// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Integration tests for the Discovery v2alpha CapabilitiesService (b.hv2).
//
// Stands up a real tonic server that mirrors the production wiring in
// `src/main.rs`:
//
//   - the LiveService is built by `live_service_server` (auth-interceptored,
//     with the uploads' end-of-stream record),
//   - the CapabilitiesService is added bare (`.add_service(...)`) — the same
//     structural auth exemption as health and reflection,
//   - the health service marks `CapabilitiesServiceServer` SERVING statically,
//   - server reflection registers BOTH file descriptor sets (building the
//     reflection service here also validates that the committed buf-generated
//     capabilities FILE_DESCRIPTOR_SET is well-formed — `build_v1` would fail
//     otherwise, exactly as `main.rs` would panic at startup).
//
// Tests then drive real gRPC clients over a plaintext channel — the same way
// the discovery operator dials the pod: no authorization metadata at all.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::oneshot;
use tonic::metadata::MetadataValue;
use tonic::transport::{Channel, Server};
use tonic::Request;
use tonic_health::pb::health_client::HealthClient;
use tonic_health::pb::HealthCheckRequest;
use tonic_reflection::server::Builder as ReflectionBuilder;

use live_edit::grpc::capabilities::CapabilitiesServiceImpl;
use live_edit::grpc::proto::live_service_client::LiveServiceClient;
use live_edit::grpc::proto::live_service_server::SERVICE_NAME as LIVE_SERVICE_NAME;
use live_edit::grpc::proto::{PingRequest, FILE_DESCRIPTOR_SET};
use live_edit::grpc::transport::live_service_server;
use live_edit::storage::test_support::TestValkey;
use live_edit_capabilities_proto::nvidia::omniverse::discovery::capabilities::v2alpha::capabilities_service_client::CapabilitiesServiceClient;
use live_edit_capabilities_proto::nvidia::omniverse::discovery::capabilities::v2alpha::capabilities_service_server::{
    CapabilitiesServiceServer, SERVICE_NAME as CAPABILITIES_SERVICE_NAME,
};
use live_edit_capabilities_proto::nvidia::omniverse::discovery::capabilities::v2alpha::{
    GrpcEndpoint, GrpcService, ListServicesRequest, ListServicesResponse, ServiceEntry,
    FILE_DESCRIPTOR_SET as CAPABILITIES_FILE_DESCRIPTOR_SET,
};

use super::helpers::{build_service, AllowAllPermissions};

// ── Server fixture ───────────────────────────────────────────────────────────

/// A running test server mirroring the production `main.rs` service wiring.
/// Holds the `TestValkey` fixture so its keyspace outlives the server and is
/// cleaned up (SCAN+UNLINK) when the fixture drops at the end of the test.
struct ProductionShapedServer {
    addr: SocketAddr,
    shutdown_tx: oneshot::Sender<()>,
    server_task: tokio::task::JoinHandle<()>,
    _tv: TestValkey,
}

impl ProductionShapedServer {
    async fn shutdown(self) {
        let _ = self.shutdown_tx.send(());
        tokio::time::timeout(Duration::from_secs(5), self.server_task)
            .await
            .expect("server did not stop within 5 seconds")
            .expect("server task panicked");
    }
}

/// Spawn a tonic server on a random port with the production service set:
/// interceptored LiveService + health + reflection + bare CapabilitiesService.
async fn spawn_production_shaped_server() -> ProductionShapedServer {
    let tv = TestValkey::setup().await;
    let service = build_service(&tv, Arc::new(AllowAllPermissions));
    let live_service = live_service_server(service);

    // Health wiring mirrors `main.rs`: the capabilities service is marked
    // SERVING statically (never flipped by the health supervisor).
    let (health_reporter, health_service) = tonic_health::server::health_reporter();
    health_reporter
        .set_serving::<CapabilitiesServiceServer<CapabilitiesServiceImpl>>()
        .await;

    // Reflection wiring mirrors `main.rs`: both descriptor sets registered.
    let reflection_service = ReflectionBuilder::configure()
        .register_encoded_file_descriptor_set(FILE_DESCRIPTOR_SET)
        .register_encoded_file_descriptor_set(CAPABILITIES_FILE_DESCRIPTOR_SET)
        .build_v1()
        .expect("reflection must build with the capabilities descriptor set registered");

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("failed to bind to random port");
    let addr = listener.local_addr().expect("no local addr");

    let (ready_tx, ready_rx) = oneshot::channel::<()>();
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();

    let server_task = tokio::spawn(async move {
        let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);
        let _ = ready_tx.send(());
        Server::builder()
            .add_service(live_service)
            .add_service(health_service)
            .add_service(reflection_service)
            // Bare add — structurally exempt from the auth interceptor,
            // exactly like `main.rs`.
            .add_service(CapabilitiesServiceServer::new(CapabilitiesServiceImpl))
            .serve_with_incoming_shutdown(incoming, async move {
                let _ = shutdown_rx.await;
            })
            .await
            .expect("test server error");
    });

    tokio::time::timeout(Duration::from_secs(5), ready_rx)
        .await
        .expect("test server spawn did not signal readiness within 5s")
        .expect("server task dropped the ready channel");

    ProductionShapedServer {
        addr,
        shutdown_tx,
        server_task,
        _tv: tv,
    }
}

/// Plain plaintext channel — carries NO authorization metadata, matching how
/// the discovery operator dials the pod.
async fn plain_channel(addr: SocketAddr) -> Channel {
    Channel::from_shared(format!("http://{addr}"))
        .expect("invalid endpoint URI")
        .connect()
        .await
        .expect("failed to connect to test server")
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn capabilities_list_services_without_credentials_returns_exact_static_payload() {
    let server = spawn_production_shaped_server().await;
    let mut client = CapabilitiesServiceClient::new(plain_channel(server.addr).await);

    // No authorization metadata attached anywhere on this request.
    let response = client
        .list_services(ListServicesRequest::default())
        .await
        .expect("ListServices without credentials must succeed")
        .into_inner();

    // Targeted assertions first for readable failures, exact-payload pin last.
    let entry = response
        .services
        .first()
        .expect("response must carry exactly one ServiceEntry");
    assert_eq!(
        entry.service_name, "live-service",
        "service_name must be the discovery \"type\" key, not the deployment id"
    );
    assert!(
        entry.rest_services.is_empty(),
        "rest_services must stay empty — live-edit is gRPC-only"
    );
    assert!(
        response.next_page_token.is_empty(),
        "next_page_token must be empty — the response is a single static page"
    );

    let expected = ListServicesResponse {
        services: vec![ServiceEntry {
            service_name: "live-service".to_owned(),
            grpc_services: vec![GrpcService {
                endpoints: vec![GrpcEndpoint {
                    service_name: LIVE_SERVICE_NAME.to_owned(),
                }],
            }],
            rest_services: vec![],
        }],
        next_page_token: String::new(),
    };
    assert_eq!(
        response, expected,
        "ListServices must return the exact static payload: one ServiceEntry \
         advertising only the fully-qualified LiveService gRPC endpoint"
    );

    server.shutdown().await;
}

/// The auth exemption must be scoped to the CapabilitiesService, not a hole
/// in the server: on the SAME server, a LiveService RPC without credentials is
/// rejected UNAUTHENTICATED by the interceptor, the same RPC with a bearer
/// token succeeds, and the credential-less CapabilitiesService call succeeds.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn auth_exemption_is_scoped_live_service_without_credentials_returns_unauthenticated() {
    let server = spawn_production_shaped_server().await;
    let channel = plain_channel(server.addr).await;

    // 1. LiveService without credentials → UNAUTHENTICATED (interceptor hit).
    let mut live_client = LiveServiceClient::new(channel.clone());
    let status = live_client
        .ping(PingRequest {
            client_timestamp_microseconds_since_epoch: 1,
        })
        .await
        .expect_err("LiveService RPC without credentials must be rejected");
    assert_eq!(
        status.code(),
        tonic::Code::Unauthenticated,
        "the auth interceptor must reject a credential-less LiveService RPC \
         with UNAUTHENTICATED, got {:?}",
        status.code()
    );

    // 2. Same RPC with a bearer token succeeds — proving the rejection above
    //    is credential absence, not a broken server.
    let mut request = Request::new(PingRequest {
        client_timestamp_microseconds_since_epoch: 42,
    });
    request.metadata_mut().insert(
        "authorization",
        MetadataValue::try_from("Bearer test-token").expect("valid metadata"),
    );
    let ping_response = live_client
        .ping(request)
        .await
        .expect("LiveService Ping with a bearer token must succeed")
        .into_inner();
    assert_eq!(
        ping_response.client_timestamp_microseconds_since_epoch, 42,
        "authenticated Ping must echo the client timestamp"
    );

    // 3. CapabilitiesService without credentials succeeds on the same server.
    let mut cap_client = CapabilitiesServiceClient::new(channel);
    let response = cap_client
        .list_services(ListServicesRequest::default())
        .await
        .expect("ListServices without credentials must succeed on the same server")
        .into_inner();
    assert_eq!(
        response.services.len(),
        1,
        "credential-less ListServices must return the single static entry"
    );

    server.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn capabilities_service_health_check_reports_serving() {
    let server = spawn_production_shaped_server().await;
    let mut client = HealthClient::new(plain_channel(server.addr).await);

    let response = client
        .check(HealthCheckRequest {
            // The generated NamedService name — the exact string a discovery
            // health probe would query.
            service: CAPABILITIES_SERVICE_NAME.to_owned(),
        })
        .await
        .expect("health check for the CapabilitiesService must succeed")
        .into_inner();

    assert_eq!(
        response.status,
        tonic_health::pb::health_check_response::ServingStatus::Serving as i32,
        "CapabilitiesService must be registered SERVING in the health service"
    );

    server.shutdown().await;
}
