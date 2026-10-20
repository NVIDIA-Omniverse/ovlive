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

//! Completion logging behind a real tonic server (SDD §17.2.1).
//!
//! The unit tests in `src/grpc/{auth_interceptor,metrics_layer,rpc_log}.rs`
//! pin each completion point in isolation. What they cannot see are the
//! premises that depend on tonic's own plumbing, and those are what this file
//! checks, on a server wired the way `main.rs` wires it — `MetricsLayer` as a
//! server layer, the `LiveService` built by `live_service_server` (behind
//! `with_interceptor(auth_interceptor)`),
//! and health, reflection and the Discovery `CapabilitiesService` added bare:
//!
//! * the completion ledger the metrics layer inserts reaches the interceptor
//!   and the handlers through tonic's `InterceptedService` (if it did not, a
//!   failure would log twice: once where it happened, once at the backstop);
//! * tonic's own rejections — the 4 MB decode limit, an undecodable body —
//!   come back through the metrics layer as trailers-only statuses the
//!   backstop can read;
//! * a client `grpc-timeout` drops the metrics layer's future (tonic's
//!   `GrpcTimeout` sits outside it), which the drop guard counts and logs;
//! * a `CreateLayer` / `UpdateLayer` upload over the size cap reaches the
//!   client as `INVALID_ARGUMENT`, a code clients do not retry, with one INFO
//!   line (b.75i);
//! * the probes Kubernetes and discovery hit continuously — health `Check`, a
//!   health `Watch` the client closes, reflection, `ListServices` — produce no
//!   line at all and exactly the one counter tick they always did. A stray
//!   line there would be one per probe, forever.
//!
//! # Capture
//!
//! Every line here is emitted from a task tonic spawned per connection, so the
//! tests run on a **current-thread** runtime under a thread-local
//! [`ThreadCapture`] (see `log_capture`). Each waits for the server-side
//! counter tick of the call it made before reading the capture: every
//! completion line is emitted before the metrics layer records the call, so
//! once the tick is visible the line is too. `TestValkey` cannot clean up on a
//! current-thread runtime, so every test ends with [`purge_keys`].

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use tokio::sync::oneshot;
use tonic::metadata::MetadataValue;
use tonic::transport::{Channel, Server};
use tonic::{Code, Request};
use tonic_reflection::server::Builder as ReflectionBuilder;

use live_edit::auth::permissions::PermissionsClient;
use live_edit::auth::{AuthError, AuthResult, Permission};
use live_edit::grpc::capabilities::CapabilitiesServiceImpl;
use live_edit::grpc::metrics_layer::MetricsLayer;
use live_edit::grpc::proto::live_service_client::LiveServiceClient;
use live_edit::grpc::proto::{
    StatLayerRequest, StatLayerResponse, SubscribeReadLayerRequest, FILE_DESCRIPTOR_SET,
};
use live_edit::grpc::service::LiveEditService;
use live_edit::grpc::transport::live_service_server;
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::Metrics;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;
use live_edit_capabilities_proto::nvidia::omniverse::discovery::capabilities::v2alpha::capabilities_service_client::CapabilitiesServiceClient;
use live_edit_capabilities_proto::nvidia::omniverse::discovery::capabilities::v2alpha::capabilities_service_server::CapabilitiesServiceServer;
use live_edit_capabilities_proto::nvidia::omniverse::discovery::capabilities::v2alpha::ListServicesRequest;

use super::helpers::{
    build_service, build_service_with_size_cap, create_session_and_layer, size_cap_message,
    AllowAllPermissions, SMALL_SIZE_CAP,
};
use super::log_capture::{
    assert_common_fields, assert_field, purge_keys, Captured, ThreadCapture, GRPC_CODE_SLUGS,
    INTERNAL_ERROR_MSG, RPC_ABANDONED_MSG, RPC_FAILURE_MSG,
};

/// Upper bound on any single wait. Unreachable on the healthy path.
const SETTLE: Duration = Duration::from_secs(10);

/// How long a probe test keeps watching for a late, stray tick or line after
/// the expected one arrived. An absence window, not a synchronisation sleep:
/// the positive half (the expected tick) has already been observed.
const ABSENCE_WINDOW: Duration = Duration::from_millis(300);

/// The metrics layer's service label, and the `service` label every counter
/// read here pins.
const SERVICE_LABEL: &str = "test";

/// Bearer for requests that must pass the interceptor. The mock permission
/// clients ignore it; it must still never be logged.
const BEARER: &str = "server-log-bearer-must-never-be-logged-a61f";

// =============================================================================
// Server fixture
// =============================================================================

/// A running server with the production service set and layer stack.
struct FullStackServer {
    addr: SocketAddr,
    /// The metrics layer's registry — the one `live_endpoint_calls_total`
    /// ticks in. Separate from the service's own, as in `main.rs`.
    layer_metrics: Arc<Metrics>,
    shutdown_tx: oneshot::Sender<()>,
    server_task: tokio::task::JoinHandle<()>,
}

impl FullStackServer {
    async fn start(svc: LiveEditService) -> Self {
        let layer_metrics = Arc::new(crate::common::metrics_scrape_only(SERVICE_LABEL));
        // 30 s / 2 s: the production defaults, so no test here trips the
        // server-side deadline by accident.
        let metrics_layer =
            MetricsLayer::new(Arc::clone(&layer_metrics), SERVICE_LABEL, 30_000, 2_000);

        let (health_reporter, health_service) = tonic_health::server::health_reporter();
        health_reporter
            .set_service_status("", tonic_health::ServingStatus::Serving)
            .await;
        let reflection_service = ReflectionBuilder::configure()
            .register_encoded_file_descriptor_set(FILE_DESCRIPTOR_SET)
            .build_v1()
            .expect("reflection must build");

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("failed to bind to a random port");
        let addr = listener.local_addr().expect("no local addr");
        let (ready_tx, ready_rx) = oneshot::channel::<()>();
        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();

        let server_task = tokio::spawn(async move {
            let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);
            let _ = ready_tx.send(());
            Server::builder()
                .layer(metrics_layer)
                .add_service(live_service_server(svc))
                .add_service(health_service)
                .add_service(reflection_service)
                .add_service(CapabilitiesServiceServer::new(CapabilitiesServiceImpl))
                .serve_with_incoming_shutdown(incoming, async move {
                    let _ = shutdown_rx.await;
                })
                .await
                .expect("test server error");
        });
        tokio::time::timeout(SETTLE, ready_rx)
            .await
            .expect("the server did not signal readiness")
            .expect("the server task dropped its ready channel");

        Self {
            addr,
            layer_metrics,
            shutdown_tx,
            server_task,
        }
    }

    async fn channel(&self) -> Channel {
        Channel::from_shared(format!("http://{}", self.addr))
            .expect("valid endpoint URI")
            .connect()
            .await
            .expect("failed to connect to the test server")
    }

    /// `live_endpoint_calls_total{endpoint, grpc_code}`.
    fn calls(&self, endpoint: &str, code: &str) -> u64 {
        sample_value(
            &render(&self.layer_metrics),
            "live_endpoint_calls_total",
            &[
                ("endpoint", endpoint),
                ("service", SERVICE_LABEL),
                ("grpc_code", code),
            ],
        )
        .unwrap_or(0.0) as u64
    }

    /// Every `live_endpoint_calls_total` tick for `endpoint`, whatever code.
    fn total_calls(&self, endpoint: &str) -> u64 {
        GRPC_CODE_SLUGS
            .iter()
            .map(|code| self.calls(endpoint, code))
            .sum()
    }

    /// Wait until `endpoint` has recorded a `code` tick. Every completion line
    /// is emitted before the call is recorded, so from here the line is in the
    /// capture.
    async fn await_call(&self, endpoint: &str, code: &str) {
        let deadline = tokio::time::Instant::now() + SETTLE;
        while self.calls(endpoint, code) == 0 {
            assert!(
                tokio::time::Instant::now() < deadline,
                "no {endpoint}/{code} tick within {SETTLE:?}"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    async fn shutdown(self) {
        let _ = self.shutdown_tx.send(());
        tokio::time::timeout(SETTLE, self.server_task)
            .await
            .expect("the server did not stop in time")
            .expect("the server task panicked");
    }
}

fn with_bearer<T>(inner: T) -> Request<T> {
    let mut request = Request::new(inner);
    request.metadata_mut().insert(
        "authorization",
        MetadataValue::try_from(format!("Bearer {BEARER}")).expect("valid metadata"),
    );
    request
}

fn stat_request(session_id: &str, layer_name: &str) -> StatLayerRequest {
    StatLayerRequest {
        session_id: session_id.to_owned(),
        layer_name: layer_name.to_owned(),
    }
}

/// Assert the capture holds exactly one completion line, and return it.
fn only_completion_line(captured: &Captured) -> &serde_json::Value {
    let lines = captured.completion_lines();
    assert_eq!(
        lines.len(),
        1,
        "exactly one completion line per failure, whichever completion point emits it; full \
         capture:\n{}",
        captured.text
    );
    lines[0]
}

// =============================================================================
// (a) The auth interceptor
// =============================================================================

/// A request with no `authorization` header logs one WARN naming the endpoint
/// the metrics layer bucketed it under, and nothing else — no second line from
/// the backstop, which would fire if the ledger never reached the interceptor.
#[tokio::test]
async fn missing_authorization_logs_one_interceptor_line_naming_the_endpoint() {
    let tv = TestValkey::setup().await;
    let server = FullStackServer::start(build_service(&tv, Arc::new(AllowAllPermissions))).await;
    let capture = ThreadCapture::start();

    let status = LiveServiceClient::new(server.channel().await)
        .stat_layer(stat_request("some-session", "some-layer"))
        .await
        .expect_err("a request without credentials must be refused");
    assert_eq!(status.code(), Code::Unauthenticated);
    server.await_call("StatLayer", "UNAUTHENTICATED").await;

    let captured = capture.snapshot();
    let event = only_completion_line(&captured);
    assert_eq!(event["fields"]["message"], RPC_FAILURE_MSG);
    assert_common_fields(event, "WARN", "StatLayer", "header", "UNAUTHENTICATED");
    assert_field(event, "rejection", "missing_header".into());
    assert_field(event, "status_message", "".into());
    assert_eq!(server.total_calls("StatLayer"), 1, "one call, one tick");

    drop(capture);
    server.shutdown().await;
    purge_keys(&tv).await;
}

// =============================================================================
// (b)–(c) The transport backstop
// =============================================================================

/// A unary request over tonic's 4 MB decode limit is refused `OUT_OF_RANGE`
/// before any handler runs, and the backstop logs it once at INFO,
/// `stage=transport`.
#[tokio::test]
async fn oversized_request_logs_one_info_out_of_range_at_the_transport_stage() {
    /// One byte past tonic's unconfigured 4 MiB default decode limit.
    const OVERSIZED_LAYER_NAME_BYTES: usize = 4 * 1024 * 1024 + 1;

    let tv = TestValkey::setup().await;
    let server = FullStackServer::start(build_service(&tv, Arc::new(AllowAllPermissions))).await;
    let capture = ThreadCapture::start();

    let status = LiveServiceClient::new(server.channel().await)
        .stat_layer(with_bearer(stat_request(
            "some-session",
            &"x".repeat(OVERSIZED_LAYER_NAME_BYTES),
        )))
        .await
        .expect_err("an oversized request must be refused");
    assert_eq!(
        status.code(),
        Code::OutOfRange,
        "precondition: tonic refuses an oversized message OUT_OF_RANGE; got {status:?}"
    );
    server.await_call("StatLayer", "OUT_OF_RANGE").await;

    let captured = capture.snapshot();
    let event = only_completion_line(&captured);
    assert_eq!(event["fields"]["message"], RPC_FAILURE_MSG);
    assert_common_fields(event, "INFO", "StatLayer", "transport", "OUT_OF_RANGE");
    assert_field(event, "status_message", status.message().into());
    captured.assert_absent_everywhere(BEARER);

    drop(capture);
    server.shutdown().await;
    purge_keys(&tv).await;
}

/// A message that fails the service's `StatLayerRequest` decode: field 1 is a
/// varint here, where `StatLayerRequest.session_id` is a string.
#[derive(Clone, PartialEq, prost::Message)]
struct WrongWireTypeRequest {
    #[prost(uint64, tag = "1")]
    session_id: u64,
}

/// A body that is not a valid `StatLayerRequest` is refused `INTERNAL` by
/// tonic, and the backstop logs it once at WARN — never ERROR, since a client
/// can send it at will and no server fault occurred.
#[tokio::test]
async fn undecodable_request_logs_one_warn_internal_at_the_transport_stage_and_no_error() {
    let tv = TestValkey::setup().await;
    let server = FullStackServer::start(build_service(&tv, Arc::new(AllowAllPermissions))).await;
    let capture = ThreadCapture::start();

    let mut grpc = tonic::client::Grpc::new(server.channel().await);
    grpc.ready().await.expect("the channel must become ready");
    let status = grpc
        .unary::<_, StatLayerResponse, _>(
            with_bearer(WrongWireTypeRequest { session_id: 42 }),
            tonic::codegen::http::uri::PathAndQuery::from_static(
                "/nvidia.omniverse.live.v1alpha.LiveService/StatLayer",
            ),
            tonic_prost::ProstCodec::default(),
        )
        .await
        .expect_err("an undecodable request must be refused");
    assert_eq!(
        status.code(),
        Code::Internal,
        "precondition: tonic refuses an undecodable body INTERNAL; got {status:?}"
    );
    server.await_call("StatLayer", "INTERNAL").await;

    let captured = capture.snapshot();
    let event = only_completion_line(&captured);
    assert_eq!(event["fields"]["message"], RPC_FAILURE_MSG);
    assert_common_fields(event, "WARN", "StatLayer", "transport", "INTERNAL");
    assert!(
        captured.with_message(INTERNAL_ERROR_MSG).is_empty(),
        "a client-caused decode failure must not produce the server-fault ERROR"
    );

    drop(capture);
    server.shutdown().await;
    purge_keys(&tv).await;
}

// =============================================================================
// (d) The drop guard, on a client `grpc-timeout`
// =============================================================================

/// A `PermissionsClient` whose check never returns in the test's lifetime —
/// the slow handler a client deadline cuts off.
struct StalledPermissions;

#[async_trait::async_trait]
impl PermissionsClient for StalledPermissions {
    async fn check_permission(
        &self,
        _token: &str,
        _path: &str,
        _permission: Permission,
    ) -> Result<AuthResult, AuthError> {
        tokio::time::sleep(Duration::from_secs(3600)).await;
        Ok(AuthResult::Allowed)
    }
}

/// A client `grpc-timeout` shorter than a slow handler is abandoned: tonic's
/// `GrpcTimeout`, outside the metrics layer, drops its future. The drop guard
/// records exactly one `CANCELLED` tick and logs one INFO `RPC abandoned before
/// completion` carrying the client's bound — where before, the call left no
/// line and no tick at all.
#[tokio::test]
async fn client_timeout_on_a_slow_handler_logs_one_abandoned_line_and_one_cancelled_tick() {
    /// The client's deadline; the handler would take an hour.
    const CLIENT_TIMEOUT: Duration = Duration::from_millis(200);

    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(StalledPermissions));
    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create("server-log-slow", "/stage", "test-user")
        .await
        .expect("session creation must succeed");
    let server = FullStackServer::start(svc).await;
    let capture = ThreadCapture::start();

    let mut request = with_bearer(stat_request(&session_id, "some-layer"));
    request.set_timeout(CLIENT_TIMEOUT);
    let status = LiveServiceClient::new(server.channel().await)
        .stat_layer(request)
        .await
        .expect_err("the call must not complete inside the client's deadline");
    assert!(
        matches!(status.code(), Code::Cancelled | Code::DeadlineExceeded),
        "precondition: the client's deadline ended the call; got {status:?}"
    );
    server.await_call("StatLayer", "CANCELLED").await;

    let captured = capture.snapshot();
    let event = only_completion_line(&captured);
    assert_eq!(event["level"], "INFO");
    assert_field(event, "message", RPC_ABANDONED_MSG.into());
    assert_field(event, "endpoint", "StatLayer".into());
    assert_field(event, "grpc_code", "CANCELLED".into());
    assert_field(event, "client_timeout_ms", 200.into());
    assert!(event["fields"]["elapsed_ms"].is_u64(), "got {event}");
    assert_eq!(
        server.total_calls("StatLayer"),
        1,
        "exactly one tick for the abandoned call — CANCELLED, and nothing else"
    );

    drop(capture);
    server.shutdown().await;
    purge_keys(&tv).await;
}

// =============================================================================
// (e)–(f) One line in total through the full stack
// =============================================================================

/// A handler failure behind the full stack produces exactly one line in total:
/// the handler boundary's. The backstop stays silent because the ledger it
/// reads is the one the boundary marked — proof it crossed tonic's interceptor.
#[tokio::test]
async fn handler_failure_behind_the_full_stack_logs_exactly_one_line() {
    let tv = TestValkey::setup().await;
    let server = FullStackServer::start(build_service(&tv, Arc::new(AllowAllPermissions))).await;
    let capture = ThreadCapture::start();

    let status = LiveServiceClient::new(server.channel().await)
        .stat_layer(with_bearer(stat_request("server-log-no-such-session", "l")))
        .await
        .expect_err("an unknown session must fail");
    assert_eq!(status.code(), Code::NotFound);
    server.await_call("StatLayer", "NOT_FOUND").await;

    let captured = capture.snapshot();
    let event = only_completion_line(&captured);
    assert_eq!(event["fields"]["message"], RPC_FAILURE_MSG);
    assert_common_fields(event, "INFO", "StatLayer", "auth", "NOT_FOUND");
    assert_field(event, "session_id", "server-log-no-such-session".into());
    captured.assert_absent_everywhere(BEARER);

    drop(capture);
    server.shutdown().await;
    purge_keys(&tv).await;
}

/// A `SubscribeReadLayer` that fails after its response stream opened — an
/// unknown layer, discovered by the subscription task — produces exactly one
/// line in total: the task's own teardown line at `stage=stream`.
///
/// The one completion point whose ledger mark crosses a task boundary: the
/// tail logs from a clone of the setup context on the spawned task, and the
/// metrics layer reads the mark when the trailer streams past on the
/// connection's task. Were the mark lost on the way, the backstop would log the
/// same `NOT_FOUND` again at `stage=transport` and this test would see two
/// lines.
#[tokio::test]
async fn streaming_teardown_behind_the_full_stack_logs_exactly_one_line_at_the_stream_stage() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create("server-log-stream", "/stage", "test-user")
        .await
        .expect("session creation must succeed");
    let server = FullStackServer::start(svc).await;
    let capture = ThreadCapture::start();

    let mut stream = LiveServiceClient::new(server.channel().await)
        .subscribe_read_layer(with_bearer(SubscribeReadLayerRequest {
            session_id: session_id.clone(),
            layer_name: "server-log-no-such-layer".to_owned(),
            object_id: 0,
            sequence_num: 0,
            client_id: "server-log-subscriber".to_owned(),
        }))
        .await
        .expect("setup must succeed; the layer lookup runs in the subscription task")
        .into_inner();
    let status = tokio::time::timeout(SETTLE, async {
        while let Some(item) = stream.next().await {
            if let Err(status) = item {
                return status;
            }
        }
        panic!("the subscription ended without a status");
    })
    .await
    .expect("the subscription did not end within the deadline");
    assert_eq!(status.code(), Code::NotFound);
    server.await_call("SubscribeReadLayer", "NOT_FOUND").await;

    let captured = capture.snapshot();
    let event = only_completion_line(&captured);
    assert_eq!(event["fields"]["message"], RPC_FAILURE_MSG);
    assert_common_fields(event, "INFO", "SubscribeReadLayer", "stream", "NOT_FOUND");
    assert_field(event, "session_id", session_id.as_str().into());
    assert_field(event, "layer_name", "server-log-no-such-layer".into());
    captured.assert_absent_everywhere(BEARER);

    drop(capture);
    server.shutdown().await;
    purge_keys(&tv).await;
}

/// A server fault behind the full stack produces exactly one ERROR — with the
/// request context — and no `RPC failed` from the backstop, and no
/// context-free fallback when tonic drops the status after writing it.
#[tokio::test]
async fn server_fault_behind_the_full_stack_logs_exactly_one_error() {
    use fred::interfaces::KeysInterface;
    const LAYER: &str = "server-log-fault-layer";

    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, "server-log-fault", LAYER).await;
    let _: () = tv
        .client()
        .pool
        .set(
            tv.key_schema().obj_header(&object_id.to_string()),
            "not-a-hash",
            None,
            None,
            false,
        )
        .await
        .expect("retyping obj:hdr must succeed");
    let server = FullStackServer::start(svc).await;
    let capture = ThreadCapture::start();

    let status = LiveServiceClient::new(server.channel().await)
        .stat_layer(with_bearer(stat_request(&session_id, LAYER)))
        .await
        .expect_err("a wrong-typed header must fail");
    assert_eq!(status.code(), Code::Internal);
    assert_eq!(
        status.message(),
        INTERNAL_ERROR_MSG,
        "the wire stays redacted"
    );
    // The server's status is converted to the trailers-only response — and so
    // dropped — before the metrics layer records the call, so from this tick
    // a missing mark would already have produced the fallback line.
    server.await_call("StatLayer", "INTERNAL").await;

    let captured = capture.snapshot();
    let event = only_completion_line(&captured);
    assert_eq!(event["fields"]["message"], INTERNAL_ERROR_MSG);
    assert_common_fields(event, "ERROR", "StatLayer", "storage", "INTERNAL");
    assert_field(event, "session_id", session_id.as_str().into());
    assert_field(event, "layer_name", LAYER.into());
    let detail = event["fields"]["error"].as_str().unwrap_or_default();
    assert!(
        detail.contains("WRONGTYPE"),
        "the ERROR carries the un-redacted detail the wire lost; got {event}"
    );

    drop(capture);
    server.shutdown().await;
    purge_keys(&tv).await;
}

// =============================================================================
// (f2) An upload over the size cap — b.75i
// =============================================================================
//
// The size cap is the one refusal a client must never retry: the upload can
// never fit, and every retry re-sends and re-buffers up to the cap before it
// is refused again. It answers `INVALID_ARGUMENT` (not retried by Kit's client
// library), where it used to answer the retried `RESOURCE_EXHAUSTED`. Driven
// through a real server so the code the client sees is the one asserted — the
// handler's `Status`, after tonic and the metrics layer — and the one
// completion line is pinned at INFO, the level a client-caused refusal gets.

// The service is built with `helpers::SMALL_SIZE_CAP` so the over-cap body stays
// far below tonic's 4 MiB decode limit; through a real server that limit would
// otherwise refuse the frame first (`OUT_OF_RANGE`, at the transport stage) and
// the cap would never run.

// The upload names a real session (and, for `UpdateLayer`, its real layer and
// object): the call is authorized and its target resolved from the header
// before any content is read (b.715), so an upload to a session that does not
// exist is refused `NOT_FOUND` from its header and never reaches the size cap.
// The cap runs while the admitted upload reads its content, so the line is
// logged at stage `body`.

/// Assert an over-cap upload was refused `INVALID_ARGUMENT` with the size-cap
/// message, ticked that code and no `RESOURCE_EXHAUSTED`, and logged exactly
/// one INFO `RPC failed` at the `body` stage.
async fn assert_size_cap_refusal(
    server: &FullStackServer,
    capture: &ThreadCapture,
    endpoint: &str,
    status: &tonic::Status,
) {
    assert_eq!(
        status.code(),
        Code::InvalidArgument,
        "an upload over the size cap can never succeed, so it must answer a code clients do \
         not retry; RESOURCE_EXHAUSTED is retried and re-sends the whole upload each time. \
         Got {status:?}"
    );
    assert_eq!(status.message(), size_cap_message(SMALL_SIZE_CAP));
    server.await_call(endpoint, "INVALID_ARGUMENT").await;
    assert_eq!(
        server.calls(endpoint, "RESOURCE_EXHAUSTED"),
        0,
        "no RESOURCE_EXHAUSTED tick"
    );

    let captured = capture.snapshot();
    let event = only_completion_line(&captured);
    assert_eq!(event["fields"]["message"], RPC_FAILURE_MSG);
    assert_common_fields(event, "INFO", endpoint, "body", "INVALID_ARGUMENT");
    assert_field(
        event,
        "status_message",
        size_cap_message(SMALL_SIZE_CAP).into(),
    );
    captured.assert_absent_everywhere(BEARER);
}

/// A `CreateLayer` upload one byte over the cap answers `INVALID_ARGUMENT`.
#[tokio::test]
async fn create_layer_over_the_size_cap_answers_invalid_argument_and_logs_one_info_line() {
    use super::helpers::{make_create_data, make_create_header};

    let tv = TestValkey::setup().await;
    let svc = build_service_with_size_cap(&tv, Arc::new(AllowAllPermissions), SMALL_SIZE_CAP);
    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create("size-cap-session", "/stage", "test-user")
        .await
        .expect("session creation must succeed");
    let server = FullStackServer::start(svc).await;
    let capture = ThreadCapture::start();

    let upload = futures::stream::iter(vec![
        make_create_header(&session_id, "size-cap-layer"),
        make_create_data(vec![0xAB; SMALL_SIZE_CAP + 1]),
    ]);
    let status = LiveServiceClient::new(server.channel().await)
        .create_layer(with_bearer(upload))
        .await
        .expect_err("an upload over the size cap must be refused");
    assert_size_cap_refusal(&server, &capture, "CreateLayer", &status).await;

    drop(capture);
    server.shutdown().await;
    purge_keys(&tv).await;
}

/// An `UpdateLayer` upload one byte over the cap answers `INVALID_ARGUMENT`.
#[tokio::test]
async fn update_layer_over_the_size_cap_answers_invalid_argument_and_logs_one_info_line() {
    use super::helpers::{make_update_data, make_update_header};

    let tv = TestValkey::setup().await;
    let svc = build_service_with_size_cap(&tv, Arc::new(AllowAllPermissions), SMALL_SIZE_CAP);
    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, "size-cap-session", "size-cap-layer").await;
    let server = FullStackServer::start(svc).await;
    let capture = ThreadCapture::start();

    let upload = futures::stream::iter(vec![
        make_update_header(&session_id, "size-cap-layer", object_id),
        make_update_data(vec![0xAB; SMALL_SIZE_CAP + 1]),
    ]);
    let status = LiveServiceClient::new(server.channel().await)
        .update_layer(with_bearer(upload))
        .await
        .expect_err("an upload over the size cap must be refused");
    assert_size_cap_refusal(&server, &capture, "UpdateLayer", &status).await;

    drop(capture);
    server.shutdown().await;
    purge_keys(&tv).await;
}

// =============================================================================
// (g) Probes stay silent
// =============================================================================
//
// Health, reflection and `ListServices` skip the interceptor but sit behind
// the same metrics layer that now carries the ledger, the backstop and the
// drop guard. Each is bucketed under the `other` endpoint. Each must add no
// line of any kind and exactly the one tick it always did.

/// The endpoint bucket every non-`LiveService` method is counted under.
const OTHER: &str = "other";

/// Assert the probe window produced no log line of any kind, then hold an
/// absence window and assert no late line or second tick arrived.
async fn assert_probe_was_silent(
    server: &FullStackServer,
    capture: &ThreadCapture,
    events_before: usize,
    what: &str,
) {
    tokio::time::sleep(ABSENCE_WINDOW).await;
    let captured = capture.snapshot();
    assert_eq!(
        captured.events.len(),
        events_before,
        "{what} must log nothing; new lines:\n{}",
        captured.events[events_before..]
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join("\n")
    );
    assert_eq!(
        server.total_calls(OTHER),
        1,
        "{what} must record exactly the one tick it always did"
    );
}

/// A normal health `Check` logs nothing and ticks `other`/`OK` once.
#[tokio::test]
async fn health_check_logs_nothing_and_ticks_once() {
    use tonic_health::pb::health_client::HealthClient;
    use tonic_health::pb::HealthCheckRequest;

    let tv = TestValkey::setup().await;
    let server = FullStackServer::start(build_service(&tv, Arc::new(AllowAllPermissions))).await;
    let mut client = HealthClient::new(server.channel().await);
    let capture = ThreadCapture::start();
    let before = capture.snapshot().events.len();

    client
        .check(HealthCheckRequest {
            service: String::new(),
        })
        .await
        .expect("the health check must succeed");
    server.await_call(OTHER, "OK").await;
    assert_probe_was_silent(&server, &capture, before, "a health Check").await;

    drop(capture);
    server.shutdown().await;
    purge_keys(&tv).await;
}

/// A health `Watch` the client closes after its first update logs nothing.
///
/// This is the shape a closed `Watch` stream has in the metrics layer: the
/// response existed and its body is dropped — `CANCELLED`, silently. It must
/// never reach the drop guard's abandoned line, which is for calls torn down
/// before any response existed.
#[tokio::test]
async fn closed_health_watch_logs_nothing_and_ticks_once() {
    use tonic_health::pb::health_client::HealthClient;
    use tonic_health::pb::HealthCheckRequest;

    let tv = TestValkey::setup().await;
    let server = FullStackServer::start(build_service(&tv, Arc::new(AllowAllPermissions))).await;
    let mut client = HealthClient::new(server.channel().await);
    let capture = ThreadCapture::start();
    let before = capture.snapshot().events.len();

    let mut updates = client
        .watch(HealthCheckRequest {
            service: String::new(),
        })
        .await
        .expect("the watch must open")
        .into_inner();
    let first = tokio::time::timeout(SETTLE, updates.next())
        .await
        .expect("the first watch update must arrive")
        .expect("the watch stream must not be empty")
        .expect("the first watch update must not be an error");
    assert_eq!(
        first.status,
        tonic_health::pb::health_check_response::ServingStatus::Serving as i32
    );
    drop(updates);
    server.await_call(OTHER, "CANCELLED").await;
    assert_probe_was_silent(&server, &capture, before, "a closed health Watch").await;

    drop(capture);
    server.shutdown().await;
    purge_keys(&tv).await;
}

/// A reflection `ListServices` exchange logs nothing and ticks once.
#[tokio::test]
async fn reflection_request_logs_nothing_and_ticks_once() {
    use tonic_reflection::pb::v1::server_reflection_client::ServerReflectionClient;
    use tonic_reflection::pb::v1::server_reflection_request::MessageRequest;
    use tonic_reflection::pb::v1::ServerReflectionRequest;

    let tv = TestValkey::setup().await;
    let server = FullStackServer::start(build_service(&tv, Arc::new(AllowAllPermissions))).await;
    let mut client = ServerReflectionClient::new(server.channel().await);
    let capture = ThreadCapture::start();
    let before = capture.snapshot().events.len();

    // One request, then the request stream ends, so the server ends the
    // exchange with an OK trailer.
    let request = ServerReflectionRequest {
        host: String::new(),
        message_request: Some(MessageRequest::ListServices(String::new())),
    };
    let mut responses = client
        .server_reflection_info(futures::stream::iter(vec![request]))
        .await
        .expect("the reflection exchange must open")
        .into_inner();
    let mut answered = 0;
    while let Some(response) = tokio::time::timeout(SETTLE, responses.next())
        .await
        .expect("the reflection exchange must finish")
    {
        response.expect("a reflection response must not be an error");
        answered += 1;
    }
    assert_eq!(answered, 1, "precondition: one request, one answer");
    server.await_call(OTHER, "OK").await;
    assert_probe_was_silent(&server, &capture, before, "a reflection request").await;

    drop(capture);
    server.shutdown().await;
    purge_keys(&tv).await;
}

/// A Discovery `ListServices` call logs nothing and ticks once.
#[tokio::test]
async fn capabilities_list_services_logs_nothing_and_ticks_once() {
    let tv = TestValkey::setup().await;
    let server = FullStackServer::start(build_service(&tv, Arc::new(AllowAllPermissions))).await;
    let mut client = CapabilitiesServiceClient::new(server.channel().await);
    let capture = ThreadCapture::start();
    let before = capture.snapshot().events.len();

    client
        .list_services(ListServicesRequest::default())
        .await
        .expect("ListServices must succeed without credentials");
    server.await_call(OTHER, "OK").await;
    assert_probe_was_silent(&server, &capture, before, "a ListServices call").await;

    drop(capture);
    server.shutdown().await;
    purge_keys(&tv).await;
}
