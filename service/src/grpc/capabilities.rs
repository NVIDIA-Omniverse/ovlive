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

//! Discovery Service v2alpha `CapabilitiesService` implementation (b.hv2).
//!
//! Implements `nvidia.omniverse.discovery.capabilities.v2alpha.CapabilitiesService`
//! so the v2 Discovery Kubernetes operator can enumerate this service. The
//! operator dials the pod's `grpc` port with a bare plaintext tonic channel
//! (no auth metadata), calls `ListServices`, and serves the resulting entries
//! over its HTTP endpoint.
//!
//! Design notes (matches the fleet-wide capabilities reference pattern):
//!
//! - **Static response** — one [`ServiceEntry`]; `page_size` / `page_token`
//!   are ignored and `next_page_token` is always empty.
//! - **`service_name` is the discovery "type" key** (`"live-service"`, the v1
//!   static-registry `type`), not the deployment id. In v2 output the
//!   discovery identity is the k8s Service object name plus the
//!   fully-qualified gRPC service name; `ServiceEntry.service_name` is a
//!   REST-name fallback the operator never surfaces for gRPC-only services.
//! - **Advertised set = the domain service only.** Health, reflection, and
//!   the CapabilitiesService itself are deliberately not advertised, matching
//!   the fleet-wide pattern. The FQ name comes from the generated
//!   `live_service_server::SERVICE_NAME` constant, never a hardcoded string.
//! - **`rest_services` stays empty** — live-edit is gRPC-only. The axum
//!   `/metrics` port is Prometheus scrape plumbing, not a service API.
//! - **Unauthenticated by design** — registered in `main.rs` via a bare
//!   `.add_service(...)`, so the per-service auth interceptor attached to
//!   `LiveServiceServer` never sees it (same structural exemption as health
//!   and reflection); discovery scans carry no credentials.

use live_edit_capabilities_proto::nvidia::omniverse::discovery::capabilities::v2alpha::capabilities_service_server::CapabilitiesService;
use live_edit_capabilities_proto::nvidia::omniverse::discovery::capabilities::v2alpha::{
    GrpcEndpoint, GrpcService, ListServicesRequest, ListServicesResponse, ServiceEntry,
};
use tonic::{Request, Response, Status};

use crate::grpc::proto::live_service_server::SERVICE_NAME as LIVE_GRPC_SERVICE_NAME;

/// The discovery "type" key for this service — the `type` field of the v1
/// static discovery registry entry (the discovery service's per-environment
/// values), kept for fleet-wide
/// consistency. NOT the deployment `id` (`"live-edit-service"`).
const SERVICE_TYPE: &str = "live-service";

/// Stateless `CapabilitiesService` handler. The response is a compile-time
/// constant shape, so no state, storage, or auth dependencies are needed.
#[derive(Default)]
pub struct CapabilitiesServiceImpl;

#[tonic::async_trait]
impl CapabilitiesService for CapabilitiesServiceImpl {
    async fn list_services(
        &self,
        _request: Request<ListServicesRequest>,
    ) -> Result<Response<ListServicesResponse>, Status> {
        let service = ServiceEntry {
            service_name: SERVICE_TYPE.to_owned(),
            grpc_services: vec![GrpcService {
                endpoints: vec![GrpcEndpoint {
                    service_name: LIVE_GRPC_SERVICE_NAME.to_owned(),
                }],
            }],
            rest_services: vec![],
        };

        Ok(Response::new(ListServicesResponse {
            services: vec![service],
            next_page_token: String::new(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact response `ListServices` must return, per the b.hv2 contract:
    /// a single `ServiceEntry` whose `service_name` is the discovery "type"
    /// key, one `GrpcService` with one `GrpcEndpoint` carrying the generated
    /// fully-qualified LiveService name, empty `rest_services`, and an empty
    /// `next_page_token`.
    fn expected_response() -> ListServicesResponse {
        ListServicesResponse {
            services: vec![ServiceEntry {
                service_name: "live-service".to_owned(),
                grpc_services: vec![GrpcService {
                    endpoints: vec![GrpcEndpoint {
                        service_name: LIVE_GRPC_SERVICE_NAME.to_owned(),
                    }],
                }],
                rest_services: vec![],
            }],
            next_page_token: String::new(),
        }
    }

    async fn call(request: ListServicesRequest) -> ListServicesResponse {
        CapabilitiesServiceImpl
            .list_services(Request::new(request))
            .await
            .expect("ListServices is static and must never fail")
            .into_inner()
    }

    #[tokio::test]
    async fn list_services_default_request_returns_exact_static_payload() {
        let response = call(ListServicesRequest::default()).await;
        assert_eq!(
            response,
            expected_response(),
            "ListServices must return exactly one ServiceEntry (service_name \
             \"live-service\", one GrpcService/GrpcEndpoint = generated \
             LiveService SERVICE_NAME, empty rest_services) with an empty \
             next_page_token"
        );
    }

    /// Pins the *literal* fully-qualified name advertised on the wire, not
    /// just "equals the generated constant". `expected_response` derives the
    /// endpoint from `SERVICE_NAME`, so an upstream proto package rename
    /// would silently track through both impl and expectation — this test
    /// converts such a rename into a visible failure, because the discovery
    /// operator's consumers key on the exact advertised string.
    #[tokio::test]
    async fn list_services_advertised_endpoint_is_fq_live_service_name() {
        let response = call(ListServicesRequest::default()).await;
        assert_eq!(
            response.services[0].grpc_services[0].endpoints[0].service_name,
            "nvidia.omniverse.live.v1alpha.LiveService",
            "the advertised gRPC endpoint must be the fully-qualified \
             LiveService name"
        );
    }

    #[tokio::test]
    async fn list_services_nonzero_page_fields_return_same_response_as_default() {
        let default_response = call(ListServicesRequest::default()).await;
        let paged_response = call(ListServicesRequest {
            page_size: 3,
            page_token: "opaque-page-token".to_owned(),
        })
        .await;
        assert_eq!(
            paged_response, default_response,
            "page_size / page_token must be ignored: a paged request returns \
             the identical response (including empty next_page_token) as the \
             default request"
        );
    }
}
