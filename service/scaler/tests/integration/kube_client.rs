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

//! `kube::KubeClient` against a `wiremock` API server on loopback: the
//! requests it sends (paths, verbs, query, content type, bearer token, body)
//! and how it reads the answers.
//!
//! The success-path mocks require the bearer token, so a request sent
//! without it, or with a stale one, matches nothing and gets wiremock's
//! default 404, which the client turns into an error the test sees.

use std::io::Write;

use live_edit_scaler::kube::{Endpoint, KubeClient, KubeError, Scale};
use reqwest::StatusCode;
use serde_json::json;
use tempfile::NamedTempFile;
use wiremock::matchers::{body_json, header, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

const NAMESPACE: &str = "editing";
const DEPLOYMENT: &str = "live-edit-service";
const SERVICE: &str = "live-edit-service";
const TOKEN: &str = "token-one";

const SLICES_PATH: &str = "/apis/discovery.k8s.io/v1/namespaces/editing/endpointslices";
const SCALE_PATH: &str = "/apis/apps/v1/namespaces/editing/deployments/live-edit-service/scale";
/// The Deployment object itself: carries the pod template, so the scaler
/// must never request it.
const DEPLOYMENT_PATH: &str = "/apis/apps/v1/namespaces/editing/deployments/live-edit-service";

/// A token file holding `token` (with the trailing newline a projected token
/// file may carry).
fn token_file(token: &str) -> NamedTempFile {
    let mut file = NamedTempFile::new().expect("create a token file");
    writeln!(file, "{token}").expect("write the token");
    file
}

fn client(server: &MockServer, token: &NamedTempFile) -> KubeClient {
    KubeClient::new(&server.uri(), token.path(), None).expect("an http client for the mock")
}

fn bearer(token: &str) -> String {
    format!("Bearer {token}")
}

fn slice_endpoint(uid: &str, address: &str, conditions: serde_json::Value) -> serde_json::Value {
    json!({
        "addresses": [address],
        "conditions": conditions,
        "targetRef": { "kind": "Pod", "name": format!("server-{uid}"), "uid": uid }
    })
}

/// The EndpointSlices of the service: what the client keeps and what it
/// skips.
///
/// - An IPv4 slice whose `metrics` port is 9091 (its other port is not the
///   metrics one): a Ready pod, an unready one, a terminating one, one with
///   no conditions at all (ready by the API's definition), and two endpoints
///   it cannot key or annotate — no `targetRef`, and a `targetRef` that is
///   not a Pod.
/// - An IPv6 slice, `metrics` port 9092, listing the first pod again (one per
///   address family; kept once, at its first listing) and a pod only it has.
/// - An FQDN slice, which is not a pod address and is skipped whole.
#[tokio::test]
async fn list_endpoints_parses_every_pod_once_with_its_state_and_metrics_port() {
    let server = MockServer::start().await;
    let token = token_file(TOKEN);
    Mock::given(method("GET"))
        .and(path(SLICES_PATH))
        .and(query_param(
            "labelSelector",
            "kubernetes.io/service-name=live-edit-service",
        ))
        .and(header("authorization", bearer(TOKEN).as_str()))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "kind": "EndpointSliceList",
            "items": [
                {
                    "addressType": "IPv4",
                    "ports": [
                        { "name": "grpc", "port": 50051 },
                        { "name": "metrics", "port": 9091 }
                    ],
                    "endpoints": [
                        slice_endpoint("uid-ready", "10.0.0.1",
                            json!({ "ready": true, "terminating": false })),
                        slice_endpoint("uid-unready", "10.0.0.2", json!({ "ready": false })),
                        slice_endpoint("uid-terminating", "10.0.0.3",
                            json!({ "ready": false, "terminating": true })),
                        slice_endpoint("uid-no-conditions", "10.0.0.4", json!({})),
                        { "addresses": ["10.0.0.5"], "conditions": { "ready": true } },
                        {
                            "addresses": ["10.0.0.6"],
                            "conditions": { "ready": true },
                            "targetRef": { "kind": "Node", "name": "node-a", "uid": "uid-node" }
                        }
                    ]
                },
                {
                    "addressType": "IPv6",
                    "ports": [{ "name": "metrics", "port": 9092 }],
                    "endpoints": [
                        slice_endpoint("uid-ready", "fd00::1", json!({ "ready": true })),
                        slice_endpoint("uid-v6", "fd00::7", json!({ "ready": true }))
                    ]
                },
                {
                    "addressType": "FQDN",
                    "ports": [{ "name": "metrics", "port": 9093 }],
                    "endpoints": [slice_endpoint("uid-fqdn", "pod.example.com", json!({}))]
                }
            ]
        })))
        .expect(1)
        .mount(&server)
        .await;

    let endpoints = client(&server, &token)
        .list_endpoints(NAMESPACE, SERVICE)
        .await
        .expect("the list must parse");

    let endpoint = |uid: &str, address: &str, port: u16, ready: bool, terminating: bool| Endpoint {
        uid: uid.to_owned(),
        name: format!("server-{uid}"),
        address: address.to_owned(),
        metrics_port: Some(port),
        ready,
        terminating,
    };
    assert_eq!(
        endpoints,
        vec![
            endpoint("uid-ready", "10.0.0.1", 9091, true, false),
            endpoint("uid-unready", "10.0.0.2", 9091, false, false),
            endpoint("uid-terminating", "10.0.0.3", 9091, false, true),
            endpoint("uid-no-conditions", "10.0.0.4", 9091, true, false),
            endpoint("uid-v6", "fd00::7", 9092, true, false),
        ]
    );
    assert_eq!(
        endpoints[4].url("/scaling/v1").as_deref(),
        Some("http://[fd00::7]:9092/scaling/v1"),
        "an IPv6 address is bracketed in the snapshot URL"
    );
}

/// A slice without a port named `metrics` gives its pods no metrics port,
/// and so no snapshot URL.
#[tokio::test]
async fn list_endpoints_without_a_metrics_port_gives_no_snapshot_url() {
    let server = MockServer::start().await;
    let token = token_file(TOKEN);
    Mock::given(method("GET"))
        .and(path(SLICES_PATH))
        .and(header("authorization", bearer(TOKEN).as_str()))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "items": [{
                "addressType": "IPv4",
                "ports": [{ "name": "grpc", "port": 50051 }],
                "endpoints": [slice_endpoint("uid-a", "10.0.0.1", json!({ "ready": true }))]
            }]
        })))
        .mount(&server)
        .await;

    let endpoints = client(&server, &token)
        .list_endpoints(NAMESPACE, SERVICE)
        .await
        .expect("the list must parse");
    assert_eq!(endpoints.len(), 1);
    assert_eq!(endpoints[0].metrics_port, None);
    assert_eq!(endpoints[0].url("/scaling/v1"), None);
}

/// `GET …/deployments/<name>/scale` reads `spec.replicas` and
/// `status.replicas`; an absent `status.replicas` (or no `status` at all)
/// reads as 0.
#[tokio::test]
async fn get_scale_reads_spec_and_status_replicas_and_an_absent_status_as_zero() {
    for (body, expected) in [
        (
            json!({ "kind": "Scale", "spec": { "replicas": 3 }, "status": { "replicas": 4, "selector": "app=x" } }),
            Scale {
                replicas: 3,
                status_replicas: 4,
            },
        ),
        (
            json!({ "spec": { "replicas": 3 }, "status": { "selector": "app=x" } }),
            Scale {
                replicas: 3,
                status_replicas: 0,
            },
        ),
        (
            json!({ "spec": { "replicas": 5 } }),
            Scale {
                replicas: 5,
                status_replicas: 0,
            },
        ),
    ] {
        let server = MockServer::start().await;
        let token = token_file(TOKEN);
        Mock::given(method("GET"))
            .and(path(SCALE_PATH))
            .and(header("authorization", bearer(TOKEN).as_str()))
            .respond_with(ResponseTemplate::new(200).set_body_json(body.clone()))
            .expect(1)
            .mount(&server)
            .await;
        let scale = client(&server, &token)
            .get_scale(NAMESPACE, DEPLOYMENT)
            .await
            .expect("the scale must parse");
        assert_eq!(scale, expected, "{body}");
    }
}

/// `patch_replicas` sends a JSON merge patch of `spec.replicas` to the
/// `/scale` subresource, and `patch_pod_deletion_cost` one of the pod's
/// deletion-cost annotation (as a string, which annotations must be).
/// Neither call, nor `get_scale`, ever requests the Deployment object
/// itself: no mock serves it, and the server's request log is checked.
#[tokio::test]
async fn patches_are_merge_patches_and_the_deployment_object_is_never_requested() {
    let server = MockServer::start().await;
    let token = token_file(TOKEN);
    Mock::given(method("PATCH"))
        .and(path(SCALE_PATH))
        .and(header("authorization", bearer(TOKEN).as_str()))
        .and(header("content-type", "application/merge-patch+json"))
        .and(body_json(json!({ "spec": { "replicas": 7 } })))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(json!({ "spec": { "replicas": 7 } })),
        )
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path(SCALE_PATH))
        .and(header("authorization", bearer(TOKEN).as_str()))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(json!({ "spec": { "replicas": 7 } })),
        )
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("PATCH"))
        .and(path("/api/v1/namespaces/editing/pods/server-a"))
        .and(header("authorization", bearer(TOKEN).as_str()))
        .and(header("content-type", "application/merge-patch+json"))
        .and(body_json(json!({
            "metadata": { "annotations": { "controller.kubernetes.io/pod-deletion-cost": "-3" } }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "kind": "Pod" })))
        .expect(1)
        .mount(&server)
        .await;

    let kube = client(&server, &token);
    kube.patch_replicas(NAMESPACE, DEPLOYMENT, 7)
        .await
        .expect("the scale patch must be accepted");
    kube.get_scale(NAMESPACE, DEPLOYMENT)
        .await
        .expect("the scale read must be accepted");
    kube.patch_pod_deletion_cost(NAMESPACE, "server-a", -3)
        .await
        .expect("the annotation patch must be accepted");

    let requests = server
        .received_requests()
        .await
        .expect("request recording is on");
    assert_eq!(requests.len(), 3, "exactly the three calls: {requests:?}");
    assert!(
        requests.iter().all(|r| r.url.path() != DEPLOYMENT_PATH),
        "the client must never request the Deployment object, which carries the pod template"
    );
}

/// The service-account token is re-read from its file on every request, so
/// a token the kubelet rotated is used on the next request.
#[tokio::test]
async fn a_rotated_token_file_is_used_on_the_next_request() {
    let server = MockServer::start().await;
    let scale = json!({ "spec": { "replicas": 3 }, "status": { "replicas": 3 } });
    for token in ["token-one", "token-two"] {
        Mock::given(method("GET"))
            .and(path(SCALE_PATH))
            .and(header("authorization", bearer(token).as_str()))
            .respond_with(ResponseTemplate::new(200).set_body_json(scale.clone()))
            .expect(1)
            .mount(&server)
            .await;
    }
    let file = token_file("token-one");
    let kube = client(&server, &file);
    kube.get_scale(NAMESPACE, DEPLOYMENT)
        .await
        .expect("the first token is accepted");

    std::fs::write(file.path(), "token-two\n").expect("rotate the token");
    kube.get_scale(NAMESPACE, DEPLOYMENT)
        .await
        .expect("the rotated token is used on the next request");
    // `expect(1)` on each mock is verified when the server drops.
}

/// A non-2xx answer is an error carrying the status, whatever the call; a
/// 2xx body that does not parse is a decode error. Neither yields a value
/// the caller could act on.
#[tokio::test]
async fn non_success_answers_and_unparsable_bodies_are_errors() {
    let server = MockServer::start().await;
    let token = token_file(TOKEN);
    Mock::given(method("GET"))
        .and(path(SLICES_PATH))
        .respond_with(ResponseTemplate::new(403).set_body_string("forbidden: endpointslices"))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path(SCALE_PATH))
        .respond_with(ResponseTemplate::new(500).set_body_string("etcd unavailable"))
        .mount(&server)
        .await;
    Mock::given(method("PATCH"))
        .and(path(SCALE_PATH))
        .respond_with(ResponseTemplate::new(409).set_body_string("conflict"))
        .mount(&server)
        .await;
    Mock::given(method("PATCH"))
        .and(path("/api/v1/namespaces/editing/pods/server-a"))
        .respond_with(ResponseTemplate::new(404).set_body_string("pods \"server-a\" not found"))
        .mount(&server)
        .await;
    let kube = client(&server, &token);

    let status_of = |e: KubeError| match e {
        KubeError::Status { status, .. } => status,
        other => panic!("expected a status error, got {other:?}"),
    };
    assert_eq!(
        status_of(kube.list_endpoints(NAMESPACE, SERVICE).await.unwrap_err()),
        StatusCode::FORBIDDEN
    );
    assert_eq!(
        status_of(kube.get_scale(NAMESPACE, DEPLOYMENT).await.unwrap_err()),
        StatusCode::INTERNAL_SERVER_ERROR
    );
    assert_eq!(
        status_of(
            kube.patch_replicas(NAMESPACE, DEPLOYMENT, 4)
                .await
                .unwrap_err()
        ),
        StatusCode::CONFLICT
    );
    assert_eq!(
        status_of(
            kube.patch_pod_deletion_cost(NAMESPACE, "server-a", 1)
                .await
                .unwrap_err()
        ),
        StatusCode::NOT_FOUND
    );

    let garbled = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path(SCALE_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_string("{ not json"))
        .mount(&garbled)
        .await;
    let err = client(&garbled, &token)
        .get_scale(NAMESPACE, DEPLOYMENT)
        .await
        .unwrap_err();
    assert!(matches!(err, KubeError::Decode { .. }), "got {err:?}");
}

/// A token file that cannot be read fails the request before anything is
/// sent.
#[tokio::test]
async fn an_unreadable_token_file_fails_the_request_without_sending_it() {
    let server = MockServer::start().await;
    let dir = tempfile::tempdir().expect("a temporary directory");
    let kube = KubeClient::new(&server.uri(), dir.path().join("absent-token"), None)
        .expect("an http client for the mock");
    let err = kube.get_scale(NAMESPACE, DEPLOYMENT).await.unwrap_err();
    assert!(matches!(err, KubeError::Token { .. }), "got {err:?}");
    assert_eq!(
        server
            .received_requests()
            .await
            .expect("recording is on")
            .len(),
        0
    );
}
