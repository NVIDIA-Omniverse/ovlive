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

//! The storage service refusing the permission probe as a malformed request
//! (`AuthResult::InvalidRequest`), end to end through the real handlers.
//!
//! # The incident
//!
//! On a dev run, clients probing scheme-less base paths (shaped like
//! `load-test/perf`) got a redacted `INTERNAL` and each call logged an
//! ERROR `internal error … auth backend returned HTTP 400`. `INTERNAL` is a
//! code Kit's client retries ten times, so one malformed path becomes a retry
//! storm against live-edit and the storage service, and operators see server
//! faults for a client mistake.
//!
//! # What is pinned
//!
//! - **Request-supplied path** (`CreateSession`, `ListSessions`): a real
//!   `RestStorageServiceClient` against a wiremock HEAD answering 400 gives
//!   `INVALID_ARGUMENT` with the approved message, one probe (not retried),
//!   and one INFO `RPC failed` line at `stage=auth` — no ERROR line.
//! - **Not every 4xx**: the same fixture answering 405 (a storage service
//!   without HEAD by-address) stays `INTERNAL` with one ERROR line naming
//!   `HTTP 405`.
//! - **Session-stored path** (the seven session-scoped handlers): a stub
//!   answering `InvalidRequest` gives `INTERNAL` and one ERROR line naming the
//!   stored-path refusal, because the client never sent that path in this
//!   RPC. Each handler is driven separately so each call site's
//!   `BasePathSource` argument is pinned.
//!
//! The log lines are the deliverable here (the ERROR line was the operator
//! symptom), so they are asserted structurally under the shared JSON capture —
//! see `rpc_failure_log_test.rs` for that rationale and
//! `log_capture.rs` for the capture scopes. The setup failures logged here are
//! all emitted inside the awaited handler, so the `WithSubscriber` scope
//! ([`capture`]) on a multi-thread runtime sees them.

use std::sync::Arc;

use fred::interfaces::SetsInterface;
use serde_json::Value;
use tonic::{Code, Request, Status};
use wiremock::matchers::method;
use wiremock::{Mock, MockServer, ResponseTemplate};

use live_edit::auth::permissions::PermissionsClient;
use live_edit::auth::{fast_retry, RestStorageServiceClient};
use live_edit::grpc::layer::{
    handle_create_layer, handle_delete_layer, handle_stat_layer, handle_update_layer,
};
use live_edit::grpc::presence::handle_join_session;
use live_edit::grpc::proto::{
    CreateSessionRequest, DeleteLayerRequest, DeleteSessionRequest, ListSessionsRequest,
    StatLayerRequest, SubscribeReadLayerRequest,
};
use live_edit::grpc::service::LiveEditService;
use live_edit::grpc::session::{
    handle_create_session, handle_delete_session, handle_list_sessions,
};
use live_edit::grpc::subscribe::handle_subscribe_read_layer;
use live_edit::grpc::BearerToken;
use live_edit::storage::test_support::TestValkey;

use super::helpers::{
    build_service, create_session, encode_join_frame, expect_err_status, join_header,
    make_create_data, make_create_header, open_join_streaming, streaming_create,
    IntoHandlerRequest as _, InvalidRequestPermissions, INVALID_REQUEST_BACKEND_STATUS,
};
use super::log_capture::{
    assert_common_fields, assert_field, capture, Captured, INTERNAL_ERROR_MSG, RPC_FAILURE_MSG,
};
use crate::common::owned_delta::{delta_to_ompf_bytes, empty_delta};

// =============================================================================
// Constants and builders
// =============================================================================

/// Bearer every request here carries. Distinctive so the whole-buffer sweep
/// cannot match an unrelated fixture.
const BEARER: &str = "izd-bearer-must-never-be-logged-91c4";

/// A scheme-less, relative base path shaped like the incident's. It is already
/// canonical, so the logged `base_path` must be exactly this string.
const SCHEMELESS_BASE_PATH: &str = "load-test/perf";

/// The approved `INVALID_ARGUMENT` wire message for a request-supplied path
/// the storage service refused with HTTP 400.
const INVALID_ARGUMENT_HTTP_400_MSG: &str = "base_path is not a resource address the storage \
                                             service accepts (storage service answered HTTP 400)";

/// A fragment of the stored-path refusal's un-redacted ERROR detail.
const STORED_PATH_DETAIL: &str = "stored base_path";

/// The base path `helpers::create_session` mints sessions under.
const STORED_BASE_PATH: &str = "/stage";

/// Auth timeout for the real REST client. Generous: the mock answers at once,
/// so this is never reached on the healthy path.
const REST_TIMEOUT_MS: u64 = 5_000;

fn with_bearer<T>(inner: T) -> Request<T> {
    let mut request = Request::new(inner);
    request.extensions_mut().insert(BearerToken::new(BEARER));
    request
}

/// [`with_bearer`] for an upload, whose request also carries the upload's
/// end-of-stream record.
fn upload_with_bearer<T>(
    stream: super::helpers::AttestedStreaming<T>,
) -> Request<tonic::codec::Streaming<T>> {
    let mut request = stream.into_handler_request();
    request.extensions_mut().insert(BearerToken::new(BEARER));
    request
}

fn create_session_request(path: &str, name: &str) -> Request<CreateSessionRequest> {
    with_bearer(CreateSessionRequest {
        path: path.to_owned(),
        name: name.to_owned(),
    })
}

fn list_sessions_request(path: &str) -> Request<ListSessionsRequest> {
    with_bearer(ListSessionsRequest {
        path: path.to_owned(),
    })
}

/// A wiremock storage service whose stat-by-address HEAD always answers
/// `status`, and a real REST permissions client pointed at it with a
/// three-attempt retry budget — so a retried status shows up as more than one
/// received request.
async fn rest_storage_answering(status: u16) -> (MockServer, Arc<dyn PermissionsClient>) {
    let mock_server = MockServer::start().await;
    Mock::given(method("HEAD"))
        .respond_with(ResponseTemplate::new(status))
        .mount(&mock_server)
        .await;
    let client = RestStorageServiceClient::new(&mock_server.uri(), REST_TIMEOUT_MS, fast_retry(3))
        .expect("the REST client must build against the mock's URI");
    (mock_server, Arc::new(client))
}

async fn probes_received(mock_server: &MockServer) -> usize {
    mock_server
        .received_requests()
        .await
        .expect("wiremock request recording is on by default")
        .len()
}

/// Assert the capture holds no ERROR-level event at all.
fn assert_no_error_event(captured: &Captured) {
    let errors: Vec<&Value> = captured
        .events
        .iter()
        .filter(|e| e["level"] == "ERROR")
        .collect();
    assert!(
        errors.is_empty(),
        "a client's malformed base path must not log at ERROR; full capture:\n{}",
        captured.text
    );
}

/// The one completion line, asserting it is `RPC failed` at INFO, `stage=auth`,
/// `INVALID_ARGUMENT`, carrying `base_path` and a status message naming the
/// storage service's status — and that nothing else in the capture is a
/// completion line or an ERROR.
fn assert_one_info_invalid_argument_line(captured: &Captured, endpoint: &str) {
    assert_eq!(
        captured.completion_lines().len(),
        1,
        "one failure, one completion line; full capture:\n{}",
        captured.text
    );
    let event = captured.only(RPC_FAILURE_MSG);
    assert_common_fields(event, "INFO", endpoint, "auth", "INVALID_ARGUMENT");
    assert_field(event, "base_path", SCHEMELESS_BASE_PATH.into());
    let status_message = event["fields"]["status_message"]
        .as_str()
        .unwrap_or_default();
    assert!(
        status_message.contains("HTTP 400"),
        "the line must carry the storage service's status; got {event}"
    );
    assert_no_error_event(captured);
    captured.assert_absent_everywhere(BEARER);
}

/// The one completion line for a server fault: ERROR `internal error` at
/// `stage=auth` whose `error` detail contains every string in `detail_has`.
fn assert_one_internal_error_line(captured: &Captured, endpoint: &str, detail_has: &[&str]) {
    assert_eq!(
        captured.completion_lines().len(),
        1,
        "one failure, one completion line; full capture:\n{}",
        captured.text
    );
    let event = captured.only(INTERNAL_ERROR_MSG);
    assert_common_fields(event, "ERROR", endpoint, "auth", "INTERNAL");
    let detail = event["fields"]["error"].as_str().unwrap_or_default();
    for needle in detail_has {
        assert!(
            detail.contains(needle),
            "the ERROR line's detail must contain `{needle}`; got {event}"
        );
    }
    captured.assert_absent_everywhere(BEARER);
}

// =============================================================================
// (a) Request-supplied path, real REST client, HTTP 400 → INVALID_ARGUMENT
// =============================================================================

/// The incident path: `ListSessions` on a scheme-less base path the storage
/// service answers 400 for. Before the fix this answered `INTERNAL` (which Kit
/// retries) and logged ERROR `internal error`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn list_sessions_storage_400_answers_invalid_argument_and_logs_one_info_line() {
    let tv = TestValkey::setup().await;
    let (mock_server, permissions) = rest_storage_answering(400).await;
    let svc = build_service(&tv, permissions);

    let (result, captured) = capture(handle_list_sessions(
        &svc,
        list_sessions_request(SCHEMELESS_BASE_PATH),
    ))
    .await;

    let status = expect_err_status(result, "a refused base path must fail ListSessions");
    assert_eq!(status.code(), Code::InvalidArgument, "got {status:?}");
    assert_eq!(status.message(), INVALID_ARGUMENT_HTTP_400_MSG);
    assert_eq!(
        probes_received(&mock_server).await,
        1,
        "a 400 is a verdict and must not be retried"
    );
    assert_one_info_invalid_argument_line(&captured, "ListSessions");
}

/// `CreateSession` on the same path answers the same, logs the same, and
/// creates nothing: the refusal lands before any storage write.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_session_storage_400_answers_invalid_argument_and_creates_nothing() {
    let tv = TestValkey::setup().await;
    let (mock_server, permissions) = rest_storage_answering(400).await;
    let svc = build_service(&tv, permissions);

    let (result, captured) = capture(handle_create_session(
        &svc,
        create_session_request(SCHEMELESS_BASE_PATH, "izd-refused-session"),
    ))
    .await;

    let status = expect_err_status(result, "a refused base path must fail CreateSession");
    assert_eq!(status.code(), Code::InvalidArgument, "got {status:?}");
    assert_eq!(status.message(), INVALID_ARGUMENT_HTTP_400_MSG);
    assert_eq!(
        probes_received(&mock_server).await,
        1,
        "a 400 is a verdict and must not be retried"
    );
    assert_one_info_invalid_argument_line(&captured, "CreateSession");

    let pool = &tv.client().pool;
    let indexed: u64 = pool
        .scard(tv.key_schema().path_index(SCHEMELESS_BASE_PATH))
        .await
        .expect("SCARD on the path index must succeed");
    let named: u64 = pool
        .scard(tv.key_schema().path_names_index(SCHEMELESS_BASE_PATH))
        .await
        .expect("SCARD on the path names index must succeed");
    assert_eq!(
        (indexed, named),
        (0, 0),
        "a refused CreateSession must not index a session under the path"
    );
}

// =============================================================================
// (b) The change did not widen to every 4xx: HTTP 405 stays INTERNAL
// =============================================================================

/// 405 means the storage service does not route HEAD by-address — a
/// deployment fault that must stay loud: redacted `INTERNAL` and one ERROR
/// line naming the status, even on a request-supplied path.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn list_sessions_storage_405_stays_internal_and_logs_one_error_line() {
    let tv = TestValkey::setup().await;
    let (mock_server, permissions) = rest_storage_answering(405).await;
    let svc = build_service(&tv, permissions);

    let (result, captured) = capture(async {
        let result = handle_list_sessions(&svc, list_sessions_request(SCHEMELESS_BASE_PATH)).await;
        // Taken apart inside the capture, so an unreported fault's drop
        // fallback would land in it as a second line.
        result
            .map(|_| ())
            .map_err(|s| (s.code(), s.message().to_owned()))
    })
    .await;

    let (code, message) = result.expect_err("a 405 must fail ListSessions");
    assert_eq!(code, Code::Internal);
    assert_eq!(message, INTERNAL_ERROR_MSG);
    assert_eq!(probes_received(&mock_server).await, 1);
    assert_one_internal_error_line(&captured, "ListSessions", &["HTTP 405"]);
}

// =============================================================================
// (c) Stub InvalidRequest: request-supplied paths answer INVALID_ARGUMENT
// =============================================================================

/// `CreateSession` passes `BasePathSource::Request`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_session_invalid_request_answers_invalid_argument() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(InvalidRequestPermissions));

    let (result, captured) = capture(handle_create_session(
        &svc,
        create_session_request(SCHEMELESS_BASE_PATH, "izd-stub-session"),
    ))
    .await;

    let status = expect_err_status(result, "must be refused");
    assert_eq!(status.code(), Code::InvalidArgument, "got {status:?}");
    assert_eq!(status.message(), INVALID_ARGUMENT_HTTP_400_MSG);
    assert_one_info_invalid_argument_line(&captured, "CreateSession");
}

/// `ListSessions` passes `BasePathSource::Request`, and the refusal is an
/// error — never the empty list `PathNotFound` answers.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn list_sessions_invalid_request_answers_invalid_argument_not_an_empty_list() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(InvalidRequestPermissions));

    let (result, captured) = capture(handle_list_sessions(
        &svc,
        list_sessions_request(SCHEMELESS_BASE_PATH),
    ))
    .await;

    let status = expect_err_status(result, "must be refused, not answered with an empty list");
    assert_eq!(status.code(), Code::InvalidArgument, "got {status:?}");
    assert_eq!(status.message(), INVALID_ARGUMENT_HTTP_400_MSG);
    assert_one_info_invalid_argument_line(&captured, "ListSessions");
}

// =============================================================================
// (d) Stub InvalidRequest: session-stored paths answer INTERNAL
// =============================================================================
//
// Sessions are minted straight through the store: every handler route to a
// session runs the permission probe this section refuses. Each test drops the
// status inside the capture so a fault the boundary failed to mark reported
// would show up as a second, context-free ERROR line.

/// The stored-path refusal for one session-scoped handler: `INTERNAL`, and
/// one ERROR line at `stage=auth` carrying the session, its stored base path,
/// and a detail naming the refusal and the storage service's status.
fn assert_stored_path_refusal(
    status: (Code, String),
    captured: &Captured,
    endpoint: &str,
    session_id: &str,
) {
    let (code, message) = status;
    assert_eq!(
        code,
        Code::Internal,
        "{endpoint}: a session's stored path refused as malformed is a server-side \
         inconsistency, not a client error; got {code:?} {message:?}"
    );
    assert_eq!(
        message, INTERNAL_ERROR_MSG,
        "{endpoint}: the wire must stay redacted"
    );
    assert_one_internal_error_line(
        captured,
        endpoint,
        &[STORED_PATH_DETAIL, INVALID_REQUEST_BACKEND_STATUS],
    );
    let event = captured.only(INTERNAL_ERROR_MSG);
    assert_field(event, "session_id", session_id.into());
    assert_field(event, "base_path", STORED_BASE_PATH.into());
}

/// Run `future` (a handler call) under the capture, reducing its error to
/// `(code, message)` inside the capture scope.
async fn capture_failure<F, T>(future: F, endpoint: &str) -> ((Code, String), Captured)
where
    F: std::future::Future<Output = Result<T, Status>>,
{
    let (result, captured) = capture(async {
        future
            .await
            .map(|_| ())
            .map_err(|s| (s.code(), s.message().to_owned()))
    })
    .await;
    match result {
        Err(status) => (status, captured),
        Ok(()) => panic!("{endpoint}: a refused stored base path must fail the RPC"),
    }
}

async fn refused_stored_path_service(name: &str) -> (TestValkey, LiveEditService, String) {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(InvalidRequestPermissions));
    let (session_id, _token) = create_session(&tv, &svc, name, true).await;
    (tv, svc, session_id)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_stored_path_invalid_request_answers_internal() {
    let (_tv, svc, session_id) = refused_stored_path_service("izd-create-layer").await;
    let stream = streaming_create(vec![
        make_create_header(&session_id, "izd-layer"),
        make_create_data(delta_to_ompf_bytes(&empty_delta(), &[])),
    ]);

    let (status, captured) = capture_failure(
        handle_create_layer(&svc, upload_with_bearer(stream)),
        "CreateLayer",
    )
    .await;

    assert_stored_path_refusal(status, &captured, "CreateLayer", &session_id);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_stored_path_invalid_request_answers_internal() {
    let (_tv, svc, session_id) = refused_stored_path_service("izd-update-layer").await;
    let request = crate::common::update_request_with_bearer(
        &session_id,
        "izd-layer",
        1,
        1000,
        "izd-client",
        BEARER,
    );

    let (status, captured) =
        capture_failure(handle_update_layer(&svc, request), "UpdateLayer").await;

    assert_stored_path_refusal(status, &captured, "UpdateLayer", &session_id);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stat_layer_stored_path_invalid_request_answers_internal() {
    let (_tv, svc, session_id) = refused_stored_path_service("izd-stat-layer").await;
    let request = with_bearer(StatLayerRequest {
        session_id: session_id.clone(),
        layer_name: "izd-layer".to_owned(),
    });

    let (status, captured) = capture_failure(handle_stat_layer(&svc, request), "StatLayer").await;

    assert_stored_path_refusal(status, &captured, "StatLayer", &session_id);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_layer_stored_path_invalid_request_answers_internal() {
    let (_tv, svc, session_id) = refused_stored_path_service("izd-delete-layer").await;
    let request = with_bearer(DeleteLayerRequest {
        session_id: session_id.clone(),
        layer_name: "izd-layer".to_owned(),
    });

    let (status, captured) =
        capture_failure(handle_delete_layer(&svc, request), "DeleteLayer").await;

    assert_stored_path_refusal(status, &captured, "DeleteLayer", &session_id);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_session_stored_path_invalid_request_answers_internal() {
    let (_tv, svc, session_id) = refused_stored_path_service("izd-delete-session").await;
    let request = with_bearer(DeleteSessionRequest {
        session_id: session_id.clone(),
    });

    let (status, captured) =
        capture_failure(handle_delete_session(&svc, request), "DeleteSession").await;

    assert_stored_path_refusal(status, &captured, "DeleteSession", &session_id);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn join_session_stored_path_invalid_request_answers_internal() {
    let (_tv, svc, session_id) = refused_stored_path_service("izd-join-session").await;
    let (stream, tx) = open_join_streaming();
    tx.send(encode_join_frame(&join_header(&session_id, "izd-app")))
        .expect("the client channel must accept the header frame");

    let (status, captured) = capture_failure(
        handle_join_session(&svc, with_bearer(stream)),
        "JoinSession",
    )
    .await;
    drop(tx);

    assert_stored_path_refusal(status, &captured, "JoinSession", &session_id);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn subscribe_read_layer_stored_path_invalid_request_answers_internal() {
    let (_tv, svc, session_id) = refused_stored_path_service("izd-subscribe").await;
    let request = with_bearer(SubscribeReadLayerRequest {
        session_id: session_id.clone(),
        layer_name: "izd-layer".to_owned(),
        object_id: 0,
        sequence_num: 0,
        client_id: "izd-subscriber".to_owned(),
    });

    let (status, captured) = capture_failure(
        handle_subscribe_read_layer(&svc, request),
        "SubscribeReadLayer",
    )
    .await;

    assert_stored_path_refusal(status, &captured, "SubscribeReadLayer", &session_id);
}
