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

//! Completion logging for the two streaming RPCs, `SubscribeReadLayer` and
//! `JoinSession` (SDD §17.2.1): every non-OK completion emits exactly one line,
//! whether the status was decided at **setup** (before the response stream
//! opened) or at **teardown** (inside the spawned task, after it opened).
//!
//! # Why these tests assert on log output
//!
//! The line is the deliverable. Before it, a failed subscription was visible
//! only as a `+1` on `live_endpoint_calls_total{grpc_code=…}`; and since every
//! transient subscribe failure reaches the client as `UNAVAILABLE`, the
//! `status_message` on this line is the only per-event record of *which* read
//! failed. So the assertions are structural — level, message, each field,
//! present or absent — never cosmetic. Each test also pins the wire status
//! next to the line, so a log assertion cannot be satisfied by a handler that
//! started answering something else.
//!
//! # Two capture scopes
//!
//! * **Setup** lines are emitted by the handler wrapper inside the awaited
//!   handler future, so they are captured with `WithSubscriber`
//!   ([`capture`]) on a multi-thread runtime.
//! * **Teardown** lines are emitted by a task the handler spawned, which
//!   inherits no dispatcher. Those tests run on a **current-thread** runtime
//!   (`#[tokio::test]`) under a thread-local [`ThreadCapture`], so the spawned
//!   task is polled on the thread holding the guard. They wait for the task's
//!   observable end — the stream-exit counter for a subscription (incremented
//!   after the line), the presence cleanup tracker for a join — and only then
//!   read the capture. `TestValkey` cannot clean up on a current-thread
//!   runtime, so each of those tests ends with [`purge_keys`].
//!
//! # A second, silent activation condition
//!
//! The tests driving a `LiveEditService` seam (`outer_send_deadline_for_test`,
//! `catch_up_chunk_seam_for_test`, `catch_up_out_of_window_seam_for_test`,
//! `lease_keeper_seam_for_test`, `presence_replay_seam_for_test`), the stream
//! multiplexer's fault queue, or a
//! storage fault injector exist only under **`test-fault-injection`**; without
//! it they are compiled out and the run reports fewer tests.
//!
//! ```text
//! cargo nextest run -p live-edit --test integration \
//!     --features test-support,test-fault-injection grpc::stream_completion_log_test
//! ```
//!
//! # Wait budget
//!
//! Every wait is a deadline, not a measurement: `SETTLE` (10 s) per awaited
//! message, drain or condition; the healthy path pays none of it. The
//! heaviest test is
//! `subscribe_layer_deleted_mid_stream_logs_one_not_found_with_caught_up_and_nothing_before`:
//! one catch-up drain (≤ 4 messages, 40 s) + two relayed updates (20 s) + one
//! terminal drain (10 s) + one stream-exit wait (10 s) ≈ 80 s at every
//! budget's full value, inside nextest's 120 s hang-killer. The slow-reader
//! test's own deadline is the 300 ms override, not a wait. The one wait that
//! is paid in full is the catch-up-capacity test's: its refusal IS
//! `TestValkey`'s 2 s capacity budget running out.

use std::sync::Arc;
use std::time::Duration;

use futures::{Stream, StreamExt};
use prost::Message;
use serde_json::Value;
use tonic::codec::{BufferSettings, Streaming};
use tonic::{Code, Request, Status};
use tonic_prost::ProstCodec;

use live_edit::auth::userinfo::UserinfoClient;
use live_edit::auth::{AuthError, UserInfo};
use live_edit::grpc::layer::{handle_delete_layer, handle_update_layer};
use live_edit::grpc::presence::handle_join_session;
use live_edit::grpc::proto::{
    join_session_request, join_session_response, subscribe_read_layer_response, DeleteLayerRequest,
    JoinSessionHeader, JoinSessionRequest, SessionMessageRequest, SubscribeReadLayerRequest,
    SubscribeReadLayerResponse,
};
use live_edit::grpc::service::LiveEditService;
use live_edit::grpc::subscribe::handle_subscribe_read_layer;
use live_edit::grpc::BearerToken;
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::{
    Metrics, STREAM_EXIT_CLIENT_DISCONNECT, STREAM_EXIT_ERROR, STREAM_EXIT_REASONS,
    STREAM_EXIT_SESSION_CANCEL, STREAM_EXIT_SHUTDOWN,
};
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;
use oodf_ffi::{CommandDiffChangeSection, OwnedDelta, StructCommand};

use super::helpers::{
    build_service, build_service_with_userinfo, create_session_and_layer, expect_err_status,
    make_request, make_update_data, make_update_header, streaming_update, AllowAllPermissions,
    DenyAllPermissions,
};
use super::log_capture::{
    assert_common_fields, assert_field, assert_field_absent, capture, purge_keys, Captured,
    ThreadCapture, INTERNAL_ERROR_MSG, RPC_FAILURE_MSG,
};

// =============================================================================
// Constants and builders
// =============================================================================

/// Upper bound on any single wait. Unreachable on the healthy path.
const SETTLE: Duration = Duration::from_secs(10);

/// Bearer every request here carries. Distinctive so the whole-buffer sweep
/// cannot match an unrelated fixture; the mock permission clients ignore it.
const BEARER: &str = "stream-log-bearer-must-never-be-logged-4d2b";

/// The subscriber's `client_id`. Distinct from the writer's `test-client`
/// (`make_update_header`) so self-echo suppression never swallows an update.
const SUBSCRIBER: &str = "stream-log-subscriber";

/// A `session_id` that fails `validate_session_id` on its ESC byte.
const CONTROL_CHAR_SESSION_ID: &str = "stream-log-bad\u{1b}[31m-session";

/// A distinctive `user_application`, so a leak into the join teardown line is
/// greppable. It passes `validate_user_application`.
const JOIN_APP: &str = "stream-log-join-app-7e21";

fn subscribe_request(
    session_id: &str,
    layer_name: &str,
    object_id: u64,
    sequence_num: u64,
) -> Request<SubscribeReadLayerRequest> {
    let mut request = Request::new(SubscribeReadLayerRequest {
        session_id: session_id.to_owned(),
        layer_name: layer_name.to_owned(),
        object_id,
        sequence_num,
        client_id: SUBSCRIBER.to_owned(),
    });
    request.extensions_mut().insert(BearerToken::new(BEARER));
    request
}

fn delete_layer_request(session_id: &str, layer_name: &str) -> Request<DeleteLayerRequest> {
    let mut request = Request::new(DeleteLayerRequest {
        session_id: session_id.to_owned(),
        layer_name: layer_name.to_owned(),
    });
    request.extensions_mut().insert(BearerToken::new(BEARER));
    request
}

/// Encode `JoinSessionRequest`s into a gRPC-framed request stream.
fn streaming_join(messages: Vec<JoinSessionRequest>) -> Streaming<JoinSessionRequest> {
    use bytes::BufMut;
    let mut buf = bytes::BytesMut::new();
    for msg in &messages {
        let encoded = msg.encode_to_vec();
        buf.put_u8(0);
        buf.put_u32(encoded.len() as u32);
        buf.extend_from_slice(&encoded);
    }
    join_stream_from_frames(buf.freeze())
}

fn join_stream_from_frames(frames: bytes::Bytes) -> Streaming<JoinSessionRequest> {
    let body = http_body_util::Full::new(frames);
    let decoder = ProstCodec::<(), JoinSessionRequest>::raw_decoder(BufferSettings::default());
    Streaming::new_request(decoder, body, None, None)
}

fn join_header(session_id: &str) -> JoinSessionRequest {
    JoinSessionRequest {
        request_type: Some(join_session_request::RequestType::Header(
            JoinSessionHeader {
                session_id: session_id.to_owned(),
                user_application: JOIN_APP.to_owned(),
                client_keepalive: false,
            },
        )),
    }
}

fn join_request(stream: Streaming<JoinSessionRequest>) -> Request<Streaming<JoinSessionRequest>> {
    let mut request = Request::new(stream);
    request.extensions_mut().insert(BearerToken::new(BEARER));
    request
}

/// A structural delta adding `section_id` under the root: each commit
/// advances the sequence and relays one steady-state update.
fn add_root_child(section_id: u64) -> OwnedDelta {
    OwnedDelta {
        is_diff: false,
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 0,
            section_id,
            section_name: format!("sec-{section_id}"),
            section_type: 0,
            section_order: 0,
        })],
        ..Default::default()
    }
}

async fn commit(svc: &LiveEditService, session_id: &str, layer_name: &str, object_id: u64, n: u64) {
    let ompf = crate::common::owned_delta::delta_to_ompf_bytes(&add_root_child(n), &[]);
    handle_update_layer(
        svc,
        make_request(streaming_update(vec![
            make_update_header(session_id, layer_name, object_id),
            make_update_data(ompf),
        ])),
    )
    .await
    .unwrap_or_else(|e| panic!("UpdateLayer (section {n}) must succeed: {e}"));
}

/// A session created straight through the store, with the cancellation token
/// `CreateSession` would have registered.
async fn create_session(tv: &TestValkey, svc: &LiveEditService, name: &str) -> String {
    let store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = store
        .create(name, "/stage", "test-user")
        .await
        .expect("session creation must succeed");
    svc.session_tokens.insert(
        session_id.clone(),
        tokio_util::sync::CancellationToken::new().into(),
    );
    session_id
}

// =============================================================================
// Stream helpers
// =============================================================================

type SubscribeStream =
    std::pin::Pin<Box<dyn Stream<Item = Result<SubscribeReadLayerResponse, Status>> + Send>>;

async fn next_item(
    stream: &mut SubscribeStream,
    what: &str,
) -> Option<Result<SubscribeReadLayerResponse, Status>> {
    tokio::time::timeout(SETTLE, stream.next())
        .await
        .unwrap_or_else(|_| panic!("{what} did not arrive within {SETTLE:?}"))
}

/// Drain the catch-up through its caught-up ack (the one `total_size == 0`
/// header), leaving the subscription in steady state.
async fn drain_catch_up(stream: &mut SubscribeStream) {
    loop {
        let msg = next_item(stream, "a catch-up message")
            .await
            .expect("the stream ended during catch-up")
            .unwrap_or_else(|s| panic!("catch-up failed: {s:?}"));
        if let Some(subscribe_read_layer_response::ResponseType::Header(h)) = msg.response_type {
            if h.total_size == 0 {
                return;
            }
        }
    }
}

/// Drain to the terminal status under one deadline; `None` for a clean EOS.
async fn drain_to_terminal(stream: &mut SubscribeStream) -> Option<Status> {
    tokio::time::timeout(SETTLE, async {
        while let Some(item) = stream.next().await {
            if let Err(status) = item {
                return Some(status);
            }
        }
        None
    })
    .await
    .unwrap_or_else(|_| panic!("the subscription did not end within {SETTLE:?}"))
}

fn stream_exits(metrics: &Metrics, reason: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_subscriber_stream_exits_total",
        &[("reason", reason)],
    )
    .unwrap_or_else(|| panic!("stream-exit series for reason={reason} is absent")) as u64
}

fn total_stream_exits(metrics: &Metrics) -> u64 {
    STREAM_EXIT_REASONS
        .iter()
        .map(|reason| stream_exits(metrics, reason))
        .sum()
}

/// Wait until the subscription task has finished its tail. The tail logs its
/// line *before* it increments the stream-exit counter, so once the counter
/// shows the exit the line (if any) is already in the capture.
async fn await_stream_exit(metrics: &Metrics, reason: &str) {
    let deadline = tokio::time::Instant::now() + SETTLE;
    while stream_exits(metrics, reason) == 0 {
        assert!(
            tokio::time::Instant::now() < deadline,
            "no `{reason}` stream exit within {SETTLE:?}; exits so far: {}",
            total_stream_exits(metrics)
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// Wait for every spawned JoinSession task — including its presence cleanup —
/// to finish.
async fn await_join_tasks(svc: &LiveEditService) {
    svc.presence_cleanup_tracker.close();
    tokio::time::timeout(SETTLE, svc.presence_cleanup_tracker.wait())
        .await
        .unwrap_or_else(|_| panic!("the JoinSession task did not finish within {SETTLE:?}"));
}

/// The single `RPC failed` line, asserting there is no other completion line.
fn only_failure(captured: &Captured) -> &Value {
    let event = captured.only(RPC_FAILURE_MSG);
    assert_eq!(
        captured.completion_lines().len(),
        1,
        "one failure, one completion line; full capture:\n{}",
        captured.text
    );
    event
}

// =============================================================================
// (a) SubscribeReadLayer setup: logged by the handler wrapper
// =============================================================================

/// An invalid `session_id` fails at `stage=header` with no `session_id` and no
/// `layer_name` on the line, and the rejected value appears nowhere.
///
/// `client_id` IS present: it is validated first, so it has been recorded by
/// the time the session validator refuses (populate-after-validate, one field
/// at a time).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn subscribe_invalid_session_id_logs_info_at_header_without_the_session_id() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let (result, captured) = capture(handle_subscribe_read_layer(
        &svc,
        subscribe_request(CONTROL_CHAR_SESSION_ID, "some-layer", 0, 0),
    ))
    .await;

    let status = expect_err_status(result, "a control-character session_id must be refused");
    assert_eq!(status.code(), Code::InvalidArgument);

    let event = only_failure(&captured);
    assert_common_fields(
        event,
        "INFO",
        "SubscribeReadLayer",
        "header",
        "INVALID_ARGUMENT",
    );
    assert_field(event, "client_id", SUBSCRIBER.into());
    assert_field_absent(event, "session_id", "its validator rejected it");
    assert_field_absent(event, "layer_name", "never reached its validator");
    assert_field_absent(event, "regime", "no stream ever opened");
    captured.assert_absent_everywhere(CONTROL_CHAR_SESSION_ID);
    captured.assert_absent_everywhere("\u{1b}[");
    captured.assert_absent_everywhere(BEARER);
}

/// An invalid `layer_name` is refused at setup — `INVALID_ARGUMENT` returned
/// from the handler (Trailers-Only) — and the rejected name never reaches the
/// log.
///
/// The regression test for validating `layer_name` at setup: before it, the
/// handler checked only for emptiness, so this name opened a stream and ended
/// mid-stream `NOT_FOUND` whose message quoted the name back — which the
/// teardown line would then have written, unvalidated, to the log.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn subscribe_invalid_layer_name_is_refused_at_setup_and_never_logged() {
    const BAD_LAYER: &str = "bad\u{1b}[2Jlayer";

    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let (session_id, _object_id) =
        create_session_and_layer(&tv, &svc, "stream-log-bad-layer", "good-layer").await;

    let (result, captured) = capture(handle_subscribe_read_layer(
        &svc,
        subscribe_request(&session_id, BAD_LAYER, 0, 0),
    ))
    .await;

    let status = expect_err_status(
        result,
        "an invalid layer_name opened a stream; it must be refused before the response head",
    );
    assert_eq!(status.code(), Code::InvalidArgument);
    assert_eq!(status.message(), "layer name contains control characters");

    let event = only_failure(&captured);
    assert_common_fields(
        event,
        "INFO",
        "SubscribeReadLayer",
        "header",
        "INVALID_ARGUMENT",
    );
    assert_field(event, "session_id", session_id.as_str().into());
    assert_field_absent(event, "layer_name", "its validator rejected it");
    captured.assert_absent_everywhere(BAD_LAYER);
    captured.assert_absent_everywhere("\u{1b}[2J");
}

/// A missing session logs INFO `NOT_FOUND` at `stage=auth`, with the request
/// identity and a non-zero `have_object_id` as `object_id`, and no
/// `base_path` (the lookup that would produce it is the one that failed).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn subscribe_missing_session_logs_info_not_found_at_auth() {
    const MISSING: &str = "stream-log-no-such-session";
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let (result, captured) = capture(handle_subscribe_read_layer(
        &svc,
        subscribe_request(MISSING, "some-layer", 7, 3),
    ))
    .await;

    let status = expect_err_status(result, "an unknown session must be refused at setup");
    assert_eq!(status.code(), Code::NotFound);

    let event = only_failure(&captured);
    assert_common_fields(event, "INFO", "SubscribeReadLayer", "auth", "NOT_FOUND");
    assert_field(event, "session_id", MISSING.into());
    assert_field(event, "layer_name", "some-layer".into());
    assert_field(event, "client_id", SUBSCRIBER.into());
    assert_field(event, "object_id", 7.into());
    assert_field_absent(event, "base_path", "the lookup that produces it failed");
    assert_field_absent(
        event,
        "current_object_id",
        "setup resolves no mapping, so no object was proven live",
    );
    captured.assert_absent_everywhere(BEARER);
}

/// A brand-new subscribe (`have_object_id = 0`) records no `object_id`: zero
/// is the protocol's "no object", not an id.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn subscribe_brand_new_setup_failure_records_no_object_id() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let (result, captured) = capture(handle_subscribe_read_layer(
        &svc,
        subscribe_request("stream-log-missing-2", "some-layer", 0, 0),
    ))
    .await;
    assert_eq!(
        expect_err_status(result, "must be refused").code(),
        Code::NotFound
    );
    assert_field_absent(only_failure(&captured), "object_id", "zero means no object");
}

/// A permission denial logs WARN `PERMISSION_DENIED` at `stage=auth` with the
/// session's `base_path`, and no bearer material anywhere.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn subscribe_permission_denied_logs_warn_at_auth_with_the_base_path() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(DenyAllPermissions));
    let session_id = create_session(&tv, &svc, "stream-log-denied").await;

    let (result, captured) = capture(handle_subscribe_read_layer(
        &svc,
        subscribe_request(&session_id, "some-layer", 0, 0),
    ))
    .await;

    let status = expect_err_status(result, "a denied caller must be refused at setup");
    assert_eq!(status.code(), Code::PermissionDenied);
    assert_eq!(status.message(), "");

    let event = only_failure(&captured);
    assert_common_fields(
        event,
        "WARN",
        "SubscribeReadLayer",
        "auth",
        "PERMISSION_DENIED",
    );
    assert_field(event, "session_id", session_id.as_str().into());
    assert_field(event, "layer_name", "some-layer".into());
    assert_field(event, "base_path", "/stage".into());
    assert_field(event, "status_message", "".into());
    captured.assert_absent_everywhere(BEARER);
}

// =============================================================================
// (b) SubscribeReadLayer teardown: logged by the subscription task's tail
// =============================================================================
//
// Current-thread runtime + `ThreadCapture` throughout — see the module doc.

/// A layer name that resolves to no mapping ends the stream `NOT_FOUND`, and
/// the tail logs one INFO line at `stage=stream` with no `regime` and no
/// object: the catch-up failed before any branch ran.
#[tokio::test]
async fn subscribe_unknown_layer_logs_one_info_not_found_at_stream_with_no_regime() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let session_id = create_session(&tv, &svc, "stream-log-unknown-layer").await;
    let capture = ThreadCapture::start();

    let mut stream: SubscribeStream =
        handle_subscribe_read_layer(&svc, subscribe_request(&session_id, "no-such-layer", 0, 0))
            .await
            .unwrap_or_else(|s| panic!("setup must succeed; the lookup runs in the task: {s:?}"))
            .into_inner();
    let status = drain_to_terminal(&mut stream)
        .await
        .expect("an unknown layer must end with a status, not a clean EOS");
    assert_eq!(status.code(), Code::NotFound);
    await_stream_exit(&svc.metrics, STREAM_EXIT_ERROR).await;

    let captured = capture.snapshot();
    let event = only_failure(&captured);
    assert_common_fields(event, "INFO", "SubscribeReadLayer", "stream", "NOT_FOUND");
    assert_field(event, "session_id", session_id.as_str().into());
    assert_field(event, "layer_name", "no-such-layer".into());
    assert_field(event, "client_id", SUBSCRIBER.into());
    assert_field(event, "base_path", "/stage".into());
    assert_field(event, "reason", STREAM_EXIT_ERROR.into());
    assert_field(event, "caught_up", false.into());
    assert_field(event, "status_message", status.message().into());
    assert_field_absent(event, "regime", "no catch-up branch was entered");
    assert_field_absent(event, "current_object_id", "no mapping resolved");
    captured.assert_absent_everywhere(BEARER);

    drop(capture);
    purge_keys(&tv).await;
}

/// A layer deleted under an established subscriber ends `NOT_FOUND` with
/// `caught_up=true`; the steady-state updates before it log nothing; and the
/// subscription produces exactly one completion line in total.
///
/// Level INFO even though the object was resolved: only the commit path
/// (`stage=storage`) escalates a resolved `NOT_FOUND` to keyspace damage, and a
/// deleted layer under a subscriber is the ordinary lifecycle.
#[tokio::test]
async fn subscribe_layer_deleted_mid_stream_logs_one_not_found_with_caught_up_and_nothing_before() {
    const LAYER: &str = "stream-log-deleted-layer";
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, "stream-log-deleted", LAYER).await;
    let capture = ThreadCapture::start();

    let mut stream: SubscribeStream =
        handle_subscribe_read_layer(&svc, subscribe_request(&session_id, LAYER, 0, 0))
            .await
            .unwrap_or_else(|s| panic!("setup must succeed: {s:?}"))
            .into_inner();
    drain_catch_up(&mut stream).await;

    // Steady-state traffic: relayed, and silent.
    for n in 1..=2 {
        commit(&svc, &session_id, LAYER, object_id, n).await;
        let _ = next_item(&mut stream, "a relayed update").await;
    }
    capture
        .snapshot()
        .assert_no_completion_line("steady-state updates on a healthy subscription");

    handle_delete_layer(&svc, delete_layer_request(&session_id, LAYER))
        .await
        .expect("DeleteLayer must succeed");
    let status = drain_to_terminal(&mut stream)
        .await
        .expect("a deleted layer must end the subscription with a status");
    assert_eq!(status.code(), Code::NotFound);
    await_stream_exit(&svc.metrics, STREAM_EXIT_ERROR).await;

    let captured = capture.snapshot();
    let event = only_failure(&captured);
    assert_common_fields(event, "INFO", "SubscribeReadLayer", "stream", "NOT_FOUND");
    assert_field(event, "caught_up", true.into());
    assert_field(event, "current_object_id", object_id.into());
    assert_field(event, "regime", "brand_new".into());
    assert_field(event, "reason", STREAM_EXIT_ERROR.into());
    assert_field(event, "layer_name", LAYER.into());

    drop(capture);
    purge_keys(&tv).await;
}

/// A registration the stream multiplexer refuses because it has shut down ends
/// the subscription as the shutdown it is: `UNAVAILABLE "server shutting
/// down"` — the status every other shutdown ending carries, which the client
/// retries — under `reason="shutdown"`, with its one completion line at INFO,
/// as every shutdown ending is (a routine event on every rolling restart). It
/// is never a clean end, never `INTERNAL`, and never the `error` bucket.
///
/// The multiplexer's shutdown is the only cause here: the service shutdown
/// token is left unfired, so `open_consumer`'s own shutdown arm cannot be what
/// ended it, and the ending comes from mapping `ConsumeError::ShuttingDown`.
#[tokio::test]
async fn subscribe_whose_stream_registration_is_refused_at_shutdown_logs_one_info_unavailable_under_shutdown(
) {
    const LAYER: &str = "stream-log-refused-registration";
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let (session_id, _object_id) =
        create_session_and_layer(&tv, &svc, "stream-log-refused", LAYER).await;
    tv.client()
        .stream_mux
        .shutdown(Duration::from_secs(1))
        .await;
    let capture = ThreadCapture::start();

    let mut stream: SubscribeStream =
        handle_subscribe_read_layer(&svc, subscribe_request(&session_id, LAYER, 0, 0))
            .await
            .unwrap_or_else(|s| panic!("setup must succeed: {s:?}"))
            .into_inner();
    let status = drain_to_terminal(&mut stream)
        .await
        .expect("a refused registration must end the subscription with a status");
    assert!(
        !svc.shutdown.is_cancelled(),
        "precondition: only the stream multiplexer shut down, not the service"
    );
    assert_eq!(status.code(), Code::Unavailable, "got {status:?}");
    assert_eq!(status.message(), "server shutting down");
    await_stream_exit(&svc.metrics, STREAM_EXIT_SHUTDOWN).await;
    assert_eq!(
        stream_exits(&svc.metrics, STREAM_EXIT_ERROR),
        0,
        "a shutdown refusal is not the `error` bucket"
    );

    let captured = capture.snapshot();
    let event = only_failure(&captured);
    assert_common_fields(event, "INFO", "SubscribeReadLayer", "stream", "UNAVAILABLE");
    assert_field(event, "reason", STREAM_EXIT_SHUTDOWN.into());
    assert_field(event, "status_message", "server shutting down".into());
    assert_field(event, "caught_up", false.into());

    drop(capture);
    purge_keys(&tv).await;
}

/// A catch-up refused its slot — every concurrent-catch-up slot stayed held
/// for the whole capacity budget — ends `UNAVAILABLE` naming
/// `LIVE_EDIT_SUBSCRIBER_MAX_CONCURRENT_CATCHUPS`, under
/// `reason="catchup_capacity"`, before any content or ack, with one WARN
/// completion line: a replica turning subscribers away for capacity is an
/// operator signal, unlike a shutdown.
///
/// `TestValkey`'s client caps catch-ups at 64 with a 2 s budget, so the test
/// holds all 64 slots itself and the refusal costs that 2 s.
#[tokio::test]
async fn subscribe_refused_a_catch_up_slot_logs_one_warn_unavailable_under_catchup_capacity() {
    use live_edit::observability::metrics::STREAM_EXIT_CATCHUP_CAPACITY;
    const LAYER: &str = "stream-log-catchup-capacity";
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let (session_id, _object_id) =
        create_session_and_layer(&tv, &svc, "stream-log-catchup-capacity", LAYER).await;
    let gate = &svc.valkey.catchup_gate;
    let limit = gate.limit().expect("TestValkey caps concurrent catch-ups");
    let far = tokio::time::Instant::now() + Duration::from_secs(3_600);
    let mut held = Vec::with_capacity(limit);
    for _ in 0..limit {
        held.push(gate.acquire(far).await.expect("a free slot"));
    }
    assert_eq!(gate.available(), 0, "precondition: every slot is held");
    let capture = ThreadCapture::start();

    let mut stream: SubscribeStream =
        handle_subscribe_read_layer(&svc, subscribe_request(&session_id, LAYER, 0, 0))
            .await
            .unwrap_or_else(|s| panic!("setup must succeed: {s:?}"))
            .into_inner();
    let first = next_item(&mut stream, "the capacity refusal")
        .await
        .expect("a refused catch-up must end with a status, not a clean close");
    let status = match first {
        Err(status) => status,
        Ok(msg) => panic!("the refusal must come before any content or ack; got {msg:?}"),
    };
    assert_eq!(status.code(), Code::Unavailable, "got {status:?}");
    assert_eq!(
        status.message(),
        "this replica is at its concurrent catch-up limit \
         (LIVE_EDIT_SUBSCRIBER_MAX_CONCURRENT_CATCHUPS); retry"
    );
    await_stream_exit(&svc.metrics, STREAM_EXIT_CATCHUP_CAPACITY).await;

    let captured = capture.snapshot();
    let event = only_failure(&captured);
    assert_common_fields(event, "WARN", "SubscribeReadLayer", "stream", "UNAVAILABLE");
    assert_field(event, "reason", STREAM_EXIT_CATCHUP_CAPACITY.into());
    assert_field(event, "status_message", status.message().into());
    assert_field(event, "caught_up", false.into());

    drop(held);
    drop(capture);
    purge_keys(&tv).await;
}

/// The sequence number the layer's object header is at now — what a client
/// that has seen every commit would reconnect with.
async fn current_seq(svc: &LiveEditService, object_id: u64) -> u64 {
    svc.layer_store
        .read_object_header(object_id)
        .await
        .expect("the object header must be readable")
        .sequence_num
}

/// Reconnect at `(object_id, have_version)`, drain the catch-up, delete the
/// layer, and return the one completion line the subscription's tail logged —
/// after asserting it is the INFO `NOT_FOUND` teardown of an established
/// subscription on that object.
///
/// Shared by the three non-brand-new regime tests below. Every other teardown
/// test subscribes brand-new, and `catch_up_brand_new` records its own regime
/// and object; these are the only cases that pin the up-to-date, in-window and
/// out-of-window set points and the object id recorded as soon as the mapping
/// resolves.
async fn teardown_line_after_reconnect(
    svc: &LiveEditService,
    session_id: &str,
    layer: &str,
    object_id: u64,
    have_version: u64,
) -> Value {
    let capture = ThreadCapture::start();
    let mut stream: SubscribeStream = handle_subscribe_read_layer(
        svc,
        subscribe_request(session_id, layer, object_id, have_version),
    )
    .await
    .unwrap_or_else(|s| panic!("setup must succeed: {s:?}"))
    .into_inner();
    drain_catch_up(&mut stream).await;

    handle_delete_layer(svc, delete_layer_request(session_id, layer))
        .await
        .expect("DeleteLayer must succeed");
    let status = drain_to_terminal(&mut stream)
        .await
        .expect("a deleted layer must end the subscription with a status");
    assert_eq!(status.code(), Code::NotFound);
    await_stream_exit(&svc.metrics, STREAM_EXIT_ERROR).await;

    let captured = capture.snapshot();
    let event = only_failure(&captured).clone();
    assert_common_fields(&event, "INFO", "SubscribeReadLayer", "stream", "NOT_FOUND");
    assert_field(&event, "caught_up", true.into());
    assert_field(&event, "object_id", object_id.into());
    assert_field(&event, "current_object_id", object_id.into());
    drop(capture);
    event
}

/// A subscriber that reconnects already up to date, then loses its layer,
/// logs `regime=up_to_date` and the object it was serving.
///
/// The up-to-date branch sets no object id of its own: the one on this line
/// comes from the set point right after the mapping resolves, so this test is
/// also what pins that one.
#[tokio::test]
async fn subscribe_teardown_after_up_to_date_catch_up_logs_the_up_to_date_regime() {
    const LAYER: &str = "stream-log-up-to-date";
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, "stream-log-up-to-date", LAYER).await;
    let seq = current_seq(&svc, object_id).await;

    let event = teardown_line_after_reconnect(&svc, &session_id, LAYER, object_id, seq).await;
    assert_field(&event, "regime", "up_to_date".into());

    purge_keys(&tv).await;
}

/// A subscriber that reconnects a few commits behind, inside the stream's
/// retention window, then loses its layer, logs `regime=in_window`.
#[tokio::test]
async fn subscribe_teardown_after_in_window_catch_up_logs_the_in_window_regime() {
    const LAYER: &str = "stream-log-in-window";
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, "stream-log-in-window", LAYER).await;
    let seen = current_seq(&svc, object_id).await;
    for n in 1..=2 {
        commit(&svc, &session_id, LAYER, object_id, n).await;
    }

    let event = teardown_line_after_reconnect(&svc, &session_id, LAYER, object_id, seen).await;
    assert_field(&event, "regime", "in_window".into());

    purge_keys(&tv).await;
}

/// A subscriber whose next entry has been trimmed off the stream reconnects
/// through the out-of-window catch-up; when it then loses its layer, the line
/// says `regime=out_of_window`.
#[tokio::test]
async fn subscribe_teardown_after_out_of_window_catch_up_logs_the_out_of_window_regime() {
    use fred::interfaces::StreamsInterface;
    use fred::types::streams::XCap;
    const LAYER: &str = "stream-log-out-of-window";

    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, "stream-log-out-of-window", LAYER).await;
    let seen = current_seq(&svc, object_id).await;
    for n in 1..=3 {
        commit(&svc, &session_id, LAYER, object_id, n).await;
    }
    // Trim away every entry the reconnector would replay in-window, so the
    // first retained entry is not `{seen + 1}-0` and the handler falls through
    // to the out-of-window regime.
    let latest = current_seq(&svc, object_id).await;
    let min_id = format!("{latest}-0");
    let cap = XCap::try_from(("MINID", "=", min_id.as_str())).expect("build MINID xcap");
    let _: () = tv
        .client()
        .pool
        .xtrim(tv.key_schema().layer_stream(object_id), cap)
        .await
        .expect("XTRIM to force out-of-window must succeed");

    let event = teardown_line_after_reconnect(&svc, &session_id, LAYER, object_id, seen).await;
    assert_field(&event, "regime", "out_of_window".into());

    purge_keys(&tv).await;
}

/// A permanent catch-up failure — a retyped `obj:hdr` — produces exactly ONE
/// line: ERROR `internal error` carrying the un-redacted detail AND the
/// subscription's identity and regime; no `RPC failed`; and no context-free
/// fallback when the client drops the status.
///
/// Before request context reached `INTERNAL`, this ERROR — the line operators
/// page on — said only which Valkey command failed, not for which session,
/// layer or subscriber.
#[tokio::test]
async fn subscribe_permanent_catch_up_failure_logs_one_error_with_context_and_regime() {
    use fred::interfaces::KeysInterface;
    const LAYER: &str = "stream-log-wrongtype-layer";

    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, "stream-log-wrongtype", LAYER).await;
    let header_key = tv.key_schema().obj_header(&object_id.to_string());
    let _: () = tv
        .client()
        .pool
        .set(&header_key, "not-a-hash", None, None, false)
        .await
        .expect("retyping obj:hdr must succeed");
    let capture = ThreadCapture::start();

    let mut stream: SubscribeStream =
        handle_subscribe_read_layer(&svc, subscribe_request(&session_id, LAYER, 0, 0))
            .await
            .unwrap_or_else(|s| panic!("setup must succeed: {s:?}"))
            .into_inner();
    let status = drain_to_terminal(&mut stream)
        .await
        .expect("a permanent catch-up failure must end with a status");
    assert_eq!(
        status.code(),
        Code::Internal,
        "precondition: the fixture must produce INTERNAL; got {status:?}"
    );
    assert_eq!(
        status.message(),
        INTERNAL_ERROR_MSG,
        "the wire stays redacted"
    );
    // The client's copy of the status goes away here, as it would when tonic
    // finishes writing the trailer. A fault the tail failed to mark reported
    // would log its context-free fallback now.
    drop(status);
    drop(stream);
    await_stream_exit(&svc.metrics, STREAM_EXIT_ERROR).await;

    let captured = capture.snapshot();
    assert!(
        captured.with_message(RPC_FAILURE_MSG).is_empty(),
        "a server fault logs `internal error`, not `RPC failed`; full capture:\n{}",
        captured.text
    );
    let event = captured.only(INTERNAL_ERROR_MSG);
    assert_common_fields(event, "ERROR", "SubscribeReadLayer", "stream", "INTERNAL");
    assert_field(event, "session_id", session_id.as_str().into());
    assert_field(event, "layer_name", LAYER.into());
    assert_field(event, "client_id", SUBSCRIBER.into());
    assert_field(event, "regime", "brand_new".into());
    assert_field(event, "current_object_id", object_id.into());
    assert_field(event, "caught_up", false.into());
    let detail = event["fields"]["error"].as_str().unwrap_or_default();
    assert!(
        detail.contains("WRONGTYPE"),
        "the ERROR line must carry the un-redacted detail the wire lost; got {event}"
    );

    drop(capture);
    purge_keys(&tv).await;
}

/// A client that leaves before its first caught-up ack was enqueued is logged
/// — `CANCELLED` at INFO, `caught_up=false`, `reason=client_disconnect`.
///
/// This is the load-test shape "never got a first response". The stream is
/// dropped before the handler's spawned task first runs: on this
/// current-thread runtime the task cannot be polled until the test yields, so
/// its very first send finds the receiver gone.
#[tokio::test]
async fn subscribe_client_leaving_before_caught_up_logs_one_info_cancelled() {
    const LAYER: &str = "stream-log-early-leave";
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let (session_id, _object_id) =
        create_session_and_layer(&tv, &svc, "stream-log-early-leave", LAYER).await;
    let capture = ThreadCapture::start();

    let stream = handle_subscribe_read_layer(&svc, subscribe_request(&session_id, LAYER, 0, 0))
        .await
        .unwrap_or_else(|s| panic!("setup must succeed: {s:?}"));
    drop(stream);
    await_stream_exit(&svc.metrics, STREAM_EXIT_CLIENT_DISCONNECT).await;

    let captured = capture.snapshot();
    let event = only_failure(&captured);
    assert_common_fields(event, "INFO", "SubscribeReadLayer", "stream", "CANCELLED");
    assert_field(event, "caught_up", false.into());
    assert_field(event, "reason", STREAM_EXIT_CLIENT_DISCONNECT.into());
    assert_field(event, "session_id", session_id.as_str().into());

    drop(capture);
    purge_keys(&tv).await;
}

/// A client that leaves an established stream — after its caught-up ack — is
/// not a failure, and logs nothing.
#[tokio::test]
async fn subscribe_client_leaving_after_caught_up_logs_nothing() {
    const LAYER: &str = "stream-log-late-leave";
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let (session_id, _object_id) =
        create_session_and_layer(&tv, &svc, "stream-log-late-leave", LAYER).await;
    let capture = ThreadCapture::start();

    let mut stream: SubscribeStream =
        handle_subscribe_read_layer(&svc, subscribe_request(&session_id, LAYER, 0, 0))
            .await
            .unwrap_or_else(|s| panic!("setup must succeed: {s:?}"))
            .into_inner();
    drain_catch_up(&mut stream).await;
    drop(stream);
    await_stream_exit(&svc.metrics, STREAM_EXIT_CLIENT_DISCONNECT).await;

    capture
        .snapshot()
        .assert_no_completion_line("a client leaving an established subscription");

    drop(capture);
    purge_keys(&tv).await;
}

/// An idle steady-state subscriber whose session token fires ends `NOT_FOUND
/// "session was deleted"` and logs one INFO line at `stage=stream` with
/// `reason=session_cancel` and `caught_up=true` — an expected ending, so INFO,
/// but no longer a silent one.
///
/// The token is cancelled directly, the deletion path whose message and reason
/// are deterministic (no layer `deleted` sentinel can race it).
#[tokio::test]
async fn subscribe_idle_session_cancel_ends_not_found_and_logs_one_info_with_its_reason() {
    const LAYER: &str = "stream-log-idle-cancel";
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let (session_id, _object_id) =
        create_session_and_layer(&tv, &svc, "stream-log-idle-cancel", LAYER).await;
    let capture = ThreadCapture::start();

    let mut stream: SubscribeStream =
        handle_subscribe_read_layer(&svc, subscribe_request(&session_id, LAYER, 0, 0))
            .await
            .unwrap_or_else(|s| panic!("setup must succeed: {s:?}"))
            .into_inner();
    drain_catch_up(&mut stream).await;
    svc.session_tokens
        .get(&session_id)
        .expect("the handler registered a session token")
        .cancel();
    let status = drain_to_terminal(&mut stream)
        .await
        .expect("an idle subscriber whose session is deleted must end with a status");
    assert_eq!(status.code(), Code::NotFound, "got {status:?}");
    await_stream_exit(&svc.metrics, STREAM_EXIT_SESSION_CANCEL).await;

    let captured = capture.snapshot();
    let event = only_failure(&captured);
    assert_common_fields(event, "INFO", "SubscribeReadLayer", "stream", "NOT_FOUND");
    assert_field(event, "reason", STREAM_EXIT_SESSION_CANCEL.into());
    assert_field(event, "caught_up", true.into());
    assert_field(event, "status_message", "session was deleted".into());

    drop(capture);
    purge_keys(&tv).await;
}

/// Seam- and injector-driven teardowns. Compiled out without
/// `test-fault-injection` (module doc).
#[cfg(feature = "test-fault-injection")]
mod seam_driven {
    use super::*;

    use live_edit::observability::metrics::STREAM_EXIT_SLOW_READER_DEADLINE;

    /// A transient catch-up read failure logs WARN `UNAVAILABLE` at
    /// `stage=stream`, with the failing read named in `status_message`, the
    /// regime, the object, and `caught_up=false`.
    ///
    /// Every transient subscribe failure is `UNAVAILABLE` on the wire, so the
    /// status message on this line is the only per-event record of which read
    /// failed — the reason the line carries it at all.
    #[tokio::test]
    async fn subscribe_transient_catch_up_failure_logs_warn_naming_the_failed_read() {
        use fred::error::{Error as FredError, ErrorKind as FredErrorKind};
        use live_edit::storage::diff_read::ObjectSnapshotReadFaultInjector;
        const LAYER: &str = "stream-log-transient-layer";

        let tv = TestValkey::setup().await;
        let svc = build_service(&tv, Arc::new(AllowAllPermissions));
        let (session_id, object_id) =
            create_session_and_layer(&tv, &svc, "stream-log-transient", LAYER).await;
        // The catch-up's atomic snapshot read takes the canned error in place
        // of its `EVALSHA` result. No checkpoint loop runs here, so nothing
        // else on the shared client can consume it.
        let injector = Arc::new(ObjectSnapshotReadFaultInjector::new());
        injector.push(FredError::new(
            FredErrorKind::Timeout,
            "injected catch-up atomic snapshot read timeout",
        ));
        tv.client()
            .install_object_snapshot_read_fault_injector_for_test(injector.clone());
        let capture = ThreadCapture::start();

        let mut stream: SubscribeStream =
            handle_subscribe_read_layer(&svc, subscribe_request(&session_id, LAYER, 0, 0))
                .await
                .unwrap_or_else(|s| panic!("setup must succeed: {s:?}"))
                .into_inner();
        let status = drain_to_terminal(&mut stream)
            .await
            .expect("a transient catch-up failure must end with a status");
        assert_eq!(status.code(), Code::Unavailable);
        assert!(
            status.message().contains("EVALSHA object_snapshot_read"),
            "precondition: the failure came from the snapshot read; got {}",
            status.message()
        );
        assert_eq!(injector.remaining(), 0, "precondition: the injector fired");
        await_stream_exit(&svc.metrics, STREAM_EXIT_ERROR).await;

        let captured = capture.snapshot();
        let event = only_failure(&captured);
        assert_common_fields(event, "WARN", "SubscribeReadLayer", "stream", "UNAVAILABLE");
        assert_field(event, "status_message", status.message().into());
        assert_field(event, "regime", "brand_new".into());
        assert_field(event, "current_object_id", object_id.into());
        assert_field(event, "caught_up", false.into());
        assert_field(event, "reason", STREAM_EXIT_ERROR.into());
        assert_field(event, "session_id", session_id.as_str().into());
        assert_field(event, "client_id", SUBSCRIBER.into());

        drop(capture);
        purge_keys(&tv).await;
    }

    /// An out-of-window catch-up whose state read answers NotFound — the
    /// object was overwritten and its keys cleaned up under the reconnector —
    /// re-bootstraps onto the new object; a teardown after that says
    /// `regime=brand_new` and names the NEW object as `current_object_id`, not
    /// the out-of-window regime or the object it reconnected to (b.1pa, kept
    /// true through b.21r's shared recovery helper).
    ///
    /// The out-of-window seam parks the reconnector after its header read and
    /// before its diff read, which is where the overwrite + cleanup lands.
    #[tokio::test]
    async fn subscribe_teardown_after_an_out_of_window_notfound_rebootstrap_logs_brand_new() {
        use fred::interfaces::StreamsInterface;
        use fred::types::streams::XCap;
        use live_edit::grpc::layer::handle_create_layer;

        use super::super::helpers::{make_create_data, make_create_header, streaming_create};
        const LAYER: &str = "stream-log-oow-rebootstrap";

        let tv = TestValkey::setup().await;
        let mut svc = build_service(&tv, Arc::new(AllowAllPermissions));
        let seam =
            Arc::new(live_edit::grpc::subscribe::CatchUpOutOfWindowSeam::new_before_compute_diff());
        svc.catch_up_out_of_window_seam_for_test = Some(Arc::clone(&seam));
        let (session_id, old_object_id) =
            create_session_and_layer(&tv, &svc, "stream-log-oow-rebootstrap", LAYER).await;
        let seen = current_seq(&svc, old_object_id).await;
        for n in 1..=3 {
            commit(&svc, &session_id, LAYER, old_object_id, n).await;
        }
        let latest = current_seq(&svc, old_object_id).await;
        let min_id = format!("{latest}-0");
        let cap = XCap::try_from(("MINID", "=", min_id.as_str())).expect("build MINID xcap");
        let _: () = tv
            .client()
            .pool
            .xtrim(tv.key_schema().layer_stream(old_object_id), cap)
            .await
            .expect("XTRIM to force out-of-window must succeed");
        let capture = ThreadCapture::start();

        let mut stream: SubscribeStream = handle_subscribe_read_layer(
            &svc,
            subscribe_request(&session_id, LAYER, old_object_id, seen),
        )
        .await
        .unwrap_or_else(|s| panic!("setup must succeed: {s:?}"))
        .into_inner();
        tokio::time::timeout(SETTLE, seam.wait_until_parked_pre_compute_diff())
            .await
            .expect("the reconnector must reach the out-of-window diff-read seam");

        let new_object_id = handle_create_layer(
            &svc,
            make_request(streaming_create(vec![
                make_create_header(&session_id, LAYER),
                make_create_data(crate::common::owned_delta::delta_to_ompf_bytes(
                    &crate::common::owned_delta::empty_delta(),
                    &[],
                )),
            ])),
        )
        .await
        .expect("the CreateLayer overwrite must succeed")
        .into_inner()
        .object_id;
        assert_ne!(new_object_id, old_object_id);
        svc.layer_store.wait_for_cleanup(SETTLE).await;
        tokio::time::timeout(SETTLE, seam.arrive_at_pre_compute_diff())
            .await
            .expect("releasing the parked reconnector must not hang");

        // The re-bootstrap's full state and caught-up ack on the new object.
        drain_catch_up(&mut stream).await;
        handle_delete_layer(&svc, delete_layer_request(&session_id, LAYER))
            .await
            .expect("DeleteLayer must succeed");
        let status = drain_to_terminal(&mut stream)
            .await
            .expect("a deleted layer must end the subscription with a status");
        assert_eq!(status.code(), Code::NotFound);
        await_stream_exit(&svc.metrics, STREAM_EXIT_ERROR).await;

        let captured = capture.snapshot();
        let event = only_failure(&captured);
        assert_common_fields(event, "INFO", "SubscribeReadLayer", "stream", "NOT_FOUND");
        assert_field(event, "regime", "brand_new".into());
        assert_field(event, "object_id", old_object_id.into());
        assert_field(event, "current_object_id", new_object_id.into());
        assert_field(event, "caught_up", true.into());

        drop(capture);
        purge_keys(&tv).await;
    }

    /// A slow reader torn down on the outer send deadline logs WARN
    /// `UNAVAILABLE` with `reason=slow_reader_deadline` and `caught_up=true`.
    #[tokio::test]
    async fn subscribe_slow_reader_teardown_logs_warn_with_its_reason() {
        const LAYER: &str = "stream-log-slow-layer";
        /// Commits written while the client reads nothing: each relays two
        /// messages, so this is comfortably past the 32 delivery slots.
        const STALL_COMMITS: u64 = 40;

        let tv = TestValkey::setup().await;
        let mut svc = build_service(&tv, Arc::new(AllowAllPermissions));
        svc.outer_send_deadline_for_test = Some(Duration::from_millis(300));
        let (session_id, object_id) =
            create_session_and_layer(&tv, &svc, "stream-log-slow", LAYER).await;
        let capture = ThreadCapture::start();

        let mut stream: SubscribeStream =
            handle_subscribe_read_layer(&svc, subscribe_request(&session_id, LAYER, 0, 0))
                .await
                .unwrap_or_else(|s| panic!("setup must succeed: {s:?}"))
                .into_inner();
        drain_catch_up(&mut stream).await;
        for n in 1..=STALL_COMMITS {
            commit(&svc, &session_id, LAYER, object_id, n).await;
        }
        await_stream_exit(&svc.metrics, STREAM_EXIT_SLOW_READER_DEADLINE).await;

        let captured = capture.snapshot();
        let event = only_failure(&captured);
        assert_common_fields(event, "WARN", "SubscribeReadLayer", "stream", "UNAVAILABLE");
        assert_field(event, "reason", STREAM_EXIT_SLOW_READER_DEADLINE.into());
        assert_field(event, "caught_up", true.into());
        assert_field(
            event,
            "status_message",
            "subscriber did not read its stream within 300ms; tearing down".into(),
        );

        drop(stream);
        drop(capture);
        purge_keys(&tv).await;
    }

    /// A permanent consumer failure mid-stream logs WARN `UNAVAILABLE` with
    /// `reason=error` and `caught_up=true`.
    ///
    /// The fault is queued only after the caught-up ack, so it lands on a
    /// steady-state read. Queued before the subscribe, it would fail the
    /// reader's first XREAD — issued as soon as the catch-up registers its
    /// stream — and end the catch-up before any content, which is a different
    /// ending (`caught_up=false`).
    #[tokio::test]
    async fn subscribe_consumer_fault_logs_warn_with_its_reason() {
        use crate::storage::mux_test_helpers::push_faults;
        const LAYER: &str = "stream-log-consumer-fault";

        let tv = TestValkey::setup().await;
        let svc = build_service(&tv, Arc::new(AllowAllPermissions));
        let (session_id, object_id) =
            create_session_and_layer(&tv, &svc, "stream-log-consumer", LAYER).await;
        let capture = ThreadCapture::start();

        let mut stream: SubscribeStream =
            handle_subscribe_read_layer(&svc, subscribe_request(&session_id, LAYER, 0, 0))
                .await
                .unwrap_or_else(|s| panic!("setup must succeed: {s:?}"))
                .into_inner();
        drain_catch_up(&mut stream).await;
        push_faults(
            tv.client(),
            1,
            fred::error::ErrorKind::Auth,
            "synthetic permanent XREAD failure",
        );
        // This commit returns the reader's in-flight read, and its next XREAD
        // takes the fault. (Had the reader not yet reissued its read when the
        // fault was queued, that read takes it and this commit is never read;
        // the ending is the same.)
        commit(&svc, &session_id, LAYER, object_id, 1).await;
        let status = drain_to_terminal(&mut stream)
            .await
            .expect("a permanent consumer failure must end with a status");
        assert_eq!(status.code(), Code::Unavailable);
        assert_eq!(
            tv.client()
                .stream_mux
                .pending_faults_for_test(live_edit::storage::stream_mux::MuxFaultSite::Xread),
            0,
            "precondition: the fault was popped"
        );
        await_stream_exit(&svc.metrics, STREAM_EXIT_ERROR).await;

        let captured = capture.snapshot();
        let event = only_failure(&captured);
        assert_common_fields(event, "WARN", "SubscribeReadLayer", "stream", "UNAVAILABLE");
        assert_field(event, "reason", STREAM_EXIT_ERROR.into());
        assert_field(event, "caught_up", true.into());
        assert_field(event, "status_message", status.message().into());

        drop(capture);
        purge_keys(&tv).await;
    }

    /// A stream reader that ends with no recorded reason (a panic drops its
    /// subscriptions' sinks) ends the RPC `UNAVAILABLE` with the one "no
    /// recorded reason" message, under `reason="error"`, with its WARN
    /// completion line. The set of endings that deliver no status is
    /// `{client_disconnect}`; a reason-less reader ending is not in it.
    #[tokio::test]
    async fn subscribe_reader_ending_without_a_reason_logs_warn_unavailable_under_error() {
        use live_edit::storage::notifications::READER_ENDED_WITHOUT_REASON_MSG;
        use live_edit::storage::stream_mux::{MuxFault, MuxFaultSite};
        const LAYER: &str = "stream-log-reasonless";

        let tv = TestValkey::setup().await;
        tv.client()
            .stream_mux
            .push_faults_for_test(MuxFaultSite::Xread, [MuxFault::Panic]);
        let svc = build_service(&tv, Arc::new(AllowAllPermissions));
        let (session_id, object_id) =
            create_session_and_layer(&tv, &svc, "stream-log-reasonless", LAYER).await;
        let capture = ThreadCapture::start();

        let mut stream: SubscribeStream =
            handle_subscribe_read_layer(&svc, subscribe_request(&session_id, LAYER, 0, 0))
                .await
                .unwrap_or_else(|s| panic!("setup must succeed: {s:?}"))
                .into_inner();
        drain_catch_up(&mut stream).await;
        // The reader's first XREAD panics; this commit is never read.
        commit(&svc, &session_id, LAYER, object_id, 1).await;
        let status = drain_to_terminal(&mut stream)
            .await
            .expect("a reader that ended with no reason must still end the RPC with a status");
        assert_eq!(status.code(), Code::Unavailable);
        assert_eq!(status.message(), READER_ENDED_WITHOUT_REASON_MSG);
        await_stream_exit(&svc.metrics, STREAM_EXIT_ERROR).await;

        let captured = capture.snapshot();
        let event = only_failure(&captured);
        assert_common_fields(event, "WARN", "SubscribeReadLayer", "stream", "UNAVAILABLE");
        assert_field(event, "reason", STREAM_EXIT_ERROR.into());
        assert_field(
            event,
            "status_message",
            READER_ENDED_WITHOUT_REASON_MSG.into(),
        );

        drop(capture);
        purge_keys(&tv).await;
    }

    /// Park a brand-new subscription's catch-up after its first chunk, let
    /// `fire` end it there, and return the one completion line its tail logged
    /// (with the status the client received), once the exit counted under
    /// `reason`.
    async fn truncated_catch_up_line(
        name: &str,
        fire: impl FnOnce(&LiveEditService, &str),
        reason: &str,
    ) -> (Status, Value) {
        use live_edit::grpc::subscribe::CatchUpChunkSeam;

        let tv = TestValkey::setup().await;
        let seam = Arc::new(CatchUpChunkSeam::new());
        let mut svc = build_service(&tv, Arc::new(AllowAllPermissions));
        svc.catch_up_chunk_seam_for_test = Some(Arc::clone(&seam));
        let (session_id, object_id) = create_session_and_layer(&tv, &svc, name, name).await;
        for n in 1..=3 {
            commit(&svc, &session_id, name, object_id, n).await;
        }
        let capture = ThreadCapture::start();

        // `object_id = 0`: a brand-new subscribe routes through the chunk loop
        // the seam sits in.
        let mut stream: SubscribeStream =
            handle_subscribe_read_layer(&svc, subscribe_request(&session_id, name, 0, 0))
                .await
                .unwrap_or_else(|s| panic!("setup must succeed: {s:?}"))
                .into_inner();
        tokio::time::timeout(SETTLE, seam.wait_until_parked())
            .await
            .expect("the subscriber never parked at the catch-up chunk seam");
        fire(&svc, &session_id);
        tokio::time::timeout(SETTLE, seam.release())
            .await
            .expect("the parked subscriber must accept its release");
        let status = drain_to_terminal(&mut stream)
            .await
            .expect("a truncated catch-up must end with a status");
        await_stream_exit(&svc.metrics, reason).await;

        let captured = capture.snapshot();
        let event = only_failure(&captured).clone();
        drop(capture);
        purge_keys(&tv).await;
        (status, event)
    }

    /// A session deleted mid catch-up logs INFO `NOT_FOUND` with
    /// `reason=session_cancel` and `caught_up=false` — the truncated-catch-up
    /// ending of a deletion.
    #[tokio::test]
    async fn subscribe_session_cancel_mid_catch_up_logs_info_not_found_with_its_reason() {
        let (status, event) = truncated_catch_up_line(
            "stream-log-cancel-mid",
            |svc, session_id| {
                svc.session_tokens
                    .get(session_id)
                    .expect("the handler registered a session token")
                    .cancel()
            },
            STREAM_EXIT_SESSION_CANCEL,
        )
        .await;
        assert_eq!(status.code(), Code::NotFound, "got {status:?}");
        assert_common_fields(&event, "INFO", "SubscribeReadLayer", "stream", "NOT_FOUND");
        assert_field(&event, "reason", STREAM_EXIT_SESSION_CANCEL.into());
        assert_field(&event, "caught_up", false.into());
        assert_field(&event, "regime", "brand_new".into());
        assert_field(&event, "status_message", "session was deleted".into());
    }

    /// A shutdown mid catch-up logs INFO `UNAVAILABLE` — not WARN — with
    /// `reason=shutdown` and `caught_up=false`.
    #[tokio::test]
    async fn subscribe_shutdown_mid_catch_up_logs_info_unavailable_with_its_reason() {
        let (status, event) = truncated_catch_up_line(
            "stream-log-shutdown-mid",
            |svc, _session_id| svc.shutdown.cancel(),
            STREAM_EXIT_SHUTDOWN,
        )
        .await;
        assert_eq!(status.code(), Code::Unavailable, "got {status:?}");
        assert_common_fields(
            &event,
            "INFO",
            "SubscribeReadLayer",
            "stream",
            "UNAVAILABLE",
        );
        assert_field(&event, "reason", STREAM_EXIT_SHUTDOWN.into());
        assert_field(&event, "caught_up", false.into());
        assert_field(&event, "regime", "brand_new".into());
        assert_field(&event, "status_message", "server shutting down".into());
    }
}

// =============================================================================
// (c) JoinSession setup: logged by the handler wrapper
// =============================================================================

/// A first message that is not a header logs INFO at `stage=header` with no
/// identifiers.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn join_first_message_not_a_header_logs_info_at_header() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let not_a_header = JoinSessionRequest {
        request_type: Some(join_session_request::RequestType::Message(
            SessionMessageRequest::default(),
        )),
    };

    let (result, captured) = capture(handle_join_session(
        &svc,
        join_request(streaming_join(vec![not_a_header])),
    ))
    .await;

    let status = expect_err_status(result, "a non-header first message must be refused");
    assert_eq!(status.code(), Code::InvalidArgument);
    let event = only_failure(&captured);
    assert_common_fields(event, "INFO", "JoinSession", "header", "INVALID_ARGUMENT");
    assert_field(
        event,
        "status_message",
        "first message must be a JoinSessionHeader".into(),
    );
    assert_field_absent(event, "session_id", "no header was read");
    captured.assert_absent_everywhere(BEARER);
}

/// A first message that is not valid protobuf passes tonic's own `INTERNAL`
/// through unchanged — and it is logged as one WARN `RPC failed`, never as an
/// ERROR: it carries no server-fault source, and a client can send it at will.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn join_undecodable_first_message_logs_one_warn_and_no_error() {
    use bytes::BufMut;
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    // One well-framed message whose payload is not a JoinSessionRequest: 0xFF
    // is a field tag with an invalid wire type.
    let mut frames = bytes::BytesMut::new();
    let payload = [0xFFu8, 0xFF, 0xFF, 0xFF];
    frames.put_u8(0);
    frames.put_u32(payload.len() as u32);
    frames.extend_from_slice(&payload);

    let (result, captured) = capture(handle_join_session(
        &svc,
        join_request(join_stream_from_frames(frames.freeze())),
    ))
    .await;

    let status = expect_err_status(result, "an undecodable first message must be refused");
    assert_eq!(
        status.code(),
        Code::Internal,
        "precondition: tonic answers a decode failure INTERNAL; got {status:?}"
    );
    let event = only_failure(&captured);
    assert_common_fields(event, "WARN", "JoinSession", "header", "INTERNAL");
    assert!(
        captured.with_message(INTERNAL_ERROR_MSG).is_empty(),
        "a client-caused decode failure must not produce the server-fault ERROR; full \
         capture:\n{}",
        captured.text
    );
}

/// An invalid `session_id` logs INFO at `stage=header` with no identifiers,
/// and the rejected value appears nowhere.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn join_invalid_session_id_logs_info_at_header_with_no_identifiers() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let (result, captured) = capture(handle_join_session(
        &svc,
        join_request(streaming_join(vec![join_header(CONTROL_CHAR_SESSION_ID)])),
    ))
    .await;

    assert_eq!(
        expect_err_status(result, "must be refused").code(),
        Code::InvalidArgument
    );
    let event = only_failure(&captured);
    assert_common_fields(event, "INFO", "JoinSession", "header", "INVALID_ARGUMENT");
    for absent in ["session_id", "base_path", "connection_id"] {
        assert_field_absent(event, absent, "nothing validated yet");
    }
    captured.assert_absent_everywhere(CONTROL_CHAR_SESSION_ID);
    captured.assert_absent_everywhere("\u{1b}[");
}

/// A missing session logs INFO `NOT_FOUND` at `stage=auth`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn join_missing_session_logs_info_not_found_at_auth() {
    const MISSING: &str = "stream-log-join-missing";
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let (result, captured) = capture(handle_join_session(
        &svc,
        join_request(streaming_join(vec![join_header(MISSING)])),
    ))
    .await;

    assert_eq!(
        expect_err_status(result, "must be refused").code(),
        Code::NotFound
    );
    let event = only_failure(&captured);
    assert_common_fields(event, "INFO", "JoinSession", "auth", "NOT_FOUND");
    assert_field(event, "session_id", MISSING.into());
    assert_field_absent(event, "base_path", "the lookup that produces it failed");
    assert_field_absent(event, "connection_id", "minted only after auth");
}

/// A permission denial logs WARN `PERMISSION_DENIED` at `stage=auth` with the
/// `base_path`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn join_permission_denied_logs_warn_at_auth_with_the_base_path() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(DenyAllPermissions));
    let session_id = create_session(&tv, &svc, "stream-log-join-denied").await;

    let (result, captured) = capture(handle_join_session(
        &svc,
        join_request(streaming_join(vec![join_header(&session_id)])),
    ))
    .await;

    assert_eq!(
        expect_err_status(result, "must be refused").code(),
        Code::PermissionDenied
    );
    let event = only_failure(&captured);
    assert_common_fields(event, "WARN", "JoinSession", "auth", "PERMISSION_DENIED");
    assert_field(event, "session_id", session_id.as_str().into());
    assert_field(event, "base_path", "/stage".into());
    captured.assert_absent_everywhere(BEARER);
}

/// A `UserinfoClient` whose backend times out.
struct TimingOutUserinfo;

#[async_trait::async_trait]
impl UserinfoClient for TimingOutUserinfo {
    async fn get_current_user(&self, _token: &str) -> Result<UserInfo, AuthError> {
        Err(AuthError::Timeout)
    }
}

/// A transient userinfo failure logs WARN `UNAVAILABLE` at `stage=auth`, with
/// the session and `base_path` it had resolved.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn join_transient_userinfo_failure_logs_warn_unavailable_at_auth() {
    let tv = TestValkey::setup().await;
    let svc = build_service_with_userinfo(
        &tv,
        Arc::new(AllowAllPermissions),
        Arc::new(TimingOutUserinfo),
    );
    let session_id = create_session(&tv, &svc, "stream-log-join-userinfo").await;

    let (result, captured) = capture(handle_join_session(
        &svc,
        join_request(streaming_join(vec![join_header(&session_id)])),
    ))
    .await;

    let status = expect_err_status(result, "a userinfo outage must be refused at setup");
    assert_eq!(status.code(), Code::Unavailable);
    let event = only_failure(&captured);
    assert_common_fields(event, "WARN", "JoinSession", "auth", "UNAVAILABLE");
    assert_field(event, "session_id", session_id.as_str().into());
    assert_field(event, "base_path", "/stage".into());
    assert_field(event, "status_message", status.message().into());
}

/// A userinfo `InvalidResponse` maps to `ServiceError::Internal`, a server
/// fault: one ERROR `internal error` at `stage=auth`, carrying the session,
/// and no `RPC failed`. Driven with the in-process JWT decoder and an opaque
/// (non-JWT) bearer, the route `join_session_test.rs` uses for the same
/// status.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn join_userinfo_server_fault_logs_one_error_with_the_session() {
    let tv = TestValkey::setup().await;
    let svc = build_service_with_userinfo(
        &tv,
        Arc::new(AllowAllPermissions),
        Arc::new(live_edit::auth::JwtUserinfoClient::new()),
    );
    let session_id = create_session(&tv, &svc, "stream-log-join-opaque").await;

    // The status is dropped inside the capture, so a fault the wrapper failed
    // to mark reported would show up as a second, context-free ERROR.
    let (code, captured) = capture(async {
        let result = handle_join_session(
            &svc,
            join_request(streaming_join(vec![join_header(&session_id)])),
        )
        .await;
        expect_err_status(result, "an opaque (non-JWT) bearer must be refused").code()
    })
    .await;

    assert_eq!(code, Code::Internal);
    assert!(captured.with_message(RPC_FAILURE_MSG).is_empty());
    let event = captured.only(INTERNAL_ERROR_MSG);
    assert_common_fields(event, "ERROR", "JoinSession", "auth", "INTERNAL");
    assert_field(event, "session_id", session_id.as_str().into());
    assert_field(event, "base_path", "/stage".into());
    captured.assert_absent_everywhere(BEARER);
}

/// A join that arrives once service shutdown has begun is refused
/// `UNAVAILABLE` and logged once, INFO, at `stage=admission`. At
/// `stage=auth` the refusal would read as an auth-backend outage, and at WARN
/// every rolling restart would look like a dependency failure.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn join_during_shutdown_logs_one_info_unavailable_at_admission() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let session_id = create_session(&tv, &svc, "stream-log-join-shutdown").await;
    svc.shutdown.cancel();

    let (result, captured) = capture(handle_join_session(
        &svc,
        join_request(streaming_join(vec![join_header(&session_id)])),
    ))
    .await;

    let status = expect_err_status(result, "a join during shutdown must be refused at setup");
    assert_eq!(status.code(), Code::Unavailable);
    let event = only_failure(&captured);
    assert_common_fields(event, "INFO", "JoinSession", "admission", "UNAVAILABLE");
    assert_field(event, "session_id", session_id.as_str().into());
    assert_field(event, "status_message", "server shutting down".into());
    captured.assert_absent_everywhere(BEARER);
}

/// A join the handler admitted, whose task finds shutdown already begun at its
/// start-of-task check, ends `UNAVAILABLE` and logs exactly one INFO line at
/// `stage=stream` — logged by the task itself, since no teardown follows to
/// log it, and not a second time by anything else.
///
/// Current-thread runtime: the handler returns without yielding after the
/// spawn, so the task has not run when the test cancels the token.
#[tokio::test]
async fn join_shutdown_before_task_start_logs_one_info_unavailable_at_stream() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let session_id = create_session(&tv, &svc, "stream-log-join-task-start").await;
    let capture = ThreadCapture::start();

    let mut stream = handle_join_session(
        &svc,
        join_request(streaming_join(vec![join_header(&session_id)])),
    )
    .await
    .unwrap_or_else(|s| panic!("the handler must admit the join before shutdown: {s:?}"))
    .into_inner();
    svc.shutdown.cancel();
    let status = tokio::time::timeout(SETTLE, async {
        while let Some(item) = stream.next().await {
            if let Err(status) = item {
                return Some(status);
            }
        }
        None
    })
    .await
    .expect("the join stream did not end within the deadline")
    .expect("a join stopped by shutdown at task start must end UNAVAILABLE, not OK");
    assert_eq!(status.code(), Code::Unavailable);
    await_join_tasks(&svc).await;

    let captured = capture.snapshot();
    let event = only_failure(&captured);
    assert_common_fields(event, "INFO", "JoinSession", "stream", "UNAVAILABLE");
    assert_field(event, "session_id", session_id.as_str().into());
    assert_field(event, "status_message", "server shutting down".into());

    drop(capture);
    purge_keys(&tv).await;
}

/// An established join ended by shutdown logs one INFO `UNAVAILABLE` line at
/// `stage=stream` **before its teardown** (SDD §13.2): teardown awaits Valkey
/// and the shutdown sequence bounds how long it waits for the task, so a line
/// logged after it could be lost with the task.
///
/// The ordering is made observable with the lease keeper seam. The keeper is
/// parked inside a step before its refresh; teardown then waits out its
/// bounded wait for the keeper before it unregisters the connection. The test
/// requires the line to be in the capture while the connection is still a
/// member — that is, before teardown's unregister — and then releases the
/// keeper and checks teardown completed.
///
/// Compiled only with `test-fault-injection` (the keeper seam).
#[cfg(feature = "test-fault-injection")]
#[tokio::test]
async fn join_shutdown_of_an_established_join_logs_one_info_before_its_teardown() {
    use fred::interfaces::SetsInterface;
    use live_edit::grpc::presence::LeaseKeeperSeam;

    let tv = TestValkey::setup().await;
    let keeper_seam = Arc::new(LeaseKeeperSeam::new());
    let svc = LiveEditService {
        presence_refresh_interval_secs: 1,
        presence_keepalive_interval_ms: live_edit::presence::DEFAULT_PRESENCE_KEEPALIVE_INTERVAL_MS,
        lease_keeper_seam_for_test: Some(Arc::clone(&keeper_seam)),
        ..build_service(&tv, Arc::new(AllowAllPermissions))
    };
    let session_id = create_session(&tv, &svc, "stream-log-join-shutdown-loop").await;
    let capture = ThreadCapture::start();

    let (mut stream, tx) = super::helpers::open_join(&svc, &session_id, JOIN_APP).await;
    let connection_id = super::helpers::expect_session_connected(&mut stream, SETTLE).await;
    let gate = &keeper_seam.before_refresh;
    gate.arm();
    tokio::time::timeout(SETTLE, gate.wait_until_parked())
        .await
        .expect("the lease keeper's step did not park before its refresh");

    svc.shutdown.cancel();
    async fn is_member(tv: &TestValkey, session_id: &str, connection_id: &str) -> bool {
        tv.client()
            .maintenance_pool
            .next()
            .sismember::<bool, _, _>(tv.key_schema().presence_members(session_id), connection_id)
            .await
            .expect("SISMEMBER must succeed")
    }
    let deadline = tokio::time::Instant::now() + SETTLE;
    loop {
        if !capture.snapshot().with_message(RPC_FAILURE_MSG).is_empty() {
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "the shutdown ending was not logged within {SETTLE:?}"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(
        is_member(&tv, &session_id, &connection_id).await,
        "the completion line must be logged before teardown unregisters the connection; the \
         connection was already gone when the line appeared"
    );

    gate.release();
    let (_, status) = super::helpers::read_to_end(&mut stream, "the join after shutdown").await;
    let status = status.expect("an established join ended by shutdown must end UNAVAILABLE");
    assert_eq!(status.code(), Code::Unavailable);
    await_join_tasks(&svc).await;
    assert!(
        !is_member(&tv, &session_id, &connection_id).await,
        "teardown must have unregistered the connection"
    );

    let captured = capture.snapshot();
    let event = only_failure(&captured);
    assert_common_fields(event, "INFO", "JoinSession", "stream", "UNAVAILABLE");
    assert_field(event, "connection_id", connection_id.as_str().into());
    assert_field(event, "status_message", "server shutting down".into());

    drop(tx);
    drop(capture);
    purge_keys(&tv).await;
}

// ── A join ended mid-replay (SDD SR-4.4) ─────────────────────────────────────
//
// The replay is held where only its own cancel arms can end it: blocked on a
// send to a client that read `SessionConnected` and stopped, with page one
// still holding members to send. The bidi loop is never reached, so every line
// (or its absence) here is the replay exit's.

/// Seed `ONE_PAGE_OVERFILLING_THE_CHANNEL` other members into `session_id`,
/// open a join on `svc` (whose replay seam is `replay`), read its
/// `SessionConnected` and nothing more, and wait until the replay is blocked
/// on a send: its one pipeline issued and `RESPONSE_DELIVERY_SLOTS`
/// `PeerJoined` enqueued. Returns the response stream, the client sender and
/// the connection id.
#[cfg(feature = "test-fault-injection")]
async fn join_blocked_mid_replay(
    tv: &TestValkey,
    svc: &LiveEditService,
    replay: &live_edit::grpc::presence::PresenceReplaySeam,
    session_id: &str,
) -> (
    super::helpers::JoinResponseStream,
    tokio::sync::mpsc::UnboundedSender<bytes::Bytes>,
    String,
) {
    use super::helpers::{
        seed_presence_members, seeded_cids, ONE_PAGE_OVERFILLING_THE_CHANNEL,
        RESPONSE_DELIVERY_SLOTS,
    };

    seed_presence_members(
        tv,
        session_id,
        &seeded_cids(ONE_PAGE_OVERFILLING_THE_CHANNEL),
    )
    .await;
    let (mut stream, tx) = super::helpers::open_join(svc, session_id, JOIN_APP).await;
    let connection_id = super::helpers::expect_session_connected(&mut stream, SETTLE).await;
    tokio::time::timeout(
        SETTLE,
        replay.wait_for_peer_joined_sent(RESPONSE_DELIVERY_SLOTS),
    )
    .await
    .expect("the replay did not fill the response channel");
    assert_eq!(
        replay.pipelines_issued(),
        1,
        "precondition: the replay is on its only page"
    );
    assert_eq!(
        replay.peer_joined_sent_count(),
        RESPONSE_DELIVERY_SLOTS,
        "precondition: the replay is blocked on a send"
    );
    (stream, tx, connection_id)
}

/// A shutdown while the replay is blocked on a send ends the join through the
/// replay's own shutdown arm, with exactly one INFO `UNAVAILABLE` line at
/// `stage=stream` logged **before teardown**, as a shutdown in the loop logs
/// it.
///
/// The ordering is made observable as in
/// `join_shutdown_of_an_established_join_logs_one_info_before_its_teardown`:
/// the lease keeper is parked inside a step, so teardown waits out its bound
/// for the keeper before it unregisters, and the line must be in the capture
/// while the connection is still a member.
#[cfg(feature = "test-fault-injection")]
#[tokio::test]
async fn join_shutdown_mid_replay_logs_one_info_unavailable_before_its_teardown() {
    use live_edit::grpc::presence::{LeaseKeeperSeam, ObservedReplayResult, PresenceReplaySeam};

    let tv = TestValkey::setup().await;
    let keeper_seam = Arc::new(LeaseKeeperSeam::new());
    let replay_seam = Arc::new(PresenceReplaySeam::new());
    let svc = LiveEditService {
        presence_refresh_interval_secs: 1,
        presence_keepalive_interval_ms: live_edit::presence::DEFAULT_PRESENCE_KEEPALIVE_INTERVAL_MS,
        lease_keeper_seam_for_test: Some(Arc::clone(&keeper_seam)),
        presence_replay_seam_for_test: Some(Arc::clone(&replay_seam)),
        ..build_service(&tv, Arc::new(AllowAllPermissions))
    };
    let session_id = create_session(&tv, &svc, "stream-log-join-shutdown-replay").await;
    let capture = ThreadCapture::start();

    let (mut stream, tx, connection_id) =
        join_blocked_mid_replay(&tv, &svc, &replay_seam, &session_id).await;
    let gate = &keeper_seam.before_refresh;
    gate.arm();
    tokio::time::timeout(SETTLE, gate.wait_until_parked())
        .await
        .expect("the lease keeper's step did not park before its refresh");

    svc.shutdown.cancel();
    let result = tokio::time::timeout(SETTLE, replay_seam.wait_for_result())
        .await
        .expect("the replay did not end within the deadline");
    assert_eq!(
        result,
        ObservedReplayResult::ShuttingDown,
        "the replay's own shutdown arm must end the join"
    );
    let deadline = tokio::time::Instant::now() + SETTLE;
    while capture.snapshot().with_message(RPC_FAILURE_MSG).is_empty() {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the shutdown ending was not logged within {SETTLE:?}"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(
        super::helpers::is_member(&tv, &session_id, &connection_id).await,
        "the completion line must be logged before teardown unregisters the connection; the \
         connection was already gone when the line appeared"
    );

    gate.release();
    let (_, status) = super::helpers::read_to_end(&mut stream, "the join after shutdown").await;
    let status = status.expect("a join ended by shutdown mid-replay must end UNAVAILABLE");
    assert_eq!(status.code(), Code::Unavailable);
    await_join_tasks(&svc).await;
    assert!(
        !super::helpers::is_member(&tv, &session_id, &connection_id).await,
        "teardown must have unregistered the connection"
    );

    let captured = capture.snapshot();
    let event = only_failure(&captured);
    assert_common_fields(event, "INFO", "JoinSession", "stream", "UNAVAILABLE");
    assert_field(event, "connection_id", connection_id.as_str().into());
    assert_field(event, "status_message", "server shutting down".into());

    drop(tx);
    drop(capture);
    purge_keys(&tv).await;
}

/// A session deletion while the replay is blocked on a send ends the join
/// through the replay's own session-token arm, `OK`, and logs no completion
/// line: a deletion is not an RPC failure.
#[cfg(feature = "test-fault-injection")]
#[tokio::test]
async fn join_session_deleted_mid_replay_ends_ok_and_logs_nothing() {
    use live_edit::grpc::presence::{ObservedReplayResult, PresenceReplaySeam};

    let tv = TestValkey::setup().await;
    let replay_seam = Arc::new(PresenceReplaySeam::new());
    let svc = LiveEditService {
        presence_replay_seam_for_test: Some(Arc::clone(&replay_seam)),
        ..build_service(&tv, Arc::new(AllowAllPermissions))
    };
    let session_id = create_session(&tv, &svc, "stream-log-join-delete-replay").await;
    let capture = ThreadCapture::start();

    let (mut stream, tx, _) = join_blocked_mid_replay(&tv, &svc, &replay_seam, &session_id).await;
    svc.session_tokens
        .get(&session_id)
        .expect("the session's token is registered")
        .cancel_deleted();
    let result = tokio::time::timeout(SETTLE, replay_seam.wait_for_result())
        .await
        .expect("the replay did not end within the deadline");
    assert_eq!(
        result,
        ObservedReplayResult::SessionCancelled,
        "the replay's own session-token arm must end the join"
    );
    let (_, status) =
        super::helpers::read_to_end(&mut stream, "the join after its session was deleted").await;
    assert!(
        status.is_none(),
        "a deletion ends the join OK; got {status:?}"
    );
    await_join_tasks(&svc).await;

    capture
        .snapshot()
        .assert_no_completion_line("a JoinSession whose session was deleted mid-replay");

    drop(tx);
    drop(capture);
    purge_keys(&tv).await;
}

/// A client that leaves while the replay is blocked on a send ends the join
/// `CANCELLED`, logged exactly once, at INFO, `stage=stream`, with the
/// connection id.
#[cfg(feature = "test-fault-injection")]
#[tokio::test]
async fn join_client_leaving_mid_replay_logs_one_info_cancelled_at_stream() {
    use live_edit::grpc::presence::{ObservedReplayResult, PresenceReplaySeam};

    let tv = TestValkey::setup().await;
    let replay_seam = Arc::new(PresenceReplaySeam::new());
    let svc = LiveEditService {
        presence_replay_seam_for_test: Some(Arc::clone(&replay_seam)),
        ..build_service(&tv, Arc::new(AllowAllPermissions))
    };
    let session_id = create_session(&tv, &svc, "stream-log-join-leave-replay").await;
    let capture = ThreadCapture::start();

    let (stream, tx, connection_id) =
        join_blocked_mid_replay(&tv, &svc, &replay_seam, &session_id).await;
    drop(stream);
    let result = tokio::time::timeout(SETTLE, replay_seam.wait_for_result())
        .await
        .expect("the replay did not end within the deadline");
    assert_eq!(result, ObservedReplayResult::Failed(Code::Cancelled));
    await_join_tasks(&svc).await;

    let captured = capture.snapshot();
    let event = only_failure(&captured);
    assert_common_fields(event, "INFO", "JoinSession", "stream", "CANCELLED");
    assert_field(event, "connection_id", connection_id.as_str().into());

    drop(tx);
    drop(capture);
    purge_keys(&tv).await;
}

// =============================================================================
// (d) JoinSession teardown: logged by the spawned task, before cleanup
// =============================================================================

/// A client that leaves before `SessionConnected` is sent is logged —
/// `CANCELLED` at INFO, `stage=stream`, with the server-minted
/// `connection_id` — and the user's name and application are not.
///
/// The response stream is dropped before the spawned task first runs (the
/// runtime is current-thread), so the task's `SessionConnected` send finds the
/// receiver gone.
#[tokio::test]
async fn join_client_leaving_before_session_connected_logs_info_cancelled_at_stream() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let session_id = create_session(&tv, &svc, "stream-log-join-leave").await;
    let capture = ThreadCapture::start();

    let response = handle_join_session(
        &svc,
        join_request(streaming_join(vec![join_header(&session_id)])),
    )
    .await
    .unwrap_or_else(|s| panic!("setup must succeed: {s:?}"));
    drop(response);
    await_join_tasks(&svc).await;

    let captured = capture.snapshot();
    let event = only_failure(&captured);
    assert_common_fields(event, "INFO", "JoinSession", "stream", "CANCELLED");
    assert_field(event, "session_id", session_id.as_str().into());
    assert_field(event, "base_path", "/stage".into());
    let connection_id = event["fields"]["connection_id"]
        .as_str()
        .unwrap_or_default();
    assert!(
        uuid::Uuid::parse_str(connection_id).is_ok(),
        "the teardown line must carry the server-minted connection_id; got {event}"
    );
    // The user's display name and application are never on the line.
    let line = event.to_string();
    for private in [JOIN_APP, "Test User", "testuser"] {
        assert!(!line.contains(private), "`{private}` leaked onto {line}");
    }

    drop(capture);
    purge_keys(&tv).await;
}

/// A presence registration that fails permanently is a server fault logged by
/// the join task, before it returns: exactly one ERROR `internal error` at
/// `stage=stream`, carrying the `connection_id` and the un-redacted detail, no
/// `RPC failed`, and no context-free fallback once the task and the client's
/// copy of the status are gone.
///
/// The session's presence-members set is replaced with a plain string, so the
/// registration's `SADD` fails `WRONGTYPE` — a permanent error, hence
/// `INTERNAL`. Registration runs in the spawned task, after the handler
/// returned `Ok`, so this is the one teardown path that fails before the
/// bidirectional flow even starts.
#[tokio::test]
async fn join_registration_failure_logs_one_error_at_stream_with_the_connection_id() {
    use fred::interfaces::KeysInterface;

    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let session_id = create_session(&tv, &svc, "stream-log-join-register").await;
    let _: () = tv
        .client()
        .pool
        .set(
            tv.key_schema().presence_members(&session_id),
            "not-a-set",
            None,
            None,
            false,
        )
        .await
        .expect("retyping the presence-members key must succeed");
    let capture = ThreadCapture::start();

    let mut stream = handle_join_session(
        &svc,
        join_request(streaming_join(vec![join_header(&session_id)])),
    )
    .await
    .unwrap_or_else(|s| panic!("setup must succeed; registration runs in the task: {s:?}"))
    .into_inner();
    let status = tokio::time::timeout(SETTLE, async {
        while let Some(item) = stream.next().await {
            if let Err(status) = item {
                return status;
            }
        }
        panic!("the join stream ended without a status");
    })
    .await
    .expect("the join stream did not end within the deadline");
    assert_eq!(
        status.code(),
        Code::Internal,
        "precondition: a WRONGTYPE registration answers INTERNAL; got {status:?}"
    );
    assert_eq!(
        status.message(),
        INTERNAL_ERROR_MSG,
        "the wire stays redacted"
    );
    // The client's copy goes first, then the task: a fault the task failed to
    // mark reported would log its fallback at one of these two drops.
    drop(status);
    drop(stream);
    await_join_tasks(&svc).await;

    let captured = capture.snapshot();
    assert!(
        captured.with_message(RPC_FAILURE_MSG).is_empty(),
        "a server fault logs `internal error`, not `RPC failed`; full capture:\n{}",
        captured.text
    );
    let event = captured.only(INTERNAL_ERROR_MSG);
    assert_common_fields(event, "ERROR", "JoinSession", "stream", "INTERNAL");
    assert_field(event, "session_id", session_id.as_str().into());
    assert_field(event, "base_path", "/stage".into());
    let connection_id = event["fields"]["connection_id"]
        .as_str()
        .unwrap_or_default();
    assert!(
        uuid::Uuid::parse_str(connection_id).is_ok(),
        "the registration line must carry the server-minted connection_id; got {event}"
    );
    let detail = event["fields"]["error"].as_str().unwrap_or_default();
    assert!(
        detail.contains("WRONGTYPE"),
        "the ERROR line must carry the un-redacted detail the wire lost; got {event}"
    );

    drop(capture);
    purge_keys(&tv).await;
}

/// A transient failure in the spawned task, after `SessionConnected`, logs
/// WARN `UNAVAILABLE` at `stage=stream` with the `connection_id`.
#[cfg(feature = "test-fault-injection")]
#[tokio::test]
async fn join_transient_in_task_failure_logs_warn_unavailable_at_stream() {
    use std::collections::VecDeque;
    use std::sync::Mutex;

    use live_edit::error::ServiceError;
    const INJECTED: &str = "transient I/O: HGET session base_path failed (Timeout): injected";

    let tv = TestValkey::setup().await;
    let fault_queue: Arc<Mutex<VecDeque<ServiceError>>> = Arc::new(Mutex::new(VecDeque::new()));
    let base = build_service(&tv, Arc::new(AllowAllPermissions));
    let svc = LiveEditService {
        session_store: Arc::new(
            SessionStore::new(tv.client().clone(), tv.key_schema().clone())
                .with_get_base_path_fault_injector_for_test(fault_queue.clone()),
        ),
        ..base
    };
    let session_id = svc
        .session_store
        .create("stream-log-join-in-task", "/stage", "test-user")
        .await
        .expect("session creation must succeed");
    // Warm the handler's cache so its own lookup cannot take the fault.
    svc.session_base_path_cache
        .get_base_path(&svc.session_store, &session_id)
        .await
        .expect("warming the base-path cache must succeed");
    fault_queue
        .lock()
        .unwrap()
        .push_back(ServiceError::Aborted(INJECTED.to_owned()));
    let capture = ThreadCapture::start();

    let mut stream = handle_join_session(
        &svc,
        join_request(streaming_join(vec![join_header(&session_id)])),
    )
    .await
    .unwrap_or_else(|s| panic!("setup must succeed: {s:?}"))
    .into_inner();
    let status = tokio::time::timeout(SETTLE, async {
        while let Some(item) = stream.next().await {
            if let Err(status) = item {
                return status;
            }
        }
        panic!("the join stream ended without a status");
    })
    .await
    .expect("the join stream did not end within the deadline");
    assert_eq!(status.code(), Code::Unavailable);
    await_join_tasks(&svc).await;

    let captured = capture.snapshot();
    let event = only_failure(&captured);
    assert_common_fields(event, "WARN", "JoinSession", "stream", "UNAVAILABLE");
    assert_field(event, "status_message", INJECTED.into());
    assert!(
        event["fields"]["connection_id"].is_string(),
        "the in-task line must carry the connection_id; got {event}"
    );

    drop(capture);
    purge_keys(&tv).await;
}

/// A normal join and leave logs nothing: the client stream ends after its
/// header, the bidirectional loop returns `Ok`, and cleanup runs.
#[tokio::test]
async fn join_normal_join_and_leave_logs_nothing() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let session_id = create_session(&tv, &svc, "stream-log-join-ok").await;
    let capture = ThreadCapture::start();

    let mut stream = handle_join_session(
        &svc,
        join_request(streaming_join(vec![join_header(&session_id)])),
    )
    .await
    .unwrap_or_else(|s| panic!("setup must succeed: {s:?}"))
    .into_inner();
    let first = tokio::time::timeout(SETTLE, stream.next())
        .await
        .expect("SessionConnected must arrive")
        .expect("the stream must not be empty")
        .expect("the first message must not be an error");
    assert!(
        matches!(
            first.response_type,
            Some(join_session_response::ResponseType::Connected(_))
        ),
        "precondition: the join succeeded"
    );
    await_join_tasks(&svc).await;
    drop(stream);

    capture
        .snapshot()
        .assert_no_completion_line("a JoinSession that joined and left normally");

    drop(capture);
    purge_keys(&tv).await;
}
