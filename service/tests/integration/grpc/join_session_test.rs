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

// Integration tests for the JoinSession bidirectional streaming handler.
//
// Tests call `handle_join_session` directly with a LiveEditService backed by
// real Valkey (TestValkey fixture), mock auth, and gRPC-framed streaming
// requests.
//
// JoinSession is bidirectional streaming: the client sends a header then
// messages, and the server responds with SessionConnected, membership replay,
// and presence events. Tests verify the connection handshake, auth, and
// cleanup paths.

use std::sync::Arc;

use bytes::BufMut;
use futures::StreamExt;
use prost::Message;
use tonic::codec::{BufferSettings, Streaming};
use tonic::Request;
use tonic_prost::ProstCodec;

use live_edit::auth::userinfo::UserinfoClient;
use live_edit::grpc::presence::handle_join_session;
use live_edit::grpc::proto::{
    join_session_request, join_session_response, JoinSessionHeader, JoinSessionRequest,
};
use live_edit::grpc::BearerToken;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;

use super::helpers::*;

// ── Streaming helper ────────────────────────────────────────────────────────

/// Encode JoinSessionRequest messages into a gRPC-framed body.
fn streaming_join(messages: Vec<JoinSessionRequest>) -> Streaming<JoinSessionRequest> {
    let mut buf = bytes::BytesMut::new();
    for msg in &messages {
        let encoded = msg.encode_to_vec();
        buf.put_u8(0); // no compression
        buf.put_u32(encoded.len() as u32);
        buf.extend_from_slice(&encoded);
    }
    let body = http_body_util::Full::new(buf.freeze());
    let decoder = ProstCodec::<(), JoinSessionRequest>::raw_decoder(BufferSettings::default());
    Streaming::new_request(decoder, body, None, None)
}

/// Build a JoinSessionRequest header message.
fn make_join_header(session_id: &str, user_application: &str) -> JoinSessionRequest {
    JoinSessionRequest {
        request_type: Some(join_session_request::RequestType::Header(
            JoinSessionHeader {
                session_id: session_id.to_owned(),
                user_application: user_application.to_owned(),
                client_keepalive: false,
            },
        )),
    }
}

/// Wrap a JoinSession streaming request with a bearer token.
fn make_join_request(
    stream: Streaming<JoinSessionRequest>,
) -> Request<Streaming<JoinSessionRequest>> {
    let mut request = Request::new(stream);
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    request
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

// ── 1. Happy path: connect and receive SessionConnected ─────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn join_session_happy_path_returns_session_connected() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    // Create a session first.
    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = session_store
        .create("join-happy", "/stage", "test-user")
        .await
        .expect("session creation must succeed");

    // Insert a CancellationToken (normally done by handle_create_session).
    svc.session_tokens.insert(
        session_id.clone(),
        tokio_util::sync::CancellationToken::new().into(),
    );

    let stream = streaming_join(vec![make_join_header(&session_id, "test-app")]);
    let request = make_join_request(stream);

    let response = handle_join_session(&svc, request)
        .await
        .expect("join_session must succeed");

    let mut response_stream = response.into_inner();

    // First message should be SessionConnected with a non-empty connection_id.
    let first_msg = tokio::time::timeout(std::time::Duration::from_secs(5), response_stream.next())
        .await
        .expect("must receive first message within timeout")
        .expect("stream must not be empty")
        .expect("first message must not be an error");

    match first_msg.response_type {
        Some(join_session_response::ResponseType::Connected(connected)) => {
            assert!(
                !connected.connection_id.is_empty(),
                "connection_id must be non-empty"
            );
        }
        other => panic!("expected SessionConnected, got: {other:?}"),
    }
}

// ── 2. Session not found → NOT_FOUND ────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn join_session_session_not_found_returns_not_found() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let stream = streaming_join(vec![make_join_header("nonexistent-session", "test-app")]);
    let request = make_join_request(stream);

    // The handler reads the header then checks session existence.
    // Since we pre-encode the stream, the header is read then get_base_path fails.
    let result = handle_join_session(&svc, request).await;

    // JoinSession may return the error either at the handler level or via the
    // stream. Check both paths.
    match result {
        Err(status) => {
            assert_eq!(
                status.code(),
                tonic::Code::NotFound,
                "non-existent session must return NOT_FOUND, got: {}",
                status.code()
            );
        }
        Ok(response) => {
            // Error might come through the stream.
            let mut stream = response.into_inner();
            let msg = tokio::time::timeout(std::time::Duration::from_secs(5), stream.next())
                .await
                .expect("must receive message within timeout")
                .expect("stream must not be empty");

            let status = msg.expect_err("message must be an error for non-existent session");
            assert_eq!(
                status.code(),
                tonic::Code::NotFound,
                "non-existent session must return NOT_FOUND via stream, got: {}",
                status.code()
            );
        }
    }
}

// ── 3. Auth denied → PERMISSION_DENIED ──────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn join_session_auth_denied_returns_permission_denied() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(DenyAllPermissions));

    // Create a session with a direct store call (bypassing auth).
    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = session_store
        .create("join-denied", "/stage", "test-user")
        .await
        .expect("session creation must succeed");

    let stream = streaming_join(vec![make_join_header(&session_id, "test-app")]);
    let request = make_join_request(stream);

    let result = handle_join_session(&svc, request).await;

    match result {
        Err(status) => {
            assert_eq!(
                status.code(),
                tonic::Code::PermissionDenied,
                "denied auth must return PERMISSION_DENIED, got: {}",
                status.code()
            );
        }
        Ok(response) => {
            let mut stream = response.into_inner();
            let msg = tokio::time::timeout(std::time::Duration::from_secs(5), stream.next())
                .await
                .expect("must receive message within timeout")
                .expect("stream must not be empty");

            let status = msg.expect_err("message must be an error for denied auth");
            assert_eq!(
                status.code(),
                tonic::Code::PermissionDenied,
                "denied auth must return PERMISSION_DENIED via stream, got: {}",
                status.code()
            );
        }
    }
}

// ── 4. Empty session_id → INVALID_ARGUMENT ──────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn join_session_empty_session_id_returns_invalid_argument() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let stream = streaming_join(vec![make_join_header("", "test-app")]);
    let request = make_join_request(stream);

    let result = handle_join_session(&svc, request).await;

    match result {
        Err(status) => {
            assert_eq!(
                status.code(),
                tonic::Code::InvalidArgument,
                "empty session_id must return INVALID_ARGUMENT, got: {}",
                status.code()
            );
        }
        Ok(response) => {
            let mut stream = response.into_inner();
            let msg = tokio::time::timeout(std::time::Duration::from_secs(5), stream.next())
                .await
                .expect("must receive message within timeout")
                .expect("stream must not be empty");

            let status = msg.expect_err("message must be an error for empty session_id");
            assert_eq!(
                status.code(),
                tonic::Code::InvalidArgument,
                "empty session_id must return INVALID_ARGUMENT via stream, got: {}",
                status.code()
            );
        }
    }
}

// ── b.39x PM follow-up: validator wiring (session_id + user_application) ──
//
// The b.39x regression risk is structural: each of the 7 untrusted-string
// ingress handlers must call `validate_session_id` / `validate_user_application`
// before the value reaches storage, auth, tracing, or `presence_user` HSET.
// A future refactor that drops one of those calls would slip past the
// unit tests in `validation.rs` entirely — those test the validators in
// isolation, not the handler wiring.
//
// These integration tests pin the wiring on two representative ingress
// shapes: `handle_join_session` (bidi-stream, exercises both
// `validate_session_id` and `validate_user_application`) and
// `handle_create_layer` (client-stream, exercises `validate_session_id`).
// A NUL byte / >256-byte `session_id` would slip past the pre-fix
// `is_empty()` check; a control-char `user_application` would slip past
// the pre-fix "no validation at all" pipeline.
//
// `handle_join_session` may surface the InvalidArgument either as the
// top-level handler Err or as the first stream message — we accept either
// shape, mirroring the empty-session_id test above.

/// Helper: assert the handler returned InvalidArgument either at the
/// handler level or via the first message of the response stream.
async fn assert_join_invalid_argument(
    svc: &live_edit::grpc::service::LiveEditService,
    stream: Streaming<JoinSessionRequest>,
    case_label: &str,
) {
    let request = make_join_request(stream);
    let result = handle_join_session(svc, request).await;
    match result {
        Err(status) => {
            assert_eq!(
                status.code(),
                tonic::Code::InvalidArgument,
                "{case_label}: must return INVALID_ARGUMENT, got: {}",
                status.code()
            );
        }
        Ok(response) => {
            let mut stream = response.into_inner();
            let msg = tokio::time::timeout(std::time::Duration::from_secs(5), stream.next())
                .await
                .expect("must receive message within timeout")
                .expect("stream must not be empty");
            let status = msg.expect_err(case_label);
            assert_eq!(
                status.code(),
                tonic::Code::InvalidArgument,
                "{case_label}: must return INVALID_ARGUMENT via stream, got: {}",
                status.code()
            );
        }
    }
}

// b.39x PM-follow-up: NUL byte in `session_id` must reach the validator
// (`validate_session_id`) before the handler hits storage. Pre-fix the
// `is_empty()` shortcut accepted any non-empty string; a stray NUL would
// have flowed into Valkey key construction and tracing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn join_session_session_id_with_nul_byte_returns_invalid_argument() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let stream = streaming_join(vec![make_join_header("abc\0def", "test-app")]);
    assert_join_invalid_argument(&svc, stream, "session_id with NUL byte").await;
}

// b.39x PM-follow-up: `session_id` >256 bytes (`MAX_SESSION_ID_BYTES`)
// must be rejected by `validate_session_id`. The validator's length cap
// would never be reached if a future refactor dropped the
// `validate_session_id` call from `handle_join_session`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn join_session_session_id_over_length_cap_returns_invalid_argument() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    // 257 bytes — one over the 256-byte cap.
    let oversized = "a".repeat(257);
    let stream = streaming_join(vec![make_join_header(&oversized, "test-app")]);
    assert_join_invalid_argument(&svc, stream, "session_id over 256-byte cap").await;
}

// b.39x PM-follow-up: control-character `user_application` must be
// rejected by `validate_user_application` before it reaches the
// `presence_user` HSET or the `PeerJoined` broadcast. The pre-fix handler
// had NO `user_application` validation at all, so a control byte would
// have poisoned every peer's view.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn join_session_user_application_with_control_char_returns_invalid_argument() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    // Use a real session_id so the test is exercising user_application
    // validation specifically — not session_id validation.
    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = session_store
        .create("user-app-control-char-session", "/stage", "test-user")
        .await
        .expect("session creation must succeed");

    // ANSI escape (0x1B) in user_application — a control character, must
    // be rejected by the validator.
    let stream = streaming_join(vec![make_join_header(&session_id, "Bad\x1b[31mApp")]);
    assert_join_invalid_argument(&svc, stream, "user_application with control char").await;
}

// b.39x PM-follow-up: streaming ingress shape — `handle_create_layer`
// must run `validate_session_id` before the storage / accumulation
// pipeline. A NUL byte in the session_id would slip past the pre-fix
// `is_empty()` check and corrupt Valkey key construction. Distinct from
// the existing `create_layer_invalid_session_id_rejected_before_body_buffering`
// test (which uses an empty session_id and tests the buffer-ordering
// guarantee from b.y2t).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_session_id_with_nul_byte_returns_invalid_argument() {
    use live_edit::grpc::layer::handle_create_layer;

    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let stream = streaming_create(vec![make_create_header("abc\0def", "some-layer")]);
    let request = make_request(stream);

    let status = handle_create_layer(&svc, request)
        .await
        .expect_err("session_id with NUL must fail");

    assert_eq!(
        status.code(),
        tonic::Code::InvalidArgument,
        "session_id with NUL byte must return INVALID_ARGUMENT, got: {} — msg: {}",
        status.code(),
        status.message()
    );
}

// ── 5. Empty stream (no header) → INVALID_ARGUMENT ──────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn join_session_empty_stream_returns_invalid_argument() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    // Send no messages at all.
    let stream = streaming_join(vec![]);
    let request = make_join_request(stream);

    let result = handle_join_session(&svc, request).await;

    match result {
        Err(status) => {
            assert_eq!(
                status.code(),
                tonic::Code::InvalidArgument,
                "empty stream must return INVALID_ARGUMENT, got: {}",
                status.code()
            );
        }
        Ok(response) => {
            let mut stream = response.into_inner();
            let msg = tokio::time::timeout(std::time::Duration::from_secs(5), stream.next())
                .await
                .expect("must receive message within timeout")
                .expect("stream must not be empty");

            let status = msg.expect_err("message must be an error for empty stream");
            assert_eq!(
                status.code(),
                tonic::Code::InvalidArgument,
                "empty stream must return INVALID_ARGUMENT via stream, got: {}",
                status.code()
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SR-2.x: JoinSession through a real JwtUserinfoClient (b.g3z t1 Task 4)
// ═══════════════════════════════════════════════════════════════════════════════
//
// The JWT happy-path and opaque-bearer tests pin SR-5.4 for the
// bidirectional-streaming JoinSession RPC. Between them sit the JWT
// Auth0-subject test, the two userinfo-not-found tests, which reach the same
// claim chain through the fallback decorator, and the raw-payload test, which
// publishes literal `joined` payloads instead of having a peer join, and
// receives them on its own JoinSession (SDD §13.1).
// `handle_join_session` resolves the bearer's display identity via
// `userinfo_client.get_current_user` AFTER permissions has accepted the token,
// and the join's register script then publishes a `PeerJoined` event whose
// `user_name` is the resolved claim.
//
// The happy-path, Auth0-subject and both userinfo-not-found tests share
// `join_and_observe_peer_joined`, which observes the broadcast PeerJoined
// locally via the presence router (the "simpler shape" called out in the
// subtask): it pre-subscribes to the router so the entry exists, runs the
// Valkey bridge task so events arrive end-to-end, manually `SUBSCRIBE`s on
// Valkey (the bidi handler's own `subscribe_presence_channel` would only fire
// if it were the first router subscriber, which it isn't once the helper has
// subscribed first), then drives a real JoinSession with the test's bearer and
// returns the PeerJoined's user. The happy-path bearer is a JWT carrying
// `name = "Ada Lovelace"`, and the test asserts that claim flows through.
//
// The opaque-bearer test mirrors the `assert_join_invalid_argument` dual-shape
// pattern but for `tonic::Code::Internal` and the verbatim SR-2.4 substring
// (kept in lock-step with the engineer's landing in `src/auth/userinfo_jwt.rs`).

/// JoinSession wrap-request variant that lets the caller supply the bearer
/// token directly. The existing `make_join_request` always sets
/// `"test-token"`, which is fine for the mock-userinfo tests above but
/// would be rejected by a real `JwtUserinfoClient` (not a 3-segment JWT).
fn make_join_request_with_bearer(
    stream: Streaming<JoinSessionRequest>,
    bearer: &str,
) -> Request<Streaming<JoinSessionRequest>> {
    let mut request = Request::new(stream);
    request.extensions_mut().insert(BearerToken::new(bearer));
    request
}

/// Create a session named `session_name`, drive a real JoinSession into it
/// with `bearer` and the application `test-app`, and return the user of the
/// `PeerJoined` event the join broadcasts, as a peer's replica parses it off
/// the presence pub/sub wire.
///
/// Observes the broadcast locally via the presence router: pre-subscribe to
/// the router so the presence receiver (and the router entry
/// `spawn_presence_subscriber` checks via `has_session`) exists before the
/// join's register publishes; run the Valkey -> router bridge task so events
/// published to `le:presence:events:{sid}` arrive end-to-end; and `SUBSCRIBE`
/// on Valkey by hand, because the bidi handler only calls
/// `subscribe_presence_channel` when it is the first router subscriber, which
/// it is not once this helper has subscribed. Panics if the join is refused,
/// if its first response is not SessionConnected, or if no PeerJoined arrives
/// within the timeout.
async fn join_and_observe_peer_joined(
    svc: &live_edit::grpc::service::LiveEditService,
    tv: &TestValkey,
    session_name: &str,
    bearer: &str,
) -> live_edit::presence::ConnectedUser {
    use live_edit::presence::{
        spawn_presence_subscriber, subscribe_presence_channel, PresenceEvent,
    };

    // Pre-create the session and install the CancellationToken (mirrors the
    // existing `join_session_happy_path_returns_session_connected` setup).
    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = session_store
        .create(session_name, "/stage", "test-user")
        .await
        .expect("session creation must succeed");
    svc.session_tokens.insert(
        session_id.clone(),
        tokio_util::sync::CancellationToken::new().into(),
    );

    let (mut event_rx, _first) = svc.presence_router.subscribe(&session_id);
    let presence_cancel = tokio_util::sync::CancellationToken::new();
    let subscriber_handle = spawn_presence_subscriber(
        tv.client().subscriber.clone(),
        Arc::clone(&svc.presence_router),
        tv.key_schema().clone(),
        Arc::clone(&svc.metrics),
        presence_cancel.clone(),
    );
    subscribe_presence_channel(&tv.client().subscriber, tv.key_schema(), &session_id)
        .await
        .expect("SUBSCRIBE on presence events channel must succeed");

    let stream = streaming_join(vec![make_join_header(&session_id, "test-app")]);
    let response = handle_join_session(svc, make_join_request_with_bearer(stream, bearer))
        .await
        .expect("JoinSession must succeed for an authorized caller");
    let mut response_stream = response.into_inner();

    // Drain at least the SessionConnected message so the bidi loop has
    // definitely reached the PUBLISH step.
    let first_msg = tokio::time::timeout(std::time::Duration::from_secs(5), response_stream.next())
        .await
        .expect("must receive SessionConnected within timeout")
        .expect("stream must not be empty")
        .expect("first response must not be an error");
    match first_msg.response_type {
        Some(join_session_response::ResponseType::Connected(_)) => {}
        other => panic!("expected SessionConnected, got {other:?}"),
    }

    // The PeerJoined event for this client's own connection must arrive on
    // the locally-subscribed receiver. Walk until we see a PeerJoined — the
    // bridge task may interleave PeerLeft / PeerMessage cousins for prior
    // tests in unlikely cases, but the cleanest assertion is just to find
    // the first PeerJoined.
    let user = tokio::time::timeout(std::time::Duration::from_secs(5), async {
        loop {
            match event_rx.recv().await.as_deref() {
                Ok(PresenceEvent::PeerJoined(user)) => return user.clone(),
                Ok(_) => continue,
                Err(e) => panic!("router receiver closed unexpectedly: {e:?}"),
            }
        }
    })
    .await
    .expect("must receive PeerJoined within timeout");

    // Teardown: cancel the bridge so it doesn't keep churning after the test.
    presence_cancel.cancel();
    let _ = tokio::time::timeout(std::time::Duration::from_secs(5), subscriber_handle).await;

    user
}

// ── JWT happy path: PeerJoined.user_name carries the resolved display claim ──

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn join_session_jwt_happy_path_peer_joined_carries_name_claim() {
    let tv = TestValkey::setup().await;
    let userinfo = Arc::new(live_edit::auth::JwtUserinfoClient::new());
    let svc = build_service_with_userinfo(&tv, Arc::new(AllowAllPermissions), userinfo);

    // A real JoinSession with a JWT bearer carrying `name = "Ada Lovelace"`.
    let jwt = make_jwt(&serde_json::json!({ "name": "Ada Lovelace" }));
    let user = join_and_observe_peer_joined(&svc, &tv, "join-jwt-happy", &jwt).await;

    assert_eq!(
        user.user_name, "Ada Lovelace",
        "JWT `name` claim must flow through to PeerJoined.user_name, got: {user:?}",
    );
}

// ── Auth0 subject: a `|` in the name reaches live peers whole (SDD §13.1) ────
//
// A JWT-only deployment labels a token without a human-name claim by its
// `sub`, and an Auth0 subject is `provider|id`. The presence wire escapes the
// `|`, so the peer's replica parses the whole subject as the name and the
// application stays its own field.

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn join_session_jwt_auth0_subject_peer_joined_carries_the_whole_subject_and_app() {
    let tv = TestValkey::setup().await;
    let userinfo = Arc::new(live_edit::auth::JwtUserinfoClient::new());
    let svc = build_service_with_userinfo(&tv, Arc::new(AllowAllPermissions), userinfo);

    let jwt = make_jwt(&serde_json::json!({ "sub": "auth0|5f8a1b2c" }));
    let user = join_and_observe_peer_joined(&svc, &tv, "join-jwt-auth0", &jwt).await;

    assert_eq!(
        (user.user_name.as_str(), user.user_application.as_str()),
        ("auth0|5f8a1b2c", "test-app"),
        "the PeerJoined a peer parses must carry the registered name and application"
    );
}

// ── Userinfo "no such user": PeerJoined.user_name carries the app label ─────
//
// With the userinfo service and authorization both on, an authorized app
// identity the userinfo directory does not hold used to fail JoinSession
// `INTERNAL`. The userinfo client `build_userinfo_client` builds for that
// deployment answers from the bearer's claims; the ` (app)` label must reach
// `PeerJoined.user_name` across the presence pub/sub wire intact. Same
// observation shape as the JWT happy path above.

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn join_session_userinfo_not_found_peer_joined_carries_jwt_app_label() {
    let tv = TestValkey::setup().await;
    let (userinfo, _userinfo_server) = userinfo_answering_not_found().await;
    let svc = build_service_with_userinfo(&tv, Arc::new(AllowAllPermissions), userinfo);

    let jwt = unlisted_app_jwt("Asset Agent");
    let user = join_and_observe_peer_joined(&svc, &tv, "join-userinfo-not-found", &jwt).await;

    assert_eq!(user.user_name, "Asset Agent (app)");
}

/// An app display name holding `|` is used as the label, and the label with
/// its ` (app)` marker reaches the peer whole: the wire escapes the `|`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn join_session_userinfo_not_found_app_label_with_a_pipe_reaches_peers_whole() {
    let tv = TestValkey::setup().await;
    let (userinfo, _userinfo_server) = userinfo_answering_not_found().await;
    let svc = build_service_with_userinfo(&tv, Arc::new(AllowAllPermissions), userinfo);

    let jwt = unlisted_app_jwt("Asset|Agent");
    let user = join_and_observe_peer_joined(&svc, &tv, "join-userinfo-pipe-label", &jwt).await;

    assert_eq!(
        (user.user_name.as_str(), user.user_application.as_str()),
        ("Asset|Agent (app)", "test-app")
    );
}

// ── Raw `joined` payloads on the wire (SDD §13.1) ────────────────────────────
//
// Literal payloads published on the session's presence channel, as a newer
// and an older replica write them, must reach a connected JoinSession client
// as exactly the fields the publishing replica registered: a newer replica's
// `%7C` / `%25` escapes are decoded, and an older replica's raw name with an
// extra `|` keeps the older cut (the rest stays in the application).

/// Each raw payload published, and the `(connection_id, user_name,
/// user_application)` the client must receive for it.
const RAW_JOINED_CASES: [(&str, (&str, &str, &str)); 2] = [
    (
        "joined|ghost-new|auth0%7C5f8a%25x|Kit%7COmniverse",
        ("ghost-new", "auth0|5f8a%x", "Kit|Omniverse"),
    ),
    (
        "joined|ghost-old|Grace Hopper|Kit|Omniverse",
        ("ghost-old", "Grace Hopper", "Kit|Omniverse"),
    ),
];

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn join_session_raw_joined_payloads_reach_the_client_as_the_registered_fields() {
    use fred::interfaces::PubsubInterface;
    use live_edit::presence::spawn_presence_subscriber;

    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let presence_cancel = tokio_util::sync::CancellationToken::new();
    let subscriber_handle = spawn_presence_subscriber(
        tv.client().subscriber.clone(),
        Arc::clone(&svc.presence_router),
        tv.key_schema().clone(),
        Arc::clone(&svc.metrics),
        presence_cancel.clone(),
    );
    let (session_id, _token) = create_session(&tv, &svc, "join-raw-joined", true).await;

    // The only local join is the router's first subscriber, so its handler
    // SUBSCRIBEs on Valkey, and SessionConnected follows that subscribe.
    let (mut stream, tx) = open_join(&svc, &session_id, "app-a").await;
    expect_session_connected(&mut stream, std::time::Duration::from_secs(5)).await;

    let channel = tv.key_schema().presence_events_channel(&session_id);
    for (payload, _) in RAW_JOINED_CASES {
        tv.client()
            .pool
            .next()
            .publish::<(), _, _>(&channel, payload)
            .await
            .expect("PUBLISH must succeed");
    }

    // Collect the injected peers' PeerJoined messages, by connection id, under
    // one deadline for the whole drain. A payload the replica drops shows as
    // a missing entry in the assertions below.
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
    let mut received = std::collections::BTreeMap::new();
    while received.len() < RAW_JOINED_CASES.len() {
        let Ok(next) = tokio::time::timeout_at(deadline, stream.next()).await else {
            break;
        };
        let msg = next
            .expect("the join stream must stay open")
            .expect("the join stream must not fail");
        if let Some(join_session_response::ResponseType::PeerJoined(pj)) = msg.response_type {
            if pj.connection_id.starts_with("ghost-") {
                received.insert(pj.connection_id, (pj.user_name, pj.user_application));
            }
        }
    }

    for (payload, (connection_id, user_name, user_application)) in RAW_JOINED_CASES {
        assert_eq!(
            received
                .get(connection_id)
                .map(|(n, a)| (n.as_str(), a.as_str())),
            Some((user_name, user_application)),
            "published {payload:?}"
        );
    }

    drop(tx);
    presence_cancel.cancel();
    let _ = tokio::time::timeout(std::time::Duration::from_secs(5), subscriber_handle).await;
}

// ── Opaque-bearer rejection: redacted INTERNAL wire shape + AuthError pin ───
//
// Status-message contract — important. SR-2.4 pins the verbatim wording on
// the `AuthError::InvalidResponse(msg)` boundary that the production
// `JwtUserinfoClient::get_current_user` returns; that string flows through
// `ServiceError::Internal(format!("auth backend: {e}"))` and lands in the
// operator's `tracing::error!(error = %msg, ...)` log. The
// `ServiceError → tonic::Status` impl deliberately redacts the wire message
// to a generic `"internal error"` (`src/error.rs` line ~206) so backend
// detail does not leak to clients. This is a security contract, not a bug —
// see the `ServiceError::Internal` arm and its
// `internal_status_does_not_leak_message` regression test.
//
// The verbatim-wording check therefore lives in two places:
//   1. `src/auth/userinfo_jwt.rs::tests` pins the `AuthError` boundary.
//   2. `SR_2_4_OPAQUE_TOKEN_MSG` in `super::helpers` is a shared `pub const`
//      referenced from this file (and the sibling SR-2.x integration tests),
//      so a Task 2 reword surfaces as a compile-time drift signal even though
//      the wire never sees the string.
//
// What we CAN assert from the integration-test position: the redacted wire
// shape (`INTERNAL` + `"internal error"`). Both halves of the contract are
// exercised; the verbatim-wire-string check the Subtask body originally
// asked for is structurally impossible past the security redaction and is
// left to the unit-level pin (mirrored in this file via the cross-file
// `JwtUserinfoClient` call below).

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn join_session_opaque_bearer_returns_internal_with_sr_2_4_message() {
    let tv = TestValkey::setup().await;
    let userinfo = Arc::new(live_edit::auth::JwtUserinfoClient::new());
    let svc = build_service_with_userinfo(&tv, Arc::new(AllowAllPermissions), userinfo);

    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = session_store
        .create("join-opaque", "/stage", "test-user")
        .await
        .expect("session creation must succeed");
    svc.session_tokens.insert(
        session_id.clone(),
        tokio_util::sync::CancellationToken::new().into(),
    );

    let stream = streaming_join(vec![make_join_header(&session_id, "test-app")]);
    let request = make_join_request_with_bearer(stream, "opaque-token-not-a-jwt");

    // `handle_join_session` surfaces userinfo errors either at the handler
    // return (when the userinfo call runs to completion before the response
    // stream is constructed) or as the first stream message (when the bidi
    // task converts the late error). Mirror `assert_join_invalid_argument`'s
    // dual-shape pattern — but assert `Internal` + the security-redacted
    // wire message (NOT the verbatim SR-2.4 string, which is redacted off
    // the wire by `ServiceError::Internal → tonic::Status` mapping).
    let result = handle_join_session(&svc, request).await;
    match result {
        Err(status) => {
            assert_eq!(
                status.code(),
                tonic::Code::Internal,
                "opaque bearer must return INTERNAL, got: {} — msg: {}",
                status.code(),
                status.message(),
            );
            assert_eq!(
                status.message(),
                "internal error",
                "wire-side message must remain the security-redacted placeholder; \
                 backend detail (incl. SR-2.4 wording {SR_2_4_OPAQUE_TOKEN_MSG:?}) \
                 must never reach the client. Got: {:?}",
                status.message(),
            );
        }
        Ok(response) => {
            let mut stream = response.into_inner();
            let msg = tokio::time::timeout(std::time::Duration::from_secs(5), stream.next())
                .await
                .expect("must receive message within timeout")
                .expect("stream must not be empty");

            let status = msg.expect_err("first stream message must be an error for opaque bearer");
            assert_eq!(
                status.code(),
                tonic::Code::Internal,
                "opaque bearer must return INTERNAL via stream, got: {} — msg: {}",
                status.code(),
                status.message(),
            );
            assert_eq!(
                status.message(),
                "internal error",
                "stream-side message must remain the security-redacted placeholder; \
                 backend detail (incl. SR-2.4 wording {SR_2_4_OPAQUE_TOKEN_MSG:?}) \
                 must never reach the client. Got: {:?}",
                status.message(),
            );
        }
    }

    // Cross-file pin: also exercise the production decoder directly with the
    // same opaque bearer to confirm the underlying `AuthError` carries the
    // verbatim SR-2.4 wording. This is what makes `SR_2_4_OPAQUE_TOKEN_MSG`
    // load-bearing — a reword in `JwtUserinfoClient::get_current_user`
    // surfaces here even though the gRPC wire path redacts the string.
    let decoder = live_edit::auth::JwtUserinfoClient::new();
    let auth_err = decoder
        .get_current_user("opaque-token-not-a-jwt")
        .await
        .expect_err("opaque bearer must return AuthError at the decoder boundary");
    match auth_err {
        live_edit::auth::AuthError::InvalidResponse(msg) => {
            assert_eq!(
                msg, SR_2_4_OPAQUE_TOKEN_MSG,
                "JwtUserinfoClient must return the verbatim SR-2.4 wording at the \
                 AuthError boundary — Task 2 contract. Got: {msg:?}",
            );
        }
        other => panic!("expected AuthError::InvalidResponse with SR-2.4 wording, got {other:?}"),
    }
}

// ── b.27b: an escaped `Aborted` ends the join stream UNAVAILABLE ────────────
//
// After registering, and before SessionConnected, the spawned join task
// re-checks the session in `acquire_session_cancel`
// (`SessionStore::get_base_path`). In production that
// read maps a transient Valkey failure to `ServiceError::Unavailable` itself,
// so the `ServiceError::Aborted` injected here is synthetic: it stands in for
// any in-process retry signal that escapes onto the join stream. The test
// guards that such an `Aborted` reaches the client as `UNAVAILABLE`, which the
// Kit client retries, and never as ABORTED, which it treats as terminal.
//
// The handler's own step-2 lookup goes through the base-path cache, so the
// test warms that cache first; the queued fault is then consumed by the
// in-task call, which reads the store directly. The handler returning `Ok`
// proves the fault did not fire in the handler.
// Feature-gated: without `--features test-fault-injection` this test is
// compiled out.

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn join_session_transient_in_task_session_check_ends_the_stream_unavailable() {
    use std::collections::VecDeque;
    use std::sync::Mutex;

    use live_edit::error::ServiceError;

    // One deadline for the whole drain, however many items precede the status.
    const DRAIN_DEADLINE: std::time::Duration = std::time::Duration::from_secs(30);
    const INJECTED: &str =
        "transient I/O: HGET session base_path failed (Timeout): injected in-task blip";

    let tv = TestValkey::setup().await;
    let fault_queue: Arc<Mutex<VecDeque<ServiceError>>> = Arc::new(Mutex::new(VecDeque::new()));
    let base = build_service(&tv, Arc::new(AllowAllPermissions));
    let svc = live_edit::grpc::service::LiveEditService {
        session_store: Arc::new(
            SessionStore::new(tv.client().clone(), tv.key_schema().clone())
                .with_get_base_path_fault_injector_for_test(fault_queue.clone()),
        ),
        ..base
    };
    let session_id = svc
        .session_store
        .create("join-transient", "/stage", "test-user")
        .await
        .expect("session creation must succeed");

    // Warm the handler's base-path cache so its step-2 lookup cannot consume
    // the fault queued next.
    svc.session_base_path_cache
        .get_base_path(&svc.session_store, &session_id)
        .await
        .expect("warming the base-path cache must succeed");
    fault_queue
        .lock()
        .unwrap()
        .push_back(ServiceError::Aborted(INJECTED.to_owned()));

    let request = make_join_request(streaming_join(vec![make_join_header(
        &session_id,
        "test-app",
    )]));
    let mut response_stream = handle_join_session(&svc, request)
        .await
        .expect("handler must return Ok — the fault fires inside the spawned join task")
        .into_inner();

    let (delivered, status) = tokio::time::timeout(DRAIN_DEADLINE, async {
        let mut delivered = Vec::new();
        loop {
            match response_stream.next().await {
                Some(Ok(msg)) => delivered.push(msg),
                Some(Err(status)) => {
                    assert!(
                        response_stream.next().await.is_none(),
                        "the error status must be the stream's terminal item"
                    );
                    return (delivered, status);
                }
                None => panic!(
                    "the join stream ended with no status after {} message(s): a bare EOS is \
                     gRPC OK, which the Kit client does not retry",
                    delivered.len()
                ),
            }
        }
    })
    .await
    .expect("the join stream neither terminated nor errored within the drain deadline");

    assert!(
        delivered.is_empty(),
        "the session re-check precedes SessionConnected, so the failure must be the stream's \
         first item; got: {delivered:?}"
    );
    assert_eq!(
        status.code(),
        tonic::Code::Unavailable,
        "a transient in-task failure must end the join stream UNAVAILABLE so the Kit client \
         retries (it treats ABORTED as terminal); got {} ({})",
        status.code(),
        status.message()
    );
    assert_eq!(
        status.message(),
        INJECTED,
        "the status must carry the injected failure verbatim, proving it came from the \
         in-task session check"
    );
    assert!(
        fault_queue.lock().unwrap().is_empty(),
        "the queued fault must have been consumed"
    );
}
