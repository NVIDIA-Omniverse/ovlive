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

// Integration tests for the CreateLayer handler's auth, parsing and
// external-value pipeline, and for the layer's root section it supplies to
// content that leaves the layer empty (SDD §9.7).
//
// Tests call `handle_create_layer` directly with a LiveEditService backed by
// a real Valkey (TestValkey fixture), mock auth clients, and gRPC-framed
// streaming requests.

use std::sync::Arc;

use fred::interfaces::KeysInterface;
use futures::StreamExt;
use live_edit::auth::userinfo::UserinfoClient;
use live_edit::auth::JwtUserinfoClient;
use live_edit::domain::ompf::OmpfContainer;
use live_edit::domain::types::ObjectState;
use live_edit::grpc::layer::{handle_create_layer, handle_stat_layer, handle_update_layer};
use live_edit::grpc::proto::{
    subscribe_read_layer_response, StatLayerRequest, SubscribeReadLayerHeader,
    SubscribeReadLayerRequest,
};
use live_edit::grpc::service::LiveEditService;
use live_edit::grpc::subscribe::handle_subscribe_read_layer;
use live_edit::grpc::BearerToken;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;
use oodf_ffi::{CommandDiffChangeSection, OwnedDelta, StructCommand};
use serde_json::json;
use tonic::Request;

use crate::common::owned_delta::{delta_to_ompf_bytes, empty_delta, with_root};

use super::helpers::*;

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

// ── 1. Happy path with delta content ─────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_happy_path_with_delta_returns_object_id() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    // Create a session so get_base_path succeeds.
    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = session_store
        .create("test-session", "/stage", "test-user")
        .await
        .expect("session creation must succeed");

    // Build valid OMPF bytes containing an OODF delta.
    let ompf_bytes = delta_to_ompf_bytes(&empty_delta(), &[]);

    let stream = streaming_create(vec![
        make_create_header(&session_id, "my-layer"),
        make_create_data(ompf_bytes),
    ]);
    let request = make_request(stream);

    let response = handle_create_layer(&svc, request)
        .await
        .expect("happy path must succeed");

    // The handler returns an object_id in the response.
    // Since this is a fresh ObjectState, object_id is 0 (placeholder).
    let resp = response.into_inner();
    // Just verify we got a response without error; the exact object_id value
    // depends on the delta engine's behavior for a fresh state.
    assert!(
        resp.object_id > 0,
        "object_id must be positive (allocated via Valkey INCR)"
    );
}

// ── 2. Empty content creates the layer holding only its root (SDD §9.7) ─────

/// A header with no chunks, and an empty delta, each create the layer holding
/// only its root section: a new subscriber's catch-up message #1, at sequence
/// 1, carries exactly `DiffChangeSection(parent 0, section 1, "", type 7,
/// order 0)`, and the stored object holds that one row.
///
/// The regression test for a header-only create Kit could not open: before
/// the layer-root rule the layer was created with no section at all, and
/// message #1 carried no structural command.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_empty_content_creates_a_layer_holding_only_its_root_section() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = session_store
        .create("empty-layer-session", "/stage", "test-user")
        .await
        .expect("session creation must succeed");

    let cases = [
        ("header only", "header-only-layer", None),
        (
            "empty delta",
            "empty-delta-layer",
            Some(delta_to_ompf_bytes(&empty_delta(), &[])),
        ),
    ];
    for (case, layer, body) in cases {
        let mut messages = vec![make_create_header(&session_id, layer)];
        messages.extend(body.map(make_create_data));
        let object_id = handle_create_layer(&svc, make_request(streaming_create(messages)))
            .await
            .unwrap_or_else(|s| panic!("{case}: empty content must create the layer: {s:?}"))
            .into_inner()
            .object_id;

        let (header, delta) = catch_up_first_message(&svc, &session_id, layer).await;
        assert_eq!(
            (header.object_id, header.sequence_num),
            (object_id, 1),
            "{case}: message #1 is the created object at sequence 1"
        );
        assert_eq!(
            delta.struct_commands,
            vec![supplied_root()],
            "{case}: message #1 must carry exactly the layer's root section"
        );
        assert!(
            delta.set_fields.is_empty() && delta.time_samples.is_empty(),
            "{case}: the root holds no field or time sample"
        );

        let stored = stored_state(&tv, object_id).await;
        assert_eq!(
            stored
                .sections
                .values()
                .map(|s| (
                    s.section_id,
                    s.parent_id,
                    s.name.as_str(),
                    s.section_type,
                    s.section_order
                ))
                .collect::<Vec<_>>(),
            [(1, 0, "", 7, 0)],
            "{case}: the stored object holds only the root row"
        );
    }
}

// ── 3. Malformed OMPF → INVALID_ARGUMENT ────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_malformed_ompf_returns_invalid_argument() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = session_store
        .create("malformed-session", "/stage", "test-user")
        .await
        .expect("session creation must succeed");

    // Random bytes — not valid OMPF.
    let garbage = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03, 0x04];

    let stream = streaming_create(vec![
        make_create_header(&session_id, "bad-layer"),
        make_create_data(garbage),
    ]);
    let request = make_request(stream);

    let status = handle_create_layer(&svc, request)
        .await
        .expect_err("malformed OMPF must fail");

    assert_eq!(
        status.code(),
        tonic::Code::InvalidArgument,
        "malformed OMPF must return INVALID_ARGUMENT, got: {}",
        status.code()
    );
    assert!(
        status.message().contains("OMPF") || status.message().contains("malformed"),
        "error message should mention OMPF or malformed, got: {}",
        status.message()
    );
}

// ── 4. Session not found → NOT_FOUND ────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_session_not_found_returns_not_found() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    // Use a non-existent session_id — no session was created in Valkey.
    let stream = streaming_create(vec![make_create_header(
        "nonexistent-session-id",
        "some-layer",
    )]);
    let request = make_request(stream);

    let status = handle_create_layer(&svc, request)
        .await
        .expect_err("non-existent session must fail");

    assert_eq!(
        status.code(),
        tonic::Code::NotFound,
        "non-existent session must return NOT_FOUND, got: {}",
        status.code()
    );
}

// ── 5. Invalid layer name → INVALID_ARGUMENT ────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_invalid_layer_name_returns_invalid_argument() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = session_store
        .create("name-validation-session", "/stage", "test-user")
        .await
        .expect("session creation must succeed");

    // Prohibited layer name: path traversal with ".."
    let stream = streaming_create(vec![make_create_header(&session_id, "../../etc/passwd")]);
    let request = make_request(stream);

    let status = handle_create_layer(&svc, request)
        .await
        .expect_err("invalid layer name must fail");

    assert_eq!(
        status.code(),
        tonic::Code::InvalidArgument,
        "invalid layer name must return INVALID_ARGUMENT, got: {}",
        status.code()
    );
}

// ── 6. Auth denied → PERMISSION_DENIED ──────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_auth_denied_returns_permission_denied() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(DenyAllPermissions));

    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = session_store
        .create("auth-denied-session", "/stage", "test-user")
        .await
        .expect("session creation must succeed");

    let stream = streaming_create(vec![make_create_header(&session_id, "some-layer")]);
    let request = make_request(stream);

    let status = handle_create_layer(&svc, request)
        .await
        .expect_err("denied auth must fail");

    assert_eq!(
        status.code(),
        tonic::Code::PermissionDenied,
        "denied auth must return PERMISSION_DENIED, got: {}",
        status.code()
    );
}

// ── 7. b.y2t — invalid header rejected BEFORE buffering body ────────────────
//
// DoS amplification regression: an invalid session_id or layer_name in the
// header must be rejected before the server buffers `max_layer_size`
// bytes of attacker-chosen junk. The service is built with a small
// `max_layer_size` (`helpers::SMALL_SIZE_CAP`) and we send a body
// one byte over it after an invalid header.
//
// A buffer-first ingest would run the buffering loop, trip the size cap,
// and refuse on that — the header never inspected. Validating the header
// before any chunk is buffered makes the header's own refusal win.
//
// Both refusals answer INVALID_ARGUMENT (the size cap stopped answering the
// retried RESOURCE_EXHAUSTED in b.75i), so the CODE cannot tell them apart.
// The MESSAGE can: each test pins the header validator's message and asserts
// the size cap's is absent (`helpers::assert_header_refused_before_buffering`).
//
// The cap is small on purpose. Against the default 10 MB cap the over-cap body
// exceeds tonic's 4 MiB per-message decode limit, so a buffer-first
// implementation is stopped by the decoder (CANCELLED) before the size cap ever
// runs, and the scenario this section names is never exercised.

/// Body one byte over the test service's `max_layer_size`, and well under the
/// decoder's limit, so only the header check stands between it and the size
/// cap.
const OVER_LIMIT_CHUNK_SIZE: usize = SMALL_SIZE_CAP + 1;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_invalid_session_id_rejected_before_body_buffering() {
    let tv = TestValkey::setup().await;
    let svc = build_service_with_size_cap(&tv, Arc::new(AllowAllPermissions), SMALL_SIZE_CAP);

    // Invalid header: empty session_id. Body is larger than max_layer_size,
    // so a buffer-first implementation would refuse on the size cap instead.
    let stream = streaming_create(vec![
        make_create_header("", "some-layer"),
        make_create_data(vec![0xAB; OVER_LIMIT_CHUNK_SIZE]),
    ]);
    let request = make_request(stream);

    let status = handle_create_layer(&svc, request)
        .await
        .expect_err("invalid session_id must fail");

    assert_header_refused_before_buffering(&status, "session_id must not be empty");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_invalid_layer_name_rejected_before_body_buffering() {
    let tv = TestValkey::setup().await;
    let svc = build_service_with_size_cap(&tv, Arc::new(AllowAllPermissions), SMALL_SIZE_CAP);

    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = session_store
        .create("early-reject-session", "/stage", "test-user")
        .await
        .expect("session creation must succeed");

    // Invalid header: layer_name with path-traversal. Body > max_layer_size.
    let stream = streaming_create(vec![
        make_create_header(&session_id, "../../etc/passwd"),
        make_create_data(vec![0xAB; OVER_LIMIT_CHUNK_SIZE]),
    ]);
    let request = make_request(stream);

    let status = handle_create_layer(&svc, request)
        .await
        .expect_err("invalid layer_name must fail");

    assert_header_refused_before_buffering(&status, "..");
}

// ── 8. SR-2.x — JwtUserinfoClient end-to-end via CreateLayer ────────────────
//
// These tests pin the `userinfo-service-disabled` deployment shape (SR-2.x):
// when `LiveEditService` is wired with the real `JwtUserinfoClient`, the JWT
// payload's display claim flows verbatim into the layer's `created_by`
// (observed by `StatLayer` as `created_by_user_name`); and a non-JWT bearer
// surfaces as `INTERNAL` with the verbatim SR-2.4 wording. One test covers the
// same claim chain reached the other way: the userinfo service on, with
// authorization, answering "no such user", so the fallback decorator decodes
// the bearer.
//
// The verbatim SR-2.4 string lives in `super::helpers` as a shared `pub const`
// (`SR_2_4_OPAQUE_TOKEN_MSG`) — drift here is a Task 2 contract violation.

/// Build a `CreateLayerRequest` stream wrapped in a `tonic::Request` with the
/// caller-supplied bearer token. Distinct from the shared `make_request`
/// helper (which always inserts `"test-token"`) — these tests need to control
/// the bearer because the JWT decoder reads its payload from that exact
/// extension.
fn make_create_request_with_bearer(
    stream: AttestedStreaming<live_edit::grpc::proto::CreateLayerRequest>,
    bearer: &str,
) -> Request<tonic::codec::Streaming<live_edit::grpc::proto::CreateLayerRequest>> {
    let mut request = stream.into_handler_request();
    request.extensions_mut().insert(BearerToken::new(bearer));
    request
}

/// Build a `StatLayerRequest` carrying the supplied bearer. Lifted from the
/// pattern in `layer_management_test.rs` so the stat readback uses the same
/// bearer as the create call.
fn stat_request_with_bearer(
    session_id: &str,
    layer_name: &str,
    bearer: &str,
) -> Request<StatLayerRequest> {
    let mut request = Request::new(StatLayerRequest {
        session_id: session_id.to_owned(),
        layer_name: layer_name.to_owned(),
    });
    request.extensions_mut().insert(BearerToken::new(bearer));
    request
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_jwt_userinfo_happy_path_flows_name_claim_into_stat_response() {
    // Happy-path JWT: with a real `JwtUserinfoClient` injected, the `name`
    // claim must surface through `created_by_user_name` on the layer's stat
    // response — proves the decoded display name reaches the object header's
    // `created_by` field on `CreateLayer`.
    let tv = TestValkey::setup().await;
    let svc = build_service_with_userinfo(
        &tv,
        Arc::new(AllowAllPermissions),
        Arc::new(JwtUserinfoClient::new()),
    );

    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = session_store
        .create("jwt-create-happy", "/stage", "test-user")
        .await
        .expect("session creation must succeed");

    let jwt = make_jwt(&json!({ "name": "Grace Hopper" }));

    let ompf_bytes = delta_to_ompf_bytes(&empty_delta(), &[]);
    let stream = streaming_create(vec![
        make_create_header(&session_id, "jwt-happy-layer"),
        make_create_data(ompf_bytes),
    ]);
    let request = make_create_request_with_bearer(stream, &jwt);

    let response = handle_create_layer(&svc, request)
        .await
        .expect("CreateLayer with valid JWT must succeed");
    assert!(
        response.into_inner().object_id > 0,
        "object_id must be positive (allocated via Valkey INCR)"
    );

    // Stat the freshly-created layer and assert the JWT's display claim
    // flowed through to `created_by_user_name`. Use the same JWT bearer so
    // permissions/userinfo see consistent input across both calls.
    let stat = handle_stat_layer(
        &svc,
        stat_request_with_bearer(&session_id, "jwt-happy-layer", &jwt),
    )
    .await
    .expect("StatLayer on freshly-created layer must succeed")
    .into_inner();

    assert_eq!(
        stat.created_by_user_name, "Grace Hopper",
        "JWT `name` claim must flow through to created_by_user_name, got: {:?}",
        stat.created_by_user_name,
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_jwt_userinfo_preferred_username_fallback_flows_into_stat_response() {
    // Optional fallback-chain coverage: with `name` absent, the
    // `preferred_username` claim must win the fallback chain and surface as
    // `created_by_user_name`. Low-cost extra coverage per the Subtask body's
    // "encouraged but not required" note.
    let tv = TestValkey::setup().await;
    let svc = build_service_with_userinfo(
        &tv,
        Arc::new(AllowAllPermissions),
        Arc::new(JwtUserinfoClient::new()),
    );

    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = session_store
        .create("jwt-create-fallback", "/stage", "test-user")
        .await
        .expect("session creation must succeed");

    // `name` absent → fallback chain falls through to `preferred_username`.
    let jwt = make_jwt(&json!({
        "preferred_username": "grace@example.com",
        "sub": "should-not-win",
    }));

    let ompf_bytes = delta_to_ompf_bytes(&empty_delta(), &[]);
    let stream = streaming_create(vec![
        make_create_header(&session_id, "jwt-fallback-layer"),
        make_create_data(ompf_bytes),
    ]);
    let request = make_create_request_with_bearer(stream, &jwt);

    handle_create_layer(&svc, request)
        .await
        .expect("CreateLayer with preferred_username-only JWT must succeed");

    let stat = handle_stat_layer(
        &svc,
        stat_request_with_bearer(&session_id, "jwt-fallback-layer", &jwt),
    )
    .await
    .expect("StatLayer on freshly-created layer must succeed")
    .into_inner();

    assert_eq!(
        stat.created_by_user_name, "grace@example.com",
        "preferred_username fallback must win when `name` is absent, got: {:?}",
        stat.created_by_user_name,
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_userinfo_not_found_uses_jwt_app_label_for_created_by() {
    // With the userinfo service and authorization both on, an authorized app
    // identity the userinfo directory does not hold used to fail CreateLayer
    // `INTERNAL`. The userinfo client `build_userinfo_client` builds for that
    // deployment answers from the bearer's claims; the app label must reach
    // the layer's `created_by`, read back by `StatLayer`.
    let tv = TestValkey::setup().await;
    let (userinfo, _userinfo_server) = userinfo_answering_not_found().await;
    let svc = build_service_with_userinfo(&tv, Arc::new(AllowAllPermissions), userinfo);

    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = session_store
        .create("userinfo-not-found", "/stage", "test-user")
        .await
        .expect("session creation must succeed");

    let jwt = unlisted_app_jwt("Asset Agent");
    let stream = streaming_create(vec![
        make_create_header(&session_id, "app-layer"),
        make_create_data(delta_to_ompf_bytes(&empty_delta(), &[])),
    ]);
    handle_create_layer(&svc, make_create_request_with_bearer(stream, &jwt))
        .await
        .expect("CreateLayer must succeed for an authorized caller the directory does not hold");

    let stat = handle_stat_layer(
        &svc,
        stat_request_with_bearer(&session_id, "app-layer", &jwt),
    )
    .await
    .expect("StatLayer on freshly-created layer must succeed")
    .into_inner();

    assert_eq!(stat.created_by_user_name, "Asset Agent (app)");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_jwt_userinfo_opaque_bearer_returns_internal_with_sr_2_4_message() {
    // Opaque-bearer rejection: a non-JWT bearer string must surface as
    // `INTERNAL` at the CreateLayer wire boundary.
    //
    // Status-message contract — important. SR-2.4 pins the verbatim wording
    // on the `AuthError::InvalidResponse(msg)` boundary that the production
    // `JwtUserinfoClient::get_current_user` returns; that string flows
    // through `ServiceError::Internal(format!("auth backend: {e}"))` and
    // lands in the operator's `tracing::error!(error = %msg, ...)` log. The
    // `ServiceError → tonic::Status` impl deliberately redacts the wire
    // message to a generic `"internal error"` (`src/error.rs` line ~206) so
    // backend detail does not leak to clients. This is a security contract,
    // not a bug — see the `ServiceError::Internal` arm and its
    // `internal_status_does_not_leak_message` regression test.
    //
    // The verbatim-wording check therefore lives in two places:
    //   1. `src/auth/userinfo_jwt.rs::tests` pins the `AuthError` boundary.
    //   2. Below we pin `SR_2_4_OPAQUE_TOKEN_MSG` as a `const` referenced
    //      from this file, so a Task 2 reword surfaces as a compile-time
    //      drift signal even though the wire never sees the string.
    //
    // What we CAN assert from the integration-test position: the redacted
    // wire shape (`INTERNAL` + `"internal error"`). Both halves of the
    // contract are exercised; the verbatim-wire-string check the Subtask
    // body asks for is structurally impossible past the security redaction
    // and is left to the unit-level pin.
    let tv = TestValkey::setup().await;
    let svc = build_service_with_userinfo(
        &tv,
        Arc::new(AllowAllPermissions),
        Arc::new(JwtUserinfoClient::new()),
    );

    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = session_store
        .create("jwt-create-opaque", "/stage", "test-user")
        .await
        .expect("session creation must succeed");

    let ompf_bytes = delta_to_ompf_bytes(&empty_delta(), &[]);
    let stream = streaming_create(vec![
        make_create_header(&session_id, "opaque-layer"),
        make_create_data(ompf_bytes),
    ]);
    // Non-JWT bearer: single segment, no '.' separators — guarantees the
    // decoder bails at the segment-count check (one of the SR-2.4 paths).
    let request = make_create_request_with_bearer(stream, "opaque-bearer-token");

    let status = handle_create_layer(&svc, request)
        .await
        .expect_err("opaque (non-JWT) bearer must fail at the userinfo step");

    assert_eq!(
        status.code(),
        tonic::Code::Internal,
        "opaque bearer must surface as INTERNAL, got: {} — msg: {}",
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

    // Cross-file pin: also exercise the production decoder directly with
    // the same opaque bearer to confirm the underlying `AuthError`
    // carries the verbatim SR-2.4 wording. This is what makes
    // `SR_2_4_OPAQUE_TOKEN_MSG` load-bearing — a reword in
    // `JwtUserinfoClient::get_current_user` surfaces here even though the
    // gRPC wire path redacts the string.
    let decoder = JwtUserinfoClient::new();
    let auth_err = decoder
        .get_current_user("opaque-bearer-token")
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

// ── b.ot3: CreateLayer plans with a discarding mutation sink ─────────────────

/// CreateLayer applies its delta through `apply_delta_into` with a
/// `MutationSink::Discard`, so the persistence mutations it used to build and
/// then drop are never built at all. The object must still materialize
/// identically.
///
/// This is where the ticket's measurement was taken: CreateLayer materializes
/// the whole object from `ObjectState` via `queue_object_state` and never reads
/// the mutation list, yet an 8,000-sibling delta paid 256,032,000 bytes of
/// `u64` child vectors to build one. The counters and the state/stats
/// equivalence of the two sinks are pinned in `src/domain/delta.rs`
/// (`apply_delta_into_discard_leaves_identical_state_and_identical_stats`);
/// what only this tier can say is that the *real* CreateLayer call site, with
/// the sink it actually passes, persists what a collecting run would have.
///
/// The oracle is the collecting engine itself: the same delta applied to a
/// fresh `ObjectState` through `apply_delta`, compared against what came back
/// out of Valkey.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_with_many_siblings_persists_what_a_collecting_run_would() {
    const N: u64 = 500;

    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = session_store
        .create("b-ot3-create-session", "/stage", "test-user")
        .await
        .expect("session creation must succeed");

    let delta = oodf_ffi::OwnedDelta {
        is_diff: false,
        struct_commands: (1..=N)
            .map(|i| {
                oodf_ffi::StructCommand::CreateSection(oodf_ffi::CommandCreateSection {
                    parent_id: 0,
                    section_id: i,
                    section_name: format!("s{i}"),
                    section_type: 1,
                })
            })
            .collect(),
        ..Default::default()
    };

    let stream = streaming_create(vec![
        make_create_header(&session_id, "b-ot3-layer"),
        make_create_data(delta_to_ompf_bytes(&delta, &[])),
    ]);
    let object_id = handle_create_layer(&svc, make_request(stream))
        .await
        .expect("CreateLayer with many siblings must succeed")
        .into_inner()
        .object_id;

    // What the collecting engine would have produced for the same delta.
    let mut expected = live_edit::domain::types::ObjectState::default();
    live_edit::domain::delta::apply_delta(&mut expected, Some(delta))
        .expect("the reference application must succeed");

    let snapshot = live_edit::storage::test_support::read_object_state_for_test(
        tv.client(),
        tv.key_schema(),
        &object_id.to_string(),
    )
    .await
    .expect("snapshot load must succeed");
    let persisted = live_edit::storage::diff_read::snapshot_to_object_state(&snapshot);

    assert_eq!(
        persisted.children.get(&0),
        expected.children.get(&0),
        "the root's ordered children list must survive the discarding sink intact"
    );
    assert_eq!(
        persisted.children.get(&0).map(Vec::len).unwrap_or_default() as u64,
        N,
        "every sibling landed"
    );
    assert_eq!(
        persisted.sections.len(),
        expected.sections.len(),
        "every section row landed"
    );
    for (sid, section) in &expected.sections {
        let stored = persisted
            .sections
            .get(sid)
            .unwrap_or_else(|| panic!("section {sid} must be persisted"));
        assert_eq!(
            (
                stored.parent_id,
                stored.name.as_str(),
                stored.section_type,
                stored.section_order
            ),
            (
                section.parent_id,
                section.name.as_str(),
                section.section_type,
                section.section_order
            ),
            "section {sid} must match the collecting run"
        );
    }
}

// Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
/// Every state-rebuilding call site that never reads the mutation list still
/// routes through `apply_delta_into` (or, for stream replay and the checkpoint
/// base decoder, `apply_delta_into_mode`) with a `MutationSink::Discard`.
///
/// This is a deterministic **source** assertion, and deliberately so. The
/// ticket's acceptance criterion "CreateLayer must not construct a discarded
/// mutation list" is a call-site contract with no behavioural shadow: reverting
/// the creation step to `apply_delta_with_stats` and dropping the returned
/// `Vec` persists a byte-identical object, so
/// `create_layer_with_many_siblings_persists_what_a_collecting_run_would` above
/// — which compares persisted state against a collecting reference run, and
/// every other test in the suite — stays green through the regression. The
/// allocation the criterion forbids is invisible to every oracle except the
/// code itself.
///
/// CreateLayer's apply lives in its creation step,
/// `domain::layer_root::build_creation_state` (`src/domain/layer_root.rs`),
/// which the handler (`src/grpc/layer.rs`) calls for both a create and a
/// re-send's rebuild. So the creation step is scanned as a site, and the
/// handler is scanned for the opposite: it must build through that step and
/// call no apply entry point of its own, or a collecting call could come back
/// beside it.
///
/// It covers all three sites the ticket converted rather than the CreateLayer
/// one alone: the two checkpoint decoders rebuild an `ObjectState` from a delta
/// and likewise drop the mutations, they regress the same way, and scanning
/// them costs two more `include_str!`s. Stream replay (`src/domain/replay.rs`) is
/// here for the same reason: it rebuilds a state from committed entries and
/// never reads the mutations, so a collecting sink there would rebuild the
/// mutation list on every replayed entry. `src/storage/layer.rs` is
/// deliberately NOT here — `update_layer` *consumes* the mutation list to
/// build its write plan, so `apply_delta_with_stats` is correct there and
/// pinning its absence would be wrong.
///
/// Shape borrowed from `compose_module_does_no_io` in
/// `src/checkpoint/compose.rs`, anchor self-check included: the stripper runs
/// over real source, and without the anchors a stripping bug would make the
/// "no collecting call" half pass over an empty string.
#[test]
fn state_rebuilding_call_sites_still_discard_their_mutations() {
    /// Production span only — everything before the module's `#[cfg(test)]`,
    /// with line comments removed so prose mentioning a banned spelling (of
    /// which these modules have plenty) cannot trip the scan.
    fn non_test_source(path: &str, src: &str) -> String {
        let cut = src.find("#[cfg(test)]").unwrap_or_else(|| {
            panic!("{path} must have a #[cfg(test)] block for the scan to cut at")
        });
        let non_test = &src[..cut];
        assert!(
            !non_test.contains("/*"),
            "{path}: this guard strips line comments only; a block comment would let a \
             collecting call hide from it"
        );
        non_test
            .lines()
            .map(|line| line.split("//").next().unwrap_or(line))
            .collect::<Vec<_>>()
            .join("\n")
    }

    // (path, source, anchor proving the stripper left real code behind, the
    // discarding entry point the site calls)
    let sites: &[(&str, &str, &str, &str)] = &[
        (
            // CreateLayer's creation step: the content's apply and the apply
            // of a supplied root both discard.
            "src/domain/layer_root.rs",
            include_str!("../../../src/domain/layer_root.rs"),
            "pub fn build_creation_state(",
            "apply_delta_into(",
        ),
        (
            // The base decoder runs the engine in `ApplyMode::RestoreBase`, so
            // it too reaches it through the explicit-mode entry point.
            "src/checkpoint/apply.rs",
            include_str!("../../../src/checkpoint/apply.rs"),
            "pub(crate) fn decode_checkpoint_ompf(",
            "apply_delta_into_mode(",
        ),
        (
            "src/checkpoint/compose.rs",
            include_str!("../../../src/checkpoint/compose.rs"),
            "pub fn compose_lineage(",
            "apply_delta_into(",
        ),
        (
            // Stream replay runs the same engine in `ApplyMode::Replay`, so it
            // reaches it through the explicit-mode entry point. Generic over
            // the state's row form (the structural cache replays onto its
            // compact entry), hence the type parameter in the anchor.
            "src/domain/replay.rs",
            include_str!("../../../src/domain/replay.rs"),
            "pub fn replay_stream_record<R: ValueRows>(",
            "apply_delta_into_mode(",
        ),
    ];

    for (path, src, anchor, call) in sites {
        let code = non_test_source(path, src);

        // Self-check first: every assertion below is vacuous over an empty or
        // over-stripped string.
        assert!(
            code.contains(anchor),
            "{path}: comment stripping removed real code (`{anchor}` is gone); the scan \
             below would be vacuous"
        );

        assert!(
            code.contains(call),
            "{path} must apply its delta through `{call}..)` (b.ot3)"
        );
        assert!(
            code.contains("MutationSink::Discard"),
            "{path} calls `{call}..)` but no longer passes a \
             `MutationSink::Discard`; a collecting sink here rebuilds the per-command \
             mutation list this call site never reads — the 244 MiB-from-289 KiB \
             allocation b.ot3 removed"
        );
        assert!(
            !code.contains("apply_delta_with_stats("),
            "{path} must not call `apply_delta_with_stats` (b.ot3): it returns the \
             materialized mutation `Vec` that this call site drops on the floor. Rebuild \
             state via `apply_delta_into(.., &mut MutationSink::Discard, ..)` instead"
        );
        assert!(
            !code.contains("apply_delta("),
            "{path} must not call `apply_delta` (b.ot3); see the \
             `apply_delta_with_stats` note above — it is the same collecting engine \
             without the stats out-param"
        );
    }

    // The CreateLayer handler builds its state only through the creation step
    // scanned above, for a create and for a re-send's rebuild alike.
    let path = "src/grpc/layer.rs";
    let code = non_test_source(path, include_str!("../../../src/grpc/layer.rs"));
    assert!(
        code.contains("async fn create_layer_inner("),
        "{path}: comment stripping removed real code (`async fn create_layer_inner(` is \
         gone); the scan below would be vacuous"
    );
    assert_eq!(
        code.matches("build_creation_state(").count(),
        2,
        "{path} must build CreateLayer's state through `build_creation_state` at both \
         sites, the create and the re-send's rebuild"
    );
    for call in [
        "apply_delta(",
        "apply_delta_with_stats(",
        "apply_delta_into(",
        "apply_delta_into_mode(",
    ] {
        assert!(
            !code.contains(call),
            "{path} must not call `{call}..)` itself (b.ot3): CreateLayer applies its \
             content only through `build_creation_state`, which discards the mutations"
        );
    }
}

// ── G3: wrong-length external hash → INVALID_ARGUMENT, no blob, no object ────

/// CreateLayer stores a delta's external blobs before it allocates an
/// `object_id` (`store_external_values`, then `apply_delta_into`, then
/// `create_layer`). The collector used to zero-pad a 31-byte hash and truncate
/// a 33-byte one, so the blob was stored under a hash the client never sent
/// before `apply_delta_into` refused the delta. It now refuses the hash itself:
/// INVALID_ARGUMENT, no blob stored, and no object allocated or mapped.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_wrong_length_external_hash_returns_invalid_argument_and_stores_nothing() {
    use fred::interfaces::{KeysInterface, SetsInterface};
    use live_edit::storage::external_values::MockExternalValueStore;

    let tv = TestValkey::setup().await;
    let ext_store = Arc::new(MockExternalValueStore::new());
    let svc = live_edit::grpc::service::LiveEditService {
        ext_value_store: ext_store.clone(),
        ..build_service(&tv, Arc::new(AllowAllPermissions))
    };
    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create("g3-create-session", "/stage", "test-user")
        .await
        .expect("session creation must succeed");

    let blob = b"g3-external-blob".as_slice();
    // The field sits on the layer's root section, which the body creates: a
    // body leaving the field with no top-level section is refused for that
    // instead, which would make the 32-byte control below fail for the wrong
    // reason.
    let body = |hash: Vec<u8>| {
        let delta = with_root(crate::common::owned_delta::delta_with_set_field_external(
            1,
            "tex",
            hash,
            blob.len() as u64,
            1,
            0,
        ));
        delta_to_ompf_bytes(&delta, &[blob])
    };
    let counter_key = tv.key_schema().counter_object_id();
    let counter_before: Option<String> = tv
        .client()
        .pool
        .get(&counter_key)
        .await
        .expect("GET object-id counter must succeed");

    for len in [31usize, 33] {
        let layer = format!("g3-create-layer-{len}");
        let stream = streaming_create(vec![
            make_create_header(&session_id, &layer),
            make_create_data(body(vec![0xAB; len])),
        ]);
        let status = handle_create_layer(&svc, make_request(stream))
            .await
            .expect_err("a wrong-length external hash must be refused");

        assert_eq!(
            status.code(),
            tonic::Code::InvalidArgument,
            "{len}-byte hash: expected INVALID_ARGUMENT, got {:?}: {}",
            status.code(),
            status.message()
        );
        assert!(
            status.message().contains(&format!(
                "invalid external-value hash length: got {len} bytes, expected 32"
            )),
            "{len}-byte hash: the refusal must name the hash length, got: {}",
            status.message()
        );
        assert_eq!(
            ext_store.entry_count(),
            0,
            "{len}-byte hash: no external blob may be stored for a refused CreateLayer — \
             pre-fix it was stored under a padded / truncated hash"
        );
        let mapped: u64 = tv
            .client()
            .pool
            .exists(tv.key_schema().layer_mapping(&session_id, &layer))
            .await
            .expect("EXISTS layer mapping must succeed");
        assert_eq!(mapped, 0, "{len}-byte hash: no layer may be created");
    }

    let counter_after: Option<String> = tv
        .client()
        .pool
        .get(&counter_key)
        .await
        .expect("GET object-id counter must succeed");
    assert_eq!(
        counter_after, counter_before,
        "a refused CreateLayer must not allocate an object_id"
    );
    let layers: Vec<String> = tv
        .client()
        .pool
        .smembers(tv.key_schema().layers_index(&session_id))
        .await
        .expect("SMEMBERS layers index must succeed");
    assert!(
        layers.is_empty(),
        "a refused CreateLayer must not index a layer: {layers:?}"
    );

    // Control: the same body with the blob's 32-byte content hash creates the
    // layer and stores its blob, so the store is observably wired to this
    // handler and the zero above is not vacuous.
    let content_hash = live_edit::storage::external_values::ext_content_hash(blob);
    let stream = streaming_create(vec![
        make_create_header(&session_id, "g3-create-layer-32"),
        make_create_data(body(content_hash.to_vec())),
    ]);
    handle_create_layer(&svc, make_request(stream))
        .await
        .expect("a 32-byte external hash must create the layer");
    assert_eq!(
        ext_store.ref_count(&content_hash),
        Some(1),
        "the control CreateLayer must store its blob under the hash it named"
    );
}

// ── The layer's root section (SDD §9.7) ─────────────────────────────────────

/// Bound on one catch-up read, from the subscribe call to message #1's last
/// chunk: a handful of store round trips when healthy.
const CATCH_UP_BUDGET: std::time::Duration = std::time::Duration::from_secs(30);

/// The root section the service supplies to a create that leaves the layer
/// empty, as catch-up restates it.
fn supplied_root() -> StructCommand {
    StructCommand::DiffChangeSection(CommandDiffChangeSection {
        parent_id: 0,
        section_id: 1,
        section_name: String::new(),
        section_type: 7,
        section_order: 0,
    })
}

/// `object_id`'s stored state, read back from Valkey.
async fn stored_state(tv: &TestValkey, object_id: u64) -> ObjectState {
    let snapshot = live_edit::storage::test_support::read_object_state_for_test(
        tv.client(),
        tv.key_schema(),
        &object_id.to_string(),
    )
    .await
    .expect("snapshot load must succeed");
    live_edit::storage::diff_read::snapshot_to_object_state(&snapshot)
}

/// A brand-new subscriber's catch-up of `layer`: message #1's header and its
/// decoded delta. The whole read runs under [`CATCH_UP_BUDGET`].
async fn catch_up_first_message(
    svc: &LiveEditService,
    session_id: &str,
    layer: &str,
) -> (SubscribeReadLayerHeader, OwnedDelta) {
    let mut request = Request::new(SubscribeReadLayerRequest {
        session_id: session_id.to_owned(),
        layer_name: layer.to_owned(),
        object_id: 0,
        sequence_num: 0,
        client_id: "root-reader".to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    tokio::time::timeout(CATCH_UP_BUDGET, async {
        let mut stream = handle_subscribe_read_layer(svc, request)
            .await
            .expect("the subscription must start")
            .into_inner();
        let header = match stream.next().await {
            Some(Ok(message)) => match message.response_type {
                Some(subscribe_read_layer_response::ResponseType::Header(h)) => h,
                other => panic!("expected the catch-up's header, got {other:?}"),
            },
            other => panic!("expected the catch-up's header, got {other:?}"),
        };
        let mut body = Vec::with_capacity(header.total_size as usize);
        while (body.len() as u64) < header.total_size {
            match stream.next().await {
                Some(Ok(message)) => match message.response_type {
                    Some(subscribe_read_layer_response::ResponseType::Chunk(c)) => {
                        body.extend_from_slice(&c.data)
                    }
                    other => panic!("expected a catch-up chunk, got {other:?}"),
                },
                other => panic!("the catch-up ended early: {other:?}"),
            }
        }
        let container = OmpfContainer::parse(&body).expect("the catch-up is an OMPF");
        let delta = if container.delta_bytes.is_empty() {
            OwnedDelta::default()
        } else {
            oodf_ffi::parse_delta(container.delta_bytes).expect("the catch-up's delta parses")
        };
        (header, delta)
    })
    .await
    .expect("the catch-up's message #1 did not arrive within its budget")
}

/// The object-id counter's raw value.
async fn object_id_counter(tv: &TestValkey) -> Option<String> {
    tv.client()
        .pool
        .get(tv.key_schema().counter_object_id())
        .await
        .expect("GET object-id counter must succeed")
}

/// One `CreateLayer` of `layer` carrying `body` as its one chunk, answering
/// its object id.
async fn create_with_body(
    svc: &LiveEditService,
    session_id: &str,
    layer: &str,
    body: Vec<u8>,
) -> Result<u64, tonic::Status> {
    handle_create_layer(
        svc,
        make_request(streaming_create(vec![
            make_create_header(session_id, layer),
            make_create_data(body),
        ])),
    )
    .await
    .map(|response| response.into_inner().object_id)
}

/// Content leaving only an orphan section row — a `DiffChangeSection` under a
/// parent with no row, which the engine stores — leaves data but no top-level
/// section: refused `INVALID_ARGUMENT` naming the rule, with nothing created.
/// `StatLayer` answers `NOT_FOUND`, no mapping exists, and the object-id
/// counter did not move.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_orphan_only_content_is_refused_invalid_argument_and_creates_nothing() {
    const LAYER: &str = "orphan-only-layer";
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create("orphan-only-session", "/stage", "test-user")
        .await
        .expect("session creation must succeed");
    let orphan = OwnedDelta {
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 42,
            section_id: 5,
            section_name: "orphan".to_owned(),
            section_type: 0,
            section_order: 0,
        })],
        ..OwnedDelta::default()
    };
    let counter_before = object_id_counter(&tv).await;

    let status = create_with_body(&svc, &session_id, LAYER, delta_to_ompf_bytes(&orphan, &[]))
        .await
        .expect_err("orphan-only content must be refused");

    assert_eq!(
        status.code(),
        tonic::Code::InvalidArgument,
        "got {:?}: {}",
        status.code(),
        status.message()
    );
    assert!(
        status.message().contains("no top-level section"),
        "the refusal must name the layer-root rule, got: {}",
        status.message()
    );
    let stat = handle_stat_layer(
        &svc,
        stat_request_with_bearer(&session_id, LAYER, "test-token"),
    )
    .await
    .expect_err("no layer may exist after the refusal");
    assert_eq!(stat.code(), tonic::Code::NotFound);
    let mapped: u64 = tv
        .client()
        .pool
        .exists(tv.key_schema().layer_mapping(&session_id, LAYER))
        .await
        .expect("EXISTS layer mapping must succeed");
    assert_eq!(mapped, 0, "no layer mapping may be written");
    assert_eq!(
        object_id_counter(&tv).await,
        counter_before,
        "a refused CreateLayer must not allocate an object_id"
    );
}

/// The root the service supplies is exactly the root Kit creates: a
/// header-only create and Kit's captured create content
/// (`tools/fixtures/gze_inbound_deltas.json`, entry 1: a reset holding only
/// the root) leave identical stored objects, apart from the object id.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_header_only_stores_the_same_object_as_kits_captured_create() {
    use base64::{engine::general_purpose::STANDARD as B64, Engine};

    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create("kit-parity-session", "/stage", "test-user")
        .await
        .expect("session creation must succeed");
    let path = format!(
        "{}/tools/fixtures/gze_inbound_deltas.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let fixture: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&path).unwrap_or_else(|e| panic!("read fixture {path}: {e}")),
    )
    .unwrap_or_else(|e| panic!("parse fixture {path}: {e}"));
    let kit_create = &fixture[1];
    assert_eq!(kit_create["kind"], "CreateLayer", "entry 1 is Kit's create");
    let kit_body = B64
        .decode(
            kit_create["b64"]
                .as_str()
                .expect("the entry's b64 is a string"),
        )
        .expect("the entry's b64 decodes");

    let header_only = handle_create_layer(
        &svc,
        make_request(streaming_create(vec![make_create_header(
            &session_id,
            "header-only-layer",
        )])),
    )
    .await
    .expect("a header-only create must commit")
    .into_inner()
    .object_id;
    let kit = create_with_body(&svc, &session_id, "kit-layer", kit_body)
        .await
        .expect("Kit's create must commit");

    let mut supplied = stored_state(&tv, header_only).await;
    let mut kits = stored_state(&tv, kit).await;
    assert!(
        !kits.sections.is_empty(),
        "precondition: Kit's create holds its root"
    );
    supplied.object_id = 0;
    kits.object_id = 0;
    assert_eq!(
        supplied, kits,
        "a header-only create must store exactly the object Kit's own create stores"
    );
}

/// The supplied root takes section id 1, so a client that then creates its
/// own root as section 1 collides with it: the `UpdateLayer` is refused
/// `INVALID_ARGUMENT` "duplicate section_id 1", and the layer stays at
/// sequence 1.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_create_section_1_after_an_empty_create_is_refused_duplicate_section() {
    const LAYER: &str = "collision-layer";
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, "collision-session", LAYER).await;

    let status = handle_update_layer(
        &svc,
        make_request(streaming_update(vec![
            make_update_header(&session_id, LAYER, object_id),
            make_update_data(delta_to_ompf_bytes(&with_root(empty_delta()), &[])),
        ])),
    )
    .await
    .expect_err("creating section 1 again must be refused");

    assert_eq!(
        status.code(),
        tonic::Code::InvalidArgument,
        "got {:?}: {}",
        status.code(),
        status.message()
    );
    assert!(
        status.message().contains("duplicate section_id 1"),
        "the refusal must name the collision, got: {}",
        status.message()
    );
    assert_eq!(
        stored_state(&tv, object_id).await.sequence_num,
        1,
        "nothing applied"
    );
}

/// `CreateLayer completed` carries `root_supplied = true` when the service
/// supplied the root, and no `root_supplied` field when the content kept its
/// own (a `None` is left off the event, never logged `false`).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_completed_line_carries_root_supplied_only_for_a_supplied_root() {
    use super::log_capture::{assert_field, assert_field_absent, capture};
    const COMPLETED: &str = "CreateLayer completed";

    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create("root-log-session", "/stage", "test-user")
        .await
        .expect("session creation must succeed");

    let (supplied, log) = capture(create_with_body(
        &svc,
        &session_id,
        "supplied-layer",
        delta_to_ompf_bytes(&empty_delta(), &[]),
    ))
    .await;
    supplied.expect("an empty create must commit");
    assert_field(log.only(COMPLETED), "root_supplied", true.into());

    let (kept, log) = capture(create_with_body(
        &svc,
        &session_id,
        "rooted-layer",
        delta_to_ompf_bytes(&with_root(empty_delta()), &[]),
    ))
    .await;
    kept.expect("a rooted create must commit");
    assert_field_absent(
        log.only(COMPLETED),
        "root_supplied",
        "the content kept its own root",
    );
}
