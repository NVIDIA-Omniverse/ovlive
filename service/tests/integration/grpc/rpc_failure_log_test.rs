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

//! b.94g — every non-OK completion of a unary layer/session RPC emits exactly
//! one structured completion line, end-to-end through the real handlers: `RPC
//! failed`, or — for a server fault — ERROR `internal error` with the same
//! request fields.
//!
//! # The incident this file pins
//!
//! A client-rejected `UpdateLayer` (FAILED_PRECONDITION, stale `object_id`) on
//! a development deployment left **zero** trace in the pod log. The handler's early
//! return was a bare `.into()`, `From<ServiceError> for Status` logged only its
//! `Internal` arm, and the metrics layer's `endpoint_calls` tick carried a gRPC
//! code and nothing else — no `session_id`, no `layer_name`, no submitted vs.
//! resolved `object_id`, no message. Diagnosis meant inferring the cause from
//! the surrounding CreateLayer / subscribe / checkpoint lines for the same
//! object.
//!
//! # Why these tests assert on log output at all
//!
//! `docs-internal/test-writing-guide.md` § What NOT to Test warns off asserting
//! exact log messages, on the grounds that they are observability rather than
//! contract. b.94g is the case that inverts it: here the line *is* the
//! deliverable. An operator running one `grep 'RPC failed'` and reading
//! `object_id` / `current_object_id` off the hit is the entire remediation, so
//! a silent downgrade to `debug!`, a dropped identifier, or a doubled emission
//! would restore exactly the blindness the ticket was filed for — while every
//! status-code test in this directory stayed green. The assertions are
//! therefore deliberately structural (level, message, field presence/absence),
//! not cosmetic (no format, ordering, or timestamp is pinned).
//!
//! # What is asserted, and what is not
//!
//! Each test pins the **wire status alongside the log line**. The ticket's
//! hardest constraint is that no status code or message the client sees may
//! change, and a log-only test cannot notice if one did. The redundancy with
//! `update_layer_test.rs` / `update_layer_recovery_arm_test.rs` is the point:
//! those files own the status contract, these re-read it so a log assertion
//! can never be satisfied by a handler that started answering differently.
//!
//! Level assignment itself (17 codes × 8 stages × 2 resolution states) is
//! covered exhaustively — and without a Valkey round trip — by the unit matrix
//! in `src/grpc/rpc_log.rs`. The streaming RPCs and the completion points
//! outside any handler have their own files: `stream_completion_log_test.rs`
//! and `completion_log_server_test.rs`. This file covers the wiring the unit tests cannot
//! see: that the real handlers reach the stages they claim, populate the
//! context in the b.39x-safe order, and emit once per RPC.
//!
//! Section (h) also pins one handler-owned line beside the boundary: the
//! `DeleteSession` cascade's per-layer WARN, the only record of the failing
//! layer and progress counts when the failure is a redacted `INTERNAL`.
//! Section (j) pins the `UpdateLayer` store-call summary (attempts, path,
//! conflicts by kind, commit size) the failure line carries (SDD SR-2.6), and
//! that a call whose snapshot-fence wait reached its cap is logged by that
//! line alone, on every commit path.
//!
//! # Capture mechanism
//!
//! The shared `super::log_capture::capture`: a scoped JSON capture
//! subscriber (`tracing::instrument::WithSubscriber`), never a global one — these tests run on a multi-thread runtime and await
//! Valkey I/O, so a thread-local dispatcher would stop capturing at the first
//! `.await` that resumes on another worker, which reads exactly like "the line
//! was never emitted". JSON so an absent field is a *missing key* rather than
//! a `null`: `tracing` records nothing for a `None`, and the b.39x ordering
//! (never log an unvalidated identifier) is only assertable if absence is
//! distinguishable from emptiness.

use std::sync::Arc;

use fred::interfaces::{HashesInterface, KeysInterface, SetsInterface};
use serde_json::Value;
use tonic::codec::Streaming;
use tonic::{Code, Request};

use live_edit::grpc::layer::{
    handle_create_layer, handle_delete_layer, handle_stat_layer, handle_update_layer,
};
use live_edit::grpc::proto::{
    update_layer_request, CreateSessionRequest, DeleteLayerRequest, DeleteSessionRequest,
    ListSessionsRequest, StatLayerRequest, UpdateLayerHeader, UpdateLayerRequest,
};
#[cfg(feature = "test-fault-injection")]
use live_edit::grpc::service::LiveEditService;
use live_edit::grpc::session::{
    handle_create_session, handle_delete_session, handle_list_sessions,
};
use live_edit::grpc::BearerToken;
use live_edit::storage::keys;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;

use super::helpers::{
    build_service, create_session_and_layer, make_create_data, make_create_header,
    make_update_data, streaming_create, streaming_update, AllowAllPermissions, AttestedStreaming,
    DenyAllPermissions, IntoHandlerRequest as _,
};
#[cfg(feature = "test-fault-injection")]
use super::helpers::{build_service_with_layer_store, layer_store_with_name_faults};
use super::log_capture::{
    assert_common_fields, assert_field, assert_field_absent, capture, Captured, INTERNAL_ERROR_MSG,
    RPC_FAILURE_MSG,
};
use crate::common::{
    owned_delta::{delta_to_ompf_bytes, empty_delta},
    wipe_object_keys, EMPTY_HEADER_SENTINEL_MSG, PARTIAL_HEADER_SENTINEL_MSG,
};

// =============================================================================
// Constants
// =============================================================================

/// The `warn!` the `DeleteSession` cascade emits when a layer deletion fails.
/// It carries the failing layer and the progress counts, which the wire message
/// loses whenever the failure is redacted as `INTERNAL`.
const CASCADE_WARN_MSG: &str = "DeleteSession cascade aborted on layer deletion failure";

/// Bearer token every request in this file carries.
///
/// Deliberately distinctive rather than the tree-wide `"test-token"`: every
/// test greps the WHOLE capture buffer for it, and a value that also appears
/// in unrelated fixtures would make the grep meaningless. Not a real
/// credential — the mock permissions clients ignore it entirely.
const B94G_BEARER: &str = "b94g-bearer-must-never-be-logged-7f3a";

/// The `client_id` UpdateLayer requests in this file carry.
const B94G_CLIENT_ID: &str = "b94g-client";

/// The header timestamp UpdateLayer requests in this file carry.
///
/// Named because the two builders below reach it by different routes — one
/// through `common::update_request_with_bearer`, one by building the header
/// inline — and a divergence would silently make the chunked and header-only
/// cases different requests.
const B94G_TIMESTAMP_MICROS: u64 = 1000;

/// A `session_id` that fails `validate_session_id` on its ESC byte (0x1B).
///
/// The b.39x threat model in one value: pre-b.39x an unvalidated `session_id`
/// carrying ANSI escapes reached tracing output. The failure boundary is one
/// more tracing sink, so the same string must not reach it either.
const CONTROL_CHAR_SESSION_ID: &str = "b94g-bad\u{1b}[31m-session";

/// Caller spelling of a base path, and the canonical form the service
/// authorizes and keys storage under (b.uoc). Same pair `session_test.rs`
/// uses for the canonicalization contract.
const RAW_BASE_PATH: &str = "/main/Playground/Jdoe@example.com";
const CANONICAL_BASE_PATH: &str = "/main/Playground/Jdoe%40example.com";

// =============================================================================
// Capture assertions
// =============================================================================

/// This file's assertions over the shared [`Captured`], kept as methods so the
/// call sites read as before.
trait FailureLogAssertions {
    /// The single `RPC failed` event, asserting there is exactly one.
    ///
    /// The count is checked here rather than in each caller so no test can
    /// accidentally assert on the first of two duplicated events (the
    /// double-emission regression) and pass.
    fn only_failure(&self) -> &Value;

    /// Assert the RPC logged no `RPC failed` event at all.
    fn assert_no_failure(&self, ctx: &str);

    /// Assert the bearer token appears nowhere in the captured output.
    ///
    /// Swept over the raw buffer, not the parsed fields: the point is to catch
    /// a token that leaked into a field this file does not name — a status
    /// message quoting an upstream error, a future field, a span. `BearerToken`
    /// has a redacted `Debug`, and `PermissionDenied` / `Unauthenticated`
    /// carry an empty message by construction, so this should hold trivially;
    /// it is a canary, and a canary that only checks the fields you already
    /// thought of is not one. It delegates to
    /// [`Captured::assert_absent_everywhere`], so the JSON-escaped form is
    /// swept too.
    fn assert_no_bearer_material(&self);
}

impl FailureLogAssertions for Captured {
    fn only_failure(&self) -> &Value {
        self.only(RPC_FAILURE_MSG)
    }

    fn assert_no_failure(&self, ctx: &str) {
        let failures = self.with_message(RPC_FAILURE_MSG);
        assert!(
            failures.is_empty(),
            "{ctx}: expected no `{RPC_FAILURE_MSG}` event; got {failures:?}"
        );
    }

    fn assert_no_bearer_material(&self) {
        self.assert_absent_everywhere(B94G_BEARER);
    }
}

// =============================================================================
// Request builders
// =============================================================================

/// Attach [`B94G_BEARER`] to a request.
fn with_bearer<T>(inner: T) -> Request<T> {
    let mut request = Request::new(inner);
    request
        .extensions_mut()
        .insert(BearerToken::new(B94G_BEARER));
    request
}

/// Attach [`B94G_BEARER`] to an upload's request, which also carries the
/// upload's end-of-stream record.
fn upload_with_bearer<T>(stream: AttestedStreaming<T>) -> Request<tonic::codec::Streaming<T>> {
    let mut request = stream.into_handler_request();
    request
        .extensions_mut()
        .insert(BearerToken::new(B94G_BEARER));
    request
}

/// A header-only `UpdateLayer` request (empty OMPF → SDD §8.4 empty commit).
///
/// Delegates to `common::update_request_with_bearer`, the bearer-parameterised
/// form of the shared builder: this file's canary needs a token that appears
/// nowhere else in the tree, which is exactly what that parameter is for. The
/// chunked sibling below stays local — the shared helper builds a header and
/// nothing after it.
fn update_request(
    session_id: &str,
    layer_name: &str,
    object_id: u64,
) -> Request<Streaming<UpdateLayerRequest>> {
    crate::common::update_request_with_bearer(
        session_id,
        layer_name,
        object_id,
        B94G_TIMESTAMP_MICROS,
        B94G_CLIENT_ID,
        B94G_BEARER,
    )
}

/// An `UpdateLayer` request carrying `chunks` of OMPF payload after the header.
fn update_request_with_chunks(
    session_id: &str,
    layer_name: &str,
    object_id: u64,
    chunks: Vec<Vec<u8>>,
) -> Request<Streaming<UpdateLayerRequest>> {
    let header = UpdateLayerRequest {
        message_type: Some(update_layer_request::MessageType::Header(
            UpdateLayerHeader {
                session_id: session_id.to_owned(),
                layer_name: layer_name.to_owned(),
                object_id,
                timestamp_microseconds_since_epoch: B94G_TIMESTAMP_MICROS,
                client_id: B94G_CLIENT_ID.to_owned(),
                operation_id: String::new(),
            },
        )),
    };
    let mut messages = vec![header];
    messages.extend(chunks.into_iter().map(make_update_data));
    upload_with_bearer(streaming_update(messages))
}

fn stat_request(session_id: &str, layer_name: &str) -> Request<StatLayerRequest> {
    with_bearer(StatLayerRequest {
        session_id: session_id.to_owned(),
        layer_name: layer_name.to_owned(),
    })
}

fn delete_layer_request(session_id: &str, layer_name: &str) -> Request<DeleteLayerRequest> {
    with_bearer(DeleteLayerRequest {
        session_id: session_id.to_owned(),
        layer_name: layer_name.to_owned(),
    })
}

fn delete_session_request(session_id: &str) -> Request<DeleteSessionRequest> {
    with_bearer(DeleteSessionRequest {
        session_id: session_id.to_owned(),
    })
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

// =============================================================================
// (a) The ticket's incident path: a stale object_id
// =============================================================================

/// The reported incident, end to end: an `UpdateLayer` carrying a stale
/// `object_id` logs one INFO line carrying BOTH ids and the full request
/// identity.
///
/// `object_id` (submitted) and `current_object_id` (resolved) are the two
/// facts the dev incident had to be reconstructed from — they existed only
/// inside the client-visible status detail string, which the server never
/// recorded. `stage = "resolve"` is what keeps the level at INFO: the mapping
/// resolved, so the object is provably live and this is a stale client, not
/// keyspace damage.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_object_id_mismatch_logs_one_info_failure_with_both_object_ids() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let layer_name = "b94g-mismatch-layer";
    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, "b94g-mismatch-session", layer_name).await;

    let stale = object_id + 1;
    let (result, captured) = capture(handle_update_layer(
        &svc,
        update_request(&session_id, layer_name, stale),
    ))
    .await;

    // The wire contract, unchanged. Re-asserted here (it is also pinned by
    // `update_layer_test.rs`) so no log assertion below can be satisfied by a
    // handler that started answering something else.
    let status = result.expect_err("a stale object_id must fail the RPC");
    assert_eq!(status.code(), Code::FailedPrecondition);
    assert_eq!(
        status.message(),
        format!("object_id mismatch: expected {object_id}, got {stale}")
    );

    let event = captured.only_failure();
    assert_common_fields(
        event,
        "INFO",
        "UpdateLayer",
        "resolve",
        "FAILED_PRECONDITION",
    );
    assert_field(event, "session_id", session_id.as_str().into());
    assert_field(event, "layer_name", layer_name.into());
    assert_field(event, "client_id", B94G_CLIENT_ID.into());
    assert_field(event, "object_id", stale.into());
    assert_field(event, "current_object_id", object_id.into());
    assert_field(event, "status_message", status.message().into());
    captured.assert_no_bearer_material();
}

// =============================================================================
// (b) Storage sentinels: the server-side shapes that share a client code
// =============================================================================

/// A commit refused by the partial-header sentinel logs at WARN, at
/// `stage = "storage"`, with the object resolved.
///
/// Same gRPC code as the stale-client case above (FAILED_PRECONDITION) and a
/// deliberately different level: the mapping proved the object live and the
/// *commit path* then refused it, which is damaged keyspace, not a client
/// mistake. That the two are distinguishable at all is the whole reason the
/// level is computed from `(stage, object resolved)` rather than from the
/// status message — the sentinel strings are module-private to `src/storage`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_partial_header_logs_one_warn_failure_at_the_storage_stage() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let layer_name = "b94g-partial-layer";
    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, "b94g-partial-session", layer_name).await;

    // Damage rather than delete: `seq` alone goes, so the header is PRESENT
    // and the commit path (not the recovery arm) refuses it. Same fixture as
    // `update_layer_recovery_arm_test.rs`.
    let header_key = tv.key_schema().obj_header(&object_id.to_string());
    let _: i64 = tv
        .client()
        .pool
        .hdel(&header_key, "seq")
        .await
        .expect("HDEL seq must succeed");

    let (result, captured) = capture(handle_update_layer(
        &svc,
        update_request(&session_id, layer_name, object_id),
    ))
    .await;

    let status = result.expect_err("a commit against a partial header must fail the RPC");
    assert_eq!(status.code(), Code::FailedPrecondition);
    assert_eq!(status.message(), PARTIAL_HEADER_SENTINEL_MSG);

    let event = captured.only_failure();
    assert_common_fields(
        event,
        "WARN",
        "UpdateLayer",
        "storage",
        "FAILED_PRECONDITION",
    );
    assert_field(event, "current_object_id", object_id.into());
    assert_field(event, "object_id", object_id.into());
    assert_field(event, "status_message", PARTIAL_HEADER_SENTINEL_MSG.into());
    captured.assert_no_bearer_material();
}

/// A commit against a wiped header (mapping alive) logs at WARN, at
/// `stage = "storage"`.
///
/// The sibling sentinel: NOT_FOUND raised by the commit path against an object
/// the mapping just proved live is a wiped `obj:hdr` under a live layer — the
/// natural symptom of partial keyspace loss, and the shape a `NOT_FOUND` from
/// mapping resolution (an unknown layer name, INFO) must never be confused
/// with.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_empty_header_logs_one_warn_failure_at_the_storage_stage() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let layer_name = "b94g-empty-header-layer";
    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, "b94g-empty-header-session", layer_name).await;

    // Wipes `le:{object_id}:obj:*` and leaves the layer mapping intact, so the
    // handler resolves the mapping and meets the empty header at the COMMIT.
    // `build_service` wires a disabled recovery hook, so the merged NotFound
    // arm short-circuits to `NotRecoverable` and surfaces the sentinel.
    wipe_object_keys(&tv, object_id).await;

    let (result, captured) = capture(handle_update_layer(
        &svc,
        update_request(&session_id, layer_name, object_id),
    ))
    .await;

    let status = result.expect_err("a commit against a wiped header must fail the RPC");
    assert_eq!(status.code(), Code::NotFound);
    assert_eq!(status.message(), EMPTY_HEADER_SENTINEL_MSG);

    let event = captured.only_failure();
    assert_common_fields(event, "WARN", "UpdateLayer", "storage", "NOT_FOUND");
    assert_field(event, "current_object_id", object_id.into());
    assert_field(event, "status_message", EMPTY_HEADER_SENTINEL_MSG.into());
    captured.assert_no_bearer_material();
}

/// `DeleteLayer` of an absent layer logs at INFO even though it reaches
/// `stage = "storage"`.
///
/// The counterexample that keeps the escalation honest: the pair
/// `(Storage, object resolved)` is the discriminator, not the stage alone.
/// `DeleteLayer` resolves no mapping of its own, so its NOT_FOUND is the
/// ordinary "you named a layer that does not exist" client shape and must not
/// read as keyspace damage — otherwise every idempotent re-delete would WARN.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_layer_missing_layer_logs_info_with_no_resolved_object_id() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let (session_id, _object_id) =
        create_session_and_layer(&tv, &svc, "b94g-delete-session", "b94g-real-layer").await;

    let (result, captured) = capture(handle_delete_layer(
        &svc,
        delete_layer_request(&session_id, "b94g-no-such-layer"),
    ))
    .await;

    let status = result.expect_err("deleting an absent layer must fail the RPC");
    assert_eq!(status.code(), Code::NotFound);

    let event = captured.only_failure();
    assert_common_fields(event, "INFO", "DeleteLayer", "storage", "NOT_FOUND");
    assert_field(event, "session_id", session_id.as_str().into());
    assert_field(event, "layer_name", "b94g-no-such-layer".into());
    assert_field_absent(
        event,
        "current_object_id",
        "DeleteLayer resolves no mapping of its own, so it can never claim the object was \
         proven live — and that claim is what escalates NOT_FOUND to WARN",
    );
    captured.assert_no_bearer_material();
}

/// A malformed OMPF payload logs at INFO, at `stage = "parse"`.
///
/// The one stage no other test in this file reaches, and the reason `stage` is
/// a field at all: `INVALID_ARGUMENT` from the header validators and
/// `INVALID_ARGUMENT` from the delta parser are the same code on the same
/// endpoint, and only the stage tells an operator whether the client sent a
/// bad identifier or a bad payload. The mapping has already resolved by this
/// point, so the line carries both object ids — and stays INFO, because a
/// resolved object only escalates a failure raised by the commit itself.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_malformed_payload_logs_info_at_the_parse_stage() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let layer_name = "b94g-parse-layer";
    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, "b94g-parse-session", layer_name).await;

    // Not an OMPF container: the magic word alone is wrong, which the pure-Rust
    // container parser rejects before any OODF verification runs.
    let garbage = vec![0xde, 0xad, 0xbe, 0xef, 0x00, 0x01, 0x02, 0x03];
    let (result, captured) = capture(handle_update_layer(
        &svc,
        update_request_with_chunks(&session_id, layer_name, object_id, vec![garbage]),
    ))
    .await;

    let status = result.expect_err("a malformed OMPF payload must fail the RPC");
    assert_eq!(status.code(), Code::InvalidArgument);

    let event = captured.only_failure();
    assert_common_fields(event, "INFO", "UpdateLayer", "parse", "INVALID_ARGUMENT");
    assert_field(event, "object_id", object_id.into());
    assert_field(event, "current_object_id", object_id.into());
    assert_field(event, "status_message", status.message().into());
    captured.assert_no_bearer_material();
}

// =============================================================================
// (c) Success paths stay silent
// =============================================================================

/// A successful `UpdateLayer` emits no boundary failure line.
///
/// Asserted as "no `RPC failed` event", NOT as "nothing at INFO": the storage
/// layer's own `UpdateLayer committed` INFO line is a separate, known site
/// (b.p4t), and a test that banned all INFO would fail for the wrong reason
/// and invite someone to silence the wrong line.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn successful_update_layer_logs_no_failure_line() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let layer_name = "b94g-success-layer";
    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, "b94g-success-session", layer_name).await;

    let (result, captured) = capture(handle_update_layer(
        &svc,
        update_request(&session_id, layer_name, object_id),
    ))
    .await;

    let response = result
        .expect("a matching object_id must commit")
        .into_inner();
    assert!(
        response.sequence_num > 0,
        "precondition: the commit must have advanced the sequence, else the silence below is \
         vacuous"
    );
    captured.assert_no_failure("a successful UpdateLayer");
    captured.assert_no_bearer_material();
}

/// A successful `CreateLayer` emits no boundary failure line.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn successful_create_layer_logs_no_failure_line() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let (session_id, _object_id) =
        create_session_and_layer(&tv, &svc, "b94g-create-ok-session", "b94g-first-layer").await;

    let ompf = delta_to_ompf_bytes(&empty_delta(), &[]);
    let stream = streaming_create(vec![
        make_create_header(&session_id, "b94g-second-layer"),
        make_create_data(ompf),
    ]);

    let (result, captured) = capture(handle_create_layer(&svc, upload_with_bearer(stream))).await;

    let response = result.expect("CreateLayer must succeed").into_inner();
    assert!(
        response.object_id > 0,
        "precondition: an object must actually have been minted"
    );
    captured.assert_no_failure("a successful CreateLayer");
    captured.assert_no_bearer_material();
}

// =============================================================================
// (d) One line per failed RPC — no more, no fewer
// =============================================================================

/// Two failing RPCs under one capture produce exactly two lines, one each.
///
/// The regression fence against double emission. `only_failure()` already
/// asserts "exactly one" per test, but a shared or static emitter that fired
/// twice per *call* would be caught by that while one that fired once per
/// *process* would not. Counting across two distinct RPCs pins the 1:1
/// relationship itself: a doubled line would silently double every `RPC
/// failed` count an operator greps.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn each_failed_rpc_logs_exactly_one_failure_line() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let layer_name = "b94g-count-layer";
    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, "b94g-count-session", layer_name).await;

    let (codes, captured) = capture(async {
        let first =
            handle_update_layer(&svc, update_request(&session_id, layer_name, object_id + 1))
                .await
                .expect_err("stale object_id must fail")
                .code();
        let second = handle_stat_layer(&svc, stat_request(&session_id, "b94g-no-such-layer"))
            .await
            .expect_err("an absent layer must fail StatLayer")
            .code();
        (first, second)
    })
    .await;

    assert_eq!(
        codes,
        (Code::FailedPrecondition, Code::NotFound),
        "precondition: both RPCs must have failed, and for the intended reasons"
    );

    let failures = captured.with_message(RPC_FAILURE_MSG);
    assert_eq!(
        failures.len(),
        2,
        "two failed RPCs must produce exactly two lines; got {} — full capture:\n{}",
        failures.len(),
        captured.text
    );
    let endpoints: Vec<&Value> = failures.iter().map(|e| &e["fields"]["endpoint"]).collect();
    assert_eq!(
        endpoints,
        vec!["UpdateLayer", "StatLayer"],
        "each line must name its own RPC, in completion order"
    );
    captured.assert_no_bearer_material();
}

// =============================================================================
// (e) b.39x ordering: never log an identifier its validator rejected
// =============================================================================

/// A malformed `session_id` fails at `stage = "header"`, the field is absent
/// from the boundary event, and the rejected string appears NOWHERE in the
/// whole capture buffer.
///
/// b.39x closed a hole where an unvalidated `session_id` — up to 64 KB, ANSI
/// escapes included — flowed into Valkey keys, pub/sub channels and tracing
/// output. The failure boundary is one more tracing sink, so every `ctx`
/// record sits *below* its validator. This is the test that would fail if one
/// were ever hoisted above.
///
/// The buffer-wide claim is backed by [`Captured::assert_absent_everywhere`],
/// which sweeps the escaped rendering as well as the literal one — the
/// rejected value carries an ESC byte, and JSON output never contains one
/// literally, so a raw-only sweep would pass unconditionally.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_invalid_session_id_logs_no_session_id_field() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let (result, captured) = capture(handle_update_layer(
        &svc,
        update_request(CONTROL_CHAR_SESSION_ID, "b94g-layer", 7),
    ))
    .await;

    let status = result.expect_err("a control-character session_id must fail the RPC");
    assert_eq!(status.code(), Code::InvalidArgument);

    let event = captured.only_failure();
    assert_common_fields(event, "INFO", "UpdateLayer", "header", "INVALID_ARGUMENT");
    assert_field_absent(
        event,
        "session_id",
        "the value was rejected by `validate_session_id`, so it is attacker-controlled and must \
         never be recorded",
    );
    // The whole rejected value, then the escape sequence on its own — the
    // second catches a partial leak (a truncated or reformatted spelling that
    // still carries the control bytes) that a whole-value sweep would miss.
    captured.assert_absent_everywhere(CONTROL_CHAR_SESSION_ID);
    captured.assert_absent_everywhere("\u{1b}[");
    captured.assert_no_bearer_material();
}

/// `object_id = 0` is rejected before the id is recorded, so the line carries
/// the (already validated) `session_id` but no `object_id`.
///
/// The finer-grained half of the same ordering rule: the populate calls are
/// interleaved with the validators one field at a time, not batched after all
/// of them. A line that carried `object_id = 0` would be reporting a value the
/// handler had just declared invalid.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_zero_object_id_logs_the_session_id_but_no_object_id() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let layer_name = "b94g-zero-oid-layer";
    let (session_id, _object_id) =
        create_session_and_layer(&tv, &svc, "b94g-zero-oid-session", layer_name).await;

    let (result, captured) = capture(handle_update_layer(
        &svc,
        update_request(&session_id, layer_name, 0),
    ))
    .await;

    let status = result.expect_err("object_id = 0 must fail the RPC");
    assert_eq!(status.code(), Code::InvalidArgument);

    let event = captured.only_failure();
    assert_common_fields(event, "INFO", "UpdateLayer", "header", "INVALID_ARGUMENT");
    assert_field(event, "session_id", session_id.as_str().into());
    assert_field_absent(
        event,
        "object_id",
        "the zero id was rejected before `submitted_object_id` runs",
    );
    assert_field_absent(
        event,
        "layer_name",
        "the layer-name validator runs after the object_id check, so nothing recorded it",
    );
    captured.assert_no_bearer_material();
}

/// A header-only `UpdateLayer` request carrying `operation_id` (b.v6d).
fn update_request_with_operation(
    session_id: &str,
    layer_name: &str,
    object_id: u64,
    operation_id: &str,
) -> Request<Streaming<UpdateLayerRequest>> {
    upload_with_bearer(streaming_update(vec![UpdateLayerRequest {
        message_type: Some(update_layer_request::MessageType::Header(
            UpdateLayerHeader {
                session_id: session_id.to_owned(),
                layer_name: layer_name.to_owned(),
                object_id,
                timestamp_microseconds_since_epoch: B94G_TIMESTAMP_MICROS,
                client_id: B94G_CLIENT_ID.to_owned(),
                operation_id: operation_id.to_owned(),
            },
        )),
    }]))
}

/// b.v6d T18: an `operation_id` its validator rejected is never recorded. The
/// line carries the identifiers validated before it (`session_id`,
/// `layer_name`, `client_id`) and no `operation_id` field, and the rejected
/// value appears nowhere in the capture buffer.
///
/// The id is logged verbatim at INFO once valid, so the b.39x ordering applies
/// to it exactly as to `session_id` above.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_invalid_operation_id_logs_no_operation_id_field() {
    const REJECTED: &str = "v6d rejected {operation} id";
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let layer_name = "bv6d-log-invalid-op-layer";
    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, "bv6d-log-invalid-op-session", layer_name).await;

    let (result, captured) = capture(handle_update_layer(
        &svc,
        update_request_with_operation(&session_id, layer_name, object_id, REJECTED),
    ))
    .await;

    let status = result.expect_err("a malformed operation_id must fail the RPC");
    assert_eq!(status.code(), Code::InvalidArgument);

    let event = captured.only_failure();
    assert_common_fields(event, "INFO", "UpdateLayer", "header", "INVALID_ARGUMENT");
    assert_field(event, "session_id", session_id.as_str().into());
    assert_field(event, "layer_name", layer_name.into());
    assert_field(event, "client_id", B94G_CLIENT_ID.into());
    assert_field_absent(
        event,
        "operation_id",
        "the value was rejected by `validate_operation_id`, so it must never be recorded",
    );
    captured.assert_absent_everywhere(REJECTED);
    captured.assert_no_bearer_material();
}

/// b.v6d T18: once validated, the `operation_id` rides every failure line of the
/// call, so an operator can follow one write intent across a client's re-sends.
/// Here a stale `object_id` fails the call after validation.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_failure_after_validation_logs_the_operation_id() {
    const OPERATION_ID: &str = "bv6d-logged-operation-1";
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let layer_name = "bv6d-log-op-layer";
    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, "bv6d-log-op-session", layer_name).await;

    let (result, captured) = capture(handle_update_layer(
        &svc,
        update_request_with_operation(&session_id, layer_name, object_id + 1, OPERATION_ID),
    ))
    .await;

    let status = result.expect_err("a stale object_id must fail the RPC");
    assert_eq!(status.code(), Code::FailedPrecondition);

    let event = captured.only_failure();
    assert_common_fields(
        event,
        "INFO",
        "UpdateLayer",
        "resolve",
        "FAILED_PRECONDITION",
    );
    assert_field(event, "operation_id", OPERATION_ID.into());
    captured.assert_no_bearer_material();
}

/// b.v6d T18: the `CreateLayer` twin of the rejected-id case — the header's
/// session and layer were recorded, the rejected `operation_id` was not.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_invalid_operation_id_logs_no_operation_id_field() {
    const REJECTED: &str = "v6d:rejected:create:id";
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create("bv6d-log-create-op", "/bv6d-log-create-op", "test-user")
        .await
        .expect("session creation must succeed");
    let layer_name = "bv6d-log-create-op-layer";

    let mut header = make_create_header(&session_id, layer_name);
    if let Some(live_edit::grpc::proto::create_layer_request::MessageType::Header(h)) =
        header.message_type.as_mut()
    {
        h.operation_id = REJECTED.to_owned();
    }
    let (result, captured) = capture(handle_create_layer(
        &svc,
        upload_with_bearer(streaming_create(vec![header])),
    ))
    .await;

    let status = result.expect_err("a malformed operation_id must fail the RPC");
    assert_eq!(status.code(), Code::InvalidArgument);

    let event = captured.only_failure();
    assert_common_fields(event, "INFO", "CreateLayer", "header", "INVALID_ARGUMENT");
    assert_field(event, "session_id", session_id.as_str().into());
    assert_field(event, "layer_name", layer_name.into());
    assert_field_absent(
        event,
        "operation_id",
        "the value was rejected by `validate_operation_id`, so it must never be recorded",
    );
    captured.assert_absent_everywhere(REJECTED);
    captured.assert_no_bearer_material();
}

/// b.v6d T18: the `CreateLayer` twin of the validated-id case — once the
/// header validator accepts the `operation_id`, every failure line of the call
/// carries it. Here the write-permission check denies the call after
/// validation.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_failure_after_validation_logs_the_operation_id() {
    const OPERATION_ID: &str = "bv6d-logged-create-operation-1";
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(DenyAllPermissions));
    // Minted through the store directly: every handler route to a session runs
    // the permission probe this test denies.
    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create(
            "bv6d-log-create-op-ok",
            "/bv6d-log-create-op-ok",
            "test-user",
        )
        .await
        .expect("session creation must succeed");
    let layer_name = "bv6d-log-create-op-ok-layer";

    let mut header = make_create_header(&session_id, layer_name);
    if let Some(live_edit::grpc::proto::create_layer_request::MessageType::Header(h)) =
        header.message_type.as_mut()
    {
        h.operation_id = OPERATION_ID.to_owned();
    }
    let ompf = delta_to_ompf_bytes(&empty_delta(), &[]);
    let (result, captured) = capture(handle_create_layer(
        &svc,
        upload_with_bearer(streaming_create(vec![header, make_create_data(ompf)])),
    ))
    .await;

    let status = result.expect_err("a denied CreateLayer must fail the RPC");
    assert_eq!(status.code(), Code::PermissionDenied);

    let event = captured.only_failure();
    assert_common_fields(event, "WARN", "CreateLayer", "auth", "PERMISSION_DENIED");
    assert_field(event, "operation_id", OPERATION_ID.into());
    captured.assert_no_bearer_material();
}

// =============================================================================
// (f) A server fault reports once, at ERROR, with the request context
// =============================================================================

/// An INTERNAL server fault emits exactly ONE line — ERROR `internal error`
/// carrying both the un-redacted detail and the request fields — and no `RPC
/// failed`; the context-free fallback never fires.
///
/// The ERROR line is the one operators page on, and it used to be the one line
/// with no request context: the conversion that logged it could not see the
/// session, layer or object. The conversion now attaches the detail to the
/// `Status` and the handler boundary logs it with the context it has; this
/// test pins all three halves of that — one line, context present, detail
/// present — and the absence of the fault's own `Drop` fallback, which would
/// fire if the boundary logged without marking the fault reported. The status
/// is dropped inside the capture for exactly that reason.
///
/// The fault is induced by retyping `obj:hdr` from a Hash to a String, so the
/// handler's `HGETALL` fails `WRONGTYPE`. `map_fred_err_for_grpc` classifies
/// that as permanent (not in the transient kinds), hence `Internal`. No
/// fault-injection feature is needed, so this test compiles and runs in every
/// shape of the suite.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stat_layer_internal_error_logs_one_error_with_the_request_context() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let layer_name = "b94g-internal-layer";
    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, "b94g-internal-session", layer_name).await;

    let header_key = tv.key_schema().obj_header(&object_id.to_string());
    let _: () = tv
        .client()
        .pool
        .set(&header_key, "not-a-hash", None, None, false)
        .await
        .expect("retyping obj:hdr to a String must succeed");

    let ((code, message), captured) = capture(async {
        let status = handle_stat_layer(&svc, stat_request(&session_id, layer_name))
            .await
            .expect_err("a wrong-typed header must fail the RPC");
        // Dropped here, inside the capture: an unreported fault would log its
        // fallback line now.
        (status.code(), status.message().to_owned())
    })
    .await;

    assert_eq!(
        code,
        Code::Internal,
        "precondition: the fixture must produce an INTERNAL, else nothing below applies"
    );
    assert_eq!(
        message, INTERNAL_ERROR_MSG,
        "the client must still see only the sanitized message"
    );

    captured.assert_no_failure("a server fault logs `internal error`, not `RPC failed`");
    let internals = captured.with_message(INTERNAL_ERROR_MSG);
    assert_eq!(
        internals.len(),
        1,
        "a server fault must be reported exactly once — a second, context-free line is the \
         fault's Drop fallback; got {} — full capture:\n{}",
        internals.len(),
        captured.text
    );
    let event = internals[0];
    assert_common_fields(event, "ERROR", "StatLayer", "storage", "INTERNAL");
    assert_field(event, "session_id", session_id.as_str().into());
    assert_field(event, "layer_name", layer_name.into());
    assert_field(event, "base_path", "/stage".into());
    assert_field(event, "current_object_id", object_id.into());
    assert_field(event, "status_message", INTERNAL_ERROR_MSG.into());
    let detail = event["fields"]["error"].as_str().unwrap_or_default();
    assert!(
        detail.contains("WRONGTYPE"),
        "the ERROR line must carry the un-redacted detail the wire lost; got {event}"
    );
    captured.assert_no_bearer_material();
}

// =============================================================================
// (g) Session RPCs ride the same boundary
// =============================================================================

/// `DeleteSession` on an unknown session logs one INFO line at
/// `stage = "auth"` naming the session.
///
/// The session handlers were wired to the same boundary rather than left as a
/// follow-up, so the ticket's `grep 'RPC failed'` covers session CRUD too. The
/// stage is `auth` because the base-path lookup that authorization needs is
/// what discovers the session is gone.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_session_missing_session_logs_info_at_the_auth_stage() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    const MISSING: &str = "b94g-no-such-session";

    let (result, captured) =
        capture(handle_delete_session(&svc, delete_session_request(MISSING))).await;

    let status = result.expect_err("deleting an unknown session must fail the RPC");
    assert_eq!(status.code(), Code::NotFound);

    let event = captured.only_failure();
    assert_common_fields(event, "INFO", "DeleteSession", "auth", "NOT_FOUND");
    assert_field(event, "session_id", MISSING.into());
    assert_field_absent(
        event,
        "base_path",
        "the lookup that would have produced it is the one that failed",
    );
    captured.assert_no_bearer_material();
}

/// A store stall on `ListSessions`' names read answers `UNAVAILABLE` and logs
/// one WARN `RPC failed` at the storage stage — never the server-fault ERROR.
///
/// This pins the log-noise half of b.62m. Before it, the `SMEMBERS` failure
/// was hand-wrapped as `Internal`, so every store blip on a ListSessions poll
/// answered a redacted `INTERNAL` and logged one ERROR `internal error` line —
/// the line operators page on — for what is a retriable backing-store
/// condition. The timeout is fred's real one: every data-pool connection is
/// held behind a server-side `BLPOP`, so the read outlives the fixture's 5 s
/// command timeout. The handler is called directly because the served RPC's
/// `LIVE_EDIT_SESSION_LIST_TIMEOUT_MS` deadline (below the command timeout)
/// would end it as `DEADLINE_EXCEEDED` first; a fast-failing transient error
/// (IO, canceled, …) takes this same path under that deadline.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn list_sessions_store_timeout_logs_one_warn_unavailable_and_no_internal_error() {
    use live_edit::storage::test_support::release_all;

    const BASE_PATH: &str = "/b62m-list-stall";

    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let stalls = tv.stall_data_pool().await;
    let (result, captured) =
        capture(handle_list_sessions(&svc, list_sessions_request(BASE_PATH))).await;
    release_all(stalls).await;

    let status = result.expect_err("a timed-out names read must fail the RPC");
    assert_eq!(
        status.code(),
        Code::Unavailable,
        "a store timeout is a retriable UNAVAILABLE, not INTERNAL; got {status:?}"
    );
    let op_and_kind = format!(
        "SMEMBERS path names index (list) failed ({}): ",
        fred::error::ErrorKind::Timeout.to_str()
    );
    assert!(
        status.message().starts_with(&op_and_kind),
        "the UNAVAILABLE message must start {op_and_kind:?}; got {:?}",
        status.message()
    );

    let event = captured.only_failure();
    assert_common_fields(event, "WARN", "ListSessions", "storage", "UNAVAILABLE");
    assert_field(event, "base_path", BASE_PATH.into());
    let internals = captured.with_message(INTERNAL_ERROR_MSG);
    assert!(
        internals.is_empty(),
        "a store timeout must not produce the server-fault ERROR line; got {internals:?} — \
         full capture:\n{}",
        captured.text
    );
    captured.assert_no_bearer_material();
}

// =============================================================================
// (h) The DeleteSession cascade's per-layer WARN
// =============================================================================
//
// A layer-deletion failure inside the cascade logs its own WARN carrying the
// failing layer and the progress counts. For a permanent (`INTERNAL`) failure
// that WARN is the only record of them: the wire message is redacted, and the
// boundary's ERROR carries the session but not the cascade's progress. For a
// transient failure it sits alongside the boundary's `UNAVAILABLE` line. Both
// halves are pinned here.

/// Add an empty layer named `layer_name` to `session_id`.
#[cfg(feature = "test-fault-injection")]
async fn add_layer(svc: &LiveEditService, session_id: &str, layer_name: &str) {
    let ompf = delta_to_ompf_bytes(&empty_delta(), &[]);
    let stream = streaming_create(vec![
        make_create_header(session_id, layer_name),
        make_create_data(ompf),
    ]);
    handle_create_layer(svc, upload_with_bearer(stream))
        .await
        .unwrap_or_else(|e| panic!("creating layer {layer_name} must succeed: {e}"));
}

/// The single cascade WARN, asserting there is exactly one and that it carries
/// the session, the failing layer and the three progress counts.
fn only_cascade_warn<'a>(
    captured: &'a Captured,
    session_id: &str,
    layer_name: &str,
    deleted_before_failure: usize,
    total: usize,
) -> &'a Value {
    let warns = captured.with_message(CASCADE_WARN_MSG);
    assert_eq!(
        warns.len(),
        1,
        "expected exactly one `{CASCADE_WARN_MSG}` event; got {} — full capture:\n{}",
        warns.len(),
        captured.text
    );
    let event = warns[0];
    assert_eq!(event["level"], "WARN", "wrong level; full event: {event}");
    assert_field(event, "session_id", session_id.into());
    assert_field(event, "layer_name", layer_name.into());
    assert_field(
        event,
        "layers_deleted_before_failure",
        deleted_before_failure.into(),
    );
    assert_field(
        event,
        "layers_remaining",
        (total - deleted_before_failure - 1).into(),
    );
    assert_field(event, "layers_total", total.into());
    event
}

/// A permanent layer failure answers a redacted `INTERNAL`, logs no `RPC
/// failed` and exactly one ERROR `internal error` carrying the session, and the
/// cascade WARN carries the layer and the progress counts the wire message
/// lost.
///
/// The fault is the corrupt layer mapping `delete_session_test.rs` plants: a
/// non-numeric mapping value that `delete_layer` rejects as `Internal`. No
/// fault-injection feature is needed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_session_permanent_cascade_failure_logs_the_cascade_warn_with_counts() {
    const CORRUPT_LAYER: &str = "b27b-corrupt-layer";

    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let (session_id, _object_id) =
        create_session_and_layer(&tv, &svc, "b27b-cascade-permanent", "b27b-good-layer").await;

    tv.client()
        .pool
        .sadd::<(), _, _>(
            tv.key_schema().layers_index(&session_id),
            keys::hex_encode(CORRUPT_LAYER),
        )
        .await
        .expect("SADD of the corrupt layer's index entry must succeed");
    tv.client()
        .pool
        .set::<(), _, _>(
            tv.key_schema().layer_mapping(&session_id, CORRUPT_LAYER),
            "not-a-number",
            None,
            None,
            false,
        )
        .await
        .expect("SET of the corrupt mapping must succeed");

    // The cascade walks the order `list_layer_names` returns; read it now so
    // the expected counts do not depend on which layer SMEMBERS yields first.
    let order = svc
        .layer_store
        .list_layer_names(&session_id)
        .await
        .expect("list_layer_names must succeed");
    assert_eq!(order.len(), 2, "precondition: two layers in the index");
    let failing_idx = order
        .iter()
        .position(|name| name == CORRUPT_LAYER)
        .expect("precondition: the corrupt layer must be listed");

    // The status is dropped inside the capture, so an unreported fault's
    // context-free fallback line would be counted below.
    let ((code, message), captured) = capture(async {
        let status = handle_delete_session(&svc, delete_session_request(&session_id))
            .await
            .expect_err("a corrupt layer mapping must fail the cascade");
        (status.code(), status.message().to_owned())
    })
    .await;

    assert_eq!(
        code,
        Code::Internal,
        "precondition: a permanent layer failure must answer INTERNAL, else this is not the \
         redacted case"
    );
    assert_eq!(message, INTERNAL_ERROR_MSG);

    captured.assert_no_failure("a permanent cascade failure (INTERNAL)");
    let internals = captured.with_message(INTERNAL_ERROR_MSG);
    assert_eq!(
        internals.len(),
        1,
        "a server fault must be reported exactly once; got {} — full capture:\n{}",
        internals.len(),
        captured.text
    );
    assert_eq!(internals[0]["level"], "ERROR");
    assert_field(internals[0], "endpoint", "DeleteSession".into());
    assert_field(internals[0], "session_id", session_id.as_str().into());

    let warn = only_cascade_warn(&captured, &session_id, CORRUPT_LAYER, failing_idx, 2);
    let error = warn["fields"]["error"].as_str().unwrap_or_default();
    assert!(
        error.starts_with("internal error: "),
        "the cascade WARN must carry the unredacted `Internal` detail the wire lost; got \
         {error:?} — full event: {warn}"
    );
    captured.assert_no_bearer_material();
}

/// A transient layer failure answers `UNAVAILABLE`, logs exactly one boundary
/// WARN for it at the storage stage, and the cascade WARN carries the same
/// progress counts the wire message does.
///
/// The fault is a `ServiceError::Aborted` returned by the per-name `LayerStore`
/// seam for the second layer the cascade meets, so the counts are non-trivial
/// (one deleted, none remaining, two total).
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_session_transient_cascade_failure_logs_one_unavailable_failure_and_the_cascade_warn(
) {
    use std::collections::HashMap;
    use std::sync::Mutex;

    use live_edit::error::ServiceError;

    let tv = TestValkey::setup().await;
    let fault_map: Arc<Mutex<HashMap<String, ServiceError>>> = Arc::new(Mutex::new(HashMap::new()));
    let svc =
        build_service_with_layer_store(&tv, layer_store_with_name_faults(&tv, fault_map.clone()));
    let (session_id, _object_id) =
        create_session_and_layer(&tv, &svc, "b27b-cascade-transient", "b27b-layer-0").await;
    add_layer(&svc, &session_id, "b27b-layer-1").await;

    let order = svc
        .layer_store
        .list_layer_names(&session_id)
        .await
        .expect("list_layer_names must succeed");
    assert_eq!(order.len(), 2, "precondition: two layers in the index");
    let target = order[1].clone();
    fault_map.lock().unwrap().insert(
        target.clone(),
        ServiceError::Aborted("transient I/O: synthetic cascade blip".to_owned()),
    );

    let (result, captured) = capture(handle_delete_session(
        &svc,
        delete_session_request(&session_id),
    ))
    .await;

    let status = result.expect_err("a transient layer failure must fail the cascade");
    assert_eq!(
        status.code(),
        Code::Unavailable,
        "a transient layer failure must answer retriable UNAVAILABLE; got {status:?}"
    );
    assert!(
        status.message().contains(&target) && status.message().contains("after 1 of 2"),
        "the UNAVAILABLE message must name the layer and the counts; got {}",
        status.message()
    );

    let event = captured.only_failure();
    assert_common_fields(event, "WARN", "DeleteSession", "storage", "UNAVAILABLE");
    assert_field(event, "session_id", session_id.as_str().into());
    assert!(
        captured.with_message(INTERNAL_ERROR_MSG).is_empty(),
        "a transient failure must not produce the server-fault ERROR line; full capture:\n{}",
        captured.text
    );

    let warn = only_cascade_warn(&captured, &session_id, &target, 1, 2);
    assert_field(
        warn,
        "error",
        "aborted: transient I/O: synthetic cascade blip".into(),
    );
    captured.assert_no_bearer_material();
}

/// A denied `CreateSession` logs at WARN carrying the CANONICAL base path.
///
/// Two contracts in one line. WARN because PRD §7.2 makes authorization
/// denials a security signal — a denial burst is worth noticing, unlike a
/// stale client. And the recorded `base_path` is the canonical spelling
/// (b.uoc), never the caller's: it is the string that was authorized and that
/// storage is keyed under, so an operator correlating a denial with a
/// permission-service decision must see the same value on both sides.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_session_permission_denied_logs_warn_with_the_canonical_base_path() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(DenyAllPermissions));

    let (result, captured) = capture(handle_create_session(
        &svc,
        create_session_request(RAW_BASE_PATH, "b94g-denied-session"),
    ))
    .await;

    let status = result.expect_err("a denied CreateSession must fail the RPC");
    assert_eq!(status.code(), Code::PermissionDenied);
    assert_eq!(
        status.message(),
        "",
        "PermissionDenied carries an empty message by construction — nothing about the token or \
         the decision may reach the client"
    );

    let event = captured.only_failure();
    assert_common_fields(event, "WARN", "CreateSession", "auth", "PERMISSION_DENIED");
    assert_field(event, "base_path", CANONICAL_BASE_PATH.into());
    // Scoped to the boundary event, NOT `Captured::assert_absent_everywhere`:
    // the handler's own `CreateSession canonicalized base_path` INFO line
    // deliberately carries the RAW spelling in its `received` field — the line
    // exists to surface the rewrite to the caller — and it is emitted at INFO,
    // so it is inside this capture. A buffer-wide sweep would therefore fail on
    // a line that is behaving exactly as designed. The contract under test is
    // about what the FAILURE line records, and that is what is asserted.
    assert!(
        !event.to_string().contains(RAW_BASE_PATH),
        "the caller's raw spelling must not appear on the failure line — the canonical one is \
         what was authorized and keyed: {event}"
    );
    assert_field_absent(
        event,
        "session_id",
        "CreateSession's session id is minted by storage, which the denial never reaches",
    );
    captured.assert_no_bearer_material();
}

// =============================================================================
// (i) Every wrapped handler has its Err arm pinned
// =============================================================================
//
// Seven handlers carry the boundary wrapper: CreateLayer, UpdateLayer,
// StatLayer, DeleteLayer, CreateSession, DeleteSession, ListSessions. The
// sections above reach five of them. The two below close the set, so a
// handler whose wrapper was dropped in a refactor cannot pass this file by
// simply not being represented in it. Both take the auth-denial route because
// it is the one failure every handler can be driven into without a
// per-endpoint fixture, which keeps the coverage floor cheap to hold.

/// A denied `CreateLayer` logs one WARN at `stage = "auth"` carrying the full
/// request identity and NEITHER object id.
///
/// The absent ids are the interesting half. `CreateLayer` mints the object id
/// in the very call that could fail, and nothing after that call can fail the
/// RPC — so no `CreateLayer` failure event can ever carry an `object_id`, and
/// it resolves no mapping, so `current_object_id` is structurally impossible
/// too. That second absence is what keeps the level honest: were a
/// `resolved_object_id` ever recorded here, a later NOT_FOUND at `Storage`
/// would read as keyspace damage instead of the client shape it is.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_permission_denied_logs_warn_at_auth_with_neither_object_id() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(DenyAllPermissions));

    // Minted through the store directly, not `create_session_and_layer`: every
    // handler route to a session runs the same permission probe this test is
    // denying, so the fixture cannot be built through one.
    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = session_store
        .create("b94g-denied-create-layer-session", "/stage", "test-user")
        .await
        .expect("session creation must succeed");

    let layer_name = "b94g-denied-layer";
    let ompf = delta_to_ompf_bytes(&empty_delta(), &[]);
    let stream = streaming_create(vec![
        make_create_header(&session_id, layer_name),
        make_create_data(ompf),
    ]);

    let (result, captured) = capture(handle_create_layer(&svc, upload_with_bearer(stream))).await;

    let status = result.expect_err("a denied CreateLayer must fail the RPC");
    assert_eq!(status.code(), Code::PermissionDenied);
    assert_eq!(
        status.message(),
        "",
        "PermissionDenied carries an empty message by construction — nothing about the token or \
         the decision may reach the client"
    );

    let event = captured.only_failure();
    assert_common_fields(event, "WARN", "CreateLayer", "auth", "PERMISSION_DENIED");
    assert_field(event, "session_id", session_id.as_str().into());
    assert_field(event, "layer_name", layer_name.into());
    assert_field(event, "base_path", "/stage".into());
    assert_field(event, "status_message", "".into());
    assert_field_absent(
        event,
        "object_id",
        "CreateLayer mints the object id in the call that could fail, so no failure event can \
         carry one",
    );
    assert_field_absent(
        event,
        "current_object_id",
        "CreateLayer resolves no layer mapping, so it can never claim an object was proven live",
    );
    captured.assert_no_bearer_material();
}

/// A denied `ListSessions` logs one WARN at `stage = "auth"` carrying the
/// CANONICAL base path and no session id.
///
/// `ListSessions` records `base_path` *before* it enters the `Auth` stage —
/// the canonicalization is what produces the string auth is then asked about.
/// This pins that ordering from the outside: an auth-stage denial that arrived
/// with no `base_path` on the line would mean the record had drifted below the
/// probe, leaving an operator correlating a denial burst with the permission
/// service's own decisions nothing to join on.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn list_sessions_permission_denied_logs_warn_with_the_canonical_base_path() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(DenyAllPermissions));

    let (result, captured) = capture(handle_list_sessions(
        &svc,
        list_sessions_request(RAW_BASE_PATH),
    ))
    .await;

    let status = result.expect_err("a denied ListSessions must fail the RPC");
    assert_eq!(status.code(), Code::PermissionDenied);
    assert_eq!(status.message(), "");

    let event = captured.only_failure();
    assert_common_fields(event, "WARN", "ListSessions", "auth", "PERMISSION_DENIED");
    assert_field(event, "base_path", CANONICAL_BASE_PATH.into());
    // Event-scoped like the CreateSession sibling, and for a subtler reason:
    // `ListSessions` logs its canonicalization notice at `debug!` (it is a
    // polled RPC, so an INFO copy per poll would be noise), which sits below
    // this capture's INFO ceiling. A buffer-wide sweep would therefore pass
    // today by accident and start failing the moment that line's level was
    // raised — so the assertion is pinned to the failure line, which is what
    // the contract is actually about.
    assert!(
        !event.to_string().contains(RAW_BASE_PATH),
        "the caller's raw spelling must not appear on the failure line — the canonical one is \
         what was authorized and what the path index was read under: {event}"
    );
    assert_field_absent(
        event,
        "session_id",
        "ListSessions names no session; it is scoped to a base path",
    );
    assert_field_absent(event, "layer_name", "ListSessions names no layer either");
    captured.assert_no_bearer_material();
}

// =============================================================================
// (j) UpdateLayer's failure line carries its store call's summary
// =============================================================================
//
// Spec SDD SR-2.6 / SR-1.2: a failed `UpdateLayer` has no sequence number to
// join a fixture's write to, so its one failure line carries what the store
// call did — `attempt`, `path`, one count per conflict kind, `records`,
// `bytes` — beside the request's session, layer and client ids. A failure
// before the store is called carries none of them.

/// A stale `object_id` fails at `stage = "resolve"`, before the store is
/// called, so the failure line carries no call-summary field at all — not a
/// zero, which would read as "the store made attempts and lost no race".
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_failure_before_the_store_logs_no_call_summary() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let layer_name = "summary-before-store-layer";
    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, "summary-before-store-session", layer_name).await;

    let (result, captured) = capture(handle_update_layer(
        &svc,
        update_request(&session_id, layer_name, object_id + 1),
    ))
    .await;

    assert_eq!(
        result
            .expect_err("a stale object_id must fail the RPC")
            .code(),
        Code::FailedPrecondition,
        "precondition: the RPC must fail at mapping resolution, before the store"
    );
    let event = captured.only_failure();
    assert_field(event, "stage", "resolve".into());
    for field in [
        "attempt",
        "path",
        "watch_conflict",
        "watch_read_moved",
        "seq_fence",
        "cache_diverged",
        "snapshot_fence_wait",
        "records",
        "bytes",
    ] {
        assert_field_absent(event, field, "the RPC failed before calling the store");
    }
}

#[cfg(feature = "test-fault-injection")]
mod update_layer_call_summary {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use fred::clients::Client;
    use futures::FutureExt;

    use live_edit::config::UpdateLayerBackend;
    use live_edit::observability::metrics::UPDATE_LAYER_CONFLICT_KINDS;
    use live_edit::storage::layer::LayerStore;
    use live_edit::storage::watched_txn::WatchWindowHook;

    use super::*;
    use crate::common::owned_delta::delta_with_set_field_inline;
    use crate::storage::snapshot_read_support::{monitor_fence, test_valkey_addr, MonitorTap};
    use crate::storage::update_layer_commit_count_test::watch_commit_sent_bytes;

    /// The retry budget of the store under test: every attempt loses a race,
    /// so the call makes exactly this many and exhausts.
    const MAX_ATTEMPTS: u32 = 3;

    /// An `UpdateLayer` whose every attempt's `EXEC` is cancelled by a write to
    /// its WATCHed header exhausts its retries. Its single failure line carries
    /// `attempt=3`, `path="watch"`, `watch_conflict=3` and 0 for every other
    /// kind, `records` (one per field the delta sets) and `bytes`, beside the
    /// request's session, layer and client ids; no commit line is logged; and
    /// the client still sees the unchanged `UNAVAILABLE` exhaustion message.
    ///
    /// `bytes` is checked against a control: the conflicts are switched off and
    /// the same request is sent again, at the same store state (each conflict
    /// re-set `seq` to its own value, so nothing moved). That commit builds the
    /// same commands — its random commit token is always 32 hex digits and its
    /// microsecond server timestamp keeps its digit count — so its bytes, read
    /// off the store's `MONITOR` feed and logged on its own commit line, are
    /// the bytes the exhausted call's last attempt built.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn update_layer_retry_exhaustion_logs_one_failure_with_the_calls_attempts_path_conflicts_and_commit_size(
    ) {
        let tv = TestValkey::setup().await;
        let plain = build_service(&tv, Arc::new(AllowAllPermissions));
        let layer_name = "summary-exhausted-layer";
        let (session_id, object_id) =
            create_session_and_layer(&tv, &plain, "summary-exhausted-session", layer_name).await;

        let armed = Arc::new(AtomicBool::new(true));
        let conflicts_staged = Arc::new(AtomicUsize::new(0));
        let hook: WatchWindowHook = {
            let writer = tv.client().pool.next().clone();
            let header_key = tv.key_schema().obj_header(&object_id.to_string());
            let armed = Arc::clone(&armed);
            let conflicts_staged = Arc::clone(&conflicts_staged);
            Arc::new(move |_watched: &Client| {
                let writer = writer.clone();
                let header_key = header_key.clone();
                let armed = Arc::clone(&armed);
                let conflicts_staged = Arc::clone(&conflicts_staged);
                async move {
                    if !armed.load(Ordering::SeqCst) {
                        return;
                    }
                    // Re-set `seq` to its own value: the write cancels the
                    // EXEC, and the next attempt meets the same state.
                    let seq: String = writer
                        .hget(&header_key, "seq")
                        .await
                        .expect("HGET seq must succeed");
                    let _: () = writer
                        .hset(&header_key, [("seq", seq)])
                        .await
                        .expect("HSET seq must succeed");
                    conflicts_staged.fetch_add(1, Ordering::SeqCst);
                }
                .boxed()
            })
        };
        let store = Arc::new(
            LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
                .with_update_layer_backend(UpdateLayerBackend::Txn)
                .with_update_layer_retry_max_attempts(MAX_ATTEMPTS)
                .with_watch_window_hook_for_test(hook),
        );
        let svc = build_service_with_layer_store(&tv, store);
        let delta = delta_with_set_field_inline(1, "color", vec![0x63], 0);
        let records = delta.set_fields.len() as u64;
        let ompf = delta_to_ompf_bytes(&delta, &[]);
        let request =
            || update_request_with_chunks(&session_id, layer_name, object_id, vec![ompf.clone()]);

        let (result, captured) = capture(handle_update_layer(&svc, request())).await;

        // The wire contract, unchanged by the summary.
        let status = result.expect_err("an exhausted UpdateLayer must fail the RPC");
        assert_eq!(status.code(), Code::Unavailable);
        assert_eq!(
            status.message(),
            format!(
                "update_layer retriable failures, {MAX_ATTEMPTS} attempts exhausted (last cause: \
                 concurrent write conflict, retry)"
            ),
            "the client-visible message must be the exhaustion message, with no summary field \
             in it"
        );
        assert_eq!(
            conflicts_staged.load(Ordering::SeqCst),
            MAX_ATTEMPTS as usize,
            "precondition: every attempt's EXEC must have been cancelled"
        );

        // Exactly one completion line, and no commit line.
        assert_eq!(
            captured.completion_lines().len(),
            1,
            "one failed RPC, one completion line; full capture:\n{}",
            captured.text
        );
        assert!(
            captured.with_message("UpdateLayer committed").is_empty(),
            "nothing committed, so no commit line; full capture:\n{}",
            captured.text
        );
        let event = captured.only_failure();
        assert_common_fields(event, "WARN", "UpdateLayer", "storage", "UNAVAILABLE");
        assert_field(event, "session_id", session_id.as_str().into());
        assert_field(event, "layer_name", layer_name.into());
        assert_field(event, "client_id", B94G_CLIENT_ID.into());
        assert_field(event, "object_id", object_id.into());
        assert_field(event, "status_message", status.message().into());
        assert_field(event, "attempt", MAX_ATTEMPTS.into());
        assert_field(event, "path", "watch".into());
        for kind in UPDATE_LAYER_CONFLICT_KINDS {
            let want = if *kind == "watch_conflict" {
                u64::from(MAX_ATTEMPTS)
            } else {
                0
            };
            assert_field(event, kind, want.into());
        }
        assert_field(event, "records", records.into());
        captured.assert_no_bearer_material();

        // Control: the same request, uncontended, under MONITOR.
        armed.store(false, Ordering::SeqCst);
        let mut tap = MonitorTap::start(&test_valkey_addr()).await;
        let (control, control_captured) = capture(handle_update_layer(&svc, request())).await;
        let fence = monitor_fence(tv.prefix());
        let _: Option<String> = tv
            .client()
            .pool
            .get(&fence)
            .await
            .expect("the fence GET must succeed");
        let feed = tap.drain_until(&fence).await;
        let sequence_num = control
            .expect("the uncontended request must commit")
            .into_inner()
            .sequence_num;
        let sent = watch_commit_sent_bytes(
            &feed,
            &tv.key_schema().layer_stream(object_id),
            &format!("{sequence_num}-0"),
        );
        let commit_line = control_captured.only("UpdateLayer committed");
        assert_field(commit_line, "bytes", sent.into());
        assert_field(commit_line, "records", records.into());

        assert_field(event, "bytes", sent.into());
    }

    /// T3 / M2: an `UpdateLayer` whose snapshot-fence wait reaches its cap is
    /// logged ONCE — the RPC's completion line — on every commit path: the
    /// fast path, the structural route (a cold start, `structural_full`) and
    /// the WATCH path. The store's cap is 1 ms, so the first refusal already
    /// cannot sleep: the client sees the retryable `UNAVAILABLE` naming the
    /// waited span, the cap, the object and the path; the completion line
    /// carries it at WARN with the call's `snapshot_fence_wait` conflict and
    /// path; no other WARN or ERROR is emitted (the wait helper logs nothing of
    /// its own); `live_edit_update_layer_fence_timeout_total` ticks once; and
    /// nothing is written.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn update_layer_reaching_its_fence_wait_cap_logs_one_line_on_every_path() {
        use fred::interfaces::HashesInterface;
        use live_edit::observability::metrics::test_util::{render, sample_value};

        let cases: [(&str, UpdateLayerBackend, oodf_ffi::OwnedDelta, &str); 3] = [
            (
                "fast",
                UpdateLayerBackend::Lua,
                delta_with_set_field_inline(1, "color", vec![0x63], 0),
                "fast",
            ),
            (
                "structural",
                UpdateLayerBackend::Lua,
                oodf_ffi::OwnedDelta {
                    struct_commands: vec![oodf_ffi::StructCommand::CreateSection(
                        oodf_ffi::CommandCreateSection {
                            parent_id: 0,
                            section_id: 100,
                            section_name: "child".to_owned(),
                            section_type: 1,
                        },
                    )],
                    ..Default::default()
                },
                "structural_full",
            ),
            (
                "watch",
                UpdateLayerBackend::Txn,
                delta_with_set_field_inline(1, "color", vec![0x63], 0),
                "watch",
            ),
        ];
        for (case, backend, delta, path) in cases {
            let tv = TestValkey::setup().await;
            let plain = build_service(&tv, Arc::new(AllowAllPermissions));
            let layer_name = format!("fence-cap-{case}-layer");
            let (session_id, object_id) = create_session_and_layer(
                &tv,
                &plain,
                &format!("fence-cap-{case}-session"),
                &layer_name,
            )
            .await;
            let metrics = Arc::new(crate::common::metrics_scrape_only("fence-cap-log"));
            let store = Arc::new(
                LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
                    .with_update_layer_backend(backend)
                    .with_fence_wait_deadline(
                        std::time::Duration::from_millis(1),
                        Arc::new(|| None),
                    )
                    .with_metrics(Arc::clone(&metrics)),
            );
            let svc = build_service_with_layer_store(&tv, store);
            // A reader lease live on any store clock (its deadline is in the
            // year 5138).
            let oid = object_id.to_string();
            let _: () = tv
                .client()
                .pool
                .hset(
                    tv.key_schema().obj_snapshot_fence(&oid),
                    vec![("r:cap", "99999999999999"), ("epoch_start", "0")],
                )
                .await
                .expect("staging the fence must succeed");
            let header_key = tv.key_schema().obj_header(&oid);
            let header_before: std::collections::HashMap<String, String> = tv
                .client()
                .pool
                .hgetall(&header_key)
                .await
                .expect("HGETALL must succeed");
            let ompf = delta_to_ompf_bytes(&delta, &[]);

            let (result, captured) = capture(handle_update_layer(
                &svc,
                update_request_with_chunks(&session_id, &layer_name, object_id, vec![ompf]),
            ))
            .await;

            let status = result.expect_err("a capped fence wait must fail the RPC");
            assert_eq!(status.code(), Code::Unavailable, "{case}");
            assert_eq!(
                status.message(),
                format!(
                    "update_layer waited 0 ms of its 1 ms snapshot-fence wait cap on object \
                     {object_id} ({path}); nothing was written, retry"
                ),
                "{case}: the client-visible text names the wait, the cap, the object and the \
                 path"
            );
            crate::common::refusal_detail::assert_refusal_detail(
                &status,
                live_edit::error::refusal_reason::SNAPSHOT_FENCE_TIMEOUT,
                &[],
                live_edit::error::CAPACITY_RETRY_DELAY,
            );
            assert_eq!(
                captured.completion_lines().len(),
                1,
                "{case}: one failed RPC, one completion line; full capture:\n{}",
                captured.text
            );
            let loud: Vec<&Value> = captured
                .events
                .iter()
                .filter(|e| e["level"] == "WARN" || e["level"] == "ERROR")
                .collect();
            assert_eq!(
                loud.len(),
                1,
                "{case}: the completion line is the only WARN / ERROR — the fence wait logs \
                 nothing of its own; full capture:\n{}",
                captured.text
            );
            let event = captured.only_failure();
            assert_common_fields(event, "WARN", "UpdateLayer", "storage", "UNAVAILABLE");
            assert_field(event, "status_message", status.message().into());
            assert_field(event, "path", path.into());
            assert_field(event, "snapshot_fence_wait", 1_u64.into());
            assert_eq!(
                sample_value(
                    &render(&metrics),
                    "live_edit_update_layer_fence_timeout_total",
                    &[]
                ),
                Some(1.0),
                "{case}: the cap ticks the timeout counter once"
            );
            let header_after: std::collections::HashMap<String, String> = tv
                .client()
                .pool
                .hgetall(&header_key)
                .await
                .expect("HGETALL must succeed");
            assert_eq!(header_after, header_before, "{case}: nothing written");
        }
    }
}
