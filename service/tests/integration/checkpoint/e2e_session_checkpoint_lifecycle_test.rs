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

//! End-to-end test for Epic AC #1 — session-checkpoint writes triggered by
//! lifecycle events (Subtask `t3.fgj.7v.o7.hr` under Task `t2.fgj.7v.o7`,
//! Epic `t1.fgj.7v`, Bee `b.fgj`).
//!
//! Task 4 (`t2.fgj.7v.p9`) ships the dispatcher primitive
//! (`SessionCheckpointWriter::enqueue_session_checkpoint_write` /
//! `enqueue_session_checkpoint_delete`) and wires it into the four lifecycle
//! call sites: `SessionStore::create` (put), `LayerStore::create_layer`
//! (put), `LayerStore::delete_layer` (put), and `SessionStore::delete`
//! (delete). Task 4's own integration tests pin the dispatch-routing
//! (`lifecycle_delete_session_test.rs`) and the error metric
//! (`lifecycle_session_writes_error_test.rs`) using counting / failing
//! adapters; neither verifies the on-disk **payload** the writes deposit
//! end-to-end through real RPC entry points.
//!
//! This file closes that gap. Each test drives a real gRPC handler
//! (`handle_create_session`, `handle_create_layer`, `handle_delete_layer`,
//! `handle_delete_session`) through the [`CheckpointHarness`] wiring, waits
//! for the dispatcher's fire-and-forget put-task to land on the LocalDisk
//! adapter, then decodes the resulting blob with
//! [`SessionCheckpoint::from_bytes`] and asserts every field is the value
//! Task 1's struct exposes — `schema_version`, `session_id`, `name`,
//! `base_path`, `created_by`, `created_at`, `layer_mappings`, `layers_index`,
//! `path_name_members`, `path_index_session_id_member`.
//!
//! The `checkpoint_writes_total{kind="session", result="success"}` metric
//! is also asserted at each step: the four lifecycle events covered here
//! (CreateSession, CreateLayer, DeleteLayer, DeleteSession) all enqueue
//! through the same dispatcher and all four route their `Ok(_)` outcomes
//! through the shared `handle_put_outcome` shim
//! (`src/checkpoint/session_writer.rs:204`), so the success cell ticks on
//! every clean lifecycle event — including DeleteSession's delete-path.
//! The asymmetry to verify on the DeleteSession path is therefore NOT a
//! missing metric tick but the post-call `Ok(None)` from
//! `get_session_checkpoint`: a regression that wired a `_write` instead of
//! a `_delete` would leave the blob in place. See
//! `delete_session_calls_delete_session_checkpoint` below.
//!
//! Each test calls [`setup_checkpoint_harness`] fresh so the metric
//! snapshots and the on-disk checkpoint store state are independent. Wait
//! cadences are bounded-poll (no fixed sleeps) per
//! `docs-internal/test-writing-guide.md` `## Synchronization Patterns`.
//!
//! Sandbox env note: this test uses `TestValkey::setup`, which reads
//! `TEST_VALKEY_URL`. Under the project sandbox the URL must include the
//! password and `host.docker.internal` host; see CLAUDE.md.

use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant};

use live_edit::checkpoint::session_payload::SESSION_CHECKPOINT_SCHEMA_VERSION;
use live_edit::checkpoint::{CheckpointStore, SessionCheckpoint};
use live_edit::grpc::layer::{handle_create_layer, handle_delete_layer};
use live_edit::grpc::proto::{CreateSessionRequest, DeleteLayerRequest, DeleteSessionRequest};
use live_edit::grpc::session::{handle_create_session, handle_delete_session};
use live_edit::observability::metrics::SharedMetrics;
use live_edit::storage::keys::KeySchema;

use super::helpers::*;
use crate::common::owned_delta::{delta_to_ompf_bytes, empty_delta};

// Stable values the four tests below use. The session name is unique per
// test so concurrent execution against a shared Valkey instance does not
// collide on the path-name uniqueness index — the AC #1 metric and payload
// assertions are independent of the live path-index, but a leftover ghost
// member from an earlier failed run would still flake CreateSession.
const SESSION_BASE_PATH: &str = "/stage/e2e-lifecycle";
const SESSION_NAME_META: &str = "e2e-lifecycle-meta-session";
const SESSION_NAME_CREATE_LAYER: &str = "e2e-lifecycle-create-layer-session";
const SESSION_NAME_DELETE_LAYER: &str = "e2e-lifecycle-delete-layer-session";
const SESSION_NAME_DELETE_SESSION: &str = "e2e-lifecycle-delete-session-session";
const LAYER_NAME: &str = "lifecycle-layer";
// The harness wires `StubUserinfo` which returns `display_name = "Test User"`.
// `SessionStore::create_once` writes this into the `le:{sid}:sess.created_by`
// hash field; the assembler reads it back through the same field name. Pinned
// here so a future widening of the stub surfaces as a single-point fix.
const TEST_USER_DISPLAY_NAME: &str = "Test User";

// ── Streaming framing helpers — imported from
// `tests/integration/checkpoint/helpers.rs` via `use super::helpers::*;`
// (b.dzd Phase 1 deduplication). ─────────────────────────────────────────

// ── Metric helpers ──────────────────────────────────────────────────────

/// Read the live `live_edit_checkpoint_writes_total{result, kind="session"}`
/// counter value. The metric is `CounterVec` (f64); cells are NOT labelled
/// by service (unlike `checkpoint_recovery_attempts_total`), so the read
/// shape here is intentionally narrower than [`session_recovery_metric`].
fn session_checkpoint_writes_metric(metrics: &SharedMetrics, result: &str) -> f64 {
    live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(metrics),
        "live_edit_checkpoint_writes_total",
        &[("result", result), ("kind", "session")],
    )
    .unwrap_or(0.0)
}

// ── Polling helpers ─────────────────────────────────────────────────────

/// Bounded-poll [`CheckpointStore::get_session_checkpoint`] until the stored
/// bytes differ from `previous`, returning the new blob bytes on success.
/// Panics with a diagnostic on deadline expiry.
///
/// Load-bearing for the three lifecycle scenarios that build on top of a
/// pre-existing checkpoint blob: CreateLayer extends the previous
/// CreateSession-time blob with a new layer-mapping; DeleteLayer shrinks the
/// previous CreateLayer-time blob. The writer's enqueue path is
/// fire-and-forget, so a successful RPC return does not guarantee the put-task
/// has landed — and the put bytes for the NEW state may or may not have
/// finished landing by the time we observe a `Some(_)` from the store. Polling
/// on byte-inequality is the precise synchronization primitive: it converges
/// only after the writer has flushed the NEW state.
async fn wait_for_session_checkpoint_bytes_change(
    store: &std::sync::Arc<dyn CheckpointStore>,
    session_id: &str,
    previous: &bytes::Bytes,
    deadline_secs: u64,
) -> bytes::Bytes {
    let deadline = Instant::now() + Duration::from_secs(deadline_secs);
    let poll_interval = Duration::from_millis(50);
    loop {
        match store.get_session_checkpoint(session_id).await {
            Ok(Some(blob)) if blob.bytes != *previous => return blob.bytes,
            Ok(_) => {
                // Either no blob landed yet, or the on-disk bytes still match
                // the pre-call snapshot — the writer's put-task has not yet
                // emitted the NEW state. Keep polling.
            }
            Err(err) => {
                panic!(
                    "get_session_checkpoint failed during wait_for_session_checkpoint_bytes_change \
                     (session_id={session_id}, deadline_secs={deadline_secs}): {err}"
                );
            }
        }
        if Instant::now() >= deadline {
            panic!(
                "wait_for_session_checkpoint_bytes_change timed out after {deadline_secs}s \
                 waiting for new bytes for session_id={session_id}. The lifecycle event's \
                 enqueue_session_checkpoint_write either did not fire (wiring regression) \
                 or its spawned put-task is starved."
            );
        }
        tokio::time::sleep(poll_interval).await;
    }
}

/// Bounded-poll the `checkpoint_writes_total{result, kind="session"}` cell
/// until it reaches `expected`, panicking on deadline expiry.
///
/// Mirrors the polling rationale in
/// [`wait_for_session_checkpoint_bytes_change`]: the writer's success-metric
/// increment fires inside the spawned put-task, AFTER the adapter returns. A
/// caller that read the metric immediately after a `wait_for_session_checkpoint`
/// would race with the increment site and observe stale values.
async fn wait_for_writes_metric(
    metrics: &SharedMetrics,
    result: &str,
    expected: f64,
    deadline_secs: u64,
) {
    let deadline = Instant::now() + Duration::from_secs(deadline_secs);
    let poll_interval = Duration::from_millis(50);
    loop {
        let observed = session_checkpoint_writes_metric(metrics, result);
        if (observed - expected).abs() < f64::EPSILON {
            return;
        }
        if Instant::now() >= deadline {
            panic!(
                "wait_for_writes_metric timed out after {deadline_secs}s waiting for \
                 checkpoint_writes_total{{kind=\"session\", result={result:?}}} to reach \
                 {expected} (last observed {observed})"
            );
        }
        tokio::time::sleep(poll_interval).await;
    }
}

/// Decode the current session-checkpoint blob for `session_id` and return
/// the typed [`SessionCheckpoint`]. Panics on absent blob or decode failure
/// — the caller is responsible for landing a blob first via either
/// [`wait_for_session_checkpoint`] or
/// [`wait_for_session_checkpoint_bytes_change`].
async fn read_session_checkpoint(
    store: &std::sync::Arc<dyn CheckpointStore>,
    session_id: &str,
) -> (SessionCheckpoint, bytes::Bytes) {
    let blob = store
        .get_session_checkpoint(session_id)
        .await
        .expect("get_session_checkpoint must succeed against LocalDisk adapter")
        .expect("a session-checkpoint blob MUST be present at this point in the test");
    let decoded = SessionCheckpoint::from_bytes(&blob.bytes)
        .expect("SessionCheckpoint::from_bytes must succeed on a well-formed payload");
    (decoded, blob.bytes)
}

// ── Composite drivers ───────────────────────────────────────────────────

/// CreateSession through `handle_create_session`. Returns the minted
/// `session_id`. Used by every test in this file as the leading lifecycle
/// step.
async fn drive_create_session(harness: &CheckpointHarness, session_name: &str) -> String {
    let response = handle_create_session(
        &harness.svc,
        make_unary_token_request(CreateSessionRequest {
            path: SESSION_BASE_PATH.to_owned(),
            name: session_name.to_owned(),
        }),
    )
    .await
    .expect("CreateSession must succeed against an AllowAllPermissions harness");
    let session_id = response.into_inner().session_id;
    assert!(
        !session_id.is_empty(),
        "CreateSession must mint a non-empty session_id"
    );
    session_id
}

/// CreateLayer through `handle_create_layer`. Returns the minted
/// `object_id`.
async fn drive_create_layer(
    harness: &CheckpointHarness,
    session_id: &str,
    layer_name: &str,
) -> u64 {
    let ompf_bytes = delta_to_ompf_bytes(&empty_delta(), &[]);
    let stream = streaming_create(vec![
        make_create_header(session_id, layer_name),
        make_create_data(ompf_bytes),
    ]);
    let response = handle_create_layer(&harness.svc, make_streaming_token_request(stream))
        .await
        .expect("CreateLayer must succeed for harness setup");
    let object_id = response.into_inner().object_id;
    assert!(object_id > 0, "CreateLayer must mint a non-zero object_id");
    object_id
}

// =============================================================================
// Scenario 1 — CreateSession lands a SessionCheckpoint with every meta field.
// =============================================================================

/// Pin AC #1's `CreateSession` half: the lifecycle hook fires
/// `enqueue_session_checkpoint_write`, the dispatcher's spawned put-task
/// lands a blob on the LocalDisk adapter, and that blob decodes to a
/// `SessionCheckpoint` whose every meta field equals the input passed to
/// `handle_create_session`.
///
/// Field-by-field coverage:
///
/// - `schema_version == SESSION_CHECKPOINT_SCHEMA_VERSION` — the assembler
///   stamps the current envelope version, not the `default_schema_version`
///   fallback.
/// - `session_id == minted session_id` — provenance.
/// - `name == SESSION_NAME_META` — the `le:{sid}:sess.name` hash field.
/// - `base_path == SESSION_BASE_PATH` — the `le:{sid}:sess.base_path` field.
/// - `created_by == TEST_USER_DISPLAY_NAME` — pinned by `StubUserinfo`.
/// - `created_at > 0` — the `now_micros()` capture (we cannot pin a literal
///   value; the assembler's contract is "non-zero").
/// - `layer_mappings.is_empty()` — no CreateLayer yet, so the per-session
///   layer-mapping table is empty.
/// - `layers_index.is_empty()` — same: no layers, no layers-index members.
/// - `path_name_members == {session_name\0session_id}` — the assembler
///   emits EXACTLY ONE path-name composite per session, irrespective of
///   layer count. The composite mirrors what `SessionStore::create_once`
///   writes on the live path (`src/storage/session.rs:308`). Emitting a
///   per-LAYER composite (a pre-fix shape) broke `ListSessions` (returned
///   N duplicate entries), `create_once`'s duplicate-name check (any
///   composite would falsely match the new name), and `DeleteSession`'s
///   SREM (only the `session_name\0session_id` composite is removed).
/// - `path_index_session_id_member == session_id` — the legacy
///   session-id-only path-index member.
///
/// Metric assertion: `checkpoint_writes_total{kind="session",
/// result="success"}` increments by EXACTLY 1.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn create_session_enqueues_session_checkpoint_with_meta_fields() {
    let harness = setup_checkpoint_harness(1).await;

    let success_pre = session_checkpoint_writes_metric(&harness.metrics, "success");
    let error_pre = session_checkpoint_writes_metric(&harness.metrics, "error");

    let session_id = drive_create_session(&harness, SESSION_NAME_META).await;

    wait_for_session_checkpoint(&harness.store, &session_id, 10).await;
    wait_for_writes_metric(&harness.metrics, "success", success_pre + 1.0, 10).await;

    let (checkpoint, _bytes) = read_session_checkpoint(&harness.store, &session_id).await;

    // Schema version + provenance.
    assert_eq!(
        checkpoint.schema_version, SESSION_CHECKPOINT_SCHEMA_VERSION,
        "schema_version MUST match SESSION_CHECKPOINT_SCHEMA_VERSION; got {} vs expected {}",
        checkpoint.schema_version, SESSION_CHECKPOINT_SCHEMA_VERSION
    );
    assert_eq!(
        checkpoint.session_id, session_id,
        "checkpoint.session_id MUST equal the minted session_id"
    );

    // Session-meta hash fields (`name`, `base_path`, `created_by`,
    // `created_at`). The first three are pinned to the literal inputs the
    // RPC carried; `created_at` is asserted as non-zero (the assembler's
    // contract is `now_micros()` at CreateSession time, which we cannot
    // pin to a literal value here).
    assert_eq!(
        checkpoint.name, SESSION_NAME_META,
        "checkpoint.name MUST equal the CreateSession `name` input"
    );
    assert_eq!(
        checkpoint.base_path, SESSION_BASE_PATH,
        "checkpoint.base_path MUST equal the CreateSession `path` input"
    );
    assert_eq!(
        checkpoint.created_by, TEST_USER_DISPLAY_NAME,
        "checkpoint.created_by MUST equal the StubUserinfo display_name"
    );
    assert!(
        checkpoint.created_at > 0,
        "checkpoint.created_at MUST be a non-zero now_micros() capture; got {}",
        checkpoint.created_at
    );

    // No layers yet → layer_mappings and layers_index are empty.
    assert_eq!(
        checkpoint.layer_mappings,
        BTreeMap::<String, u64>::new(),
        "post-CreateSession layer_mappings MUST be empty (no CreateLayer yet)"
    );
    assert_eq!(
        checkpoint.layers_index,
        BTreeSet::<String>::new(),
        "post-CreateSession layers_index MUST be empty (no CreateLayer yet)"
    );

    // Path-name composite: exactly one member — `session_name\0session_id`.
    // The assembler emits one composite per session (matching the live
    // `SessionStore::create_once` write), irrespective of layer count.
    let expected_session_composite = KeySchema::path_name_member(SESSION_NAME_META, &session_id);
    assert_eq!(
        checkpoint.path_name_members,
        BTreeSet::from([expected_session_composite]),
        "post-CreateSession path_name_members MUST contain exactly the \
         session-name composite (`KeySchema::path_name_member(SESSION_NAME_META, \
         session_id)`); got {:?}",
        checkpoint.path_name_members
    );

    // Legacy session-id-only path-index member.
    assert_eq!(
        checkpoint.path_index_session_id_member, session_id,
        "path_index_session_id_member MUST equal the session_id"
    );

    // Success metric incremented by EXACTLY 1. No error tick.
    assert_eq!(
        session_checkpoint_writes_metric(&harness.metrics, "success"),
        success_pre + 1.0,
        "checkpoint_writes_total{{kind=\"session\", result=\"success\"}} MUST have \
         incremented by exactly 1 on CreateSession's enqueue"
    );
    assert_eq!(
        session_checkpoint_writes_metric(&harness.metrics, "error"),
        error_pre,
        "checkpoint_writes_total{{kind=\"session\", result=\"error\"}} MUST NOT tick \
         on a successful put"
    );

    harness.cancel.cancel();
    let _ = harness.loop_handle.await;
}

// =============================================================================
// Scenario 2 — CreateLayer extends the checkpoint with the new layer-mapping.
// =============================================================================

/// Pin AC #1's `CreateLayer` half: a successful `handle_create_layer` call
/// AFTER a CreateSession fires a second enqueue_session_checkpoint_write
/// through `LayerStore::create_layer`'s lifecycle hook. The dispatcher's
/// second put-task lands a NEW blob whose `layer_mappings` includes the new
/// layer name → object_id, and whose `layers_index` includes the new layer
/// name — i.e. the post-CreateLayer state is faithfully reflected in the
/// checkpoint payload.
///
/// We capture the pre-CreateLayer blob bytes and bounded-poll on
/// byte-inequality (`wait_for_session_checkpoint_bytes_change`) so the
/// assertion phase sees the NEW state, not the previous CreateSession-time
/// snapshot. Without this, the test would race the writer's put-task and
/// flake under load.
///
/// Metric assertion: the success cell increments to `success_pre + 2.0`
/// (one for CreateSession, one for CreateLayer).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn create_layer_extends_session_checkpoint_with_mapping() {
    let harness = setup_checkpoint_harness(1).await;

    let success_pre = session_checkpoint_writes_metric(&harness.metrics, "success");
    let error_pre = session_checkpoint_writes_metric(&harness.metrics, "error");

    // CreateSession + wait for the first blob.
    let session_id = drive_create_session(&harness, SESSION_NAME_CREATE_LAYER).await;
    wait_for_session_checkpoint(&harness.store, &session_id, 10).await;
    wait_for_writes_metric(&harness.metrics, "success", success_pre + 1.0, 10).await;

    // Snapshot pre-CreateLayer blob bytes — the polling primitive needs a
    // baseline to test byte-inequality against.
    let (pre_checkpoint, pre_bytes) = read_session_checkpoint(&harness.store, &session_id).await;
    assert!(
        pre_checkpoint.layer_mappings.is_empty(),
        "pre-CreateLayer layer_mappings MUST be empty (otherwise the test \
         baseline is wrong)"
    );

    // CreateLayer + wait for the NEW blob and the NEW metric tick.
    let object_id = drive_create_layer(&harness, &session_id, LAYER_NAME).await;
    let _new_bytes =
        wait_for_session_checkpoint_bytes_change(&harness.store, &session_id, &pre_bytes, 10).await;
    wait_for_writes_metric(&harness.metrics, "success", success_pre + 2.0, 10).await;

    let (checkpoint, _bytes) = read_session_checkpoint(&harness.store, &session_id).await;

    // Session-meta unchanged through CreateLayer.
    assert_eq!(checkpoint.session_id, session_id);
    assert_eq!(checkpoint.name, SESSION_NAME_CREATE_LAYER);
    assert_eq!(checkpoint.base_path, SESSION_BASE_PATH);
    assert_eq!(checkpoint.created_by, TEST_USER_DISPLAY_NAME);

    // layer_mappings now carries EXACTLY one entry: the new layer name →
    // the fresh object_id.
    let expected_mappings: BTreeMap<String, u64> =
        BTreeMap::from([(LAYER_NAME.to_string(), object_id)]);
    assert_eq!(
        checkpoint.layer_mappings, expected_mappings,
        "post-CreateLayer layer_mappings MUST contain exactly one entry \
         ({LAYER_NAME} → {object_id}); got {:?}",
        checkpoint.layer_mappings
    );

    // layers_index now contains EXACTLY the new layer name. The assembler
    // emits the unhexed name (matches `layer_mappings.keys()`); a regression
    // that emitted the hex-encoded form would diverge here.
    let expected_layers_index: BTreeSet<String> = BTreeSet::from([LAYER_NAME.to_string()]);
    assert_eq!(
        checkpoint.layers_index, expected_layers_index,
        "post-CreateLayer layers_index MUST contain exactly the new layer name; got {:?}",
        checkpoint.layers_index
    );

    // Path-name composite: exactly one member — the
    // `session_name\0session_id` composite. The assembler emits one
    // composite per session, irrespective of layer count, matching the
    // live `SessionStore::create_once` write at
    // `src/storage/session.rs:308`. Adding a layer does NOT mint a new
    // composite; the set stays at exactly one member. A regression that
    // bypassed `add_path_name_member` and inlined a `format!("{name}:{sid}")`
    // would fail this assertion (the helper uses NUL, not colon, as
    // separator).
    let expected_composite = KeySchema::path_name_member(SESSION_NAME_CREATE_LAYER, &session_id);
    assert_eq!(
        checkpoint.path_name_members,
        BTreeSet::from([expected_composite]),
        "post-CreateLayer path_name_members MUST contain exactly the session-name \
         composite (`KeySchema::path_name_member(SESSION_NAME_CREATE_LAYER, \
         session_id)`); got {:?}",
        checkpoint.path_name_members
    );
    assert_eq!(checkpoint.path_index_session_id_member, session_id);

    // Success metric +2 (one each for CreateSession and CreateLayer), no error.
    assert_eq!(
        session_checkpoint_writes_metric(&harness.metrics, "success"),
        success_pre + 2.0,
        "checkpoint_writes_total{{kind=\"session\", result=\"success\"}} MUST be \
         success_pre + 2 after CreateSession + CreateLayer"
    );
    assert_eq!(
        session_checkpoint_writes_metric(&harness.metrics, "error"),
        error_pre,
        "checkpoint_writes_total{{kind=\"session\", result=\"error\"}} MUST NOT tick"
    );

    harness.cancel.cancel();
    let _ = harness.loop_handle.await;
}

// =============================================================================
// Scenario 3 — DeleteLayer shrinks the checkpoint's layer-mappings.
// =============================================================================

/// Pin AC #1's `DeleteLayer` half: after CreateSession + CreateLayer have
/// landed the extended-state blob, `handle_delete_layer` fires a third
/// enqueue_session_checkpoint_write through `LayerStore::delete_layer`'s
/// lifecycle hook. The new blob's `layer_mappings` and `layers_index` are
/// both EMPTY (the only layer was just deleted), but the session itself is
/// still present in the blob — DeleteLayer does NOT cascade into a session
/// delete.
///
/// This case is the cleanest contrast to Scenario 4 (DeleteSession), which
/// goes through `delete_session_checkpoint` (no put, no metric tick).
/// Scenario 3 stays on the put-path and increments the success cell.
///
/// Metric assertion: success cell reaches `success_pre + 3.0` (CreateSession +
/// CreateLayer + DeleteLayer).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn delete_layer_shrinks_session_checkpoint_mapping() {
    let harness = setup_checkpoint_harness(1).await;

    let success_pre = session_checkpoint_writes_metric(&harness.metrics, "success");
    let error_pre = session_checkpoint_writes_metric(&harness.metrics, "error");

    let session_id = drive_create_session(&harness, SESSION_NAME_DELETE_LAYER).await;
    wait_for_session_checkpoint(&harness.store, &session_id, 10).await;
    wait_for_writes_metric(&harness.metrics, "success", success_pre + 1.0, 10).await;

    let (_pre_cs, pre_cs_bytes) = read_session_checkpoint(&harness.store, &session_id).await;
    let _object_id = drive_create_layer(&harness, &session_id, LAYER_NAME).await;
    let post_cl_bytes =
        wait_for_session_checkpoint_bytes_change(&harness.store, &session_id, &pre_cs_bytes, 10)
            .await;
    wait_for_writes_metric(&harness.metrics, "success", success_pre + 2.0, 10).await;

    // DeleteLayer through the real handler. Lifecycle hook fires; the
    // dispatcher's third put-task lands the shrunken state.
    let _response = handle_delete_layer(
        &harness.svc,
        make_unary_token_request(DeleteLayerRequest {
            session_id: session_id.clone(),
            layer_name: LAYER_NAME.to_owned(),
        }),
    )
    .await
    .expect("DeleteLayer must succeed against a freshly-created layer");

    let _post_dl_bytes =
        wait_for_session_checkpoint_bytes_change(&harness.store, &session_id, &post_cl_bytes, 10)
            .await;
    wait_for_writes_metric(&harness.metrics, "success", success_pre + 3.0, 10).await;

    let (checkpoint, _bytes) = read_session_checkpoint(&harness.store, &session_id).await;

    // Session itself remains.
    assert_eq!(checkpoint.session_id, session_id);
    assert_eq!(checkpoint.name, SESSION_NAME_DELETE_LAYER);
    assert_eq!(checkpoint.base_path, SESSION_BASE_PATH);
    assert_eq!(checkpoint.created_by, TEST_USER_DISPLAY_NAME);
    assert!(checkpoint.created_at > 0);

    // layer_mappings and layers_index are BOTH empty post-DeleteLayer.
    assert_eq!(
        checkpoint.layer_mappings,
        BTreeMap::<String, u64>::new(),
        "post-DeleteLayer layer_mappings MUST be empty"
    );
    assert_eq!(
        checkpoint.layers_index,
        BTreeSet::<String>::new(),
        "post-DeleteLayer layers_index MUST be empty"
    );

    // Path-name composite: still exactly the session-name composite. The
    // assembler emits one composite per session, not per layer — deleting
    // the only layer does NOT remove the composite. A regression that
    // emitted a per-layer composite would fail here (the set would drop
    // to empty when the layer disappeared).
    let expected_composite = KeySchema::path_name_member(SESSION_NAME_DELETE_LAYER, &session_id);
    assert_eq!(
        checkpoint.path_name_members,
        BTreeSet::from([expected_composite]),
        "post-DeleteLayer path_name_members MUST still contain exactly the \
         session-name composite (the session itself persists; only the layer \
         was deleted); got {:?}",
        checkpoint.path_name_members
    );
    assert_eq!(checkpoint.path_index_session_id_member, session_id);

    // Success metric +3 (CreateSession + CreateLayer + DeleteLayer), no error.
    assert_eq!(
        session_checkpoint_writes_metric(&harness.metrics, "success"),
        success_pre + 3.0,
        "checkpoint_writes_total{{kind=\"session\", result=\"success\"}} MUST be \
         success_pre + 3 after CreateSession + CreateLayer + DeleteLayer"
    );
    assert_eq!(
        session_checkpoint_writes_metric(&harness.metrics, "error"),
        error_pre,
        "checkpoint_writes_total{{kind=\"session\", result=\"error\"}} MUST NOT tick"
    );

    harness.cancel.cancel();
    let _ = harness.loop_handle.await;
}

// =============================================================================
// Scenario 4 — DeleteSession routes through delete_session_checkpoint.
// =============================================================================

/// Pin AC #1's `DeleteSession` asymmetry: `handle_delete_session`'s
/// success-arm fires `enqueue_session_checkpoint_delete` (NOT `_write`).
/// The dispatcher's spawned delete-task calls `delete_session_checkpoint`
/// on the LocalDisk adapter rather than `put_session_checkpoint`. The
/// load-bearing observable for this path is the post-call `Ok(None)` from
/// the adapter — `get_session_checkpoint(session_id)` returns no blob
/// because the delete branch actually fired (not a stale put that left the
/// blob in place).
///
/// This complements `lifecycle_delete_session_test.rs` (which pins the
/// call-site dispatch routing via a counting adapter) by exercising the
/// end-to-end RPC + adapter state — the only way the test can verify the
/// adapter's on-disk blob is gone is to bounded-poll on
/// `get_session_checkpoint == Ok(None)`.
///
/// Drive shape:
///
/// 1. CreateSession + wait for the put-task to land the first blob. Success
///    metric ticks to `success_pre + 1.0`.
/// 2. DeleteSession through the real handler. The success-arm enqueues
///    a delete; the dispatcher's spawned delete-task calls
///    `delete_session_checkpoint` on the adapter.
/// 3. Bounded-poll `get_session_checkpoint(session_id)` until it returns
///    `Ok(None)`. This is the load-bearing observable: a regression that
///    erroneously enqueued a `_write` (or a no-op) would leave the blob
///    present here.
///
/// Note on the success metric: `SessionCheckpointWriter::enqueue_session_checkpoint_delete`
/// routes its `Ok(())` outcome through the same `handle_put_outcome` shim
/// that the put path uses (see `src/checkpoint/session_writer.rs:204`), so
/// a clean delete ALSO ticks `result="success"`. The success cell therefore
/// reaches `success_pre + 2.0` post-DeleteSession (one for CreateSession,
/// one for DeleteSession). This is the actual Task 4 dispatch contract.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn delete_session_calls_delete_session_checkpoint() {
    let harness = setup_checkpoint_harness(1).await;

    let success_pre = session_checkpoint_writes_metric(&harness.metrics, "success");
    let error_pre = session_checkpoint_writes_metric(&harness.metrics, "error");

    let session_id = drive_create_session(&harness, SESSION_NAME_DELETE_SESSION).await;
    wait_for_session_checkpoint(&harness.store, &session_id, 10).await;
    wait_for_writes_metric(&harness.metrics, "success", success_pre + 1.0, 10).await;

    // Sanity-check the pre-delete blob: it should reflect CreateSession's
    // result. We're only verifying the session-meta state — the path-index
    // members and emptiness of layer_mappings are pinned in Scenario 1.
    let (pre_delete_checkpoint, _pre_delete_bytes) =
        read_session_checkpoint(&harness.store, &session_id).await;
    assert_eq!(pre_delete_checkpoint.session_id, session_id);
    assert_eq!(pre_delete_checkpoint.name, SESSION_NAME_DELETE_SESSION);

    // Snapshot the success metric immediately before DeleteSession so the
    // post-call assertion has a tight baseline. (If a concurrent test were
    // ticking this counter, the per-harness `Metrics` instance would still
    // isolate us — each setup_checkpoint_harness call builds a fresh
    // `Registry` + `Metrics` — but capturing the snapshot here keeps the
    // assertion self-evidently sound.)
    let success_before_delete = session_checkpoint_writes_metric(&harness.metrics, "success");

    let _response = handle_delete_session(
        &harness.svc,
        make_unary_token_request(DeleteSessionRequest {
            session_id: session_id.clone(),
        }),
    )
    .await
    .expect("DeleteSession must succeed against a freshly-created session");

    // Bounded-poll until the delete-task lands. `wait_for_session_checkpoint_absent`
    // is the load-bearing observable for this scenario — a regression that
    // wired `_write` would never observe `Ok(None)` and would time out here.
    // (Task 4's `lifecycle_delete_session_test.rs` already pins the
    // dispatch-routing via a counting adapter; this test exercises the
    // end-to-end RPC + adapter state through real bounded polling.)
    wait_for_session_checkpoint_absent(&harness.store, &session_id, 10).await;

    // The success cell DID tick: `enqueue_session_checkpoint_delete` routes
    // its `Ok(())` outcome through the same `handle_put_outcome` shim that
    // the put path uses (see `src/checkpoint/session_writer.rs:204`), so a
    // clean delete folds into the same `result="success"` cell as a clean
    // put. The success cell therefore reaches `success_before_delete + 1.0`
    // post-DeleteSession. A regression that swallowed the delete-success
    // (e.g. forgot to emit the metric on the delete path) would leave this
    // assertion at `success_before_delete`, masking the silent metric loss.
    wait_for_writes_metric(&harness.metrics, "success", success_before_delete + 1.0, 10).await;
    assert_eq!(
        session_checkpoint_writes_metric(&harness.metrics, "error"),
        error_pre,
        "checkpoint_writes_total{{kind=\"session\", result=\"error\"}} MUST NOT tick \
         on a clean DeleteSession"
    );

    harness.cancel.cancel();
    let _ = harness.loop_handle.await;
}
