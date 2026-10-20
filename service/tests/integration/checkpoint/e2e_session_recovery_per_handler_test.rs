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

//! End-to-end session-recovery integration tests for the per-handler
//! wiring choke points (Subtask `t3.fgj.7v.fp.xx` under Task `t2.fgj.7v.fp`,
//! Epic `t1.fgj.7v`, Bee `b.fgj`).
//!
//! Task 6 wires `try_recover_session` into seven session-resolving gRPC
//! handlers via two choke points:
//!
//! - [`SessionBasePathCache::get_base_path`](live_edit::grpc::session_cache::SessionBasePathCache)
//!   — covers CreateLayer / UpdateLayer / DeleteLayer / StatLayer /
//!   DeleteSession / JoinSession / SubscribeReadLayer (six handlers).
//! - `reconcile_ghost_members` inside `handle_list_sessions` — covers
//!   the seventh (ListSessions bypasses the cache entirely).
//!
//! Task 7's umbrella E2E body exercises UpdateLayer (AC #2 — full Valkey
//! wipe → next UpdateLayer triggers the full recovery chain) and
//! ListSessions (AC #7 — post-recovery shape). The OTHER cache-funnel
//! handlers — CreateLayer, DeleteLayer, DeleteSession — are NOT explicitly
//! exercised by Task 7's named ACs, which (per the parent Task 6 body) is
//! a "silent bug" risk: an engineer wiring a new session-resolving
//! handler that forgets the cache funnel would compile cleanly and pass
//! every existing test.
//!
//! Inline `#[cfg(test)]` tests in Task 6 verify the function-call shape
//! via the cache wrapper but cannot verify the actual gRPC handler
//! dispatch path through `LiveEditService` (auth middleware, BearerToken
//! extraction, ServiceError → tonic::Status mapping, real `SessionStore`
//! and `LayerStore` interaction). The only way to lock the wiring
//! contract end-to-end is to wipe `le:{sid}:sess` and dispatch each
//! handler individually through the real RPC entry point. This file
//! does that for the three remaining handlers; the resulting test
//! surface plus the umbrella tests covers every cache-funnel handler
//! explicitly named in Task 6's wiring contract.
//!
//! Each test follows the same five-step shape:
//!
//! 1. Bring up [`CheckpointHarness`] with a `LocalDisk` adapter and
//!    `checkpoint_enabled=true`. The harness's `SessionStore` /
//!    `LayerStore` are wired with `with_checkpoint_writer`, and its
//!    `SessionBasePathCache` is wired with `with_recovery_hook` —
//!    matching the production `main.rs` wiring exactly.
//! 2. `CreateSession` (via the real `handle_create_session` RPC entry
//!    point) + `CreateLayer` (via `handle_create_layer`). Both lifecycle
//!    hooks fire fire-and-forget session-checkpoint writes through the
//!    shared dispatcher.
//! 3. [`wait_for_session_checkpoint`] blocks until the dispatcher's
//!    spawned put-task has landed the blob on the
//!    `LocalDiskCheckpointStore`.
//! 4. [`wipe_session_state`] UNLINKs every `le:{sid}*` key — session
//!    metadata hash, every layer-mapping, the layers-index — simulating
//!    the data-loss surface SR-2.7 is designed to recover from. The
//!    path-index (`le:path:{hex(base_path)}*`) survives intact.
//! 5. Dispatch the handler under test through its real RPC entry point.
//!    Recovery fires from the cache funnel; the original session +
//!    layer-mappings are re-materialized; the handler proceeds against
//!    the restored state.
//!
//! Each test asserts:
//!
//! - The `live_edit_checkpoint_recovery_attempts_total{result="restored",
//!   scope="session"}` cell incremented by EXACTLY 1 — pinning the
//!   at-most-once contract for the cache's `recovered_once` latch.
//! - `restored` is the only outcome cell that ticks. `not_recoverable` /
//!   `error` / `already_present` stay at their pre-call snapshots.
//! - Post-call Valkey state matches what the handler would produce
//!   absent the wipe (modulo the operation being tested).
//!
//! Sandbox env note: this test uses `TestValkey::setup`, which reads
//! `TEST_VALKEY_URL`. Under the project sandbox the URL must include the
//! password and `host.docker.internal` host; see CLAUDE.md.

use fred::interfaces::{HashesInterface, KeysInterface, SetsInterface};

#[cfg(feature = "test-checkpoint-fault-injection")]
use live_edit::checkpoint::session::clear_apply_step_fault_injectors_for_test;
use live_edit::grpc::layer::{handle_create_layer, handle_delete_layer};
use live_edit::grpc::proto::{CreateSessionRequest, DeleteLayerRequest, DeleteSessionRequest};
use live_edit::grpc::session::{handle_create_session, handle_delete_session};
use live_edit::observability::metrics::SharedMetrics;

use super::helpers::*;
use crate::common::owned_delta::{delta_to_ompf_bytes, empty_delta};

const SESSION_BASE_PATH: &str = "/stage";
const SESSION_NAME_CREATE_LAYER: &str = "e2e-session-recovery-create-layer-session";
const SESSION_NAME_DELETE_LAYER: &str = "e2e-session-recovery-delete-layer-session";
const SESSION_NAME_DELETE_SESSION: &str = "e2e-session-recovery-delete-session-session";
const LAYER_NAME_ORIGINAL: &str = "original-layer";
const LAYER_NAME_NEW_CREATE: &str = "new-layer-after-recovery";

// ── Streaming framing helpers — imported from
// `tests/integration/checkpoint/helpers.rs` via `use super::helpers::*;`
// (b.dzd Phase 1 deduplication). ─────────────────────────────────────────

/// Probe the live label-cell counter value for a given outcome label on
/// the `live_edit_checkpoint_recovery_attempts_total{result, scope="session",
/// service}` metric. `scope="session"` (NOT `"layer"`) is the load-bearing
/// difference vs. the helper in `e2e_update_layer_recovery_test.rs` —
/// session-recovery and layer-recovery tick disjoint metric cells, so the
/// per-handler tests must inspect the session scope to assert their
/// invariants.
fn session_recovery_metric(metrics: &SharedMetrics, result: &str) -> u64 {
    live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(metrics),
        "live_edit_checkpoint_recovery_attempts_total",
        &[
            ("result", result),
            ("scope", "session"),
            ("service", SERVICE_LABEL),
        ],
    )
    .unwrap_or(0.0) as u64
}

// ── Composite harness driver — CreateSession + CreateLayer + wait ────────────
//
// All three tests start from the same baseline: a session with one layer,
// a session-checkpoint blob landed on disk, baseline metric snapshot
// captured. Lifting this into a helper keeps each test body focused on
// the wipe → handler-call → assert cycle that distinguishes it from its
// siblings.

struct HarnessBaseline {
    session_id: String,
    object_id: u64,
    restored_pre: u64,
    not_recoverable_pre: u64,
    error_pre: u64,
    already_present_pre: u64,
}

/// CreateSession + CreateLayer + wait_for_session_checkpoint, capturing the
/// pre-wipe baseline of the four session-scope recovery-metric cells.
///
/// CreateSession goes through `handle_create_session` (the real gRPC entry
/// point) so the wiring through `LiveEditService` is exercised end-to-end.
/// `SessionStore::create`'s success arm enqueues a fire-and-forget
/// session-checkpoint write through the dispatcher; the subsequent
/// CreateLayer also enqueues one (via `LayerStore::create_layer`'s
/// success arm). Both write the same `session_id` — `wait_for_session_checkpoint`
/// only needs to see one blob land, but a regression that stops enqueuing
/// from either path would still surface on a downstream wipe-and-recover
/// test, just via a different timeout.
async fn create_session_and_layer_and_wait_for_checkpoint(
    harness: &CheckpointHarness,
    session_name: &str,
    layer_name: &str,
) -> HarnessBaseline {
    // CreateSession through the real RPC entry point.
    let create_session_resp = handle_create_session(
        &harness.svc,
        make_unary_token_request(CreateSessionRequest {
            path: SESSION_BASE_PATH.to_owned(),
            name: session_name.to_owned(),
        }),
    )
    .await
    .expect("CreateSession must succeed against an AllowAllPermissions harness");
    let session_id = create_session_resp.into_inner().session_id;
    assert!(
        !session_id.is_empty(),
        "CreateSession must mint a non-empty session_id"
    );

    // CreateLayer through the real client-streaming entry point.
    let ompf_bytes = delta_to_ompf_bytes(&empty_delta(), &[]);
    let stream = streaming_create(vec![
        make_create_header(&session_id, layer_name),
        make_create_data(ompf_bytes),
    ]);
    let create_layer_resp = handle_create_layer(&harness.svc, make_streaming_token_request(stream))
        .await
        .expect("CreateLayer must succeed for harness setup");
    let object_id = create_layer_resp.into_inner().object_id;
    assert!(object_id > 0, "CreateLayer must mint a non-zero object_id");

    // Wait for the session-checkpoint dispatcher's spawned puts to land
    // on disk AND for the writer's TaskTracker to drain. Both
    // CreateSession and CreateLayer enqueue a fire-and-forget put-task;
    // observing one blob via `wait_for_session_checkpoint` alone does
    // NOT prove the second task is also finished. Under fast Valkey the
    // post-recovery layer-mapping assertion can race a late
    // CreateSession write that overwrites the post-CreateLayer state
    // with the pre-CreateLayer (empty layer_mappings) snapshot.
    // `_quiesced` waits for `tracker.is_empty()` after the blob lands.
    wait_for_session_checkpoint_quiesced(harness, &session_id, 10).await;

    // Pre-wipe baseline of the four session-scope recovery-metric cells.
    // A fresh harness starts every cell at zero, but the snapshot pattern
    // is robust to any future `Metrics::new` wiring change that pre-seeds
    // a cell.
    let restored_pre = session_recovery_metric(&harness.metrics, "restored");
    let not_recoverable_pre = session_recovery_metric(&harness.metrics, "not_recoverable");
    let error_pre = session_recovery_metric(&harness.metrics, "error");
    let already_present_pre = session_recovery_metric(&harness.metrics, "already_present");

    HarnessBaseline {
        session_id,
        object_id,
        restored_pre,
        not_recoverable_pre,
        error_pre,
        already_present_pre,
    }
}

/// Confirm that the wipe actually emptied `le:{sid}:sess`. A failed wipe
/// would let the next handler call succeed without exercising the
/// recovery branch, silently passing the rest of the test. Run AFTER
/// `wipe_session_state` to catch test-infrastructure regressions before
/// the assertion phase obscures them.
async fn assert_session_meta_absent_after_wipe(harness: &CheckpointHarness, session_id: &str) {
    let session_meta_key = harness.tv.key_schema().session_meta(session_id);
    let exists: bool = harness
        .tv
        .client()
        .pool
        .exists(&session_meta_key)
        .await
        .expect("EXISTS sess probe must succeed against healthy Valkey");
    assert!(
        !exists,
        "wipe_session_state must UNLINK le:{{session_id}}:sess; surviving key here \
         means the scan-pattern missed it and recovery would silently no-op via \
         the pre-acquire AlreadyPresent short-circuit"
    );
}

/// Drop the cached `base_path` entry for `session_id` on the harness's
/// `SessionBasePathCache`. Load-bearing: the baseline CreateLayer call in
/// every test populates the cache with the live `base_path`, and the
/// cache's 2-second positive TTL is long enough that the post-wipe
/// handler call inside the same test would HIT the cache and never
/// consult `session_store.get_base_path` — bypassing the recovery
/// branch entirely. Without this explicit invalidation, every test in
/// this file would silently observe `restored == 0` (recovery never
/// fired) AND a successful handler call (the cache HIT returned the
/// pre-wipe `base_path`), masking the wiring regression we're trying to
/// pin.
///
/// In production, the `session:deleted` pub/sub fanout drives the
/// cross-replica cache invalidation; under E2E test we have no peer
/// replica writing the wipe, so the cache must be invalidated here.
fn invalidate_session_cache(harness: &CheckpointHarness, session_id: &str) {
    harness.svc.session_base_path_cache.invalidate(session_id);
}

/// Post-recovery assertion shared by every test: the four session-scope
/// recovery cells must show EXACTLY ONE `restored` increment relative to
/// the pre-call snapshot, with every other cell unchanged. This is the
/// load-bearing at-most-once contract the cache's `recovered_once` latch
/// is supposed to enforce — a regression that double-fired recovery
/// would show `restored >= 2`, and a regression that bypassed recovery
/// entirely would show `restored == 0` plus a handler-level NotFound.
fn assert_exactly_one_restored(harness: &CheckpointHarness, baseline: &HarnessBaseline) {
    assert_eq!(
        session_recovery_metric(&harness.metrics, "restored") - baseline.restored_pre,
        1,
        "exactly ONE session-recovery attempt must have reported `restored` \
         (at-most-once contract via `recovered_once` latch); restored_pre={}",
        baseline.restored_pre
    );
    assert_eq!(
        session_recovery_metric(&harness.metrics, "not_recoverable"),
        baseline.not_recoverable_pre,
        "post-recovery `not_recoverable` MUST equal pre-call snapshot — the \
         checkpoint blob was readable so the apply branch ran, not the \
         not_recoverable arm"
    );
    assert_eq!(
        session_recovery_metric(&harness.metrics, "error"),
        baseline.error_pre,
        "post-recovery `error` MUST equal pre-call snapshot — no transient \
         backend failure was injected"
    );
    assert_eq!(
        session_recovery_metric(&harness.metrics, "already_present"),
        baseline.already_present_pre,
        "post-recovery `already_present` MUST equal pre-call snapshot — the \
         wipe drove the pre-acquire EXISTS check to false, so the cell that \
         tracks the short-circuit path stays untouched"
    );
}

// =============================================================================
// Test 1 — CreateLayer triggers session recovery and proceeds.
// =============================================================================

/// Pin the wiring of `SessionBasePathCache::get_base_path` into
/// `handle_create_layer` (Subtask `fp.rd`).
///
/// Why each step is load-bearing:
///
/// - The wipe leaves the path-index intact (SR-2.7 asymmetric self-healing)
///   so the apply path's path-name uniqueness check still passes when the
///   recovery hook re-materializes the session-scoped keys.
/// - The post-wipe CreateLayer targets a NEW layer name (not the original)
///   so the test exercises CreateLayer's create-fresh-mapping path, not
///   an overwrite. A regression that erroneously hit the overwrite path
///   on the new name would show up as a mapping pointing to the wrong
///   `object_id` post-call.
/// - Assertion (a): post-call session-meta is present — the recovery hook's
///   apply ran successfully through `apply_session_checkpoint`.
/// - Assertion (b): the ORIGINAL layer's mapping is restored to the
///   pre-wipe `object_id` — Task 5 applier's
///   `layer_mappings`-restoration contract.
/// - Assertion (c): the NEW layer's mapping is present, pointing to a
///   distinct object_id minted by the post-recovery CreateLayer — proves
///   the handler proceeded past the recovery branch and continued its
///   normal flow.
#[allow(clippy::await_holding_lock)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn recovery_fires_on_create_layer_session_absent() {
    // Serialize against fault-injection tests in the sibling
    // `e2e_session_recovery_atomicity_test` binary. Those tests install
    // entries on the process-global `APPLY_STEP_FAULT_INJECTORS` map,
    // which our `handle_*` calls also pop on their recovery path —
    // without the gate a concurrent test's queued failure can be
    // consumed by our recovery and surface here as a spurious
    // `Unavailable`. The clear drains any leftover state from a
    // previous fault-injection test in the same process.
    let _serial = lock_fault_injector_serial();
    #[cfg(feature = "test-checkpoint-fault-injection")]
    clear_apply_step_fault_injectors_for_test();
    let harness = setup_checkpoint_harness(1).await;

    let baseline = create_session_and_layer_and_wait_for_checkpoint(
        &harness,
        SESSION_NAME_CREATE_LAYER,
        LAYER_NAME_ORIGINAL,
    )
    .await;
    let HarnessBaseline {
        session_id,
        object_id: original_object_id,
        ..
    } = &baseline;

    // Wipe `le:{sid}*` (session metadata + layer-mappings + layers-index).
    // Path-index left intact.
    wipe_session_state(&harness.tv, session_id).await;
    assert_session_meta_absent_after_wipe(&harness, session_id).await;
    invalidate_session_cache(&harness, session_id);

    // Dispatch CreateLayer for a NEW layer name through the real handler.
    // The cache funnel's `get_base_path` sees NotFound, fires session
    // recovery, the apply rebuilds session-meta + the original
    // layer-mapping + layers-index; then CreateLayer continues against
    // the restored session and mints a new mapping for the new name.
    let ompf_bytes = delta_to_ompf_bytes(&empty_delta(), &[]);
    let stream = streaming_create(vec![
        make_create_header(session_id, LAYER_NAME_NEW_CREATE),
        make_create_data(ompf_bytes),
    ]);
    let response = handle_create_layer(&harness.svc, make_streaming_token_request(stream))
        .await
        .expect(
            "CreateLayer MUST succeed via the session-recovery branch in the cache funnel; \
                 a NotFound here means session_recovery did not fire (the cache's \
                 with_recovery_hook wiring is broken) or the apply pipeline failed",
        );
    let new_object_id = response.into_inner().object_id;
    assert!(
        new_object_id > 0,
        "post-recovery CreateLayer must mint a non-zero object_id; got {new_object_id}"
    );
    assert_ne!(
        new_object_id, *original_object_id,
        "post-recovery CreateLayer must mint a FRESH object_id distinct from the \
         pre-wipe original ({original_object_id}); collision means the counter \
         leaked into recovery, which would resurrect a stale object_id"
    );

    // Exactly ONE session-recovery attempt landed in the `restored` cell.
    assert_exactly_one_restored(&harness, &baseline);

    // (a) session-meta is present post-recovery.
    let session_meta_key = harness.tv.key_schema().session_meta(session_id);
    let meta_after: std::collections::HashMap<String, String> = harness
        .tv
        .client()
        .pool
        .hgetall(&session_meta_key)
        .await
        .expect("HGETALL session_meta must succeed");
    assert!(
        !meta_after.is_empty(),
        "post-recovery session_meta MUST be present; got empty hash. The \
         apply_session_checkpoint pipeline failed to write the session-meta fields."
    );
    assert_eq!(
        meta_after.get("name"),
        Some(&SESSION_NAME_CREATE_LAYER.to_string()),
        "post-recovery session_meta.name MUST equal the original session name; \
         got {:?}",
        meta_after.get("name")
    );

    // (b) original layer's mapping restored to the pre-wipe object_id.
    let original_mapping_key = harness
        .tv
        .key_schema()
        .layer_mapping(session_id, LAYER_NAME_ORIGINAL);
    let original_mapping: Option<String> = harness
        .tv
        .client()
        .pool
        .get(&original_mapping_key)
        .await
        .expect("GET original layer_mapping must succeed");
    assert_eq!(
        original_mapping,
        Some(original_object_id.to_string()),
        "post-recovery, the original layer's mapping MUST resolve to the pre-wipe \
         object_id ({original_object_id}); a missing or different value means \
         apply_session_checkpoint did not restore the layer_mappings table"
    );

    // (c) new layer's mapping is present and points to the freshly-minted
    // object_id from the post-recovery CreateLayer.
    let new_mapping_key = harness
        .tv
        .key_schema()
        .layer_mapping(session_id, LAYER_NAME_NEW_CREATE);
    let new_mapping: Option<String> = harness
        .tv
        .client()
        .pool
        .get(&new_mapping_key)
        .await
        .expect("GET new layer_mapping must succeed");
    assert_eq!(
        new_mapping,
        Some(new_object_id.to_string()),
        "post-recovery CreateLayer's new layer mapping MUST point at the new \
         object_id ({new_object_id}); proves the handler resumed normal flow \
         past the recovery branch"
    );

    // Graceful shutdown of the checkpoint loop. Drops are sufficient for
    // cleanup but cancelling + awaiting yields a deterministic teardown.
    harness.cancel.cancel();
    let _ = harness.loop_handle.await;
}

// =============================================================================
// Test 2 — DeleteLayer triggers session recovery and proceeds.
// =============================================================================

/// Pin the wiring of `SessionBasePathCache::get_base_path` into
/// `handle_delete_layer` (Subtask `fp.rd`).
///
/// Why each step is load-bearing:
///
/// - DeleteLayer targets the ORIGINAL layer (the one whose mapping the
///   recovery hook must restore). A test that deleted a never-created
///   layer would short-circuit before reaching the cache funnel.
/// - Assertion (a): post-call session-meta is STILL present — DeleteLayer
///   only removes one layer's mapping, not the session itself.
/// - Assertion (b): the original layer's mapping is GONE post-call — the
///   recovery branch restored it, then DeleteLayer's cascade removed it
///   (so the test pins both halves: recovery happened, DeleteLayer
///   succeeded against the restored state).
/// - Assertion (c): the layers-index is updated — the layer name is no
///   longer in the index after DeleteLayer's SREM.
#[allow(clippy::await_holding_lock)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn recovery_fires_on_delete_layer_session_absent() {
    // See `recovery_fires_on_create_layer_session_absent` above for the
    // rationale on holding `lock_fault_injector_serial` in non-FI tests.
    let _serial = lock_fault_injector_serial();
    #[cfg(feature = "test-checkpoint-fault-injection")]
    clear_apply_step_fault_injectors_for_test();
    let harness = setup_checkpoint_harness(1).await;

    let baseline = create_session_and_layer_and_wait_for_checkpoint(
        &harness,
        SESSION_NAME_DELETE_LAYER,
        LAYER_NAME_ORIGINAL,
    )
    .await;
    let HarnessBaseline { session_id, .. } = &baseline;

    wipe_session_state(&harness.tv, session_id).await;
    assert_session_meta_absent_after_wipe(&harness, session_id).await;
    invalidate_session_cache(&harness, session_id);

    // Dispatch DeleteLayer through the real handler. The cache funnel's
    // `get_base_path` sees NotFound, fires session recovery, the apply
    // re-materializes session-meta + the original layer-mapping +
    // layers-index; then DeleteLayer continues, resolves the mapping
    // (now present again), and removes it.
    let _response = handle_delete_layer(
        &harness.svc,
        make_unary_token_request(DeleteLayerRequest {
            session_id: session_id.clone(),
            layer_name: LAYER_NAME_ORIGINAL.to_owned(),
        }),
    )
    .await
    .expect(
        "DeleteLayer MUST succeed via the session-recovery branch in the cache funnel; \
         a NotFound here means session_recovery did not fire, or recovery fired but \
         did not restore the layer_mappings table (Task 5 applier regression)",
    );

    // Exactly ONE session-recovery attempt landed in the `restored` cell.
    assert_exactly_one_restored(&harness, &baseline);

    // (a) session-meta is STILL present post-call — DeleteLayer does not
    // remove the session.
    let session_meta_key = harness.tv.key_schema().session_meta(session_id);
    let session_meta_exists: bool = harness
        .tv
        .client()
        .pool
        .exists(&session_meta_key)
        .await
        .expect("EXISTS session_meta probe must succeed");
    assert!(
        session_meta_exists,
        "post-DeleteLayer session_meta MUST still be present — DeleteLayer only \
         removes one layer's mapping, not the session. If this assertion fires \
         the test exercised the wrong handler or DeleteLayer leaked a cascade \
         that touched the session-meta hash."
    );

    // (b) original layer's mapping is GONE post-call.
    let mapping_key = harness
        .tv
        .key_schema()
        .layer_mapping(session_id, LAYER_NAME_ORIGINAL);
    let mapping_after: Option<String> = harness
        .tv
        .client()
        .pool
        .get(&mapping_key)
        .await
        .expect("GET mapping must succeed");
    assert_eq!(
        mapping_after, None,
        "post-DeleteLayer the original layer's mapping MUST be gone; a \
         surviving value here means DeleteLayer did not actually run its \
         cascade (e.g. it short-circuited on a NotFound that recovery should \
         have prevented)"
    );

    // (c) layers-index no longer carries the deleted layer name.
    let layers_index_key = harness.tv.key_schema().layers_index(session_id);
    let layers_after: Vec<String> = harness
        .tv
        .client()
        .pool
        .smembers::<Vec<String>, _>(&layers_index_key)
        .await
        .expect("SMEMBERS layers_index must succeed");
    let hex_original = live_edit::storage::keys::hex_encode(LAYER_NAME_ORIGINAL);
    assert!(
        !layers_after.contains(&hex_original),
        "post-DeleteLayer layers_index MUST NOT carry hex({LAYER_NAME_ORIGINAL}); \
         got {layers_after:?}"
    );

    harness.cancel.cancel();
    let _ = harness.loop_handle.await;
}

// =============================================================================
// Test 3 — DeleteSession triggers session recovery and then deletes.
// =============================================================================

/// Pin the wiring of `SessionBasePathCache::get_base_path` into
/// `handle_delete_session` (Subtask `fp.rd`).
///
/// Why each step is load-bearing:
///
/// - DeleteSession is the only handler in this file where recovery
///   "succeeds and then deletes everything recovery just rebuilt" — the
///   post-call state is the same as if the wipe had been a permanent
///   loss followed by a clean DeleteSession. The test pins that the
///   transient recovery + delete cascade observe each other correctly:
///   recovery restores the mapping table, DeleteSession's cascade
///   iterates it and removes every layer, then deletes the session
///   metadata.
/// - The lifecycle-hook contract from Task 4 of `t1.fgj.7v` is that
///   `SessionStore::delete` ALSO enqueues a
///   `enqueue_session_checkpoint_delete` — so post-call, the
///   session-checkpoint blob is also gone. We assert this through
///   `wait_for_session_checkpoint_absent`.
/// - Assertion (a): the recovery counter shows EXACTLY ONE `restored`
///   attempt — Task 6's `recovered_once` latch at-most-once contract.
/// - Assertion (b): session-meta is gone post-call (DeleteSession
///   cascade succeeded).
/// - Assertion (c): the session-checkpoint blob on disk is also gone
///   (Task 4 lifecycle-hook contract).
#[allow(clippy::await_holding_lock)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn recovery_fires_on_delete_session_then_deletes() {
    // See `recovery_fires_on_create_layer_session_absent` above for the
    // rationale on holding `lock_fault_injector_serial` in non-FI tests.
    let _serial = lock_fault_injector_serial();
    #[cfg(feature = "test-checkpoint-fault-injection")]
    clear_apply_step_fault_injectors_for_test();
    let harness = setup_checkpoint_harness(1).await;

    let baseline = create_session_and_layer_and_wait_for_checkpoint(
        &harness,
        SESSION_NAME_DELETE_SESSION,
        LAYER_NAME_ORIGINAL,
    )
    .await;
    let HarnessBaseline { session_id, .. } = &baseline;

    wipe_session_state(&harness.tv, session_id).await;
    assert_session_meta_absent_after_wipe(&harness, session_id).await;
    invalidate_session_cache(&harness, session_id);

    // Dispatch DeleteSession through the real handler. The cache funnel's
    // `get_base_path` sees NotFound, fires session recovery, the apply
    // restores session-meta + layer-mapping + layers-index; then
    // DeleteSession cascade iterates the restored layers-index, removes
    // each layer (via `LayerStore::delete_layer` which itself uses the
    // restored mapping), then deletes session-meta + path-index entries.
    let _response = handle_delete_session(
        &harness.svc,
        make_unary_token_request(DeleteSessionRequest {
            session_id: session_id.clone(),
        }),
    )
    .await
    .expect(
        "DeleteSession MUST succeed via the session-recovery branch in the cache \
         funnel; a NotFound here means session_recovery did not fire, or the cascade \
         hit a layer-mapping mismatch (apply_session_checkpoint did not restore \
         the layer_mappings or layers_index tables symmetrically)",
    );

    // (a) Exactly ONE session-recovery attempt landed in the `restored` cell.
    assert_exactly_one_restored(&harness, &baseline);

    // (b) session-meta is GONE post-call.
    let session_meta_key = harness.tv.key_schema().session_meta(session_id);
    let session_meta_exists: bool = harness
        .tv
        .client()
        .pool
        .exists(&session_meta_key)
        .await
        .expect("EXISTS session_meta probe must succeed");
    assert!(
        !session_meta_exists,
        "post-DeleteSession session_meta MUST be gone; a surviving key here \
         means DeleteSession did not actually run its delete-cascade \
         (e.g. it returned early after recovery, or the cascade leaked)"
    );

    // (c) the session-checkpoint blob on disk is also gone. Task 4's
    // lifecycle-hook contract: `SessionStore::delete` enqueues
    // `enqueue_session_checkpoint_delete`, which calls
    // `delete_session_checkpoint` on the adapter.
    //
    // That delete is dispatched on the same fire-and-forget writer
    // TaskTracker as the put-tasks. Drain the tracker first so the delete
    // is guaranteed to have run before we poll for absence; otherwise a
    // loaded / cold-cache runner can starve the delete task past the 10s
    // absent-poll budget, surfacing as a spurious timeout here.
    wait_for_writer_idle(&harness.session_checkpoint_tracker, 10).await;
    wait_for_session_checkpoint_absent(&harness.store, session_id, 10).await;

    harness.cancel.cancel();
    let _ = harness.loop_handle.await;
}

/// Regression guard for the deterministic CI failure in
/// `recovery_fires_on_delete_session_then_deletes` (blob never went absent →
/// 30s `wait_for_session_checkpoint_absent` timeout).
///
/// Root cause: the `DeleteSession` cascade deleted each layer via
/// `LayerStore::delete_layer`, which enqueues a fire-and-forget session-
/// checkpoint **rewrite**. That rewrite raced the session-checkpoint **DELETE**
/// that `SessionStore::delete` enqueues on the same writer TaskTracker — a
/// rewrite whose `build_session_checkpoint` read the still-present session and
/// landed its PUT after the DELETE resurrected the tombstoned blob. The fix:
/// the cascade calls `delete_layer_during_session_delete`, which suppresses the
/// per-layer rewrite.
///
/// This pins the fix **deterministically** (independent of task scheduling):
/// the teardown-variant delete must leave the session-checkpoint blob byte-for-
/// byte unchanged (no rewrite enqueued). The umbrella
/// `recovery_fires_on_delete_session_then_deletes` test asserts the same
/// invariant end-to-end, but only flakes under the specific CI task ordering;
/// this one fails on every run if the gating regresses.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_layer_during_session_delete_does_not_rewrite_session_checkpoint() {
    let harness = setup_checkpoint_harness(1).await;
    let HarnessBaseline { session_id, .. } = create_session_and_layer_and_wait_for_checkpoint(
        &harness,
        SESSION_NAME_DELETE_SESSION,
        LAYER_NAME_ORIGINAL,
    )
    .await;

    let before = harness
        .store
        .get_session_checkpoint(&session_id)
        .await
        .expect("get_session_checkpoint before must succeed")
        .expect("session-checkpoint blob must exist after setup")
        .bytes;

    // Teardown-variant delete — the path the DeleteSession cascade uses. It must
    // NOT enqueue a session-checkpoint rewrite.
    harness
        .svc
        .layer_store
        .delete_layer_during_session_delete(
            &session_id,
            LAYER_NAME_ORIGINAL,
            harness.svc.ext_value_store.clone(),
        )
        .await
        .expect("teardown-variant delete_layer must succeed");

    // Drain the writer so any erroneously-enqueued rewrite would have landed.
    wait_for_writer_idle(&harness.session_checkpoint_tracker, 10).await;

    let after = harness
        .store
        .get_session_checkpoint(&session_id)
        .await
        .expect("get_session_checkpoint after must succeed")
        .expect("session-checkpoint blob must still exist (no DELETE was enqueued here)")
        .bytes;

    assert_eq!(
        before, after,
        "teardown-variant layer delete MUST NOT rewrite the session checkpoint; a \
         changed blob means delete_layer_during_session_delete enqueued a rewrite, \
         which races SessionStore::delete's tombstone and can resurrect the blob"
    );

    harness.cancel.cancel();
    let _ = harness.loop_handle.await;
}

/// Companion contrast to the guard above: the STANDALONE `delete_layer` entry
/// point (a real `DeleteLayer` on a session that survives) DOES rewrite the
/// session checkpoint to reflect the smaller layer set (SR-2.7 per-layer
/// refresh). Pins the contrast so a future change can't "fix" the resurrection
/// by suppressing the rewrite everywhere — which would silently drop the
/// per-layer refresh contract for standalone DeleteLayer.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn standalone_delete_layer_rewrites_session_checkpoint() {
    let harness = setup_checkpoint_harness(1).await;
    let HarnessBaseline { session_id, .. } = create_session_and_layer_and_wait_for_checkpoint(
        &harness,
        SESSION_NAME_DELETE_LAYER,
        LAYER_NAME_ORIGINAL,
    )
    .await;

    let before = harness
        .store
        .get_session_checkpoint(&session_id)
        .await
        .expect("get_session_checkpoint before must succeed")
        .expect("session-checkpoint blob must exist after setup")
        .bytes;

    harness
        .svc
        .layer_store
        .delete_layer(
            &session_id,
            LAYER_NAME_ORIGINAL,
            harness.svc.ext_value_store.clone(),
        )
        .await
        .expect("standalone delete_layer must succeed");

    wait_for_writer_idle(&harness.session_checkpoint_tracker, 10).await;

    let after = harness
        .store
        .get_session_checkpoint(&session_id)
        .await
        .expect("get_session_checkpoint after must succeed")
        .expect("session-checkpoint blob must still exist (the session survives)")
        .bytes;

    assert_ne!(
        before, after,
        "standalone delete_layer MUST rewrite the session checkpoint to reflect the \
         removed layer (SR-2.7 per-layer refresh)"
    );

    harness.cancel.cancel();
    let _ = harness.loop_handle.await;
}
