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

//! End-to-end test for Epic AC #3 — atomic session-scoped restore (Step B
//! fault injection) under Task `t2.fgj.7v.o7`, Epic `t1.fgj.7v`, Bee
//! `b.fgj`. Subtask `t3.fgj.7v.o7.6f`.
//!
//! # What this file pins
//!
//! Task 5's recovery applier (`apply_session_checkpoint`,
//! `src/checkpoint/session.rs`) splits into two independently-atomic
//! MULTI/EXEC transactions per SR-2.7:
//!
//! - **Step A** — path-index SADDs on the `{hex(base_path)}` slot
//!   (`le:path:{hex(base_path)}` legacy session-id set +
//!   `le:path:{hex(base_path)}:names` composite set).
//! - **Step B** — session-scoped HSET + SETs + SADDs on the
//!   `{session_id}` slot (`le:{sid}:sess`, `le:{sid}:layer:{hex(name)}`,
//!   `le:{sid}:layers`).
//!
//! Epic AC #3 asserts that a mid-step failure between Step A and Step B
//! leaves the path-index half fully committed (Step A is internally
//! atomic) AND the session-scope half fully absent (Step B's MULTI/EXEC
//! never reached EXEC). The "ghost-but-functional" failure mode — session-
//! scope half present without path-index — is the unrecoverable variant
//! SR-2.7's ordering exists to prevent; the opposite shape (path-index
//! members pointing at a missing session) is recoverable because
//! `ListSessions`' opportunistic-cleanup walk reconciles ghost members.
//!
//! Three scenarios:
//!
//! 1. `step_b_session_scope_failure_leaves_path_index_landed_and_no_session_state`
//!    — install a single Step B failure, dispatch the session-resolving
//!    RPC, assert `Code::Unavailable`, assert path-index landed AND
//!    session-scope absent, assert `error{scope=session}` ticked +1.
//! 2. `recovery_retry_after_step_b_failure_converges` — drain the
//!    injector after the first failure, dispatch a SECOND RPC, assert
//!    `Ok`. Re-running Step A is a no-op (idempotent SADD pins the Task 5
//!    idempotency contract end-to-end); Step B now succeeds; the
//!    `restored{scope=session}` cell ticks +1.
//! 3. `partial_step_b_failure_does_not_corrupt_existing_path_index` —
//!    pin the "Step A is not rolled back by a downstream Step B failure"
//!    invariant. Ghost-member reconciliation belongs to the orphan-sweep
//!    / `ListSessions` cleanup path (Task 7 AC #5 / AC #7), NOT to the
//!    applier — the applier's contract is purely additive.
//!
//! # Why DeleteLayer is the session-resolving RPC under test
//!
//! Every Task-6 cache-funnel handler (CreateLayer / UpdateLayer /
//! DeleteLayer / StatLayer / DeleteSession / JoinSession /
//! SubscribeReadLayer) routes its session-id → base_path resolution
//! through `SessionBasePathCache::get_base_path`, so any of them would
//! exercise Step B via session-recovery. We pick DeleteLayer for three
//! reasons:
//!
//! - Unary (no chunked-stream framing) — minimal scaffolding cost.
//! - Does not depend on `obj:hdr` being present post-recovery — the
//!   handler delegates to `LayerStore::delete_layer`, which resolves the
//!   layer-mapping (restored by session-recovery) and then issues
//!   idempotent UNLINKs across the per-`{object_id}` key family.
//!   Scenario 2's "RPC succeeds" assertion is independent of the
//!   layer-recovery branch firing.
//! - Production-realistic — the same wiring choke point as the other
//!   six session-resolving handlers.
//!
//! # Sandbox note
//!
//! Uses `TestValkey::setup`, which reads `TEST_VALKEY_URL`. Under the
//! project sandbox the URL must include the password and
//! `host.docker.internal` host; see CLAUDE.md.

#![cfg(feature = "test-checkpoint-fault-injection")]

use std::collections::{HashMap, VecDeque};

use fred::interfaces::{HashesInterface, KeysInterface, SetsInterface};
use tonic::Code;

use live_edit::checkpoint::session::{
    clear_apply_step_fault_injectors_for_test, set_apply_step_fault_injector_for_test, ApplyStep,
};
use live_edit::checkpoint::store::CheckpointError;
use live_edit::grpc::layer::{handle_create_layer, handle_delete_layer};
use live_edit::grpc::proto::{CreateSessionRequest, DeleteLayerRequest};
use live_edit::grpc::session::handle_create_session;
use live_edit::storage::keys::{hex_encode, KeySchema};

use super::helpers::*;
use crate::common::owned_delta::{delta_to_ompf_bytes, empty_delta};

// ── Test constants ───────────────────────────────────────────────────────────
//
// Per-scenario base paths and session names keep the three tests isolated
// even when the process-global `APPLY_STEP_FAULT_INJECTORS` map leaks
// across `cargo nextest` shared-process boundaries — each scenario calls
// `clear_apply_step_fault_injectors_for_test()` at top AND bottom, and
// uses a distinct `(base_path, session_name)` so the live Valkey
// keyspaces never collide.

const SESSION_BASE_PATH_STEP_B_FAIL: &str = "/stage/atomicity-step-b-fail";
const SESSION_BASE_PATH_RETRY: &str = "/stage/atomicity-retry";
const SESSION_BASE_PATH_NO_CORRUPT: &str = "/stage/atomicity-no-corrupt";
const SESSION_NAME_STEP_B_FAIL: &str = "atomicity-step-b-fail-session";
const SESSION_NAME_RETRY: &str = "atomicity-retry-session";
const SESSION_NAME_NO_CORRUPT: &str = "atomicity-no-corrupt-session";
const LAYER_NAME: &str = "atomicity-original-layer";

// ── Cross-test serialization ────────────────────────────────────────────────
//
// `APPLY_STEP_FAULT_INJECTORS` is process-global state per its docstring
// contract — the seam's queues live in a `OnceLock<Mutex<HashMap<...>>>`
// shared across every test in this binary. Both this file's Step B
// scenarios AND `e2e_session_recovery_path_index_atomicity_test.rs`'s
// Step A scenarios install/pop entries on the same map; without
// cross-file serialization, one file's
// `set_apply_step_fault_injector_for_test` install can be consumed by the
// OTHER file's `apply_session_checkpoint` call — observable as the OTHER
// test seeing an unexpected `Err` AND this test seeing an unexpected
// `Ok` (the fault never fired).
//
// The fix is the cross-file `APPLY_STEP_FAULT_INJECTOR_TEST_SERIAL`
// `Mutex` declared once in `helpers.rs`, acquired via
// [`lock_fault_injector_serial`] at the top of every scenario in both
// files. See that gate's docstring for the locking-discipline rationale
// (poison recovery, `std::sync::Mutex` choice, etc.).

// ── Streaming framing helpers — imported from
// `tests/integration/checkpoint/helpers.rs` via `use super::helpers::*;`
// (b.dzd Phase 1 deduplication). ─────────────────────────────────────────

// ── Composite harness driver ────────────────────────────────────────────────
//
// All three scenarios share the same baseline: CreateSession +
// CreateLayer through the real RPC entry points (so the lifecycle hook
// fires, the session-checkpoint dispatcher lands a blob on disk), wait
// for the blob, then `full_valkey_wipe` to simulate the data-loss
// surface the recovery path is designed to defend against. The Step B
// fault injection is installed AFTER the baseline so it only affects
// the apply path under test, not the pre-wipe lifecycle writes.
//
// The path-name composite the assembler emits is exactly one per session —
// `session_name\0session_id` — matching what `SessionStore::create_once`
// writes on the live path (`src/storage/session.rs:308`). The path-index
// session-id member is bare `session_id`. Both pinned to
// `path_index_after_step_b_fail` / `path_names_after_step_b_fail`
// post-apply.

struct AtomicityBaseline {
    session_id: String,
    /// Pre-call snapshot of every session-scope recovery-metric cell. A
    /// future `Metrics::new` wiring change that pre-seeds a cell would
    /// shift the absolute values; the snapshot pattern is robust to
    /// that.
    error_pre: u64,
    restored_pre: u64,
    not_recoverable_pre: u64,
    already_present_pre: u64,
}

/// CreateSession + CreateLayer + wait for the session checkpoint to land,
/// then capture the pre-call session-scope recovery-metric snapshot.
async fn create_session_and_layer_and_wait_for_checkpoint(
    harness: &CheckpointHarness,
    base_path: &str,
    session_name: &str,
    layer_name: &str,
) -> AtomicityBaseline {
    let create_session_resp = handle_create_session(
        &harness.svc,
        make_unary_token_request(CreateSessionRequest {
            path: base_path.to_owned(),
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

    let ompf_bytes = delta_to_ompf_bytes(&empty_delta(), &[]);
    let stream = streaming_create(vec![
        make_create_header(&session_id, layer_name),
        make_create_data(ompf_bytes),
    ]);
    let create_layer_resp = handle_create_layer(&harness.svc, make_streaming_token_request(stream))
        .await
        .expect("CreateLayer must succeed for the atomicity baseline");
    let object_id = create_layer_resp.into_inner().object_id;
    assert!(
        object_id > 0,
        "CreateLayer must mint a non-zero object_id; got {object_id}"
    );

    // Block until the session-checkpoint dispatcher's spawned put-tasks
    // have landed AND the writer's TaskTracker is drained. CreateSession
    // and CreateLayer each enqueue a fire-and-forget put; observing one
    // blob via `wait_for_session_checkpoint` alone does not prove the
    // other has finished. Under fast Valkey a downstream
    // `full_valkey_wipe` + recovery cycle would race the still-in-flight
    // task — overwriting the post-CreateLayer state with a stale
    // post-CreateSession snapshot and surfacing as the wrong recovery
    // outcome (Step B failure path becomes a `not_recoverable` outcome).
    wait_for_session_checkpoint_quiesced(harness, &session_id, 10).await;

    AtomicityBaseline {
        session_id,
        error_pre: session_recovery_metric(harness, "error"),
        restored_pre: session_recovery_metric(harness, "restored"),
        not_recoverable_pre: session_recovery_metric(harness, "not_recoverable"),
        already_present_pre: session_recovery_metric(harness, "already_present"),
    }
}

/// Drop the cached `base_path` entry for `session_id` on the harness's
/// `SessionBasePathCache`. Load-bearing: the baseline CreateLayer call
/// populates the cache with the live `base_path`; without invalidation,
/// the post-wipe handler dispatch would HIT the cache and skip the
/// session-recovery branch entirely, silently bypassing the Step B
/// fault-injection path under test.
///
/// In production the `session:deleted` pub/sub fanout drives the
/// cross-replica cache invalidation; under E2E test we have no peer
/// replica writing the wipe, so the cache must be invalidated here.
fn invalidate_session_cache(harness: &CheckpointHarness, session_id: &str) {
    harness.svc.session_base_path_cache.invalidate(session_id);
}

/// Probe `EXISTS le:{sid}:sess` to confirm the wipe landed before the
/// fault-injection RPC dispatches. A failed wipe would let the next
/// `get_base_path` succeed via the live HGETALL path (no NotFound, no
/// recovery branch, no fault-injection), silently masking the wiring
/// regression the test exists to catch.
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
        "full_valkey_wipe must UNLINK le:{{sid}}:sess; a surviving key here means \
         the SCAN missed it and session-recovery would silently no-op via the \
         pre-acquire AlreadyPresent short-circuit, bypassing the Step B \
         fault-injection path under test"
    );
}

/// Read `SMEMBERS le:path:{hex(base_path)}` and assert membership shape.
async fn assert_path_index_carries_session(
    harness: &CheckpointHarness,
    base_path: &str,
    session_id: &str,
) {
    let path_index_key = harness.tv.key_schema().path_index(base_path);
    let members: Vec<String> = harness
        .tv
        .client()
        .pool
        .smembers(&path_index_key)
        .await
        .expect("SMEMBERS path_index must succeed");
    assert!(
        members.contains(&session_id.to_string()),
        "Step A path-index legacy set MUST carry session_id={session_id} \
         post-Step-B-failure; got members={members:?}. A missing entry here \
         means Step A's MULTI/EXEC was rolled back by the Step B injection \
         (atomicity invariant broken) or Step A never ran at all (apply \
         pipeline regression)."
    );
    let sismember: bool = harness
        .tv
        .client()
        .pool
        .sismember(&path_index_key, session_id)
        .await
        .expect("SISMEMBER path_index must succeed");
    assert!(
        sismember,
        "SISMEMBER le:path:{{hex(base_path)}} MUST return 1 for session_id={session_id} \
         after Step A commit; got 0"
    );
}

/// Read `SMEMBERS le:path:{hex(base_path)}:names` and assert the
/// expected `(session_name, session_id)` composite landed. The
/// assembler emits exactly one `path_names_index` composite per session,
/// matching the live `SessionStore::create_once` write at
/// `src/storage/session.rs:308`.
async fn assert_path_names_carries_expected_composite(
    harness: &CheckpointHarness,
    base_path: &str,
    session_id: &str,
    session_name: &str,
) {
    let path_names_key = harness.tv.key_schema().path_names_index(base_path);
    let members: Vec<String> = harness
        .tv
        .client()
        .pool
        .smembers(&path_names_key)
        .await
        .expect("SMEMBERS path_names must succeed");
    let expected = KeySchema::path_name_member(session_name, session_id);
    assert!(
        members.contains(&expected),
        "Step A path-names composite set MUST carry the composite \
         (session_name={session_name}, session_id={session_id}) post-Step-B-failure; \
         expected member={expected:?}; got members={members:?}. A missing \
         composite here means Step A's path-name SADD was rolled back by \
         the Step B injection (cross-step atomicity invariant broken) or \
         the assembler did not emit the composite (Task 3 assembler \
         regression)."
    );
}

/// Assert every Step B target key (`le:{sid}:sess`, `le:{sid}:layer:{hex(name)}`,
/// `le:{sid}:layers`) is absent in the live keyspace. Pins the
/// "Step B's MULTI/EXEC never reached EXEC" invariant.
async fn assert_session_scope_absent(
    harness: &CheckpointHarness,
    session_id: &str,
    layer_name: &str,
) {
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
        "Step B session-meta hash MUST be absent post-Step-B-failure — its \
         HSET was queued inside the same MULTI/EXEC that the injection \
         aborted, so the EXEC never ran. EXISTS=1 here means the injection \
         seam fired AFTER the MULTI/EXEC committed (per-step seam \
         placement regression)."
    );

    let layers_index_key = harness.tv.key_schema().layers_index(session_id);
    let layers_index_exists: bool = harness
        .tv
        .client()
        .pool
        .exists(&layers_index_key)
        .await
        .expect("EXISTS layers_index probe must succeed");
    assert!(
        !layers_index_exists,
        "Step B layers-index set MUST be absent post-Step-B-failure; \
         EXISTS=1 means the SADD on `le:{{sid}}:layers` committed despite \
         the injection."
    );

    let mapping_key = harness
        .tv
        .key_schema()
        .layer_mapping(session_id, layer_name);
    let mapping_exists: bool = harness
        .tv
        .client()
        .pool
        .exists(&mapping_key)
        .await
        .expect("EXISTS layer_mapping probe must succeed");
    assert!(
        !mapping_exists,
        "Step B layer-mapping SET MUST be absent post-Step-B-failure; \
         EXISTS=1 means the SET on `le:{{sid}}:layer:{{hex({layer_name})}}` \
         committed despite the injection."
    );
}

// =============================================================================
// Scenario 1 — Step B fault leaves Step A landed atomically + session-scope absent.
// =============================================================================

/// Pin Epic AC #3 — Step B atomicity. A single Step B failure during
/// `apply_session_checkpoint` MUST leave Step A's path-index writes
/// fully committed AND every Step B target key absent. The RPC
/// surfaces `Code::Unavailable` (the standard retriable-error mapping
/// for `CheckpointError::Transient` via `ServiceError::Unavailable`).
///
/// Step-by-step rationale:
///
/// 1. `clear_apply_step_fault_injectors_for_test()` at top — the
///    `APPLY_STEP_FAULT_INJECTORS` map is process-global, so a sibling
///    test in the same `cargo test` invocation could have left a queue
///    populated. Clearing here makes this test order-independent.
/// 2. `setup_checkpoint_harness` with `tick_secs=1` for fast checkpoint
///    convergence.
/// 3. `create_session_and_layer_and_wait_for_checkpoint` runs the real
///    `handle_create_session` + `handle_create_layer` RPCs so the
///    lifecycle hook fires and the session-checkpoint dispatcher lands
///    a blob on disk. Pre-call recovery-metric snapshot captured.
/// 4. `full_valkey_wipe` SCANs + UNLINKs EVERY `tv.prefix()*` key.
///    Sanity assertion on `le:{sid}:sess` catches test-infrastructure
///    regressions before the fault-injection phase obscures them.
/// 5. `invalidate_session_cache` — see helper docstring.
/// 6. Install ONE Step B failure via the per-step fault-injection seam.
///    `Transient` (not `Permanent`) so the cache funnel maps it to
///    `ServiceError::Unavailable` per `src/grpc/session_cache.rs`
///    `match err { Transient => Unavailable, ... }`.
/// 7. Dispatch `handle_delete_layer` through the real RPC entry point.
///    The cache funnel's `get_base_path` sees NotFound (le:{sid}:sess
///    is gone), fires session-recovery, the recovery hook runs Step A
///    successfully then short-circuits Step B on the injected
///    failure, propagates `Err(Transient)` → `SessionRecoveryOutcome::Error`
///    → `ServiceError::Unavailable` → `Code::Unavailable`.
/// 8. Assert `Code::Unavailable`.
/// 9. Assert Step A's path-index writes landed atomically — both the
///    legacy session-id set and the composite name set carry the
///    expected members.
/// 10. Assert every Step B target key is absent.
/// 11. Assert `error{scope=session}` cell ticked by exactly 1; no
///     other session-scope cell ticked.
/// 12. Graceful shutdown + clear injectors at the bottom (cross-test
///     bleed defence).
// `clippy::await_holding_lock` — intentional: the `std::sync::Mutex`
// guard exists precisely to serialize this test against its two siblings
// in this file (they all consume the same process-global fault-injector
// queue). No other async task contends for this guard within a single
// test invocation, so the standard "don't hold a non-async lock across
// await" rule doesn't apply — the lock has zero in-test contention and
// the cross-test contention is the whole point.
#[allow(clippy::await_holding_lock)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn step_b_session_scope_failure_leaves_path_index_landed_and_no_session_state() {
    let _serial = lock_fault_injector_serial();
    clear_apply_step_fault_injectors_for_test();
    let harness = setup_checkpoint_harness(1).await;

    let baseline = create_session_and_layer_and_wait_for_checkpoint(
        &harness,
        SESSION_BASE_PATH_STEP_B_FAIL,
        SESSION_NAME_STEP_B_FAIL,
        LAYER_NAME,
    )
    .await;
    let session_id = baseline.session_id.clone();

    // Full Valkey wipe — every key under the test prefix gone.
    full_valkey_wipe(&harness.tv).await;
    assert_session_meta_absent_after_wipe(&harness, &session_id).await;
    invalidate_session_cache(&harness, &session_id);

    // Install a single Step B failure. `Transient` so the cache funnel's
    // error mapper routes through `Unavailable` (retriable per
    // SDD §18.1 transient-error contract).
    let mut queue: VecDeque<CheckpointError> = VecDeque::new();
    queue.push_back(CheckpointError::Transient {
        source: "injected: simulated Step B EXEC failure for AC #3".into(),
    });
    set_apply_step_fault_injector_for_test(ApplyStep::StepB, queue);

    // Dispatch DeleteLayer. The cache funnel's `get_base_path` sees the
    // wiped `le:{sid}:sess`, fires session-recovery, the apply pipeline
    // commits Step A, then short-circuits Step B on the injected error,
    // surfacing Transient → ServiceError::Unavailable → Code::Unavailable.
    let status = handle_delete_layer(
        &harness.svc,
        make_unary_token_request(DeleteLayerRequest {
            session_id: session_id.clone(),
            layer_name: LAYER_NAME.to_owned(),
        }),
    )
    .await
    .expect_err(
        "DeleteLayer MUST fail with Unavailable when session-recovery's Step B is \
         fault-injected. Ok here means: (a) the injection seam never fired \
         (per-step plumbing regression), (b) the cache funnel did not invoke the \
         recovery hook (with_recovery_hook wiring regression), or (c) the apply \
         pipeline swallowed the Transient and pretended to succeed.",
    );
    assert_eq!(
        status.code(),
        Code::Unavailable,
        "Step B Transient injection MUST surface as gRPC Code::Unavailable per \
         src/grpc/session_cache.rs's `match err {{ Transient => Unavailable, ... }}` \
         arm; got {status:?}"
    );

    // Step A atomicity: both path-index halves committed under the
    // `{hex(base_path)}` slot.
    assert_path_index_carries_session(&harness, SESSION_BASE_PATH_STEP_B_FAIL, &session_id).await;
    assert_path_names_carries_expected_composite(
        &harness,
        SESSION_BASE_PATH_STEP_B_FAIL,
        &session_id,
        SESSION_NAME_STEP_B_FAIL,
    )
    .await;

    // Step B never reached EXEC: every session-scope key absent.
    assert_session_scope_absent(&harness, &session_id, LAYER_NAME).await;

    // Recovery-metric: `error{scope=session}` ticked by exactly 1; all
    // other session-scope cells unchanged. The `error` arm is hit by
    // `src/checkpoint/session_recovery.rs:393-400`'s `Err(_)` match arm
    // — apply returns Err, the hook increments `"error"` then returns
    // `SessionRecoveryOutcome::Error`.
    assert_eq!(
        session_recovery_metric(&harness, "error") - baseline.error_pre,
        1,
        "exactly ONE session-recovery attempt must have reported `error` after \
         the Step B injection; restored_pre={}, error_pre={}",
        baseline.restored_pre,
        baseline.error_pre,
    );
    assert_eq!(
        session_recovery_metric(&harness, "restored"),
        baseline.restored_pre,
        "post-Step-B-failure `restored{{session}}` MUST equal pre-call snapshot — \
         a non-zero delta here means the apply pipeline somehow succeeded \
         despite the injection."
    );
    assert_eq!(
        session_recovery_metric(&harness, "not_recoverable"),
        baseline.not_recoverable_pre,
        "post-Step-B-failure `not_recoverable{{session}}` MUST equal pre-call \
         snapshot — the checkpoint blob was readable so the apply branch ran, \
         not the NoCheckpoint arm."
    );
    assert_eq!(
        session_recovery_metric(&harness, "already_present"),
        baseline.already_present_pre,
        "post-Step-B-failure `already_present{{session}}` MUST equal pre-call \
         snapshot — the wipe drove the pre-acquire EXISTS check to false."
    );

    clear_apply_step_fault_injectors_for_test();
    harness.cancel.cancel();
    let _ = harness.loop_handle.await;
}

// =============================================================================
// Scenario 2 — Retry after Step B failure converges (Task 5 idempotency
// contract pinned end-to-end).
// =============================================================================

/// Pin Task 5's idempotency contract: a re-applied `SessionCheckpoint`
/// after a Step B-only failure converges to the same materialized
/// state. The second `try_recover_session` call:
///
/// - Re-runs Step A on the `{hex(base_path)}` slot. Both SADDs are
///   idempotent (SADD of an existing member is a no-op returning 0), so
///   Step A is effectively a no-op the second time around — the
///   path-index members already landed by the first call's Step A are
///   unchanged.
/// - Runs Step B for real (fault injector is empty after a single pop).
///   HSET / SET / SADD are all idempotent at the per-key level, and the
///   wiped session-scope keys land cleanly.
///
/// The DeleteLayer RPC then proceeds against the restored session
/// state: `get_base_path` returns the base path, the layer-mapping
/// resolves (Step B restored it), and the delete-cascade UNLINKs the
/// per-`(session_id, layer_name)` family.
///
/// Step-by-step rationale follows scenario 1, with the additional steps:
///
/// 9. After the first RPC fails, drain the injector via
///    `clear_apply_step_fault_injectors_for_test()`. A second
///    `try_recover_session` call would now find an empty queue, run
///    Step A (no-op) and Step B (succeeds).
/// 10. Invalidate the session cache a SECOND time — the first DeleteLayer's
///     `get_base_path` call did NOT populate the cache (it returned an
///     error before the success arm cached anything), so this is a no-op
///     in practice, but it documents the intent and protects against a
///     future refactor that would pre-cache on the error path.
/// 11. Dispatch DeleteLayer a SECOND time. `Ok` is the success signal;
///     the recovery branch fired again (via the cache-funnel re-attempt),
///     Step A no-op'd, Step B succeeded.
/// 12. Assert `restored{scope=session}` ticked by exactly 1 (the second
///     call's successful apply); `error{scope=session}` still equals 1
///     (the first call's failure).
// See `step_b_session_scope_failure_leaves_path_index_landed_and_no_session_state`
// for the rationale on `clippy::await_holding_lock`.
#[allow(clippy::await_holding_lock)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn recovery_retry_after_step_b_failure_converges() {
    let _serial = lock_fault_injector_serial();
    clear_apply_step_fault_injectors_for_test();
    let harness = setup_checkpoint_harness(1).await;

    let baseline = create_session_and_layer_and_wait_for_checkpoint(
        &harness,
        SESSION_BASE_PATH_RETRY,
        SESSION_NAME_RETRY,
        LAYER_NAME,
    )
    .await;
    let session_id = baseline.session_id.clone();

    full_valkey_wipe(&harness.tv).await;
    assert_session_meta_absent_after_wipe(&harness, &session_id).await;
    invalidate_session_cache(&harness, &session_id);

    // ── First RPC: Step B injected to fail ──────────────────────────────
    let mut queue: VecDeque<CheckpointError> = VecDeque::new();
    queue.push_back(CheckpointError::Transient {
        source: "injected: first attempt Step B EXEC failure".into(),
    });
    set_apply_step_fault_injector_for_test(ApplyStep::StepB, queue);

    let first_status = handle_delete_layer(
        &harness.svc,
        make_unary_token_request(DeleteLayerRequest {
            session_id: session_id.clone(),
            layer_name: LAYER_NAME.to_owned(),
        }),
    )
    .await
    .expect_err(
        "First DeleteLayer must fail with Unavailable when Step B is injected to fail \
         (mirrors scenario 1's primary assertion; if the first call were to succeed, \
         the retry-convergence test below would be vacuous).",
    );
    assert_eq!(
        first_status.code(),
        Code::Unavailable,
        "first DeleteLayer must surface Code::Unavailable for the Step B \
         Transient injection; got {first_status:?}"
    );

    // Pre-retry metric snapshot: `error` MUST have ticked by 1, every
    // other cell unchanged.
    assert_eq!(
        session_recovery_metric(&harness, "error") - baseline.error_pre,
        1,
        "after first (failed) RPC, `error{{session}}` MUST have ticked by 1"
    );
    assert_eq!(
        session_recovery_metric(&harness, "restored"),
        baseline.restored_pre,
        "after first (failed) RPC, `restored{{session}}` MUST be unchanged"
    );

    // ── Drain injector, re-invalidate cache, dispatch retry ──────────────
    clear_apply_step_fault_injectors_for_test();

    // Re-invalidate the cache. The first call's `get_base_path` failed
    // before reaching the cache-insert arm (the cache only caches on
    // `Ok(base_path)` per session_cache.rs:170-176), so this is a
    // defensive no-op — but it guarantees the second call exercises the
    // recovery branch even if a future refactor pre-caches on the error
    // path.
    invalidate_session_cache(&harness, &session_id);

    let retry_resp = handle_delete_layer(
        &harness.svc,
        make_unary_token_request(DeleteLayerRequest {
            session_id: session_id.clone(),
            layer_name: LAYER_NAME.to_owned(),
        }),
    )
    .await
    .expect(
        "Retry DeleteLayer MUST succeed after the fault injector is drained — \
         Task 5's idempotency contract requires that re-applying the checkpoint \
         converges. A failure here means: (a) Step A's idempotent SADD broke \
         (the path-index members landed by the first call's Step A are now \
         blocking the second call somehow), (b) Step B's idempotency broke \
         (HSET / SET / SADD over the wiped session-scope keys does not converge), \
         or (c) the recovery branch did not fire on the retry (cache funnel \
         caching the error somehow).",
    );
    drop(retry_resp);

    // `restored{session}` ticked by exactly 1 (the retry's success);
    // `error` still 1 from the first call. The other cells stay at
    // pre-call snapshot values.
    assert_eq!(
        session_recovery_metric(&harness, "restored") - baseline.restored_pre,
        1,
        "retry DeleteLayer MUST tick `restored{{session}}` by exactly 1 — pins \
         Task 5's idempotency contract end-to-end. restored_pre={}, observed delta={}",
        baseline.restored_pre,
        session_recovery_metric(&harness, "restored") - baseline.restored_pre,
    );
    assert_eq!(
        session_recovery_metric(&harness, "error") - baseline.error_pre,
        1,
        "after retry, `error{{session}}` total delta MUST still be 1 (from the \
         first failed call); a delta of 2 means the retry's apply ALSO errored, \
         contradicting the success-on-retry contract"
    );
    assert_eq!(
        session_recovery_metric(&harness, "not_recoverable"),
        baseline.not_recoverable_pre,
        "`not_recoverable{{session}}` MUST equal pre-call snapshot after retry"
    );
    assert_eq!(
        session_recovery_metric(&harness, "already_present"),
        baseline.already_present_pre,
        "`already_present{{session}}` MUST equal pre-call snapshot after retry"
    );

    // Confirm Step B's writes actually landed on the retry — session-meta
    // is present (the HSET hit), and the layers-index carries the
    // hex-encoded layer name (the SADD hit). The mapping key would also
    // have landed initially, but DeleteLayer's cascade UNLINKs it as
    // part of the delete (which is precisely the point — the RPC
    // succeeded against the restored state). We assert on session-meta
    // and layers-index because they are observably present post-DeleteLayer.
    let session_meta_key = harness.tv.key_schema().session_meta(&session_id);
    let meta_after: HashMap<String, String> = harness
        .tv
        .client()
        .pool
        .hgetall(&session_meta_key)
        .await
        .expect("HGETALL session_meta post-retry must succeed");
    assert!(
        !meta_after.is_empty(),
        "post-retry session_meta MUST be present — Step B's HSET landed on the \
         retry path; got empty hash"
    );
    assert_eq!(
        meta_after.get("name"),
        Some(&SESSION_NAME_RETRY.to_string()),
        "post-retry session_meta.name MUST equal the original session name; \
         got {:?}",
        meta_after.get("name")
    );
    assert_eq!(
        meta_after.get("base_path"),
        Some(&SESSION_BASE_PATH_RETRY.to_string()),
        "post-retry session_meta.base_path MUST equal the original base_path; \
         got {:?}",
        meta_after.get("base_path")
    );

    // DeleteLayer's cascade removes the layer name from the layers-index
    // via SREM. The layers-index key itself may or may not still exist
    // depending on whether other layers were present (we only created
    // one), so we assert the deleted name is NOT a member — the
    // observable post-DeleteLayer invariant.
    let layers_index_key = harness.tv.key_schema().layers_index(&session_id);
    let layers_after: Vec<String> = harness
        .tv
        .client()
        .pool
        .smembers(&layers_index_key)
        .await
        .expect("SMEMBERS layers_index post-retry must succeed");
    let hex_layer = hex_encode(LAYER_NAME);
    assert!(
        !layers_after.contains(&hex_layer),
        "post-DeleteLayer cascade MUST have SREMed hex({LAYER_NAME}) from \
         layers_index; got {layers_after:?}. A surviving member here means \
         the cascade short-circuited."
    );

    clear_apply_step_fault_injectors_for_test();
    harness.cancel.cancel();
    let _ = harness.loop_handle.await;
}

// =============================================================================
// Scenario 3 — Step B failure does not corrupt the path-index (no rollback).
// =============================================================================

/// Pin the "applier is purely additive" invariant: a Step B failure
/// MUST NOT roll back, mutate, or otherwise corrupt Step A's path-index
/// writes. Even though Step A's SADDs may leave path-index members
/// pointing at a non-existent session (the "ghost-but-functional"
/// failure mode's mirror image), reconciliation is owned by
/// `ListSessions`' opportunistic-cleanup walk (SDD §7.13) and the
/// orphan-sweep — NOT by the applier.
///
/// This scenario is the "scenario 1 plus end-state pinning" — scenario 1
/// asserts the post-Step-B-failure shape immediately; this scenario
/// further asserts that:
///
/// - The exact path-index members landed by Step A are still present
///   (byte-equality with the expected `(session_id, layer_name\0session_id)`
///   shape).
/// - SCARD on both path-index keys returns the expected counts.
/// - A SECOND independent observation pass (after a short delay
///   simulating a downstream caller) yields the same shape — no hidden
///   background task is reconciling the ghost members on its own.
///
/// Together these pin the contract that the applier is purely
/// additive: it never DELs, SREMs, or HDELs. Adding a destructive
/// pre-clean step in `apply_session_checkpoint` would be visible here
/// as a path-index member count below the expected post-Step-A value.
// See `step_b_session_scope_failure_leaves_path_index_landed_and_no_session_state`
// for the rationale on `clippy::await_holding_lock`.
//
// Harness wiring: `setup_checkpoint_harness` spawns ONLY
// `spawn_checkpoint_loop` — no orphan-sweep, no presence-sweeper, no
// session-tokens-sweep, no GC. The two SMEMBERS observation passes
// below are therefore run back-to-back (no inter-pass sleep). They
// verify in-test path-index stability against a path-index that no
// background reconciler in this harness is configured to touch. If a
// future change ADDS a reconciler to `setup_checkpoint_harness`, the
// reconciler MUST be armed by an explicit deterministic seam following
// the b.sxc pattern (e.g., a `Notify`-gated "one tick" trigger) and the
// observation pair below must be updated to await that signal — never
// reintroduce a wall-clock `sleep` here.
#[allow(clippy::await_holding_lock)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn partial_step_b_failure_does_not_corrupt_existing_path_index() {
    let _serial = lock_fault_injector_serial();
    clear_apply_step_fault_injectors_for_test();
    let harness = setup_checkpoint_harness(1).await;

    // Harness-wiring assertion. `setup_checkpoint_harness` only spawns
    // `spawn_checkpoint_loop` (see helpers.rs `setup_checkpoint_harness`
    // body — no `spawn_orphan_sweep`, no `spawn_presence_sweeper`, no
    // `spawn_session_tokens_sweep`, no `spawn_gc`). The two
    // back-to-back SMEMBERS observation passes below assume that no
    // other background actor can mutate the path-index between calls.
    // Asserting the loop handle is still live confirms the harness
    // hasn't gone sideways (e.g., the loop panicked during setup) and
    // is the only reconciler we need to reason about. A future
    // regression that wires an additional background task into this
    // harness MUST add a deterministic seam (per the b.sxc pattern)
    // and re-justify this assertion shape before merging.
    assert!(
        !harness.loop_handle.is_finished(),
        "checkpoint loop handle must still be live — the harness wires \
         spawn_checkpoint_loop and no other background task; any future \
         additional reconciler (orphan-sweep, presence-sweeper, gc) must \
         arrive with a deterministic seam following the b.sxc pattern."
    );

    let baseline = create_session_and_layer_and_wait_for_checkpoint(
        &harness,
        SESSION_BASE_PATH_NO_CORRUPT,
        SESSION_NAME_NO_CORRUPT,
        LAYER_NAME,
    )
    .await;
    let session_id = baseline.session_id.clone();

    full_valkey_wipe(&harness.tv).await;
    assert_session_meta_absent_after_wipe(&harness, &session_id).await;
    invalidate_session_cache(&harness, &session_id);

    let mut queue: VecDeque<CheckpointError> = VecDeque::new();
    queue.push_back(CheckpointError::Transient {
        source: "injected: Step B EXEC failure for no-corrupt scenario".into(),
    });
    set_apply_step_fault_injector_for_test(ApplyStep::StepB, queue);

    let status = handle_delete_layer(
        &harness.svc,
        make_unary_token_request(DeleteLayerRequest {
            session_id: session_id.clone(),
            layer_name: LAYER_NAME.to_owned(),
        }),
    )
    .await
    .expect_err("DeleteLayer must fail with Unavailable when Step B is injected to fail");
    assert_eq!(status.code(), Code::Unavailable);

    // First observation: immediately post-failure.
    let path_index_key = harness
        .tv
        .key_schema()
        .path_index(SESSION_BASE_PATH_NO_CORRUPT);
    let path_names_key = harness
        .tv
        .key_schema()
        .path_names_index(SESSION_BASE_PATH_NO_CORRUPT);

    let path_index_members_first: Vec<String> = harness
        .tv
        .client()
        .pool
        .smembers(&path_index_key)
        .await
        .expect("SMEMBERS path_index (first observation) must succeed");
    let path_names_members_first: Vec<String> = harness
        .tv
        .client()
        .pool
        .smembers(&path_names_key)
        .await
        .expect("SMEMBERS path_names (first observation) must succeed");

    assert!(
        path_index_members_first.contains(&session_id),
        "first observation: path_index MUST carry session_id={session_id}; got \
         {path_index_members_first:?}"
    );
    assert_eq!(
        path_index_members_first.len(),
        1,
        "first observation: path_index SCARD MUST equal 1 (one Step A SADD on \
         the legacy session-id-only set); got {} members: {:?}",
        path_index_members_first.len(),
        path_index_members_first
    );

    let expected_composite = KeySchema::path_name_member(SESSION_NAME_NO_CORRUPT, &session_id);
    assert!(
        path_names_members_first.contains(&expected_composite),
        "first observation: path_names MUST carry composite {expected_composite:?}; \
         got {path_names_members_first:?}"
    );
    assert_eq!(
        path_names_members_first.len(),
        1,
        "first observation: path_names SCARD MUST equal 1 (one Step A SADD of \
         the `session_name\\0session_id` composite — matches the live \
         `SessionStore::create_once` write at `src/storage/session.rs:308`); \
         got {} members: {:?}",
        path_names_members_first.len(),
        path_names_members_first
    );

    // Second observation pass — back-to-back with the first (no sleep).
    // The harness wires `spawn_checkpoint_loop` and no other background
    // task (see the harness-wiring assertion at the top of this scenario
    // and the doc-comment above the `#[tokio::test]` attribute). The
    // checkpoint loop never touches `path_index` / `path_names_index`
    // keys, so the only actor that could mutate them between the two
    // SMEMBERS calls is the applier itself. The byte-equality assertion
    // below therefore directly pins the "applier is purely additive"
    // contract — no sleep is required to "give a reconciler time to
    // run" because there is no reconciler in this harness. A wall-clock
    // sleep here would buy nothing but flake budget.
    let path_index_members_second: Vec<String> = harness
        .tv
        .client()
        .pool
        .smembers(&path_index_key)
        .await
        .expect("SMEMBERS path_index (second observation) must succeed");
    let path_names_members_second: Vec<String> = harness
        .tv
        .client()
        .pool
        .smembers(&path_names_key)
        .await
        .expect("SMEMBERS path_names (second observation) must succeed");

    // Sort both observations so non-deterministic SMEMBERS ordering
    // doesn't make the comparison flaky.
    let mut first_index_sorted = path_index_members_first;
    let mut second_index_sorted = path_index_members_second;
    let mut first_names_sorted = path_names_members_first;
    let mut second_names_sorted = path_names_members_second;
    first_index_sorted.sort();
    second_index_sorted.sort();
    first_names_sorted.sort();
    second_names_sorted.sort();

    assert_eq!(
        second_index_sorted, first_index_sorted,
        "path_index members MUST be byte-identical across the two observations \
         — a divergence indicates a background reconciliation task touched the \
         path-index in the immediate post-failure window, which would violate \
         the applier's purely-additive contract OR indicate a non-applier \
         actor is rolling back Step A."
    );
    assert_eq!(
        second_names_sorted, first_names_sorted,
        "path_names members MUST be byte-identical across the two observations \
         — same rationale as path_index."
    );

    // Session-scope keys still absent on the second observation —
    // doubles as a sanity-check that the Step B inject didn't
    // accidentally fire post-MULTI/EXEC.
    assert_session_scope_absent(&harness, &session_id, LAYER_NAME).await;

    clear_apply_step_fault_injectors_for_test();
    harness.cancel.cancel();
    let _ = harness.loop_handle.await;
}
