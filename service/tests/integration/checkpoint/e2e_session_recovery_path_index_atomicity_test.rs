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

//! End-to-end integration tests for Epic AC #4 — path-index restoration
//! (Step A) is independently atomic, and Step A is independent from Step B
//! (Subtask `t3.fgj.7v.o7.xt` under Task `t2.fgj.7v.o7`, Epic `t1.fgj.7v`,
//! Bee `b.fgj`).
//!
//! Epic AC #4 pairs with AC #3: Task 5's session-recovery applier
//! ([`apply_session_checkpoint`](live_edit::checkpoint::session)) splits the
//! restore into two independently-atomic MULTI/EXEC transactions because
//! the path-index keys (`le:path:{hex(base_path)}*`, hashed on
//! `{hex(base_path)}`) and the session-scoped keys (`le:{sid}:sess`,
//! `le:{sid}:layers`, `le:{sid}:layer:{hex(name)}`, hashed on
//! `{session_id}`) cannot share a Cluster slot. The split is ordered
//! path-index-FIRST per SR-2.7 because that asymmetry makes a Step-A-only
//! state self-healable: a ghost path-index member surfaces in
//! `ListSessions::reconcile_ghost_members` and the next recovery walk
//! either restores the session (turning the ghost into a live member) or
//! sweeps the orphan. The reverse — session-scoped keys present without a
//! path-index entry — would be a ghost-but-functional session with NO
//! self-healing path.
//!
//! Sibling Subtask `t3.fgj.7v.o7.6f` covers Step B injection (AC #3
//! contract); this Subtask covers:
//!
//! 1. **Step A injection leaves NO partial state.** Path-index MULTI/EXEC
//!    fails before any SADD lands; Step B never runs (the applier
//!    short-circuits on Step A's `Err` return).
//! 2. **Step A atomicity under partial-member failure.** Task 5 batches
//!    every path-index SADD (legacy `path_index` + every composite
//!    `path_names_index` member) into ONE MULTI/EXEC, so a fault BEFORE
//!    EXEC leaves BOTH sets empty — no "some members landed, others
//!    didn't" partial state.
//! 3. **Cross-step independence — Step A succeeds + Step B fails leaves
//!    the path-index populated AND demonstrates SR-2.7's self-healing
//!    contract.** A subsequent `ListSessions` walk surfaces the ghost
//!    path-name member (Step A wrote it) and `reconcile_ghost_members`
//!    SREMs it once the corresponding session-checkpoint blob is gone.
//!
//! All three scenarios consume the per-step fault-injection seam exposed
//! by `live_edit::checkpoint::session::{ApplyStep,
//! set_apply_step_fault_injector_for_test, clear_apply_step_fault_injectors_for_test}`
//! under the `test-checkpoint-fault-injection` cargo feature. Each
//! scenario clears the queues at the top of its body so a prior leftover
//! cannot bleed across tests (the queues are process-global state — see
//! the seam's docstring).
//!
//! Each scenario also acquires the cross-file
//! `APPLY_STEP_FAULT_INJECTOR_TEST_SERIAL` mutex from `helpers.rs` via
//! [`lock_fault_injector_serial`] BEFORE clearing the queues. The map is
//! shared with `e2e_session_recovery_atomicity_test.rs`'s Step B
//! scenarios; without the cross-file gate, one file's installed fault
//! could be popped by the other file's `apply_session_checkpoint` call
//! mid-flight.
//!
//! ## Why each scenario uses a session-resolving RPC to drive recovery
//!
//! The applier is only invoked through `SessionRecoveryHook::try_recover_session`,
//! which is in turn only invoked from the per-handler cache funnel
//! (`SessionBasePathCache::get_base_path`) and `reconcile_ghost_members`
//! in `handle_list_sessions`. Direct unit-test reach into
//! `apply_session_checkpoint` would not exercise the wiring path that
//! flows the applier's error classification through to the gRPC caller —
//! the integration-test surface here pins that the `Transient` →
//! `Unavailable` mapping in the cache funnel still surfaces the
//! injected error verbatim to the client.
//!
//! Sandbox env note: this test uses `TestValkey::setup`, which reads
//! `TEST_VALKEY_URL`. Under the project sandbox the URL must include the
//! password and `host.docker.internal` host; see CLAUDE.md.

#![cfg(feature = "test-checkpoint-fault-injection")]

use std::collections::VecDeque;

use fred::interfaces::{KeysInterface, SetsInterface};
use tonic::Code;

use live_edit::checkpoint::session::{
    clear_apply_step_fault_injectors_for_test, set_apply_step_fault_injector_for_test, ApplyStep,
};
use live_edit::checkpoint::CheckpointError;
use live_edit::grpc::layer::handle_create_layer;
use live_edit::grpc::proto::{CreateSessionRequest, ListSessionsRequest};
use live_edit::grpc::session::{handle_create_session, handle_list_sessions};
use live_edit::storage::keys::KeySchema;

use super::helpers::*;
use crate::common::owned_delta::{delta_to_ompf_bytes, empty_delta};
// Each scenario's session-recovery failure surfaces from `CreateLayer`'s
// base-path lookup, before its commit marker, so the `UNAVAILABLE` carries the
// b.xd1 pre-commit hint (`UPSTREAM_UNAVAILABLE`): this call applied nothing.
use crate::common::refusal_detail::assert_upstream_unavailable;

// ── Test constants ───────────────────────────────────────────────────────────
//
// Distinct `base_path` values per scenario keep the three test bodies'
// path-index keys disjoint inside one Valkey instance — proptest-style
// cross-test bleed is impossible by construction even if the tests run
// concurrently against a shared TEST_VALKEY_URL.

const SCENARIO_1_BASE_PATH: &str = "/stage/path-index-atomicity-step-a-only";
const SCENARIO_1_SESSION_NAME: &str = "path-index-atomicity-step-a-only";
const SCENARIO_1_LAYER_NAME: &str = "trigger-layer";

const SCENARIO_2_BASE_PATH: &str = "/stage/path-index-atomicity-partial-member";
const SCENARIO_2_SESSION_NAME: &str = "path-index-atomicity-partial-member";
const SCENARIO_2_LAYER_NAME: &str = "trigger-layer";

const SCENARIO_3_BASE_PATH: &str = "/stage/path-index-atomicity-cross-step-self-heal";
const SCENARIO_3_SESSION_NAME: &str = "path-index-atomicity-cross-step-self-heal";
const SCENARIO_3_LAYER_NAME: &str = "trigger-layer";

// ── Streaming-request framing helpers — imported from
// `tests/integration/checkpoint/helpers.rs` via `use super::helpers::*;`
// (b.dzd Phase 1 deduplication). ─────────────────────────────────────────

// ── Shared baseline setup ────────────────────────────────────────────────────
//
// CreateSession + CreateLayer + wait for the session-checkpoint blob to
// land on disk, then wipe `le:{sid}*` (preserving the LocalDisk-backed
// blob and the path-index `*:names` set per `wipe_session_state`'s
// contract). The wipe is the cue that subsequent session-resolving RPCs
// will trigger recovery via `SessionBasePathCache::get_base_path`'s
// NotFound branch.
//
// Returns the minted `session_id`. CreateLayer's `object_id` is
// intentionally discarded — none of the scenarios in this file assert
// post-recovery layer state; the AC under test is about path-index
// atomicity.

async fn create_session_layer_and_wipe(
    harness: &CheckpointHarness,
    base_path: &str,
    session_name: &str,
    layer_name: &str,
) -> String {
    let create_session_resp = handle_create_session(
        &harness.svc,
        make_unary_token_request(CreateSessionRequest {
            path: base_path.to_owned(),
            name: session_name.to_owned(),
        }),
    )
    .await
    .expect("CreateSession must succeed against the AllowAllPermissions harness");
    let session_id = create_session_resp.into_inner().session_id;
    assert!(
        !session_id.is_empty(),
        "CreateSession must mint a non-empty session_id"
    );

    // CreateLayer ensures the session-checkpoint blob carries layer
    // state in `layer_mappings` / `layers_index`; the wipe-and-recover
    // path through the applier exercises both Step A's SADDs into the
    // path-index AND Step B's HSET / SET / SADD into the session-scoped
    // keys. A CreateSession-only baseline would still trigger recovery,
    // but Step B's payload would be near-empty and a path-index
    // contribution regression would be the only AC under test — we want
    // both halves of the applier to be exercised.
    let ompf_bytes = delta_to_ompf_bytes(&empty_delta(), &[]);
    let stream = streaming_create(vec![
        make_create_header(&session_id, layer_name),
        make_create_data(ompf_bytes),
    ]);
    let _create_layer_resp =
        handle_create_layer(&harness.svc, make_streaming_token_request(stream))
            .await
            .expect("CreateLayer must succeed for baseline setup");

    // Wait for the fire-and-forget session-checkpoint writer to land
    // the blob on disk before the wipe step removes the live Valkey
    // state. Without the blob, recovery's Step 5 (`get_session_checkpoint`)
    // would short-circuit at `NoCheckpoint` and the fault injectors on
    // Steps A/B would never fire — silent test-passes-for-wrong-reason.
    wait_for_session_checkpoint(&harness.store, &session_id, 10).await;

    // Wipe `le:{sid}*`: session_meta hash, every layer-mapping, the
    // layers-index. Path-index (`le:path:{hex(base_path)}*`) survives —
    // SR-2.7's asymmetric self-healing contract relies on the path-index
    // surviving session-scoped wipes. The recovery path will re-execute
    // Step A (idempotent SADD on the surviving members) and Step B
    // (rebuilds session_meta from scratch).
    wipe_session_state(&harness.tv, &session_id).await;

    // Drop the cache so the post-wipe RPC's `get_base_path` actually
    // consults `session_store` (rather than HIT on the pre-wipe entry).
    // Production cross-replica invalidation drives this via the
    // `session:deleted` pub/sub fanout; under E2E test we invalidate
    // explicitly. Without this, the cache HIT on the pre-wipe base_path
    // would let the post-wipe RPC succeed WITHOUT consulting recovery
    // — Step A/B fault injectors would never fire and every test in
    // this file would silently pass for the wrong reason.
    harness.svc.session_base_path_cache.invalidate(&session_id);

    session_id
}

// ── Fault-injection error constructor ────────────────────────────────────────
//
// `CheckpointError::Transient` is the natural classification for a Step
// A/B EXEC failure — it matches what real Valkey-side failures would
// surface as in production, and it is the variant that the cache funnel
// maps to `ServiceError::Unavailable` (vs. `Internal` for Permanent /
// Conflict). The scenarios in this file all want the failure to surface
// as `Unavailable` to the gRPC caller (`Code::Unavailable`), so the
// queued error MUST be `Transient`. A regression that flipped Step A's
// classification to `Permanent` would surface here as the RPC returning
// `Internal` and the scenarios in this file would fail at the status-
// code assertion — exactly the wiring contract we want pinned.

fn make_step_a_fault(label: &str) -> CheckpointError {
    CheckpointError::Transient {
        source: format!("step-a fault-injection: {label}").into(),
    }
}

fn make_step_b_fault(label: &str) -> CheckpointError {
    CheckpointError::Transient {
        source: format!("step-b fault-injection: {label}").into(),
    }
}

// =============================================================================
// Scenario 1 — Step A path-index failure leaves NO partial state.
// =============================================================================

/// Pin Epic AC #4's first contract: a Step A failure leaves zero
/// path-index members landed AND zero session-scoped members landed.
/// Step B is never entered (the applier short-circuits on Step A's `Err`
/// return before reaching `pop_apply_step_fault(StepB)`).
///
/// Why each step is load-bearing:
///
/// - The baseline CreateSession + CreateLayer + wait-for-checkpoint
///   shape ensures the session-checkpoint blob is on disk before the
///   wipe. Without it, the post-wipe recovery would short-circuit at
///   Step 5 (`NoCheckpoint`) and the Step A fault would never be
///   consumed from the queue — the test would pass silently with
///   `restored == 0` and the path-index empty (because the wipe step
///   alone does not populate the path-index from the recovery side).
/// - `clear_apply_step_fault_injectors_for_test()` at the top drains
///   any leftover queue from a prior test invocation in the same
///   process. The fault-injection state is process-global per the
///   seam's contract; without the clear, a parallel-running scenario
///   could install a queue entry that this test consumes.
/// - `wipe_session_state` removes `le:{sid}:sess` and `le:{sid}:layer:*`
///   but preserves the path-index entries that CreateSession wrote.
///   After the wipe, the path-index members are still there from the
///   live-write path. So to assert "Step A failure left NO path-index
///   members landed BY RECOVERY", we instead assert that the wipe
///   removed the path-index too — and the proper helper here is to
///   wipe BOTH the path-index AND the session-scope. The
///   `full_valkey_wipe` helper does this AND clears the checkpoint
///   blob (which is on disk, NOT in Valkey, so survives), then we
///   restore the checkpoint by referencing the LocalDisk blob the
///   adapter already holds via the harness's `store` handle.
/// - Actually the cleanest setup is: `full_valkey_wipe` removes ALL
///   live state including the path-index; the LocalDisk checkpoint
///   adapter survives (LocalDisk tempdir is OUTSIDE the Valkey
///   keyspace); recovery's Step A is what would re-materialize the
///   path-index — and that's the step we inject failure into. So the
///   post-call invariant `EXISTS le:path:{hex(base_path)} == 0` is
///   directly observable.
/// - The Step A fault is queued BEFORE the RPC dispatch so the applier
///   pops it on entry to Step A and short-circuits before any SADD
///   touches the path-index. Recovery propagates the `Transient` error
///   through `SessionRecoveryOutcome::Error` and the cache funnel
///   surfaces it as `ServiceError::Unavailable` → gRPC `Unavailable`.
/// - The four storage assertions cover the four observable surfaces:
///   path-index (legacy `path_index` set), path-names-index, session-
///   meta, layers-index. The Subtask body explicitly names path-index
///   + path-names + session-meta; we add layers-index for completeness
///   because a partial Step B run could leak there too (Step B is
///   never reached here, but defense-in-depth).
/// - The metric assertion `error += 1` pins the recovery hook's
///   error-arm tick. `restored` / `not_recoverable` / `already_present`
///   stay at their pre-call snapshots — the apply pipeline ran but
///   never reached the success arms.
// `clippy::await_holding_lock` — intentional: the cross-file
// `std::sync::Mutex` guard exists precisely to serialize this scenario
// against its file-local siblings AND against the Step B scenarios in
// `e2e_session_recovery_atomicity_test.rs` (they all consume the same
// process-global fault-injector queue map). No other async task contends
// for this guard within a single test invocation, so the standard
// "don't hold a non-async lock across await" rule doesn't apply.
#[allow(clippy::await_holding_lock)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn step_a_path_index_failure_leaves_no_partial_state() {
    let _serial = lock_fault_injector_serial();
    clear_apply_step_fault_injectors_for_test();

    let harness = setup_checkpoint_harness(1).await;

    let session_id = create_session_layer_and_wipe(
        &harness,
        SCENARIO_1_BASE_PATH,
        SCENARIO_1_SESSION_NAME,
        SCENARIO_1_LAYER_NAME,
    )
    .await;

    // Wipe the path-index too — `wipe_session_state` preserves it (by
    // SR-2.7 design), but for this scenario we need a clean slate so
    // the assertion "Step A failure left zero path-index members" is
    // unambiguous. The LocalDisk-backed session-checkpoint blob stays
    // intact because LocalDisk is outside the Valkey keyspace.
    let path_index_key = harness.tv.key_schema().path_index(SCENARIO_1_BASE_PATH);
    let path_names_key = harness
        .tv
        .key_schema()
        .path_names_index(SCENARIO_1_BASE_PATH);
    let _: () = harness
        .tv
        .client()
        .pool
        .del::<(), _>((path_index_key.clone(), path_names_key.clone()))
        .await
        .expect("DEL path-index keys must succeed");

    // Pre-call sanity: both path-index keys are absent. A surviving
    // key here means the cleanup DEL did not actually fire and the
    // post-call assertion would be meaningless.
    let pre_call_path_index_exists: bool = harness
        .tv
        .client()
        .pool
        .exists(&path_index_key)
        .await
        .expect("EXISTS path_index probe must succeed");
    assert!(
        !pre_call_path_index_exists,
        "pre-call sanity: path_index key MUST be absent before the Step A injection — \
         a surviving key invalidates the post-call assertion"
    );

    // Capture pre-call recovery-metric snapshot so the assertion phase
    // tracks only this scenario's recovery activity.
    let restored_pre = session_recovery_metric(&harness, "restored");
    let not_recoverable_pre = session_recovery_metric(&harness, "not_recoverable");
    let error_pre = session_recovery_metric(&harness, "error");
    let already_present_pre = session_recovery_metric(&harness, "already_present");

    // Queue a single Step A fault. The applier pops it on entry and
    // returns `Err(Transient)` without touching any path-index key.
    let mut queue = VecDeque::new();
    queue.push_back(make_step_a_fault("scenario_1_short_circuit"));
    set_apply_step_fault_injector_for_test(ApplyStep::StepA, queue);

    // Dispatch a session-resolving RPC. CreateLayer goes through the
    // cache funnel's `get_base_path`, which sees NotFound (post-wipe),
    // fires `try_recover_session`, the applier pops the Step A fault
    // and returns the transient error, the recovery hook surfaces
    // `SessionRecoveryOutcome::Error(Transient)`, the cache funnel
    // maps it to `ServiceError::Unavailable`, and tonic returns
    // `Code::Unavailable`.
    let ompf_bytes = delta_to_ompf_bytes(&empty_delta(), &[]);
    let stream = streaming_create(vec![
        make_create_header(&session_id, "post-recovery-layer"),
        make_create_data(ompf_bytes),
    ]);
    let result = handle_create_layer(&harness.svc, make_streaming_token_request(stream)).await;
    let err = result.expect_err(
        "CreateLayer MUST fail when Step A fault is injected — recovery returns Error, \
         cache funnel propagates Unavailable. A Ok(...) here means the fault injector \
         did not fire (test-checkpoint-fault-injection feature gate misconfigured) or \
         the cache funnel silently swallowed the recovery error.",
    );
    assert_eq!(
        err.code(),
        Code::Unavailable,
        "Step A Transient fault MUST map to gRPC Unavailable (transient classification \
         per `session_cache.rs`'s match arm); got code={:?}, message={}",
        err.code(),
        err.message()
    );
    assert_upstream_unavailable(&err);

    // (a) Path-index legacy set: EXISTS le:path:{hex(base_path)} == 0.
    // The Subtask body pins this assertion explicitly. A surviving key
    // would indicate Step A's SADD landed before the fault injector
    // short-circuited — a regression that would re-open the partial-
    // failure mode AC #4 is designed to prevent.
    let post_call_path_index_exists: bool = harness
        .tv
        .client()
        .pool
        .exists(&path_index_key)
        .await
        .expect("EXISTS path_index post-call probe must succeed");
    assert!(
        !post_call_path_index_exists,
        "post-call EXISTS le:path:{{hex(base_path)}} MUST be 0 — Step A short-circuited \
         BEFORE any SADD landed (the applier pops the fault BEFORE issuing the MULTI). \
         A surviving key indicates Step A executed the SADD before consulting the \
         injector — Task 5 ordering regression."
    );

    // (b) Path-names-index: SCARD le:path:{hex(base_path)}:names == 0.
    let path_names_cardinality: u64 = harness
        .tv
        .client()
        .pool
        .scard(&path_names_key)
        .await
        .expect("SCARD path_names_index post-call probe must succeed");
    assert_eq!(
        path_names_cardinality, 0,
        "post-call SCARD le:path:{{hex(base_path)}}:names MUST be 0 — Step A's MULTI \
         block batched both legacy + composite SADDs together; the fault short- \
         circuited before EXEC so the entire transaction is a no-op. Got cardinality {}",
        path_names_cardinality
    );

    // (c) Session-meta: EXISTS le:{sid}:sess == 0 — Step B never ran.
    let session_meta_key = harness.tv.key_schema().session_meta(&session_id);
    let session_meta_exists: bool = harness
        .tv
        .client()
        .pool
        .exists(&session_meta_key)
        .await
        .expect("EXISTS session_meta post-call probe must succeed");
    assert!(
        !session_meta_exists,
        "post-call EXISTS le:{{sid}}:sess MUST be 0 — Step A failed FIRST, so Step B \
         (which writes session_meta) was never reached. A surviving session_meta hash \
         indicates the applier ran Step B regardless of Step A's outcome — a step- \
         ordering regression that would break SR-2.7's asymmetric self-healing."
    );

    // (d) Layers-index: defense-in-depth assertion that Step B's SADDs
    // also did not partially leak. `layers_index` is in the {sid} slot
    // alongside session_meta in Step B's MULTI block, so this is the
    // same MULTI as session-meta — but pinning both surfaces would
    // catch a regression that split Step B internally.
    let layers_index_key = harness.tv.key_schema().layers_index(&session_id);
    let layers_index_exists: bool = harness
        .tv
        .client()
        .pool
        .exists(&layers_index_key)
        .await
        .expect("EXISTS layers_index post-call probe must succeed");
    assert!(
        !layers_index_exists,
        "post-call EXISTS le:{{sid}}:layers MUST be 0 — Step B never ran. \
         A surviving key indicates Step B's SADD landed despite Step A's error \
         — Task 5 step-ordering regression."
    );

    // (e) Recovery-metric error += 1. The Step A fault propagates
    // through `SessionRecoveryOutcome::Error(Transient)` which ticks
    // the `error` cell. Other outcome cells are unchanged.
    assert_eq!(
        session_recovery_metric(&harness, "error") - error_pre,
        1,
        "exactly ONE session-recovery attempt MUST have reported `error` — the Step A \
         fault injection. Got delta={}; pre-call snapshot={error_pre}",
        session_recovery_metric(&harness, "error") - error_pre,
    );
    assert_eq!(
        session_recovery_metric(&harness, "restored"),
        restored_pre,
        "post-call `restored` MUST equal pre-call snapshot — Step A failure means the \
         apply pipeline never reached the success arm"
    );
    assert_eq!(
        session_recovery_metric(&harness, "not_recoverable"),
        not_recoverable_pre,
        "post-call `not_recoverable` MUST equal pre-call snapshot — the checkpoint blob \
         was present so the apply branch ran"
    );
    assert_eq!(
        session_recovery_metric(&harness, "already_present"),
        already_present_pre,
        "post-call `already_present` MUST equal pre-call snapshot — the wipe drove the \
         pre-acquire EXISTS check to false, so the short-circuit cell stays untouched"
    );

    clear_apply_step_fault_injectors_for_test();
    harness.cancel.cancel();
    let _ = harness.loop_handle.await;
}

// =============================================================================
// Scenario 2 — Step A atomicity under partial-member failure.
// =============================================================================

/// Pin Epic AC #4's atomicity contract: Task 5 batches ALL path-index
/// SADDs (legacy `path_index` + every composite `path_names_index`
/// member) into ONE MULTI/EXEC block. A fault BEFORE EXEC therefore
/// leaves BOTH sets empty — no "some members landed, others didn't"
/// partial state.
///
/// Why each step is load-bearing:
///
/// - Same baseline shape as scenario 1: create a real session through
///   the gRPC entry points so the checkpoint blob carries the full
///   `(legacy_member, composite_member)` pair that Step A will replay.
///   A synthetic in-memory blob would not exercise the real
///   composition path.
/// - The Step A fault is queued BEFORE the RPC. The applier's
///   `pop_apply_step_fault(StepA)` consults the queue BEFORE entering
///   the MULTI block — so the entire Step A transaction (BOTH the
///   legacy SADD and every composite SADD) is short-circuited atomically.
/// - The scenario distinguishes itself from scenario 1 by asserting
///   BOTH path-index sets are empty AFTER the failure (scenario 1
///   asserts EXISTS=0 on each individually; this assertion is
///   conceptually a paired-empty invariant). The two scenarios appear
///   redundant on the surface — they're not. Scenario 1 pins
///   "side-effect-free short-circuit"; scenario 2 pins "no partial
///   member visibility". A regression that pre-DEL'd both sets before
///   MULTI (idempotency hardening) would still pass scenario 1 but
///   fail scenario 2 — observable subtle bug.
/// - Same `Unavailable` gRPC status assertion as scenario 1.
/// - Same metric assertion (`error += 1`).
// See `step_a_path_index_failure_leaves_no_partial_state` for the
// rationale on `clippy::await_holding_lock`.
#[allow(clippy::await_holding_lock)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn step_a_atomicity_under_partial_member_failure() {
    let _serial = lock_fault_injector_serial();
    clear_apply_step_fault_injectors_for_test();

    let harness = setup_checkpoint_harness(1).await;

    let session_id = create_session_layer_and_wipe(
        &harness,
        SCENARIO_2_BASE_PATH,
        SCENARIO_2_SESSION_NAME,
        SCENARIO_2_LAYER_NAME,
    )
    .await;

    // Wipe the path-index too. See scenario 1's rationale.
    let path_index_key = harness.tv.key_schema().path_index(SCENARIO_2_BASE_PATH);
    let path_names_key = harness
        .tv
        .key_schema()
        .path_names_index(SCENARIO_2_BASE_PATH);
    let _: () = harness
        .tv
        .client()
        .pool
        .del::<(), _>((path_index_key.clone(), path_names_key.clone()))
        .await
        .expect("DEL path-index keys must succeed");

    // Pre-call sanity: both sets are absent.
    let pre_call_legacy: u64 = harness
        .tv
        .client()
        .pool
        .scard(&path_index_key)
        .await
        .expect("SCARD legacy path_index pre-call must succeed");
    let pre_call_names: u64 = harness
        .tv
        .client()
        .pool
        .scard(&path_names_key)
        .await
        .expect("SCARD path_names_index pre-call must succeed");
    assert_eq!(
        pre_call_legacy, 0,
        "pre-call sanity: legacy path_index MUST be empty after wipe (got cardinality {})",
        pre_call_legacy
    );
    assert_eq!(
        pre_call_names, 0,
        "pre-call sanity: path_names_index MUST be empty after wipe (got cardinality {})",
        pre_call_names
    );

    let error_pre = session_recovery_metric(&harness, "error");

    let mut queue = VecDeque::new();
    queue.push_back(make_step_a_fault("scenario_2_partial_member"));
    set_apply_step_fault_injector_for_test(ApplyStep::StepA, queue);

    let ompf_bytes = delta_to_ompf_bytes(&empty_delta(), &[]);
    let stream = streaming_create(vec![
        make_create_header(&session_id, "post-recovery-layer"),
        make_create_data(ompf_bytes),
    ]);
    let result = handle_create_layer(&harness.svc, make_streaming_token_request(stream)).await;
    let err = result.expect_err(
        "CreateLayer MUST fail when Step A fault is injected; same wiring contract as \
         scenario 1",
    );
    assert_eq!(
        err.code(),
        Code::Unavailable,
        "Step A Transient fault MUST map to gRPC Unavailable; got code={:?}, message={}",
        err.code(),
        err.message()
    );
    assert_upstream_unavailable(&err);

    // Paired-empty assertion: BOTH path-index sets MUST still be empty.
    // The Subtask body pins this explicitly: "assert BOTH
    // le:path:{{hex(base_path)}} AND le:path:{{hex(base_path)}}:names
    // are empty — no partial SADD landed."
    let post_call_legacy: u64 = harness
        .tv
        .client()
        .pool
        .scard(&path_index_key)
        .await
        .expect("SCARD legacy path_index post-call must succeed");
    let post_call_names: u64 = harness
        .tv
        .client()
        .pool
        .scard(&path_names_key)
        .await
        .expect("SCARD path_names_index post-call must succeed");
    assert_eq!(
        post_call_legacy, 0,
        "post-call SCARD legacy path_index MUST be 0 — Step A's MULTI block batched \
         both SADDs together; the fault short-circuited before EXEC. A nonzero \
         cardinality means the legacy SADD landed outside the MULTI block (Task 5 \
         atomicity regression). Got {}",
        post_call_legacy
    );
    assert_eq!(
        post_call_names, 0,
        "post-call SCARD path_names_index MUST be 0 — same atomicity contract as the \
         legacy set. Got cardinality {}. An asymmetric result (legacy=0, names>0 or \
         vice versa) is the load-bearing partial-member regression this scenario \
         specifically pins.",
        post_call_names
    );

    // Metric assertion: error += 1.
    assert_eq!(
        session_recovery_metric(&harness, "error") - error_pre,
        1,
        "exactly ONE session-recovery attempt MUST have reported `error` — the Step A \
         fault injection. Got delta={}; pre-call snapshot={error_pre}",
        session_recovery_metric(&harness, "error") - error_pre,
    );

    clear_apply_step_fault_injectors_for_test();
    harness.cancel.cancel();
    let _ = harness.loop_handle.await;
}

// =============================================================================
// Scenario 3 — Step A alone is self-healing via ListSessions cleanup.
// =============================================================================

/// Pin Epic AC #4's cross-step independence + SR-2.7 self-healing
/// contract: when Step A succeeds + Step B fails on the FIRST recovery
/// attempt, the resulting Step-A-only state (path-index members present,
/// session_meta absent) is reconciled by the next `ListSessions` walk's
/// `reconcile_ghost_members`. The Subtask body asserts the
/// opportunistic-cleanup branch SREMs the ghost path-name member; the
/// load-bearing SREM arm is `SessionRecoveryOutcome::NoCheckpoint` per
/// `reconcile_ghost_members`'s match (the `Error` arm conservatively
/// skips SREM, and `Recovered + Ok(Some(meta))` appends to the response).
///
/// To drive `NoCheckpoint`, we delete the session-checkpoint blob AFTER
/// the first (Step-B-failed) recovery — modelling the realistic scenario
/// where a partial recovery left a path-index ghost AND a downstream
/// `DeleteSession` later removed the blob (the blob's lifecycle is
/// orthogonal to the path-index member's lifecycle in a multi-replica
/// world; a peer replica may have completed the recovery, then deleted
/// the session and its blob, leaving this replica's stale path-index
/// member). The `ListSessions` walk then sees the ghost, recovery
/// reports `NoCheckpoint`, and the opportunistic-cleanup branch SREMs.
///
/// ## One-composite shape — load-bearing detail
///
/// The session-checkpoint assembler emits EXACTLY ONE `path_name_member`
/// per session — the `session_name\0session_id` composite that mirrors
/// what `SessionStore::create_once` writes on the live path
/// (`src/storage/session.rs:308`). Step A of recovery re-SADDs the same
/// composite; the re-SADD is idempotent against the surviving
/// CreateSession-written member, so post-Step-A `path_names_index`
/// carries EXACTLY ONE composite. The `ListSessions` walk surfaces it
/// as ONE ghost (HGETALL session_meta is empty for the corresponding
/// sid), so `reconcile_ghost_members` invokes `try_recover_session`
/// once. That single call hits the `NoCheckpoint` arm (the blob was
/// deleted between RPCs), so `not_recoverable` ticks by EXACTLY 1 and
/// the composite is SREM'd.
///
/// Why each step is load-bearing:
///
/// - The Step B fault is queued first; the first RPC (CreateLayer)
///   triggers cache-funnel recovery → Step A succeeds (path-index
///   re-materialised from the LocalDisk blob; the recovery applier's
///   re-SADD of the `session_name\0session_id` composite is idempotent
///   against the surviving CreateSession-written member) → Step B's
///   queued fault short-circuits → recovery returns `Error(Transient)`
///   → the cache funnel surfaces `Unavailable` to the gRPC caller. The
///   resulting on-disk state is: path-index members present (one
///   composite), session_meta absent. This is exactly the Step-A-only
///   state SR-2.7 is designed to self-heal.
/// - We assert the post-first-RPC state directly: legacy path_index
///   has the session_id member, path_names_index has the one
///   session-name composite, session_meta is still absent. A regression
///   that made Step A non-atomic would surface here (asymmetric set
///   state).
/// - Between the two RPCs we delete the LocalDisk-backed checkpoint
///   blob — this is what flips the `try_recover_session` outcome from
///   `Recovered` (the second ListSessions walk would otherwise restore
///   the session) to `NoCheckpoint` (SREM arm). Modelling the
///   peer-replica-deleted-the-blob race; in production a path-index
///   ghost can outlive its checkpoint blob if a peer replica completes
///   the lifecycle (DeleteSession → blob delete) before this replica
///   notices the ghost.
/// - `ListSessions` walks the path-index, finds the single ghost,
///   `reconcile_ghost_members` invokes `try_recover_session` once, it
///   returns `NoCheckpoint` (the blob is gone), and `srem_ghost_member`
///   SREMs from BOTH `path_names_index` AND legacy `path_index`. We
///   assert post-state cardinality == 0 on both sets.
/// - The post-ListSessions assertions pin SR-2.7's self-healing:
///   neither set carries the stale members anymore; the response is
///   empty (no recovered session); the `not_recoverable` cell ticks
///   by exactly 1 (one composite ghost).
/// - Note: `ListSessions` bypasses `SessionBasePathCache` entirely
///   (the handler reads the path-index walk directly), so no cache
///   invalidation is needed between the two RPCs.
// See `step_a_path_index_failure_leaves_no_partial_state` for the
// rationale on `clippy::await_holding_lock`.
#[allow(clippy::await_holding_lock)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn step_a_alone_is_self_healing_via_list_sessions_cleanup() {
    let _serial = lock_fault_injector_serial();
    clear_apply_step_fault_injectors_for_test();

    let harness = setup_checkpoint_harness(1).await;

    let session_id = create_session_layer_and_wipe(
        &harness,
        SCENARIO_3_BASE_PATH,
        SCENARIO_3_SESSION_NAME,
        SCENARIO_3_LAYER_NAME,
    )
    .await;

    // Note: scenario 3 does NOT wipe the path-index keys — Step A's
    // recovery SADDs are idempotent against the surviving CreateSession-
    // written members, so the post-Step-A state is the same regardless
    // of whether the path-index was wiped beforehand. Leaving the
    // CreateSession-written members in place tests the realistic
    // "path-index survived, session-scope wiped" disaster scenario.

    let path_index_key = harness.tv.key_schema().path_index(SCENARIO_3_BASE_PATH);
    let path_names_key = harness
        .tv
        .key_schema()
        .path_names_index(SCENARIO_3_BASE_PATH);
    // CreateSession writes `session_name\0session_id` to path_names_index.
    // The session-checkpoint assembler emits the SAME composite (one per
    // session, irrespective of layer count — see
    // `src/checkpoint/session.rs`'s post-loop `add_path_name_member`
    // call). Step A of recovery re-SADDs the composite idempotently
    // against the surviving CreateSession-written member; no second
    // composite is produced.
    let session_name_composite = KeySchema::path_name_member(SCENARIO_3_SESSION_NAME, &session_id);

    // Sanity-check the precondition: post-wipe, the path-index still
    // carries the CreateSession-written members (wipe_session_state
    // preserves them per SR-2.7's asymmetric self-healing contract).
    let pre_call_legacy_member: bool = harness
        .tv
        .client()
        .pool
        .sismember(&path_index_key, &session_id)
        .await
        .expect("SISMEMBER legacy path_index pre-call must succeed");
    let pre_call_session_composite: bool = harness
        .tv
        .client()
        .pool
        .sismember(&path_names_key, session_name_composite.clone())
        .await
        .expect("SISMEMBER session-name composite pre-call must succeed");
    let pre_call_path_names_cardinality: u64 = harness
        .tv
        .client()
        .pool
        .scard(&path_names_key)
        .await
        .expect("SCARD path_names_index pre-call must succeed");
    assert!(
        pre_call_legacy_member,
        "pre-call sanity: legacy path_index MUST carry the session_id member \
         post-wipe (wipe_session_state preserves path-index by SR-2.7 design). \
         A missing member here means `wipe_session_state` regressed and the rest \
         of the scenario is unsound."
    );
    assert!(
        pre_call_session_composite,
        "pre-call sanity: path_names_index MUST carry the session-name composite \
         post-wipe — CreateSession's live-write SADD survives `wipe_session_state`"
    );
    assert_eq!(
        pre_call_path_names_cardinality, 1,
        "pre-call sanity: path_names_index MUST contain exactly the one \
         session-name composite. Any other composite indicates a pre-fix \
         per-layer composite leaked through (Bee-level review BLOCKER), or \
         a peer write polluted the index."
    );

    let restored_pre = session_recovery_metric(&harness, "restored");
    let not_recoverable_pre = session_recovery_metric(&harness, "not_recoverable");
    let error_pre = session_recovery_metric(&harness, "error");

    // Queue a single Step B fault. Step A runs normally (idempotent
    // re-SADD of the surviving path-index members); Step B short-
    // circuits before any session-scoped key is written.
    let mut queue_b = VecDeque::new();
    queue_b.push_back(make_step_b_fault("scenario_3_first_recovery_attempt"));
    set_apply_step_fault_injector_for_test(ApplyStep::StepB, queue_b);

    // First RPC: CreateLayer drives cache-funnel recovery. Step B
    // fault → SessionRecoveryOutcome::Error(Transient) → cache funnel
    // returns ServiceError::Unavailable → tonic Status::Unavailable.
    let ompf_bytes = delta_to_ompf_bytes(&empty_delta(), &[]);
    let stream = streaming_create(vec![
        make_create_header(&session_id, "post-recovery-layer"),
        make_create_data(ompf_bytes),
    ]);
    let result = handle_create_layer(&harness.svc, make_streaming_token_request(stream)).await;
    let err = result.expect_err(
        "CreateLayer MUST fail on the first attempt — Step B fault propagates \
         through Error(Transient) → Unavailable.",
    );
    assert_eq!(
        err.code(),
        Code::Unavailable,
        "Step B Transient fault MUST map to gRPC Unavailable on the first attempt; \
         got code={:?}, message={}",
        err.code(),
        err.message()
    );
    assert_upstream_unavailable(&err);

    // Confirm the on-disk state matches the SR-2.7 Step-A-only invariant:
    // path-index members present, session_meta absent.
    let session_meta_key = harness.tv.key_schema().session_meta(&session_id);
    let session_meta_after_step_b_fail: bool = harness
        .tv
        .client()
        .pool
        .exists(&session_meta_key)
        .await
        .expect("EXISTS session_meta post-first-RPC probe must succeed");
    assert!(
        !session_meta_after_step_b_fail,
        "after Step B fault, session_meta MUST be absent — the Step B MULTI block \
         contains all session-scoped writes (HSET session_meta, SET layer_mapping, \
         SADD layers_index) and short-circuits atomically. A surviving session_meta \
         indicates Step B wrote it before consulting the injector — Task 5 step- \
         ordering regression."
    );

    let legacy_after_step_b_fail: bool = harness
        .tv
        .client()
        .pool
        .sismember(&path_index_key, &session_id)
        .await
        .expect("SISMEMBER legacy path_index post-first-RPC must succeed");
    let session_composite_after_step_b_fail: bool = harness
        .tv
        .client()
        .pool
        .sismember(&path_names_key, session_name_composite.clone())
        .await
        .expect("SISMEMBER session-name composite post-first-RPC must succeed");
    let path_names_cardinality_after_step_b_fail: u64 = harness
        .tv
        .client()
        .pool
        .scard(&path_names_key)
        .await
        .expect("SCARD path_names_index post-first-RPC must succeed");
    assert!(
        legacy_after_step_b_fail,
        "after Step B fault, the legacy path_index member MUST still be present — \
         Step A succeeded BEFORE Step B failed; this is the Step-A-only ghost state \
         SR-2.7 is designed to self-heal."
    );
    assert!(
        session_composite_after_step_b_fail,
        "after Step B fault, the session-name composite member MUST still be \
         present — written by CreateSession's live-write path and idempotently \
         re-SADD'd by Step A of recovery."
    );
    assert_eq!(
        path_names_cardinality_after_step_b_fail, 1,
        "after Step B fault, path_names_index MUST contain exactly the one \
         session-name composite. The session-checkpoint assembler emits exactly \
         one composite per session (matching the live `SessionStore::create_once` \
         write at `src/storage/session.rs:308`), and Step A's re-SADD is \
         idempotent against the surviving CreateSession-written member. Any \
         second composite indicates the pre-fix per-layer composite regression."
    );

    // Drain the (now-empty) Step B queue defensively. The first
    // recovery consumed the entry; future recoveries will run Step B
    // normally. We want the SECOND walk to NOT have a fault.
    clear_apply_step_fault_injectors_for_test();

    // Delete the session-checkpoint blob. This flips the SECOND
    // recovery attempt from `Recovered` (would restore the session,
    // reconciler appends to response) to `NoCheckpoint` (SREM arm).
    // Models the realistic peer-replica race described in the test
    // docstring above.
    harness
        .store
        .delete_session_checkpoint(&session_id)
        .await
        .expect("delete_session_checkpoint must succeed against the LocalDisk adapter");

    // Second RPC: ListSessions walks the path-index, finds the ghost
    // (HGETALL session_meta is empty), invokes try_recover_session,
    // which sees no checkpoint blob and returns NoCheckpoint. The
    // reconciler's NoCheckpoint arm SREMs from both path-index sets.
    let list_response = handle_list_sessions(
        &harness.svc,
        make_unary_token_request(ListSessionsRequest {
            path: SCENARIO_3_BASE_PATH.to_owned(),
        }),
    )
    .await
    .expect(
        "ListSessions MUST succeed against the ghost-only base_path — the \
         opportunistic-cleanup is silent and recovery's NoCheckpoint arm cleanly \
         sweeps the orphan.",
    );
    let proto_sessions = list_response.into_inner().sessions;

    // The response carries NO recovered session — the blob was
    // deleted, so the ghost was SREM'd via the NoCheckpoint arm rather
    // than appended via the Recovered arm.
    assert!(
        proto_sessions.is_empty(),
        "ListSessions response MUST be empty — the ghost has no checkpoint blob, \
         so the NoCheckpoint arm SREMs and excludes. A non-empty response means \
         either the blob deletion failed (the second recovery succeeded via \
         Recovered) or the reconciler's NoCheckpoint arm appended to the response \
         (a regression that would break the SREM contract)."
    );

    // The Subtask body pins the SREM assertion explicitly: "Assert the
    // existing opportunistic-cleanup branch SREMs the ghost path-name
    // member." The single composite ghost results in one SREM call via
    // reconcile_ghost_members.
    let session_composite_after_list: bool = harness
        .tv
        .client()
        .pool
        .sismember(&path_names_key, session_name_composite.clone())
        .await
        .expect("SISMEMBER session-name composite post-ListSessions must succeed");
    assert!(
        !session_composite_after_list,
        "post-ListSessions SISMEMBER MUST be 0 for the session-name composite — \
         reconcile_ghost_members's NoCheckpoint arm calls srem_ghost_member, \
         which SREMs from path_names_index. A surviving member means the SREM \
         never fired (regression in the NoCheckpoint arm)."
    );
    let path_names_cardinality_after: u64 = harness
        .tv
        .client()
        .pool
        .scard(&path_names_key)
        .await
        .expect("SCARD path_names_index post-ListSessions must succeed");
    assert_eq!(
        path_names_cardinality_after, 0,
        "post-ListSessions path_names_index MUST be empty — the single composite \
         ghost SREM'd via the NoCheckpoint arm. A nonzero cardinality means \
         a second composite leaked from somewhere unexpected, or the SREM failed \
         silently. Got cardinality {}.",
        path_names_cardinality_after
    );
    let legacy_after_list: bool = harness
        .tv
        .client()
        .pool
        .sismember(&path_index_key, &session_id)
        .await
        .expect("SISMEMBER legacy path_index post-ListSessions probe must succeed");
    assert!(
        !legacy_after_list,
        "post-ListSessions SISMEMBER MUST be 0 for the legacy path_index member — \
         srem_ghost_member SREMs from both sets symmetrically."
    );

    // Metric assertions:
    //
    // - `error` ticked by 1 (the first recovery's Step B fault).
    // - `not_recoverable` ticked by 1 (the single composite ghost in the
    //   second recovery's ListSessions walk — see the docstring's
    //   "one-composite shape" section).
    // - `restored` stays at the pre-call snapshot (no successful
    //   apply happened).
    //
    // Use `wait_for_recovery_metric` for the tick assertion because
    // ListSessions's reconciler may complete the metric tick on a
    // separate async path. Per `docs-internal/test-writing-guide.md`'s
    // synchronization-patterns guidance and the helper's docstring,
    // bounded-poll-on-observable is the right pattern.
    assert_eq!(
        session_recovery_metric(&harness, "error") - error_pre,
        1,
        "exactly ONE session-recovery attempt MUST have reported `error` — the \
         first RPC's Step B fault. Got delta={}; pre-call snapshot={error_pre}",
        session_recovery_metric(&harness, "error") - error_pre,
    );
    wait_for_recovery_metric(
        &harness,
        "session",
        "not_recoverable",
        not_recoverable_pre + 1,
        10,
    )
    .await;
    assert_eq!(
        session_recovery_metric(&harness, "restored"),
        restored_pre,
        "post-call `restored` MUST equal pre-call snapshot — both recovery attempts \
         ended on error/not_recoverable arms, never on the success arm"
    );

    // Final guard: ensure no fault-injector queues survive (so a
    // parallel-running test in the same process cannot consume one of
    // ours by mistake).
    clear_apply_step_fault_injectors_for_test();

    harness.cancel.cancel();
    let _ = harness.loop_handle.await;
}

// ── Cross-test bleed safeguard ───────────────────────────────────────────────
//
// All three scenarios call `clear_apply_step_fault_injectors_for_test()`
// at the top and the bottom of their bodies. The seam's queues are
// process-global state per its docstring contract — explicit drain at
// both ends gives a clean handoff regardless of test execution order or
// parallel scheduling within the same `cargo test` invocation.
