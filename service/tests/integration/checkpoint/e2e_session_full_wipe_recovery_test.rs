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

//! End-to-end test for Epic AC #2 — full Valkey wipe recovery
//! (Subtask `t3.fgj.7v.o7.c6` under Task `t2.fgj.7v.o7`, Epic `t1.fgj.7v`,
//! Bee `b.fgj`).
//!
//! Pins the headline composition contract for the Epic: after a write →
//! checkpoint → **full namespace wipe** → next-UpdateLayer cycle, the
//! handler MUST land BOTH recovery hops in series:
//!
//! 1. **Session recovery first** — `SessionBasePathCache::get_base_path`
//!    funnels the absent `le:{sid}:sess` lookup into
//!    `SessionRecoveryHook`. The two-step apply re-materializes
//!    `session_meta`, the layer-mappings table, the layers-index, AND
//!    the path-index (`le:path:{hex(base_path)}` + `:names`) — the last
//!    of which is gone because `full_valkey_wipe` wiped EVERY `le:*`
//!    key, including the path-index that the per-handler tests in
//!    `e2e_session_recovery_per_handler_test.rs` deliberately preserve
//!    (SR-2.7 asymmetric self-healing — the path-index survival is the
//!    contract for THAT failure mode, not this one).
//! 2. **Layer recovery second** — once `update_layer_once` resolves the
//!    layer mapping (now restored by step 1) and HGETALLs `obj:hdr`, the
//!    empty header surfaces `NotFound` to the handler, which trips the
//!    layer-recovery branch and runs `apply_checkpoint` against the
//!    object's checkpoint generation. The re-issued commit then lands
//!    against the restored materialized state.
//!
//! Two recovery hops → exactly two metric increments. The session-scope
//! and layer-scope cells on
//! `live_edit_checkpoint_recovery_attempts_total{result="restored", scope, service}`
//! MUST each tick by exactly 1:
//!
//! - `restored{scope="session"} +1` → session recovery fired AND the
//!   apply pipeline rebuilt session-state.
//! - `restored{scope="layer"} +1` → layer recovery fired AND the apply
//!   pipeline rebuilt object-state.
//! - `0` on either side ⇒ broken wiring (the cache funnel or the
//!   per-handler recovery branch short-circuited).
//! - `>1` on either side ⇒ broken `recovered_once` latch (the handler
//!   fell through to a second recovery attempt rather than returning
//!   from the first).
//!
//! The post-`UpdateLayer` assertions then pin the END-STATE: a successful
//! `ListSessions(base_path)` (proves the path-index restore landed end-
//! to-end, since `ListSessions` reads `path_names_index` and surfaces the
//! restored session) AND a successful `read_object_state_for_test` (proves the
//! object's materialized state was rebuilt, including the new SetField
//! from the post-recovery commit).
//!
//! Sandbox env note: this test uses `TestValkey::setup`, which reads
//! `TEST_VALKEY_URL`. Under the project sandbox the URL must include the
//! password and `host.docker.internal` host; see CLAUDE.md.

use fred::interfaces::{HashesInterface, KeysInterface};

use live_edit::grpc::layer::{handle_create_layer, handle_update_layer};
use live_edit::grpc::proto::{CreateSessionRequest, ListSessionsRequest};
use live_edit::grpc::session::{handle_create_session, handle_list_sessions};
use live_edit::storage::test_support::read_object_state_for_test;
use oodf_ffi::{
    CommandDiffChangeSection, OwnedDelta, SetField, StructCommand, TimeSample as OodfTimeSample,
};

use super::helpers::*;
use crate::common::owned_delta::delta_to_ompf_bytes;

const SESSION_NAME: &str = "e2e-full-wipe-recovery-session";
const LAYER_NAME: &str = "e2e-full-wipe-recovery-layer";
const CLIENT_ID: &str = "e2e-full-wipe-recovery-client";
const SESSION_BASE_PATH: &str = "/stage/full-wipe";

// Section + field constants for the initial / post-recovery delta payloads.
// `SECTION_ID` is the shared parent for the initial seed section + every
// SetField the test issues. `POST_RECOVERY_FIELD_KEY` is the marker the
// final `read_object_state_for_test` checks for — its presence proves the
// commit re-issue inside the layer-recovery branch actually landed
// (catching a regression that only restored the checkpoint and skipped
// the re-issue).
const SECTION_ID: u64 = 73;
const POST_RECOVERY_FIELD_KEY: &str = "post-full-wipe-recovery-field";

// ── Streaming framing helpers ────────────────────────────────────────────────
//
// Imported from `tests/integration/checkpoint/helpers.rs` via
// `use super::helpers::*;` (b.dzd Phase 1 deduplication).

// ── Delta fixture builders ───────────────────────────────────────────────────
//
// `populated_initial_delta` seeds the layer with one section + one inline
// field so the post-recovery `read_object_state_for_test` has restorable
// sections / fields to observe. Without populated content, asserting
// "post-recovery state contains restored sections/fields" would be
// vacuous (the snapshot would be empty either way).
//
// `delta_with_field_update` and `delta_with_time_sample` mirror the
// patterns in `e2e_update_layer_recovery_test.rs` so the burst phase
// exercises both Lua-fast-path branches in `update_layer`.

fn populated_initial_delta() -> OwnedDelta {
    OwnedDelta {
        is_diff: true,
        base_version: 0,
        source_format: "usd".to_string(),
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 0,
            section_id: SECTION_ID,
            section_name: "e2e-full-wipe-section".to_string(),
            section_type: 1,
            section_order: 0,
        })],
        set_fields: vec![SetField {
            section_id: SECTION_ID,
            key_name: "initial-field".to_string(),
            value_or_ext_hash: vec![0xAB, 0xCD, 0xEF],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 0,
        }],
        ..Default::default()
    }
}

fn delta_with_field_update(set_order: u64, marker: u8) -> OwnedDelta {
    // `base_version: 1` (any non-zero) marks this as an INCREMENTAL diff,
    // not a reset. With `base_version == 0` plus `is_diff == true` the
    // domain `apply_delta` engine takes the reset-delta branch (SDD
    // §9.1.3) which wipes existing sections / fields / time samples
    // before applying — effectively erasing the seed delta from
    // CreateLayer. See `e2e_update_layer_recovery_test.rs` for the full
    // rationale (same comment block).
    OwnedDelta {
        is_diff: true,
        base_version: 1,
        set_fields: vec![SetField {
            section_id: SECTION_ID,
            key_name: format!("burst-field-{set_order}"),
            value_or_ext_hash: vec![marker, marker.wrapping_add(1), marker.wrapping_add(2)],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order,
        }],
        ..Default::default()
    }
}

fn delta_with_time_sample(time_value: f64, marker: u8) -> OwnedDelta {
    OwnedDelta {
        is_diff: true,
        base_version: 1,
        time_samples: vec![OodfTimeSample {
            section_id: SECTION_ID,
            time: time_value,
            value_or_ext_hash: vec![marker],
            ext_value_size: 0,
            ext_value_index: 0,
        }],
        ..Default::default()
    }
}

/// The post-wipe UpdateLayer carries a SetField on a NEW key
/// (`POST_RECOVERY_FIELD_KEY`) so a post-recovery `read_object_state_for_test`
/// can independently verify the new write landed on top of the restored
/// state. A regression that lost the new write (e.g. a recovery loop
/// that re-ran apply without re-issuing the commit) would show all
/// original fields restored but the marker field missing.
fn delta_post_recovery() -> OwnedDelta {
    OwnedDelta {
        is_diff: true,
        base_version: 1,
        set_fields: vec![SetField {
            section_id: SECTION_ID,
            key_name: POST_RECOVERY_FIELD_KEY.to_string(),
            value_or_ext_hash: vec![0xCA, 0xFE, 0xBA, 0xBE],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 99,
        }],
        ..Default::default()
    }
}

// ── Test driver helpers ──────────────────────────────────────────────────────

/// Issue a single `CreateLayer` carrying the populated initial delta.
/// Returns the freshly-minted `object_id`.
async fn create_layer_with_initial_state(harness: &CheckpointHarness, session_id: &str) -> u64 {
    let ompf = delta_to_ompf_bytes(&populated_initial_delta(), &[]);
    let stream = streaming_create(vec![
        make_create_header(session_id, LAYER_NAME),
        make_create_data(ompf),
    ]);
    let response = handle_create_layer(&harness.svc, make_streaming_token_request(stream))
        .await
        .expect("CreateLayer must succeed for the full-wipe E2E baseline");
    response.into_inner().object_id
}

/// Issue a single `UpdateLayer` and return the produced `sequence_num`.
async fn issue_update(
    harness: &CheckpointHarness,
    session_id: &str,
    object_id: u64,
    delta: OwnedDelta,
    client_timestamp: u64,
) -> u64 {
    let ompf = delta_to_ompf_bytes(&delta, &[]);
    let stream = streaming_update(vec![
        make_update_header(
            session_id,
            LAYER_NAME,
            object_id,
            client_timestamp,
            CLIENT_ID,
        ),
        make_update_data(ompf),
    ]);
    let response = handle_update_layer(&harness.svc, make_streaming_token_request(stream))
        .await
        .expect(
            "UpdateLayer must succeed in the pre-wipe burst / post-recovery happy path \
             — a failure here means the recovery chain (session-recovery → layer-recovery) \
             did not compose end-to-end",
        );
    response.into_inner().sequence_num
}

/// Snapshot of the (session, layer) recovery-metric cells at a given
/// moment. Captured once before the wipe and re-read after each
/// recovery hop so the assertion phase can pin
/// `delta(restored{session}) == 1` AND `delta(restored{layer}) == 1`.
struct RecoveryBaseline {
    session_restored: u64,
    session_not_recoverable: u64,
    session_error: u64,
    session_already_present: u64,
    layer_restored: u64,
    layer_not_recoverable: u64,
    layer_error: u64,
}

fn capture_recovery_baseline(harness: &CheckpointHarness) -> RecoveryBaseline {
    RecoveryBaseline {
        session_restored: session_recovery_metric(harness, "restored"),
        session_not_recoverable: session_recovery_metric(harness, "not_recoverable"),
        session_error: session_recovery_metric(harness, "error"),
        session_already_present: session_recovery_metric(harness, "already_present"),
        layer_restored: layer_recovery_metric(harness, "restored"),
        layer_not_recoverable: layer_recovery_metric(harness, "not_recoverable"),
        layer_error: layer_recovery_metric(harness, "error"),
    }
}

/// `live_edit_checkpoint_recovery_attempts_total{result, scope="layer",
/// service=SERVICE_LABEL}` cell reader. `scope="layer"` (NOT `"session"`)
/// is the load-bearing difference vs. [`session_recovery_metric`] in
/// helpers.rs — this test is the umbrella E2E that asserts BOTH cells
/// tick, so we need both readers.
fn layer_recovery_metric(harness: &CheckpointHarness, result: &str) -> u64 {
    live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(&harness.metrics),
        "live_edit_checkpoint_recovery_attempts_total",
        &[
            ("result", result),
            ("scope", "layer"),
            ("service", SERVICE_LABEL),
        ],
    )
    .unwrap_or(0.0) as u64
}

/// Drop the cached `base_path` entry for `session_id` on the harness's
/// `SessionBasePathCache`. Load-bearing for the same reason as the
/// per-handler tests: the pre-wipe CreateLayer / UpdateLayer calls
/// populate the cache with the live `base_path`; the cache's positive
/// TTL is long enough that the post-wipe UpdateLayer inside the same
/// test would HIT the cache and never consult `session_store.get_base_path`,
/// bypassing session-recovery entirely. Without this explicit
/// invalidation the test would silently observe `restored{session} == 0`
/// (recovery never fired) AND a successful first-leg auth path (cache
/// HIT returned the pre-wipe `base_path`), masking the wiring regression
/// we're trying to pin.
///
/// In production the `session:deleted` pub/sub fanout drives the cross-
/// replica cache invalidation; under E2E test we have no peer replica
/// writing the wipe, so the cache must be invalidated here.
fn invalidate_session_cache(harness: &CheckpointHarness, session_id: &str) {
    harness.svc.session_base_path_cache.invalidate(session_id);
}

// =============================================================================
// Headline E2E test — Epic AC #2 (full Valkey wipe → composed recovery).
// =============================================================================

/// Pin Epic AC #2: a full Valkey wipe followed by an `UpdateLayer`
/// against the original `(session_id, layer_name)` MUST compose
/// session-recovery (first hop) with layer-recovery (second hop) and
/// land Ok with a `sequence_num > pre_wipe_max_seq`.
///
/// Why this test is load-bearing for the whole Epic:
///
/// - Tasks 1–6 each have their own per-Task tests (atomicity of the
///   two-step apply, path-index restore correctness, ghost-member
///   reconciliation, etc.). This test is the COMPOSITION assertion:
///   the two-hop chain WORKS, end-to-end, through the real gRPC entry
///   point, against a live Valkey, with the live checkpoint store. A
///   regression in any one of the six earlier Tasks would surface here
///   as either a failed `UpdateLayer` Ok, a mismatched metric, or a
///   missing entry in the post-recovery `ListSessions` / snapshot
///   assertions.
/// - The metric assertion (`restored{session} +1` AND `restored{layer}
///   +1`) is the ONE place where both recovery scopes are pinned in the
///   same test. The per-Task tests assert each scope's increment in
///   isolation; only here do we verify they BOTH tick on the same
///   handler dispatch — proving the composition contract (session first,
///   then layer) holds.
///
/// Step-by-step rationale:
///
/// 1. `CreateSession` through the real `handle_create_session` RPC
///    entry point. The lifecycle hook fires fire-and-forget session-
///    checkpoint write through the dispatcher. `wait_for_session_checkpoint`
///    blocks until the blob lands.
/// 2. `CreateLayer` with a POPULATED initial delta — one
///    `DiffChangeSection` + one `SetField`. The non-empty initial state
///    is what the post-recovery `read_object_state_for_test` assertion will
///    verify was restored; an empty delta would make the "snapshot has
///    sections" check vacuous.
/// 3. Burst of UpdateLayers (3 commits, mixing field + time-sample
///    payloads) drives `sequence_num` well above the post-CreateLayer
///    seed so the "strictly greater than" assertion below has
///    unambiguous signal. Mixed payloads exercise both Lua-fast-path
///    branches in `update_layer`, matching the realistic shape commits
///    take in production.
/// 4. `wait_for_checkpoint` against the latest `object_id` blocks until
///    the leader-elected loop has landed at least one layer-checkpoint
///    generation. With the harness's `tick_secs=1` and
///    `interval=quiesce=0`, this converges within a single tick.
/// 5. `full_valkey_wipe` SCANs + UNLINKs EVERY `tv.prefix()*` key —
///    `le:{sid}:sess`, the layer-mapping, `le:{oid}:obj:*`, the path-
///    index, the live-update stream, GC bookkeeping, the external-value
///    store, counters. The post-wipe Valkey state is empty under the
///    test prefix. Sanity assertions on three specific keys (sess,
///    mapping, obj:hdr) catch test-infrastructure regressions before
///    the assertion phase obscures them.
/// 6. `invalidate_session_cache` drops the in-process
///    `SessionBasePathCache` entry — see the helper docstring for why.
/// 7. `handle_update_layer` MUST succeed. Two hops in series:
///    a. The cache funnel's `get_base_path` sees NotFound (since
///       `le:{sid}:sess` is gone), fires session-recovery, the apply
///       re-materializes session_meta + layer-mappings + layers-index +
///       path-index.
///    b. `update_layer_once` resolves the (now-restored) mapping,
///       HGETALLs the empty `obj:hdr`, surfaces NotFound, the layer-
///       recovery branch fires, `apply_checkpoint` rebuilds object
///       state, the commit is re-issued. Returns Ok with
///       `sequence_num > pre_wipe_max_seq`.
/// 8. Both `restored{session}` AND `restored{layer}` MUST have ticked
///    by exactly one. 0 on either side ⇒ broken wiring; >1 ⇒ broken
///    `recovered_once` latch.
/// 9. `handle_list_sessions(base_path)` MUST return exactly one session
///    matching the original. Proves the path-index restore landed end-
///    to-end (ListSessions reads the path-index — without the restore,
///    it would return an empty list).
/// 10. `read_object_state_for_test(object_id)` MUST return restored
///     materialized state — same object_id, the original section
///     present, AND the post-recovery SetField present. The last item
///     proves the commit re-issue inside the layer-recovery branch
///     actually landed (a regression that only restored the checkpoint
///     and skipped the re-issue would show the original section but
///     no marker field).
/// 11. Graceful shutdown of the checkpoint loop — same drain pattern as
///     the sibling tests.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn full_valkey_wipe_recovers_session_and_layer_then_update_succeeds() {
    // Step 1: Harness with a fast-converging checkpoint loop.
    let harness = setup_checkpoint_harness(1).await;

    // Step 2: CreateSession through the real RPC entry point. The
    // session-checkpoint writer is wired on the harness's SessionStore
    // (helpers.rs), so the lifecycle hook in `SessionStore::create`'s
    // success arm enqueues a fire-and-forget put through the dispatcher.
    let create_session_resp = handle_create_session(
        &harness.svc,
        make_unary_token_request(CreateSessionRequest {
            path: SESSION_BASE_PATH.to_owned(),
            name: SESSION_NAME.to_owned(),
        }),
    )
    .await
    .expect("CreateSession must succeed against an AllowAllPermissions harness");
    let session_id = create_session_resp.into_inner().session_id;
    assert!(
        !session_id.is_empty(),
        "CreateSession must mint a non-empty session_id"
    );

    // Wait for the session-checkpoint dispatcher's spawned put to land
    // on disk. Without this, a subsequent wipe-and-recover would race
    // the writer task and observe `NoCheckpoint` from the recovery hook
    // — the test would fail at the metric assertion rather than the
    // wiring check it's targeting.
    wait_for_session_checkpoint(&harness.store, &session_id, 10).await;

    // Capture the pre-CreateLayer session-checkpoint blob bytes so the
    // subsequent CreateLayer-driven write can be observed to mutate
    // them. The subtask body asks for "BOTH (a) new session checkpoint
    // (bytes-differ from pre) and (b) layer checkpoint" — this is the
    // pre side of the (a) assertion.
    let pre_create_layer_session_bytes = harness
        .store
        .get_session_checkpoint(&session_id)
        .await
        .expect("get_session_checkpoint must succeed once a blob has landed")
        .expect("session-checkpoint blob must be present after wait_for_session_checkpoint")
        .bytes;

    // Step 3: CreateLayer with a populated initial state. CreateLayer's
    // success arm ALSO enqueues a session-checkpoint write (the layer-
    // mappings table changed), so we wait for the NEXT session-checkpoint
    // blob to land — distinguished by bytes-differ against the pre-call
    // snapshot above. Without this discipline, the wipe-and-recover
    // step further down would race CreateLayer's writer task and the
    // restored session might be missing the freshly-created layer
    // mapping.
    let object_id = create_layer_with_initial_state(&harness, &session_id).await;
    assert!(
        object_id > 0,
        "CreateLayer must mint a non-zero object_id; got {object_id}"
    );

    wait_for_post_create_layer_session_checkpoint(
        &harness,
        &session_id,
        pre_create_layer_session_bytes.as_ref(),
        10,
    )
    .await;

    // Wait for the first layer-checkpoint generation to land. The
    // populated CreateLayer above mints the object's seed state; the
    // loop's first tick after this snapshots it.
    let _seed_generation = wait_for_checkpoint(&harness.store, object_id, 10).await;

    // Step 4: Burst of UpdateLayers — three commits mixing field +
    // time-sample payloads. Each carries `base_version: 1` so the
    // domain engine takes the INCREMENTAL diff branch (NOT a reset),
    // preserving the seed section / field across the burst.
    let mut pre_wipe_max_seq: u64 = 0;
    let updates = [
        (delta_with_field_update(1, 0x21), 2_001_u64),
        (delta_with_time_sample(0.0, 0x22), 2_002_u64),
        (delta_with_field_update(3, 0x23), 2_003_u64),
    ];
    for (delta, ts) in updates {
        let seq = issue_update(&harness, &session_id, object_id, delta, ts).await;
        assert!(
            seq > pre_wipe_max_seq,
            "UpdateLayer sequence_num must advance monotonically; \
             previous max={pre_wipe_max_seq}, got {seq}"
        );
        pre_wipe_max_seq = seq;
    }
    assert!(
        pre_wipe_max_seq >= 4,
        "pre-wipe max sequence_num must be at least 4 (1 from CreateLayer + 3 \
         UpdateLayers); got {pre_wipe_max_seq}"
    );

    // Wait for a POST-BURST layer-checkpoint generation. The seed
    // generation captured above is from the immediate post-CreateLayer
    // tick; we need at least one generation that reflects the burst's
    // committed updates so the post-wipe recovery has a generation
    // with `snap_seq >= pre_wipe_max_seq` to apply. Capturing here
    // (rather than re-fetching at use site) keeps the harness's busy-
    // wait surface minimal.
    wait_for_post_burst_layer_checkpoint(&harness, object_id, pre_wipe_max_seq, 10).await;

    // Step 5: Full-namespace wipe. SCANs + UNLINKs every `tv.prefix()*`
    // key — session-meta, mapping, obj:*, path-index, stream, GC, ext
    // values, counters. The recovery path must rebuild EVERY piece of
    // this from scratch.
    full_valkey_wipe(&harness.tv).await;

    // Sanity check: three specific keys we expect to be absent post-
    // wipe. A failed wipe would let the next UpdateLayer succeed
    // without exercising the recovery branch, silently passing the
    // assertion phase.
    let session_meta_key = harness.tv.key_schema().session_meta(&session_id);
    let mapping_key = harness
        .tv
        .key_schema()
        .layer_mapping(&session_id, LAYER_NAME);
    let obj_header_key = harness.tv.key_schema().obj_header(&object_id.to_string());
    let session_meta_exists: bool = harness
        .tv
        .client()
        .pool
        .exists(&session_meta_key)
        .await
        .expect("EXISTS sess probe must succeed");
    let mapping_exists: bool = harness
        .tv
        .client()
        .pool
        .exists(&mapping_key)
        .await
        .expect("EXISTS mapping probe must succeed");
    let obj_header_after: std::collections::HashMap<String, String> = harness
        .tv
        .client()
        .pool
        .hgetall(&obj_header_key)
        .await
        .expect("HGETALL obj:hdr probe must succeed");
    assert!(
        !session_meta_exists,
        "full_valkey_wipe must remove le:{{sid}}:sess; a surviving key here means \
         the SCAN missed it and session-recovery would silently no-op via the \
         pre-acquire AlreadyPresent short-circuit"
    );
    assert!(
        !mapping_exists,
        "full_valkey_wipe must remove le:{{sid}}:layer:{{hex(name)}}; a surviving \
         mapping means the second-hop recovery would target restored object state \
         (not what AC #2 is exercising)"
    );
    assert!(
        obj_header_after.is_empty(),
        "full_valkey_wipe must leave obj:hdr empty; got {} fields: {:?}. A \
         surviving header means layer-recovery never fires (the HGETALL inside \
         update_layer_once succeeds and the handler proceeds without consulting \
         the recovery hook).",
        obj_header_after.len(),
        obj_header_after.keys().collect::<Vec<_>>()
    );

    // Cache invalidation — see helper docstring.
    invalidate_session_cache(&harness, &session_id);

    // Step 6: Pre-call recovery-metric baselines. Fresh harness ⇒ every
    // cell should be 0, but the snapshot pattern is robust to any
    // future `Metrics::new` wiring change that pre-seeds a cell.
    let baseline = capture_recovery_baseline(&harness);
    assert_eq!(
        baseline.session_restored, 0,
        "pre-wipe `restored{{session}}` MUST be zero (fresh harness Metrics); got {}",
        baseline.session_restored
    );
    assert_eq!(
        baseline.layer_restored, 0,
        "pre-wipe `restored{{layer}}` MUST be zero (fresh harness Metrics); got {}",
        baseline.layer_restored
    );

    // Step 7: Dispatch the post-wipe UpdateLayer. This MUST trigger
    // BOTH recovery hops in series and return Ok with a `sequence_num
    // > pre_wipe_max_seq`. The handler-level recovery branch in
    // `handle_update_layer` runs the layer-recovery; the cache funnel
    // in `update_layer_once`'s session-resolution path runs the
    // session-recovery.
    let post_recovery_seq = issue_update(
        &harness,
        &session_id,
        object_id,
        delta_post_recovery(),
        3_000,
    )
    .await;

    // Step 8: post-recovery sequence_num MUST be strictly greater than
    // the pre-wipe max. The apply pipeline seeds the restored header at
    // `restore_floor + checkpoint_seq_safety_margin` (b.8y5) — the floor
    // being the checkpoint's `snap_seq` (which equals `pre_wipe_max_seq`
    // because the loop captured the post-burst state; the wipe took the
    // per-object stream, so the high-water probe contributes nothing) —
    // and the re-issued UpdateLayer bumps it once more.
    assert!(
        post_recovery_seq > pre_wipe_max_seq,
        "Epic AC #2: post-recovery sequence_num ({post_recovery_seq}) MUST be \
         strictly greater than pre-wipe max ({pre_wipe_max_seq})"
    );

    // Step 9: Exactly ONE recovery attempt on EACH scope. This is the
    // composition assertion the rest of the Epic builds on.
    //
    // We use `wait_for_recovery_metric` (bounded poll) rather than a
    // bare equality check because the layer-recovery metric increment
    // happens inside `RecoveryHook::recover_object`, which is awaited
    // inside the handler — so by the time `handle_update_layer`
    // returns, both increments are visible. The bounded poll covers
    // the slim window where a future refactor moves either increment
    // to a fire-and-forget background task; without it the test would
    // be flaky against that change rather than catching it.
    wait_for_recovery_metric(
        &harness,
        "session",
        "restored",
        baseline.session_restored + 1,
        10,
    )
    .await;
    wait_for_recovery_metric(
        &harness,
        "layer",
        "restored",
        baseline.layer_restored + 1,
        10,
    )
    .await;

    // The other cells must NOT have ticked — pins the "exactly one
    // restored" contract across both scopes.
    assert_eq!(
        session_recovery_metric(&harness, "not_recoverable"),
        baseline.session_not_recoverable,
        "post-recovery `not_recoverable{{session}}` MUST equal pre-call snapshot — \
         the session-checkpoint blob was readable so the apply branch ran, not \
         the not_recoverable arm"
    );
    assert_eq!(
        session_recovery_metric(&harness, "error"),
        baseline.session_error,
        "post-recovery `error{{session}}` MUST equal pre-call snapshot — no \
         transient backend failure was injected"
    );
    assert_eq!(
        session_recovery_metric(&harness, "already_present"),
        baseline.session_already_present,
        "post-recovery `already_present{{session}}` MUST equal pre-call snapshot — \
         the wipe drove the pre-acquire EXISTS check to false, so the cell that \
         tracks the short-circuit path stays untouched"
    );
    assert_eq!(
        layer_recovery_metric(&harness, "not_recoverable"),
        baseline.layer_not_recoverable,
        "post-recovery `not_recoverable{{layer}}` MUST equal pre-call snapshot — \
         the layer-checkpoint generation was readable"
    );
    assert_eq!(
        layer_recovery_metric(&harness, "error"),
        baseline.layer_error,
        "post-recovery `error{{layer}}` MUST equal pre-call snapshot — no \
         transient backend failure was injected"
    );

    // Step 10: ListSessions on the original base_path MUST return
    // exactly one session matching the original. Proves the path-index
    // restore landed end-to-end — ListSessions reads
    // `path_names_index` and surfaces the recovered session; without
    // the restore, it would return an empty list (or an inconsistent
    // ghost-member result).
    let list_resp = handle_list_sessions(
        &harness.svc,
        make_unary_token_request(ListSessionsRequest {
            path: SESSION_BASE_PATH.to_owned(),
        }),
    )
    .await
    .expect(
        "post-recovery ListSessions MUST succeed; a failure here means the \
         path-index was not restored by session-recovery (Task 5 applier \
         regression) or the ghost-member reconciliation diverged",
    );
    let sessions = list_resp.into_inner().sessions;
    assert_eq!(
        sessions.len(),
        1,
        "post-recovery ListSessions MUST return exactly one session under \
         base_path={SESSION_BASE_PATH}; got {} sessions: {:?}",
        sessions.len(),
        sessions
            .iter()
            .map(|s| (s.session_id.clone(), s.name.clone()))
            .collect::<Vec<_>>()
    );
    let recovered = &sessions[0];
    assert_eq!(
        recovered.session_id, session_id,
        "post-recovery ListSessions session_id MUST match the pre-wipe captured \
         id ({session_id}); got {}",
        recovered.session_id
    );
    assert_eq!(
        recovered.name, SESSION_NAME,
        "post-recovery ListSessions session.name MUST equal the original \
         session name ({SESSION_NAME}); got {}",
        recovered.name
    );

    // Step 11: read_object_state_for_test returns the restored materialized
    // state. Three layered checks:
    //   (a) snapshot.object_id matches the pre-wipe captured id — apply
    //       restored STATE under the same `object_id` rather than re-
    //       minting a new one (Epic AC #9 invariant, partially observable
    //       here).
    //   (b) the original section (from CreateLayer) is present — apply
    //       replayed the checkpoint generation faithfully.
    //   (c) the post-recovery SetField (POST_RECOVERY_FIELD_KEY) is
    //       present — the commit re-issue inside the layer-recovery
    //       branch actually landed (proves the recovery branch did more
    //       than just restore the checkpoint).
    let snapshot = read_object_state_for_test(
        harness.tv.client(),
        harness.tv.key_schema(),
        &object_id.to_string(),
    )
    .await
    .expect(
        "post-recovery read_object_state_for_test MUST succeed — the apply pipeline \
         must have restored the obj:hdr header. A NotFound here means \
         layer-recovery did not actually apply the checkpoint generation.",
    );

    assert_eq!(
        snapshot.object_id, object_id,
        "post-recovery snapshot.object_id MUST equal pre-wipe captured \
         object_id ({object_id}); got {}. A mismatch would indicate apply \
         re-minted the object instead of restoring it.",
        snapshot.object_id
    );
    assert!(
        snapshot.sections.contains_key(&SECTION_ID),
        "post-recovery snapshot MUST contain the original section_id={SECTION_ID} \
         restored from the layer-checkpoint generation; got sections={:?}",
        snapshot.sections.keys().collect::<Vec<_>>()
    );
    let post_recovery_field_present = snapshot
        .fields
        .iter()
        .any(|f| f.section_id == SECTION_ID && f.key_name == POST_RECOVERY_FIELD_KEY);
    assert!(
        post_recovery_field_present,
        "post-recovery snapshot MUST contain the SetField '{POST_RECOVERY_FIELD_KEY}' \
         landed by the re-issued UpdateLayer (proves the recovery branch re-issued \
         the commit, not just restored the checkpoint); got fields={:?}",
        snapshot
            .fields
            .iter()
            .map(|f| (f.section_id, &f.key_name))
            .collect::<Vec<_>>()
    );

    // Graceful shutdown of the checkpoint loop. Drops are sufficient
    // for cleanup but cancelling + awaiting yields a deterministic
    // teardown and removes the slim window where a still-running tick
    // could race with the test runtime's shutdown.
    harness.cancel.cancel();
    let _ = harness.loop_handle.await;
}

// ── Bytes-differ polling helpers ─────────────────────────────────────────────
//
// The CreateLayer-driven session-checkpoint write is fire-and-forget
// through the dispatcher; a polling helper that just checks "blob is
// present" returns immediately because the pre-CreateSession blob is
// still there. We need a poll that returns only once the BYTES have
// changed — proving the CreateLayer-driven write actually landed and
// the restored session will carry the freshly-created layer mapping.
//
// Same applies to the post-burst layer-checkpoint generation: the seed
// generation from the immediate-post-CreateLayer tick has `snap_seq` ≈
// 1 (from CreateLayer's seq bump); we need a generation that reflects
// the burst's `pre_wipe_max_seq` so the restore-and-re-issue chain
// returns a `sequence_num > pre_wipe_max_seq`. The
// `list_layer_checkpoints` response carries `snap_seq` per handle; we
// poll until a handle with `snap_seq >= pre_wipe_max_seq` lands.

async fn wait_for_post_create_layer_session_checkpoint(
    harness: &CheckpointHarness,
    session_id: &str,
    pre_create_layer_blob: &[u8],
    deadline_secs: u64,
) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(deadline_secs);
    let poll_interval = std::time::Duration::from_millis(50);
    loop {
        match harness.store.get_session_checkpoint(session_id).await {
            Ok(Some(blob)) if blob.bytes.as_ref() != pre_create_layer_blob => return,
            Ok(_) => {
                // Writer's post-CreateLayer put has not landed yet — keep
                // polling. The pre-CreateLayer blob is still the most
                // recent on the store; the bytes-differ check fails.
            }
            Err(err) => {
                panic!(
                    "get_session_checkpoint failed during post-CreateLayer wait \
                     (session_id={session_id}, deadline_secs={deadline_secs}): {err}"
                );
            }
        }
        if std::time::Instant::now() >= deadline {
            panic!(
                "wait_for_post_create_layer_session_checkpoint timed out after \
                 {deadline_secs}s — the CreateLayer-driven session-checkpoint write \
                 never landed (bytes did not differ from the pre-CreateLayer blob). \
                 Common causes: LayerStore was constructed without \
                 `with_checkpoint_writer` (harness wiring regression), or the \
                 lifecycle hook in `LayerStore::create_layer` short-circuited \
                 before reaching the enqueue call."
            );
        }
        tokio::time::sleep(poll_interval).await;
    }
}

async fn wait_for_post_burst_layer_checkpoint(
    harness: &CheckpointHarness,
    object_id: u64,
    pre_wipe_max_seq: u64,
    deadline_secs: u64,
) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(deadline_secs);
    let poll_interval = std::time::Duration::from_millis(50);
    loop {
        match harness.store.list_layer_checkpoints(object_id).await {
            Ok(handles) => {
                if handles.iter().any(|h| h.snap_seq >= pre_wipe_max_seq) {
                    return;
                }
            }
            Err(err) => {
                panic!(
                    "list_layer_checkpoints failed during post-burst wait \
                     (object_id={object_id}, pre_wipe_max_seq={pre_wipe_max_seq}, \
                     deadline_secs={deadline_secs}): {err}"
                );
            }
        }
        if std::time::Instant::now() >= deadline {
            panic!(
                "wait_for_post_burst_layer_checkpoint timed out after {deadline_secs}s \
                 — no layer-checkpoint generation with snap_seq >= {pre_wipe_max_seq} \
                 landed for object_id={object_id}. The leader-elected loop may have \
                 skipped a tick (idle-gate regression) or the burst's UpdateLayer \
                 calls did not advance `last_committed_seq` on the loop's \
                 watermark map."
            );
        }
        tokio::time::sleep(poll_interval).await;
    }
}
