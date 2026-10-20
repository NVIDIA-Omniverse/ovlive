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

//! End-to-end recovery test for `UpdateLayer` (Epic AC #7, Subtask
//! `t3.fgj.ya.wg.rt` under Task `t2.fgj.ya.wg`, Epic `t1.fgj.ya`, Bee
//! `b.fgj`).
//!
//! Drives the full write → checkpoint → wipe → recover → re-write cycle
//! through the production gRPC handler:
//!
//! 1. `CheckpointHarness` brings up Valkey + a LocalDisk-backed
//!    `LiveEditService` with an ENABLED `RecoveryHook` (`tick_secs=1`,
//!    `interval=0`, `quiesce=0` so the leader-elected loop produces a
//!    generation immediately after each committed write — see helpers.rs
//!    rationale).
//! 2. `CreateSession` + `CreateLayer` (non-empty initial OMPF carrying
//!    one section and an inline field). The first
//!    `UpdateLayer`s then drive the post-create sequence number forward.
//! 3. After the burst of UpdateLayers, `wait_for_checkpoint` blocks
//!    until the loop lands at least one generation under the
//!    `LocalDiskCheckpointStore`.
//! 4. `wipe_layer_state` `UNLINK`s every `le:{object_id}:obj:*` key,
//!    simulating the data-loss surface (failover to a fresh replica,
//!    persistence wipe). The layer mapping
//!    (`le:{session_id}:layer:{hex(name)}`) is left intact — so
//!    `get_layer_mapping` in `handle_update_layer` still resolves, and
//!    the recovery branch fires from the second NotFound surface
//!    (`update_layer` returning `NotFound` on the empty header HGETALL
//!    inside `update_layer_once`). See `src/grpc/layer.rs` step 6 + the
//!    SR-2.4 comment block.
//! 5. A subsequent `UpdateLayer` triggers the recovery branch, applies
//!    the checkpoint via `apply_checkpoint`, re-issues the commit, and
//!    succeeds — producing a `sequence_num` strictly greater than the
//!    pre-wipe max.
//! 6. The `live_edit_checkpoint_recovery_attempts_total{result="restored"}`
//!    counter MUST equal exactly 1 (the at-most-once contract pins one
//!    recovery per RPC).
//! 7. Post-recovery `read_object_state_for_test` must observe a non-empty
//!    header (the restored sections / fields plus the new write) and the
//!    snapshot's `object_id` field must match the pre-wipe captured id
//!    (the apply pipeline restores STATE under the same `object_id`
//!    without re-minting a new generation).
//!
//! Sandbox env note: this test uses `TestValkey::setup`, which reads
//! `TEST_VALKEY_URL`. Under the project sandbox the URL must include the
//! password and `host.docker.internal` host; see CLAUDE.md.

use fred::interfaces::HashesInterface;

use live_edit::grpc::layer::{handle_create_layer, handle_update_layer};
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::read_object_state_for_test;
use oodf_ffi::{
    CommandDiffChangeSection, OwnedDelta, SetField, StructCommand, TimeSample as OodfTimeSample,
};

use super::helpers::*;

use crate::common::owned_delta::delta_to_ompf_bytes;

const SESSION_NAME: &str = "e2e-update-recovery-session";
const LAYER_NAME: &str = "e2e-update-recovery-layer";
const CLIENT_ID: &str = "e2e-update-recovery-client";
const CREATED_BY: &str = "e2e-update-recovery-user";
const SESSION_BASE_PATH: &str = "/stage";

// ── Streaming framing helpers ────────────────────────────────────────────────
//
// Imported from `tests/integration/checkpoint/helpers.rs` via
// `use super::helpers::*;` (b.dzd Phase 1 deduplication).

// ── Delta fixture builders ───────────────────────────────────────────────────
//
// `populated_initial_delta` seeds the layer with one section + one inline
// field so the post-restore `read_object_state_for_test` has restorable sections /
// fields to observe. Without populated content, asserting "post-recovery
// state contains restored sections/fields" would be vacuous (the snapshot
// would be empty either way).
//
// `delta_with_field_update` builds a small diff carrying one SetField update
// — the per-UpdateLayer payload used both during the pre-wipe writes and
// for the post-wipe recovery write.
//
// `delta_with_time_sample_update` carries one time-sample. Mixing field +
// time-sample updates exercises both Lua-fast-path branches in update_layer
// so we get coverage of the real shape commits take in production.

const SECTION_ID: u64 = 42;
const FIELD_KEY: &str = "post-recovery-field";

fn populated_initial_delta() -> OwnedDelta {
    OwnedDelta {
        is_diff: true,
        base_version: 0,
        source_format: "usd".to_string(),
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 0,
            section_id: SECTION_ID,
            section_name: "e2e-section".to_string(),
            section_type: 1,
            section_order: 0,
        })],
        set_fields: vec![SetField {
            section_id: SECTION_ID,
            key_name: "initial-field".to_string(),
            value_or_ext_hash: vec![0xAA, 0xBB, 0xCC],
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
    // before applying — effectively erasing prior commits. The storage
    // layer does not validate `base_version` against the current
    // `sequence_num` (OCC happens via the WATCH/MULTI/EXEC or Lua
    // CAS-on-seq path, not base_version equality), so a small constant
    // is safe and is the documented shape for incremental updates.
    OwnedDelta {
        is_diff: true,
        base_version: 1,
        set_fields: vec![SetField {
            section_id: SECTION_ID,
            key_name: format!("field-{set_order}"),
            value_or_ext_hash: vec![marker, marker.wrapping_add(1), marker.wrapping_add(2)],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order,
        }],
        ..Default::default()
    }
}

fn delta_with_time_sample(_set_order: u64, time_value: f64, marker: u8) -> OwnedDelta {
    // Incremental diff — see `delta_with_field_update` for the
    // `base_version != 0` rationale (reset-delta avoidance).
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

// ── Final recovery payload ───────────────────────────────────────────────────
//
// The post-wipe UpdateLayer carries a SetField on a NEW key
// (`post-recovery-field`) so a post-recovery `read_object_state_for_test` can
// independently verify the new write landed on top of the restored
// state. A regression that lost the new write (e.g. a recovery loop that
// re-ran apply without re-issuing the commit) would show all original
// fields restored but the marker field missing.

fn delta_post_recovery() -> OwnedDelta {
    // Incremental diff (NOT a reset) — see `delta_with_field_update`.
    // The post-recovery write MUST land on top of the restored
    // checkpoint state, not wipe it; otherwise the snapshot assertion
    // below would observe only this field and none of the originals.
    OwnedDelta {
        is_diff: true,
        base_version: 1,
        set_fields: vec![SetField {
            section_id: SECTION_ID,
            key_name: FIELD_KEY.to_string(),
            value_or_ext_hash: vec![0xFE, 0xED, 0xFA, 0xCE],
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
    let request = make_streaming_token_request(stream);
    let response = handle_create_layer(&harness.svc, request)
        .await
        .expect("CreateLayer must succeed for E2E recovery harness setup");
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
    let request = make_streaming_token_request(stream);
    let response = handle_update_layer(&harness.svc, request)
        .await
        .expect("UpdateLayer must succeed in the pre-wipe / post-recovery happy path");
    response.into_inner().sequence_num
}

/// Probe the live label-cell counter value for a given outcome label on
/// the `live_edit_checkpoint_recovery_attempts_total{result, service}`
/// metric. Mirrors the `recovery_metric` helper in
/// `recovery_hook_test.rs` — re-implemented locally because the
/// `test-checkpoint-fault-injection`-gated module is not reachable from
/// this test file.
fn recovery_metric(harness: &CheckpointHarness, result: &str) -> u64 {
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

// =============================================================================
// Headline E2E test — Epic AC #7 for UpdateLayer.
// =============================================================================

/// Headline acceptance test pinning Epic AC #7 for the UpdateLayer RPC:
/// after a write → checkpoint → wipe cycle, a subsequent UpdateLayer
/// MUST succeed via the recovery branch and return a `sequence_num`
/// strictly greater than the pre-wipe max.
///
/// Why each step is load-bearing:
///
/// - The initial CreateLayer carries a populated delta so the
///   post-recovery snapshot has something to verify (without sections /
///   fields, "restored materialized state" would assert nothing).
/// - The burst of UpdateLayers (4 commits, mixing field + time-sample
///   updates) drives `sequence_num` well above 1 so the
///   "strictly greater than" assertion below has unambiguous signal
///   even if a regression collapsed the post-recovery sequence to a
///   small value.
/// - `wait_for_checkpoint(..., deadline_secs=10)` blocks until the
///   leader-elected loop has landed at least one BASE generation — the
///   helper's kind predicate, and what this test needs: the handle it
///   returns is restored from, and only a base carries full state. The
///   harness sets `tick_secs=1` and `interval=quiesce=0` so the loop
///   skips the idle gate after every commit; the deadline is sized for
///   CI jitter on the 1-second tick.
/// - `wipe_layer_state` UNLINKs only `le:{oid}:obj:*` (not the layer
///   mapping). This drives the engineer's wired recovery branch in
///   `handle_update_layer` from the SECOND NotFound surface — the
///   `update_layer` call's empty-header `NotFound` inside
///   `update_layer_once` (b.fgj SR-2.4 step 6) — exercising the full
///   recovery path: `try_restore_layer` → admission acquire →
///   apply_checkpoint → re-issue update_layer.
/// - The recovery-metric assertion (`restored == 1`) pins the
///   at-most-once contract: a regression that double-fired recovery
///   (e.g. dropped the `recovered_once` latch) would show `restored >=
///   2`, and a regression that bypassed recovery entirely would show
///   `restored == 0` plus a `NotFound` from the handler.
/// - The post-recovery snapshot assertion verifies the restored
///   materialized state. Three layered checks:
///     1. `snapshot.object_id == captured_object_id` — apply restored
///        state under the original `object_id` (no re-mint).
///     2. The original section (from the initial CreateLayer) is
///        present — apply faithfully replayed the checkpoint generation.
///     3. The new SetField from the post-wipe UpdateLayer is present —
///        the commit re-issue actually landed (proving the recovery
///        branch did more than just restore the checkpoint).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn update_layer_recovers_layer_state_after_valkey_wipe() {
    // Step 1: Harness with a fast-converging checkpoint loop.
    let harness = setup_checkpoint_harness(1).await;

    // Step 2: CreateSession (we go through the SessionStore directly
    // rather than CreateSession RPC because we don't need to exercise
    // the gRPC surface for session creation — the recovery target is
    // UpdateLayer).
    let session_store =
        SessionStore::new(harness.tv.client().clone(), harness.tv.key_schema().clone());
    let session_id = session_store
        .create(SESSION_NAME, SESSION_BASE_PATH, CREATED_BY)
        .await
        .expect("session creation must succeed");

    // Step 3: CreateLayer with a populated initial state. The
    // CreateLayer path mints the `object_id` we'll capture for the
    // wipe-and-recover assertions below.
    let object_id = create_layer_with_initial_state(&harness, &session_id).await;
    assert!(
        object_id > 0,
        "CreateLayer must mint a non-zero object_id; got {object_id}"
    );

    // Step 4: Issue 4 UpdateLayers and capture the highest acknowledged
    // sequence_num. Mixing field + time-sample updates exercises both
    // Lua-fast-path branches in update_layer so the post-wipe recovery
    // path is set up against the realistic shape commits take in
    // production.
    let mut pre_wipe_max_seq: u64 = 0;
    let updates = [
        (delta_with_field_update(1, 0x11), 1_001_u64),
        (delta_with_time_sample(2, 0.0, 0x22), 1_002_u64),
        (delta_with_field_update(3, 0x33), 1_003_u64),
        (delta_with_time_sample(4, 1.0, 0x44), 1_004_u64),
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
        pre_wipe_max_seq >= 5,
        "pre-wipe max sequence_num must be at least 5 (1 from CreateLayer + 4 \
         UpdateLayers); got {pre_wipe_max_seq}"
    );

    // Step 5: Wait for the leader-elected checkpoint loop to land at
    // least one generation. With `tick_secs=1` and `interval=quiesce=0`
    // the loop should produce a generation within a single tick window;
    // a 10 s deadline is comfortably above CI jitter.
    let _handle = wait_for_checkpoint(&harness.store, object_id, 10).await;

    // Sanity check: pre-wipe baseline of the `restored` metric cell is
    // zero. The harness wires a fresh Registry / Metrics inside
    // setup_checkpoint_harness so there is no leakage from prior tests.
    assert_eq!(
        recovery_metric(&harness, "restored"),
        0,
        "pre-wipe `restored` recovery metric MUST be zero (fresh harness Metrics)"
    );

    // Step 6: Wipe `le:{object_id}:obj:*` only. Leaves the layer mapping
    // intact so `get_layer_mapping` resolves; the empty `obj:hdr` then
    // surfaces NotFound from `update_layer_once`, tripping the recovery
    // branch in `handle_update_layer` (b.fgj SR-2.4 step 6).
    wipe_layer_state(&harness.tv, object_id).await;

    // Sanity: confirm the wipe actually emptied the header. A failed
    // wipe would let the next UpdateLayer succeed without exercising
    // the recovery branch, silently passing the rest of the test.
    let header_key = harness.tv.key_schema().obj_header(&object_id.to_string());
    let post_wipe_header: std::collections::HashMap<String, String> = harness
        .tv
        .client()
        .pool
        .hgetall(&header_key)
        .await
        .expect("HGETALL obj:hdr must succeed");
    assert!(
        post_wipe_header.is_empty(),
        "wipe_layer_state must leave obj:hdr empty; got {} fields: {:?}",
        post_wipe_header.len(),
        post_wipe_header.keys().collect::<Vec<_>>()
    );

    // Step 7: Issue the post-wipe UpdateLayer against the same
    // (session_id, layer_name, object_id). This MUST trigger the
    // recovery branch, apply the checkpoint, re-issue the commit, and
    // return Ok with a sequence_num > pre_wipe_max_seq.
    let post_recovery_seq = issue_update(
        &harness,
        &session_id,
        object_id,
        delta_post_recovery(),
        2_000,
    )
    .await;

    // Step 8: Post-recovery sequence_num MUST be strictly greater than
    // the pre-wipe max. The apply pipeline seeds the restored header at
    // `restore_floor + checkpoint_seq_safety_margin` (b.8y5) — the floor
    // being `max(snap_seq, stream high-water)`, here just `snap_seq`
    // because the wipe took the per-object stream with it — and the
    // re-issued UpdateLayer bumps it once more. The resulting value is
    // therefore `snap_seq + safety_margin + 1` at minimum, which (because
    // snap_seq covers the last committed write) always clears
    // `pre_wipe_max_seq`.
    assert!(
        post_recovery_seq > pre_wipe_max_seq,
        "Epic AC #7: post-recovery sequence_num ({post_recovery_seq}) MUST be \
         strictly greater than pre-wipe max ({pre_wipe_max_seq})"
    );

    // Step 9: Exactly ONE recovery attempt landed in the `restored`
    // metric cell. Pins the at-most-once contract for this RPC.
    assert_eq!(
        recovery_metric(&harness, "restored"),
        1,
        "Epic AC #7: exactly ONE recovery attempt must have reported `restored`; \
         got {}. A value of 0 means the recovery branch never fired (test wipe \
         did not surface NotFound to the handler); a value >= 2 means the \
         `recovered_once` latch is broken and the handler double-fired recovery.",
        recovery_metric(&harness, "restored")
    );
    assert_eq!(
        recovery_metric(&harness, "not_recoverable"),
        0,
        "post-recovery `not_recoverable` cell must stay zero — the checkpoint \
         was readable and the apply pipeline succeeded"
    );
    assert_eq!(
        recovery_metric(&harness, "error"),
        0,
        "post-recovery `error` cell must stay zero — no transient backend failure \
         was injected"
    );

    // Step 10: Post-recovery materialized state. `read_object_state_for_test`
    // returns NotFound if the header is empty — its success implies the
    // apply pipeline restored the header. We then verify:
    //   (a) the snapshot's `object_id` matches the captured pre-wipe id
    //       — apply restored STATE under the same `object_id` rather
    //       than re-minting a new one (Epic AC #9 invariant, but
    //       partially observable here).
    //   (b) the original section from CreateLayer is present — apply
    //       replayed the checkpoint generation faithfully.
    //   (c) the post-recovery SetField (`post-recovery-field`) is
    //       present — the commit re-issue inside the recovery branch
    //       actually landed.
    let snapshot = read_object_state_for_test(
        harness.tv.client(),
        harness.tv.key_schema(),
        &object_id.to_string(),
    )
    .await
    .expect(
        "post-recovery read_object_state_for_test must succeed (apply pipeline \
         must have restored the header)",
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
         restored from the checkpoint; got sections={:?}",
        snapshot.sections.keys().collect::<Vec<_>>()
    );

    let post_recovery_field_present = snapshot
        .fields
        .iter()
        .any(|f| f.section_id == SECTION_ID && f.key_name == FIELD_KEY);
    assert!(
        post_recovery_field_present,
        "post-recovery snapshot MUST contain the SetField '{FIELD_KEY}' landed \
         by the re-issued UpdateLayer (proves the recovery branch re-issued the \
         commit, not just restored the checkpoint); got fields={:?}",
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
