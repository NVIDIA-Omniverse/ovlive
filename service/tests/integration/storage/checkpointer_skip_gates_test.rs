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

// Task t2.fgj.ya.9j Subtasks .pi (skip-if-no-commits) + .ws (quiesce gate).
// Body lands via the dispatched test-writers.
//
// .pi — Skip-if-no-commits gate: per SDD SR-2.2 (AC#5), a layer whose live
// `seq` equals `last_checkpoint_seq` has not committed anything new since the
// last checkpoint and MUST be skipped. The operator-visible signal is the
// `live_edit_checkpoint_writes_total{result="skipped", kind="layer"}` counter.
// A regression that drops the `seq == last_checkpoint_seq` early-return in
// `run_one_tick` would re-checkpoint idle layers forever and burn adapter I/O
// (plus mis-attributing successive snapshots to the same seq cursor).
//
// To isolate the skip-if-no-commits gate from the quiesce + interval gates we
// run the loop with `interval = 0` + `quiesce = 0`, so the only gate that can
// fire is the seq-equals-cursor one. tick is small so observation latency is
// well under the polling deadlines.
//
// Sandbox env note: this test uses `TestValkey::setup_with_pool_size`, which
// reads `TEST_VALKEY_URL`. Under the project sandbox set the URL to
// `redis://:${VALKEY_PASSWORD}@host.docker.internal:6379`.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use fred::interfaces::HashesInterface;
use tempfile::TempDir;
use tokio::time::{sleep, timeout, Instant};
use tokio_util::sync::CancellationToken;

use live_edit::checkpoint::adapters::LocalDiskCheckpointStore;
use live_edit::checkpoint::{
    spawn_checkpoint_loop, CheckpointLoopConfig, CheckpointStore, TailCaptureConfig,
};
use live_edit::domain::types::ObjectState;
use live_edit::observability::metrics::{Metrics, SharedMetrics};
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::relay::RelayedUpdates;
use live_edit::storage::test_support::TestValkey;
use oodf_ffi::OwnedDelta;

// ────────────────────────────────────────────────────────────────────────
// Helpers — mirror the patterns established by the sibling end-to-end and
// metrics tests so a future reader doesn't have to context-switch between
// helper styles when chasing a skip-gate regression.
// ────────────────────────────────────────────────────────────────────────

const REPLICA_ID: &str = "skip-gates-test-replica";

/// Construct a `SharedMetrics` over a fresh `Registry`. Each test owns its
/// own metrics so cross-test interference is impossible.
fn make_metrics() -> SharedMetrics {
    Arc::new(crate::common::metrics_scrape_only(REPLICA_ID))
}

/// Read `live_edit_checkpoint_writes_total{result=<result>, kind="layer"}`.
/// Counter is `f64` but only ever bumped by 1.0, so casting to `u64` is exact.
fn read_writes(metrics: &Metrics, result: &str) -> u64 {
    live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(metrics),
        "live_edit_checkpoint_writes_total",
        &[("result", result), ("kind", "layer")],
    )
    .unwrap_or(0.0) as u64
}

/// Read `last_checkpoint_seq` off `le:{oid}:obj:hdr`. Returns `None` when the
/// field is absent (the loop has not yet written the cursor for this oid).
async fn read_last_checkpoint_seq(tv: &TestValkey, object_id: u64) -> Option<u64> {
    let key = tv.key_schema().obj_header(&object_id.to_string());
    let raw: Option<String> = tv
        .client()
        .pool
        .hget(&key, "last_checkpoint_seq")
        .await
        .expect("HGET last_checkpoint_seq must succeed");
    raw.and_then(|s| s.parse::<u64>().ok())
}

/// Bounded-poll until `last_checkpoint_seq` for `object_id` is `Some(v)` with
/// `v >= min`. Returns the observed cursor. 10 s deadline matches the
/// sibling tests' polling budget.
async fn wait_for_cursor_at_least(tv: &TestValkey, object_id: u64, min: u64) -> u64 {
    let deadline = Duration::from_secs(10);
    let result = timeout(deadline, async {
        loop {
            if let Some(cursor) = read_last_checkpoint_seq(tv, object_id).await {
                if cursor >= min {
                    return cursor;
                }
            }
            sleep(Duration::from_millis(25)).await;
        }
    })
    .await;
    result.unwrap_or_else(|_| {
        panic!(
            "timed out waiting for last_checkpoint_seq>={min} on oid={object_id} \
             within {deadline:?}"
        )
    })
}

/// Bounded-poll an async predicate. Returns `true` when the predicate
/// returns `true` before the deadline, `false` otherwise. Same shape as the
/// sibling metrics test's `poll_until` for cross-file consistency.
async fn poll_until<F, Fut>(deadline: Duration, mut pred: F) -> bool
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = bool>,
{
    let start = Instant::now();
    while start.elapsed() < deadline {
        if pred().await {
            return true;
        }
        sleep(Duration::from_millis(25)).await;
    }
    pred().await
}

/// Count `.ompf` files under `<root>/.live-edit/layers/`, bucketed by oid
/// parsed from the `<oid>.<snap_seq>.ompf` filename prefix. Used to assert
/// retention-side state, since `LocalDiskCheckpointStore` persists each
/// generation as a discrete file.
fn count_ompf_files_per_oid(layers_dir: &std::path::Path) -> HashMap<u64, usize> {
    let mut by_oid: HashMap<u64, usize> = HashMap::new();
    let read = std::fs::read_dir(layers_dir).expect("read_dir layers must succeed");
    for entry in read {
        let entry = entry.expect("dir entry must succeed");
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.ends_with(".ompf") {
            continue;
        }
        if let Some(oid_str) = name.split('.').next() {
            if let Ok(oid) = oid_str.parse::<u64>() {
                *by_oid.entry(oid).or_insert(0) += 1;
            }
        }
    }
    by_oid
}

/// Seed one populated layer in a fresh session and return `(session_id,
/// layer_name, object_id)`. Each layer gets its own session to keep the
/// (`session_id`, `layer_name`) namespace independent.
async fn seed_one_layer(tv: &TestValkey, suffix: &str) -> (String, String, u64) {
    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let layer_store = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
    let session_id = session_store
        .create(
            &format!("skip-gates-session-{suffix}"),
            &format!("/stage/skip-gates-{suffix}"),
            "skip-gates-test-user",
        )
        .await
        .expect("session create must succeed");
    let layer_name = format!("skip-gates-layer-{suffix}");
    let initial_state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usda".to_string(),
        ..Default::default()
    };
    let oid = layer_store
        .create_layer(
            &session_id,
            &layer_name,
            "skip-gates-test-user",
            &initial_state,
            Arc::new(MockExternalValueStore::new()),
        )
        .await
        .expect("create_layer must succeed");
    (session_id, layer_name, oid)
}

/// Issue one empty-delta UpdateLayer so `obj:hdr seq` advances past the prior
/// `last_checkpoint_seq`. Empty `OwnedDelta::default()` is a no-op on state
/// but still bumps `seq` and XADDs to the per-object stream — exactly what the
/// "touched between ticks" leg of the per-layer test needs.
async fn empty_update(
    tv: &TestValkey,
    session_id: &str,
    layer_name: &str,
    object_id: u64,
    nonce: u64,
) {
    let layer_store = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
    layer_store
        .update_layer_relayed(
            session_id,
            layer_name,
            object_id,
            Some(OwnedDelta::default()),
            "skip-gates-test-client",
            nonce,
        )
        .await
        .expect("update_layer must succeed");
}

// ────────────────────────────────────────────────────────────────────────
// .pi-1 — Skip gate prevents re-checkpoint when no new commits land.
//
// Setup:
//   - Seed one layer (CreateLayer XADDs to the stream; live seq = 1).
//   - Spawn the loop with interval=0 + quiesce=0 so the ONLY active gate is
//     skip-if-no-commits.
//
// Phase 1: wait for the first checkpoint to land. The loop sees
// `seq=1 > last_checkpoint_seq=0`, runs the assembler, writes generation 1,
// and HSETs `last_checkpoint_seq=1` (well, snap_seq — which strictly tracks
// the live seq at assembly time). One `.ompf` lands on disk.
//
// Phase 2: do NOT issue any UpdateLayers. Snapshot the metrics, the on-disk
// file count, and the cursor. Now wait long enough for several ticks
// (each tick is 50 ms; 1 s gives ≥ 20 ticks). Re-read every observable:
//   - `result="skipped"` counter must have incremented by ≥ 1 (idle layer
//     is observed once per tick → multiple skips actually accumulate, but we
//     only assert ≥ 1 per the AC).
//   - The on-disk `.ompf` file count must still be 1 (a regression that
//     dropped the early-return would have produced a fresh generation per
//     tick, all retention-pruned down to retain_generations=2).
//   - `last_checkpoint_seq` must be unchanged (the cursor only advances on a
//     successful assembler tick).
//
// A regression that drops `if seq == last_checkpoint_seq { skipped; continue }`
// in `run_one_tick` fails this test: phase-2 ticks would re-snapshot the
// idle layer and the on-disk file count or cursor would drift.
// ────────────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn skip_gate_prevents_recheckpoint_when_no_commits() {
    let tv = TestValkey::setup_with_pool_size(8).await;
    let metrics = make_metrics();

    // Seed: one layer in its own session.
    let (_session_id, _layer_name, oid) = seed_one_layer(&tv, "single").await;

    // Tempdir-scoped LocalDisk adapter — RAII-cleaned on drop.
    let root: TempDir = tempfile::tempdir().expect("tempdir creation must succeed");
    let store: Arc<dyn CheckpointStore> = Arc::new(
        LocalDiskCheckpointStore::new(root.path().to_path_buf())
            .expect("LocalDiskCheckpointStore::new must succeed on a fresh root"),
    );

    // Spawn the loop. interval=0 + quiesce=0 isolates the skip-if-no-commits
    // gate as the only thing that could skip. tick=50ms gives ~20 ticks/sec.
    let config = CheckpointLoopConfig {
        tick: Duration::from_millis(50),
        interval: Duration::ZERO,
        quiesce: Duration::ZERO,
        retain_generations: 2,
        // Epic 6 counters-sync cadence; pushed well above the test runtime
        // so the counters-sidecar timer does not fire during this test.
        counters_sync: Duration::from_secs(3600),
        reconcile_interval: Duration::from_secs(3600),
        leader: crate::common::test_leader(REPLICA_ID),
        // Tails pinned explicitly OFF. b.3n1 promoted the binary default
        // to `true`; this suite predates the tail feature and was written
        // against an inert tail path, so the flag is part of the fixture
        // rather than something to inherit.
        tail: TailCaptureConfig {
            enabled: false,
            ..TailCaptureConfig::default()
        },
    };
    let cancel = CancellationToken::new();
    let handle = spawn_checkpoint_loop(
        tv.client().clone(),
        tv.key_schema().clone(),
        Arc::clone(&store),
        config,
        Some(Arc::clone(&metrics)),
        cancel.clone(),
    );

    // Phase 1: wait for the first checkpoint. After CreateLayer the live seq
    // is at least 1 and the loop's seq-vs-cursor check (1 > 0) lets the
    // assembler run. We poll for the cursor advancing to >= 1 — this is the
    // observable proof the assembler-Ok HSET path ran.
    let first_cursor = wait_for_cursor_at_least(&tv, oid, 1).await;
    assert!(
        first_cursor >= 1,
        "first-wave cursor must be >= 1 after CreateLayer; got {first_cursor}"
    );

    // Capture the on-disk file count for this oid at the start of phase 2.
    // The layers_dir is created by LocalDiskCheckpointStore::new; we read it
    // after the loop has had a chance to write generation 1.
    let layers_dir = root.path().join(".live-edit").join("layers");
    // The first checkpoint must have produced at least one .ompf for this oid.
    let files_after_phase1 = count_ompf_files_per_oid(&layers_dir);
    let phase1_count = *files_after_phase1.get(&oid).unwrap_or(&0);
    assert!(
        phase1_count >= 1,
        "phase 1 must produce at least one .ompf for oid={oid}; got {phase1_count} \
         (files_after_phase1={files_after_phase1:?})"
    );

    // Snapshot the skipped counter BEFORE phase 2 begins. The loop may have
    // emitted zero or more skips so far; we only care about the delta.
    let skipped_before_phase2 = read_writes(&metrics, "skipped");

    // Phase 2: do NOT issue any UpdateLayers. Wait for the skipped counter
    // to advance by at least 1 — the skip-if-no-commits gate fires on every
    // tick that observes the idle layer.
    let metrics_for_poll = metrics.clone();
    let saw_skipped = poll_until(Duration::from_secs(5), || {
        let m = metrics_for_poll.clone();
        async move { read_writes(&m, "skipped") > skipped_before_phase2 }
    })
    .await;
    assert!(
        saw_skipped,
        "result=\"skipped\" counter must increment within 5 s while the layer is idle; \
         a regression that dropped the `seq == last_checkpoint_seq` early-return in \
         run_one_tick would re-checkpoint the idle layer instead of skipping it"
    );

    // Stop the loop BEFORE reading the quiescent state — otherwise a late
    // tick could slip a write in between our observations.
    cancel.cancel();
    timeout(Duration::from_secs(5), handle)
        .await
        .expect("checkpoint loop must exit within 5 s of cancellation")
        .expect("checkpoint loop must not panic on shutdown");

    // The cursor must NOT have advanced past its phase-1 value. The
    // skip-if-no-commits gate does not bump `last_checkpoint_seq` (only the
    // assembler-Ok HSET arm does), so an idle layer's cursor is stable
    // across an arbitrary number of skipped ticks.
    let final_cursor = read_last_checkpoint_seq(&tv, oid)
        .await
        .expect("last_checkpoint_seq must remain set after phase 2");
    assert_eq!(
        final_cursor, first_cursor,
        "last_checkpoint_seq must be unchanged after idle phase; \
         phase1={first_cursor}, post-phase2={final_cursor} — a regression that \
         dropped the skip-if-no-commits gate would have advanced the cursor on \
         each idle-layer tick"
    );

    // The on-disk `.ompf` count for this oid must still be exactly the
    // phase-1 count. A regression that re-checkpointed the idle layer would
    // have created additional generations (and retention prune would have
    // capped them at 2, so a `phase1_count=1` test would see `2` here).
    let files_final = count_ompf_files_per_oid(&layers_dir);
    let final_count = *files_final.get(&oid).unwrap_or(&0);
    assert_eq!(
        final_count, phase1_count,
        "on-disk .ompf count for oid={oid} must be unchanged across the idle phase; \
         phase1={phase1_count}, final={final_count} (files_final={files_final:?}) — \
         re-checkpointing idle layers indicates the skip-if-no-commits gate did not fire"
    );

    // Final-state skipped counter must reflect the observed increment.
    let skipped_after = read_writes(&metrics, "skipped");
    assert!(
        skipped_after > skipped_before_phase2,
        "final skipped counter ({skipped_after}) must exceed phase-2-start \
         snapshot ({skipped_before_phase2})"
    );
}

// ────────────────────────────────────────────────────────────────────────
// .pi-2 — Skip gate is per-layer, not per-tick.
//
// Setup:
//   - Seed two layers (A and B) in separate sessions.
//   - Wait for the first checkpoint on BOTH so the cursor is set on each.
//   - Issue ONE empty UpdateLayer to layer A only — layer B stays idle.
//   - Run for additional ticks. Assert:
//       - Layer A gets re-checkpointed (`success` counter advances; A's cursor
//         advances; A's on-disk .ompf count advances toward `retain=2`).
//       - Layer B stays idle (no new generation on disk; B's cursor unchanged).
//       - `skipped` counter advances by at least 1 (B was skipped on at least
//         one tick during the observation window).
//
// A regression that made the skip-gate "tick-wide" rather than "per-layer"
// (e.g. an early-return out of the per-layer loop on the first idle layer
// encountered) would fail this test: depending on iteration order, EITHER A
// would not be re-checkpointed (if B is iterated first and the loop bails)
// OR B's idle status would not produce a skip increment (if A is iterated
// first and the loop bails). Either failure mode hits one of the asserts.
// ────────────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn skip_gate_is_per_layer_not_per_tick() {
    let tv = TestValkey::setup_with_pool_size(8).await;
    let metrics = make_metrics();

    // Seed two layers in distinct sessions so the layer-name namespace is
    // independent and the loop's SCAN sees two unambiguous obj:hdr keys.
    let (session_a, layer_a, oid_a) = seed_one_layer(&tv, "a").await;
    let (_session_b, _layer_b, oid_b) = seed_one_layer(&tv, "b").await;

    // Tempdir-scoped LocalDisk adapter.
    let root: TempDir = tempfile::tempdir().expect("tempdir creation must succeed");
    let store: Arc<dyn CheckpointStore> = Arc::new(
        LocalDiskCheckpointStore::new(root.path().to_path_buf())
            .expect("LocalDiskCheckpointStore::new must succeed on a fresh root"),
    );

    // Spawn the loop. Same gate-isolation config as case 1.
    let config = CheckpointLoopConfig {
        tick: Duration::from_millis(50),
        interval: Duration::ZERO,
        quiesce: Duration::ZERO,
        retain_generations: 2,
        // Epic 6 counters-sync cadence; pushed well above the test runtime
        // so the counters-sidecar timer does not fire during this test.
        counters_sync: Duration::from_secs(3600),
        reconcile_interval: Duration::from_secs(3600),
        leader: crate::common::test_leader(REPLICA_ID),
        // Tails pinned explicitly OFF. b.3n1 promoted the binary default
        // to `true`; this suite predates the tail feature and was written
        // against an inert tail path, so the flag is part of the fixture
        // rather than something to inherit.
        tail: TailCaptureConfig {
            enabled: false,
            ..TailCaptureConfig::default()
        },
    };
    let cancel = CancellationToken::new();
    let handle = spawn_checkpoint_loop(
        tv.client().clone(),
        tv.key_schema().clone(),
        Arc::clone(&store),
        config,
        Some(Arc::clone(&metrics)),
        cancel.clone(),
    );

    // Wait for first-wave checkpoints on BOTH layers. The cursor presence is
    // the observable proof the assembler ran for that oid.
    let cursor_a_phase1 = wait_for_cursor_at_least(&tv, oid_a, 1).await;
    let cursor_b_phase1 = wait_for_cursor_at_least(&tv, oid_b, 1).await;
    assert!(cursor_a_phase1 >= 1 && cursor_b_phase1 >= 1);

    let layers_dir = root.path().join(".live-edit").join("layers");
    let files_after_phase1 = count_ompf_files_per_oid(&layers_dir);
    let count_a_phase1 = *files_after_phase1.get(&oid_a).unwrap_or(&0);
    let count_b_phase1 = *files_after_phase1.get(&oid_b).unwrap_or(&0);
    assert!(
        count_a_phase1 >= 1 && count_b_phase1 >= 1,
        "phase 1 must produce at least one .ompf per oid; got a={count_a_phase1}, \
         b={count_b_phase1} (files_after_phase1={files_after_phase1:?})"
    );

    // Snapshot counters before the touched/untouched divergence.
    let skipped_before_phase2 = read_writes(&metrics, "skipped");
    let success_before_phase2 = read_writes(&metrics, "success");

    // Touch ONLY layer A. The empty-delta UpdateLayer advances `obj:hdr seq`
    // past A's current cursor; B's seq stays at its phase-1 value.
    empty_update(&tv, &session_a, &layer_a, oid_a, 7_777).await;

    // Wait for A's cursor to advance past its phase-1 value — observable
    // proof the assembler ran AGAIN for A (a successful tick).
    let advance_deadline = Duration::from_secs(10);
    let cursor_a_phase2 = timeout(advance_deadline, async {
        loop {
            if let Some(cur) = read_last_checkpoint_seq(&tv, oid_a).await {
                if cur > cursor_a_phase1 {
                    return cur;
                }
            }
            sleep(Duration::from_millis(25)).await;
        }
    })
    .await
    .unwrap_or_else(|_| {
        panic!(
            "layer A cursor did not advance past {cursor_a_phase1} within \
             {advance_deadline:?} after the second-wave UpdateLayer — the \
             touched-layer re-checkpoint path is broken"
        )
    });
    assert!(
        cursor_a_phase2 > cursor_a_phase1,
        "layer A cursor must strictly advance after the touched-layer commit; \
         phase1={cursor_a_phase1}, phase2={cursor_a_phase2}"
    );

    // Wait for the skipped counter to advance — observable proof that B's
    // idle status was processed as a skip on at least one of the ticks
    // between phase-1-end and now. A regression that aborted the per-tick
    // loop after the first idle layer (or before reaching B) would never
    // produce this increment.
    let metrics_for_poll = metrics.clone();
    let saw_skipped = poll_until(Duration::from_secs(5), || {
        let m = metrics_for_poll.clone();
        async move { read_writes(&m, "skipped") > skipped_before_phase2 }
    })
    .await;
    assert!(
        saw_skipped,
        "result=\"skipped\" counter must increment within 5 s — the untouched \
         layer B should hit the skip-if-no-commits gate while the touched \
         layer A is being re-checkpointed"
    );

    // Stop the loop and read quiescent final state.
    cancel.cancel();
    timeout(Duration::from_secs(5), handle)
        .await
        .expect("checkpoint loop must exit within 5 s of cancellation")
        .expect("checkpoint loop must not panic on shutdown");

    // Layer B's cursor must be UNCHANGED — it was idle across the whole
    // observation window.
    let cursor_b_final = read_last_checkpoint_seq(&tv, oid_b)
        .await
        .expect("last_checkpoint_seq must remain set for oid_b");
    assert_eq!(
        cursor_b_final, cursor_b_phase1,
        "layer B (idle) cursor must be unchanged across phase 2; \
         phase1={cursor_b_phase1}, final={cursor_b_final} — a regression that \
         dropped the per-layer skip gate would have re-checkpointed B and \
         advanced its cursor"
    );

    // On-disk file counts: A may have additional generations (capped at
    // retain_generations=2 by retention prune); B must have the same count
    // as phase 1.
    let files_final = count_ompf_files_per_oid(&layers_dir);
    let count_a_final = *files_final.get(&oid_a).unwrap_or(&0);
    let count_b_final = *files_final.get(&oid_b).unwrap_or(&0);
    assert_eq!(
        count_b_final, count_b_phase1,
        "layer B (idle) on-disk .ompf count must be unchanged across phase 2; \
         phase1={count_b_phase1}, final={count_b_final} — re-checkpointing an \
         idle layer is the regression this assertion catches"
    );
    assert!(
        count_a_final >= count_a_phase1,
        "layer A (touched) on-disk .ompf count must NOT decrease; \
         phase1={count_a_phase1}, final={count_a_final}"
    );
    // A's count may equal count_a_phase1 if the retention prune is keeping
    // pace, or it may be retain_generations=2 (i.e. larger). The strict
    // proof that A was re-checkpointed is the cursor advance above.

    // Final-state success counter must have advanced — at least one extra
    // success increment for A's re-checkpoint.
    let success_after = read_writes(&metrics, "success");
    assert!(
        success_after > success_before_phase2,
        "result=\"success\" counter must advance after the touched-layer \
         commit; before={success_before_phase2}, after={success_after} — the \
         touched layer A re-checkpoint should have produced at least one \
         success increment"
    );
}

// ────────────────────────────────────────────────────────────────────────
// .ws — Quiesce + interval gates (OR-gated, Task 9j AC, SDD SR-2.2).
//
// The checkpoint loop reads both gate timestamps off `le:{oid}:obj:hdr`:
//
//   - `last_server_ts` — the commit timestamp every commit path stamps, in
//     MICROSECONDS.
//   - `last_checkpoint_ms` — the loop's own post-write stamp, in
//     MILLISECONDS; absent means "never checkpointed".
//
// Each tick applies the OR-gated rule with DISTINCT references (b.yut):
//
//     write when (now - last_commit) >= quiesce_ms          [quiesce arm]
//             OR (now - last_checkpoint) >= interval_ms     [interval arm]
//
// Per the Task 9j Acceptance Criteria: "a layer idle for ≥
// LIVE_EDIT_CHECKPOINT_QUIESCE_SECS since its last commit gets a final
// checkpoint even if the interval has not elapsed." Quiesce is an
// EARLY-write trigger (an operator-visible "snapshot once writes fall
// quiet" knob), not an additional delay layered on top of interval.
// Interval is the cadence upper bound — every layer gets a checkpoint at
// least every interval, even under continuous write load.
//
// Test strategy
// -------------
// Drive the gate by writing the two header columns directly, per layer,
// after CreateLayer. That is exactly the surface production writes: no
// synthetic stream entry is involved, so a regression that reverts to
// parsing stream entry IDs (b.yut) cannot be masked by a fixture that
// fabricates wall-clock-shaped IDs production never emits.
//
// The config is default-SHAPED (quiesce below interval, same 1:2 ratio as the
// 30 s / 60 s defaults) but scaled up 10× to `quiesce = 300 s`,
// `interval = 600 s`, with the three cases' backdated ages scaled to match.
// The scaling is purely anti-flake headroom and buys ~250 s of it: every
// backdated age is a fixed point in the past, so real time elapsing during the
// test walks each case TOWARD its thresholds. At the un-scaled 30 s / 60 s
// numbers, Case A (the negative) sat only 25 s away from legitimately becoming
// quiesced — a stalled runner would flip it and the failure would read as a
// gate regression. The absolute wall-clock cost is unchanged: no test waits
// out a threshold, they are all reached by backdating.
//
//   - Case A — commit 50 s ago, checkpointed just now: 50 < 300 (quiesce)
//                            AND 0 < 600 (interval) → SKIP. Baseline: a
//                            freshly-committed, freshly-checkpointed layer
//                            is below both floors.
//   - Case B — commit 350 s ago, checkpointed 100 s ago: 350 ≥ 300 (quiesce
//                            ELAPSED) while the interval has NOT → WRITE.
//                            The quiesce floor is sufficient on its own.
//                            **Catches a regression that re-imposes
//                            AND-gating.**
//   - Case C — commit 10 s ago, checkpointed 650 s ago: the layer is under
//                            active write (10 < 300, quiesce NOT elapsed)
//                            but 650 ≥ 600 (interval ELAPSED) → WRITE. This
//                            is the b.yut case: a continuously-written layer
//                            MUST still be checkpointed at the cadence
//                            upper bound. Pre-fix — with both arms reading
//                            idle time — such a layer was never
//                            checkpointed at all.
//
// Sharing one fixture and one loop across the three cases (each layer in its
// own session for namespace isolation) keeps the test under one Valkey setup
// + one loop spin-up. The loop sees three layers per tick and decides
// independently per-layer; cross-layer interference is structurally
// impossible because the gate reads per-layer state.
//
// Regression coverage summary
// ---------------------------
//   * Drop the gate entirely (always write) → Case A writes → its cursor +
//     on-disk count asserts fail.
//   * Re-impose AND-gating (write only when BOTH thresholds elapsed) → Case
//     B does NOT write → its cursor + on-disk count asserts fail.
//   * Drop the quiesce arm (only the interval arm triggers writes) → Case B
//     does NOT write → its cursor + on-disk count asserts fail.
//   * Point the interval arm back at the commit timestamp (b.yut) → Case C
//     does NOT write → its cursor wait times out.
//   * Break the write path entirely (always skip) → Cases B and C never
//     write → both cursor waits time out.
//
// Why not a `now_fn` injection seam: the loop calls `SystemTime::now()`
// inline (no DI seam exists in `checkpointer.rs`). Backdating the real
// header columns drives the gate without modifying production code, and the
// pure `checkpoint_due` helper covers the clock-free table-driven cases.
// ────────────────────────────────────────────────────────────────────────

/// Backdate `obj:hdr last_server_ts` — the layer's last-commit timestamp —
/// to `age_ms` milliseconds ago.
///
/// **Units**: `last_server_ts` is written by production in MICROSECONDS
/// (`domain::time::now_micros`), so the helper multiplies by 1000 on the way
/// in. The loop divides by 1000 on the way out. A fixture that wrote
/// milliseconds here would make every layer look ~1970-era-old and every
/// gate arm pass — which is the shape of the b.yut defect itself.
async fn set_last_commit_age(tv: &TestValkey, object_id: u64, age_ms: u64) {
    let header_key = tv.key_schema().obj_header(&object_id.to_string());
    let server_ts_micros = now_unix_ms().saturating_sub(age_ms) * 1_000;
    let _: () = tv
        .client()
        .pool
        .hset(
            &header_key,
            ("last_server_ts", server_ts_micros.to_string()),
        )
        .await
        .expect("HSET last_server_ts must succeed");
}

/// Backdate `obj:hdr last_checkpoint_ms` — the loop's own cadence stamp — to
/// `age_ms` milliseconds ago. Milliseconds, matching what the loop writes on
/// its success arm. Setting this is how a test says "this layer already has
/// a recent checkpoint", which is the only way to hold the interval arm
/// closed.
async fn set_last_checkpoint_age(tv: &TestValkey, object_id: u64, age_ms: u64) {
    let header_key = tv.key_schema().obj_header(&object_id.to_string());
    let stamp_ms = now_unix_ms().saturating_sub(age_ms);
    let _: () = tv
        .client()
        .pool
        .hset(&header_key, ("last_checkpoint_ms", stamp_ms.to_string()))
        .await
        .expect("HSET last_checkpoint_ms must succeed");
}

/// Current wall-clock milliseconds since the UNIX epoch, matching
/// `checkpoint::now_unix_ms` in production.
fn now_unix_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn quiesce_and_interval_gates_or_gated_quiesce_triggers_early_write() {
    /// Gate thresholds under test — default-SHAPED (quiesce = half the
    /// interval) scaled 10× for anti-flake headroom. See the block comment.
    const QUIESCE: Duration = Duration::from_secs(300);
    const INTERVAL: Duration = Duration::from_secs(600);
    /// Case A's backdated commit age. The distance from here to `QUIESCE`
    /// (250 s) is the margin a stalled runner has to eat before Case A would
    /// LEGITIMATELY become quiesced and the negative assertion would stop
    /// meaning what it says.
    const CASE_A_COMMIT_AGE: Duration = Duration::from_secs(50);

    let tv = TestValkey::setup_with_pool_size(8).await;
    let metrics = make_metrics();

    // Seed three layers, one per gate-window case. Each in its own session
    // to keep the (`session_id`, `layer_name`) namespace independent.
    let (_sess_a, _name_a, oid_a) = seed_one_layer(&tv, "ws-fresh").await;
    let (_sess_b, _name_b, oid_b) = seed_one_layer(&tv, "ws-quiesced").await;
    let (_sess_c, _name_c, oid_c) = seed_one_layer(&tv, "ws-hot").await;

    // Backdate the two gate columns per layer so each case exercises a
    // different arm (see the block comment above):
    //   A: commit 50 s ago, checkpoint now       → neither arm → SKIP
    //   B: commit 350 s ago, checkpoint 100 s ago → quiesce arm → WRITE
    //   C: commit 10 s ago, checkpoint 650 s ago  → interval arm → WRITE
    set_last_commit_age(&tv, oid_a, CASE_A_COMMIT_AGE.as_millis() as u64).await;
    set_last_checkpoint_age(&tv, oid_a, 0).await;
    set_last_commit_age(&tv, oid_b, 350_000).await;
    set_last_checkpoint_age(&tv, oid_b, 100_000).await;
    set_last_commit_age(&tv, oid_c, 10_000).await;
    set_last_checkpoint_age(&tv, oid_c, 650_000).await;
    // Every backdated age above is a FIXED point in the past, so real time
    // elapsing from here walks each case toward its thresholds. Case A's
    // negative assertion is the one that can flip, so time it from here.
    let fixture_at = Instant::now();

    // Tempdir-scoped LocalDisk adapter — RAII-cleaned on drop.
    let root: TempDir = tempfile::tempdir().expect("tempdir creation must succeed");
    let store: Arc<dyn CheckpointStore> = Arc::new(
        LocalDiskCheckpointStore::new(root.path().to_path_buf())
            .expect("LocalDiskCheckpointStore::new must succeed on a fresh root"),
    );

    // Default-shaped config scaled 10×: interval=600 s, quiesce=300 s,
    // retain=2. `tick=50 ms` keeps observation latency low so the test runs
    // quickly — the thresholds are reached by backdating, never by waiting.
    let config = CheckpointLoopConfig {
        tick: Duration::from_millis(50),
        interval: INTERVAL,
        quiesce: QUIESCE,
        retain_generations: 2,
        // Epic 6 counters-sync cadence; pushed well above the test runtime
        // so the counters-sidecar timer does not fire during this test.
        counters_sync: Duration::from_secs(3600),
        reconcile_interval: Duration::from_secs(3600),
        leader: crate::common::test_leader(REPLICA_ID),
        // Tails pinned explicitly OFF. b.3n1 promoted the binary default
        // to `true`; this suite predates the tail feature and was written
        // against an inert tail path, so the flag is part of the fixture
        // rather than something to inherit.
        tail: TailCaptureConfig {
            enabled: false,
            ..TailCaptureConfig::default()
        },
    };
    let cancel = CancellationToken::new();
    let handle = spawn_checkpoint_loop(
        tv.client().clone(),
        tv.key_schema().clone(),
        Arc::clone(&store),
        config,
        Some(Arc::clone(&metrics)),
        cancel.clone(),
    );

    // Case C: positive write-path proof via the INTERVAL upper-bound
    // trigger on a layer that is NOT quiesced (last commit 1 s ago, well
    // inside the 30 s quiesce floor). The loop sees 65 s since the last
    // checkpoint ≥ interval 60 s → assembler runs → cursor advances to
    // >= 1. This is the b.yut regression surface: an implementation whose
    // interval arm measures time-since-commit rather than
    // time-since-checkpoint leaves this hot layer permanently
    // un-checkpointed and this poll times out.
    let cursor_c = wait_for_cursor_at_least(&tv, oid_c, 1).await;
    assert!(
        cursor_c >= 1,
        "Case C cursor must be >= 1 after the loop processes a continuously-written layer \
         whose last checkpoint is older than the interval; got {cursor_c}"
    );

    // Case B: positive write-path proof via the QUIESCE early-write
    // trigger. The loop sees 35 s since the last commit ≥ quiesce 30 s →
    // assembler runs even though only 10 s have passed since the last
    // checkpoint (interval 60 s has NOT elapsed). A regression that
    // re-imposes AND-gating (write only when both thresholds elapsed)
    // would keep Case B in SKIP and this poll times out.
    let cursor_b = wait_for_cursor_at_least(&tv, oid_b, 1).await;
    assert!(
        cursor_b >= 1,
        "Case B cursor must be >= 1 after the loop processes a quiesce-elapsed layer \
         (per Task 9j AC: quiesce is an early-write trigger, not a delay); got {cursor_b}"
    );

    // Wait for the skipped counter to advance — proof that Case A (fresh)
    // was visited in the per-layer loop and hit the skip branch. A
    // regression that aborts the per-layer loop after the first layer
    // would never produce this increment.
    let metrics_for_poll = metrics.clone();
    let saw_skipped = poll_until(Duration::from_secs(10), || {
        let m = metrics_for_poll.clone();
        async move { read_writes(&m, "skipped") >= 1 }
    })
    .await;
    assert!(
        saw_skipped,
        "result=\"skipped\" counter must accumulate >= 1 within 10 s for the fresh layer \
         (Case A: commit {CASE_A_COMMIT_AGE:?} ago, checkpoint just now — below both \
         floors); a regression that drops the gate entirely would instead snapshot Case A \
         as success"
    );

    // Stop the loop BEFORE reading negative assertions on Case A —
    // otherwise a late tick could slip a Case A write in between
    // observations.
    cancel.cancel();
    timeout(Duration::from_secs(5), handle)
        .await
        .expect("checkpoint loop must exit within 5 s of cancellation")
        .expect("checkpoint loop must not panic on shutdown");

    // Case A: fresh layer (commit 50 s ago, checkpoint just now) must NOT
    // have been checkpointed — neither the quiesce (300 s) nor the interval
    // (600 s) threshold is reached, so the gate skips. The
    // `last_checkpoint_seq` field must still be absent on `obj:hdr` (the
    // fixture seeded `last_checkpoint_ms` only, so a written cursor here
    // can only have come from the loop). A regression that dropped the gate
    // entirely would have written it.
    //
    // Precondition first: Case A's commit age is a fixed point in the past,
    // so a runner stalled for longer than the remaining margin would make the
    // layer LEGITIMATELY quiesced and a checkpoint the correct outcome.
    // Asserting that separately keeps a stall from being reported as a gate
    // regression.
    let elapsed_since_fixture = fixture_at.elapsed();
    let margin = QUIESCE - CASE_A_COMMIT_AGE;
    assert!(
        elapsed_since_fixture < margin,
        "TEST-RUNNER STALL, not a gate regression: {elapsed_since_fixture:?} of \
         real time elapsed between backdating Case A's columns and this \
         observation, eating the {margin:?} margin between its backdated commit \
         age ({CASE_A_COMMIT_AGE:?}) and the {QUIESCE:?} quiesce floor. Case A is \
         now legitimately quiesced and the loop would be RIGHT to checkpoint it. \
         Re-run; if this recurs, scale the fixture ages and thresholds together"
    );
    let cursor_a = read_last_checkpoint_seq(&tv, oid_a).await;
    assert!(
        cursor_a.is_none(),
        "Case A (commit {CASE_A_COMMIT_AGE:?} ago, checkpoint just now — below \
         both the {QUIESCE:?} quiesce and {INTERVAL:?} interval floors) must NOT \
         have a last_checkpoint_seq written; got {cursor_a:?} — a regression that \
         dropped the quiesce/interval gate would have snapshotted this fresh layer"
    );

    // On-disk: at least one .ompf for both B and C, zero for A.
    let layers_dir = root.path().join(".live-edit").join("layers");
    assert!(
        layers_dir.exists(),
        "layers dir must exist after Case B and Case C's writes at {layers_dir:?}"
    );
    let files = count_ompf_files_per_oid(&layers_dir);
    let count_a = *files.get(&oid_a).unwrap_or(&0);
    let count_b = *files.get(&oid_b).unwrap_or(&0);
    let count_c = *files.get(&oid_c).unwrap_or(&0);
    assert_eq!(
        count_a, 0,
        "Case A must have produced ZERO .ompf files on disk; got {count_a} \
         (files={files:?}) — the gate must skip a fresh layer below both floors"
    );
    assert!(
        count_b >= 1,
        "Case B must have produced at least one .ompf file on disk; got {count_b} \
         (files={files:?}) — quiesce-elapsed layer must be snapshotted via the \
         early-write trigger (Task 9j AC)"
    );
    assert!(
        count_c >= 1,
        "Case C must have produced at least one .ompf file on disk; got {count_c} \
         (files={files:?}) — a layer whose last checkpoint is older than the interval \
         must be snapshotted even while it is being actively written (b.yut)"
    );

    // Final skipped counter must reflect at least one skip for Case A.
    // (Cases B and C write, so they bump `success`, not `skipped`.)
    let skipped_final = read_writes(&metrics, "skipped");
    assert!(
        skipped_final >= 1,
        "result=\"skipped\" must accumulate >= 1 increment for Case A's fresh-layer \
         skip; got {skipped_final} — fewer would mean Case A was never observed in \
         the per-layer loop"
    );

    // Final success counter must be >= 2 — one for Case B (quiesce early
    // trigger), one for Case C (interval upper bound on a hot layer).
    let success_final = read_writes(&metrics, "success");
    assert!(
        success_final >= 2,
        "result=\"success\" must be >= 2 — Case B (quiesce-elapsed) and Case C \
         (interval-elapsed while under write) should each have produced a success \
         increment; got {success_final}"
    );
}
