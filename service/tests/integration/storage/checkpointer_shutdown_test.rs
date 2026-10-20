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

// Task t2.fgj.ya.9j Subtask .uy — graceful shutdown of the leader-elected
// checkpoint loop.
//
// AC: "Loop joins existing graceful-shutdown via `TaskTracker`" + the Task
// body's cancellation contract ("completes in-flight per-layer write …
// never leaves a half-written checkpoint generation"). We verify three
// shutdown invariants:
//
//   Case A — TaskTracker join.  When the loop is registered through a
//     `tokio_util::task::TaskTracker`, closing the tracker + firing the
//     cancellation token MUST drain the task within the
//     `LIVE_EDIT_SHUTDOWN_TIMEOUT_SECS` (5 s) budget — including the
//     pathological case where the loop is sleeping inside `timer.tick()`
//     on a long-tick configuration (the same shape `main.rs` ships).
//     This is the load-bearing AC: a regression that drops the cancellation
//     branch from the loop's `tokio::select!` would wedge tracker.wait().
//
//   Case B — mid-tick cancellation leaves no half-written generation.
//     Spawn the loop with a very short tick so it's actively writing
//     when cancellation fires, then assert the on-disk OMPF / sidecar /
//     meta triplets stay paired and `last_checkpoint_seq` on `obj:hdr`
//     never advances past a generation that's only partially on disk.
//     We loop this race ~10 times to catch the in-flight case. The
//     checkpointer's design (per `run_one_tick`'s docblock + `select!`
//     placement) is to check cancellation BEFORE the assembler call and
//     run the post-write HSET to completion afterward — we exercise both
//     halves of that contract.
//
//   Case C — shutdown bounded by the 5 s budget.  Even on a soak run that
//     stresses the in-flight path, every cancel + join must complete
//     inside the SDD §5.6 budget. We assert per-iteration latency in
//     Case B and a dedicated soak assert here.
//
// LocalDisk adapter chosen over InMemory for Case B because the on-disk
// OMPF + ext-refs.json + meta.json triplet is the exact failure surface
// the AC names ("OMPF/sidecar pairs"). InMemory would still detect
// `last_checkpoint_seq` drift but cannot prove the durable artifacts
// are consistent.

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use fred::interfaces::HashesInterface;
use tempfile::TempDir;
use tokio::time::timeout;
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;

use live_edit::checkpoint::adapters::{InMemoryCheckpointStore, LocalDiskCheckpointStore};
use live_edit::checkpoint::{
    spawn_checkpoint_loop, CheckpointLoopConfig, CheckpointStore, TailCaptureConfig,
};
use live_edit::domain::types::ObjectState;
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::relay::RelayedUpdates;
use live_edit::storage::test_support::TestValkey;
use oodf_ffi::OwnedDelta;

/// The shutdown budget the SDD §5.6 sequence enforces in production
/// (`LIVE_EDIT_SHUTDOWN_TIMEOUT_SECS = 5`). All tests in this file MUST
/// stay strictly under this — if a regression pushes a join past 5 s, the
/// real shutdown sequence would also drop the task on the floor.
const SHUTDOWN_BUDGET: Duration = Duration::from_secs(5);

fn mock_ext_store() -> Arc<MockExternalValueStore> {
    Arc::new(MockExternalValueStore::new())
}

fn layer_store(tv: &TestValkey) -> LayerStore {
    LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
}

fn session_store(tv: &TestValkey) -> SessionStore {
    SessionStore::new(tv.client().clone(), tv.key_schema().clone())
}

async fn create_test_session(tv: &TestValkey, label: &str) -> String {
    session_store(tv)
        .create(label, "/stage", "shutdown-test-user")
        .await
        .expect("session creation must succeed")
}

/// Seed a single populated layer the checkpoint loop can walk. Returns the
/// `object_id`. CreateLayer stamps `obj:hdr last_server_ts`, so the loop's
/// gate finds a commit timestamp on the first tick — paired with
/// `quiesce=0` + `interval=0` in our configs, every commit becomes
/// immediately due.
async fn seed_layer(tv: &TestValkey, session_id: &str, layer_name: &str) -> u64 {
    let initial_state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usd".to_string(),
        ..Default::default()
    };
    layer_store(tv)
        .create_layer(
            session_id,
            layer_name,
            "shutdown-test-user",
            &initial_state,
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed")
}

/// Drive a number of `update_layer` commits against `(session_id,
/// layer_name, oid)`, each advancing `obj:hdr seq` by one. Returns once
/// every commit has landed.
async fn drive_commits(ls: &LayerStore, session_id: &str, layer_name: &str, oid: u64, count: u64) {
    for i in 0..count {
        ls.update_layer_relayed(
            session_id,
            layer_name,
            oid,
            Some(OwnedDelta::default()),
            "shutdown-test-client",
            1_000 + i,
        )
        .await
        .expect("update_layer must succeed");
    }
}

/// =============================================================================
/// Case A — Clean shutdown joins TaskTracker within the 5 s budget.
/// =============================================================================
///
/// Long tick (10 s) keeps the loop sleeping inside `timer.tick()` when
/// cancellation fires. The `TaskTracker` wait MUST return within the
/// shutdown budget — if `spawn_checkpoint_loop` ever lost its
/// `tokio::select!` cancellation arm, this test's `timeout(...)` around
/// `tracker.wait()` would fire and surface the regression.
///
/// We adopt the production-side `TaskTracker::spawn` shape (used elsewhere
/// for WatchGuard UNWATCH and presence cleanup) by wrapping the
/// `JoinHandle<()>` returned from `spawn_checkpoint_loop` inside a
/// tracker-spawned future that awaits it. This mirrors how `main.rs`
/// composes background-task joins under a shared tracker while keeping
/// `spawn_checkpoint_loop`'s existing `JoinHandle` return type.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn shutdown_case_a_clean_cancellation_joins_tracker_within_budget() {
    let tv = TestValkey::setup().await;
    let keys = tv.key_schema().clone();

    let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());

    // 10 s tick — long enough that the loop is guaranteed to be sleeping
    // inside `timer.tick()` at cancellation time. The select arm on
    // `cancel.cancelled()` is the only path out; if it ever regressed,
    // the join would block on the next tick (well past the 5 s budget).
    let config = CheckpointLoopConfig {
        tick: Duration::from_secs(10),
        interval: Duration::from_secs(60),
        quiesce: Duration::from_secs(0),
        retain_generations: 2,
        // Epic 6 counters-sync cadence; pushed well above the test runtime
        // so the counters-sidecar timer does not fire during this test.
        counters_sync: Duration::from_secs(3600),
        reconcile_interval: Duration::from_secs(3600),
        leader: crate::common::test_leader("case-a-replica"),
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
        keys,
        Arc::clone(&store),
        config,
        None,
        cancel.clone(),
    );

    // Register the loop's `JoinHandle` through a fresh `TaskTracker`. The
    // tracker-spawned future awaits the loop's join; `tracker.wait()`
    // returns once every tracked future completes — i.e. once the loop
    // task has actually exited. This is the production-equivalent shape:
    // background tasks register their work through a shared tracker so
    // shutdown can wait on every in-flight task before tearing down Valkey
    // connections.
    //
    // `loop_joined` is the regression-guard for the AC's "removing
    // `TaskTracker::spawn` registration times out Case A" clause. The
    // spawned future flips this to `true` ONLY after the inner JoinHandle
    // has resolved (i.e. the loop task actually exited). If a regression
    // dropped the `tracker.spawn(...)` call, `tracker.wait()` would still
    // return immediately (empty tracker), but `loop_joined` would stay
    // `false` and the post-wait assert would fire. This is the strict
    // form of the AC: every tracker join must observably gate on the
    // loop's exit, not just trivially return because nothing was tracked.
    let loop_joined = Arc::new(AtomicBool::new(false));
    let loop_joined_for_task = Arc::clone(&loop_joined);
    let tracker = TaskTracker::new();
    tracker.spawn(async move {
        handle
            .await
            .expect("checkpoint loop must not panic on clean cancellation");
        loop_joined_for_task.store(true, Ordering::SeqCst);
    });
    tracker.close();

    // Give the loop a moment to actually park inside `timer.tick()`. A
    // freshly spawned task may not have reached the select yet, which
    // would let `cancel.cancel()` race with the first tick path — still
    // correct (the cancel happens-before-or-after the loop body runs)
    // but reduces the "loop is asleep" coverage of this test.
    tokio::time::sleep(Duration::from_millis(50)).await;

    let started = Instant::now();
    cancel.cancel();

    timeout(SHUTDOWN_BUDGET, tracker.wait()).await.expect(
        "checkpoint loop MUST drain its TaskTracker within the 5 s SDD shutdown budget; \
             a regression here means a hung tracker.wait() in main.rs's shutdown sequence",
    );

    let elapsed = started.elapsed();
    assert!(
        elapsed < SHUTDOWN_BUDGET,
        "TaskTracker drain took {elapsed:?}, exceeds the {SHUTDOWN_BUDGET:?} budget",
    );
    assert!(
        loop_joined.load(Ordering::SeqCst),
        "tracker.wait() returned but the spawned future never observed the loop's \
         JoinHandle resolving — this would happen if a regression removed the \
         `tracker.spawn(...)` call (tracker.wait() then returns immediately \
         because nothing is tracked) or if the inner await never completed",
    );
}

/// =============================================================================
/// Case B — Mid-tick cancellation leaves no half-written checkpoint
///          generation on disk.
/// =============================================================================
///
/// Spawn with a very short tick so the loop is actively doing per-layer
/// work when we cancel. After each iteration we assert:
///
///   1. For every `<oid>.<snap_seq>.ompf` on disk under the LocalDisk root,
///      the matching `<oid>.<snap_seq>.meta.json` is also present (the
///      LocalDisk adapter writes them as a pair in `put_layer_checkpoint`).
///      If a sidecar were missing, the apply path would observe a
///      half-written generation it cannot consistently reconstruct.
///
///   2. The `last_checkpoint_seq` on `obj:hdr` (when present) refers to a
///      `snap_seq` that exists on disk — i.e. the per-tick HSET that
///      advances the cursor never lands ahead of a durable OMPF.
///      Equivalently: if the cursor says "we checkpointed up through
///      snap_seq=N", an OMPF for that N MUST exist.
///
/// Looped 10× to maximise the chance of catching the in-flight race;
/// each iteration uses a fresh TestValkey + tempdir so prior state can't
/// mask a regression.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn shutdown_case_b_mid_tick_cancellation_leaves_no_orphan_artifacts() {
    for iteration in 0..10 {
        let tv = TestValkey::setup_with_pool_size(8).await;
        let keys = tv.key_schema().clone();
        let ls = layer_store(&tv);
        let session_id = create_test_session(&tv, "case-b-session").await;
        let layer_name = format!("case-b-layer-{iteration}");

        // Seed a layer with several pending commits so the per-tick work
        // has multiple put paths to potentially be racing on at the
        // cancel moment.
        let oid = seed_layer(&tv, &session_id, &layer_name).await;
        drive_commits(&ls, &session_id, &layer_name, oid, 5).await;

        // LocalDisk adapter so we can inspect the durable triplet
        // (`.ompf`, `.ext-refs.json`, `.meta.json`) directly. Tempdir
        // isolation per iteration keeps state from leaking between race
        // attempts.
        let tempdir = TempDir::new().expect("tempdir must be creatable");
        let store: Arc<dyn CheckpointStore> = Arc::new(
            LocalDiskCheckpointStore::new(tempdir.path().to_path_buf())
                .expect("LocalDiskCheckpointStore::new must succeed against a fresh tempdir"),
        );

        // Short tick (10 ms) so the loop is actively assembling when we
        // cancel; `interval = 0`, `quiesce = 0` keeps the skip-if-idle
        // gates from suppressing per-tick work. `retain_generations = 4`
        // is loose enough that the prune isn't immediately churning, so
        // most of the per-tick cost is in the assembler + put path —
        // exactly where we want the cancellation to interleave.
        let config = CheckpointLoopConfig {
            tick: Duration::from_millis(10),
            interval: Duration::ZERO,
            quiesce: Duration::ZERO,
            retain_generations: 4,
            // Epic 6 counters-sync cadence; pushed well above the test runtime
            // so the counters-sidecar timer does not fire during this test.
            counters_sync: Duration::from_secs(3600),
            reconcile_interval: Duration::from_secs(3600),
            leader: crate::common::test_leader(&format!("case-b-replica-{iteration}")),
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
            keys.clone(),
            Arc::clone(&store),
            config,
            None,
            cancel.clone(),
        );

        // Variable per-iteration delay (1 ms .. 100 ms in 10 ms steps)
        // hits a spread of points across the per-tick assembly so we
        // sample the "just woke up", "mid-assembler", and
        // "post-put / about to HSET cursor" stages of the loop body. A
        // single fixed delay would only ever interrupt one phase.
        let pre_cancel_delay = Duration::from_millis(1 + (iteration as u64) * 10);
        tokio::time::sleep(pre_cancel_delay).await;

        let started = Instant::now();
        cancel.cancel();

        timeout(SHUTDOWN_BUDGET, handle)
            .await
            .unwrap_or_else(|_| {
                panic!(
                    "iteration {iteration}: checkpoint loop must join within {SHUTDOWN_BUDGET:?} \
                     of cancellation; pre_cancel_delay={pre_cancel_delay:?}"
                )
            })
            .expect("checkpoint loop task must not panic on cancellation");
        let join_elapsed = started.elapsed();
        assert!(
            join_elapsed < SHUTDOWN_BUDGET,
            "iteration {iteration}: cancel→join took {join_elapsed:?}, exceeds {SHUTDOWN_BUDGET:?} \
             budget (Case C is a stricter version of this assertion)",
        );

        // ── Assertion 1: every on-disk OMPF has its meta sidecar paired.
        let on_disk = scan_layer_artifacts(tempdir.path()).await;
        for snap_seq in &on_disk.ompf_snap_seqs {
            assert!(
                on_disk.meta_snap_seqs.contains(snap_seq),
                "iteration {iteration}: snap_seq {snap_seq} has an OMPF but no meta.json — \
                 half-written generation visible on disk after mid-tick cancellation. \
                 ompf={:?}, meta={:?}",
                on_disk.ompf_snap_seqs,
                on_disk.meta_snap_seqs,
            );
        }

        // Symmetric check: no orphan meta or ext-refs without an OMPF.
        // The LocalDisk adapter writes the OMPF FIRST (so the only legal
        // dangling state would be ompf-without-meta on a crash between
        // the two writes). A meta-without-ompf shouldn't be reachable
        // through `put_layer_checkpoint`, so this is a defense-in-depth
        // assertion that catches a future refactor swapping the write
        // order.
        for snap_seq in &on_disk.meta_snap_seqs {
            assert!(
                on_disk.ompf_snap_seqs.contains(snap_seq),
                "iteration {iteration}: snap_seq {snap_seq} has meta.json but no OMPF — \
                 LocalDisk write order must persist OMPF before meta.json. \
                 ompf={:?}, meta={:?}",
                on_disk.ompf_snap_seqs,
                on_disk.meta_snap_seqs,
            );
        }

        // ── Assertion 2: `last_checkpoint_seq` cursor never refers to a
        // missing generation. The loop's design is to HSET the cursor
        // AFTER the assembler put has succeeded — a cursor value of N
        // therefore implies an OMPF for snap_seq=N exists on disk.
        let header_key = keys.obj_header(&oid.to_string());
        let cursor_raw: Option<String> = tv
            .client()
            .pool
            .hget(&header_key, "last_checkpoint_seq")
            .await
            .expect("HGET last_checkpoint_seq must succeed");
        if let Some(raw) = cursor_raw {
            if let Ok(cursor) = raw.parse::<u64>() {
                if cursor > 0 {
                    assert!(
                        on_disk.ompf_snap_seqs.contains(&cursor),
                        "iteration {iteration}: obj:hdr last_checkpoint_seq={cursor} but no \
                         matching OMPF on disk — the post-put HSET landed ahead of a \
                         durable generation, which violates the loop's cursor-after-put \
                         invariant. on-disk snap_seqs={:?}",
                        on_disk.ompf_snap_seqs,
                    );
                }
            }
        }
    }
}

/// =============================================================================
/// Case C — Worst-case shutdown still bounded by the 5 s SDD budget.
/// =============================================================================
///
/// Stress the in-flight assembler by driving many commits then cancelling
/// while the loop is mid-burst. The join MUST complete inside the
/// shutdown budget — the leader-lock TTL (30 s) and the assembler's own
/// retry budget would otherwise dominate. This is a soak version of
/// Case B's per-iteration latency assertion: if every iteration in
/// Case B passed individually but the soak shape revealed a slow path
/// (e.g. a single tick's assembler taking > 5 s), this test surfaces it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn shutdown_case_c_worst_case_in_flight_assembler_joins_within_budget() {
    let tv = TestValkey::setup_with_pool_size(8).await;
    let keys = tv.key_schema().clone();
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv, "case-c-session").await;

    // Seed a handful of layers — more per-tick work means the
    // cancellation more reliably lands inside the SCAN/assemble loop.
    let mut oids = Vec::new();
    for i in 0..4 {
        let layer_name = format!("case-c-layer-{i}");
        let oid = seed_layer(&tv, &session_id, &layer_name).await;
        drive_commits(&ls, &session_id, &layer_name, oid, 3).await;
        oids.push(oid);
    }

    let tempdir = TempDir::new().expect("tempdir must be creatable");
    let store: Arc<dyn CheckpointStore> = Arc::new(
        LocalDiskCheckpointStore::new(tempdir.path().to_path_buf())
            .expect("LocalDiskCheckpointStore::new must succeed against a fresh tempdir"),
    );

    let config = CheckpointLoopConfig {
        tick: Duration::from_millis(5),
        interval: Duration::ZERO,
        quiesce: Duration::ZERO,
        retain_generations: 2,
        // Epic 6 counters-sync cadence; pushed well above the test runtime
        // so the counters-sidecar timer does not fire during this test.
        counters_sync: Duration::from_secs(3600),
        reconcile_interval: Duration::from_secs(3600),
        leader: crate::common::test_leader("case-c-replica"),
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
        keys,
        store,
        config,
        None,
        cancel.clone(),
    );

    // Let the loop chew on the seeded work for a few ticks so the
    // assembler is reliably in-flight when we cancel.
    tokio::time::sleep(Duration::from_millis(150)).await;

    let started = Instant::now();
    cancel.cancel();

    timeout(SHUTDOWN_BUDGET, handle)
        .await
        .expect(
            "Case C: worst-case in-flight checkpoint loop MUST join within the 5 s SDD \
             shutdown budget; a longer join would extend past terminationGracePeriodSeconds",
        )
        .expect("checkpoint loop task must not panic on cancellation");

    let elapsed = started.elapsed();
    assert!(
        elapsed < SHUTDOWN_BUDGET,
        "Case C: cancel→join took {elapsed:?}, exceeds {SHUTDOWN_BUDGET:?} budget",
    );
}

/// On-disk artifacts the LocalDisk adapter writes per checkpoint: the OMPF
/// blob, the optional ext-refs sidecar, and the meta JSON. Each value is
/// the set of `snap_seq`s observed for the test's single layer (Case B
/// uses one layer per iteration; the file-name parser pulls the snap_seq
/// from `<oid>.<snap_seq>.<ext>`).
struct OnDiskArtifacts {
    ompf_snap_seqs: HashSet<u64>,
    meta_snap_seqs: HashSet<u64>,
}

/// Walk `<root>/.live-edit/layers/` and bucket every file by extension and
/// `snap_seq`. Mirrors the LocalDisk adapter's filename grammar
/// (`<oid>.<snap_seq>.{ompf,ext-refs.json,meta.json}`). Temp files
/// (`*.tmp`) from in-flight atomic writes are deliberately ignored: the
/// adapter cleans them up on success and best-effort removes them on
/// error, so a leaked temp file is a low-severity garbage condition that
/// doesn't constitute a "half-written generation".
async fn scan_layer_artifacts(root: &std::path::Path) -> OnDiskArtifacts {
    let layers_dir = root.join(".live-edit").join("layers");
    let mut ompf = HashSet::new();
    let mut meta = HashSet::new();

    let mut reader = match tokio::fs::read_dir(&layers_dir).await {
        Ok(r) => r,
        // First-tick cancellation can fire before the adapter has
        // touched the layers dir at all — that's "nothing on disk yet",
        // which is the strongest possible no-half-write outcome.
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            return OnDiskArtifacts {
                ompf_snap_seqs: ompf,
                meta_snap_seqs: meta,
            };
        }
        Err(err) => panic!("read_dir {layers_dir:?} failed: {err}"),
    };
    while let Some(entry) = reader
        .next_entry()
        .await
        .expect("read_dir iteration must succeed")
    {
        let file_name = entry.file_name();
        let Some(name) = file_name.to_str() else {
            continue;
        };
        // Skip in-flight atomic-write tempfiles. The adapter uses a UUID
        // suffix so a temp lingering after cancellation is a separately-
        // tracked low-severity garbage condition, NOT a published
        // generation.
        if name.ends_with(".tmp") {
            continue;
        }
        if let Some(snap_seq) = parse_snap_seq_from_filename(name, ".ompf") {
            ompf.insert(snap_seq);
        } else if let Some(snap_seq) = parse_snap_seq_from_filename(name, ".meta.json") {
            meta.insert(snap_seq);
        }
        // ext-refs.json files exist in the same dir; we don't pair on
        // them in this test (the LocalDisk adapter only writes
        // ext-refs.json when `meta.sidecar_bytes` is `Some(_)`, and the
        // adapter writes it AFTER the OMPF but BEFORE the meta.json,
        // so a missing ext-refs alongside an OMPF + meta is also a
        // half-written generation — but the meta-vs-ompf pairing above
        // is the load-bearing one for the apply-path correctness the
        // AC names. A future stricter test could pair ext-refs too.).
    }
    OnDiskArtifacts {
        ompf_snap_seqs: ompf,
        meta_snap_seqs: meta,
    }
}

/// Parse `<oid>.<snap_seq><suffix>` -> Some(snap_seq). Returns None for
/// filenames that don't match the expected shape.
fn parse_snap_seq_from_filename(name: &str, suffix: &str) -> Option<u64> {
    let stem = name.strip_suffix(suffix)?;
    let dot = stem.rfind('.')?;
    let (_oid, snap_seq_str) = stem.split_at(dot);
    let snap_seq_str = snap_seq_str.strip_prefix('.')?;
    snap_seq_str.parse::<u64>().ok()
}
