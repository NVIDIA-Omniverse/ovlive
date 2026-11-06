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

//! From the hold's write bound on, nothing writes the store or the backup
//! store (SDD SR-6.2; Task 3's acceptance criterion, on which the restore
//! Job's quiet period rests).
//!
//! | Test | What it pins |
//! |---|---|
//! | [`from_the_hold_write_bound_on_nothing_writes_the_store_or_the_backup_store`] | with every inventory writer busy before entry, the store and the backup store are unchanged over the window that starts at the hold's write bound |
//! | [`a_cleanup_pass_in_flight_at_entry_starts_nothing_after_it_and_ends_aborted`] | a checkpoint clean-up pass parked in a backup-store delete at entry: the task ends aborted at entry, no backup-store call starts after entry, the parked delete is the only one that lands, and nothing writes the store from entry + the command timeout on |
//!
//! # The whole replica
//!
//! The replica here is composed in process from the production spawn
//! functions, each registered on a `HoldWriterAborts` the way `main.rs`
//! registers it (its handle, and its cancel token, under its inventory
//! entry): the checkpoint loop over a LocalDisk backup store, the
//! external-value mirror workers with a backlog the backup store drains
//! slowly, the external-value backup repair at a 2 s pass with no age gate
//! (so it re-sends and confirms the backlog the aborted mirror leaves), the
//! checkpoint clean-up at a 1 s interval (an `le:gc:leader` acquire and a
//! pass every second), the external-value GC loop with its `le:gc:leader`
//! refresh ticker, the orphan sweep, the reference-count reconciler at a 1 s
//! interval and the presence sweeper. The layer store's cleanup tracker and
//! an on-access fence token are registered too, with no work behind them.
//! The busy writers are shown busy before entry — the three leases held and
//! refreshed, a checkpoint written, mirror `PUT`s landing, the repair
//! handling values, the clean-up ticking — then the marker is written, the
//! real hold poll reads it, and the hold reactor runs the entry routine.
//!
//! From entry + `grpc::hold::hold_write_bound(D, T)` — the RPC deadline
//! [`RPC_DEADLINE`] plus the test client's command timeout T plus fred's
//! 45 s write wait, D + T + 45 s (51 s at the fixture's 5 s command
//! timeout), the window the criterion names — the
//! store's keyspace under the test's prefix and the backup store's files are
//! snapshotted, and again [`QUIET_WINDOW`] later. Between the two, no key may
//! appear or change value, and no TTL may be extended — a lease refresh is a
//! write. A key may vanish only by TTL expiry: it had a TTL at the first
//! snapshot. The backup store's files must be identical. The three leases
//! held before entry must already be gone at the first snapshot: they lapse
//! within the write bound.
//!
//! **What this does not cover.** `main.rs`'s own registration calls are not
//! exercised: the binary runs only in production. The test's registration
//! mirrors them, `tests/misc/hold_wiring_scan.rs` pins that `main.rs`
//! registers every inventory entry, a task or token for every writer it
//! spawns (the backup repair's among them) and the task handle of each writer
//! whose token does not stop a pass in flight (the clean-up's among them),
//! the inventory itself is pinned in `hold_reactor_test.rs`, and at runtime
//! the serving point logs an ERROR naming any entry left unregistered. Idle
//! here, and covered elsewhere:
//!
//! - the session-checkpoint dispatch and base bootstraps (recorded as not
//!   spawned; no lifecycle event follows entry): the tracker tests in
//!   `src/checkpoint/session_writer.rs` and
//!   `tests/integration/checkpoint/first_base_bootstrap_test.rs`;
//! - the layer store's tracked cleanups: `src/storage/layer.rs`'s tests;
//! - the on-access repairs and restores (a fresh token, wired to nothing
//!   here): `hold_fence_test.rs` and `hold_fence_sites_test.rs`;
//! - the orphan sweep, which runs one pass at start and then every 5
//!   minutes, so left running it would write nothing inside the window: the
//!   entry routine's enumeration test;
//! - a GC pass in flight at entry, which this composition does not stage:
//!   `hold_writer_abort_test.rs`;
//! - a clean-up pass in flight at entry, which this composition does not
//!   stage either (its backup store holds nothing for the clean-up to
//!   delete): the second test here.
//!
//! # A clean-up pass in flight at entry
//!
//! The second test runs the real clean-up task alone, registered as
//! `main.rs` registers it, with a pass parked in its first backup-store
//! delete when the real marker, poll and reactor enter the hold. The parked
//! delete runs on a task of its own, as a LocalDisk delete already handed to
//! the blocking pool does, so the hold's abort drops the pass but not that
//! call: it is the one in-flight step the bound in `src/dr/hold/writers.rs`
//! lets land after entry. A pass the abort did not stop would run its
//! post-delete step and its settle once the delete returns (store writes
//! after the command timeout), and one that also read no cancellation would
//! start the next delete (a backup-store call after entry).
//!
//! **Not covered there:** a Valkey command already handed to fred at entry
//! (bounded by the command timeout; the window starts after it) and a
//! StorageApi request already sent (the backend's, as for every writer).
//!
//! # Waits
//!
//! Setup waits are deadlines ([`SETTLE`], 10 s, for the mirror `PUT`s, the
//! backup repair's first handled value, the three leases and the clean-up's
//! first tick together, and the entry; the checkpoint wait clamps to its
//! helper's 30 s floor; the startup hold read is bounded by the fixture's 5 s
//! command timeout). The deliberate pauses are the write bound (51 s at the
//! fixture's 5 s command timeout) and [`QUIET_WINDOW`]. Every wait at its
//! full budget sums to about 156 s, past nextest's 120 s hang-killer, so
//! `.config/nextest.toml` gives this test its own 180 s deadline; a healthy
//! run takes about 85 s.
//!
//! The in-flight test's waits are deadlines too — the pass reaching its
//! delete (10 s, the park's entry budget), the startup hold read (5 s, the
//! command timeout), the entry, the task's end and the parked delete's
//! landing ([`SETTLE`] each) — and its deliberate pauses are the command
//! timeout (5 s) and [`IN_FLIGHT_WINDOW`] (5 s): about 55 s at full budget,
//! inside the 120 s hang-killer; a healthy run takes about 10 s.

use std::collections::{BTreeMap, HashSet};
use std::num::NonZeroU64;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use fred::interfaces::{ClientLike, KeysInterface, SortedSetsInterface};
use futures::future::BoxFuture;
use tokio_util::sync::CancellationToken;

use live_edit::checkpoint::adapters::{InMemoryCheckpointStore, LocalDiskCheckpointStore};
use live_edit::checkpoint::{
    ext_mirror_channel, spawn_checkpoint_cleanup, spawn_ext_backup_repair,
    spawn_ext_mirror_workers, CheckpointBlob, CheckpointError, CheckpointHandle,
    CheckpointLoopConfig, CheckpointMeta, CheckpointStore, CollectHeal, ExtBackupRepair,
    ExtBackupRepairTiming, ExtMirrorReceiver, ExtMirrorSender, LayerArtifact, LiveExtSet,
    TailCaptureConfig,
};
use live_edit::dr::hold::{
    evaluate_hold_at_startup, spawn_hold_poll, HoldObservation, HoldPollSettings, HoldWriter,
    HOLD_WRITERS,
};
use live_edit::grpc::graceful_shutdown::GracefulShutdown;
use live_edit::grpc::hold::{
    hold_write_bound, spawn_hold_reactor, HoldEntryParts, HoldExit, HoldGate, HoldGateState,
    HoldReactorSettings, HoldServingPoint, HoldWriterAborts,
};
use live_edit::observability::metrics::SharedMetrics;
use live_edit::presence::{spawn_presence_sweeper, SweeperBudget};
use live_edit::storage::ext_backup::ExtBackupMarks;
use live_edit::storage::ext_reconcile::{ext_reconcile_grace, spawn_ext_ref_reconcile};
use live_edit::storage::external_values::{spawn_gc_task, GcPolicy, ValkeyExternalValueStore};
use live_edit::storage::leader_lease::LeaderIdentity;
use live_edit::storage::orphan_sweep::spawn_orphan_sweep;
use live_edit::storage::test_support::{HoldMarkerSeed, TestValkey};

use super::hold_support::{answering, Park, ParkedStore, COMMAND_TIMEOUT_MS};
use super::support::{keyspace, keyspace_writes, WriteRules};
use crate::checkpoint::helpers::{
    setup_checkpoint_harness_with_options, tails_off_options, wait_for_checkpoint, SERVICE_LABEL,
};
use crate::grpc::helpers::create_session_and_layer;

/// Bound on each setup wait. Unreachable on the healthy path.
const SETTLE: Duration = Duration::from_secs(10);

/// D, the RPC deadline the write bound is computed from. No RPC is in flight
/// across entry in this composition; the window still starts where the
/// criterion starts it, at `hold_write_bound(D, T)` after entry.
const RPC_DEADLINE: Duration = Duration::from_secs(1);

/// The window over which nothing may write.
const QUIET_WINDOW: Duration = Duration::from_secs(30);

/// How far a TTL must have fallen across [`QUIET_WINDOW`] for it not to have
/// been refreshed: half the window, so runner jitter in the window's length
/// cannot read as a refresh, while a refresh (which resets a 30 s lease to
/// 30 s) always does.
const TTL_MUST_FALL_BY_MS: i64 = (QUIET_WINDOW.as_millis() / 2) as i64;

/// The hold poll's interval: the marker is read within a fraction of a
/// second.
const POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Interval of the reconciler and the presence sweeper: each runs many times
/// in the window if left running.
const BUSY_INTERVAL: Duration = Duration::from_secs(1);

/// External values stored before entry, and how long the backup store takes
/// over each mirror `PUT`: a backlog of about 150 s, so a mirror worker left
/// running would still be landing `PUT`s at the window's end (about 81 s
/// after entry: the 51 s write bound, then the 30 s window).
const MIRROR_BACKLOG: usize = 600;
/// The mirror's byte budget: far above the backlog's charge (600 small values,
/// at most a KiB each), so the budget never drops a copy.
const MIRROR_BUDGET_BYTES: NonZeroU64 = NonZeroU64::new(64 * 1024 * 1024).unwrap();
const MIRROR_PUT_DELAY: Duration = Duration::from_millis(250);

/// The backup store identity the mirror workers and the backup repair
/// confirm copies under.
const BACKUP_STORE_ID: &str = "hold-quiescence-backup";

/// The backup repair's timing: a pass every 2 s that acts on every value at
/// once (no age gate), so a repair left running would keep re-sending the
/// values the aborted mirror never delivered and confirming them on their
/// meta through the window. Production runs a pass every 5 minutes; the
/// writes a pass makes are the same.
fn busy_repair_timing() -> ExtBackupRepairTiming {
    ExtBackupRepairTiming {
        interval: Duration::from_secs(2),
        age_gate: Duration::ZERO,
        audit_age: Duration::from_secs(24 * 60 * 60),
        audit_per_pass: 2_000,
        reserve_wait: Duration::from_millis(300),
        reserve_retry: Duration::from_millis(20),
        upload_ceiling: Duration::from_secs(5),
        work_budget: Duration::from_millis(1_500),
    }
}

/// The checkpoint clean-up's config. `main.rs` hands the clean-up a clone of
/// the checkpoint loop's config; this one has the harness loop's shape, tails
/// off, with `leader` and a pass every `reconcile_interval`.
fn cleanup_config(leader: LeaderIdentity, reconcile_interval: Duration) -> CheckpointLoopConfig {
    CheckpointLoopConfig {
        tick: Duration::from_secs(1),
        interval: Duration::ZERO,
        quiesce: Duration::ZERO,
        retain_generations: 2,
        counters_sync: Duration::from_secs(3600),
        reconcile_interval,
        leader,
        tail: TailCaptureConfig {
            enabled: false,
            ..TailCaptureConfig::default()
        },
    }
}

/// The layer the checkpoint loop checkpoints before entry.
const LAYER: &str = "quiescence-layer";

/// What a write is over [`QUIET_WINDOW`]: a TTL that fell by less than
/// [`TTL_MUST_FALL_BY_MS`] was refreshed, and a key with a TTL may have
/// expired.
const WINDOW_WRITES: WriteRules = WriteRules {
    ttl_must_fall_by_ms: TTL_MUST_FALL_BY_MS,
    ttl_keys_may_expire: true,
};

/// Every file under `root`, by relative path, with its bytes.
fn backup_files(root: &Path) -> BTreeMap<String, Vec<u8>> {
    fn walk(root: &Path, dir: &Path, out: &mut BTreeMap<String, Vec<u8>>) {
        for entry in std::fs::read_dir(dir).expect("read a backup-store directory") {
            let path = entry.expect("a directory entry").path();
            if path.is_dir() {
                walk(root, &path, out);
            } else {
                let rel = path
                    .strip_prefix(root)
                    .expect("under the root")
                    .to_string_lossy()
                    .into_owned();
                out.insert(rel, std::fs::read(&path).expect("read a backup-store file"));
            }
        }
    }
    let mut out = BTreeMap::new();
    walk(root, root, &mut out);
    out
}

/// An exit seam that does nothing: the test never clears the hold.
struct NoExit;

impl HoldExit for NoExit {
    fn exit(&self, _code: i32) -> BoxFuture<'static, ()> {
        Box::pin(async {})
    }
}

async fn pttl(tv: &TestValkey, key: &str) -> i64 {
    tv.client().pool.pttl(key).await.expect("PTTL must succeed")
}

/// With the busy writers of the inventory working before entry, the store's
/// keyspace and the backup store are unchanged over the 30 s window that
/// starts at hold entry + `hold_write_bound(D, T)`, apart from TTL expiry.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn from_the_hold_write_bound_on_nothing_writes_the_store_or_the_backup_store() {
    // ── The backup store: LocalDisk, its external-value PUTs slowed ──────────
    let backup_root = tempfile::tempdir().expect("a backup-store root");
    let backup = ParkedStore::with_ext_put_delay(
        Arc::new(
            LocalDiskCheckpointStore::new(backup_root.path().to_path_buf())
                .expect("a LocalDisk store on a fresh root"),
        ),
        MIRROR_PUT_DELAY,
    );

    // ── The replica: the harness's service and checkpoint loop, its external-
    // value store mirroring into the backup store ────────────────────────────
    // The mirror's channel needs the harness's metrics, so the factory builds
    // it and parks the receiver here for the workers spawned below. The byte
    // budget is far above the backlog's charge, so no copy is dropped
    // `budget_full` (asserted once the backlog is stored).
    // The backup repair shares the mirror's byte budget through a clone of
    // its sender, as `main.rs` hands it one; the factory parks that too.
    let mirror_rx_slot: Arc<std::sync::Mutex<Option<ExtMirrorReceiver>>> = Arc::default();
    let repair_tx_slot: Arc<std::sync::Mutex<Option<ExtMirrorSender>>> = Arc::default();
    let factory_rx_slot = Arc::clone(&mirror_rx_slot);
    let factory_repair_slot = Arc::clone(&repair_tx_slot);
    let mut opts = tails_off_options(1);
    opts.store = Some(Arc::clone(&backup) as Arc<dyn CheckpointStore>);
    opts.ext_value_store_factory = Some(Box::new(move |ctx| {
        let (mirror_tx, mirror_rx) = ext_mirror_channel(
            MIRROR_BACKLOG * 2,
            MIRROR_BUDGET_BYTES,
            Arc::clone(ctx.metrics),
        );
        *factory_rx_slot
            .lock()
            .expect("the receiver slot is never poisoned") = Some(mirror_rx);
        *factory_repair_slot
            .lock()
            .expect("the sender slot is never poisoned") = Some(mirror_tx.clone());
        Arc::new(
            ValkeyExternalValueStore::new(ctx.tv.client().clone(), ctx.tv.key_schema().clone())
                .with_metrics(Arc::clone(ctx.metrics))
                .with_ext_mirror_sender(mirror_tx),
        )
    }));
    let harness = setup_checkpoint_harness_with_options(opts).await;
    let mirror_rx = mirror_rx_slot
        .lock()
        .expect("the receiver slot is never poisoned")
        .take()
        .expect("the harness built the external-value store through the factory");
    let repair_tx = repair_tx_slot
        .lock()
        .expect("the sender slot is never poisoned")
        .take()
        .expect("the factory parked the repair's sender");
    let tv = &harness.tv;
    let client = tv.client().clone();
    let keys = tv.key_schema().clone();
    let metrics = Arc::clone(&harness.metrics);
    let leader = crate::common::test_leader(SERVICE_LABEL);
    let ext_store = Arc::clone(&harness.svc.ext_value_store);

    // ── Every writer of the inventory, spawned and registered as main.rs does ─
    let writers = Arc::new(HoldWriterAborts::new());
    let gc_cancel = CancellationToken::new();
    let gc = spawn_gc_task(
        Arc::clone(&ext_store),
        client.clone(),
        keys.clone(),
        leader.clone(),
        GcPolicy::PRODUCTION,
        gc_cancel.clone(),
        Some(Arc::clone(&metrics)),
    );
    writers.register_task(HoldWriter::ExtGc, gc.abort_handle());
    writers.register_token(HoldWriter::ExtGc, gc_cancel.clone());
    let orphan_cancel = CancellationToken::new();
    let orphan = spawn_orphan_sweep(
        client.clone(),
        keys.clone(),
        Arc::clone(&ext_store),
        Arc::clone(&harness.svc.layer_store),
        leader.clone(),
        orphan_cancel.clone(),
        Some(Arc::clone(&metrics)),
    );
    writers.register_task(HoldWriter::OrphanSweep, orphan.abort_handle());
    writers.register_token(HoldWriter::OrphanSweep, orphan_cancel.clone());
    let reconcile_cancel = CancellationToken::new();
    let reconcile = spawn_ext_ref_reconcile(
        client.clone(),
        keys.clone(),
        leader.clone(),
        BUSY_INTERVAL,
        ext_reconcile_grace(RPC_DEADLINE.as_millis() as u64),
        reconcile_cancel.clone(),
        Some(Arc::clone(&metrics)),
    );
    writers.register_task(HoldWriter::ExtRefReconcile, reconcile.abort_handle());
    writers.register_token(HoldWriter::ExtRefReconcile, reconcile_cancel.clone());
    writers.register_task(
        HoldWriter::CheckpointLoop,
        harness.loop_handle.abort_handle(),
    );
    writers.register_token(HoldWriter::CheckpointLoop, harness.cancel.clone());
    writers.register_not_spawned(
        HoldWriter::SessionCheckpoints,
        "no lifecycle event follows entry here; the tracker abort is unit-tested",
    );
    let mirror_cancel = CancellationToken::new();
    // With the confirmation marks, as `main.rs` spawns them: each landed copy
    // is confirmed on the value's meta, a store write the hold must stop too.
    let mirror_workers = spawn_ext_mirror_workers(
        Arc::clone(&backup) as Arc<dyn CheckpointStore>,
        mirror_rx,
        1,
        Arc::clone(&metrics),
        Some(ExtBackupMarks::new(
            client.clone(),
            keys.clone(),
            BACKUP_STORE_ID.to_owned(),
            Some(Arc::clone(&metrics)),
        )),
        mirror_cancel.clone(),
    );
    for worker in &mirror_workers {
        writers.register_task(HoldWriter::ExtMirrorWorkers, worker.abort_handle());
    }
    writers.register_token(HoldWriter::ExtMirrorWorkers, mirror_cancel.clone());
    // The backup repair, spawned beside the workers as `main.rs` spawns it
    // and registered under its own entry (G1): it confirms copies on their
    // meta and re-sends the ones the backup store lacks.
    let cleanup_heal_tx = repair_tx.clone();
    let repair_cancel = CancellationToken::new();
    let repair = spawn_ext_backup_repair(
        ExtBackupRepair::new(
            client.clone(),
            keys.clone(),
            Arc::clone(&backup) as Arc<dyn CheckpointStore>,
            repair_tx,
            BACKUP_STORE_ID.to_owned(),
            Duration::from_secs(5),
            Some(Arc::clone(&metrics)),
        )
        .with_timing(busy_repair_timing()),
        leader.clone(),
        repair_cancel.clone(),
    );
    writers.register_task(HoldWriter::ExtBackupRepair, repair.abort_handle());
    writers.register_token(HoldWriter::ExtBackupRepair, repair_cancel.clone());
    // The checkpoint clean-up, spawned beside the repair as `main.rs` spawns
    // it — with the harness loop's leader, so `le:checkpoint:leader` carries
    // the current build's marked value its passes need, and a heal over the
    // mirror's byte budget confirming under the backup marks — and registered
    // like the checkpoint loop: its handle and its token. Left running, it
    // takes `le:gc:leader` and runs a pass every second.
    let cleanup_cancel = CancellationToken::new();
    let cleanup = spawn_checkpoint_cleanup(
        client.clone(),
        keys.clone(),
        Arc::clone(&backup) as Arc<dyn CheckpointStore>,
        cleanup_config(leader.clone(), BUSY_INTERVAL),
        Some(CollectHeal::new(
            cleanup_heal_tx,
            ExtBackupMarks::new(
                client.clone(),
                keys.clone(),
                BACKUP_STORE_ID.to_owned(),
                Some(Arc::clone(&metrics)),
            ),
        )),
        Some(Arc::clone(&metrics)),
        cleanup_cancel.clone(),
    );
    writers.register_task(HoldWriter::CheckpointCleanup, cleanup.abort_handle());
    writers.register_token(HoldWriter::CheckpointCleanup, cleanup_cancel.clone());
    writers.register_token(
        HoldWriter::LayerCleanups,
        harness.svc.layer_store.cleanup_abort_token(),
    );
    let background = CancellationToken::new();
    let sweeper = spawn_presence_sweeper(
        client.clone(),
        keys.clone(),
        leader.clone(),
        SweeperBudget::production(),
        Arc::clone(&metrics),
        BUSY_INTERVAL,
        background.clone(),
    );
    writers.register_task(HoldWriter::PresenceAndSessionTasks, sweeper.abort_handle());
    writers.register_token(HoldWriter::PresenceAndSessionTasks, background.clone());
    writers.register_token(HoldWriter::OnAccessRepairs, CancellationToken::new());
    assert!(
        writers.unregistered().is_empty(),
        "every inventory entry is registered or recorded as not spawned"
    );

    // ── Busy before entry ────────────────────────────────────────────────────
    let (_session_id, object_id) =
        create_session_and_layer(tv, &harness.svc, "quiescence", LAYER).await;
    wait_for_checkpoint(&harness.store, object_id, SETTLE.as_secs()).await;
    let blobs: Vec<([u8; 32], Vec<u8>)> = (0..MIRROR_BACKLOG)
        .map(|i| {
            let mut hash = [0x7Eu8; 32];
            hash[..8].copy_from_slice(&(i as u64).to_be_bytes());
            (hash, format!("mirrored value {i}").into_bytes())
        })
        .collect();
    let pairs: Vec<(&[u8; 32], &[u8])> = blobs.iter().map(|(h, d)| (h, d.as_slice())).collect();
    ext_store
        .store_values(&pairs)
        .await
        .expect("storing the backlog's values must succeed");
    let budget_full = live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(&metrics),
        "live_edit_checkpoint_ext_mirror_dropped_total",
        &[("reason", "budget_full")],
    )
    .unwrap_or(0.0);
    assert_eq!(
        budget_full, 0.0,
        "precondition: every backlog copy is queued; a budget drop would shrink the backlog \
         this test drains through hold entry"
    );
    let deadline = tokio::time::Instant::now() + SETTLE;
    while backup.ext_puts() < 2 {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the mirror workers did not land two PUTs within {SETTLE:?}"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    // The backup repair handling values: re-sending ones the mirror has not
    // reached yet, or confirming ones it has.
    let repair_handled = || {
        let text = live_edit::observability::metrics::test_util::render(&metrics);
        ["resent", "already_stored"]
            .iter()
            .map(|outcome| {
                live_edit::observability::metrics::test_util::sample_value(
                    &text,
                    "live_edit_checkpoint_ext_backup_repair_total",
                    &[("outcome", outcome)],
                )
                .unwrap_or(0.0)
            })
            .sum::<f64>()
    };
    let deadline = tokio::time::Instant::now() + SETTLE;
    while repair_handled() < 1.0 {
        assert!(
            tokio::time::Instant::now() < deadline,
            "precondition: the backup repair handled no value within {SETTLE:?}"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    // One deadline for the three leases and the clean-up's tick together.
    let deadline = tokio::time::Instant::now() + SETTLE;
    for (lease, what) in [
        (keys.gc_leader_lock(), "the GC loop's and the clean-up's"),
        (keys.checkpoint_leader_lock(), "the checkpoint loop's"),
        (
            keys.presence_sweeper_leader_lock(),
            "the presence sweeper's",
        ),
    ] {
        while pttl(tv, &lease).await <= 0 {
            assert!(
                tokio::time::Instant::now() < deadline,
                "precondition: {what} lease ({lease}) is never held"
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }
    // Only a clean-up tick publishes the external-value tombstone backlog.
    while live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(&metrics),
        "live_edit_checkpoint_ext_tombstone_backlog",
        &[],
    )
    .is_none()
    {
        assert!(
            tokio::time::Instant::now() < deadline,
            "precondition: the checkpoint clean-up never ran a tick"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    // ── Hold entry: the marker, the real poll, the reactor ───────────────────
    let initial: HoldObservation =
        evaluate_hold_at_startup(&client, &keys, answering(COMMAND_TIMEOUT_MS))
            .await
            .expect("the startup hold evaluation must answer");
    let poll_shutdown = CancellationToken::new();
    let (verdicts, _poll) = spawn_hold_poll(
        client.clone(),
        keys.clone(),
        HoldPollSettings::new(POLL_INTERVAL, COMMAND_TIMEOUT_MS),
        initial,
        poll_shutdown.clone(),
    );
    let point = HoldServingPoint::new();
    let _reactor = spawn_hold_reactor(
        verdicts.clone(),
        point.clone(),
        poll_shutdown.clone(),
        HoldReactorSettings {
            heartbeat_interval: Duration::from_secs(300),
            exit: Arc::new(NoExit),
            verdict_stale: tokio::sync::watch::Sender::new(false),
            held: tokio::sync::watch::Sender::new(false),
        },
    );
    let gate = HoldGate::new(verdicts);
    assert!(point.reach(HoldEntryParts {
        gate: gate.clone(),
        streams: harness.svc.shutdown.clone(),
        writers: Arc::clone(&writers),
        metrics: Arc::clone(&metrics),
    }));
    tv.write_hold_marker(&HoldMarkerSeed::default()).await;
    let deadline = tokio::time::Instant::now() + SETTLE;
    while gate.state() != HoldGateState::Held {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the hold was not entered within {SETTLE:?}"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let entered = tokio::time::Instant::now();
    let puts_at_entry = backup.ext_puts();
    assert!(
        puts_at_entry < MIRROR_BACKLOG,
        "precondition: the mirror backlog is still draining at entry"
    );

    // ── The window, from the write bound the criterion names ────────────────
    let command_timeout = client.pool.next().perf_config().default_command_timeout;
    tokio::time::sleep_until(entered + hold_write_bound(RPC_DEADLINE, command_timeout)).await;
    let store_before = keyspace(tv).await;
    let files_before = backup_files(backup_root.path());
    tokio::time::sleep(QUIET_WINDOW).await;
    let store_after = keyspace(tv).await;
    let files_after = backup_files(backup_root.path());

    // The three leases the busy writers held before entry have lapsed by the
    // write bound: their refreshers stopped at entry, and each lease's TTL
    // ran out inside it. No lease of a hold-capable writer outlives the
    // window a restore Job's quiet period covers.
    for (lease, what) in [
        (keys.gc_leader_lock(), "the GC loop's and the clean-up's"),
        (keys.checkpoint_leader_lock(), "the checkpoint loop's"),
        (
            keys.presence_sweeper_leader_lock(),
            "the presence sweeper's",
        ),
    ] {
        assert!(
            !store_before.contains_key(&lease),
            "{what} lease ({lease}) is still held at the hold's write bound: {:?}",
            store_before.get(&lease)
        );
    }
    let writes = keyspace_writes(&store_before, &store_after, WINDOW_WRITES);
    assert!(
        writes.is_empty(),
        "nothing may write the store from the hold's write bound on; over {QUIET_WINDOW:?}:\n{}",
        writes.join("\n")
    );
    let changed: Vec<&String> = files_before
        .keys()
        .chain(files_after.keys())
        .filter(|path| files_before.get(*path) != files_after.get(*path))
        .collect();
    assert!(
        changed.is_empty(),
        "nothing may write the backup store from the hold's write bound on; {} files \
         changed, the first: {:?} ({} mirror PUTs at entry, {} now)",
        changed.len(),
        changed.iter().take(5).collect::<Vec<_>>(),
        puts_at_entry,
        backup.ext_puts()
    );
    assert_eq!(gate.state(), HoldGateState::Held, "the replica stayed held");

    poll_shutdown.cancel();
}

// ═════════════════════════════════════════════════════════════════════════════
// A clean-up pass in flight at entry
// ═════════════════════════════════════════════════════════════════════════════

/// The in-flight test's clean-up interval: left running, the clean-up takes
/// `le:gc:leader` and starts a pass this often. Its pass budget is a quarter
/// of it, 500 ms, ample for an in-memory backup store.
const IN_FLIGHT_INTERVAL: Duration = Duration::from_secs(2);

/// How long the in-flight test watches after the parked delete is let
/// through: more than two clean-up intervals.
const IN_FLIGHT_WINDOW: Duration = Duration::from_secs(5);

/// What a write is over [`IN_FLIGHT_WINDOW`], by the rule of
/// [`WINDOW_WRITES`].
const IN_FLIGHT_WRITES: WriteRules = WriteRules {
    ttl_must_fall_by_ms: (IN_FLIGHT_WINDOW.as_millis() / 2) as i64,
    ttl_keys_may_expire: true,
};

/// The backup copies only aged tombstones name: the collect deletes them one
/// after another, and the first delete is the one parked at entry.
const COLLECTABLE: [&str; 4] = ["c1ea00", "c1ea01", "c1ea02", "c1ea03"];

/// The meta of a retained backup whose ext-refs sidecar names one value, none
/// of [`COLLECTABLE`]: the collect's live set is then not empty, so its own
/// listing licenses deletes.
fn retained_base_meta() -> CheckpointMeta {
    CheckpointMeta {
        creator_replica_id: "hold-cleanup-in-flight".to_owned(),
        written_at_unix_ms: 1,
        snap_seq: Some(1),
        session_id: "hold-cleanup-session".to_owned(),
        layer_name: "hold-cleanup-layer".to_owned(),
        sidecar_bytes: Some(Bytes::from_static(br#"{"ext_hashes":["0a1b2c"]}"#)),
        ..CheckpointMeta::default()
    }
}

/// A backup store over an in-memory one that records every call as it
/// starts, and runs its first `delete_ext_blob` the way a LocalDisk delete
/// already handed to the blocking pool runs: on a task of its own, which
/// waits on [`Self::park`] and then deletes. Aborting the caller drops only
/// the caller's wait; the delete lands once the park is released, and
/// [`Self::landed`] turns true.
struct InFlightStore {
    inner: Arc<InMemoryCheckpointStore>,
    park: Arc<Park>,
    calls: Mutex<Vec<String>>,
    parked_one: AtomicBool,
    landed: Arc<AtomicBool>,
}

impl InFlightStore {
    fn over(inner: Arc<InMemoryCheckpointStore>) -> Arc<Self> {
        Arc::new(Self {
            inner,
            park: Arc::new(Park::default()),
            calls: Mutex::new(Vec::new()),
            parked_one: AtomicBool::new(false),
            landed: Arc::new(AtomicBool::new(false)),
        })
    }

    fn record(&self, call: String) {
        self.calls.lock().unwrap().push(call);
    }

    /// Every call so far, in the order they started.
    fn calls(&self) -> Vec<String> {
        self.calls.lock().unwrap().clone()
    }
}

#[async_trait]
impl CheckpointStore for InFlightStore {
    async fn put_layer_checkpoint(
        &self,
        object_id: u64,
        snap_seq: u64,
        ompf: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        self.record(format!("put_layer_checkpoint {object_id}.{snap_seq}"));
        self.inner
            .put_layer_checkpoint(object_id, snap_seq, ompf, meta)
            .await
    }
    async fn get_latest_layer_checkpoint(
        &self,
        object_id: u64,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.record(format!("get_latest_layer_checkpoint {object_id}"));
        self.inner.get_latest_layer_checkpoint(object_id).await
    }
    async fn get_layer_checkpoint(
        &self,
        handle: &CheckpointHandle,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.record(format!("get_layer_checkpoint {}", handle.object_id));
        self.inner.get_layer_checkpoint(handle).await
    }
    async fn list_layer_checkpoints(
        &self,
        object_id: u64,
    ) -> Result<Vec<CheckpointHandle>, CheckpointError> {
        self.record(format!("list_layer_checkpoints {object_id}"));
        self.inner.list_layer_checkpoints(object_id).await
    }
    async fn delete_layer_checkpoint(
        &self,
        handle: CheckpointHandle,
    ) -> Result<(), CheckpointError> {
        self.record(format!("delete_layer_checkpoint {}", handle.object_id));
        self.inner.delete_layer_checkpoint(handle).await
    }
    async fn put_session_checkpoint(
        &self,
        session_id: &str,
        body: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        self.record(format!("put_session_checkpoint {session_id}"));
        self.inner
            .put_session_checkpoint(session_id, body, meta)
            .await
    }
    async fn get_session_checkpoint(
        &self,
        session_id: &str,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.record(format!("get_session_checkpoint {session_id}"));
        self.inner.get_session_checkpoint(session_id).await
    }
    async fn delete_session_checkpoint(&self, session_id: &str) -> Result<(), CheckpointError> {
        self.record(format!("delete_session_checkpoint {session_id}"));
        self.inner.delete_session_checkpoint(session_id).await
    }
    async fn list_session_checkpoint_ids(&self) -> Result<Vec<String>, CheckpointError> {
        self.record("list_session_checkpoint_ids".to_owned());
        self.inner.list_session_checkpoint_ids().await
    }
    async fn put_ext_blob(&self, hash_hex: &str, bytes: Bytes) -> Result<(), CheckpointError> {
        self.record(format!("put_ext_blob {hash_hex}"));
        self.inner.put_ext_blob(hash_hex, bytes).await
    }
    async fn get_ext_blob(&self, hash_hex: &str) -> Result<Option<Bytes>, CheckpointError> {
        self.record(format!("get_ext_blob {hash_hex}"));
        self.inner.get_ext_blob(hash_hex).await
    }
    async fn ext_blob_exists(&self, hash_hex: &str) -> Result<bool, CheckpointError> {
        self.record(format!("ext_blob_exists {hash_hex}"));
        self.inner.ext_blob_exists(hash_hex).await
    }
    async fn verify_list_route(&self) -> Result<(), CheckpointError> {
        self.record("verify_list_route".to_owned());
        self.inner.verify_list_route().await
    }
    async fn delete_ext_blob(&self, hash_hex: &str) -> Result<(), CheckpointError> {
        self.record(format!("delete_ext_blob {hash_hex}"));
        if self.parked_one.swap(true, Ordering::SeqCst) {
            return self.inner.delete_ext_blob(hash_hex).await;
        }
        let inner = Arc::clone(&self.inner);
        let park = Arc::clone(&self.park);
        let landed = Arc::clone(&self.landed);
        let hash = hash_hex.to_owned();
        tokio::spawn(async move {
            park.hold().await;
            let deleted = inner.delete_ext_blob(&hash).await;
            landed.store(true, Ordering::SeqCst);
            deleted
        })
        .await
        .expect("the in-flight delete does not panic")
    }
    async fn put_counters_sidecar(&self, body: Bytes) -> Result<(), CheckpointError> {
        self.record("put_counters_sidecar".to_owned());
        self.inner.put_counters_sidecar(body).await
    }
    async fn get_counters_sidecar(&self) -> Result<Option<Bytes>, CheckpointError> {
        self.record("get_counters_sidecar".to_owned());
        self.inner.get_counters_sidecar().await
    }
    async fn list_all_object_ids(&self) -> Result<Vec<u64>, CheckpointError> {
        self.record("list_all_object_ids".to_owned());
        self.inner.list_all_object_ids().await
    }
    async fn list_referenced_ext_hashes(&self) -> Result<HashSet<String>, CheckpointError> {
        self.record("list_referenced_ext_hashes".to_owned());
        self.inner.list_referenced_ext_hashes().await
    }
    async fn list_live_ext_set(&self) -> Result<LiveExtSet, CheckpointError> {
        self.record("list_live_ext_set".to_owned());
        self.inner.list_live_ext_set().await
    }
    async fn read_ext_refs_sidecar(
        &self,
        handle: &CheckpointHandle,
    ) -> Result<HashSet<String>, CheckpointError> {
        self.record(format!("read_ext_refs_sidecar {}", handle.object_id));
        self.inner.read_ext_refs_sidecar(handle).await
    }
    async fn list_layer_artifacts(&self) -> Result<Vec<LayerArtifact>, CheckpointError> {
        self.record("list_layer_artifacts".to_owned());
        self.inner.list_layer_artifacts().await
    }
    async fn delete_layer_temp_artifacts(
        &self,
        object_id: u64,
        snap_seq: u64,
    ) -> Result<usize, CheckpointError> {
        self.record(format!(
            "delete_layer_temp_artifacts {object_id}.{snap_seq}"
        ));
        self.inner
            .delete_layer_temp_artifacts(object_id, snap_seq)
            .await
    }
    fn publish_worst_case(&self) -> Duration {
        self.inner.publish_worst_case()
    }
}

/// A checkpoint clean-up pass in flight at hold entry stops at the step it
/// is on (SDD SR-6.2, and the clean-up's bound in `src/dr/hold/writers.rs`).
///
/// The real clean-up task, registered as `main.rs` registers it (its handle
/// and its token under `HoldWriter::CheckpointCleanup`), runs a pass whose
/// collect deletes the four backup copies only aged tombstones name; the
/// first delete parks (see [`InFlightStore`]). With the pass parked, the real
/// marker, poll and reactor enter the hold. Then:
///
/// - the task ends at entry, aborted, while its pass is still parked: not at
///   its next interval, and not by its token, which a pass reads only before
///   a delete;
/// - from entry + the command timeout, with the parked delete let through,
///   nothing writes the store over [`IN_FLIGHT_WINDOW`]: no post-delete
///   step, settle, re-date or `le:gc:leader` refresh;
/// - no backup-store call starts after entry, and of the four copies only
///   the one whose delete was in flight is gone.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_cleanup_pass_in_flight_at_entry_starts_nothing_after_it_and_ends_aborted() {
    let tv = TestValkey::setup().await;
    let client = tv.client().clone();
    let keys = tv.key_schema().clone();
    let metrics: SharedMetrics =
        Arc::new(crate::common::metrics_scrape_only("hold-cleanup-in-flight"));
    let leader = crate::common::test_leader("hold-cleanup-in-flight");

    // ── The backup store: a retained backup, and four copies only aged
    // tombstones name ─────────────────────────────────────────────────────────
    let inner = Arc::new(InMemoryCheckpointStore::new());
    inner
        .put_layer_checkpoint(1, 1, Bytes::from_static(b"base"), retained_base_meta())
        .await
        .expect("seeding a retained backup");
    for hash in COLLECTABLE {
        inner
            .put_ext_blob(hash, Bytes::from_static(b"copy"))
            .await
            .expect("seeding a backup copy");
    }
    let store = InFlightStore::over(Arc::clone(&inner));
    let _: () = client
        .pool
        .zadd(
            keys.checkpoint_reconcile_tombstones(),
            None,
            None,
            false,
            false,
            COLLECTABLE.iter().map(|h| (1.0, *h)).collect::<Vec<_>>(),
        )
        .await
        .expect("seeding aged tombstones");
    // `le:checkpoint:leader` carrying this build's marked value, as the
    // checkpoint loop's acquire leaves it: the clean-up's gate lets the pass
    // run.
    let _: () = client
        .pool
        .set(
            keys.checkpoint_leader_lock(),
            leader.checkpoint_lock_value(),
            Some(fred::types::Expiration::PX(60_000)),
            None,
            false,
        )
        .await
        .expect("SET le:checkpoint:leader");

    // ── The clean-up, spawned and registered as main.rs does ────────────────
    let (mirror_tx, _mirror_rx) = ext_mirror_channel(16, MIRROR_BUDGET_BYTES, Arc::clone(&metrics));
    let cancel = CancellationToken::new();
    let cleanup = spawn_checkpoint_cleanup(
        client.clone(),
        keys.clone(),
        Arc::clone(&store) as Arc<dyn CheckpointStore>,
        cleanup_config(leader.clone(), IN_FLIGHT_INTERVAL),
        Some(CollectHeal::new(
            mirror_tx,
            ExtBackupMarks::new(
                client.clone(),
                keys.clone(),
                BACKUP_STORE_ID.to_owned(),
                Some(Arc::clone(&metrics)),
            ),
        )),
        Some(Arc::clone(&metrics)),
        cancel.clone(),
    );
    let writers = Arc::new(HoldWriterAborts::new());
    writers.register_task(HoldWriter::CheckpointCleanup, cleanup.abort_handle());
    writers.register_token(HoldWriter::CheckpointCleanup, cancel.clone());
    for writer in HOLD_WRITERS {
        if writer != HoldWriter::CheckpointCleanup {
            writers.register_not_spawned(writer, "only the checkpoint clean-up runs here");
        }
    }
    assert!(
        writers.unregistered().is_empty(),
        "every inventory entry is registered or recorded as not spawned"
    );

    // ── A pass in flight: parked in its first backup-store delete ──────────
    store
        .park
        .entered("the clean-up pass's first backup-store delete")
        .await;

    // ── Hold entry: the marker, the real poll, the reactor ──────────────────
    let initial: HoldObservation =
        evaluate_hold_at_startup(&client, &keys, answering(COMMAND_TIMEOUT_MS))
            .await
            .expect("the startup hold evaluation must answer");
    let poll_shutdown = CancellationToken::new();
    let (verdicts, _poll) = spawn_hold_poll(
        client.clone(),
        keys.clone(),
        HoldPollSettings::new(POLL_INTERVAL, COMMAND_TIMEOUT_MS),
        initial,
        poll_shutdown.clone(),
    );
    let point = HoldServingPoint::new();
    let _reactor = spawn_hold_reactor(
        verdicts.clone(),
        point.clone(),
        poll_shutdown.clone(),
        HoldReactorSettings {
            heartbeat_interval: Duration::from_secs(300),
            exit: Arc::new(NoExit),
            verdict_stale: tokio::sync::watch::Sender::new(false),
            held: tokio::sync::watch::Sender::new(false),
        },
    );
    let gate = HoldGate::new(verdicts);
    assert!(point.reach(HoldEntryParts {
        gate: gate.clone(),
        streams: GracefulShutdown::new(),
        writers: Arc::clone(&writers),
        metrics: Arc::clone(&metrics),
    }));
    tv.write_hold_marker(&HoldMarkerSeed::default()).await;
    let deadline = tokio::time::Instant::now() + SETTLE;
    while !writers.is_aborted() {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the hold was not entered within {SETTLE:?}"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let entered = tokio::time::Instant::now();
    let calls_at_entry = store.calls();
    let in_flight: Vec<&str> = calls_at_entry
        .iter()
        .filter_map(|c| c.strip_prefix("delete_ext_blob "))
        .collect();
    let [parked] = in_flight.as_slice() else {
        panic!("precondition: one backup-store delete in flight at entry, got {in_flight:?}");
    };
    let parked = (*parked).to_owned();

    // The task ends at entry, aborted, its pass still parked.
    let deadline = tokio::time::Instant::now() + SETTLE;
    while !cleanup.is_finished() && tokio::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(
        cleanup.is_finished(),
        "the clean-up task is still running {SETTLE:?} after hold entry, with its pass parked \
         mid-delete: the hold did not stop it"
    );
    let ended = cleanup.await;
    assert!(
        ended
            .expect_err("the clean-up task ends only by the hold's abort here")
            .is_cancelled(),
        "the clean-up task ends aborted"
    );

    // ── From entry + the command timeout, the parked delete let through ─────
    let command_timeout = client.pool.next().perf_config().default_command_timeout;
    tokio::time::sleep_until(entered + command_timeout).await;
    let store_before = keyspace(&tv).await;
    store.park.release();
    let deadline = tokio::time::Instant::now() + SETTLE;
    while !store.landed.load(Ordering::SeqCst) {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the delete in flight at entry did not land within {SETTLE:?} of its release"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    tokio::time::sleep(IN_FLIGHT_WINDOW).await;
    let store_after = keyspace(&tv).await;

    let calls = store.calls();
    let started_after_entry = &calls[calls_at_entry.len()..];
    assert!(
        started_after_entry.is_empty(),
        "no backup-store call may start after hold entry; the clean-up started \
         {started_after_entry:?}"
    );
    for hash in COLLECTABLE {
        let held = inner
            .ext_blob_exists(hash)
            .await
            .expect("the in-memory store answers");
        assert_eq!(
            held,
            hash != parked,
            "only the copy whose delete was in flight at entry ({parked}) is gone; {hash} is \
             {}",
            if held { "still held" } else { "gone" }
        );
    }
    let writes = keyspace_writes(&store_before, &store_after, IN_FLIGHT_WRITES);
    assert!(
        writes.is_empty(),
        "nothing may write the store from hold entry + the command timeout ({command_timeout:?}) \
         on, with the delete in flight at entry let through; over {IN_FLIGHT_WINDOW:?}:\n{}",
        writes.join("\n")
    );
    assert_eq!(gate.state(), HoldGateState::Held, "the replica stayed held");

    poll_shutdown.cancel();
}
