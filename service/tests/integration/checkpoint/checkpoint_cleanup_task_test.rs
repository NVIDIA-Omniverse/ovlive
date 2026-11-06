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

//! b.g65 N1 — the checkpoint clean-up runs as its own task on `le:gc:leader`,
//! not as a timer of the checkpoint loop under `le:checkpoint:leader`, so
//! layer backups never wait for it.
//!
//! | Test | What it pins |
//! |---|---|
//! | [`layer_backups_and_the_counters_sync_run_while_a_cleanup_pass_is_parked`] | a clean-up pass parked in its first read does not stop the per-layer tick writing a backup or the counters tick writing its sidecar |
//! | [`the_checkpoint_loop_alone_runs_no_cleanup_pass`] | the loop has no clean-up arm: with only the loop running, an orphaned partial publish is never even registered |
//! | [`the_cleanup_task_exits_promptly_on_cancel_after_its_first_pass_starts`] | the task's cancel: once its first pass has registered a candidate, a cancel ends the task promptly, mid-pass or between passes |
//! | [`a_failed_redate_leaves_its_members_where_they_were_and_the_pass_goes_on`] | a re-date that Valkey refuses (D4) is not fatal: the batch's deletes still run, and the member stays at its old score |
//! | [`a_cancel_during_a_pass_starts_no_further_pass_though_the_interval_has_elapsed`] | the task's wait between passes is biased toward the cancel: a cancel that lands mid-pass starts no further pass, though the interval has elapsed too |
//!
//! Every wait is bounded by [`WAIT`] (10 s); the longest test,
//! `layer_backups_…_parked`, has four (three conditions and the join), 40 s,
//! inside nextest's 120 s hang-killer. The biased-cancel test's 32 trials
//! each wait at most one [`WAIT`] for the pass to reach its delete and 5 s for
//! the join, but a healthy trial takes well under a tenth of a second.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use fred::interfaces::{KeysInterface, SortedSetsInterface};
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;

use live_edit::checkpoint::adapters::{InMemoryCheckpointStore, LocalDiskCheckpointStore};
use live_edit::checkpoint::{
    CheckpointBlob, CheckpointError, CheckpointHandle, CheckpointLoopConfig, CheckpointMeta,
    CheckpointStore, LayerArtifact, LiveExtSet, TailCaptureConfig,
};
use live_edit::domain::types::ObjectState;
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;

use super::helpers::DelegatingCheckpointStore;
use crate::storage::snapshot_read_support::{
    client_through, error_reply_script, script_frames_naming, test_valkey_addr, ProxyAction,
    ScriptReplyProxy,
};

/// Bound on every wait for a background condition.
const WAIT: Duration = Duration::from_secs(10);

/// Poll interval inside [`WAIT`].
const POLL: Duration = Duration::from_millis(20);

/// A config whose loop ticks every 50 ms with every layer due, whose counters
/// sync runs every `counters_sync`, and whose clean-up runs once at start and
/// then not again within the test.
fn config(replica: &str, counters_sync: Duration) -> CheckpointLoopConfig {
    CheckpointLoopConfig {
        tick: Duration::from_millis(50),
        interval: Duration::ZERO,
        quiesce: Duration::ZERO,
        retain_generations: 2,
        counters_sync,
        reconcile_interval: Duration::from_secs(3600),
        leader: crate::common::test_leader(replica),
        tail: TailCaptureConfig {
            enabled: false,
            ..TailCaptureConfig::default()
        },
    }
}

/// `le:checkpoint:leader` held by `config.leader`'s marked value, as the
/// loop's first acquire would leave it: the clean-up gate then lets the first
/// pass run whatever the order the two tasks start in, and the loop
/// recognises the value as its own.
async fn plant_own_checkpoint_lock(tv: &TestValkey, config: &CheckpointLoopConfig) {
    let _: () = tv
        .client()
        .pool
        .set(
            tv.key_schema().checkpoint_leader_lock(),
            config.leader.checkpoint_lock_value(),
            Some(fred::types::Expiration::PX(30_000)),
            None,
            false,
        )
        .await
        .expect("SET le:checkpoint:leader");
}

async fn eventually<F, Fut>(what: &str, mut condition: F)
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = bool>,
{
    let deadline = tokio::time::Instant::now() + WAIT;
    loop {
        if condition().await {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "timed out after {WAIT:?} waiting for: {what}"
        );
        tokio::time::sleep(POLL).await;
    }
}

/// `<root>/.live-edit/layers`, the LocalDisk adapter's flat artifact
/// directory (spelled out, as `adapter_local_disk_test.rs` does: the adapter
/// keeps its layout constants private).
fn layers_dir(root: &std::path::Path) -> std::path::PathBuf {
    root.join(".live-edit").join("layers")
}

async fn create_layer(tv: &TestValkey, name: &str) -> u64 {
    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create(name, &format!("/{name}"), "cleanup-task-user")
        .await
        .expect("session creation must succeed");
    LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .create_layer(
            &session_id,
            name,
            "cleanup-task-user",
            &ObjectState {
                sequence_num: 1,
                first_version: 1,
                source_format: "usda".to_owned(),
                ..Default::default()
            },
            Arc::new(MockExternalValueStore::new()),
        )
        .await
        .expect("create_layer must succeed")
}

/// A store over `inner` that counts its counters-sidecar reads (every counters
/// tick reads the sidecar first) and, when
/// `park_first_enumeration`, parks its FIRST `list_layer_artifacts` — the
/// clean-up pass's orphan enumeration, the first read of every pass — until
/// the test opens [`Self::gate`]. Nothing in the loop's ticks calls
/// `list_layer_artifacts`.
///
/// With `park_ext_deletes`, it also holds every `delete_ext_blob` — the
/// collect's delete of a backup copy, a write no cancel abandons — until the
/// test adds a permit to [`Self::delete_gate`], counting the deletes that
/// reached it.
struct ParkingStore {
    inner: DelegatingCheckpointStore<dyn CheckpointStore>,
    park_first_enumeration: bool,
    parked: AtomicUsize,
    gate: Semaphore,
    counters_reads: AtomicUsize,
    park_ext_deletes: bool,
    ext_deletes_entered: AtomicUsize,
    delete_gate: Semaphore,
}

impl ParkingStore {
    fn over(inner: Arc<dyn CheckpointStore>, park_first_enumeration: bool) -> Arc<Self> {
        Arc::new(Self {
            inner: DelegatingCheckpointStore::new(inner),
            park_first_enumeration,
            parked: AtomicUsize::new(0),
            gate: Semaphore::new(0),
            counters_reads: AtomicUsize::new(0),
            park_ext_deletes: false,
            ext_deletes_entered: AtomicUsize::new(0),
            delete_gate: Semaphore::new(0),
        })
    }

    fn new() -> Arc<Self> {
        Self::over(Arc::new(InMemoryCheckpointStore::new()), true)
    }

    /// Over `inner`, holding every `delete_ext_blob` and nothing else.
    fn parking_ext_deletes(inner: Arc<dyn CheckpointStore>) -> Arc<Self> {
        Arc::new(Self {
            inner: DelegatingCheckpointStore::new(inner),
            park_first_enumeration: false,
            parked: AtomicUsize::new(0),
            gate: Semaphore::new(0),
            counters_reads: AtomicUsize::new(0),
            park_ext_deletes: true,
            ext_deletes_entered: AtomicUsize::new(0),
            delete_gate: Semaphore::new(0),
        })
    }
}

#[async_trait::async_trait]
impl CheckpointStore for ParkingStore {
    async fn list_layer_artifacts(&self) -> Result<Vec<LayerArtifact>, CheckpointError> {
        if self.parked.fetch_add(1, Ordering::SeqCst) == 0 && self.park_first_enumeration {
            let _open = self.gate.acquire().await.expect("the gate is never closed");
        }
        self.inner.list_layer_artifacts().await
    }

    // ── Verbatim forwarding ──────────────────────────────────────────────
    async fn put_layer_checkpoint(
        &self,
        object_id: u64,
        snap_seq: u64,
        ompf: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        self.inner
            .put_layer_checkpoint(object_id, snap_seq, ompf, meta)
            .await
    }
    async fn get_latest_layer_checkpoint(
        &self,
        object_id: u64,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.inner.get_latest_layer_checkpoint(object_id).await
    }
    async fn get_layer_checkpoint(
        &self,
        handle: &CheckpointHandle,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.inner.get_layer_checkpoint(handle).await
    }
    async fn list_layer_checkpoints(
        &self,
        object_id: u64,
    ) -> Result<Vec<CheckpointHandle>, CheckpointError> {
        self.inner.list_layer_checkpoints(object_id).await
    }
    async fn delete_layer_checkpoint(
        &self,
        handle: CheckpointHandle,
    ) -> Result<(), CheckpointError> {
        self.inner.delete_layer_checkpoint(handle).await
    }
    async fn put_session_checkpoint(
        &self,
        session_id: &str,
        body: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        self.inner
            .put_session_checkpoint(session_id, body, meta)
            .await
    }
    async fn get_session_checkpoint(
        &self,
        session_id: &str,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.inner.get_session_checkpoint(session_id).await
    }
    async fn delete_session_checkpoint(&self, session_id: &str) -> Result<(), CheckpointError> {
        self.inner.delete_session_checkpoint(session_id).await
    }
    async fn list_session_checkpoint_ids(&self) -> Result<Vec<String>, CheckpointError> {
        self.inner.list_session_checkpoint_ids().await
    }
    async fn put_ext_blob(&self, hash_hex: &str, bytes: Bytes) -> Result<(), CheckpointError> {
        self.inner.put_ext_blob(hash_hex, bytes).await
    }
    async fn get_ext_blob(&self, hash_hex: &str) -> Result<Option<Bytes>, CheckpointError> {
        self.inner.get_ext_blob(hash_hex).await
    }
    async fn ext_blob_exists(&self, hash_hex: &str) -> Result<bool, CheckpointError> {
        self.inner.ext_blob_exists(hash_hex).await
    }
    async fn delete_ext_blob(&self, hash_hex: &str) -> Result<(), CheckpointError> {
        if self.park_ext_deletes {
            self.ext_deletes_entered.fetch_add(1, Ordering::SeqCst);
            self.delete_gate
                .acquire()
                .await
                .expect("the gate is never closed")
                .forget();
        }
        self.inner.delete_ext_blob(hash_hex).await
    }
    async fn put_counters_sidecar(&self, body: Bytes) -> Result<(), CheckpointError> {
        self.inner.put_counters_sidecar(body).await
    }
    async fn get_counters_sidecar(&self) -> Result<Option<Bytes>, CheckpointError> {
        self.counters_reads.fetch_add(1, Ordering::SeqCst);
        self.inner.get_counters_sidecar().await
    }
    async fn list_all_object_ids(&self) -> Result<Vec<u64>, CheckpointError> {
        self.inner.list_all_object_ids().await
    }
    async fn list_referenced_ext_hashes(
        &self,
    ) -> Result<std::collections::HashSet<String>, CheckpointError> {
        self.inner.list_referenced_ext_hashes().await
    }
    async fn list_live_ext_set(&self) -> Result<LiveExtSet, CheckpointError> {
        self.inner.list_live_ext_set().await
    }
    async fn read_ext_refs_sidecar(
        &self,
        handle: &CheckpointHandle,
    ) -> Result<std::collections::HashSet<String>, CheckpointError> {
        self.inner.read_ext_refs_sidecar(handle).await
    }
    async fn delete_layer_temp_artifacts(
        &self,
        object_id: u64,
        snap_seq: u64,
    ) -> Result<usize, CheckpointError> {
        self.inner
            .delete_layer_temp_artifacts(object_id, snap_seq)
            .await
    }
    fn publish_worst_case(&self) -> Duration {
        self.inner.publish_worst_case()
    }
}

/// N1: with a clean-up pass parked in its first read — the orphan
/// enumeration, which on a large store can take minutes — the checkpoint
/// loop's per-layer tick still writes a new layer's backup and its counters
/// tick still writes the counters sidecar.
///
/// Before b.g65 the clean-up ran as the third timer of the loop's one
/// `select!`, under the same `le:checkpoint:leader` lease, so a parked pass
/// stopped every backup on every replica until it ended (about 9 minutes an
/// hour on a development deployment).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn layer_backups_and_the_counters_sync_run_while_a_cleanup_pass_is_parked() {
    let tv = TestValkey::setup_with_pool_size(8).await;
    let cfg = config("cleanup-task-n1", Duration::from_millis(200));
    plant_own_checkpoint_lock(&tv, &cfg).await;
    let parking = ParkingStore::new();
    let store: Arc<dyn CheckpointStore> = parking.clone();
    let cancel = CancellationToken::new();
    let handle = crate::common::spawn_checkpoint_loop_and_cleanup(
        tv.client().clone(),
        tv.key_schema().clone(),
        Arc::clone(&store),
        cfg,
        cancel.clone(),
    );
    eventually(
        "the clean-up pass parks in its orphan enumeration",
        || async { parking.parked.load(Ordering::SeqCst) >= 1 },
    )
    .await;

    let oid = create_layer(&tv, "cleanup-task-n1-layer").await;
    eventually("the loop writes the new layer's backup", || async {
        !store
            .list_layer_checkpoints(oid)
            .await
            .expect("list_layer_checkpoints")
            .is_empty()
    })
    .await;
    eventually("the counters tick writes its sidecar", || async {
        store
            .get_counters_sidecar()
            .await
            .expect("get_counters_sidecar")
            .is_some()
    })
    .await;
    assert_eq!(
        parking.parked.load(Ordering::SeqCst),
        1,
        "and the clean-up pass was still parked throughout"
    );

    parking.gate.add_permits(1);
    cancel.cancel();
    tokio::time::timeout(WAIT, handle)
        .await
        .expect("both tasks join once the pass is released and cancelled")
        .expect("neither task panics");
}

/// N1: the checkpoint loop has no clean-up arm. With only the loop running —
/// its per-layer and counters ticks — an orphaned partial publish on the store
/// is never registered as a candidate (nor listed at all) while the counters
/// tick runs three times. Before b.g65 the loop's third timer, here due every
/// 100 ms beside the 200 ms counters timer, ran a pass on every one of those
/// rounds and registered it at the first.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_checkpoint_loop_alone_runs_no_cleanup_pass() {
    let tv = TestValkey::setup_with_pool_size(8).await;
    let dir = tempfile::tempdir().expect("tempdir");
    let disk = LocalDiskCheckpointStore::new(dir.path().to_path_buf()).expect("LocalDisk store");
    let layers = layers_dir(dir.path());
    // A publish that died after its sidecar: an orphan candidate for any
    // clean-up pass.
    tokio::fs::write(layers.join("9.1.ext-refs.json"), br#"{"ext_hashes":[]}"#)
        .await
        .expect("planting the orphan sidecar");
    let observed = ParkingStore::over(Arc::new(disk), false);
    let store: Arc<dyn CheckpointStore> = observed.clone();
    let mut cfg = config("cleanup-task-loop-only", Duration::from_millis(200));
    cfg.reconcile_interval = Duration::from_millis(100);
    plant_own_checkpoint_lock(&tv, &cfg).await;
    create_layer(&tv, "cleanup-task-loop-only-layer").await;
    let cancel = CancellationToken::new();
    let handle = live_edit::checkpoint::spawn_checkpoint_loop(
        tv.client().clone(),
        tv.key_schema().clone(),
        Arc::clone(&store),
        cfg,
        None,
        cancel.clone(),
    );

    eventually("the loop's counters tick runs three times", || async {
        observed.counters_reads.load(Ordering::SeqCst) >= 3
    })
    .await;
    assert_eq!(
        observed.parked.load(Ordering::SeqCst),
        0,
        "the loop never enumerated the layers namespace, which only a clean-up pass does"
    );
    let registered: Vec<String> = tv
        .client()
        .pool
        .zrange(
            tv.key_schema().checkpoint_reconcile_orphans(),
            0,
            -1,
            None,
            false,
            None,
            false,
        )
        .await
        .expect("ZRANGE :orphans");
    cancel.cancel();
    tokio::time::timeout(WAIT, handle)
        .await
        .expect("the loop joins")
        .expect("the loop does not panic");

    assert!(
        registered.is_empty(),
        "the loop alone registered orphan candidates {registered:?}: it still runs a clean-up pass"
    );
}

/// The clean-up task's cancel: once its first pass (run at start) has
/// registered the store's orphan candidate, a cancel ends the task promptly —
/// it does not wait out its hour-long interval. The cancel can land while
/// that pass is still in its later phases (which check the cancel before each
/// member and batch) or after it, between passes; either way the task ends.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_cleanup_task_exits_promptly_on_cancel_after_its_first_pass_starts() {
    let tv = TestValkey::setup().await;
    let dir = tempfile::tempdir().expect("tempdir");
    let disk = LocalDiskCheckpointStore::new(dir.path().to_path_buf()).expect("LocalDisk store");
    let layers = layers_dir(dir.path());
    tokio::fs::write(layers.join("9.1.ext-refs.json"), br#"{"ext_hashes":[]}"#)
        .await
        .expect("planting the orphan sidecar");
    let store: Arc<dyn CheckpointStore> = Arc::new(disk);
    let cfg = config("cleanup-task-cancel", Duration::from_secs(3600));
    plant_own_checkpoint_lock(&tv, &cfg).await;
    let cancel = CancellationToken::new();
    let handle = live_edit::checkpoint::spawn_checkpoint_cleanup(
        tv.client().clone(),
        tv.key_schema().clone(),
        store,
        cfg,
        None,
        None,
        cancel.clone(),
    );
    let orphans = tv.key_schema().checkpoint_reconcile_orphans();
    eventually("the first pass registers the orphan candidate", || async {
        let n: u64 = tv.client().pool.zcard(&orphans).await.expect("ZCARD");
        n == 1
    })
    .await;

    cancel.cancel();
    tokio::time::timeout(Duration::from_secs(5), handle)
        .await
        .expect("the task ends at once on cancel, not after its hour-long interval")
        .expect("the task does not panic");
}

/// A metadata-only base generation for `object_id` whose ext-refs sidecar
/// names `hashes`: a retained backup, so the collect's live set is not empty.
fn base_naming(object_id: u64, hashes: &[&str]) -> (u64, u64, Bytes, CheckpointMeta) {
    let sidecar = serde_json::to_vec(&serde_json::json!({ "ext_hashes": hashes }))
        .expect("sidecar JSON encodes");
    (
        object_id,
        1,
        Bytes::from_static(b"base"),
        CheckpointMeta {
            creator_replica_id: "cleanup-task-test".to_owned(),
            written_at_unix_ms: 1,
            snap_seq: Some(1),
            session_id: "cleanup-task-session".to_owned(),
            layer_name: "cleanup-task-layer".to_owned(),
            sidecar_bytes: Some(Bytes::from(sidecar)),
            ..CheckpointMeta::default()
        },
    )
}

/// A failed re-date is not fatal (b.g65, D4). The collect claims two aged
/// tombstones in one batch: `HELD`, whose value Valkey still holds, so the
/// collect keeps it and re-dates its tombstone; and `FREE`, which nobody holds
/// or names. Every `recon_redate` call answers an error (a `ScriptReplyProxy`
/// in front of the task's client, matching the script's SHA). The pass goes
/// on: it deletes `FREE`'s backup copy, which happens after the re-date in the
/// same batch. `HELD`'s tombstone stays, at its old score.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_failed_redate_leaves_its_members_where_they_were_and_the_pass_goes_on() {
    use fred::interfaces::HashesInterface;
    const HELD: &str = "a11ce0";
    const FREE: &str = "f4ee00";
    let tv = TestValkey::setup().await;
    let sha = live_edit::storage::recon_fence::RECON_REDATE_SCRIPT
        .sha1()
        .to_string();
    let proxy = ScriptReplyProxy::start(
        test_valkey_addr(),
        script_frames_naming(
            sha,
            ProxyAction::Eval(error_reply_script("ERR staged re-date failure")),
        ),
    )
    .await;
    let proxied = client_through(&proxy).await;
    let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
    let (object_id, snap_seq, bytes, meta) = base_naming(1, &["0ther"]);
    store
        .put_layer_checkpoint(object_id, snap_seq, bytes, meta)
        .await
        .expect("seeding a retained backup");
    store
        .put_ext_blob(FREE, Bytes::from_static(b"bytes"))
        .await
        .expect("put_ext_blob");
    let keys = tv.key_schema().clone();
    let _: u64 = tv
        .client()
        .pool
        .hset(keys.ext_meta(HELD), vec![("ref_count", "1")])
        .await
        .expect("Valkey holds HELD");
    let tombs = keys.checkpoint_reconcile_tombstones();
    let _: () = tv
        .client()
        .pool
        .zadd(
            &tombs,
            None,
            None,
            false,
            false,
            vec![(1.0, HELD), (1.0, FREE)],
        )
        .await
        .expect("seeding aged tombstones");
    let cfg = config("cleanup-task-redate", Duration::from_secs(3600));
    plant_own_checkpoint_lock(&tv, &cfg).await;
    let cancel = CancellationToken::new();
    let handle = live_edit::checkpoint::spawn_checkpoint_cleanup(
        proxied,
        keys.clone(),
        Arc::clone(&store),
        cfg,
        None,
        None,
        cancel.clone(),
    );

    eventually(
        "the collect deletes FREE's copy after the failed re-date",
        || async {
            store
                .get_ext_blob(FREE)
                .await
                .expect("get_ext_blob")
                .is_none()
        },
    )
    .await;
    assert!(proxy.hits() >= 1, "fixture: the re-date was failed");
    let held_score: Option<f64> = tv.client().pool.zscore(&tombs, HELD).await.expect("ZSCORE");
    assert_eq!(
        held_score,
        Some(1.0),
        "HELD stays registered, at its old score: a failed re-date costs only head-of-line \
         order"
    );

    cancel.cancel();
    tokio::time::timeout(WAIT, handle)
        .await
        .expect("the task joins")
        .expect("the task does not panic");
}

/// The trials [`a_cancel_during_a_pass_starts_no_further_pass_though_the_interval_has_elapsed`]
/// runs. An unbiased task takes the elapsed interval over the cancel half the
/// time, so it survives every trial with probability 2⁻³².
const CANCEL_TRIALS: usize = 32;

/// Permits that open a gate for good.
const GATE_OPEN: usize = 1 << 20;

/// **Regression (b.jt5's checkpoint fold): once the cancel fires, an
/// interval that has also elapsed starts no further pass.** The task's
/// `select!` between passes is biased toward the cancel. Each trial holds a
/// pass at the collect's delete of an aged tombstone's backup copy — a write,
/// which a cancel does not abandon — for [`OVERRUN`], three intervals; cancels;
/// and lets the delete through, so when the pass returns the cancel and the
/// interval are both ready. Every pass starts by acquiring or refreshing
/// `le:gc:leader`, which sets its TTL to `LEADER_LOCK_TTL_MS`; the trial
/// stamps an hour-long TTL while the pass is held, so a further pass shows as
/// the TTL back under it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_cancel_during_a_pass_starts_no_further_pass_though_the_interval_has_elapsed() {
    use live_edit::storage::external_values::LEADER_LOCK_TTL_MS;

    /// The task's interval between passes: short, so a pass held for
    /// [`OVERRUN`] has let it elapse.
    const INTERVAL: Duration = Duration::from_millis(10);
    /// How long each trial holds its pass: three intervals. It is the
    /// interval's own period that must pass here, not another task.
    const OVERRUN: Duration = Duration::from_millis(30);
    const SENTINEL_TTL_MS: i64 = 3_600_000;
    const FREE: &str = "f4ee01";

    let tv = TestValkey::setup().await;
    let keys = tv.key_schema().clone();
    let tombs = keys.checkpoint_reconcile_tombstones();
    let gc_lock = keys.gc_leader_lock();

    for trial in 0..CANCEL_TRIALS {
        // A retained backup, so the live set is not empty, and an aged
        // tombstone for a copy nothing names or holds: the collect deletes it.
        let inner: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
        let (object_id, snap_seq, bytes, meta) = base_naming(1, &["0ther"]);
        inner
            .put_layer_checkpoint(object_id, snap_seq, bytes, meta)
            .await
            .expect("seeding a retained backup");
        inner
            .put_ext_blob(FREE, Bytes::from_static(b"bytes"))
            .await
            .expect("put_ext_blob");
        let _: () = tv
            .client()
            .pool
            .zadd(&tombs, None, None, false, false, vec![(1.0, FREE)])
            .await
            .expect("seeding an aged tombstone");
        let store = ParkingStore::parking_ext_deletes(Arc::clone(&inner));
        let cfg = CheckpointLoopConfig {
            reconcile_interval: INTERVAL,
            ..config("cleanup-task-biased-cancel", Duration::from_secs(3600))
        };
        plant_own_checkpoint_lock(&tv, &cfg).await;
        let cancel = CancellationToken::new();
        let handle = live_edit::checkpoint::spawn_checkpoint_cleanup(
            tv.client().clone(),
            keys.clone(),
            Arc::clone(&store) as Arc<dyn CheckpointStore>,
            cfg,
            None,
            None,
            cancel.clone(),
        );

        eventually("the collect reaches the copy's delete", || async {
            store.ext_deletes_entered.load(Ordering::SeqCst) == 1
        })
        .await;
        let stamped: bool = tv
            .client()
            .pool
            .pexpire(&gc_lock, SENTINEL_TTL_MS, None)
            .await
            .expect("PEXPIRE le:gc:leader");
        assert!(
            stamped,
            "trial {trial}: fixture: the task holds le:gc:leader"
        );
        tokio::time::sleep(OVERRUN).await;

        cancel.cancel();
        store.delete_gate.add_permits(GATE_OPEN);
        tokio::time::timeout(Duration::from_secs(5), handle)
            .await
            .unwrap_or_else(|_| panic!("trial {trial}: the task ends on its cancel"))
            .expect("the task does not panic");

        let ttl: i64 = tv.client().pool.pttl(&gc_lock).await.expect("PTTL");
        assert!(
            ttl > LEADER_LOCK_TTL_MS as i64,
            "trial {trial}: a pass started after the cancel (it set le:gc:leader's TTL to {ttl} \
             ms), though the cancel was ready when the pass in flight returned"
        );
        assert!(
            inner
                .get_ext_blob(FREE)
                .await
                .expect("get_ext_blob")
                .is_none(),
            "trial {trial}: the delete in flight at the cancel ran to its end"
        );
    }
}
