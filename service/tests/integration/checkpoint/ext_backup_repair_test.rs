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

//! b.wsx — the external-value backup repair (`checkpoint::ext_backup_repair`)
//! and the backup confirmation it acts on (`storage::ext_backup`), against a
//! real Valkey and a recording backup store.
//!
//! A copy the ext-mirror loses (dropped, failed, discarded at shutdown) leaves
//! its value in Valkey with no copy in the backup store and no record of the
//! loss. The confirmation records the opposite: `backed_up_in = <store id>:<ms>`
//! on the value's meta, written only once the backup store answered that it
//! holds the bytes. Every test here drives the repair's pass, the mirror
//! worker's confirmation or the confirm script directly, and reads the outcome
//! from three places: the pass's [`PassSummary`] (and the counters it ticks),
//! the meta's field in Valkey, and the recording store's log of existence
//! checks and uploads.
//!
//! **The recording store** ([`RepairStore`]) wraps an `InMemoryCheckpointStore`
//! (which keeps its own copy of each upload, so the charge a re-send's bytes
//! carry returns once the upload call has returned or been dropped; a
//! LocalDisk write instead holds the bytes, and the charge, until its write
//! ends, which `local_disk`'s own tests cover). It logs every `ext_blob_exists`
//! and `put_ext_blob` by hash, tracks how many of each are in flight and the
//! peak, can hold either behind a gate, fail the next N of either, fail one
//! hash's check once another check has answered absent, and can stage a
//! collection (the GC's `DEL meta data`) of one hash just before it answers a
//! check or an upload.
//!
//! **Valkey replies** the repair must see fail (a failed size read, a failed
//! read of the bytes, a length that changed, a value collected or shrunk
//! between two ranges of its read, a failed confirmation) are staged with
//! `ScriptReplyProxy` on a client only the repair uses, so no other client can
//! take the fault. The same proxy, staging nothing, records the ranges a read
//! asks for.
//!
//! **Ages.** The tests run the pass with an age gate of 0 and first wait until
//! Valkey's clock has passed every stamp they stored
//! ([`Rig::wait_until_stored_before_now`]), so "older than the gate" is an
//! event, not a sleep; the age-gate test sets an old stamp outright.
//!
//! **What this file does not cover:** the clean-up fence (b.g65 adds it to both
//! upload sites), and the collect's clear of the field, which is a
//! `checkpointer` unit test (it needs the reconciliation harness).

use std::num::NonZeroU64;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bytes::Bytes;
use fred::interfaces::{HashesInterface, KeysInterface};
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;

use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
use live_edit::checkpoint::ext_backup_repair::{
    PassError, PassSummary, EXT_BACKUP_PROBE_CONCURRENCY, META_CHUNK,
};
use live_edit::checkpoint::ext_mirror::item_charge_kib;
use live_edit::checkpoint::{
    ext_mirror_channel, spawn_ext_backup_repair, spawn_ext_mirror_workers, CheckpointBlob,
    CheckpointError, CheckpointHandle, CheckpointMeta, CheckpointStore, ExtBackupRepair,
    ExtBackupRepairTiming, ExtMirrorReceiver, ExtMirrorSender, LayerArtifact, LiveExtSet,
};
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::{
    SharedMetrics, EXT_MIRROR_DROP_BUDGET_FULL, EXT_MIRROR_DROP_SHUTDOWN,
    EXT_MIRROR_STATE_IN_FLIGHT, EXT_MIRROR_STATE_QUEUED,
};
use live_edit::storage::client::ValkeyClient;
use live_edit::storage::ext_backup::{
    confirm_ext_backup, parse_backed_up_in, read_ext_value_bytes, valkey_time_ms, ExtBackupMarks,
    ExtValueRead, BACKED_UP_IN_FIELD, EXT_VALUE_READ_CHUNK_BYTES,
};
use live_edit::storage::external_values::{
    acquire_gc_leader_lock, ext_content_hash, ExternalValueStore, GcLeaderAcquire,
    ValkeyExternalValueStore, GC_LEADER_TAKEOVER_BELOW_MS, LEADER_LOCK_TTL_MS,
};
use live_edit::storage::keys::{sha256_to_hex, KeySchema};
use live_edit::storage::leader_lease::LeaderIdentity;
use live_edit::storage::test_support::TestValkey;

use super::helpers::DelegatingCheckpointStore;
use crate::storage::snapshot_read_support::{
    client_through, error_reply_script, test_valkey_addr, ProxyAction, ProxyRule, ScriptReplyProxy,
};

/// The `service` label on the metrics, and the repair's replica label.
const SERVICE_LABEL: &str = "ext-backup-repair-test";

/// The backup store identity the repair and the mirror workers confirm under.
const STORE_ID: &str = "0123456789abcdef";

/// Another backup store's identity: a confirmation under it is no
/// confirmation here.
const OTHER_STORE_ID: &str = "fedcba9876543210";

/// The adapter call ceiling handed to the repair, which bounds every
/// existence check; [`timing`] uses it as the upload ceiling too.
const CALL_CEILING: Duration = Duration::from_secs(5);

/// How long any wait for a background condition may take before the test
/// fails.
const WAIT_BUDGET: Duration = Duration::from_secs(10);

/// One day in Valkey milliseconds: the production audit age.
const DAY_MS: u64 = 86_400_000;

const KIB: u64 = 1024;

/// Permits that open a gate for good.
const GATE_OPEN_PERMITS: usize = 1 << 20;

// ─────────────────────────────────────────────────────────────────────────────
// The recording backup store
// ─────────────────────────────────────────────────────────────────────────────

/// Where a staged collection lands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CollectAt {
    /// Just before the store answers the hash's existence check.
    Check,
    /// Just before the store takes the hash's upload.
    Upload,
}

/// A collection of one hash (the GC's `DEL meta data`) the store stages.
struct StagedCollect {
    at: CollectAt,
    hash_hex: String,
    client: ValkeyClient,
    keys: KeySchema,
}

/// One kind of call's bookkeeping: every hash in call order, how many are in
/// flight, the peak, a gate each call passes, and how many next calls fail.
struct CallLog {
    hashes: Mutex<Vec<String>>,
    in_flight: AtomicUsize,
    peak: AtomicUsize,
    gate: Semaphore,
    fail_next: AtomicUsize,
}

impl CallLog {
    fn new(held: bool) -> Self {
        Self {
            hashes: Mutex::new(Vec::new()),
            in_flight: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            gate: Semaphore::new(if held { 0 } else { GATE_OPEN_PERMITS }),
            fail_next: AtomicUsize::new(0),
        }
    }

    /// Record a call's entry; the returned guard counts it out of flight when
    /// dropped, the call's future dropped mid-way included.
    fn enter(&self, hash_hex: &str) -> InFlight<'_> {
        self.hashes
            .lock()
            .expect("the call log is never poisoned")
            .push(hash_hex.to_owned());
        let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(now, Ordering::SeqCst);
        InFlight(&self.in_flight)
    }

    /// Whether this call is one the test asked to fail.
    fn take_failure(&self) -> bool {
        self.fail_next
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
            .is_ok()
    }

    fn calls(&self) -> Vec<String> {
        self.hashes
            .lock()
            .expect("the call log is never poisoned")
            .clone()
    }
}

struct InFlight<'a>(&'a AtomicUsize);

impl Drop for InFlight<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

/// The backup store the tests hand the repair and the mirror workers (module
/// docs).
struct RepairStore {
    inner: DelegatingCheckpointStore<InMemoryCheckpointStore>,
    checks: CallLog,
    uploads: CallLog,
    collect: Mutex<Option<StagedCollect>>,
    /// A hash whose check fails, but only once another check has answered
    /// absent ([`Self::fail_check_after_an_absent_answer`]).
    fail_after_absent: Mutex<Option<String>>,
    /// How many checks have answered absent (`Ok(false)`).
    absent_answers: AtomicUsize,
}

impl RepairStore {
    fn new() -> Arc<Self> {
        Self::with_gates(false, false)
    }

    /// A store whose existence checks (`checks_held`) and uploads
    /// (`uploads_held`) wait inside the call until the test opens the gate.
    fn with_gates(checks_held: bool, uploads_held: bool) -> Arc<Self> {
        Arc::new(Self {
            inner: DelegatingCheckpointStore::new(Arc::new(InMemoryCheckpointStore::new())),
            checks: CallLog::new(checks_held),
            uploads: CallLog::new(uploads_held),
            collect: Mutex::new(None),
            fail_after_absent: Mutex::new(None),
            absent_answers: AtomicUsize::new(0),
        })
    }

    /// Put `bytes` in the backup store directly, as an earlier copy would
    /// have: not logged as an upload.
    async fn seed(&self, hash_hex: &str, bytes: &[u8]) {
        self.inner
            .inner
            .put_ext_blob(hash_hex, Bytes::copy_from_slice(bytes))
            .await
            .expect("seeding the backup store succeeds");
    }

    /// The bytes the backup store holds under `hash_hex`.
    async fn held(&self, hash_hex: &str) -> Option<Bytes> {
        self.inner
            .inner
            .get_ext_blob(hash_hex)
            .await
            .expect("reading the backup store succeeds")
    }

    fn open_checks(&self) {
        self.checks.gate.add_permits(GATE_OPEN_PERMITS);
    }

    fn open_uploads(&self) {
        self.uploads.gate.add_permits(GATE_OPEN_PERMITS);
    }

    fn fail_next_checks(&self, n: usize) {
        self.checks.fail_next.store(n, Ordering::SeqCst);
    }

    fn fail_next_uploads(&self, n: usize) {
        self.uploads.fail_next.store(n, Ordering::SeqCst);
    }

    /// Fail `hash_hex`'s check, but only once another check has returned
    /// absent: the failure lands after the pass has an absent value in hand.
    fn fail_check_after_an_absent_answer(&self, hash_hex: &str) {
        *self.fail_after_absent.lock().expect("never poisoned") = Some(hash_hex.to_owned());
    }

    /// Whether `hash_hex`'s check is the one
    /// [`Self::fail_check_after_an_absent_answer`] staged; if so, wait (within
    /// [`WAIT_BUDGET`]) until another check has answered absent.
    async fn wait_to_fail_after_an_absent_answer(&self, hash_hex: &str) -> bool {
        let staged = self
            .fail_after_absent
            .lock()
            .expect("never poisoned")
            .clone();
        if staged.as_deref() != Some(hash_hex) {
            return false;
        }
        let deadline = Instant::now() + WAIT_BUDGET;
        while self.absent_answers.load(Ordering::SeqCst) == 0 {
            assert!(
                Instant::now() < deadline,
                "no other check answered absent before {hash_hex}'s staged failure"
            );
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        true
    }

    fn stage_collect(&self, at: CollectAt, hash_hex: &str, tv: &TestValkey) {
        *self.collect.lock().expect("never poisoned") = Some(StagedCollect {
            at,
            hash_hex: hash_hex.to_owned(),
            client: tv.client().clone(),
            keys: tv.key_schema().clone(),
        });
    }

    /// Run the staged collection when it is due here, once.
    async fn collect_if_due(&self, at: CollectAt, hash_hex: &str) {
        let staged = {
            let mut slot = self.collect.lock().expect("never poisoned");
            match slot.as_ref() {
                Some(s) if s.at == at && s.hash_hex == hash_hex => slot.take(),
                _ => None,
            }
        };
        if let Some(s) = staged {
            let _: u64 = s
                .client
                .pool
                .del(vec![s.keys.ext_meta(hash_hex), s.keys.ext_data(hash_hex)])
                .await
                .expect("the staged collection's DEL succeeds");
        }
    }

    fn checked(&self) -> Vec<String> {
        self.checks.calls()
    }

    fn uploaded(&self) -> Vec<String> {
        self.uploads.calls()
    }
}

#[async_trait::async_trait]
impl CheckpointStore for RepairStore {
    // ── The two instrumented methods ─────────────────────────────────────
    async fn ext_blob_exists(&self, hash_hex: &str) -> Result<bool, CheckpointError> {
        let _in_flight = self.checks.enter(hash_hex);
        let _pass = self
            .checks
            .gate
            .acquire()
            .await
            .expect("the gate is never closed");
        if self.checks.take_failure() || self.wait_to_fail_after_an_absent_answer(hash_hex).await {
            return Err(CheckpointError::Transient {
                source: "synthetic ext_blob_exists failure for the backup repair test".into(),
            });
        }
        self.collect_if_due(CollectAt::Check, hash_hex).await;
        let answer = self.inner.ext_blob_exists(hash_hex).await;
        if matches!(answer, Ok(false)) {
            self.absent_answers.fetch_add(1, Ordering::SeqCst);
        }
        answer
    }

    async fn put_ext_blob(&self, hash_hex: &str, bytes: Bytes) -> Result<(), CheckpointError> {
        let _in_flight = self.uploads.enter(hash_hex);
        let _pass = self
            .uploads
            .gate
            .acquire()
            .await
            .expect("the gate is never closed");
        if self.uploads.take_failure() {
            return Err(CheckpointError::Transient {
                source: "synthetic put_ext_blob failure for the backup repair test".into(),
            });
        }
        self.collect_if_due(CollectAt::Upload, hash_hex).await;
        self.inner.put_ext_blob(hash_hex, bytes).await
    }

    // ── Verbatim forwarding, method by method ────────────────────────────
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

    async fn get_ext_blob(&self, hash_hex: &str) -> Result<Option<Bytes>, CheckpointError> {
        self.inner.get_ext_blob(hash_hex).await
    }

    async fn delete_ext_blob(&self, hash_hex: &str) -> Result<(), CheckpointError> {
        self.inner.delete_ext_blob(hash_hex).await
    }

    async fn put_counters_sidecar(&self, body: Bytes) -> Result<(), CheckpointError> {
        self.inner.put_counters_sidecar(body).await
    }

    async fn get_counters_sidecar(&self) -> Result<Option<Bytes>, CheckpointError> {
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

    async fn list_layer_artifacts(&self) -> Result<Vec<LayerArtifact>, CheckpointError> {
        self.inner.list_layer_artifacts().await
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

// ─────────────────────────────────────────────────────────────────────────────
// The rig
// ─────────────────────────────────────────────────────────────────────────────

/// One external value: its bytes and their content hash.
#[derive(Clone)]
struct Value {
    hash: [u8; 32],
    hex: String,
    bytes: Vec<u8>,
}

/// A value of `len` bytes, told apart from others by `tag`.
fn value(tag: u16, len: usize) -> Value {
    let mut bytes = vec![(tag % 251) as u8; len.max(2)];
    bytes[..2].copy_from_slice(&tag.to_le_bytes());
    bytes.truncate(len);
    let hash = ext_content_hash(&bytes);
    Value {
        hash,
        hex: sha256_to_hex(&hash),
        bytes,
    }
}

/// The repair timing every test starts from: an age gate of 0 (the tests wait
/// for Valkey's clock instead), the production audit age and cap, a short
/// wait for budget room, and a time budget as long as the interval, so it
/// never cuts a pass the interval does not.
fn timing() -> ExtBackupRepairTiming {
    ExtBackupRepairTiming {
        interval: Duration::from_secs(60),
        age_gate: Duration::ZERO,
        audit_age: Duration::from_millis(DAY_MS),
        audit_per_pass: 2_000,
        reserve_wait: Duration::from_millis(300),
        reserve_retry: Duration::from_millis(20),
        upload_ceiling: CALL_CEILING,
        work_budget: Duration::from_secs(60),
    }
}

/// A Valkey fixture, metrics, the recording store and a mirror channel whose
/// receiver the rig holds, so the channel stays open and nothing drains it
/// unless a test starts workers.
struct Rig {
    tv: TestValkey,
    metrics: SharedMetrics,
    store: Arc<RepairStore>,
    sender: ExtMirrorSender,
    rx: Option<ExtMirrorReceiver>,
}

impl Rig {
    async fn new(store: Arc<RepairStore>, budget_bytes: u64) -> Self {
        let tv = TestValkey::setup().await;
        let metrics: SharedMetrics = Arc::new(crate::common::metrics_scrape_only(SERVICE_LABEL));
        let (sender, rx) = ext_mirror_channel(
            64,
            NonZeroU64::new(budget_bytes).expect("a test budget is non-zero"),
            Arc::clone(&metrics),
        );
        Self {
            tv,
            metrics,
            store,
            sender,
            rx: Some(rx),
        }
    }

    fn client(&self) -> &ValkeyClient {
        self.tv.client()
    }

    fn keys(&self) -> &KeySchema {
        self.tv.key_schema()
    }

    fn backup_store(&self) -> Arc<dyn CheckpointStore> {
        Arc::clone(&self.store) as Arc<dyn CheckpointStore>
    }

    /// The store path with no mirror wired: a value stored through it is
    /// never copied, so it is owed from the start.
    fn unmirrored(&self) -> ValkeyExternalValueStore {
        ValkeyExternalValueStore::new(self.client().clone(), self.keys().clone())
            .with_metrics(Arc::clone(&self.metrics))
    }

    /// The store path with the mirror wired, as production builds it.
    fn mirrored(&self) -> ValkeyExternalValueStore {
        self.unmirrored()
            .with_ext_mirror_sender(self.sender.clone())
    }

    async fn store_unmirrored(&self, values: &[&Value]) {
        let ext = self.unmirrored();
        for v in values {
            ext.store_values(&[(&v.hash, v.bytes.as_slice())])
                .await
                .expect("store_values succeeds");
        }
    }

    /// Start `workers` mirror workers on the held receiver, confirming under
    /// [`STORE_ID`].
    fn start_workers(
        &mut self,
        workers: usize,
        cancel: &CancellationToken,
    ) -> Vec<tokio::task::JoinHandle<()>> {
        let rx = self.rx.take().expect("the receiver is still held");
        let marks = ExtBackupMarks::new(
            self.client().clone(),
            self.keys().clone(),
            STORE_ID.to_owned(),
            Some(Arc::clone(&self.metrics)),
        );
        spawn_ext_mirror_workers(
            self.backup_store(),
            rx,
            workers,
            Arc::clone(&self.metrics),
            Some(marks),
            cancel.clone(),
        )
    }

    /// A repair over the rig's Valkey, store and budget, with `timing`.
    fn repair(&self, timing: ExtBackupRepairTiming) -> ExtBackupRepair {
        self.repair_on(self.client().clone(), timing)
    }

    /// [`Self::repair`] on another client of the same store (a proxied one).
    fn repair_on(&self, client: ValkeyClient, timing: ExtBackupRepairTiming) -> ExtBackupRepair {
        ExtBackupRepair::new(
            client,
            self.keys().clone(),
            self.backup_store(),
            self.sender.clone(),
            STORE_ID.to_owned(),
            CALL_CEILING,
            Some(Arc::clone(&self.metrics)),
        )
        .with_timing(timing)
    }

    async fn pass(&self, repair: &ExtBackupRepair) -> PassSummary {
        tokio::time::timeout(WAIT_BUDGET, repair.run_pass(&CancellationToken::new()))
            .await
            .expect("the pass ends within the budget")
            .unwrap_or_else(|e| panic!("the pass completes: {e}"))
    }

    async fn meta_field(&self, hash_hex: &str, field: &str) -> Option<String> {
        self.client()
            .pool
            .hget(self.keys().ext_meta(hash_hex), field)
            .await
            .expect("HGET ext:meta")
    }

    /// The value's `backed_up_in`, as stored.
    async fn confirmation(&self, hash_hex: &str) -> Option<String> {
        self.meta_field(hash_hex, BACKED_UP_IN_FIELD).await
    }

    async fn stamp(&self, hash_hex: &str) -> Option<String> {
        self.meta_field(hash_hex, "last_store_ms").await
    }

    async fn set_meta_field(&self, hash_hex: &str, field: &str, value: String) {
        let _: u64 = self
            .client()
            .pool
            .hset(self.keys().ext_meta(hash_hex), vec![(field, value)])
            .await
            .expect("HSET ext:meta");
    }

    async fn exists(&self, key: String) -> bool {
        let n: u64 = self.client().pool.exists(key).await.expect("EXISTS");
        n == 1
    }

    async fn now_ms(&self) -> u64 {
        valkey_time_ms(self.client()).await.expect("TIME")
    }

    /// Wait until Valkey's clock is past every stamp `values` carry, so an age
    /// gate of 0 finds each one stored before the pass's clock.
    async fn wait_until_stored_before_now(&self, values: &[&Value]) {
        let mut newest = 0u64;
        for v in values {
            if let Some(ms) = self.stamp(&v.hex).await.and_then(|s| s.parse::<u64>().ok()) {
                newest = newest.max(ms);
            }
        }
        let deadline = Instant::now() + WAIT_BUDGET;
        while self.now_ms().await <= newest {
            assert!(
                Instant::now() < deadline,
                "Valkey's clock never passed {newest}"
            );
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    }

    fn render(&self) -> String {
        render(&self.metrics)
    }

    fn repair_count(&self, outcome: &str) -> u64 {
        sample_value(
            &self.render(),
            "live_edit_checkpoint_ext_backup_repair_total",
            &[("outcome", outcome)],
        )
        .unwrap_or_else(|| panic!("outcome {outcome:?} is zero-seeded")) as u64
    }

    fn mark_failed(&self, op: &str) -> u64 {
        sample_value(
            &self.render(),
            "live_edit_checkpoint_ext_backup_mark_failed_total",
            &[("op", op)],
        )
        .unwrap_or_else(|| panic!("op {op:?} is zero-seeded")) as u64
    }

    fn owed_gauge(&self) -> Option<u64> {
        sample_value(
            &self.render(),
            "live_edit_checkpoint_ext_backup_owed",
            &[("service", SERVICE_LABEL)],
        )
        .map(|v| v as u64)
    }

    fn mirror_bytes(&self, state: &str) -> u64 {
        sample_value(
            &self.render(),
            "live_edit_checkpoint_ext_mirror_bytes",
            &[("state", state)],
        )
        .unwrap_or_else(|| panic!("state {state:?} is zero-seeded")) as u64
    }

    fn dropped(&self, reason: &str) -> u64 {
        sample_value(
            &self.render(),
            "live_edit_checkpoint_ext_mirror_dropped_total",
            &[("reason", reason)],
        )
        .unwrap_or(0.0) as u64
    }

    fn charged(&self) -> u64 {
        self.sender.budget_bytes() - self.sender.available_budget_bytes()
    }

    fn assert_budget_whole(&self, step: &str) {
        assert_eq!(
            self.sender.available_budget_bytes(),
            self.sender.budget_bytes(),
            "{step}: every charge has returned"
        );
    }
}

/// The charge, in bytes, of one value of `len` bytes.
fn charge_of(len: u64) -> u64 {
    item_charge_kib(len) * KIB
}

/// Assert `field` is a confirmation under `store_id`, and return its time.
fn assert_confirmed_under(field: Option<&str>, store_id: &str, what: &str) -> u64 {
    let field = field.unwrap_or_else(|| panic!("{what}: no confirmation"));
    let (id, ms) =
        parse_backed_up_in(field).unwrap_or_else(|| panic!("{what}: {field:?} does not parse"));
    assert_eq!(id, store_id, "{what}: confirmed under the wrong store");
    ms
}

/// Poll `probe` until it holds, within [`WAIT_BUDGET`].
async fn wait_for<F: FnMut() -> bool>(what: &str, mut probe: F) {
    let deadline = Instant::now() + WAIT_BUDGET;
    while !probe() {
        assert!(Instant::now() < deadline, "never happened: {what}");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// Wait until `hash_hex` carries a confirmation, within [`WAIT_BUDGET`].
async fn wait_confirmed(rig: &Rig, hash_hex: &str) {
    let deadline = Instant::now() + WAIT_BUDGET;
    while rig.confirmation(hash_hex).await.is_none() {
        assert!(Instant::now() < deadline, "{hash_hex} is never confirmed");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// Cancel `handles`' tasks and join them within [`WAIT_BUDGET`].
async fn shut_down(cancel: CancellationToken, handles: Vec<tokio::task::JoinHandle<()>>) {
    cancel.cancel();
    for h in handles {
        tokio::time::timeout(WAIT_BUDGET, h)
            .await
            .expect("the task joins after the cancel")
            .expect("the task does not panic");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Lost copies are re-sent (T1–T3)
// ─────────────────────────────────────────────────────────────────────────────

/// T1: a copy still queued when the mirror's receiver is dropped at shutdown
/// is discarded (`dropped{reason="shutdown"}`) and never confirmed; one pass
/// finds the value owed, uploads Valkey's bytes and confirms them under this
/// store's identity. The reservation's charge returns once the upload is done
/// with the bytes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_copy_discarded_at_shutdown_is_resent_and_confirmed_by_one_pass() {
    let mut rig = Rig::new(RepairStore::new(), 1 << 20).await;
    let v = value(1, 4 * KIB as usize);
    rig.mirrored()
        .store_values(&[(&v.hash, v.bytes.as_slice())])
        .await
        .expect("store_values succeeds");
    assert_eq!(
        rig.charged(),
        charge_of(4 * KIB),
        "fixture: the copy is queued"
    );
    drop(rig.rx.take());
    assert_eq!(
        rig.dropped(EXT_MIRROR_DROP_SHUTDOWN),
        1,
        "fixture: discarded"
    );
    assert!(rig.store.held(&v.hex).await.is_none());
    assert_eq!(rig.confirmation(&v.hex).await, None);

    rig.wait_until_stored_before_now(&[&v]).await;
    let summary = rig.pass(&rig.repair(timing())).await;

    assert_eq!(
        (summary.owed_seen, summary.resent, summary.owed),
        (1, 1, 0),
        "{summary:?}"
    );
    assert_eq!(rig.repair_count("resent"), 1);
    assert_eq!(rig.store.uploaded(), vec![v.hex.clone()]);
    assert_eq!(
        rig.store.held(&v.hex).await.as_deref(),
        Some(v.bytes.as_slice()),
        "the backup store holds Valkey's bytes"
    );
    assert_confirmed_under(
        rig.confirmation(&v.hex).await.as_deref(),
        STORE_ID,
        "resent",
    );
    rig.assert_budget_whole("after the pass");
}

/// T2 and T19: a mirror worker whose upload fails leaves the value
/// unconfirmed (it confirms only after a successful upload); one pass
/// re-sends it and confirms it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_copy_whose_upload_failed_once_is_left_unconfirmed_and_resent_by_one_pass() {
    let store = RepairStore::new();
    store.fail_next_uploads(1);
    let mut rig = Rig::new(store, 1 << 20).await;
    let cancel = CancellationToken::new();
    let workers = rig.start_workers(1, &cancel);
    let failed = value(2, 2 * KIB as usize);
    let delivered = value(3, 2 * KIB as usize);
    let ext = rig.mirrored();
    ext.store_values(&[(&failed.hash, failed.bytes.as_slice())])
        .await
        .expect("store_values succeeds");
    ext.store_values(&[(&delivered.hash, delivered.bytes.as_slice())])
        .await
        .expect("store_values succeeds");
    // One worker takes items in order, so once the second copy is confirmed
    // the first one's upload, and anything after it, is done.
    wait_confirmed(&rig, &delivered.hex).await;
    assert_eq!(
        rig.store.uploaded(),
        vec![failed.hex.clone(), delivered.hex.clone()],
        "fixture: both copies were attempted, in order"
    );
    assert_eq!(
        rig.confirmation(&failed.hex).await,
        None,
        "a failed upload is never confirmed"
    );
    assert_eq!(rig.mark_failed("confirm"), 0);

    rig.wait_until_stored_before_now(&[&failed, &delivered])
        .await;
    let summary = rig.pass(&rig.repair(timing())).await;

    assert_eq!(
        (
            summary.backed,
            summary.owed_seen,
            summary.resent,
            summary.owed
        ),
        (1, 1, 1, 0),
        "{summary:?}"
    );
    assert_eq!(rig.store.checked(), vec![failed.hex.clone()]);
    assert_eq!(
        rig.store.held(&failed.hex).await.as_deref(),
        Some(failed.bytes.as_slice())
    );
    assert_confirmed_under(
        rig.confirmation(&failed.hex).await.as_deref(),
        STORE_ID,
        "re-sent",
    );
    shut_down(cancel, workers).await;
}

/// T3: a copy the mirror drops `budget_full` is never confirmed, while the
/// copy that took the budget is delivered and confirmed by its worker; one
/// pass leaves the confirmed value alone (no check, no upload) and re-sends
/// the dropped one.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_copy_dropped_budget_full_is_resent_and_the_delivered_one_is_left_alone() {
    let len = 4 * KIB;
    let mut rig = Rig::new(RepairStore::new(), charge_of(len)).await;
    let admitted = value(4, len as usize);
    let dropped = value(5, len as usize);
    let ext = rig.mirrored();
    ext.store_values(&[(&admitted.hash, admitted.bytes.as_slice())])
        .await
        .expect("store_values succeeds");
    ext.store_values(&[(&dropped.hash, dropped.bytes.as_slice())])
        .await
        .expect("store_values succeeds");
    assert_eq!(rig.dropped(EXT_MIRROR_DROP_BUDGET_FULL), 1, "fixture");

    let cancel = CancellationToken::new();
    let workers = rig.start_workers(1, &cancel);
    wait_confirmed(&rig, &admitted.hex).await;
    assert_eq!(rig.confirmation(&dropped.hex).await, None);
    rig.wait_until_stored_before_now(&[&admitted, &dropped])
        .await;

    let summary = rig.pass(&rig.repair(timing())).await;
    assert_eq!(
        (
            summary.backed,
            summary.owed_seen,
            summary.resent,
            summary.owed
        ),
        (1, 1, 1, 0),
        "{summary:?}"
    );
    assert_eq!(
        rig.store.checked(),
        vec![dropped.hex.clone()],
        "the confirmed value is not checked"
    );
    assert_eq!(
        rig.store.uploaded(),
        vec![admitted.hex.clone(), dropped.hex.clone()],
        "the worker's upload, then the repair's"
    );
    assert_confirmed_under(
        rig.confirmation(&dropped.hex).await.as_deref(),
        STORE_ID,
        "re-sent",
    );
    shut_down(cancel, workers).await;
    rig.assert_budget_whole("after the pass");
}

// ─────────────────────────────────────────────────────────────────────────────
// Classification against the store (T4–T7, T15)
// ─────────────────────────────────────────────────────────────────────────────

/// T4: an unconfirmed value the backup store already holds is confirmed with
/// no upload.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_unconfirmed_value_the_backup_store_holds_is_confirmed_with_no_upload() {
    let rig = Rig::new(RepairStore::new(), 1 << 20).await;
    let v = value(6, 3 * KIB as usize);
    rig.store_unmirrored(&[&v]).await;
    rig.store.seed(&v.hex, &v.bytes).await;
    rig.wait_until_stored_before_now(&[&v]).await;

    let summary = rig.pass(&rig.repair(timing())).await;

    assert_eq!(
        (summary.already_stored, summary.resent, summary.owed),
        (1, 0, 0),
        "{summary:?}"
    );
    assert_eq!(rig.repair_count("already_stored"), 1);
    assert_eq!(rig.store.checked(), vec![v.hex.clone()]);
    assert!(rig.store.uploaded().is_empty(), "no upload");
    assert_confirmed_under(rig.confirmation(&v.hex).await.as_deref(), STORE_ID, "held");
}

/// T5: a value confirmed under this store's identity is neither checked nor
/// uploaded, and its confirmation is left as it was.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_value_confirmed_under_this_store_is_neither_checked_nor_uploaded() {
    let rig = Rig::new(RepairStore::new(), 1 << 20).await;
    let v = value(7, 3 * KIB as usize);
    rig.store_unmirrored(&[&v]).await;
    let stamp = rig
        .stamp(&v.hex)
        .await
        .expect("stored values carry a stamp");
    assert!(
        confirm_ext_backup(rig.client(), rig.keys(), &v.hex, STORE_ID, &stamp)
            .await
            .expect("the confirm script runs"),
        "fixture: confirmed"
    );
    let confirmed = rig.confirmation(&v.hex).await;
    rig.wait_until_stored_before_now(&[&v]).await;

    let summary = rig.pass(&rig.repair(timing())).await;

    assert_eq!(
        (summary.backed, summary.owed_seen, summary.owed),
        (1, 0, 0),
        "{summary:?}"
    );
    assert!(rig.store.checked().is_empty(), "no existence check");
    assert!(rig.store.uploaded().is_empty(), "no upload");
    assert_eq!(rig.confirmation(&v.hex).await, confirmed, "left as it was");
}

/// T6: confirmations under another backup store's identity (the adapter now
/// points elsewhere) count as none: every such value is checked against this
/// store, confirmed when it holds the bytes and re-sent when it does not, and
/// each ends confirmed under this store's identity.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn values_confirmed_under_another_store_identity_are_all_checked_again() {
    let rig = Rig::new(RepairStore::new(), 1 << 20).await;
    let held = value(8, 2 * KIB as usize);
    let missing = value(9, 2 * KIB as usize);
    rig.store_unmirrored(&[&held, &missing]).await;
    rig.store.seed(&held.hex, &held.bytes).await;
    for v in [&held, &missing] {
        let stamp = rig.stamp(&v.hex).await.expect("stamped");
        assert!(
            confirm_ext_backup(rig.client(), rig.keys(), &v.hex, OTHER_STORE_ID, &stamp)
                .await
                .expect("the confirm script runs"),
            "fixture: confirmed under the other store"
        );
    }
    rig.wait_until_stored_before_now(&[&held, &missing]).await;

    let summary = rig.pass(&rig.repair(timing())).await;

    assert_eq!(
        (summary.owed_seen, summary.already_stored, summary.resent),
        (2, 1, 1),
        "{summary:?}"
    );
    let mut checked = rig.store.checked();
    checked.sort();
    let mut both = vec![held.hex.clone(), missing.hex.clone()];
    both.sort();
    assert_eq!(checked, both, "both are checked");
    assert_eq!(rig.store.uploaded(), vec![missing.hex.clone()]);
    for v in [&held, &missing] {
        assert_confirmed_under(rig.confirmation(&v.hex).await.as_deref(), STORE_ID, &v.hex);
    }
}

/// T7: the age gate. A value stored within it is young, left to the mirror
/// (not checked, not counted owed); one stored before it is checked and
/// re-sent.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_age_gate_leaves_a_young_value_to_the_mirror_and_checks_an_old_one() {
    let rig = Rig::new(RepairStore::new(), 1 << 20).await;
    let young = value(10, KIB as usize);
    let old = value(11, KIB as usize);
    rig.store_unmirrored(&[&young, &old]).await;
    let an_hour_ago = rig.now_ms().await - 3_600_000;
    rig.set_meta_field(&old.hex, "last_store_ms", an_hour_ago.to_string())
        .await;
    let gated = ExtBackupRepairTiming {
        age_gate: Duration::from_secs(600),
        ..timing()
    };

    let summary = rig.pass(&rig.repair(gated)).await;

    assert_eq!(
        (
            summary.young,
            summary.owed_seen,
            summary.resent,
            summary.owed
        ),
        (1, 1, 1, 0),
        "{summary:?}"
    );
    assert_eq!(
        rig.store.checked(),
        vec![old.hex.clone()],
        "only the old one"
    );
    assert_eq!(rig.confirmation(&young.hex).await, None);
    assert_confirmed_under(rig.confirmation(&old.hex).await.as_deref(), STORE_ID, "old");
}

/// T15: metas a recount or an older build wrote (no stamp, no confirmation)
/// and garbled confirmations are owed and checked; each ends confirmed under
/// this store's identity. A stampless meta is confirmed through the empty
/// stamp, which matches only a meta with no `last_store_ms`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn metas_with_no_fields_or_a_garbled_confirmation_are_checked_and_confirmed() {
    let rig = Rig::new(RepairStore::new(), 1 << 20).await;
    let stampless = value(12, KIB as usize);
    // A recount's meta: the reference count alone, beside the data.
    let _: () = rig
        .client()
        .pool
        .set(
            rig.keys().ext_data(&stampless.hex),
            stampless.bytes.clone(),
            None,
            None,
            false,
        )
        .await
        .expect("SET ext:data");
    rig.set_meta_field(&stampless.hex, "ref_count", "1".to_owned())
        .await;
    rig.store.seed(&stampless.hex, &stampless.bytes).await;

    let garbled: Vec<(Value, String)> = [
        "garbage".to_owned(),
        ":123".to_owned(),
        format!("{STORE_ID}:"),
        format!("{STORE_ID}:12x"),
    ]
    .into_iter()
    .enumerate()
    .map(|(i, field)| (value(20 + i as u16, KIB as usize), field))
    .collect();
    for (i, (v, field)) in garbled.iter().enumerate() {
        rig.store_unmirrored(&[v]).await;
        rig.set_meta_field(&v.hex, BACKED_UP_IN_FIELD, field.clone())
            .await;
        if i % 2 == 0 {
            rig.store.seed(&v.hex, &v.bytes).await;
        }
    }
    let all: Vec<&Value> = std::iter::once(&stampless)
        .chain(garbled.iter().map(|(v, _)| v))
        .collect();
    rig.wait_until_stored_before_now(&all).await;

    let summary = rig.pass(&rig.repair(timing())).await;

    assert_eq!(
        (
            summary.owed_seen,
            summary.already_stored,
            summary.resent,
            summary.owed
        ),
        (5, 3, 2, 0),
        "{summary:?}"
    );
    let mut checked = rig.store.checked();
    checked.sort();
    let mut expected: Vec<String> = all.iter().map(|v| v.hex.clone()).collect();
    expected.sort();
    assert_eq!(checked, expected, "every one is checked");
    for v in &all {
        assert_confirmed_under(rig.confirmation(&v.hex).await.as_deref(), STORE_ID, &v.hex);
    }
    assert_eq!(
        rig.stamp(&stampless.hex).await,
        None,
        "the confirmation adds no stamp"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Collection mid-pass, and the confirm script (T8, T9)
// ─────────────────────────────────────────────────────────────────────────────

/// T8: a value the GC collects after the walk read it and before the repair
/// reads its bytes is `value_gone`: nothing is uploaded, no charge is held,
/// and no meta is created for it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_value_collected_before_its_bytes_are_read_is_value_gone_and_never_uploaded() {
    let rig = Rig::new(RepairStore::new(), 1 << 20).await;
    let v = value(30, 2 * KIB as usize);
    rig.store_unmirrored(&[&v]).await;
    rig.store.stage_collect(CollectAt::Check, &v.hex, &rig.tv);
    rig.wait_until_stored_before_now(&[&v]).await;

    let summary = rig.pass(&rig.repair(timing())).await;

    assert_eq!(rig.store.checked(), vec![v.hex.clone()], "fixture: checked");
    assert_eq!(
        (summary.value_gone, summary.resent, summary.owed),
        (1, 0, 0),
        "{summary:?}"
    );
    assert_eq!(rig.repair_count("value_gone"), 1);
    assert!(rig.store.uploaded().is_empty(), "nothing is uploaded");
    assert!(rig.store.held(&v.hex).await.is_none());
    assert!(!rig.exists(rig.keys().ext_meta(&v.hex)).await, "no meta");
    rig.assert_budget_whole("after the pass");
}

/// T8: a value the GC collects while its upload is in flight is uploaded (a
/// leak in the backup store at worst) but its confirmation is declined: the
/// confirm script never creates a meta.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_value_collected_during_its_upload_gets_no_confirmation_and_no_meta() {
    let rig = Rig::new(RepairStore::new(), 1 << 20).await;
    let v = value(31, 2 * KIB as usize);
    rig.store_unmirrored(&[&v]).await;
    rig.store.stage_collect(CollectAt::Upload, &v.hex, &rig.tv);
    rig.wait_until_stored_before_now(&[&v]).await;

    let summary = rig.pass(&rig.repair(timing())).await;

    assert_eq!(summary.resent, 1, "the upload landed: {summary:?}");
    assert!(
        !rig.exists(rig.keys().ext_meta(&v.hex)).await,
        "the confirmation created no meta for the collected value"
    );
    assert!(!rig.exists(rig.keys().ext_data(&v.hex)).await);
}

/// T9: the confirm script writes `backed_up_in` only on a meta that exists and
/// carries the caller's stamp: an absent meta and a stale stamp write
/// nothing, a matching stamp writes `<store id>:<Valkey ms>`, and a later
/// matching call re-stamps it (the audit's renewal).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_confirm_script_writes_only_on_an_existing_meta_carrying_the_callers_stamp() {
    let rig = Rig::new(RepairStore::new(), 1 << 20).await;
    let (client, keys) = (rig.client(), rig.keys());
    let absent = value(40, KIB as usize);
    assert!(
        !confirm_ext_backup(client, keys, &absent.hex, STORE_ID, "1760000000000")
            .await
            .expect("the confirm script runs"),
        "an absent meta declines"
    );
    assert!(
        !rig.exists(keys.ext_meta(&absent.hex)).await,
        "and stays absent"
    );

    let v = value(41, KIB as usize);
    rig.store_unmirrored(&[&v]).await;
    let stamp: u64 = rig
        .stamp(&v.hex)
        .await
        .expect("stamped")
        .parse()
        .expect("a decimal stamp");
    let stale = (stamp - 1).to_string();
    assert!(
        !confirm_ext_backup(client, keys, &v.hex, STORE_ID, &stale)
            .await
            .expect("the confirm script runs"),
        "a stale stamp declines"
    );
    assert_eq!(rig.confirmation(&v.hex).await, None, "and writes nothing");

    let before = rig.now_ms().await;
    assert!(
        confirm_ext_backup(client, keys, &v.hex, STORE_ID, &stamp.to_string())
            .await
            .expect("the confirm script runs")
    );
    let after = rig.now_ms().await;
    let ms = assert_confirmed_under(rig.confirmation(&v.hex).await.as_deref(), STORE_ID, "match");
    assert!(
        (before..=after).contains(&ms),
        "the confirmation carries Valkey's clock: {before} <= {ms} <= {after}"
    );
    assert_eq!(
        rig.stamp(&v.hex).await,
        Some(stamp.to_string()),
        "the stamp is untouched"
    );

    assert!(
        confirm_ext_backup(client, keys, &v.hex, OTHER_STORE_ID, &stamp.to_string())
            .await
            .expect("the confirm script runs")
    );
    assert_confirmed_under(
        rig.confirmation(&v.hex).await.as_deref(),
        OTHER_STORE_ID,
        "a later confirmation replaces the field",
    );
}

/// T9, the empty stamp: `""` stands for "the meta carried no
/// `last_store_ms`". It confirms a stampless meta (a recount's), declines a
/// stamped one, and never creates an absent one; a non-empty stamp declines a
/// stampless meta.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_confirm_scripts_empty_stamp_confirms_only_a_stampless_meta_that_exists() {
    let rig = Rig::new(RepairStore::new(), 1 << 20).await;
    let (client, keys) = (rig.client(), rig.keys());
    let absent = value(42, KIB as usize);
    assert!(!confirm_ext_backup(client, keys, &absent.hex, STORE_ID, "")
        .await
        .expect("the confirm script runs"));
    assert!(
        !rig.exists(keys.ext_meta(&absent.hex)).await,
        "the empty stamp never creates a meta"
    );

    let stamped = value(43, KIB as usize);
    rig.store_unmirrored(&[&stamped]).await;
    assert!(
        !confirm_ext_backup(client, keys, &stamped.hex, STORE_ID, "")
            .await
            .expect("the confirm script runs"),
        "a stamped meta declines the empty stamp"
    );
    assert_eq!(rig.confirmation(&stamped.hex).await, None);

    let stampless = value(44, KIB as usize);
    rig.set_meta_field(&stampless.hex, "ref_count", "1".to_owned())
        .await;
    assert!(
        !confirm_ext_backup(client, keys, &stampless.hex, STORE_ID, "1760000000000")
            .await
            .expect("the confirm script runs"),
        "a stampless meta declines a stamp"
    );
    assert!(
        confirm_ext_backup(client, keys, &stampless.hex, STORE_ID, "")
            .await
            .expect("the confirm script runs")
    );
    assert_confirmed_under(
        rig.confirmation(&stampless.hex).await.as_deref(),
        STORE_ID,
        "stampless",
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// The byte budget (T12, T13)
// ─────────────────────────────────────────────────────────────────────────────

/// T12: while a re-send's upload is in flight, its reservation is counted
/// `in_flight` beside a hot-path copy queued meanwhile, and `queued +
/// in_flight` is the budget's charged bytes; the charge returns when the
/// upload is done with the bytes (here when the recording store's call
/// returns, which keeps a copy), leaving only the queued copy's.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_resends_charge_is_in_flight_beside_queued_copies_until_its_upload_drops_the_bytes() {
    let rig = Rig::new(RepairStore::with_gates(false, true), 64 * KIB).await;
    let v = value(50, 10 * KIB as usize);
    rig.store_unmirrored(&[&v]).await;
    rig.wait_until_stored_before_now(&[&v]).await;
    let repair = rig.repair(timing());
    let cancel = CancellationToken::new();
    let pass = tokio::spawn({
        let cancel = cancel.clone();
        async move { repair.run_pass(&cancel).await }
    });
    wait_for("the re-send's upload is in flight", || {
        rig.store.uploads.in_flight.load(Ordering::SeqCst) == 1
    })
    .await;

    let queued = value(51, 4 * KIB as usize);
    rig.sender.send(
        queued.hex.clone(),
        rig.sender.copy_value(&queued.bytes),
        "0".to_owned(),
    );
    let (q, f) = (
        rig.mirror_bytes(EXT_MIRROR_STATE_QUEUED),
        rig.mirror_bytes(EXT_MIRROR_STATE_IN_FLIGHT),
    );
    assert_eq!(
        (q, f),
        (charge_of(4 * KIB), charge_of(10 * KIB)),
        "the queued copy and the re-send"
    );
    assert_eq!(
        q + f,
        rig.charged(),
        "queued + in_flight is the charged bytes"
    );

    rig.store.open_uploads();
    let summary = tokio::time::timeout(WAIT_BUDGET, pass)
        .await
        .expect("the pass ends")
        .expect("the pass task does not panic")
        .expect("the pass completes");
    assert_eq!(summary.resent, 1, "{summary:?}");
    assert_eq!(
        (
            rig.mirror_bytes(EXT_MIRROR_STATE_QUEUED),
            rig.mirror_bytes(EXT_MIRROR_STATE_IN_FLIGHT)
        ),
        (charge_of(4 * KIB), 0),
        "the re-send's charge returned"
    );
    assert_eq!(rig.charged(), charge_of(4 * KIB));
}

/// T12: with the budget full of queued copies the repair finds no room for
/// its whole wait, counts `budget_busy` once and stops re-sending for the
/// pass, leaving both values owed; sends on the store path meanwhile are
/// refused at once (`budget_full`), never held behind the repair, and the
/// repair holds no charge afterwards.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_full_budget_ends_resending_with_budget_busy_and_sends_meanwhile_never_wait() {
    let rig = Rig::new(RepairStore::new(), 2 * charge_of(KIB)).await;
    for i in 0..2u16 {
        let filler = value(60 + i, KIB as usize);
        rig.sender.send(
            filler.hex.clone(),
            rig.sender.copy_value(&filler.bytes),
            "0".to_owned(),
        );
    }
    assert_eq!(rig.sender.available_budget_bytes(), 0, "fixture: full");
    let a = value(62, KIB as usize);
    let b = value(63, KIB as usize);
    rig.store_unmirrored(&[&a, &b]).await;
    rig.wait_until_stored_before_now(&[&a, &b]).await;

    let repair = rig.repair(timing());
    let pass = tokio::spawn(async move { repair.run_pass(&CancellationToken::new()).await });
    wait_for("both values are checked", || rig.store.checked().len() == 2).await;
    const SENDS: u64 = 100;
    for i in 0..SENDS {
        let v = value(1_000 + i as u16, KIB as usize);
        rig.sender.send(
            v.hex.clone(),
            rig.sender.copy_value(&v.bytes),
            "0".to_owned(),
        );
    }
    assert_eq!(
        rig.dropped(EXT_MIRROR_DROP_BUDGET_FULL),
        SENDS,
        "every send returned, refused budget_full"
    );
    let summary = tokio::time::timeout(WAIT_BUDGET, pass)
        .await
        .expect("the pass ends")
        .expect("the pass task does not panic")
        .expect("the pass completes");

    assert_eq!(
        (
            summary.budget_busy,
            summary.resent,
            summary.resend_failed,
            summary.owed
        ),
        (1, 0, 0, 2),
        "one budget_busy stops re-sending; both stay owed: {summary:?}"
    );
    assert_eq!(rig.repair_count("budget_busy"), 1);
    assert!(rig.store.uploaded().is_empty());
    assert_eq!(
        rig.charged(),
        2 * charge_of(KIB),
        "only the queued copies hold charges"
    );
}

/// A re-send that finds no room asks the budget again every `reserve_retry`
/// for up to `reserve_wait`, so room freed within the wait is taken: with a
/// reservation filling the budget, dropped once the value's check has
/// answered, the value is re-sent, not `budget_busy`.
///
/// Limit: the drop lands at the test's next poll after the check answers,
/// which is usually after the repair's first refused ask (one `STRLEN` round
/// trip later). A refused reservation cannot be observed, so no ordering
/// guarantees it: a run whose drop lands earlier passes without exercising the
/// retry. Red/green evidence (b.wsx test round 2): with the retry loop in
/// `reserve` removed (a refusal returned at once), this test failed with
/// `budget_busy: 1`, `(resent, budget_busy, owed) = (0, 1, 1)`; restored, it
/// passed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_resend_takes_budget_room_freed_while_it_waits() {
    let len = 4 * KIB;
    let rig = Rig::new(RepairStore::new(), charge_of(len)).await;
    let holder = rig
        .sender
        .try_reserve(len)
        .expect("fixture: the budget has room for one value");
    assert_eq!(rig.sender.available_budget_bytes(), 0, "fixture: full");
    let v = value(65, len as usize);
    rig.store_unmirrored(&[&v]).await;
    rig.wait_until_stored_before_now(&[&v]).await;
    let repair = rig.repair(ExtBackupRepairTiming {
        reserve_wait: WAIT_BUDGET,
        ..timing()
    });
    let pass = tokio::spawn(async move { repair.run_pass(&CancellationToken::new()).await });
    wait_for("the value's check has answered", || {
        rig.store.checked().len() == 1 && rig.store.checks.in_flight.load(Ordering::SeqCst) == 0
    })
    .await;
    drop(holder);

    let summary = tokio::time::timeout(WAIT_BUDGET, pass)
        .await
        .expect("the pass ends")
        .expect("the pass task does not panic")
        .expect("the pass completes");
    assert_eq!(
        (summary.resent, summary.budget_busy, summary.owed),
        (1, 0, 0),
        "{summary:?}"
    );
    assert_eq!(rig.store.uploaded(), vec![v.hex.clone()]);
    assert_confirmed_under(
        rig.confirmation(&v.hex).await.as_deref(),
        STORE_ID,
        "resent",
    );
    rig.assert_budget_whole("after the pass");
}

/// T13: a value larger than the mirror's whole budget is `oversized`: it takes
/// no reservation, stays owed, and does not stop re-sending for the pass. With
/// two oversized values and a small one, in whatever order the walk finds
/// them, every one is re-sent or counted: both oversized, the small one
/// uploaded. A pass that stopped re-sending at an oversized value would leave
/// a later one uncounted or the small one unsent.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_value_larger_than_the_whole_budget_is_oversized_stays_owed_and_the_pass_goes_on() {
    let rig = Rig::new(RepairStore::new(), 8 * KIB).await;
    let big = value(70, 16 * KIB as usize);
    let bigger = value(72, 24 * KIB as usize);
    let small = value(71, KIB as usize);
    rig.store_unmirrored(&[&big, &bigger, &small]).await;
    rig.wait_until_stored_before_now(&[&big, &bigger, &small])
        .await;

    let summary = rig.pass(&rig.repair(timing())).await;

    assert_eq!(
        (summary.oversized, summary.resent, summary.owed),
        (2, 1, 2),
        "{summary:?}"
    );
    assert_eq!(rig.repair_count("oversized"), 2);
    assert_eq!(rig.store.uploaded(), vec![small.hex.clone()]);
    for v in [&big, &bigger] {
        assert_eq!(rig.confirmation(&v.hex).await, None, "stays owed");
    }
    rig.assert_budget_whole("after the pass");
}

// ─────────────────────────────────────────────────────────────────────────────
// Leadership (T14)
// ─────────────────────────────────────────────────────────────────────────────

/// The repair task's interval in the leadership tests: one pass, then a wait
/// the test cancels.
const LONG_INTERVAL: Duration = Duration::from_secs(3_600);

/// T14: a replica that does not hold `le:gc:leader` runs no pass (no check, no
/// upload, no confirmation) and publishes 0 on the owed gauge, so a demoted
/// leader's last reading does not pin the fleet's `max()`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_non_leader_runs_no_pass_and_publishes_zero_owed() {
    let rig = Rig::new(RepairStore::new(), 1 << 20).await;
    let peer = LeaderIdentity::new("peer-replica", "peer-instance");
    let verdict = acquire_gc_leader_lock(
        rig.client(),
        &rig.keys().gc_leader_lock(),
        &peer,
        LEADER_LOCK_TTL_MS,
        GC_LEADER_TAKEOVER_BELOW_MS,
    )
    .await
    .expect("the peer takes the lock");
    assert!(matches!(verdict, GcLeaderAcquire::Acquired), "{verdict:?}");
    let v = value(80, KIB as usize);
    rig.store_unmirrored(&[&v]).await;
    rig.wait_until_stored_before_now(&[&v]).await;

    let cancel = CancellationToken::new();
    let task = spawn_ext_backup_repair(
        rig.repair(ExtBackupRepairTiming {
            interval: LONG_INTERVAL,
            ..timing()
        }),
        LeaderIdentity::new(SERVICE_LABEL, "instance-a"),
        cancel.clone(),
    );
    wait_for("the owed gauge is published", || rig.owed_gauge().is_some()).await;
    shut_down(cancel, vec![task]).await;

    assert_eq!(rig.owed_gauge(), Some(0), "a non-leader publishes 0");
    assert!(rig.store.checked().is_empty(), "no pass ran");
    assert!(rig.store.uploaded().is_empty());
    assert_eq!(rig.confirmation(&v.hex).await, None);
}

/// T14: the replica holding `le:gc:leader` runs the pass and publishes its
/// owed reading: here one oversized value that stays owed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_leader_runs_the_pass_and_publishes_what_stays_owed() {
    let rig = Rig::new(RepairStore::new(), 8 * KIB).await;
    let big = value(81, 16 * KIB as usize);
    let small = value(82, KIB as usize);
    rig.store_unmirrored(&[&big, &small]).await;
    rig.wait_until_stored_before_now(&[&big, &small]).await;

    let cancel = CancellationToken::new();
    let task = spawn_ext_backup_repair(
        rig.repair(ExtBackupRepairTiming {
            interval: LONG_INTERVAL,
            ..timing()
        }),
        LeaderIdentity::new(SERVICE_LABEL, "instance-a"),
        cancel.clone(),
    );
    wait_for("the owed gauge is published", || rig.owed_gauge().is_some()).await;
    shut_down(cancel, vec![task]).await;

    assert_eq!(rig.owed_gauge(), Some(1), "the oversized value stays owed");
    assert_eq!(rig.store.uploaded(), vec![small.hex.clone()]);
}

/// T14: two passes at once (two leaders briefly overlapping after a takeover)
/// converge on the state one pass reaches: every value in the backup store
/// and confirmed under this store's identity, every charge returned; a later
/// pass finds all of them backed and does nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_concurrent_passes_converge_on_the_state_one_pass_reaches() {
    let rig = Rig::new(RepairStore::new(), 1 << 20).await;
    let values: Vec<Value> = (0..6u16).map(|i| value(90 + i, 2 * KIB as usize)).collect();
    let refs: Vec<&Value> = values.iter().collect();
    rig.store_unmirrored(&refs).await;
    for v in values.iter().step_by(2) {
        rig.store.seed(&v.hex, &v.bytes).await;
    }
    rig.wait_until_stored_before_now(&refs).await;

    let (first, second) = (rig.repair(timing()), rig.repair(timing()));
    let (a, b) = tokio::join!(rig.pass(&first), rig.pass(&second));
    assert_eq!(
        (a.owed, b.owed),
        (0, 0),
        "each pass leaves nothing owed: {a:?} {b:?}"
    );

    for v in &values {
        assert_eq!(
            rig.store.held(&v.hex).await.as_deref(),
            Some(v.bytes.as_slice()),
            "{} is backed up",
            v.hex
        );
        assert_confirmed_under(rig.confirmation(&v.hex).await.as_deref(), STORE_ID, &v.hex);
    }
    rig.assert_budget_whole("after both passes");
    let (checked, uploaded) = (rig.store.checked().len(), rig.store.uploaded().len());
    let third = rig.pass(&rig.repair(timing())).await;
    assert_eq!((third.backed, third.owed_seen), (6, 0), "{third:?}");
    assert_eq!(
        (rig.store.checked().len(), rig.store.uploaded().len()),
        (checked, uploaded),
        "a converged store needs no further check or upload"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// The audit (T16)
// ─────────────────────────────────────────────────────────────────────────────

/// A confirmation under this store's identity, `age_ms` old.
async fn confirm_aged(rig: &Rig, v: &Value, age_ms: u64) -> String {
    let field = format!("{STORE_ID}:{}", rig.now_ms().await - age_ms);
    rig.set_meta_field(&v.hex, BACKED_UP_IN_FIELD, field.clone())
        .await;
    field
}

/// T16: a value confirmed more than a day ago whose copy was deleted from the
/// backup store out of band is found missing by the audit
/// (`confirmed_copy_missing`), re-sent and confirmed afresh. The audit is
/// served after the walk's owed values.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_audit_resends_a_confirmed_copy_deleted_out_of_band_after_the_owed_values() {
    let rig = Rig::new(RepairStore::new(), 1 << 20).await;
    let audited = value(100, 2 * KIB as usize);
    let owed = value(101, 2 * KIB as usize);
    rig.store_unmirrored(&[&audited, &owed]).await;
    let old = confirm_aged(&rig, &audited, DAY_MS + 3_600_000).await;
    rig.wait_until_stored_before_now(&[&audited, &owed]).await;

    let summary = rig.pass(&rig.repair(timing())).await;

    assert_eq!(
        (
            summary.audited,
            summary.confirmed_copy_missing,
            summary.resent,
            summary.owed
        ),
        (1, 1, 2, 0),
        "{summary:?}"
    );
    assert_eq!(rig.repair_count("confirmed_copy_missing"), 1);
    assert_eq!(
        rig.store.checked(),
        vec![owed.hex.clone(), audited.hex.clone()],
        "the owed value first, the audit after the walk"
    );
    assert_eq!(
        rig.store.held(&audited.hex).await.as_deref(),
        Some(audited.bytes.as_slice())
    );
    let renewed = assert_confirmed_under(
        rig.confirmation(&audited.hex).await.as_deref(),
        STORE_ID,
        "audited",
    );
    let (_, old_ms) = parse_backed_up_in(&old).expect("fixture parses");
    assert!(renewed > old_ms, "confirmed afresh");
}

/// T16: one pass audits at most its cap; the rest wait for later passes,
/// each audited value re-stamped (so it is backed next time), until every one
/// has been checked.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_audit_checks_at_most_its_cap_per_pass_and_the_rest_in_later_passes() {
    let rig = Rig::new(RepairStore::new(), 1 << 20).await;
    let values: Vec<Value> = (0..5u16).map(|i| value(110 + i, KIB as usize)).collect();
    let refs: Vec<&Value> = values.iter().collect();
    rig.store_unmirrored(&refs).await;
    for v in &values {
        rig.store.seed(&v.hex, &v.bytes).await;
        confirm_aged(&rig, v, DAY_MS + 3_600_000).await;
    }
    rig.wait_until_stored_before_now(&refs).await;
    let capped = ExtBackupRepairTiming {
        audit_per_pass: 2,
        ..timing()
    };
    let repair = rig.repair(capped);

    // (audited, deferred to a later pass, backed) per pass.
    for (pass, (audited, deferred, backed)) in
        [(2, 3, 0), (2, 1, 2), (1, 0, 4)].into_iter().enumerate()
    {
        let s = rig.pass(&repair).await;
        assert_eq!(
            (s.audited, s.audit_deferred, s.backed, s.already_stored),
            (audited, deferred, backed, audited),
            "pass {pass}: {s:?}"
        );
    }
    let mut checked = rig.store.checked();
    assert_eq!(checked.len(), 5, "five checks in all");
    checked.sort();
    checked.dedup();
    assert_eq!(checked.len(), 5, "each value audited exactly once");
    assert!(rig.store.uploaded().is_empty());
}

// ─────────────────────────────────────────────────────────────────────────────
// Failures: Valkey replies staged through a proxy, and the backup store's
// ─────────────────────────────────────────────────────────────────────────────

/// A rule answering every `name` command whose arguments name an `ext:data`
/// key with `action`.
fn ext_data_command(name: &'static str, action: ProxyAction) -> ProxyRule {
    Arc::new(move |args: &[Vec<u8>]| {
        let cmd = args.first()?.to_ascii_uppercase();
        (cmd == name.as_bytes()
            && args
                .iter()
                .any(|a| String::from_utf8_lossy(a).contains(":ext:data:")))
        .then(|| action.clone())
    })
}

/// Run one pass whose Valkey client goes through a proxy applying `rule`.
async fn pass_through_proxy(rig: &Rig, rule: ProxyRule) -> (PassSummary, usize) {
    let proxy = ScriptReplyProxy::start(test_valkey_addr(), rule).await;
    let client = client_through(&proxy).await;
    let summary = rig.pass(&rig.repair_on(client, timing())).await;
    (summary, proxy.hits())
}

/// Three owed values the backup store lacks, stored before now.
async fn three_owed(rig: &Rig, first_tag: u16) -> Vec<Value> {
    let values: Vec<Value> = (0..3u16)
        .map(|i| value(first_tag + i, 2 * KIB as usize))
        .collect();
    let refs: Vec<&Value> = values.iter().collect();
    rig.store_unmirrored(&refs).await;
    rig.wait_until_stored_before_now(&refs).await;
    values
}

/// A failed size read (`STRLEN`) is `resend_failed` for that value alone: the
/// pass goes on to the next one, so all three are tried, none uploaded, all
/// owed, and no charge is held.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_failed_size_read_fails_that_value_alone_and_the_pass_goes_on() {
    let rig = Rig::new(RepairStore::new(), 1 << 20).await;
    three_owed(&rig, 120).await;
    let (summary, hits) = pass_through_proxy(
        &rig,
        ext_data_command(
            "STRLEN",
            ProxyAction::Eval(error_reply_script("ERR staged STRLEN failure")),
        ),
    )
    .await;
    assert_eq!(hits, 3, "fixture: each value's size read was failed");
    assert_eq!(
        (summary.resend_failed, summary.resent, summary.owed),
        (3, 0, 3),
        "{summary:?}"
    );
    assert_eq!(rig.repair_count("resend_failed"), 3);
    assert!(rig.store.uploaded().is_empty());
    rig.assert_budget_whole("after the pass");
}

/// A failed read of the bytes (`GETRANGE`; each 2 KiB value is one range) is
/// `resend_failed` for that value alone, and its reservation's charge returns.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_failed_read_of_the_bytes_fails_that_value_alone_and_returns_its_charge() {
    let rig = Rig::new(RepairStore::new(), 1 << 20).await;
    three_owed(&rig, 125).await;
    let (summary, hits) = pass_through_proxy(
        &rig,
        ext_data_command(
            "GETRANGE",
            ProxyAction::Eval(error_reply_script("ERR staged GETRANGE failure")),
        ),
    )
    .await;
    assert_eq!(hits, 3, "fixture: each value's read was failed");
    assert_eq!(
        (summary.resend_failed, summary.resent, summary.owed),
        (3, 0, 3),
        "{summary:?}"
    );
    assert!(rig.store.uploaded().is_empty());
    rig.assert_budget_whole("after the pass");
}

/// A value whose length changed between its size read and its read (staged:
/// each value's first `STRLEN`, the size read, answers 1; the read's closing
/// `STRLEN` sees the real length) is not uploaded: `resend_failed` for that
/// value alone.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_length_that_changed_since_the_size_read_fails_that_value_alone() {
    let rig = Rig::new(RepairStore::new(), 1 << 20).await;
    three_owed(&rig, 130).await;
    let staged = Arc::new(Mutex::new(std::collections::HashSet::<Vec<u8>>::new()));
    let first_size_read: ProxyRule = Arc::new(move |args: &[Vec<u8>]| {
        let key = args.get(1)?;
        (args.first()?.eq_ignore_ascii_case(b"STRLEN")
            && String::from_utf8_lossy(key).contains(":ext:data:")
            && staged.lock().expect("staged keys").insert(key.clone()))
        .then(|| ProxyAction::Eval("return 1".to_owned()))
    });
    let (summary, hits) = pass_through_proxy(&rig, first_size_read).await;
    assert_eq!(hits, 3, "fixture: each value's size read was staged");
    assert_eq!(
        (summary.resend_failed, summary.resent, summary.owed),
        (3, 0, 3),
        "{summary:?}"
    );
    assert!(
        rig.store.uploaded().is_empty(),
        "nothing of the wrong size is uploaded"
    );
    rig.assert_budget_whole("after the pass");
}

/// A failed upload ends re-sending for the pass: one `resend_failed`, one
/// upload attempted, all three values owed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_failed_upload_ends_resending_for_the_pass() {
    let store = RepairStore::new();
    store.fail_next_uploads(usize::MAX);
    let rig = Rig::new(store, 1 << 20).await;
    three_owed(&rig, 135).await;

    let summary = rig.pass(&rig.repair(timing())).await;

    assert_eq!(
        (summary.resend_failed, summary.resent, summary.owed),
        (1, 0, 3),
        "{summary:?}"
    );
    assert_eq!(rig.store.uploaded().len(), 1, "one upload attempted");
    rig.assert_budget_whole("after the pass");
}

/// T21: an upload held past the timing's upload ceiling (300 ms) ends
/// re-sending for the pass, as a failed one does: one upload attempted, all
/// three values owed. The call ceiling is 60 s, longer than the pass's
/// [`WAIT_BUDGET`], so an upload bounded by the call ceiling (or by the longer
/// of the two) would run the pass into that budget instead. The recording
/// store keeps no handle on the bytes of the upload the pass stopped waiting
/// for, so its charge is back once the pass ends.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_upload_past_the_upload_ceiling_ends_resending_for_the_pass() {
    let rig = Rig::new(RepairStore::with_gates(false, true), 1 << 20).await;
    three_owed(&rig, 140).await;
    let call_ceiling = Duration::from_secs(60);
    assert!(call_ceiling > WAIT_BUDGET, "fixture");
    let repair = ExtBackupRepair::new(
        rig.client().clone(),
        rig.keys().clone(),
        rig.backup_store(),
        rig.sender.clone(),
        STORE_ID.to_owned(),
        call_ceiling,
        Some(Arc::clone(&rig.metrics)),
    )
    .with_timing(ExtBackupRepairTiming {
        upload_ceiling: Duration::from_millis(300),
        ..timing()
    });

    let summary = rig.pass(&repair).await;

    assert_eq!(
        (summary.resend_failed, summary.resent, summary.owed),
        (1, 0, 3),
        "{summary:?}"
    );
    assert_eq!(rig.repair_count("resend_failed"), 1);
    assert_eq!(rig.store.uploaded().len(), 1, "one upload attempted");
    rig.assert_budget_whole("after the pass");
    rig.store.open_uploads();
}

/// T22: the upload is bounded by the upload ceiling, not the adapter call
/// ceiling, which on the local adapters bounds a rename. An upload that takes
/// longer than the call ceiling (300 ms here: the gate opens a second after
/// the upload started) and finishes within the upload ceiling (the rig's 5 s)
/// is `resent` and confirmed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_upload_slower_than_the_call_ceiling_within_the_upload_ceiling_is_resent() {
    let rig = Rig::new(RepairStore::with_gates(false, true), 1 << 20).await;
    let v = value(145, 2 * KIB as usize);
    rig.store_unmirrored(&[&v]).await;
    rig.wait_until_stored_before_now(&[&v]).await;
    let repair = ExtBackupRepair::new(
        rig.client().clone(),
        rig.keys().clone(),
        rig.backup_store(),
        rig.sender.clone(),
        STORE_ID.to_owned(),
        Duration::from_millis(300),
        Some(Arc::clone(&rig.metrics)),
    )
    .with_timing(timing());
    assert_eq!(repair.timing().upload_ceiling, CALL_CEILING, "fixture");
    let pass = tokio::spawn(async move { repair.run_pass(&CancellationToken::new()).await });
    wait_for("the upload is in flight", || {
        rig.store.uploads.in_flight.load(Ordering::SeqCst) == 1
    })
    .await;
    // Not a wait for an event: the upload is made slower than the call
    // ceiling. A sleep lasts at least this long, so the call ceiling has
    // always passed when the gate opens.
    tokio::time::sleep(Duration::from_secs(1)).await;
    rig.store.open_uploads();

    let summary = tokio::time::timeout(WAIT_BUDGET, pass)
        .await
        .expect("the pass ends")
        .expect("the pass task does not panic")
        .expect("the pass completes");
    assert_eq!(
        (summary.resent, summary.resend_failed, summary.owed),
        (1, 0, 0),
        "{summary:?}"
    );
    assert_eq!(
        rig.store.held(&v.hex).await.as_deref(),
        Some(v.bytes.as_slice())
    );
    assert_confirmed_under(rig.confirmation(&v.hex).await.as_deref(), STORE_ID, "slow");
    rig.assert_budget_whole("after the pass");
}

/// A failed existence check ends checking and re-sending for the pass, but the
/// walk finishes, across chunks, so the owed reading counts every value: of
/// 130 owed values (two chunks), at most one probe batch is checked, none is
/// uploaded, and all 130 are owed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_failed_existence_check_stops_checking_but_the_walk_counts_every_owed_value() {
    let store = RepairStore::new();
    store.fail_next_checks(usize::MAX);
    let rig = Rig::new(store, 1 << 20).await;
    const VALUES: u16 = 130;
    let values: Vec<Value> = (0..VALUES).map(|i| value(2_000 + i, 64)).collect();
    let refs: Vec<&Value> = values.iter().collect();
    rig.store_unmirrored(&refs).await;
    rig.wait_until_stored_before_now(&refs).await;

    let summary = rig.pass(&rig.repair(timing())).await;

    assert_eq!(
        (
            summary.metas,
            summary.owed_seen,
            summary.probe_failed,
            summary.owed
        ),
        (130, 130, 1, 130),
        "{summary:?}"
    );
    assert_eq!(rig.repair_count("probe_failed"), 1);
    assert!(
        rig.store.checked().len()
            <= live_edit::checkpoint::ext_backup_repair::EXT_BACKUP_PROBE_CONCURRENCY,
        "checking stopped at the first failure: {} checks",
        rig.store.checked().len()
    );
    assert!(rig.store.uploaded().is_empty());
}

/// An existence check that outlasts the call ceiling (300 ms; the checks are
/// held) is a failed check: `probe_failed` once, checking and re-sending end
/// for the pass, and every value stays owed, none uploaded.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_existence_check_past_the_call_ceiling_is_probe_failed_and_every_value_stays_owed() {
    let rig = Rig::new(RepairStore::with_gates(true, false), 1 << 20).await;
    three_owed(&rig, 200).await;
    let repair = ExtBackupRepair::new(
        rig.client().clone(),
        rig.keys().clone(),
        rig.backup_store(),
        rig.sender.clone(),
        STORE_ID.to_owned(),
        Duration::from_millis(300),
        Some(Arc::clone(&rig.metrics)),
    )
    .with_timing(timing());

    let summary = rig.pass(&repair).await;

    assert_eq!(
        (
            summary.owed_seen,
            summary.probe_failed,
            summary.already_stored,
            summary.resent,
            summary.owed
        ),
        (3, 1, 0, 0, 3),
        "{summary:?}"
    );
    assert_eq!(rig.repair_count("probe_failed"), 1);
    assert!(rig.store.uploaded().is_empty(), "no upload");
    rig.store.open_checks();
}

/// A failed check also stops re-sending the values the pass already found
/// absent: of two owed values the store lacks, one's check answers absent
/// and only then does the other's fail. The absent one is not uploaded and
/// both stay owed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_failed_check_also_stops_resending_a_value_already_found_absent() {
    let rig = Rig::new(RepairStore::new(), 1 << 20).await;
    let absent = value(205, 2 * KIB as usize);
    let failing = value(206, 2 * KIB as usize);
    rig.store_unmirrored(&[&absent, &failing]).await;
    rig.wait_until_stored_before_now(&[&absent, &failing]).await;
    rig.store.fail_check_after_an_absent_answer(&failing.hex);

    let summary = rig.pass(&rig.repair(timing())).await;

    assert_eq!(
        rig.store.absent_answers.load(Ordering::SeqCst),
        1,
        "fixture: one check answered absent before the other failed"
    );
    assert_eq!(
        (
            summary.owed_seen,
            summary.probe_failed,
            summary.resent,
            summary.owed
        ),
        (2, 1, 0, 2),
        "{summary:?}"
    );
    assert!(
        rig.store.uploaded().is_empty(),
        "the value found absent is not re-sent after the failed check"
    );
    assert_eq!(rig.confirmation(&absent.hex).await, None);
}

/// A confirmation that fails during the daily audit (staged: the confirm
/// script's call answers an error) is counted (`mark_failed{op="confirm"}`)
/// and leaves the old confirmation in place: the value stays in the audit
/// class, not owed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_confirmation_failing_in_the_audit_leaves_the_old_one_in_place() {
    let rig = Rig::new(RepairStore::new(), 1 << 20).await;
    let v = value(150, 2 * KIB as usize);
    rig.store_unmirrored(&[&v]).await;
    rig.store.seed(&v.hex, &v.bytes).await;
    let old = confirm_aged(&rig, &v, DAY_MS + 3_600_000).await;
    rig.wait_until_stored_before_now(&[&v]).await;
    let needle = v.hex.clone();
    let rule: ProxyRule = Arc::new(move |args: &[Vec<u8>]| {
        let cmd = args.first()?.to_ascii_uppercase();
        ((cmd == b"EVALSHA" || cmd == b"EVAL")
            && args
                .iter()
                .any(|a| String::from_utf8_lossy(a).contains(needle.as_str())))
        .then(|| ProxyAction::Eval(error_reply_script("ERR staged confirm failure")))
    });

    let (summary, hits) = pass_through_proxy(&rig, rule).await;

    assert!(hits >= 1, "fixture: the confirmation was failed");
    assert_eq!(
        (summary.audited, summary.already_stored, summary.owed),
        (1, 1, 0),
        "{summary:?}"
    );
    assert_eq!(rig.mark_failed("confirm"), 1);
    assert_eq!(
        rig.confirmation(&v.hex).await.as_deref(),
        Some(old.as_str()),
        "the old confirmation stays"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Concurrency bounds (T17)
// ─────────────────────────────────────────────────────────────────────────────

/// T17: the pass runs at most `EXT_BACKUP_PROBE_CONCURRENCY` (8) existence
/// checks at once, and reaches that many, and uploads one value at a time.
/// Checks and uploads are held at their gates until the bound is reached, so
/// a looser bound would show in the peak.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn checks_run_at_most_eight_at_once_and_uploads_one_at_a_time() {
    let rig = Rig::new(RepairStore::with_gates(true, true), 1 << 20).await;
    let values: Vec<Value> = (0..20u16).map(|i| value(160 + i, KIB as usize)).collect();
    let refs: Vec<&Value> = values.iter().collect();
    rig.store_unmirrored(&refs).await;
    rig.wait_until_stored_before_now(&refs).await;
    let repair = rig.repair(timing());
    let pass = tokio::spawn(async move { repair.run_pass(&CancellationToken::new()).await });

    let bound = live_edit::checkpoint::ext_backup_repair::EXT_BACKUP_PROBE_CONCURRENCY;
    wait_for("the checks reach the bound", || {
        rig.store.checks.in_flight.load(Ordering::SeqCst) == bound
    })
    .await;
    rig.store.open_checks();
    wait_for("an upload is in flight", || {
        rig.store.uploads.in_flight.load(Ordering::SeqCst) == 1
    })
    .await;
    rig.store.open_uploads();
    let summary = tokio::time::timeout(WAIT_BUDGET, pass)
        .await
        .expect("the pass ends")
        .expect("the pass task does not panic")
        .expect("the pass completes");

    assert_eq!(summary.resent, 20, "{summary:?}");
    assert_eq!(rig.store.checks.peak.load(Ordering::SeqCst), bound);
    assert_eq!(rig.store.uploads.peak.load(Ordering::SeqCst), 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// The mirror worker's confirmation (T19)
// ─────────────────────────────────────────────────────────────────────────────

/// T19: the mirror worker confirms a copy only once its upload has returned
/// successfully: no confirmation while the upload is in flight, one under
/// this store's identity after it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_mirror_worker_confirms_a_copy_only_after_its_upload_succeeds() {
    let mut rig = Rig::new(RepairStore::with_gates(false, true), 1 << 20).await;
    let cancel = CancellationToken::new();
    let workers = rig.start_workers(1, &cancel);
    let v = value(170, 2 * KIB as usize);
    rig.mirrored()
        .store_values(&[(&v.hash, v.bytes.as_slice())])
        .await
        .expect("store_values succeeds");
    wait_for("the worker's upload is in flight", || {
        rig.store.uploads.in_flight.load(Ordering::SeqCst) == 1
    })
    .await;
    assert_eq!(
        rig.confirmation(&v.hex).await,
        None,
        "not confirmed while the upload is in flight"
    );

    rig.store.open_uploads();
    wait_confirmed(&rig, &v.hex).await;
    let ms = assert_confirmed_under(
        rig.confirmation(&v.hex).await.as_deref(),
        STORE_ID,
        "mirror",
    );
    let stamp: u64 = rig
        .stamp(&v.hex)
        .await
        .expect("stamped")
        .parse()
        .expect("decimal");
    assert!(ms >= stamp, "confirmed after the store: {ms} >= {stamp}");
    shut_down(cancel, workers).await;
}

/// T19: a store of the same hash after the copy was made (here its stamp is
/// moved while the worker's upload is in flight) declines the worker's
/// confirmation: the copy is left to the repair's next check.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_mirror_worker_declines_to_confirm_a_copy_older_than_the_metas_last_store() {
    let mut rig = Rig::new(RepairStore::with_gates(false, true), 1 << 20).await;
    let cancel = CancellationToken::new();
    let workers = rig.start_workers(1, &cancel);
    let v = value(171, 2 * KIB as usize);
    let sentinel = value(172, 2 * KIB as usize);
    let ext = rig.mirrored();
    ext.store_values(&[(&v.hash, v.bytes.as_slice())])
        .await
        .expect("store_values succeeds");
    wait_for("the worker's upload is in flight", || {
        rig.store.uploads.in_flight.load(Ordering::SeqCst) == 1
    })
    .await;
    rig.set_meta_field(&v.hex, "last_store_ms", "1".to_owned())
        .await;
    rig.store.open_uploads();
    // One worker takes items in order: once the next copy is confirmed, the
    // first one's confirmation has been tried.
    ext.store_values(&[(&sentinel.hash, sentinel.bytes.as_slice())])
        .await
        .expect("store_values succeeds");
    wait_confirmed(&rig, &sentinel.hex).await;

    assert_eq!(rig.confirmation(&v.hex).await, None, "declined");
    assert_eq!(rig.mark_failed("confirm"), 0, "a decline is not a failure");
    shut_down(cancel, workers).await;
}

// ─────────────────────────────────────────────────────────────────────────────
// The recovery path (b.mhq F3)
// ─────────────────────────────────────────────────────────────────────────────

/// A value written back to Valkey by fetch-on-miss recovery
/// (`ExtRecoveryRewrite::store_recovered_values`, `ExtMirroring::Skip`) sends
/// the mirror no copy, so no worker uploads or confirms it; one pass checks
/// it, finds the bytes it was recovered from, and confirms it
/// (`already_stored`) with no upload.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_value_recovered_from_the_backup_store_is_confirmed_by_one_pass_with_no_upload() {
    let mut rig = Rig::new(RepairStore::new(), 1 << 20).await;
    let cancel = CancellationToken::new();
    let workers = rig.start_workers(1, &cancel);
    let v = value(180, 3 * KIB as usize);
    rig.store.seed(&v.hex, &v.bytes).await;
    let ext = rig.mirrored().with_checkpoint_store(rig.backup_store());

    let read = ext
        .read_batch(&[v.hash])
        .await
        .expect("the miss recovers from the backup store");
    assert_eq!(
        read.get(&v.hash).map(Vec::as_slice),
        Some(v.bytes.as_slice()),
        "fixture: recovered"
    );
    assert!(
        rig.exists(rig.keys().ext_meta(&v.hex)).await,
        "fixture: the recovery wrote the value back"
    );
    rig.assert_budget_whole("no mirror copy was sent");
    assert!(rig.store.uploaded().is_empty(), "no worker upload");
    assert_eq!(
        rig.confirmation(&v.hex).await,
        None,
        "no worker confirmation"
    );

    rig.wait_until_stored_before_now(&[&v]).await;
    let summary = rig.pass(&rig.repair(timing())).await;

    assert_eq!(
        (summary.already_stored, summary.resent, summary.owed),
        (1, 0, 0),
        "{summary:?}"
    );
    assert_eq!(rig.store.checked(), vec![v.hex.clone()]);
    assert!(rig.store.uploaded().is_empty(), "no upload");
    assert_confirmed_under(
        rig.confirmation(&v.hex).await.as_deref(),
        STORE_ID,
        "recovered",
    );
    shut_down(cancel, workers).await;
}

// ─────────────────────────────────────────────────────────────────────────────
// The ranged read (F1): one buffer of the value's length, one range at a time
// ─────────────────────────────────────────────────────────────────────────────

/// [`EXT_VALUE_READ_CHUNK_BYTES`], the most one range of the read asks for.
const CHUNK: u64 = EXT_VALUE_READ_CHUNK_BYTES as u64;

/// A value of `len` bytes whose every position differs from its neighbours,
/// so a range read at the wrong offset, or joined out of order, does not
/// compare equal.
fn patterned(tag: u16, len: usize) -> Value {
    let bytes: Vec<u8> = (0..len).map(|i| ((i % 251) as u8) ^ (tag as u8)).collect();
    let hash = ext_content_hash(&bytes);
    Value {
        hash,
        hex: sha256_to_hex(&hash),
        bytes,
    }
}

/// Store `bytes` under `hash_hex`'s data key alone, with no meta: all the
/// reader reads.
async fn set_data(rig: &Rig, hash_hex: &str, bytes: &[u8]) {
    let _: () = rig
        .client()
        .pool
        .set(
            rig.keys().ext_data(hash_hex),
            bytes.to_vec(),
            None,
            None,
            false,
        )
        .await
        .expect("SET ext:data");
}

/// `read_ext_value_bytes` of `hash_hex`, expecting `len` bytes.
async fn read_value(rig: &Rig, hash_hex: &str, len: u64) -> ExtValueRead {
    read_ext_value_bytes(rig.client(), rig.keys(), hash_hex, len)
        .await
        .expect("the read's Valkey calls succeed")
}

/// The stored length a [`ExtValueRead::Changed`] reports; panics on any other
/// answer.
fn changed_to(read: ExtValueRead, what: &str) -> u64 {
    match read {
        ExtValueRead::Changed { now_len } => now_len,
        ExtValueRead::Bytes(b) => panic!("{what}: read {} bytes, not Changed", b.len()),
        ExtValueRead::Gone => panic!("{what}: Gone, not Changed"),
    }
}

/// A `GETRANGE` frame's key, start and end, when `args` is one.
fn getrange_of(args: &[Vec<u8>]) -> Option<(Vec<u8>, u64, u64)> {
    if !args.first()?.eq_ignore_ascii_case(b"GETRANGE") {
        return None;
    }
    let number = |arg: &Vec<u8>| String::from_utf8_lossy(arg).parse::<u64>().ok();
    Some((
        args.get(1)?.clone(),
        number(args.get(2)?)?,
        number(args.get(3)?)?,
    ))
}

/// F1: a value of two and a half ranges is read whole, into one buffer of
/// exactly its length that no other handle shares: what the read holds is
/// the value, which the re-send's charge was taken for, and not a reply
/// buffer around it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_ranged_read_holds_a_value_in_one_unshared_buffer_of_exactly_its_length() {
    let rig = Rig::new(RepairStore::new(), 1 << 20).await;
    let v = patterned(300, (5 * CHUNK / 2) as usize);
    set_data(&rig, &v.hex, &v.bytes).await;

    let bytes = match read_value(&rig, &v.hex, v.bytes.len() as u64).await {
        ExtValueRead::Bytes(bytes) => bytes,
        other => panic!("the stored value reads whole: {other:?}"),
    };

    assert!(
        bytes.as_ref() == v.bytes.as_slice(),
        "the value, whole and in order ({} of {} bytes read)",
        bytes.len(),
        v.bytes.len()
    );
    let buffer = bytes
        .try_into_mut()
        .unwrap_or_else(|_| panic!("the read's buffer is shared with another handle"));
    assert_eq!(
        buffer.capacity(),
        v.bytes.len(),
        "the buffer is exactly the value's length"
    );
}

/// F1: a stored value of another length than the read expects is `Changed`,
/// reporting the stored length: a shorter one answers a short range (the
/// first, or in a longer read the second), a longer one fails the closing
/// `STRLEN`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_ranged_read_answers_changed_with_the_stored_length_when_it_differs() {
    let rig = Rig::new(RepairStore::new(), 1 << 20).await;
    let v = patterned(301, 1_000);
    set_data(&rig, &v.hex, &v.bytes).await;
    assert_eq!(
        changed_to(read_value(&rig, &v.hex, 2_000).await, "expected longer"),
        1_000
    );
    assert_eq!(
        changed_to(read_value(&rig, &v.hex, 500).await, "expected shorter"),
        1_000
    );

    let two_ranges = patterned(302, (CHUNK + 10) as usize);
    set_data(&rig, &two_ranges.hex, &two_ranges.bytes).await;
    assert_eq!(
        changed_to(
            read_value(&rig, &two_ranges.hex, 2 * CHUNK).await,
            "short at the second range"
        ),
        CHUNK + 10
    );
}

/// F1: an absent value is `Gone`, whatever length the read expects, zero
/// included.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_ranged_read_answers_gone_for_an_absent_value_whatever_length_it_expects() {
    let rig = Rig::new(RepairStore::new(), 1 << 20).await;
    let absent = patterned(303, 16);
    for len in [0, 10, 2 * CHUNK] {
        let read = read_value(&rig, &absent.hex, len).await;
        assert!(matches!(read, ExtValueRead::Gone), "len {len}: {read:?}");
    }
}

/// F1: an absent value and an empty one both read length 0; the read tells
/// them apart. An empty value expected empty reads as its zero bytes, an
/// empty value expected longer is `Changed` to 0, and a non-empty value
/// expected empty is `Changed` to its length (an absent one is `Gone`, the
/// test above).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_ranged_read_tells_an_empty_value_from_an_absent_one() {
    let rig = Rig::new(RepairStore::new(), 1 << 20).await;
    let empty = patterned(304, 0);
    set_data(&rig, &empty.hex, &[]).await;
    let read = read_value(&rig, &empty.hex, 0).await;
    assert!(
        matches!(&read, ExtValueRead::Bytes(b) if b.is_empty()),
        "an empty value reads empty: {read:?}"
    );
    assert_eq!(
        changed_to(read_value(&rig, &empty.hex, 5).await, "empty, expected 5"),
        0
    );

    let three = patterned(305, 3);
    set_data(&rig, &three.hex, &three.bytes).await;
    assert_eq!(
        changed_to(read_value(&rig, &three.hex, 0).await, "3 bytes, expected 0"),
        3
    );
}

/// F1: a re-send reads its value one range at a time, each of at most
/// [`EXT_VALUE_READ_CHUNK_BYTES`], back to back from 0 to the value's end, and
/// re-sends it whole. No reply is larger than one range, so no read buffer
/// outgrows the connection's counted one.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_resend_reads_its_value_one_range_of_at_most_a_chunk_at_a_time() {
    let rig = Rig::new(RepairStore::new(), 1 << 20).await;
    let v = patterned(310, (7 * CHUNK / 2) as usize);
    rig.store_unmirrored(&[&v]).await;
    rig.wait_until_stored_before_now(&[&v]).await;
    let data_key = rig.keys().ext_data(&v.hex).into_bytes();
    let ranges: Arc<Mutex<Vec<(u64, u64)>>> = Arc::default();
    let rule: ProxyRule = Arc::new({
        let ranges = Arc::clone(&ranges);
        move |args: &[Vec<u8>]| {
            if let Some((key, start, end)) = getrange_of(args) {
                if key == data_key {
                    ranges.lock().expect("never poisoned").push((start, end));
                }
            }
            None
        }
    });
    let proxy = ScriptReplyProxy::start(test_valkey_addr(), rule).await;
    let client = client_through(&proxy).await;

    let summary = rig.pass(&rig.repair_on(client, timing())).await;

    assert_eq!((summary.resent, summary.owed), (1, 0), "{summary:?}");
    let len = v.bytes.len() as u64;
    let expected: Vec<(u64, u64)> = (0..len)
        .step_by(EXT_VALUE_READ_CHUNK_BYTES)
        .map(|start| (start, (start + CHUNK).min(len) - 1))
        .collect();
    assert_eq!(expected.len(), 4, "fixture: four ranges, the last a half");
    assert_eq!(
        *ranges.lock().expect("never poisoned"),
        expected,
        "one range of at most a chunk at a time, in order"
    );
    let held = rig.store.held(&v.hex).await.expect("the copy is stored");
    assert!(held.as_ref() == v.bytes.as_slice(), "the copy is the value");
    assert_confirmed_under(
        rig.confirmation(&v.hex).await.as_deref(),
        STORE_ID,
        "ranged",
    );
    rig.assert_budget_whole("after the pass");
}

/// A rule staging, in place of the second range of `v`'s read, a script
/// that runs `before` and then reads that range.
fn at_second_range(rig: &Rig, v: &Value, before: String) -> ProxyRule {
    let data_key = rig.keys().ext_data(&v.hex);
    let data_bytes = data_key.clone().into_bytes();
    Arc::new(move |args: &[Vec<u8>]| {
        let (key, start, end) = getrange_of(args)?;
        (key == data_bytes && start == CHUNK).then(|| {
            ProxyAction::Eval(format!(
                "{before} return redis.call('GETRANGE', '{data_key}', {start}, {end})"
            ))
        })
    })
}

/// F1: a value the GC collects between two ranges of its read (staged: the
/// second range's `GETRANGE` runs after a `DEL` of the value's meta and data)
/// is `value_gone`: nothing is uploaded, the charge returns, and no meta is
/// created for it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_value_collected_between_two_ranges_of_its_read_is_value_gone() {
    let rig = Rig::new(RepairStore::new(), 1 << 20).await;
    let v = patterned(320, (7 * CHUNK / 2) as usize);
    rig.store_unmirrored(&[&v]).await;
    rig.wait_until_stored_before_now(&[&v]).await;
    let collect = format!(
        "redis.call('DEL', '{}', '{}')",
        rig.keys().ext_meta(&v.hex),
        rig.keys().ext_data(&v.hex)
    );

    let (summary, hits) = pass_through_proxy(&rig, at_second_range(&rig, &v, collect)).await;

    assert_eq!(hits, 1, "fixture: the collect landed after the first range");
    assert_eq!(
        (
            summary.value_gone,
            summary.resent,
            summary.resend_failed,
            summary.owed
        ),
        (1, 0, 0, 0),
        "{summary:?}"
    );
    assert_eq!(rig.repair_count("value_gone"), 1);
    assert!(rig.store.uploaded().is_empty(), "nothing is uploaded");
    assert!(!rig.exists(rig.keys().ext_meta(&v.hex)).await, "no meta");
    rig.assert_budget_whole("after the pass");
}

/// F1: a value rewritten at another length between two ranges of its read
/// (staged: before the second range, it is cut to its first range and 100
/// bytes more, as a heal rewrites a damaged copy) is `resend_failed` for that
/// value alone: its read ends at the short range, none of it is uploaded, its
/// charge returns, and the pass re-sends the two other owed values.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_value_rewritten_shorter_between_two_ranges_fails_that_value_alone() {
    let rig = Rig::new(RepairStore::new(), 1 << 20).await;
    let big = patterned(330, (7 * CHUNK / 2) as usize);
    let small = [value(331, 2 * KIB as usize), value(332, 2 * KIB as usize)];
    rig.store_unmirrored(&[&big, &small[0], &small[1]]).await;
    rig.wait_until_stored_before_now(&[&big, &small[0], &small[1]])
        .await;
    let data_key = rig.keys().ext_data(&big.hex);
    let shrink = format!(
        "local v = redis.call('GET', '{data_key}') \
         redis.call('SET', '{data_key}', string.sub(v, 1, {}))",
        CHUNK + 100
    );

    let (summary, hits) = pass_through_proxy(&rig, at_second_range(&rig, &big, shrink)).await;

    assert_eq!(hits, 1, "fixture: the rewrite landed after the first range");
    assert_eq!(
        (
            summary.resend_failed,
            summary.resent,
            summary.value_gone,
            summary.owed
        ),
        (1, 2, 0, 1),
        "{summary:?}"
    );
    let mut uploaded = rig.store.uploaded();
    uploaded.sort();
    let mut expected: Vec<String> = small.iter().map(|v| v.hex.clone()).collect();
    expected.sort();
    assert_eq!(uploaded, expected, "only the two other values are uploaded");
    assert_eq!(rig.confirmation(&big.hex).await, None, "stays owed");
    rig.assert_budget_whole("after the pass");
}

// ─────────────────────────────────────────────────────────────────────────────
// The time budget (F2): a pass cut short still publishes its full owed count
// ─────────────────────────────────────────────────────────────────────────────

/// The timing of the time-budget tests: a 2 s budget for checks and re-sends,
/// well inside a 6 s interval, itself well inside [`WAIT_BUDGET`]. The budget
/// runs from the walk's start, so it is seconds long: the walk reaches its
/// first chunk's checks (after a `SCAN` of the shared test keyspace) long
/// before it runs out, even on a loaded runner, and it is the held checks or
/// uploads that keep the work going until it does. A budget that never cut
/// the pass would leave it to the deadline, which publishes nothing, before
/// the test's own wait ran out.
fn budgeted() -> ExtBackupRepairTiming {
    ExtBackupRepairTiming {
        interval: Duration::from_secs(6),
        work_budget: Duration::from_secs(2),
        ..timing()
    }
}

/// More owed values than one meta chunk holds, so the walk reads a second
/// chunk after the time budget has run out: [`META_CHUNK`] + 12.
const PAST_ONE_CHUNK: u16 = META_CHUNK as u16 + 12;

/// `n` owed values of 1 KiB the backup store lacks, stored before now.
async fn owed_values(rig: &Rig, first_tag: u16, n: u16) -> Vec<Value> {
    let values: Vec<Value> = (0..n).map(|i| value(first_tag + i, KIB as usize)).collect();
    let refs: Vec<&Value> = values.iter().collect();
    rig.store_unmirrored(&refs).await;
    rig.wait_until_stored_before_now(&refs).await;
    values
}

/// F2: a pass whose time budget runs out while its checks are held stops
/// checking, finishes its walk and completes, before its deadline. Of
/// [`PAST_ONE_CHUNK`] owed values, at most 8 checks were ever sent (and are
/// dropped), the rest of the first chunk is never checked, and the second
/// chunk is read only after the budget ran out: the owed reading counts every
/// value of both chunks. A walk that stopped with the work would count only
/// the first chunk.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_pass_whose_time_budget_runs_out_mid_check_still_counts_every_owed_value() {
    let rig = Rig::new(RepairStore::with_gates(true, false), 1 << 20).await;
    owed_values(&rig, 400, PAST_ONE_CHUNK).await;

    let summary = rig.pass(&rig.repair(budgeted())).await;

    let n = u64::from(PAST_ONE_CHUNK);
    assert!(summary.work_budget_spent, "{summary:?}");
    assert_eq!(
        (
            summary.metas,
            summary.owed_seen,
            summary.owed,
            summary.probe_failed,
            summary.already_stored,
            summary.resent
        ),
        (n, n, n, 0, 0, 0),
        "{summary:?}"
    );
    assert!(
        rig.store.checked().len() <= EXT_BACKUP_PROBE_CONCURRENCY,
        "only the first checks were ever sent: {}",
        rig.store.checked().len()
    );
    assert!(rig.store.uploaded().is_empty());
    rig.store.open_checks();
}

/// F2: a pass whose time budget runs out during an upload drops the upload,
/// stops re-sending and completes: of three owed values the store lacks, one
/// upload was attempted, none re-sent or failed, all three are owed, and the
/// dropped upload's charge is back.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_pass_whose_time_budget_runs_out_mid_upload_still_counts_every_owed_value() {
    let rig = Rig::new(RepairStore::with_gates(false, true), 1 << 20).await;
    three_owed(&rig, 410).await;

    let summary = rig.pass(&rig.repair(budgeted())).await;

    assert!(summary.work_budget_spent, "{summary:?}");
    assert_eq!(
        (summary.resent, summary.resend_failed, summary.owed),
        (0, 0, 3),
        "{summary:?}"
    );
    assert_eq!(rig.store.uploaded().len(), 1, "one upload attempted");
    rig.assert_budget_whole("after the pass");
    rig.store.open_uploads();
}

/// F2: a leader's pass cut by its time budget publishes its full owed count:
/// all [`PAST_ONE_CHUNK`] values, unchecked ones included, the second meta
/// chunk read only after the budget ran out among them. The gauge reports
/// the backlog while the backup store is too slow to work it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_leaders_pass_cut_by_its_time_budget_publishes_the_full_owed_count() {
    let rig = Rig::new(RepairStore::with_gates(true, false), 1 << 20).await;
    owed_values(&rig, 600, PAST_ONE_CHUNK).await;

    let cancel = CancellationToken::new();
    let task = spawn_ext_backup_repair(
        rig.repair(budgeted()),
        LeaderIdentity::new(SERVICE_LABEL, "instance-a"),
        cancel.clone(),
    );
    wait_for("the owed gauge is published", || rig.owed_gauge().is_some()).await;
    shut_down(cancel, vec![task]).await;

    assert_eq!(
        rig.owed_gauge(),
        Some(u64::from(PAST_ONE_CHUNK)),
        "every owed value is counted"
    );
    assert!(
        rig.store.checked().len() <= EXT_BACKUP_PROBE_CONCURRENCY,
        "only the first checks were ever sent: {}",
        rig.store.checked().len()
    );
    rig.store.open_checks();
}

// ─────────────────────────────────────────────────────────────────────────────
// An unfinished pass publishes nothing (T20)
// ─────────────────────────────────────────────────────────────────────────────

/// T20: a leader's pass cancelled mid-check (shutdown) publishes no owed
/// reading — unknown is not 0 — and holds no charge.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_pass_cancelled_mid_check_publishes_no_owed_reading() {
    let rig = Rig::new(RepairStore::with_gates(true, false), 1 << 20).await;
    let v = value(190, KIB as usize);
    rig.store_unmirrored(&[&v]).await;
    rig.wait_until_stored_before_now(&[&v]).await;

    let cancel = CancellationToken::new();
    let task = spawn_ext_backup_repair(
        rig.repair(ExtBackupRepairTiming {
            interval: LONG_INTERVAL,
            ..timing()
        }),
        LeaderIdentity::new(SERVICE_LABEL, "instance-a"),
        cancel.clone(),
    );
    wait_for("the pass is checking", || {
        rig.store.checks.in_flight.load(Ordering::SeqCst) == 1
    })
    .await;
    shut_down(cancel, vec![task]).await;

    assert_eq!(rig.owed_gauge(), None, "a cancelled pass publishes nothing");
    assert!(rig.store.uploaded().is_empty());
    rig.store.open_checks();
}

/// T20: a pass cancelled during an upload ends `Cancelled`, and the
/// reservation's charge returns with the abandoned upload's bytes: here at
/// once, since the recording store holds them only in the dropped call (a
/// LocalDisk write would hold them, and the charge, until it ends).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_pass_cancelled_mid_upload_ends_cancelled_and_returns_the_charge() {
    let rig = Rig::new(RepairStore::with_gates(false, true), 1 << 20).await;
    let v = value(191, 8 * KIB as usize);
    rig.store_unmirrored(&[&v]).await;
    rig.wait_until_stored_before_now(&[&v]).await;
    let repair = rig.repair(timing());
    let cancel = CancellationToken::new();
    let pass = tokio::spawn({
        let cancel = cancel.clone();
        async move { repair.run_pass(&cancel).await }
    });
    wait_for("the upload is in flight", || {
        rig.store.uploads.in_flight.load(Ordering::SeqCst) == 1
    })
    .await;
    assert_eq!(rig.charged(), charge_of(8 * KIB), "fixture: reserved");

    cancel.cancel();
    let ended = tokio::time::timeout(WAIT_BUDGET, pass)
        .await
        .expect("the pass ends")
        .expect("the pass task does not panic");
    assert!(matches!(ended, Err(PassError::Cancelled)), "{ended:?}");
    rig.assert_budget_whole("after the cancel");
    assert_eq!(rig.confirmation(&v.hex).await, None);
    rig.store.open_uploads();
}

/// T20: a pass that runs past its deadline (the interval) ends `Deadline`.
/// The time budget ([`timing`]'s 60 s) outlasts the 300 ms interval, so the
/// held check is cut by the deadline, the backstop a walk slower than the
/// interval reaches; a budget that runs out first is
/// [`a_pass_whose_time_budget_runs_out_mid_check_still_counts_every_owed_value`].
/// That the task publishes nothing for it is
/// [`a_leaders_pass_past_its_deadline_publishes_no_owed_reading`].
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_pass_past_its_deadline_ends_deadline() {
    let rig = Rig::new(RepairStore::with_gates(true, false), 1 << 20).await;
    let v = value(192, KIB as usize);
    rig.store_unmirrored(&[&v]).await;
    rig.wait_until_stored_before_now(&[&v]).await;
    let repair = rig.repair(ExtBackupRepairTiming {
        interval: Duration::from_millis(300),
        ..timing()
    });

    let ended = tokio::time::timeout(WAIT_BUDGET, repair.run_pass(&CancellationToken::new()))
        .await
        .expect("the pass ends");

    assert!(matches!(ended, Err(PassError::Deadline)), "{ended:?}");
    rig.store.open_checks();
}

/// A leader's pass that runs past its deadline (a 300 ms interval, shorter
/// than [`timing`]'s 60 s time budget; the checks are held) publishes no owed
/// reading. The task's second pass checking the value shows the first one has
/// ended, and with its check held only the deadline ends it; the second is
/// cancelled by the shutdown. A pass whose time budget runs out first does
/// publish: [`a_leaders_pass_cut_by_its_time_budget_publishes_the_full_owed_count`].
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_leaders_pass_past_its_deadline_publishes_no_owed_reading() {
    let rig = Rig::new(RepairStore::with_gates(true, false), 1 << 20).await;
    let v = value(193, KIB as usize);
    rig.store_unmirrored(&[&v]).await;
    rig.wait_until_stored_before_now(&[&v]).await;

    let cancel = CancellationToken::new();
    let task = spawn_ext_backup_repair(
        rig.repair(ExtBackupRepairTiming {
            interval: Duration::from_millis(300),
            ..timing()
        }),
        LeaderIdentity::new(SERVICE_LABEL, "instance-a"),
        cancel.clone(),
    );
    wait_for("a second pass is checking", || {
        rig.store.checked().len() >= 2
    })
    .await;
    shut_down(cancel, vec![task]).await;

    assert_eq!(
        rig.owed_gauge(),
        None,
        "a pass past its deadline publishes nothing"
    );
    assert!(rig.store.uploaded().is_empty());
    rig.store.open_checks();
}

/// A leader's pass whose walk fails (staged: every `HMGET` of an `ext:meta`
/// key answers an error) publishes no owed reading: the walk did not count
/// every value.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_leaders_pass_whose_walk_fails_publishes_no_owed_reading() {
    let rig = Rig::new(RepairStore::new(), 1 << 20).await;
    let v = value(194, KIB as usize);
    rig.store_unmirrored(&[&v]).await;
    rig.wait_until_stored_before_now(&[&v]).await;
    let rule: ProxyRule = Arc::new(|args: &[Vec<u8>]| {
        let cmd = args.first()?.to_ascii_uppercase();
        (cmd == b"HMGET"
            && args
                .iter()
                .any(|a| String::from_utf8_lossy(a).contains(":ext:meta:")))
        .then(|| ProxyAction::Eval(error_reply_script("ERR staged HMGET failure")))
    });
    let proxy = ScriptReplyProxy::start(test_valkey_addr(), rule).await;
    let client = client_through(&proxy).await;

    let cancel = CancellationToken::new();
    let task = spawn_ext_backup_repair(
        rig.repair_on(
            client,
            ExtBackupRepairTiming {
                interval: LONG_INTERVAL,
                ..timing()
            },
        ),
        LeaderIdentity::new(SERVICE_LABEL, "instance-a"),
        cancel.clone(),
    );
    // The walk's `HMGET` is not raced against the cancel, so once it has
    // been sent the pass ends on its failure whenever the shutdown lands.
    wait_for("the walk's HMGET was failed", || proxy.hits() >= 1).await;
    shut_down(cancel, vec![task]).await;

    assert_eq!(
        rig.owed_gauge(),
        None,
        "a pass whose walk failed publishes nothing"
    );
    assert!(rig.store.checked().is_empty(), "no value was checked");
}
