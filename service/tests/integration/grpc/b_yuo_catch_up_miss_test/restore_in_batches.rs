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

//! D1 — a genuine miss's restore of backup-only values before its layout, in
//! batches charged to the catch-up payload budget `D_c` (SDD §11.2, §12.3,
//! §22.22), through `handle_subscribe_read_layer` against a real Valkey and a
//! real `ValkeyExternalValueStore`, and, where only the batching is the
//! subject, through `storage::ompf_payload::restore_in_batches` directly.
//!
//! Every layer here is in the shape a disaster-recovery restore leaves: each
//! value's meta is in the store without its data, and only the backup store
//! (the checkpoint adapter, [`CountingAdapter`]) holds its bytes. So a
//! catch-up misses every value, re-reads under its `{gc}` read hold, finds
//! them still missing — genuinely — and restores them.
//!
//! | Test | Pins |
//! |---|---|
//! | T2 | [`every_value_a_restore_fetches_is_covered_by_a_share_of_the_payload_budget_reserved_before_it`] |
//! | T3 | [`a_restore_that_cannot_reserve_the_payload_budget_is_refused_catchup_capacity_before_content_holding_nothing`], [`a_restore_refused_after_its_first_batch_keeps_that_batch_stored_and_its_retry_restores_only_the_rest`] |
//! | T4 | [`six_catch_ups_restoring_over_three_layers_on_one_and_a_half_batches_of_budget_are_all_served`], [`a_join_wave_on_a_restored_layer_fetches_each_value_from_the_backup_store_once`] |
//! | TR-1, the contended join wave (the probe after a batch's reservation) | [`a_contended_join_wave_fetches_each_value_once_its_waiters_re_probing_after_their_grant`] |
//! | TR-2, a batch's grant through its re-probe, a failure and a disconnect | [`a_batch_whose_re_probe_finds_part_of_it_stored_keeps_only_the_rest_s_shares_before_any_fetch`], [`a_batch_with_one_value_the_backup_store_lost_writes_the_others_back_and_ends_internal_holding_nothing`], [`a_client_disconnecting_while_its_batch_is_fetched_leaves_nothing_held`]; the wait histogram's count per batch, T2 |
//! | T6 (b.cyy) | [`backup_only_values_are_fetched_ext_restore_parallelism_at_a_time`] |
//! | b.cyy's slot-hold A/B, on a real clock (ignored, manual, report only) | [`slot_hold_ab_is_reported`] |
//! | PD-7b | [`the_recovery_counter_labels_a_meta_without_data_present_and_an_unknown_hash_absent`], [`the_recovery_counter_is_zero_seeded_over_result_and_meta`]; at the catch-up, T2 |
//!
//! b.ntm's two pins rewritten for D1 (T5) are in the parent module:
//! `a_genuine_miss_waiting_for_the_payload_budget_holds_nothing_of_it_while_it_waits`
//! and `two_catch_ups_restoring_at_once_are_both_served_without_a_capacity_refusal`.
//! The batches' formation and charge (T9), the fill's one-at-a-time
//! restore (T7) and the fill's charge for bytes beyond a value's laid-out
//! length are pinned in `storage::ompf_payload`'s unit tests; the charge for
//! a restored value's bytes beyond its declared size, held until its
//! write-back returns and given back when the write-back fails, in
//! `storage::external_values`' `restore_excess_charge_tests`; and the restore
//! phase's heap (T1) in `tests/delivery_memory`.
//!
//! **Waits.** Every wait is a deadline ([`DEADLINE`], 30 s, per message,
//! park or polled condition), unreachable on the healthy path. Paid in full:
//! the 2 s capacity budget the two T3 refusals run out. The six-catch-up T4
//! test raises its catch-ups' capacity budget to [`WIDE_BUDGET_MS`] so the
//! time it holds the payload budget before it lets go is never what ends
//! them; the join wave stops its catch-ups in a store call, which no capacity
//! budget is charged for. TR-1 raises its catch-ups' capacity budget the same
//! way, and TR-2's disconnect polls, within [`DEADLINE`], for what its
//! catch-up held to be returned.

use std::sync::Mutex as StdMutex;

use live_edit::error::{refusal_reason, CAPACITY_RETRY_DELAY};
use live_edit::observability::metrics::{
    EXT_RECOVERY_METAS, EXT_RECOVERY_META_ABSENT, EXT_RECOVERY_META_PRESENT, EXT_RECOVERY_RESULTS,
    STREAM_EXIT_CATCHUP_CAPACITY,
};
use live_edit::storage::diff_read::RestoreLayout;
use live_edit::storage::external_values::{charge_nothing, MockExternalValueStore};
use live_edit::storage::memory_budget::{MemoryBudget, CATCHUP_PAYLOAD_SERIES};
use live_edit::storage::ompf_payload::{
    restore_charge_bytes, restore_in_batches, RestoredLength, EXT_RESTORE_CHARGE_CENTI,
    EXT_RESTORE_PARALLELISM,
};

use super::*;
use crate::common::refusal_detail::assert_refusal_detail;

/// A catch-up capacity budget wide enough that only a deadlock could run it
/// out while a T4 test holds the payload budget and then lets go.
const WIDE_BUDGET_MS: u64 = 20_000;

/// The share of the catch-up payload budget a restore of a value of `len`
/// bytes reserves: its charge at the catch-up's restore-only rate,
/// [`EXT_RESTORE_CHARGE_CENTI`] hundredths per byte, in the budget's whole
/// KiB. Not the upload path's `EXT_EXPANSION_CENTI`: T2 fails if the
/// catch-up passes that instead.
fn share(len: usize) -> u64 {
    restore_charge_bytes(len as u64, EXT_RESTORE_CHARGE_CENTI).div_ceil(1024) * 1024
}

/// `count` distinct 2 KiB values, `base` seeding the first.
fn values(base: u8, count: usize) -> Vec<Vec<u8>> {
    (0..count)
        .map(|i| blob(base.wrapping_add(i as u8)))
        .collect()
}

// ═════════════════════════════════════════════════════════════════════════════
// Fixture
// ═════════════════════════════════════════════════════════════════════════════

/// The backup store: an `InMemoryCheckpointStore` whose `get_ext_blob` — the
/// fetch-on-miss consult — counts its calls and the most in flight at once,
/// after an injected `latency` when there is one; a stalled adapter
/// ([`Self::stalled`]) holds every fetch in flight until [`Self::unstall`]
/// lets it answer. Every other method is forwarded, the defaulted ones
/// included: an omitted arm would substitute the trait's default, not pass
/// through.
struct CountingAdapter {
    inner: InMemoryCheckpointStore,
    latency: Option<Duration>,
    stall: Option<tokio::sync::Semaphore>,
    fetches: AtomicUsize,
    in_flight: AtomicUsize,
    max_in_flight: AtomicUsize,
}

impl CountingAdapter {
    fn new(latency: Option<Duration>) -> Self {
        Self {
            inner: InMemoryCheckpointStore::new(),
            latency,
            stall: None,
            fetches: AtomicUsize::new(0),
            in_flight: AtomicUsize::new(0),
            max_in_flight: AtomicUsize::new(0),
        }
    }

    /// An adapter whose fetches wait, in flight, for [`Self::unstall`].
    fn stalled() -> Self {
        Self {
            stall: Some(tokio::sync::Semaphore::new(0)),
            ..Self::new(None)
        }
    }

    /// Let `fetches` stalled fetches answer.
    fn unstall(&self, fetches: usize) {
        self.stall
            .as_ref()
            .expect("a stalled adapter")
            .add_permits(fetches);
    }

    fn in_flight(&self) -> usize {
        self.in_flight.load(Ordering::SeqCst)
    }

    /// Hold `value`'s bytes, as the mirror left them there.
    async fn hold(&self, value: &[u8]) {
        self.inner
            .put_ext_blob(
                &sha256_to_hex(&ext_content_hash(value)),
                Bytes::from(value.to_vec()),
            )
            .await
            .expect("seed the backup store");
    }

    fn fetches(&self) -> usize {
        self.fetches.load(Ordering::SeqCst)
    }

    fn max_in_flight(&self) -> usize {
        self.max_in_flight.load(Ordering::SeqCst)
    }
}

#[async_trait::async_trait]
impl CheckpointStore for CountingAdapter {
    // ── The one instrumented method ──────────────────────────────────────
    async fn get_ext_blob(&self, hash_hex: &str) -> Result<Option<Bytes>, CheckpointError> {
        self.fetches.fetch_add(1, Ordering::SeqCst);
        let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_in_flight.fetch_max(now, Ordering::SeqCst);
        if let Some(latency) = self.latency {
            tokio::time::sleep(latency).await;
        }
        if let Some(stall) = &self.stall {
            stall
                .acquire()
                .await
                .expect("the stall is never closed")
                .forget();
        }
        let fetched = self.inner.get_ext_blob(hash_hex).await;
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
        fetched
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

    async fn ext_blob_exists(&self, hash_hex: &str) -> Result<bool, CheckpointError> {
        self.inner.ext_blob_exists(hash_hex).await
    }

    async fn verify_list_route(&self) -> Result<(), CheckpointError> {
        self.inner.verify_list_route().await
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

    async fn list_live_ext_set(
        &self,
    ) -> Result<live_edit::checkpoint::LiveExtSet, CheckpointError> {
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

/// A real store that watches a catch-up's restore:
///
/// - each `recover_value` call, its hash, the calls in flight once it has
///   started and, given [`Self::budget`], the catch-up payload budget's
///   reserved bytes as it starts (T2);
/// - with `hook_after_restore` = n, each of the first n calls, once its
///   restore has returned, counts itself parked ([`Self::wait_hooked`] waits
///   for all n) and waits for [`Self::go`] before it answers, so its share of
///   the batch's grant is still held (T3's second batch);
/// - its length probes (`probe_value_lengths`) by call index across every
///   catch-up of the service: the indices in `park_probes` announce the park
///   ([`Self::wait_parked`]), record how many hashes they were asked for,
///   and wait for [`Self::release`] before they delegate. A catch-up whose
///   values are all missing makes three probes before its restore — its
///   first build's, its re-read's, its fetch-on-miss layout's — so its
///   fourth is its first batch's re-probe (T4's join wave).
///
/// Every trait method is forwarded; should the trait gain a defaulted one,
/// forward it too: an omitted arm would substitute the default body, not
/// pass through.
struct WatchingStore {
    inner: ValkeyExternalValueStore,
    budget: std::sync::OnceLock<Arc<MemoryBudget>>,
    restored: StdMutex<Vec<[u8; 32]>>,
    in_flight: AtomicUsize,
    max_in_flight: AtomicUsize,
    /// `(calls in flight, the budget's reserved bytes)` as each restore
    /// started.
    entries: StdMutex<Vec<(usize, u64)>>,
    hook_after_restore: Option<usize>,
    hooked: AtomicUsize,
    go: tokio::sync::Semaphore,
    park_probes: Vec<usize>,
    probes: AtomicUsize,
    parked_probe_sizes: StdMutex<Vec<usize>>,
    parked: tokio::sync::Notify,
    released: tokio::sync::Notify,
}

impl WatchingStore {
    fn new(
        inner: ValkeyExternalValueStore,
        park_probes: Vec<usize>,
        hook_after_restore: Option<usize>,
    ) -> Self {
        Self {
            inner,
            budget: std::sync::OnceLock::new(),
            restored: StdMutex::new(Vec::new()),
            in_flight: AtomicUsize::new(0),
            max_in_flight: AtomicUsize::new(0),
            entries: StdMutex::new(Vec::new()),
            hook_after_restore,
            hooked: AtomicUsize::new(0),
            go: tokio::sync::Semaphore::new(0),
            park_probes,
            probes: AtomicUsize::new(0),
            parked_probe_sizes: StdMutex::new(Vec::new()),
            parked: tokio::sync::Notify::new(),
            released: tokio::sync::Notify::new(),
        }
    }

    fn restores(&self) -> usize {
        self.restored.lock().expect("restored").len()
    }

    fn restored(&self) -> Vec<[u8; 32]> {
        self.restored.lock().expect("restored").clone()
    }

    fn entries(&self) -> Vec<(usize, u64)> {
        self.entries.lock().expect("entries").clone()
    }

    /// Wait until every hooked restore has returned and parked.
    async fn wait_hooked(&self) {
        let hooked = self
            .hook_after_restore
            .expect("a store with hooked restores");
        until(
            || self.hooked.load(Ordering::SeqCst) == hooked,
            "every hooked restore returned and parked",
        )
        .await;
    }

    /// Let every hooked restore answer.
    fn go(&self) {
        self.go.add_permits(
            self.hook_after_restore
                .expect("a store with hooked restores"),
        );
    }

    async fn wait_parked(&self) {
        tokio::time::timeout(DEADLINE, self.parked.notified())
            .await
            .expect("no catch-up reached its parking probe");
    }

    fn release(&self) {
        self.released.notify_one();
    }

    fn parked_probe_sizes(&self) -> Vec<usize> {
        self.parked_probe_sizes
            .lock()
            .expect("parked_probe_sizes")
            .clone()
    }
}

#[async_trait::async_trait]
impl ExternalValueStore for WatchingStore {
    async fn store_values(&self, values: &[(&[u8; 32], &[u8])]) -> Result<(), ServiceError> {
        self.inner.store_values(values).await
    }

    async fn store_verified_values(
        &self,
        values: &[(&[u8; 32], &[u8])],
    ) -> Result<(), ServiceError> {
        self.inner.store_verified_values(values).await
    }

    fn ext_value_verify(&self) -> ExtValueVerify<'_> {
        self.inner.ext_value_verify()
    }

    async fn load_values(
        &self,
        hashes: &[&[u8; 32]],
    ) -> Result<Vec<([u8; 32], Vec<u8>)>, ServiceError> {
        self.inner.load_values(hashes).await
    }

    async fn read_batch(
        &self,
        hashes: &[[u8; 32]],
    ) -> Result<std::collections::HashMap<[u8; 32], Vec<u8>>, ServiceError> {
        self.inner.read_batch(hashes).await
    }

    async fn probe_values(&self, hashes: &[[u8; 32]]) -> Result<ExtValueProbe, ServiceError> {
        self.inner.probe_values(hashes).await
    }

    async fn probe_value_lengths(
        &self,
        hashes: &[[u8; 32]],
    ) -> Result<ExtValueLengths, ServiceError> {
        let call = self.probes.fetch_add(1, Ordering::SeqCst);
        if self.park_probes.contains(&call) {
            self.parked_probe_sizes
                .lock()
                .expect("parked_probe_sizes")
                .push(hashes.len());
            self.parked.notify_one();
            self.released.notified().await;
        }
        self.inner.probe_value_lengths(hashes).await
    }

    async fn read_value_range(
        &self,
        hash: &[u8; 32],
        start: u64,
        len: usize,
    ) -> Result<Bytes, ServiceError> {
        self.inner.read_value_range(hash, start, len).await
    }

    async fn recover_value(
        &self,
        hash: &[u8; 32],
        on_fetched: live_edit::storage::external_values::OnFetched<'_>,
    ) -> Result<Bytes, ServiceError> {
        let call = {
            let mut restored = self.restored.lock().expect("restored");
            restored.push(*hash);
            restored.len()
        };
        let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_in_flight.fetch_max(now, Ordering::SeqCst);
        if let Some(budget) = self.budget.get() {
            self.entries
                .lock()
                .expect("entries")
                .push((now, budget.reserved_bytes()));
        }
        let restored = self.inner.recover_value(hash, on_fetched).await;
        if self.hook_after_restore.is_some_and(|hooked| call <= hooked) {
            self.hooked.fetch_add(1, Ordering::SeqCst);
            self.go
                .acquire()
                .await
                .expect("go is never closed")
                .forget();
        }
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
        restored
    }

    async fn enqueue_retirement_attempt(
        &self,
        object_id: &str,
        attempt: &str,
        releases: &[ExtRelease],
        send_by: std::time::Instant,
    ) -> Result<live_edit::storage::external_values::ReleaseAttemptOutcome, ServiceError> {
        self.inner
            .enqueue_retirement_attempt(object_id, attempt, releases, send_by)
            .await
    }

    async fn process_pending(
        &self,
        policy: &GcPolicy,
        state: &mut GcPassState,
    ) -> Result<GcBatchResult, ServiceError> {
        self.inner.process_pending(policy, state).await
    }

    fn ext_release_gate(&self) -> Option<&Arc<ExtReleaseGate>> {
        self.inner.ext_release_gate()
    }
}

/// How [`restoring_service`] shapes its service.
struct Shape {
    /// The catch-up payload budget `D_c`, in bytes (whole KiB).
    payload_budget_bytes: u64,
    /// The catch-up capacity budget, when not the fixture's 2 s.
    capacity_budget_ms: Option<u64>,
    /// [`WatchingStore`]'s `park_probes`.
    park_probes: Vec<usize>,
    /// [`WatchingStore`]'s `hook_after_restore`.
    hook_after_restore: Option<usize>,
    /// Whether the backup store is [`CountingAdapter::stalled`].
    stalled_fetches: bool,
}

impl Shape {
    fn budget(payload_budget_bytes: u64) -> Self {
        Self {
            payload_budget_bytes,
            capacity_budget_ms: None,
            park_probes: Vec::new(),
            hook_after_restore: None,
            stalled_fetches: false,
        }
    }
}

/// A service on `tv` whose external values live in a real store wrapped in a
/// [`WatchingStore`], with a [`CountingAdapter`] (stalled when `shape` says
/// so) as its backup store and a catch-up payload budget of `shape`'s size on
/// the service's metrics (so its wait outcomes are counted); its catch-ups
/// run on a client with `shape`'s capacity budget when it names one.
async fn restoring_service(
    tv: &TestValkey,
    shape: Shape,
) -> (
    Arc<LiveEditService>,
    Arc<WatchingStore>,
    Arc<CountingAdapter>,
    Arc<MemoryBudget>,
) {
    let base = build_service(tv, Arc::new(AllowAllPermissions));
    let adapter = Arc::new(if shape.stalled_fetches {
        CountingAdapter::stalled()
    } else {
        CountingAdapter::new(None)
    });
    let inner = ValkeyExternalValueStore::new(tv.client().clone(), tv.key_schema().clone())
        .with_metrics(Arc::clone(&base.metrics))
        .with_checkpoint_store(Arc::clone(&adapter) as Arc<dyn CheckpointStore>);
    let store = Arc::new(WatchingStore::new(
        inner,
        shape.park_probes,
        shape.hook_after_restore,
    ));
    let mut svc = LiveEditService {
        ext_value_store: Arc::clone(&store) as Arc<dyn ExternalValueStore>,
        ..base
    };
    if let Some(ms) = shape.capacity_budget_ms {
        let mut config = crate::common::test_config();
        config.subscriber_admission_wait_ms = ms;
        svc.valkey = live_edit::storage::test_support::connect_test_client(&config)
            .await
            .expect("a catch-up client with a wide capacity budget connects");
    }
    let budget = Arc::new(MemoryBudget::with_series(
        shape.payload_budget_bytes / 1024,
        Arc::new(StdMutex::new(Some(Arc::clone(&svc.metrics)))),
        &CATCHUP_PAYLOAD_SERIES,
    ));
    svc.valkey.catchup_payload_budget = Arc::clone(&budget);
    assert!(
        store.budget.set(Arc::clone(&budget)).is_ok(),
        "the store watches one budget"
    );
    (Arc::new(svc), store, adapter, budget)
}

/// A layer `layer` in `session_id` (created when `None`) whose one commit
/// sets a field to each of `values`, left as a disaster-recovery restore
/// leaves it: every value's meta stored without its data, its bytes held by
/// `adapter` alone. Returns the session and the layer's object.
async fn backup_only_layer(
    tv: &TestValkey,
    svc: &LiveEditService,
    adapter: &CountingAdapter,
    session_id: Option<&str>,
    layer: &str,
    values: &[Vec<u8>],
) -> (String, u64) {
    let (session_id, object_id) = match session_id {
        Some(session_id) => (
            session_id.to_owned(),
            create_layer(svc, session_id, layer).await,
        ),
        None => create_session_and_layer(tv, svc, "d1-restore", layer).await,
    };
    let delta = OwnedDelta {
        struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
            parent_id: 0,
            section_id: SECTION,
            section_name: "Mesh".to_owned(),
            section_type: 1,
        })],
        set_fields: values
            .iter()
            .enumerate()
            .map(|(i, value)| SetField {
                section_id: SECTION,
                key_name: format!("v{i:02}"),
                value_or_ext_hash: ext_content_hash(value).to_vec(),
                ext_value_size: value.len() as u64,
                ext_value_index: i as u32 + 1,
                set_order: i as u64,
            })
            .collect(),
        ..Default::default()
    };
    let parts: Vec<&[u8]> = values.iter().map(Vec::as_slice).collect();
    upload(
        svc,
        &session_id,
        layer,
        object_id,
        delta_to_ompf_bytes(&delta, &parts),
    )
    .await;
    for value in values {
        adapter.hold(value).await;
        let removed: i64 = tv
            .client()
            .pool
            .unlink(
                tv.key_schema()
                    .ext_data(&sha256_to_hex(&ext_content_hash(value))),
            )
            .await
            .expect("UNLINK ext:data");
        assert_eq!(removed, 1, "precondition: the value was stored");
    }
    (session_id, object_id)
}

/// A brand-new subscription to `layer` from `client`.
async fn subscribe_as(
    svc: &LiveEditService,
    session_id: &str,
    layer: &str,
    client: &str,
) -> SubscribeStream {
    let mut request = subscribe_request(session_id, layer, 0, 0);
    request.get_mut().client_id = client.to_owned();
    let response = handle_subscribe_read_layer(svc, request)
        .await
        .expect("SubscribeReadLayer must open");
    Box::pin(response.into_inner())
}

/// Take a brand-new catch-up's content and its caught-up ack, and check it
/// carries every one of `values`.
async fn expect_served(stream: &mut SubscribeStream, object_id: u64, values: &[Vec<u8>]) {
    let (content, body) = next_content(stream, "a restored layer's catch-up").await;
    assert_eq!((content.object_id, content.sequence_num), (object_id, 2));
    let parts = carried(&body);
    for value in values {
        assert!(
            parts.contains(value),
            "every restored value is delivered as the backup store held it"
        );
    }
    expect_caught_up_ack(stream, object_id, 2).await;
}

/// `live_edit_subscriber_stream_exits_total{reason}`.
fn exits(metrics: &Metrics, reason: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_subscriber_stream_exits_total",
        &[("reason", reason)],
    )
    .expect("every exit reason is seeded") as u64
}

/// Wait, within [`DEADLINE`], until `condition` holds.
async fn until(mut condition: impl FnMut() -> bool, what: &str) {
    let deadline = tokio::time::Instant::now() + DEADLINE;
    while !condition() {
        assert!(
            tokio::time::Instant::now() < deadline,
            "{what} did not hold within {DEADLINE:?}"
        );
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
}

/// One `live_edit_checkpoint_ext_recovery_total{result, meta}` cell.
fn recovery_cell(metrics: &Metrics, result: &str, meta: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_checkpoint_ext_recovery_total",
        &[("result", result), ("meta", meta)],
    )
    .unwrap_or_else(|| panic!("recovery cell {{result={result}, meta={meta}}} is zero-seeded"))
        as u64
}

/// `live_edit_catchup_payload_wait_seconds_count`, summed over its `outcome`
/// label: one observation per reservation of the catch-up payload budget,
/// whether a restore batch's or a payload's.
fn payload_waits(metrics: &Metrics) -> u64 {
    live_edit::observability::metrics::test_util::sample_sum(
        &render(metrics),
        "live_edit_catchup_payload_wait_seconds_count",
        &[],
    )
    .unwrap_or(0.0) as u64
}

/// `value`'s meta `ref_count`, as the store holds it.
async fn ref_count(tv: &TestValkey, value: &[u8]) -> Option<i64> {
    tv.client()
        .pool
        .hget(
            tv.key_schema()
                .ext_meta(&sha256_to_hex(&ext_content_hash(value))),
            "ref_count",
        )
        .await
        .expect("HGET ext_meta ref_count")
}

/// The catch-up payload budget as `(reserved, over-committed, waiters)`.
fn held(budget: &MemoryBudget) -> (u64, u64, usize) {
    (
        budget.reserved_bytes(),
        budget.overcommit_bytes(),
        budget.waiters(),
    )
}

// ═════════════════════════════════════════════════════════════════════════════
// T2 — the charge held while the values are fetched
// ═════════════════════════════════════════════════════════════════════════════

/// T2: a catch-up restoring 12 backup-only values on a catch-up payload
/// budget of two values' charges restores them two at a time, and every
/// fetch is covered as it starts: the budget then holds at least one share
/// (at the restore-only `EXT_RESTORE_CHARGE_CENTI`, whole KiB) for every
/// restore in flight, reserved before any of them started. The content carries every value, each restored once
/// through the backup store — each counted `recovered` with `meta="present"`,
/// the shape a restore's recount leaves (PD-7b) — and the budget is all
/// returned. The restore's time shows in
/// `live_edit_catchup_payload_wait_seconds`: one observation per batch's
/// reservation (6) and one for the payload's.
///
/// Fails without the change (in-place perturbations): with the batch's
/// reservation skipped, the budget holds nothing as the restores start; with
/// each batch taken by a non-blocking grant (`try_grant`, which records no
/// wait), the wait histogram counts the payload's reservation alone; with the
/// catch-up passing the upload path's `EXT_EXPANSION_CENTI` (3.09) instead of
/// `EXT_RESTORE_CHARGE_CENTI`, the budget holds less than a share for each
/// restore in flight as it starts.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn every_value_a_restore_fetches_is_covered_by_a_share_of_the_payload_budget_reserved_before_it(
) {
    const VALUES: usize = 12;
    const BATCHES: u64 = 6;
    let tv = TestValkey::setup().await;
    let layer_values = values(0x10, VALUES);
    let one = share(layer_values[0].len());
    let (svc, store, adapter, budget) = restoring_service(&tv, Shape::budget(2 * one)).await;
    let (session_id, object_id) =
        backup_only_layer(&tv, &svc, &adapter, None, "t2-covered", &layer_values).await;

    let mut stream = subscribe_as(&svc, &session_id, "t2-covered", "t2").await;
    expect_served(&mut stream, object_id, &layer_values).await;

    let entries = store.entries();
    assert_eq!(
        entries.len(),
        VALUES,
        "every value restored once: {entries:?}"
    );
    for (in_flight, reserved) in &entries {
        assert!(
            *reserved >= *in_flight as u64 * one,
            "as a restore starts, the budget holds a share of {one} B for each of the \
             {in_flight} in flight; it holds {reserved} B ({entries:?})"
        );
    }
    assert_eq!(
        store.max_in_flight.load(Ordering::SeqCst),
        2,
        "two values' charges fit the budget, so two are restored at once"
    );
    assert_eq!(
        adapter.fetches(),
        VALUES,
        "each fetched from the backup store once"
    );
    assert_eq!(
        (
            recovery_cell(&svc.metrics, "recovered", EXT_RECOVERY_META_PRESENT),
            recovery_cell(&svc.metrics, "recovered", EXT_RECOVERY_META_ABSENT),
        ),
        (VALUES as u64, 0),
        "PD-7b: every restore of a meta without data is counted meta=\"present\""
    );
    assert_eq!(
        payload_waits(&svc.metrics),
        BATCHES + 1,
        "one wait observation per restore batch's reservation and one for the payload's"
    );
    for value in &layer_values {
        assert!(blob_stored(&tv, value).await, "every value written back");
    }
    assert_eq!(hold_fields(&tv).await, 0, "the hold is released");
    assert_eq!(
        (
            budget.reserved_bytes(),
            budget.overcommit_bytes(),
            budget.waiters()
        ),
        (0, 0, 0),
        "the catch-up payload budget is all returned"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// T3 — a restore out of capacity budget
// ═════════════════════════════════════════════════════════════════════════════

/// T3: a catch-up whose restore cannot reserve the catch-up payload budget —
/// another holder keeps all of it past the catch-up's 2 s capacity budget —
/// is refused before any content: `UNAVAILABLE` with the `catchup_capacity`
/// refusal and its retry hint, its exit counted `catchup_capacity` and its
/// wait `refused_timeout`. It restored nothing (nothing fetched, nothing
/// written back), holds nothing of the budget afterwards — no grant, no
/// top-up, no place in its queue — and its read hold is released.
///
/// Fails without the change (in-place perturbation): with the restore's
/// reservation skipped, the values are restored before the payload's
/// reservation is refused.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_restore_that_cannot_reserve_the_payload_budget_is_refused_catchup_capacity_before_content_holding_nothing(
) {
    let tv = TestValkey::setup().await;
    let layer_values = values(0x30, 4);
    let (svc, store, adapter, budget) = restoring_service(&tv, Shape::budget(64 << 10)).await;
    let (session_id, _object_id) =
        backup_only_layer(&tv, &svc, &adapter, None, "t3-refused", &layer_values).await;
    let held = budget
        .reserve(budget.total_bytes(), None)
        .await
        .expect("an idle budget grants its whole total");

    let mut stream = subscribe_as(&svc, &session_id, "t3-refused", "t3").await;
    let status = expect_ended_before_content(&mut stream, "a restore out of capacity").await;
    assert_refusal_detail(
        &status,
        refusal_reason::CATCHUP_CAPACITY,
        &[],
        CAPACITY_RETRY_DELAY,
    );
    until(
        || exits(&svc.metrics, STREAM_EXIT_CATCHUP_CAPACITY) == 1,
        "the refusal is counted catchup_capacity",
    )
    .await;
    assert_eq!(
        sample_value(
            &render(&svc.metrics),
            "live_edit_catchup_payload_wait_seconds_count",
            &[("outcome", "refused_timeout")],
        ),
        Some(1.0),
        "the restore's wait is counted refused_timeout"
    );
    assert_eq!(
        (store.restores(), adapter.fetches()),
        (0, 0),
        "nothing restored, nothing fetched"
    );
    for value in &layer_values {
        assert!(!blob_stored(&tv, value).await, "nothing written back");
    }
    assert_eq!(
        (
            budget.reserved_bytes(),
            budget.overcommit_bytes(),
            budget.waiters()
        ),
        (budget.total_bytes(), 0, 0),
        "the refused catch-up holds nothing of the budget: only the other holder's grant"
    );
    assert_eq!(hold_fields(&tv).await, 0, "the hold is released");
    drop(held);
    assert_eq!(budget.reserved_bytes(), 0);
}

/// T3, the second batch: a catch-up restoring 8 backup-only values on a
/// catch-up payload budget of exactly one batch's charge restores its first
/// batch of 4, and then another holder takes the whole budget before the
/// second batch reserves; that reservation runs the catch-up's capacity
/// budget out, and the catch-up is refused `catchup_capacity` before any
/// content. While the first batch's restores have returned but not yet
/// answered, the batch holds exactly its charge: the whole budget. The first
/// batch's values stay written back. The client's retry, once the holder
/// lets go, restores only the other 4 and is served all 8: each value
/// fetched from the backup store once in all.
///
/// Fails without the change (in-place perturbations): with all 8 values cut
/// into one batch, the refused catch-up has restored and fetched all 8 —
/// `(8, 8)` where only the first batch's `(4, 4)` is expected; with the
/// batch's reservation skipped, the first batch holds none of the budget, so
/// the whole-budget grant is made at once and never queues behind it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_restore_refused_after_its_first_batch_keeps_that_batch_stored_and_its_retry_restores_only_the_rest(
) {
    const VALUES: usize = 8;
    let tv = TestValkey::setup().await;
    let layer_values = values(0x40, VALUES);
    let batch = EXT_RESTORE_PARALLELISM as u64 * share(layer_values[0].len());
    let (svc, store, adapter, budget) = restoring_service(
        &tv,
        Shape {
            hook_after_restore: Some(EXT_RESTORE_PARALLELISM),
            ..Shape::budget(batch)
        },
    )
    .await;
    let (session_id, object_id) =
        backup_only_layer(&tv, &svc, &adapter, None, "t3-second", &layer_values).await;

    let mut stream = subscribe_as(&svc, &session_id, "t3-second", "t3-first-try").await;
    // The first batch's restores have all returned and wait, each still
    // holding its share: queue a grant of the whole budget behind the batch's
    // shares before the batch ends, so it is granted the moment they are
    // repaid, ahead of the second batch.
    store.wait_hooked().await;
    assert_eq!(
        budget.reserved_bytes(),
        batch,
        "the first batch holds its whole charge while its restores are held"
    );
    let taker = {
        let budget = Arc::clone(&budget);
        tokio::spawn(async move { budget.reserve(budget.total_bytes(), None).await })
    };
    until(|| budget.waiters() == 1, "the whole-budget grant queues").await;
    store.go();
    let held = tokio::time::timeout(DEADLINE, taker)
        .await
        .expect("the whole budget is granted once the first batch is repaid")
        .expect("the taker did not panic")
        .expect("the whole budget is granted");

    let status = expect_ended_before_content(&mut stream, "the second batch refused").await;
    assert_refusal_detail(
        &status,
        refusal_reason::CATCHUP_CAPACITY,
        &[],
        CAPACITY_RETRY_DELAY,
    );
    let first_batch = store.restored();
    assert_eq!(
        (first_batch.len(), adapter.fetches()),
        (EXT_RESTORE_PARALLELISM, EXT_RESTORE_PARALLELISM),
        "only the first batch was restored"
    );
    for value in &layer_values {
        let hash = ext_content_hash(value);
        assert_eq!(
            blob_stored(&tv, value).await,
            first_batch.contains(&hash),
            "the first batch's values stay written back, and only they"
        );
    }
    drop(held);

    let mut retry = subscribe_as(&svc, &session_id, "t3-second", "t3-retry").await;
    expect_served(&mut retry, object_id, &layer_values).await;
    let restored = store.restored();
    assert_eq!(
        restored.len(),
        VALUES,
        "the retry restored 4 more: {restored:?}"
    );
    for hash in &restored[EXT_RESTORE_PARALLELISM..] {
        assert!(
            !first_batch.contains(hash),
            "the retry restores only the values the refused catch-up did not"
        );
    }
    assert_eq!(adapter.fetches(), VALUES, "each value fetched once in all");
    assert_eq!(hold_fields(&tv).await, 0, "the holds are released");
    assert_eq!(
        (
            budget.reserved_bytes(),
            budget.overcommit_bytes(),
            budget.waiters()
        ),
        (0, 0, 0),
        "the catch-up payload budget is all returned"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// T4 — no deadlock, and the join wave
// ═════════════════════════════════════════════════════════════════════════════

/// T4: six catch-ups, two on each of three layers of 8 backup-only values
/// (two batches each), restore on a catch-up payload budget of one and a
/// half batches' charges. The budget is held by another while every one of
/// them reaches its first batch's reservation and queues, having restored
/// nothing; then it is let go. Since no catch-up keeps a batch's charge while
/// it reserves its next batch or its payload, none waits on itself or on
/// another's held charge, and all six are served, none refused, with nothing
/// of the budget left held or queued.
///
/// Fails without the change (in-place perturbation): a catch-up that keeps
/// each batch's charge until its whole restore returns holds one batch while
/// its second waits for a batch that cannot fit beside it, and is refused
/// `catchup_capacity` at its capacity budget's end.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn six_catch_ups_restoring_over_three_layers_on_one_and_a_half_batches_of_budget_are_all_served(
) {
    const LAYERS: usize = 3;
    const VALUES: usize = 8;
    const CATCH_UPS_PER_LAYER: usize = 2;
    let tv = TestValkey::setup().await;
    let batch = EXT_RESTORE_PARALLELISM as u64 * share(blob(0).len());
    let (svc, store, adapter, budget) = restoring_service(
        &tv,
        Shape {
            capacity_budget_ms: Some(WIDE_BUDGET_MS),
            ..Shape::budget(batch * 3 / 2)
        },
    )
    .await;
    let mut layers = Vec::new();
    let mut session = None;
    for l in 0..LAYERS {
        let name = format!("t4-layer-{l}");
        let layer_values = values(0x50 + (l * VALUES) as u8, VALUES);
        let (session_id, object_id) = backup_only_layer(
            &tv,
            &svc,
            &adapter,
            session.as_deref(),
            &name,
            &layer_values,
        )
        .await;
        session = Some(session_id);
        layers.push((name, object_id, layer_values));
    }
    let session_id = session.expect("a session");
    let held = budget
        .reserve(budget.total_bytes(), None)
        .await
        .expect("an idle budget grants its whole total");

    let mut streams = Vec::new();
    for (name, object_id, layer_values) in &layers {
        for c in 0..CATCH_UPS_PER_LAYER {
            let stream = subscribe_as(&svc, &session_id, name, &format!("{name}-{c}")).await;
            streams.push((stream, *object_id, layer_values));
        }
    }
    until(
        || budget.waiters() == LAYERS * CATCH_UPS_PER_LAYER,
        "every catch-up queues for its first batch",
    )
    .await;
    assert_eq!(store.restores(), 0, "nothing restored while they wait");
    drop(held);

    futures::future::join_all(
        streams.iter_mut().map(|(stream, object_id, layer_values)| {
            expect_served(stream, *object_id, layer_values)
        }),
    )
    .await;
    assert_eq!(
        exits(&svc.metrics, STREAM_EXIT_CATCHUP_CAPACITY),
        0,
        "none refused"
    );
    assert!(
        adapter.fetches() >= LAYERS * VALUES,
        "every value fetched at least once"
    );
    assert_eq!(hold_fields(&tv).await, 0, "the holds are released");
    assert_eq!(
        (
            budget.reserved_bytes(),
            budget.overcommit_bytes(),
            budget.waiters()
        ),
        (0, 0, 0),
        "the catch-up payload budget is all returned"
    );
}

/// T4, the join wave: four catch-ups of one layer of 8 backup-only values
/// (two batches each). Three of them are stopped at their first batch's
/// re-probe — after their layouts found every value missing — while the
/// fourth restores every value and is served; then the three go on. Each
/// re-probes its batches, finds every value stored again, and lays it out
/// at its stored length with neither a fetch nor a charge: the wave fetches
/// each value from the backup store exactly once, a stopped catch-up reserves
/// nothing for its restore (its batches' first probe finds every value stored
/// before any reservation), and all four are served.
///
/// Fails without the change (in-place perturbations): without the probe
/// before each batch's reservation, a stopped catch-up reserves its first
/// batch before it finds the values stored, so here it holds the budget
/// while stopped and the first catch-up ends `CATCHUP_CAPACITY` (where it
/// does not, the reservation count shows it: 12, not 6); without both
/// probes, each stopped catch-up fetches every value again (four fetches of
/// each).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_join_wave_on_a_restored_layer_fetches_each_value_from_the_backup_store_once() {
    const VALUES: usize = 8;
    const STOPPED: usize = 3;
    /// Length probes a catch-up of this layer makes before its first
    /// batch's re-probe (see [`WatchingStore`]).
    const PROBES_BEFORE_RESTORE: usize = 3;
    let tv = TestValkey::setup().await;
    let layer_values = values(0x70, VALUES);
    let park_probes: Vec<usize> = (0..STOPPED)
        .map(|i| i * (PROBES_BEFORE_RESTORE + 1) + PROBES_BEFORE_RESTORE)
        .collect();
    let (svc, store, adapter, budget) = restoring_service(
        &tv,
        Shape {
            park_probes,
            ..Shape::budget(1 << 20)
        },
    )
    .await;
    let (session_id, object_id) =
        backup_only_layer(&tv, &svc, &adapter, None, "t4-wave", &layer_values).await;

    let mut stopped = Vec::new();
    for i in 0..STOPPED {
        stopped.push(subscribe_as(&svc, &session_id, "t4-wave", &format!("stopped-{i}")).await);
        store.wait_parked().await;
    }
    assert_eq!(
        store.parked_probe_sizes(),
        vec![EXT_RESTORE_PARALLELISM; STOPPED],
        "fixture self-check: each stopped catch-up stands at its first batch's re-probe"
    );
    assert_eq!(
        store.restores(),
        0,
        "fixture self-check: nothing restored yet"
    );

    let mut first = subscribe_as(&svc, &session_id, "t4-wave", "first").await;
    expect_served(&mut first, object_id, &layer_values).await;
    assert_eq!(
        adapter.fetches(),
        VALUES,
        "the first catch-up fetched every value"
    );
    for _ in 0..STOPPED {
        store.release();
    }
    // Read at once: each payload's reservation is clamped to the budget's
    // total and held until its stream takes its pages, so the others wait on
    // whichever stream is granted first.
    futures::future::join_all(
        stopped
            .iter_mut()
            .map(|stream| expect_served(stream, object_id, &layer_values)),
    )
    .await;

    assert_eq!(
        (adapter.fetches(), store.restores()),
        (VALUES, VALUES),
        "the wave fetched and restored each value once: the stopped catch-ups' re-probes found \
         them stored"
    );
    assert_eq!(
        payload_waits(&svc.metrics),
        2 + (STOPPED as u64 + 1),
        "the first catch-up reserved its 2 batches, and each of the 4 catch-ups its payload; a \
         stopped catch-up reserved nothing for its restore, its batches' first probe having \
         found every value stored"
    );
    assert_eq!(
        atomic_misses(&svc.metrics),
        (STOPPED as u64 + 1, STOPPED as u64 + 1),
        "fixture self-check: every catch-up of the wave missed genuinely"
    );
    assert_eq!(hold_fields(&tv).await, 0, "the holds are released");
    assert_eq!(
        (
            budget.reserved_bytes(),
            budget.overcommit_bytes(),
            budget.waiters()
        ),
        (0, 0, 0),
        "the catch-up payload budget is all returned"
    );
}

/// TR-1, the contended join wave: two catch-ups of one layer of 8
/// backup-only values (two batches each) on a catch-up payload budget of
/// exactly one batch's charge, which another holder fills. Each probes its
/// first batch, finds every value missing and queues for its reservation —
/// A, then B — having fetched nothing. Once the holder lets go, A restores
/// its first batch while B waits; B's reservation is granted only once A has
/// repaid it, and B's probe after the grant finds those values stored and
/// gives their shares back before fetching anything. The batches after go
/// the same way, one holder at a time, so the wave fetches and restores each
/// value exactly once; both catch-ups are served; every value's meta
/// `ref_count` is the recount's, untouched by the write-backs; and the
/// budget is all returned, no one queued.
///
/// Fails without the change (in-place perturbation): without the probe
/// after the reservation, B fetches again the values A restored while it
/// waited, and the wave makes at least 12 fetches (observed: 16; one run
/// ended a catch-up `UNAVAILABLE` at its read-back instead).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_contended_join_wave_fetches_each_value_once_its_waiters_re_probing_after_their_grant() {
    const VALUES: usize = 8;
    let tv = TestValkey::setup().await;
    let layer_values = values(0xC0, VALUES);
    let batch = EXT_RESTORE_PARALLELISM as u64 * share(layer_values[0].len());
    let (svc, store, adapter, budget) = restoring_service(
        &tv,
        Shape {
            capacity_budget_ms: Some(WIDE_BUDGET_MS),
            ..Shape::budget(batch)
        },
    )
    .await;
    let (session_id, object_id) =
        backup_only_layer(&tv, &svc, &adapter, None, "tr1-wave", &layer_values).await;
    let mut recounted = Vec::new();
    for value in &layer_values {
        recounted.push(ref_count(&tv, value).await);
    }
    assert!(
        recounted.iter().all(|count| count.is_some_and(|c| c > 0)),
        "fixture self-check: every value's meta is stored without its data: {recounted:?}"
    );
    let holder = budget
        .reserve(budget.total_bytes(), None)
        .await
        .expect("an idle budget grants its whole total");

    let mut a = subscribe_as(&svc, &session_id, "tr1-wave", "tr1-a").await;
    until(|| budget.waiters() == 1, "A queues for its first batch").await;
    let mut b = subscribe_as(&svc, &session_id, "tr1-wave", "tr1-b").await;
    until(|| budget.waiters() == 2, "B queues for its first batch").await;
    assert_eq!(
        (adapter.fetches(), store.restores()),
        (0, 0),
        "fixture self-check: both wait having probed and fetched nothing"
    );
    drop(holder);

    // Read at once: each payload's reservation is clamped to the budget's
    // total and held until its stream takes its pages.
    futures::future::join(
        expect_served(&mut a, object_id, &layer_values),
        expect_served(&mut b, object_id, &layer_values),
    )
    .await;

    assert_eq!(
        (adapter.fetches(), store.restores()),
        (VALUES, VALUES),
        "the contended wave fetched and restored each value once"
    );
    for (value, recount) in layer_values.iter().zip(&recounted) {
        assert_eq!(
            ref_count(&tv, value).await,
            *recount,
            "a write-back keeps the recount's ref_count"
        );
    }
    assert_eq!(hold_fields(&tv).await, 0, "the holds are released");
    assert_eq!(
        held(&budget),
        (0, 0, 0),
        "the catch-up payload budget is all returned"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// TR-2 — a batch's grant through its re-probe, a failure, a disconnect
// ═════════════════════════════════════════════════════════════════════════════

/// TR-2 (i): when the probe after a batch's reservation finds part of the
/// batch stored — here 2 of its 4 values, stored by another while the
/// reservation was made — the batch keeps only the other values' shares and
/// gives the rest back before it fetches anything: as each of the two
/// remaining restores starts, the budget holds exactly their two shares.
/// Only those two are fetched; the two found stored are answered at their
/// stored length, not restored; the budget is all returned.
///
/// Fails without the change (in-place perturbation): with the grant not
/// shrunk, the budget still holds all four shares as the restores start.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_batch_whose_re_probe_finds_part_of_it_stored_keeps_only_the_rest_s_shares_before_any_fetch(
) {
    let tv = TestValkey::setup().await;
    let adapter = Arc::new(CountingAdapter::new(None));
    let inner = ValkeyExternalValueStore::new(tv.client().clone(), tv.key_schema().clone())
        .with_checkpoint_store(Arc::clone(&adapter) as Arc<dyn CheckpointStore>);
    let store = Arc::new(WatchingStore::new(inner, Vec::new(), None));
    let layer_values = values(0xD0, EXT_RESTORE_PARALLELISM);
    let one = share(layer_values[0].len());
    for value in &layer_values {
        let _: () = tv
            .client()
            .pool
            .hset(
                tv.key_schema()
                    .ext_meta(&sha256_to_hex(&ext_content_hash(value))),
                vec![("ref_count", "1")],
            )
            .await
            .expect("HSET ext_meta: a meta without data");
        adapter.hold(value).await;
    }
    let budget = Arc::new(MemoryBudget::with_series(
        1 << 10,
        Arc::new(StdMutex::new(None)),
        &CATCHUP_PAYLOAD_SERIES,
    ));
    assert!(store.budget.set(Arc::clone(&budget)).is_ok());
    // Stored by another between the batch's first probe and its reservation.
    let stored_meanwhile = [&layer_values[1], &layer_values[3]];
    let restores = RestoreLayout::restore_for_test(
        layer_values
            .iter()
            .map(|value| (ext_content_hash(value), value.len() as u64)),
    );

    let reserved_at_grant = Arc::new(StdMutex::new(Vec::new()));
    let lengths = restore_in_batches(
        store.as_ref(),
        &restores,
        EXT_RESTORE_CHARGE_CENTI,
        budget.total_bytes(),
        |bytes| {
            let budget = Arc::clone(&budget);
            let other = ValkeyExternalValueStore::new(tv.client().clone(), tv.key_schema().clone());
            let stored_meanwhile: Vec<Vec<u8>> =
                stored_meanwhile.iter().map(|v| v.to_vec()).collect();
            let reserved_at_grant = Arc::clone(&reserved_at_grant);
            async move {
                let grant = budget.reserve(bytes, None).await.map_err(|refusal| {
                    ServiceError::Internal(format!("an idle budget refused: {refusal:?}"))
                })?;
                reserved_at_grant
                    .lock()
                    .expect("reserved_at_grant")
                    .push(budget.reserved_bytes());
                for value in &stored_meanwhile {
                    other
                        .store_values(&[(&ext_content_hash(value), value.as_slice())])
                        .await?;
                }
                Ok::<_, ServiceError>(grant)
            }
        },
    )
    .await
    .expect("the batch restores");

    assert_eq!(
        *reserved_at_grant.lock().expect("reserved_at_grant"),
        vec![EXT_RESTORE_PARALLELISM as u64 * one],
        "fixture self-check: one batch, reserved for all four values"
    );
    assert_eq!(
        store.entries(),
        vec![(1, 2 * one), (2, 2 * one)],
        "as each remaining restore starts, the batch holds only the two remaining shares"
    );
    assert_eq!(
        adapter.fetches(),
        2,
        "only the two still missing are fetched"
    );
    for (i, value) in layer_values.iter().enumerate() {
        let restored = !stored_meanwhile.contains(&value);
        assert_eq!(
            lengths.get(&ext_content_hash(value)),
            Some(&RestoredLength {
                len: value.len() as u64,
                restored,
            }),
            "value {i}: answered at its stored length, restored only when this restore fetched \
             it"
        );
        assert!(blob_stored(&tv, value).await, "value {i} is stored");
    }
    assert_eq!(held(&budget), (0, 0, 0), "the budget is all returned");
}

/// TR-2 (ii): a catch-up whose batch of 4 backup-only values holds one the
/// backup store has lost restores the other 3 — each fetched once and
/// written back, where it stays — and then ends `INTERNAL` before any
/// content, the lost value genuinely missing. Every share of the batch is
/// repaid and nothing else of the budget is held; the read hold is released.
///
/// Fails without the change (in-place perturbation): with the batch's
/// restores joined fail-fast (`try_join_all`), the lost value's error drops
/// the other restores mid-flight, before they are fetched and written back
/// (observed: 1 fetch of the 4).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_batch_with_one_value_the_backup_store_lost_writes_the_others_back_and_ends_internal_holding_nothing(
) {
    let tv = TestValkey::setup().await;
    let layer_values = values(0xE0, EXT_RESTORE_PARALLELISM);
    let lost = &layer_values[1];
    let (svc, store, adapter, budget) = restoring_service(&tv, Shape::budget(1 << 20)).await;
    let (session_id, _object_id) =
        backup_only_layer(&tv, &svc, &adapter, None, "tr2-lost", &layer_values).await;
    adapter
        .inner
        .delete_ext_blob(&sha256_to_hex(&ext_content_hash(lost)))
        .await
        .expect("the backup store loses one value");

    let mut stream = subscribe_as(&svc, &session_id, "tr2-lost", "tr2-lost").await;
    let status = expect_ended_before_content(&mut stream, "a batch with a lost value").await;

    assert_eq!(status.code(), Code::Internal, "{status:?}");
    assert_eq!(
        (store.restores(), adapter.fetches()),
        (EXT_RESTORE_PARALLELISM, EXT_RESTORE_PARALLELISM),
        "every value of the batch was restored once"
    );
    assert_eq!(
        (
            recovery_cell(&svc.metrics, "recovered", EXT_RECOVERY_META_PRESENT),
            recovery_cell(&svc.metrics, "adapter_miss", EXT_RECOVERY_META_PRESENT),
        ),
        (EXT_RESTORE_PARALLELISM as u64 - 1, 1),
        "three restored, one the backup store lacks"
    );
    for value in &layer_values {
        assert_eq!(
            blob_stored(&tv, value).await,
            value != lost,
            "the others stay written back; the lost one is not stored"
        );
    }
    assert_eq!(hold_fields(&tv).await, 0, "the hold is released");
    assert_eq!(
        held(&budget),
        (0, 0, 0),
        "every share is repaid, nothing of the budget held"
    );
}

/// TR-2 (iii): a client that disconnects while its catch-up's first batch
/// is being fetched — all 4 fetches held in flight at the backup store —
/// leaves nothing held, without waiting for those fetches: the build races
/// the subscription's endings, so the disconnect drops it with the batch's
/// shares and its `{gc}` read hold. Within [`DEADLINE`], with the fetches
/// still stalled, the catch-up payload budget is all returned, no one
/// queued, and the read hold is released.
///
/// Fails without the change (in-place perturbation): with the brand-new
/// build no longer raced against the subscription's endings, the disconnect
/// leaves the batch's shares held behind its stalled fetches.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_client_disconnecting_while_its_batch_is_fetched_leaves_nothing_held() {
    let tv = TestValkey::setup().await;
    let layer_values = values(0xE8, EXT_RESTORE_PARALLELISM);
    let (svc, _store, adapter, budget) = restoring_service(
        &tv,
        Shape {
            stalled_fetches: true,
            ..Shape::budget(1 << 20)
        },
    )
    .await;
    let (session_id, _object_id) =
        backup_only_layer(&tv, &svc, &adapter, None, "tr2-gone", &layer_values).await;

    let stream = subscribe_as(&svc, &session_id, "tr2-gone", "tr2-gone").await;
    until(
        || adapter.in_flight() == EXT_RESTORE_PARALLELISM,
        "the batch's fetches are all in flight",
    )
    .await;
    assert!(
        budget.reserved_bytes() > 0,
        "fixture self-check: the batch's charge is held while it is fetched"
    );
    drop(stream);

    let deadline = tokio::time::Instant::now() + DEADLINE;
    loop {
        let holds = hold_fields(&tv).await;
        if held(&budget) == (0, 0, 0) && holds == 0 {
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "within {DEADLINE:?} of the disconnect, its fetches still stalled, the budget is \
             returned and the hold released: budget {:?}, hold fields {holds}",
            held(&budget)
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    // Nothing waits on the stalled fetches any more; let them go regardless.
    adapter.unstall(EXT_RESTORE_PARALLELISM);
}

// ═════════════════════════════════════════════════════════════════════════════
// T6 (b.cyy) — bounded parallelism
// ═════════════════════════════════════════════════════════════════════════════

/// The backup-store latency T6 injects into every fetch.
const FETCH_LATENCY: Duration = Duration::from_millis(100);

/// The restore phase of `count` backup-only values through
/// `restore_in_batches` on a store whose backup store answers each fetch
/// after [`FETCH_LATENCY`], with a budget of `budget_total` bytes: the
/// adapter afterwards, and the phase's duration on the runtime's clock.
async fn timed_restore(count: usize, budget_total: u64) -> (Arc<CountingAdapter>, Duration) {
    let adapter = Arc::new(CountingAdapter::new(Some(FETCH_LATENCY)));
    let store = MockExternalValueStore::new()
        .with_checkpoint_store(Arc::clone(&adapter) as Arc<dyn CheckpointStore>);
    let layer_values = values(0x90, count);
    let mut restores = Vec::new();
    for value in &layer_values {
        let hash = ext_content_hash(value);
        store.seed_meta_without_data_for_test(&hash, 1);
        adapter.hold(value).await;
        restores.push((hash, value.len() as u64));
    }
    let restores = RestoreLayout::restore_for_test(restores);
    let budget = Arc::new(MemoryBudget::with_series(
        budget_total / 1024,
        Arc::new(StdMutex::new(None)),
        &CATCHUP_PAYLOAD_SERIES,
    ));
    let started = tokio::time::Instant::now();
    let lengths = restore_in_batches(
        &store,
        &restores,
        EXT_RESTORE_CHARGE_CENTI,
        budget.total_bytes(),
        |bytes| {
            let budget = Arc::clone(&budget);
            async move {
                budget.reserve(bytes, None).await.map_err(|refusal| {
                    ServiceError::Internal(format!("an idle budget refused: {refusal:?}"))
                })
            }
        },
    )
    .await
    .expect("every value restores");
    let took = started.elapsed();
    assert_eq!(lengths.len(), count, "every value answered");
    assert!(
        layer_values
            .iter()
            .all(|value| store.has_data(&ext_content_hash(value))),
        "every value written back"
    );
    (adapter, took)
}

/// T6 (b.cyy): a catch-up restores its backup-only values
/// [`EXT_RESTORE_PARALLELISM`] at a time — more than one — so the backup
/// store sees exactly that many fetches in flight at once, never more, and
/// 16 values at 100 ms a fetch take 4 fetch latencies, not 16. On tokio's
/// paused clock (deterministic: it advances only to the next timer, never
/// with the host), against the one-at-a-time restore it replaced (a budget
/// fitting one value's charge, so every batch is one value), the two
/// durations are printed for the record: model figures, not measurements.
///
/// Fails without the change (in-place perturbation): with
/// `EXT_RESTORE_PARALLELISM` set to 1, one fetch is in flight at a time.
#[tokio::test(start_paused = true)]
async fn backup_only_values_are_fetched_ext_restore_parallelism_at_a_time() {
    const VALUES: usize = 16;
    let one = share(blob(0).len());

    let (batched, batched_took) = timed_restore(VALUES, 64 * one).await;
    let (serial, serial_took) = timed_restore(VALUES, one).await;
    eprintln!(
        "T6 [model, paused clock]: restoring {VALUES} backup-only values at {FETCH_LATENCY:?} per \
         backup-store fetch: one at a time (before) {serial_took:?}; {EXT_RESTORE_PARALLELISM} at \
         a time (after) {batched_took:?}"
    );

    assert!(
        batched.max_in_flight() > 1,
        "more than one backup-store fetch in flight at once"
    );
    assert_eq!(
        batched.max_in_flight(),
        EXT_RESTORE_PARALLELISM,
        "exactly EXT_RESTORE_PARALLELISM fetches in flight at once, never more"
    );
    assert_eq!(batched.fetches(), VALUES, "each value fetched once");
    assert_eq!(
        batched_took,
        FETCH_LATENCY * VALUES.div_ceil(EXT_RESTORE_PARALLELISM) as u32,
        "one fetch latency per batch"
    );
    assert_eq!(
        (serial.max_in_flight(), serial_took),
        (1, FETCH_LATENCY * VALUES as u32),
        "fixture self-check: the one-at-a-time restore pays every fetch's latency in turn"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// b.cyy — the slot-hold A/B, on a real clock and a real store (manual)
// ═════════════════════════════════════════════════════════════════════════════

/// The backup-store latency the A/B injects into every fetch: 50 ms, the low
/// end of the 50–200 ms estimated for one backup-store read.
const AB_FETCH_LATENCY: Duration = Duration::from_millis(50);

/// The backup-only values each arm of the A/B restores.
const AB_VALUES: usize = 256;

/// Each A/B value's size: 256 KiB.
const AB_VALUE_BYTES: usize = 256 * 1024;

/// `AB_VALUES` distinct values of `AB_VALUE_BYTES`, `seed` setting them
/// apart from another arm's.
fn ab_values(seed: u8) -> Vec<Vec<u8>> {
    (0..AB_VALUES)
        .map(|i| {
            let mut value = vec![seed; AB_VALUE_BYTES];
            value[..8].copy_from_slice(&(i as u64).to_le_bytes());
            value
        })
        .collect()
}

/// `/proc/loadavg`'s first three fields, or "unavailable".
fn loadavg() -> String {
    std::fs::read_to_string("/proc/loadavg").map_or_else(
        |_| "unavailable".to_owned(),
        |text| {
            text.split_whitespace()
                .take(3)
                .collect::<Vec<_>>()
                .join(" ")
        },
    )
}

/// One arm of the A/B: [`AB_VALUES`] backup-only values seeded by `seed` (each
/// stored, then its data unlinked, so its meta stands without data and only
/// the backup store holds its bytes), restored through `restore_in_batches`
/// on a real `ValkeyExternalValueStore` whose backup store answers each fetch
/// after [`AB_FETCH_LATENCY`], with a catch-up payload budget of
/// `budget_total` bytes. Returns the adapter and the restore phase's duration
/// on the wall clock.
async fn real_clock_restore(
    tv: &TestValkey,
    seed: u8,
    budget_total: u64,
) -> (Arc<CountingAdapter>, Duration) {
    let adapter = Arc::new(CountingAdapter::new(Some(AB_FETCH_LATENCY)));
    let store = ValkeyExternalValueStore::new(tv.client().clone(), tv.key_schema().clone())
        .with_checkpoint_store(Arc::clone(&adapter) as Arc<dyn CheckpointStore>);
    let layer_values = ab_values(seed);
    let mut restores = Vec::with_capacity(AB_VALUES);
    for value in &layer_values {
        let hash = ext_content_hash(value);
        store
            .store_values(&[(&hash, value.as_slice())])
            .await
            .expect("seed the value and its meta");
        let removed: i64 = tv
            .client()
            .pool
            .unlink(tv.key_schema().ext_data(&sha256_to_hex(&hash)))
            .await
            .expect("UNLINK ext:data");
        assert_eq!(
            removed, 1,
            "fixture: the value was stored before its unlink"
        );
        adapter.hold(value).await;
        restores.push((hash, value.len() as u64));
    }
    let hashes: Vec<[u8; 32]> = restores.iter().map(|(hash, _)| *hash).collect();
    let restores = RestoreLayout::restore_for_test(restores);
    let budget = Arc::new(MemoryBudget::with_series(
        budget_total / 1024,
        Arc::new(StdMutex::new(None)),
        &CATCHUP_PAYLOAD_SERIES,
    ));
    let started = std::time::Instant::now();
    let lengths = restore_in_batches(
        &store,
        &restores,
        EXT_RESTORE_CHARGE_CENTI,
        budget.total_bytes(),
        |bytes| {
            let budget = Arc::clone(&budget);
            async move {
                budget.reserve(bytes, None).await.map_err(|refusal| {
                    ServiceError::Internal(format!("an idle budget refused: {refusal:?}"))
                })
            }
        },
    )
    .await
    .expect("every value restores");
    let took = started.elapsed();
    assert_eq!(lengths.len(), AB_VALUES, "every value answered");
    assert!(
        lengths
            .values()
            .all(|l| l.restored && l.len == AB_VALUE_BYTES as u64),
        "every value restored at its full length"
    );
    let stored = store
        .probe_value_lengths(&hashes)
        .await
        .expect("the probe after the restore");
    assert!(
        stored.missing.is_empty(),
        "every value written back to the store"
    );
    (adapter, took)
}

/// b.cyy's slot-hold A/B, report only: the restore phase of one catch-up of
/// [`AB_VALUES`] backup-only values of 256 KiB, on a real store and a real
/// clock, each backup-store fetch answering after [`AB_FETCH_LATENCY`]. The
/// restore phase is the only part of a catch-up's slot hold that differs
/// between the arms. Before: a payload budget of one value's share, so every
/// batch is one value (one fetch at a time). After: the reference plan's
/// catch-up payload budget, so batches of [`EXT_RESTORE_PARALLELISM`]. It
/// prints both durations, their ratio and the load average at start and end;
/// the T6 test above is the paused-clock model of the same comparison.
///
/// It asserts only its fixture: every value restored and written back in
/// each arm, one backup-store fetch per value, one in flight at a time
/// before and `EXT_RESTORE_PARALLELISM` after. The durations are a
/// wall-clock measurement under whatever load the host carries, so they are
/// reported, never asserted.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "manual: report only — b.cyy slot-hold A/B"]
async fn slot_hold_ab_is_reported() {
    let load_start = loadavg();
    let tv = TestValkey::setup().await;
    let one = share(AB_VALUE_BYTES);
    let reference = live_edit::config::PodMemoryPlan::reference().catchup_payload_budget_bytes();

    let (serial, serial_took) = real_clock_restore(&tv, 0x31, one).await;
    let (parallel, parallel_took) = real_clock_restore(&tv, 0x32, reference).await;
    let load_end = loadavg();

    eprintln!(
        "b.cyy slot-hold A/B [measured, real clock]: {AB_VALUES} backup-only values of \
         {AB_VALUE_BYTES} B at {AB_FETCH_LATENCY:?} per backup-store fetch; one at a time \
         (before, a budget of {one} B) {serial_took:?}; {EXT_RESTORE_PARALLELISM} at a time \
         (after, a budget of {reference} B) {parallel_took:?}; ratio {:.2}; loadavg at start \
         {load_start}, at end {load_end}",
        serial_took.as_secs_f64() / parallel_took.as_secs_f64()
    );

    assert_eq!(
        (serial.fetches(), parallel.fetches()),
        (AB_VALUES, AB_VALUES),
        "fixture: each value fetched once in each arm"
    );
    assert_eq!(
        (serial.max_in_flight(), parallel.max_in_flight()),
        (1, EXT_RESTORE_PARALLELISM),
        "fixture: one fetch in flight at a time before, EXT_RESTORE_PARALLELISM after"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// PD-7b — the recovery counter's `meta` label
// ═════════════════════════════════════════════════════════════════════════════

/// PD-7b: `live_edit_checkpoint_ext_recovery_total` labels each consult of
/// the backup store by the value's meta: a value whose meta is stored
/// without its data — the shape a restore's recount leaves — is
/// `meta="present"`, and one whose meta is missing too — a loss outside a
/// restore — is `meta="absent"`, whatever the backup store answers. Here
/// two present-meta values it holds and one it lacks, and one absent-meta
/// value it holds, are restored through `recover_value`: the cells read 2
/// and 1 `recovered`, 1 and 0 `adapter_miss`, every other cell 0. The counts
/// differ between the shapes, so a label given to the wrong shape shows.
///
/// Fails without the change (in-place perturbation): with the two labels
/// swapped, each consult lands in the other shape's cell.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_recovery_counter_labels_a_meta_without_data_present_and_an_unknown_hash_absent() {
    let tv = TestValkey::setup().await;
    let base = build_service(&tv, Arc::new(AllowAllPermissions));
    let adapter = Arc::new(CountingAdapter::new(None));
    let store = ValkeyExternalValueStore::new(tv.client().clone(), tv.key_schema().clone())
        .with_metrics(Arc::clone(&base.metrics))
        .with_checkpoint_store(Arc::clone(&adapter) as Arc<dyn CheckpointStore>);
    let present_held = [blob(0xA1), blob(0xA2)];
    let present_lost = blob(0xA3);
    let absent_held = blob(0xA4);
    for value in present_held.iter().chain([&present_lost]) {
        let _: () = tv
            .client()
            .pool
            .hset(
                tv.key_schema()
                    .ext_meta(&sha256_to_hex(&ext_content_hash(value))),
                vec![("ref_count", "1")],
            )
            .await
            .expect("HSET ext_meta: a meta without data");
    }
    for value in present_held.iter().chain([&absent_held]) {
        adapter.hold(value).await;
    }

    for value in present_held.iter().chain([&absent_held]) {
        let restored = store
            .recover_value(&ext_content_hash(value), &charge_nothing)
            .await
            .expect("a value the backup store holds restores");
        assert_eq!(&restored[..], &value[..]);
    }
    store
        .recover_value(&ext_content_hash(&present_lost), &charge_nothing)
        .await
        .expect_err("a value the backup store lacks fails its restore");

    for result in EXT_RECOVERY_RESULTS {
        for meta in EXT_RECOVERY_METAS {
            let expected = match (*result, *meta) {
                ("recovered", EXT_RECOVERY_META_PRESENT) => 2,
                ("recovered", EXT_RECOVERY_META_ABSENT) => 1,
                ("adapter_miss", EXT_RECOVERY_META_PRESENT) => 1,
                _ => 0,
            };
            assert_eq!(
                recovery_cell(&base.metrics, result, meta),
                expected,
                "{{result={result}, meta={meta}}}"
            );
        }
    }
}

/// PD-7b: every `{result, meta}` cell of
/// `live_edit_checkpoint_ext_recovery_total` — five results by two metas —
/// is registered at 0 before any consult, so a rate over either meta reads 0,
/// not absent, on a replica that never restored a value.
#[test]
fn the_recovery_counter_is_zero_seeded_over_result_and_meta() {
    let metrics = Metrics::new("d1-recovery-zero-seed");
    assert_eq!(
        (EXT_RECOVERY_RESULTS.len(), EXT_RECOVERY_METAS.len()),
        (5, 2)
    );
    for result in EXT_RECOVERY_RESULTS {
        for meta in EXT_RECOVERY_METAS {
            assert_eq!(recovery_cell(&metrics, result, meta), 0, "{result}/{meta}");
        }
    }
}
