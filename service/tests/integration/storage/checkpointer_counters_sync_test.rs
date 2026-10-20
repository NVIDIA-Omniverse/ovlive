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

// Task t2.fgj.mh.fc Subtask .h7 — integration tests for the SR-5.1
// periodic counters-sidecar write in the leader-elected checkpoint loop
// (Epic `t1.fgj.mh`, Bee `b.fgj`).
//
// The sibling engineer Subtask `.8j` lands the per-tick counters-sync
// step inside `src/checkpoint/checkpointer.rs::counters_leader_tick` /
// `run_counters_sync_tick`. The step runs on its own
// `LIVE_EDIT_CHECKPOINT_COUNTERS_SYNC_SECS` timer, is leader-gated on
// the existing `le:checkpoint:leader` lock, and writes a
// `{"max_object_id_observed": <u64>, "updated_at": "<RFC3339>"}` JSON
// blob through `CheckpointStore::put_counters_sidecar`.
//
// Monotonic rule (SDD SR-2.4, Plan b.tv3 Task t2.tv3.r3.ev): each tick
// reads the stored sidecar FIRST, then the live counter, and writes
// only when the counter strictly exceeds the stored
// `max_object_id_observed` read in that same tick (above 0 when no
// sidecar is stored). There is no in-process memory of what the tick
// last wrote or read; the per-tick read is the whole gate on the value.
// The loop keeps one bit instead: whether this process has seen the
// sidecar exist (read one that parsed, or wrote one). Once it has, a
// read answering "absent" — the backup store's cached not-found for a
// sidecar that exists — is an error, not "no floor". A counter below the
// stored value is skipped (never written); a read or parse failure of
// the stored sidecar, or that "absent", writes nothing and counts
// `error`; a `put_counters_sidecar` failure is retried by the next tick,
// which re-reads both values.
//
// Scenarios 1–5 pin the original SR-5.1 write behaviour; 6–13 pin the
// monotonic rule:
//   6. a fresh leader whose counter is below the stored sidecar never
//      lowers it, and still writes once the counter passes it;
//   7. a leadership hand-off after a simulated counter rewind never
//      lowers it, on the new leader or on the old one after it regains
//      the lock;
//   8. a lower body that lands late (a deposed leader's in-flight PUT)
//      is raised back by the next counters tick;
//   9. the stored sidecar is read before the live counter;
//  10. a failed sidecar read writes nothing, and the next tick writes;
//  11. stored bytes that do not parse write nothing until they parse;
//  12. an "absent" answer after the tick read a stored sidecar writes
//      nothing, with the counter rewound below it;
//  13. an "absent" answer after this process wrote the sidecar writes
//      nothing, with the counter rewound below it.
//
// All scenarios share heavy harness setup (Valkey + spawned loop
// + counting wrapper + `CounterStore`), so they live in a single file
// and reuse one `CountingCheckpointStore` boilerplate impl plus one
// `poll_until` helper. Mirrors the layout of
// `tests/integration/storage/checkpointer_leader_race_test.rs`.
//
// The fault-injection scenarios (#4, #10) need the `with_fault_injector`
// seam on `InMemoryCheckpointStore`, so the whole file is gated on
// `test-checkpoint-fault-injection`; without that feature every test
// here compiles out. The `#[tokio::test]`s are otherwise independent
// and have no inter-test dependencies.
//
// Waits for "the loop ran N more counters ticks" count completed ticks
// on the loop's own metrics: every tick that reaches the counters-sync
// body bumps exactly one of `live_edit_checkpoint_writes_total{result=
// success|skipped|error, kind="counters_sidecar"}`, so the count is a
// tick counter that needs no sleep.
//
// These tests use `TestValkey::setup_with_pool_size`, which reads
// `TEST_VALKEY_URL`.

#![cfg(feature = "test-checkpoint-fault-injection")]

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use fred::interfaces::KeysInterface;
use fred::types::Expiration;
use tokio::sync::Mutex;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use live_edit::checkpoint::adapters::{CheckpointFaultInjector, InMemoryCheckpointStore};
use live_edit::checkpoint::{
    spawn_checkpoint_loop, CheckpointBlob, CheckpointError, CheckpointHandle, CheckpointLoopConfig,
    CheckpointMeta, CheckpointStore, CountersSidecar, TailCaptureConfig,
};
use live_edit::observability::metrics::{Metrics, SharedMetrics};
use live_edit::storage::client::ValkeyClient;
use live_edit::storage::counters::CounterStore;
use live_edit::storage::external_values::LEADER_LOCK_TTL_MS;
use live_edit::storage::test_support::TestValkey;

// ────────────────────────────────────────────────────────────────────────
// Harness: counting wrapper that records every `put_counters_sidecar`
// call with the request body and observed result, and counts every
// `get_counters_sidecar` call, while delegating every other trait
// method straight through to an inner shared `InMemoryCheckpointStore`.
// The wrapper's identity is by-reference; callers wrap it in
// `Arc<CountingCheckpointStore>` to share between the spawned loop and
// the asserting test body.
// ────────────────────────────────────────────────────────────────────────

/// One recorded `put_counters_sidecar` invocation: the body bytes
/// passed into the call, the `Result` the underlying adapter returned,
/// and how many `get_counters_sidecar` calls the wrapper had seen when
/// the PUT was issued. Recording happens AFTER the inner call so the
/// result is available; the body bytes are recorded regardless of
/// success so scenario #4 can prove the failed attempt was actually
/// observed.
#[derive(Debug)]
struct PutCountersCall {
    body: Bytes,
    /// `Ok(())` on success, the error stringified on failure. We
    /// stringify so the recording can be compared by-value without
    /// requiring `CheckpointError: Clone` (it isn't, because its
    /// `Transient` source is a `Box<dyn Error + Send + Sync>`).
    result_err: Option<String>,
    /// `get_counters_sidecar` calls made through this wrapper before
    /// this PUT. Each counters tick issues exactly one GET before its
    /// PUT (and the per-layer and reconcile ticks issue none), so with
    /// no test-body reads in between this is the 1-based index of the
    /// tick that wrote: scenarios #9 and #10 use it to name the tick a
    /// write came from.
    gets_before: u64,
}

/// Scenario #9's read-order probe. On the FIRST `get_counters_sidecar`
/// through the wrapper, and before that call reads the inner store, it
/// raises the live `le:counter:object_id` to `raise_counter_to` and
/// stores a sidecar of `store_sidecar_at` straight into the inner
/// store. A tick that reads the sidecar first sees
/// `store_sidecar_at`, then the raised counter, and writes; a tick that
/// reads the counter first sees the counter from before the raise,
/// which is below `store_sidecar_at`, and skips.
struct FirstGetRaise {
    client: ValkeyClient,
    counter_key: String,
    raise_counter_to: u64,
    store_sidecar_at: u64,
    fired: AtomicBool,
}

impl FirstGetRaise {
    async fn fire(&self, inner: &InMemoryCheckpointStore) {
        let _: () = self
            .client
            .pool
            .set(
                &self.counter_key,
                self.raise_counter_to.to_string(),
                None,
                None,
                false,
            )
            .await
            .expect("first-GET hook: SET le:counter:object_id must succeed");
        inner
            .put_counters_sidecar(sidecar_body(self.store_sidecar_at))
            .await
            .expect("first-GET hook: storing the sidecar in the inner store must succeed");
    }
}

/// Counting wrapper composed atop `crate::checkpoint::helpers::DelegatingCheckpointStore`
/// (b.dzd Phase 2). Overrides `put_counters_sidecar` to record every
/// invocation in a `Mutex<Vec<...>>` and `get_counters_sidecar` to count
/// calls (and run scenario #9's optional [`FirstGetRaise`], or answer
/// scenarios #12 and #13's stale "absent"); every other trait method
/// delegates verbatim through the embedded `DelegatingCheckpointStore`.
/// The inner adapter is wrapped in `Arc` so scenarios #5 and #7 can hand
/// the same backing state to two different counting wrappers (one per
/// replica).
struct CountingCheckpointStore {
    delegate: crate::checkpoint::helpers::DelegatingCheckpointStore<InMemoryCheckpointStore>,
    put_counters_calls: Mutex<Vec<PutCountersCall>>,
    get_counters_calls: AtomicU64,
    first_get_raise: Option<FirstGetRaise>,
    /// When `answer_absent` gets set.
    stale_absent: StaleAbsent,
    /// Once set, every `get_counters_sidecar` answers `Ok(None)` without
    /// reading the inner store, whatever it holds: the storage-api backend's
    /// cached not-found answer for a sidecar that exists.
    answer_absent: AtomicBool,
}

/// When a [`CountingCheckpointStore`] starts answering its stale "absent"
/// (scenarios #12 and #13). Each trigger fires inside the wrapper, between
/// two of the loop's own calls, so no tick can read across it.
enum StaleAbsent {
    Never,
    /// From the `get_counters_sidecar` after the first `n`.
    AfterGets(u64),
    /// From the first `get_counters_sidecar` after a `put_counters_sidecar`
    /// that succeeded.
    AfterSuccessfulPut,
}

impl CountingCheckpointStore {
    /// Wrap an existing shared inner adapter. The caller decides
    /// whether the inner is fresh (`Arc::new(InMemoryCheckpointStore::new())`)
    /// or shared across wrappers (scenarios #5 and #7).
    fn new(inner: Arc<InMemoryCheckpointStore>) -> Self {
        Self {
            delegate: crate::checkpoint::helpers::DelegatingCheckpointStore::new(inner),
            put_counters_calls: Mutex::new(Vec::new()),
            get_counters_calls: AtomicU64::new(0),
            first_get_raise: None,
            stale_absent: StaleAbsent::Never,
            answer_absent: AtomicBool::new(false),
        }
    }

    /// Arm scenario #9's read-order probe on this wrapper.
    fn with_first_get_raise(mut self, hook: FirstGetRaise) -> Self {
        self.first_get_raise = Some(hook);
        self
    }

    /// Answer `Ok(None)` from `get_counters_sidecar` once `trigger` fires,
    /// while the inner store keeps whatever it holds.
    fn with_stale_absent(mut self, trigger: StaleAbsent) -> Self {
        self.stale_absent = trigger;
        self
    }

    /// Snapshot of every recorded `put_counters_sidecar` invocation,
    /// in call-order. The wrapper's recording site sits AFTER the
    /// inner call returns so call-order equals adapter-order.
    async fn put_counters_calls_snapshot(&self) -> Vec<PutCountersCall> {
        self.put_counters_calls
            .lock()
            .await
            .iter()
            .map(|c| PutCountersCall {
                body: c.body.clone(),
                result_err: c.result_err.clone(),
                gets_before: c.gets_before,
            })
            .collect()
    }
}

#[async_trait]
impl CheckpointStore for CountingCheckpointStore {
    async fn put_layer_checkpoint(
        &self,
        object_id: u64,
        snap_seq: u64,
        ompf: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        self.delegate
            .put_layer_checkpoint(object_id, snap_seq, ompf, meta)
            .await
    }

    async fn get_latest_layer_checkpoint(
        &self,
        object_id: u64,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.delegate.get_latest_layer_checkpoint(object_id).await
    }

    // Forwarded per `CheckpointStore::get_layer_checkpoint`'s MUST — its trait
    // default returns `NotImplemented`, which would mask a generation the
    // delegate holds rather than merely leaving the call un-instrumented.
    async fn get_layer_checkpoint(
        &self,
        handle: &CheckpointHandle,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.delegate.get_layer_checkpoint(handle).await
    }

    async fn list_layer_checkpoints(
        &self,
        object_id: u64,
    ) -> Result<Vec<CheckpointHandle>, CheckpointError> {
        self.delegate.list_layer_checkpoints(object_id).await
    }

    async fn delete_layer_checkpoint(
        &self,
        handle: CheckpointHandle,
    ) -> Result<(), CheckpointError> {
        self.delegate.delete_layer_checkpoint(handle).await
    }

    async fn put_session_checkpoint(
        &self,
        session_id: &str,
        body: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        self.delegate
            .put_session_checkpoint(session_id, body, meta)
            .await
    }

    async fn get_session_checkpoint(
        &self,
        session_id: &str,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.delegate.get_session_checkpoint(session_id).await
    }

    async fn delete_session_checkpoint(&self, session_id: &str) -> Result<(), CheckpointError> {
        self.delegate.delete_session_checkpoint(session_id).await
    }

    async fn list_session_checkpoint_ids(&self) -> Result<Vec<String>, CheckpointError> {
        self.delegate.list_session_checkpoint_ids().await
    }

    async fn put_ext_blob(&self, hash_hex: &str, bytes: Bytes) -> Result<(), CheckpointError> {
        self.delegate.put_ext_blob(hash_hex, bytes).await
    }

    async fn get_ext_blob(&self, hash_hex: &str) -> Result<Option<Bytes>, CheckpointError> {
        self.delegate.get_ext_blob(hash_hex).await
    }

    async fn ext_blob_exists(&self, hash_hex: &str) -> Result<bool, CheckpointError> {
        self.delegate.ext_blob_exists(hash_hex).await
    }

    async fn delete_ext_blob(&self, hash_hex: &str) -> Result<(), CheckpointError> {
        self.delegate.delete_ext_blob(hash_hex).await
    }

    async fn put_counters_sidecar(&self, body: Bytes) -> Result<(), CheckpointError> {
        let gets_before = self.get_counters_calls.load(Ordering::SeqCst);
        // Delegate FIRST so failure injection on the inner adapter is
        // honoured and the recorded result matches what the loop's
        // failure-classification arm sees.
        let result = self.delegate.put_counters_sidecar(body.clone()).await;
        // Record AFTER the inner call so the recorded `result_err`
        // captures the actual observed outcome — scenario #4 asserts
        // the first recorded call carries an Err and the second
        // carries an Ok.
        let result_err = result.as_ref().err().map(|e| format!("{e}"));
        self.put_counters_calls.lock().await.push(PutCountersCall {
            body,
            result_err,
            gets_before,
        });
        if result.is_ok() && matches!(self.stale_absent, StaleAbsent::AfterSuccessfulPut) {
            self.answer_absent.store(true, Ordering::SeqCst);
        }
        result
    }

    async fn get_counters_sidecar(&self) -> Result<Option<Bytes>, CheckpointError> {
        let gets = self.get_counters_calls.fetch_add(1, Ordering::SeqCst) + 1;
        if let Some(hook) = &self.first_get_raise {
            if !hook.fired.swap(true, Ordering::SeqCst) {
                hook.fire(&self.delegate.inner).await;
            }
        }
        if let StaleAbsent::AfterGets(n) = self.stale_absent {
            if gets > n {
                self.answer_absent.store(true, Ordering::SeqCst);
            }
        }
        if self.answer_absent.load(Ordering::SeqCst) {
            return Ok(None);
        }
        self.delegate.get_counters_sidecar().await
    }

    async fn list_all_object_ids(&self) -> Result<Vec<u64>, CheckpointError> {
        self.delegate.list_all_object_ids().await
    }

    async fn list_referenced_ext_hashes(&self) -> Result<HashSet<String>, CheckpointError> {
        self.delegate.list_referenced_ext_hashes().await
    }

    async fn list_live_ext_set(
        &self,
    ) -> Result<live_edit::checkpoint::LiveExtSet, CheckpointError> {
        self.delegate.list_live_ext_set().await
    }
}

// ────────────────────────────────────────────────────────────────────────
// Generic helpers shared across the scenarios.
// ────────────────────────────────────────────────────────────────────────

/// Bounded-poll a future-returning closure until it yields `true` or
/// the deadline expires. Same shape as
/// `checkpointer_leader_race_test.rs::poll_until` — every wait in
/// this file sits behind an explicit deadline so a hung loop surfaces
/// as a deterministic timeout rather than a 60 s test stall.
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
        // 25 ms between probes keeps the poll cost modest against the
        // 100 ms per-layer tick and the 200 ms counters-sync timer.
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    pred().await
}

/// Construct a base `CheckpointLoopConfig` with the per-layer tick at
/// 100 ms and the counters-sync timer at 200 ms. With these knobs each
/// observation window of 1–2 s sees several counters-sync ticks —
/// enough for "advance writes once" / "stable counter no-op" assertions
/// to hold robustly under CI scheduling jitter. `interval` and
/// `quiesce` are kept at zero so the per-layer skip gates do not
/// interfere with the leader-lock probe cadence; this test exercises
/// only the counters-sync timer arm of the `tokio::select!`.
fn loop_config(replica_id: &str) -> CheckpointLoopConfig {
    CheckpointLoopConfig {
        tick: Duration::from_millis(100),
        interval: Duration::ZERO,
        quiesce: Duration::ZERO,
        retain_generations: 2,
        // 200 ms cadence so every test window sees at least 3 ticks
        // inside a 1 s observation budget (each scenario's `poll_until`
        // deadlines are 3–8 s, giving generous CI headroom).
        counters_sync: Duration::from_millis(200),
        reconcile_interval: Duration::from_secs(3600),
        leader: crate::common::test_leader(replica_id),
        // Tails pinned explicitly OFF. b.3n1 promoted the binary default
        // to `true`; this suite predates the tail feature and was written
        // against an inert tail path, so the flag is part of the fixture
        // rather than something to inherit.
        tail: TailCaptureConfig {
            enabled: false,
            ..TailCaptureConfig::default()
        },
    }
}

/// Cancel a `CancellationToken` and bounded-join the spawned task. Mirrors
/// the teardown idiom from `checkpointer_leader_race_test.rs` — a
/// stuck loop surfaces as a clear assertion failure here rather than
/// as a 60 s `cargo test` hang.
async fn cancel_and_join(cancel: &CancellationToken, handle: tokio::task::JoinHandle<()>) {
    cancel.cancel();
    let join_outcome = tokio::time::timeout(Duration::from_secs(5), handle).await;
    assert!(
        join_outcome.is_ok(),
        "checkpoint loop join must complete within 5 s of cancellation; \
         a stuck shutdown would have wedged main.rs's drain budget too",
    );
    join_outcome
        .expect("timeout already checked")
        .expect("checkpoint loop task must not panic on shutdown");
}

/// One-shot fault injector for one counters-sidecar op: returns
/// `Transient` on the FIRST call against `target_op` and `Ok(())`
/// forever after (every other op passes). Scenario #4 targets
/// `put_counters_sidecar` to drive the loop's PUT-failure arm;
/// scenario #10 targets `get_counters_sidecar` to drive its
/// sidecar-read-failure arm. Latched via `AtomicBool` so a concurrent
/// re-entry (impossible in practice with a single leader, but cheap to
/// guard) would still fire at most once.
struct CountersOneShotFailure {
    target_op: &'static str,
    fired: AtomicBool,
}

impl CountersOneShotFailure {
    fn new(target_op: &'static str) -> Self {
        Self {
            target_op,
            fired: AtomicBool::new(false),
        }
    }
}

impl CheckpointFaultInjector for CountersOneShotFailure {
    fn maybe_inject(&self, op: &str) -> Result<(), CheckpointError> {
        if op == self.target_op && !self.fired.swap(true, Ordering::SeqCst) {
            Err(CheckpointError::Transient {
                source: Box::new(std::io::Error::other(format!("synthetic {op} failure"))),
            })
        } else {
            Ok(())
        }
    }
}

/// Parse the persisted counters-sidecar body and assert it is
/// well-formed: `max_object_id_observed == expected` and `updated_at`
/// parses as RFC3339. Returns the parsed value for downstream
/// assertions.
fn decode_and_check_sidecar(bytes: &Bytes, expected_max: u64) -> CountersSidecar {
    let sidecar: CountersSidecar = serde_json::from_slice(bytes)
        .unwrap_or_else(|e| panic!("counters sidecar JSON must parse via shared type: {e}"));
    assert_eq!(
        sidecar.max_object_id_observed, expected_max,
        "counters sidecar max_object_id_observed must equal the live counter value at write time",
    );
    chrono::DateTime::parse_from_rfc3339(&sidecar.updated_at).unwrap_or_else(|e| {
        panic!(
            "counters sidecar updated_at must parse as RFC3339; got {:?} ({e})",
            sidecar.updated_at
        )
    });
    sidecar
}

/// Budget for every "wait for the loop" poll added with scenarios
/// 6–11: ~50 counters ticks at the 200 ms cadence. A deadline, not a
/// sleep — the healthy path returns within a few ticks.
const LOOP_WAIT: Duration = Duration::from_secs(10);

/// A well-formed counters-sidecar body carrying `max`, as the loop
/// itself would write it. Used to stand a sidecar up in the inner store
/// directly, bypassing the loop and the counting wrapper.
fn sidecar_body(max: u64) -> Bytes {
    Bytes::from(
        serde_json::to_vec(&CountersSidecar {
            max_object_id_observed: max,
            updated_at: chrono::Utc::now().to_rfc3339(),
        })
        .expect("CountersSidecar must serialize"),
    )
}

/// The `max_object_id_observed` currently stored in `inner`, read
/// straight from the inner store so the read is not counted by any
/// wrapper. Panics when nothing is stored or the bytes do not parse.
async fn stored_max(inner: &InMemoryCheckpointStore) -> u64 {
    let bytes = inner
        .get_counters_sidecar()
        .await
        .expect("inner get_counters_sidecar must succeed")
        .expect("a counters sidecar must be stored");
    serde_json::from_slice::<CountersSidecar>(&bytes)
        .expect("stored counters sidecar must parse")
        .max_object_id_observed
}

/// Every recorded PUT's `max_object_id_observed`, in call order.
fn recorded_maxes(calls: &[PutCountersCall]) -> Vec<u64> {
    calls
        .iter()
        .map(|c| {
            serde_json::from_slice::<CountersSidecar>(&c.body)
                .expect("every recorded body must decode")
                .max_object_id_observed
        })
        .collect()
}

/// Set `le:counter:object_id` to `value` directly. Stands in for any
/// path that moves the live counter (allocation, or a rewind the
/// service cannot cause itself).
async fn set_object_id_counter(tv: &TestValkey, value: u64) {
    let _: () = tv
        .client()
        .pool
        .set(
            tv.key_schema().counter_object_id(),
            value.to_string(),
            None,
            None,
            false,
        )
        .await
        .expect("SET le:counter:object_id must succeed");
}

/// Hand `le:checkpoint:leader` to the replica whose lock value is
/// `lock_value`, with the production TTL. The acquire script refreshes
/// a lock already holding the caller's value and refuses any other, so
/// the named replica leads from its next probe on and every other
/// replica stops.
async fn pin_leader_lock(tv: &TestValkey, lock_value: &str) {
    let _: () = tv
        .client()
        .pool
        .set(
            tv.key_schema().checkpoint_leader_lock(),
            lock_value,
            Some(Expiration::PX(LEADER_LOCK_TTL_MS as i64)),
            None,
            false,
        )
        .await
        .expect("SET le:checkpoint:leader must succeed");
}

fn make_metrics(label: &str) -> SharedMetrics {
    Arc::new(crate::common::metrics_scrape_only(label))
}

/// `live_edit_checkpoint_writes_total{result, kind="counters_sidecar"}`.
fn counters_writes(metrics: &Metrics, result: &str) -> u64 {
    live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(metrics),
        "live_edit_checkpoint_writes_total",
        &[("result", result), ("kind", "counters_sidecar")],
    )
    .unwrap_or(0.0) as u64
}

/// Completed counters ticks on the loop that owns `metrics`: every tick
/// that reaches the counters-sync body bumps exactly one result.
fn counters_ticks(metrics: &Metrics) -> u64 {
    ["success", "skipped", "error"]
        .iter()
        .map(|result| counters_writes(metrics, result))
        .sum()
}

/// Wait until the loop that owns `metrics` has completed at least
/// `at_least` counters ticks in total, failing with `what` if it does
/// not within [`LOOP_WAIT`].
async fn wait_for_counters_ticks(metrics: &SharedMetrics, at_least: u64, what: &str) {
    let metrics_for_poll = Arc::clone(metrics);
    let reached = poll_until(LOOP_WAIT, || {
        let metrics = Arc::clone(&metrics_for_poll);
        async move { counters_ticks(&metrics) >= at_least }
    })
    .await;
    assert!(
        reached,
        "{what}: expected ≥ {at_least} completed counters ticks within {LOOP_WAIT:?}, saw {}",
        counters_ticks(metrics),
    );
}

/// Wait until the loop that owns `metrics` has counted at least
/// `at_least` counters ticks with `result`, failing with `what` if it
/// does not within [`LOOP_WAIT`]. A `success` is counted only after the
/// PUT returned, so once this returns for `success` the counting
/// wrapper has already recorded that PUT.
async fn wait_for_counters_result(
    metrics: &SharedMetrics,
    result: &'static str,
    at_least: u64,
    what: &str,
) {
    let metrics_for_poll = Arc::clone(metrics);
    let reached = poll_until(LOOP_WAIT, || {
        let metrics = Arc::clone(&metrics_for_poll);
        async move { counters_writes(&metrics, result) >= at_least }
    })
    .await;
    assert!(
        reached,
        "{what}: expected ≥ {at_least} counters ticks with result={result} within \
         {LOOP_WAIT:?}, saw {}",
        counters_writes(metrics, result),
    );
}

// ════════════════════════════════════════════════════════════════════════
// Scenario 1 — counter advance → tick writes the JSON blob.
//
// Drive `counters.next_object_id()` five times so the live counter
// lands at 5. Wait for the counting wrapper to record at least one
// `put_counters_sidecar` call (3 s deadline → ~15 counters-sync ticks
// at the 200 ms cadence). Round-trip the persisted body through
// `store.get_counters_sidecar` and parse via `CountersSidecar`; assert
// `max_object_id_observed == 5` and `updated_at` is RFC3339.
// ════════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn counters_sync_advance_writes_sidecar() {
    let tv = TestValkey::setup_with_pool_size(8).await;
    let keys = tv.key_schema().clone();
    let counters = CounterStore::new(tv.client().clone(), keys.clone());

    let inner = Arc::new(InMemoryCheckpointStore::new());
    let counting = Arc::new(CountingCheckpointStore::new(inner));
    let store: Arc<dyn CheckpointStore> = counting.clone();

    // b.b2j: `next_object_id` fails closed on an absent `le:counter:object_id`,
    // so establish the key the way the startup path does before allocating.
    counters
        .initialize_object_id_counter_if_absent()
        .await
        .expect("establishing le:counter:object_id must succeed");

    // Drive the live counter to 5 BEFORE spawning the loop so the
    // first counters-sync tick observes a non-zero starting value.
    for expected in 1..=5_u64 {
        let id = counters
            .next_object_id()
            .await
            .expect("next_object_id must succeed")
            .expect("counter key was just established, so allocation must not fail closed");
        assert_eq!(id, expected, "INCR must yield the expected next id");
    }

    let cancel = CancellationToken::new();
    let handle = spawn_checkpoint_loop(
        tv.client().clone(),
        keys.clone(),
        store,
        loop_config("counters-sync-advance-replica"),
        None,
        cancel.clone(),
    );

    // Wait up to 10 s for at least one recorded write. The 200 ms
    // counters-sync cadence + 100 ms tick give the loop ~50 attempts
    // inside the budget; on a healthy CI host the first write lands
    // within the first 1–2 ticks. The 10 s budget matches the sibling
    // `checkpointer_metrics_test.rs::case_c_label_cardinality_is_bounded`
    // pattern and absorbs scheduler / Valkey contention when the full
    // workspace test suite runs in parallel.
    let counting_for_poll = Arc::clone(&counting);
    let saw_write = poll_until(Duration::from_secs(10), || {
        let counting = Arc::clone(&counting_for_poll);
        async move { !counting.put_counters_calls_snapshot().await.is_empty() }
    })
    .await;
    assert!(
        saw_write,
        "leader must issue at least one put_counters_sidecar within 10 s — \
         the counter advanced to 5 BEFORE the loop spawned, so the first \
         counters-sync tick must observe no stored sidecar and n=5, and write",
    );

    // Round-trip the persisted bytes through the store's read path so
    // we exercise the same retrieval contract that Task 4's startup
    // recovery hook depends on.
    let persisted = counting
        .get_counters_sidecar()
        .await
        .expect("get_counters_sidecar must succeed")
        .expect("counters sidecar must be present after the first write");
    let _ = decode_and_check_sidecar(&persisted, 5);

    cancel_and_join(&cancel, handle).await;
}

// ════════════════════════════════════════════════════════════════════════
// Scenario 2 — stable counter across two ticks → second tick is a no-op.
//
// Drive `next_object_id()` to 3. Wait for ONE recorded write.
// Record the snapshot length. Sleep long enough for several more
// counters-sync ticks to elapse (`2 × tick + counters_sync` worth at
// minimum), then assert the snapshot length is UNCHANGED. Every later
// tick reads the stored sidecar back at 3 with the counter still at 3,
// so this pins "no PUT when the counter equals the stored value".
// ════════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn counters_sync_stable_counter_is_noop() {
    let tv = TestValkey::setup_with_pool_size(8).await;
    let keys = tv.key_schema().clone();
    let counters = CounterStore::new(tv.client().clone(), keys.clone());

    let inner = Arc::new(InMemoryCheckpointStore::new());
    let counting = Arc::new(CountingCheckpointStore::new(inner));
    let store: Arc<dyn CheckpointStore> = counting.clone();

    // b.b2j: establish `le:counter:object_id` before allocating — the
    // allocator fails closed on an absent counter.
    counters
        .initialize_object_id_counter_if_absent()
        .await
        .unwrap();
    for _ in 0..3_u64 {
        counters.next_object_id().await.unwrap().unwrap();
    }

    let cancel = CancellationToken::new();
    let handle = spawn_checkpoint_loop(
        tv.client().clone(),
        keys.clone(),
        store,
        loop_config("counters-sync-stable-replica"),
        None,
        cancel.clone(),
    );

    // First write: bounded-poll to confirm the loop made it past the
    // first counters-sync tick.
    let counting_for_poll = Arc::clone(&counting);
    let saw_first = poll_until(Duration::from_secs(10), || {
        let counting = Arc::clone(&counting_for_poll);
        async move { !counting.put_counters_calls_snapshot().await.is_empty() }
    })
    .await;
    assert!(
        saw_first,
        "first counters-sync write must land within 10 s of spawn",
    );
    let baseline_calls = counting.put_counters_calls_snapshot().await;
    let baseline_len = baseline_calls.len();
    assert_eq!(
        baseline_len, 1,
        "exactly one write must land before the counter goes stable; \
         saw {baseline_len} writes",
    );
    // First write observed `n=3` (the live counter at spawn time).
    let first = decode_and_check_sidecar(&baseline_calls[0].body, 3);
    assert_eq!(first.max_object_id_observed, 3);

    // Soak: 1.5 s lets at least 7 more counters-sync ticks elapse at
    // the 200 ms cadence. The Subtask body asks for ≥ 2 × tick +
    // counters_sync_interval worth of real time (≈ 400 ms), so 1.5 s
    // is generously above that.
    tokio::time::sleep(Duration::from_millis(1500)).await;

    let after_soak = counting.put_counters_calls_snapshot().await;
    assert_eq!(
        after_soak.len(),
        baseline_len,
        "with the counter stable, subsequent ticks MUST be no-ops; \
         observed {} new writes during the 1.5 s soak",
        after_soak.len() - baseline_len,
    );

    cancel_and_join(&cancel, handle).await;
}

// ════════════════════════════════════════════════════════════════════════
// Scenario 3 — advancing counter across multiple ticks → each tick
// writes once.
//
// Drive counter to 2 → wait for one write at `n=2`. Drive to 4 → wait
// for write at `n=4`. Drive to 5 → wait for write at `n=5`. At the end
// snapshot.len() == 3 and recorded max_object_id_observed sequence is
// `[2, 4, 5]`.
// ════════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn counters_sync_advancing_counter_writes_each_tick() {
    let tv = TestValkey::setup_with_pool_size(8).await;
    let keys = tv.key_schema().clone();
    let counters = CounterStore::new(tv.client().clone(), keys.clone());

    let inner = Arc::new(InMemoryCheckpointStore::new());
    let counting = Arc::new(CountingCheckpointStore::new(inner));
    let store: Arc<dyn CheckpointStore> = counting.clone();

    // b.b2j: establish `le:counter:object_id` before allocating — the
    // allocator fails closed on an absent counter.
    counters
        .initialize_object_id_counter_if_absent()
        .await
        .unwrap();

    // Allocate ids 1 and 2 BEFORE spawning so the first tick observes
    // `n=2` rather than racing against the loop. Subsequent
    // increments are interleaved with bounded-polls.
    counters.next_object_id().await.unwrap().unwrap();
    counters.next_object_id().await.unwrap().unwrap();

    let cancel = CancellationToken::new();
    let handle = spawn_checkpoint_loop(
        tv.client().clone(),
        keys.clone(),
        store,
        loop_config("counters-sync-advancing-replica"),
        None,
        cancel.clone(),
    );

    // Wait for the first write to observe `n=2`.
    let counting_for_poll = Arc::clone(&counting);
    let saw_two = poll_until(Duration::from_secs(10), || {
        let counting = Arc::clone(&counting_for_poll);
        async move {
            let calls = counting.put_counters_calls_snapshot().await;
            calls.iter().any(|c| {
                serde_json::from_slice::<CountersSidecar>(&c.body)
                    .map(|s| s.max_object_id_observed == 2)
                    .unwrap_or(false)
            })
        }
    })
    .await;
    assert!(saw_two, "first write at n=2 must land within 10 s");

    // Drive counter to 4 (ids 3 and 4).
    counters.next_object_id().await.unwrap().unwrap();
    counters.next_object_id().await.unwrap().unwrap();

    let counting_for_poll = Arc::clone(&counting);
    let saw_four = poll_until(Duration::from_secs(10), || {
        let counting = Arc::clone(&counting_for_poll);
        async move {
            let calls = counting.put_counters_calls_snapshot().await;
            calls.iter().any(|c| {
                serde_json::from_slice::<CountersSidecar>(&c.body)
                    .map(|s| s.max_object_id_observed == 4)
                    .unwrap_or(false)
            })
        }
    })
    .await;
    assert!(saw_four, "second write at n=4 must land within 10 s");

    // Drive counter to 5 (id 5).
    counters.next_object_id().await.unwrap().unwrap();

    let counting_for_poll = Arc::clone(&counting);
    let saw_five = poll_until(Duration::from_secs(10), || {
        let counting = Arc::clone(&counting_for_poll);
        async move {
            let calls = counting.put_counters_calls_snapshot().await;
            calls.iter().any(|c| {
                serde_json::from_slice::<CountersSidecar>(&c.body)
                    .map(|s| s.max_object_id_observed == 5)
                    .unwrap_or(false)
            })
        }
    })
    .await;
    assert!(saw_five, "third write at n=5 must land within 10 s");

    // Pin the recorded sequence exactly. We rely on the no-advance
    // gate to filter out duplicate writes between the three advances —
    // there were no stable-counter intervals long enough for the loop
    // to see the counter as advanced twice at the same value.
    let final_calls = counting.put_counters_calls_snapshot().await;
    assert_eq!(
        final_calls.len(),
        3,
        "exactly three writes must land — one per counter advance window. \
         Recorded {} calls; payload n-values: {:?}",
        final_calls.len(),
        final_calls
            .iter()
            .map(|c| serde_json::from_slice::<CountersSidecar>(&c.body)
                .map(|s| s.max_object_id_observed)
                .unwrap_or(u64::MAX))
            .collect::<Vec<_>>(),
    );
    let observed_sequence: Vec<u64> = final_calls
        .iter()
        .map(|c| {
            serde_json::from_slice::<CountersSidecar>(&c.body)
                .expect("every recorded body must decode")
                .max_object_id_observed
        })
        .collect();
    assert_eq!(
        observed_sequence,
        vec![2, 4, 5],
        "recorded max_object_id_observed sequence must be [2, 4, 5]",
    );

    // Every recorded call must also have succeeded — no transient
    // failures expected on the plain in-memory adapter.
    for (idx, call) in final_calls.iter().enumerate() {
        assert!(
            call.result_err.is_none(),
            "recorded call #{idx} must have succeeded; got error: {:?}",
            call.result_err,
        );
    }

    cancel_and_join(&cancel, handle).await;
}

// ════════════════════════════════════════════════════════════════════════
// Scenario 4 — `put_counters_sidecar` failure → next tick retries; with
// the counter unchanged, the retry carries the same value.
//
// Install a one-shot `Transient` injector on the inner adapter keyed
// on `op == "put_counters_sidecar"`. Drive counter to 3. Wait until
// the wrapper has recorded ≥ 2 calls. Assert the FIRST recorded call
// has `result_err = Some(_)` and the SECOND has the same
// `max_object_id_observed = 3` and `result_err = None`. The
// `get_counters_sidecar` post-read confirms the persisted body
// reflects the retry's payload.
// ════════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn counters_sync_put_failure_retries_with_same_value() {
    let tv = TestValkey::setup_with_pool_size(8).await;
    let keys = tv.key_schema().clone();
    let counters = CounterStore::new(tv.client().clone(), keys.clone());

    // Install the injector ON the inner adapter so the failure is
    // observable through the trait method's return value (the
    // counting wrapper observes the failed result post-delegate).
    let injector: Arc<dyn CheckpointFaultInjector> =
        Arc::new(CountersOneShotFailure::new("put_counters_sidecar"));
    let inner = Arc::new(InMemoryCheckpointStore::new().with_fault_injector(injector));
    let counting = Arc::new(CountingCheckpointStore::new(inner));
    let store: Arc<dyn CheckpointStore> = counting.clone();

    // b.b2j: establish `le:counter:object_id` before allocating — the
    // allocator fails closed on an absent counter.
    counters
        .initialize_object_id_counter_if_absent()
        .await
        .unwrap();
    for _ in 0..3_u64 {
        counters.next_object_id().await.unwrap().unwrap();
    }

    let cancel = CancellationToken::new();
    let handle = spawn_checkpoint_loop(
        tv.client().clone(),
        keys.clone(),
        store,
        loop_config("counters-sync-retry-replica"),
        None,
        cancel.clone(),
    );

    // Wait until the wrapper has recorded ≥ 2 calls. The first call
    // hits the one-shot injector and surfaces `Transient`; the loop
    // logs warn and the store still holds no sidecar, so the NEXT
    // counters-sync tick re-reads both values and re-fires with the
    // same `n=3`. 5 s budget gives ~25 ticks at the 200 ms cadence.
    let counting_for_poll = Arc::clone(&counting);
    let saw_two = poll_until(Duration::from_secs(5), || {
        let counting = Arc::clone(&counting_for_poll);
        async move { counting.put_counters_calls_snapshot().await.len() >= 2 }
    })
    .await;
    assert!(
        saw_two,
        "loop must retry put_counters_sidecar on the next tick after \
         a transient failure; only {} calls observed within 5 s",
        counting.put_counters_calls_snapshot().await.len(),
    );

    let calls = counting.put_counters_calls_snapshot().await;
    assert!(
        calls.len() >= 2,
        "expected ≥ 2 recorded calls, saw {}",
        calls.len(),
    );

    // Call #1 hit the injector → `result_err = Some(_)`.
    let first = &calls[0];
    assert!(
        first.result_err.is_some(),
        "first recorded call must have surfaced the injected Transient \
         failure; got result_err = None (the injector did not fire?)",
    );
    let first_sidecar = decode_and_check_sidecar(&first.body, 3);
    assert_eq!(first_sidecar.max_object_id_observed, 3);

    // Call #2 retried with the SAME observed value AND succeeded —
    // call #1's failure left nothing stored that would gate the retry.
    let second = &calls[1];
    assert!(
        second.result_err.is_none(),
        "second recorded call must have succeeded; got result_err = {:?}",
        second.result_err,
    );
    let second_sidecar = decode_and_check_sidecar(&second.body, 3);
    assert_eq!(
        second_sidecar.max_object_id_observed, 3,
        "retry MUST carry the same `max_object_id_observed` as the failed \
         attempt — the counter did not move and a failed PUT must not \
         suppress the retry",
    );

    // The persisted body reflects the successful retry.
    let persisted = counting
        .get_counters_sidecar()
        .await
        .expect("get_counters_sidecar must succeed after the retry landed")
        .expect("persisted counters sidecar must be present");
    let persisted_sidecar = decode_and_check_sidecar(&persisted, 3);
    assert_eq!(persisted_sidecar.max_object_id_observed, 3);

    cancel_and_join(&cancel, handle).await;
}

// ════════════════════════════════════════════════════════════════════════
// Scenario 5 — non-leader replica observes no counters-sidecar writes.
//
// Spawn two `spawn_checkpoint_loop` instances against the same Valkey
// + KeySchema with distinct leader identities and a SHARED inner
// `InMemoryCheckpointStore` behind two distinct counting wrappers
// (one per replica). Drive counter to 5. Wait for either wrapper to
// record a write; soak; assert exactly one wrapper recorded writes
// (the leader) and the other recorded zero. Cross-check against
// `GET le:checkpoint:leader` to pin which replica's lock value
// (`LeaderIdentity::lock_value()`, `label:instance`) won.
// ════════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn counters_sync_non_leader_observes_no_writes() {
    let tv = TestValkey::setup_with_pool_size(8).await;
    let keys = tv.key_schema().clone();
    let counters = CounterStore::new(tv.client().clone(), keys.clone());

    // Shared inner adapter so both replicas observe the same backing
    // state — modelling the production wiring where two pods write to
    // the same durable adapter (same LocalDisk volume or same Azure
    // container).
    let inner_shared = Arc::new(InMemoryCheckpointStore::new());
    let counting_leader = Arc::new(CountingCheckpointStore::new(inner_shared.clone()));
    let counting_follower = Arc::new(CountingCheckpointStore::new(inner_shared.clone()));
    let store_a: Arc<dyn CheckpointStore> = counting_leader.clone();
    let store_b: Arc<dyn CheckpointStore> = counting_follower.clone();

    // b.b2j: establish `le:counter:object_id` before allocating — the
    // allocator fails closed on an absent counter.
    counters
        .initialize_object_id_counter_if_absent()
        .await
        .unwrap();
    for _ in 0..5_u64 {
        counters.next_object_id().await.unwrap().unwrap();
    }

    const REPLICA_LEADER: &str = "replica-leader";
    const REPLICA_FOLLOWER: &str = "replica-follower";
    let config_a = loop_config(REPLICA_LEADER);
    let config_b = loop_config(REPLICA_FOLLOWER);
    // What each replica writes into the lock: `loop_config` builds its
    // leader with `test_leader`, so these are the two possible lock values.
    let lock_a = crate::common::test_leader(REPLICA_LEADER).lock_value();
    let lock_b = crate::common::test_leader(REPLICA_FOLLOWER).lock_value();

    let cancel = CancellationToken::new();
    let handle_a = spawn_checkpoint_loop(
        tv.client().clone(),
        keys.clone(),
        store_a,
        config_a,
        None,
        cancel.clone(),
    );
    let handle_b = spawn_checkpoint_loop(
        tv.client().clone(),
        keys.clone(),
        store_b,
        config_b,
        None,
        cancel.clone(),
    );

    // Wait for the leader lock to be acquired by one of the two
    // configured replicas; up to 5 s budget. Same primitive as the
    // sibling leader-race test.
    let lock_key = keys.checkpoint_leader_lock();
    let pool = tv.client().pool.clone();
    let lock_key_for_poll = lock_key.clone();
    let lock_acquired = poll_until(Duration::from_secs(5), || {
        let pool = pool.clone();
        let key = lock_key_for_poll.clone();
        let (lock_a, lock_b) = (lock_a.clone(), lock_b.clone());
        async move {
            let value: Option<String> = pool.get(&key).await.unwrap_or(None);
            value.is_some_and(|v| v == lock_a || v == lock_b)
        }
    })
    .await;
    assert!(
        lock_acquired,
        "le:checkpoint:leader must be acquired by exactly one of the two \
         replicas within 5 s",
    );

    let leader_lock_value: String = pool
        .get::<Option<String>, _>(&lock_key)
        .await
        .expect("GET checkpoint leader lock must succeed")
        .expect("leader lock value must be present after the bounded-poll wait");
    assert!(
        leader_lock_value == lock_a || leader_lock_value == lock_b,
        "leader lock value must be exactly one of the two configured \
         replicas' lock values ({lock_a:?} / {lock_b:?}), got {leader_lock_value:?}",
    );

    // Wait for either wrapper to record a write. The leader's counter
    // is non-zero at spawn time so the first counters-sync tick must
    // fire `put_counters_sidecar`. 5 s budget gives plenty of room.
    let leader_for_poll = Arc::clone(&counting_leader);
    let follower_for_poll = Arc::clone(&counting_follower);
    let saw_any = poll_until(Duration::from_secs(5), || {
        let l = Arc::clone(&leader_for_poll);
        let f = Arc::clone(&follower_for_poll);
        async move {
            !l.put_counters_calls_snapshot().await.is_empty()
                || !f.put_counters_calls_snapshot().await.is_empty()
        }
    })
    .await;
    assert!(
        saw_any,
        "at least one of the two replicas must record a counters-sidecar \
         write within 5 s",
    );

    // Soak: ≥ 3 × counters_sync_interval (200 ms × 3 = 600 ms) of
    // real time. We use 1.5 s for CI headroom, well above the floor.
    // The leader lock's 30 s TTL keeps the same replica as leader
    // throughout, so a regression that bypassed the lock would
    // surface writes from the follower in this window.
    tokio::time::sleep(Duration::from_millis(1500)).await;

    let leader_calls = if leader_lock_value == lock_a {
        counting_leader.put_counters_calls_snapshot().await
    } else {
        counting_follower.put_counters_calls_snapshot().await
    };
    let follower_calls = if leader_lock_value == lock_a {
        counting_follower.put_counters_calls_snapshot().await
    } else {
        counting_leader.put_counters_calls_snapshot().await
    };

    assert!(
        !leader_calls.is_empty(),
        "the lock-winning replica ({leader_lock_value}) MUST have recorded ≥ 1 \
         counters-sidecar write during the soak; recorded zero",
    );
    assert!(
        follower_calls.is_empty(),
        "the non-leader replica MUST NOT have recorded any \
         counters-sidecar writes; saw {} (lock holder is {})",
        follower_calls.len(),
        leader_lock_value,
    );

    // Sanity: the leader's first recorded body decodes and reflects
    // the counter at write time (n=5 — driven pre-spawn).
    let _ = decode_and_check_sidecar(&leader_calls[0].body, 5);

    // ── Teardown ────────────────────────────────────────────────────
    cancel.cancel();
    for h in [handle_a, handle_b] {
        let join_outcome = tokio::time::timeout(Duration::from_secs(5), h).await;
        assert!(
            join_outcome.is_ok(),
            "checkpoint loop join must complete within 5 s of cancellation",
        );
        join_outcome
            .expect("timeout already checked")
            .expect("checkpoint loop task must not panic on shutdown");
    }
}

// ════════════════════════════════════════════════════════════════════════
// Scenario 6 — a fresh leader whose live counter is below the stored
// sidecar never lowers it.
//
// Stand a sidecar of 50 up in the store and set the live counter to 7,
// then start one loop: its first counters tick is this process's first,
// so nothing in-process knows what was stored. Over three ticks no PUT
// is issued, each tick counts `skipped`, and 50 stays stored. Raising
// the counter to 60 then writes 60: the stored value is a floor, not a
// latch.
//
// "Higher counter still writes" with NO stored sidecar is scenarios 1
// and 3; "no PUT when the counter equals the stored value" is
// scenario 2.
// ════════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn counters_sync_fresh_leader_with_lower_counter_never_lowers_sidecar() {
    const STORED_FLOOR: u64 = 50;
    const LOWER_COUNTER: u64 = 7;
    const HIGHER_COUNTER: u64 = 60;

    let tv = TestValkey::setup_with_pool_size(8).await;
    let keys = tv.key_schema().clone();

    let inner = Arc::new(InMemoryCheckpointStore::new());
    inner
        .put_counters_sidecar(sidecar_body(STORED_FLOOR))
        .await
        .expect("pre-seeding the stored sidecar must succeed");
    let counting = Arc::new(CountingCheckpointStore::new(Arc::clone(&inner)));
    let store: Arc<dyn CheckpointStore> = counting.clone();
    set_object_id_counter(&tv, LOWER_COUNTER).await;

    let metrics = make_metrics("counters-sync-fresh-leader-lower");
    let cancel = CancellationToken::new();
    let handle = spawn_checkpoint_loop(
        tv.client().clone(),
        keys.clone(),
        store,
        loop_config("counters-sync-fresh-leader-lower-replica"),
        Some(Arc::clone(&metrics)),
        cancel.clone(),
    );

    wait_for_counters_ticks(&metrics, 3, "fresh leader below the stored sidecar").await;

    let calls = counting.put_counters_calls_snapshot().await;
    assert!(
        calls.is_empty(),
        "a leader whose live counter ({LOWER_COUNTER}) is below the stored sidecar \
         ({STORED_FLOOR}) must not write; recorded PUTs of {:?}",
        recorded_maxes(&calls),
    );
    assert_eq!(
        stored_max(&inner).await,
        STORED_FLOOR,
        "the stored sidecar must still hold the floor",
    );
    assert_eq!(
        counters_writes(&metrics, "error"),
        0,
        "a counter below the stored sidecar is a skip, not an error",
    );

    set_object_id_counter(&tv, HIGHER_COUNTER).await;
    wait_for_counters_result(&metrics, "success", 1, "counter raised past the floor").await;
    assert_eq!(
        recorded_maxes(&counting.put_counters_calls_snapshot().await),
        vec![HIGHER_COUNTER],
        "once the counter passes the stored value, exactly one PUT of the new counter lands",
    );
    assert_eq!(stored_max(&inner).await, HIGHER_COUNTER);

    cancel_and_join(&cancel, handle).await;
}

// ════════════════════════════════════════════════════════════════════════
// Scenario 7 — leadership hand-off after a simulated counter rewind
// never lowers the sidecar.
//
// Two loops (replicas A and B) share one inner store, as two pods share
// one durable adapter. The lock is pinned to A, which writes 10. The
// counter is then rewound to 4 (a loss the service cannot cause itself,
// set directly) and the lock is handed to B, which has never led and so
// holds no memory of 10: three of B's counters ticks run and none
// writes. The lock then goes back to A: three more of A's ticks run and
// none writes. 10 stays stored throughout.
// ════════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn counters_sync_leadership_handoff_after_rewind_never_lowers_sidecar() {
    const HIGH: u64 = 10;
    const REWOUND: u64 = 4;
    const REPLICA_A: &str = "counters-handoff-replica-a";
    const REPLICA_B: &str = "counters-handoff-replica-b";

    let tv = TestValkey::setup_with_pool_size(8).await;
    let keys = tv.key_schema().clone();

    let inner_shared = Arc::new(InMemoryCheckpointStore::new());
    let counting_a = Arc::new(CountingCheckpointStore::new(Arc::clone(&inner_shared)));
    let counting_b = Arc::new(CountingCheckpointStore::new(Arc::clone(&inner_shared)));
    let store_a: Arc<dyn CheckpointStore> = counting_a.clone();
    let store_b: Arc<dyn CheckpointStore> = counting_b.clone();
    let metrics_a = make_metrics("counters-handoff-a");
    let metrics_b = make_metrics("counters-handoff-b");
    let lock_a = crate::common::test_leader(REPLICA_A).lock_value();
    let lock_b = crate::common::test_leader(REPLICA_B).lock_value();

    set_object_id_counter(&tv, HIGH).await;
    // Pinned before either loop starts, so A leads from its first probe
    // and B never does until the hand-off.
    pin_leader_lock(&tv, &lock_a).await;

    let cancel = CancellationToken::new();
    let handle_a = spawn_checkpoint_loop(
        tv.client().clone(),
        keys.clone(),
        store_a,
        loop_config(REPLICA_A),
        Some(Arc::clone(&metrics_a)),
        cancel.clone(),
    );
    let handle_b = spawn_checkpoint_loop(
        tv.client().clone(),
        keys.clone(),
        store_b,
        loop_config(REPLICA_B),
        Some(Arc::clone(&metrics_b)),
        cancel.clone(),
    );

    wait_for_counters_result(&metrics_a, "success", 1, "replica A writes the high value").await;
    assert_eq!(
        recorded_maxes(&counting_a.put_counters_calls_snapshot().await),
        vec![HIGH],
    );
    assert_eq!(stored_max(&inner_shared).await, HIGH);
    assert_eq!(
        counters_ticks(&metrics_b),
        0,
        "replica B must not have run a counters tick while A held the lock",
    );

    // Rewind, then hand the lock to B.
    set_object_id_counter(&tv, REWOUND).await;
    pin_leader_lock(&tv, &lock_b).await;
    wait_for_counters_ticks(&metrics_b, 3, "replica B leads after the hand-off").await;

    let b_calls = counting_b.put_counters_calls_snapshot().await;
    assert!(
        b_calls.is_empty(),
        "the new leader must not lower the stored sidecar ({HIGH}) to the rewound \
         counter ({REWOUND}); replica B recorded PUTs of {:?}",
        recorded_maxes(&b_calls),
    );
    assert_eq!(
        stored_max(&inner_shared).await,
        HIGH,
        "the stored sidecar must still hold {HIGH} after the hand-off",
    );

    // Hand the lock back to A.
    let a_ticks_before = counters_ticks(&metrics_a);
    pin_leader_lock(&tv, &lock_a).await;
    wait_for_counters_ticks(
        &metrics_a,
        a_ticks_before + 3,
        "replica A leads again after regaining the lock",
    )
    .await;

    assert_eq!(
        recorded_maxes(&counting_a.put_counters_calls_snapshot().await),
        vec![HIGH],
        "the old leader must not write below the stored sidecar after regaining the lock",
    );
    assert!(
        counting_b.put_counters_calls_snapshot().await.is_empty(),
        "replica B must still have recorded no PUT",
    );
    assert_eq!(stored_max(&inner_shared).await, HIGH);

    // ── Teardown ────────────────────────────────────────────────────
    cancel.cancel();
    for h in [handle_a, handle_b] {
        let join_outcome = tokio::time::timeout(Duration::from_secs(5), h).await;
        assert!(
            join_outcome.is_ok(),
            "checkpoint loop join must complete within 5 s of cancellation",
        );
        join_outcome
            .expect("timeout already checked")
            .expect("checkpoint loop task must not panic on shutdown");
    }
}

// ════════════════════════════════════════════════════════════════════════
// Scenario 8 — a lower body that lands late is raised back by the next
// counters tick.
//
// The loop writes 8. A body of 3 is then put straight into the inner
// store, standing in for a deposed leader's PUT that was already in
// flight and landed after the higher one; the counter stays at 8.
// Within two further counters ticks the stored value is 8 again. Two,
// because a tick already past its sidecar read when the late body landed
// may still be the first one counted; the second starts after it.
//
// The gate is the stored value read in each tick and nothing else: a
// gate that also required the counter to exceed what this process last
// wrote (8) would never repair it while the counter stays at 8.
// ════════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn counters_sync_late_lower_write_is_repaired_by_next_tick() {
    const HIGH: u64 = 8;
    const LATE_LOW: u64 = 3;

    let tv = TestValkey::setup_with_pool_size(8).await;
    let keys = tv.key_schema().clone();

    let inner = Arc::new(InMemoryCheckpointStore::new());
    let counting = Arc::new(CountingCheckpointStore::new(Arc::clone(&inner)));
    let store: Arc<dyn CheckpointStore> = counting.clone();
    set_object_id_counter(&tv, HIGH).await;

    let metrics = make_metrics("counters-sync-late-write");
    let cancel = CancellationToken::new();
    let handle = spawn_checkpoint_loop(
        tv.client().clone(),
        keys.clone(),
        store,
        loop_config("counters-sync-late-write-replica"),
        Some(Arc::clone(&metrics)),
        cancel.clone(),
    );

    wait_for_counters_result(&metrics, "success", 1, "the loop writes the high value").await;
    assert_eq!(stored_max(&inner).await, HIGH);

    // The late PUT lands, bypassing the loop and the counting wrapper.
    inner
        .put_counters_sidecar(sidecar_body(LATE_LOW))
        .await
        .expect("landing the late lower body must succeed");
    assert_eq!(stored_max(&inner).await, LATE_LOW);
    let ticks_after_late_write = counters_ticks(&metrics);

    wait_for_counters_ticks(
        &metrics,
        ticks_after_late_write + 2,
        "two counters ticks after the late write",
    )
    .await;

    assert_eq!(
        stored_max(&inner).await,
        HIGH,
        "within two counters ticks of a late lower write ({LATE_LOW}), the stored sidecar \
         must be raised back to the live counter ({HIGH})",
    );
    assert_eq!(
        recorded_maxes(&counting.put_counters_calls_snapshot().await),
        vec![HIGH, HIGH],
        "the repair is one more PUT of the unchanged counter",
    );

    cancel_and_join(&cancel, handle).await;
}

// ════════════════════════════════════════════════════════════════════════
// Scenario 9 — the stored sidecar is read before the live counter.
//
// The live counter starts at C0 = 5 with nothing stored. The wrapper's
// first `get_counters_sidecar` (see `FirstGetRaise`) raises the counter
// to C0 + 2 and stores C0 + 1 before it reads, as a peer leader's write
// and further allocations would between two reads. Read in the required
// order, the first tick sees C0 + 1 and then C0 + 2, and writes C0 + 2.
// Read the other way round, it would see C0 and then C0 + 1, take
// C0 < C0 + 1 for a counter that went backwards, and skip; the PUT
// would come only from a later tick. So the test pins the PUT of C0 + 2
// to the first tick (`gets_before == 1`).
// ════════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn counters_sync_reads_stored_sidecar_before_live_counter() {
    const C0: u64 = 5;

    let tv = TestValkey::setup_with_pool_size(8).await;
    let keys = tv.key_schema().clone();

    let inner = Arc::new(InMemoryCheckpointStore::new());
    let counting = Arc::new(
        CountingCheckpointStore::new(Arc::clone(&inner)).with_first_get_raise(FirstGetRaise {
            client: tv.client().clone(),
            counter_key: keys.counter_object_id(),
            raise_counter_to: C0 + 2,
            store_sidecar_at: C0 + 1,
            fired: AtomicBool::new(false),
        }),
    );
    let store: Arc<dyn CheckpointStore> = counting.clone();
    set_object_id_counter(&tv, C0).await;

    let metrics = make_metrics("counters-sync-read-order");
    let cancel = CancellationToken::new();
    let handle = spawn_checkpoint_loop(
        tv.client().clone(),
        keys.clone(),
        store,
        loop_config("counters-sync-read-order-replica"),
        Some(Arc::clone(&metrics)),
        cancel.clone(),
    );

    wait_for_counters_result(&metrics, "success", 1, "the read-order tick writes").await;

    let calls = counting.put_counters_calls_snapshot().await;
    let first = calls.first().expect("a success implies a recorded PUT");
    decode_and_check_sidecar(&first.body, C0 + 2);
    assert_eq!(
        first.gets_before, 1,
        "the PUT of C0 + 2 must come from the first counters tick — the one whose sidecar \
         read raised the counter; a later tick means the first read the counter before the \
         sidecar and took the raise for a counter below the stored value",
    );
    assert_eq!(stored_max(&inner).await, C0 + 2);

    cancel_and_join(&cancel, handle).await;
}

// ════════════════════════════════════════════════════════════════════════
// Scenario 10 — a failed sidecar read writes nothing; the next tick
// writes once the read succeeds.
//
// 20 is stored and the live counter is 25. A one-shot `Transient`
// injector fails the first `get_counters_sidecar`. That tick counts
// `error` and issues no PUT; the next tick reads 20 and writes 25. The
// one recorded PUT therefore follows the second sidecar read
// (`gets_before == 2`), and exactly one `error` is counted.
// ════════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn counters_sync_sidecar_read_failure_writes_nothing_until_read_succeeds() {
    const STORED: u64 = 20;
    const LIVE: u64 = 25;

    let tv = TestValkey::setup_with_pool_size(8).await;
    let keys = tv.key_schema().clone();

    let injector: Arc<dyn CheckpointFaultInjector> =
        Arc::new(CountersOneShotFailure::new("get_counters_sidecar"));
    let inner = Arc::new(InMemoryCheckpointStore::new().with_fault_injector(injector));
    // The injector targets reads only, so this pre-seed PUT passes.
    inner
        .put_counters_sidecar(sidecar_body(STORED))
        .await
        .expect("pre-seeding the stored sidecar must succeed");
    let counting = Arc::new(CountingCheckpointStore::new(Arc::clone(&inner)));
    let store: Arc<dyn CheckpointStore> = counting.clone();
    set_object_id_counter(&tv, LIVE).await;

    let metrics = make_metrics("counters-sync-read-failure");
    let cancel = CancellationToken::new();
    let handle = spawn_checkpoint_loop(
        tv.client().clone(),
        keys.clone(),
        store,
        loop_config("counters-sync-read-failure-replica"),
        Some(Arc::clone(&metrics)),
        cancel.clone(),
    );

    wait_for_counters_result(&metrics, "success", 1, "the tick after the failed read").await;

    let calls = counting.put_counters_calls_snapshot().await;
    assert_eq!(
        recorded_maxes(&calls),
        vec![LIVE],
        "exactly one PUT, of the live counter, must land",
    );
    assert_eq!(
        calls[0].gets_before, 2,
        "the PUT must come from the tick after the failed sidecar read; the failing tick \
         must write nothing",
    );
    assert_eq!(
        counters_writes(&metrics, "error"),
        1,
        "the failed sidecar read must be counted as one `error`",
    );
    assert_eq!(stored_max(&inner).await, LIVE);

    cancel_and_join(&cancel, handle).await;
}

// ════════════════════════════════════════════════════════════════════════
// Scenario 11 — stored bytes that do not parse write nothing until they
// parse.
//
// The store holds bytes that are not a counters sidecar and the live
// counter is 25. Over three ticks each counts `error`, no PUT is issued
// and the bytes stay as they were: with no floor to compare against, a
// write could lower the stored value. Once a well-formed sidecar of 20
// is stored, the next tick writes 25.
// ════════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn counters_sync_unparseable_sidecar_writes_nothing_until_it_parses() {
    const GARBAGE: &[u8] = b"not a counters sidecar";
    const REPAIRED: u64 = 20;
    const LIVE: u64 = 25;

    let tv = TestValkey::setup_with_pool_size(8).await;
    let keys = tv.key_schema().clone();

    let inner = Arc::new(InMemoryCheckpointStore::new());
    inner
        .put_counters_sidecar(Bytes::from_static(GARBAGE))
        .await
        .expect("storing the unparseable bytes must succeed");
    let counting = Arc::new(CountingCheckpointStore::new(Arc::clone(&inner)));
    let store: Arc<dyn CheckpointStore> = counting.clone();
    set_object_id_counter(&tv, LIVE).await;

    let metrics = make_metrics("counters-sync-parse-failure");
    let cancel = CancellationToken::new();
    let handle = spawn_checkpoint_loop(
        tv.client().clone(),
        keys.clone(),
        store,
        loop_config("counters-sync-parse-failure-replica"),
        Some(Arc::clone(&metrics)),
        cancel.clone(),
    );

    wait_for_counters_ticks(&metrics, 3, "ticks over the unparseable sidecar").await;

    let calls = counting.put_counters_calls_snapshot().await;
    assert!(
        calls.is_empty(),
        "no PUT may be issued while the stored sidecar does not parse; recorded PUTs of {:?}",
        recorded_maxes(&calls),
    );
    assert!(
        counters_writes(&metrics, "error") >= 3,
        "every tick over the unparseable sidecar must count `error`; saw {}",
        counters_writes(&metrics, "error"),
    );
    let still_stored = inner
        .get_counters_sidecar()
        .await
        .expect("inner get_counters_sidecar must succeed")
        .expect("the unparseable bytes must still be stored");
    assert_eq!(
        still_stored.as_ref(),
        GARBAGE,
        "the unparseable stored bytes must be left as they were",
    );

    inner
        .put_counters_sidecar(sidecar_body(REPAIRED))
        .await
        .expect("storing a well-formed sidecar must succeed");
    wait_for_counters_result(&metrics, "success", 1, "the tick after the bytes parse").await;

    assert_eq!(
        recorded_maxes(&counting.put_counters_calls_snapshot().await),
        vec![LIVE],
        "once the stored sidecar parses, the next tick writes the live counter",
    );
    assert_eq!(stored_max(&inner).await, LIVE);

    cancel_and_join(&cancel, handle).await;
}

// ════════════════════════════════════════════════════════════════════════
// Scenario 12 — an "absent" answer after the tick read a stored sidecar
// writes nothing.
//
// 50 is stored and the live counter is rewound to 7. The first counters
// tick reads 50 and skips; from the second on, the wrapper answers
// "absent" while 50 stays stored, as a storage-api pod's cached
// not-found would. Over three such ticks each counts `error` and none
// writes, so 50 stays stored. Taken as "no floor", the first of them
// would write 7.
// ════════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn counters_sync_absent_answer_after_reading_the_sidecar_never_lowers_it() {
    const STORED: u64 = 50;
    const REWOUND: u64 = 7;

    let tv = TestValkey::setup_with_pool_size(8).await;
    let keys = tv.key_schema().clone();

    let inner = Arc::new(InMemoryCheckpointStore::new());
    inner
        .put_counters_sidecar(sidecar_body(STORED))
        .await
        .expect("pre-seeding the stored sidecar must succeed");
    let counting = Arc::new(
        CountingCheckpointStore::new(Arc::clone(&inner))
            .with_stale_absent(StaleAbsent::AfterGets(1)),
    );
    let store: Arc<dyn CheckpointStore> = counting.clone();
    set_object_id_counter(&tv, REWOUND).await;

    let metrics = make_metrics("counters-sync-absent-after-read");
    let cancel = CancellationToken::new();
    let handle = spawn_checkpoint_loop(
        tv.client().clone(),
        keys.clone(),
        store,
        loop_config("counters-sync-absent-after-read-replica"),
        Some(Arc::clone(&metrics)),
        cancel.clone(),
    );

    // One tick reads the stored sidecar; three more are answered "absent".
    wait_for_counters_ticks(&metrics, 4, "ticks after the sidecar was read").await;

    let calls = counting.put_counters_calls_snapshot().await;
    assert!(
        calls.is_empty(),
        "an \"absent\" answer after this process read the sidecar ({STORED}) is not \"no \
         floor\": the rewound counter ({REWOUND}) must not be written; recorded PUTs of {:?}",
        recorded_maxes(&calls),
    );
    assert_eq!(
        counters_writes(&metrics, "skipped"),
        1,
        "only the first tick read the stored sidecar, and skipped below it",
    );
    assert!(
        counters_writes(&metrics, "error") >= 3,
        "every tick answered \"absent\" must count `error`; saw {}",
        counters_writes(&metrics, "error"),
    );
    assert_eq!(
        stored_max(&inner).await,
        STORED,
        "the stored sidecar must still hold {STORED}",
    );

    cancel_and_join(&cancel, handle).await;
}

// ════════════════════════════════════════════════════════════════════════
// Scenario 13 — an "absent" answer after this process wrote the sidecar
// writes nothing.
//
// Nothing is stored and the live counter is 10. The first counters tick
// reads "absent" before this process has seen a sidecar, so there is no
// floor, and it writes 10. From then on the wrapper answers "absent"
// while 10 stays stored, so no tick ever reads the sidecar back: only
// the process's own write can have marked the sidecar seen. The counter
// is then rewound to 4. Over three more ticks each counts `error` and
// none writes, so 10 stays stored. Taken as "no floor", each would write.
// ════════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn counters_sync_absent_answer_after_writing_the_sidecar_never_lowers_it() {
    const WRITTEN: u64 = 10;
    const REWOUND: u64 = 4;

    let tv = TestValkey::setup_with_pool_size(8).await;
    let keys = tv.key_schema().clone();

    let inner = Arc::new(InMemoryCheckpointStore::new());
    let counting = Arc::new(
        CountingCheckpointStore::new(Arc::clone(&inner))
            .with_stale_absent(StaleAbsent::AfterSuccessfulPut),
    );
    let store: Arc<dyn CheckpointStore> = counting.clone();
    set_object_id_counter(&tv, WRITTEN).await;

    let metrics = make_metrics("counters-sync-absent-after-write");
    let cancel = CancellationToken::new();
    let handle = spawn_checkpoint_loop(
        tv.client().clone(),
        keys.clone(),
        store,
        loop_config("counters-sync-absent-after-write-replica"),
        Some(Arc::clone(&metrics)),
        cancel.clone(),
    );

    wait_for_counters_result(&metrics, "success", 1, "the first tick writes").await;
    let calls = counting.put_counters_calls_snapshot().await;
    assert_eq!(
        recorded_maxes(&calls),
        vec![WRITTEN],
        "an \"absent\" answer before this process has seen a sidecar means no floor: the \
         first tick writes the live counter",
    );
    assert_eq!(calls[0].gets_before, 1, "the write is the first tick's");

    set_object_id_counter(&tv, REWOUND).await;
    let ticks_at_rewind = counters_ticks(&metrics);
    wait_for_counters_ticks(
        &metrics,
        ticks_at_rewind + 3,
        "ticks after the counter was rewound",
    )
    .await;

    let calls = counting.put_counters_calls_snapshot().await;
    assert_eq!(
        recorded_maxes(&calls),
        vec![WRITTEN],
        "an \"absent\" answer after this process wrote the sidecar ({WRITTEN}) is not \"no \
         floor\": nothing more may be written, the rewound counter ({REWOUND}) least of all",
    );
    assert_eq!(
        counters_writes(&metrics, "skipped"),
        0,
        "no tick read the sidecar back, so only the process's own write marked it seen",
    );
    assert!(
        counters_writes(&metrics, "error") >= 3,
        "every tick answered \"absent\" after the write must count `error`; saw {}",
        counters_writes(&metrics, "error"),
    );
    assert_eq!(
        stored_max(&inner).await,
        WRITTEN,
        "the stored sidecar must still hold {WRITTEN}",
    );

    cancel_and_join(&cancel, handle).await;
}
