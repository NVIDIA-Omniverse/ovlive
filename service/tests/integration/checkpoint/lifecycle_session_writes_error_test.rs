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

//! Subtask `t3.fgj.7v.p9.am` — error-path coverage for the
//! `SessionCheckpointWriter` dispatch chain (Task `t2.fgj.7v.p9`).
//!
//! These tests pin two load-bearing invariants of the session-checkpoint
//! error-handling contract:
//!
//! 1. **Failure-is-swallowed.** When `put_session_checkpoint` returns
//!    `Err(_)`, the lifecycle RPC has already returned `Ok(_)` to the
//!    client — so `SessionStore::create` MUST surface `Ok(session_id)`
//!    even though the off-hot-path checkpoint dispatch fails.
//! 2. **Uniform metric scheme (no Transient/Permanent split).** Both
//!    `CheckpointError::Transient` and `CheckpointError::Permanent` fold
//!    into the same `live_edit_checkpoint_writes_total{kind="session",
//!    result="error"}` cell. The metric does NOT split by error class —
//!    triage between transient and permanent is via the WARN log content,
//!    not the metric label. Each test asserts BOTH the increment AND the
//!    no-split contract (no `result="error_transient"` or `result=
//!    "error_permanent"` cells exist on the registry).
//! 3. **The b.443 ADAPTER-phase retry ladder, and its metric unit.** The
//!    dispatcher retries a `Transient` adapter failure up to
//!    `MAX_ADAPTER_ATTEMPTS` times; `Permanent` / `NotFound` / `Conflict` /
//!    `NotImplemented` reproduce identically and are abandoned after one
//!    attempt. In this phase the `result="error"` counter's unit is the
//!    *adapter call*, not the coalesced operation, so a fully-failing
//!    transient operation ticks it once per attempt. The asymmetry between
//!    the two failure modes below (3 ticks vs 1) IS the
//!    retry-classification contract: a regression that started retrying
//!    `Permanent` errors, or that stopped retrying `Transient` ones, moves
//!    one of the two numbers.
//!
//!    **Scope note.** Every failure here is an ADAPTER-phase failure —
//!    nothing has landed, so retrying the whole operation is free and each
//!    attempt is genuinely its own adapter call. The PUBLISH phase counts
//!    differently on purpose: once the adapter call has landed the operation
//!    is never re-run, intermediate publish failures tick nothing, and a
//!    publish that exhausts its budget ticks `error` exactly ONCE for the
//!    whole operation. Those cells are pinned in
//!    `lifecycle_session_publish_watermark_test.rs::publish_faults`; do not
//!    generalise the per-attempt unit asserted below to that phase.
//!
//! Sandbox env note: these tests use `TestValkey::setup`, which reads
//! `TEST_VALKEY_URL`. Under the project sandbox the URL must include the
//! password and `host.docker.internal` host; see CLAUDE.md.

#![cfg(feature = "test-checkpoint-fault-injection")]

use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use tokio_util::task::TaskTracker;

use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
use live_edit::checkpoint::{
    CheckpointBlob, CheckpointError, CheckpointHandle, CheckpointMeta, CheckpointStore,
    SessionCheckpointWriter,
};
use live_edit::observability::metrics::SharedMetrics;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;

/// Service label stamped on the per-test `Metrics` registry. Stable across
/// both test cases so the metric-cell assertions read the same labels.
const SERVICE_LABEL: &str = "test";

/// Mirror of `MAX_ADAPTER_ATTEMPTS` in `src/checkpoint/session_writer.rs`
/// (b.443) — the number of adapter calls one coalesced operation makes
/// before the dispatch task abandons it.
///
/// Deliberately mirrored rather than imported: the production constant is a
/// private module item, and exporting it purely for a test would widen the
/// dispatcher's public surface. The pin is one-directional but loud — if the
/// production ladder changes without this constant following, the
/// `failed_put_transient_increments_error_metric` assertion below fails with
/// the observed-vs-expected tick count in the message, which names exactly
/// what to reconcile.
const MAX_ADAPTER_ATTEMPTS: u64 = 3;

/// Failure mode the wrapper synthesises on every `put_session_checkpoint`.
/// Both classifications fold into the same `result="error"` cell per the
/// uniform-metric contract documented at the top of this file.
#[derive(Debug, Clone, Copy)]
enum FailureMode {
    Transient,
    Permanent,
}

/// `CheckpointStore` wrapper that delegates every method to an inner store
/// EXCEPT `put_session_checkpoint`, which synthesises a classified error
/// per the configured [`FailureMode`].
///
/// We hand-roll this instead of going through the
/// `CheckpointFaultInjector` seam on the underlying `InMemoryCheckpointStore`
/// because the injector trait fires `maybe_inject` on every method (layer
/// puts, session puts, ext-blob puts, ...) and an op-label-keyed filter
/// would couple this test to the stable string the in-memory adapter uses
/// at its `maybe_inject` call site (`"put_session_checkpoint"`). A
/// dedicated wrapper keeps the test self-contained: only the one method
/// fails, every other call passes through to the inner store cleanly.
///
/// `fail_budget` is the number of leading `put_session_checkpoint` calls that
/// fail; every call after that delegates to the inner store and succeeds.
/// `usize::MAX` is the always-fail mode the two original tests use;
/// a finite budget drives the b.443 retry-recovers case.
struct FailingSessionPutStore {
    inner: Arc<InMemoryCheckpointStore>,
    mode: FailureMode,
    fail_budget: AtomicUsize,
    put_calls: AtomicUsize,
    /// `session_id` of the most recent `put_session_checkpoint` call, so a
    /// test can read the deposited blob back without re-plumbing the id out
    /// of the driver.
    last_put_session_id: Mutex<Option<String>>,
}

impl FailingSessionPutStore {
    /// Fail every `put_session_checkpoint`, forever.
    fn always_failing(mode: FailureMode) -> Self {
        Self::with_fail_budget(mode, usize::MAX)
    }

    /// Fail the first `fail_budget` `put_session_checkpoint` calls, then
    /// delegate to the inner in-memory store.
    fn with_fail_budget(mode: FailureMode, fail_budget: usize) -> Self {
        Self {
            inner: Arc::new(InMemoryCheckpointStore::new()),
            mode,
            fail_budget: AtomicUsize::new(fail_budget),
            put_calls: AtomicUsize::new(0),
            last_put_session_id: Mutex::new(None),
        }
    }

    fn put_calls(&self) -> usize {
        self.put_calls.load(Ordering::Acquire)
    }

    fn last_put_session_id(&self) -> Option<String> {
        self.last_put_session_id
            .lock()
            .expect("last_put_session_id mutex poisoned")
            .clone()
    }

    fn synthesise_error(&self) -> CheckpointError {
        match self.mode {
            FailureMode::Transient => CheckpointError::Transient {
                source: "synthetic transient failure for p9.am test".into(),
            },
            FailureMode::Permanent => CheckpointError::Permanent {
                source: "synthetic permanent failure for p9.am test".into(),
            },
        }
    }
}

#[async_trait]
impl CheckpointStore for FailingSessionPutStore {
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

    // Forwarded per `CheckpointStore::get_layer_checkpoint`'s MUST — its trait
    // default returns `NotImplemented`, which would mask a generation the
    // inner store holds rather than merely leaving the call un-instrumented.
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
        // This is the load-bearing failure point — the writer's dispatch
        // chain reaches this method only when `build_session_checkpoint`
        // returned `Ok(Some(_))` and `to_bytes()` succeeded. The classified
        // error returned here MUST tick `result="error"` (NOT `result=
        // "error_transient"` / `result="error_permanent"`) per the uniform-
        // metric contract.
        self.put_calls.fetch_add(1, Ordering::AcqRel);
        *self
            .last_put_session_id
            .lock()
            .expect("last_put_session_id mutex poisoned") = Some(session_id.to_string());
        // Consume one unit of the failure budget. `fetch_update` keeps the
        // saturating floor at zero without a read-modify-write race, so a
        // finite budget fails EXACTLY that many calls even if the dispatcher
        // ever parallelised its retries.
        let consumed = self
            .fail_budget
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |remaining| {
                (remaining > 0).then_some(remaining.saturating_sub(1))
            })
            .is_ok();
        if consumed {
            return Err(self.synthesise_error());
        }
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

    async fn list_referenced_ext_hashes(&self) -> Result<HashSet<String>, CheckpointError> {
        self.inner.list_referenced_ext_hashes().await
    }

    async fn list_live_ext_set(
        &self,
    ) -> Result<live_edit::checkpoint::LiveExtSet, CheckpointError> {
        self.inner.list_live_ext_set().await
    }
}

/// Read every `(result, kind)` pair currently materialised on the
/// `live_edit_checkpoint_writes_total` metric family.
///
/// Used by the no-split assertion below: a metric pre-init shape change
/// that adds (say) `result="error_transient"` would surface as a new pair
/// in the returned set, failing the assertion at the closest possible
/// call site rather than at a distant operator dashboard.
fn collect_checkpoint_writes_label_pairs(metrics: &SharedMetrics) -> HashSet<(String, String)> {
    let text = live_edit::observability::metrics::test_util::render(metrics);
    let mut set = HashSet::new();
    for line in text.lines() {
        let Some(rest) = line.strip_prefix("live_edit_checkpoint_writes_total") else {
            continue;
        };
        if !rest.starts_with('{') {
            continue;
        }
        let extract = |label: &str| -> String {
            let needle = format!("{label}=\"");
            if let Some(i) = line.find(&needle) {
                let after = &line[i + needle.len()..];
                if let Some(end) = after.find('"') {
                    return after[..end].to_owned();
                }
            }
            String::new()
        };
        set.insert((extract("result"), extract("kind")));
    }
    set
}

/// Drive `SessionStore::create` against a writer wired to `failing_store`,
/// await the in-flight `TaskTracker`, and assert the swallowed-error contract
/// on the RPC's own return value.
///
/// Hands back the metric registry, the materialised
/// `checkpoint_writes_total` label pairs, and the store itself (so a caller
/// can assert on the adapter-call count) — each test's variant-specific
/// assertions stay close to its own body.
async fn drive_put_and_collect(
    failing_store: FailingSessionPutStore,
) -> (
    SharedMetrics,
    HashSet<(String, String)>,
    Arc<FailingSessionPutStore>,
) {
    let tv = TestValkey::setup().await;

    let metrics: SharedMetrics = Arc::new(crate::common::metrics_scrape_only(SERVICE_LABEL));

    let failing_store = Arc::new(failing_store);
    let store: Arc<dyn CheckpointStore> = Arc::clone(&failing_store) as Arc<dyn CheckpointStore>;

    let tracker = TaskTracker::new();
    let writer = Arc::new(SessionCheckpointWriter::new(
        Arc::clone(&store),
        tv.client().clone(),
        tv.key_schema().clone(),
        tracker.clone(),
        Arc::clone(&metrics),
        SERVICE_LABEL.to_string(),
        true,
    ));

    // Wire the writer into the SessionStore so `create`'s success arm
    // enqueues a fire-and-forget checkpoint write through the failing
    // adapter. Using the canonical `with_checkpoint_writer` builder
    // mirrors `main.rs`'s wire-up exactly — a downstream rename of the
    // builder name surfaces here as a compile error rather than a
    // silent drift.
    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .with_checkpoint_writer(Arc::clone(&writer));

    // A fixed session name is safe across the (now three) callers of this
    // driver: each gets its own `TestValkey`, whose `test_{uuid}:` key prefix
    // gives it a private keyspace including the path-name uniqueness index.
    let session_id = session_store
        .create(
            "p9-am-failing-put-session",
            "/p9-am/failing-put",
            "test-user",
        )
        .await
        .expect("SessionStore::create MUST return Ok(_) even when the off-hot-path checkpoint dispatch fails: the failure is swallowed per the SessionCheckpointWriter contract");
    assert!(
        !session_id.is_empty(),
        "session_id MUST be a non-empty UUID4 — empty value indicates an unexpected early-return path"
    );

    // Close + drain the tracker so the spawned task has reached its
    // post-`put_session_checkpoint` arm and ticked the metric before we
    // read it. `wait()` returns once every spawned task has completed —
    // since each spawned task either ticks the metric or short-circuits
    // silently, no further synchronization is needed. Wrap in a
    // `tokio::time::timeout` so a hang in the writer (which would be a
    // real bug, not a test-only flake) fails fast with a clear message
    // instead of stalling the suite.
    tracker.close();
    tokio::time::timeout(Duration::from_secs(10), tracker.wait())
        .await
        .expect(
            "tracker.wait() must drain inside 10s — a hang here points at a SessionCheckpointWriter \
             bug, not a test-only flake",
        );

    let label_pairs = collect_checkpoint_writes_label_pairs(&metrics);
    (metrics, label_pairs, failing_store)
}

/// Read one `live_edit_checkpoint_writes_total{result,kind="session"}` cell.
fn session_writes_cell(metrics: &SharedMetrics, result: &str) -> u64 {
    live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(metrics),
        "live_edit_checkpoint_writes_total",
        &[("result", result), ("kind", "session")],
    )
    .unwrap_or(0.0) as u64
}

/// **Transient-failure case.** `put_session_checkpoint` returns
/// `Err(CheckpointError::Transient { .. })` on every attempt. The writer's
/// `handle_put_outcome` MUST fold each failed attempt into the generic
/// `result="error"` cell (NOT a `result="error_transient"` cell) on
/// `live_edit_checkpoint_writes_total{kind="session"}`. The lifecycle
/// RPC has already returned success — `SessionStore::create` therefore
/// surfaces `Ok(session_id)`.
///
/// **The tick count is `MAX_ADAPTER_ATTEMPTS`, not 1 (b.443).** Before
/// b.443 the dispatcher made exactly one adapter call per lifecycle event,
/// so a single transient Valkey/adapter blip silently dropped the write
/// entirely. It now retries a `Transient` classification up to
/// `MAX_ADAPTER_ATTEMPTS` times with a 50 ms → 200 ms backoff, and the
/// counter's unit is the adapter call, so a fully-failing operation ticks
/// once per attempt. Asserting the exact count (rather than `>= 1`) pins
/// both halves: the ladder ran to its end, and it stopped there rather than
/// spinning inside the shutdown drain budget.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn failed_put_transient_increments_error_metric() {
    let (metrics, label_pairs, store) = drive_put_and_collect(
        FailingSessionPutStore::always_failing(FailureMode::Transient),
    )
    .await;

    let error_cell = session_writes_cell(&metrics, "error");
    assert_eq!(
        error_cell, MAX_ADAPTER_ATTEMPTS,
        "live_edit_checkpoint_writes_total{{kind=\"session\", result=\"error\"}} MUST tick once \
         per failed adapter ATTEMPT — {MAX_ADAPTER_ATTEMPTS} for an always-Transient \
         put_session_checkpoint (b.443 bounded retry ladder); got {error_cell}. A value of 1 \
         means the retry ladder was removed (a single blip once again drops the write silently); \
         a larger value means the attempt cap moved"
    );

    let put_calls = store.put_calls() as u64;
    assert_eq!(
        put_calls, MAX_ADAPTER_ATTEMPTS,
        "put_session_checkpoint MUST be called exactly {MAX_ADAPTER_ATTEMPTS} times — the metric \
         and the adapter-call count are the same unit, so a divergence here means a tick was \
         emitted without an adapter call (or vice versa); got {put_calls}"
    );

    let success_cell = session_writes_cell(&metrics, "success");
    assert_eq!(
        success_cell, 0,
        "live_edit_checkpoint_writes_total{{kind=\"session\", result=\"success\"}} \
         MUST NOT tick when every attempt failed; got {success_cell}"
    );

    // No-split contract: a `result="error_transient"` cell MUST NOT exist
    // on the registry. If a future change starts splitting Transient and
    // Permanent into separate cells, the increment + success assertions
    // above would still pass (the new cell would tick instead of the
    // generic `error` cell) but for the wrong reason — this assertion is
    // the contract pin that catches the split at its closest possible
    // call site.
    assert!(
        !label_pairs.contains(&("error_transient".to_owned(), "session".to_owned())),
        "no-split contract: `result=\"error_transient\"` cell must NOT exist on \
         live_edit_checkpoint_writes_total — Transient errors MUST fold into the generic \
         `result=\"error\"` cell; got cells {label_pairs:?}"
    );
    assert!(
        !label_pairs.contains(&("error_permanent".to_owned(), "session".to_owned())),
        "no-split contract: `result=\"error_permanent\"` cell must NOT exist on \
         live_edit_checkpoint_writes_total — Permanent errors MUST fold into the generic \
         `result=\"error\"` cell; got cells {label_pairs:?}"
    );
}

/// **Permanent-failure case.** Symmetric with the transient case above:
/// `put_session_checkpoint` returns `Err(CheckpointError::Permanent {
/// .. })`, the writer folds it into the same `result="error"` cell, and
/// the no-split assertions hold identically. Running both classes
/// through the same metric-cell assertion is the load-bearing proof of
/// the uniform-metric contract — if a future change started splitting
/// the two classes, ONE of these tests would still pass (against the
/// new class-specific cell) while this one's `error` assertion would
/// fail, surfacing the split immediately.
///
/// **The tick count stays 1 (b.443).** `Permanent` is not retried — it
/// reproduces identically on every attempt, so the ladder abandons after the
/// first call. That asymmetry against
/// `failed_put_transient_increments_error_metric`'s
/// `MAX_ADAPTER_ATTEMPTS` ticks is the retry-classification contract; a
/// regression that retried everything uniformly moves this number to 3.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn failed_put_permanent_increments_error_metric() {
    let (metrics, label_pairs, store) = drive_put_and_collect(
        FailingSessionPutStore::always_failing(FailureMode::Permanent),
    )
    .await;

    let error_cell = session_writes_cell(&metrics, "error");
    assert_eq!(
        error_cell, 1,
        "live_edit_checkpoint_writes_total{{kind=\"session\", result=\"error\"}} \
         MUST increment by exactly 1 on a Permanent put_session_checkpoint failure — Permanent \
         is NOT retried (it reproduces identically), unlike Transient which ticks \
         {MAX_ADAPTER_ATTEMPTS} times; got {error_cell}"
    );

    let put_calls = store.put_calls();
    assert_eq!(
        put_calls, 1,
        "put_session_checkpoint MUST be called exactly once for a Permanent failure — retrying \
         a Permanent classification burns the shutdown drain budget for a call that cannot \
         succeed; got {put_calls}"
    );

    let success_cell = session_writes_cell(&metrics, "success");
    assert_eq!(
        success_cell, 0,
        "live_edit_checkpoint_writes_total{{kind=\"session\", result=\"success\"}} \
         MUST NOT tick when the put failed; got {success_cell}"
    );

    assert!(
        !label_pairs.contains(&("error_transient".to_owned(), "session".to_owned())),
        "no-split contract: `result=\"error_transient\"` cell must NOT exist on \
         live_edit_checkpoint_writes_total — Transient errors MUST fold into the generic \
         `result=\"error\"` cell; got cells {label_pairs:?}"
    );
    assert!(
        !label_pairs.contains(&("error_permanent".to_owned(), "session".to_owned())),
        "no-split contract: `result=\"error_permanent\"` cell must NOT exist on \
         live_edit_checkpoint_writes_total — Permanent errors MUST fold into the generic \
         `result=\"error\"` cell; got cells {label_pairs:?}"
    );
}

/// **The retry actually recovers (b.443).** The two tests above pin what
/// happens when every attempt fails; neither shows that the ladder is worth
/// having. This one does: `put_session_checkpoint` fails once with a
/// `Transient` classification and succeeds on attempt 2, and the checkpoint
/// lands.
///
/// Without the retry the session checkpoint would simply be missing until
/// some later lifecycle event on the same session happened to rewrite it —
/// and the session checkpoint is the root of reachability for DR, so
/// "missing" means every layer created in that session is unrestorable in
/// the interim. That is the durability gap the ladder closes, and this test
/// is the one that fails if the ladder is reduced back to a single attempt.
///
/// The metric shape is the mixed one the counter's per-adapter-call unit
/// implies: one `error` tick for the failed attempt, one `success` tick for
/// the attempt that landed. Note that the `success` tick sits BEHIND the
/// publish — an operation whose bytes landed but whose ordering was never
/// established does not report success — so this test also exercises the
/// ordinary accepted-publish path end to end.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn transient_put_failure_recovers_on_retry() {
    let (metrics, _label_pairs, store) = drive_put_and_collect(
        FailingSessionPutStore::with_fail_budget(FailureMode::Transient, 1),
    )
    .await;

    let put_calls = store.put_calls();
    assert_eq!(
        put_calls, 2,
        "put_session_checkpoint MUST be called twice — attempt 1 fails Transient, attempt 2 \
         succeeds. A count of 1 means the retry ladder is gone and a single transient blip once \
         again drops the session checkpoint silently; got {put_calls}"
    );

    let success_cell = session_writes_cell(&metrics, "success");
    assert_eq!(
        success_cell, 1,
        "live_edit_checkpoint_writes_total{{kind=\"session\", result=\"success\"}} MUST tick \
         exactly once — the retried attempt landed, so the operation as a whole succeeded; \
         got {success_cell}"
    );

    let error_cell = session_writes_cell(&metrics, "error");
    assert_eq!(
        error_cell, 1,
        "live_edit_checkpoint_writes_total{{kind=\"session\", result=\"error\"}} MUST tick \
         exactly once — the counter's unit is the adapter call, and exactly one call failed. \
         A zero here would hide real backend flakiness from operators behind a successful \
         retry; got {error_cell}"
    );

    // The blob is actually durable, not merely reported as written. A
    // success tick with no blob behind it would mean the retry re-ran the
    // metric arm without re-running the adapter call.
    let session_id = store
        .last_put_session_id()
        .expect("put_session_checkpoint must have been reached at least once");
    let blob = store
        .inner
        .get_session_checkpoint(&session_id)
        .await
        .expect("get_session_checkpoint must not fail against the in-memory store");
    assert!(
        blob.is_some(),
        "the retried put MUST have deposited a session checkpoint in the adapter for \
         {session_id}"
    );
}
