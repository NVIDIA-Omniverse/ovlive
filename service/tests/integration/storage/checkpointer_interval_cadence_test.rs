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

// b.yut — the checkpoint tick gate's two arms must reference two different
// clocks, and the commit clock must be read in the unit production writes it.
//
// Pre-fix, the loop derived "time since last commit" from the leading half of
// the newest per-object stream entry ID. That half is a SEQUENCE NUMBER, not
// milliseconds-since-epoch (both commit paths XADD an explicit
// `{sequence_num}-0` ID — SDD §12.2), so `idle_ms` evaluated to ≈ 1.77e12 for
// every layer on every tick. Both gate arms compared against that value, so
// both always passed: every layer with a new commit was re-checkpointed on
// EVERY tick, `LIVE_EDIT_CHECKPOINT_QUIESCE_SECS` and
// `LIVE_EDIT_CHECKPOINT_INTERVAL_SECS` were both inert, and the write volume
// ran at the tick cadence (6× the intended rate at the default 10 s tick /
// 60 s interval).
//
// The tests here are the regression surface, in increasing scope:
//
//   1. `gate_measures_time_since_a_real_commit_not_a_stream_entry_id_*`
//      — the units contract, driven by a REAL `update_layer` commit with no
//      injected timestamps anywhere. It pins that the value the loop derives
//      as `last_commit_ms` is a wall clock within seconds of now. Run once
//      per commit backend (Lua fast path / WATCH-MULTI-EXEC slow path), which
//      are two independent writers of the one column the gate reads.
//   2. `interval_arm_checkpoints_a_layer_that_is_never_idle`
//      — the headline behaviour: a layer committed every ~50 ms is still
//      checkpointed on the interval cadence, repeatedly, while writes are in
//      flight. This is the guarantee `LIVE_EDIT_CHECKPOINT_INTERVAL_SECS`
//      names and the one the b.yut report says was missing.
//   3. `interval_arm_bounds_checkpoint_write_volume_for_a_hot_layer`
//      — the other side of the same coin: the cadence is an upper bound on
//      the gap, which makes it a bound on write VOLUME too. Pre-fix this
//      layer would have been checkpointed on every one of ~80 ticks.
//   4. `unusable_commit_timestamp_still_checkpoints_on_the_interval_arm_and_samples_pending_age`
//      and
//      5. `unusable_commit_timestamp_on_the_skip_arm_samples_pending_age_and_writes_nothing`
//      — degraded mode. A header with no readable commit clock loses the
//      quiesce arm and NOTHING else: the interval arm still decides, in both
//      directions, and the layer stays fully observable. Refusing such a layer
//      outright would leave it permanently outside DR — the same class of
//      silent gap b.yut closed. Since b.jq3 the staleness histogram measures
//      against `last_checkpoint_ms` rather than the commit clock, so these two
//      tests pin the INVERSION of what they used to: a missing commit clock no
//      longer suppresses the observation.
//
// The sibling `checkpointer_skip_gates_test.rs` covers the per-arm decision
// table with backdated header columns (fast, no waiting); this file
// deliberately pays real wall-clock time so that nothing in the
// commit-timestamp path is faked — a fixture that writes the gate's input
// columns cannot, by construction, catch a regression in how the commit path
// produces them or in what unit the loop reads them.
//
// Sandbox env note: `TestValkey::setup_with_pool_size` reads
// `TEST_VALKEY_URL`; inside the dev container it is preset to
// `redis://valkey:6379`.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use fred::interfaces::HashesInterface;
use tokio::time::{sleep, timeout, Instant};
use tokio_util::sync::CancellationToken;

use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
use live_edit::checkpoint::{
    spawn_checkpoint_loop, CheckpointLoopConfig, CheckpointStore, TailCaptureConfig,
};
use live_edit::config::UpdateLayerBackend;
use live_edit::domain::types::ObjectState;
use live_edit::observability::metrics::{Metrics, SharedMetrics};
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::relay::RelayedUpdates;
use live_edit::storage::test_support::TestValkey;
use oodf_ffi::OwnedDelta;

use crate::common::read_header_u64;

// ────────────────────────────────────────────────────────────────────────
// Helpers — same shapes as the sibling checkpointer tests so a reader
// chasing a cadence regression does not have to context-switch styles.
// ────────────────────────────────────────────────────────────────────────

const REPLICA_ID: &str = "interval-cadence-test-replica";

fn make_metrics() -> SharedMetrics {
    Arc::new(crate::common::metrics_scrape_only(REPLICA_ID))
}

/// Read `live_edit_checkpoint_writes_total{result=<result>, kind="layer"}`.
fn read_writes(metrics: &Metrics, result: &str) -> u64 {
    live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(metrics),
        "live_edit_checkpoint_writes_total",
        &[("result", result), ("kind", "layer")],
    )
    .unwrap_or(0.0) as u64
}

/// The `live_edit_checkpoint_pending_age_seconds` cumulative bucket ladder and
/// total sample count, both parsed out of ONE render.
///
/// The pending-age histogram is the loop's own derived
/// `now_ms - last_checkpoint_ms`, in seconds, sampled ONLY for layers that
/// still carry commits their newest durable generation does not (b.jq3). It
/// deliberately does NOT project the commit clock: idle time and checkpoint
/// staleness are different quantities, and the predecessor metric
/// (`live_edit_checkpoint_lag_seconds`) conflated them, which made a rising
/// p99 mean "traffic stopped" on a perfectly healthy fleet.
///
/// **One render, every count.** The whole ladder is returned rather than a
/// single bound's count because the callers below compare two bounds against
/// each other (`within_5s` vs `within_30s` vs `total`). Rendering per bound
/// would compare two different scrapes: while the loop is still ticking, a
/// sample landing between the two renders makes an `assert_eq!` across them
/// spuriously fail. Deriving every count from one snapshot removes the race
/// rather than papering it over with a retry.
///
/// The bucket line is parsed rather than matched on an `le="…"` literal so
/// the assertion does not depend on how the exporter formats a float bound
/// (`15` vs `15.0`). Mirrors `pending_age_histogram` in the sibling metrics
/// test.
fn pending_age_buckets(metrics: &Metrics) -> (Vec<(f64, u64)>, u64) {
    let text = live_edit::observability::metrics::test_util::render(metrics);
    let mut buckets = Vec::new();
    for line in text.lines() {
        let Some(rest) = line
            .trim()
            .strip_prefix("live_edit_checkpoint_pending_age_seconds_bucket")
        else {
            continue;
        };
        let Some(le_at) = rest.find("le=\"") else {
            continue;
        };
        let after = &rest[le_at + 4..];
        let Some(end) = after.find('"') else {
            continue;
        };
        let Ok(le) = after[..end].parse::<f64>() else {
            continue; // `+Inf`
        };
        let value = line
            .rsplit_once(char::is_whitespace)
            .and_then(|(_, v)| v.trim().parse::<u64>().ok())
            .unwrap_or(0);
        buckets.push((le, value));
    }
    // Parsed out of the SAME rendered text as the buckets above.
    let total = live_edit::observability::metrics::test_util::sample_value(
        &text,
        "live_edit_checkpoint_pending_age_seconds_count",
        &[],
    )
    .unwrap_or(0.0) as u64;
    (buckets, total)
}

/// Cumulative sample count at or below `bound` seconds, read off a ladder from
/// [`pending_age_buckets`]. Buckets are cumulative, so the largest boundary at
/// or below `bound` carries the count; `max` rather than "last" so the result
/// does not depend on the exporter's bucket ordering.
fn at_or_below(buckets: &[(f64, u64)], bound: f64) -> u64 {
    buckets
        .iter()
        .filter(|(le, _)| *le <= bound)
        .map(|(_, count)| *count)
        .max()
        .unwrap_or(0)
}

/// Current value of `live_edit_checkpoint_oldest_pending_age_seconds{service}`,
/// or `None` when the gauge has never been published (an OTel gauge emits no
/// family until its first `record()`). The `None` / `Some(0.0)` distinction is
/// load-bearing in the assertions below: "the leader published a clean tick" and
/// "the gauge was never wired at all" are different claims.
fn oldest_pending_age_gauge(metrics: &Metrics, service: &str) -> Option<f64> {
    live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(metrics),
        "live_edit_checkpoint_oldest_pending_age_seconds",
        &[("service", service)],
    )
}

/// Current wall-clock time as milliseconds since the Unix epoch. Matches the
/// production helper in `src/checkpoint/checkpointer.rs`, so a stamp written
/// through it is on the same clock the loop reads.
fn now_unix_ms() -> u64 {
    u64::try_from(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock must not be pre-epoch")
            .as_millis(),
    )
    .expect("wall clock fits in u64 ms")
}

/// Bounded-poll an async predicate. Returns `true` when it holds before the
/// deadline. Same shape as the sibling checkpointer tests' `poll_until` so a
/// reader chasing a cadence regression does not have to context-switch styles.
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

/// Read `last_checkpoint_seq` off `le:{oid}:obj:hdr`, `None` when absent.
async fn read_last_checkpoint_seq(tv: &TestValkey, object_id: u64) -> Option<u64> {
    read_header_u64(tv, object_id, "last_checkpoint_seq").await
}

/// Bounded-poll until `last_checkpoint_seq` for `object_id` is present and
/// strictly greater than `above`. Returns the observed cursor, or `None` on
/// deadline. `above = 0` means "any cursor at all" (the loop only ever writes
/// a `snap_seq >= 1`).
async fn wait_for_cursor_above(
    tv: &TestValkey,
    object_id: u64,
    above: u64,
    deadline: Duration,
) -> Option<u64> {
    timeout(deadline, async {
        loop {
            if let Some(cursor) = read_last_checkpoint_seq(tv, object_id).await {
                if cursor > above {
                    return cursor;
                }
            }
            sleep(Duration::from_millis(25)).await;
        }
    })
    .await
    .ok()
}

/// Seed one populated layer in its own session. Returns
/// `(session_id, layer_name, object_id)`.
async fn seed_one_layer(tv: &TestValkey, suffix: &str) -> (String, String, u64) {
    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let layer_store = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
    let session_id = session_store
        .create(
            &format!("cadence-session-{suffix}"),
            &format!("/stage/cadence-{suffix}"),
            "cadence-test-user",
        )
        .await
        .expect("session create must succeed");
    let layer_name = format!("cadence-layer-{suffix}");
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
            "cadence-test-user",
            &initial_state,
            Arc::new(MockExternalValueStore::new()),
        )
        .await
        .expect("create_layer must succeed");
    (session_id, layer_name, oid)
}

/// One real empty-delta commit through the production write path. Empty is a
/// no-op on state but still advances `seq` and re-stamps `obj:hdr
/// last_server_ts` — which is exactly the surface under test.
///
/// Parameterized over the commit backend: `last_server_ts` is stamped by two
/// INDEPENDENT implementations — the Lua fast path (`update_layer_fast.lua`)
/// and the WATCH/MULTI/EXEC slow path (`SetObjectHeader`) — and the gate reads
/// one column from both. A units or omission regression in either one is
/// invisible to a test that only ever drives the default backend.
async fn real_commit(
    tv: &TestValkey,
    backend: UpdateLayerBackend,
    session_id: &str,
    layer_name: &str,
    object_id: u64,
) {
    let layer_store = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_backend(backend);
    layer_store
        .update_layer_relayed(
            session_id,
            layer_name,
            object_id,
            Some(OwnedDelta::default()),
            "cadence-test-client",
            0,
        )
        .await
        .expect("update_layer must succeed");
}

/// Handle to a background task committing to one layer every `cadence`.
struct HotWriter {
    stop: Arc<AtomicBool>,
    commits: Arc<AtomicUsize>,
    handle: tokio::task::JoinHandle<()>,
}

impl HotWriter {
    /// Commit an empty delta every `cadence` until stopped. Real commits
    /// through `LayerStore::update_layer` — no injected timestamps — so the
    /// gate sees exactly what production would see for a layer under
    /// continuous write.
    fn spawn(
        tv: &TestValkey,
        session_id: &str,
        layer_name: &str,
        oid: u64,
        cadence: Duration,
    ) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let commits = Arc::new(AtomicUsize::new(0));
        let layer_store = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
        let session_id = session_id.to_string();
        let layer_name = layer_name.to_string();
        let handle = {
            let stop = stop.clone();
            let commits = commits.clone();
            tokio::spawn(async move {
                while !stop.load(Ordering::Relaxed) {
                    if layer_store
                        .update_layer_relayed(
                            &session_id,
                            &layer_name,
                            oid,
                            Some(OwnedDelta::default()),
                            "cadence-hot-writer",
                            0,
                        )
                        .await
                        .is_err()
                    {
                        // The pool is torn down at test exit; stop quietly
                        // rather than poisoning the join with a panic.
                        return;
                    }
                    commits.fetch_add(1, Ordering::Relaxed);
                    sleep(cadence).await;
                }
            })
        };
        Self {
            stop,
            commits,
            handle,
        }
    }

    fn commits(&self) -> usize {
        self.commits.load(Ordering::Relaxed)
    }

    async fn stop(self) {
        self.stop.store(true, Ordering::Relaxed);
        self.handle.await.expect("hot writer must not panic");
    }
}

/// Cancel the loop and join it under a bounded timeout.
async fn cancel_and_join(cancel: CancellationToken, handle: tokio::task::JoinHandle<()>) {
    cancel.cancel();
    timeout(Duration::from_secs(5), handle)
        .await
        .expect("checkpoint loop must exit within 5 s of cancellation")
        .expect("checkpoint loop must not panic on shutdown");
}

// ────────────────────────────────────────────────────────────────────────
// 1 — Units contract on a REAL commit.
//
// Nothing here fabricates a commit timestamp: the layer is created and then
// committed through `LayerStore::update_layer`, so `obj:hdr last_server_ts`
// is whatever the production write path stamped. The loop then has to derive
// a `last_commit_ms` from it that is genuinely "a moment ago".
//
// Three independent assertions pin that, each failing loudly under a
// different reversion:
//
//   (a) The raw column is microsecond-scale and within seconds of now. This
//       is the producer half of the contract — if a commit path ever starts
//       writing milliseconds, the loop's conversion silently backdates every
//       layer to ~1970 and both arms start passing again. (This assertion
//       reads the column, not the loop, so it is the one that survives a
//       rewrite of the consumer.)
//   (b) The gate does NOT fire before `quiesce` has actually elapsed since
//       that commit. Under the pre-fix stream-ID read `idle_ms` was ≈ 1.77e12
//       — 56 years — so the very first tick wrote a checkpoint. The negative
//       window catches it with ~2 s of slack.
//   (c) The gate DOES fire once `quiesce` elapses, and every pending-age
//       sample the loop published is ≤ 30 s. Dropping the µs→ms conversion puts
//       `last_commit_ms` ~56,000 years in the FUTURE, where `saturating_sub`
//       floors idle at 0: the layer never quiesces, this poll times out, and
//       (b) alone would not have caught it.
//
// `interval` is parked at an hour so only the quiesce arm can fire, and
// `last_checkpoint_ms` is stamped with the real current clock so the
// "never checkpointed" shortcut (absent column → 0 → interval arm fires
// immediately) does not pre-empt the arm under test. That single stamp is a
// genuine `now`, not a fabricated age.
//
// Run once per commit backend. `last_server_ts` has two independent writers —
// the Lua fast path (`update_layer_fast.lua`) and the WATCH/MULTI/EXEC slow
// path (`LayerStore::update_layer`'s `SetObjectHeader`) — and the gate reads
// the single column both produce. Driving only the default (Lua) backend
// would leave a units or omission regression in the slow path invisible here,
// even though `LIVE_EDIT_UPDATE_LAYER_BACKEND=txn` is a supported production
// configuration and structural-command deltas take that path unconditionally.
// ────────────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn gate_measures_time_since_a_real_commit_not_a_stream_entry_id_lua_backend() {
    assert_gate_measures_a_real_commit(UpdateLayerBackend::Lua, "real-commit-lua").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn gate_measures_time_since_a_real_commit_not_a_stream_entry_id_txn_backend() {
    assert_gate_measures_a_real_commit(UpdateLayerBackend::Txn, "real-commit-txn").await;
}

async fn assert_gate_measures_a_real_commit(backend: UpdateLayerBackend, suffix: &str) {
    /// Time the quiesce arm needs after the commit. Long enough that the
    /// negative window below has multi-second slack on a loaded runner,
    /// short enough to keep the test around 4 s.
    const QUIESCE: Duration = Duration::from_secs(3);
    /// How long after the commit the negative assertion is taken. 2 s of
    /// margin under `QUIESCE`.
    const NEGATIVE_WINDOW: Duration = Duration::from_secs(1);

    let tv = TestValkey::setup_with_pool_size(8).await;
    let metrics = make_metrics();
    let (session_id, layer_name, oid) = seed_one_layer(&tv, suffix).await;

    // A real commit through the production path. Everything the gate reads
    // about "when did this layer last change" originates here.
    real_commit(&tv, backend, &session_id, &layer_name, oid).await;
    let commit_at = Instant::now();

    // (a) Producer half of the units contract, read straight off the column
    // production wrote.
    let last_server_ts = read_header_u64(&tv, oid, "last_server_ts")
        .await
        .expect("every commit path must stamp obj:hdr last_server_ts");
    let now_micros = live_edit::domain::time::now_micros();
    let commit_age_micros = now_micros.saturating_sub(last_server_ts);
    assert!(
        commit_age_micros < 10_000_000,
        "obj:hdr last_server_ts must be MICROSECONDS since the epoch and must \
         name the commit that just happened; got {last_server_ts} against a \
         current µs clock of {now_micros} (difference {commit_age_micros} µs). \
         A millisecond-scale value here reads as ~1970 to the checkpoint gate \
         and re-opens b.yut"
    );
    assert!(
        last_server_ts <= now_micros,
        "obj:hdr last_server_ts ({last_server_ts}) must not be ahead of the \
         current µs clock ({now_micros})"
    );

    // Close the interval arm with a real `now` so the quiesce arm is the only
    // thing that can open the gate. Absent, the column reads as 0 = "never
    // checkpointed" and the interval arm fires on the first tick regardless
    // of the commit clock.
    let now_ms = u64::try_from(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock must not be pre-epoch")
            .as_millis(),
    )
    .expect("wall clock fits in u64 ms");
    let header_key = tv.key_schema().obj_header(&oid.to_string());
    let _: () = tv
        .client()
        .pool
        .hset(&header_key, ("last_checkpoint_ms", now_ms.to_string()))
        .await
        .expect("HSET last_checkpoint_ms must succeed");

    let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
    let cancel = CancellationToken::new();
    let config = CheckpointLoopConfig {
        tick: Duration::from_millis(50),
        interval: Duration::from_secs(3600),
        quiesce: QUIESCE,
        retain_generations: 2,
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
    let handle = spawn_checkpoint_loop(
        tv.client().clone(),
        tv.key_schema().clone(),
        store,
        config,
        Some(metrics.clone()),
        cancel.clone(),
    );

    // (b) Negative window: nothing may be checkpointed yet.
    let elapsed = commit_at.elapsed();
    if elapsed < NEGATIVE_WINDOW {
        sleep(NEGATIVE_WINDOW - elapsed).await;
    }
    // Precondition, checked immediately before the negative read: the
    // observation must still be INSIDE the quiesce floor. A runner that
    // stalled (loaded CI box, a paused container, a slow Valkey setup) can
    // push `commit_at.elapsed()` past `QUIESCE`, at which point a checkpoint
    // is the CORRECT behaviour and the assertion below would fail for a
    // reason that has nothing to do with the gate. Separating the two failure
    // modes here means a red build is never ambiguous.
    let elapsed_at_read = commit_at.elapsed();
    assert!(
        elapsed_at_read < QUIESCE,
        "TEST-RUNNER STALL, not a gate regression: {elapsed_at_read:?} elapsed \
         between the commit and the negative observation, which is already past \
         the {QUIESCE:?} quiesce floor — the loop would be RIGHT to have \
         checkpointed the layer by now. Re-run; if this recurs, raise QUIESCE \
         and NEGATIVE_WINDOW together"
    );
    let early_cursor = read_last_checkpoint_seq(&tv, oid).await;
    let early_success = read_writes(&metrics, "success");
    assert!(
        early_cursor.is_none() && early_success == 0,
        "only {:?} has passed since the commit, well inside the {QUIESCE:?} \
         quiesce floor, so the layer must NOT have been checkpointed yet; got \
         last_checkpoint_seq={early_cursor:?} and {early_success} success \
         write(s). A gate that derives its commit time from the stream entry \
         ID reads ~56 years of idleness and writes on the first tick (b.yut)",
        commit_at.elapsed()
    );

    // (c) Positive window: the quiesce arm must fire once the floor is
    // genuinely past. Generous deadline — the assertion is "eventually", the
    // sharp edge is (b).
    let wrote = wait_for_cursor_above(&tv, oid, 0, Duration::from_secs(20))
        .await
        .is_some();
    assert!(
        wrote,
        "the layer must be checkpointed once {QUIESCE:?} has elapsed since its \
         last commit; the cursor never appeared. A gate that reads \
         last_server_ts as milliseconds places the last commit ~56,000 years \
         in the future, where saturating_sub floors idle time at 0 and the \
         layer never quiesces"
    );

    cancel_and_join(cancel, handle).await;

    // The pending-age projection of the same run. This layer has un-persisted
    // commits throughout (seq=2 against an absent cursor) and a REAL
    // `last_checkpoint_ms` stamped moments ago, so every sample the loop takes
    // — on the pre-quiesce Skip arm and on the success arm — is a small
    // positive number.
    //
    // What this catches: a regression that folds the "never checkpointed"
    // sentinel into the histogram as `now - 0` records ~1.77e9 seconds and
    // lands outside the <=15 s bucket, poisoning `_sum` for the whole fleet.
    // That is the same defect shape b.yut fixed on the commit clock, and b.jq3
    // is where it could have been re-introduced under a new metric name.
    let (buckets, total) = pending_age_buckets(&metrics);
    let within_30s = at_or_below(&buckets, 30.0);
    assert!(
        total >= 1,
        "the loop must have taken at least one pending-age sample for a layer \
         with un-persisted commits and a durable generation stamped moments ago"
    );
    assert_eq!(
        within_30s, total,
        "every live_edit_checkpoint_pending_age_seconds sample must land in the \
         <=30 s bucket for a layer whose last_checkpoint_ms was stamped seconds \
         ago; {within_30s} of {total} did. A sample outside it means the loop is \
         measuring against something other than that stamp — a sentinel-valued \
         `now - 0`, or the µs commit clock read as ms"
    );

    // The tick aggregate, and — the half that needs the `Some`/`None`
    // distinction — the leader published it at all.
    //
    // The range is `0.0..=30.0` rather than a positive band because the gauge
    // is an END-OF-TICK-PENDING maximum: a layer this tick WROTE is no longer
    // pending and contributes nothing, so the last published tick reads 0.0 if
    // it was the one that checkpointed this layer, and the layer's own small
    // age if it was a pre-quiesce Skip tick. Both are correct; which one the
    // run lands on depends on where cancellation fell. What is pinned here is
    // the upper bound — a maximum outside the distribution above would mean
    // the gauge is fed from a different quantity than the histogram.
    let oldest = oldest_pending_age_gauge(&metrics, REPLICA_ID);
    let oldest = oldest.expect(
        "the leader must publish live_edit_checkpoint_oldest_pending_age_seconds \
         once per completed tick; an absent series means the gauge is unwired \
         or every tick bailed before the SCAN drained",
    );
    assert!(
        (0.0..=30.0).contains(&oldest),
        "the per-tick maximum must agree with the distribution above for a \
         single-layer fixture; got {oldest}s"
    );
}

// ────────────────────────────────────────────────────────────────────────
// 2 — The headline b.yut regression: the interval arm is a cadence upper
//     bound, so it must fire on a layer that is NEVER idle.
//
// A background task commits every ~50 ms, so the layer's idle time never
// exceeds a tick — under the pre-fix gate (both arms measuring idle time)
// such a layer becomes eligible only if `min(quiesce, interval)` is smaller
// than the commit cadence, i.e. essentially never. Here `quiesce` is parked
// at an hour precisely so the quiesce arm CANNOT be what lets a write
// through: every checkpoint observed below is attributable to the interval
// arm.
//
// The assertion is not merely "one checkpoint happened" — the first one is
// explainable by `last_checkpoint_ms` being absent ("never checkpointed").
// It is that the cursor advances at least TWICE, with the second advance
// landing while commits are still in flight, which can only come from the
// interval arm measuring time since the last CHECKPOINT.
// ────────────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn interval_arm_checkpoints_a_layer_that_is_never_idle() {
    let tv = TestValkey::setup_with_pool_size(8).await;
    let metrics = make_metrics();
    let (session_id, layer_name, oid) = seed_one_layer(&tv, "never-idle").await;

    let writer = HotWriter::spawn(
        &tv,
        &session_id,
        &layer_name,
        oid,
        Duration::from_millis(50),
    );

    let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
    let cancel = CancellationToken::new();
    let config = CheckpointLoopConfig {
        tick: Duration::from_millis(50),
        interval: Duration::from_secs(1),
        // An hour: the quiesce arm is structurally unable to fire, so every
        // write below is the interval arm's doing.
        quiesce: Duration::from_secs(3600),
        retain_generations: 2,
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
    let handle = spawn_checkpoint_loop(
        tv.client().clone(),
        tv.key_schema().clone(),
        store,
        config,
        Some(metrics.clone()),
        cancel.clone(),
    );

    // First advance. Explainable on its own by "never checkpointed", so it is
    // only a staging point for the real assertion.
    let first_cursor = wait_for_cursor_above(&tv, oid, 0, Duration::from_secs(20))
        .await
        .unwrap_or_else(|| {
            panic!(
                "a continuously-written layer must be checkpointed at least \
                 once within 20 s; observed {} writer commits",
                writer.commits()
            )
        });
    let commits_at_first = writer.commits();

    // Second advance — the b.yut claim. Under an interval arm that measures
    // time since the last COMMIT this never happens: the layer is committed
    // every 50 ms and can never accumulate a full second of idleness.
    let second = wait_for_cursor_above(&tv, oid, first_cursor, Duration::from_secs(20)).await;
    // Read the writer's progress BEFORE stopping it: the point of the test is
    // that the second checkpoint landed mid-write, not after the layer fell
    // quiet.
    let commits_at_second = writer.commits();

    let second_cursor = second.unwrap_or_else(|| {
        panic!(
            "last_checkpoint_seq must advance a SECOND time within 20 s of the \
             first while the layer is committed every ~50 ms — that advance is \
             the LIVE_EDIT_CHECKPOINT_INTERVAL_SECS upper bound doing its job. \
             It stalled at {first_cursor} after {commits_at_second} writer \
             commits. Pre-b.yut both gate arms measured time since the last \
             COMMIT, so a layer under continuous write was never checkpointed \
             on a cadence at all"
        )
    });
    assert!(
        second_cursor > first_cursor,
        "the second checkpoint must name a strictly newer cut than the first \
         ({second_cursor} vs {first_cursor})"
    );
    assert!(
        commits_at_second > commits_at_first,
        "commits must still have been in flight across the two checkpoints \
         (first observed at {commits_at_first} commits, second at \
         {commits_at_second}) — otherwise the layer went quiet and the advance \
         proves nothing about a hot layer"
    );

    cancel_and_join(cancel, handle).await;
    writer.stop().await;

    let successes = read_writes(&metrics, "success");
    assert!(
        successes >= 2,
        "at least two successful layer checkpoints must have been counted for \
         a continuously-written layer over two interval periods; got \
         {successes}"
    );
}

// ────────────────────────────────────────────────────────────────────────
// 3 — The write-volume bound.
//
// Same hot-layer shape, `interval = 2 s`, observed for ~4 s. The interval arm
// stamps `last_checkpoint_ms` on every success, so successive checkpoints of
// one layer are at least `interval` apart: over a 4 s window that is 2-3
// writes, whatever the tick rate.
//
// Pre-fix the gate passed on EVERY tick for any layer with a new commit, so
// the same window at a 50 ms tick produced ~80 writes — the 6× (here 27×)
// amplification the b.yut report describes, in the operator-visible counter.
// The assertion is a closed range rather than a ceiling so that a regression
// in the other direction (the interval arm going inert again, e.g. by
// `last_checkpoint_ms` never being stamped) also fails here.
// ────────────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn interval_arm_bounds_checkpoint_write_volume_for_a_hot_layer() {
    /// Cadence under test.
    const INTERVAL: Duration = Duration::from_secs(2);
    /// Observation window. Two full interval periods plus slack, so the
    /// expected write count is 2 (loop start drifts past the first boundary)
    /// or 3 (t≈0, t≈2, t≈4); 4 is arithmetically impossible at a 2 s minimum
    /// spacing.
    const WINDOW: Duration = Duration::from_millis(4_200);
    /// The tick rate the volume must NOT track. 4.2 s / 50 ms ≈ 84 ticks.
    const TICK: Duration = Duration::from_millis(50);

    let tv = TestValkey::setup_with_pool_size(8).await;
    let metrics = make_metrics();
    let (session_id, layer_name, oid) = seed_one_layer(&tv, "volume-bound").await;

    let writer = HotWriter::spawn(
        &tv,
        &session_id,
        &layer_name,
        oid,
        Duration::from_millis(50),
    );

    let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
    let cancel = CancellationToken::new();
    let config = CheckpointLoopConfig {
        tick: TICK,
        interval: INTERVAL,
        quiesce: Duration::from_secs(3600),
        retain_generations: 2,
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
    let handle = spawn_checkpoint_loop(
        tv.client().clone(),
        tv.key_schema().clone(),
        store,
        config,
        Some(metrics.clone()),
        cancel.clone(),
    );

    // A fixed observation window, not a coordination sleep: the quantity
    // under test IS "how many writes per unit time", so the window has to be
    // wall-clock and has to be the same on every runner.
    sleep(WINDOW).await;

    cancel_and_join(cancel, handle).await;
    let commits = writer.commits();
    writer.stop().await;

    let successes = read_writes(&metrics, "success");
    assert!(
        (2..=3).contains(&successes),
        "over a {WINDOW:?} window at interval={INTERVAL:?} a continuously-\
         written layer must be checkpointed 2-3 times (successive checkpoints \
         are >= interval apart because last_checkpoint_ms is stamped on each \
         success); got {successes} against {commits} writer commits at a \
         {TICK:?} tick. ~80 means the gate is passing on every tick — the \
         b.yut defect, where LIVE_EDIT_CHECKPOINT_INTERVAL_SECS was inert and \
         write volume tracked the tick rate. 0-1 means the interval arm has \
         gone inert in the other direction"
    );
}

// ────────────────────────────────────────────────────────────────────────
// 4 — A header with NO usable commit timestamp runs the gate in DEGRADED
//     MODE: the quiesce arm is unavailable, the interval arm still decides,
//     and pending age is STILL sampled for that layer.
//
// The gate has exactly one input it cannot obtain elsewhere, and b.yut is what
// happens when a plausible-looking default stands in for it: the pre-fix loop
// treated an unreadable commit time as "now", which quietly made both arms
// unfalsifiable. The replacement refuses to guess — but refusing to guess is
// NOT the same as refusing the layer. Dropping the layer from the tick would
// leave it permanently outside DR, which is the same class of silent gap
// b.yut exists to close. So:
//
//   - the quiesce arm is unavailable (no commit clock to measure idleness),
//   - the interval arm is untouched (it measures against `last_checkpoint_ms`
//     and needs no commit clock at all), so the layer is STILL checkpointed
//     on cadence, and
//   - `live_edit_checkpoint_pending_age_seconds` still records for it.
//
// That last bullet is the b.jq3 INVERSION and the reason this test exists in
// its current shape. Under the predecessor metric a missing commit clock cost
// the layer its observation entirely, because the quantity being measured was
// "seconds since the last commit" — unobtainable without the clock. Pending
// age measures against `last_checkpoint_ms` instead, so a layer with no commit
// clock at all is still fully observable: its durable copy has an age, its
// commits are un-persisted, and staleness is exactly what the operator needs
// to see. Losing the commit clock must cost the layer the quiesce arm and
// NOTHING else — not its DR coverage, and not its visibility.
//
// Two layers, because "no usable commit timestamp" has two distinct shapes and
// only one of them is a data-shape violation:
//
//   A — `last_server_ts` ABSENT (HDEL'd). The unparseable / corrupted case.
//   B — `last_server_ts` present and literally `"0"`. REACHABLE in production,
//       not defensive: the assembler folds a missing `last_server_ts` to `0`
//       into `CheckpointMeta::server_ts`, and `apply_checkpoint` →
//       `write_object` stamps that `0` straight back onto the restored header.
//       A gate that read it as a real epoch timestamp would compute ~56 years
//       of idleness and fire the quiesce arm unconditionally, forever. The
//       zero-fold lives in `run_one_tick` (not in the pure `checkpoint_due`),
//       so this is the surface that pins it.
//
// Both layers are given a REAL, backdated `last_checkpoint_ms` — 7 s old — so
// they enter the tick as "has a durable generation, has un-persisted commits",
// which is precisely the population the histogram is scoped to. 7 s is chosen
// to sit between two bucket boundaries: a sample derived from anything other
// than that column (a fabricated 0.0, the absent commit clock folded to 0, a
// post-write recomputation) lands in the `<=5` bucket instead and fails.
//
// `interval = 0` + `quiesce = 1 hour` makes the two arms distinguishable: the
// quiesce arm cannot fire even for a layer WITH a commit clock, so every write
// observed here is attributable to the interval arm. (The pre-reshape fixture
// set both thresholds to zero, which could not tell "the interval arm carried
// the layer" apart from "the quiesce arm did".)
//
// Not asserted: the degraded-mode WARN. Capturing it needs a process-global
// `tracing` subscriber, which is unsound in a multi-test integration binary,
// and the test-writing guide rules exact log messages out as observability
// rather than contract.
// ────────────────────────────────────────────────────────────────────────

/// Backdated durable-generation stamp shared by both layers in test 4. Sits
/// strictly between the `<=5` and `<=15` histogram bucket boundaries.
const DURABLE_STAMP_AGE_MS: u64 = 7_000;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unusable_commit_timestamp_still_checkpoints_on_the_interval_arm_and_samples_pending_age() {
    let tv = TestValkey::setup_with_pool_size(8).await;
    let metrics = make_metrics();
    let (_session_a, _layer_a, oid_absent) = seed_one_layer(&tv, "commit-ts-absent").await;
    let (_session_b, _layer_b, oid_zero) = seed_one_layer(&tv, "commit-ts-zero").await;

    // Layer A: remove the gate's commit-clock input entirely. Every other
    // precondition holds — the header is otherwise intact and `seq` (1) is
    // ahead of `last_checkpoint_seq` (absent → 0), so the skip-if-no-commits
    // gate does not fire either.
    let header_a = tv.key_schema().obj_header(&oid_absent.to_string());
    let removed: u64 = tv
        .client()
        .pool
        .hdel(&header_a, "last_server_ts")
        .await
        .expect("HDEL last_server_ts must succeed");
    assert_eq!(
        removed, 1,
        "create_layer must have stamped obj:hdr last_server_ts for the HDEL to \
         remove; the fixture is not exercising what it thinks it is"
    );

    // Layer B: the reachable literal zero a restored header carries.
    let header_b = tv.key_schema().obj_header(&oid_zero.to_string());
    let _: () = tv
        .client()
        .pool
        .hset(&header_b, ("last_server_ts", "0"))
        .await
        .expect("HSET last_server_ts=0 must succeed");

    // Both layers get a durable-generation stamp 7 s in the past. Without it
    // they read as "never checkpointed", which is a categorically different
    // operator situation (the `layers_never_written` gauge, covered by Case I
    // in `checkpointer_metrics_test.rs`) and produces no histogram sample at
    // all.
    let now_ms = now_unix_ms();
    for header in [&header_a, &header_b] {
        let _: () = tv
            .client()
            .pool
            .hset(
                header,
                (
                    "last_checkpoint_ms",
                    now_ms.saturating_sub(DURABLE_STAMP_AGE_MS).to_string(),
                ),
            )
            .await
            .expect("HSET backdated last_checkpoint_ms must succeed");
    }

    let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
    let cancel = CancellationToken::new();
    let config = CheckpointLoopConfig {
        tick: Duration::from_millis(50),
        // Interval at zero: the cadence arm is due on every tick. Quiesce at
        // an hour: structurally unable to fire, so the two arms are
        // distinguishable and every write below is the interval arm's.
        interval: Duration::ZERO,
        quiesce: Duration::from_secs(3600),
        retain_generations: 2,
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
    let handle = spawn_checkpoint_loop(
        tv.client().clone(),
        tv.key_schema().clone(),
        store,
        config,
        Some(metrics.clone()),
        cancel.clone(),
    );

    // Both layers must be checkpointed despite having no usable commit clock.
    // A loop that refused them would time out here — and would be leaving a
    // layer permanently outside DR in production.
    let cursor_absent = wait_for_cursor_above(&tv, oid_absent, 0, Duration::from_secs(20));
    let cursor_zero = wait_for_cursor_above(&tv, oid_zero, 0, Duration::from_secs(20));
    assert!(
        cursor_absent.await.is_some(),
        "a layer whose obj:hdr carries NO last_server_ts must still be \
         checkpointed by the interval arm — that arm measures against \
         last_checkpoint_ms and needs no commit clock. Refusing the layer \
         outright leaves it permanently outside DR (b.yut review R2)"
    );
    assert!(
        cursor_zero.await.is_some(),
        "a layer whose obj:hdr carries last_server_ts=0 — the value a restored \
         header actually carries — must still be checkpointed by the interval \
         arm, and must NOT be treated as a 1970 commit"
    );

    // Let both layers go round at least twice more, now through the
    // skip-if-no-commits arm (their `seq` equals the cursor the success arm
    // just wrote). Four skips is two full post-write ticks over two layers.
    let metrics_for_poll = metrics.clone();
    let saw_skips = poll_until(Duration::from_secs(20), || {
        let m = metrics_for_poll.clone();
        async move { read_writes(&m, "skipped") >= 4 }
    })
    .await;
    assert!(
        saw_skips,
        "both layers must have been observed at least twice more after their \
         checkpoint landed (the skip-if-no-commits arm); got {} skip(s)",
        read_writes(&metrics, "skipped")
    );

    // Stop the loop before reading the histogram so no late tick can slip a
    // sample in after the exact-count assertion below is set up.
    cancel_and_join(cancel, handle).await;

    let (buckets, total) = pending_age_buckets(&metrics);
    let within_5s = at_or_below(&buckets, 5.0);
    let within_30s = at_or_below(&buckets, 30.0);

    // The inversion, stated as an exact count: one sample per layer, taken on
    // the success arm, and nothing afterwards.
    assert_eq!(
        total, 2,
        "each of the two layers must contribute EXACTLY one pending-age sample \
         — the one its successful checkpoint took — and nothing after that, \
         because a fully-checkpointed layer is no longer pending. Got {total}. \
         Zero means a missing commit clock is still suppressing the \
         observation, which is the predecessor metric's behaviour and leaves a \
         degraded layer invisible; more than two means the skip-if-no-commits \
         arm is sampling idle layers, which is the quiet-fleet false positive \
         b.jq3 removed"
    );
    assert_eq!(
        within_5s, 0,
        "no sample may land in the <=5 s bucket: the durable stamp is \
         {DURABLE_STAMP_AGE_MS} ms old, so a sub-5 s value can only come from \
         measuring the wrong thing — a fabricated 0.0 for the absent commit \
         clock, or a post-write recomputation against the stamp the success arm \
         had just written (which yields ~0 for every layer and re-masks the \
         tail the metric exists to expose)"
    );
    assert_eq!(
        within_30s, total,
        "both samples must land in the <=30 s bucket, derived from a 7 s-old \
         last_checkpoint_ms; {within_30s} of {total} did"
    );
}

// ────────────────────────────────────────────────────────────────────────
// 5 — Degraded mode on the SKIP side: with both thresholds closed, a layer
//     with no usable commit timestamp is skipped, writes nothing, and STILL
//     samples pending age off its durable-generation stamp.
//
// Test 4 covers the arms a due layer takes. This covers the fourth arm — the
// gate's own `Skip` — which is the one an operator sees most of the time in a
// steady-state fleet, and the arm that carries the healthy fleet's whole
// steady-state distribution. Nothing is written on it either way, so the two
// observables below are the ONLY trace a regression would leave:
//
//   - `last_checkpoint_seq` still absent. This is now where a fabricated
//     commit clock surfaces: quiesce is parked at an hour, so a loop that
//     folded the missing `last_server_ts` to 0 would read ~56 years of
//     idleness, fire the quiesce arm, and write. A cursor appearing here is
//     that regression (b.yut).
//   - A pending-age sample derived from `last_checkpoint_ms`, not from the
//     absent commit clock. The stamp is backdated 7 s, between two bucket
//     boundaries, so a value sourced from anywhere else lands in `<=5`.
//
// `last_checkpoint_ms` is 7 s old against a 1-hour interval, so the interval
// arm is closed too and `Skip` is the only reachable decision.
// ────────────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unusable_commit_timestamp_on_the_skip_arm_samples_pending_age_and_writes_nothing() {
    let tv = TestValkey::setup_with_pool_size(8).await;
    let metrics = make_metrics();
    let (_session_id, _layer_name, oid) = seed_one_layer(&tv, "skip-arm-no-ts").await;

    let header_key = tv.key_schema().obj_header(&oid.to_string());
    let removed: u64 = tv
        .client()
        .pool
        .hdel(&header_key, "last_server_ts")
        .await
        .expect("HDEL last_server_ts must succeed");
    assert_eq!(removed, 1, "the fixture must have a column to remove");

    // Close the interval arm, and give the histogram something to measure: a
    // genuine durable-generation stamp 7 s in the past, far inside the 1-hour
    // interval.
    let now_ms = now_unix_ms();
    let _: () = tv
        .client()
        .pool
        .hset(
            &header_key,
            (
                "last_checkpoint_ms",
                now_ms.saturating_sub(DURABLE_STAMP_AGE_MS).to_string(),
            ),
        )
        .await
        .expect("HSET last_checkpoint_ms must succeed");

    let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
    let cancel = CancellationToken::new();
    let config = CheckpointLoopConfig {
        tick: Duration::from_millis(50),
        interval: Duration::from_secs(3600),
        quiesce: Duration::from_secs(3600),
        retain_generations: 2,
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
    let handle = spawn_checkpoint_loop(
        tv.client().clone(),
        tv.key_schema().clone(),
        store,
        config,
        Some(metrics.clone()),
        cancel.clone(),
    );

    // The gate's Skip arm is the observable: `result="skipped"` accumulates
    // while nothing is written.
    let metrics_for_poll = metrics.clone();
    let saw_skip = poll_until(Duration::from_secs(20), || {
        let m = metrics_for_poll.clone();
        async move { read_writes(&m, "skipped") >= 1 }
    })
    .await;
    assert!(
        saw_skip,
        "a layer below both thresholds must be counted as result=\"skipped\"; \
         the counter never moved, so the loop is not reaching this layer at all"
    );

    let cursor = read_last_checkpoint_seq(&tv, oid).await;
    assert!(
        cursor.is_none(),
        "nothing may be persisted on the Skip arm; got \
         last_checkpoint_seq={cursor:?}. With quiesce parked at an hour, a \
         cursor here means the missing last_server_ts was folded to 0 and read \
         as ~56 years of idleness (b.yut)"
    );

    // The Skip arm's pending-age samples, taken while only that arm can run.
    // Every one of them must come from the 7 s-old durable stamp.
    //
    // The loop is deliberately STILL RUNNING here (the liveness control below
    // needs it), so all three counts come off a single render — two renders
    // would let a tick land a sample between them and make the
    // `within_30s == total` comparison span two different scrapes.
    let (buckets, total) = pending_age_buckets(&metrics);
    let within_5s = at_or_below(&buckets, 5.0);
    let within_30s = at_or_below(&buckets, 30.0);
    assert!(
        total >= 1,
        "the Skip arm must sample pending age — this layer has un-persisted \
         commits and a durable generation, which is the histogram's entire \
         scope, and it carries the steady-state distribution of a healthy \
         fleet. A missing commit clock costs the layer the quiesce arm and \
         nothing else (b.jq3)"
    );
    assert_eq!(
        within_5s, 0,
        "no sample may land in the <=5 s bucket: the durable stamp is \
         {DURABLE_STAMP_AGE_MS} ms old, so a sub-5 s value means the sample is \
         sourced from the absent commit clock folded to 0 rather than from \
         last_checkpoint_ms"
    );
    assert_eq!(
        within_30s, total,
        "all {total} Skip-arm samples must land in the <=30 s bucket; \
         {within_30s} did"
    );

    // The per-tick aggregate agrees, and is non-zero: this is the gauge an
    // operator alerts on, and a fleet with one stuck layer must not read 0.
    // Unambiguous under the end-of-tick-pending semantics — nothing is ever
    // written here, so this layer is pending when every tick ends and IS the
    // maximum, with no success arm to zero it.
    //
    // Polled rather than read outright: the aggregates are published only
    // after the tick's SCAN drains, so the tick that bumped `skipped` above
    // may not have reached the publish yet.
    let metrics_for_poll = metrics.clone();
    let published = poll_until(Duration::from_secs(10), || {
        let m = metrics_for_poll.clone();
        async move { oldest_pending_age_gauge(&m, REPLICA_ID).is_some() }
    })
    .await;
    assert!(
        published,
        "the leader must publish the oldest-pending-age gauge once a tick's \
         SCAN drains"
    );
    let oldest = oldest_pending_age_gauge(&metrics, REPLICA_ID)
        .expect("the leader must publish the oldest-pending-age gauge each tick");
    assert!(
        (5.0..=30.0).contains(&oldest),
        "the per-tick maximum over a single pending layer must be that layer's \
         own age (~{DURABLE_STAMP_AGE_MS} ms); got {oldest}s"
    );

    // Liveness control: open the interval arm by backdating the cadence stamp
    // two hours. The SAME loop must now check the SAME layer in — proving the
    // skip above was the gate's decision, not a wedged or leaderless loop.
    let _: () = tv
        .client()
        .pool
        .hset(
            &header_key,
            (
                "last_checkpoint_ms",
                now_ms.saturating_sub(7_200_000).to_string(),
            ),
        )
        .await
        .expect("HSET backdated last_checkpoint_ms must succeed");
    let recovered = wait_for_cursor_above(&tv, oid, 0, Duration::from_secs(20)).await;
    cancel_and_join(cancel, handle).await;
    assert!(
        recovered.is_some(),
        "once the interval arm opens, the layer must be checkpointed even \
         though it still has no commit clock — the skip above has to be the \
         gate's Skip decision, not a wedged loop or a blanket refusal of \
         timestamp-less layers"
    );
}

// ────────────────────────────────────────────────────────────────────────
// 6 — b.vnw: the success arm RETIRES the restore bump in the same HSET.
//
// A DR restore seeds `seq` a whole band above its restore floor while
// stamping `last_checkpoint_seq = snap_seq`, and records the band it applied
// in `obj:hdr restore_seq_bump` so the next assembly can subtract it and
// measure real commits rather than the artificial jump. That correction must
// apply to EXACTLY ONE assembly. Once a post-restore generation lands,
// `last_checkpoint_seq` is that generation, the jump is behind it, and a
// surviving bump would silently deflate every later span by a full band —
// re-opening the very sequence-reuse window the span exists to close, and
// doing it quietly, because a deflated span reads as a cold layer.
//
// So the success arm writes `restore_seq_bump = 0` in the SAME HSET that
// advances the two cursor columns (b.4fh/b.yut reasoning, third column): all
// three describe one generation and must never disagree about it. The sibling
// `checkpointer_metrics_test.rs` Case D pins the other arm — a
// `SkippedNotQuiesced` tick must leave an outstanding bump owed.
//
// The fixture also lands on the `saturating_sub` arm on purpose. The planted
// bump (4242) far exceeds the fixture's `snap_seq` (1), so the raw difference
// `snap_seq − last_checkpoint_seq − restore_seq_bump` goes negative and must
// fold to 0 — the documented "no measurement available" sentinel — rather
// than wrapping to ~1.8e19, which as a band would trip the
// `F64_SAFE_INT_MAX` refusal on the next restore and leave the layer
// unrecoverable.
//
// Exactly one generation is written despite both gates being wide open: after
// the first success `last_checkpoint_seq == seq`, so the skip-if-no-commits
// gate closes and nothing further touches the header. That makes the
// single-blob assertion below deterministic rather than a race.
// ────────────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn success_arm_clears_the_restore_seq_bump_in_the_same_hset_as_the_cursor() {
    const PLANTED_RESTORE_BUMP: u64 = 4242;

    let tv = TestValkey::setup_with_pool_size(8).await;
    let (_session_id, _layer_name, oid) = seed_one_layer(&tv, "restore-bump-retire").await;

    // Plant an outstanding restore bump, as `apply_checkpoint` would have.
    let header_key = tv.key_schema().obj_header(&oid.to_string());
    let _: () = tv
        .client()
        .pool
        .hset(
            &header_key,
            ("restore_seq_bump", PLANTED_RESTORE_BUMP.to_string()),
        )
        .await
        .expect("HSET restore_seq_bump must succeed");
    assert_eq!(
        read_header_u64(&tv, oid, "restore_seq_bump").await,
        Some(PLANTED_RESTORE_BUMP),
        "fixture precondition: the bump must be outstanding before the loop runs"
    );

    // Both gates open so the first tick reaches the assembler and writes.
    let store = Arc::new(InMemoryCheckpointStore::new());
    let store_for_loop: Arc<dyn CheckpointStore> = store.clone();
    let cancel = CancellationToken::new();
    let config = CheckpointLoopConfig {
        tick: Duration::from_millis(50),
        interval: Duration::ZERO,
        quiesce: Duration::ZERO,
        retain_generations: 2,
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
    let handle = spawn_checkpoint_loop(
        tv.client().clone(),
        tv.key_schema().clone(),
        store_for_loop,
        config,
        None,
        cancel.clone(),
    );

    let cursor = wait_for_cursor_above(&tv, oid, 0, Duration::from_secs(20)).await;
    cancel_and_join(cancel, handle).await;
    assert!(
        cursor.is_some(),
        "an idle layer with both gates open must be checkpointed within 20 s; \
         without a landed generation this case cannot observe the success arm \
         at all"
    );

    assert_eq!(
        read_header_u64(&tv, oid, "restore_seq_bump").await,
        Some(0),
        "b.vnw: the success arm MUST clear restore_seq_bump to 0. A surviving \
         {PLANTED_RESTORE_BUMP} means every future assembly of this layer \
         subtracts a band that is already behind last_checkpoint_seq, \
         deflating observed_commit_span by that much and under-sizing the \
         guard band on the next DR restore"
    );
    // Same-HSET claim: the column that was cleared and the columns that were
    // advanced must all be present together. `restore_seq_bump` reading 0 with
    // NO cadence stamp beside it would mean the clear escaped into a separate
    // write, where a crash between the two can strand it.
    assert!(
        read_header_u64(&tv, oid, "last_checkpoint_ms")
            .await
            .is_some(),
        "the cadence stamp must have landed alongside the cleared bump — all \
         three cursor columns ride one HSET on the success arm"
    );

    // The generation that ran the correction must record the saturated 0
    // rather than a wrapped u64.
    let handles = store
        .list_layer_checkpoints(oid)
        .await
        .expect("list must succeed");
    assert_eq!(
        handles.len(),
        1,
        "the skip-if-no-commits gate must close after the first success, so \
         exactly one generation may exist; got {}",
        handles.len()
    );
    let blob = store
        .get_latest_layer_checkpoint(oid)
        .await
        .expect("get must succeed")
        .expect("a generation must exist");
    assert_eq!(
        blob.meta.observed_commit_span, 0,
        "a bump larger than snap_seq must saturate the span to 0 — the \
         'no measurement available' sentinel that folds the next restore's \
         band back to the configured margin. A wrapped value here would be \
         read as an astronomically large band and trip the F64_SAFE_INT_MAX \
         refusal on restore, leaving the layer unrecoverable"
    );
}
