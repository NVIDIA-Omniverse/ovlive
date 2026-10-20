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

// Task t2.fgj.ya.9j Subtask .by — metrics: writes_total{result, kind}
// + the pending-age histogram/gauges + label cardinality (SDD SR-2.2, b.jq3).
//
// Two metrics are load-bearing for the leader-elected periodic checkpoint
// loop:
//
//   - `live_edit_checkpoint_writes_total{result, kind}` — incremented on
//     every per-layer tick outcome. `result ∈ {success, error, skipped}`,
//     `kind = "layer"` in Epic 1. The pre-init in `Metrics::new` materializes
//     all three permutations at startup so a Prometheus `rate(... [5m])`
//     alert sees a continuous derivative from boot. Operators key on
//     `result="error"` for "adapter failing" alerts.
//
//   - `live_edit_checkpoint_pending_age_seconds` — histogram of
//     `now - last_checkpoint_ms` per layer at tick time, sampled ONLY for
//     layers whose newest durable generation does not carry every commit they
//     have. Buckets at `[5, 15, 30, 60, 120, 300, 600, 1800, 3600, 7200]`.
//     p99 above `LIVE_EDIT_CHECKPOINT_INTERVAL_SECS + tick + assembly` means
//     the loop fell behind.
//
//     The pending-scoping is the b.jq3 fix and the thing most of the cases
//     below exist to pin. The predecessor metric
//     (`live_edit_checkpoint_lag_seconds`) measured time since the last
//     COMMIT on the skip and success arms, so an idle, fully-backed-up layer's
//     value climbed without bound on a perfectly healthy fleet — a rising p99
//     meant "traffic stopped", and the documented `> 300s` alert false-fired
//     on any quiet fleet.
//
//   - `live_edit_checkpoint_oldest_pending_age_seconds` and
//     `live_edit_checkpoint_layers_never_written` — per-tick aggregate gauges
//     by `service`, published once per completed tick. A p99 over thousands of
//     layers structurally cannot see the ONE layer wedged behind a poison
//     payload; the max gauge can. Non-leader replicas publish zeros (an OTel
//     gauge never decays, so a demoted leader would otherwise pin the fleet's
//     `max()` high forever); a FAILED leader probe publishes nothing, because
//     unknown is not healthy.
//
// Sibling tests (.s2 end-to-end, .hd retention, .pi/.ws skip-gates) exercise
// success / skipped paths incidentally. This file adds the EXPLICIT error
// path, the EXPLICIT pending-age bucket + gauge assertions, and the label
// cardinality guard so a regression that drops the `inc_*` call on the
// error path, hardcodes `kind="foo"`, or accidentally introduces a new
// `kind` value is caught by a dedicated test rather than slipping through
// behind the success-path tests.
//
// DELIBERATELY NOT COVERED, here or anywhere — five arms of the loop with no
// test seam, listed together so their absence reads as a decision rather than
// an oversight. Each is restated at the site it belongs to:
//
//   - leader-probe `Err(_)` — publishes nothing (Case J's banner).
//   - cancellation mid-SCAN — publishes nothing.
//   - truncated SCAN — publishes nothing; needs a fred scanner faulted
//     mid-stream, and `test-checkpoint-fault-injection` fault-injects the
//     ADAPTER, not the Valkey client.
//   - cursor-HSET failure after a successful `put` — records as pending;
//     needs the HSET to fail while the adapter call succeeded.
//   - assembler `Err(NotFound)` — records nothing (the layer is gone, so
//     there is no DR obligation left); needs a won delete/assemble race.
//
// The last four are enumerated with their reasoning on `pending_age_tests` in
// `src/checkpoint/checkpointer.rs`, which is where the accumulator's
// observation map lives.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use bytes::Bytes;
use fred::interfaces::HashesInterface;
use tokio::time::{sleep, timeout, Instant};
use tokio_util::sync::CancellationToken;

use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
use live_edit::checkpoint::{
    spawn_checkpoint_loop, CheckpointBlob, CheckpointError, CheckpointHandle, CheckpointLoopConfig,
    CheckpointMeta, CheckpointStore, TailCaptureConfig,
};
use live_edit::domain::types::ObjectState;
use live_edit::observability::metrics::{Metrics, SharedMetrics};
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::relay::RelayedUpdates;
use live_edit::storage::test_support::TestValkey;

#[cfg(feature = "test-checkpoint-fault-injection")]
use crate::checkpoint::helpers::unprovable_cut_client;

/// The tail knobs every `CheckpointLoopConfig` in this file runs under: capture
/// pinned explicitly OFF, every other knob at its shipped value.
///
/// Pinned rather than inherited from [`TailCaptureConfig::default`], whose
/// `enabled` became `true` on 2026-09-11 (b.3n1). This file's subject is the
/// checkpoint loop's METRIC SURFACE — the exact `{result} × {kind}` label set on
/// `writes_total`, the stable-cut vocabulary, and (Case C) the claim that every
/// `tail_total{outcome}` cell exists at zero and stays there. Every one of those
/// is a statement about a loop whose tail path never executed, so the flag that
/// makes it never execute is part of the fixture, not something to track.
fn tails_off() -> TailCaptureConfig {
    TailCaptureConfig {
        enabled: false,
        ..TailCaptureConfig::default()
    }
}

// ────────────────────────────────────────────────────────────────────────
// Erroring-on-demand `CheckpointStore` wrapper.
//
// Adapts the `CountingCheckpointStore` pattern from
// `checkpointer_leader_race_test.rs`: every non-target trait method
// delegates to an inner `InMemoryCheckpointStore`, but
// `put_layer_checkpoint` consults an `AtomicBool` "fail" flag and returns
// `CheckpointError::Transient` while it is set. Test Case A flips the flag
// off after observing one error increment so the next tick can succeed
// normally — pinning the loop's "errors do NOT abort the loop" contract.
// ────────────────────────────────────────────────────────────────────────

struct ErroringCheckpointStore {
    inner: InMemoryCheckpointStore,
    fail_put_layer: AtomicBool,
    put_layer_attempts: AtomicUsize,
}

impl ErroringCheckpointStore {
    fn new() -> Self {
        Self {
            inner: InMemoryCheckpointStore::new(),
            fail_put_layer: AtomicBool::new(true),
            put_layer_attempts: AtomicUsize::new(0),
        }
    }

    fn stop_failing(&self) {
        self.fail_put_layer.store(false, Ordering::SeqCst);
    }

    fn put_layer_attempt_count(&self) -> usize {
        self.put_layer_attempts.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl CheckpointStore for ErroringCheckpointStore {
    async fn put_layer_checkpoint(
        &self,
        object_id: u64,
        snap_seq: u64,
        ompf: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        self.put_layer_attempts.fetch_add(1, Ordering::SeqCst);
        if self.fail_put_layer.load(Ordering::SeqCst) {
            return Err(CheckpointError::Transient {
                source: "test-injected transient adapter failure".to_string().into(),
            });
        }
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

// ────────────────────────────────────────────────────────────────────────
// Small helpers shared across cases.
// ────────────────────────────────────────────────────────────────────────

fn make_metrics() -> SharedMetrics {
    Arc::new(crate::common::metrics_scrape_only(
        "checkpointer-metrics-test",
    ))
}

/// Read the current value of `live_edit_checkpoint_writes_total{result, kind="layer"}`.
fn read_writes_counter(metrics: &Metrics, result: &str) -> u64 {
    live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(metrics),
        "live_edit_checkpoint_writes_total",
        &[("result", result), ("kind", "layer")],
    )
    .unwrap_or(0.0) as u64
}

/// Read the current sample count on
/// `live_edit_checkpoint_pending_age_seconds`.
fn read_pending_age_sample_count(metrics: &Metrics) -> u64 {
    live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(metrics),
        "live_edit_checkpoint_pending_age_seconds_count",
        &[],
    )
    .unwrap_or(0.0) as u64
}

/// Read `live_edit_checkpoint_oldest_pending_age_seconds{service}`. `None`
/// when the gauge has never been published — an OTel gauge emits no family
/// until its first `record()`, and the `None` / `Some(0.0)` distinction
/// separates "this replica reported a clean tick" from "this replica reported
/// nothing at all", which is exactly the contract on the leader-probe `Err`
/// arm.
fn read_oldest_pending_age(metrics: &Metrics, service: &str) -> Option<f64> {
    live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(metrics),
        "live_edit_checkpoint_oldest_pending_age_seconds",
        &[("service", service)],
    )
}

/// Read `live_edit_checkpoint_layers_never_written{service}`. `None` semantics
/// as for [`read_oldest_pending_age`].
fn read_layers_never_written(metrics: &Metrics, service: &str) -> Option<u64> {
    live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(metrics),
        "live_edit_checkpoint_layers_never_written",
        &[("service", service)],
    )
    .map(|v| v as u64)
}

/// Parse the `live_edit_checkpoint_pending_age_seconds` histogram from
/// rendered text into `(le upper-bound, cumulative count)` pairs (finite
/// buckets only, in wire order) plus the `_count` / `_sum` aggregates.
fn pending_age_histogram(metrics: &Metrics) -> (Vec<(f64, u64)>, u64, f64) {
    let text = live_edit::observability::metrics::test_util::render(metrics);
    let mut buckets = Vec::new();
    for line in text.lines() {
        let Some(rest) = line.strip_prefix("live_edit_checkpoint_pending_age_seconds_bucket")
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
        if &after[..end] == "+Inf" {
            continue;
        }
        let Ok(bound) = after[..end].parse::<f64>() else {
            continue;
        };
        let val = line
            .rsplit_once(char::is_whitespace)
            .and_then(|(_, v)| v.trim().parse::<u64>().ok())
            .unwrap_or(0);
        buckets.push((bound, val));
    }
    let count = live_edit::observability::metrics::test_util::sample_value(
        &text,
        "live_edit_checkpoint_pending_age_seconds_count",
        &[],
    )
    .unwrap_or(0.0) as u64;
    let sum = live_edit::observability::metrics::test_util::sample_value(
        &text,
        "live_edit_checkpoint_pending_age_seconds_sum",
        &[],
    )
    .unwrap_or(0.0);
    (buckets, count, sum)
}

/// Seed one populated layer so the loop has work to do. Returns the
/// `object_id`. Mirrors the helpers in the sibling retention / leader-race
/// tests so each metric-focused test pays the same setup cost without
/// pulling in their full assertion machinery.
async fn seed_layer(tv: &TestValkey, layer_name: &str) -> u64 {
    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = session_store
        .create("checkpoint-metrics-session", "/stage", "metrics-test-user")
        .await
        .expect("session creation must succeed");

    let layer_store = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
    let state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usda".to_string(),
        ..Default::default()
    };
    layer_store
        .create_layer(
            &session_id,
            layer_name,
            "metrics-test-user",
            &state,
            Arc::new(MockExternalValueStore::new()),
        )
        .await
        .expect("create_layer must succeed")
}

/// Current wall-clock time as milliseconds since the Unix epoch. Matches the
/// production helper in `src/checkpoint/checkpointer.rs`.
fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

/// Pin deterministic clock ages into the loop's gate logic without waiting
/// real time, by writing the two `obj:hdr` columns the gate actually reads.
///
/// The two ages are independent knobs and the tests below lean on the
/// distinction: `commit_age_ms` drives the quiesce arm, `checkpoint_age_ms`
/// drives the interval arm AND is the sole input to the pending-age histogram
/// (b.jq3). Several cases set them to deliberately different values so a
/// sample's bucket proves which column it came from.
///
///   - `last_server_ts = now - commit_age_ms`, in **MICROSECONDS** (that is
///     the unit every production commit path writes, via
///     `domain::time::now_micros`; the loop converts µs → ms). Writing
///     milliseconds here would backdate the layer to ~1970 and make every
///     gate arm pass — the b.yut defect shape.
///   - `last_checkpoint_ms = now - checkpoint_age_ms`, in milliseconds,
///     matching what the loop stamps on its own success arm. Without this
///     the interval arm sees "never checkpointed" and fires regardless of
///     how recent the commit was, so a test that wants the SKIP path must
///     set it.
///
/// This deliberately drives the real surface rather than fabricating a
/// wall-clock-shaped stream entry ID: production XADDs an explicit
/// `{sequence_num}-0` ID, so no such entry ever exists in a live system.
async fn set_gate_timestamps(
    tv: &TestValkey,
    object_id: u64,
    commit_age_ms: u64,
    checkpoint_age_ms: u64,
) {
    let header_key = tv.key_schema().obj_header(&object_id.to_string());
    let now_ms = now_unix_ms();
    let last_server_ts_micros = now_ms.saturating_sub(commit_age_ms) * 1_000;
    let last_checkpoint_ms = now_ms.saturating_sub(checkpoint_age_ms);
    let _: () = tv
        .client()
        .pool
        .hset(
            &header_key,
            vec![
                ("last_server_ts", last_server_ts_micros.to_string()),
                ("last_checkpoint_ms", last_checkpoint_ms.to_string()),
            ],
        )
        .await
        .expect("HSET of the gate timestamp columns must succeed");
}

/// Bounded-poll an async predicate. Mirrors the helper in
/// `checkpointer_leader_race_test.rs` — every condition wait sits behind a
/// deadline so a regression that wedges the loop surfaces as a deterministic
/// timeout, not a 60 s hang.
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

/// Cancel the loop and bounded-join the handle. Pulled into a helper so each
/// test exits the same way; a 5 s join budget matches the sibling tests'
/// shutdown discipline.
async fn cancel_and_join(cancel: CancellationToken, handle: tokio::task::JoinHandle<()>) {
    cancel.cancel();
    timeout(Duration::from_secs(5), handle)
        .await
        .expect("checkpoint loop must exit within 5 s of cancellation")
        .expect("checkpoint loop task must not panic on shutdown");
}

// ────────────────────────────────────────────────────────────────────────
// Case A — error path increments `result="error"` and the loop survives.
//
// Drives the assembler-failure branch of the loop (line 528 in
// `src/checkpoint/checkpointer.rs`). Seeds a real layer, points the loop
// at an `ErroringCheckpointStore` whose `put_layer_checkpoint` returns
// `Transient`, runs ticks, and asserts:
//
//   1. The `result="error"` counter increments by at least 1.
//   2. `last_checkpoint_seq` on `obj:hdr` stays unchanged at 0 (the cursor
//      HSET only fires AFTER a successful assembler return).
//   3. The loop did NOT panic — verified by flipping the fail-flag off
//      after the first observed error and watching `result="success"` go
//      up. If the loop had aborted on the first error, that wouldn't happen.
//
// A regression dropping the `inc_checkpoint_write("error", "layer")` call
// from the assembler-Err arm makes assertion 1 fail.
// ────────────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn case_a_error_path_increments_error_counter_and_loop_survives() {
    let tv = TestValkey::setup_with_pool_size(8).await;
    let metrics = make_metrics();

    // Seed a real layer so the loop's SCAN finds work, then advance past
    // the create+immediate-checkpoint race by waiting until obj:hdr exists.
    let oid = seed_layer(&tv, "err-layer").await;

    // Plumb the erroring store into the loop. interval=0, quiesce=0 means
    // both skip-gates pass immediately, so every tick reaches the assembler
    // (which calls our `put_layer_checkpoint` and gets Transient back).
    // tick=50ms gives ~20 ticks/sec of observation density.
    let store: Arc<ErroringCheckpointStore> = Arc::new(ErroringCheckpointStore::new());
    let store_for_loop: Arc<dyn CheckpointStore> = store.clone();
    let cancel = CancellationToken::new();
    let config = CheckpointLoopConfig {
        tick: Duration::from_millis(50),
        interval: Duration::ZERO,
        quiesce: Duration::ZERO,
        retain_generations: 2,
        // Epic 6 counters-sync cadence; pushed well above the test runtime
        // so the counters-sidecar timer does not fire during this test.
        counters_sync: Duration::from_secs(3600),
        reconcile_interval: Duration::from_secs(3600),
        leader: crate::common::test_leader("case-a-replica"),
        tail: tails_off(),
    };
    let handle = spawn_checkpoint_loop(
        tv.client().clone(),
        tv.key_schema().clone(),
        store_for_loop,
        config,
        Some(metrics.clone()),
        cancel.clone(),
    );

    // Wait for the first error increment (≤5 s gives ≥80 ticks). The loop's
    // observed `result="error"` increment is the load-bearing signal Case A
    // is built around.
    let error_before = read_writes_counter(&metrics, "error");
    let metrics_for_poll = metrics.clone();
    let saw_error = poll_until(Duration::from_secs(5), || {
        let m = metrics_for_poll.clone();
        async move { read_writes_counter(&m, "error") > error_before }
    })
    .await;
    assert!(
        saw_error,
        "result=\"error\" counter must increment within 5 s of the first \
         erroring tick; pre-SR-2.2 the assembler-Err arm of the loop \
         either swallowed the error silently or bumped the wrong label",
    );

    // The cursor MUST stay unchanged. The HSET of `last_checkpoint_seq`
    // happens only on the assembler-Ok path; an erroring tick advancing the
    // cursor would mean the next tick mistakenly thinks the layer is
    // checkpointed and skips it forever (silent data-loss path).
    let header_key = format!("{}{{{oid}}}:obj:hdr", tv.key_schema().prefix());
    let cursor: Option<String> = tv
        .client()
        .pool
        .hget(&header_key, "last_checkpoint_seq")
        .await
        .expect("HGET last_checkpoint_seq must succeed");
    assert!(
        cursor.is_none() || cursor.as_deref() == Some("0"),
        "last_checkpoint_seq must remain absent/0 across erroring ticks; \
         got {cursor:?} — the cursor HSET ran on a path where the put failed, \
         which would silently mask future commits as already-checkpointed",
    );

    let attempts_during_failure = store.put_layer_attempt_count();
    assert!(
        attempts_during_failure >= 1,
        "the loop must have called put_layer_checkpoint at least once before \
         we flip the fail-flag; got {attempts_during_failure}"
    );

    // Loop-survival proof: flip the adapter from failing to succeeding and
    // wait for a `result="success"` increment. If the first error had
    // panicked the spawned task, the JoinHandle would be done and the
    // success counter would never move.
    store.stop_failing();
    let success_before = read_writes_counter(&metrics, "success");
    let metrics_for_success_poll = metrics.clone();
    let saw_success = poll_until(Duration::from_secs(5), || {
        let m = metrics_for_success_poll.clone();
        async move { read_writes_counter(&m, "success") > success_before }
    })
    .await;
    assert!(
        saw_success,
        "after flipping the adapter to succeed, the loop must record at \
         least one result=\"success\" tick within 5 s; if the loop had \
         panicked / aborted on the earlier error, the JoinHandle would be \
         done and the success counter would never advance"
    );

    // Final-state error count must be at least 1 (the original observed
    // increment) AND the success count must be at least 1 (post-flip).
    let final_error = read_writes_counter(&metrics, "error");
    let final_success = read_writes_counter(&metrics, "success");
    assert!(
        final_error >= 1,
        "final error count must be >= 1, got {final_error}"
    );
    assert!(
        final_success >= 1,
        "final success count must be >= 1 (post-flip ticks succeeded), got {final_success}"
    );

    cancel_and_join(cancel, handle).await;
}

// ────────────────────────────────────────────────────────────────────────
// Case B — pending-age histogram observed in the ~7 s bucket when a skip-gate
//          fires.
//
// Drives the skip branch of the quiesce/interval gate — the arm that carries
// a healthy fleet's entire steady-state distribution. The layer's `obj:hdr`
// commit stamp is set to "just now" and its CHECKPOINT stamp backdated ~7 s,
// then the loop runs with quiesce=60s + interval=60s so neither arm fires:
// ~0 s since commit < 60 s quiesce, and 7 s since checkpoint < 60 s interval.
// The skip path observes `pending_age ≈ 7.0` into the histogram.
//
// The two stamps are deliberately the OPPOSITE way round from the fixture
// this case used before b.jq3 (commit 7 s ago, checkpoint now). That swap is
// the point: the histogram is `now - last_checkpoint_ms`, not
// `now - last_commit_ms`, so a loop still reading the commit clock produces
// ~0 here and fails the `<=5` assertion below. The layer's `seq` (1) is ahead
// of `last_checkpoint_seq` (absent → 0), so it is genuinely pending and in
// scope for the histogram at all.
//
// Bucket boundaries are `[5, 15, 30, 60, 120, 300, 600, 1800, 3600, 7200]`.
// A 7 s sample lands in the `<=15` bucket (cumulative_count ≥ 1) but NOT
// in the `<=5` bucket. We assert bucket MEMBERSHIP, not an exact float
// (per the subtask AC).
// ────────────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn case_b_pending_age_histogram_observed_for_skipped_layer() {
    let tv = TestValkey::setup_with_pool_size(8).await;
    let metrics = make_metrics();

    let oid = seed_layer(&tv, "pending-age-layer").await;

    // Fresh commit stamp, checkpoint stamp backdated ~7 s: the loop computes
    // `pending_age ≈ 7.0` while both gate arms stay closed.
    set_gate_timestamps(&tv, oid, 0, 7_000).await;

    // Force the skip path. interval=60s + quiesce=60s ensures
    // `now - last_commit (~0 s) < quiesce (60 s)` AND
    // `now - last_checkpoint (~7 s) < interval (60 s)`, so the gate skips on
    // every tick — and every tick samples the pending age.
    let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
    let cancel = CancellationToken::new();
    let config = CheckpointLoopConfig {
        tick: Duration::from_millis(50),
        interval: Duration::from_secs(60),
        quiesce: Duration::from_secs(60),
        retain_generations: 2,
        // Epic 6 counters-sync cadence; pushed well above the test runtime
        // so the counters-sidecar timer does not fire during this test.
        counters_sync: Duration::from_secs(3600),
        reconcile_interval: Duration::from_secs(3600),
        leader: crate::common::test_leader("case-b-replica"),
        tail: tails_off(),
    };
    let handle = spawn_checkpoint_loop(
        tv.client().clone(),
        tv.key_schema().clone(),
        store,
        config,
        Some(metrics.clone()),
        cancel.clone(),
    );

    // Wait for the histogram to record at least one observation.
    let metrics_for_poll = metrics.clone();
    let saw_observation = poll_until(Duration::from_secs(5), || {
        let m = metrics_for_poll.clone();
        async move { read_pending_age_sample_count(&m) >= 1 }
    })
    .await;
    assert!(
        saw_observation,
        "the pending-age histogram must receive at least one observation \
         within 5 s; either the skip-gate did not fire or the loop did not \
         sample pending age on the gate's Skip arm — the arm that carries a \
         healthy fleet's whole steady-state distribution",
    );

    // Skipped counter must have moved alongside the histogram — they are
    // observed together on the skip path in `run_one_tick`.
    let skipped_count = read_writes_counter(&metrics, "skipped");
    assert!(
        skipped_count >= 1,
        "result=\"skipped\" counter must have incremented when the \
         quiesce/interval gate skipped the layer; got {skipped_count}"
    );

    // Stop the loop BEFORE inspecting the histogram so a late tick can't
    // slip an extra sample between the gather and the bucket assertion.
    cancel_and_join(cancel, handle).await;

    // The retired family must not be on the wire. `live_edit_checkpoint_lag_
    // seconds` was REPLACED by the pending-age histogram, not supplemented by
    // it: a loop that fed both would export two series measuring different
    // quantities under names an operator reads as synonyms, and the old one
    // carries exactly the quiet-fleet false positive b.jq3 removed. An OTel
    // family renders only once something records into it, so this fires
    // precisely on a regression that restores the old observation call.
    let text = live_edit::observability::metrics::test_util::render(&metrics);
    assert!(
        !text.contains("live_edit_checkpoint_lag_seconds"),
        "the pre-b.jq3 lag histogram must be gone from the scrape, not \
         double-written alongside its replacement"
    );

    // Bucket-membership assertion. Inspect the histogram via the rendered
    // /metrics text so we read the same wire shape Prometheus scrapes —
    // guards against a regression that changes the metric's internal storage
    // but leaves a getter API working.
    let (buckets, sample_count, sample_sum) = pending_age_histogram(&metrics);

    // Sanity: bucket boundaries must match SDD SR-2.2.
    let expected_bounds: &[f64] = &[
        5.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0, 3600.0, 7200.0,
    ];
    assert_eq!(
        buckets.len(),
        expected_bounds.len(),
        "the pending-age histogram must have exactly {} explicit buckets per \
         SDD SR-2.2; got {}",
        expected_bounds.len(),
        buckets.len()
    );
    for ((upper, _), expected) in buckets.iter().zip(expected_bounds) {
        assert!(
            (upper - expected).abs() < f64::EPSILON,
            "pending-age histogram bucket boundaries must match SDD SR-2.2 \
             ({expected_bounds:?}); got upper_bound {upper} where {expected} was expected",
        );
    }

    // ~7 s sample → cumulative_count of the `<=5` bucket must NOT include
    // it, while the `<=15` bucket MUST include it. This is the load-bearing
    // bucket-membership assertion: a regression that fed the histogram a
    // wrong unit (e.g. milliseconds vs seconds) would land in the wrong
    // bucket and fail one of these two checks.
    let le5_cum = buckets[0].1;
    let le15_cum = buckets[1].1;
    assert_eq!(
        le5_cum, 0,
        "no pending-age sample should fall into the <=5s bucket — the \
         injected CHECKPOINT stamp is ~7s in the past so the smallest expected \
         bucket is <=15s (cumulative count of <=5s bucket = {le5_cum}). A \
         sample here means the loop measured the commit clock, which this \
         fixture stamps at ~now, instead of last_checkpoint_ms"
    );
    assert!(
        le15_cum >= 1,
        "the ~7s pending-age sample must land in the <=15s cumulative bucket; \
         cumulative_count for <=15s was {le15_cum}, expected >= 1"
    );

    // Belt-and-braces: sample_sum / sample_count ≈ 7s (we don't assert
    // exact equality, just that the mean sits between the <=5s and <=15s
    // boundaries — same bucket-membership guarantee at the aggregate level).
    assert!(sample_count >= 1, "sample_count must be >= 1");
    let mean = sample_sum / sample_count as f64;
    assert!(
        (5.0..=15.0).contains(&mean),
        "mean pending-age observation must fall in (5s, 15s] — bucket \
         membership at the aggregate level; got mean = {mean:.3}s over \
         {sample_count} samples"
    );

    // The per-tick aggregate agrees with the distribution. One pending layer,
    // so the max IS that layer's age; and it has a durable generation, so the
    // never-written gauge must stay at zero.
    let oldest = read_oldest_pending_age(&metrics, "case-b-replica")
        .expect("the leader must publish the oldest-pending-age gauge each tick");
    assert!(
        (5.0..=15.0).contains(&oldest),
        "the per-tick maximum over a single ~7s-pending layer must be that \
         layer's own age; got {oldest}s"
    );
    assert_eq!(
        read_layers_never_written(&metrics, "case-b-replica"),
        Some(0),
        "this layer HAS a durable generation (last_checkpoint_ms is stamped \
         ~7s ago), so it belongs on the histogram, not on the never-written \
         gauge — the two populations must not overlap"
    );
}

// ────────────────────────────────────────────────────────────────────────
// Case C — label cardinality: exactly {success, error, skipped} ×
// {layer, session, counters_sidecar, layer_tail}.
//
// `Metrics::new` pre-initialises every Epic-1/2/6 cell so a fresh registry
// already exposes all permutations at zero. This test gathers the
// counter family and asserts the EXACT label set — guarding two regression
// shapes simultaneously:
//
//   1. Adding a stray `result` value (e.g. `result="aborted"`) without
//      updating the SDD + pre-init loop.
//   2. Hardcoding a wrong `kind` label (e.g. `kind="checkpoint"`,
//      `kind="foo"`) inside the loop body so the per-tick increment opens
//      a NEW series alongside the pre-init zero series and dashboards
//      key off the wrong slice.
//
// Epic 1 introduced `kind="layer"`; Epic 2 (`t1.fgj.7v`) added
// `kind="session"` for the session-checkpoint write path; Epic 6
// (`t1.fgj.mh` / SDD SR-5.1) added `kind="counters_sidecar"` for the
// periodic counters-sync tick; the tail-generation work (SDD SR-8.2)
// added `kind="layer_tail"` for the stream-replay generation that extends
// a lineage without rebuilding full state — a re-base deliberately does
// NOT get its own kind, because what a re-base writes is a base and so
// counts as `kind="layer"`. The pre-init loop in `Metrics::new`
// materialises all four kinds at zero.
//
// We then drive one of each `result` value through the loop end-to-end
// (using the helpers from Cases A + B) and re-gather. The final assertion
// is that the gathered label set is STILL exactly the pre-init shape —
// no new `kind` ever appeared.
//
// The test closes with the flag-off statement for the tail feature: with
// `tails_off()` (capture pinned disabled) driving all three phases, every
// `live_edit_checkpoint_tail_total{outcome}` cell must read exactly 0 —
// pre-initialised, present, and unmoved.
// ────────────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn case_c_label_cardinality_is_bounded() {
    let metrics = make_metrics();

    // Initial-state check: BEFORE any tick runs, all Epic-1/2/6 cells
    // must exist at zero. This pins the `Metrics::new` pre-init contract
    // (the `for result × for kind` nested loop in src/observability/metrics.rs)
    // — a regression that dropped the pre-init loop would leave the family
    // empty until the first tick, and a regression that dropped a `kind`
    // from the inner list would leave dashboards keyed off a missing series.
    //
    // Both vocabularies below are spelled out as LITERAL arrays on purpose,
    // and must stay that way: importing the production consts
    // (`CHECKPOINT_WRITE_KINDS` / the `result` list) would make both sides of
    // this comparison derive from one definition, so adding a fifth kind
    // would seed the new cell AND grow `expected` in lockstep and the
    // assertion would pass by construction. The guard's whole mechanism is
    // that `expected` restates the vocabulary INDEPENDENTLY while
    // `gather_writes_label_pairs` reads what `Metrics::new` actually seeded —
    // a widening reaches this file only through a deliberate human edit here.
    let initial_pairs = gather_writes_label_pairs(&metrics);
    let expected: HashSet<(String, String)> = ["success", "error", "skipped"]
        .iter()
        .flat_map(|r| {
            ["layer", "session", "counters_sidecar", "layer_tail"]
                .iter()
                .map(move |k| ((*r).to_string(), (*k).to_string()))
        })
        .collect();
    assert_eq!(
        initial_pairs, expected,
        "pre-init label set on live_edit_checkpoint_writes_total must be \
         exactly {{success, error, skipped}} × \
         {{layer, session, counters_sidecar, layer_tail}}; \
         got {initial_pairs:?}"
    );

    // Drive the loop through ALL THREE result paths so any
    // post-init label-set drift surfaces:
    //   - Case A's `ErroringCheckpointStore` → `result="error"`.
    //   - Case B's quiesce-gate firing → `result="skipped"`.
    //   - Stop-failing + ordinary tick → `result="success"`.
    let tv = TestValkey::setup_with_pool_size(8).await;
    let oid = seed_layer(&tv, "cardinality-layer").await;

    // Backdate the gate timestamps so the skip path can fire while the layer
    // is otherwise eligible (seq=1 > last_checkpoint_seq=0). Same shape as
    // Case B: commit ~7 s ago, checkpoint stamped "now".
    set_gate_timestamps(&tv, oid, 7_000, 0).await;

    // Drive the skipped → error → success sequence inside ONE spawned loop
    // so the `le:checkpoint:leader` Valkey lock (30 s TTL) stays held by the
    // same replica throughout. Cancelling a sub-loop and spawning a fresh
    // one with a different replica id would hit the lock-still-held window
    // and the second loop would sit idle until the TTL expired.
    //
    // Initial config: quiesce=60s + interval=60s so the per-layer gate
    // skips immediately (commit ~7 s ago < 60 s, checkpoint stamp ~now
    // < 60 s). After the skipped counter ticks, we want the gate to open —
    // but cargo doesn't let us mutate the config in a running loop. Instead
    // we time the phases:
    //
    //   phase 1: quiesce=60s + interval=60s, adapter failing → only skipped
    //            moves.
    //   phase 2: cancel + respawn with SAME replica_id + quiesce=0 +
    //            interval=0, adapter still failing → error moves.
    //   phase 3: stop-failing → success moves on the next tick.
    //
    // The same-replica-id refresh path (`LEADER_ACQUIRE_OR_REFRESH_SCRIPT`,
    // shared with the GC + presence sweepers) reclaims the lock on respawn
    // because the value-matches branch refreshes the TTL rather than
    // failing on a competing claim.
    let replica_id = "case-c-replica".to_string();
    let store: Arc<ErroringCheckpointStore> = Arc::new(ErroringCheckpointStore::new());

    // Phase 1: skipped path.
    let store_for_loop1: Arc<dyn CheckpointStore> = store.clone();
    let cancel_1 = CancellationToken::new();
    let config_phase1 = CheckpointLoopConfig {
        tick: Duration::from_millis(50),
        interval: Duration::from_secs(60),
        quiesce: Duration::from_secs(60),
        retain_generations: 2,
        // Epic 6 counters-sync cadence; pushed well above the test runtime
        // so the counters-sidecar timer does not fire during this test.
        counters_sync: Duration::from_secs(3600),
        reconcile_interval: Duration::from_secs(3600),
        leader: crate::common::test_leader(&replica_id),
        tail: tails_off(),
    };
    let handle_1 = spawn_checkpoint_loop(
        tv.client().clone(),
        tv.key_schema().clone(),
        store_for_loop1,
        config_phase1,
        Some(metrics.clone()),
        cancel_1.clone(),
    );
    let metrics_for_skipped = metrics.clone();
    let saw_skipped = poll_until(Duration::from_secs(10), || {
        let m = metrics_for_skipped.clone();
        async move { read_writes_counter(&m, "skipped") >= 1 }
    })
    .await;
    assert!(saw_skipped, "result=\"skipped\" must increment within 10s");
    cancel_and_join(cancel_1, handle_1).await;

    // Phase 2: same replica_id reclaims the leader lock; quiesce=0 lets
    // every tick reach the (still-failing) assembler.
    let store_for_loop2: Arc<dyn CheckpointStore> = store.clone();
    let cancel_2 = CancellationToken::new();
    let config_phase2 = CheckpointLoopConfig {
        tick: Duration::from_millis(50),
        interval: Duration::ZERO,
        quiesce: Duration::ZERO,
        retain_generations: 2,
        // Epic 6 counters-sync cadence; pushed well above the test runtime
        // so the counters-sidecar timer does not fire during this test.
        counters_sync: Duration::from_secs(3600),
        reconcile_interval: Duration::from_secs(3600),
        leader: crate::common::test_leader(&replica_id),
        tail: tails_off(),
    };
    let handle_2 = spawn_checkpoint_loop(
        tv.client().clone(),
        tv.key_schema().clone(),
        store_for_loop2,
        config_phase2,
        Some(metrics.clone()),
        cancel_2.clone(),
    );
    let metrics_for_error = metrics.clone();
    let saw_error = poll_until(Duration::from_secs(10), || {
        let m = metrics_for_error.clone();
        async move { read_writes_counter(&m, "error") >= 1 }
    })
    .await;
    assert!(saw_error, "result=\"error\" must increment within 10s");

    // Phase 3: stop the adapter from failing inside the SAME spawned loop
    // so the next eligible tick lands a success increment without needing
    // another respawn (and another leader-lock refresh).
    store.stop_failing();
    let metrics_for_success = metrics.clone();
    let saw_success = poll_until(Duration::from_secs(10), || {
        let m = metrics_for_success.clone();
        async move { read_writes_counter(&m, "success") >= 1 }
    })
    .await;
    assert!(saw_success, "result=\"success\" must increment within 10s");
    cancel_and_join(cancel_2, handle_2).await;

    // After driving every cell at least once, re-gather and assert the
    // label set is STILL exactly the pre-init shape. A regression that
    // hardcoded `kind="foo"` or introduced an out-of-band `result` value
    // would surface here as an extra pair in `final_pairs`.
    let final_pairs = gather_writes_label_pairs(&metrics);
    assert_eq!(
        final_pairs, expected,
        "post-tick label set on live_edit_checkpoint_writes_total must \
         still be exactly {{success, error, skipped}} × \
         {{layer, session, counters_sidecar, layer_tail}}; \
         got {final_pairs:?} — a regression hardcoded a different `kind` \
         label or introduced an out-of-band `result` value"
    );

    // Flag-off zero-movement: every phase above ran with `tails_off()`, i.e.
    // tail capture DISABLED — pinned there explicitly, since the binary
    // default is `true` (b.3n1). The strongest statement available at this
    // tier that "the operator opt-out changes nothing" is that the tail
    // counter's cells all exist (pre-init contract) and all still read
    // exactly 0 — no outcome, not even a `refused_*` one, moved. A refusal
    // tick here would mean the capture path ran and declined, which is a
    // different (and wrong) story from never having run.
    //
    // Same independent-restatement discipline as `expected` above: the
    // outcome vocabulary is a LITERAL array, deliberately not an import of
    // `CHECKPOINT_TAIL_OUTCOMES`, so a widening of the production const
    // cannot silently widen what this test claims to have checked.
    //
    // `live_edit_checkpoint_tail_total` is outcome-labelled ONLY (no
    // `service` label), so an outcome-only match on the rendered text is an
    // EXACT match on one series, not a subset match that could latch onto a
    // differently-labelled sibling.
    //
    // `refused_lineage_unrootable` is the TENTH member, added by Epic 5's SR-7
    // adjudication (`t2.dfh.be.s8`): the checkpoint-store-damage arms split out
    // of `refused_no_base`. Restated here by the deliberate human edit this
    // literal exists to force — the whole point of not importing the const.
    let tail_outcomes = [
        "captured",
        "captured_capped",
        "rebased",
        "refused_trimmed",
        "refused_missing_bytes",
        "refused_sentinel",
        "refused_seq_gap",
        "refused_no_base",
        "refused_chain_full",
        "refused_lineage_unrootable",
    ];
    let tail_cells = gather_tail_outcome_cells(&metrics);
    let observed_outcomes: HashSet<String> = tail_cells.keys().cloned().collect();
    let expected_outcomes: HashSet<String> =
        tail_outcomes.iter().map(|o| (*o).to_string()).collect();
    assert_eq!(
        observed_outcomes, expected_outcomes,
        "pre-init outcome set on live_edit_checkpoint_tail_total must be exactly \
         {{captured, captured_capped, rebased, refused_trimmed, refused_missing_bytes, \
         refused_sentinel, refused_seq_gap, refused_no_base, refused_chain_full, \
         refused_lineage_unrootable}}; got {observed_outcomes:?}"
    );
    for outcome in tail_outcomes {
        let value = tail_cells
            .get(outcome)
            .copied()
            .expect("outcome set was just asserted equal, so every cell is present");
        assert_eq!(
            value, 0.0,
            "live_edit_checkpoint_tail_total{{outcome={outcome:?}}} MUST still read exactly 0 \
             after the full three-phase drive: every loop in this test ran with \
             `tails_off()` (tail capture pinned off), so the tail path must not have \
             executed at all — not even to refuse"
        );
    }
}

/// Render `live_edit_checkpoint_tail_total` and return every materialised
/// `outcome` cell as `outcome -> value`.
///
/// Parses the family directly rather than probing named cells with
/// `sample_value` so an out-of-band `outcome` (one the pre-init loop never
/// seeded) shows up as an EXTRA key instead of being invisible to a
/// per-name lookup.
fn gather_tail_outcome_cells(metrics: &Metrics) -> HashMap<String, f64> {
    let text = live_edit::observability::metrics::test_util::render(metrics);
    let mut out: HashMap<String, f64> = HashMap::new();
    for line in text.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("live_edit_checkpoint_tail_total") else {
            continue;
        };
        if !rest.starts_with('{') {
            continue;
        }
        let needle = "outcome=\"";
        let Some(i) = line.find(needle) else {
            continue;
        };
        let after = &line[i + needle.len()..];
        let Some(end) = after.find('"') else {
            continue;
        };
        let outcome = after[..end].to_owned();
        let value = line
            .rsplit_once(char::is_whitespace)
            .and_then(|(_, v)| v.trim().parse::<f64>().ok())
            .expect("every rendered sample line must end in a parseable value");
        out.insert(outcome, value);
    }
    out
}

/// Render `live_edit_checkpoint_writes_total` and return the set of
/// `(result, kind)` label pairs present on it.
fn gather_writes_label_pairs(metrics: &Metrics) -> HashSet<(String, String)> {
    let text = live_edit::observability::metrics::test_util::render(metrics);
    let mut out: HashSet<(String, String)> = HashSet::new();
    for line in text.lines() {
        let Some(rest) = line.strip_prefix("live_edit_checkpoint_writes_total") else {
            continue;
        };
        if !rest.starts_with('{') {
            continue;
        }
        let extract = |label: &str| -> Option<String> {
            let needle = format!("{label}=\"");
            let i = line.find(&needle)?;
            let after = &line[i + needle.len()..];
            let end = after.find('"')?;
            Some(after[..end].to_owned())
        };
        let result = extract("result").expect("`result` label must be present on every series");
        let kind = extract("kind").expect("`kind` label must be present on every series");
        out.insert((result, kind));
    }
    out
}

// ────────────────────────────────────────────────────────────────────────
// Case D — b.pw6: the unprovable-cut arm ticks its own outcome counter,
//          reuses `result="skipped"`, and persists NOTHING.
//
// Drives the `AssemblyDecision::SkippedNotQuiesced` arm of the loop. A hot
// writer commits against a wide layer while the loop ticks with both gates
// open (`quiesce = interval = 0`), so every tick reaches the assembler.
//
// The loop runs on `unprovable_cut_client` (`checkpoint::helpers`), the
// test-only seam that has the layer's every assembly pin above every cut, so
// its proven read is judged regressed. That is what makes the arm reachable:
// every read proves its cut — atomically at or below the atomic-read limits,
// under the snapshot fence above them (Spec SDD SR-4.1, SR-11.5) — so the only
// route to this arm is a header-sequence regression, which a real loop cannot
// be made to hit on cue. The seam needs `test-checkpoint-fault-injection`, so
// this case (and Case K) is compiled only under that feature — compiled out,
// not skipped, without it.
//
// Asserted:
//
//   1. `live_edit_checkpoint_stable_cut_total{outcome="unverified"}` rises —
//      the DR-exposure signal operators alert on.
//   2. `live_edit_checkpoint_writes_total{result="skipped", kind="layer"}`
//      rises by at least as much — the skip arm reuses the existing cell
//      rather than widening that counter's pinned label cardinality (Case C
//      is the guard on that).
//   3. The layer is observed as pending (b.jq3): the arm persists nothing by
//      design, so its commits stay outside DR and it must feed the aggregates
//      like any other pending layer. This fixture has no durable generation
//      at all, so it lands on `layers_never_written` rather than on the
//      histogram; Case K is the same arm for a layer that HAS one.
//   4. Nothing was persisted off a regressed cut, stated as two invariants:
//      generations on the adapter never outnumber the ticks that verified a
//      cut (none, with the seam armed), and `last_checkpoint_seq` never runs
//      ahead of the newest generation actually on the adapter (absent cursor
//      and empty adapter both read as 0). The cursor is the load-bearing half
//      — advancing it on a tick that wrote no blob would silence the next
//      tick's skip-if-no-commits gate and make the omission permanent.
//
// The loop is cancelled while the seam is still armed, so no tick can verify
// between the observation and the assertions.
// ────────────────────────────────────────────────────────────────────────

/// Read `live_edit_checkpoint_stable_cut_total{outcome, service}`. The
/// `service` label is required in the match: `Metrics::new` pre-materializes
/// both outcomes at zero under the *metrics* service label, so an
/// outcome-only (subset) match can return that pre-init 0 series instead of
/// the loop's.
fn read_stable_cut_counter(metrics: &Metrics, outcome: &str, service: &str) -> u64 {
    live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(metrics),
        "live_edit_checkpoint_stable_cut_total",
        &[("outcome", outcome), ("service", service)],
    )
    .unwrap_or(0.0) as u64
}

/// Seed a wide layer — 120 sections, `drive_captured_tail`'s number — for the
/// hot-writer cases. Returns `(object_id, session_id)`; the caller drives
/// commits with `layer_name`.
async fn seed_wide_layer(tv: &TestValkey, layer_name: &str) -> (u64, String) {
    use live_edit::domain::types::{Field, FieldValue, Section};

    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = session_store
        .create("checkpoint-metrics-session", "/stage", "metrics-test-user")
        .await
        .expect("session creation must succeed");

    let mut state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usda".to_string(),
        ..Default::default()
    };
    let ids: Vec<u64> = (1..=120u64).collect();
    for (order, sid) in ids.iter().enumerate() {
        state.sections.insert(
            *sid,
            Section {
                section_id: *sid,
                parent_id: 0,
                name: format!("seed-{sid}"),
                section_type: 1,
                section_order: order as u32,
                first_version: 1,
                last_version: 1,
            },
        );
        state.fields.insert(
            (*sid, "seed-field".to_string()),
            Field {
                section_id: *sid,
                key_name: "seed-field".to_string(),
                value: FieldValue::Inline(vec![0x5E, 0xED]),
                set_order: 0,
                authoring_set_order: 0,
                version: 1,
            },
        );
    }
    state.children.insert(0, ids);

    let layer_store = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
    let oid = layer_store
        .create_layer(
            &session_id,
            layer_name,
            "metrics-test-user",
            &state,
            Arc::new(MockExternalValueStore::new()),
        )
        .await
        .expect("create_layer must succeed");
    (oid, session_id)
}

/// Skipped ticks Case D waits for before it stops the loop: more than one, so
/// the sampler watches the header across repeated skip arms, not one.
#[cfg(feature = "test-checkpoint-fault-injection")]
const CASE_D_SKIPPED_TICKS: u64 = 3;

/// Cursor samples Case D's sampler must have taken before its verdict means
/// anything (one `HGETALL` every 10 ms).
#[cfg(feature = "test-checkpoint-fault-injection")]
const CASE_D_MIN_CURSOR_SAMPLES: usize = 10;

#[cfg(feature = "test-checkpoint-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn case_d_unprovable_cut_ticks_unverified_and_persists_nothing() {
    use oodf_ffi::{CommandCreateSection, OwnedDelta, StructCommand};

    const REPLICA: &str = "case-d-replica";
    let tv = TestValkey::setup_with_pool_size(8).await;
    let metrics = make_metrics();
    let layer_name = "not-quiesced-layer";
    let (oid, session_id) = seed_wide_layer(&tv, layer_name).await;

    // b.vnw — plant an outstanding restore bump so the "persists NOTHING"
    // claim covers the third cursor column too. The success arm clears this
    // column to 0 in the same HSET that advances the seq cursor; the
    // `SkippedNotQuiesced` arm must leave it exactly as found. Clearing it
    // without a generation landing would be the silent half of the bug: the
    // restore's artificial jump is still ahead of `last_checkpoint_seq`, so
    // the next attempt's assembly would measure a full band of jump as commit
    // traffic, inflate `observed_commit_span`, and size the NEXT restore's
    // band off it.
    //
    // The value is arbitrary but must be recognisable and must not collide
    // with anything else on this header.
    const PLANTED_RESTORE_BUMP: &str = "4242";
    let _: () = tv
        .client()
        .pool
        .hset(
            tv.key_schema().obj_header(&oid.to_string()),
            ("restore_seq_bump", PLANTED_RESTORE_BUMP),
        )
        .await
        .expect("HSET restore_seq_bump must succeed");

    // Hot writer: one CreateSection commit per iteration, so the header
    // sequence keeps moving and the layer stays due on every tick.
    let stop = Arc::new(AtomicBool::new(false));
    let commits = Arc::new(AtomicUsize::new(0));
    let writer = {
        let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
        let stop = stop.clone();
        let commits = commits.clone();
        let session_id = session_id.clone();
        tokio::spawn(async move {
            let mut next_section_id = 10_000u64;
            while !stop.load(Ordering::Relaxed) {
                let delta = OwnedDelta {
                    struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
                        parent_id: 0,
                        section_id: next_section_id,
                        section_name: format!("hot-{next_section_id}"),
                        section_type: 1,
                    })],
                    ..Default::default()
                };
                if ls
                    .update_layer_relayed(
                        &session_id,
                        layer_name,
                        oid,
                        Some(delta),
                        "hot-writer",
                        0,
                    )
                    .await
                    .is_err()
                {
                    // The pool is torn down at test exit; stop quietly rather
                    // than poisoning the join with a panic.
                    return;
                }
                commits.fetch_add(1, Ordering::Relaxed);
                next_section_id += 1;
            }
        })
    };

    // Both gates open so every tick reaches the assembler.
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
        leader: crate::common::test_leader(REPLICA),
        tail: tails_off(),
    };
    let skipped_before = read_writes_counter(&metrics, "skipped");

    // b.yut — concurrent observer for the ONE header state the two cursor
    // columns can never legitimately be in: `last_checkpoint_ms` present while
    // `last_checkpoint_seq` is absent.
    //
    // Why that state is unreachable, for every arm this loop takes: the
    // success arm writes both columns in a SINGLE `HSET` (so they land
    // together or not at all), and no skip arm writes either. The only way to
    // observe the pair half-written is for a skip arm to have started stamping
    // the cadence clock on its own — the insidious half of the b.pw6 /b.yut
    // failure, because it leaves no trace in the audit trail the post-hoc
    // assertions below inspect, yet it defers the layer's next interval-arm
    // eligibility by a full `LIVE_EDIT_CHECKPOINT_INTERVAL_SECS`.
    //
    // Post-hoc assertions cannot see it: by the time the loop is cancelled the
    // success arm may have landed and papered the transient over. Sampling
    // throughout the run is the only way to catch a *transient* half-write,
    // and it is unconditional — it does not depend on which arm any one tick
    // took.
    //
    // Each sample is one `HGETALL`, so the pair is read atomically by the
    // server: a false positive from reading the two fields in two round-trips
    // is structurally impossible here.
    //
    // (`written_at_ms` is non-optional on the `apply_checkpoint` restore path
    // as of the R2 semantics; that changes nothing for this test, which drives
    // the periodic LOOP's arms, where the single-HSET invariant is the whole
    // claim.)
    let header_key = format!("{}{{{oid}}}:obj:hdr", tv.key_schema().prefix());
    let sampler_stop = Arc::new(AtomicBool::new(false));
    let half_written_observations = Arc::new(AtomicUsize::new(0));
    let samples_taken = Arc::new(AtomicUsize::new(0));
    let sampler = {
        let client = tv.client().clone();
        let header_key = header_key.clone();
        let stop = sampler_stop.clone();
        let half_written = half_written_observations.clone();
        let samples = samples_taken.clone();
        tokio::spawn(async move {
            while !stop.load(Ordering::Relaxed) {
                let Ok(header) = client
                    .pool
                    .hgetall::<HashMap<String, String>, _>(&header_key)
                    .await
                else {
                    // The pool is torn down at test exit; stop quietly.
                    return;
                };
                samples.fetch_add(1, Ordering::Relaxed);
                if header.contains_key("last_checkpoint_ms")
                    && !header.contains_key("last_checkpoint_seq")
                {
                    half_written.fetch_add(1, Ordering::Relaxed);
                }
                sleep(Duration::from_millis(10)).await;
            }
        })
    };

    // The unprovable-cut seam, armed for this layer — see the case banner.
    let handle = spawn_checkpoint_loop(
        unprovable_cut_client(&tv, oid),
        tv.key_schema().clone(),
        store_for_loop,
        config,
        Some(metrics.clone()),
        cancel.clone(),
    );

    // Run until several ticks have taken the skip arm AND the sampler has
    // watched the header across them. A tick's one proven read is judged
    // regressed in a few milliseconds, so stopping at the first unverified
    // tick would leave the sampler only a handful of 10 ms periods — too few
    // for its verdict to cover the skip arms it exists to watch.
    let metrics_for_poll = metrics.clone();
    let samples_for_poll = samples_taken.clone();
    let saw_unverified = poll_until(Duration::from_secs(20), || {
        let m = metrics_for_poll.clone();
        let samples = samples_for_poll.clone();
        async move {
            read_stable_cut_counter(&m, "unverified", REPLICA) >= CASE_D_SKIPPED_TICKS
                && samples.load(Ordering::Relaxed) >= CASE_D_MIN_CURSOR_SAMPLES
        }
    })
    .await;
    assert!(
        saw_unverified,
        "a layer whose proven read the seam has judged regressed must produce \
         {CASE_D_SKIPPED_TICKS} outcome=\"unverified\" stable-cut ticks, watched by at least \
         {CASE_D_MIN_CURSOR_SAMPLES} cursor samples, within 20 s; observed {} unverified \
         tick(s), {} sample(s) and {} writer commits. Pre-b.pw6 the assembler had no such arm \
         at all — it stamped the post-fan-out sequence onto a snapshot it never proved.",
        read_stable_cut_counter(&metrics, "unverified", REPLICA),
        samples_taken.load(Ordering::Relaxed),
        commits.load(Ordering::Relaxed)
    );

    // Cancel while the seam is still armed: a tick that verified would
    // legitimately persist and invalidate the "nothing persisted" assertions
    // below.
    cancel_and_join(cancel, handle).await;
    stop.store(true, Ordering::Relaxed);
    writer.await.expect("hot-writer task must not panic");
    sampler_stop.store(true, Ordering::Relaxed);
    sampler.await.expect("cursor sampler task must not panic");

    // The sampler's verdict, checked before anything else: a single
    // observation of the half-written pair is a failure, and the sample count
    // guards against a sampler that silently died on its first HGETALL.
    let half_written = half_written_observations.load(Ordering::Relaxed);
    let samples = samples_taken.load(Ordering::Relaxed);
    assert!(
        samples >= CASE_D_MIN_CURSOR_SAMPLES,
        "the concurrent cursor sampler must have taken samples throughout the \
         run for its verdict to mean anything; it took {samples}"
    );
    assert_eq!(
        half_written, 0,
        "obj:hdr for object {oid} was observed {half_written} time(s) (out of \
         {samples} samples) carrying last_checkpoint_ms WITHOUT \
         last_checkpoint_seq. That pair is written by one HSET on the success \
         arm and by neither of the skip arms, so the state is unreachable — \
         observing it means a skip arm has started stamping the cadence clock \
         on a cut it never persisted, deferring the layer's next interval-arm \
         eligibility by a full LIVE_EDIT_CHECKPOINT_INTERVAL_SECS (b.yut)"
    );

    let unverified = read_stable_cut_counter(&metrics, "unverified", REPLICA);
    let verified = read_stable_cut_counter(&metrics, "verified", REPLICA);
    let skipped_delta = read_writes_counter(&metrics, "skipped") - skipped_before;

    assert!(unverified >= 1, "precondition re-check after cancellation");
    assert_eq!(
        verified, 0,
        "the seam skips every assembly of this layer, so no tick may verify; \
         unverified={unverified}"
    );
    assert!(
        skipped_delta >= unverified,
        "the skip arm must also bump writes_total{{result=\"skipped\", \
         kind=\"layer\"}} (it deliberately reuses that cell rather than \
         widening the counter's label cardinality); skipped_delta={skipped_delta} \
         unverified={unverified}"
    );

    // b.jq3 — the pending-age half of this arm, which is otherwise the only
    // row of the accumulator's observation map with no test behind it. A layer
    // the assembler refused to prove persisted nothing BY DESIGN, so its
    // commits are still outside DR and it must be observed exactly like any
    // other pending layer.
    //
    // This fixture never injects `last_checkpoint_ms` and no skip arm may
    // stamp one, so an un-raced run observes the layer as NEVER-WRITTEN:
    // outside DR entirely, not stale by N seconds. That routing is why the
    // count is a separate gauge — `now - 0` is ~1.77e9 s, which would peg the
    // histogram's `+Inf` bucket and poison `_sum` (the b.yut defect shape).
    //
    // All three are stated as implications on `verified`: a tick that verified
    // would stamp a real generation, after which the layer is legitimately
    // stale-rather-than-absent and every count below legitimately changes.
    // With the seam armed `verified` is 0 (asserted above), so each implication
    // reduces to its left-hand side.
    let aggregates_published = read_oldest_pending_age(&metrics, REPLICA).is_some();
    assert!(
        aggregates_published,
        "the leader ticked for seconds before cancellation, so it must have \
         published its per-tick aggregates at least once; an absent series \
         means no tick ever drained its SCAN"
    );
    assert!(
        read_layers_never_written(&metrics, REPLICA) == Some(1) || verified >= 1,
        "a layer left pending by the SkippedNotQuiesced arm with no durable \
         generation at all must be counted on the never-written gauge — that \
         arm persists nothing by design, so silence there is a layer sitting \
         outside DR with no signal at all ({verified} verified / {unverified} \
         unverified ticks); got {:?}",
        read_layers_never_written(&metrics, REPLICA)
    );
    assert!(
        read_pending_age_sample_count(&metrics) == 0 || verified >= 1,
        "and it must contribute NO histogram sample: there is no finite age \
         for a layer that has never been checkpointed, so folding one in would \
         record ~1.77e9 s and poison _sum for the whole fleet; got {} sample(s) \
         over {verified} verified tick(s)",
        read_pending_age_sample_count(&metrics)
    );
    assert!(
        read_oldest_pending_age(&metrics, REPLICA) == Some(0.0) || verified >= 1,
        "and it must leave the max gauge at 0.0 for the same reason — the \
         never-written sentinel must never reach the age number line; got {:?}",
        read_oldest_pending_age(&metrics, REPLICA)
    );

    // Nothing may be persisted off a regressed cut. Both assertions are sound
    // whatever arm each tick took: generations can only be created by a
    // verified assembly, and the cursor only by a written one.
    let handles = store
        .list_layer_checkpoints(oid)
        .await
        .expect("list must succeed");
    let generations = handles.len() as u64;
    assert!(
        generations <= verified,
        "the adapter holds {generations} generation(s) for object {oid} but only \
         {verified} assembly/-ies proved a cut — a generation was written off an \
         unproven snapshot (b.pw6)"
    );

    // `header_key` is the one built for the concurrent sampler above.
    let cursor: Option<String> = tv
        .client()
        .pool
        .hget(&header_key, "last_checkpoint_seq")
        .await
        .expect("HGET last_checkpoint_seq must succeed");
    let cursor_seq: u64 = cursor
        .as_deref()
        .map(|raw| {
            raw.parse().unwrap_or_else(|e| {
                panic!("last_checkpoint_seq must be numeric; got {raw:?} ({e})")
            })
        })
        .unwrap_or(0);

    // The cursor may never name a cut the adapter does not actually hold.
    // Advancing it past what was written is the half of the b.pw6 hole that
    // makes an omission permanent: the next tick's skip-if-no-commits gate fires
    // on a generation that was never persisted, so nothing re-snapshots the
    // layer until its next commit.
    //
    // Deliberately stated as one *unconditional* comparison rather than as an
    // `if verified == 0 { cursor must be absent }` arm, so it holds whatever arm
    // each tick took. Here `unwrap_or(0)` carries both halves of the claim:
    // with no generation on the adapter the ceiling is 0, so ANY cursor value
    // fails; with generations present, the cursor may not exceed the newest
    // one's `snap_seq`.
    let newest_generation_seq = handles.iter().map(|h| h.snap_seq).max().unwrap_or(0);
    assert!(
        cursor_seq <= newest_generation_seq,
        "last_checkpoint_seq={cursor_seq} is ahead of every generation the \
         adapter holds for object {oid} (newest snap_seq={newest_generation_seq} \
         across {generations} generation(s); {verified} verified / {unverified} \
         unverified ticks) — the cursor advanced off a cut whose blob was never \
         written (b.pw6)"
    );

    // b.yut — "persist nothing" now covers TWO header columns. The success
    // arm stamps `last_checkpoint_seq` and `last_checkpoint_ms` in one HSET;
    // the `SkippedNotQuiesced` arm must stamp neither.
    //
    // Stamping only `last_checkpoint_ms` off an unproven cut would be the
    // more insidious half: it leaves no trace in the audit trail the previous
    // assertion inspects, yet it defers the layer's next interval-arm
    // eligibility by a full `LIVE_EDIT_CHECKPOINT_INTERVAL_SECS` — an
    // interval-long DR gap opened by the very path that exists to avoid
    // writing an unsound blob.
    //
    // Stated as the two unconditional implications, for the same reason the
    // cursor assertion above is unconditional: they hold whatever arm each
    // tick took.
    //   (i)  the column may only exist if some tick actually wrote a
    //        generation, and
    //   (ii) it must agree with the seq cursor about whether one did — they
    //        travel in a single HSET, so exactly one of them being present is
    //        proof the two arms have drifted apart.
    let checkpoint_ms: Option<String> = tv
        .client()
        .pool
        .hget(&header_key, "last_checkpoint_ms")
        .await
        .expect("HGET last_checkpoint_ms must succeed");
    assert!(
        checkpoint_ms.is_none() || verified >= 1,
        "obj:hdr last_checkpoint_ms={checkpoint_ms:?} was stamped on object \
         {oid} although no assembly proved a cut ({verified} verified / \
         {unverified} unverified ticks) — the SkippedNotQuiesced arm must \
         persist NOTHING, and stamping the cadence clock there defers the \
         layer's next interval-arm eligibility by a full interval (b.yut)"
    );
    assert_eq!(
        checkpoint_ms.is_some(),
        cursor.is_some(),
        "the two cursor columns are written in ONE HSET on the success arm and \
         neither is written on the skip arm, so they must agree about whether a \
         generation landed; got last_checkpoint_seq={cursor:?} and \
         last_checkpoint_ms={checkpoint_ms:?} for object {oid}"
    );

    // b.vnw — "persist nothing" now covers a THIRD column, and this one is an
    // erasure rather than a write: the success arm clears `restore_seq_bump`
    // to 0 in the same HSET, so the skip arm's obligation is to leave the
    // planted value untouched.
    //
    // Clearing it off an unproven cut is the insidious shape here. It leaves
    // no trace in the audit trail either of the two assertions above inspect,
    // yet it means the restore's artificial sequence jump — still ahead of
    // `last_checkpoint_seq`, because no generation landed — gets measured as
    // commit traffic by the next attempt's assembly. That inflates
    // `observed_commit_span` by a full band and sizes the NEXT restore's band
    // off the inflated number.
    //
    // Stated as the same unconditional implication as its siblings, for the
    // same reason: it holds whatever arm each tick took.
    let restore_bump: Option<String> = tv
        .client()
        .pool
        .hget(&header_key, "restore_seq_bump")
        .await
        .expect("HGET restore_seq_bump must succeed");
    assert!(
        restore_bump.as_deref() == Some(PLANTED_RESTORE_BUMP) || verified >= 1,
        "obj:hdr restore_seq_bump={restore_bump:?} on object {oid} was changed \
         from the planted {PLANTED_RESTORE_BUMP} although no assembly proved a \
         cut ({verified} verified / {unverified} unverified ticks) — the \
         SkippedNotQuiesced arm must leave the outstanding restore bump owed, \
         or the next attempt measures a full band of artificial jump as commit \
         traffic (b.vnw)"
    );
}

// ────────────────────────────────────────────────────────────────────────
// Case D′ — Case D's fixture at the SHIPPED atomic-read limits: a hot layer
//           at or below them ticks only `outcome="verified"` and is written
//           tick after tick (Spec SDD SR-4.1, SR-8.6).
//
// The same wide layer and an even hotter writer (field-only commits ride
// `update_layer`'s Lua fast path, roughly an order of magnitude faster than
// Case D's structural ones), with the loop on the fixture's own client — the
// production posture. The assembler reads the layer in one atomic read that
// proves a cut however hot it is, so:
//
//   1. `stable_cut_total{outcome="unverified"}` stays at ZERO. On this path
//      only a header-sequence regression reaches that cell, and nothing here
//      regresses the header — which is what makes the SR-8.6 alert recipe
//      ("any increase from a layer at or below the limits") sound.
//   2. Several ticks VERIFY while the writer is still running, and the cursor
//      names a generation the adapter actually holds.
//
// Before the atomic read this fixture was Case D: the quiescence proof lost
// to the writer and the unverified cell rose within a few ticks, so (1) is the
// assertion that fails on the pre-atomic loop.
// ────────────────────────────────────────────────────────────────────────

/// Verified ticks Case D′ waits for while the writer keeps committing. Three
/// is enough to show the layer is written tick after tick rather than once on
/// a lucky tick, and at the 50 ms tick it arrives in well under a second.
const HOT_AT_THE_LIMITS_VERIFIED_TICKS: u64 = 3;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn case_d_prime_hot_layer_at_the_atomic_read_limits_verifies_every_tick_and_never_ticks_unverified(
) {
    use oodf_ffi::{OwnedDelta, SetField};

    const REPLICA: &str = "case-d-prime-replica";
    let tv = TestValkey::setup_with_pool_size(8).await;
    let metrics = make_metrics();
    let layer_name = "hot-at-the-limits-layer";
    let (oid, session_id) = seed_wide_layer(&tv, layer_name).await;

    let stop = Arc::new(AtomicBool::new(false));
    let commits = Arc::new(AtomicUsize::new(0));
    let writer = {
        let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
        let stop = stop.clone();
        let commits = commits.clone();
        let session_id = session_id.clone();
        tokio::spawn(async move {
            let mut nonce = 30_000u64;
            while !stop.load(Ordering::Relaxed) {
                let delta = OwnedDelta {
                    set_fields: vec![SetField {
                        // Section 1 exists in every `seed_wide_layer` layer.
                        section_id: 1,
                        key_name: "hot-field".to_string(),
                        value_or_ext_hash: nonce.to_le_bytes().to_vec(),
                        ext_value_size: 0,
                        ext_value_index: 0,
                        set_order: nonce,
                    }],
                    ..Default::default()
                };
                if ls
                    .update_layer_relayed(
                        &session_id,
                        layer_name,
                        oid,
                        Some(delta),
                        "hot-writer",
                        0,
                    )
                    .await
                    .is_err()
                {
                    // The pool is torn down at test exit; stop quietly.
                    return;
                }
                commits.fetch_add(1, Ordering::Relaxed);
                nonce += 1;
            }
        })
    };
    // Meet the first tick with an already-hot layer.
    assert!(
        poll_until(Duration::from_secs(20), || {
            let commits = commits.clone();
            async move { commits.load(Ordering::Relaxed) >= 20 }
        })
        .await,
        "the hot writer must warm up before the loop starts"
    );

    let store = Arc::new(InMemoryCheckpointStore::new());
    let store_for_loop: Arc<dyn CheckpointStore> = store.clone();
    let cancel = CancellationToken::new();
    let commits_at_start = commits.load(Ordering::Relaxed);
    let handle = spawn_checkpoint_loop(
        tv.client().clone(),
        tv.key_schema().clone(),
        store_for_loop,
        CheckpointLoopConfig {
            tick: Duration::from_millis(50),
            interval: Duration::ZERO,
            quiesce: Duration::ZERO,
            retain_generations: 2,
            counters_sync: Duration::from_secs(3600),
            reconcile_interval: Duration::from_secs(3600),
            leader: crate::common::test_leader(REPLICA),
            tail: tails_off(),
        },
        Some(metrics.clone()),
        cancel.clone(),
    );

    let metrics_for_poll = metrics.clone();
    let reached = poll_until(Duration::from_secs(20), || {
        let m = metrics_for_poll.clone();
        async move {
            read_stable_cut_counter(&m, "verified", REPLICA) >= HOT_AT_THE_LIMITS_VERIFIED_TICKS
        }
    })
    .await;

    // Cancel WHILE the writer is still hammering, as Case D does: every tick
    // counted below ran against a hot layer.
    cancel_and_join(cancel, handle).await;
    let commits_during = commits.load(Ordering::Relaxed) - commits_at_start;
    stop.store(true, Ordering::Relaxed);
    writer.await.expect("hot-writer task must not panic");

    let verified = read_stable_cut_counter(&metrics, "verified", REPLICA);
    let unverified = read_stable_cut_counter(&metrics, "unverified", REPLICA);
    assert!(
        reached,
        "a hot layer at or below the atomic-read limits must verify on tick after tick; only \
         {verified} verified tick(s) within 20 s ({unverified} unverified, {commits_during} \
         writer commits during the run)"
    );
    assert!(
        commits_during >= HOT_AT_THE_LIMITS_VERIFIED_TICKS as usize,
        "the writer must have kept committing while the loop ran, or the layer was not hot; it \
         landed {commits_during}"
    );
    assert_eq!(
        unverified, 0,
        "at or below the atomic-read limits only a header-sequence regression may tick \
         outcome=\"unverified\" (Spec SDD SR-4.1, SR-8.6), and nothing here regresses the header. \
         A non-zero count is a read losing its proof to the writer, which only the retired \
         quiescence loop ever did — Case D reaches that arm only through its unprovable-cut \
         seam ({verified} verified ticks)"
    );

    let handles = store
        .list_layer_checkpoints(oid)
        .await
        .expect("list must succeed");
    let newest_generation_seq = handles.iter().map(|h| h.snap_seq).max().unwrap_or(0);
    let cursor: Option<String> = tv
        .client()
        .pool
        .hget(
            tv.key_schema().obj_header(&oid.to_string()),
            "last_checkpoint_seq",
        )
        .await
        .expect("HGET last_checkpoint_seq must succeed");
    let cursor_seq: u64 = cursor
        .as_deref()
        .and_then(|raw| raw.parse().ok())
        .unwrap_or_else(|| {
            panic!("verified ticks wrote generations, so last_checkpoint_seq must be stamped and numeric; got {cursor:?}")
        });
    assert!(
        cursor_seq > 1 && cursor_seq <= newest_generation_seq,
        "the cursor must have advanced over the writer's commits (past the seeded sequence 1) \
         without running ahead of the newest generation the adapter holds for object {oid}: \
         cursor {cursor_seq}, newest generation {newest_generation_seq} ({} generation(s) after \
         retention)",
        handles.len()
    );
}

// ────────────────────────────────────────────────────────────────────────
// Case E — the other half of the stable-cut metric: a quiescent layer ticks
//          `outcome="verified"` and never `outcome="unverified"`.
//
// Case D pins the unverified arm; without this one a regression that ticked
// `unverified` unconditionally would still pass.
// ────────────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn case_e_quiescent_layer_ticks_verified_and_never_unverified() {
    const REPLICA: &str = "case-e-replica";
    let tv = TestValkey::setup_with_pool_size(8).await;
    let metrics = make_metrics();
    let oid = seed_layer(&tv, "quiescent-layer").await;

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
        leader: crate::common::test_leader(REPLICA),
        tail: tails_off(),
    };
    let handle = spawn_checkpoint_loop(
        tv.client().clone(),
        tv.key_schema().clone(),
        store_for_loop,
        config,
        Some(metrics.clone()),
        cancel.clone(),
    );

    let metrics_for_poll = metrics.clone();
    let saw_success = poll_until(Duration::from_secs(10), || {
        let m = metrics_for_poll.clone();
        async move { read_writes_counter(&m, "success") >= 1 }
    })
    .await;
    assert!(
        saw_success,
        "an idle layer with both gates open must be checkpointed within 10 s"
    );
    cancel_and_join(cancel, handle).await;

    let verified = read_stable_cut_counter(&metrics, "verified", REPLICA);
    let unverified = read_stable_cut_counter(&metrics, "unverified", REPLICA);

    assert!(
        verified >= 1,
        "every assembly the loop actually runs must record a stable-cut \
         outcome; a written generation with no `verified` tick means the \
         instrument is not wired to the write path"
    );
    assert_eq!(
        unverified, 0,
        "nothing was writing to this layer, so no assembly may report an \
         unprovable cut"
    );

    // Sanity that the loop really did the work the metric claims.
    assert!(
        !store
            .list_layer_checkpoints(oid)
            .await
            .expect("list must succeed")
            .is_empty(),
        "a verified assembly must have persisted a generation"
    );
}

// ────────────────────────────────────────────────────────────────────────
// Case F — the pre-gate SEQUENCE refusal is operator-visible on BOTH the
//          counter and the staleness histogram.
//
// A header whose `seq` column is missing or unparseable is a data-shape
// violation and the one remaining refusal that drops a layer from the tick
// entirely: without a sequence number neither the skip-if-no-commits gate nor
// the assembler has anything to anchor on. Two things must be true of it:
//
//   1. It bumps `writes_total{result="skipped", kind="layer"}`. A WARN alone
//      is invisible on every dashboard, and a layer stuck in this state is
//      silently outside DR. `"skipped"` and not `"error"`: the adapter is
//      healthy and was never called, and the metrics doc's adapter-health
//      alert keys on `result="error"`, so misfiling this refusal there would
//      page an operator toward the wrong subsystem. Label cardinality is
//      unchanged — both values already exist on the counter.
//   2. It DOES sample pending age. This inverts what the case asserted before
//      b.jq3, and the inversion is the improvement. Under the predecessor
//      metric — "seconds since the last commit" — the refusal arm had nothing
//      to report, because it returns before establishing that the header even
//      carries a readable commit timestamp. Pending age needs no commit clock:
//      it measures `now - last_checkpoint_ms`, both of which are in hand at
//      the refusal. And the conservative reading is the correct one — this
//      layer is frozen out of DR until a writer repairs the column, which is
//      the most severe form of the staleness the histogram exists to measure.
//      Silence here is precisely the blind spot that let a permanently
//      un-checkpointable layer sit undetected.
//
// The fixture stamps a FRESH commit clock and a 7 s-old cadence clock, so the
// sample's provenance is pinned as well as its existence: a value derived from
// the commit clock would land in the `<=5` bucket and fail.
//
// Both gate thresholds are zero so every other gate in `run_one_tick` is wide
// open: the only thing that can hold this layer back is the seq refusal.
// ────────────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn case_f_missing_seq_column_ticks_skipped_and_samples_pending_age() {
    const REPLICA: &str = "case-f-replica";
    let tv = TestValkey::setup_with_pool_size(8).await;
    let metrics = make_metrics();
    let oid = seed_layer(&tv, "seq-refusal-layer").await;

    // Healthy, recent commit clock plus a cadence stamp 7 s in the past.
    // Nothing about the timestamps is what holds this layer back; the 7 s
    // backdate exists so the pending-age sample below has a bucket-checkable
    // provenance.
    set_gate_timestamps(&tv, oid, 0, 7_000).await;

    // Remove the sequence column — the one pre-gate refusal.
    let header_key = tv.key_schema().obj_header(&oid.to_string());
    let removed: u64 = tv
        .client()
        .pool
        .hdel(&header_key, "seq")
        .await
        .expect("HDEL seq must succeed");
    assert_eq!(
        removed, 1,
        "create_layer must have stamped obj:hdr seq for the HDEL to remove; \
         the fixture is not exercising what it thinks it is"
    );

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
        leader: crate::common::test_leader(REPLICA),
        tail: tails_off(),
    };
    let handle = spawn_checkpoint_loop(
        tv.client().clone(),
        tv.key_schema().clone(),
        store_for_loop,
        config,
        Some(metrics.clone()),
        cancel.clone(),
    );

    let metrics_for_poll = metrics.clone();
    let saw_skipped = poll_until(Duration::from_secs(10), || {
        let m = metrics_for_poll.clone();
        async move { read_writes_counter(&m, "skipped") >= 1 }
    })
    .await;
    assert!(
        saw_skipped,
        "a header with no `seq` column must bump \
         writes_total{{result=\"skipped\", kind=\"layer\"}} — a WARN alone is \
         invisible on a dashboard and the layer is silently outside DR (b.yut)"
    );

    assert_eq!(
        read_writes_counter(&metrics, "error"),
        0,
        "the seq refusal must NOT be counted as result=\"error\": the adapter \
         is healthy and was never called, and the metrics doc's adapter-health \
         alert keys on `error`, so filing it there points operators at the \
         wrong subsystem"
    );
    // The b.jq3 inversion: the refusal arm samples, and the sample comes from
    // the cadence clock, not the commit clock.
    let (buckets, total, _) = pending_age_histogram(&metrics);
    assert!(
        total >= 1,
        "a layer refused at the seq check must STILL sample pending age: it is \
         permanently un-checkpointable until a writer restores the column, \
         which is the most severe staleness there is. Silence here is the blind \
         spot b.jq3 closed — the predecessor metric could not report this arm \
         because it measured seconds-since-commit and the refusal returns \
         before a commit clock is established"
    );
    assert_eq!(
        buckets[0].1, 0,
        "no sample may fall in the <=5s bucket: this fixture's commit clock is \
         stamped at ~now while its cadence clock is 7s old, so a sub-5s value \
         means the refusal arm is measuring the wrong column"
    );
    assert_eq!(
        buckets[2].1, total,
        "all {total} samples must sit inside the <=30s bucket — a value outside \
         it means the never-checkpointed sentinel (`now - 0` ≈ 1.77e9 s) leaked \
         into the histogram and poisoned _sum"
    );
    // The per-tick gauges are published only after the SCAN drains, so a tick
    // that has bumped the counter may not have published yet. Poll for the
    // series rather than reading it on the first observed skip.
    let metrics_for_poll = metrics.clone();
    let published = poll_until(Duration::from_secs(10), || {
        let m = metrics_for_poll.clone();
        async move { read_layers_never_written(&m, REPLICA).is_some() }
    })
    .await;
    assert!(
        published,
        "the leader must publish its per-tick aggregates once the SCAN drains"
    );
    assert_eq!(
        read_layers_never_written(&metrics, REPLICA),
        Some(0),
        "this layer has a durable generation stamped 7s ago, so the refusal \
         puts it on the histogram, not on the never-written gauge"
    );
    assert_eq!(
        read_writes_counter(&metrics, "success"),
        0,
        "a header with no `seq` column must never be checkpointed"
    );
    assert!(
        store
            .list_layer_checkpoints(oid)
            .await
            .expect("list must succeed")
            .is_empty(),
        "no generation may be persisted for a layer refused at the seq check"
    );
    let cursor: Option<String> = tv
        .client()
        .pool
        .hget(&header_key, "last_checkpoint_seq")
        .await
        .expect("HGET last_checkpoint_seq must succeed");
    assert!(
        cursor.is_none(),
        "no cursor may be stamped for a layer refused at the seq check; got \
         {cursor:?}"
    );

    // Liveness control: restore the column and watch the SAME loop check the
    // SAME layer in. Without this the negatives above (no success, no
    // generation, no cursor) would also be satisfied by a loop that had
    // crashed or never acquired leadership.
    let _: () = tv
        .client()
        .pool
        .hset(&header_key, ("seq", "1"))
        .await
        .expect("HSET seq must succeed");
    let metrics_for_poll = metrics.clone();
    let recovered = poll_until(Duration::from_secs(20), || {
        let m = metrics_for_poll.clone();
        async move { read_writes_counter(&m, "success") >= 1 }
    })
    .await;
    cancel_and_join(cancel, handle).await;
    assert!(
        recovered,
        "once `seq` is restored the layer must be checkpointed — the refusal \
         above has to be the missing-seq check, not a wedged or leaderless \
         loop; got success={}",
        read_writes_counter(&metrics, "success")
    );
}

// ────────────────────────────────────────────────────────────────────────
// Case G — b.jq3, THE regression guard: an idle, fully-checkpointed layer
//          produces NO staleness sample and never moves the max gauge.
//
// This is the defect the ticket exists to fix, stated as a test. The
// predecessor metric (`live_edit_checkpoint_lag_seconds`) was observed on the
// skip-if-no-commits arm and measured time since the last COMMIT, so a layer
// that had been backed up and then left alone contributed a sample on EVERY
// tick, each one larger than the last. On a quiet fleet the p99 climbed
// without bound while every layer was perfectly protected, and the `> 300s`
// alert recipe then published to operators false-fired. Operators
// could not tell "the checkpoint loop has fallen behind" from "nobody is
// editing anything", which is the difference between a page and a no-op.
//
// The fix scopes the histogram to layers with un-persisted commits. This
// layer has none after its first checkpoint lands, so from that moment on it
// must be metrically silent no matter how many ticks pass.
//
// Both gates are wide open (`interval = quiesce = 0`) so the layer is
// checkpointed on the first eligible tick and every tick after it takes the
// skip-if-no-commits arm — the exact arm that used to sample.
//
// The positive control at the end is load-bearing: without it, "zero samples"
// would also be satisfied by an unwired instrument or a loop that never ran.
// ────────────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn case_g_idle_fully_checkpointed_layer_never_samples_pending_age() {
    const REPLICA: &str = "case-g-replica";
    /// How many post-checkpoint ticks the layer must survive silently. Each
    /// one is a skip-if-no-commits pass, and each one would have produced a
    /// sample under the predecessor metric.
    const IDLE_TICKS: u64 = 5;

    let tv = TestValkey::setup_with_pool_size(8).await;
    let metrics = make_metrics();
    let oid = seed_layer(&tv, "idle-layer").await;

    let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
    let cancel = CancellationToken::new();
    let config = CheckpointLoopConfig {
        tick: Duration::from_millis(50),
        interval: Duration::ZERO,
        quiesce: Duration::ZERO,
        retain_generations: 2,
        counters_sync: Duration::from_secs(3600),
        reconcile_interval: Duration::from_secs(3600),
        leader: crate::common::test_leader(REPLICA),
        tail: tails_off(),
    };
    let handle = spawn_checkpoint_loop(
        tv.client().clone(),
        tv.key_schema().clone(),
        store,
        config,
        Some(metrics.clone()),
        cancel.clone(),
    );

    // Phase 1 — the layer gets its one and only checkpoint.
    let metrics_for_poll = metrics.clone();
    let saw_success = poll_until(Duration::from_secs(10), || {
        let m = metrics_for_poll.clone();
        async move { read_writes_counter(&m, "success") >= 1 }
    })
    .await;
    assert!(
        saw_success,
        "a fresh layer with both gates open must be checkpointed within 10 s"
    );

    // Phase 2 — let it sit. Every tick from here takes the skip-if-no-commits
    // arm, and its `last_checkpoint_ms` ages by another 50 ms each time.
    let metrics_for_poll = metrics.clone();
    let saw_idle_ticks = poll_until(Duration::from_secs(10), || {
        let m = metrics_for_poll.clone();
        async move { read_writes_counter(&m, "skipped") >= IDLE_TICKS }
    })
    .await;
    assert!(
        saw_idle_ticks,
        "the loop must have re-evaluated the checkpointed layer at least \
         {IDLE_TICKS} times; got {} skip(s)",
        read_writes_counter(&metrics, "skipped")
    );

    // The verdict.
    assert_eq!(
        read_pending_age_sample_count(&metrics),
        0,
        "an idle, fully-checkpointed layer must contribute NOTHING to \
         live_edit_checkpoint_pending_age_seconds, however many ticks pass. \
         Pre-b.jq3 the skip-if-no-commits arm sampled time-since-last-COMMIT \
         here, so a quiet fleet's p99 climbed without bound while every layer \
         was perfectly protected — the false positive that made the documented \
         `> 300s` alert unusable. (The one checkpoint this layer did take was \
         its first ever, so it had no finite pre-write age to report either.)"
    );
    assert_eq!(
        read_oldest_pending_age(&metrics, REPLICA),
        Some(0.0),
        "the leader must publish the max gauge on every completed tick, and a \
         tick that found nothing pending must publish 0.0 — `None` here means \
         the gauge was never wired or no tick ever drained its SCAN, and a \
         non-zero value means an idle layer moved a signal it has no business \
         moving"
    );
    assert_eq!(
        read_layers_never_written(&metrics, REPLICA),
        Some(0),
        "the layer's first-ever checkpoint SUCCEEDED, so as of the end of that \
         tick it is inside DR and must not be counted as never-written — \
         counting it would make the gauge non-zero on every healthy fleet that \
         creates layers"
    );

    // Positive control: make the SAME layer pending again against a durable
    // generation that is genuinely old, and watch the same instrument fire.
    // 450 s sits strictly BETWEEN the `<=300` and `<=600` bucket boundaries,
    // so the control pins that the value tracks `last_checkpoint_ms` and does
    // so with slack in both directions — an age injected exactly ON a boundary
    // spills into the next bucket the moment the tick's clock read lands a
    // millisecond later.
    let header_key = tv.key_schema().obj_header(&oid.to_string());
    set_gate_timestamps(&tv, oid, 0, 450_000).await;
    let _: () = tv
        .client()
        .pool
        .hset(&header_key, ("seq", "2"))
        .await
        .expect("HSET seq must succeed");

    let metrics_for_poll = metrics.clone();
    let saw_sample = poll_until(Duration::from_secs(10), || {
        let m = metrics_for_poll.clone();
        async move { read_pending_age_sample_count(&m) >= 1 }
    })
    .await;
    cancel_and_join(cancel, handle).await;
    assert!(
        saw_sample,
        "once the layer has un-persisted commits again the histogram must \
         record — the zero above has to be the pending-scoping, not an unwired \
         instrument or a wedged loop"
    );

    let (buckets, total, _) = pending_age_histogram(&metrics);
    assert_eq!(
        buckets[5].1, 0,
        "the control's durable generation is 450 s old, so nothing may land at \
         or below the <=300s boundary; a sample there means the value is not \
         derived from last_checkpoint_ms"
    );
    assert!(
        buckets[6].1 >= 1,
        "the ~450 s sample must land in the <=600s cumulative bucket; got \
         {} of {total}",
        buckets[6].1
    );
}

// ────────────────────────────────────────────────────────────────────────
// Case H — a layer whose adapter always fails samples pending age, and the
//          per-tick max gauge CLIMBS across ticks.
//
// The assembler-`Err` arm is the one place where the DR gap is both real and
// growing, and before b.jq3 it was the one arm the histogram could not see:
// the old metric was observed on the skip and success arms only, so a layer
// wedged behind a poison payload or a broken adapter produced
// `writes_total{result="error"}` and nothing else. An operator watching
// staleness saw a healthy distribution while a layer drifted arbitrarily far
// out of DR.
//
// Two claims, and the second is why the gauge exists at all:
//
//   1. The error arm samples. Case A already pins the counter; this pins the
//      staleness signal.
//   2. `live_edit_checkpoint_oldest_pending_age_seconds` grows monotonically
//      while the layer stays stuck. A histogram p99 over a large fleet
//      structurally cannot surface one wedged layer; the max can, and its
//      growth is what distinguishes "stuck" from "briefly behind".
//
// The layer is given a 30 s-old durable generation so the first sample is
// already well clear of zero and the growth assertion has a clean baseline.
// ────────────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn case_h_failing_adapter_samples_pending_age_and_the_max_gauge_climbs() {
    const REPLICA: &str = "case-h-replica";
    /// Baseline staleness injected into the fixture.
    const DURABLE_AGE_MS: u64 = 30_000;
    /// How much the maximum must be seen to grow. A wall-clock quantity is
    /// unavoidable here: the thing under test IS "does this value increase as
    /// time passes while the layer stays stuck".
    const REQUIRED_GROWTH_SECS: f64 = 1.0;
    /// How long to wait for that growth. Deliberately far longer than the
    /// ~1 s it takes on an idle machine: this is a deadline, not a
    /// measurement window, so a starved runner takes longer to pass rather
    /// than failing a fixed-sleep-then-read.
    const GROWTH_DEADLINE: Duration = Duration::from_secs(10);

    let tv = TestValkey::setup_with_pool_size(8).await;
    let metrics = make_metrics();
    let oid = seed_layer(&tv, "wedged-layer").await;
    set_gate_timestamps(&tv, oid, 0, DURABLE_AGE_MS).await;

    // Never flipped off: this layer stays wedged for the whole test.
    let store: Arc<ErroringCheckpointStore> = Arc::new(ErroringCheckpointStore::new());
    let store_for_loop: Arc<dyn CheckpointStore> = store.clone();
    let cancel = CancellationToken::new();
    let config = CheckpointLoopConfig {
        tick: Duration::from_millis(50),
        interval: Duration::ZERO,
        quiesce: Duration::ZERO,
        retain_generations: 2,
        counters_sync: Duration::from_secs(3600),
        reconcile_interval: Duration::from_secs(3600),
        leader: crate::common::test_leader(REPLICA),
        tail: tails_off(),
    };
    let handle = spawn_checkpoint_loop(
        tv.client().clone(),
        tv.key_schema().clone(),
        store_for_loop,
        config,
        Some(metrics.clone()),
        cancel.clone(),
    );

    let metrics_for_poll = metrics.clone();
    let saw_error_sample =
        poll_until(Duration::from_secs(10), || {
            let m = metrics_for_poll.clone();
            async move {
                read_writes_counter(&m, "error") >= 1 && read_pending_age_sample_count(&m) >= 1
            }
        })
        .await;
    assert!(
        saw_error_sample,
        "a layer whose adapter keeps failing must produce BOTH an error tick \
         and a pending-age sample. Pre-b.jq3 it produced only the counter: a \
         failing adapter is the primary cause of checkpoint staleness and was \
         the one arm the staleness histogram could not see; got error={} \
         samples={}",
        read_writes_counter(&metrics, "error"),
        read_pending_age_sample_count(&metrics)
    );

    // The per-tick gauges are published only after the SCAN drains, so the
    // tick that bumped the counter above may not have reached the publish yet.
    let metrics_for_poll = metrics.clone();
    let published = poll_until(Duration::from_secs(10), || {
        let m = metrics_for_poll.clone();
        async move { read_oldest_pending_age(&m, REPLICA).is_some() }
    })
    .await;
    assert!(
        published,
        "the leader must publish the max gauge once a tick's SCAN drains"
    );
    let first_max = read_oldest_pending_age(&metrics, REPLICA)
        .expect("the leader must publish the max gauge on every completed tick");
    assert!(
        first_max >= (DURABLE_AGE_MS as f64) / 1000.0,
        "the wedged layer's durable generation is already {DURABLE_AGE_MS} ms \
         old, so the per-tick maximum must be at least that; got {first_max}s"
    );
    let samples_before = read_pending_age_sample_count(&metrics);

    // Let it stay stuck. Each tick re-reads the clock, so the published
    // maximum must track real elapsed time.
    //
    // Polled to a deadline rather than slept-then-read once: the claim is
    // "this value climbs", and a fixed sleep turns a runner that was descheduled
    // for a second — or a tick that has not published since the sleep ended —
    // into a failure of a claim that is actually true. The deadline preserves
    // the claim exactly (a gauge that never climbs still fails, after 10 s)
    // while letting a loaded machine take longer to demonstrate it.
    let metrics_for_poll = metrics.clone();
    let climbed = poll_until(GROWTH_DEADLINE, || {
        let m = metrics_for_poll.clone();
        async move {
            read_oldest_pending_age(&m, REPLICA).unwrap_or(0.0) >= first_max + REQUIRED_GROWTH_SECS
        }
    })
    .await;
    let second_max = read_oldest_pending_age(&metrics, REPLICA)
        .expect("the gauge must still be published after further ticks");
    assert!(
        climbed,
        "the per-tick maximum must CLIMB while a layer stays wedged — that \
         growth is what distinguishes a layer stuck out of DR from one briefly \
         behind, and it is the signal a fleet-wide p99 structurally cannot \
         show. Got {first_max}s then {second_max}s across a \
         {GROWTH_DEADLINE:?} deadline"
    );
    assert!(
        read_pending_age_sample_count(&metrics) > samples_before,
        "every failing tick must contribute another sample; the count stalled \
         at {samples_before}"
    );
    assert_eq!(
        read_layers_never_written(&metrics, REPLICA),
        Some(0),
        "this layer has a durable generation, so it is stale rather than \
         outside DR entirely; the never-written gauge must not double-count it"
    );

    cancel_and_join(cancel, handle).await;
}

// ────────────────────────────────────────────────────────────────────────
// Case I — `live_edit_checkpoint_layers_never_written` counts a pending layer
//          with NO durable generation, and stops counting it the moment its
//          first checkpoint lands.
//
// "Never checkpointed" is a categorically different operator situation from
// "stale by N seconds": the layer is outside DR entirely, not merely behind.
// It also has no finite age — `now - 0` is ~1.77e9 seconds — so folding it
// into the histogram would peg every sample into `+Inf` and poison `_sum`,
// which is exactly the defect shape b.yut fixed on the commit clock. It gets
// its own counting gauge instead, and the histogram must never see it.
//
// The second half is the subtle one. A layer whose FIRST EVER checkpoint
// succeeds on this tick had no durable generation when the loop measured it,
// but has one now, and the gauge reports layers outside DR *as of the end of
// the tick*. Counting the first successful write would make the gauge
// non-zero on every healthy fleet that is creating layers — reproducing, on a
// new instrument, the very false-positive class b.jq3 exists to remove.
//
// The layer is held in the never-written state by an adapter that fails, then
// released by flipping the adapter to succeed inside the SAME running loop —
// so the before/after comparison cannot be confounded by a leader-lock
// handover.
// ────────────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn case_i_never_written_gauge_counts_undr_layers_and_clears_on_first_write() {
    const REPLICA: &str = "case-i-replica";
    let tv = TestValkey::setup_with_pool_size(8).await;
    let metrics = make_metrics();
    let oid = seed_layer(&tv, "never-written-layer").await;

    // Deliberately NO `last_checkpoint_ms` injection: absent → 0 → "never
    // checkpointed", which is also what makes the interval arm due on the
    // first tick.
    let header_key = tv.key_schema().obj_header(&oid.to_string());
    let stamp: Option<String> = tv
        .client()
        .pool
        .hget(&header_key, "last_checkpoint_ms")
        .await
        .expect("HGET last_checkpoint_ms must succeed");
    assert!(
        stamp.is_none(),
        "the fixture depends on a freshly created layer having NO durable \
         generation stamp; got {stamp:?}"
    );

    let store: Arc<ErroringCheckpointStore> = Arc::new(ErroringCheckpointStore::new());
    let store_for_loop: Arc<dyn CheckpointStore> = store.clone();
    let cancel = CancellationToken::new();
    let config = CheckpointLoopConfig {
        tick: Duration::from_millis(50),
        interval: Duration::ZERO,
        quiesce: Duration::ZERO,
        retain_generations: 2,
        counters_sync: Duration::from_secs(3600),
        reconcile_interval: Duration::from_secs(3600),
        leader: crate::common::test_leader(REPLICA),
        tail: tails_off(),
    };
    let handle = spawn_checkpoint_loop(
        tv.client().clone(),
        tv.key_schema().clone(),
        store_for_loop,
        config,
        Some(metrics.clone()),
        cancel.clone(),
    );

    // Phase 1 — held outside DR by the failing adapter.
    let metrics_for_poll = metrics.clone();
    let counted = poll_until(Duration::from_secs(10), || {
        let m = metrics_for_poll.clone();
        async move { read_layers_never_written(&m, REPLICA) == Some(1) }
    })
    .await;
    assert!(
        counted,
        "a layer with un-persisted commits and no durable generation must be \
         counted on live_edit_checkpoint_layers_never_written; got {:?}",
        read_layers_never_written(&metrics, REPLICA)
    );
    assert_eq!(
        read_pending_age_sample_count(&metrics),
        0,
        "a never-checkpointed layer has no finite age; `now - 0` is ~1.77e9 \
         seconds, and folding that sentinel into the histogram would peg every \
         sample into +Inf and poison _sum — the b.yut defect shape, under a new \
         metric name"
    );
    assert_eq!(
        read_oldest_pending_age(&metrics, REPLICA),
        Some(0.0),
        "for the same reason, a never-written layer must not move the max \
         gauge — the sentinel would pin it at ~1.77e9 forever"
    );

    // Phase 2 — its first checkpoint lands, in the same loop.
    store.stop_failing();
    let metrics_for_poll = metrics.clone();
    let cleared = poll_until(Duration::from_secs(10), || {
        let m = metrics_for_poll.clone();
        async move {
            read_writes_counter(&m, "success") >= 1
                && read_layers_never_written(&m, REPLICA) == Some(0)
        }
    })
    .await;
    cancel_and_join(cancel, handle).await;
    assert!(
        cleared,
        "once the layer's FIRST checkpoint succeeds it is inside DR as of the \
         end of that tick and must stop being counted; got success={} \
         never_written={:?}. Counting a first successful write would make this \
         gauge non-zero on every healthy fleet that creates layers",
        read_writes_counter(&metrics, "success"),
        read_layers_never_written(&metrics, REPLICA)
    );
    assert_eq!(
        read_pending_age_sample_count(&metrics),
        0,
        "the first-ever write has no finite PRE-write age to report either, so \
         the histogram stays empty across this whole test; got {} sample(s)",
        read_pending_age_sample_count(&metrics)
    );
}

// ────────────────────────────────────────────────────────────────────────
// Case J — a replica that does NOT hold the leader lock publishes zeros on
//          both per-tick gauges, without disturbing the leader's own series.
//
// An OTel `Gauge` retains its last recorded value indefinitely — no decay, no
// staleness marker. A replica that held the lock, observed a large maximum,
// and then lost it would keep exporting that maximum forever. Since the
// fleet-level query is `max()` across replicas, one demoted leader's
// fossilised value would pin the whole fleet's reading high and defeat the
// alert. Publishing zero is the truthful reading for a replica that evaluated
// no layers, and it is what lets `max()` be correct.
//
// Driven end-to-end rather than through the accumulator: the leader-probe
// `Ok(false)` arm is loop control flow, and the claim is about what a SECOND
// replica exports while a first one is mid-incident. Loop A is wedged on a
// failing adapter against a 600 s-stale layer, so its maximum is large and
// unmistakable; loop B joins against the same Valkey, loses the lock race,
// and must publish its own zeroed series alongside — not instead of — A's.
//
// Not covered here: the leader-probe `Err(_)` arm, which publishes NOTHING
// (an unknown leader state is not evidence of a healthy fleet, and zero would
// assert one). Reaching it needs a Valkey that fails only the lock EVAL while
// the rest of the loop keeps working, which no seam in this crate exposes.
// ────────────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn case_j_non_leader_replica_publishes_zeroed_pending_aggregates() {
    const LEADER: &str = "case-j-leader";
    const FOLLOWER: &str = "case-j-follower";
    /// Staleness injected into the leader's view. Far enough above zero that
    /// the follower's published zero cannot be confused for it.
    const DURABLE_AGE_MS: u64 = 600_000;

    let tv = TestValkey::setup_with_pool_size(8).await;
    let metrics = make_metrics();
    let oid = seed_layer(&tv, "leader-view-layer").await;
    set_gate_timestamps(&tv, oid, 0, DURABLE_AGE_MS).await;

    // Leader: wedged adapter keeps the layer pending tick after tick, so its
    // maximum stays large for the whole test.
    let leader_store: Arc<dyn CheckpointStore> = Arc::new(ErroringCheckpointStore::new());
    let leader_cancel = CancellationToken::new();
    let leader_handle = spawn_checkpoint_loop(
        tv.client().clone(),
        tv.key_schema().clone(),
        leader_store,
        CheckpointLoopConfig {
            tick: Duration::from_millis(50),
            interval: Duration::ZERO,
            quiesce: Duration::ZERO,
            retain_generations: 2,
            counters_sync: Duration::from_secs(3600),
            reconcile_interval: Duration::from_secs(3600),
            leader: crate::common::test_leader(LEADER),
            tail: tails_off(),
        },
        Some(metrics.clone()),
        leader_cancel.clone(),
    );

    // Wait until the leader is demonstrably the leader before the follower
    // starts, so the lock race has exactly one possible outcome.
    let metrics_for_poll = metrics.clone();
    let leader_reporting = poll_until(Duration::from_secs(10), || {
        let m = metrics_for_poll.clone();
        async move { read_oldest_pending_age(&m, LEADER).unwrap_or(0.0) >= 300.0 }
    })
    .await;
    assert!(
        leader_reporting,
        "the first loop must acquire the lock and report the wedged layer's \
         ~{}s staleness before the follower starts; got {:?}",
        DURABLE_AGE_MS / 1000,
        read_oldest_pending_age(&metrics, LEADER)
    );

    // Follower: same Valkey, different replica id, so every leader probe
    // returns Ok(false) and it runs no tick at all.
    let follower_store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
    let follower_cancel = CancellationToken::new();
    let follower_handle = spawn_checkpoint_loop(
        tv.client().clone(),
        tv.key_schema().clone(),
        follower_store,
        CheckpointLoopConfig {
            tick: Duration::from_millis(50),
            interval: Duration::ZERO,
            quiesce: Duration::ZERO,
            retain_generations: 2,
            counters_sync: Duration::from_secs(3600),
            reconcile_interval: Duration::from_secs(3600),
            leader: crate::common::test_leader(FOLLOWER),
            tail: tails_off(),
        },
        Some(metrics.clone()),
        follower_cancel.clone(),
    );

    let metrics_for_poll = metrics.clone();
    let follower_published = poll_until(Duration::from_secs(10), || {
        let m = metrics_for_poll.clone();
        async move { read_oldest_pending_age(&m, FOLLOWER).is_some() }
    })
    .await;
    assert!(
        follower_published,
        "a non-leader replica must PUBLISH its zeroed aggregates rather than \
         stay silent: an OTel gauge never decays, so a replica that goes quiet \
         after losing the lock keeps exporting its last observed maximum \
         forever and pins the fleet-wide max() high"
    );

    assert_eq!(
        read_oldest_pending_age(&metrics, FOLLOWER),
        Some(0.0),
        "a replica that evaluated no layers must publish 0.0 — that is the \
         truthful value, and it is what makes max() across replicas correct"
    );
    assert_eq!(
        read_layers_never_written(&metrics, FOLLOWER),
        Some(0),
        "both aggregates are zeroed on the non-leader arm, not just the max"
    );

    // The leader's series is untouched by the follower's zero: they are
    // distinct `service` label values, which is why the documented aggregation
    // is max() and never sum().
    let leader_max = read_oldest_pending_age(&metrics, LEADER).unwrap_or(0.0);
    assert!(
        leader_max >= 300.0,
        "the follower's zero must not erase or dilute the leader's reading; \
         the leader still sees a wedged layer but reported {leader_max}s"
    );

    cancel_and_join(follower_cancel, follower_handle).await;
    cancel_and_join(leader_cancel, leader_handle).await;
}

// ────────────────────────────────────────────────────────────────────────
// Case K — the SkippedNotQuiesced arm for a layer that HAS a durable
//          generation: it samples the histogram off that generation's age and
//          feeds the per-tick maximum.
//
// Case D drives the same arm, but its fixture has never been checkpointed, so
// the layer is routed to `layers_never_written` and touches neither the
// histogram nor the max. That leaves the arm's *other* shape untested: a layer
// that was checkpointed at some point and whose ticks have since been skipped
// (a header-sequence regression). Such a layer is stale by a measurable amount
// and drifting further every tick, which is precisely the population the
// histogram and the max gauge exist to describe.
//
// The fixture is Case D's hot writer over a layer whose `last_checkpoint_ms`
// is backdated 7 s, with the loop on Case D's `unprovable_cut_client` (see
// Case D's banner for why the arm needs it, and for the feature gate).
// That 7 s sits between two bucket boundaries, so a sample
// derived from anything else — the commit clock (stamped at ~now by the
// writer), or a post-write recomputation — lands in `<=5` and fails.
//
// Note the fixture deliberately stamps `last_checkpoint_ms` WITHOUT a matching
// `last_checkpoint_seq`. Case D asserts that pair is never observed
// half-written *by the loop*, and it owns that invariant; here the half-pair is
// injected on purpose, to hold the layer both pending (no cursor) and
// previously-checkpointed (a cadence stamp) at once. The two tests must not be
// merged for exactly this reason.
//
// The seam skips every assembly of the layer, so no tick verifies — asserted,
// as in Case D — and the injected cadence stamp is never refreshed: the layer
// stays pending, aging from the injected 7 s, for the whole run.
// ────────────────────────────────────────────────────────────────────────

#[cfg(feature = "test-checkpoint-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn case_k_unprovable_cut_on_a_stale_layer_samples_pending_age_and_the_max() {
    use oodf_ffi::{CommandCreateSection, OwnedDelta, StructCommand};

    const REPLICA: &str = "case-k-replica";
    /// Age of the injected durable generation. Strictly between the `<=5` and
    /// `<=15` bucket boundaries, so bucket membership proves provenance.
    const DURABLE_AGE_MS: u64 = 7_000;

    let tv = TestValkey::setup_with_pool_size(8).await;
    let metrics = make_metrics();
    let layer_name = "stale-not-quiesced-layer";
    let (oid, session_id) = seed_wide_layer(&tv, layer_name).await;

    // A real cadence stamp, and a commit clock at ~now. The commit clock is
    // then continuously refreshed by the writer below, so a sample taken off
    // the commit clock would land near zero.
    set_gate_timestamps(&tv, oid, 0, DURABLE_AGE_MS).await;

    let stop = Arc::new(AtomicBool::new(false));
    let commits = Arc::new(AtomicUsize::new(0));
    let writer = {
        let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
        let stop = stop.clone();
        let commits = commits.clone();
        let session_id = session_id.clone();
        tokio::spawn(async move {
            let mut next_section_id = 20_000u64;
            while !stop.load(Ordering::Relaxed) {
                let delta = OwnedDelta {
                    struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
                        parent_id: 0,
                        section_id: next_section_id,
                        section_name: format!("hot-{next_section_id}"),
                        section_type: 1,
                    })],
                    ..Default::default()
                };
                if ls
                    .update_layer_relayed(
                        &session_id,
                        layer_name,
                        oid,
                        Some(delta),
                        "hot-writer",
                        0,
                    )
                    .await
                    .is_err()
                {
                    // The pool is torn down at test exit; stop quietly.
                    return;
                }
                commits.fetch_add(1, Ordering::Relaxed);
                next_section_id += 1;
            }
        })
    };

    // Both gates open so every tick reaches the assembler; the seam skips
    // every assembly of the layer (Case D's banner).
    let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
    let cancel = CancellationToken::new();
    let handle = spawn_checkpoint_loop(
        unprovable_cut_client(&tv, oid),
        tv.key_schema().clone(),
        store,
        CheckpointLoopConfig {
            tick: Duration::from_millis(50),
            interval: Duration::ZERO,
            quiesce: Duration::ZERO,
            retain_generations: 2,
            counters_sync: Duration::from_secs(3600),
            reconcile_interval: Duration::from_secs(3600),
            leader: crate::common::test_leader(REPLICA),
            tail: tails_off(),
        },
        Some(metrics.clone()),
        cancel.clone(),
    );

    let metrics_for_poll = metrics.clone();
    let saw_unverified = poll_until(Duration::from_secs(20), || {
        let m = metrics_for_poll.clone();
        async move { read_stable_cut_counter(&m, "unverified", REPLICA) >= 1 }
    })
    .await;
    assert!(
        saw_unverified,
        "a layer whose proven read the seam has judged regressed must produce at \
         least one outcome=\"unverified\" stable-cut tick within 20 s; observed {} \
         writer commits",
        commits.load(Ordering::Relaxed)
    );

    // Cancel while the writer is still hammering and the seam armed, so the
    // layer is pending when the last published tick ends.
    cancel_and_join(cancel, handle).await;
    stop.store(true, Ordering::Relaxed);
    writer.await.expect("hot-writer task must not panic");

    let verified = read_stable_cut_counter(&metrics, "verified", REPLICA);
    let unverified = read_stable_cut_counter(&metrics, "unverified", REPLICA);
    assert_eq!(
        verified, 0,
        "the seam skips every assembly of this layer, so no tick may verify; \
         unverified={unverified}"
    );

    // The histogram half. Stated as "at least one sample landed strictly
    // between the two boundaries": the FIRST observation of this layer is taken
    // against the injected 7 s stamp, and on a slow run later ones age past
    // 15 s, so the (5, 15] band is the one every run must reach.
    let (buckets, total, _) = pending_age_histogram(&metrics);
    assert!(
        total >= 1,
        "a layer the assembler refused to prove must still sample pending age: \
         it persisted nothing by design, so its commits are outside DR and it \
         is exactly as stale as any other pending layer ({verified} verified / \
         {unverified} unverified ticks). Silence here is the blind spot b.jq3 \
         closed"
    );
    let le5 = buckets[0].1;
    let le15 = buckets[1].1;
    assert!(
        // `le15 > le5`, i.e. at least one sample in the (5, 15] band.
        le15 > le5,
        "at least one sample must land strictly between the <=5s and <=15s \
         boundaries — that is the injected {DURABLE_AGE_MS} ms durable stamp. \
         A run where every sample sits at or below 5 s means the arm measured \
         the commit clock (stamped at ~now by the hot writer) or recomputed the \
         age after a write, not last_checkpoint_ms; got le5={le5} le15={le15} \
         of {total}"
    );

    // The never-written gauge must stay clear: this layer HAS a durable
    // generation for the whole test — the injection guarantees it — so the two
    // populations never overlap.
    assert_eq!(
        read_layers_never_written(&metrics, REPLICA),
        Some(0),
        "a layer with a durable generation is stale, not outside DR entirely; \
         counting it on the never-written gauge would double-count it against \
         the histogram sample above"
    );

    // The max-gauge half, and the R2 semantics that make it meaningful: the
    // gauge is an END-OF-TICK-PENDING maximum, and a layer this arm skipped is
    // still pending when the tick ends, so it MUST set the maximum. (A layer
    // the success arm wrote must not — that asymmetry is pinned on the
    // accumulator itself in `checkpointer.rs::pending_age_tests`.)
    let oldest = read_oldest_pending_age(&metrics, REPLICA)
        .expect("the leader must publish the max gauge once a tick's SCAN drains");
    assert!(
        oldest >= (DURABLE_AGE_MS as f64) / 1000.0,
        "the SkippedNotQuiesced arm leaves the layer pending at the end of the \
         tick, and no tick ever wrote, so the published maximum must be at least \
         the layer's injected {DURABLE_AGE_MS} ms age; got {oldest}s. A zero here \
         means the arm records nothing and one wedged-hot layer is invisible to \
         the gauge operators page on"
    );
}
