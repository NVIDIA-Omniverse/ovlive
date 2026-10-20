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

// Integration tests for the server-side UpdateLayer retry loop (b.vi2).
//
// The engineer added a bounded retry around the WATCH/MULTI/EXEC sequence in
// `LayerStore::update_layer` so that transient WATCH-cancel aborts are
// retried transparently with full-jitter backoff. These tests lock in:
//
//   1. Under high contention (N=50 writers hammering one object) every writer
//      commits — recovering from the ~21% pre-fix ceiling — and the retry loop
//      is provably the mechanism (retries were taken, none exhausted). The race
//      itself is the shared `update_layer_transaction_test::
//      run_fifty_writer_race` scenario, so the commit-rate contract is stated
//      once for the whole suite rather than once per copy of the race.
//   2. When the retry budget is exhausted, `update_layer` returns
//      `ServiceError::Aborted` (retry exhaustion) whose message includes the
//      retry count ("N attempts exhausted").
//   3. Metrics fire: `update_layer_retries_total{committed|exhausted}` counts
//      every retry attempt taken across all update_layer calls (a commit on
//      attempt N contributes N-1 to `committed`; an exhaustion contributes
//      `max_attempts - 1` to `exhausted`). `update_layer_attempts_histogram`
//      records the per-commit attempt count.
//
// Tests run against a real Valkey instance via the TestValkey fixture and
// use per-test key prefixes so they can run in parallel alongside the rest
// of the storage suite.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use bytes::Bytes;
use fred::interfaces::StreamsInterface;

use live_edit::config::UpdateLayerBackend;
use live_edit::domain::types::ObjectState;
use live_edit::error::ServiceError;
use live_edit::observability::metrics::Metrics;
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::relay::RelayedUpdates;
use live_edit::storage::test_support::TestValkey;
use oodf_ffi::OwnedDelta;

fn mock_ext_store() -> Arc<MockExternalValueStore> {
    Arc::new(MockExternalValueStore::new())
}

/// Build a LayerStore pinned to the legacy `UpdateLayerBackend::Txn` path
/// with the default retry budget (32). b.67u made `Lua` the default; these
/// tests exercise WATCH/MULTI/EXEC retry semantics (WATCH-conflict retries,
/// WATCH-path fred-error injection) which do not apply to the Lua fast
/// path, so the Txn path is the correct backend to pin here.
fn layer_store_default(tv: &TestValkey) -> LayerStore {
    LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_backend(UpdateLayerBackend::Txn)
}

/// Build a LayerStore with a caller-controlled retry budget and an attached
/// Metrics instance, pinned to `UpdateLayerBackend::Txn` for the same
/// reason as `layer_store_default`.
fn layer_store_with(tv: &TestValkey, max_attempts: u32, metrics: Arc<Metrics>) -> LayerStore {
    LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_retry_max_attempts(max_attempts)
        .with_update_layer_backend(UpdateLayerBackend::Txn)
        .with_metrics(metrics)
}

async fn create_test_session(tv: &TestValkey) -> String {
    let ss = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    ss.create("test-session", "/stage", "test-user")
        .await
        .expect("session creation must succeed")
}

async fn create_test_layer(ls: &LayerStore, session_id: &str, layer_name: &str) -> u64 {
    let state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usd".to_string(),
        ..Default::default()
    };
    ls.create_layer(
        session_id,
        layer_name,
        "test-user",
        &state,
        mock_ext_store(),
    )
    .await
    .expect("create_layer must succeed")
}

fn build_valid_delta() -> OwnedDelta {
    OwnedDelta::default()
}

/// The `path` label every store in this file commits on: each one is pinned to
/// `UpdateLayerBackend::Txn`, so every attempt takes the WATCH slow path
/// (`path="watch"`), and the helpers below read that path's series only.
const WATCH_PATH: &str = "watch";

/// Read `live_edit_update_layer_retries_total{path="watch",outcome}` out of the
/// rendered scrape exposition. Returns 0 if the series is missing.
/// `update_layer_retries_total` pre-initialises every `(path, outcome)` cell,
/// so this returns 0 without panicking for unseen outcomes.
fn retry_counter(metrics: &Metrics, outcome: &str) -> u64 {
    use live_edit::observability::metrics::test_util::{render, sample_value};
    sample_value(
        &render(metrics),
        "live_edit_update_layer_retries_total",
        &[("path", WATCH_PATH), ("outcome", outcome)],
    )
    .unwrap_or(0.0) as u64
}

/// Render the `path="watch"` attempts histogram and return (sample_count,
/// bucket_counts). `bucket_counts` maps `le` upper-bound → cumulative count so
/// callers can assert specific buckets observed non-zero values. `+Inf` maps to
/// `f64::INFINITY`.
fn gather_attempts_histogram(metrics: &Metrics) -> (u64, Vec<(f64, u64)>) {
    use live_edit::observability::metrics::test_util::{render, sample_value};
    let text = render(metrics);
    let count = sample_value(
        &text,
        "live_edit_update_layer_attempts_histogram_count",
        &[("path", WATCH_PATH)],
    )
    .unwrap_or(0.0) as u64;
    let path_label = format!("path=\"{WATCH_PATH}\"");
    let mut buckets = Vec::new();
    for line in text.lines() {
        let Some(rest) = line.strip_prefix("live_edit_update_layer_attempts_histogram_bucket")
        else {
            continue;
        };
        if !rest.contains(&path_label) {
            continue;
        }
        let Some(le_at) = rest.find("le=\"") else {
            continue;
        };
        let after = &rest[le_at + 4..];
        let Some(end) = after.find('"') else {
            continue;
        };
        let bound = if &after[..end] == "+Inf" {
            f64::INFINITY
        } else {
            after[..end].parse::<f64>().unwrap_or(f64::NAN)
        };
        let val = line
            .rsplit_once(char::is_whitespace)
            .and_then(|(_, v)| v.trim().parse::<u64>().ok())
            .unwrap_or(0);
        buckets.push((bound, val));
    }
    (count, buckets)
}

/// Sum of `attempt` values observed in the `path="watch"` attempts histogram.
/// Total retries across all successful commits is `histogram_sum -
/// sample_count`.
fn histogram_sum(metrics: &Metrics) -> f64 {
    use live_edit::observability::metrics::test_util::{render, sample_value};
    sample_value(
        &render(metrics),
        "live_edit_update_layer_attempts_histogram_sum",
        &[("path", WATCH_PATH)],
    )
    .unwrap_or(0.0)
}

// =============================================================================
// 1. Regression test: N=50 concurrent writers all commit, by consuming retries
// =============================================================================
//
// Pre-fix, `update_layer` returned `Aborted` on every WATCH-cancel — observed
// success rate at N=50 was ~21%. With the retry loop (default 32 attempts,
// full-jitter backoff), every writer commits.
//
// This test used to own a second, hand-written copy of that race with its own
// `>= 99%` tolerance (as `update_layer_fifty_concurrent_writers_commit_rate_
// at_least_99pct`), which drifted against the identical scenario in
// `update_layer_transaction_test` — same 50 writers, same object, same backend,
// but 0 aborts tolerated there and 1 tolerated here. It now drives the SAME
// helper, so the suite states one contract in one place and this test adds only
// what it uniquely claims: that the retry loop is the mechanism.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn update_layer_fifty_concurrent_writers_commit_by_consuming_retries() {
    const WRITERS: usize = 50;

    // The shared scenario. Every commit-rate assertion lives inside it: all 50
    // writers commit (zero aborts), on distinct gap-free sequences, each with
    // its stream entry, with the three race-guard counters at zero.
    let (_tv, _oid, metrics) = super::update_layer_transaction_test::run_fifty_writer_race(
        "retry-regression",
        Vec::new(),
        (0..WRITERS).map(|_| build_valid_delta()).collect(),
        "50 empty-delta writers (retry-loop claims)",
    )
    .await;

    // ── What only this test claims: the retry loop did the work ──────────────
    //
    // "All 50 committed" is compatible with two very different runs: one where
    // the writers never actually overlapped (a vacuous pass — the barrier
    // failed to create contention, or a future change serialized them upstream)
    // and one where the retry loop absorbed real WATCH-cancel losses. The
    // histogram is what tells them apart, and it is the reason this test is not
    // redundant with its sibling.
    let (samples, _buckets) = gather_attempts_histogram(&metrics);
    assert_eq!(
        samples, WRITERS as u64,
        "the attempts histogram must carry exactly one sample per COMMIT — it is observed once \
         per successful `update_layer`, so a different count means the run was not the {WRITERS} \
         commits the shared helper just asserted"
    );

    let total_attempts = histogram_sum(&metrics);
    assert!(
        total_attempts > samples as f64,
        "no writer retried: {WRITERS} commits took {total_attempts} attempts in total, i.e. every \
         one committed first try. With {WRITERS} writers released from one barrier onto a single \
         WATCHed header, losses are structural — so this reads as the contention never happening, \
         which would make the commit-rate assertions in the shared helper vacuous against the \
         pre-b.vi2 ~21% ceiling they exist to fence"
    );
    assert_eq!(
        retry_counter(&metrics, "committed") as f64,
        total_attempts - samples as f64,
        "`update_layer_retries_total{{outcome=\"committed\"}}` counts every retry TAKEN, so it \
         must equal (total attempts − commits) exactly: a commit on attempt N contributes N−1. A \
         mismatch means the counter and the histogram disagree about the same run, and an \
         operator reading a retry-rate dashboard would be reading a number nothing produces"
    );
    assert_eq!(
        retry_counter(&metrics, "exhausted"),
        0,
        "no writer may exhaust its budget — this is the same fact as the helper's zero-abort \
         assertion, restated on the metric an operator actually alerts on. A nonzero here with \
         zero aborts would mean the counter fires on commits"
    );
}

// =============================================================================
// 2. Retry exhaustion: low retry budget + contention → `Aborted` "attempts exhausted"
// =============================================================================
//
// Configure the LayerStore with `max_attempts=2` so every writer after the
// first commit in a given contention window exhausts its retry budget
// quickly. With N=10 writers and only 2 attempts each, at least one writer
// must return `Aborted` with a message matching "attempts exhausted" — the
// shared retry loop's exhaustion-path literal (`update_layer` runs it as
// `retry_aborted_with_jitter_and_uncounted_waits`) carrying the attempt count
// (b.gsd; same wording as `SessionStore::create`).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn update_layer_retry_exhaustion_surfaces_exhausted_error_message() {
    const WRITERS: usize = 10;
    const MAX_ATTEMPTS: u32 = 2;
    let pool_size = (WRITERS as u32) + 4;

    let tv = TestValkey::setup_with_pool_size(pool_size).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("retry-test"));
    let ls = layer_store_with(&tv, MAX_ATTEMPTS, metrics.clone());
    let session_id = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &session_id, "retry-exhaustion").await;

    let delta = build_valid_delta();
    let barrier = Arc::new(tokio::sync::Barrier::new(WRITERS));

    let mut handles = Vec::with_capacity(WRITERS);
    for i in 0..WRITERS {
        let ls_i = ls.clone();
        let sid_i = session_id.clone();
        let d_i = delta.clone();
        let barrier_i = barrier.clone();
        let client_id = format!("writer-{i}");
        let client_ts = 1000u64 + i as u64;
        handles.push(tokio::spawn(async move {
            barrier_i.wait().await;
            ls_i.update_layer_relayed(
                &sid_i,
                "retry-exhaustion",
                oid,
                Some(d_i),
                &client_id,
                client_ts,
            )
            .await
        }));
    }

    let mut successes: u32 = 0;
    let mut exhausted_msgs: Vec<String> = Vec::new();
    let mut other_errors: Vec<String> = Vec::new();
    for h in handles {
        match h.await.expect("writer task must not panic") {
            Ok(_) => successes += 1,
            Err(ServiceError::Aborted(msg)) => exhausted_msgs.push(msg),
            Err(e) => other_errors.push(format!("{e:?}")),
        }
    }

    assert!(
        other_errors.is_empty(),
        "unexpected non-Aborted errors from contended writers: {other_errors:?}"
    );

    // With a 2-attempt budget across 10 racing writers, we expect at least
    // one writer to exhaust its budget and surface the exhaustion error.
    // The engineer's literal is `"concurrent write conflict, {N} retries
    // exhausted"` with N = MAX_ATTEMPTS.
    assert!(
        !exhausted_msgs.is_empty(),
        "expected at least one writer to exhaust its retry budget at max_attempts={MAX_ATTEMPTS} \
         across {WRITERS} writers; got {successes} successes and 0 aborts"
    );

    for msg in &exhausted_msgs {
        assert!(
            msg.contains("attempts exhausted"),
            "exhaustion-path Aborted message must contain the 'attempts exhausted' literal \
             (clients and operators rely on this to distinguish exhaustion from one-shot \
              conflicts); got: {msg:?}"
        );
        assert!(
            msg.contains(&MAX_ATTEMPTS.to_string()),
            "exhaustion-path Aborted message must include the configured attempt count \
             ({MAX_ATTEMPTS}) so operators can correlate with LIVE_EDIT_UPDATE_LAYER_RETRY_MAX_ATTEMPTS; \
             got: {msg:?}"
        );
    }

    // At least one writer must have committed — the exhaustion path only
    // fires if someone else won the race, so a total wipeout would mean
    // even the first attempt never committed, which is an unrelated bug.
    assert!(
        successes >= 1,
        "at least one writer must commit (retry exhaustion is always relative to a winner); \
         got {successes} successes and {} exhausted",
        exhausted_msgs.len()
    );

    // Exhaustion path should also fire the `exhausted` retry counter.
    // Counter semantic: each exhausted call adds (MAX_ATTEMPTS - 1) retries,
    // since exhaustion fires at attempt == max_attempts and that many retries
    // were taken before the caller gave up.
    let exhausted_count = retry_counter(&metrics, "exhausted");
    let expected_exhausted = exhausted_msgs.len() as u64 * (MAX_ATTEMPTS as u64 - 1);
    assert_eq!(
        exhausted_count,
        expected_exhausted,
        "update_layer_retries_total{{outcome=\"exhausted\"}} must equal \
         exhausted_writers * (MAX_ATTEMPTS - 1); got counter={exhausted_count}, \
         exhausted_msgs={}, MAX_ATTEMPTS={MAX_ATTEMPTS}",
        exhausted_msgs.len()
    );
}

// =============================================================================
// 3. Metrics test: `update_layer_retries_total{committed}` increments on a
//    retry that eventually succeeds, and the attempts histogram records.
// =============================================================================
//
// Running N=20 writers with the default retry budget (32) produces near-100%
// commit rate, where most commits either succeed on the first attempt or
// required retries. We assert that the `committed` counter is strictly
// greater than zero (i.e. at least some writers needed a retry) and that
// `committed + retry_counter_exhausted == 0` is not the case.
//
// We also gather the attempts histogram and assert its sample count equals
// the number of successful commits — proving the observation wiring fires
// exactly once per commit.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn update_layer_retry_metrics_track_committed_and_attempts() {
    const WRITERS: usize = 20;
    let pool_size = (WRITERS as u32) + 4;

    let tv = TestValkey::setup_with_pool_size(pool_size).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("retry-metrics"));
    // Default retry budget (32) so all writers commit — lets us assert the
    // `committed` counter advances without racing the `exhausted` path.
    let ls = layer_store_with(&tv, 32, metrics.clone());
    let session_id = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &session_id, "retry-metrics").await;

    let delta = build_valid_delta();
    let barrier = Arc::new(tokio::sync::Barrier::new(WRITERS));

    let mut handles = Vec::with_capacity(WRITERS);
    for i in 0..WRITERS {
        let ls_i = ls.clone();
        let sid_i = session_id.clone();
        let d_i = delta.clone();
        let barrier_i = barrier.clone();
        let client_id = format!("writer-{i}");
        let client_ts = 1000u64 + i as u64;
        handles.push(tokio::spawn(async move {
            barrier_i.wait().await;
            ls_i.update_layer_relayed(
                &sid_i,
                "retry-metrics",
                oid,
                Some(d_i),
                &client_id,
                client_ts,
            )
            .await
        }));
    }

    let mut successes: u32 = 0;
    let mut other_errors: Vec<String> = Vec::new();
    for h in handles {
        match h.await.expect("writer task must not panic") {
            Ok(_) => successes += 1,
            Err(ServiceError::Aborted(_)) => {
                // Should not happen at default budget with N=20; flag loudly.
                panic!(
                    "unexpected Aborted at default retry budget (32) and N=20 writers — \
                     suggests the retry loop isn't absorbing WATCH conflicts"
                );
            }
            Err(e) => other_errors.push(format!("{e:?}")),
        }
    }
    assert!(
        other_errors.is_empty(),
        "unexpected non-Aborted errors from retry-metrics test: {other_errors:?}"
    );
    assert_eq!(
        successes, WRITERS as u32,
        "with default retry budget of 32, every writer must commit; got {successes}/{WRITERS}"
    );

    // Retry counters.
    let committed = retry_counter(&metrics, "committed");
    let exhausted = retry_counter(&metrics, "exhausted");
    assert_eq!(
        exhausted, 0,
        "no writer should exhaust at default budget; got exhausted={exhausted}"
    );
    assert!(
        committed >= 1,
        "at least one writer must have needed a retry to commit under {WRITERS}-way contention; \
         if committed=0, the test isn't exercising the retry path at all — got committed={committed}"
    );
    // Counter semantic: `committed` counts every retry attempt taken across
    // all successful commits (not once per retrying call). Exact relation:
    // `committed == sum(attempts over commits) - number_of_commits`, i.e.
    // histogram_sum - sample_count.
    let h_sum = histogram_sum(&metrics);
    let expected_committed = (h_sum as u64) - WRITERS as u64;
    assert_eq!(
        committed, expected_committed,
        "committed retries must equal total_attempts - successful_commits \
         (histogram.sum - histogram.count); got committed={committed}, \
         expected={expected_committed} (h_sum={h_sum}, WRITERS={WRITERS})"
    );

    // Attempts histogram — one observation per successful commit. The
    // engineer records the attempt index (1 = first-try success, N = Nth
    // attempt) so the sample count must equal the number of Ok() returns.
    let (sample_count, buckets) = gather_attempts_histogram(&metrics);
    assert_eq!(
        sample_count, WRITERS as u64,
        "attempts histogram must record exactly one observation per successful commit; \
         got sample_count={sample_count}, WRITERS={WRITERS}"
    );

    // The first bucket upper-bound is 1.0 (first-try successes). At least
    // one of the 20 writers should be a first-try success; at least one
    // should have required more than one attempt (since we saw committed >=
    // 1). The bucket 1.0 count should therefore be strictly between 0 and
    // WRITERS — a partial-bucket signal that the histogram resolves retry
    // counts, not just total commits.
    let bucket_1 = buckets
        .iter()
        .find(|(ub, _)| (*ub - 1.0).abs() < f64::EPSILON)
        .map(|(_, c)| *c)
        .expect("histogram must have a 1.0 bucket (first-try successes)");
    // bucket_1 is the cumulative count of observations ≤ 1.0. With committed
    // ≥ 1 retries, at least one sample must land above 1.0, so bucket_1
    // must be strictly less than sample_count.
    assert!(
        bucket_1 < sample_count,
        "attempts histogram bucket ≤1 must be strictly less than sample_count when retries \
         occurred (committed={committed}); got bucket_1={bucket_1}, sample_count={sample_count}"
    );
}

// =============================================================================
// 4. Metrics test (exhaustion path): `update_layer_attempts_histogram` does
//    NOT record for exhausted writers — only committed writers get an
//    observation. This pins the semantics of the histogram: "attempts per
//    successful commit", not "attempts per call".
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn update_layer_attempts_histogram_records_only_on_commit_not_exhaustion() {
    const WRITERS: usize = 8;
    const MAX_ATTEMPTS: u32 = 1;
    let pool_size = (WRITERS as u32) + 4;

    let tv = TestValkey::setup_with_pool_size(pool_size).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("histo-exhaust"));
    // max_attempts=1 means zero retries — one shot, exhaustion on any
    // WATCH-cancel. With N=8 contenders, exactly one should commit and the
    // rest should exhaust.
    let ls = layer_store_with(&tv, MAX_ATTEMPTS, metrics.clone());
    let session_id = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &session_id, "histo-exhaust").await;

    let delta = build_valid_delta();
    let barrier = Arc::new(tokio::sync::Barrier::new(WRITERS));

    let mut handles = Vec::with_capacity(WRITERS);
    for i in 0..WRITERS {
        let ls_i = ls.clone();
        let sid_i = session_id.clone();
        let d_i = delta.clone();
        let barrier_i = barrier.clone();
        let client_id = format!("writer-{i}");
        let client_ts = 1000u64 + i as u64;
        handles.push(tokio::spawn(async move {
            barrier_i.wait().await;
            ls_i.update_layer_relayed(
                &sid_i,
                "histo-exhaust",
                oid,
                Some(d_i),
                &client_id,
                client_ts,
            )
            .await
        }));
    }

    let mut successes: u32 = 0;
    let mut exhausted: u32 = 0;
    for h in handles {
        match h.await.expect("writer task must not panic") {
            Ok(_) => successes += 1,
            Err(ServiceError::Aborted(msg)) => {
                assert!(
                    msg.contains("attempts exhausted"),
                    "with max_attempts=1, any Aborted error must be the exhaustion literal; \
                     got: {msg:?}"
                );
                exhausted += 1;
            }
            Err(e) => panic!("unexpected error from writer: {e:?}"),
        }
    }
    assert!(
        successes >= 1,
        "at least one writer must commit even with max_attempts=1"
    );
    assert!(
        exhausted >= 1,
        "with max_attempts=1 and {WRITERS} contenders, at least one writer must exhaust; \
         got successes={successes} exhausted={exhausted}"
    );

    // Histogram must record ONLY for the successful commits, not for the
    // exhausted callers.
    let (sample_count, _) = gather_attempts_histogram(&metrics);
    assert_eq!(
        sample_count, successes as u64,
        "attempts histogram must record one observation per successful commit, \
         NOT for exhausted callers; got sample_count={sample_count}, successes={successes}, \
         exhausted={exhausted}"
    );

    // Counter sanity: with max_attempts=1, zero retries are possible per call
    // (exhaustion fires immediately on the first conflict, adding
    // max_attempts-1 == 0 retries). Both counters must therefore stay at 0,
    // confirming the per-retry semantic — if the counter still fired
    // once-per-call, `exhausted_counter` would equal the exhausted-writer
    // count and this test would fail.
    let committed_counter = retry_counter(&metrics, "committed");
    let exhausted_counter = retry_counter(&metrics, "exhausted");
    assert_eq!(
        committed_counter, 0,
        "with max_attempts=1, no writer can record a 'committed retry' — any success was \
         first-try; got committed_counter={committed_counter}"
    );
    assert_eq!(
        exhausted_counter, 0,
        "with max_attempts=1, exhausted callers add (max_attempts - 1) = 0 retries to the \
         counter — got exhausted_counter={exhausted_counter}, exhausted_writers={exhausted}"
    );
}

// =============================================================================
// 5. First-try success records 1 in the histogram (not 0, not uncounted).
//    Pins the observation semantics: `observe_update_layer_attempts(1)` fires
//    on every success, even when no retry was needed.
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_first_try_success_records_one_in_attempts_histogram() {
    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("first-try"));
    let ls = layer_store_with(&tv, 32, metrics.clone());
    let session_id = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &session_id, "first-try").await;

    // Solo writer — no contention, first attempt wins.
    let (seq, _ts) = ls
        .update_layer_relayed(
            &session_id,
            "first-try",
            oid,
            Some(build_valid_delta()),
            "solo-client",
            1000,
        )
        .await
        .expect("solo writer must commit on first attempt");
    assert!(seq > 0, "sequence must advance, got {seq}");

    let (sample_count, buckets) = gather_attempts_histogram(&metrics);
    assert_eq!(
        sample_count, 1,
        "solo commit must record exactly one histogram sample, got {sample_count}"
    );
    // `bucket[upper_bound=1.0]` is cumulative — a first-try commit records
    // the value 1.0, which falls in the ≤1 bucket. So bucket_1 must be 1.
    let bucket_1 = buckets
        .iter()
        .find(|(ub, _)| (*ub - 1.0).abs() < f64::EPSILON)
        .map(|(_, c)| *c)
        .expect("histogram must have a 1.0 bucket");
    assert_eq!(
        bucket_1, 1,
        "first-try success must land in bucket ≤1 (records value 1.0); got bucket_1={bucket_1}"
    );

    // The retry counters both stay at zero: no committed retry, no exhaustion.
    assert_eq!(retry_counter(&metrics, "committed"), 0);
    assert_eq!(retry_counter(&metrics, "exhausted"), 0);
}

// =============================================================================
// 6. Retry still produces gap-free sequence numbers and a consistent stream.
//    Under retry, each successful commit must still yield a distinct {seq}-0
//    entry; the stream XLEN must still equal 1 (CreateLayer) + successes.
//    This catches a retry-loop bug where a retried writer might XADD with a
//    stale seq, duplicate an existing stream entry, or skip a seq.
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn update_layer_retry_preserves_stream_integrity_under_contention() {
    const WRITERS: usize = 15;
    let pool_size = (WRITERS as u32) + 4;

    let tv = TestValkey::setup_with_pool_size(pool_size).await;
    let ls = layer_store_default(&tv);
    let keys = tv.key_schema();
    let session_id = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &session_id, "retry-stream-integrity").await;

    let delta = build_valid_delta();
    let barrier = Arc::new(tokio::sync::Barrier::new(WRITERS));

    let mut handles = Vec::with_capacity(WRITERS);
    for i in 0..WRITERS {
        let ls_i = ls.clone();
        let sid_i = session_id.clone();
        let d_i = delta.clone();
        let barrier_i = barrier.clone();
        let client_id = format!("writer-{i}");
        let client_ts = 1000u64 + i as u64;
        handles.push(tokio::spawn(async move {
            barrier_i.wait().await;
            ls_i.update_layer_relayed(
                &sid_i,
                "retry-stream-integrity",
                oid,
                Some(d_i),
                &client_id,
                client_ts,
            )
            .await
        }));
    }

    let mut successes: Vec<u64> = Vec::new();
    for h in handles {
        match h.await.expect("writer task must not panic") {
            Ok((seq, _)) => successes.push(seq),
            Err(ServiceError::Aborted(msg)) => {
                panic!("no writer should abort at default retry budget; got: {msg}");
            }
            Err(e) => panic!("unexpected error: {e:?}"),
        }
    }

    // Distinct seqs.
    let mut unique: std::collections::HashSet<u64> = std::collections::HashSet::new();
    for seq in &successes {
        assert!(
            unique.insert(*seq),
            "two writers reported the same seq={seq} under retry; successes={successes:?}"
        );
    }

    // Contiguous from 2 (CreateLayer is seq=1).
    let mut sorted = successes.clone();
    sorted.sort();
    for (i, &seq) in sorted.iter().enumerate() {
        let expected = 2 + i as u64;
        assert_eq!(
            seq, expected,
            "retried commits must still produce contiguous seqs 2..=N+1; got sorted={sorted:?}"
        );
    }

    // Stream XLEN == 1 (CreateLayer) + successes.
    let stream_key = keys.layer_stream(oid);
    // Fields read as bytes: an entry's relay `ompf` field is binary.
    let entries: Vec<(String, HashMap<String, Bytes>)> = tv
        .client()
        .pool
        .xrange_values(&stream_key, "-", "+", None)
        .await
        .expect("XRANGE must succeed");
    assert_eq!(
        entries.len(),
        1 + successes.len(),
        "stream XLEN must match 1 (CreateLayer) + {} committed updates; entries={entries:?}",
        successes.len()
    );

    // Every committed seq must have a matching {seq}-0 stream entry.
    for seq in &successes {
        let want_id = format!("{seq}-0");
        assert!(
            entries.iter().any(|(id, _)| id == &want_id),
            "retried commit reported Ok(seq={seq}) but stream has no {want_id} entry; \
             entries={entries:?}"
        );
    }
}

// =============================================================================
// b.1dz regression tests: transient fred I/O errors flow through the retry
// loop instead of surfacing as Internal.
// =============================================================================
//
// Pre-fix, `update_layer_once` mapped every fred error during WATCH / HGETALL
// / snapshot load / MULTI queue to `ServiceError::Internal(...)`. The outer
// retry loop matches only `Aborted`, so a transient timeout / IO / canceled /
// cluster / backpressure error surfaced as `Internal` — the caller got a
// spurious failure even though a retry would have succeeded.
//
// The engineer's fix introduces `is_transient_fred_error` +
// `map_snapshot_load_err` helpers that turn transient kinds into
// `Aborted("transient I/O: <op> failed (<kind>): ...")`, letting the existing
// retry loop pick them up on a fresh txn connection.
//
// These tests exercise the retry path by injecting a synthetic transient
// `Aborted("transient I/O: ...")` error on the first attempt via the
// `with_fault_injector_for_test` seam added to `LayerStore`. The seam is the
// minimum surgical change needed to let the retry loop run under test without
// requiring a real fred fault-injection wrapper or a Valkey restart.
//
// Pre-fix regression: the same injected error would still route through the
// retry loop (matches `Aborted`), so injecting `Aborted(...)` alone doesn't
// directly falsify the pre-fix behavior. That is covered by the unit tests in
// `src/storage/client.rs` — `map_snapshot_load_err_timeout_maps_to_aborted_with_prefix`
// would fail against pre-fix code because pre-fix mapped Timeout to
// `Internal`, not `Aborted`. What these integration tests add on top is the
// end-to-end proof that (a) the retry loop does recover and commit when a
// transient-I/O-flavored `Aborted` is returned, and (b) the retry metrics
// fire with `outcome="committed"` on that path — the precise Metrics
// requirement from the ticket.

fn transient_io_err() -> ServiceError {
    // Shape matches the engineer's `map_snapshot_load_err` output for a
    // transient fred kind. The retry loop matches on the Aborted variant
    // itself, not the message, but using the real prefix keeps the test
    // readable and ensures grep-ability alongside production traces. The
    // kind label is pulled from fred's `ErrorKind::to_str()` rather than
    // hardcoded so a fred bump that renames the label doesn't fabricate a
    // "wrong-looking" test message.
    let kind_label = fred::error::ErrorKind::Timeout.to_str();
    ServiceError::Aborted(format!(
        "transient I/O: HGETALL obj header failed ({kind_label}): synthetic test"
    ))
}

/// Build a fault-injection queue pre-loaded with the given errors. The queue
/// is consumed FIFO by `update_layer_once`: index 0 hits attempt 1, index 1
/// hits attempt 2, etc.
fn injector(errs: Vec<ServiceError>) -> Arc<Mutex<VecDeque<ServiceError>>> {
    Arc::new(Mutex::new(errs.into_iter().collect()))
}

// Requirement 1 (ticket Tests #1): "mock a LayerStore with a fault-injecting
// fred wrapper that returns a transient error on the first HGETALL and
// succeeds on the second. Assert the call succeeds within the retry budget
// with one retry."
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_recovers_from_single_transient_io_on_first_attempt() {
    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("b1dz-transient-retry"));

    // Inject one transient-I/O error on attempt 1. Attempt 2 gets an empty
    // queue, runs the real Valkey sequence, and must commit.
    let queue = injector(vec![transient_io_err()]);

    let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_retry_max_attempts(32)
        .with_update_layer_backend(UpdateLayerBackend::Txn)
        .with_metrics(metrics.clone())
        .with_fault_injector_for_test(queue.clone());

    let session_id = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &session_id, "b1dz-single-retry").await;

    let (seq, _ts) = ls
        .update_layer_relayed(
            &session_id,
            "b1dz-single-retry",
            oid,
            Some(build_valid_delta()),
            "solo-client",
            1000,
        )
        .await
        .expect(
            "update_layer must recover from a single transient-I/O Aborted and commit on \
             the second attempt within the default retry budget — pre-b.1dz this failed \
             because transient fred errors surfaced as Internal, which the retry loop \
             doesn't match",
        );
    assert!(seq > 0, "sequence must advance, got {seq}");

    // Injector drained — exactly one fault consumed.
    assert_eq!(
        queue.lock().unwrap().len(),
        0,
        "fault injector queue should be fully drained after one retry"
    );

    // Attempts histogram: one sample recording value 2.0 (commit on the
    // second attempt). Sample_count == 1; sample_sum == 2.
    let (sample_count, _) = gather_attempts_histogram(&metrics);
    assert_eq!(
        sample_count, 1,
        "exactly one commit observed; got sample_count={sample_count}"
    );
    let h_sum = histogram_sum(&metrics);
    assert!(
        (h_sum - 2.0).abs() < f64::EPSILON,
        "commit on second attempt records value 2 in the histogram; got sum={h_sum}"
    );
}

// Requirement 2 (ticket Tests #2): "confirm update_layer_retries_total
// {outcome=\"committed\"} increments when a transient-I/O-triggered retry
// eventually commits." Models on the existing retry-metric tests but triggers
// via injected transient I/O rather than a WATCH conflict.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_transient_io_retry_increments_committed_counter() {
    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("b1dz-committed-counter"));

    // Three transient-I/O failures up front, then commit on attempt 4.
    let queue = injector(vec![
        transient_io_err(),
        transient_io_err(),
        transient_io_err(),
    ]);

    let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_retry_max_attempts(32)
        .with_update_layer_backend(UpdateLayerBackend::Txn)
        .with_metrics(metrics.clone())
        .with_fault_injector_for_test(queue.clone());

    let session_id = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &session_id, "b1dz-committed-counter").await;

    ls.update_layer_relayed(
        &session_id,
        "b1dz-committed-counter",
        oid,
        Some(build_valid_delta()),
        "solo-client",
        2000,
    )
    .await
    .expect("update_layer must commit after 3 injected transient-I/O retries");

    // Fault queue drained — all 3 faults consumed.
    assert_eq!(queue.lock().unwrap().len(), 0);

    // `committed` counter semantics (b.vi2): each commit contributes
    // (attempt_number - 1) retries. One commit on attempt 4 → 3 retries.
    let committed = retry_counter(&metrics, "committed");
    assert_eq!(
        committed, 3,
        "update_layer_retries_total{{outcome=\"committed\"}} must increment by exactly \
         (attempts - 1) when a transient-I/O-triggered retry eventually commits; \
         expected 3 (commit on attempt 4), got {committed} — pre-b.1dz the counter \
         would stay at 0 because the transient error would surface as Internal and \
         bypass the retry loop entirely"
    );

    // `exhausted` must remain zero — this scenario commits within budget.
    assert_eq!(
        retry_counter(&metrics, "exhausted"),
        0,
        "committed-path transient I/O retry must not fire the exhausted counter"
    );

    // Histogram: one sample, value 4 (commit on attempt 4).
    let (sample_count, _) = gather_attempts_histogram(&metrics);
    assert_eq!(sample_count, 1);
    let h_sum = histogram_sum(&metrics);
    assert!(
        (h_sum - 4.0).abs() < f64::EPSILON,
        "expected histogram sum=4 (commit on attempt 4); got {h_sum}"
    );
}

// Pins the non-transient-Internal escape hatch (b.1dz scope discipline): a
// non-transient error surfaced by `map_snapshot_load_err` as
// `ServiceError::Internal(...)` MUST break out of the retry loop, not be
// silently retried. This guards against a regression where someone widens
// the retry-loop match to `Err(_)` and hides real protocol / parse / config
// bugs behind silent retries.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_internal_error_is_not_retried() {
    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only(
        "b1dz-no-retry-on-internal",
    ));

    // Synthetic Internal — matches what `map_snapshot_load_err` produces for
    // a non-transient fred kind (Protocol / Parse / Auth / Config). Kind
    // label is sourced from fred's own `ErrorKind::to_str()` so a fred bump
    // that renames labels keeps the test message consistent with production.
    let protocol_label = fred::error::ErrorKind::Protocol.to_str();
    let queue = injector(vec![ServiceError::Internal(format!(
        "HGETALL obj header failed: {protocol_label}: malformed frame"
    ))]);

    let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_retry_max_attempts(32)
        .with_update_layer_backend(UpdateLayerBackend::Txn)
        .with_metrics(metrics.clone())
        .with_fault_injector_for_test(queue.clone());

    let session_id = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &session_id, "b1dz-no-retry-on-internal").await;

    let result = ls
        .update_layer_relayed(
            &session_id,
            "b1dz-no-retry-on-internal",
            oid,
            Some(build_valid_delta()),
            "solo-client",
            3000,
        )
        .await;

    match result {
        Err(ServiceError::Internal(msg)) => {
            let protocol_label = fred::error::ErrorKind::Protocol.to_str();
            assert!(
                msg.contains(protocol_label),
                "Internal error must bubble up verbatim so operators can diagnose root \
                 cause; expected substring `{protocol_label}`, got: {msg:?}"
            );
        }
        Ok(_) => panic!(
            "Internal error must surface immediately, not be silently retried — otherwise \
             persistent bugs (bad serialization, protocol errors, config failures) would be \
             masked forever. Observed Ok, which means the retry loop swallowed the Internal \
             (b.1dz scope-discipline regression)."
        ),
        Err(other) => panic!("expected Internal, got {other:?}"),
    }

    // The queue should have been drained by exactly one call — no retries.
    assert_eq!(
        queue.lock().unwrap().len(),
        0,
        "injector queue must be drained by exactly one attempt; a remaining entry would \
         mean the retry loop swallowed the Internal error and tried again"
    );

    // No commit → no histogram sample, no retry counters.
    let (sample_count, _) = gather_attempts_histogram(&metrics);
    assert_eq!(sample_count, 0);
    assert_eq!(retry_counter(&metrics, "committed"), 0);
    assert_eq!(retry_counter(&metrics, "exhausted"), 0);
}

// =============================================================================
// b.1dz fred-layer regression tests: the full fred::error → map_snapshot_load_err
// → retry-loop chain is exercised end-to-end.
// =============================================================================
//
// The tests above (`update_layer_recovers_from_single_transient_io_on_first_attempt`,
// `update_layer_transient_io_retry_increments_committed_counter`) inject a
// pre-cooked `ServiceError::Aborted("transient I/O: ...")` via
// `with_fault_injector_for_test`. Those verify that the retry loop's match
// arms are wired for the cooked-Aborted shape — but pre-b.1dz, the outer
// retry loop ALSO matched `Err(ServiceError::Aborted(_))` message-agnostically
// (it just retried the WATCH-cancel path). An injected cooked-Aborted therefore
// flows through retry on pre-fix code too, so those tests do not prove the
// actual bug fix: the `map_snapshot_load_err` classification that routes fred
// errors through the retry path.
//
// The tests in this section close that regression gap by injecting a raw
// `fred::error::Error` at the fred-client layer. The seam (new field
// `fred_error_injector_for_test` on `LayerStore`) substitutes the injected
// fred error for the result of the WATCH round-trip inside the WATCH path's
// attempt (`update_layer_once_watch_attempt`) and pipes it through the same `map_snapshot_load_err("WATCH obj header", ...)`
// call the production path uses. That exercises the full chain:
//
//   fred::error::Error(Timeout) →
//     map_snapshot_load_err →
//       ServiceError::Aborted("transient I/O: ...") →
//         update_layer retry loop →
//           second attempt runs against real Valkey →
//             commit
//
// On pre-fix code this chain is broken at step 2: a fred Timeout mapped to
// `ServiceError::Internal(...)`, which the retry loop does not match, so the
// first-attempt error would surface directly as Internal — the `.expect(...)`
// below would fail. That is the regression these tests lock in.

use fred::error::ErrorKind as FredErrorKind;

fn synthetic_fred_err(kind: FredErrorKind) -> fred::error::Error {
    fred::error::Error::new(kind, "synthetic test error")
}

fn fred_injector(errs: Vec<fred::error::Error>) -> Arc<Mutex<VecDeque<fred::error::Error>>> {
    Arc::new(Mutex::new(errs.into_iter().collect()))
}

// Regression test: a raw fred Timeout error on attempt 1 must flow through
// `map_snapshot_load_err` into `Aborted("transient I/O: ...")`, retry, and
// commit on attempt 2.
//
// PRE-FIX reasoning: pre-b.1dz, `update_layer_once` would have mapped this
// fred Timeout to `ServiceError::Internal(...)` (the pre-fix code path had
// no `map_snapshot_load_err` helper and wrapped every fred error as
// `Internal`). The retry loop matches only `Err(ServiceError::Aborted(_))`,
// so an Internal on attempt 1 would return immediately without retrying,
// `update_layer` would surface `Err(Internal(...))`, and the `.expect(...)`
// on the `ls.update_layer(...)` call below would panic. This test therefore
// provably fails against pre-fix code.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_fred_timeout_on_first_watch_flows_through_retry_loop() {
    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only(
        "b1dz-fred-timeout-chain",
    ));

    // One raw fred Timeout on attempt 1; empty on attempt 2 → real Valkey
    // WATCH/MULTI/EXEC runs and commits.
    let queue = fred_injector(vec![synthetic_fred_err(FredErrorKind::Timeout)]);

    let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_retry_max_attempts(32)
        .with_update_layer_backend(UpdateLayerBackend::Txn)
        .with_metrics(metrics.clone())
        .with_fred_error_injector_for_test(queue.clone());

    let session_id = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &session_id, "b1dz-fred-timeout").await;

    let (seq, _ts) = ls
        .update_layer_relayed(
            &session_id,
            "b1dz-fred-timeout",
            oid,
            Some(build_valid_delta()),
            "solo-client",
            5000,
        )
        .await
        .expect(
            "update_layer must commit on attempt 2 after attempt 1's raw fred Timeout is \
             classified by map_snapshot_load_err as Aborted(\"transient I/O: ...\") and picked \
             up by the retry loop. Pre-b.1dz this fails because fred::error::Error(Timeout) \
             mapped to ServiceError::Internal(...) which the retry loop does not match — \
             first-attempt Internal surfaces immediately and this .expect panics.",
        );
    assert!(
        seq > 0,
        "sequence must advance past create_layer's seq; got {seq}"
    );

    // Fred-error queue must be fully drained — the injected error was
    // consumed on attempt 1.
    assert_eq!(
        queue.lock().unwrap().len(),
        0,
        "fred error injector queue should have been drained by exactly one attempt"
    );

    // `committed` counter: commit on attempt 2 ⇒ 1 retry.
    let committed = retry_counter(&metrics, "committed");
    assert_eq!(
        committed, 1,
        "commit on attempt 2 must record exactly 1 committed-retry; got {committed}"
    );
    assert_eq!(retry_counter(&metrics, "exhausted"), 0);

    // Histogram: exactly one sample with value 2.
    let (sample_count, _) = gather_attempts_histogram(&metrics);
    assert_eq!(sample_count, 1);
    let h_sum = histogram_sum(&metrics);
    assert!(
        (h_sum - 2.0).abs() < f64::EPSILON,
        "histogram sum must equal 2 (single commit on attempt 2); got {h_sum}"
    );
}

// Regression test: a raw fred Cluster error (MOVED/ASK redirection mid-
// resharding) on attempt 1 must ALSO flow through the classification →
// retry chain. Pins the full transient-kind table end-to-end so that the
// retry recovery covers every kind classified as transient by
// `is_transient_fred_error`, not just Timeout.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_fred_cluster_error_on_first_watch_flows_through_retry_loop() {
    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only(
        "b1dz-fred-cluster-chain",
    ));

    let queue = fred_injector(vec![synthetic_fred_err(FredErrorKind::Cluster)]);

    let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_retry_max_attempts(32)
        .with_update_layer_backend(UpdateLayerBackend::Txn)
        .with_metrics(metrics.clone())
        .with_fred_error_injector_for_test(queue.clone());

    let session_id = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &session_id, "b1dz-fred-cluster").await;

    ls.update_layer_relayed(
        &session_id,
        "b1dz-fred-cluster",
        oid,
        Some(build_valid_delta()),
        "solo-client",
        6000,
    )
    .await
    .expect(
        "a raw fred Cluster error must be classified as transient and retried; pre-b.1dz \
         this would have surfaced as Internal and bypassed the retry loop",
    );

    assert_eq!(queue.lock().unwrap().len(), 0);
    assert_eq!(retry_counter(&metrics, "committed"), 1);
    assert_eq!(retry_counter(&metrics, "exhausted"), 0);
}

// Regression test: a raw fred Protocol error on attempt 1 must be classified
// as NON-transient by `map_snapshot_load_err` (Protocol isn't in the
// transient set) and surface as `ServiceError::Internal(...)` without
// retrying. This is the other side of the b.1dz classification contract —
// the retry loop must not widen to cover real bugs.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_fred_protocol_error_surfaces_internal_without_retry() {
    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only(
        "b1dz-fred-protocol-no-retry",
    ));

    let queue = fred_injector(vec![synthetic_fred_err(FredErrorKind::Protocol)]);

    let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_retry_max_attempts(32)
        .with_update_layer_backend(UpdateLayerBackend::Txn)
        .with_metrics(metrics.clone())
        .with_fred_error_injector_for_test(queue.clone());

    let session_id = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &session_id, "b1dz-fred-protocol").await;

    let result = ls
        .update_layer_relayed(
            &session_id,
            "b1dz-fred-protocol",
            oid,
            Some(build_valid_delta()),
            "solo-client",
            7000,
        )
        .await;

    match result {
        Err(ServiceError::Internal(msg)) => {
            let protocol_label = FredErrorKind::Protocol.to_str();
            assert!(
                msg.contains(protocol_label),
                "non-transient fred Protocol error must surface with fred's own ErrorKind \
                 label (`{protocol_label}`) so operators can diagnose; got: {msg:?}"
            );
            assert!(
                !msg.starts_with("transient I/O:"),
                "non-transient error must not be misclassified as transient; got: {msg:?}"
            );
        }
        Ok(_) => panic!(
            "a non-transient fred error (Protocol) MUST surface as Internal, not be \
             silently retried — otherwise real protocol/parse/config bugs would be masked \
             forever. Observed Ok, which means the retry loop over-matched (b.1dz scope \
             regression)."
        ),
        Err(other) => panic!("expected Internal, got {other:?}"),
    }

    // Injector drained by one attempt — no retry was issued.
    assert_eq!(
        queue.lock().unwrap().len(),
        0,
        "fred error injector queue must be drained by exactly one attempt; a remaining \
         entry would mean the retry loop retried a non-transient error"
    );

    // No commit → no histogram sample, no retry counters.
    let (sample_count, _) = gather_attempts_histogram(&metrics);
    assert_eq!(sample_count, 0);
    assert_eq!(retry_counter(&metrics, "committed"), 0);
    assert_eq!(retry_counter(&metrics, "exhausted"), 0);
}

// =============================================================================
// b.xd1 — the store not serving (`-LOADING`, `-BUSY`) is absorbed by the loop
// =============================================================================
//
// A node replaying its dataset after a restart answers `-LOADING` to every
// command, and a node behind a long script answers `-BUSY`. fred maps neither,
// so both arrive as `ErrorKind::Unknown`, which was permanent: the first one
// ended the UpdateLayer `INTERNAL`. b.xd1 classifies them transient
// (`storage::client::is_store_not_serving_reply`), so the retry loop absorbs a
// burst shorter than its budget, and a burst that outlasts the budget answers
// the retriable exhaustion (`UNAVAILABLE` on the wire), never `INTERNAL`.

/// The store's restart reply, as fred hands it over.
fn loading_err() -> fred::error::Error {
    fred::error::Error::new(
        FredErrorKind::Unknown,
        live_edit::storage::client::STORE_NOT_SERVING_SHAPES[0],
    )
}

/// The stream length of `oid`'s per-object stream: one entry per commit.
async fn stream_len(tv: &TestValkey, oid: u64) -> u64 {
    tv.client()
        .pool
        .xlen(tv.key_schema().layer_stream(oid))
        .await
        .expect("XLEN must succeed")
}

/// A burst of not-serving replies on the WATCH — one per
/// `STORE_NOT_SERVING_SHAPES` row, fewer than the budget — is retried
/// through: the commit lands once, on the attempt after the burst, and
/// `retries_total{outcome="committed"}` rises by the burst's length.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_rides_out_a_not_serving_burst_on_the_watch_and_commits_once() {
    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("xd1-not-serving-burst"));
    let burst: Vec<fred::error::Error> = live_edit::storage::client::STORE_NOT_SERVING_SHAPES
        .iter()
        .map(|reply| fred::error::Error::new(FredErrorKind::Unknown, *reply))
        .collect();
    let burst_len = burst.len() as u64;
    let queue = fred_injector(burst);

    let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_retry_max_attempts(32)
        .with_update_layer_backend(UpdateLayerBackend::Txn)
        .with_metrics(metrics.clone())
        .with_fred_error_injector_for_test(queue.clone());
    let session_id = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &session_id, "xd1-not-serving-burst").await;
    let xlen_before = stream_len(&tv, oid).await;

    ls.update_layer_relayed(
        &session_id,
        "xd1-not-serving-burst",
        oid,
        Some(build_valid_delta()),
        "solo-client",
        8000,
    )
    .await
    .expect(
        "a burst of -LOADING / -BUSY shorter than the retry budget must be retried through; \
         pre-b.xd1 the first one answered INTERNAL and the update was lost",
    );

    assert_eq!(
        queue.lock().unwrap().len(),
        0,
        "every injected reply was consumed, one per attempt"
    );
    assert_eq!(
        stream_len(&tv, oid).await,
        xlen_before + 1,
        "the update landed exactly once"
    );
    assert_eq!(
        retry_counter(&metrics, "committed"),
        burst_len,
        "one committed retry per refused attempt"
    );
    assert_eq!(retry_counter(&metrics, "exhausted"), 0);
}

/// A `-LOADING` burst that outlasts the budget answers the loop's exhaustion
/// — `Aborted` "… attempts exhausted …", which reaches the client as
/// `UNAVAILABLE` (retriable) — and never `INTERNAL`. Nothing lands.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_loading_past_the_retry_budget_answers_unavailable_exhaustion_not_internal() {
    const MAX_ATTEMPTS: u32 = 4;
    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("xd1-loading-exhausted"));
    let queue = fred_injector((0..MAX_ATTEMPTS).map(|_| loading_err()).collect());

    let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_retry_max_attempts(MAX_ATTEMPTS)
        .with_update_layer_backend(UpdateLayerBackend::Txn)
        .with_metrics(metrics.clone())
        .with_fred_error_injector_for_test(queue.clone());
    let session_id = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &session_id, "xd1-loading-exhausted").await;
    let xlen_before = stream_len(&tv, oid).await;

    let result = ls
        .update_layer_relayed(
            &session_id,
            "xd1-loading-exhausted",
            oid,
            Some(build_valid_delta()),
            "solo-client",
            8100,
        )
        .await;

    let err = match result {
        Err(e) => e,
        Ok(_) => panic!("every attempt was refused -LOADING; the update cannot have committed"),
    };
    match &err {
        ServiceError::Aborted(msg) => {
            assert!(
                msg.contains(&format!("{MAX_ATTEMPTS} attempts exhausted")),
                "the loop's exhaustion names its budget; got {msg:?}"
            );
            assert!(
                msg.contains("LOADING"),
                "the last cause is the store's reply; got {msg:?}"
            );
        }
        other => panic!(
            "a -LOADING that outlasts the budget must end in the loop's retriable exhaustion; \
             pre-b.xd1 the first one answered INTERNAL; got {other:?}"
        ),
    }
    assert_eq!(
        tonic::Status::from(err).code(),
        tonic::Code::Unavailable,
        "the client sees UNAVAILABLE, which it retries — not INTERNAL"
    );
    assert_eq!(queue.lock().unwrap().len(), 0, "one reply per attempt");
    assert_eq!(stream_len(&tv, oid).await, xlen_before, "nothing landed");
    assert_eq!(
        retry_counter(&metrics, "exhausted"),
        u64::from(MAX_ATTEMPTS - 1)
    );
    assert_eq!(retry_counter(&metrics, "committed"), 0);
}

/// A real `-BUSY` reply — fred's own parsing of it, through a reply proxy —
/// to the Lua fast path's `EVALSHA update_layer_fast` is retried, and the
/// update commits once. The script did not run (the store refuses before
/// executing), so the retry is a first execution.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_fast_path_evalsha_refused_busy_is_retried_and_commits_once() {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use live_edit::observability::metrics::test_util::{render, sample_value};
    use live_edit::storage::lua::{UpdateScriptMode, UPDATE_LAYER_FAST_SCRIPT};

    use crate::storage::snapshot_read_support::{
        error_reply_script, test_valkey_addr, ProxyAction, ProxyRule, ScriptReplyProxy,
    };

    const BUSY_REPLY: &str = live_edit::storage::client::STORE_NOT_SERVING_SHAPES[2];
    let armed = Arc::new(AtomicBool::new(false));
    let refused = Arc::new(AtomicUsize::new(0));
    let rule: ProxyRule = {
        let (armed, refused) = (Arc::clone(&armed), Arc::clone(&refused));
        let sha = UPDATE_LAYER_FAST_SCRIPT.sha1().to_string();
        Arc::new(move |args: &[Vec<u8>]| {
            let is_fast_commit = args
                .first()
                .is_some_and(|a| a.eq_ignore_ascii_case(b"EVALSHA"))
                && args
                    .get(1)
                    .is_some_and(|a| String::from_utf8_lossy(a).eq_ignore_ascii_case(&sha))
                && args
                    .iter()
                    .any(|a| a.as_slice() == UpdateScriptMode::Fast.as_str().as_bytes());
            // Only the first fast-path commit after arming is refused.
            (armed.load(Ordering::SeqCst)
                && is_fast_commit
                && refused.fetch_add(1, Ordering::SeqCst) == 0)
                .then(|| ProxyAction::Eval(error_reply_script(BUSY_REPLY)))
        })
    };
    let proxy = ScriptReplyProxy::start(test_valkey_addr(), rule).await;
    let tv = TestValkey::setup_with_url(proxy.url(), 4).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("xd1-fast-busy"));
    let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_backend(UpdateLayerBackend::Lua)
        .with_metrics(metrics.clone());
    let session_id = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &session_id, "xd1-fast-busy").await;
    let field = |value: u8| OwnedDelta {
        set_fields: vec![oodf_ffi::SetField {
            section_id: 1,
            key_name: "xd1-busy".to_owned(),
            value_or_ext_hash: vec![value],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 0,
        }],
        ..Default::default()
    };
    // An unrefused first update loads the script, so the refused one below is
    // the commit's own EVALSHA rather than the first of a NOSCRIPT reload.
    ls.update_layer_relayed(
        &session_id,
        "xd1-fast-busy",
        oid,
        Some(field(1)),
        "solo-client",
        1,
    )
    .await
    .expect("the warm-up update must commit");
    let xlen_before = stream_len(&tv, oid).await;

    armed.store(true, Ordering::SeqCst);
    let result = ls
        .update_layer_relayed(
            &session_id,
            "xd1-fast-busy",
            oid,
            Some(field(2)),
            "solo-client",
            2,
        )
        .await;
    armed.store(false, Ordering::SeqCst);

    result.unwrap_or_else(|e| {
        panic!(
            "a -BUSY answering the fast path's EVALSHA must be retried; pre-b.xd1 it answered \
             INTERNAL; got {e:?}"
        )
    });
    assert_eq!(proxy.hits(), 1, "exactly one EVALSHA was refused");
    assert!(
        refused.load(Ordering::SeqCst) >= 2,
        "the retry sent the commit's EVALSHA again"
    );
    assert_eq!(
        stream_len(&tv, oid).await,
        xlen_before + 1,
        "the update landed exactly once"
    );
    assert_eq!(
        sample_value(
            &render(&metrics),
            "live_edit_update_layer_retries_total",
            &[("path", "fast"), ("outcome", "committed")],
        ),
        Some(1.0),
        "one retry, committed on the fast path"
    );
}

// =============================================================================
// Shared delta — retries preserve delta content
// =============================================================================
//
// The retry loop holds the call's delta as one shared
// `Arc<Option<OwnedDelta>>` (b.715) and every attempt borrows it; the
// WATCH/MULTI/EXEC slow path clones it once per attempt, at the apply.
// Functional contract: every attempt, including the final one, must see the
// same delta content as attempt 1 — no attempt may consume the shared delta
// and leave a later one an empty/None delta.
//
// Test: inject 3 transient WATCH errors via the fred-error seam, confirm
// the 4th attempt commits, and observe the committed delta's content
// through the layer's stored field value (written by apply_delta on the
// successful attempt). A regression that dropped or consumed the delta on
// an earlier attempt would commit an empty delta on attempt 4, so the field
// value would be absent.

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_retries_preserve_delta_content_across_attempts() {
    use fred::interfaces::HashesInterface;
    use live_edit::storage::keys::KeySchema;
    use oodf_ffi::SetField;

    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only(
        "bk1r-pending-delta-preserved",
    ));

    // Three transient fred Timeouts on the WATCH round-trip → retry loop
    // loops through attempts 1/2/3, each replaying the same delta.
    // Attempt 4 runs real Valkey WATCH/MULTI/EXEC with the same delta and
    // commits.
    let queue = fred_injector(vec![
        synthetic_fred_err(FredErrorKind::Timeout),
        synthetic_fred_err(FredErrorKind::Timeout),
        synthetic_fred_err(FredErrorKind::Timeout),
    ]);

    let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        // max_attempts=4 means the final attempt is exactly the post-retry
        // commit: the last attempt the loop allows still sees the whole delta.
        .with_update_layer_retry_max_attempts(4)
        .with_update_layer_backend(UpdateLayerBackend::Txn)
        .with_metrics(metrics.clone())
        .with_fred_error_injector_for_test(queue.clone());

    let session_id = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &session_id, "bk1r-pending-preserved").await;

    // Distinctive payload so the stored field value can be verified byte-
    // for-byte after commit. A regression that replaced the delta with a
    // default on a retry would write an empty value instead.
    let distinctive_value: Vec<u8> = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0xCA, 0xFE];

    let delta = oodf_ffi::OwnedDelta {
        set_fields: vec![SetField {
            section_id: 1,
            key_name: "marker".to_owned(),
            value_or_ext_hash: distinctive_value.clone(),
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 0,
        }],
        ..Default::default()
    };

    let (seq, _ts) = ls
        .update_layer_relayed(
            &session_id,
            "bk1r-pending-preserved",
            oid,
            Some(delta),
            "bk1r-client",
            9000,
        )
        .await
        .expect("update_layer must commit on attempt 4 after 3 transient retries");

    // Queue drained — exactly three injected errors, one per retry.
    assert_eq!(queue.lock().unwrap().len(), 0);

    // committed retry counter: commit on attempt 4 ⇒ 3 retries.
    assert_eq!(retry_counter(&metrics, "committed"), 3);
    assert_eq!(retry_counter(&metrics, "exhausted"), 0);

    // Field value verification: read the stored field and assert its
    // `value` hash-field matches the distinctive payload (base64-encoded
    // per SDD §7.1). A silent delta-drop on the final attempt would write
    // no field at all; a delta-truncate would write a mangled / shorter
    // value. Either regression is caught here.
    let keys = tv.key_schema();
    let oid_str = oid.to_string();
    let field_key = keys.obj_field(&oid_str, "1", "marker");
    let fmap: std::collections::HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(&field_key)
        .await
        .expect("HGETALL stored field must succeed");
    let stored_b64 = fmap
        .get("value")
        .expect("stored field must have an inline `value` (delta preserved across retries)");
    use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
    let stored_bytes = BASE64
        .decode(stored_b64)
        .expect("stored value must be valid base64");
    assert_eq!(
        stored_bytes, distinctive_value,
        "committed field value must match the original distinctive payload exactly — \
         a mismatched value would indicate the retry loop dropped or truncated the \
         pending delta between attempts"
    );

    // Index bookkeeping: the fields index must list the member at the new
    // seq as score. One more cross-check that `apply_delta` ran on the
    // final attempt with the full delta.
    use fred::interfaces::SortedSetsInterface;
    let idx_key = keys.obj_fields_index(&oid_str);
    let members: Vec<(String, f64)> = tv
        .client()
        .pool
        .zrange(&idx_key, 0, -1, None, false, None, true)
        .await
        .expect("ZRANGE fields index must succeed");
    let expected_member = KeySchema::field_member("1", "marker");
    let found = members.iter().find(|(m, _)| m == &expected_member);
    assert!(
        found.is_some(),
        "fields index must contain the committed field's member; got {members:?}"
    );
    assert_eq!(
        found.unwrap().1 as u64,
        seq,
        "index score must equal the committed seq (version); a different score would \
         signal apply_delta ran against stale state"
    );
}
