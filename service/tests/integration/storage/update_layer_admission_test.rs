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

// Integration tests for the b.3gx per-object `update_layer` admission.
//
// b.3gx introduced an in-process admission entry keyed by `object_id` on
// `LayerStore` whose `writers` semaphore caps concurrent `update_layer` calls
// against a single object at `update_layer_per_object_concurrency` (default
// 16). The permit is acquired BEFORE the exclusive txn-pool slot, so waiters
// never starve transaction-pool connections while blocked. Paired with the
// raised retry budget (32 → 128), this collapses the `(K-1/K)^attempts`
// WATCH-conflict exhaustion tail from ~31% ABORTED at N=100 down to
// effectively zero.
//
// The entry (`ObjectAdmission`) also carries a one-permit `structural`
// semaphore (structural-edits spec SR-7.2): under the `lua` backend a
// structural or reset call takes it before `writers`. Section 8 at the end of
// this file covers that permit; sections 1-7 below cover `writers` and the
// map entry's lifecycle, and use field-only deltas, which take `writers` alone
// on every backend.
//
// Sections 1-7 lock in two non-overlapping invariants:
//
//   1. **Contention collapse** — N=100 concurrent `update_layer` writers on
//      one object all commit (no ABORTED), and the average attempts-per-
//      commit stays well below the raised 128 cap. Pre-b.3gx the same load
//      produced ~31% ABORTED at attempt=32 with median attempts also near
//      the cap.
//
//   2. **Semaphore-map leak regression** — the `semaphores` `DashMap` drains
//      back to its baseline size as objects are retired via DeleteLayer and
//      CreateLayer overwrite. Without the `.semaphores.remove(&oid)` calls
//      added in `delete_layer` and the overwrite branch of `create_layer`,
//      this map would grow O(N) in the number of retired object_ids for
//      the lifetime of the process.
//
//      Important nuance (flagged by PM-3gx): because `update_layer` lazily
//      reinserts via `entry().or_insert_with()`, a straggler writer that
//      arrives AFTER `semaphores.remove(&oid)` has fired will re-populate
//      that oid's entry before failing NOT_FOUND on HGETALL. object_ids are
//      monotonic INCR counters so the re-inserted entry will never be
//      revisited. The precise invariant is therefore "map does not grow
//      O(N) with the number of retired object_ids under the common case
//      (no racing stragglers)" plus a weaker "stays bounded by the number
//      of concurrently-in-flight stragglers under racing retirement". The
//      tests below cover both: 3 & 4 assert exact-empty under purely
//      sequential retirement (no straggler can race, so exact-empty is the
//      honest invariant), and 5 asserts bounded growth under deliberate
//      straggler-vs-retirement races.
//
// The tests require a running Valkey (default `redis://127.0.0.1:6379` or
// `TEST_VALKEY_URL`) and use per-test key prefixes via `TestValkey` so they
// run in parallel with the rest of the storage suite.

use std::sync::Arc;

use live_edit::domain::types::ObjectState;
use live_edit::error::ServiceError;
use live_edit::observability::metrics::Metrics;
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::relay::RelayedUpdates;
use live_edit::storage::test_support::TestValkey;
use oodf_ffi::OwnedDelta;

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

fn mock_ext_store() -> Arc<MockExternalValueStore> {
    Arc::new(MockExternalValueStore::new())
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

/// The `path` label the stores the two helpers below are used with commit on:
/// the default `lua` backend with an empty (field-only) delta takes the Lua
/// fast path, `path="fast"`, on every attempt.
const FAST_PATH: &str = "fast";

/// Pull the `live_edit_update_layer_attempts_histogram{path="fast"}`
/// sample_count and sample_sum. `sample_count` equals the number of successful
/// commits; `sample_sum` is the sum of attempt-index values recorded across
/// them (1 = first-try success, 2 = commit on second attempt, etc.). Mean
/// attempts-per-commit is therefore `sample_sum / sample_count`.
fn attempts_histogram(metrics: &Metrics) -> (u64, f64) {
    use live_edit::observability::metrics::test_util::{render, sample_value};
    let text = render(metrics);
    let path = [("path", FAST_PATH)];
    let count = sample_value(
        &text,
        "live_edit_update_layer_attempts_histogram_count",
        &path,
    )
    .unwrap_or(0.0) as u64;
    let sum = sample_value(
        &text,
        "live_edit_update_layer_attempts_histogram_sum",
        &path,
    )
    .unwrap_or(0.0);
    (count, sum)
}

/// Count of `update_layer` calls that exhausted their retry budget and
/// surfaced ABORTED (`update_layer_retries_total{path="fast",outcome="exhausted"}`
/// / (max_attempts - 1) — but we'd rather not do that arithmetic, so we
/// assert against the caller-side ABORTED count instead).
fn exhausted_counter(metrics: &Metrics) -> u64 {
    live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(metrics),
        "live_edit_update_layer_retries_total",
        &[("path", FAST_PATH), ("outcome", "exhausted")],
    )
    .unwrap_or(0.0) as u64
}

// =============================================================================
// 1. Concurrent-writer contention collapse (b.3gx primary regression).
//
// Spawn N=100 concurrent `update_layer` calls on one object. With the
// admission semaphore capped at K=8 and the retry budget at 128, all 100
// writers must commit (no ABORTED). The attempts-per-commit distribution
// must stay well below the 128 cap — a loose upper bound on the mean
// catches the regression cleanly without over-constraining on specific
// percentiles, which are noisy in CI and on contended dev-laptop runs.
//
// PRE-FIX REASONING: pre-b.3gx, the same load saw `(N-1)/N ≈ 31/32 per
// attempt` conflict probability per round and 32-attempt exhaustion at
// `(31/32)^32 ≈ 35.7%`. The measured rate at N=100 was 31.3% ABORTED.
// Without the admission cap, this test would see dozens of ABORTED
// results and the `successes == WRITERS` assertion would fail.
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn update_layer_one_hundred_concurrent_writers_all_commit_with_admission() {
    const WRITERS: usize = 100;
    // Admission cap pinned to the original b.3gx regression value so the test
    // continues to exercise the worst contention scenario that motivated the
    // admission semaphore. Intentionally lower than the current `K=16` default
    // (b.i2j) — raising the test's K would weaken the regression guarantee by
    // shrinking the (N-K)/K conflict window this test is designed to stress.
    const ADMISSION_K: u32 = 8;
    const RETRY_ATTEMPTS: u32 = 128;
    // The pool must be large enough that the ADMISSION_K admitted writers
    // can all hold txn-pool slots simultaneously. Extra headroom covers the
    // session-store and layer-index operations running against the same
    // pool. Headroom > K suffices because the admission gate ensures no
    // more than K `update_layer_once` bodies are inside the pool-acquire
    // window for this object at any time.
    let pool_size: u32 = ADMISSION_K + 8;

    let tv = TestValkey::setup_with_pool_size(pool_size).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("b3gx-stress"));
    let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_retry_max_attempts(RETRY_ATTEMPTS)
        .with_update_layer_per_object_concurrency(ADMISSION_K)
        .with_metrics(metrics.clone());
    let session_id = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &session_id, "b3gx-stress-layer").await;

    let delta = build_valid_delta();
    // Barrier so every writer enters `update_layer` at roughly the same
    // instant. Without it, the first handful of spawned tasks can finish
    // their entire admission-permit + WATCH/MULTI/EXEC cycle before later
    // tasks enter — the contention window this test exercises disappears.
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
                "b3gx-stress-layer",
                oid,
                Some(d_i),
                &client_id,
                client_ts,
            )
            .await
        }));
    }

    let mut successes: u32 = 0;
    let mut aborts: Vec<String> = Vec::new();
    let mut other_errors: Vec<String> = Vec::new();
    for h in handles {
        match h.await.expect("writer task must not panic") {
            Ok(_) => successes += 1,
            Err(ServiceError::Aborted(msg)) => aborts.push(msg),
            Err(e) => other_errors.push(format!("{e:?}")),
        }
    }

    assert!(
        other_errors.is_empty(),
        "unexpected non-Aborted errors from {WRITERS} concurrent writers: {other_errors:?}"
    );
    assert_eq!(
        successes as usize,
        WRITERS,
        "b.3gx regression: all {WRITERS} concurrent writers must commit with \
         admission_k={ADMISSION_K} + retry_attempts={RETRY_ATTEMPTS}; got \
         successes={successes} aborts={} (first aborted message: {:?}) — \
         pre-b.3gx this was ~68.7%% ok at N=100",
        aborts.len(),
        aborts.first()
    );
    assert!(
        aborts.is_empty(),
        "no writer should surface ABORTED at the paired (K=8, attempts=128) regression \
         pairing; got {} aborts with first message {:?}",
        aborts.len(),
        aborts.first()
    );
    assert_eq!(
        exhausted_counter(&metrics),
        0,
        "update_layer_retries_total{{outcome=\"exhausted\"}} must stay at zero — any \
         increment here means at least one writer exhausted its retry budget, which \
         contradicts the 0 ABORTED result and indicates a metrics-wiring bug"
    );

    // Attempts-per-commit must stay well below the 128 cap. At K=8 admission,
    // per-round win probability is ~1/8, so the mean attempts-per-commit
    // over 100 commits should land near a small double-digit number (the
    // geometric expectation is K=8). We assert a loose upper bound of 32
    // — well under 128 — so the test remains robust against CI scheduling
    // variance while still catching a regression to the pre-fix "near-cap
    // mean attempts" behavior. sample_count must equal WRITERS because
    // every writer committed and each commit records exactly one sample.
    let (sample_count, sample_sum) = attempts_histogram(&metrics);
    assert_eq!(
        sample_count, WRITERS as u64,
        "attempts histogram must record one observation per successful commit; \
         got sample_count={sample_count}, WRITERS={WRITERS}"
    );
    let mean_attempts = sample_sum / sample_count as f64;
    assert!(
        mean_attempts < 32.0,
        "mean attempts-per-commit must stay well under the retry cap; got \
         mean={mean_attempts:.2} over {sample_count} commits (sum={sample_sum}). \
         Pre-b.3gx this was near 32 (the old cap) because admission was unbounded; \
         a regression here means the admission semaphore isn't actually limiting \
         the contention window."
    );
    assert!(
        mean_attempts <= RETRY_ATTEMPTS as f64,
        "sanity: mean attempts cannot exceed the cap; got mean={mean_attempts:.2}, \
         cap={RETRY_ATTEMPTS}"
    );
}

// =============================================================================
// 2. Admission cap actually serializes writers — with K=1 the attempts-per-
//    commit histogram collapses to 1.0 for every commit. Pins the admission
//    mechanism as the causal control, not merely a parameter with a
//    coincidentally-correct default. With the gate allowing only one writer
//    through at a time, every WATCH/MULTI/EXEC runs without contention from
//    this LayerStore and every commit is a first-try success.
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn update_layer_admission_cap_one_serializes_writers_every_commit_first_try() {
    const WRITERS: usize = 20;
    const ADMISSION_K: u32 = 1;
    let pool_size: u32 = (WRITERS as u32) + 4;

    let tv = TestValkey::setup_with_pool_size(pool_size).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("b3gx-serialize"));
    let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_retry_max_attempts(128)
        .with_update_layer_per_object_concurrency(ADMISSION_K)
        .with_metrics(metrics.clone());
    let session_id = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &session_id, "b3gx-serialize-layer").await;

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
                "b3gx-serialize-layer",
                oid,
                Some(d_i),
                &client_id,
                client_ts,
            )
            .await
        }));
    }

    let mut successes: u32 = 0;
    for h in handles {
        match h.await.expect("writer task must not panic") {
            Ok(_) => successes += 1,
            Err(e) => {
                panic!("writer must commit at K=1 — serial admission has no contention; got {e:?}")
            }
        }
    }
    assert_eq!(successes as usize, WRITERS);

    // With K=1 the admission gate serializes every writer. The WATCH
    // window on this object therefore contains exactly one writer at a
    // time, so every commit is a first-try success. Histogram sum =
    // count = WRITERS.
    let (sample_count, sample_sum) = attempts_histogram(&metrics);
    assert_eq!(sample_count, WRITERS as u64);
    assert!(
        (sample_sum - WRITERS as f64).abs() < f64::EPSILON,
        "with admission K=1 every commit must be a first-try success (attempt index 1.0); \
         got sample_sum={sample_sum}, WRITERS={WRITERS} — if sample_sum > WRITERS the \
         admission gate is not actually serializing writers and retries are still \
         happening (b.3gx admission-not-wired regression)"
    );
    assert_eq!(
        exhausted_counter(&metrics),
        0,
        "no writer can exhaust at K=1 — serial admission means zero WATCH conflicts"
    );
}

// =============================================================================
// 3. Semaphore-map drains on DeleteLayer — NO-STRAGGLER invariant
//    (b.3gx leak-regression test, sequential cycles).
//
// Cycle `create_layer` + `update_layer` + `delete_layer` N times against
// distinct layer names in the same session. Each `update_layer` awaits to
// completion BEFORE `delete_layer` runs, so there is no window for a
// straggler to race retirement and re-insert via `entry().or_insert_with()`.
// Under this purely sequential shape the map must return to EXACTLY zero
// after each cycle — a stronger assertion than "bounded growth" and the
// cleanest falsifier for a regression where the engineer's
// `self.semaphores.remove(&object_id)` call in `delete_layer` step 7 is
// dropped.
//
// This test deliberately does NOT exercise the straggler-race path — see
// test 5 (`semaphore_map_bounded_under_straggler_race`) for that weaker
// bounded-growth invariant. Splitting these keeps each regression signal
// focused and falsifiable: test 3 pins "no leak on the happy path", test 5
// pins "no unbounded growth even under deliberate races". If 3 fails, the
// remove() call is missing; if 5 fails, the map grows linearly with
// racing stragglers (a different bug).
//
// PRE-FIX REASONING: without `self.semaphores.remove(&object_id)` in
// `delete_layer`, the count would grow monotonically to N and never drain
// for the lifetime of the process. That is the leak the ticket flags.
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn layer_store_semaphore_map_drains_after_delete_layer_cycles() {
    const CYCLES: usize = 12;

    let tv = TestValkey::setup().await;
    // `DEFAULT_UPDATE_LAYER_PER_OBJECT_CONCURRENCY` is the default in
    // `LayerStore::new`, so no explicit `.with_...` call is needed; the
    // default is what production runs with and is what this regression
    // test should exercise. A metrics registry isn't required — this test
    // asserts only on the map cardinality.
    let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
    let session_id = create_test_session(&tv).await;

    assert_eq!(
        ls.semaphore_count(),
        0,
        "baseline: a fresh LayerStore must have zero semaphore entries before any update_layer"
    );

    for i in 0..CYCLES {
        let layer_name = format!("drain-cycle-{i}");
        let oid = create_test_layer(&ls, &session_id, &layer_name).await;

        // One `update_layer` triggers lazy semaphore-map insertion. Without
        // this call the map would stay empty and delete_layer's remove()
        // would be a no-op — the test would pass vacuously and miss a
        // regression where create_layer accidentally touched the map.
        ls.update_layer_relayed(
            &session_id,
            &layer_name,
            oid,
            Some(build_valid_delta()),
            "solo-client",
            1000 + i as u64,
        )
        .await
        .expect("update_layer must commit for the semaphore entry to be populated");

        assert_eq!(
            ls.semaphore_count(),
            1,
            "after update_layer on cycle {i}, the map must hold exactly one entry for \
             oid={oid}; got count={}",
            ls.semaphore_count()
        );

        ls.delete_layer(&session_id, &layer_name, mock_ext_store())
            .await
            .expect("delete_layer must succeed");

        assert_eq!(
            ls.semaphore_count(),
            0,
            "b.3gx leak regression: delete_layer on cycle {i} (oid={oid}) must retire \
             the per-object admission semaphore entry (step 7 of delete_layer); \
             got remaining count={}",
            ls.semaphore_count()
        );
    }

    assert_eq!(
        ls.semaphore_count(),
        0,
        "after {CYCLES} full create/update/delete cycles, the semaphore map must be \
         empty — any residual entries are a leak on the delete_layer path"
    );
}

// =============================================================================
// 4. Semaphore-map drains on CreateLayer overwrite — NO-STRAGGLER invariant
//    (b.3gx leak-regression test, overwrite branch, sequential cycles).
//
// A fresh layer name gets an initial object_id; overwriting it allocates a
// NEW object_id and must retire the OLD one's semaphore entry. The test
// awaits the `update_layer` on each object before the next overwrite, so
// no straggler can race the `self.semaphores.remove(&old_oid)` call in
// `create_layer`. Under this sequential shape the map holds EXACTLY one
// entry (the current live object's) throughout, and exactly one entry at
// the end.
//
// PRE-FIX REASONING: without `self.semaphores.remove(&old_oid)` in the
// overwrite branch of `create_layer`, every overwrite would leak the
// previous object's semaphore while the new object populated a fresh
// entry on its first `update_layer` call. Over time the map would grow
// O(number-of-overwrites). Final-count == 1 proves only the current
// (live) object's entry survives the happy-path overwrite sequence.
//
// Straggler-race behaviour on the overwrite branch is covered separately
// in test 5 (`semaphore_map_bounded_under_straggler_race`).
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn layer_store_semaphore_map_retires_old_object_on_create_layer_overwrite() {
    const OVERWRITES: usize = 10;

    let tv = TestValkey::setup().await;
    let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
    let session_id = create_test_session(&tv).await;

    // First create — allocates object_id #1 for "overwrite-layer".
    let state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usd".to_string(),
        ..Default::default()
    };
    let mut current_oid = ls
        .create_layer(
            &session_id,
            "overwrite-layer",
            "alice",
            &state,
            mock_ext_store(),
        )
        .await
        .expect("initial create_layer must succeed");

    // Populate the semaphore entry for the current live object_id.
    ls.update_layer_relayed(
        &session_id,
        "overwrite-layer",
        current_oid,
        Some(build_valid_delta()),
        "solo-client",
        1000,
    )
    .await
    .expect("first update_layer must commit");
    assert_eq!(
        ls.semaphore_count(),
        1,
        "after the initial create + update, the map must hold exactly one entry"
    );

    for i in 0..OVERWRITES {
        // Overwrite: allocates a fresh object_id, flips the mapping, XADDs
        // `deleted` on the OLD stream, and must remove the old oid's
        // semaphore entry.
        let new_oid = ls
            .create_layer(
                &session_id,
                "overwrite-layer",
                "alice",
                &state,
                mock_ext_store(),
            )
            .await
            .expect("overwrite create_layer must succeed");
        assert_ne!(
            new_oid, current_oid,
            "each overwrite must allocate a fresh monotonic object_id"
        );

        // Immediately after overwrite, the OLD oid's semaphore entry must
        // already be gone — `create_layer` retires it inline. The NEW oid
        // has not been touched by `update_layer` yet, so the lazy-insert
        // path has not populated it. The map therefore must be empty
        // here. This is stricter than the final post-cycle check below
        // and directly falsifies a regression where the removal call in
        // the overwrite branch is dropped.
        assert_eq!(
            ls.semaphore_count(),
            0,
            "b.3gx leak regression: after create_layer overwrite on iteration {i} \
             (old_oid={current_oid} → new_oid={new_oid}), the old object's semaphore \
             entry must have been retired AND the new object's entry must not yet \
             exist (update_layer hasn't been called on it); got count={}",
            ls.semaphore_count()
        );

        current_oid = new_oid;

        // Now populate the new object's semaphore entry so the next
        // iteration has something to retire.
        ls.update_layer_relayed(
            &session_id,
            "overwrite-layer",
            current_oid,
            Some(build_valid_delta()),
            "solo-client",
            2000 + i as u64,
        )
        .await
        .expect("update_layer on overwritten layer must commit");

        assert_eq!(
            ls.semaphore_count(),
            1,
            "after update_layer on new oid={current_oid}, the map must again hold \
             exactly one entry (for the new object only)"
        );
    }

    assert_eq!(
        ls.semaphore_count(),
        1,
        "after {OVERWRITES} overwrites + live update, exactly one semaphore entry (for \
         the current live object_id={current_oid}) must remain. Any higher count means \
         retired object entries are leaking on the create_layer overwrite path."
    );
}

// =============================================================================
// 5. Semaphore-map stays bounded under straggler-race retirement
//    (b.3gx leak-regression test, concurrent/racing cycles).
//
// The preceding tests (3 & 4) assert that purely sequential create/update/
// delete cycles leave the map at EXACTLY zero entries — the happy-path
// invariant. But `update_layer` re-inserts lazily via
// `entry().or_insert_with()`: a straggler writer that invokes
// `update_layer(retired_oid, ...)` AFTER `delete_layer` has already fired
// its `semaphores.remove(&oid)` will re-populate that oid's entry before
// failing NotFound on HGETALL. object_ids are monotonic INCR counters so
// the re-inserted entry is never revisited — each such straggler leaves
// exactly one residual entry behind.
//
// PM-3gx flagged this nuance: the tight "returns to exact baseline"
// assertion is only honest under the no-straggler shape. Under straggler
// races the honest invariant is "map size is bounded by the number of
// concurrently-in-flight stragglers at the observation moment", not zero.
//
// This test stages M stragglers (deterministic, sequential — each
// straggler runs to completion before the next cycle starts) against
// M distinct retired object_ids, then runs a further BG no-straggler
// cycles, and asserts:
//
//   (a) after M straggler cycles: `semaphore_count() == M` — each
//       straggler re-inserted exactly one entry for the retired oid.
//   (b) after BG additional no-straggler cycles: `semaphore_count() < M + BG`
//       — specifically, no-straggler cycles must NOT grow the map.
//       (Exact equality: M, because no-straggler cycles leave zero.)
//   (c) the total is bounded by M, NOT by M + BG. This falsifies a
//       regression where no-straggler retirement path accidentally starts
//       leaking.
//
// The construction uses `delete_layer` BEFORE `update_layer` so the
// straggler's `or_insert_with` fires strictly after the `remove()`. No
// wall-clock sleep is needed — the ordering is deterministic by program
// structure, not by timing (keeps the test non-flaky).
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn layer_store_semaphore_map_bounded_under_straggler_race_retirement() {
    const STRAGGLER_CYCLES: usize = 5;
    // Background no-straggler cycles. Chosen larger than STRAGGLER_CYCLES so
    // a hypothetical "every cycle leaks one entry" regression would produce
    // a count of STRAGGLER_CYCLES + BACKGROUND_CYCLES = 15 instead of
    // STRAGGLER_CYCLES = 5, producing a bright-line signal.
    const BACKGROUND_CYCLES: usize = 10;

    let tv = TestValkey::setup().await;
    let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
    let session_id = create_test_session(&tv).await;

    assert_eq!(ls.semaphore_count(), 0, "baseline");

    // Phase 1: stage STRAGGLER_CYCLES straggler races. Each iteration:
    //   (a) create_layer → fresh oid
    //   (b) delete_layer → retires the (empty) semaphore entry
    //   (c) update_layer on the retired oid → `or_insert_with` re-inserts
    //       one entry, then HGETALL fails with NotFound (failed_precondition).
    // The straggler's `update_layer` returns an error, but the map
    // carries the orphaned entry forward.
    for i in 0..STRAGGLER_CYCLES {
        let layer_name = format!("straggler-{i}");
        let oid = create_test_layer(&ls, &session_id, &layer_name).await;

        // Retire BEFORE any update_layer has populated the entry. Note that
        // `delete_layer` calls `semaphores.remove(&oid)` even when the map
        // had no entry for that oid — that's a no-op, but it sets up the
        // structural precondition that the subsequent `update_layer`'s
        // `or_insert_with` fires against an explicitly-not-present entry.
        ls.delete_layer(&session_id, &layer_name, mock_ext_store())
            .await
            .expect("delete_layer on freshly-created layer must succeed");

        // Straggler: update_layer arriving after retirement. Must:
        //   - re-insert a semaphore entry (lazy `or_insert_with`)
        //   - fail with a non-retriable NotFound / FailedPrecondition because
        //     the object header has been UNLINKed. The layer mapping has
        //     also been DELed, so the caller sees a `NotFound` from the
        //     mapping resolution OR a `FailedPrecondition` from the
        //     stale-object-id check inside `update_layer_once`.
        //
        // In production the caller would have held a stale `object_id`
        // reference from a previous StatLayer. We emulate that by passing
        // the pre-delete `oid` directly. The error surface here matters
        // less than the semaphore-map side effect — any non-`Ok` return
        // from the straggler is acceptable; what matters is the map
        // cardinality delta.
        let result = ls
            .update_layer_relayed(
                &session_id,
                &layer_name,
                oid,
                Some(build_valid_delta()),
                "straggler-client",
                5000 + i as u64,
            )
            .await;
        assert!(
            result.is_err(),
            "straggler update_layer on retired oid={oid} must fail (object was deleted); \
             got Ok — the delete_layer cleanup did not actually remove the object"
        );

        assert_eq!(
            ls.semaphore_count(),
            i + 1,
            "after {} straggler cycles, exactly that many semaphore entries should \
             remain (each straggler re-inserts one entry for its retired oid and \
             never removes it); got count={}",
            i + 1,
            ls.semaphore_count()
        );
    }

    assert_eq!(
        ls.semaphore_count(),
        STRAGGLER_CYCLES,
        "after {STRAGGLER_CYCLES} straggler cycles, map must hold exactly that many \
         residual entries (one per retired oid that a straggler re-inserted) — a \
         departure here means either stragglers are NOT re-inserting (contradicting \
         the known `or_insert_with` behaviour, possible if someone inserts an \
         explicit retirement-check) or stragglers are leaking MORE than one entry \
         each (a regression that would compound under real workloads)"
    );

    // Phase 2: BACKGROUND no-straggler cycles. These exercise the happy
    // path repeatedly and must NOT grow the semaphore map. If the map
    // count grows beyond the Phase-1 total, a non-straggler cycle is
    // leaking — which would be a regression on delete_layer step 7
    // specifically for concurrent-workload profiles.
    for i in 0..BACKGROUND_CYCLES {
        let layer_name = format!("bg-{i}");
        let oid = create_test_layer(&ls, &session_id, &layer_name).await;
        ls.update_layer_relayed(
            &session_id,
            &layer_name,
            oid,
            Some(build_valid_delta()),
            "bg-client",
            9000 + i as u64,
        )
        .await
        .expect("bg update_layer must commit");
        ls.delete_layer(&session_id, &layer_name, mock_ext_store())
            .await
            .expect("bg delete_layer must succeed");
    }

    let final_count = ls.semaphore_count();
    assert_eq!(
        final_count, STRAGGLER_CYCLES,
        "after {BACKGROUND_CYCLES} additional no-straggler cycles, the map must \
         still hold exactly {STRAGGLER_CYCLES} residual entries — zero new leaks \
         from the happy-path. Any increase means the no-straggler retirement \
         path is leaking, which contradicts the sequential-cycle invariant \
         covered by test 3. Got final_count={final_count}."
    );
    // Explicit bounded-growth assertion, as PM-3gx requested: the map size \
    // is bounded by the number of straggler races ever staged — NOT by the \
    // total number of retirement cycles. STRAGGLER_CYCLES + BACKGROUND_CYCLES \
    // = 15 cycles total; the map size must be STRAGGLER_CYCLES = 5, i.e. \
    // one-third of total cycles. This falsifies a regression where ordinary \
    // retirement begins leaking.
    assert!(
        final_count < STRAGGLER_CYCLES + BACKGROUND_CYCLES,
        "semaphore-map growth must be sub-linear in total retirement cycles: \
         map size {final_count} vs. total cycles {} (stragglers={STRAGGLER_CYCLES}, \
         background={BACKGROUND_CYCLES}). If the count equals total cycles every \
         retirement is leaking an entry — a full regression of b.3gx's cleanup.",
        STRAGGLER_CYCLES + BACKGROUND_CYCLES
    );
}

// =============================================================================
// 6. Semaphore-map drains on `delete_layer` MID-OP failure
//    (b.xet leak-regression test, error path).
//
// The b.3gx leak tests (3 & 4) cover the *happy-path* retirement: every
// step of `delete_layer` / `create_layer` succeeds, and the post-commit
// `.remove(&oid)` calls fire. b.xet observed that if ANY Valkey step of
// `delete_layer` (steps 2-6: DEL mapping, XADD deleted, ext-GC schedule,
// UNLINK object keys, SREM layers index) fails with `ServiceError::Internal`,
// the pre-fix code returned via `?` BEFORE reaching step 7's
// `.remove(&object_id)`. The entry was orphaned in the map forever —
// object_ids are monotonic so no future caller revisits it, making it a
// true (slow) memory leak.
//
// The engineer's fix replaces the explicit tail `.remove()` with a
// `SemaphoreMapGuard` whose `Drop` impl removes the entry on every exit
// (success OR error). This test falsifies the pre-fix behavior by
// injecting a synthetic `ServiceError::Internal` via the
// `delete_layer_fault_injector_for_test` seam AFTER the guard is
// installed and asserting that `semaphore_count()` drops to zero even
// though the function returned an error.
//
// PRE-FIX REASONING: without the RAII guard, an Internal return at any
// step between guard-install and tail `.remove()` would leave the map
// holding a (1) entry after the call. With the guard, Drop runs on the
// `?` / `return Err(...)` unwind and the entry is removed.
// =============================================================================

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_layer_mid_op_internal_error_retires_semaphore_entry() {
    use std::collections::VecDeque;
    use std::sync::Mutex;

    let tv = TestValkey::setup().await;

    // Queue that the delete_layer fault injector will drain. Shared between
    // the test and the LayerStore via Arc<Mutex<...>>.
    let injector: Arc<Mutex<VecDeque<ServiceError>>> = Arc::new(Mutex::new(VecDeque::new()));

    let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_delete_layer_fault_injector_for_test(injector.clone());
    let session_id = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &session_id, "xet-delete-layer").await;

    // Populate the semaphore entry by running one successful `update_layer`.
    // Without this, `delete_layer`'s guard would Drop on an entry that was
    // never inserted and the test would pass vacuously — we'd be asserting
    // the map is empty when it started empty.
    ls.update_layer_relayed(
        &session_id,
        "xet-delete-layer",
        oid,
        Some(build_valid_delta()),
        "solo-client",
        1000,
    )
    .await
    .expect("update_layer must commit to populate the semaphore entry");
    assert_eq!(
        ls.semaphore_count(),
        1,
        "precondition: the semaphore entry for oid={oid} must be present before \
         the injected failure so the test can observe its retirement"
    );

    // Prime the fault injector with a single `Internal` error. `delete_layer`
    // pops it AFTER installing the semaphore guard and short-circuits with
    // `Err(err)`. The guard's Drop runs on unwind and (if the fix is in
    // place) removes the entry.
    injector.lock().unwrap().push_back(ServiceError::Internal(
        "injected mid-op delete_layer failure (b.xet)".to_string(),
    ));

    let result = ls
        .delete_layer(&session_id, "xet-delete-layer", mock_ext_store())
        .await;

    match result {
        Err(ServiceError::Internal(msg)) => {
            assert!(
                msg.contains("injected mid-op delete_layer failure"),
                "expected the injected error message to surface verbatim, got: {msg}"
            );
        }
        Err(other) => panic!(
            "expected Err(ServiceError::Internal(_)) from the injected failure; got {other:?}"
        ),
        Ok(()) => panic!(
            "delete_layer must not succeed when an Internal error is queued on the \
             fault injector — the injector short-circuits before step 2 (DEL mapping)"
        ),
    }

    // b.xet core assertion: the semaphore map entry for the retired oid must
    // be gone, even though the function returned an error. Pre-fix this
    // would be 1 (the guard did not exist and the explicit tail `.remove()`
    // never ran). Post-fix the guard's Drop fired on the `return Err(...)`
    // unwind and removed the entry.
    assert_eq!(
        ls.semaphore_count(),
        0,
        "b.xet leak regression: delete_layer returned Err(Internal) mid-op but the \
         per-object admission semaphore entry for oid={oid} is still in the map. \
         The SemaphoreMapGuard must remove the entry on Drop (both success and \
         error exit paths), not only on the successful tail `.remove()` call. \
         Got remaining count={}",
        ls.semaphore_count()
    );
}

// =============================================================================
// 7. Semaphore-map drains on a `create_layer` overwrite whose wake-up fails
//    (b.xet leak-regression test, overwrite wake-up failure path).
//
// The `create_layer` overwrite tail retires the OLD object's semaphore entry
// through an RAII guard installed before the old-stream wake-up (the
// `deleted` entry for subscribers still tailing the old object). Before
// b.xet the entry was removed explicitly after that wake-up, so a wake-up
// failure that returned early orphaned it.
//
// The test injects a raw `fred::error::Error` via the
// `create_layer_overwrite_xadd_fault_injector_for_test` seam, in place of the
// tail's inline wake-up — the attempt made while the guard is held. Since
// b.ua9 a failed wake-up is logged and retried by the spawned cleanup, never
// answered, whatever the fred kind, so one test loops over a transient `IO`
// and a permanent `Protocol`. For each it asserts:
//
//   1. The overwrite answers OK with its new object: its commit landed.
//   2. The OLD object's semaphore entry is NOT in the map (the guard's Drop
//      fired once the inline wake-up was done).
//   3. The NEW object's entry is also absent (never populated — no
//      `update_layer` ran on it), so the final map count is exactly 0.
//
// PRE-FIX REASONING (b.xet): without the RAII guard, the wake-up failure
// would skip the explicit removal and leave the old oid's entry in the map.
// A sustained cycle of "overwrite, wake-up fails" would grow the map
// O(number-of-failed-overwrites). Post-fix the map count drops to zero on
// every overwrite regardless of whether the wake-up succeeds.
// =============================================================================

/// Run the b.xet overwrite scenario with `injected` substituted for the
/// tail's inline old-stream wake-up: create a layer, populate its object's
/// semaphore entry with one `update_layer`, then overwrite it.
///
/// Returns the overwrite's new object id, the OLD object id, and the
/// semaphore-map count left behind once the overwrite has returned.
#[cfg(feature = "test-fault-injection")]
async fn overwrite_with_failed_wake_up(injected: fred::error::Error) -> (u64, u64, usize) {
    use std::collections::VecDeque;
    use std::sync::Mutex;

    let tv = TestValkey::setup().await;

    let injector: Arc<Mutex<VecDeque<fred::error::Error>>> = Arc::new(Mutex::new(VecDeque::new()));

    let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_create_layer_overwrite_xadd_fault_injector_for_test(injector.clone());
    let session_id = create_test_session(&tv).await;

    // Initial create: allocates the OLD object_id for "xet-overwrite-layer".
    // The first `create_layer` call is NOT an overwrite (no pre-existing
    // mapping), so the injector queue is not consumed here — only an
    // overwrite's tail makes a wake-up attempt.
    let state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usd".to_string(),
        ..Default::default()
    };
    let old_oid = ls
        .create_layer(
            &session_id,
            "xet-overwrite-layer",
            "alice",
            &state,
            mock_ext_store(),
        )
        .await
        .expect("initial create_layer must succeed");

    // Populate the semaphore entry for the OLD oid by running one
    // `update_layer`. Without this, there would be no entry for the
    // overwrite's guard to retire — the test would pass vacuously.
    ls.update_layer_relayed(
        &session_id,
        "xet-overwrite-layer",
        old_oid,
        Some(build_valid_delta()),
        "solo-client",
        1000,
    )
    .await
    .expect("update_layer must commit to populate the OLD oid's semaphore entry");
    assert_eq!(
        ls.semaphore_count(),
        1,
        "precondition: the semaphore entry for old_oid={old_oid} must be present \
         before the overwrite so the test can observe its retirement when its wake-up fails"
    );

    // Prime the injector for the tail's inline wake-up.
    injector.lock().unwrap().push_back(injected);

    // Overwrite: allocates a NEW object_id, writes the new object, fences the
    // old one, flips the mapping, then runs the tail — whose inline wake-up
    // takes the injected error. The commit landed, so the call answers OK;
    // the spawned cleanup retries the wake-up for real.
    let new_oid = ls
        .create_layer(
            &session_id,
            "xet-overwrite-layer",
            "alice",
            &state,
            mock_ext_store(),
        )
        .await
        .expect("an overwrite whose wake-up failed must answer OK: its commit landed");
    assert!(
        injector.lock().unwrap().is_empty(),
        "the overwrite's inline wake-up must have taken the injected error"
    );
    (new_oid, old_oid, ls.semaphore_count())
}

/// b.xet core assertion: the OLD oid's semaphore entry is gone even though the
/// overwrite's wake-up failed. The NEW oid's entry was never populated (no
/// `update_layer` ran on it), so the expected count is exactly 0. Pre-b.xet
/// this would be 1 (the old oid would still be resident because the explicit
/// tail `.remove(&old_oid)` never ran).
#[cfg(feature = "test-fault-injection")]
fn assert_old_semaphore_entry_retired(
    kind: &fred::error::ErrorKind,
    new_oid: u64,
    old_oid: u64,
    remaining: usize,
) {
    assert_ne!(
        new_oid, old_oid,
        "{kind:?}: the overwrite answers its NEW object"
    );
    assert_eq!(
        remaining, 0,
        "{kind:?}: b.xet leak regression: the create_layer overwrite's wake-up failed but the old \
         object's (old_oid={old_oid}) admission semaphore entry is still in the map. The \
         SemaphoreMapGuard in the overwrite tail must remove the OLD oid's entry on Drop \
         whether or not the wake-up landed. Got remaining count={remaining}"
    );
}

/// A failed inline wake-up of the overwrite, of a transient fred kind (`IO`)
/// or a permanent one (`Protocol`): either way the call answers OK (the commit
/// landed; the cleanup retries the wake-up), and the old object's semaphore
/// entry is still retired (b.xet). A failed wake-up is never answered,
/// whatever its class.
///
/// Before b.ua9 the failure was answered after the commit point — a retriable
/// `Unavailable` for a transient kind (b.62m), and before that a non-retriable
/// `INTERNAL`.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_overwrite_wake_up_failure_of_any_kind_answers_ok_and_retires_old_semaphore_entry(
) {
    for kind in [fred::error::ErrorKind::IO, fred::error::ErrorKind::Protocol] {
        let (new_oid, old_oid, remaining) = overwrite_with_failed_wake_up(fred::error::Error::new(
            kind.clone(),
            "injected wake-up failure on overwrite",
        ))
        .await;
        assert_old_semaphore_entry_retired(&kind, new_oid, old_oid, remaining);
    }
}

// =============================================================================
// 8. The `structural` permit (structural-edits spec SR-7.2).
//
// Under the `lua` backend a structural or reset `update_layer` call takes its
// object's one-permit `structural` semaphore FIRST, then a `writers` permit,
// and holds both for the whole call; a fast-path call takes `writers` alone.
// Under `lua-watch-structural` and `txn` every call takes `writers` alone, as
// before the permit existed. These tests pin:
//
//   - a queue of structural and reset calls waiting on `structural` holds no
//     `writers` permit, so a fast-path call is admitted and commits meanwhile;
//   - under `lua` a structural or reset call is not admitted while another
//     structural call on the object runs; under the two rollback values K of
//     them run at once;
//   - both permits are held across a call's backoff sleeps: parked after the
//     sleep, the call still holds them (read through
//     `admission_permits_available_for_test`), and a structural call queued
//     there commits after it;
//   - both permits are released when a call holding or waiting on them is
//     dropped — including one holding `structural` while it waits on
//     `writers` — and when an attempt panics;
//   - DeleteLayer / CreateLayer-overwrite retirement drops the map entry and
//     the structural-cache entry while calls holding the old entry finish
//     with NotFound / FailedPrecondition and re-insert nothing.
//
// How "admitted" is observed: a call's first poll runs `update_layer` up to
// its first `.await` that can pend. Nothing awaits before the admission wait
// (`update_layer_with_summary` classifies the delta and goes straight to
// `admit_update_layer`), and an available permit is granted on that first
// poll, so after one `Pending` poll a call either waits on a semaphore or
// holds its `writers` permit. A fast-path call that then commits on a store
// whose every other `writers` permit is held proves the polled calls hold
// none. The polls run under `tokio::task::coop::unconstrained`, so tokio's
// cooperative budget cannot answer `Pending` before a call reaches its
// semaphore.
//
// How "running" is observed under `lua`: a structural or reset call reaches
// its object through the structural cache's read side, which ticks
// `live_edit_structural_cache_lookups_total` once per attempt after
// admission, so the lookups summed are the structural or reset attempts that
// got past admission. The holder is parked between its attempts (its first
// attempt failed by the fault injector before it reads, so it has read
// nothing yet), holding both of its permits — the backoff test pins that
// hold. Under the rollback values a structural call takes the WATCH path, so
// the overlap tests count WATCH-window entries instead.
//
// Wait budget: every wait is one STEP_BUDGET (15 s) deadline, except the
// backoff test's wait for its holder, which is two (30 s) because the hook's
// STEP_BUDGET wait for the fast call runs inside it; the healthy path takes
// milliseconds. Worst case per test, every wait at its full budget: the queue
// test waits six times (90 s: the holder's park, the fast call, the holder,
// three queued calls), the not-admitted and retirement tests four times
// (60 s), the drop and panic tests three times (45 s), the backoff test 45 s
// (the holder's 30 s, then W's 15 s), the overlap tests once (15 s) — all
// under nextest's 120 s hang-killer.
//
// Feature-gated: the WATCH-window and between-attempts hooks and the fault
// injector exist only under `test-fault-injection`, so without that feature
// this section compiles to nothing.
// =============================================================================

#[cfg(feature = "test-fault-injection")]
mod structural_admission {
    use std::collections::VecDeque;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    use std::time::Duration;

    use fred::clients::Client;
    use futures::FutureExt;
    use live_edit::config::UpdateLayerBackend;
    use live_edit::storage::layer::BetweenAttemptsHook;
    use live_edit::storage::watched_txn::WatchWindowHook;
    use oodf_ffi::{CommandCreateSection, StructCommand};
    use tokio::sync::Notify;
    use tokio::task::JoinHandle;

    use super::*;
    use crate::storage::structural_cache_retirement_test::{assert_retired, cache_entry_for};

    /// Deadline on any one wait below. Each healthy step returns in
    /// milliseconds; this only turns a hang into a failure with a message.
    const STEP_BUDGET: Duration = Duration::from_secs(15);

    /// Connections in the fixture's pools: at most three calls hold an
    /// exclusive txn slot at once below, plus the fixture's own traffic.
    const POOL_SIZE: u32 = 8;

    type CallResult = Result<(u64, u64), ServiceError>;
    type Call = Pin<Box<dyn Future<Output = CallResult> + Send>>;

    struct Fixture {
        _tv: TestValkey,
        ls: Arc<LayerStore>,
        session_id: String,
        layer: &'static str,
        oid: u64,
        /// Hooked onto the store's client, where the structural cache's read
        /// side reports.
        metrics: Arc<Metrics>,
    }

    impl Fixture {
        /// `live_edit_structural_cache_lookups_total`, summed over its
        /// outcomes: one per structural or reset attempt on the `lua` backend
        /// that got past admission and read the object (see the section
        /// header).
        fn structural_reads(&self) -> u64 {
            use live_edit::observability::metrics::test_util::render;
            render(&self.metrics)
                .lines()
                .filter(|l| l.starts_with("live_edit_structural_cache_lookups_total{"))
                .filter_map(|l| l.rsplit_once(' ')?.1.parse::<f64>().ok())
                .sum::<f64>() as u64
        }
    }

    /// A session and one layer on a store `configure` builds from a plain one.
    async fn fixture(
        layer: &'static str,
        configure: impl FnOnce(LayerStore) -> LayerStore,
    ) -> Fixture {
        let tv = TestValkey::setup_with_pool_size(POOL_SIZE).await;
        let metrics = Arc::new(crate::common::metrics_scrape_only(layer));
        tv.client()
            .install_observability_hooks(Arc::clone(&metrics))
            .await;
        let ls = Arc::new(configure(LayerStore::new(
            tv.client().clone(),
            tv.key_schema().clone(),
            4096,
        )));
        let session_id = create_test_session(&tv).await;
        let oid = create_test_layer(&ls, &session_id, layer).await;
        Fixture {
            _tv: tv,
            ls,
            session_id,
            layer,
            oid,
            metrics,
        }
    }

    /// One `update_layer` call on the fixture's object, as an owned future
    /// that has not been polled.
    fn call(fx: &Fixture, delta: OwnedDelta, client_id: &str) -> Call {
        call_on(
            Arc::clone(&fx.ls),
            Target {
                session_id: fx.session_id.clone(),
                layer: fx.layer,
                oid: fx.oid,
            },
            delta,
            client_id,
        )
    }

    /// The object a call is made on: what a test hook needs to make one.
    #[derive(Clone)]
    struct Target {
        session_id: String,
        layer: &'static str,
        oid: u64,
    }

    /// [`call`], on `ls` and `target` rather than a [`Fixture`].
    fn call_on(ls: Arc<LayerStore>, target: Target, delta: OwnedDelta, client_id: &str) -> Call {
        let client_id = client_id.to_owned();
        Box::pin(async move {
            ls.update_layer_relayed(
                &target.session_id,
                target.layer,
                target.oid,
                Some(delta),
                &client_id,
                1,
            )
            .await
        })
    }

    /// Poll `call` once, asserting it pends (it waits on a semaphore, or holds
    /// its `writers` permit and waits on its first round trip; see the section
    /// header).
    async fn first_poll_pends(call: &mut Call, what: &str) {
        let first = tokio::task::coop::unconstrained(async { futures::poll!(call.as_mut()) }).await;
        assert!(
            first.is_pending(),
            "fixture: {what} must wait on its first poll, not complete"
        );
    }

    /// [`first_poll_pends`], then run the call to its end on a task.
    async fn queue(mut call: Call, what: &str) -> JoinHandle<CallResult> {
        first_poll_pends(&mut call, what).await;
        tokio::spawn(call)
    }

    async fn settle(handle: JoinHandle<CallResult>, what: &str) -> CallResult {
        settle_within(handle, STEP_BUDGET, what).await
    }

    /// [`settle`] with a caller-chosen deadline, for a call whose own wait
    /// contains another STEP_BUDGET wait.
    async fn settle_within(
        handle: JoinHandle<CallResult>,
        budget: Duration,
        what: &str,
    ) -> CallResult {
        tokio::time::timeout(budget, handle)
            .await
            .unwrap_or_else(|_| panic!("{what} did not finish within {budget:?}"))
            .unwrap_or_else(|e| panic!("{what}'s task must not panic: {e}"))
    }

    async fn commits(call: Call, what: &str) {
        tokio::time::timeout(STEP_BUDGET, call)
            .await
            .unwrap_or_else(|_| {
                panic!("{what} was not admitted within {STEP_BUDGET:?}: a permit is still held")
            })
            .unwrap_or_else(|e| panic!("{what} must commit; got {e:?}"));
    }

    /// One `CreateSection` under the root: a structural call — the structural
    /// route under `lua`, the WATCH path under the two rollback values.
    fn structural_delta(section_id: u64) -> OwnedDelta {
        OwnedDelta {
            struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
                parent_id: 0,
                section_id,
                section_name: format!("admission-{section_id}"),
                section_type: 1,
            })],
            ..Default::default()
        }
    }

    /// A whole-object reset (`is_diff` with `base_version` 0): the reset class.
    fn reset_delta() -> OwnedDelta {
        OwnedDelta {
            is_diff: true,
            base_version: 0,
            source_format: "usd".to_owned(),
            ..Default::default()
        }
    }

    /// Parks the FIRST call made on a `lua` store between its attempts: the
    /// store's fault injector fails the first attempt of all, before it
    /// reads anything, and the between-attempts hook parks that call's retry
    /// until [`HolderPark::release`] — past admission, holding both of its
    /// permits. Later retries pass straight through.
    struct HolderPark {
        entered: Arc<Notify>,
        release: Arc<Notify>,
    }

    impl HolderPark {
        /// The park, and the store configuration installing it on a `lua`
        /// store admitting `k` writers per object.
        fn new(k: u32) -> (Self, impl FnOnce(LayerStore) -> LayerStore) {
            let park = HolderPark {
                entered: Arc::new(Notify::new()),
                release: Arc::new(Notify::new()),
            };
            let fired = Arc::new(AtomicUsize::new(0));
            let (entered, release) = (Arc::clone(&park.entered), Arc::clone(&park.release));
            let hook: BetweenAttemptsHook = Arc::new(move |_attempt: u32| {
                let first = fired.fetch_add(1, Ordering::SeqCst) == 0;
                let (entered, release) = (Arc::clone(&entered), Arc::clone(&release));
                async move {
                    if first {
                        entered.notify_one();
                        release.notified().await;
                    }
                }
                .boxed()
            });
            let injected: Arc<Mutex<VecDeque<ServiceError>>> =
                Arc::new(Mutex::new(VecDeque::from([ServiceError::Aborted(
                    "injected retriable failure: parks the holder between its attempts".to_owned(),
                )])));
            let configure = move |ls: LayerStore| {
                ls.with_update_layer_backend(UpdateLayerBackend::Lua)
                    .with_update_layer_per_object_concurrency(k)
                    .with_fault_injector_for_test(injected)
                    .with_between_attempts_hook_for_test(hook)
            };
            (park, configure)
        }

        /// Wait until the first call is parked between its attempts.
        async fn wait_entered(&self) {
            tokio::time::timeout(STEP_BUDGET, self.entered.notified())
                .await
                .expect("the holding call must park between its attempts");
        }

        fn release(&self) {
            self.release.notify_one();
        }
    }

    /// A structural call parked between its attempts on the fixture's object,
    /// holding the object's `structural` permit and one `writers` permit.
    /// It has read nothing yet.
    async fn park_structural_holder(fx: &Fixture, park: &HolderPark) -> JoinHandle<CallResult> {
        let holder = tokio::spawn(call(fx, structural_delta(1), "holder"));
        park.wait_entered().await;
        assert_eq!(
            fx.structural_reads(),
            0,
            "fixture: the parked holder failed its first attempt before reading"
        );
        holder
    }

    // ── A structural queue does not starve fast-path edits ──────────────────

    /// The Epic's acceptance: with structural and reset calls queued on the
    /// object's `structural` permit behind a running structural call, a
    /// fast-path call is admitted and commits while they wait.
    ///
    /// K is one more than the queue, so the fast call gets a `writers` permit
    /// only if no queued call holds one: were `writers` taken before
    /// `structural`, the three queued calls would hold three and the running
    /// call the fourth, and the fast call would wait out its deadline.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn update_layer_lua_fast_call_commits_while_structural_and_reset_calls_queue() {
        let queued_deltas = [structural_delta(2), reset_delta(), structural_delta(3)];
        let k = queued_deltas.len() as u32 + 1;
        let (park, configure) = HolderPark::new(k);
        let fx = fixture("admission-queue", configure).await;
        let holder = park_structural_holder(&fx, &park).await;

        let mut queued = Vec::new();
        for (i, delta) in queued_deltas.into_iter().enumerate() {
            let what = format!("queued call {i}");
            queued.push(queue(call(&fx, delta, &what), &what).await);
        }

        commits(
            call(&fx, build_valid_delta(), "fast"),
            "the fast call, with three structural and reset calls queued on the object",
        )
        .await;
        assert_eq!(
            fx.structural_reads(),
            0,
            "no queued structural or reset call may get past admission while the holder runs"
        );

        park.release();
        settle(holder, "the holder")
            .await
            .expect("the holder must commit once released");
        for (i, handle) in queued.into_iter().enumerate() {
            settle(handle, &format!("queued call {i}"))
                .await
                .unwrap_or_else(|e| panic!("queued call {i} must commit once admitted; got {e:?}"));
        }
    }

    // ── Under `lua`, structural calls on one object never overlap ───────────

    /// With K = 2 and a structural call running (holding one `writers`
    /// permit), a second structural-class call is polled into the admission
    /// queue, and then a fast call commits: the fast call took the only other
    /// `writers` permit, so the queued call holds none — it has not been
    /// admitted — and no structural attempt has read the object. Once the
    /// running call finishes, the queued one runs and commits.
    async fn queued_call_is_not_admitted_while_a_structural_call_runs(
        layer: &'static str,
        queued_delta: OwnedDelta,
    ) {
        let (park, configure) = HolderPark::new(2);
        let fx = fixture(layer, configure).await;
        let holder = park_structural_holder(&fx, &park).await;
        let queued = queue(call(&fx, queued_delta, "queued"), "the queued call").await;

        commits(
            call(&fx, build_valid_delta(), "fast"),
            "the fast call (its commit proves the queued call holds no writers permit)",
        )
        .await;
        assert_eq!(
            fx.structural_reads(),
            0,
            "{layer}: the queued call must not get past admission while another structural \
             call on the object runs"
        );

        park.release();
        settle(holder, "the holder")
            .await
            .expect("the holder must commit once released");
        settle(queued, "the queued call")
            .await
            .expect("the queued call must commit once the holder is done");
        assert!(
            fx.structural_reads() >= 2,
            "{layer}: the queued call must have run its structural attempt after the holder"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn update_layer_lua_structural_call_is_not_admitted_while_another_runs() {
        queued_call_is_not_admitted_while_a_structural_call_runs(
            "admission-serial-structural",
            structural_delta(2),
        )
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn update_layer_lua_reset_call_is_not_admitted_while_a_structural_call_runs() {
        queued_call_is_not_admitted_while_a_structural_call_runs(
            "admission-serial-reset",
            reset_delta(),
        )
        .await;
    }

    /// Under the two rollback values a structural or reset call takes
    /// `writers` alone, as before the `structural` permit existed: K = 3 of
    /// them are inside their WATCH windows at once. The hook holds each of the
    /// first three WATCH entries until all three have arrived, so the calls
    /// finish only if all three were admitted together.
    async fn k_structural_calls_run_at_once(layer: &'static str, backend: UpdateLayerBackend) {
        let deltas = [structural_delta(1), reset_delta(), structural_delta(2)];
        let together = deltas.len();
        let barrier = Arc::new(tokio::sync::Barrier::new(together));
        let entries = Arc::new(AtomicUsize::new(0));
        let hook: WatchWindowHook = {
            let (barrier, entries) = (Arc::clone(&barrier), Arc::clone(&entries));
            Arc::new(move |_watched: &Client| {
                let first_wave = entries.fetch_add(1, Ordering::SeqCst) < together;
                let barrier = Arc::clone(&barrier);
                async move {
                    if first_wave {
                        barrier.wait().await;
                    }
                }
                .boxed()
            })
        };
        let fx = fixture(layer, move |ls| {
            ls.with_update_layer_backend(backend)
                .with_update_layer_per_object_concurrency(together as u32)
                .with_watch_window_hook_for_test(hook)
        })
        .await;

        let calls: Vec<_> = deltas
            .into_iter()
            .enumerate()
            .map(|(i, delta)| tokio::spawn(call(&fx, delta, &format!("writer-{i}"))))
            .collect();
        let outcomes = tokio::time::timeout(STEP_BUDGET, futures::future::join_all(calls))
            .await
            .unwrap_or_else(|_| {
                panic!(
                    "{layer}: the {together} structural and reset calls did not all reach their \
                     WATCH windows at once — {} WATCH entries seen; under {backend:?} they take \
                     only writers permits, {together} of which are free",
                    entries.load(Ordering::SeqCst)
                )
            });
        for (i, outcome) in outcomes.into_iter().enumerate() {
            outcome
                .unwrap_or_else(|e| panic!("{layer}: writer-{i}'s task must not panic: {e}"))
                .unwrap_or_else(|e| panic!("{layer}: writer-{i} must commit; got {e:?}"));
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn update_layer_lua_watch_structural_runs_k_structural_calls_on_one_object_at_once() {
        k_structural_calls_run_at_once(
            "admission-overlap-lws",
            UpdateLayerBackend::LuaWatchStructural,
        )
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn update_layer_txn_runs_k_structural_calls_on_one_object_at_once() {
        k_structural_calls_run_at_once("admission-overlap-txn", UpdateLayerBackend::Txn).await;
    }

    // ── Both permits are RAII ───────────────────────────────────────────────

    /// A `lua` store admitting `writers` writers per object whose FIRST
    /// attempt of all fails with a retriable `Aborted`, and whose
    /// between-attempts hook runs `on_retry` on the first retry it sees. The
    /// first call made on it is then inside its retry loop, past its backoff
    /// sleep, holding every permit its call takes: a structural call the
    /// `structural` permit and a `writers` one, a fast call a `writers` one.
    async fn retrying_holder_fixture(
        layer: &'static str,
        writers: u32,
        on_retry: impl Fn() -> futures::future::BoxFuture<'static, ()> + Send + Sync + 'static,
    ) -> Fixture {
        let injected: Arc<Mutex<VecDeque<ServiceError>>> =
            Arc::new(Mutex::new(VecDeque::from([ServiceError::Aborted(
                "injected retriable failure".to_owned(),
            )])));
        let fired = Arc::new(AtomicUsize::new(0));
        let hook: BetweenAttemptsHook = Arc::new(move |_attempt: u32| {
            if fired.fetch_add(1, Ordering::SeqCst) == 0 {
                on_retry()
            } else {
                async {}.boxed()
            }
        });
        fixture(layer, move |ls| {
            ls.with_update_layer_backend(UpdateLayerBackend::Lua)
                .with_update_layer_per_object_concurrency(writers)
                .with_fault_injector_for_test(injected)
                .with_between_attempts_hook_for_test(hook)
        })
        .await
    }

    /// After the permits are back, both kinds of call are admitted again: a
    /// structural call (the `structural` and `writers` permits) and a fast
    /// call (the only `writers` permit).
    async fn both_permits_are_free(fx: &Fixture, context: &str) {
        commits(
            call(fx, structural_delta(10), "after-structural"),
            &format!("{context}: a structural call afterwards"),
        )
        .await;
        commits(
            call(fx, build_valid_delta(), "after-fast"),
            &format!("{context}: a fast call afterwards"),
        )
        .await;
    }

    /// A call dropped while it holds both permits (parked between its retry
    /// attempts — what an RPC deadline does to a call in its backoff), and
    /// calls dropped while they wait on `structural` and on `writers`, release
    /// everything: afterwards a structural call and a fast call commit.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn update_layer_lua_calls_dropped_holding_or_waiting_on_permits_release_them() {
        let parked = Arc::new(Notify::new());
        let fx = retrying_holder_fixture("admission-drop", 1, {
            let parked = Arc::clone(&parked);
            move || {
                let parked = Arc::clone(&parked);
                async move {
                    parked.notify_one();
                    futures::future::pending::<()>().await;
                }
                .boxed()
            }
        })
        .await;

        let mut holder = call(&fx, structural_delta(1), "holder");
        drive_until_parked(&mut holder, &parked, "the holder").await;

        let mut on_structural = call(&fx, structural_delta(2), "waits-on-structural");
        first_poll_pends(&mut on_structural, "a structural call behind the holder").await;
        let mut on_writers = call(&fx, build_valid_delta(), "waits-on-writers");
        first_poll_pends(&mut on_writers, "a fast call behind the holder").await;

        drop(on_structural);
        drop(on_writers);
        drop(holder);

        both_permits_are_free(&fx, "after the holder and both waiters were dropped").await;
    }

    /// A call whose attempt panics releases both permits as it unwinds:
    /// afterwards a structural call and a fast call commit.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn update_layer_lua_panicking_attempt_releases_both_permits() {
        let fx = retrying_holder_fixture("admission-panic", 1, || {
            async { panic!("injected panic between attempts") }.boxed()
        })
        .await;

        let holder = tokio::spawn(call(&fx, structural_delta(1), "holder"));
        let joined = tokio::time::timeout(STEP_BUDGET, holder)
            .await
            .expect("the holder must reach its panicking retry");
        assert!(
            joined.as_ref().is_err_and(|e| e.is_panic()),
            "fixture: the holder's task must have panicked; got {joined:?}"
        );

        both_permits_are_free(&fx, "after the holder panicked").await;
    }

    /// Poll `holder` on this task until its between-attempts hook signals
    /// `parked`: the call is then past its backoff sleep, holding its permits.
    async fn drive_until_parked(holder: &mut Call, parked: &Notify, what: &str) {
        tokio::time::timeout(STEP_BUDGET, async {
            tokio::select! {
                biased;
                () = parked.notified() => {}
                returned = holder.as_mut() => panic!(
                    "{what} must park between its attempts, not return; it returned \
                     {returned:?}"
                ),
            }
        })
        .await
        .unwrap_or_else(|_| panic!("{what} must reach its second attempt"));
    }

    /// A structural call dropped while it holds `structural` and waits on
    /// `writers` releases `structural` — the state an RPC deadline catches when
    /// the object's writers are saturated. K = 1: a fast call parked between
    /// its attempts holds the only `writers` permit, so a structural call
    /// polled once takes the free `structural` permit and queues on `writers`.
    /// It is dropped there; once the fast call finishes, a structural call must
    /// commit. A `structural` permit released only by a value built after both
    /// acquisitions (for instance `forget` plus `add_permits` in a `Drop`)
    /// would be lost here and wedge every later structural call on the object.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn update_layer_lua_call_dropped_holding_structural_while_waiting_on_writers_releases_it()
    {
        let parked = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let fx = retrying_holder_fixture("admission-drop-mid", 1, {
            let (parked, release) = (Arc::clone(&parked), Arc::clone(&release));
            move || {
                let (parked, release) = (Arc::clone(&parked), Arc::clone(&release));
                async move {
                    parked.notify_one();
                    release.notified().await;
                }
                .boxed()
            }
        })
        .await;

        let mut holder = call(&fx, build_valid_delta(), "fast-holder");
        drive_until_parked(&mut holder, &parked, "the fast holder").await;
        let mut dropped = call(&fx, structural_delta(2), "dropped");
        first_poll_pends(
            &mut dropped,
            "a structural call behind the fast holder (it takes `structural`, then waits on \
             `writers`)",
        )
        .await;
        drop(dropped);

        release.notify_one();
        tokio::time::timeout(STEP_BUDGET, holder)
            .await
            .expect("the fast holder must finish once released")
            .expect("the fast holder must commit");

        commits(
            call(&fx, structural_delta(3), "after"),
            "a structural call after one was dropped holding `structural` and waiting on \
             `writers`",
        )
        .await;
    }

    /// What the holder's between-attempts hook in
    /// [`update_layer_lua_structural_call_holds_its_permits_across_its_backoff`]
    /// left behind: the object's free `(structural, writers)` permits as the
    /// hook found them, the structural call W it queued, and the fast call's
    /// result.
    #[derive(Default)]
    struct DuringBackoff {
        free_permits: Option<Option<(usize, usize)>>,
        queued: Option<JoinHandle<CallResult>>,
        fast: Option<CallResult>,
    }

    /// A structural call holds both of its permits across its backoff sleep,
    /// not only within an attempt. K = 2: a structural holder H fails its first
    /// attempt and parks between its attempts, after its backoff sleep. There
    /// the hook reads the object's free permits, before anything else runs on
    /// the object: no `structural` permit and K - 1 `writers` permits, because
    /// H still holds one of each. Then a second structural call W is polled
    /// once (queued on `structural`) and spawned, and a fast call F commits on
    /// the other `writers` permit. H then commits, and W after it.
    ///
    /// The permit reading pins the hold itself, whatever the scheduling: it
    /// fails if H's permits were released before the sleep, whether or not
    /// they are taken back for attempt 2. Each commit then takes the next
    /// sequence number (the id of its layer-stream entry), so the order is read
    /// off the returned sequences: F, then H, then W — the behaviour the hold
    /// buys. Were the permits taken per attempt instead, W's first poll would
    /// take the free `structural` permit and H's attempt 2 would queue behind
    /// W, so W would commit before H.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn update_layer_lua_structural_call_holds_its_permits_across_its_backoff() {
        const WRITERS: u32 = 2;
        let target: Arc<std::sync::OnceLock<(std::sync::Weak<LayerStore>, Target)>> =
            Arc::new(std::sync::OnceLock::new());
        let during: Arc<Mutex<DuringBackoff>> = Arc::default();
        let fx = retrying_holder_fixture("admission-backoff", WRITERS, {
            let (target, during) = (Arc::clone(&target), Arc::clone(&during));
            move || {
                let (target, during) = (Arc::clone(&target), Arc::clone(&during));
                async move {
                    let (ls, target) = target.get().expect("fixture: the target is set").clone();
                    let ls = ls
                        .upgrade()
                        .expect("fixture: the store outlives the holder");
                    during.lock().unwrap().free_permits =
                        Some(ls.admission_permits_available_for_test(target.oid));
                    let mut queued =
                        call_on(Arc::clone(&ls), target.clone(), structural_delta(2), "w");
                    first_poll_pends(&mut queued, "W, polled during the holder's backoff").await;
                    during.lock().unwrap().queued = Some(tokio::spawn(queued));
                    let fast = tokio::time::timeout(
                        STEP_BUDGET,
                        call_on(ls, target, build_valid_delta(), "fast"),
                    )
                    .await
                    .unwrap_or_else(|_| {
                        Err(ServiceError::Internal(
                            "the fast call was not admitted during the holder's backoff".into(),
                        ))
                    });
                    during.lock().unwrap().fast = Some(fast);
                }
                .boxed()
            }
        })
        .await;
        let fixture_target = Target {
            session_id: fx.session_id.clone(),
            layer: fx.layer,
            oid: fx.oid,
        };
        assert!(
            target.set((Arc::downgrade(&fx.ls), fixture_target)).is_ok(),
            "fixture: the target is set once"
        );

        let holder = tokio::spawn(call(&fx, structural_delta(1), "holder"));
        // Twice the step budget: the hook's own STEP_BUDGET wait for the fast
        // call runs inside this one, so a refused fast call reports itself
        // before the holder's wait expires.
        let (holder_seq, _) = settle_within(holder, 2 * STEP_BUDGET, "the holder")
            .await
            .expect("the holder must commit on its second attempt");
        let DuringBackoff {
            free_permits,
            queued,
            fast,
        } = std::mem::take(&mut *during.lock().unwrap());
        assert_eq!(
            free_permits.expect("fixture: the holder's between-attempts hook must have run"),
            Some((0, WRITERS as usize - 1)),
            "parked after its backoff sleep, the holder must still hold the object's \
             `structural` permit and one of its {WRITERS} `writers` permits — (free structural, \
             free writers)"
        );
        let (fast_seq, _) = fast
            .expect("fixture: the holder's between-attempts hook must have run")
            .expect("the fast call must commit during the holder's backoff");
        let (queued_seq, _) = settle(
            queued.expect("fixture: W was queued during the holder's backoff"),
            "W",
        )
        .await
        .expect("W must commit once the holder is done");

        assert!(
            fast_seq < holder_seq,
            "the fast call committed during the holder's backoff, so before the holder; got \
             fast={fast_seq}, holder={holder_seq}"
        );
        assert!(
            holder_seq < queued_seq,
            "W was queued while the holder held `structural` across its backoff, so W must \
             commit after the holder; got holder={holder_seq}, W={queued_seq}"
        );
    }

    // ── Retirement while calls hold the old entry ───────────────────────────

    /// Fail unless `outcome` is how a call on a retired object ends today.
    fn assert_retired_object_refusal(outcome: &CallResult, what: &str) {
        assert!(
            matches!(
                outcome,
                Err(ServiceError::NotFound(_) | ServiceError::FailedPrecondition(_))
            ),
            "{what} on the retired object must end NotFound / FailedPrecondition; got {outcome:?}"
        );
    }

    /// A structural call parked between its attempts (holding its object's
    /// admission entry and both permits) and a structural call queued on that
    /// entry's `structural` permit, while `retire` retires the object: the map
    /// entry and the object's structural-cache entry go at once, both calls
    /// then end NotFound / FailedPrecondition, and neither re-inserts an
    /// entry — they hold the old one, cloned out before the retirement.
    async fn retirement_under_running_calls(
        layer: &'static str,
        retire: impl for<'a> FnOnce(&'a Fixture) -> futures::future::BoxFuture<'a, ()>,
    ) {
        let (park, configure) = HolderPark::new(16);
        let fx = fixture(layer, configure).await;
        let holder = park_structural_holder(&fx, &park).await;
        let queued = queue(call(&fx, structural_delta(2), "queued"), "the queued call").await;
        cache_entry_for(&fx.ls, fx.oid);
        assert_eq!(
            fx.ls.semaphore_count(),
            1,
            "precondition: the object's admission entry is in the map"
        );

        tokio::time::timeout(STEP_BUDGET, retire(&fx))
            .await
            .expect("the retirement must finish while the calls hold the object's entry");

        assert_eq!(
            fx.ls.semaphore_count(),
            0,
            "{layer}: the retirement must drop the object's admission entry"
        );
        assert_retired(&fx.ls, fx.oid, layer);

        park.release();
        assert_retired_object_refusal(&settle(holder, "the holder").await, "the holder");
        assert_retired_object_refusal(&settle(queued, "the queued call").await, "the queued call");
        assert_eq!(
            fx.ls.semaphore_count(),
            0,
            "{layer}: calls that cloned the entry before the retirement must not put it back"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn delete_layer_retires_admission_and_cache_entries_under_running_structural_calls() {
        retirement_under_running_calls("admission-retire-delete", |fx| {
            async move {
                fx.ls
                    .delete_layer(&fx.session_id, fx.layer, mock_ext_store())
                    .await
                    .expect("delete_layer must succeed");
            }
            .boxed()
        })
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn create_layer_overwrite_retires_admission_and_cache_entries_under_running_structural_calls(
    ) {
        retirement_under_running_calls("admission-retire-overwrite", |fx| {
            async move {
                let new_oid = create_test_layer(&fx.ls, &fx.session_id, fx.layer).await;
                assert_ne!(new_oid, fx.oid, "an overwrite allocates a fresh object id");
            }
            .boxed()
        })
        .await;
    }
}
