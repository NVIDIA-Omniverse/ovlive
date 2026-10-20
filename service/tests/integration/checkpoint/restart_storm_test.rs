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

//! End-to-end DR restart-storm integration tests — Task `t2.fgj.mh.zp`,
//! Epic `t1.fgj.mh`, Bee `b.fgj`.
//!
//! This file co-locates the two restart-storm test bodies that share
//! the [`restart_storm_harness`] module:
//!
//! 1. `no_object_id_collision_under_restart_storm` (Subtask
//!    `t3.fgj.mh.zp.vy`, this Subtask) — Epic 6 AC #4 positive
//!    invariant. 100 sequential restarts × 50 `CreateLayer` calls
//!    under `safety_margin = 1000` MUST produce 5000 unique
//!    `object_id`s, AND the per-iteration maximum `object_id` MUST be
//!    strictly increasing across iterations.
//!
//! 2. `restart_storm_with_zero_safety_margin_surfaces_collision`
//!    (Subtask `t3.fgj.mh.zp.d8`) — negative-control proof that the
//!    safety margin is the load-bearing piece. Added by the sibling
//!    Subtask; the file pattern mirrors
//!    `e2e_object_id_preservation_test.rs` (one file, multiple
//!    `#[tokio::test]` functions).
//!
//! Both tests are `#[ignore]` per the slow-test convention; run via:
//!
//! ```text
//! cargo test -p live-edit --test integration --features \
//!     oodf-ffi/internal-bridge,test-checkpoint-fault-injection -- \
//!     --ignored --nocapture checkpoint::restart_storm_test
//! ```
//!
//! ## Why the strictly-increasing-max invariant is load-bearing
//!
//! The uniqueness invariant (5000 distinct ids) is the broad regression
//! surface — any global counter regression that recycles ids surfaces
//! there. But the strictly-increasing per-iteration max is the
//! *cleaner* regression surface, because it isolates the
//! restart-boundary bug class specifically:
//!
//! - If Task 4's `seed_object_id_counter` forgot to add the
//!   `safety_margin` term, the post-restart counter would land at
//!   exactly the pre-restart sidecar value. The very next CreateLayer
//!   would mint `object_id = pre_restart_max - K` (where K is the
//!   sidecar-vs-counter gap at the moment of the wipe), which is `<=
//!   pre_restart_max`. The strictly-increasing assertion fires
//!   loudly.
//!
//! - If Task 3's counters-sidecar producer tick missed the last batch
//!   of CreateLayer increments before the restart (e.g. the leader
//!   election never converged, or the tick cadence drifted past the
//!   harness's pre-restart `wait_for_counters_tick` budget), the
//!   sidecar would carry a stale value. Recovery would seed at
//!   `stale_value + safety_margin`, which can still land below
//!   `pre_restart_max` if the in-flight burst exceeded the safety
//!   margin. Same assertion fires.
//!
//! The uniqueness assertion catches both bugs too, but only by
//! observation of a duplicate id far downstream — the
//! strictly-increasing-max assertion catches them at the exact
//! restart boundary where the regression occurred.
//!
//! Sandbox env note: `TestValkey::setup` reads `TEST_VALKEY_URL`. Under
//! the project sandbox the URL must include the password and
//! `host.docker.internal` host; see CLAUDE.md.

#![cfg(feature = "test-checkpoint-fault-injection")]

use std::collections::HashSet;
use std::time::{Duration, Instant};

use live_edit::storage::session::SessionStore;

use super::restart_storm_harness::{
    create_layer_once, current_counter_value, wait_for_counters_tick, RestartStormHarness,
};

/// AC #4 dimensions. Pinned as `const` so the body reads cleanly and so
/// a future budget-reduction attempt that violates the parent Task's
/// "do NOT run with fewer than 100 iterations or fewer than 50 layers"
/// constraint surfaces as a literal-edit on these constants rather than
/// a silent change buried in a `for` head.
const ITERATIONS: usize = 100;
const LAYERS_PER_ITERATION: usize = 50;
const TOTAL_LAYERS: usize = ITERATIONS * LAYERS_PER_ITERATION;

/// Counter safety margin for the storm. Deliberately below the production
/// default (`CHECKPOINT_COUNTER_SAFETY_MARGIN_DEFAULT`, 1,000,000): the
/// seeding arithmetic is the same at any margin, and 1000 still exceeds the
/// ≤ 50 ids each iteration allocates between sidecar syncs, so the positive
/// test exercises production's seeding with small, readable floors.
///
/// This is the OBJECT-ID COUNTER margin consumed by
/// `seed_object_id_counter` — not the per-layer
/// `LIVE_EDIT_CHECKPOINT_SEQ_SAFETY_MARGIN` that `apply_checkpoint` adds
/// above a restored layer's sequence floor (whose default is 1000). The two
/// knobs read similarly, which is exactly why this comment names the right
/// one.
const SAFETY_MARGIN: u64 = 1000;

/// Counters-sidecar tick cadence. 1s keeps the producer-side blob
/// converging quickly between iterations without contending the
/// leader-elected loop's other work.
const COUNTERS_SYNC_SECS: u64 = 1;

/// Wall-clock budget for `wait_for_counters_tick` between batches.
/// 5s comfortably covers the 1s tick cadence plus jitter; if a
/// tick is missing here, the parent Task's "Task 3 producer tick
/// is not firing" diagnostic surfaces.
const COUNTERS_TICK_BUDGET: Duration = Duration::from_secs(5);

const SESSION_BASE_PATH: &str = "/stage/restart-storm";
const CREATED_BY: &str = "restart-storm-test";

/// Epic 6 AC #4 positive invariant: 100 sequential restarts ×
/// 50 CreateLayer calls produce 5000 unique `object_id`s, with the
/// per-iteration max strictly increasing across iterations.
///
/// Slow test — `#[ignore]` per the project's slow-test convention.
/// Run via:
///
/// ```text
/// cargo test -p live-edit --test integration --features \
///     oodf-ffi/internal-bridge,test-checkpoint-fault-injection -- \
///     --ignored --nocapture \
///     checkpoint::restart_storm_test::no_object_id_collision_under_restart_storm
/// ```
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "manual: slow, 100 restarts × 50 CreateLayer calls must give unique object_ids with a \
            strictly increasing max; run with --ignored (module docs)"]
async fn no_object_id_collision_under_restart_storm() {
    // Best-effort tracing init — `.ok()` because other tests in the
    // same binary may have already installed a global subscriber.
    tracing_subscriber::fmt()
        .with_env_filter("warn")
        .try_init()
        .ok();

    let mut harness = RestartStormHarness::setup_initial(SAFETY_MARGIN, COUNTERS_SYNC_SECS).await;

    // Pre-allocate at the exact final cardinality so a runaway test
    // (mis-edit of the loop bounds) does not silently rely on
    // `HashSet` re-hashing.
    let mut observed_ids: HashSet<u64> = HashSet::with_capacity(TOTAL_LAYERS);
    // First-N duplicate diagnostic. Kept as a Vec so the failure
    // message can render (iteration, layer_idx, colliding_id) tuples
    // in discovery order.
    let mut duplicates: Vec<(usize, usize, u64)> = Vec::new();
    let mut per_iter_max: Vec<u64> = Vec::with_capacity(ITERATIONS);

    let test_started = Instant::now();

    for iteration in 0..ITERATIONS {
        // ── Recreate the session. The prior restart's prefix-wipe
        //    cleared every `{prefix}*` key including the session
        //    meta + path index, so a fresh `SessionStore::create`
        //    is required. The session name carries the iteration
        //    index so a hung iteration surfaces in valkey-cli /
        //    Cluster diagnostics. ────────────────────────────────
        let session_store =
            SessionStore::new(harness.tv.client().clone(), harness.tv.key_schema().clone());
        let session_name = format!("restart-storm-session-iter-{iteration}");
        let session_id = session_store
            .create(&session_name, SESSION_BASE_PATH, CREATED_BY)
            .await
            .unwrap_or_else(|e| {
                panic!(
                    "SessionStore::create must succeed on iteration {iteration}; got error: {e}. \
                     A failure here on iteration 0 points at a TestValkey wiring regression; \
                     a failure on iteration N>0 points at a `restart()` wipe gap that left a \
                     conflicting path-index key behind."
                )
            });

        // ── Issue 50 CreateLayer calls. Each layer name carries
        //    iter+layer_idx so a duplicate `object_id` failure can
        //    be traced back to a specific (iter, idx) pair. ──────
        let mut iter_max: u64 = 0;
        for layer_idx in 0..LAYERS_PER_ITERATION {
            let layer_name = format!("iter_{iteration}_layer_{layer_idx}");
            let object_id = create_layer_once(&harness.svc, &session_id, &layer_name)
                .await
                .unwrap_or_else(|e| {
                    panic!(
                        "CreateLayer must succeed (iter={iteration}, idx={layer_idx}); \
                             got status: {e}. observed_ids.len()={}",
                        observed_ids.len()
                    )
                });
            let fresh = observed_ids.insert(object_id);
            if !fresh {
                duplicates.push((iteration, layer_idx, object_id));
                // Continue to populate `duplicates` so the final
                // assertion's diagnostic can list ALL collisions
                // discovered in the first failing iteration — useful
                // when triaging a counter-recycle regression that
                // produces a deterministic collision pattern. The
                // overall test still fails at the final
                // `assert_eq!`.
            }
            if object_id > iter_max {
                iter_max = object_id;
            }
        }

        assert!(
            iter_max > 0,
            "iter_max must be non-zero after 50 CreateLayers on iteration {iteration} \
             (the counter starts at 0 and INCR-then-read yields >= 1)"
        );

        // ── Wait for the counters-sidecar tick BEFORE the restart.
        //
        //    The lower bound on the wait MUST be `Instant::now()`
        //    captured AFTER the 50-CreateLayer burst completes, NOT
        //    the iteration's start instant. The producer-side tick
        //    captures `GET le:counter:object_id` at tick time, so a
        //    tick that fires mid-burst captures only a partial
        //    counter advance — leaving the post-restart seed routine
        //    at `partial_max + safety_margin` rather than at
        //    `iter_max + safety_margin`. If `partial_max + margin <
        //    iter_max`, the next iteration's CreateLayers collide
        //    with the previous-iteration's tail.
        //
        //    The parent Subtask body literally names
        //    `iter_start_instant` here, but doing so violates the
        //    body's own stated intent ("ensure the latest counter
        //    state is in the sidecar before the restart"). The
        //    deviation is intentional — see the module docstring's
        //    "Why the strictly-increasing-max invariant is
        //    load-bearing" section for the bug class this guards.
        let burst_completed = Instant::now();
        wait_for_counters_tick(&harness, burst_completed, COUNTERS_TICK_BUDGET).await;

        // ── The load-bearing restart. ─────────────────────────────
        harness = harness.restart().await;

        // ── Post-restart counter-floor assertion. Task 4's seed
        //    routine reads the sidecar's pre-restart max and seeds
        //    the live counter to `max + safety_margin`. Catching a
        //    regression where the seed routine forgets the safety
        //    margin (or applies it as `max(snap, snap)` rather than
        //    `max(snap, snap + margin)`) is the load-bearing
        //    failure surface here. ─────────────────────────────────
        let post_seed_counter = current_counter_value(&harness.tv).await;
        let expected_floor = iter_max + SAFETY_MARGIN;
        assert!(
            post_seed_counter >= expected_floor,
            "post-restart counter must satisfy >= per_iter_max + safety_margin \
             (iteration={iteration}, per_iter_max={iter_max}, \
              safety_margin={SAFETY_MARGIN}, expected_floor={expected_floor}, \
              actual={post_seed_counter}). A failure here points at Task 4's \
             seed_object_id_counter: either the safety_margin term was dropped, \
             or the sidecar's pre-restart max was stale (Task 3 tick missed the \
             final batch — but wait_for_counters_tick above should have ruled \
             that out)."
        );

        per_iter_max.push(iter_max);
    }

    let total_elapsed = test_started.elapsed();
    tracing::warn!(
        iterations = ITERATIONS,
        layers_per_iter = LAYERS_PER_ITERATION,
        total_layers = TOTAL_LAYERS,
        observed_ids = observed_ids.len(),
        elapsed_secs = total_elapsed.as_secs_f64(),
        "restart-storm positive test loop complete; running final assertions"
    );

    // ── Final uniqueness assertion. Renders the first 10
    //    duplicates inline so the failure message is actionable
    //    without re-running with extra logging. ──────────────────
    if !duplicates.is_empty() {
        let head: Vec<String> = duplicates
            .iter()
            .take(10)
            .map(|(iter, idx, oid)| format!("(iter={iter}, idx={idx}, object_id={oid})"))
            .collect();
        panic!(
            "object_id collision regression: observed {} duplicates across \
             {ITERATIONS} iterations × {LAYERS_PER_ITERATION} layers \
             (observed_ids.len()={}, expected={TOTAL_LAYERS}). \
             First 10 duplicates: [{}]. \
             Total duplicates: {}.",
            duplicates.len(),
            observed_ids.len(),
            head.join(", "),
            duplicates.len(),
        );
    }
    assert_eq!(
        observed_ids.len(),
        TOTAL_LAYERS,
        "expected {TOTAL_LAYERS} unique object_ids across the full restart-storm \
         (observed {} unique). No duplicates were recorded, which means the \
         shortfall came from missing CreateLayer responses — check the per-iteration \
         panic messages above.",
        observed_ids.len()
    );

    // ── Strictly-increasing per-iteration max. This is the
    //    cleaner restart-boundary regression surface — see
    //    module docstring. ────────────────────────────────────
    for i in 1..per_iter_max.len() {
        assert!(
            per_iter_max[i] > per_iter_max[i - 1],
            "per-iteration max object_id must be strictly increasing across restarts: \
             iteration {prev_iter} max={prev_max}, iteration {curr_iter} max={curr_max}. \
             A non-strict advance points at Task 4's seed routine (safety_margin term \
             dropped) or Task 3's producer tick (sidecar carrying stale max). \
             Full per_iter_max sequence: {per_iter_max:?}",
            prev_iter = i - 1,
            curr_iter = i,
            prev_max = per_iter_max[i - 1],
            curr_max = per_iter_max[i],
        );
    }

    // Drop the harness explicitly so the graceful-shutdown path runs
    // before the test exits — surfaces leaks in the per-restart
    // cancellation token / loop-handle plumbing.
    drop(harness);
}

// =============================================================================
// Test — safety_margin=0 negative control surfaces at least one collision.
// =============================================================================

/// Negative control for Epic 6 AC #4 — runs the restart-storm loop
/// with `LIVE_EDIT_CHECKPOINT_COUNTER_SAFETY_MARGIN=0` (the production
/// default is `1000`) and asserts that at least one `object_id`
/// collision is observed across 10 iterations × 50 layers per
/// iteration.
///
/// Subtask `t3.fgj.mh.zp.d8` under Task `t2.fgj.mh.zp`.
///
/// ## Why this test PASSES when it sees collisions (INVERSE-LOGIC)
///
/// Quoting parent Task `t2.fgj.mh.zp`'s body:
///
/// > This validates that the safety margin is what's preventing
/// > collisions, not some other coincidence.
///
/// The positive sibling above (`no_object_id_collision_under_restart_storm`)
/// demonstrates that with `safety_margin=1000`, 100 iterations × 50
/// layers produce 5000 unique `object_id`s. That test's pass condition
/// is "uniqueness".
///
/// THIS test's pass condition is the opposite: "collisions DO occur
/// when safety_margin=0". The mechanism the negative control surfaces
/// is the race between:
///
/// 1. CreateLayer increments the Valkey `counter:object_id` key.
/// 2. The Task-3 counters-sidecar tick reads the current counter and
///    writes it to the checkpoint adapter's sidecar blob.
/// 3. The simulated restart wipes the Valkey counter key but leaves
///    the sidecar blob intact.
/// 4. The Task-4 startup seed routine reads the sidecar blob and
///    seeds the Valkey counter to `sidecar.max_object_id_observed +
///    safety_margin`, via a seed-if-below script that leaves a
///    counter already at-or-above that value untouched.
///
/// With `safety_margin=0`, step (4) seeds the counter at exactly
/// `sidecar.max_object_id` — which is the value the sidecar tick
/// observed on its LAST FIRING before the restart, not the latest
/// counter value. Any CreateLayer call that incremented the counter
/// AFTER the sidecar tick's last firing and BEFORE the restart leaves
/// a "lost" range: the post-restart counter starts back at the
/// sidecar's stale value, and the next CreateLayer collides with the
/// pre-restart-allocated `object_id`(s) in that lost range.
///
/// The collisions are EXPECTED — the test fails LOUDLY if zero
/// collisions occur, because that would either mean:
/// - the `safety_margin=0` override is not being applied (something
///   is still defending the post-restart counter advance), OR
/// - the counters-sidecar tick is firing fast enough to fully cover
///   the per-iteration 50-CreateLayer burst (which would defeat the
///   margin's purpose entirely and break the positive sibling's
///   regression surface too), OR
/// - the test is no longer driving the restart-storm shape it
///   intended to (a wiring regression — much harder to diagnose
///   without this inverse-logic guard).
///
/// ## Do NOT "fix" this test into asserting uniqueness
///
/// A future maintainer who reads "the test asserts collisions occur"
/// and "fixes" it into asserting uniqueness would silently break the
/// load-bearing AC #4 regression surface. Do not soften, generalize,
/// or invert this test's assertions. If the test starts failing
/// spuriously (zero collisions observed), the right diagnostic path
/// is to investigate whether the safety_margin gate is leaking, NOT
/// to flip the assertion.
///
/// ## Tick-wait omitted DELIBERATELY
///
/// Unlike the positive sibling test above, this test does NOT call
/// `wait_for_counters_tick` between batches. The collision the
/// negative control demonstrates depends on the counter advancing
/// FASTER than the sidecar tick — waiting for the tick would mask
/// the bug by ensuring the sidecar always observes the latest
/// counter pre-restart. A 50ms sleep between batches keeps the loop
/// from busy-spinning without coupling to the sidecar cadence.
///
/// ## Sanity guards
///
/// - `assert!(!collisions.is_empty(), ...)` — the primary
///   load-bearing invariant. A future regression that accidentally
///   defended `safety_margin=0` against collisions MUST fail this.
/// - `assert!(observed_ids.len() < 500, ...)` — paranoid consistency
///   check. The expected size of `observed_ids` under collisions is
///   strictly LESS than 500 (= 10 × 50, the perfect-uniqueness
///   mark). A bug that inflates the set above 500 OR a state where
///   uniqueness was perfectly preserved but the collision counter
///   nevertheless registered a hit would surface here.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "manual: slow negative control, a zero safety margin must surface an object_id \
            collision; run with --ignored (module docs)"]
async fn restart_storm_with_zero_safety_margin_surfaces_collision() {
    // Best-effort tracing init — `.ok()` because other tests in the
    // same binary may have already installed a global subscriber
    // (the positive sibling above does the same).
    tracing_subscriber::fmt()
        .with_env_filter("warn")
        .try_init()
        .ok();

    // safety_margin=0 is the load-bearing knob this test exercises.
    // counters_sync_secs=1 mirrors the positive sibling — the Task-3
    // ticking still fires (the parent Task body's "Do NOT remove the
    // counters_sync_secs knob" constraint), but the safety margin is
    // zeroed out so the collisions surface inside the 10-iteration
    // budget.
    let mut harness = RestartStormHarness::setup_initial(0, COUNTERS_SYNC_SECS).await;

    // Set of every `object_id` observed across the whole run. Used
    // to detect collisions via `HashSet::insert`'s bool return.
    let mut observed_ids: HashSet<u64> = HashSet::new();

    // Records every (iteration, layer_idx, object_id) tuple that
    // collided with an earlier observation. Sized for the failure
    // diagnostic path — we surface the first 10 collisions on
    // assertion failure but record every one for completeness.
    let mut collisions: Vec<(usize, usize, u64)> = Vec::new();

    // Negative-control loop dimensions. 10 iterations × 50 layers
    // is the parent Task body's "10 iterations is enough to surface
    // this" budget.
    const NEG_ITERATIONS: usize = 10;
    const NEG_LAYERS_PER_ITERATION: usize = 50;
    const NEG_PERFECT_MARK: usize = NEG_ITERATIONS * NEG_LAYERS_PER_ITERATION;

    let test_started = Instant::now();

    for iteration in 0..NEG_ITERATIONS {
        // Recreate the session — the previous restart wiped its
        // Valkey keys (the harness's prefix-scoped SCAN + UNLINK
        // removed `le:{session_id}:sess` along with the layer
        // state). A fresh session per iteration keeps the
        // per-iteration burst self-contained and matches the
        // production restart shape where a replica boot finds a
        // wiped Valkey.
        let session_store =
            SessionStore::new(harness.tv.client().clone(), harness.tv.key_schema().clone());
        let session_name = format!("restart-storm-neg-session-iter-{iteration}");
        let session_id = session_store
            .create(&session_name, SESSION_BASE_PATH, CREATED_BY)
            .await
            .unwrap_or_else(|e| {
                panic!(
                    "SessionStore::create must succeed on iteration {iteration}; got error: {e}. \
                     A failure on iteration 0 points at TestValkey wiring; a failure on \
                     iteration N>0 points at a `restart()` wipe gap that left a path-index \
                     key behind."
                )
            });

        // Issue 50 CreateLayer calls. Each gets a unique layer name
        // so duplicate-layer-name validation does not mask the
        // object_id collision surface. Layer names carry the
        // (iteration, layer_idx) pair so a collision diagnostic can
        // point at the specific call.
        for layer_idx in 0..NEG_LAYERS_PER_ITERATION {
            let layer_name = format!("neg-iter-{iteration}-layer-{layer_idx}");
            let object_id = create_layer_once(&harness.svc, &session_id, &layer_name)
                .await
                .unwrap_or_else(|status| {
                    panic!(
                        "CreateLayer must succeed in negative-control test \
                         (iter={iteration}, idx={layer_idx}); got status: {status}. \
                         observed_ids.len()={}, collisions.len()={}",
                        observed_ids.len(),
                        collisions.len()
                    )
                });

            // The load-bearing observation: HashSet::insert returns
            // `false` if the object_id was already in the set —
            // i.e., a collision. We record but DO NOT panic: the
            // test wants to OBSERVE the full collision shape across
            // the whole run, not abort on the first hit. The final
            // assertion below converts the "no collisions observed"
            // case into a test failure.
            let fresh = observed_ids.insert(object_id);
            if !fresh {
                collisions.push((iteration, layer_idx, object_id));
            }
        }

        // Inter-batch sleep — deterministic without coupling to the
        // sidecar tick. See the doc-comment above for why this test
        // DOES NOT call `wait_for_counters_tick` here: doing so
        // would let the sidecar always observe the latest counter
        // pre-restart, defeating the collision the negative control
        // demonstrates.
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Restart — the load-bearing simulation step. Consumes the
        // current harness and returns a fresh one bound to the same
        // TestValkey prefix + tempdir.
        harness = harness.restart().await;
    }

    let total_elapsed = test_started.elapsed();
    tracing::warn!(
        iterations = NEG_ITERATIONS,
        layers_per_iter = NEG_LAYERS_PER_ITERATION,
        observed_ids = observed_ids.len(),
        collisions = collisions.len(),
        elapsed_secs = total_elapsed.as_secs_f64(),
        "negative-control loop complete; running final assertions"
    );

    // Diagnostic dump: surface the first 10 collisions on the
    // stderr stream so a CI failure carries the actionable
    // information without requiring a re-run with --nocapture. Each
    // line is unique so log-correlation across CI shards is trivial.
    // Done BEFORE the assertion so the dump always lands; if the
    // assertion fires the diagnostic is in the captured output.
    for (iter, layer_idx, oid) in collisions.iter().take(10) {
        eprintln!(
            "[negative-control collision] iter={iter} layer_idx={layer_idx} \
             colliding_object_id={oid}"
        );
    }

    // Environmental-flake guard. The collision-surfacing race depends on the
    // 10 × 50-CreateLayer bursts running faster than the 1s sidecar tick.
    // If the test loop ran so slowly that every burst landed inside its own
    // sidecar window (e.g., a heavily-contended CI host), the sidecar would
    // observe the latest counter pre-restart and seed loss-lessly even with
    // safety_margin=0, masking the bug we want to surface. The budget is
    // generous (10 iterations × ~1s each on idle hardware = ~10s expected;
    // we cap at 30s) — exceeding it points at CI-host slowness rather than
    // a leaking safety_margin gate. This is a SIBLING diagnostic — it does
    // NOT replace the load-bearing `collisions.is_empty()` assertion below.
    assert!(
        total_elapsed < Duration::from_secs(30),
        "negative-control loop ran in {:?}, exceeding the 30s flake-guard budget. \
         A failure on the immediately-following collisions assertion likely indicates \
         environmental slowness (slow CI host, host contention, scheduler stalls) \
         rather than a safety_margin regression — re-run on a quieter host before \
         treating this as a real bug. \
         (collisions.len()={}, observed_ids.len()={})",
        total_elapsed,
        collisions.len(),
        observed_ids.len()
    );

    // Primary load-bearing assertion. A future regression that
    // accidentally defended `safety_margin=0` against collisions —
    // by tightening the sidecar tick, by adding an unrelated counter
    // flush on shutdown, or by any other coincidence — MUST fail
    // here. Do NOT soften this assertion into "collisions or
    // uniqueness, either is fine".
    assert!(
        !collisions.is_empty(),
        "expected safety_margin=0 to surface at least one object_id collision across \
         {NEG_ITERATIONS} iterations × {NEG_LAYERS_PER_ITERATION} layers; got 0 — either \
         the safety_margin gate is being applied despite the override, or the sidecar \
         tick is firing faster than the test budgeted for. \
         observed_ids.len()={observed}",
        observed = observed_ids.len()
    );

    // Sanity assertion — uniqueness MUST drop below the
    // 500-perfect mark by at least one collision. This is a
    // paranoid consistency check: the primary assertion above proves
    // `collisions.len() >= 1`, and the perfect mark is
    // `NEG_ITERATIONS * NEG_LAYERS_PER_ITERATION = 500`, so
    // `observed_ids.len()` must be strictly less than 500 (because
    // every collision is by definition a non-fresh insert into the
    // set). A failure here would indicate a logic bug in the test
    // itself — e.g., a collision recorded against an object_id that
    // was NOT in the set, which the HashSet::insert semantics make
    // impossible. The assertion is cheap insurance against a future
    // test-helper rewrite that breaks the collision-recording
    // contract.
    assert!(
        observed_ids.len() < NEG_PERFECT_MARK,
        "negative-control consistency check failed: observed_ids.len() = {} MUST \
         be strictly less than the perfect-uniqueness mark ({NEG_PERFECT_MARK}) when \
         at least one collision was recorded (collisions.len() = {}). This is a \
         test-internal invariant; a failure here indicates the collision-recording \
         path drifted from HashSet::insert semantics.",
        observed_ids.len(),
        collisions.len()
    );

    // Cross-check (diagnostic only — not asserted because exact
    // value is timing-dependent): the post-restart counter value
    // SHOULD be small with safety_margin=0, far below what a
    // `safety_margin=1000` run would produce after 10 restarts
    // (≥ 10 × 1000 = 10000). Reading via the harness's TestValkey
    // client avoids going through the LiveEditService shell.
    let final_counter = current_counter_value(&harness.tv).await;
    tracing::warn!(
        final_counter,
        collisions = collisions.len(),
        "negative-control test complete: collisions surfaced as expected"
    );

    // Drop the harness explicitly so the graceful-shutdown path
    // runs before the test exits — surfaces leaks in the
    // per-restart cancellation token / loop-handle plumbing
    // (matches the positive sibling above).
    drop(harness);
}
