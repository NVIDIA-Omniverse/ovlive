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

//! Integration tests for `seed_object_id_counter` + readiness watch signal
//! (SDD SR-5.3 / SR-5.4). Subtask `t3.fgj.mh.4u.ja` under Task
//! `t2.fgj.mh.4u`, Epic `t1.fgj.mh`.
//!
//! One `#[tokio::test]` function per parent-Task Success Criteria bullet,
//! plus the b.b2j-era additions (2b, and Scenario 7's retry cases at the
//! bottom of the file):
//!
//! 1. `seeded_outcome_advances_counter` — sidecar present, counter below
//!    `seed = max_object_id_observed + safety_margin` → `Seeded { seed,
//!    new_counter == seed }` and the next `next_object_id()` returns
//!    `seed + 1`.
//! 2. `absent_outcome_leaves_counter_unchanged` — no sidecar written →
//!    `Absent` and the counter retains its pre-call value (the routine
//!    must not allocate or reset).
//!    - `absent_outcome_establishes_counter_key_at_zero` — the same
//!      no-sidecar path against a counter key that does not exist at all →
//!      `Absent` AND `le:counter:object_id` established at `0` (b.b2j). The
//!      write is what makes `next_object_id`'s absent-counter verdict mean
//!      "the keyspace was lost" rather than "nobody has created a layer yet".
//! 3. `noop_outcome_when_counter_already_above_seed` — sidecar present
//!    but `le:counter:object_id` already at-or-above `seed` →
//!    `NoOp { existing }` and the counter is untouched.
//! 4. `parse_failed_on_corrupt_sidecar_json` — corrupt bytes AND a
//!    schema-mismatch JSON case both surface as `Err(ParseFailed)`; the
//!    counter is NOT advanced.
//! 5. `readiness_signal_flips_on_success_stays_false_on_error` — drives a
//!    thin test wrapper that mirrors the `main.rs` orchestrator-side
//!    decision (flip the `watch::Sender<bool>` to `true` on `Ok`, leave
//!    it `false` on `Err`). Three sub-cases — `Seeded`, `Absent`,
//!    `ParseFailed` — covered in one test function with a fresh channel
//!    each.
//! 6. `counter_seed_secs_histogram_observed_at_least_once` — asserts the
//!    `live_edit_checkpoint_counter_seed_secs` histogram family is
//!    registered AND its sample count + sum advance on every routine
//!    entry (not only on `Seeded`).
//!
//! Sandbox env note: these tests use `TestValkey::setup`, which reads
//! `TEST_VALKEY_URL`. Under the project sandbox the URL must include the
//! password and `host.docker.internal` host; see CLAUDE.md.

use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use fred::interfaces::KeysInterface;
use tokio::sync::watch;
use tokio::time::timeout;

use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
use live_edit::checkpoint::counters::CountersSidecar;
use live_edit::checkpoint::{seed_object_id_counter, CheckpointStore, SeedError, SeedOutcome};
use live_edit::observability::metrics::SharedMetrics;
use live_edit::storage::counters::CounterStore;
use live_edit::storage::test_support::TestValkey;

const SERVICE_LABEL: &str = "counter-seed-test-replica";

// ─────────────────────────────────────────────────────────────────────────
// Shared helpers.
// ─────────────────────────────────────────────────────────────────────────

/// Build a fresh `SharedMetrics`. Scenario 6 confirms the
/// `live_edit_checkpoint_counter_seed_secs` family is exposed by rendering the
/// scrape exposition (`test_util::render`) rather than a `prometheus::Registry`.
fn make_metrics() -> SharedMetrics {
    Arc::new(crate::common::metrics_scrape_only(SERVICE_LABEL))
}

/// Encode a `CountersSidecar` via `serde_json::to_vec` and persist it
/// through `CheckpointStore::put_counters_sidecar`. Scenarios 1, 3, 5,
/// and 6 use this path to pin the producer/consumer contract (round-trip
/// through the shared struct, not hand-rolled JSON). Scenario 4
/// INTENTIONALLY hand-rolls bytes to exercise the parse-failure arm.
async fn put_sidecar(store: &dyn CheckpointStore, max_object_id_observed: u64) {
    let sidecar = CountersSidecar {
        max_object_id_observed,
        updated_at: "2026-05-22T16:00:00Z".to_string(),
    };
    let bytes = serde_json::to_vec(&sidecar).expect("CountersSidecar serialization must succeed");
    store
        .put_counters_sidecar(Bytes::from(bytes))
        .await
        .expect("put_counters_sidecar must succeed on InMemory adapter");
}

/// Delete `le:counter:object_id` so a subsequent `next_object_id()` starts
/// from `1` and a subsequent `seed_object_id_counter` exercises the
/// "counter strictly below seed" arm of the conditional Lua. Uses the
/// `KeySchema` from `TestValkey` so the test-prefix is honoured.
async fn wipe_object_id_counter(tv: &TestValkey) {
    let key = tv.key_schema().counter_object_id();
    let _: () = tv
        .client()
        .pool
        .del(key)
        .await
        .expect("DEL le:counter:object_id must succeed");
}

/// SET `le:counter:object_id` to an explicit value. Scenario 3 uses this
/// to drive the "counter already at-or-above seed" arm.
async fn set_object_id_counter(tv: &TestValkey, value: u64) {
    let key = tv.key_schema().counter_object_id();
    let _: () = tv
        .client()
        .pool
        .set(key, value.to_string(), None, None, false)
        .await
        .expect("SET le:counter:object_id must succeed");
}

/// Read the RAW `le:counter:object_id` value — `None` when the key does not
/// exist.
///
/// Deliberately not [`read_counter`], which routes through
/// `CounterStore::current_object_id` and reports `0` for BOTH "absent" and
/// "present and zero". Since b.b2j that conflation is exactly what the
/// `Absent` arm's `SET NX 0` exists to remove — allocation treats an absent
/// counter as "the Valkey keyspace was lost", a verdict that is only
/// unambiguous if the key is otherwise guaranteed to exist — so the test that
/// pins it needs to see the two states apart.
async fn raw_counter(tv: &TestValkey) -> Option<String> {
    tv.client()
        .pool
        .get(tv.key_schema().counter_object_id())
        .await
        .expect("GET le:counter:object_id must succeed")
}

/// Read `le:counter:object_id` via the public `CounterStore` API. Returns
/// `0` for the pre-first-allocate state.
async fn read_counter(counters: &CounterStore) -> u64 {
    counters
        .current_object_id()
        .await
        .expect("current_object_id read must succeed")
}

/// Thin wrapper that mirrors `main.rs`'s orchestrator-side decision: call
/// the routine, and on `Ok(_)` flip the readiness watch signal to `true`.
/// On `Err(_)` the wrapper does NOT touch the signal — the supervisor
/// keeps readiness at NOT_SERVING and the operator sees the failing probe.
///
/// All three `SeedOutcome` variants (`Absent`, `Seeded`, `NoOp`) flip the
/// gate per the parent Task body — `Absent` is the legitimate
/// "no sidecar to seed, AOF is doing the job" path.
async fn run_with_readiness_wrapper(
    store: Arc<dyn CheckpointStore>,
    counters: Arc<CounterStore>,
    safety_margin: u64,
    metrics: SharedMetrics,
    tx: &watch::Sender<bool>,
) -> Result<SeedOutcome, SeedError> {
    let result = seed_object_id_counter(store, counters, safety_margin, metrics).await;
    if result.is_ok() {
        // `send` only errors if every receiver has been dropped; in tests
        // the caller holds the receiver across the wrapper call, so any
        // send-failure here is a test-setup bug rather than a production
        // condition — fall back to a hard assertion so the bug surfaces
        // at the closest possible call site.
        tx.send(true).expect("readiness watch send must succeed");
    }
    result
}

// ─────────────────────────────────────────────────────────────────────────
// Scenario 1 — `Seeded` outcome advances counter to seed.
// ─────────────────────────────────────────────────────────────────────────

/// Sidecar with `max_object_id_observed = 5000`, safety margin `1000`,
/// counter wiped → `Seeded { seed: 6000, new_counter: 6000 }` and the
/// next `next_object_id()` returns `6001`.
///
/// Pre-fix regression mode (Lua conditional broken or wired with
/// `<=` instead of `<`): the counter would either remain at `0` or
/// advance past `seed`, both of which would fail the
/// `next_object_id() == 6001` post-condition.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn seeded_outcome_advances_counter() {
    let tv = TestValkey::setup().await;
    let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
    let counters = Arc::new(CounterStore::new(
        tv.client().clone(),
        tv.key_schema().clone(),
    ));
    let metrics = make_metrics();

    put_sidecar(store.as_ref(), 5000).await;
    wipe_object_id_counter(&tv).await;

    let outcome = seed_object_id_counter(Arc::clone(&store), Arc::clone(&counters), 1000, metrics)
        .await
        .expect("seed_object_id_counter must succeed on happy-path sidecar");

    assert!(
        matches!(
            outcome,
            SeedOutcome::Seeded {
                seed: 6000,
                new_counter: 6000
            }
        ),
        "expected Seeded {{ seed: 6000, new_counter: 6000 }}; got {outcome:?}"
    );

    let next = counters
        .next_object_id()
        .await
        .expect("next_object_id must succeed after seeding")
        .expect("seeding established the counter key, so allocation must not fail closed");
    assert_eq!(
        next, 6001,
        "next_object_id after a Seeded outcome must INCR from the seed value: \
         expected 6001 (= seed + 1); got {next}. A different value here means \
         the Lua conditional-SET wrote a wrong value, or the post-script INCR \
         saw a stale read."
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Scenario 2 — `Absent` outcome leaves counter untouched.
// ─────────────────────────────────────────────────────────────────────────

/// No sidecar present, counter at `3` (after three pre-call
/// `next_object_id` calls) → `Absent` and the counter is unchanged
/// (post-call `next_object_id()` returns `4`).
///
/// Pre-fix regression mode (the routine accidentally writes a default
/// seed of `0` or the safety margin alone): the counter would either
/// reset to `0` / `1000` or refuse to advance, both failing the
/// `next_object_id() == 4` post-condition.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn absent_outcome_leaves_counter_unchanged() {
    let tv = TestValkey::setup().await;
    let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
    let counters = Arc::new(CounterStore::new(
        tv.client().clone(),
        tv.key_schema().clone(),
    ));
    let metrics = make_metrics();

    // Intentionally do NOT call `put_sidecar` — `Absent` is the
    // pre-first-checkpoint state. Drive the counter to `3` so the
    // post-call assertion can distinguish "unchanged" from a routine
    // that accidentally writes `0` or `safety_margin`.
    // b.b2j: the allocator fails closed on an absent `le:counter:object_id`,
    // so establish the key the way the startup path does before pre-rolling.
    counters
        .initialize_object_id_counter_if_absent()
        .await
        .expect("establishing le:counter:object_id must succeed");
    for _ in 0..3 {
        let _ = counters
            .next_object_id()
            .await
            .expect("next_object_id pre-roll must succeed")
            .expect("counter key was just established, so allocation must not fail closed");
    }
    assert_eq!(
        read_counter(&counters).await,
        3,
        "pre-call sanity: counter should sit at 3 after three INCRs"
    );

    let outcome = seed_object_id_counter(Arc::clone(&store), Arc::clone(&counters), 1000, metrics)
        .await
        .expect("seed_object_id_counter must succeed on no-sidecar path");

    assert!(
        matches!(outcome, SeedOutcome::Absent),
        "expected Absent on the no-sidecar path; got {outcome:?}"
    );

    let next = counters
        .next_object_id()
        .await
        .expect("next_object_id must succeed after Absent")
        .expect("the counter key established above must survive the Absent arm");
    assert_eq!(
        next, 4,
        "next_object_id after an Absent outcome must continue the pre-call \
         counter line (3 → 4); got {next}. A different value means the routine \
         touched le:counter:object_id despite no sidecar being present."
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Scenario 2b — the `Absent` arm's `SET NX 0` side effect (b.b2j).
// ─────────────────────────────────────────────────────────────────────────

/// No sidecar AND no counter key → `Absent`, **and the routine establishes
/// `le:counter:object_id` at `0` on its way out**.
///
/// The sibling [`absent_outcome_leaves_counter_unchanged`] deliberately
/// pre-establishes the counter, so it pins the "does not disturb an existing
/// counter" half and is structurally blind to this one. This test covers the
/// genuinely-virgin case — first boot of a deployment with an adapter
/// configured but nothing checkpointed yet — where the write actually happens.
///
/// The write is what makes the whole b.b2j fail-closed design coherent.
/// `next_object_id` refuses to allocate against an absent counter and reports
/// that absence as "the Valkey keyspace was lost underneath this process". That
/// reading is only sound if the key is otherwise guaranteed to exist: nothing
/// in the service ever deletes it, and this `SET NX` is what creates it in the
/// first place. Drop this line and the very first `CreateLayer` of every fresh
/// deployment takes the keyspace-loss branch — a spurious repair, a spurious
/// ERROR-level "investigate the backing store" on a perfectly healthy pod.
///
/// `NX`, not a bare `SET`, and this test pins the difference at the value: `0`
/// rather than a peer's real seed would rewind a live counter, which is a
/// strictly worse bug than the one being fixed. (The race itself is pinned by
/// `initialize_object_id_counter_if_absent_does_not_stomp_a_seeded_counter` in
/// `src/storage/counters.rs`; here the claim is only that the routine's Absent
/// arm performs the write at all, at the right value.)
///
/// The trailing allocation assertion closes the loop: an established-at-zero
/// counter must hand the first `CreateLayer` id `1`. Asserting the key's
/// existence alone would pass on a routine that wrote some other value.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn absent_outcome_establishes_counter_key_at_zero() {
    let tv = TestValkey::setup().await;
    let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
    let counters = Arc::new(CounterStore::new(
        tv.client().clone(),
        tv.key_schema().clone(),
    ));
    let metrics = make_metrics();

    // Intentionally no `put_sidecar` and — unlike the sibling test — no
    // pre-establishment either. `TestValkey` mints a fresh per-test prefix, so
    // the counter is absent by construction, which is the first-boot shape.
    assert_eq!(
        raw_counter(&tv).await,
        None,
        "fixture precondition: a fresh test prefix must have no counter key at all \
         (not merely a counter reading zero)"
    );

    let outcome = seed_object_id_counter(Arc::clone(&store), Arc::clone(&counters), 1000, metrics)
        .await
        .expect("seed_object_id_counter must succeed on the no-sidecar path");

    assert!(
        matches!(outcome, SeedOutcome::Absent),
        "expected Absent on the no-sidecar path; got {outcome:?}"
    );

    assert_eq!(
        raw_counter(&tv).await.as_deref(),
        Some("0"),
        "the Absent arm MUST leave le:counter:object_id existing and holding `0`. Absent \
         here means the allocator's fail-closed branch fires on every first CreateLayer of \
         every fresh deployment, reporting keyspace loss on a healthy pod; a value other \
         than `0` means the first allocation skips ids nothing ever claimed"
    );

    let first = counters
        .next_object_id()
        .await
        .expect("next_object_id must succeed once the counter exists")
        .expect("the Absent arm established the key, so allocation must not fail closed");
    assert_eq!(
        first, 1,
        "an established-at-0 counter must hand the first CreateLayer id 1; got {first}"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Scenario 3 — `NoOp` outcome when counter already above seed.
// ─────────────────────────────────────────────────────────────────────────

/// Sidecar with `max_object_id_observed = 5000`, safety margin `1000`
/// (so `seed = 6000`), but `le:counter:object_id` already at `7000` →
/// `NoOp { existing: 7000 }` and the counter is unchanged (post-call
/// `next_object_id()` returns `7001`).
///
/// Pre-fix regression mode (the Lua condition is `>=` instead of `<`):
/// the counter would be re-SET to `6000` and the post-call
/// `next_object_id()` would return `6001`, NOT `7001`. Pinning `7001`
/// rejects both the silently-rewinding-counter regression AND a
/// "no-conditional, always-SET" implementation.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn noop_outcome_when_counter_already_above_seed() {
    let tv = TestValkey::setup().await;
    let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
    let counters = Arc::new(CounterStore::new(
        tv.client().clone(),
        tv.key_schema().clone(),
    ));
    let metrics = make_metrics();

    put_sidecar(store.as_ref(), 5000).await;
    set_object_id_counter(&tv, 7000).await;

    let outcome = seed_object_id_counter(Arc::clone(&store), Arc::clone(&counters), 1000, metrics)
        .await
        .expect("seed_object_id_counter must succeed on NoOp path");

    assert!(
        matches!(outcome, SeedOutcome::NoOp { existing: 7000 }),
        "expected NoOp {{ existing: 7000 }} (sidecar says 5000+1000=6000 but \
         counter already at 7000); got {outcome:?}"
    );

    let next = counters
        .next_object_id()
        .await
        .expect("next_object_id must succeed after NoOp")
        .expect("the counter was pre-set to 7000, so allocation must not fail closed");
    assert_eq!(
        next, 7001,
        "next_object_id after a NoOp outcome must continue from the existing \
         counter value (7000 → 7001); got {next}. Any other value means the \
         routine silently rewrote le:counter:object_id despite the conditional \
         Lua's no-op branch."
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Scenario 4 — `SeedError::ParseFailed` on corrupt / mismatched sidecar.
// ─────────────────────────────────────────────────────────────────────────

/// Both a corrupt-bytes sidecar and a schema-mismatch JSON sidecar
/// surface as `Err(SeedError::ParseFailed)`. In both cases the counter
/// is NOT advanced.
///
/// The two-case structure pins that `ParseFailed` covers BOTH the
/// "not valid JSON at all" path and the "valid JSON but missing required
/// field" path — `serde_json::from_slice::<CountersSidecar>` rejects
/// both, and the routine maps either onto the same `ParseFailed` variant
/// without quietly defaulting `max_object_id_observed` to `0` (which
/// would produce a `Seeded { seed: safety_margin, ... }` outcome that
/// silently rewinds the counter past every checkpointed `object_id`).
///
/// Each case uses a fresh `TestValkey` to keep per-case namespace
/// isolation; the `InMemoryCheckpointStore` is fresh too so the corrupt
/// bytes from case A do not leak into case B.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn parse_failed_on_corrupt_sidecar_json() {
    // Case A: garbage bytes (not even valid UTF-8 in places, definitely
    // not JSON). Intentionally hand-rolled per the Subtask body's
    // "Scenario 4 INTENTIONALLY hand-rolls bytes" note.
    {
        let tv = TestValkey::setup().await;
        let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
        let counters = Arc::new(CounterStore::new(
            tv.client().clone(),
            tv.key_schema().clone(),
        ));
        let metrics = make_metrics();

        store
            .put_counters_sidecar(Bytes::from_static(b"\x00not-json{"))
            .await
            .expect("put_counters_sidecar must succeed on bytes (parse happens later)");
        wipe_object_id_counter(&tv).await;

        let result =
            seed_object_id_counter(Arc::clone(&store), Arc::clone(&counters), 1000, metrics).await;
        assert!(
            matches!(result, Err(SeedError::ParseFailed(_))),
            "Case A (corrupt bytes): expected Err(SeedError::ParseFailed(_)); got {result:?}"
        );

        // Counter MUST NOT have been advanced. Pre-call wipe puts it at
        // absent (`current_object_id` returns 0); a routine that
        // silently defaulted `max_object_id_observed = 0` on parse
        // failure would still produce `seed = 0 + 1000 = 1000` and the
        // counter would be 1000 here.
        assert_eq!(
            read_counter(&counters).await,
            0,
            "Case A: counter must remain at 0 on ParseFailed — a non-zero value \
             here means the routine silently defaulted on parse failure and \
             advanced the counter (which would later collide with checkpointed \
             object_ids)."
        );
    }

    // Case B: well-formed JSON but the required field is missing
    // (`max_object_id_observed` absent). `serde_json::from_slice::<
    // CountersSidecar>` must reject this with the same `ParseFailed`
    // arm — the routine must NOT silently default the missing field.
    {
        let tv = TestValkey::setup().await;
        let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
        let counters = Arc::new(CounterStore::new(
            tv.client().clone(),
            tv.key_schema().clone(),
        ));
        let metrics = make_metrics();

        store
            .put_counters_sidecar(Bytes::from_static(
                br#"{"updated_at": "2026-05-21T16:00:00Z"}"#,
            ))
            .await
            .expect("put_counters_sidecar must succeed on bytes");
        wipe_object_id_counter(&tv).await;

        let result =
            seed_object_id_counter(Arc::clone(&store), Arc::clone(&counters), 1000, metrics).await;
        assert!(
            matches!(result, Err(SeedError::ParseFailed(_))),
            "Case B (schema mismatch): expected Err(SeedError::ParseFailed(_)); got {result:?}"
        );

        assert_eq!(
            read_counter(&counters).await,
            0,
            "Case B: counter must remain at 0 on schema-mismatch ParseFailed — \
             same rationale as Case A."
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Scenario 5 — readiness watch signal transitions.
// ─────────────────────────────────────────────────────────────────────────

/// Three sub-cases driving the test wrapper (`run_with_readiness_wrapper`)
/// that mirrors `main.rs`'s orchestrator-side decision:
///
/// - `Seeded` outcome → watch flips `false → true`.
/// - `Absent` outcome → watch flips `false → true` (per parent Task body
///   "Success Criteria … flips … after a successful Seeded or Absent
///   outcome").
/// - `ParseFailed` outcome → watch STAYS at `false` (the
///   orchestrator-side decision does NOT touch the sender on `Err`, so
///   the supervisor keeps readiness at NOT_SERVING and Kubernetes pulls
///   the pod).
///
/// Three sub-cases share one `#[tokio::test]` because each case
/// constructs a fresh `(tx, rx) = watch::channel(false)` — no cross-case
/// state pollution. Per the Subtask "no naked tokio::time::sleep" rule
/// we use `*rx.borrow()` synchronously after the awaited wrapper call;
/// there is no across-task wait, so `rx.changed().await` + `timeout` are
/// unnecessary here.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn readiness_signal_flips_on_success_stays_false_on_error() {
    // ── Sub-case A: success via `Seeded` flips the gate.
    {
        let tv = TestValkey::setup().await;
        let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
        let counters = Arc::new(CounterStore::new(
            tv.client().clone(),
            tv.key_schema().clone(),
        ));
        let metrics = make_metrics();

        put_sidecar(store.as_ref(), 5000).await;
        wipe_object_id_counter(&tv).await;

        let (tx, rx) = watch::channel(false);
        assert!(!*rx.borrow(), "pre-call sanity: watch must start at false");

        let result = run_with_readiness_wrapper(
            Arc::clone(&store),
            Arc::clone(&counters),
            1000,
            metrics,
            &tx,
        )
        .await
        .expect("wrapper must return Ok on Seeded path");
        assert!(
            matches!(result, SeedOutcome::Seeded { .. }),
            "Sub-case A: expected Seeded outcome; got {result:?}"
        );

        assert!(
            *rx.borrow(),
            "Sub-case A: watch must flip to true after a Seeded outcome — the \
             orchestrator-side decision (mirrored by `run_with_readiness_wrapper`) \
             flips on every `Ok(_)`. A false value here would mean the wrapper \
             diverged from main.rs's contract."
        );
    }

    // ── Sub-case B: success via `Absent` ALSO flips the gate.
    {
        let tv = TestValkey::setup().await;
        let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
        let counters = Arc::new(CounterStore::new(
            tv.client().clone(),
            tv.key_schema().clone(),
        ));
        let metrics = make_metrics();

        // Intentionally no put_sidecar — `Absent` is the legitimate
        // "no sidecar to seed" path and the Task body says it MUST
        // flip the gate.
        wipe_object_id_counter(&tv).await;

        let (tx, rx) = watch::channel(false);
        assert!(!*rx.borrow(), "pre-call sanity: watch must start at false");

        let result = run_with_readiness_wrapper(
            Arc::clone(&store),
            Arc::clone(&counters),
            1000,
            metrics,
            &tx,
        )
        .await
        .expect("wrapper must return Ok on Absent path");
        assert!(
            matches!(result, SeedOutcome::Absent),
            "Sub-case B: expected Absent outcome; got {result:?}"
        );

        assert!(
            *rx.borrow(),
            "Sub-case B: watch must flip to true after an Absent outcome — \
             `Absent` is the legitimate no-sidecar path (e.g., first-ever \
             deployment) and the parent Task body explicitly requires the \
             gate to flip. A false value here would block readiness on every \
             fresh deployment forever."
        );
    }

    // ── Sub-case C: `ParseFailed` keeps the gate at false.
    {
        let tv = TestValkey::setup().await;
        let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
        let counters = Arc::new(CounterStore::new(
            tv.client().clone(),
            tv.key_schema().clone(),
        ));
        let metrics = make_metrics();

        store
            .put_counters_sidecar(Bytes::from_static(b"\x00not-json{"))
            .await
            .expect("put_counters_sidecar must succeed on bytes");
        wipe_object_id_counter(&tv).await;

        let (tx, rx) = watch::channel(false);
        assert!(!*rx.borrow(), "pre-call sanity: watch must start at false");

        let result = run_with_readiness_wrapper(
            Arc::clone(&store),
            Arc::clone(&counters),
            1000,
            metrics,
            &tx,
        )
        .await;
        assert!(
            matches!(result, Err(SeedError::ParseFailed(_))),
            "Sub-case C: expected Err(SeedError::ParseFailed(_)); got {result:?}"
        );

        assert!(
            !*rx.borrow(),
            "Sub-case C: watch must STAY at false after a ParseFailed error — \
             the orchestrator-side decision does not touch the sender on \
             Err, so the supervisor keeps readiness at NOT_SERVING and \
             Kubernetes pulls the pod (the operator-facing safety net for a \
             corrupt sidecar). A true value here would route CreateLayer \
             traffic at a replica with an un-seeded counter."
        );

        // Defensive: bound any latent watch propagation. The wrapper
        // mutates the sender synchronously on success and not at all on
        // error, so a `changed()` notification on the `false → false`
        // path must not arrive — wait with a tight 200 ms budget and
        // assert it does not fire. The bound matches the "no naked
        // sleep, bound with timeout" Subtask rule.
        let mut rx_mut = rx;
        let waited = timeout(Duration::from_millis(200), rx_mut.changed()).await;
        assert!(
            waited.is_err(),
            "Sub-case C: a `changed()` notification must NOT fire on the \
             ParseFailed path (sender was never touched). Receiving one here \
             means the wrapper accidentally `send`s on Err — see \
             `run_with_readiness_wrapper`'s contract."
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Scenario 6 — `live_edit_checkpoint_counter_seed_secs` histogram observed.
// ─────────────────────────────────────────────────────────────────────────

/// One `Seeded` call followed by one `NoOp` call (the counter is now
/// at-or-above seed). The histogram must:
///
/// 1. Be registered on the `prometheus::Registry` (visible to scrapes).
/// 2. Have `get_sample_count() >= 1` after the first call.
/// 3. Have `get_sample_sum() > 0.0` (real elapsed time was observed).
/// 4. Have `get_sample_count() >= 2` after the second call — pins that
///    the histogram fires on EVERY routine entry (not only the `Seeded`
///    arm). The routine's `ObserveOnDrop` guard is armed on every exit
///    path except `SeedError::Store`; the `NoOp` arm leaves it armed,
///    so two routine calls must produce two samples.
///
/// Pre-fix regression mode where the histogram is only observed on
/// `Seeded` (forgotten `Drop` impl, or an early-return-without-observe
/// path through `NoOp`): `sample_count` would stick at `1` after the
/// second call, failing the `>= 2` assertion.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn counter_seed_secs_histogram_observed_at_least_once() {
    let tv = TestValkey::setup().await;
    let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
    let counters = Arc::new(CounterStore::new(
        tv.client().clone(),
        tv.key_schema().clone(),
    ));
    let metrics = make_metrics();

    // Pre-call sanity: the histogram family is absent until first observation.
    assert!(
        !live_edit::observability::metrics::test_util::has_metric(
            &live_edit::observability::metrics::test_util::render(&metrics),
            "live_edit_checkpoint_counter_seed_secs",
        ),
        "fresh Metrics must expose no samples on \
         live_edit_checkpoint_counter_seed_secs before the routine runs"
    );

    put_sidecar(store.as_ref(), 1000).await;
    wipe_object_id_counter(&tv).await;

    // First call — exercises the `Seeded` arm.
    let outcome_a = seed_object_id_counter(
        Arc::clone(&store),
        Arc::clone(&counters),
        1000,
        metrics.clone(),
    )
    .await
    .expect("first routine call must succeed");
    assert!(
        matches!(outcome_a, SeedOutcome::Seeded { .. }),
        "Scenario 6 pre-condition: first call must produce Seeded; got {outcome_a:?}"
    );

    let count_after_first = live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(&metrics),
        "live_edit_checkpoint_counter_seed_secs_count",
        &[],
    )
    .unwrap_or(0.0);
    let sum_after_first = live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(&metrics),
        "live_edit_checkpoint_counter_seed_secs_sum",
        &[],
    )
    .unwrap_or(0.0);
    assert!(
        count_after_first >= 1.0,
        "after one routine call sample_count must be >= 1; got {count_after_first}. \
         Zero here means the `ObserveOnDrop` guard never fired — the histogram \
         contract is broken."
    );
    assert!(
        sum_after_first > 0.0,
        "after one routine call sample_sum must be > 0.0 (real elapsed time); \
         got {sum_after_first}. A zero sum with a non-zero count would mean \
         the observer recorded zero-duration samples (a wall-clock bug)."
    );

    // The histogram family must appear in the rendered `/metrics` exposition —
    // that is what Prometheus actually scrapes. Asserting on the rendered
    // scrape surface (not an in-memory handle) catches a regression where the
    // metric is observed but not wired into the scrape reader.
    assert!(
        live_edit::observability::metrics::test_util::has_metric(
            &live_edit::observability::metrics::test_util::render(&metrics),
            "live_edit_checkpoint_counter_seed_secs",
        ),
        "live_edit_checkpoint_counter_seed_secs family must appear in the rendered \
         /metrics output — Prometheus scrapes this surface. Missing here means the \
         metric was created without being wired into the scrape reader."
    );

    // Push the counter strictly past `seed` so the second routine call
    // hits the genuine `NoOp` arm (where `new_counter > seed`). Without
    // this INCR the second call would land on the `new_counter == seed`
    // boundary that the routine's docstring explicitly maps to
    // `Seeded` ("a NoOp masquerading as a Seeded" — see
    // `src/checkpoint/counter_seed.rs` lines 207-219). The histogram
    // assertion does not care which arm fires — only that the routine
    // observes the histogram on every entry — but pinning the test on
    // the genuine `NoOp` outcome (rather than the masquerade) gives
    // operators reading the test as documentation a clearer signal.
    let _ = counters
        .next_object_id()
        .await
        .expect("next_object_id must succeed to push counter past seed")
        .expect("the first routine call seeded the counter, so allocation must not fail closed");

    // Second call — counter is now strictly above the seed, so this
    // hits the genuine `NoOp` arm. The histogram MUST still observe a
    // sample.
    let outcome_b = seed_object_id_counter(
        Arc::clone(&store),
        Arc::clone(&counters),
        1000,
        metrics.clone(),
    )
    .await
    .expect("second routine call must succeed");
    assert!(
        matches!(outcome_b, SeedOutcome::NoOp { .. }),
        "Scenario 6 pre-condition: second call must produce NoOp (counter \
         pushed strictly past seed by the intervening next_object_id call); \
         got {outcome_b:?}"
    );

    let count_after_second = live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(&metrics),
        "live_edit_checkpoint_counter_seed_secs_count",
        &[],
    )
    .unwrap_or(0.0);
    assert!(
        count_after_second >= 2.0,
        "after two routine calls sample_count must be >= 2 (one per call, \
         including the NoOp arm); got {count_after_second}. A stuck-at-1 \
         value would mean the histogram is only observed on the Seeded arm \
         — the ObserveOnDrop guard must remain armed across NoOp."
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Scenario 7 — `seed_object_id_counter_with_retry` startup token-race
// recovery. Regression coverage for the bug where a one-shot counters-sidecar
// fetch failure (the credential-provisioner sidecar had not written the
// service token yet at pod startup) wedged readiness NOT_SERVING permanently
// because the liveness probe targets the bare gRPC server (already SERVING)
// and never restarts the pod to retry the one-shot seed.
//
// Gated on `test-checkpoint-fault-injection` because the
// `InMemoryCheckpointStore::with_fault_injector` builder + the
// `CheckpointFaultInjector` trait are compiled out of production library
// builds (same pattern as e2e_ext_value_recovery_test.rs).
// ─────────────────────────────────────────────────────────────────────────

#[cfg(feature = "test-checkpoint-fault-injection")]
use live_edit::checkpoint::adapters::{
    CheckpointFaultInjector, InMemoryCheckpointStore as InMemFI,
};
#[cfg(feature = "test-checkpoint-fault-injection")]
use live_edit::checkpoint::{seed_object_id_counter_with_retry, CheckpointError};
#[cfg(feature = "test-checkpoint-fault-injection")]
use std::sync::atomic::{AtomicU32, Ordering};

/// Counting fault injector that fails the first `fail_first`
/// `get_counters_sidecar` calls with a `Transient` error, then lets every
/// subsequent call through. `calls` records the total number of
/// `get_counters_sidecar` invocations so a test can assert how many attempts
/// the retry wrapper made. Non-`get_counters_sidecar` ops always pass (so a
/// `put_counters_sidecar` used to stage a fixture is not short-circuited).
#[cfg(feature = "test-checkpoint-fault-injection")]
struct FlakyCountersFetch {
    fail_first: u32,
    calls: AtomicU32,
}

#[cfg(feature = "test-checkpoint-fault-injection")]
impl CheckpointFaultInjector for FlakyCountersFetch {
    fn maybe_inject(&self, op: &str) -> Result<(), CheckpointError> {
        if op == "get_counters_sidecar" {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            if n < self.fail_first {
                return Err(CheckpointError::Transient {
                    source: "injected: service token not written yet".into(),
                });
            }
        }
        Ok(())
    }
}

/// 7a — the fetch fails the first 3 times (sidecar token not yet written),
/// then succeeds. `seed_object_id_counter_with_retry` must retry past the
/// failures and return `Seeded`, having made exactly 4 fetch attempts.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[cfg(feature = "test-checkpoint-fault-injection")]
async fn with_retry_recovers_after_transient_fetch_failures() {
    let tv = TestValkey::setup().await;
    let metrics = make_metrics();
    let counters = Arc::new(CounterStore::new(
        tv.client().clone(),
        tv.key_schema().clone(),
    ));
    wipe_object_id_counter(&tv).await;

    let injector = Arc::new(FlakyCountersFetch {
        fail_first: 3,
        calls: AtomicU32::new(0),
    });
    let store: Arc<dyn CheckpointStore> =
        Arc::new(InMemFI::new().with_fault_injector(injector.clone()));
    // Stage a sidecar so the eventual successful fetch yields a real seed.
    put_sidecar(store.as_ref(), 5000).await;

    let outcome = seed_object_id_counter_with_retry(
        Arc::clone(&store),
        Arc::clone(&counters),
        1000,
        metrics,
        10,             // max_attempts — well above the 3 injected failures
        Duration::ZERO, // no real sleep in tests
    )
    .await
    .expect("with_retry must succeed once the injected failures are exhausted");

    assert_eq!(
        outcome,
        SeedOutcome::Seeded {
            seed: 6000,
            new_counter: 6000
        },
        "after retrying past the transient fetch failures the seed must advance the counter to max+margin"
    );
    assert_eq!(
        injector.calls.load(Ordering::SeqCst),
        4,
        "expected 3 failed fetch attempts + 1 success = 4 total get_counters_sidecar calls"
    );
    assert_eq!(
        read_counter(&counters).await,
        6000,
        "le:counter:object_id must be seeded to 6000 after retry recovery"
    );
}

/// 7b — the fetch fails on every attempt. The wrapper must give up after
/// exactly `max_attempts` and surface `SeedError::Store` (readiness stays
/// NOT_SERVING — a genuinely unreachable adapter / never-written token is a
/// real failure, not something to retry forever).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[cfg(feature = "test-checkpoint-fault-injection")]
async fn with_retry_gives_up_after_max_attempts() {
    let tv = TestValkey::setup().await;
    let metrics = make_metrics();
    let counters = Arc::new(CounterStore::new(
        tv.client().clone(),
        tv.key_schema().clone(),
    ));

    let injector = Arc::new(FlakyCountersFetch {
        fail_first: u32::MAX, // always fail
        calls: AtomicU32::new(0),
    });
    let store: Arc<dyn CheckpointStore> =
        Arc::new(InMemFI::new().with_fault_injector(injector.clone()));

    let err = seed_object_id_counter_with_retry(
        Arc::clone(&store),
        Arc::clone(&counters),
        1000,
        metrics,
        3, // max_attempts
        Duration::ZERO,
    )
    .await
    .expect_err("with_retry must surface the Store error after exhausting the budget");

    assert!(
        matches!(err, SeedError::Store(_)),
        "exhausted retry must surface SeedError::Store; got {err:?}"
    );
    assert_eq!(
        injector.calls.load(Ordering::SeqCst),
        3,
        "with max_attempts=3 the wrapper must make exactly 3 fetch attempts before giving up"
    );
}

/// 7c — a parse failure (corrupt sidecar bytes) is NOT retried: it is a
/// genuinely permanent condition, so the wrapper returns `ParseFailed` after
/// exactly one fetch even with a large retry budget.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[cfg(feature = "test-checkpoint-fault-injection")]
async fn with_retry_does_not_retry_parse_failure() {
    let tv = TestValkey::setup().await;
    let metrics = make_metrics();
    let counters = Arc::new(CounterStore::new(
        tv.client().clone(),
        tv.key_schema().clone(),
    ));

    // Injector never fails the fetch (fail_first = 0); it only counts calls so
    // we can assert no retry occurred.
    let injector = Arc::new(FlakyCountersFetch {
        fail_first: 0,
        calls: AtomicU32::new(0),
    });
    let store: Arc<dyn CheckpointStore> =
        Arc::new(InMemFI::new().with_fault_injector(injector.clone()));
    // Stage corrupt (non-JSON) sidecar bytes so the fetch succeeds but the
    // parse fails.
    store
        .put_counters_sidecar(bytes::Bytes::from_static(b"{ not valid json"))
        .await
        .expect("staging corrupt sidecar bytes must succeed");

    let err = seed_object_id_counter_with_retry(
        Arc::clone(&store),
        Arc::clone(&counters),
        1000,
        metrics,
        5, // generous budget — must NOT be consumed on a parse failure
        Duration::ZERO,
    )
    .await
    .expect_err("corrupt sidecar must surface ParseFailed");

    assert!(
        matches!(err, SeedError::ParseFailed(_)),
        "corrupt sidecar bytes must surface SeedError::ParseFailed; got {err:?}"
    );
    assert_eq!(
        injector.calls.load(Ordering::SeqCst),
        1,
        "ParseFailed must not be retried — exactly one fetch attempt expected"
    );
}
