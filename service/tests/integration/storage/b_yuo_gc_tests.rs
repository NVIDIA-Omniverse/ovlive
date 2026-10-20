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

//! b.yuo — the external-value release ledger, the age-gated GC loop, and the
//! at-most-once decrement, against a real Valkey.
//!
//! # What lives here
//!
//! | Area | Tests |
//! |---|---|
//! | Age gate (test 1) | [`a_member_younger_than_the_age_floor_waits_and_is_collected_after_it`] |
//! | Byte trigger (test 2; policy 11, only the excess over T) | [`at_the_pressure_threshold_only_the_excess_is_collected_oldest_first`], [`a_member_of_zero_bytes_counts_nothing_toward_the_excess`], [`a_member_past_the_age_floor_is_collected_whatever_the_excess`], [`under_pressure_a_live_read_hold_clips_the_excess_too`], [`below_the_pressure_threshold_the_age_floor_applies`], [`production_collects_from_5_s_and_the_excess_over_the_threshold_at_once`], [`without_a_budget_no_pass_is_under_pressure`] |
//! | Hold clipping (test 3, `gc_select.lua`'s read of the hold HASH; un-aged members stamped first since b.1jt) | [`a_live_read_hold_protects_members_released_after_its_start`], [`an_unaged_member_is_never_collected_under_a_read_hold`], [`an_expired_read_hold_field_is_ignored_by_the_selection`], [`the_earliest_live_hold_start_clips_the_selection`], [`an_unaged_member_waits_while_any_read_hold_field_lives`] |
//! | Read hold lifecycle (H-a … H-e: `ext_read_hold.lua`, `ReadHold`) | [`a_released_hold_moves_the_clip_to_the_next_holders_own_start`], [`a_hold_dropped_unreleased_deletes_its_field`], [`a_renewal_after_the_field_expired_answers_lapsed_and_writes_nothing`], [`a_resent_take_keeps_the_fields_start`], [`a_dead_holders_field_stops_clipping_at_expiry_and_is_pruned`] |
//! | Live hold count (`holds_live`, the `live_edit_ext_read_holds` gauge) | [`the_leader_publishes_the_live_read_hold_count`] |
//! | Ledger accounting (test 4's non-property arms) | [`a_release_accounts_its_blob_size_and_the_total_returns_to_zero`], [`a_failed_size_read_still_enqueues_the_release_at_zero_bytes`], [`a_commit_surplus_release_accounts_zero_bytes`], [`a_byte_count_with_a_leading_zero_is_refused_encoding_with_nothing_written`] |
//! | H1 fred re-send, script level | [`an_identical_decrement_rerun_is_replayed_and_decrements_once`], [`a_replay_after_delete_and_restore_keeps_the_restored_blob`] |
//! | H2 fred re-send through `process_pending` | [`a_resent_decrement_through_the_pass_applies_once`] |
//! | H3 re-claim of an already-decremented claim (in place since b.1jt) | [`a_reclaimed_claim_whose_decrement_applied_is_replayed`], [`a_reclaimed_claim_whose_decrement_deleted_the_blob_keeps_the_restored_one`] |
//! | H4 per-pass re-claim (in place since b.1jt) | [`the_loop_reclaims_and_collects_a_stuck_claim_without_a_restart`] |
//! | H5 fence | [`a_decrement_older_than_the_fence_is_stale_and_writes_nothing`], [`a_late_copy_after_settle_and_teardown_is_stale`], [`a_pass_whose_decrement_outlives_the_fence_leaves_the_member_inflight`] |
//! | H6 teardown | [`a_settled_record_is_torn_down_after_the_retention`], [`unmark_keeps_a_record_newer_than_the_settle`], [`settled_retire_removes_an_entry_only_while_its_score_matches`], [`a_settled_member_back_in_pending_is_not_torn_down`], [`an_identical_member_after_teardown_decrements_normally`] |
//! | H7 underflow | `b_j9j_tests::gc_underflow_total_increments_on_synthetic_double_decrement` |
//! | H8 older-build leftovers | [`the_claim_self_heals_an_orphan_with_an_age_entry`], [`the_audit_settles_an_orphan_without_an_age_entry`] |
//! | H9 per-member wrong type | [`a_wrong_typed_replay_record_fails_only_its_member`] |
//! | `more` | [`a_full_batch_of_failing_unmarks_does_not_run_the_next_pass_at_once`], [`a_full_batch_of_failing_decrements_does_not_run_the_next_pass_at_once`] |
//! | Gauges (test 9) | [`only_the_leader_publishes_the_ledger_gauges`] |
//!
//! Elsewhere: the H10 key-shape tests beside `KeySchema`
//! (`src/storage/keys.rs`), H11 and the `PRODUCTION` pin in
//! `src/storage/external_values.rs`, the gate in
//! `src/storage/ext_release_gate.rs` and
//! `tests/integration/grpc/ext_release_gate_test.rs`, the commit release
//! record in `b_yuo_wrel_tests.rs`, an older build sharing the store (b.1jt:
//! the marked lock, foreign claims, the enqueue's replay refusal) in
//! `b_1jt_rollout_tests.rs`, the ledger/replay state machine in
//! `tests/proptests.rs`, and the catch-up side of the read hold (the miss
//! protocol that takes, renews and releases it) in
//! `tests/integration/grpc/b_yuo_catch_up_miss_test.rs`, and the
//! out-of-memory collection, which collects at any age (b.pa1), in
//! `oom_collection_test.rs`.
//!
//! # Driving the scripts directly
//!
//! The script-level tests call the GC scripts with the KEYS and ARGV
//! `process_pending` passes, so a claim's time can be read back and a
//! decrement re-run with identical arguments — fred's re-send of a command
//! whose reply was lost. A small `fence_ms` stands in for `F` where a test
//! needs a copy to outlive its fence in real time; through `process_pending`,
//! `with_decrement_fence_for_test` does the same. The read hold tests plant
//! fields directly where a field's start or expiry must be placed exactly (an
//! already-expired field, two starts in a chosen order), and take them
//! through `ReadHold` or `ext_read_hold.lua` where the lifecycle is the
//! subject.

use std::sync::Arc;
use std::time::Duration;

use fred::interfaces::{
    ClientLike, HashesInterface, KeysInterface, SetsInterface, SortedSetsInterface,
};
use fred::types::{ClusterHash, CustomCommand, Value};
use tokio_util::sync::CancellationToken;

use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::Metrics;
use live_edit::storage::ext_release_gate::ExtReleaseGate;
use live_edit::storage::external_values::{
    process_pending_gc, spawn_gc_task_with_leader, ExtRelease, ExternalValueStore, GcPassState,
    GcPolicy, ReadHold, ReadHoldRenewal, ReadHoldTiming, ValkeyExternalValueStore,
    EXT_READ_HOLD_SCRIPT, EXT_RELEASE_ENQUEUE_SCRIPT, GC_CLAIM_PENDING_SCRIPT, GC_COMPLETE_SCRIPT,
    GC_DECREMENT_AND_COLLECT_SCRIPT, GC_DECREMENT_FENCE_MS, GC_SETTLED_RETIRE_SCRIPT,
    GC_UNMARK_SCRIPT,
};
use live_edit::storage::keys::{sha256_to_hex, KeySchema};
use live_edit::storage::test_support::TestValkey;

// ═════════════════════════════════════════════════════════════════════════════
// Fixture
// ═════════════════════════════════════════════════════════════════════════════

/// The object every release here names (a retirement), unless a test needs
/// two distinct members on one hash.
const OBJ: u64 = 7_700;

/// A second releasing object.
const OBJ_B: u64 = 7_701;

fn hash(byte: u8) -> [u8; 32] {
    [byte; 32]
}

fn member(h: [u8; 32], object_id: u64) -> String {
    ExtRelease::retirement(h, object_id).member()
}

/// `IMMEDIATE`'s shape with the given floors, replay retention and re-claim
/// age.
fn policy(
    min_age: Duration,
    pressure: Duration,
    retention: Duration,
    reclaim: Duration,
) -> GcPolicy {
    GcPolicy {
        min_age,
        pressure_min_age: pressure,
        replay_record_retention: retention,
        reclaim_after: reclaim,
        ..GcPolicy::IMMEDIATE
    }
}

/// [`GcPolicy::IMMEDIATE`], except that replay records outlive the run, so a
/// replay of a collected member meets its record.
fn immediate_keeping_records() -> GcPolicy {
    GcPolicy {
        replay_record_retention: Duration::from_secs(3600),
        ..GcPolicy::IMMEDIATE
    }
}

fn metrics() -> Arc<Metrics> {
    Arc::new(crate::common::metrics_scrape_only("b-yuo-gc"))
}

fn store(tv: &TestValkey, metrics: &Arc<Metrics>) -> ValkeyExternalValueStore {
    ValkeyExternalValueStore::new(tv.client().clone(), tv.key_schema().clone())
        .with_metrics(Arc::clone(metrics))
}

/// [`store`] with a release-budget gate at `budget`, so the pressure
/// threshold is `budget / 2`.
fn gated_store(tv: &TestValkey, metrics: &Arc<Metrics>, budget: u64) -> ValkeyExternalValueStore {
    store(tv, metrics).with_ext_release_gate(Arc::new(ExtReleaseGate::new(
        tv.client().clone(),
        tv.key_schema(),
        budget,
        Duration::from_secs(30),
        None,
    )))
}

/// Store `len` bytes under `h` (one reference each call).
async fn put(store: &ValkeyExternalValueStore, h: [u8; 32], len: usize) {
    store
        .store_values(&[(&h, vec![0x5a; len].as_slice())])
        .await
        .expect("store_values must succeed");
}

async fn release(store: &ValkeyExternalValueStore, h: [u8; 32], object_id: u64) {
    store
        .schedule_release(&[ExtRelease::retirement(h, object_id)])
        .await
        .expect("schedule_release must succeed");
}

async fn pass(
    store: &ValkeyExternalValueStore,
    policy: &GcPolicy,
) -> live_edit::storage::external_values::GcBatchResult {
    store
        .process_pending(policy, &mut GcPassState::new())
        .await
        .expect("a GC pass must succeed")
}

/// Valkey's own clock, ms.
async fn valkey_now_ms(tv: &TestValkey) -> u64 {
    let reply: Vec<String> = tv
        .client()
        .pool
        .custom(
            CustomCommand::new_static("TIME", ClusterHash::FirstKey, false),
            Vec::<Value>::new(),
        )
        .await
        .expect("TIME must succeed");
    let secs: u64 = reply[0].parse().expect("TIME seconds");
    let usecs: u64 = reply[1].parse().expect("TIME microseconds");
    secs * 1000 + usecs / 1000
}

// ── keyspace readers ─────────────────────────────────────────────────────────

async fn ref_count(tv: &TestValkey, h: [u8; 32]) -> Option<i64> {
    tv.client()
        .pool
        .hget(tv.key_schema().ext_meta(&sha256_to_hex(&h)), "ref_count")
        .await
        .expect("HGET ref_count")
}

async fn data_present(tv: &TestValkey, h: [u8; 32]) -> bool {
    tv.client()
        .pool
        .exists(tv.key_schema().ext_data(&sha256_to_hex(&h)))
        .await
        .expect("EXISTS ext:data")
}

async fn meta_present(tv: &TestValkey, h: [u8; 32]) -> bool {
    tv.client()
        .pool
        .exists(tv.key_schema().ext_meta(&sha256_to_hex(&h)))
        .await
        .expect("EXISTS ext:meta")
}

async fn total(tv: &TestValkey) -> i64 {
    let raw: Option<i64> = tv
        .client()
        .pool
        .get(tv.key_schema().ext_pending_total())
        .await
        .expect("GET pending:total");
    raw.unwrap_or(0)
}

async fn bytes_of(tv: &TestValkey, m: &str) -> Option<i64> {
    tv.client()
        .pool
        .hget(tv.key_schema().ext_pending_bytes(), m)
        .await
        .expect("HGET pending:bytes")
}

async fn zscore(tv: &TestValkey, key: String, m: &str) -> Option<f64> {
    tv.client().pool.zscore(key, m).await.expect("ZSCORE")
}

async fn age_of(tv: &TestValkey, m: &str) -> Option<f64> {
    zscore(tv, tv.key_schema().ext_pending_age(), m).await
}

async fn settled_of(tv: &TestValkey, m: &str) -> Option<f64> {
    zscore(tv, tv.key_schema().ext_settled(), m).await
}

async fn record_of(tv: &TestValkey, h: [u8; 32], m: &str) -> Option<f64> {
    zscore(tv, tv.key_schema().ext_applied(&sha256_to_hex(&h)), m).await
}

async fn in_pending(tv: &TestValkey, m: &str) -> bool {
    tv.client()
        .pool
        .sismember(tv.key_schema().ext_pending(), m)
        .await
        .expect("SISMEMBER pending")
}

async fn in_inflight(tv: &TestValkey, m: &str) -> bool {
    zscore(tv, tv.key_schema().ext_inflight(), m)
        .await
        .is_some()
}

async fn zcard(tv: &TestValkey, key: String) -> u64 {
    tv.client().pool.zcard(key).await.expect("ZCARD")
}

// ── the scripts, called as `process_pending` calls them ──────────────────────

fn claim_keys(keys: &KeySchema) -> Vec<String> {
    vec![
        keys.ext_pending(),
        keys.ext_inflight(),
        keys.ext_pending_age(),
        keys.ext_pending_bytes(),
        keys.ext_pending_total(),
        keys.ext_settled(),
    ]
}

/// `gc_claim_pending.lua`: the claim time in ms, or 0.
async fn claim(tv: &TestValkey, m: &str) -> i64 {
    GC_CLAIM_PENDING_SCRIPT
        .evalsha_with_reload::<i64, _, _>(
            tv.client().maintenance_pool.next(),
            claim_keys(tv.key_schema()),
            vec![m.to_owned()],
        )
        .await
        .expect("gc_claim_pending must run")
}

/// `gc_decrement_and_collect.lua` for member `m` of hash `h`: its reply
/// `(action, n, freed_bytes)`, `freed_bytes` the deleted blob's length on a
/// deleting branch and 0 on every other (b.pa1).
async fn decrement(
    tv: &TestValkey,
    h: [u8; 32],
    m: &str,
    claim_ms: i64,
    fence_ms: u64,
) -> (String, i64, i64) {
    let hex = sha256_to_hex(&h);
    let keys = tv.key_schema();
    GC_DECREMENT_AND_COLLECT_SCRIPT
        .evalsha_with_reload::<(String, i64, i64), _, _>(
            tv.client().maintenance_pool.next(),
            vec![
                keys.ext_meta(&hex),
                keys.ext_data(&hex),
                keys.ext_applied(&hex),
            ],
            vec![m.to_owned(), claim_ms.to_string(), fence_ms.to_string()],
        )
        .await
        .expect("gc_decrement_and_collect must run")
}

/// `gc_complete.lua`: 1 when it released the claim and settled the member.
async fn complete(tv: &TestValkey, m: &str) -> i64 {
    let keys = tv.key_schema();
    GC_COMPLETE_SCRIPT
        .evalsha_with_reload::<i64, _, _>(
            tv.client().maintenance_pool.next(),
            vec![
                keys.ext_inflight(),
                keys.ext_pending(),
                keys.ext_pending_bytes(),
                keys.ext_pending_total(),
                keys.ext_settled(),
            ],
            vec![m.to_owned()],
        )
        .await
        .expect("gc_complete must run")
}

/// `live_edit_gc_decrement_total{outcome}`.
fn decrements(metrics: &Metrics, outcome: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_gc_decrement_total",
        &[("outcome", outcome)],
    )
    .unwrap_or_else(|| panic!("gc_decrement_total{{outcome={outcome}}} is zero-seeded")) as u64
}

/// `live_edit_gc_underflow_total`. The counter is zero-seeded when the
/// metrics are built, so a missing sample is a failure, not a zero.
#[cfg(feature = "test-fault-injection")]
fn underflows(metrics: &Metrics) -> u64 {
    sample_value(&render(metrics), "live_edit_gc_underflow_total", &[])
        .expect("live_edit_gc_underflow_total is zero-seeded") as u64
}

fn batch_errors(metrics: &Metrics, kind: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_gc_sweep_batch_errors_total",
        &[("kind", kind)],
    )
    .unwrap_or(0.0) as u64
}

// ═════════════════════════════════════════════════════════════════════════════
// Test 1 — the age gate
// ═════════════════════════════════════════════════════════════════════════════

/// A member younger than the age floor is not collected, and the pass reports
/// when it falls due; once it is older, it is collected.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_member_younger_than_the_age_floor_waits_and_is_collected_after_it() {
    let tv = TestValkey::setup().await;
    let m = metrics();
    let store = store(&tv, &m);
    let floor = Duration::from_millis(1_500);
    let policy = policy(floor, floor, Duration::ZERO, Duration::from_secs(30));
    let h = hash(0x11);
    put(&store, h, 64).await;
    release(&store, h, OBJ).await;

    let young = pass(&store, &policy).await;
    assert_eq!(
        young.processed, 0,
        "a release younger than the age floor must not be claimed"
    );
    assert_eq!(ref_count(&tv, h).await, Some(1));
    let due = young
        .ledger
        .and_then(|l| l.next_due_in)
        .expect("the pass reports when the oldest member falls due");
    assert!(due <= floor && due > Duration::ZERO, "due in {due:?}");

    tokio::time::sleep(floor + Duration::from_millis(300)).await;
    let old = pass(&store, &policy).await;
    assert_eq!(
        (old.processed, old.deleted),
        (1, 1),
        "past the floor it is collected"
    );
    assert!(!data_present(&tv, h).await);
}

// ═════════════════════════════════════════════════════════════════════════════
// Test 2 — the byte trigger
// ═════════════════════════════════════════════════════════════════════════════

/// The pressure threshold is half the budget.
const BUDGET: u64 = 4_000;
/// Each release here accounts this many bytes, so two reach the threshold.
const BLOB: usize = 1_000;

/// The policy every test-2 case runs: an hour's age floor `G` and no
/// pressure floor, so a member younger than `G` is collected only as part of
/// the excess over `T`.
fn pressure_policy() -> GcPolicy {
    policy(
        Duration::from_secs(3600),
        Duration::ZERO,
        Duration::ZERO,
        Duration::from_secs(30),
    )
}

/// Release `h` with `len` bytes, then wait 20 ms so the next release gets a
/// later `pending:age` score: the order of a sequence of these is the age
/// order.
async fn release_in_turn(store: &ValkeyExternalValueStore, h: [u8; 32], len: usize) {
    put(store, h, len).await;
    release(store, h, OBJ).await;
    tokio::time::sleep(Duration::from_millis(20)).await;
}

/// At or above `T = H / 2` a pass collects only the excess over `T` (policy
/// 11), oldest first, although no member is as old as `G`: with three
/// 1 000-byte members and `T` = 2 000, the two oldest go and the total falls
/// to 1 000, below `T`; the newest stays, and the next pass, below `T`,
/// collects nothing.
///
/// The member strings sort against age — the oldest's (`2323…`) sorts last —
/// and `gc_select.lua`'s `ZRANGEBYSCORE` breaks equal scores by member, so a
/// selection that ignored age (or ages collapsed to one score) would take the
/// two newest instead. A selection of the whole due batch would take all
/// three.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn at_the_pressure_threshold_only_the_excess_is_collected_oldest_first() {
    let tv = TestValkey::setup().await;
    let m = metrics();
    let store = gated_store(&tv, &m, BUDGET);
    let (oldest, middle, newest) = (hash(0x23), hash(0x22), hash(0x21));
    assert!(
        member(oldest, OBJ) > member(middle, OBJ) && member(middle, OBJ) > member(newest, OBJ),
        "precondition: the member strings sort against age"
    );
    for h in [oldest, middle, newest] {
        release_in_turn(&store, h, BLOB).await;
    }
    assert_eq!(
        total(&tv).await,
        3 * BLOB as i64,
        "precondition: 1 000 bytes above T"
    );

    let p = pass(&store, &pressure_policy()).await;
    assert_eq!(
        (p.processed, p.deleted),
        (2, 2),
        "under pressure only the excess over T is collected, though nothing is as old as G"
    );
    assert!(
        !data_present(&tv, oldest).await && !data_present(&tv, middle).await,
        "the two OLDEST members are the ones collected"
    );
    assert!(
        in_pending(&tv, &member(newest, OBJ)).await,
        "the newest stays pending"
    );
    assert_eq!(ref_count(&tv, newest).await, Some(1));
    assert_eq!(total(&tv).await, BLOB as i64, "the total is below T");

    let below = pass(&store, &pressure_policy()).await;
    assert_eq!(below.processed, 0, "below T the hour's floor applies again");
}

/// A member whose `pending:bytes` are 0 — or absent — counts nothing toward
/// the excess: it is taken in its turn, and the members after it are taken
/// until their own bytes cover the excess. Here two such members are the
/// oldest; the excess over `T` is 1 000 bytes; so both go, then two 1 000-byte
/// members (the second of which takes the sum past the excess), and the
/// newest stays. Counted at their blobs' size, the two would have covered the
/// excess themselves and only they would go.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_member_of_zero_bytes_counts_nothing_toward_the_excess() {
    let tv = TestValkey::setup().await;
    let m = metrics();
    let store = gated_store(&tv, &m, BUDGET);
    let (zero, absent) = (hash(0x2A), hash(0x2B));
    let (first, second, newest) = (hash(0x2C), hash(0x2D), hash(0x2E));
    for h in [zero, absent, first, second, newest] {
        release_in_turn(&store, h, BLOB).await;
    }
    // Account the two oldest at 0 bytes: one field written "0", one removed.
    let bytes_key = tv.key_schema().ext_pending_bytes();
    let _: () = tv
        .client()
        .pool
        .hset(&bytes_key, (member(zero, OBJ), "0"))
        .await
        .expect("HSET pending:bytes");
    let removed: i64 = tv
        .client()
        .pool
        .hdel(&bytes_key, member(absent, OBJ))
        .await
        .expect("HDEL pending:bytes");
    assert_eq!(removed, 1);
    let _: i64 = tv
        .client()
        .pool
        .decr_by(tv.key_schema().ext_pending_total(), 2 * BLOB as i64)
        .await
        .expect("DECRBY pending:total");
    assert_eq!(
        total(&tv).await,
        3 * BLOB as i64,
        "precondition: 1 000 bytes above T, all in the three newest"
    );

    let p = pass(&store, &pressure_policy()).await;
    assert_eq!(
        (p.processed, p.deleted),
        (4, 4),
        "the two zero-byte members plus the two that cover the excess"
    );
    assert!(
        in_pending(&tv, &member(newest, OBJ)).await,
        "the newest stays pending"
    );
    assert_eq!(total(&tv).await, BLOB as i64);
}

/// A member past `G` is collected whatever the excess: its bytes count
/// toward the excess, but the excess never holds it back. Here a 2 500-byte
/// member past `G` alone covers the 2 000-byte excess; the 500-byte member
/// past `G` after it is still collected; the young member is not. A rule
/// that capped every member at the excess would leave the second one.
///
/// "Past `G`" is staged by backdating the two members' `pending:age` scores
/// an hour, so the test does not wait out a floor.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_member_past_the_age_floor_is_collected_whatever_the_excess() {
    let tv = TestValkey::setup().await;
    let m = metrics();
    let store = gated_store(&tv, &m, BUDGET);
    let (big, small, young) = (hash(0x2F), hash(0x30), hash(0x3B));
    release_in_turn(&store, big, 2_500).await;
    release_in_turn(&store, small, 500).await;
    release_in_turn(&store, young, BLOB).await;
    let hour_ago = valkey_now_ms(&tv).await - 3_600_000 - 60_000;
    for (h, backdate) in [(big, hour_ago), (small, hour_ago + 1)] {
        let _: i64 = tv
            .client()
            .pool
            .zadd(
                tv.key_schema().ext_pending_age(),
                Some(fred::types::SetOptions::XX),
                None,
                true,
                false,
                (backdate as f64, member(h, OBJ)),
            )
            .await
            .expect("ZADD XX pending:age");
    }
    assert_eq!(total(&tv).await, 4_000, "precondition: 2 000 bytes above T");

    let p = pass(&store, &pressure_policy()).await;
    assert_eq!(
        (p.processed, p.deleted),
        (2, 2),
        "both members past G are collected, the young one is not"
    );
    assert!(!data_present(&tv, big).await && !data_present(&tv, small).await);
    assert_eq!(ref_count(&tv, young).await, Some(1));
    assert_eq!(total(&tv).await, BLOB as i64);
}

/// Under pressure a live read hold clips the excess selection too: with
/// `T` = 2 000 and 3 000 bytes released — `old` before the hold's start, `a`
/// and `b` after it — only `old` is collected, although the excess (1 000)
/// would otherwise take `a` as well. The held releases stay, and the total
/// stays at or above `T` for as long as the hold lives. Without the clip on
/// the excess cutoff (`gc_select.lua`'s `excess_cutoff > clip` line) the
/// excess branch, which runs only with a pressure floor below `G`, would
/// collect `a`, released after the hold began.
///
/// No hold margin, so each member's side of the start follows from its age;
/// the start sits 150 ms after `old`'s release and at least 150 ms before
/// `a`'s.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn under_pressure_a_live_read_hold_clips_the_excess_too() {
    let tv = TestValkey::setup().await;
    let m = metrics();
    let store = gated_store(&tv, &m, BUDGET).with_hold_margin_for_test(Duration::ZERO);
    let (old, a, b) = (hash(0x45), hash(0x46), hash(0x47));
    release_in_turn(&store, old, BLOB).await;
    let old_age = age_of(&tv, &member(old, OBJ)).await.expect("aged") as u64;
    plant_hold(&tv, old_age + 150, Duration::from_secs(600)).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    release_in_turn(&store, a, BLOB).await;
    release_in_turn(&store, b, BLOB).await;
    assert!(
        age_of(&tv, &member(a, OBJ)).await.expect("aged") as u64 > old_age + 150,
        "precondition: `a` was released after the hold's start"
    );
    assert_eq!(
        total(&tv).await,
        3 * BLOB as i64,
        "precondition: 1 000 bytes above T"
    );

    let p = pass(&store, &pressure_policy()).await;
    assert_eq!(
        (p.processed, p.deleted),
        (1, 1),
        "only the release from before the hold's start is collected"
    );
    assert!(!data_present(&tv, old).await);
    for (h, what) in [(a, "a"), (b, "b")] {
        assert_eq!(
            ref_count(&tv, h).await,
            Some(1),
            "`{what}`, released after the hold's start, is protected under pressure"
        );
    }
    assert_eq!(total(&tv).await, 2 * BLOB as i64);
}

/// Below the threshold the normal floor applies: nothing younger than it is
/// collected, however low the pressure floor.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn below_the_pressure_threshold_the_age_floor_applies() {
    let tv = TestValkey::setup().await;
    let m = metrics();
    let store = gated_store(&tv, &m, BUDGET);
    let policy = policy(
        Duration::from_secs(3600),
        Duration::ZERO,
        Duration::ZERO,
        Duration::from_secs(30),
    );
    let h = hash(0x23);
    put(&store, h, BLOB).await;
    release(&store, h, OBJ).await;
    assert!(total(&tv).await < (BUDGET / 2) as i64);

    let p = pass(&store, &policy).await;
    assert_eq!(
        p.processed, 0,
        "below T a member younger than G is not collected"
    );
    assert_eq!(ref_count(&tv, h).await, Some(1));
}

/// `GcPolicy::PRODUCTION`, driven through a real pass: below `T` nothing
/// younger than its 5 s floor is collected and the pass reports the member
/// due within 5 s; above `T` the excess over `T` is collected at once, oldest
/// first, with no age floor (policy 11 with a pressure floor of 0).
///
/// With `T` = 2 000: one 1 000-byte release is below `T`, so the pass
/// collects nothing and reports it due within 5 s. Two more releases take the
/// total to 3 000, 1 000 above `T`, so the next pass collects the two oldest
/// (the second takes the selected bytes past the excess, and is the last) and
/// leaves the newest; the total falls to 1 000. The member strings sort
/// against age, as in [`at_the_pressure_threshold_only_the_excess_is_collected_oldest_first`],
/// so a selection that ignored age would take a different pair.
///
/// It fails on the old 60 s / 60 s floors (the member is reported due in
/// about 60 s, and above `T` nothing is collected), and on 5 s / 5 s (above
/// `T` nothing younger than 5 s is collected). The pass above `T` is checked
/// to have run while every member was younger than the 5 s floor, on the
/// store's own clock: the oldest member's `pending:age` score against `TIME`
/// once the pass is done.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn production_collects_from_5_s_and_the_excess_over_the_threshold_at_once() {
    let tv = TestValkey::setup().await;
    let m = metrics();
    let store = gated_store(&tv, &m, BUDGET);
    let (oldest, middle, newest) = (hash(0x28), hash(0x27), hash(0x26));
    assert!(
        member(oldest, OBJ) > member(middle, OBJ) && member(middle, OBJ) > member(newest, OBJ),
        "precondition: the member strings sort against age"
    );

    // Below T: the 5 s floor.
    release_in_turn(&store, oldest, BLOB).await;
    assert!(
        total(&tv).await < (BUDGET / 2) as i64,
        "precondition: below T"
    );
    let below = pass(&store, &GcPolicy::PRODUCTION).await;
    assert_eq!(
        below.processed, 0,
        "below T nothing younger than the 5 s floor is collected"
    );
    assert_eq!(ref_count(&tv, oldest).await, Some(1));
    let due = below
        .ledger
        .and_then(|l| l.next_due_in)
        .expect("the pass reports when the member falls due");
    assert!(
        due > Duration::ZERO && due <= Duration::from_secs(5),
        "the member falls due at the 5 s floor; got {due:?}"
    );

    // Above T: the excess, at once, oldest first.
    release_in_turn(&store, middle, BLOB).await;
    release_in_turn(&store, newest, BLOB).await;
    assert_eq!(
        total(&tv).await,
        3 * BLOB as i64,
        "precondition: 1 000 bytes above T"
    );
    let oldest_age = age_of(&tv, &member(oldest, OBJ))
        .await
        .expect("the oldest member is aged") as u64;
    let above = pass(&store, &GcPolicy::PRODUCTION).await;
    let after = valkey_now_ms(&tv).await;
    assert!(
        after.saturating_sub(oldest_age) < 5_000,
        "precondition: on the store's clock every member was younger than the 5 s floor when \
         the pass ran (the oldest was aged at {oldest_age}, the pass ended by {after})"
    );
    assert_eq!(
        (above.processed, above.deleted),
        (2, 2),
        "above T the excess over T is collected with no age floor"
    );
    assert!(
        !data_present(&tv, oldest).await && !data_present(&tv, middle).await,
        "the two OLDEST members are the ones collected"
    );
    assert!(
        in_pending(&tv, &member(newest, OBJ)).await,
        "the newest stays pending"
    );
    assert_eq!(ref_count(&tv, newest).await, Some(1));
    assert_eq!(total(&tv).await, BLOB as i64, "the total is below T");
}

/// A store without a gate has no pressure threshold: however large the
/// total, the normal floor applies.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn without_a_budget_no_pass_is_under_pressure() {
    let tv = TestValkey::setup().await;
    let m = metrics();
    let store = store(&tv, &m);
    let policy = policy(
        Duration::from_secs(3600),
        Duration::ZERO,
        Duration::ZERO,
        Duration::from_secs(30),
    );
    let h = hash(0x25);
    put(&store, h, 50_000).await;
    release(&store, h, OBJ).await;
    let p = pass(&store, &policy).await;
    assert_eq!(p.processed, 0);
}

// ═════════════════════════════════════════════════════════════════════════════
// Test 3 — hold clipping (`gc_select.lua` reads the per-holder hold HASH)
// ═════════════════════════════════════════════════════════════════════════════

/// Plant one read hold field starting at `start_ms`, live for `ttl` from now
/// on the store's clock, as `ext_read_hold.lua` writes one: value
/// `<start_ms>:<expires_ms>`, and the key's expiry at the same TTL.
async fn plant_hold(tv: &TestValkey, start_ms: u64, ttl: Duration) {
    let expires_ms = valkey_now_ms(tv).await + ttl.as_millis() as u64;
    plant_hold_field(tv, "fixture:0", start_ms, expires_ms, ttl).await;
}

/// Plant read hold field `field` with value `<start_ms>:<expires_ms>` and set
/// the key's expiry to `key_ttl`, so a field can be planted already expired
/// (`expires_ms` in the past) while the key — and every other field on it —
/// stays: the state a holder that died leaves while another holder renews.
async fn plant_hold_field(
    tv: &TestValkey,
    field: &str,
    start_ms: u64,
    expires_ms: u64,
    key_ttl: Duration,
) {
    let key = tv.key_schema().ext_read_hold();
    let _: () = tv
        .client()
        .pool
        .hset(&key, (field.to_owned(), format!("{start_ms}:{expires_ms}")))
        .await
        .expect("HSET read_hold");
    let _: bool = tv
        .client()
        .pool
        .pexpire(&key, key_ttl.as_millis() as i64, None)
        .await
        .expect("PEXPIRE read_hold");
}

/// `HDEL` one read hold field: what [`ReadHold::release`] sends.
async fn delete_hold_field(tv: &TestValkey, field: &str) {
    let deleted: i64 = tv
        .client()
        .pool
        .hdel(tv.key_schema().ext_read_hold(), field)
        .await
        .expect("HDEL read_hold");
    assert_eq!(deleted, 1, "precondition: field {field} was on the hold");
}

/// The `(start_ms, expires_ms)` read hold field `field` holds, or `None`.
async fn hold_field(tv: &TestValkey, field: &str) -> Option<(u64, u64)> {
    let raw: Option<String> = tv
        .client()
        .pool
        .hget(tv.key_schema().ext_read_hold(), field)
        .await
        .expect("HGET read_hold");
    raw.map(|value| {
        let (start, expires) = value
            .split_once(':')
            .unwrap_or_else(|| panic!("a hold field's value is <start>:<expires>; got {value}"));
        (
            start.parse().expect("a numeric start"),
            expires.parse().expect("a numeric expiry"),
        )
    })
}

/// `scripts/ext_read_hold.lua` in `mode` (`take` | `renew`) for `field` with
/// a lifetime of `ttl`, called as [`ReadHold`] calls it: the hold's start, or
/// 0 from a `renew` that found its field lapsed.
async fn hold_script(tv: &TestValkey, mode: &str, field: &str, ttl: Duration) -> i64 {
    EXT_READ_HOLD_SCRIPT
        .evalsha_with_reload::<i64, _, _>(
            tv.client().pool.next(),
            vec![tv.key_schema().ext_read_hold()],
            vec![
                mode.to_owned(),
                field.to_owned(),
                ttl.as_millis().to_string(),
            ],
        )
        .await
        .expect("ext_read_hold must run")
}

/// Wait until the store's clock is past `ms`.
async fn wait_for_store_time_past(tv: &TestValkey, ms: u64) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while valkey_now_ms(tv).await <= ms {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the store's clock never passed {ms}"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// End the planted read hold now: `PEXPIRE … 1`, then wait until the store no
/// longer reports the key. A hold planted with a long TTL and ended this way
/// cannot lapse early under a slow runner, and the test does not sleep out a
/// TTL.
async fn expire_hold(tv: &TestValkey) {
    let key = tv.key_schema().ext_read_hold();
    let set: bool = tv
        .client()
        .pool
        .pexpire(&key, 1, None)
        .await
        .expect("PEXPIRE read_hold");
    assert!(set, "precondition: the hold was still live");
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    loop {
        let live: bool = tv.client().pool.exists(&key).await.expect("EXISTS");
        if !live {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "the hold never expired"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// While a read hold lives, no member enqueued at or after its start less the
/// one-second margin is collected — even above the pressure threshold — while
/// older members are; its expiry releases the protected ones.
///
/// The hold's start is placed from the margin member's own `pending:age`
/// score (+300 ms), not from the wall clock, so each member's side of the
/// margin follows from sleeps alone (each a lower bound on elapsed store
/// time): `old` is ≥ 1.4 s before the margin member, so ≥ 1.1 s before the
/// start, past the 1 s margin; the margin member is 300 ms before the start;
/// `after` is released ≥ 400 ms after the margin member, so after the start.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_live_read_hold_protects_members_released_after_its_start() {
    let tv = TestValkey::setup().await;
    let m = metrics();
    // Every release here is far above T.
    let store = gated_store(&tv, &m, 100);
    let policy = policy(
        Duration::ZERO,
        Duration::ZERO,
        Duration::ZERO,
        Duration::from_secs(30),
    );
    let (old, margin, after) = (hash(0x31), hash(0x32), hash(0x33));

    put(&store, old, BLOB).await;
    release(&store, old, OBJ).await;
    tokio::time::sleep(Duration::from_millis(1_400)).await;
    // Released inside the margin: 300 ms before the hold's start.
    put(&store, margin, BLOB).await;
    release(&store, margin, OBJ).await;
    let margin_age = age_of(&tv, &member(margin, OBJ))
        .await
        .expect("the margin member is aged") as u64;
    plant_hold(&tv, margin_age + 300, Duration::from_secs(600)).await;
    tokio::time::sleep(Duration::from_millis(400)).await;
    put(&store, after, BLOB).await;
    release(&store, after, OBJ).await;
    assert!(
        age_of(&tv, &member(after, OBJ)).await.expect("aged") as u64 > margin_age + 300,
        "precondition: `after` was released after the hold's start"
    );

    let held = pass(&store, &policy).await;
    assert_eq!(
        (held.processed, held.deleted),
        (1, 1),
        "only the release from before the hold's start less the margin is collected"
    );
    assert!(
        !data_present(&tv, old).await,
        "the old release is collected"
    );
    for (h, what) in [(margin, "inside the margin"), (after, "after the start")] {
        assert_eq!(
            ref_count(&tv, h).await,
            Some(1),
            "the release {what} is protected while the hold lives, though the total is above T"
        );
    }

    expire_hold(&tv).await;
    let expired = pass(&store, &policy).await;
    assert_eq!(
        (expired.processed, expired.deleted),
        (2, 2),
        "the hold's expiry releases the protected members"
    );
}

/// An un-aged member (an older build's bare `SADD`) is never collected while
/// a read hold lives: the pass that first sees it stamps it with the store's
/// time (b.1jt) — after the hold began — so the hold clips it like a member
/// enqueued then; it is collected once the hold is gone.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_unaged_member_is_never_collected_under_a_read_hold() {
    let tv = TestValkey::setup().await;
    let m = metrics();
    let store = store(&tv, &m);
    let h = hash(0x34);
    put(&store, h, BLOB).await;
    let bare = member(h, OBJ);
    let _: () = tv
        .client()
        .pool
        .sadd(tv.key_schema().ext_pending(), bare.as_str())
        .await
        .expect("bare SADD");
    plant_hold(&tv, valkey_now_ms(&tv).await, Duration::from_secs(600)).await;

    let held = pass(&store, &GcPolicy::IMMEDIATE).await;
    assert_eq!(
        (held.processed, held.stamped),
        (0, 1),
        "an un-aged member is stamped, and waits out the hold"
    );
    assert_eq!(pass(&store, &GcPolicy::IMMEDIATE).await.processed, 0);
    assert_eq!(ref_count(&tv, h).await, Some(1));

    expire_hold(&tv).await;
    let free = pass(&store, &GcPolicy::IMMEDIATE).await;
    assert_eq!((free.processed, free.deleted), (1, 1));
}

/// A field whose `expires_ms` has passed — a holder that died, its key kept
/// alive by another write — clips nothing: an aged member released after its
/// start is collected on the first pass, and an un-aged member — stamped by
/// that pass (b.1jt) — on the next, with the field still on the hold
/// (selection is read-only, so it was the field's expiry, not its absence,
/// that the passes honoured).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_expired_read_hold_field_is_ignored_by_the_selection() {
    let tv = TestValkey::setup().await;
    let m = metrics();
    let store = store(&tv, &m);
    let (aged, bare) = (hash(0x35), hash(0x36));
    let now = valkey_now_ms(&tv).await;
    // Started a minute before every release here, expired a second ago; the
    // key itself lives on for ten minutes.
    plant_hold_field(
        &tv,
        "dead:0",
        now - 60_000,
        now - 1_000,
        Duration::from_secs(600),
    )
    .await;
    put(&store, aged, BLOB).await;
    release(&store, aged, OBJ).await;
    put(&store, bare, BLOB).await;
    let _: () = tv
        .client()
        .pool
        .sadd(tv.key_schema().ext_pending(), member(bare, OBJ).as_str())
        .await
        .expect("bare SADD");

    let swept = pass(&store, &GcPolicy::IMMEDIATE).await;
    assert_eq!(
        (swept.processed, swept.deleted, swept.stamped),
        (1, 1, 1),
        "an expired field does not clip the aged member released after its start; the \
         un-aged one is stamped"
    );
    assert!(!data_present(&tv, aged).await);
    let stamped = pass(&store, &GcPolicy::IMMEDIATE).await;
    assert_eq!(
        (stamped.processed, stamped.deleted),
        (1, 1),
        "nor does it hold back the stamped member"
    );
    assert!(
        hold_field(&tv, "dead:0").await.is_some(),
        "precondition: the expired field was on the hold throughout"
    );
}

/// With several live fields, the clip is the EARLIEST live start, whichever
/// field the store lists first; deleting that field (its holder's release)
/// moves the clip to the next one at once, and deleting the last frees
/// everything. The fields are planted latest-first so a selection that used
/// the first field it read would clip at the later start.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_earliest_live_hold_start_clips_the_selection() {
    let tv = TestValkey::setup().await;
    let m = metrics();
    // No margin: each member's side of a start follows from its age alone.
    let store = store(&tv, &m).with_hold_margin_for_test(Duration::ZERO);
    let (first, middle, last) = (hash(0x37), hash(0x38), hash(0x39));
    for (i, h) in [first, middle, last].into_iter().enumerate() {
        if i > 0 {
            tokio::time::sleep(Duration::from_millis(300)).await;
        }
        put(&store, h, BLOB).await;
        release(&store, h, OBJ).await;
    }
    let age = |h: [u8; 32]| {
        let tv = &tv;
        async move { age_of(tv, &member(h, OBJ)).await.expect("aged") as u64 }
    };
    let (first_age, middle_age) = (age(first).await, age(middle).await);
    assert!(
        middle_age >= first_age + 300 && age(last).await >= middle_age + 300,
        "precondition: the releases are at least 300 ms apart on the store's clock"
    );
    let far = valkey_now_ms(&tv).await + 600_000;
    plant_hold_field(
        &tv,
        "late:0",
        middle_age + 150,
        far,
        Duration::from_secs(600),
    )
    .await;
    plant_hold_field(
        &tv,
        "early:0",
        first_age + 150,
        far,
        Duration::from_secs(600),
    )
    .await;

    let both = pass(&store, &GcPolicy::IMMEDIATE).await;
    assert_eq!(
        (both.processed, both.deleted),
        (1, 1),
        "only the release before the earliest live start is collected"
    );
    assert!(!data_present(&tv, first).await);

    delete_hold_field(&tv, "early:0").await;
    let late_only = pass(&store, &GcPolicy::IMMEDIATE).await;
    assert_eq!(
        (late_only.processed, late_only.deleted),
        (1, 1),
        "the earlier holder's release moves the clip to the later holder's start"
    );
    assert!(!data_present(&tv, middle).await);
    assert_eq!(
        ref_count(&tv, last).await,
        Some(1),
        "the release after the later start stays protected"
    );

    delete_hold_field(&tv, "late:0").await;
    let free = pass(&store, &GcPolicy::IMMEDIATE).await;
    assert_eq!((free.processed, free.deleted), (1, 1));
}

/// An un-aged member — stamped by the first pass that sees it (b.1jt) — waits
/// while ANY field is live, here one live field beside one expired one, and is
/// collected once only the expired field remains.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_unaged_member_waits_while_any_read_hold_field_lives() {
    let tv = TestValkey::setup().await;
    let m = metrics();
    let store = store(&tv, &m);
    let h = hash(0x3A);
    put(&store, h, BLOB).await;
    let _: () = tv
        .client()
        .pool
        .sadd(tv.key_schema().ext_pending(), member(h, OBJ).as_str())
        .await
        .expect("bare SADD");
    let now = valkey_now_ms(&tv).await;
    let ten_minutes = Duration::from_secs(600);
    plant_hold_field(&tv, "dead:0", now - 60_000, now - 1_000, ten_minutes).await;
    plant_hold_field(&tv, "live:0", now, now + 600_000, ten_minutes).await;

    let held = pass(&store, &GcPolicy::IMMEDIATE).await;
    assert_eq!(held.processed, 0, "one live field holds the un-aged member");
    assert_eq!(ref_count(&tv, h).await, Some(1));

    delete_hold_field(&tv, "live:0").await;
    let free = pass(&store, &GcPolicy::IMMEDIATE).await;
    assert_eq!(
        (free.processed, free.deleted),
        (1, 1),
        "with only an expired field left, the un-aged member is collected"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// The read hold's lifecycle — `ext_read_hold.lua` and `ReadHold` (H-a … H-e)
// ═════════════════════════════════════════════════════════════════════════════

/// H-a, no chaining: holders A and B overlap; A's release moves the clip to
/// B's own start at once, so a release enqueued between the two starts is
/// collected while B still holds. Each holder's field keeps its own start —
/// a later holder neither inherits nor extends an earlier one's.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_released_hold_moves_the_clip_to_the_next_holders_own_start() {
    let tv = TestValkey::setup().await;
    let m = metrics();
    let store = store(&tv, &m).with_hold_margin_for_test(Duration::ZERO);
    let (before, between, after) = (hash(0x41), hash(0x42), hash(0x43));
    const GAP: Duration = Duration::from_millis(200);

    put(&store, before, BLOB).await;
    release(&store, before, OBJ).await;
    tokio::time::sleep(GAP).await;
    let a = ReadHold::take(tv.client(), tv.key_schema(), OBJ)
        .await
        .expect("holder A takes its hold");
    tokio::time::sleep(GAP).await;
    put(&store, between, BLOB).await;
    release(&store, between, OBJ).await;
    tokio::time::sleep(GAP).await;
    let b = ReadHold::take(tv.client(), tv.key_schema(), OBJ_B)
        .await
        .expect("holder B takes its hold");
    tokio::time::sleep(GAP).await;
    put(&store, after, BLOB).await;
    release(&store, after, OBJ).await;

    let (a_start, _) = hold_field(&tv, a.field()).await.expect("A's field");
    let (b_start, _) = hold_field(&tv, b.field()).await.expect("B's field");
    let age = |h: [u8; 32]| {
        let tv = &tv;
        async move { age_of(tv, &member(h, OBJ)).await.expect("aged") as u64 }
    };
    assert!(
        age(before).await < a_start
            && a_start <= age(between).await
            && age(between).await + GAP.as_millis() as u64 <= age(after).await,
        "precondition: before < A's start <= between, and after is at least a gap later"
    );
    assert!(
        age(between).await < b_start && b_start <= age(after).await,
        "B's field starts at B's own take, after the release between the holds — not at A's \
         start ({a_start}); got {b_start}"
    );

    let both = pass(&store, &GcPolicy::IMMEDIATE).await;
    assert_eq!(
        (both.processed, both.deleted),
        (1, 1),
        "under both holds only the release before A's start is collected"
    );

    let a_field = a.field().to_owned();
    a.release().await;
    assert_eq!(
        (
            hold_field(&tv, &a_field).await,
            hold_field(&tv, b.field()).await.map(|(start, _)| start)
        ),
        (None, Some(b_start)),
        "A's release deletes its field and only its field; B keeps its own start"
    );
    let b_only = pass(&store, &GcPolicy::IMMEDIATE).await;
    assert_eq!(
        (b_only.processed, b_only.deleted),
        (1, 1),
        "with A released the clip is B's own start: the release between the starts is \
         collected (a hold that kept the earliest start would still protect it)"
    );
    assert!(!data_present(&tv, between).await);
    assert_eq!(ref_count(&tv, after).await, Some(1), "B still protects");

    b.release().await;
    assert_eq!(hold_field_count(&tv).await, 0);
    let free = pass(&store, &GcPolicy::IMMEDIATE).await;
    assert_eq!((free.processed, free.deleted), (1, 1));
}

/// A `ReadHold` dropped unreleased — its catch-up cancelled mid-rebuild —
/// spawns the `HDEL` of its field: the field goes within seconds, long
/// before its 15 s TTL could lapse it, and the other holder's field stays.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_hold_dropped_unreleased_deletes_its_field() {
    let tv = TestValkey::setup().await;
    let dropped = ReadHold::take(tv.client(), tv.key_schema(), OBJ)
        .await
        .expect("the dropped holder takes its hold");
    let kept = ReadHold::take(tv.client(), tv.key_schema(), OBJ_B)
        .await
        .expect("the other holder takes its hold");
    let field = dropped.field().to_owned();
    let (_, expires) = hold_field(&tv, &field).await.expect("taken");

    drop(dropped);

    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    while hold_field(&tv, &field).await.is_some() {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the dropped hold's field was not deleted"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(
        valkey_now_ms(&tv).await < expires,
        "precondition: the field went before its TTL could lapse it"
    );
    assert!(hold_field(&tv, kept.field()).await.is_some());
    kept.release().await;
}

/// The number of fields on the read hold.
async fn hold_field_count(tv: &TestValkey) -> u64 {
    tv.client()
        .pool
        .hlen(tv.key_schema().ext_read_hold())
        .await
        .expect("HLEN read_hold")
}

/// H-c, server-side lapse: a renewal that runs after its field expired —
/// here the field is still on the hold, kept by another holder's write —
/// answers [`ReadHoldRenewal::Lapsed`], deletes the field, and writes nothing
/// else: the other holder's field and the key's expiry are untouched, and a
/// second renewal finds nothing to revive. A renewal of a live field
/// extends it, keeps its start, and writes a deadline one TTL past the store
/// time it ran at.
///
/// The TTL keeps the store's clock off every assertion that is not about it
/// (the test-writing guide's rule (c)): twice the command timeout of the
/// client the renewal runs on, so the renewal of a live field cannot find it
/// lapsed unless it ran later than that client's own commands may; and the
/// field's expiry is staged — its value rewritten to a deadline the store's
/// clock has passed — rather than waited out.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_renewal_after_the_field_expired_answers_lapsed_and_writes_nothing() {
    let tv = TestValkey::setup().await;
    let command_timeout = tv
        .client()
        .pool
        .next()
        .perf_config()
        .default_command_timeout;
    assert!(
        command_timeout > Duration::ZERO,
        "precondition: the client bounds its commands"
    );
    let ttl = command_timeout * 2;
    tv.client().set_read_hold_timing_for_test(ReadHoldTiming {
        ttl,
        renew_every: ttl / 4,
    });
    let ttl_ms = ttl.as_millis() as u64;
    let hold = ReadHold::take(tv.client(), tv.key_schema(), OBJ)
        .await
        .expect("the hold is taken");
    let (start, expires) = hold_field(&tv, hold.field()).await.expect("taken");
    let taken_at = expires - ttl_ms;
    // The renewal must run at a later store millisecond than the take, so an
    // extension is visible.
    wait_for_store_time_past(&tv, taken_at).await;
    assert_eq!(
        hold.renew().await.expect("renew runs"),
        ReadHoldRenewal::Live
    );
    let (renewed_start, renewed_expires) = hold_field(&tv, hold.field()).await.expect("renewed");
    let now = valkey_now_ms(&tv).await;
    assert_eq!(renewed_start, start, "a renewal keeps the field's start");
    assert!(
        renewed_expires > expires,
        "a renewal extends the field: {renewed_expires} > {expires}"
    );
    let renewed_at = renewed_expires - ttl_ms;
    assert!(
        taken_at < renewed_at && renewed_at <= now,
        "the renewal's deadline is one TTL past the store time it ran at, after the take and \
         not after now: ran at {renewed_at}, taken at {taken_at}, now {now}"
    );

    // Another holder, written with a ten-minute lifetime, keeps the key — and
    // so this holder's expired field — on the store. This holder's field is
    // then made expired: its deadline rewritten to one the store's clock has
    // passed, its start kept.
    let other = format!("{OBJ_B}:other");
    assert!(hold_script(&tv, "take", &other, Duration::from_secs(600)).await > 0);
    let other_value = hold_field(&tv, &other).await.expect("the other field");
    let lapsed_at = valkey_now_ms(&tv).await - 1;
    plant_hold_field(
        &tv,
        hold.field(),
        start,
        lapsed_at,
        Duration::from_secs(600),
    )
    .await;
    assert_eq!(
        hold_field(&tv, hold.field()).await,
        Some((start, lapsed_at)),
        "precondition: the expired field is still on the hold"
    );
    let key_ttl_before: i64 = tv
        .client()
        .pool
        .pttl(tv.key_schema().ext_read_hold())
        .await
        .expect("PTTL");

    assert_eq!(
        hold.renew().await.expect("renew runs"),
        ReadHoldRenewal::Lapsed,
        "a renewal after the field expired answers LAPSED"
    );
    assert_eq!(
        hold_field(&tv, hold.field()).await,
        None,
        "the lapsed field is deleted, not revived"
    );
    assert_eq!(
        hold_field(&tv, &other).await,
        Some(other_value),
        "the lapsed renewal touches no other holder's field"
    );
    let key_ttl_after: i64 = tv
        .client()
        .pool
        .pttl(tv.key_schema().ext_read_hold())
        .await
        .expect("PTTL");
    assert!(
        key_ttl_after <= key_ttl_before,
        "the lapsed renewal does not refresh the key's expiry: {key_ttl_after} > \
         {key_ttl_before}"
    );
    assert_eq!(
        hold.renew().await.expect("renew runs"),
        ReadHoldRenewal::Lapsed,
        "nothing revives it later either"
    );
    assert_eq!(hold_field(&tv, hold.field()).await, None);
    hold.release().await;
}

/// H-d, re-send: a `take` sent twice for one field — fred's re-send of an
/// `EVALSHA` whose reply was lost — keeps the first write's start and
/// extends the field. Another holder's `take` starts its own field.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_resent_take_keeps_the_fields_start() {
    let tv = TestValkey::setup().await;
    let field = format!("{OBJ}:resent");
    let ttl = Duration::from_secs(60);

    let first = hold_script(&tv, "take", &field, ttl).await;
    let (_, first_expires) = hold_field(&tv, &field).await.expect("taken");
    tokio::time::sleep(Duration::from_millis(50)).await;
    let resent = hold_script(&tv, "take", &field, ttl).await;

    assert_eq!(resent, first, "the re-sent take answers the first start");
    let (start, expires) = hold_field(&tv, &field).await.expect("still held");
    assert_eq!(start as i64, first, "the field keeps the first start");
    assert!(
        expires > first_expires,
        "the re-send extends the field: {expires} > {first_expires}"
    );

    let other = format!("{OBJ_B}:other");
    let other_start = hold_script(&tv, "take", &other, ttl).await;
    assert!(
        other_start > first,
        "another holder starts its own field at its own time"
    );
    assert_eq!(hold_field_count(&tv).await, 2);
}

/// H-e, lapse after a crash: a field whose holder died (never renewed, never
/// released) clips while it lives and stops at its expiry, though it stays on
/// the hold while another holder keeps renewing; the next `take` prunes it;
/// and once nobody writes, the key goes away one TTL after its last write.
///
/// Slow (about 4 s): the dead field must outlive one pass and then expire
/// in real store time, and the key must lapse in real time too.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_dead_holders_field_stops_clipping_at_expiry_and_is_pruned() {
    let tv = TestValkey::setup().await;
    let m = metrics();
    let store = store(&tv, &m).with_hold_margin_for_test(Duration::ZERO);
    let h = hash(0x44);
    let ttl = Duration::from_secs(2);
    let dead = format!("{OBJ}:dead");
    let renewing = format!("{OBJ_B}:renewing");

    assert!(hold_script(&tv, "take", &dead, ttl).await > 0);
    tokio::time::sleep(Duration::from_millis(100)).await;
    put(&store, h, BLOB).await;
    release(&store, h, OBJ).await;
    tokio::time::sleep(Duration::from_millis(100)).await;
    // Started after `h`'s release, so it never protects `h` itself.
    assert!(hold_script(&tv, "take", &renewing, ttl).await > 0);
    let (_, dead_expires) = hold_field(&tv, &dead).await.expect("the dead field");

    let held = pass(&store, &GcPolicy::IMMEDIATE).await;
    assert!(
        valkey_now_ms(&tv).await < dead_expires,
        "precondition: the pass ran while the dead holder's field was live"
    );
    assert_eq!(held.processed, 0, "the dead holder's live field clips");
    assert_eq!(ref_count(&tv, h).await, Some(1));

    // The live holder renews; the dead one does not.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while valkey_now_ms(&tv).await <= dead_expires {
        assert!(tokio::time::Instant::now() < deadline);
        assert!(hold_script(&tv, "renew", &renewing, ttl).await > 0);
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(hold_script(&tv, "renew", &renewing, ttl).await > 0);
    assert!(
        hold_field(&tv, &dead).await.is_some(),
        "precondition: the expired field is still on the hold"
    );
    let lapsed = pass(&store, &GcPolicy::IMMEDIATE).await;
    assert_eq!(
        (lapsed.processed, lapsed.deleted),
        (1, 1),
        "an expired field stops clipping at its expiry"
    );

    let next = format!("{OBJ}:next");
    assert!(hold_script(&tv, "take", &next, ttl).await > 0);
    assert_eq!(
        hold_field(&tv, &dead).await,
        None,
        "the next take prunes the expired field"
    );
    assert!(hold_field(&tv, &renewing).await.is_some());

    // Nobody writes again: the key lapses one TTL after the last write.
    let key = tv.key_schema().ext_read_hold();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    loop {
        let present: bool = tv.client().pool.exists(&key).await.expect("EXISTS");
        if !present {
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "the hold key never lapsed after its last write"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// The live read hold gauge
// ═════════════════════════════════════════════════════════════════════════════

/// The leader publishes `live_edit_ext_read_holds` — the live fields its
/// pass's selection counted, an expired field not among them — and a
/// non-leader publishes 0.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_leader_publishes_the_live_read_hold_count() {
    let tv = TestValkey::setup().await;
    let m = metrics();
    let s: Arc<dyn ExternalValueStore> = Arc::new(store(&tv, &m));
    let now = valkey_now_ms(&tv).await;
    let ten_minutes = Duration::from_secs(600);
    plant_hold_field(&tv, "live:1", now, now + 600_000, ten_minutes).await;
    plant_hold_field(&tv, "live:2", now, now + 600_000, ten_minutes).await;
    plant_hold_field(&tv, "dead:0", now - 60_000, now - 1_000, ten_minutes).await;
    let hold_everything = policy(
        Duration::from_secs(3600),
        Duration::from_secs(3600),
        Duration::from_secs(3600),
        Duration::from_secs(30),
    );
    let run = |leader: bool| {
        let cancel = CancellationToken::new();
        let handle = spawn_gc_task_with_leader(
            Arc::clone(&s),
            hold_everything,
            cancel.clone(),
            Some(Arc::clone(&m)),
            move || async move { leader },
        );
        (cancel, handle)
    };

    let (cancel, handle) = run(true);
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while gauge(&m, "live_edit_ext_read_holds") != 2.0 {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the leader never published two live holds; last {}",
            gauge(&m, "live_edit_ext_read_holds")
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    cancel.cancel();
    handle.await.unwrap();

    let (cancel, handle) = run(false);
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while gauge(&m, "live_edit_ext_read_holds") != 0.0 {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the non-leader never zeroed the gauge"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    cancel.cancel();
    handle.await.unwrap();
    assert_eq!(
        hold_field_count(&tv).await,
        3,
        "precondition held throughout: two live fields and one expired"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Ledger accounting
// ═════════════════════════════════════════════════════════════════════════════

/// A release accounts its blob's size; a replayed enqueue counts nothing
/// twice; the claim and completion take the bytes out exactly once and settle
/// the member.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_release_accounts_its_blob_size_and_the_total_returns_to_zero() {
    let tv = TestValkey::setup().await;
    let m = metrics();
    let store = store(&tv, &m);
    let h = hash(0x41);
    put(&store, h, 1_234).await;
    release(&store, h, OBJ).await;
    release(&store, h, OBJ).await;
    let mem = member(h, OBJ);
    assert_eq!(bytes_of(&tv, &mem).await, Some(1_234));
    assert_eq!(
        total(&tv).await,
        1_234,
        "a replayed enqueue is not counted twice"
    );
    assert!(age_of(&tv, &mem).await.is_some());

    let stats = process_pending_gc(&store, Some(&m), immediate_keeping_records()).await;
    assert_eq!(stats.deleted, 1);
    assert_eq!(total(&tv).await, 0);
    assert_eq!(bytes_of(&tv, &mem).await, None);
    assert_eq!(age_of(&tv, &mem).await, None);
    assert!(settled_of(&tv, &mem).await.is_some(), "the member settled");
}

/// A release whose blob size cannot be read is still enqueued, accounting 0
/// bytes: under-counting the budget is the lesser harm next to losing a
/// release. A wrong-typed data key makes the `STRLEN` fail.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_failed_size_read_still_enqueues_the_release_at_zero_bytes() {
    let tv = TestValkey::setup().await;
    let m = metrics();
    let store = store(&tv, &m);
    let h = hash(0x42);
    let _: () = tv
        .client()
        .pool
        .hset(
            tv.key_schema().ext_data(&sha256_to_hex(&h)),
            [("not", "a-blob")],
        )
        .await
        .expect("plant a wrong-typed data key");
    release(&store, h, OBJ).await;
    let mem = member(h, OBJ);
    assert!(in_pending(&tv, &mem).await, "the release is enqueued");
    assert_eq!(bytes_of(&tv, &mem).await, Some(0));
    assert_eq!(total(&tv).await, 0);
}

/// A commit's `surplus_acquire` release — the hand-back of the pre-flight `+1`
/// of a hash the object still references — accounts 0 bytes: its decrement
/// cannot free the blob. (A `rewrite` release of the same commit path accounts
/// the blob's size: `b_yuo_wrel_tests::lost_reply::lost_reply_enqueues_once`
/// asserts its 22 bytes.)
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_commit_surplus_release_accounts_zero_bytes() {
    use super::b_qpf_tests::{
        create_session, empty_state, ext_hash, layer_store, pending_members, pre_store,
    };
    use crate::common::owned_delta::delta_with_set_field_external;
    use live_edit::config::UpdateLayerBackend;
    use live_edit::storage::test_support::relay::RelayedUpdates;

    let tv = TestValkey::setup().await;
    let m = metrics();
    let ls = layer_store(&tv, UpdateLayerBackend::Lua, &m);
    let ext = Arc::new(store(&tv, &m));
    let session_id = create_session(&tv, "b-yuo-surplus").await;
    let oid = ls
        .create_layer(
            &session_id,
            "b-yuo-surplus",
            "u",
            &empty_state(),
            ext.clone(),
        )
        .await
        .expect("create_layer");
    let a = ext_hash(0x43);
    let delta = |order| delta_with_set_field_external(1, "color", a.to_vec(), 1_024, 1, order);

    pre_store(&ext, &[a]).await;
    ls.update_layer_relayed(&session_id, "b-yuo-surplus", oid, Some(delta(0)), "w", 1)
        .await
        .expect("first commit");
    // The same hash again: the commit owes back its pre-flight `+1`.
    pre_store(&ext, &[a]).await;
    let (seq, _) = ls
        .update_layer_relayed(&session_id, "b-yuo-surplus", oid, Some(delta(1)), "w", 2)
        .await
        .expect("second commit");
    let surplus = ExtRelease::commit(a, oid, seq).member();
    assert_eq!(pending_members(&tv).await, vec![surplus.clone()]);
    assert_eq!(
        bytes_of(&tv, &surplus).await,
        Some(0),
        "a surplus_acquire release accounts 0 bytes"
    );
    assert_eq!(total(&tv).await, 0);
}

/// `ext_release_enqueue.lua` called as the plain enqueue calls it — six keys
/// since b.1jt, the last two (`inflight`, `settled`) read for its replay
/// refusal — with `argv` as its (member, bytes) pairs. Replies
/// `(added, refused)`.
async fn enqueue_script(
    tv: &TestValkey,
    argv: Vec<String>,
) -> Result<(i64, i64), fred::error::Error> {
    let keys = tv.key_schema();
    EXT_RELEASE_ENQUEUE_SCRIPT
        .evalsha_with_reload::<(i64, i64), _, _>(
            tv.client().pool.next(),
            vec![
                keys.ext_pending(),
                keys.ext_pending_age(),
                keys.ext_pending_bytes(),
                keys.ext_pending_total(),
                keys.ext_inflight(),
                keys.ext_settled(),
            ],
            argv,
        )
        .await
}

/// A byte count `INCRBY` would refuse — a leading zero (`007`, `00`) — is
/// refused `ENCODING` before the first write, so nothing of the call is
/// written, not even a canonical pair ahead of it: no member on `pending`,
/// no age, no bytes, no total. `0` itself is canonical and accepted.
///
/// Before the canonical check the script took any digit string: the `SADD`
/// and `ZADD` of the pair ran, and `INCRBY` then refused the value, leaving a
/// member on `pending` the ledger never counted.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_byte_count_with_a_leading_zero_is_refused_encoding_with_nothing_written() {
    let tv = TestValkey::setup().await;
    let (good, bad) = (member(hash(0x6C), OBJ), member(hash(0x6D), OBJ));
    for spelling in ["007", "00"] {
        let refused = enqueue_script(
            &tv,
            vec![
                good.clone(),
                "16".to_owned(),
                bad.clone(),
                spelling.to_owned(),
            ],
        )
        .await
        .expect_err("a non-canonical byte count must be refused");
        assert!(
            refused.details().contains(
                "ENCODING: bytes for a release member must be a canonical non-negative decimal"
            ),
            "{spelling:?}: refused ENCODING, before any write; got {refused:?}"
        );
        for m in [&good, &bad] {
            assert!(
                !in_pending(&tv, m).await,
                "{spelling:?}: no member enqueued"
            );
            assert_eq!(age_of(&tv, m).await, None, "{spelling:?}: no age");
            assert_eq!(bytes_of(&tv, m).await, None, "{spelling:?}: no bytes");
        }
        assert_eq!(total(&tv).await, 0, "{spelling:?}: the total is untouched");
    }

    let added = enqueue_script(&tv, vec![bad.clone(), "0".to_owned()])
        .await
        .expect("`0` is a canonical byte count");
    assert_eq!(added, (1, 0), "added, nothing refused");
    assert!(in_pending(&tv, &bad).await);
    assert_eq!(bytes_of(&tv, &bad).await, Some(0));
    assert_eq!(total(&tv).await, 0);
}

// ═════════════════════════════════════════════════════════════════════════════
// H1 — fred's re-send of the decrement, at the script
// ═════════════════════════════════════════════════════════════════════════════

/// A decrement re-run with identical arguments — fred's transparent re-send
/// of an `EVALSHA` whose connection dropped after the server ran it — answers
/// `replayed` and decrements nothing: a shared blob keeps its other reference.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_identical_decrement_rerun_is_replayed_and_decrements_once() {
    let tv = TestValkey::setup().await;
    let m = metrics();
    let store = store(&tv, &m);
    let h = hash(0x51);
    put(&store, h, 64).await;
    put(&store, h, 64).await;
    release(&store, h, OBJ).await;
    let mem = member(h, OBJ);
    let claim_ms = claim(&tv, &mem).await;
    assert!(claim_ms > 0);

    let first = decrement(&tv, h, &mem, claim_ms, GC_DECREMENT_FENCE_MS).await;
    let second = decrement(&tv, h, &mem, claim_ms, GC_DECREMENT_FENCE_MS).await;
    assert_eq!(first, ("decremented".to_owned(), 1, 0));
    assert_eq!(
        second,
        ("replayed".to_owned(), 1, 0),
        "the re-send is a replay. Before b.yuo it decremented again: ref_count 0 and the blob \
         deleted under its other holder"
    );
    assert_eq!(ref_count(&tv, h).await, Some(1));
    assert!(data_present(&tv, h).await);
    assert_eq!(
        record_of(&tv, h, &mem).await,
        Some(claim_ms as f64),
        "the replay record holds the claim time"
    );
}

/// The single-reference variant: the decrement deletes the blob, the same
/// content is stored again by a new reference, and then the identical
/// decrement arrives — `replayed`, and the restored blob survives. The record
/// is a separate key exactly so this holds: the meta was deleted with the
/// blob.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_replay_after_delete_and_restore_keeps_the_restored_blob() {
    let tv = TestValkey::setup().await;
    let m = metrics();
    let store = store(&tv, &m);
    let h = hash(0x52);
    put(&store, h, 64).await;
    release(&store, h, OBJ).await;
    let mem = member(h, OBJ);
    let claim_ms = claim(&tv, &mem).await;

    assert_eq!(
        decrement(&tv, h, &mem, claim_ms, GC_DECREMENT_FENCE_MS).await,
        ("deleted".to_owned(), 0, 64),
        "the delete answers the blob's length as the bytes it freed"
    );
    assert!(!meta_present(&tv, h).await && !data_present(&tv, h).await);
    put(&store, h, 64).await;
    assert_eq!(
        decrement(&tv, h, &mem, claim_ms, GC_DECREMENT_FENCE_MS).await,
        ("replayed".to_owned(), 1, 0)
    );
    assert_eq!(ref_count(&tv, h).await, Some(1));
    assert!(
        data_present(&tv, h).await,
        "the restored blob survives the replay"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// H2 — fred's re-send of the decrement, through `process_pending`
// ═════════════════════════════════════════════════════════════════════════════

/// `ExecuteTwice` at the decrement site: the pass sends the decrement twice
/// with identical arguments and sees the second reply. The blob loses one
/// reference, the member settles once, the decrement counter reads one
/// `replayed` (the first reply was the lost one), and nothing underflows.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_resent_decrement_through_the_pass_applies_once() {
    use live_edit::storage::external_values::{PendingFault, ProcessPendingFaultInjector};
    let tv = TestValkey::setup().await;
    let m = metrics();
    let faults: ProcessPendingFaultInjector = Arc::new(std::sync::Mutex::new(
        std::collections::VecDeque::from(vec![(1usize, PendingFault::ExecuteTwice)]),
    ));
    let store = store(&tv, &m).with_process_pending_fault_injector_for_test(faults.clone());
    let h = hash(0x53);
    put(&store, h, 64).await;
    put(&store, h, 64).await;
    release(&store, h, OBJ).await;

    let p = pass(&store, &immediate_keeping_records()).await;
    assert!(faults.lock().unwrap().is_empty(), "the re-send was staged");
    assert_eq!((p.processed, p.replayed, p.deleted), (1, 1, 0));
    assert_eq!(ref_count(&tv, h).await, Some(1), "one decrement, not two");
    assert!(data_present(&tv, h).await);
    assert_eq!(
        zcard(&tv, tv.key_schema().ext_settled()).await,
        1,
        "settled once"
    );
    assert_eq!(
        (decrements(&m, "replayed"), decrements(&m, "applied")),
        (1, 0)
    );
    assert_eq!(underflows(&m), 0);
    assert_eq!(total(&tv).await, 0);
}

// ═════════════════════════════════════════════════════════════════════════════
// H3 — the re-claim of a claim whose decrement already applied
// ═════════════════════════════════════════════════════════════════════════════

/// Leave `h`'s member in `inflight` with its decrement applied — the
/// completion fails transiently — then age its claim past the re-claim age
/// D, keeping this build's encoded score (b.1jt), so the next pass re-claims
/// it in place. With `restore_before_reclaim`, the same content is stored
/// again first.
#[cfg(feature = "test-fault-injection")]
async fn decrement_then_strand_and_age(
    tv: &TestValkey,
    m: &Arc<Metrics>,
    h: [u8; 32],
    restore_before_reclaim: bool,
) {
    use live_edit::storage::external_values::{PendingFault, ProcessPendingFaultInjector};
    let faults: ProcessPendingFaultInjector = Arc::new(std::sync::Mutex::new(
        std::collections::VecDeque::from(vec![(
            2usize,
            PendingFault::Fail(fred::error::Error::new(
                fred::error::ErrorKind::Timeout,
                "b.yuo: the completion's reply is lost",
            )),
        )]),
    ));
    let faulted = store(tv, m).with_process_pending_fault_injector_for_test(faults.clone());
    let first = pass(&faulted, &immediate_keeping_records()).await;
    assert!(
        faults.lock().unwrap().is_empty(),
        "the completion fault fired"
    );
    assert_eq!(first.processed, 1);
    let mem = member(h, OBJ);
    assert!(
        in_inflight(tv, &mem).await,
        "precondition: stranded in inflight"
    );

    if restore_before_reclaim {
        put(&store(tv, m), h, 64).await;
    }
    crate::storage::b_1jt_rollout_tests::plant_own_claim(
        tv,
        &mem,
        crate::storage::b_1jt_rollout_tests::STALE_OWN_CLAIM_AGE,
    )
    .await;
}

/// A shared blob's member decremented before its completion was lost, then
/// re-claimed in place: the next pass answers `replayed`, the blob keeps its
/// other reference, and the ledger total returns to zero. Before b.yuo the
/// re-claim decremented again and deleted the blob under its other holder.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_reclaimed_claim_whose_decrement_applied_is_replayed() {
    let tv = TestValkey::setup().await;
    let m = metrics();
    let h = hash(0x61);
    let s = store(&tv, &m);
    put(&s, h, 64).await;
    put(&s, h, 64).await;
    release(&s, h, OBJ).await;
    decrement_then_strand_and_age(&tv, &m, h, false).await;
    assert_eq!(
        ref_count(&tv, h).await,
        Some(1),
        "precondition: decremented once"
    );

    let next = pass(&s, &immediate_keeping_records()).await;
    assert_eq!(
        (next.reclaimed, next.processed, next.replayed, next.deleted),
        (1, 1, 1, 0)
    );
    assert!(
        !in_pending(&tv, &member(h, OBJ)).await,
        "re-claimed in place, never back in pending"
    );
    assert_eq!(ref_count(&tv, h).await, Some(1));
    assert!(data_present(&tv, h).await);
    assert_eq!(total(&tv).await, 0, "the bytes leave the total once");
    assert_eq!(underflows(&m), 0);
}

/// The single-reference variant: the decrement deleted the blob, the content
/// was stored again before the re-claim, and the re-claim's replay keeps the
/// new blob.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_reclaimed_claim_whose_decrement_deleted_the_blob_keeps_the_restored_one() {
    let tv = TestValkey::setup().await;
    let m = metrics();
    let h = hash(0x62);
    let s = store(&tv, &m);
    put(&s, h, 64).await;
    release(&s, h, OBJ).await;
    decrement_then_strand_and_age(&tv, &m, h, true).await;
    assert_eq!(ref_count(&tv, h).await, Some(1), "precondition: re-stored");

    let next = pass(&s, &immediate_keeping_records()).await;
    assert_eq!((next.replayed, next.deleted), (1, 0));
    assert_eq!(ref_count(&tv, h).await, Some(1));
    assert!(data_present(&tv, h).await, "the new blob survives");
    assert_eq!(total(&tv).await, 0);
}

// ═════════════════════════════════════════════════════════════════════════════
// H4 — the per-pass re-claim
// ═════════════════════════════════════════════════════════════════════════════

/// A member a transient decrement error left in `inflight` is re-claimed in
/// place and collected by the running loop itself, with no restart, and the
/// ledger total — what the upload gate reads — returns to zero; the member is
/// never seen in `pending` meanwhile (b.1jt). Before b.yuo only a process
/// start drained `inflight`, so the member and its bytes stayed until some
/// replica restarted. The re-claim age is shortened to 300 ms here.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_loop_reclaims_and_collects_a_stuck_claim_without_a_restart() {
    use live_edit::storage::external_values::{PendingFault, ProcessPendingFaultInjector};
    let tv = TestValkey::setup().await;
    let m = metrics();
    let faults: ProcessPendingFaultInjector = Arc::new(std::sync::Mutex::new(
        std::collections::VecDeque::from(vec![(
            1usize,
            PendingFault::Fail(fred::error::Error::new(
                fred::error::ErrorKind::Timeout,
                "b.yuo: the decrement times out",
            )),
        )]),
    ));
    let s = Arc::new(store(&tv, &m).with_process_pending_fault_injector_for_test(faults.clone()));
    let h = hash(0x71);
    put(&s, h, 2_000).await;
    release(&s, h, OBJ).await;
    let mem = member(h, OBJ);
    let policy = policy(
        Duration::ZERO,
        Duration::ZERO,
        Duration::ZERO,
        Duration::from_millis(300),
    );

    let cancel = CancellationToken::new();
    let handle = spawn_gc_task_with_leader(
        s.clone() as Arc<dyn ExternalValueStore>,
        policy,
        cancel.clone(),
        Some(Arc::clone(&m)),
        || async { true },
    );
    let deadline = tokio::time::Instant::now() + Duration::from_secs(15);
    loop {
        if faults.lock().unwrap().is_empty()
            && !data_present(&tv, h).await
            && !in_inflight(&tv, &mem).await
            && total(&tv).await == 0
        {
            break;
        }
        assert!(
            !faults.lock().unwrap().is_empty() || !in_pending(&tv, &mem).await,
            "a claimed member is never moved back to pending"
        );
        assert!(
            tokio::time::Instant::now() < deadline,
            "the loop never re-claimed the stuck claim: inflight={}, data={}, total={}",
            in_inflight(&tv, &mem).await,
            data_present(&tv, h).await,
            total(&tv).await
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    cancel.cancel();
    tokio::time::timeout(Duration::from_secs(5), handle)
        .await
        .expect("the loop stops on cancel")
        .expect("the loop must not panic");
    assert_eq!(underflows(&m), 0);
}

// ═════════════════════════════════════════════════════════════════════════════
// H5 — the claim-age fence
// ═════════════════════════════════════════════════════════════════════════════

/// A decrement executing more than the fence after its claim is refused
/// `stale`: no decrement and no replay record.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_decrement_older_than_the_fence_is_stale_and_writes_nothing() {
    let tv = TestValkey::setup().await;
    let m = metrics();
    let s = store(&tv, &m);
    let h = hash(0x81);
    put(&s, h, 64).await;
    let mem = member(h, OBJ);
    let claim_ms = (valkey_now_ms(&tv).await - GC_DECREMENT_FENCE_MS - 1_000) as i64;

    let (action, age, freed_bytes) = decrement(&tv, h, &mem, claim_ms, GC_DECREMENT_FENCE_MS).await;
    assert_eq!((action.as_str(), freed_bytes), ("stale", 0));
    assert!(
        age > GC_DECREMENT_FENCE_MS as i64,
        "the reply carries the claim age"
    );
    assert_eq!(ref_count(&tv, h).await, Some(1), "nothing decremented");
    assert_eq!(record_of(&tv, h, &mem).await, None, "nothing recorded");
}

/// A late copy of a decrement — written by fred after its caller gave up, or
/// by a sweeper paused after its claim — arriving after its member settled and
/// its record was torn down is refused `stale`. Without the fence it would
/// find no record and decrement a second time. `fence_ms` is 300 ms here so
/// the copy can outlive it in real time.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_late_copy_after_settle_and_teardown_is_stale() {
    let tv = TestValkey::setup().await;
    let m = metrics();
    let s = store(&tv, &m);
    let h = hash(0x82);
    put(&s, h, 64).await;
    put(&s, h, 64).await;
    release(&s, h, OBJ).await;
    let mem = member(h, OBJ);
    const FENCE: u64 = 300;

    let claim_ms = claim(&tv, &mem).await;
    assert_eq!(
        decrement(&tv, h, &mem, claim_ms, FENCE).await,
        ("decremented".to_owned(), 1, 0)
    );
    assert_eq!(complete(&tv, &mem).await, 1);
    // U = 0: the next pass tears the record down.
    let p = pass(&s, &GcPolicy::IMMEDIATE).await;
    assert_eq!(p.torn_down, 1);
    assert_eq!(
        record_of(&tv, h, &mem).await,
        None,
        "precondition: torn down"
    );

    tokio::time::sleep(Duration::from_millis(FENCE + 200)).await;
    let late = decrement(&tv, h, &mem, claim_ms, FENCE).await;
    assert_eq!(late.0, "stale", "the late copy must be refused");
    assert_eq!(
        ref_count(&tv, h).await,
        Some(1),
        "the other holder's reference survives"
    );
    assert_eq!(record_of(&tv, h, &mem).await, None);
}

/// Maintenance connections of the GC store in
/// [`a_pass_whose_decrement_outlives_the_fence_leaves_the_member_inflight`]:
/// more than the commands a one-member pass sends before its decrement
/// (selection, claim), so fred's round-robin pool never wraps the decrement
/// onto a connection the test has already released.
const STALE_PASS_MAINTENANCE_POOL: u32 = 8;

/// A pass whose decrement executes more than the claim-age fence after its
/// claim — a store stall between the two — is refused `stale` by the
/// decrement script, and the pass leaves the member where a later pass re-claims
/// it: still in `inflight`, its ledger bytes still counted, NOT completed
/// (no settle entry, no replay record) and nothing decremented. It counts
/// `gc_decrement_total{outcome="stale"}` and `GcBatchResult::stale`.
/// Completing a stale member would settle a release whose decrement never
/// applied: a silent leak of the blob.
///
/// The stall is real: every maintenance connection of the GC store's own
/// client is held in a server-side `BLPOP` ([`ConnStall`]); each command the
/// pass queues behind a held connection is released at once, except the one
/// queued after the claim landed — the decrement — which is held past the
/// fence first. The fence is shortened to 100 ms
/// (`with_decrement_fence_for_test`), below the policy's 30 s re-claim age as in
/// production (15 s against 30 s), so no ordering production cannot reach is
/// exercised.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_pass_whose_decrement_outlives_the_fence_leaves_the_member_inflight() {
    use live_edit::storage::test_support::{connect_test_client, stall_pool, ConnStall};
    const FENCE: Duration = Duration::from_millis(100);
    let tv = TestValkey::setup().await;
    let m = metrics();
    let policy = policy(
        Duration::ZERO,
        Duration::ZERO,
        Duration::from_secs(3600),
        Duration::from_secs(30),
    );
    assert!(FENCE < policy.reclaim_after, "the fence stays below D");
    let gc_client = connect_test_client(&live_edit::config::Config {
        maintenance_pool_size: STALE_PASS_MAINTENANCE_POOL,
        valkey_connect_timeout_secs: 30,
        ..crate::common::test_config()
    })
    .await
    .expect("the GC store's own client must connect");
    let gc = ValkeyExternalValueStore::new(gc_client.clone(), tv.key_schema().clone())
        .with_metrics(Arc::clone(&m))
        .with_decrement_fence_for_test(FENCE);
    let h = hash(0x83);
    let s = store(&tv, &m);
    put(&s, h, 64).await;
    put(&s, h, 64).await;
    release(&s, h, OBJ).await;
    let mem = member(h, OBJ);

    let observer = tv.client().pool.next().clone();
    let mut stalls: Vec<Option<ConnStall>> = stall_pool(
        &gc_client.maintenance_pool,
        &observer,
        &format!("{}conn-stall:gc-stale:", tv.prefix()),
    )
    .await
    .into_iter()
    .map(Some)
    .collect();
    let mut baseline = Vec::new();
    for stall in stalls.iter().flatten() {
        baseline.push(stall.net_in().await.expect("the stalled connection exists"));
    }

    let pass_task = tokio::spawn(async move {
        gc.process_pending(&policy, &mut GcPassState::new())
            .await
            .expect("a pass whose decrement is refused stale must not fail")
    });
    let deadline = tokio::time::Instant::now() + Duration::from_secs(20);
    let mut held_the_decrement = false;
    while !held_the_decrement {
        for (i, slot) in stalls.iter_mut().enumerate() {
            let Some(stall) = slot.as_ref() else {
                continue;
            };
            if stall.net_in().await.expect("the stalled connection exists") == baseline[i] {
                continue;
            }
            // A pass command waits behind this connection. Once the claim has
            // landed, the only command the pass sends before completing the
            // member is its decrement: hold it past the fence.
            if in_inflight(&tv, &mem).await {
                tokio::time::sleep(FENCE * 4).await;
                held_the_decrement = true;
            }
            let released = slot.take().expect("checked above").release().await;
            assert!(
                matches!(&released, Ok(v) if !v.is_null()),
                "the stall must have held until its release; got {released:?}"
            );
            if held_the_decrement {
                break;
            }
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "the pass never queued its decrement behind a held connection"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    for stall in stalls.into_iter().flatten() {
        stall.release().await.expect("the stall must release");
    }
    let p = tokio::time::timeout(Duration::from_secs(20), pass_task)
        .await
        .expect("the pass must finish once released")
        .expect("the pass must not panic");

    assert_eq!(
        (p.processed, p.stale, p.replayed, p.deleted),
        (1, 1, 0, 0),
        "the claimed member's decrement was refused stale"
    );
    assert!(!p.more, "a stale member is not progress");
    assert_eq!(decrements(&m, "stale"), 1);
    assert_eq!(
        (decrements(&m, "applied"), decrements(&m, "replayed")),
        (0, 0)
    );
    assert!(
        in_inflight(&tv, &mem).await,
        "a stale member stays in inflight, to be re-claimed in place"
    );
    assert!(!in_pending(&tv, &mem).await);
    assert_eq!(
        settled_of(&tv, &mem).await,
        None,
        "a stale member is not completed: completing it would settle a release whose decrement \
         never applied"
    );
    assert_eq!(
        bytes_of(&tv, &mem).await,
        Some(64),
        "its bytes stay counted"
    );
    assert_eq!(total(&tv).await, 64);
    assert_eq!(ref_count(&tv, h).await, Some(2), "nothing decremented");
    assert_eq!(record_of(&tv, h, &mem).await, None, "nothing recorded");
}

// ═════════════════════════════════════════════════════════════════════════════
// H6 — teardown through the settle carrier
// ═════════════════════════════════════════════════════════════════════════════

/// A settled member's replay record survives the retention U and is torn
/// down after it, with its settle entry.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_settled_record_is_torn_down_after_the_retention() {
    let tv = TestValkey::setup().await;
    let m = metrics();
    let s = store(&tv, &m);
    let h = hash(0x91);
    put(&s, h, 64).await;
    put(&s, h, 64).await;
    release(&s, h, OBJ).await;
    let mem = member(h, OBJ);
    let retention = Duration::from_millis(800);
    let policy = policy(
        Duration::ZERO,
        Duration::ZERO,
        retention,
        Duration::from_secs(30),
    );

    let first = pass(&s, &policy).await;
    assert_eq!(first.processed, 1);
    assert!(record_of(&tv, h, &mem).await.is_some());
    assert!(settled_of(&tv, &mem).await.is_some());
    let early = pass(&s, &policy).await;
    assert_eq!(early.torn_down, 0, "inside U the record is kept");
    assert!(record_of(&tv, h, &mem).await.is_some());

    tokio::time::sleep(retention + Duration::from_millis(300)).await;
    let after = pass(&s, &policy).await;
    assert_eq!(after.torn_down, 1);
    assert_eq!(record_of(&tv, h, &mem).await, None);
    assert_eq!(settled_of(&tv, &mem).await, None);
    let applied_left: bool = tv
        .client()
        .pool
        .exists(tv.key_schema().ext_applied(&sha256_to_hex(&h)))
        .await
        .unwrap();
    assert!(
        !applied_left,
        "the record key vanishes with its last member"
    );
}

/// `gc_unmark.lua` removes a record only while its score is at or below the
/// settle time it is given: a record newer than the settle belongs to a newer
/// lifecycle and is kept.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unmark_keeps_a_record_newer_than_the_settle() {
    let tv = TestValkey::setup().await;
    let h = hash(0x92);
    let mem = member(h, OBJ);
    let applied = tv.key_schema().ext_applied(&sha256_to_hex(&h));
    let _: () = tv
        .client()
        .pool
        .zadd(&applied, None, None, false, false, (5_000.0, mem.as_str()))
        .await
        .unwrap();
    let unmark = |settled_ms: u64| {
        let applied = applied.clone();
        let mem = mem.clone();
        let tv = &tv;
        async move {
            GC_UNMARK_SCRIPT
                .evalsha_with_reload::<i64, _, _>(
                    tv.client().maintenance_pool.next(),
                    vec![applied],
                    vec![mem, settled_ms.to_string()],
                )
                .await
                .expect("gc_unmark must run")
        }
    };
    assert_eq!(unmark(4_999).await, 0, "a newer record is kept");
    assert_eq!(record_of(&tv, h, &mem).await, Some(5_000.0));
    assert_eq!(
        unmark(5_000).await,
        1,
        "a record no newer than the settle goes"
    );
    assert_eq!(record_of(&tv, h, &mem).await, None);
    assert_eq!(unmark(5_000).await, 0, "idempotent");
}

/// `gc_settled_retire.lua` removes a settle entry only while its score still
/// equals the one read: a member that settled again keeps its newer entry.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn settled_retire_removes_an_entry_only_while_its_score_matches() {
    let tv = TestValkey::setup().await;
    let settled = tv.key_schema().ext_settled();
    let mem = member(hash(0x93), OBJ);
    let _: () = tv
        .client()
        .pool
        .zadd(&settled, None, None, false, false, (7_000.0, mem.as_str()))
        .await
        .unwrap();
    let retire = |score: &str| {
        let settled = settled.clone();
        let argv = vec![mem.clone(), score.to_owned()];
        let tv = &tv;
        async move {
            GC_SETTLED_RETIRE_SCRIPT
                .evalsha_with_reload::<i64, _, _>(
                    tv.client().maintenance_pool.next(),
                    vec![settled],
                    argv,
                )
                .await
                .expect("gc_settled_retire must run")
        }
    };
    assert_eq!(retire("6000").await, 0, "a stale score retires nothing");
    assert_eq!(settled_of(&tv, &mem).await, Some(7_000.0));
    assert_eq!(retire("7000").await, 1);
    assert_eq!(settled_of(&tv, &mem).await, None);
}

/// A settled member that is back in `pending` (enqueued again) is skipped by
/// the teardown: its record must answer the new copy's replay. Since b.1jt
/// this build's own enqueue refuses a settled member, so only an older
/// build's bare `SADD` puts one back.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_settled_member_back_in_pending_is_not_torn_down() {
    let tv = TestValkey::setup().await;
    let m = metrics();
    let s = store(&tv, &m);
    let h = hash(0x94);
    put(&s, h, 64).await;
    put(&s, h, 64).await;
    release(&s, h, OBJ).await;
    let mem = member(h, OBJ);
    let first = pass(&s, &immediate_keeping_records()).await;
    assert_eq!(first.processed, 1);
    // The identical member again, from an older build's bare `SADD`.
    let _: () = tv
        .client()
        .pool
        .sadd(tv.key_schema().ext_pending(), mem.as_str())
        .await
        .expect("bare SADD");
    assert!(in_pending(&tv, &mem).await);

    // U = 0 makes the settle due; the hour-old floor keeps the new copy
    // pending through the pass.
    let p = pass(
        &s,
        &policy(
            Duration::from_secs(3600),
            Duration::from_secs(3600),
            Duration::ZERO,
            Duration::from_secs(30),
        ),
    )
    .await;
    assert_eq!(p.torn_down, 0, "a member back in pending is not torn down");
    assert!(record_of(&tv, h, &mem).await.is_some());

    // And the new copy is answered as the replay it is.
    let replay = pass(&s, &immediate_keeping_records()).await;
    assert_eq!((replay.processed, replay.replayed), (1, 1));
    assert_eq!(ref_count(&tv, h).await, Some(1));
}

/// After teardown an identical member is a new lifecycle: it decrements
/// normally (policy 8's window has passed).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_identical_member_after_teardown_decrements_normally() {
    let tv = TestValkey::setup().await;
    let m = metrics();
    let s = store(&tv, &m);
    let h = hash(0x95);
    put(&s, h, 64).await;
    put(&s, h, 64).await;
    release(&s, h, OBJ).await;
    let mem = member(h, OBJ);
    let stats = process_pending_gc(&s, Some(&m), GcPolicy::IMMEDIATE).await;
    assert_eq!((stats.processed, stats.torn_down), (1, 1));
    assert_eq!(record_of(&tv, h, &mem).await, None);

    release(&s, h, OBJ).await;
    let again = process_pending_gc(&s, Some(&m), GcPolicy::IMMEDIATE).await;
    assert_eq!((again.processed, again.replayed, again.deleted), (1, 0, 1));
    assert_eq!(ref_count(&tv, h).await, None);
}

// ═════════════════════════════════════════════════════════════════════════════
// H8 — an older build's leftovers
// ═════════════════════════════════════════════════════════════════════════════

/// Plant an orphan an older build leaves when it completes a member this
/// build decremented with a bare `ZREM inflight`: the member in neither set,
/// its ledger bytes counted in the total, and its replay record. With
/// `aged`, it also keeps an age entry (an older build's claim never removes
/// it).
async fn plant_orphan(tv: &TestValkey, h: [u8; 32], m: &str, aged: bool) {
    let keys = tv.key_schema();
    let pool = &tv.client().pool;
    let _: () = pool
        .hset(keys.ext_pending_bytes(), [(m, "1000")])
        .await
        .unwrap();
    let _: () = pool
        .set(keys.ext_pending_total(), "1000", None, None, false)
        .await
        .unwrap();
    let _: () = pool
        .zadd(
            keys.ext_applied(&sha256_to_hex(&h)),
            None,
            None,
            false,
            false,
            (1.0, m),
        )
        .await
        .unwrap();
    if aged {
        let _: () = pool
            .zadd(keys.ext_pending_age(), None, None, false, false, (1.0, m))
            .await
            .unwrap();
    }
}

/// The claim's self-heal settles an orphan the selection reaches through its
/// leftover age entry: its age and bytes go, the total is corrected, and its
/// record is torn down after U. The passes share one loop state whose audit
/// cycle already ran (on an empty ledger) and is not due again, so the
/// self-heal is the only repair that can act.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_claim_self_heals_an_orphan_with_an_age_entry() {
    let tv = TestValkey::setup().await;
    let m = metrics();
    let s = store(&tv, &m);
    let h = hash(0xA1);
    let mem = member(h, OBJ);
    let retention = Duration::from_millis(400);
    let policy = policy(
        Duration::ZERO,
        Duration::ZERO,
        retention,
        Duration::from_secs(30),
    );
    let mut state = GcPassState::new();
    let first = s.process_pending(&policy, &mut state).await.unwrap();
    assert_eq!(
        first.audit_repaired, 0,
        "the audit cycle ran on an empty ledger"
    );
    plant_orphan(&tv, h, &mem, true).await;

    let healed = s.process_pending(&policy, &mut state).await.unwrap();
    assert_eq!(healed.audit_repaired, 0, "the audit is not due again");
    assert_eq!(
        healed.processed, 0,
        "nothing was claimed: the member is in no set"
    );
    assert_eq!(age_of(&tv, &mem).await, None);
    assert_eq!(bytes_of(&tv, &mem).await, None);
    assert_eq!(total(&tv).await, 0, "the orphan's bytes leave the total");
    assert!(settled_of(&tv, &mem).await.is_some(), "the orphan settled");

    tokio::time::sleep(retention + Duration::from_millis(300)).await;
    let later = pass(&s, &policy).await;
    assert_eq!(later.torn_down, 1);
    assert_eq!(
        record_of(&tv, h, &mem).await,
        None,
        "its record is torn down"
    );
}

/// The ledger audit settles an orphan with no age entry — one the selection
/// cannot reach — on the first pass of a fresh leadership, and its record is
/// torn down after U.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_audit_settles_an_orphan_without_an_age_entry() {
    let tv = TestValkey::setup().await;
    let m = metrics();
    let s = store(&tv, &m);
    let h = hash(0xA2);
    let mem = member(h, OBJ);
    plant_orphan(&tv, h, &mem, false).await;
    let retention = Duration::from_millis(400);
    let policy = policy(
        Duration::ZERO,
        Duration::ZERO,
        retention,
        Duration::from_secs(30),
    );

    let mut state = GcPassState::new();
    let audited = s.process_pending(&policy, &mut state).await.unwrap();
    assert_eq!(audited.audit_repaired, 1);
    assert_eq!(bytes_of(&tv, &mem).await, None);
    assert_eq!(total(&tv).await, 0);
    assert!(settled_of(&tv, &mem).await.is_some());

    tokio::time::sleep(retention + Duration::from_millis(300)).await;
    let later = s.process_pending(&policy, &mut state).await.unwrap();
    assert_eq!(later.torn_down, 1);
    assert_eq!(record_of(&tv, h, &mem).await, None);
}

// ═════════════════════════════════════════════════════════════════════════════
// H9 — a member whose own keys fail
// ═════════════════════════════════════════════════════════════════════════════

/// Plant a string where `h`'s replay record lives, so its decrement fails
/// with a non-transient `WRONGTYPE`.
async fn poison_record(tv: &TestValkey, h: [u8; 32]) {
    let _: () = tv
        .client()
        .pool
        .set(
            tv.key_schema().ext_applied(&sha256_to_hex(&h)),
            "foreign",
            None,
            None,
            false,
        )
        .await
        .unwrap();
}

/// A wrong-typed value at one member's `le:ext:applied:{h}` fails only that
/// member: it stays in `inflight`, the error is counted, and the rest of the
/// pass collects. Ending the pass instead would let one bad member, selected
/// oldest first, stop collection for good.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_wrong_typed_replay_record_fails_only_its_member() {
    let tv = TestValkey::setup().await;
    let m = metrics();
    let s = store(&tv, &m);
    let (bad, good) = (hash(0xB1), hash(0xB2));
    for h in [bad, good] {
        put(&s, h, 64).await;
        release(&s, h, OBJ).await;
    }
    poison_record(&tv, bad).await;

    let p = pass(&s, &GcPolicy::IMMEDIATE).await;
    assert_eq!(
        (p.processed, p.deleted),
        (2, 1),
        "the good member is collected"
    );
    assert!(!data_present(&tv, good).await);
    assert_eq!(ref_count(&tv, bad).await, Some(1));
    assert!(
        in_inflight(&tv, &member(bad, OBJ)).await,
        "the bad member waits in inflight for its re-claim"
    );
    assert_eq!(
        batch_errors(&m, "internal"),
        1,
        "the member's error is counted"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// The pass-again (`more`) rule
// ═════════════════════════════════════════════════════════════════════════════

/// The settled-batch and member-batch size the pass's `more` rule keys on
/// (`GC_SETTLED_BATCH_SIZE`, `GC_BATCH_SIZE`).
const FULL_BATCH: usize = 100;

/// A full batch of due settle entries whose unmarks all fail tears nothing
/// down, and the pass does NOT ask to run again at once — while a shard is
/// down the loop must sleep, not spin on the same pairs. The control: the
/// same full batch with healthy records is torn down and does ask.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_full_batch_of_failing_unmarks_does_not_run_the_next_pass_at_once() {
    let tv = TestValkey::setup().await;
    let m = metrics();
    let s = store(&tv, &m);
    let quiet = policy(
        Duration::from_secs(3600),
        Duration::from_secs(3600),
        Duration::ZERO,
        Duration::from_secs(30),
    );
    let settled = tv.key_schema().ext_settled();
    for i in 0..FULL_BATCH {
        let h = hash_n(i);
        let _: () = tv
            .client()
            .pool
            .zadd(
                &settled,
                None,
                None,
                false,
                false,
                (1.0, member(h, OBJ).as_str()),
            )
            .await
            .unwrap();
        poison_record(&tv, h).await;
    }

    let failing = pass(&s, &quiet).await;
    assert_eq!(failing.torn_down, 0, "every unmark failed");
    assert!(
        !failing.more,
        "a pass whose unmarks all failed must sleep like an idle one"
    );
    assert_eq!(
        zcard(&tv, settled.clone()).await,
        FULL_BATCH as u64,
        "the pairs wait"
    );

    for i in 0..FULL_BATCH {
        let _: i64 = tv
            .client()
            .pool
            .del(tv.key_schema().ext_applied(&sha256_to_hex(&hash_n(i))))
            .await
            .unwrap();
    }
    let healthy = pass(&s, &quiet).await;
    assert_eq!(healthy.torn_down, FULL_BATCH);
    assert!(
        healthy.more,
        "a full batch actually torn down runs the next pass at once"
    );
}

/// A full batch of members whose decrements all fail (`Left`) collects
/// nothing, and the pass does NOT ask to run again at once. The control: a
/// full batch that collects does.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_full_batch_of_failing_decrements_does_not_run_the_next_pass_at_once() {
    let tv = TestValkey::setup().await;
    let m = metrics();
    let s = store(&tv, &m);
    for i in 0..FULL_BATCH {
        let h = hash_n(i);
        put(&s, h, 16).await;
        release(&s, h, OBJ).await;
        poison_record(&tv, h).await;
    }
    let failing = pass(&s, &GcPolicy::IMMEDIATE).await;
    assert_eq!((failing.processed, failing.deleted), (FULL_BATCH, 0));
    assert!(
        !failing.more,
        "a full batch whose decrements all failed must sleep like an idle one"
    );

    for i in 0..FULL_BATCH {
        let h = hash_n(1_000 + i);
        put(&s, h, 16).await;
        release(&s, h, OBJ_B).await;
    }
    let healthy = pass(&s, &GcPolicy::IMMEDIATE).await;
    assert_eq!(healthy.deleted, FULL_BATCH);
    assert!(
        healthy.more,
        "a full batch that collected runs the next pass at once"
    );
}

/// A distinct 32-byte hash per index.
fn hash_n(i: usize) -> [u8; 32] {
    let mut h = [0xC0; 32];
    h[..8].copy_from_slice(&(i as u64).to_be_bytes());
    h
}

// ═════════════════════════════════════════════════════════════════════════════
// Test 9 — the gauges are leader-published
// ═════════════════════════════════════════════════════════════════════════════

fn gauge(metrics: &Metrics, name: &str) -> f64 {
    sample_value(&render(metrics), name, &[]).unwrap_or(0.0)
}

/// The leader publishes the ledger's levels each pass; a non-leader publishes
/// zeros, so a demoted leader does not leave its last level standing in a
/// fleet-wide `max()`.
///
/// The ledger holds one member of each kind the gauges count: a pending
/// member (3 000 bytes, aged), an inflight one (500 bytes, claimed and not
/// completed) and a settled one (its bytes already out of the total). So the
/// leader publishes 3 500 bytes, a count of 2 — pending plus inflight — a
/// settled count of 1 and an oldest age above zero; the non-leader zeros all
/// four.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn only_the_leader_publishes_the_ledger_gauges() {
    let tv = TestValkey::setup().await;
    let m = metrics();
    let s: Arc<dyn ExternalValueStore> = Arc::new(store(&tv, &m));
    let (h, inflight, settled) = (hash(0xD1), hash(0xD2), hash(0xD3));
    for (hh, len) in [(h, 3_000), (inflight, 500), (settled, 700)] {
        put(&store(&tv, &m), hh, len).await;
        release(&store(&tv, &m), hh, OBJ).await;
    }
    let claimed = claim(&tv, &member(inflight, OBJ)).await;
    assert!(claimed > 0, "precondition: the inflight member is claimed");
    let done = member(settled, OBJ);
    let claim_ms = claim(&tv, &done).await;
    assert_eq!(
        decrement(&tv, settled, &done, claim_ms, GC_DECREMENT_FENCE_MS)
            .await
            .0,
        "deleted"
    );
    assert_eq!(complete(&tv, &done).await, 1, "precondition: settled");
    // The pending member's age is measured in whole milliseconds.
    tokio::time::sleep(Duration::from_millis(20)).await;
    // Nothing is due (an hour's floor), nothing is re-claimed (30 s), and the
    // settled member's record outlives the run (an hour's retention).
    let hold = policy(
        Duration::from_secs(3600),
        Duration::from_secs(3600),
        Duration::from_secs(3600),
        Duration::from_secs(30),
    );

    let run = |leader: bool| {
        let cancel = CancellationToken::new();
        let handle = spawn_gc_task_with_leader(
            Arc::clone(&s),
            hold,
            cancel.clone(),
            Some(Arc::clone(&m)),
            move || async move { leader },
        );
        (cancel, handle)
    };

    let (cancel, handle) = run(true);
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while gauge(&m, "live_edit_ext_release_pending_bytes") != 3_500.0 {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the leader never published"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    cancel.cancel();
    handle.await.unwrap();
    assert_eq!(
        gauge(&m, "live_edit_ext_release_pending_count"),
        2.0,
        "the count covers pending AND inflight"
    );
    assert_eq!(gauge(&m, "live_edit_ext_release_settled_count"), 1.0);
    let oldest = gauge(&m, "live_edit_ext_release_oldest_age_seconds");
    assert!(
        oldest > 0.0,
        "the leader publishes the oldest pending member's age; got {oldest}"
    );

    let (cancel, handle) = run(false);
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while gauge(&m, "live_edit_ext_release_pending_bytes") != 0.0 {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the non-leader never zeroed"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    cancel.cancel();
    handle.await.unwrap();
    for name in [
        "live_edit_ext_release_pending_count",
        "live_edit_ext_release_settled_count",
        "live_edit_ext_release_oldest_age_seconds",
    ] {
        assert_eq!(gauge(&m, name), 0.0, "the non-leader zeros {name}");
    }
    assert_eq!(
        total(&tv).await,
        3_500,
        "the store still holds the releases"
    );
    assert!(
        in_inflight(&tv, &member(inflight, OBJ)).await && settled_of(&tv, &done).await.is_some(),
        "precondition held throughout: one inflight member, one settled"
    );
}
