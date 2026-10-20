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

//! b.1jt — a store shared with an older build: rolling upgrades and
//! rollbacks across the GC protocol marker, against a real Valkey.
//!
//! Every build before the marker (0.1.0 through 0.2.0-rc.48) runs a GC this
//! build cannot change, and all of them share four traits: an unmarked leader
//! lock taken by an equality CAS, a claim scored with the bare claim time, a
//! drain that moves `[0, now − 30 s]` claims back to `pending`, and SET
//! retirement carriers. 0.1.0 through 0.2.0-rc.47 also decrement record-blind
//! (no replay record), complete a member with a bare `ZREM`, and drain only at
//! startup; 0.2.0-rc.48 has the replay record, `gc_complete.lua` and a drain
//! every pass. The record-blind builds are the dangerous ones, and their
//! scripts are vendored verbatim under `tests/fixtures/legacy_gc_rc47/` (0.1.0
//! and 0.1.1 differ only in headers and comments). The tests here run them
//! against the store as an older replica would, beside this build's own code,
//! and check that the state this build leaves is safe: no second decrement, a
//! `ref_count` that errs high (a leak the reconciler repairs, permanent if it
//! is turned off) and never low.
//!
//! | Design test | Tests |
//! |---|---|
//! | T1 — the marked lock | [`an_unmarked_gc_lock_is_taken_over_in_both_older_spellings`], [`a_lower_or_malformed_marked_version_is_taken_over_as_older`], [`a_same_version_peer_is_kept_above_the_threshold_and_taken_at_or_below_it`], [`a_second_acquire_by_the_holder_is_refreshed_and_restores_the_ttl`], [`a_higher_version_holder_is_kept_at_any_ttl_unless_it_has_none`], [`a_lower_version_acquires_the_key_once_a_higher_version_holders_key_lapses`], [`the_older_builds_cas_never_matches_a_marked_lock_and_writes_nothing`], [`two_marked_loops_hand_the_lock_over_without_it_ever_lapsing`], [`the_gc_task_warns_and_counts_only_a_takeover_from_an_older_build`], [`a_leaders_progressing_pass_past_the_ttl_and_the_stall_bound_never_lets_the_key_lapse`], [`a_gc_loop_that_panics_stops_refreshing_the_lock_and_its_handle_completes`], [`a_wedged_gc_loop_stops_renewing_the_lock_and_a_peer_takes_it_over`], [`a_failing_refresh_logs_one_warn_counts_every_failure_and_one_info_on_recovery`], [`a_slow_attempt_audit_page_marks_progress_and_keeps_the_lock`], [`a_slow_teardown_and_member_step_mark_progress_and_keep_the_lock`] |
//! | T2 — after a takeover | [`an_older_drain_and_sweep_leave_this_builds_decremented_claim_alone`] |
//! | T3 — a rollback | [`a_rollback_decrements_nothing_this_build_claimed_and_this_build_finishes_it_once`], [`an_older_carrier_claim_or_deferral_fails_on_this_builds_carrier_and_writes_nothing`] |
//! | T4 — un-aged members | [`an_unaged_member_is_stamped_first_and_collected_only_on_a_later_pass`], [`a_stamped_member_waits_out_the_age_floor_from_its_stamp`], [`a_hold_started_within_the_margin_after_the_stamp_protects_a_stamped_member`] |
//! | T5 — an older build's claims | [`an_older_builds_claim_past_the_fence_is_dropped_never_decremented`], [`the_foreign_claim_drop_writes_nothing_for_a_pair_any_guard_refuses`] |
//! | T6 — this build's stale claims | [`a_stale_own_claim_is_reclaimed_in_place_and_never_written_to_pending`], [`of_two_concurrent_reclaims_of_one_claim_exactly_one_wins`] |
//! | T7 — the enqueue's replay refusal | [`the_enqueue_refuses_a_member_claimed_settled_or_accounted_and_writes_nothing`], [`an_all_refused_attempt_enqueues_nothing_and_still_records_the_attempt_landed`], [`a_resent_enqueue_after_an_older_sweep_collected_the_member_is_refused`], [`a_resent_enqueue_of_a_settled_member_succeeds_and_enqueues_nothing`] |
//! | T8 — the ledger after an older leader | [`a_takeover_corrects_the_total_an_older_leaders_bare_completions_left`] |
//! | Guard (source scan) | [`only_the_release_enqueue_adds_to_pending_and_only_the_claim_scripts_score_inflight`], [`the_write_scanner_classifies_keys_by_their_local_and_skips_comments`], [`the_rust_write_scanner_sees_the_turbofish_form`] |
//!
//! Elsewhere: an older build's SET carrier (T9) in `src/storage/orphan_sweep.rs`'s
//! b.1jt section; the reconciler's owed walk over both carrier types and both
//! claim encodings, and its forfeit on a carrier of any other type (T10), in
//! `src/storage/ext_reconcile.rs`; the release
//! budget's logging (T11) beside `Config` (`src/config.rs`) and in
//! `b_yuo_store_memory_tests.rs`; its derivation (T12) beside `Config`; the
//! claim-score offset literal in `src/storage/external_values.rs`; and the GC
//! state machine with re-claims, foreign drops and stamping in
//! `tests/proptests.rs`.
//!
//! **What fails on the build before b.1jt (c9463288).** That build's own
//! scripts and API differ (no marked lock, no re-claim, no drop, no stamp, a
//! four-key enqueue), so these tests do not compile against it; each
//! behavioural test's doc says which of its assertions the old behaviour
//! breaks, and why.
//!
//! The lock tests drive `acquire_gc_leader_lock` with the production TTL and
//! threshold except where a hand-over between two loops must happen inside
//! the test: there both are scaled down together, keeping production's ratio
//! of the loop's 1 s call interval to the 10 s takeover threshold. The tests
//! of the GC task itself (`spawn_gc_task`, with its refresh ticker) run its
//! production constants. The ticker renews the lock once a second only while
//! the GC loop keeps marking progress (`GC_PROGRESS_STALL_AFTER`, 20 s), so
//! the test of a progressing pass that outlives the TTL plus that bound runs
//! for about 57 s, the test of a wedged pass, which loses the lock about 40 s
//! in, for about 45 s, and the tests of single slow steps, which must each run
//! past the bound, for about 30 s (an audit page) and 55 s (a teardown and a
//! member step). Their slowness comes from a proxy that holds chosen store
//! calls a few hundred ms each, well under the command timeout.

use std::collections::HashMap;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::Duration;

use fred::interfaces::{
    ClientLike, HashesInterface, KeysInterface, SetsInterface, SortedSetsInterface,
};
use fred::types::scripts::Script;
use fred::types::{ClusterHash, CustomCommand, Expiration, Value};
use tokio_util::sync::CancellationToken;
use tracing_subscriber::fmt::MakeWriter;

use live_edit::error::ServiceError;
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::Metrics;
use live_edit::storage::ext_release_gate::ExtReleaseGate;
use live_edit::storage::external_values::{
    acquire_gc_leader_lock, process_pending_gc, spawn_gc_task, ExtRelease, ExtValueProbe,
    ExtValueVerify, ExternalValueStore, GcBatchResult, GcLeaderAcquire, GcPassState, GcPolicy,
    ReleaseAttemptOutcome, ValkeyExternalValueStore, EXT_RELEASE_ENQUEUE_SCRIPT,
    GC_CLAIM_PENDING_SCRIPT, GC_CLAIM_SCORE_OFFSET_MS, GC_DECREMENT_AND_COLLECT_SCRIPT,
    GC_DECREMENT_FENCE_MS, GC_DROP_FOREIGN_CLAIMS_SCRIPT, GC_LEADER_TAKEOVER_BELOW_MS,
    GC_PROGRESS_STALL_AFTER, GC_RECLAIM_STALE_SCRIPT, GC_UNMARK_SCRIPT, LEADER_LOCK_TTL_MS,
};
use live_edit::storage::keys::sha256_to_hex;
use live_edit::storage::layer::schedule_old_object_ext_gc_for_test;
use live_edit::storage::leader_lease::{
    LeaderIdentity, GC_LEADER_MARKER_PREFIX, GC_LEADER_PROTOCOL_VERSION,
};
use live_edit::storage::test_support::TestValkey;

use crate::storage::lua_source_scan::{compiled_scripts, lua_code};
use crate::storage::snapshot_read_support::{
    client_through, error_reply_script, monitor_fence, test_valkey_addr, MonitorTap, ProxyAction,
    ProxyRule, ScriptReplyProxy,
};

// ═════════════════════════════════════════════════════════════════════════════
// The older build's scripts (fixtures)
// ═════════════════════════════════════════════════════════════════════════════

/// The older build's claim: `SREM pending`, then `ZADD inflight <now_ms>` —
/// a bare score. KEYS `[pending, inflight]`, ARGV `[member]`; 1 or 0.
static LEGACY_CLAIM: LazyLock<Script> = LazyLock::new(|| {
    Script::from_lua(include_str!(
        "../../fixtures/legacy_gc_rc47/gc_claim_pending.lua"
    ))
});

/// The older build's record-blind decrement: `HINCRBY ref_count -1`, delete
/// at zero. KEYS `[meta, data]`; `{ action, ref_count }`.
static LEGACY_DECREMENT: LazyLock<Script> = LazyLock::new(|| {
    Script::from_lua(include_str!(
        "../../fixtures/legacy_gc_rc47/gc_decrement_and_collect.lua"
    ))
});

/// The older build's startup drain: every claim scored in `[0, now − ttl]`
/// back to `pending`. KEYS `[inflight, pending]`, ARGV `[ttl_ms]`;
/// `{ drained, inflight_left }`.
static LEGACY_DRAIN: LazyLock<Script> = LazyLock::new(|| {
    Script::from_lua(include_str!(
        "../../fixtures/legacy_gc_rc47/drain_inflight_to_pending.lua"
    ))
});

/// The older build's destructive carrier claim (`SCARD` / `SMEMBERS`, then
/// `DEL`). KEYS `[carrier, header, index]`; `{ verdict, count, members }`.
static LEGACY_CARRIER_CLAIM: LazyLock<Script> = LazyLock::new(|| {
    Script::from_lua(include_str!(
        "../../fixtures/legacy_gc_rc47/ext_retire_retry_claim.lua"
    ))
});

/// The older build's retirement deferral (`SADD carrier`, flag the header).
/// KEYS `[carrier, header]`, ARGV members; `{ added, flagged }`.
static LEGACY_DEFER: LazyLock<Script> = LazyLock::new(|| {
    Script::from_lua(include_str!(
        "../../fixtures/legacy_gc_rc47/ext_retire_defer.lua"
    ))
});

/// The older build's leader-lock CAS: claim an absent key, refresh an equal
/// one, else 0. KEYS `[lock]`, ARGV `[value, ttl_ms]`.
static LEGACY_LOCK_CAS: LazyLock<Script> = LazyLock::new(|| {
    Script::from_lua(include_str!(
        "../../fixtures/legacy_gc_rc47/leader_acquire_or_refresh.lua"
    ))
});

/// The lock value an older build writes (`service_label:instance`; 0.1.0
/// wrote the bare label).
const LEGACY_LOCK_VALUE: &str = "older-replica:6f1c0d9e";

async fn legacy_claim(tv: &TestValkey, m: &str) -> i64 {
    let keys = tv.key_schema();
    LEGACY_CLAIM
        .evalsha_with_reload::<i64, _, _>(
            tv.client().maintenance_pool.next(),
            vec![keys.ext_pending(), keys.ext_inflight()],
            vec![m.to_owned()],
        )
        .await
        .expect("the older claim must run")
}

async fn legacy_decrement(tv: &TestValkey, h: [u8; 32]) -> (String, i64) {
    let hex = sha256_to_hex(&h);
    let keys = tv.key_schema();
    LEGACY_DECREMENT
        .evalsha_with_reload::<(String, i64), _, _>(
            tv.client().maintenance_pool.next(),
            vec![keys.ext_meta(&hex), keys.ext_data(&hex)],
            Vec::<String>::new(),
        )
        .await
        .expect("the older decrement must run")
}

/// The older build's startup drain with drain age `ttl_ms` (production
/// passes 30 000; 0 is the widest it can ever reach). Returns how many
/// claims it moved back to `pending`.
async fn legacy_drain(tv: &TestValkey, ttl_ms: u64) -> i64 {
    let keys = tv.key_schema();
    let (drained, _left): (i64, i64) = LEGACY_DRAIN
        .evalsha_with_reload(
            tv.client().maintenance_pool.next(),
            vec![keys.ext_inflight(), keys.ext_pending()],
            vec![ttl_ms.to_string()],
        )
        .await
        .expect("the older drain must run");
    drained
}

/// One pass of the older build's GC sweep, as its loop runs one: `SRANDMEMBER
/// pending`, then per member the claim, the record-blind decrement and a bare
/// `ZREM inflight` — no ledger, no replay record, no read hold. Returns the
/// members it decremented.
async fn legacy_gc_pass(tv: &TestValkey) -> Vec<String> {
    let keys = tv.key_schema();
    let sample: Vec<String> = tv
        .client()
        .pool
        .srandmember(keys.ext_pending(), Some(100))
        .await
        .expect("SRANDMEMBER pending");
    let mut decremented = Vec::new();
    for m in sample {
        if legacy_claim(tv, &m).await == 0 {
            continue;
        }
        let hex = m.split(':').next().expect("a member starts with its hash");
        let h = hash_from_hex(hex);
        legacy_decrement(tv, h).await;
        let _: () = tv
            .client()
            .pool
            .zrem(keys.ext_inflight(), m.as_str())
            .await
            .expect("bare ZREM inflight");
        decremented.push(m);
    }
    decremented
}

fn hash_from_hex(hex: &str) -> [u8; 32] {
    let mut h = [0u8; 32];
    for (i, byte) in h.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).expect("hex");
    }
    h
}

// ═════════════════════════════════════════════════════════════════════════════
// Fixture
// ═════════════════════════════════════════════════════════════════════════════

/// The object every release here names, unless a test needs two.
const OBJ: u64 = 7_900;

fn hash(byte: u8) -> [u8; 32] {
    [byte; 32]
}

fn member(h: [u8; 32], object_id: u64) -> String {
    ExtRelease::retirement(h, object_id).member()
}

fn metrics() -> Arc<Metrics> {
    Arc::new(crate::common::metrics_scrape_only("b-1jt-rollout"))
}

fn store(tv: &TestValkey, metrics: &Arc<Metrics>) -> ValkeyExternalValueStore {
    ValkeyExternalValueStore::new(tv.client().clone(), tv.key_schema().clone())
        .with_metrics(Arc::clone(metrics))
}

/// [`GcPolicy::IMMEDIATE`], except that replay records outlive the test, so a
/// re-run of a decided member meets its record.
fn immediate_keeping_records() -> GcPolicy {
    GcPolicy {
        replay_record_retention: Duration::from_secs(3600),
        ..GcPolicy::IMMEDIATE
    }
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

async fn pass(store: &ValkeyExternalValueStore, policy: &GcPolicy) -> GcBatchResult {
    store
        .process_pending(policy, &mut GcPassState::new())
        .await
        .expect("a GC pass must succeed")
}

/// Valkey's own clock, ms — the clock every GC script reads.
pub(super) async fn valkey_now_ms(tv: &TestValkey) -> u64 {
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

/// How old a claim of this build's must be for the production re-claim age
/// D (30 s, `GcPolicy::reclaim_after`) to reach it, with a second's slack.
pub(super) const STALE_OWN_CLAIM_AGE: Duration = Duration::from_millis(LEADER_LOCK_TTL_MS + 1_000);

/// Put `m` in `inflight` as THIS build scores a claim — the store's clock
/// `claimed_ago` back, plus `GC_CLAIM_SCORE_OFFSET_MS` (b.1jt). With
/// [`STALE_OWN_CLAIM_AGE`] the next pass re-claims it in place: the state a
/// sweeper that crashed, or whose step failed, leaves behind.
pub(super) async fn plant_own_claim(tv: &TestValkey, m: &str, claimed_ago: Duration) {
    let score = valkey_now_ms(tv).await - claimed_ago.as_millis() as u64 + GC_CLAIM_SCORE_OFFSET_MS;
    let _: () = tv
        .client()
        .pool
        .zadd(
            tv.key_schema().ext_inflight(),
            None,
            None,
            false,
            false,
            (score as f64, m),
        )
        .await
        .expect("ZADD inflight (this build's encoded claim)");
}

/// Put `m` in `inflight` as an OLDER build scores a claim: the bare claim
/// time, `claim_ms`.
async fn plant_older_claim(tv: &TestValkey, m: &str, claim_ms: u64) {
    let _: () = tv
        .client()
        .pool
        .zadd(
            tv.key_schema().ext_inflight(),
            None,
            None,
            false,
            false,
            (claim_ms as f64, m),
        )
        .await
        .expect("ZADD inflight (an older build's bare claim)");
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

async fn in_pending(tv: &TestValkey, m: &str) -> bool {
    tv.client()
        .pool
        .sismember(tv.key_schema().ext_pending(), m)
        .await
        .expect("SISMEMBER pending")
}

async fn zscore(tv: &TestValkey, key: String, m: &str) -> Option<f64> {
    tv.client().pool.zscore(key, m).await.expect("ZSCORE")
}

async fn inflight_score(tv: &TestValkey, m: &str) -> Option<f64> {
    zscore(tv, tv.key_schema().ext_inflight(), m).await
}

async fn age_of(tv: &TestValkey, m: &str) -> Option<f64> {
    zscore(tv, tv.key_schema().ext_pending_age(), m).await
}

async fn settled_of(tv: &TestValkey, m: &str) -> Option<f64> {
    zscore(tv, tv.key_schema().ext_settled(), m).await
}

async fn bytes_of(tv: &TestValkey, m: &str) -> Option<i64> {
    tv.client()
        .pool
        .hget(tv.key_schema().ext_pending_bytes(), m)
        .await
        .expect("HGET pending:bytes")
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

/// `live_edit_gc_older_build_total{event}`; zero-seeded, so absence means a
/// different `Metrics`.
fn older_build_events(metrics: &Metrics, event: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_gc_older_build_total",
        &[("event", event)],
    )
    .unwrap_or_else(|| panic!("live_edit_gc_older_build_total{{event=\"{event}\"}} is zero-seeded"))
        as u64
}

/// `live_edit_gc_decrement_total{outcome}`.
fn decrements(metrics: &Metrics, outcome: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_gc_decrement_total",
        &[("outcome", outcome)],
    )
    .unwrap_or_else(|| panic!("live_edit_gc_decrement_total{{outcome=\"{outcome}\"}} is seeded"))
        as u64
}

// ═════════════════════════════════════════════════════════════════════════════
// T1 — the protocol-marked leader lock
// ═════════════════════════════════════════════════════════════════════════════

fn this_build(label: &str) -> LeaderIdentity {
    LeaderIdentity::new(label, "this-build")
}

/// A marked value as a build of protocol `version` writes it.
fn marked(version: u32, label: &str) -> String {
    format!("{GC_LEADER_MARKER_PREFIX}{version}|{label}:peer-instance")
}

async fn plant_lock(tv: &TestValkey, value: &str, px_ms: Option<i64>) {
    let _: () = tv
        .client()
        .pool
        .set(
            tv.key_schema().gc_leader_lock(),
            value,
            px_ms.map(Expiration::PX),
            None,
            false,
        )
        .await
        .expect("SET the planted lock");
}

async fn lock_value(tv: &TestValkey) -> Option<String> {
    tv.client()
        .pool
        .get(tv.key_schema().gc_leader_lock())
        .await
        .expect("GET le:gc:leader")
}

async fn lock_pttl(tv: &TestValkey) -> i64 {
    tv.client()
        .pool
        .pttl(tv.key_schema().gc_leader_lock())
        .await
        .expect("PTTL le:gc:leader")
}

/// One production-shaped acquire (30 s TTL, 10 s threshold) as `who`.
async fn acquire(tv: &TestValkey, who: &LeaderIdentity) -> GcLeaderAcquire {
    acquire_gc_leader_lock(
        tv.client(),
        &tv.key_schema().gc_leader_lock(),
        who,
        LEADER_LOCK_TTL_MS,
        GC_LEADER_TAKEOVER_BELOW_MS,
    )
    .await
    .expect("gc_leader_acquire must run")
}

/// The older build's CAS as `value`, production TTL.
async fn legacy_cas(tv: &TestValkey, value: &str) -> i64 {
    LEGACY_LOCK_CAS
        .evalsha_with_reload::<i64, _, _>(
            tv.client().maintenance_pool.next(),
            vec![tv.key_schema().gc_leader_lock()],
            vec![value.to_owned(), LEADER_LOCK_TTL_MS.to_string()],
        )
        .await
        .expect("the older CAS must run")
}

/// A lock with no protocol marker — an older build's, in either spelling it
/// ever wrote (`label:instance`, or 0.1.0's bare label) — is taken over at
/// once, with the production TTL, whatever TTL it had left.
///
/// **On the build before b.1jt** the lock script was the plain CAS: a live
/// unmarked lock was another process's, and this build waited it out.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_unmarked_gc_lock_is_taken_over_in_both_older_spellings() {
    let tv = TestValkey::setup().await;
    let me = this_build("b1jt-unmarked");
    for older in ["older-replica", LEGACY_LOCK_VALUE] {
        plant_lock(&tv, older, Some(30_000)).await;
        assert_eq!(
            acquire(&tv, &me).await,
            GcLeaderAcquire::TookOverOlder {
                displaced: older.to_owned()
            },
            "{older}: a value without the marker is an older build's"
        );
        assert_eq!(lock_value(&tv).await, Some(me.gc_lock_value()));
        assert!(
            lock_pttl(&tv).await > GC_LEADER_TAKEOVER_BELOW_MS as i64,
            "{older}: the takeover writes the full TTL"
        );
    }
}

/// A marked value of a LOWER version is an older build's too, and so is one
/// whose marker is malformed (no decimal version, or no `|`).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_lower_or_malformed_marked_version_is_taken_over_as_older() {
    let tv = TestValkey::setup().await;
    let me = this_build("b1jt-lower");
    let lower = marked(GC_LEADER_PROTOCOL_VERSION - 1, "older-marked");
    for older in [
        lower.as_str(),
        "le-gc/x|replica:1",
        "le-gc/3replica:1",
        "le-gc/|replica:1",
    ] {
        plant_lock(&tv, older, Some(30_000)).await;
        assert_eq!(
            acquire(&tv, &me).await,
            GcLeaderAcquire::TookOverOlder {
                displaced: older.to_owned()
            },
            "{older}"
        );
        assert_eq!(lock_value(&tv).await, Some(me.gc_lock_value()));
    }
}

/// The TTL a held peer's lock is planted with: above the takeover threshold
/// and below the TTL any acquire writes, so a `PTTL` at or under it after the
/// call proves the caller that found the lock held wrote nothing — a stray
/// `PEXPIRE` or `SET` of the full TTL would read above it.
const HELD_PEER_PX: i64 = 20_000;

/// A marked peer of this build's version — here one with the same service
/// label and another instance — is left alone while its TTL is above the
/// takeover threshold (nothing written, its TTL untouched), and taken over
/// once it is at or below it, or when it has no TTL at all.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_same_version_peer_is_kept_above_the_threshold_and_taken_at_or_below_it() {
    let tv = TestValkey::setup().await;
    let me = this_build("b1jt-peer");
    let peer = LeaderIdentity::new("b1jt-peer", "other-instance").gc_lock_value();

    plant_lock(&tv, &peer, Some(HELD_PEER_PX)).await;
    assert_eq!(
        acquire(&tv, &me).await,
        GcLeaderAcquire::Held {
            holder: peer.clone()
        }
    );
    assert_eq!(lock_value(&tv).await, Some(peer.clone()));
    assert!(
        lock_pttl(&tv).await <= HELD_PEER_PX,
        "a held lock is not refreshed by the caller that found it held"
    );

    let near_expiry = (GC_LEADER_TAKEOVER_BELOW_MS / 2) as i64;
    for (px, what) in [
        (Some(near_expiry), "at or below the threshold"),
        (None, "with no TTL"),
    ] {
        plant_lock(&tv, &peer, px).await;
        assert_eq!(
            acquire(&tv, &me).await,
            GcLeaderAcquire::TookOverExpiring {
                displaced: peer.clone()
            },
            "a same-version peer {what} is taken over"
        );
        assert_eq!(lock_value(&tv).await, Some(me.gc_lock_value()));
        assert!(lock_pttl(&tv).await > GC_LEADER_TAKEOVER_BELOW_MS as i64);
    }
}

/// A second acquire by the identity that holds the lock is `refreshed`: the
/// value stays, and the TTL is restored to the full lease, whether what was
/// left was above the takeover threshold or at or below it. The script must
/// recognise its own value before it reads the marker: a misreading would
/// answer `held` above the threshold (and the lease would not be extended)
/// or `took_over_expiring` below it.
///
/// **On the build before b.1jt** the GC lock ran the plain CAS, which has no
/// verdicts; the script this pins did not exist.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_second_acquire_by_the_holder_is_refreshed_and_restores_the_ttl() {
    let tv = TestValkey::setup().await;
    let me = this_build("b1jt-refresh");
    assert_eq!(acquire(&tv, &me).await, GcLeaderAcquire::Acquired);

    for left in [HELD_PEER_PX, (GC_LEADER_TAKEOVER_BELOW_MS / 2) as i64] {
        let _: bool = tv
            .client()
            .pool
            .pexpire(tv.key_schema().gc_leader_lock(), left, None)
            .await
            .expect("PEXPIRE le:gc:leader");
        assert!(lock_pttl(&tv).await <= left, "precondition: {left} ms left");
        assert_eq!(
            acquire(&tv, &me).await,
            GcLeaderAcquire::Refreshed,
            "{left} ms left: the holder's own value"
        );
        assert_eq!(lock_value(&tv).await, Some(me.gc_lock_value()));
        assert!(
            lock_pttl(&tv).await > HELD_PEER_PX,
            "{left} ms left: the refresh restores the full TTL"
        );
    }
}

/// A HIGHER version is a newer GC protocol: its holder is left alone at any
/// TTL — near expiry included (the routed rule) — so this build never leads
/// across it while it refreshes; only a higher-version value with no TTL,
/// which would never lapse, is taken over.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_higher_version_holder_is_kept_at_any_ttl_unless_it_has_none() {
    let tv = TestValkey::setup().await;
    let me = this_build("b1jt-higher");
    let newer = marked(GC_LEADER_PROTOCOL_VERSION + 1, "newer");
    let near_expiry = (GC_LEADER_TAKEOVER_BELOW_MS / 2) as i64;
    for px in [HELD_PEER_PX, near_expiry] {
        plant_lock(&tv, &newer, Some(px)).await;
        assert_eq!(
            acquire(&tv, &me).await,
            GcLeaderAcquire::Held {
                holder: newer.clone()
            },
            "PX {px}: a newer protocol's live holder is never taken over"
        );
        assert_eq!(lock_value(&tv).await, Some(newer.clone()));
        assert!(lock_pttl(&tv).await <= px, "PX {px}: nothing written");
    }

    plant_lock(&tv, &newer, None).await;
    assert_eq!(
        acquire(&tv, &me).await,
        GcLeaderAcquire::TookOverExpiring {
            displaced: newer.clone()
        },
        "a marked value with no TTL would never lapse, whatever its version"
    );
    assert_eq!(lock_value(&tv).await, Some(me.gc_lock_value()));
}

/// The documented lapse window, one version up: a higher-version holder that
/// stops refreshing lapses within its TTL, and this build — the lower marked
/// version — then acquires the absent key. The lock alone never keeps an
/// older protocol out across a lapse; store state does (SDD §11.3).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_lower_version_acquires_the_key_once_a_higher_version_holders_key_lapses() {
    let tv = TestValkey::setup().await;
    let me = this_build("b1jt-lapse");
    let newer = marked(GC_LEADER_PROTOCOL_VERSION + 1, "newer");
    plant_lock(&tv, &newer, Some(300)).await;
    assert!(matches!(
        acquire(&tv, &me).await,
        GcLeaderAcquire::Held { .. }
    ));

    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while lock_value(&tv).await.is_some() {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the 300 ms lock never lapsed"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(acquire(&tv, &me).await, GcLeaderAcquire::Acquired);
    assert_eq!(lock_value(&tv).await, Some(me.gc_lock_value()));
}

/// The older build's CAS — its acquire and its refresh in one — never matches
/// a marked value: it answers 0 and writes nothing, neither the value nor the
/// TTL. So once this build holds the key, no older GC, orphan sweep or
/// reconciler starts a pass while the key stays marked.
///
/// **On the build before b.1jt** this build wrote the bare `label:instance`
/// the older CAS reads as just another holder — and when it lapsed, the older
/// build's acquire won it just as easily (see
/// [`two_marked_loops_hand_the_lock_over_without_it_ever_lapsing`]).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_older_builds_cas_never_matches_a_marked_lock_and_writes_nothing() {
    let tv = TestValkey::setup().await;
    let me = this_build("b1jt-cas");
    assert_eq!(acquire(&tv, &me).await, GcLeaderAcquire::Acquired);
    let pttl_before = lock_pttl(&tv).await;

    for older in [LEGACY_LOCK_VALUE, "older-replica"] {
        assert_eq!(
            legacy_cas(&tv, older).await,
            0,
            "{older}: the older build does not lead"
        );
    }
    assert_eq!(lock_value(&tv).await, Some(me.gc_lock_value()));
    assert!(
        lock_pttl(&tv).await <= pttl_before,
        "the older CAS refreshed nothing"
    );
}

/// Two marked loops at one version hand the lock over without it ever being
/// absent: the follower takes an expiring lock over before it lapses, so the
/// older build's CAS — which can only claim an absent key — never succeeds,
/// even while the leader has stopped.
///
/// Scaled: each loop calls every 100 ms against a 3 s TTL with a 1 s takeover
/// threshold — production's 1 s calls, 30 s TTL and 10 s threshold divided by
/// ten, the same ratio of call interval to threshold. The test samples the key
/// (`GET`) and runs the older CAS on every 10 ms tick while the leader is
/// stopped, until the follower holds the key.
///
/// **On the build before b.1jt** a non-leader never took a live lock over, so
/// a stopped leader's key always lapsed for a moment, and the older CAS won it
/// then.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_marked_loops_hand_the_lock_over_without_it_ever_lapsing() {
    const TTL_MS: u64 = 3_000;
    const TAKEOVER_BELOW_MS: u64 = 1_000;
    const CALL_EVERY: Duration = Duration::from_millis(100);
    let tv = TestValkey::setup_with_pool_size(4).await;
    let key = tv.key_schema().gc_leader_lock();
    let (a, b) = (
        LeaderIdentity::new("b1jt-loops", "loop-a"),
        LeaderIdentity::new("b1jt-loops", "loop-b"),
    );
    let spawn_loop = |who: LeaderIdentity, cancel: CancellationToken| {
        let (client, key) = (tv.client().clone(), key.clone());
        tokio::spawn(async move {
            while !cancel.is_cancelled() {
                acquire_gc_leader_lock(&client, &key, &who, TTL_MS, TAKEOVER_BELOW_MS)
                    .await
                    .expect("gc_leader_acquire must run");
                tokio::select! {
                    () = cancel.cancelled() => {}
                    () = tokio::time::sleep(CALL_EVERY) => {}
                }
            }
        })
    };
    let (cancel_a, cancel_b) = (CancellationToken::new(), CancellationToken::new());
    let loop_a = spawn_loop(a.clone(), cancel_a.clone());
    let loop_b = spawn_loop(b.clone(), cancel_b.clone());

    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    let leader_value = loop {
        if let Some(v) = lock_value(&tv).await {
            break v;
        }
        assert!(tokio::time::Instant::now() < deadline, "nobody acquired");
        tokio::time::sleep(Duration::from_millis(10)).await;
    };
    let (stop, follower) = if leader_value == a.gc_lock_value() {
        (cancel_a.clone(), b.gc_lock_value())
    } else {
        assert_eq!(leader_value, b.gc_lock_value());
        (cancel_b.clone(), a.gc_lock_value())
    };
    stop.cancel();

    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    let mut samples = 0usize;
    loop {
        let value = lock_value(&tv).await;
        samples += 1;
        assert!(
            value.is_some(),
            "sample {samples}: the key was absent — the follower let it lapse"
        );
        assert_eq!(
            legacy_cas(&tv, LEGACY_LOCK_VALUE).await,
            0,
            "sample {samples}: the older CAS must never win the key"
        );
        if value.as_deref() == Some(follower.as_str()) {
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "the follower never took the stopped leader's lock over"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    cancel_a.cancel();
    cancel_b.cancel();
    for handle in [loop_a, loop_b] {
        tokio::time::timeout(Duration::from_secs(5), handle)
            .await
            .expect("the loop stops on cancel")
            .expect("the loop must not panic");
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// T1 — the lock as the GC task holds it: verdict logging, the progress-gated
// mid-pass refresh ticker and its failure logging, a loop that dies or wedges
// ═════════════════════════════════════════════════════════════════════════════

/// What [`GcTaskStore`] does on top of the real store.
enum Twist {
    /// Every retirement attempt the real store ran (enqueued, or already
    /// landed) is answered with an error, as when its reply is lost: the
    /// attempt landed, and the retirement cannot know it.
    LoseRetireReply,
    /// The first `process_pending` raises `entered`, then waits for `release`
    /// before it runs, marking no progress meanwhile: a GC pass that is
    /// wedged — hung, not ended — for as long as the test likes.
    HoldFirstPass {
        entered: Arc<AtomicBool>,
        release: CancellationToken,
    },
    /// The first `process_pending` raises `entered`, then runs the real
    /// store's pass every `every` until `hold` has passed, and returns the
    /// last one's result: one long GC pass that keeps making progress. Each
    /// inner pass marks progress on the loop's heartbeat, which `state`
    /// carries, as a long real pass does after every step and member.
    ProgressingFirstPass {
        entered: Arc<AtomicBool>,
        hold: Duration,
        every: Duration,
    },
    /// `process_pending` panics: a GC loop that dies while the process lives.
    PanicOnPass,
}

/// The real store, with one [`Twist`]; every other method delegates.
struct GcTaskStore {
    inner: ValkeyExternalValueStore,
    twist: Twist,
    passes: AtomicUsize,
}

impl GcTaskStore {
    fn new(inner: ValkeyExternalValueStore, twist: Twist) -> Self {
        Self {
            inner,
            twist,
            passes: AtomicUsize::new(0),
        }
    }

    /// How many passes have started.
    fn passes(&self) -> usize {
        self.passes.load(Ordering::SeqCst)
    }
}

#[async_trait::async_trait]
impl ExternalValueStore for GcTaskStore {
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
    ) -> Result<HashMap<[u8; 32], Vec<u8>>, ServiceError> {
        self.inner.read_batch(hashes).await
    }

    async fn probe_values(&self, hashes: &[[u8; 32]]) -> Result<ExtValueProbe, ServiceError> {
        self.inner.probe_values(hashes).await
    }

    async fn enqueue_retirement_attempt(
        &self,
        object_id: &str,
        attempt: &str,
        releases: &[ExtRelease],
        send_by: std::time::Instant,
    ) -> Result<ReleaseAttemptOutcome, ServiceError> {
        let outcome = self
            .inner
            .enqueue_retirement_attempt(object_id, attempt, releases, send_by)
            .await?;
        match (&self.twist, outcome) {
            (
                Twist::LoseRetireReply,
                ReleaseAttemptOutcome::Enqueued | ReleaseAttemptOutcome::AlreadyLanded,
            ) => Err(ServiceError::Unavailable(
                "b.1jt test: the attempt ran and its reply was lost".to_owned(),
            )),
            (_, outcome) => Ok(outcome),
        }
    }

    async fn process_pending(
        &self,
        policy: &GcPolicy,
        state: &mut GcPassState,
    ) -> Result<GcBatchResult, ServiceError> {
        let first = self.passes.fetch_add(1, Ordering::SeqCst) == 0;
        match &self.twist {
            Twist::HoldFirstPass { entered, release } if first => {
                entered.store(true, Ordering::SeqCst);
                release.cancelled().await;
            }
            Twist::ProgressingFirstPass {
                entered,
                hold,
                every,
            } if first => {
                entered.store(true, Ordering::SeqCst);
                let started = tokio::time::Instant::now();
                loop {
                    let result = self.inner.process_pending(policy, state).await;
                    if started.elapsed() >= *hold {
                        return result;
                    }
                    tokio::time::sleep(*every).await;
                }
            }
            Twist::PanicOnPass => panic!("b.1jt test: the GC pass panics"),
            _ => {}
        }
        self.inner.process_pending(policy, state).await
    }

    fn ext_release_gate(&self) -> Option<&Arc<ExtReleaseGate>> {
        self.inner.ext_release_gate()
    }
}

/// A capturing log writer for the GC task's lines. Install it with
/// [`capture_logs`] on a current-thread runtime: the GC task is spawned, so it
/// is polled on the thread the capture is installed on.
#[derive(Clone)]
struct CapturedLog(Arc<Mutex<Vec<u8>>>);

impl<'a> MakeWriter<'a> for CapturedLog {
    type Writer = CapturedLog;
    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

impl Write for CapturedLog {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl CapturedLog {
    fn text(&self) -> String {
        String::from_utf8(self.0.lock().unwrap().clone()).expect("UTF-8 log output")
    }

    /// The captured lines at `level` (e.g. `" WARN "`) containing every one
    /// of `texts`.
    fn count(&self, level: &str, texts: &[&str]) -> usize {
        self.text()
            .lines()
            .filter(|line| line.contains(level) && texts.iter().all(|t| line.contains(t)))
            .count()
    }
}

/// Capture this thread's log lines under `filter` until the guard drops.
fn capture_logs(filter: &str) -> (CapturedLog, tracing::subscriber::DefaultGuard) {
    let log = CapturedLog(Arc::new(Mutex::new(Vec::new())));
    let guard = tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_writer(log.clone())
            .with_env_filter(tracing_subscriber::EnvFilter::new(filter))
            .with_ansi(false)
            .finish(),
    );
    (log, guard)
}

/// Wait until `le:gc:leader` holds `who`'s value.
async fn wait_for_lock(tv: &TestValkey, who: &LeaderIdentity) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while lock_value(tv).await != Some(who.gc_lock_value()) {
        assert!(
            tokio::time::Instant::now() < deadline,
            "{} never took the lock: {:?}",
            who.gc_lock_value(),
            lock_value(tv).await
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// Run this build's GC task (`spawn_gc_task`, so with its refresh ticker)
/// over the real store as `who` until it holds the lock, then for 1.5 s more,
/// so the ticker has called too; then stop it.
async fn run_gc_task_until_it_leads(tv: &TestValkey, who: &LeaderIdentity, m_: &Arc<Metrics>) {
    let cancel = CancellationToken::new();
    let handle = spawn_gc_task(
        Arc::new(store(tv, m_)) as Arc<dyn ExternalValueStore>,
        tv.client().clone(),
        tv.key_schema().clone(),
        who.clone(),
        GcPolicy::PRODUCTION,
        cancel.clone(),
        Some(Arc::clone(m_)),
    );
    wait_for_lock(tv, who).await;
    tokio::time::sleep(Duration::from_millis(1_500)).await;
    cancel.cancel();
    tokio::time::timeout(Duration::from_secs(5), handle)
        .await
        .expect("the GC task stops on cancel")
        .expect("the GC task must not panic");
}

/// How the GC task reports a takeover (`try_acquire_gc_leader`): from a
/// same-version peer whose lock is expiring, one INFO line and no tick of
/// `live_edit_gc_older_build_total{event="lock_taken_over"}` — that counter is
/// an operator's signal that an older build was displaced, and a routine
/// failover must not inflate it; from an older build (no marker), one WARN
/// line naming the displaced value, and one tick. The ticker's later calls
/// find the task's own value (`refreshed`) and log nothing.
///
/// Current-thread runtime with a thread-local log capture: the GC task and its
/// ticker are spawned, so they are polled on the thread the capture is
/// installed on.
///
/// **On the build before b.1jt** neither planted lock was taken over: the
/// task waited both out, logging nothing.
#[tokio::test]
async fn the_gc_task_warns_and_counts_only_a_takeover_from_an_older_build() {
    const OLDER_TEXT: &str = "took over the GC leader lock from an older build";
    const EXPIRING_TEXT: &str = "took over the GC leader lock from a peer that stopped";
    let tv = TestValkey::setup().await;
    let (log, _capture) = capture_logs("live_edit::storage::external_values=info");
    let m_ = metrics();
    let me = this_build("b1jt-verdicts");
    let peer = LeaderIdentity::new("b1jt-verdicts", "other-instance").gc_lock_value();

    plant_lock(&tv, &peer, Some((GC_LEADER_TAKEOVER_BELOW_MS / 2) as i64)).await;
    run_gc_task_until_it_leads(&tv, &me, &m_).await;
    assert_eq!(
        log.count(" INFO ", &[EXPIRING_TEXT, peer.as_str()]),
        1,
        "one INFO for the expiring peer's takeover; got {}",
        log.text()
    );
    assert_eq!(log.count(" WARN ", &[OLDER_TEXT]), 0, "got {}", log.text());
    assert_eq!(
        older_build_events(&m_, "lock_taken_over"),
        0,
        "a same-version takeover is not an older build's"
    );

    plant_lock(&tv, LEGACY_LOCK_VALUE, Some(30_000)).await;
    run_gc_task_until_it_leads(&tv, &me, &m_).await;
    assert_eq!(
        log.count(" WARN ", &[OLDER_TEXT, LEGACY_LOCK_VALUE]),
        1,
        "one WARN naming the older build's value; got {}",
        log.text()
    );
    assert_eq!(
        log.count(" INFO ", &[EXPIRING_TEXT]),
        1,
        "and no second INFO; got {}",
        log.text()
    );
    assert_eq!(older_build_events(&m_, "lock_taken_over"), 1);
}

/// INV-1's mid-pass refresh: a leader whose one GC pass runs longer than the
/// lock's 30 s TTL plus the progress bound `GC_PROGRESS_STALL_AFTER` (20 s)
/// still never lets `le:gc:leader` lapse while the pass keeps making progress,
/// because the refresh ticker that `spawn_gc_task` runs beside the loop renews
/// once a second as long as the loop's last progress mark is younger than the
/// bound. The pass here runs the real store's pass every 2 s inside one outer
/// pass, so progress is marked from inside `process_pending` — as a long real
/// pass marks it after every step and member — and never by the loop's own
/// per-wake-up mark. Throughout 54 s of that pass the key is sampled every
/// 50 ms: it always holds the leader's value, and the older build's CAS, which
/// can only claim an absent key, never wins it. This is the single-leader
/// case: no peer of this build exists to take an expiring key over.
///
/// The production TTL, bound and ticker interval, unscaled (they are
/// constants), so this test runs for about 57 s. 54 s is past the latest a
/// key could lapse if the in-pass marks were ignored: the ticker would stop
/// renewing 20 s after the pass began, and the key would lapse 30 s later.
///
/// **On the build before b.1jt**, and with the ticker removed, the lock was
/// refreshed only between passes: the key lapsed 30 s into the pass, and the
/// older CAS won it. With the in-pass progress marks removed, the key lapses
/// about 49 s into the pass.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_leaders_progressing_pass_past_the_ttl_and_the_stall_bound_never_lets_the_key_lapse() {
    const HOLD: Duration = Duration::from_millis(
        LEADER_LOCK_TTL_MS + GC_PROGRESS_STALL_AFTER.as_millis() as u64 + 4_000,
    );
    let tv = TestValkey::setup_with_pool_size(4).await;
    let m_ = metrics();
    let me = this_build("b1jt-midpass");
    let entered = Arc::new(AtomicBool::new(false));
    let s = Arc::new(GcTaskStore::new(
        store(&tv, &m_),
        Twist::ProgressingFirstPass {
            entered: Arc::clone(&entered),
            hold: HOLD + Duration::from_secs(1),
            every: Duration::from_secs(2),
        },
    ));
    let cancel = CancellationToken::new();
    let handle = spawn_gc_task(
        s.clone() as Arc<dyn ExternalValueStore>,
        tv.client().clone(),
        tv.key_schema().clone(),
        me.clone(),
        GcPolicy::PRODUCTION,
        cancel.clone(),
        Some(Arc::clone(&m_)),
    );

    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while !entered.load(Ordering::SeqCst) {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the GC task never started a pass"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let held_since = tokio::time::Instant::now();
    let mut samples = 0usize;
    while held_since.elapsed() < HOLD {
        let value = lock_value(&tv).await;
        samples += 1;
        assert_eq!(
            value,
            Some(me.gc_lock_value()),
            "sample {samples}, {:?} into the long pass: the leader's key lapsed or changed hands",
            held_since.elapsed()
        );
        assert_eq!(
            legacy_cas(&tv, LEGACY_LOCK_VALUE).await,
            0,
            "sample {samples}, {:?} into the long pass: the older CAS must never win the key",
            held_since.elapsed()
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert_eq!(
        s.passes(),
        1,
        "precondition: one outer pass, running throughout"
    );
    assert!(
        lock_pttl(&tv).await > GC_LEADER_TAKEOVER_BELOW_MS as i64,
        "the key was refreshed during the pass, past the TTL it was taken with"
    );

    cancel.cancel();
    tokio::time::timeout(Duration::from_secs(15), handle)
        .await
        .expect("the GC task stops on cancel")
        .expect("the GC task must not panic");
}

/// A GC loop that dies while its process lives (a panic in a pass) takes its
/// refresh ticker with it: the handle `spawn_gc_task` returned completes at
/// once, without `cancel`, re-raising the panic; one ERROR line says the loop
/// ended with `panicked=true`; and nothing refreshes `le:gc:leader` any more,
/// so its TTL runs down and a peer of this build takes it over once it is at
/// or below the takeover threshold, without waiting for the progress bound. A
/// ticker that outlived its loop would keep this replica's lock alive with no
/// GC pass behind it until the bound ran out (20 s after the loop's last
/// progress mark).
///
/// The peer's takeover is shown without waiting out the 30 s TTL: the test
/// shortens the key's TTL to 12 s (the peer is `held`), then waits 2.5 s —
/// past two ticker intervals and well inside the progress bound, so a live
/// ticker would have restored the full TTL — and the peer takes the key over.
///
/// Current-thread runtime with a thread-local log capture, as above.
///
/// **On the build before b.1jt** a dead loop was logged nowhere: that build
/// has no "GC loop ended before shutdown" line, so the one-ERROR count fails.
/// (It had no ticker either.) In this tree, a ticker that is not stopped when
/// its loop ends keeps the handle from completing, and the `timeout` below
/// fails.
#[tokio::test]
async fn a_gc_loop_that_panics_stops_refreshing_the_lock_and_its_handle_completes() {
    const ENDED_TEXT: &str = "GC loop ended before shutdown";
    let tv = TestValkey::setup().await;
    let (log, _capture) = capture_logs("live_edit::storage::external_values=error");
    let m_ = metrics();
    let me = this_build("b1jt-dead-loop");
    let s = Arc::new(GcTaskStore::new(store(&tv, &m_), Twist::PanicOnPass));
    let cancel = CancellationToken::new();
    let handle = spawn_gc_task(
        s.clone() as Arc<dyn ExternalValueStore>,
        tv.client().clone(),
        tv.key_schema().clone(),
        me.clone(),
        GcPolicy::PRODUCTION,
        cancel.clone(),
        Some(Arc::clone(&m_)),
    );

    let joined = tokio::time::timeout(Duration::from_secs(10), handle)
        .await
        .expect("the handle completes once the loop dies, without cancel");
    let err = joined.expect_err("the loop's panic is re-raised through the handle");
    assert!(err.is_panic(), "{err:?}");
    assert!(!cancel.is_cancelled(), "precondition: nothing cancelled it");
    assert_eq!(
        s.passes(),
        1,
        "precondition: the loop died in its first pass"
    );
    assert_eq!(
        lock_value(&tv).await,
        Some(me.gc_lock_value()),
        "precondition: it led"
    );
    assert_eq!(
        log.count("ERROR", &[ENDED_TEXT, "panicked=true"]),
        1,
        "one ERROR line for the dead loop; got {}",
        log.text()
    );

    let at_death = lock_pttl(&tv).await;
    tokio::time::sleep(Duration::from_millis(2_500)).await;
    let later = lock_pttl(&tv).await;
    assert!(
        later <= at_death - 2_000,
        "nothing refreshed the lock after the loop died: PTTL {at_death} then {later}"
    );

    let peer = LeaderIdentity::new("b1jt-dead-loop", "peer-instance");
    let _: bool = tv
        .client()
        .pool
        .pexpire(tv.key_schema().gc_leader_lock(), 12_000, None)
        .await
        .expect("PEXPIRE le:gc:leader");
    assert_eq!(
        acquire(&tv, &peer).await,
        GcLeaderAcquire::Held {
            holder: me.gc_lock_value()
        },
        "above the threshold the peer waits"
    );
    tokio::time::sleep(Duration::from_millis(2_500)).await;
    assert_eq!(
        acquire(&tv, &peer).await,
        GcLeaderAcquire::TookOverExpiring {
            displaced: me.gc_lock_value()
        },
        "with no refresh, the TTL ran down below the threshold and the peer took the lock"
    );
    assert_eq!(
        log.count("ERROR", &[ENDED_TEXT]),
        1,
        "still one ERROR line; got {}",
        log.text()
    );
}

/// A GC loop that is wedged — a pass that awaits forever without ending, so
/// the loop neither progresses nor dies — gives the lock up: the refresh
/// ticker renews `le:gc:leader` only while the loop's last progress mark is
/// younger than `GC_PROGRESS_STALL_AFTER` (20 s), then stops with one WARN, so
/// the key's TTL runs down and a peer of this build takes it over at the
/// production threshold — about 40 s into the wedge, never later than the TTL
/// plus the bound. Until the bound the ticker keeps renewing, so the takeover
/// comes no sooner than the TTL after the wedge began. The wedged task's
/// handle does not finish meanwhile: it is neither dead nor cancelled. Once
/// the pass is released, the loop marks progress again and the ticker logs
/// one INFO as it resumes — counted two ticker intervals later, so an INFO
/// repeated on every tick would show.
///
/// The production TTL, bound, threshold and ticker interval, unscaled, so
/// this test runs for about 45 s. Current-thread runtime with a thread-local
/// log capture, as above.
///
/// **On the build before b.1jt** there was no ticker: the lock, refreshed only
/// between passes, was taken over about 20 s into the wedge, so the lower
/// bound fails, and no stall WARN was logged. With a ticker that renews
/// regardless of progress, the wedged loop keeps the lock forever and the
/// deadline fails; with one that does not reset its stalled state on resuming,
/// the INFO repeats every tick and the one-INFO count fails.
#[tokio::test]
async fn a_wedged_gc_loop_stops_renewing_the_lock_and_a_peer_takes_it_over() {
    const STALL_TEXT: &str = "the GC loop has made no progress for too long";
    const RESUMED_TEXT: &str = "resumed refreshing the GC leader lock";
    let bound = Duration::from_millis(LEADER_LOCK_TTL_MS) + GC_PROGRESS_STALL_AFTER;
    let tv = TestValkey::setup().await;
    let (log, _capture) = capture_logs("live_edit::storage::external_values=info");
    let m_ = metrics();
    let me = this_build("b1jt-wedged");
    let peer = LeaderIdentity::new("b1jt-wedged", "peer-instance");
    let (entered, release) = (Arc::new(AtomicBool::new(false)), CancellationToken::new());
    let s = Arc::new(GcTaskStore::new(
        store(&tv, &m_),
        Twist::HoldFirstPass {
            entered: Arc::clone(&entered),
            release: release.clone(),
        },
    ));
    let cancel = CancellationToken::new();
    let handle = spawn_gc_task(
        s.clone() as Arc<dyn ExternalValueStore>,
        tv.client().clone(),
        tv.key_schema().clone(),
        me.clone(),
        GcPolicy::PRODUCTION,
        cancel.clone(),
        Some(Arc::clone(&m_)),
    );
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while !entered.load(Ordering::SeqCst) {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the GC task never started a pass"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let wedged_since = tokio::time::Instant::now();
    assert_eq!(
        lock_value(&tv).await,
        Some(me.gc_lock_value()),
        "precondition: it leads"
    );

    let taken_after = loop {
        match acquire(&tv, &peer).await {
            GcLeaderAcquire::Held { holder } => assert_eq!(holder, me.gc_lock_value()),
            GcLeaderAcquire::TookOverExpiring { displaced } => {
                assert_eq!(displaced, me.gc_lock_value());
                break wedged_since.elapsed();
            }
            GcLeaderAcquire::Acquired => break wedged_since.elapsed(),
            other => panic!("unexpected verdict for the peer: {other:?}"),
        }
        assert!(
            !handle.is_finished(),
            "the wedged task's handle must not finish"
        );
        assert!(
            wedged_since.elapsed() < bound + Duration::from_secs(10),
            "the wedged loop never gave the lock up; got {}",
            log.text()
        );
        tokio::time::sleep(Duration::from_millis(250)).await;
    };
    assert!(
        taken_after >= Duration::from_millis(LEADER_LOCK_TTL_MS),
        "the ticker kept renewing until the bound, so the key outlives the TTL: taken over \
         after {taken_after:?}"
    );
    assert!(
        taken_after <= bound,
        "the lock is given up within the TTL plus the bound: taken over after {taken_after:?}"
    );
    assert!(!handle.is_finished(), "still wedged, not ended");
    assert_eq!(
        log.count(" WARN ", &[STALL_TEXT]),
        1,
        "one WARN as the ticker stops renewing; got {}",
        log.text()
    );
    assert_eq!(
        log.count(" INFO ", &[RESUMED_TEXT]),
        0,
        "got {}",
        log.text()
    );

    release.cancel();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    while log.count(" INFO ", &[RESUMED_TEXT]) == 0 {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the ticker never resumed once the loop progressed; got {}",
            log.text()
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    // Two more ticker intervals: a ticker that did not reset its stalled state
    // would log the INFO again on each of them.
    tokio::time::sleep(Duration::from_millis(2_500)).await;
    assert_eq!(
        log.count(" INFO ", &[RESUMED_TEXT]),
        1,
        "one INFO as the ticker resumes, not one per tick; got {}",
        log.text()
    );
    assert_eq!(log.count(" WARN ", &[STALL_TEXT]), 1, "got {}", log.text());
    assert_eq!(
        lock_value(&tv).await,
        Some(peer.gc_lock_value()),
        "the peer keeps the key it took over"
    );

    cancel.cancel();
    tokio::time::timeout(Duration::from_secs(10), handle)
        .await
        .expect("the GC task stops on cancel")
        .expect("the GC task must not panic");
}

/// `live_edit_gc_sweep_batch_errors_total{kind="leader_lock_error"}`.
fn leader_lock_errors(metrics: &Metrics) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_gc_sweep_batch_errors_total",
        &[("kind", "leader_lock_error")],
    )
    .unwrap_or(0.0) as u64
}

/// The refresh ticker logs a run of failed calls on its edges only: one WARN
/// when its calls start failing — not one a second — and one INFO when one
/// succeeds again, while `gc_sweep_batch_errors_total{kind="leader_lock_error"}`
/// counts every failure, the ticker's and the loop's gate's alike. The store
/// is reached through a proxy that, while the test says so, answers every
/// call naming `le:gc:leader` with an error; the proxy's match count is the
/// number of failed calls. The INFO is counted two ticker intervals after it
/// first appears, so an INFO repeated on every later success would show.
///
/// Current-thread runtime with a thread-local log capture, as above.
///
/// **On the build before b.1jt** there was no ticker. With a ticker that logs
/// every failure, the one-WARN count fails; with one that does not reset its
/// failing state on success, the INFO repeats and the one-INFO count fails.
#[tokio::test]
async fn a_failing_refresh_logs_one_warn_counts_every_failure_and_one_info_on_recovery() {
    const FAILED_TEXT: &str = "GC leader lock refresh failed";
    const RECOVERED_TEXT: &str = "GC leader lock refresh succeeded again";
    let tv = TestValkey::setup().await;
    let lock_key = tv.key_schema().gc_leader_lock();
    let failing = Arc::new(AtomicBool::new(false));
    let rule: ProxyRule = {
        let failing = Arc::clone(&failing);
        Arc::new(move |args: &[Vec<u8>]| {
            let script = args.first().is_some_and(|name| {
                name.eq_ignore_ascii_case(b"EVALSHA") || name.eq_ignore_ascii_case(b"EVAL")
            });
            let on_lock = args.iter().any(|a| a.as_slice() == lock_key.as_bytes());
            (failing.load(Ordering::SeqCst) && script && on_lock).then(|| {
                ProxyAction::Eval(error_reply_script("ERR b.1jt test: staged lock failure"))
            })
        })
    };
    let proxy = ScriptReplyProxy::start(test_valkey_addr(), rule).await;
    let client = client_through(&proxy).await;
    let (log, _capture) = capture_logs("live_edit::storage::external_values=info");
    let m_ = metrics();
    let me = this_build("b1jt-ticker-failures");
    let cancel = CancellationToken::new();
    let handle = spawn_gc_task(
        Arc::new(store(&tv, &m_)) as Arc<dyn ExternalValueStore>,
        client,
        tv.key_schema().clone(),
        me.clone(),
        GcPolicy::PRODUCTION,
        cancel.clone(),
        Some(Arc::clone(&m_)),
    );
    wait_for_lock(&tv, &me).await;
    tokio::time::sleep(Duration::from_millis(1_200)).await;
    assert_eq!(leader_lock_errors(&m_), 0, "precondition: healthy");

    failing.store(true, Ordering::SeqCst);
    tokio::time::sleep(Duration::from_millis(4_500)).await;
    failing.store(false, Ordering::SeqCst);

    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    while log.count(" INFO ", &[RECOVERED_TEXT]) == 0 {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the ticker never logged its recovery; got {}",
            log.text()
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    // Two more ticker intervals: a ticker that did not reset its failing state
    // would log the INFO again on each successful call.
    tokio::time::sleep(Duration::from_millis(2_500)).await;
    let failures = proxy.hits();
    assert!(
        failures >= 6,
        "precondition: the ticker and the gate each failed several times: {failures}"
    );
    assert_eq!(
        leader_lock_errors(&m_),
        failures as u64,
        "every failed call is counted"
    );
    assert_eq!(
        log.count(" WARN ", &[FAILED_TEXT]),
        1,
        "one WARN for the run of failures, not one per call; got {}",
        log.text()
    );
    assert_eq!(
        log.count(" INFO ", &[RECOVERED_TEXT]),
        1,
        "one INFO on recovery, not one per later call; got {}",
        log.text()
    );

    cancel.cancel();
    tokio::time::timeout(Duration::from_secs(10), handle)
        .await
        .expect("the GC task stops on cancel")
        .expect("the GC task must not panic");
    drop(proxy);
}

/// The WARN the refresh ticker logs when it stops renewing for want of
/// progress.
const GC_STALL_WARN_TEXT: &str = "the GC loop has made no progress for too long";

/// A store whose client reaches Valkey through a proxy that, while `slow` is
/// set, holds each frame `delay_for` names for that long before forwarding it
/// — so every frame behind it on the connection waits too — and the proxy.
/// The GC task's gate and ticker keep the test's direct client, so only the
/// pass is slow.
async fn slow_store(
    tv: &TestValkey,
    m_: &Arc<Metrics>,
    slow: Arc<AtomicBool>,
    delay_for: impl Fn(&[Vec<u8>]) -> Option<Duration> + Send + Sync + 'static,
) -> (ScriptReplyProxy, ValkeyExternalValueStore) {
    let rule: ProxyRule = Arc::new(move |args: &[Vec<u8>]| {
        if !slow.load(Ordering::SeqCst) {
            return None;
        }
        delay_for(args).map(ProxyAction::Delay)
    });
    let proxy = ScriptReplyProxy::start(test_valkey_addr(), rule).await;
    let client = client_through(&proxy).await;
    let s =
        ValkeyExternalValueStore::new(client, tv.key_schema().clone()).with_metrics(Arc::clone(m_));
    (proxy, s)
}

/// Whether `args` is an `EVALSHA` of `script`.
fn is_evalsha_of(args: &[Vec<u8>], script: &Script) -> bool {
    args.first()
        .is_some_and(|name| name.eq_ignore_ascii_case(b"EVALSHA"))
        && args
            .get(1)
            .is_some_and(|sha| sha.as_slice() == script.sha1().as_bytes())
}

/// One sample of a leader's hold on `le:gc:leader` while its pass is slow,
/// `since` the slow part began: the key holds `me`'s value with a TTL above
/// the takeover threshold, a peer at the production threshold is refused
/// (`Held`), and the ticker has not logged the stall WARN. Returns the PTTL.
async fn assert_leads_unstalled(
    tv: &TestValkey,
    me: &LeaderIdentity,
    peer: &LeaderIdentity,
    log: &CapturedLog,
    since: tokio::time::Instant,
) -> i64 {
    let at = since.elapsed();
    assert_eq!(
        log.count(" WARN ", &[GC_STALL_WARN_TEXT]),
        0,
        "{at:?} into the slow pass: the ticker stopped renewing for want of progress; got {}",
        log.text()
    );
    assert_eq!(
        lock_value(tv).await,
        Some(me.gc_lock_value()),
        "{at:?} into the slow pass: the key left the leader"
    );
    let pttl = lock_pttl(tv).await;
    assert!(
        pttl > GC_LEADER_TAKEOVER_BELOW_MS as i64,
        "{at:?} into the slow pass: PTTL {pttl} fell to the takeover threshold"
    );
    assert_eq!(
        acquire(tv, peer).await,
        GcLeaderAcquire::Held {
            holder: me.gc_lock_value()
        },
        "{at:?} into the slow pass: a peer must not take the key over"
    );
    pttl
}

/// The enqueue-attempt audit page (step 7 of a pass) marks progress after its
/// page script, after every `HEXISTS` against a record's carrier and after
/// its `forget` script, so a page whose calls are each fast but whose total
/// runs past `GC_PROGRESS_STALL_AFTER` keeps the ticker renewing the lock.
///
/// 100 `landed` records older than the retention are seeded on
/// `le:ext:{gc}:enqueue_attempt`, none of them named by a carrier, so the
/// first pass of the leadership (which starts the audit cycle) checks each
/// with an `HEXISTS` and forgets them all. The store's client reaches Valkey
/// through a proxy that holds every `HEXISTS` 300 ms, so the page runs about
/// 30 s. Throughout, sampled every 250 ms: the key holds the leader's value,
/// its PTTL is reset near 30 s at least every 3 s and never falls to the
/// takeover threshold, a peer is refused, and no stall WARN is logged. The
/// page is checked to be still running past the bound (fewer than 100
/// `HEXISTS` by then) and, at the end, to have removed every record.
///
/// Current-thread runtime with a thread-local log capture, as above. About
/// 32 s.
///
/// **Without the page's progress marks** the last mark before the page is
/// step 6's, the ticker stops renewing 20 s into the page, and the stall WARN
/// fails the test.
#[tokio::test]
async fn a_slow_attempt_audit_page_marks_progress_and_keeps_the_lock() {
    const RECORDS: usize = 100;
    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();
    let records: Vec<(String, String)> = (0..RECORDS)
        .map(|i| (format!("{i:032x}"), format!("landed:1:{}", 79_200 + i)))
        .collect();
    let _: () = tv
        .client()
        .pool
        .hset(keys.ext_enqueue_attempts(), records)
        .await
        .expect("seed the landed records");
    let (log, _capture) = capture_logs("live_edit::storage::external_values=info");
    let m_ = metrics();
    let slow = Arc::new(AtomicBool::new(true));
    let (proxy, s) = slow_store(&tv, &m_, Arc::clone(&slow), |args| {
        args.first()
            .is_some_and(|name| name.eq_ignore_ascii_case(b"HEXISTS"))
            .then_some(Duration::from_millis(300))
    })
    .await;
    let me = this_build("b1jt-slow-audit");
    let peer = LeaderIdentity::new("b1jt-slow-audit", "peer-instance");
    let cancel = CancellationToken::new();
    let handle = spawn_gc_task(
        Arc::new(s) as Arc<dyn ExternalValueStore>,
        tv.client().clone(),
        keys.clone(),
        me.clone(),
        GcPolicy::PRODUCTION,
        cancel.clone(),
        Some(Arc::clone(&m_)),
    );
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while proxy.hits() == 0 {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the audit page never started"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    let started = tokio::time::Instant::now();
    let mut last_reset = started;
    let mut hexists_past_bound = None;
    loop {
        let left: i64 = tv
            .client()
            .pool
            .hlen(keys.ext_enqueue_attempts())
            .await
            .expect("HLEN enqueue_attempt");
        if left == 0 {
            break;
        }
        let pttl = assert_leads_unstalled(&tv, &me, &peer, &log, started).await;
        if pttl >= 28_000 {
            last_reset = tokio::time::Instant::now();
        }
        assert!(
            last_reset.elapsed() <= Duration::from_secs(3),
            "{:?} into the page: the PTTL was not reset for {:?}",
            started.elapsed(),
            last_reset.elapsed()
        );
        if hexists_past_bound.is_none()
            && started.elapsed() >= GC_PROGRESS_STALL_AFTER + Duration::from_secs(1)
        {
            hexists_past_bound = Some(proxy.hits());
        }
        assert!(
            started.elapsed() < Duration::from_secs(60),
            "the audit page never finished: {} HEXISTS",
            proxy.hits()
        );
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    assert!(
        hexists_past_bound.is_some_and(|hits| hits < RECORDS),
        "precondition: the page was still running past the progress bound: {hexists_past_bound:?}"
    );
    assert_eq!(proxy.hits(), RECORDS, "one HEXISTS per record");
    assert_eq!(
        log.count(" WARN ", &[GC_STALL_WARN_TEXT]),
        0,
        "got {}",
        log.text()
    );

    slow.store(false, Ordering::SeqCst);
    cancel.cancel();
    tokio::time::timeout(Duration::from_secs(10), handle)
        .await
        .expect("the GC task stops on cancel")
        .expect("the GC task must not panic");
    drop(proxy);
}

/// One real pass that is slow inside its steps keeps the lock: the settled
/// replay-record teardown (step 4) marks progress as each unmark returns, and
/// the member step (step 5) marks it after every member, so neither step's
/// total — each past `GC_PROGRESS_STALL_AFTER` here — stops the ticker
/// renewing, though no step-boundary mark lands for longer than the bound.
///
/// 100 settled entries past the retention (`GcPolicy::IMMEDIATE`, U = 0) and
/// 100 due pending members are seeded. The store's client reaches Valkey
/// through a proxy that holds every `gc_unmark` 230 ms (about 23 s for the
/// teardown) and every claim and decrement 150 ms (about 30 s for the members;
/// at most 16 calls are in flight on the connection, so no call waits longer
/// than about 4 s, under the 5 s command timeout). Throughout, sampled every 250 ms, the key holds the leader's
/// value with a TTL above the takeover threshold, a peer is refused, and no
/// stall WARN is logged. Each step is checked to have run past the bound.
///
/// Current-thread runtime with a thread-local log capture, as above. About
/// 55 s.
///
/// **Without the teardown's per-unmark marks** the stall WARN fires about
/// 20 s into the teardown; **without the per-member marks**, about 20 s after
/// the teardown ends. Either fails the test.
#[tokio::test]
async fn a_slow_teardown_and_member_step_mark_progress_and_keep_the_lock() {
    const COUNT: u8 = 100;
    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();
    let direct = store(&tv, &metrics());
    for i in 1..=COUNT {
        put(&direct, hash(i), 16).await;
        release(&direct, hash(i), OBJ).await;
    }
    let seeded_settled: Vec<(f64, String)> = (1..=COUNT)
        .map(|i| (1.0, member(hash(i), 79_300)))
        .collect();
    let _: () = tv
        .client()
        .pool
        .zadd(keys.ext_settled(), None, None, false, false, seeded_settled)
        .await
        .expect("seed the settled entries");
    let (log, _capture) = capture_logs("live_edit::storage::external_values=info");
    let m_ = metrics();
    let slow = Arc::new(AtomicBool::new(true));
    let (proxy, s) = slow_store(&tv, &m_, Arc::clone(&slow), |args| {
        if is_evalsha_of(args, &GC_UNMARK_SCRIPT) {
            Some(Duration::from_millis(230))
        } else if is_evalsha_of(args, &GC_CLAIM_PENDING_SCRIPT)
            || is_evalsha_of(args, &GC_DECREMENT_AND_COLLECT_SCRIPT)
        {
            Some(Duration::from_millis(150))
        } else {
            None
        }
    })
    .await;
    let me = this_build("b1jt-slow-steps");
    let peer = LeaderIdentity::new("b1jt-slow-steps", "peer-instance");
    let cancel = CancellationToken::new();
    let handle = spawn_gc_task(
        Arc::new(s) as Arc<dyn ExternalValueStore>,
        tv.client().clone(),
        keys.clone(),
        me.clone(),
        GcPolicy::IMMEDIATE,
        cancel.clone(),
        Some(Arc::clone(&m_)),
    );
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while proxy.hits() == 0 {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the teardown never started"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    let started = tokio::time::Instant::now();
    let mut teardown_took = None;
    let members_took = loop {
        let settled: i64 = tv
            .client()
            .pool
            .zcard(keys.ext_settled())
            .await
            .expect("ZCARD settled");
        if teardown_took.is_none() && settled != i64::from(COUNT) {
            teardown_took = Some(started.elapsed());
        }
        let pending: i64 = tv
            .client()
            .pool
            .scard(keys.ext_pending())
            .await
            .expect("SCARD pending");
        let inflight: i64 = tv
            .client()
            .pool
            .zcard(keys.ext_inflight())
            .await
            .expect("ZCARD inflight");
        if let Some(teardown) = teardown_took {
            if pending == 0 && inflight == 0 {
                break started.elapsed() - teardown;
            }
        }
        assert_leads_unstalled(&tv, &me, &peer, &log, started).await;
        assert!(
            started.elapsed() < Duration::from_secs(90),
            "the slow pass never finished: {} delayed calls",
            proxy.hits()
        );
        tokio::time::sleep(Duration::from_millis(250)).await;
    };
    let teardown_took = teardown_took.expect("the teardown ended");
    for (step, took) in [("teardown", teardown_took), ("member step", members_took)] {
        assert!(
            took > GC_PROGRESS_STALL_AFTER + Duration::from_secs(1),
            "precondition: the {step} ran past the progress bound: {took:?}"
        );
    }
    assert!(
        !data_present(&tv, hash(1)).await,
        "the members were collected"
    );
    assert_eq!(
        log.count(" WARN ", &[GC_STALL_WARN_TEXT]),
        0,
        "got {}",
        log.text()
    );

    slow.store(false, Ordering::SeqCst);
    cancel.cancel();
    tokio::time::timeout(Duration::from_secs(15), handle)
        .await
        .expect("the GC task stops on cancel")
        .expect("the GC task must not panic");
    drop(proxy);
}

// ═════════════════════════════════════════════════════════════════════════════
// T2 — after a takeover: an older sweep still running, an older replica
// starting
// ═════════════════════════════════════════════════════════════════════════════

/// Claim `m` and apply its decrement with this build's scripts, then stop: the
/// completion never runs. Returns the claim time.
async fn claim_and_decrement(tv: &TestValkey, h: [u8; 32], m: &str) -> i64 {
    let keys = tv.key_schema();
    let claim_ms: i64 = GC_CLAIM_PENDING_SCRIPT
        .evalsha_with_reload(
            tv.client().maintenance_pool.next(),
            vec![
                keys.ext_pending(),
                keys.ext_inflight(),
                keys.ext_pending_age(),
                keys.ext_pending_bytes(),
                keys.ext_pending_total(),
                keys.ext_settled(),
            ],
            vec![m.to_owned()],
        )
        .await
        .expect("gc_claim_pending must run");
    assert!(claim_ms > 0, "precondition: claimed");
    let hex = sha256_to_hex(&h);
    let (action, _, _): (String, i64, i64) = GC_DECREMENT_AND_COLLECT_SCRIPT
        .evalsha_with_reload(
            tv.client().maintenance_pool.next(),
            vec![
                keys.ext_meta(&hex),
                keys.ext_data(&hex),
                keys.ext_applied(&hex),
            ],
            vec![
                m.to_owned(),
                claim_ms.to_string(),
                GC_DECREMENT_FENCE_MS.to_string(),
            ],
        )
        .await
        .expect("gc_decrement_and_collect must run");
    assert_eq!(action, "decremented", "precondition: the decrement applied");
    claim_ms
}

/// A member this build claimed and decremented, whose completion was withheld
/// — a sweeper that crashed between the two, or a pass still running when an
/// older replica starts — is out of reach of every older-build path: its
/// startup drain (production age, and the widest age 0) and its sweep leave
/// it in `inflight`, never in `pending`, and decrement nothing. This build
/// then re-claims it in place, and its decrement is answered `replayed`; the
/// member settles, and a blob shared with another reference keeps it.
///
/// **On the build before b.1jt** the claim was scored with the bare time, so
/// the older drain moved it back to `pending` and the older sweep decremented
/// it a second time: `ref_count` 0 and the blob deleted under its other
/// holder.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_older_drain_and_sweep_leave_this_builds_decremented_claim_alone() {
    let tv = TestValkey::setup().await;
    let m_ = metrics();
    let s = store(&tv, &m_);
    let h = hash(0x11);
    let m = member(h, OBJ);
    put(&s, h, 64).await;
    put(&s, h, 64).await;
    release(&s, h, OBJ).await;
    claim_and_decrement(&tv, h, &m).await;
    assert_eq!(ref_count(&tv, h).await, Some(1), "precondition");

    for ttl_ms in [LEADER_LOCK_TTL_MS, 0] {
        assert_eq!(
            legacy_drain(&tv, ttl_ms).await,
            0,
            "drain age {ttl_ms}: an encoded claim is outside every older drain's range"
        );
    }
    assert!(
        legacy_gc_pass(&tv).await.is_empty(),
        "the older sweep finds nothing in pending"
    );
    assert_eq!(
        legacy_claim(&tv, &m).await,
        0,
        "the older claim cannot take a member that is not in pending"
    );
    assert!(!in_pending(&tv, &m).await);
    assert!(inflight_score(&tv, &m)
        .await
        .is_some_and(|s| s >= GC_CLAIM_SCORE_OFFSET_MS as f64));
    assert_eq!(ref_count(&tv, h).await, Some(1), "no second decrement");
    assert!(data_present(&tv, h).await);

    plant_own_claim(&tv, &m, STALE_OWN_CLAIM_AGE).await;
    let p = pass(&s, &immediate_keeping_records()).await;
    assert_eq!(
        (p.reclaimed, p.processed, p.replayed, p.deleted),
        (1, 1, 1, 0),
        "re-claimed in place and answered as the replay it is"
    );
    assert_eq!(ref_count(&tv, h).await, Some(1));
    assert!(data_present(&tv, h).await);
    assert!(settled_of(&tv, &m).await.is_some(), "the member settled");
    assert_eq!(inflight_score(&tv, &m).await, None);
    assert!(!in_pending(&tv, &m).await);
}

// ═════════════════════════════════════════════════════════════════════════════
// T3 — a rollback: only older builds run, then this build returns
// ═════════════════════════════════════════════════════════════════════════════

/// A rollback strands two of this build's claims — one whose decrement applied
/// and one whose did not — and the older builds take over: each replica's
/// startup drain, then their GC passes. None of it decrements either member,
/// so `ref_count` errs high (the not-applied member still owes its decrement)
/// and never low. When this build returns, it re-claims both: the applied one
/// is answered `replayed`, and the other is decremented exactly once.
///
/// **On the build before b.1jt** both claims had bare scores: the older drain
/// moved both back to `pending` and the older GC decremented each — a second
/// decrement of the applied one.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_rollback_decrements_nothing_this_build_claimed_and_this_build_finishes_it_once() {
    let tv = TestValkey::setup().await;
    let m_ = metrics();
    let s = store(&tv, &m_);
    let (applied, not_applied) = (hash(0x21), hash(0x22));
    let (m_applied, m_not_applied) = (member(applied, OBJ), member(not_applied, OBJ));
    for h in [applied, not_applied] {
        put(&s, h, 64).await;
        put(&s, h, 64).await;
        release(&s, h, OBJ).await;
    }
    claim_and_decrement(&tv, applied, &m_applied).await;
    let keys = tv.key_schema();
    let claimed: i64 = GC_CLAIM_PENDING_SCRIPT
        .evalsha_with_reload(
            tv.client().maintenance_pool.next(),
            vec![
                keys.ext_pending(),
                keys.ext_inflight(),
                keys.ext_pending_age(),
                keys.ext_pending_bytes(),
                keys.ext_pending_total(),
                keys.ext_settled(),
            ],
            vec![m_not_applied.clone()],
        )
        .await
        .expect("gc_claim_pending must run");
    assert!(
        claimed > 0,
        "precondition: the second member is claimed only"
    );

    // The rollback: two older replicas start (each drains), then their GC
    // runs a few passes.
    for _ in 0..2 {
        assert_eq!(legacy_drain(&tv, LEADER_LOCK_TTL_MS).await, 0);
    }
    assert_eq!(legacy_drain(&tv, 0).await, 0, "not even at drain age 0");
    for _ in 0..3 {
        assert!(legacy_gc_pass(&tv).await.is_empty());
    }
    assert_eq!(
        (
            ref_count(&tv, applied).await,
            ref_count(&tv, not_applied).await
        ),
        (Some(1), Some(2)),
        "the older builds decremented nothing: high for the member still owed, never low"
    );

    // This build returns.
    for m in [&m_applied, &m_not_applied] {
        plant_own_claim(&tv, m, STALE_OWN_CLAIM_AGE).await;
    }
    let stats = process_pending_gc(&s, Some(&m_), immediate_keeping_records()).await;
    assert_eq!(
        (stats.processed, stats.replayed, stats.deleted),
        (2, 1, 0),
        "both re-claimed; the applied one is a replay"
    );
    assert_eq!(
        (
            ref_count(&tv, applied).await,
            ref_count(&tv, not_applied).await
        ),
        (Some(1), Some(1)),
        "each member's decrement applied exactly once"
    );
    assert_eq!(
        (decrements(&m_, "applied"), decrements(&m_, "replayed")),
        (1, 1)
    );
    for m in [&m_applied, &m_not_applied] {
        assert_eq!(inflight_score(&tv, m).await, None);
        assert!(!in_pending(&tv, m).await);
    }
}

/// This build's retirement carrier is a HASH (b.1jt). An older build's carrier
/// claim reads a carrier with `SCARD` / `SMEMBERS` and its deferral with
/// `SADD`, so on this build's carrier each fails `WRONGTYPE` before writing
/// anything — whether the object is live or dead — and the carrier, its
/// attempt token and the header's flag are left exactly as they were: an
/// older build can neither pay an obligation this build recorded a second
/// time nor delete it.
///
/// The carrier is made by this build's own retirement
/// (`schedule_old_object_ext_gc`: `ext_retire_begin.lua` records the member,
/// the attempt token and the header's flag, then the attempt-mode enqueue
/// runs). Its attempt lands but both replies are lost, so the retirement
/// defers: the carrier stays, its token recorded `landed` on `{gc}`. This
/// build's GC then collects the member (settled) before the rollback, so a
/// second payment would be a second decrement. The older scripts run over the
/// live object first, then again after the callers' teardown (header and
/// `obj:ext` gone).
///
/// **On the build before b.1jt** the begin wrote the carrier as a SET: the
/// older claim read it, deleted it and handed its member back to `pending`,
/// and the first `expect_err` below fails.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_older_carrier_claim_or_deferral_fails_on_this_builds_carrier_and_writes_nothing() {
    const OID: &str = "79001";
    let tv = TestValkey::setup().await;
    let m_ = metrics();
    let s = store(&tv, &m_);
    let keys = tv.key_schema();
    let h = hash(0x23);
    let m = member(h, 79_001);
    put(&s, h, 64).await;
    put(&s, h, 64).await;
    // The retiring object, as the retirement reads it: a header and an
    // `obj:ext` index naming `h`.
    let _: () = tv
        .client()
        .pool
        .hset(keys.obj_header(OID), [("seq", "1")])
        .await
        .expect("seed the header");
    let _: () = tv
        .client()
        .pool
        .hset(keys.obj_ext_index(OID), [(sha256_to_hex(&h), "1")])
        .await
        .expect("seed obj:ext");
    let lossy = GcTaskStore::new(store(&tv, &m_), Twist::LoseRetireReply);
    schedule_old_object_ext_gc_for_test(tv.client(), keys, OID, &lossy, None)
        .await
        .expect_err("both replies were lost, so the retirement defers onto its carrier");

    let carrier = keys.obj_ext_retire_carrier(OID);
    let attempts = keys.obj_ext_retire_attempts(OID);
    let fields: Vec<String> = tv.client().pool.hkeys(&carrier).await.expect("HKEYS");
    assert_eq!(
        fields,
        vec![m.clone()],
        "precondition: the retirement's carrier, a HASH holding the member"
    );
    let tokens: Vec<String> = tv.client().pool.hkeys(&attempts).await.expect("HKEYS");
    assert_eq!(tokens.len(), 1, "precondition: one attempt token");
    let record: Option<String> = tv
        .client()
        .pool
        .hget(keys.ext_enqueue_attempts(), tokens[0].as_str())
        .await
        .expect("HGET the {gc} record");
    assert!(
        record.as_deref().is_some_and(|r| r.starts_with("landed:")),
        "precondition: the attempt landed: {record:?}"
    );
    assert_eq!(pass(&s, &immediate_keeping_records()).await.processed, 1);
    assert!(
        settled_of(&tv, &m).await.is_some(),
        "precondition: collected"
    );
    assert_eq!(ref_count(&tv, h).await, Some(1), "precondition");

    for live in [true, false] {
        if !live {
            let _: i64 = tv
                .client()
                .pool
                .del(vec![keys.obj_header(OID), keys.obj_ext_index(OID)])
                .await
                .expect("the callers' teardown");
        }
        let claim = LEGACY_CARRIER_CLAIM
            .evalsha_with_reload::<Value, _, _>(
                tv.client().pool.next(),
                vec![
                    carrier.clone(),
                    keys.obj_header(OID),
                    keys.obj_ext_index(OID),
                ],
                Vec::<String>::new(),
            )
            .await
            .expect_err("the older carrier claim must fail on a HASH carrier");
        assert!(
            claim.details().contains("WRONGTYPE"),
            "live={live}: {claim:?}"
        );
        let defer = LEGACY_DEFER
            .evalsha_with_reload::<Value, _, _>(
                tv.client().pool.next(),
                vec![carrier.clone(), keys.obj_header(OID)],
                vec![m.clone()],
            )
            .await
            .expect_err("the older deferral must fail on a HASH carrier");
        assert!(
            defer.details().contains("WRONGTYPE"),
            "live={live}: {defer:?}"
        );

        let fields: Vec<String> = tv.client().pool.hkeys(&carrier).await.expect("HKEYS");
        assert_eq!(
            fields,
            vec![m.clone()],
            "live={live}: the carrier is untouched"
        );
        let tokens_now: Vec<String> = tv.client().pool.hkeys(&attempts).await.expect("HKEYS");
        assert_eq!(tokens_now, tokens, "live={live}: the attempt token stays");
        assert!(
            !in_pending(&tv, &m).await,
            "live={live}: nothing re-enqueued"
        );
        let flag: Option<String> = tv
            .client()
            .pool
            .hget(keys.obj_header(OID), "ext_gc_scheduled")
            .await
            .expect("HGET flag");
        assert_eq!(
            flag.as_deref(),
            live.then_some("1"),
            "live={live}: the older claim's cancel never cleared the flag, and the older \
             deferral flagged no torn-down header"
        );
    }
    assert_eq!(ref_count(&tv, h).await, Some(1), "no second decrement");
}

// ═════════════════════════════════════════════════════════════════════════════
// T4 — un-aged members (an older build's bare `SADD`)
// ═════════════════════════════════════════════════════════════════════════════

async fn bare_sadd(tv: &TestValkey, m: &str) {
    let _: () = tv
        .client()
        .pool
        .sadd(tv.key_schema().ext_pending(), m)
        .await
        .expect("bare SADD (an older build's enqueue)");
}

/// A member an older build enqueued with a bare `SADD` — no age entry — is
/// never collected un-aged: the first pass stamps it with the store's time
/// (counted on `live_edit_gc_older_build_total{event="release_stamped"}`) and
/// collects nothing, and only a later pass, past the age floor from the stamp
/// (0 here), collects it.
///
/// **On the build before b.1jt** the first pass collected it at once.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_unaged_member_is_stamped_first_and_collected_only_on_a_later_pass() {
    let tv = TestValkey::setup().await;
    let m_ = metrics();
    let s = store(&tv, &m_);
    let h = hash(0x31);
    let m = member(h, OBJ);
    put(&s, h, 64).await;
    bare_sadd(&tv, &m).await;

    let before = valkey_now_ms(&tv).await;
    let first = pass(&s, &GcPolicy::IMMEDIATE).await;
    assert_eq!(
        (first.stamped, first.processed),
        (1, 0),
        "stamped, not collected"
    );
    assert!(in_pending(&tv, &m).await);
    let stamp = age_of(&tv, &m).await.expect("the member now has an age") as u64;
    assert!(stamp >= before, "the stamp is the store's current time");
    assert_eq!(older_build_events(&m_, "release_stamped"), 1);
    assert_eq!(ref_count(&tv, h).await, Some(1));

    let second = pass(&s, &GcPolicy::IMMEDIATE).await;
    assert_eq!(
        (second.processed, second.deleted, second.stamped),
        (1, 1, 0)
    );
    assert!(!data_present(&tv, h).await);
}

/// A stamped member waits out the age floor G, measured from its stamp, like
/// a member this build enqueued: not collected before the stamp is G old, and
/// collected after.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_stamped_member_waits_out_the_age_floor_from_its_stamp() {
    const FLOOR: Duration = Duration::from_secs(2);
    let tv = TestValkey::setup().await;
    let m_ = metrics();
    let s = store(&tv, &m_);
    let policy = GcPolicy {
        min_age: FLOOR,
        ..GcPolicy::IMMEDIATE
    };
    let h = hash(0x32);
    let m = member(h, OBJ);
    put(&s, h, 64).await;
    bare_sadd(&tv, &m).await;

    assert_eq!(pass(&s, &policy).await.stamped, 1);
    let stamp = age_of(&tv, &m).await.expect("stamped") as u64;
    let early = pass(&s, &policy).await;
    if valkey_now_ms(&tv).await < stamp + FLOOR.as_millis() as u64 {
        assert_eq!(early.processed, 0, "younger than G from its stamp");
    }

    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while valkey_now_ms(&tv).await <= stamp + FLOOR.as_millis() as u64 {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the store's clock stood still"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let due = pass(&s, &policy).await;
    assert_eq!(
        (due.processed, due.deleted),
        (1, 1),
        "past G from its stamp"
    );
}

/// The read-hold clip measured from the stamp: a hold whose start is after the
/// stamp but within the one-second margin protects the stamped member, and its
/// expiry frees it. Every catch-up that missed a blob takes such a hold, so an
/// older build's release cannot be collected under a reader.
///
/// **On the build before b.1jt** the member was collected by the pass that
/// first saw it, before any hold existed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_hold_started_within_the_margin_after_the_stamp_protects_a_stamped_member() {
    let tv = TestValkey::setup().await;
    let m_ = metrics();
    let s = store(&tv, &m_);
    let h = hash(0x33);
    let m = member(h, OBJ);
    put(&s, h, 64).await;
    bare_sadd(&tv, &m).await;
    assert_eq!(pass(&s, &GcPolicy::IMMEDIATE).await.stamped, 1);
    let stamp = age_of(&tv, &m).await.expect("stamped") as u64;

    // A hold that started 500 ms after the stamp — inside the 1 s margin.
    let hold_key = tv.key_schema().ext_read_hold();
    let expires = valkey_now_ms(&tv).await + 600_000;
    let _: () = tv
        .client()
        .pool
        .hset(
            &hold_key,
            [("b1jt:0", format!("{}:{expires}", stamp + 500))],
        )
        .await
        .expect("plant the read hold");
    let _: bool = tv
        .client()
        .pool
        .pexpire(&hold_key, 600_000, None)
        .await
        .expect("PEXPIRE read_hold");

    let held = pass(&s, &GcPolicy::IMMEDIATE).await;
    assert_eq!(held.processed, 0, "the hold protects the stamped member");
    assert_eq!(ref_count(&tv, h).await, Some(1));

    let _: i64 = tv
        .client()
        .pool
        .del(&hold_key)
        .await
        .expect("DEL read_hold");
    let free = pass(&s, &GcPolicy::IMMEDIATE).await;
    assert_eq!((free.processed, free.deleted), (1, 1));
}

// ═════════════════════════════════════════════════════════════════════════════
// T5 — an older build's stranded claims
// ═════════════════════════════════════════════════════════════════════════════

/// A claim scored below the offset is an older build's, whose record-blind
/// decrement may or may not have applied. Past the decrement fence F this
/// build drops it WITHOUT decrementing: out of `inflight`, its bytes out of
/// the ledger total (an rc.48-style claim keeps a ledger field), settled, and
/// counted on `live_edit_gc_older_build_total{event="claim_dropped"}`. A
/// younger one — its older sweeper may still be working on it — is left
/// exactly as it is. A dropped member that an older build also re-enqueued
/// stays in `pending` with its own ledger entry.
///
/// **On the build before b.1jt** the pass drained a claim past 30 s back to
/// `pending` and decremented it again; one between F and 30 s stayed stuck.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_older_builds_claim_past_the_fence_is_dropped_never_decremented() {
    let tv = TestValkey::setup().await;
    let m_ = metrics();
    let s = store(&tv, &m_);
    let keys = tv.key_schema();
    let now = valkey_now_ms(&tv).await;
    let (past_fence, ancient, young, re_enqueued) =
        (hash(0x41), hash(0x42), hash(0x43), hash(0x44));
    for h in [past_fence, ancient, young, re_enqueued] {
        put(&s, h, 64).await;
        put(&s, h, 64).await;
    }
    let m_past = member(past_fence, OBJ);
    let m_ancient = member(ancient, OBJ);
    let m_young = member(young, OBJ);
    let m_re = member(re_enqueued, OBJ);
    plant_older_claim(&tv, &m_past, now - GC_DECREMENT_FENCE_MS - 1_000).await;
    plant_older_claim(&tv, &m_ancient, 1).await;
    plant_older_claim(&tv, &m_young, now - 5_000).await;
    plant_older_claim(&tv, &m_re, now - GC_DECREMENT_FENCE_MS - 1_000).await;
    bare_sadd(&tv, &m_re).await;
    let _: () = tv
        .client()
        .pool
        .hset(
            keys.ext_pending_bytes(),
            [(m_past.as_str(), "1000"), (m_re.as_str(), "500")],
        )
        .await
        .expect("plant the ledger fields");
    let _: () = tv
        .client()
        .pool
        .set(keys.ext_pending_total(), "1500", None, None, false)
        .await
        .expect("plant the total");
    let young_score = inflight_score(&tv, &m_young).await;

    // An hour's floor: the re-enqueued copy in `pending` is not collected.
    let p = pass(
        &s,
        &GcPolicy {
            min_age: Duration::from_secs(3600),
            ..GcPolicy::IMMEDIATE
        },
    )
    .await;

    assert_eq!(
        p.foreign_dropped, 3,
        "past the fence, ancient and re-enqueued"
    );
    assert_eq!(
        p.processed, 0,
        "nothing an older build claimed is decremented"
    );
    for h in [past_fence, ancient, young, re_enqueued] {
        assert_eq!(ref_count(&tv, h).await, Some(2), "{h:?}: no decrement");
    }
    for m in [&m_past, &m_ancient, &m_re] {
        assert_eq!(inflight_score(&tv, m).await, None, "{m}: dropped");
    }
    assert_eq!(bytes_of(&tv, &m_past).await, None);
    assert!(settled_of(&tv, &m_past).await.is_some());
    assert!(settled_of(&tv, &m_ancient).await.is_some());
    assert!(
        in_pending(&tv, &m_re).await,
        "the re-enqueued copy stays owed"
    );
    assert_eq!(bytes_of(&tv, &m_re).await, Some(500), "with its own bytes");
    assert_eq!(settled_of(&tv, &m_re).await, None, "and unsettled");
    assert_eq!(
        total(&tv).await,
        500,
        "the dropped claim's bytes left the total once"
    );
    assert_eq!(
        inflight_score(&tv, &m_young).await,
        young_score,
        "a claim younger than F is left to its sweeper"
    );
    assert_eq!(older_build_events(&m_, "claim_dropped"), 3);
}

/// `gc_drop_foreign_claims.lua` called as the GC pass calls it — its six
/// `{gc}` ledger keys and the `{gc}` migration sentinel as `KEYS[7]` (b.cyu)
/// — with `offset` and fence `F` as given; returns the members it dropped.
async fn drop_foreign(
    tv: &TestValkey,
    offset: u64,
    fence_ms: i64,
    pairs: &[(&str, f64)],
) -> Vec<String> {
    let keys = tv.key_schema();
    let mut argv = vec![offset.to_string(), fence_ms.to_string()];
    for (m, score) in pairs {
        argv.push((*m).to_owned());
        argv.push(format!("{score}"));
    }
    GC_DROP_FOREIGN_CLAIMS_SCRIPT
        .evalsha_with_reload(
            tv.client().maintenance_pool.next(),
            vec![
                keys.ext_inflight(),
                keys.ext_pending(),
                keys.ext_pending_age(),
                keys.ext_pending_bytes(),
                keys.ext_pending_total(),
                keys.ext_settled(),
                keys.migration_sentinel("gc"),
            ],
            argv,
        )
        .await
        .expect("gc_drop_foreign_claims must run")
}

/// The drop's per-pair guards: a pair writes nothing unless the claim's score
/// is still exactly the one the selection read, is below the offset (an older
/// build's bare score, not this build's encoded one), and is at least F old.
/// Each guard is isolated: a changed score with every other condition met; a
/// claim of this build's under a fence so negative that only the offset can
/// refuse it; a bare score exactly at the offset passed, then one above it;
/// and an unchanged bare claim under the offset that is younger than F (1 s
/// old, the production fence). "Nothing" is checked on every key the drop writes — the claim's score,
/// its ledger bytes field, the total, its age entry and `settled`. A pair that
/// qualifies, in the same call as the changed one, is dropped, so the call
/// shape is the one that drops.
///
/// **On the build before b.1jt** there was no drop script: a claim past 30 s,
/// whoever made it, was drained back to `pending`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_foreign_claim_drop_writes_nothing_for_a_pair_any_guard_refuses() {
    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();
    let now = valkey_now_ms(&tv).await;
    let (moved, qualifies, own, boundary, young) = (
        member(hash(0x45), OBJ),
        member(hash(0x46), OBJ),
        member(hash(0x47), OBJ),
        member(hash(0x48), OBJ),
        member(hash(0x49), OBJ),
    );
    let old = (now - GC_DECREMENT_FENCE_MS - 1_000) as f64;
    let young_score = (now - 1_000) as f64;
    plant_older_claim(&tv, &moved, old as u64).await;
    plant_older_claim(&tv, &qualifies, old as u64).await;
    plant_own_claim(&tv, &own, STALE_OWN_CLAIM_AGE).await;
    plant_older_claim(&tv, &boundary, 1_000).await;
    plant_older_claim(&tv, &young, young_score as u64).await;
    let guarded = [
        moved.as_str(),
        own.as_str(),
        boundary.as_str(),
        young.as_str(),
    ];
    let _: () = tv
        .client()
        .pool
        .hset(
            keys.ext_pending_bytes(),
            guarded.iter().map(|m| (*m, "10")).collect::<Vec<_>>(),
        )
        .await
        .expect("plant the ledger fields");
    let _: () = tv
        .client()
        .pool
        .set(keys.ext_pending_total(), "40", None, None, false)
        .await
        .expect("plant the total");
    for m in guarded {
        let _: () = tv
            .client()
            .pool
            .zadd(keys.ext_pending_age(), None, None, false, false, (1.0, m))
            .await
            .expect("plant the age entry");
    }
    let own_score = inflight_score(&tv, &own).await.expect("planted");

    assert_eq!(
        drop_foreign(
            &tv,
            GC_CLAIM_SCORE_OFFSET_MS,
            GC_DECREMENT_FENCE_MS as i64,
            &[(moved.as_str(), old + 1.0), (qualifies.as_str(), old)]
        )
        .await,
        vec![qualifies.clone()],
        "a changed score is refused; the pair beside it, unchanged, is dropped"
    );
    assert_eq!(
        drop_foreign(
            &tv,
            GC_CLAIM_SCORE_OFFSET_MS,
            -2 * GC_CLAIM_SCORE_OFFSET_MS as i64,
            &[(own.as_str(), own_score)]
        )
        .await,
        Vec::<String>::new(),
        "this build's claim, scored at or above the offset, is never dropped"
    );
    assert_eq!(
        drop_foreign(&tv, 1_000, 0, &[(boundary.as_str(), 1_000.0)]).await,
        Vec::<String>::new(),
        "a score exactly at the offset is not below it"
    );
    assert_eq!(
        drop_foreign(
            &tv,
            GC_CLAIM_SCORE_OFFSET_MS,
            GC_DECREMENT_FENCE_MS as i64,
            &[(young.as_str(), young_score)]
        )
        .await,
        Vec::<String>::new(),
        "an older build's claim younger than F is left to its sweeper"
    );

    for (m, score) in [
        (&moved, old),
        (&own, own_score),
        (&boundary, 1_000.0),
        (&young, young_score),
    ] {
        assert_eq!(inflight_score(&tv, m).await, Some(score), "{m}: claim kept");
        assert_eq!(bytes_of(&tv, m).await, Some(10), "{m}: bytes kept");
        assert_eq!(age_of(&tv, m).await, Some(1.0), "{m}: age kept");
        assert_eq!(settled_of(&tv, m).await, None, "{m}: not settled");
    }
    assert_eq!(total(&tv).await, 40, "the total is untouched");

    assert_eq!(
        drop_foreign(&tv, 1_001, 0, &[(boundary.as_str(), 1_000.0)]).await,
        vec![boundary.clone()],
        "control: one millisecond under the offset, the same pair is dropped"
    );
    assert_eq!(total(&tv).await, 30, "and its bytes leave the total");
}

// ═════════════════════════════════════════════════════════════════════════════
// T6 — this build's own stale claims
// ═════════════════════════════════════════════════════════════════════════════

/// A stale claim of this build's is re-claimed IN PLACE and decided in the
/// same pass; no command ever writes it to `pending` — the step an older
/// build's record-blind GC could inherit and decrement a second time. The
/// `MONITOR` feed of the whole pass is the evidence: no `SADD` / `SMOVE` on
/// this test's `pending` key at any point.
///
/// **On the build before b.1jt** the pass's drain `SADD`ed it to `pending`,
/// and only the next pass claimed it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_stale_own_claim_is_reclaimed_in_place_and_never_written_to_pending() {
    let tv = TestValkey::setup().await;
    let m_ = metrics();
    let s = store(&tv, &m_);
    let h = hash(0x51);
    let m = member(h, OBJ);
    put(&s, h, 64).await;
    put(&s, h, 64).await;
    plant_own_claim(&tv, &m, STALE_OWN_CLAIM_AGE).await;
    let pending_key = tv.key_schema().ext_pending();

    let mut tap = MonitorTap::start(&test_valkey_addr()).await;
    let p = pass(&s, &immediate_keeping_records()).await;
    let fence = monitor_fence(tv.prefix());
    let _: bool = tv.client().pool.exists(&fence).await.expect("EXISTS fence");
    let feed = tap.drain_until(&fence).await;

    assert_eq!((p.reclaimed, p.processed, p.deleted), (1, 1, 0));
    assert_eq!(ref_count(&tv, h).await, Some(1), "decremented once");
    let writes: Vec<_> = feed
        .iter()
        .filter(|c| {
            matches!(c.name().as_str(), "SADD" | "SMOVE")
                && c.args.iter().any(|a| a == &pending_key)
        })
        .collect();
    assert!(
        writes.is_empty(),
        "nothing may write this test's pending key during the pass: {writes:?}"
    );
    assert!(
        feed.iter().any(|c| c.name() == "ZADD"
            && c.args.iter().any(|a| a == &tv.key_schema().ext_inflight())
            && c.args.iter().any(|a| a == &m)),
        "precondition: the feed saw the in-place re-claim's ZADD"
    );
}

/// Two passes that re-claim one stale claim at once — two leaders, briefly —
/// are decided by the re-claim script's score check: exactly one gets a claim
/// time, the other 0. The script also refuses an older build's bare-scored
/// claim and a claim whose score changed since it was read.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn of_two_concurrent_reclaims_of_one_claim_exactly_one_wins() {
    let tv = TestValkey::setup_with_pool_size(4).await;
    let m = member(hash(0x52), OBJ);
    plant_own_claim(&tv, &m, STALE_OWN_CLAIM_AGE).await;
    let read = inflight_score(&tv, &m).await.expect("planted");
    let expected = format!("{read}");
    let reclaim = |score: String| {
        let (client, inflight, m) = (
            tv.client().clone(),
            tv.key_schema().ext_inflight(),
            m.clone(),
        );
        async move {
            GC_RECLAIM_STALE_SCRIPT
                .evalsha_with_reload::<i64, _, _>(
                    client.pool.next(),
                    vec![inflight],
                    vec![m, score, GC_CLAIM_SCORE_OFFSET_MS.to_string()],
                )
                .await
                .expect("gc_reclaim_stale must run")
        }
    };
    let barrier = Arc::new(tokio::sync::Barrier::new(2));
    let racers: Vec<_> = (0..2)
        .map(|_| {
            let (barrier, run) = (Arc::clone(&barrier), reclaim(expected.clone()));
            tokio::spawn(async move {
                barrier.wait().await;
                run.await
            })
        })
        .collect();
    let mut wins = 0;
    for racer in racers {
        if racer.await.expect("racer must not panic") > 0 {
            wins += 1;
        }
    }
    assert_eq!(wins, 1, "exactly one re-claim wins");
    let rescored = inflight_score(&tv, &m).await.expect("still claimed");
    assert!(rescored > read && rescored >= GC_CLAIM_SCORE_OFFSET_MS as f64);

    assert_eq!(
        reclaim(expected).await,
        0,
        "a changed score is not re-claimed"
    );
    let older = member(hash(0x53), OBJ);
    plant_older_claim(&tv, &older, 1).await;
    assert_eq!(
        GC_RECLAIM_STALE_SCRIPT
            .evalsha_with_reload::<i64, _, _>(
                tv.client().pool.next(),
                vec![tv.key_schema().ext_inflight()],
                vec![
                    older.clone(),
                    "1".to_owned(),
                    GC_CLAIM_SCORE_OFFSET_MS.to_string()
                ],
            )
            .await
            .expect("gc_reclaim_stale must run"),
        0,
        "an older build's claim is never re-claimed"
    );
    assert_eq!(inflight_score(&tv, &older).await, Some(1.0));
}

// ═════════════════════════════════════════════════════════════════════════════
// T7 — the enqueue refuses a replay
// ═════════════════════════════════════════════════════════════════════════════

/// `ext_release_enqueue.lua` as the plain enqueue calls it: `{ added, refused }`.
async fn enqueue(tv: &TestValkey, pairs: &[(&str, &str)]) -> (i64, i64) {
    let keys = tv.key_schema();
    let argv: Vec<String> = pairs
        .iter()
        .flat_map(|(m, b)| [(*m).to_owned(), (*b).to_owned()])
        .collect();
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
        .expect("ext_release_enqueue must run")
}

/// A member that is not in `pending` but is in `inflight` (claimed), in
/// `settled` (decided, within the replay window) or still holds a ledger bytes
/// field (accounted) was enqueued once already: the enqueue refuses it and
/// writes nothing for it — no `pending`, no age, no bytes, no total. A fresh
/// member in the same call is enqueued, and a member still in `pending` is
/// the idempotent replay it always was.
///
/// **On the build before b.1jt** the enqueue `SADD`ed every one of them back
/// to `pending`, for a second decrement.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_enqueue_refuses_a_member_claimed_settled_or_accounted_and_writes_nothing() {
    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();
    let (claimed, settled, accounted, fresh) = (
        member(hash(0x61), OBJ),
        member(hash(0x62), OBJ),
        member(hash(0x63), OBJ),
        member(hash(0x64), OBJ),
    );
    plant_own_claim(&tv, &claimed, Duration::ZERO).await;
    let _: () = tv
        .client()
        .pool
        .zadd(
            keys.ext_settled(),
            None,
            None,
            false,
            false,
            (1.0, settled.as_str()),
        )
        .await
        .expect("plant settled");
    let _: () = tv
        .client()
        .pool
        .hset(keys.ext_pending_bytes(), [(accounted.as_str(), "7")])
        .await
        .expect("plant the bytes field");
    let _: () = tv
        .client()
        .pool
        .set(keys.ext_pending_total(), "7", None, None, false)
        .await
        .expect("plant the total");

    let reply = enqueue(
        &tv,
        &[
            (&claimed, "100"),
            (&settled, "100"),
            (&accounted, "100"),
            (&fresh, "100"),
        ],
    )
    .await;
    assert_eq!(reply, (1, 3), "one added, three refused");
    for m in [&claimed, &settled, &accounted] {
        assert!(!in_pending(&tv, m).await, "{m}: not re-enqueued");
        assert_eq!(age_of(&tv, m).await, None, "{m}: no age");
    }
    assert_eq!(bytes_of(&tv, &claimed).await, None);
    assert_eq!(bytes_of(&tv, &settled).await, None);
    assert_eq!(
        bytes_of(&tv, &accounted).await,
        Some(7),
        "its field is not overwritten"
    );
    assert!(in_pending(&tv, &fresh).await);
    assert_eq!(
        total(&tv).await,
        107,
        "only the fresh member's bytes were added"
    );

    assert_eq!(
        enqueue(&tv, &[(&fresh, "100")]).await,
        (0, 0),
        "a member still in pending is an idempotent replay, not a refusal"
    );
    assert_eq!(total(&tv).await, 107);
}

/// The refusal in attempt mode — a retirement's attempt, or the orphan sweep's
/// re-drive of a carrier — when every member is refused: nothing is enqueued,
/// the reply is `{ "enqueued", 0, n }`, and the attempt is still recorded
/// `landed:<ms>:<oid>`, because it ran and each refused member was already
/// owed through its earlier enqueue. A re-send of the same attempt is then
/// `already`. Without the record the sweep would read the attempt as never
/// run: it would resolve it void and settle the carrier again, or re-drive it
/// every tick.
///
/// **On the build before b.1jt** the attempt enqueued both members again
/// (`{ "enqueued", 2 }`).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_all_refused_attempt_enqueues_nothing_and_still_records_the_attempt_landed() {
    const TOKEN: &str = "0123456789abcdef0123456789abcdef";
    const OID: &str = "79002";
    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();
    let (claimed, settled) = (member(hash(0x67), 79_002), member(hash(0x68), 79_002));
    plant_own_claim(&tv, &claimed, Duration::ZERO).await;
    let _: () = tv
        .client()
        .pool
        .zadd(
            keys.ext_settled(),
            None,
            None,
            false,
            false,
            (1.0, settled.as_str()),
        )
        .await
        .expect("plant settled");
    let attempt = || {
        EXT_RELEASE_ENQUEUE_SCRIPT.evalsha_with_reload::<(String, i64, i64), _, _>(
            tv.client().pool.next(),
            vec![
                keys.ext_pending(),
                keys.ext_pending_age(),
                keys.ext_pending_bytes(),
                keys.ext_pending_total(),
                keys.ext_inflight(),
                keys.ext_settled(),
                keys.ext_enqueue_attempts(),
            ],
            vec![
                "@attempt".to_owned(),
                TOKEN.to_owned(),
                OID.to_owned(),
                claimed.clone(),
                "100".to_owned(),
                settled.clone(),
                "100".to_owned(),
            ],
        )
    };

    assert_eq!(
        attempt().await.expect("ext_release_enqueue must run"),
        ("enqueued".to_owned(), 0, 2),
        "it ran: nothing added, both refused"
    );
    for m in [&claimed, &settled] {
        assert!(!in_pending(&tv, m).await, "{m}: not enqueued");
        assert_eq!(age_of(&tv, m).await, None, "{m}: no age");
        assert_eq!(bytes_of(&tv, m).await, None, "{m}: no bytes");
    }
    assert_eq!(total(&tv).await, 0);
    let record: Option<String> = tv
        .client()
        .pool
        .hget(keys.ext_enqueue_attempts(), TOKEN)
        .await
        .expect("HGET the {gc} record");
    let record = record.expect("the attempt is recorded although nothing was enqueued");
    assert!(
        record.starts_with("landed:") && record.ends_with(&format!(":{OID}")),
        "recorded landed for its object: {record}"
    );

    assert_eq!(
        attempt().await.expect("ext_release_enqueue must run"),
        ("already".to_owned(), 0, 0),
        "a re-send of the landed attempt"
    );
}

/// End to end: this build enqueues a release, an older build's sweep — still
/// running after the takeover — collects it (claim, record-blind decrement,
/// bare `ZREM`), and then a fred re-send of the original enqueue arrives. It
/// is refused (the member still holds its ledger bytes field), so the release
/// is decremented exactly once; this build's next passes settle the leftover
/// ledger entries without decrementing.
///
/// **On the build before b.1jt** the re-sent enqueue put the member back in
/// `pending` and this build's GC decremented it a second time — the blob
/// deleted under its other holder.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_resent_enqueue_after_an_older_sweep_collected_the_member_is_refused() {
    let tv = TestValkey::setup().await;
    let m_ = metrics();
    let s = store(&tv, &m_);
    let h = hash(0x65);
    let m = member(h, OBJ);
    put(&s, h, 64).await;
    put(&s, h, 64).await;
    release(&s, h, OBJ).await;

    assert_eq!(
        legacy_gc_pass(&tv).await,
        vec![m.clone()],
        "the older sweep collected it"
    );
    assert_eq!(ref_count(&tv, h).await, Some(1));

    release(&s, h, OBJ).await;
    assert!(!in_pending(&tv, &m).await, "the re-send is refused");
    let stats = process_pending_gc(&s, Some(&m_), GcPolicy::IMMEDIATE).await;
    assert_eq!(stats.processed, 0);
    assert_eq!(ref_count(&tv, h).await, Some(1), "exactly one decrement");
    assert!(data_present(&tv, h).await);
    assert_eq!(
        bytes_of(&tv, &m).await,
        None,
        "the leftover ledger entry settled"
    );
    assert_eq!(total(&tv).await, 0);
}

/// The plain enqueue's refusal is not a failure: `schedule_release` of a
/// member this build already collected (settled, inside the replay window)
/// succeeds and enqueues nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_resent_enqueue_of_a_settled_member_succeeds_and_enqueues_nothing() {
    let tv = TestValkey::setup().await;
    let m_ = metrics();
    let s = store(&tv, &m_);
    let h = hash(0x66);
    let m = member(h, OBJ);
    put(&s, h, 64).await;
    put(&s, h, 64).await;
    release(&s, h, OBJ).await;
    assert_eq!(pass(&s, &immediate_keeping_records()).await.processed, 1);
    assert!(settled_of(&tv, &m).await.is_some());

    release(&s, h, OBJ).await;
    assert!(!in_pending(&tv, &m).await);
    assert_eq!(pass(&s, &immediate_keeping_records()).await.processed, 0);
    assert_eq!(ref_count(&tv, h).await, Some(1));
}

// ═════════════════════════════════════════════════════════════════════════════
// T8 — the ledger after an older leader
// ═════════════════════════════════════════════════════════════════════════════

/// While an older build leads, its bare-`ZREM` completions never lower the
/// ledger total, so it only rises. Once this build's GC loop takes the lock
/// over — at once, from an unmarked holder, counted on
/// `live_edit_gc_older_build_total{event="lock_taken_over"}` — its first pass
/// audits the ledger, and the total equals the bytes of the members still in
/// `pending ∪ inflight`.
///
/// **On the build before b.1jt** this build's loop could not take a live
/// unmarked lock over, so while the older holder kept it the total stayed
/// inflated: uploads carrying external values wait at the budget.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_takeover_corrects_the_total_an_older_leaders_bare_completions_left() {
    const BYTES: usize = 1_000;
    let tv = TestValkey::setup().await;
    let m_ = metrics();
    let s = Arc::new(store(&tv, &m_));
    let hs = [hash(0x71), hash(0x72), hash(0x73)];
    for h in hs {
        put(&s, h, BYTES).await;
        release(&s, h, OBJ).await;
    }
    assert_eq!(total(&tv).await, 3 * BYTES as i64, "precondition");

    // The older build leads and collects two of them.
    plant_lock(&tv, LEGACY_LOCK_VALUE, Some(60_000)).await;
    for h in &hs[..2] {
        let m = member(*h, OBJ);
        assert_eq!(legacy_claim(&tv, &m).await, 1);
        legacy_decrement(&tv, *h).await;
        let _: () = tv
            .client()
            .pool
            .zrem(tv.key_schema().ext_inflight(), m.as_str())
            .await
            .expect("bare ZREM");
    }
    assert_eq!(
        total(&tv).await,
        3 * BYTES as i64,
        "the older completions lowered nothing"
    );

    // This build's loop; an hour's floor so it collects nothing itself.
    let cancel = CancellationToken::new();
    let handle = spawn_gc_task(
        s.clone() as Arc<dyn ExternalValueStore>,
        tv.client().clone(),
        tv.key_schema().clone(),
        this_build("b1jt-ledger"),
        GcPolicy {
            min_age: Duration::from_secs(3600),
            ..GcPolicy::PRODUCTION
        },
        cancel.clone(),
        Some(Arc::clone(&m_)),
    );
    let deadline = tokio::time::Instant::now() + Duration::from_secs(15);
    while total(&tv).await != BYTES as i64 {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the total was never corrected: {} (lock {:?})",
            total(&tv).await,
            lock_value(&tv).await
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    cancel.cancel();
    tokio::time::timeout(Duration::from_secs(5), handle)
        .await
        .expect("the loop stops on cancel")
        .expect("the loop must not panic");

    assert_eq!(
        lock_value(&tv).await,
        Some(this_build("b1jt-ledger").gc_lock_value())
    );
    assert_eq!(older_build_events(&m_, "lock_taken_over"), 1);
    assert!(
        in_pending(&tv, &member(hs[2], OBJ)).await,
        "the uncollected one is still owed"
    );
    assert_eq!(ref_count(&tv, hs[2]).await, Some(1));
}

// ═════════════════════════════════════════════════════════════════════════════
// Guard — who may write `pending` and `inflight`
// ═════════════════════════════════════════════════════════════════════════════

/// The writes a Lua script makes to the GC `pending` set (`SADD`, `SMOVE` into
/// it) and to `inflight` scores (`ZADD`, `ZINCRBY`), found by the local each
/// script binds its `KEYS` slot to: a name containing `pending` (and none of
/// `age`, `bytes`, `total`) is the pending set, one containing `inflight` the
/// claim set. A write the scanner cannot classify this way is counted apart:
/// one whose key is a literal `KEYS[n]` in any script, and, in a `{gc}`
/// script (one that binds a `pending` or `inflight` local), one whose key is
/// anything but a `KEYS` local. Line comments are skipped (the shared
/// [`lua_code`] scanner). Returns
/// `(pending_writes, inflight_writes, unclassified_writes)`.
fn gc_set_writes(script: &str) -> (usize, usize, usize) {
    let code: Vec<String> = lua_code(script);
    let mut key_vars = Vec::new();
    let mut pending_vars = Vec::new();
    let mut inflight_vars = Vec::new();
    for line in &code {
        let Some(rest) = line.trim().strip_prefix("local ") else {
            continue;
        };
        let Some((name, value)) = rest.split_once('=') else {
            continue;
        };
        let (name, value) = (name.trim(), value.trim());
        if !value.starts_with("KEYS[") {
            continue;
        }
        key_vars.push(name.to_owned());
        if name.contains("inflight") {
            inflight_vars.push(name.to_owned());
        } else if name.contains("pending")
            && !["age", "bytes", "total"].iter().any(|x| name.contains(x))
        {
            pending_vars.push(name.to_owned());
        }
    }
    let call_args = |line: &str, command: &str| -> Option<Vec<String>> {
        let (at, needle) = ["redis.call(", "redis.pcall("]
            .iter()
            .flat_map(|call| ['"', '\''].map(|q| format!("{call}{q}{command}{q},")))
            .find_map(|needle| line.find(&needle).map(|at| (at, needle)))?;
        let start = at + needle.len();
        let args = &line[start..];
        let end = args.find(')').unwrap_or(args.len());
        Some(
            args[..end]
                .split(',')
                .map(|a| a.trim().to_owned())
                .collect(),
        )
    };
    let gc_script = key_vars
        .iter()
        .any(|v| v.contains("pending") || v.contains("inflight"));
    let unclassified =
        |key: &str| key.starts_with("KEYS[") || (gc_script && !key_vars.iter().any(|v| v == key));
    let mut pending_writes = 0;
    let mut inflight_writes = 0;
    let mut unclassified_writes = 0;
    for line in &code {
        // (command, the position of the key it writes)
        for (command, at) in [("SADD", 0), ("SMOVE", 1), ("ZADD", 0), ("ZINCRBY", 0)] {
            let Some(args) = call_args(line, command) else {
                continue;
            };
            let Some(key) = args.get(at) else {
                unclassified_writes += 1;
                continue;
            };
            if unclassified(key) {
                unclassified_writes += 1;
            } else if matches!(command, "SADD" | "SMOVE") {
                pending_writes += usize::from(pending_vars.contains(key));
            } else {
                inflight_writes += usize::from(inflight_vars.contains(key));
            }
        }
    }
    (pending_writes, inflight_writes, unclassified_writes)
}

/// INV-2/INV-3 (b.1jt), as a source rule: among the scripts the service runs,
/// only `ext_release_enqueue.lua` adds a member to the GC `pending` set, and
/// only the claim (`gc_claim_pending.lua`) and the in-place re-claim
/// (`gc_reclaim_stale.lua`) write an `inflight` score. A new writer of
/// `pending` could hand an older build's record-blind GC a member this build
/// may have decremented; a new writer of an `inflight` score could write a
/// bare score an older drain re-queues, or one this build reads as an older
/// build's. And no Rust code under `src/` (outside test modules) `SADD`s the
/// pending key or `ZADD`s the inflight key directly.
#[test]
fn only_the_release_enqueue_adds_to_pending_and_only_the_claim_scripts_score_inflight() {
    let scripts = compiled_scripts();
    assert!(
        scripts.iter().any(|(n, _)| n == "gc_claim_pending.lua") && scripts.len() > 20,
        "the scan must see the compiled scripts: {:?}",
        scripts.iter().map(|(n, _)| n).collect::<Vec<_>>()
    );
    let mut pending_writers = Vec::new();
    let mut inflight_writers = Vec::new();
    let mut unclassified_writers = Vec::new();
    for (name, text) in &scripts {
        let (pending, inflight, unclassified) = gc_set_writes(text);
        if pending > 0 {
            pending_writers.push(name.clone());
        }
        if inflight > 0 {
            inflight_writers.push(name.clone());
        }
        if unclassified > 0 {
            unclassified_writers.push(name.clone());
        }
    }
    assert_eq!(
        pending_writers,
        vec!["ext_release_enqueue.lua".to_owned()],
        "only the release enqueue may add to le:ext:{{gc}}:pending"
    );
    assert_eq!(
        inflight_writers,
        vec![
            "gc_claim_pending.lua".to_owned(),
            "gc_reclaim_stale.lua".to_owned()
        ],
        "only the claim and the in-place re-claim may write an inflight score"
    );
    assert!(
        unclassified_writers.is_empty(),
        "a set write the scan cannot attribute to a key (a literal KEYS[n], or a key that is \
         not a KEYS local in a {{gc}} script) could be a write to pending or inflight; bind \
         the key to a named local: {unclassified_writers:?}"
    );

    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut direct = Vec::new();
    let mut stack = vec![root.join("src")];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).expect("read src dir") {
            let path = entry.expect("dir entry").path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if path.extension().is_none_or(|e| e != "rs") || path.ends_with("test_support.rs") {
                continue;
            }
            let text = std::fs::read_to_string(&path).expect("read source");
            let production = text.split("#[cfg(test)]\nmod tests").next().unwrap_or("");
            for call in direct_gc_set_writes(production) {
                direct.push(format!("{}: {call}", path.display()));
            }
        }
    }
    assert!(
        direct.is_empty(),
        "no Rust code may write pending or an inflight score directly: {direct:?}"
    );
}

/// The Rust calls in `source` that write the GC `pending` set or `inflight`
/// scores directly: a fred `.sadd` / `.smove` naming `ext_pending()`, or a
/// `.zadd` / `.zincrby` naming `ext_inflight()`, in its plain form (`(`) or
/// with a turbofish (`::<`), up to the statement's `;`.
fn direct_gc_set_writes(source: &str) -> Vec<String> {
    let mut found = Vec::new();
    for (method, key) in [
        (".sadd", "ext_pending()"),
        (".smove", "ext_pending()"),
        (".zadd", "ext_inflight()"),
        (".zincrby", "ext_inflight()"),
    ] {
        for (i, _) in source.match_indices(method) {
            let after = &source[i + method.len()..];
            if !(after.starts_with('(') || after.starts_with("::<")) {
                continue;
            }
            let call = &source[i..source.len().min(i + 200)];
            let call = &call[..call.find(';').unwrap_or(call.len())];
            if call.contains(key) {
                found.push(call.trim().to_owned());
            }
        }
    }
    found
}

/// The scanner behind
/// [`only_the_release_enqueue_adds_to_pending_and_only_the_claim_scripts_score_inflight`]:
/// it classifies a write by the local its key is bound to, counts `SMOVE`
/// by its destination, ignores the ledger's `pending:*` keys, skips
/// commented-out code, and counts apart a write it cannot classify — so a
/// pass is not the scanner seeing nothing.
#[test]
fn the_write_scanner_classifies_keys_by_their_local_and_skips_comments() {
    let script = r#"
local pending_key  = KEYS[1]
local inflight_key = KEYS[2]
local age_key      = KEYS[3]
local pending_bytes = KEYS[4]
redis.call("SADD", pending_key, member)
redis.call("SMOVE", other_key, pending_key, member)
redis.call("ZADD", inflight_key, now_ms, member)
redis.pcall('ZINCRBY', inflight_key, 1, member)
redis.call("ZADD", age_key, "NX", now_ms, member)
redis.call("SADD", pending_bytes, member)
-- redis.call("SADD", pending_key, member)
x = 1 -- redis.call("ZADD", inflight_key, 0, member)
"#;
    assert_eq!(gc_set_writes(script), (2, 2, 0));
    assert_eq!(
        gc_set_writes("local p = KEYS[1]\nredis.call(\"SADD\", p, m)"),
        (0, 0, 0),
        "a non-{{gc}} script's set write through a KEYS local"
    );
    assert_eq!(
        gc_set_writes("local p = KEYS[1]\nredis.call(\"SADD\", KEYS[1], m)"),
        (0, 0, 1),
        "a literal KEYS[n] key, in any script"
    );
    assert_eq!(
        gc_set_writes(
            "local pending_key = KEYS[1]\nlocal k = pending_key\nredis.call(\"SADD\", k, m)"
        ),
        (0, 0, 1),
        "in a {{gc}} script, a key that is not a KEYS local"
    );
}

/// The Rust half of the guard's scan finds a direct write in both call
/// spellings — `.sadd(` and the turbofish `.sadd::<u64, _, _>(` that
/// production already uses elsewhere — and ignores a write to another key or
/// a method that only starts with the same name.
#[test]
fn the_rust_write_scanner_sees_the_turbofish_form() {
    let source = r#"
        let _: u64 = pool.sadd(keys.ext_pending(), member).await?;
        let _ = pool.sadd::<u64, _, _>(keys.ext_pending(), member).await?;
        let _: () = pool.zadd::<(), _, _>(keys.ext_inflight(), None, None, false, false, (1.0, m)).await?;
        let _: i64 = pool.zincrby(keys.ext_inflight(), 1.0, m).await?;
        let _: u64 = pool.smove(keys.ext_inflight(), keys.ext_pending(), m).await?;
        let _: u64 = pool.sadd::<u64, _, _>(&index_key, member).await?;
        let _ = pool.sadd_all(keys.ext_pending(), member);
    "#;
    let found = direct_gc_set_writes(source);
    assert_eq!(found.len(), 5, "{found:#?}");
    assert!(found
        .iter()
        .any(|c| c.starts_with(".sadd::<u64, _, _>(keys.ext_pending()")));
}
