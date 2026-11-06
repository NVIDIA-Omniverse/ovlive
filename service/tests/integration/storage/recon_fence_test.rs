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

//! The checkpoint clean-up's `{recon}` fence scripts, the value-slot
//! post-delete step and the dead-object reap's withdrawal of an object's
//! announcements, each driven through its Rust wrapper against a real Valkey
//! (b.g65, SDD §22.17.5): the contracts the collect's delete rule and every
//! writer's refusal rest on.
//!
//! | Test | Script | What it pins |
//! |---|---|---|
//! | [`rearm_re_arms_only_tombstoned_hashes_and_never_lowers_a_score`] | `recon_ext_rearm` | `ZADD GT` on `:rearm` for tomb members only; a later run never lowers a score; the tombstone is never re-scored |
//! | [`rearm_answers_the_valkey_clock_and_expires_its_key_in_a_day`] | `recon_ext_rearm` | the reply's clock is Valkey's `TIME`; `:rearm` carries a 24 h `PEXPIRE` |
//! | [`rearm_reports_a_hash_collected_inside_the_lookback_and_not_one_before_it`] | `recon_ext_rearm` | the lookback boundary: a `:collected` entry inside the window refuses, one before it does not |
//! | [`rearm_refuses_a_stem_announced_within_the_grace_and_writes_nothing_for_it`] | `recon_ext_rearm` | the stem part: refused within the grace, nothing written for the stem; healed otherwise |
//! | [`commit_licenses_only_at_or_below_the_threshold_and_records_before_the_delete`] | `recon_ext_commit` | `licensed` / `rearmed` / `too_young`; `:collected` written for a license; a re-armed tombstone re-dated `XX GT`; no member created |
//! | [`commit_trims_its_two_horizons_and_expires_collected_in_a_day`] | `recon_ext_commit` | `:collected` kept 24 h, `:rearm` two graces; `PEXPIRE` on `:collected` |
//! | [`settle_retires_dates_and_reports_each_group_as_its_contract_says`] | `recon_ext_settle` | spared, deleted (unless re-armed), not-live, healed and kept |
//! | [`orphan_commit_licenses_only_a_member_no_publish_healed_and_announces_it`] | `recon_orphan_commit` | license and announcement in one run; a member a publish removed is refused |
//! | [`an_orphan_reap_and_a_publish_of_the_same_stem_exclude_each_other_in_either_order`] | `recon_orphan_commit` + `recon_ext_rearm` | N3's two interleavings: writer first → the reap is not licensed; reap first → the publish is refused |
//! | [`tomb_record_pins_the_first_sighting_and_announces_the_stems`] | `recon_tomb_record` | `ZADD NX` on `:tomb`; stems announced, trimmed and expired |
//! | [`redate_moves_existing_members_later_only`] | `recon_redate` | `XX` never creates, `GT` never lowers, the Valkey clock |
//! | [`clear_writes_only_on_an_existing_meta_and_withdraws_the_confirmation`] | `ext_backup_clear` | absent meta → nothing written, "not live"; present → `backed_up_in` removed, `backup_cleared_ms` stamped, "live" |
//! | [`confirm_declines_a_reliance_start_at_or_before_the_clear_and_accepts_a_later_one`] | `ext_backup_confirm` | the N6 comparison, and an older caller's empty `ARGV[3]` behaving as before |
//! | [`commit_more_than_one_grace_past_t0_licenses_nothing_and_writes_nothing`] | `recon_ext_commit` | the script-local `t0` bound: past it every verdict is `too_young`, with no `:collected` write, no re-date and no `:rearm` trim |
//! | [`commit_within_one_grace_of_t0_still_licenses`] | `recon_ext_commit` | inside the `t0` bound the commit licenses as before |
//! | [`settle_keeps_a_spared_hash_with_a_delete_record_until_it_is_settled_healed`] | `recon_ext_settle` | a spared hash with a `:collected` entry keeps its tombstone and re-arm until a run settles it `healed` |
//! | [`rearm_refused_for_a_collected_hash_leaves_the_stems_entries_in_place`] | `recon_ext_rearm` | a publish refused for a collected hash heals neither the stem's orphan member nor its announcement |
//! | [`rearm_refused_in_an_earlier_run_does_not_heal_the_stem_in_the_last_run`] | `recon_ext_rearm` (wrapper) | above `RECON_FENCE_CHUNK` hashes, a refusal in an earlier run keeps the last run from healing the stem |
//! | [`withdrawing_an_objects_announcements_removes_only_that_objects_stems`] | `withdraw_object_announcements` (`ZSCAN` + `ZREM`) | object `12`'s stems go; `123.*` and `1234.*` keep their scores |
//!
//! Every test runs in its own `TestValkey` namespace; the keys are cleaned up
//! by its `Drop`.

use std::time::Duration;

use fred::interfaces::{HashesInterface, KeysInterface, SortedSetsInterface};

use live_edit::storage::ext_backup::{
    clear_ext_backup, confirm_ext_backup, valkey_time_ms, BACKED_UP_IN_FIELD,
    BACKUP_CLEARED_MS_FIELD,
};
use live_edit::storage::recon_fence::{
    announced_stems_of_object, recon_ext_commit, recon_ext_rearm, recon_ext_settle,
    recon_orphan_commit, recon_redate, recon_tomb_record, withdraw_object_announcements,
    CommitVerdict, SettleGroups, RECON_FENCE_CHUNK,
};
use live_edit::storage::test_support::TestValkey;

/// A grace the tests hand the scripts: a minute, the production floor.
const GRACE: Duration = Duration::from_secs(60);

/// 24 h, the `PEXPIRE` and `:collected` horizon.
const DAY_MS: i64 = 86_400_000;

async fn now_ms(tv: &TestValkey) -> u64 {
    valkey_time_ms(tv.client())
        .await
        .expect("TIME must succeed")
}

async fn zadd(tv: &TestValkey, key: &str, score_ms: u64, member: &str) {
    let _: () = tv
        .client()
        .pool
        .zadd(key, None, None, false, false, (score_ms as f64, member))
        .await
        .expect("ZADD must succeed");
}

async fn zscore(tv: &TestValkey, key: &str, member: &str) -> Option<f64> {
    tv.client()
        .pool
        .zscore(key, member)
        .await
        .expect("ZSCORE must succeed")
}

async fn pttl(tv: &TestValkey, key: &str) -> i64 {
    tv.client().pool.pttl(key).await.expect("PTTL must succeed")
}

fn owned(hashes: &[&str]) -> Vec<String> {
    hashes.iter().map(|h| (*h).to_owned()).collect()
}

/// A writer re-arms only the hashes on `:tomb` (an untombstoned hash cannot
/// be collected), with `ZADD GT`, so a run whose clock reads earlier — a
/// planted later score here — never lowers a re-arm; and the tombstone's own
/// score is never touched.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rearm_re_arms_only_tombstoned_hashes_and_never_lowers_a_score() {
    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();
    zadd(
        &tv,
        &keys.checkpoint_reconcile_tombstones(),
        7,
        "tombstoned",
    )
    .await;
    let before = now_ms(&tv).await;

    let reply = recon_ext_rearm(
        tv.client(),
        keys,
        &owned(&["tombstoned", "untombstoned"]),
        Duration::from_secs(60),
        GRACE,
        None,
    )
    .await
    .expect("re-arm");

    assert!(!reply.refuses(), "{reply:?}");
    let rearm = keys.checkpoint_reconcile_rearm();
    let score = zscore(&tv, &rearm, "tombstoned")
        .await
        .expect("a tombstoned hash is re-armed");
    assert!(score >= before as f64, "at the Valkey clock");
    assert_eq!(
        zscore(&tv, &rearm, "untombstoned").await,
        None,
        "an untombstoned hash is not re-armed"
    );
    assert_eq!(
        zscore(&tv, &keys.checkpoint_reconcile_tombstones(), "tombstoned").await,
        Some(7.0),
        "the tombstone is never re-scored by a writer"
    );

    let later = reply.now_ms + 3_600_000;
    zadd(&tv, &rearm, later, "tombstoned").await;
    recon_ext_rearm(
        tv.client(),
        keys,
        &owned(&["tombstoned"]),
        Duration::from_secs(60),
        GRACE,
        None,
    )
    .await
    .expect("re-arm");
    assert_eq!(
        zscore(&tv, &rearm, "tombstoned").await,
        Some(later as f64),
        "`GT`: a run never lowers a re-arm"
    );
}

/// The re-arm's reply carries the Valkey clock of its run (the collect's
/// snapshot clock reads it), and a run that re-armed sets `:rearm`'s 24 h
/// `PEXPIRE`, so a rollback leaves nothing behind.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rearm_answers_the_valkey_clock_and_expires_its_key_in_a_day() {
    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();
    zadd(&tv, &keys.checkpoint_reconcile_tombstones(), 1, "h").await;

    let before = now_ms(&tv).await;
    let reply = recon_ext_rearm(
        tv.client(),
        keys,
        &owned(&["h"]),
        Duration::ZERO,
        GRACE,
        None,
    )
    .await
    .expect("re-arm");
    let after = now_ms(&tv).await;

    assert!(
        (before..=after).contains(&reply.now_ms),
        "the reply's clock is Valkey's TIME: {before} <= {} <= {after}",
        reply.now_ms
    );
    let ttl = pttl(&tv, &keys.checkpoint_reconcile_rearm()).await;
    assert!(
        ttl > DAY_MS - 60_000 && ttl <= DAY_MS,
        ":rearm expires in 24 h; PTTL {ttl}"
    );
}

/// R7's window: a hash with a `:collected` entry at or after `now − lookback`
/// is returned (the writer refuses); one recorded before the window is not.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rearm_reports_a_hash_collected_inside_the_lookback_and_not_one_before_it() {
    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();
    let now = now_ms(&tv).await;
    let collected = keys.checkpoint_reconcile_collected();
    zadd(&tv, &collected, now - 10_000, "inside").await;
    zadd(&tv, &collected, now - 120_000, "before").await;

    let reply = recon_ext_rearm(
        tv.client(),
        keys,
        &owned(&["inside", "before", "never"]),
        Duration::from_secs(60),
        GRACE,
        None,
    )
    .await
    .expect("re-arm");

    assert_eq!(
        reply.collected,
        vec!["inside".to_owned()],
        "only the entry inside the 60 s lookback refuses"
    );
    assert!(reply.refuses());
}

/// N3's writer half: a stem announced on `:deleting` within the grace
/// refuses, and the run writes nothing for the stem (its `:orphans` member
/// and the announcement stay); an announcement older than the grace does not
/// refuse, and the run removes both the orphan member and the stale
/// announcement.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rearm_refuses_a_stem_announced_within_the_grace_and_writes_nothing_for_it() {
    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();
    let (orphans, deleting) = (
        keys.checkpoint_reconcile_orphans(),
        keys.checkpoint_reconcile_deleting(),
    );
    let now = now_ms(&tv).await;
    zadd(&tv, &orphans, 1, "9.4").await;
    zadd(&tv, &deleting, now, "9.4").await;

    let refused = recon_ext_rearm(tv.client(), keys, &[], Duration::ZERO, GRACE, Some("9.4"))
        .await
        .expect("re-arm");
    assert!(refused.stem_refused && refused.refuses(), "{refused:?}");
    assert!(
        zscore(&tv, &orphans, "9.4").await.is_some(),
        "the orphan member stays"
    );
    assert!(
        zscore(&tv, &deleting, "9.4").await.is_some(),
        "and the announcement"
    );

    zadd(&tv, &deleting, now - 2 * GRACE.as_millis() as u64, "9.4").await;
    let healed = recon_ext_rearm(tv.client(), keys, &[], Duration::ZERO, GRACE, Some("9.4"))
        .await
        .expect("re-arm");
    assert!(!healed.refuses(), "{healed:?}");
    assert_eq!(
        zscore(&tv, &orphans, "9.4").await,
        None,
        "the publish heals its orphan"
    );
    assert_eq!(
        zscore(&tv, &deleting, "9.4").await,
        None,
        "and removes the stale announcement"
    );
}

/// R3: the delete-time fence licenses a hash only when its tombstone AND its
/// latest re-arm are at or below the threshold, recording it on `:collected`
/// at the run's clock; a hash re-armed after the threshold is refused and its
/// tombstone re-dated (`XX GT`, to the run's clock); a tombstone newer than the
/// threshold — or no longer a member — is "too young" and nothing is written
/// for it (no tombstone is created).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_licenses_only_at_or_below_the_threshold_and_records_before_the_delete() {
    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();
    let (tomb, rearm, collected) = (
        keys.checkpoint_reconcile_tombstones(),
        keys.checkpoint_reconcile_rearm(),
        keys.checkpoint_reconcile_collected(),
    );
    let now = now_ms(&tv).await;
    let threshold = now - 60_000;
    zadd(&tv, &tomb, threshold, "at-threshold").await;
    zadd(&tv, &tomb, threshold - 1, "re-armed-before").await;
    zadd(&tv, &rearm, threshold, "re-armed-before").await;
    zadd(&tv, &tomb, threshold - 1, "re-armed-after").await;
    zadd(&tv, &rearm, threshold + 1, "re-armed-after").await;
    zadd(&tv, &tomb, threshold + 1, "young").await;

    // `t0` is one grace after the threshold.
    let verdicts = recon_ext_commit(
        tv.client(),
        keys,
        threshold + GRACE.as_millis() as u64,
        GRACE,
        &owned(&[
            "at-threshold",
            "re-armed-before",
            "re-armed-after",
            "young",
            "absent",
        ]),
    )
    .await
    .expect("commit");

    assert_eq!(
        verdicts,
        vec![
            CommitVerdict::Licensed,
            CommitVerdict::Licensed,
            CommitVerdict::Rearmed,
            CommitVerdict::TooYoung,
            CommitVerdict::TooYoung,
        ]
    );
    for licensed in ["at-threshold", "re-armed-before"] {
        assert!(
            zscore(&tv, &collected, licensed)
                .await
                .is_some_and(|s| s >= now as f64),
            "{licensed}: recorded on `:collected` at the run's clock, before any delete"
        );
    }
    assert_eq!(zscore(&tv, &collected, "re-armed-after").await, None);
    assert!(
        zscore(&tv, &tomb, "re-armed-after")
            .await
            .is_some_and(|s| s >= now as f64),
        "a refused, re-armed tombstone is re-dated to the run's clock"
    );
    assert_eq!(
        zscore(&tv, &tomb, "young").await,
        Some((threshold + 1) as f64),
        "a too-young tombstone is left as it was"
    );
    assert_eq!(
        zscore(&tv, &tomb, "absent").await,
        None,
        "`XX`: the fence never creates a tombstone"
    );
}

/// R8: a licensing run trims `:collected` to 24 h and sets its 24 h
/// `PEXPIRE`; every run trims `:rearm` to two graces — an older re-arm is at
/// or below any later threshold and can refuse nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_trims_its_two_horizons_and_expires_collected_in_a_day() {
    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();
    let (tomb, rearm, collected) = (
        keys.checkpoint_reconcile_tombstones(),
        keys.checkpoint_reconcile_rearm(),
        keys.checkpoint_reconcile_collected(),
    );
    let now = now_ms(&tv).await;
    let grace_ms = GRACE.as_millis() as u64;
    zadd(
        &tv,
        &collected,
        now - DAY_MS as u64 - 60_000,
        "stale-record",
    )
    .await;
    zadd(&tv, &collected, now - 60_000, "fresh-record").await;
    zadd(&tv, &rearm, now - 2 * grace_ms - 60_000, "stale-rearm").await;
    zadd(&tv, &rearm, now - grace_ms, "fresh-rearm").await;
    zadd(&tv, &tomb, 1, "licensed").await;

    recon_ext_commit(tv.client(), keys, now, GRACE, &owned(&["licensed"]))
        .await
        .expect("commit");

    assert_eq!(
        zscore(&tv, &collected, "stale-record").await,
        None,
        "24 h trim"
    );
    assert!(zscore(&tv, &collected, "fresh-record").await.is_some());
    assert_eq!(
        zscore(&tv, &rearm, "stale-rearm").await,
        None,
        "2 × grace trim"
    );
    assert!(zscore(&tv, &rearm, "fresh-rearm").await.is_some());
    let ttl = pttl(&tv, &collected).await;
    assert!(ttl > DAY_MS - 60_000 && ttl <= DAY_MS, "PTTL {ttl}");
}

/// R6: one settle run retires the spared hashes' tombstone and re-arm, the
/// deleted ones' tombstone unless re-armed after the threshold, re-dates a
/// not-live delete's `:collected` entry to the run's clock (`XX`), drops a
/// healed one's, and returns the kept hashes that still have a `:collected`
/// entry (heal-on-keep's work list).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn settle_retires_dates_and_reports_each_group_as_its_contract_says() {
    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();
    let (tomb, rearm, collected) = (
        keys.checkpoint_reconcile_tombstones(),
        keys.checkpoint_reconcile_rearm(),
        keys.checkpoint_reconcile_collected(),
    );
    let now = now_ms(&tv).await;
    let threshold = now - 60_000;
    for hash in [
        "spared",
        "deleted",
        "deleted-re-armed",
        "kept-with-record",
        "kept",
    ] {
        zadd(&tv, &tomb, 1, hash).await;
    }
    zadd(&tv, &rearm, 1, "spared").await;
    zadd(&tv, &rearm, threshold + 1, "deleted-re-armed").await;
    zadd(&tv, &collected, 5, "not-live").await;
    zadd(&tv, &collected, 5, "healed").await;
    zadd(&tv, &collected, 5, "kept-with-record").await;

    let unsettled = recon_ext_settle(
        tv.client(),
        keys,
        threshold,
        SettleGroups {
            spared: &owned(&["spared"]),
            deleted: &owned(&["deleted", "deleted-re-armed"]),
            not_live: &owned(&["not-live"]),
            healed: &owned(&["healed"]),
            kept: &owned(&["kept-with-record", "kept"]),
        },
    )
    .await
    .expect("settle");

    assert_eq!(unsettled, vec!["kept-with-record".to_owned()]);
    assert_eq!(zscore(&tv, &tomb, "spared").await, None);
    assert_eq!(zscore(&tv, &rearm, "spared").await, None);
    assert_eq!(zscore(&tv, &tomb, "deleted").await, None);
    assert!(
        zscore(&tv, &tomb, "deleted-re-armed").await.is_some(),
        "a deleted hash a writer re-armed after the threshold keeps its tombstone"
    );
    assert!(
        zscore(&tv, &collected, "not-live")
            .await
            .is_some_and(|s| s >= now as f64),
        "a not-live delete's record is re-dated to the settle's clock"
    );
    assert_eq!(zscore(&tv, &collected, "healed").await, None);
    assert!(
        zscore(&tv, &tomb, "kept").await.is_some(),
        "kept is only read"
    );
}

/// N3's reap half: the orphan commit licenses a claimed stem only while its
/// member is still on `:orphans`, announcing it on `:deleting` (trimmed, 24 h
/// `PEXPIRE`) in the same run; a member a publish removed is refused and
/// nothing is written.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn orphan_commit_licenses_only_a_member_no_publish_healed_and_announces_it() {
    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();
    let (orphans, deleting) = (
        keys.checkpoint_reconcile_orphans(),
        keys.checkpoint_reconcile_deleting(),
    );
    let now = now_ms(&tv).await;
    zadd(&tv, &orphans, 1, "5.1").await;
    zadd(
        &tv,
        &deleting,
        now - 2 * GRACE.as_millis() as u64 - 60_000,
        "stale.1",
    )
    .await;

    assert!(recon_orphan_commit(tv.client(), keys, "5.1", GRACE)
        .await
        .expect("commit"));
    assert!(
        zscore(&tv, &deleting, "5.1")
            .await
            .is_some_and(|s| s >= now as f64),
        "announced at the run's clock"
    );
    assert_eq!(
        zscore(&tv, &deleting, "stale.1").await,
        None,
        "2 × grace trim"
    );
    let ttl = pttl(&tv, &deleting).await;
    assert!(ttl > DAY_MS - 60_000 && ttl <= DAY_MS, "PTTL {ttl}");

    assert!(
        !recon_orphan_commit(tv.client(), keys, "6.1", GRACE)
            .await
            .expect("commit"),
        "a member no longer on `:orphans` (a publish healed it) is not licensed"
    );
    assert_eq!(
        zscore(&tv, &deleting, "6.1").await,
        None,
        "and not announced"
    );
}

/// N3, both interleavings of an orphan reap and a re-publish of the SAME stem
/// (a failed publish re-written at the same sequence). Exactly one of the two
/// may act, whichever reaches Valkey first:
///
/// - **writer first:** the publish's re-arm removes the stem's `:orphans`
///   member, so the reap's commit is not licensed and deletes nothing — the
///   re-put generation survives intact;
/// - **reap first:** the commit announces the stem on `:deleting`, so the
///   publish's re-arm refuses (the writer retries after a grace, when the reap
///   has finished) — it never PUTs a generation the reap is deleting.
///
/// Before b.g65 only holding the backup lease kept the two apart.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_orphan_reap_and_a_publish_of_the_same_stem_exclude_each_other_in_either_order() {
    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();
    let orphans = keys.checkpoint_reconcile_orphans();

    // Writer first.
    zadd(&tv, &orphans, 1, "21.3").await;
    let publish = recon_ext_rearm(tv.client(), keys, &[], Duration::ZERO, GRACE, Some("21.3"))
        .await
        .expect("re-arm");
    assert!(!publish.refuses(), "the writer proceeds: {publish:?}");
    assert!(
        !recon_orphan_commit(tv.client(), keys, "21.3", GRACE)
            .await
            .expect("commit"),
        "and the reap that comes after it is not licensed"
    );

    // Reap first.
    zadd(&tv, &orphans, 1, "22.3").await;
    assert!(
        recon_orphan_commit(tv.client(), keys, "22.3", GRACE)
            .await
            .expect("commit"),
        "the reap is licensed"
    );
    let publish = recon_ext_rearm(tv.client(), keys, &[], Duration::ZERO, GRACE, Some("22.3"))
        .await
        .expect("re-arm");
    assert!(
        publish.stem_refused,
        "and the publish that comes after it is refused: {publish:?}"
    );
}

/// The deleters' record: hashes tombstoned with `ZADD NX` (the first sighting
/// is what the claim's grace measures from, so a repeat never moves it) and
/// the stems announced, in one run.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tomb_record_pins_the_first_sighting_and_announces_the_stems() {
    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();
    let (tomb, deleting) = (
        keys.checkpoint_reconcile_tombstones(),
        keys.checkpoint_reconcile_deleting(),
    );
    zadd(&tv, &tomb, 3, "seen-before").await;
    let now = now_ms(&tv).await;

    recon_tomb_record(
        tv.client(),
        keys,
        GRACE,
        &owned(&["40.2"]),
        &owned(&["seen-before", "new"]),
    )
    .await
    .expect("record");

    assert_eq!(zscore(&tv, &tomb, "seen-before").await, Some(3.0), "`NX`");
    assert!(zscore(&tv, &tomb, "new")
        .await
        .is_some_and(|s| s >= now as f64));
    assert!(zscore(&tv, &deleting, "40.2")
        .await
        .is_some_and(|s| s >= now as f64));
    let ttl = pttl(&tv, &deleting).await;
    assert!(ttl > DAY_MS - 60_000 && ttl <= DAY_MS, "PTTL {ttl}");
    assert!(
        pttl(&tv, &tomb).await < 0,
        "`:tomb` carries no TTL: it is the only record of owed reclamation"
    );
}

/// D4's re-date: existing members move to the Valkey clock, a member absent
/// from the list stays absent (`XX`), and a member already later stays later
/// (`GT`).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn redate_moves_existing_members_later_only() {
    let tv = TestValkey::setup().await;
    let dead = tv.key_schema().checkpoint_reconcile_dead();
    let now = now_ms(&tv).await;
    let future = now + 3_600_000;
    zadd(&tv, &dead, 1, "old").await;
    zadd(&tv, &dead, future, "later").await;

    let moved = recon_redate(tv.client(), &dead, &owned(&["old", "later", "absent"]))
        .await
        .expect("re-date");

    assert_eq!(moved, 1, "only `old` moved");
    assert!(zscore(&tv, &dead, "old")
        .await
        .is_some_and(|s| s >= now as f64));
    assert_eq!(
        zscore(&tv, &dead, "later").await,
        Some(future as f64),
        "`GT`"
    );
    assert_eq!(zscore(&tv, &dead, "absent").await, None, "`XX`");
}

/// N6, the clear (R5's post-delete step): on an absent meta it answers "not
/// live" and writes nothing — it never creates a meta; on a present one it
/// removes `backed_up_in`, stamps `backup_cleared_ms` at the Valkey clock and
/// answers "live", leaving the other fields.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn clear_writes_only_on_an_existing_meta_and_withdraws_the_confirmation() {
    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();

    assert!(!clear_ext_backup(tv.client(), keys, "absent")
        .await
        .expect("clear"));
    let created: u64 = tv
        .client()
        .pool
        .exists(keys.ext_meta("absent"))
        .await
        .expect("EXISTS");
    assert_eq!(created, 0, "no meta is created");

    let _: u64 = tv
        .client()
        .pool
        .hset(
            keys.ext_meta("present"),
            vec![
                ("ref_count", "1"),
                ("last_store_ms", "123"),
                (BACKED_UP_IN_FIELD, "0123456789abcdef:100"),
            ],
        )
        .await
        .expect("HSET");
    let before = now_ms(&tv).await;
    assert!(clear_ext_backup(tv.client(), keys, "present")
        .await
        .expect("clear"));
    let fields: std::collections::HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(keys.ext_meta("present"))
        .await
        .expect("HGETALL");
    assert_eq!(
        fields.get(BACKED_UP_IN_FIELD),
        None,
        "the confirmation is withdrawn"
    );
    let cleared: u64 = fields
        .get(BACKUP_CLEARED_MS_FIELD)
        .expect("the clear is stamped")
        .parse()
        .expect("decimal ms");
    assert!(cleared >= before, "at the Valkey clock");
    assert_eq!(fields.get("last_store_ms").map(String::as_str), Some("123"));
    assert_eq!(fields.get("ref_count").map(String::as_str), Some("1"));
}

/// N6, the confirm: a confirmation whose caller started relying on the copy
/// at or before the collect's clear is declined (the copy it relied on was
/// deleted); one that started after it is accepted; and an older caller's
/// empty reliance start confirms as before b.g65.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn confirm_declines_a_reliance_start_at_or_before_the_clear_and_accepts_a_later_one() {
    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();
    let _: u64 = tv
        .client()
        .pool
        .hset(
            keys.ext_meta("h"),
            vec![
                ("ref_count", "1"),
                ("last_store_ms", "123"),
                (BACKUP_CLEARED_MS_FIELD, "5000"),
            ],
        )
        .await
        .expect("HSET");
    let client = tv.client();
    let confirmed = |start: Option<u64>| async move {
        confirm_ext_backup(client, keys, "h", "0123456789abcdef", "123", start)
            .await
            .expect("confirm")
    };

    assert!(
        !confirmed(Some(4_999)).await,
        "a start before the clear is declined"
    );
    assert!(
        !confirmed(Some(5_000)).await,
        "a start AT the clear is declined"
    );
    let unchanged: Option<String> = tv
        .client()
        .pool
        .hget(keys.ext_meta("h"), BACKED_UP_IN_FIELD)
        .await
        .expect("HGET");
    assert_eq!(unchanged, None, "a declined confirm writes nothing");
    assert!(
        confirmed(Some(5_001)).await,
        "a start after the clear is accepted"
    );
    let _: u64 = tv
        .client()
        .pool
        .hdel(keys.ext_meta("h"), BACKED_UP_IN_FIELD)
        .await
        .expect("HDEL");
    assert!(
        confirmed(None).await,
        "an older caller's empty reliance start skips the comparison, as before b.g65"
    );
}

/// R3's own bound (F6): a commit run more than one grace past the collect's
/// `t0` licenses nothing, because by then a peer runner's trim of `:rearm`
/// (entries older than two graces) may already have removed a re-arm newer
/// than the threshold. Every candidate answers `too_young` — the one that
/// would be licensed and the one that would be re-armed alike — and the run
/// writes nothing: no `:collected` entry, no tombstone re-dated, no `:rearm`
/// trim. `t0` is one millisecond past the bound as read before the call, so
/// the script's own clock, read after, is past it too.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_more_than_one_grace_past_t0_licenses_nothing_and_writes_nothing() {
    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();
    let (tomb, rearm, collected) = (
        keys.checkpoint_reconcile_tombstones(),
        keys.checkpoint_reconcile_rearm(),
        keys.checkpoint_reconcile_collected(),
    );
    let grace_ms = GRACE.as_millis() as u64;
    let now = now_ms(&tv).await;
    let t0 = now - grace_ms - 1;
    let threshold = t0 - grace_ms;
    zadd(&tv, &tomb, 1, "would-license").await;
    zadd(&tv, &tomb, 1, "would-re-arm").await;
    zadd(&tv, &rearm, threshold + 1, "would-re-arm").await;
    zadd(&tv, &rearm, now - 2 * grace_ms - 60_000, "trimmable-rearm").await;

    let verdicts = recon_ext_commit(
        tv.client(),
        keys,
        t0,
        GRACE,
        &owned(&["would-license", "would-re-arm"]),
    )
    .await
    .expect("commit");

    assert_eq!(
        verdicts,
        vec![CommitVerdict::TooYoung, CommitVerdict::TooYoung],
        "a run past the bound licenses nothing"
    );
    assert_eq!(
        zscore(&tv, &collected, "would-license").await,
        None,
        "no `:collected` entry is written"
    );
    assert_eq!(
        zscore(&tv, &tomb, "would-license").await,
        Some(1.0),
        "the tombstone is untouched"
    );
    assert_eq!(
        zscore(&tv, &tomb, "would-re-arm").await,
        Some(1.0),
        "a re-armed tombstone is not re-dated either"
    );
    assert!(
        zscore(&tv, &rearm, "trimmable-rearm").await.is_some(),
        "and `:rearm` is not trimmed"
    );
}

/// R3's own bound, the inside of it (F6): a commit run within one grace of
/// `t0` licenses as before. Exactly `now − t0 = grace` cannot be staged —
/// the script reads its own clock after this test's — so `t0` here sits half
/// a grace (30 s) inside the bound: a NOSCRIPT reload or a contended shared
/// Valkey between this test's `TIME` and the script's cannot use that up.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_within_one_grace_of_t0_still_licenses() {
    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();
    let (tomb, collected) = (
        keys.checkpoint_reconcile_tombstones(),
        keys.checkpoint_reconcile_collected(),
    );
    let grace_ms = GRACE.as_millis() as u64;
    let now = now_ms(&tv).await;
    zadd(&tv, &tomb, 1, "licensed").await;

    let verdicts = recon_ext_commit(
        tv.client(),
        keys,
        now - grace_ms / 2,
        GRACE,
        &owned(&["licensed"]),
    )
    .await
    .expect("commit");

    assert_eq!(verdicts, vec![CommitVerdict::Licensed]);
    assert!(
        zscore(&tv, &collected, "licensed")
            .await
            .is_some_and(|s| s >= now as f64),
        "recorded on `:collected`"
    );
}

/// R6 for a spared hash with an unsettled delete record (F3): the settle keeps
/// its tombstone and its re-arm while it has a `:collected` entry and is not in
/// the run's `healed` group — an earlier delete of its backup copy was never
/// settled, and a retained backup names it, so a later claim must serve it
/// again for heal-on-keep — and retires both, dropping the entry, in the run
/// that settles it as `healed`. A spared hash with no record retires at once,
/// as before.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn settle_keeps_a_spared_hash_with_a_delete_record_until_it_is_settled_healed() {
    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();
    let (tomb, rearm, collected) = (
        keys.checkpoint_reconcile_tombstones(),
        keys.checkpoint_reconcile_rearm(),
        keys.checkpoint_reconcile_collected(),
    );
    let threshold = now_ms(&tv).await - 60_000;
    for hash in ["spared-with-record", "spared"] {
        zadd(&tv, &tomb, 1, hash).await;
        zadd(&tv, &rearm, 1, hash).await;
    }
    zadd(&tv, &collected, 5, "spared-with-record").await;

    recon_ext_settle(
        tv.client(),
        keys,
        threshold,
        SettleGroups {
            spared: &owned(&["spared-with-record", "spared"]),
            ..SettleGroups::default()
        },
    )
    .await
    .expect("settle");

    assert_eq!(
        zscore(&tv, &tomb, "spared-with-record").await,
        Some(1.0),
        "not healed: the tombstone stays, so a later claim serves the hash again"
    );
    assert_eq!(
        zscore(&tv, &rearm, "spared-with-record").await,
        Some(1.0),
        "and so does its re-arm"
    );
    assert_eq!(
        zscore(&tv, &collected, "spared-with-record").await,
        Some(5.0),
        "and its delete record"
    );
    assert_eq!(
        zscore(&tv, &tomb, "spared").await,
        None,
        "a spared hash with no record retires"
    );
    assert_eq!(zscore(&tv, &rearm, "spared").await, None);

    recon_ext_settle(
        tv.client(),
        keys,
        threshold,
        SettleGroups {
            spared: &owned(&["spared-with-record"]),
            healed: &owned(&["spared-with-record"]),
            ..SettleGroups::default()
        },
    )
    .await
    .expect("settle");

    assert_eq!(
        zscore(&tv, &tomb, "spared-with-record").await,
        None,
        "healed: the tombstone retires"
    );
    assert_eq!(zscore(&tv, &rearm, "spared-with-record").await, None);
    assert_eq!(
        zscore(&tv, &collected, "spared-with-record").await,
        None,
        "and the record is dropped"
    );
}

/// A refused publish writes nothing for its stem (F13), whichever refusal it
/// is: a hash the collect recorded on `:collected` inside the lookback
/// refuses the publish just as an announced stem does, so the stem's orphan
/// member and its announcement stay — the run heals neither. The
/// announcement is older than the grace, so the stem itself is not what
/// refuses.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rearm_refused_for_a_collected_hash_leaves_the_stems_entries_in_place() {
    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();
    let (orphans, deleting, collected) = (
        keys.checkpoint_reconcile_orphans(),
        keys.checkpoint_reconcile_deleting(),
        keys.checkpoint_reconcile_collected(),
    );
    let now = now_ms(&tv).await;
    let stale_announcement = now - 2 * GRACE.as_millis() as u64;
    zadd(&tv, &orphans, 1, "31.4").await;
    zadd(&tv, &deleting, stale_announcement, "31.4").await;
    zadd(&tv, &collected, now - 1_000, "collected-hash").await;

    let reply = recon_ext_rearm(
        tv.client(),
        keys,
        &owned(&["collected-hash"]),
        Duration::from_secs(60),
        GRACE,
        Some("31.4"),
    )
    .await
    .expect("re-arm");

    assert!(
        reply.refuses() && !reply.stem_refused,
        "fixture: refused for the collected hash, not the stem: {reply:?}"
    );
    assert_eq!(
        zscore(&tv, &orphans, "31.4").await,
        Some(1.0),
        "the stem's orphan member stays"
    );
    assert_eq!(
        zscore(&tv, &deleting, "31.4").await,
        Some(stale_announcement as f64),
        "and so does its announcement"
    );
}

/// The same rule across runs (F13): a publish naming more than
/// [`RECON_FENCE_CHUNK`] hashes goes out in several runs, and when the first
/// run refuses on a collected hash, the stem — which rides the last run — is
/// not healed by it, although that last run finds nothing collected. The
/// control, the same publish with nothing collected, heals the stem.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rearm_refused_in_an_earlier_run_does_not_heal_the_stem_in_the_last_run() {
    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();
    let (orphans, deleting, collected) = (
        keys.checkpoint_reconcile_orphans(),
        keys.checkpoint_reconcile_deleting(),
        keys.checkpoint_reconcile_collected(),
    );
    let now = now_ms(&tv).await;
    let stale_announcement = now - 2 * GRACE.as_millis() as u64;
    // Two runs: the first 256 hashes, then the rest.
    let hashes: Vec<String> = (0..RECON_FENCE_CHUNK + 10)
        .map(|i| format!("chunked-{i:04}"))
        .collect();
    assert!(hashes.len() > RECON_FENCE_CHUNK);

    for (stem, collected_hash) in [("41.4", Some(&hashes[0])), ("42.4", None)] {
        zadd(&tv, &orphans, 1, stem).await;
        zadd(&tv, &deleting, stale_announcement, stem).await;
        if let Some(hash) = collected_hash {
            zadd(&tv, &collected, now - 1_000, hash).await;
        }

        let reply = recon_ext_rearm(
            tv.client(),
            keys,
            &hashes,
            Duration::from_secs(60),
            GRACE,
            Some(stem),
        )
        .await
        .expect("re-arm");

        if collected_hash.is_some() {
            assert_eq!(
                reply.collected,
                vec![hashes[0].clone()],
                "fixture: the first run refuses on the collected hash"
            );
            assert_eq!(
                zscore(&tv, &orphans, stem).await,
                Some(1.0),
                "{stem}: refused, so the stem's orphan member stays"
            );
            assert_eq!(
                zscore(&tv, &deleting, stem).await,
                Some(stale_announcement as f64),
                "{stem}: and its announcement"
            );
            let _: i64 = tv
                .client()
                .pool
                .zrem(&collected, hashes[0].as_str())
                .await
                .expect("ZREM");
        } else {
            assert!(!reply.refuses(), "control: nothing refuses: {reply:?}");
            assert_eq!(
                zscore(&tv, &orphans, stem).await,
                None,
                "control: the publish heals its orphan member"
            );
            assert_eq!(
                zscore(&tv, &deleting, stem).await,
                None,
                "control: and the stale announcement"
            );
        }
    }
}

/// The dead-object reap's withdrawal works from `:deleting` itself, by object
/// (b.g65): it finds every member naming object `12`'s stems — a base's
/// `12.3` and a tail's `12.4.tail` — and removes exactly those. A neighbour
/// whose id starts with the same digits (`123`, `1234`) is a different object,
/// and each of its members keeps its score.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn withdrawing_an_objects_announcements_removes_only_that_objects_stems() {
    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();
    let deleting = keys.checkpoint_reconcile_deleting();
    let neighbours = [("123.4", 11), ("123.4.tail", 12), ("1234.1", 13)];
    zadd(&tv, &deleting, 7, "12.3").await;
    zadd(&tv, &deleting, 8, "12.4.tail").await;
    for (member, score) in neighbours {
        zadd(&tv, &deleting, score, member).await;
    }

    let mut found = announced_stems_of_object(tv.client(), keys, 12)
        .await
        .expect("scan");
    found.sort();
    assert_eq!(
        found,
        owned(&["12.3", "12.4.tail"]),
        "object 12's stems, and only those"
    );

    let withdrawn = withdraw_object_announcements(tv.client(), keys, 12)
        .await
        .expect("withdraw");

    assert_eq!(withdrawn, 2);
    assert_eq!(zscore(&tv, &deleting, "12.3").await, None);
    assert_eq!(zscore(&tv, &deleting, "12.4.tail").await, None);
    for (member, score) in neighbours {
        assert_eq!(
            zscore(&tv, &deleting, member).await,
            Some(score as f64),
            "{member} names another object and keeps its score"
        );
    }
}
