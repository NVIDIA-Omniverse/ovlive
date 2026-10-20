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

//! b.8jc — the signed-zero merge (`storage::time_sample_merge`) against a real
//! store, through its public API.
//!
//! An object's legacy `-0.0` time-sample rows — written by an older build,
//! which stored the client's bits — are planted here by hand, as that build's
//! commit wrote them: the row's `value` or `hash`/`size` columns and its
//! `version`, its `obj:tss` member at score `version`, and, for an external
//! row, its `obj:ext` count. This build never writes one.
//!
//! | Test | Pins |
//! |---|---|
//! | [`merge_applies_rule_b_to_every_planted_pair_and_rekeys_a_lone_row`] | rule B's decision on inline and external pairs, a lone `-0.0` row re-keyed, the readers' full read unchanged, no `obj:ext` write, no release, the counter cells |
//! | [`merge_leaves_a_pair_with_an_undecodable_row_alone`] | `undecodable`: nothing written |
//! | [`a_second_merge_writes_nothing_not_even_the_stamp`] | idempotency: a re-read finds nothing to merge and sends no script |
//! | [`a_stale_plan_applied_after_a_merge_skips_every_row_and_writes_nothing`] | the script's per-row skip of a `-0.0` key already gone, and its no-write reply, for a second leader's plan read before the first merge |
//! | [`a_keep_whose_positive_zero_row_is_gone_is_skipped_and_writes_nothing`] | the per-row skip of a keep with no `+0.0` row: the section's only time-0 sample is never dropped |
//! | [`a_negative_zero_key_with_no_member_is_skipped_and_writes_nothing`] | the per-row skip of a `-0.0` key with no `obj:tss` member |
//! | [`an_unlisted_positive_zero_key_is_overwritten_by_a_take_of_the_served_row`] | a `+0.0` key with no `obj:tss` member planned as absent: the `-0.0` row readers serve is taken over it, counted `merged`, converging in one pass with the readers' full read unchanged |
//! | [`a_keep_whose_positive_zero_member_is_gone_is_skipped_and_writes_nothing`] | the per-row skip of a keep whose `+0.0` member went after the read: the `-0.0` row readers serve is never dropped; the next merge converges |
//! | [`a_commit_between_the_read_and_the_script_answers_changed_and_writes_nothing`] | `CHANGED` |
//! | [`a_live_snapshot_fence_answers_fenced_and_writes_nothing`] | `FENCED` |
//! | [`a_superseded_or_deleted_object_answers_not_live_and_writes_nothing`] | `NOT_LIVE`, at the read and at the script (`superseded`, `ext_gc_scheduled`) |
//! | [`two_concurrent_merges_merge_each_row_once`] | two leaders at once |
//! | [`more_legacy_sections_than_one_call_takes_are_all_merged`] | more sections than one call takes, merged over two calls |
//! | [`the_merge_stamp_aborts_a_watched_commit_planned_before_it`] | the header stamp fires a watched txn commit's `WATCH` (`test-fault-injection`) |
//!
//! The walk that runs the merge inside a reconciler pass — the audit lowering
//! the over-count a dropped row leaves, a row written after a pass merged by
//! the next, and a failed merge counted without forfeiting the pass — is
//! pinned in-crate, in `storage::ext_reconcile`'s tests
//! (`index_audit::signed_zero_merge`), where a single pass can be driven.

use std::collections::HashMap;
use std::sync::Arc;

use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use fred::interfaces::{HashesInterface, KeysInterface, SetsInterface, SortedSetsInterface};

use live_edit::domain::types::{ObjectState, Section};
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::{Metrics, TIME_SAMPLE_SIGNED_ZERO_MERGE_OUTCOMES};
use live_edit::storage::diff_read::{
    read_object_snapshot, snapshot_to_object_state, SnapshotRead, SnapshotReadConsumer,
};
use live_edit::storage::keys::KeySchema;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::test_support::relay::RelayedUpdates;
use live_edit::storage::test_support::structural_commit::{
    dump_object_keyspace, CommitTokens, ObjectKeyspaceDump,
};
use live_edit::storage::test_support::TestValkey;
use live_edit::storage::time_sample_merge::{
    apply_signed_zero_merge, merge_signed_zero_rows, negative_zero_member_section,
    negative_zero_sections, plan_signed_zero_merge, MergeAction, MergePlan, MergeReply,
    MergeSummary, PlanRead, PlannedRow, HEADER_FIELD_TIME_SAMPLES_MERGED_MS,
    MERGE_MAX_ROWS_PER_CALL, NEGATIVE_ZERO_TIME_HEX,
};

use crate::common::owned_delta::delta_with_set_field_inline;
use crate::storage::snapshot_read_atomic_test::{create, create_session, layer_store};

/// The first section id of the fixture object; it holds sections
/// `FIRST_SECTION..FIRST_SECTION + SECTIONS` under the root.
const FIRST_SECTION: u64 = 21;
const SECTIONS: u64 = 8;
const WRITER: &str = "merge-test-writer";
/// A snapshot-fence lease deadline (store-clock ms) no test run reaches: the
/// year 2286, so the planted lease is live whatever the store's clock reads.
const FAR_FUTURE_DEADLINE_MS: &str = "9999999999999";

/// External hashes, each the 32-byte repetition of its tag.
const X: [u8; 32] = [0x58; 32];
const Y: [u8; 32] = [0x59; 32];
const Z: [u8; 32] = [0x5A; 32];

/// One planted row's value.
#[derive(Debug, Clone, Copy)]
enum Planted {
    Inline(&'static [u8]),
    External([u8; 32]),
    /// An external row whose `hash` column the readers' decoder refuses.
    Malformed,
}

fn hex(hash: &[u8; 32]) -> String {
    hash.iter().map(|b| format!("{b:02x}")).collect()
}

/// The stored columns of a row holding `value` at `version`, as every writer
/// spells them (`storage::layer::encode_value_fields` plus `version`).
fn columns(value: Planted, version: u64) -> HashMap<String, String> {
    let mut out = HashMap::from([("version".to_owned(), version.to_string())]);
    match value {
        Planted::Inline(bytes) => {
            out.insert("value".to_owned(), BASE64.encode(bytes));
        }
        Planted::External(hash) => {
            out.insert("hash".to_owned(), hex(&hash));
            out.insert("size".to_owned(), "64".to_owned());
        }
        Planted::Malformed => {
            out.insert("hash".to_owned(), "zz".to_owned());
            out.insert("size".to_owned(), "64".to_owned());
        }
    }
    out
}

/// The fixture object: `SECTIONS` sections under the root and no row.
fn sections_state() -> ObjectState {
    let mut state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usda".to_owned(),
        ..Default::default()
    };
    let ids: Vec<u64> = (FIRST_SECTION..FIRST_SECTION + SECTIONS).collect();
    for (order, &id) in ids.iter().enumerate() {
        state.sections.insert(
            id,
            Section {
                section_id: id,
                parent_id: 0,
                name: format!("S{id}"),
                section_type: 1,
                section_order: order as u32,
                first_version: 1,
                last_version: 1,
            },
        );
    }
    state.children.insert(0, ids);
    state
}

struct Fx {
    tv: TestValkey,
    ls: LayerStore,
    session_id: String,
    layer: String,
    oid: u64,
}

impl Fx {
    async fn new(layer: &str) -> Self {
        let tv = TestValkey::setup().await;
        let ls = layer_store(&tv);
        let session_id = create_session(&tv, layer).await;
        let oid = create(&ls, &session_id, layer, &sections_state()).await;
        Self {
            tv,
            ls,
            session_id,
            layer: layer.to_owned(),
            oid,
        }
    }

    fn oid_str(&self) -> String {
        self.oid.to_string()
    }

    fn keys(&self) -> &KeySchema {
        self.tv.key_schema()
    }

    fn row_key(&self, section: u64, time: f64) -> String {
        self.keys()
            .obj_time_sample(&self.oid_str(), &section.to_string(), time)
    }

    /// Plant `section`'s row at `time` as an older build's commit wrote it:
    /// the row, its `obj:tss` member at score `version`, and for an external
    /// row its `obj:ext` count.
    async fn plant(&self, section: u64, time: f64, version: u64, value: Planted) {
        self.plant_unlisted(section, time, version, value).await;
        let _: i64 = self
            .tv
            .client()
            .pool
            .zadd(
                self.keys().obj_time_samples_index(&self.oid_str()),
                None,
                None,
                false,
                false,
                (
                    version as f64,
                    KeySchema::time_sample_member(&section.to_string(), time),
                ),
            )
            .await
            .expect("ZADD the planted member");
    }

    /// [`Self::plant`] without the `obj:tss` member: the row key and, for an
    /// external row, its `obj:ext` count — a row whose member drifted away,
    /// which no reader finds.
    async fn plant_unlisted(&self, section: u64, time: f64, version: u64, value: Planted) {
        let pool = &self.tv.client().pool;
        let _: () = pool
            .hset(self.row_key(section, time), columns(value, version))
            .await
            .expect("HSET the planted row");
        if let Planted::External(hash) = value {
            let _: i64 = pool
                .hincrby(self.keys().obj_ext_index(&self.oid_str()), hex(&hash), 1)
                .await
                .expect("HINCRBY obj:ext");
        }
    }

    async fn row(&self, section: u64, time: f64) -> HashMap<String, String> {
        self.tv
            .client()
            .pool
            .hgetall(self.row_key(section, time))
            .await
            .expect("HGETALL a row")
    }

    /// `obj:tss` as `member → score`.
    async fn members(&self) -> HashMap<String, f64> {
        let pairs: Vec<(String, f64)> = self
            .tv
            .client()
            .pool
            .zrange(
                self.keys().obj_time_samples_index(&self.oid_str()),
                0,
                -1,
                None,
                false,
                None,
                true,
            )
            .await
            .expect("ZRANGE obj:tss WITHSCORES");
        pairs.into_iter().collect()
    }

    async fn member_list(&self) -> Vec<String> {
        self.members().await.into_keys().collect()
    }

    /// The sections the walk would hand the merge: the `-0.0` members'.
    async fn legacy_sections(&self) -> Vec<String> {
        negative_zero_sections(&self.member_list().await)
    }

    async fn ext_index(&self) -> HashMap<String, String> {
        self.tv
            .client()
            .pool
            .hgetall(self.keys().obj_ext_index(&self.oid_str()))
            .await
            .expect("HGETALL obj:ext")
    }

    async fn pending_releases(&self) -> Vec<String> {
        self.tv
            .client()
            .pool
            .smembers(self.keys().ext_pending())
            .await
            .expect("SMEMBERS ext pending")
    }

    async fn header(&self) -> HashMap<String, String> {
        self.tv
            .client()
            .pool
            .hgetall(self.keys().obj_header(&self.oid_str()))
            .await
            .expect("HGETALL obj:hdr")
    }

    async fn dump(&self) -> ObjectKeyspaceDump {
        dump_object_keyspace(
            self.tv.client(),
            self.keys(),
            self.oid,
            CommitTokens::Verbatim,
        )
        .await
    }

    /// What every full reader serves: the atomic snapshot read, decoded.
    async fn served(&self) -> ObjectState {
        match read_object_snapshot(
            self.tv.client(),
            self.keys(),
            &self.oid_str(),
            SnapshotReadConsumer::Catchup,
        )
        .await
        {
            Ok(SnapshotRead::Snapshot(snapshot)) => snapshot_to_object_state(&snapshot.state),
            Ok(SnapshotRead::TooLarge(t)) => panic!("the fixture is within every ceiling: {t:?}"),
            Err(e) => panic!("the snapshot read must succeed: {e:?}"),
        }
    }

    /// One merge call's read and decisions, on a live object.
    async fn plan(&self, sections: &[String]) -> MergePlan {
        match plan_signed_zero_merge(self.tv.client(), self.keys(), &self.oid_str(), sections)
            .await
            .expect("the merge read succeeds")
        {
            PlanRead::Planned(plan) => plan,
            PlanRead::NotLive => panic!("the object is live"),
        }
    }

    async fn merge(&self, sections: &[String], metrics: Option<&Metrics>) -> MergeSummary {
        merge_signed_zero_rows(
            self.tv.client(),
            self.keys(),
            &self.oid_str(),
            sections,
            metrics,
        )
        .await
    }

    /// A fast-path commit on the object, through the store.
    async fn commit(&self) {
        self.ls
            .update_layer_relayed(
                &self.session_id,
                &self.layer,
                self.oid,
                Some(delta_with_set_field_inline(
                    FIRST_SECTION,
                    "label",
                    vec![0x2A],
                    1,
                )),
                WRITER,
                0,
            )
            .await
            .expect("the commit lands");
    }

    /// No `-0.0` row key or member of `section` remains.
    async fn assert_no_negative_zero(&self, section: u64, context: &str) {
        let exists: i64 = self
            .tv
            .client()
            .pool
            .exists(self.row_key(section, -0.0))
            .await
            .expect("EXISTS");
        assert_eq!(exists, 0, "{context}: section {section}'s -0.0 row is gone");
        assert!(
            !self
                .members()
                .await
                .contains_key(&KeySchema::time_sample_member(&section.to_string(), -0.0)),
            "{context}: section {section}'s -0.0 member is gone"
        );
    }
}

/// `live_edit_time_sample_signed_zero_merge_total`'s cells, all of them
/// (every cell is zero-seeded).
fn merge_cells(metrics: &Metrics) -> HashMap<&'static str, u64> {
    let text = render(metrics);
    TIME_SAMPLE_SIGNED_ZERO_MERGE_OUTCOMES
        .iter()
        .map(|&outcome| {
            let value = sample_value(
                &text,
                "live_edit_time_sample_signed_zero_merge_total",
                &[("outcome", outcome)],
            )
            .unwrap_or_else(|| panic!("the {outcome} cell is zero-seeded"));
            (outcome, value as u64)
        })
        .collect()
}

/// `cells` with every other outcome at zero.
fn only(cells: &[(&'static str, u64)]) -> HashMap<&'static str, u64> {
    let mut out: HashMap<&'static str, u64> = TIME_SAMPLE_SIGNED_ZERO_MERGE_OUTCOMES
        .iter()
        .map(|&o| (o, 0))
        .collect();
    out.extend(cells.iter().copied());
    out
}

/// Whether `time` is `-0.0` (by its sign bit, which `==` ignores).
fn is_negative_zero(time: f64) -> bool {
    time == 0.0 && time.is_sign_negative()
}

/// Every served time sample is held at `+0.0`'s bits where its time is 0.
fn assert_no_served_negative_zero(state: &ObjectState, context: &str) {
    for ((section, key), row) in &state.time_samples {
        assert!(
            !(is_negative_zero(key.into_inner()) || is_negative_zero(row.time.into_inner())),
            "{context}: section {section} is served at -0.0"
        );
    }
}

/// One planted section of the decision table: its `+0.0` row (version and
/// value), if any, its `-0.0` row, and whether the `-0.0` row survives.
type PlantedPair = (u64, Option<(u64, Planted)>, (u64, Planted), bool);

/// Rule B on every combination the decision table names, inline and
/// external, plus a lone `-0.0` row: each section ends with one row, at
/// `+0.0`, holding the survivor's exact stored columns, its member at the
/// survivor's version, and no `-0.0` key or member. What the readers serve
/// is the same before and after, `obj:ext` is not written, nothing is
/// released, the header carries the merge stamp, and the counter adds 7
/// `merged` and 1 `rekeyed`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn merge_applies_rule_b_to_every_planted_pair_and_rekeys_a_lone_row() {
    let fx = Fx::new("merge-table").await;
    let metrics = crate::common::metrics_scrape_only("merge-table");
    use Planted::{External as E, Inline as I};
    let pairs: [PlantedPair; 8] = [
        (21, Some((1, I(b"a"))), (2, I(b"b")), true),
        (22, Some((2, I(b"a"))), (1, I(b"b")), false),
        (23, Some((3, I(b""))), (3, I(b"b")), true),
        (24, Some((3, I(b"a"))), (3, I(b"")), false),
        (25, Some((3, I(b"a"))), (3, I(b"b")), false),
        (26, Some((1, E(X))), (2, E(Y)), true),
        (27, Some((2, E(X))), (1, E(Y)), false),
        (28, None, (4, E(Z)), true),
    ];
    for &(section, positive, (version, value), _) in &pairs {
        if let Some((pv, pvalue)) = positive {
            fx.plant(section, 0.0, pv, pvalue).await;
        }
        fx.plant(section, -0.0, version, value).await;
    }
    let served_before = fx.served().await;
    let ext_before = fx.ext_index().await;
    let sections = fx.legacy_sections().await;
    assert_eq!(sections.len(), 8, "fixture: eight -0.0 members");

    let summary = fx.merge(&sections, Some(&metrics)).await;

    assert_eq!(
        summary,
        MergeSummary {
            merged: 7,
            rekeyed: 1,
            ..MergeSummary::default()
        }
    );
    assert_eq!(summary.rows_merged(), 8);
    for &(section, positive, negative, take) in &pairs {
        let (version, value) = if take {
            negative
        } else {
            positive.expect("a keep has a +0.0 row")
        };
        let context = format!("section {section}");
        assert_eq!(
            fx.row(section, 0.0).await,
            columns(value, version),
            "{context}: the +0.0 row holds the survivor's columns"
        );
        assert_eq!(
            fx.members()
                .await
                .get(&KeySchema::time_sample_member(&section.to_string(), 0.0)),
            Some(&(version as f64)),
            "{context}: the +0.0 member is at the survivor's version"
        );
        fx.assert_no_negative_zero(section, &context).await;
    }
    let served_after = fx.served().await;
    assert_eq!(
        served_after, served_before,
        "the readers serve the same state before and after the merge"
    );
    assert_no_served_negative_zero(&served_before, "before");
    assert_no_served_negative_zero(&served_after, "after");
    assert_eq!(
        fx.ext_index().await,
        ext_before,
        "the merge writes no obj:ext field: the dropped external rows leave X and Y \
         over-counted, the leak direction, for the reconciler's audit"
    );
    assert!(
        fx.pending_releases().await.is_empty(),
        "nothing is released"
    );
    assert!(
        fx.header()
            .await
            .contains_key(HEADER_FIELD_TIME_SAMPLES_MERGED_MS),
        "the merge stamped the header"
    );
    assert_eq!(
        merge_cells(&metrics),
        only(&[("merged", 7), ("rekeyed", 1)])
    );
}

/// A pair whose `+0.0` row carries a `hash` the readers' decoder refuses is
/// left alone and counted `undecodable`: nothing is written, the stamp
/// included.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn merge_leaves_a_pair_with_an_undecodable_row_alone() {
    let fx = Fx::new("merge-undecodable").await;
    let metrics = crate::common::metrics_scrape_only("merge-undecodable");
    fx.plant(FIRST_SECTION, 0.0, 1, Planted::Malformed).await;
    fx.plant(FIRST_SECTION, -0.0, 2, Planted::Inline(b"b"))
        .await;
    let before = fx.dump().await;

    let summary = fx.merge(&fx.legacy_sections().await, Some(&metrics)).await;

    assert_eq!(
        summary,
        MergeSummary {
            undecodable: 1,
            ..MergeSummary::default()
        }
    );
    assert_eq!(fx.dump().await, before, "nothing is written");
    assert_eq!(merge_cells(&metrics), only(&[("undecodable", 1)]));
}

/// A second merge over the same sections that reads after the first merge
/// applied — the next pass's over a stale listing, or a second leader's —
/// finds every `-0.0` row absent at its read and sends no script: nothing is
/// written, not even the header stamp, and a fresh listing names no section.
/// A second leader's plan read *before* the first merge applied does reach
/// the script; that case is
/// [`a_stale_plan_applied_after_a_merge_skips_every_row_and_writes_nothing`].
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_second_merge_writes_nothing_not_even_the_stamp() {
    let fx = Fx::new("merge-idempotent").await;
    fx.plant(21, 0.0, 1, Planted::Inline(b"a")).await;
    fx.plant(21, -0.0, 2, Planted::Inline(b"b")).await;
    fx.plant(22, -0.0, 3, Planted::External(X)).await;
    let sections = fx.legacy_sections().await;
    let first = fx.merge(&sections, None).await;
    assert_eq!(
        first.rows_merged(),
        2,
        "fixture: the first merge merged both"
    );
    let after_first = fx.dump().await;
    let replan = fx.plan(&sections).await;
    assert!(
        replan.rows.is_empty() && replan.absent == 2,
        "a fresh read finds both -0.0 rows absent, so no script is sent: {replan:?}"
    );

    let second = fx.merge(&sections, None).await;

    assert_eq!(second, MergeSummary::default(), "nothing to merge");
    assert_eq!(
        fx.dump().await,
        after_first,
        "the second merge wrote nothing, the header stamp included"
    );
    assert!(fx.legacy_sections().await.is_empty());
}

/// Two plans read from the same state — two replicas each believing they
/// lead, both reading before either applies — applied one after the other.
/// The first merges every row. The second still passes the `seq` guard (a
/// merge does not advance `seq`), so its script runs, and finds every `-0.0`
/// key gone: its per-row skip leaves each row as it is and, with no row to
/// act on, it writes nothing at all, not even the header stamp. Without the
/// skip, the stale plan's takes would `RENAME` keys that no longer exist and
/// abort the script after its stamp.
///
/// The fixture covers each action: a keep, a take over a `+0.0` row, and a
/// lone `-0.0` row's re-key.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_stale_plan_applied_after_a_merge_skips_every_row_and_writes_nothing() {
    let fx = Fx::new("merge-stale-plan").await;
    fx.plant(21, 0.0, 2, Planted::Inline(b"a")).await;
    fx.plant(21, -0.0, 1, Planted::Inline(b"b")).await;
    fx.plant(22, 0.0, 1, Planted::Inline(b"a")).await;
    fx.plant(22, -0.0, 2, Planted::Inline(b"b")).await;
    fx.plant(23, -0.0, 3, Planted::External(X)).await;
    let sections = fx.legacy_sections().await;
    let first = fx.plan(&sections).await;
    let stale = fx.plan(&sections).await;
    assert_eq!(first, stale, "fixture: both plans read the same state");
    let actions: HashMap<String, MergeAction> = stale
        .rows
        .iter()
        .map(|r| (r.section_id.clone(), r.action))
        .collect();
    assert_eq!(
        actions,
        HashMap::from([
            ("21".to_owned(), MergeAction::Keep),
            ("22".to_owned(), MergeAction::Take),
            ("23".to_owned(), MergeAction::Take),
        ]),
        "fixture: a keep, a take and a re-key, so the stale plan sends a script"
    );

    let applied = apply_signed_zero_merge(fx.tv.client(), fx.keys(), &first)
        .await
        .expect("the first script runs");
    assert_eq!(
        applied,
        MergeReply::Merged {
            merged: 2,
            rekeyed: 1,
            skipped: 0
        },
        "fixture: the first plan merges every row"
    );
    // Drop the first merge's stamp, so a stamp from the stale plan is seen
    // whatever the store's clock reads between the two.
    let removed: i64 = fx
        .tv
        .client()
        .pool
        .hdel(
            fx.keys().obj_header(&fx.oid_str()),
            HEADER_FIELD_TIME_SAMPLES_MERGED_MS,
        )
        .await
        .expect("HDEL the first merge's stamp");
    assert_eq!(removed, 1, "fixture: the first merge stamped the header");
    let after_first = fx.dump().await;

    let reply = apply_signed_zero_merge(fx.tv.client(), fx.keys(), &stale)
        .await
        .expect("the stale plan's script runs");

    assert_eq!(
        reply,
        MergeReply::Merged {
            merged: 0,
            rekeyed: 0,
            skipped: 3
        },
        "every row of the stale plan is skipped"
    );
    assert_eq!(fx.dump().await, after_first, "the stale plan wrote nothing");
    assert!(
        !fx.header()
            .await
            .contains_key(HEADER_FIELD_TIME_SAMPLES_MERGED_MS),
        "the stale plan wrote no stamp"
    );
    for section in 21..=23 {
        fx.assert_no_negative_zero(section, "after both plans")
            .await;
    }
}

/// A keep whose `+0.0` row is gone by the time the script runs is skipped:
/// dropping the `-0.0` row would drop the section's only sample at time 0.
/// The plan is hand-built — a lone `-0.0` row, sent as a keep, the shape a
/// plan read while a `+0.0` row existed has once that row is gone — and the
/// script answers `Merged{0,0,1}`, leaves the `-0.0` row and member as they
/// are, and writes no stamp. A keep whose `+0.0` key is left but whose member
/// is gone is skipped too:
/// [`a_keep_whose_positive_zero_member_is_gone_is_skipped_and_writes_nothing`].
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_keep_whose_positive_zero_row_is_gone_is_skipped_and_writes_nothing() {
    let fx = Fx::new("merge-keep-gone").await;
    fx.plant(21, -0.0, 2, Planted::Inline(b"b")).await;
    let read = fx.plan(&fx.legacy_sections().await).await;
    let plan = MergePlan {
        rows: vec![PlannedRow {
            section_id: "21".to_owned(),
            action: MergeAction::Keep,
        }],
        ..read
    };
    let before = fx.dump().await;

    let reply = apply_signed_zero_merge(fx.tv.client(), fx.keys(), &plan)
        .await
        .expect("the script runs");

    assert_eq!(
        reply,
        MergeReply::Merged {
            merged: 0,
            rekeyed: 0,
            skipped: 1
        }
    );
    assert_eq!(fx.dump().await, before, "nothing is written");
    assert_eq!(
        fx.row(21, -0.0).await,
        columns(Planted::Inline(b"b"), 2),
        "the section's only time-0 sample is kept"
    );
    assert_eq!(
        fx.members()
            .await
            .get(&KeySchema::time_sample_member("21", -0.0)),
        Some(&2.0),
        "its member is kept"
    );
    assert!(
        !fx.header()
            .await
            .contains_key(HEADER_FIELD_TIME_SAMPLES_MERGED_MS),
        "no stamp"
    );
}

/// A `-0.0` row key with no `obj:tss` member is not a row readers see, so the
/// script leaves it as it is: a take of it answers `Merged{0,0,1}`, moves
/// nothing to `+0.0`, adds no member and writes no stamp.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_negative_zero_key_with_no_member_is_skipped_and_writes_nothing() {
    let fx = Fx::new("merge-no-member").await;
    let unlisted = columns(Planted::Inline(b"b"), 2);
    let _: () = fx
        .tv
        .client()
        .pool
        .hset(fx.row_key(21, -0.0), unlisted.clone())
        .await
        .expect("HSET a -0.0 row with no member");
    let plan = fx.plan(&["21".to_owned()]).await;
    assert_eq!(
        plan.rows.iter().map(|r| r.action).collect::<Vec<_>>(),
        vec![MergeAction::Take],
        "fixture: the read sees a lone -0.0 row and plans its re-key"
    );
    let before = fx.dump().await;

    let reply = apply_signed_zero_merge(fx.tv.client(), fx.keys(), &plan)
        .await
        .expect("the script runs");

    assert_eq!(
        reply,
        MergeReply::Merged {
            merged: 0,
            rekeyed: 0,
            skipped: 1
        }
    );
    assert_eq!(fx.dump().await, before, "nothing is written");
    assert_eq!(fx.row(21, -0.0).await, unlisted, "the -0.0 key is left");
    assert!(fx.row(21, 0.0).await.is_empty(), "nothing moved to +0.0");
    assert!(fx.members().await.is_empty(), "no member is added");
    assert!(
        !fx.header()
            .await
            .contains_key(HEADER_FIELD_TIME_SAMPLES_MERGED_MS),
        "no stamp"
    );
}

/// The versions of `section`'s samples served at time 0.
fn served_zero_versions(state: &ObjectState, section: u64) -> Vec<u64> {
    state
        .time_samples
        .iter()
        .filter(|((s, key), _)| *s == section && key.into_inner() == 0.0)
        .map(|(_, row)| row.version)
        .collect()
}

/// A `+0.0` row key whose `obj:tss` member is gone — keyspace drift — is a
/// row no reader finds, so readers serve the section's `-0.0` row. The merge
/// plans the unlisted row as absent and takes the `-0.0` row, even though
/// rule B would keep the `+0.0` row (its version is higher): the `-0.0` row
/// moves over the unlisted key, its member to `+0.0` at the same score, and
/// the take counts `merged` since the `+0.0` key existed. One pass converges:
/// what readers serve is unchanged, `obj:ext` is not written (the unlisted
/// external row's count is left as a surplus, the leak direction), nothing is
/// released, and a second merge writes nothing.
///
/// Deciding keep instead would leave the script's backstop (pinned by
/// [`a_keep_whose_positive_zero_member_is_gone_is_skipped_and_writes_nothing`])
/// skipping the section on every pass, and it would never converge.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_unlisted_positive_zero_key_is_overwritten_by_a_take_of_the_served_row() {
    let fx = Fx::new("merge-unlisted-positive").await;
    let metrics = crate::common::metrics_scrape_only("merge-unlisted-positive");
    fx.plant_unlisted(21, 0.0, 5, Planted::External(X)).await;
    fx.plant(21, -0.0, 2, Planted::Inline(b"b")).await;
    let served_before = fx.served().await;
    assert_eq!(
        served_zero_versions(&served_before, 21),
        vec![2],
        "fixture: readers serve the -0.0 row, not the unlisted +0.0 one"
    );
    let ext_before = fx.ext_index().await;
    let sections = fx.legacy_sections().await;
    assert_eq!(
        fx.plan(&sections)
            .await
            .rows
            .iter()
            .map(|r| r.action)
            .collect::<Vec<_>>(),
        vec![MergeAction::Take],
        "the unlisted +0.0 row is planned as absent, so the -0.0 row is taken"
    );

    let summary = fx.merge(&sections, Some(&metrics)).await;

    assert_eq!(
        summary,
        MergeSummary {
            merged: 1,
            ..MergeSummary::default()
        },
        "a take over an existing +0.0 key counts merged"
    );
    assert_eq!(merge_cells(&metrics), only(&[("merged", 1)]));
    assert_eq!(
        fx.row(21, 0.0).await,
        columns(Planted::Inline(b"b"), 2),
        "the +0.0 key holds the -0.0 row's columns"
    );
    assert_eq!(
        fx.members().await,
        HashMap::from([(KeySchema::time_sample_member("21", 0.0), 2.0)]),
        "one member, at +0.0, at the -0.0 member's score"
    );
    fx.assert_no_negative_zero(21, "after the merge").await;
    let served_after = fx.served().await;
    assert_eq!(
        served_after, served_before,
        "the readers serve the same state before and after the merge"
    );
    assert_no_served_negative_zero(&served_after, "after");
    assert_eq!(
        fx.ext_index().await,
        ext_before,
        "the merge writes no obj:ext field"
    );
    assert!(
        fx.pending_releases().await.is_empty(),
        "nothing is released"
    );
    assert!(
        fx.header()
            .await
            .contains_key(HEADER_FIELD_TIME_SAMPLES_MERGED_MS),
        "the merge stamped the header"
    );
    assert!(
        fx.legacy_sections().await.is_empty(),
        "no -0.0 member is left"
    );
    let after_first = fx.dump().await;

    let second = fx.merge(&sections, None).await;

    assert_eq!(second, MergeSummary::default(), "nothing left to merge");
    assert_eq!(
        fx.dump().await,
        after_first,
        "the second merge wrote nothing, the header stamp included"
    );
}

/// A keep planned while both zero-time rows were listed, applied after the
/// `+0.0` member went (keyspace drift between the read and the script), is
/// skipped: readers no longer find the `+0.0` row and serve the `-0.0` one,
/// so dropping the `-0.0` row would change what they serve. The script
/// answers `Merged{0,0,1}`, leaves both keys and the `-0.0` member as they
/// are, and writes no stamp. The next merge reads afresh, takes the `-0.0`
/// row, and converges.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_keep_whose_positive_zero_member_is_gone_is_skipped_and_writes_nothing() {
    let fx = Fx::new("merge-keep-member-gone").await;
    fx.plant(21, 0.0, 5, Planted::Inline(b"a")).await;
    fx.plant(21, -0.0, 2, Planted::Inline(b"b")).await;
    let sections = fx.legacy_sections().await;
    let plan = fx.plan(&sections).await;
    assert_eq!(
        plan.rows.iter().map(|r| r.action).collect::<Vec<_>>(),
        vec![MergeAction::Keep],
        "fixture: with both rows listed, the newer +0.0 row is kept"
    );
    let removed: i64 = fx
        .tv
        .client()
        .pool
        .zrem(
            fx.keys().obj_time_samples_index(&fx.oid_str()),
            KeySchema::time_sample_member("21", 0.0),
        )
        .await
        .expect("ZREM the +0.0 member");
    assert_eq!(removed, 1, "fixture: the +0.0 member was listed");
    let served_before = fx.served().await;
    assert_eq!(
        served_zero_versions(&served_before, 21),
        vec![2],
        "fixture: readers now serve the -0.0 row"
    );
    let before = fx.dump().await;

    let reply = apply_signed_zero_merge(fx.tv.client(), fx.keys(), &plan)
        .await
        .expect("the script runs");

    assert_eq!(
        reply,
        MergeReply::Merged {
            merged: 0,
            rekeyed: 0,
            skipped: 1
        }
    );
    assert_eq!(fx.dump().await, before, "nothing is written");
    assert_eq!(
        fx.row(21, -0.0).await,
        columns(Planted::Inline(b"b"), 2),
        "the -0.0 row readers serve is kept"
    );
    assert_eq!(
        fx.members().await,
        HashMap::from([(KeySchema::time_sample_member("21", -0.0), 2.0)]),
        "its member is kept, and no +0.0 member is added"
    );
    assert_eq!(
        fx.row(21, 0.0).await,
        columns(Planted::Inline(b"a"), 5),
        "the unlisted +0.0 key is left"
    );
    assert!(
        !fx.header()
            .await
            .contains_key(HEADER_FIELD_TIME_SAMPLES_MERGED_MS),
        "no stamp"
    );
    assert_eq!(
        fx.served().await,
        served_before,
        "the readers serve the same state"
    );

    let next = fx.merge(&sections, None).await;

    assert_eq!(
        next,
        MergeSummary {
            merged: 1,
            ..MergeSummary::default()
        },
        "a fresh read takes the -0.0 row"
    );
    assert_eq!(
        fx.row(21, 0.0).await,
        columns(Planted::Inline(b"b"), 2),
        "the +0.0 key holds the -0.0 row's columns"
    );
    fx.assert_no_negative_zero(21, "after the next merge").await;
    assert_eq!(fx.served().await, served_before);
}

/// A commit landing between the merge's read and its script moves `seq`, so
/// the script answers `CHANGED` and writes nothing; the next merge reads
/// afresh and merges.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_commit_between_the_read_and_the_script_answers_changed_and_writes_nothing() {
    let fx = Fx::new("merge-changed").await;
    fx.plant(21, 0.0, 1, Planted::Inline(b"a")).await;
    fx.plant(21, -0.0, 2, Planted::Inline(b"b")).await;
    let sections = fx.legacy_sections().await;
    let PlanRead::Planned(plan) =
        plan_signed_zero_merge(fx.tv.client(), fx.keys(), &fx.oid_str(), &sections)
            .await
            .expect("the read succeeds")
    else {
        panic!("the object is live");
    };
    assert_eq!(
        plan.rows.iter().map(|r| r.action).collect::<Vec<_>>(),
        vec![MergeAction::Take],
        "fixture: the newer -0.0 row is taken"
    );

    fx.commit().await;
    let after_commit = fx.dump().await;
    let reply = apply_signed_zero_merge(fx.tv.client(), fx.keys(), &plan)
        .await
        .expect("the script runs");

    assert_eq!(reply, MergeReply::Changed);
    assert_eq!(fx.dump().await, after_commit, "CHANGED writes nothing");

    let retry = fx.merge(&sections, None).await;
    assert_eq!(retry.merged, 1, "a fresh read merges");
    fx.assert_no_negative_zero(21, "after the retry").await;
}

/// A live snapshot-fence lease on the object answers `FENCED` and writes
/// nothing; once it is gone the merge lands.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_live_snapshot_fence_answers_fenced_and_writes_nothing() {
    let fx = Fx::new("merge-fenced").await;
    let metrics = crate::common::metrics_scrape_only("merge-fenced");
    fx.plant(21, -0.0, 2, Planted::Inline(b"b")).await;
    fx.plant(22, -0.0, 2, Planted::Inline(b"c")).await;
    let fence = fx.keys().obj_snapshot_fence(&fx.oid_str());
    let pool = &fx.tv.client().pool;
    let _: () = pool
        .hset(&fence, ("r:merge-test-reader", FAR_FUTURE_DEADLINE_MS))
        .await
        .expect("plant a live lease");
    let before = fx.dump().await;
    let sections = fx.legacy_sections().await;

    let summary = fx.merge(&sections, Some(&metrics)).await;

    assert_eq!(
        summary,
        MergeSummary {
            fenced: 2,
            ..MergeSummary::default()
        }
    );
    assert_eq!(fx.dump().await, before, "FENCED writes nothing");
    assert_eq!(merge_cells(&metrics), only(&[("fenced", 2)]));

    let _: i64 = pool
        .hdel(&fence, "r:merge-test-reader")
        .await
        .expect("drop the lease");
    assert_eq!(fx.merge(&sections, None).await.rekeyed, 2);
}

/// A retiring object — its header marked `superseded`, or
/// `ext_gc_scheduled` — answers `NOT_LIVE` at the script, and a deleted
/// header answers it at the read; none of them writes anything.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_superseded_or_deleted_object_answers_not_live_and_writes_nothing() {
    let fx = Fx::new("merge-not-live").await;
    fx.plant(21, -0.0, 2, Planted::Inline(b"b")).await;
    let sections = fx.legacy_sections().await;
    let PlanRead::Planned(plan) =
        plan_signed_zero_merge(fx.tv.client(), fx.keys(), &fx.oid_str(), &sections)
            .await
            .expect("the read succeeds")
    else {
        panic!("the object is live");
    };
    let header_key = fx.keys().obj_header(&fx.oid_str());
    let pool = &fx.tv.client().pool;
    let _: () = pool
        .hset(&header_key, ("superseded", "1"))
        .await
        .expect("mark the header superseded");
    let before = fx.dump().await;

    let reply = apply_signed_zero_merge(fx.tv.client(), fx.keys(), &plan)
        .await
        .expect("the script runs");

    assert_eq!(reply, MergeReply::NotLive, "a superseded object");
    assert_eq!(fx.dump().await, before, "NOT_LIVE writes nothing");

    let _: i64 = pool
        .hdel(&header_key, "superseded")
        .await
        .expect("clear superseded");
    let _: () = pool
        .hset(&header_key, ("ext_gc_scheduled", "1"))
        .await
        .expect("mark the header ext_gc_scheduled");
    let before = fx.dump().await;

    let reply = apply_signed_zero_merge(fx.tv.client(), fx.keys(), &plan)
        .await
        .expect("the script runs");

    assert_eq!(reply, MergeReply::NotLive, "an ext_gc_scheduled object");
    assert_eq!(fx.dump().await, before, "NOT_LIVE writes nothing");

    let _: i64 = pool.del(&header_key).await.expect("DEL the header");
    let before = fx.dump().await;
    let summary = fx.merge(&sections, None).await;
    assert_eq!(
        summary,
        MergeSummary {
            not_live: 1,
            ..MergeSummary::default()
        },
        "a deleted header"
    );
    assert_eq!(fx.dump().await, before, "NOT_LIVE writes nothing");
}

/// Two merges of the same object at once — two replicas each believing they
/// lead — merge each row exactly once between them and leave the rows the
/// one rule says.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn two_concurrent_merges_merge_each_row_once() {
    let fx = Arc::new(Fx::new("merge-concurrent").await);
    for section in FIRST_SECTION..FIRST_SECTION + SECTIONS {
        fx.plant(section, 0.0, 1, Planted::Inline(b"a")).await;
        fx.plant(section, -0.0, 2, Planted::Inline(b"b")).await;
    }
    let sections = Arc::new(fx.legacy_sections().await);
    let barrier = Arc::new(tokio::sync::Barrier::new(2));

    let run = |fx: Arc<Fx>, sections: Arc<Vec<String>>, barrier: Arc<tokio::sync::Barrier>| {
        tokio::spawn(async move {
            barrier.wait().await;
            fx.merge(&sections, None).await
        })
    };
    let a = run(Arc::clone(&fx), Arc::clone(&sections), Arc::clone(&barrier));
    let b = run(Arc::clone(&fx), Arc::clone(&sections), Arc::clone(&barrier));
    let (a, b) = (a.await.expect("merge A"), b.await.expect("merge B"));

    assert_eq!(
        a.rows_merged() + b.rows_merged(),
        SECTIONS,
        "each row is merged exactly once between the two: {a:?} {b:?}"
    );
    assert_eq!(
        (a.failed, b.failed, a.not_live, b.not_live),
        (0, 0, 0, 0),
        "{a:?} {b:?}"
    );
    for section in FIRST_SECTION..FIRST_SECTION + SECTIONS {
        assert_eq!(
            fx.row(section, 0.0).await,
            columns(Planted::Inline(b"b"), 2)
        );
        fx.assert_no_negative_zero(section, "after both merges")
            .await;
    }
}

/// An object with one more legacy section than one call takes is merged in
/// two calls, and every row is re-keyed. One call cannot take them all: the
/// read refuses more than `MERGE_MAX_ROWS_PER_CALL` sections and so does the
/// script, so a summary covering every row shows `merge_signed_zero_rows`
/// split them.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn more_legacy_sections_than_one_call_takes_are_all_merged() {
    const FIRST: u64 = 1_000;
    let count = MERGE_MAX_ROWS_PER_CALL as u64 + 1;
    let fx = Fx::new("merge-chunked").await;
    let metrics = crate::common::metrics_scrape_only("merge-chunked");
    for section in FIRST..FIRST + count {
        fx.plant(section, -0.0, 2, Planted::Inline(b"b")).await;
    }
    let sections = fx.legacy_sections().await;
    assert_eq!(
        sections.len() as u64,
        count,
        "fixture: one -0.0 member per section"
    );

    let summary = fx.merge(&sections, Some(&metrics)).await;

    assert_eq!(
        summary,
        MergeSummary {
            rekeyed: count,
            ..MergeSummary::default()
        }
    );
    assert_eq!(merge_cells(&metrics), only(&[("rekeyed", count)]));
    let members = fx.members().await;
    for section in FIRST..FIRST + count {
        let context = format!("section {section}");
        assert_eq!(
            fx.row(section, 0.0).await,
            columns(Planted::Inline(b"b"), 2),
            "{context}: the row moved to +0.0"
        );
        assert_eq!(
            members.get(&KeySchema::time_sample_member(&section.to_string(), 0.0)),
            Some(&2.0),
            "{context}: the +0.0 member is at the row's version"
        );
        assert!(
            fx.row(section, -0.0).await.is_empty(),
            "{context}: the -0.0 row is gone"
        );
    }
    assert_eq!(
        members.len() as u64,
        count,
        "one member per section, every one at +0.0"
    );
}

/// The merge's header stamp fires the `WATCH` of a watched txn commit planned
/// before it, so that commit cannot `EXEC` rows planned from the pre-merge
/// keyspace: its first attempt is aborted and the retry, which reads afresh,
/// lands. A merge with nothing to merge writes no stamp, and the commit it
/// runs inside lands on its first attempt — the control that shows the
/// retry is the stamp's doing.
///
/// The merge runs inside the WATCH window through the store's test hook
/// (`LayerStore::with_watch_window_hook_for_test`), so it needs
/// `test-fault-injection` and is compiled out without it.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_merge_stamp_aborts_a_watched_commit_planned_before_it() {
    use std::sync::atomic::{AtomicU64, Ordering};

    use live_edit::config::UpdateLayerBackend;
    use live_edit::storage::watched_txn::WatchWindowHook;

    let fx = Fx::new("merge-watch").await;
    let windows = Arc::new(AtomicU64::new(0));
    let (client, keys, oid) = (fx.tv.client().clone(), fx.keys().clone(), fx.oid_str());
    let section = FIRST_SECTION.to_string();
    let hook: WatchWindowHook = {
        let windows = Arc::clone(&windows);
        Arc::new(move |_guarded| {
            let (windows, client, keys, oid, section) = (
                Arc::clone(&windows),
                client.clone(),
                keys.clone(),
                oid.clone(),
                section.clone(),
            );
            Box::pin(async move {
                // Only the first window of each commit merges: the retry's
                // window must find nothing left to stamp.
                if windows.fetch_add(1, Ordering::SeqCst) == 0 {
                    merge_signed_zero_rows(&client, &keys, &oid, &[section], None).await;
                }
            })
        })
    };
    let watch_store = LayerStore::new(fx.tv.client().clone(), fx.keys().clone(), 4096)
        .with_update_layer_backend(UpdateLayerBackend::Txn)
        .with_watch_window_hook_for_test(hook);
    let commit = |value: u8| {
        watch_store.update_layer_relayed(
            &fx.session_id,
            &fx.layer,
            fx.oid,
            Some(delta_with_set_field_inline(
                FIRST_SECTION,
                "label",
                vec![value],
                1,
            )),
            WRITER,
            0,
        )
    };

    // Control: nothing to merge, so no stamp, one window.
    commit(1).await.expect("the control commit lands");
    assert_eq!(
        windows.swap(0, Ordering::SeqCst),
        1,
        "control: a merge with nothing to merge writes no stamp, so the commit lands on its \
         first attempt"
    );

    fx.plant(FIRST_SECTION, -0.0, 1, Planted::Inline(b"b"))
        .await;
    commit(2).await.expect("the retry lands");

    assert_eq!(
        windows.load(Ordering::SeqCst),
        2,
        "the merge's stamp aborted the first attempt, and the retry's window ran"
    );
    fx.assert_no_negative_zero(FIRST_SECTION, "after the commit")
        .await;
    assert!(
        fx.header()
            .await
            .contains_key(HEADER_FIELD_TIME_SAMPLES_MERGED_MS),
        "the merge stamped the header"
    );
}

/// The member parser the walk detects legacy rows by: a `-0.0` member names
/// its section; a `+0.0` member, another time and a member without a time
/// part name none.
#[test]
fn negative_zero_member_section_names_only_a_negative_zero_member() {
    assert_eq!(
        negative_zero_member_section(&format!("42:{NEGATIVE_ZERO_TIME_HEX}")),
        Some("42")
    );
    assert_eq!(
        negative_zero_member_section(&KeySchema::time_sample_member("42", -0.0)),
        Some("42")
    );
    assert_eq!(
        negative_zero_member_section(&KeySchema::time_sample_member("42", 0.0)),
        None
    );
    assert_eq!(
        negative_zero_member_section(&KeySchema::time_sample_member("42", 1.5)),
        None
    );
    assert_eq!(negative_zero_member_section("42"), None);
    assert_eq!(
        negative_zero_sections(&[
            KeySchema::time_sample_member("1", 0.0),
            KeySchema::time_sample_member("2", -0.0),
            KeySchema::time_sample_member("3", -0.0),
        ]),
        vec!["2".to_owned(), "3".to_owned()]
    );
}
