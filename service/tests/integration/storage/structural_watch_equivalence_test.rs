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

//! Structural mode's `obj:ext` accounting on a drifted index, beside the WATCH
//! path's (plan `b.dkt` Epic 5 Task 4; Spec SDD SR-10.3, SDD §7.5.6).
//!
//! The keyspace-equivalence proptest (`tests/proptests.rs`,
//! `structural_watch_equivalence`) pins the general rule. This file pins its
//! named cases exactly. On a drift-free index the two paths leave the same
//! keyspace. On a drifted one, for every external hash a structural commit
//! touches, structural mode writes the raise-only floor
//! `max(stored + delta, untouched + ends)`: the cache's untouched-row counts
//! plus the touched rows' final writes can raise a count or suppress a
//! release, but never lower a count or cause a release. The WATCH path's
//! snapshot branch writes `recount + delta` when `delta != 0` (`plan_ext_index`
//! in `src/storage/layer.rs`). So the paths agree when an under-count changes
//! (T1, T3), structural mode also repairs an under-count the commit leaves
//! unchanged (T4), it keeps an over-count where the WATCH path drops and
//! releases the hash (T5), and on a held hash the index lacks it releases
//! nothing where the WATCH path releases (T6). T2 runs a cache that is wrong
//! about an untouched row and shows the floor cannot release; its companion
//! runs one wrong about the touched row, so the untouched-row count block
//! carries no entry for the hash the script settles, and shows the same.
//!
//! Every case also plants drift on a hash the commit does not touch, which
//! stays as planted, and every case with a WATCH twin requires every key but
//! `obj:ext` and `obj:wrel` to be equal on the two paths (T7). `obj:wrel`, the
//! commit release ring, holds an entry naming what the commit released, so
//! where the paths' releases differ by design (T5, T6) their entries differ
//! too: each path's entry is checked against its own releases, and the two
//! entries are compared only where the releases agree. Each case's planted count is
//! chosen so the count or release structural mode is expected to leave
//! differs from what the rule it rules out would leave — `stored + delta`
//! alone (T1, T3, T4, T6b), the cache's recount alone (T2, T5), or the WATCH
//! path's recount (T4, T5, T6) — which each test's doc and assertion
//! messages name.
//!
//! Each delta carries a structural command, because a field-only delta on the
//! `Txn` backend takes the WATCH path's non-snapshot branch, which reads its
//! pre-image from the stored index.

use std::collections::BTreeSet;
use std::sync::Arc;

use bytes::Bytes;
use fred::interfaces::HashesInterface;
use oodf_ffi::{OwnedDelta, SetField};

use live_edit::config::UpdateLayerBackend;
use live_edit::domain::delta::StorageMutation;
use live_edit::domain::types::{Field, FieldValue, ObjectState};
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::Metrics;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::lua::{encode_mutations, is_fast_path};
use live_edit::storage::structural::CommitOutcome;
use live_edit::storage::test_support::structural_commit::{
    dump_object_keyspace, CommitTokens, DumpedValue, ObjectKeyspaceDump,
};
use live_edit::storage::test_support::TestValkey;

use crate::common::owned_delta::delta_to_ompf_bytes;
use crate::storage::b_qpf_tests::{
    ext_hash, ext_hash_hex, ext_hash_vec, ext_ref_released, ext_store, obj_ext_sorted,
    pending_members, pre_store, EXT_SIZE, REASON_REWRITE, REASON_SURPLUS,
};
use crate::storage::snapshot_read_atomic_test::{
    create, create_cmd, create_session, delete_cmd, every_shape_state, layer_store, structural,
    CUBE, SIBLING, WORLD,
};
use crate::storage::structural_commit_test::{
    hand_built, Fixture, CLIENT_TS, CUBE_FIELD, NEW_SECTION, WRITER,
};

/// The hash under test: `every_shape_state`'s `CUBE` field `mesh` holds it,
/// and each case adds rows holding it.
const H: u8 = 0xAB;
/// `every_shape_state`'s `CUBE` sample at `1.5` holds it. Never drifted, so
/// deleting `CUBE` releases it as `rewrite` on both paths.
const CD: u8 = 0xCD;
/// Held by one row on `SIBLING` that no commit here touches.
const UNTOUCHED: u8 = 0xEF;
/// The count planted for [`UNTOUCHED`], whose one row holds it.
const UNTOUCHED_PLANTED: i64 = 4;
/// The field key [`UNTOUCHED`]'s row is stored under.
const UNTOUCHED_FIELD: &str = "tex";
/// The field key the cases' extra [`H`] rows are stored under.
const H_FIELD: &str = "skin";

/// The WATCH path's layer, beside the structural side's.
const WATCH_LAYER: &str = "ext-drift-watch";

/// An external field holding `tag`.
fn external_field(section_id: u64, key_name: &str, tag: u8) -> Field {
    Field {
        section_id,
        key_name: key_name.to_owned(),
        value: FieldValue::External {
            hash: ext_hash(tag),
            size: EXT_SIZE,
            ext_value_index: 0,
        },
        set_order: 5,
        authoring_set_order: 105,
        version: 1,
    }
}

/// `every_shape_state`, plus a [`H_FIELD`] field holding [`H`] on each of
/// `extra_h_sections`, plus [`UNTOUCHED`]'s one row on `SIBLING`. So `H` is
/// held by `1 + extra_h_sections.len()` rows, `CD` by one, `UNTOUCHED` by one.
fn state_with(extra_h_sections: &[u64]) -> ObjectState {
    let mut state = every_shape_state();
    let rows = extra_h_sections
        .iter()
        .map(|section| external_field(*section, H_FIELD, H))
        .chain([external_field(SIBLING, UNTOUCHED_FIELD, UNTOUCHED)]);
    for field in rows {
        state
            .fields
            .insert((field.section_id, field.key_name.clone()), field);
    }
    state
}

/// `obj:ext` as `(hash_hex, count)` pairs, sorted.
fn ext_index(rows: &[(u8, i64)]) -> Vec<(String, i64)> {
    let mut rows: Vec<(String, i64)> = rows
        .iter()
        .map(|(tag, count)| (ext_hash_hex(*tag), *count))
        .collect();
    rows.sort();
    rows
}

/// What one commit released: the change in each reason's counter, and the
/// hashes of the pending members it added.
#[derive(Debug, PartialEq, Eq)]
struct Released {
    rewrite: u64,
    surplus_acquire: u64,
    hashes: BTreeSet<String>,
}

fn released(rewrite: u64, surplus_acquire: u64, hashes: &[u8]) -> Released {
    Released {
        rewrite,
        surplus_acquire,
        hashes: hashes.iter().map(|tag| ext_hash_hex(*tag)).collect(),
    }
}

/// A delta that creates a section under `WORLD` (so the WATCH path takes its
/// snapshot branch) and sets `(section, key)` fields to [`H`].
fn create_and_set_h(fields: &[(u64, &str)]) -> OwnedDelta {
    OwnedDelta {
        struct_commands: vec![create_cmd(WORLD, NEW_SECTION, "Lamp")],
        set_fields: fields
            .iter()
            .enumerate()
            .map(|(i, (section_id, key))| SetField {
                section_id: *section_id,
                key_name: (*key).to_owned(),
                value_or_ext_hash: ext_hash_vec(H),
                ext_value_size: EXT_SIZE,
                ext_value_index: 1,
                set_order: i as u64,
            })
            .collect(),
        ..Default::default()
    }
}

/// The two objects one case compares: the same state, created in one session
/// as two layers. The structural side commits through `commit_structural`,
/// the WATCH side through `update_layer` on the `Txn` backend. Each side's
/// store has its own metrics registry, so release counts are per side.
struct Twins {
    fx: Fixture,
    structural_metrics: Arc<Metrics>,
    structural_ls: LayerStore,
    watch_oid: u64,
    watch_metrics: Arc<Metrics>,
    watch_ls: LayerStore,
}

/// Seed `state` into both objects, require `create_layer` to have indexed it
/// exactly as `indexed` says, then plant `planted` (`Some(count)` writes the
/// count, `None` removes the field) on both.
async fn twins(
    layer: &'static str,
    state: &ObjectState,
    indexed: &[(u8, i64)],
    planted: &[(u8, Option<i64>)],
) -> Twins {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_session(&tv, layer).await;
    let oid = create(&ls, &session_id, layer, state).await;
    let watch_oid = create(&ls, &session_id, WATCH_LAYER, state).await;
    let structural_metrics = Arc::new(crate::common::metrics_scrape_only(&format!(
        "{layer}-structural"
    )));
    let watch_metrics = Arc::new(crate::common::metrics_scrape_only(&format!(
        "{layer}-watch"
    )));
    let twins = Twins {
        structural_ls: layer_store(&tv).with_metrics(structural_metrics.clone()),
        watch_ls: layer_store(&tv)
            .with_update_layer_backend(UpdateLayerBackend::Txn)
            .with_metrics(watch_metrics.clone()),
        structural_metrics,
        watch_metrics,
        watch_oid,
        fx: Fixture {
            tv,
            ls,
            session_id,
            layer,
            oid,
        },
    };
    for oid in [twins.fx.oid, twins.watch_oid] {
        assert_eq!(
            obj_ext_sorted(&twins.fx.tv, oid).await,
            ext_index(indexed),
            "fixture precondition: create_layer indexes the seed's row count per hash"
        );
        for (tag, count) in planted {
            plant(&twins.fx.tv, oid, *tag, *count).await;
        }
    }
    assert_eq!(
        comparable_dump(&twins.fx.tv, twins.fx.oid).await,
        comparable_dump(&twins.fx.tv, twins.watch_oid).await,
        "fixture precondition: the two objects are seeded and drifted identically"
    );
    twins
}

/// Plant `count` for `tag` in object `oid`'s `obj:ext` out of band, or remove
/// the field (`None`), which must exist.
async fn plant(tv: &TestValkey, oid: u64, tag: u8, count: Option<i64>) {
    let key = tv.key_schema().obj_ext_index(&oid.to_string());
    match count {
        Some(count) => {
            let _: () = tv
                .client()
                .pool
                .hset(key, [(ext_hash_hex(tag), count.to_string())])
                .await
                .expect("HSET obj:ext must succeed");
        }
        None => {
            let removed: u64 = tv
                .client()
                .pool
                .hdel(key, ext_hash_hex(tag))
                .await
                .expect("HDEL obj:ext must succeed");
            assert_eq!(removed, 1, "fixture precondition: the field was indexed");
        }
    }
}

/// Object `oid`'s keyspace, comparable across two layers of one session:
/// tokens normalized (the WATCH path mints its own) and the header's
/// `layer_name` replaced.
async fn comparable_dump(tv: &TestValkey, oid: u64) -> ObjectKeyspaceDump {
    let mut dump =
        dump_object_keyspace(tv.client(), tv.key_schema(), oid, CommitTokens::Normalized).await;
    if let Some(DumpedValue::Hash(header)) = dump.keys.get_mut("obj:hdr") {
        header.insert("layer_name".to_owned(), "<layer_name>".to_owned());
    }
    dump
}

/// The hashes of the pending members added since `before`.
async fn pending_hashes_since(tv: &TestValkey, before: &BTreeSet<String>) -> BTreeSet<String> {
    pending_members(tv)
        .await
        .into_iter()
        .filter(|member| !before.contains(member))
        .map(|member| member.split(':').next().unwrap_or_default().to_owned())
        .collect()
}

impl Twins {
    fn tv(&self) -> &TestValkey {
        &self.fx.tv
    }

    /// Commit `delta` on both objects, structural mode first, each after the
    /// pre-flight `+1` for `referenced` the handler pays; return what each
    /// released.
    async fn commit_both(&self, delta: OwnedDelta, referenced: &[u8]) -> (Released, Released) {
        assert!(
            !is_fast_path(&Some(delta.clone())),
            "fixture precondition: the delta must carry a structural command, or the WATCH path \
             skips its snapshot read and trusts the stored index too"
        );
        let hashes: Vec<[u8; 32]> = referenced.iter().map(|tag| ext_hash(*tag)).collect();
        let ext = ext_store(self.tv());

        pre_store(&ext, &hashes).await;
        let planned = self.fx.plan(delta.clone(), "tok-ext-drift").await;
        let structural = self
            .collect(&self.structural_metrics, async {
                match self
                    .fx
                    .commit_with(&self.structural_ls, &planned.commit)
                    .await
                    .0
                {
                    Ok(CommitOutcome::Committed { .. }) => {}
                    other => panic!("the structural commit must land; got {other:?}"),
                }
            })
            .await;

        pre_store(&ext, &hashes).await;
        let watch = self
            .collect(&self.watch_metrics, async {
                self.watch_ls
                    .update_layer(
                        &self.fx.session_id,
                        WATCH_LAYER,
                        self.watch_oid,
                        Some(delta.clone()),
                        Some(Bytes::from(delta_to_ompf_bytes(&delta, &[]))),
                        WRITER,
                        CLIENT_TS,
                    )
                    .await
                    .expect("the WATCH-path commit must land");
            })
            .await;
        (structural, watch)
    }

    /// Run `commit` and return what it released, by `metrics`' counters and
    /// the pending set.
    async fn collect(
        &self,
        metrics: &Metrics,
        commit: impl std::future::Future<Output = ()>,
    ) -> Released {
        let rewrite = ext_ref_released(metrics, REASON_REWRITE);
        let surplus = ext_ref_released(metrics, REASON_SURPLUS);
        let before: BTreeSet<String> = pending_members(self.tv()).await.into_iter().collect();
        commit.await;
        Released {
            rewrite: ext_ref_released(metrics, REASON_REWRITE) - rewrite,
            surplus_acquire: ext_ref_released(metrics, REASON_SURPLUS) - surplus,
            hashes: pending_hashes_since(self.tv(), &before).await,
        }
    }

    /// The structural side's settle drift so far, `(under, over)`:
    /// `live_edit_ext_index_drift_total{site="structural_settle"}` — the
    /// touched hashes whose `stored + delta` fell below / stood above the
    /// recount `untouched + ends` (b.nvf merge rule (c)).
    fn structural_settle_drift(&self) -> (u64, u64) {
        let text = render(&self.structural_metrics);
        let cell = |direction: &str| {
            sample_value(
                &text,
                "live_edit_ext_index_drift_total",
                &[("site", "structural_settle"), ("direction", direction)],
            )
            .unwrap_or_else(|| {
                panic!("structural_settle/{direction} must be zero-seeded, so never absent")
            }) as u64
        };
        (cell("under"), cell("over"))
    }

    /// Each object's `obj:ext`, structural side first.
    async fn ext_indexes(&self) -> (Vec<(String, i64)>, Vec<(String, i64)>) {
        (
            obj_ext_sorted(self.tv(), self.fx.oid).await,
            obj_ext_sorted(self.tv(), self.watch_oid).await,
        )
    }

    /// T7: every key but `obj:ext` and `obj:wrel` is equal on the two paths,
    /// and the untouched hash keeps its planted count on both.
    ///
    /// `obj:wrel`, the commit release ring, holds an entry naming what each
    /// path's commit released, and on a drifted index the two paths' releases
    /// differ by design (T5, T6). So each path's entry is checked against its
    /// own releases (`structural_released`, `watch_released`) instead, and the
    /// two entries are compared to each other only when the releases agree.
    async fn assert_untouched_and_other_keys_equal(
        &self,
        structural_released: &Released,
        watch_released: &Released,
    ) {
        let (structural_ext, watch_ext) = self.ext_indexes().await;
        let untouched = (ext_hash_hex(UNTOUCHED), UNTOUCHED_PLANTED);
        assert!(
            structural_ext.contains(&untouched) && watch_ext.contains(&untouched),
            "the untouched hash keeps its planted count on both paths; structural {structural_ext:?}, \
             WATCH {watch_ext:?}"
        );
        let mut structural = comparable_dump(self.tv(), self.fx.oid).await;
        let mut watched = comparable_dump(self.tv(), self.watch_oid).await;
        let structural_record = own_release_record(
            self.tv(),
            self.fx.oid,
            &structural,
            structural_released,
            "structural",
        )
        .await;
        let watch_record =
            own_release_record(self.tv(), self.watch_oid, &watched, watch_released, "WATCH").await;
        if structural_released == watch_released {
            assert_eq!(
                structural_record, watch_record,
                "the two paths released the same, so their ring entries name the same tokens"
            );
        }
        for dump in [&mut structural, &mut watched] {
            dump.keys.remove("obj:ext");
            dump.keys.remove("obj:wrel");
        }
        assert_eq!(
            structural, watched,
            "every key but obj:ext and obj:wrel is equal on the two paths"
        );
    }
}

/// Require object `oid`'s commit release ring (`obj:wrel`), as `dump` read
/// it, to agree with what its one commit released: when that commit released
/// anything, the ring's one entry (`|t|<token>`) names the commit's sequence
/// and exactly its release tokens — its hashes, and as many `rewrite` /
/// `surplus_acquire` tokens as the reason counters ticked; when it released
/// nothing, the commit wrote no entry, and none from an earlier commit exists
/// here. The ring holds nothing but its own field shapes (`|t|`, `|s|`,
/// `|n|`): no field is named after the writer's `client_id`. Returns the
/// entry's tokens, sorted (the paths plan their releases in different
/// orders, and the entry's one reader treats them as a set).
async fn own_release_record(
    tv: &TestValkey,
    oid: u64,
    dump: &ObjectKeyspaceDump,
    released: &Released,
    side: &str,
) -> Option<(u64, Vec<String>)> {
    let record = match dump.keys.get("obj:wrel") {
        None => None,
        Some(DumpedValue::Hash(ring)) => {
            assert!(
                ring.keys()
                    .all(|f| f.starts_with("|t|") || f.starts_with("|s|") || f == "|n|"),
                "{side}: the ring holds only its own field shapes, none named after {WRITER:?}; \
                 got {ring:?}"
            );
            let entries: Vec<&String> = ring
                .iter()
                .filter(|(field, _)| field.starts_with("|t|"))
                .map(|(_, value)| value)
                .collect();
            assert!(
                entries.len() <= 1,
                "{side}: one commit per object here, so at most one ring entry; got {ring:?}"
            );
            entries.first().map(|value| (*value).clone())
        }
        Some(other) => panic!("{side}: obj:wrel must be a hash; got {other:?}"),
    };
    let parsed = record.as_deref().map(|value| {
        let mut parts = value.split(' ');
        let seq: u64 = parts
            .next()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| panic!("{side}: the entry starts with a sequence; got {value:?}"));
        let mut tokens: Vec<String> = parts.map(str::to_owned).collect();
        tokens.sort();
        (seq, tokens)
    });
    if released.hashes.is_empty() {
        assert_eq!(
            parsed, None,
            "{side}: the commit released nothing, so it wrote no ring entry"
        );
        return None;
    }
    let seq = crate::common::read_header_u64(tv, oid, "seq")
        .await
        .expect("the header carries a sequence");
    let (recorded_seq, tokens) = parsed
        .unwrap_or_else(|| panic!("{side}: the commit released {released:?} but wrote no entry"));
    assert_eq!(
        recorded_seq, seq,
        "{side}: the ring entry names the releasing commit"
    );
    let hashes: BTreeSet<String> = tokens
        .iter()
        .map(|t| t.rsplit(':').next().unwrap_or_default().to_owned())
        .collect();
    let reasons = |reason: &str| {
        tokens
            .iter()
            .filter(|t| t.split(':').next() == Some(reason))
            .count() as u64
    };
    assert_eq!(
        (reasons(REASON_REWRITE), reasons(REASON_SURPLUS), hashes),
        (
            released.rewrite,
            released.surplus_acquire,
            released.hashes.clone()
        ),
        "{side}: the ring entry names exactly the commit's releases; got {tokens:?}"
    );
    assert_eq!(
        tokens.len() as u64,
        released.rewrite + released.surplus_acquire,
        "{side}: one token per release; got {tokens:?}"
    );
    Some((recorded_seq, tokens))
}

/// **T1 — the decider's case, and the regression the floor fixes.** Three
/// rows hold `H`, indexed as one. Deleting `CUBE` cascades over one of them
/// (and `CD`'s one row). `stored + delta` is `0` while two rows still hold
/// `H`: the floor writes `2` and releases nothing for `H`, which is the WATCH
/// path's recount — the two paths agree on the keyspace and the releases.
/// Applying the transitions to the stored count (`after = before + delta`)
/// would `HDEL` `H` and release it as `rewrite`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_delete_to_zero_on_an_under_count_keeps_the_row_count_and_releases_nothing() {
    let twins = twins(
        "ext-drift-t1",
        &state_with(&[WORLD, SIBLING]),
        &[(H, 3), (CD, 1), (UNTOUCHED, 1)],
        &[(H, Some(1)), (UNTOUCHED, Some(UNTOUCHED_PLANTED))],
    )
    .await;

    let (structural, watch) = twins
        .commit_both(structural(vec![delete_cmd(WORLD, CUBE)]), &[])
        .await;

    let (structural_ext, watch_ext) = twins.ext_indexes().await;
    let expected = ext_index(&[(H, 2), (UNTOUCHED, UNTOUCHED_PLANTED)]);
    assert_eq!(
        structural_ext, expected,
        "structural mode writes H's row count (2), not stored + delta (0), and CD's last row \
         leaving drops it"
    );
    assert_eq!(watch_ext, expected, "the WATCH path writes the same");
    assert_eq!(
        structural,
        released(1, 0, &[CD]),
        "structural mode releases only CD, as rewrite: H is still held"
    );
    assert_eq!(watch, structural, "the WATCH path releases the same");
    assert_eq!(
        twins.structural_settle_drift(),
        (1, 0),
        "the settle counts H as under-drift (stored + delta 0 below its recount 2); CD's \
         stored + delta equals its recount (0)"
    );
    twins
        .assert_untouched_and_other_keys_equal(&structural, &watch)
        .await;
}

/// **T3 — an under-count the commit changes.** Two rows hold `H`, indexed as
/// one; the delta adds a third. Both paths write `3` (structural mode's floor,
/// the WATCH path's recount plus one), and both hand back the pre-flight `+1`
/// as `surplus_acquire`, since the object already held `H`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_write_on_an_under_count_raises_it_to_the_row_count_as_the_watch_path_does() {
    let twins = twins(
        "ext-drift-t3",
        &state_with(&[WORLD]),
        &[(H, 2), (CD, 1), (UNTOUCHED, 1)],
        &[(H, Some(1)), (UNTOUCHED, Some(UNTOUCHED_PLANTED))],
    )
    .await;

    let (structural, watch) = twins
        .commit_both(create_and_set_h(&[(SIBLING, H_FIELD)]), &[H])
        .await;

    let (structural_ext, watch_ext) = twins.ext_indexes().await;
    let expected = ext_index(&[(H, 3), (CD, 1), (UNTOUCHED, UNTOUCHED_PLANTED)]);
    assert_eq!(
        structural_ext, expected,
        "structural mode writes H's row count (3), not stored + delta (2)"
    );
    assert_eq!(watch_ext, expected, "the WATCH path writes the same");
    assert_eq!(
        structural,
        released(0, 1, &[H]),
        "the pre-flight +1 for H is handed back as surplus_acquire"
    );
    assert_eq!(watch, structural, "the WATCH path releases the same");
    twins
        .assert_untouched_and_other_keys_equal(&structural, &watch)
        .await;
}

/// **T4 — an under-count the commit leaves unchanged (a pinned difference).**
/// Two rows hold `H`, indexed as one; the delta rewrites one of them to `H`
/// again. Structural mode repairs the count to `2`; the WATCH path writes
/// nothing (`delta == 0`) and keeps `1`. Both hand back the pre-flight `+1`:
/// structural mode because the rewritten row held `H` before and after, the
/// WATCH path because its recount is above zero.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_same_hash_rewrite_repairs_an_under_count_the_watch_path_leaves() {
    let twins = twins(
        "ext-drift-t4",
        &state_with(&[WORLD]),
        &[(H, 2), (CD, 1), (UNTOUCHED, 1)],
        &[(H, Some(1)), (UNTOUCHED, Some(UNTOUCHED_PLANTED))],
    )
    .await;

    let (structural, watch) = twins
        .commit_both(create_and_set_h(&[(WORLD, H_FIELD)]), &[H])
        .await;

    let (structural_ext, watch_ext) = twins.ext_indexes().await;
    assert_eq!(
        structural_ext,
        ext_index(&[(H, 2), (CD, 1), (UNTOUCHED, UNTOUCHED_PLANTED)]),
        "structural mode raises H to its row count (2) although the commit nets no change"
    );
    assert_eq!(
        watch_ext,
        ext_index(&[(H, 1), (CD, 1), (UNTOUCHED, UNTOUCHED_PLANTED)]),
        "the WATCH path writes nothing when the commit nets no change, keeping the planted 1"
    );
    assert_eq!(
        structural,
        released(0, 1, &[H]),
        "structural mode hands back the pre-flight +1: the rewritten row held H before and after"
    );
    assert_eq!(watch, structural, "the WATCH path releases the same");
    twins
        .assert_untouched_and_other_keys_equal(&structural, &watch)
        .await;
}

/// **T5 — an over-count (a pinned difference, the leak direction).** One row
/// holds `H`, indexed as three; deleting `CUBE` deletes it. Structural mode
/// keeps `stored + delta = 2` and releases nothing for `H`: the floor never
/// lowers a count or causes a release. The WATCH path recounts, drops `H` and
/// releases it as `rewrite`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_delete_on_an_over_count_keeps_the_excess_where_the_watch_path_releases() {
    let twins = twins(
        "ext-drift-t5",
        &state_with(&[]),
        &[(H, 1), (CD, 1), (UNTOUCHED, 1)],
        &[(H, Some(3)), (UNTOUCHED, Some(UNTOUCHED_PLANTED))],
    )
    .await;

    let (structural, watch) = twins
        .commit_both(structural(vec![delete_cmd(WORLD, CUBE)]), &[])
        .await;

    let (structural_ext, watch_ext) = twins.ext_indexes().await;
    assert_eq!(
        structural_ext,
        ext_index(&[(H, 2), (UNTOUCHED, UNTOUCHED_PLANTED)]),
        "structural mode keeps H at stored + delta (2)"
    );
    assert_eq!(
        watch_ext,
        ext_index(&[(UNTOUCHED, UNTOUCHED_PLANTED)]),
        "the WATCH path drops H: its recount reaches 0"
    );
    assert_eq!(
        structural,
        released(1, 0, &[CD]),
        "structural mode releases only CD"
    );
    assert_eq!(
        watch,
        released(2, 0, &[H, CD]),
        "the WATCH path releases H and CD, both as rewrite"
    );
    assert_eq!(
        twins.structural_settle_drift(),
        (0, 1),
        "the settle counts H as over-drift (stored + delta 2 above its recount 0), kept for \
         the audit to lower; CD matches"
    );
    twins
        .assert_untouched_and_other_keys_equal(&structural, &watch)
        .await;
}

/// **T6a — absent but held, the row deleted (a pinned difference, the leak
/// direction).** One row holds `H`, which the index does not name; deleting
/// `CUBE` deletes it. Structural mode's count is `max(0 - 1, 0) = 0` on a field
/// that does not exist, so it writes nothing, and with no stored count it
/// releases nothing. The WATCH path recounts one row and releases `H` as
/// `rewrite`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_delete_of_a_held_hash_the_index_lacks_releases_nothing() {
    let twins = twins(
        "ext-drift-t6a",
        &state_with(&[]),
        &[(H, 1), (CD, 1), (UNTOUCHED, 1)],
        &[(H, None), (UNTOUCHED, Some(UNTOUCHED_PLANTED))],
    )
    .await;

    let (structural, watch) = twins
        .commit_both(structural(vec![delete_cmd(WORLD, CUBE)]), &[])
        .await;

    let (structural_ext, watch_ext) = twins.ext_indexes().await;
    let expected = ext_index(&[(UNTOUCHED, UNTOUCHED_PLANTED)]);
    assert_eq!(
        structural_ext, expected,
        "structural mode writes nothing for H"
    );
    assert_eq!(
        watch_ext, expected,
        "the WATCH path's HDEL of H removes nothing"
    );
    assert_eq!(
        structural,
        released(1, 0, &[CD]),
        "structural mode releases only CD"
    );
    assert_eq!(
        watch,
        released(2, 0, &[H, CD]),
        "the WATCH path releases H and CD, both as rewrite"
    );
    assert_eq!(
        twins.structural_settle_drift(),
        (1, 0),
        "the settle counts H as under-drift (absent, so stored + delta -1, below its recount \
         0); CD matches"
    );
    twins
        .assert_untouched_and_other_keys_equal(&structural, &watch)
        .await;
}

/// **T6b — absent but held, a new row added (a pinned difference, the leak
/// direction).** One untouched row holds `H`, which the index does not name;
/// the delta adds a second row holding `H`. Both paths write `2`. Structural
/// mode sees no stored count and no row holding `H` both before and after, so
/// it keeps the pre-flight `+1`. The WATCH path recounts one row and hands
/// the `+1` back as `surplus_acquire`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_write_of_a_held_hash_the_index_lacks_keeps_the_pre_flight_increment() {
    let twins = twins(
        "ext-drift-t6b",
        &state_with(&[]),
        &[(H, 1), (CD, 1), (UNTOUCHED, 1)],
        &[(H, None), (UNTOUCHED, Some(UNTOUCHED_PLANTED))],
    )
    .await;

    let (structural, watch) = twins
        .commit_both(create_and_set_h(&[(SIBLING, H_FIELD)]), &[H])
        .await;

    let (structural_ext, watch_ext) = twins.ext_indexes().await;
    let expected = ext_index(&[(H, 2), (CD, 1), (UNTOUCHED, UNTOUCHED_PLANTED)]);
    assert_eq!(
        structural_ext, expected,
        "structural mode writes H's row count (2), not stored + delta (1)"
    );
    assert_eq!(watch_ext, expected, "the WATCH path writes the same");
    assert_eq!(
        structural,
        released(0, 0, &[]),
        "structural mode releases nothing: the pre-flight +1 is kept"
    );
    assert_eq!(
        watch,
        released(0, 1, &[H]),
        "the WATCH path hands the pre-flight +1 back as surplus_acquire"
    );
    twins
        .assert_untouched_and_other_keys_equal(&structural, &watch)
        .await;
}

/// **T2 — a cache wrong about an untouched row cannot release.** The store
/// holds two rows with `H` (`CUBE`'s `mesh` and `WORLD`'s [`H_FIELD`]),
/// indexed as two. The commit is hand-built on a pre-plan state that lacks
/// `WORLD`'s row, and deletes `CUBE`'s, the only row that state believes
/// holds `H`. The `DF` existence digest passes (the row exists), so the
/// commit lands. Its untouched-row count for `H` is `0` and no touched row
/// ends holding `H`, so the cache's recount is `0`. The floor still writes
/// `stored - 1 = 1` and releases nothing. Writing the recount instead
/// (`after = untouched + ends`) would `HDEL` `H` and release it as
/// `rewrite` while `WORLD`'s row still holds it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_commit_on_a_cache_missing_an_untouched_row_does_not_release_the_hash() {
    commit_on_a_wrong_cache_deleting_cubes_row(
        "ext-drift-t2",
        |believed| {
            assert!(
                believed
                    .fields
                    .remove(&(WORLD, H_FIELD.to_owned()))
                    .is_some(),
                "fixture precondition: the full read returned WORLD's row holding H"
            );
        },
        // `H`, with no untouched row the cache believes holds it.
        vec!["1".to_owned(), ext_hash_hex(H), "0".to_owned()],
    )
    .await;
}

/// **A cache wrong about the touched row cannot release either.** The store
/// is T2's: two rows hold `H` (`CUBE`'s `mesh` and `WORLD`'s [`H_FIELD`]),
/// indexed as two. The pre-plan state believes `CUBE`'s `mesh` is inline, so
/// no touched row names `H` in the cache and the commit's untouched-row count
/// block carries no entry for it (`K = 0`). Deleting that row (the `DF`
/// existence digest passes: the row exists) reads `H` from the stored row,
/// so the script settles a hash the block does not carry: its untouched count
/// reads 0, and the floor writes `stored - 1 = 1` and releases nothing. This
/// is the one shape that reaches the script's fallback for a hash with no
/// block entry; the other named cases plan on a correct cache, and T2's wrong
/// belief still leaves `H` an entry.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_commit_on_a_cache_wrong_about_the_touched_row_does_not_release_the_hash() {
    commit_on_a_wrong_cache_deleting_cubes_row(
        "ext-drift-wrong-touched",
        |believed| {
            let mesh = believed
                .fields
                .get_mut(&(CUBE, CUBE_FIELD.to_owned()))
                .expect("fixture precondition: the full read returned CUBE's row");
            assert!(
                matches!(mesh.value, FieldValue::External { hash, .. } if hash == ext_hash(H)),
                "fixture precondition: CUBE's row holds H"
            );
            mesh.value = FieldValue::Inline(vec![0x5A]);
        },
        vec!["0".to_owned()],
    )
    .await;
}

/// Seed T2's store (`H` held by `CUBE`'s `mesh` and `WORLD`'s [`H_FIELD`],
/// indexed as two; [`UNTOUCHED`] planted), hand-build a commit deleting
/// `CUBE`'s `mesh` on the pre-plan state `believe` makes wrong, require its
/// untouched-row count block to be `expected_block`, commit it, and require
/// `H` to end at `stored - 1 = 1` with nothing released.
async fn commit_on_a_wrong_cache_deleting_cubes_row(
    layer: &'static str,
    believe: impl FnOnce(&mut ObjectState),
    expected_block: Vec<String>,
) {
    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only(layer));
    let ls = layer_store(&tv).with_metrics(metrics.clone());
    let session_id = create_session(&tv, layer).await;
    let oid = create(&ls, &session_id, layer, &state_with(&[WORLD])).await;
    let fx = Fixture {
        tv,
        ls,
        session_id,
        layer,
        oid,
    };
    assert_eq!(
        obj_ext_sorted(&fx.tv, oid).await,
        ext_index(&[(H, 2), (CD, 1), (UNTOUCHED, 1)]),
        "fixture precondition: the store indexes both rows holding H"
    );
    plant(&fx.tv, oid, UNTOUCHED, Some(UNTOUCHED_PLANTED)).await;

    // The base plan only lends its cut; its mutations are replaced.
    let base = fx
        .plan(
            structural(vec![create_cmd(WORLD, NEW_SECTION, "Lamp")]),
            "tok-ext-wrong-cache",
        )
        .await;
    let mut believed = base.pre_state.clone();
    believe(&mut believed);
    let commit = hand_built(
        &base,
        &believed,
        vec![StorageMutation::DeleteField {
            section_id: CUBE,
            key_name: CUBE_FIELD.to_owned(),
        }],
    );
    let encoded = encode_mutations(
        &commit.mutations,
        &commit.pre_image_digests,
        commit.next_set_order,
        fx.keys(),
        &fx.oid_str(),
    )
    .expect("the hand-built plan encodes");
    assert_eq!(
        encoded.untouched_ext_block, expected_block,
        "fixture precondition: the untouched-row count block the wrong belief yields"
    );
    let ext_metrics = (
        ext_ref_released(&metrics, REASON_REWRITE),
        ext_ref_released(&metrics, REASON_SURPLUS),
    );
    let before: BTreeSet<String> = pending_members(&fx.tv).await.into_iter().collect();

    fx.commit_landed(&commit).await;

    assert!(
        fx.exists(&fx.field_key(WORLD, H_FIELD)).await
            && !fx.exists(&fx.field_key(CUBE, CUBE_FIELD)).await,
        "the commit deleted CUBE's row and left WORLD's"
    );
    assert_eq!(
        obj_ext_sorted(&fx.tv, oid).await,
        ext_index(&[(H, 1), (CD, 1), (UNTOUCHED, UNTOUCHED_PLANTED)]),
        "H ends at stored - 1 (1), not the cache's recount (0) and not a row count missing \
         from the block; the untouched hash keeps its planted count"
    );
    assert_eq!(
        (
            ext_ref_released(&metrics, REASON_REWRITE) - ext_metrics.0,
            ext_ref_released(&metrics, REASON_SURPLUS) - ext_metrics.1,
            pending_hashes_since(&fx.tv, &before).await,
        ),
        (0, 0, BTreeSet::new()),
        "the commit releases nothing: WORLD's row still holds H"
    );
}
