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

//! External-value clean-up's reference walk (b.fsz MR-D4, D4-T1 and D4-T9;
//! `storage::retirement::walk_object_references`): what a retiring object's
//! release reads in place of a decode of the whole object.
//!
//! The walk pages `obj:flds` and `obj:tss` by rank, reads each indexed value
//! row by the bounded value-row read and decodes its hash by the readers'
//! strict rule (a row holding a `value` names no hash), then pages `obj:ext` by
//! `HSCAN`, and folds everything into one set of hashes. The oracle here is
//! independent of the walk's code: every indexed row read whole (`HGETALL`)
//! and decoded by that rule, united with `obj:ext`'s fields. The walk's set
//! must equal it, and must contain what the full decode's
//! `collect_external_hashes` names together with `obj:ext` — the union the
//! release read before the walk replaced it.
//!
//! | Test | What it pins |
//! |---|---|
//! | [`reference_walk_releases_exactly_the_rows_by_the_readers_rule_and_obj_ext_at_every_page_size`] | over inline, external and legacy dual-shape rows, a row whose hash `obj:ext` does not count, an `obj:ext` field no row names, an undecodable `obj:ext` field, an index member with no row and a row no index names: the set equals the oracle at pages of 1, 2 and the reference page, and contains the full decode's union; the undecodable field is counted, not released |
//! | [`reference_walk_of_a_signed_zero_pair_releases_both_rows_hashes`] | a legacy `0.0` / `-0.0` time-sample pair written straight into the store (b.8jc refuses `-0.0` at ingest) with two different hashes: the walk releases both rows' hashes, where the full decode resolves the pair to one sample |
//! | [`reference_walk_of_an_object_without_obj_ext_releases_its_rows_hashes`] | a pre-b.qpf object with no `obj:ext`: the rows alone |
//! | [`reference_walk_of_a_header_without_a_numeric_seq_walks_on`] | `seq` absent, `seq` not a number, and the single-column `superseded=1` header with every other key behind it: the walk releases what the rest of the object holds (SR-1.5) |
//! | [`reference_walk_of_a_superseded_ghost_header_is_the_empty_set`] | the retirement fence's ghost with no keys behind it: `Ok`, nothing to release |
//! | [`reference_walk_of_an_absent_header_is_not_found_and_records_it`] | `NotFound`, recorded once as `snapshot_read_total{consumer="ext_gc",path="unfenced",outcome="not_found"}` |
//! | [`reference_walk_records_one_snapshot_read_per_walk_under_ext_gc_unfenced`] | D4-T9: one `ok` per walk however many pages it took, its duration and size histograms with the walk's own tallies, nothing on the snapshot consumers' cells; and a `DeleteLayer` ticks it exactly once more |
//! | [`retirement_release_counts_each_undecodable_obj_ext_field`] | the release path (`schedule_old_object_ext_gc`) counts each undecodable field on `ext_ref_undecodable_total{site="retirement_index_field"}` and releases every decodable reference |
//! | [`walk_failures::a_walk_meeting_a_wrong_typed_key_ticks_exactly_its_family_once`] | a wrong-typed `obj:flds`, `obj:tss`, value row or `obj:ext`: `Internal`, one `permanent_error`, exactly its own `key_family` cell once |
//! | `walk_failures::injected::a_walk_failing_transiently_answers_aborted_and_records_transient_error` | a timeout: `Aborted`, one `transient_error`, no refusal kind or family ticked |
//! | `walk_failures::injected::a_walk_failing_permanently_answers_internal_and_ticks_its_refusal_kind` | a protocol error: `Internal`, one `permanent_error`, `kind="protocol"` once |
//! | `walk_failures::injected::a_walk_refused_loading_or_busy_is_retriable_and_counts_nothing` | every `-LOADING` / `-BUSY` shape: `Aborted` with the retry prefix, nothing counted |
//! | `walk_failures::injected::a_delete_whose_walk_fails_still_tears_the_object_down_and_releases_nothing` | a `DeleteLayer` whose walk fails transiently or permanently answers OK, tears the object down and enqueues no release |
//!
//! A malformed `hash` column fails the walk `Internal` beside every snapshot
//! read: `g3_external_hash_tests.rs`. The walk's thread identity (its windows
//! on a `watch-read` thread) is pinned in `src/storage/fanout_thread.rs`
//! (D4-T8). Requires Valkey (`TEST_VALKEY_URL`); the `walk_failures::injected`
//! rows arm the snapshot-read seam and need the `test-fault-injection`
//! feature.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use fred::interfaces::{HashesInterface, KeysInterface, SortedSetsInterface};
use ordered_float::OrderedFloat;

use live_edit::domain::types::{FieldValue, TimeSample};
use live_edit::error::ServiceError;
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::Metrics;
use live_edit::storage::diff_read::{
    read_object_snapshot, snapshot_to_object_state, SnapshotRead, SnapshotReadConsumer,
};
use live_edit::storage::external_values::{collect_external_hashes, MockExternalValueStore};
use live_edit::storage::keys::KeySchema;
use live_edit::storage::layer::schedule_old_object_ext_gc_for_test;
use live_edit::storage::retirement::{
    walk_object_references, RetirementPaging, RetirementWalk, WalkedReferences,
};
use live_edit::storage::test_support::TestValkey;

use crate::storage::snapshot_read_atomic_test::{
    create, create_session, every_shape_state, ext_hash, layer_store, mock_ext_store, CUBE,
    SIBLING, WORLD,
};

/// The every-shape object's external field (`(CUBE, "mesh")`) hash.
const MESH: [u8; 32] = [0xAB; 32];
/// The every-shape object's external time sample (`(CUBE, 1.5)`) hash.
const SAMPLE: [u8; 32] = [0xCD; 32];

/// The lowercase hex the store keeps a hash in.
fn hex(hash: &[u8; 32]) -> String {
    hash.iter().map(|b| format!("{b:02x}")).collect()
}

/// The oracle's strict hash decode: exactly 64 hex digits.
fn decode(hex: &str) -> Option<[u8; 32]> {
    if hex.len() != 64 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).ok()?;
    }
    Some(out)
}

/// The reference pod's page: `P` index members (what a `LayerStore` starts
/// with).
fn reference_page_len() -> u64 {
    RetirementPaging::reference().page_len()
}

/// The walk of `oid` at `page_len` index members per page, unreserved.
async fn walk(tv: &TestValkey, oid: u64, page_len: u64) -> Result<WalkedReferences, ServiceError> {
    walk_object_references(
        tv.client(),
        tv.key_schema(),
        &oid.to_string(),
        &RetirementWalk::unreserved().with_page_len(page_len),
    )
    .await
}

/// The members of a sorted index, by rank.
async fn index_members(tv: &TestValkey, key: &str) -> Vec<String> {
    tv.client()
        .pool
        .zrange(key, 0, -1, None, false, None, false)
        .await
        .expect("ZRANGE of the value index must succeed")
}

/// The oracle's rows half: every row `obj:flds` and `obj:tss` name, read whole
/// and decoded by the readers' rule — a row holding a `value` (beside a `hash`
/// or not) is inline and names no hash; a row holding a `hash` alone names
/// it; an absent row names nothing.
async fn readers_rule_rows(tv: &TestValkey, oid: u64) -> BTreeSet<[u8; 32]> {
    let (keys, oid_str) = (tv.key_schema(), oid.to_string());
    let mut row_keys = Vec::new();
    for member in index_members(tv, &keys.obj_fields_index(&oid_str)).await {
        row_keys.push(keys.obj_field_from_member(&oid_str, &member));
    }
    for member in index_members(tv, &keys.obj_time_samples_index(&oid_str)).await {
        row_keys.push(keys.obj_ts_from_member(&oid_str, &member));
    }
    let mut hashes = BTreeSet::new();
    for key in row_keys {
        let row: HashMap<String, String> = tv
            .client()
            .pool
            .hgetall(&key)
            .await
            .expect("HGETALL of a value row must succeed");
        if row.contains_key("value") {
            continue;
        }
        if let Some(column) = row.get("hash") {
            hashes
                .insert(decode(column).unwrap_or_else(|| panic!("fixture: {key}'s hash decodes")));
        }
    }
    hashes
}

/// The oracle's index half: `obj:ext`'s decodable fields, and how many it
/// holds that do not decode.
async fn ext_index(tv: &TestValkey, oid: u64) -> (BTreeSet<[u8; 32]>, u64) {
    let fields: Vec<String> = tv
        .client()
        .pool
        .hkeys(tv.key_schema().obj_ext_index(&oid.to_string()))
        .await
        .expect("HKEYS obj:ext must succeed");
    let mut hashes = BTreeSet::new();
    let mut undecodable = 0;
    for field in fields {
        match decode(&field) {
            Some(hash) => {
                hashes.insert(hash);
            }
            None => undecodable += 1,
        }
    }
    (hashes, undecodable)
}

/// What the release read before the walk: `collect_external_hashes` over the
/// full decode of the object (the catch-up's atomic read).
async fn full_decode_hashes(tv: &TestValkey, oid: u64) -> BTreeSet<[u8; 32]> {
    let read = read_object_snapshot(
        tv.client(),
        tv.key_schema(),
        &oid.to_string(),
        SnapshotReadConsumer::Catchup,
    )
    .await
    .expect("the full read of the fixture must succeed");
    let SnapshotRead::Snapshot(snapshot) = read else {
        panic!("fixture: the object is within the atomic-read ceilings");
    };
    collect_external_hashes(&snapshot_to_object_state(&snapshot.state))
        .into_iter()
        .collect()
}

async fn hset(tv: &TestValkey, key: &str, columns: &[(&str, &str)]) {
    let _: () = tv
        .client()
        .pool
        .hset(key, columns.to_vec())
        .await
        .expect("HSET of the planted key must succeed");
}

async fn zadd(tv: &TestValkey, key: &str, member: String) {
    let _: i64 = tv
        .client()
        .pool
        .zadd(key, None, None, false, false, (1_000.0, member))
        .await
        .expect("ZADD of the planted member must succeed");
}

/// The every-shape object (`snapshot_read_atomic_test::every_shape_state`:
/// an inline and an external field, an inline and an external time sample,
/// `obj:ext` counting the two external hashes) under `layer`, and its id.
async fn every_shape_object(tv: &TestValkey, layer: &str) -> u64 {
    let ls = layer_store(tv);
    let session_id = create_session(tv, layer).await;
    let oid = create(&ls, &session_id, layer, &every_shape_state()).await;
    let (indexed, _) = ext_index(tv, oid).await;
    assert_eq!(
        indexed,
        BTreeSet::from([MESH, SAMPLE]),
        "fixture: obj:ext counts the two external hashes the object was created with"
    );
    oid
}

/// D4-T1, the shapes: the walk's release set is exactly the readers'-rule
/// rows united with `obj:ext`, on every page size, and contains the full
/// decode's union.
///
/// Planted beside the every-shape object's rows:
/// - `(WORLD, "color")`, its inline row, given a `hash` of `0x11` too — the
///   legacy dual-shape row: inline by the readers' rule, so `0x11` (which
///   `obj:ext` does not count) is not released;
/// - `obj:ext` stripped of `SAMPLE`, which its row still names: released
///   through the row;
/// - `obj:ext` given `0x33`, which no row names: released through the index;
/// - `obj:ext` given a field that is no hash: counted as undecodable, never
///   released;
/// - an `obj:flds` member `(WORLD, "ghost")` with no row: read as absent;
/// - a row `(SIBLING, "stray")` of hash `0x44` no index names: not walked.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reference_walk_releases_exactly_the_rows_by_the_readers_rule_and_obj_ext_at_every_page_size(
) {
    let tv = TestValkey::setup().await;
    let oid = every_shape_object(&tv, "walk-shapes").await;
    let (keys, oid_str) = (tv.key_schema(), oid.to_string());
    let (world, sibling) = (WORLD.to_string(), SIBLING.to_string());
    let (dual, index_only, stray) = (ext_hash(0x11), ext_hash(0x33), ext_hash(0x44));

    hset(
        &tv,
        &keys.obj_field(&oid_str, &world, "color"),
        &[("hash", &hex(&dual)), ("size", "3")],
    )
    .await;
    let _: i64 = tv
        .client()
        .pool
        .hdel(keys.obj_ext_index(&oid_str), hex(&SAMPLE))
        .await
        .expect("HDEL obj:ext must succeed");
    hset(
        &tv,
        &keys.obj_ext_index(&oid_str),
        &[(&hex(&index_only), "1"), ("not-a-hash", "1")],
    )
    .await;
    zadd(
        &tv,
        &keys.obj_fields_index(&oid_str),
        KeySchema::field_member(&world, "ghost"),
    )
    .await;
    hset(
        &tv,
        &keys.obj_field(&oid_str, &sibling, "stray"),
        &[
            ("hash", &hex(&stray)),
            ("size", "9"),
            ("set_order", "9"),
            ("version", "1"),
        ],
    )
    .await;

    let rows = readers_rule_rows(&tv, oid).await;
    let (indexed, undecodable) = ext_index(&tv, oid).await;
    let oracle: BTreeSet<[u8; 32]> = rows.union(&indexed).copied().collect();
    assert_eq!(
        oracle,
        BTreeSet::from([MESH, SAMPLE, index_only]),
        "fixture self-check: the oracle names the external rows and the index-only field, not \
         the dual-shape row's or the unindexed row's hash"
    );
    let mut legacy_union = full_decode_hashes(&tv, oid).await;
    legacy_union.extend(indexed.iter().copied());
    let indexed_rows = (index_members(&tv, &keys.obj_fields_index(&oid_str))
        .await
        .len()
        + index_members(&tv, &keys.obj_time_samples_index(&oid_str))
            .await
            .len()) as u64;

    for page_len in [1, 2, reference_page_len()] {
        let walked = walk(&tv, oid, page_len)
            .await
            .unwrap_or_else(|e| panic!("page {page_len}: the walk must succeed; got {e:?}"));
        let set: BTreeSet<[u8; 32]> = walked.hashes.iter().copied().collect();
        assert_eq!(
            set, oracle,
            "page {page_len}: the walk releases exactly the indexed rows' hashes by the readers' \
             rule and obj:ext's fields"
        );
        assert!(
            set.is_superset(&legacy_union),
            "page {page_len}: the walk must release everything the full decode's union did; \
             missing {:?}",
            legacy_union.difference(&set).collect::<Vec<_>>()
        );
        assert!(
            !set.contains(&dual) && !set.contains(&stray),
            "page {page_len}: a dual-shape row is inline, and a row no index names is no part \
             of the object"
        );
        assert_eq!(
            walked.undecodable, undecodable,
            "page {page_len}: the undecodable obj:ext field is counted, not released"
        );
        assert_eq!(
            walked.rows, indexed_rows,
            "page {page_len}: every indexed row is read once, the ghost member's absent row \
             included"
        );
    }
}

/// D4-T1, the signed-zero pair: a legacy `(CUBE, 0.0)` / `(CUBE, -0.0)` pair,
/// each row its own stored key and `obj:tss` member, holding two different
/// hashes — written straight into the store, since this build refuses `-0.0`
/// at ingest (b.8jc). The walk reads each stored row and releases both
/// hashes; the full decode resolves the pair to one sample
/// (`domain::types::resolve_signed_zero_pair`), so the union the release read
/// before the walk named only one of them, and the other leaked.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reference_walk_of_a_signed_zero_pair_releases_both_rows_hashes() {
    let (positive, negative) = (ext_hash(0x5A), ext_hash(0xA5));
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let layer = "walk-signed-zero";
    let session_id = create_session(&tv, layer).await;
    let mut state = every_shape_state();
    state.time_samples.insert(
        (CUBE, OrderedFloat(0.0)),
        TimeSample {
            section_id: CUBE,
            time: OrderedFloat(0.0),
            value: FieldValue::External {
                hash: positive,
                size: 64,
                ext_value_index: 0,
            },
            version: 1,
        },
    );
    let oid = create(&ls, &session_id, layer, &state).await;
    let (keys, oid_str, cube) = (tv.key_schema(), oid.to_string(), CUBE.to_string());

    let mut row: HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(keys.obj_time_sample(&oid_str, &cube, 0.0))
        .await
        .expect("HGETALL the 0.0 row must succeed");
    assert_eq!(
        row.get("hash"),
        Some(&hex(&positive)),
        "fixture: the 0.0 row is external"
    );
    row.insert("hash".to_owned(), hex(&negative));
    let _: () = tv
        .client()
        .pool
        .hset(keys.obj_time_sample(&oid_str, &cube, -0.0), row)
        .await
        .expect("HSET the -0.0 row must succeed");
    zadd(
        &tv,
        &keys.obj_time_samples_index(&oid_str),
        KeySchema::time_sample_member(&cube, -0.0),
    )
    .await;

    let decoded = full_decode_hashes(&tv, oid).await;
    assert_eq!(
        [positive, negative]
            .iter()
            .filter(|h| decoded.contains(*h))
            .count(),
        1,
        "fixture: the full decode resolves the signed-zero pair to one sample, so it names one \
         of the two hashes; got {decoded:?}"
    );
    let walked = walk(&tv, oid, reference_page_len())
        .await
        .expect("the walk must succeed");
    assert!(
        walked.hashes.contains(&positive) && walked.hashes.contains(&negative),
        "the walk reads each stored row, so it releases both rows' hashes"
    );
}

/// D4-T1, a pre-b.qpf object: no `obj:ext` at all. The walk's set is its
/// rows' hashes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reference_walk_of_an_object_without_obj_ext_releases_its_rows_hashes() {
    let tv = TestValkey::setup().await;
    let oid = every_shape_object(&tv, "walk-no-ext").await;
    let _: i64 = tv
        .client()
        .pool
        .del(tv.key_schema().obj_ext_index(&oid.to_string()))
        .await
        .expect("DEL obj:ext must succeed");

    let walked = walk(&tv, oid, reference_page_len())
        .await
        .expect("the walk must succeed with no obj:ext");
    assert_eq!(
        walked.hashes.iter().copied().collect::<BTreeSet<_>>(),
        BTreeSet::from([MESH, SAMPLE]),
        "the rows' hashes alone, the rolling-deploy cover for an object materialized before \
         obj:ext existed"
    );
    assert_eq!(walked.undecodable, 0);
}

/// D4-T1, a header without a numeric `seq` (SR-1.5): `seq` absent, `seq` not
/// a number, and the retirement fence's single-column `superseded=1` header
/// (every other column gone) over the object's remaining keys. The walk goes
/// on and releases what the rest of the object holds — refusing would strand
/// every external value it references.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reference_walk_of_a_header_without_a_numeric_seq_walks_on() {
    let tv = TestValkey::setup().await;
    let oid = every_shape_object(&tv, "walk-no-seq").await;
    let header_key = tv.key_schema().obj_header(&oid.to_string());

    for shape in ["seq absent", "seq not a number", "superseded=1 alone"] {
        let _: i64 = tv
            .client()
            .pool
            .del(&header_key)
            .await
            .expect("DEL obj:hdr must succeed");
        match shape {
            "seq absent" => hset(&tv, &header_key, &[("src_fmt", "usda")]).await,
            "seq not a number" => hset(&tv, &header_key, &[("seq", "not-a-number")]).await,
            _ => hset(&tv, &header_key, &[("superseded", "1")]).await,
        }
        let walked = walk(&tv, oid, reference_page_len())
            .await
            .unwrap_or_else(|e| panic!("{shape}: the walk must go on; got {e:?}"));
        assert_eq!(
            walked.hashes.iter().copied().collect::<BTreeSet<_>>(),
            BTreeSet::from([MESH, SAMPLE]),
            "{shape}: the walk releases what the object's remaining keys hold"
        );
    }
}

/// The retirement fence's ghost — a `superseded=1` header with no keys behind
/// it (SDD §7.5.4): the walk answers `Ok` with nothing to release.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reference_walk_of_a_superseded_ghost_header_is_the_empty_set() {
    const GHOST: u64 = 770_042;
    let tv = TestValkey::setup().await;
    hset(
        &tv,
        &tv.key_schema().obj_header(&GHOST.to_string()),
        &[("superseded", "1")],
    )
    .await;

    let walked = walk(&tv, GHOST, reference_page_len())
        .await
        .expect("a ghost header is walked, not refused");
    assert!(walked.hashes.is_empty() && walked.undecodable == 0 && walked.rows == 0);
}

/// `live_edit_snapshot_read_total{consumer, path, outcome}`, absent read as
/// a panic: every `ext_gc` / `unfenced` cell is zero-seeded.
fn snapshot_reads(metrics: &Metrics, consumer: &str, path: &str, outcome: &str) -> f64 {
    sample_value(
        &render(metrics),
        "live_edit_snapshot_read_total",
        &[("consumer", consumer), ("path", path), ("outcome", outcome)],
    )
    .unwrap_or_else(|| {
        panic!("snapshot_read_total{{{consumer},{path},{outcome}}} must be zero-seeded")
    })
}

/// A metrics registry installed on `tv`'s client: the walk records through
/// the client's hooks.
async fn installed_metrics(tv: &TestValkey, label: &str) -> Arc<Metrics> {
    let metrics = Arc::new(crate::common::metrics_scrape_only(label));
    tv.client()
        .install_observability_hooks(Arc::clone(&metrics))
        .await;
    metrics
}

/// D4-T1 / D4-T9: an absent header is `NotFound`, recorded once under
/// `ext_gc` / `unfenced` / `not_found`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reference_walk_of_an_absent_header_is_not_found_and_records_it() {
    let tv = TestValkey::setup().await;
    let metrics = installed_metrics(&tv, "walk-absent").await;

    let err = walk(&tv, 987_654_321, reference_page_len())
        .await
        .expect_err("an absent header is no object to walk");
    assert!(
        matches!(err, ServiceError::NotFound(_)),
        "an absent header is NotFound; got {err:?}"
    );
    assert_eq!(
        snapshot_reads(&metrics, "ext_gc", "unfenced", "not_found"),
        1.0
    );
    assert_eq!(snapshot_reads(&metrics, "ext_gc", "unfenced", "ok"), 0.0);
}

/// D4-T9: one `live_edit_snapshot_read_total{consumer="ext_gc",
/// path="unfenced",outcome="ok"}` per walk, however many pages the walk took
/// (here one row per page), with one duration observation and one size
/// observation of the walk's own tallies (`reply_bytes`: the strings it
/// received; `entries`: the rows it read), and nothing on any snapshot
/// consumer's cell. A `DeleteLayer`, whose release runs the walk once, ticks
/// it exactly once more.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reference_walk_records_one_snapshot_read_per_walk_under_ext_gc_unfenced() {
    let tv = TestValkey::setup().await;
    let metrics = installed_metrics(&tv, "walk-metrics").await;
    let ls = layer_store(&tv);
    let layer = "walk-metrics";
    let session_id = create_session(&tv, layer).await;
    let oid = create(&ls, &session_id, layer, &every_shape_state()).await;

    let walked = walk(&tv, oid, 1).await.expect("the walk must succeed");
    assert!(
        walked.rows >= 4,
        "fixture: four value rows, each on a page of its own"
    );
    let text = render(&metrics);
    let count = |name: &str, labels: &[(&str, &str)]| sample_value(&text, name, labels);
    assert_eq!(
        snapshot_reads(&metrics, "ext_gc", "unfenced", "ok"),
        1.0,
        "one tick per walk, not per page"
    );
    assert_eq!(
        count(
            "live_edit_snapshot_read_duration_seconds_count",
            &[("consumer", "ext_gc"), ("path", "unfenced")]
        ),
        Some(1.0)
    );
    assert_eq!(
        (
            count(
                "live_edit_snapshot_read_reply_bytes_sum",
                &[("consumer", "ext_gc")]
            ),
            count(
                "live_edit_snapshot_read_entries_sum",
                &[("consumer", "ext_gc")]
            ),
        ),
        (Some(walked.bytes as f64), Some(walked.rows as f64)),
        "the size histograms record the walk's own tallies"
    );
    for consumer in ["catchup", "checkpoint", "structural_update"] {
        for path in ["atomic", "unfenced"] {
            if let Some(ticks) = count(
                "live_edit_snapshot_read_total",
                &[("consumer", consumer), ("path", path), ("outcome", "ok")],
            ) {
                assert_eq!(ticks, 0.0, "the walk is no {consumer} read");
            }
        }
    }

    ls.delete_layer(&session_id, layer, mock_ext_store())
        .await
        .expect("DeleteLayer must succeed");
    assert_eq!(
        snapshot_reads(&metrics, "ext_gc", "unfenced", "ok"),
        2.0,
        "a DeleteLayer's release walks the object exactly once"
    );
}

/// The release path counts what the walk could not decode: each undecodable
/// `obj:ext` field ticks `live_edit_ext_ref_undecodable_total{site=
/// "retirement_index_field"}` once (b.yda), and every decodable reference is
/// released.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn retirement_release_counts_each_undecodable_obj_ext_field() {
    let tv = TestValkey::setup().await;
    let oid = every_shape_object(&tv, "walk-undecodable").await;
    hset(
        &tv,
        &tv.key_schema().obj_ext_index(&oid.to_string()),
        &[("not-a-hash", "1"), ("zz", "1")],
    )
    .await;
    let metrics = crate::common::metrics_scrape_only("walk-undecodable");
    let ext = Arc::new(MockExternalValueStore::new());

    schedule_old_object_ext_gc_for_test(
        tv.client(),
        tv.key_schema(),
        &oid.to_string(),
        &*ext,
        Some(&metrics),
    )
    .await
    .expect("the release must succeed");

    assert_eq!(
        sample_value(
            &render(&metrics),
            "live_edit_ext_ref_undecodable_total",
            &[("site", "retirement_index_field")],
        ),
        Some(2.0),
        "one tick per undecodable obj:ext field"
    );
    assert!(
        ext.is_pending(&MESH) && ext.is_pending(&SAMPLE),
        "every decodable reference is released"
    );
}

/// The walk's own failure handling (b.fsz MR-D4 test review): each failed
/// command is classified under its own key family on the snapshot lane —
/// transient → `Aborted`, recorded `transient_error`; permanent → `Internal`,
/// recorded `permanent_error` with its refusal kind ticked; the store's
/// not-serving `-LOADING` / `-BUSY` retryable and counted nowhere; a
/// wrong-typed `obj:flds`, `obj:tss`, value row or `obj:ext` ticks exactly
/// its own `key_family` cell once — and a retirement route whose walk failed
/// still tears the object down, with no release enqueued. The injected rows
/// arm the snapshot-read seam, so they need `test-fault-injection`; the
/// wrong-typed rows plant real foreign keys.
mod walk_failures {
    use super::*;

    use crate::storage::b_atj_tests::assert_only_family_ticked;

    /// The every-shape object under `layer`, a metrics registry installed on
    /// `tv`'s client, and the object's id.
    async fn walked_object(tv: &TestValkey, layer: &str) -> (Arc<Metrics>, u64) {
        let metrics = installed_metrics(tv, layer).await;
        let oid = every_shape_object(tv, layer).await;
        (metrics, oid)
    }

    /// Replace `key` with a string.
    async fn make_foreign(tv: &TestValkey, key: &str) {
        let _: i64 = tv.client().pool.del(key).await.expect("DEL must succeed");
        let _: () = tv
            .client()
            .pool
            .set(key, "foreign", None, None, false)
            .await
            .expect("SET of the foreign value must succeed");
    }

    /// A wrong-typed key under each family the walk reads past its header:
    /// the walk answers `Internal`, records one `permanent_error`, and ticks
    /// exactly that family's `key_family` cell once.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_walk_meeting_a_wrong_typed_key_ticks_exactly_its_family_once() {
        for family in ["fields_index", "time_samples_index", "field", "ext_index"] {
            let tv = TestValkey::setup().await;
            let (metrics, oid) = walked_object(&tv, &format!("walk-wrongtype-{family}")).await;
            let (keys, oid_str) = (tv.key_schema(), oid.to_string());
            let key = match family {
                "fields_index" => keys.obj_fields_index(&oid_str),
                "time_samples_index" => keys.obj_time_samples_index(&oid_str),
                "field" => keys.obj_field(&oid_str, &CUBE.to_string(), "mesh"),
                _ => keys.obj_ext_index(&oid_str),
            };
            make_foreign(&tv, &key).await;

            let err = walk(&tv, oid, reference_page_len())
                .await
                .expect_err("a wrong-typed key fails the walk");
            assert!(
                matches!(err, ServiceError::Internal(_)),
                "{family}: a wrong-typed key is permanent; got {err:?}"
            );
            assert_only_family_ticked(&metrics, family, 1);
            assert_eq!(
                snapshot_reads(&metrics, "ext_gc", "unfenced", "permanent_error"),
                1.0,
                "{family}: one walk, one permanent_error"
            );
        }
    }

    #[cfg(feature = "test-fault-injection")]
    mod injected {
        use super::*;

        use fred::error::{Error as FredError, ErrorKind as FredErrorKind};
        use live_edit::storage::client::STORE_NOT_SERVING_SHAPES;

        use crate::storage::b_atj_tests::assert_no_family_ticked;
        use crate::storage::b_muu_tests::{
            assert_no_refusal_kind_ticked, assert_only_refusal_kind_ticked,
        };
        use live_edit::storage::diff_read::{
            ObjectSnapshotReadFaultInjector, ObjectSnapshotReadFaultTarget, ObjectSnapshotReader,
        };

        /// A seam injector installed on `tv`'s client.
        fn injector(tv: &TestValkey) -> Arc<ObjectSnapshotReadFaultInjector> {
            let injector = Arc::new(ObjectSnapshotReadFaultInjector::new());
            tv.client()
                .install_object_snapshot_read_fault_injector_for_test(Arc::clone(&injector));
            injector
        }

        /// The walk's first command of `family`.
        fn first_of(family: &str) -> ObjectSnapshotReadFaultTarget {
            ObjectSnapshotReadFaultTarget::nth_of_family(
                ObjectSnapshotReader::UnfencedFanout,
                family,
                0,
            )
        }

        /// A transient failure (a timeout on the first `obj:flds` page) is
        /// `Aborted`, recorded once as `transient_error`, and ticks no
        /// refusal kind and no key family.
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn a_walk_failing_transiently_answers_aborted_and_records_transient_error() {
            let tv = TestValkey::setup().await;
            let (metrics, oid) = walked_object(&tv, "walk-transient").await;
            injector(&tv).push_at(
                first_of("fields_index"),
                FredError::new(FredErrorKind::Timeout, "injected walk page timeout"),
            );

            let err = walk(&tv, oid, reference_page_len())
                .await
                .expect_err("the injected timeout fails the walk");
            assert!(
                matches!(err, ServiceError::Aborted(_)),
                "a timeout is transient; got {err:?}"
            );
            assert_eq!(
                snapshot_reads(&metrics, "ext_gc", "unfenced", "transient_error"),
                1.0
            );
            assert_no_refusal_kind_ticked(&metrics);
            assert_no_family_ticked(&metrics);
        }

        /// A permanent failure (a protocol error on the `HSCAN obj:ext` step)
        /// is `Internal`, recorded once as `permanent_error`, and ticks its
        /// refusal kind once and no key family.
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn a_walk_failing_permanently_answers_internal_and_ticks_its_refusal_kind() {
            let tv = TestValkey::setup().await;
            let (metrics, oid) = walked_object(&tv, "walk-permanent").await;
            injector(&tv).push_at(
                first_of("ext_index"),
                FredError::new(FredErrorKind::Protocol, "injected RESP framing violation"),
            );

            let err = walk(&tv, oid, reference_page_len())
                .await
                .expect_err("the injected protocol error fails the walk");
            assert!(
                matches!(err, ServiceError::Internal(_)),
                "a protocol error is permanent; got {err:?}"
            );
            assert_eq!(
                snapshot_reads(&metrics, "ext_gc", "unfenced", "permanent_error"),
                1.0
            );
            assert_only_refusal_kind_ticked(&metrics, "protocol", 1);
            assert_no_family_ticked(&metrics);
        }

        /// The store's not-serving replies (`-LOADING`, `-BUSY`, bare or after
        /// an `EXECABORT`) on a walk's value-row read are retryable: `Aborted`
        /// with the `transient I/O:` prefix, and no refusal kind or key family
        /// ticked.
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn a_walk_refused_loading_or_busy_is_retriable_and_counts_nothing() {
            let tv = TestValkey::setup().await;
            let (metrics, oid) = walked_object(&tv, "walk-not-serving").await;
            let injector = injector(&tv);
            for reply in STORE_NOT_SERVING_SHAPES {
                injector.push_at(
                    first_of("field"),
                    FredError::new(FredErrorKind::Unknown, reply),
                );
                let err = walk(&tv, oid, reference_page_len())
                    .await
                    .expect_err("the injected reply fails the walk");
                let ServiceError::Aborted(msg) = &err else {
                    panic!("{reply:?} is the store not serving: retryable; got {err:?}");
                };
                assert!(
                    msg.starts_with("transient I/O:") && msg.contains(reply),
                    "the retry prefix and the reply; got {msg:?}"
                );
            }
            assert_eq!(injector.remaining(), 0, "fixture: every reply was taken");
            assert_no_refusal_kind_ticked(&metrics);
            assert_no_family_ticked(&metrics);
        }

        /// A `DeleteLayer` whose walk fails — transiently (an injected timeout
        /// on its `HSCAN obj:ext` step), or permanently (a wrong-typed value
        /// row) — answers OK, tears the object down, and enqueues no release:
        /// the failure is WARNed and the teardown does not depend on it.
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn a_delete_whose_walk_fails_still_tears_the_object_down_and_releases_nothing() {
            for permanent in [false, true] {
                let tv = TestValkey::setup().await;
                let layer = "walk-fails-delete";
                let metrics = installed_metrics(&tv, layer).await;
                let ls = layer_store(&tv);
                let session_id = create_session(&tv, layer).await;
                let oid = create(&ls, &session_id, layer, &every_shape_state()).await;
                if permanent {
                    make_foreign(
                        &tv,
                        &tv.key_schema()
                            .obj_field(&oid.to_string(), &CUBE.to_string(), "mesh"),
                    )
                    .await;
                } else {
                    injector(&tv).push_at(
                        first_of("ext_index"),
                        FredError::new(FredErrorKind::Timeout, "injected walk timeout"),
                    );
                }
                let ext = Arc::new(MockExternalValueStore::new());

                ls.delete_layer(&session_id, layer, ext.clone())
                    .await
                    .unwrap_or_else(|e| {
                        panic!("permanent={permanent}: the delete answers OK; got {e:?}")
                    });

                let outcome = if permanent {
                    "permanent_error"
                } else {
                    "transient_error"
                };
                assert_eq!(
                    snapshot_reads(&metrics, "ext_gc", "unfenced", outcome),
                    1.0,
                    "permanent={permanent}: the walk failed"
                );
                assert_eq!(
                    crate::common::scan_object_keys(tv.client(), tv.prefix(), oid)
                        .await
                        .into_iter()
                        .map(|k| k.as_str_lossy().into_owned())
                        .filter(|k| !k.ends_with(":obj:stream") && !k.contains(":obj:extret"))
                        .collect::<Vec<_>>(),
                    Vec::<String>::new(),
                    "permanent={permanent}: the object is torn down"
                );
                assert_eq!(
                    ext.pending_count(),
                    0,
                    "permanent={permanent}: no release was enqueued"
                );
            }
        }
    }
}
