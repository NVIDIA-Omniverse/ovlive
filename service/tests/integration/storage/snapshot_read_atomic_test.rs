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

//! The atomic snapshot read — `storage::diff_read::read_object_snapshot` over
//! `scripts/object_snapshot_read.lua` (b.y3m Epic 2 Task 4; Spec SDD SR-1).
//!
//! One `EVALSHA` returns the whole object, and because a script runs
//! atomically the state it returns is exactly the state at the header `seq` it
//! returns — the cut. What is pinned here:
//!
//! | Area | What it proves |
//! |---|---|
//! | Equality with the reference read | On every state the object's own writes produce — `write_object`, `write_restored_object` over a wiped keyspace, structural and fast-path `UpdateLayer` commits — the atomic read returns exactly what the test-support reference reader `read_object_state_for_test` (plain reads of every key) returns, marked and unmarked (SR-1.2) |
//! | The marked / unmarked children rule | an unmarked object with a partial `obj:chs` still returns every list (the regression); a marked object reads only the lists `obj:chs` names (the restore-over-leftovers shape, the one intended difference) |
//! | RESP2 shape | index members whose hashes are gone decode, and are skipped exactly as the reference read skips them |
//! | The cut | `cut` and the header timestamps are the header's; the tallies count what the read read |
//! | Never tears | under a concurrent writer every read equals the serial state at its cut |
//! | `NOT_FOUND` | a header without a numeric `seq`, for every atomic consumer (the absent header is in `diff_read_test`); a non-numeric object id is `Internal` |
//! | Ceilings | the early-out reads no entry; an object at exactly its item bound reads whole; a well-formed layer under its bound never trips the item tally; inline values past the byte limit and a children list of non-live ids stop mid-read |
//! | Refusals | a wrong-typed key of each of the eleven families ticks exactly its own `key_family` cell; keys the read never touches tick nothing |
//! | Metrics | one `live_edit_snapshot_read_total` tick per read under its consumer and outcome; size histograms only for a read that returned state |
//! | Key set | every key the script touches carries the object's hash tag, it issues only reads, and it never reads `obj:wtok` / `obj:ext` / `obj:stream` / `obj:extret` / `obj:extret:attempts` / `obj:snapfence` / `obj:snapintent` |
//! | Width pinning | the four `SNAPSHOT_*_HASH_WIDTH` constants equal the widest row every writer writes |
//! | Reply shapes | a reply in another format, or malformed, is `Internal` |
//! | Startup self-test | passes on the test store; a store loading, busy, refusing for a slot's state (Valkey's texts and AMR's) or dropping the connection is `Unavailable` (not fatal), decided by the reply and not the deadline; a real refusal is fatal |
//! | Fault seam (`test-fault-injection`) | the in-script cluster refusals are transient and tick no refusal counter; anchoring of that match |
//!
//! The structural and ext-GC consumers never take this path; their unfenced
//! fan-out is `snapshot_read_unfenced_test.rs`, which shares this file's
//! fixture builders. The cluster-gated twin is `snapshot_read_cluster_test.rs`; the instruments
//! ([`MonitorTap`], [`ScriptReplyProxy`], [`SnapshotView`]) are in
//! `snapshot_read_support.rs`; the predicate and key-construction unit tests
//! are in `src/storage/diff_read.rs`; the random-history property is in
//! `tests/proptests.rs`; the zero-seeded series in
//! `observability_metrics_test.rs`.
//!
//! The fault-seam module is `#[cfg(feature = "test-fault-injection")]` and is
//! compiled out — not skipped — without that feature.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use fred::interfaces::{
    HashesInterface, KeysInterface, ListInterface, SetsInterface, SortedSetsInterface,
};
use ordered_float::OrderedFloat;
use tracing::instrument::WithSubscriber;

use live_edit::checkpoint::apply::apply_checkpoint;
use live_edit::config::UpdateLayerBackend;
use live_edit::domain::types::{Field, FieldValue, ObjectState, Section, TimeSample};
use live_edit::error::ServiceError;
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::Metrics;
use live_edit::storage::client::ValkeyClient;
use live_edit::storage::diff_read::{
    read_object_snapshot, run_snapshot_read_self_test, SnapshotCardinalities, SnapshotLimit,
    SnapshotReadConsumer, SnapshotReadPath, SnapshotSelfTestFailure, SnapshotTooLargeAt,
    SNAPSHOT_FIELD_HASH_WIDTH, SNAPSHOT_SECTION_HASH_WIDTH, SNAPSHOT_SELF_TEST_OBJECT_ID,
    SNAPSHOT_TIME_SAMPLE_HASH_WIDTH, SNAPSHOT_TOMBSTONE_HASH_WIDTH,
};
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::keys::KeySchema;
use live_edit::storage::layer::{LayerStore, CHS_COMPLETE_MARKED, HEADER_FIELD_CHS_COMPLETE};
use live_edit::storage::session::SessionStore;
use live_edit::storage::snapshot_fence::{
    self, AcquireVerdict, IntentVerdict, LayerSize, ReleaseVerdict,
};
use live_edit::storage::test_support::relay::RelayedUpdates;
use live_edit::storage::test_support::{read_object_state_for_test, TestValkey};
use oodf_ffi::{
    CommandCreateSection, CommandDeleteSection, CommandMoveSection, OwnedDelta, SetField,
    StructCommand, TimeSample as OodfTimeSample,
};

use crate::common::owned_delta::{
    delta_with_set_field_external, delta_with_set_field_inline, delta_with_time_sample_external,
    delta_with_time_sample_inline,
};
use crate::storage::b_atj_tests::{
    assert_keyspace_type_errors_flat, assert_no_family_ticked, assert_only_family_ticked,
    create_section_delta, delete_section_delta, Fixture, DOOMED_SECTION_ID, FIELD_NAME,
    SAMPLE_TIME, SECTION_ID,
};
use crate::storage::b_muu_tests::assert_no_refusal_kind_ticked;
use crate::storage::checkpoint_apply_test::build_checkpoint_ompf;
use crate::storage::snapshot_read_support::{
    assert_atomic_equals_reference, client_through, client_with_limits, error_reply_script,
    monitor_fence, read_atomic, read_atomic_as, read_err, read_too_large, script_frames_naming,
    test_valkey_addr, view, MonitorTap, MonitoredCommand, ProxyAction, ScriptReplyProxy,
};

const CREATOR: &str = "snapshot-read-user";
const WRITER: &str = "snapshot-read-writer";

// ═════════════════════════════════════════════════════════════════════════════
// Fixture ids and builders
// ═════════════════════════════════════════════════════════════════════════════
//
// Root → [WORLD, SIBLING]; WORLD → [CUBE]; DANGLING_PARENT → [ORPHAN]. The
// object owns three children lists — `obj:ch:0`, `obj:ch:1`, `obj:ch:77` — and
// `obj:chs` = {0, 1, 77}. ORPHAN's parent names no section (SDD §10.4), so its
// list is reachable only through `obj:chs`: both readers must leave it out
// (the parent set is the root plus the live sections, b.evm unchanged).
//
// The ids, builders and fixture helpers below that are `pub(crate)` are shared
// with four other files; check them before changing the tree or its rows:
// - `snapshot_read_unfenced_test` compares the unfenced fan-out with the
//   atomic read on these same objects;
// - `g3_external_hash_tests` seeds its read-side objects from
//   `every_shape_state` and corrupts named rows of it: `CUBE`'s external
//   `"mesh"` field and `1.5` sample, and `WORLD`'s inline `"color"` field;
// - `structural_commit_test` plans structural commits on `every_shape_state`
//   and names its rows: `CUBE`'s external `"mesh"` field and `1.5` sample (the
//   only rows holding their hashes, so deleting `CUBE` releases both),
//   `WORLD`'s inline `"color"` field and `0.5` sample, `WORLD`'s list
//   `[CUBE]`, the root's `[WORLD, SIBLING]`, and no list under `SIBLING`;
// - `structural_guard_parity_test` commits on the same `every_shape_state`
//   object (through `structural_commit_test`'s fixture) and depends on the
//   same rows, plus: `DANGLING_PARENT` having no section row and its list
//   `[ORPHAN]` already named in `obj:chs` at create (the dangling-parent heal
//   test relies on both); `SIBLING` as a root child with no list, and `CUBE` as
//   `WORLD`'s only child, so the `obj:chs` test's `create_cmd` / `delete_cmd`
//   steps under `ROOT`, `WORLD`, `SIBLING` and `CUBE` create, empty and delete
//   the lists it checks with `indexed_parents` / `listed_parents`; and the
//   marker `unmark` strips.

pub(crate) const ROOT: u64 = 0;
pub(crate) const WORLD: u64 = 1;
pub(crate) const CUBE: u64 = 2;
pub(crate) const SIBLING: u64 = 3;
const ORPHAN: u64 = 4;
pub(crate) const DANGLING_PARENT: u64 = 77;

pub(crate) fn mock_ext_store() -> Arc<MockExternalValueStore> {
    Arc::new(MockExternalValueStore::new())
}

/// Lua backend, named explicitly: several tests here are about which commit
/// path wrote a row.
pub(crate) fn layer_store(tv: &TestValkey) -> LayerStore {
    LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_backend(UpdateLayerBackend::Lua)
}

pub(crate) async fn create_session(tv: &TestValkey, name: &str) -> String {
    SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create(name, &format!("/{name}"), CREATOR)
        .await
        .expect("session creation must succeed")
}

fn section(section_id: u64, parent_id: u64, name: &str, section_order: u32) -> Section {
    Section {
        section_id,
        parent_id,
        name: name.to_string(),
        section_type: 1,
        section_order,
        first_version: 1,
        last_version: 1,
    }
}

pub(crate) fn ext_hash(tag: u8) -> [u8; 32] {
    [tag; 32]
}

fn inline_field(section_id: u64, key: &str, value: Vec<u8>) -> Field {
    Field {
        section_id,
        key_name: key.to_string(),
        value: FieldValue::Inline(value),
        set_order: 3,
        authoring_set_order: 103,
        version: 1,
    }
}

fn external_field(section_id: u64, key: &str, tag: u8) -> Field {
    Field {
        section_id,
        key_name: key.to_string(),
        value: FieldValue::External {
            hash: ext_hash(tag),
            size: 4_096,
            ext_value_index: 0,
        },
        set_order: 4,
        authoring_set_order: 104,
        version: 1,
    }
}

fn inline_sample(section_id: u64, time: f64, value: Vec<u8>) -> TimeSample {
    TimeSample {
        section_id,
        time: OrderedFloat(time),
        value: FieldValue::Inline(value),
        version: 1,
    }
}

fn external_sample(section_id: u64, time: f64, tag: u8) -> TimeSample {
    TimeSample {
        section_id,
        time: OrderedFloat(time),
        value: FieldValue::External {
            hash: ext_hash(tag),
            size: 2_048,
            ext_value_index: 0,
        },
        version: 1,
    }
}

fn base_state() -> ObjectState {
    ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usda".to_string(),
        next_set_order: 10,
        ..Default::default()
    }
}

/// The tree in the block comment above, plus one inline and one external row
/// of each value family.
pub(crate) fn every_shape_state() -> ObjectState {
    let mut state = base_state();
    for s in [
        section(WORLD, ROOT, "World", 0),
        section(CUBE, WORLD, "Cube", 0),
        section(SIBLING, ROOT, "Sibling", 1),
        section(ORPHAN, DANGLING_PARENT, "Orphan", 0),
    ] {
        state.sections.insert(s.section_id, s);
    }
    state.children.insert(ROOT, vec![WORLD, SIBLING]);
    state.children.insert(WORLD, vec![CUBE]);
    state.children.insert(DANGLING_PARENT, vec![ORPHAN]);
    for f in [
        inline_field(WORLD, "color", vec![0xFF, 0x00, 0x80]),
        external_field(CUBE, "mesh", 0xAB),
    ] {
        state.fields.insert((f.section_id, f.key_name.clone()), f);
    }
    for t in [
        inline_sample(WORLD, 0.5, vec![0x01, 0x02]),
        external_sample(CUBE, 1.5, 0xCD),
    ] {
        state.time_samples.insert((t.section_id, t.time), t);
    }
    state
}

/// The item bound SR-1.2 step 2 computes for [`every_shape_state`] as
/// created: four live sections, no tombstone, two fields, two time samples,
/// and — marked — the three `obj:chs` members.
pub(crate) fn every_shape_item_bound(marked: bool) -> u64 {
    4 * (1 + SNAPSHOT_SECTION_HASH_WIDTH + 1)
        + 2 * (1 + SNAPSHOT_FIELD_HASH_WIDTH)
        + 2 * (1 + SNAPSHOT_TIME_SAMPLE_HASH_WIDTH)
        + if marked { 3 } else { 0 }
}

pub(crate) async fn create(
    ls: &LayerStore,
    session_id: &str,
    name: &str,
    state: &ObjectState,
) -> u64 {
    ls.create_layer(session_id, name, CREATOR, state, mock_ext_store())
        .await
        .expect("create_layer must succeed")
}

/// Strip the `chs_complete` marker: the header an object materialized before
/// the marker existed carried exactly this shape (Task 2's
/// `unmarked_object_stays_unmarked_through_structural_and_fast_path_commits`
/// uses the same construction).
pub(crate) async fn unmark(tv: &TestValkey, oid: u64) {
    let removed: u64 = tv
        .client()
        .pool
        .hdel(
            tv.key_schema().obj_header(&oid.to_string()),
            HEADER_FIELD_CHS_COMPLETE,
        )
        .await
        .expect("HDEL obj:hdr chs_complete must succeed");
    assert_eq!(
        removed, 1,
        "fixture precondition: the object must have been marked, or it was never the shape this \
         test simulates"
    );
}

async fn marker(tv: &TestValkey, oid: u64) -> Option<String> {
    tv.client()
        .pool
        .hget(
            tv.key_schema().obj_header(&oid.to_string()),
            HEADER_FIELD_CHS_COMPLETE,
        )
        .await
        .expect("HGET obj:hdr chs_complete must succeed")
}

pub(crate) async fn commit(
    ls: &LayerStore,
    session_id: &str,
    layer: &str,
    oid: u64,
    delta: OwnedDelta,
) {
    ls.update_layer_relayed(session_id, layer, oid, Some(delta), WRITER, 7_000)
        .await
        .expect("the fixture commit must land");
}

pub(crate) fn structural(cmds: Vec<StructCommand>) -> OwnedDelta {
    OwnedDelta {
        is_diff: false,
        struct_commands: cmds,
        ..Default::default()
    }
}

pub(crate) fn create_cmd(parent_id: u64, section_id: u64, name: &str) -> StructCommand {
    StructCommand::CreateSection(CommandCreateSection {
        parent_id,
        section_id,
        section_name: name.to_string(),
        section_type: 1,
    })
}

pub(crate) fn move_cmd(
    section_id: u64,
    old_parent_id: u64,
    new_parent_id: u64,
    name: &str,
) -> StructCommand {
    StructCommand::MoveSection(CommandMoveSection {
        old_parent_id,
        new_parent_id,
        section_id,
        new_name: name.to_string(),
    })
}

pub(crate) fn delete_cmd(parent_id: u64, section_id: u64) -> StructCommand {
    StructCommand::DeleteSection(CommandDeleteSection {
        parent_id,
        section_id,
    })
}

/// Every parent with an existing `obj:ch:*` key, from a keyspace SCAN — an
/// oracle independent of `obj:chs` and of both readers.
pub(crate) async fn listed_parents(tv: &TestValkey, oid: u64) -> BTreeSet<u64> {
    let list_prefix = format!("{}{{{oid}}}:obj:ch:", tv.prefix());
    crate::common::scan_object_keys(tv.client(), tv.prefix(), oid)
        .await
        .into_iter()
        .filter_map(|k| {
            k.as_str_lossy()
                .strip_prefix(&list_prefix)
                .and_then(|p| p.parse().ok())
        })
        .collect()
}

pub(crate) async fn indexed_parents(tv: &TestValkey, oid: u64) -> BTreeSet<u64> {
    let members: Vec<String> = tv
        .client()
        .pool
        .smembers(tv.key_schema().obj_children_index(&oid.to_string()))
        .await
        .expect("SMEMBERS obj:chs must succeed");
    members
        .into_iter()
        .map(|m| m.parse().expect("obj:chs members are ids"))
        .collect()
}

// ═════════════════════════════════════════════════════════════════════════════
// 1. Equality with the reference read on states the object's own writes produce
// ═════════════════════════════════════════════════════════════════════════════

async fn equality_on_a_created_object(marked: bool) {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_session(&tv, "sr-eq-create").await;
    let oid = create(&ls, &session_id, "sr-eq-create", &every_shape_state()).await;
    if !marked {
        unmark(&tv, oid).await;
    }
    assert!(
        listed_parents(&tv, oid).await.contains(&DANGLING_PARENT),
        "fixture precondition: the object must own a list under a parent that names no section, \
         or the dangling-parent half of the equality below is vacuous"
    );

    let atomic = assert_atomic_equals_reference(
        tv.client(),
        tv.key_schema(),
        oid,
        if marked {
            "created, marked"
        } else {
            "created, unmarked"
        },
    )
    .await;
    assert!(
        !atomic.state.children.contains_key(&DANGLING_PARENT),
        "a list under a dangling parent stays out of the snapshot on both readers (the parent set \
         is the root plus the live sections; b.evm is unchanged) — on a marked object that list's \
         parent IS in `obj:chs`, so the script must still filter it against the live sections"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_object_snapshot_equals_the_reference_read_on_a_created_marked_object() {
    equality_on_a_created_object(true).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_object_snapshot_equals_the_reference_read_on_a_created_unmarked_object() {
    equality_on_a_created_object(false).await;
}

/// A structural `CreateSection`, a `MoveSection` that empties one list and
/// starts another, a `DeleteSection` that writes a tombstone and rewrites the
/// root list, then fast-path inline and external rows — equality after each.
async fn equality_after_commits(marked: bool) {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let layer = "sr-eq-commits";
    let session_id = create_session(&tv, layer).await;
    let oid = create(&ls, &session_id, layer, &every_shape_state()).await;
    if !marked {
        unmark(&tv, oid).await;
    }
    const NEW: u64 = 5;

    let steps: Vec<(&str, OwnedDelta)> = vec![
        (
            "CreateSection under a leaf",
            structural(vec![create_cmd(CUBE, NEW, "New")]),
        ),
        (
            "MoveSection of WORLD's only child to SIBLING",
            structural(vec![move_cmd(CUBE, WORLD, SIBLING, "Cube")]),
        ),
        (
            "DeleteSection of the now childless WORLD",
            structural(vec![delete_cmd(ROOT, WORLD)]),
        ),
        (
            "fast-path inline field",
            delta_with_set_field_inline(SIBLING, "label", vec![0x2A], 5),
        ),
        (
            "fast-path external field",
            delta_with_set_field_external(CUBE, "mesh", ext_hash(0xEE).to_vec(), 9_000, 1, 6),
        ),
        (
            "fast-path inline time sample",
            delta_with_time_sample_inline(NEW, 3.25, vec![0x07]),
        ),
        (
            "fast-path external time sample",
            delta_with_time_sample_external(NEW, 4.5, ext_hash(0x77).to_vec(), 1_000, 1),
        ),
    ];
    for (what, delta) in steps {
        commit(&ls, &session_id, layer, oid, delta).await;
        assert_atomic_equals_reference(
            tv.client(),
            tv.key_schema(),
            oid,
            &format!(
                "after {what} ({})",
                if marked { "marked" } else { "unmarked" }
            ),
        )
        .await;
    }
    let atomic = read_atomic(tv.client(), tv.key_schema(), oid).await;
    assert!(
        atomic
            .state
            .tombstones
            .iter()
            .any(|(sid, _, _)| *sid == WORLD),
        "fixture precondition: the DeleteSection must have written WORLD's tombstone, or the \
         tombstone family was never compared"
    );
    assert_eq!(
        marker(&tv, oid).await.as_deref(),
        marked.then_some(CHS_COMPLETE_MARKED),
        "the commits must leave the marker as the fixture set it, or this test compared the \
         other read path"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_object_snapshot_equals_the_reference_read_after_structural_and_fast_path_commits_marked(
) {
    equality_after_commits(true).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_object_snapshot_equals_the_reference_read_after_structural_and_fast_path_commits_unmarked(
) {
    equality_after_commits(false).await;
}

/// `write_restored_object` over a WIPED keyspace — the restore fixture SR-1.2
/// sanctions for equality. (A restore over leftover keys whose index is gone
/// is the one shape where the readers differ by design; see
/// [`read_object_snapshot_of_a_marked_object_reads_only_the_lists_its_children_index_names`].)
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_object_snapshot_equals_the_reference_read_on_an_object_restored_over_a_wiped_keyspace(
) {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let layer = "sr-eq-restore";
    let session_id = create_session(&tv, layer).await;
    let oid = create(&ls, &session_id, layer, &every_shape_state()).await;
    commit(
        &ls,
        &session_id,
        layer,
        oid,
        structural(vec![create_cmd(CUBE, 5, "New")]),
    )
    .await;
    let (ompf, snap_seq, meta) = build_checkpoint_ompf(&tv, oid).await;
    crate::common::wipe_object_keys(&tv, oid).await;

    apply_checkpoint(
        &ls,
        oid,
        ompf,
        snap_seq,
        &meta,
        None,
        &crate::common::test_config(),
    )
    .await
    .expect("apply_checkpoint must succeed on a wiped obj keyspace");

    assert_eq!(
        marker(&tv, oid).await.as_deref(),
        Some(CHS_COMPLETE_MARKED),
        "fixture precondition: the restore must have marked the object, or the marked read path \
         was never exercised"
    );
    assert_atomic_equals_reference(
        tv.client(),
        tv.key_schema(),
        oid,
        "restored over a wiped keyspace",
    )
    .await;
}

/// **The unmarked-partial-index regression** (Spec SDD SR-1.7). An object
/// materialized before the children index existed carries every list but no
/// `obj:chs`; one later `SetChildren` indexes only the list it wrote. Such an
/// object is unmarked, so the read must probe the root plus every section and
/// return every list. A script that trusted `obj:chs` here would return only
/// the one indexed list and silently drop the rest of the tree from every
/// catch-up and checkpoint of the layer.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_object_snapshot_of_an_unmarked_object_with_a_partial_children_index_returns_every_children_list(
) {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let layer = "sr-unmarked-partial";
    let session_id = create_session(&tv, layer).await;
    let oid = create(&ls, &session_id, layer, &every_shape_state()).await;

    // Lists written without the index and without the marker: the pre-b.jdr
    // object. Then one `SetChildren` (CUBE gains its first child).
    unmark(&tv, oid).await;
    let _: i64 = tv
        .client()
        .pool
        .del(tv.key_schema().obj_children_index(&oid.to_string()))
        .await
        .expect("DEL obj:chs must succeed");
    const NEW: u64 = 5;
    commit(
        &ls,
        &session_id,
        layer,
        oid,
        structural(vec![create_cmd(CUBE, NEW, "New")]),
    )
    .await;

    assert_eq!(
        indexed_parents(&tv, oid).await,
        BTreeSet::from([CUBE]),
        "fixture precondition: `obj:chs` must name only the list the one `SetChildren` wrote — a \
         partial index"
    );
    assert_eq!(
        marker(&tv, oid).await,
        None,
        "fixture precondition: the object must be unmarked"
    );
    let listed = listed_parents(&tv, oid).await;
    for parent in [ROOT, WORLD, CUBE] {
        assert!(
            listed.contains(&parent),
            "fixture precondition: `obj:ch:{parent}` must exist; listed: {listed:?}"
        );
    }

    let atomic = assert_atomic_equals_reference(
        tv.client(),
        tv.key_schema(),
        oid,
        "unmarked object with a partial obj:chs",
    )
    .await;
    assert_eq!(
        atomic.state.children.keys().copied().collect::<BTreeSet<_>>(),
        BTreeSet::from([ROOT, WORLD, CUBE]),
        "an unmarked object's `obj:chs` may be partial, so the read must probe the root plus every \
         live section and return every list; returning only the indexed CUBE list drops the root's \
         and WORLD's children from every catch-up and checkpoint"
    );
}

/// The one shape where the readers differ **by design** (Spec SDD SR-1.2, the
/// restore-over-leftovers exception): a list under a live section that a
/// marked object's `obj:chs` does not name. A DR restore over leftover keys
/// whose index is gone leaves exactly this — a stray list beside a marked
/// header — and the marked read leaves it out, matching the restored state;
/// the reference reader (`read_object_state_for_test`) returns it.
///
/// Staged directly, since the restore's own route to it is a crashed cleanup:
/// the key shape is what matters. This is also the discriminating twin of the
/// regression above — a script that ignored the marker and probed every live
/// section on a marked object would return the stray list and fail here.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_object_snapshot_of_a_marked_object_reads_only_the_lists_its_children_index_names() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let layer = "sr-marked-unindexed";
    let session_id = create_session(&tv, layer).await;
    let oid = create(&ls, &session_id, layer, &every_shape_state()).await;
    // SIBLING is live and childless; give it an unindexed list.
    let _: i64 = tv
        .client()
        .pool
        .rpush(
            tv.key_schema()
                .obj_children(&oid.to_string(), &SIBLING.to_string()),
            vec![CUBE.to_string()],
        )
        .await
        .expect("RPUSH of the stray list must succeed");
    assert!(
        !indexed_parents(&tv, oid).await.contains(&SIBLING),
        "fixture precondition: the stray list must be unindexed"
    );

    let reference = read_object_state_for_test(tv.client(), tv.key_schema(), &oid.to_string())
        .await
        .unwrap_or_else(|e| panic!("the reference read must succeed; got {e:?}"));
    let atomic = read_atomic(tv.client(), tv.key_schema(), oid).await;

    assert_eq!(
        reference.children.get(&SIBLING),
        Some(&vec![CUBE]),
        "fixture precondition: the reference read probes every live section, so it must return the \
         stray list"
    );
    let mut expected = view(&reference);
    expected.children.remove(&SIBLING);
    assert_eq!(
        view(&atomic.state),
        expected,
        "on a marked object the read trusts `obj:chs` (every list write maintains it in the same \
         transaction), so a list it does not name is not part of the object: the atomic read must \
         equal the reference read minus exactly that list"
    );
}

/// RESP2 shape (Spec SDD SR-1.3): index members whose hashes are gone come
/// back as `{id, {}}` — never a nil inside an array, which a RESP2 connection
/// would truncate the reply at — and decode to "skip", exactly as the reference
/// read's empty `HGETALL`s do. Every family at once, marked and unmarked (the
/// unmarked read also probes the ghost section's absent children list).
///
/// The skip is not silent on the atomic path for the section index (Spec SDD
/// SR-4.1): under one atomic cut an `obj:secs` member without its hash is a
/// corrupt index, not a race, so the read names it on
/// `corrupt_section_index_members` for the consumer to judge (the checkpoint
/// assembler refuses the layer). The unfenced fan-out leaves that list empty:
/// there a missing hash can be a commit landing between rounds, and its
/// consumers do not consult it. The ghost tombstone, field and time sample are
/// not section-index members and are not reported.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_object_snapshot_decodes_index_members_whose_hashes_are_absent() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let layer = "sr-resp2-absent";
    let session_id = create_session(&tv, layer).await;
    let oid = create(&ls, &session_id, layer, &every_shape_state()).await;
    let oid_str = oid.to_string();
    let keys = tv.key_schema();
    let pool = &tv.client().pool;
    const GHOST: u64 = 99;
    let _: i64 = pool
        .sadd(keys.obj_sections_index(&oid_str), GHOST.to_string())
        .await
        .unwrap();
    let _: i64 = pool
        .sadd(keys.obj_deleted_sections_index(&oid_str), GHOST.to_string())
        .await
        .unwrap();
    let _: i64 = pool
        .zadd(
            keys.obj_fields_index(&oid_str),
            None,
            None,
            false,
            false,
            (1.0, KeySchema::field_member(&WORLD.to_string(), "ghost")),
        )
        .await
        .unwrap();
    let _: i64 = pool
        .zadd(
            keys.obj_time_samples_index(&oid_str),
            None,
            None,
            false,
            false,
            (1.0, KeySchema::time_sample_member(&WORLD.to_string(), 9.0)),
        )
        .await
        .unwrap();

    for marked in [true, false] {
        if !marked {
            unmark(&tv, oid).await;
        }
        let atomic = assert_atomic_equals_reference(
            tv.client(),
            keys,
            oid,
            &format!(
                "absent hashes behind every index ({})",
                if marked { "marked" } else { "unmarked" }
            ),
        )
        .await;
        assert!(
            !atomic.state.sections.contains_key(&GHOST)
                && !atomic
                    .state
                    .tombstones
                    .iter()
                    .any(|(sid, _, _)| *sid == GHOST)
                && atomic.state.fields.len() == 2
                && atomic.state.time_samples.len() == 2,
            "an index member whose hash is gone must decode as absent, not as a zeroed entry"
        );
        assert_eq!(
            atomic.corrupt_section_index_members,
            vec![GHOST.to_string()],
            "the atomic read must name the `obj:secs` member whose hash is gone — and only it — \
             as a corrupt index ({}); the checkpoint assembler's refusal depends on it",
            if marked { "marked" } else { "unmarked" }
        );

        // The structural WATCH path's slim fan-out (the unfenced fan-out's one
        // reader) and the cold start's slim atomic read: the absent hashes
        // are absent rows on the compact form too, and the ghost section is
        // skipped.
        for consumer in [
            live_edit::storage::diff_read::SlimSnapshotReadConsumer::StructuralUpdate,
            live_edit::storage::diff_read::SlimSnapshotReadConsumer::StructuralCache,
        ] {
            let slim = match live_edit::storage::diff_read::read_object_snapshot_slim(
                tv.client(),
                keys,
                &oid_str,
                consumer,
            )
            .await
            {
                Ok(live_edit::storage::diff_read::SnapshotReadOf::Snapshot(snapshot)) => snapshot,
                Ok(live_edit::storage::diff_read::SnapshotReadOf::TooLarge(t)) => {
                    panic!("{consumer:?}: the fixture is within every ceiling; got {t:?}")
                }
                Err(e) => panic!("{consumer:?}: the slim read must succeed; got {e:?}"),
            };
            assert_eq!(
                slim.state,
                live_edit::storage::test_support::compact_snapshot(&atomic.state),
                "{consumer:?}: the slim read is the compact projection of the full atomic read, \
                 absent rows and the ghost section left out alike"
            );
            if matches!(
                consumer,
                live_edit::storage::diff_read::SlimSnapshotReadConsumer::StructuralUpdate
            ) {
                assert_eq!(slim.path, SnapshotReadPath::Unfenced);
                assert!(
                    slim.corrupt_section_index_members.is_empty(),
                    "the unfenced fan-out must not report a corrupt index: a hash missing \
                     between its rounds can be a concurrent delete, not corruption; got {:?}",
                    slim.corrupt_section_index_members
                );
            }
        }
    }
}

/// An `obj:secs` member that is not a decimal section id names no section even
/// when a hash exists under it: the decoder reads it as section `0` on every
/// path (so the state matches the reference read), and the atomic read reports it on
/// `corrupt_section_index_members` (Spec SDD SR-4.1). The hash is planted so the
/// member is reported for its id alone, not for a missing hash.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_object_snapshot_reports_a_non_decimal_section_index_member_as_corrupt() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let layer = "sr-non-decimal-member";
    let session_id = create_session(&tv, layer).await;
    let oid = create(&ls, &session_id, layer, &every_shape_state()).await;
    let oid_str = oid.to_string();
    let keys = tv.key_schema();
    let pool = &tv.client().pool;
    const NON_DECIMAL: &str = "not-a-section-id";
    let _: i64 = pool
        .sadd(keys.obj_sections_index(&oid_str), NON_DECIMAL.to_string())
        .await
        .expect("SADD of the non-decimal member must succeed");
    let _: i64 = pool
        .hset(
            keys.obj_section(&oid_str, NON_DECIMAL),
            vec![
                ("parent_id", "0".to_string()),
                ("name", "non-decimal".to_string()),
                ("type", "1".to_string()),
                ("order", "0".to_string()),
                ("first_ver", "1".to_string()),
                ("last_ver", "1".to_string()),
            ],
        )
        .await
        .expect("HSET of the member's section hash must succeed");

    let atomic = read_atomic_as(tv.client(), keys, oid, SnapshotReadConsumer::Checkpoint).await;
    assert_eq!(
        atomic.corrupt_section_index_members,
        vec![NON_DECIMAL.to_string()],
        "a non-decimal `obj:secs` member is a corrupt index even with a hash behind it"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// 2. The cut and the tallies
// ═════════════════════════════════════════════════════════════════════════════

/// `cut` is the header `seq` the read returned with, and the header timestamps
/// ride with it (SR-1.3: the whole header is read inside the script).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_object_snapshot_cut_and_timestamps_are_the_headers_at_the_cut() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let layer = "sr-cut";
    let session_id = create_session(&tv, layer).await;
    let oid = create(&ls, &session_id, layer, &every_shape_state()).await;
    const CLIENT_TS: u64 = 424_242;
    ls.update_layer_relayed(
        &session_id,
        layer,
        oid,
        Some(delta_with_set_field_inline(WORLD, "color", vec![9], 1)),
        WRITER,
        CLIENT_TS,
    )
    .await
    .expect("the commit must land");
    let header: HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(tv.key_schema().obj_header(&oid.to_string()))
        .await
        .expect("HGETALL obj:hdr must succeed");
    let column = |name: &str| -> u64 {
        header
            .get(name)
            .unwrap_or_else(|| panic!("the header must carry {name}"))
            .parse()
            .unwrap_or_else(|e| panic!("{name} must be numeric: {e}"))
    };

    let atomic = read_atomic(tv.client(), tv.key_schema(), oid).await;

    assert_eq!(
        (atomic.cut, atomic.state.sequence_num),
        (column("seq"), column("seq")),
        "the cut is the header `seq` read inside the script, and the state is the state at it"
    );
    assert_eq!(
        (atomic.last_server_ts, atomic.last_client_ts),
        (column("last_server_ts"), CLIENT_TS),
        "the header timestamps must be the ones at the cut"
    );
    assert_eq!(atomic.path, SnapshotReadPath::Atomic);
}

/// The three tallies, recomputed from the store by the SR-1.2 definition:
/// items are index members, hash fields, `obj:chs` members (marked) and list
/// elements; bytes are the lengths of every one of those strings; entries are
/// one per hash and one per children list read. The limits are checked
/// against these numbers, so a tally that drifted from its definition moves
/// the ceiling.
async fn expected_tallies(tv: &TestValkey, oid: u64, marked: bool) -> (u64, u64, u64) {
    let keys = tv.key_schema();
    let o = oid.to_string();
    let pool = &tv.client().pool;
    let (mut items, mut bytes, mut entries) = (0u64, 0u64, 0u64);
    fn list(l: &[String], items: &mut u64, bytes: &mut u64) {
        *items += l.len() as u64;
        *bytes += l.iter().map(|s| s.len() as u64).sum::<u64>();
    }
    let secs: Vec<String> = pool.smembers(keys.obj_sections_index(&o)).await.unwrap();
    let dels: Vec<String> = pool
        .smembers(keys.obj_deleted_sections_index(&o))
        .await
        .unwrap();
    let flds: Vec<String> = pool
        .zrange(keys.obj_fields_index(&o), 0, -1, None, false, None, false)
        .await
        .unwrap();
    let tss: Vec<String> = pool
        .zrange(
            keys.obj_time_samples_index(&o),
            0,
            -1,
            None,
            false,
            None,
            false,
        )
        .await
        .unwrap();
    for index in [&secs, &dels, &flds, &tss] {
        list(index, &mut items, &mut bytes);
    }
    let mut hash_keys: Vec<String> = Vec::new();
    hash_keys.extend(secs.iter().map(|s| keys.obj_section(&o, s)));
    hash_keys.extend(dels.iter().map(|s| keys.obj_deleted_section(&o, s)));
    hash_keys.extend(flds.iter().map(|m| keys.obj_field_from_member(&o, m)));
    hash_keys.extend(tss.iter().map(|m| keys.obj_ts_from_member(&o, m)));
    for key in hash_keys {
        let h: HashMap<String, String> = pool.hgetall(&key).await.unwrap();
        items += h.len() as u64;
        bytes += h
            .iter()
            .map(|(k, v)| (k.len() + v.len()) as u64)
            .sum::<u64>();
        entries += 1;
    }
    let parents: Vec<String> = if marked {
        let chs: Vec<String> = pool.smembers(keys.obj_children_index(&o)).await.unwrap();
        list(&chs, &mut items, &mut bytes);
        chs.into_iter()
            .filter(|p| p == "0" || secs.contains(p))
            .collect()
    } else {
        secs.iter()
            .cloned()
            .chain(std::iter::once("0".to_string()))
            .collect()
    };
    for p in parents {
        let l: Vec<String> = pool.lrange(keys.obj_children(&o, &p), 0, -1).await.unwrap();
        list(&l, &mut items, &mut bytes);
        entries += 1;
    }
    (items, bytes, entries)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_object_snapshot_tallies_count_every_item_byte_and_entry_it_read() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let layer = "sr-tallies";
    let session_id = create_session(&tv, layer).await;
    let oid = create(&ls, &session_id, layer, &every_shape_state()).await;
    commit(
        &ls,
        &session_id,
        layer,
        oid,
        structural(vec![create_cmd(ROOT, 9, "Doomed")]),
    )
    .await;
    commit(
        &ls,
        &session_id,
        layer,
        oid,
        structural(vec![delete_cmd(ROOT, 9)]),
    )
    .await;

    for marked in [true, false] {
        if !marked {
            unmark(&tv, oid).await;
        }
        let atomic = read_atomic(tv.client(), tv.key_schema(), oid).await;
        assert_eq!(
            (atomic.item_count, atomic.reply_bytes, atomic.entry_count),
            expected_tallies(&tv, oid, marked).await,
            "(items, bytes, entries) must be the SR-1.2 tallies of what the read read ({}) — the \
             item and byte limits are checked against exactly these",
            if marked {
                "marked: obj:chs members counted, only indexed live lists read"
            } else {
                "unmarked: the root plus every section's list read"
            }
        );
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// 3. NOT_FOUND
// ═════════════════════════════════════════════════════════════════════════════

// The absent-header case is `diff_read_test::read_object_snapshot_nonexistent_object_returns_not_found`.
// The structural and ext-GC consumers, which never take the atomic path, are
// `snapshot_read_unfenced_test`'s — including the proof that neither ever runs
// this script.

/// Object ids are decimal `u64`s; any other id is a caller bug, refused as
/// `Internal` before the script runs.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_object_snapshot_of_a_non_numeric_object_id_is_internal() {
    let tv = TestValkey::setup().await;
    let result = read_object_snapshot(
        tv.client(),
        tv.key_schema(),
        "not-an-id",
        SnapshotReadConsumer::Catchup,
    )
    .await;
    assert!(
        matches!(&result, Err(ServiceError::Internal(m)) if m.contains("non-numeric object_id")),
        "a non-numeric object id must be refused as Internal; got {:?}",
        result.err()
    );
}

/// A header without a numeric `seq` — absent or garbage — is `NotFound` for
/// every consumer that takes the atomic path (Spec SDD SR-1.5,
/// `read_object_header`'s rule). The `ext_gc` empty-state rule belongs to the
/// unfenced path (`snapshot_read_unfenced_test`).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_object_snapshot_of_a_header_without_a_numeric_seq_is_not_found_for_every_atomic_consumer(
) {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let layer = "sr-partial-hdr";
    let session_id = create_session(&tv, layer).await;
    let oid = create(&ls, &session_id, layer, &every_shape_state()).await;
    let header_key = tv.key_schema().obj_header(&oid.to_string());

    let removed: i64 = tv.client().pool.hdel(&header_key, "seq").await.unwrap();
    assert_eq!(
        removed, 1,
        "fixture precondition: the seq column must have been present"
    );
    for seq in [None, Some("not-a-number")] {
        if let Some(raw) = seq {
            let _: () = tv
                .client()
                .pool
                .hset(&header_key, [("seq", raw)])
                .await
                .unwrap();
        }
        for consumer in [
            SnapshotReadConsumer::Catchup,
            SnapshotReadConsumer::CatchupLegacyPerCommit,
            SnapshotReadConsumer::Checkpoint,
        ] {
            let result =
                read_object_snapshot(tv.client(), tv.key_schema(), &oid.to_string(), consumer)
                    .await;
            assert!(
                matches!(result, Err(ServiceError::NotFound(_))),
                "consumer {consumer:?}, seq {seq:?}: a header without a numeric seq is not a live \
                 object for this consumer; got {:?}",
                result.err()
            );
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// 4. Ceilings
// ═════════════════════════════════════════════════════════════════════════════

/// The early-out (SR-1.2 step 2): with the item bound one over the limit the
/// script answers from the header and the cardinalities alone. The `MONITOR`
/// record pins "reads no entry" as the exact command sequence. A marked
/// object's bound counts its `obj:chs` members and the answer carries that
/// cardinality; an unmarked object's bound leaves `obj:chs` out, the script
/// never touches the key, and the answer carries no children-index
/// cardinality.
async fn early_out(marked: bool) {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let layer = "sr-early-out";
    let session_id = create_session(&tv, layer).await;
    let oid = create(&ls, &session_id, layer, &every_shape_state()).await;
    if !marked {
        unmark(&tv, oid).await;
    }
    let bound = every_shape_item_bound(marked);
    let client = client_with_limits(bound - 1, u64::MAX / 2).await;
    let mut tap = MonitorTap::start(&test_valkey_addr()).await;

    let too_large = read_too_large(&client, tv.key_schema(), oid).await;
    let fence = monitor_fence(tv.prefix());
    let _: i64 = tv.client().pool.exists(&fence).await.unwrap();
    let script = script_commands(
        tap.drain_until(&fence).await,
        &tv.key_schema().obj_header(&oid.to_string()),
    );

    assert_eq!(
        too_large.at,
        SnapshotTooLargeAt::EarlyOut,
        "a bound over the limit must stop at step 2"
    );
    assert_eq!(
        (too_large.item_bound, too_large.cardinalities),
        (
            bound,
            SnapshotCardinalities {
                sections: 4,
                tombstones: 0,
                fields: 2,
                time_samples: 2,
                children_index: marked.then_some(3),
            }
        ),
        "every TOO_LARGE carries step 2's cardinalities and item bound — Epic 3/5 size the layer \
         from them (SR-11.1)"
    );
    let names: Vec<String> = script.iter().map(|c| c.name()).collect();
    let expected: &[&str] = if marked {
        &["HGETALL", "SCARD", "SCARD", "ZCARD", "ZCARD", "SCARD"]
    } else {
        &["HGETALL", "SCARD", "SCARD", "ZCARD", "ZCARD"]
    };
    assert_eq!(
        names, expected,
        "the early-out must cost one header read and the cardinalities, nothing more — an entry \
         read here is shard time the ceiling exists to avoid; script commands: {script:?}"
    );
    let chs_key = tv.key_schema().obj_children_index(&oid.to_string());
    assert_eq!(
        script.iter().any(|c| c.args.get(1) == Some(&chs_key)),
        marked,
        "only a marked object's early-out reads `obj:chs`; script commands: {script:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_object_snapshot_over_its_item_bound_answers_too_large_before_reading_any_entry() {
    early_out(true).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_object_snapshot_of_an_unmarked_object_over_its_item_bound_early_outs_without_obj_chs()
{
    early_out(false).await;
}

/// The commands `object_snapshot_read.lua` issued for the read whose `EVALSHA`
/// named `header_key`: the `lua` lines straight after that `EVALSHA` line. A
/// script runs atomically, so nothing interleaves with them.
fn script_commands(seen: Vec<MonitoredCommand>, header_key: &str) -> Vec<MonitoredCommand> {
    let start = seen
        .iter()
        .rposition(|c| {
            c.name() == "EVALSHA" && c.args.get(3).map(String::as_str) == Some(header_key)
        })
        .unwrap_or_else(|| panic!("the MONITOR record holds no EVALSHA naming {header_key}"));
    seen.into_iter()
        .skip(start + 1)
        .take_while(MonitoredCommand::issued_by_script)
        .collect()
}

/// An object whose every row has its family's full width and whose every live
/// section sits in a live parent's list: the step-2 bound is exact, so the
/// running item tally lands on it. Root → [A], A → [B], a tombstone, external
/// rows only.
async fn create_exact_bound_object(tv: &TestValkey, ls: &LayerStore, layer: &str) -> (u64, u64) {
    const A: u64 = 1;
    const B: u64 = 2;
    const DOOMED: u64 = 3;
    let session_id = create_session(tv, layer).await;
    let mut state = base_state();
    state.sections.insert(A, section(A, ROOT, "A", 0));
    state.sections.insert(B, section(B, A, "B", 0));
    state.children.insert(ROOT, vec![A]);
    state.children.insert(A, vec![B]);
    for f in [external_field(A, "fa", 1), external_field(B, "fb", 2)] {
        state.fields.insert((f.section_id, f.key_name.clone()), f);
    }
    let t = external_sample(B, 1.0, 3);
    state.time_samples.insert((t.section_id, t.time), t);
    let oid = create(ls, &session_id, layer, &state).await;
    commit(
        ls,
        &session_id,
        layer,
        oid,
        structural(vec![create_cmd(ROOT, DOOMED, "Doomed")]),
    )
    .await;
    commit(
        ls,
        &session_id,
        layer,
        oid,
        structural(vec![delete_cmd(ROOT, DOOMED)]),
    )
    .await;
    // Two sections, one tombstone, two fields, one sample, `obj:chs` = {0, A}.
    let bound = 2 * (1 + SNAPSHOT_SECTION_HASH_WIDTH + 1)
        + (1 + SNAPSHOT_TOMBSTONE_HASH_WIDTH)
        + 2 * (1 + SNAPSHOT_FIELD_HASH_WIDTH)
        + (1 + SNAPSHOT_TIME_SAMPLE_HASH_WIDTH)
        + 2;
    (oid, bound)
}

/// The boundary, both sides: at an item limit equal to the bound the object is
/// read whole and its tally lands exactly on the bound (the tally check is
/// strict, `items > limit`); one below, it early-outs. Equality of tally and
/// bound on this fixture is the independent check that the bound counts what
/// the read reads.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_object_snapshot_at_exactly_its_item_bound_reads_whole_and_one_below_early_outs() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let (oid, bound) = create_exact_bound_object(&tv, &ls, "sr-exact-bound").await;

    let at = read_atomic(
        &client_with_limits(bound, u64::MAX / 2).await,
        tv.key_schema(),
        oid,
    )
    .await;
    assert_eq!(
        at.item_count, bound,
        "on an object whose rows all have their family's full width, the running item tally must \
         land exactly on step 2's bound — a gap means the bound and the tally count different \
         things"
    );
    let below = read_too_large(
        &client_with_limits(bound - 1, u64::MAX / 2).await,
        tv.key_schema(),
        oid,
    )
    .await;
    assert_eq!(
        (below.at, below.item_bound),
        (SnapshotTooLargeAt::EarlyOut, bound),
        "one item under the bound must early-out"
    );
}

/// A well-formed layer whose bound is at or under the item limit never trips
/// the item tally mid-read (SR-1.2): inline rows use fewer columns than the
/// widths count, and a dangling-parent section adds a bound slot no list
/// fills, so the tally stays under the bound and the read completes at
/// limit = bound.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_object_snapshot_of_a_well_formed_layer_under_its_bound_never_trips_the_item_tally() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let layer = "sr-under-bound";
    let session_id = create_session(&tv, layer).await;
    let oid = create(&ls, &session_id, layer, &every_shape_state()).await;
    let bound = every_shape_item_bound(true);

    let atomic = read_atomic(
        &client_with_limits(bound, u64::MAX / 2).await,
        tv.key_schema(),
        oid,
    )
    .await;
    assert!(
        atomic.item_count <= bound,
        "the tally ({}) must stay within step 2's bound ({bound}) on a well-formed object",
        atomic.item_count
    );
}

/// Few entries, large inline values: the byte tally crosses the byte limit
/// mid-read (SR-1.2 running cost check), with the overshoot bounded by the
/// one entry that crossed it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_object_snapshot_with_inline_values_past_the_byte_limit_answers_too_large_mid_read() {
    const VALUE_LEN: usize = 3_000;
    const BYTE_LIMIT: u64 = 5_000;
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let layer = "sr-bytes";
    let session_id = create_session(&tv, layer).await;
    let mut state = base_state();
    state
        .sections
        .insert(WORLD, section(WORLD, ROOT, "World", 0));
    state.children.insert(ROOT, vec![WORLD]);
    for key in ["a", "b", "c"] {
        let f = inline_field(WORLD, key, vec![0x5A; VALUE_LEN]);
        state.fields.insert((WORLD, key.to_string()), f);
    }
    let oid = create(&ls, &session_id, layer, &state).await;
    let largest_entry = largest_field_hash_bytes(&tv, oid).await;

    let too_large = read_too_large(
        &client_with_limits(10_000, BYTE_LIMIT).await,
        tv.key_schema(),
        oid,
    )
    .await;

    let SnapshotTooLargeAt::MidRead {
        limit,
        item_tally,
        byte_tally,
    } = too_large.at
    else {
        panic!(
            "inline values past the byte limit must stop mid-read; got {:?}",
            too_large.at
        );
    };
    assert_eq!(limit, SnapshotLimit::Bytes);
    assert!(
        byte_tally > BYTE_LIMIT && byte_tally <= BYTE_LIMIT + largest_entry,
        "the byte tally ({byte_tally}) must have crossed the limit ({BYTE_LIMIT}) by at most the one \
         entry that crossed it ({largest_entry} B) — the overshoot bound SR-1.2 states"
    );
    assert!(
        item_tally > 0,
        "a mid-read answer carries the tallies so far"
    );
    assert_eq!(too_large.cardinalities.fields, 3);
}

async fn largest_field_hash_bytes(tv: &TestValkey, oid: u64) -> u64 {
    let o = oid.to_string();
    let members: Vec<String> = tv
        .client()
        .pool
        .zrange(
            tv.key_schema().obj_fields_index(&o),
            0,
            -1,
            None,
            false,
            None,
            false,
        )
        .await
        .unwrap();
    let mut largest = 0;
    for m in members {
        let h: HashMap<String, String> = tv
            .client()
            .pool
            .hgetall(tv.key_schema().obj_field_from_member(&o, &m))
            .await
            .unwrap();
        largest = largest.max(h.iter().map(|(k, v)| (k.len() + v.len()) as u64).sum());
    }
    largest
}

/// The anomaly safety net (SR-1.2): a children list holding ids that are not
/// live sections passes the early-out (the bound counts at most one list
/// element per live section) and trips the item tally mid-read. Counted as
/// `outcome="too_large"` and WARN-logged naming the object and the tallies —
/// the line is the deliverable here, the one evidence that a writer broke the
/// width or tree invariant, so it is pinned.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_object_snapshot_of_a_children_list_of_non_live_ids_trips_the_item_tally_and_warns() {
    const ITEM_LIMIT: u64 = 50;
    const BOGUS: u64 = 200;
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let layer = "sr-anomaly";
    let session_id = create_session(&tv, layer).await;
    let mut state = base_state();
    state
        .sections
        .insert(WORLD, section(WORLD, ROOT, "World", 0));
    state.children.insert(ROOT, vec![WORLD]);
    let oid = create(&ls, &session_id, layer, &state).await;
    let bogus: Vec<String> = (1_000..1_000 + BOGUS).map(|i| i.to_string()).collect();
    let _: i64 = tv
        .client()
        .pool
        .rpush(tv.key_schema().obj_children(&oid.to_string(), "0"), bogus)
        .await
        .unwrap();
    let bound = (1 + SNAPSHOT_SECTION_HASH_WIDTH + 1) + 1;
    assert!(
        bound <= ITEM_LIMIT,
        "fixture precondition: the early-out must pass"
    );
    let client = client_with_limits(ITEM_LIMIT, u64::MAX / 2).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("sr-anomaly"));
    client
        .install_observability_hooks(Arc::clone(&metrics))
        .await;

    let (too_large, log) = capture_warn_json(read_too_large(&client, tv.key_schema(), oid)).await;

    let SnapshotTooLargeAt::MidRead {
        limit, item_tally, ..
    } = too_large.at
    else {
        panic!(
            "a list of non-live ids must stop mid-read; got {:?}",
            too_large.at
        );
    };
    assert_eq!((limit, too_large.item_bound), (SnapshotLimit::Items, bound));
    assert!(item_tally > ITEM_LIMIT);
    assert_eq!(snapshot_reads(&metrics, "catchup", "too_large"), 1);
    let warn = log
        .iter()
        .find(|e| {
            e["level"] == "WARN"
                && e["fields"]["message"]
                    .as_str()
                    .is_some_and(|m| m.contains("item tally"))
        })
        .unwrap_or_else(|| panic!("the item-tally anomaly must WARN; captured: {log:?}"));
    let fields = &warn["fields"];
    assert_eq!(fields["object_id"], oid.to_string());
    assert_eq!(fields["item_bound"], bound);
    assert_eq!(fields["item_tally"], item_tally);
    assert_eq!(fields["item_limit"], ITEM_LIMIT);
}

/// Run `future` under a JSON subscriber admitting WARN and above, returning its
/// output parsed per line. `WithSubscriber` is right here: the read is awaited
/// inline, not spawned.
async fn capture_warn_json<F: std::future::Future>(
    future: F,
) -> (F::Output, Vec<serde_json::Value>) {
    let buf = Arc::new(StdMutex::new(Vec::<u8>::new()));
    let sink = Arc::clone(&buf);
    let subscriber = tracing_subscriber::fmt()
        .json()
        .with_max_level(tracing::Level::WARN)
        .with_writer(move || SharedBuf(Arc::clone(&sink)))
        .finish();
    let out = future
        .with_subscriber(tracing::Dispatch::new(subscriber))
        .await;
    let text = String::from_utf8(buf.lock().unwrap().clone()).expect("JSON log is UTF-8");
    let events = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("log line {l:?}: {e}")))
        .collect();
    (out, events)
}

struct SharedBuf(Arc<StdMutex<Vec<u8>>>);

impl std::io::Write for SharedBuf {
    fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(data);
        Ok(data.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// 5. Per-key refusals (SR-1.4)
// ═════════════════════════════════════════════════════════════════════════════
//
// Each staging plants a string on one key the script reads, on the shared b.atj
// `Fixture` (so the "every other cell flat" assertions run over the same key
// inventory b.atj's are calibrated on). The script's `redis.pcall` turns the
// `WRONGTYPE` into `SNAPSHOT_KEY_REFUSED: <family> <key> <raw>`, and exactly
// that family's `live_edit_snapshot_read_type_error_total` cell must tick.

async fn assert_atomic_wrongtype_ticks_only(fx: &Fixture, key: &str, family: &str) {
    fx.plant_string(key).await;

    let err = read_err(fx.tv.client(), fx.tv.key_schema(), fx.object_id).await;

    let ServiceError::Internal(msg) = &err else {
        panic!("a wrong-typed key is permanent, so the read must be Internal; got {err:?}");
    };
    assert!(
        msg.contains("EVALSHA object_snapshot_read (in-script read)") && msg.contains("WRONGTYPE"),
        "the refusal must be the script's per-key one for this plant; got {msg:?}"
    );
    assert_only_family_ticked(&fx.metrics, family, 1);
    assert_no_refusal_kind_ticked(&fx.metrics);
    assert_keyspace_type_errors_flat(&fx.metrics);
    assert_eq!(snapshot_reads(&fx.metrics, "catchup", "permanent_error"), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn atomic_read_of_a_wrongtype_object_header_ticks_header() {
    let fx = Fixture::empty("sr-ref-hdr").await;
    let key = fx.tv.key_schema().obj_header(&fx.oid());
    assert_atomic_wrongtype_ticks_only(&fx, &key, "header").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn atomic_read_of_a_wrongtype_sections_index_ticks_sections_index() {
    let fx = Fixture::empty("sr-ref-secs").await;
    let key = fx.tv.key_schema().obj_sections_index(&fx.oid());
    assert_atomic_wrongtype_ticks_only(&fx, &key, "sections_index").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn atomic_read_of_a_wrongtype_deleted_sections_index_ticks_deleted_sections_index() {
    let fx = Fixture::empty("sr-ref-dels").await;
    let key = fx.tv.key_schema().obj_deleted_sections_index(&fx.oid());
    assert_atomic_wrongtype_ticks_only(&fx, &key, "deleted_sections_index").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn atomic_read_of_a_wrongtype_fields_index_ticks_fields_index() {
    let fx = Fixture::seeded("sr-ref-flds").await;
    let key = fx.tv.key_schema().obj_fields_index(&fx.oid());
    assert_atomic_wrongtype_ticks_only(&fx, &key, "fields_index").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn atomic_read_of_a_wrongtype_time_samples_index_ticks_time_samples_index() {
    let fx = Fixture::seeded("sr-ref-tss").await;
    let key = fx.tv.key_schema().obj_time_samples_index(&fx.oid());
    assert_atomic_wrongtype_ticks_only(&fx, &key, "time_samples_index").await;
}

/// `obj:chs` — the family only `read_object_snapshot` touches (here its atomic
/// read; its unfenced fan-out in `snapshot_read_unfenced_test`), and only on a
/// marked object. Labelled `children_index`, never `children` (SR-1.4).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn atomic_read_of_a_wrongtype_children_index_ticks_children_index() {
    let fx = Fixture::seeded("sr-ref-chs").await;
    let key = fx.tv.key_schema().obj_children_index(&fx.oid());
    assert_atomic_wrongtype_ticks_only(&fx, &key, "children_index").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn atomic_read_of_a_wrongtype_section_hash_ticks_section() {
    let fx = Fixture::seeded("sr-ref-sec").await;
    let key = fx
        .tv
        .key_schema()
        .obj_section(&fx.oid(), &SECTION_ID.to_string());
    assert_atomic_wrongtype_ticks_only(&fx, &key, "section").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn atomic_read_of_a_wrongtype_tombstone_hash_ticks_deleted_section() {
    let fx = Fixture::empty("sr-ref-del").await;
    fx.update(create_section_delta(DOOMED_SECTION_ID, "Doomed"))
        .await
        .expect("the setup CreateSection must land");
    fx.update(delete_section_delta(DOOMED_SECTION_ID))
        .await
        .expect("the setup DeleteSection must land and write the tombstone");
    let key = fx
        .tv
        .key_schema()
        .obj_deleted_section(&fx.oid(), &DOOMED_SECTION_ID.to_string());
    assert_atomic_wrongtype_ticks_only(&fx, &key, "deleted_section").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn atomic_read_of_a_wrongtype_field_row_ticks_field() {
    let fx = Fixture::seeded("sr-ref-fld").await;
    let key = fx
        .tv
        .key_schema()
        .obj_field(&fx.oid(), &SECTION_ID.to_string(), FIELD_NAME);
    assert_atomic_wrongtype_ticks_only(&fx, &key, "field").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn atomic_read_of_a_wrongtype_time_sample_row_ticks_time_sample() {
    let fx = Fixture::seeded("sr-ref-ts").await;
    let key = fx
        .tv
        .key_schema()
        .obj_time_sample(&fx.oid(), &SECTION_ID.to_string(), SAMPLE_TIME);
    assert_atomic_wrongtype_ticks_only(&fx, &key, "time_sample").await;
}

/// On a marked object the root's list is read because `obj:chs` names "0".
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn atomic_read_of_a_wrongtype_children_list_ticks_children() {
    let fx = Fixture::seeded("sr-ref-ch").await;
    let key = fx.tv.key_schema().obj_children(&fx.oid(), "0");
    assert_atomic_wrongtype_ticks_only(&fx, &key, "children").await;
}

/// The keys the read must never touch: the commit-token record (`obj:wtok`,
/// b.75i), the external-value index, the live stream, the ext-retire carrier
/// and its attempt tokens, and the snapshot fence and writer-intent keys (`obj:snapfence`,
/// `obj:snapintent`; the fence protocol's, not the read's). A foreign value on
/// any of them leaves the read healthy and every counter flat — and so does one
/// on `obj:chs` of an UNMARKED object, which the unmarked read does not consult.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn atomic_read_over_wrongtype_keys_it_never_reads_ticks_nothing() {
    let fx = Fixture::seeded("sr-ref-untouched").await;
    let keys = fx.tv.key_schema();
    for key in [
        keys.obj_write_token(&fx.oid()),
        keys.obj_ext_index(&fx.oid()),
        keys.layer_stream(fx.object_id),
        keys.obj_ext_retire_carrier(&fx.oid()),
        keys.obj_ext_retire_attempts(&fx.oid()),
        keys.obj_snapshot_fence(&fx.oid()),
        keys.obj_snapshot_intent(&fx.oid()),
    ] {
        fx.plant_string(&key).await;
    }
    let marked = read_atomic(fx.tv.client(), keys, fx.object_id).await;
    assert_eq!(
        marked.state.sections.len(),
        1,
        "the read must still carry the fixture's section"
    );

    let removed: i64 = fx
        .tv
        .client()
        .pool
        .hdel(keys.obj_header(&fx.oid()), HEADER_FIELD_CHS_COMPLETE)
        .await
        .unwrap();
    assert_eq!(
        removed, 1,
        "fixture precondition: the object must have been marked"
    );
    fx.plant_string(&keys.obj_children_index(&fx.oid())).await;
    read_atomic(fx.tv.client(), keys, fx.object_id).await;

    assert_no_family_ticked(&fx.metrics);
    assert_no_refusal_kind_ticked(&fx.metrics);
    assert_eq!(snapshot_reads(&fx.metrics, "catchup", "ok"), 2);
}

// ═════════════════════════════════════════════════════════════════════════════
// 6. Metrics (SR-8.1, SR-8.2)
// ═════════════════════════════════════════════════════════════════════════════

/// `live_edit_snapshot_read_total{consumer, path="atomic", outcome}`. Panics on
/// an absent cell: every atomic cell is zero-seeded, so absence means the
/// handle never reached the read.
fn snapshot_reads(metrics: &Metrics, consumer: &str, outcome: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_snapshot_read_total",
        &[
            ("consumer", consumer),
            ("path", "atomic"),
            ("outcome", outcome),
        ],
    )
    .unwrap_or_else(|| {
        panic!(
            "live_edit_snapshot_read_total{{consumer=\"{consumer}\",path=\"atomic\",outcome=\
             \"{outcome}\"}} is absent — it is zero-seeded, so this handle never reached the read"
        )
    }) as u64
}

/// One tick per read, under the consumer that read and the outcome it had; the
/// duration histogram records every read, the size histograms only a read
/// that returned state, with the read's own tallies.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_object_snapshot_records_one_outcome_per_read_under_its_consumer() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let layer = "sr-metrics";
    let session_id = create_session(&tv, layer).await;
    let oid = create(&ls, &session_id, layer, &every_shape_state()).await;
    let client = client_with_limits(every_shape_item_bound(true) - 1, u64::MAX / 2).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("sr-metrics"));
    tv.client()
        .install_observability_hooks(Arc::clone(&metrics))
        .await;
    client
        .install_observability_hooks(Arc::clone(&metrics))
        .await;

    let ok = read_atomic_as(
        tv.client(),
        tv.key_schema(),
        oid,
        SnapshotReadConsumer::Checkpoint,
    )
    .await;
    let missing = read_object_snapshot(
        tv.client(),
        tv.key_schema(),
        "987654321",
        SnapshotReadConsumer::CatchupLegacyPerCommit,
    )
    .await;
    assert!(matches!(missing, Err(ServiceError::NotFound(_))));
    read_too_large(&client, tv.key_schema(), oid).await;

    let expected: BTreeMap<(&str, &str), u64> = BTreeMap::from([
        (("checkpoint", "ok"), 1),
        (("catchup_legacy_per_commit", "not_found"), 1),
        (("catchup", "too_large"), 1),
    ]);
    for consumer in ["catchup", "catchup_legacy_per_commit", "checkpoint"] {
        for outcome in [
            "ok",
            "not_found",
            "too_large",
            "transient_error",
            "permanent_error",
        ] {
            assert_eq!(
                snapshot_reads(&metrics, consumer, outcome),
                expected.get(&(consumer, outcome)).copied().unwrap_or(0),
                "consumer={consumer}, outcome={outcome}: exactly the three reads above may tick, \
                 each once, under its own consumer and outcome"
            );
        }
    }

    let text = render(&metrics);
    let count = |name: &str, labels: &[(&str, &str)]| sample_value(&text, name, labels);
    assert_eq!(
        count(
            "live_edit_snapshot_read_duration_seconds_count",
            &[("consumer", "checkpoint"), ("path", "atomic")]
        ),
        Some(1.0),
        "every read records its wall time"
    );
    assert_eq!(
        count(
            "live_edit_snapshot_read_duration_seconds_count",
            &[("consumer", "catchup"), ("path", "atomic")]
        ),
        Some(1.0),
        "a TOO_LARGE answer is a completed read and records its wall time too"
    );
    assert_eq!(
        (
            count(
                "live_edit_snapshot_read_reply_bytes_sum",
                &[("consumer", "checkpoint")]
            ),
            count(
                "live_edit_snapshot_read_entries_sum",
                &[("consumer", "checkpoint")]
            ),
        ),
        (Some(ok.reply_bytes as f64), Some(ok.entry_count as f64)),
        "the size histograms record the successful read's own tallies"
    );
    assert_eq!(
        (
            count(
                "live_edit_snapshot_read_reply_bytes_count",
                &[("consumer", "catchup")]
            ),
            count(
                "live_edit_snapshot_read_reply_bytes_count",
                &[("consumer", "catchup_legacy_per_commit")]
            ),
        ),
        (None, None),
        "a read that returned no state records no size"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// 7. Never tears
// ═════════════════════════════════════════════════════════════════════════════

/// Readers loop on `read_object_snapshot` while one writer commits a mix of
/// fast-path rows and structural `CreateSection` / `MoveSection` /
/// `DeleteSection`. After each commit the writer — the only writer — records
/// the quiescent state under its `seq`, through the reference reader: the
/// serial state at that cut.
/// Every atomic read must equal the serial state at the cut it reports. A read
/// that blended two commits (the pre-atomic tear) would carry a state no
/// commit produced.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn read_object_snapshot_racing_a_writer_returns_the_serial_state_at_its_cut() {
    const BLOCKS: u64 = 10;
    const READERS: usize = 3;
    let tv = Arc::new(TestValkey::setup_with_pool_size(6).await);
    let ls = layer_store(&tv);
    let layer = "sr-never-tears";
    let session_id = create_session(&tv, layer).await;
    let oid = create(&ls, &session_id, layer, &every_shape_state()).await;

    let serial: Arc<StdMutex<BTreeMap<u64, crate::storage::snapshot_read_support::SnapshotView>>> =
        Arc::new(StdMutex::new(BTreeMap::new()));
    let record = |snapshot: &live_edit::storage::diff_read::ObjectSnapshot| {
        serial
            .lock()
            .unwrap()
            .insert(snapshot.sequence_num, view(snapshot));
    };
    record(
        &read_object_state_for_test(tv.client(), tv.key_schema(), &oid.to_string())
            .await
            .unwrap(),
    );

    let done = Arc::new(AtomicBool::new(false));
    let start = Arc::new(tokio::sync::Barrier::new(READERS + 1));
    let mut readers = Vec::new();
    for _ in 0..READERS {
        let (tv, done, start) = (Arc::clone(&tv), Arc::clone(&done), Arc::clone(&start));
        readers.push(tokio::spawn(async move {
            start.wait().await;
            let mut seen = Vec::new();
            loop {
                let finished = done.load(Ordering::SeqCst);
                let a = read_atomic(tv.client(), tv.key_schema(), oid).await;
                seen.push((a.cut, view(&a.state)));
                if finished {
                    return seen;
                }
            }
        }));
    }

    start.wait().await;
    for b in 0..BLOCKS {
        let s = 100 + b;
        let name = format!("s{b}");
        let ops = [
            delta_with_set_field_inline(WORLD, &format!("k{}", b % 3), vec![b as u8], b),
            delta_with_time_sample_inline(CUBE, b as f64, vec![b as u8]),
            structural(vec![create_cmd(WORLD, s, &name)]),
            structural(vec![move_cmd(s, WORLD, SIBLING, &name)]),
            delta_with_set_field_external(s, "mesh", ext_hash(b as u8).to_vec(), 100 + b, 1, 1),
            structural(vec![delete_cmd(SIBLING, s)]),
        ];
        for delta in ops {
            commit(&ls, &session_id, layer, oid, delta).await;
            record(
                &read_object_state_for_test(tv.client(), tv.key_schema(), &oid.to_string())
                    .await
                    .unwrap(),
            );
        }
    }
    done.store(true, Ordering::SeqCst);

    let serial = serial.lock().unwrap().clone();
    let mut cuts = BTreeSet::new();
    for reader in readers {
        let seen = reader.await.expect("a reader task must not panic");
        let mut last = 0;
        for (cut, state) in seen {
            let expected = serial.get(&cut).unwrap_or_else(|| {
                panic!("an atomic read reported cut {cut}, which no commit produced")
            });
            assert_eq!(
                &state, expected,
                "the atomic read at cut {cut} must be exactly the serial state at that cut"
            );
            assert!(cut >= last, "one reader's cuts must never go backwards");
            last = cut;
            cuts.insert(cut);
        }
    }
    assert!(
        cuts.len() >= 2,
        "the readers must have observed more than one cut, or no read ever raced a commit and the \
         assertion above is vacuous; cuts: {cuts:?}"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// 8. The key set (Spec SDD SR-1.1, SR-2.2)
// ═════════════════════════════════════════════════════════════════════════════

/// What the script touched, from the store's own command feed — asserted on
/// the key set itself, never on a refusal (the local cluster and AMR accept an
/// undeclared other-slot key on the same node, so a refusal cannot prove
/// containment). The object carries every key family, including the
/// commit-token record a commit through `UpdateLayer` leaves (`obj:wtok`), its
/// stream, and the snapshot fence and writer-intent keys (`obj:snapfence`,
/// `obj:snapintent`), so "never read" is not vacuous.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn object_snapshot_read_touches_only_read_commands_on_keys_under_the_objects_hash_tag() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let layer = "sr-key-set";
    let session_id = create_session(&tv, layer).await;
    let oid = create(&ls, &session_id, layer, &every_shape_state()).await;
    commit(
        &ls,
        &session_id,
        layer,
        oid,
        structural(vec![create_cmd(ROOT, 9, "Doomed")]),
    )
    .await;
    commit(
        &ls,
        &session_id,
        layer,
        oid,
        structural(vec![delete_cmd(ROOT, 9)]),
    )
    .await;
    commit(
        &ls,
        &session_id,
        layer,
        oid,
        delta_with_set_field_inline(WORLD, "w", vec![1], 1),
    )
    .await;
    let o = oid.to_string();
    let keys = tv.key_schema();
    // The snapshot fence and writer-intent keys, as their scripts leave them: a
    // closed fenced epoch, then a registered intent (held to the end of the
    // test so its key stays). The size only sets their lifetimes — the intent
    // lasts `intent_ttl(L)`, minutes at this size — so neither lapses mid-test.
    let fenced_size = LayerSize {
        items: 10_000_000,
        bytes: 120_000_000,
    };
    match snapshot_fence::acquire_fence(tv.client(), keys, oid, fenced_size)
        .await
        .unwrap()
    {
        AcquireVerdict::Granted(lease) => assert_eq!(
            lease.release(Some(fenced_size)).await.unwrap(),
            ReleaseVerdict::Verified
        ),
        other => panic!("fixture: the fence acquire must grant; got {other:?}"),
    }
    let _intent = match snapshot_fence::register_intent(tv.client(), keys, oid, fenced_size)
        .await
        .unwrap()
    {
        IntentVerdict::Registered(intent) => intent,
        other => panic!("fixture: the intent must register; got {other:?}"),
    };
    for (key, what) in [
        (keys.obj_write_token(&o), "obj:wtok"),
        (keys.layer_stream(oid), "obj:stream"),
        (keys.obj_snapshot_fence(&o), "obj:snapfence"),
        (keys.obj_snapshot_intent(&o), "obj:snapintent"),
    ] {
        let n: i64 = tv.client().pool.exists(&key).await.unwrap();
        assert_eq!(
            n, 1,
            "fixture precondition: {what} must exist, or 'never read' is vacuous"
        );
    }
    let header_key = keys.obj_header(&o);
    let tag = format!("{}{{{oid}}}:obj:", tv.prefix());

    for marked in [true, false] {
        if !marked {
            unmark(&tv, oid).await;
        }
        let mut tap = MonitorTap::start(&test_valkey_addr()).await;
        read_atomic(tv.client(), keys, oid).await;
        let fence = monitor_fence(tv.prefix());
        let _: i64 = tv.client().pool.exists(&fence).await.unwrap();
        let script = script_commands(tap.drain_until(&fence).await, &header_key);
        let label = if marked { "marked" } else { "unmarked" };

        for c in &script {
            assert!(
                ["HGETALL", "SCARD", "ZCARD", "SMEMBERS", "ZRANGE", "LRANGE"]
                    .contains(&c.name().as_str()),
                "{label}: the script must issue only reads; it issued {c:?}"
            );
            let key = c
                .args
                .get(1)
                .unwrap_or_else(|| panic!("{label}: {c:?} names no key"));
            assert!(
                key.starts_with(&tag),
                "{label}: every key the script touches must carry the object's hash tag ({tag}…); \
                 it touched {key}"
            );
            for never in [
                "wtok",
                "ext",
                "stream",
                "extret",
                "extret:attempts",
                "snapfence",
                "snapintent",
            ] {
                assert_ne!(
                    key.strip_prefix(&tag),
                    Some(never),
                    "{label}: the snapshot read must never read obj:{never}"
                );
            }
        }
        let families: BTreeSet<&str> = script
            .iter()
            .filter_map(|c| c.args.get(1)?.strip_prefix(&tag))
            .map(|rest| rest.split(':').next().unwrap_or(rest))
            .collect();
        let expected: BTreeSet<&str> = if marked {
            BTreeSet::from([
                "hdr", "secs", "dels", "flds", "tss", "chs", "sec", "del", "fld", "ts", "ch",
            ])
        } else {
            BTreeSet::from([
                "hdr", "secs", "dels", "flds", "tss", "sec", "del", "fld", "ts", "ch",
            ])
        };
        assert_eq!(
            families, expected,
            "{label}: the record must show every family the read reads and no other — the marked \
             read consults obj:chs, the unmarked one never does"
        );
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// 9. Width pinning (SR-1.2 step 2)
// ═════════════════════════════════════════════════════════════════════════════

/// The widest row of each family a writer wrote, by `HKEYS`.
async fn widest_rows(tv: &TestValkey, oid: u64) -> BTreeMap<&'static str, (u64, Vec<String>)> {
    let tag = format!("{}{{{oid}}}:obj:", tv.prefix());
    let mut widest: BTreeMap<&'static str, (u64, Vec<String>)> = BTreeMap::new();
    for key in crate::common::scan_object_keys(tv.client(), tv.prefix(), oid).await {
        let key = key.as_str_lossy().to_string();
        let Some(rest) = key.strip_prefix(&tag) else {
            continue;
        };
        // `sec:{id}` and `del:{id}`, never the `secs` / `dels` index keys.
        let family = match rest.split(':').next() {
            Some("sec") => "section",
            Some("del") => "tombstone",
            Some("fld") => "field",
            Some("ts") => "time_sample",
            _ => continue,
        };
        let mut columns: Vec<String> = tv.client().pool.hkeys(&key).await.unwrap();
        columns.sort();
        let entry = widest.entry(family).or_insert((0, Vec::new()));
        if columns.len() as u64 > entry.0 {
            *entry = (columns.len() as u64, columns);
        }
    }
    widest
}

/// The four width constants equal the widest row every writer of section,
/// tombstone, field and time-sample hashes writes: `queue_object_state`
/// (CreateLayer; a DR restore shares it), `execute_mutation`'s `SetSection` /
/// `CreateTombstone` / `SetField` / `SetTimeSample` arms (the Txn backend), and
/// the Lua fast path's `SF` / `ST` arms — each value row in both shapes, since
/// the external one is the wide one. A writer that adds a column to any row
/// pushes that family past its constant and fails here: the bound would
/// under-count, letting an object pass the early-out and trip the item tally
/// mid-read instead.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn snapshot_hash_widths_equal_the_widest_row_every_writer_writes() {
    let tv = TestValkey::setup().await;
    let widths: BTreeMap<&str, u64> = BTreeMap::from([
        ("section", SNAPSHOT_SECTION_HASH_WIDTH),
        ("tombstone", SNAPSHOT_TOMBSTONE_HASH_WIDTH),
        ("field", SNAPSHOT_FIELD_HASH_WIDTH),
        ("time_sample", SNAPSHOT_TIME_SAMPLE_HASH_WIDTH),
    ]);
    let session_id = create_session(&tv, "sr-widths").await;
    const HOST: u64 = 1;
    let mut host_only = base_state();
    host_only
        .sections
        .insert(HOST, section(HOST, ROOT, "Host", 0));
    host_only.children.insert(ROOT, vec![HOST]);

    // queue_object_state.
    let lua = layer_store(&tv);
    let mut materialized = every_shape_state();
    materialized
        .fields
        .insert((WORLD, "wide".into()), external_field(WORLD, "wide", 9));
    materialized
        .time_samples
        .insert((WORLD, OrderedFloat(8.0)), external_sample(WORLD, 8.0, 8));
    let a = create(&lua, &session_id, "sr-widths-materialize", &materialized).await;

    // execute_mutation, every arm through the Txn backend.
    let txn = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_backend(UpdateLayerBackend::Txn);
    let b = create(&txn, &session_id, "sr-widths-txn", &host_only).await;
    let row_delta = |extra: Vec<StructCommand>| OwnedDelta {
        struct_commands: extra,
        set_fields: vec![
            SetField {
                section_id: HOST,
                key_name: "i".into(),
                value_or_ext_hash: vec![1],
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 1,
            },
            SetField {
                section_id: HOST,
                key_name: "e".into(),
                value_or_ext_hash: ext_hash(5).to_vec(),
                ext_value_size: 64,
                ext_value_index: 1,
                set_order: 2,
            },
        ],
        time_samples: vec![
            OodfTimeSample {
                section_id: HOST,
                time: 1.0,
                value_or_ext_hash: vec![1],
                ext_value_size: 0,
                ext_value_index: 0,
            },
            OodfTimeSample {
                section_id: HOST,
                time: 2.0,
                value_or_ext_hash: ext_hash(6).to_vec(),
                ext_value_size: 64,
                ext_value_index: 1,
            },
        ],
        ..Default::default()
    };
    commit(
        &txn,
        &session_id,
        "sr-widths-txn",
        b,
        row_delta(vec![create_cmd(HOST, 2, "Doomed")]),
    )
    .await;
    commit(
        &txn,
        &session_id,
        "sr-widths-txn",
        b,
        structural(vec![delete_cmd(HOST, 2)]),
    )
    .await;

    // The Lua fast path's SF / ST arms.
    let c = create(&lua, &session_id, "sr-widths-lua", &host_only).await;
    let fast = row_delta(Vec::new());
    assert!(
        live_edit::storage::lua::is_fast_path(&Some(fast.clone())),
        "fixture precondition"
    );
    commit(&lua, &session_id, "sr-widths-lua", c, fast).await;

    for (writer, oid, families) in [
        (
            "queue_object_state",
            a,
            vec!["section", "field", "time_sample"],
        ),
        (
            "execute_mutation",
            b,
            vec!["section", "tombstone", "field", "time_sample"],
        ),
        ("update_layer_fast.lua", c, vec!["field", "time_sample"]),
    ] {
        let widest = widest_rows(&tv, oid).await;
        for family in families {
            let (width, columns) = widest.get(family).unwrap_or_else(|| {
                panic!("fixture precondition: {writer} must have written a {family} row")
            });
            assert_eq!(
                *width, widths[family],
                "{writer}'s widest {family} row has {width} columns {columns:?}, but the item bound \
                 counts {} per {family} hash. Raise the SNAPSHOT_*_HASH_WIDTH constant with the \
                 writer, or the early-out under-counts (Spec SDD SR-1.2 step 2)",
                widths[family]
            );
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// 10. Reply shapes (SR-1.3) — through the reply proxy
// ═════════════════════════════════════════════════════════════════════════════

/// A reply from a script this binary does not carry — another format tag — or
/// one whose shape is wrong is refused as `Internal`, never decoded into an
/// empty or partial state.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_object_snapshot_of_a_reply_in_another_format_or_shape_is_internal() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let layer = "sr-reply-shape";
    let session_id = create_session(&tv, layer).await;
    let oid = create(&ls, &session_id, layer, &every_shape_state()).await;
    let header_key = tv.key_schema().obj_header(&oid.to_string());

    for (reply, expect) in [
        ("return {'object-snapshot-read-v0', 'NOT_FOUND'}", "format"),
        (
            "return {'object-snapshot-read-v1', 'OK'}",
            "has 2 elements, expected 9",
        ),
        (
            "return {'object-snapshot-read-v1', 'SOMETHING'}",
            "unknown status",
        ),
        (
            "return {'object-snapshot-read-v1', 'TOO_LARGE', 'bound', {1, 2}, 9, {0, 0, 0}}",
            "cardinalities",
        ),
        (
            "return {'object-snapshot-read-v1', 'TOO_LARGE', 'sideways', {1, 0, 0, 0, -1, 0}, 9, \
             {0, 0, 0}}",
            "unknown TOO_LARGE limit",
        ),
        (
            "return {'object-snapshot-read-v1', 'TOO_LARGE', 'items', {1, 0, 0, 0, -1, 0}, 9, \
             {1, 2}}",
            "tallies are not three integers",
        ),
        (
            "return {'object-snapshot-read-v1', 'OK', {'seq', '5'}, {{'1'}}, {}, {}, {}, {}, \
             {1, 1, 1}}",
            "a sections entry is not an {id, body} pair",
        ),
        (
            "return {'object-snapshot-read-v1', 'OK', {'seq'}, {}, {}, {}, {}, {}, {0, 0, 0}}",
            "odd-length",
        ),
    ] {
        let proxy = ScriptReplyProxy::start(
            test_valkey_addr(),
            script_frames_naming(header_key.clone(), ProxyAction::Eval(reply.to_string())),
        )
        .await;
        let err = read_err(&client_through(&proxy).await, tv.key_schema(), oid).await;
        assert!(
            proxy.hits() >= 1,
            "the proxy must have substituted the reply"
        );
        assert!(
            matches!(&err, ServiceError::Internal(m) if m.contains(expect)),
            "reply {reply:?} must be refused as Internal ({expect:?}); got {err:?}"
        );
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// 11. The startup self-test (SR-1.8)
// ═════════════════════════════════════════════════════════════════════════════

const SELF_TEST_DEADLINE: Duration = Duration::from_secs(5);

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_snapshot_read_self_test_passes_on_the_test_store() {
    let tv = TestValkey::setup().await;
    assert_eq!(
        run_snapshot_read_self_test(tv.client(), tv.key_schema(), SELF_TEST_DEADLINE).await,
        Ok(()),
        "the test store allows a script's undeclared same-slot read"
    );
    assert_eq!(
        run_snapshot_read_self_test(tv.client(), &KeySchema::default(), SELF_TEST_DEADLINE).await,
        Ok(()),
        "and so it does under the production key prefix"
    );
}

/// The self-test's deadline in [`self_test_verdict`]: short, so the stall
/// case returns quickly, and far above a local proxied round trip.
const PROXIED_SELF_TEST_DEADLINE: Duration = Duration::from_millis(500);

/// The detail the self-test carries when its deadline, not a reply, decided
/// the verdict. A test of a reply's classification must not accept it: under
/// load a slow proxied round trip yields the same `Unavailable` whatever the
/// classification of the reply would have been.
const DEADLINE_DETAIL: &str = "no reply within";

/// The self-test's verdict when the proxy answers its `EVALSHA` as `action`
/// directs: with a substituted one-line script's reply, by closing the
/// connection, or never.
async fn self_test_verdict(action: ProxyAction) -> Result<(), SnapshotSelfTestFailure> {
    let proxy = ScriptReplyProxy::start(
        test_valkey_addr(),
        script_frames_naming(SNAPSHOT_SELF_TEST_OBJECT_ID.to_string(), action),
    )
    .await;
    let client: ValkeyClient = client_through(&proxy).await;
    let keys = KeySchema::new(&format!("test_selftest_{}:", uuid::Uuid::new_v4()));
    let verdict = run_snapshot_read_self_test(&client, &keys, PROXIED_SELF_TEST_DEADLINE).await;
    assert!(
        proxy.hits() >= 1,
        "the proxy must have intercepted the self-test's EVALSHA"
    );
    verdict
}

/// Assert `verdict` is `Unavailable` decided by the staged reply — its detail
/// carries `staged` — and not by the deadline.
fn assert_unavailable_from_reply(verdict: &Result<(), SnapshotSelfTestFailure>, staged: &str) {
    let Err(SnapshotSelfTestFailure::Unavailable(detail)) = verdict else {
        panic!(
            "{staged:?} is not a verdict on the store's script policy: the self-test must be \
             Unavailable (non-fatal), never a refusal (exit); got {verdict:?}"
        );
    };
    assert!(
        detail.contains(staged) && !detail.contains(DEADLINE_DETAIL),
        "the Unavailable must come from the staged reply {staged:?}, not from the deadline; \
         detail: {detail:?}"
    );
}

fn probe_refusal(raw: &str) -> String {
    format!("SNAPSHOT_SELFTEST_REFUSED: test_selftest:{{{SNAPSHOT_SELF_TEST_OBJECT_ID}}}:obj:secs {raw}")
}

/// **F1 regression.** A store temporarily not serving — a restarted primary
/// replaying its data (`-LOADING`) or busy behind a long script (`-BUSY`) —
/// answers the self-test's `EVALSHA` that way although `connect` succeeded.
/// fred hands both over as `ErrorKind::Unknown`; read as a refusal they would
/// exit the process and crash-loop the pod through the whole replay. They
/// must be `Unavailable`: logged, and startup continues. (Since b.xd1 the
/// transient predicate calls both transient, which is one of the two guards
/// that keep them off the fatal arm; `is_backend_unavailable_reply` is the
/// other, and the one that covers the write-gated `-MISCONF` / `-READONLY` /
/// `-NOREPLICAS`.)
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_snapshot_read_self_test_is_unavailable_when_the_store_is_loading_or_busy() {
    for reply in [
        "LOADING Valkey is loading the dataset in memory",
        "BUSY Valkey is busy running a script. You can only call SCRIPT KILL or FUNCTION KILL or SHUTDOWN NOSAVE.",
    ] {
        let verdict = self_test_verdict(ProxyAction::Eval(error_reply_script(reply))).await;
        assert_unavailable_from_reply(&verdict, reply);
    }
}

/// **F1 regression, cluster half.** The probe key's slot migrating, not served,
/// or the cluster down makes the store refuse the probe for the slot's state —
/// arriving as a tagged self-test refusal carrying one of the five
/// cluster-state texts (Valkey's four, and Azure Managed Redis's wording of the
/// non-local-key one, bare or with the `ERR ` code). That is a state of the
/// slot, not a store that refuses undeclared keys, so it must be
/// `Unavailable`, not the fatal `UndeclaredKeyRefused`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_snapshot_read_self_test_is_unavailable_when_the_cluster_refuses_for_the_slots_state() {
    for raw in [
        "ERR Script attempted to access a non local key in a cluster node",
        "ERR Unable to execute command 'EXISTS' in script because undeclared keys were accessed during rehashing of the slot",
        "ERR Script attempted to execute a command while the cluster is down",
        "ERR Script attempted to access a slot not served",
        "Lua script attempted to access a non local key in a cluster node",
        "ERR Lua script attempted to access a non local key in a cluster node",
    ] {
        let verdict =
            self_test_verdict(ProxyAction::Eval(error_reply_script(&probe_refusal(raw)))).await;
        assert_unavailable_from_reply(&verdict, raw);
    }
}

/// The other direction: a store that genuinely refuses the undeclared
/// same-slot read stops startup (every atomic read would fail the same way).
/// That includes AMR's non-local-key wording when it arrives decorated as an
/// uncaught script error (a location prefix, or the 7.x script suffix): the
/// match is anchored, and those shapes are not what `redis.pcall` returns.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_snapshot_read_self_test_fails_startup_when_the_store_refuses_the_undeclared_key() {
    for raw in [
        "ERR Script attempted to access keys that do not hash to the same slot",
        "@user_script: 30: Lua script attempted to access a non local key in a cluster node",
        "Lua script attempted to access a non local key in a cluster node script: \
         b08075fc7374844c5b04707915156ba3a0cd5466, on @user_script:30.",
    ] {
        let verdict =
            self_test_verdict(ProxyAction::Eval(error_reply_script(&probe_refusal(raw)))).await;
        assert!(
            matches!(
                verdict,
                Err(SnapshotSelfTestFailure::UndeclaredKeyRefused(_))
            ),
            "{raw:?} is not one of the five anchored cluster-state texts, so it is the store's \
             policy: fatal; got {verdict:?}"
        );
    }
}

/// A connection the store drops under the self-test's `EVALSHA` is transient
/// IO (`is_transient_fred_error`): `Unavailable`, decided by the error and
/// not by the deadline.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_snapshot_read_self_test_is_unavailable_when_the_connection_drops() {
    let verdict = self_test_verdict(ProxyAction::Close).await;
    let Err(SnapshotSelfTestFailure::Unavailable(detail)) = &verdict else {
        panic!("a dropped connection is transient: Unavailable; got {verdict:?}");
    };
    assert!(
        !detail.contains(DEADLINE_DETAIL),
        "the Unavailable must come from the dropped connection, not the deadline; detail: \
         {detail:?}"
    );
}

/// A store that refuses the script itself — an ACL denial — or answers in a
/// format this binary does not carry is `ScriptRefused` (fatal), and one that
/// never answers within the deadline is `Unavailable`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_snapshot_read_self_test_refuses_a_denied_or_foreign_script_and_tolerates_no_reply() {
    let denied = self_test_verdict(ProxyAction::Eval(error_reply_script(
        "NOPERM User default has no permissions to run the evalsha command",
    )))
    .await;
    assert!(
        matches!(denied, Err(SnapshotSelfTestFailure::ScriptRefused(_))),
        "an ACL denial is permanent: fatal; got {denied:?}"
    );
    let foreign = self_test_verdict(ProxyAction::Eval(
        "return {'object-snapshot-read-v0', 'SELFTEST', 0}".to_string(),
    ))
    .await;
    assert!(
        matches!(foreign, Err(SnapshotSelfTestFailure::ScriptRefused(_))),
        "a reply in another format means another script ran: fatal; got {foreign:?}"
    );
    let silent = self_test_verdict(ProxyAction::Stall).await;
    assert!(
        matches!(&silent, Err(SnapshotSelfTestFailure::Unavailable(d)) if d.contains(DEADLINE_DETAIL)),
        "no reply within the deadline is not a verdict on the store; got {silent:?}"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// 12. The fault seam: in-script cluster refusals and the classifier
// ═════════════════════════════════════════════════════════════════════════════

/// `ObjectSnapshotReadFaultInjector` answers the `EVALSHA` with a canned error
/// that takes the real classifier, so the tagged shapes a live single-node
/// store cannot produce are staged here. Needs `--features
/// test-fault-injection`; compiled out without it.
#[cfg(feature = "test-fault-injection")]
mod fault_seam {
    use super::*;

    use fred::error::{Error as FredError, ErrorKind as FredErrorKind};
    use live_edit::storage::diff_read::ObjectSnapshotReadFaultInjector;

    /// Inject `injected` as the next read's `EVALSHA` result and return the
    /// read's error, asserting the canned error was consumed.
    async fn read_with_injected(fx: &Fixture, injected: FredError) -> ServiceError {
        let injector = Arc::new(ObjectSnapshotReadFaultInjector::new());
        injector.push(injected);
        fx.tv
            .client()
            .install_object_snapshot_read_fault_injector_for_test(Arc::clone(&injector));
        let err = read_err(fx.tv.client(), fx.tv.key_schema(), fx.object_id).await;
        assert_eq!(
            injector.remaining(),
            0,
            "the canned error must have answered the read"
        );
        err
    }

    fn tagged(family: &str, key: &str, raw: &str) -> FredError {
        FredError::new(
            FredErrorKind::Unknown,
            format!("SNAPSHOT_KEY_REFUSED: {family} {key} {raw}"),
        )
    }

    /// The five texts for a script's undeclared key refused for the slot's
    /// state — Valkey's four and Azure Managed Redis's wording of the
    /// non-local-key one — with and without the `ERR ` code `redis.pcall` adds,
    /// tagged under several families: each is the in-script form of a cluster
    /// redirect, so the read is transient (`Aborted`, its message naming the
    /// refused key), ticks no refusal counter under any family, and counts as
    /// `outcome="transient_error"` — without this, every read of an object in
    /// a migrating slot would fail non-retryably for the whole resharding.
    /// (The classifier attributes these to the header family; for a transient
    /// error that attribution moves no counter, so nothing here can observe
    /// it.)
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn atomic_read_refused_for_a_slots_state_is_transient_and_ticks_no_refusal_counter() {
        let fx = Fixture::seeded("sr-seam-cluster").await;
        let key = fx.tv.key_schema().obj_children_index(&fx.oid());
        let raws = [
            "ERR Script attempted to access a non local key in a cluster node",
            "Script attempted to access a non local key in a cluster node",
            "ERR Unable to execute command 'SMEMBERS' in script because undeclared keys were accessed during rehashing of the slot",
            "ERR Script attempted to execute a command while the cluster is down",
            "ERR Script attempted to access a slot not served",
            "Lua script attempted to access a non local key in a cluster node",
            "ERR Lua script attempted to access a non local key in a cluster node",
        ];
        const FAMILIES: [&str; 7] = [
            "children_index",
            "section",
            "children",
            "fields_index",
            "header",
            "time_sample",
            "deleted_sections_index",
        ];
        for (raw, family) in raws.iter().zip(FAMILIES) {
            let err = read_with_injected(&fx, tagged(family, &key, raw)).await;
            assert!(
                matches!(&err, ServiceError::Aborted(m) if m.contains(&format!("the cluster refused the in-script read of {key}"))),
                "{raw:?} under {family} is a slot state, so the read is transient and names the \
                 refused key; got {err:?}"
            );
        }
        assert_no_family_ticked(&fx.metrics);
        assert_no_refusal_kind_ticked(&fx.metrics);
        assert_eq!(
            snapshot_reads(&fx.metrics, "catchup", "transient_error"),
            raws.len() as u64
        );
        assert_eq!(snapshot_reads(&fx.metrics, "catchup", "permanent_error"), 0);
    }

    /// The match is exact and anchored: a refusal that merely CONTAINS a
    /// cluster-state text is somebody else's error and stays a permanent
    /// per-key refusal, counted on `refused_total{kind="unknown"}`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn atomic_read_refused_with_a_cluster_text_inside_another_error_stays_permanent() {
        let fx = Fixture::seeded("sr-seam-decoy").await;
        let key = fx.tv.key_schema().obj_sections_index(&fx.oid());
        let err = read_with_injected(
            &fx,
            tagged(
                "sections_index",
                &key,
                "ERR wrapped: Script attempted to access a non local key in a cluster node",
            ),
        )
        .await;
        assert!(
            matches!(&err, ServiceError::Internal(m) if m.contains("in-script read")),
            "a substring is not the anchored text; got {err:?}"
        );
        crate::storage::b_muu_tests::assert_only_refusal_kind_ticked(&fx.metrics, "unknown", 1);
        assert_no_family_ticked(&fx.metrics);
        assert_eq!(snapshot_reads(&fx.metrics, "catchup", "permanent_error"), 1);
    }

    /// A failure of the `EVALSHA` itself is classified by the unchanged
    /// `map_snapshot_load_err`: a timeout is transient and counts nothing; a
    /// tag naming a family this binary does not know is a script/binary
    /// mismatch, permanent, attributed to the header.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn atomic_read_script_level_failures_take_the_unchanged_classifier() {
        let fx = Fixture::seeded("sr-seam-script").await;
        let timeout = read_with_injected(
            &fx,
            FredError::new(FredErrorKind::Timeout, "Request timed out."),
        )
        .await;
        assert!(
            matches!(timeout, ServiceError::Aborted(_)),
            "a timeout is transient; got {timeout:?}"
        );
        assert_no_refusal_kind_ticked(&fx.metrics);

        let unknown_family =
            read_with_injected(&fx, tagged("no_such_family", "k", "WRONGTYPE Operation")).await;
        assert!(
            matches!(&unknown_family, ServiceError::Internal(m) if m.starts_with("EVALSHA object_snapshot_read failed")),
            "an unknown family is not trusted as a per-key refusal; got {unknown_family:?}"
        );
        crate::storage::b_muu_tests::assert_only_refusal_kind_ticked(&fx.metrics, "unknown", 1);
        assert_no_family_ticked(&fx.metrics);
        assert_eq!(
            (
                snapshot_reads(&fx.metrics, "catchup", "transient_error"),
                snapshot_reads(&fx.metrics, "catchup", "permanent_error"),
            ),
            (1, 1)
        );
    }
}
