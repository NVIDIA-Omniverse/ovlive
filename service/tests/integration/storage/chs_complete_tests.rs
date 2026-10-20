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

//! Integration tests for the children-index completeness marker — the
//! `obj:hdr` column `chs_complete`
//! ([`HEADER_FIELD_CHS_COMPLETE`] = [`CHS_COMPLETE_MARKED`]) and the
//! `le:{oid}:obj:chs` index it vouches for (Spec SDD SR-1.7).
//!
//! # What the marker promises
//!
//! A reader that finds the marker may read only the children lists `obj:chs`
//! names, instead of probing the root plus every section. On a marked object,
//! then, a missing index member is not benign the way it is for retirement
//! (b.jdr): it hides a live list from that reader. So the contract pinned here
//! is the strict one:
//!
//! > on a marked object, `SMEMBERS obj:chs` equals the set of parent ids that
//! > have an existing `le:{oid}:obj:ch:*` key.
//!
//! The right-hand side is computed by **SCANning the keyspace**, never from
//! `obj:chs` or the snapshot loader, so the oracle is independent of both the
//! index under test and the loader's own parent set (which reads `obj:secs`
//! plus the root and cannot see a list under a dangling parent).
//!
//! # Which writes are covered
//!
//! Every arm that creates, rewrites or deletes a children list, each driven
//! through its real entry point and each followed by the equality assertion:
//!
//! | Arm | Driven by |
//! |---|---|
//! | `write_object` (`queue_object_state`) | `create_layer` |
//! | `write_restored_object` (`queue_object_state`) | `apply_checkpoint` over a wiped keyspace |
//! | `execute_mutation` `SetChildren`, non-empty | `CreateSection` / `DiffChangeSection` commit |
//! | `execute_mutation` `SetChildren`, empty | `MoveSection` of a parent's only child |
//! | `execute_mutation` `DeleteChildrenList` | `DeleteSection` cascade |
//! | reset (`DeleteChildrenList` per parent, then the rebuild's `SetChildren`) | `is_diff && base_version == 0` commit |
//!
//! The restore case runs on a **wiped** keyspace on purpose. A restore over
//! index-less residue can leave a stray pre-index list under a parent that
//! restores as a live childless section: the restore does not `DEL` it and the
//! index does not name it, beside a marked header. That is a known shape,
//! documented on `HEADER_FIELD_CHS_COMPLETE` and owned by the read (Task 4) —
//! no write of the object produced it, so it is not a write this file pins.
//!
//! # The marker itself
//!
//! - Present after `create_layer` and after a DR restore — the two
//!   `queue_object_state` entry points, its only writer.
//! - Unchanged by every later commit of a marked object: a structural commit,
//!   a reset, and a Lua fast-path commit.
//! - Never conjured onto an unmarked object: there is no migration, so an
//!   object materialized before the marker existed stays unmarked through a
//!   structural commit and a fast-path commit.
//!
//! The unmarked-partial-index *read* regression needs the new atomic read and
//! lands with it (Task 4), not here.

use std::collections::BTreeSet;
use std::sync::Arc;

use fred::interfaces::{HashesInterface, SetsInterface};

use live_edit::checkpoint::apply::apply_checkpoint;
use live_edit::config::UpdateLayerBackend;
use live_edit::domain::types::{ObjectState, Section};
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::layer::{LayerStore, CHS_COMPLETE_MARKED, HEADER_FIELD_CHS_COMPLETE};
use live_edit::storage::lua::is_fast_path;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::relay::RelayedUpdates;
use live_edit::storage::test_support::TestValkey;
use oodf_ffi::{
    CommandCreateSection, CommandDeleteSection, CommandDiffChangeSection, CommandMoveSection,
    OwnedDelta, StructCommand,
};

use crate::common::owned_delta::delta_with_set_field_inline;
use crate::storage::checkpoint_apply_test::build_checkpoint_ompf;

const WRITER: &str = "chs-complete-writer";
const CREATOR: &str = "chs-complete-user";

// ── Fixture section ids ──────────────────────────────────────────────────────
//
// Root → [WORLD, SIBLING]; WORLD → [CUBE]; DANGLING_PARENT → [ORPHAN];
// SIBLING → [] (an explicit empty vec). So the created object owns exactly
// three children lists — `obj:ch:0`, `obj:ch:1`, `obj:ch:77` — and covers every
// shape `queue_object_state`'s children loop distinguishes: the synthetic root
// (a member of no other index), a live-section parent, a dangling parent (in
// neither `obj:secs` nor `obj:dels`), and a parent whose empty vec must write
// neither a key nor an index member.

/// The synthetic root's parent id.
const ROOT: u64 = 0;
/// Top-level section owning [`CUBE`].
const WORLD: u64 = 1;
/// Leaf under [`WORLD`].
const CUBE: u64 = 2;
/// Top-level section carrying an explicit empty children vec at create time.
const SIBLING: u64 = 3;
/// A live section whose `parent_id` names no section (SDD §10.4).
const ORPHAN: u64 = 4;
/// The parent id [`ORPHAN`] hangs under; names no section.
const DANGLING_PARENT: u64 = 77;

fn mock_ext_store() -> Arc<MockExternalValueStore> {
    Arc::new(MockExternalValueStore::new())
}

/// The Lua backend is named explicitly rather than left to `LayerStore::new`'s
/// default, because the fast-path commits below are about that backend.
fn layer_store(tv: &TestValkey) -> LayerStore {
    LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_backend(UpdateLayerBackend::Lua)
}

async fn create_test_session(tv: &TestValkey) -> String {
    SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create("chs-complete-session", "/stage", CREATOR)
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

/// See the fixture-id block above for the tree.
fn state_with_every_children_list_shape() -> ObjectState {
    let mut state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usda".to_string(),
        next_set_order: 10,
        ..Default::default()
    };
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
    state.children.insert(SIBLING, Vec::new());
    state.children.insert(DANGLING_PARENT, vec![ORPHAN]);
    state
}

async fn create_marked_layer(ls: &LayerStore, session_id: &str, layer: &str) -> u64 {
    ls.create_layer(
        session_id,
        layer,
        CREATOR,
        &state_with_every_children_list_shape(),
        mock_ext_store(),
    )
    .await
    .expect("create_layer must succeed")
}

/// Commit `delta`, which must route down the WATCH/MULTI/EXEC slow path where
/// `execute_mutation` runs. Asserted, not assumed: a delta the fast path
/// admitted would never reach the arm the caller is testing.
async fn commit_structural(
    ls: &LayerStore,
    session_id: &str,
    layer: &str,
    oid: u64,
    delta: OwnedDelta,
) {
    let delta = Some(delta);
    assert!(
        !is_fast_path(&delta),
        "fixture precondition: this delta must take the txn slow path, or \
         `execute_mutation`'s children arms never run"
    );
    ls.update_layer_relayed(session_id, layer, oid, delta, WRITER, 0)
        .await
        .expect("structural update_layer must succeed");
}

/// Commit one inline `SetField` on [`WORLD`], asserting it qualifies for the
/// Lua fast path (the store is pinned to the Lua backend in [`layer_store`]).
async fn commit_fast_path(ls: &LayerStore, session_id: &str, layer: &str, oid: u64) {
    let delta = Some(delta_with_set_field_inline(WORLD, "color", vec![0x2A], 1));
    assert!(
        is_fast_path(&delta),
        "fixture precondition: a field-only delta must qualify for the Lua \
         fast path, or this commit is a second slow-path commit in disguise"
    );
    ls.update_layer_relayed(session_id, layer, oid, delta, WRITER, 0)
        .await
        .expect("fast-path update_layer must succeed");
}

fn struct_delta(cmds: Vec<StructCommand>) -> OwnedDelta {
    OwnedDelta {
        is_diff: false,
        struct_commands: cmds,
        ..Default::default()
    }
}

fn create_cmd(parent_id: u64, section_id: u64, name: &str) -> StructCommand {
    StructCommand::CreateSection(CommandCreateSection {
        parent_id,
        section_id,
        section_name: name.to_string(),
        section_type: 1,
    })
}

// ── Readers ──────────────────────────────────────────────────────────────────

/// The marker column, `None` when absent.
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

/// `SMEMBERS le:{oid}:obj:chs`.
async fn indexed_parents(tv: &TestValkey, oid: u64) -> BTreeSet<String> {
    let members: Vec<String> = tv
        .client()
        .pool
        .smembers(tv.key_schema().obj_children_index(&oid.to_string()))
        .await
        .expect("SMEMBERS obj:chs must succeed");
    members.into_iter().collect()
}

/// Every parent id with an existing `le:{oid}:obj:ch:{parent}` key, found by
/// SCANning the object's keyspace — the oracle, independent of `obj:chs`.
///
/// The family is matched on the literal `obj:ch:` prefix, colon included, so
/// the index key `obj:chs` itself is not mistaken for a list.
async fn listed_parents(tv: &TestValkey, oid: u64) -> BTreeSet<String> {
    let full_prefix = format!("{}{{{oid}}}:obj:ch:", tv.prefix());
    crate::common::scan_object_keys(tv.client(), tv.prefix(), oid)
        .await
        .into_iter()
        .filter_map(|key| {
            key.as_str_lossy()
                .strip_prefix(&full_prefix)
                .map(str::to_string)
        })
        .collect()
}

fn ids(parents: &[u64]) -> Vec<String> {
    parents.iter().map(u64::to_string).collect()
}

/// The SR-1.7 equality, after `arm`: `obj:chs` names exactly the parents that
/// own a children list.
///
/// `must_list` / `must_not_list` are fixture preconditions on the SCAN side
/// only — that the arm really created or removed the lists it is meant to.
/// Without them the equality could pass vacuously (an arm that wrote nothing
/// leaves the two sides as equal as before).
async fn assert_index_names_exactly_the_lists(
    tv: &TestValkey,
    oid: u64,
    arm: &str,
    must_list: &[u64],
    must_not_list: &[u64],
) {
    let listed = listed_parents(tv, oid).await;
    for parent in ids(must_list) {
        assert!(
            listed.contains(&parent),
            "fixture precondition after {arm}: `obj:ch:{parent}` must exist, or \
             this arm did not write the list the assertion below is about. \
             Listed: {listed:?}"
        );
    }
    for parent in ids(must_not_list) {
        assert!(
            !listed.contains(&parent),
            "fixture precondition after {arm}: `obj:ch:{parent}` must be gone, \
             or this arm did not remove the list the assertion below is about. \
             Listed: {listed:?}"
        );
    }

    let indexed = indexed_parents(tv, oid).await;
    assert_eq!(
        indexed,
        listed,
        "after {arm}, on a marked object `obj:chs` must name exactly the parents \
         that own an `obj:ch:*` key (SR-1.7). Indexed but no list: {:?} — a \
         reader trusting the index probes a list that is not there. Listed but \
         not indexed: {:?} — a reader trusting the index never reads that list",
        indexed.difference(&listed).collect::<Vec<_>>(),
        listed.difference(&indexed).collect::<Vec<_>>(),
    );
}

// =============================================================================
// 1. The two `queue_object_state` entry points: marker present, index exact
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_marks_the_object_and_indexes_exactly_its_children_lists() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;

    let oid = create_marked_layer(&ls, &session_id, "chs-create").await;

    assert_eq!(
        marker(&tv, oid).await.as_deref(),
        Some(CHS_COMPLETE_MARKED),
        "CreateLayer materializes through `queue_object_state`, which must stamp \
         `{HEADER_FIELD_CHS_COMPLETE}` = `{CHS_COMPLETE_MARKED}` in the same \
         transaction as the lists it vouches for"
    );
    assert_index_names_exactly_the_lists(
        &tv,
        oid,
        "CreateLayer (`write_object`)",
        &[ROOT, WORLD, DANGLING_PARENT],
        // SIBLING's empty vec writes no key; CUBE and ORPHAN are leaves.
        &[SIBLING, CUBE, ORPHAN],
    )
    .await;
}

/// The restore rebuilds the lists through the engine (the checkpoint's reset
/// delta of `DiffChangeSection`s), which keeps no list under a parent with no
/// row (SDD §10.4). So unlike CreateLayer, which materializes the fixture's
/// hand-built `children[DANGLING_PARENT]` as given, the restored object holds
/// [`ORPHAN`]'s row with no `obj:ch:77`, and the index names neither.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_marks_the_object_and_indexes_exactly_its_children_lists() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;

    let oid = create_marked_layer(&ls, &session_id, "chs-restore").await;
    let (ompf, snap_seq, meta) = build_checkpoint_ompf(&tv, oid).await;
    // A clean keyspace, so the header, every list and the index can come only
    // from the restore — the old marker went with the old header.
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
        "a DR restore materializes through `queue_object_state` \
         (`write_restored_object`), which must stamp the marker exactly as \
         CreateLayer does — a restored object is complete by the same argument"
    );
    assert_index_names_exactly_the_lists(
        &tv,
        oid,
        "DR restore (`write_restored_object`)",
        &[ROOT, WORLD],
        &[CUBE, ORPHAN, DANGLING_PARENT],
    )
    .await;
}

// =============================================================================
// 2. `execute_mutation`'s children arms keep the index exact on a marked object
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn set_children_with_new_lists_keeps_the_index_equal_to_the_lists() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;
    let layer = "chs-set-children";
    let oid = create_marked_layer(&ls, &session_id, layer).await;

    // One new list, under a live leaf (CUBE gains its first child). The same
    // commit's `DiffChangeSection` — the one command that does not validate
    // its parent — makes a row under a second dangling parent, which gets no
    // list: the engine keeps none under a parent with no row (SDD §10.4), so
    // neither the key nor an index member appears for it.
    const NEW_CHILD: u64 = 5;
    const MUTATION_ORPHAN: u64 = 6;
    const SECOND_DANGLING_PARENT: u64 = 88;
    commit_structural(
        &ls,
        &session_id,
        layer,
        oid,
        struct_delta(vec![
            create_cmd(CUBE, NEW_CHILD, "Child"),
            StructCommand::DiffChangeSection(CommandDiffChangeSection {
                parent_id: SECOND_DANGLING_PARENT,
                section_id: MUTATION_ORPHAN,
                section_name: "MutationOrphan".to_string(),
                section_type: 1,
                section_order: 0,
            }),
        ]),
    )
    .await;

    assert_index_names_exactly_the_lists(
        &tv,
        oid,
        "`SetChildren` with a non-empty list",
        &[ROOT, WORLD, DANGLING_PARENT, CUBE],
        &[SECOND_DANGLING_PARENT],
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn set_children_emptying_a_list_keeps_the_index_equal_to_the_lists() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;
    let layer = "chs-set-children-empty";
    let oid = create_marked_layer(&ls, &session_id, layer).await;

    // Moving WORLD's only child to SIBLING empties WORLD's list (the
    // `SetChildren`-with-empty-vec fork: DEL and SREM) and gives SIBLING its
    // first one (the non-empty fork).
    commit_structural(
        &ls,
        &session_id,
        layer,
        oid,
        struct_delta(vec![StructCommand::MoveSection(CommandMoveSection {
            old_parent_id: WORLD,
            new_parent_id: SIBLING,
            section_id: CUBE,
            new_name: "Cube".to_string(),
        })]),
    )
    .await;

    assert_index_names_exactly_the_lists(
        &tv,
        oid,
        "`SetChildren` with an empty list",
        &[ROOT, SIBLING, DANGLING_PARENT],
        &[WORLD],
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_children_list_keeps_the_index_equal_to_the_lists() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;
    let layer = "chs-delete-children";
    let oid = create_marked_layer(&ls, &session_id, layer).await;

    // Deleting WORLD cascades over {WORLD, CUBE}; WORLD's list goes with it
    // through `DeleteChildrenList`, and the root's list is rewritten without
    // WORLD.
    commit_structural(
        &ls,
        &session_id,
        layer,
        oid,
        struct_delta(vec![StructCommand::DeleteSection(CommandDeleteSection {
            parent_id: ROOT,
            section_id: WORLD,
        })]),
    )
    .await;

    assert_index_names_exactly_the_lists(
        &tv,
        oid,
        "`DeleteChildrenList` (section-delete cascade)",
        &[ROOT, DANGLING_PARENT],
        &[WORLD],
    )
    .await;
}

/// A reset erases every list the loaded state holds (one `DeleteChildrenList`
/// per parent) and the same delta then rebuilds a fresh tree (`SetChildren`).
///
/// The dangling parent's list is deliberately named in neither precondition:
/// a reset cannot see it (the snapshot loader never reads `obj:chs`), so today
/// `obj:ch:77` and its index member both survive — a known gap owned by
/// b.evm. Either outcome keeps the two sides equal, which is all this test
/// claims, so a fix to b.evm leaves it green.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reset_keeps_the_index_equal_to_the_lists() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;
    let layer = "chs-reset";
    let oid = create_marked_layer(&ls, &session_id, layer).await;

    const FRESH: u64 = 10;
    const FRESH_CHILD: u64 = 11;
    commit_structural(
        &ls,
        &session_id,
        layer,
        oid,
        OwnedDelta {
            is_diff: true,
            base_version: 0,
            source_format: "usda".to_string(),
            struct_commands: vec![
                create_cmd(ROOT, FRESH, "Fresh"),
                create_cmd(FRESH, FRESH_CHILD, "FreshChild"),
            ],
            ..Default::default()
        },
    )
    .await;

    assert_index_names_exactly_the_lists(
        &tv,
        oid,
        "a reset that rebuilds the tree",
        &[ROOT, FRESH],
        &[WORLD],
    )
    .await;
}

// =============================================================================
// 3. The marker's lifecycle across later commits
// =============================================================================

/// Only `queue_object_state` writes the column and nothing clears it: a
/// structural commit, a reset and a Lua fast-path commit each leave a marked
/// object marked.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn marked_object_keeps_its_marker_through_structural_reset_and_fast_path_commits() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;
    let layer = "chs-marker-kept";
    let oid = create_marked_layer(&ls, &session_id, layer).await;
    assert_eq!(
        marker(&tv, oid).await.as_deref(),
        Some(CHS_COMPLETE_MARKED),
        "fixture precondition: the object must start marked"
    );

    commit_structural(
        &ls,
        &session_id,
        layer,
        oid,
        struct_delta(vec![create_cmd(CUBE, 5, "Child")]),
    )
    .await;
    assert_eq!(
        marker(&tv, oid).await.as_deref(),
        Some(CHS_COMPLETE_MARKED),
        "a structural commit must leave the marker unchanged — its children \
         arms maintain `obj:chs` in their own transaction, so the index stays \
         complete and nothing may clear the column"
    );

    commit_structural(
        &ls,
        &session_id,
        layer,
        oid,
        OwnedDelta {
            is_diff: true,
            base_version: 0,
            source_format: "usda".to_string(),
            struct_commands: vec![create_cmd(ROOT, 10, "Fresh")],
            ..Default::default()
        },
    )
    .await;
    assert_eq!(
        marker(&tv, oid).await.as_deref(),
        Some(CHS_COMPLETE_MARKED),
        "a reset must keep the marker true: it rewrites lists through the same \
         index-maintaining arms, and does not rewrite the header"
    );

    commit_fast_path(&ls, &session_id, layer, oid).await;
    assert_eq!(
        marker(&tv, oid).await.as_deref(),
        Some(CHS_COMPLETE_MARKED),
        "a Lua fast-path commit touches no children key and must leave the \
         marker unchanged"
    );
}

/// No migration: an object materialized before the marker existed stays
/// unmarked through later commits, because nothing but `queue_object_state`
/// writes the column.
///
/// The pre-marker object is today's materialization with the column taken
/// away — the materializer before this change wrote exactly the same header
/// minus `chs_complete`, and no sequence of RPCs can produce that shape any
/// more. Its `obj:chs` happens to be complete; the point is only that a later
/// writer does not promote it, since in general such an object's index may be
/// partial (an object from before the index existed, then structurally
/// edited, holds members for only the lists written since).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unmarked_object_stays_unmarked_through_structural_and_fast_path_commits() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;
    let layer = "chs-pre-marker";
    let oid = create_marked_layer(&ls, &session_id, layer).await;

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
        "arrangement: the column must have existed before this HDEL, or the \
         object was never the shape this test simulates"
    );

    commit_structural(
        &ls,
        &session_id,
        layer,
        oid,
        struct_delta(vec![create_cmd(CUBE, 5, "Child")]),
    )
    .await;
    assert_eq!(
        marker(&tv, oid).await,
        None,
        "a structural commit must not mark a pre-marker object: it writes \
         `obj:chs` members only for the lists it touches, so it cannot vouch \
         for an index it did not build"
    );

    commit_fast_path(&ls, &session_id, layer, oid).await;
    assert_eq!(
        marker(&tv, oid).await,
        None,
        "a Lua fast-path commit must not mark a pre-marker object"
    );
}
