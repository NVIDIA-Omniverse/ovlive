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

//! Integration tests for b.a4k — no children list under a parent with no row
//! (SDD §10.4).
//!
//! A `DiffChangeSection` may name a parent that has no row; the row it writes
//! is an *orphan*, listed nowhere. When the parent is born — a `CreateSection`
//! or a new `DiffChangeSection` row — every row naming it joins its list in
//! `(section_order, section_id)` order before the delta's next command.
//!
//! Every snapshot read keeps only the lists under the root and the live
//! sections, and each commit plans on such a read. A list kept under a
//! missing parent was therefore invisible to every later commit, and once the
//! parent was born the stored list was stale: it named an orphan that had
//! moved away or been deleted, survived a reset, lost a first orphan to a
//! second, or outlived a create-then-delete of its parent. Every reader
//! (catch-up, snapshot, checkpoint) then served a hierarchy whose lists
//! disagreed with its rows.
//!
//! # The parity check
//!
//! Each shape is one of the ticket's deterministic replays, run through
//! `LayerStore::update_layer` as two tests — a module named for the shape's
//! outcome (e.g. `update_layer_create_section_birth_lists_its_orphan`)
//! holding `lua` and `txn`, one per backend, so each fails on its own — with a
//! field-only commit after every structural one (the Lua backend's fast path).
//! After **every** commit, for every parent — the root and every live
//! section, an absent list counting as empty — four answers must be equal:
//!
//! 1. the fresh atomic snapshot (`read_object_snapshot`, catch-up consumer);
//! 2. the structural fan-out (`read_object_snapshot`, structural consumer);
//! 3. the reference: the same deltas applied in order with `apply_delta` to
//!    the created state;
//! 4. the per-parent sibling order `plan_faithful_snapshot` emits from the
//!    atomic read — for the parents the root reaches, since the faithful
//!    snapshot drops the rest.
//!
//! The keyspace is checked too: every `obj:ch:{id}` key (found by SCAN) names
//! the root or a live section, and `obj:chs` names exactly the keys that
//! exist. After the last commit one more structural delta must apply — in a
//! debug build its planner's cross-map check runs over the stored state.
//!
//! The last five tests cover the lists earlier builds stored under a parent while
//! it had no row: a commit creating that parent deletes them, on a marked and
//! on an unmarked object, and a commit creating another id leaves them alone.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use fred::interfaces::{ListInterface, SetsInterface};

use live_edit::config::UpdateLayerBackend;
use live_edit::domain::delta::apply_delta;
use live_edit::domain::diff::{DiffEngine, DiffPlan};
use live_edit::domain::types::{CompactState, ObjectState, Section};
use live_edit::storage::diff_read::{
    read_object_snapshot, read_object_snapshot_slim, snapshot_to_object_state, ObjectSnapshot,
    SlimSnapshotReadConsumer, SnapshotRead, SnapshotReadConsumer, SnapshotReadOf,
};
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::relay::RelayedUpdates;
use live_edit::storage::test_support::TestValkey;
use oodf_ffi::{
    CommandCreateSection, CommandDeleteSection, CommandDiffChangeSection, CommandMoveSection,
    CommandReorderChildren, OwnedDelta, StructCommand,
};

use crate::common::owned_delta::delta_with_set_field_inline;

const WRITER: &str = "b-a4k-writer";
const CREATOR: &str = "b-a4k-user";

/// The live top-level section every replay starts with.
const P: u64 = 2;
/// The parent id the orphans name until it is born.
const X: u64 = 9;

fn mock_ext_store() -> Arc<MockExternalValueStore> {
    Arc::new(MockExternalValueStore::new())
}

async fn create_test_session(tv: &TestValkey) -> String {
    SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create("b-a4k-session", "/stage", CREATOR)
        .await
        .expect("session creation must succeed")
}

fn section(id: u64, parent: u64, name: &str, order: u32) -> Section {
    Section {
        section_id: id,
        parent_id: parent,
        name: name.to_string(),
        section_type: 1,
        section_order: order,
        first_version: 1,
        last_version: 1,
    }
}

/// Root → `[P]`, plus `extra` rows (each listed under its parent, which must
/// be the root or `P`).
fn start_state(extra: &[Section]) -> ObjectState {
    let mut state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usda".to_string(),
        ..Default::default()
    };
    state.sections.insert(P, section(P, 0, "P", 0));
    state.children.insert(0, vec![P]);
    for s in extra {
        state.sections.insert(s.section_id, s.clone());
        state
            .children
            .entry(s.parent_id)
            .or_default()
            .push(s.section_id);
    }
    state
}

// ── Command builders ─────────────────────────────────────────────────────────

fn create(parent_id: u64, section_id: u64, name: &str) -> StructCommand {
    StructCommand::CreateSection(CommandCreateSection {
        parent_id,
        section_id,
        section_name: name.to_string(),
        section_type: 1,
    })
}

fn change(parent_id: u64, section_id: u64, name: &str, section_order: u32) -> StructCommand {
    StructCommand::DiffChangeSection(CommandDiffChangeSection {
        parent_id,
        section_id,
        section_name: name.to_string(),
        section_type: 1,
        section_order,
    })
}

fn delete(section_id: u64) -> StructCommand {
    StructCommand::DeleteSection(CommandDeleteSection {
        parent_id: 0,
        section_id,
    })
}

fn move_to(section_id: u64, old_parent_id: u64, new_parent_id: u64, name: &str) -> StructCommand {
    StructCommand::MoveSection(CommandMoveSection {
        old_parent_id,
        new_parent_id,
        section_id,
        new_name: name.to_string(),
    })
}

fn reorder(parent_id: u64, children: Vec<u64>) -> StructCommand {
    StructCommand::ReorderChildren(CommandReorderChildren {
        section_id: parent_id,
        children_list_id: 0,
        children_list: children,
    })
}

fn structural(cmds: Vec<StructCommand>) -> OwnedDelta {
    OwnedDelta {
        is_diff: false,
        struct_commands: cmds,
        ..Default::default()
    }
}

fn reset() -> OwnedDelta {
    OwnedDelta {
        is_diff: true,
        base_version: 0,
        source_format: "usda".to_string(),
        ..Default::default()
    }
}

// ── Readers ──────────────────────────────────────────────────────────────────

async fn read_as(tv: &TestValkey, oid: u64, consumer: SnapshotReadConsumer) -> ObjectSnapshot {
    match read_object_snapshot(tv.client(), tv.key_schema(), &oid.to_string(), consumer)
        .await
        .expect("snapshot read must succeed")
    {
        SnapshotRead::Snapshot(snapshot) => snapshot.state,
        SnapshotRead::TooLarge(answer) => panic!("a test layer is never too large: {answer:?}"),
    }
}

/// The structural WATCH path's read of `oid`: the unfenced fan-out decoded
/// into the compact form.
async fn read_structural_fanout(tv: &TestValkey, oid: u64) -> CompactState {
    match read_object_snapshot_slim(
        tv.client(),
        tv.key_schema(),
        &oid.to_string(),
        SlimSnapshotReadConsumer::StructuralUpdate,
    )
    .await
    .expect("snapshot read must succeed")
    {
        SnapshotReadOf::Snapshot(snapshot) => snapshot.state,
        SnapshotReadOf::TooLarge(answer) => {
            panic!("the fan-out is never too large: {answer:?}")
        }
    }
}

/// Every parent id with an existing `obj:ch:{id}` key, by SCAN — independent
/// of `obj:chs` and of every reader.
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

async fn indexed_parents(tv: &TestValkey, oid: u64) -> BTreeSet<String> {
    let members: Vec<String> = tv
        .client()
        .pool
        .smembers(tv.key_schema().obj_children_index(&oid.to_string()))
        .await
        .expect("SMEMBERS obj:chs must succeed");
    members.into_iter().collect()
}

async fn stored_list(tv: &TestValkey, oid: u64, parent: u64) -> Vec<String> {
    tv.client()
        .pool
        .lrange(
            tv.key_schema()
                .obj_children(&oid.to_string(), &parent.to_string()),
            0,
            -1,
        )
        .await
        .expect("LRANGE obj:ch must succeed")
}

/// The list under each of `parents`, an absent list counting as empty.
fn lists_of(
    children: &std::collections::HashMap<u64, Vec<u64>>,
    parents: &BTreeSet<u64>,
) -> BTreeMap<u64, Vec<u64>> {
    parents
        .iter()
        .map(|p| (*p, children.get(p).cloned().unwrap_or_default()))
        .collect()
}

/// The per-parent sibling order `plan_faithful_snapshot` emits: each emitted
/// section under its parent, by the emitted sibling index.
fn faithful_lists(state: &ObjectState) -> BTreeMap<u64, Vec<u64>> {
    let DiffPlan::FullContent { sections, .. } =
        DiffEngine::plan_faithful_snapshot(state).expect("the faithful snapshot must plan")
    else {
        panic!("the faithful snapshot plans FullContent");
    };
    let mut by_parent: BTreeMap<u64, Vec<(u32, u64)>> = BTreeMap::new();
    for s in &sections {
        by_parent
            .entry(s.parent_id)
            .or_default()
            .push((s.section_order, s.section_id));
    }
    by_parent
        .into_iter()
        .map(|(p, mut rows)| {
            rows.sort_unstable();
            (p, rows.into_iter().map(|(_, id)| id).collect())
        })
        .collect()
}

/// The root and every live section the root reaches through the lists.
fn reachable_parents(state: &ObjectState) -> BTreeSet<u64> {
    let mut seen = BTreeSet::new();
    let mut stack = vec![0];
    while let Some(p) = stack.pop() {
        if !seen.insert(p) {
            continue;
        }
        if let Some(children) = state.children.get(&p) {
            stack.extend(children.iter().filter(|c| state.sections.contains_key(*c)));
        }
    }
    seen
}

// ── The replay harness ───────────────────────────────────────────────────────

/// One object under replay: the store it commits through and the reference
/// state the same deltas are applied to in memory.
struct Replay<'a> {
    tv: &'a TestValkey,
    ls: LayerStore,
    session_id: String,
    layer: String,
    oid: u64,
    reference: ObjectState,
    label: String,
    commits: usize,
}

impl<'a> Replay<'a> {
    async fn start(
        tv: &'a TestValkey,
        session_id: &str,
        backend: UpdateLayerBackend,
        shape: &str,
        start: ObjectState,
    ) -> Self {
        let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
            .with_update_layer_backend(backend);
        let layer = format!("{shape}-{backend:?}");
        let oid = ls
            .create_layer(session_id, &layer, CREATOR, &start, mock_ext_store())
            .await
            .expect("create_layer must succeed");
        let replay = Self {
            tv,
            ls,
            session_id: session_id.to_string(),
            layer,
            oid,
            reference: start,
            label: format!("{shape} on the {backend:?} backend"),
            commits: 0,
        };
        replay.check("after create_layer").await;
        replay
    }

    /// Commit `delta` through `update_layer` and apply it to the reference,
    /// then check the parity.
    async fn commit(&mut self, delta: OwnedDelta) {
        self.commits += 1;
        let at = format!(
            "after commit {} ({:?})",
            self.commits, delta.struct_commands
        );
        self.ls
            .update_layer_relayed(
                &self.session_id,
                &self.layer,
                self.oid,
                Some(delta.clone()),
                WRITER,
                0,
            )
            .await
            .unwrap_or_else(|e| panic!("{}: update_layer {at} failed: {e:?}", self.label));
        apply_delta(&mut self.reference, Some(delta))
            .unwrap_or_else(|e| panic!("{}: reference apply {at} failed: {e:?}", self.label));
        self.check(&at).await;
    }

    /// A structural commit followed by a field-only one on `P` — the Lua
    /// backend's fast path.
    async fn commit_with_field(&mut self, delta: OwnedDelta) {
        self.commit(delta).await;
        let n = self.commits as u8;
        self.commit(delta_with_set_field_inline(P, "f", vec![n], 0))
            .await;
    }

    /// One more structural delta after the replay: its planner reads the
    /// stored state, and in a debug build checks it.
    async fn finish(mut self) {
        self.commit(structural(vec![create(0, 50, "tail")])).await;
    }

    async fn check(&self, at: &str) {
        let label = &self.label;
        let atomic = read_as(self.tv, self.oid, SnapshotReadConsumer::Catchup).await;
        let fanout = read_structural_fanout(self.tv, self.oid).await;

        let live: BTreeSet<u64> = self.reference.sections.keys().copied().collect();
        assert_eq!(
            atomic.sections.keys().copied().collect::<BTreeSet<u64>>(),
            live,
            "{label} {at}: the stored rows are the reference's"
        );
        let parents: BTreeSet<u64> = std::iter::once(0).chain(live.iter().copied()).collect();

        let reference = lists_of(&self.reference.children, &parents);
        assert_eq!(
            lists_of(&atomic.children, &parents),
            reference,
            "{label} {at}: the fresh atomic snapshot's lists must be the reference's"
        );
        assert_eq!(
            lists_of(&fanout.children, &parents),
            reference,
            "{label} {at}: the structural fan-out's lists must be the reference's"
        );

        let read_state = snapshot_to_object_state(&atomic);
        let faithful = faithful_lists(&read_state);
        for parent in reachable_parents(&read_state) {
            let emitted = faithful.get(&parent).cloned().unwrap_or_default();
            let expected: Vec<u64> = reference[&parent]
                .iter()
                .copied()
                .filter(|c| live.contains(c))
                .collect();
            assert_eq!(
                emitted, expected,
                "{label} {at}: the faithful snapshot's sibling order under {parent} must \
                 be the reference's"
            );
        }

        let listed = listed_parents(self.tv, self.oid).await;
        for parent in &listed {
            let id: u64 = parent.parse().expect("a children key names a decimal id");
            assert!(
                parents.contains(&id),
                "{label} {at}: `obj:ch:{id}` exists under an id that is neither the root nor \
                 a live section. Listed: {listed:?}"
            );
        }
        assert_eq!(
            indexed_parents(self.tv, self.oid).await,
            listed,
            "{label} {at}: `obj:chs` must name exactly the children keys that exist"
        );
    }
}

/// One replayed shape: the created state and the structural deltas that
/// follow it.
struct Shape {
    start: ObjectState,
    deltas: Vec<OwnedDelta>,
}

/// Replay `shape` on `backend`, checking the parity after every commit.
async fn replay(name: &str, backend: UpdateLayerBackend, shape: Shape) {
    let tv = TestValkey::setup().await;
    let session_id = create_test_session(&tv).await;
    let mut replay = Replay::start(&tv, &session_id, backend, name, shape.start).await;
    for delta in shape.deltas {
        replay.commit_with_field(delta).await;
    }
    replay.finish().await;
}

/// One module per shape, holding one test per backend — `lua` and `txn` — so
/// every shape on every backend passes or fails on its own. The module is
/// named for the shape's outcome; the shape itself is the function given.
macro_rules! replay_tests {
    ($($test:ident => $shape:ident;)*) => {
        $(
            mod $test {
                use super::*;

                #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
                async fn lua() {
                    replay(stringify!($shape), UpdateLayerBackend::Lua, $shape()).await;
                }

                #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
                async fn txn() {
                    replay(stringify!($shape), UpdateLayerBackend::Txn, $shape()).await;
                }
            }
        )*
    };
}

// =============================================================================
// 1. Adoption
// =============================================================================

/// An orphan under X, then X is born by `birth`, then a child is created
/// under X: the stored list holds the orphan, then the child.
fn adoption(birth: StructCommand) -> Shape {
    Shape {
        start: start_state(&[]),
        deltas: vec![
            structural(vec![change(X, 1, "o", 0)]),
            structural(vec![birth]),
            structural(vec![create(X, 3, "n")]),
        ],
    }
}

/// X born by `CreateSection`.
fn adopt_by_create() -> Shape {
    adoption(create(0, X, "X"))
}

/// X born by a new `DiffChangeSection` row.
fn adopt_by_change() -> Shape {
    adoption(change(0, X, "X", 1))
}

/// The birth and a command building the new list in one delta — the shapes
/// b461447a left as "the known remaining case". Two orphans, keyed out of id
/// order, so the adopted list's order is visible.
fn same_delta(follow: StructCommand) -> Shape {
    Shape {
        start: start_state(&[section(5, P, "m", 0)]),
        deltas: vec![
            structural(vec![change(X, 1, "o1", 2), change(X, 4, "o4", 0)]),
            structural(vec![create(0, X, "X"), follow]),
        ],
    }
}

/// A `CreateSection` under X after the birth.
fn same_delta_create() -> Shape {
    same_delta(create(X, 3, "n"))
}

/// A `DiffChangeSection` under X after the birth.
fn same_delta_change() -> Shape {
    same_delta(change(X, 3, "n", 1))
}

/// A `MoveSection` from P into X after the birth.
fn same_delta_move() -> Shape {
    same_delta(move_to(5, P, X, "m"))
}

/// A `ReorderChildren` of X after the birth.
fn same_delta_reorder() -> Shape {
    same_delta(reorder(X, vec![1, 4]))
}

replay_tests! {
    update_layer_create_section_birth_lists_its_orphan => adopt_by_create;
    update_layer_diff_change_section_birth_lists_its_orphan => adopt_by_change;
    update_layer_birth_then_create_under_it_in_one_delta_lists_both => same_delta_create;
    update_layer_birth_then_diff_change_under_it_in_one_delta_lists_both => same_delta_change;
    update_layer_birth_then_move_into_it_in_one_delta_lists_both => same_delta_move;
    update_layer_birth_then_reorder_of_it_in_one_delta_orders_the_adopted_rows => same_delta_reorder;
}

// =============================================================================
// 2. The orphan leaves before its parent is born
// =============================================================================

/// The orphan leaves by `away`, then X is born and given a child. X's list
/// must not name the orphan.
fn orphan_leaves(away: OwnedDelta) -> Shape {
    Shape {
        start: start_state(&[]),
        deltas: vec![
            structural(vec![change(X, 1, "o", 0)]),
            away,
            structural(vec![create(0, X, "X")]),
            structural(vec![create(X, 3, "n")]),
        ],
    }
}

/// The orphan is moved under P by `MoveSection`.
fn moved_away_by_move() -> Shape {
    orphan_leaves(structural(vec![move_to(1, X, P, "o")]))
}

/// The orphan is moved under P by a reparenting `DiffChangeSection`.
fn moved_away_by_change() -> Shape {
    orphan_leaves(structural(vec![change(P, 1, "o", 0)]))
}

/// The orphan is deleted.
fn orphan_deleted() -> Shape {
    orphan_leaves(structural(vec![delete(1)]))
}

/// A reset erases the orphan.
fn orphan_reset() -> Shape {
    orphan_leaves(reset())
}

/// Two orphans under X from two commits, then X is born. X's list must hold
/// both, in key order.
fn second_orphan() -> Shape {
    Shape {
        start: start_state(&[]),
        deltas: vec![
            structural(vec![change(X, 1, "o1", 0)]),
            structural(vec![change(X, 4, "o4", 1)]),
            structural(vec![create(0, X, "X")]),
            structural(vec![create(X, 3, "n")]),
        ],
    }
}

replay_tests! {
    update_layer_orphan_moved_away_by_move_is_not_listed_under_its_old_parent => moved_away_by_move;
    update_layer_orphan_moved_away_by_change_is_not_listed_under_its_old_parent => moved_away_by_change;
    update_layer_deleted_orphan_is_not_listed_under_its_parent => orphan_deleted;
    update_layer_orphan_erased_by_a_reset_is_not_listed_under_its_parent => orphan_reset;
    update_layer_second_orphan_does_not_displace_the_first => second_orphan;
}

// =============================================================================
// 3. The parent is born and deleted in one delta
// =============================================================================

/// The ticket's replay with ids 1 and 9: orphan 1 under the missing 9; one
/// delta creates 9 and deletes it (taking the adopted 1 along); a later delta
/// creates 9 again; one more structural delta follows.
fn create_then_delete() -> Shape {
    Shape {
        start: start_state(&[]),
        deltas: vec![
            structural(vec![change(9, 1, "o", 0)]),
            structural(vec![create(0, 9, "X"), delete(9)]),
            structural(vec![create(0, 9, "X")]),
            structural(vec![create(9, 3, "n")]),
        ],
    }
}

/// The same through a cascade from an ancestor of the just-created parent:
/// one delta creates A, creates X under A, and deletes A.
fn ancestor_cascade() -> Shape {
    const A: u64 = 6;
    Shape {
        start: start_state(&[]),
        deltas: vec![
            structural(vec![change(X, 1, "o", 0)]),
            structural(vec![create(0, A, "A"), create(A, X, "X"), delete(A)]),
            structural(vec![create(0, X, "X")]),
            structural(vec![create(X, 3, "n")]),
        ],
    }
}

replay_tests! {
    update_layer_create_then_delete_of_the_parent_takes_its_orphan_along => create_then_delete;
    update_layer_cascade_from_an_ancestor_takes_the_adopted_orphan_along => ancestor_cascade;
}

// =============================================================================
// 4. Lists earlier builds stored under a parent with no row
// =============================================================================

async fn commit_structural(
    ls: &LayerStore,
    session_id: &str,
    layer: &str,
    oid: u64,
    cmds: Vec<StructCommand>,
) {
    ls.update_layer_relayed(session_id, layer, oid, Some(structural(cmds)), WRITER, 0)
        .await
        .expect("the structural commit must succeed");
}

/// Inject `obj:ch:{X}` = `list` and the `obj:chs` member `X` — what an
/// earlier build's commit stored for a row under X while X had no row.
async fn inject_earlier_build_list(tv: &TestValkey, oid: u64, list: &[u64]) {
    let oid_str = oid.to_string();
    let pushed: i64 = tv
        .client()
        .pool
        .rpush(
            tv.key_schema().obj_children(&oid_str, &X.to_string()),
            list.iter().map(u64::to_string).collect::<Vec<String>>(),
        )
        .await
        .expect("RPUSH obj:ch must succeed");
    assert_eq!(
        pushed,
        list.len() as i64,
        "arrangement: no list existed under X before the injection"
    );
    let _: i64 = tv
        .client()
        .pool
        .sadd(tv.key_schema().obj_children_index(&oid_str), X.to_string())
        .await
        .expect("SADD obj:chs must succeed");
}

/// A commit creating X deletes the list an earlier build stored under X:
/// afterwards `obj:ch:{X}` is absent (nothing to adopt) or exactly the
/// adopted list (orphan 1 adopted), and `obj:chs` names exactly the keys that
/// exist — on a marked object, where the read reports the left-out list, and
/// on an unmarked one, where it cannot and the commit deletes unconditionally.
///
/// The injected lists are stale on purpose: they name 77, an id with no row,
/// so a list left in place shows up as a wrong list rather than as a
/// coincidentally right one.
///
/// One test per (marked, with an adopted orphan) case.
async fn assert_creating_x_deletes_the_earlier_builds_list(marked: bool, with_orphan: bool) {
    const GONE: u64 = 77;

    let tv = TestValkey::setup().await;
    let session_id = create_test_session(&tv).await;
    let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);

    let label = format!("marked={marked} with_orphan={with_orphan}");
    let layer = "earlier-build";
    let oid = ls
        .create_layer(
            &session_id,
            layer,
            CREATOR,
            &start_state(&[]),
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");
    if !marked {
        crate::storage::snapshot_read_atomic_test::unmark(&tv, oid).await;
    }
    let injected: Vec<u64> = if with_orphan {
        commit_structural(&ls, &session_id, layer, oid, vec![change(X, 1, "o", 0)]).await;
        vec![GONE, 1]
    } else {
        vec![GONE]
    };
    inject_earlier_build_list(&tv, oid, &injected).await;

    commit_structural(&ls, &session_id, layer, oid, vec![create(0, X, "X")]).await;

    let expected: Vec<String> = if with_orphan {
        vec!["1".to_string()]
    } else {
        Vec::new()
    };
    assert_eq!(
        stored_list(&tv, oid, X).await,
        expected,
        "{label}: X's stored list is exactly what the engine built — the injected \
         list is gone"
    );
    let listed = listed_parents(&tv, oid).await;
    assert_eq!(
        indexed_parents(&tv, oid).await,
        listed,
        "{label}: `obj:chs` names exactly the keys that exist"
    );
    assert_eq!(
        listed.contains(&X.to_string()),
        with_orphan,
        "{label}: X is listed exactly when it has a child"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_creating_a_parent_on_a_marked_object_deletes_the_list_an_earlier_build_stored(
) {
    assert_creating_x_deletes_the_earlier_builds_list(true, false).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_creating_a_parent_with_an_orphan_on_a_marked_object_replaces_the_list_an_earlier_build_stored(
) {
    assert_creating_x_deletes_the_earlier_builds_list(true, true).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_creating_a_parent_on_an_unmarked_object_deletes_the_list_an_earlier_build_stored(
) {
    assert_creating_x_deletes_the_earlier_builds_list(false, false).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_creating_a_parent_with_an_orphan_on_an_unmarked_object_replaces_the_list_an_earlier_build_stored(
) {
    assert_creating_x_deletes_the_earlier_builds_list(false, true).await;
}

/// On a marked object, a commit creating an id other than X leaves the list
/// an earlier build stored under X — and its `obj:chs` member — alone: the
/// deletion touches only ids the commit creates.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_creating_another_id_leaves_an_earlier_builds_list_alone() {
    const GONE: u64 = 77;
    const OTHER: u64 = 10;

    let tv = TestValkey::setup().await;
    let session_id = create_test_session(&tv).await;
    let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
    let layer = "earlier-build-other-id";
    let oid = ls
        .create_layer(
            &session_id,
            layer,
            CREATOR,
            &start_state(&[]),
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");
    inject_earlier_build_list(&tv, oid, &[GONE]).await;

    commit_structural(
        &ls,
        &session_id,
        layer,
        oid,
        vec![create(0, OTHER, "other")],
    )
    .await;

    assert_eq!(
        stored_list(&tv, oid, X).await,
        vec![GONE.to_string()],
        "the list under X is untouched"
    );
    assert!(
        indexed_parents(&tv, oid).await.contains(&X.to_string()),
        "and so is its `obj:chs` member"
    );
}
