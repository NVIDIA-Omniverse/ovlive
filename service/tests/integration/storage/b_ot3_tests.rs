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

//! Integration tests for b.ot3 — bounded structural delta allocation and CPU
//! work (SDD §7.7).
//!
//! The domain-side counters and orderings are pinned inline in
//! `src/domain/delta.rs`. This file answers the question those cannot: does the
//! *persisted* keyspace still agree, now that N per-command `SetChildren`s have
//! become one write per dirty parent at the close of the structural phase?
//!
//! # The oracle
//!
//! Most tests here drive the same command sequence twice against two layers in
//! one session — once as a single `update_layer` (the batched delta) and once
//! as one `update_layer` per command (the reference) — and compare the whole
//! persisted surface: `obj:sec:*`, the `obj:ch:*` lists **in order**, the
//! `obj:chs` index, `obj:fld:*`, `obj:ts:*`, `obj:del:*` and `obj:ext`. Every
//! one of those is a column of [`PersistedShape`], so a test added later
//! inherits the whole comparison rather than the subset its author remembered.
//!
//! One command per `update_layer` is one command per `ChildPlan`: every
//! per-delta cache is rebuilt from a freshly loaded snapshot, which is by
//! construction what the pre-b.ot3 engine did on every command. So the
//! reference layer is the pre-fix answer for everything except the coalescing
//! itself, and any divergence is a coalescing or caching defect rather than a
//! deliberate semantic change.
//!
//! Version columns are excluded from the comparison: N commits advance
//! `sequence_num` N times where one commit advances it once, and
//! `first_ver` / `last_ver` / `set_order` follow. Everything else is compared
//! literally.
//!
//! # What only this tier can see
//!
//! Two of the tests below pin a *key* that must not survive, rather than a
//! value that differs:
//!
//! - [`reset_deletes_the_root_children_key_and_drops_it_from_the_index`] — the
//!   synthetic root `0` is never a section, so `apply_reset`'s per-section loop
//!   could not emit its `DeleteChildrenList`. Pre-b.ot3 a reset left
//!   `le:{oid}:obj:ch:0` populated with dead ids and `"0"` in `obj:chs`, which
//!   the next snapshot load read straight back into `state.children[0]`.
//! - [`child_under_a_ghost_parent_cascaded_in_one_delta_leaves_no_key_and_survives_unlisted`]
//!   — a row created under a ghost id (listed, no row of its own) inside the
//!   delta that cascades the ghost's lister away. Pre-b.ot3 the inline
//!   `SetChildren` wrote the ghost's key and the cascade's ghost arm never
//!   deleted it; b.ot3's finalization pass deleted it in the batched delta
//!   only. Since b.a4k no list is ever kept under an id with no row (SDD
//!   §10.4), so no commit writes the key and the row survives as an orphan
//!   the cascade cannot reach.
//!
//! The reset test guards a leak on a `noeviction` store, of the same family as
//! b.9fn / b.jdr (whose own coverage lives in `b_9fn_tests.rs`) — bounded
//! there by the `obj:chs`-driven retirement inventory, but live-object state
//! corruption until then. The ghost test guards the no-list rule: no key is
//! ever written under the ghost, so there is nothing to leak.

use std::sync::Arc;

use fred::interfaces::{KeysInterface, ListInterface, SetsInterface};

use live_edit::domain::types::{ObjectState, Section};
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::read_object_state_for_test;
use live_edit::storage::test_support::relay::RelayedUpdates;
use live_edit::storage::test_support::TestValkey;
use oodf_ffi::{
    CommandCreateSection, CommandDeleteSection, CommandDiffChangeSection, CommandMoveSection,
    CommandReorderChildren, OwnedDelta, SetField, StructCommand, TimeSample,
};

use crate::storage::b_qpf_tests::{ext_hash, ext_hash_hex, ext_store, obj_ext_sorted, pre_store};

const WRITER: &str = "b-ot3-writer";

fn mock_ext_store() -> Arc<MockExternalValueStore> {
    Arc::new(MockExternalValueStore::new())
}

fn layer_store(tv: &TestValkey) -> LayerStore {
    LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
}

async fn create_test_session(tv: &TestValkey) -> String {
    SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create("b-ot3-session", "/stage", "b-ot3-user")
        .await
        .expect("session creation must succeed")
}

// ── Command builders ─────────────────────────────────────────────────────────

fn create_cmd(parent_id: u64, section_id: u64, name: &str) -> StructCommand {
    StructCommand::CreateSection(CommandCreateSection {
        parent_id,
        section_id,
        section_name: name.to_string(),
        section_type: 1,
    })
}

fn change_cmd(parent_id: u64, section_id: u64, name: &str, section_order: u32) -> StructCommand {
    StructCommand::DiffChangeSection(CommandDiffChangeSection {
        parent_id,
        section_id,
        section_name: name.to_string(),
        section_type: 1,
        section_order,
    })
}

fn delete_cmd(section_id: u64) -> StructCommand {
    StructCommand::DeleteSection(CommandDeleteSection {
        parent_id: 0,
        section_id,
    })
}

fn move_cmd(
    section_id: u64,
    old_parent_id: u64,
    new_parent_id: u64,
    new_name: &str,
) -> StructCommand {
    StructCommand::MoveSection(CommandMoveSection {
        old_parent_id,
        new_parent_id,
        section_id,
        new_name: new_name.to_string(),
    })
}

fn reorder_cmd(parent_id: u64, children: Vec<u64>) -> StructCommand {
    StructCommand::ReorderChildren(CommandReorderChildren {
        section_id: parent_id,
        children_list_id: 0,
        children_list: children,
    })
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

// ── Commit drivers ───────────────────────────────────────────────────────────

/// Commit every command in ONE `update_layer` — the delta shape b.ot3 coalesces.
async fn commit_batched(
    ls: &LayerStore,
    session_id: &str,
    layer: &str,
    oid: u64,
    cmds: Vec<StructCommand>,
) {
    ls.update_layer_relayed(
        session_id,
        layer,
        oid,
        Some(OwnedDelta {
            is_diff: false,
            struct_commands: cmds,
            ..Default::default()
        }),
        WRITER,
        0,
    )
    .await
    .expect("batched update_layer must succeed");
}

/// Commit each command as its own `update_layer` — the reference. Each commit
/// reloads the snapshot, so every per-delta cache is derived fresh: the
/// pre-b.ot3 per-command behaviour, expressed through the shipped engine.
async fn commit_one_per_delta(
    ls: &LayerStore,
    session_id: &str,
    layer: &str,
    oid: u64,
    cmds: Vec<StructCommand>,
) {
    for cmd in cmds {
        ls.update_layer_relayed(
            session_id,
            layer,
            oid,
            Some(OwnedDelta {
                is_diff: false,
                struct_commands: vec![cmd],
                ..Default::default()
            }),
            WRITER,
            0,
        )
        .await
        .expect("single-command update_layer must succeed");
    }
}

// ── Persisted-surface readers ────────────────────────────────────────────────

/// Sorted `SMEMBERS le:{oid}:obj:chs`.
async fn children_index(tv: &TestValkey, oid: u64) -> Vec<String> {
    let mut members: Vec<String> = tv
        .client()
        .pool
        .smembers(tv.key_schema().obj_children_index(&oid.to_string()))
        .await
        .expect("SMEMBERS obj:chs must succeed");
    members.sort();
    members
}

async fn children_key_exists(tv: &TestValkey, oid: u64, parent_id: u64) -> bool {
    tv.client()
        .pool
        .exists(
            tv.key_schema()
                .obj_children(&oid.to_string(), &parent_id.to_string()),
        )
        .await
        .expect("EXISTS obj:ch:{parent} must succeed")
}

/// `LRANGE le:{oid}:obj:ch:{parent} 0 -1`.
async fn children_list(tv: &TestValkey, oid: u64, parent_id: u64) -> Vec<String> {
    tv.client()
        .pool
        .lrange(
            tv.key_schema()
                .obj_children(&oid.to_string(), &parent_id.to_string()),
            0,
            -1,
        )
        .await
        .expect("LRANGE obj:ch:{parent} must succeed")
}

/// The whole version-free persisted surface of an object.
///
/// Children are read through `obj:chs` plus the root rather than through the
/// snapshot loader: the loader only `LRANGE`s parents that are live sections
/// (plus the root), so a children key left behind under a dangling or ghost id
/// is invisible to it — and an orphaned key is exactly what two of these tests
/// exist to catch.
///
/// `ext_refs` is part of the shape rather than a per-test hand-rolled read so
/// that a future equivalence test carrying external values inherits the
/// reference-count comparison instead of silently getting a weaker one; it is
/// empty (and equal on both sides) for every test that uses no external values.
#[derive(Debug, PartialEq, Eq)]
struct PersistedShape {
    sections: Vec<(u64, u64, String, u8, u32)>,
    children_index: Vec<String>,
    children: Vec<(String, Vec<String>)>,
    field_keys: Vec<(u64, String)>,
    time_sample_keys: Vec<(u64, String)>,
    tombstone_ids: Vec<u64>,
    /// `obj:ext` as sorted `(hash_hex, ref_count)` rows.
    ext_refs: Vec<(String, i64)>,
}

async fn persisted_shape(tv: &TestValkey, oid: u64) -> PersistedShape {
    let snapshot = read_object_state_for_test(tv.client(), tv.key_schema(), &oid.to_string())
        .await
        .expect("snapshot load must succeed");

    let mut sections: Vec<(u64, u64, String, u8, u32)> = snapshot
        .sections
        .values()
        .map(|s| {
            (
                s.section_id,
                s.parent_id,
                s.name.clone(),
                s.section_type,
                s.section_order,
            )
        })
        .collect();
    sections.sort();

    let index = children_index(tv, oid).await;
    // The root is read unconditionally: it is a member of no index, which is
    // the whole point of b.9fn, and its presence/absence is load-bearing here.
    let mut parents: Vec<String> = index.clone();
    if !parents.iter().any(|p| p == "0") {
        parents.push("0".to_string());
    }
    parents.sort();
    let mut children = Vec::with_capacity(parents.len());
    for parent in &parents {
        let list: Vec<String> = tv
            .client()
            .pool
            .lrange(
                tv.key_schema().obj_children(&oid.to_string(), parent),
                0,
                -1,
            )
            .await
            .expect("LRANGE obj:ch must succeed");
        children.push((parent.clone(), list));
    }

    let mut field_keys: Vec<(u64, String)> = snapshot
        .fields
        .iter()
        .map(|f| (f.section_id, f.key_name.clone()))
        .collect();
    field_keys.sort();

    // Times are compared as their persisted spelling rather than as floats —
    // an epsilon comparison has no meaning for a map KEY.
    let mut time_sample_keys: Vec<(u64, String)> = snapshot
        .time_samples
        .iter()
        .map(|t| (t.section_id, t.time.0.to_string()))
        .collect();
    time_sample_keys.sort();

    let mut tombstone_ids: Vec<u64> = snapshot.tombstones.iter().map(|(sid, _, _)| *sid).collect();
    tombstone_ids.sort();

    PersistedShape {
        sections,
        children_index: index,
        children,
        field_keys,
        time_sample_keys,
        tombstone_ids,
        ext_refs: obj_ext_sorted(tv, oid).await,
    }
}

// =============================================================================
// 1. Reset — the synthetic root's children key
// =============================================================================

/// A reset delta over a layer with top-level sections must leave
/// `le:{oid}:obj:ch:0` **absent** and `"0"` out of `obj:chs`.
///
/// `apply_reset` erases `state.children` wholesale but its per-section loop can
/// only emit a `DeleteChildrenList` for ids that had a `state.sections` row.
/// The synthetic root never does. Pre-b.ot3 the key therefore survived every
/// reset carrying the ids of sections the same delta had just tombstoned, and
/// the next snapshot read read them straight back into
/// `state.children[0]` — tripping the cross-map debug assert and re-persisting
/// ghosts in release builds.
///
/// The fix folds the reset into the finalization pass: every children-list key
/// is registered as dirty before the clear, so the root — and any other
/// non-section parent **present in the loaded state** — gets its delete from
/// the finalizer.
///
/// # Scope boundary: a pre-existing DANGLING parent is not reachable from here
///
/// The fix reaches exactly as far as `state.children` does, and the snapshot
/// reads take their parent set from the root plus the live sections
/// (`SMEMBERS obj:secs`) — on a marked object only the lists `obj:chs` names
/// under such a parent (`src/storage/diff_read.rs`'s fan-out,
/// `object_snapshot_read.lua`). A children key written under a parent id that
/// names no section — the tolerated dangling-parent state of SDD §10.4 — is
/// therefore
/// never `LRANGE`d back, so it is never in `state.children`, so `apply_reset`'s
/// registration loop cannot mark it dirty and the finalizer never sees it.
/// Measured, not inferred: with `children[77] = [orphan]` in the created
/// object, `obj:ch:77` still exists and `"77"` is still in `obj:chs` after the
/// reset above.
///
/// That is a pre-existing gap, not a b.ot3 regression (pre-fix the root leaked
/// the same way), and it is bounded at object retirement by b.jdr's
/// `obj:chs`-driven UNLINK inventory — `b_9fn_tests.rs` covers that. It is left
/// unasserted here rather than pinned, because pinning it would make a future
/// fix red; it is owned by follow-up issue **b.evm** ("Reset leaves
/// dangling-parent children keys and obj:chs members in place (sdd.md §7.7)").
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reset_deletes_the_root_children_key_and_drops_it_from_the_index() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;

    let mut state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usda".to_string(),
        ..Default::default()
    };
    state.sections.insert(1, section(1, 0, "World", 0));
    state.sections.insert(2, section(2, 0, "Sibling", 1));
    state.sections.insert(3, section(3, 1, "Cube", 0));
    state.children.insert(0, vec![1, 2]);
    state.children.insert(1, vec![3]);

    let oid = ls
        .create_layer(
            &session_id,
            "reset-layer",
            "b-ot3",
            &state,
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");

    // Precondition: without these the post-reset assertions are vacuous.
    assert!(
        children_key_exists(&tv, oid, 0).await,
        "fixture precondition: the root children key must exist before the reset"
    );
    assert!(
        children_index(&tv, oid).await.contains(&"0".to_string()),
        "fixture precondition: the root must be indexed before the reset"
    );

    ls.update_layer_relayed(
        &session_id,
        "reset-layer",
        oid,
        Some(OwnedDelta {
            is_diff: true,
            base_version: 0,
            source_format: "usda".to_string(),
            ..Default::default()
        }),
        WRITER,
        0,
    )
    .await
    .expect("reset update_layer must succeed");

    assert!(
        !children_key_exists(&tv, oid, 0).await,
        "the reset must delete the root children key, not leave it holding dead ids"
    );
    assert!(
        !children_index(&tv, oid).await.contains(&"0".to_string()),
        "the root must leave obj:chs with its key (SDD §7.7: member ⟺ key exists)"
    );
    // The per-section arm still works, so the whole children surface is gone.
    assert!(
        !children_key_exists(&tv, oid, 1).await,
        "a live section's children key is deleted by the reset's own loop"
    );
    assert!(
        children_index(&tv, oid).await.is_empty(),
        "no parent survives a reset that erased every section"
    );
}

// =============================================================================
// 2. Ghost parent — a list created and cascaded inside one delta
// =============================================================================

/// One delta creates a row under a **ghost** id (listed in a children list, no
/// section row of its own) and then cascade-deletes the section that listed
/// the ghost. Afterwards no children key exists under the ghost, the ghost is
/// not a member of `obj:chs`, and the new row survives as an orphan, listed
/// nowhere.
///
/// The ghost has no row, so the engine keeps no list under it (SDD §10.4):
/// the `DiffChangeSection` makes an orphan, and the cascade — which follows
/// the lists — walks section 1 and the ghost without reaching it.
/// `cascade_delete_section`'s ghost arm emits nothing; there is no row, no
/// tombstone and no field to erase. Committing the two commands one per
/// `update_layer` reaches the same rows — the orphan's list membership no
/// longer depends on the split — but that form cannot be compared here: its
/// first commit ends with section 1 still listing the ghost, which a debug
/// build's delta-boundary cross-map check refuses, so the batched delta is
/// asserted against the keyspace directly.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn child_under_a_ghost_parent_cascaded_in_one_delta_leaves_no_key_and_survives_unlisted() {
    const GHOST: u64 = 4242;
    const ORPHAN: u64 = 5;

    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;

    let mut state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usda".to_string(),
        ..Default::default()
    };
    state.sections.insert(1, section(1, 0, "World", 0));
    state.children.insert(0, vec![1]);
    // Section 1 lists an id that has no row — a ghost. Stored state can hold
    // this (SDD §10.4 tolerates the dual of it), and a snapshot read
    // `LRANGE`s section 1's list back verbatim, so the engine meets it.
    state.children.insert(1, vec![GHOST]);

    let batched_oid = ls
        .create_layer(
            &session_id,
            "ghost-layer",
            "b-ot3",
            &state,
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");

    commit_batched(
        &ls,
        &session_id,
        "ghost-layer",
        batched_oid,
        vec![
            // A row under the ghost: an orphan, listed nowhere.
            change_cmd(GHOST, ORPHAN, "UnderGhost", 0),
            // Cascades through section 1 and the ghost.
            delete_cmd(1),
        ],
    )
    .await;

    assert!(
        !children_key_exists(&tv, batched_oid, GHOST).await,
        "no children list exists under the ghost"
    );
    assert!(
        !children_index(&tv, batched_oid)
            .await
            .contains(&GHOST.to_string()),
        "the ghost is not in obj:chs"
    );
    assert!(
        !children_key_exists(&tv, batched_oid, 1).await,
        "the cascade-deleted section's own list is gone"
    );
    let snapshot =
        read_object_state_for_test(tv.client(), tv.key_schema(), &batched_oid.to_string())
            .await
            .expect("snapshot load must succeed");
    assert_eq!(
        snapshot
            .sections
            .values()
            .map(|s| (s.section_id, s.parent_id))
            .collect::<Vec<_>>(),
        vec![(ORPHAN, GHOST)],
        "section 1 is gone and the orphan survives under the ghost"
    );
    assert!(
        snapshot.children.values().all(|l| !l.contains(&ORPHAN)),
        "the orphan is listed nowhere: {:?}",
        snapshot.children
    );
}

// =============================================================================
// 3. Large subtree delete — persisted equivalence, external values included
// =============================================================================

/// A subtree delete spanning many unrelated fields and time samples, plus a
/// creation and a field write in the same delta, must persist exactly what the
/// one-command-per-delta reference persists.
///
/// This is the acceptance criterion's Valkey-equivalence arm for the deferred
/// purge: b.ot3 moved every cascade's `DeleteField` / `DeleteTimeSample` out of
/// the per-section loop into one pass at the close of the structural phase.
/// The pass has to run **before** the delta's own `SetField`s, because
/// `ext_rows_written` folds the mutation list in order and a late delete would
/// release an external-value reference the commit still holds — which is why
/// `obj:ext` is part of the comparison rather than an afterthought.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn large_subtree_delete_persists_identically_to_one_command_per_delta() {
    const SUBTREE: u64 = 40;
    const UNRELATED: u64 = 40;
    /// The external value on a section inside the doomed subtree — its
    /// reference is released by the cascade.
    const DOOMED_EXT: u8 = 0xD0;
    /// The external value on a surviving section — its reference must not move.
    const KEPT_EXT: u8 = 0xE5;

    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let ext = ext_store(&tv);
    let session_id = create_test_session(&tv).await;

    /// Section 1 holds the doomed subtree, section 2 the survivors.
    fn arrangement_state() -> ObjectState {
        let mut state = ObjectState {
            sequence_num: 1,
            first_version: 1,
            source_format: "usda".to_string(),
            ..Default::default()
        };
        state.sections.insert(1, section(1, 0, "Doomed", 0));
        state.sections.insert(2, section(2, 0, "Keep", 1));
        state.children.insert(0, vec![1, 2]);
        let mut doomed = Vec::new();
        let mut kept = Vec::new();
        for i in 0..SUBTREE {
            state
                .sections
                .insert(100 + i, section(100 + i, 1, &format!("d{i}"), i as u32));
            doomed.push(100 + i);
        }
        for i in 0..UNRELATED {
            state
                .sections
                .insert(1000 + i, section(1000 + i, 2, &format!("k{i}"), i as u32));
            kept.push(1000 + i);
        }
        state.children.insert(1, doomed);
        state.children.insert(2, kept);
        state
    }

    /// One field and one time sample per leaf, plus the two external values.
    fn value_delta() -> OwnedDelta {
        let mut set_fields = Vec::new();
        let mut time_samples = Vec::new();
        for i in 0..SUBTREE {
            set_fields.push(SetField {
                section_id: 100 + i,
                key_name: "f".to_string(),
                value_or_ext_hash: vec![1],
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 0,
            });
            time_samples.push(TimeSample {
                section_id: 100 + i,
                time: 1.0,
                value_or_ext_hash: vec![1],
                ext_value_size: 0,
                ext_value_index: 0,
            });
        }
        for i in 0..UNRELATED {
            set_fields.push(SetField {
                section_id: 1000 + i,
                key_name: "f".to_string(),
                value_or_ext_hash: vec![2],
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 0,
            });
            time_samples.push(TimeSample {
                section_id: 1000 + i,
                time: 1.0,
                value_or_ext_hash: vec![2],
                ext_value_size: 0,
                ext_value_index: 0,
            });
        }
        set_fields.push(SetField {
            section_id: 100,
            key_name: "ext".to_string(),
            value_or_ext_hash: ext_hash(DOOMED_EXT).to_vec(),
            ext_value_size: 16,
            ext_value_index: 0,
            set_order: 0,
        });
        set_fields.push(SetField {
            section_id: 1000,
            key_name: "ext".to_string(),
            value_or_ext_hash: ext_hash(KEPT_EXT).to_vec(),
            ext_value_size: 16,
            ext_value_index: 1,
            set_order: 0,
        });
        OwnedDelta {
            is_diff: false,
            set_fields,
            time_samples,
            ..Default::default()
        }
    }

    let mut oids = Vec::new();
    for layer in ["batched-layer", "reference-layer"] {
        let oid = ls
            .create_layer(
                &session_id,
                layer,
                "b-ot3",
                &arrangement_state(),
                mock_ext_store(),
            )
            .await
            .expect("create_layer must succeed");
        // The pre-flight `store_values` the gRPC layer pays: one increment per
        // DISTINCT hash per RPC.
        pre_store(&ext, &[ext_hash(DOOMED_EXT), ext_hash(KEPT_EXT)]).await;
        ls.update_layer_relayed(&session_id, layer, oid, Some(value_delta()), WRITER, 0)
            .await
            .expect("value-seeding update_layer must succeed");
        oids.push(oid);
    }
    let (batched_oid, reference_oid) = (oids[0], oids[1]);

    let cmds = vec![
        delete_cmd(1),
        create_cmd(0, 99, "Fresh"),
        change_cmd(2, 1000, "k0", 7),
    ];
    commit_batched(&ls, &session_id, "batched-layer", batched_oid, cmds.clone()).await;
    commit_one_per_delta(&ls, &session_id, "reference-layer", reference_oid, cmds).await;

    let batched = persisted_shape(&tv, batched_oid).await;
    let reference = persisted_shape(&tv, reference_oid).await;
    assert_eq!(
        batched, reference,
        "one commit must persist what N commits persist"
    );

    // The doomed subtree really did go, and the survivors really did stay —
    // otherwise the equality above could hold over two equally-broken objects.
    assert_eq!(
        batched.field_keys.len() as u64,
        UNRELATED + 1,
        "the unrelated fields and the surviving external-value field remain"
    );
    assert!(
        batched.field_keys.iter().all(|(sid, _)| *sid >= 1000),
        "no field of the doomed subtree survived: {:?}",
        batched.field_keys
    );
    assert_eq!(
        batched.tombstone_ids.len() as u64,
        SUBTREE + 1,
        "the deleted root and every descendant is tombstoned"
    );

    // External-value references. The batched-vs-reference comparison is already
    // made by the `PersistedShape` equality above (`ext_refs` is a column of the
    // shape); what remains here is the non-vacuity half that equality cannot
    // give — the doomed row's reference is actually released and the surviving
    // row's is actually retained, rather than both objects being equally wrong.
    assert_eq!(
        batched.ext_refs,
        vec![(ext_hash_hex(KEPT_EXT), 1)],
        "only the surviving row's external value is still referenced by the object"
    );
}

// =============================================================================
// 4. Mixed conflict sequence — persisted equivalence
// =============================================================================

/// A duplicate-name conflict sequence — rename-duplicate, create, move,
/// reorder, delete — must persist identically whether committed as one delta
/// or five.
///
/// The domain-side twin
/// (`apply_delta_mixed_conflict_sequence_matches_one_per_delta`) compares the
/// in-memory state; this compares what actually landed, which is where the
/// coalesced children writes and the `obj:chs` membership fork are observable.
/// The lowest-`section_id` tie-break (b.ot3 P2) is what makes the two runs
/// comparable at all: pre-fix the conflict target came from a `HashMap`'s
/// layout, so the two layers could legally have cascade-deleted different
/// subtrees.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mixed_conflict_sequence_persists_identically_to_one_command_per_delta() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;

    fn arrangement_state() -> ObjectState {
        let mut state = ObjectState {
            sequence_num: 1,
            first_version: 1,
            source_format: "usda".to_string(),
            ..Default::default()
        };
        state.sections.insert(1, section(1, 0, "P1", 0));
        state.sections.insert(2, section(2, 0, "P2", 1));
        state.sections.insert(10, section(10, 1, "A", 0));
        state.sections.insert(20, section(20, 1, "B", 1));
        state.sections.insert(30, section(30, 1, "C", 2));
        state.sections.insert(40, section(40, 2, "Mover", 0));
        state.children.insert(0, vec![1, 2]);
        state.children.insert(1, vec![10, 20, 30]);
        state.children.insert(2, vec![40]);
        state
    }

    let mut oids = Vec::new();
    for layer in ["conflict-batched", "conflict-reference"] {
        oids.push(
            ls.create_layer(
                &session_id,
                layer,
                "b-ot3",
                &arrangement_state(),
                mock_ext_store(),
            )
            .await
            .expect("create_layer must succeed"),
        );
    }
    let (batched_oid, reference_oid) = (oids[0], oids[1]);

    let cmds = vec![
        // Rename 20 onto 10's name — two "A" siblings under parent 1.
        move_cmd(20, 1, 1, "A"),
        // Create onto "A" → cascades the LOWEST holder, 10.
        create_cmd(1, 50, "A"),
        // Move onto "A" → holders are now {20, 50}; the lowest is 20.
        move_cmd(40, 2, 1, "A"),
        // Verbatim order over what is left.
        reorder_cmd(1, vec![50, 30, 40]),
        delete_cmd(30),
    ];
    commit_batched(
        &ls,
        &session_id,
        "conflict-batched",
        batched_oid,
        cmds.clone(),
    )
    .await;
    commit_one_per_delta(&ls, &session_id, "conflict-reference", reference_oid, cmds).await;

    let batched = persisted_shape(&tv, batched_oid).await;
    let reference = persisted_shape(&tv, reference_oid).await;
    assert_eq!(
        batched, reference,
        "one commit must persist what N commits persist"
    );

    assert_eq!(
        children_list(&tv, batched_oid, 1).await,
        vec!["50".to_string(), "40".to_string()],
        "the settled order of parent 1's list"
    );
    assert!(
        !batched
            .sections
            .iter()
            .any(|(sid, ..)| *sid == 10 || *sid == 20),
        "both conflict resolutions took the lowest-id holder: {:?}",
        batched.sections
    );
    assert_eq!(
        children_list(&tv, batched_oid, 2).await,
        Vec::<String>::new(),
        "parent 2's list emptied when its only child moved away"
    );
    assert!(
        !children_index(&tv, batched_oid)
            .await
            .contains(&"2".to_string()),
        "an emptied list leaves obj:chs with its key"
    );
}

// =============================================================================
// 5. Many siblings in one delta — the coalesced write carries the whole list
// =============================================================================

/// The coalescing acceptance criterion, at the persistence tier: N siblings
/// created in one delta must land as one ordered list of N ids.
///
/// Pre-b.ot3 the same delta emitted N `SetChildren`, each a `DEL` + `RPUSH` of
/// a growing prefix, which the commit executed in full — N(N+1)/2 ids pushed
/// through the transaction to reach the same end state. The risk a coalescing
/// fix carries is the opposite one: writing the *first* list instead of the
/// last, or skipping the write entirely for a parent it decided was clean. Both
/// show up here as a wrong or missing list.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn many_siblings_in_one_delta_persist_as_one_ordered_list() {
    const N: u64 = 300;

    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;

    let state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usda".to_string(),
        ..Default::default()
    };
    let oid = ls
        .create_layer(
            &session_id,
            "siblings-layer",
            "b-ot3",
            &state,
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");

    commit_batched(
        &ls,
        &session_id,
        "siblings-layer",
        oid,
        (1..=N)
            .map(|i| create_cmd(0, i, &format!("s{i}")))
            .collect(),
    )
    .await;

    let persisted = children_list(&tv, oid, 0).await;
    let expected: Vec<String> = (1..=N).map(|i| i.to_string()).collect();
    assert_eq!(
        persisted, expected,
        "the single coalesced write carries the whole settled list, in order"
    );
    assert!(
        children_index(&tv, oid).await.contains(&"0".to_string()),
        "a non-empty list puts its parent in obj:chs"
    );

    // And the snapshot reads back what the delta meant.
    let snapshot = read_object_state_for_test(tv.client(), tv.key_schema(), &oid.to_string())
        .await
        .expect("snapshot load must succeed");
    assert_eq!(snapshot.sections.len() as u64, N);
    assert_eq!(
        snapshot.children.get(&0).map(Vec::len).unwrap_or_default() as u64,
        N,
        "the loader reads the settled list back"
    );
}

// =============================================================================
// 6. Delete then recreate the same parent inside one delta
// =============================================================================

/// One delta cascade-deletes section 1 — taking `obj:ch:1` with it — creates
/// section 1 again, and lists a fresh child under it. The key must exist
/// afterwards holding the recreated list, and `"1"` must be back in
/// `obj:chs`.
///
/// This is where the finalizer's ordering argument meets `execute_mutation`.
/// The domain-side twin
/// (`apply_delta_delete_then_recreate_parent_emits_delete_before_the_rebuild`)
/// pins only the mutation *order*; the two mutations are executed as
/// `DEL` + `SREM obj:chs` (`DeleteChildrenList`) and then
/// `DEL` + `RPUSH` + `SADD obj:chs` (`SetChildren`) inside one MULTI/EXEC, so a
/// mis-ordering does not merely lose the list — it strands the parent on the
/// wrong side of SDD §7.7's "member present ⟺ children key exists": the
/// finalizer's `SADD` landing first and the cascade's `SREM` second leaves the
/// key populated with nothing indexing it.
///
/// The re-creation is what lets the list come back: between the delete and
/// the create, id 1 has no row, and the engine keeps no list under an id with
/// no row (SDD §10.4).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_then_recreate_parent_in_one_delta_persists_identically_to_one_command_per_delta() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;

    fn arrangement_state() -> ObjectState {
        let mut state = ObjectState {
            sequence_num: 1,
            first_version: 1,
            source_format: "usda".to_string(),
            ..Default::default()
        };
        state.sections.insert(1, section(1, 0, "P", 0));
        state.sections.insert(2, section(2, 1, "C", 0));
        // Section 2 has a child of its own, so `obj:ch:2` exists and the
        // cascade really has a second list to delete and un-index.
        state.sections.insert(3, section(3, 2, "G", 0));
        state.children.insert(0, vec![1]);
        state.children.insert(1, vec![2]);
        state.children.insert(2, vec![3]);
        state
    }

    let mut oids = Vec::new();
    for layer in ["recreate-batched", "recreate-reference"] {
        oids.push(
            ls.create_layer(
                &session_id,
                layer,
                "b-ot3",
                &arrangement_state(),
                mock_ext_store(),
            )
            .await
            .expect("create_layer must succeed"),
        );
    }
    let (batched_oid, reference_oid) = (oids[0], oids[1]);

    // Preconditions: without the keys the post-delta assertions are vacuous.
    assert!(
        children_key_exists(&tv, batched_oid, 1).await,
        "fixture precondition: parent 1's children key must exist before the delta"
    );
    assert!(
        children_key_exists(&tv, batched_oid, 2).await
            && children_index(&tv, batched_oid)
                .await
                .contains(&"2".to_string()),
        "fixture precondition: section 2's children key and its obj:chs member must \
         exist before the delta"
    );

    let cmds = vec![
        delete_cmd(1),
        create_cmd(0, 1, "P"),
        change_cmd(1, 9, "New", 0),
    ];
    commit_batched(
        &ls,
        &session_id,
        "recreate-batched",
        batched_oid,
        cmds.clone(),
    )
    .await;
    commit_one_per_delta(&ls, &session_id, "recreate-reference", reference_oid, cmds).await;

    let batched = persisted_shape(&tv, batched_oid).await;
    let reference = persisted_shape(&tv, reference_oid).await;
    assert_eq!(
        batched, reference,
        "one commit must persist what three commits persist"
    );

    assert_eq!(
        children_list(&tv, batched_oid, 1).await,
        vec!["9".to_string()],
        "the recreated list survives the cascade's DEL that preceded it"
    );
    assert!(
        children_index(&tv, batched_oid)
            .await
            .contains(&"1".to_string()),
        "the re-materialized parent is back in obj:chs, matching its key"
    );
    assert_eq!(
        children_list(&tv, batched_oid, 0).await,
        vec!["1".to_string()],
        "the root lists the re-created section"
    );
    // The other half of the §7.7 biconditional, on a parent that really did
    // go away: the cascaded section 2's list, which existed before the delta
    // (asserted above), is gone from both the keyspace and the index.
    assert!(
        !children_key_exists(&tv, batched_oid, 2).await,
        "the cascaded child's own list is gone"
    );
    assert!(
        !children_index(&tv, batched_oid)
            .await
            .contains(&"2".to_string()),
        "and so is its index member"
    );

    let snapshot =
        read_object_state_for_test(tv.client(), tv.key_schema(), &batched_oid.to_string())
            .await
            .expect("snapshot load must succeed");
    let mut survivors: Vec<u64> = snapshot.sections.keys().copied().collect();
    survivors.sort_unstable();
    assert_eq!(
        survivors,
        vec![1, 9],
        "the re-created section and its new child survive; the cascaded \
         subtree does not: {:?}",
        snapshot.sections
    );
}

// =============================================================================
// 7. Ascending ordered siblings (b.sao) — the cache-maintenance workload
// =============================================================================

/// The b.sao workload at the persistence tier: N fresh siblings inserted under
/// a **non-empty** parent at ascending non-zero `section_order`s, in one delta,
/// against the one-command-per-`update_layer` reference.
///
/// Section 5 above drives the many-siblings shape with `create_cmd` only, and
/// `CommandCreateSection` carries no `sectionOrder` at all — so the command
/// family b.sao is about had no persisted coverage, and neither did the
/// insertion *positions* that an order-bearing command computes. Those
/// positions are exactly what the per-delta `ParentCache` answers, and exactly
/// what a stale or wrongly-reused cache gets wrong; in the list they are
/// persisted as `LRANGE` order, which the reference layer independently
/// derives one command at a time from a freshly loaded snapshot.
///
/// The delta also empties a second parent, so the run covers the other two
/// §7.7 clauses in the same comparison: the empty-list delete and
/// `obj:chs` membership ⟺ children-key existence. Field and time-sample rows on
/// the emptied parent's child give the deferred cascade purge something to
/// erase, so purge-before-`SetField` ordering is exercised rather than assumed
/// (both are columns of [`PersistedShape`], so the comparison inherits them).
///
/// Sizes are kept modest deliberately: the reference arm is one Valkey
/// round-trip per command, so the cost of this test is N commits, not N
/// commands. The asymptotic claim is the unit tier's job — `src/domain/delta.rs`
/// asserts the cache counters out to 8,000 siblings — and this tier's job is
/// that what lands in Valkey still agrees.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ascending_ordered_siblings_persist_identically_to_one_command_per_delta() {
    /// Children parent 1 already holds, at orders 1..=M.
    const M: u64 = 6;
    /// New ascending siblings inserted under parent 1.
    const N: u64 = 40;
    /// The parent whose list the delta empties.
    const EMPTIED: u64 = 2;
    const DOOMED_CHILD: u64 = 500;

    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;

    fn arrangement_state() -> ObjectState {
        let mut state = ObjectState {
            sequence_num: 1,
            first_version: 1,
            source_format: "usda".to_string(),
            ..Default::default()
        };
        state.sections.insert(1, section(1, 0, "Growing", 0));
        state
            .sections
            .insert(EMPTIED, section(EMPTIED, 0, "Emptied", 1));
        state.children.insert(0, vec![1, EMPTIED]);
        let mut existing = Vec::new();
        for i in 1..=M {
            state
                .sections
                .insert(100 + i, section(100 + i, 1, &format!("e{i}"), i as u32));
            existing.push(100 + i);
        }
        state.children.insert(1, existing);
        state
            .sections
            .insert(DOOMED_CHILD, section(DOOMED_CHILD, EMPTIED, "Doomed", 0));
        state.children.insert(EMPTIED, vec![DOOMED_CHILD]);
        state
    }

    /// Rows on the child the delta cascades away, so the deferred purge pass has
    /// work to do in the same commit as the sibling insertions.
    fn value_delta() -> OwnedDelta {
        OwnedDelta {
            is_diff: false,
            set_fields: vec![SetField {
                section_id: DOOMED_CHILD,
                key_name: "f".to_string(),
                value_or_ext_hash: vec![3],
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 0,
            }],
            time_samples: vec![TimeSample {
                section_id: DOOMED_CHILD,
                time: 1.0,
                value_or_ext_hash: vec![3],
                ext_value_size: 0,
                ext_value_index: 0,
            }],
            ..Default::default()
        }
    }

    let mut oids = Vec::new();
    for layer in ["ascending-batched", "ascending-reference"] {
        let oid = ls
            .create_layer(
                &session_id,
                layer,
                "b-ot3",
                &arrangement_state(),
                mock_ext_store(),
            )
            .await
            .expect("create_layer must succeed");
        ls.update_layer_relayed(&session_id, layer, oid, Some(value_delta()), WRITER, 0)
            .await
            .expect("value-seeding update_layer must succeed");
        oids.push(oid);
    }
    let (batched_oid, reference_oid) = (oids[0], oids[1]);

    // Ascending non-zero orders, every one above everything parent 1 already
    // holds: each insertion appends, and the settled order is command order.
    let mut cmds: Vec<StructCommand> = (1..=N)
        .map(|j| change_cmd(1, 1000 + j, &format!("n{j}"), (M + j) as u32))
        .collect();
    // Empties parent 2's list in the same delta.
    cmds.push(delete_cmd(DOOMED_CHILD));

    commit_batched(
        &ls,
        &session_id,
        "ascending-batched",
        batched_oid,
        cmds.clone(),
    )
    .await;
    commit_one_per_delta(&ls, &session_id, "ascending-reference", reference_oid, cmds).await;

    let batched = persisted_shape(&tv, batched_oid).await;
    let reference = persisted_shape(&tv, reference_oid).await;
    assert_eq!(
        batched, reference,
        "one commit must persist what N commits persist, insertion positions included"
    );

    // Non-vacuity: the equality above is satisfied by two equally-broken
    // objects, so state what the delta must actually have persisted.
    let expected: Vec<String> = (1..=M)
        .map(|i| (100 + i).to_string())
        .chain((1..=N).map(|j| (1000 + j).to_string()))
        .collect();
    assert_eq!(
        children_list(&tv, batched_oid, 1).await,
        expected,
        "the coalesced write carries the pre-existing children followed by the new \
         siblings in ascending order — the positions the per-delta cache computed"
    );

    // SDD §7.7's biconditional, on both sides.
    assert!(
        children_index(&tv, batched_oid)
            .await
            .contains(&"1".to_string()),
        "a non-empty list puts its parent in obj:chs"
    );
    assert!(
        !children_key_exists(&tv, batched_oid, EMPTIED).await,
        "the emptied parent's children key is deleted, not left holding a dead id"
    );
    assert!(
        !children_index(&tv, batched_oid)
            .await
            .contains(&EMPTIED.to_string()),
        "and it leaves obj:chs together with its key"
    );

    // The deferred purge really ran in the same commit as the insertions.
    assert!(
        batched.field_keys.is_empty() && batched.time_sample_keys.is_empty(),
        "the cascaded child's field and time-sample rows were purged: {:?} / {:?}",
        batched.field_keys,
        batched.time_sample_keys
    );
    assert_eq!(
        batched.tombstone_ids,
        vec![DOOMED_CHILD],
        "exactly the deleted child is tombstoned"
    );
}
