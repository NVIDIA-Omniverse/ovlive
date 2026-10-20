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

//! Integration tests for b.9fn and its follow-up b.jdr — children-ordering
//! lists (`le:{oid}:obj:ch:{parent_id}`) that leaked on every object
//! retirement because no index named their parent.
//!
//! # The bug (b.9fn) — the synthetic root
//!
//! Both retirement paths built their UNLINK inventory by walking the object's
//! own indices: `obj:secs` for live sections and `obj:dels` for tombstones,
//! emitting `obj:ch:{sid}` for each id found. Parent id `0` is the *synthetic
//! root* — it names the top-level section ordering, is not itself a section,
//! and is therefore a member of neither index. So `write_object` RPUSHes
//! `obj:ch:0` for every layer that has any top-level section (i.e. every real
//! layer), and no cleanup path ever reaped it: one leaked key per retired
//! object id, forever, on a `noeviction` store.
//!
//! The fix routes both paths through the pure
//! `KeySchema::object_cleanup_keys` helper, which emits the root children key
//! unconditionally.
//!
//! # The follow-up bug (b.jdr) — dangling parents
//!
//! b.9fn closed the root case by *hard-coding* the one non-section parent id
//! the schema reserves. It did not close the general case, which its own
//! doc-comment named as a residual gap: `apply_diff_change_section` does not
//! validate that `parent_id` names an existing section (validation rejects
//! only self-parenting), and a *dangling* parent is a deliberately tolerated
//! state — SDD §10.4, and the server itself emits such deltas in catch-up
//! diffs. So `le:{oid}:obj:ch:{dangling}` gets written for a parent id in
//! neither `obj:secs` nor `obj:dels`, and the reconstruction
//! *sections ∪ tombstones ∪ root* could not name it. Same leak as b.9fn, at a
//! lower rate.
//!
//! The fix records the parent ids in a new index, `le:{oid}:obj:chs`
//! (`KeySchema::obj_children_index`), maintained inside the same MULTI/EXEC as
//! the children lists themselves, and unions it into the inventory.
//!
//! Since b.a4k the engine keeps no list under a parent with no row (SDD
//! §10.4), so no commit writes a dangling-parent list any more; the lists
//! earlier builds stored remain until their object retires, and retirement
//! must still reap them. Two dangling-parent lists are arranged here:
//!
//! - **`write_object`** — SADDs every parent with a non-empty child vec,
//!   materializing a hand-built state as given. Driven by the
//!   [`DANGLING_PARENT_WRITE_OBJECT`] entry in [`state_with_full_key_surface`].
//! - **An earlier build's update path** — the list `execute_mutation`'s
//!   `SetChildren` arm wrote for a `DiffChangeSection` under a dangling
//!   parent, with its `obj:chs` member. [`orphan_with_an_earlier_builds_list`]
//!   commits that `DiffChangeSection` (which today makes a row and no list)
//!   and injects the list and member with raw writes, naming
//!   [`DANGLING_PARENT_MUTATION`].
//!
//! The two ids are deliberately *different* so a leak names which list
//! survived rather than reading as a generic leak.
//!
//! Unit coverage of the inventory itself lives in `src/storage/keys.rs`
//! (section F); this file pins the *behavioural* callers against a live
//! Valkey. Tests 1 and 2 are the leak tripwires on the two `cleanup_old_object`
//! entry points. Test 3 retires an object carrying **no** `obj:chs` — the
//! rolling-deploy shape — and pins exactly how far the retained pre-b.jdr
//! reconstruction still reaches. Tests 4 and 5 pin the halves of the index's
//! *membership ⟺ key existence* contract that a leak tripwire cannot see: the
//! write-side skip for an empty child vec, and the two SREM forks.
//!
//! # Why both callers, separately
//!
//! `cleanup_old_object` has two entry points with materially different error
//! and timing postures, and a regression could plausibly hit one and not the
//! other:
//!
//! - **CreateLayer overwrite** — cleanup runs in a task spawned on the
//!   `LayerStore` cleanup tracker *after* `create_layer` has already returned
//!   the new object id, and it swallows errors with `tracing::warn`. A test
//!   must join the tracker via `wait_for_cleanup` or it races the UNLINK.
//! - **DeleteLayer** — cleanup runs synchronously inside the RPC, inside the
//!   retry helper, and propagates errors to the caller.
//!
//! The orphan-sweep caller (`orphan_sweep::cleanup_orphaned_object`) is the
//! third retirement route and is covered in `orphan_sweep_test.rs`
//! (`sweep_cleans_all_sub_keys_of_orphaned_object`).
//!
//! # Assertion shape
//!
//! Rather than probing `obj:ch:0` alone, each test SCANs the whole
//! `le:{oid}:obj:*` namespace and asserts every survivor is `obj:stream`
//! — which survives *legitimately*, PEXPIRE'd with a grace window so
//! subscribers lagging between XREAD round-trips still observe the `deleted`
//! sentinel. That makes these tests a general leak tripwire: any future key
//! family added to the write surface but not to the cleanup inventory fails
//! here, not just the one b.9fn happened to miss.
//!
//! **The tripwire is only as wide as the fixture.** A SCAN-based leak
//! assertion can only catch a family the arrangement actually materialized —
//! a fixture that writes sections and children alone would keep passing if a
//! regression dropped `obj:flds` / `obj:fld:*` / `obj:tss` / `obj:ts:*` /
//! `obj:dels` / `obj:del:*` from `object_cleanup_keys`. So the arrangement
//! deliberately lights up **every** family the write surface can produce:
//!
//! - `state_with_full_key_surface` seeds sections, all three children-list
//!   shapes (root / live-section parent / dangling parent), fields, and time
//!   samples (the four leaf shapes `write_object` MULTI/EXECs);
//! - a `DeleteSection` `update_layer` commit is driven *before* retirement to
//!   land the tombstone families, which are unreachable from `ObjectState`
//!   alone — it carries no `deleted_sections` map, so tombstones exist only
//!   as a side effect of an applied delete command;
//! - a `DiffChangeSection` `update_layer` commit is driven *before* retirement
//!   to land an orphan row, and the dangling-parent children list an earlier
//!   build's mutation path stored for it is injected beside it, with its
//!   `obj:chs` member;
//! - a fenced read's acquire and release and a writer-intent registration,
//!   through `storage::snapshot_fence`, land the snapshot fence
//!   (`obj:snapfence`) and writer-intent (`obj:snapintent`) keys, which no
//!   index names.
//!
//! `assert_full_key_surface_present` pins that arrangement so a fixture
//! regression surfaces as a failed precondition rather than as a quietly
//! narrowed tripwire.

use std::sync::Arc;
use std::time::Duration;

use fred::interfaces::{KeysInterface, ListInterface, SetsInterface};
use ordered_float::OrderedFloat;

use live_edit::domain::types::{Field, FieldValue, ObjectState, Section, TimeSample};
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::snapshot_fence::{
    self, AcquireVerdict, IntentVerdict, LayerSize, ReleaseVerdict, WriterIntent,
};
use live_edit::storage::test_support::relay::RelayedUpdates;
use live_edit::storage::test_support::TestValkey;
use oodf_ffi::{
    CommandDeleteSection, CommandDiffChangeSection, CommandMoveSection, OwnedDelta, StructCommand,
};

const CLEANUP_JOIN_BUDGET: Duration = Duration::from_secs(10);

/// The section the pre-retirement `DeleteSection` commit tombstones. A leaf
/// child of the synthetic root with no fields or time samples of its own, so
/// deleting it adds the tombstone families without cascading away any of the
/// field / time-sample state the rest of the arrangement depends on.
const DOOMED_SECTION_ID: u64 = 3;

/// b.jdr — a parent id that names no section, reached through the
/// **`write_object`** surface: [`state_with_full_key_surface`] seeds
/// `children[77] = [ORPHAN_SECTION_ID]` directly, so the MULTI/EXEC in
/// `write_object` RPUSHes `obj:ch:77` and SADDs `77` into `obj:chs`.
const DANGLING_PARENT_WRITE_OBJECT: u64 = 77;

/// The orphan section parented to [`DANGLING_PARENT_WRITE_OBJECT`]. A real
/// member of `obj:secs` whose `parent_id` resolves to nothing — the tolerated
/// state of SDD §10.4, not corruption.
const ORPHAN_SECTION_ID: u64 = 4;

/// b.jdr — a *second*, distinct dangling parent id, whose list is the one an
/// earlier build's **`execute_mutation` `SetChildren`** arm stored, arranged
/// by [`orphan_with_an_earlier_builds_list`]. Distinct from
/// [`DANGLING_PARENT_WRITE_OBJECT`] on purpose: a leak of `obj:ch:77` and a
/// leak of `obj:ch:88` name different lists, and the failure message says
/// which.
const DANGLING_PARENT_MUTATION: u64 = 88;

/// The section [`orphan_with_an_earlier_builds_list`] creates under
/// [`DANGLING_PARENT_MUTATION`].
const MUTATION_ORPHAN_SECTION_ID: u64 = 5;

fn mock_ext_store() -> Arc<MockExternalValueStore> {
    Arc::new(MockExternalValueStore::new())
}

fn layer_store(tv: &TestValkey) -> LayerStore {
    LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
}

async fn create_test_session(tv: &TestValkey) -> String {
    SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create("b-9fn-session", "/stage", "b-9fn-user")
        .await
        .expect("session creation must succeed")
}

/// An `ObjectState` covering every leaf shape `write_object` materializes, so
/// the post-retirement SCAN is a tripwire on the *whole* `obj:*` inventory
/// rather than on the section/children corner of it.
///
/// Section tree — two levels plus a doomed leaf plus an orphan, so
/// materialization writes **all three** children-list shapes:
///
/// - `obj:ch:0` → `[1, 3]` — the root ordering. Parent 0 is in no index; this
///   is the key b.9fn leaked.
/// - `obj:ch:1` → `[2]` — a per-section ordering. Parent 1 *is* a member of
///   `obj:secs`, so the pre-fix enumeration already reaped it.
/// - `obj:ch:77` → `[4]` — a *dangling* parent ordering
///   ([`DANGLING_PARENT_WRITE_OBJECT`]). Section 4 exists and is a member of
///   `obj:secs`, but its parent 77 names no section, so 77 is in neither
///   `obj:secs` nor `obj:dels` and no reconstruction can produce this key.
///   This is what b.jdr leaked, and it is reachable only through the recorded
///   `obj:chs` index.
///
/// Keeping all three in one fixture is what makes the assertions diagnostic
/// rather than merely red: pre-b.9fn, `obj:ch:1` was reaped and `obj:ch:0`
/// survived; pre-b.jdr, both were reaped and `obj:ch:77` survived. Each
/// survivor isolates a different failure of the enumeration rather than
/// reading as "children lists are never cleaned up".
///
/// An orphan is *not* corrupt state — `apply_diff_change_section` creates
/// section 4's shape and validation permits it (SDD §10.4). The list under 77
/// is what an earlier build stored with it; the current engine stores none,
/// but `write_object` materializes this hand-built state as given.
///
/// Fields and time samples (shape copied from `rich_object_state` in
/// `checkpoint_apply_test.rs`) hang off sections 1 and 2 — *not* section
/// [`DOOMED_SECTION_ID`], whose later `DeleteSection` would cascade them away
/// again.
fn state_with_full_key_surface() -> ObjectState {
    let mut state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usda".to_string(),
        next_set_order: 10,
        ..Default::default()
    };

    state.sections.insert(
        1,
        Section {
            section_id: 1,
            parent_id: 0,
            name: "World".to_string(),
            section_type: 1,
            section_order: 0,
            first_version: 1,
            last_version: 1,
        },
    );
    state.sections.insert(
        2,
        Section {
            section_id: 2,
            parent_id: 1,
            name: "Cube".to_string(),
            section_type: 2,
            section_order: 0,
            first_version: 1,
            last_version: 1,
        },
    );
    state.sections.insert(
        DOOMED_SECTION_ID,
        Section {
            section_id: DOOMED_SECTION_ID,
            parent_id: 0,
            name: "Doomed".to_string(),
            section_type: 1,
            section_order: 1,
            first_version: 1,
            last_version: 1,
        },
    );

    // b.jdr: an orphan section whose parent id names no section. Legal state
    // (SDD §10.4) — the diff engine emits such sections after the reachable
    // DFS walk rather than dropping them.
    state.sections.insert(
        ORPHAN_SECTION_ID,
        Section {
            section_id: ORPHAN_SECTION_ID,
            parent_id: DANGLING_PARENT_WRITE_OBJECT,
            name: "Orphan".to_string(),
            section_type: 1,
            section_order: 0,
            first_version: 1,
            last_version: 1,
        },
    );

    state.children.insert(0, vec![1, DOOMED_SECTION_ID]);
    state.children.insert(1, vec![2]);
    state
        .children
        .insert(DANGLING_PARENT_WRITE_OBJECT, vec![ORPHAN_SECTION_ID]);

    for (section_id, key_name, value, order) in [
        (1_u64, "color", vec![0xFF_u8, 0x00, 0x80], 1_u64),
        (2, "size", vec![0x42], 2),
    ] {
        state.fields.insert(
            (section_id, key_name.to_string()),
            Field {
                section_id,
                key_name: key_name.to_string(),
                value: FieldValue::Inline(value),
                set_order: order,
                authoring_set_order: order,
                version: 1,
            },
        );
    }

    for (section_id, time, value) in [(1_u64, 0.0_f64, vec![0xAA_u8, 0xBB]), (2, 1.5, vec![0xCC])] {
        state.time_samples.insert(
            (section_id, OrderedFloat(time)),
            TimeSample {
                section_id,
                time: OrderedFloat(time),
                value: FieldValue::Inline(value),
                version: 1,
            },
        );
    }

    state
}

/// Commit a `DeleteSection` for [`DOOMED_SECTION_ID`] so `obj:dels` and
/// `obj:del:{sid}` exist at retirement time.
///
/// This has to go through a real `update_layer` commit: `ObjectState` has no
/// `deleted_sections` map, so tombstones are not reachable from the
/// `create_layer` arrangement at all — they exist only as a side effect of
/// `cascade_delete_section` running over an applied delete command. Without
/// this step, dropping the tombstone families from `object_cleanup_keys`
/// would leave both tests green.
///
/// `DeleteSection`, not `DiffDeleteSection`: the latter is *silently ignored*
/// on the ingest path (SDD §9.1.4f, `domain/delta.rs`), so it produces no
/// tombstone and no mutation of any kind. Being a structural command,
/// `DeleteSection` also routes this commit down the WATCH/MULTI/EXEC slow
/// path, which is the path that writes tombstones.
async fn tombstone_doomed_section(ls: &LayerStore, session_id: &str, layer_name: &str, oid: u64) {
    let delta = OwnedDelta {
        struct_commands: vec![StructCommand::DeleteSection(CommandDeleteSection {
            parent_id: 0,
            section_id: DOOMED_SECTION_ID,
        })],
        ..Default::default()
    };

    ls.update_layer_relayed(session_id, layer_name, oid, Some(delta), "b-9fn-writer", 0)
        .await
        .expect("DeleteSection update_layer must succeed");
}

/// b.jdr — commit a real `DiffChangeSection` that creates
/// [`MUTATION_ORPHAN_SECTION_ID`] under the dangling parent
/// [`DANGLING_PARENT_MUTATION`], then inject `le:{oid}:obj:ch:88` = `[5]` and
/// the `obj:chs` member `88` with raw writes — the list and member an earlier
/// build's **mutation** path wrote for that commit.
///
/// `DiffChangeSection` specifically, because it is the only command that
/// names an unvalidated parent: `apply_diff_change_section` takes the
/// new-section branch here (section 5 is not in state) and makes an orphan
/// row under 88 — the tolerated state of SDD §10.4. Earlier builds'
/// `insert_child_into_parent(88, 5)` also stored a list under 88, through
/// `execute_mutation`'s `SetChildren` arm, which SADDs one parent id at a
/// time. The current engine keeps no list under a parent with no row, so the
/// commit writes none; the injection reproduces what such an object still
/// holds, so retirement's `obj:chs`-driven reaping of it stays covered. (The
/// `SetChildren` arm's own SADD and SREM are covered by tests 4 and 5 below
/// and by `chs_complete_tests.rs`.)
async fn orphan_with_an_earlier_builds_list(
    tv: &TestValkey,
    ls: &LayerStore,
    session_id: &str,
    layer_name: &str,
    oid: u64,
) {
    let delta = OwnedDelta {
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: DANGLING_PARENT_MUTATION,
            section_id: MUTATION_ORPHAN_SECTION_ID,
            section_name: "MutationOrphan".to_string(),
            section_type: 1,
            section_order: 0,
        })],
        ..Default::default()
    };

    ls.update_layer_relayed(session_id, layer_name, oid, Some(delta), "b-9fn-writer", 0)
        .await
        .expect("DiffChangeSection update_layer must succeed");

    let oid_str = oid.to_string();
    let pushed: i64 = tv
        .client()
        .pool
        .rpush(
            tv.key_schema()
                .obj_children(&oid_str, &DANGLING_PARENT_MUTATION.to_string()),
            MUTATION_ORPHAN_SECTION_ID.to_string(),
        )
        .await
        .expect("RPUSH obj:ch:88 must succeed");
    assert_eq!(
        pushed, 1,
        "arrangement: the commit itself wrote no list under the dangling parent"
    );
    let added: i64 = tv
        .client()
        .pool
        .sadd(
            tv.key_schema().obj_children_index(&oid_str),
            DANGLING_PARENT_MUTATION.to_string(),
        )
        .await
        .expect("SADD obj:chs must succeed");
    assert_eq!(
        added, 1,
        "arrangement: the commit itself indexed no list under the dangling parent"
    );
}

/// The layer size [`fence_and_intent`] records on the fence. Far above any real
/// fixture: it is chosen for its limits, not its contents. The key outlives the
/// gap by `key_linger(L)` (about an hour at this size) and an intent lasts
/// `intent_ttl(L)` (about three minutes), so neither can lapse on its own between the
/// presence precondition and the retirement, which would let a leak go unseen.
const FENCED_LAYER_SIZE: LayerSize = LayerSize {
    items: 10_000_000,
    bytes: 120_000_000,
};

/// Materialize the snapshot fence (`obj:snapfence`) and writer-intent
/// (`obj:snapintent`) keys on `oid` the way production writes them: a fenced
/// read's acquire and release (which leave the fence key behind with the gap
/// and the recorded size), then a WATCH-path writer's intent registration.
///
/// Returns the intent's guard: the caller holds it past the retirement,
/// because dropping it removes the intent (its spawned `HDEL`) and with it the
/// key the tripwire must see reaped. Its `HDEL` on drop, after retirement,
/// removes nothing and creates nothing.
async fn fence_and_intent(tv: &TestValkey, oid: u64) -> Box<WriterIntent> {
    let lease =
        match snapshot_fence::acquire_fence(tv.client(), tv.key_schema(), oid, FENCED_LAYER_SIZE)
            .await
            .expect("acquire_fence must answer")
        {
            AcquireVerdict::Granted(lease) => lease,
            other => panic!("fixture: the fence acquire must grant; got {other:?}"),
        };
    let released = lease
        .release(Some(FENCED_LAYER_SIZE))
        .await
        .expect("the fence release must answer");
    assert_eq!(
        released,
        ReleaseVerdict::Verified,
        "fixture: nothing commits between the acquire and the release"
    );
    match snapshot_fence::register_intent(tv.client(), tv.key_schema(), oid, FENCED_LAYER_SIZE)
        .await
        .expect("register_intent must answer")
    {
        IntentVerdict::Registered(intent) => intent,
        other => panic!("fixture: no reader is live, so the intent must register; got {other:?}"),
    }
}

/// Arrange the full pre-retirement key surface on `oid`: tombstone families,
/// the earlier build's dangling-parent children list and the snapshot fence
/// and writer-intent keys, then assert every family the tripwire depends on
/// actually landed. Returns the intent guard [`fence_and_intent`] describes;
/// hold it until the assertions are done.
async fn arrange_full_key_surface(
    tv: &TestValkey,
    ls: &LayerStore,
    session_id: &str,
    layer_name: &str,
    oid: u64,
) -> Box<WriterIntent> {
    tombstone_doomed_section(ls, session_id, layer_name, oid).await;
    orphan_with_an_earlier_builds_list(tv, ls, session_id, layer_name, oid).await;
    let intent = fence_and_intent(tv, oid).await;
    assert_full_key_surface_present(tv, oid).await;
    intent
}

/// Every live `le:{prefix}{oid}:obj:*` key, as bare suffixes (`obj:hdr`,
/// `obj:ch:0`, …) so failure messages read as key families rather than
/// prefixed noise. The cluster-aware SCAN itself lives in
/// `crate::common::scan_object_keys`.
async fn surviving_obj_key_suffixes(tv: &TestValkey, object_id: u64) -> Vec<String> {
    let full_prefix = format!("{}{{{object_id}}}:", tv.prefix());
    let mut found: Vec<String> =
        crate::common::scan_object_keys(tv.client(), tv.prefix(), object_id)
            .await
            .into_iter()
            .map(|key| {
                let key = key.as_str_lossy().into_owned();
                key.strip_prefix(&full_prefix).unwrap_or(&key).to_string()
            })
            .collect();
    found.sort();
    found
}

/// Every `obj:*` family the arrangement must have materialized, as the literal
/// key suffix (exact match) or family prefix (`…:` — at least one member
/// required).
///
/// Hand-written literals rather than re-derived from `KeySchema`: a rename
/// must fail here rather than be masked by both sides calling the same
/// renamed builder.
const REQUIRED_FAMILIES: &[&str] = &[
    "obj:hdr",
    "obj:secs",
    "obj:sec:",
    "obj:flds",
    "obj:fld:",
    "obj:tss",
    "obj:ts:",
    "obj:dels",
    "obj:del:",
    // The b.9fn key — root parent id 0, member of no index.
    "obj:ch:0",
    // A per-section children list, which the pre-fix enumeration did reap.
    "obj:ch:1",
    // The b.jdr index itself. Its own absence would silently narrow the
    // tripwire back to the pre-b.jdr reconstruction.
    "obj:chs",
    // The b.jdr keys — children lists under dangling parents, in neither
    // `obj:secs` nor `obj:dels`. `ch:77` is written by `write_object`,
    // `ch:88` injected as an earlier build's `SetChildren` arm wrote it.
    "obj:ch:77",
    "obj:ch:88",
    "obj:stream",
    // b.75i — the UpdateLayer commit-token record. The arrangement's own
    // `update_layer` calls write it, so a retirement path that dropped it from
    // `object_cleanup_keys` shows up below as a leaked `obj:wtok`.
    "obj:wtok",
    // b.y3m Epic 3 — the snapshot fence and the writer-intent key, written by
    // the arrangement's `fence_and_intent`. Neither is named by any index, so
    // a retirement path that dropped them from `object_cleanup_keys` shows up
    // below as a leaked `obj:snapfence` / `obj:snapintent`.
    "obj:snapfence",
    "obj:snapintent",
];

/// Assert the arrangement lit up every `obj:*` family before retirement.
///
/// Without this the post-retirement assertions degrade silently: a family the
/// fixture stopped writing is a family the leak tripwire stops covering, and
/// nothing about the resulting green run says so.
async fn assert_full_key_surface_present(tv: &TestValkey, object_id: u64) {
    let present = surviving_obj_key_suffixes(tv, object_id).await;

    for family in REQUIRED_FAMILIES {
        let matched = if family.ends_with(':') {
            present.iter().any(|k| k.starts_with(family))
        } else {
            present.iter().any(|k| k == family)
        };
        assert!(
            matched,
            "fixture precondition: no `{family}` key was materialized before \
             retirement, so the post-retirement SCAN cannot detect a leak in \
             that family. Present: {present:?}"
        );
    }

    // Belt-and-suspenders on the one family the b.9fn fix is about: EXISTS
    // straight through the schema builder, so a SCAN that silently missed a
    // shard cannot fake the root children list into existence.
    let root_children = tv.key_schema().obj_children(&object_id.to_string(), "0");
    let exists: bool = tv
        .client()
        .pool
        .exists(&root_children)
        .await
        .expect("EXISTS must succeed");
    assert!(
        exists,
        "fixture precondition: root children list ({root_children}) must exist \
         before retirement"
    );

    // b.jdr: the tripwire below can only reap a dangling parent's children
    // list if the index actually recorded that parent — nothing else in the
    // keyspace names it. Assert the *contents*, not just the key's existence,
    // so a write surface that stopped SADDing degrades to a red precondition
    // here instead of a quietly narrowed leak assertion.
    let index_key = tv.key_schema().obj_children_index(&object_id.to_string());
    let mut indexed: Vec<String> = tv
        .client()
        .pool
        .smembers(&index_key)
        .await
        .expect("SMEMBERS obj:chs must succeed");
    indexed.sort();
    for (parent_id, surface) in [
        (DANGLING_PARENT_WRITE_OBJECT, "write_object"),
        (
            DANGLING_PARENT_MUTATION,
            "earlier-build SetChildren (injected)",
        ),
    ] {
        assert!(
            indexed.contains(&parent_id.to_string()),
            "fixture precondition: dangling parent {parent_id} must be recorded \
             in {index_key} by the `{surface}` write surface — its children \
             list is in no other index, so an unrecorded parent makes the \
             post-retirement SCAN blind to that leak. Indexed: {indexed:?}"
        );
    }
}

/// The one key family that legitimately outlives `cleanup_old_object`.
const EXPECTED_SURVIVOR: &str = "obj:stream";

fn assert_only_stream_survives(survivors: &[String], context: &str) {
    assert!(
        !survivors.iter().any(|k| k == "obj:ch:0"),
        "{context}: the root children list `obj:ch:0` must be UNLINKed. It is in \
         neither `obj:secs` nor `obj:dels`, so before b.9fn the index-walking \
         enumeration never produced it and it leaked one key per retired object \
         id forever. Survivors: {survivors:?}"
    );
    // b.jdr, named separately from the catch-all below so a regression says
    // *which* dangling-parent list survived.
    for (parent_id, surface) in [
        (DANGLING_PARENT_WRITE_OBJECT, "`write_object`"),
        (
            DANGLING_PARENT_MUTATION,
            "the injection (an earlier build's `SetChildren` arm)",
        ),
    ] {
        let key = format!("obj:ch:{parent_id}");
        assert!(
            !survivors.contains(&key),
            "{context}: the children list `{key}` under a DANGLING parent must \
             be UNLINKed. Parent {parent_id} names no section, so it is in \
             neither `obj:secs` nor `obj:dels` and the pre-b.jdr \
             sections ∪ tombstones ∪ root reconstruction could not produce this \
             key. It is reachable only via the `obj:chs` index, which \
             {surface} maintains — so this survivor means either that SADD or \
             the retirement-side SMEMBERS regressed. Survivors: {survivors:?}"
        );
    }
    // Subset, not set-equality. The property under test is "nothing but the
    // stream survives" — requiring `obj:stream` to still be *present* would
    // additionally couple the test to `STREAM_DELETE_GRACE_MS` (30 s), so a
    // contended run that stalled past the PEXPIRE between retirement and the
    // SCAN would go red for a reason that has nothing to do with key leaks.
    let leaked: Vec<&String> = survivors
        .iter()
        .filter(|k| *k != EXPECTED_SURVIVOR)
        .collect();
    assert!(
        leaked.is_empty(),
        "{context}: `obj:stream` is the ONLY key family that may outlive object \
         retirement (PEXPIRE'd with grace so lagging subscribers still observe \
         the `deleted` sentinel). Leaked key families missing from \
         `KeySchema::object_cleanup_keys`: {leaked:?} (all survivors: \
         {survivors:?})"
    );
}

// =============================================================================
// 1. CreateLayer overwrite — spawned, error-swallowing cleanup
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_overwrite_unlinks_root_and_dangling_children_lists_of_old_object() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;
    let layer_name = "b9fn-overwrite";

    // Arrange: materialize an object covering every `obj:*` family, then
    // tombstone a section so `obj:dels` / `obj:del:*` land too.
    let old_oid = ls
        .create_layer(
            &session_id,
            layer_name,
            "b-9fn-user",
            &state_with_full_key_surface(),
            mock_ext_store(),
        )
        .await
        .expect("initial create_layer must succeed");
    let _intent = arrange_full_key_surface(&tv, &ls, &session_id, layer_name, old_oid).await;

    // Act: overwrite the same (session, layer) pair. This mints a fresh
    // object id and spawns `cleanup_old_object` for the old one.
    let new_oid = ls
        .create_layer(
            &session_id,
            layer_name,
            "b-9fn-user",
            &state_with_full_key_surface(),
            mock_ext_store(),
        )
        .await
        .expect("overwrite create_layer must succeed");
    assert_ne!(new_oid, old_oid, "overwrite must mint a fresh object_id");

    // The overwrite cleanup is spawned on the cleanup tracker and does NOT
    // block `create_layer`'s return — join it, or the assertions race the
    // UNLINK.
    ls.wait_for_cleanup(CLEANUP_JOIN_BUDGET).await;

    // Assert.
    let survivors = surviving_obj_key_suffixes(&tv, old_oid).await;
    assert_only_stream_survives(&survivors, "after CreateLayer overwrite");
}

// =============================================================================
// 2. DeleteLayer — synchronous, error-propagating cleanup
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_layer_unlinks_root_and_dangling_children_lists() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;
    let layer_name = "b9fn-delete";

    // Arrange.
    let oid = ls
        .create_layer(
            &session_id,
            layer_name,
            "b-9fn-user",
            &state_with_full_key_surface(),
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");
    let _intent = arrange_full_key_surface(&tv, &ls, &session_id, layer_name, oid).await;

    // Act: DeleteLayer runs `cleanup_old_object` inline (SDD §7.11 step 4),
    // so no tracker join is needed for the cleanup itself. The
    // `wait_for_cleanup` below only drains any unrelated tracked task so the
    // SCAN sees a settled keyspace.
    ls.delete_layer(&session_id, layer_name, mock_ext_store())
        .await
        .expect("delete_layer must succeed");
    ls.wait_for_cleanup(CLEANUP_JOIN_BUDGET).await;

    // Assert.
    let survivors = surviving_obj_key_suffixes(&tv, oid).await;
    assert_only_stream_survives(&survivors, "after DeleteLayer");
}

// =============================================================================
// 3. b.jdr rolling deploy — retirement of an object that carries NO `obj:chs`
// =============================================================================

/// An object materialized by a **pre-b.jdr replica** has no `obj:chs` key: the
/// index ships with the fix and nothing backfills it, so during a rolling
/// deploy a post-b.jdr replica will retire objects whose children lists were
/// written by a replica that never SADDed a parent id. `SMEMBERS` of a missing
/// key is an empty array, not an error, so the enumeration in
/// `cleanup_old_object` degrades to `&[]` and `object_cleanup_keys` falls back
/// to the pre-b.jdr *sections ∪ tombstones ∪ root* reconstruction.
///
/// This test pins two things the two tripwires above cannot:
///
/// 1. **Cleanup tolerates an absent index end-to-end.** The `SMEMBERS
///    obj:chs` added to both retirement paths propagates its errors rather
///    than `unwrap_or_default()`ing them (b.u96 item 1), so "key missing" has
///    to be distinguishable from "call failed" at the fred layer for the
///    rolling-deploy story to hold at all. Here it is exercised against a live
///    Valkey rather than argued from the API contract.
/// 2. **Exactly how much the retained reconstruction still covers.** That
///    fallback is not dead code kept for symmetry — it is what bounds the leak
///    for pre-upgrade objects, and `object_cleanup_keys` keeps emitting the
///    root key unconditionally *because of* this case. So the surviving set is
///    asserted precisely: `obj:ch:0` (unconditional) and `obj:ch:1`
///    (section-derived, parent 1 is in `obj:secs`) are still reaped, while
///    `obj:ch:77` / `obj:ch:88` — the dangling parents — leak.
///
/// **Those two survivors are the expected, pre-fix behaviour, not a bug this
/// test tolerates.** Without the index there is nothing anywhere in the
/// keyspace that names parent 77 or 88, so no cleanup path can reach them; the
/// residual leak is bounded by the pre-upgrade objects and ends when they are
/// retired. If a future change makes dangling parents reachable without the
/// index, this test goes red and should be *tightened*, not deleted.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_layer_without_a_children_index_still_reaps_the_reconstructible_children_lists() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;
    let layer_name = "bjdr-no-index";

    // Arrange: the same full key surface as tests 1 and 2 — including the
    // index-precondition assertions, which prove the index really was there
    // before the DEL below removes it.
    let oid = ls
        .create_layer(
            &session_id,
            layer_name,
            "b-9fn-user",
            &state_with_full_key_surface(),
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");
    let _intent = arrange_full_key_surface(&tv, &ls, &session_id, layer_name, oid).await;

    // Simulate the pre-b.jdr materialization: drop the index, keep every
    // children list it named. This is the only way to reach the state — the
    // current write surfaces maintain the index inside the same MULTI/EXEC as
    // the lists, so no sequence of RPCs can produce lists without it.
    let index_key = tv.key_schema().obj_children_index(&oid.to_string());
    let removed: i64 = tv
        .client()
        .pool
        .del(&index_key)
        .await
        .expect("DEL obj:chs must succeed");
    assert_eq!(
        removed, 1,
        "arrangement: the index must have existed before this DEL, otherwise \
         the fallback below is not the thing under test"
    );

    // Act.
    ls.delete_layer(&session_id, layer_name, mock_ext_store())
        .await
        .expect(
            "delete_layer must succeed with no `obj:chs`: SMEMBERS of a missing \
             key is an empty array, not an error, so the enumeration degrades \
             instead of failing the RPC",
        );
    ls.wait_for_cleanup(CLEANUP_JOIN_BUDGET).await;

    // Assert.
    let survivors = surviving_obj_key_suffixes(&tv, oid).await;

    // The reconstruction's coverage, positively: the root key (emitted
    // unconditionally, which is exactly the b.9fn cover this case justifies)
    // and every parent that is a live section or a tombstone.
    for (key, why) in [
        (
            "obj:ch:0",
            "the synthetic root, emitted unconditionally by \
             `object_cleanup_keys` precisely so pre-b.jdr objects keep their \
             b.9fn fix",
        ),
        (
            "obj:ch:1",
            "parent 1 is a live section, so the sections-index walk names it \
             without any help from `obj:chs`",
        ),
    ] {
        assert!(
            !survivors.contains(&key.to_string()),
            "an absent `obj:chs` must not cost the reconstruction its own \
             coverage: `{key}` must still be UNLINKed — {why}. Survivors: \
             {survivors:?}"
        );
    }

    // The reconstruction's ceiling, negatively: dangling parents are
    // unreachable without the index, and that residual leak is what the index
    // exists to close for objects written by an upgraded replica.
    let dangling_keys: Vec<String> = [DANGLING_PARENT_WRITE_OBJECT, DANGLING_PARENT_MUTATION]
        .iter()
        .map(|parent_id| format!("obj:ch:{parent_id}"))
        .collect();
    for (parent_id, key) in [DANGLING_PARENT_WRITE_OBJECT, DANGLING_PARENT_MUTATION]
        .iter()
        .zip(&dangling_keys)
    {
        assert!(
            survivors.contains(key),
            "PRE-FIX BEHAVIOUR, asserted deliberately: with no `obj:chs`, \
             parent {parent_id} is named nowhere in the keyspace, so `{key}` \
             is expected to survive retirement. Its absence means dangling \
             parents became reachable some other way — a real improvement, but \
             one that changes what this test is pinning, so tighten the \
             assertion rather than deleting it. Survivors: {survivors:?}"
        );
    }

    // Nothing else may leak. Subset rather than set-equality for `obj:stream`
    // for the same reason as `assert_only_stream_survives` — it is PEXPIRE'd
    // with a 30 s grace, and a contended run could outlive it.
    let unexpected: Vec<&String> = survivors
        .iter()
        .filter(|k| *k != EXPECTED_SURVIVOR && !dangling_keys.contains(k))
        .collect();
    assert!(
        unexpected.is_empty(),
        "with no `obj:chs`, the pre-b.jdr reconstruction must still reap every \
         `obj:*` family it can name — only the two dangling-parent children \
         lists and the grace-expiring stream may remain. Unexpectedly leaked: \
         {unexpected:?} (all survivors: {survivors:?})"
    );
}

// =============================================================================
// 4. b.jdr — the index's write-side skip: an empty children vec is not indexed
// =============================================================================

/// Sorted `SMEMBERS le:{oid}:obj:chs`.
async fn children_index_members(tv: &TestValkey, object_id: u64) -> Vec<String> {
    let mut members: Vec<String> = tv
        .client()
        .pool
        .smembers(tv.key_schema().obj_children_index(&object_id.to_string()))
        .await
        .expect("SMEMBERS obj:chs must succeed");
    members.sort();
    members
}

async fn children_key_exists(tv: &TestValkey, object_id: u64, parent_id: u64) -> bool {
    tv.client()
        .pool
        .exists(
            tv.key_schema()
                .obj_children(&object_id.to_string(), &parent_id.to_string()),
        )
        .await
        .expect("EXISTS obj:ch:{parent} must succeed")
}

/// Root → `[1, 5]`, section 1 → `[2]`, section 5 → `[]` — an **explicit empty
/// vec**, the shape `remove_child_from_parent` legitimately leaves behind in
/// `state.children` when the last child leaves a parent (and which
/// `apply_checkpoint`'s restore therefore replays into `write_object`).
///
/// Shared by the write-side test directly below and by the SREM-fork test
/// after it: the same three-section tree is both the arrangement whose
/// *initial* index contents are under test, and the starting position the
/// MoveSection / DeleteSection sequence mutates.
fn state_with_an_empty_children_vec() -> ObjectState {
    let mut state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usda".to_string(),
        next_set_order: 10,
        ..Default::default()
    };
    for (section_id, parent_id, name, order) in [
        (1_u64, 0_u64, "World", 0_u32),
        (2, 1, "Cube", 0),
        (5, 0, "Sibling", 1),
    ] {
        state.sections.insert(
            section_id,
            Section {
                section_id,
                parent_id,
                name: name.to_string(),
                section_type: 1,
                section_order: order,
                first_version: 1,
                last_version: 1,
            },
        );
    }
    state.children.insert(0, vec![1, 5]);
    state.children.insert(1, vec![2]);
    state.children.insert(5, Vec::new());
    state
}

/// The write-side half of the *membership ⟺ key existence* contract, on the
/// one input where the two can diverge: a parent carrying an empty child vec.
/// `write_object`'s children loop RPUSHes nothing for it, so an unconditional
/// SADD would index a parent whose key does not exist — breaking the contract
/// on the very first write, before any mutation has run.
///
/// Separate from the SREM-fork test below because it is a different surface
/// (`write_object`'s `is_empty` skip vs. `execute_mutation`'s `SetChildren` /
/// `DeleteChildrenList` arms) reached by a different call (`create_layer` vs.
/// `update_layer`), and because `write_object` is also the checkpoint-restore
/// materializer: a restored object must come back with the same index contents
/// the live one had, and this is the input where that is not automatic.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn write_object_does_not_index_a_parent_with_an_empty_children_vec() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;

    let oid = ls
        .create_layer(
            &session_id,
            "bjdr-empty-child-vec",
            "b-9fn-user",
            &state_with_an_empty_children_vec(),
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");

    assert_eq!(
        children_index_members(&tv, oid).await,
        vec!["0".to_string(), "1".to_string()],
        "write_object must index exactly the parents that own a NON-EMPTY \
         children list — section 5's entry is an empty vec, so no `obj:ch:5` \
         key is RPUSHed and indexing it would break the membership ⟺ key \
         existence contract on the very first write"
    );
    assert!(
        !children_key_exists(&tv, oid, 5).await,
        "an empty children vec must materialize no key — the premise of the \
         assertion above"
    );
    // The non-empty parents are asserted positively too: an index that matched
    // the expectation above by SADDing nothing at all would otherwise pass.
    for parent_id in [0_u64, 1] {
        assert!(
            children_key_exists(&tv, oid, parent_id).await,
            "parent {parent_id} owns a non-empty children list, so \
             `obj:ch:{parent_id}` must exist alongside its index member"
        );
    }
}

// =============================================================================
// 5. b.jdr — the index's SREM forks: a parent whose list went away must not
//    stay indexed
// =============================================================================

/// The `obj:chs` contract is *membership ⟺ key existence*. The leak tripwires
/// exercise its SADD half and the test above exercises the write-side skip;
/// this pins the two SREM halves — the forks in `execute_mutation` where a
/// children list stops existing:
///
/// - **`SetChildren` with an empty vec.** `remove_child_from_parent` emits
///   exactly this when the last child leaves a parent; the arm DELs the key
///   and RPUSHes nothing, so an unconditional SADD would index a parent whose
///   key does not exist.
/// - **`DeleteChildrenList`.** `cascade_delete_section` emits one per section
///   in the doomed subtree.
///
/// Why it matters even though a stale member is *individually* benign (it
/// costs one UNLINK of a nonexistent key at retirement): the set is unbounded
/// and long-lived. A hot layer that re-parents sections all session accretes
/// one permanent `obj:chs` member per parent that ever held a child, on a
/// `noeviction` store — the same unbounded-growth shape b.9fn and b.jdr were
/// filed for, just relocated from the key space into one set.
///
/// Both commands are structural, so both commits route down the
/// WATCH/MULTI/EXEC slow path where `execute_mutation` lives.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn children_index_drops_parents_whose_list_was_emptied_or_deleted() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;
    let layer_name = "bjdr-srem-forks";

    // Arrange: root → [1, 5], section 1 → [2], section 5 → []. What
    // `write_object` makes of that empty vec is the subject of
    // `write_object_does_not_index_a_parent_with_an_empty_children_vec` above;
    // here it is only the starting position the two mutations below move from.
    let oid = ls
        .create_layer(
            &session_id,
            layer_name,
            "b-9fn-user",
            &state_with_an_empty_children_vec(),
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");

    assert!(
        children_index_members(&tv, oid)
            .await
            .contains(&"1".to_string()),
        "precondition: parent 1 owns [2] at create time, so it must start out \
         indexed — the SREM fork below only means anything if there is a \
         member for it to remove"
    );

    // Act 1 — SetChildren-with-empty-vec fork. Moving section 2 from parent 1
    // to parent 5 empties parent 1's list (`remove_child_from_parent`) and
    // creates parent 5's (`insert_child_into_parent`), so this single commit
    // drives the SREM and the SADD forks of the same arm at once.
    let move_delta = OwnedDelta {
        struct_commands: vec![StructCommand::MoveSection(CommandMoveSection {
            old_parent_id: 1,
            new_parent_id: 5,
            section_id: 2,
            new_name: "Cube".to_string(),
        })],
        ..Default::default()
    };
    ls.update_layer_relayed(
        &session_id,
        layer_name,
        oid,
        Some(move_delta),
        "b-9fn-writer",
        0,
    )
    .await
    .expect("MoveSection update_layer must succeed");

    assert!(
        !children_key_exists(&tv, oid, 1).await,
        "precondition: moving the only child away must DEL `obj:ch:1`"
    );
    assert!(
        children_key_exists(&tv, oid, 5).await,
        "precondition: the move must RPUSH `obj:ch:5`"
    );
    assert_eq!(
        children_index_members(&tv, oid).await,
        vec!["0".to_string(), "5".to_string()],
        "`SetChildren` with an empty vec must SREM the parent from `obj:chs`: \
         parent 1's key was DELed, and membership tracks key existence. \
         Leaving `1` behind accretes a permanent member per re-parented \
         parent for the object's whole life"
    );

    // Act 2 — DeleteChildrenList fork. Deleting section 5 cascades over the
    // subtree {5, 2}, emitting a `DeleteChildrenList` per section.
    let delete_delta = OwnedDelta {
        struct_commands: vec![StructCommand::DeleteSection(CommandDeleteSection {
            parent_id: 0,
            section_id: 5,
        })],
        ..Default::default()
    };
    ls.update_layer_relayed(
        &session_id,
        layer_name,
        oid,
        Some(delete_delta),
        "b-9fn-writer",
        0,
    )
    .await
    .expect("DeleteSection update_layer must succeed");

    assert!(
        !children_key_exists(&tv, oid, 5).await,
        "precondition: the cascade must DEL `obj:ch:5`"
    );
    assert_eq!(
        children_index_members(&tv, oid).await,
        vec!["0".to_string()],
        "`DeleteChildrenList` must SREM the parent from `obj:chs`. Only the \
         root survives — it still owns [1]. (The cascade also emits a \
         `DeleteChildrenList` for section 2, whose id was never a member; \
         that SREM is a harmless no-op, which is the benign-drift direction \
         the index's contract allows.)"
    );
    assert!(
        children_key_exists(&tv, oid, 0).await,
        "the root children list must be untouched by either fork"
    );
}
