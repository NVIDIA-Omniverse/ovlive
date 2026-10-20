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

//! The planning engine's writer: every write `domain::delta` makes to an
//! object's state goes through a function here. That covers section rows,
//! children lists, value rows and the header scalars, and each function
//! reports its write to the plan's [`EngineHooks`].
//!
//! One choke point serves two kinds of plan:
//!
//! - **Every caller but the structural cache** plans with [`NoHooks`]: every
//!   hook is an empty `#[inline(always)]` body behind
//!   [`EngineHooks::OBSERVES`] `= false`, so each writer compiles to the bare
//!   map operation it wraps and nothing is cloned for a hook.
//! - **The structural cache** plans in place on its held entry, and its
//!   stream repair replays foreign commits into it, with hooks that keep the
//!   entry's persistent plan indexes (`domain::plan_index::IndexedHooks`)
//!   and, for a plan, its pre-image journal (`domain::plan_journal`). Those
//!   hooks must see every write, or an index or a rollback silently drifts
//!   from the state. So no write in
//!   `domain::delta` may bypass this module: a guard test there
//!   (`no_direct_table_or_header_mutation_outside_the_writer_module`) fails
//!   on any direct table or header mutation in that file.
//!
//! Writers that remove a key from one of the state's hash tables, or make an
//! insert call on one, also report it with the table's room before the write
//! ([`EngineHooks::removing`], [`EngineHooks::insert_call`]), so a journalled
//! plan can charge the table rebuild its rollback may need before the write
//! that can make it necessary (`domain::plan_journal`, "Rollback without
//! growth"). A removal also carries the table's capacity before the write
//! (inside a purge, before the purge began), which the plan indexes' capacity
//! high-water samples so the entry is never weighed below the buckets a
//! tombstone hides (`domain::plan_index::PlanIndexes::held_capacity`).
//!
//! Each writer reports the value a key held before its write and the value
//! it holds after. For section rows and children lists that is a
//! `*_before` / `*_after` pair around the write, so an in-place edit needs
//! no clone. For value rows it is one event carrying both, because the row
//! store reports what it replaced. A reset is the exception. With hooks it
//! moves the whole tables out at once ([`take_for_reset`]) and hands them to
//! the hooks after its deletes are emitted ([`finish_reset`]), so hooks that
//! keep the pre-reset state can take it whole, in O(1), rather than key by
//! key. Without hooks it is the engine's in-place drain, allocating nothing.
//!
//! Pure logic: zero I/O, zero async (SDD §9).

use std::collections::{HashMap, HashSet};

use ordered_float::OrderedFloat;

use crate::domain::types::{
    Field, FieldRows, FieldValue, NameId, ObjectStateOf, SampleRows, Section, TimeSample, ValueRows,
};

/// What a value-row write did to the interned field names (the compact form
/// only; the full form interns nothing).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NameChange {
    /// No name was interned or released.
    Unchanged,
    /// The write interned a name no row used before.
    Interned,
    /// The write removed the last row using its name, which was released.
    Released,
}

/// The header scalars as they stand before a header write
/// ([`EngineHooks::header_before`]).
#[derive(Debug, Clone, Copy)]
pub struct HeaderImage<'s> {
    pub sequence_num: u64,
    pub first_version: u64,
    pub next_set_order: u64,
    pub source_format: &'s String,
}

impl<'s> HeaderImage<'s> {
    fn of<R: ValueRows>(state: &'s ObjectStateOf<R>) -> Self {
        Self {
            sequence_num: state.sequence_num,
            first_version: state.first_version,
            next_set_order: state.next_set_order,
            source_format: &state.source_format,
        }
    }
}

/// A hash table of a structural cache entry: the state's four tables, the
/// field-name interner's name → id table, and the plan indexes' two maps
/// (`domain::plan_index`). The registry every per-table rule iterates: a
/// plan's rollback rebuilds a table that lacks the room to re-create the keys
/// the plan removed (`domain::plan_journal`), and the entry's weighers read
/// each table at its held capacity (`domain::plan_index::PlanIndexes::held_capacity`).
/// A new hash table in the state or the indexes is one more variant here,
/// in [`Self::ALL`] (the state's before the indexes'), with its arm in each
/// exhaustive per-table lookup — [`Self::in_state`],
/// `domain::plan_index::state_table_shape` or `PlanIndexes::own_table_shape`,
/// `domain::plan_index::table_slot_bytes`,
/// `storage::structural_cache::PlannedInserts::calls` — and every weigher
/// then reads it at its held capacity. Two obligations come with it:
///
/// - a removal from it notes its capacity first, as every removal does
///   (`PlanIndexes::note_capacity`, through the writer's removal hook for a
///   state table);
/// - the cache's compaction shrinks it to fit
///   (`storage::structural_cache::compact_state` for the state's tables,
///   `PlanIndexes::compact` for the indexes'), since the high-water reset at
///   compaction is sound only for a table compaction left minimal; the
///   test-support settle check asserts it.
///
/// A map field added without a variant fails the test-support guard's build
/// (`domain::plan_index::hash_tables_for_test`) until it is classified.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RebuildTable {
    Sections,
    Children,
    FieldRows,
    SampleRows,
    NameIds,
    NameIndex,
    ExtCounts,
}

impl RebuildTable {
    /// Every table, in [`Self::index`] order.
    pub const ALL: [RebuildTable; 7] = [
        RebuildTable::Sections,
        RebuildTable::Children,
        RebuildTable::FieldRows,
        RebuildTable::SampleRows,
        RebuildTable::NameIds,
        RebuildTable::NameIndex,
        RebuildTable::ExtCounts,
    ];

    /// The number of tables: the size of every per-table array.
    pub const COUNT: usize = Self::ALL.len();

    /// The number of the state's own tables ([`Self::in_state`]), which come
    /// first in [`Self::ALL`]: indexes `0..STATE_COUNT`.
    pub const STATE_COUNT: usize = {
        let mut count = 0;
        let mut i = 0;
        while i < Self::ALL.len() {
            if Self::ALL[i].in_state() {
                count += 1;
            }
            i += 1;
        }
        count
    };

    /// The table's position in [`Self::ALL`].
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Whether the table is the state's own (or its interner's), not one of
    /// the plan indexes' maps: the tables an entry without plan indexes
    /// holds.
    pub const fn in_state(self) -> bool {
        match self {
            RebuildTable::Sections
            | RebuildTable::Children
            | RebuildTable::FieldRows
            | RebuildTable::SampleRows
            | RebuildTable::NameIds => true,
            RebuildTable::NameIndex | RebuildTable::ExtCounts => false,
        }
    }
}

// `ALL` lists each table at its `index()`, and the state's tables first, so
// a per-table array is indexed by `index()` and a per-state-table one by the
// same index below `STATE_COUNT`.
const _: () = {
    let mut i = 0;
    while i < RebuildTable::ALL.len() {
        assert!(RebuildTable::ALL[i].index() == i);
        assert!(RebuildTable::ALL[i].in_state() == (i < RebuildTable::STATE_COUNT));
        i += 1;
    }
};

/// The key a write is about to remove from one of the state's tables
/// ([`EngineHooks::removing`]).
#[derive(Debug, Clone, Copy)]
pub(crate) enum RemovedKey<'k> {
    Section(u64),
    List(u64),
    Field(u64, &'k str),
    Sample(u64, u64),
    /// An interned field name, released by the last row using it, and the id
    /// it held.
    FieldName(&'k str, NameId),
}

impl RemovedKey<'_> {
    /// The table the key leaves.
    pub(crate) fn table(self) -> RebuildTable {
        match self {
            RemovedKey::Section(_) => RebuildTable::Sections,
            RemovedKey::List(_) => RebuildTable::Children,
            RemovedKey::Field(..) => RebuildTable::FieldRows,
            RemovedKey::Sample(..) => RebuildTable::SampleRows,
            RemovedKey::FieldName(..) => RebuildTable::NameIds,
        }
    }
}

/// A hash map's room: the entries it takes before its next insert call
/// would grow it.
fn map_room<K, V, S>(map: &HashMap<K, V, S>) -> usize {
    map.capacity().saturating_sub(map.len())
}

/// The tables a reset moved out of the state ([`take_for_reset`]): every row
/// and list the object held before it.
///
/// An un-hooked reset moves only the section and children maps out, and
/// drains the value rows in place, so its value-row stores are `None` here.
pub(crate) struct ResetTables<R: ValueRows> {
    pub(crate) sections: HashMap<u64, Section>,
    pub(crate) children: HashMap<u64, Vec<u64>>,
    pub(crate) fields: Option<R::Fields>,
    pub(crate) time_samples: Option<R::Samples>,
}

impl<R: ValueRows> ResetTables<R> {
    /// An un-hooked reset's drain of the section rows, handing each over
    /// owned, in the order iterating them would.
    pub(crate) fn drain_sections(&mut self) -> std::collections::hash_map::Drain<'_, u64, Section> {
        self.sections.drain()
    }

    /// An un-hooked reset's clear of the children lists.
    pub(crate) fn clear_children(&mut self) {
        self.children.clear();
    }
}

/// A reset's field-row deletes: `on_removed(section_id, key_name)` once per
/// row, in drain order. Read from the moved-out store when the hooks took
/// one; otherwise drained from `state` in place ([`FieldRows::drain_rows`]).
pub(crate) fn reset_fields<R: ValueRows>(
    state: &mut ObjectStateOf<R>,
    taken: &ResetTables<R>,
    on_removed: impl FnMut(u64, &str),
) {
    match &taken.fields {
        Some(fields) => fields.for_each_in_drain_order(on_removed),
        None => state.fields.drain_rows(on_removed),
    }
}

/// A reset's time-sample deletes, as [`reset_fields`].
pub(crate) fn reset_samples<R: ValueRows>(
    state: &mut ObjectStateOf<R>,
    taken: &ResetTables<R>,
    on_removed: impl FnMut(u64, OrderedFloat<f64>),
) {
    match &taken.time_samples {
        Some(samples) => samples.for_each_in_drain_order(on_removed),
        None => state.time_samples.drain_rows(on_removed),
    }
}

/// What a plan reports each write to. Every method has an empty default body,
/// so an implementation overrides only the events it reads. The writer calls
/// a hook only when [`Self::OBSERVES`] is set.
pub(crate) trait EngineHooks<R: ValueRows> {
    /// Whether the writer calls the hooks at all. `false` for [`NoHooks`], so
    /// each writer compiles to its bare map operation.
    const OBSERVES: bool;

    /// The row `id` is about to be written: `old` is what it holds now
    /// (`None` when it has no row).
    #[inline(always)]
    fn section_before(&mut self, _id: u64, _old: Option<&Section>) {}

    /// The row `id` was written: `new` is what it holds now (`None` once
    /// removed).
    #[inline(always)]
    fn section_after(&mut self, _id: u64, _new: Option<&Section>) {}

    /// The children list keyed `parent` is about to be written: `old` is the
    /// list now (`None` when there is none).
    #[inline(always)]
    fn list_before(&mut self, _parent: u64, _old: Option<&Vec<u64>>) {}

    /// The children list keyed `parent` was written: `new` is the list now
    /// (`None` once removed).
    #[inline(always)]
    fn list_after(&mut self, _parent: u64, _new: Option<&Vec<u64>>) {}

    /// `child` was inserted into `parent`'s children list: an id that list
    /// did not hold. The only write that can make a section reachable, or
    /// deeper, in the section tree.
    #[inline(always)]
    fn child_listed(&mut self, _parent: u64, _child: u64) {}

    /// The field row `(section_id, name)` was written: `old` and `new` are
    /// its external hash before and after (`None` for no row, `Some(None)`
    /// for an inline row).
    #[inline(always)]
    fn field_written(
        &mut self,
        _section_id: u64,
        _name: &str,
        _old: Option<Option<[u8; 32]>>,
        _new: Option<Option<[u8; 32]>>,
        _names: NameChange,
    ) {
    }

    /// The time-sample row `(section_id, time_bits)` was written, as
    /// [`Self::field_written`].
    #[inline(always)]
    fn sample_written(
        &mut self,
        _section_id: u64,
        _time_bits: u64,
        _old: Option<Option<[u8; 32]>>,
        _new: Option<Option<[u8; 32]>>,
    ) {
    }

    /// A header scalar is about to be written: `header` is every scalar as
    /// it stands now.
    #[inline(always)]
    fn header_before(&mut self, _header: HeaderImage<'_>) {}

    /// `source_format` was written: `new` is its value now.
    #[inline(always)]
    fn source_format_after(&mut self, _new: &String) {}

    /// A reset moved these tables out of the state, which now holds none of
    /// them ([`take_for_reset`]); its deletes are already emitted. The
    /// default drops them.
    #[inline(always)]
    fn reset(&mut self, _taken: ResetTables<R>) {}

    /// An insert call is about to land on the state table `table` (new key
    /// or overwrite: std's `HashMap::insert` reserves a slot either way),
    /// which takes `room` more entries before it grows.
    #[inline(always)]
    fn insert_call(&mut self, _table: RebuildTable, _room: usize) {}

    /// `key` is about to be removed from its state table, which takes `room`
    /// more entries before it grows and reports `capacity` usable slots
    /// (`HashMap::capacity`) before the removal (inside a purge, both
    /// measured before the purge began: a removal never lowers the room nor
    /// raises the capacity).
    /// Called after the write's `*_before` / `*_written` event for the key,
    /// so a journal has recorded its pre-image. The capacity is what the
    /// plan indexes' capacity high-water samples, before the removal can
    /// leave a tombstone that hides buckets from it
    /// (`domain::plan_index::PlanIndexes::note_capacity`).
    #[inline(always)]
    fn removing(&mut self, _key: RemovedKey<'_>, _room: usize, _capacity: usize) {}

    // The lookups below are not writes: they let hooks that keep persistent
    // indexes answer what the engine otherwise rebuilds per delta. Each
    // default answers "not kept here", and the engine uses its own.

    /// `domain::delta::find_child_by_name`'s answer — the lowest id among
    /// the rows of `sections` naming `(parent_id, name)`, other than
    /// `except_id` — from a persistent index: `Some(answer)`, or `None` when
    /// the hooks keep none.
    #[inline(always)]
    fn child_by_name(
        &self,
        _sections: &HashMap<u64, Section>,
        _parent_id: u64,
        _name: &str,
        _except_id: Option<u64>,
    ) -> Option<Option<u64>> {
        None
    }

    /// Push the id of every row naming `parent_id`, listed or not, onto
    /// `out` from a persistent index and answer `true`; `false` (and nothing
    /// pushed) when the hooks keep none.
    #[inline(always)]
    fn rows_naming(&self, _parent_id: u64, _out: &mut Vec<u64>) -> bool {
        false
    }

    /// The delta's end-of-delta section-tree depth verdict, from an
    /// incremental check: `Some(verdict)`, or `None` when the hooks keep none
    /// and the engine runs its full walk. `walk(n)` is called before the
    /// check's walk grows by `n` sections, so a bounded plan's charge covers
    /// it first.
    #[inline(always)]
    fn depth_verdict(
        &mut self,
        _state: &ObjectStateOf<R>,
        _walk: &mut dyn FnMut(u64),
    ) -> Option<Result<(), crate::domain::delta::DeltaError>> {
        None
    }
}

/// The hooks of every plan outside the structural cache: none.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct NoHooks;

impl<R: ValueRows> EngineHooks<R> for NoHooks {
    const OBSERVES: bool = false;
}

// ── Section rows ──────────────────────────────────────────────────────────────

/// Insert `section` under its `section_id`, returning the row it replaced.
pub(crate) fn insert_section<R: ValueRows, H: EngineHooks<R>>(
    state: &mut ObjectStateOf<R>,
    hooks: &mut H,
    section: Section,
) -> Option<Section> {
    let id = section.section_id;
    if H::OBSERVES {
        hooks.insert_call(RebuildTable::Sections, map_room(&state.sections));
        hooks.section_before(id, state.sections.get(&id));
    }
    let old = state.sections.insert(id, section);
    if H::OBSERVES {
        hooks.section_after(id, state.sections.get(&id));
    }
    old
}

/// Remove the row `id`, returning it; nothing is reported when there is none.
pub(crate) fn remove_section<R: ValueRows, H: EngineHooks<R>>(
    state: &mut ObjectStateOf<R>,
    hooks: &mut H,
    id: u64,
) -> Option<Section> {
    if H::OBSERVES {
        let room = map_room(&state.sections);
        let capacity = state.sections.capacity();
        let old = state.sections.get(&id)?;
        hooks.section_before(id, Some(old));
        hooks.removing(RemovedKey::Section(id), room, capacity);
    }
    let old = state.sections.remove(&id);
    if H::OBSERVES {
        hooks.section_after(id, None);
    }
    old
}

/// Edit the row `id` in place through `edit`, returning what it returns;
/// `None` (and nothing reported) when there is no such row. `edit` must not
/// change the row's `section_id`.
pub(crate) fn update_section<R: ValueRows, H: EngineHooks<R>, T>(
    state: &mut ObjectStateOf<R>,
    hooks: &mut H,
    id: u64,
    edit: impl FnOnce(&mut Section) -> T,
) -> Option<T> {
    let row = state.sections.get_mut(&id)?;
    if H::OBSERVES {
        hooks.section_before(id, Some(row));
    }
    let out = edit(row);
    debug_assert_eq!(row.section_id, id, "a row edit changed its section_id");
    if H::OBSERVES {
        hooks.section_after(id, Some(row));
    }
    Some(out)
}

// ── Children lists ────────────────────────────────────────────────────────────

/// Make sure `parent` has a children list, creating an empty one if not.
pub(crate) fn ensure_list<R: ValueRows, H: EngineHooks<R>>(
    state: &mut ObjectStateOf<R>,
    hooks: &mut H,
    parent: u64,
) {
    if state.children.contains_key(&parent) {
        return;
    }
    if H::OBSERVES {
        hooks.insert_call(RebuildTable::Children, map_room(&state.children));
        hooks.list_before(parent, None);
    }
    state.children.insert(parent, Vec::new());
    if H::OBSERVES {
        hooks.list_after(parent, state.children.get(&parent));
    }
}

/// Install `list` as `parent`'s children list, returning the list it
/// replaced.
pub(crate) fn set_list<R: ValueRows, H: EngineHooks<R>>(
    state: &mut ObjectStateOf<R>,
    hooks: &mut H,
    parent: u64,
    list: Vec<u64>,
) -> Option<Vec<u64>> {
    if H::OBSERVES {
        hooks.insert_call(RebuildTable::Children, map_room(&state.children));
        hooks.list_before(parent, state.children.get(&parent));
    }
    let old = state.children.insert(parent, list);
    if H::OBSERVES {
        hooks.list_after(parent, state.children.get(&parent));
    }
    old
}

/// Remove `parent`'s children list, returning it; nothing is reported when
/// there is none.
pub(crate) fn remove_list<R: ValueRows, H: EngineHooks<R>>(
    state: &mut ObjectStateOf<R>,
    hooks: &mut H,
    parent: u64,
) -> Option<Vec<u64>> {
    if H::OBSERVES {
        let room = map_room(&state.children);
        let capacity = state.children.capacity();
        let old = state.children.get(&parent)?;
        hooks.list_before(parent, Some(old));
        hooks.removing(RemovedKey::List(parent), room, capacity);
    }
    let old = state.children.remove(&parent);
    if H::OBSERVES {
        hooks.list_after(parent, None);
    }
    old
}

/// Edit `parent`'s children list in place through `edit`, returning what it
/// returns; `None` (and nothing reported) when there is no list. An edit that
/// adds an id the list did not hold goes through [`insert_listed_child`]
/// instead, which also reports the listing.
pub(crate) fn update_list<R: ValueRows, H: EngineHooks<R>, T>(
    state: &mut ObjectStateOf<R>,
    hooks: &mut H,
    parent: u64,
    edit: impl FnOnce(&mut Vec<u64>) -> T,
) -> Option<T> {
    let list = state.children.get_mut(&parent)?;
    if H::OBSERVES {
        hooks.list_before(parent, Some(list));
    }
    let out = edit(list);
    if H::OBSERVES {
        hooks.list_after(parent, Some(list));
    }
    Some(out)
}

/// [`update_list`] with the section rows lent beside the list, for an edit
/// that reads them while it writes the list (a stable sort by stored key).
/// `edit` may not change which ids the list holds.
pub(crate) fn update_list_with_sections<R: ValueRows, H: EngineHooks<R>, T>(
    state: &mut ObjectStateOf<R>,
    hooks: &mut H,
    parent: u64,
    edit: impl FnOnce(&mut Vec<u64>, &HashMap<u64, Section>) -> T,
) -> Option<T> {
    let ObjectStateOf {
        sections, children, ..
    } = state;
    let list = children.get_mut(&parent)?;
    if H::OBSERVES {
        hooks.list_before(parent, Some(list));
    }
    let out = edit(list, sections);
    if H::OBSERVES {
        hooks.list_after(parent, Some(list));
    }
    Some(out)
}

/// Insert `child`, an id `parent`'s list does not hold, at `position` in it.
/// The one write that lists an id, so the one that reports
/// [`EngineHooks::child_listed`]. The list must exist ([`ensure_list`]).
pub(crate) fn insert_listed_child<R: ValueRows, H: EngineHooks<R>>(
    state: &mut ObjectStateOf<R>,
    hooks: &mut H,
    parent: u64,
    position: usize,
    child: u64,
) {
    let inserted = update_list(state, hooks, parent, |list| list.insert(position, child));
    debug_assert!(
        inserted.is_some(),
        "insert_listed_child under a parent with no list"
    );
    if H::OBSERVES && inserted.is_some() {
        hooks.child_listed(parent, child);
    }
}

// ── Value rows ────────────────────────────────────────────────────────────────

fn external_hash(value: &FieldValue) -> Option<[u8; 32]> {
    match value {
        FieldValue::External { hash, .. } => Some(*hash),
        FieldValue::Inline(_) => None,
    }
}

/// Insert `field`, replacing the row with its key.
pub(crate) fn insert_field<R: ValueRows, H: EngineHooks<R>>(
    state: &mut ObjectStateOf<R>,
    hooks: &mut H,
    field: Field,
) {
    if !H::OBSERVES {
        state.fields.insert_row(field);
        return;
    }
    let section_id = field.section_id;
    let name = field.key_name.clone();
    let new = external_hash(&field.value);
    hooks.insert_call(RebuildTable::FieldRows, state.fields.row_room());
    if state.fields.interns(&name) {
        hooks.insert_call(RebuildTable::NameIds, state.fields.name_room());
    }
    let write = state.fields.insert_row(field);
    let names = if write.name_interned {
        NameChange::Interned
    } else {
        NameChange::Unchanged
    };
    hooks.field_written(section_id, &name, write.replaced, Some(new), names);
}

/// Insert `sample`, replacing the row with its key.
pub(crate) fn insert_sample<R: ValueRows, H: EngineHooks<R>>(
    state: &mut ObjectStateOf<R>,
    hooks: &mut H,
    sample: TimeSample,
) {
    if !H::OBSERVES {
        state.time_samples.insert_row(sample);
        return;
    }
    let section_id = sample.section_id;
    let time_bits = sample.time.into_inner().to_bits();
    let new = external_hash(&sample.value);
    hooks.insert_call(RebuildTable::SampleRows, state.time_samples.row_room());
    let replaced = state.time_samples.insert_row(sample);
    hooks.sample_written(section_id, time_bits, replaced, Some(new));
}

/// Remove every field and time-sample row of the sections in `section_ids`,
/// the fields first: `on_field(section_id, key_name)` and
/// `on_sample(section_id, time)` once per removed row, as
/// [`FieldRows::purge_sections`] and [`SampleRows::purge_sections`] call
/// them. On the compact form the field purge costs O(min(D × N, F)) for
/// `D` sections, `N` interned-name id slots (live and free) and `F` field
/// rows, and the sample purge O(T) for `T` sample rows, nothing when there
/// are none (`domain::types::PURGE_PROBE_RATIO`); on the full form,
/// O(F + T).
pub(crate) fn purge_value_rows<R: ValueRows, H: EngineHooks<R>>(
    state: &mut ObjectStateOf<R>,
    hooks: &mut H,
    section_ids: &HashSet<u64>,
    mut on_field: impl FnMut(u64, &str),
    mut on_sample: impl FnMut(u64, OrderedFloat<f64>),
) {
    // Each table's room and capacity as the purge begins: the purge only
    // removes — whether by `retain` or, on the compact field purge's probe
    // arm, by `remove` — and an erase never lowers a table's room nor raises
    // its capacity, so the room never falls below this while it runs, and
    // the capacity before any of its removals is this one.
    let field_room = state.fields.row_room();
    let name_room = state.fields.name_room();
    let sample_room = state.time_samples.row_room();
    let field_capacity = state.fields.row_capacity();
    let name_capacity = state.fields.name_capacity();
    let sample_capacity = state.time_samples.row_capacity();
    state
        .fields
        .purge_sections_with(section_ids, |section_id, name, hash, released| {
            if H::OBSERVES {
                let names = if released.is_some() {
                    NameChange::Released
                } else {
                    NameChange::Unchanged
                };
                hooks.field_written(section_id, name, Some(hash.copied()), None, names);
                hooks.removing(
                    RemovedKey::Field(section_id, name),
                    field_room,
                    field_capacity,
                );
                if let Some(id) = released {
                    hooks.removing(RemovedKey::FieldName(name, id), name_room, name_capacity);
                }
            }
            on_field(section_id, name);
        });
    state
        .time_samples
        .purge_sections_with(section_ids, |section_id, time, hash| {
            if H::OBSERVES {
                let time_bits = time.into_inner().to_bits();
                hooks.sample_written(section_id, time_bits, Some(hash.copied()), None);
                hooks.removing(
                    RemovedKey::Sample(section_id, time_bits),
                    sample_room,
                    sample_capacity,
                );
            }
            on_sample(section_id, time);
        });
}

// ── Header scalars ────────────────────────────────────────────────────────────

/// Set `state.sequence_num`.
pub(crate) fn set_sequence_num<R: ValueRows, H: EngineHooks<R>>(
    state: &mut ObjectStateOf<R>,
    hooks: &mut H,
    value: u64,
) {
    if H::OBSERVES {
        hooks.header_before(HeaderImage::of(state));
    }
    state.sequence_num = value;
}

/// Set `state.first_version`.
pub(crate) fn set_first_version<R: ValueRows, H: EngineHooks<R>>(
    state: &mut ObjectStateOf<R>,
    hooks: &mut H,
    value: u64,
) {
    if H::OBSERVES {
        hooks.header_before(HeaderImage::of(state));
    }
    state.first_version = value;
}

/// Set `state.next_set_order`.
pub(crate) fn set_next_set_order<R: ValueRows, H: EngineHooks<R>>(
    state: &mut ObjectStateOf<R>,
    hooks: &mut H,
    value: u64,
) {
    if H::OBSERVES {
        hooks.header_before(HeaderImage::of(state));
    }
    state.next_set_order = value;
}

/// Set `state.source_format` to `value`.
pub(crate) fn set_source_format<R: ValueRows, H: EngineHooks<R>>(
    state: &mut ObjectStateOf<R>,
    hooks: &mut H,
    value: &str,
) {
    if H::OBSERVES {
        hooks.header_before(HeaderImage::of(state));
    }
    state.source_format = value.to_string();
    if H::OBSERVES {
        hooks.source_format_after(&state.source_format);
    }
}

// ── Reset ─────────────────────────────────────────────────────────────────────

/// Move every section row, children list and value row out of `state`, for a
/// reset to emit its deletes from, between this and [`finish_reset`].
///
/// - **Without hooks** (`!H::OBSERVES`) this is the engine's in-place reset
///   split in two. Only the section and children maps move out; `state`
///   holds zero-capacity placeholders with their own hashers, which allocate
///   nothing. The reset *drains* the returned maps as it emits — moving each
///   section's name into its mutation — and the value rows in place
///   ([`reset_fields`], [`reset_samples`]), and [`finish_reset`] moves the
///   drained maps back. They come back empty with their buckets, exactly as
///   a drain or a clear in place leaves them, so nothing is allocated and
///   later inserts land where they always did.
/// - **With hooks** every table goes to [`EngineHooks::reset`] whole, so the
///   reset only *reads* them as it emits. `state` keeps empty tables of the
///   old capacity and hasher instead.
pub(crate) fn take_for_reset<R: ValueRows, H: EngineHooks<R>>(
    state: &mut ObjectStateOf<R>,
) -> ResetTables<R> {
    if H::OBSERVES {
        fn emptied<K, V>(map: &HashMap<K, V>) -> HashMap<K, V> {
            HashMap::with_capacity_and_hasher(map.capacity(), map.hasher().clone())
        }
        let sections = emptied(&state.sections);
        let children = emptied(&state.children);
        return ResetTables {
            sections: std::mem::replace(&mut state.sections, sections),
            children: std::mem::replace(&mut state.children, children),
            fields: Some(state.fields.take_all()),
            time_samples: Some(state.time_samples.take_all()),
        };
    }
    // Placeholders until `finish_reset` moves the drained maps back; nothing
    // is inserted into them in between.
    let sections = HashMap::with_hasher(state.sections.hasher().clone());
    let children = HashMap::with_hasher(state.children.hasher().clone());
    ResetTables {
        sections: std::mem::replace(&mut state.sections, sections),
        children: std::mem::replace(&mut state.children, children),
        fields: None,
        time_samples: None,
    }
}

/// Close a reset begun by [`take_for_reset`]. With hooks, hand them the
/// tables ([`EngineHooks::reset`]). Without, move the section and children
/// maps — which the reset has drained — back into `state` in place of the
/// placeholders.
pub(crate) fn finish_reset<R: ValueRows, H: EngineHooks<R>>(
    state: &mut ObjectStateOf<R>,
    hooks: &mut H,
    taken: ResetTables<R>,
) {
    if H::OBSERVES {
        hooks.reset(taken);
        return;
    }
    debug_assert!(
        taken.sections.is_empty() && taken.children.is_empty(),
        "an un-hooked reset must drain its maps before moving them back"
    );
    state.sections = taken.sections;
    state.children = taken.children;
}
