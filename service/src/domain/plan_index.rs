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

//! The structural write cache's persistent plan indexes ([`PlanIndexes`]):
//! what a structural plan reads beside the state, kept across plans instead
//! of rebuilt from the whole layer on every delta. They live with the cache
//! entry (`storage::structural_cache::CacheEntry`), never in the state, so no
//! reader type changes and the state's own weight estimate is unchanged.
//!
//! - **The name index**: for each parent id, the `(name hash, section id)` of
//!   every row naming that parent, listed or not, sorted (one parent-keyed
//!   map of sorted vectors). It answers the engine's `(parent_id, name)`
//!   conflict lookup by the same rule as its per-delta index (the lowest
//!   surviving id, re-verified against the row) and a birth's adoption (every
//!   row naming the newborn). The engine's per-delta `NameIndex` and
//!   `AdoptionIndex` builds, and their O(sections) passes, are skipped.
//! - **External-hash counts** over the field and time-sample rows, and the
//!   number of external rows of each kind, so the count of untouched rows
//!   naming a hash is read in O(touched rows).
//! - **Running allocation sums**: section names, children lists and interned
//!   field names — the parts of `storage::structural_cache::estimate_state_bytes`
//!   that walk the whole state — so the entry's weight can be read in O(1),
//!   from these and the tables' held capacities.
//! - **A capacity high-water per hash table** of the table registry
//!   ([`RebuildTable`]: the state's tables and these indexes' own maps): the
//!   largest `capacity()` each reported just before a removal since the
//!   entry was last compacted. A removal can leave a tombstone, which lowers
//!   `capacity()` without freeing a bucket, so a table thinned that way reads
//!   one power of two low once half its usable slots are tombstones. Every
//!   weigher reads a table at its held capacity, `max(high-water,
//!   capacity())` ([`PlanIndexes::held_capacity`]), never below the buckets
//!   it holds. A table's buckets change only when an insert grows it — after
//!   which, as after an in-place rehash, `capacity()` reads them exactly
//!   until the next removal, which samples it first — at compaction, which
//!   resets the high-water, at a reset, which swaps in fresh tables and
//!   [`PlanIndexes::emptied`]'s zeroed high-water together, and at a
//!   rollback's rebuild, whose smaller table the high-water over-reads until
//!   the compaction that follows. So the O(1) weight equals the estimate at
//!   every compaction and stays at or above it while a table thinned since
//!   then hides buckets from the estimate's `capacity()` reading. Sampled
//!   through the writer's removal hook
//!   (`domain::delta_write::EngineHooks::removing`), the indexes' own map
//!   removals, [`normalize_touched`] and the start of a rollback
//!   (`domain::plan_journal`).
//! - **`depth_ok`**: the state passed the full section-tree depth walk and
//!   the shape rules the incremental check relies on ([`PlanIndexes::build`]).
//!   While it is set, a plan checks only what its own listings can have made
//!   deeper ([`IndexedHooks`]).
//!
//! [`PlanIndexes::build`] makes them in one pass over the state, at a cold
//! start or reload, after the fenced read has released its fence. Every
//! write a hooked plan or replay makes reaches [`IndexedHooks`] through the
//! engine's writer (`domain::delta_write`), which keeps them in step.
//!
//! Pure logic: zero I/O, zero async (SDD §9).

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::mem::size_of;

use crate::domain::alloc_model::{
    allocation_bytes, table_bytes, table_growth_bytes, table_growth_bytes_held,
};
use crate::domain::delta::{check_section_tree_depth_capped, DeltaError, RebuildProjector};
use crate::domain::delta_write::{
    EngineHooks, HeaderImage, NameChange, RebuildTable, RemovedKey, ResetTables,
};
use crate::domain::limits::MAX_SECTION_TREE_DEPTH;
use crate::domain::plan_journal::{PlanJournal, RowImage};
use crate::domain::types::{
    rebuild_map_at, CompactFields, CompactRows, CompactSamples, CompactState, FieldRows,
    NameInterner, ObjectStateOf, SampleRows, Section,
};

/// One name-index entry: a row's name hash and its section id.
type NameEntry = (u32, u64);

/// The name hash the index keys rows by: std's `DefaultHasher` (SipHash with
/// fixed keys, so the same in every process) truncated to 32 bits. Entries
/// are candidates, re-verified against the row, so a collision costs one
/// lookup, never a wrong answer.
fn name_hash(name: &str) -> u32 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    name.hash(&mut hasher);
    hasher.finish() as u32
}

/// The heap bytes one name-index vector of `capacity` entries holds.
fn entries_bytes(capacity: usize) -> u64 {
    allocation_bytes((capacity as u64).saturating_mul(size_of::<NameEntry>() as u64))
}

/// The heap bytes one children list of `capacity` ids holds, as the state's
/// weight estimate charges it.
pub fn list_bytes(capacity: usize) -> u64 {
    allocation_bytes((capacity as u64).saturating_mul(size_of::<u64>() as u64))
}

/// The heap bytes of one interned field name of `len` bytes, as the state's
/// weight estimate charges it.
pub fn interned_name_bytes(len: usize) -> u64 {
    allocation_bytes(NameInterner::name_allocation_bytes(len) as u64)
}

/// The structural cache entry's persistent plan indexes (module docs).
#[derive(Debug, Default)]
pub struct PlanIndexes {
    /// Parent id → `(name hash, section id)` of every row naming it, sorted
    /// and non-empty (a parent with no row naming it has no entry).
    names: HashMap<u64, Vec<NameEntry>>,
    /// The heap bytes of `names`' vectors, at their capacities.
    names_vec_bytes: u64,
    /// External hash → the field and time-sample rows holding it.
    ext_counts: HashMap<[u8; 32], u32>,
    /// Field rows holding an external hash.
    ext_field_rows: u64,
    /// Time-sample rows holding an external hash.
    ext_sample_rows: u64,
    /// Σ [`interned_name_bytes`] over the interned field names.
    interned_names_bytes: u64,
    /// Σ `allocation_bytes(name.capacity())` over the section rows.
    section_names_bytes: u64,
    /// Σ [`list_bytes`] over the children lists, at their capacities.
    lists_bytes: u64,
    /// The state passed the full depth walk and the shape rules
    /// ([`Self::build`]); cleared by a plan whose listings broke them.
    depth_ok: bool,
    /// Per [`RebuildTable`], in [`RebuildTable::index`] order — the state's
    /// tables and these indexes' own maps — the largest `capacity()` the
    /// table reported just before a removal since the entry was last
    /// compacted; 0 at [`Self::build`], [`Self::emptied`] and right after the
    /// cache's compaction ([`Self::reset_capacity_high_water`]). A removal can
    /// leave a tombstone, which lowers `capacity()` without freeing a bucket,
    /// so every weigher reads a table at its held capacity,
    /// `max(high-water, capacity())` ([`Self::held_capacity`]): never fewer
    /// buckets than the table holds (module docs). Sampled only before
    /// removals ([`Self::note_capacity`]), never after an insert: a table
    /// grown since its last removal reads its new buckets from `capacity()`
    /// itself, and an insert-only plan or replay leaves it at 0.
    capacity_high_water: [usize; RebuildTable::COUNT],
}

impl Clone for PlanIndexes {
    /// A field-wise clone, except that the name vectors' running sum is
    /// recomputed: a cloned `Vec` is allocated at its length, not at the
    /// original's capacity, so the original's sum would overstate the clone.
    /// The capacity high-water is copied: a cloned map keeps its buckets and
    /// its tombstones.
    fn clone(&self) -> Self {
        let names = self.names.clone();
        let names_vec_bytes = names.values().fold(0u64, |sum, entries| {
            sum.saturating_add(entries_bytes(entries.capacity()))
        });
        Self {
            names,
            names_vec_bytes,
            ext_counts: self.ext_counts.clone(),
            ext_field_rows: self.ext_field_rows,
            ext_sample_rows: self.ext_sample_rows,
            interned_names_bytes: self.interned_names_bytes,
            section_names_bytes: self.section_names_bytes,
            lists_bytes: self.lists_bytes,
            depth_ok: self.depth_ok,
            capacity_high_water: self.capacity_high_water,
        }
    }
}

// ── The hash-table registry ─────────────────────────────────────────────────

/// The bytes one slot of `table` takes, as the entry's weight estimate
/// counts it (`storage::structural_cache::estimate_state_bytes`,
/// [`PlanIndexes::weight_bytes`]).
pub fn table_slot_bytes(table: RebuildTable) -> u64 {
    let bytes = match table {
        RebuildTable::Sections => size_of::<(u64, Section)>(),
        RebuildTable::Children => size_of::<(u64, Vec<u64>)>(),
        RebuildTable::FieldRows => CompactFields::ROW_SLOT_BYTES,
        RebuildTable::SampleRows => CompactSamples::ROW_SLOT_BYTES,
        RebuildTable::NameIds => NameInterner::ID_ENTRY_BYTES,
        RebuildTable::NameIndex => PlanIndexes::NAME_SLOT_BYTES,
        RebuildTable::ExtCounts => PlanIndexes::EXT_SLOT_BYTES,
    };
    bytes as u64
}

/// `(len, capacity())` of the state's table `table`; `None` for one of the
/// plan indexes' own maps ([`RebuildTable::in_state`]).
pub fn state_table_shape(table: RebuildTable, state: &CompactState) -> Option<(usize, usize)> {
    match table {
        RebuildTable::Sections => Some((state.sections.len(), state.sections.capacity())),
        RebuildTable::Children => Some((state.children.len(), state.children.capacity())),
        RebuildTable::FieldRows => Some((FieldRows::len(&state.fields), state.fields.capacity())),
        RebuildTable::SampleRows => Some((
            SampleRows::len(&state.time_samples),
            state.time_samples.capacity(),
        )),
        RebuildTable::NameIds => {
            let names = state.fields.names();
            Some((names.len(), names.ids_capacity()))
        }
        RebuildTable::NameIndex | RebuildTable::ExtCounts => None,
    }
}

/// `(len, capacity())` of `table`, the state's or `indexes`' own.
pub fn table_shape(
    table: RebuildTable,
    state: &CompactState,
    indexes: &PlanIndexes,
) -> (usize, usize) {
    state_table_shape(table, state)
        .or_else(|| indexes.own_table_shape(table))
        .unwrap_or((0, 0))
}

impl PlanIndexes {
    /// The indexes of `state`, in one pass over its rows and lists, with
    /// `depth_ok` from the full depth walk and the shape rules
    /// ([`shape_ok`]).
    pub fn build(state: &CompactState) -> Self {
        let mut indexes = PlanIndexes::default();
        for section in state.sections.values() {
            indexes.section_names_bytes = indexes
                .section_names_bytes
                .saturating_add(allocation_bytes(section.name.capacity() as u64));
            // Pushed unsorted, sorted per parent below: each vector grows by
            // doubling from one entry, as an insert-maintained one does.
            let entries = indexes
                .names
                .entry(section.parent_id)
                .or_insert_with(|| Vec::with_capacity(1));
            entries.push((name_hash(&section.name), section.section_id));
        }
        for entries in indexes.names.values_mut() {
            entries.sort_unstable();
            indexes.names_vec_bytes = indexes
                .names_vec_bytes
                .saturating_add(entries_bytes(entries.capacity()));
        }
        for list in state.children.values() {
            indexes.lists_bytes = indexes
                .lists_bytes
                .saturating_add(list_bytes(list.capacity()));
        }
        for name in state.fields.names().names() {
            indexes.interned_names_bytes = indexes
                .interned_names_bytes
                .saturating_add(interned_name_bytes(name.len()));
        }
        state.fields.for_each(|_, _, hash| {
            if let Some(hash) = hash {
                indexes.ext_field_rows += 1;
                indexes.count_hash(*hash, true);
            }
        });
        state.time_samples.for_each(|_, _, hash| {
            if let Some(hash) = hash {
                indexes.ext_sample_rows += 1;
                indexes.count_hash(*hash, true);
            }
        });
        indexes.depth_ok = shape_ok(state)
            && check_section_tree_depth_capped(state, MAX_SECTION_TREE_DEPTH).is_ok();
        indexes
    }

    /// The most bytes [`Self::build`] holds at once beyond the state, for a
    /// state of `sections` section rows, `list_entries` children-list entries
    /// in all and `value_rows` field and time-sample rows in all (the upper
    /// bound on its distinct external hashes). An upper bound from the counts
    /// alone, so a cold start can charge the build before it runs. It sums:
    ///
    /// - **the name index's map**, filled one parent key at a time (at most
    ///   one per row), at its last growth: the table it grows to and the one
    ///   before it, alive at once while the items move;
    /// - **the name index's vectors**, each filled by pushes from a capacity
    ///   of one: every vector at a capacity of at most twice its length (so
    ///   at most two entries' room per row), plus the buffer the vector
    ///   growing last is moving out of (at most one entry's room per row);
    /// - **the external-hash counts**, filled one hash at a time (at most one
    ///   per value row), at their last growth, as the name map;
    /// - **the shape check's** member set for one list (the largest is at
    ///   most every list entry), and **the depth walk's** visited set
    ///   (pre-sized to the sections) and stack — the walk runs only on a
    ///   state whose shape passed, where each section is listed at most once,
    ///   so the stack grows by pushes to at most the sections, its last
    ///   growth holding the buffer it grows to and the one before it. Both
    ///   are dropped before the build returns.
    pub fn build_peak_bytes(sections: u64, list_entries: u64, value_rows: u64) -> u64 {
        let usize_of = |n: u64| usize::try_from(n).unwrap_or(usize::MAX);
        let names_map =
            table_growth_bytes(0, 0, sections, size_of::<(u64, Vec<NameEntry>)>() as u64);
        let names_vectors = sections.saturating_mul(entries_bytes(1)).saturating_mul(3);
        let ext_counts = table_growth_bytes(0, 0, value_rows, size_of::<([u8; 32], u32)>() as u64);
        let shape = table_bytes(usize_of(list_entries), size_of::<u64>() as u64);
        let stack_entry = size_of::<(u64, usize)>() as u64;
        let walk = table_bytes(usize_of(sections), size_of::<u64>() as u64)
            .saturating_add(allocation_bytes(
                sections
                    .saturating_mul(2)
                    .max(4)
                    .saturating_mul(stack_entry),
            ))
            .saturating_add(allocation_bytes(sections.saturating_mul(stack_entry)));
        (size_of::<PlanIndexes>() as u64)
            .saturating_add(names_map)
            .saturating_add(names_vectors)
            .saturating_add(ext_counts)
            .saturating_add(shape)
            .saturating_add(walk)
    }

    /// [`Self::build_peak_bytes`] at `state`'s counts: its section rows, its
    /// children lists' entries (summed, O(lists)) and its field and
    /// time-sample rows.
    pub fn build_peak_bytes_for(state: &CompactState) -> u64 {
        let count = |n: usize| u64::try_from(n).unwrap_or(u64::MAX);
        let list_entries = state
            .children
            .values()
            .fold(0u64, |sum, list| sum.saturating_add(count(list.len())));
        Self::build_peak_bytes(
            count(state.sections.len()),
            list_entries,
            count(state.fields.len()).saturating_add(count(state.time_samples.len())),
        )
    }

    /// Whether the state these indexes describe passed the full depth walk
    /// and the shape rules, so a plan may check its depth incrementally.
    pub fn depth_ok(&self) -> bool {
        self.depth_ok
    }

    /// The bytes the indexes hold: the struct itself (its capacity
    /// high-water included), their maps at their held capacities
    /// ([`Self::held_capacity`]) and the name vectors at their capacities.
    /// O(1).
    pub fn weight_bytes(&self) -> u64 {
        self.weight_with_vectors(self.names_vec_bytes)
    }

    /// [`Self::weight_bytes`] with the name vectors at `vectors` bytes: the
    /// struct, and each of the indexes' own maps in the table registry
    /// ([`RebuildTable`]) at its held capacity.
    fn weight_with_vectors(&self, vectors: u64) -> u64 {
        RebuildTable::ALL
            .iter()
            .filter_map(|&table| {
                let (_, capacity) = self.own_table_shape(table)?;
                Some(table_bytes(
                    self.held_capacity(table, capacity),
                    table_slot_bytes(table),
                ))
            })
            .fold(
                (size_of::<PlanIndexes>() as u64).saturating_add(vectors),
                u64::saturating_add,
            )
    }

    /// The bytes the indexes' own maps can allocate while a plan makes up to
    /// `calls(map)` insert calls on each — on the name index's parent-keyed
    /// map, the plan's name-key inserts; on the external-hash counts, at most
    /// one per value row the plan writes holding an external hash, so the
    /// cached route passes the delta's external set fields and time samples
    /// (`storage::structural_cache::PlannedInserts::calls`): for each map in
    /// the table registry ([`RebuildTable`]), what
    /// [`table_growth_bytes_held`] says its growth through those calls can
    /// hold at once — nothing while its length plus its calls stays within
    /// its capacity, else at least the doubled table hashbrown grows to,
    /// whatever tombstones its removals (`name_remove`, `count_hash`) left,
    /// doubled from its held capacity ([`Self::held_capacity`]), which those
    /// tombstones cannot hide. After a reset (`emptied`) the maps start
    /// empty, fresh, so they are priced from zero. The name vectors are not
    /// here: each grows with its parent's listing, which the plan projects at
    /// the parent's whole length first.
    ///
    /// A key a write removes and adds back counts as an insert call: a row
    /// write takes the row's entry out before it puts it back, and the last
    /// entry out takes its parent's key (or its hash's) with it. A rollback's
    /// re-creations are not here: it rebuilds a map without the room at its
    /// pre-plan length instead of growing it ([`Self::restore_name_index`],
    /// [`Self::restore_ext_counts`]), charged on its own.
    pub fn growth_bytes(&self, calls: impl Fn(RebuildTable) -> u64, emptied: bool) -> u64 {
        RebuildTable::ALL
            .iter()
            .filter_map(|&table| {
                let (len, capacity) = self.own_table_shape(table)?;
                let slot = table_slot_bytes(table);
                Some(if emptied {
                    table_growth_bytes(0, 0, calls(table), slot)
                } else {
                    table_growth_bytes_held(
                        len,
                        capacity,
                        self.held_capacity(table, capacity),
                        calls(table),
                        slot,
                    )
                })
            })
            .fold(0, u64::saturating_add)
    }

    /// [`Self::weight_bytes`] recomputed by walking the name vectors, for a
    /// check that the running sum has not drifted.
    pub fn weight_bytes_recomputed(&self) -> u64 {
        let vectors = self.names.values().fold(0u64, |sum, entries| {
            sum.saturating_add(entries_bytes(entries.capacity()))
        });
        self.weight_with_vectors(vectors)
    }

    /// `(len, capacity())` of the indexes' own map `table`; `None` for one of
    /// the state's tables ([`RebuildTable::in_state`]).
    pub fn own_table_shape(&self, table: RebuildTable) -> Option<(usize, usize)> {
        match table {
            RebuildTable::NameIndex => Some((self.names.len(), self.names.capacity())),
            RebuildTable::ExtCounts => Some((self.ext_counts.len(), self.ext_counts.capacity())),
            RebuildTable::Sections
            | RebuildTable::Children
            | RebuildTable::FieldRows
            | RebuildTable::SampleRows
            | RebuildTable::NameIds => None,
        }
    }

    /// `table`, which reports `capacity` usable slots, is about to lose a
    /// key: raise its capacity high-water to `capacity` (field docs). Called
    /// immediately before every removal from a table of a hooked plan or
    /// replay — through the writer's [`EngineHooks::removing`], by the
    /// indexes' own removals, by `normalize_touched` and at a rollback's
    /// start — so it reads the table before the removal's tombstone can hide
    /// any of its buckets. O(1).
    pub fn note_capacity(&mut self, table: RebuildTable, capacity: usize) {
        let high_water = &mut self.capacity_high_water[table.index()];
        *high_water = (*high_water).max(capacity);
    }

    /// The capacity a weigher reads `table` at, which reports `capacity`
    /// usable slots now: `max(high-water, capacity)`. Never fewer buckets
    /// than the table holds, and never more than the most it held since the
    /// entry was compacted (more only after a rollback rebuilt the table
    /// smaller, which the compaction after every rollback undoes). The room
    /// a table takes before an insert call grows it is still `capacity −
    /// len`. O(1).
    pub fn held_capacity(&self, table: RebuildTable, capacity: usize) -> usize {
        self.capacity_high_water[table.index()].max(capacity)
    }

    /// Zero every capacity high-water: the cache's compaction just left each
    /// table minimal for its length, whose `capacity()` reads its exact
    /// bucket count (`domain::alloc_model::table_bytes`). Called after the
    /// compaction and before the weigh
    /// (`storage::structural_cache::CacheEntry::compact_and_weigh`): weighed
    /// before it, an entry would be stored at the uncompacted tables' bucket
    /// counts.
    pub fn reset_capacity_high_water(&mut self) {
        self.capacity_high_water = [0; RebuildTable::COUNT];
    }

    /// `table`'s capacity high-water (field docs): 0 when nothing was
    /// removed from it since the entry was built, emptied or compacted.
    pub fn capacity_high_water(&self, table: RebuildTable) -> usize {
        self.capacity_high_water[table.index()]
    }

    /// Whether these indexes describe the same rows, hashes and allocation
    /// sums as `other` — a fresh [`Self::build`] of the same state, say —
    /// whatever the capacities of their own maps. `depth_ok` is not
    /// compared: a plan never sets it again once it is clear.
    pub fn same_content_as(&self, other: &PlanIndexes) -> bool {
        self.names.len() == other.names.len()
            && self.names.iter().all(|(parent, entries)| {
                other
                    .names
                    .get(parent)
                    .is_some_and(|theirs| theirs.as_slice() == entries.as_slice())
            })
            && self.ext_counts == other.ext_counts
            && self.ext_field_rows == other.ext_field_rows
            && self.ext_sample_rows == other.ext_sample_rows
            && self.interned_names_bytes == other.interned_names_bytes
            && self.section_names_bytes == other.section_names_bytes
            && self.lists_bytes == other.lists_bytes
    }

    /// One slot of the name index's parent-keyed map.
    pub const NAME_SLOT_BYTES: usize = size_of::<(u64, Vec<NameEntry>)>();

    /// One slot of the external-hash counts.
    pub const EXT_SLOT_BYTES: usize = size_of::<([u8; 32], u32)>();

    /// The parent keys of the name index.
    pub fn parent_keys(&self) -> usize {
        self.names.len()
    }

    /// The distinct external hashes counted.
    pub fn hash_keys(&self) -> usize {
        self.ext_counts.len()
    }

    /// The parent keys the name index's map takes before an insert call
    /// would grow it.
    pub fn parent_key_room(&self) -> usize {
        self.names.capacity().saturating_sub(self.names.len())
    }

    /// `(len, capacity)` of `parent_id`'s name-index vector — the
    /// `(name hash, section id)` entries of every row naming it — or `(0, 0)`
    /// when no row names it (no key, no vector). Unlike the children lists,
    /// these vectors are never trimmed to their length — not by a rollback,
    /// a removal or [`Self::compact`] — so while its parent keeps a key a
    /// vector keeps the largest capacity it ever had, and can have room: a
    /// caller pricing a write that adds entries under `parent_id`
    /// (`storage::structural_cache::estimate_replay_record_growth`) reads
    /// whether it reallocates from this. The length is also the count of rows
    /// naming `parent_id`, live or not: the orphans a birth of that id
    /// adopts. O(1).
    pub fn name_vector_room(&self, parent_id: u64) -> (usize, usize) {
        self.names
            .get(&parent_id)
            .map_or((0, 0), |entries| (entries.len(), entries.capacity()))
    }

    /// One name-index vector entry's bytes, `(name hash, section id)`.
    pub const NAME_ENTRY_BYTES: usize = size_of::<NameEntry>();

    /// The hashes the external-hash counts take before an insert call would
    /// grow them.
    pub fn hash_key_room(&self) -> usize {
        self.ext_counts
            .capacity()
            .saturating_sub(self.ext_counts.len())
    }

    /// Whether taking `row`'s entry out of the name index removes its
    /// parent's key (the entry is its vector's last).
    fn entry_out_removes_key(&self, row: &Section) -> bool {
        self.names
            .get(&row.parent_id)
            .is_some_and(|entries| entries.len() == 1)
    }

    /// Whether putting `row`'s entry in makes an insert call on the name
    /// index's map (its parent has no key yet).
    fn entry_in_adds_key(&self, row: &Section) -> bool {
        !self.names.contains_key(&row.parent_id)
    }

    /// The count of `hash`.
    fn hash_count(&self, hash: &[u8; 32]) -> u32 {
        self.ext_counts.get(hash).copied().unwrap_or(0)
    }

    /// A rollback's restore of the name index and the section-name sum
    /// (`domain::plan_journal`), from each touched row's `(current, pre-plan)`
    /// value, called while the state still holds the current ones. A row
    /// whose `(parent, name hash, id)` entry the plan did not change is
    /// skipped (its name allocation still re-counted). Otherwise, in order:
    ///
    /// 1. every changed current entry is taken out of its parent's vector,
    ///    the key kept even when its vector empties;
    /// 2. every changed pre-plan entry whose parent has a key goes back in.
    ///    The vector then holds exactly its parent's pre-plan entries. One
    ///    whose key survived the whole plan never needs more than the
    ///    capacity it had before the plan (a `Vec` never shrinks here), so
    ///    it never reallocates. One whose key the plan dropped and
    ///    re-created can be shorter than that; it is rebuilt once at its
    ///    restored length instead of growing a step at a time;
    /// 3. the keys whose vectors are still empty — keys the plan created —
    ///    are dropped;
    /// 4. pre-plan entries whose parent has no key — a key the plan dropped —
    ///    are re-created, each parent's vector at its restored length, into
    ///    a map first rebuilt at `pre_parent_keys` ([`rebuild_map_at`]) when
    ///    it lacks the room.
    ///
    /// The vectors of steps 2 and 4 allocate at most the pre-plan vector of
    /// their parent. The plan projected that allocation, an upper bound, when
    /// it dropped the parent's key: the dropped vector's capacity
    /// (`PlanJournal::note_vector_drop`). Returns whether the map was rebuilt.
    ///
    /// So the restore gives back the index's contents, not its layout: a
    /// vector the plan grew keeps its grown capacity and one rebuilt or
    /// re-created sits at its restored length, so [`Self::weight_bytes`] can
    /// end above or below its pre-plan value after the rollback (and
    /// [`Self::compact`] shrinks only the two maps). A call that suspends on
    /// a refused plan growth prices its resumed attempt's CR1 on the entry
    /// as put back for that reason (`storage::layer`, the `Overflow` arm). A
    /// resumed plan that drops such a vector projects it at the capacity it
    /// then has, which is not predicted, so it can ask once more. A grown
    /// capacity is kept for as long as the entry is cached, so rolled-back
    /// plans that grow different parents' vectors add up; the cache bounds
    /// the result by evicting, when it is stored, an entry heavier than it
    /// can hold (`storage::structural_cache`, `Control::store`).
    pub(crate) fn restore_name_index<'s>(
        &mut self,
        rows: impl Iterator<Item = (Option<&'s Section>, Option<&'s Section>)> + Clone,
        pre_parent_keys: usize,
    ) -> bool {
        let entry = |row: &Section| (row.parent_id, name_hash(&row.name), row.section_id);
        // 1. Take the changed current entries out, keeping every key.
        let mut taken_from: Vec<u64> = Vec::new();
        for (current, pre) in rows.clone() {
            if let Some(current) = current {
                self.section_names_bytes = self
                    .section_names_bytes
                    .saturating_sub(allocation_bytes(current.name.capacity() as u64));
            }
            if let Some(pre) = pre {
                self.section_names_bytes = self
                    .section_names_bytes
                    .saturating_add(allocation_bytes(pre.name.capacity() as u64));
            }
            let Some(current) = current else {
                continue;
            };
            if pre.map(entry) == Some(entry(current)) {
                continue;
            }
            let (parent, hash, id) = entry(current);
            if let Some(entries) = self.names.get_mut(&parent) {
                if let Ok(position) = entries.binary_search(&(hash, id)) {
                    entries.remove(position);
                }
                taken_from.push(parent);
            }
        }
        // 2. Put the changed pre-plan entries back, per parent.
        let mut returning: Vec<(u64, u32, u64)> = rows
            .filter_map(|(current, pre)| {
                let pre = pre?;
                (current.map(entry) != Some(entry(pre))).then(|| entry(pre))
            })
            .collect();
        returning.sort_unstable();
        returning.dedup();
        let mut deferred: Vec<(u64, u32, u64)> = Vec::new();
        for group in returning.chunk_by(|a, b| a.0 == b.0) {
            let parent = group[0].0;
            let Some(entries) = self.names.get_mut(&parent) else {
                deferred.extend_from_slice(group);
                continue;
            };
            let restored = entries.len().saturating_add(group.len());
            if entries.capacity() >= restored {
                for &(_, hash, id) in group {
                    if let Err(position) = entries.binary_search(&(hash, id)) {
                        entries.insert(position, (hash, id));
                    }
                }
            } else {
                // A key the plan dropped and re-created: its vector is
                // rebuilt once at the restored length.
                let mut rebuilt: Vec<NameEntry> = Vec::with_capacity(restored);
                rebuilt.extend_from_slice(entries);
                rebuilt.extend(group.iter().map(|&(_, hash, id)| (hash, id)));
                rebuilt.sort_unstable();
                rebuilt.dedup();
                let before = entries.capacity();
                *entries = rebuilt;
                self.names_vec_bytes = self
                    .names_vec_bytes
                    .saturating_sub(entries_bytes(before))
                    .saturating_add(entries_bytes(entries.capacity()));
            }
        }
        // 3. Drop the keys left empty: keys the plan created.
        taken_from.sort_unstable();
        taken_from.dedup();
        for parent in taken_from {
            if self.names.get(&parent).is_some_and(Vec::is_empty) {
                self.note_capacity(RebuildTable::NameIndex, self.names.capacity());
                if let Some(entries) = self.names.remove(&parent) {
                    self.names_vec_bytes = self
                        .names_vec_bytes
                        .saturating_sub(entries_bytes(entries.capacity()));
                }
            }
        }
        // 4. Re-create the keys the plan dropped.
        let keys = deferred.chunk_by(|a, b| a.0 == b.0).count();
        let rebuilt = self.parent_key_room() < keys;
        if rebuilt {
            rebuild_map_at(&mut self.names, pre_parent_keys);
        }
        for group in deferred.chunk_by(|a, b| a.0 == b.0) {
            let mut entries: Vec<NameEntry> = Vec::with_capacity(group.len());
            entries.extend(group.iter().map(|&(_, hash, id)| (hash, id)));
            self.names_vec_bytes = self
                .names_vec_bytes
                .saturating_add(entries_bytes(entries.capacity()));
            self.names.insert(group[0].0, entries);
        }
        rebuilt
    }

    /// The heap bytes of `parent`'s name-index vector, at its capacity; 0
    /// with no key.
    fn parent_vector_bytes(&self, parent: u64) -> u64 {
        self.names
            .get(&parent)
            .map_or(0, |entries| entries_bytes(entries.capacity()))
    }

    /// A rollback's restore of the external-hash counts and the external-row
    /// counters, from each touched field and time-sample row's
    /// `(current, pre-plan)` image, read while the state still holds the
    /// current ones. A pre-plan hash is counted back before a current one is
    /// counted out, so a hash's key is removed only when the plan created
    /// it; hashes whose key the plan removed are re-created last, into a map
    /// first rebuilt at `pre_hash_keys` when it lacks the room. Returns
    /// whether it was rebuilt.
    pub(crate) fn restore_ext_counts(
        &mut self,
        fields: &[(RowImage, RowImage)],
        samples: &[(RowImage, RowImage)],
        pre_hash_keys: usize,
    ) -> bool {
        let hash_of = |image: RowImage| image.flatten();
        let changed = fields
            .iter()
            .chain(samples)
            .filter(|(current, pre)| hash_of(*current) != hash_of(*pre));
        let mut deferred: Vec<[u8; 32]> = Vec::new();
        for &(_, pre) in changed.clone() {
            if let Some(hash) = hash_of(pre) {
                match self.ext_counts.get_mut(&hash) {
                    Some(count) => *count += 1,
                    None => deferred.push(hash),
                }
            }
        }
        for &(current, _) in changed {
            if let Some(hash) = hash_of(current) {
                self.count_hash(hash, false);
            }
        }
        for (rows, images) in [
            (&mut self.ext_field_rows, fields),
            (&mut self.ext_sample_rows, samples),
        ] {
            for &(current, pre) in images {
                if hash_of(current).is_some() {
                    *rows = rows.saturating_sub(1);
                }
                if hash_of(pre).is_some() {
                    *rows = rows.saturating_add(1);
                }
            }
        }
        deferred.sort_unstable();
        let mut keys = deferred.clone();
        keys.dedup();
        let rebuilt = self.hash_key_room() < keys.len();
        if rebuilt {
            rebuild_map_at(&mut self.ext_counts, pre_hash_keys);
        }
        for group in deferred.chunk_by(|a, b| a == b) {
            let count = u32::try_from(group.len()).unwrap_or(u32::MAX);
            self.ext_counts.insert(group[0], count);
        }
        rebuilt
    }

    /// An interned field name was interned or released outside a hooked
    /// write (a rollback's): keep the interned-name sum.
    pub(crate) fn interned_name_changed(&mut self, name: &str, change: NameChange) {
        match change {
            NameChange::Unchanged => {}
            NameChange::Interned => {
                self.interned_names_bytes = self
                    .interned_names_bytes
                    .saturating_add(interned_name_bytes(name.len()));
            }
            NameChange::Released => {
                self.interned_names_bytes = self
                    .interned_names_bytes
                    .saturating_sub(interned_name_bytes(name.len()));
            }
        }
    }

    /// The field and time-sample rows holding `hash`.
    pub fn ext_count(&self, hash: &[u8; 32]) -> u64 {
        self.ext_counts.get(hash).copied().map_or(0, u64::from)
    }

    /// Field rows holding an external hash.
    pub fn external_field_rows(&self) -> u64 {
        self.ext_field_rows
    }

    /// Time-sample rows holding an external hash.
    pub fn external_sample_rows(&self) -> u64 {
        self.ext_sample_rows
    }

    /// Σ of the interned field names' allocations.
    pub fn interned_names_bytes(&self) -> u64 {
        self.interned_names_bytes
    }

    /// Σ of the section names' allocations.
    pub fn section_names_bytes(&self) -> u64 {
        self.section_names_bytes
    }

    /// Σ of the children lists' allocations.
    pub fn lists_bytes(&self) -> u64 {
        self.lists_bytes
    }

    /// The lowest id among the rows of `sections` naming `(parent_id, name)`,
    /// other than `except_id`: the engine's `find_child_by_name` rule, each
    /// candidate re-verified against its row.
    pub(crate) fn child_by_name(
        &self,
        sections: &HashMap<u64, Section>,
        parent_id: u64,
        name: &str,
        except_id: Option<u64>,
    ) -> Option<u64> {
        let entries = self.names.get(&parent_id)?;
        let hash = name_hash(name);
        let start = entries.partition_point(|&(h, _)| h < hash);
        entries[start..]
            .iter()
            .take_while(|&&(h, _)| h == hash)
            .map(|&(_, id)| id)
            .find(|&id| {
                except_id != Some(id)
                    && sections
                        .get(&id)
                        .is_some_and(|row| row.parent_id == parent_id && row.name == name)
            })
    }

    /// The ids of every row of `sections` at `(parent_id, name)`, each
    /// candidate re-verified against its row as [`Self::child_by_name`]'s
    /// are, in ascending order. O(log L + matches) for `parent_id`'s `L`
    /// rows; allocates nothing. The rows a command introducing
    /// `(parent_id, name)` can cascade-delete as a same-name sibling
    /// (`storage::structural_cache::estimate_replay_record_growth`).
    pub(crate) fn rows_named<'a>(
        &'a self,
        sections: &'a HashMap<u64, Section>,
        parent_id: u64,
        name: &'a str,
    ) -> impl Iterator<Item = u64> + 'a {
        let entries = self.names.get(&parent_id).map_or(&[][..], Vec::as_slice);
        let hash = name_hash(name);
        let start = entries.partition_point(|&(h, _)| h < hash);
        entries[start..]
            .iter()
            .take_while(move |&&(h, _)| h == hash)
            .map(|&(_, id)| id)
            .filter(move |id| {
                sections
                    .get(id)
                    .is_some_and(|row| row.parent_id == parent_id && row.name == name)
            })
    }

    /// The ids of every row naming `parent_id`, listed or not, in no
    /// particular order.
    pub(crate) fn rows_naming(&self, parent_id: u64) -> impl Iterator<Item = u64> + '_ {
        self.names
            .get(&parent_id)
            .into_iter()
            .flat_map(|entries| entries.iter().map(|&(_, id)| id))
    }

    fn count_hash(&mut self, hash: [u8; 32], add: bool) {
        if add {
            *self.ext_counts.entry(hash).or_insert(0) += 1;
        } else if let Some(count) = self.ext_counts.get_mut(&hash) {
            *count -= 1;
            if *count == 0 {
                self.note_capacity(RebuildTable::ExtCounts, self.ext_counts.capacity());
                self.ext_counts.remove(&hash);
            }
        }
    }

    fn name_insert(&mut self, parent_id: u64, hash: u32, id: u64) {
        let entries = match self.names.get_mut(&parent_id) {
            Some(entries) => entries,
            None => {
                self.names_vec_bytes = self.names_vec_bytes.saturating_add(entries_bytes(1));
                self.names
                    .entry(parent_id)
                    .or_insert_with(|| Vec::with_capacity(1))
            }
        };
        if let Err(position) = entries.binary_search(&(hash, id)) {
            let before = entries.capacity();
            entries.insert(position, (hash, id));
            if entries.capacity() != before {
                self.names_vec_bytes = self
                    .names_vec_bytes
                    .saturating_sub(entries_bytes(before))
                    .saturating_add(entries_bytes(entries.capacity()));
            }
        }
    }

    fn name_remove(&mut self, parent_id: u64, hash: u32, id: u64) {
        let Some(entries) = self.names.get_mut(&parent_id) else {
            return;
        };
        if let Ok(position) = entries.binary_search(&(hash, id)) {
            entries.remove(position);
        }
        if entries.is_empty() {
            let capacity = entries.capacity();
            self.note_capacity(RebuildTable::NameIndex, self.names.capacity());
            self.names.remove(&parent_id);
            self.names_vec_bytes = self.names_vec_bytes.saturating_sub(entries_bytes(capacity));
        }
    }

    /// The row `old` left the state: its name entry and name allocation go.
    pub(crate) fn section_out(&mut self, old: &Section) {
        self.name_remove(old.parent_id, name_hash(&old.name), old.section_id);
        self.section_names_bytes = self
            .section_names_bytes
            .saturating_sub(allocation_bytes(old.name.capacity() as u64));
    }

    /// The row `new` entered the state: its name entry and name allocation
    /// come.
    pub(crate) fn section_in(&mut self, new: &Section) {
        self.name_insert(new.parent_id, name_hash(&new.name), new.section_id);
        self.section_names_bytes = self
            .section_names_bytes
            .saturating_add(allocation_bytes(new.name.capacity() as u64));
    }

    /// A children list of `old` capacity (`None`: none) became one of `new`
    /// capacity (`None`: removed).
    pub(crate) fn list_replaced(&mut self, old: Option<usize>, new: Option<usize>) {
        if let Some(old) = old {
            self.lists_bytes = self.lists_bytes.saturating_sub(list_bytes(old));
        }
        if let Some(new) = new {
            self.lists_bytes = self.lists_bytes.saturating_add(list_bytes(new));
        }
    }

    /// A field row named `name` went from `old` to `new` (each `None` for no
    /// row, `Some(None)` for an inline one), interning or releasing `name` as
    /// `names` says.
    pub(crate) fn field_row_written(
        &mut self,
        name: &str,
        old: Option<Option<[u8; 32]>>,
        new: Option<Option<[u8; 32]>>,
        names: NameChange,
    ) {
        self.value_row_written(|ix| &mut ix.ext_field_rows, old, new);
        match names {
            NameChange::Unchanged => {}
            NameChange::Interned => {
                self.interned_names_bytes = self
                    .interned_names_bytes
                    .saturating_add(interned_name_bytes(name.len()));
            }
            NameChange::Released => {
                self.interned_names_bytes = self
                    .interned_names_bytes
                    .saturating_sub(interned_name_bytes(name.len()));
            }
        }
    }

    /// A time-sample row went from `old` to `new`, as
    /// [`Self::field_row_written`].
    pub(crate) fn sample_row_written(
        &mut self,
        old: Option<Option<[u8; 32]>>,
        new: Option<Option<[u8; 32]>>,
    ) {
        self.value_row_written(|ix| &mut ix.ext_sample_rows, old, new);
    }

    /// Set `depth_ok`: a rollback restores the plan's starting flag.
    pub(crate) fn set_depth_ok(&mut self, depth_ok: bool) {
        self.depth_ok = depth_ok;
    }

    fn value_row_written(
        &mut self,
        external_rows: fn(&mut Self) -> &mut u64,
        old: Option<Option<[u8; 32]>>,
        new: Option<Option<[u8; 32]>>,
    ) {
        if let Some(Some(hash)) = old {
            self.count_hash(hash, false);
            let rows = external_rows(self);
            *rows = rows.saturating_sub(1);
        }
        if let Some(Some(hash)) = new {
            self.count_hash(hash, true);
            *external_rows(self) += 1;
        }
    }

    /// The indexes of an empty state, keeping `depth_ok`: what a reset
    /// leaves.
    pub(crate) fn emptied(&self) -> PlanIndexes {
        PlanIndexes {
            depth_ok: self.depth_ok,
            ..PlanIndexes::default()
        }
    }

    /// Release the spare capacity of the indexes' own maps, as the cache
    /// compacts the state's tables before it weighs an entry. A no-op on maps
    /// already minimal for their length.
    pub fn compact(&mut self) {
        self.names.shrink_to_fit();
        self.ext_counts.shrink_to_fit();
    }
}

/// `domain::replay::normalize_children` over the children lists in `lists`
/// only: each that is empty, or keyed by neither the root nor a live row, is
/// dropped, its allocation taken off `indexes`' list sum and the children
/// table's capacity noted before the removal
/// ([`PlanIndexes::note_capacity`]); each that stays is
/// shrunk to its length (`Vec::shrink_to_fit`), the sum following its
/// capacity. The parent ids of the non-empty lists dropped as dangling are
/// returned, sorted ascending and unique.
///
/// The shrink is what keeps a cached entry's lists exact across commits:
/// the engine grows a list in place (`delta_write`), and the cache's
/// compaction (`compact_state`) trims tables, not lists, so without it every
/// list a plan or replayed record touched would keep up to twice its length,
/// and the entry's capacity-charged weight would drift up against the
/// per-entry cap. O(touched lists): one reallocation per list that has spare
/// capacity.
///
/// Exact for a plan or a replayed record from a normalized state, given the
/// lists it wrote (`PlanTouches::lists`): the state held no empty list and no
/// list under a missing parent before it, and the engine's only writes that
/// can leave either — a list emptied, a parent's row removed — write that
/// list too (a cascade removes the deleted row's list with it; a reset takes
/// every list).
pub fn normalize_touched(
    state: &mut CompactState,
    indexes: &mut PlanIndexes,
    lists: &HashSet<u64>,
) -> Vec<u64> {
    let mut dangling = Vec::new();
    for &parent_id in lists {
        let live = parent_id == 0 || state.sections.contains_key(&parent_id);
        let Some(list) = state.children.get_mut(&parent_id) else {
            continue;
        };
        if !list.is_empty() && live {
            // A list kept: trimmed to its length, as a cold start's lists
            // are, so a list the engine grew in place does not keep up to
            // twice its length in the entry (and in its weight).
            let before = list.capacity();
            list.shrink_to_fit();
            if list.capacity() != before {
                indexes.list_replaced(Some(before), Some(list.capacity()));
            }
            continue;
        }
        if !list.is_empty() {
            dangling.push(parent_id);
        }
        // A removal from the children table: its capacity first, before a
        // tombstone can hide its buckets (`PlanIndexes::note_capacity`).
        indexes.note_capacity(RebuildTable::Children, state.children.capacity());
        if let Some(list) = state.children.remove(&parent_id) {
            indexes.lists_bytes = indexes
                .lists_bytes
                .saturating_sub(list_bytes(list.capacity()));
        }
    }
    dangling.sort_unstable();
    dangling
}

/// Whether `state` has the shape the incremental depth check relies on, the
/// shape the engine keeps once it holds:
///
/// 1. every section whose parent is the root or a live row is listed in its
///    parent's children list;
/// 2. every listed id has a row naming that list's owner;
/// 3. every children list is keyed by the root or a live row;
/// 4. no list holds an id twice.
///
/// Under 1–3 a section's parent pointer and its listing agree, so its depth
/// can be read by walking parent pointers; under 4 a list's members are each
/// reached once.
pub fn shape_ok(state: &CompactState) -> bool {
    let listed_ok = state.children.iter().all(|(&parent_id, list)| {
        (parent_id == 0 || state.sections.contains_key(&parent_id))
            && list.iter().all(|child| {
                state
                    .sections
                    .get(child)
                    .is_some_and(|row| row.parent_id == parent_id)
            })
            && {
                let mut seen = HashSet::with_capacity(list.len());
                list.iter().all(|child| seen.insert(*child))
            }
    });
    if !listed_ok {
        return false;
    }
    // Rule 1, by counting: under rules 2–4 every listed id is a distinct row
    // naming its list's owner, so the rows whose parent keeps a list number
    // at least the listed ids, and equal them exactly when every such row is
    // listed.
    let listed: usize = state.children.values().map(Vec::len).sum();
    let listable = state
        .sections
        .values()
        .filter(|row| row.parent_id == 0 || state.sections.contains_key(&row.parent_id))
        .count();
    listed == listable
        && state
            .sections
            .values()
            .filter(|row| row.parent_id == 0 || state.sections.contains_key(&row.parent_id))
            .all(|row| state.children.contains_key(&row.parent_id))
}

/// The end-of-delta depth verdict for a plan whose state passed the full
/// walk (`depth_ok`), from the ids its listings inserted, `listed` (each
/// `(parent, child)`): exactly [`check_section_tree_depth_capped`]'s verdict
/// at `cap` on such a state, in work proportional to the listed subtrees and
/// their depths rather than to the layer. `None` when the state's shape no
/// longer supports the argument and the caller must run the full walk and
/// clear the flag.
///
/// A section can become deeper, or newly reachable from the root, only if it
/// or an ancestor was inserted into a list by the delta, and every such
/// insert is in `listed`. So each listed child that is still a row naming
/// its list's owner is walked up its parent pointers to the root (memoised);
/// one that does not reach the root is detached, and the walk skips it, as
/// the full walk does. From each reachable one its listed subtree is walked
/// down with its depth. Every other section keeps the depth it had when the
/// state last passed.
///
/// A breach reports `depth: cap + 1`, as the full walk does on such a state:
/// its DFS meets a section at depth `cap + 1` before anything deeper.
fn incremental_depth_verdict(
    sections: &HashMap<u64, Section>,
    children: &HashMap<u64, Vec<u64>>,
    listed: &[(u64, u64)],
    cap: usize,
    walk: &mut dyn FnMut(u64),
) -> Option<Result<(), DeltaError>> {
    let too_deep = || {
        Err(DeltaError::SectionTreeTooDeep {
            depth: cap + 1,
            cap,
        })
    };
    // Depths found walking up, by id; the root is at 0.
    let mut depth_of: HashMap<u64, Option<usize>> = HashMap::new();
    let mut roots: Vec<(u64, usize)> = Vec::new();
    for &(parent_id, child_id) in listed {
        let still_listed = || {
            children
                .get(&parent_id)
                .is_some_and(|list| list.contains(&child_id))
        };
        let Some(row) = sections.get(&child_id) else {
            // Listed and later deleted: no row, nothing to walk — unless it
            // is still listed, a ghost listing the full walk would descend
            // into and the shape argument does not cover.
            if still_listed() {
                return None;
            }
            continue;
        };
        if row.parent_id != parent_id {
            // Listed under `parent_id`, then moved elsewhere by the same
            // delta: its later listing is in `listed` too. Still listed here
            // means a listing that disagrees with its row.
            if still_listed() {
                return None;
            }
            continue;
        }
        if !still_listed() {
            // Listed, then taken out again (and not deleted): unreachable
            // through this listing; if it was relisted, that is in `listed`.
            continue;
        }
        // Walk up: the path until a known depth, the root, a missing row or
        // a repeat (a parent-pointer cycle, which never reaches the root).
        let mut path: Vec<u64> = Vec::new();
        let mut on_path: HashSet<u64> = HashSet::new();
        let mut cursor = child_id;
        let base: Option<usize> = loop {
            if cursor == 0 {
                break Some(0);
            }
            if let Some(&known) = depth_of.get(&cursor) {
                break known;
            }
            // One hop up, projected before the walk's scratch grows by it:
            // its `on_path` and `path` entries, and the `depth_of` entry
            // recorded for it below. (A repeat projects one hop it does not
            // take, the safe direction.)
            walk(1);
            if !on_path.insert(cursor) {
                break None;
            }
            if path.len() > cap + 1 {
                // Deeper than any reachable section may be: let the full walk
                // decide (and report) it.
                return None;
            }
            path.push(cursor);
            match sections.get(&cursor) {
                Some(row) => cursor = row.parent_id,
                None => break None,
            }
        };
        // Record the path's depths: the last element sits directly under the
        // known base.
        let mut depth = base;
        for &id in path.iter().rev() {
            depth = depth.map(|d| d + 1);
            depth_of.insert(id, depth);
        }
        if let Some(Some(depth)) = depth_of.get(&child_id) {
            roots.push((child_id, *depth));
        }
    }
    // Walk each reachable listed subtree down, sharing one visited set. Each
    // root, and each list before it is pushed, is projected first (`walk`),
    // as each hop of the walk up was.
    let mut visited: HashSet<u64> = HashSet::new();
    let mut stack: Vec<(u64, usize)> = Vec::new();
    for (root, depth) in roots {
        walk(1);
        stack.push((root, depth));
        while let Some((id, depth)) = stack.pop() {
            if !visited.insert(id) {
                continue;
            }
            if depth > cap {
                return Some(too_deep());
            }
            if let Some(list) = children.get(&id) {
                walk(list.len() as u64);
                stack.extend(list.iter().map(|&child| (child, depth + 1)));
            }
        }
    }
    Some(Ok(()))
}

/// The depth verdict for `state` after a plan that listed `listed`, at `cap`:
/// the incremental verdict while `depth_ok` holds, the full walk otherwise
/// (and when the incremental check declines). `depth_ok` is cleared when the
/// incremental check finds a listing that breaks the shape it relies on.
/// `walk(n)` is called before the walk grows by `n` sections: on the
/// incremental walk, per hop of each walk up (before its memo and path
/// entries) and per list (and root) of each walk down; every section once
/// before a full walk.
pub(crate) fn depth_verdict<R: crate::domain::types::ValueRows>(
    state: &ObjectStateOf<R>,
    listed: &[(u64, u64)],
    depth_ok: &mut bool,
    cap: usize,
    walk: &mut dyn FnMut(u64),
) -> Result<(), DeltaError> {
    if *depth_ok {
        if let Some(verdict) =
            incremental_depth_verdict(&state.sections, &state.children, listed, cap, walk)
        {
            return verdict;
        }
        *depth_ok = false;
    }
    walk(state.sections.len() as u64);
    check_section_tree_depth_capped(state, cap)
}

/// The engine hooks of the structural cache's hooked plans and replays: they
/// keep the entry's [`PlanIndexes`] in step with every write, answer the
/// engine's name and adoption lookups from them, record what the delta
/// touched ([`PlanTouches`]), and give its depth verdict incrementally. A
/// plan in place also records, in its [`PlanJournal`], the pre-plan value of
/// every key on its first touch, so it can be rolled back; a replay has none.
///
/// With a journal, the hooks also count, per [`RebuildTable`], the removals
/// a rollback may have to re-create, and project a table's rollback rebuild
/// through `projector` (the bounded plan's meter) before the write that can
/// make it necessary (`PlanJournal::note_removal`,
/// `PlanJournal::note_insert_call`; module docs of `domain::plan_journal`).
/// They also project, as the plan drops a name-index parent key, that key's
/// vector at its capacity, the most a rollback restoring the parent's
/// entries allocates (`PlanJournal::note_vector_drop`; a later drop of the
/// same parent projects again, an over-count).
pub(crate) struct IndexedHooks<'e> {
    indexes: &'e mut PlanIndexes,
    touches: &'e mut PlanTouches,
    journal: Option<&'e mut PlanJournal>,
    projector: Option<&'e dyn RebuildProjector>,
}

/// What a hooked plan or replay touched, beyond the mutations it emits.
#[derive(Debug, Default, Clone)]
pub struct PlanTouches {
    /// `(parent, child)` for every id a list gained, in write order.
    pub listed: Vec<(u64, u64)>,
    /// Every children-list key written (created, edited or removed).
    pub lists: HashSet<u64>,
    /// Whether the delta was a reset.
    pub reset: bool,
}

impl<'e> IndexedHooks<'e> {
    /// Hooks keeping `indexes` and recording into `touches`.
    pub(crate) fn new(indexes: &'e mut PlanIndexes, touches: &'e mut PlanTouches) -> Self {
        Self {
            indexes,
            touches,
            journal: None,
            projector: None,
        }
    }

    /// The same hooks, also journalling every first touch into `journal`
    /// and projecting each rollback rebuild through `projector`.
    pub(crate) fn journalled(
        mut self,
        journal: &'e mut PlanJournal,
        projector: &'e dyn RebuildProjector,
    ) -> Self {
        self.journal = Some(journal);
        self.projector = Some(projector);
        self
    }

    /// A key is about to leave `table`, with `room`: counted, and its
    /// rebuild projected when it could now be needed.
    fn note_removal(&mut self, table: RebuildTable, room: usize, pre_existing: bool) {
        let Some(journal) = self.journal.as_deref_mut() else {
            return;
        };
        if let Some(bytes) = journal.note_removal(table, room as u64, pre_existing) {
            if let Some(projector) = self.projector {
                projector.project_rebuild(bytes);
            }
        }
    }

    /// An insert call is about to land on `table`, with `room`.
    fn note_insert_call(&mut self, table: RebuildTable, room: usize) {
        let Some(journal) = self.journal.as_deref_mut() else {
            return;
        };
        if let Some(bytes) = journal.note_insert_call(table, room as u64) {
            if let Some(projector) = self.projector {
                projector.project_rebuild(bytes);
            }
        }
    }

    /// The external-hash counts' side of a value row going from `old` to
    /// `new`: the removal of `old`'s hash key when its count is about to
    /// reach 0, and the insert call that puts `new`'s back or adds it.
    fn note_hash_change(&mut self, old: RowImage, new: RowImage) {
        let (old, new) = (old.flatten(), new.flatten());
        if old == new && old.is_some_and(|hash| self.indexes.hash_count(&hash) > 1) {
            return;
        }
        let room = self.indexes.hash_key_room();
        let mut gone = None;
        if let Some(hash) = old {
            if self.indexes.hash_count(&hash) == 1 {
                gone = Some(hash);
                self.note_removal(RebuildTable::ExtCounts, room, true);
            }
        }
        if let Some(hash) = new {
            if self.indexes.hash_count(&hash) == 0 || gone == Some(hash) {
                self.note_insert_call(RebuildTable::ExtCounts, room);
            }
        }
    }
}

/// The incremental depth verdict at `cap` for a state that passed the full
/// walk, given the `(parent, child)` listings a delta made: `None` when the
/// state's shape no longer supports it (the caller would run the full walk).
/// For equivalence tests, which compare it with [`full_depth_verdict`] at a
/// small cap so breaches are frequent.
#[cfg(any(test, feature = "test-support"))]
pub fn incremental_depth_verdict_for_test(
    state: &CompactState,
    listed: &[(u64, u64)],
    cap: usize,
) -> Option<Result<(), DeltaError>> {
    incremental_depth_verdict(&state.sections, &state.children, listed, cap, &mut |_| {})
}

/// The engine's full section-tree depth walk at `cap`, the verdict
/// [`incremental_depth_verdict_for_test`] must reproduce.
#[cfg(any(test, feature = "test-support"))]
pub fn full_depth_verdict(state: &CompactState, cap: usize) -> Result<(), DeltaError> {
    check_section_tree_depth_capped(state, cap)
}

/// **Test support: the table-registry guard's enumeration.** Every hash
/// table `state` owns — its own maps, its field store's and its name
/// interner's — each with the [`RebuildTable`] it is registered as and its
/// capacities (`domain::types::MapCapacities`: `capacity()` and the true
/// capacity of the buckets it holds). Destructured field by field, with no
/// `..`, here and in the stores' own enumerations
/// (`CompactFields::hash_tables_for_test`,
/// `CompactSamples::hash_tables_for_test`), so a field added to the state or
/// a store fails to build until it is classified: a hash table, listed with
/// its registry entry, or not one. The guard test checks that the registry
/// reads each listed table and lists none twice.
#[cfg(any(test, feature = "test-support"))]
pub fn state_hash_tables_for_test(
    state: &CompactState,
) -> Vec<(RebuildTable, crate::domain::types::MapCapacities)> {
    use crate::domain::types::MapCapacities;
    let ObjectStateOf {
        object_id: _,
        sequence_num: _,
        first_version: _,
        source_format: _,
        next_set_order: _,
        sections,
        children,
        fields,
        time_samples,
    } = state;
    let (field_rows, name_ids) = fields.hash_tables_for_test();
    vec![
        (RebuildTable::Sections, MapCapacities::of(sections)),
        (RebuildTable::Children, MapCapacities::of(children)),
        (RebuildTable::FieldRows, field_rows),
        (
            RebuildTable::SampleRows,
            time_samples.hash_tables_for_test(),
        ),
        (RebuildTable::NameIds, name_ids),
    ]
}

/// **Test support.** [`state_hash_tables_for_test`] of `state`, then
/// `indexes`' own ([`PlanIndexes::hash_tables_for_test`]): every hash table
/// of an entry holding plan indexes.
#[cfg(any(test, feature = "test-support"))]
pub fn hash_tables_for_test(
    state: &CompactState,
    indexes: &PlanIndexes,
) -> Vec<(RebuildTable, crate::domain::types::MapCapacities)> {
    let mut tables = state_hash_tables_for_test(state);
    tables.extend(indexes.hash_tables_for_test());
    tables
}

/// **Test support.** The true capacity (`domain::types::true_capacity_for_test`)
/// of the state's table `table`: the usable slots of the buckets it holds,
/// which its `capacity()` under-reads by its tombstones. `None` for one of
/// the plan indexes' own maps. O(the table).
#[cfg(any(test, feature = "test-support"))]
pub fn true_state_table_capacity_for_test(
    table: RebuildTable,
    state: &CompactState,
) -> Option<usize> {
    state_hash_tables_for_test(state)
        .into_iter()
        .find(|&(listed, _)| listed == table)
        .map(|(_, capacities)| capacities.true_capacity)
}

#[cfg(any(test, feature = "test-support"))]
impl PlanIndexes {
    /// **Test support.** The indexes' own hash tables with their registry
    /// entries and capacities, destructured as
    /// [`state_hash_tables_for_test`] does the state.
    pub fn hash_tables_for_test(&self) -> Vec<(RebuildTable, crate::domain::types::MapCapacities)> {
        use crate::domain::types::MapCapacities;
        let PlanIndexes {
            names,
            names_vec_bytes: _,
            ext_counts,
            ext_field_rows: _,
            ext_sample_rows: _,
            interned_names_bytes: _,
            section_names_bytes: _,
            lists_bytes: _,
            depth_ok: _,
            capacity_high_water: _,
        } = self;
        vec![
            (RebuildTable::NameIndex, MapCapacities::of(names)),
            (RebuildTable::ExtCounts, MapCapacities::of(ext_counts)),
        ]
    }

    /// **Test support.** [`Self::weight_bytes`] with each of the indexes' own
    /// maps at its true capacity (`domain::types::true_capacity_for_test`)
    /// instead of its held one: what they really hold. O(the maps).
    pub fn weight_bytes_at_true_capacity_for_test(&self) -> u64 {
        self.hash_tables_for_test().into_iter().fold(
            (size_of::<PlanIndexes>() as u64).saturating_add(self.names_vec_bytes),
            |sum, (table, capacities)| {
                sum.saturating_add(table_bytes(
                    capacities.true_capacity,
                    table_slot_bytes(table),
                ))
            },
        )
    }
}

impl EngineHooks<CompactRows> for IndexedHooks<'_> {
    const OBSERVES: bool = true;

    fn section_before(&mut self, id: u64, old: Option<&Section>) {
        if let Some(journal) = self.journal.as_deref_mut() {
            journal.section_touched(id, old);
        }
        if let Some(old) = old {
            if self.indexes.entry_out_removes_key(old) {
                let room = self.indexes.parent_key_room();
                self.note_removal(RebuildTable::NameIndex, room, true);
                // The vector goes with its key. A rollback rebuilds or
                // re-creates the parent's vector at its pre-plan length at
                // most, which the first vector dropped for that parent —
                // the pre-plan one, perhaps grown — covers; a later drop of
                // the same parent's key projects again, an over-count.
                let bytes = self.indexes.parent_vector_bytes(old.parent_id);
                if let Some(journal) = self.journal.as_deref_mut() {
                    if let Some(bytes) = journal.note_vector_drop(bytes) {
                        if let Some(projector) = self.projector {
                            projector.project_rebuild(bytes);
                        }
                    }
                }
            }
            self.indexes.section_out(old);
        }
    }

    fn section_after(&mut self, _id: u64, new: Option<&Section>) {
        if let Some(new) = new {
            if self.indexes.entry_in_adds_key(new) {
                let room = self.indexes.parent_key_room();
                self.note_insert_call(RebuildTable::NameIndex, room);
            }
            self.indexes.section_in(new);
        }
    }

    fn list_before(&mut self, parent: u64, old: Option<&Vec<u64>>) {
        if let Some(journal) = self.journal.as_deref_mut() {
            journal.list_touched(parent, old);
        }
        self.touches.lists.insert(parent);
        self.indexes.list_replaced(old.map(Vec::capacity), None);
    }

    fn list_after(&mut self, _parent: u64, new: Option<&Vec<u64>>) {
        self.indexes.list_replaced(None, new.map(Vec::capacity));
    }

    fn child_listed(&mut self, parent: u64, child: u64) {
        self.touches.listed.push((parent, child));
    }

    fn field_written(
        &mut self,
        section_id: u64,
        name: &str,
        old: Option<Option<[u8; 32]>>,
        new: Option<Option<[u8; 32]>>,
        names: NameChange,
    ) {
        if let Some(journal) = self.journal.as_deref_mut() {
            journal.field_touched(section_id, name, old);
            if names == NameChange::Interned {
                journal.name_interned(name);
            }
        }
        self.note_hash_change(old, new);
        self.indexes.field_row_written(name, old, new, names);
    }

    fn sample_written(
        &mut self,
        section_id: u64,
        time_bits: u64,
        old: Option<Option<[u8; 32]>>,
        new: Option<Option<[u8; 32]>>,
    ) {
        if let Some(journal) = self.journal.as_deref_mut() {
            journal.sample_touched(section_id, time_bits, old);
        }
        self.note_hash_change(old, new);
        self.indexes.sample_row_written(old, new);
    }

    fn insert_call(&mut self, table: RebuildTable, room: usize) {
        self.note_insert_call(table, room);
    }

    fn removing(&mut self, key: RemovedKey<'_>, room: usize, capacity: usize) {
        self.indexes.note_capacity(key.table(), capacity);
        if let (RemovedKey::FieldName(name, id), Some(journal)) = (key, self.journal.as_deref_mut())
        {
            journal.name_released(name, id);
        }
        let pre_existing = self
            .journal
            .as_deref()
            .is_some_and(|journal| journal.key_pre_existed(key));
        self.note_removal(key.table(), room, pre_existing);
    }

    fn header_before(&mut self, header: HeaderImage<'_>) {
        if let Some(journal) = self.journal.as_deref_mut() {
            journal.header_touched(header);
        }
    }

    fn reset(&mut self, taken: ResetTables<CompactRows>) {
        self.touches.reset = true;
        self.touches.lists.extend(taken.children.keys().copied());
        let emptied = self.indexes.emptied();
        let before = std::mem::replace(&mut *self.indexes, emptied);
        if let Some(journal) = self.journal.as_deref_mut() {
            journal.reset_taken(taken, before);
        }
    }

    fn child_by_name(
        &self,
        sections: &HashMap<u64, Section>,
        parent_id: u64,
        name: &str,
        except_id: Option<u64>,
    ) -> Option<Option<u64>> {
        Some(
            self.indexes
                .child_by_name(sections, parent_id, name, except_id),
        )
    }

    fn rows_naming(&self, parent_id: u64, out: &mut Vec<u64>) -> bool {
        out.extend(self.indexes.rows_naming(parent_id));
        true
    }

    fn depth_verdict(
        &mut self,
        state: &CompactState,
        walk: &mut dyn FnMut(u64),
    ) -> Option<Result<(), DeltaError>> {
        Some(depth_verdict(
            state,
            &self.touches.listed,
            &mut self.indexes.depth_ok,
            MAX_SECTION_TREE_DEPTH,
            walk,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(id: u64, parent: u64, name: &str) -> Section {
        Section {
            section_id: id,
            parent_id: parent,
            name: name.to_owned(),
            section_type: 0,
            section_order: 0,
            first_version: 1,
            last_version: 1,
        }
    }

    fn state_of(rows: &[(u64, u64, &str)]) -> CompactState {
        let mut state = CompactState::default();
        for &(id, parent, name) in rows {
            state.sections.insert(id, row(id, parent, name));
        }
        for &(id, parent, _) in rows {
            if parent == 0 || state.sections.contains_key(&parent) {
                state.children.entry(parent).or_default().push(id);
            }
        }
        state
    }

    #[test]
    fn child_by_name_answers_the_lowest_surviving_id_of_a_duplicate_name() {
        let state = state_of(&[(7, 0, "a"), (3, 0, "a"), (5, 0, "b")]);
        let indexes = PlanIndexes::build(&state);
        assert_eq!(
            indexes.child_by_name(&state.sections, 0, "a", None),
            Some(3)
        );
        assert_eq!(
            indexes.child_by_name(&state.sections, 0, "a", Some(3)),
            Some(7)
        );
        assert_eq!(indexes.child_by_name(&state.sections, 0, "c", None), None);
    }

    #[test]
    fn build_sets_depth_ok_only_on_a_well_shaped_state() {
        let well = state_of(&[(1, 0, "a"), (2, 1, "b")]);
        assert!(PlanIndexes::build(&well).depth_ok());
        let mut stale = well.clone();
        stale.children.get_mut(&0).expect("root list").push(2);
        assert!(!PlanIndexes::build(&stale).depth_ok());
    }

    #[test]
    fn incremental_verdict_reports_the_full_walks_breach_at_a_small_cap() {
        // A chain 1 → 2 → 3 under the root: depth 3.
        let state = state_of(&[(1, 0, "a"), (2, 1, "b"), (3, 2, "c")]);
        let listed = [(2, 3)];
        let full = check_section_tree_depth_capped(&state, 2)
            .unwrap_err()
            .to_string();
        let incremental = incremental_depth_verdict_for_test(&state, &listed, 2)
            .expect("well-shaped")
            .unwrap_err()
            .to_string();
        assert_eq!(incremental, full);
        assert!(incremental_depth_verdict_for_test(&state, &listed, 3)
            .expect("well-shaped")
            .is_ok());
    }

    /// A chain `1 → 2 → … → n` under the root.
    fn chain_of(n: u64) -> CompactState {
        let rows: Vec<(u64, u64, &str)> = (1..=n).map(|id| (id, id - 1, "c")).collect();
        state_of(&rows)
    }

    /// T3 at the real cap: a chain of `MAX_SECTION_TREE_DEPTH` sections passes
    /// the full walk; a child listed under its deepest section breaches it,
    /// and the incremental verdict reports the full walk's exact error
    /// (`depth: MAX + 1`); a child listed there and moved back to the root by
    /// the same delta (briefly too deep, legal at the end) passes both.
    #[test]
    fn incremental_verdict_equals_the_full_walk_at_the_real_cap() {
        let cap = MAX_SECTION_TREE_DEPTH;
        let deepest = cap as u64;
        let mut state = chain_of(deepest);
        assert!(
            check_section_tree_depth_capped(&state, cap).is_ok(),
            "fixture: a chain at the cap passes"
        );
        let extra = deepest + 1;
        state.sections.insert(extra, row(extra, deepest, "x"));
        state.children.entry(deepest).or_default().push(extra);
        let incremental = incremental_depth_verdict_for_test(&state, &[(deepest, extra)], cap)
            .expect("well-shaped");
        let full = check_section_tree_depth_capped(&state, cap);
        assert!(
            matches!(
                incremental,
                Err(DeltaError::SectionTreeTooDeep { depth, cap: c }) if depth == cap + 1 && c == cap
            ),
            "the breach reports depth MAX + 1: {incremental:?}"
        );
        assert_eq!(
            incremental.map_err(|e| e.to_string()),
            full.map_err(|e| e.to_string())
        );

        // Listed under the deepest section, then moved back under the root.
        state
            .children
            .get_mut(&deepest)
            .expect("the deepest list")
            .retain(|&id| id != extra);
        if state.children.get(&deepest).is_some_and(Vec::is_empty) {
            state.children.remove(&deepest);
        }
        state.sections.get_mut(&extra).expect("the row").parent_id = 0;
        state.children.entry(0).or_default().push(extra);
        let listed = [(deepest, extra), (0, extra)];
        assert_eq!(
            incremental_depth_verdict_for_test(&state, &listed, cap)
                .expect("well-shaped")
                .map_err(|e| e.to_string()),
            check_section_tree_depth_capped(&state, cap).map_err(|e| e.to_string()),
        );
        assert!(check_section_tree_depth_capped(&state, cap).is_ok());
    }

    /// T3's detached cycle: two rows naming each other, each listed in the
    /// other's list, reach the root by no path; a listing into the cycle is
    /// skipped by the incremental check as by the full walk, which never
    /// enters it.
    #[test]
    fn incremental_verdict_skips_a_detached_cycle_as_the_full_walk_does() {
        let mut state = state_of(&[(1, 0, "a")]);
        state.sections.insert(10, row(10, 11, "p"));
        state.sections.insert(11, row(11, 10, "q"));
        state.children.insert(10, vec![11]);
        state.children.insert(11, vec![10]);
        assert!(
            shape_ok(&state),
            "fixture: the cycle is listed consistently"
        );
        for cap in [1, 2, MAX_SECTION_TREE_DEPTH] {
            assert_eq!(
                incremental_depth_verdict_for_test(&state, &[(10, 11)], cap)
                    .expect("well-shaped")
                    .map_err(|e| e.to_string()),
                check_section_tree_depth_capped(&state, cap).map_err(|e| e.to_string()),
                "cap {cap}"
            );
        }
    }

    /// `build_peak_bytes_for` is `build_peak_bytes` at the state's counts —
    /// its sections, its lists' entries summed, its field and time-sample
    /// rows — and bounds what the build leaves allocated, on an empty state,
    /// a flat one, a deep one, and one with value rows.
    #[test]
    fn build_peak_bytes_for_is_the_peak_at_the_states_counts_and_bounds_the_build() {
        let empty = CompactState::default();
        let flat = state_of(
            &(1..=1_000u64)
                .map(|id| (id, 0, "f"))
                .collect::<Vec<(u64, u64, &str)>>(),
        );
        let deep = chain_of(2_000);
        let mut with_rows = state_of(&[(1, 0, "a"), (2, 1, "b"), (3, 1, "c")]);
        for id in 1..=3u64 {
            for (i, name) in ["n0", "n1", "n2", "n3"].into_iter().enumerate() {
                with_rows
                    .fields
                    .insert(id, name, (i % 2 == 0).then_some([id as u8; 32]));
            }
            with_rows
                .time_samples
                .insert(id, 1.0f64.to_bits(), Some([9; 32]));
        }
        for (label, state) in [
            ("empty", &empty),
            ("flat", &flat),
            ("deep", &deep),
            ("value rows", &with_rows),
        ] {
            let list_entries: u64 = state.children.values().map(|l| l.len() as u64).sum();
            let value_rows =
                (FieldRows::len(&state.fields) + SampleRows::len(&state.time_samples)) as u64;
            assert_eq!(
                PlanIndexes::build_peak_bytes_for(state),
                PlanIndexes::build_peak_bytes(
                    state.sections.len() as u64,
                    list_entries,
                    value_rows
                ),
                "{label}: the peak at the state's counts"
            );
            assert!(
                PlanIndexes::build_peak_bytes_for(state)
                    >= PlanIndexes::build(state).weight_bytes(),
                "{label}: the peak bounds the indexes the build leaves"
            );
        }
        // Monotone in each count.
        let base = PlanIndexes::build_peak_bytes(100, 100, 100);
        for grown in [
            PlanIndexes::build_peak_bytes(200, 100, 100),
            PlanIndexes::build_peak_bytes(100, 200, 100),
            PlanIndexes::build_peak_bytes(100, 100, 200),
        ] {
            assert!(grown > base, "{grown} > {base}");
        }
    }

    /// `normalize_touched` over a touched non-empty list keyed by a parent
    /// with no row (dangling) and a touched empty list under a live parent
    /// does what `domain::replay::normalize_children` does to the whole
    /// state: both lists dropped, the dangling parent reported, the state
    /// equal; and the indexes stay in step — their content equals a rebuild
    /// of the normalized state (the list allocation sum included) and their
    /// weight its recomputation. The dangling branch is one no generated
    /// plan reaches (a plan never lists under a missing parent), so the
    /// properties in `tests/proptests.rs` leave it to this test.
    #[test]
    fn normalize_touched_drops_a_dangling_list_and_an_empty_one_as_normalize_children_does() {
        const DANGLING_PARENT: u64 = 50;
        const ORPHAN: u64 = 7;
        let mut state = state_of(&[(1, 0, "a"), (2, 1, "b")]);
        // An orphan naming the missing parent, listed under it as an earlier
        // build left it, and an empty list with spare capacity under the live
        // row 2.
        state
            .sections
            .insert(ORPHAN, row(ORPHAN, DANGLING_PARENT, "o"));
        state.children.insert(DANGLING_PARENT, vec![ORPHAN]);
        state.children.insert(2, Vec::with_capacity(4));
        let mut indexes = PlanIndexes::build(&state);
        let mut expected = state.clone();
        let expected_dangling = crate::domain::replay::normalize_children(&mut expected);
        assert_eq!(
            expected_dangling,
            vec![DANGLING_PARENT],
            "fixture: the full normalization reports the dangling parent"
        );

        let dangling = normalize_touched(
            &mut state,
            &mut indexes,
            &HashSet::from([DANGLING_PARENT, 2]),
        );

        assert_eq!(state, expected, "the state normalize_children leaves");
        assert_eq!(dangling, expected_dangling, "the same dangling parents");
        assert!(
            !state.children.contains_key(&DANGLING_PARENT) && !state.children.contains_key(&2),
            "both lists dropped"
        );
        assert!(
            indexes.same_content_as(&PlanIndexes::build(&state)),
            "the indexes, list allocation sum included, equal a rebuild"
        );
        assert_eq!(indexes.weight_bytes(), indexes.weight_bytes_recomputed());
    }

    /// The name-index vectors a rollback allocates — a vector rebuilt for a
    /// parent whose key the plan dropped and re-created short (step 2 of
    /// `restore_name_index`), a key re-created whole (step 4) — and the tables
    /// it rebuilds are within what the plan projected for its rollback
    /// (`RewriteCount::rollback_rebuild_bytes`): the shadow check compares the
    /// rebuilt tables with the projected set, but not the vectors with the
    /// dropped-vector projection, so this does. Allocations are read from
    /// the name map's vectors before and after the rollback: a parent new to
    /// the map, or whose vector moved, allocated its vector.
    #[test]
    fn rollback_allocates_its_name_vectors_within_the_plans_projection() {
        use crate::domain::delta::{
            apply_delta_bounded_indexed, ApplyDeltaStats, BoundedPlan, MutationAllowance,
            RewriteCount,
        };
        use crate::domain::plan_journal::PlanJournal;
        use oodf_ffi::{CommandCreateSection, CommandDeleteSection, OwnedDelta, StructCommand};

        let create = |parent_id: u64, section_id: u64| {
            StructCommand::CreateSection(CommandCreateSection {
                parent_id,
                section_id,
                section_name: format!("n{section_id}"),
                section_type: 1,
            })
        };
        let delete = |section_id: u64| {
            StructCommand::DeleteSection(CommandDeleteSection {
                parent_id: 0,
                section_id,
            })
        };
        // A parent (1) with `n` children (ids from 100), and a sibling (2).
        let parent_with = |n: u64| {
            let mut rows = vec![(1, 0, "p"), (2, 0, "q")];
            rows.extend((100..100 + n).map(|id| (id, 1, "k")));
            state_of(&rows)
        };
        let cases: Vec<(&str, CompactState, Vec<StructCommand>)> = vec![
            (
                "every child deleted, one created: the key dropped and re-created short, \
                 its vector rebuilt at its restored length (step 2)",
                parent_with(5),
                (100..105).map(delete).chain([create(1, 500)]).collect(),
            ),
            (
                "the parent deleted with its 1,025 children: the key re-created whole at \
                 its length below its capacity (step 4)",
                parent_with(1_025),
                vec![delete(1)],
            ),
        ];
        for (label, pre, commands) in cases {
            let mut state = pre.clone();
            let mut indexes = PlanIndexes::build(&state);
            indexes.compact();
            let mut journal = PlanJournal::new(&state, &indexes);
            let mut touches = PlanTouches::default();
            let projected = std::cell::Cell::new(0u64);
            // Every price 0, so the plan asks exactly when its projected
            // rollback rebuilds grow; each answer covers them.
            let allowance = MutationAllowance {
                bytes: 0,
                row_bytes: 0,
                list_entry_bytes: 0,
                index_entry_bytes: 0,
                adoption_entry_bytes: 0,
                walk_entry_bytes: 0,
            };
            let mut budget = |needed: RewriteCount| {
                projected.set(needed.rollback_rebuild_bytes);
                Some(needed.rollback_rebuild_bytes)
            };
            let plan = apply_delta_bounded_indexed(
                &mut state,
                Some(OwnedDelta {
                    struct_commands: commands,
                    ..OwnedDelta::default()
                }),
                allowance,
                &mut budget,
                &mut ApplyDeltaStats::default(),
                &mut indexes,
                &mut touches,
                Some(&mut journal),
            );
            assert!(
                matches!(plan, Ok(BoundedPlan::Within { .. })),
                "{label}: the plan applies"
            );
            let before: HashMap<u64, *const NameEntry> = indexes
                .names
                .iter()
                .map(|(&parent, entries)| (parent, entries.as_ptr()))
                .collect();
            // The plan's projections split: each table projected once at
            // its rebuild, the rest the vectors it dropped with their keys.
            let projected_tables = journal.projected_rebuilds();
            let table_projections: u64 = RebuildTable::ALL
                .iter()
                .filter(|&&t| projected_tables.contains(t))
                .map(|&t| journal.rebuild_bytes(t))
                .sum();
            let vector_projections = projected.get() - table_projections;
            let rebuilt = journal
                .rollback(&mut state, &mut indexes)
                .expect("the rollback");
            assert_eq!(state, pre, "{label}: the contents are restored");
            assert!(
                rebuilt.is_subset(projected_tables),
                "{label}: every table rebuilt was projected"
            );
            let vectors: u64 = indexes
                .names
                .iter()
                .filter(|(parent, entries)| before.get(parent) != Some(&entries.as_ptr()))
                .map(|(_, entries)| entries_bytes(entries.capacity()))
                .sum();
            assert!(
                vectors > 0,
                "{label}: fixture: the rollback allocated a name-index vector"
            );
            assert!(
                vectors <= vector_projections,
                "{label}: the rollback allocated {vectors} B of name-index vectors, past the \
                 {vector_projections} B of dropped vectors its plan projected"
            );
        }
    }
}
