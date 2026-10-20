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

//! The pre-image journal of a structural plan made in place on the cache's
//! held entry ([`PlanJournal`]), and the pre-plan view it gives
//! ([`PreImageView`]).
//!
//! The structural cache plans each attempt on its held entry itself, not on a
//! copy (`storage::structural_cache::CheckedOut::plan_in_place`). Every write
//! the plan makes reaches the journal through the engine's writer
//! (`domain::delta_write`) and the plan's hooks (`domain::plan_index`), which
//! record each key's value **on its first touch**: section rows, children
//! lists, field and time-sample rows, and the header scalars. A reset hands
//! over the whole pre-reset tables and indexes in O(1) instead.
//!
//! - **Rollback** ([`PlanJournal::rollback`]) puts every recorded key back to
//!   its pre-plan value and fixes the plan indexes to match, key by key — or
//!   swaps a reset's tables and indexes back whole — then restores the header
//!   scalars and `depth_ok`. Each key is restored from its own record, so the
//!   result does not depend on the order of the writes or on where the plan
//!   stopped (an error, an overflow, or a commit that did not land).
//! - **The pre-plan view** ([`PlanJournal::view`]) answers a key from its
//!   record when the plan touched it and from the state otherwise, so the
//!   commit's pre-image digests, its size gate and its untouched-row counts
//!   read the state as it stood before the plan without a copy of it.
//!
//! The journal costs O(touched keys): one clone of each touched section row
//! and children list, one record per field name the plan released or
//! interned, and nothing for a key the plan never wrote.
//!
//! # Rollback without growth
//!
//! std's `HashMap::insert` reserves a slot before it looks its key up, so an
//! insert call on a table with no room left grows it — at least doubling its
//! buckets — even when it only overwrites; `entry()` reserves only for a
//! vacant key. A plan can leave a table with no room without anything
//! charging for it: its inserts can use up exactly the room left, and its
//! removals leave tombstones that do not give room back. So the rollback
//! never restores through an overwriting insert:
//!
//! - **Restore in place, insert only to re-create.** Per table, the keys the
//!   plan created are removed first; the keys it rewrote are restored in
//!   place (`get_mut` and assignment; for value rows
//!   `restore_hash_in_place`, keeping the row's name id and leaving the
//!   interner alone); only then are the keys it removed re-created — the one
//!   kind of insert call a rollback makes. The plan indexes remove a key of
//!   their maps only when the plan created it and re-create one only when
//!   the plan removed it: the external-hash counts put a pre-plan count back
//!   before taking the current one out; the name index takes the changed
//!   current entries out of their vectors keeping every key, puts the
//!   pre-plan entries back, then drops the keys left empty and re-creates
//!   the keys still missing (`PlanIndexes::restore_name_index`). Field
//!   names the plan's rows used are released before the removed rows' names
//!   are acquired again, and each name the plan released goes back at the id
//!   it held before the plan (the journal records it at the name's first
//!   release, `PlanJournal::name_released`); a pre-plan row the plan left
//!   keyed by such a name at a new id — the plan released the name and
//!   interned it again — is taken out and re-created at that id, its key
//!   being one the plan removed. So the interner's id table never grows,
//!   and its live and free ids are what they were before the plan
//!   (`types::NameInterner`).
//! - **Rebuild a short table, never grow it.** A table whose room is below
//!   its re-creations is first rebuilt at its pre-plan length
//!   (`types::rebuild_map_at`): the minimal, tombstone-free table for that
//!   length, never more buckets than it had, which the cache's compaction
//!   then leaves alone. It allocates one table of that size, freeing the
//!   old one after the swap; with enough room nothing is allocated. The
//!   tables are the state's four, the interner's name → id table and the
//!   plan indexes' two maps ([`RebuildTable`]); the journal records each
//!   one's pre-plan length.
//! - **Charge it before the write that can make it necessary.** While the
//!   plan runs, its hooks count each table's removals the rollback may have
//!   to re-create (`PlanJournal::note_removal`) and, the first time a
//!   removal or an insert call could leave that count above the table's
//!   room, project the table's rebuild, `table_bytes(pre-plan length)`,
//!   through the plan's meter, exactly
//!   (`domain::delta::RewriteCount::rollback_rebuild_bytes`). A name-index
//!   vector the plan drops with its parent key is projected as it goes, at
//!   its capacity: the most a rollback that restores that parent's entries
//!   allocates. After the plan, [`PlanJournal::rollback_rebuild_bytes`] says
//!   what a rollback could still rebuild — an upper bound, each rebuild
//!   priced exactly; the cached route's CR2 keeps it in its charge, so a
//!   rollback after it (a `STALE` commit, a dropped call) runs inside it.
//! - **Contents, not layout.** A rollback restores the state's contents
//!   and, once the put-back compacts it, its state weight byte for byte
//!   (the field-name interner's included: the same ids live and free), not
//!   its tables' layout: a table it did not rebuild keeps the tombstones of
//!   the keys it removed, so it can have less room than before the plan
//!   (the rollback notes each state table's capacity before its first
//!   removal, so the entry is still weighed at the buckets those tombstones
//!   hide, `PlanIndexes::note_capacity`; a table it rebuilt smaller is
//!   over-read at the old capacity until the compaction that follows);
//!   the interner's free list can be in another order and hold free slots
//!   the plan appended until compaction trims them; and a name-index vector
//!   can keep a grown capacity, which counts toward the index weight, not
//!   the state's. Each rollback restores from its own plan's records, so
//!   repeated rolled-back attempts on one entry leave no residue in its
//!   state that builds up. Its plan indexes can carry one: each name-index
//!   vector keeps the largest capacity it ever had (as it does through
//!   committed deletes), so rolled-back plans that grow different parents'
//!   vectors add up, and the index weight has no bound relative to the
//!   state. The cache evicts an entry grown heavier than it can hold when
//!   it is stored (`storage::structural_cache`, `Control::store`), and the
//!   next lookup cold-starts it with fresh indexes. A resumed plan of the
//!   same delta projects from the room and capacities it meets, so it can
//!   project a rebuild, or a larger dropped vector, that the first plan did
//!   not, and ask once more. Those projections are not predicted; each is
//!   still asked before its write.
//!
//! Pure logic: zero I/O, zero async (SDD §9).

use std::collections::HashMap;
use std::fmt;

use crate::domain::alloc_model::table_bytes;
use crate::domain::delta_write::{HeaderImage, NameChange, RebuildTable, RemovedKey, ResetTables};
use crate::domain::plan_index::{state_table_shape, table_shape, table_slot_bytes, PlanIndexes};
use crate::domain::types::{
    rebuild_map_at, CompactRows, CompactState, FieldRowImage, FieldRows, NameId, SampleRows,
    Section,
};

/// A value row's image: `None` for no row, `Some(None)` for an inline row,
/// `Some(Some(hash))` for an external one.
pub type RowImage = Option<Option<[u8; 32]>>;

/// The header scalars before the plan's first header write.
#[derive(Debug, Clone)]
struct SavedHeader {
    sequence_num: u64,
    first_version: u64,
    next_set_order: u64,
    source_format: String,
}

/// The lengths of the state's tables before the plan.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PrePlanLens {
    pub sections: u64,
    pub children: u64,
    pub fields: u64,
    pub time_samples: u64,
}

/// A reset's pre-reset tables and indexes, moved in whole.
struct ResetRecord {
    tables: ResetTables<CompactRows>,
    indexes: PlanIndexes,
}

/// A journal that cannot restore its plan's state: a reset arrived after a
/// key was already recorded. The engine always runs a reset first, so this
/// is a defect, never a delta's doing; the caller drops the entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TornJournal;

/// Which of the [`RebuildTable`]s a plan's rollback rebuilt, or whose rebuild
/// a plan projected (module docs, "Rollback without growth").
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RebuildSet([bool; RebuildTable::COUNT]);

impl RebuildSet {
    /// Whether `table` is in the set.
    pub fn contains(self, table: RebuildTable) -> bool {
        self.0[table.index()]
    }

    /// Whether no table is in the set.
    pub fn is_empty(self) -> bool {
        !self.0.iter().any(|&member| member)
    }

    /// Whether every table in `self` is also in `other`.
    pub fn is_subset(self, other: RebuildSet) -> bool {
        RebuildTable::ALL
            .iter()
            .all(|&table| !self.contains(table) || other.contains(table))
    }

    fn insert(&mut self, table: RebuildTable) {
        self.0[table.index()] = true;
    }
}

/// The entries `table` takes before an insert call would grow it: its
/// `capacity() − len()`, from the table registry
/// (`domain::plan_index::table_shape`).
fn room(table: RebuildTable, state: &CompactState, indexes: &PlanIndexes) -> u64 {
    let (len, capacity) = table_shape(table, state, indexes);
    capacity.saturating_sub(len) as u64
}

/// The journal's one rebuild trigger: a rollback that has `removed` keys to
/// re-create into a table with `room` entries of room cannot re-create them
/// all without an insert call growing it, so it rebuilds the table at its
/// pre-plan length instead. Read by [`PlanJournal::note_removal`],
/// [`PlanJournal::note_insert_call`] and
/// [`PlanJournal::rollback_rebuild_bytes`].
fn over_room(removed: u64, room: u64) -> bool {
    removed > room
}

/// The plan's count, per [`RebuildTable`], of the removals that its rollback
/// may have to re-create (`r_t`), and which tables' rebuilds it projected.
#[derive(Debug, Default)]
struct RebuildTracker {
    /// Removals of keys that existed before the plan (for the interner and
    /// the plan indexes' maps, every removal: over-counting only projects
    /// earlier).
    removed: [u64; RebuildTable::COUNT],
    projected: RebuildSet,
    /// The name-index vectors the plan dropped with their parent keys, in
    /// bytes at their capacities: what restoring those parents' vectors can
    /// allocate, an upper bound.
    dropped_vector_bytes: u64,
}

/// The first-touch pre-images of one plan made in place (module docs).
pub struct PlanJournal {
    lens: PrePlanLens,
    /// Every [`RebuildTable`]'s length before the plan, in
    /// [`RebuildTable::index`] order: the size a rollback rebuilds it at.
    rebuild_lens: [u64; RebuildTable::COUNT],
    rebuild: RebuildTracker,
    depth_ok_at_start: bool,
    header: Option<SavedHeader>,
    sections: HashMap<u64, Option<Section>>,
    lists: HashMap<u64, Option<Vec<u64>>>,
    fields: HashMap<u64, HashMap<Box<str>, RowImage>>,
    /// Each field name the plan released or interned, by its first such
    /// change: `Some(id)` for a name it released first — a pre-plan name,
    /// and the id it held before the plan — and `None` for one it interned
    /// first, which no row used before the plan.
    names: HashMap<Box<str>, Option<NameId>>,
    samples: HashMap<(u64, u64), RowImage>,
    reset: Option<Box<ResetRecord>>,
    torn: bool,
}

impl fmt::Debug for PlanJournal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PlanJournal")
            .field("lens", &self.lens)
            .field("header_touched", &self.header.is_some())
            .field("sections", &self.sections.len())
            .field("lists", &self.lists.len())
            .field("field_sections", &self.fields.len())
            .field("names", &self.names.len())
            .field("samples", &self.samples.len())
            .field("reset", &self.reset.is_some())
            .finish()
    }
}

/// The image of field row `(section_id, name)` in `fields`.
fn field_image(fields: &impl FieldRows, section_id: u64, name: &str) -> RowImage {
    fields
        .contains(section_id, name)
        .then(|| fields.ext_hash(section_id, name))
}

/// The image of time-sample row `(section_id, time_bits)` in `samples`.
fn sample_image(samples: &impl SampleRows, section_id: u64, time_bits: u64) -> RowImage {
    samples
        .contains(section_id, time_bits)
        .then(|| samples.ext_hash(section_id, time_bits))
}

impl PlanJournal {
    /// An empty journal for a plan about to run on `state`, whose indexes are
    /// `indexes`.
    pub fn new(state: &CompactState, indexes: &PlanIndexes) -> Self {
        let lens = PrePlanLens {
            sections: state.sections.len() as u64,
            children: state.children.len() as u64,
            fields: FieldRows::len(&state.fields) as u64,
            time_samples: SampleRows::len(&state.time_samples) as u64,
        };
        Self {
            lens,
            rebuild_lens: RebuildTable::ALL
                .map(|table| table_shape(table, state, indexes).0 as u64),
            rebuild: RebuildTracker::default(),
            depth_ok_at_start: indexes.depth_ok(),
            header: None,
            sections: HashMap::new(),
            lists: HashMap::new(),
            fields: HashMap::new(),
            names: HashMap::new(),
            samples: HashMap::new(),
            reset: None,
            torn: false,
        }
    }

    /// The state's table lengths before the plan.
    pub fn pre_plan_lens(&self) -> PrePlanLens {
        self.lens
    }

    /// What a rollback's rebuild of `table` allocates: the table at its
    /// pre-plan length.
    pub fn rebuild_bytes(&self, table: RebuildTable) -> u64 {
        let len = self.rebuild_lens[table.index()];
        table_bytes(
            usize::try_from(len).unwrap_or(usize::MAX),
            table_slot_bytes(table),
        )
    }

    /// The tables whose rollback rebuild the plan projected.
    pub fn projected_rebuilds(&self) -> RebuildSet {
        self.rebuild.projected
    }

    /// Whether the removed `key` held a row or list before the plan, by its
    /// first-touch record. An interned name is always taken to (over-counting
    /// only projects earlier).
    pub(crate) fn key_pre_existed(&self, key: RemovedKey<'_>) -> bool {
        match key {
            RemovedKey::Section(id) => matches!(self.sections.get(&id), Some(Some(_))),
            RemovedKey::List(parent) => matches!(self.lists.get(&parent), Some(Some(_))),
            RemovedKey::Field(section_id, name) => matches!(
                self.fields.get(&section_id).and_then(|rows| rows.get(name)),
                Some(Some(_))
            ),
            RemovedKey::Sample(section_id, time_bits) => {
                matches!(self.samples.get(&(section_id, time_bits)), Some(Some(_)))
            }
            RemovedKey::FieldName(..) => true,
        }
    }

    /// A key is about to leave `table`, which has `room` entries of room:
    /// count it toward the rollback's re-creations when it existed before
    /// the plan, and answer the bytes to project when that count could now
    /// pass the table's room — the first time only. Nothing after a reset,
    /// whose rollback swaps the tables back whole.
    pub(crate) fn note_removal(
        &mut self,
        table: RebuildTable,
        room: u64,
        pre_existing: bool,
    ) -> Option<u64> {
        if self.reset.is_some() || !pre_existing {
            return None;
        }
        let removed = &mut self.rebuild.removed[table.index()];
        *removed = removed.saturating_add(1);
        over_room(*removed, room)
            .then(|| self.project(table))
            .flatten()
    }

    /// An insert call is about to land on `table`, which has `room` entries
    /// of room: answer the bytes to project when the re-creations counted so
    /// far could pass the room left after it — the first time only.
    pub(crate) fn note_insert_call(&mut self, table: RebuildTable, room: u64) -> Option<u64> {
        if self.reset.is_some() {
            return None;
        }
        let removed = self.rebuild.removed[table.index()];
        (removed > 0 && over_room(removed, room.saturating_sub(1)))
            .then(|| self.project(table))
            .flatten()
    }

    /// The plan is about to drop a name-index parent key together with its
    /// vector of `bytes`: a rollback may rebuild that parent's vector, at
    /// most its pre-plan size, which this vector's covers when the key
    /// existed before the plan
    /// (`domain::plan_index::PlanIndexes::restore_name_index`). Counted
    /// toward [`Self::rollback_rebuild_bytes`] and answered, to project at
    /// once; nothing after a reset.
    pub(crate) fn note_vector_drop(&mut self, bytes: u64) -> Option<u64> {
        if self.reset.is_some() || bytes == 0 {
            return None;
        }
        self.rebuild.dropped_vector_bytes = self.rebuild.dropped_vector_bytes.saturating_add(bytes);
        Some(bytes)
    }

    /// Mark `table`'s rebuild projected, answering its bytes the first time.
    fn project(&mut self, table: RebuildTable) -> Option<u64> {
        if self.rebuild.projected.contains(table) {
            return None;
        }
        self.rebuild.projected.insert(table);
        Some(self.rebuild_bytes(table))
    }

    /// What a rollback of the plan as it stands could rebuild, in bytes — an
    /// upper bound, each rebuild priced exactly: the sum of
    /// [`Self::rebuild_bytes`] over the tables whose room is now below the
    /// plan's count of removals for them, plus the name-index vectors the
    /// plan dropped with their keys (`Self::note_vector_drop`). O(tables).
    ///
    /// The rollback can do less: the count over-counts the interner's and
    /// the index maps' removals, counts a pre-plan key the plan removed and
    /// then created again, and the rollback measures a table's room only
    /// after removing the plan's own keys. So a delta that deletes one row
    /// and creates one, in a table at its load limit, is held a rebuild its
    /// rollback never makes. CR2 holds this through the commit, so a
    /// rollback after it (`STALE`, `FENCED`, a transport or in-doubt error, a
    /// dropped call) runs inside the charge. 0 after a reset.
    pub fn rollback_rebuild_bytes(&self, state: &CompactState, indexes: &PlanIndexes) -> u64 {
        if self.reset.is_some() {
            return 0;
        }
        RebuildTable::ALL
            .iter()
            .filter(|&&table| {
                over_room(
                    self.rebuild.removed[table.index()],
                    room(table, state, indexes),
                )
            })
            .fold(self.rebuild.dropped_vector_bytes, |sum, &table| {
                sum.saturating_add(self.rebuild_bytes(table))
            })
    }

    /// Whether the plan touched no key and made no header write.
    pub fn is_empty(&self) -> bool {
        self.header.is_none()
            && self.sections.is_empty()
            && self.lists.is_empty()
            && self.fields.is_empty()
            && self.samples.is_empty()
            && self.reset.is_none()
    }

    fn has_key_records(&self) -> bool {
        !(self.sections.is_empty()
            && self.lists.is_empty()
            && self.fields.is_empty()
            && self.samples.is_empty())
    }

    pub(crate) fn section_touched(&mut self, id: u64, old: Option<&Section>) {
        if self.reset.is_none() {
            self.sections.entry(id).or_insert_with(|| old.cloned());
        }
    }

    pub(crate) fn list_touched(&mut self, parent: u64, old: Option<&Vec<u64>>) {
        if self.reset.is_none() {
            self.lists.entry(parent).or_insert_with(|| old.cloned());
        }
    }

    pub(crate) fn field_touched(&mut self, section_id: u64, name: &str, old: RowImage) {
        if self.reset.is_some() {
            return;
        }
        let rows = self.fields.entry(section_id).or_default();
        if !rows.contains_key(name) {
            rows.insert(Box::from(name), old);
        }
    }

    /// The plan released field name `name`, which held `id`: on the name's
    /// first change this is its pre-plan id.
    pub(crate) fn name_released(&mut self, name: &str, id: NameId) {
        if self.reset.is_none() && !self.names.contains_key(name) {
            self.names.insert(Box::from(name), Some(id));
        }
    }

    /// The plan interned field name `name`: on the name's first change, one
    /// no row used before the plan.
    pub(crate) fn name_interned(&mut self, name: &str) {
        if self.reset.is_none() && !self.names.contains_key(name) {
            self.names.insert(Box::from(name), None);
        }
    }

    pub(crate) fn sample_touched(&mut self, section_id: u64, time_bits: u64, old: RowImage) {
        if self.reset.is_none() {
            self.samples.entry((section_id, time_bits)).or_insert(old);
        }
    }

    pub(crate) fn header_touched(&mut self, header: HeaderImage<'_>) {
        if self.header.is_none() {
            self.header = Some(SavedHeader {
                sequence_num: header.sequence_num,
                first_version: header.first_version,
                next_set_order: header.next_set_order,
                source_format: header.source_format.clone(),
            });
        }
    }

    pub(crate) fn reset_taken(&mut self, tables: ResetTables<CompactRows>, indexes: PlanIndexes) {
        // A reset is a delta's first step, so nothing is recorded before it.
        // If something were, the swap below would put back the state at the
        // reset rather than before the plan.
        if self.has_key_records() || self.reset.is_some() {
            self.torn = true;
        }
        self.reset = Some(Box::new(ResetRecord { tables, indexes }));
    }

    /// Put `state` and `indexes` back exactly as they stood before the plan
    /// (module docs): their contents, not their layout. Consumes the
    /// journal, and answers which tables it rebuilt.
    ///
    /// # Errors
    ///
    /// [`TornJournal`] when the journal cannot say what that was; `state`
    /// and `indexes` are then partly restored, and the caller must drop them.
    pub fn rollback(
        self,
        state: &mut CompactState,
        indexes: &mut PlanIndexes,
    ) -> Result<RebuildSet, TornJournal> {
        if self.torn {
            return Err(TornJournal);
        }
        let mut rebuilt = RebuildSet::default();
        let Self {
            lens,
            rebuild_lens,
            depth_ok_at_start,
            header,
            sections,
            lists,
            fields,
            names: name_ids,
            samples,
            reset,
            ..
        } = self;
        let pre_len = |table: RebuildTable| {
            usize::try_from(rebuild_lens[table.index()]).unwrap_or(usize::MAX)
        };
        match reset {
            Some(record) => {
                let ResetRecord {
                    tables,
                    indexes: before,
                } = *record;
                let (Some(field_rows), Some(sample_rows)) = (tables.fields, tables.time_samples)
                else {
                    return Err(TornJournal);
                };
                state.sections = tables.sections;
                state.children = tables.children;
                state.fields = field_rows;
                state.time_samples = sample_rows;
                *indexes = before;
            }
            None => {
                // The rollback removes keys from the state's tables below:
                // each table's capacity is noted once, before the first of
                // them can leave a tombstone that hides its buckets
                // (`PlanIndexes::note_capacity`). The indexes' own maps note
                // theirs at their removals.
                for table in RebuildTable::ALL {
                    if let Some((_, capacity)) = state_table_shape(table, state) {
                        indexes.note_capacity(table, capacity);
                    }
                }
                // The plan indexes first, while the state still holds the
                // plan's rows: the name index and the section-name sum from
                // each touched row's current and pre-plan value, the
                // external-hash counts from each touched value row's.
                if indexes.restore_name_index(
                    sections
                        .iter()
                        .map(|(id, pre)| (state.sections.get(id), pre.as_ref())),
                    pre_len(RebuildTable::NameIndex),
                ) {
                    rebuilt.insert(RebuildTable::NameIndex);
                }
                let field_images: Vec<(RowImage, RowImage)> = fields
                    .iter()
                    .flat_map(|(&section_id, rows)| {
                        let current = &state.fields;
                        rows.iter()
                            .map(move |(name, &pre)| (field_image(current, section_id, name), pre))
                    })
                    .collect();
                let sample_images: Vec<(RowImage, RowImage)> = samples
                    .iter()
                    .map(|(&(section_id, time_bits), &pre)| {
                        (
                            sample_image(&state.time_samples, section_id, time_bits),
                            pre,
                        )
                    })
                    .collect();
                if indexes.restore_ext_counts(
                    &field_images,
                    &sample_images,
                    pre_len(RebuildTable::ExtCounts),
                ) {
                    rebuilt.insert(RebuildTable::ExtCounts);
                }
                // The test-support failpoint's panic mode fires here.
                crate::domain::delta::rollback_rebuild_failpoint();

                // Section rows: the plan's own rows removed, the rows it
                // rewrote restored in place, then the rows it removed
                // re-created — into a table rebuilt at its pre-plan length
                // first if it lacks the room.
                let mut recreate: Vec<Section> = Vec::new();
                for (id, pre) in sections {
                    match (state.sections.get_mut(&id), pre) {
                        (Some(slot), Some(row)) => *slot = row,
                        (Some(_), None) => {
                            state.sections.remove(&id);
                        }
                        (None, Some(row)) => recreate.push(row),
                        (None, None) => {}
                    }
                }
                if state
                    .sections
                    .capacity()
                    .saturating_sub(state.sections.len())
                    < recreate.len()
                {
                    rebuild_map_at(&mut state.sections, pre_len(RebuildTable::Sections));
                    rebuilt.insert(RebuildTable::Sections);
                }
                for row in recreate {
                    state.sections.insert(row.section_id, row);
                }

                // Children lists, the same way, their allocation sum kept.
                let mut recreate: Vec<(u64, Vec<u64>)> = Vec::new();
                for (parent, pre) in lists {
                    let current = state.children.get(&parent).map(Vec::capacity);
                    indexes.list_replaced(current, pre.as_ref().map(Vec::capacity));
                    match (state.children.get_mut(&parent), pre) {
                        (Some(slot), Some(list)) => *slot = list,
                        (Some(_), None) => {
                            state.children.remove(&parent);
                        }
                        (None, Some(list)) => recreate.push((parent, list)),
                        (None, None) => {}
                    }
                }
                if state
                    .children
                    .capacity()
                    .saturating_sub(state.children.len())
                    < recreate.len()
                {
                    rebuild_map_at(&mut state.children, pre_len(RebuildTable::Children));
                    rebuilt.insert(RebuildTable::Children);
                }
                for (parent, list) in recreate {
                    state.children.insert(parent, list);
                }

                // Field rows: the plan's own rows removed first (releasing
                // their names), the rows it overwrote restored in place (the
                // name and the interner untouched), then the rows it removed
                // re-created — each table rebuilt first if it lacks the room.
                // A pre-plan name the plan released goes back at its
                // pre-plan id, so the interner's live and free ids are what
                // they were: a pre-plan row the plan left keyed by such a
                // name at another id (released, then interned again) is
                // taken out with the plan's rows and re-created too. A key
                // the plan removed is the only one re-created.
                let mut recreate: Vec<FieldRowImage> = Vec::new();
                for (section_id, rows) in fields {
                    for (name, pre) in rows {
                        let moved_name =
                            name_ids
                                .get(&name)
                                .copied()
                                .flatten()
                                .is_some_and(|pre_id| {
                                    state.fields.names().id_of(&name) != Some(pre_id)
                                });
                        match pre {
                            Some(hash) if !moved_name => {
                                if state
                                    .fields
                                    .restore_hash_in_place(section_id, &name, hash)
                                    .is_none()
                                {
                                    recreate.push((section_id, name, hash));
                                }
                            }
                            _ => {
                                if let Some((_, released)) = state.fields.remove(section_id, &name)
                                {
                                    if released {
                                        indexes.interned_name_changed(&name, NameChange::Released);
                                    }
                                }
                                if let Some(hash) = pre {
                                    recreate.push((section_id, name, hash));
                                }
                            }
                        }
                    }
                }
                let new_names = {
                    let mut names: Vec<&str> = recreate
                        .iter()
                        .map(|(_, name, _)| &**name)
                        .filter(|name| state.fields.names().id_of(name).is_none())
                        .collect();
                    names.sort_unstable();
                    names.dedup();
                    names.len()
                };
                if state.fields.name_room() < new_names {
                    state
                        .fields
                        .rebuild_names_at(pre_len(RebuildTable::NameIds));
                    rebuilt.insert(RebuildTable::NameIds);
                }
                if state.fields.row_room() < recreate.len() {
                    state
                        .fields
                        .rebuild_rows_at(pre_len(RebuildTable::FieldRows));
                    rebuilt.insert(RebuildTable::FieldRows);
                }
                // The rows of the names the plan released, at their names'
                // pre-plan ids, first; then the rest, whose names were never
                // released and keep their ids.
                let mut at_pre_ids = Vec::new();
                let mut rest = Vec::new();
                for (section_id, name, hash) in recreate {
                    match name_ids.get(&name).copied().flatten() {
                        Some(pre_id) => at_pre_ids.push((section_id, name, pre_id, hash)),
                        None => rest.push((section_id, name, hash)),
                    }
                }
                let unplaced = state.fields.recreate_at_pre_ids(at_pre_ids, |name| {
                    indexes.interned_name_changed(name, NameChange::Interned);
                });
                debug_assert!(
                    unplaced.is_empty(),
                    "every released name's pre-plan id is free once the plan's rows are out"
                );
                for (section_id, name, hash) in rest.into_iter().chain(unplaced) {
                    let write = state.fields.insert(section_id, &name, hash);
                    if write.name_interned {
                        indexes.interned_name_changed(&name, NameChange::Interned);
                    }
                }

                // Time-sample rows, the same way.
                let mut recreate: Vec<(u64, u64, Option<[u8; 32]>)> = Vec::new();
                for ((section_id, time_bits), pre) in samples {
                    match pre {
                        None => {
                            state.time_samples.remove(section_id, time_bits);
                        }
                        Some(hash) => {
                            if state
                                .time_samples
                                .restore_hash_in_place(section_id, time_bits, hash)
                                .is_none()
                            {
                                recreate.push((section_id, time_bits, hash));
                            }
                        }
                    }
                }
                if state.time_samples.row_room() < recreate.len() {
                    state
                        .time_samples
                        .rebuild_rows_at(pre_len(RebuildTable::SampleRows));
                    rebuilt.insert(RebuildTable::SampleRows);
                }
                for (section_id, time_bits, hash) in recreate {
                    state.time_samples.insert(section_id, time_bits, hash);
                }
                debug_assert_eq!(state.sections.len() as u64, lens.sections);
                debug_assert_eq!(state.children.len() as u64, lens.children);
            }
        }
        if let Some(saved) = header {
            state.sequence_num = saved.sequence_num;
            state.first_version = saved.first_version;
            state.next_set_order = saved.next_set_order;
            state.source_format = saved.source_format;
        }
        indexes.set_depth_ok(depth_ok_at_start);
        Ok(rebuilt)
    }

    /// The state as it stood before the plan, read through the journal:
    /// `state` and `indexes` are the planned (post-plan) entry's.
    pub fn view<'a>(
        &'a self,
        state: &'a CompactState,
        indexes: &'a PlanIndexes,
    ) -> PreImageView<'a> {
        PreImageView {
            state,
            indexes,
            journal: self,
        }
    }
}

/// The pre-plan state of an entry planned in place ([`PlanJournal::view`]):
/// each key from the journal when the plan touched it, from the state
/// otherwise; after a reset, every table from the pre-reset tables.
#[derive(Clone, Copy)]
pub struct PreImageView<'a> {
    state: &'a CompactState,
    indexes: &'a PlanIndexes,
    journal: &'a PlanJournal,
}

impl fmt::Debug for PreImageView<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreImageView")
            .field("journal", self.journal)
            .finish_non_exhaustive()
    }
}

impl<'a> PreImageView<'a> {
    fn reset(&self) -> Option<&'a ResetTables<CompactRows>> {
        self.journal.reset.as_deref().map(|record| &record.tables)
    }

    /// The section row `id` held.
    pub fn section(&self, id: u64) -> Option<&'a Section> {
        if let Some(tables) = self.reset() {
            return tables.sections.get(&id);
        }
        match self.journal.sections.get(&id) {
            Some(pre) => pre.as_ref(),
            None => self.state.sections.get(&id),
        }
    }

    /// The children list `parent` held.
    pub fn list(&self, parent: u64) -> Option<&'a [u64]> {
        if let Some(tables) = self.reset() {
            return tables.children.get(&parent).map(Vec::as_slice);
        }
        match self.journal.lists.get(&parent) {
            Some(pre) => pre.as_deref(),
            None => self.state.children.get(&parent).map(Vec::as_slice),
        }
    }

    /// The image of field row `(section_id, name)`.
    pub fn field(&self, section_id: u64, name: &str) -> RowImage {
        if let Some(tables) = self.reset() {
            return tables
                .fields
                .as_ref()
                .and_then(|fields| field_image(fields, section_id, name));
        }
        match self
            .journal
            .fields
            .get(&section_id)
            .and_then(|rows| rows.get(name))
        {
            Some(pre) => *pre,
            None => field_image(&self.state.fields, section_id, name),
        }
    }

    /// The image of time-sample row `(section_id, time_bits)`.
    pub fn sample(&self, section_id: u64, time_bits: u64) -> RowImage {
        if let Some(tables) = self.reset() {
            return tables
                .time_samples
                .as_ref()
                .and_then(|samples| sample_image(samples, section_id, time_bits));
        }
        match self.journal.samples.get(&(section_id, time_bits)) {
            Some(pre) => *pre,
            None => sample_image(&self.state.time_samples, section_id, time_bits),
        }
    }

    /// The table lengths before the plan.
    pub fn lens(&self) -> PrePlanLens {
        self.journal.lens
    }

    /// The field and time-sample rows holding `hash` **after** the plan: the
    /// planned state's count, which the pre-plan untouched-row count is read
    /// from (untouched rows are the same before and after).
    pub fn post_ext_count(&self, hash: &[u8; 32]) -> u64 {
        self.indexes.ext_count(hash)
    }
}

/// The rollback's restore-in-place rule and its rebuild bound, on states the
/// tests build to sit at their tables' load limits. The differential
/// properties (every step, every refused ask, plan → rollback → re-plan) are
/// in `tests/proptests.rs`, `in_place_plan_equivalence`.
#[cfg(test)]
mod tests {
    use std::mem::size_of;

    use oodf_ffi::{
        CommandCreateSection, CommandDeleteSection, CommandMoveSection, CommandReorderChildren,
        OwnedDelta, SetField, StructCommand, TimeSample as OodfTimeSample,
    };

    use super::*;
    use crate::domain::delta::{
        apply_delta_bounded_indexed, ApplyDeltaStats, BoundedPlan, MutationAllowance, RewriteCount,
    };
    use crate::domain::plan_index::PlanTouches;
    use crate::domain::types::{compact, Field, FieldValue, ObjectState};

    fn section(id: u64, parent_id: u64) -> Section {
        Section {
            section_id: id,
            parent_id,
            name: format!("s{id}"),
            section_type: 6,
            section_order: 0,
            first_version: 1,
            last_version: 1,
        }
    }

    /// `rows` as `(id, parent)` sections listed under their parents in id
    /// order, `fields` as `(section, name)` inline rows and `samples` as
    /// `(section, time)` inline rows, compacted so every table is minimal for
    /// its length.
    fn state_of(
        rows: &[(u64, u64)],
        fields: &[(u64, &str)],
        samples: &[(u64, f64)],
    ) -> CompactState {
        let mut full = ObjectState {
            object_id: 1,
            sequence_num: 5,
            first_version: 1,
            source_format: "usd".to_owned(),
            ..ObjectState::default()
        };
        for &(id, parent) in rows {
            full.sections.insert(id, section(id, parent));
            full.children.entry(parent).or_default().push(id);
        }
        for &(section_id, name) in fields {
            full.fields.insert(
                (section_id, name.to_owned()),
                Field {
                    section_id,
                    key_name: name.to_owned(),
                    value: FieldValue::Inline(vec![1]),
                    set_order: 1,
                    authoring_set_order: 1,
                    version: 1,
                },
            );
        }
        let mut state = compact(&full);
        for &(section_id, time) in samples {
            state.time_samples.insert(section_id, time.to_bits(), None);
        }
        state.sections.shrink_to_fit();
        state.children.shrink_to_fit();
        FieldRows::shrink_to_fit(&mut state.fields);
        SampleRows::shrink_to_fit(&mut state.time_samples);
        state
    }

    fn indexes_of(state: &CompactState) -> PlanIndexes {
        let mut indexes = PlanIndexes::build(state);
        indexes.compact();
        indexes
    }

    /// Every [`RebuildTable`]'s capacity, in [`RebuildTable::ALL`] order.
    fn capacities(state: &CompactState, indexes: &PlanIndexes) -> [usize; 7] {
        [
            state.sections.capacity(),
            state.children.capacity(),
            state.fields.capacity(),
            state.time_samples.capacity(),
            state.fields.names().ids_capacity(),
            indexes.parent_keys() + indexes.parent_key_room(),
            indexes.hash_keys() + indexes.hash_key_room(),
        ]
    }

    /// `delta` planned in place on `state` under a journal, unbounded but for
    /// its rollback rebuilds, whose projections it records: returns the
    /// plan's result, its journal and the rollback rebuild bytes it projected.
    fn plan(
        state: &mut CompactState,
        indexes: &mut PlanIndexes,
        delta: OwnedDelta,
    ) -> (
        Result<BoundedPlan, crate::domain::delta::DeltaError>,
        PlanJournal,
        u64,
    ) {
        let mut journal = PlanJournal::new(state, indexes);
        let mut touches = PlanTouches::default();
        let projected = std::cell::Cell::new(0u64);
        // Every count free but the rollback rebuilds (bytes, 1:1), so the
        // plan asks exactly when its projected rebuilds grow.
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
        let result = apply_delta_bounded_indexed(
            state,
            Some(delta),
            allowance,
            &mut budget,
            &mut ApplyDeltaStats::default(),
            indexes,
            &mut touches,
            Some(&mut journal),
        );
        (result, journal, projected.get())
    }

    /// R-T2: an overwrite-only plan — a rename, a move between two listed
    /// parents, a reorder, and field and time-sample overwrites — on tables at
    /// `capacity() == len()` rolls back with no table's capacity changed by
    /// the rollback (it restores rewritten keys in place: an insert call on a
    /// full table would grow it), and the state is the pre-plan state. The
    /// rename and the move write rows and lists in place, so the sections
    /// table is at its pre-plan capacity throughout.
    #[test]
    fn rollback_of_an_overwrite_only_plan_on_full_tables_changes_no_capacity() {
        let pre = state_of(
            &[(1, 0), (2, 0), (3, 0), (4, 1), (5, 1), (6, 2), (7, 2)],
            &[
                (1, "f0"),
                (2, "f0"),
                (3, "f1"),
                (4, "f1"),
                (5, "f2"),
                (6, "f2"),
                (7, "f0"),
            ],
            &[(1, 0.0), (2, 1.0), (3, 2.0)],
        );
        let pre_indexes = indexes_of(&pre);
        let full = capacities(&pre, &pre_indexes);
        let lens = [
            pre.sections.len(),
            pre.children.len(),
            FieldRows::len(&pre.fields),
            SampleRows::len(&pre.time_samples),
            pre.fields.names().len(),
            pre_indexes.parent_keys(),
        ];
        assert_eq!(
            full[..6],
            lens,
            "fixture: every table but the empty external-hash counts is at capacity() == len()"
        );
        let delta = OwnedDelta {
            struct_commands: vec![
                StructCommand::MoveSection(CommandMoveSection {
                    old_parent_id: 1,
                    new_parent_id: 1,
                    section_id: 4,
                    new_name: "renamed".to_owned(),
                }),
                StructCommand::MoveSection(CommandMoveSection {
                    old_parent_id: 1,
                    new_parent_id: 2,
                    section_id: 5,
                    new_name: "s5".to_owned(),
                }),
                StructCommand::ReorderChildren(CommandReorderChildren {
                    section_id: 0,
                    children_list_id: 0,
                    children_list: vec![3, 2, 1],
                }),
            ],
            set_fields: vec![SetField {
                section_id: 1,
                key_name: "f0".to_owned(),
                value_or_ext_hash: vec![2],
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 1,
            }],
            time_samples: vec![OodfTimeSample {
                section_id: 1,
                time: 0.0,
                value_or_ext_hash: vec![3],
                ext_value_size: 0,
                ext_value_index: 0,
            }],
            ..OwnedDelta::default()
        };
        let mut state = pre.clone();
        let mut indexes = indexes_of(&state);
        let (result, journal, _) = plan(&mut state, &mut indexes, delta);
        assert!(
            matches!(result, Ok(BoundedPlan::Within { .. })),
            "the plan applies: {result:?}"
        );
        assert_eq!(
            state.sections.capacity(),
            full[0],
            "fixture: the rename and the move write the sections table in place"
        );
        let planned = capacities(&state, &indexes);
        let rebuilt = journal
            .rollback(&mut state, &mut indexes)
            .expect("the rollback");
        assert!(
            rebuilt.is_empty(),
            "nothing to re-create, so nothing rebuilt"
        );
        assert_eq!(state, pre, "the rollback restores the pre-plan state");
        assert_eq!(
            capacities(&state, &indexes),
            planned,
            "the rollback changes no table's capacity"
        );
        assert_eq!(
            state.sections.capacity(),
            full[0],
            "the sections table at its pre-plan capacity"
        );
        assert!(indexes.same_content_as(&PlanIndexes::build(&state)));
    }

    /// `rollback_rebuild_bytes` after a plan with room for every
    /// re-creation is the dropped name-index vectors alone: 0 when no key is
    /// dropped, and a dropped key's vector at its capacity when one is.
    #[test]
    fn rollback_rebuild_bytes_counts_a_dropped_vector_and_nothing_with_room() {
        // Parent 1 with two children, parents 2 and 3 with one each; room in
        // every table for the one removal each case makes (the name index's
        // map holds four parent keys at capacity 7).
        let mut pre = state_of(
            &[(1, 0), (2, 0), (3, 0), (10, 1), (11, 1), (20, 2), (30, 3)],
            &[],
            &[],
        );
        pre.sections.reserve(16);
        pre.children.reserve(16);
        let delete = |section_id: u64| OwnedDelta {
            struct_commands: vec![StructCommand::DeleteSection(CommandDeleteSection {
                parent_id: 0,
                section_id,
            })],
            ..OwnedDelta::default()
        };

        let mut state = pre.clone();
        let mut indexes = PlanIndexes::build(&state);
        let (result, journal, projected) = plan(&mut state, &mut indexes, delete(10));
        assert!(matches!(result, Ok(BoundedPlan::Within { .. })));
        assert_eq!(
            journal.rollback_rebuild_bytes(&state, &indexes),
            0,
            "a removal with room and no dropped key: nothing to rebuild"
        );
        assert_eq!(projected, 0, "and nothing projected");

        let mut state = pre.clone();
        let mut indexes = PlanIndexes::build(&state);
        assert!(
            indexes.parent_key_room() >= 1,
            "fixture: the name index's map has room to re-create a key"
        );
        // Parent 2's one-entry vector of `(name hash, id)`, as the build
        // leaves it: capacity 1.
        let dropped = crate::domain::alloc_model::allocation_bytes(size_of::<(u32, u64)>() as u64);
        let (result, journal, projected) = plan(&mut state, &mut indexes, delete(20));
        assert!(matches!(result, Ok(BoundedPlan::Within { .. })));
        assert_eq!(
            journal.rollback_rebuild_bytes(&state, &indexes),
            dropped,
            "deleting parent 2's only child drops its key: its vector, at its capacity"
        );
        assert_eq!(projected, dropped, "projected as the key was dropped");
    }

    /// `rollback_rebuild_bytes` after a plan that removes more keys of a
    /// table than it has room for prices that table's rebuild at its
    /// pre-plan length, `rebuild_bytes`, and bounds what the rollback then
    /// allocates; and a reset's is 0 (its rollback swaps the tables back).
    /// This is the pricing half of the sweep; its bucket and branch half —
    /// the rebuilt table at no more buckets than before the plan, the
    /// indexes and the weight restored, and both branches (rebuilt and not)
    /// reached — is `tests/proptests.rs`'s
    /// `rollback_rebuilds_a_short_table_at_its_pre_plan_length_and_both_branches_are_hit`,
    /// over the same deletes and creates.
    ///
    /// Whether a rollback rebuilds turns on how hashbrown marks the erases:
    /// a slot becomes a tombstone, its room not coming back, only when the
    /// run of non-EMPTY control bytes through it is at least the control
    /// group's width — 8 on aarch64 (NEON), 16 on x86_64 (SSE2) — so a
    /// table of at most one group's buckets never holds one. The full
    /// tables here are of 32, 64 and 128 buckets (28, 56 and 112 rows),
    /// each over `SHORT_TABLE_LAYOUTS` fresh layouts (`state_of` builds new
    /// maps, so a new `RandomState`), since a layout can lack any run of 16:
    /// with its `b`/8 EMPTY bytes placed at random, a table at no room has
    /// about `(b/8)(1 - 16/(7b/8))^(b/8 - 1)` such runs (Poisson), so lacks
    /// one with probability about 0.5 at 64 buckets and 0.2 at 128. At
    /// x86_64 every 64- and 128-bucket layout lacks one with probability
    /// about 0.5^8 × 0.2^8, below 1e-7; hashbrown's probing clusters the
    /// full slots, so less in fact.
    #[test]
    fn rollback_rebuild_bytes_prices_a_short_table_at_its_pre_plan_length_and_resets_at_zero() {
        /// The fresh hash layouts each table size is swept over.
        const SHORT_TABLE_LAYOUTS: usize = 8;
        // Tables of at least two control groups at aarch64's width (a
        // smaller one marks every erase empty, so its room always comes
        // back), each full.
        let full_of = |n: u64| {
            let rows: Vec<(u64, u64)> = (1..=n).map(|id| (id, 0)).collect();
            let state = state_of(&rows, &[], &[]);
            assert_eq!(
                state.sections.capacity(),
                state.sections.len(),
                "fixture: the sections table of {n} is full"
            );
            state
        };
        let mut reached_short = false;
        for n in [28u64, 56, 112] {
            for _layout in 0..SHORT_TABLE_LAYOUTS {
                let pre = full_of(n);
                for victim in 1..=n {
                    for fresh in [1_000 + victim, 5_000 + 3 * victim, 77_777 * victim] {
                        let context = format!("{n} sections, delete {victim}, create {fresh}");
                        let mut state = pre.clone();
                        let mut indexes = PlanIndexes::build(&state);
                        let delta = OwnedDelta {
                            struct_commands: vec![
                                StructCommand::DeleteSection(CommandDeleteSection {
                                    parent_id: 0,
                                    section_id: victim,
                                }),
                                StructCommand::CreateSection(CommandCreateSection {
                                    parent_id: 0,
                                    section_id: fresh,
                                    section_name: format!("n{fresh}"),
                                    section_type: 6,
                                }),
                            ],
                            ..OwnedDelta::default()
                        };
                        let (result, journal, projected) = plan(&mut state, &mut indexes, delta);
                        assert!(
                            matches!(result, Ok(BoundedPlan::Within { .. })),
                            "{context}"
                        );
                        let bound = journal.rollback_rebuild_bytes(&state, &indexes);
                        let sections_rebuild = journal.rebuild_bytes(RebuildTable::Sections);
                        assert_eq!(
                            sections_rebuild,
                            table_bytes(n as usize, table_slot_bytes(RebuildTable::Sections)),
                            "{context}: a rebuild is the table at its pre-plan length"
                        );
                        assert!(
                            bound <= projected,
                            "{context}: the bound ({bound}) is within what was projected \
                         ({projected})"
                        );
                        let rebuilt = journal
                            .rollback(&mut state, &mut indexes)
                            .expect("rollback");
                        assert_eq!(state, pre, "{context}: restored");
                        if rebuilt.contains(RebuildTable::Sections) {
                            reached_short = true;
                            assert!(
                                bound >= sections_rebuild,
                                "{context}: the rollback rebuilt the sections table, which the \
                             bound ({bound}) priced"
                            );
                        }
                    }
                }
            }
        }
        assert!(
            reached_short,
            "fixture: some rollback rebuilt the sections table"
        );

        let pre = full_of(28);
        let mut state = pre.clone();
        let mut indexes = PlanIndexes::build(&state);
        let (result, journal, _) = plan(
            &mut state,
            &mut indexes,
            OwnedDelta {
                is_diff: true,
                base_version: 0,
                ..OwnedDelta::default()
            },
        );
        assert!(
            matches!(result, Ok(BoundedPlan::Within { .. })),
            "{result:?}"
        );
        assert_eq!(
            journal.rollback_rebuild_bytes(&state, &indexes),
            0,
            "a reset's rollback swaps its tables back whole"
        );
        journal
            .rollback(&mut state, &mut indexes)
            .expect("rollback");
        assert_eq!(state, pre, "the reset rolled back");
    }
}
