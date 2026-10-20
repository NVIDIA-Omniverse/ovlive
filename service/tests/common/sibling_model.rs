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

//! Independent models of how two clients order a scene's siblings, built only
//! from the deltas the wire carries (b.xje, b.ze1).
//!
//! They are the receiving side's answer, written from the clients' published
//! rules rather than from the server's engine, so a test can ask "does a
//! client that applied these deltas hold the server's child order?" without
//! asking the engine to grade itself. `src/domain/delta.rs`'s `ClientModel` is
//! the twin for the domain unit tests, which cannot reach this file; keep the
//! two in step.
//!
//! Each [`Reader`] is a **model**: a transliteration of one client's ordering
//! rules, not the client's code. The client revisions and source anchors each
//! one was written from are recorded in `docs-internal/cross-repo-debugging.md`
//! § Sibling order and move rules. No model proves what a client does with the
//! bytes; only a run of the real client does.
//!
//! Both readers keep each parent's children as **one sequence** across every
//! child list (prims, properties, variant sets), ordered by a per-parent key,
//! and a USD child list is that sequence filtered to the list's children
//! ([`SiblingModel::child_lists`]). Compare child lists, not the sequence:
//! how the lists interleave is invisible to USD.
//!
//! What the two readers share:
//!
//! * `CreateSection`, and a `MoveSection` onto a new parent, **append**: the
//!   child takes the largest key among its new siblings + 1 and lands last. A
//!   create under a missing non-root parent is ignored.
//! * `MoveSection` classifies against the parent the client holds the section
//!   under, never the carried `old_parent_id` — except that a carried parent
//!   equal to the destination is a rename. A missing destination, a
//!   destination under the section, or an ancestor holding the name there
//!   deletes the section; another holder of the name is deleted instead. A
//!   section already under the destination keeps its key and position.
//! * `DeleteSection`, and a catch-up's `DiffDeleteSection`, remove the subtree.
//!   A name conflict on a create deletes the lowest-id holder, as on the
//!   server. A `DiffChangeSection` row deletes no holder: neither client does.
//! * A row whose parent has no row — an *orphan*, which a create or a move
//!   whose name-conflict cascade deletes the parent it names makes (the
//!   holder's subtree loops back to it) — is listed nowhere. When its parent is
//!   born by a create, every row naming it joins its list in `(key, id)` order
//!   before the next command. No real client holds orphan rows: this is the
//!   server's SDD §10.4 rule, modelled so the parity checks extend to histories
//!   that hold them.
//!
//! Where they differ:
//!
//! * [`Reader::KitV1`], the Kit client's default (v1) tree. Keys live in one
//!   space per parent, shared by every child list. A `DiffChangeSection`
//!   applies in wire order: a new row, a row moving to a new parent, or an
//!   orphan row the model holds, is appended (largest key + 1); then, when
//!   its key differs from the command's
//!   `sectionOrder`, it takes that value as its key, and a sibling already
//!   holding it (in any child list) moves to the parent's largest key + 1
//!   (*displacement*). A change whose parent the client does not hold is
//!   deferred until the parent arrives, after the delta's other commands. A
//!   live `ReorderChildren` *slot-fills* within one child list: the named
//!   children, each once and only if they are the parent's children, take the
//!   named children's sorted keys, in command order; every other child keeps
//!   its key. Kit never renumbers.
//! * [`Reader::Ovlive`], the C++ client library. A catch-up reads
//!   `sectionOrder` as an index: per delta, it removes retracted sections and
//!   upserts every `DiffChangeSection` row with the command's value as its
//!   order; then, for each parent with a restated row, it keeps the parent's
//!   other listed children in their current order, stable-sorts the restated
//!   rows by `sectionOrder`, inserts each at `min(sectionOrder, len)`, and
//!   renumbers the parent's whole sequence `0..n-1`. A live `ReorderChildren`
//!   applies OpenUSD's `SdfApplyListOrdering` ([`sdf_apply_list_ordering`],
//!   transliterated from OpenUSD's splice walk, not from the server's engine)
//!   separately to each of the parent's child lists ([`child_list`], from the
//!   sections' `SdfSpecType`s), each list keeping the positions it holds in
//!   the sequence, ignoring ids that are not the parent's children and
//!   repeats, then renumbers the sequence `0..n-1`. That is also the
//!   service's rule (SDD §7.7), so for this reader a live baseline matches
//!   the server after any reorder.
//!
//! The pseudo-root is not special-cased: the root (`parent_id` 0) is placed
//! like any other parent, which only matters for synthetic tests that put
//! several sections under it.
//!
//! Callers feed only deltas the server accepted.

use std::collections::{BTreeMap, HashMap, HashSet};

use live_edit::domain::types::ObjectState;
use oodf_ffi::{CommandDiffChangeSection, OwnedDelta, StructCommand};

/// Which client's reading of the wire a [`SiblingModel`] follows (module doc).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reader {
    /// The Kit client's default v1 tree: shared per-parent keys, catch-up
    /// orders read as keys with displacement, live reorders slot-filled.
    KitV1,
    /// The C++ client library: catch-up orders read as indexes into the
    /// parent's sequence, then renumbered; live reorders by USD's rule per
    /// child list.
    Ovlive,
}

/// Both readers, for tests that run each.
pub const READERS: [Reader; 2] = [Reader::KitV1, Reader::Ovlive];

/// One section as a client holds it; `ty` is its `SdfSpecType`.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ModelRow {
    parent: u64,
    name: String,
    ty: u8,
    key: u32,
}

/// A child list of one parent: `(parent_id, child list)`, where the list is
/// [`child_list`]'s value for the parent's and child's types.
pub type ChildListKey = (u64, Option<u8>);

/// `SdfApplyListOrdering(list, order)`, transliterated from OpenUSD's
/// `SdfListOp<T>::_ReorderKeysHelper` (`pxr/usd/sdf/listOp.cpp:674-726`) and
/// its entry point `SdfApplyListOrdering` (`:897-921`) @ `ee47c679`.
///
/// `scratch` stands for USD's linked list, with a tombstone where an item has
/// been spliced out; `search` maps each item to its position (the last
/// occurrence wins, as `search[*i] = i` does). For the first occurrence in
/// `order` of each item the list holds, the walk splices out that item and
/// every following item `order` does not name, up to the next named item still
/// in the list, onto the end of `result`. Whatever is left in `scratch` was
/// before every named item and goes in front, in its order.
fn sdf_apply_list_ordering(list: &[u64], order: &[u64]) -> Vec<u64> {
    if order.is_empty() || list.is_empty() {
        return list.to_vec();
    }
    let mut order_set = HashSet::new();
    let unique_order: Vec<u64> = order
        .iter()
        .copied()
        .filter(|id| order_set.insert(*id))
        .collect();
    let mut scratch: Vec<Option<u64>> = list.iter().copied().map(Some).collect();
    let search: HashMap<u64, usize> = list.iter().enumerate().map(|(i, id)| (*id, i)).collect();
    let mut result = Vec::new();
    for id in &unique_order {
        let Some(&j) = search.get(id) else {
            continue;
        };
        let mut e = j + 1;
        while e < scratch.len() && scratch[e].is_none_or(|x| !order_set.contains(&x)) {
            e += 1;
        }
        result.extend(scratch[j..e].iter_mut().filter_map(Option::take));
    }
    let mut front: Vec<u64> = scratch.into_iter().flatten().collect();
    front.extend(result);
    front
}

/// Which child list of its parent a child belongs to, from the two sections'
/// `SdfSpecType`s, written from the C++ client library's child-list table (the
/// table at the revision recorded in `docs-internal/cross-repo-debugging.md`
/// § Sibling order and move rules). Values (OpenUSD
/// `pxr/usd/sdf/types.h:71-89`): Unknown 0, Attribute 1, Connection 2,
/// Expression 3, Mapper 4, MapperArg 5, Prim 6, PseudoRoot 7, Relationship 8,
/// RelationshipTarget 9, Variant 10, VariantSet 11.
///
/// `None` is the one fallback list: every pair the table does not name, a
/// root parent (no row, so no type) and an Unknown type.
pub fn child_list(parent_ty: Option<u8>, child_ty: u8) -> Option<u8> {
    match (parent_ty?, child_ty) {
        // PseudoRoot, Prim, Variant → child prims.
        (7 | 6 | 10, 6) => Some(0),
        // Prim, Variant → properties: attributes and relationships together.
        (6 | 10, 1 | 8) => Some(1),
        // Prim, Variant → variant sets.
        (6 | 10, 11) => Some(2),
        // VariantSet → variants.
        (11, 10) => Some(0),
        // Attribute → connections, expressions, mappers.
        (1, 2) => Some(0),
        (1, 3) => Some(1),
        (1, 4) => Some(2),
        // Relationship → targets, expressions.
        (8, 9) => Some(0),
        (8, 3) => Some(1),
        // RelationshipTarget → attributes.
        (9, 1) => Some(0),
        // Mapper → mapper args.
        (4, 5) => Some(0),
        _ => None,
    }
}

/// Split each parent's sequence into its child lists, keeping the sequence's
/// order within each list. `ty_of` gives a section's `SdfSpecType`, or `None`
/// when it has no row; an entry with no row is skipped.
fn split_child_lists<'a>(
    lists: impl IntoIterator<Item = (u64, &'a Vec<u64>)>,
    ty_of: impl Fn(u64) -> Option<u8>,
) -> BTreeMap<ChildListKey, Vec<u64>> {
    let mut out: BTreeMap<ChildListKey, Vec<u64>> = BTreeMap::new();
    for (parent, list) in lists {
        let parent_ty = ty_of(parent);
        for &id in list {
            let Some(ty) = ty_of(id) else {
                continue;
            };
            out.entry((parent, child_list(parent_ty, ty)))
                .or_default()
                .push(id);
        }
    }
    out
}

/// The server's stored order, split into child lists: each stored children
/// list filtered to the parent's true children (an entry whose row names
/// another parent, or with no row, is not a child) and grouped by
/// [`child_list`].
pub fn server_child_lists(state: &ObjectState) -> BTreeMap<ChildListKey, Vec<u64>> {
    let true_children: BTreeMap<u64, Vec<u64>> = state
        .children
        .iter()
        .map(|(parent, list)| {
            let mut seen = HashSet::new();
            let kept: Vec<u64> = list
                .iter()
                .copied()
                .filter(|id| {
                    state
                        .sections
                        .get(id)
                        .is_some_and(|s| s.parent_id == *parent)
                        && seen.insert(*id)
                })
                .collect();
            (*parent, kept)
        })
        .filter(|(_, l)| !l.is_empty())
        .collect();
    split_child_lists(true_children.iter().map(|(p, l)| (*p, l)), |id| {
        state.sections.get(&id).map(|s| s.section_type)
    })
}

/// A client's sections and children lists, as one [`Reader`] builds them.
#[derive(Clone, Debug)]
pub struct SiblingModel {
    reader: Reader,
    rows: BTreeMap<u64, ModelRow>,
    /// Each parent's sequence of children, in the reader's order (ascending
    /// key; keys are unique per parent between deltas).
    lists: BTreeMap<u64, Vec<u64>>,
}

impl SiblingModel {
    /// An empty scene read by `reader`.
    pub fn new(reader: Reader) -> Self {
        Self {
            reader,
            rows: BTreeMap::new(),
            lists: BTreeMap::new(),
        }
    }

    /// The reader this model follows.
    pub fn reader(&self) -> Reader {
        self.reader
    }

    /// Apply one delta: a reset (`is_diff` with `base_version` 0) clears the
    /// model first, then the structural commands apply in order, each by the
    /// reader's rule (module doc). Fields and time samples carry no sibling
    /// order and are ignored.
    pub fn apply(&mut self, delta: &OwnedDelta) {
        if delta.is_diff && delta.base_version == 0 {
            *self = Self::new(self.reader);
        }
        // Kit: changes whose parent is missing, with their command index.
        let mut deferred: Vec<(usize, CommandDiffChangeSection)> = Vec::new();
        // Kit: the last index at which each section was retracted.
        let mut retracted_at: HashMap<u64, usize> = HashMap::new();
        // Ovlive: every restated row, `(parent, sectionOrder, id)`, in
        // command order.
        let mut restated: Vec<(u64, u32, u64)> = Vec::new();
        for (index, cmd) in delta.struct_commands.iter().enumerate() {
            match cmd {
                StructCommand::CreateSection(c) => {
                    self.create(c.parent_id, c.section_id, &c.section_name, c.section_type);
                }
                StructCommand::DiffChangeSection(c) => match self.reader {
                    Reader::KitV1 => {
                        if !self.kit_change(c) {
                            deferred.push((index, c.clone()));
                        }
                    }
                    Reader::Ovlive => {
                        self.ovlive_upsert(c);
                        restated.push((c.parent_id, c.section_order, c.section_id));
                    }
                },
                StructCommand::DeleteSection(c) => self.delete(c.section_id),
                StructCommand::DiffDeleteSection(c) => {
                    retracted_at.insert(c.section_id, index);
                    self.delete(c.section_id);
                }
                StructCommand::MoveSection(c) => {
                    self.move_section(c.section_id, c.old_parent_id, c.new_parent_id, &c.new_name);
                }
                StructCommand::ReorderChildren(c) => self.reorder(c.section_id, &c.children_list),
            }
        }
        match self.reader {
            Reader::KitV1 => self.kit_apply_deferred(deferred, &retracted_at),
            Reader::Ovlive => self.ovlive_place(&restated),
        }
    }

    /// Place a section as a client already holds it: last in its parent's
    /// sequence, of type `ty`, at `key`. The caller passes the sequence in
    /// order with increasing keys. Unlike a create there is no name-conflict
    /// cascade — a scene a client holds can carry duplicate names under one
    /// parent, which a same-parent rename produces.
    pub fn hold(&mut self, parent: u64, id: u64, name: &str, ty: u8, key: u32) {
        self.rows.insert(
            id,
            ModelRow {
                parent,
                name: name.to_string(),
                ty,
                key,
            },
        );
        self.lists.entry(parent).or_default().push(id);
    }

    /// Hold an orphan row: a section whose parent has no row, listed nowhere
    /// (module doc).
    pub fn hold_orphan(&mut self, parent: u64, id: u64, name: &str, ty: u8, key: u32) {
        self.rows.insert(
            id,
            ModelRow {
                parent,
                name: name.to_string(),
                ty,
                key,
            },
        );
    }

    /// The key the model holds `id` at, if it holds `id` under `parent`.
    pub fn key_under(&self, parent: u64, id: u64) -> Option<u32> {
        self.rows
            .get(&id)
            .filter(|r| r.parent == parent)
            .map(|r| r.key)
    }

    /// Every non-empty sequence of children, by parent: the reader's order
    /// across all child lists.
    pub fn lists(&self) -> BTreeMap<u64, Vec<u64>> {
        self.lists
            .iter()
            .filter(|(_, l)| !l.is_empty())
            .map(|(p, l)| (*p, l.clone()))
            .collect()
    }

    /// Every non-empty USD child list, by `(parent, child list)`: what the
    /// client's USD layer shows.
    pub fn child_lists(&self) -> BTreeMap<ChildListKey, Vec<u64>> {
        split_child_lists(self.lists.iter().map(|(p, l)| (*p, l)), |id| {
            self.rows.get(&id).map(|r| r.ty)
        })
    }

    /// The lowest-id section named `name` under `parent`, other than `except`.
    fn holder(&self, parent: u64, name: &str, except: u64) -> Option<u64> {
        self.rows
            .iter()
            .find(|(id, r)| **id != except && r.parent == parent && r.name == name)
            .map(|(id, _)| *id)
    }

    /// `candidate` is `ancestor` or sits under it.
    fn is_descendant(&self, mut candidate: u64, ancestor: u64) -> bool {
        let mut seen = HashSet::new();
        loop {
            if candidate == ancestor {
                return true;
            }
            if !seen.insert(candidate) {
                return false;
            }
            match self.rows.get(&candidate) {
                Some(row) => candidate = row.parent,
                None => return false,
            }
        }
    }

    /// One above the largest key among `parent`'s listed children; 0 under
    /// an empty parent.
    fn append_key(&self, parent: u64) -> u32 {
        self.lists
            .get(&parent)
            .and_then(|l| l.iter().map(|id| self.rows[id].key).max())
            .map_or(0, |largest| largest.saturating_add(1))
    }

    fn unlink(&mut self, parent: u64, id: u64) {
        if let Some(list) = self.lists.get_mut(&parent) {
            list.retain(|c| *c != id);
        }
    }

    /// Re-sort `parent`'s sequence by key (stable, so equal keys keep their
    /// order).
    fn sort_by_key(&mut self, parent: u64) {
        let rows = &self.rows;
        if let Some(list) = self.lists.get_mut(&parent) {
            list.sort_by_key(|c| rows[c].key);
        }
    }

    /// List `id` under `parent` after every sibling whose key is ≤ its own —
    /// unless `parent` is neither the root nor a row, when `id` is an orphan
    /// and is listed nowhere.
    fn insert_by_key(&mut self, parent: u64, id: u64) {
        if parent != 0 && !self.rows.contains_key(&parent) {
            return;
        }
        let key = self.rows[&id].key;
        let rows = &self.rows;
        let list = self.lists.entry(parent).or_default();
        let at = list
            .iter()
            .position(|c| rows[c].key > key)
            .unwrap_or(list.len());
        list.insert(at, id);
    }

    /// `id` was just given a row by a create: every row already naming it
    /// joins its list, in `(key, id)` order (the server's SDD §10.4 rule).
    fn adopt(&mut self, id: u64) {
        if id == 0 {
            return;
        }
        let mut orphans: Vec<(u32, u64)> = self
            .rows
            .iter()
            .filter(|(cid, r)| r.parent == id && **cid != id)
            .map(|(cid, r)| (r.key, *cid))
            .collect();
        orphans.sort_unstable();
        for (_, cid) in orphans {
            self.insert_by_key(id, cid);
        }
    }

    fn create(&mut self, parent: u64, id: u64, name: &str, ty: u8) {
        if self.rows.contains_key(&id) || (parent != 0 && !self.rows.contains_key(&parent)) {
            return;
        }
        if let Some(holder) = self.holder(parent, name, id) {
            self.delete(holder);
        }
        let key = self.append_key(parent);
        self.rows.insert(
            id,
            ModelRow {
                parent,
                name: name.to_string(),
                ty,
                key,
            },
        );
        // The holder's cascade can take the parent with it (a holder whose
        // subtree loops back to the parent); the new row is then an orphan.
        if parent == 0 || self.rows.contains_key(&parent) {
            self.lists.entry(parent).or_default().push(id);
        }
        self.adopt(id);
    }

    /// The Kit client's `DiffChangeSection`: link a new row, or relink a row
    /// whose parent changed, at the parent's append key, then install the
    /// command's order as the key, displacing a sibling holding it. Returns
    /// `false`, applying nothing, when the parent is not held.
    fn kit_change(&mut self, c: &CommandDiffChangeSection) -> bool {
        let (parent, id) = (c.parent_id, c.section_id);
        if parent != 0 && !self.rows.contains_key(&parent) {
            return false;
        }
        match self.rows.get(&id).map(|r| r.parent) {
            None => {
                let key = self.append_key(parent);
                self.rows.insert(
                    id,
                    ModelRow {
                        parent,
                        name: c.section_name.clone(),
                        ty: c.section_type,
                        key,
                    },
                );
                self.lists.entry(parent).or_default().push(id);
            }
            Some(old_parent) => {
                // An orphan row the model holds (the server's adoption rule;
                // no Kit client holds one) is not listed under its parent, so
                // it is linked like a row moving in.
                let unlisted = self.lists.get(&parent).is_none_or(|l| !l.contains(&id));
                if old_parent != parent || unlisted {
                    self.unlink(old_parent, id);
                    let key = self.append_key(parent);
                    let row = self.rows.get_mut(&id).expect("checked above");
                    row.parent = parent;
                    row.key = key;
                    self.lists.entry(parent).or_default().push(id);
                }
                let row = self.rows.get_mut(&id).expect("checked above");
                row.name = c.section_name.clone();
                row.ty = c.section_type;
            }
        }
        if self.rows[&id].key != c.section_order {
            self.kit_set_new_order(parent, id, c.section_order);
        }
        true
    }

    /// The Kit client's displacement: a sibling of `id` holding `order` (in
    /// any child list) moves to the parent's largest key + 1, then `id` takes
    /// `order`.
    fn kit_set_new_order(&mut self, parent: u64, id: u64, order: u32) {
        let holder = self
            .lists
            .get(&parent)
            .and_then(|l| l.iter().find(|c| **c != id && self.rows[c].key == order))
            .copied();
        if let Some(holder) = holder {
            let next = self.append_key(parent);
            self.rows
                .get_mut(&holder)
                .expect("a listed child has a row")
                .key = next;
        }
        self.rows.get_mut(&id).expect("linked above").key = order;
        self.sort_by_key(parent);
    }

    /// The Kit client's deferred changes: once the delta's commands have
    /// applied, a change whose parent has arrived applies, repeatedly, unless
    /// a later retraction in the delta superseded it.
    fn kit_apply_deferred(
        &mut self,
        mut pending: Vec<(usize, CommandDiffChangeSection)>,
        retracted_at: &HashMap<u64, usize>,
    ) {
        loop {
            let before = pending.len();
            let mut still = Vec::new();
            for (index, c) in pending {
                if retracted_at
                    .get(&c.section_id)
                    .is_some_and(|&at| at > index)
                {
                    continue;
                }
                if !self.kit_change(&c) {
                    still.push((index, c));
                }
            }
            pending = still;
            if pending.is_empty() || pending.len() == before {
                return;
            }
        }
    }

    /// The C++ client library's `DiffChangeSection` before placement: a new
    /// row takes the command's order as its order; an existing row is
    /// rewritten, leaving its old parent's sequence when its parent changes.
    /// [`ovlive_place`](Self::ovlive_place) lists it.
    fn ovlive_upsert(&mut self, c: &CommandDiffChangeSection) {
        let id = c.section_id;
        if let Some(old_parent) = self.rows.get(&id).map(|r| r.parent) {
            self.unlink(old_parent, id);
        }
        self.rows.insert(
            id,
            ModelRow {
                parent: c.parent_id,
                name: c.section_name.clone(),
                ty: c.section_type,
                key: c.section_order,
            },
        );
    }

    /// The C++ client library's placement of a catch-up's restated rows, per
    /// parent: the other listed children in their current order, then each
    /// restated row (stable-sorted by its order) inserted at
    /// `min(order, len)`, and the whole sequence renumbered `0..n-1`. A
    /// restated row under a parent with no row stays an orphan.
    fn ovlive_place(&mut self, restated: &[(u64, u32, u64)]) {
        if restated.is_empty() {
            return;
        }
        let restated_ids: HashSet<u64> = restated.iter().map(|(_, _, id)| *id).collect();
        let mut by_parent: BTreeMap<u64, Vec<(u32, u64)>> = BTreeMap::new();
        for &(parent, order, id) in restated {
            by_parent.entry(parent).or_default().push((order, id));
        }
        for (parent, mut entries) in by_parent {
            if parent != 0 && !self.rows.contains_key(&parent) {
                continue;
            }
            let mut others: Vec<(u32, u64)> = self
                .lists
                .get(&parent)
                .map(|l| {
                    l.iter()
                        .filter(|id| !restated_ids.contains(id))
                        .map(|id| (self.rows[id].key, *id))
                        .collect()
                })
                .unwrap_or_default();
            others.sort_unstable();
            let mut children: Vec<u64> = others.into_iter().map(|(_, id)| id).collect();
            entries.sort_by_key(|(order, _)| *order);
            for (order, id) in entries {
                // A row a later command in the delta retracted, or moved
                // under another parent, is not placed here.
                if self.rows.get(&id).is_none_or(|r| r.parent != parent) || children.contains(&id) {
                    continue;
                }
                let at = (order as usize).min(children.len());
                children.insert(at, id);
            }
            for (position, id) in children.iter().enumerate() {
                self.rows.get_mut(id).expect("a placed child has a row").key =
                    u32::try_from(position).expect("a model list is short");
            }
            self.lists.insert(parent, children);
        }
    }

    fn delete(&mut self, id: u64) {
        let Some(parent) = self.rows.get(&id).map(|r| r.parent) else {
            return;
        };
        self.unlink(parent, id);
        let mut stack = vec![id];
        let mut seen = HashSet::new();
        while let Some(sid) = stack.pop() {
            if !seen.insert(sid) {
                continue;
            }
            self.rows.remove(&sid);
            if let Some(children) = self.lists.remove(&sid) {
                stack.extend(children);
            }
        }
    }

    fn move_section(&mut self, id: u64, carried: u64, dest: u64, name: &str) {
        let Some(current) = self.rows.get(&id).map(|r| r.parent) else {
            return;
        };
        if carried == dest {
            self.rows.get_mut(&id).expect("checked above").name = name.to_string();
            return;
        }
        if dest != 0 && !self.rows.contains_key(&dest) {
            self.delete(id);
            return;
        }
        if self.is_descendant(dest, id) {
            self.delete(id);
            return;
        }
        if let Some(holder) = self.holder(dest, name, id) {
            if self.is_descendant(id, holder) {
                self.delete(id);
                return;
            }
            self.delete(holder);
        }
        if dest == current {
            self.rows.get_mut(&id).expect("checked above").name = name.to_string();
            return;
        }
        self.unlink(current, id);
        let key = self.append_key(dest);
        let row = self.rows.get_mut(&id).expect("checked above");
        row.parent = dest;
        row.name = name.to_string();
        row.key = key;
        // The holder's cascade can take the destination with it (a holder
        // whose subtree loops back to the destination); the section then
        // sits under a parent with no row, an orphan listed nowhere.
        if dest == 0 || self.rows.contains_key(&dest) {
            self.lists.entry(dest).or_default().push(id);
        }
    }

    fn reorder(&mut self, parent: u64, named: &[u64]) {
        let list = self.lists.get(&parent).cloned().unwrap_or_default();
        if !named.iter().any(|id| list.contains(id)) {
            return;
        }
        let parent_ty = self.rows.get(&parent).map(|r| r.ty);
        let list_of = |id: &u64| child_list(parent_ty, self.rows[id].ty);
        match self.reader {
            Reader::KitV1 => {
                // Slot fill per child list: the named children, each once,
                // take their own sorted keys in command order.
                let mut seen = HashSet::new();
                let mut by_list: BTreeMap<Option<u8>, Vec<u64>> = BTreeMap::new();
                for id in named {
                    if list.contains(id) && seen.insert(*id) {
                        by_list.entry(list_of(id)).or_default().push(*id);
                    }
                }
                for members in by_list.values() {
                    let mut keys: Vec<u32> = members.iter().map(|id| self.rows[id].key).collect();
                    keys.sort_unstable();
                    for (id, key) in members.iter().zip(keys) {
                        self.rows.get_mut(id).expect("a listed child has a row").key = key;
                    }
                }
                self.sort_by_key(parent);
            }
            Reader::Ovlive => {
                let mut groups: BTreeMap<Option<u8>, Vec<usize>> = BTreeMap::new();
                for (position, id) in list.iter().enumerate() {
                    groups.entry(list_of(id)).or_default().push(position);
                }
                let mut placed = list.clone();
                for positions in groups.values() {
                    let members: Vec<u64> = positions.iter().map(|&p| list[p]).collect();
                    for (&p, id) in positions
                        .iter()
                        .zip(sdf_apply_list_ordering(&members, named))
                    {
                        placed[p] = id;
                    }
                }
                for (position, id) in placed.iter().enumerate() {
                    self.rows.get_mut(id).expect("a listed child has a row").key =
                        u32::try_from(position).expect("a model list is short");
                }
                self.lists.insert(parent, placed);
            }
        }
    }
}
