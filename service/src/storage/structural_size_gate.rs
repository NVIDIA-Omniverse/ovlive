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

//! The structural route's size gate: how much work one structural commit
//! may do in its single `EVALSHA` before the call commits through the
//! WATCH/MULTI/EXEC path instead.
//!
//! # Why a gate
//!
//! A structural commit is one `EVALSHA` of `update_layer_fast.lua`, and a
//! script holds its shard for its whole run. Its run time grows with the
//! plan: about 10–12 µs per planned mutation in the sandbox. A large reset
//! runs for seconds. Past the 5 s Valkey command timeout the writer's call
//! fails although its commit lands, and past `lua-time-limit` (5 s) every
//! other client of the shard is answered `BUSY` until the script ends. The
//! WATCH path's `EXEC` is not a script, so other clients wait but are never
//! refused, and its hold is shorter at the same work: about 3–8× from about
//! 1,000 units up, near parity at the two smallest edits measured on the
//! 11,000-asset layer. It pays for that with a read of the whole layer on
//! every attempt, which is why the gate sends only the largest commits
//! there.
//!
//! # The two gates
//!
//! Both compare a commit's work ([`StructuralWork`]) with
//! [`STRUCTURAL_COMMIT_WORK_LIMIT`], and a commit over it takes the WATCH path
//! for the rest of its call (`UpdateLayerCall::fall_back_to_watch`):
//!
//! - **After planning** (`LayerStore::update_layer_once_structural`): the
//!   plan's exact work ([`StructuralWork::of_plan`]), on every structural or
//!   reset attempt, between the plan and `commit_structural`. A warm cache
//!   entry always reaches this gate.
//! - **Before a reset's cold start** (`storage::structural_cache_read`): a
//!   reset whose object has no cache entry is estimated from the probe's
//!   index cardinalities and the delta's own rows
//!   ([`reset_work_lower_bound`]). Over the limit, the call skips the cold
//!   start's full read, which its plan could not use, and goes straight to
//!   the WATCH path. The estimate is a lower bound on the plan's work, so it
//!   sends a reset to the WATCH path only when its plan would be over the
//!   limit too; a reset it lets through is caught after planning if it is
//!   over.
//!
//! The gates use no configuration: the limit and the weight below are
//! internal constants, in this one place. Only a `test-support` build can
//! change the limit, per store
//! (`LayerStore::with_structural_commit_work_limit_for_test`).

use std::time::Duration;

use crate::domain::delta::StorageMutation;
use crate::storage::diff_read::SnapshotCardinalities;
use crate::storage::structural::{children_list_digest_held, PreState};

/// The most work ([`StructuralWork::units`]) one structural commit may do:
/// 100,000 work units. A plan over it commits through the WATCH path.
///
/// **What it costs.** Measured 2026-10-06 on the sandbox store (plan `b.dkt`,
/// Epic 6, Task 11; the whole-call crossover of deletes of K placed assets
/// from layers of 1,000, 4,000 and 11,000 placed assets): at 102,781 units the
/// structural `EVALSHA` held its shard for 0.94–1.04 s at p50, 1.4–1.6 s scaled
/// by 1.502 (the measured Azure Managed Redis to sandbox cost ratio, a read
/// factor applied to a write), with single runs up to about twice the p50.
///
/// **The trade.** The structural route holds one shard for up to about 1.5 s
/// (p50, scaled) for edits of up to about 3,000 placed assets, in exchange for
/// never re-reading the layer: the WATCH path reads the whole layer on every
/// attempt, and a commit landing during that read voids it. The whole call
/// favoured the structural route at every size measured — 2.6 to 76 times
/// faster end to end — and its store time stayed lower until an edit removed
/// about 80 % of the layer. So the limit is set by the hold, not by the
/// crossover: above it the hold nears the 5 s command timeout and
/// `lua-time-limit` once scaled, where other clients of the shard are
/// answered `BUSY`.
///
/// What it sends to the WATCH path: the clear or reset of a layer of about
/// 3,000 or more placed assets (about 34 units per asset), deletes or
/// replaces of about 3,000 or more placed assets in one edit, and any edit
/// that adds enough content to plan over 100,000 work units on its own.
///
/// **When to lower it.** A structural commit holding its shard above 2.5 s on
/// dev or at a customer. Two signals show it: samples above the 2.5 s bucket
/// boundary of `live_edit_update_layer_structural_commit_seconds` (its
/// `le="2.5"` count below its total count), and the slow-commit WARN
/// ([`STRUCTURAL_COMMIT_SLOW_WARN`]) with an `elapsed_ms` above 2500.
///
/// **Tests.** A `test-support` build can gate one store at another limit,
/// both gates at once (`LayerStore::with_structural_commit_work_limit_for_test`),
/// so tests reach either side of it with small layers; a release build has no
/// such override and always gates here.
pub const STRUCTURAL_COMMIT_WORK_LIMIT: u64 = 100_000;

/// Children-list entries per work unit: 20. A commit that rewrites or deletes
/// a children list `RPUSH`es the new list and, for its pre-image digest,
/// `LRANGE`s and hashes the stored one (`update_layer_fast.lua`'s digest
/// check), so a one-prim edit under a parent of 100,000 children handles
/// about 200,000 entries.
///
/// Measured 2026-10-06 (same source as [`STRUCTURAL_COMMIT_WORK_LIMIT`]): a
/// one-asset placement under a parent of 11,110 children took 10.6 ms longer
/// than under a parent of 100, for about 22,020 more entries: 0.48 µs per
/// entry, against about 10–12 µs per planned mutation. So one entry is about
/// 1/20 of a mutation.
pub const STRUCTURAL_CHILDREN_ENTRIES_PER_WORK_UNIT: u64 = 20;

/// A structural `EVALSHA` that took longer than this logs a WARN with the
/// plan's work and the layer's size: 2 s, below the 2.5 s hold that is the
/// trigger to lower [`STRUCTURAL_COMMIT_WORK_LIMIT`], and well below the 5 s
/// command timeout.
pub const STRUCTURAL_COMMIT_SLOW_WARN: Duration = Duration::from_secs(2);

/// One structural commit's work: what its single `EVALSHA` handles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StructuralWork {
    /// The plan's mutations: one script record each.
    pub mutations: u64,
    /// The children-list entries the script handles for the plan's list
    /// records: each `SetChildren`'s new entries, which it `RPUSH`es, plus,
    /// once per list, the pre-plan entries of each rewritten or deleted list
    /// whose digest it checks, which it `LRANGE`s and hashes.
    pub children_entries: u64,
}

impl StructuralWork {
    /// The work of `mutations`, planned on `pre_state` (the cache entry's
    /// state before the plan). A list's pre-plan entries count once, on the
    /// first record that touches it, and only for a list that carries a
    /// pre-image digest (`structural::children_list_digest_held`), as the
    /// script reads each such list once; a list under a dangling parent
    /// carries no digest and is not read.
    pub fn of_plan(mutations: &[StorageMutation], pre_state: &impl PreState) -> Self {
        let mut hashed = std::collections::HashSet::new();
        let mut children_entries: u64 = 0;
        for mutation in mutations {
            let parent = match mutation {
                StorageMutation::SetChildren {
                    section_id,
                    children,
                } => {
                    children_entries = children_entries.saturating_add(children.len() as u64);
                    *section_id
                }
                StorageMutation::DeleteChildrenList { section_id } => *section_id,
                _ => continue,
            };
            if children_list_digest_held(parent, pre_state) && hashed.insert(parent) {
                let held = pre_state.list(parent).map_or(0, <[u64]>::len);
                children_entries = children_entries.saturating_add(held as u64);
            }
        }
        Self {
            mutations: mutations.len() as u64,
            children_entries,
        }
    }

    /// The work in units: the mutations plus the children-list entries
    /// divided by [`STRUCTURAL_CHILDREN_ENTRIES_PER_WORK_UNIT`] (rounded
    /// down).
    pub fn units(self) -> u64 {
        self.mutations
            .saturating_add(self.children_entries / STRUCTURAL_CHILDREN_ENTRIES_PER_WORK_UNIT)
    }

    /// Whether the work is over [`STRUCTURAL_COMMIT_WORK_LIMIT`], the
    /// constant only. The gate itself compares with the limit in force,
    /// [`Self::exceeds`], which a `test-support` store can override.
    pub fn over_limit(self) -> bool {
        self.exceeds(STRUCTURAL_COMMIT_WORK_LIMIT)
    }

    /// Whether the work is over `limit` units: the gate's comparison, with
    /// the limit in force (the constant, or a test store's override,
    /// `LayerStore::with_structural_commit_work_limit_for_test`).
    pub fn exceeds(self, limit: u64) -> bool {
        self.units() > limit
    }
}

/// A lower bound on the work of a reset of a layer with the index
/// cardinalities `cards` (the cache probe's) by a delta carrying
/// `content_rows` field and time-sample rows: `3 × sections + fields +
/// time samples + content_rows` units.
///
/// **Why it is a lower bound.** A reset deletes everything the layer holds
/// before it applies the delta (`domain::delta`'s `apply_reset`): per live
/// section a `DeleteSection`, a `CreateTombstone` and a `DeleteChildrenList`,
/// and one delete per field and per time sample. On a well-formed layer those
/// rows are exactly the index members the probe counts, so this is the
/// clear's mutation count less its few fixed records (the root's list, the
/// header) and less the delete it adds per section whose time 0 the layer
/// holds at one stored spelling only (rule E in `domain::delta`: a reset
/// deletes a section's time 0 at both `0.0` and `-0.0`). Each field and time
/// sample the delta carries then plans exactly one more mutation. Left out,
/// so the bound stays under the plan: the delta's structural commands (one
/// can plan nothing, such as a create under a parent the reset removed),
/// rule E's added deletes, the children-list entries the clear hashes (about
/// one per section, divided by 20), and existing tombstones, which a reset
/// does not touch. Measured
/// 2026-10-06: a clear of 11,000 placed assets (55,112 sections) planned
/// 374,561 mutations; with its 209,223 field deletes the bound is 374,559.
///
/// **Where it can be over the plan.** When the layer shrinks between the
/// probe and the WATCH path's own read, or an index holds members whose rows
/// are missing (a corrupt index over-counts). Either way the reset commits
/// through the WATCH path, which is correct at any size; only its route
/// changed.
pub fn reset_work_lower_bound(cards: &SnapshotCardinalities, content_rows: u64) -> u64 {
    cards
        .sections
        .saturating_mul(3)
        .saturating_add(cards.fields)
        .saturating_add(cards.time_samples)
        .saturating_add(content_rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::types::{ObjectState, Section};

    fn section(section_id: u64, parent_id: u64) -> Section {
        Section {
            section_id,
            parent_id,
            name: format!("s{section_id}"),
            section_type: 0,
            section_order: 0,
            first_version: 1,
            last_version: 0,
        }
    }

    /// A state whose root lists `root_children` sections, the first of which
    /// lists `grandchildren` more.
    fn state(root_children: u64, grandchildren: u64) -> ObjectState {
        let mut state = ObjectState::default();
        let root: Vec<u64> = (1..=root_children).collect();
        for id in &root {
            state.sections.insert(*id, section(*id, 0));
        }
        let first = 1;
        let under_first: Vec<u64> = (1000..1000 + grandchildren).collect();
        for id in &under_first {
            state.sections.insert(*id, section(*id, first));
        }
        state.children.insert(0, root);
        if grandchildren > 0 {
            state.children.insert(first, under_first);
        }
        state
    }

    #[test]
    fn units_are_mutations_plus_entries_over_twenty_rounded_down() {
        let work = StructuralWork {
            mutations: 7,
            children_entries: 59,
        };
        assert_eq!(work.units(), 7 + 2);
        assert_eq!(StructuralWork::default().units(), 0);
    }

    /// The gate's comparison, strictly above the limit, at the production
    /// limit (`over_limit`) and at any limit in force (`exceeds`, which a
    /// test store's override reaches): work exactly at the limit — its
    /// children entries rounded down — stays on the structural route.
    #[test]
    fn over_limit_and_exceeds_are_strictly_above_the_limit() {
        let at = StructuralWork {
            mutations: STRUCTURAL_COMMIT_WORK_LIMIT,
            children_entries: STRUCTURAL_CHILDREN_ENTRIES_PER_WORK_UNIT - 1,
        };
        assert!(!at.over_limit(), "at the production limit");
        let above = StructuralWork {
            children_entries: STRUCTURAL_CHILDREN_ENTRIES_PER_WORK_UNIT,
            ..at
        };
        assert!(above.over_limit(), "one unit over the production limit");

        let work = StructuralWork {
            mutations: 1_990,
            children_entries: 10 * STRUCTURAL_CHILDREN_ENTRIES_PER_WORK_UNIT,
        };
        assert_eq!(work.units(), 2_000);
        assert!(!work.exceeds(2_000), "at a given limit");
        assert!(work.exceeds(1_999), "one unit over a given limit");
        assert!(!work.exceeds(2_001), "under a given limit");
    }

    /// The production limit and weight, as the operator documentation states
    /// them: a store without the test-support
    /// override gates at 100,000 units, one children-list entry a twentieth
    /// of one.
    #[test]
    fn production_limit_is_100_000_units_at_20_entries_per_unit() {
        assert_eq!(STRUCTURAL_COMMIT_WORK_LIMIT, 100_000);
        assert_eq!(STRUCTURAL_CHILDREN_ENTRIES_PER_WORK_UNIT, 20);
    }

    #[test]
    fn of_plan_counts_new_entries_and_each_held_lists_pre_plan_entries_once() {
        let pre = state(3, 40);
        let mutations = vec![
            StorageMutation::SetChildren {
                section_id: 1,
                children: (1000..1041).collect(),
            },
            // A second record for the same list: its new entries count, its
            // pre-plan entries do not again.
            StorageMutation::SetChildren {
                section_id: 1,
                children: (1000..1042).collect(),
            },
            StorageMutation::DeleteChildrenList { section_id: 0 },
        ];
        let work = StructuralWork::of_plan(&mutations, &pre);
        assert_eq!(work.mutations, 3);
        assert_eq!(work.children_entries, 41 + 42 + 40 + 3);
    }

    #[test]
    fn of_plan_counts_no_pre_plan_entries_for_a_list_under_a_dangling_parent() {
        let mut pre = state(1, 0);
        // A stored list the entry holds under a parent that is not a live
        // section: it carries no digest, so the script does not read it.
        pre.children.insert(77, vec![5, 6, 7]);
        let mutations = vec![StorageMutation::DeleteChildrenList { section_id: 77 }];
        let work = StructuralWork::of_plan(&mutations, &pre);
        assert_eq!(work.children_entries, 0);
        assert_eq!(work.mutations, 1);
    }

    #[test]
    fn a_one_child_edit_under_a_huge_parent_counts_by_its_entries() {
        let pre = state(1, 100_000);
        let mut children: Vec<u64> = (1000..101_000).collect();
        children.push(9);
        let mutations = vec![StorageMutation::SetChildren {
            section_id: 1,
            children,
        }];
        let work = StructuralWork::of_plan(&mutations, &pre);
        assert_eq!(work.children_entries, 100_001 + 100_000);
        assert_eq!(work.units(), 1 + 200_001 / 20);
    }

    #[test]
    fn reset_bound_is_three_per_section_plus_rows_and_content() {
        let cards = SnapshotCardinalities {
            sections: 55_112,
            tombstones: 9,
            fields: 209_223,
            time_samples: 4,
            children_index: Some(11_112),
        };
        assert_eq!(
            reset_work_lower_bound(&cards, 10),
            3 * 55_112 + 209_223 + 4 + 10
        );
    }

    /// The cardinalities the cache probe reads for `state` on a well-formed
    /// layer: one index member per live section, field and time sample.
    fn cards_of(state: &ObjectState) -> SnapshotCardinalities {
        SnapshotCardinalities {
            sections: state.sections.len() as u64,
            tombstones: 0,
            fields: state.fields.len() as u64,
            time_samples: state.time_samples.len() as u64,
            children_index: Some(state.children.len() as u64),
        }
    }

    /// [`state`] with `rows` fields and `rows` time samples on each section.
    fn state_with_rows(root_children: u64, grandchildren: u64, rows: u64) -> ObjectState {
        use crate::domain::types::{Field, FieldValue, TimeSample};
        use ordered_float::OrderedFloat;

        let mut state = state(root_children, grandchildren);
        let ids: Vec<u64> = state.sections.keys().copied().collect();
        for section_id in ids {
            for row in 0..rows {
                let key_name = format!("f{row}");
                state.fields.insert(
                    (section_id, key_name.clone()),
                    Field {
                        section_id,
                        key_name,
                        value: FieldValue::Inline(vec![1]),
                        set_order: 1,
                        authoring_set_order: 1,
                        version: 1,
                    },
                );
                let time = OrderedFloat(row as f64);
                state.time_samples.insert(
                    (section_id, time),
                    TimeSample {
                        section_id,
                        time,
                        value: FieldValue::Inline(vec![2]),
                        version: 1,
                    },
                );
            }
        }
        state
    }

    /// A reset delta that creates one section under the root and writes
    /// `fields` fields and `samples` time samples on it.
    fn reset_with_content(section_id: u64, fields: u64, samples: u64) -> oodf_ffi::OwnedDelta {
        oodf_ffi::OwnedDelta {
            is_diff: true,
            base_version: 0,
            struct_commands: vec![oodf_ffi::StructCommand::CreateSection(
                oodf_ffi::CommandCreateSection {
                    parent_id: 0,
                    section_id,
                    section_name: "Fresh".to_string(),
                    section_type: 1,
                },
            )],
            set_fields: (0..fields)
                .map(|i| oodf_ffi::SetField {
                    section_id,
                    key_name: format!("k{i}"),
                    value_or_ext_hash: vec![3],
                    ext_value_size: 0,
                    ext_value_index: 0,
                    set_order: i + 1,
                })
                .collect(),
            time_samples: (0..samples)
                .map(|i| oodf_ffi::TimeSample {
                    section_id,
                    time: i as f64,
                    value_or_ext_hash: vec![4],
                    ext_value_size: 0,
                    ext_value_index: 0,
                })
                .collect(),
            source_format: "usda".to_string(),
        }
    }

    /// The bound's claim against the planner itself: a clear of a
    /// well-formed layer (no delta content) plans exactly the bound plus its
    /// two fixed records (the root's list and the header) plus rule E's added
    /// delete per section whose time 0 the layer holds at one stored spelling
    /// (`domain::delta`: a reset deletes a section's time 0 at both `0.0` and
    /// `-0.0`), so the bound is the clear's work less those, never more than
    /// it.
    #[test]
    fn reset_bound_is_the_clears_planned_mutations_less_its_fixed_records_and_rule_e_deletes() {
        let pre = state_with_rows(3, 4, 2);
        let mut planned = pre.clone();
        let mutations = crate::domain::delta::apply_delta(
            &mut planned,
            Some(oodf_ffi::OwnedDelta {
                is_diff: true,
                base_version: 0,
                source_format: "usda".to_string(),
                ..Default::default()
            }),
        )
        .expect("a reset of a well-formed state plans");
        let bound = reset_work_lower_bound(&cards_of(&pre), 0);
        assert_eq!(
            bound,
            3 * 7 + 7 * 2 + 7 * 2,
            "fixture: seven sections with two fields and two time samples each"
        );
        let sections_holding_time_zero = pre
            .time_samples
            .keys()
            .filter(|(_, time)| time.into_inner() == 0.0)
            .count() as u64;
        assert_eq!(
            sections_holding_time_zero, 7,
            "fixture: every section's first sample is at time 0, held at +0.0"
        );
        assert_eq!(
            mutations.len() as u64,
            bound + 2 + sections_holding_time_zero
        );
    }

    /// With delta content, the bound stays under the plan's work: the plan
    /// also creates the delta's section and rewrites the root's list, which
    /// the bound leaves out.
    #[test]
    fn reset_bound_with_content_rows_stays_under_the_plans_work() {
        let pre = state_with_rows(3, 4, 2);
        let delta = reset_with_content(500, 40, 25);
        let mut planned = pre.clone();
        let mutations = crate::domain::delta::apply_delta(&mut planned, Some(delta.clone()))
            .expect("a reset with content plans");
        let content_rows = (delta.set_fields.len() + delta.time_samples.len()) as u64;
        let bound = reset_work_lower_bound(&cards_of(&pre), content_rows);
        let work = StructuralWork::of_plan(&mutations, &pre);
        assert!(
            bound < work.units(),
            "the bound ({bound}) must stay under the plan's work ({}) for the pre-read gate to \
             send only resets whose plan is over the limit too",
            work.units()
        );
    }

    #[test]
    fn reset_bound_saturates() {
        let cards = SnapshotCardinalities {
            sections: u64::MAX,
            tombstones: 0,
            fields: 1,
            time_samples: 1,
            children_index: None,
        };
        assert_eq!(reset_work_lower_bound(&cards, 1), u64::MAX);
    }
}
