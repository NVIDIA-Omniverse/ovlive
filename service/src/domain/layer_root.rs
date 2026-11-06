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

//! The layer's root section (SDD §9.7): the state a `CreateLayer` commits,
//! and the check that keeps an `UpdateLayer` from removing a layer's last
//! top-level section.
//!
//! Every reader of a layer — USD, Kit and the other live clients
//! (`docs-internal/cross-repo-debugging.md`) — expects the layer's pseudo-root
//! as a section whose `parent_id` is 0. Kit hangs on a layer with none: it
//! opens it, then never finishes creating the first prim it authors there. So:
//!
//! - [`build_creation_state`] builds the state a `CreateLayer` commits. Content
//!   that leaves at least one top-level section is kept exactly as sent.
//!   Content that leaves the layer completely empty (no section, field or
//!   time sample: a header with no chunks, an empty delta, a root created and
//!   deleted again) gets the service-supplied root instead — exactly the state
//!   Kit's own create content builds. Content that leaves sections or value
//!   rows but no top-level section is refused, since supplying a root into it
//!   would attach or adopt data the client did not put under one.
//! - [`check_keeps_top_level_section`] refuses an update that would take a
//!   layer from at least one top-level section to none. It runs at the two
//!   structural commit routes (`storage::layer`'s WATCH path and its cached
//!   structural route); the Lua fast path commits only deltas that cannot
//!   change the tree (`storage::lua::is_fast_path`).
//!
//! Neither rule is in the delta engine (`domain::delta`), and storage-level
//! creates (`LayerStore::create_layer*`) do not apply them: checkpoint decode,
//! compose, stream replay and the structural cache's repair re-apply stored
//! history through that engine, and must reproduce it exactly — layers stored
//! without a root before this rule included. A layer that already has no
//! top-level section is never refused an update for it: it had none to lose.

use oodf_ffi::{CommandDiffChangeSection, OwnedDelta, StructCommand};

use crate::domain::delta::{apply_delta_into, ApplyDeltaStats, DeltaError, MutationSink};
use crate::domain::types::{ObjectState, ObjectStateOf, ValueRows};

/// The `section_id` of the root the service supplies: the id Kit gives the
/// pseudo-root of every layer it creates. Not
/// `storage::structural::ROOT_SECTION_ID`, which names the synthetic parent 0.
pub const SUPPLIED_ROOT_SECTION_ID: u64 = 1;

/// The `section_type` of the root the service supplies: the type Kit gives the
/// pseudo-root of every layer it creates.
pub const SUPPLIED_ROOT_SECTION_TYPE: u8 = 7;

/// The state a `CreateLayer` commits, from [`build_creation_state`].
#[derive(Debug)]
pub struct CreationState {
    /// The layer's first state, at sequence 1.
    pub state: ObjectState,
    /// Whether the service supplied the root section: the content left the
    /// layer completely empty, so `state` holds only the root.
    pub root_supplied: bool,
}

/// Whether `state` holds a top-level section: some id listed under parent 0
/// (`state.children[&0]`) has a row whose `parent_id` is 0. Allocates nothing,
/// and costs one step per listed top-level section.
pub fn has_top_level_section<R: ValueRows>(state: &ObjectStateOf<R>) -> bool {
    state.children.get(&0).is_some_and(|ids| {
        ids.iter().any(|id| {
            state
                .sections
                .get(id)
                .is_some_and(|section| section.parent_id == 0)
        })
    })
}

/// Build the state a `CreateLayer` with content `delta` commits (SDD §9.7).
///
/// 1. Apply `delta` to a fresh state, discarding the mutations: a create
///    materializes the whole object from the state, never from a mutation
///    list. `stats` counts this apply only.
/// 2. If the result has a top-level section, return it unchanged.
/// 3. If it holds no section, field or time sample, return instead the state
///    Kit's own create content builds on a fresh state: a reset holding
///    `DiffChangeSection { parent_id: 0, section_id: 1, section_name: "",
///    section_type: 7, section_order: 0 }`, with the source format step 1
///    stored. One apply on a fresh state, so the sequence is still 1; its
///    stats are not added to `stats`, so nothing is counted twice.
/// 4. Otherwise refuse it ([`DeltaError::CreateWithoutTopLevelSection`]).
///
/// Every error of step 1's apply is returned as it is.
///
/// Used by the `CreateLayer` handler, for both a create and the rebuild of a
/// re-send answered `ALREADY_COMMITTED`, so the two build the same state.
pub fn build_creation_state(
    delta: Option<OwnedDelta>,
    stats: &mut ApplyDeltaStats,
) -> Result<CreationState, DeltaError> {
    let mut state = ObjectState::default();
    apply_delta_into(&mut state, delta, &mut MutationSink::Discard, stats)?;
    if has_top_level_section(&state) {
        return Ok(CreationState {
            state,
            root_supplied: false,
        });
    }
    if !state.sections.is_empty() || !state.fields.is_empty() || !state.time_samples.is_empty() {
        return Err(DeltaError::CreateWithoutTopLevelSection);
    }
    let mut supplied = ObjectState::default();
    let mut supplied_stats = ApplyDeltaStats::default();
    apply_delta_into(
        &mut supplied,
        Some(supplied_root_create(state.source_format)),
        &mut MutationSink::Discard,
        &mut supplied_stats,
    )?;
    Ok(CreationState {
        state: supplied,
        root_supplied: true,
    })
}

/// Kit's own create content for an empty layer: a reset creating only the
/// pseudo-root, carrying `source_format`.
fn supplied_root_create(source_format: String) -> OwnedDelta {
    OwnedDelta {
        is_diff: true,
        base_version: 0,
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 0,
            section_id: SUPPLIED_ROOT_SECTION_ID,
            section_name: String::new(),
            section_type: SUPPLIED_ROOT_SECTION_TYPE,
            section_order: 0,
        })],
        set_fields: Vec::new(),
        time_samples: Vec::new(),
        source_format,
    }
}

/// Refuse an update plan that took a layer from at least one top-level
/// section to none ([`DeltaError::RemovesLastTopLevelSection`]).
///
/// `had_top_level` is [`has_top_level_section`] of the state the plan ran on,
/// taken before the plan; `planned` is that state after it. Called by both
/// structural commit routes after a plan that fits its allowance, before any
/// commit work. A layer stored without a top-level section is never refused
/// here.
pub fn check_keeps_top_level_section<R: ValueRows>(
    had_top_level: bool,
    planned: &ObjectStateOf<R>,
) -> Result<(), DeltaError> {
    if had_top_level && !has_top_level_section(planned) {
        return Err(DeltaError::RemovesLastTopLevelSection);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::delta::apply_delta;
    use crate::domain::types::Section;
    use oodf_ffi::{CommandCreateSection, CommandDeleteSection, SetField, TimeSample};

    /// A `source_format` a client could send, to show which one the supplied
    /// root's reset carries.
    const SOURCE_FORMAT: &str = "usda";

    /// The OODF bytes of the Kit client library's no-op delta, the content
    /// its create sends for an empty layer: the OMPF part only, as
    /// little-endian words. An ordinary (not reset) delta whose four vectors
    /// are present and empty, with no `source_format`.
    const KIT_CLIENT_LIBRARY_NO_OP_DELTA: [u32; 15] = [
        0x0000_0018,
        0x4644_4f4f,
        0x0014_0010,
        0x0000_0000,
        0x0008_0004,
        0x0010_000c,
        0x0000_0010,
        0x0000_001c,
        0x0000_0014,
        0x0000_000c,
        0x0000_0004,
        0x0000_0000,
        0x0000_0000,
        0x0000_0000,
        0x0000_0000,
    ];

    /// Kit's own create content for an empty layer, as captured
    /// (`tools/fixtures/gze_inbound_deltas.json`, entries 0 and 1, which are
    /// identical; decoded by hand): a reset holding only
    /// `DiffChangeSection { parent_id: 0, section_id: 1, section_name: "",
    /// section_type: 7, section_order: 0 }`. The capture carries no
    /// `source_format`; `source_format` is set here for the cases that send
    /// one. Written with literals, not the module's constants, so a wrong
    /// constant fails the comparison.
    fn kit_create_content(source_format: &str) -> OwnedDelta {
        OwnedDelta {
            is_diff: true,
            base_version: 0,
            struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
                parent_id: 0,
                section_id: 1,
                section_name: String::new(),
                section_type: 7,
                section_order: 0,
            })],
            set_fields: Vec::new(),
            time_samples: Vec::new(),
            source_format: source_format.to_owned(),
        }
    }

    /// The state the delta engine alone builds for `delta` on a fresh layer.
    fn engine_state(delta: Option<OwnedDelta>) -> ObjectState {
        let mut state = ObjectState::default();
        apply_delta(&mut state, delta).expect("the engine applies the content");
        state
    }

    fn build(delta: Option<OwnedDelta>) -> Result<CreationState, DeltaError> {
        build_creation_state(delta, &mut ApplyDeltaStats::default())
    }

    fn ordinary(struct_commands: Vec<StructCommand>) -> OwnedDelta {
        OwnedDelta {
            struct_commands,
            ..OwnedDelta::default()
        }
    }

    fn create_section(parent_id: u64, section_id: u64, name: &str, ty: u8) -> StructCommand {
        StructCommand::CreateSection(CommandCreateSection {
            parent_id,
            section_id,
            section_name: name.to_owned(),
            section_type: ty,
        })
    }

    fn change_section(parent_id: u64, section_id: u64, name: &str, ty: u8) -> StructCommand {
        StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id,
            section_id,
            section_name: name.to_owned(),
            section_type: ty,
            section_order: 0,
        })
    }

    fn inline_field(section_id: u64, key_name: &str) -> SetField {
        SetField {
            section_id,
            key_name: key_name.to_owned(),
            value_or_ext_hash: b"value".to_vec(),
            ..SetField::default()
        }
    }

    fn inline_sample(section_id: u64, time: f64) -> TimeSample {
        TimeSample {
            section_id,
            time,
            value_or_ext_hash: b"sample".to_vec(),
            ..TimeSample::default()
        }
    }

    fn section_row(section_id: u64, parent_id: u64) -> Section {
        Section {
            section_id,
            parent_id,
            name: format!("s{section_id}"),
            section_type: 0,
            section_order: 0,
            first_version: 1,
            last_version: 1,
        }
    }

    /// Assert `delta` creates the layer holding only the supplied root: the
    /// state Kit's own create content builds, carrying `source_format`, at
    /// sequence 1.
    fn assert_root_supplied(case: &str, delta: Option<OwnedDelta>, source_format: &str) {
        let created = build(delta).unwrap_or_else(|e| panic!("{case}: refused: {e}"));
        assert!(created.root_supplied, "{case}: the root must be supplied");
        assert_eq!(created.state.sequence_num, 1, "{case}: sequence");
        assert_eq!(
            created.state,
            engine_state(Some(kit_create_content(source_format))),
            "{case}: the state must equal Kit's own create content's"
        );
        let root = &created.state.sections[&1];
        assert_eq!(
            (
                root.parent_id,
                root.name.as_str(),
                root.section_type,
                root.section_order
            ),
            (0, "", 7, 0),
            "{case}: the supplied root row"
        );
        assert_eq!(
            created.state.sections.len(),
            1,
            "{case}: the root is the only section"
        );
    }

    /// Assert `delta` keeps its own content: the state the engine builds for
    /// it, with no root supplied.
    fn assert_content_kept(case: &str, delta: OwnedDelta) {
        let expected = engine_state(Some(delta.clone()));
        let created = build(Some(delta)).unwrap_or_else(|e| panic!("{case}: refused: {e}"));
        assert!(!created.root_supplied, "{case}: no root may be supplied");
        assert_eq!(created.state, expected, "{case}: the content's own state");
    }

    /// Assert `delta` is refused as data without a top-level section.
    fn assert_refused(case: &str, delta: OwnedDelta) {
        let result = build(Some(delta));
        assert!(
            matches!(result, Err(DeltaError::CreateWithoutTopLevelSection)),
            "{case}: expected CreateWithoutTopLevelSection, got {result:?}"
        );
    }

    #[test]
    fn build_creation_state_no_content_supplies_the_root() {
        assert_root_supplied("no content", None, "");
    }

    #[test]
    fn build_creation_state_empty_ordinary_delta_supplies_the_root() {
        assert_root_supplied("empty ordinary delta", Some(OwnedDelta::default()), "");
    }

    #[test]
    fn build_creation_state_kit_client_library_no_op_delta_supplies_the_root() {
        let bytes: Vec<u8> = KIT_CLIENT_LIBRARY_NO_OP_DELTA
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect();
        let delta = oodf_ffi::parse_delta(&bytes).expect("the no-op delta parses");
        assert!(
            !delta.is_diff && delta.struct_commands.is_empty(),
            "the no-op delta is an ordinary empty delta, got {delta:?}"
        );
        assert_root_supplied("the Kit client library's no-op delta", Some(delta), "");
    }

    /// The supplied root's reset carries the `source_format` the content's own
    /// apply stored: a reset stores the one it carries.
    #[test]
    fn build_creation_state_empty_reset_supplies_the_root_with_its_source_format() {
        let delta = OwnedDelta {
            is_diff: true,
            base_version: 0,
            source_format: SOURCE_FORMAT.to_owned(),
            ..OwnedDelta::default()
        };
        assert_root_supplied("empty reset", Some(delta), SOURCE_FORMAT);
    }

    /// An ordinary delta stores no `source_format`, so neither does the root
    /// supplied for it.
    #[test]
    fn build_creation_state_empty_ordinary_delta_with_a_source_format_supplies_the_root_without_it()
    {
        let delta = OwnedDelta {
            source_format: SOURCE_FORMAT.to_owned(),
            ..OwnedDelta::default()
        };
        assert_root_supplied("empty ordinary delta with a source format", Some(delta), "");
    }

    #[test]
    fn build_creation_state_root_created_and_deleted_in_one_delta_supplies_the_root() {
        let delta = ordinary(vec![
            create_section(0, 1, "", 7),
            StructCommand::DeleteSection(CommandDeleteSection {
                parent_id: 0,
                section_id: 1,
            }),
        ]);
        assert_root_supplied("root created and deleted", Some(delta), "");
    }

    /// A `CreateSection` the engine skips for its missing parent leaves the
    /// layer empty, so the root is supplied; the skip is counted once, from
    /// the content's own apply.
    #[test]
    fn build_creation_state_skipped_create_supplies_the_root_and_counts_the_skip_once() {
        let mut stats = ApplyDeltaStats::default();
        let created = build_creation_state(
            Some(ordinary(vec![create_section(42, 5, "orphan", 0)])),
            &mut stats,
        )
        .expect("the create commits");
        assert!(created.root_supplied, "the root must be supplied");
        assert_eq!(stats.create_section_skipped_missing_parent, 1);
    }

    #[test]
    fn build_creation_state_one_top_level_section_of_each_type_is_kept() {
        for ty in [0, 6, 7] {
            assert_content_kept(
                &format!("top-level section of type {ty}"),
                ordinary(vec![create_section(0, 1, "", ty)]),
            );
        }
    }

    #[test]
    fn build_creation_state_top_level_section_with_another_id_is_kept() {
        assert_content_kept(
            "top-level section 9",
            ordinary(vec![create_section(0, 9, "root", 7)]),
        );
    }

    #[test]
    fn build_creation_state_two_top_level_sections_are_kept() {
        assert_content_kept(
            "two top-level sections",
            ordinary(vec![
                create_section(0, 1, "a", 7),
                create_section(0, 2, "b", 0),
            ]),
        );
    }

    /// Kit's own create content keeps its root as sent: the root is not
    /// supplied a second time.
    #[test]
    fn build_creation_state_kit_create_content_is_kept() {
        assert_content_kept("Kit's create content", kit_create_content(""));
    }

    /// A rooted create with data under the root is kept with its data.
    #[test]
    fn build_creation_state_rooted_content_with_fields_and_samples_is_kept() {
        let mut delta = ordinary(vec![
            create_section(0, 1, "", 7),
            create_section(1, 2, "prim", 0),
        ]);
        delta.set_fields = vec![inline_field(2, "attr")];
        delta.time_samples = vec![inline_sample(2, 1.0)];
        assert_content_kept("rooted content with data", delta);
    }

    /// An orphan section row: a `DiffChangeSection` under a parent with no
    /// row is stored, unlike a skipped `CreateSection`.
    #[test]
    fn build_creation_state_orphan_rows_only_are_refused() {
        assert_refused(
            "orphan section rows",
            ordinary(vec![change_section(42, 5, "orphan", 0)]),
        );
    }

    #[test]
    fn build_creation_state_fields_on_section_1_only_are_refused() {
        let delta = OwnedDelta {
            set_fields: vec![inline_field(1, "metadata")],
            ..OwnedDelta::default()
        };
        assert_refused("fields on section 1 only", delta);
    }

    #[test]
    fn build_creation_state_time_samples_only_are_refused() {
        let delta = OwnedDelta {
            time_samples: vec![inline_sample(1, 0.0)],
            ..OwnedDelta::default()
        };
        assert_refused("time samples only", delta);
    }

    /// An engine error is returned as it is, not turned into the root rule's
    /// refusal or a supplied root.
    #[test]
    fn build_creation_state_engine_error_is_returned_unchanged() {
        let result = build(Some(ordinary(vec![
            create_section(0, 1, "", 7),
            create_section(0, 1, "", 7),
        ])));
        assert!(
            matches!(result, Err(DeltaError::DuplicateSection { .. })),
            "expected the engine's DuplicateSection, got {result:?}"
        );
    }

    #[test]
    fn has_top_level_section_empty_state_is_false() {
        assert!(!has_top_level_section(&ObjectState::default()));
    }

    #[test]
    fn has_top_level_section_orphans_only_is_false() {
        let state = engine_state(Some(ordinary(vec![change_section(42, 5, "orphan", 0)])));
        assert!(!state.sections.is_empty(), "the orphan row is stored");
        assert!(!has_top_level_section(&state));
    }

    #[test]
    fn has_top_level_section_listed_id_without_a_row_is_false() {
        let mut state = ObjectState::default();
        state.children.insert(0, vec![5]);
        assert!(!has_top_level_section(&state));
    }

    #[test]
    fn has_top_level_section_listed_id_with_a_foreign_parent_is_false() {
        let mut state = ObjectState::default();
        state.sections.insert(5, section_row(5, 3));
        state.children.insert(0, vec![5]);
        assert!(!has_top_level_section(&state));
    }

    /// A top-level row the list under 0 does not name does not count: the
    /// predicate reads the list.
    #[test]
    fn has_top_level_section_unlisted_top_level_row_is_false() {
        let mut state = ObjectState::default();
        state.sections.insert(5, section_row(5, 0));
        assert!(!has_top_level_section(&state));
    }

    #[test]
    fn has_top_level_section_listed_top_level_row_is_true() {
        let mut state = ObjectState::default();
        state.sections.insert(3, section_row(3, 3));
        state.sections.insert(5, section_row(5, 0));
        state.children.insert(0, vec![3, 5]);
        assert!(has_top_level_section(&state));
    }

    #[test]
    fn check_keeps_top_level_section_refuses_only_losing_the_last_one() {
        let empty = ObjectState::default();
        let rooted = engine_state(Some(kit_create_content("")));
        assert!(matches!(
            check_keeps_top_level_section(true, &empty),
            Err(DeltaError::RemovesLastTopLevelSection)
        ));
        assert!(
            check_keeps_top_level_section(true, &rooted).is_ok(),
            "kept a top-level section"
        );
        assert!(
            check_keeps_top_level_section(false, &empty).is_ok(),
            "a layer with no top-level section has none to lose"
        );
        assert!(
            check_keeps_top_level_section(false, &rooted).is_ok(),
            "gained one"
        );
    }
}
