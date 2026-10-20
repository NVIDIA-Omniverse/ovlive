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

//! Task 3.4.A — positive-path round-trip coverage of every StructCommand
//! variant and of SetField / TimeSample through the full
//! idiomatic → bridge → C++ FlatBuffers → bridge → idiomatic pipeline.
//!
//! Each test constructs an `OwnedDelta`, calls `build_delta`, then `parse_delta`,
//! and asserts `PartialEq` equality on the idiomatic value. Byte-level
//! comparison is avoided because FlatBuffers offset ordering is not guaranteed
//! to be byte-stable across runs.

use oodf_ffi::{
    build_delta, parse_delta, CommandCreateSection, CommandDeleteSection, CommandDiffChangeSection,
    CommandDiffDeleteSection, CommandMoveSection, CommandReorderChildren, OwnedDelta, SetField,
    StructCommand, TimeSample,
};

// ── Tiny constructors for readability ───────────────────────────────────────

fn create_section() -> StructCommand {
    StructCommand::CreateSection(CommandCreateSection {
        parent_id: 1,
        section_id: 2,
        section_name: "sec-create".to_string(),
        section_type: 3,
    })
}

fn delete_section() -> StructCommand {
    StructCommand::DeleteSection(CommandDeleteSection {
        parent_id: 10,
        section_id: 11,
    })
}

fn move_section() -> StructCommand {
    StructCommand::MoveSection(CommandMoveSection {
        old_parent_id: 20,
        new_parent_id: 21,
        section_id: 22,
        new_name: "moved".to_string(),
    })
}

fn reorder_children() -> StructCommand {
    StructCommand::ReorderChildren(CommandReorderChildren {
        section_id: 30,
        children_list_id: 2,
        children_list: vec![31, 32, 33, 34],
    })
}

fn diff_change_section() -> StructCommand {
    StructCommand::DiffChangeSection(CommandDiffChangeSection {
        parent_id: 40,
        section_id: 41,
        section_name: "diff".to_string(),
        section_type: 5,
        section_order: 7,
    })
}

fn diff_delete_section() -> StructCommand {
    StructCommand::DiffDeleteSection(CommandDiffDeleteSection { section_id: 50 })
}

fn delta_with_single_command(cmd: StructCommand) -> OwnedDelta {
    OwnedDelta {
        is_diff: false,
        base_version: 0,
        struct_commands: vec![cmd],
        set_fields: vec![],
        time_samples: vec![],
        source_format: String::new(),
    }
}

fn round_trip(delta: &OwnedDelta) -> OwnedDelta {
    let bytes = build_delta(delta).expect("build_delta cannot fail on idiomatic input");
    parse_delta(&bytes).expect("round-trip parse must succeed on a well-formed delta")
}

// ── One test per StructCommand variant ──────────────────────────────────────

#[test]
fn round_trip_create_section_only() {
    let original = delta_with_single_command(create_section());
    assert_eq!(round_trip(&original), original);
}

#[test]
fn round_trip_delete_section_only() {
    let original = delta_with_single_command(delete_section());
    assert_eq!(round_trip(&original), original);
}

#[test]
fn round_trip_move_section_only() {
    let original = delta_with_single_command(move_section());
    assert_eq!(round_trip(&original), original);
}

#[test]
fn round_trip_reorder_children_only() {
    let original = delta_with_single_command(reorder_children());
    assert_eq!(round_trip(&original), original);
}

#[test]
fn round_trip_diff_change_section_only() {
    let original = delta_with_single_command(diff_change_section());
    assert_eq!(round_trip(&original), original);
}

#[test]
fn round_trip_diff_delete_section_only() {
    let original = delta_with_single_command(diff_delete_section());
    assert_eq!(round_trip(&original), original);
}

// Empty-delta round-trip is covered by the proptest in
// oodf-ffi/tests/proptests.rs (arb_owned_delta generates 0..=N commands,
// fields, and time-samples — the zero-of-everything case is sampled).

// ── Mixed: every variant once in one delta ──────────────────────────────────

#[test]
fn round_trip_mixed_all_variants() {
    let original = OwnedDelta {
        is_diff: true,
        base_version: 123,
        struct_commands: vec![
            create_section(),
            delete_section(),
            move_section(),
            reorder_children(),
            diff_change_section(),
            diff_delete_section(),
        ],
        set_fields: vec![],
        time_samples: vec![],
        source_format: "usda".to_string(),
    };
    assert_eq!(round_trip(&original), original);
}

// ── SetField-heavy: ~20 entries, mix inline + external ──────────────────────

#[test]
fn round_trip_set_field_heavy_inline_and_external() {
    let mut set_fields: Vec<SetField> = Vec::new();
    for i in 0..20u64 {
        let inline = i % 2 == 0;
        if inline {
            set_fields.push(SetField {
                section_id: 1000 + i,
                key_name: format!("key_inline_{i}"),
                value_or_ext_hash: vec![i as u8, (i + 1) as u8, (i + 2) as u8, (i + 3) as u8],
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: i,
            });
        } else {
            // External reference: value_or_ext_hash is a 32-byte content hash,
            // ext_value_size > 0, ext_value_index points into the OMPF parts
            // table (any non-zero index is fine for a round-trip test).
            let mut hash = vec![0u8; 32];
            for (j, slot) in hash.iter_mut().enumerate() {
                *slot = ((i as usize + j) % 256) as u8;
            }
            set_fields.push(SetField {
                section_id: 2000 + i,
                key_name: format!("key_ext_{i}"),
                value_or_ext_hash: hash,
                ext_value_size: 4096 + i,
                ext_value_index: (i + 1) as u32,
                set_order: i + 100,
            });
        }
    }

    let original = OwnedDelta {
        is_diff: false,
        base_version: 9,
        struct_commands: vec![],
        set_fields,
        time_samples: vec![],
        source_format: String::new(),
    };
    assert_eq!(round_trip(&original), original);
}

// ── TimeSample-heavy: ~20 entries with distinct (section_id, time) ─────────

#[test]
fn round_trip_time_sample_heavy_distinct_keys() {
    let mut time_samples: Vec<TimeSample> = Vec::new();
    for i in 0..20u64 {
        // Use multiples of 0.5 to keep values exactly representable in f64
        // (avoids any spurious floating-point inequality on round-trip).
        let time = (i as f64) * 0.5;
        time_samples.push(TimeSample {
            section_id: 5000 + i,
            time,
            value_or_ext_hash: vec![i as u8; 8],
            ext_value_size: if i % 3 == 0 { 0 } else { 256 + i },
            ext_value_index: if i % 3 == 0 { 0 } else { (i + 1) as u32 },
        });
    }

    let original = OwnedDelta {
        is_diff: true,
        base_version: 17,
        struct_commands: vec![],
        set_fields: vec![],
        time_samples,
        source_format: "usdc".to_string(),
    };
    assert_eq!(round_trip(&original), original);
}
