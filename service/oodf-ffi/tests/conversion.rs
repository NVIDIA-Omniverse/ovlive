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

//! Conversion-layer integration tests.
//!
//! Exercises the `From`/`TryFrom` impls in `oodf_ffi::convert` that translate
//! between the cxx bridge types (`oodf_ffi::bridge::ffi::*`) and the idiomatic
//! Rust mirrors (`oodf_ffi::types::*`). No C++ is invoked — these are pure
//! Rust-side round-trip and tag-handling tests.
//!
//! The bridge and convert modules are `pub` + `#[doc(hidden)]` specifically
//! so this integration test can reach bridge types; production callers should
//! still go through `oodf_ffi::types` and the public API.

use oodf_ffi::bridge::ffi;
use oodf_ffi::error::OodfError;
use oodf_ffi::types::{
    CommandCreateSection, CommandDeleteSection, CommandDiffChangeSection, CommandDiffDeleteSection,
    CommandMoveSection, CommandReorderChildren, OwnedDelta, SetField, StructCommand, TimeSample,
};

// ── Fixture helpers ─────────────────────────────────────────────────────────

fn sample_create_section() -> CommandCreateSection {
    CommandCreateSection {
        parent_id: 1,
        section_id: 2,
        section_name: "sec".to_string(),
        section_type: 3,
    }
}

fn sample_delete_section() -> CommandDeleteSection {
    CommandDeleteSection {
        parent_id: 10,
        section_id: 11,
    }
}

fn sample_move_section() -> CommandMoveSection {
    CommandMoveSection {
        old_parent_id: 20,
        new_parent_id: 21,
        section_id: 22,
        new_name: "renamed".to_string(),
    }
}

fn sample_reorder_children() -> CommandReorderChildren {
    CommandReorderChildren {
        section_id: 30,
        children_list_id: 2,
        children_list: vec![31, 32, 33],
    }
}

fn sample_diff_change_section() -> CommandDiffChangeSection {
    CommandDiffChangeSection {
        parent_id: 40,
        section_id: 41,
        section_name: "diff".to_string(),
        section_type: 5,
        section_order: 7,
    }
}

fn sample_diff_delete_section() -> CommandDiffDeleteSection {
    CommandDiffDeleteSection { section_id: 50 }
}

// ── CreateSection ───────────────────────────────────────────────────────────

#[test]
fn from_idiomatic_create_section_sets_correct_tag_and_payload() {
    let payload = sample_create_section();
    let bridge: ffi::StructCommand = StructCommand::CreateSection(payload.clone()).into();

    assert_eq!(bridge.tag.repr, ffi::CommandTag::CreateSection.repr);
    assert_eq!(bridge.create_section.parent_id, payload.parent_id);
    assert_eq!(bridge.create_section.section_id, payload.section_id);
    assert_eq!(bridge.create_section.section_name, payload.section_name);
    assert_eq!(bridge.create_section.section_type, payload.section_type);

    // Inactive payload fields must be default-constructed.
    assert_eq!(bridge.delete_section.section_id, 0);
    assert_eq!(bridge.move_section.section_id, 0);
    assert_eq!(bridge.reorder_children.section_id, 0);
    assert!(bridge.reorder_children.children_list.is_empty());
    assert_eq!(bridge.diff_change_section.section_id, 0);
    assert_eq!(bridge.diff_delete_section.section_id, 0);
}

#[test]
fn try_from_bridge_create_section_returns_correct_variant() {
    let payload = sample_create_section();
    let bridge = ffi::StructCommand {
        tag: ffi::CommandTag::CreateSection,
        create_section: ffi::CommandCreateSection {
            parent_id: payload.parent_id,
            section_id: payload.section_id,
            section_name: payload.section_name.clone(),
            section_type: payload.section_type,
        },
        ..Default::default()
    };

    let idiomatic = StructCommand::try_from(bridge).expect("valid tag converts");
    assert_eq!(idiomatic, StructCommand::CreateSection(payload));
}

#[test]
fn round_trip_create_section_preserves_data() {
    let original = StructCommand::CreateSection(sample_create_section());
    let bridge: ffi::StructCommand = original.clone().into();
    let back = StructCommand::try_from(bridge).expect("round-trip succeeds");
    assert_eq!(original, back);
}

// ── DeleteSection ───────────────────────────────────────────────────────────

#[test]
fn from_idiomatic_delete_section_sets_correct_tag_and_payload() {
    let payload = sample_delete_section();
    let bridge: ffi::StructCommand = StructCommand::DeleteSection(payload.clone()).into();

    assert_eq!(bridge.tag.repr, ffi::CommandTag::DeleteSection.repr);
    assert_eq!(bridge.delete_section.parent_id, payload.parent_id);
    assert_eq!(bridge.delete_section.section_id, payload.section_id);

    assert_eq!(bridge.create_section.section_id, 0);
    assert!(bridge.create_section.section_name.is_empty());
    assert_eq!(bridge.move_section.section_id, 0);
    assert!(bridge.reorder_children.children_list.is_empty());
    assert_eq!(bridge.diff_change_section.section_id, 0);
    assert_eq!(bridge.diff_delete_section.section_id, 0);
}

#[test]
fn try_from_bridge_delete_section_returns_correct_variant() {
    let payload = sample_delete_section();
    let bridge = ffi::StructCommand {
        tag: ffi::CommandTag::DeleteSection,
        delete_section: ffi::CommandDeleteSection {
            parent_id: payload.parent_id,
            section_id: payload.section_id,
        },
        ..Default::default()
    };

    let idiomatic = StructCommand::try_from(bridge).expect("valid tag converts");
    assert_eq!(idiomatic, StructCommand::DeleteSection(payload));
}

#[test]
fn round_trip_delete_section_preserves_data() {
    let original = StructCommand::DeleteSection(sample_delete_section());
    let bridge: ffi::StructCommand = original.clone().into();
    let back = StructCommand::try_from(bridge).expect("round-trip succeeds");
    assert_eq!(original, back);
}

// ── MoveSection ─────────────────────────────────────────────────────────────

#[test]
fn from_idiomatic_move_section_sets_correct_tag_and_payload() {
    let payload = sample_move_section();
    let bridge: ffi::StructCommand = StructCommand::MoveSection(payload.clone()).into();

    assert_eq!(bridge.tag.repr, ffi::CommandTag::MoveSection.repr);
    assert_eq!(bridge.move_section.old_parent_id, payload.old_parent_id);
    assert_eq!(bridge.move_section.new_parent_id, payload.new_parent_id);
    assert_eq!(bridge.move_section.section_id, payload.section_id);
    assert_eq!(bridge.move_section.new_name, payload.new_name);

    assert_eq!(bridge.create_section.section_id, 0);
    assert_eq!(bridge.delete_section.section_id, 0);
    assert!(bridge.reorder_children.children_list.is_empty());
    assert_eq!(bridge.diff_change_section.section_id, 0);
    assert_eq!(bridge.diff_delete_section.section_id, 0);
}

#[test]
fn try_from_bridge_move_section_returns_correct_variant() {
    let payload = sample_move_section();
    let bridge = ffi::StructCommand {
        tag: ffi::CommandTag::MoveSection,
        move_section: ffi::CommandMoveSection {
            old_parent_id: payload.old_parent_id,
            new_parent_id: payload.new_parent_id,
            section_id: payload.section_id,
            new_name: payload.new_name.clone(),
        },
        ..Default::default()
    };

    let idiomatic = StructCommand::try_from(bridge).expect("valid tag converts");
    assert_eq!(idiomatic, StructCommand::MoveSection(payload));
}

#[test]
fn round_trip_move_section_preserves_data() {
    let original = StructCommand::MoveSection(sample_move_section());
    let bridge: ffi::StructCommand = original.clone().into();
    let back = StructCommand::try_from(bridge).expect("round-trip succeeds");
    assert_eq!(original, back);
}

// ── ReorderChildren ─────────────────────────────────────────────────────────

#[test]
fn from_idiomatic_reorder_children_sets_correct_tag_and_payload() {
    let payload = sample_reorder_children();
    let bridge: ffi::StructCommand = StructCommand::ReorderChildren(payload.clone()).into();

    assert_eq!(bridge.tag.repr, ffi::CommandTag::ReorderChildren.repr);
    assert_eq!(bridge.reorder_children.section_id, payload.section_id);
    assert_eq!(
        bridge.reorder_children.children_list_id,
        payload.children_list_id
    );
    assert_eq!(bridge.reorder_children.children_list, payload.children_list);

    assert_eq!(bridge.create_section.section_id, 0);
    assert_eq!(bridge.delete_section.section_id, 0);
    assert_eq!(bridge.move_section.section_id, 0);
    assert_eq!(bridge.diff_change_section.section_id, 0);
    assert_eq!(bridge.diff_delete_section.section_id, 0);
}

#[test]
fn try_from_bridge_reorder_children_returns_correct_variant() {
    let payload = sample_reorder_children();
    let bridge = ffi::StructCommand {
        tag: ffi::CommandTag::ReorderChildren,
        reorder_children: ffi::CommandReorderChildren {
            section_id: payload.section_id,
            children_list_id: payload.children_list_id,
            children_list: payload.children_list.clone(),
        },
        ..Default::default()
    };

    let idiomatic = StructCommand::try_from(bridge).expect("valid tag converts");
    assert_eq!(idiomatic, StructCommand::ReorderChildren(payload));
}

#[test]
fn round_trip_reorder_children_preserves_data() {
    let original = StructCommand::ReorderChildren(sample_reorder_children());
    let bridge: ffi::StructCommand = original.clone().into();
    let back = StructCommand::try_from(bridge).expect("round-trip succeeds");
    assert_eq!(original, back);
}

// ── DiffChangeSection ───────────────────────────────────────────────────────

#[test]
fn from_idiomatic_diff_change_section_sets_correct_tag_and_payload() {
    let payload = sample_diff_change_section();
    let bridge: ffi::StructCommand = StructCommand::DiffChangeSection(payload.clone()).into();

    assert_eq!(bridge.tag.repr, ffi::CommandTag::DiffChangeSection.repr);
    assert_eq!(bridge.diff_change_section.parent_id, payload.parent_id);
    assert_eq!(bridge.diff_change_section.section_id, payload.section_id);
    assert_eq!(
        bridge.diff_change_section.section_name,
        payload.section_name
    );
    assert_eq!(
        bridge.diff_change_section.section_type,
        payload.section_type
    );
    assert_eq!(
        bridge.diff_change_section.section_order,
        payload.section_order
    );

    assert_eq!(bridge.create_section.section_id, 0);
    assert_eq!(bridge.delete_section.section_id, 0);
    assert_eq!(bridge.move_section.section_id, 0);
    assert!(bridge.reorder_children.children_list.is_empty());
    assert_eq!(bridge.diff_delete_section.section_id, 0);
}

#[test]
fn try_from_bridge_diff_change_section_returns_correct_variant() {
    let payload = sample_diff_change_section();
    let bridge = ffi::StructCommand {
        tag: ffi::CommandTag::DiffChangeSection,
        diff_change_section: ffi::CommandDiffChangeSection {
            parent_id: payload.parent_id,
            section_id: payload.section_id,
            section_name: payload.section_name.clone(),
            section_type: payload.section_type,
            section_order: payload.section_order,
        },
        ..Default::default()
    };

    let idiomatic = StructCommand::try_from(bridge).expect("valid tag converts");
    assert_eq!(idiomatic, StructCommand::DiffChangeSection(payload));
}

#[test]
fn round_trip_diff_change_section_preserves_data() {
    let original = StructCommand::DiffChangeSection(sample_diff_change_section());
    let bridge: ffi::StructCommand = original.clone().into();
    let back = StructCommand::try_from(bridge).expect("round-trip succeeds");
    assert_eq!(original, back);
}

// ── DiffDeleteSection ───────────────────────────────────────────────────────

#[test]
fn from_idiomatic_diff_delete_section_sets_correct_tag_and_payload() {
    let payload = sample_diff_delete_section();
    let bridge: ffi::StructCommand = StructCommand::DiffDeleteSection(payload.clone()).into();

    assert_eq!(bridge.tag.repr, ffi::CommandTag::DiffDeleteSection.repr);
    assert_eq!(bridge.diff_delete_section.section_id, payload.section_id);

    assert_eq!(bridge.create_section.section_id, 0);
    assert_eq!(bridge.delete_section.section_id, 0);
    assert_eq!(bridge.move_section.section_id, 0);
    assert!(bridge.reorder_children.children_list.is_empty());
    assert_eq!(bridge.diff_change_section.section_id, 0);
}

#[test]
fn try_from_bridge_diff_delete_section_returns_correct_variant() {
    let payload = sample_diff_delete_section();
    let bridge = ffi::StructCommand {
        tag: ffi::CommandTag::DiffDeleteSection,
        diff_delete_section: ffi::CommandDiffDeleteSection {
            section_id: payload.section_id,
        },
        ..Default::default()
    };

    let idiomatic = StructCommand::try_from(bridge).expect("valid tag converts");
    assert_eq!(idiomatic, StructCommand::DiffDeleteSection(payload));
}

#[test]
fn round_trip_diff_delete_section_preserves_data() {
    let original = StructCommand::DiffDeleteSection(sample_diff_delete_section());
    let bridge: ffi::StructCommand = original.clone().into();
    let back = StructCommand::try_from(bridge).expect("round-trip succeeds");
    assert_eq!(original, back);
}

// ── Invalid / missing tag handling ──────────────────────────────────────────

#[test]
fn try_from_invalid_tag_returns_err() {
    // Construct a bridge StructCommand with an out-of-range tag byte. cxx
    // shared enums are `#[repr(transparent)] struct { repr: u8 }`, so any u8
    // is syntactically representable — the conversion layer must surface this
    // as `UnknownCommandTagAt { index: 0, tag: 99 }` rather than panicking.
    let bridge = ffi::StructCommand {
        tag: ffi::CommandTag { repr: 99 },
        ..Default::default()
    };

    let err = StructCommand::try_from(bridge).expect_err("invalid tag must error");
    assert!(matches!(
        err,
        OodfError::UnknownCommandTagAt { index: 0, tag: 99 }
    ));
}

#[test]
fn try_from_tag_none_returns_err() {
    // NONE is the default and carries no payload — treat it the same as any
    // other unknown tag so missing/unset tags never silently succeed.
    let bridge = ffi::StructCommand {
        tag: ffi::CommandTag::NONE,
        ..Default::default()
    };

    let err = StructCommand::try_from(bridge).expect_err("NONE tag must error");
    assert!(matches!(
        err,
        OodfError::UnknownCommandTagAt { index: 0, tag: 0 }
    ));
}

// ── SetField / TimeSample ───────────────────────────────────────────────────

#[test]
fn set_field_round_trip_inline() {
    // Inline case: value_or_ext_hash carries the value directly, ext_value_*
    // fields are zero.
    let original = SetField {
        section_id: 100,
        key_name: "inline_key".to_string(),
        value_or_ext_hash: vec![0xde, 0xad, 0xbe, 0xef],
        ext_value_size: 0,
        ext_value_index: 0,
        set_order: 7,
    };

    let bridge: ffi::SetField = original.clone().into();
    let back: SetField = bridge.into();
    assert_eq!(original, back);
}

#[test]
fn set_field_round_trip_external() {
    // External case: value_or_ext_hash is a hash reference and ext_value_size
    // / ext_value_index are populated to point at the out-of-band blob.
    let original = SetField {
        section_id: 200,
        key_name: "external_key".to_string(),
        value_or_ext_hash: vec![0x01, 0x02, 0x03, 0x04, 0x05],
        ext_value_size: 1024,
        ext_value_index: 5,
        set_order: 42,
    };

    let bridge: ffi::SetField = original.clone().into();
    let back: SetField = bridge.into();
    assert_eq!(original, back);
}

#[test]
fn time_sample_round_trip_exact_f64() {
    // 1.5 is exactly representable in IEEE 754 binary64, so assert_eq on the
    // round-trip value is safe (no epsilon needed).
    let original = TimeSample {
        section_id: 300,
        time: 1.5,
        value_or_ext_hash: vec![9, 8, 7, 6],
        ext_value_size: 4,
        ext_value_index: 0,
    };

    let bridge: ffi::TimeSample = original.clone().into();
    let back: TimeSample = bridge.into();
    assert_eq!(original, back);
}

/// The bridge → owned conversion of a time sample whose `time` has `bits`:
/// the owned time's bits.
fn converted_time_bits(bits: u64) -> u64 {
    let bridge = ffi::TimeSample {
        section_id: 1,
        time: f64::from_bits(bits),
        value_or_ext_hash: vec![1],
        ext_value_size: 0,
        ext_value_index: 0,
    };
    let owned: TimeSample = bridge.into();
    owned.time.to_bits()
}

/// b.8jc: the parse direction canonicalizes a time code. `-0.0` converts to
/// `+0.0` (bits 0), so a parsed delta never carries `-0.0`; `+0.0` and an
/// ordinary time keep their bits; and a NaN keeps its exact payload — a
/// signalling NaN included, which `t + 0.0` would have quieted — for
/// validation to refuse as the client sent it.
#[test]
fn time_sample_from_bridge_canonicalizes_negative_zero_and_keeps_every_other_bit_pattern() {
    assert_eq!(converted_time_bits((-0.0f64).to_bits()), 0, "-0.0 → +0.0");
    assert_eq!(converted_time_bits(0.0f64.to_bits()), 0, "+0.0 unchanged");
    assert_eq!(
        converted_time_bits(1.5f64.to_bits()),
        1.5f64.to_bits(),
        "an ordinary time unchanged"
    );
    for nan in [
        0x7FF0_0000_0000_0001_u64,
        0x7FF8_0000_0000_0001,
        0xFFF4_0000_0000_00AB,
    ] {
        assert!(
            f64::from_bits(nan).is_nan(),
            "fixture: {nan:#018X} is a NaN"
        );
        assert_eq!(
            converted_time_bits(nan),
            nan,
            "the NaN payload {nan:#018X} must pass through bit for bit"
        );
    }
    for infinity in [f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(converted_time_bits(infinity.to_bits()), infinity.to_bits());
    }
}

/// The build direction writes the time as given: canonicalization belongs to
/// the parse direction alone.
#[test]
fn time_sample_into_bridge_writes_negative_zero_as_given() {
    let owned = TimeSample {
        section_id: 1,
        time: -0.0,
        value_or_ext_hash: vec![1],
        ext_value_size: 0,
        ext_value_index: 0,
    };
    let bridge: ffi::TimeSample = owned.into();
    assert_eq!(bridge.time.to_bits(), (-0.0f64).to_bits());
}

// ── Full OwnedDelta round-trip ──────────────────────────────────────────────

#[test]
fn owned_delta_full_round_trip() {
    // Build an OwnedDelta exercising every StructCommand variant plus
    // SetField (inline + external) and TimeSample. Convert through the bridge
    // and back, then assert structural equality.
    let original = OwnedDelta {
        is_diff: true,
        base_version: 42,
        struct_commands: vec![
            StructCommand::CreateSection(sample_create_section()),
            StructCommand::DeleteSection(sample_delete_section()),
            StructCommand::MoveSection(sample_move_section()),
            StructCommand::ReorderChildren(sample_reorder_children()),
            StructCommand::DiffChangeSection(sample_diff_change_section()),
            StructCommand::DiffDeleteSection(sample_diff_delete_section()),
        ],
        set_fields: vec![
            SetField {
                section_id: 1,
                key_name: "inline".to_string(),
                value_or_ext_hash: vec![0xaa, 0xbb],
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 1,
            },
            SetField {
                section_id: 2,
                key_name: "external".to_string(),
                value_or_ext_hash: vec![0xcc, 0xdd, 0xee],
                ext_value_size: 2048,
                ext_value_index: 11,
                set_order: 2,
            },
        ],
        time_samples: vec![TimeSample {
            section_id: 3,
            time: 2.25,
            value_or_ext_hash: vec![1, 2, 3, 4],
            ext_value_size: 4,
            ext_value_index: 0,
        }],
        source_format: "usd".to_string(),
    };

    let bridge: ffi::OwnedDelta = (&original).into();
    let back = OwnedDelta::try_from(bridge).expect("full round-trip must succeed");
    assert_eq!(original, back);
}
