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

//! b.8uk regression tests — UTF-8 hardening (H-ffi-1) and bridge-conversion
//! error-fidelity (H-ffi-2).
//!
//! Each test builds a verifier-valid OODF buffer via `build_delta`, patches in
//! a known invalid byte sequence at a specific schema-field offset, and asserts
//! `parse_delta` returns the typed error variant (not `OodfError::InternalError`
//! as it would have before b.8uk). The shared helper `patch_utf8_run` locates a
//! FlatBuffers string field by its current value and overwrites its first byte
//! with an invalid UTF-8 lead byte (0xFF).
//!
//! Tag-range coverage:
//! - `OodfError::InvalidUtf8(_)` with field-name substring ∈ {sourceFormat,
//!   CommandCreateSection.sectionName, CommandDiffChangeSection.sectionName,
//!   CommandMoveSection.newName, SetField.keyName}. `TimeSample` has no
//!   string fields so there is no UTF-8 test for it.
//! - `OodfError::UnknownCommandTagAt { index, tag }` with a bad tag placed
//!   at a non-zero index inside the `struct_commands` vector — the b.8uk
//!   commit replaced `UnknownCommandTag(u8)` with the indexed variant so the
//!   caller can identify *which* command was malformed.

use oodf_ffi::{
    bridge::ffi, build_delta, parse_delta, CommandCreateSection, CommandDiffChangeSection,
    CommandMoveSection, OodfError, OwnedDelta, SetField, StructCommand,
};

// ── Shared helper: find & patch the first byte of a FB string run ──────────

/// Locate `needle` in `bytes` and overwrite its first byte with `0xFF`.
///
/// `0xFF` is a guaranteed-invalid UTF-8 lead byte (no legal UTF-8 leading byte
/// falls in `0xFE..=0xFF`), so `is_valid_utf8` in `oodf.cpp` must reject the
/// resulting string and emit `OodfError::InvalidUtf8`. The needle is searched
/// as a contiguous byte run — FlatBuffers string payload bytes appear verbatim
/// in the wire buffer immediately after their 4-byte length prefix, so we pick
/// a needle that is long enough to be unique within the buffer.
///
/// Panics if `needle` does not appear exactly once (ambiguous patches can
/// accidentally corrupt a different field and invalidate the test).
fn patch_first_byte_to_invalid_utf8(bytes: &mut [u8], needle: &[u8]) {
    let positions: Vec<usize> = bytes
        .windows(needle.len())
        .enumerate()
        .filter_map(|(i, w)| if w == needle { Some(i) } else { None })
        .collect();
    assert_eq!(
        positions.len(),
        1,
        "needle {:?} must appear exactly once in the built buffer (got {} matches)",
        String::from_utf8_lossy(needle),
        positions.len()
    );
    bytes[positions[0]] = 0xFF;
}

// ── H-ffi-1: UTF-8 checks per schema field ─────────────────────────────────

#[test]
fn parse_delta_rejects_non_utf8_in_source_format_field() {
    let delta = OwnedDelta {
        source_format: "unique-source-format-marker".to_string(),
        ..Default::default()
    };
    let mut bytes = build_delta(&delta).expect("build_delta cannot fail");
    patch_first_byte_to_invalid_utf8(&mut bytes, b"unique-source-format-marker");

    let err = parse_delta(&bytes).expect_err("non-UTF-8 sourceFormat must be rejected");
    match err {
        OodfError::InvalidUtf8(msg) => {
            assert!(
                msg.contains("sourceFormat"),
                "InvalidUtf8 message must name the sourceFormat field, got {msg:?}"
            );
            assert!(
                msg.contains("offset"),
                "InvalidUtf8 message should carry the offset of the first bad byte, got {msg:?}"
            );
        }
        other => panic!("expected OodfError::InvalidUtf8, got {other:?}"),
    }
}

#[test]
fn parse_delta_rejects_non_utf8_in_create_section_section_name() {
    let delta = OwnedDelta {
        struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
            parent_id: 1,
            section_id: 2,
            section_name: "unique-create-section-name".to_string(),
            section_type: 3,
        })],
        ..Default::default()
    };
    let mut bytes = build_delta(&delta).expect("build_delta cannot fail");
    patch_first_byte_to_invalid_utf8(&mut bytes, b"unique-create-section-name");

    let err = parse_delta(&bytes).expect_err("non-UTF-8 sectionName must be rejected");
    match err {
        OodfError::InvalidUtf8(msg) => {
            assert!(
                msg.contains("CommandCreateSection.sectionName"),
                "InvalidUtf8 message must name CommandCreateSection.sectionName, got {msg:?}"
            );
        }
        other => panic!("expected OodfError::InvalidUtf8, got {other:?}"),
    }
}

#[test]
fn parse_delta_rejects_non_utf8_in_diff_change_section_section_name() {
    let delta = OwnedDelta {
        is_diff: true,
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 4,
            section_id: 5,
            section_name: "unique-diff-change-name".to_string(),
            section_type: 6,
            section_order: 7,
        })],
        ..Default::default()
    };
    let mut bytes = build_delta(&delta).expect("build_delta cannot fail");
    patch_first_byte_to_invalid_utf8(&mut bytes, b"unique-diff-change-name");

    let err = parse_delta(&bytes).expect_err("non-UTF-8 sectionName must be rejected");
    match err {
        OodfError::InvalidUtf8(msg) => {
            assert!(
                msg.contains("CommandDiffChangeSection.sectionName"),
                "InvalidUtf8 message must name CommandDiffChangeSection.sectionName, got {msg:?}"
            );
        }
        other => panic!("expected OodfError::InvalidUtf8, got {other:?}"),
    }
}

#[test]
fn parse_delta_rejects_non_utf8_in_move_section_new_name() {
    let delta = OwnedDelta {
        struct_commands: vec![StructCommand::MoveSection(CommandMoveSection {
            old_parent_id: 8,
            new_parent_id: 9,
            section_id: 10,
            new_name: "unique-move-new-name".to_string(),
        })],
        ..Default::default()
    };
    let mut bytes = build_delta(&delta).expect("build_delta cannot fail");
    patch_first_byte_to_invalid_utf8(&mut bytes, b"unique-move-new-name");

    let err = parse_delta(&bytes).expect_err("non-UTF-8 newName must be rejected");
    match err {
        OodfError::InvalidUtf8(msg) => {
            assert!(
                msg.contains("CommandMoveSection.newName"),
                "InvalidUtf8 message must name CommandMoveSection.newName, got {msg:?}"
            );
        }
        other => panic!("expected OodfError::InvalidUtf8, got {other:?}"),
    }
}

#[test]
fn parse_delta_rejects_non_utf8_in_set_field_key_name() {
    let delta = OwnedDelta {
        set_fields: vec![SetField {
            section_id: 11,
            key_name: "unique-set-field-key".to_string(),
            value_or_ext_hash: vec![0xDE, 0xAD],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 0,
        }],
        ..Default::default()
    };
    let mut bytes = build_delta(&delta).expect("build_delta cannot fail");
    patch_first_byte_to_invalid_utf8(&mut bytes, b"unique-set-field-key");

    let err = parse_delta(&bytes).expect_err("non-UTF-8 keyName must be rejected");
    match err {
        OodfError::InvalidUtf8(msg) => {
            assert!(
                msg.contains("SetField.keyName"),
                "InvalidUtf8 message must name SetField.keyName, got {msg:?}"
            );
        }
        other => panic!("expected OodfError::InvalidUtf8, got {other:?}"),
    }
}

#[test]
fn parse_delta_rejects_invalid_utf8_continuation_byte() {
    // Cover the non-0xFF branch of `is_valid_utf8`: a valid 2-byte lead
    // (0xC3) followed by a byte outside the 0x80..=0xBF continuation range
    // also has to be rejected. The patched sequence (0xC3 0x28) is the
    // canonical "not a valid continuation" example from the RFC 3629 tests.
    let delta = OwnedDelta {
        set_fields: vec![SetField {
            section_id: 99,
            key_name: "continuation-check-xyz".to_string(),
            value_or_ext_hash: vec![0u8],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 0,
        }],
        ..Default::default()
    };
    let mut bytes = build_delta(&delta).expect("build_delta cannot fail");
    // Find the key name run and patch its first two bytes.
    let needle = b"continuation-check-xyz";
    let pos = bytes
        .windows(needle.len())
        .position(|w| w == needle)
        .expect("keyName needle present in buffer");
    bytes[pos] = 0xC3;
    bytes[pos + 1] = 0x28;

    let err =
        parse_delta(&bytes).expect_err("bad continuation byte after 2-byte lead must be rejected");
    assert!(
        matches!(err, OodfError::InvalidUtf8(_)),
        "expected InvalidUtf8, got {err:?}"
    );
}

// ── H-ffi-2: UnknownCommandTagAt carries the vector index ──────────────────

#[test]
fn try_from_owned_delta_reports_bad_tag_at_non_zero_index() {
    // Build a bridge OwnedDelta whose struct_commands[0] is valid and
    // struct_commands[1] has an out-of-range tag. The bridge conversion must
    // surface the specific `index: 1` so callers can identify which command
    // was malformed rather than receiving an index-0 default.
    let good = ffi::StructCommand {
        tag: ffi::CommandTag::CreateSection,
        create_section: ffi::CommandCreateSection {
            parent_id: 1,
            section_id: 2,
            section_name: "ok".to_string(),
            section_type: 3,
        },
        ..Default::default()
    };
    let bad = ffi::StructCommand {
        tag: ffi::CommandTag { repr: 99 },
        ..Default::default()
    };
    let bridge = ffi::OwnedDelta {
        struct_commands: vec![good, bad],
        ..Default::default()
    };

    let err =
        oodf_ffi::types::OwnedDelta::try_from(bridge).expect_err("out-of-range tag must surface");
    assert!(
        matches!(err, OodfError::UnknownCommandTagAt { index: 1, tag: 99 }),
        "expected UnknownCommandTagAt {{ index: 1, tag: 99 }}, got {err:?}"
    );
}

#[test]
fn try_from_owned_delta_reports_bad_tag_at_third_position() {
    // Index-2 coverage on a 3-element vector: guards against off-by-one in
    // the `enumerate()` walk in `TryFrom<ffi::OwnedDelta>`. cxx shared structs
    // do not derive Clone, so build each element independently.
    let mk_ok = || ffi::StructCommand {
        tag: ffi::CommandTag::DiffDeleteSection,
        diff_delete_section: ffi::CommandDiffDeleteSection { section_id: 7 },
        ..Default::default()
    };
    let bridge = ffi::OwnedDelta {
        struct_commands: vec![
            mk_ok(),
            mk_ok(),
            ffi::StructCommand {
                tag: ffi::CommandTag { repr: 200 },
                ..Default::default()
            },
        ],
        ..Default::default()
    };

    let err =
        oodf_ffi::types::OwnedDelta::try_from(bridge).expect_err("out-of-range tag must surface");
    assert!(
        matches!(err, OodfError::UnknownCommandTagAt { index: 2, tag: 200 }),
        "expected UnknownCommandTagAt {{ index: 2, tag: 200 }}, got {err:?}"
    );
}

// ── Variant-name pin: rename would break this block at compile time ────────

#[test]
fn pin_error_variant_names_and_payload_shapes() {
    // If any of these variants gets renamed, this block fails to compile, so
    // the rename has to be a deliberate downstream-visible change rather than
    // an accidental API drift. Each arm also exercises the payload shape:
    //   InvalidUtf8 → single String
    //   UnknownCommandTagAt → named fields { index: usize, tag: u8 }
    //   UnknownCommandType → single u8
    let a = OodfError::InvalidUtf8("sectionName @ offset 4".to_string());
    let b = OodfError::UnknownCommandTagAt {
        index: 1usize,
        tag: 99u8,
    };
    let c = OodfError::UnknownCommandType(42u8);
    for err in [a, b, c] {
        let _ = format!("{err}");
    }
}

// ── Tag-range pre-FFI (UnknownCommandType) parse-time assertion ────────────
//
// The primary UnknownCommandType malformed-input regression lives in
// oodf-ffi/tests/malformed_input.rs (`out_of_range_struct_command_type_is_
// rejected`). That test pins `OodfError::UnknownCommandType(99)`, which is
// the same variant we want to assert here — keeping a single source of truth
// for the parse-time discriminant rejection. Re-asserting here would be a
// pure duplicate.
