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

//! Task 3.4.C — property-based coverage for `oodf-ffi`.
//!
//! Three properties:
//! 1. Round-trip: `parse_delta(build_delta(&d)) == d` for every `arb_owned_delta`.
//! 2. Fuzz safety: `parse_delta(arbitrary_bytes)` never panics.
//! 3. The entry-count probe on hostile headers: an over-limit buffer (built
//!    once, a few MB) with its header bytes mutated — the bytes the probe
//!    reads before the verifier — never crashes, is refused `TooManyTables`
//!    only with counts over the limit that the buffer can hold, or is
//!    refused by the verifier; a mutation that shortens a vector into a
//!    valid buffer parses only within the limit.
//!
//! Per CLAUDE.md: proptest MUST live in its own test binary, separate from
//! any `#[tokio::test]` multi-thread runtime (which deadlocks at process
//! exit when it shares a binary with proptest). This file is a pure sync
//! test binary — no tokio, no async.
//!
//! `PROPTEST_CASES` env var overrides the default case count (256). Case
//! persistence to `proptest-regressions/` is explicitly disabled via
//! `ProptestConfig::failure_persistence = None` so the test does not need a
//! writable project subdirectory at `cargo test` time (CI-friendly).

use oodf_ffi::{
    build_delta, parse_delta, CommandCreateSection, CommandDeleteSection, CommandDiffChangeSection,
    CommandDiffDeleteSection, CommandMoveSection, CommandReorderChildren, OodfError, OwnedDelta,
    SetField, StructCommand, TimeSample, MAX_DELTA_TABLES,
};
use proptest::prelude::*;

// ── Arbitrary generators ────────────────────────────────────────────────────

fn arb_command_create_section() -> impl Strategy<Value = CommandCreateSection> {
    (any::<u64>(), any::<u64>(), ".{0,32}", any::<u8>()).prop_map(
        |(parent_id, section_id, section_name, section_type)| CommandCreateSection {
            parent_id,
            section_id,
            section_name,
            section_type,
        },
    )
}

fn arb_command_delete_section() -> impl Strategy<Value = CommandDeleteSection> {
    (any::<u64>(), any::<u64>()).prop_map(|(parent_id, section_id)| CommandDeleteSection {
        parent_id,
        section_id,
    })
}

fn arb_command_move_section() -> impl Strategy<Value = CommandMoveSection> {
    (any::<u64>(), any::<u64>(), any::<u64>(), ".{0,32}").prop_map(
        |(old_parent_id, new_parent_id, section_id, new_name)| CommandMoveSection {
            old_parent_id,
            new_parent_id,
            section_id,
            new_name,
        },
    )
}

fn arb_command_reorder_children() -> impl Strategy<Value = CommandReorderChildren> {
    (
        any::<u64>(),
        any::<u8>(),
        prop::collection::vec(any::<u64>(), 0..6),
    )
        .prop_map(
            |(section_id, children_list_id, children_list)| CommandReorderChildren {
                section_id,
                children_list_id,
                children_list,
            },
        )
}

fn arb_command_diff_change_section() -> impl Strategy<Value = CommandDiffChangeSection> {
    (
        any::<u64>(),
        any::<u64>(),
        ".{0,32}",
        any::<u8>(),
        any::<u32>(),
    )
        .prop_map(
            |(parent_id, section_id, section_name, section_type, section_order)| {
                CommandDiffChangeSection {
                    parent_id,
                    section_id,
                    section_name,
                    section_type,
                    section_order,
                }
            },
        )
}

fn arb_command_diff_delete_section() -> impl Strategy<Value = CommandDiffDeleteSection> {
    any::<u64>().prop_map(|section_id| CommandDiffDeleteSection { section_id })
}

fn arb_struct_command() -> impl Strategy<Value = StructCommand> {
    prop_oneof![
        arb_command_create_section().prop_map(StructCommand::CreateSection),
        arb_command_delete_section().prop_map(StructCommand::DeleteSection),
        arb_command_move_section().prop_map(StructCommand::MoveSection),
        arb_command_reorder_children().prop_map(StructCommand::ReorderChildren),
        arb_command_diff_change_section().prop_map(StructCommand::DiffChangeSection),
        arb_command_diff_delete_section().prop_map(StructCommand::DiffDeleteSection),
    ]
}

fn arb_set_field() -> impl Strategy<Value = SetField> {
    // Mix inline (ext_value_size == 0, short value_or_ext_hash) and external
    // (ext_value_size > 0, 32-byte hash in value_or_ext_hash) cases. Using
    // `prop_oneof!` gives the two shapes roughly equal weight.
    prop_oneof![
        (
            any::<u64>(),
            ".{0,24}",
            prop::collection::vec(any::<u8>(), 0..32),
            any::<u64>(),
        )
            .prop_map(
                |(section_id, key_name, value_or_ext_hash, set_order)| SetField {
                    section_id,
                    key_name,
                    value_or_ext_hash,
                    ext_value_size: 0,
                    ext_value_index: 0,
                    set_order,
                }
            ),
        (
            any::<u64>(),
            ".{0,24}",
            prop::array::uniform32(any::<u8>()).prop_map(|a| a.to_vec()),
            1u64..=1 << 20,
            1u32..=64,
            any::<u64>(),
        )
            .prop_map(
                |(
                    section_id,
                    key_name,
                    value_or_ext_hash,
                    ext_value_size,
                    ext_value_index,
                    set_order,
                )| SetField {
                    section_id,
                    key_name,
                    value_or_ext_hash,
                    ext_value_size,
                    ext_value_index,
                    set_order,
                }
            ),
    ]
}

fn arb_time_sample() -> impl Strategy<Value = TimeSample> {
    // Restrict time to exactly representable f64 values (multiples of 0.25)
    // so semantic equality on round-trip is stable. Mix inline + external
    // value forms.
    (
        any::<u64>(),
        prop::sample::select(vec![
            -1024.0, -1.0, -0.25, 0.0, 0.25, 0.5, 1.0, 2.5, 100.0, 1024.0,
        ]),
        prop::collection::vec(any::<u8>(), 0..32),
        prop_oneof![Just((0u64, 0u32)), (1u64..=1 << 20, 1u32..=64)],
    )
        .prop_map(
            |(section_id, time, value_or_ext_hash, (ext_value_size, ext_value_index))| TimeSample {
                section_id,
                time,
                value_or_ext_hash,
                ext_value_size,
                ext_value_index,
            },
        )
}

fn arb_owned_delta() -> impl Strategy<Value = OwnedDelta> {
    (
        any::<bool>(),
        any::<u64>(),
        prop::collection::vec(arb_struct_command(), 0..=5),
        prop::collection::vec(arb_set_field(), 0..=5),
        prop::collection::vec(arb_time_sample(), 0..=3),
        ".{0,16}",
    )
        .prop_map(
            |(is_diff, base_version, struct_commands, set_fields, time_samples, source_format)| {
                OwnedDelta {
                    is_diff,
                    base_version,
                    struct_commands,
                    set_fields,
                    time_samples,
                    source_format,
                }
            },
        )
}

// ── Properties ──────────────────────────────────────────────────────────────

fn proptest_config() -> ProptestConfig {
    // Default 256 cases; honor PROPTEST_CASES env override. failure_persistence
    // is None so we don't require a writable `proptest-regressions/` dir in CI.
    let cases = std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(256);
    ProptestConfig {
        cases,
        failure_persistence: None,
        ..ProptestConfig::default()
    }
}

proptest! {
    #![proptest_config(proptest_config())]

    #[test]
    fn round_trip_preserves_owned_delta(delta in arb_owned_delta()) {
        let bytes = build_delta(&delta).expect("build_delta cannot fail on idiomatic input");
        let parsed = parse_delta(&bytes).expect("well-formed delta must round-trip parse");
        prop_assert_eq!(parsed, delta);
    }

    #[test]
    fn parse_delta_never_panics_on_arbitrary_bytes(
        bytes in prop::collection::vec(any::<u8>(), 0..=1024),
    ) {
        // Exercise the fuzz-safety contract: no panic for any input, only
        // Ok(_) or Err(_). We deliberately do NOT assert anything about the
        // returned value — a malformed-but-structurally-valid buffer could
        // parse successfully, and that's still fine.
        let _ = parse_delta(&bytes);
    }

    #[test]
    fn parse_delta_over_the_entry_limit_with_mutated_header_bytes_never_admits_or_miscounts(
        mutations in prop::collection::vec((any::<prop::sample::Index>(), any::<u8>()), 1..=8),
    ) {
        let fixture = over_limit_fixture();
        let mut bytes = fixture.bytes.clone();
        for (at, value) in &mutations {
            bytes[*at.get(&fixture.header_positions)] = *value;
        }
        let max_entries = MAX_DELTA_TABLES as u64 - 1;
        // Every count the probe reports was bounds-checked: an offset vector
        // spends 4 bytes per element, so no honest count exceeds this.
        let max_count = bytes.len() as u64 / 4;
        match parse_delta(&bytes) {
            Err(OodfError::TooManyTables {
                struct_commands,
                set_fields,
                time_samples,
                max_entries: limit,
            }) => {
                prop_assert_eq!(limit, max_entries);
                prop_assert!(
                    struct_commands + set_fields + time_samples > max_entries,
                    "a refusal under the limit: {} + {} + {}",
                    struct_commands, set_fields, time_samples
                );
                prop_assert!(
                    struct_commands <= max_count && set_fields <= max_count
                        && time_samples <= max_count,
                    "a count the buffer cannot hold — read from an unchecked header: \
                     {} / {} / {} in {} bytes",
                    struct_commands, set_fields, time_samples, bytes.len()
                );
            }
            // A header the probe cannot read falls through to the verifier (or,
            // for a mutated identifier, never gets past the magic check); a
            // header rewritten into another valid shape can reach the
            // accessors' own checks.
            Err(OodfError::VerifierFailed(_))
            | Err(OodfError::BadMagic)
            | Err(OodfError::UnknownCommandType(_))
            | Err(OodfError::InvalidUtf8(_)) => {}
            Err(other) => prop_assert!(false, "an unexpected refusal: {:?}", other),
            Ok(delta) => {
                // A mutation that shortened a vector can yield a delta the
                // verifier accepts; it must then be within the limit.
                let total = (delta.struct_commands.len() + delta.set_fields.len()
                    + delta.time_samples.len()) as u64;
                prop_assert!(total <= max_entries, "parsed {} entries", total);
            }
        }
    }
}

/// An over-limit delta (60,000 delete commands + 60,000 inline set fields,
/// each kind under any per-kind cap, the sum over the entry limit) and the
/// byte positions of its headers: the root offset, the file identifier, the
/// root table's vtable and inline fields, and the length word and first
/// element offset of each of the three vectors — every byte the entry-count
/// probe reads, and their neighbours.
struct OverLimitFixture {
    bytes: Vec<u8>,
    header_positions: Vec<usize>,
}

fn over_limit_fixture() -> &'static OverLimitFixture {
    static FIXTURE: std::sync::OnceLock<OverLimitFixture> = std::sync::OnceLock::new();
    FIXTURE.get_or_init(|| {
        let delta = OwnedDelta {
            struct_commands: (0..60_000u64)
                .map(|i| {
                    StructCommand::DeleteSection(CommandDeleteSection {
                        parent_id: 0,
                        section_id: i + 1,
                    })
                })
                .collect(),
            set_fields: (0..60_000)
                .map(|i| SetField {
                    section_id: 1,
                    key_name: format!("k{i}"),
                    value_or_ext_hash: vec![1],
                    ext_value_size: 0,
                    ext_value_index: 0,
                    set_order: 0,
                })
                .collect(),
            ..Default::default()
        };
        let bytes = build_delta(&delta).expect("build_delta cannot fail here");
        assert!(
            matches!(parse_delta(&bytes), Err(OodfError::TooManyTables { .. })),
            "the unmutated fixture must be refused by the probe"
        );
        let u32_at = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
        let root = u32_at(0);
        let back = i32::from_le_bytes(bytes[root..root + 4].try_into().unwrap());
        let vtable = usize::try_from(root as i64 - i64::from(back)).expect("vtable inside buffer");
        let vtable_len = usize::from(u16::from_le_bytes([bytes[vtable], bytes[vtable + 1]]));
        let table_len = usize::from(u16::from_le_bytes([bytes[vtable + 2], bytes[vtable + 3]]));
        let mut positions: Vec<usize> = (0..8).collect();
        positions.extend(vtable..vtable + vtable_len);
        positions.extend(root..root + table_len);
        // The vectors' vtable slots: structCommands_type, structCommands,
        // setFields, timeSamples (absent here, so skipped).
        for vt in [8usize, 10, 12, 14] {
            let field = usize::from(u16::from_le_bytes([
                bytes[vtable + vt],
                bytes[vtable + vt + 1],
            ]));
            if field == 0 {
                continue;
            }
            let slot = root + field;
            let length = slot + u32_at(slot);
            positions.extend(length..length + 8);
        }
        positions.sort_unstable();
        positions.dedup();
        positions.retain(|&p| p < bytes.len());
        OverLimitFixture {
            bytes,
            header_positions: positions,
        }
    })
}
