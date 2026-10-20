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

// Shared OwnedDelta fixture helpers for integration tests.
//
// Post Task 4.4 migration, all Delta bytes are built via
// `oodf_ffi::build_delta` over an idiomatic `OwnedDelta`. These small
// composable helpers centralize the most common fixture shapes so test files
// don't repeat OwnedDelta literals.

#![allow(dead_code)]

use live_edit::domain::ompf::OmpfContainer;
use oodf_ffi::{CommandDiffChangeSection, OwnedDelta, SetField, StructCommand, TimeSample};

/// An empty OwnedDelta — `is_diff=false`, `base_version=0`, no commands or
/// fields. Equivalent to the old `build_delta_bytes(false, 0)` fixture.
pub fn empty_delta() -> OwnedDelta {
    OwnedDelta::default()
}

/// A reset delta (`is_diff=true`, `base_version=0`) with the given
/// `source_format`. Use when tests need the SDD §9.1.3 reset path.
pub fn reset_delta(source_format: &str) -> OwnedDelta {
    OwnedDelta {
        is_diff: true,
        base_version: 0,
        source_format: source_format.to_string(),
        ..Default::default()
    }
}

/// A diff delta that carries a single `CommandDiffChangeSection`. Use when a
/// test needs to create/update exactly one section via the diff command.
pub fn delta_with_diff_change_section(
    parent_id: u64,
    section_id: u64,
    section_name: &str,
    section_type: u8,
    section_order: u32,
) -> OwnedDelta {
    OwnedDelta {
        is_diff: true,
        base_version: 0,
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id,
            section_id,
            section_name: section_name.to_string(),
            section_type,
            section_order,
        })],
        ..Default::default()
    }
}

/// A delta carrying one inline SetField (value stored directly, no external
/// blob).
pub fn delta_with_set_field_inline(
    section_id: u64,
    key: &str,
    value: Vec<u8>,
    set_order: u64,
) -> OwnedDelta {
    OwnedDelta {
        set_fields: vec![SetField {
            section_id,
            key_name: key.to_string(),
            value_or_ext_hash: value,
            ext_value_size: 0,
            ext_value_index: 0,
            set_order,
        }],
        ..Default::default()
    }
}

/// A field-only delta setting `count` one-byte inline fields on `section_id`,
/// named `{name_prefix}{i:03}` for `i` in `0..count`, at `set_order` 1 to
/// `count`. With names the layer does not hold yet, it grows a compact
/// structural-cache entry by a row and an interned name per field (several
/// kilobytes at 64 fields), where one large inline value would grow it by
/// nothing: the compact entry holds no value.
pub fn delta_with_new_field_names(section_id: u64, name_prefix: &str, count: usize) -> OwnedDelta {
    OwnedDelta {
        set_fields: (0..count)
            .map(|i| SetField {
                section_id,
                key_name: format!("{name_prefix}{i:03}"),
                value_or_ext_hash: vec![0xA5],
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: i as u64 + 1,
            })
            .collect(),
        ..Default::default()
    }
}

/// A delta carrying one external-reference SetField: `value_or_ext_hash`
/// holds `hash` verbatim — 32 bytes for a valid delta; any other length is
/// refused `INVALID_ARGUMENT` (the G3 tests pass one on purpose) — and
/// `ext_value_size` / `ext_value_index` point to the out-of-band blob in the
/// OMPF container.
pub fn delta_with_set_field_external(
    section_id: u64,
    key: &str,
    hash: Vec<u8>,
    ext_value_size: u64,
    ext_value_index: u32,
    set_order: u64,
) -> OwnedDelta {
    OwnedDelta {
        set_fields: vec![SetField {
            section_id,
            key_name: key.to_string(),
            value_or_ext_hash: hash,
            ext_value_size,
            ext_value_index,
            set_order,
        }],
        ..Default::default()
    }
}

/// A delta carrying one inline time sample (value stored directly, no
/// external blob). The time-sample twin of [`delta_with_set_field_inline`];
/// time samples share `encode_value_fields` / `decode_field_value` with
/// fields, so a rewrite test needs both shapes on both row kinds.
pub fn delta_with_time_sample_inline(section_id: u64, time: f64, value: Vec<u8>) -> OwnedDelta {
    OwnedDelta {
        time_samples: vec![TimeSample {
            section_id,
            time,
            value_or_ext_hash: value,
            ext_value_size: 0,
            ext_value_index: 0,
        }],
        ..Default::default()
    }
}

/// A delta carrying one external-reference time sample: `value_or_ext_hash`
/// holds `hash` verbatim (32 bytes for a valid delta), and `ext_value_size` /
/// `ext_value_index` point to the out-of-band blob in the OMPF container. The
/// time-sample twin of
/// [`delta_with_set_field_external`].
pub fn delta_with_time_sample_external(
    section_id: u64,
    time: f64,
    hash: Vec<u8>,
    ext_value_size: u64,
    ext_value_index: u32,
) -> OwnedDelta {
    OwnedDelta {
        time_samples: vec![TimeSample {
            section_id,
            time,
            value_or_ext_hash: hash,
            ext_value_size,
            ext_value_index,
        }],
        ..Default::default()
    }
}

/// Build an OMPF container from an `OwnedDelta` plus any external-value
/// blobs. Replaces the old `build_delta_bytes` + `OmpfContainer::serialize`
/// pair that flatc-era tests used.
pub fn delta_to_ompf_bytes(delta: &OwnedDelta, external_values: &[&[u8]]) -> Vec<u8> {
    let delta_bytes =
        oodf_ffi::build_delta(delta).expect("test inputs are idiomatic, build cannot fail");
    OmpfContainer::serialize(&delta_bytes, external_values)
        .expect("test inputs never exceed u32::MAX OMPF parts")
}

/// A time no fixture uses for anything else, which a test building a delta
/// that must physically carry `-0.0` gives the sample first
/// ([`build_delta_with_negative_zero_times`]).
pub const NEGATIVE_ZERO_SENTINEL: f64 = 1_234.567_890_123;
/// A second such time, for a delta with two samples to patch.
pub const NEGATIVE_ZERO_SENTINEL_2: f64 = 9_876.543_210_987;

/// The positions at which `pattern` occurs in `bytes`, overlaps included.
fn occurrences(bytes: &[u8], pattern: &[u8; 8]) -> Vec<usize> {
    bytes
        .windows(8)
        .enumerate()
        .filter(|(_, w)| *w == pattern)
        .map(|(at, _)| at)
        .collect()
}

/// The OODF bytes of `delta` with each sample time in `sentinels` rewritten
/// to `-0.0` in place: the delta as a client that keeps signed-zero time
/// bits (one that forces FlatBuffers defaults) sends it.
///
/// `oodf_ffi::build_delta` cannot write `-0.0`: FlatBuffers drops a field
/// equal to its default, and `-0.0 == 0.0`, so a sample built at `-0.0`
/// arrives at `+0.0` and a test sending it would prove nothing. So the sample
/// is built at a sentinel time, which FlatBuffers stores, and its 8
/// little-endian bytes are overwritten with `-0.0`'s. Each sentinel must
/// occur exactly once in the built bytes, the `-0.0` pattern not at all
/// before patching and exactly once per sentinel after: a collision with any
/// other byte run fails the fixture rather than patching the wrong bytes.
///
/// Panics on a fixture that breaks either count.
pub fn build_delta_with_negative_zero_times(delta: &OwnedDelta, sentinels: &[f64]) -> Vec<u8> {
    let negative_zero = (-0.0f64).to_le_bytes();
    let mut bytes =
        oodf_ffi::build_delta(delta).expect("test inputs are idiomatic, build cannot fail");
    assert!(
        occurrences(&bytes, &negative_zero).is_empty(),
        "fixture: the built delta must not carry the -0.0 byte pattern before patching"
    );
    for &sentinel in sentinels {
        let at = occurrences(&bytes, &sentinel.to_le_bytes());
        assert_eq!(
            at.len(),
            1,
            "fixture: sentinel time {sentinel} must occur exactly once in the built delta"
        );
        bytes[at[0]..at[0] + 8].copy_from_slice(&negative_zero);
    }
    assert_eq!(
        occurrences(&bytes, &negative_zero).len(),
        sentinels.len(),
        "fixture: the patched delta carries -0.0 exactly once per sentinel"
    );
    bytes
}

/// [`build_delta_with_negative_zero_times`] in an OMPF container with
/// `external_values`, as [`delta_to_ompf_bytes`] builds one.
pub fn delta_to_ompf_bytes_with_negative_zero_times(
    delta: &OwnedDelta,
    sentinels: &[f64],
    external_values: &[&[u8]],
) -> Vec<u8> {
    let delta_bytes = build_delta_with_negative_zero_times(delta, sentinels);
    OmpfContainer::serialize(&delta_bytes, external_values)
        .expect("test inputs never exceed u32::MAX OMPF parts")
}

/// Whether `bytes` carry the little-endian `-0.0` pattern anywhere.
pub fn carries_negative_zero(bytes: &[u8]) -> bool {
    !occurrences(bytes, &(-0.0f64).to_le_bytes()).is_empty()
}

/// `delta`'s delta part as [`delta_to_ompf_bytes`] carries it in an OMPF, in
/// bytes: the built OODF buffer, which a structural cache repair counts for
/// each replayed record.
pub fn delta_part_len(delta: &OwnedDelta) -> u64 {
    oodf_ffi::build_delta(delta)
        .expect("test inputs are idiomatic, build cannot fail")
        .len() as u64
}
