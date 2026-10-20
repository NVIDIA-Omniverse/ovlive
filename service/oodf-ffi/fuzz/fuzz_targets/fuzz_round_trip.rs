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

//! b.vvs item 10 — structure-aware round-trip fuzz target.
//!
//! Generates an arbitrary `OwnedDelta` from libfuzzer's mutator-driven byte
//! stream and asserts `parse_delta(build_delta(x))` is byte-identical and
//! structurally equal to `x`. Catches:
//!
//!  * Build → parse asymmetries (a `build_delta`-emitted field that
//!    `parse_delta` cannot read back).
//!  * `OwnedDelta` round-trip drift introduced by future schema additions
//!    where one side is updated but the other is not.
//!  * Panics in either `build_delta` or `parse_delta` on adversarial-but-
//!    type-system-legal inputs (e.g. extreme sizes, NaN time samples,
//!    pathological vector lengths).
//!
//! Seed the corpus with the `tests/fixtures/oodf/*.oodf` files for
//! coverage of the realistic-shape branches:
//!
//! ```sh
//! cd oodf-ffi
//! mkdir -p corpus/fuzz_round_trip
//! cp ../tests/fixtures/oodf/*.oodf corpus/fuzz_round_trip/
//! cargo +nightly fuzz run fuzz_round_trip -- -max_total_time=60
//! ```

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use oodf_ffi::{
    build_delta, parse_delta, CommandCreateSection, CommandDeleteSection,
    CommandDiffChangeSection, CommandDiffDeleteSection, CommandMoveSection,
    CommandReorderChildren, OwnedDelta, SetField, StructCommand, TimeSample,
};

// Defensive caps to keep individual fuzz cases under the C++-side
// `kMaxCommands = 1_000_000` ceiling — without them every iteration could
// burn seconds allocating a million-element vector and the fuzzer would
// spend its time-budget on a single uninteresting input.
const MAX_VEC_LEN: usize = 64;
const MAX_BYTES_LEN: usize = 256;
const MAX_STRING_LEN: usize = 64;

fn ascii_string(u: &mut Unstructured<'_>) -> arbitrary::Result<String> {
    // Restrict to ASCII so we never run afoul of the C++-side InvalidUtf8
    // check (which is exercised independently by the existing
    // utf8_and_tag_regressions integration test). Keeps the fuzzer focused
    // on round-trip drift rather than UTF-8 rejection paths.
    let len = u.int_in_range(0..=MAX_STRING_LEN)?;
    let mut s = String::with_capacity(len);
    for _ in 0..len {
        let b = u.int_in_range(0x20u8..=0x7E)?;
        s.push(b as char);
    }
    Ok(s)
}

fn byte_vec(u: &mut Unstructured<'_>) -> arbitrary::Result<Vec<u8>> {
    let len = u.int_in_range(0..=MAX_BYTES_LEN)?;
    let mut v = Vec::with_capacity(len);
    for _ in 0..len {
        v.push(u.arbitrary::<u8>()?);
    }
    Ok(v)
}

fn u64_vec(u: &mut Unstructured<'_>) -> arbitrary::Result<Vec<u64>> {
    let len = u.int_in_range(0..=MAX_VEC_LEN)?;
    let mut v = Vec::with_capacity(len);
    for _ in 0..len {
        v.push(u.arbitrary::<u64>()?);
    }
    Ok(v)
}

fn arbitrary_struct_command(u: &mut Unstructured<'_>) -> arbitrary::Result<StructCommand> {
    match u.int_in_range(0u8..=5)? {
        0 => Ok(StructCommand::CreateSection(CommandCreateSection {
            parent_id: u.arbitrary()?,
            section_id: u.arbitrary()?,
            section_name: ascii_string(u)?,
            section_type: u.arbitrary()?,
        })),
        1 => Ok(StructCommand::DeleteSection(CommandDeleteSection {
            parent_id: u.arbitrary()?,
            section_id: u.arbitrary()?,
        })),
        2 => Ok(StructCommand::MoveSection(CommandMoveSection {
            old_parent_id: u.arbitrary()?,
            new_parent_id: u.arbitrary()?,
            section_id: u.arbitrary()?,
            new_name: ascii_string(u)?,
        })),
        3 => Ok(StructCommand::ReorderChildren(CommandReorderChildren {
            section_id: u.arbitrary()?,
            children_list_id: u.arbitrary()?,
            children_list: u64_vec(u)?,
        })),
        4 => Ok(StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: u.arbitrary()?,
            section_id: u.arbitrary()?,
            section_name: ascii_string(u)?,
            section_type: u.arbitrary()?,
            section_order: u.arbitrary()?,
        })),
        _ => Ok(StructCommand::DiffDeleteSection(CommandDiffDeleteSection {
            section_id: u.arbitrary()?,
        })),
    }
}

fn arbitrary_owned_delta(u: &mut Unstructured<'_>) -> arbitrary::Result<OwnedDelta> {
    let cmd_count = u.int_in_range(0..=MAX_VEC_LEN)?;
    let mut struct_commands = Vec::with_capacity(cmd_count);
    for _ in 0..cmd_count {
        struct_commands.push(arbitrary_struct_command(u)?);
    }

    let field_count = u.int_in_range(0..=MAX_VEC_LEN)?;
    let mut set_fields = Vec::with_capacity(field_count);
    for _ in 0..field_count {
        set_fields.push(SetField {
            section_id: u.arbitrary()?,
            key_name: ascii_string(u)?,
            value_or_ext_hash: byte_vec(u)?,
            ext_value_size: u.arbitrary()?,
            ext_value_index: u.arbitrary()?,
            set_order: u.arbitrary()?,
        });
    }

    let sample_count = u.int_in_range(0..=MAX_VEC_LEN)?;
    let mut time_samples = Vec::with_capacity(sample_count);
    for _ in 0..sample_count {
        time_samples.push(TimeSample {
            section_id: u.arbitrary()?,
            // Drive `time` from raw bits to cover NaN / ±inf / subnormals.
            time: f64::from_bits(u.arbitrary()?),
            value_or_ext_hash: byte_vec(u)?,
            ext_value_size: u.arbitrary()?,
            ext_value_index: u.arbitrary()?,
        });
    }

    Ok(OwnedDelta {
        is_diff: u.arbitrary()?,
        base_version: u.arbitrary()?,
        struct_commands,
        set_fields,
        time_samples,
        source_format: ascii_string(u)?,
    })
}

fuzz_target!(|data: &[u8]| {
    let mut u = Unstructured::new(data);
    let delta = match arbitrary_owned_delta(&mut u) {
        Ok(d) => d,
        Err(_) => return, // not enough entropy in this case — let libfuzzer move on
    };

    // build_delta on idiomatic Rust types should never fail in practice.
    // If it does, surface as a fuzzer crash so we can investigate.
    let bytes = build_delta(&delta).expect("build_delta of arbitrary idiomatic OwnedDelta");

    // parse_delta must accept what build_delta produces.
    let parsed = parse_delta(&bytes).expect("round-trip parse_delta failed for own-built bytes");

    // Structural equality. Note: TimeSample.time is f64 which is `PartialEq`
    // but not `Eq` because NaN != NaN. The delta we built may contain NaN
    // samples; if so, byte-equality is still expected so we compare via
    // bit-pattern equality on each sample.
    assert_eq!(delta.is_diff, parsed.is_diff, "is_diff mismatch");
    assert_eq!(delta.base_version, parsed.base_version, "base_version mismatch");
    assert_eq!(delta.source_format, parsed.source_format, "source_format mismatch");
    assert_eq!(
        delta.struct_commands, parsed.struct_commands,
        "struct_commands mismatch"
    );
    assert_eq!(delta.set_fields, parsed.set_fields, "set_fields mismatch");
    assert_eq!(
        delta.time_samples.len(),
        parsed.time_samples.len(),
        "time_samples length mismatch"
    );
    for (orig, back) in delta.time_samples.iter().zip(parsed.time_samples.iter()) {
        assert_eq!(orig.section_id, back.section_id);
        assert_eq!(orig.value_or_ext_hash, back.value_or_ext_hash);
        assert_eq!(orig.ext_value_size, back.ext_value_size);
        assert_eq!(orig.ext_value_index, back.ext_value_index);
        assert_eq!(
            orig.time.to_bits(),
            back.time.to_bits(),
            "time bit-pattern drift for section_id {}",
            orig.section_id
        );
    }
});
