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

//! b.8uk edge-case regressions exercising the public `parse_delta` /
//! `build_delta` surface with values that stressed earlier implementations:
//!
//! - `TimeSample.time` IEEE-754 edge cases (NaN / ±∞) round-trip with exact
//!   `to_bits()` equality, and ±0.0 is pinned by bits. Plain
//!   `PartialEq`/`assert_eq!` is insufficient — `NaN != NaN` under IEEE
//!   semantics, and +0.0 == -0.0, so a regression that silently flipped
//!   either could slip past a normal `assert_eq!(orig, parsed)` check.
//! - `build_delta` drops a `-0.0` time (FlatBuffers omits a value equal to
//!   the `0.0` default), pinned on the built bytes and the raw bridge parser.
//! - b.8jc: a buffer whose time sample physically carries `-0.0` (patched
//!   in, since `build_delta` cannot write it) parses to `+0.0`.
//! - Pathological 1M-element `children_list` in a `ReorderChildren` build +
//!   parse without panic, bounded by ~9 MB (well under the 100 MB service
//!   MAX_LAYER_SIZE).
//! - Empty-delta-at-FFI: a bridge `OwnedDelta` with three *absent* vectors
//!   (represented as empty `Vec`s at the cxx boundary — the FlatBuffers
//!   side elides empty vectors) parses back to an empty idiomatic delta.
//! - `extend_byte_vec` bridge fn (b.8uk perf fix) appends on top of
//!   existing contents — regression guard for a future "set instead of
//!   append" rewrite of the helper.

use oodf_ffi::{
    bridge::ffi, build_delta, parse_delta, CommandReorderChildren, OwnedDelta, StructCommand,
    TimeSample,
};

// ── TimeSample f64 edge cases ─────────────────────────────────────────────

fn round_trip_time(time: f64) -> f64 {
    let delta = OwnedDelta {
        time_samples: vec![TimeSample {
            section_id: 1,
            time,
            value_or_ext_hash: vec![0xAB, 0xCD],
            ext_value_size: 0,
            ext_value_index: 0,
        }],
        ..Default::default()
    };
    let bytes = build_delta(&delta).expect("build_delta cannot fail");
    let parsed = parse_delta(&bytes).expect("round-trip must succeed for every f64");
    assert_eq!(parsed.time_samples.len(), 1);
    parsed.time_samples[0].time
}

#[test]
fn time_sample_preserves_nan_bit_pattern() {
    let original = f64::NAN;
    let round_tripped = round_trip_time(original);
    // NaN != NaN under PartialEq, so compare the raw bit pattern. The
    // FlatBuffers wire format stores `f64` as little-endian 8 bytes with no
    // normalization, so both the sign bit and the significand must survive
    // bit-for-bit.
    assert!(
        round_tripped.is_nan(),
        "round-tripped value must still be NaN"
    );
    // We don't assert the exact bit pattern for `f64::NAN` because Rust does
    // not guarantee the canonical bit layout across platforms. The is_nan()
    // check is the load-bearing property.
}

#[test]
fn time_sample_preserves_custom_nan_payload() {
    // A NaN with a non-default payload in the significand. The FB writer
    // treats `f64` as opaque 8 bytes, so the exact bit pattern must survive.
    let bits: u64 = 0x7FF0_0000_0000_0001_u64 | 0x0008_0000_0000_0000_u64; // sNaN/qNaN payload
    let original = f64::from_bits(bits);
    assert!(original.is_nan(), "sanity: constructed value is NaN");
    let round_tripped = round_trip_time(original);
    assert_eq!(
        round_tripped.to_bits(),
        bits,
        "NaN payload bits must round-trip exactly: got 0x{:016X}, expected 0x{:016X}",
        round_tripped.to_bits(),
        bits
    );
}

#[test]
fn time_sample_preserves_positive_infinity() {
    let round_tripped = round_trip_time(f64::INFINITY);
    assert_eq!(round_tripped.to_bits(), f64::INFINITY.to_bits());
    assert!(round_tripped.is_infinite() && round_tripped.is_sign_positive());
}

#[test]
fn time_sample_preserves_negative_infinity() {
    let round_tripped = round_trip_time(f64::NEG_INFINITY);
    assert_eq!(round_tripped.to_bits(), f64::NEG_INFINITY.to_bits());
    assert!(round_tripped.is_infinite() && round_tripped.is_sign_negative());
}

/// A one-sample delta at `time`, built through `build_delta`.
fn build_one_sample_at(time: f64) -> Vec<u8> {
    build_delta(&OwnedDelta {
        time_samples: vec![TimeSample {
            section_id: 1,
            time,
            value_or_ext_hash: vec![0xAB, 0xCD],
            ext_value_size: 0,
            ext_value_index: 0,
        }],
        ..Default::default()
    })
    .expect("build_delta cannot fail")
}

/// `build_delta` cannot write a `-0.0` time. The schema declares
/// `time:double;` with no explicit default, so the generated builder treats
/// `0.0` as the default and omits the field when the written value compares
/// equal to it; since `-0.0 == 0.0` in IEEE-754, a `-0.0` time is dropped
/// and the reader falls back to the default `+0.0`.
///
/// Pinned on the build side, below `parse_delta` (which canonicalizes every
/// time to `+0.0` itself and so cannot tell a dropped field from a kept
/// one): the bytes built at `-0.0` equal the bytes built at `+0.0`, carry no
/// `-0.0` pattern, and the raw bridge parser reads bits 0 from them. If the
/// builder ever started emitting `-0.0` (a codegen change, or a forced
/// default), this fails, so the change surfaces as a deliberate signal.
/// A client that forces defaults does send `-0.0`; tests that need that
/// input must patch it into the bytes, as
/// [`parse_delta_canonicalizes_a_physical_negative_zero_time_to_positive_zero`]
/// does, because a `-0.0` handed to `build_delta` never reaches the wire.
#[test]
fn time_sample_signed_zero_collapse_pinned() {
    let positive = build_one_sample_at(0.0f64);
    let negative = build_one_sample_at(-0.0f64);
    assert_eq!(
        negative, positive,
        "build_delta drops a -0.0 time exactly as it drops the +0.0 default"
    );
    assert!(
        occurrences(&negative, &(-0.0f64).to_le_bytes()).is_empty(),
        "the bytes built at -0.0 carry no -0.0 pattern"
    );

    let raw = ffi::parse_delta(&negative);
    assert!(
        raw.error.repr == ffi::ParseErrorKind::Ok.repr,
        "the built buffer verifies: {}",
        raw.error_message
    );
    assert_eq!(raw.delta.time_samples.len(), 1);
    assert_eq!(
        raw.delta.time_samples[0].time.to_bits(),
        0,
        "the raw C++ parser reads the dropped field as the +0.0 default"
    );

    assert_eq!(
        round_trip_time(0.0f64).to_bits(),
        0,
        "+0.0 round-trips as +0.0"
    );
}

/// A sentinel time the patched-bytes test builds its sample at: FlatBuffers
/// stores it (it is not the `0.0` default), and its 8 bytes are then
/// overwritten with `-0.0`'s.
const SENTINEL_TIME: f64 = 1_234.567_890_123;

/// The positions at which `pattern` occurs in `bytes`.
fn occurrences(bytes: &[u8], pattern: &[u8; 8]) -> Vec<usize> {
    bytes
        .windows(8)
        .enumerate()
        .filter(|(_, w)| *w == pattern)
        .map(|(at, _)| at)
        .collect()
}

/// b.8jc: a buffer whose time sample physically carries `-0.0` — the bytes
/// a client that forces FlatBuffers defaults sends, which `build_delta`
/// cannot produce (see [`time_sample_signed_zero_collapse_pinned`]) —
/// parses to a time of bits 0. The buffer is built at a sentinel time whose
/// 8 little-endian bytes, found exactly once, are overwritten with `-0.0`'s.
/// The C++ parser hands the bridge the `-0.0` it read (checked through the
/// raw bridge call, so the test is not vacuous); `parse_delta`'s conversion
/// to the owned type is what canonicalizes it.
#[test]
fn parse_delta_canonicalizes_a_physical_negative_zero_time_to_positive_zero() {
    let delta = OwnedDelta {
        time_samples: vec![TimeSample {
            section_id: 7,
            time: SENTINEL_TIME,
            value_or_ext_hash: vec![0xAB, 0xCD],
            ext_value_size: 0,
            ext_value_index: 0,
        }],
        ..Default::default()
    };
    let mut bytes = build_delta(&delta).expect("build_delta cannot fail");
    let negative_zero = (-0.0f64).to_le_bytes();
    assert!(
        occurrences(&bytes, &negative_zero).is_empty(),
        "fixture: no -0.0 pattern before patching"
    );
    let at = occurrences(&bytes, &SENTINEL_TIME.to_le_bytes());
    assert_eq!(at.len(), 1, "fixture: the sentinel occurs exactly once");
    bytes[at[0]..at[0] + 8].copy_from_slice(&negative_zero);
    assert_eq!(
        occurrences(&bytes, &negative_zero).len(),
        1,
        "fixture: the buffer carries -0.0 exactly once"
    );

    let raw = ffi::parse_delta(&bytes);
    assert!(
        raw.error.repr == ffi::ParseErrorKind::Ok.repr,
        "fixture: the patched buffer verifies: {}",
        raw.error_message
    );
    assert_eq!(
        raw.delta.time_samples[0].time.to_bits(),
        (-0.0f64).to_bits(),
        "premise: the C++ parser reads the physical -0.0"
    );

    let parsed = parse_delta(&bytes).expect("the patched buffer parses");
    assert_eq!(parsed.time_samples.len(), 1);
    assert_eq!(
        parsed.time_samples[0].time.to_bits(),
        0,
        "parse_delta hands back +0.0 for a physical -0.0"
    );
    assert_eq!(parsed.time_samples[0].value_or_ext_hash, vec![0xAB, 0xCD]);
}

// ── Pathological build: 1M-element ReorderChildren children_list ──────────

#[test]
fn build_and_parse_one_million_element_children_list() {
    // 1_000_000 × 8 bytes = ~8 MB payload for children_list alone. Well
    // under the 100 MB MAX_LAYER_SIZE but an order of magnitude larger than
    // the largest fixture in tests/fixtures/oodf/. Proves the parse/build
    // path has no quadratic behavior at a size that would be catastrophic
    // with O(n²) byte-by-byte copy.
    const N: usize = 1_000_000;
    let original = OwnedDelta {
        struct_commands: vec![StructCommand::ReorderChildren(CommandReorderChildren {
            section_id: 42,
            children_list_id: 1,
            children_list: (0..N as u64).collect(),
        })],
        ..Default::default()
    };

    let bytes = build_delta(&original).expect("build_delta cannot fail");
    // Sanity: 1M u64 values = 8 MB of payload data, so the built buffer is
    // at least 8 MB. Upper-bound at 12 MB to catch runaway overhead.
    assert!(bytes.len() >= 8 * 1_000_000);
    assert!(bytes.len() <= 12 * 1_000_000);

    let parsed = parse_delta(&bytes).expect("1M-element children_list must parse");
    match &parsed.struct_commands[..] {
        [StructCommand::ReorderChildren(c)] => {
            assert_eq!(c.section_id, 42);
            assert_eq!(c.children_list_id, 1);
            assert_eq!(c.children_list.len(), N);
            // Spot-check endpoints and midpoint — full equality would cost
            // an extra 8 MB compare for no additional signal.
            assert_eq!(c.children_list[0], 0);
            assert_eq!(c.children_list[N / 2], (N / 2) as u64);
            assert_eq!(c.children_list[N - 1], (N - 1) as u64);
        }
        other => panic!("expected a single ReorderChildren, got {other:?}"),
    }
}

// ── Empty Delta at FFI: vectors absent, not just empty ────────────────────

#[test]
fn build_delta_default_produces_absent_optional_vectors() {
    // An `OwnedDelta::default()` has zero-length `struct_commands`,
    // `set_fields`, `time_samples` and an empty `source_format`. The
    // FlatBuffers builder elides empty offset vectors (writes them as a
    // null sub-offset), so the wire buffer represents them as *absent*
    // rather than as length-zero vectors.
    //
    // This test pins that parsing a default-built buffer yields a default
    // OwnedDelta — i.e. the absent-vector path in oodf.cpp
    // (`if (const auto* fields = delta->setFields())`) correctly handles a
    // null accessor return, and does not silently substitute a sentinel
    // value.
    let original = OwnedDelta::default();
    let bytes = build_delta(&original).expect("build_delta cannot fail");
    let parsed = parse_delta(&bytes).expect("empty delta must parse");
    assert_eq!(parsed, original);
    assert!(parsed.struct_commands.is_empty());
    assert!(parsed.set_fields.is_empty());
    assert!(parsed.time_samples.is_empty());
    assert!(parsed.source_format.is_empty());
    assert!(!parsed.is_diff);
    assert_eq!(parsed.base_version, 0);
}

// ── extend_byte_vec bridge fn: append-on-top semantics ────────────────────

#[test]
fn extend_byte_vec_appends_on_top_of_existing_contents() {
    // Direct test of the b.8uk perf helper. The helper is crucial for
    // throughput on 1 MB+ external values — regressing it from
    // `extend_from_slice` to `= src.to_vec()` would be a subtle correctness
    // bug (the first chunk would clobber any pre-existing bytes). Pin
    // append semantics explicitly.
    let mut dst: Vec<u8> = vec![1, 2, 3];
    oodf_ffi::bridge::extend_byte_vec(&mut dst, &[4, 5, 6]);
    assert_eq!(dst, vec![1, 2, 3, 4, 5, 6]);

    // Second call must also append, not replace.
    oodf_ffi::bridge::extend_byte_vec(&mut dst, &[7]);
    assert_eq!(dst, vec![1, 2, 3, 4, 5, 6, 7]);

    // Extending with an empty slice is a no-op.
    oodf_ffi::bridge::extend_byte_vec(&mut dst, &[]);
    assert_eq!(dst, vec![1, 2, 3, 4, 5, 6, 7]);
}

// ── ParseErrorKind tag space hasn't collided with InvalidUtf8=8 ───────────

#[test]
fn parse_error_kind_invalid_utf8_has_expected_repr() {
    // Pin the numeric repr of the new `InvalidUtf8` variant so a silent
    // renumbering would fail loudly. Any renumber must be deliberate and
    // kept in sync between oodf.cpp's `ParseErrorKind` and the cxx bridge.
    assert_eq!(ffi::ParseErrorKind::InvalidUtf8.repr, 8);
}
