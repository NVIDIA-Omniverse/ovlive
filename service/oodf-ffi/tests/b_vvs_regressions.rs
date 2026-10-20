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

//! b.vvs regression tests for items 2 (verifier bounds) and 8 (build_delta
//! fallibility). Lives in `oodf-ffi/tests/` because both regressions exercise
//! the C++ FFI surface — item 2 needs the cxx bridge to construct an
//! over-sized OwnedDelta that the in-process `parse_delta` then has to reject
//! quickly, and item 8 needs `bridge::ffi::StructCommand` with a raw out-of-
//! range `tag.repr` to drive the new `BuildErrorKind::UnknownCommandTag`
//! path. Both require `--features internal-bridge`, which is the same gate
//! the existing `edge_cases.rs` tests use.
//!
//! AC1 — verifier bounds (item 2). The DoS shape the C++ fix defends against
//! is a verifier-valid buffer claiming millions of nested tables: the
//! library-default `max_tables = 1_000_000` would walk every one of them
//! before returning. b.vvs lowers `max_tables` to 100_000
//! (`kVerifierMaxTables`, equal to `oodf_ffi::MAX_DELTA_TABLES`), so a buffer
//! carrying ≥ 200_000 `StructCommand` tables must be rejected before the
//! verifier walks them all. An honest such buffer never reaches the verifier
//! now: the entry-count probe (`probe_entry_counts`) reads its vector lengths
//! first and refuses it `TooManyTables` in O(1) (`entry_count_probe.rs`
//! covers the probe). The verifier's cap stays as the backstop for a buffer
//! the probe cannot count, so the timing test here forges a vector header the
//! verifier checks only AFTER walking the 200_000 commands: the probe fails
//! on it and falls through, and the verifier must still stop at its table
//! cap. The wall-time bound asserted here is *deliberately generous* —
//! 1.5 seconds on the slowest possible build (debug + sanitizer + cold cache)
//! is still an order of magnitude under what walking 200K tables would cost.
//! The point is to catch a regression that removes the verifier-bounds
//! tightening entirely, not to land a tight performance assertion that
//! starts flaking under CI load. The pre-b.vvs default walked every one of
//! 200K tables and observably ran > 5 seconds in single-thread debug
//! builds.
//!
//! AC5 — build_delta fallibility (item 8). Before b.vvs the C++ side called
//! `std::abort()` when it observed a `StructCommand` with an out-of-range
//! `CommandTag` discriminant. That kill blew away every concurrent in-flight
//! RPC sharing the process, even though only one of them sent the bad tag.
//! The fix routes the abort through a `BuildResult { error:
//! BuildErrorKind::UnknownCommandTag, error_message, bytes: empty }` so the
//! Rust side surfaces the failure as a clean `OodfError::BuildUnknownCommandTag`
//! to the offending caller. Idiomatic Rust types prevent the bad-tag shape
//! at the API surface (the enum-typed `StructCommand` cannot carry an
//! invalid `tag.repr`), so the regression test has to reach below the public
//! API via the `internal-bridge` feature to construct a malformed bridge
//! `StructCommand`. The Rust-side `lib::build_delta` only accepts
//! `&OwnedDelta`, so we call into `bridge::ffi::build_delta` directly.

use oodf_ffi::{
    bridge::{
        self,
        ffi::{self, CommandTag},
    },
    parse_delta, CommandCreateSection, OodfError, OwnedDelta, StructCommand,
};
use std::time::{Duration, Instant};

// ── AC1: verifier rejects an oversized table payload quickly ───────────────

/// The OODF bytes of a delta whose `structCommands` vector carries `n` valid
/// `CommandDeleteSection` tables, followed by `samples` inline time samples.
///
/// `CommandDeleteSection` is the cheapest variant to emit: no strings, no
/// nested vectors, two `u64` fields per table. The buffer therefore scales
/// at roughly the FlatBuffers per-table overhead (~40 bytes) rather than the
/// per-payload byte budget — exactly the shape the verifier-tables cap
/// defends against (lots of tables, modest byte total).
fn delete_sections_then_samples(n: usize, samples: usize) -> Vec<u8> {
    use oodf_ffi::{CommandDeleteSection, TimeSample};
    let original = OwnedDelta {
        struct_commands: (0..n)
            .map(|i| {
                StructCommand::DeleteSection(CommandDeleteSection {
                    parent_id: i as u64,
                    section_id: (i + 1) as u64,
                })
            })
            .collect(),
        time_samples: (0..samples)
            .map(|i| TimeSample {
                section_id: 1,
                time: i as f64,
                value_or_ext_hash: vec![1],
                ext_value_size: 0,
                ext_value_index: 0,
            })
            .collect(),
        ..Default::default()
    };
    oodf_ffi::build_delta(&original).expect("build cannot fail on idiomatic input")
}

/// `parse_delta(bytes)` and the wall-clock time it took.
fn time_parse(bytes: &[u8]) -> (Result<OwnedDelta, OodfError>, Duration) {
    let start = Instant::now();
    let parsed = parse_delta(bytes);
    (parsed, start.elapsed())
}

/// Byte position of the length word of the root table's `timeSamples`
/// vector. FlatBuffers layout: the root `uoffset_t` at byte 0 points at the
/// root table, whose first four bytes are a signed back-offset to its
/// vtable; `timeSamples` is the schema's sixth field slot (isDiff,
/// baseVersion, structCommands_type, structCommands, setFields, timeSamples),
/// so its vtable entry sits at byte `4 + 2 * 5`.
fn time_samples_length_pos(buf: &[u8]) -> usize {
    const VT_TIMESAMPLES: usize = 14;
    let u32_at = |at: usize| u32::from_le_bytes(buf[at..at + 4].try_into().unwrap()) as usize;
    let root = u32_at(0);
    let back = i32::from_le_bytes(buf[root..root + 4].try_into().unwrap());
    let vtable = usize::try_from(root as i64 - i64::from(back)).expect("vtable inside buffer");
    let field = usize::from(u16::from_le_bytes([
        buf[vtable + VT_TIMESAMPLES],
        buf[vtable + VT_TIMESAMPLES + 1],
    ]));
    assert_ne!(field, 0, "the fixture must carry a timeSamples vector");
    let slot = root + field;
    slot + u32_at(slot)
}

/// The verifier's table cap is the backstop for a buffer the entry-count
/// probe cannot count, and it still stops the walk early: 200,000 delete
/// commands followed by a `timeSamples` vector whose length word is forged to
/// `u32::MAX`. The probe bounds-checks every vector header before it counts,
/// so it fails on the forged one and falls through; the verifier checks
/// fields in schema order, so it reaches the 200,000 command tables BEFORE the
/// forged header and must stop at its `max_tables = 100_000` cap.
#[test]
fn verifier_rejects_oversized_struct_commands_table_vector_quickly() {
    // 200_000 tables ≥ verifier's `max_tables = 100_000`. The verifier walks
    // tables in vector order and short-circuits on the first violation, so
    // this stops within the first ~100K iterations — far faster than the
    // library-default 1M-table walk that pre-b.vvs would have completed.
    //
    // The cap on `kMaxCommands` (1_000_000) is independently enforced AFTER
    // verification in `parse_delta`, so any rejection here is the verifier
    // talking. We assert the variant is `VerifierFailed` (not `TooManyTables`,
    // which would mean the probe counted a forged header, nor
    // `UnknownCommandType` or anything else) to pin the fault attribution.
    let mut bytes = delete_sections_then_samples(200_000, 1);
    let len_pos = time_samples_length_pos(&bytes);
    assert_eq!(
        &bytes[len_pos..len_pos + 4],
        &1u32.to_le_bytes(),
        "located the timeSamples length word"
    );
    bytes[len_pos..len_pos + 4].copy_from_slice(&u32::MAX.to_le_bytes());

    let (result, elapsed) = time_parse(&bytes);
    let err = result.expect_err("verifier must reject the 200K-table buffer");
    assert!(
        matches!(err, OodfError::VerifierFailed(_)),
        "expected VerifierFailed (verifier max_tables bound), got {err:?}"
    );

    // Wall-clock bound: deliberately generous so CI variance doesn't make
    // this flaky. The pre-fix verifier walked all 200K tables and observably
    // exceeded 5 s in single-thread debug builds; the fix returns in tens of
    // milliseconds. 1.5 s leaves >10× headroom against CI jitter while still
    // catching a regression that removes the bound entirely.
    assert!(
        elapsed < Duration::from_millis(1500),
        "verifier walked too far on a 200K-table payload: took {elapsed:?}; \
         expected <1.5s with max_tables=100K bound (was unbounded ~5s+ pre-b.vvs)"
    );
}

/// The same 200,000 commands with honest headers never reach the verifier:
/// the entry-count probe refuses them `TooManyTables` with the counts.
#[test]
fn oversized_struct_commands_table_vector_with_honest_headers_is_refused_by_the_probe() {
    let (result, _) = time_parse(&delete_sections_then_samples(200_000, 0));
    match result {
        Err(OodfError::TooManyTables {
            struct_commands,
            set_fields,
            time_samples,
            max_entries,
        }) => assert_eq!(
            (struct_commands, set_fields, time_samples, max_entries),
            (200_000, 0, 0, oodf_ffi::MAX_DELTA_TABLES as u64 - 1)
        ),
        other => panic!("expected TooManyTables, got {other:?}"),
    }
}

// Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
#[test]
fn verifier_max_tables_constant_documented_in_cpp() {
    // The verifier-bound constants must exist in oodf.cpp, and the table cap
    // must equal the Rust constant the service derives its per-delta entry
    // limit from (`oodf_ffi::MAX_DELTA_TABLES`): the two are written in two
    // languages, so this test is what keeps them one value. The load-bearing
    // behaviour assertions live in the tests above and in
    // `entry_count_probe.rs`.
    let cpp_source = include_str!("../cpp/oodf.cpp");
    assert!(
        cpp_source.contains("kVerifierMaxTables"),
        "oodf.cpp must define `kVerifierMaxTables` (b.vvs item 2); rename or removal is \
         a regression that drops the verifier-tables DoS defense"
    );
    assert!(
        cpp_source.contains("kVerifierMaxDepth"),
        "oodf.cpp must define `kVerifierMaxDepth` (b.vvs item 2)"
    );
    let literal = cpp_constant(cpp_source, "kVerifierMaxTables");
    assert_eq!(
        literal,
        oodf_ffi::MAX_DELTA_TABLES,
        "oodf.cpp's `kVerifierMaxTables` must equal `oodf_ffi::MAX_DELTA_TABLES`; change both \
         together (and get the security review the constant's comment asks for)"
    );
}

/// The value of `constexpr size_t <name> = <literal>;` in `source`, with C++14
/// digit separators (`100'000`) removed.
fn cpp_constant(source: &str, name: &str) -> usize {
    let needle = format!("constexpr size_t {name} = ");
    let start = source
        .find(&needle)
        .unwrap_or_else(|| panic!("oodf.cpp must define `{needle}…;`"))
        + needle.len();
    let end = start
        + source[start..]
            .find(';')
            .unwrap_or_else(|| panic!("`{name}`'s definition must end with `;`"));
    let digits: String = source[start..end].chars().filter(|c| *c != '\'').collect();
    digits.trim().parse().unwrap_or_else(|e| {
        panic!(
            "`{name}` must be an integer literal; got {:?}: {e}",
            &source[start..end]
        )
    })
}

// ── AC5: build_delta surfaces UnknownCommandTag instead of aborting ────────

/// Build a minimal valid `ffi::OwnedDelta` carrying exactly one
/// `StructCommand` with the given raw `tag` byte. Other payload fields are
/// left at default (`Default::default()`), which is safe regardless of the
/// tag value — the C++ `build_delta` switch dispatches on `tag` before
/// reading any payload, and an out-of-range tag deterministically lands in
/// the `default:` branch that now returns `BuildErrorKind::UnknownCommandTag`.
fn ffi_delta_with_raw_tag(repr: u8) -> ffi::OwnedDelta {
    ffi::OwnedDelta {
        struct_commands: vec![ffi::StructCommand {
            tag: CommandTag { repr },
            ..Default::default()
        }],
        ..Default::default()
    }
}

#[test]
fn ffi_build_delta_returns_unknown_command_tag_for_out_of_range_repr() {
    // `repr = 7` is one past the highest declared `CommandTag`
    // (DiffDeleteSection = 6). The bridge admits any `u8`, so this exercises
    // the post-b.vvs `default:` branch in oodf.cpp's build dispatch which
    // used to call `std::abort()` and now returns a typed BuildResult.
    let delta = ffi_delta_with_raw_tag(7);
    let result = ffi::build_delta(&delta);
    assert_eq!(
        result.error.repr,
        ffi::BuildErrorKind::UnknownCommandTag.repr,
        "expected BuildErrorKind::UnknownCommandTag (repr={}), got repr={} \
         (message: {:?})",
        ffi::BuildErrorKind::UnknownCommandTag.repr,
        result.error.repr,
        result.error_message
    );
    // The error message must carry the offending tag byte and the command
    // index so an operator log can pinpoint the bad bridge caller without
    // re-running with extra instrumentation.
    assert!(
        result.error_message.contains('7'),
        "error_message must mention the offending tag value 7; got: {:?}",
        result.error_message
    );
    assert!(
        result.error_message.contains("index 0"),
        "error_message must mention the command index; got: {:?}",
        result.error_message
    );
    // bytes vector is meaningless when error != Ok — pin it empty so a
    // future regression that returns partial bytes is caught.
    assert!(
        result.bytes.is_empty(),
        "BuildResult.bytes must be empty when error != Ok; got {} bytes",
        result.bytes.len()
    );
}

#[test]
fn ffi_build_delta_returns_unknown_command_tag_for_none_repr() {
    // `repr = 0` is the `NONE` discriminant — the cxx-generated default for
    // a `CommandTag` shared enum and the same value a `Default`-constructed
    // bridge `StructCommand` lands on. A bridge caller that forgot to set
    // `tag` therefore hits this path; pre-b.vvs that was `std::abort()`,
    // post-b.vvs it surfaces as `UnknownCommandTag` so the offending caller
    // can be informed without taking the process down.
    let delta = ffi_delta_with_raw_tag(0);
    let result = ffi::build_delta(&delta);
    assert_eq!(
        result.error.repr,
        ffi::BuildErrorKind::UnknownCommandTag.repr,
        "NONE tag must surface as UnknownCommandTag, got repr={} (message: {:?})",
        result.error.repr,
        result.error_message
    );
}

#[test]
fn ffi_build_delta_succeeds_for_every_legal_tag() {
    // Sanity: each of the six declared `CommandTag` values produces an
    // `Ok` BuildResult when paired with a default payload. This pins the
    // dispatch table so a regression that mis-handles a legal tag fires
    // alongside the bad-tag assertions above instead of as a confusing
    // round-trip failure elsewhere.
    for tag in [
        CommandTag::CreateSection,
        CommandTag::DeleteSection,
        CommandTag::MoveSection,
        CommandTag::ReorderChildren,
        CommandTag::DiffChangeSection,
        CommandTag::DiffDeleteSection,
    ] {
        let delta = ffi::OwnedDelta {
            struct_commands: vec![ffi::StructCommand {
                tag,
                ..Default::default()
            }],
            ..Default::default()
        };
        let result = ffi::build_delta(&delta);
        assert_eq!(
            result.error.repr,
            ffi::BuildErrorKind::Ok.repr,
            "legal tag repr={} must produce Ok; got repr={} (message: {:?})",
            tag.repr,
            result.error.repr,
            result.error_message
        );
        assert!(
            !result.bytes.is_empty(),
            "successful build must emit non-empty bytes for tag repr={}",
            tag.repr
        );
    }
}

#[test]
fn public_build_delta_maps_unknown_command_tag_into_oodf_error() {
    // End-to-end: the public Rust wrapper `oodf_ffi::build_delta` is typed
    // `Result<Vec<u8>, OodfError>` after b.vvs item 8. The wrapper cannot be
    // invoked with a malformed `StructCommand` (the idiomatic enum prevents
    // it), but the `from_build_ffi` mapping must still produce the right
    // `OodfError` variant when the bridge returns
    // `BuildErrorKind::UnknownCommandTag`. We round-trip through
    // `OodfError::from_build_ffi` directly to verify the mapping, then
    // verify the wrapper rejects the matching shape end-to-end via the
    // bridge.
    let result = ffi::build_delta(&ffi_delta_with_raw_tag(42));
    let err = OodfError::from_build_ffi(&result);
    match err {
        OodfError::BuildUnknownCommandTag(msg) => {
            assert!(
                msg.contains("42"),
                "BuildUnknownCommandTag payload must carry the C++-side message \
                 including the offending tag; got: {msg}"
            );
        }
        other => panic!("expected BuildUnknownCommandTag, got {other:?}"),
    }
}

// ── AC5 / process-survival sanity: bad tag does NOT take the process down ──

#[test]
fn bad_tag_build_does_not_terminate_process_concurrent_good_builds_still_work() {
    // The whole point of b.vvs item 8: a bridge caller's bad tag must NOT
    // tear down the process. A weak version of "process still alive" is to
    // make a bad-tag call, then immediately issue a normal build_delta on
    // the same thread and assert success. If b.vvs ever regressed back to
    // `std::abort()`, the bad-tag call would terminate the test binary and
    // this test would surface as a "test thread panicked / process killed"
    // failure rather than a clean assertion failure — either way, the
    // regression is loud.
    let bad = ffi::build_delta(&ffi_delta_with_raw_tag(99));
    assert_eq!(
        bad.error.repr,
        ffi::BuildErrorKind::UnknownCommandTag.repr,
        "sanity: bad-tag call must return UnknownCommandTag (got repr={})",
        bad.error.repr
    );

    // Now issue a normal build_delta on an idiomatic OwnedDelta. Pre-b.vvs
    // the process would already be dead; post-b.vvs this succeeds because
    // the bad-tag failure is data, not a process exit.
    let good = OwnedDelta {
        struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
            parent_id: 1,
            section_id: 2,
            section_name: "still-alive".to_string(),
            section_type: 0,
        })],
        ..Default::default()
    };
    let bytes = oodf_ffi::build_delta(&good)
        .expect("post-bad-tag build_delta must still succeed (process must not have aborted)");
    let parsed = parse_delta(&bytes).expect("round-trip after bad-tag must succeed");
    assert_eq!(parsed, good);
}

// ── AC4 reinforcement: bridge::extend_byte_vec exists but is doc-hidden ────

#[test]
fn bridge_extend_byte_vec_is_reachable_with_internal_bridge_feature() {
    // The cxx bridge's `extend_byte_vec` Rust impl is `#[doc(hidden)] pub`
    // so cxx can find it. The `bridge` module itself is gated `pub(crate)`
    // unless `internal-bridge` is enabled (b.vvs item 7). This test runs
    // under `--features internal-bridge` (oodf-ffi tests' default invocation
    // per CLAUDE.md `## Build Commands`), so the function MUST be callable.
    // The companion negative-case test (bridge module NOT reachable from
    // live-edit) lives in `tests/oodf_bridge_visibility_test.rs` at the
    // workspace root, where `internal-bridge` is intentionally disabled.
    let mut dst: Vec<u8> = vec![0xAA];
    bridge::extend_byte_vec(&mut dst, &[0xBB, 0xCC]);
    assert_eq!(dst, vec![0xAA, 0xBB, 0xCC]);
}
