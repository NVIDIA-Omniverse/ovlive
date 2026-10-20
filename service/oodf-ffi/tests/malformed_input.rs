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

//! Task 3.4.B — negative-path coverage through the public `parse_delta` API.
//!
//! Every test uses `matches!` (not `.is_err()`) so the specific `OodfError`
//! variant is pinned. Epic 2's C++ parser is the source of truth for which
//! error kind a given malformed input produces — where Epic 2 has a choice
//! between two kinds (e.g. `TooShort` vs. `VerifierFailed`), each assertion
//! pins the set actually observed at the time of writing.
//!
//! Coverage notes (per `OodfError` variants in `oodf-ffi/src/error.rs`):
//! - `EmptyInput`, `TooShort`, `BadMagic`, `VerifierFailed`, `UnknownCommandType`
//!   are exercised directly by the tests in this file.
//! - `UnknownCommandTag` is bridge-layer and exercised via
//!   `bridge::ffi::CommandTag::NONE` → `StructCommand::try_from(_)`.
//! - `AllocationFailed` requires fault injection (allocator failure inside the
//!   C++ core); it's covered conceptually in Epic 2's C++ tests and cannot be
//!   reliably triggered from Rust without allocator hooks.
//! - `InternalError` is raised when the bridge's `ParseErrorKind::Ok` path
//!   reports an inconsistent value; not reproducible without mocking the C++
//!   parser. Epic 2 unit tests exercise the trigger condition.
//! - `Ffi(_)` is reserved-for-future — documented, not exercised.

use oodf_ffi::{
    build_delta, parse_delta, CommandCreateSection, OodfError, OwnedDelta, StructCommand,
};

// ── Shape-level malformed inputs ────────────────────────────────────────────

#[test]
fn empty_input_returns_empty_input_error() {
    let err = parse_delta(&[]).expect_err("empty input must be rejected");
    assert!(matches!(err, OodfError::EmptyInput), "got {err:?}");
}

#[test]
fn four_zero_bytes_returns_too_short() {
    // A 4-byte buffer falls below `kMinFlatBufferHeader = 8`, so oodf.cpp's
    // pre-verifier size guard rejects it with `TooShort` before any
    // FlatBuffers accessor runs. The earlier assertion accepted
    // `VerifierFailed` as a legitimate alternative "for future-proofing",
    // but that was a hedge that let a real reordering regression pass
    // silently. Pin the exact current outcome — a deliberate future
    // reordering change must update both oodf.cpp and this assertion.
    let err = parse_delta(&[0u8; 4]).expect_err("4-byte buffer must be rejected");
    assert!(matches!(err, OodfError::TooShort), "got {err:?}");
}

#[test]
fn random_bytes_never_panic_and_return_bad_magic() {
    // Each seed produces a 64-byte buffer that clears `kMinFlatBufferHeader`
    // (8 bytes) but fails `DeltaBufferHasIdentifier`, which compares
    // bytes[4..8] to the ASCII string "OODF". None of these patterns spell
    // OODF, so `BadMagic` fires before the verifier is invoked — pin
    // exactly that outcome. The load-bearing property (no panic) is already
    // implicit in `expect_err`; the variant pin catches reorderings.
    for seed in [0xABu8, 0xFFu8, 0x5Au8, 0x00u8] {
        let buf = [seed; 64];
        let err = parse_delta(&buf).expect_err("random bytes must be rejected");
        assert!(
            matches!(err, OodfError::BadMagic),
            "seed=0x{seed:02x}: got {err:?}"
        );
    }
}

// ── Content-level malformed inputs (hex-patch a known-good buffer) ─────────

#[test]
fn wrong_file_identifier_returns_bad_magic() {
    // Build a valid OODF buffer, then clobber the 4-byte file_identifier at
    // bytes 4..8 (FlatBuffers contract: file_identifier lives immediately
    // after the 4-byte root uoffset_t). oodf.cpp runs
    // `DeltaBufferHasIdentifier` ahead of the verifier, so a clobbered
    // identifier deterministically surfaces as `BadMagic`. Pin that —
    // accepting `VerifierFailed` as a "future-proof" alternative would
    // mask a real change in the parse-time ordering.
    let mut bytes = build_delta(&OwnedDelta::default()).expect("build_delta cannot fail");
    assert!(bytes.len() >= 8, "need at least 8 bytes to patch magic");
    assert_eq!(
        &bytes[4..8],
        b"OODF",
        "sanity: build_delta must emit OODF identifier"
    );
    bytes[4..8].copy_from_slice(b"XXXX");

    let err = parse_delta(&bytes).expect_err("bad file_identifier must be rejected");
    assert!(matches!(err, OodfError::BadMagic), "got {err:?}");
}

#[test]
fn out_of_range_struct_command_type_is_rejected() {
    // Build a valid Delta with exactly one CreateSection command, whose type
    // vector therefore contains a single byte with value 1 (CommandTag::
    // CreateSection). Find that byte by scanning for the FB u8-vector length
    // prefix `01 00 00 00 01` and flip the discriminant to 99.
    let original = OwnedDelta {
        struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
            parent_id: 1,
            section_id: 2,
            section_name: "patch-target".to_string(),
            section_type: 3,
        })],
        ..Default::default()
    };
    let mut bytes = build_delta(&original).expect("build_delta cannot fail");

    // FB u8-vector encoding: 4-byte little-endian length, then the bytes.
    // A length-1 vector whose single byte is `0x01` matches the pattern
    // `01 00 00 00 01`. There is exactly one such length-1 type vector in
    // a delta with one StructCommand, so the first match is the one to patch.
    let pattern = [0x01u8, 0x00, 0x00, 0x00, 0x01];
    let idx = bytes
        .windows(pattern.len())
        .position(|w| w == pattern)
        .expect("length-1 type-vector pattern must appear in a single-CreateSection delta");
    bytes[idx + 4] = 99;

    // The flatc-generated `VerifyStructCommandVector` accepts any u8 in the
    // discriminant vector (its `default:` branch returns true for unknown
    // tags — see oodf.cpp's `StructCommand_NONE / default` case). The
    // verifier therefore does not reject tag=99; oodf.cpp's own dispatch
    // switch does, returning `UnknownCommandType(99)`. Pin that exact
    // variant and payload so a flatc-side change that starts rejecting
    // unknown tags at verify time surfaces as a test failure rather than
    // silently shifting the error path.
    let err = parse_delta(&bytes).expect_err("out-of-range command type must be rejected");
    assert!(
        matches!(err, OodfError::UnknownCommandType(99)),
        "got {err:?}"
    );
}

// Bridge-layer NONE-tag rejection is covered by `try_from_tag_none_returns_err`
// in oodf-ffi/tests/conversion.rs — same assertion, same path, kept there
// because that's the conversion-layer's home.
