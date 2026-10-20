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

//! Task 3.4.C — wire-format regression test anchored on the Postman OODF
//! payload (280 bytes — the FlatBuffers slice that lives inside the Postman
//! OMPF chunk after the 16-byte OMPF header).
//!
//! This fixture is the canonical "known-good buffer" from the Epic 2 C++
//! parser tests. Keeping a Rust-side copy of it pinned here ensures the
//! bridge + conversion layers don't silently regress on a real wire payload.

use oodf_ffi::{build_delta, parse_delta, StructCommand};

const FIXTURE: &[u8] = include_bytes!("fixtures/postman_chunk.bin");

#[test]
fn fixture_parses_into_two_diff_change_sections() {
    assert_eq!(
        FIXTURE.len(),
        280,
        "Postman OODF fixture must be exactly 280 bytes"
    );

    let delta = parse_delta(FIXTURE).expect("Postman fixture must parse as a valid OODF buffer");

    assert_eq!(
        delta.struct_commands.len(),
        2,
        "Postman fixture is known to carry exactly 2 struct commands"
    );
    for (i, cmd) in delta.struct_commands.iter().enumerate() {
        assert!(
            matches!(cmd, StructCommand::DiffChangeSection(_)),
            "Postman fixture struct_commands[{i}] must be DiffChangeSection, got {cmd:?}"
        );
    }
}

#[test]
fn fixture_round_trips_semantically() {
    // parse → build → parse yields the same idiomatic OwnedDelta. Byte-level
    // equality is not asserted: FlatBuffers offset ordering is unspecified, so
    // the re-built buffer is allowed to differ byte-for-byte from FIXTURE.
    let first = parse_delta(FIXTURE).expect("first parse must succeed");
    let rebuilt = build_delta(&first).expect("build_delta cannot fail on a parsed-back OwnedDelta");
    let third = parse_delta(&rebuilt).expect("re-parse of rebuilt bytes must succeed");
    assert_eq!(
        first, third,
        "round-tripped delta must be PartialEq-equal to the original parse"
    );
}
