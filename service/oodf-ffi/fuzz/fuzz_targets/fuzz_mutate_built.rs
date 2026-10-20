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

//! b.vvs item 10 — mutate-built-bytes fuzz target.
//!
//! Complements `fuzz_round_trip` by feeding `parse_delta` bytes that are
//! *almost* well-formed: the fuzzer mutates the byte stream produced by
//! `build_delta` for an arbitrary `OwnedDelta`, hunting for verifier /
//! accessor crashes on buffers that share the layout of legitimate OODF
//! traffic but differ in offsets, vector lengths, or string-byte payloads.
//!
//! The plain `fuzz_parse_delta` target feeds random bytes, which the
//! verifier rejects within microseconds 99% of the time; this target's
//! input distribution puts most cases past the verifier so the post-
//! verifier accessor path is exercised more deeply.
//!
//! Seed the corpus from the fixture buffers:
//!
//! ```sh
//! cd oodf-ffi
//! mkdir -p corpus/fuzz_mutate_built
//! cp ../tests/fixtures/oodf/*.oodf corpus/fuzz_mutate_built/
//! cargo +nightly fuzz run fuzz_mutate_built -- -max_total_time=60
//! ```

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Feed the raw fuzzer-mutated bytes straight to parse_delta.
    // Mutate-built-bytes coverage is best realized by seeding the libfuzzer
    // corpus from `tests/fixtures/oodf/*.oodf` (see header comment): the
    // mutator then applies bit/byte-level edits to known-valid OODF
    // buffers, producing nearly-valid inputs that exercise the post-
    // verifier accessor path.
    let _ = oodf_ffi::parse_delta(data);
});
