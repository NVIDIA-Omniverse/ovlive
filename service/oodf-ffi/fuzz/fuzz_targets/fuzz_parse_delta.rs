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

//! Task 5.4 — fuzz target for `oodf_ffi::parse_delta`.
//!
//! Feeds arbitrary byte slices into the parser and asserts the function
//! returns — never panics or aborts. Both `Ok(_)` and `Err(_)` are acceptable
//! outcomes; what we're catching is unexpected panics, UB, or runaway
//! allocations in the FFI / conversion layer. The C++ FlatBuffers verifier
//! upstream is already well-fuzzed; this target focuses on our Rust-side
//! error-handling wiring.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // We deliberately ignore the return value. A libfuzzer run will flag any
    // panic / abort / timeout as a crash automatically.
    let _ = oodf_ffi::parse_delta(data);
});
