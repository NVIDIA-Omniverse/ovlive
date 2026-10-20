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

// Compile-only regression check: confirms that `include/oodf.h` is
// self-contained when included alongside the cxx-generated bridge header.
// Catches missing-include regressions (e.g. forgetting <cstdint> or
// "rust/cxx.h") before they break the main crate build.
//
// This TU is compiled by a secondary `cc::Build` invocation in
// `oodf-ffi/build.rs` into a static archive that is not linked into any Rust
// artifact — only its successful compilation matters.

#include "oodf.h"

// The cxx-generated header carries the declarations for `rust::String`,
// `rust::Vec`, and `rust::Slice` that the public API uses (transitively
// included via `oodf.h` — include it explicitly here to keep the
// self-containment check honest).
#include "oodf-ffi/src/bridge.rs.h"

namespace {

// Instantiate the public types in a trivial context so that any missing
// definitions surface as compile errors here rather than deep in Epic 3's
// bridge code.
[[maybe_unused]] oodf::ParseResult make_parse_result() {
    oodf::ParseResult result{};
    result.error = oodf::ParseErrorKind::Ok;
    return result;
}

[[maybe_unused]] oodf::StructCommand make_struct_command() {
    oodf::StructCommand cmd{};
    cmd.tag = oodf::CommandTag::NONE;
    return cmd;
}

[[maybe_unused]] oodf::OwnedDelta make_owned_delta() {
    return oodf::OwnedDelta{};
}

}  // namespace
