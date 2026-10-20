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

// Hand-rolled test runner for the oodf-ffi C++ standalone tests.
//
// The shared infrastructure (TEST_CASE / EXPECT_* macros, registry(), the
// `semantic_equal` helper) lives in `test_main.h`. This file owns only main()
// — every per-variant / error-path test case lives in a sibling TU
// (errors.cpp, round_trip.cpp, …). The Rust wrapper at
// `oodf-ffi/tests/cpp_tests.rs` compiles all `*.cpp` files in this directory
// (except `header_compiles.cpp`) into a single binary and checks its exit
// code.
//
// ── Coverage map ────────────────────────────────────────────────────────────
// Every non-Ok ParseErrorKind variant has an error-path test in errors.cpp;
// every CommandTag variant (except NONE, which `build_delta` rejects via
// `BuildErrorKind::UnknownCommandTag` since b.vvs item 8) has a round-trip
// test in round_trip.cpp:
//
//   ParseErrorKind::EmptyInput           → error_empty_input_returns_EmptyInput
//   ParseErrorKind::TooShort             → error_too_short_one_byte, …_seven_bytes
//   ParseErrorKind::BadMagic             → error_bad_magic_wrong_file_identifier
//   ParseErrorKind::VerifierFailed       → error_verifier_rejects_{truncated_buffer,
//                                               corrupted_root_offset,
//                                               oversized_vector_length}
//   ParseErrorKind::UnknownCommandType   → error_unknown_command_type_discriminant
//   ParseErrorKind::AllocationFailed     → (rationale in errors.cpp — not
//                                           exercised here)
//   ParseErrorKind::InternalError        → (rationale in errors.cpp — not
//                                           exercised here)
//   ParseErrorKind::InvalidUtf8          → (covered by Rust-side tests; the
//                                           explicit UTF-8 pre-check in
//                                           oodf.cpp is exercised by
//                                           malformed_input.rs and the
//                                           oodf-ffi proptests)
//
//   CommandTag::CreateSection            → round_trip_create_section_variant
//   CommandTag::DeleteSection            → round_trip_delete_section_variant
//   CommandTag::MoveSection              → round_trip_move_section_variant
//   CommandTag::ReorderChildren          → round_trip_reorder_children_variant
//   CommandTag::DiffChangeSection        → round_trip_diff_change_section_variant
//   CommandTag::DiffDeleteSection        → round_trip_diff_delete_section_variant
//
// Additional positive-path tests: round_trip_set_field_heavy_*,
// round_trip_time_sample_heavy_*, round_trip_mixed_struct_command_variants_*,
// round_trip_postman_regression_*, wire_layout_file_identifier_byte_check,
// wire_layout_split_structcommands_vtable_slots.

#include "test_main.h"

#include <cstdio>
#include <cstdlib>
#include <exception>

int main() {
    auto& cases = ::oodf_test::registry();
    std::printf("oodf-ffi cpp test harness: running %zu test(s)\n", cases.size());
    std::fflush(stdout);

    size_t passed = 0;
    size_t failed = 0;
    for (const auto& tc : cases) {
        std::printf("[ RUN      ] %s\n", tc.name);
        std::fflush(stdout);
        try {
            tc.fn();
            std::printf("[       OK ] %s\n", tc.name);
            ++passed;
        } catch (const std::exception& e) {
            std::printf("[  FAILED  ] %s\n    %s\n", tc.name, e.what());
            ++failed;
            // Fail fast: stop at the first failure so the diagnostic is not
            // buried under cascading failures.
            std::fflush(stdout);
            std::fprintf(stderr,
                         "oodf-ffi cpp test harness: aborting after first failure in %s\n",
                         tc.name);
            return 1;
        } catch (...) {
            std::printf("[  FAILED  ] %s (unknown exception)\n", tc.name);
            ++failed;
            std::fflush(stdout);
            return 1;
        }
        std::fflush(stdout);
    }

    std::printf("oodf-ffi cpp test harness: %zu passed, %zu failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
