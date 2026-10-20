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

#pragma once

// Shared infrastructure for the oodf-ffi C++ test harness: the `TEST_CASE`
// macro, `EXPECT_*` assertions, and the `semantic_equal` helper used by the
// round-trip suite. The harness itself (main() + the registry vector) lives
// in `test_main.cpp`. Split this way so subtask 2.4.B/C TUs can `#include`
// the macros without dragging in `main()`.

#include "oodf.h"

#include <cstddef>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace oodf_test {

struct TestCase {
    const char* name;
    void (*fn)();
};

// Global registry shared across every TU in the test binary. C++17 inline
// guarantees all calls resolve to one function with one static local, so the
// registry is unique even though this header is included in many TUs.
inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> r;
    return r;
}

struct Registrar {
    Registrar(const char* name, void (*fn)()) {
        registry().push_back(TestCase{name, fn});
    }
};

// Assertion helper — throws std::runtime_error so the harness's per-case
// catch block turns the failure into a formatted diagnostic. We deliberately
// avoid <cassert>/abort so that failure output reaches stderr before the
// process dies.
[[noreturn]] inline void fail(const char* file, int line, const std::string& msg) {
    std::string full = std::string(file) + ":" + std::to_string(line) + ": " + msg;
    throw std::runtime_error(full);
}

// ── semantic_equal ──────────────────────────────────────────────────────────
// Field-by-field, vector-element-by-vector-element equality between two
// `OwnedDelta` values. Vectors are ORDER-SENSITIVE: the OODF wire format
// preserves the authoring order of structCommands / setFields / timeSamples,
// and the diff-replay engine relies on that ordering. Do NOT substitute a
// set-based or multiset-based comparison here; divergent orderings must fail
// this check.

inline bool bytes_equal(const rust::Vec<uint8_t>& a, const rust::Vec<uint8_t>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

inline bool u64s_equal(const rust::Vec<uint64_t>& a, const rust::Vec<uint64_t>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

inline bool string_equal(const rust::String& a, const rust::String& b) {
    return std::string(a) == std::string(b);
}

inline bool semantic_equal_cmd(const oodf::StructCommand& a, const oodf::StructCommand& b) {
    if (a.tag != b.tag) return false;
    switch (a.tag) {
        case oodf::CommandTag::NONE:
            return true;
        case oodf::CommandTag::CreateSection:
            return a.create_section.parent_id == b.create_section.parent_id &&
                   a.create_section.section_id == b.create_section.section_id &&
                   string_equal(a.create_section.section_name, b.create_section.section_name) &&
                   a.create_section.section_type == b.create_section.section_type;
        case oodf::CommandTag::DeleteSection:
            return a.delete_section.parent_id == b.delete_section.parent_id &&
                   a.delete_section.section_id == b.delete_section.section_id;
        case oodf::CommandTag::MoveSection:
            return a.move_section.old_parent_id == b.move_section.old_parent_id &&
                   a.move_section.new_parent_id == b.move_section.new_parent_id &&
                   a.move_section.section_id == b.move_section.section_id &&
                   string_equal(a.move_section.new_name, b.move_section.new_name);
        case oodf::CommandTag::ReorderChildren:
            return a.reorder_children.section_id == b.reorder_children.section_id &&
                   a.reorder_children.children_list_id == b.reorder_children.children_list_id &&
                   u64s_equal(a.reorder_children.children_list, b.reorder_children.children_list);
        case oodf::CommandTag::DiffChangeSection:
            return a.diff_change_section.parent_id == b.diff_change_section.parent_id &&
                   a.diff_change_section.section_id == b.diff_change_section.section_id &&
                   string_equal(a.diff_change_section.section_name, b.diff_change_section.section_name) &&
                   a.diff_change_section.section_type == b.diff_change_section.section_type &&
                   a.diff_change_section.section_order == b.diff_change_section.section_order;
        case oodf::CommandTag::DiffDeleteSection:
            return a.diff_delete_section.section_id == b.diff_delete_section.section_id;
    }
    return false;
}

inline bool semantic_equal_set_field(const oodf::SetField& a, const oodf::SetField& b) {
    return a.section_id == b.section_id &&
           string_equal(a.key_name, b.key_name) &&
           bytes_equal(a.value_or_ext_hash, b.value_or_ext_hash) &&
           a.ext_value_size == b.ext_value_size &&
           a.ext_value_index == b.ext_value_index &&
           a.set_order == b.set_order;
}

inline bool semantic_equal_time_sample(const oodf::TimeSample& a, const oodf::TimeSample& b) {
    return a.section_id == b.section_id &&
           a.time == b.time &&
           bytes_equal(a.value_or_ext_hash, b.value_or_ext_hash) &&
           a.ext_value_size == b.ext_value_size &&
           a.ext_value_index == b.ext_value_index;
}

inline bool semantic_equal(const oodf::OwnedDelta& a, const oodf::OwnedDelta& b) {
    if (a.is_diff != b.is_diff) return false;
    if (a.base_version != b.base_version) return false;
    if (!string_equal(a.source_format, b.source_format)) return false;
    if (a.struct_commands.size() != b.struct_commands.size()) return false;
    for (size_t i = 0; i < a.struct_commands.size(); ++i) {
        if (!semantic_equal_cmd(a.struct_commands[i], b.struct_commands[i])) return false;
    }
    if (a.set_fields.size() != b.set_fields.size()) return false;
    for (size_t i = 0; i < a.set_fields.size(); ++i) {
        if (!semantic_equal_set_field(a.set_fields[i], b.set_fields[i])) return false;
    }
    if (a.time_samples.size() != b.time_samples.size()) return false;
    for (size_t i = 0; i < a.time_samples.size(); ++i) {
        if (!semantic_equal_time_sample(a.time_samples[i], b.time_samples[i])) return false;
    }
    return true;
}

}  // namespace oodf_test

// TEST_CASE(name) — declares a per-TU function + self-registers it into the
// shared registry. Use unique names per TU; the macro mangles the function
// symbol as `test_<name>` so two TUs with the same `TEST_CASE(foo)` would
// cause a link error (a feature, not a bug — tests should not collide).
#define TEST_CASE(name)                                                           \
    static void test_##name();                                                    \
    static ::oodf_test::Registrar registrar_##name(#name, test_##name);           \
    static void test_##name()

#define EXPECT_TRUE(cond)                                                         \
    do {                                                                          \
        if (!(cond)) {                                                            \
            ::oodf_test::fail(__FILE__, __LINE__,                                 \
                              std::string("EXPECT_TRUE failed: ") + #cond);       \
        }                                                                         \
    } while (0)

#define EXPECT_EQ(a, b)                                                           \
    do {                                                                          \
        auto _a = (a);                                                            \
        auto _b = (b);                                                            \
        if (!(_a == _b)) {                                                        \
            ::oodf_test::fail(__FILE__, __LINE__,                                 \
                              std::string("EXPECT_EQ failed: ") + #a " == " #b);  \
        }                                                                         \
    } while (0)
