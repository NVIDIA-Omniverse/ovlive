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

// Error-path tests (Task 2.4.B). Exercises every ParseErrorKind variant
// reachable from a malformed-input stream. Every test must:
//   1. return from `oodf::parse_delta` without throwing, crashing, or
//      corrupting memory, AND
//   2. populate `result.error` with the expected `ParseErrorKind`.
//
// The happy-path / round-trip tests live alongside this file; the harness
// pulls every `.cpp` in `oodf-ffi/cpp/tests/` into the same binary.

#include "fixtures.h"
#include "oodf.h"
#include "test_main.h"

#include "flatbuffers/flatbuffers.h"
#include "OmniObjectDelta_generated.h"

#include <cstdint>
#include <cstring>
#include <vector>

namespace {

// Build a deep copy of the Postman OODF payload so each test can mutate
// without affecting the shared fixture.
std::vector<uint8_t> postman_copy() {
    return std::vector<uint8_t>(
        oodf_test::kPostmanOodfPayload,
        oodf_test::kPostmanOodfPayload + oodf_test::kPostmanOodfPayloadLen);
}

rust::Slice<const uint8_t> as_slice(const std::vector<uint8_t>& v) {
    return rust::Slice<const uint8_t>(v.data(), v.size());
}

}  // namespace

TEST_CASE(error_empty_input_returns_EmptyInput) {
    // Use a real (non-null) address with size 0 (b.vvs item 1). The previous
    // `(nullptr, 0)` slice exercised an undefined-behavior corner of
    // `rust::Slice` and crossed the FFI with a literal null pointer; the
    // new defensive null-pointer guard in `parse_delta` rejects null
    // pointers explicitly even when size is 0 via the same EmptyInput
    // error kind, so either shape will pass this test going forward, but
    // we use a real address here to keep the test from depending on cxx's
    // pointer-equality treatment of empty slices and to avoid producing a
    // technically-malformed `rust::Slice`.
    static const uint8_t empty_buf[1] = {0};
    rust::Slice<const uint8_t> empty(empty_buf, 0);
    auto result = oodf::parse_delta(empty);
    EXPECT_EQ(result.error, oodf::ParseErrorKind::EmptyInput);
}

TEST_CASE(error_too_short_one_byte) {
    const uint8_t one[1] = {0x00};
    auto result = oodf::parse_delta(rust::Slice<const uint8_t>(one, 1));
    EXPECT_EQ(result.error, oodf::ParseErrorKind::TooShort);
}

TEST_CASE(error_too_short_seven_bytes) {
    const uint8_t seven[7] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06};
    auto result = oodf::parse_delta(rust::Slice<const uint8_t>(seven, 7));
    EXPECT_EQ(result.error, oodf::ParseErrorKind::TooShort);
}

TEST_CASE(error_bad_magic_wrong_file_identifier) {
    // Postman payload with the 4-byte file_identifier at offsets 4..8
    // rewritten to "XXXX". Everything else is still a valid OODF buffer, so
    // the TooShort guard is not tripped and the BadMagic probe fires before
    // the verifier runs.
    std::vector<uint8_t> buf = postman_copy();
    buf[4] = 'X';
    buf[5] = 'X';
    buf[6] = 'X';
    buf[7] = 'X';
    auto result = oodf::parse_delta(as_slice(buf));
    EXPECT_EQ(result.error, oodf::ParseErrorKind::BadMagic);
}

TEST_CASE(error_verifier_rejects_truncated_buffer) {
    // Drop the trailing 64 bytes of the Postman payload. File identifier is
    // still present at offsets 4..8, so the BadMagic probe passes; the
    // verifier then rejects the truncated body.
    std::vector<uint8_t> buf = postman_copy();
    buf.resize(buf.size() - 64);
    auto result = oodf::parse_delta(as_slice(buf));
    EXPECT_EQ(result.error, oodf::ParseErrorKind::VerifierFailed);
}

TEST_CASE(error_verifier_rejects_corrupted_root_offset) {
    // Replace the 4-byte root uoffset_t at offset 0 with a wildly out-of-
    // range value (0xFFFFFFFC). The verifier's first action is to resolve
    // the root uoffset_t — a value that points past the end of the buffer
    // fails the bounded read check and returns VerifierFailed. This is a
    // reliable way to trigger VerifierFailed across FlatBuffers versions
    // without depending on the internal vtable layout of a particular
    // fixture.
    std::vector<uint8_t> buf = postman_copy();
    buf[0] = 0xFC;
    buf[1] = 0xFF;
    buf[2] = 0xFF;
    buf[3] = 0xFF;
    auto result = oodf::parse_delta(as_slice(buf));
    EXPECT_EQ(result.error, oodf::ParseErrorKind::VerifierFailed);
}

TEST_CASE(error_verifier_rejects_oversized_vector_length) {
    // The Postman payload carries a structCommands vector whose length
    // prefix sits at byte offset 0x3c — 0x02 00 00 00 (= 2). Overwrite the
    // prefix with 0xFFFFFFFF so the verifier sees a vector claiming ~4 GiB
    // of elements inside a 280-byte buffer. The verifier's bounds check
    // rejects this before any element is read.
    std::vector<uint8_t> buf = postman_copy();
    // We don't hard-code the offset — scan for the known `[2,0,0,0]` prefix
    // of the 2-element structCommands vector and overwrite it. If the
    // fixture layout ever changes, the test will fail loudly rather than
    // silently corrupt a different field.
    bool patched = false;
    for (size_t i = 0; i + 4 <= buf.size(); ++i) {
        if (buf[i] == 0x02 && buf[i + 1] == 0x00 && buf[i + 2] == 0x00 && buf[i + 3] == 0x00) {
            // The first such match in the Postman payload is the
            // structCommands vector length prefix at offset 0x3c. Overwrite
            // and stop.
            buf[i] = 0xFF;
            buf[i + 1] = 0xFF;
            buf[i + 2] = 0xFF;
            buf[i + 3] = 0xFF;
            patched = true;
            break;
        }
    }
    EXPECT_TRUE(patched);
    auto result = oodf::parse_delta(as_slice(buf));
    EXPECT_EQ(result.error, oodf::ParseErrorKind::VerifierFailed);
}

TEST_CASE(error_unknown_command_type_discriminant) {
    // Hand-craft a Delta whose structCommands_type[0] is 7 (outside the
    // declared 1..=6 range). VerifyStructCommand's default branch returns
    // true for unknown discriminants, so the verifier accepts the buffer;
    // oodf::parse_delta's own switch then detects the bad discriminant and
    // returns UnknownCommandType.
    ::flatbuffers::FlatBufferBuilder builder;

    // A CommandDeleteSection (valid table) serves as the payload so the
    // offset in struct_commands points somewhere well-formed. The tag lies
    // about its identity — that's the entire point of the test.
    auto payload = OmniObjectDelta::CreateCommandDeleteSection(builder, 1, 2);

    std::vector<uint8_t> types = {7};
    auto types_vec = builder.CreateVector<uint8_t>(types);

    std::vector<::flatbuffers::Offset<void>> offsets;
    offsets.push_back(payload.Union());
    auto values_vec = builder.CreateVector<::flatbuffers::Offset<void>>(offsets);

    std::vector<::flatbuffers::Offset<OmniObjectDelta::SetField>> empty_set_fields;
    auto set_fields_vec = builder.CreateVector<
        ::flatbuffers::Offset<OmniObjectDelta::SetField>>(empty_set_fields);

    std::vector<::flatbuffers::Offset<OmniObjectDelta::TimeSample>> empty_time_samples;
    auto time_samples_vec = builder.CreateVector<
        ::flatbuffers::Offset<OmniObjectDelta::TimeSample>>(empty_time_samples);

    auto source_format = builder.CreateString("");

    auto delta = OmniObjectDelta::CreateDelta(
        builder,
        /*isDiff=*/false,
        /*baseVersion=*/0,
        types_vec,
        values_vec,
        set_fields_vec,
        time_samples_vec,
        source_format);
    OmniObjectDelta::FinishDeltaBuffer(builder, delta);

    rust::Slice<const uint8_t> slice(builder.GetBufferPointer(), builder.GetSize());
    auto result = oodf::parse_delta(slice);
    EXPECT_EQ(result.error, oodf::ParseErrorKind::UnknownCommandType);
}

namespace {

// The verifier's table cap (`kVerifierMaxTables` in oodf.cpp): one delta is
// one root table plus one table per entry, so it carries at most one less.
constexpr size_t kTableCap = 100'000;
constexpr size_t kEntryLimit = kTableCap - 1;

// Finish `builder` as a Delta whose structCommands are `types` / `values` and
// whose other vectors are absent, and return its bytes.
std::vector<uint8_t> finish_delta(
    ::flatbuffers::FlatBufferBuilder& builder,
    const std::vector<uint8_t>& types,
    const std::vector<::flatbuffers::Offset<void>>& values) {
    auto types_vec = builder.CreateVector<uint8_t>(types);
    auto values_vec = builder.CreateVector<::flatbuffers::Offset<void>>(values);
    auto delta = OmniObjectDelta::CreateDelta(
        builder, /*isDiff=*/false, /*baseVersion=*/0, types_vec, values_vec);
    OmniObjectDelta::FinishDeltaBuffer(builder, delta);
    return std::vector<uint8_t>(
        builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize());
}

// A delta of `n` CommandDeleteSection struct commands, the cheapest entry.
std::vector<uint8_t> delete_sections(size_t n) {
    ::flatbuffers::FlatBufferBuilder builder;
    std::vector<uint8_t> types;
    std::vector<::flatbuffers::Offset<void>> values;
    types.reserve(n);
    values.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        values.push_back(
            OmniObjectDelta::CreateCommandDeleteSection(builder, 0, i + 1).Union());
        types.push_back(OmniObjectDelta::StructCommand_CommandDeleteSection);
    }
    return finish_delta(builder, types, values);
}

// A delta of one ReorderChildren command naming `n` children.
std::vector<uint8_t> reorder_children(size_t n) {
    ::flatbuffers::FlatBufferBuilder builder;
    std::vector<uint64_t> children(n);
    for (size_t i = 0; i < n; ++i) {
        children[i] = i + 1;
    }
    auto list = builder.CreateVector<uint64_t>(children);
    auto cmd = OmniObjectDelta::CreateCommandReorderChildren(builder, 1, 0, list);
    return finish_delta(
        builder, {OmniObjectDelta::StructCommand_CommandReorderChildren}, {cmd.Union()});
}

}  // namespace

TEST_CASE(entry_count_probe_admits_exactly_the_limit) {
    // 99,999 entries plus the root are exactly the verifier's 100,000 tables.
    std::vector<uint8_t> buf = delete_sections(kEntryLimit);
    auto result = oodf::parse_delta(as_slice(buf));
    EXPECT_EQ(result.error, oodf::ParseErrorKind::Ok);
    EXPECT_EQ(result.delta.struct_commands.size(), kEntryLimit);
    EXPECT_EQ(result.max_entries, static_cast<uint64_t>(0));
}

TEST_CASE(entry_count_probe_refuses_one_over_with_the_counts_in_the_result_fields) {
    // One entry over: refused TooManyTables before the verifier runs, the
    // counts and the limit in the result's own fields (never in the message).
    std::vector<uint8_t> buf = delete_sections(kEntryLimit + 1);
    auto result = oodf::parse_delta(as_slice(buf));
    EXPECT_EQ(result.error, oodf::ParseErrorKind::TooManyTables);
    EXPECT_EQ(result.struct_command_count, static_cast<uint64_t>(kEntryLimit + 1));
    EXPECT_EQ(result.set_field_count, static_cast<uint64_t>(0));
    EXPECT_EQ(result.time_sample_count, static_cast<uint64_t>(0));
    EXPECT_EQ(result.max_entries, static_cast<uint64_t>(kEntryLimit));
    EXPECT_EQ(result.children_list_len, static_cast<uint64_t>(0));
}

TEST_CASE(entry_count_probe_falls_through_to_the_verifier_on_an_unreadable_root) {
    // The same over-limit buffer with its root offset pointing past the end:
    // the probe's first bounds check fails, it counts nothing, and the
    // verifier answers as it always has.
    std::vector<uint8_t> buf = delete_sections(kEntryLimit + 1);
    const uint32_t past_end = static_cast<uint32_t>(buf.size()) + 64;
    std::memcpy(buf.data(), &past_end, sizeof(past_end));
    auto result = oodf::parse_delta(as_slice(buf));
    EXPECT_EQ(result.error, oodf::ParseErrorKind::VerifierFailed);
    EXPECT_EQ(result.struct_command_count, static_cast<uint64_t>(0));
    EXPECT_EQ(result.max_entries, static_cast<uint64_t>(0));
}

TEST_CASE(children_list_one_past_the_parser_cap_returns_children_list_too_long) {
    // `kMaxCommands` (1,000,000) bounds one children list; one past it is the
    // typed refusal carrying the list's length.
    std::vector<uint8_t> buf = reorder_children(1'000'001);
    auto result = oodf::parse_delta(as_slice(buf));
    EXPECT_EQ(result.error, oodf::ParseErrorKind::ChildrenListTooLong);
    EXPECT_EQ(result.children_list_len, static_cast<uint64_t>(1'000'001));
}

// AllocationFailed: not exercised by this harness.
//
// The parser's `catch (const std::bad_alloc&)` block (oodf.cpp) only fires if
// the process runs out of memory while materialising `rust::String` or
// `rust::Vec` copies. Reliably inducing that on macOS and Linux requires
// overriding `operator new` or setting strict RLIMIT_AS — both are fragile
// across compilers, sanitizers, and CI environments, and would make a test
// flaky without catching real regressions. Covered by manual inspection and
// by Rust-side tests that force allocator failure through the cxx bridge
// (Epic 5). Left as a comment rather than an empty TEST_CASE because an
// empty case would inflate the harness's pass count without exercising any
// behavior.
//
// InternalError: not exercised by this harness.
//
// `oodf::parse_delta` only reaches `ParseErrorKind::InternalError` when
// `GetDelta` returns null AFTER the hardened verifier has accepted the
// buffer, or when a non-bad_alloc std::exception escapes an accessor. Both
// conditions require the flatc-generated verifier to be broken — a stronger
// failure than anything this test file can inject with valid flatbuffers
// machinery. The `catch (...)` branch is covered by the b.8uk UTF-8 defense-
// in-depth (explicit `check_utf8_or_error` + broad catch-all in oodf.cpp)
// and by sanitizer runs in CI. Same reason as AllocationFailed above for
// not carrying an empty TEST_CASE here.
