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

// Round-trip tests (Task 2.4.C). Positive-path regression net for every
// CommandTag variant plus SetField / TimeSample coverage, the Postman
// fixture, and byte-level wire-layout checks.
//
// Each test constructs an OwnedDelta with a known shape, calls build_delta,
// parses the resulting bytes back, and asserts `semantic_equal` between the
// original and the reparse. The file_identifier and vtable-layout checks run
// alongside the happy-path round trips to lock in drop-in compatibility
// with the legacy C++ live-edit service.
//
// Coverage map (CommandTag variant ↔ test name):
//     CreateSection     → round_trip_create_section_variant
//     DeleteSection     → round_trip_delete_section_variant
//     MoveSection       → round_trip_move_section_variant
//     ReorderChildren   → round_trip_reorder_children_variant
//     DiffChangeSection → round_trip_diff_change_section_variant
//     DiffDeleteSection → round_trip_diff_delete_section_variant
// The NONE discriminant is never produced by build_delta (see oodf.cpp's
// switch statement — since b.vvs item 8 it surfaces as a
// `BuildErrorKind::UnknownCommandTag` BuildResult instead of `std::abort()`)
// so it does not appear here.

#include "fixtures.h"
#include "oodf.h"
#include "test_main.h"

#include <cstdint>
#include <cstring>
#include <vector>

namespace {

rust::Slice<const uint8_t> buf_slice(const rust::Vec<uint8_t>& v) {
    return rust::Slice<const uint8_t>(v.data(), v.size());
}

// Construct a byte Vec from a std::initializer_list. `rust::Vec<u8>` has a
// runtime `cxxbridge1$rust_vec$u8$*` implementation provided by the cxx
// crate (primitive specialization), so unlike the oodf types this "just
// works" without test-only shims.
rust::Vec<uint8_t> make_bytes(std::initializer_list<uint8_t> bytes) {
    rust::Vec<uint8_t> out;
    out.reserve(bytes.size());
    for (uint8_t b : bytes) {
        out.push_back(b);
    }
    return out;
}

// Build a minimal OwnedDelta carrying exactly one StructCommand of the named
// variant. Other fields default to sensible zeros so round-trips focus on
// that variant.
oodf::OwnedDelta make_single_create_section() {
    oodf::StructCommand cmd{};
    cmd.tag = oodf::CommandTag::CreateSection;
    cmd.create_section.parent_id = 1;
    cmd.create_section.section_id = 42;
    cmd.create_section.section_name = rust::String("PrimCreate");
    cmd.create_section.section_type = 3;

    oodf::OwnedDelta d{};
    d.is_diff = false;
    d.base_version = 7;
    d.source_format = rust::String("");
    d.struct_commands.push_back(std::move(cmd));
    return d;
}

oodf::OwnedDelta make_single_delete_section() {
    oodf::StructCommand cmd{};
    cmd.tag = oodf::CommandTag::DeleteSection;
    cmd.delete_section.parent_id = 9;
    cmd.delete_section.section_id = 17;

    oodf::OwnedDelta d{};
    d.is_diff = false;
    d.base_version = 0;
    d.source_format = rust::String("");
    d.struct_commands.push_back(std::move(cmd));
    return d;
}

oodf::OwnedDelta make_single_move_section() {
    oodf::StructCommand cmd{};
    cmd.tag = oodf::CommandTag::MoveSection;
    cmd.move_section.old_parent_id = 1;
    cmd.move_section.new_parent_id = 2;
    cmd.move_section.section_id = 99;
    cmd.move_section.new_name = rust::String("renamed");

    oodf::OwnedDelta d{};
    d.is_diff = false;
    d.base_version = 0;
    d.source_format = rust::String("");
    d.struct_commands.push_back(std::move(cmd));
    return d;
}

oodf::OwnedDelta make_single_reorder_children() {
    oodf::StructCommand cmd{};
    cmd.tag = oodf::CommandTag::ReorderChildren;
    cmd.reorder_children.section_id = 1;
    cmd.reorder_children.children_list_id = 2;
    // Non-trivial vector to exercise the flatbuffers vector serialization.
    cmd.reorder_children.children_list.push_back(10);
    cmd.reorder_children.children_list.push_back(20);
    cmd.reorder_children.children_list.push_back(30);

    oodf::OwnedDelta d{};
    d.is_diff = false;
    d.base_version = 0;
    d.source_format = rust::String("");
    d.struct_commands.push_back(std::move(cmd));
    return d;
}

oodf::OwnedDelta make_single_diff_change() {
    oodf::StructCommand cmd{};
    cmd.tag = oodf::CommandTag::DiffChangeSection;
    cmd.diff_change_section.parent_id = 4;
    cmd.diff_change_section.section_id = 55;
    cmd.diff_change_section.section_name = rust::String("PrimDiff");
    cmd.diff_change_section.section_type = 1;
    cmd.diff_change_section.section_order = 7;

    oodf::OwnedDelta d{};
    d.is_diff = true;
    d.base_version = 5;
    d.source_format = rust::String("usd");
    d.struct_commands.push_back(std::move(cmd));
    return d;
}

oodf::OwnedDelta make_single_diff_delete() {
    oodf::StructCommand cmd{};
    cmd.tag = oodf::CommandTag::DiffDeleteSection;
    cmd.diff_delete_section.section_id = 123;

    oodf::OwnedDelta d{};
    d.is_diff = true;
    d.base_version = 5;
    d.source_format = rust::String("usd");
    d.struct_commands.push_back(std::move(cmd));
    return d;
}

// Shared round-trip driver: build → check OODF magic → parse → semantic_equal.
//
// build_delta now returns a `BuildResult` (b.vvs item 8). Extract `.bytes`
// and require `.error == Ok` for the happy-path tests.
void round_trip(const oodf::OwnedDelta& input) {
    auto built = oodf::build_delta(input);
    EXPECT_EQ(built.error, oodf::BuildErrorKind::Ok);
    const auto& bytes = built.bytes;
    EXPECT_TRUE(bytes.size() >= 8);
    // Bytes 4..8 carry the FlatBuffers file_identifier. This is what
    // DeltaBufferHasIdentifier checks on the parse side.
    EXPECT_EQ(static_cast<char>(bytes[4]), 'O');
    EXPECT_EQ(static_cast<char>(bytes[5]), 'O');
    EXPECT_EQ(static_cast<char>(bytes[6]), 'D');
    EXPECT_EQ(static_cast<char>(bytes[7]), 'F');

    auto result = oodf::parse_delta(buf_slice(bytes));
    EXPECT_EQ(result.error, oodf::ParseErrorKind::Ok);
    EXPECT_TRUE(::oodf_test::semantic_equal(input, result.delta));
}

}  // namespace

TEST_CASE(round_trip_create_section_variant) {
    round_trip(make_single_create_section());
}

TEST_CASE(round_trip_delete_section_variant) {
    round_trip(make_single_delete_section());
}

TEST_CASE(round_trip_move_section_variant) {
    round_trip(make_single_move_section());
}

TEST_CASE(round_trip_reorder_children_variant) {
    round_trip(make_single_reorder_children());
}

TEST_CASE(round_trip_diff_change_section_variant) {
    round_trip(make_single_diff_change());
}

TEST_CASE(round_trip_diff_delete_section_variant) {
    round_trip(make_single_diff_delete());
}

TEST_CASE(round_trip_set_field_heavy_inline_and_external_mix) {
    oodf::OwnedDelta d{};
    d.is_diff = false;
    d.base_version = 0;
    d.source_format = rust::String("");

    // Inline SetField: payload bytes live directly in valueOrExtHash.
    // extValueSize == 0 signals inline, per the schema comment.
    oodf::SetField inline_a{};
    inline_a.section_id = 1;
    inline_a.key_name = rust::String("inline_a");
    inline_a.value_or_ext_hash = make_bytes({0x01, 0x02, 0x03});
    inline_a.ext_value_size = 0;
    inline_a.ext_value_index = 0;
    inline_a.set_order = 1;
    d.set_fields.push_back(std::move(inline_a));

    oodf::SetField inline_b{};
    inline_b.section_id = 1;
    inline_b.key_name = rust::String("inline_b");
    inline_b.value_or_ext_hash = make_bytes({0xde, 0xad, 0xbe, 0xef});
    inline_b.ext_value_size = 0;
    inline_b.ext_value_index = 0;
    inline_b.set_order = 2;
    d.set_fields.push_back(std::move(inline_b));

    // External SetField: valueOrExtHash is a 32-byte SHA-256 hash, and
    // extValueSize / extValueIndex describe the external blob.
    oodf::SetField external{};
    external.section_id = 2;
    external.key_name = rust::String("ext_large");
    external.value_or_ext_hash = rust::Vec<uint8_t>();
    external.value_or_ext_hash.reserve(32);
    for (uint8_t i = 0; i < 32; ++i) {
        external.value_or_ext_hash.push_back(static_cast<uint8_t>(i * 7u + 1u));
    }
    external.ext_value_size = 1024 * 1024;
    external.ext_value_index = 42;
    external.set_order = 3;
    d.set_fields.push_back(std::move(external));

    round_trip(d);
}

TEST_CASE(round_trip_time_sample_heavy_inline_and_external_mix) {
    oodf::OwnedDelta d{};
    d.is_diff = false;
    d.base_version = 0;
    d.source_format = rust::String("");

    oodf::TimeSample inline_a{};
    inline_a.section_id = 11;
    inline_a.time = 0.0;
    inline_a.value_or_ext_hash = make_bytes({0xaa});
    inline_a.ext_value_size = 0;
    inline_a.ext_value_index = 0;
    d.time_samples.push_back(std::move(inline_a));

    oodf::TimeSample inline_b{};
    inline_b.section_id = 11;
    inline_b.time = 1.5;
    inline_b.value_or_ext_hash = make_bytes({0xbb, 0xcc});
    inline_b.ext_value_size = 0;
    inline_b.ext_value_index = 0;
    d.time_samples.push_back(std::move(inline_b));

    oodf::TimeSample external{};
    external.section_id = 12;
    external.time = 2.25;
    external.value_or_ext_hash = rust::Vec<uint8_t>();
    external.value_or_ext_hash.reserve(32);
    for (uint8_t i = 0; i < 32; ++i) {
        external.value_or_ext_hash.push_back(static_cast<uint8_t>(0xf0u ^ i));
    }
    external.ext_value_size = 1 << 20;
    external.ext_value_index = 9;
    d.time_samples.push_back(std::move(external));

    round_trip(d);
}

TEST_CASE(round_trip_mixed_struct_command_variants_in_one_delta) {
    oodf::OwnedDelta d{};
    d.is_diff = false;
    d.base_version = 0;
    d.source_format = rust::String("");

    // One StructCommand of every non-NONE CommandTag variant, in the order
    // they are declared in include/oodf.h. Vectors are order-sensitive, so
    // this pins the ordering through build → parse.
    {
        oodf::StructCommand c{};
        c.tag = oodf::CommandTag::CreateSection;
        c.create_section.parent_id = 1;
        c.create_section.section_id = 2;
        c.create_section.section_name = rust::String("c0");
        c.create_section.section_type = 1;
        d.struct_commands.push_back(std::move(c));
    }
    {
        oodf::StructCommand c{};
        c.tag = oodf::CommandTag::DeleteSection;
        c.delete_section.parent_id = 1;
        c.delete_section.section_id = 3;
        d.struct_commands.push_back(std::move(c));
    }
    {
        oodf::StructCommand c{};
        c.tag = oodf::CommandTag::MoveSection;
        c.move_section.old_parent_id = 1;
        c.move_section.new_parent_id = 4;
        c.move_section.section_id = 5;
        c.move_section.new_name = rust::String("moved");
        d.struct_commands.push_back(std::move(c));
    }
    {
        oodf::StructCommand c{};
        c.tag = oodf::CommandTag::ReorderChildren;
        c.reorder_children.section_id = 6;
        c.reorder_children.children_list_id = 1;
        c.reorder_children.children_list.push_back(100);
        c.reorder_children.children_list.push_back(200);
        d.struct_commands.push_back(std::move(c));
    }
    {
        oodf::StructCommand c{};
        c.tag = oodf::CommandTag::DiffChangeSection;
        c.diff_change_section.parent_id = 7;
        c.diff_change_section.section_id = 8;
        c.diff_change_section.section_name = rust::String("diffed");
        c.diff_change_section.section_type = 2;
        c.diff_change_section.section_order = 9;
        d.struct_commands.push_back(std::move(c));
    }
    {
        oodf::StructCommand c{};
        c.tag = oodf::CommandTag::DiffDeleteSection;
        c.diff_delete_section.section_id = 10;
        d.struct_commands.push_back(std::move(c));
    }

    round_trip(d);
}

TEST_CASE(round_trip_postman_regression_parses_and_survives_rebuild) {
    // First parse: the canonical Postman 280-byte OODF payload must parse
    // cleanly and surface exactly two DiffChangeSection commands, matching
    // the byte-level analysis in Bee b.duy's regression log.
    rust::Slice<const uint8_t> slice(oodf_test::kPostmanOodfPayload,
                                     oodf_test::kPostmanOodfPayloadLen);
    auto first = oodf::parse_delta(slice);
    EXPECT_EQ(first.error, oodf::ParseErrorKind::Ok);
    EXPECT_EQ(first.delta.struct_commands.size(), static_cast<std::size_t>(2));
    EXPECT_EQ(first.delta.struct_commands[0].tag,
              oodf::CommandTag::DiffChangeSection);
    EXPECT_EQ(first.delta.struct_commands[1].tag,
              oodf::CommandTag::DiffChangeSection);

    // Second half: build → parse → compare against the first parse.
    // FlatBuffers does not guarantee byte-for-byte equality under re-build
    // (offset ordering within a table can vary), so this is semantic
    // equality, not byte equality.
    auto rebuilt = oodf::build_delta(first.delta);
    EXPECT_EQ(rebuilt.error, oodf::BuildErrorKind::Ok);
    auto second = oodf::parse_delta(buf_slice(rebuilt.bytes));
    EXPECT_EQ(second.error, oodf::ParseErrorKind::Ok);
    EXPECT_TRUE(::oodf_test::semantic_equal(first.delta, second.delta));
}

TEST_CASE(wire_layout_file_identifier_byte_check) {
    // build_delta must emit the "OODF" file_identifier at bytes 4..8. This
    // is the contract parse_delta's BufferHasIdentifier check relies on,
    // and downstream C++ consumers gate on identically.
    auto delta = make_single_delete_section();
    auto built = oodf::build_delta(delta);
    EXPECT_EQ(built.error, oodf::BuildErrorKind::Ok);
    const auto& bytes = built.bytes;
    EXPECT_TRUE(bytes.size() >= 8);
    const uint8_t kOodf[4] = {'O', 'O', 'D', 'F'};
    EXPECT_EQ(0, std::memcmp(bytes.data() + 4, kOodf, 4));
}

TEST_CASE(wire_layout_split_structcommands_vtable_slots) {
    // Drop-in compatibility with the legacy C++ live-edit service depends on
    // the Delta table exposing TWO adjacent vtable slots for the
    // structCommands union vector: one for the type-vector
    // (structCommands_type) and one for the offset-vector (structCommands).
    // Rust flatc emits a single combined slot and is NOT wire-compatible.
    // We assert by walking the root uoffset_t to the root table's vtable
    // and confirming the expected entry count.
    //
    // FlatBuffers vtable layout (all little-endian):
    //   root_uoffset_t at buffer[0..4] → points to root table's soffset_t.
    //   root_soffset_t at root_table[0..4] → signed back-offset to vtable.
    //   vtable[0..2] = vtable length in bytes.
    //   vtable[2..4] = object (inline) size.
    //   vtable[4..] = per-field offsets (uint16_t each).
    //
    // The Delta table in OmniObjectDelta.fbs has 7 fields:
    //   isDiff, baseVersion, structCommands_type, structCommands,
    //   setFields, timeSamples, sourceFormat.
    // Any collapse of structCommands_type + structCommands into a single
    // slot would drop the vtable entry count, which this test would catch.

    auto delta = make_single_create_section();
    auto built = oodf::build_delta(delta);
    EXPECT_EQ(built.error, oodf::BuildErrorKind::Ok);
    const auto& bytes = built.bytes;
    EXPECT_TRUE(bytes.size() >= 32);

    auto read_u32 = [&](std::size_t off) {
        return static_cast<uint32_t>(bytes[off]) |
               (static_cast<uint32_t>(bytes[off + 1]) << 8) |
               (static_cast<uint32_t>(bytes[off + 2]) << 16) |
               (static_cast<uint32_t>(bytes[off + 3]) << 24);
    };
    auto read_u16 = [&](std::size_t off) {
        return static_cast<uint16_t>(bytes[off]) |
               (static_cast<uint16_t>(bytes[off + 1]) << 8);
    };

    const uint32_t root_uoffset = read_u32(0);
    EXPECT_TRUE(root_uoffset >= 4 && root_uoffset < bytes.size());

    // Root table starts at bytes[root_uoffset]. The first four bytes are a
    // signed soffset_t pointing BACK to the vtable.
    const uint32_t root_soffset = read_u32(root_uoffset);
    // Per FlatBuffers contract, soffset_t = vtable_offset - table_offset.
    // The vtable lives at `root_uoffset - soffset` (note: soffset is stored
    // as the absolute back-distance).
    EXPECT_TRUE(root_soffset <= root_uoffset);
    const std::size_t vtable_off = root_uoffset - root_soffset;
    EXPECT_TRUE(vtable_off + 4 <= bytes.size());

    const uint16_t vtable_len = read_u16(vtable_off);
    const uint16_t object_size = read_u16(vtable_off + 2);
    // vtable length is bytes, 2 bytes header + 2 bytes per field slot.
    // Delta has 7 fields → expect vtable_len == 2 + 2 + 2*7 = 18.
    // But trailing null slots may be trimmed by flatc; the floor we care
    // about is: enough slots to include BOTH structCommands_type (field 3)
    // AND structCommands (field 4). Slot offsets in bytes:
    //   field 0 (isDiff)            at vtable[4..6]
    //   field 1 (baseVersion)       at vtable[6..8]
    //   field 2 (structCommands_type) at vtable[8..10]
    //   field 3 (structCommands)    at vtable[10..12]
    //   field 4 (setFields)         at vtable[12..14]
    //   field 5 (timeSamples)       at vtable[14..16]
    //   field 6 (sourceFormat)      at vtable[16..18]
    EXPECT_TRUE(vtable_len >= 12);  // must include slots for fields 2 AND 3
    EXPECT_TRUE(object_size >= 4);  // sanity
    (void)object_size;

    const uint16_t slot_struct_cmds_type = read_u16(vtable_off + 8);
    const uint16_t slot_struct_cmds = read_u16(vtable_off + 10);
    // Both slots MUST be non-zero (present) for the input we built, which
    // contains a single StructCommand.
    EXPECT_TRUE(slot_struct_cmds_type != 0);
    EXPECT_TRUE(slot_struct_cmds != 0);
    // And they MUST be distinct — a collapsed single-slot layout would set
    // only one and leave the other at 0, which the previous asserts would
    // have caught, but an explicit inequality makes the intent obvious.
    EXPECT_TRUE(slot_struct_cmds_type != slot_struct_cmds);
}
