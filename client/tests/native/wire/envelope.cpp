// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "OmpfEnvelope.h"
#include <SHA256.h>
#include "schema/ObjectDeltaBuilder.h"
#include <cassert>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <optional>
#include <tuple>

using namespace omni::usd::live::wire;
using namespace OmniObjectDelta;
namespace {
void put(std::string& out, size_t at, uint64_t value, size_t count) {
    for (size_t i = 0; i < count; ++i) out[at + i] = char(value >> (8 * i));
}
std::string pack(const flatbuffers::FlatBufferBuilder& builder, const std::vector<std::string>& external = {}, size_t padding = 0) {
    const size_t count = external.size() + 1, header = 8 + count * 8;
    std::string result(header + padding, '\0');
    put(result, 0, 0x46504d4f, 4); put(result, 4, count, 4); put(result, 8, result.size(), 8);
    result.append(reinterpret_cast<const char*>(builder.GetBufferPointer()), builder.GetSize());
    for (size_t i = 0; i < external.size(); ++i) {
        put(result, 16 + i * 8, result.size(), 8); result += external[i];
    }
    return result;
}
std::string noop() {
    // Independent literal from the pinned provider's gen_noop_delta output.
    const uint32_t words[] = {0x46504d4f,1,16,0,24,0x46444f4f,0x00140010,0,0x00080004,0x0010000c,16,28,20,12,4,0,0,0,0};
    std::string result(sizeof(words), '\0');
    for (size_t i = 0; i < sizeof(words) / sizeof(words[0]); ++i) put(result, i * 4, words[i], 4);
    return result;
}
void bad(std::string_view input, const EnvelopeLimits& limits = {}, std::optional<ErrorKind> kind = {}) {
    bool rejected = false;
    try { (void)decodeEnvelope(input, limits); }
    catch (const DecodeError& error) { rejected = true; if (kind) assert(error.kind == *kind); }
    assert(rejected);
}
std::string built(const std::function<void(OmniObjectDeltaBuilder&)>& write, bool diff = false, uint64_t base = 0) {
    OmniObjectDeltaBuilder b; write(b); b.Finish(diff, base); return pack(b.fbb);
}
void headersAndNoop() {
    const auto valid = noop(); assert(valid.size() == 76);
    auto result = decodeEnvelope(valid);
    assert(!result.isDiff && result.baseVersion == 0 && result.fields.empty() && result.commands.empty() && result.samples.empty());
    for (size_t length = 0; length < valid.size(); ++length) bad(std::string_view(valid).substr(0, length));
    for (const auto& [offset, value, size] : std::vector<std::tuple<size_t, uint64_t, size_t>>{
        {0,0,4},{4,0,4},{4,UINT32_MAX,4},{8,0,8},{8,15,8},{8,76,8},{8,UINT64_MAX,8}}) {
        auto bytes = valid; put(bytes, offset, value, size); bad(bytes);
    }
    flatbuffers::FlatBufferBuilder absent;
    FinishDeltaBuffer(absent, CreateDelta(absent));
    auto padded = pack(absent, {}, 3); // No invented canonical-padding/alignment restriction.
    std::string unaligned = "x" + padded;
    assert(decodeEnvelope(std::string_view(unaligned).substr(1)).commands.empty());
    auto limits = EnvelopeLimits{}; limits.maxBytes = 75;
    bad(valid, limits, ErrorKind::ResourceLimit);
    auto shortPart = valid; put(shortPart, 4, 2, 4); put(shortPart, 8, 24, 8); put(shortPart, 16, 28, 8); bad(shortPart);
    const auto* table = reinterpret_cast<const flatbuffers::Table*>(GetDelta(valid.data() + 16));
    const auto vt = size_t(reinterpret_cast<const char*>(table->GetVTable()) - valid.data());
    auto emptyVtable = valid; put(emptyVtable, vt, 0, 2); bad(emptyVtable);
    auto shortObject = valid; put(shortObject, vt + 2, 4, 2); bad(shortObject);
    std::cout << "header/noop/unaligned/truncation contracts passed" << std::endl;
}
void commandsAndFields() {
    auto bytes = built([](auto& b) {
        auto& f = b.fbb;
        b.AddStructCommand(CreateCommandCreateSection(f, 1, UINT64_MAX, f.CreateString("Prim"), 6));
        b.AddStructCommand(CreateCommandMoveSection(f, 1, 2, UINT64_MAX, f.CreateString("Moved")));
        b.AddStructCommand(CreateCommandDeleteSection(f, 2, UINT64_MAX));
        const uint64_t children[]{10,20};
        b.AddStructCommand(CreateCommandReorderChildren(f, 1, 0, f.CreateVector(children, 2)));
        const uint8_t value[]{3,0,42,0,0,0};
        b.AddSetField(CreateSetField(f, 1, f.CreateString("one"), f.CreateVector(value, sizeof(value)), 0, 0, 1));
        b.AddSetField(CreateSetField(f, 1, f.CreateString("two"), 0, 0, 0, 1)); // Repeated setOrder and deletion are valid.
        b.AddTimeSample(CreateTimeSample(f, 2, -3.5, f.CreateVector(value, sizeof(value))));
    }, false, 57);
    auto decoded = decodeEnvelope(bytes); bytes.assign(bytes.size(), 'x');
    assert(!decoded.isDiff && decoded.baseVersion == 57 && decoded.commands.size() == 4);
    const auto& create = decoded.commands[0]; const auto& move = decoded.commands[1];
    const auto& erase = decoded.commands[2]; const auto& reorder = decoded.commands[3];
    assert(create.kind == Command::Kind::Create && create.section == UINT64_MAX && create.parent == 1 && create.sectionType == 6 && create.name == "Prim");
    assert(move.kind == Command::Kind::Move && move.section == UINT64_MAX && move.oldParent == 1 && move.parent == 2 && move.name == "Moved");
    assert(erase.kind == Command::Kind::Delete && erase.section == UINT64_MAX && erase.parent == 2);
    assert(reorder.kind == Command::Kind::Reorder && reorder.section == 1 && reorder.childList == 0);
    assert((reorder.children == std::vector<uint64_t>{10,20}));
    assert(decoded.fields.size() == 2 && decoded.fields[1].value.empty());
    assert(decoded.fields[0].name == "one" && decoded.fields[1].name == "two");
    for (const auto& field : decoded.fields) assert(field.section == 1 && field.setOrder == 1);
    assert(decoded.fields[0].value.bytes() == std::string("\x03\0\x2a\0\0\0", 6));
    assert(decoded.samples.size() == 1 && decoded.samples[0].section == 2 && decoded.samples[0].time == -3.5 && decoded.samples[0].value.bytes() == decoded.fields[0].value.bytes());
    auto diff = built([](auto& b) {
        b.AddStructCommand(CreateCommandDiffChangeSection(b.fbb, 0, 1, b.fbb.CreateString(""), 7, 0));
        b.AddStructCommand(CreateCommandDiffChangeSection(b.fbb, 1, 2, b.fbb.CreateString("P"), 6, 19));
        b.AddStructCommand(CreateCommandDiffDeleteSection(b.fbb, 3));
    }, true, 123);
    auto parsedDiff = decodeEnvelope(diff);
    assert(parsedDiff.isDiff && parsedDiff.baseVersion == 123 && parsedDiff.commands.size() == 3);
    assert(parsedDiff.commands[0].kind == Command::Kind::DiffChange && parsedDiff.commands[0].section == 1 && parsedDiff.commands[0].parent == 0 && parsedDiff.commands[0].name.empty() && parsedDiff.commands[0].sectionType == 7);
    assert(parsedDiff.commands[1].kind == Command::Kind::DiffChange && parsedDiff.commands[1].section == 2 && parsedDiff.commands[1].parent == 1 && parsedDiff.commands[1].name == "P" && parsedDiff.commands[1].sectionType == 6 && parsedDiff.commands[1].sectionOrder == 19);
    assert(parsedDiff.commands[2].kind == Command::Kind::DiffDelete && parsedDiff.commands[2].section == 3);
    bad(built([](auto& b) { b.AddStructCommand(CreateCommandDeleteSection(b.fbb, 1, 2)); }, true));
    bad(built([](auto& b) { b.AddStructCommand(CreateCommandDiffDeleteSection(b.fbb, 2)); }));
    bad(built([](auto& b) { b.AddStructCommand(CreateCommandCreateSection(b.fbb, 1, 2, 0, 6)); }));
    bad(built([](auto& b) { b.AddStructCommand(CreateCommandCreateSection(b.fbb, 1, 2, b.fbb.CreateString("P"), 255)); }));
    bad(built([](auto& b) { b.AddStructCommand(CreateCommandDiffChangeSection(b.fbb, 0, 1, b.fbb.CreateString("bad"), 7)); }, true));
    bad(built([](auto& b) { b.AddTimeSample(CreateTimeSample(b.fbb, 2, std::numeric_limits<double>::quiet_NaN())); }));
    bad(built([](auto& b) { b.AddSetField(CreateSetField(b.fbb, 0, b.fbb.CreateString("bad"))); }));
    bad(built([](auto& b) { b.AddSetField(CreateSetField(b.fbb, 2, b.fbb.CreateString("a\0b", 3))); }));
    bad(built([](auto& b) { b.AddSetField(CreateSetField(b.fbb, 2, b.fbb.CreateString("\xc0\x80", 2))); }));
    // An empty inline value erases, as an absent one does: the service's compacted snapshot keeps erased fields that way.
    const auto emptied = decodeEnvelope(built([](auto& b) {
        b.AddSetField(CreateSetField(b.fbb, 2, b.fbb.CreateString("x"), b.fbb.CreateVector(std::vector<uint8_t>{})));
        b.AddTimeSample(CreateTimeSample(b.fbb, 2, 1, b.fbb.CreateVector(std::vector<uint8_t>{})));
    }, true));
    assert(emptied.fields.size() == 1 && emptied.fields[0].name == "x" && emptied.fields[0].value.empty());
    assert(emptied.samples.size() == 1 && emptied.samples[0].time == 1 && emptied.samples[0].value.empty());
    // A reorder list is an order: any ID and any repeat decode, in list order.
    const auto repeated = decodeEnvelope(built([](auto& b) {
        const uint64_t ids[]{2,2,0,1,5,2}; b.AddStructCommand(CreateCommandReorderChildren(b.fbb,5,0,b.fbb.CreateVector(ids,6)));
    }));
    assert(repeated.commands.size() == 1 && (repeated.commands[0].children == std::vector<uint64_t>{2,2,0,1,5,2}));
    // Unknown union tags must fail even though the pinned generated verifier accepts them.
    auto unknown = built([](auto& b) { b.structCommandTypes.push_back(255); b.structCommandOffsets.push_back(CreateCommandDeleteSection(b.fbb,1,2).Union()); });
    bad(unknown);
    // Corrupt a known table-vector offset; never form an unchecked outside pointer.
    auto indirect = built([](auto& b) { b.AddSetField(CreateSetField(b.fbb, 1, b.fbb.CreateString("x"))); });
    const auto* d = GetDelta(indirect.data() + 16);
    const size_t at = reinterpret_cast<const char*>(d->setFields()->Data()) - indirect.data();
    put(indirect, at, UINT32_MAX, 4); bad(indirect);
    auto limits = EnvelopeLimits{}; limits.maxStringBytes = 2;
    bad(built([](auto& b) { b.AddSetField(CreateSetField(b.fbb,1,b.fbb.CreateString("long"))); }), limits, ErrorKind::ResourceLimit);
    limits = {}; limits.maxRecords = 1; bad(diff, limits, ErrorKind::ResourceLimit);
    for (double time : {-std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()}) {
        const auto result = decodeEnvelope(built([time](auto& b) { b.AddTimeSample(CreateTimeSample(b.fbb,2,time)); }));
        assert(result.samples.size() == 1 && result.samples[0].time == time && result.samples[0].value.empty());
    }
    std::cout << "command/field/semantic/ownership contracts passed" << std::endl;
}
std::vector<uint8_t> hex(const char* text) {
    std::vector<uint8_t> result;
    for (size_t i = 0; text[i]; i += 2) result.push_back(uint8_t(std::stoul(std::string(text + i, 2), nullptr, 16)));
    return result;
}
void hashAlignment() {
    // Independently calculated hashlib digests; exercise every input/output
    // alignment, padding boundary, buffered tail and multiple direct blocks.
    const std::pair<size_t, const char*> fixtures[] = {
        {0, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
        {55, "463eb28e72f82e0a96c0a4cc53690c571281131f672aa229e0d45ae59b598b59"},
        {56, "da2ae4d6b36748f2a318f23e7ab1dfdf45acdc9d049bd80e59de82a60895f562"},
        {63, "29af2686fd53374a36b0846694cc342177e428d1647515f078784d69cdb9e488"},
        {64, "fdeab9acf3710362bd2658cdc9a29e8f9c757fcf9811603a8c447cd1d9151108"},
        {65, "4bfd2c8b6f1eec7a2afeb48b934ee4b2694182027e6d0fc075074f2fabb31781"},
        {128, "471fb943aa23c511f6f72f8d1652d9c880cfa392ad80503120547703e56a2be5"},
    };
    alignas(8) uint8_t input[136] = {};
    alignas(8) uint8_t output[40] = {};
    for (const auto& [length, digest] : fixtures) {
        const auto expected = hex(digest);
        for (size_t in = 0; in < 8; ++in) {
            for (size_t i = 0; i < length; ++i) input[in + i] = uint8_t(i);
            for (size_t out = 0; out < 8; ++out) {
                computeSHA256(input + in, length, output + out);
                assert(std::memcmp(output + out, expected.data(), 32) == 0);
            }
        }
    }
    std::cout << "SHA256 independent digest and input/output alignment passed" << std::endl;
}
void externalValues() {
    // Independently calculated with Python hashlib over bytes(i % 251).
    const std::pair<size_t, const char*> fixtures[] = {
        {1,"1406e05881e299367766d313e26c05564ec91bf721d31726bd6e46e60689539a"},
        {1048576,"004fbf44b4c0a58632164fc9ec00e8f35e330037fd4425dd947cf65d930762c2"},
        {1048607,"f5c9e6694fcbe188c96f8a1945622f16b6ef5808bc40e4cd9f1abb085f3d7e2b"}};
    for (const auto& [size, digest] : fixtures) {
        std::string payload(size, '\0'); for (size_t i = 0; i < size; ++i) payload[i] = char(i % 251);
        OmniObjectDeltaBuilder b; const auto hash = b.fbb.CreateVector(hex(digest));
        b.AddSetField(CreateSetField(b.fbb, 2, b.fbb.CreateString("a"), hash, size, 1, 1));
        b.AddSetField(CreateSetField(b.fbb, 2, b.fbb.CreateString("b"), hash, size, 1, 1)); // Shared part is legal.
        b.Finish(false, 0); auto packed = pack(b.fbb, {payload}, 1);
        auto decoded = decodeEnvelope(packed);
        assert(decoded.fields[0].value.bytes() == payload && decoded.fields[1].value.bytes() == payload);
        auto limits = EnvelopeLimits{}; limits.maxReferencedValueBytes = size;
        bad(packed, limits, ErrorKind::ResourceLimit);
        packed.back() ^= 1; bad(packed, {}, ErrorKind::Integrity);
        packed.pop_back(); bad(packed);
    }
    bad(built([](auto& b) { b.AddSetField(CreateSetField(b.fbb,1,b.fbb.CreateString("x"),0,0,1)); }));
    bad(built([](auto& b) { b.AddSetField(CreateSetField(b.fbb,1,b.fbb.CreateString("x"),0,10,1)); }));
    {
        std::string first(1, '\0'), second(1048607, '\0');
        for (size_t i = 0; i < second.size(); ++i) second[i] = char(i % 251);
        OmniObjectDeltaBuilder b;
        b.AddSetField(CreateSetField(b.fbb, 2, b.fbb.CreateString("second"), b.fbb.CreateVector(hex(fixtures[2].second)), second.size(), 2));
        b.AddTimeSample(CreateTimeSample(b.fbb, 3, 8, b.fbb.CreateVector(hex(fixtures[0].second)), first.size(), 1));
        b.Finish(false, 0); const auto result = decodeEnvelope(pack(b.fbb, {first, second}));
        assert(result.fields[0].value.bytes() == second && result.samples[0].value.bytes() == first);
    }
    {
        OmniObjectDeltaBuilder b; const auto correct = hex(fixtures[0].second); auto wrong = correct; wrong[0] ^= 1;
        b.AddSetField(CreateSetField(b.fbb, 2, b.fbb.CreateString("first"), b.fbb.CreateVector(correct), 1, 1));
        b.AddSetField(CreateSetField(b.fbb, 2, b.fbb.CreateString("bad-alias"), b.fbb.CreateVector(wrong), 1, 1));
        b.Finish(false, 0); bad(pack(b.fbb, {std::string(1, '\0')}), {}, ErrorKind::Integrity);
    }
    std::cout << "external one-block/multiblock hashes and budgets passed" << std::endl;
}
void fuzz(uint64_t seed) {
    std::mt19937_64 random(seed);
    std::vector<std::string> corpus{noop(), built([](auto& b) {
        b.AddStructCommand(CreateCommandCreateSection(b.fbb, 1, 2, b.fbb.CreateString("P"), 6));
        b.AddSetField(CreateSetField(b.fbb, 2, b.fbb.CreateString("field"), b.fbb.CreateVector(std::vector<uint8_t>{1,2,3})));
        b.AddTimeSample(CreateTimeSample(b.fbb, 2, 1.5, b.fbb.CreateVector(std::vector<uint8_t>{4,5,6})));
    })};
    {
        OmniObjectDeltaBuilder b;
        b.AddStructCommand(CreateCommandDiffChangeSection(b.fbb, 0, 1, b.fbb.CreateString(""), 7, 1));
        b.Finish(true, 0); corpus.push_back(pack(b.fbb));
    }
    {
        OmniObjectDeltaBuilder b;
        b.AddSetField(CreateSetField(b.fbb, 2, b.fbb.CreateString("external"),
            b.fbb.CreateVector(hex("1406e05881e299367766d313e26c05564ec91bf721d31726bd6e46e60689539a")), 1, 1));
        b.Finish(false, 0); corpus.push_back(pack(b.fbb, {std::string(1, '\0')}));
    }
    for (const auto& base : corpus) (void)decodeEnvelope(base);
    for (unsigned n = 0; n < 10000; ++n) {
        auto data = corpus[n % corpus.size()];
        if (n % 3 == 0) data.resize(random() % (data.size() + 80));
        for (unsigned m = 0; m < 1 + n % 8; ++m) if (!data.empty()) data[random() % data.size()] = char(random());
        try { (void)decodeEnvelope(data); } catch (const DecodeError&) {}
    }
    std::cout << "mutated envelope histories=10000 seed=" << seed << std::endl;
}
}
void envelopeEncoding(uint64_t seed);
int main(int argc, char** argv) {
    const uint64_t seed = argc == 2 ? std::stoull(argv[1]) : 42;
    std::cout << "master_seed=" << seed << " scenario=ompf-envelope" << std::endl;
    headersAndNoop(); commandsAndFields(); hashAlignment(); externalValues(); fuzz(seed); envelopeEncoding(seed);
    // The runner requires this final line; it counts the six phases above.
    std::cout << "live-cases passed=6" << std::endl;
}
