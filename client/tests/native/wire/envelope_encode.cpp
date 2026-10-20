// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "OmpfEncode.h"
#include "schema/OmniObjectDelta_generated.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

using namespace omni::usd::live::wire;
using namespace OmniObjectDelta;
namespace {
std::string context;
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(context+": "+message);
}
uint64_t integer(std::string_view bytes, size_t offset, size_t width) {
    check(width <= 8 && offset <= bytes.size() && width <= bytes.size()-offset,"raw integer outside buffer");
    uint64_t result = 0;
    for (size_t i = 0; i < width; ++i) result |= uint64_t(uint8_t(bytes[offset+i])) << (8*i);
    return result;
}
uint64_t bits(double number) {
    uint64_t result; static_assert(sizeof(result) == sizeof(number));
    std::memcpy(&result,&number,sizeof(result)); return result;
}
std::string_view text(const flatbuffers::String* string) {
    check(string != nullptr,"missing raw string"); return {string->c_str(),string->size()};
}
template<class T> size_t count(const T* vector) { return vector ? vector->size() : 0; }
std::string pattern(size_t size) {
    std::string result(size,'\0');
    for (size_t i = 0; i < size; ++i) result[i] = char(i % 251);
    return result;
}
std::string hex(const char* encoded) {
    std::string result;
    for (size_t i = 0; encoded[i]; i += 2) result += char(std::stoul(std::string(encoded+i,2),nullptr,16));
    return result;
}
void sameCommand(const Command& a, const Command& b) {
    check(a.kind == b.kind && a.section == b.section && a.parent == b.parent && a.oldParent == b.oldParent &&
        a.sectionType == b.sectionType && a.childList == b.childList && a.sectionOrder == b.sectionOrder &&
        a.name == b.name && a.children == b.children,"encoding changed command input");
}
struct Snapshot {
    EnvelopeInput metadata;
    std::vector<std::string> fields, samples;
    explicit Snapshot(const EnvelopeInput& input) : metadata(input) {
        for (const auto& field : input.fields) fields.emplace_back(field.value);
        for (const auto& sample : input.samples) samples.emplace_back(sample.value);
    }
    void unchanged(const EnvelopeInput& input) const {
        check(input.isDiff == metadata.isDiff && input.baseVersion == metadata.baseVersion &&
            input.sourceFormat == metadata.sourceFormat && input.commands.size() == metadata.commands.size() &&
            input.fields.size() == fields.size() && input.samples.size() == samples.size(),"encoding changed envelope input");
        for (size_t i = 0; i < input.commands.size(); ++i) sameCommand(input.commands[i],metadata.commands[i]);
        for (size_t i = 0; i < input.fields.size(); ++i) {
            const auto& a = input.fields[i]; const auto& b = metadata.fields[i];
            check(a.section == b.section && a.setOrder == b.setOrder && a.name == b.name && a.value == fields[i],
                "encoding changed field or borrowed field bytes");
        }
        for (size_t i = 0; i < input.samples.size(); ++i) {
            const auto& a = input.samples[i]; const auto& b = metadata.samples[i];
            check(a.section == b.section && bits(a.time) == bits(b.time) && a.value == samples[i],
                "encoding changed sample or borrowed sample bytes");
        }
    }
};
// This oracle deliberately never calls decodeEnvelope or the production SHA
// implementation. Inspect framing directly, then use only the generated schema
// and verifier over an aligned copy of the bounded first part.
struct Raw {
    const std::string& bytes;
    std::vector<size_t> offsets;
    std::vector<uint64_t> aligned;
    const Delta* delta = nullptr;
    explicit Raw(const std::string& input) : bytes(input) {
        check(integer(bytes,0,4) == UINT32_C(0x46504d4f),"wrong OMPF magic");
        const size_t parts = size_t(integer(bytes,4,4));
        check(parts > 0 && parts <= (bytes.size()-8)/8,"invalid raw part count");
        const size_t header = 8+parts*8;
        for (size_t i = 0; i < parts; ++i) {
            const size_t offset = size_t(integer(bytes,8+i*8,8));
            check(offset >= header && offset < bytes.size(),"raw part offset outside payload");
            check(offsets.empty() || offset > offsets.back(),"empty, reordered or overlapping raw parts");
            offsets.push_back(offset);
        }
        offsets.push_back(bytes.size());
        const auto first = part(0);
        check(first.size() >= 8 && first.substr(4,4) == "OODF","wrong FlatBuffer file identifier");
        aligned.resize((first.size()+7)/8);
        std::memcpy(aligned.data(),first.data(),first.size());
        flatbuffers::Verifier verifier(reinterpret_cast<const uint8_t*>(aligned.data()),first.size());
        check(VerifyDeltaBuffer(verifier),"generated schema verifier rejected encoded part");
        delta = GetDelta(aligned.data());
    }
    size_t parts() const { return offsets.size()-1; }
    std::string_view part(size_t index) const {
        check(index < parts(),"external part index outside framing");
        return std::string_view(bytes).substr(offsets[index],offsets[index+1]-offsets[index]);
    }
    void command(size_t index, const Command& expected) const {
        const auto tag = delta->structCommands_type()->Get(index);
        const void* payload = delta->structCommands()->Get(index);
        switch (expected.kind) {
        case Command::Kind::Create: {
            check(tag == StructCommand_CommandCreateSection,"wrong create union tag");
            const auto* c = static_cast<const CommandCreateSection*>(payload);
            check(c->sectionId() == expected.section && c->parentId() == expected.parent &&
                c->sectionType() == expected.sectionType && text(c->sectionName()) == expected.name,"create metadata differs"); break;
        }
        case Command::Kind::Delete: {
            check(tag == StructCommand_CommandDeleteSection,"wrong delete union tag");
            const auto* c = static_cast<const CommandDeleteSection*>(payload);
            check(c->sectionId() == expected.section && c->parentId() == expected.parent,"delete metadata differs"); break;
        }
        case Command::Kind::Move: {
            check(tag == StructCommand_CommandMoveSection,"wrong move union tag");
            const auto* c = static_cast<const CommandMoveSection*>(payload);
            check(c->sectionId() == expected.section && c->oldParentId() == expected.oldParent &&
                c->newParentId() == expected.parent && text(c->newName()) == expected.name,"move metadata differs"); break;
        }
        case Command::Kind::Reorder: {
            check(tag == StructCommand_CommandReorderChildren,"wrong reorder union tag");
            const auto* c = static_cast<const CommandReorderChildren*>(payload);
            check(c->sectionId() == expected.section && c->childrenListId() == expected.childList &&
                c->childrenList() && c->childrenList()->size() == expected.children.size(),"reorder metadata differs");
            for (size_t i = 0; i < expected.children.size(); ++i)
                check(c->childrenList()->Get(i) == expected.children[i],"child order/content differs");
            break;
        }
        case Command::Kind::DiffChange: {
            check(tag == StructCommand_CommandDiffChangeSection,"wrong diff change union tag");
            const auto* c = static_cast<const CommandDiffChangeSection*>(payload);
            check(c->sectionId() == expected.section && c->parentId() == expected.parent &&
                c->sectionType() == expected.sectionType && c->sectionOrder() == expected.sectionOrder &&
                text(c->sectionName()) == expected.name,"diff change metadata differs"); break;
        }
        case Command::Kind::DiffDelete: {
            check(tag == StructCommand_CommandDiffDeleteSection,"wrong diff delete union tag");
            const auto* c = static_cast<const CommandDiffDeleteSection*>(payload);
            check(c->sectionId() == expected.section,"diff delete metadata differs"); break;
        }
        }
    }
    template<class Record>
    void value(const Record* record, std::string_view expected, size_t threshold, size_t& external) const {
        const auto* data = record->valueOrExtHash();
        if (expected.empty()) {
            check(data == nullptr && record->extValueSize() == 0 && record->extValueIndex() == 0,
                "deletion was not an absent value vector");
        } else if (expected.size() < threshold) {
            check(data && data->size() == expected.size() && record->extValueSize() == 0 && record->extValueIndex() == 0,
                "inline value metadata differs");
            check(std::string_view(reinterpret_cast<const char*>(data->Data()),data->size()) == expected,"inline bytes differ");
        } else {
            check(data && data->size() == 32 && record->extValueSize() == expected.size() && record->extValueIndex() == external,
                "external value size, hash shape or field/sample part order differs");
            check(part(external) == expected,"external bytes differ"); ++external;
        }
    }
    void matches(const EnvelopeInput& expected, size_t threshold = 65535) const {
        check(delta->isDiff() == expected.isDiff && delta->baseVersion() == expected.baseVersion,"delta mode/base version differs");
        const std::string_view format = delta->sourceFormat() ? text(delta->sourceFormat()) : std::string_view{};
        check(format == expected.sourceFormat,"source format differs");
        check(count(delta->structCommands()) == expected.commands.size() &&
            count(delta->structCommands_type()) == expected.commands.size(),"command count differs");
        for (size_t i = 0; i < expected.commands.size(); ++i) command(i,expected.commands[i]);
        check(count(delta->setFields()) == expected.fields.size() && count(delta->timeSamples()) == expected.samples.size(),
            "field/sample count differs");
        size_t external = 1;
        for (size_t i = 0; i < expected.fields.size(); ++i) {
            const auto* f = delta->setFields()->Get(i); const auto& source = expected.fields[i];
            check(f->sectionId() == source.section && f->setOrder() == source.setOrder && text(f->keyName()) == source.name,
                "field identity/order/name differs");
            value(f,source.value,threshold,external);
        }
        for (size_t i = 0; i < expected.samples.size(); ++i) {
            const auto* s = delta->timeSamples()->Get(i); const auto& source = expected.samples[i];
            check(s->sectionId() == source.section && bits(s->time()) == bits(source.time),"sample section/time bits differ");
            value(s,source.value,threshold,external);
        }
        check(parts() == external,"unexpected or omitted external parts");
    }
};
std::string good(const EnvelopeInput& input, const EnvelopeEncodeLimits& limits = {}) {
    const Snapshot before(input);
    auto output = encodeEnvelope(input,limits);
    before.unchanged(input);
    Raw(output).matches(input,limits.externalValueThreshold);
    return output;
}
void bad(const EnvelopeInput& input, ErrorKind kind, const EnvelopeEncodeLimits& limits = {}) {
    const Snapshot before(input);
    std::string output = "unpublished sentinel";
    bool rejected = false;
    try { output = encodeEnvelope(input,limits); }
    catch (const DecodeError& error) { check(error.kind == kind,"wrong envelope encoding error kind"); rejected = true; }
    check(rejected,"invalid/over-budget envelope was accepted");
    check(output == "unpublished sentinel","failed envelope encoding exposed partial output");
    before.unchanged(input);
}
Command make(Command::Kind kind, uint64_t section) { Command c{}; c.kind = kind; c.section = section; return c; }
EnvelopeInput ordinary() {
    EnvelopeInput input; input.baseVersion = UINT64_MAX;
    auto create = make(Command::Kind::Create,UINT64_MAX); create.parent = 1; create.name = "Prim"; create.sectionType = 6;
    auto move = make(Command::Kind::Move,UINT64_MAX); move.oldParent = 1; move.parent = 2; move.name = "Renamed";
    auto erase = make(Command::Kind::Delete,UINT64_MAX); erase.parent = 2;
    auto reorder = make(Command::Kind::Reorder,2); reorder.childList = 2; reorder.children = {19,7,11};
    input.commands = {create,move,erase,reorder};
    return input;
}
void metadata() {
    context = "envelope encoder metadata";
    const auto noop = good({});
    check(Raw(noop).parts() == 1,"noop contains external parts");
    const std::string a("\x03\0\x2a\0\0\0",6), b("\x0a\0\x03\0\0\0" "a\0b",9);
    auto input = ordinary();
    input.fields = {{2,31,"first",a},{3,31,"second",{}},{2,0,"third",b}};
    input.samples = {{2,-3.5,b},{3,-0.0,a},{4,std::numeric_limits<double>::infinity(),{}},
        {4,-std::numeric_limits<double>::infinity(),a}};
    good(input);
    input.commands.back().children.clear(); good(input); // Empty reorder remains a present empty ID vector.
    EnvelopeInput diff; diff.isDiff = true; diff.baseVersion = UINT64_C(0xfedcba9876543210); diff.sourceFormat = "usda";
    auto root = make(Command::Kind::DiffChange,1); root.sectionType = 7; root.sectionOrder = 37;
    auto child = make(Command::Kind::DiffChange,19); child.parent = 1; child.sectionType = 6; child.sectionOrder = UINT32_MAX; child.name = "Child";
    diff.commands = {root,child,make(Command::Kind::DiffDelete,11)};
    diff.fields = {{1,UINT64_MAX,"comment",b}}; diff.samples = {{19,-0.0,a}};
    good(diff);
    // Returned bytes own their metadata and payload after inputs are changed.
    const Snapshot before(input);
    auto output = good(input);
    input.commands.clear(); input.fields.clear(); input.samples.clear();
    Raw(output).matches(before.metadata);
}
void externalParts() {
    context = "envelope encoder external framing and independent hashes";
    const auto inlineBytes = pattern(65534), externalBytes = pattern(65535);
    EnvelopeInput boundary; boundary.fields = {{2,0,"inline",inlineBytes},{2,0,"external",externalBytes}};
    const auto output = good(boundary); const Raw parsed(output);
    check(parsed.parts() == 2,"default threshold boundary differs");
    const auto* hash = parsed.delta->setFields()->Get(1)->valueOrExtHash();
    check(std::string_view(reinterpret_cast<const char*>(hash->Data()),hash->size()) ==
        hex("88f44eb280210885d300225265bf618dfb15daee217f13518251253b8773b2a1"),"threshold external tree hash differs");
    // Python hashlib fixtures: SHA256(concatenated SHA256 of each 1MiB block).
    // One byte, one full block and a full block plus a short tail distinguish
    // tree hashing from ordinary SHA256 and exercise the tail/block boundary.
    const std::string one = pattern(1), block = pattern(1048576), tail = pattern(1048607);
    EnvelopeInput mixed;
    mixed.fields = {{2,4,"first",one},{2,4,"delete",{}},{3,8,"second",tail}};
    mixed.samples = {{4,1,block},{5,2,{}}};
    EnvelopeEncodeLimits limits; limits.externalValueThreshold = 1;
    const auto encoded = good(mixed,limits); const Raw raw(encoded);
    check(raw.parts() == 4,"one part per external field/sample was not emitted");
    const auto checkHash = [](const auto* record, const char* digest) {
        const auto* hash = record->valueOrExtHash();
        check(hash && std::string_view(reinterpret_cast<const char*>(hash->Data()),hash->size()) == hex(digest),
            "external hash differs from independent hashlib fixture");
    };
    checkHash(raw.delta->setFields()->Get(0),"1406e05881e299367766d313e26c05564ec91bf721d31726bd6e46e60689539a");
    checkHash(raw.delta->setFields()->Get(2),"f5c9e6694fcbe188c96f8a1945622f16b6ef5808bc40e4cd9f1abb085f3d7e2b");
    checkHash(raw.delta->timeSamples()->Get(0),"004fbf44b4c0a58632164fc9ec00e8f35e330037fd4425dd947cf65d930762c2");
    check(raw.part(1) == one && raw.part(2) == tail && raw.part(3) == block,"field/sample external byte ordering differs");
    std::string borrowed = externalBytes;
    EnvelopeInput transient; transient.fields = {{2,0,"owned",borrowed}};
    const auto owned = good(transient);
    borrowed.assign(borrowed.size(),'x'); transient.fields.clear();
    check(Raw(owned).part(1) == externalBytes,"returned envelope retained a borrowed external payload");
}
void rejectionAndLimits() {
    context = "envelope encoder rejection and budgets";
    const auto base = ordinary();
    auto input = base; input.sourceFormat = "usda"; bad(input,ErrorKind::Malformed);
    input = base; input.isDiff = true; bad(input,ErrorKind::Malformed);
    input = {}; input.commands = {make(Command::Kind::DiffDelete,2)}; bad(input,ErrorKind::Malformed);
    input = base; input.commands[0].sectionType = 7; bad(input,ErrorKind::Malformed);
    input = base; input.commands[0].name.clear(); bad(input,ErrorKind::Malformed);
    input = base; input.commands[0].parent = input.commands[0].section; bad(input,ErrorKind::Malformed);
    input = base; input.commands[1].oldParent = 0; bad(input,ErrorKind::Malformed);
    input = base; input.commands[3].children = {3,3}; bad(input,ErrorKind::Malformed);
    input = base; input.commands[3].children = {1}; bad(input,ErrorKind::Malformed);
    input = base; input.commands[3].childList = 3; bad(input,ErrorKind::Malformed);
    // Irrelevant metadata has no wire representation. Reject rather than lose
    // these fields while returning an apparently valid complete envelope.
    for (size_t index = 0; index < base.commands.size(); ++index) {
        input = base; input.commands[index].sectionOrder = 1; bad(input,ErrorKind::Malformed);
    }
    input = base; input.commands[0].oldParent = 4; bad(input,ErrorKind::Malformed);
    input = base; input.commands[0].children = {3}; bad(input,ErrorKind::Malformed);
    input = base; input.commands[1].sectionType = 6; bad(input,ErrorKind::Malformed);
    input = base; input.commands[2].name = "discarded"; bad(input,ErrorKind::Malformed);
    input = base; input.commands[3].parent = 1; bad(input,ErrorKind::Malformed);
    input = {}; input.isDiff = true; auto erase = make(Command::Kind::DiffDelete,3); erase.parent = 1;
    input.commands = {erase}; bad(input,ErrorKind::Malformed);
    input = {}; input.fields = {{0,0,"field",{}}}; bad(input,ErrorKind::Malformed);
    input.fields = {{2,0,"",{}}}; bad(input,ErrorKind::Malformed);
    input.fields[0].name = std::string("a\0b",3); bad(input,ErrorKind::Malformed);
    input.fields[0].name = std::string("\xc0\x80",2); bad(input,ErrorKind::Malformed);
    input = {}; input.samples = {{0,0,{}}}; bad(input,ErrorKind::Malformed);
    input.samples = {{2,std::numeric_limits<double>::quiet_NaN(),{}}}; bad(input,ErrorKind::Malformed);
    input = {}; input.isDiff = true; auto root = make(Command::Kind::DiffChange,1); root.sectionType = 7; root.name = "bad";
    input.commands = {root}; bad(input,ErrorKind::Malformed);
    input = {}; input.commands = {make(static_cast<Command::Kind>(255),3)}; bad(input,ErrorKind::Malformed);

    const std::string bytes("a\0b",3);
    input = {}; input.fields = {{2,4,"a",bytes},{2,4,"b",bytes}}; input.samples = {{3,0,bytes}};
    const auto encoded = good(input);
    auto limits = EnvelopeEncodeLimits{}; limits.content.maxBytes = encoded.size(); good(input,limits);
    --limits.content.maxBytes; bad(input,ErrorKind::ResourceLimit,limits);
    limits = {}; limits.content.maxRecords = 3; good(input,limits);
    limits.content.maxRecords = 2; bad(input,ErrorKind::ResourceLimit,limits);
    limits = {}; limits.content.maxStringBytes = 2; good(input,limits);
    limits.content.maxStringBytes = 1; bad(input,ErrorKind::ResourceLimit,limits);
    limits = {}; limits.content.maxReferencedValueBytes = 9; good(input,limits);
    limits.content.maxReferencedValueBytes = 8; bad(input,ErrorKind::ResourceLimit,limits);
    limits = {}; limits.maxBuilderBytes = 1; bad(input,ErrorKind::ResourceLimit,limits);
    limits = {}; limits.maxScratchBytes = 1; bad(input,ErrorKind::ResourceLimit,limits);
    limits = {}; limits.externalValueThreshold = 1; limits.content.maxParts = 4; good(input,limits);
    limits.content.maxParts = 3; bad(input,ErrorKind::ResourceLimit,limits);
    input = base; limits = {}; limits.content.maxRecords = 7; good(input,limits); // 4 commands + 3 reordered IDs.
    limits.content.maxRecords = 6; bad(input,ErrorKind::ResourceLimit,limits);
    const std::string largeInline = pattern(65534);
    input = {}; input.fields = {{2,0,"growth",largeInline}};
    limits = {}; limits.maxBuilderBytes = 1024; // Allows the initial buffer, then forces a bounded growth failure.
    bad(input,ErrorKind::ResourceLimit,limits); good(input);
    // Failure at the final sample must not change earlier borrowed bytes or
    // publish an envelope prefix. A subsequent valid invocation still succeeds.
    input = {}; input.fields = {{2,0,"first",bytes}};
    input.samples = {{3,0,bytes},{4,std::numeric_limits<double>::quiet_NaN(),bytes}};
    bad(input,ErrorKind::Malformed); input.samples.pop_back(); good(input);
}
void histories(uint64_t seed) {
    std::mt19937_64 random(seed);
    uint64_t fingerprint = UINT64_C(14695981039346656037);
    size_t externalCount = 0;
    for (size_t step = 0; step < 150; ++step) {
        context = "envelope encoder history seed "+std::to_string(seed)+" step "+std::to_string(step);
        EnvelopeInput input; input.isDiff = (step % 2) != 0; input.baseVersion = random();
        if (input.isDiff) input.sourceFormat = "usda";
        const size_t n = 1+random()%8;
        // Reserve before storing string_views; every input payload has its own
        // stable backing string, with distinguishable sizes and content.
        std::vector<std::string> values; values.reserve(n*2);
        for (size_t i = 0; i < n*2; ++i) {
            std::string bytes(random()%40,'\0');
            for (auto& c : bytes) c = char(random());
            values.push_back(std::move(bytes));
        }
        for (size_t i = 0; i < n; ++i) {
            const uint64_t section = i+2;
            auto command = make(input.isDiff ? Command::Kind::DiffChange : Command::Kind::Create,section);
            command.parent = 1; command.name = "P"+std::to_string(i); command.sectionType = 6;
            if (input.isDiff) command.sectionOrder = uint32_t(random());
            input.commands.push_back(command);
            input.fields.push_back({section,random()%3,"f"+std::to_string(i),values[i]});
            input.samples.push_back({section,double(int(random()%100)-50)/4.0,values[n+i]});
        }
        if (!input.isDiff) {
            auto move = make(Command::Kind::Move,n+1); move.oldParent = 1; move.parent = n+2; move.name = "Moved";
            input.commands.push_back(move);
            auto erase = make(Command::Kind::Delete,n+1); erase.parent = n+2; input.commands.push_back(erase);
            auto reorder = make(Command::Kind::Reorder,1); reorder.childList = step%3;
            for (size_t i = n; i > 0; --i) reorder.children.push_back(i+1);
            input.commands.push_back(reorder);
        } else input.commands.push_back(make(Command::Kind::DiffDelete,n+20));
        EnvelopeEncodeLimits limits; limits.externalValueThreshold = 17;
        const auto output = good(input,limits);
        externalCount += Raw(output).parts()-1;
        for (unsigned char c : output) { fingerprint ^= c; fingerprint *= UINT64_C(1099511628211); }
    }
    check(externalCount > 0,"seeded histories never exercised external values");
    std::cout << "envelope encoder seed=" << seed << " histories=150 external=" << externalCount
              << " fingerprint=" << fingerprint << std::endl;
}
} // namespace
void envelopeEncoding(uint64_t seed) {
    std::cout << "envelope encoder seed=" << seed << std::endl;
    metadata(); externalParts(); rejectionAndLimits(); histories(seed);
    std::cout << "independent raw envelope encoder framing/commands/hashes/budgets passed" << std::endl;
}
