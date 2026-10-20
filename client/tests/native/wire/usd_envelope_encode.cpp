// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "UsdValueCodec.h"
#include "schema/OmniObjectDelta_generated.h"
#include <pxr/base/vt/array.h>
#include <pxr/base/vt/dictionary.h>
#include <pxr/usd/sdf/types.h>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

PXR_NAMESPACE_USING_DIRECTIVE
using namespace omni::usd::live::wire;
using namespace OmniObjectDelta;
namespace {
std::string context;
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(context+": "+message);
}
std::string le(uint64_t number, size_t width) {
    std::string result;
    for (size_t i = 0; i < width; ++i) result += char(number >> (8*i));
    return result;
}
uint64_t read(std::string_view bytes, size_t at, size_t width) {
    check(width <= 8 && at <= bytes.size() && width <= bytes.size()-at,"raw read outside multipart buffer");
    uint64_t result = 0;
    for (size_t i = 0; i < width; ++i) result |= uint64_t(uint8_t(bytes[at+i])) << (8*i);
    return result;
}
uint64_t bits(double number) {
    uint64_t result; static_assert(sizeof(result) == sizeof(number));
    std::memcpy(&result,&number,sizeof(result)); return result;
}
std::string_view string(const flatbuffers::String* value) {
    check(value != nullptr,"missing metadata string"); return {value->c_str(),value->size()};
}
std::string hex(const char* text) {
    std::string result;
    for (size_t i = 0; text[i]; i += 2) result += char(std::stoul(std::string(text+i,2),nullptr,16));
    return result;
}
// Direct framing/schema inspection: neither decodeEnvelope nor decodeValue is
// used as an oracle for this outgoing composition boundary.
struct Raw {
    const std::string& bytes;
    std::vector<size_t> offsets;
    std::vector<uint64_t> aligned;
    const Delta* delta;
    explicit Raw(const std::string& encoded) : bytes(encoded) {
        check(read(bytes,0,4) == UINT32_C(0x46504d4f),"wrong multipart magic");
        const size_t count = size_t(read(bytes,4,4));
        check(count > 0 && count <= (bytes.size()-8)/8,"invalid multipart count");
        const size_t header = 8+8*count;
        for (size_t i = 0; i < count; ++i) {
            const size_t offset = size_t(read(bytes,8+8*i,8));
            check(offset >= header && offset < bytes.size() && (offsets.empty() || offset > offsets.back()),
                "multipart offset invalid or overlapping");
            offsets.push_back(offset);
        }
        offsets.push_back(bytes.size());
        const auto first = part(0);
        check(first.size() >= 8 && first.substr(4,4) == "OODF","wrong FlatBuffer identifier");
        aligned.resize((first.size()+7)/8); std::memcpy(aligned.data(),first.data(),first.size());
        flatbuffers::Verifier verifier(reinterpret_cast<const uint8_t*>(aligned.data()),first.size());
        check(VerifyDeltaBuffer(verifier),"generated schema rejected outgoing envelope");
        delta = GetDelta(aligned.data());
    }
    size_t parts() const { return offsets.size()-1; }
    std::string_view part(size_t index) const {
        check(index < parts(),"part index outside multipart table");
        return std::string_view(bytes).substr(offsets[index],offsets[index+1]-offsets[index]);
    }
    template<class Record>
    void value(const Record* record, const std::string& expected, size_t external = 0, const char* digest = nullptr) const {
        const auto* data = record->valueOrExtHash();
        check(record->extValueIndex() == external,"wrong field/sample external order");
        if (expected.empty()) {
            check(!external && !data && record->extValueSize() == 0,"deletion is not an absent vector");
        } else if (!external) {
            check(data && record->extValueSize() == 0 && data->size() == expected.size(),"inline value metadata differs");
            check(std::string_view(reinterpret_cast<const char*>(data->Data()),data->size()) == expected,
                "inline USD bytes differ from hand-authored expectation");
        } else {
            check(data && data->size() == 32 && record->extValueSize() == expected.size(),"external value metadata differs");
            check(part(external) == expected,"external USD bytes differ from hand-authored expectation");
            if (digest) check(std::string_view(reinterpret_cast<const char*>(data->Data()),data->size()) == hex(digest),
                "external tree hash differs from independent hashlib constant");
        }
    }
};
struct Snapshot {
    EnvelopeInput metadata;
    DecodedValues values;
    std::vector<std::string> fieldBytes, sampleBytes;
    Snapshot(const EnvelopeInput& m, const DecodedValues& v) : metadata(m), values(v) {
        for (const auto& field : m.fields) fieldBytes.emplace_back(field.value);
        for (const auto& sample : m.samples) sampleBytes.emplace_back(sample.value);
    }
    void unchanged(const EnvelopeInput& m, const DecodedValues& v) const {
        check(m.isDiff == metadata.isDiff && m.baseVersion == metadata.baseVersion && m.sourceFormat == metadata.sourceFormat &&
            m.commands.size() == metadata.commands.size() && m.fields.size() == metadata.fields.size() &&
            m.samples.size() == metadata.samples.size(),"bridge changed metadata input");
        for (size_t i = 0; i < m.commands.size(); ++i) {
            const auto& a = m.commands[i]; const auto& b = metadata.commands[i];
            check(a.kind == b.kind && a.section == b.section && a.parent == b.parent && a.oldParent == b.oldParent &&
                a.sectionType == b.sectionType && a.childList == b.childList && a.sectionOrder == b.sectionOrder &&
                a.name == b.name && a.children == b.children,"bridge changed command metadata");
        }
        for (size_t i = 0; i < m.fields.size(); ++i) {
            const auto& a = m.fields[i]; const auto& b = metadata.fields[i];
            check(a.section == b.section && a.setOrder == b.setOrder && a.name == b.name && a.value == fieldBytes[i],
                "bridge changed field metadata or borrowed bytes");
        }
        for (size_t i = 0; i < m.samples.size(); ++i) {
            const auto& a = m.samples[i]; const auto& b = metadata.samples[i];
            check(a.section == b.section && bits(a.time) == bits(b.time) && a.value == sampleBytes[i],
                "bridge changed sample metadata or borrowed bytes");
        }
        check(v.fields == values.fields && v.samples == values.samples,"bridge changed authored USD input");
    }
};
std::string good(const EnvelopeInput& metadata, const DecodedValues& values,
                 const EnvelopeEncodeLimits& envelope = {}, const EncodeLimits& limits = {}) {
    const Snapshot before(metadata,values);
    auto result = encodeUsdEnvelope(metadata,values,envelope,limits);
    before.unchanged(metadata,values); return result;
}
void bad(const EnvelopeInput& metadata, const DecodedValues& values, ErrorKind kind,
         const EnvelopeEncodeLimits& envelope = {}, const EncodeLimits& limits = {}) {
    const Snapshot before(metadata,values);
    std::string unpublished = "caller-owned sentinel";
    bool rejected = false;
    try { unpublished = encodeUsdEnvelope(metadata,values,envelope,limits); }
    catch (const DecodeError& error) { check(error.kind == kind,"wrong typed-envelope error kind"); rejected = true; }
    check(rejected,"typed-envelope bridge accepted invalid/over-budget input");
    check(unpublished == "caller-owned sentinel","failed bridge published a partial envelope");
    before.unchanged(metadata,values);
}
void outgoing() {
    context = "typed USD envelope exact outgoing bytes";
    EnvelopeInput metadata; metadata.baseVersion = UINT64_C(0xfedcba9876543210);
    Command command{}; command.kind = Command::Kind::Create; command.section = 7;
    command.parent = 1; command.sectionType = 6; command.name = "Prim";
    metadata.commands = {command};
    metadata.fields = {{7,19,"number",{}},{7,19,"delete",{}},{7,2,"block",{}},{9,0,"external",{}}};
    metadata.samples = {{9,-0.0,{}},{7,4.5,{}},{7,std::numeric_limits<double>::infinity(),{}}};
    const std::string large = "large-field-"+std::string(10,'x'), embedded("a\0b",3);
    DecodedValues values{{VtValue(-123),VtValue(),VtValue(SdfValueBlock()),VtValue(large)},
        {VtValue(VtArray<int>{17,-31,99}),VtValue(embedded),VtValue(-0.0)}};
    EnvelopeEncodeLimits limits; limits.externalValueThreshold = 17;
    const auto output = good(metadata,values,limits);
    const Raw raw(output);
    check(raw.parts() == 3 && !raw.delta->isDiff() && raw.delta->baseVersion() == metadata.baseVersion,
        "typed bridge framing/mode/base differs");
    check(raw.delta->structCommands_type() && raw.delta->structCommands_type()->size() == 1 &&
        raw.delta->structCommands_type()->Get(0) == StructCommand_CommandCreateSection &&
        raw.delta->structCommands() && raw.delta->structCommands()->size() == 1,"typed bridge lost command order/type");
    const auto* created = static_cast<const CommandCreateSection*>(raw.delta->structCommands()->Get(0));
    check(created->sectionId() == 7 && created->parentId() == 1 && created->sectionType() == 6 &&
        string(created->sectionName()) == "Prim","typed bridge changed create metadata");
    const auto* fields = raw.delta->setFields(); const auto* samples = raw.delta->timeSamples();
    check(fields && fields->size() == metadata.fields.size() && samples && samples->size() == metadata.samples.size(),
        "typed bridge field/sample counts differ");
    for (size_t i = 0; i < metadata.fields.size(); ++i) {
        const auto* field = fields->Get(i); const auto& expected = metadata.fields[i];
        check(field->sectionId() == expected.section && field->setOrder() == expected.setOrder && string(field->keyName()) == expected.name,
            "typed bridge field pairing/order differs");
    }
    for (size_t i = 0; i < metadata.samples.size(); ++i) {
        const auto* sample = samples->Get(i); const auto& expected = metadata.samples[i];
        check(sample->sectionId() == expected.section && bits(sample->time()) == bits(expected.time),"typed bridge sample pairing/time bits differ");
    }
    // Independent version-zero bytes, including canonical ValueBlock byte zero.
    raw.value(fields->Get(0),le(3,2)+le(uint32_t(-123),4));
    raw.value(fields->Get(1),{});
    raw.value(fields->Get(2),le(51,2)+le(0,1));
    const auto fieldBytes = le(10,2)+le(large.size(),4)+large;
    const auto sampleBytes = le(0x83,2)+le(3,8)+le(17,4)+le(uint32_t(-31),4)+le(99,4);
    // Constants generated independently with Python hashlib over encoded bytes.
    raw.value(fields->Get(3),fieldBytes,1,"8ea44c825e05ba6d906ced564cad1a765879a409dda4ff409ce77713bbc5b676");
    raw.value(samples->Get(0),sampleBytes,2,"9b4fecedfed2892080c1b3fdcfe4b596df2c4d6d866ed557e2ad297dc531813b");
    raw.value(samples->Get(1),le(10,2)+le(3,4)+embedded);
    raw.value(samples->Get(2),le(9,2)+le(UINT64_C(0x8000000000000000),8));
    metadata = {}; values = {};
    check(raw.part(1) == fieldBytes && raw.part(2) == sampleBytes,"typed output retained temporary encoded value buffers");

    EnvelopeInput diff; diff.isDiff = true; diff.baseVersion = 91; diff.sourceFormat = "usda";
    command = {}; command.kind = Command::Kind::DiffChange; command.section = 1; command.sectionType = 7; command.sectionOrder = 23;
    diff.commands = {command}; diff.fields = {{1,5,"comment",{}}};
    const auto diffOutput = good(diff,DecodedValues{{VtValue(embedded)},{}});
    const Raw diffRaw(diffOutput);
    check(diffRaw.delta->isDiff() && diffRaw.delta->baseVersion() == 91 && string(diffRaw.delta->sourceFormat()) == "usda",
        "typed bridge lost diff metadata");
    check(diffRaw.delta->structCommands_type() && diffRaw.delta->structCommands_type()->size() == 1 &&
        diffRaw.delta->structCommands() && diffRaw.delta->structCommands()->size() == 1 &&
        diffRaw.delta->structCommands_type()->Get(0) == StructCommand_CommandDiffChangeSection,"typed bridge lost diff command type");
    const auto* root = static_cast<const CommandDiffChangeSection*>(diffRaw.delta->structCommands()->Get(0));
    check(root->sectionId() == 1 && root->parentId() == 0 && root->sectionType() == 7 && root->sectionOrder() == 23 &&
        string(root->sectionName()).empty(),"typed bridge lost diff root metadata");
    check(diffRaw.delta->setFields() && diffRaw.delta->setFields()->size() == 1,"typed diff lost field");
    diffRaw.value(diffRaw.delta->setFields()->Get(0),le(10,2)+le(3,4)+embedded);
    const auto noop = good({},{}); const Raw empty(noop);
    check(empty.parts() == 1 && !empty.delta->isDiff() && empty.delta->baseVersion() == 0 &&
        (!empty.delta->structCommands() || empty.delta->structCommands()->size() == 0) &&
        (!empty.delta->setFields() || empty.delta->setFields()->size() == 0) &&
        (!empty.delta->timeSamples() || empty.delta->timeSamples()->size() == 0),
        "typed empty batch is not an ordinary single-part noop");
}
void failuresAndBudgets() {
    context = "typed USD envelope validation order and budgets";
    EnvelopeInput metadata; metadata.fields = {{2,3,"a",{}}}; metadata.samples = {{3,0,{}}};
    const VtValue shared(std::string("abc"));
    const DecodedValues values{{shared},{shared}};
    bad(metadata,DecodedValues{{},values.samples},ErrorKind::Malformed);
    bad(metadata,DecodedValues{values.fields,{}},ErrorKind::Malformed);
    auto ambiguous = metadata; const std::string alreadyEncoded("\x03\0\x01\0\0\0",6);
    ambiguous.fields[0].value = alreadyEncoded; bad(ambiguous,values,ErrorKind::Malformed);
    ambiguous = metadata; ambiguous.samples[0].value = alreadyEncoded; bad(ambiguous,values,ErrorKind::Malformed);
    VtDictionary unsupported; unsupported["nested"] = VtValue(std::vector<int>{1,2});
    const DecodedValues invalid{{shared},{VtValue(unsupported)}};
    bad(metadata,invalid,ErrorKind::Unsupported);
    auto invalidMetadata = metadata; invalidMetadata.fields[0].section = 0;
    bad(invalidMetadata,invalid,ErrorKind::Malformed); // Metadata failure must precede unsupported USD traversal.
    EnvelopeEncodeLimits envelope; envelope.content.maxRecords = 1;
    bad(metadata,invalid,ErrorKind::ResourceLimit,envelope);
    envelope = {}; envelope.content.maxStringBytes = 0;
    bad(metadata,invalid,ErrorKind::ResourceLimit,envelope);
    envelope = {}; envelope.maxScratchBytes = 1;
    bad(metadata,invalid,ErrorKind::ResourceLimit,envelope);

    EncodeLimits limits; limits.maxEncodedBytes = 18;
    const auto output = good(metadata,values,{},limits); // Each tagged string is9 bytes; budget is shared across fields/samples.
    limits.maxEncodedBytes = 17; bad(metadata,values,ErrorKind::ResourceLimit,{},limits);
    limits = {}; limits.maxStringBytes = 6; good(metadata,values,{},limits);
    limits.maxStringBytes = 5; bad(metadata,values,ErrorKind::ResourceLimit,{},limits);
    limits = {}; limits.maxElements = 2; good(metadata,values,{},limits);
    limits.maxElements = 1; bad(metadata,values,ErrorKind::ResourceLimit,{},limits);
    envelope = {}; envelope.content.maxReferencedValueBytes = 18; good(metadata,values,envelope);
    envelope.content.maxReferencedValueBytes = 17; bad(metadata,values,ErrorKind::ResourceLimit,envelope);
    envelope = {}; envelope.content.maxBytes = output.size(); good(metadata,values,envelope);
    --envelope.content.maxBytes; bad(metadata,values,ErrorKind::ResourceLimit,envelope);
    // Metadata alone has no external parts. This limit fails only when the
    // completely encoded values are placed into the final multipart envelope.
    envelope = {}; envelope.externalValueThreshold = 1; envelope.content.maxParts = 3; good(metadata,values,envelope);
    envelope.content.maxParts = 2; bad(metadata,values,ErrorKind::ResourceLimit,envelope);
    envelope = {}; envelope.maxBuilderBytes = 1; bad(metadata,values,ErrorKind::ResourceLimit,envelope);
    const auto afterFailure = good(metadata,values);
    check(afterFailure == output,"failure changed later bridge output for unchanged input");
}
} // namespace
void usdEnvelopeEncoding() {
    outgoing(); failuresAndBudgets();
    std::cout << "typed USD envelope independent bytes/metadata/quotas/failure preservation passed" << std::endl;
}
