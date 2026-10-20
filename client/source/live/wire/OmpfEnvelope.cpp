// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "OmpfEnvelope.h"
#include "schema/OmniObjectDelta_generated.h"
#include "OmpfHash.h"
#include "OmpfValidation.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <initializer_list>
#include <type_traits>
#include <optional>

namespace omni::usd::live::wire {
class EnvelopeStorage {
public:
    explicit EnvelopeStorage(std::string_view bytes) : words((bytes.size() + 7) / 8) {
        std::memcpy(words.data(), bytes.data(), bytes.size());
    }
    const uint8_t* data() const { return reinterpret_cast<const uint8_t*>(words.data()); }
private:
    // Explicit alignment even if the caller supplies an unaligned substring.
    std::vector<uint64_t> words;
};
std::string_view ValueBytes::bytes() const {
    return size_ ? std::string_view(reinterpret_cast<const char*>(storage_->data() + offset_), size_) :
                   std::string_view{};
}
namespace {
using namespace OmniObjectDelta;
[[noreturn]] void malformed(const char* message) { throw DecodeError(ErrorKind::Malformed, message); }
void require(bool okay, const char* message) {
    if (!okay) malformed(message);
}
uint64_t integer(const uint8_t* bytes, unsigned count) {
    uint64_t value = 0;
    for (unsigned i = 0; i < count; ++i) value |= uint64_t(bytes[i]) << (8 * i);
    return value;
}
void charge(size_t count, size_t& remaining, const char* message) {
    if (count > remaining) throw DecodeError(ErrorKind::ResourceLimit, message);
    remaining -= count;
}
// The pinned generated verifier accepts unknown union tags and reads vector
// offsets before checking their destinations. Check every indirect offset before
// Get(), and reject unknown tags before visiting their tables. Schema stays exact.
bool tableShape(flatbuffers::Verifier& v, const void* pointer, std::initializer_list<size_t> widths) {
    // Called only after table verification established a bounded vtable.
    const auto* table = reinterpret_cast<const flatbuffers::Table*>(pointer);
    const auto* vt = table->GetVTable();
    const auto size = flatbuffers::ReadScalar<flatbuffers::voffset_t>(vt);
    if (size < 4 || !v.VerifyFromPointer(vt, size)) return false;
    const auto objectSize = flatbuffers::ReadScalar<flatbuffers::voffset_t>(vt + 2);
    if (objectSize < 4 || !v.VerifyFromPointer(reinterpret_cast<const uint8_t*>(pointer), objectSize)) return false;
    size_t index = 0;
    for (auto width : widths) {
        const auto field = flatbuffers::voffset_t(4 + 2 * index++);
        if (field >= size) break;
        const auto offset = flatbuffers::ReadScalar<flatbuffers::voffset_t>(vt + field);
        if (offset && (offset < 4 || offset > objectSize || width > size_t(objectSize - offset))) return false;
    }
    return true;
}
struct StrictDelta : private flatbuffers::Table {
    template<class T>
    bool tables(flatbuffers::Verifier& v, const flatbuffers::Vector<flatbuffers::Offset<T>>* list) const {
        if (!list) return true;
        for (flatbuffers::uoffset_t i = 0; i < list->size(); ++i) {
            if (!v.VerifyOffset(list->Data() + size_t(i) * sizeof(flatbuffers::uoffset_t), 0)) return false;
            if (!list->Get(i)->Verify(v)) return false;
            if constexpr (std::is_same<T, OmniObjectDelta::SetField>::value) {
                if (!tableShape(v, list->Get(i), {8, 4, 4, 8, 4, 8})) return false;
            } else {
                if (!tableShape(v, list->Get(i), {8, 8, 4, 8, 4})) return false;
            }
        }
        return true;
    }
    bool Verify(flatbuffers::Verifier& v) const {
        const auto* d = reinterpret_cast<const Delta*>(this);
        if (!(VerifyTableStart(v) && tableShape(v, this, {1, 8, 4, 4, 4, 4, 4}) &&
                VerifyField<uint8_t>(v, Delta::VT_ISDIFF) && VerifyField<uint64_t>(v, Delta::VT_BASEVERSION) &&
                VerifyOffset(v, Delta::VT_STRUCTCOMMANDS_TYPE) && v.VerifyVector(d->structCommands_type()) &&
                VerifyOffset(v, Delta::VT_STRUCTCOMMANDS) && v.VerifyVector(d->structCommands()) &&
                VerifyOffset(v, Delta::VT_SETFIELDS) && v.VerifyVector(d->setFields()) &&
                VerifyOffset(v, Delta::VT_TIMESAMPLES) && v.VerifyVector(d->timeSamples()) &&
                VerifyOffset(v, Delta::VT_SOURCEFORMAT) && v.VerifyString(d->sourceFormat())))
            return false;
        return GetField<uint8_t>(Delta::VT_ISDIFF, 0) <= 1 && v.EndTable();
    }
    bool VerifyContents(flatbuffers::Verifier& v, size_t remaining) const {
        // VerifyBuffer has checked the root and every vector span. Charge their
        // combined table count before following any table offsets.
        const auto* d = reinterpret_cast<const Delta*>(this);
        auto types = d->structCommands_type();
        auto commands = d->structCommands();
        if (bool(types) != bool(commands) || (types && types->size() != commands->size())) return false;
        if (commands) charge(commands->size(), remaining, "OMPF record limit exceeded");
        if (d->setFields()) charge(d->setFields()->size(), remaining, "OMPF record limit exceeded");
        if (d->timeSamples()) charge(d->timeSamples()->size(), remaining, "OMPF record limit exceeded");
        if (commands)
            for (flatbuffers::uoffset_t i = 0; i < commands->size(); ++i) {
                const auto type = types->Get(i);
                if (type < StructCommand_CommandCreateSection || type > StructCommand_CommandDiffDeleteSection)
                    return false;
                if (!v.VerifyOffset(commands->Data() + size_t(i) * sizeof(flatbuffers::uoffset_t), 0)) return false;
                if (!VerifyStructCommand(v, commands->Get(i), static_cast<StructCommand>(type))) return false;
                const auto* ptr = commands->Get(i);
                switch (type) {
                case StructCommand_CommandCreateSection:
                    if (!tableShape(v, ptr, {8, 8, 4, 1})) return false;
                    break;
                case StructCommand_CommandDeleteSection:
                    if (!tableShape(v, ptr, {8, 8})) return false;
                    break;
                case StructCommand_CommandMoveSection:
                    if (!tableShape(v, ptr, {8, 8, 8, 4})) return false;
                    break;
                case StructCommand_CommandReorderChildren:
                    if (!tableShape(v, ptr, {8, 1, 4})) return false;
                    break;
                case StructCommand_CommandDiffChangeSection:
                    if (!tableShape(v, ptr, {8, 8, 4, 1, 4})) return false;
                    break;
                case StructCommand_CommandDiffDeleteSection:
                    if (!tableShape(v, ptr, {8})) return false;
                    break;
                }
            }
        return tables(v, d->setFields()) && tables(v, d->timeSamples());
    }
};
}
class EnvelopeParser {
public:
    EnvelopeParser(std::string_view bytes, const EnvelopeLimits& limits)
        : limits_(limits), records_(limits.maxRecords), strings_(limits.maxStringBytes),
          values_(limits.maxReferencedValueBytes) {
        if (!limits.maxBytes || !limits.maxParts || !limits.maxRecords || limits.maxRecords == UINT32_MAX)
            throw std::invalid_argument("invalid OMPF limits");
        if (bytes.size() > limits.maxBytes || bytes.size() >= FLATBUFFERS_MAX_BUFFER_SIZE)
            throw DecodeError(ErrorKind::ResourceLimit, "OMPF byte limit exceeded");
        require(bytes.size() >= 16, "truncated OMPF header");
        const auto* input = reinterpret_cast<const uint8_t*>(bytes.data());
        require(integer(input, 4) == 0x46504d4f, "invalid OMPF magic");
        const auto count = integer(input + 4, 4);
        require(count != 0, "OMPF has no delta part");
        if (count > limits.maxParts) throw DecodeError(ErrorKind::ResourceLimit, "OMPF part limit exceeded");
        require(count <= (bytes.size() - 8) / 8, "truncated OMPF offsets");
        const size_t header = 8 + size_t(count) * 8;
        offsets_.reserve(size_t(count) + 1);
        for (size_t i = 0; i < count; ++i) {
            const auto offset = integer(input + 8 + i * 8, 8);
            require(offset >= header && offset < bytes.size(), "OMPF offset outside content");
            require(offsets_.empty() || offset > offsets_.back(), "OMPF parts overlap or are empty");
            offsets_.push_back(size_t(offset));
        }
        offsets_.push_back(bytes.size());
        storage_ = std::make_shared<EnvelopeStorage>(bytes);
        hashes_.resize(size_t(count));
    }
    Envelope parse() {
        // Padding before part zero is allowed, and need not align its start.
        // Copy just the FlatBuffer into aligned storage for verifier access.
        const size_t size = offsets_[1] - offsets_[0];
        auto aligned = std::make_shared<EnvelopeStorage>(
            std::string_view(reinterpret_cast<const char*>(storage_->data() + offsets_[0]), size));
        flatbuffers::Verifier verifier(aligned->data(), size, 64, limits_.maxRecords + 1);
        require(verifier.VerifyBuffer<StrictDelta>("OODF"), "invalid OODF FlatBuffer");
        require(flatbuffers::GetRoot<StrictDelta>(aligned->data())->VerifyContents(verifier, limits_.maxRecords),
            "invalid OODF contents");
        const auto* delta = GetDelta(aligned->data());
        Envelope result;
        result.isDiff = delta->isDiff();
        result.baseVersion = delta->baseVersion();
        if (delta->sourceFormat()) result.sourceFormat = name(delta->sourceFormat(), true);
        require(result.isDiff || result.sourceFormat.empty(), "sourceFormat on ordinary delta");
        if (const auto* commands = delta->structCommands()) {
            charge(commands->size(), records_, "OMPF record limit exceeded");
            result.commands.reserve(commands->size());
            for (flatbuffers::uoffset_t i = 0; i < commands->size(); ++i)
                result.commands.push_back(
                    command(delta->structCommands_type()->Get(i), commands->Get(i), result.isDiff));
        }
        if (const auto* fields = delta->setFields()) {
            charge(fields->size(), records_, "OMPF record limit exceeded");
            result.fields.reserve(fields->size());
            for (const auto* field : *fields) {
                require(field->sectionId() != 0, "invalid field section ID");
                Field value;
                value.section = field->sectionId();
                value.setOrder = field->setOrder();
                value.name = name(field->keyName());
                value.value = bytes(field->valueOrExtHash(), field->extValueSize(), field->extValueIndex(), aligned,
                    size);
                result.fields.push_back(std::move(value));
            }
        }
        if (const auto* samples = delta->timeSamples()) {
            charge(samples->size(), records_, "OMPF record limit exceeded");
            result.samples.reserve(samples->size());
            for (const auto* sample : *samples) {
                require(sample->sectionId() != 0 && !std::isnan(sample->time()), "invalid time-sample identity/time");
                Sample value;
                value.section = sample->sectionId();
                value.time = sample->time();
                value.value = bytes(sample->valueOrExtHash(), sample->extValueSize(), sample->extValueIndex(), aligned,
                    size);
                result.samples.push_back(std::move(value));
            }
        }
        return result;
    }
private:
    const EnvelopeLimits limits_;
    size_t records_, strings_, values_;
    std::shared_ptr<const EnvelopeStorage> storage_;
    std::vector<size_t> offsets_;
    std::vector<std::optional<std::array<uint8_t, 32>>> hashes_;
    std::string name(const flatbuffers::String* value, bool empty = false) {
        require(value && (empty || value->size() != 0), "missing required name");
        charge(value->size(), strings_, "OMPF string limit exceeded");
        const std::string_view text(value->c_str(), value->size());
        require(detail::validNameUtf8(text), "invalid UTF-8 or NUL in name");
        return std::string(text);
    }
    ValueBytes bytes(const flatbuffers::Vector<uint8_t>* value, uint64_t externalSize, uint32_t index,
        const std::shared_ptr<const EnvelopeStorage>& aligned, size_t flatSize) {
        ValueBytes result;
        if (!externalSize) {
            require(index == 0, "inline value with external index");
            // An absent or empty inline value erases (design section 9). The service's compacted snapshot
            // keeps an erased field as an empty value.
            if (!value || value->size() == 0) return result;
            charge(value->size(), values_, "OMPF referenced-value limit exceeded");
            result.storage_ = aligned;
            result.offset_ = size_t(value->Data() - aligned->data());
            result.size_ = value->size();
            require(result.offset_ <= flatSize && result.size_ <= flatSize - result.offset_, "invalid inline span");
        } else {
            require(index > 0 && index < hashes_.size(), "invalid external part index");
            require(externalSize == offsets_[index + 1] - offsets_[index], "external part size mismatch");
            require(value && value->size() == 32, "external hash must have 32 bytes");
            charge(size_t(externalSize), values_, "OMPF referenced-value limit exceeded");
            if (!hashes_[index]) {
                hashes_[index] = detail::blockHash(std::string_view(
                    reinterpret_cast<const char*>(storage_->data() + offsets_[index]), size_t(externalSize)));
            }
            if (!std::equal(hashes_[index]->begin(), hashes_[index]->end(), value->begin()))
                throw DecodeError(ErrorKind::Integrity, "external value hash mismatch");
            result.storage_ = storage_;
            result.offset_ = offsets_[index];
            result.size_ = size_t(externalSize);
        }
        return result;
    }
    Command command(uint8_t tag, const void* ptr, bool diff) {
        require(diff == (tag >= StructCommand_CommandDiffChangeSection), "command does not match delta mode");
        Command result{};
        switch (tag) {
        case StructCommand_CommandCreateSection: {
            const auto* c = static_cast<const CommandCreateSection*>(ptr);
            result.kind = Command::Kind::Create;
            result.section = c->sectionId();
            result.parent = c->parentId();
            result.sectionType = c->sectionType();
            result.name = name(c->sectionName());
            break;
        }
        case StructCommand_CommandDeleteSection: {
            const auto* c = static_cast<const CommandDeleteSection*>(ptr);
            result.kind = Command::Kind::Delete;
            result.section = c->sectionId();
            result.parent = c->parentId();
            break;
        }
        case StructCommand_CommandMoveSection: {
            const auto* c = static_cast<const CommandMoveSection*>(ptr);
            result.kind = Command::Kind::Move;
            result.section = c->sectionId();
            result.parent = c->newParentId();
            result.oldParent = c->oldParentId();
            result.name = name(c->newName());
            require(result.oldParent != 0 && result.oldParent != result.section, "invalid old parent ID");
            break;
        }
        case StructCommand_CommandReorderChildren: {
            const auto* c = static_cast<const CommandReorderChildren*>(ptr);
            result.kind = Command::Kind::Reorder;
            result.section = c->sectionId();
            result.childList = c->childrenListId();
            require(result.section != 0 && result.childList < 3 && c->childrenList(), "invalid child reorder");
            charge(c->childrenList()->size(), records_, "OMPF child-ID limit exceeded");
            // The list is an order, which USD and the service apply by ignoring
            // IDs that are not children of the section and every repeat after
            // an ID's first occurrence, so any ID and any repeat decode.
            result.children.assign(c->childrenList()->begin(), c->childrenList()->end());
            return result;
        }
        case StructCommand_CommandDiffChangeSection: {
            const auto* c = static_cast<const CommandDiffChangeSection*>(ptr);
            result.kind = Command::Kind::DiffChange;
            result.section = c->sectionId();
            result.parent = c->parentId();
            result.sectionType = c->sectionType();
            result.sectionOrder = c->sectionOrder();
            result.name = name(c->sectionName(), result.section == 1);
            if (result.section == 1) {
                require(result.parent == 0 && result.sectionType == 7 && result.name.empty(), "invalid diff root");
                return result;
            }
            break;
        }
        case StructCommand_CommandDiffDeleteSection: {
            const auto* c = static_cast<const CommandDiffDeleteSection*>(ptr);
            result.kind = Command::Kind::DiffDelete;
            result.section = c->sectionId();
            require(result.section > 1, "invalid diff deletion ID");
            return result;
        }
        default: malformed("unknown structural command");
        }
        require(result.section > 1 && result.parent != 0 && result.parent != result.section,
            "invalid section/parent ID");
        if (result.kind == Command::Kind::Create || result.kind == Command::Kind::DiffChange)
            require(result.sectionType >= 1 && result.sectionType <= 11 && result.sectionType != 7,
                "invalid section type");
        return result;
    }
};
Envelope decodeEnvelope(std::string_view bytes, const EnvelopeLimits& limits) {
    return EnvelopeParser(bytes, limits).parse();
}
} // namespace omni::usd::live::wire
