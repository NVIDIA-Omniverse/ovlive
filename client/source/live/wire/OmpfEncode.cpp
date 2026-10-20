// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "OmpfEncode.h"
#include "OmpfEncodeInternal.h"
#include "OmpfHash.h"
#include "OmpfValidation.h"
#include "schema/OmniObjectDelta_generated.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

namespace omni::usd::live::wire {
namespace {
using namespace OmniObjectDelta;
template<class T> struct OffsetTable;
template<class T> struct OffsetTable<flatbuffers::Offset<T>> {
    using type = T;
};
[[noreturn]] void limit(const char* message) { throw DecodeError(ErrorKind::ResourceLimit, message); }
void require(bool good, const char* message) {
    if (!good) throw DecodeError(ErrorKind::Malformed, message);
}
void charge(size_t size, size_t& remaining, const char* message) {
    if (size > remaining) limit(message);
    remaining -= size;
}
void chargeProduct(size_t n, size_t width, size_t& remaining, const char* message) {
    if (n > remaining / width) limit(message);
    remaining -= n * width;
}
class BoundedAllocator final : public flatbuffers::Allocator {
public:
    explicit BoundedAllocator(size_t maximum) : maximum_(maximum) {}
    uint8_t* allocate(size_t size) override {
        if (size >= FLATBUFFERS_MAX_BUFFER_SIZE || size > maximum_ - live_) limit("OMPF builder memory limit exceeded");
        for (auto& slot : slots_)
            if (!slot.pointer) {
                auto* result = new uint8_t[size];
                slot = {result, size};
                live_ += size;
                return result;
            }
        limit("OMPF builder allocation overlap exceeded");
    }
    void deallocate(uint8_t* p, size_t) override {
        // The pinned builder updates reserved size before growing. If growth
        // throws, its destructor reports that new size for the old pointer.
        // Track actual allocations rather than trusting the supplied size.
        for (auto& slot : slots_)
            if (slot.pointer == p) {
                live_ -= slot.size;
                slot = {};
                delete[] p;
                return;
            }
        std::terminate();
    }
private:
    struct Slot {
        uint8_t* pointer = nullptr;
        size_t size = 0;
    };
    std::array<Slot, 2> slots_{};
    const size_t maximum_;
    size_t live_ = 0;
};
struct Plan {
    size_t externalCount = 0, externalBytes = 0;
};
Plan preflight(const EnvelopeInput& input, const EnvelopeEncodeLimits& options) {
    const auto& limits = options.content;
    if (!limits.maxBytes || !limits.maxParts || !limits.maxRecords || limits.maxRecords == UINT32_MAX ||
        !options.externalValueThreshold || !options.maxBuilderBytes)
        throw std::invalid_argument("invalid OMPF encoding limits");
    size_t records = limits.maxRecords, strings = limits.maxStringBytes, values = limits.maxReferencedValueBytes;
    size_t wire = std::min<size_t>(limits.maxBytes, FLATBUFFERS_MAX_BUFFER_SIZE - 1), scratch = options.maxScratchBytes;
    // Lower bound only; final FlatBuffer tables/padding are checked exactly.
    charge(16, wire, "OMPF encoded-byte limit exceeded");
    for (size_t n : {input.commands.size(), input.fields.size(), input.samples.size()}) {
        charge(n, records, "OMPF record limit exceeded");
        chargeProduct(n, 4, wire, "OMPF encoded-byte limit exceeded");
        chargeProduct(n, sizeof(flatbuffers::uoffset_t), scratch, "OMPF scratch limit exceeded");
    }
    charge(input.commands.size(), scratch, "OMPF scratch limit exceeded"); // union tags
    charge(input.commands.size(), wire, "OMPF encoded-byte limit exceeded");
    auto name = [&](std::string_view value, bool empty = false) {
        require(empty || !value.empty(), "missing required OMPF name");
        charge(value.size(), strings, "OMPF string limit exceeded");
        charge(value.size(), wire, "OMPF encoded-byte limit exceeded");
        require(detail::validNameUtf8(value), "invalid OMPF name UTF-8 or NUL");
    };
    require(input.isDiff || input.sourceFormat.empty(), "sourceFormat on ordinary delta");
    if (!input.sourceFormat.empty()) name(input.sourceFormat);
    for (const auto& c : input.commands) {
        const bool diffCommand = c.kind == Command::Kind::DiffChange || c.kind == Command::Kind::DiffDelete;
        require(input.isDiff == diffCommand, "OMPF command does not match delta mode");
        if (c.kind != Command::Kind::Reorder) require(c.children.empty() && !c.childList, "unused OMPF child metadata");
        if (c.kind != Command::Kind::Move) require(!c.oldParent, "unused OMPF old parent");
        if (c.kind != Command::Kind::DiffChange) require(!c.sectionOrder, "unused OMPF section order");
        if (c.kind != Command::Kind::Create && c.kind != Command::Kind::DiffChange)
            require(!c.sectionType, "unused OMPF section type");
        switch (c.kind) {
        case Command::Kind::Create:
        case Command::Kind::DiffChange:
            name(c.name, c.kind == Command::Kind::DiffChange && c.section == 1);
            if (c.kind == Command::Kind::DiffChange && c.section == 1) {
                require(!c.parent && c.sectionType == 7 && c.name.empty(), "invalid OMPF diff root");
                break;
            }
            require(c.section > 1 && c.parent && c.parent != c.section, "invalid OMPF section/parent");
            require(c.sectionType >= 1 && c.sectionType <= 11 && c.sectionType != 7, "invalid OMPF section type");
            break;
        case Command::Kind::Move:
            name(c.name);
            require(c.section > 1 && c.parent && c.oldParent && c.parent != c.section && c.oldParent != c.section,
                "invalid OMPF move IDs");
            break;
        case Command::Kind::Delete:
            require(c.name.empty() && c.section > 1 && c.parent && c.parent != c.section, "invalid OMPF deletion");
            break;
        case Command::Kind::DiffDelete:
            require(c.name.empty() && c.section > 1 && !c.parent, "invalid OMPF diff deletion");
            break;
        case Command::Kind::Reorder: {
            require(c.section && !c.parent && c.name.empty() && c.childList < 3, "invalid OMPF reorder");
            charge(c.children.size(), records, "OMPF child-ID limit exceeded");
            chargeProduct(c.children.size(), 8, wire, "OMPF encoded-byte limit exceeded");
            chargeProduct(c.children.size(), sizeof(uint64_t), scratch, "OMPF scratch limit exceeded");
            auto sorted = c.children;
            for (auto id : sorted) require(id > 1 && id != c.section, "invalid OMPF reordered child");
            std::sort(sorted.begin(), sorted.end());
            require(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end(), "duplicate OMPF reordered child");
            break;
        }
        default: require(false, "unknown OMPF command");
        }
    }
    Plan plan;
    auto value = [&](std::string_view bytes) {
        charge(bytes.size(), values, "OMPF referenced-value limit exceeded");
        charge(bytes.size(), wire, "OMPF encoded-byte limit exceeded");
        if (!bytes.empty() && bytes.size() >= options.externalValueThreshold) {
            if (++plan.externalCount >= limits.maxParts) limit("OMPF part limit exceeded");
            charge(8 + 32, wire, "OMPF encoded-byte limit exceeded"); // offset + hash bytes
            charge(sizeof(std::string_view), scratch, "OMPF scratch limit exceeded");
            charge(detail::blockHashScratch(bytes.size()), scratch, "OMPF scratch limit exceeded");
            plan.externalBytes += bytes.size(); // Already bounded by total wire budget.
        }
    };
    for (const auto& f : input.fields) {
        require(f.section != 0, "invalid OMPF field section");
        name(f.name);
        value(f.value);
    }
    for (const auto& s : input.samples) {
        require(s.section && !std::isnan(s.time), "invalid OMPF sample identity/time");
        value(s.value);
    }
    return plan;
}
struct Parts {
    flatbuffers::Offset<flatbuffers::Vector<uint8_t>> data;
    uint64_t size = 0;
    uint32_t index = 0;
};
void integer(char* out, uint64_t value, size_t width) {
    for (size_t i = 0; i < width; ++i) out[i] = char(value >> (8 * i));
}
}
void detail::validateEnvelopeInput(const EnvelopeInput& input, const EnvelopeEncodeLimits& options) {
    (void)preflight(input, options);
}
std::string encodeEnvelope(const EnvelopeInput& input, const EnvelopeEncodeLimits& options) {
    const auto plan = preflight(input, options);
    BoundedAllocator allocator(options.maxBuilderBytes);
    flatbuffers::FlatBufferBuilder b(1024, &allocator);
    b.DedupVtables(false); // Bound builder work independently of vtable history.
    std::vector<uint8_t> types;
    types.reserve(input.commands.size());
    std::vector<flatbuffers::Offset<void>> commands;
    commands.reserve(input.commands.size());
    std::vector<flatbuffers::Offset<SetField>> fields;
    fields.reserve(input.fields.size());
    std::vector<flatbuffers::Offset<TimeSample>> samples;
    samples.reserve(input.samples.size());
    std::vector<std::string_view> external;
    external.reserve(plan.externalCount);
    auto add = [&](auto command) {
        using Table = typename OffsetTable<decltype(command)>::type;
        types.push_back(StructCommandTraits<Table>::enum_value);
        commands.push_back(command.Union());
    };
    for (const auto& c : input.commands) switch (c.kind) {
        case Command::Kind::Create:
            add(CreateCommandCreateSection(b, c.parent, c.section, b.CreateString(c.name), c.sectionType));
            break;
        case Command::Kind::Delete: add(CreateCommandDeleteSection(b, c.parent, c.section)); break;
        case Command::Kind::Move:
            add(CreateCommandMoveSection(b, c.oldParent, c.parent, c.section, b.CreateString(c.name)));
            break;
        case Command::Kind::Reorder:
            add(CreateCommandReorderChildren(b, c.section, c.childList, b.CreateVector(c.children)));
            break;
        case Command::Kind::DiffChange:
            add(CreateCommandDiffChangeSection(b, c.parent, c.section, b.CreateString(c.name), c.sectionType,
                c.sectionOrder));
            break;
        case Command::Kind::DiffDelete: add(CreateCommandDiffDeleteSection(b, c.section)); break;
        }
    auto value = [&](std::string_view bytes) {
        Parts result;
        if (bytes.empty()) return result; // Deletion must omit the vector.
        if (bytes.size() < options.externalValueThreshold) {
            result.data = b.CreateVector(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size());
        } else {
            const auto hash = detail::blockHash(bytes);
            result.data = b.CreateVector(hash.data(), hash.size());
            result.size = bytes.size();
            external.push_back(bytes);
            result.index = uint32_t(external.size());
        }
        return result;
    };
    for (const auto& f : input.fields) {
        const auto bytes = value(f.value);
        fields.push_back(
            CreateSetField(b, f.section, b.CreateString(f.name), bytes.data, bytes.size, bytes.index, f.setOrder));
    }
    for (const auto& s : input.samples) {
        const auto bytes = value(s.value);
        b.ForceDefaults(true); // Keep signed-zero time bits as well as infinities.
        samples.push_back(CreateTimeSample(b, s.section, s.time, bytes.data, bytes.size, bytes.index));
        b.ForceDefaults(false);
    }
    const auto root = CreateDeltaDirect(b, input.isDiff, input.baseVersion, &types, &commands, &fields, &samples,
        input.sourceFormat.empty() ? nullptr : input.sourceFormat.c_str());
    FinishDeltaBuffer(b, root);
    const size_t header = 16 + 8 * plan.externalCount;
    size_t remaining = std::min<size_t>(options.content.maxBytes, FLATBUFFERS_MAX_BUFFER_SIZE - 1);
    charge(header, remaining, "OMPF encoded-byte limit exceeded");
    charge(b.GetSize(), remaining, "OMPF encoded-byte limit exceeded");
    charge(plan.externalBytes, remaining, "OMPF encoded-byte limit exceeded");
    const size_t total = header + b.GetSize() + plan.externalBytes;
    std::string result(total, '\0');
    integer(result.data(), 0x46504d4f, 4);
    integer(result.data() + 4, 1 + plan.externalCount, 4);
    integer(result.data() + 8, header, 8);
    std::memcpy(result.data() + header, b.GetBufferPointer(), b.GetSize());
    size_t offset = header + b.GetSize();
    for (size_t i = 0; i < external.size(); ++i) {
        integer(result.data() + 16 + i * 8, offset, 8);
        std::memcpy(result.data() + offset, external[i].data(), external[i].size());
        offset += external[i].size();
    }
    return result;
}
} // namespace omni::usd::live::wire
