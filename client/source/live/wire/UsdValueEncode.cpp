// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "UsdValueWriter.h"
#include <typeindex>
#include <unordered_map>

namespace omni::usd::live::wire {
namespace detail {
void Writer::tagged(const VtValue& value) {
    if (value.IsEmpty()) throw DecodeError(ErrorKind::Unsupported, "nested empty USD value is not representable");
    // Fixed-size, process-owned registration; input cannot add entries. Scalar
    // RTTI and the array RTTI of array-capable types resolve to the unchanged
    // version-zero type table.
    static const std::unordered_map<std::type_index, uint16_t> types = [] {
        std::unordered_map<std::type_index, uint16_t> result;
#define USD_TYPE(NAME, ID, CPP, VERSION) \
        result.emplace(typeid(CPP), ID); \
        if (hasArrayForm(ID)) result.emplace(typeid(VtArray<CPP>), ID | 0x80);
#include "UsdValueTypes.inc"
        return result;
    }();
    const auto found = types.find(value.GetTypeid());
    if (found == types.end()) throw DecodeError(ErrorKind::Unsupported, "unsupported USD value for encoding");
    if ((found->second & 0x7f) <= 30) numeric(found->second, value);
    else compound(found->second, value);
}
namespace {
void pathTopology(const SdfPath& path, EncodeBudget& budget, size_t& nameBytes) {
    for (const auto& prefix : path.GetAncestorsRange()) {
        spend(1, budget.elements, "USD encoding path element limit exceeded");
        if (prefix.IsTargetPath() || prefix.IsMapperPath()) {
            EncodeDepth depth(budget);
            pathTopology(prefix.GetTargetPath(), budget, nameBytes);
        } else {
            // Borrowed names let ordinary paths fail a tiny byte budget without
            // first materializing their entire string. Variant-set names are
            // not exposed this way by the pinned SDK (see path() below).
            const auto size = prefix.GetNameToken().GetString().size();
            spend(size, nameBytes, "USD encoding path string limit exceeded");
        }
    }
}
}
void Writer::path(const SdfPath& value) {
    if (budget_.encoded < 4) throw DecodeError(ErrorKind::ResourceLimit, "USD encoded-byte limit exceeded");
    size_t names = std::min(budget_.strings, budget_.encoded - 4);
    pathTopology(value, budget_, names);
    // GetString returns a borrowed cached string. USD may allocate that cache
    // on first use, including variant-set text not exposed as borrowed tokens.
    // Our limits bound output and traversal, not these SDK-owned cache bytes.
    // Target recursion has already been bounded before calling into USD.
    string(value.GetString());
}
std::string emit(const VtValue& value, size_t size, EncodeBudget& budget) {
    if (size > std::string().max_size())
        throw DecodeError(ErrorKind::ResourceLimit, "USD output string limit exceeded");
    std::string result(size, '\0');
    Writer writer(budget, &result);
    writer.top(value);
    check(writer.size() == size, "USD value changed during encoding");
    return result;
}
} // namespace detail
std::string encodeValue(const PXR_NS::VtValue& value, const EncodeLimits& limits) {
    detail::EncodeBudget budget(limits);
    detail::Writer measure(budget);
    measure.top(value);
    detail::EncodeBudget emission(limits);
    return detail::emit(value, measure.size(), emission);
}
EncodedValues encodeValues(const DecodedValues& values, const EncodeLimits& limits) {
    if (values.fields.size() > limits.maxElements || values.samples.size() > limits.maxElements - values.fields.size())
        throw DecodeError(ErrorKind::ResourceLimit, "USD encoding batch element limit exceeded");
    detail::EncodeBudget budget(limits);
    detail::Writer measure(budget);
    std::vector<size_t> sizes;
    sizes.reserve(values.fields.size() + values.samples.size());
    for (const auto* source : {&values.fields, &values.samples})
        for (const auto& value : *source) {
            const auto before = measure.size();
            measure.top(value);
            sizes.push_back(measure.size() - before);
        }
    // Every type, shape, offset and quota has passed before allocating payloads.
    EncodedValues result;
    result.fields.reserve(values.fields.size());
    result.samples.reserve(values.samples.size());
    detail::EncodeBudget emission(limits);
    size_t index = 0;
    for (const auto& value : values.fields) result.fields.push_back(detail::emit(value, sizes[index++], emission));
    for (const auto& value : values.samples) result.samples.push_back(detail::emit(value, sizes[index++], emission));
    return result;
}
size_t measureValues(const DecodedValues& values, const EncodeLimits& limits, size_t* elementsUsed, size_t* stringsUsed,
    size_t* depthUsed) {
    if (values.fields.size() > limits.maxElements || values.samples.size() > limits.maxElements - values.fields.size())
        throw DecodeError(ErrorKind::ResourceLimit, "USD encoding batch element limit exceeded");
    detail::EncodeBudget budget(limits);
    detail::Writer measure(budget);
    for (const auto* source : {&values.fields, &values.samples})
        for (const auto& value : *source) measure.top(value);
    if (elementsUsed) *elementsUsed = limits.maxElements - budget.elements;
    if (stringsUsed) *stringsUsed = limits.maxStringBytes - budget.strings;
    if (depthUsed) *depthUsed = budget.deepest;
    return measure.size();
}
} // namespace omni::usd::live::wire
