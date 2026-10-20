// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "UsdValueReader.h"

namespace omni::usd::live::wire {
namespace detail {
void Reader::assetString(const std::string& text) {
    for (size_t i = 0; i < text.size();) {
        const uint8_t first = uint8_t(text[i++]);
        uint32_t value = first, minimum = 0;
        unsigned count = 0;
        if (first < 0x80) {
        } else if (first >= 0xc2 && first <= 0xdf) {
            count = 1;
            value &= 31;
            minimum = 0x80;
        } else if (first >= 0xe0 && first <= 0xef) {
            count = 2;
            value &= 15;
            minimum = 0x800;
        } else if (first >= 0xf0 && first <= 0xf4) {
            count = 3;
            value &= 7;
            minimum = 0x10000;
        } else invalid("invalid USD asset-path UTF-8");
        check(count <= text.size() - i, "truncated USD asset-path UTF-8");
        while (count--) {
            const uint8_t next = uint8_t(text[i++]);
            check((next & 0xc0) == 0x80, "invalid USD asset-path UTF-8");
            value = (value << 6) | (next & 63);
        }
        check(value >= minimum && value <= 0x10ffff && !(value >= 0xd800 && value <= 0xdfff),
            "invalid USD asset-path UTF-8");
        check(value >= 0x20 && !(value >= 0x7f && value <= 0x9f), "control character in USD asset path");
    }
}
void Reader::pathStructure(const std::string& text) {
    if (text.empty()) return;
    size_t nodes = 1, nesting = 0;
    for (unsigned char ch : text) {
        // This deliberately overcounts: USD's syntax validator still decides
        // validity. Every path/target/variant node must fit admission before
        // USD allocates or recursively parses it.
        if (ch == '/' || ch == '.' || ch == '[' || ch == ']' || ch == '{' || ch == '}') ++nodes;
        if (ch == '[' || ch == '{') {
            if (nesting >= budget_.maxDepth - budget_.depth)
                throw DecodeError(ErrorKind::ResourceLimit, "USD path nesting limit exceeded");
            ++nesting;
        } else if ((ch == ']' || ch == '}') && nesting) --nesting;
    }
    spend(nodes, budget_.elements, "USD path element limit exceeded");
    budget_.storage(nodes, 256); // conservative parser/interned-node charge.
    budget_.storage(text.size() + 1, 2); // parser and interned name copies.
}
VtValue Reader::tagged() {
    const uint16_t tag = uint16_t(integer(2));
    if ((tag >> 8) != 0) throw DecodeError(ErrorKind::Unsupported, "unsupported USD value version");
    const unsigned id = tag & 0x7f;
    // Kit writes a scalar or array value of a type outside the table as type 0 with no body, and reads it as
    // an empty value.
    if (id == 0) {
        spend(1, budget_.elements, "USD element limit exceeded");
        return {};
    }
    if (id > 64) throw DecodeError(ErrorKind::Unsupported, "unsupported USD value type");
    if ((tag & 0x80) && !hasArrayForm(id))
        throw DecodeError(ErrorKind::Unsupported, "USD value type has no array form");
    return id <= 30 ? numeric(tag) : compound(tag);
}
VtValue decode(std::string_view bytes, Budget& budget) {
    spend(bytes.size(), budget.encoded, "USD encoded-byte limit exceeded");
    if (bytes.empty()) { // Envelope absence, not an encoded unknown/nested value.
        spend(1, budget.elements, "USD element limit exceeded");
        budget.storage(1, sizeof(VtValue));
        return {};
    }
    Reader reader(bytes, budget);
    auto result = reader.tagged();
    check(reader.left() == 0, "trailing bytes in USD value");
    return result;
}
} // namespace detail
PXR_NS::VtValue decodeValue(std::string_view bytes, const ValueLimits& limits) {
    detail::Budget budget(limits);
    return detail::decode(bytes, budget);
}
DecodedValues decodeValues(const Envelope& envelope, const ValueLimits& limits, ValueUsage* usage) {
    detail::Budget budget(limits);
    budget.storage(envelope.fields.size(), sizeof(PXR_NS::VtValue));
    budget.storage(envelope.samples.size(), sizeof(PXR_NS::VtValue));
    DecodedValues result;
    result.fields.reserve(envelope.fields.size());
    result.samples.reserve(envelope.samples.size());
    for (const auto& field : envelope.fields) result.fields.push_back(detail::decode(field.value.bytes(), budget));
    for (const auto& sample : envelope.samples) result.samples.push_back(detail::decode(sample.value.bytes(), budget));
    if (usage)
        *usage = {limits.maxEncodedBytes - budget.encoded, limits.maxDecodedBytes - budget.decoded,
            limits.maxElements - budget.elements, limits.maxComparisonWork - budget.comparison};
    return result;
}
} // namespace omni::usd::live::wire
