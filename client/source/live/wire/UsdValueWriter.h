// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "UsdValueReader.h"

namespace omni::usd::live::wire::detail {
struct EncodeBudget {
    explicit EncodeBudget(const EncodeLimits& limits)
        : encoded(limits.maxEncodedBytes), elements(limits.maxElements), strings(limits.maxStringBytes),
          maxDepth(limits.maxDepth) {
        if (!maxDepth || maxDepth > 256) throw std::invalid_argument("invalid USD encoding depth limit");
    }
    size_t encoded, elements, strings, maxDepth, depth = 0;
    size_t deepest = 0; // The greatest nesting depth any measured value reached.
};
struct EncodeDepth {
    EncodeBudget& budget;
    explicit EncodeDepth(EncodeBudget& b) : budget(b) {
        if (b.depth == b.maxDepth) throw DecodeError(ErrorKind::ResourceLimit, "USD encoding nesting limit exceeded");
        if (++b.depth > b.deepest) b.deepest = b.depth;
    }
    ~EncodeDepth() { --budget.depth; }
};
class Writer {
public:
    Writer(EncodeBudget& budget, std::string* output = nullptr) : budget_(budget), output_(output) {}
    size_t size() const { return position_; }
    void bytes(std::string_view data) {
        spend(data.size(), budget_.encoded, "USD encoded-byte limit exceeded");
        check(data.size() <= SIZE_MAX - position_, "USD encoded size overflow");
        if (output_) {
            check(position_ <= output_->size() && data.size() <= output_->size() - position_,
                "USD value changed during encoding");
            if (!data.empty()) std::memcpy(output_->data() + position_, data.data(), data.size());
        }
        position_ += data.size();
    }
    void integer(uint64_t value, size_t width) {
        char data[8];
        for (size_t i = 0; i < width; ++i) data[i] = char(value >> (8 * i));
        bytes(std::string_view(data, width));
    }
    void string(const std::string& value) {
        if (value.size() > UINT32_MAX) throw DecodeError(ErrorKind::ResourceLimit, "USD string wire limit exceeded");
        spend(value.size(), budget_.strings, "USD encoding string-byte limit exceeded");
        integer(value.size(), 4);
        bytes(value);
    }
    void count(size_t n, size_t minimum) {
        if (n > budget_.elements || (minimum && n > budget_.encoded / minimum))
            throw DecodeError(ErrorKind::ResourceLimit, "USD encoding container limit exceeded");
        integer(n, 8);
    }
    void top(const VtValue& value) {
        if (value.IsEmpty()) spend(1, budget_.elements, "USD encoding element limit exceeded");
        else tagged(value);
    }
    void tagged(const VtValue&);
    void numeric(uint16_t, const VtValue&);
    void compound(uint16_t, const VtValue&);
    void path(const SdfPath&);
    template<ValueType V> void typed(uint16_t tag, const VtValue& value) {
        using T = typename WireType<V>::Type;
        if constexpr (hasArrayForm(unsigned(V))) {
            integer(tag, 2);
            if (!(tag & 0x80)) {
                write(value.UncheckedGet<T>());
                return;
            }
            EncodeDepth depth(budget_);
            spend(1, budget_.elements, "USD encoding element limit exceeded");
            const auto& values = value.UncheckedGet<VtArray<T>>();
            count(values.size(), minimumBytes<T>());
            for (const auto& item : values) write(item);
        } else {
            if (tag & 0x80) throw DecodeError(ErrorKind::Unsupported, "USD value type has no array form");
            integer(tag, 2);
            write(value.UncheckedGet<T>());
        }
    }
    template<class T> void write(const T& value) {
        EncodeDepth depth(budget_);
        spend(1, budget_.elements, "USD encoding element limit exceeded");
        if constexpr (std::is_same_v<T, bool>) {
            integer(value ? 1 : 0, 1);
        } else if constexpr (std::is_integral_v<T>) {
            integer(static_cast<std::make_unsigned_t<T>>(value), sizeof(T));
        } else if constexpr (std::is_floating_point_v<T>) {
            static_assert(std::numeric_limits<T>::is_iec559);
            using U = std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>;
            U bits;
            std::memcpy(&bits, &value, sizeof(T));
            integer(bits, sizeof(T));
        } else if constexpr (std::is_same_v<T, GfHalf>) {
            integer(value.bits(), 2);
        } else if constexpr (std::is_enum_v<T>) {
            // Reading an out-of-range enum as an enum can itself be UB. Inspect
            // its fixed-width representation before any cast or comparison.
            static_assert(sizeof(T) == 4);
            uint32_t raw;
            std::memcpy(&raw, &value, 4);
            constexpr auto end = [] {
                if constexpr (std::is_same_v<T, SdfSpecifier>) return int(SdfNumSpecifiers);
                else if constexpr (std::is_same_v<T, SdfPermission>) return int(SdfNumPermissions);
                else if constexpr (std::is_same_v<T, SdfVariability>) return int(SdfNumVariabilities);
                else {
                    static_assert(std::is_same_v<T, SdfSpecType>);
                    return int(SdfNumSpecTypes);
                }
            }();
            check(raw < uint32_t(end), "invalid USD enum for encoding");
            integer(raw, 4);
        } else if constexpr (GfIsGfVec<T>::value) {
            for (size_t i = 0; i < T::dimension; ++i) write(value[i]);
        } else if constexpr (GfIsGfMatrix<T>::value) {
            for (size_t i = 0; i < T::numRows * T::numColumns; ++i) write(value.GetArray()[i]);
        } else if constexpr (GfIsGfQuat<T>::value) {
            write(value.GetImaginary());
            write(value.GetReal());
        } else if constexpr (GfIsGfRange<T>::value) {
            write(value.GetMin());
            write(value.GetMax());
        } else if constexpr (std::is_same_v<T, std::string>) {
            string(value);
        } else if constexpr (std::is_same_v<T, TfToken>) {
            const auto& text = value.GetString();
            string(text);
            check(text.find('\0') == std::string::npos, "NUL in USD token for encoding");
        } else if constexpr (std::is_same_v<T, SdfPath>) {
            path(value);
        } else if constexpr (std::is_same_v<T, SdfAssetPath>) {
            string(value.GetAuthoredPath());
            Reader::assetString(value.GetAuthoredPath());
        } else if constexpr (std::is_same_v<T, SdfLayerOffset>) {
            check(value.IsValid(), "invalid USD layer offset for encoding");
            write(value.GetOffset());
            write(value.GetScale());
        } else if constexpr (std::is_same_v<T, SdfReference>) {
            string(value.GetAssetPath());
            Reader::assetString(value.GetAssetPath());
            write(value.GetPrimPath());
            write(value.GetLayerOffset());
            write(value.GetCustomData());
        } else if constexpr (std::is_same_v<T, SdfPayload>) {
            const auto& offset = value.GetLayerOffset();
            if (offset.GetOffset() != 0.0 || offset.GetScale() != 1.0)
                throw DecodeError(ErrorKind::Unsupported, kPayloadOffsetUnsupported);
            string(value.GetAssetPath());
            Reader::assetString(value.GetAssetPath());
            write(value.GetPrimPath());
        } else if constexpr (Vector<T>::value) {
            count(value.size(), minimumBytes<typename Vector<T>::Item>());
            for (const auto& item : value) write(item);
        } else if constexpr (std::is_same_v<T, VtDictionary> || std::is_same_v<T, SdfVariantSelectionMap> ||
            std::is_same_v<T, SdfTimeSampleMap>) {
            using Key = typename T::key_type;
            using Value = typename T::mapped_type;
            count(value.size(), minimumBytes<Key>() + minimumBytes<Value>());
            for (const auto& [key, item] : value) {
                if constexpr (std::is_same_v<Key, double>)
                    check(!std::isnan(key), "invalid USD sample time for encoding");
                write(key);
                write(item);
            }
        } else if constexpr (List<T>::value) {
            const unsigned bits = (value.IsExplicit() ? 1 : 0) | (!value.GetExplicitItems().empty() ? 2 : 0) |
                (!value.GetAddedItems().empty() ? 4 : 0) | (!value.GetDeletedItems().empty() ? 8 : 0) |
                (!value.GetOrderedItems().empty() ? 16 : 0) | (!value.GetPrependedItems().empty() ? 32 : 0) |
                (!value.GetAppendedItems().empty() ? 64 : 0);
            check((!(bits & 2) || (bits & 1)) && (!(bits & 1) || !(bits & 0x7c)),
                "invalid USD list-op mode for encoding");
            integer(bits, 1);
            if (bits & 2) write(value.GetExplicitItems());
            if (bits & 4) write(value.GetAddedItems());
            if (bits & 32) write(value.GetPrependedItems());
            if (bits & 64) write(value.GetAppendedItems());
            if (bits & 8) write(value.GetDeletedItems());
            if (bits & 16) write(value.GetOrderedItems());
        } else if constexpr (std::is_same_v<T, SdfValueBlock>) {
            integer(0, 1); // Deterministic byte for the reserved empty-object slot.
        } else if constexpr (std::is_same_v<T, VtValue>) {
            tagged(value);
        } else if constexpr (std::is_same_v<T, SdfUnregisteredValue>) {
            const VtValue& nested = value.GetValue();
            if (!nested.IsHolding<std::string>() && !nested.IsHolding<VtDictionary>() &&
                !nested.IsHolding<SdfUnregisteredValueListOp>())
                throw DecodeError(ErrorKind::Unsupported, "unsupported USD unregistered value for encoding");
            tagged(nested);
        } else if constexpr (std::is_same_v<T, SdfTimeCode>) {
            write(value.GetValue());
        } else static_assert(!sizeof(T), "unhandled wire value type");
    }
private:
    EncodeBudget& budget_;
    std::string* output_;
    size_t position_ = 0;
};
} // namespace omni::usd::live::wire::detail
