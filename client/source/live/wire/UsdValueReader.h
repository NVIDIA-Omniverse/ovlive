// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "UsdValueCodec.h"
#include "SdfIncludes.h"
#include <pxr/usd/sdf/timeCode.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <type_traits>

namespace omni::usd::live::wire::detail {
using namespace PXR_NS;
static_assert(sizeof(int) == 4 && sizeof(unsigned int) == 4 && sizeof(GfHalf) == 2 && sizeof(float) == 4 &&
        sizeof(double) == 8,
    "unsupported USD value scalar ABI");
enum class ValueType : unsigned {
#define USD_TYPE(NAME, ID, CPP, VERSION) NAME = ID,
#include "UsdValueTypes.inc"
};
template<ValueType> struct WireType;
#define USD_TYPE(NAME, ID, CPP, VERSION) template<> struct WireType<ValueType::NAME> { using Type = CPP; };
#include "UsdValueTypes.inc"
// USD stores arrays only of the numeric, string, token, asset and Gf types
// (IDs 1-30) and TimeCode (58): the crate format's array-capable types and
// SdfValueTypeNames' array names. No other type has a VtArray form on the
// wire, in either direction.
constexpr bool hasArrayForm(unsigned id) noexcept {
    return (id >= 1 && id <= 30) || id == unsigned(ValueType::TimeCode);
}

[[noreturn]] inline void invalid(const char* message) { throw DecodeError(ErrorKind::Malformed, message); }
inline void check(bool condition, const char* message) {
    if (!condition) invalid(message);
}
inline void spend(size_t amount, size_t& left, const char* message) {
    if (amount > left) throw DecodeError(ErrorKind::ResourceLimit, message);
    left -= amount;
}
struct Budget {
    explicit Budget(const ValueLimits& l)
        : encoded(l.maxEncodedBytes), decoded(l.maxDecodedBytes), elements(l.maxElements), strings(l.maxStringBytes),
          maxDepth(l.maxDepth), comparison(l.maxComparisonWork) {
        if (!maxDepth || maxDepth > 256) throw std::invalid_argument("invalid USD value depth limit");
    }
    size_t encoded, decoded, elements, strings, maxDepth, comparison, depth = 0;
    void storage(size_t count, size_t size) {
        if (size && count > decoded / size)
            throw DecodeError(ErrorKind::ResourceLimit, "USD decoded-byte limit exceeded");
        spend(count * size, decoded, "USD decoded-byte limit exceeded");
    }
};
struct Depth {
    Budget& budget;
    explicit Depth(Budget& b) : budget(b) {
        if (b.depth == b.maxDepth) throw DecodeError(ErrorKind::ResourceLimit, "USD value nesting limit exceeded");
        ++b.depth;
    }
    ~Depth() { --budget.depth; }
};
template<class T> struct Vector : std::false_type {};
template<class T, class A> struct Vector<std::vector<T, A>> : std::true_type {
    using Item = T;
};
template<class T> struct List : std::false_type {};
template<class T> struct List<SdfListOp<T>> : std::true_type {
    using Item = T;
};
template<class T> constexpr size_t minimumBytes() {
    if constexpr (std::is_arithmetic_v<T> || std::is_enum_v<T> || std::is_same_v<T, GfHalf>) return sizeof(T);
    else if constexpr (GfIsGfVec<T>::value) return T::dimension * sizeof(typename T::ScalarType);
    else if constexpr (GfIsGfMatrix<T>::value) return T::numRows * T::numColumns * sizeof(typename T::ScalarType);
    else if constexpr (GfIsGfQuat<T>::value) return 4 * sizeof(typename T::ScalarType);
    else if constexpr (GfIsGfRange<T>::value) return 2 * T::dimension * sizeof(typename T::ScalarType);
    else if constexpr (std::is_same_v<T, std::string> || std::is_same_v<T, TfToken> || std::is_same_v<T, SdfPath> ||
        std::is_same_v<T, SdfAssetPath>)
        return 4;
    else if constexpr (List<T>::value || std::is_same_v<T, SdfValueBlock>) return 1;
    else if constexpr (std::is_same_v<T, VtValue> || std::is_same_v<T, SdfUnregisteredValue>) return 2;
    else if constexpr (std::is_same_v<T, SdfReference>) return 32;
    else if constexpr (std::is_same_v<T, SdfLayerOffset>) return 16;
    else return 8; // count prefix, time code, or two payload strings.
}
class Reader {
public:
    Reader(std::string_view bytes, Budget& budget) : bytes_(bytes), budget_(budget) {}
    size_t left() const { return bytes_.size() - position_; }
    std::string_view take(size_t size) {
        check(size <= left(), "truncated USD value");
        const auto result = bytes_.substr(position_, size);
        position_ += size;
        return result;
    }
    uint64_t integer(size_t width) {
        const auto bytes = take(width);
        uint64_t value = 0;
        for (size_t i = 0; i < width; ++i) value |= uint64_t(uint8_t(bytes[i])) << (i * 8);
        return value;
    }
    size_t count(size_t minimum, size_t storage) {
        const uint64_t count = integer(8);
        check(count <= left() / minimum, "USD container count exceeds content");
        if (count > budget_.elements) throw DecodeError(ErrorKind::ResourceLimit, "USD element limit exceeded");
        budget_.storage(size_t(count), storage);
        budget_.storage(1, 64); // container allocation/control overhead.
        return size_t(count);
    }
    std::string string() {
        const auto size = integer(4);
        const auto span = take(size_t(size));
        spend(size_t(size), budget_.strings, "USD string-byte limit exceeded");
        budget_.storage(size_t(size) + 1, 1);
        return std::string(span);
    }
    static void assetString(const std::string&);
    void pathStructure(const std::string&);
    VtValue tagged();
    VtValue numeric(uint16_t);
    VtValue compound(uint16_t);
    template<ValueType V> VtValue make(bool array) {
        using T = typename WireType<V>::Type;
        budget_.storage(1, sizeof(VtValue) + sizeof(T) + 32);
        if (!array) {
            auto value = read<T>();
            return VtValue::Take(value);
        }
        if constexpr (hasArrayForm(unsigned(V))) {
            Depth depth(budget_);
            spend(1, budget_.elements, "USD element limit exceeded");
            const size_t n = count(minimumBytes<T>(), sizeof(T));
            VtArray<T> result(n);
            for (size_t i = 0; i < n; ++i) result[i] = read<T>();
            return VtValue::Take(result);
        } else {
            throw DecodeError(ErrorKind::Unsupported, "USD value type has no array form");
        }
    }
    template<class T> T read() {
        Depth depth(budget_);
        spend(1, budget_.elements, "USD element limit exceeded");
        if constexpr (std::is_same_v<T, bool>) {
            const auto value = integer(1);
            check(value <= 1, "invalid USD boolean");
            return value != 0;
        } else if constexpr (std::is_integral_v<T>) {
            static_assert(sizeof(T) <= 8);
            using U = std::make_unsigned_t<T>;
            const U bits = U(integer(sizeof(T)));
            T result;
            std::memcpy(&result, &bits, sizeof(T));
            return result;
        } else if constexpr (std::is_floating_point_v<T>) {
            static_assert(std::numeric_limits<T>::is_iec559);
            using U = std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>;
            const U bits = U(integer(sizeof(T)));
            T result;
            std::memcpy(&result, &bits, sizeof(T));
            return result;
        } else if constexpr (std::is_same_v<T, GfHalf>) {
            GfHalf result;
            result.setBits(uint16_t(integer(2)));
            return result;
        } else if constexpr (std::is_enum_v<T>) {
            static_assert(sizeof(T) == 4);
            const auto value = integer(4);
            constexpr auto end = [] {
                if constexpr (std::is_same_v<T, SdfSpecifier>) return int(SdfNumSpecifiers);
                else if constexpr (std::is_same_v<T, SdfPermission>) return int(SdfNumPermissions);
                else if constexpr (std::is_same_v<T, SdfVariability>) return int(SdfNumVariabilities);
                else {
                    static_assert(std::is_same_v<T, SdfSpecType>);
                    return int(SdfNumSpecTypes);
                }
            }();
            check(value < uint64_t(end), "invalid USD enum");
            return T(value);
        } else if constexpr (GfIsGfVec<T>::value) {
            T result;
            for (size_t i = 0; i < T::dimension; ++i) result[i] = read<typename T::ScalarType>();
            return result;
        } else if constexpr (GfIsGfMatrix<T>::value) {
            T result;
            constexpr size_t n = T::numRows * T::numColumns;
            for (size_t i = 0; i < n; ++i) result.GetArray()[i] = read<typename T::ScalarType>();
            return result;
        } else if constexpr (GfIsGfQuat<T>::value) {
            auto imaginary = read<typename T::ImaginaryType>();
            auto real = read<typename T::ScalarType>();
            return T(real, imaginary);
        } else if constexpr (GfIsGfRange<T>::value) {
            auto minimum = read<typename T::MinMaxType>();
            auto maximum = read<typename T::MinMaxType>();
            return T(minimum, maximum);
        } else if constexpr (std::is_same_v<T, std::string>) {
            return string();
        } else if constexpr (std::is_same_v<T, TfToken>) {
            auto text = string();
            check(text.find('\0') == std::string::npos, "NUL in USD token");
            budget_.storage(text.size() + 33, 1);
            return TfToken(text);
        } else if constexpr (std::is_same_v<T, SdfPath>) {
            auto text = string();
            check(text.find('\0') == std::string::npos, "NUL in USD path");
            pathStructure(text);
            check(text.empty() || SdfPath::IsValidPathString(text), "invalid USD path");
            return text.empty() ? SdfPath() : SdfPath(text);
        } else if constexpr (std::is_same_v<T, SdfAssetPath>) {
            auto text = string();
            assetString(text);
            budget_.storage(text.size() + 1, 1);
            return SdfAssetPath(text);
        } else if constexpr (std::is_same_v<T, SdfLayerOffset>) {
            const auto offset = read<double>();
            const auto scale = read<double>();
            check(std::isfinite(offset) && std::isfinite(scale), "invalid USD layer offset");
            return SdfLayerOffset(offset, scale);
        } else if constexpr (std::is_same_v<T, SdfReference>) {
            const auto before = budget_.decoded;
            auto asset = string();
            assetString(asset);
            auto path = read<SdfPath>();
            auto offset = read<SdfLayerOffset>();
            auto data = read<VtDictionary>();
            budget_.storage(before - budget_.decoded, 1);
            return SdfReference(asset, path, offset, data);
        } else if constexpr (std::is_same_v<T, SdfPayload>) {
            auto asset = string();
            assetString(asset);
            auto path = read<SdfPath>();
            budget_.storage(asset.size() + 1, 1);
            return SdfPayload(asset, path); // v0 has no layer offset on the wire.
        } else if constexpr (Vector<T>::value) {
            using Item = typename Vector<T>::Item;
            const size_t n = count(minimumBytes<Item>(), sizeof(Item));
            T result;
            result.reserve(n);
            for (size_t i = 0; i < n; ++i) result.push_back(read<Item>());
            return result;
        } else if constexpr (std::is_same_v<T, VtDictionary> || std::is_same_v<T, SdfVariantSelectionMap> ||
            std::is_same_v<T, SdfTimeSampleMap>) {
            using Key = typename T::key_type;
            using Value = typename T::mapped_type;
            const size_t n = count(minimumBytes<Key>() + minimumBytes<Value>(), sizeof(Key) + sizeof(Value) + 64);
            T result;
            for (size_t i = 0; i < n; ++i) {
                auto key = read<Key>();
                if constexpr (std::is_same_v<Key, double>) check(!std::isnan(key), "invalid USD sample time");
                auto value = read<Value>();
                if constexpr (std::is_same_v<Key, std::string>) budget_.storage(key.size() + 1, 1);
                if constexpr (std::is_same_v<Value, VtValue>)
                    if (value.IsEmpty()) { // A type outside the table; the entry is omitted.
                        check(!result.count(key), "duplicate USD map key");
                        continue;
                    }
                check(result.insert({std::move(key), std::move(value)}).second, "duplicate USD map key");
            }
            return result;
        } else if constexpr (List<T>::value) {
            using Item = typename List<T>::Item;
            const auto bits = integer(1);
            check(!(bits & 0x80) && (!(bits & 2) || (bits & 1)) && (!(bits & 1) || !(bits & 0x7c)),
                "invalid USD list-op flags");
            T result;
            if (bits & 1) result.ClearAndMakeExplicit();
            const std::pair<unsigned, SdfListOpType> lists[] = {{2, SdfListOpTypeExplicit}, {4, SdfListOpTypeAdded},
                {32, SdfListOpTypePrepended}, {64, SdfListOpTypeAppended}, {8, SdfListOpTypeDeleted},
                {16, SdfListOpTypeOrdered}};
            for (const auto& [bit, type] : lists)
                if (bits & bit) {
                    const auto before = budget_.decoded;
                    auto items = read<std::vector<Item>>();
                // SetItems copies the vector, including nested dictionary keys.
                // Reserve a second full decoded charge before that copy.
                    budget_.storage(before - budget_.decoded, 1);
                    if constexpr (std::is_same_v<Item, SdfUnregisteredValue>) {
                    // This USD type exposes equality but no total ordering.
                    // Bound even quadratic SDK equality/deduplication before
                    // calling it; nested item storage is a work-size proxy.
                        const auto cost = before - budget_.decoded;
                        if (cost && items.size() > budget_.comparison / cost)
                            throw DecodeError(ErrorKind::ResourceLimit, "USD list comparison limit exceeded");
                        spend(items.size() * cost, budget_.comparison, "USD list comparison limit exceeded");
                    } else {
                        budget_.storage(items.size(), sizeof(size_t));
                        std::vector<size_t> sorted(items.size());
                        std::iota(sorted.begin(), sorted.end(), 0);
                        typename Sdf_ListOpTraits<Item>::ItemComparator less;
                        std::sort(sorted.begin(), sorted.end(),
                            [&](size_t a, size_t b) { return less(items[a], items[b]); });
                        check(std::adjacent_find(sorted.begin(), sorted.end(),
                                  [&](size_t a, size_t b) {
                            return !less(items[a], items[b]) && !less(items[b], items[a]);
                        }) == sorted.end(),
                            "duplicate USD list-op item");
                    }
                    check(result.SetItems(items, type), "invalid USD list-op items");
                }
            return result;
        } else if constexpr (std::is_same_v<T, SdfValueBlock>) {
            (void)integer(1);
            return SdfValueBlock(); // Legacy empty-object representation is unspecified.
        } else if constexpr (std::is_same_v<T, VtValue>) {
            return tagged();
        } else if constexpr (std::is_same_v<T, SdfUnregisteredValue>) {
            const auto before = budget_.decoded;
            auto value = tagged();
            budget_.storage(before - budget_.decoded, 1);
            if (value.IsHolding<std::string>()) return SdfUnregisteredValue(value.UncheckedGet<std::string>());
            if (value.IsHolding<VtDictionary>()) return SdfUnregisteredValue(value.UncheckedGet<VtDictionary>());
            if (value.IsHolding<SdfUnregisteredValueListOp>())
                return SdfUnregisteredValue(value.UncheckedGet<SdfUnregisteredValueListOp>());
            invalid("invalid USD unregistered value type");
        } else if constexpr (std::is_same_v<T, SdfTimeCode>) {
            return SdfTimeCode(read<double>());
        } else static_assert(!sizeof(T), "unhandled wire value type");
    }
private:
    std::string_view bytes_;
    size_t position_ = 0;
    Budget& budget_;
};
} // namespace omni::usd::live::wire::detail
