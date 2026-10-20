// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Typed JSON and command parsing for the reference graph profile's presence value families: strings, string and
// token arrays, and floating-point families by the exact IEEE-754 bits of their components, so the Python oracle
// compares representation (-0.0 and NaN payloads stay distinct). Expected values live in the Python oracle.
#pragma once
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3d.h>
#include <pxr/base/gf/vec4f.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/array.h>
#include <pxr/base/vt/value.h>
#include <cstdint>
#include <cstring>
#include <istream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

namespace graph_values {
template<class T> struct Real;
template<> struct Real<float> {
    using Scalar = float; using Bits = uint32_t; static constexpr std::size_t width = 1;
    static const float* data(const float& value) { return &value; }
    static float* data(float& value) { return &value; }
};
template<> struct Real<double> {
    using Scalar = double; using Bits = uint64_t; static constexpr std::size_t width = 1;
    static const double* data(const double& value) { return &value; }
    static double* data(double& value) { return &value; }
};
template<class Gf, class S, std::size_t N> struct GfReal {
    using Scalar = S; using Bits = std::conditional_t<sizeof(S) == 4, uint32_t, uint64_t>;
    static constexpr std::size_t width = N;
    static const S* data(const Gf& value) { return value.data(); }
    static S* data(Gf& value) { return value.data(); }
};
template<> struct Real<PXR_NS::GfVec2f> : GfReal<PXR_NS::GfVec2f, float, 2> {};
template<> struct Real<PXR_NS::GfVec3d> : GfReal<PXR_NS::GfVec3d, double, 3> {};
template<> struct Real<PXR_NS::GfVec4f> : GfReal<PXR_NS::GfVec4f, float, 4> {};
template<> struct Real<PXR_NS::GfMatrix4d> : GfReal<PXR_NS::GfMatrix4d, double, 16> {};

template<class T, class Json> void realBits(Json& out, const T& item, bool& first) {
    const auto* components = Real<T>::data(item);
    for (std::size_t i = 0; i < Real<T>::width; ++i) {
        typename Real<T>::Bits bits;
        std::memcpy(&bits, components + i, sizeof(bits));
        if (!first) out.add(",");
        first = false;
        out.number(bits);
    }
}
// A scalar double is the oracle's finite "double" type, so only its array form is projected here.
template<class T, class Json>
bool real(Json& out, const PXR_NS::VtValue& value, std::string_view family, bool scalar = true) {
    bool first = true;
    if (scalar && value.IsHolding<T>()) {
        out.add("{\"type\":\""); out.add(family); out.add("\",\"value\":[");
        realBits(out, value.UncheckedGet<T>(), first);
        out.add("]}");
        return true;
    }
    if (!value.IsHolding<PXR_NS::VtArray<T>>()) return false;
    out.add("{\"type\":\""); out.add(family); out.add("[]\",\"value\":[");
    for (const auto& item : value.UncheckedGet<PXR_NS::VtArray<T>>()) realBits(out, item, first);
    out.add("]}");
    return true;
}
template<class Json, class Item, class Text>
void textArray(Json& out, std::string_view type, const PXR_NS::VtArray<Item>& items, Text text) {
    out.add("{\"type\":\""); out.add(type); out.add("\",\"value\":[");
    bool first = true;
    for (const auto& item : items) {
        if (!first) out.add(",");
        first = false;
        out.quote(text(item));
    }
    out.add("]}");
}
// Projects one presence family and returns true, or returns false for any other value.
template<class Json> bool project(Json& out, const PXR_NS::VtValue& value) {
    using namespace PXR_NS;
    if (value.IsHolding<std::string>()) {
        out.add("{\"type\":\"string\",\"value\":"); out.quote(value.UncheckedGet<std::string>()); out.add("}");
        return true;
    }
    if (value.IsHolding<VtArray<std::string>>()) {
        textArray(out, "string[]", value.UncheckedGet<VtArray<std::string>>(),
                  [](const std::string& text) -> std::string_view { return text; });
        return true;
    }
    if (value.IsHolding<VtArray<TfToken>>()) {
        textArray(out, "token[]", value.UncheckedGet<VtArray<TfToken>>(),
                  [](const TfToken& token) -> std::string_view { return token.GetString(); });
        return true;
    }
    return real<float>(out, value, "float") || real<double>(out, value, "double", false) ||
        real<GfMatrix4d>(out, value, "matrix4d") || real<GfVec2f>(out, value, "vec2f") ||
        real<GfVec3d>(out, value, "vec3d") || real<GfVec4f>(out, value, "vec4f");
}

// Command operands: a string is "x" followed by the hex of its bytes, an array is its element count followed by
// its elements, and a floating-point component is the decimal integer of its IEEE-754 bits.
inline void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
inline std::string text(std::istream& in) {
    std::string word;
    require(bool(in >> word) && !word.empty() && word[0] == 'x' && word.size() % 2 == 1, "invalid graph text operand");
    std::string result;
    const auto digit = [](char c) {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        throw std::runtime_error("invalid graph text hex");
    };
    for (std::size_t i = 1; i < word.size(); i += 2) result.push_back(char(digit(word[i]) * 16 + digit(word[i + 1])));
    return result;
}
inline std::size_t count(std::istream& in) {
    uint64_t value = 0;
    require(bool(in >> value) && value <= 4096, "invalid graph array count");
    return std::size_t(value);
}
template<class T> T realItem(std::istream& in) {
    T item{};
    auto* components = Real<T>::data(item);
    for (std::size_t i = 0; i < Real<T>::width; ++i) {
        uint64_t bits = 0;
        require(bool(in >> bits) && bits <= std::numeric_limits<typename Real<T>::Bits>::max(),
                "invalid graph real bits");
        const auto narrow = typename Real<T>::Bits(bits);
        std::memcpy(components + i, &narrow, sizeof(narrow));
    }
    return item;
}
template<class T> PXR_NS::VtValue realValue(std::istream& in, bool array) {
    if (!array) return PXR_NS::VtValue(realItem<T>(in));
    PXR_NS::VtArray<T> items(count(in));
    for (auto& item : items) item = realItem<T>(in);
    return PXR_NS::VtValue(items);
}
// Parses "<type> <operands>" for one presence family: string, string[], token[], float, float[], double[],
// matrix4d, vec2f, vec3d, vec4f and the array form of each Gf family.
inline PXR_NS::VtValue parse(std::istream& in) {
    using namespace PXR_NS;
    std::string type;
    require(bool(in >> type), "missing graph value type");
    if (type == "string") return VtValue(text(in));
    if (type == "string[]" || type == "token[]") {
        const auto size = count(in);
        if (type == "string[]") {
            VtArray<std::string> items(size);
            for (auto& item : items) item = text(in);
            return VtValue(items);
        }
        VtArray<TfToken> items(size);
        for (auto& item : items) item = TfToken(text(in));
        return VtValue(items);
    }
    const bool array = type.size() > 2 && type.compare(type.size() - 2, 2, "[]") == 0;
    const auto family = array ? type.substr(0, type.size() - 2) : type;
    if (family == "float") return realValue<float>(in, array);
    if (family == "double" && array) return realValue<double>(in, true);
    if (family == "matrix4d") return realValue<GfMatrix4d>(in, array);
    if (family == "vec2f") return realValue<GfVec2f>(in, array);
    if (family == "vec3d") return realValue<GfVec3d>(in, array);
    if (family == "vec4f") return realValue<GfVec4f>(in, array);
    throw std::runtime_error("unknown graph value type");
}
} // namespace graph_values
