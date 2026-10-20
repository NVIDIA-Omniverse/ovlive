// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "UsdValueCodec.h"
#include "schema/ObjectDeltaBuilder.h"
#include <pxr/base/gf/half.h>
#include <pxr/base/gf/matrix2d.h>
#include <pxr/base/gf/matrix3d.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/quatd.h>
#include <pxr/base/gf/quatf.h>
#include <pxr/base/gf/quath.h>
#include <pxr/base/gf/range1d.h>
#include <pxr/base/gf/range2d.h>
#include <pxr/base/gf/range3d.h>
#include <pxr/base/gf/range1f.h>
#include <pxr/base/gf/range2f.h>
#include <pxr/base/gf/range3f.h>
#include <pxr/base/gf/vec2h.h>
#include <pxr/base/gf/vec3h.h>
#include <pxr/base/gf/vec4h.h>
#include <pxr/base/gf/vec2i.h>
#include <pxr/base/gf/vec3i.h>
#include <pxr/base/gf/vec4i.h>
#include <pxr/base/tf/type.h>
#include <pxr/base/vt/array.h>
#include <pxr/base/vt/dictionary.h>
#include <pxr/usd/sdf/assetPath.h>
#include <pxr/usd/sdf/attributeSpec.h>
#include <pxr/usd/sdf/layer.h>
#include <pxr/usd/sdf/layerOffset.h>
#include <pxr/usd/sdf/listOp.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/sdf/payload.h>
#include <pxr/usd/sdf/primSpec.h>
#include <pxr/usd/sdf/reference.h>
#include <pxr/usd/sdf/timeCode.h>
#include <pxr/usd/sdf/types.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE
using namespace omni::usd::live::wire;
using namespace OmniObjectDelta;
namespace {
std::string context;
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(context + ": " + message);
}
// These builders only spell the documented byte layout. They never call a
// production encoder or decoder, or inspect a Gf object's storage.
std::string le(uint64_t number, size_t width) {
    std::string result;
    for (size_t i = 0; i < width; ++i) result += char(number >> (i * 8));
    return result;
}
std::string text(const std::string& value) { return le(value.size(), 4) + value; }
std::string value(unsigned id, const std::string& body) { return le(id, 2) + body; }
std::string items(const std::vector<std::string>& bodies) {
    std::string result = le(bodies.size(), 8);
    for (const auto& body : bodies) result += body;
    return result;
}
std::string doubles(std::initializer_list<double> numbers) {
    std::string result;
    for (double number : numbers) {
        uint64_t bits; static_assert(sizeof(bits) == sizeof(number));
        std::memcpy(&bits, &number, sizeof(bits)); result += le(bits, 8);
    }
    return result;
}
std::string floats(std::initializer_list<float> numbers) {
    std::string result;
    for (float number : numbers) {
        uint32_t bits; static_assert(sizeof(bits) == sizeof(number));
        std::memcpy(&bits, &number, sizeof(bits)); result += le(bits, 4);
    }
    return result;
}
std::string halves(std::initializer_list<uint16_t> bits) {
    std::string result; for (auto number : bits) result += le(number, 2); return result;
}
std::string ints(std::initializer_list<int> numbers) {
    std::string result; for (auto number : numbers) result += le(uint32_t(number), 4); return result;
}
void bad(std::string_view bytes, const ValueLimits& limits = {}) {
    bool rejected = false;
    try { (void)decodeValue(bytes, limits); }
    catch (const DecodeError&) { rejected = true; }
    check(rejected, "invalid value accepted");
}
void noArrayForm(std::string_view bytes) {
    bool rejected = false;
    try { (void)decodeValue(bytes); }
    catch (const DecodeError& error) {
        check(error.kind == ErrorKind::Unsupported, "wrong array-form rejection kind"); rejected = true;
    }
    check(rejected, "value without a USD array form accepted as an array");
}
// USD stores arrays only of the numeric, string, token, asset and Gf types
// (IDs 1-30) and TimeCode (58) (the crate format's array-capable types,
// matching SdfValueTypeNames' array names). This list is the test's own
// oracle, not the production table.
bool usdArrayForm(unsigned id) { return id <= 30 || id == 58; }
void limited(std::string_view bytes, const ValueLimits& limits) {
    bool rejected = false;
    try { (void)decodeValue(bytes, limits); }
    catch (const DecodeError& error) {
        check(error.kind == ErrorKind::ResourceLimit, "wrong limit error kind"); rejected = true;
    }
    check(rejected, "value budget was not enforced");
}
void encodeBad(const VtValue& input, ErrorKind kind, const EncodeLimits& limits = {}) {
    const VtValue before = input;
    bool rejected = false;
    try { (void)encodeValue(input,limits); }
    catch (const DecodeError& error) {
        check(error.kind == kind, "wrong encoder rejection kind"); rejected = true;
    }
    check(rejected, "encoder accepted unsupported or over-budget value");
    check(input == before, "failed encoding changed input");
}
void encodeExact(const VtValue& input, const std::string& expected) {
    const VtValue before = input;
    check(encodeValue(input) == expected, "encoder differs from independently authored bytes");
    EncodeLimits exact; exact.maxEncodedBytes = expected.size();
    check(encodeValue(input,exact) == expected, "exact encoded-byte budget rejected or changed value");
    check(input == before, "successful encoding changed input");
    if (!expected.empty()) {
        --exact.maxEncodedBytes;
        encodeBad(input,ErrorKind::ResourceLimit,exact);
    }
}
std::vector<std::string> valid;
std::set<unsigned> covered;
template<class T>
void fixture(unsigned id, const std::string& body, const T& expected) {
    context = "type " + std::to_string(id);
    check(covered.insert(id).second, "duplicate primary type fixture");
    auto bytes = value(id, body);
    check(decodeValue(bytes) == VtValue(expected), "scalar fixture differs from independent USD value");
    const auto canonicalBody = id == 51 ? le(0,1) : body;
    // VtValue cannot retain a scalar VtValue wrapper; it holds the inner type.
    encodeExact(VtValue(expected),id == 52 ? body : value(id,canonicalBody));
    valid.push_back(bytes);
    VtArray<T> expectedArray(2); expectedArray[0] = expected; expectedArray[1] = expected;
    if (usdArrayForm(id)) {
        // Empty and two-element arrays also exercise count framing.
        auto empty = value(id | 0x80, items({}));
        check(decodeValue(empty) == VtValue(VtArray<T>{}), "empty array has wrong type");
        auto array = value(id | 0x80, items({body, body}));
        check(decodeValue(array) == VtValue(expectedArray), "array fixture differs from independent USD value");
        encodeExact(VtValue(VtArray<T>{}),empty);
        encodeExact(VtValue(expectedArray),value(id | 0x80,items({canonicalBody,canonicalBody})));
        valid.push_back(empty); valid.push_back(array);
    } else {
        // USD cannot store this type as an array: the flag is an unsupported
        // value in both directions, never a VtArray no USD API can use.
        // A tag with no body proves the rejection precedes any body read:
        // reading the missing count would be Malformed, not Unsupported.
        noArrayForm(value(id | 0x80, ""));
        noArrayForm(value(id | 0x80, items({})));
        noArrayForm(value(id | 0x80, items({body, body})));
        encodeBad(VtValue(VtArray<T>{}),ErrorKind::Unsupported);
        encodeBad(VtValue(expectedArray),ErrorKind::Unsupported);
    }
    // The decoder must not assume the incoming tag/body is naturally aligned.
    for (size_t padding = 1; padding <= 7; ++padding) {
        auto unaligned = std::string(padding, 'x') + bytes;
        check(decodeValue(std::string_view(unaligned).substr(padding)) == VtValue(expected), "unaligned value differs");
    }
}
template<class T>
SdfListOp<T> explicitList(std::vector<T> entries) {
    SdfListOp<T> result;
    check(result.SetExplicitItems(entries), "invalid test list fixture");
    return result;
}
std::string explicitBody(const std::vector<std::string>& entries) {
    return le(3, 1) + items(entries);
}
void numericAndGf() {
    fixture(1, le(1, 1), true);
    fixture(2, le(231, 1), uint8_t(231));
    fixture(3, le(0x87654321, 4), int(-2023406815));
    fixture(4, le(0xfedcba98, 4), 0xfedcba98u);
    fixture(5, le(UINT64_C(0x8123456789abcdef), 8), INT64_C(-9141386507638288913));
    fixture(6, le(UINT64_C(0xfedcba9876543210), 8), UINT64_C(0xfedcba9876543210));
    fixture(7, le(0xbe00, 2), GfHalf(-1.5f));
    fixture(8, le(0xc0200000, 4), -2.5f);
    fixture(9, le(UINT64_C(0x400a000000000000), 8), 3.25);
    fixture(13, doubles({1,2,3,4}), GfMatrix2d(1,2,3,4));
    fixture(14, doubles({1,2,3,4,5,6,7,8,9}), GfMatrix3d(1,2,3,4,5,6,7,8,9));
    fixture(15, doubles({1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}),
        GfMatrix4d(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16));
    // Quaternion wire order is imaginary xyz, then real. USD's constructor
    // takes real first, so a native-memory reinterpretation is not our oracle.
    fixture(16, doubles({1,2,3,4}), GfQuatd(4, GfVec3d(1,2,3)));
    fixture(17, floats({1,2,3,4}), GfQuatf(4, GfVec3f(1,2,3)));
    fixture(18, halves({0x3c00,0x4000,0x4200,0x4400}), GfQuath(GfHalf(4), GfVec3h(1,2,3)));
    fixture(19, doubles({-1,2}), GfVec2d(-1,2));
    fixture(20, floats({-1,2}), GfVec2f(-1,2));
    fixture(21, halves({0xbc00,0x4000}), GfVec2h(-1,2));
    fixture(22, ints({-1,2}), GfVec2i(-1,2));
    fixture(23, doubles({-1,2,3}), GfVec3d(-1,2,3));
    fixture(24, floats({-1,2,3}), GfVec3f(-1,2,3));
    fixture(25, halves({0xbc00,0x4000,0x4200}), GfVec3h(-1,2,3));
    fixture(26, ints({-1,2,3}), GfVec3i(-1,2,3));
    fixture(27, doubles({-1,2,3,4}), GfVec4d(-1,2,3,4));
    fixture(28, floats({-1,2,3,4}), GfVec4f(-1,2,3,4));
    fixture(29, halves({0xbc00,0x4000,0x4200,0x4400}), GfVec4h(-1,2,3,4));
    fixture(30, ints({-1,2,3,4}), GfVec4i(-1,2,3,4));
    fixture(59, doubles({-3,7}), GfRange1d(-3,7));
    fixture(60, doubles({-3,-2,7,8}), GfRange2d(GfVec2d(-3,-2), GfVec2d(7,8)));
    fixture(61, doubles({-3,-2,-1,7,8,9}), GfRange3d(GfVec3d(-3,-2,-1), GfVec3d(7,8,9)));
    fixture(62, floats({-3,7}), GfRange1f(-3,7));
    fixture(63, floats({-3,-2,7,8}), GfRange2f(GfVec2f(-3,-2), GfVec2f(7,8)));
    fixture(64, floats({-3,-2,-1,7,8,9}), GfRange3f(GfVec3f(-3,-2,-1), GfVec3f(7,8,9)));
    // Distinct array elements catch an implementation repeating the first body.
    auto array = value(0x83, items({ints({17}), ints({-31}), ints({99})}));
    check(decodeValue(array) == VtValue(VtArray<int>{17,-31,99}), "array order/content lost");
    encodeExact(VtValue(VtArray<int>{17,-31,99}),array);
    valid.push_back(array);
    const auto strings = value(0x8a,items({text("one"),text("longer two"),text("")}));
    check(decodeValue(strings) == VtValue(VtArray<std::string>{"one","longer two",""}), "variable-sized array order/content lost");
    encodeExact(VtValue(VtArray<std::string>{"one","longer two",""}),strings);
    valid.push_back(strings);
    // Keep extraction on const lvalues: this SDK's rvalue Get<T>() moves via
    // an uninitialized primitive temporary, which itself trips UBSan for bool.
    const auto falseValue = decodeValue(value(1, le(0,1)));
    const auto infiniteValue = decodeValue(value(8, le(0x7f800000,4)));
    const auto nanValue = decodeValue(value(9, le(UINT64_C(0x7ff8000000000000),8)));
    const auto negativeZeroValue = decodeValue(value(9, le(UINT64_C(0x8000000000000000),8)));
    check(!falseValue.Get<bool>(), "false decoded as true");
    check(std::isinf(infiniteValue.Get<float>()), "infinite float rejected");
    check(std::isnan(nanValue.Get<double>()), "NaN numeric rejected");
    check(std::signbit(negativeZeroValue.Get<double>()), "negative zero lost");
}
void containers() {
    const std::string withNul("a\0b", 3);
    fixture(10, text(withNul), withNul);
    fixture(11, text("authoredName"), TfToken("authoredName"));
    fixture(12, text("./asset.usda"), SdfAssetPath("./asset.usda"));
    const std::string dictBody = le(2,8) + text("count") + value(3,ints({-7})) + text("label") + value(10,text(withNul));
    VtDictionary dict; dict["count"] = -7; dict["label"] = withNul;
    fixture(31, dictBody, dict);
    fixture(32, explicitBody({text("red"),text("blue")}), explicitList<TfToken>({TfToken("red"),TfToken("blue")}));
    fixture(33, explicitBody({text("first"),text(withNul)}), explicitList<std::string>({"first",withNul}));
    fixture(34, explicitBody({text("/A"),text("/B.attr")}), explicitList<SdfPath>({SdfPath("/A"),SdfPath("/B.attr")}));
    const auto refBody = text("model.usda") + text("/Root") + doubles({-3.5,2.0}) + dictBody;
    const auto ref = SdfReference("model.usda", SdfPath("/Root"), SdfLayerOffset(-3.5,2.0), dict);
    fixture(35, explicitBody({refBody}), explicitList<SdfReference>({ref}));
    fixture(36, explicitBody({ints({-7}),ints({23})}), explicitList<int>({-7,23}));
    fixture(37, explicitBody({le(uint64_t(-INT64_C(4294967297)),8),le(17,8)}), explicitList<int64_t>({-INT64_C(4294967297),17}));
    fixture(38, explicitBody({le(0xffffffff,4),le(23,4)}), explicitList<unsigned int>({0xffffffffu,23}));
    fixture(39, explicitBody({le(UINT64_MAX,8),le(23,8)}), explicitList<uint64_t>({UINT64_MAX,23}));
    fixture(40, items({text("/World"),text("relative")}), SdfPathVector{SdfPath("/World"),SdfPath("relative")});
    fixture(41, items({text("a"),text("b")}), std::vector<TfToken>{TfToken("a"),TfToken("b")});
    fixture(42, le(2,4), SdfSpecifierClass);
    fixture(43, le(1,4), SdfPermissionPrivate);
    fixture(44, le(1,4), SdfVariabilityUniform);
    fixture(45, le(2,8)+text("look")+text("red")+text("shape")+text("round"),
        SdfVariantSelectionMap{{"look","red"},{"shape","round"}});
    const double infinity = std::numeric_limits<double>::infinity();
    fixture(46, le(4,8)+doubles({-infinity})+value(3,ints({-99}))+
        doubles({-2.5})+value(3,ints({17}))+doubles({4.25})+value(10,text(withNul))+
        doubles({infinity})+value(3,ints({99})),
        SdfTimeSampleMap{{-infinity,VtValue(-99)},{-2.5,VtValue(17)},
            {4.25,VtValue(withNul)},{infinity,VtValue(99)}});
    const auto payloadBody = text("payload.usda") + text("/Payload");
    const auto payload = SdfPayload("payload.usda", SdfPath("/Payload"));
    fixture(47, payloadBody, payload); // v0 has no layer-offset extension.
    fixture(48, items({doubles({-3}),doubles({7.5})}), std::vector<double>{-3,7.5});
    fixture(49, items({doubles({-3,2}),doubles({7.5,0.5})}),
        std::vector<SdfLayerOffset>{SdfLayerOffset(-3,2),SdfLayerOffset(7.5,0.5)});
    fixture(50, items({text("one"),text(withNul)}), std::vector<std::string>{"one",withNul});
    fixture(51, le(0xff,1), SdfValueBlock()); // The reserved byte is ignored, not a boolean.
    fixture<VtValue>(52, value(3,ints({42})), VtValue(42));
    fixture(53, value(10,text("unknownType(1)")), SdfUnregisteredValue("unknownType(1)"));
    const auto urBody = value(10,text("unknownType(2)"));
    fixture(54, explicitBody({urBody}), explicitList<SdfUnregisteredValue>({SdfUnregisteredValue("unknownType(2)")}));
    fixture(55, text("/World.attr"), SdfPath("/World.attr"));
    fixture(56, le(6,4), SdfSpecTypePrim);
    fixture(57, explicitBody({payloadBody}), explicitList<SdfPayload>({payload}));
    fixture(58, doubles({-9.5}), SdfTimeCode(-9.5));
    // The pinned USD registers timecode[]; its array form stays on the wire.
    check(SdfValueTypeNames->TimeCodeArray.GetType() == TfType::Find<VtArray<SdfTimeCode>>(), "USD timecode[] type differs");
    check(decodeValue(value(53,value(31,dictBody))) == VtValue(SdfUnregisteredValue(dict)), "unregistered dictionary differs");
    encodeExact(VtValue(SdfUnregisteredValue(dict)),value(53,value(31,dictBody)));
    const auto urList = explicitList<SdfUnregisteredValue>({SdfUnregisteredValue("unknownType(2)")});
    check(decodeValue(value(53,value(54,explicitBody({urBody})))) == VtValue(SdfUnregisteredValue(urList)),
        "unregistered list differs");
    encodeExact(VtValue(SdfUnregisteredValue(urList)),value(53,value(54,explicitBody({urBody}))));
    check(decodeValue(value(32,le(1,1))) == VtValue(explicitList<TfToken>({})), "empty explicit list lost");
    check(decodeValue(value(32,le(0,1))) == VtValue(SdfTokenListOp()), "empty nonexplicit list lost");
    encodeExact(VtValue(explicitList<TfToken>({})),value(32,le(1,1)));
    encodeExact(VtValue(SdfTokenListOp()),value(32,le(0,1)));
    // All six list vectors have different members; the byte order differs from
    // the numeric flag order for prepended/appended versus deleted/ordered.
    SdfStringListOp op;
    check(op.SetItems({"add"}, SdfListOpTypeAdded), "fixture add rejected");
    check(op.SetPrependedItems({"prepend"}), "fixture prepend rejected");
    check(op.SetAppendedItems({"append"}), "fixture append rejected");
    check(op.SetDeletedItems({"delete"}), "fixture delete rejected");
    check(op.SetItems({"order"}, SdfListOpTypeOrdered), "fixture order rejected");
    auto list = value(33, le(0x7c,1)+items({text("add")})+items({text("prepend")})+
        items({text("append")})+items({text("delete")})+items({text("order")}));
    check(decodeValue(list) == VtValue(op), "list operation vectors reordered or omitted");
    encodeExact(VtValue(op),list);
    valid.push_back(list);
    bad(value(47,payloadBody+doubles({0,1}))); // Must not silently accept a newer payload format.
}
void malformedAndBounds() {
    context = "malformed values";
    check(decodeValue({}).IsEmpty(), "empty top-level value must mean deletion");
    check(covered.size() == 64 && *covered.begin() == 1 && *covered.rbegin() == 64, "missing type coverage");
    for (const auto& bytes : valid) {
        for (size_t length = 1; length < bytes.size(); ++length) {
            context = "truncated tag " + std::to_string(uint8_t(bytes[0])) + " at " + std::to_string(length);
            bad(std::string_view(bytes).substr(0,length));
        }
        context = "trailing/version tag " + std::to_string(uint8_t(bytes[0]));
        bad(bytes + "x");
        auto version = bytes; version[1] = 1; bad(version);
    }
    context = "malformed values";
    for (unsigned id : {65u,127u,193u,255u}) bad(value(id,{}));
    // Kit writes a type outside the table, scalar or array, as type 0 with no body and reads it as an empty
    // value: an erase at the top level, an omitted entry in a dictionary.
    for (unsigned id : {0u,128u}) { check(decodeValue(value(id,{})).IsEmpty(),"type 0 is not empty"); bad(value(id,"x")); }
    check(decodeValue(value(52,value(0,{}))).IsEmpty(),"nested type 0 is not empty");
    { VtDictionary kept; kept["count"] = -7;
      check(decodeValue(value(31,le(2,8)+text("count")+value(3,ints({-7}))+text("gone")+value(0,{}))) == VtValue(kept),
            "dictionary kept a type 0 entry"); }
    bad(value(1,le(2,1))); bad(value(0x81,items({le(0,1),le(255,1)})));
    for (unsigned id : {42u,43u,44u,56u}) { bad(value(id,le(UINT32_MAX,4))); bad(value(id,le(255,4))); }
    bad(value(42,le(3,4))); bad(value(43,le(2,4))); bad(value(44,le(2,4))); bad(value(56,le(12,4)));
    bad(value(10,le(UINT32_MAX,4))); bad(value(0x83,le(UINT64_MAX,8)));
    for (unsigned id : {31u,40u,41u,45u,46u,48u,49u,50u}) bad(value(id,le(UINT64_MAX,8)));
    bad(value(32,le(3,1)+le(UINT64_MAX,8)));
    bad(value(52,{})); bad(value(53,value(3,ints({1}))));
    bad(value(11,text(std::string("a\0b",3)))); bad(value(12,text(std::string("a\0b",3))));
    bad(value(55,text(std::string("/A\0B",5))));
    bad(value(32,le(0x80,1))); bad(value(32,le(5,1)+items({text("a")})));
    bad(value(32,explicitBody({text("same"),text("same")})));
    bad(value(54,explicitBody({value(10,text("same")),value(10,text("same"))})));
    bad(value(31,le(2,8)+text("same")+value(1,le(0,1))+text("same")+value(1,le(1,1))));
    bad(value(45,le(2,8)+text("same")+text("one")+text("same")+text("two")));
    bad(value(46,le(2,8)+doubles({1})+value(1,le(0,1))+doubles({1})+value(1,le(1,1))));
    bad(value(46,le(1,8)+doubles({std::numeric_limits<double>::quiet_NaN()})+value(1,le(0,1))));
    const auto nested = value(52,value(52,value(52,value(3,ints({19})))));
    check(decodeValue(nested) == VtValue(19), "nested VtValue changed type/value");
    auto limits = ValueLimits{}; limits.maxDepth = 2; limited(nested,limits);
    const auto integerArray = value(0x83,items({ints({11}),ints({22})}));
    limits = {}; limits.maxElements = 1; limited(integerArray,limits);
    limits = {}; limits.maxStringBytes = 2; limited(value(10,text("abc")),limits);
    limits = {}; limits.maxEncodedBytes = integerArray.size()-1; limited(integerArray,limits);
    limits = {}; limits.maxDecodedBytes = 1; limited(integerArray,limits);
    limits = {}; limits.maxDecodedBytes = 0; limited(value(3,ints({1})),limits);
    limits = {}; limits.maxComparisonWork = 0;
    limited(value(54,explicitBody({value(10,text("opaque"))})),limits);
    for (const auto& invalidAsset : {std::string("line\nbreak"), std::string("\xc0\xaf",2),
        std::string("\xed\xa0\x80",3), std::string("\xf4\x90\x80\x80",4),std::string("\xe2\x82",2)}) {
        bad(value(12,text(invalidAsset)));
    }
    bad(value(55,text("/Not A Path")));
    bad(value(49,items({doubles({0,std::numeric_limits<double>::infinity()})})));
}
void pathBudgets() {
    context = "path preflight budgets";
    std::string path = "/Leaf";
    for (unsigned i = 0; i < 12; ++i) path = "/Root.rel[" + path + "]";
    check(decodeValue(value(55,text(path))) == VtValue(SdfPath(path)), "nested target path fixture invalid");
    auto limits = ValueLimits{}; limits.maxDepth = 3;
    limited(value(55,text(path)),limits);
    path.clear();
    for (unsigned i = 0; i < 48; ++i) path += "/Component" + std::to_string(i);
    check(decodeValue(value(55,text(path))) == VtValue(SdfPath(path)), "component path fixture invalid");
    limits = {}; limits.maxElements = 8; limited(value(55,text(path)),limits);
    limits = {}; limits.maxDecodedBytes = 256; limited(value(55,text(path)),limits);
}
void copyBudgets() {
    context = "USD constructor copy budgets";
    // Large authored payloads dominate object/control overhead. These bounds
    // leave room for decoding the source value, but not all conservative copy
    // charges required when constructing its enclosing USD value. They do not
    // discover implementation thresholds or assume an exact object size/RSS.
    constexpr size_t payloadSize = 64 * 1024;
    const std::string payload(payloadSize,'k');
    const auto stringBytes = value(10,text(payload));
    auto limits = ValueLimits{}; limits.maxDecodedBytes = payloadSize * 3 / 2;
    check(decodeValue(stringBytes,limits) == VtValue(payload), "source string does not fit copy budget fixture");
    const auto unregisteredString = value(53,stringBytes);
    check(decodeValue(unregisteredString) == VtValue(SdfUnregisteredValue(payload)), "large unregistered string differs");
    limited(unregisteredString,limits);

    // Put the large string in a dictionary KEY: VtValue may share a string
    // value, whereas copying the dictionary must copy its authored keys.
    const auto dictBody = le(1,8)+text(payload)+value(3,ints({17}));
    VtDictionary dict; dict[payload] = 17;
    limits.maxDecodedBytes = payloadSize * 3;
    check(decodeValue(value(31,dictBody),limits) == VtValue(dict), "source dictionary does not fit copy budget fixture");
    const auto unregisteredDict = value(53,value(31,dictBody));
    check(decodeValue(unregisteredDict) == VtValue(SdfUnregisteredValue(dict)), "large unregistered dictionary differs");
    limited(unregisteredDict,limits);

    const auto refBody = text("model.usda")+text("/Root")+doubles({0,1})+dictBody;
    const auto referenceList = value(35,explicitBody({refBody}));
    const auto expectedReference = explicitList<SdfReference>({SdfReference("model.usda",SdfPath("/Root"),SdfLayerOffset(),dict)});
    check(decodeValue(referenceList) == VtValue(expectedReference), "large reference custom data differs");
    // Both the reference constructor and the list setter copy custom data.
    // Omitting the former charge leaves this whole fixture below 6 payloads.
    limits.maxDecodedBytes = payloadSize * 6;
    limited(referenceList,limits);

    const auto listBytes = value(54,explicitBody({stringBytes}));
    const auto expectedList = explicitList<SdfUnregisteredValue>({SdfUnregisteredValue(payload)});
    check(decodeValue(listBytes,limits) == VtValue(expectedList), "source unregistered list does not fit copy budget fixture");
    const auto unregisteredList = value(53,listBytes);
    check(decodeValue(unregisteredList) == VtValue(SdfUnregisteredValue(expectedList)), "large unregistered list differs");
    limited(unregisteredList,limits);
}
Envelope envelope(const std::vector<std::string>& fields, const std::vector<std::string>& samples) {
    OmniObjectDeltaBuilder b;
    for (size_t i = 0; i < fields.size(); ++i) {
        const auto& bytes = fields[i];
        const auto serialized = bytes.empty() ? flatbuffers::Offset<flatbuffers::Vector<uint8_t>>{} :
            b.fbb.CreateVector(reinterpret_cast<const uint8_t*>(bytes.data()),bytes.size());
        b.AddSetField(CreateSetField(b.fbb,1,b.fbb.CreateString("field"+std::to_string(i)),serialized));
    }
    for (size_t i = 0; i < samples.size(); ++i) {
        const auto& bytes = samples[i];
        const auto serialized = bytes.empty() ? flatbuffers::Offset<flatbuffers::Vector<uint8_t>>{} :
            b.fbb.CreateVector(reinterpret_cast<const uint8_t*>(bytes.data()),bytes.size());
        b.AddTimeSample(CreateTimeSample(b.fbb,1,double(i),serialized));
    }
    b.Finish(false,0);
    auto bytes = le(0x46504d4f,4)+le(1,4)+le(16,8);
    bytes.append(reinterpret_cast<const char*>(b.fbb.GetBufferPointer()),b.fbb.GetSize());
    return decodeEnvelope(bytes);
}
void batch() {
    context = "shared batch budgets";
    const auto a = value(10,text("abc")), b = value(3,ints({-31}));
    auto source = envelope({a,{},b},{b,{},a});
    ValueUsage usage;
    auto decoded = decodeValues(source,{},&usage);
    check(usage.encodedBytes == 2*(a.size()+b.size()) && usage.elements == 6 &&
          usage.decodedBytes >= 6*sizeof(VtValue) && usage.comparisonWork == 0,
          "batch usage lost repeated references, deletion slots or storage");
    auto exactUsage = ValueLimits{};
    exactUsage.maxEncodedBytes = usage.encodedBytes; exactUsage.maxDecodedBytes = usage.decodedBytes;
    exactUsage.maxElements = usage.elements; exactUsage.maxComparisonWork = usage.comparisonWork;
    ValueUsage repeatedUsage;
    (void)decodeValues(source,exactUsage,&repeatedUsage);
    check(repeatedUsage.encodedBytes == usage.encodedBytes && repeatedUsage.decodedBytes == usage.decodedBytes &&
          repeatedUsage.elements == usage.elements && repeatedUsage.comparisonWork == usage.comparisonWork,
          "usage depends on configured allowance");
    ValueUsage emptyUsage{1,1,1,1};
    (void)decodeValues({}, {}, &emptyUsage);
    check(emptyUsage.encodedBytes == 0 && emptyUsage.decodedBytes == 0 && emptyUsage.elements == 0 &&
          emptyUsage.comparisonWork == 0,"empty batch reports work");
    check(decoded.fields == std::vector<VtValue>{VtValue(std::string("abc")),VtValue(),VtValue(-31)}, "batch field order/deletion lost");
    check(decoded.samples == std::vector<VtValue>{VtValue(-31),VtValue(),VtValue(std::string("abc"))}, "batch sample order/deletion lost");
    // All three references point at the SAME immutable ValueBytes. Charging
    // only distinct buffers, or resetting budgets per field/sample, must fail.
    auto shared = envelope({a},{});
    shared.fields.push_back(shared.fields.front());
    Sample sample; sample.value = shared.fields.front().value; shared.samples.push_back(sample);
    auto limitedBatch = [&](const Envelope& input, const ValueLimits& limits) {
        bool rejected = false;
        try { (void)decodeValues(input,limits); }
        catch (const DecodeError& error) { check(error.kind == ErrorKind::ResourceLimit, "wrong batch budget error"); rejected = true; }
        check(rejected,"shared batch budget not enforced");
    };
    auto limits = ValueLimits{}; limits.maxEncodedBytes = a.size()*2;
    check(decodeValue(a,limits) == VtValue(std::string("abc")), "individual encoded budget fixture invalid");
    limitedBatch(shared,limits);
    limits = {}; limits.maxStringBytes = 6;
    check(decodeValue(a,limits) == VtValue(std::string("abc")), "individual string budget fixture invalid");
    limitedBatch(shared,limits);
    const auto array = value(0x83,items({ints({1}),ints({2}),ints({3}),ints({4})}));
    auto arrays = envelope({array},{array});
    limits = {}; limits.maxElements = 5;
    check(decodeValue(array,limits) == VtValue(VtArray<int>{1,2,3,4}), "individual element budget fixture invalid");
    limitedBatch(arrays,limits);
    auto invalid = envelope({a,b},{value(1,le(2,1))});
    bool rejected = false;
    try { (void)decodeValues(invalid); } catch (const DecodeError&) { rejected = true; }
    check(rejected,"batch accepted malformed final sample");
    check(invalid.fields[0].value.bytes() == a && invalid.fields[1].value.bytes() == b,
        "failed batch changed input values");
    check(decodeValue(invalid.fields[0].value.bytes()) == VtValue(std::string("abc")), "failure poisoned later decode");
    source = {}; shared = {}; arrays = {}; invalid = {};
    check(decoded.fields[0].Get<std::string>() == "abc" && decoded.samples[0].Get<int>() == -31,
        "decoded USD values depend on envelope lifetime");
}
template<class Float, class Bits>
void encodeFloatingBits(unsigned id, std::initializer_list<Bits> patterns) {
    VtArray<Float> array;
    std::vector<std::string> bodies;
    for (Bits bits : patterns) {
        Float number; static_assert(sizeof(number) == sizeof(bits));
        std::memcpy(&number,&bits,sizeof(number));
        const VtValue input(number);
        check(encodeValue(input) == value(id,le(bits,sizeof(bits))), "floating scalar bits changed");
        Bits after; const auto& held = input.Get<Float>();
        std::memcpy(&after,&held,sizeof(after));
        check(after == bits,"floating encoding changed input bits");
        array.push_back(number); bodies.push_back(le(bits,sizeof(bits)));
    }
    const VtValue input(array);
    check(encodeValue(input) == value(id | 0x80,items(bodies)), "floating array bits changed");
    const auto& held = input.Get<VtArray<Float>>();
    size_t i = 0;
    for (Bits bits : patterns) {
        Bits after; std::memcpy(&after,&held[i++],sizeof(after));
        check(after == bits,"floating array encoding changed input bits");
    }
}
void encoderFailuresAndBits() {
    context = "encoder unsupported values and float bits";
    encodeExact(VtValue(),{});
    encodeExact(VtValue(false),value(1,le(0,1)));
    encodeFloatingBits<float,uint32_t>(8,{0x7f800000,0xff800000,0x80000000,0x7fc12345,0xffc54321});
    encodeFloatingBits<double,uint64_t>(9,{UINT64_C(0x7ff0000000000000),UINT64_C(0xfff0000000000000),
        UINT64_C(0x8000000000000000),UINT64_C(0x7ff8123456789abc),UINT64_C(0xfff8abcdef123456)});
    VtArray<GfHalf> halvesArray;
    std::vector<std::string> halfBodies;
    for (uint16_t bits : {0x7c00,0xfc00,0x8000,0x7e15,0xfe2a}) {
        GfHalf number; number.setBits(bits);
        const VtValue input(number);
        check(encodeValue(input) == value(7,le(bits,2)),"half scalar bits changed");
        check(input.Get<GfHalf>().bits() == bits,"half encoding changed input bits");
        halvesArray.push_back(number); halfBodies.push_back(le(bits,2));
    }
    check(encodeValue(VtValue(halvesArray)) == value(0x87,items(halfBodies)),"half array bits changed");
    for (const auto& unsupported : {VtValue(short(7)),VtValue(std::vector<int>{1,2,3}),
        VtValue(VtArray<short>{4,5})}) encodeBad(unsupported,ErrorKind::Unsupported);
    VtDictionary emptyNested; emptyNested["empty"] = VtValue();
    encodeBad(VtValue(emptyNested),ErrorKind::Unsupported);
    encodeBad(VtValue(VtArray<VtValue>{VtValue(1),VtValue()}),ErrorKind::Unsupported);
    encodeBad(VtValue(SdfUnregisteredValue(emptyNested)),ErrorKind::Unsupported);
    encodeBad(VtValue(SdfUnregisteredValue()),ErrorKind::Unsupported);
    VtDictionary unsupportedNested; unsupportedNested["unsupported"] = VtValue(std::vector<int>{1,2,3});
    encodeBad(VtValue(unsupportedNested),ErrorKind::Unsupported);
    encodeBad(VtValue(VtArray<VtValue>{VtValue(std::vector<int>{1,2,3})}),ErrorKind::Unsupported);
    // v0 encodes only the two payload strings. Every route must reject an
    // offset instead of silently discarding authored offset or scale data.
    for (const auto& offset : {SdfLayerOffset(3,1),SdfLayerOffset(0,2),
        SdfLayerOffset(1e-12,1),SdfLayerOffset(0,1+1e-12)}) {
        const SdfPayload payload("model.usda",SdfPath("/Root"),offset);
        VtDictionary dict; dict["payload"] = payload;
        const auto list = explicitList<SdfPayload>({payload});
        const auto wrapped = SdfUnregisteredValue(dict);
        const auto wrappedList = explicitList<SdfUnregisteredValue>({wrapped});
        for (const auto& input : {VtValue(payload),VtValue(VtArray<SdfPayload>{payload}),
            VtValue(list),VtValue(dict),VtValue(VtArray<VtValue>{VtValue(payload)}),
            VtValue(wrapped),VtValue(SdfUnregisteredValue(wrappedList))}) {
            encodeBad(input,ErrorKind::Unsupported);
        }
        for (const auto operation : {SdfListOpTypeAdded,SdfListOpTypePrepended,SdfListOpTypeAppended,
            SdfListOpTypeDeleted,SdfListOpTypeOrdered}) {
            SdfPayloadListOp opinion;
            check(opinion.SetItems({payload},operation),"invalid payload list rejection fixture");
            encodeBad(VtValue(opinion),ErrorKind::Unsupported);
        }
        // SdfLayerOffset's equality/IsIdentity use tolerance; these checks must
        // detect even offsets too small for those SDK comparisons to notice.
        check(payload.GetLayerOffset().GetOffset() == offset.GetOffset() &&
            payload.GetLayerOffset().GetScale() == offset.GetScale(),"failed payload encoding changed exact offset");
    }
    auto limits = EncodeLimits{}; limits.maxStringBytes = 2;
    encodeBad(VtValue(std::string("abc")),ErrorKind::ResourceLimit,limits);
    limits = {}; limits.maxElements = 1;
    encodeBad(VtValue(VtArray<int>{1,2}),ErrorKind::ResourceLimit,limits);
    limits.maxElements = 0;
    encodeBad(VtValue(),ErrorKind::ResourceLimit,limits);
    VtValue nested(17);
    for (unsigned i = 0; i < 8; ++i) { VtDictionary layer; layer["child"] = nested; nested = VtValue(layer); }
    check(!encodeValue(nested).empty(),"nested encoder fixture invalid");
    limits = {}; limits.maxDepth = 2;
    encodeBad(nested,ErrorKind::ResourceLimit,limits);
}
void encoderAuthoredAssetsAndPaths() {
    context = "encoder authored assets and path topology";
    const std::string authored = "`\"asset_${MODEL}.usda\"`";
    const std::string resolved = "/resolved/"+std::string(256,'r')+".usda";
    const SdfAssetPath asset(SdfAssetPathParams().Authored(authored).Evaluated("evaluated.usda").Resolved(resolved));
    const SdfAssetPath other(SdfAssetPathParams().Authored("other.usda").Evaluated("other-evaluated.usda").Resolved("/other.usda"));
    check(asset.GetAuthoredPath() == authored && asset.GetAssetPath() == "evaluated.usda",
        "cached asset fixture did not distinguish authored/evaluated paths");
    // Stock USD serialization independently establishes which part of an
    // asset with evaluated/resolved caches is authored scene data.
    const auto layer = SdfLayer::CreateAnonymous("asset-authored-oracle.usda");
    check(bool(layer),"stock USD layer creation failed");
    const auto prim = SdfPrimSpec::New(layer,"Root",SdfSpecifierDef);
    check(bool(prim),"stock USD prim creation failed");
    const auto attribute = SdfAttributeSpec::New(prim,"asset",SdfValueTypeNames->Asset);
    check(bool(attribute),"stock USD asset attribute creation failed");
    attribute->SetDefaultValue(VtValue(asset));
    std::string exported;
    check(layer->ExportToString(&exported),"stock USD asset export failed");
    check(exported.find(authored) != std::string::npos &&
        exported.find("evaluated.usda") == std::string::npos && exported.find(resolved) == std::string::npos,
        "stock USD export did not preserve authored asset independently of caches");
    const auto assetBytes = value(12,text(authored));
    encodeExact(VtValue(asset),assetBytes);
    encodeExact(VtValue(VtArray<SdfAssetPath>{asset,other}),value(0x8c,items({text(authored),text("other.usda")})));
    VtDictionary dict; dict["asset"] = asset;
    const auto dictBytes = value(31,le(1,8)+text("asset")+assetBytes);
    encodeExact(VtValue(dict),dictBytes);
    encodeExact(VtValue(SdfUnregisteredValue(dict)),value(53,dictBytes));
    encodeBad(VtValue(VtArray<VtValue>{VtValue(asset),VtValue(dict)}),ErrorKind::Unsupported);
    VtDictionary outer; outer["nested"] = VtValue(dict);
    encodeExact(VtValue(outer),value(31,le(1,8)+text("nested")+dictBytes));
    auto limits = EncodeLimits{}; limits.maxStringBytes = authored.size();
    check(encodeValue(VtValue(asset),limits) == assetBytes,"asset caches charged as authored string bytes");
    check(asset.GetAuthoredPath() == authored && asset.GetEvaluatedPath() == "evaluated.usda" &&
        asset.GetResolvedPath() == resolved,"encoding changed authored asset or its caches");

    for (const std::string path : {".","..","../A","../..","../A.p","/A{v=x}.p",
        "/Root{model=hero}","/Root.rel[/Target]"}) {
        const SdfPath source(path);
        check(!source.IsEmpty(),"invalid path fixture");
        encodeExact(VtValue(source),value(55,text(path)));
    }
    const SdfPath mapper = SdfPath("/Source.attr").AppendMapper(SdfPath("/Target.attr"));
    const SdfPath mapperArg = mapper.AppendMapperArg(TfToken("argument"));
    check(mapper.IsMapperPath() && mapperArg.IsMapperArgPath(),"SDK mapper path fixture invalid");
    encodeExact(VtValue(mapperArg),value(55,text("/Source.attr.mapper[/Target.attr].argument")));
    std::string nested = "/Leaf";
    for (unsigned i = 0; i < 12; ++i) nested = "/Root.rel["+nested+"]";
    const VtValue targetPath{SdfPath(nested)};
    encodeExact(targetPath,value(55,text(nested)));
    limits = {}; limits.maxDepth = 3;
    encodeBad(targetPath,ErrorKind::ResourceLimit,limits);
    std::string many = "/Root{model=hero}";
    for (unsigned i = 0; i < 32; ++i) many += (i == 0 ? "Child" : "/Child")+std::to_string(i);
    const VtValue componentPath{SdfPath(many)};
    encodeExact(componentPath,value(55,text(many)));
    limits = {}; limits.maxElements = 4;
    encodeBad(componentPath,ErrorKind::ResourceLimit,limits);
    limits = {}; limits.maxStringBytes = 1;
    encodeBad(componentPath,ErrorKind::ResourceLimit,limits);
    limits = {}; limits.maxEncodedBytes = 7;
    encodeBad(componentPath,ErrorKind::ResourceLimit,limits);
}
void encoderBatch() {
    context = "encoder shared batch budgets";
    const VtValue a(std::string("abc")), b(-31);
    const DecodedValues input{{a,VtValue(),b},{b,VtValue(),a}};
    const auto before = input;
    const auto result = encodeValues(input);
    size_t elementsUsed = 0;
    check(measureValues(input,{},&elementsUsed) == 30 && elementsUsed == 6,
          "retained measurement usage omits values/deletions");
    size_t emptyElements = 1;
    check(measureValues({}, {}, &emptyElements) == 0 && emptyElements == 0,
          "empty retained measurement reports work");
    check(result.fields == std::vector<std::string>{value(10,text("abc")),{},value(3,ints({-31}))},"encoded field order/deletion differs");
    check(result.samples == std::vector<std::string>{value(3,ints({-31})),{},value(10,text("abc"))},"encoded sample order/deletion differs");
    check(input.fields == before.fields && input.samples == before.samples,"encoding batch changed inputs");
    auto reject = [](const DecodedValues& input, ErrorKind kind, const EncodeLimits& limits) {
        const auto before = input;
        EncodedValues output{{"sentinel field"},{"sentinel sample"}};
        bool rejected = false;
        try { output = encodeValues(input,limits); }
        catch (const DecodeError& error) { check(error.kind == kind,"wrong batch encoding error"); rejected = true; }
        check(rejected,"batch encoding accepted unsupported or over-budget input");
        check(output.fields == std::vector<std::string>{"sentinel field"} &&
            output.samples == std::vector<std::string>{"sentinel sample"},"failed batch exposed partial output");
        check(input.fields == before.fields && input.samples == before.samples,"failed batch encoding changed input");
    };
    const DecodedValues aliases{{a,a},{a}}; // Copies share a single VtValue allocation.
    auto limits = EncodeLimits{}; limits.maxEncodedBytes = value(10,text("abc")).size()*2;
    check(encodeValue(a,limits) == value(10,text("abc")),"individual encoder byte budget fixture invalid");
    reject(aliases,ErrorKind::ResourceLimit,limits);
    limits = {}; limits.maxStringBytes = 6;
    check(encodeValue(a,limits) == value(10,text("abc")),"individual encoder string budget fixture invalid");
    reject(aliases,ErrorKind::ResourceLimit,limits);
    const VtValue array(VtArray<int>{1,2,3,4});
    limits = {}; limits.maxElements = 5;
    check(encodeValue(array,limits) == value(0x83,items({ints({1}),ints({2}),ints({3}),ints({4})})),
        "individual encoder element budget fixture invalid");
    reject(DecodedValues{{array},{array}},ErrorKind::ResourceLimit,limits);
    reject(DecodedValues{{a,b},{a,VtValue(std::vector<int>{7})}},ErrorKind::Unsupported,{});
    check(encodeValues(input).fields == result.fields,"failed batch poisoned later encoding");
}
void encoderHistories(uint64_t seed) {
    std::mt19937_64 random(seed);
    DecodedValues state; state.fields.resize(12); state.samples.resize(12);
    EncodedValues expected; expected.fields.resize(12); expected.samples.resize(12);
    uint64_t fingerprint = UINT64_C(14695981039346656037);
    for (size_t step = 0; step < 200; ++step) {
        context = "encoder history seed " + std::to_string(seed) + " step " + std::to_string(step);
        const bool sample = random() % 2;
        const size_t index = random() % 12;
        VtValue authored;
        std::string bytes;
        const int number = int(random() % 20001)-10000;
        std::string string;
        const size_t length = random() % 24;
        for (size_t i = 0; i < length; ++i) string += char(random() % 256);
        switch (random() % 5) {
        case 0: break; // Delete the authored value at this field/sample slot.
        case 1: authored = VtValue(number); bytes = value(3,ints({number})); break;
        case 2: authored = VtValue(string); bytes = value(10,text(string)); break;
        case 3:
            authored = VtValue(VtArray<int>{number,number+1,-number});
            bytes = value(0x83,items({ints({number}),ints({number+1}),ints({-number})})); break;
        default: {
            VtDictionary dict; dict["count"] = number; dict["text"] = string;
            authored = VtValue(dict);
            bytes = value(31,le(2,8)+text("count")+value(3,ints({number}))+text("text")+value(10,text(string))); break;
        }
        }
        (sample ? state.samples : state.fields)[index] = authored;
        (sample ? expected.samples : expected.fields)[index] = bytes;
        const auto before = state;
        const auto encoded = encodeValues(state);
        check(encoded.fields == expected.fields && encoded.samples == expected.samples,"seeded history diverged from independent byte oracle");
        check(state.fields == before.fields && state.samples == before.samples,"seeded history encoding changed source values");
        for (const auto& field : expected.fields) for (unsigned char c : field) { fingerprint ^= c; fingerprint *= UINT64_C(1099511628211); }
        for (const auto& item : expected.samples) for (unsigned char c : item) { fingerprint ^= c; fingerprint *= UINT64_C(1099511628211); }
    }
    std::cout << "encoder history seed=" << seed << " steps=200 fingerprint=" << fingerprint << std::endl;
}
void mutations(uint64_t seed) {
    std::mt19937_64 random(seed);
    ValueLimits limits;
    limits.maxEncodedBytes = 65536; limits.maxDecodedBytes = 65536;
    limits.maxStringBytes = 16384; limits.maxElements = 512;
    limits.maxDepth = 16; limits.maxComparisonWork = 131072;
    size_t accepted = 0, rejected = 0;
    uint64_t fingerprint = UINT64_C(14695981039346656037);
    // Safety fuzzing is separate from the exact-value oracle above: successful
    // mutated values need not equal their seed. Every attempted input is bounded,
    // and seed + iteration alone reconstructs failures.
    for (size_t iteration = 0; iteration < 3000; ++iteration) {
        context = "mutation seed " + std::to_string(seed) + " iteration " + std::to_string(iteration);
        std::string bytes = valid[random() % valid.size()];
        switch (random() % 5) {
        case 0: {
            const size_t at = random() % bytes.size();
            bytes[at] ^= char(1u << (random() % 8)); break;
        }
        case 1: bytes.resize(random() % (bytes.size()+1)); break;
        case 2: {
            const size_t at = random() % (bytes.size()+1);
            bytes.insert(at,1,char(random())); break;
        }
        case 3: {
            const size_t start = random() % bytes.size();
            const size_t count = std::min(size_t(8),bytes.size()-start);
            bytes.replace(start,count,count,char(0xff)); break;
        }
        default: bytes += valid[random() % valid.size()]; break;
        }
        for (unsigned char c : bytes) { fingerprint ^= c; fingerprint *= UINT64_C(1099511628211); }
        try { (void)decodeValue(bytes,limits); ++accepted; }
        catch (const DecodeError&) { ++rejected; }
    }
    check(accepted > 0 && rejected > 0, "mutation corpus did not exercise both outcomes");
    std::cout << "mutation seed=" << seed << " accepted=" << accepted << " rejected=" << rejected
              << " fingerprint=" << fingerprint << std::endl;
}
} // namespace
void usdEnvelopeEncoding();
int main(int argc, char** argv) {
    try {
        const uint64_t seed = argc > 1 ? std::stoull(argv[1]) : 42;
        std::cout << "USD value codec seed=" << seed << std::endl;
        numericAndGf(); containers(); malformedAndBounds(); pathBudgets(); copyBudgets(); batch();
        encoderFailuresAndBits(); encoderAuthoredAssetsAndPaths(); encoderBatch(); encoderHistories(seed); mutations(seed);
        context = "typed USD envelope bridge"; usdEnvelopeEncoding();
        std::cout << "USD value codec: all 64 scalar IDs and their USD array forms, independent values, malformed inputs and shared budgets passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "USD value codec test failed (" << context << "): " << error.what() << '\n'; return 1;
    }
}
