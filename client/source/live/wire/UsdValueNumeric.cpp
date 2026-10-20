// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "UsdValueReader.h"

namespace omni::usd::live::wire::detail {
VtValue Reader::numeric(uint16_t tag) {
    const bool array = (tag & 0x80) != 0;
    switch (tag & 0x7f) {
    case unsigned(ValueType::Bool): return make<ValueType::Bool>(array);
    case unsigned(ValueType::UChar): return make<ValueType::UChar>(array);
    case unsigned(ValueType::Int): return make<ValueType::Int>(array);
    case unsigned(ValueType::UInt): return make<ValueType::UInt>(array);
    case unsigned(ValueType::Int64): return make<ValueType::Int64>(array);
    case unsigned(ValueType::UInt64): return make<ValueType::UInt64>(array);
    case unsigned(ValueType::Half): return make<ValueType::Half>(array);
    case unsigned(ValueType::Float): return make<ValueType::Float>(array);
    case unsigned(ValueType::Double): return make<ValueType::Double>(array);
    case unsigned(ValueType::String): return make<ValueType::String>(array);
    case unsigned(ValueType::Token): return make<ValueType::Token>(array);
    case unsigned(ValueType::AssetPath): return make<ValueType::AssetPath>(array);
    case unsigned(ValueType::Matrix2d): return make<ValueType::Matrix2d>(array);
    case unsigned(ValueType::Matrix3d): return make<ValueType::Matrix3d>(array);
    case unsigned(ValueType::Matrix4d): return make<ValueType::Matrix4d>(array);
    case unsigned(ValueType::Quatd): return make<ValueType::Quatd>(array);
    case unsigned(ValueType::Quatf): return make<ValueType::Quatf>(array);
    case unsigned(ValueType::Quath): return make<ValueType::Quath>(array);
    case unsigned(ValueType::Vec2d): return make<ValueType::Vec2d>(array);
    case unsigned(ValueType::Vec2f): return make<ValueType::Vec2f>(array);
    case unsigned(ValueType::Vec2h): return make<ValueType::Vec2h>(array);
    case unsigned(ValueType::Vec2i): return make<ValueType::Vec2i>(array);
    case unsigned(ValueType::Vec3d): return make<ValueType::Vec3d>(array);
    case unsigned(ValueType::Vec3f): return make<ValueType::Vec3f>(array);
    case unsigned(ValueType::Vec3h): return make<ValueType::Vec3h>(array);
    case unsigned(ValueType::Vec3i): return make<ValueType::Vec3i>(array);
    case unsigned(ValueType::Vec4d): return make<ValueType::Vec4d>(array);
    case unsigned(ValueType::Vec4f): return make<ValueType::Vec4f>(array);
    case unsigned(ValueType::Vec4h): return make<ValueType::Vec4h>(array);
    case unsigned(ValueType::Vec4i): return make<ValueType::Vec4i>(array);
    default: throw DecodeError(ErrorKind::Unsupported, "unsupported USD value type");
    }
}
} // namespace omni::usd::live::wire::detail
