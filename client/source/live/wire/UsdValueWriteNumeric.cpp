// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "UsdValueWriter.h"

namespace omni::usd::live::wire::detail {
void Writer::numeric(uint16_t tag, const VtValue& value) {
    switch (tag & 0x7f) {
    case unsigned(ValueType::Bool): typed<ValueType::Bool>(tag, value); return;
    case unsigned(ValueType::UChar): typed<ValueType::UChar>(tag, value); return;
    case unsigned(ValueType::Int): typed<ValueType::Int>(tag, value); return;
    case unsigned(ValueType::UInt): typed<ValueType::UInt>(tag, value); return;
    case unsigned(ValueType::Int64): typed<ValueType::Int64>(tag, value); return;
    case unsigned(ValueType::UInt64): typed<ValueType::UInt64>(tag, value); return;
    case unsigned(ValueType::Half): typed<ValueType::Half>(tag, value); return;
    case unsigned(ValueType::Float): typed<ValueType::Float>(tag, value); return;
    case unsigned(ValueType::Double): typed<ValueType::Double>(tag, value); return;
    case unsigned(ValueType::String): typed<ValueType::String>(tag, value); return;
    case unsigned(ValueType::Token): typed<ValueType::Token>(tag, value); return;
    case unsigned(ValueType::AssetPath): typed<ValueType::AssetPath>(tag, value); return;
    case unsigned(ValueType::Matrix2d): typed<ValueType::Matrix2d>(tag, value); return;
    case unsigned(ValueType::Matrix3d): typed<ValueType::Matrix3d>(tag, value); return;
    case unsigned(ValueType::Matrix4d): typed<ValueType::Matrix4d>(tag, value); return;
    case unsigned(ValueType::Quatd): typed<ValueType::Quatd>(tag, value); return;
    case unsigned(ValueType::Quatf): typed<ValueType::Quatf>(tag, value); return;
    case unsigned(ValueType::Quath): typed<ValueType::Quath>(tag, value); return;
    case unsigned(ValueType::Vec2d): typed<ValueType::Vec2d>(tag, value); return;
    case unsigned(ValueType::Vec2f): typed<ValueType::Vec2f>(tag, value); return;
    case unsigned(ValueType::Vec2h): typed<ValueType::Vec2h>(tag, value); return;
    case unsigned(ValueType::Vec2i): typed<ValueType::Vec2i>(tag, value); return;
    case unsigned(ValueType::Vec3d): typed<ValueType::Vec3d>(tag, value); return;
    case unsigned(ValueType::Vec3f): typed<ValueType::Vec3f>(tag, value); return;
    case unsigned(ValueType::Vec3h): typed<ValueType::Vec3h>(tag, value); return;
    case unsigned(ValueType::Vec3i): typed<ValueType::Vec3i>(tag, value); return;
    case unsigned(ValueType::Vec4d): typed<ValueType::Vec4d>(tag, value); return;
    case unsigned(ValueType::Vec4f): typed<ValueType::Vec4f>(tag, value); return;
    case unsigned(ValueType::Vec4h): typed<ValueType::Vec4h>(tag, value); return;
    case unsigned(ValueType::Vec4i): typed<ValueType::Vec4i>(tag, value); return;
    default: throw DecodeError(ErrorKind::Unsupported, "unsupported USD encoding type");
    }
}
} // namespace omni::usd::live::wire::detail
