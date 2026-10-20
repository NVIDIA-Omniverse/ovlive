// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "UsdValueReader.h"

namespace omni::usd::live::wire::detail {
VtValue Reader::compound(uint16_t tag) {
    const bool array = (tag & 0x80) != 0;
    switch (tag & 0x7f) {
    case unsigned(ValueType::Dictionary): return make<ValueType::Dictionary>(array);
    case unsigned(ValueType::TokenListOp): return make<ValueType::TokenListOp>(array);
    case unsigned(ValueType::StringListOp): return make<ValueType::StringListOp>(array);
    case unsigned(ValueType::PathListOp): return make<ValueType::PathListOp>(array);
    case unsigned(ValueType::ReferenceListOp): return make<ValueType::ReferenceListOp>(array);
    case unsigned(ValueType::IntListOp): return make<ValueType::IntListOp>(array);
    case unsigned(ValueType::Int64ListOp): return make<ValueType::Int64ListOp>(array);
    case unsigned(ValueType::UIntListOp): return make<ValueType::UIntListOp>(array);
    case unsigned(ValueType::UInt64ListOp): return make<ValueType::UInt64ListOp>(array);
    case unsigned(ValueType::PathVector): return make<ValueType::PathVector>(array);
    case unsigned(ValueType::TokenVector): return make<ValueType::TokenVector>(array);
    case unsigned(ValueType::Specifier): return make<ValueType::Specifier>(array);
    case unsigned(ValueType::Permission): return make<ValueType::Permission>(array);
    case unsigned(ValueType::Variability): return make<ValueType::Variability>(array);
    case unsigned(ValueType::VariantSelectionMap): return make<ValueType::VariantSelectionMap>(array);
    case unsigned(ValueType::TimeSamples): return make<ValueType::TimeSamples>(array);
    case unsigned(ValueType::Payload): return make<ValueType::Payload>(array);
    case unsigned(ValueType::DoubleVector): return make<ValueType::DoubleVector>(array);
    case unsigned(ValueType::LayerOffsetVector): return make<ValueType::LayerOffsetVector>(array);
    case unsigned(ValueType::StringVector): return make<ValueType::StringVector>(array);
    case unsigned(ValueType::ValueBlock): return make<ValueType::ValueBlock>(array);
    case unsigned(ValueType::Value): return make<ValueType::Value>(array);
    case unsigned(ValueType::UnregisteredValue): return make<ValueType::UnregisteredValue>(array);
    case unsigned(ValueType::UnregisteredValueListOp): return make<ValueType::UnregisteredValueListOp>(array);
    case unsigned(ValueType::Path): return make<ValueType::Path>(array);
    case unsigned(ValueType::SpecType): return make<ValueType::SpecType>(array);
    case unsigned(ValueType::PayloadListOp): return make<ValueType::PayloadListOp>(array);
    case unsigned(ValueType::TimeCode): return make<ValueType::TimeCode>(array);
    case unsigned(ValueType::Range1d): return make<ValueType::Range1d>(array);
    case unsigned(ValueType::Range2d): return make<ValueType::Range2d>(array);
    case unsigned(ValueType::Range3d): return make<ValueType::Range3d>(array);
    case unsigned(ValueType::Range1f): return make<ValueType::Range1f>(array);
    case unsigned(ValueType::Range2f): return make<ValueType::Range2f>(array);
    case unsigned(ValueType::Range3f): return make<ValueType::Range3f>(array);
    default: throw DecodeError(ErrorKind::Unsupported, "unsupported USD value type");
    }
}
} // namespace omni::usd::live::wire::detail
