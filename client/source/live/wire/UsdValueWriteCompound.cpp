// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "UsdValueWriter.h"

namespace omni::usd::live::wire::detail {
void Writer::compound(uint16_t tag, const VtValue& value) {
    switch (tag & 0x7f) {
    case unsigned(ValueType::Dictionary): typed<ValueType::Dictionary>(tag, value); return;
    case unsigned(ValueType::TokenListOp): typed<ValueType::TokenListOp>(tag, value); return;
    case unsigned(ValueType::StringListOp): typed<ValueType::StringListOp>(tag, value); return;
    case unsigned(ValueType::PathListOp): typed<ValueType::PathListOp>(tag, value); return;
    case unsigned(ValueType::ReferenceListOp): typed<ValueType::ReferenceListOp>(tag, value); return;
    case unsigned(ValueType::IntListOp): typed<ValueType::IntListOp>(tag, value); return;
    case unsigned(ValueType::Int64ListOp): typed<ValueType::Int64ListOp>(tag, value); return;
    case unsigned(ValueType::UIntListOp): typed<ValueType::UIntListOp>(tag, value); return;
    case unsigned(ValueType::UInt64ListOp): typed<ValueType::UInt64ListOp>(tag, value); return;
    case unsigned(ValueType::PathVector): typed<ValueType::PathVector>(tag, value); return;
    case unsigned(ValueType::TokenVector): typed<ValueType::TokenVector>(tag, value); return;
    case unsigned(ValueType::Specifier): typed<ValueType::Specifier>(tag, value); return;
    case unsigned(ValueType::Permission): typed<ValueType::Permission>(tag, value); return;
    case unsigned(ValueType::Variability): typed<ValueType::Variability>(tag, value); return;
    case unsigned(ValueType::VariantSelectionMap): typed<ValueType::VariantSelectionMap>(tag, value); return;
    case unsigned(ValueType::TimeSamples): typed<ValueType::TimeSamples>(tag, value); return;
    case unsigned(ValueType::Payload): typed<ValueType::Payload>(tag, value); return;
    case unsigned(ValueType::DoubleVector): typed<ValueType::DoubleVector>(tag, value); return;
    case unsigned(ValueType::LayerOffsetVector): typed<ValueType::LayerOffsetVector>(tag, value); return;
    case unsigned(ValueType::StringVector): typed<ValueType::StringVector>(tag, value); return;
    case unsigned(ValueType::ValueBlock): typed<ValueType::ValueBlock>(tag, value); return;
    case unsigned(ValueType::Value): typed<ValueType::Value>(tag, value); return;
    case unsigned(ValueType::UnregisteredValue): typed<ValueType::UnregisteredValue>(tag, value); return;
    case unsigned(ValueType::UnregisteredValueListOp): typed<ValueType::UnregisteredValueListOp>(tag, value); return;
    case unsigned(ValueType::Path): typed<ValueType::Path>(tag, value); return;
    case unsigned(ValueType::SpecType): typed<ValueType::SpecType>(tag, value); return;
    case unsigned(ValueType::PayloadListOp): typed<ValueType::PayloadListOp>(tag, value); return;
    case unsigned(ValueType::TimeCode): typed<ValueType::TimeCode>(tag, value); return;
    case unsigned(ValueType::Range1d): typed<ValueType::Range1d>(tag, value); return;
    case unsigned(ValueType::Range2d): typed<ValueType::Range2d>(tag, value); return;
    case unsigned(ValueType::Range3d): typed<ValueType::Range3d>(tag, value); return;
    case unsigned(ValueType::Range1f): typed<ValueType::Range1f>(tag, value); return;
    case unsigned(ValueType::Range2f): typed<ValueType::Range2f>(tag, value); return;
    case unsigned(ValueType::Range3f): typed<ValueType::Range3f>(tag, value); return;
    default: throw DecodeError(ErrorKind::Unsupported, "unsupported USD encoding type");
    }
}
} // namespace omni::usd::live::wire::detail
