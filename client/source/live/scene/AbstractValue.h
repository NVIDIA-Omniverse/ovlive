// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <pxr/base/vt/value.h>
#include <pxr/usd/sdf/abstractData.h>

namespace omni::usd::live {
// Extract a foreign abstract value into `output`, which the caller owns so a
// failed extraction's value retires where the caller chooses. `initial` is
// the result when extraction succeeds without assigning, as stock SdfData
// extracts into an existing field slot. Stock VtValue assignment of a
// remote-stored type records the type before it allocates; extraction starts
// from a zero-filled local placeholder so a throwing assignment still leaves
// `output` destructible. Returns GetValue's result.
bool extractAbstractValue(const PXR_NS::SdfAbstractDataConstValue& value, const PXR_NS::VtValue& initial,
    PXR_NS::VtValue& output);
} // namespace omni::usd::live
