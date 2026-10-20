// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <pxr/usd/sdf/abstractData.h>
#include <pxr/usd/sdf/data.h>

// Test oracle: an independent stock SdfData copy, compared with Sdf's own
// spec-and-field equality. Values compare with VtValue equality, so fixtures
// that use it do not depend on NaN payloads or signed zero.
inline PXR_NS::SdfDataRefPtr copyData(const PXR_NS::SdfAbstractDataConstPtr& source) {
    auto copy = PXR_NS::TfCreateRefPtr(new PXR_NS::SdfData);
    copy->CopyFrom(source);
    return copy;
}
inline bool sameData(const PXR_NS::SdfAbstractDataConstPtr& a, const PXR_NS::SdfAbstractDataConstPtr& b) {
    return a && b && a->Equals(PXR_NS::SdfAbstractDataRefPtr(const_cast<PXR_NS::SdfAbstractData*>(get_pointer(b))));
}
