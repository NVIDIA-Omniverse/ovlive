// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/value.h>
#include <pxr/usd/sdf/abstractData.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/sdf/types.h>
#include <memory>
#include <vector>

namespace omni::usd::live {
// One mutation a host applied to a live layer's backing, in call order. It is
// the attempted call, not proof that storage changed. CopyFrom records never
// come from a host backing, which refuses CopyFrom before mutation, and the
// field-edit compiler rejects them.
enum class SdfOperationKind {
    CreateSpec,
    EraseSpec,
    MoveSpec,
    SetField,
    EraseField,
    SetDictKey,
    EraseDictKey,
    SetSample,
    EraseSample,
    CopyFrom
};
struct SdfOperation {
    explicit SdfOperation(SdfOperationKind operation) : kind(operation) {}
    SdfOperationKind kind;
    PXR_NS::SdfPath path, destination;
    PXR_NS::SdfSpecType specType = PXR_NS::SdfSpecTypeUnknown;
    PXR_NS::TfToken field, key;
    double time = 0;
    PXR_NS::VtValue value;
    PXR_NS::SdfAbstractDataConstRefPtr replacement;
};
// An immutable operation log and the storage it produced. Both owners are
// independent of the host layer and may be shared with concurrent readers.
struct SdfOperationLog {
    PXR_NS::SdfAbstractDataConstRefPtr data;
    std::shared_ptr<const std::vector<SdfOperation>> operations;
};
} // namespace omni::usd::live
