// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "HostSceneProjection.h"
#include "SdfOperation.h"
#include <memory>
#include <utility>
#include <vector>

namespace omni::usd::live::test {
// A production host backing over `base`. Every mutation through it records
// exactly as a live layer's host edit buffer does.
inline PXR_NS::TfRefPtr<HostSceneProjection> recordingBacking(const WireScenePtr& base) {
    return PXR_NS::TfCreateRefPtr(new HostSceneProjection(base, std::make_shared<HostFailureLatch>()));
}
// The backing's buffered operations and a capture of its storage, as one
// immutable log. The backing keeps its buffer.
inline std::shared_ptr<const SdfOperationLog> recordedLog(const HostSceneProjection& backing) {
    const auto edits = backing.pendingLocalEdits();
    return std::make_shared<const SdfOperationLog>(SdfOperationLog{edits.data, edits.operations});
}
// Author through a fresh recording backing and return its log.
template <class Author>
std::shared_ptr<const SdfOperationLog> capturedLog(const WireScenePtr& base, Author&& author) {
    const auto backing = recordingBacking(base);
    author(static_cast<PXR_NS::SdfAbstractData&>(*backing));
    return recordedLog(*backing);
}
// `log` with one more record and the same data, for records a host backing
// refuses before mutation (CopyFrom) or never produces.
inline std::shared_ptr<const SdfOperationLog> withOperation(const SdfOperationLog& log, SdfOperation operation) {
    auto operations = std::make_shared<std::vector<SdfOperation>>(*log.operations);
    operations->push_back(std::move(operation));
    return std::make_shared<const SdfOperationLog>(SdfOperationLog{log.data, std::move(operations)});
}
} // namespace omni::usd::live::test
