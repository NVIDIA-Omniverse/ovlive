// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "SceneAttachment.h"
#include "SceneCreation.h"
#include <pxr/usd/sdf/layer.h>
#include <memory>
#include <mutex>

namespace omni::usd::live {
// The observable state of one USD creation workflow. Its wrapper mutates it on
// the workflow's original serial host; snapshot() is thread-safe and never
// waits, so off-host close waits and the file-format binding read it.
class UsdCreationWorkflow final {
public:
    struct Snapshot {
        std::shared_ptr<SceneAttachment> attachment;
        bool openingAttempted = false, activated = false, closing = false, activeCall = false;
        // The hook abandoned opening before actor construction.
        bool openingAbandoned = false;
    };
    UsdCreationWorkflow(std::shared_ptr<Runtime> runtime, std::shared_ptr<SceneCreation> creation)
        : runtime(std::move(runtime)), creation(std::move(creation)) {}
    const std::shared_ptr<Runtime> runtime;
    const std::shared_ptr<SceneCreation> creation;
    Snapshot snapshot() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return snapshot_;
    }
    void begin() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        snapshot_.activeCall = true;
    }
    // The previous attachment owner retires after the lock is released.
    void finish(Snapshot value) noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            std::swap(snapshot_, value);
        }
    }
private:
    mutable std::mutex mutex_;
    Snapshot snapshot_;
};
// Original host, inside USD CreateNew before the target is exposed: marks the
// canonical, unattached target as this workflow's initializing layer. No RPC,
// registration or capture happens here.
void prepareInitializingCreationTarget(PXR_NS::SdfLayer&, const std::shared_ptr<UsdCreationWorkflow>&);
// Deferred bootstrap publication for the borrowed initializing target.
SceneAttachment::Publication initializingCreationPublication(PXR_NS::SdfLayer&,
    const std::shared_ptr<UsdCreationWorkflow>&);
// Original host only; the workflow already owns this confirmed actor.
void bindInitializingCreationAttachment(PXR_NS::SdfLayer&, const std::shared_ptr<UsdCreationWorkflow>&,
    const std::shared_ptr<SceneAttachment>&);
} // namespace omni::usd::live
