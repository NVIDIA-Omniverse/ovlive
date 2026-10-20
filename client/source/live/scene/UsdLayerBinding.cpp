// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "UsdLayerBinding.h"

using namespace PXR_NS;
namespace omni::usd::live {
UsdLayerBinding::~UsdLayerBinding() { detach(); }

void UsdLayerBinding::bind(SdfLayer* layer) {
    if (!layer) throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
    // Creation of USD's weak bookkeeping may allocate. The caller protects
    // the layer here, and preparation leaves this binding unchanged on failure.
    SdfLayerHandle prepared(layer);
    std::lock_guard<std::mutex> lock(mutex_);
    if (detached_) throw OmniUsdLiveResult(OMNI_USD_LIVE_CANCELLED);
    if (bound_) throw OmniUsdLiveResult(OMNI_USD_LIVE_ALREADY_EXISTS);
    layer_ = std::move(prepared);
    bound_ = true;
}

void UsdLayerBinding::attachAttachment(const std::shared_ptr<SceneAttachment>& attachment) {
    if (!attachment) throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
    std::lock_guard<std::mutex> lock(mutex_);
    if (detached_) throw OmniUsdLiveResult(OMNI_USD_LIVE_CANCELLED);
    if (!bound_) throw OmniUsdLiveResult(OMNI_USD_LIVE_BUSY);
    if (attachment_) throw OmniUsdLiveResult(OMNI_USD_LIVE_ALREADY_EXISTS);
    attachment_ = attachment;
}

std::shared_ptr<SceneAttachment> UsdLayerBinding::attachment() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (detached_) throw OmniUsdLiveResult(OMNI_USD_LIVE_CANCELLED);
    return attachment_;
}

SdfLayerRefPtr UsdLayerBinding::pinLayer() const {
    SdfLayerRefPtr layer;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (detached_ || !bound_) throw OmniUsdLiveResult(OMNI_USD_LIVE_CANCELLED);
        // The state delegate's destructor calls detach() during the layer's
        // member teardown, before TfRefBase goes. Holding this lock therefore
        // keeps TfRefBase intact even if final destruction has begun;
        // AddRefIfNonzero either acquires ownership or returns null.
        layer = TfCreateRefPtrFromProtectedWeakPtr(layer_);
    }
    if (!layer) throw OmniUsdLiveResult(OMNI_USD_LIVE_CANCELLED);
    return layer;
}

void UsdLayerBinding::publish(SceneJournal::SnapshotPtr snapshot, const Publisher& publisher) {
    if (!snapshot || !publisher) throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
    const auto layer = pinLayer();
    publisher(layer.operator->(), std::move(snapshot));
    // Even on exception, dropping the final strong reference is outside the
    // binding mutex: layer destruction may synchronously reenter detach().
}

void UsdLayerBinding::detach() noexcept {
    const auto attachment = detachWithoutClosing();
    if (attachment) {
        try {
            (void)attachment->close(0, true); // The layer is gone: nothing else closes this backing.
        } catch (...) { /* Destructor-safe best effort; attachment still retires below. */
        }
    }
}

std::shared_ptr<SceneAttachment> UsdLayerBinding::detachWithoutClosing() noexcept {
    std::shared_ptr<SceneAttachment> attachment;
    SdfLayerHandle layer;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        detached_ = true;
        layer = std::move(layer_);
        attachment = std::move(attachment_);
    }
    // No owner or host callback is destroyed while protecting weak promotion.
    return attachment;
}
} // namespace omni::usd::live
