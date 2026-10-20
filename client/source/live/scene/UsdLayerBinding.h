// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "SceneAttachment.h"
#include <pxr/usd/sdf/layer.h>
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>

namespace omni::usd::live {
// Lifetime bridge for a live SdfLayer and its one canonical attachment.
// The layer's state delegate owns this binding; the binding owns only a WEAK
// layer handle.
// Actor/host publication callbacks must capture a weak binding, not the layer
// or attachment, to avoid an ownership cycle.
//
// Mandatory construction/destruction protocol: bind() runs while construction
// or an existing strong owner protects the layer. The layer alone owns its
// state delegate, which calls detach() from its destructor during the layer's
// member teardown, before the layer's TfRefBase and TfWeakBase teardown.
// The shared mutex then protects TfRefBase throughout weak-to-strong promotion,
// including the interval after refcount reaches zero but before destruction.
// This class supplies lifetime safety, not a host USD access/publication lock.
class UsdLayerBinding final {
public:
    using Publisher = std::function<void(PXR_NS::SdfLayer*, SceneJournal::SnapshotPtr)>;
    UsdLayerBinding() = default;
    ~UsdLayerBinding();
    UsdLayerBinding(const UsdLayerBinding&) = delete;
    UsdLayerBinding& operator=(const UsdLayerBinding&) = delete;

    // One use, including across detach. Null is INVALID_ARGUMENT, an already
    // bound live object is ALREADY_EXISTS, and a detached binding is CANCELLED.
    void bind(PXR_NS::SdfLayer*);
    // Caller keeps the incoming attachment on failure; no publication or close
    // runs under this mutex. Only one nonnull attachment may be installed.
    void attachAttachment(const std::shared_ptr<SceneAttachment>&);
    // Null before attachment installation; CANCELLED after detach. Returned
    // ownership keeps the attachment alive independently of layer destruction.
    std::shared_ptr<SceneAttachment> attachment() const;
    // Pin only lifetime, without accessing layer contents or host state.
    // Protected promotion and final release occur under the same rules as
    // publish(); callers still marshal content access to the serial host.
    PXR_NS::SdfLayerRefPtr pinLayer() const;
    // Caller supplies the host-safe context and complete change-block boundary.
    // Promotion is protected by mutex_; publisher and final strong-layer
    // release run outside it. A missing/expiring layer rejects with CANCELLED.
    void publish(SceneJournal::SnapshotPtr, const Publisher&);
    // Idempotent. Clear publication access before releasing ownership. A
    // releasing close(0), which lets a later open wait for the key, and
    // attachment retirement run outside the mutex and never block here.
    void detach() noexcept;
    // Invalidate weak promotion FIRST and return the attachment without closing it; the caller owns its close.
    std::shared_ptr<SceneAttachment> detachWithoutClosing() noexcept;
    // Thread-safe summary of the bound layer's host edit buffer. The host sets
    // it when a mutation records an edit and clears it once a flush has
    // admitted every buffered record as attachment work, or when a healthy
    // layer's buffer is empty. A failed flush leaves it set, and so does a
    // failed layer. Off-host readers use it instead of the host-confined
    // projection.
    void notePendingLocalEdits(bool pending) noexcept { pendingLocalEdits_.store(pending, std::memory_order_release); }
    bool hasPendingLocalEdits() const noexcept { return pendingLocalEdits_.load(std::memory_order_acquire); }
    // Count of the host edit buffer's records that no flush has admitted, and
    // the transient cause of a flush deferral that holds them (OK otherwise).
    // The host sets them after every record and every flush attempt; true
    // asks the caller to wake the attachment's actor. Status reads them
    // through the shared owner, which never keeps this binding alive.
    bool noteBufferedEdits(uint64_t count, OmniUsdLiveResult blocked = OMNI_USD_LIVE_OK) noexcept {
        return bufferedEdits_->note(count, blocked);
    }
    std::shared_ptr<BufferedEditCount> bufferedEdits() const noexcept { return bufferedEdits_; }

private:
    std::atomic<bool> pendingLocalEdits_{false};
    const std::shared_ptr<BufferedEditCount> bufferedEdits_ = std::make_shared<BufferedEditCount>();
    mutable std::mutex mutex_;
    PXR_NS::SdfLayerHandle layer_;
    std::shared_ptr<SceneAttachment> attachment_;
    bool bound_ = false, detached_ = false;
};
} // namespace omni::usd::live
