// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "UsdLayerCreation.h"
#include <cstddef>

namespace omni::usd::live {
// Process-owned, never evicting retention. Reserve BEFORE preparation and
// install BEFORE issuance. A successfully activated implicit creation leaves
// through retireAdopted(); failed or uncertain workflows stay until transfer
// or explicit cleanup. A lease excludes transfer/cleanup while a hook or
// finalization is using the owner. Dropping a populated lease only clears BUSY;
// it never closes or discards the retained workflow, including during unwinding.
// The process root deliberately has no static destructor: final USD destruction
// is legal only on its original host, through explicit cleanup or transfer.
constexpr size_t kRetainedUsdCreationCapacity = 64;
struct RetainedUsdCreationInfo {
    uint64_t subjectId = 0;
    std::string liveUri;
};
class UsdCreationLease final {
public:
    UsdCreationLease() = default;
    UsdCreationLease(UsdCreationLease&&) noexcept;
    UsdCreationLease& operator=(UsdCreationLease&&) noexcept;
    ~UsdCreationLease();
    UsdCreationLease(const UsdCreationLease&) = delete;
    UsdCreationLease& operator=(const UsdCreationLease&) = delete;
    // Install is host-confined and accepts only an unstarted prepared owner.
    // On failure caller retains ownership; on success value becomes empty.
    void install(std::shared_ptr<UsdLayerCreation>& value);
    UsdLayerCreation& workflow() const;
    // Original-host only. Remove retention, returning its sole workflow root.
    // Caller must allocate any public wrapper BEFORE this ownership transfer.
    std::shared_ptr<UsdLayerCreation> take();
    // Original-host poll-only destructive cleanup; BUSY keeps the slot. A call
    // that overlaps another creation call returns BUSY untouched; otherwise the
    // workflow is marked closing even when stop() returns BUSY, so a later
    // activation is CANCELLED: SdfLayer::Export any content to keep before
    // requesting cleanup.
    OmniUsdLiveResult close() noexcept;
    // Original-host only, for an implicit creation whose activation succeeded:
    // adoptIntoLayer(), then destroy the workflow and free the slot, and only
    // then drop the layer reference it held across them, so an unreferenced
    // destination starts closing after the slot is free. A failed adoption
    // returns its reason and keeps the workflow retained.
    OmniUsdLiveResult retireAdopted() noexcept;
    // Test seam: retireAdopted calls hook(context) on the host after a
    // successful adoption, before it frees the slot; null disables it. It loads
    // the hook and then the context, so hooks must tolerate a null context.
    static void setAfterAdoptionForTest(void (*hook)(void*) noexcept, void* context) noexcept;
private:
    explicit UsdCreationLease(size_t slot) noexcept : slot_(slot) {}
    void reset() noexcept;
    size_t slot_ = kRetainedUsdCreationCapacity;
    friend UsdCreationLease reserveUsdCreation();
    friend UsdCreationLease acquireUsdCreation(uint64_t);
};
UsdCreationLease reserveUsdCreation(); // Throws RESOURCE_EXHAUSTED at capacity.
UsdCreationLease acquireUsdCreation(uint64_t subject); // BUSY if leased, NOT_FOUND if gone.
// Thread-neutral, no USD, no host callbacks. Ordered successor observation,
// not a pin or a transaction across calls. Reserved slots without a subject
// are invisible; occupied slots remain visible while leased or retiring.
bool nextRetainedUsdCreation(uint64_t afterSubject, RetainedUsdCreationInfo&);
} // namespace omni::usd::live
