// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "UsdCreationRegistry.h"
#include <array>
#include <atomic>
#include <mutex>
#include <utility>

namespace omni::usd::live {
namespace {
std::atomic<void (*)(void*) noexcept> afterAdoption{nullptr};
std::atomic<void*> afterAdoptionContext{nullptr};
struct Slot {
    bool occupied = false, busy = false;
    RetainedUsdCreationInfo info;
    std::shared_ptr<UsdLayerCreation> owner;
};
struct Registry {
    std::mutex mutex;
    std::array<Slot, kRetainedUsdCreationCapacity> slots;
};
Registry& registry() {
    // Runtime/Host cannot own this root: workflows themselves retain both.
    // Intentional process lifetime also avoids off-host static teardown.
    static Registry* const root = new Registry;
    return *root;
}
}
UsdCreationLease::UsdCreationLease(UsdCreationLease&& other) noexcept
    : slot_(std::exchange(other.slot_, kRetainedUsdCreationCapacity)) {}
UsdCreationLease& UsdCreationLease::operator=(UsdCreationLease&& other) noexcept {
    if (this != &other) {
        reset();
        slot_ = std::exchange(other.slot_, kRetainedUsdCreationCapacity);
    }
    return *this;
}
UsdCreationLease::~UsdCreationLease() { reset(); }
void UsdCreationLease::reset() noexcept {
    if (slot_ == kRetainedUsdCreationCapacity) return;
    auto& root = registry();
    std::lock_guard<std::mutex> lock(root.mutex);
    auto& slot = root.slots[slot_];
    slot.busy = false;
    if (!slot.owner) {
        slot.occupied = false;
        slot.info.subjectId = 0;
        slot.info.liveUri.clear();
    }
    slot_ = kRetainedUsdCreationCapacity;
}
UsdCreationLease reserveUsdCreation() {
    auto& root = registry();
    std::lock_guard<std::mutex> lock(root.mutex);
    for (size_t i = 0; i != root.slots.size(); ++i) {
        auto& slot = root.slots[i];
        if (!slot.occupied) {
            slot.occupied = slot.busy = true;
            return UsdCreationLease(i);
        }
    }
    throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
}
UsdCreationLease acquireUsdCreation(uint64_t subject) {
    if (!subject) throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
    auto& root = registry();
    std::lock_guard<std::mutex> lock(root.mutex);
    for (size_t i = 0; i != root.slots.size(); ++i) {
        auto& slot = root.slots[i];
        if (slot.occupied && slot.info.subjectId == subject) {
            if (slot.busy) throw OmniUsdLiveResult(OMNI_USD_LIVE_BUSY);
            slot.busy = true;
            return UsdCreationLease(i);
        }
    }
    throw OmniUsdLiveResult(OMNI_USD_LIVE_NOT_FOUND);
}
void UsdCreationLease::install(std::shared_ptr<UsdLayerCreation>& value) {
    if (slot_ == kRetainedUsdCreationCapacity || !value) throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
    value->requireTransferHost(); // May invoke host code; never under registry mutex.
    const auto& finite = value->creation();
    const auto snapshot = finite->snapshot();
    if (snapshot.started || snapshot.closing || snapshot.phase != SceneCreation::Phase::Prepared)
        throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
    RetainedUsdCreationInfo info{finite->status()->id(), finite->admission()->canonicalUri()};
    auto& root = registry();
    std::lock_guard<std::mutex> lock(root.mutex);
    auto& slot = root.slots[slot_];
    if (slot.owner) throw OmniUsdLiveResult(OMNI_USD_LIVE_ALREADY_EXISTS);
    for (const auto& existing : root.slots)
        if (existing.occupied && existing.info.subjectId == info.subjectId)
            throw OmniUsdLiveResult(OMNI_USD_LIVE_ALREADY_EXISTS);
    slot.info = std::move(info);
    slot.owner = std::move(value);
}
UsdLayerCreation& UsdCreationLease::workflow() const {
    if (slot_ == kRetainedUsdCreationCapacity) throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
    // Exclusive lease protects owner; enumeration reads only cached info.
    auto& owner = registry().slots[slot_].owner;
    if (!owner) throw OmniUsdLiveResult(OMNI_USD_LIVE_NOT_FOUND);
    return *owner;
}
std::shared_ptr<UsdLayerCreation> UsdCreationLease::take() {
    workflow().requireTransferHost();
    std::shared_ptr<UsdLayerCreation> result;
    {
        auto& root = registry();
        std::lock_guard<std::mutex> lock(root.mutex);
        result = std::move(root.slots[slot_].owner);
    }
    reset();
    return result;
}
OmniUsdLiveResult UsdCreationLease::close() noexcept {
    try {
        auto& owner = workflow();
        owner.requireTransferHost();
        const auto result = owner.close();
        if (result != OMNI_USD_LIVE_OK) return result;
        std::shared_ptr<UsdLayerCreation> retired;
        {
            auto& root = registry();
            std::lock_guard<std::mutex> lock(root.mutex);
            retired = std::move(root.slots[slot_].owner);
        }
        // Keep the slot occupied/BUSY through final USD/callback destruction.
        // Reentrant discovery is safe; reentrant acquisition returns BUSY.
        retired.reset();
        reset();
        return OMNI_USD_LIVE_OK;
    } catch (OmniUsdLiveResult result) { return result; } catch (const std::bad_alloc&) {
        return OMNI_USD_LIVE_RESOURCE_EXHAUSTED;
    } catch (...) { return OMNI_USD_LIVE_INTERNAL_ERROR; }
}
void UsdCreationLease::setAfterAdoptionForTest(void (*hook)(void*) noexcept, void* context) noexcept {
    if (!hook) afterAdoption.store(nullptr, std::memory_order_release);
    afterAdoptionContext.store(context, std::memory_order_release);
    if (hook) afterAdoption.store(hook, std::memory_order_release);
}
OmniUsdLiveResult UsdCreationLease::retireAdopted() noexcept {
    try {
        auto& owner = workflow();
        owner.requireTransferHost();
        // The last reference to an unreferenced destination starts its close,
        // so it drops only after the slot is free: a CLOSED observer never
        // finds the slot listed. Declared first, it is also released last when
        // a later step throws.
        auto layer = owner.layer();
        const auto result = owner.adoptIntoLayer();
        if (result != OMNI_USD_LIVE_OK) return result;
        if (const auto hook = afterAdoption.load(std::memory_order_acquire))
            hook(afterAdoptionContext.load(std::memory_order_acquire));
        std::shared_ptr<UsdLayerCreation> retired;
        {
            auto& root = registry();
            std::lock_guard<std::mutex> lock(root.mutex);
            retired = std::move(root.slots[slot_].owner);
        }
        // As in close(): the slot stays occupied/BUSY through destruction.
        retired.reset();
        reset();
        layer.Reset(); // Outside the registry lock and every workflow call.
        return OMNI_USD_LIVE_OK;
    } catch (OmniUsdLiveResult result) { return result; } catch (const std::bad_alloc&) {
        return OMNI_USD_LIVE_RESOURCE_EXHAUSTED;
    } catch (...) { return OMNI_USD_LIVE_INTERNAL_ERROR; }
}
bool nextRetainedUsdCreation(uint64_t afterSubject, RetainedUsdCreationInfo& output) {
    RetainedUsdCreationInfo value;
    {
        auto& root = registry();
        std::lock_guard<std::mutex> lock(root.mutex);
        const Slot* next = nullptr;
        for (const auto& slot : root.slots)
            if (slot.occupied && slot.info.subjectId > afterSubject &&
                (!next || slot.info.subjectId < next->info.subjectId))
                next = &slot;
        if (!next) return false;
        value = next->info;
    }
    output = std::move(value);
    return true;
}
} // namespace omni::usd::live
