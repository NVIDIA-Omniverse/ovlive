// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "HostLayerState.h"
#include <cassert>

namespace omni::usd::live {
HostLayerState::HostLayerState(WireScenePtr initial) : HostLayerState(std::move(initial), Limits{}) {}

HostLayerState::HostLayerState(WireScenePtr initial, Limits limits)
    : limits_(limits), fault_(std::make_shared<HostFailureLatch>()),
      authorized_(PXR_NS::TfCreateRefPtr(new HostSceneProjection(std::move(initial), fault_))), observed_(authorized_) {
}

HostLayerState::HostLayerState(PXR_NS::TfRefPtr<HostSceneProjection> initial) noexcept
    : limits_(), fault_(initial->fault()), authorized_(std::move(initial)), observed_(authorized_) {
    assert(authorized_ && authorized_->baseline() && fault_);
    observe(observed_);
}

PXR_NS::TfRefPtr<HostSceneProjection> HostLayerState::prepare(WireScenePtr scene) const {
    // Every projection of this layer shares its host edit buffer quota.
    return PXR_NS::TfCreateRefPtr(new HostSceneProjection(std::move(scene), fault_, authorized_->limits()));
}

void HostLayerState::fail(Failure reason) noexcept {
    if (failure_ == Failure::None) failure_ = reason;
    fault_->fail(reason == Failure::PublicationFailed ? OMNI_USD_LIVE_INTERNAL_ERROR :
            reason == Failure::GenerationExhausted    ? OMNI_USD_LIVE_RESOURCE_EXHAUSTED :
                                                        OMNI_USD_LIVE_UNSUPPORTED);
}

bool HostLayerState::observe(PXR_NS::SdfAbstractDataConstRefPtr current) noexcept {
    // Finish the transition before releasing possibly foreign data. Its
    // destructor may reenter the serial host; do not overwrite that later
    // observation or return success based on pre-retirement state.
    if (current != observed_ && replacementGeneration_ < limits_.generation) ++replacementGeneration_;
    auto retired = std::move(observed_);
    observed_ = current;
    // Installation provenance survives a mutation raised before the first
    // post-install observation. Never adopt a different replacement.
    if (candidate_ && current == candidate_) candidateObserved_ = true;
    if (fault_->result().load(std::memory_order_acquire) != OMNI_USD_LIVE_OK) fail(Failure::HostMutation);
    if (generationExhausted()) fail(Failure::GenerationExhausted);
    if (failure_ == Failure::None) {
        if ((!candidate_ || current != candidate_) && (current != authorized_ || candidateObserved_))
            fail(Failure::BackingReplaced);
    }
    retired.Reset();
    return failure_ == Failure::None;
}

HostLayerState::Admission HostLayerState::begin(const PXR_NS::SdfAbstractDataConstRefPtr& current,
    const PXR_NS::TfRefPtr<HostSceneProjection>& candidate) noexcept {
    if (!observe(current)) return Admission::Faulted;
    if (candidate_ || publicationActive_) return Admission::Busy;
    if (!candidate || candidate == authorized_ || candidate->fault() != fault_) return Admission::InvalidCandidate;
    candidate_ = candidate;
    candidateObserved_ = false;
    publicationActive_ = true;
    return Admission::Accepted;
}

HostLayerState::Admission HostLayerState::beginInPlace(const PXR_NS::SdfAbstractDataConstRefPtr& current) noexcept {
    if (!observe(current)) return Admission::Faulted;
    if (candidate_ || publicationActive_) return Admission::Busy;
    if (current != authorized_) return Admission::InvalidCandidate;
    publicationActive_ = true;
    return Admission::Accepted;
}

bool HostLayerState::finishInPlace(const PXR_NS::SdfAbstractDataConstRefPtr& current) noexcept {
    if (!publicationActive_ || candidate_) return false;
    publicationActive_ = false;
    // A backing other than the authorized projection fails observation.
    if (!observe(current)) {
        publicationFailed_ = true;
        return false;
    }
    return true;
}

OmniUsdLiveResult HostLayerState::cause() const noexcept {
    const auto result = fault_->result().load(std::memory_order_acquire);
    return result != OMNI_USD_LIVE_OK ? result : OmniUsdLiveResult(OMNI_USD_LIVE_UNSUPPORTED);
}

OmniUsdLiveResult HostLayerState::admissionResult(Admission admission) const noexcept {
    switch (admission) {
    case Admission::Accepted: return OMNI_USD_LIVE_OK;
    case Admission::Busy: return OMNI_USD_LIVE_BUSY;
    case Admission::InvalidCandidate: return OMNI_USD_LIVE_INTERNAL_ERROR;
    case Admission::Faulted: break;
    }
    return cause();
}

bool HostLayerState::finish(const PXR_NS::SdfAbstractDataConstRefPtr& current) noexcept {
    if (!publicationActive_) return false;
    if (!observe(current)) {
        publicationFailed_ = true;
        publicationActive_ = false;
        return false;
    }
    if (current != candidate_) {
        abort();
        return false;
    }
    auto retired = std::move(authorized_);
    authorized_ = std::move(candidate_);
    candidateObserved_ = false;
    publicationActive_ = false;
    retired.Reset();
    return failure_ == Failure::None;
}

bool HostLayerState::abort() noexcept {
    if (!publicationActive_) return false;
    publicationActive_ = false;
    publicationFailed_ = true;
    if (fault_->result().load(std::memory_order_acquire) != OMNI_USD_LIVE_OK) fail(Failure::HostMutation);
    fail(Failure::PublicationFailed);
    return true;
}

bool HostLayerState::generationExhausted() const noexcept { return replacementGeneration_ == limits_.generation; }

} // namespace omni::usd::live
