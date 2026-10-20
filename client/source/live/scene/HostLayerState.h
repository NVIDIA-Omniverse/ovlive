// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "HostSceneProjection.h"
#include <limits>

namespace omni::usd::live {
// Private per-layer containment state. All methods require the serial USD host. No explicit callbacks, permission changes, or USD edits.
// Retiring an owner may invoke foreign data/value destructors, after internal
// transitions are consistent; reentrant observations retain permanent faults.
// Observation must run at replacement/notice and API boundaries; pointer
// sampling cannot discover an unobserved swap away and back.
class HostLayerState final {
public:
    enum class Failure { None, HostMutation, BackingReplaced, PublicationFailed, GenerationExhausted };
    enum class Admission { Accepted, Busy, InvalidCandidate, Faulted };
    struct Limits {
        uint64_t generation = std::numeric_limits<uint64_t>::max();
    };
    explicit HostLayerState(WireScenePtr initial);
    HostLayerState(WireScenePtr initial, Limits);
    // Adopt the exact initial projection once, before exposing the layer.
    // Never use this to authorize a later replacement. Existing mutation faults
    // remain latched; adoption does not clone data or invent new provenance.
    // Precondition: a service-baseline projection this format created. This
    // path only retains owners; it cannot allocate or throw.
    explicit HostLayerState(PXR_NS::TfRefPtr<HostSceneProjection> initial) noexcept;
    HostLayerState(const HostLayerState&) = delete;
    HostLayerState& operator=(const HostLayerState&) = delete;

    const PXR_NS::TfRefPtr<HostSceneProjection>& authorized() const noexcept { return authorized_; }
    const PXR_NS::TfRefPtr<HostSceneProjection>& candidate() const noexcept { return candidate_; }
    const PXR_NS::SdfAbstractDataConstRefPtr& observed() const noexcept { return observed_; }
    const std::shared_ptr<HostFailureLatch>& fault() const noexcept { return fault_; }
    const WireScenePtr& provenance() const noexcept {
        return candidateObserved_ && candidate_ ? candidate_->baseline() : authorized_->baseline();
    }
    Failure failure() const noexcept { return failure_; }
    bool publicationFailed() const noexcept { return publicationFailed_; }
    bool installing() const noexcept { return publicationActive_; }
    uint64_t replacementGeneration() const noexcept { return replacementGeneration_; }

    // Eager preparation has no state transition: allocation failure preserves
    // the authorized owner and any installation. begin() checks admission.
    PXR_NS::TfRefPtr<HostSceneProjection> prepare(WireScenePtr) const;
    Admission begin(const PXR_NS::SdfAbstractDataConstRefPtr& current,
        const PXR_NS::TfRefPtr<HostSceneProjection>& candidate) noexcept;
    // An incremental publication into the authorized projection, which stays
    // the backing: Accepted when `current` is the authorized projection and no
    // installation is in progress. installing() holds until finishInPlace or
    // abort. The caller reseats the projection and sends its notices.
    Admission beginInPlace(const PXR_NS::SdfAbstractDataConstRefPtr& current) noexcept;
    // Call after the publication's synchronous notices. Failure, including a
    // backing other than the authorized projection, marks the publication
    // failed.
    bool finishInPlace(const PXR_NS::SdfAbstractDataConstRefPtr& current) noexcept;
    // The first latched host failure cause, shared with the actor's host
    // failure latch; UNSUPPORTED when no cause is latched.
    OmniUsdLiveResult cause() const noexcept;
    // Public result of a begin() outcome. A faulted layer reports cause(); an
    // invalid candidate is an internal invariant violation.
    OmniUsdLiveResult admissionResult(Admission admission) const noexcept;
    // Retains actual current backing even after failure. Owners pin mutable
    // storage, not a frozen copy of its contents; capture after host quiescence.
    // Retention is bounded to authorized, candidate, and latest observed owners.
    bool observe(PXR_NS::SdfAbstractDataConstRefPtr current) noexcept;
    // Call after installation AND synchronous notices. Failure keeps both
    // provenance owners; never adopt an arbitrary current backing as success.
    bool finish(const PXR_NS::SdfAbstractDataConstRefPtr& current) noexcept;
    // Call if the enclosing installation throws, even before replacing USD.
    // Publication failure remains separate from the first host fault reason.
    bool abort() noexcept;
private:
    void fail(Failure reason) noexcept;
    bool generationExhausted() const noexcept;
    const Limits limits_;
    const std::shared_ptr<HostFailureLatch> fault_;
    PXR_NS::TfRefPtr<HostSceneProjection> authorized_, candidate_;
    PXR_NS::SdfAbstractDataConstRefPtr observed_;
    Failure failure_ = Failure::None;
    bool candidateObserved_ = false, publicationFailed_ = false, publicationActive_ = false;
    uint64_t replacementGeneration_ = 0;
};
} // namespace omni::usd::live
