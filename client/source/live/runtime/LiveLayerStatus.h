// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "LiveLayerAdmission.h"
#include "LiveStatus.h"

namespace omni::usd::live {
// One canonical subject across finite creation, bootstrap and attachment. No
// registry/admission backreference owns this object. The last owner removes
// the subject, after which ordinary status-observer CLOSED retention applies.
class LayerStatus final {
public:
    static std::shared_ptr<LayerStatus> create(std::shared_ptr<Runtime>, std::shared_ptr<LayerAdmission>,
        bool creation = false);
    ~LayerStatus();
    LayerStatus(const LayerStatus&) = delete;
    LayerStatus& operator=(const LayerStatus&) = delete;
    uint64_t id() const noexcept { return subject_; }
    // Throws INVALID_ARGUMENT for a different runtime or reservation owner.
    void validate(const std::shared_ptr<Runtime>&, const std::shared_ptr<LayerAdmission>&) const;
    // Creation cannot publish READY. After handoff all creation-side updates,
    // including close/destruction, become successful no-ops.
    OmniUsdLiveResult publishCreation(const OmniUsdLiveStatus&, bool confirmed = false) noexcept;
    // One-way, once-only publication authority. Creation must be confirmed;
    // ordinary actor subjects need no receipt. Terminal/closing cannot hand off.
    OmniUsdLiveResult canHandoff() const noexcept;
    OmniUsdLiveResult handoff() noexcept;
    // `advance` publishes a new revision even when no field changed.
    OmniUsdLiveResult publishActor(const OmniUsdLiveStatus&, bool advance = false) noexcept;
    // Latch local bootstrap failure without losing the latest actor counts or
    // closing the subject. Works on either side of the authority handoff.
    OmniUsdLiveResult failOpening(OmniUsdLiveResult) noexcept;
    // A synchronous creation hook abandoned bootstrap while its finite create
    // may still settle. Keep pending status nonterminal; a late confirmation
    // becomes local OPEN failure, while rejection/uncertainty keeps its actual
    // CREATE result and transport evidence. Never cancels/retries the request.
    OmniUsdLiveResult abandonOpening(OmniUsdLiveResult) noexcept;
private:
    LayerStatus(std::shared_ptr<Runtime>, std::shared_ptr<LayerAdmission>, bool);
    OmniUsdLiveResult publishLocked(const OmniUsdLiveStatus&, bool advance = false) noexcept;
    OmniUsdLiveResult handoffResultLocked() const noexcept;
    const std::shared_ptr<Runtime> runtime_;
    const std::shared_ptr<LayerAdmission> admission_;
    const bool creation_;
    uint64_t subject_ = 0;
    mutable std::mutex mutex_;
    OmniUsdLiveStatus latest_{};
    bool confirmed_ = false, handedOff_ = false;
    OmniUsdLiveResult abandonedOpening_ = OMNI_USD_LIVE_OK;
};
} // namespace omni::usd::live
