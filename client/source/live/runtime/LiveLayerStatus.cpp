// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "LiveLayerStatus.h"

namespace omni::usd::live {
namespace {
bool terminal(const OmniUsdLiveStatus& status) noexcept {
    return status.state == OMNI_USD_LIVE_STATE_FAILED ||
        (status.result != OMNI_USD_LIVE_OK && status.recovery_action == OMNI_USD_LIVE_RECOVERY_PRESERVE_AND_REOPEN);
}
bool closing(const OmniUsdLiveStatus& status) noexcept {
    return status.state == OMNI_USD_LIVE_STATE_CLOSING || status.state == OMNI_USD_LIVE_STATE_CLOSED;
}
}
LayerStatus::LayerStatus(std::shared_ptr<Runtime> runtime, std::shared_ptr<LayerAdmission> admission, bool creation)
    : runtime_(std::move(runtime)), admission_(std::move(admission)), creation_(creation) {
    latest_.struct_size = sizeof(latest_);
    latest_.abi_version = OMNI_USD_LIVE_ABI_VERSION;
    latest_.subject_kind = OMNI_USD_LIVE_SUBJECT_LAYER;
    latest_.state = OMNI_USD_LIVE_STATE_OPENING;
    latest_.operation = creation ? OMNI_USD_LIVE_OPERATION_CREATE : OMNI_USD_LIVE_OPERATION_OPEN;
    latest_.progress_known = !creation;
}
std::shared_ptr<LayerStatus> LayerStatus::create(std::shared_ptr<Runtime> runtime,
    std::shared_ptr<LayerAdmission> admission, bool creation) {
    if (!runtime) throw OmniUsdLiveResult(OMNI_USD_LIVE_NOT_INITIALIZED);
    if (!admission) throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
    if (runtime->closing) throw OmniUsdLiveResult(OMNI_USD_LIVE_BUSY);
    runtime->layerAdmissions->validate(admission, admission->endpoint(), admission->session(), admission->layer());
    auto owner = std::shared_ptr<LayerStatus>(new LayerStatus(std::move(runtime), std::move(admission), creation));
    owner->subject_ = owner->runtime_->statuses->add(owner->admission_->endpoint(), 0,
        owner->admission_->canonicalUri(), owner->latest_);
    (void)owner->runtime_->statuses->snapshot(owner->subject_, &owner->latest_);
    return owner;
}
LayerStatus::~LayerStatus() {
    // Keep Runtime and canonical admission alive through removal, and release
    // their strong owners only after this body (outside every status mutex).
    if (subject_) runtime_->statuses->remove(subject_);
}
void LayerStatus::validate(const std::shared_ptr<Runtime>& runtime,
    const std::shared_ptr<LayerAdmission>& admission) const {
    if (runtime.get() != runtime_.get() || runtime.owner_before(runtime_) || runtime_.owner_before(runtime) ||
        admission.get() != admission_.get() || admission.owner_before(admission_) || admission_.owner_before(admission))
        throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
}
OmniUsdLiveResult LayerStatus::publishLocked(const OmniUsdLiveStatus& value, bool advance) noexcept {
    const auto result = runtime_->statuses->publish(subject_, value, advance);
    if (result != OMNI_USD_LIVE_OK) return result;
    // Cache the actual latched cause/counts, never an input that the hub may
    // have rejected as stale. Hub locks never invoke user callbacks.
    return runtime_->statuses->snapshot(subject_, &latest_);
}
OmniUsdLiveResult LayerStatus::publishCreation(const OmniUsdLiveStatus& value, bool confirmed) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (handedOff_) return OMNI_USD_LIVE_OK;
    if (!creation_ || value.state == OMNI_USD_LIVE_STATE_READY ||
        (confirmed && (value.result != OMNI_USD_LIVE_OK || terminal(value) || closing(value))))
        return OMNI_USD_LIVE_INVALID_ARGUMENT;
    if (!terminal(latest_) && terminal(value) && latest_.has_unconfirmed_edits && !value.pending_write_count &&
        !value.has_unconfirmed_edits) {
        // A definite finite rejection/unissued result settles the one create
        // obligation, without asserting any actor/read progress. Clear that
        // ownership before the generic status hub latches its terminal cause.
        auto settled = value;
        settled.state = OMNI_USD_LIVE_STATE_OPENING;
        settled.result = OMNI_USD_LIVE_OK;
        settled.recovery_action = OMNI_USD_LIVE_RECOVERY_NONE;
        settled.grpc_status_present = 0;
        settled.grpc_status = 0;
        const auto result = publishLocked(settled);
        if (result != OMNI_USD_LIVE_OK) return result;
    }
    auto candidate = value;
    if (abandonedOpening_ != OMNI_USD_LIVE_OK && !terminal(value) && !closing(value)) {
        if (confirmed) {
            // Settle finite ownership BEFORE latching terminal opening failure.
            // Otherwise the hub must conservatively retain the previous edit
            // flag even though the confirmed create has no pending obligation.
            const auto settled = publishLocked(value);
            if (settled != OMNI_USD_LIVE_OK) return settled;
            candidate.state = OMNI_USD_LIVE_STATE_FAILED;
            candidate.operation = OMNI_USD_LIVE_OPERATION_OPEN;
            candidate.recovery_action = OMNI_USD_LIVE_RECOVERY_PRESERVE_AND_REOPEN;
        }
        candidate.result = abandonedOpening_;
        candidate.grpc_status_present = 0;
        candidate.grpc_status = 0;
    } else if (terminal(value)) abandonedOpening_ = OMNI_USD_LIVE_OK;
    const auto result = publishLocked(candidate);
    if (result == OMNI_USD_LIVE_OK && confirmed && !terminal(latest_) && !closing(latest_)) confirmed_ = true;
    return result;
}
OmniUsdLiveResult LayerStatus::abandonOpening(OmniUsdLiveResult result) noexcept {
    if (result == OMNI_USD_LIVE_OK || !definedResult(result) || result == OMNI_USD_LIVE_OUTCOME_UNKNOWN)
        return OMNI_USD_LIVE_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(mutex_);
    if (!creation_ || handedOff_) return OMNI_USD_LIVE_INVALID_ARGUMENT;
    if (terminal(latest_) || closing(latest_)) return OMNI_USD_LIVE_OK;
    if (abandonedOpening_ == OMNI_USD_LIVE_OK) abandonedOpening_ = result;
    auto failure = latest_;
    failure.result = abandonedOpening_;
    failure.grpc_status_present = 0;
    failure.grpc_status = 0;
    if (confirmed_) {
        failure.state = OMNI_USD_LIVE_STATE_FAILED;
        failure.operation = OMNI_USD_LIVE_OPERATION_OPEN;
        failure.recovery_action = OMNI_USD_LIVE_RECOVERY_PRESERVE_AND_REOPEN;
    } // Pending OPENING has no terminal incident/recovery action to latch.
    return publishLocked(failure);
}
OmniUsdLiveResult LayerStatus::handoffResultLocked() const noexcept {
    if (handedOff_) return OMNI_USD_LIVE_ALREADY_EXISTS;
    if (closing(latest_)) return OMNI_USD_LIVE_CANCELLED;
    if (terminal(latest_)) return latest_.result;
    if (creation_ && !confirmed_) return OMNI_USD_LIVE_BUSY;
    return OMNI_USD_LIVE_OK;
}
OmniUsdLiveResult LayerStatus::canHandoff() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return handoffResultLocked();
}
OmniUsdLiveResult LayerStatus::handoff() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto result = handoffResultLocked();
    if (result != OMNI_USD_LIVE_OK) return result;
    handedOff_ = true;
    return OMNI_USD_LIVE_OK;
}
OmniUsdLiveResult LayerStatus::publishActor(const OmniUsdLiveStatus& value, bool advance) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!handedOff_) return OMNI_USD_LIVE_BUSY;
    return publishLocked(value, advance);
}
OmniUsdLiveResult LayerStatus::failOpening(OmniUsdLiveResult result) noexcept {
    if (result == OMNI_USD_LIVE_OK || !definedResult(result)) return OMNI_USD_LIVE_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(mutex_);
    auto failure = latest_;
    failure.state = OMNI_USD_LIVE_STATE_FAILED;
    failure.result = result;
    failure.operation = OMNI_USD_LIVE_OPERATION_OPEN;
    failure.grpc_status_present = 0;
    failure.grpc_status = 0;
    failure.recovery_action = OMNI_USD_LIVE_RECOVERY_PRESERVE_AND_REOPEN;
    return publishLocked(failure);
}
} // namespace omni::usd::live
