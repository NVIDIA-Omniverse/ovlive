// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "LiveDrain.h"
#include <algorithm>
#include <condition_variable>
#include <limits>

namespace omni::usd::live {
struct DrainSignal {
    std::mutex mutex;
    std::condition_variable changed;
    uint64_t revision = 0;
    void notify() noexcept {
        std::lock_guard<std::mutex> lock(mutex);
        ++revision;
        changed.notify_all();
    }
};

DrainState::DrainState(std::shared_ptr<DrainSignal> signal) : signal_(std::move(signal)) {}

std::shared_ptr<DrainTicket> DrainState::prepare() {
    return std::shared_ptr<DrainTicket>(new DrainTicket(shared_from_this()));
}

DrainState::Fence DrainState::capture() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return {lastOrdinal_, result_};
}

OmniUsdLiveResult DrainState::poll(const Fence& fence) const { return probe(fence).result; }

DrainState::Probe DrainState::probe(const Fence& fence) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (fence.cutoff > lastOrdinal_) return {OMNI_USD_LIVE_INVALID_ARGUMENT, false};
    if (fence.baselineResult != OMNI_USD_LIVE_OK)
        return {result_ == OMNI_USD_LIVE_OUTCOME_UNKNOWN ? result_ : fence.baselineResult, false};
    // Failure is attached to the obligations that existed when it happened,
    // not merely to the current intrusive list. Cleanup must not erase it.
    if (failedFrom_ && failedFrom_ <= fence.cutoff) return {result_, false};
    for (auto* ticket = first_; ticket; ticket = ticket->next_)
        if (ticket->ordinal_ <= fence.cutoff) return {OMNI_USD_LIVE_BUSY, true};
    return {OMNI_USD_LIVE_OK, false};
}

bool DrainState::fail(OmniUsdLiveResult result, bool admitting) noexcept {
    if (result == OMNI_USD_LIVE_OK) return false;
    bool first;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        first = result_ == OMNI_USD_LIVE_OK;
        admitting_ = admitting && (result_ == OMNI_USD_LIVE_OK || admitting_);
        if (result_ == OMNI_USD_LIVE_OK || result == OMNI_USD_LIVE_OUTCOME_UNKNOWN) result_ = result;
        for (auto* ticket = first_; ticket; ticket = ticket->next_)
            if (!failedFrom_ || ticket->ordinal_ < failedFrom_) failedFrom_ = ticket->ordinal_;
    }
    signal_->notify();
    return first;
}

void DrainState::close() noexcept {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
        if (first_) {
            if (result_ == OMNI_USD_LIVE_OK) result_ = OMNI_USD_LIVE_CANCELLED;
            for (auto* ticket = first_; ticket; ticket = ticket->next_)
                if (!failedFrom_ || ticket->ordinal_ < failedFrom_) failedFrom_ = ticket->ordinal_;
        }
    }
    signal_->notify();
}

DrainTicket::DrainTicket(std::shared_ptr<DrainState> state) : state_(std::move(state)) {}
DrainTicket::~DrainTicket() { complete(); }

OmniUsdLiveResult DrainTicket::admit() noexcept {
    {
        std::lock_guard<std::mutex> lock(state_->mutex_);
        if (phase_ != Phase::Prepared) return admissionResult_;
        if ((state_->result_ != OMNI_USD_LIVE_OK && !state_->admitting_) || state_->closed_) {
            phase_ = Phase::Retired;
            admissionResult_ = state_->result_ != OMNI_USD_LIVE_OK ? state_->result_ :
                                                                     OmniUsdLiveResult(OMNI_USD_LIVE_CANCELLED);
            return admissionResult_;
        }
        if (state_->lastOrdinal_ == std::numeric_limits<uint64_t>::max()) {
            if (state_->result_ == OMNI_USD_LIVE_OK) state_->result_ = OMNI_USD_LIVE_RESOURCE_EXHAUSTED;
            state_->admitting_ = false;
            for (auto* ticket = state_->first_; ticket; ticket = ticket->next_)
                if (!state_->failedFrom_ || ticket->ordinal_ < state_->failedFrom_)
                    state_->failedFrom_ = ticket->ordinal_;
            phase_ = Phase::Retired;
            admissionResult_ = OMNI_USD_LIVE_RESOURCE_EXHAUSTED;
        } else {
            ordinal_ = ++state_->lastOrdinal_;
            next_ = state_->first_;
            if (next_) next_->previous_ = this;
            state_->first_ = this;
            phase_ = Phase::Pending;
            admissionResult_ = OMNI_USD_LIVE_OK;
        }
    }
    state_->signal_->notify();
    return admissionResult_;
}

void DrainTicket::complete() noexcept {
    bool changed = false;
    {
        std::lock_guard<std::mutex> lock(state_->mutex_);
        if (phase_ == Phase::Pending) {
            if (previous_) previous_->next_ = next_;
            else state_->first_ = next_;
            if (next_) next_->previous_ = previous_;
            previous_ = next_ = nullptr;
            changed = true;
        }
        phase_ = Phase::Retired;
    }
    if (changed) state_->signal_->notify();
}

DrainRegistry::DrainRegistry(size_t maxStates) : maxStates_(maxStates), signal_(std::make_shared<DrainSignal>()) {}

std::shared_ptr<DrainState> DrainRegistry::makeState() {
    // Keep the candidate owner outside the lock's scope, including exceptions.
    auto state = std::shared_ptr<DrainState>(new DrainState(signal_));
    {
        std::lock_guard<std::mutex> lock(mutex_);
        states_.erase(std::remove_if(states_.begin(), states_.end(), [](const auto& value) { return value.expired(); }),
            states_.end());
        if (states_.size() >= maxStates_) throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
        states_.push_back(state);
    }
    return state;
}

DrainRegistry::Capture DrainRegistry::capture() {
    std::vector<std::shared_ptr<DrainState>> owners;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        states_.erase(std::remove_if(states_.begin(), states_.end(), [](const auto& value) { return value.expired(); }),
            states_.end());
        // Allocate before weak locking. Once strong owners exist, push_back
        // cannot allocate/throw and their destruction occurs after unlock.
        owners.reserve(states_.size());
        for (const auto& weak : states_)
            if (auto state = weak.lock()) owners.push_back(std::move(state));
    }
    Capture result;
    result.targets.reserve(owners.size());
    for (auto& state : owners) {
        const auto fence = state->capture();
        result.targets.push_back({std::move(state), fence});
    }
    return result;
}

OmniUsdLiveResult DrainRegistry::wait(const Capture& captured, Clock::time_point deadline, bool allowBlock) const {
    for (;;) {
        uint64_t observed;
        {
            std::lock_guard<std::mutex> lock(signal_->mutex);
            observed = signal_->revision;
        }
        bool pending = false;
        OmniUsdLiveResult failure = OMNI_USD_LIVE_OK;
        for (const auto& target : captured.targets) {
            if (!target.state || target.state->signal_ != signal_) return OMNI_USD_LIVE_INVALID_ARGUMENT;
            const auto result = target.state->probe(target.fence);
            if (result.pending) pending = true;
            else if (result.result == OMNI_USD_LIVE_OUTCOME_UNKNOWN || failure == OMNI_USD_LIVE_OK)
                failure = result.result;
        }
        if (failure != OMNI_USD_LIVE_OK) return failure;
        if (!pending) return OMNI_USD_LIVE_OK;
        if (!allowBlock) return OMNI_USD_LIVE_BUSY;
        if (Clock::now() >= deadline) return OMNI_USD_LIVE_TIMEOUT;
        std::unique_lock<std::mutex> lock(signal_->mutex);
        // Take the revision before polling, then test it while locked before
        // sleeping. Completion/failure between those points cannot be lost.
        signal_->changed.wait_until(lock, deadline, [&] { return signal_->revision != observed; });
    }
}

OmniUsdLiveResult DrainRegistry::wait(uint32_t timeoutMs, bool allowBlock) {
    const auto deadline = Clock::now() + std::chrono::milliseconds(timeoutMs);
    return wait(capture(), deadline, allowBlock);
}
} // namespace omni::usd::live
