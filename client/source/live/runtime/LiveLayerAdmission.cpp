// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "LiveLayerAdmission.h"
#include "LiveUri.h"
#include <condition_variable>
#include <mutex>
#include <vector>

namespace omni::usd::live {
struct LayerAdmission::State {
    struct Slot {
        OmniUsdLiveEndpoint endpoint = 0;
        std::string uri;
        std::weak_ptr<LayerAdmission> owner;
        bool retiring = false, pinned = false;
    };
    std::mutex mutex;
    std::condition_variable released;
    size_t waiting = 0;
    std::vector<Slot> slots;
    explicit State(size_t capacity) : slots(capacity) {}
};
LayerAdmission::LayerAdmission(std::shared_ptr<State> state, EndpointLease endpoint, std::string session,
    std::string layer, std::string uri)
    : state_(std::move(state)), endpoint_(std::move(endpoint)), session_(std::move(session)), layer_(std::move(layer)),
      uri_(std::move(uri)) {}
LayerAdmission::~LayerAdmission() {
    // The weak slot owner has already expired; the endpoint lease retires
    // after this body, outside the registry mutex.
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->released.notify_all();
}
void LayerAdmission::markRetiring() noexcept {
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->slots[slot_].retiring = true;
}
void LayerAdmission::pinToHost(bool pinned) noexcept {
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->slots[slot_].pinned = pinned;
    state_->released.notify_all();
}
LayerAdmissionRegistry::LayerAdmissionRegistry(Runtime& runtime, size_t capacity)
    : runtime_(runtime), state_(std::make_shared<LayerAdmission::State>(capacity)) {}
LayerAdmissionRegistry::~LayerAdmissionRegistry() = default;

std::shared_ptr<LayerAdmission> LayerAdmissionRegistry::reserve(OmniUsdLiveEndpoint endpoint, std::string_view session,
    std::string_view layer, std::optional<std::chrono::steady_clock::time_point> deadline) {
    auto lease = runtime_.acquireEndpoint(endpoint);
    if (!lease) throw OmniUsdLiveResult(runtime_.closing ? OMNI_USD_LIVE_BUSY : OMNI_USD_LIVE_INVALID_HANDLE);
    auto uri = makeLayerUri(makeSessionUri(lease->options.authority.canonical, session), layer);
    // Every fallible allocation precedes the slot publication. Keep the local
    // strong owner outside the lock's lifetime: rollback never destroys an
    // endpoint dependency while holding the admission mutex.
    auto candidate = std::shared_ptr<LayerAdmission>(
        new LayerAdmission(state_, std::move(lease), std::string(session), std::string(layer), uri));
    std::unique_lock<std::mutex> lock(state_->mutex);
    size_t available;
    for (;;) {
        available = state_->slots.size();
        const LayerAdmission::State::Slot* duplicate = nullptr;
        for (size_t i = 0; i < state_->slots.size(); ++i) {
            const auto& slot = state_->slots[i];
            // Do not lock weak owners here: dropping a temporary last strong owner
            // under this mutex could run endpoint teardown under a registry lock.
            if (slot.owner.expired()) {
                if (available == state_->slots.size()) available = i;
            } else if (slot.endpoint == endpoint && slot.uri == uri) {
                duplicate = &slot;
                break;
            }
        }
        if (!duplicate) break;
        if (!deadline || !duplicate->retiring || duplicate->pinned)
            throw OmniUsdLiveResult(OMNI_USD_LIVE_ALREADY_EXISTS);
        ++state_->waiting;
        const auto status = state_->released.wait_until(lock, *deadline);
        --state_->waiting;
        if (status == std::cv_status::timeout) {
            // Recheck once: the release may coincide with the deadline.
            deadline.reset();
        }
    }
    if (available == state_->slots.size()) throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    auto& slot = state_->slots[available];
    slot.uri.swap(uri);
    slot.endpoint = endpoint;
    slot.owner = candidate;
    slot.retiring = slot.pinned = false;
    candidate->slot_ = available;
    return candidate;
}
size_t LayerAdmissionRegistry::waiting() const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->waiting;
}
void LayerAdmissionRegistry::validate(const std::shared_ptr<LayerAdmission>& reservation, OmniUsdLiveEndpoint endpoint,
    std::string_view session, std::string_view layer) const {
    if (!reservation || reservation->state_ != state_ || reservation->endpoint() != endpoint ||
        reservation->session() != session || reservation->layer() != layer)
        throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
}
void LayerAdmissionRegistry::claim(const std::shared_ptr<LayerAdmission>& reservation, OmniUsdLiveEndpoint endpoint,
    std::string_view session, std::string_view layer) const {
    validate(reservation, endpoint, session, layer);
    bool expected = false;
    if (!reservation->claimed_.compare_exchange_strong(expected, true, std::memory_order_acq_rel))
        throw OmniUsdLiveResult(OMNI_USD_LIVE_ALREADY_EXISTS);
}
} // namespace omni::usd::live
