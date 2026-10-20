// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "LiveRuntime.h"
#include <atomic>
#include <chrono>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace omni::usd::live {
class LayerAdmissionRegistry;
// Canonical ownership reservation, before discovery/status/RPC. The token owns
// an endpoint dependency and shared registry state, never Runtime itself. Copies
// share one reservation; the last owner releases admission. Closing or failure
// does not release it while actor or discovery owners still retain it.
class LayerAdmission final {
public:
    // Wakes reserves waiting for this reservation's release.
    ~LayerAdmission();
    // One-way: the owning backing is closing and its owner has let go, so
    // this reservation goes once the backing is quiescent. A reserve with a
    // deadline waits for that release.
    void markRetiring() noexcept;
    // A retiring owner that cannot release before a queued host job retires
    // is pinned; a reserve then refuses at once instead of possibly waiting
    // on its own host. Unpinning wakes waiting reserves.
    void pinToHost(bool pinned) noexcept;
    LayerAdmission(const LayerAdmission&) = delete;
    LayerAdmission& operator=(const LayerAdmission&) = delete;
    OmniUsdLiveEndpoint endpoint() const noexcept { return endpoint_->handle; }
    const std::string& canonicalUri() const noexcept { return uri_; }
    const std::string& session() const noexcept { return session_; }
    const std::string& layer() const noexcept { return layer_; }
private:
    friend class LayerAdmissionRegistry;
    struct State;
    LayerAdmission(std::shared_ptr<State> state, EndpointLease endpoint, std::string session, std::string layer,
        std::string uri);
    std::shared_ptr<State> state_;
    size_t slot_ = 0; // Set before publication; the slot is this owner's until it expires.
    EndpointLease endpoint_;
    const std::string session_, layer_, uri_;
    std::atomic<bool> claimed_{false};
};

class LayerAdmissionRegistry final {
public:
    explicit LayerAdmissionRegistry(Runtime& runtime, size_t capacity);
    ~LayerAdmissionRegistry();
    LayerAdmissionRegistry(const LayerAdmissionRegistry&) = delete;
    LayerAdmissionRegistry& operator=(const LayerAdmissionRegistry&) = delete;
    // Decoded identities are validated and escaped using the real live:// URI
    // builders. Duplicate live ownership => ALREADY_EXISTS; full live capacity
    // => RESOURCE_EXHAUSTED. Allocation/validation failures publish nothing.
    // With a deadline, a duplicate held by a retiring owner is waited for
    // until its release or the deadline, which then returns ALREADY_EXISTS;
    // the caller must be allowed to block. Any other duplicate, including a
    // retiring one pinned to its host, is refused at once.
    std::shared_ptr<LayerAdmission> reserve(OmniUsdLiveEndpoint endpoint, std::string_view session,
        std::string_view layer, std::optional<std::chrono::steady_clock::time_point> deadline = {});
    // Reserves waiting for a retiring owner (test observation).
    size_t waiting() const;
    void validate(const std::shared_ptr<LayerAdmission>& reservation, OmniUsdLiveEndpoint endpoint,
        std::string_view session, std::string_view layer) const;
    // One-shot backing-state attachment. A claimed token may still be shared
    // for discovery ownership, but never bind a second independent
    // actor. Failure after claim retains the reservation until owners drop it.
    void claim(const std::shared_ptr<LayerAdmission>& reservation, OmniUsdLiveEndpoint endpoint,
        std::string_view session, std::string_view layer) const;
private:
    Runtime& runtime_;
    std::shared_ptr<LayerAdmission::State> state_;
};
} // namespace omni::usd::live
