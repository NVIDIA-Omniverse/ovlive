// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "LiveRuntime.h"
#include "LiveUri.h"
#include <chrono>
#include <memory>
#include <string>

namespace omni::usd::live {
class Request;
// Logical joins survive reconnect; messages are copied FIFO acceptance only,
// nondurable, and a failed in-flight write may be replayed/duplicated. No scene
// ownership or host pump. User callbacks run serially on operational workers.
class SessionRegistry final {
public:
    using Clock = std::chrono::steady_clock;
    explicit SessionRegistry(Runtime& runtime);
    ~SessionRegistry();
    SessionRegistry(const SessionRegistry&) = delete;
    SessionRegistry& operator=(const SessionRegistry&) = delete;
    // Throws typed result/allocation exceptions before admission; never invokes
    // a callback on failed admission. URI must be a parsed session URI.
    OmniUsdLiveSession join(const LiveUri& uri, std::string app, void* user, OmniUsdLiveSessionCallback callback);
    // Success releases handle only after all CQ work and callback execution is
    // quiescent. Timeout retains ownership; worker-dependent waits return BUSY.
    OmniUsdLiveResult leave(OmniUsdLiveSession session, uint32_t timeoutMs);
    OmniUsdLiveResult status(OmniUsdLiveSession session, OmniUsdLiveStatus* output);
    // Returned finite request reports local queue acceptance. Cancellation only
    // suppresses delivery; it cannot retract accepted message bytes.
    std::shared_ptr<Request> send(OmniUsdLiveSession session, const void* bytes, uint64_t size, void* user,
        OmniUsdLiveResultCallback callback);
    bool hasSessions() const noexcept;
    // Test seam: sessions joined afterwards call hook in send once the message
    // and its request are prepared, before acceptance is committed, with the
    // prepared request handle.
    void setAfterPrepareForTest(void (*hook)(void*, uint64_t) noexcept, void* context) noexcept;
    // Test seam: the session treats its peer reconciliation grace as expired from its next step.
    void expirePeersForTest(OmniUsdLiveSession session) noexcept;
    // Test seam: the session treats its keepalive interval as elapsed at its
    // next idle step on a connection that reported one, sending one keepalive.
    void expireKeepaliveForTest(OmniUsdLiveSession session) noexcept;
    void beginShutdown() noexcept;
    bool waitForShutdown(Clock::time_point deadline);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace omni::usd::live
