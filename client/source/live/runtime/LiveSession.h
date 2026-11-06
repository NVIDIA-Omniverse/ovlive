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
    // a callback on failed admission. URI must be a parsed session URI; flags
    // holds only OMNI_USD_LIVE_JOIN_* bits.
    OmniUsdLiveSession join(const LiveUri& uri, std::string app, void* user, OmniUsdLiveSessionCallback callback,
        uint32_t flags = 0);
    // Success releases handle only after all CQ work and callback execution is
    // quiescent. Timeout retains ownership; worker-dependent waits return BUSY.
    OmniUsdLiveResult leave(OmniUsdLiveSession session, uint32_t timeoutMs);
    OmniUsdLiveResult status(OmniUsdLiveSession session, OmniUsdLiveStatus* output);
    // Returned finite request reports local queue acceptance. Cancellation only
    // suppresses delivery; it cannot retract accepted message bytes.
    std::shared_ptr<Request> send(OmniUsdLiveSession session, const void* bytes, uint64_t size, void* user,
        OmniUsdLiveResultCallback callback);
    // Issues a merge-owner control frame on the join's current connection.
    // Throws a typed result before admission (UNSUPPORTED, UNAVAILABLE, ...)
    // and never invokes the callback then; otherwise the callback runs exactly
    // once unless the request is cancelled.
    std::shared_ptr<Request> requestMergeOwner(OmniUsdLiveSession session, uint32_t action, void* user,
        OmniUsdLiveMergeOwnerCallback callback);
    OmniUsdLiveResult mergeOwner(OmniUsdLiveSession session, OmniUsdLiveMergeOwner* output);
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
    // Test seam: at the session's next step with a join attempt, that attempt's
    // first-response deadline counts as passed if it is a reconnect attempt
    // still awaiting its Connected; otherwise the step discards the request.
    void expireAttemptForTest(OmniUsdLiveSession session) noexcept;
    void beginShutdown() noexcept;
    bool waitForShutdown(Clock::time_point deadline);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace omni::usd::live
