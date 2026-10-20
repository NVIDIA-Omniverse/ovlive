// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "LiveRuntime.h"
#include "LiveDispatch.h"
#include "LiveDiagnostics.h"
#include <functional>

namespace omni::usd::live {
struct ScheduledStep {
    bool done = false;
    std::chrono::steady_clock::time_point next = std::chrono::steady_clock::time_point::max();
};
class ScheduledOperation {
public:
    virtual ~ScheduledOperation() = default;
    // Internal state transitions only; never invoke host code on the timer thread.
    virtual ScheduledStep advance(std::chrono::steady_clock::time_point now) noexcept = 0;
    uint64_t wakeRevision() const noexcept { return wakeRevision_.load(std::memory_order_acquire); }
private:
    friend class AuthRegistry;
    std::atomic<uint64_t> wakeRevision_{0};
};
class AuthCallback final : public DispatchJob {
public:
    AuthCallback(DispatchExecutor& executor, EndpointLease lease, OmniUsdLiveCallbackHandle handle, void* user,
        OmniUsdLiveAuthRequiredCallback callback);
    void notify(const OmniUsdLiveAuthRequired& event); // Endpoint mutex held.
    OmniUsdLiveResult cancel(uint32_t timeoutMs);
    void detach(); // Caller retains this observer; releases endpoint-list ownership.
private:
    DispatchExecutor& executor_;
    EndpointLease endpoint_;
    const OmniUsdLiveCallbackHandle handle_;
    void* user_;
    OmniUsdLiveAuthRequiredCallback callback_;
    std::mutex mutex_;
    std::condition_variable changed_;
    OmniUsdLiveAuthRequired pending_{};
    uint64_t lastChallenge_ = 0;
    bool queued_ = false, active_ = false, cancelled_ = false;
    void dispatch() noexcept override;
};

class AuthRegistry final {
public:
    explicit AuthRegistry(Runtime& runtime);
    ~AuthRegistry();
    OmniUsdLiveResult subscribe(OmniUsdLiveEndpoint endpoint, void* user, OmniUsdLiveAuthRequiredCallback callback,
        OmniUsdLiveCallbackHandle* output);
    // Private barrier is used only by ownership tests, never by the host ABI.
    OmniUsdLiveResult unsubscribe(OmniUsdLiveCallbackHandle handle, uint32_t timeoutMs,
        const std::function<void()>& beforeDetachForTest = {});
    OmniUsdLiveResult snapshot(OmniUsdLiveEndpoint endpoint, OmniUsdLiveAuthState* state);
    OmniUsdLiveResult replace(OmniUsdLiveEndpoint endpoint, uint64_t generation, const char* token,
        uint64_t expiresAtMs, uint64_t* newGeneration);
    OmniUsdLiveResult fail(OmniUsdLiveEndpoint endpoint, uint64_t challenge);
    // Internal completion evidence; only the captured generation is affected.
    void completed(Endpoint& endpoint, uint64_t generation, grpc::StatusCode code) noexcept;
    // Private supplied-clock completion seam; no host ABI entry point.
    void completedAt(Endpoint& endpoint, uint64_t generation, grpc::StatusCode code, uint64_t nowMs,
        std::chrono::steady_clock::time_point now) noexcept;
    // Endpoint mutex held. The 2-argument form samples the steady clock.
    bool usable(const Endpoint& endpoint, uint64_t nowMs) const noexcept;
    bool usable(const Endpoint& endpoint, uint64_t nowMs, std::chrono::steady_clock::time_point now) const noexcept;
    // Clock seam for deterministic expiry/deadline cases; endpoint mutex held.
    void evaluate(Endpoint& endpoint, uint64_t nowMs, std::chrono::steady_clock::time_point now);
    // Terminal controller failure, including internal timer errors. Endpoint
    // mutex held; latch once and notify all parked operations independently of logs.
    void failClosed(Endpoint& endpoint, AuthFailureCause cause, std::chrono::steady_clock::time_point now);
    // Absolute timer deadline from the same clock sample as evaluate; endpoint
    // mutex held. Later endpoints/operations must not extend this deadline.
    std::chrono::steady_clock::time_point nextEvaluation(const Endpoint& endpoint, uint64_t nowMs,
        std::chrono::steady_clock::time_point now) const noexcept;
    // Final endpoint removal only; endpoint mutex held. No host callbacks.
    void closed(Endpoint& endpoint, std::chrono::steady_clock::time_point now) noexcept;
    void wake();
    void reserve(std::shared_ptr<ScheduledOperation> operation);
    void activate(ScheduledOperation& operation);
    void wake(ScheduledOperation& operation);
    // Publish a worker-selected deadline without manufacturing an external wake.
    void schedule(ScheduledOperation& operation, std::chrono::steady_clock::time_point next, uint64_t observedWake);
    void discard(ScheduledOperation& operation); // Unpublished admission rollback only.
    void beginShutdown();
    bool waitForShutdown(std::chrono::steady_clock::time_point deadline);
private:
    // Milliseconds of token lifetime left by the earlier of the wall-clock expiry
    // and its steady-clock measure; UINT64_MAX without an expiry, 0 once expired.
    uint64_t remainingMs(const Endpoint& endpoint, uint64_t nowMs,
        std::chrono::steady_clock::time_point now) const noexcept;
    Runtime& runtime_;
    std::mutex mutex_;
    std::unordered_map<OmniUsdLiveCallbackHandle, std::shared_ptr<AuthCallback>> callbacks_;
    std::mutex timerMutex_;
    std::condition_variable timerChanged_, timerStopped_;
    bool stopping_ = false, stopped_ = false;
    uint64_t wakeRevision_ = 0;
    struct Entry {
        std::shared_ptr<ScheduledOperation> operation;
        std::chrono::steady_clock::time_point next = std::chrono::steady_clock::time_point::max();
        uint64_t revision = 0;
        bool active = false;
    };
    std::unordered_map<ScheduledOperation*, Entry> operations_;
    std::thread timer_;
    void challenge(Endpoint& endpoint, uint32_t reason, bool rejected);
    void publishBlocked(Endpoint& endpoint, uint64_t nowMs);
    void finishWait(Endpoint& endpoint, AuthWaitResolution resolution,
        std::chrono::steady_clock::time_point now) noexcept;
    void recordFailure(Endpoint& endpoint, AuthFailureCause cause, std::chrono::steady_clock::time_point now) noexcept;
    void completedImpl(Endpoint& endpoint, uint64_t generation, grpc::StatusCode code, bool suppliedClock,
        uint64_t nowMs, std::chrono::steady_clock::time_point now) noexcept;
    void work() noexcept;
};
} // namespace omni::usd::live
