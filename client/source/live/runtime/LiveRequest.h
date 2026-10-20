// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "LiveRuntime.h"
#include "LiveDispatch.h"
#include <chrono>
#include <functional>
#include <vector>

namespace omni::usd::live {
class RequestRegistry;

// A finite operation depends on its endpoint until its handle is released,
// which requires Done. Release drops the endpoint lease synchronously, so a
// released request is no endpoint dependent even while a worker or transport
// still retains the object. Transport owners must call finish exactly once,
// including after cancellation, and retain the Request until then.
class Request final : public DispatchJob {
public:
    const OmniUsdLiveRequest handle;
    const uint32_t operation;
    // cancelTransport is an internal nonblocking, thread-safe cancellation hook.
    // It must not invoke host code. It remains safe after transport completion.
    Request(RequestRegistry& registry, OmniUsdLiveRequest handle, uint32_t operation, EndpointLease endpoint,
        std::function<void()> cancelTransport);
    void finish(OmniUsdLiveRequestError error, std::function<void()> callback) noexcept;
    OmniUsdLiveResult wait(uint32_t timeoutMs);
    OmniUsdLiveResult cancel(uint32_t timeoutMs);
    OmniUsdLiveResult result(OmniUsdLiveResult* output);
    OmniUsdLiveResult error(OmniUsdLiveRequestError* output);
    bool callbackThrew() const;
    // Internal issuance boundary, serialized against cancellation. Action must
    // not invoke host code and must have completed fallible preparation already.
    bool issueIfNotCancelled(const std::function<void()>& action);
private:
    friend class RequestRegistry;
    enum class Phase { Pending, Queued, Running, Retiring, Done };
    RequestRegistry& registry_;
    EndpointLease endpoint_;
    const std::function<void()> cancelTransport_;
    mutable std::mutex mutex_;
    std::condition_variable changed_;
    Phase phase_ = Phase::Pending;
    bool cancelled_ = false;
    bool callbackThrew_ = false;
    OmniUsdLiveRequestError error_{};
    std::function<void()> callback_;
    void dispatch() noexcept override;
    void releaseEndpoint() noexcept; // Registry release only, after Done.
};

class RequestRegistry final {
public:
    RequestRegistry(Runtime& runtime, uint32_t capacity, uint32_t workers);
    ~RequestRegistry();
    std::shared_ptr<Request> create(OmniUsdLiveEndpoint endpoint, uint32_t operation,
        std::function<void()> cancelTransport, bool timing = false);
    std::shared_ptr<Request> find(OmniUsdLiveRequest handle);
    OmniUsdLiveResult release(OmniUsdLiveRequest handle);
    void beginShutdown(); // Runtime only calls this after all endpoint owners are gone.
    bool waitForShutdown(std::chrono::steady_clock::time_point deadline);
    DispatchExecutor& executor() { return executor_; }
private:
    friend class Request;
    Runtime& runtime_;
    const uint32_t capacity_;
    std::mutex registryMutex_;
    std::unordered_map<OmniUsdLiveRequest, std::shared_ptr<Request>> requests_;
    // One fixed auxiliary clock-probe slot cannot consume scene/request quota.
    OmniUsdLiveRequest timing_ = 0;
    DispatchExecutor executor_;
};
} // namespace omni::usd::live
