// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "LiveRequest.h"
#include "LiveDiagnostics.h"
#include <cassert>
#include <cstring>
#include <utility>

namespace omni::usd::live {
namespace {
std::atomic<uint64_t> nextRequest{1};
OmniUsdLiveRequest allocateHandle() {
    uint64_t ordinal = nextRequest.load();
    do {
        if (ordinal >= (UINT64_C(1) << 56)) throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    } while (!nextRequest.compare_exchange_weak(ordinal, ordinal + 1));
    return (UINT64_C(3) << 56) | ordinal;
}
} // namespace

Request::Request(RequestRegistry& registry, OmniUsdLiveRequest value, uint32_t action, EndpointLease endpoint,
    std::function<void()> cancelTransport)
    : handle(value), operation(action), registry_(registry), endpoint_(std::move(endpoint)),
      cancelTransport_(std::move(cancelTransport)) {}

void Request::finish(OmniUsdLiveRequestError error, std::function<void()> callback) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (phase_ != Phase::Pending) return; // A duplicate transport completion can never deliver twice.
    error_ = error;
    error_.struct_size = sizeof(error_);
    error_.abi_version = OMNI_USD_LIVE_ABI_VERSION;
    error_.operation = operation;
    error_.request_id = handle;
    // Arbitrary transport/host detail is never copied into the public error
    // record. A later explicitly authorized sanitizer may add bounded detail.
    std::memset(error_.detail, 0, sizeof(error_.detail));
    if (!callback) {
        phase_ = Phase::Done;
        changed_.notify_all();
    } else {
        callback_ = std::move(callback);
        // Cancelled callbacks still retire their captures on the executor, not
        // on the CQ or cancelling thread. Quiescence includes this cleanup.
        phase_ = cancelled_ ? Phase::Retiring : Phase::Queued;
        registry_.executor().enqueue(*this);
    }
}

OmniUsdLiveResult Request::wait(uint32_t timeoutMs) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (phase_ == Phase::Done) return OMNI_USD_LIVE_OK;
    // Blocking a callback worker on another unfinished callback can exhaust the
    // pool (including mutual waits). Conservatively reject that dependency.
    if (DispatchExecutor::isWorker()) return OMNI_USD_LIVE_BUSY;
    return changed_.wait_for(lock, std::chrono::milliseconds(timeoutMs), [&] { return phase_ == Phase::Done; }) ?
        OMNI_USD_LIVE_OK :
        OMNI_USD_LIVE_TIMEOUT;
}

OmniUsdLiveResult Request::cancel(uint32_t timeoutMs) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
    bool cancelTransport = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!cancelled_) {
            cancelled_ = true;
            cancelTransport = phase_ == Phase::Pending;
        }
        if (phase_ == Phase::Queued) phase_ = Phase::Retiring; // Atomically suppress invocation, retain cleanup work.
    }
    // Internal cancellation is nonblocking and never runs under request locks.
    if (cancelTransport && cancelTransport_) cancelTransport_();
    std::unique_lock<std::mutex> lock(mutex_);
    if (phase_ == Phase::Done) return OMNI_USD_LIVE_OK;
    if (DispatchExecutor::isWorker()) return OMNI_USD_LIVE_BUSY;
    return changed_.wait_until(lock, deadline, [&] { return phase_ == Phase::Done; }) ? OMNI_USD_LIVE_OK :
                                                                                        OMNI_USD_LIVE_TIMEOUT;
}

OmniUsdLiveResult Request::result(OmniUsdLiveResult* output) {
    if (!output) return OMNI_USD_LIVE_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(mutex_);
    if (phase_ != Phase::Done) return OMNI_USD_LIVE_BUSY;
    *output = error_.result;
    return OMNI_USD_LIVE_OK;
}

OmniUsdLiveResult Request::error(OmniUsdLiveRequestError* output) {
    uint32_t size = 0;
    if (output) std::memcpy(&size, output, sizeof(size));
    if (!output || size < sizeof(*output)) return OMNI_USD_LIVE_INVALID_ARGUMENT;
    if ((output->abi_version >> 16) != (OMNI_USD_LIVE_ABI_VERSION >> 16)) return OMNI_USD_LIVE_UNSUPPORTED;
    std::lock_guard<std::mutex> lock(mutex_);
    if (phase_ != Phase::Done) return OMNI_USD_LIVE_BUSY;
    auto snapshot = error_;
    snapshot.struct_size = size;
    std::memcpy(output, &snapshot, sizeof(snapshot));
    return OMNI_USD_LIVE_OK;
}

bool Request::callbackThrew() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return callbackThrew_;
}

bool Request::issueIfNotCancelled(const std::function<void()>& action) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (cancelled_ || phase_ != Phase::Pending) return false;
    action();
    return true;
}

void Request::dispatch() noexcept {
    std::function<void()> callback;
    bool invoke = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (phase_ != Phase::Queued && phase_ != Phase::Retiring) return;
        invoke = phase_ == Phase::Queued;
        phase_ = invoke ? Phase::Running : Phase::Retiring;
        callback = std::move(callback_);
    }
    bool threw = false;
    {
        try {
            if (invoke) callback();
        } catch (...) {
            // Preserve the actual operation outcome (especially committed writes).
            // Callback exceptions are a separate diagnostic, never an RPC failure.
            threw = true;
        }
        callback = {};
    }
    if (threw)
        registry_.runtime_.diagnostics->emit(OMNI_USD_LIVE_RECORD_CALLBACK_FAILED, endpoint_->handle, 0,
            error_.incident_id,
            {{DiagnosticKey::CallbackKind, static_cast<uint32_t>(CallbackKind::Request)},
                {DiagnosticKey::Request, handle}, {DiagnosticKey::Operation, operation},
                {DiagnosticKey::Attempt, error_.attempt_id}});
    {
        std::lock_guard<std::mutex> lock(mutex_);
        callbackThrew_ = threw;
        phase_ = Phase::Done;
        changed_.notify_all();
    }
}

void Request::releaseEndpoint() noexcept {
    // Done has been published: no worker or transport path reads the lease.
    EndpointLease retired = std::move(endpoint_);
}

RequestRegistry::RequestRegistry(Runtime& runtime, uint32_t capacity, uint32_t count)
    : runtime_(runtime), capacity_(capacity), executor_(count) {}

RequestRegistry::~RequestRegistry() = default;

std::shared_ptr<Request> RequestRegistry::create(OmniUsdLiveEndpoint endpoint, uint32_t operation,
    std::function<void()> cancelTransport, bool timing) {
    if (timing && operation != OMNI_USD_LIVE_OPERATION_PING)
        throw std::invalid_argument("auxiliary request must be a timing probe");
    auto lease = runtime_.acquireEndpoint(endpoint);
    if (!lease) throw OmniUsdLiveResult(runtime_.closing ? OMNI_USD_LIVE_BUSY : OMNI_USD_LIVE_INVALID_HANDLE);
    auto request = std::make_shared<Request>(*this, allocateHandle(), operation, std::move(lease),
        std::move(cancelTransport));
    std::lock_guard<std::mutex> lock(registryMutex_);
    if (timing ? timing_ != 0 : requests_.size() - size_t(timing_ != 0) >= capacity_) {
        runtime_.diagnostics->admittedRequestSlot(timing, false);
        throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    }
    requests_.emplace(request->handle, request);
    if (timing) timing_ = request->handle;
    runtime_.diagnostics->admittedRequestSlot(timing, true);
    return request;
}

std::shared_ptr<Request> RequestRegistry::find(OmniUsdLiveRequest handle) {
    std::lock_guard<std::mutex> lock(registryMutex_);
    const auto found = requests_.find(handle);
    return found == requests_.end() ? nullptr : found->second;
}

OmniUsdLiveResult RequestRegistry::release(OmniUsdLiveRequest handle) {
    auto request = find(handle);
    if (!request) return OMNI_USD_LIVE_INVALID_HANDLE;
    auto result = request->cancel(0);
    if (result != OMNI_USD_LIVE_OK)
        return result == OMNI_USD_LIVE_TIMEOUT ? OmniUsdLiveResult(OMNI_USD_LIVE_BUSY) : result;
    {
        std::lock_guard<std::mutex> lock(registryMutex_);
        const bool timing = timing_ == handle;
        if (!requests_.erase(handle)) return OMNI_USD_LIVE_INVALID_HANDLE;
        if (timing) timing_ = 0;
        runtime_.diagnostics->releasedRequestSlot(timing);
    }
    // Exactly one release erases the handle. Its dependency ends here, not
    // when the last worker or transport reference drops the object.
    request->releaseEndpoint();
    return OMNI_USD_LIVE_OK;
}

void RequestRegistry::beginShutdown() { executor_.beginShutdown(); }

bool RequestRegistry::waitForShutdown(std::chrono::steady_clock::time_point deadline) {
    return executor_.waitForShutdown(deadline);
}
} // namespace omni::usd::live
