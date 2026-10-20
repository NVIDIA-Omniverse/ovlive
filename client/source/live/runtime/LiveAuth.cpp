// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "LiveAuth.h"
#include "LiveRequest.h"
#include "LiveStatus.h"
#include "LiveDiagnostics.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace omni::usd::live {
namespace {
std::atomic<uint64_t> nextCallback{1}, nextChallenge{1};
uint64_t allocate(std::atomic<uint64_t>& source, uint64_t limit) {
    auto value = source.load();
    do {
        if (value >= limit) throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    } while (!source.compare_exchange_weak(value, value + 1));
    return value;
}
}

AuthCallback::AuthCallback(DispatchExecutor& executor, EndpointLease lease, OmniUsdLiveCallbackHandle handle,
    void* user, OmniUsdLiveAuthRequiredCallback callback)
    : executor_(executor), endpoint_(std::move(lease)), handle_(handle), user_(user), callback_(callback) {}
void AuthCallback::notify(const OmniUsdLiveAuthRequired& event) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (cancelled_ || !event.challenge_id || event.challenge_id == lastChallenge_) return;
    lastChallenge_ = event.challenge_id;
    pending_ = event; // Superseded generations need no refresh from this observer.
    if (!active_ && !queued_) {
        queued_ = true;
        executor_.enqueue(*this);
    }
}
void AuthCallback::dispatch() noexcept {
    OmniUsdLiveAuthRequired event{};
    bool deliver = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        queued_ = false;
        active_ = true;
        event = pending_;
        pending_ = {};
        deliver = !cancelled_;
    }
    if (deliver) {
        std::lock_guard<std::mutex> lock(endpoint_->mutex);
        deliver = !endpoint_->closed && !endpoint_->authFailed &&
            endpoint_->challenge.challenge_id == event.challenge_id;
    }
    if (deliver) {
        try {
            callback_(user_, &event);
        } catch (...) {
            endpoint_->diagnostics.emit(OMNI_USD_LIVE_RECORD_CALLBACK_FAILED, event.endpoint, event.endpoint, 0,
                {{DiagnosticKey::CallbackKind, static_cast<uint32_t>(CallbackKind::Auth)},
                    {DiagnosticKey::Callback, handle_}, {DiagnosticKey::Operation, OMNI_USD_LIVE_OPERATION_AUTH},
                    {DiagnosticKey::AuthGeneration, event.rejected_generation},
                    {DiagnosticKey::Challenge, event.challenge_id}});
        }
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        active_ = false;
        if (!cancelled_ && pending_.challenge_id) {
            queued_ = true;
            executor_.enqueue(*this);
        }
        changed_.notify_all();
    }
}
OmniUsdLiveResult AuthCallback::cancel(uint32_t timeoutMs) {
    std::unique_lock<std::mutex> lock(mutex_);
    cancelled_ = true;
    pending_ = {};
    if (!active_ && !queued_) return OMNI_USD_LIVE_OK;
    if (DispatchExecutor::isWorker()) return OMNI_USD_LIVE_BUSY;
    return changed_.wait_for(lock, std::chrono::milliseconds(timeoutMs), [&] { return !active_ && !queued_; }) ?
        OMNI_USD_LIVE_OK :
        OMNI_USD_LIVE_TIMEOUT;
}
void AuthCallback::detach() {
    std::lock_guard<std::mutex> lock(endpoint_->mutex);
    auto& observers = endpoint_->authObservers;
    observers.erase(std::remove_if(observers.begin(), observers.end(),
                        [this](const auto& observer) { return observer.get() == this; }),
        observers.end());
}

AuthRegistry::AuthRegistry(Runtime& runtime) : runtime_(runtime), timer_([this] { work(); }) {}
AuthRegistry::~AuthRegistry() {
    beginShutdown();
    timer_.join();
}
void AuthRegistry::finishWait(Endpoint& endpoint, AuthWaitResolution resolution,
    std::chrono::steady_clock::time_point now) noexcept {
    if (!endpoint.authWaitActive) return;
    endpoint.authWaitActive = false;
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - endpoint.authWaitStarted).count();
    runtime_.diagnostics->endAuthWait(resolution, static_cast<uint64_t>(std::max<int64_t>(0, elapsed)));
}
void AuthRegistry::closed(Endpoint& endpoint, std::chrono::steady_clock::time_point now) noexcept {
    finishWait(endpoint, AuthWaitResolution::Closed, now);
}
void AuthRegistry::recordFailure(Endpoint& endpoint, AuthFailureCause cause,
    std::chrono::steady_clock::time_point now) noexcept {
    // Called where the failure serial advances, never from repeated evaluation.
    finishWait(endpoint, AuthWaitResolution::Failed, now);
    runtime_.diagnostics->observeAuthFailure(cause);
    runtime_.diagnostics->emit(OMNI_USD_LIVE_RECORD_AUTH_FAILED, endpoint.handle, endpoint.status.subject_id, 0,
        {{DiagnosticKey::Operation, OMNI_USD_LIVE_OPERATION_AUTH},
            {DiagnosticKey::Result, OMNI_USD_LIVE_UNAUTHENTICATED},
            {DiagnosticKey::AuthGeneration, endpoint.tokenGeneration},
            {DiagnosticKey::Challenge, endpoint.challenge.challenge_id},
            {DiagnosticKey::Reason, endpoint.challenge.reason},
            {DiagnosticKey::AuthFailureCause, static_cast<uint32_t>(cause)}});
}
uint64_t AuthRegistry::remainingMs(const Endpoint& endpoint, uint64_t nowMs,
    std::chrono::steady_clock::time_point now) const noexcept {
    if (!endpoint.expiresAtMs) return UINT64_MAX;
    const uint64_t wall = endpoint.expiresAtMs > nowMs ? endpoint.expiresAtMs - nowMs : 0;
    if (endpoint.expiresAtSteady == std::chrono::steady_clock::time_point{}) return wall;
    if (now >= endpoint.expiresAtSteady) return 0;
    const auto steady = std::chrono::duration_cast<std::chrono::milliseconds>(endpoint.expiresAtSteady - now).count();
    return std::min<uint64_t>(wall, static_cast<uint64_t>(steady));
}
bool AuthRegistry::usable(const Endpoint& endpoint, uint64_t nowMs) const noexcept {
    return usable(endpoint, nowMs, std::chrono::steady_clock::now());
}
bool AuthRegistry::usable(const Endpoint& endpoint, uint64_t nowMs,
    std::chrono::steady_clock::time_point now) const noexcept {
    return !endpoint.closed && !endpoint.authFailed && !endpoint.generationRejected && !endpoint.token.empty() &&
        remainingMs(endpoint, nowMs, now) > 0;
}
void AuthRegistry::failClosed(Endpoint& endpoint, AuthFailureCause cause, std::chrono::steady_clock::time_point now) {
    if (endpoint.closed || endpoint.authFailed) return;
    ++endpoint.authFailureSerial;
    endpoint.authFailed = true;
    recordFailure(endpoint, cause, now);
    wake();
}
void AuthRegistry::publishBlocked(Endpoint& endpoint, uint64_t nowMs) {
    auto& status = endpoint.status;
    if (status.state == OMNI_USD_LIVE_STATE_AUTH_WAIT && status.result == OMNI_USD_LIVE_UNAUTHENTICATED &&
        status.grpc_status_present == (endpoint.generationRejected ? 1u : 0u))
        return;
    // Token state is not scene recovery state. Do not clear an existing incident.
    if (status.state == OMNI_USD_LIVE_STATE_FAILED) return;
    status.state = OMNI_USD_LIVE_STATE_AUTH_WAIT;
    status.result = OMNI_USD_LIVE_UNAUTHENTICATED;
    status.operation = OMNI_USD_LIVE_OPERATION_AUTH;
    status.grpc_status_present = endpoint.generationRejected ? 1 : 0;
    status.grpc_status = endpoint.generationRejected ? grpc::StatusCode::UNAUTHENTICATED : 0;
    status.recovery_action = OMNI_USD_LIVE_RECOVERY_REFRESH_AUTH;
    status.changed_at_ms = nowMs;
    ++status.revision;
    runtime_.statuses->publish(status.subject_id, status);
}
void AuthRegistry::challenge(Endpoint& endpoint, uint32_t reason, bool rejected) {
    if (endpoint.closed || endpoint.authFailed || endpoint.challenge.challenge_id) return;
    auto& event = endpoint.challenge;
    event.struct_size = sizeof(event);
    event.abi_version = OMNI_USD_LIVE_ABI_VERSION;
    event.endpoint = endpoint.handle;
    event.rejected_generation = endpoint.tokenGeneration;
    event.challenge_id = allocate(nextChallenge, std::numeric_limits<uint64_t>::max());
    event.reason = reason;
    event.grpc_status_present = rejected ? 1 : 0;
    event.grpc_status = rejected ? grpc::StatusCode::UNAUTHENTICATED : 0;
    runtime_.diagnostics->emit(OMNI_USD_LIVE_RECORD_AUTH_CHALLENGE, endpoint.handle, endpoint.status.subject_id, 0,
        {{DiagnosticKey::Operation, OMNI_USD_LIVE_OPERATION_AUTH},
            {DiagnosticKey::AuthGeneration, endpoint.tokenGeneration}, {DiagnosticKey::Challenge, event.challenge_id},
            {DiagnosticKey::Reason, reason}});
    for (const auto& observer : endpoint.authObservers) observer->notify(event);
}
void AuthRegistry::evaluate(Endpoint& endpoint, uint64_t nowMs, std::chrono::steady_clock::time_point now) {
    if (endpoint.closed) return;
    const bool valid = usable(endpoint, nowMs, now);
    // authWaitActive marks the unusable interval, so only the first evaluation
    // of an interval observes the usable-to-unusable transition.
    bool transition = false;
    if (!valid) {
        if (!endpoint.authFailed && !endpoint.authWaitActive) {
            endpoint.authWaitStarted = now;
            endpoint.authWaitActive = true;
            runtime_.diagnostics->startAuthWait();
            transition = true;
        }
        publishBlocked(endpoint, nowMs);
        if (endpoint.authDeadline == std::chrono::steady_clock::time_point{})
            endpoint.authDeadline = now + std::chrono::milliseconds(endpoint.options.authWaitMs);
    }
    if (endpoint.authDeadline != std::chrono::steady_clock::time_point{} && now >= endpoint.authDeadline) {
        failClosed(endpoint, AuthFailureCause::Deadline, now);
        publishBlocked(endpoint, nowMs);
        return;
    }
    if (endpoint.authFailed) return;
    // Usable to unusable (expiry on either clock, rejection or a missing token):
    // parked operations re-evaluate at once instead of at their own next
    // deadline. Repeated evaluation within the interval wakes nothing, and a
    // deadline failure above wakes inside failClosed instead.
    if (transition) wake();
    if (valid) finishWait(endpoint, AuthWaitResolution::Usable, now);
    const auto remaining = remainingMs(endpoint, nowMs, now);
    const bool expiring = remaining != UINT64_MAX && remaining <= runtime_.options.values.auth_expiry_lead_ms;
    if (!valid || expiring)
        challenge(endpoint,
            endpoint.generationRejected ? OMNI_USD_LIVE_AUTH_REJECTED :
                endpoint.token.empty()  ? OMNI_USD_LIVE_AUTH_MISSING :
                                          OMNI_USD_LIVE_AUTH_EXPIRING,
            endpoint.generationRejected);
}
std::chrono::steady_clock::time_point AuthRegistry::nextEvaluation(const Endpoint& endpoint, uint64_t nowMs,
    std::chrono::steady_clock::time_point now) const noexcept {
    using Clock = std::chrono::steady_clock;
    auto next = Clock::time_point::max();
    if (endpoint.closed || endpoint.authFailed) return next;
    const auto remaining = remainingMs(endpoint, nowMs, now);
    if (remaining != UINT64_MAX && remaining > 0) {
        auto until = remaining;
        if (!endpoint.challenge.challenge_id) {
            const auto lead = runtime_.options.values.auth_expiry_lead_ms;
            until = until > lead ? until - lead : 1;
        }
        // Revisit wall time at least once a second while expiry is pending.
        const auto delay = std::chrono::milliseconds(std::min<uint64_t>(until, 1000));
        next = now > Clock::time_point::max() - delay ? Clock::time_point::max() : now + delay;
    }
    if (endpoint.authDeadline != Clock::time_point{}) next = std::min(next, endpoint.authDeadline);
    return next;
}
void AuthRegistry::completed(Endpoint& endpoint, uint64_t generation, grpc::StatusCode code) noexcept {
    completedImpl(endpoint, generation, code, false, 0, {});
}
void AuthRegistry::completedAt(Endpoint& endpoint, uint64_t generation, grpc::StatusCode code, uint64_t nowMs,
    std::chrono::steady_clock::time_point now) noexcept {
    completedImpl(endpoint, generation, code, true, nowMs, now);
}
void AuthRegistry::completedImpl(Endpoint& endpoint, uint64_t generation, grpc::StatusCode code, bool suppliedClock,
    uint64_t nowMs, std::chrono::steady_clock::time_point now) noexcept {
    bool changed = false;
    try {
        std::lock_guard<std::mutex> lock(endpoint.mutex);
        if (endpoint.closed || generation != endpoint.tokenGeneration) return;
        if (!suppliedClock) {
            nowMs = wallTimeMs();
            now = std::chrono::steady_clock::now();
        }
        if (code == grpc::StatusCode::OK) {
            endpoint.rejectedGenerations = 0;
            // A wall-clock reversal can make an expired token appear locally
            // usable again. A completed old RPC cannot undo the monotonic auth
            // wait deadline or misclassify its interval as successfully resolved.
            if (endpoint.authDeadline != std::chrono::steady_clock::time_point{} && now >= endpoint.authDeadline)
                evaluate(endpoint, nowMs, now);
            if (usable(endpoint, nowMs, now)) finishWait(endpoint, AuthWaitResolution::Usable, now);
            if (usable(endpoint, nowMs, now) && endpoint.status.state != OMNI_USD_LIVE_STATE_READY) {
                auto& status = endpoint.status;
                status.state = OMNI_USD_LIVE_STATE_READY;
                status.result = OMNI_USD_LIVE_OK;
                status.operation = OMNI_USD_LIVE_OPERATION_NONE;
                status.recovery_action = OMNI_USD_LIVE_RECOVERY_NONE;
                status.grpc_status_present = 1;
                status.grpc_status = 0;
                status.changed_at_ms = nowMs;
                ++status.revision;
                runtime_.statuses->publish(status.subject_id, status);
                changed = true;
            }
        } else if (code == grpc::StatusCode::UNAUTHENTICATED && !endpoint.generationRejected) {
            endpoint.generationRejected = true;
            const bool failedBefore = endpoint.authFailed, waitingBefore = endpoint.authWaitActive;
            const auto serial = endpoint.authFailureSerial;
            // The initial generation plus at most N replacement generations.
            if (endpoint.rejectedGenerations >= runtime_.options.values.max_rejected_token_generations) {
                failClosed(endpoint, AuthFailureCause::RejectionLimit, now);
            } else ++endpoint.rejectedGenerations;
            evaluate(endpoint, nowMs, now);
            // A new failure (failClosed) or the usable-to-unusable transition in
            // evaluate wakes every scheduled operation once; a rejection inside
            // an existing unusable interval or after a failure wakes below.
            changed = (failedBefore || waitingBefore) && endpoint.authFailureSerial == serial;
        }
    } catch (...) {
        // Exhausted challenge IDs cannot silently permit repeated rejected calls.
        std::lock_guard<std::mutex> lock(endpoint.mutex);
        failClosed(endpoint, AuthFailureCause::Internal, now);
        publishBlocked(endpoint, nowMs);
    }
    // Only an auth state change wakes every scheduled operation: READY
    // publication or a rejected generation (a terminal failure wakes inside
    // failClosed, the usable-to-unusable transition inside evaluate). A
    // completing operation wakes itself, so a read frame, write or unary
    // finish costs no scan of other layers' operations.
    if (changed) wake();
}
OmniUsdLiveResult AuthRegistry::subscribe(OmniUsdLiveEndpoint endpoint, void* user,
    OmniUsdLiveAuthRequiredCallback callback, OmniUsdLiveCallbackHandle* output) {
    auto lease = runtime_.acquireEndpoint(endpoint);
    if (!lease) return runtime_.closing ? OMNI_USD_LIVE_BUSY : OMNI_USD_LIVE_INVALID_HANDLE;
    auto retained = runtime_.findEndpoint(endpoint);
    const auto handle = (UINT64_C(4) << 56) | allocate(nextCallback, UINT64_C(1) << 56);
    auto observer = std::make_shared<AuthCallback>(runtime_.requests->executor(), std::move(lease), handle, user,
        callback);
    std::lock_guard<std::mutex> lock(mutex_);
    if (callbacks_.size() >= runtime_.options.values.max_auth_callbacks) return OMNI_USD_LIVE_RESOURCE_EXHAUSTED;
    std::lock_guard<std::mutex> endpointLock(retained->mutex);
    auto& observers = retained->authObservers;
    observers.reserve(observers.size() + 1);
    // Evaluate before admission; allocation failure cannot leave a public handle.
    evaluate(*retained, wallTimeMs(), std::chrono::steady_clock::now());
    callbacks_.emplace(handle, observer);
    observers.push_back(observer);
    if (!retained->authFailed) observer->notify(retained->challenge);
    *output = handle;
    return OMNI_USD_LIVE_OK;
}
OmniUsdLiveResult AuthRegistry::unsubscribe(OmniUsdLiveCallbackHandle handle, uint32_t timeoutMs,
    const std::function<void()>& beforeDetachForTest) {
    std::shared_ptr<AuthCallback> observer;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = callbacks_.find(handle);
        if (found == callbacks_.end()) return OMNI_USD_LIVE_INVALID_HANDLE;
        observer = found->second;
    }
    auto result = observer->cancel(timeoutMs);
    if (result != OMNI_USD_LIVE_OK) return result;
    if (beforeDetachForTest) beforeDetachForTest();
    observer->detach(); // Retain local ownership until after the endpoint lock is released.
    std::lock_guard<std::mutex> lock(mutex_);
    return callbacks_.erase(handle) ? OMNI_USD_LIVE_OK : OMNI_USD_LIVE_INVALID_HANDLE;
}
OmniUsdLiveResult AuthRegistry::snapshot(OmniUsdLiveEndpoint endpoint, OmniUsdLiveAuthState* output) {
    auto retained = runtime_.findEndpoint(endpoint);
    if (!retained) return OMNI_USD_LIVE_INVALID_HANDLE;
    std::lock_guard<std::mutex> lock(retained->mutex);
    if (retained->closed) return OMNI_USD_LIVE_INVALID_HANDLE;
    OmniUsdLiveAuthState state{};
    state.struct_size = output->struct_size;
    state.abi_version = OMNI_USD_LIVE_ABI_VERSION;
    state.generation = retained->tokenGeneration;
    state.expires_at_ms = retained->expiresAtMs;
    state.challenge_id = retained->authFailed ? 0 : retained->challenge.challenge_id;
    state.reason = retained->authFailed ? 0 : retained->challenge.reason;
    state.state = retained->authFailed  ? uint32_t(OMNI_USD_LIVE_STATE_FAILED) :
        usable(*retained, wallTimeMs()) ? retained->status.state :
                                          uint32_t(OMNI_USD_LIVE_STATE_AUTH_WAIT);
    std::memcpy(output, &state, sizeof(state));
    return OMNI_USD_LIVE_OK;
}
OmniUsdLiveResult AuthRegistry::replace(OmniUsdLiveEndpoint endpoint, uint64_t generation, const char* token,
    uint64_t expiry, uint64_t* output) {
    if (!token) return OMNI_USD_LIVE_INVALID_ARGUMENT;
    size_t length = 0;
    while (token[length]) {
        if (length == 65536) return OMNI_USD_LIVE_RESOURCE_EXHAUSTED;
        if (static_cast<unsigned char>(token[length]) <= 0x20 || static_cast<unsigned char>(token[length]) >= 0x7f)
            return OMNI_USD_LIVE_INVALID_ARGUMENT;
        ++length;
    }
    std::string copied(token, length);
    auto retained = runtime_.findEndpoint(endpoint);
    if (!retained) return OMNI_USD_LIVE_INVALID_HANDLE;
    {
        std::lock_guard<std::mutex> lock(retained->mutex);
        if (retained->closed) return OMNI_USD_LIVE_INVALID_HANDLE;
        if (retained->tokenGeneration != generation) {
            runtime_.diagnostics->staleAuthReplacement();
            return OMNI_USD_LIVE_STALE_TOKEN_GENERATION;
        }
        if (generation == UINT64_MAX) return OMNI_USD_LIVE_RESOURCE_EXHAUSTED;
        const auto replacedChallenge = retained->challenge.challenge_id;
        retained->token = std::move(copied);
        retained->tokenGeneration++;
        retained->expiresAtMs = expiry;
        retained->expiresAtSteady = steadyExpiry(expiry, wallTimeMs(), std::chrono::steady_clock::now());
        retained->challenge = {};
        retained->authDeadline = {};
        retained->generationRejected = false;
        retained->authFailed = false;
        if (usable(*retained, wallTimeMs()))
            finishWait(*retained, AuthWaitResolution::Usable, std::chrono::steady_clock::now());
        // A token does not establish connectivity/readiness or clear incidents.
        if (retained->status.state == OMNI_USD_LIVE_STATE_AUTH_WAIT) {
            retained->status.state = OMNI_USD_LIVE_STATE_OPENING;
            retained->status.result = OMNI_USD_LIVE_OK;
            retained->status.operation = OMNI_USD_LIVE_OPERATION_OPEN;
            retained->status.grpc_status_present = 0;
            retained->status.grpc_status = 0;
            retained->status.recovery_action = OMNI_USD_LIVE_RECOVERY_NONE;
            retained->status.changed_at_ms = wallTimeMs();
            ++retained->status.revision;
            runtime_.statuses->publish(retained->status.subject_id, retained->status);
        }
        *output = retained->tokenGeneration;
        runtime_.diagnostics->emit(OMNI_USD_LIVE_RECORD_AUTH_REPLACED, retained->handle, retained->status.subject_id, 0,
            {{DiagnosticKey::Operation, OMNI_USD_LIVE_OPERATION_AUTH}, {DiagnosticKey::Result, OMNI_USD_LIVE_OK},
                {DiagnosticKey::AuthGeneration, retained->tokenGeneration},
                {DiagnosticKey::Challenge, replacedChallenge}});
    }
    wake();
    return OMNI_USD_LIVE_OK;
}
OmniUsdLiveResult AuthRegistry::fail(OmniUsdLiveEndpoint endpoint, uint64_t challengeId) {
    auto retained = runtime_.findEndpoint(endpoint);
    if (!retained) return OMNI_USD_LIVE_INVALID_HANDLE;
    std::lock_guard<std::mutex> lock(retained->mutex);
    if (retained->closed) return OMNI_USD_LIVE_INVALID_HANDLE;
    if (!challengeId || retained->challenge.challenge_id != challengeId || retained->authFailed)
        return OMNI_USD_LIVE_NOT_FOUND;
    failClosed(*retained, AuthFailureCause::Provider, std::chrono::steady_clock::now());
    publishBlocked(*retained, wallTimeMs());
    return OMNI_USD_LIVE_OK;
}
void AuthRegistry::wake() {
    std::lock_guard<std::mutex> lock(timerMutex_);
    ++wakeRevision_;
    for (auto& entry : operations_)
        if (entry.second.active) {
            entry.second.operation->wakeRevision_.fetch_add(1, std::memory_order_release);
            entry.second.next = std::chrono::steady_clock::now();
            ++entry.second.revision;
        }
    timerChanged_.notify_one();
}
void AuthRegistry::reserve(std::shared_ptr<ScheduledOperation> operation) {
    std::lock_guard<std::mutex> lock(timerMutex_);
    if (stopping_) throw OmniUsdLiveResult(OMNI_USD_LIVE_BUSY);
    if (operations_.size() >= runtime_.options.values.max_requests)
        throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    auto* key = operation.get();
    operations_.emplace(key, Entry{std::move(operation)});
}
void AuthRegistry::activate(ScheduledOperation& operation) {
    std::lock_guard<std::mutex> lock(timerMutex_);
    auto& entry = operations_.at(&operation);
    entry.active = true;
    entry.next = std::chrono::steady_clock::now();
    ++entry.revision;
    ++wakeRevision_;
    timerChanged_.notify_one();
}
void AuthRegistry::wake(ScheduledOperation& operation) {
    std::lock_guard<std::mutex> lock(timerMutex_);
    const auto found = operations_.find(&operation);
    if (found != operations_.end() && found->second.active) {
        operation.wakeRevision_.fetch_add(1, std::memory_order_release);
        found->second.next = std::chrono::steady_clock::now();
        ++found->second.revision;
        ++wakeRevision_;
        timerChanged_.notify_one();
    }
}
void AuthRegistry::schedule(ScheduledOperation& operation, std::chrono::steady_clock::time_point next,
    uint64_t observedWake) {
    std::lock_guard<std::mutex> lock(timerMutex_);
    const auto found = operations_.find(&operation);
    if (found != operations_.end() && found->second.active) {
        // A global/auth wake can race before the timer calls advance again.
        // It must not be overwritten by a worker's stale future deadline.
        found->second.next = operation.wakeRevision() == observedWake ? next : std::chrono::steady_clock::now();
        ++found->second.revision;
        ++wakeRevision_;
        timerChanged_.notify_one();
    }
}
void AuthRegistry::discard(ScheduledOperation& operation) {
    std::shared_ptr<ScheduledOperation> retained;
    {
        std::lock_guard<std::mutex> lock(timerMutex_);
        const auto found = operations_.find(&operation);
        if (found != operations_.end()) {
            retained = std::move(found->second.operation);
            operations_.erase(found);
        }
    } // Destruction may release endpoint leases; never destroy under timer locks.
}
void AuthRegistry::work() noexcept {
    std::unique_lock<std::mutex> lock(timerMutex_);
    while (!stopping_) {
        const auto revision = wakeRevision_;
        auto authNext = std::chrono::steady_clock::time_point::max();
        lock.unlock();
        {
            // No host callbacks or allocations while scanning; each endpoint
            // reserves its observer work nodes at registration. A bounded scan
            // also notices wall-clock jumps without creating one thread per endpoint.
            std::lock_guard<std::mutex> registryLock(runtime_.mutex);
            for (const auto& entry : runtime_.endpoints) {
                std::lock_guard<std::mutex> endpointLock(entry.second->mutex);
                const auto wall = wallTimeMs();
                const auto now = std::chrono::steady_clock::now();
                auto& endpoint = *entry.second;
                try {
                    evaluate(endpoint, wall, now);
                } catch (...) {
                    failClosed(endpoint, AuthFailureCause::Internal, now);
                    publishBlocked(endpoint, wallTimeMs());
                }
                authNext = std::min(authNext, nextEvaluation(endpoint, wall, now));
            }
        }
        lock.lock();
        // Drive due finite operations outside every scheduler/runtime lock.
        // Revisions preserve cancellation/completion wakes racing an advance.
        auto nextWake = authNext;
        size_t budget = operations_.size();
        for (;;) {
            auto earliest = operations_.end();
            for (auto it = operations_.begin(); it != operations_.end(); ++it)
                if (it->second.active && (earliest == operations_.end() || it->second.next < earliest->second.next))
                    earliest = it;
            const auto now = std::chrono::steady_clock::now();
            if (stopping_ || earliest == operations_.end() || earliest->second.next > now) {
                nextWake = authNext;
                if (earliest != operations_.end()) nextWake = std::min(nextWake, earliest->second.next);
                break;
            }
            if (!budget-- || now >= authNext) {
                // Revisit expiry even while RPC completions keep operations due.
                nextWake = now;
                break;
            }
            auto operation = earliest->second.operation;
            const auto entryRevision = earliest->second.revision;
            lock.unlock();
            const auto result = operation->advance(std::chrono::steady_clock::now());
            lock.lock();
            const auto found = operations_.find(operation.get());
            if (found != operations_.end()) {
                if (result.done) operations_.erase(found);
                else if (found->second.revision == entryRevision) found->second.next = result.next;
            }
            lock.unlock();
            operation.reset();
            lock.lock();
        }
        if (nextWake == std::chrono::steady_clock::time_point::max())
            timerChanged_.wait(lock, [&] { return stopping_ || revision != wakeRevision_; });
        else timerChanged_.wait_until(lock, nextWake, [&] { return stopping_ || revision != wakeRevision_; });
    }
    stopped_ = true;
    timerStopped_.notify_all();
}
void AuthRegistry::beginShutdown() {
    std::lock_guard<std::mutex> lock(timerMutex_);
    stopping_ = true;
    timerChanged_.notify_one();
}
bool AuthRegistry::waitForShutdown(std::chrono::steady_clock::time_point deadline) {
    std::unique_lock<std::mutex> lock(timerMutex_);
    return timerStopped_.wait_until(lock, deadline, [&] { return stopped_; });
}
} // namespace omni::usd::live
