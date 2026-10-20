// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "LiveAuth.h"
#include "LiveDeadline.h"
#include "LiveRequest.h"
#include "LiveUnary.h"
#include <cassert>
#include <future>
#include <vector>

using namespace omni::usd::live;
namespace {
template <class T> T record()
{
    T value{};
    value.struct_size = sizeof(value);
    value.abi_version = OMNI_USD_LIVE_ABI_VERSION;
    return value;
}
struct Observer {
    std::mutex mutex;
    std::condition_variable changed;
    std::vector<OmniUsdLiveAuthRequired> events;
    bool block = false, release = false;
    OmniUsdLiveCallbackHandle handle = 0;
    std::shared_ptr<Request> unfinished;
    bool selfCancel = false;
    static void OMNI_USD_LIVE_CALL callback(void* user, const OmniUsdLiveAuthRequired* event)
    {
        auto& self = *static_cast<Observer*>(user);
        auto snapshot = record<OmniUsdLiveAuthState>();
        assert(omniUsdLiveGetAuthState(event->endpoint, &snapshot) == OMNI_USD_LIVE_OK);
        assert(snapshot.generation == event->rejected_generation);
        if (self.selfCancel)
            assert(omniUsdLiveUnregisterCallback(self.handle, 1000) == OMNI_USD_LIVE_BUSY);
        if (self.unfinished)
            assert(self.unfinished->wait(1000) == OMNI_USD_LIVE_BUSY);
        std::unique_lock<std::mutex> lock(self.mutex);
        self.events.push_back(*event);
        self.changed.notify_all();
        if (self.block)
            self.changed.wait(lock, [&] { return self.release; });
    }
    OmniUsdLiveAuthRequired wait(size_t count)
    {
        std::unique_lock<std::mutex> lock(mutex);
        assert(changed.wait_for(lock, std::chrono::seconds(3), [&] { return events.size() >= count; }));
        return events[count - 1];
    }
};
OmniUsdLiveAuthState state(OmniUsdLiveEndpoint endpoint)
{
    auto result = record<OmniUsdLiveAuthState>();
    assert(omniUsdLiveGetAuthState(endpoint, &result) == OMNI_USD_LIVE_OK);
    return result;
}
uint64_t replace(OmniUsdLiveEndpoint endpoint, uint64_t generation, const char* token = "synthetic-new")
{
    uint64_t output = 0;
    assert(omniUsdLiveSetAccessToken(endpoint, generation, token, 0, &output) == OMNI_USD_LIVE_OK);
    assert(output == generation + 1);
    return output;
}
void unregisterEndpoint(OmniUsdLiveEndpoint endpoint)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    for (;;) {
        auto result = omniUsdLiveUnregisterEndpoint(endpoint, 0);
        if (result == OMNI_USD_LIVE_OK)
            return;
        assert(result == OMNI_USD_LIVE_BUSY && std::chrono::steady_clock::now() < deadline);
        std::this_thread::yield();
    }
}
void testAuthFailureWake()
{
    using Clock = std::chrono::steady_clock;
    auto options = record<OmniUsdLiveRuntimeOptions>();
    options.flags = OMNI_USD_LIVE_RUNTIME_DISABLE_DIAGNOSTIC_HISTORY;
    assert(omniUsdLiveInitialize(&options) == OMNI_USD_LIVE_OK);
    auto endpointOptions = record<OmniUsdLiveEndpointOptions>();
    endpointOptions.authority = "auth-failure-wake.example.invalid";
    endpointOptions.access_token = "synthetic-initial";
    OmniUsdLiveEndpoint endpoint = 0;
    assert(omniUsdLiveRegisterEndpoint(&endpointOptions, &endpoint) == OMNI_USD_LIVE_OK);
    auto runtime = getRuntime();
    auto retained = runtime->findEndpoint(endpoint);
    struct Parked final : ScheduledOperation {
        std::shared_ptr<Endpoint> endpoint;
        bool hold = false;
        unsigned calls = 0;
        std::promise<void> entered, release, failed;
        std::shared_future<void> gate = release.get_future().share();
        ScheduledStep advance(Clock::time_point) noexcept override
        {
            if (!calls++) {
                entered.set_value();
                if (hold) gate.wait();
                return {false, Clock::time_point::max()};
            }
            {
                std::lock_guard<std::mutex> lock(endpoint->mutex);
                assert(endpoint->authFailed);
            }
            failed.set_value();
            return {true, {}};
        }
    };
    uint64_t generation = 1, serial = 0;
    for (bool hold : {false, true}) {
        auto operation = std::make_shared<Parked>();
        operation->endpoint = retained;
        operation->hold = hold;
        auto entered = operation->entered.get_future(), failed = operation->failed.get_future();
        runtime->auth->reserve(operation);
        runtime->auth->activate(*operation);
        assert(entered.wait_for(std::chrono::seconds(3)) == std::future_status::ready);
        // Explicitly park it even when the timer has not retired advance yet.
        runtime->auth->schedule(*operation, Clock::time_point::max(), operation->wakeRevision());
        if (!hold) {
            // The single scheduler cannot enter this barrier until it has
            // committed the tested operation's first return/deadline.
            struct Barrier final : ScheduledOperation {
                std::promise<void> entered;
                ScheduledStep advance(Clock::time_point) noexcept override
                { entered.set_value(); return {true, {}}; }
            };
            auto barrier = std::make_shared<Barrier>();
            auto parked = barrier->entered.get_future();
            runtime->auth->reserve(barrier);
            runtime->auth->activate(*barrier);
            assert(parked.wait_for(std::chrono::seconds(3)) == std::future_status::ready);
        }
        {
            std::lock_guard<std::mutex> lock(retained->mutex);
            assert(!retained->authFailed && retained->authFailureSerial == serial);
            runtime->auth->failClosed(*retained, AuthFailureCause::Internal, Clock::now());
            runtime->auth->failClosed(*retained, AuthFailureCause::Deadline, Clock::now());
            assert(retained->authFailed && retained->authFailureSerial == ++serial);
        }
        if (hold) operation->release.set_value();
        assert(failed.wait_for(std::chrono::seconds(3)) == std::future_status::ready);
        assert(state(endpoint).state == OMNI_USD_LIVE_STATE_FAILED);
        const auto diagnostic = runtime->diagnostics->snapshot(0, 100);
        assert(diagnostic.records.empty());
        assert(diagnostic.metrics.authFailuresByCause[3] == serial);
        assert(diagnostic.metrics.authFailuresByCause[1] == 0);
        generation = replace(endpoint, generation);
    }
    retained.reset();
    unregisterEndpoint(endpoint);
    runtime.reset();
    assert(omniUsdLiveShutdown(3000) == OMNI_USD_LIVE_OK);
}
void testAuthDeadlineSelection()
{
    using Clock = std::chrono::steady_clock;
    using Ms = std::chrono::milliseconds;
    auto options = record<OmniUsdLiveRuntimeOptions>();
    options.default_auth_wait_timeout_ms = 5000;
    options.auth_expiry_lead_ms = 30000;
    assert(omniUsdLiveInitialize(&options) == OMNI_USD_LIVE_OK);
    auto runtime = getRuntime();
    runtime->auth->beginShutdown();
    assert(runtime->auth->waitForShutdown(Clock::now() + std::chrono::seconds(3)));
    auto endpointOptions = record<OmniUsdLiveEndpointOptions>();
    endpointOptions.authority = "auth-deadlines.example.invalid";
    endpointOptions.access_token = "synthetic-timer";
    OmniUsdLiveEndpoint endpoint = 0;
    assert(omniUsdLiveRegisterEndpoint(&endpointOptions, &endpoint) == OMNI_USD_LIVE_OK);
    auto retained = runtime->findEndpoint(endpoint);
    endpointOptions.authority = "auth-later-scan.example.invalid";
    endpointOptions.expires_at_ms = UINT64_MAX;
    OmniUsdLiveEndpoint laterEndpoint = 0;
    assert(omniUsdLiveRegisterEndpoint(&endpointOptions, &laterEndpoint) == OMNI_USD_LIVE_OK);
    auto later = runtime->findEndpoint(laterEndpoint);
    {
        std::lock_guard<std::mutex> lock(retained->mutex);
        const auto start = Clock::time_point{} + std::chrono::seconds(200);
        constexpr uint64_t wall = 100000;
        auto& auth = *runtime->auth;
        assert(auth.nextEvaluation(*retained, wall, start) == Clock::time_point::max());
        retained->expiresAtMs = wall + 30025;
        auth.evaluate(*retained, wall, start);
        assert(!retained->challenge.challenge_id);
        const auto first = auth.nextEvaluation(*retained, wall, start);
        assert(first == start + Ms(25)); // First proactive refresh threshold.
        auth.evaluate(*retained, wall + 25, first);
        assert(retained->challenge.challenge_id && retained->authDeadline == Clock::time_point{});
        assert(auth.nextEvaluation(*retained, wall + 25, first) == start + Ms(1025));
        // Wall time leaps ahead. Once expired, refresh has a monotonic budget.
        auth.evaluate(*retained, wall + 31000, start + Ms(50));
        const auto deadline = start + Ms(5050);
        assert(retained->authDeadline == deadline && !retained->authFailed);
        auto earliest = auth.nextEvaluation(*retained, wall + 31000, start + Ms(50));
        assert(earliest == deadline);
        // Simulate later endpoint scans consuming time. Their future poll must
        // not shift the first endpoint's deadline by the scan duration.
        {
            std::lock_guard<std::mutex> laterLock(later->mutex);
            const auto laterPoll = auth.nextEvaluation(*later, wall, start + Ms(4900));
            assert(laterPoll == start + Ms(5900));
            earliest = std::min(earliest, laterPoll);
            assert(earliest == deadline && earliest < start + Ms(6000));
            assert(auth.nextEvaluation(*later, wall, Clock::time_point::max() - Ms(1)) ==
                   Clock::time_point::max());
            later->closed = true;
            assert(auth.nextEvaluation(*later, wall, start) == Clock::time_point::max());
            later->closed = false;
        }
        // A wall-clock reversal never extends an already-started auth deadline.
        assert(auth.nextEvaluation(*retained, wall, start + Ms(4900)) == deadline);
        assert(auth.nextEvaluation(*retained, wall + 31000, deadline + Ms(300)) == deadline);
        auth.evaluate(*retained, wall, deadline);
        assert(retained->authFailed && retained->authFailureSerial == 1);
        assert(auth.nextEvaluation(*retained, wall, deadline) == Clock::time_point::max());
    }
    retained.reset();
    later.reset();
    unregisterEndpoint(endpoint);
    unregisterEndpoint(laterEndpoint);
    runtime.reset();
    assert(omniUsdLiveShutdown(3000) == OMNI_USD_LIVE_OK);
}
// A completion that leaves the endpoint's auth state unchanged wakes no parked
// operation (each completing operation wakes itself). Only a state change, READY
// publication or a rejected generation, wakes every scheduled operation.
void testCompletionWakesOnlyOnAuthChange()
{
    using Clock = std::chrono::steady_clock;
    auto options = record<OmniUsdLiveRuntimeOptions>();
    options.flags = OMNI_USD_LIVE_RUNTIME_DISABLE_DIAGNOSTIC_HISTORY;
    assert(omniUsdLiveInitialize(&options) == OMNI_USD_LIVE_OK);
    auto endpointOptions = record<OmniUsdLiveEndpointOptions>();
    endpointOptions.authority = "auth-completion-wake.example.invalid";
    endpointOptions.access_token = "synthetic-initial";
    OmniUsdLiveEndpoint endpoint = 0;
    assert(omniUsdLiveRegisterEndpoint(&endpointOptions, &endpoint) == OMNI_USD_LIVE_OK);
    auto runtime = getRuntime();
    auto retained = runtime->findEndpoint(endpoint);
    struct Idle final : ScheduledOperation {
        ScheduledStep advance(Clock::time_point) noexcept override { return {false, Clock::time_point::max()}; }
    };
    std::vector<std::shared_ptr<Idle>> actors;
    for (int i = 0; i < 8; ++i) {
        actors.push_back(std::make_shared<Idle>());
        runtime->auth->reserve(actors.back());
        runtime->auth->activate(*actors.back());
    }
    // The read stream's own operation, which wakes itself for each frame.
    auto stream = std::make_shared<Idle>();
    runtime->auth->reserve(stream);
    runtime->auth->activate(*stream);
    // Quiesce the timer so only the completions below change wake revisions.
    runtime->auth->beginShutdown();
    assert(runtime->auth->waitForShutdown(Clock::now() + std::chrono::seconds(3)));
    const auto revisions = [&] {
        std::vector<uint64_t> result;
        for (const auto& actor : actors) result.push_back(actor->wakeRevision());
        return result;
    };
    uint64_t generation = 0;
    {
        std::lock_guard<std::mutex> lock(retained->mutex);
        generation = retained->tokenGeneration;
        // A registered endpoint with a token starts OPENING, so the first OK publishes READY.
        assert(retained->status.state != OMNI_USD_LIVE_STATE_READY);
    }
    const auto opening = revisions();
    runtime->auth->completed(*retained, generation, grpc::StatusCode::OK); // Publishes READY.
    const auto ready = revisions();
    for (size_t i = 0; i < actors.size(); ++i)
        assert(ready[i] == opening[i] + 1);
    // One read stream's frames and other layers' unary/write finishes: each
    // frame steps exactly one operation, the stream itself.
    for (int frame = 0; frame < 100; ++frame) {
        const auto streamBefore = stream->wakeRevision();
        runtime->auth->completed(*retained, generation, grpc::StatusCode::OK);
        runtime->auth->wake(*stream);
        runtime->auth->completed(*retained, generation, grpc::StatusCode::NOT_FOUND);
        runtime->auth->completed(*retained, generation, grpc::StatusCode::PERMISSION_DENIED);
        runtime->auth->completed(*retained, generation - 1, grpc::StatusCode::UNAUTHENTICATED);
        assert(stream->wakeRevision() == streamBefore + 1);
    }
    assert(revisions() == ready);
    runtime->auth->completed(*retained, generation, grpc::StatusCode::UNAUTHENTICATED);
    const auto rejected = revisions();
    for (size_t i = 0; i < actors.size(); ++i)
        assert(rejected[i] == ready[i] + 1);
    // A generation already marked rejected changes no auth state.
    runtime->auth->completed(*retained, generation, grpc::StatusCode::UNAUTHENTICATED);
    assert(revisions() == rejected);
    for (const auto& actor : actors)
        runtime->auth->discard(*actor);
    runtime->auth->discard(*stream);
    retained.reset();
    unregisterEndpoint(endpoint);
    runtime.reset();
    assert(omniUsdLiveShutdown(3000) == OMNI_USD_LIVE_OK);
}
// A token's lifetime is also measured on the steady clock from installation, so
// a backward wall-clock step cannot keep an expired token usable, and the
// usable-to-expired transition wakes parked operations to re-evaluate.
void testTokenExpiryIgnoresWallClockStepBack()
{
    using Clock = std::chrono::steady_clock;
    auto options = record<OmniUsdLiveRuntimeOptions>();
    options.flags = OMNI_USD_LIVE_RUNTIME_DISABLE_DIAGNOSTIC_HISTORY;
    assert(omniUsdLiveInitialize(&options) == OMNI_USD_LIVE_OK);
    auto endpointOptions = record<OmniUsdLiveEndpointOptions>();
    endpointOptions.authority = "auth-steady-expiry.example.invalid";
    endpointOptions.access_token = "synthetic-initial";
    OmniUsdLiveEndpoint endpoint = 0;
    assert(omniUsdLiveRegisterEndpoint(&endpointOptions, &endpoint) == OMNI_USD_LIVE_OK);
    auto runtime = getRuntime();
    auto retained = runtime->findEndpoint(endpoint);
    struct Idle final : ScheduledOperation {
        ScheduledStep advance(Clock::time_point) noexcept override { return {false, Clock::time_point::max()}; }
    };
    auto parked = std::make_shared<Idle>();
    runtime->auth->reserve(parked);
    runtime->auth->activate(*parked);
    // Quiesce the timer: only the supplied clocks below evaluate the token.
    runtime->auth->beginShutdown();
    assert(runtime->auth->waitForShutdown(Clock::now() + std::chrono::seconds(3)));
    const auto installedWall = wallTimeMs();
    uint64_t generation = 0;
    assert(omniUsdLiveSetAccessToken(endpoint, state(endpoint).generation, "synthetic-lifetime",
                                     installedWall + 60000, &generation) == OMNI_USD_LIVE_OK);
    {
        std::lock_guard<std::mutex> lock(retained->mutex);
        const auto expiry = retained->expiresAtSteady;
        assert(expiry != Clock::time_point{});
        const auto steppedBack = installedWall - 10000; // Wall clock stepped back 10 s after install.
        assert(runtime->auth->usable(*retained, steppedBack, expiry - std::chrono::milliseconds(1)));
        assert(!runtime->auth->usable(*retained, steppedBack, expiry));
        const auto before = parked->wakeRevision();
        runtime->auth->evaluate(*retained, steppedBack, expiry);
        assert(retained->challenge.challenge_id && retained->challenge.reason == OMNI_USD_LIVE_AUTH_EXPIRING);
        assert(retained->status.state == OMNI_USD_LIVE_STATE_AUTH_WAIT);
        assert(parked->wakeRevision() == before + 1);
        runtime->auth->evaluate(*retained, steppedBack, expiry + std::chrono::milliseconds(5));
        assert(parked->wakeRevision() == before + 1); // Only the transition wakes.
    }
    {
        // A rejection inside the unusable interval is its own auth change: one wake.
        Clock::time_point expiry;
        { std::lock_guard<std::mutex> lock(retained->mutex); expiry = retained->expiresAtSteady; }
        const auto before = parked->wakeRevision();
        runtime->auth->completedAt(*retained, generation, grpc::StatusCode::UNAUTHENTICATED,
                                   installedWall - 10000, expiry + std::chrono::milliseconds(10));
        std::lock_guard<std::mutex> lock(retained->mutex);
        assert(retained->generationRejected && !retained->authFailed);
        assert(parked->wakeRevision() == before + 1);
    }
    {
        // A rejection that reaches the auth-wait deadline fails the endpoint; that
        // terminal failure is the single wake.
        assert(omniUsdLiveSetAccessToken(endpoint, generation, "synthetic-deadline",
                                         wallTimeMs() + 60000, &generation) == OMNI_USD_LIVE_OK);
        Clock::time_point deadline;
        uint64_t serial = 0;
        {
            std::lock_guard<std::mutex> lock(retained->mutex);
            const auto expiry = retained->expiresAtSteady;
            runtime->auth->evaluate(*retained, installedWall - 10000, expiry);
            assert(retained->authWaitActive && retained->authDeadline != Clock::time_point{});
            deadline = retained->authDeadline;
            serial = retained->authFailureSerial;
        }
        const auto before = parked->wakeRevision();
        runtime->auth->completedAt(*retained, generation, grpc::StatusCode::UNAUTHENTICATED,
                                   installedWall - 10000, deadline);
        std::lock_guard<std::mutex> lock(retained->mutex);
        // Below the rejection limit: the deadline, not the limit, fails the endpoint.
        assert(retained->rejectedGenerations == 2);
        assert(retained->authFailed && retained->authFailureSerial == serial + 1);
        assert(parked->wakeRevision() == before + 1);
    }
    runtime->auth->discard(*parked);
    retained.reset();
    unregisterEndpoint(endpoint);
    runtime.reset();
    assert(omniUsdLiveShutdown(3000) == OMNI_USD_LIVE_OK);
}

// Design 6.2 step 6: a wait that has reached or passed its operation deadline
// without a usable credential ends with the auth error; with a usable
// credential it ends with TIMEOUT. Equal request and auth-wait deadlines
// share one expiry instant.
void testExpiredWaitCause()
{
    using Clock = std::chrono::steady_clock;
    const auto start = Clock::time_point{} + std::chrono::hours(1);
    const auto request = start + std::chrono::milliseconds(30000);
    assert(expiredWait(request - std::chrono::nanoseconds(1), request, true) == OMNI_USD_LIVE_OK);
    assert(expiredWait(request, request, true) == OMNI_USD_LIVE_UNAUTHENTICATED);
    assert(expiredWait(request + std::chrono::nanoseconds(1), request, true) == OMNI_USD_LIVE_UNAUTHENTICATED);
    assert(expiredWait(request, request, false) == OMNI_USD_LIVE_TIMEOUT);
    assert(expiredWait(request + std::chrono::hours(1), request, false) == OMNI_USD_LIVE_TIMEOUT);
}
// Final unregister never waits for dependents: with one it returns BUSY at
// once whatever the timeout (Live.h), keeping the reference; after the
// dependent releases, a later call succeeds.
void testFinalUnregisterDoesNotWait()
{
    auto options = record<OmniUsdLiveRuntimeOptions>();
    options.flags = OMNI_USD_LIVE_RUNTIME_DISABLE_DIAGNOSTIC_HISTORY;
    assert(omniUsdLiveInitialize(&options) == OMNI_USD_LIVE_OK);
    auto endpointOptions = record<OmniUsdLiveEndpointOptions>();
    endpointOptions.authority = "unregister-busy.example.invalid";
    endpointOptions.access_token = "synthetic-initial";
    OmniUsdLiveEndpoint endpoint = 0;
    assert(omniUsdLiveRegisterEndpoint(&endpointOptions, &endpoint) == OMNI_USD_LIVE_OK);
    auto runtime = getRuntime();
    auto lease = runtime->acquireEndpoint(endpoint); // Stands in for a live layer, session or request.
    assert(lease);
    // A waiting implementation would exceed this test's CTest timeout.
    assert(omniUsdLiveUnregisterEndpoint(endpoint, UINT32_MAX) == OMNI_USD_LIVE_BUSY);
    assert(runtime->findEndpoint(endpoint)); // The reference is retained for retry.
    lease = {};
    assert(omniUsdLiveUnregisterEndpoint(endpoint, 0) == OMNI_USD_LIVE_OK);
    assert(!runtime->findEndpoint(endpoint));
    runtime.reset();
    assert(omniUsdLiveShutdown(3000) == OMNI_USD_LIVE_OK);
}
// A reconnect wait for attempt n is uniform in [0, 2 × min(maximum, initial ×
// 2^(n−1))): every draw lies in that window, a fixed population of subjects
// averages the nominal step, and the draws reach both ends of the window.
// The inputs are fixed, so these population checks are exact, not statistical.
void testReconnectDelay()
{
    constexpr uint32_t initial = 1000, maximum = 60000;
    constexpr uint64_t subjects = 4096;
    for (uint64_t attempt = 1; attempt <= 16; ++attempt) {
        const uint64_t nominal = std::min<uint64_t>(uint64_t(initial) << (attempt - 1), maximum);
        uint64_t sum = 0, lowest = UINT64_MAX, highest = 0;
        for (uint64_t subject = 1; subject <= subjects; ++subject) {
            const auto delay = reconnectDelay("seed-118", subject, attempt, initial, maximum);
            assert(delay == reconnectDelay("seed-118", subject, attempt, initial, maximum));
            assert(delay < 2 * nominal);
            sum += delay;
            lowest = std::min(lowest, delay);
            highest = std::max(highest, delay);
        }
        const auto mean = sum / subjects;
        assert(mean * 100 >= nominal * 97 && mean * 100 <= nominal * 103);
        assert(lowest * 100 < 2 * nominal && highest * 100 > 2 * nominal * 99);
    }
    // Layers of one runtime draw independently: subjects and runtime seeds
    // each change the schedule, and a schedule depends on its attempt.
    unsigned distinctSubjects = 0, distinctSeeds = 0, distinctAttempts = 0;
    for (uint64_t subject = 1; subject <= 64; ++subject) {
        distinctSubjects += reconnectDelay("seed-118", subject, 1, initial, maximum) !=
            reconnectDelay("seed-118", subject + 1, 1, initial, maximum);
        distinctSeeds += reconnectDelay("seed-118-a", subject, 3, initial, maximum) !=
            reconnectDelay("seed-118-b", subject, 3, initial, maximum);
        distinctAttempts += reconnectDelay("seed-118", subject, 7, initial, maximum) !=
            reconnectDelay("seed-118", subject, 8, initial, maximum);
    }
    assert(distinctSubjects >= 60 && distinctSeeds >= 60 && distinctAttempts >= 60);
    // The window never overflows, whatever the attempt and bounds.
    for (uint64_t attempt : {uint64_t(1), uint64_t(33), uint64_t(64), uint64_t(1000), UINT64_MAX}) {
        assert(reconnectDelay("seed-118", 9, attempt, UINT32_MAX, UINT32_MAX) < 2 * uint64_t(UINT32_MAX));
        assert(reconnectDelay("seed-118", 9, attempt, 1, UINT32_MAX) < 2 * uint64_t(UINT32_MAX));
    }
    assert(reconnectDelay("seed-118", 9, 1, 1, 1) < 2);
}
// A reissue after a refusal with a service retry delay waits
// min(max(hint, the attempt's reconnect draw), maximum): the hint is the least
// wait below the cap, the reissue keeps its own backoff when that is longer,
// and nothing is added on top of the hint.
void testHintedReconnectDelay()
{
    constexpr uint32_t initial = 1000, maximum = 60000;
    constexpr uint64_t subjects = 4096;
    for (const uint64_t hint :
        {uint64_t(0), uint64_t(1), uint64_t(250), uint64_t(1500), uint64_t(15000), uint64_t(30000)}) {
        for (uint64_t attempt = 1; attempt <= 8; ++attempt) {
            const uint64_t window = 2 * std::min<uint64_t>(uint64_t(initial) << (attempt - 1), maximum);
            uint64_t atHint = 0, aboveHint = 0;
            for (uint64_t subject = 1; subject <= subjects; ++subject) {
                const auto delay = hintedReconnectDelay("seed-128", subject, attempt, hint, initial, maximum);
                const auto own = reconnectDelay("seed-128", subject, attempt, initial, maximum);
                assert(delay == hintedReconnectDelay("seed-128", subject, attempt, hint, initial, maximum));
                assert(delay == std::min<uint64_t>(std::max(hint, own), maximum));
                assert(delay >= hint && delay <= maximum);
                atHint += delay == hint;
                aboveHint += delay > hint;
            }
            // A hint at or above the attempt's window is the wait exactly; a
            // shorter one leaves the longer draws of the window in place.
            if (hint >= window) assert(atHint == subjects);
            if (hint * 2 < window) assert(aboveHint > 0);
        }
    }
    // A hint at or above the maximum waits the maximum, and so does a draw
    // above it.
    for (const uint64_t hint : {uint64_t(maximum), uint64_t(maximum) + 1, UINT64_MAX})
        for (uint64_t attempt = 1; attempt <= 16; ++attempt)
            assert(hintedReconnectDelay("seed-128", 9, attempt, hint, initial, maximum) == maximum);
    uint64_t capped = 0;
    for (uint64_t subject = 1; subject <= subjects; ++subject) {
        const auto delay = hintedReconnectDelay("seed-128", subject, 16, 1, initial, maximum);
        assert(delay <= maximum);
        capped += delay == maximum;
    }
    assert(capped > 0);
    assert(hintedReconnectDelay("seed-128", 9, 1, 5, 0, 0) == 0);
}
}

void testAuthController()
{
    testAuthFailureWake();
    testAuthDeadlineSelection();
    testCompletionWakesOnlyOnAuthChange();
    testTokenExpiryIgnoresWallClockStepBack();
    testExpiredWaitCause();
    testFinalUnregisterDoesNotWait();
    // Equal request histories have different retry schedules across runtime
    // seeds; a saved seed reproduces every delay without statistical assertions.
    bool different = false;
    for (uint64_t attempt = 1; attempt <= 16; ++attempt) {
        const auto first = retryDelay("seed-42-client-a", 1, attempt, 100, 1000);
        const auto second = retryDelay("seed-42-client-b", 1, attempt, 100, 1000);
        assert(first == retryDelay("seed-42-client-a", 1, attempt, 100, 1000));
        const auto ceiling = std::min<uint64_t>(100ULL << std::min<uint64_t>(attempt - 1, 4), 1000);
        assert(first >= ceiling / 2 && first <= ceiling);
        different |= first != second;
    }
    assert(different);
    testReconnectDelay();
    testHintedReconnectDelay();
    auto options = record<OmniUsdLiveRuntimeOptions>();
    options.max_auth_callbacks = 2;
    assert(omniUsdLiveInitialize(&options) == OMNI_USD_LIVE_OK);
    auto endpointOptions = record<OmniUsdLiveEndpointOptions>();
    endpointOptions.authority = "auth.example.invalid";
    OmniUsdLiveEndpoint endpoint = 0;
    assert(omniUsdLiveRegisterEndpoint(&endpointOptions, &endpoint) == OMNI_USD_LIVE_OK);
    auto runtime = getRuntime();
    auto retained = runtime->findEndpoint(endpoint);
    Observer first, second;
    assert(omniUsdLiveRegisterAuthRequiredCallback(endpoint, &first, Observer::callback, &first.handle) == OMNI_USD_LIVE_OK);
    auto initial = first.wait(1);
    assert(initial.rejected_generation == 0 && initial.reason == OMNI_USD_LIVE_AUTH_MISSING);
    assert(!initial.grpc_status_present);
    assert(omniUsdLiveRegisterAuthRequiredCallback(endpoint, &second, Observer::callback, &second.handle) == OMNI_USD_LIVE_OK);
    assert(second.wait(1).challenge_id == initial.challenge_id);
    OmniUsdLiveCallbackHandle overflow = 99;
    assert(omniUsdLiveRegisterAuthRequiredCallback(endpoint, &second, Observer::callback, &overflow) == OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    assert(!overflow);
    assert(omniUsdLiveUnregisterEndpoint(endpoint, 0) == OMNI_USD_LIVE_BUSY);
    assert(omniUsdLiveUnregisterCallback(endpoint, 0) == OMNI_USD_LIVE_INVALID_HANDLE);

    std::atomic<uint32_t> winners{0}, stale{0};
    std::mutex readyMutex;
    std::condition_variable readyChanged;
    unsigned readyCount = 0;
    std::promise<void> start;
    auto startGate = start.get_future().share();
    std::vector<std::thread> workers;
    for (unsigned i = 0; i < 16; ++i) {
        workers.emplace_back([&, i] {
            {
                std::lock_guard<std::mutex> lock(readyMutex);
                ++readyCount;
                readyChanged.notify_all();
            }
            startGate.wait();
            const auto token = "synthetic-" + std::to_string(i);
            uint64_t generation = 99;
            auto result = omniUsdLiveSetAccessToken(endpoint, 0, token.c_str(), 0, &generation);
            if (result == OMNI_USD_LIVE_OK) {
                assert(generation == 1);
                ++winners;
            } else {
                assert(result == OMNI_USD_LIVE_STALE_TOKEN_GENERATION && !generation);
                ++stale;
            }
        });
    }
    {
        std::unique_lock<std::mutex> lock(readyMutex);
        assert(readyChanged.wait_for(lock, std::chrono::seconds(3), [&] { return readyCount == 16; }));
    }
    start.set_value();
    for (auto& worker : workers)
        worker.join();
    assert(winners == 1 && stale == 15);
    assert(runtime->diagnostics->snapshot(0, 0).metrics.authStaleReplacements == 15);
    assert(state(endpoint).generation == 1 && !state(endpoint).challenge_id);
    assert(omniUsdLiveFailAuthRefresh(endpoint, initial.challenge_id) == OMNI_USD_LIVE_NOT_FOUND);
    runtime->auth->completed(*retained, 0, grpc::StatusCode::UNAUTHENTICATED);
    assert(!state(endpoint).challenge_id); // Late old-generation rejection cannot challenge the new token.
    runtime->auth->completed(*retained, 1, grpc::StatusCode::PERMISSION_DENIED);
    assert(!state(endpoint).challenge_id);
    runtime->auth->completed(*retained, 1, grpc::StatusCode::UNAUTHENTICATED);
    auto rejected = first.wait(2);
    second.wait(2);
    assert(rejected.grpc_status_present && rejected.grpc_status == 16 && rejected.reason == OMNI_USD_LIVE_AUTH_REJECTED);
    for (int i = 0; i < 20; ++i)
        runtime->auth->completed(*retained, 1, grpc::StatusCode::UNAUTHENTICATED);
    assert(state(endpoint).challenge_id == rejected.challenge_id);
    assert(omniUsdLiveFailAuthRefresh(endpoint, rejected.challenge_id) == OMNI_USD_LIVE_OK);
    assert(state(endpoint).state == OMNI_USD_LIVE_STATE_FAILED);
    assert(omniUsdLiveFailAuthRefresh(endpoint, rejected.challenge_id) == OMNI_USD_LIVE_NOT_FOUND);
    assert(runtime->diagnostics->snapshot(0, 0).metrics.authFailuresByCause[0] == 1);
    auto generation = replace(endpoint, 1);
    runtime->auth->completed(*retained, generation, grpc::StatusCode::UNAUTHENTICATED);
    first.wait(3); second.wait(3);
    generation = replace(endpoint, generation);
    runtime->auth->completed(*retained, generation, grpc::StatusCode::UNAUTHENTICATED);
    assert(state(endpoint).state == OMNI_USD_LIVE_STATE_FAILED); // Two replacement generations were rejected.
    assert(runtime->diagnostics->snapshot(0, 0).metrics.authFailuresByCause[2] == 1);
    generation = replace(endpoint, generation);
    runtime->auth->completed(*retained, generation, grpc::StatusCode::OK);
    {
        std::lock_guard<std::mutex> lock(retained->mutex);
        assert(!retained->rejectedGenerations);
    }
    assert(omniUsdLiveUnregisterCallback(first.handle, 3000) == OMNI_USD_LIVE_OK);
    assert(omniUsdLiveUnregisterCallback(second.handle, 3000) == OMNI_USD_LIVE_OK);
    assert(first.events.size() == 3 && second.events.size() == 3);

    // Actual scheduler wakeup: do not rely only on the deterministic clock seam.
    Observer expiry;
    assert(omniUsdLiveRegisterAuthRequiredCallback(endpoint, &expiry, Observer::callback, &expiry.handle) == OMNI_USD_LIVE_OK);
    uint64_t updated = 0;
    assert(omniUsdLiveSetAccessToken(endpoint, generation, "synthetic-expiring", wallTimeMs() + 1000, &updated) == OMNI_USD_LIVE_OK);
    generation = updated;
    assert(expiry.wait(1).reason == OMNI_USD_LIVE_AUTH_EXPIRING);
    assert(omniUsdLiveUnregisterCallback(expiry.handle, 3000) == OMNI_USD_LIVE_OK);
    generation = replace(endpoint, generation);

    // A fake clock transition under the endpoint lock has no timing/sleep race.
    {
        std::lock_guard<std::mutex> lock(retained->mutex);
        const auto wall = wallTimeMs();
        const auto now = std::chrono::steady_clock::now();
        retained->expiresAtMs = wall + 1000;
        runtime->auth->evaluate(*retained, wall, now);
        assert(retained->challenge.reason == OMNI_USD_LIVE_AUTH_EXPIRING);
        assert(runtime->auth->usable(*retained, wall));
        assert(retained->authDeadline == std::chrono::steady_clock::time_point{});
        // Proactive refresh cannot expire an auth wait before protected calls block.
        runtime->auth->evaluate(*retained, wall + 999, now + std::chrono::hours(1));
        assert(!retained->authFailed);
        runtime->auth->evaluate(*retained, wall + 1000, now + std::chrono::hours(1));
        assert(!retained->authFailed && !runtime->auth->usable(*retained, wall + 1000));
        runtime->auth->evaluate(*retained, wall + 1001, retained->authDeadline);
        assert(retained->authFailed);
    }
    generation = replace(endpoint, generation);
    Observer self;
    self.selfCancel = true;
    self.unfinished = runtime->requests->create(endpoint, OMNI_USD_LIVE_OPERATION_LIST, {});
    assert(omniUsdLiveRegisterAuthRequiredCallback(endpoint, &self, Observer::callback, &self.handle) == OMNI_USD_LIVE_OK);
    runtime->auth->completed(*retained, generation, grpc::StatusCode::UNAUTHENTICATED);
    self.wait(1);
    assert(omniUsdLiveUnregisterCallback(self.handle, 3000) == OMNI_USD_LIVE_OK);
    self.unfinished->finish({}, {});
    assert(runtime->requests->release(self.unfinished->handle) == OMNI_USD_LIVE_OK);
    self.unfinished.reset();
    generation = replace(endpoint, generation);
    Observer blocked;
    blocked.block = true;
    assert(omniUsdLiveRegisterAuthRequiredCallback(endpoint, &blocked, Observer::callback, &blocked.handle) == OMNI_USD_LIVE_OK);
    runtime->auth->completed(*retained, generation, grpc::StatusCode::UNAUTHENTICATED);
    blocked.wait(1);
    assert(omniUsdLiveUnregisterCallback(blocked.handle, 0) == OMNI_USD_LIVE_TIMEOUT);
    auto other = runtime->requests->create(endpoint, OMNI_USD_LIVE_OPERATION_LIST, {});
    std::atomic<bool> delivered{false};
    other->finish({}, [&] { delivered = true; });
    assert(other->wait(2000) == OMNI_USD_LIVE_OK && delivered);
    assert(runtime->requests->release(other->handle) == OMNI_USD_LIVE_OK);
    other.reset();
    {
        std::lock_guard<std::mutex> lock(blocked.mutex);
        blocked.release = true;
        blocked.changed.notify_all();
    }
    assert(omniUsdLiveUnregisterCallback(blocked.handle, 3000) == OMNI_USD_LIVE_OK);
    // Unregister must remove endpoint-list ownership before the last observer
    // can die. Hold the endpoint lock while unregistration reaches detach, then
    // publish a challenge to the cancelled observer under that same lock.
    for (unsigned i = 0; i < 20; ++i) {
        generation = replace(endpoint, generation);
        Observer racing;
        assert(omniUsdLiveRegisterAuthRequiredCallback(endpoint, &racing, Observer::callback, &racing.handle) == OMNI_USD_LIVE_OK);
        std::unique_lock<std::mutex> lock(retained->mutex);
        assert(retained->authObservers.size() == 1);
        assert(retained->authObservers.front()->cancel(0) == OMNI_USD_LIVE_OK);
        std::promise<void> entered;
        auto enteredFuture = entered.get_future();
        auto removal = std::async(std::launch::async, [&] {
            return runtime->auth->unsubscribe(racing.handle, 3000, [&] { entered.set_value(); });
        });
        enteredFuture.wait();
        assert(removal.wait_for(std::chrono::milliseconds(0)) == std::future_status::timeout);
        retained->token.clear();
        runtime->auth->evaluate(*retained, wallTimeMs(), std::chrono::steady_clock::now());
        assert(retained->authObservers.size() == 1 && retained->challenge.challenge_id);
        lock.unlock();
        assert(removal.wait_for(std::chrono::seconds(3)) == std::future_status::ready);
        assert(removal.get() == OMNI_USD_LIVE_OK);
        assert(racing.events.empty());
    }
    uint64_t unchanged = 99;
    assert(omniUsdLiveSetAccessToken(endpoint, generation, "bad token", 0, &unchanged) == OMNI_USD_LIVE_INVALID_ARGUMENT && !unchanged);
    assert(state(endpoint).generation == generation);
    struct Extended { OmniUsdLiveAuthState state; uint64_t tail; } extended{record<OmniUsdLiveAuthState>(), 0xfeed};
    extended.state.struct_size = sizeof(extended);
    assert(omniUsdLiveGetAuthState(endpoint, &extended.state) == OMNI_USD_LIVE_OK && extended.tail == 0xfeed);
    uint32_t shortRecord = sizeof(uint32_t);
    assert(omniUsdLiveGetAuthState(endpoint, reinterpret_cast<OmniUsdLiveAuthState*>(&shortRecord)) == OMNI_USD_LIVE_INVALID_ARGUMENT);
    // A continuously due operation must not monopolize the timing worker.
    // Gate its first step, introduce expiry without waking the scheduler, and
    // require an auth callback before the hot operation is permitted to finish.
    struct Hot final : ScheduledOperation {
        std::promise<void> entered, release;
        std::shared_future<void> gate = release.get_future().share();
        std::atomic<bool> stop{false};
        std::promise<void> finished;
        unsigned calls = 0;
        ScheduledStep advance(std::chrono::steady_clock::time_point now) noexcept override
        {
            if (!calls++) { entered.set_value(); gate.wait(); }
            if (stop) { finished.set_value(); return {true, {}}; }
            return {false, now};
        }
    };
    generation = replace(endpoint, generation);
    Observer timerObserver;
    assert(omniUsdLiveRegisterAuthRequiredCallback(endpoint, &timerObserver, Observer::callback, &timerObserver.handle) == OMNI_USD_LIVE_OK);
    auto hot = std::make_shared<Hot>();
    auto hotEntered = hot->entered.get_future();
    auto hotFinished = hot->finished.get_future();
    runtime->auth->reserve(hot);
    runtime->auth->activate(*hot);
    assert(hotEntered.wait_for(std::chrono::seconds(3)) == std::future_status::ready);
    {
        std::lock_guard<std::mutex> lock(retained->mutex);
        retained->expiresAtMs = wallTimeMs() - 1;
    }
    hot->release.set_value();
    assert(timerObserver.wait(1).reason == OMNI_USD_LIVE_AUTH_EXPIRING);
    hot->stop = true;
    runtime->auth->wake(*hot);
    assert(hotFinished.wait_for(std::chrono::seconds(3)) == std::future_status::ready);
    assert(omniUsdLiveUnregisterCallback(timerObserver.handle, 3000) == OMNI_USD_LIVE_OK);
    // A worker publishing an old deadline must not erase a targeted/global
    // auth wake, even when the timing worker has not observed that wake yet.
    generation = replace(endpoint, generation);
    struct DeadlineRace final : ScheduledOperation {
        std::promise<void> entered, release, resumed;
        std::shared_future<void> gate = release.get_future().share();
        unsigned calls = 0;
        ScheduledStep advance(std::chrono::steady_clock::time_point) noexcept override
        {
            if (!calls++) {
                entered.set_value(); gate.wait();
                return {false, std::chrono::steady_clock::time_point::max()};
            }
            resumed.set_value(); return {true, {}};
        }
    };
    for (bool global : {false, true}) {
        auto operation = std::make_shared<DeadlineRace>();
        auto entered = operation->entered.get_future(), resumed = operation->resumed.get_future();
        runtime->auth->reserve(operation); runtime->auth->activate(*operation);
        assert(entered.wait_for(std::chrono::seconds(3)) == std::future_status::ready);
        auto observed = operation->wakeRevision();
        if (global) runtime->auth->wake(); else runtime->auth->wake(*operation);
        runtime->auth->schedule(*operation, std::chrono::steady_clock::time_point::max(), observed);
        operation->release.set_value();
        assert(resumed.wait_for(std::chrono::seconds(3)) == std::future_status::ready);
    }
    retained.reset();
    unregisterEndpoint(endpoint);
    assert(omniUsdLiveSetAccessToken(endpoint, generation, "late-token", 0, &unchanged) == OMNI_USD_LIVE_INVALID_HANDLE);
    runtime.reset();
    assert(omniUsdLiveShutdown(3000) == OMNI_USD_LIVE_OK);
}

void testAuthDiagnostics()
{
    using Clock = std::chrono::steady_clock;
    auto options = record<OmniUsdLiveRuntimeOptions>();
    options.flags = OMNI_USD_LIVE_RUNTIME_DISABLE_DIAGNOSTIC_HISTORY;
    options.default_auth_wait_timeout_ms = 5000;
    assert(omniUsdLiveInitialize(&options) == OMNI_USD_LIVE_OK);
    auto endpointOptions = record<OmniUsdLiveEndpointOptions>();
    endpointOptions.authority = "auth-metrics.example.invalid";
    endpointOptions.access_token = "synthetic-initial";
    OmniUsdLiveEndpoint endpoint = 0;
    assert(omniUsdLiveRegisterEndpoint(&endpointOptions, &endpoint) == OMNI_USD_LIVE_OK);
    auto runtime = getRuntime();
    // This controller test supplies both clocks explicitly. Quiesce its timer
    // using the scheduler's stopped event so real evaluations cannot race the
    // synthetic histories; reference cases exercise the running scheduler.
    runtime->auth->beginShutdown();
    assert(runtime->auth->waitForShutdown(Clock::now() + std::chrono::seconds(3)));
    auto retained = runtime->findEndpoint(endpoint);
    auto metrics = [&] { return runtime->diagnostics->snapshot(0, 0).metrics; };
    const auto start = Clock::time_point{}; // Zero is a valid observation time.
    constexpr uint64_t wall = 100000;
    {
        std::lock_guard<std::mutex> lock(retained->mutex);
        retained->expiresAtMs = wall + 100;
        runtime->auth->evaluate(*retained, wall, start);
        assert(retained->challenge.reason == OMNI_USD_LIVE_AUTH_EXPIRING);
        assert(metrics().authChallengesByReason[2] == 1 && metrics().authBlockedEndpoints == 0);
        retained->expiresAtMs = 0;
        retained->token.clear();
        runtime->auth->evaluate(*retained, wall, start);
        runtime->auth->evaluate(*retained, wall, start + std::chrono::milliseconds(5));
        assert(metrics().authBlockedEndpoints == 1 && metrics().authBlockedEndpointsHighWater == 1);
        retained->token = "synthetic-usable";
        runtime->auth->evaluate(*retained, wall, start + std::chrono::milliseconds(6));
        const auto measured = metrics();
        assert(measured.authBlockedEndpoints == 0 && measured.authWaitByResolution[0].count == 1);
        assert(measured.authWaitByResolution[0].sumMs == 6 && measured.authWaitByResolution[0].counts[2] == 1);
        assert(measured.rpcAttempts == 0 && retained->status.state != OMNI_USD_LIVE_STATE_READY);
        // Reset only the fixture's synthetic policy deadline before exposing
        // it to the real scheduler; production policy is unchanged.
        retained->authDeadline = {};
        retained->challenge = {};
        retained->expiresAtMs = wall - 1;
        runtime->auth->evaluate(*retained, wall, start);
        assert(metrics().authChallengesByReason[2] == 2);
        // A wall-clock reversal can make the token locally usable, but the
        // existing expired monotonic auth deadline still wins.
        runtime->auth->evaluate(*retained, wall - 2, retained->authDeadline);
        runtime->auth->evaluate(*retained, wall, start + std::chrono::seconds(10));
        const auto failed = metrics();
        assert(retained->authFailed && failed.authBlockedEndpoints == 0);
        assert(failed.authFailuresByCause[1] == 1 && failed.authWaitByResolution[1].count == 1);
        assert(failed.authWaitByResolution[1].sumMs == 5000 && failed.authWaitByResolution[1].counts[7] == 1);
    }
    auto generation = replace(endpoint, 1);
    {
        std::lock_guard<std::mutex> lock(retained->mutex);
        retained->expiresAtMs = wall - 1;
        runtime->auth->evaluate(*retained, wall, start);
    }
    // Late successful RPC, backwards wall time and expired monotonic deadline:
    // terminal failure wins here just as it does in evaluate(), without sleep.
    runtime->auth->completedAt(*retained, generation, grpc::StatusCode::OK,
                               wall - 2, start + std::chrono::milliseconds(5000));
    {
        std::lock_guard<std::mutex> lock(retained->mutex);
        assert(retained->authFailed && retained->status.state != OMNI_USD_LIVE_STATE_READY);
        const auto failed = metrics();
        assert(failed.authBlockedEndpoints == 0 && failed.authFailuresByCause[1] == 2);
        assert(failed.authWaitByResolution[0].count == 1 && failed.authWaitByResolution[1].count == 2);
        assert(failed.authWaitByResolution[1].sumMs == 10000);
    }
    generation = replace(endpoint, generation);
    uint64_t proactiveChallenge = 0;
    {
        std::lock_guard<std::mutex> lock(retained->mutex);
        const auto actualWall = wallTimeMs();
        retained->expiresAtMs = actualWall + 1;
        runtime->auth->evaluate(*retained, actualWall, Clock::now());
        proactiveChallenge = retained->challenge.challenge_id;
        retained->expiresAtMs = 0; // Keep the token usable outside this clock-controlled step.
    }
    assert(proactiveChallenge && omniUsdLiveFailAuthRefresh(endpoint, proactiveChallenge) == OMNI_USD_LIVE_OK);
    assert(metrics().authFailuresByCause[0] == 1 && metrics().authWaitByResolution[1].count == 2);
    assert(omniUsdLiveFailAuthRefresh(endpoint, proactiveChallenge) == OMNI_USD_LIVE_NOT_FOUND);
    assert(metrics().authFailuresByCause[0] == 1);
    generation = replace(endpoint, generation);
    {
        std::lock_guard<std::mutex> lock(retained->mutex);
        retained->token.clear();
        runtime->auth->evaluate(*retained, wallTimeMs(), Clock::now());
    }
    generation = replace(endpoint, generation, "");
    assert(metrics().authBlockedEndpoints == 1 && metrics().authWaitByResolution[0].count == 1);
    uint64_t updated = 0;
    assert(omniUsdLiveSetAccessToken(endpoint, generation, "synthetic-expired", 1, &updated) == OMNI_USD_LIVE_OK);
    generation = updated;
    assert(metrics().authBlockedEndpoints == 1 && metrics().authWaitByResolution[0].count == 1);
    generation = replace(endpoint, generation);
    assert(metrics().authBlockedEndpoints == 0 && metrics().authWaitByResolution[0].count == 2);
    // Final unregister, not owner release or BUSY, retires an active interval.
    OmniUsdLiveEndpoint duplicate = 0;
    assert(omniUsdLiveRegisterEndpoint(&endpointOptions, &duplicate) == OMNI_USD_LIVE_OK && duplicate == endpoint);
    auto lease = runtime->acquireEndpoint(endpoint);
    {
        std::lock_guard<std::mutex> lock(retained->mutex);
        retained->token.clear();
        runtime->auth->evaluate(*retained, wallTimeMs(), Clock::now());
    }
    assert(omniUsdLiveUnregisterEndpoint(endpoint, 0) == OMNI_USD_LIVE_OK);
    assert(metrics().authBlockedEndpoints == 1 && metrics().authWaitByResolution[2].count == 0);
    assert(omniUsdLiveUnregisterEndpoint(endpoint, 0) == OMNI_USD_LIVE_BUSY);
    assert(metrics().authBlockedEndpoints == 1 && metrics().authWaitByResolution[2].count == 0);
    lease = {};
    assert(omniUsdLiveUnregisterEndpoint(endpoint, 0) == OMNI_USD_LIVE_OK);
    assert(metrics().authBlockedEndpoints == 0 && metrics().authWaitByResolution[2].count == 1);
    assert(runtime->diagnostics->snapshot(0, 100).records.empty());
    retained.reset(); runtime.reset();
    assert(omniUsdLiveShutdown(3000) == OMNI_USD_LIVE_OK);
}
