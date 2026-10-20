// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "LiveRequest.h"
#include "LiveDiagnostics.h"
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <mutex>

using namespace omni::usd::live;
namespace {
template <class T> T record()
{
    T value{};
    value.struct_size = sizeof(value);
    value.abi_version = OMNI_USD_LIVE_ABI_VERSION;
    return value;
}

struct Fixture {
    std::shared_ptr<Runtime> runtime;
    OmniUsdLiveEndpoint endpoint = 0;
    explicit Fixture(uint32_t workers = 2, uint32_t limit = 32, bool historyDisabled = false)
    {
        auto options = record<OmniUsdLiveRuntimeOptions>();
        options.callback_worker_count = workers;
        options.max_requests = limit;
        options.flags = historyDisabled ? OMNI_USD_LIVE_RUNTIME_DISABLE_DIAGNOSTIC_HISTORY : 0;
        assert(omniUsdLiveInitialize(&options) == OMNI_USD_LIVE_OK);
        auto endpointOptions = record<OmniUsdLiveEndpointOptions>();
        endpointOptions.authority = "localhost";
        assert(omniUsdLiveRegisterEndpoint(&endpointOptions, &endpoint) == OMNI_USD_LIVE_OK);
        runtime = getRuntime();
    }
    ~Fixture()
    {
        // A released request drops its endpoint lease synchronously, so no
        // worker still dropping its last reference is an endpoint dependent.
        assert(omniUsdLiveUnregisterEndpoint(endpoint, 0) == OMNI_USD_LIVE_OK);
        assert(omniUsdLiveShutdown(2000) == OMNI_USD_LIVE_OK);
        runtime.reset();
    }
    std::shared_ptr<Request> make(std::function<void()> cancel = {})
    {
        return runtime->requests->create(endpoint, OMNI_USD_LIVE_OPERATION_CREATE, std::move(cancel));
    }
    void release(std::shared_ptr<Request>& request)
    {
        assert(omniUsdLiveWait(request->handle, 2000) == OMNI_USD_LIVE_OK);
        assert(omniUsdLiveReleaseRequest(request->handle) == OMNI_USD_LIVE_OK);
        request.reset();
    }
};

OmniUsdLiveRequestError outcome(OmniUsdLiveResult result)
{
    auto error = record<OmniUsdLiveRequestError>();
    error.result = result;
    return error;
}

void pendingCancellationAndResult()
{
    Fixture fixture;
    std::atomic<unsigned> cancelled{0}, callbacks{0};
    auto request = fixture.make([&] { ++cancelled; });
    const auto handle = request->handle;
    OmniUsdLiveResult result = 1234;
    assert(omniUsdLiveGetRequestStatus(handle, &result) == OMNI_USD_LIVE_BUSY && result == 1234);
    assert(omniUsdLiveWait(fixture.endpoint, 0) == OMNI_USD_LIVE_INVALID_HANDLE);
    assert(omniUsdLiveWait(handle, 0) == OMNI_USD_LIVE_TIMEOUT && cancelled == 0);
    assert(omniUsdLiveCancel(handle, 0) == OMNI_USD_LIVE_TIMEOUT && cancelled == 1);
    assert(omniUsdLiveReleaseRequest(handle) == OMNI_USD_LIVE_BUSY && cancelled == 1);
    assert(omniUsdLiveUnregisterEndpoint(fixture.endpoint, 0) == OMNI_USD_LIVE_BUSY);
    auto error = outcome(OMNI_USD_LIVE_OUTCOME_UNKNOWN);
    error.grpc_status_present = 1;
    error.grpc_status = 14;
    error.attempt_id = 71;
    error.incident_id = 17;
    std::strcpy(error.detail, "secret transport detail must not escape");
    request->finish(error, [&] { ++callbacks; });
    assert(omniUsdLiveWait(handle, 2000) == OMNI_USD_LIVE_OK && callbacks == 0);
    assert(omniUsdLiveGetRequestStatus(handle, &result) == OMNI_USD_LIVE_OK && result == OMNI_USD_LIVE_OUTCOME_UNKNOWN);
    struct Extended { OmniUsdLiveRequestError error; uint64_t tail; } output{};
    output.error = record<OmniUsdLiveRequestError>();
    output.error.struct_size = sizeof(output);
    output.tail = 9876;
    assert(omniUsdLiveGetRequestError(handle, &output.error) == OMNI_USD_LIVE_OK);
    assert(output.tail == 9876 && output.error.struct_size == sizeof(output));
    assert(output.error.request_id == handle && output.error.operation == OMNI_USD_LIVE_OPERATION_CREATE);
    assert(output.error.grpc_status == 14 && output.error.grpc_status_present == 1);
    assert(output.error.incident_id == 17 && output.error.attempt_id == 71 && output.error.detail[0] == 0);
    // A completed request depends on its endpoint until its handle is released.
    assert(omniUsdLiveUnregisterEndpoint(fixture.endpoint, 0) == OMNI_USD_LIVE_BUSY);
    assert(omniUsdLiveReleaseRequest(handle) == OMNI_USD_LIVE_OK);
    assert(omniUsdLiveGetRequestStatus(handle, &result) == OMNI_USD_LIVE_INVALID_HANDLE);
    // A transport reference retained after handle release is no endpoint dependent.
    {
        auto retained = fixture.runtime->findEndpoint(fixture.endpoint);
        std::lock_guard<std::mutex> lock(retained->mutex);
        assert(retained->dependents == 0);
    }
    request.reset();
}

void callbackQuiescenceAndReentry()
{
    Fixture fixture;
    auto request = fixture.make();
    std::promise<void> entered, resume;
    auto ready = entered.get_future();
    auto resumeFuture = resume.get_future();
    request->finish(outcome(OMNI_USD_LIVE_PERMISSION_DENIED), [&] {
        assert(omniUsdLiveWait(request->handle, 0) == OMNI_USD_LIVE_BUSY);
        assert(omniUsdLiveCancel(request->handle, 0) == OMNI_USD_LIVE_BUSY);
        assert(omniUsdLiveReleaseRequest(request->handle) == OMNI_USD_LIVE_BUSY);
        auto status = record<OmniUsdLiveStatus>();
        assert(omniUsdLiveGetEndpointStatus(fixture.endpoint, &status) == OMNI_USD_LIVE_OK);
        entered.set_value();
        resumeFuture.wait();
    });
    assert(ready.wait_for(std::chrono::seconds(2)) == std::future_status::ready);
    assert(omniUsdLiveWait(request->handle, 0) == OMNI_USD_LIVE_TIMEOUT);
    assert(omniUsdLiveCancel(request->handle, 0) == OMNI_USD_LIVE_TIMEOUT);
    OmniUsdLiveResult result = 99;
    assert(omniUsdLiveGetRequestStatus(request->handle, &result) == OMNI_USD_LIVE_BUSY && result == 99);
    resume.set_value();
    assert(omniUsdLiveWait(request->handle, 2000) == OMNI_USD_LIVE_OK);
    assert(omniUsdLiveGetRequestStatus(request->handle, &result) == OMNI_USD_LIVE_OK);
    assert(result == OMNI_USD_LIVE_PERMISSION_DENIED); // Wait success is not operation success.
    fixture.release(request);
}

void queuedCancellationAndPoolWait()
{
    Fixture fixture(1);
    auto blocker = fixture.make();
    auto queued = fixture.make();
    std::promise<void> entered, resume;
    auto ready = entered.get_future();
    auto resumeFuture = resume.get_future();
    blocker->finish(outcome(OMNI_USD_LIVE_OK), [&] {
        // A worker cannot wait for another callback needing the same pool.
        assert(omniUsdLiveWait(queued->handle, 1000) == OMNI_USD_LIVE_BUSY);
        entered.set_value();
        resumeFuture.wait();
    });
    assert(ready.wait_for(std::chrono::seconds(2)) == std::future_status::ready);
    std::atomic<unsigned> calls{0};
    queued->finish(outcome(OMNI_USD_LIVE_OK), [&] { ++calls; });
    assert(omniUsdLiveCancel(queued->handle, 0) == OMNI_USD_LIVE_TIMEOUT);
    assert(omniUsdLiveWait(queued->handle, 0) == OMNI_USD_LIVE_TIMEOUT && calls == 0);
    resume.set_value();
    fixture.release(queued);
    assert(calls == 0);
    fixture.release(blocker);
}

void cancellationCompletionRace()
{
    Fixture fixture;
    for (unsigned round = 0; round < 100; ++round) {
        std::atomic<unsigned> cancelled{0}, delivered{0};
        auto request = fixture.make([&] { ++cancelled; });
        std::promise<void> start;
        auto gate = start.get_future().share();
        std::thread finish([&] {
            gate.wait();
            request->finish(outcome(OMNI_USD_LIVE_OK), [&] { ++delivered; });
        });
        std::thread cancel([&] {
            gate.wait();
            assert(omniUsdLiveCancel(request->handle, 2000) == OMNI_USD_LIVE_OK);
        });
        start.set_value();
        finish.join();
        cancel.join();
        assert(delivered <= 1 && cancelled <= 1);
        // Cancellation never changes an independently known committed outcome.
        OmniUsdLiveResult result = 99;
        assert(omniUsdLiveGetRequestStatus(request->handle, &result) == OMNI_USD_LIVE_OK && result == OMNI_USD_LIVE_OK);
        request->finish(outcome(OMNI_USD_LIVE_INTERNAL_ERROR), [&] { delivered += 100; });
        assert(omniUsdLiveWait(request->handle, 2000) == OMNI_USD_LIVE_OK);
        assert(omniUsdLiveGetRequestStatus(request->handle, &result) == OMNI_USD_LIVE_OK && result == OMNI_USD_LIVE_OK);
        assert(delivered <= 1);
        fixture.release(request);
    }
}

void capacityAndCallbackException()
{
    Fixture fixture(1, 2);
    auto first = fixture.make();
    auto second = fixture.make();
    bool rejected = false;
    try { fixture.make(); } catch (OmniUsdLiveResult result) { rejected = result == OMNI_USD_LIVE_RESOURCE_EXHAUSTED; }
    assert(rejected);
    first->finish(outcome(OMNI_USD_LIVE_OK), [] { throw 42; });
    assert(omniUsdLiveWait(first->handle, 2000) == OMNI_USD_LIVE_OK && first->callbackThrew());
    OmniUsdLiveResult result = 99;
    assert(omniUsdLiveGetRequestStatus(first->handle, &result) == OMNI_USD_LIVE_OK && result == OMNI_USD_LIVE_OK);
    std::atomic<unsigned> calls{0};
    second->finish(outcome(OMNI_USD_LIVE_OK), [&] { ++calls; });
    assert(omniUsdLiveWait(second->handle, 2000) == OMNI_USD_LIVE_OK && calls == 1);
    fixture.release(first);
    fixture.release(second);
}

struct CaptureState {
    OmniUsdLiveRequest request = 0;
    std::promise<void> entered;
    std::promise<void> resume;
};
struct HeldCapture {
    explicit HeldCapture(std::shared_ptr<CaptureState> value) : state(std::move(value)) {}
    std::shared_ptr<CaptureState> state;
    ~HeldCapture()
    {
        // Destruction is part of callback execution for self-dependency checks.
        assert(omniUsdLiveWait(state->request, 0) == OMNI_USD_LIVE_BUSY);
        state->entered.set_value();
        state->resume.get_future().wait();
    }
};

void captureCleanupIsAsynchronousAndQuiescent()
{
    Fixture fixture(1);
    auto blocker = fixture.make();
    auto queued = fixture.make();
    std::promise<void> entered, resume;
    auto ready = entered.get_future();
    auto resumeFuture = resume.get_future();
    blocker->finish(outcome(OMNI_USD_LIVE_OK), [&] {
        entered.set_value();
        resumeFuture.wait();
    });
    assert(ready.wait_for(std::chrono::seconds(2)) == std::future_status::ready);
    auto state = std::make_shared<CaptureState>();
    state->request = queued->handle;
    auto cleanupEntered = state->entered.get_future();
    auto capture = std::make_shared<HeldCapture>(state);
    std::atomic<unsigned> calls{0};
    queued->finish(outcome(OMNI_USD_LIVE_OK), [capture, &calls] { ++calls; });
    capture.reset();
    // This must return without running the held capture destructor here.
    assert(omniUsdLiveCancel(queued->handle, 0) == OMNI_USD_LIVE_TIMEOUT);
    resume.set_value();
    assert(cleanupEntered.wait_for(std::chrono::seconds(2)) == std::future_status::ready);
    assert(omniUsdLiveWait(queued->handle, 0) == OMNI_USD_LIVE_TIMEOUT);
    assert(omniUsdLiveCancel(queued->handle, 0) == OMNI_USD_LIVE_TIMEOUT);
    assert(omniUsdLiveReleaseRequest(queued->handle) == OMNI_USD_LIVE_BUSY);
    state->resume.set_value();
    fixture.release(queued);
    fixture.release(blocker);
    assert(calls == 0);
}

template <class F> void exhausted(F&& create)
{
    bool rejected = false;
    try { create(); }
    catch (OmniUsdLiveResult result) { rejected = result == OMNI_USD_LIVE_RESOURCE_EXHAUSTED; }
    assert(rejected);
}

void timingCapacityIsIndependent()
{
    Fixture fixture(1, 1, true);
    const auto slots = [&] { return fixture.runtime->diagnostics->snapshot(0, 0).metrics.requestSlots; };
    bool invalid = false;
    try {
        fixture.runtime->requests->create(fixture.endpoint, OMNI_USD_LIVE_OPERATION_CREATE, {}, true);
    } catch (const std::invalid_argument&) { invalid = true; }
    assert(invalid);
    auto timing = [&] {
        return fixture.runtime->requests->create(fixture.endpoint, OMNI_USD_LIVE_OPERATION_PING, {}, true);
    };
    // Ordinary capacity is full, but the fixed auxiliary slot remains available.
    auto ordinary = fixture.make();
    exhausted([&] { fixture.make(); });
    auto probe = timing();
    exhausted(timing);
    auto snapshot = slots();
    for (const auto& slot : snapshot)
        assert(slot.current == 1 && slot.highWater == 1 && slot.admitted == 1 && slot.rejected == 1 && slot.released == 0);
    ordinary->finish(outcome(OMNI_USD_LIVE_OK), {});
    fixture.release(ordinary);
    snapshot = slots();
    assert(snapshot[0].current == 0 && snapshot[0].released == 1 && snapshot[1].current == 1);
    // Conversely, an existing timing request consumes none of ordinary capacity.
    ordinary = fixture.make();
    exhausted([&] { fixture.make(); });
    exhausted(timing);
    probe->finish(outcome(OMNI_USD_LIVE_OK), {});
    fixture.release(probe);
    // Releasing a probe must not accidentally release an ordinary capacity slot.
    exhausted([&] { fixture.make(); });
    probe = timing();
    probe->finish(outcome(OMNI_USD_LIVE_OK), {});
    ordinary->finish(outcome(OMNI_USD_LIVE_OK), {});
    fixture.release(probe);
    fixture.release(ordinary);
    snapshot = slots();
    assert(snapshot[0].admitted == 2 && snapshot[0].released == 2 && snapshot[0].rejected == 3);
    assert(snapshot[1].admitted == 2 && snapshot[1].released == 2 && snapshot[1].rejected == 2);
    for (const auto& slot : snapshot) assert(slot.current == 0 && slot.highWater == 1);
}

void concurrentHandleReleaseCountsOnce()
{
    Fixture fixture(2, 1, true);
    auto request = fixture.make();
    request->finish(outcome(OMNI_USD_LIVE_OK), {});
    std::promise<void> release;
    const auto gate = release.get_future().share();
    std::atomic<unsigned> succeeded{0}, invalid{0};
    std::vector<std::thread> threads;
    for (unsigned i = 0; i < 16; ++i) threads.emplace_back([&] {
        gate.wait();
        const auto result = omniUsdLiveReleaseRequest(request->handle);
        if (result == OMNI_USD_LIVE_OK) ++succeeded;
        else { assert(result == OMNI_USD_LIVE_INVALID_HANDLE); ++invalid; }
    });
    release.set_value();
    for (auto& thread : threads) thread.join();
    const auto slots = fixture.runtime->diagnostics->snapshot(0, 0).metrics.requestSlots;
    assert(succeeded == 1 && invalid == 15);
    assert(slots[0].current == 0 && slots[0].admitted == 1 && slots[0].released == 1 && slots[0].rejected == 0);
    assert(slots[1].admitted == 0);
    // An internal owner that retains the released request is no endpoint dependent.
    auto retained = fixture.runtime->findEndpoint(fixture.endpoint);
    std::lock_guard<std::mutex> lock(retained->mutex);
    assert(retained->dependents == 0);
}

void timingSlotRequiresReleaseAfterQuiescence()
{
    Fixture fixture(1, 1);
    std::atomic<unsigned> cancelled{0}, calls{0};
    auto timing = [&] {
        return fixture.runtime->requests->create(fixture.endpoint, OMNI_USD_LIVE_OPERATION_PING,
                                                 [&] { ++cancelled; }, true);
    };
    auto probe = timing();
    auto blocker = fixture.make();
    std::promise<void> entered, resume;
    auto ready = entered.get_future();
    auto resumeFuture = resume.get_future();
    blocker->finish(outcome(OMNI_USD_LIVE_OK), [&] {
        entered.set_value();
        resumeFuture.wait();
    });
    assert(ready.wait_for(std::chrono::seconds(2)) == std::future_status::ready);
    // Cancellation does not establish transport completion or free the slot.
    assert(omniUsdLiveCancel(probe->handle, 0) == OMNI_USD_LIVE_TIMEOUT && cancelled == 1);
    assert(omniUsdLiveReleaseRequest(probe->handle) == OMNI_USD_LIVE_BUSY);
    assert(fixture.runtime->diagnostics->snapshot(0, 0).metrics.requestSlots[1].current == 1);
    exhausted(timing);
    auto state = std::make_shared<CaptureState>();
    state->request = probe->handle;
    auto cleanupEntered = state->entered.get_future();
    auto capture = std::make_shared<HeldCapture>(state);
    probe->finish(outcome(OMNI_USD_LIVE_CANCELLED), [capture, &calls] { ++calls; });
    capture.reset();
    exhausted(timing); // Transport done, callback cleanup still queued.
    resume.set_value();
    assert(cleanupEntered.wait_for(std::chrono::seconds(2)) == std::future_status::ready);
    assert(omniUsdLiveWait(probe->handle, 0) == OMNI_USD_LIVE_TIMEOUT);
    assert(omniUsdLiveReleaseRequest(probe->handle) == OMNI_USD_LIVE_BUSY);
    exhausted(timing); // Suppressed callback's capture destructor is still running.
    assert(fixture.runtime->diagnostics->snapshot(0, 0).metrics.requestSlots[1].released == 0);
    state->resume.set_value();
    assert(omniUsdLiveWait(probe->handle, 2000) == OMNI_USD_LIVE_OK);
    assert(calls == 0 && cancelled == 1);
    exhausted(timing); // Quiescent but the public handle has not been released.
    assert(omniUsdLiveReleaseRequest(probe->handle) == OMNI_USD_LIVE_OK);
    assert(fixture.runtime->diagnostics->snapshot(0, 0).metrics.requestSlots[1].current == 0);
    auto replacement = timing();
    replacement->finish(outcome(OMNI_USD_LIVE_OK), {});
    fixture.release(replacement);
    fixture.release(blocker);
    // A released timing handle's retained transport owner is no endpoint dependent.
    {
        auto retained = fixture.runtime->findEndpoint(fixture.endpoint);
        std::lock_guard<std::mutex> lock(retained->mutex);
        assert(retained->dependents == 0);
    }
    assert(omniUsdLiveShutdown(0) == OMNI_USD_LIVE_BUSY);
    probe.reset(); // Fixture then proves endpoint removal and final shutdown succeed.
}
// Holds the completion-queue poller after the first tag's onComplete returns,
// while the poller still owns the tag, until the test releases it.
struct HeldPoller {
    std::mutex mutex;
    std::condition_variable changed;
    bool held = false, released = false;
    static void hook(void* context, CompletionTag&) noexcept
    {
        if (!context) return; // The poller can pair this hook with a cleared context.
        auto& self = *static_cast<HeldPoller*>(context);
        std::unique_lock<std::mutex> lock(self.mutex);
        if (self.held) return;
        self.held = true;
        self.changed.notify_all();
        // Failure-only watchdog: the test releases the poller explicitly.
        if (!self.changed.wait_for(lock, std::chrono::seconds(30), [&] { return self.released; })) std::abort();
    }
};

void OMNI_USD_LIVE_CALL listed(void* user, OmniUsdLiveResult result, uint64_t, const OmniUsdLiveSessionInfo* const*)
{
    static_cast<std::atomic<OmniUsdLiveResult>*>(user)->store(result);
}

void releasedFiniteRequestLeavesNoEndpointDependent()
{
    // One attempt: the refused connection finishes ListSessions UNAVAILABLE.
    auto options = record<OmniUsdLiveRuntimeOptions>();
    options.retry_max_attempts = 1;
    assert(omniUsdLiveInitialize(&options) == OMNI_USD_LIVE_OK);
    auto endpointOptions = record<OmniUsdLiveEndpointOptions>();
    endpointOptions.authority = "127.0.0.1:35495";
    endpointOptions.access_token = "synthetic-token";
    OmniUsdLiveEndpoint endpoint = 0;
    assert(omniUsdLiveRegisterEndpoint(&endpointOptions, &endpoint) == OMNI_USD_LIVE_OK);
    auto runtime = getRuntime();
    HeldPoller hold;
    runtime->setAfterCompleteForTest(HeldPoller::hook, &hold);
    std::atomic<OmniUsdLiveResult> delivered{OMNI_USD_LIVE_OK};
    OmniUsdLiveRequest request = 0;
    assert(omniUsdLiveListSessions(endpoint, "base", &delivered, listed, &request) == OMNI_USD_LIVE_OK);
    {
        std::unique_lock<std::mutex> lock(hold.mutex);
        assert(hold.changed.wait_for(lock, std::chrono::seconds(30), [&] { return hold.held; }));
    }
    // The poller still owns the attempt's tag, which retains the operation.
    assert(omniUsdLiveWait(request, 3000) == OMNI_USD_LIVE_OK);
    OmniUsdLiveResult result = OMNI_USD_LIVE_OK;
    assert(omniUsdLiveGetRequestStatus(request, &result) == OMNI_USD_LIVE_OK && result == OMNI_USD_LIVE_UNAVAILABLE);
    assert(delivered.load() == OMNI_USD_LIVE_UNAVAILABLE);
    assert(omniUsdLiveReleaseRequest(request) == OMNI_USD_LIVE_OK);
    assert(omniUsdLiveUnregisterEndpoint(endpoint, 3000) == OMNI_USD_LIVE_OK);
    runtime->setAfterCompleteForTest(nullptr, nullptr);
    {
        std::lock_guard<std::mutex> lock(hold.mutex);
        hold.released = true;
        hold.changed.notify_all();
    }
    runtime.reset();
    assert(omniUsdLiveShutdown(3000) == OMNI_USD_LIVE_OK);
}
} // namespace

void testRequestLifecycle()
{
    pendingCancellationAndResult();
    callbackQuiescenceAndReentry();
    queuedCancellationAndPoolWait();
    cancellationCompletionRace();
    capacityAndCallbackException();
    captureCleanupIsAsynchronousAndQuiescent();
    timingCapacityIsIndependent();
    concurrentHandleReleaseCountsOnce();
    timingSlotRequiresReleaseAfterQuiescence();
    releasedFiniteRequestLeavesNoEndpointDependent();
    std::cout << "request lifecycle: cancellation, callback quiescence/reentry, bounded admission, 100 completion races, released finite endpoint dependents passed\n";
}
