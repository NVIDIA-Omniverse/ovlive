// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "LiveStatus.h"
#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <functional>
#include <cstring>
#include <future>
#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <vector>

using namespace omni::usd::live;
namespace {
using Clock = std::chrono::steady_clock;
constexpr OmniUsdLiveEndpoint endpoint = 101, otherEndpoint = 202;
constexpr OmniUsdLiveSession session = 303;
const std::string uri = "omnilive://status.example/session/root.live";

OmniUsdLiveStatus value(uint32_t state = OMNI_USD_LIVE_STATE_OPENING)
{
    OmniUsdLiveStatus result{};
    result.struct_size = sizeof(result); result.abi_version = OMNI_USD_LIVE_ABI_VERSION;
    result.subject_kind = OMNI_USD_LIVE_SUBJECT_LAYER;
    result.state = state; result.operation = OMNI_USD_LIVE_OPERATION_OPEN;
    result.progress_known = 1;
    return result;
}
struct Event {
    OmniUsdLiveEndpoint endpoint;
    OmniUsdLiveSession session;
    std::string uri;
    OmniUsdLiveStatus status;
};
struct Observer {
    std::mutex mutex;
    std::condition_variable changed;
    std::vector<Event> events;
    std::function<void(const Event&)> action;
    std::atomic<unsigned> active{0};
    std::thread::id caller = std::this_thread::get_id();
    static void OMNI_USD_LIVE_CALL callback(void* user, const OmniUsdLiveStatusEvent* input)
    {
        auto& self = *static_cast<Observer*>(user);
        assert(std::this_thread::get_id() != self.caller);
        assert(input && input->status && input->struct_size == sizeof(*input));
        assert(input->abi_version == OMNI_USD_LIVE_ABI_VERSION);
        assert(input->status->struct_size == sizeof(OmniUsdLiveStatus));
        assert(input->status->abi_version == OMNI_USD_LIVE_ABI_VERSION);
        assert(self.active.fetch_add(1) == 0); // Each registration is serialized.
        Event event{input->endpoint, input->session, input->uri ? input->uri : "", *input->status};
        {
            std::lock_guard<std::mutex> lock(self.mutex);
            for (const auto& previous : self.events)
                if (previous.status.subject_id == event.status.subject_id)
                    assert(previous.status.revision < event.status.revision);
            self.events.push_back(event);
            self.changed.notify_all();
        }
        if (self.action) self.action(event);
        assert(self.active.fetch_sub(1) == 1);
    }
    template <class Predicate> void wait(Predicate predicate)
    {
        std::unique_lock<std::mutex> lock(mutex);
        assert(changed.wait_for(lock, std::chrono::seconds(3), [&] { return predicate(events); }));
    }
    std::vector<Event> copy()
    {
        std::lock_guard<std::mutex> lock(mutex);
        return events;
    }
};
struct Fixture {
    DispatchExecutor executor;
    StatusRegistry registry;
    Fixture(size_t subjects = 8, size_t observers = 8, uint32_t workers = 2)
        : executor(workers), registry(executor, subjects, observers) {}
    ~Fixture()
    {
        registry.beginShutdown();
        assert(registry.waitForShutdown(Clock::now() + std::chrono::seconds(3)));
        executor.beginShutdown();
        assert(executor.waitForShutdown(Clock::now() + std::chrono::seconds(3)));
    }
    OmniUsdLiveCallbackHandle subscribe(Observer& observer, OmniUsdLiveEndpoint filter = endpoint,
                                         std::shared_ptr<void> lease = {})
    {
        return registry.subscribe(filter, &observer, Observer::callback, std::move(lease));
    }
};
struct HeldWorker final : DispatchJob {
    std::promise<void> entered, resume;
    std::shared_future<void> gate = resume.get_future().share();
    void dispatch() noexcept override { entered.set_value(); gate.wait(); }
    static std::shared_ptr<HeldWorker> hold(DispatchExecutor& executor)
    {
        auto job = std::make_shared<HeldWorker>();
        auto entered = job->entered.get_future();
        executor.enqueue(*job);
        assert(entered.wait_for(std::chrono::seconds(3)) == std::future_status::ready);
        return job;
    }
};
OmniUsdLiveStatus snapshot(StatusRegistry& registry, uint64_t id)
{
    auto result = value();
    assert(registry.snapshot(id, &result) == OMNI_USD_LIVE_OK);
    return result;
}
template <class F> void rejected(F&& action, OmniUsdLiveResult expected)
{
    bool caught = false;
    try { action(); }
    catch (OmniUsdLiveResult result) { caught = result == expected; }
    assert(caught);
}
void same(const OmniUsdLiveStatus& actual, const OmniUsdLiveStatus& expected)
{
    assert(actual.subject_kind == expected.subject_kind && actual.subject_id == expected.subject_id);
    assert(actual.revision == expected.revision && actual.changed_at_ms == expected.changed_at_ms);
    assert(actual.state == expected.state && actual.result == expected.result && actual.operation == expected.operation);
    assert(actual.grpc_status_present == expected.grpc_status_present && actual.grpc_status == expected.grpc_status);
    assert(actual.recovery_action == expected.recovery_action && actual.progress_known == expected.progress_known);
    assert(actual.incident_id == expected.incident_id && actual.has_unconfirmed_edits == expected.has_unconfirmed_edits);
    assert(actual.pending_write_count == expected.pending_write_count && actual.pending_bytes == expected.pending_bytes);
    assert(actual.blocked_layers == expected.blocked_layers && actual.failed_layers == expected.failed_layers);
    assert(actual.blocked_sessions == expected.blocked_sessions && actual.failed_sessions == expected.failed_sessions);
    assert(actual.buffered_edit_count == expected.buffered_edit_count);
    assert(actual.read_sequence == expected.read_sequence && actual.applied_sequence == expected.applied_sequence);
}

void initialCoalescingAndCachedQueries()
{
    Fixture fixture(2, 1, 1);
    auto worker = HeldWorker::hold(fixture.executor);
    const auto id = fixture.registry.add(endpoint, session, uri, value());
    Observer observer;
    const auto registration = fixture.subscribe(observer);
    auto initial = snapshot(fixture.registry, id);
    assert(initial.subject_id == id && initial.revision && initial.state == OMNI_USD_LIVE_STATE_OPENING);
    auto ready = value(OMNI_USD_LIVE_STATE_READY);
    ready.pending_write_count = 2; ready.pending_bytes = 19;
    assert(fixture.registry.publish(id, ready) == OMNI_USD_LIVE_OK);
    auto reconnecting = ready;
    reconnecting.state = OMNI_USD_LIVE_STATE_RECONNECTING;
    reconnecting.recovery_action = OMNI_USD_LIVE_RECOVERY_WAIT_FOR_RETRY;
    assert(fixture.registry.publish(id, reconnecting) == OMNI_USD_LIVE_OK);
    auto failed = ready;
    failed.state = OMNI_USD_LIVE_STATE_FAILED;
    failed.result = OMNI_USD_LIVE_OUTCOME_UNKNOWN;
    failed.operation = OMNI_USD_LIVE_OPERATION_UPDATE;
    failed.incident_id = 700; failed.has_unconfirmed_edits = 1;
    failed.grpc_status_present = 1; failed.grpc_status = 14;
    failed.recovery_action = OMNI_USD_LIVE_RECOVERY_PRESERVE_AND_REOPEN;
    assert(fixture.registry.publish(id, failed) == OMNI_USD_LIVE_OK);
    assert(fixture.registry.publish(id, ready) == OMNI_USD_LIVE_OK);
    const auto latest = snapshot(fixture.registry, id);
    assert(latest.revision > initial.revision);
    assert(latest.state == OMNI_USD_LIVE_STATE_FAILED && latest.result == OMNI_USD_LIVE_OUTCOME_UNKNOWN);
    assert(latest.incident_id == 700 && latest.has_unconfirmed_edits && latest.grpc_status_present && latest.grpc_status == 14);
    auto byUri = value();
    assert(fixture.registry.layer(uri, &byUri) == OMNI_USD_LIVE_OK);
    same(byUri, latest);
    // Queries complete while the only callback worker is held and invoke no
    // callbacks. The registry has no Runtime, transport, or USD dependency.
    assert(observer.copy().empty());
    Observer rejectedObserver;
    rejected([&] { fixture.subscribe(rejectedObserver); }, OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    same(snapshot(fixture.registry, id), latest);
    worker->resume.set_value();
    observer.wait([&](const auto& events) { return !events.empty(); });
    assert(fixture.registry.unsubscribe(registration, 3000) == OMNI_USD_LIVE_OK);
    const auto events = observer.copy();
    assert(events.size() == 1 && events[0].endpoint == endpoint && events[0].session == session && events[0].uri == uri);
    same(events[0].status, latest);
    assert(rejectedObserver.copy().empty());
    assert(fixture.registry.unsubscribe(registration, 0) == OMNI_USD_LIVE_INVALID_HANDLE);
}

void closedReservationsAndNewIdentity()
{
    Fixture fixture(3, 2, 1);
    auto worker = HeldWorker::hold(fixture.executor);
    const auto oldId = fixture.registry.add(endpoint, session, uri, value());
    Observer observer, late;
    const auto registration = fixture.subscribe(observer);
    auto failed = value(OMNI_USD_LIVE_STATE_FAILED);
    failed.result = OMNI_USD_LIVE_MALFORMED_STREAM; failed.incident_id = 900;
    failed.has_unconfirmed_edits = 1; failed.pending_write_count = 3; failed.pending_bytes = 71;
    assert(fixture.registry.publish(oldId, failed) == OMNI_USD_LIVE_OK);
    assert(fixture.registry.remove(oldId) == OMNI_USD_LIVE_OK);
    auto output = value();
    assert(fixture.registry.snapshot(oldId, &output) == OMNI_USD_LIVE_INVALID_HANDLE);
    assert(fixture.registry.layer(uri, &output) == OMNI_USD_LIVE_NOT_FOUND);
    assert(fixture.registry.publish(oldId, value()) == OMNI_USD_LIVE_INVALID_HANDLE);
    const auto lateRegistration = fixture.subscribe(late);
    const auto second = fixture.registry.add(endpoint, session, uri + "-2", value());
    assert(fixture.registry.remove(second) == OMNI_USD_LIVE_OK);
    const auto third = fixture.registry.add(endpoint, session, uri + "-3", value());
    assert(fixture.registry.remove(third) == OMNI_USD_LIVE_OK);
    rejected([&] { fixture.registry.add(endpoint, session, uri, value()); }, OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    assert(observer.copy().empty() && late.copy().empty());
    worker->resume.set_value();
    observer.wait([](const auto& events) { return events.size() == 3; });
    late.wait([](const auto& events) { return events.size() == 2; });
    assert(fixture.registry.unsubscribe(registration, 3000) == OMNI_USD_LIVE_OK);
    const auto events = observer.copy();
    bool found = false;
    for (const auto& event : events) {
        assert(event.status.state == OMNI_USD_LIVE_STATE_CLOSED);
        if (event.status.subject_id == oldId) {
            found = true;
            assert(event.status.result == OMNI_USD_LIVE_MALFORMED_STREAM && event.status.incident_id == 900);
            assert(event.status.has_unconfirmed_edits && event.status.pending_write_count == 3 && event.status.pending_bytes == 71);
        }
    }
    assert(found);
    for (const auto& event : late.copy()) assert(event.status.subject_id != oldId);
    const auto reopened = fixture.registry.add(endpoint, session, uri, value());
    assert(reopened != oldId && reopened != second && reopened != third);
    const auto fresh = snapshot(fixture.registry, reopened);
    assert(fresh.state == OMNI_USD_LIVE_STATE_OPENING && fresh.result == OMNI_USD_LIVE_OK && !fresh.incident_id);
    late.wait([&](const auto& current) { return current.size() == 3 && current.back().status.subject_id == reopened; });
    assert(fixture.registry.unsubscribe(lateRegistration, 3000) == OMNI_USD_LIVE_OK);
}

void selfUnregisterAndIndependentRegistration()
{
    Fixture fixture(1, 2, 2);
    const auto id = fixture.registry.add(endpoint, session, uri, value());
    Observer slow, fast;
    OmniUsdLiveCallbackHandle slowHandle = 0;
    std::promise<void> entered, resume;
    auto enteredFuture = entered.get_future();
    auto gate = resume.get_future().share();
    slow.action = [&](const Event& event) {
        if (event.status.state != OMNI_USD_LIVE_STATE_READY) return;
        assert(fixture.registry.unsubscribe(slowHandle, 1000) == OMNI_USD_LIVE_BUSY);
        assert(snapshot(fixture.registry, id).state == OMNI_USD_LIVE_STATE_READY);
        entered.set_value();
        gate.wait();
    };
    slowHandle = fixture.subscribe(slow);
    slow.wait([](const auto& events) { return events.size() == 1; });
    assert(fixture.registry.publish(id, value(OMNI_USD_LIVE_STATE_READY)) == OMNI_USD_LIVE_OK);
    assert(enteredFuture.wait_for(std::chrono::seconds(3)) == std::future_status::ready);
    const auto fastHandle = fixture.subscribe(fast);
    fast.wait([](const auto& events) { return events.size() == 1 && events[0].status.state == OMNI_USD_LIVE_STATE_READY; });
    assert(fixture.registry.publish(id, value(OMNI_USD_LIVE_STATE_BLOCKED)) == OMNI_USD_LIVE_OK);
    fast.wait([](const auto& events) { return events.size() == 2 && events.back().status.state == OMNI_USD_LIVE_STATE_BLOCKED; });
    assert(fixture.registry.unsubscribe(slowHandle, 0) == OMNI_USD_LIVE_TIMEOUT);
    resume.set_value();
    assert(fixture.registry.unsubscribe(slowHandle, 3000) == OMNI_USD_LIVE_OK);
    assert(slow.copy().size() == 2);
    assert(fixture.registry.unsubscribe(slowHandle, 0) == OMNI_USD_LIVE_INVALID_HANDLE);
    assert(fixture.registry.unsubscribe(fastHandle, 3000) == OMNI_USD_LIVE_OK);
}

void closeWhileEarlierCallbackRuns()
{
    Fixture fixture(2, 1, 2);
    const auto id = fixture.registry.add(endpoint, session, uri, value());
    Observer observer;
    std::promise<void> entered, resume;
    auto enteredFuture = entered.get_future();
    auto gate = resume.get_future().share();
    observer.action = [&](const Event& event) {
        if (event.status.subject_id == id && event.status.state == OMNI_USD_LIVE_STATE_OPENING) {
            entered.set_value(); gate.wait();
        }
    };
    const auto registration = fixture.subscribe(observer);
    assert(enteredFuture.wait_for(std::chrono::seconds(3)) == std::future_status::ready);
    auto failed = value(OMNI_USD_LIVE_STATE_FAILED);
    failed.result = OMNI_USD_LIVE_OBJECT_REPLACED; failed.incident_id = 901;
    assert(fixture.registry.publish(id, failed) == OMNI_USD_LIVE_OK);
    assert(fixture.registry.remove(id) == OMNI_USD_LIVE_OK);
    const auto next = fixture.registry.add(endpoint, session, uri + "-next", value());
    assert(fixture.registry.remove(next) == OMNI_USD_LIVE_OK);
    rejected([&] { fixture.registry.add(endpoint, session, uri, value()); }, OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    assert(observer.copy().size() == 1); // The second executor worker cannot reenter this registration.
    resume.set_value();
    observer.wait([](const auto& events) { return events.size() == 3; });
    assert(fixture.registry.unsubscribe(registration, 3000) == OMNI_USD_LIVE_OK);
    bool closedOld = false, closedNew = false;
    for (const auto& event : observer.copy()) {
        if (event.status.state != OMNI_USD_LIVE_STATE_CLOSED) continue;
        if (event.status.subject_id == id) {
            closedOld = true;
            assert(event.status.result == OMNI_USD_LIVE_OBJECT_REPLACED && event.status.incident_id == 901);
        }
        if (event.status.subject_id == next) closedNew = true;
    }
    assert(closedOld && closedNew);
}

void leaseDestructionOutsideRegistryLocks()
{
    Fixture fixture(1, 2, 1);
    const auto id = fixture.registry.add(endpoint, session, uri, value());
    Observer observer;
    std::promise<void> destroyed;
    auto destroyedFuture = destroyed.get_future();
    struct Lease {
        StatusRegistry* registry;
        uint64_t id;
        std::promise<void>* destroyed;
        ~Lease()
        {
            assert(snapshot(*registry, id).subject_id == id);
            destroyed->set_value();
        }
    };
    auto lease = std::shared_ptr<Lease>(new Lease{&fixture.registry, id, &destroyed});
    const auto handle = fixture.subscribe(observer, endpoint, lease);
    lease.reset();
    observer.wait([](const auto& events) { return events.size() == 1; });
    auto removed = std::async(std::launch::async, [&] { return fixture.registry.unsubscribe(handle, 3000); });
    assert(removed.wait_for(std::chrono::seconds(3)) == std::future_status::ready);
    assert(removed.get() == OMNI_USD_LIVE_OK);
    assert(destroyedFuture.wait_for(std::chrono::seconds(3)) == std::future_status::ready);
}

void endpointFiltersAndIndependentChildCounts()
{
    Fixture fixture(4, 2, 1);
    auto worker = HeldWorker::hold(fixture.executor);
    auto parent = value(OMNI_USD_LIVE_STATE_READY);
    parent.subject_kind = OMNI_USD_LIVE_SUBJECT_ENDPOINT; parent.subject_id = endpoint;
    assert(fixture.registry.add(endpoint, 0, "", parent) == endpoint);
    const auto layer = fixture.registry.add(endpoint, session, uri, value());
    const auto elsewhere = fixture.registry.add(otherEndpoint, 404, uri + "-elsewhere", value());
    auto joined = value(OMNI_USD_LIVE_STATE_READY);
    joined.subject_kind = OMNI_USD_LIVE_SUBJECT_SESSION;
    const auto joinedId = fixture.registry.add(endpoint, session, "omnilive://status.example/session", joined);
    Observer scoped, global;
    const auto scopedHandle = fixture.subscribe(scoped);
    const auto globalHandle = fixture.subscribe(global, 0);
    assert(fixture.registry.publish(layer, value(OMNI_USD_LIVE_STATE_BLOCKED)) == OMNI_USD_LIVE_OK);
    joined.state = OMNI_USD_LIVE_STATE_FAILED;
    joined.result = OMNI_USD_LIVE_PERMISSION_DENIED;
    assert(fixture.registry.publish(joinedId, joined) == OMNI_USD_LIVE_OK);
    auto aggregate = snapshot(fixture.registry, endpoint);
    assert(aggregate.state == OMNI_USD_LIVE_STATE_READY);
    assert(aggregate.blocked_layers == 1 && aggregate.failed_layers == 0);
    assert(aggregate.blocked_sessions == 0 && aggregate.failed_sessions == 1);
    auto otherFailure = value(OMNI_USD_LIVE_STATE_FAILED);
    otherFailure.result = OMNI_USD_LIVE_MALFORMED_STREAM;
    assert(fixture.registry.publish(elsewhere, otherFailure) == OMNI_USD_LIVE_OK);
    aggregate = snapshot(fixture.registry, endpoint);
    assert(aggregate.failed_layers == 0); // Another endpoint cannot affect this aggregate.
    worker->resume.set_value();
    scoped.wait([](const auto& events) { return events.size() == 3; });
    global.wait([](const auto& events) { return events.size() == 4; });
    assert(fixture.registry.unsubscribe(scopedHandle, 3000) == OMNI_USD_LIVE_OK);
    assert(fixture.registry.unsubscribe(globalHandle, 3000) == OMNI_USD_LIVE_OK);
    for (const auto& event : scoped.copy()) assert(event.endpoint == endpoint && event.status.subject_id != elsewhere);
    bool sawOther = false;
    for (const auto& event : global.copy())
        if (event.status.subject_id == elsewhere) { sawOther = true; assert(event.endpoint == otherEndpoint); }
    assert(sawOther);
    assert(fixture.registry.publish(layer, value(OMNI_USD_LIVE_STATE_READY)) == OMNI_USD_LIVE_OK);
    assert(fixture.registry.remove(joinedId) == OMNI_USD_LIVE_OK);
    aggregate = snapshot(fixture.registry, endpoint);
    assert(!aggregate.blocked_layers && !aggregate.failed_layers && !aggregate.blocked_sessions && !aggregate.failed_sessions);
}

void atomicSubscriptionAndConcurrentPublication()
{
    Fixture fixture(1, 1, 2);
    const auto id = fixture.registry.add(endpoint, session, uri, value());
    for (unsigned round = 0; round < 64; ++round) {
        Observer observer;
        OmniUsdLiveCallbackHandle handle = 0;
        std::promise<void> start;
        auto gate = start.get_future().share();
        std::atomic<unsigned> ready{0};
        auto update = value(OMNI_USD_LIVE_STATE_READY);
        update.pending_write_count = round + 1; update.pending_bytes = (round + 1) * 17;
        std::thread subscribe([&] { ++ready; gate.wait(); handle = fixture.subscribe(observer); });
        std::thread publish([&] { ++ready; gate.wait(); assert(fixture.registry.publish(id, update) == OMNI_USD_LIVE_OK); });
        const auto deadline = Clock::now() + std::chrono::seconds(3);
        while (ready.load() != 2) { assert(Clock::now() < deadline); std::this_thread::yield(); }
        start.set_value(); subscribe.join(); publish.join();
        const auto latest = snapshot(fixture.registry, id);
        observer.wait([&](const auto& events) { return !events.empty() && events.back().status.revision == latest.revision; });
        assert(fixture.registry.unsubscribe(handle, 3000) == OMNI_USD_LIVE_OK);
        const auto events = observer.copy();
        assert(!events.empty() && events.size() <= 2);
        same(events.back().status, latest);
        assert(events.back().status.pending_write_count == round + 1 && events.back().status.pending_bytes == (round + 1) * 17);
    }
}

void unknownOutcomeSupersedesEarlierFailure()
{
    for (bool duringClose : {false, true}) {
        Fixture fixture;
        const auto id = fixture.registry.add(endpoint, session, uri, value());
        auto failed = value(OMNI_USD_LIVE_STATE_FAILED);
        failed.result = OMNI_USD_LIVE_PERMISSION_DENIED;
        failed.operation = OMNI_USD_LIVE_OPERATION_READ;
        failed.grpc_status_present = 1; failed.grpc_status = 7;
        failed.recovery_action = OMNI_USD_LIVE_RECOVERY_PRESERVE_AND_REOPEN;
        assert(fixture.registry.publish(id, failed) == OMNI_USD_LIVE_OK);
        const auto incident = snapshot(fixture.registry, id).incident_id;
        assert(incident);
        if (duringClose) {
            auto closing = value(OMNI_USD_LIVE_STATE_CLOSING);
            assert(fixture.registry.publish(id, closing) == OMNI_USD_LIVE_OK);
        }
        auto unknown = value(duringClose ? OMNI_USD_LIVE_STATE_CLOSING : OMNI_USD_LIVE_STATE_FAILED);
        unknown.result = OMNI_USD_LIVE_OUTCOME_UNKNOWN;
        unknown.operation = OMNI_USD_LIVE_OPERATION_UPDATE;
        unknown.grpc_status_present = 1; unknown.grpc_status = 14;
        unknown.recovery_action = OMNI_USD_LIVE_RECOVERY_PRESERVE_AND_REOPEN;
        unknown.has_unconfirmed_edits = 1; unknown.incident_id = 999;
        assert(fixture.registry.publish(id, unknown) == OMNI_USD_LIVE_OK);
        auto retained = snapshot(fixture.registry, id);
        assert(retained.state == unknown.state && retained.result == OMNI_USD_LIVE_OUTCOME_UNKNOWN);
        assert(retained.operation == OMNI_USD_LIVE_OPERATION_UPDATE && retained.grpc_status_present && retained.grpc_status == 14);
        assert(retained.incident_id == incident && retained.has_unconfirmed_edits &&
               retained.recovery_action == OMNI_USD_LIVE_RECOVERY_PRESERVE_AND_REOPEN);
        assert(fixture.registry.publish(id, value(OMNI_USD_LIVE_STATE_READY)) == OMNI_USD_LIVE_OK);
        retained = snapshot(fixture.registry, id);
        assert(retained.state == unknown.state && retained.result == OMNI_USD_LIVE_OUTCOME_UNKNOWN && retained.incident_id == incident);
    }
    {
        // A cause latched on a non-terminal state still fails the subject when
        // a later outcome becomes unknown.
        Fixture fixture;
        const auto id = fixture.registry.add(endpoint, session, uri, value());
        auto cause = value(OMNI_USD_LIVE_STATE_OPENING);
        cause.result = OMNI_USD_LIVE_PERMISSION_DENIED;
        cause.recovery_action = OMNI_USD_LIVE_RECOVERY_PRESERVE_AND_REOPEN;
        assert(fixture.registry.publish(id, cause) == OMNI_USD_LIVE_OK);
        assert(snapshot(fixture.registry, id).state == OMNI_USD_LIVE_STATE_OPENING);
        auto unknown = value(OMNI_USD_LIVE_STATE_READY);
        unknown.result = OMNI_USD_LIVE_OUTCOME_UNKNOWN;
        unknown.recovery_action = OMNI_USD_LIVE_RECOVERY_PRESERVE_AND_REOPEN;
        assert(fixture.registry.publish(id, unknown) == OMNI_USD_LIVE_OK);
        const auto failed = snapshot(fixture.registry, id);
        assert(failed.state == OMNI_USD_LIVE_STATE_FAILED && failed.result == OMNI_USD_LIVE_OUTCOME_UNKNOWN);
    }
    Fixture fixture;
    const auto id = fixture.registry.add(endpoint, session, uri, value(OMNI_USD_LIVE_STATE_READY));
    auto closing = value(OMNI_USD_LIVE_STATE_CLOSING);
    assert(fixture.registry.publish(id, closing) == OMNI_USD_LIVE_OK);
    assert(!snapshot(fixture.registry, id).incident_id);
    closing.result = OMNI_USD_LIVE_OUTCOME_UNKNOWN;
    closing.recovery_action = OMNI_USD_LIVE_RECOVERY_PRESERVE_AND_REOPEN;
    closing.has_unconfirmed_edits = 1;
    assert(fixture.registry.publish(id, closing) == OMNI_USD_LIVE_OK);
    const auto failed = snapshot(fixture.registry, id);
    assert(failed.state == OMNI_USD_LIVE_STATE_CLOSING && failed.incident_id && failed.result == OMNI_USD_LIVE_OUTCOME_UNKNOWN);
}

void identicalPublicationsDoNotCreateRevisions()
{
    Fixture fixture(2, 1, 1);
    auto worker = HeldWorker::hold(fixture.executor);
    auto endpointValue = value(OMNI_USD_LIVE_STATE_READY);
    endpointValue.subject_kind = OMNI_USD_LIVE_SUBJECT_ENDPOINT; endpointValue.subject_id = endpoint;
    assert(fixture.registry.add(endpoint, 0, "", endpointValue) == endpoint);
    const auto id = fixture.registry.add(endpoint, session, uri, value());
    Observer observer;
    const auto registration = fixture.subscribe(observer);
    const auto initial = snapshot(fixture.registry, id);
    auto repeated = initial;
    repeated.revision = UINT64_MAX; repeated.changed_at_ms = 1;
    for (unsigned i = 0; i < 32; ++i)
        assert(fixture.registry.publish(id, repeated) == OMNI_USD_LIVE_OK);
    same(snapshot(fixture.registry, id), initial);
    auto blocked = value(OMNI_USD_LIVE_STATE_BLOCKED);
    blocked.pending_write_count = 1;
    assert(fixture.registry.publish(id, blocked) == OMNI_USD_LIVE_OK);
    const auto endpointWithChild = snapshot(fixture.registry, endpoint);
    assert(endpointWithChild.blocked_layers == 1);
    // Cached endpoint producers do not own child counters. The hub restores
    // them before comparison, avoiding phantom revisions on every actor tick.
    for (unsigned i = 0; i < 32; ++i)
        assert(fixture.registry.publish(endpoint, endpointValue) == OMNI_USD_LIVE_OK);
    same(snapshot(fixture.registry, endpoint), endpointWithChild);
    const auto layerLatest = snapshot(fixture.registry, id);
    worker->resume.set_value();
    observer.wait([](const auto& events) { return events.size() == 2; });
    assert(fixture.registry.unsubscribe(registration, 3000) == OMNI_USD_LIVE_OK);
    for (const auto& event : observer.copy()) {
        if (event.status.subject_id == endpoint) same(event.status, endpointWithChild);
        else { assert(event.status.subject_id == id); same(event.status, layerLatest); }
    }
}

void authoritativeCloseRetirementAndFinalDetach()
{
    Fixture fixture;
    auto opening = value();
    opening.pending_write_count = 1; opening.pending_bytes = 42; opening.has_unconfirmed_edits = 1;
    const auto id = fixture.registry.add(endpoint, session, uri, opening);
    auto failed = opening;
    failed.state = OMNI_USD_LIVE_STATE_FAILED; failed.result = OMNI_USD_LIVE_PERMISSION_DENIED;
    failed.recovery_action = OMNI_USD_LIVE_RECOVERY_PRESERVE_AND_REOPEN;
    assert(fixture.registry.publish(id, failed) == OMNI_USD_LIVE_OK);
    const auto incident = snapshot(fixture.registry, id).incident_id;
    auto closing = failed; closing.state = OMNI_USD_LIVE_STATE_CLOSING;
    assert(fixture.registry.publish(id, closing) == OMNI_USD_LIVE_OK);
    const auto held = snapshot(fixture.registry, id);
    assert(held.pending_write_count == 1 && held.pending_bytes == 42 && held.has_unconfirmed_edits);
    assert(fixture.registry.publish(id, value(OMNI_USD_LIVE_STATE_READY)) == OMNI_USD_LIVE_OK);
    same(snapshot(fixture.registry, id), held);
    auto settled = value(OMNI_USD_LIVE_STATE_CLOSED);
    assert(fixture.registry.publish(id, settled) == OMNI_USD_LIVE_OK);
    const auto closed = snapshot(fixture.registry, id);
    assert(!closed.pending_write_count && !closed.pending_bytes && !closed.has_unconfirmed_edits);
    assert(closed.progress_known && closed.result == OMNI_USD_LIVE_PERMISSION_DENIED && closed.incident_id == incident);
    Observer observer;
    const auto registration = fixture.subscribe(observer);
    observer.wait([](const auto& events) { return events.size() == 1; });
    assert(fixture.registry.remove(id) == OMNI_USD_LIVE_OK);
    observer.wait([](const auto& events) { return events.size() == 2; });
    assert(fixture.registry.unsubscribe(registration, 3000) == OMNI_USD_LIVE_OK);
    const auto delivered = observer.copy();
    assert(delivered[0].status.state == OMNI_USD_LIVE_STATE_CLOSED && delivered[1].status.state == OMNI_USD_LIVE_STATE_CLOSED);
    assert(delivered[1].status.revision > delivered[0].status.revision);
    assert(delivered[1].status.incident_id == incident && !delivered[1].status.has_unconfirmed_edits);
}

void boundedCaptureAndInternalTransitions()
{
    struct Transition {
        OmniUsdLiveEndpoint endpoint = 0;
        OmniUsdLiveSession session = 0;
        std::array<char, 128> uri{};
        bool hasPrevious = false, attached = false;
        OmniUsdLiveStatus previous{}, current{};
    };
    struct Recorder {
        std::array<Transition, 8> rows{};
        size_t count = 0;
        static void observe(void* context, OmniUsdLiveEndpoint observedEndpoint, OmniUsdLiveSession observedSession,
                            std::string_view observedUri, const OmniUsdLiveStatus* previous,
                            const OmniUsdLiveStatus& current, bool attached, const StatusTiming&) noexcept
        {
            auto& self = *static_cast<Recorder*>(context);
            assert(self.count < self.rows.size() && observedUri.size() < self.rows[0].uri.size());
            auto& row = self.rows[self.count++];
            row.endpoint = observedEndpoint; row.session = observedSession;
            observedUri.copy(row.uri.data(), observedUri.size());
            row.hasPrevious = previous != nullptr;
            row.attached = attached;
            if (previous) row.previous = *previous;
            row.current = current;
        }
    };
    DispatchExecutor executor(1);
    Recorder recorder;
    StatusRegistry registry(executor, 2, 0, Recorder::observe, &recorder);
    std::string callerUri = uri;
    const auto a = registry.add(endpoint, session, callerUri, value());
    callerUri.assign("mutated caller storage");
    const auto b = registry.add(otherEndpoint, 404, uri + "-other", value());
    const auto initialA = snapshot(registry, a), initialB = snapshot(registry, b);
    assert(recorder.count == 2 && !recorder.rows[0].hasPrevious && !recorder.rows[1].hasPrevious);
    assert(recorder.rows[0].attached && recorder.rows[1].attached);
    assert(recorder.rows[0].endpoint == endpoint && recorder.rows[0].session == session);
    assert(std::string(recorder.rows[0].uri.data()) == uri);
    same(recorder.rows[0].current, initialA); same(recorder.rows[1].current, initialB);

    const auto named = registry.capture(0, true, UINT64_MAX);
    assert(named.size() == 2);
    for (const auto& item : named) {
        if (item.status.subject_id == a) {
            assert(item.endpoint == endpoint && item.session == session && item.uri == uri);
            same(item.status, initialA);
        } else {
            assert(item.status.subject_id == b && item.endpoint == otherEndpoint && item.session == 404 && item.uri == uri + "-other");
            same(item.status, initialB);
        }
    }
    const auto redacted = registry.capture(0, false, 2 * sizeof(StatusSnapshot));
    assert(redacted.size() == 2 && redacted[0].uri.empty() && redacted[1].uri.empty());
    rejected([&] { registry.capture(0, false, 2 * sizeof(StatusSnapshot) - 1); }, OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    rejected([&] { registry.capture(0, true, 2 * sizeof(StatusSnapshot)); }, OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    const auto selected = registry.capture(endpoint, true, UINT64_MAX);
    assert(selected.size() == 1 && selected[0].status.subject_id == a && selected[0].uri == uri);
    assert(registry.capture(999, true, 0).empty());
    assert(recorder.count == 2); // Captures and their rejected allocations are read-only.
    same(snapshot(registry, a), initialA); same(snapshot(registry, b), initialB);

    auto ready = value(OMNI_USD_LIVE_STATE_READY);
    ready.pending_write_count = 2; ready.pending_bytes = 31;
    assert(registry.publish(a, ready) == OMNI_USD_LIVE_OK && recorder.count == 3);
    assert(recorder.rows[2].hasPrevious);
    same(recorder.rows[2].previous, initialA);
    same(recorder.rows[2].current, snapshot(registry, a));
    ready.revision = UINT64_MAX; ready.changed_at_ms = 1;
    assert(registry.publish(a, ready) == OMNI_USD_LIVE_OK && recorder.count == 3);
    rejected([&] { registry.add(endpoint, session, uri, value()); }, OMNI_USD_LIVE_ALREADY_EXISTS);
    assert(recorder.count == 3);
    const auto beforeRemove = snapshot(registry, a);
    assert(registry.remove(a) == OMNI_USD_LIVE_OK && recorder.count == 4);
    assert(recorder.rows[3].hasPrevious && recorder.rows[3].current.state == OMNI_USD_LIVE_STATE_CLOSED);
    assert(!recorder.rows[3].attached && recorder.rows[2].attached);
    same(recorder.rows[3].previous, beforeRemove);
    assert(recorder.rows[3].current.subject_id == a && recorder.rows[3].current.revision > beforeRemove.revision);
    const auto after = registry.capture(0, true, UINT64_MAX);
    assert(after.size() == 1 && after[0].status.subject_id == b);
    assert(registry.remove(b) == OMNI_USD_LIVE_OK && recorder.count == 5);
    assert(registry.capture(0, true, 0).empty());
    // Owned snapshots survive both source mutations and complete detachment.
    for (const auto& item : named) {
        assert(!item.uri.empty() && item.status.state == OMNI_USD_LIVE_STATE_OPENING);
        same(item.status, item.status.subject_id == a ? initialA : initialB);
    }
    registry.beginShutdown();
    assert(registry.waitForShutdown(Clock::now() + std::chrono::seconds(3)));
    executor.beginShutdown();
    assert(executor.waitForShutdown(Clock::now() + std::chrono::seconds(3)));
}

OmniUsdLiveStatusObservation observation()
{
    OmniUsdLiveStatusObservation result{};
    result.struct_size = sizeof(result); result.abi_version = OMNI_USD_LIVE_ABI_VERSION;
    result.endpoint = 919; result.session = 818;
    return result;
}
void pullRetainsSelectedAndClosedRevisions()
{
    Fixture fixture(1, 2);
    const auto id = fixture.registry.add(endpoint, session, uri, value());
    Observer callback;
    const auto callbackHandle = fixture.subscribe(callback);
    const auto handle = fixture.registry.createObserver(endpoint);
    rejected([&] { fixture.registry.createObserver(0); }, OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    const auto initial = snapshot(fixture.registry, id);
    auto output = observation(); const auto untouched = output;
    uint64_t size = 0;
    assert(fixture.registry.read(callbackHandle, 0, &output, nullptr, &size) == OMNI_USD_LIVE_UNSUPPORTED);
    assert(fixture.registry.acknowledge(callbackHandle, id, initial.revision) == OMNI_USD_LIVE_UNSUPPORTED);
    assert(fixture.registry.read(handle, 0, &output, nullptr, &size) == OMNI_USD_LIVE_OK);
    assert(size == uri.size() + 1 && std::memcmp(&output, &untouched, sizeof(output)) == 0);
    auto failed = value(OMNI_USD_LIVE_STATE_FAILED);
    failed.result = OMNI_USD_LIVE_OUTCOME_UNKNOWN;
    failed.recovery_action = OMNI_USD_LIVE_RECOVERY_PRESERVE_AND_REOPEN;
    failed.pending_write_count = 1; failed.pending_bytes = 17; failed.has_unconfirmed_edits = 1;
    assert(fixture.registry.publish(id, failed) == OMNI_USD_LIVE_OK);
    const auto failure = snapshot(fixture.registry, id);
    assert(fixture.registry.remove(id) == OMNI_USD_LIVE_OK);
    callback.wait([](const auto& events) { return !events.empty() && events.back().status.state == OMNI_USD_LIVE_STATE_CLOSED; });
    // Callback delivery cannot discharge another observer's retained CLOSED.
    assert(fixture.registry.unsubscribe(callbackHandle, 3000) == OMNI_USD_LIVE_OK);
    rejected([&] { fixture.registry.add(endpoint, session, uri, value()); }, OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    std::array<char, 512> buffer; buffer.fill('!');
    size = 1;
    assert(fixture.registry.read(handle, 0, &output, buffer.data(), &size) == OMNI_USD_LIVE_BUFFER_TOO_SMALL);
    assert(size == uri.size() + 1 && buffer[0] == '!' && std::memcmp(&output, &untouched, sizeof(output)) == 0);
    size = buffer.size();
    assert(fixture.registry.read(handle, 0, &output, buffer.data(), &size) == OMNI_USD_LIVE_OK);
    assert(output.endpoint == endpoint && output.session == session && std::string(buffer.data()) == uri);
    same(output.status, initial);
    assert(fixture.registry.acknowledge(handle, id + 1, initial.revision) == OMNI_USD_LIVE_INVALID_ARGUMENT);
    assert(fixture.registry.acknowledge(handle, id, initial.revision + 1) == OMNI_USD_LIVE_INVALID_ARGUMENT);
    size = buffer.size();
    assert(fixture.registry.read(handle, 0, &output, buffer.data(), &size) == OMNI_USD_LIVE_OK);
    same(output.status, initial);
    assert(fixture.registry.acknowledge(handle, id, initial.revision) == OMNI_USD_LIVE_OK);
    rejected([&] { fixture.registry.add(endpoint, session, uri, value()); }, OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    size = buffer.size();
    assert(fixture.registry.read(handle, 0, &output, buffer.data(), &size) == OMNI_USD_LIVE_OK);
    assert(output.status.state == OMNI_USD_LIVE_STATE_CLOSED && output.status.revision > failure.revision);
    assert(output.status.result == OMNI_USD_LIVE_OUTCOME_UNKNOWN && output.status.incident_id == failure.incident_id);
    assert(output.status.has_unconfirmed_edits && output.status.pending_write_count == 1 && output.status.pending_bytes == 17);
    assert(fixture.registry.acknowledge(handle, id, initial.revision) == OMNI_USD_LIVE_INVALID_ARGUMENT);
    rejected([&] { fixture.registry.add(endpoint, session, uri, value()); }, OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    assert(fixture.registry.acknowledge(handle, id, output.status.revision) == OMNI_USD_LIVE_OK);
    const auto beforeTimeout = output;
    size = 123;
    assert(fixture.registry.read(handle, 0, &output, buffer.data(), &size) == OMNI_USD_LIVE_TIMEOUT);
    assert(size == 123 && std::memcmp(&output, &beforeTimeout, sizeof(output)) == 0);
    const auto next = fixture.registry.add(endpoint, session, uri, value());
    assert(next != id);
    assert(fixture.registry.unsubscribe(handle, 3000) == OMNI_USD_LIVE_OK);
    assert(fixture.registry.acknowledge(handle, next, 1) == OMNI_USD_LIVE_INVALID_HANDLE);
}
// A caller built against the 1.0 layout passes the prefix before
// buffered_edit_count and receives exactly that prefix; the bytes after it
// stay untouched. A size that splits the appended field is rejected.
void bufferedEditsAndOlderOutputPrefix()
{
    Fixture fixture(1, 1);
    const auto id = fixture.registry.add(endpoint, session, uri, value());
    auto buffered = value(OMNI_USD_LIVE_STATE_READY);
    buffered.buffered_edit_count = 3; buffered.has_unconfirmed_edits = 1;
    assert(fixture.registry.publish(id, buffered) == OMNI_USD_LIVE_OK);
    const auto first = snapshot(fixture.registry, id);
    assert(first.buffered_edit_count == 3 && first.has_unconfirmed_edits && !first.pending_write_count);
    buffered.buffered_edit_count = 4;
    assert(fixture.registry.publish(id, buffered) == OMNI_USD_LIVE_OK);
    const auto second = snapshot(fixture.registry, id);
    assert(second.buffered_edit_count == 4 && second.revision == first.revision + 1);

    constexpr size_t prefix = offsetof(OmniUsdLiveStatus, buffered_edit_count);
    static_assert(prefix < sizeof(OmniUsdLiveStatus));
    const auto older = [](uint32_t size) {
        std::array<unsigned char, sizeof(OmniUsdLiveStatus) + 8> bytes; bytes.fill(0xAB);
        const uint32_t abi = OMNI_USD_LIVE_ABI_VERSION;
        std::memcpy(bytes.data(), &size, sizeof(size)); std::memcpy(bytes.data() + sizeof(size), &abi, sizeof(abi));
        return bytes;
    };
    const auto untouchedAfter = [](const auto& bytes, size_t from) {
        for (size_t i = from; i < bytes.size(); ++i) if (bytes[i] != 0xAB) return false;
        return true;
    };
    auto bytes = older(prefix);
    auto* output = reinterpret_cast<OmniUsdLiveStatus*>(bytes.data());
    assert(fixture.registry.snapshot(id, output) == OMNI_USD_LIVE_OK && untouchedAfter(bytes, prefix));
    auto expected = second; expected.struct_size = prefix;
    assert(std::memcmp(bytes.data(), &expected, prefix) == 0);
    bytes = older(prefix);
    assert(fixture.registry.layer(uri, output) == OMNI_USD_LIVE_OK && untouchedAfter(bytes, prefix));
    assert(std::memcmp(bytes.data(), &expected, prefix) == 0);
    bytes = older(prefix + 4);
    assert(fixture.registry.snapshot(id, output) == OMNI_USD_LIVE_INVALID_ARGUMENT && untouchedAfter(bytes, 8));
    assert(fixture.registry.layer(uri, output) == OMNI_USD_LIVE_INVALID_ARGUMENT && untouchedAfter(bytes, 8));
    bytes = older(prefix - 8);
    assert(fixture.registry.snapshot(id, output) == OMNI_USD_LIVE_INVALID_ARGUMENT && untouchedAfter(bytes, 8));

    const auto handle = fixture.registry.createObserver(endpoint);
    constexpr size_t observed = offsetof(OmniUsdLiveStatusObservation, status) + prefix;
    std::array<unsigned char, sizeof(OmniUsdLiveStatusObservation) + 8> pulled; pulled.fill(0xAB);
    const uint32_t observedSize = observed, abi = OMNI_USD_LIVE_ABI_VERSION;
    std::memcpy(pulled.data(), &observedSize, sizeof(observedSize));
    std::memcpy(pulled.data() + sizeof(observedSize), &abi, sizeof(abi));
    std::array<char, 128> name{};
    uint64_t size = name.size();
    assert(fixture.registry.read(handle, 0, reinterpret_cast<OmniUsdLiveStatusObservation*>(pulled.data()),
                                 name.data(), &size) == OMNI_USD_LIVE_OK);
    assert(untouchedAfter(pulled, observed) && std::string(name.data()) == uri);
    OmniUsdLiveStatusObservation header{};
    std::memcpy(&header, pulled.data(), offsetof(OmniUsdLiveStatusObservation, status));
    assert(header.struct_size == observed && header.endpoint == endpoint && header.session == session);
    // The inline status reports the prefix size it carries.
    expected.struct_size = prefix;
    assert(std::memcmp(pulled.data() + offsetof(OmniUsdLiveStatusObservation, status), &expected, prefix) == 0);
    assert(fixture.registry.unsubscribe(handle, 3000) == OMNI_USD_LIVE_OK);
}
// Buffered records are unconfirmed edits: a stale producer cannot clear them
// after a latched failure, and an authoritative close with no pending intent
// keeps the edit flag while records remain buffered.
void bufferedEditsSurviveLatchedFailureAndClose()
{
    Fixture fixture;
    const auto id = fixture.registry.add(endpoint, session, uri, value());
    auto failed = value(OMNI_USD_LIVE_STATE_FAILED);
    failed.result = OMNI_USD_LIVE_UNSUPPORTED;
    failed.recovery_action = OMNI_USD_LIVE_RECOVERY_PRESERVE_AND_REOPEN;
    failed.buffered_edit_count = 2; failed.has_unconfirmed_edits = 1;
    assert(fixture.registry.publish(id, failed) == OMNI_USD_LIVE_OK);
    assert(fixture.registry.publish(id, value(OMNI_USD_LIVE_STATE_READY)) == OMNI_USD_LIVE_OK);
    auto current = snapshot(fixture.registry, id);
    assert(current.state == OMNI_USD_LIVE_STATE_FAILED && current.buffered_edit_count == 2 && current.has_unconfirmed_edits);
    failed.buffered_edit_count = 5; // The host keeps counting edits on a failed layer.
    assert(fixture.registry.publish(id, failed) == OMNI_USD_LIVE_OK);
    assert(snapshot(fixture.registry, id).buffered_edit_count == 5);
    auto closing = value(OMNI_USD_LIVE_STATE_CLOSING);
    closing.buffered_edit_count = 5;
    assert(fixture.registry.publish(id, closing) == OMNI_USD_LIVE_OK);
    current = snapshot(fixture.registry, id);
    assert(current.state == OMNI_USD_LIVE_STATE_CLOSING && current.buffered_edit_count == 5 && current.has_unconfirmed_edits);
}
// A burst of host records asks for one actor wake; the status publication's
// take() reads the latest count and re-arms the next request.
void bufferedEditCountWakesOncePerBurst()
{
    BufferedEditCount count;
    unsigned wakes = 0;
    for (uint64_t records = 1; records <= 10000; ++records) wakes += count.note(records);
    assert(wakes == 1 && count.load() == 10000);
    assert(count.take().count == 10000);
    assert(count.note(10001) && !count.note(10002));
    assert(count.take().count == 10002 && count.take().count == 10002);
    assert(count.note(0));
    count.abandonWake(); // No actor could be woken; the next record asks again.
    assert(count.note(1) && count.take().count == 1);
}
// A flush that empties the buffer before the woken publication reads it still
// reports the emptying once, and the registry advances the revision for it
// although no field changed.
void bufferEmptiedBeforePublicationAdvancesRevision()
{
    BufferedEditCount count;
    auto taken = count.take();
    assert(taken.count == 0 && !taken.emptied);
    assert(count.note(1));   // The record requests the wake.
    assert(!count.note(0));  // The flush runs first; its wake is coalesced.
    taken = count.take();
    assert(taken.count == 0 && taken.emptied);
    taken = count.take();
    assert(taken.count == 0 && !taken.emptied);
    // A publication that read the nonzero count leaves the emptying to an
    // ordinary change from that count.
    assert(count.note(2));
    taken = count.take();
    assert(taken.count == 2 && !taken.emptied);
    assert(count.note(0));
    taken = count.take();
    assert(taken.count == 0 && !taken.emptied);

    Fixture fixture(1, 1);
    const auto id = fixture.registry.add(endpoint, session, uri, value());
    const auto initial = snapshot(fixture.registry, id);
    assert(fixture.registry.publish(id, initial) == OMNI_USD_LIVE_OK);
    same(snapshot(fixture.registry, id), initial);
    assert(fixture.registry.publish(id, initial, true) == OMNI_USD_LIVE_OK);
    const auto advanced = snapshot(fixture.registry, id);
    assert(advanced.revision == initial.revision + 1 && advanced.buffered_edit_count == 0 &&
           advanced.state == initial.state);
}
void pullWakeCancellationAndWorkerRules()
{
    Fixture fixture(2, 3, 1);
    const auto ignored = fixture.registry.add(otherEndpoint, session, uri + "-other", value());
    (void)ignored;
    const auto handle = fixture.registry.createObserver(endpoint);
    auto output = observation(); uint64_t size = 0;
    assert(fixture.registry.read(handle, 0, &output, nullptr, &size) == OMNI_USD_LIVE_TIMEOUT);
    struct ReadJob final : DispatchJob {
        StatusRegistry& registry; uint64_t handle; std::promise<OmniUsdLiveResult> done;
        ReadJob(StatusRegistry& r, uint64_t h) : registry(r), handle(h) {}
        void dispatch() noexcept override {
            auto result = observation(); uint64_t bytes = 0;
            done.set_value(registry.read(handle, 3000, &result, nullptr, &bytes));
        }
    };
    auto worker = std::make_shared<ReadJob>(fixture.registry, handle);
    auto workerResult = worker->done.get_future(); fixture.executor.enqueue(*worker);
    assert(workerResult.wait_for(std::chrono::seconds(3)) == std::future_status::ready);
    assert(workerResult.get() == OMNI_USD_LIVE_BUSY);
    auto reading = std::async(std::launch::async, [&] {
        auto result = observation(); std::array<char,512> bytes{}; uint64_t capacity = bytes.size();
        const auto code = fixture.registry.read(handle, 3000, &result, bytes.data(), &capacity);
        return std::make_pair(code, result);
    });
    const auto added = fixture.registry.add(endpoint, session, uri, value());
    assert(reading.wait_for(std::chrono::seconds(3)) == std::future_status::ready);
    const auto received = reading.get();
    assert(received.first == OMNI_USD_LIVE_OK && received.second.status.subject_id == added);
    assert(fixture.registry.acknowledge(handle, added, received.second.status.revision) == OMNI_USD_LIVE_OK);
    // Hold cancellation cleanup, keeping the cancelled registration discoverable
    // until the reader has observed cancellation, regardless of thread ordering.
    auto held = HeldWorker::hold(fixture.executor);
    std::promise<void> entered;
    auto enteredFuture = entered.get_future();
    auto cancelled = std::async(std::launch::async, [&] {
        auto result = observation(); const auto before = result; uint64_t bytes = 77;
        entered.set_value();
        const auto code = fixture.registry.read(handle, 3000, &result, nullptr, &bytes);
        assert(bytes == 77 && std::memcmp(&result, &before, sizeof(result)) == 0);
        return code;
    });
    enteredFuture.wait();
    assert(fixture.registry.unsubscribe(handle, 0) == OMNI_USD_LIVE_TIMEOUT);
    assert(cancelled.wait_for(std::chrono::seconds(3)) == std::future_status::ready);
    assert(cancelled.get() == OMNI_USD_LIVE_CANCELLED);
    held->resume.set_value();
    assert(fixture.registry.unsubscribe(handle, 3000) == OMNI_USD_LIVE_OK);
}
void reservedStateIsNeverPublished()
{
    // State 6 is reserved (design section 16): neither admission nor publication accepts it.
    static_assert(!definedState(6u) && definedState(OMNI_USD_LIVE_STATE_BLOCKED) && definedState(OMNI_USD_LIVE_STATE_FAILED));
    Fixture fixture(2, 1, 1);
    rejected([&] { fixture.registry.add(endpoint, session, uri, value(6u)); }, OMNI_USD_LIVE_INVALID_ARGUMENT);
    const auto id = fixture.registry.add(endpoint, session, uri, value());
    const auto before = snapshot(fixture.registry, id);
    assert(fixture.registry.publish(id, value(6u)) == OMNI_USD_LIVE_INVALID_ARGUMENT);
    same(snapshot(fixture.registry, id), before);
    assert(fixture.registry.publish(id, value(OMNI_USD_LIVE_STATE_CLOSED + 1)) == OMNI_USD_LIVE_INVALID_ARGUMENT);
    assert(fixture.registry.remove(id) == OMNI_USD_LIVE_OK);
}
// The read and applied sequences refresh in place: a publication that
// changes only them keeps the revision and wakes no observer, and a later
// semantic change carries them. A caller built before they were appended
// passes the prefix before read_sequence and receives exactly that prefix.
void progressSequencesRefreshWithoutRevision()
{
    Fixture fixture(1, 1);
    Observer observer;
    const auto handle = fixture.subscribe(observer);
    const auto id = fixture.registry.add(endpoint, session, uri, value());
    auto ready = value(OMNI_USD_LIVE_STATE_READY);
    ready.read_sequence = 10; ready.applied_sequence = 4;
    assert(fixture.registry.publish(id, ready) == OMNI_USD_LIVE_OK);
    observer.wait([&](const auto& events) { return !events.empty() && events.back().status.revision == 2; });
    const auto lagging = snapshot(fixture.registry, id);
    assert(lagging.revision == 2 && lagging.read_sequence == 10 && lagging.applied_sequence == 4);
    const auto delivered = observer.copy().back().status;
    assert(delivered.read_sequence == 10 && delivered.applied_sequence == 4);
    ready.read_sequence = 12; ready.applied_sequence = 12;
    assert(fixture.registry.publish(id, ready) == OMNI_USD_LIVE_OK);
    const auto caughtUp = snapshot(fixture.registry, id);
    assert(caughtUp.revision == lagging.revision && caughtUp.changed_at_ms == lagging.changed_at_ms);
    assert(caughtUp.read_sequence == 12 && caughtUp.applied_sequence == 12);
    // A semantic change advances the revision and carries the current sequences.
    ready.pending_write_count = 1; ready.has_unconfirmed_edits = 1; ready.read_sequence = 13;
    assert(fixture.registry.publish(id, ready) == OMNI_USD_LIVE_OK);
    observer.wait([&](const auto& events) { return events.back().status.revision == 3; });
    const auto events = observer.copy();
    for (const auto& event : events) assert(event.status.revision != 2 || event.status.applied_sequence == 4);
    assert(events.back().status.read_sequence == 13 && events.back().status.applied_sequence == 12);
    // A latched failure keeps refreshing them: an issuance-only FAILED layer keeps reading.
    auto failed = value(OMNI_USD_LIVE_STATE_FAILED);
    failed.result = OMNI_USD_LIVE_UNSUPPORTED;
    failed.recovery_action = OMNI_USD_LIVE_RECOVERY_PRESERVE_AND_REOPEN;
    failed.read_sequence = 14; failed.applied_sequence = 13;
    assert(fixture.registry.publish(id, failed) == OMNI_USD_LIVE_OK);
    const auto latched = snapshot(fixture.registry, id);
    failed.read_sequence = 15; failed.applied_sequence = 15;
    assert(fixture.registry.publish(id, failed) == OMNI_USD_LIVE_OK);
    const auto later = snapshot(fixture.registry, id);
    assert(later.revision == latched.revision && later.state == OMNI_USD_LIVE_STATE_FAILED);
    assert(later.read_sequence == 15 && later.applied_sequence == 15);

    constexpr size_t prefix = offsetof(OmniUsdLiveStatus, read_sequence);
    static_assert(prefix == offsetof(OmniUsdLiveStatus, buffered_edit_count) + sizeof(uint64_t));
    std::array<unsigned char, sizeof(OmniUsdLiveStatus) + 8> bytes;
    const auto older = [&](uint32_t size) {
        bytes.fill(0xAB);
        const uint32_t abi = OMNI_USD_LIVE_ABI_VERSION;
        std::memcpy(bytes.data(), &size, sizeof(size)); std::memcpy(bytes.data() + sizeof(size), &abi, sizeof(abi));
        return reinterpret_cast<OmniUsdLiveStatus*>(bytes.data());
    };
    const auto untouchedAfter = [&](size_t from) {
        for (size_t i = from; i < bytes.size(); ++i) if (bytes[i] != 0xAB) return false;
        return true;
    };
    auto expected = later; expected.struct_size = prefix;
    assert(fixture.registry.snapshot(id, older(prefix)) == OMNI_USD_LIVE_OK && untouchedAfter(prefix));
    assert(std::memcmp(bytes.data(), &expected, prefix) == 0);
    assert(fixture.registry.layer(uri, older(prefix)) == OMNI_USD_LIVE_OK && untouchedAfter(prefix));
    assert(std::memcmp(bytes.data(), &expected, prefix) == 0);
    // Sizes that split read_sequence or applied_sequence are rejected before any write.
    for (const uint32_t split : {uint32_t(prefix + 4), uint32_t(prefix + 8), uint32_t(prefix + 12)}) {
        assert(fixture.registry.snapshot(id, older(split)) == OMNI_USD_LIVE_INVALID_ARGUMENT && untouchedAfter(8));
        assert(fixture.registry.layer(uri, older(split)) == OMNI_USD_LIVE_INVALID_ARGUMENT && untouchedAfter(8));
    }
    assert(fixture.registry.unsubscribe(handle, 3000) == OMNI_USD_LIVE_OK);
    assert(fixture.registry.remove(id) == OMNI_USD_LIVE_OK);
}
} // namespace

void statusRegistryTests()
{
    reservedStateIsNeverPublished();
    initialCoalescingAndCachedQueries();
    closedReservationsAndNewIdentity();
    selfUnregisterAndIndependentRegistration();
    closeWhileEarlierCallbackRuns();
    leaseDestructionOutsideRegistryLocks();
    endpointFiltersAndIndependentChildCounts();
    atomicSubscriptionAndConcurrentPublication();
    unknownOutcomeSupersedesEarlierFailure();
    identicalPublicationsDoNotCreateRevisions();
    authoritativeCloseRetirementAndFinalDetach();
    boundedCaptureAndInternalTransitions();
    pullRetainsSelectedAndClosedRevisions();
    pullWakeCancellationAndWorkerRules();
    bufferedEditsAndOlderOutputPrefix();
    bufferedEditsSurviveLatchedFailureAndClose();
    bufferedEditCountWakesOncePerBurst();
    bufferEmptiedBeforePublicationAdvancesRevision();
    progressSequencesRefreshWithoutRevision();
    std::cout << "status registry: bounded coalescing, retained reads/ACK, closure, ownership, and 64 subscription races passed\n";
}
