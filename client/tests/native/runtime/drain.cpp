// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "LiveDrain.h"
#include <atomic>
#include <cassert>
#include <future>
#include <iostream>
#include <thread>

using namespace omni::usd::live;
namespace {
using Clock = DrainRegistry::Clock;
using Ms = std::chrono::milliseconds;

std::shared_ptr<DrainTicket> admitted(const std::shared_ptr<DrainState>& state)
{
    auto ticket = state->prepare();
    assert(ticket->admit() == OMNI_USD_LIVE_OK);
    return ticket;
}
OmniUsdLiveResult poll(DrainRegistry& registry, const DrainRegistry::Capture& capture, bool allowBlock = true)
{
    return registry.wait(capture, Clock::now(), allowBlock);
}
void expectCapacity(DrainRegistry& registry)
{
    bool rejected = false;
    try { registry.makeState(); }
    catch (OmniUsdLiveResult result) { rejected = result == OMNI_USD_LIVE_RESOURCE_EXHAUSTED; }
    assert(rejected);
}

void capturedMembershipAndUnorderedCompletion()
{
    DrainRegistry registry(4);
    assert(registry.capture().targets.empty() && registry.wait(0) == OMNI_USD_LIVE_OK);
    auto state = registry.makeState();
    auto inactive = state->prepare();
    const auto empty = registry.capture();
    assert(empty.targets.size() == 1 && empty.targets[0].fence.cutoff == 0);
    assert(poll(registry, empty) == OMNI_USD_LIVE_OK);
    auto first = admitted(state), second = admitted(state), third = admitted(state);
    const auto captured = registry.capture();
    assert(captured.targets.size() == 1 && captured.targets[0].fence.cutoff == 3);
    auto later = admitted(state);
    assert(inactive->admit() == OMNI_USD_LIVE_OK); // Prepared before capture, admitted after it.
    auto laterState = registry.makeState();
    auto anotherLater = admitted(laterState);
    assert(poll(registry, empty) == OMNI_USD_LIVE_OK);
    third->complete();
    first->complete();
    assert(poll(registry, captured) == OMNI_USD_LIVE_TIMEOUT);
    assert(poll(registry, captured, false) == OMNI_USD_LIVE_BUSY);
    second->complete();
    assert(poll(registry, captured) == OMNI_USD_LIVE_OK);
    assert(poll(registry, captured, false) == OMNI_USD_LIVE_OK);
    assert(registry.wait(0) == OMNI_USD_LIVE_TIMEOUT);
    assert(registry.wait(0, false) == OMNI_USD_LIVE_BUSY);
    later->complete(); inactive->complete(); anotherLater->complete();
    assert(registry.wait(0) == OMNI_USD_LIVE_OK);
    // Successful admission is idempotent even after completion; it cannot
    // insert a second intrusive node or turn retired work back into pending.
    assert(second->admit() == OMNI_USD_LIVE_OK);
    second->complete();
    assert(registry.wait(0) == OMNI_USD_LIVE_OK);
    assert(state->capture().cutoff == 5);
}

void deadlineAndNonblockingWaitsDoNotMutate()
{
    DrainRegistry registry(2);
    auto a = registry.makeState(), b = registry.makeState();
    auto first = admitted(a), second = admitted(b);
    const auto captured = registry.capture();
    const auto deadline = Clock::now() + Ms(30);
    assert(registry.wait(captured, deadline) == OMNI_USD_LIVE_TIMEOUT);
    assert(Clock::now() >= deadline);
    // A timeout neither cancels nor retires either participant. The same
    // absolute deadline is already expired when the first target completes.
    assert(poll(registry, captured, false) == OMNI_USD_LIVE_BUSY);
    assert(first->admit() == OMNI_USD_LIVE_OK && second->admit() == OMNI_USD_LIVE_OK);
    first->complete();
    assert(registry.wait(captured, deadline) == OMNI_USD_LIVE_TIMEOUT);
    assert(registry.wait(captured, Clock::now() + std::chrono::hours(1), false) == OMNI_USD_LIVE_BUSY);
    second->complete();
    assert(registry.wait(captured, deadline) == OMNI_USD_LIVE_OK);
    assert(registry.wait(captured, deadline, false) == OMNI_USD_LIVE_OK);
}

void terminalFailureBypassesEarlierPendingState()
{
    DrainRegistry registry(2);
    auto earlier = registry.makeState(), failing = registry.makeState();
    auto first = admitted(earlier), second = admitted(failing);
    const auto captured = registry.capture();
    std::promise<void> ready, start;
    auto readyFuture = ready.get_future();
    auto gate = start.get_future().share();
    auto waiter = std::async(std::launch::async, [&] {
        ready.set_value();
        gate.wait();
        return registry.wait(captured, Clock::now() + std::chrono::seconds(3));
    });
    assert(readyFuture.wait_for(std::chrono::seconds(1)) == std::future_status::ready);
    start.set_value();
    failing->fail(OMNI_USD_LIVE_UNAUTHENTICATED);
    const bool prompt = waiter.wait_for(std::chrono::seconds(1)) == std::future_status::ready;
    // Cleanup also makes a broken sequential waiter exit, so reporting the
    // missed prompt failure does not rely on the outer subprocess watchdog.
    first->complete();
    const auto result = waiter.get();
    assert(prompt && result == OMNI_USD_LIVE_UNAUTHENTICATED);
    assert(poll(registry, captured, false) == OMNI_USD_LIVE_UNAUTHENTICATED);
    second->complete();
    assert(poll(registry, captured) == OMNI_USD_LIVE_UNAUTHENTICATED);
}

void failuresRemainBoundToTheirCapturedObligations()
{
    DrainRegistry registry(1);
    auto state = registry.makeState();
    auto old = admitted(state);
    const auto oldCapture = registry.capture();
    old->complete();
    auto later = admitted(state);
    const auto pending = registry.capture();
    // Only the first failure reports that it latched the result; a later
    // OUTCOME_UNKNOWN override still reports false.
    assert(!state->fail(OMNI_USD_LIVE_OK));
    assert(state->fail(OMNI_USD_LIVE_PERMISSION_DENIED));
    assert(poll(registry, oldCapture) == OMNI_USD_LIVE_OK);
    assert(poll(registry, pending) == OMNI_USD_LIVE_PERMISSION_DENIED);
    assert(!state->fail(OMNI_USD_LIVE_INTERNAL_ERROR));
    assert(poll(registry, pending) == OMNI_USD_LIVE_PERMISSION_DENIED);
    assert(!state->fail(OMNI_USD_LIVE_OUTCOME_UNKNOWN));
    state->close();
    assert(!state->fail(OMNI_USD_LIVE_CANCELLED));
    assert(poll(registry, pending) == OMNI_USD_LIVE_OUTCOME_UNKNOWN);
    later.reset(); // Destruction retires the obligation, not its known failure.
    assert(poll(registry, pending) == OMNI_USD_LIVE_OUTCOME_UNKNOWN);
    assert(poll(registry, oldCapture) == OMNI_USD_LIVE_OK);
    const auto failedEmpty = registry.capture();
    assert(poll(registry, failedEmpty) == OMNI_USD_LIVE_OUTCOME_UNKNOWN);
    assert(poll(registry, failedEmpty, false) == OMNI_USD_LIVE_OUTCOME_UNKNOWN);
}

void issuanceFailureKeepsAdmitting()
{
    // A layer that stops issuing but keeps its read open fails every existing
    // obligation and later capture while its read frames keep admitting.
    DrainRegistry registry(1);
    auto state = registry.makeState();
    auto pending = admitted(state);
    const auto before = registry.capture();
    state->fail(OMNI_USD_LIVE_INVALID_ARGUMENT, true);
    assert(poll(registry, before) == OMNI_USD_LIVE_INVALID_ARGUMENT);
    auto frame = admitted(state);
    assert(poll(registry, registry.capture()) == OMNI_USD_LIVE_INVALID_ARGUMENT);
    state->fail(OMNI_USD_LIVE_UNSUPPORTED, true);
    auto second = admitted(state);
    // An ordinary failure ends admission and keeps the first cause.
    state->fail(OMNI_USD_LIVE_MALFORMED_STREAM);
    assert(state->prepare()->admit() == OMNI_USD_LIVE_INVALID_ARGUMENT);
    state->fail(OMNI_USD_LIVE_UNSUPPORTED, true); // Never reopens admission.
    assert(state->prepare()->admit() == OMNI_USD_LIVE_INVALID_ARGUMENT);
    second->complete(); frame->complete(); pending->complete();
    assert(poll(registry, before) == OMNI_USD_LIVE_INVALID_ARGUMENT);
}

void terminalBusyIsNotPending()
{
    DrainRegistry registry(2);
    auto earlier = registry.makeState(), failing = registry.makeState();
    auto first = admitted(earlier), second = admitted(failing);
    const auto captured = registry.capture();
    assert(poll(registry, captured) == OMNI_USD_LIVE_TIMEOUT);
    failing->fail(OMNI_USD_LIVE_BUSY);
    // A terminal BUSY must beat timeout, despite an earlier pending state.
    // An expired deadline catches conflation without any timing assumption.
    assert(poll(registry, captured) == OMNI_USD_LIVE_BUSY);
    assert(registry.wait(captured, Clock::now() + std::chrono::seconds(3)) == OMNI_USD_LIVE_BUSY);
    second.reset();
    assert(poll(registry, captured) == OMNI_USD_LIVE_BUSY);
    assert(registry.wait(0) == OMNI_USD_LIVE_BUSY); // Empty failed baseline also remains terminal.
    failing->fail(OMNI_USD_LIVE_OUTCOME_UNKNOWN);
    assert(poll(registry, captured) == OMNI_USD_LIVE_OUTCOME_UNKNOWN);
    first->complete();
}

void inactiveRejectedAndClosedTickets()
{
    DrainRegistry registry(4);
    auto state = registry.makeState();
    auto abandoned = state->prepare();
    abandoned->complete();
    assert(abandoned->admit() == OMNI_USD_LIVE_CANCELLED);
    assert(abandoned->admit() == OMNI_USD_LIVE_CANCELLED);
    assert(state->capture().cutoff == 0);
    auto rejected = state->prepare();
    state->fail(OMNI_USD_LIVE_UNAUTHENTICATED);
    assert(rejected->admit() == OMNI_USD_LIVE_UNAUTHENTICATED);
    state->fail(OMNI_USD_LIVE_OUTCOME_UNKNOWN);
    assert(rejected->admit() == OMNI_USD_LIVE_UNAUTHENTICATED);
    assert(state->capture().cutoff == 0);
    auto unknown = state->prepare();
    assert(unknown->admit() == OMNI_USD_LIVE_OUTCOME_UNKNOWN);

    auto emptyClosed = registry.makeState();
    auto notAdmitted = emptyClosed->prepare();
    emptyClosed->close();
    assert(notAdmitted->admit() == OMNI_USD_LIVE_CANCELLED);
    assert(emptyClosed->poll(emptyClosed->capture()) == OMNI_USD_LIVE_OK);
    auto pendingClosed = registry.makeState();
    auto pending = admitted(pendingClosed);
    const auto beforeClose = pendingClosed->capture();
    pendingClosed->close();
    assert(pendingClosed->poll(beforeClose) == OMNI_USD_LIVE_CANCELLED);
    pending.reset();
    assert(pendingClosed->poll(beforeClose) == OMNI_USD_LIVE_CANCELLED);
    auto afterClose = pendingClosed->prepare();
    assert(afterClose->admit() == OMNI_USD_LIVE_CANCELLED);
}

void ticketAndRegistryOwnership()
{
    DrainRegistry registry(1);
    auto state = registry.makeState();
    std::weak_ptr<DrainState> weak = state;
    auto ticket = admitted(state);
    auto alias = ticket;
    auto captured = registry.capture();
    state.reset();
    ticket.reset();
    assert(!weak.expired() && poll(registry, captured) == OMNI_USD_LIVE_TIMEOUT);
    expectCapacity(registry);
    alias.reset(); // Only the final shared ticket owner completes its node.
    assert(poll(registry, captured) == OMNI_USD_LIVE_OK && !weak.expired());
    expectCapacity(registry); // The capture still owns its participant.
    captured.targets.clear();
    assert(weak.expired());
    // Repeated expired registrations must be pruned before enforcing the cap.
    for (unsigned i = 0; i < 256; ++i) {
        auto temporary = registry.makeState();
        auto prepared = temporary->prepare();
        std::weak_ptr<DrainState> observed = temporary;
        temporary.reset();
        assert(!observed.expired());
        prepared.reset();
        assert(observed.expired());
    }
    assert(registry.capture().targets.empty());
    auto closed = registry.makeState();
    closed->close();
    expectCapacity(registry); // Closed does not mean its owner has disappeared.
    closed.reset();
    assert(registry.makeState());

    // Tickets retain the notification state, not a dangling registry pointer.
    auto transientRegistry = std::make_unique<DrainRegistry>(1);
    auto retainedState = transientRegistry->makeState();
    auto retainedTicket = admitted(retainedState);
    const auto fence = retainedState->capture();
    transientRegistry.reset();
    assert(retainedState->poll(fence) == OMNI_USD_LIVE_BUSY);
    retainedTicket.reset();
    assert(retainedState->poll(fence) == OMNI_USD_LIVE_OK);
}

void concurrentAdmissionCaptureAndCompletion()
{
    DrainRegistry registry(1);
    for (unsigned round = 0; round < 64; ++round) {
        auto state = registry.makeState();
        auto ticket = state->prepare();
        DrainState::Fence fence;
        std::promise<void> start;
        auto gate = start.get_future().share();
        std::atomic<unsigned> ready{0};
        std::thread writer([&] { ++ready; gate.wait(); assert(ticket->admit() == OMNI_USD_LIVE_OK); });
        std::thread reader([&] { ++ready; gate.wait(); fence = state->capture(); });
        const auto deadline = Clock::now() + std::chrono::seconds(1);
        while (ready.load() != 2) { assert(Clock::now() < deadline); std::this_thread::yield(); }
        start.set_value(); writer.join(); reader.join();
        assert(fence.cutoff == 0 || fence.cutoff == 1);
        assert(state->poll(fence) == (fence.cutoff ? OMNI_USD_LIVE_BUSY : OMNI_USD_LIVE_OK));
        ticket->complete();
        assert(state->poll(fence) == OMNI_USD_LIVE_OK);
    }
    auto state = registry.makeState();
    auto ticket = state->prepare();
    std::promise<void> start;
    auto gate = start.get_future().share();
    std::atomic<unsigned> ready{0};
    std::vector<std::thread> writers;
    for (unsigned i = 0; i < 8; ++i)
        writers.emplace_back([&] { ++ready; gate.wait(); assert(ticket->admit() == OMNI_USD_LIVE_OK); });
    const auto deadline = Clock::now() + std::chrono::seconds(1);
    while (ready.load() != 8) { assert(Clock::now() < deadline); std::this_thread::yield(); }
    start.set_value();
    for (auto& writer : writers) writer.join();
    const auto fence = state->capture();
    assert(fence.cutoff == 1 && state->poll(fence) == OMNI_USD_LIVE_BUSY);
    ticket->complete();
    assert(state->poll(fence) == OMNI_USD_LIVE_OK);
}

void unknownHasPriorityAcrossStates()
{
    for (bool reverse : {false, true}) {
        DrainRegistry registry(2);
        auto first = registry.makeState(), second = registry.makeState();
        auto a = first->prepare(), b = second->prepare();
        assert(a->admit() == OMNI_USD_LIVE_OK && b->admit() == OMNI_USD_LIVE_OK);
        const auto captured = registry.capture();
        first->fail(reverse ? OMNI_USD_LIVE_OUTCOME_UNKNOWN : OMNI_USD_LIVE_UNAUTHENTICATED);
        second->fail(reverse ? OMNI_USD_LIVE_UNAUTHENTICATED : OMNI_USD_LIVE_OUTCOME_UNKNOWN);
        assert(registry.wait(captured, Clock::now()) == OMNI_USD_LIVE_OUTCOME_UNKNOWN);
        a->complete(); b->complete();
        assert(registry.wait(0) == OMNI_USD_LIVE_OUTCOME_UNKNOWN);
    }
}

void concurrentAdmissionAndRetirement()
{
    DrainRegistry registry(1);
    for (unsigned round = 0; round < 64; ++round) {
        auto state = registry.makeState();
        auto ticket = state->prepare();
        OmniUsdLiveResult result = OMNI_USD_LIVE_INTERNAL_ERROR;
        std::promise<void> start;
        auto gate = start.get_future().share();
        std::atomic<unsigned> ready{0};
        std::thread admission([&] { ++ready; gate.wait(); result = ticket->admit(); });
        std::thread completion([&] { ++ready; gate.wait(); ticket->complete(); });
        const auto deadline = Clock::now() + std::chrono::seconds(1);
        while (ready.load() != 2) { assert(Clock::now() < deadline); std::this_thread::yield(); }
        start.set_value(); admission.join(); completion.join();
        assert(result == OMNI_USD_LIVE_OK || result == OMNI_USD_LIVE_CANCELLED);
        const auto fence = state->capture();
        assert(fence.cutoff == (result == OMNI_USD_LIVE_OK ? 1u : 0u));
        assert(state->poll(fence) == OMNI_USD_LIVE_OK);
        assert(ticket->admit() == result); // Never resurrect an already retired node.
        assert(registry.wait(0) == OMNI_USD_LIVE_OK);
    }
}
} // namespace

void drainTrackerTests()
{
    capturedMembershipAndUnorderedCompletion();
    deadlineAndNonblockingWaitsDoNotMutate();
    terminalFailureBypassesEarlierPendingState();
    failuresRemainBoundToTheirCapturedObligations();
    terminalBusyIsNotPending();
    issuanceFailureKeepsAdmitting();
    unknownHasPriorityAcrossStates();
    inactiveRejectedAndClosedTickets();
    ticketAndRegistryOwnership();
    concurrentAdmissionCaptureAndCompletion();
    concurrentAdmissionAndRetirement();
    std::cout << "drain tracker: captured cutoffs, independent failures, ownership, and 128 admission races passed\n";
}
