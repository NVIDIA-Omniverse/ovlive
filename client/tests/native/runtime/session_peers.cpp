// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "LiveSessionPeers.h"
#include <cassert>
#include <iostream>
#include <new>

void liveTestFailAllocationsAfter(size_t) noexcept;
void liveTestRestoreAllocations() noexcept;
using namespace omni::usd::live;
namespace {
using Clock = SessionPeers::Clock;
using Identity = SessionPeers::Identity;
using Events = std::vector<SessionPeers::Event>;
using Ms = std::chrono::milliseconds;
Clock::time_point at(uint32_t value) { return Clock::time_point{} + Ms(value); }
size_t bytes(const Identity& value)
{ return value.connectionId.size() + value.userName.size() + value.appName.size(); }
void event(const Events& values, size_t index, uint32_t kind, const Identity& expected)
{
    assert(index < values.size());
    const auto& actual = values[index];
    assert(actual.kind == kind && actual.peer.connectionId == expected.connectionId &&
           actual.peer.userName == expected.userName && actual.peer.appName == expected.appName);
}
template <class Action> void rejected(OmniUsdLiveResult result, Action action)
{
    bool caught = false;
    try { action(); }
    catch (OmniUsdLiveResult error) { assert(error == result); caught = true; }
    assert(caught);
}
struct AllocationFailure {
    explicit AllocationFailure(size_t after) { liveTestFailAllocationsAfter(after); }
    ~AllocationFailure() { liveTestRestoreAllocations(); }
};

void generationReconciliation()
{
    SessionPeers peers({8, 1024, 64}, 10);
    const Identity a{"a", "same-user", "same-app"}, b{"b", "same-user", "same-app"},
                   c{"c", "new-user", "new-app"};
    rejected(OMNI_USD_LIVE_MALFORMED_STREAM, [&] { peers.joined(1, a); });
    peers.beginConnection(1, at(0));
    assert(peers.nextDeadline() == Clock::time_point::max());
    event(peers.joined(1, a), 0, OMNI_USD_LIVE_EVENT_JOINED, a);
    event(peers.joined(1, b), 0, OMNI_USD_LIVE_EVENT_JOINED, b); // Same names, distinct service IDs.
    assert(peers.joined(1, a).empty() && peers.peerCount() == 2);
    assert(peers.metadataBytes() == bytes(a) + bytes(b));
    rejected(OMNI_USD_LIVE_MALFORMED_STREAM, [&] { peers.beginConnection(1, at(5)); });
    peers.disconnected(1);
    assert(peers.expire(1, at(100)).empty() && peers.joined(1, c).empty());
    assert(peers.nextDeadline() == Clock::time_point::max() && peers.peerCount() == 2);
    peers.beginConnection(2, at(100));
    const auto deadline = peers.nextDeadline();
    assert(deadline == at(110));
    assert(peers.joined(2, a).empty()); // Re-enumeration does not repeat Joined.
    event(peers.joined(2, c), 0, OMNI_USD_LIVE_EVENT_JOINED, c);
    peers.beginConnection(1, at(1000));
    peers.disconnected(1);
    assert(peers.left(1, "a").empty() && peers.expire(1, at(1000)).empty());
    assert(peers.nextDeadline() == deadline && peers.peerCount() == 3);
    assert(peers.expire(2, at(109)).empty());
    const auto expired = peers.expire(2, at(110));
    assert(expired.size() == 1); event(expired, 0, OMNI_USD_LIVE_EVENT_LEFT, b);
    assert(peers.peerCount() == 2 && peers.metadataBytes() == bytes(a) + bytes(c));
    assert(peers.expire(2, at(200)).empty());
    event(peers.left(2, "c"), 0, OMNI_USD_LIVE_EVENT_LEFT, c);
    assert(peers.left(2, "c").empty());
    // Events own their complete metadata even after reducer mutation/destruction.
    assert(expired[0].peer.userName == b.userName);
    peers.beginConnection(3, at(210));
    peers.disconnected(3);
    assert(peers.expire(3, at(300)).empty());
    peers.beginConnection(4, at(400));
    assert(peers.expire(3, at(1000)).empty());
    event(peers.expire(4, at(410)), 0, OMNI_USD_LIVE_EVENT_LEFT, a);
    assert(peers.peerCount() == 0 && peers.metadataBytes() == 0);
}

void immediateLeftAndTimerBounds()
{
    const Identity a{"a", "", ""}, b{"b", "", ""};
    SessionPeers peers({2, 8, 8}, 0);
    peers.beginConnection(1, at(0));
    peers.joined(1, a); peers.joined(1, b);
    peers.beginConnection(2, at(1));
    event(peers.left(2, "a"), 0, OMNI_USD_LIVE_EVENT_LEFT, a);
    const auto expired = peers.expire(2, at(1));
    assert(expired.size() == 1); event(expired, 0, OMNI_USD_LIVE_EVENT_LEFT, b);
    assert(peers.nextDeadline() == Clock::time_point::max());
    peers.joined(2, a);
    peers.beginConnection(UINT64_MAX, Clock::time_point::max());
    event(peers.expire(UINT64_MAX, Clock::time_point::max()), 0, OMNI_USD_LIVE_EVENT_LEFT, a);
    SessionPeers saturated({1, 8, 8}, UINT32_MAX);
    saturated.beginConnection(1, at(0)); saturated.joined(1, a);
    saturated.beginConnection(2, Clock::time_point::max() - Ms(1));
    assert(saturated.nextDeadline() == Clock::time_point::max());
    assert(saturated.expire(2, Clock::time_point::max() - Ms(1)).empty());
    event(saturated.expire(2, Clock::time_point::max()), 0, OMNI_USD_LIVE_EVENT_LEFT, a);
}

void validationAndQuotas()
{
    const Identity a{"a", "user", "app"};
    SessionPeers peers({1, bytes(a), 16}, 5);
    peers.beginConnection(1, at(0)); peers.joined(1, a);
    rejected(OMNI_USD_LIVE_RESOURCE_EXHAUSTED, [&] { peers.joined(1, {"b", "", ""}); });
    peers.beginConnection(2, at(10));
    auto conflicting = a; conflicting.appName = "different";
    rejected(OMNI_USD_LIVE_MALFORMED_STREAM, [&] { peers.joined(2, conflicting); });
    assert(peers.peerCount() == 1 && peers.metadataBytes() == bytes(a));
    // Conflicting metadata must not refresh its generation before rejecting.
    event(peers.expire(2, at(15)), 0, OMNI_USD_LIVE_EVENT_LEFT, a);
    SessionPeers byteBound({5, 3, 16}, 1);
    byteBound.beginConnection(1, at(0));
    rejected(OMNI_USD_LIVE_RESOURCE_EXHAUSTED, [&] { byteBound.joined(1, a); });
    assert(!byteBound.peerCount() && !byteBound.metadataBytes());
    event(byteBound.joined(1, {"a", "b", "c"}), 0, OMNI_USD_LIVE_EVENT_JOINED, {"a", "b", "c"});
    for (const std::string& text : {std::string{}, std::string("a\0b", 3), std::string("\xc0\xaf", 2),
                                  std::string("\xe2\x82", 2), std::string("\xed\xa0\x80", 3),
                                  std::string("\xf4\x90\x80\x80", 4), std::string("\x80", 1)}) {
        rejected(OMNI_USD_LIVE_MALFORMED_STREAM, [&] { SessionPeers::validateText(text, false, 16); });
    }
    SessionPeers::validateText("\xf0\x9f\x8c\x90", false, 4);
    SessionPeers::validateText("", true, 4);
    rejected(OMNI_USD_LIVE_RESOURCE_EXHAUSTED, [&] { SessionPeers::validateText("four", false, 3); });
    SessionPeers malformed({4, 1024, 16}, 10);
    malformed.beginConnection(1, at(0));
    for (unsigned field = 0; field < 3; ++field) {
        auto bad = a;
        (field == 0 ? bad.connectionId : field == 1 ? bad.userName : bad.appName) = std::string("\xff", 1);
        rejected(OMNI_USD_LIVE_MALFORMED_STREAM, [&] { malformed.joined(1, bad); });
        assert(!malformed.peerCount() && !malformed.metadataBytes());
    }
    rejected(OMNI_USD_LIVE_MALFORMED_STREAM, [&] { malformed.left(1, ""); });
    rejected(OMNI_USD_LIVE_MALFORMED_STREAM, [&] { malformed.joined(2, a); });
    rejected(OMNI_USD_LIVE_INVALID_ARGUMENT, [&] { malformed.beginConnection(0, at(0)); });
    rejected(OMNI_USD_LIVE_INVALID_ARGUMENT, [&] { SessionPeers bad({0, 10, 10}, 1); });
}

void allocationAtomicity()
{
    const Identity a{std::string(80, 'a'), std::string(80, 'u'), std::string(80, 'p')};
    const Identity b{std::string(80, 'b'), std::string(80, 'v'), std::string(80, 'q')};
    const Identity c{std::string(80, 'c'), std::string(80, 'w'), std::string(80, 'r')};
    for (unsigned operation = 0; operation < 3; ++operation) {
        unsigned failures = 0; bool succeeded = false;
        for (size_t budget = 0; budget < 32; ++budget) {
            SessionPeers peers({4, 4096, 256}, 10);
            peers.beginConnection(1, at(0)); peers.joined(1, a);
            if (operation == 2) { peers.joined(1, b); peers.joined(1, c); peers.beginConnection(2, at(1)); peers.joined(2, c); }
            const auto count = peers.peerCount(), size = peers.metadataBytes();
            const auto deadline = peers.nextDeadline();
            Events result;
            try {
                AllocationFailure failure(budget);
                result = operation == 0 ? peers.joined(1, b) : operation == 1 ? peers.left(1, a.connectionId) : peers.expire(2, at(11));
                succeeded = true;
            } catch (const std::bad_alloc&) { ++failures; }
            if (succeeded) {
                assert(result.size() == (operation == 2 ? 2u : 1u));
                break;
            }
            assert(peers.peerCount() == count && peers.metadataBytes() == size && peers.nextDeadline() == deadline);
            // Retrying after failure must deliver the complete original event
            // set once; an allocation failure never silently consumes a peer.
            result = operation == 0 ? peers.joined(1, b) : operation == 1 ? peers.left(1, a.connectionId) : peers.expire(2, at(11));
            assert(result.size() == (operation == 2 ? 2u : 1u));
        }
        assert(succeeded && failures >= 4);
    }
}
}

// The keepalive interval runs from the last completed write, never from a
// queued one, and a keepalive never preempts a write in flight or a queued message.
void keepaliveSchedule()
{
    SessionKeepalive disabled;
    disabled.writeCompleted(at(100)); // The join header.
    disabled.connected(0);
    assert(!disabled.enabled() && disabled.deadline() == Clock::time_point::max());
    assert(!disabled.due(at(1000000), false, false) && !disabled.due(Clock::time_point::max(), false, false));

    SessionKeepalive keepalive;
    assert(!keepalive.enabled() && !keepalive.due(Clock::time_point::max(), false, false));
    keepalive.writeCompleted(at(100)); // The join header.
    keepalive.connected(50);
    assert(keepalive.enabled() && keepalive.deadline() == at(150));
    assert(!keepalive.due(at(149), false, false) && keepalive.due(at(150), false, false));
    // A write in flight or a queued message goes first; it restarts the interval when it completes.
    assert(!keepalive.due(at(150), true, false) && !keepalive.due(at(150), false, true));
    assert(!keepalive.due(Clock::time_point::max(), true, false));
    // An active stream sends none: each completed write moves the deadline a full interval on.
    for (uint32_t written = 140; written <= 1000; written += 40) {
        keepalive.writeCompleted(at(written));
        assert(keepalive.deadline() == at(written + 50));
        assert(!keepalive.due(at(written + 49), false, false));
    }
    assert(keepalive.due(at(1050), false, false));
    // A write that completed at an earlier time never moves the deadline back.
    keepalive.writeCompleted(at(1060)); // The keepalive itself.
    keepalive.writeCompleted(at(1000));
    assert(keepalive.deadline() == at(1110) && !keepalive.due(at(1109), false, false));
    // Before its header write completes, an attempt is never idle: the header is in flight.
    SessionKeepalive early;
    early.connected(50);
    assert(!early.due(at(1000), true, false));
    early.writeCompleted(at(900));
    assert(early.deadline() == at(950) && !early.due(at(949), false, false) && early.due(at(950), false, false));
    // An interval past the clock's range saturates; only an expired-for-test step reaches it.
    SessionKeepalive saturated;
    saturated.writeCompleted(Clock::time_point::max() - Ms(1));
    saturated.connected(UINT64_MAX);
    assert(saturated.deadline() == Clock::time_point::max());
    assert(!saturated.due(Clock::time_point::max() - Ms(1), false, false));
    assert(saturated.due(Clock::time_point::max(), false, false));
    SessionKeepalive wide;
    wide.writeCompleted(at(0));
    wide.connected(UINT64_MAX);
    assert(wide.deadline() == Clock::time_point::max());
}

void sessionPeersTests()
{
    generationReconciliation(); immediateLeftAndTimerBounds(); validationAndQuotas(); allocationAtomicity();
    keepaliveSchedule();
    std::cout << "session peers: generation grace, identities, metadata quotas, UTF-8, allocation atomicity and keepalive schedule passed\n";
}
