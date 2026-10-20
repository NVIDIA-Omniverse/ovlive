// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "LiveUri.h"
#include <omni/usd/live/Live.h>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace omni::usd::live {
// Pure serial-worker reconciliation. Connection IDs are service identities;
// matching names never merge distinct peers. The controller owns its own
// connection ID and emits Connection before passing any peer messages here.
class SessionPeers final {
public:
    using Clock = std::chrono::steady_clock;
    struct Limits {
        size_t maxPeers = 1024;
        size_t maxBytes = 256u * 1024 * 1024;
        size_t maxTextBytes = kMaxUriBytes;
    };
    struct Identity {
        std::string connectionId, userName, appName;
    };
    struct Event {
        uint32_t kind;
        Identity peer;
    };
    SessionPeers(Limits limits, uint32_t graceMs);

    // Nonzero, increasing local stream generations. Stale generations do
    // nothing; repeating the current Connection is a malformed stream.
    void beginConnection(uint64_t generation, Clock::time_point now);
    // Freeze old-generation grace/events during an outage, retaining peers.
    void disconnected(uint64_t generation) noexcept;
    std::vector<Event> joined(uint64_t generation, const Identity& peer);
    std::vector<Event> left(uint64_t generation, std::string_view connectionId);
    std::vector<Event> expire(uint64_t generation, Clock::time_point now);
    Clock::time_point nextDeadline() const noexcept;
    uint64_t generation() const noexcept { return generation_; }
    size_t peerCount() const noexcept { return peers_.size(); }
    // Logical retained UTF-8 bytes (IDs + user names + application names),
    // excluding allocator/container overhead. Peer count bounds node overhead.
    size_t metadataBytes() const noexcept { return bytes_; }
    // Shared with controller self/sender validation. Embedded NUL and invalid
    // UTF-8 are malformed; oversize text is RESOURCE_EXHAUSTED. No allocation.
    static void validateText(std::string_view text, bool allowEmpty, size_t bound);

private:
    struct Peer {
        std::string userName, appName;
        uint64_t seenGeneration;
    };
    const Limits limits_;
    const uint32_t graceMs_;
    std::map<std::string, Peer, std::less<>> peers_;
    size_t bytes_ = 0;
    uint64_t generation_ = 0;
    bool connected_ = false, gracePending_ = false;
    Clock::time_point deadline_ = Clock::time_point::max();
    bool current(uint64_t generation) const;
};

// Write-silence schedule of the client keepalive on one JoinSession attempt
// (design section 15). Pure; the session worker owns it. Only a completed
// write restarts the interval: one write is outstanding per stream, so a
// queued or slow message has not reached the service yet.
class SessionKeepalive final {
public:
    using Clock = std::chrono::steady_clock;
    // Applies the validated Connection's keepalive_interval_ms; zero, which a
    // service without the client keepalive reports, disables the schedule.
    void connected(uint64_t intervalMs) noexcept { intervalMs_ = intervalMs; }
    // Any write on the stream, the header, a message or a keepalive, that
    // finished at `at`. A completion reported out of order never moves the
    // interval back.
    void writeCompleted(Clock::time_point at) noexcept {
        if (at > lastWrite_) lastWrite_ = at;
    }
    bool enabled() const noexcept { return intervalMs_ != 0; }
    // A full interval after the last completed write, saturating at max();
    // max() when disabled.
    Clock::time_point deadline() const noexcept;
    // Whether to write a keepalive at `now`: enabled, no write in flight, no
    // message queued and the deadline reached. A keepalive never preempts a
    // write, and a queued message goes out instead of one.
    bool due(Clock::time_point now, bool writing, bool queued) const noexcept {
        return enabled() && !writing && !queued && now >= deadline();
    }

private:
    uint64_t intervalMs_ = 0;
    Clock::time_point lastWrite_ = Clock::time_point::min();
};
} // namespace omni::usd::live
