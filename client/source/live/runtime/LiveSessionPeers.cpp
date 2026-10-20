// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "LiveSessionPeers.h"
#include <algorithm>
#include <limits>

namespace omni::usd::live {
SessionPeers::SessionPeers(Limits limits, uint32_t graceMs) : limits_(limits), graceMs_(graceMs) {
    if (!limits_.maxPeers || !limits_.maxBytes || !limits_.maxTextBytes || limits_.maxTextBytes > kMaxUriBytes)
        throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
}

void SessionPeers::validateText(std::string_view text, bool allowEmpty, size_t bound) {
    if (text.size() > bound || text.size() > kMaxUriBytes) throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    if (!allowEmpty && text.empty()) throw OmniUsdLiveResult(OMNI_USD_LIVE_MALFORMED_STREAM);
    for (size_t index = 0; index < text.size();) {
        uint32_t point = static_cast<unsigned char>(text[index++]);
        uint32_t minimum = 0;
        unsigned extra = 0;
        if (point >= 0x80) {
            if (point >= 0xc2 && point <= 0xdf) {
                point &= 0x1f;
                extra = 1;
                minimum = 0x80;
            } else if (point >= 0xe0 && point <= 0xef) {
                point &= 0x0f;
                extra = 2;
                minimum = 0x800;
            } else if (point >= 0xf0 && point <= 0xf4) {
                point &= 0x07;
                extra = 3;
                minimum = 0x10000;
            } else throw OmniUsdLiveResult(OMNI_USD_LIVE_MALFORMED_STREAM);
            if (extra > text.size() - index) throw OmniUsdLiveResult(OMNI_USD_LIVE_MALFORMED_STREAM);
            while (extra--) {
                const auto next = static_cast<unsigned char>(text[index++]);
                if ((next & 0xc0) != 0x80) throw OmniUsdLiveResult(OMNI_USD_LIVE_MALFORMED_STREAM);
                point = (point << 6) | (next & 0x3f);
            }
        }
        if (!point || point < minimum || point > 0x10ffff || (point >= 0xd800 && point <= 0xdfff))
            throw OmniUsdLiveResult(OMNI_USD_LIVE_MALFORMED_STREAM);
    }
}

bool SessionPeers::current(uint64_t generation) const {
    if (!generation) throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
    if (generation < generation_ || (generation == generation_ && !connected_)) return false;
    if (generation != generation_) throw OmniUsdLiveResult(OMNI_USD_LIVE_MALFORMED_STREAM);
    return true;
}

void SessionPeers::beginConnection(uint64_t generation, Clock::time_point now) {
    if (!generation) throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
    if (generation < generation_) return;
    if (generation == generation_) throw OmniUsdLiveResult(OMNI_USD_LIVE_MALFORMED_STREAM);
    const auto delay = std::chrono::duration_cast<Clock::duration>(std::chrono::milliseconds(graceMs_));
    const auto deadline = now > Clock::time_point::max() - delay ? Clock::time_point::max() : now + delay;
    generation_ = generation;
    connected_ = true;
    gracePending_ = !peers_.empty();
    deadline_ = gracePending_ ? deadline : Clock::time_point::max();
}

void SessionPeers::disconnected(uint64_t generation) noexcept {
    if (generation != generation_) return;
    connected_ = false;
    gracePending_ = false;
    deadline_ = Clock::time_point::max();
}

std::vector<SessionPeers::Event> SessionPeers::joined(uint64_t generation, const Identity& peer) {
    if (!current(generation)) return {};
    validateText(peer.connectionId, false, limits_.maxTextBytes);
    validateText(peer.userName, true, limits_.maxTextBytes);
    validateText(peer.appName, true, limits_.maxTextBytes);
    auto found = peers_.find(peer.connectionId);
    if (found != peers_.end()) {
        if (found->second.userName != peer.userName || found->second.appName != peer.appName)
            throw OmniUsdLiveResult(OMNI_USD_LIVE_MALFORMED_STREAM);
        found->second.seenGeneration = generation;
        return {}; // Presence refresh, including duplicate initial enumeration.
    }
    // Each field is bounded by 64KiB, so this sum cannot overflow size_t on
    // supported 32/64-bit hosts. Check count/aggregate bounds before copying.
    const auto size = peer.connectionId.size() + peer.userName.size() + peer.appName.size();
    if (peers_.size() >= limits_.maxPeers || size > limits_.maxBytes - bytes_)
        throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    std::vector<Event> events;
    events.push_back({OMNI_USD_LIVE_EVENT_JOINED, peer});
    // Event copies and the map allocation all precede the first state change.
    peers_.emplace(peer.connectionId, Peer{peer.userName, peer.appName, generation});
    bytes_ += size;
    return events;
}

std::vector<SessionPeers::Event> SessionPeers::left(uint64_t generation, std::string_view connectionId) {
    if (!current(generation)) return {};
    validateText(connectionId, false, limits_.maxTextBytes);
    const auto found = peers_.find(connectionId);
    if (found == peers_.end()) return {};
    const auto& peer = found->second;
    std::vector<Event> events;
    events.push_back({OMNI_USD_LIVE_EVENT_LEFT, {found->first, peer.userName, peer.appName}});
    bytes_ -= found->first.size() + peer.userName.size() + peer.appName.size();
    peers_.erase(found);
    return events;
}

std::vector<SessionPeers::Event> SessionPeers::expire(uint64_t generation, Clock::time_point now) {
    if (!current(generation) || !gracePending_ || now < deadline_) return {};
    std::vector<Event> events;
    const auto absent = static_cast<size_t>(std::count_if(peers_.begin(), peers_.end(),
        [&](const auto& item) { return item.second.seenGeneration != generation; }));
    events.reserve(absent);
    for (const auto& item : peers_) {
        const auto& peer = item.second;
        if (peer.seenGeneration != generation)
            events.push_back({OMNI_USD_LIVE_EVENT_LEFT, {item.first, peer.userName, peer.appName}});
    }
    // An allocation failure above retains every old peer and the armed timer.
    // Ordered map traversal gives deterministic callback order for a timeout.
    for (auto iter = peers_.begin(); iter != peers_.end();) {
        if (iter->second.seenGeneration != generation) {
            bytes_ -= iter->first.size() + iter->second.userName.size() + iter->second.appName.size();
            iter = peers_.erase(iter);
        } else ++iter;
    }
    gracePending_ = false;
    deadline_ = Clock::time_point::max();
    return events;
}

SessionPeers::Clock::time_point SessionPeers::nextDeadline() const noexcept {
    return connected_ && gracePending_ ? deadline_ : Clock::time_point::max();
}
SessionKeepalive::Clock::time_point SessionKeepalive::deadline() const noexcept {
    if (!enabled()) return Clock::time_point::max();
    // Before the header write completes there is no base; the header is then
    // in flight, so due() is false whatever the deadline.
    const auto base = lastWrite_ == Clock::time_point::min() ? Clock::time_point{} : lastWrite_;
    const auto room = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::time_point::max() - base);
    if (intervalMs_ >= static_cast<uint64_t>(room.count())) return Clock::time_point::max();
    return base + std::chrono::milliseconds(intervalMs_);
}
} // namespace omni::usd::live
