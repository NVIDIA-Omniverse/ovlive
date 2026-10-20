// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "LiveJitter.h"
#include <algorithm>
#include <limits>
#include <cassert>
#include <type_traits>

namespace omni::usd::live {
namespace {
uint64_t add(uint64_t a, uint64_t b) noexcept { return b > UINT64_MAX - a ? UINT64_MAX : a + b; }
uint64_t multiply(uint64_t a, uint64_t b) noexcept { return b && a > UINT64_MAX / b ? UINT64_MAX : a * b; }
uint64_t elapsed(JitterEstimator::Clock::time_point a, JitterEstimator::Clock::time_point b) noexcept {
    if (a <= b) return 0;
    // Convert each operand first: subtraction of extreme native time points
    // can overflow the clock's signed duration even when microseconds fit.
    const auto x = std::chrono::duration_cast<std::chrono::microseconds>(a.time_since_epoch()).count();
    const auto y = std::chrono::duration_cast<std::chrono::microseconds>(b.time_since_epoch()).count();
    return uint64_t(x) - uint64_t(y);
}
uint64_t smooth(uint64_t base, uint64_t target, uint64_t elapsedUs) noexcept {
    const auto limit = elapsedUs / 100;
    return target >= base ? base + std::min(target - base, limit) : base - std::min(base - target, limit);
}
JitterEstimator::Clock::time_point later(JitterEstimator::Clock::time_point now, uint64_t us) noexcept {
    using Clock = JitterEstimator::Clock;
    // All callers cap us at uint32 milliseconds. Convert that bounded duration
    // first, then compare the addition without subtracting extreme time points.
    const auto delta = std::chrono::duration_cast<Clock::duration>(std::chrono::microseconds(us));
    if (now > Clock::time_point::max() - delta) return Clock::time_point::max();
    return now + delta;
}
}
void JitterEstimator::reset() noexcept { *this = JitterEstimator{}; }
void JitterEstimator::sample(uint64_t latency, uint64_t server, Clock::time_point received) noexcept {
    if (!server || (valid_ && received < baseTime_)) return;
    history_[next_] = latency;
    next_ = (next_ + 1) % history_.size();
    count_ = std::min(count_ + 1, history_.size());
    uint64_t target = 0;
    if (count_ < 5) {
        uint64_t remainder = 0;
        for (size_t i = 0; i < count_; ++i) {
            target += history_[i] / count_;
            remainder += history_[i] % count_;
        }
        target += remainder / count_;
    } else {
        auto sorted = history_; // Sorting must not corrupt ring eviction order.
        std::sort(sorted.begin(), sorted.begin() + count_);
        const auto q1 = sorted[count_ / 4], q3 = sorted[count_ * 3 / 4];
        const auto fence = add(q3, multiply(q3 - q1, 2));
        for (size_t i = count_; i; --i)
            if (sorted[i - 1] <= fence) {
                target = sorted[i - 1];
                break;
            }
    }
    if (valid_) {
        const auto dt = elapsed(received, baseTime_);
        serverBase_ = add(smooth(serverBase_, serverTarget_, dt), dt);
        latencyBase_ = smooth(latencyBase_, latencyTarget_, dt);
    } else {
        serverBase_ = add(server, latency);
        latencyBase_ = target;
    }
    serverTarget_ = add(server, latency);
    latencyTarget_ = target;
    baseTime_ = received;
    valid_ = true;
}
JitterEstimator::Decision JitterEstimator::decide(uint64_t server, Clock::time_point received, Clock::time_point now,
    const OmniUsdLiveJitterParams& params) const noexcept {
    const auto maximum = uint64_t(params.delay_maximum_ms) * 1000;
    Decision decision{now};
    if (!maximum) return decision;
    decision.sampled = valid_ && server;
    decision.sampleAgeUs = decision.sampled ? elapsed(now, baseTime_) : 0;
    const auto constant = uint64_t(params.delay_constant_ms) * 1000;
    const auto dt = decision.sampleAgeUs;
    const auto latency = smooth(latencyBase_, latencyTarget_, dt);
    const auto hold = decision.targetHoldUs = decision.sampled ?
        std::min(add(constant, multiply(params.delay_multiple, latency)), maximum) :
        std::min(constant, maximum);
    const auto deadline = later(received, maximum);
    if (now >= deadline) return decision;
    if (!decision.sampled) {
        decision.due = std::min(deadline, later(received, hold));
        return decision;
    }
    const auto estimated = add(smooth(serverBase_, serverTarget_, dt), dt);
    const auto remaining = estimated < hold ? std::min(add(hold - estimated, server), maximum) :
        server > estimated - hold           ? std::min(server - (estimated - hold), maximum) :
                                              0;
    decision.due = std::min(deadline, later(now, remaining));
    return decision;
}

void JitterWait::begin(uint64_t work, Clock::time_point checked, Clock::time_point due) noexcept {
    assert(!active_ && due > checked);
    active_ = true;
    work_ = work;
    checked_ = checked;
    due_ = due;
    heldTicks_ = lateTicks_ = 0;
}
void JitterWait::charge(Clock::time_point checked) noexcept {
    if (checked <= checked_) return;
    using Unsigned = std::make_unsigned_t<Clock::duration::rep>;
    const auto ticks = [](Clock::time_point end, Clock::time_point start) -> uint64_t {
        return end <= start ? 0 : Unsigned(end.time_since_epoch().count()) - Unsigned(start.time_since_epoch().count());
    };
    heldTicks_ = add(heldTicks_, ticks(std::min(checked, due_), checked_));
    lateTicks_ = add(lateTicks_, ticks(checked, std::max(checked_, due_)));
    checked_ = checked;
}
void JitterWait::observe(Clock::time_point checked, Clock::time_point due) noexcept {
    assert(active_);
    if (checked < checked_) return;
    charge(checked);
    due_ = due;
}
std::optional<JitterWait::Measurement> JitterWait::finish(Clock::time_point checked) noexcept {
    if (!active_) return {};
    charge(checked);
    active_ = false;
    const auto ms = [](uint64_t ticks) {
        return std::chrono::duration_cast<std::chrono::duration<uint64_t, std::milli>>(
            std::chrono::duration<uint64_t, Clock::period>(ticks))
            .count();
    };
    return Measurement{ms(heldTicks_), ms(lateTicks_)};
}
} // namespace omni::usd::live
