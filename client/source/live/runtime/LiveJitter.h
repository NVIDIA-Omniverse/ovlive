// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <omni/usd/live/Live.h>
#include <array>
#include <chrono>
#include <cstdint>
#include <optional>

namespace omni::usd::live {
// Private estimator, confined to one layer actor worker/read generation.
// No wall-clock arithmetic, allocation, sleeping or host processing hook.
class JitterEstimator {
public:
    using Clock = std::chrono::steady_clock;
    void reset() noexcept;
    void sample(uint64_t oneWayUs, uint64_t serverUs, Clock::time_point received) noexcept;
    struct Decision {
        Clock::time_point due;
        uint64_t targetHoldUs = 0, sampleAgeUs = 0;
        bool sampled = false;
    };
    Decision decide(uint64_t serverUs, Clock::time_point received, Clock::time_point now,
        const OmniUsdLiveJitterParams& params) const noexcept;
    Clock::time_point due(uint64_t serverUs, Clock::time_point received, Clock::time_point now,
        const OmniUsdLiveJitterParams& params) const noexcept {
        return decide(serverUs, received, now, params).due;
    }
private:
    std::array<uint64_t, 30> history_{};
    size_t count_ = 0, next_ = 0;
    bool valid_ = false;
    Clock::time_point baseTime_{};
    uint64_t serverBase_ = 0, serverTarget_ = 0, latencyBase_ = 0, latencyTarget_ = 0;
};

// Worker-only observation of one head-effect waiting episode. Intervals use
// the last observed due boundary; configuration is never retroactive.
// Accumulate native ticks before conversion so sub-ms wakes do not lose time.
enum class JitterWaitResolution : uint32_t { Eligible = 0, Preempted = 1, Failed = 2, Closed = 3 };
class JitterWait {
public:
    using Clock = JitterEstimator::Clock;
    struct Measurement {
        uint64_t holdMs, postDueMs;
    };
    bool active() const noexcept { return active_; }
    uint64_t work() const noexcept { return work_; }
    void begin(uint64_t work, Clock::time_point checked, Clock::time_point due) noexcept;
    void observe(Clock::time_point checked, Clock::time_point due) noexcept;
    std::optional<Measurement> finish(Clock::time_point checked) noexcept;
private:
    void charge(Clock::time_point checked) noexcept;
    bool active_ = false;
    uint64_t work_ = 0, heldTicks_ = 0, lateTicks_ = 0;
    Clock::time_point checked_{}, due_{};
};
} // namespace omni::usd::live
