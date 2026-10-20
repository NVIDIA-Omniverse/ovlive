// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <omni/usd/live/Live.h>
#include <string>
#include <chrono>

namespace omni::usd::live {
// Inherit the caller's absolute budget through synchronous USD resolver calls.
// Thread-confined lexical scopes can only tighten a deadline; unwinding restores
// the previous one. No timers, allocations or callbacks are involved.
class MetadataDeadlineScope final {
public:
    using TimePoint = std::chrono::steady_clock::time_point;
    explicit MetadataDeadlineScope(TimePoint) noexcept;
    ~MetadataDeadlineScope();
    MetadataDeadlineScope(const MetadataDeadlineScope&) = delete;
    MetadataDeadlineScope& operator=(const MetadataDeadlineScope&) = delete;
    static TimePoint limit(TimePoint) noexcept;
private:
    const TimePoint previous_;
};
// Private synchronous resolver bridge. Uses finite-operation auth/retry policy
// and one initial-open deadline, including callback delivery. Never blocks a
// dispatch worker. Failure leaves output unchanged; timed-out work retains its
// owned callback state and retires asynchronously without a public handle leak.
// Optional private wire timestamp preserves submillisecond changes for Ar;
// the public ABI's explicitly millisecond metadata remains unchanged.
OmniUsdLiveResult statLayerMetadata(const std::string& uri, OmniUsdLiveLayerInfo& output,
    uint64_t* modifiedTimeUs = nullptr) noexcept;
}
