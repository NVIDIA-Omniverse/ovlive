// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace omni::usd::live {
inline constexpr std::array<uint64_t, 8> kMetricDurationBoundsMs{1, 5, 10, 50, 100, 500, 1000, 5000};
inline size_t metricDurationBucket(uint64_t durationMs) noexcept {
    size_t bucket = 0;
    while (bucket < kMetricDurationBoundsMs.size() && durationMs > kMetricDurationBoundsMs[bucket]) ++bucket;
    return bucket;
}
struct MetricDuration {
    uint64_t count = 0, sumMs = 0;
    // Inclusive, noncumulative bounds; the final bucket is +Inf. Each field
    // saturates independently. Caller supplies synchronization and clock values.
    std::array<uint64_t, kMetricDurationBoundsMs.size() + 1> counts{};
    void observe(uint64_t durationMs) noexcept {
        if (count != UINT64_MAX) ++count;
        sumMs += std::min(UINT64_MAX - sumMs, durationMs);
        auto& bucket = counts[metricDurationBucket(durationMs)];
        if (bucket != UINT64_MAX) ++bucket;
    }
};
} // namespace omni::usd::live
