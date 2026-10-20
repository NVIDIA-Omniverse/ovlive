// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <omni/usd/live/Live.h>
#include <grpc/support/time.h>
#include <chrono>

namespace omni::usd::live {
// RPC budgets measure elapsed time. Keep the absolute deadline in gRPC's own
// monotonic clock domain, including the gap between context setup and issue.
// Do not assume std::chrono::steady_clock shares gRPC's epoch. The optional
// monotonic sample is an explicit clock seam for deterministic boundary tests.
inline gpr_timespec rpcDeadlineAfter(std::chrono::milliseconds budget,
    gpr_timespec now = gpr_now(GPR_CLOCK_MONOTONIC)) noexcept {
    return gpr_time_add(now, gpr_time_from_millis(budget.count(), GPR_TIMESPAN));
}

// Result for a wait whose operation deadline (request, initial open) has been
// reached. A wait that last evaluated without a usable credential ends with the
// auth error, whichever of that deadline and the auth-wait deadline the
// scheduler observes first (design section 14); any other expired wait is TIMEOUT.
// OK while the deadline is still ahead.
inline OmniUsdLiveResult expiredWait(std::chrono::steady_clock::time_point now,
    std::chrono::steady_clock::time_point deadline, bool credentialStarved) noexcept {
    if (now < deadline) return OMNI_USD_LIVE_OK;
    return credentialStarved ? OMNI_USD_LIVE_UNAUTHENTICATED : OMNI_USD_LIVE_TIMEOUT;
}
} // namespace omni::usd::live
