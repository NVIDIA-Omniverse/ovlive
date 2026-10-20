// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <omni/usd/live/Live.h>
#include <new>
#include <stdexcept>

namespace omni::usd::live {
// The public result of the exception being handled: a thrown OmniUsdLiveResult
// is itself, an allocation or length failure is RESOURCE_EXHAUSTED, an
// std::invalid_argument is INVALID_ARGUMENT and anything else is
// INTERNAL_ERROR. Call only inside a catch handler. A caller that maps further
// exception types catches them first and delegates here from catch (...).
inline OmniUsdLiveResult currentExceptionResult() noexcept {
    try {
        throw;
    } catch (OmniUsdLiveResult result) { return result; } catch (const std::bad_alloc&) {
        return OMNI_USD_LIVE_RESOURCE_EXHAUSTED;
    } catch (const std::length_error&) {
        return OMNI_USD_LIVE_RESOURCE_EXHAUSTED;
    } catch (const std::invalid_argument&) { return OMNI_USD_LIVE_INVALID_ARGUMENT; } catch (...) {
        return OMNI_USD_LIVE_INTERNAL_ERROR;
    }
}
// Runs `action`; an escaping exception becomes currentExceptionResult().
template<class Action> OmniUsdLiveResult resultBoundary(Action&& action) noexcept {
    try {
        return action();
    } catch (...) { return currentExceptionResult(); }
}
} // namespace omni::usd::live
