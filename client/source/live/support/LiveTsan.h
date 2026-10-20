// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

// ThreadSanitizer cannot see a happens-before edge that runs through
// uninstrumented code (the gRPC completion queue, USD and TBB); these annotate
// the product handoffs across it. No-ops unless the build uses
// -fsanitize=thread.
#if defined(__SANITIZE_THREAD__)
#    define LIVE_TSAN 1
#elif defined(__has_feature)
#    if __has_feature(thread_sanitizer)
#        define LIVE_TSAN 1
#    endif
#endif
#if defined(LIVE_TSAN)
#    include <sanitizer/tsan_interface.h>
#    define LIVE_TSAN_RELEASE(address) __tsan_release(const_cast<void*>(static_cast<const void*>(address)))
#    define LIVE_TSAN_ACQUIRE(address) __tsan_acquire(const_cast<void*>(static_cast<const void*>(address)))
#else
#    define LIVE_TSAN_RELEASE(address) ((void)0)
#    define LIVE_TSAN_ACQUIRE(address) ((void)0)
#endif
