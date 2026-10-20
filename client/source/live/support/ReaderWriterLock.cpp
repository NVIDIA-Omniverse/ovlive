// SPDX-FileCopyrightText: Copyright (c) 2018-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "ReaderWriterLock.h"
#include "ThreadingUtils.h"

#if defined(__linux__)
#    include <sys/syscall.h>
#    include <unistd.h>
#elif defined(__APPLE__)
#    include <pthread.h>
#elif defined(_WIN32)
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    include <windows.h>
#endif

namespace threading {
void tprintf(std::stringstream& stm, const char* format) { stm << format; }

uint64_t GetThreadId() {
#ifdef _WIN32
    return GetCurrentThreadId();
#elif defined(__linux__)
    return syscall(__NR_gettid);
#elif defined(__APPLE__)
    uint64_t threadId = 0;
    pthread_threadid_np(nullptr, &threadId);
    return threadId;
#endif
}

} // namespace threading
