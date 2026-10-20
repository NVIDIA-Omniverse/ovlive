// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Deadlocks two worker threads on purpose for stack_dump_check.py: each parks
// forever in liveStackDumpDeadlockOnPurpose. A third thread blocks every
// signal, so it never answers a dump and each dump counts it as missing. The
// main thread prints ready once all three are parked.
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <mutex>
#include <pthread.h>
#include <thread>

namespace {
std::mutex mutex;
std::condition_variable changed;
int parked = 0;
}

extern "C" __attribute__((noinline)) void liveStackDumpDeadlockOnPurpose() {
    std::unique_lock<std::mutex> lock(mutex);
    ++parked;
    changed.notify_all();
    // The counter is incremented under the mutex this wait releases, so a
    // thread the main thread counts is already parked.
    changed.wait(lock, [] { return false; });
}

void neverAnswers() {
    sigset_t all;
    sigfillset(&all);
    pthread_sigmask(SIG_BLOCK, &all, nullptr);
    liveStackDumpDeadlockOnPurpose();
}

int main() {
    std::thread first(liveStackDumpDeadlockOnPurpose), second(liveStackDumpDeadlockOnPurpose), third(neverAnswers);
    {
        std::unique_lock<std::mutex> lock(mutex);
        changed.wait(lock, [] { return parked == 3; });
    }
    std::printf("ready\n");
    std::fflush(stdout);
    first.join();
    second.join();
    third.join();
    return 0;
}
