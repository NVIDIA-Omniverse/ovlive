// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Deadlocks worker threads on purpose for stack_dump_check.py: each parks
// forever in liveStackDumpDeadlockOnPurpose. A third thread blocks every
// signal, so it never answers a dump and each dump counts it as missing. On
// x86-64 and aarch64 a fourth parks below liveStackDumpBadUnwindFrame, whose
// unwind information is wrong on purpose, so unwinding it faults as unwinding a
// frame without accurate asynchronous unwind tables can. The main thread prints
// ready once all of them are parked.
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

// The frame's CFA is a callee-saved register that holds 16, so the unwinder reads its return address from address 8.
#if defined(__x86_64__)
#define LIVE_BAD_UNWIND 1
asm(R"(
    .pushsection .text
    .globl liveStackDumpBadUnwindFrame
    .type liveStackDumpBadUnwindFrame, @function
liveStackDumpBadUnwindFrame:
    .cfi_startproc
    push %r12
    .cfi_adjust_cfa_offset 8
    mov $16, %r12
    .cfi_def_cfa %r12, 0
    call liveStackDumpDeadlockOnPurpose
    ud2
    .cfi_endproc
    .size liveStackDumpBadUnwindFrame, .-liveStackDumpBadUnwindFrame
    .popsection
)");
#elif defined(__aarch64__)
#define LIVE_BAD_UNWIND 1
asm(R"(
    .pushsection .text
    .globl liveStackDumpBadUnwindFrame
    .type liveStackDumpBadUnwindFrame, %function
liveStackDumpBadUnwindFrame:
    .cfi_startproc
    stp x29, x30, [sp, #-32]!
    str x19, [sp, #16]
    mov x19, #16
    .cfi_def_cfa x19, 0
    .cfi_offset x29, -16
    .cfi_offset x30, -8
    bl liveStackDumpDeadlockOnPurpose
    brk #0
    .cfi_endproc
    .size liveStackDumpBadUnwindFrame, .-liveStackDumpBadUnwindFrame
    .popsection
)");
#else
#define LIVE_BAD_UNWIND 0
#endif
#if LIVE_BAD_UNWIND
extern "C" void liveStackDumpBadUnwindFrame();
#endif

void neverAnswers() {
    sigset_t all;
    sigfillset(&all);
    pthread_sigmask(SIG_BLOCK, &all, nullptr);
    liveStackDumpDeadlockOnPurpose();
}

int main() {
    std::thread first(liveStackDumpDeadlockOnPurpose), second(liveStackDumpDeadlockOnPurpose), third(neverAnswers);
#if LIVE_BAD_UNWIND
    std::thread fourth(liveStackDumpBadUnwindFrame);
#endif
    {
        std::unique_lock<std::mutex> lock(mutex);
        changed.wait(lock, [] { return parked == 3 + LIVE_BAD_UNWIND; });
    }
    std::printf("ready\n");
    std::fflush(stdout);
    first.join();
    second.join();
    third.join();
#if LIVE_BAD_UNWIND
    fourth.join();
#endif
    return 0;
}
