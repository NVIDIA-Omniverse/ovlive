// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Stack dump cases that need the dumper's own state or a gate in its unwinding, for stack_dump_check.py. It compiles
// StackDump.cpp into itself and is linked with --wrap=_Unwind_Backtrace,--wrap=nanosleep. Each case runs with
// LIVE_STACK_DUMP_PREFIX set,
// prints done once its dumps have ended, and leaves the dumps for the check:
//   full-guards  Every guard is held when SIGQUIT arrives, so the raising thread writes no block; the dump's missing
//                count includes it.
//   late-answer  A thread blocks the thread signal through one dump and unblocks it during the next, so it answers
//                both dumps at once; only its answer to the open dump counts, and a thread that never answers stays
//                missing.
//   nested-quit  SIGQUIT reaches a thread while it is still answering a dump that has ended, and the unwinding of the
//                dump it then starts faults; the fault is contained to that dump, which reaches its end line, and a
//                further dump completes.
#include "StackDump.cpp"

#include <cstring>
#include <thread>

namespace {
std::atomic<long> gated{0};       // The thread whose unwinding the wrapper gates and then faults.
std::atomic<int> gatedCalls{0};
std::atomic<bool> gateOpen{false};
std::atomic<long> waiter{0};       // The thread whose waits in a dump the nanosleep wrapper records.
std::atomic<int> raised{0};        // The dumps that thread has raised.
std::atomic<int> waited{0};        // The dump, of those, in which it last waited.
volatile uintptr_t faultAddress = 16;

__attribute__((noinline, no_sanitize("address", "thread"))) void fault() {
    *reinterpret_cast<volatile int*>(faultAddress) = 0;
}

// Failure-only bound for every wait on an observable predicate.
template <typename Predicate> bool waitFor(Predicate predicate, int seconds = 20) {
    for (int i = 0; i < seconds * 100; ++i, pause10ms())
        if (predicate()) return true;
    return false;
}

int count(const char* needle) {
    char text[1 << 16];
    const int fd = open(dumpPath, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return 0;
    const auto n = read(fd, text, sizeof text - 1);
    close(fd);
    if (n <= 0) return 0;
    text[n] = 0;
    int found = 0;
    for (const char* at = text; (at = std::strstr(at, needle)); ++at) ++found;
    return found;
}

// A field of /proc/self/status or /proc/self/task/<tid>/status, as a number in `base`.
unsigned long long statusField(const char* path, const char* field, int base) {
    char text[4096];
    const int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return 0;
    const auto n = read(fd, text, sizeof text - 1);
    close(fd);
    if (n <= 0) return 0;
    text[n] = 0;
    const char* at = std::strstr(text, field);
    return at ? std::strtoull(at + std::strlen(field), nullptr, base) : 0;
}

void blockThreadSignal(int how) {
    sigset_t set;
    sigemptyset(&set);
    sigaddset(&set, threadSignal());
    pthread_sigmask(how, &set, nullptr);
}

int fail(const char* message) {
    std::fprintf(stderr, "stack dump case: %s\n", message);
    return 1;
}

int fullGuards() {
    for (auto& guard : guards) guard.tid.store(1);
    liveTestDumpStacks();
    for (auto& guard : guards) guard.tid.store(0);
    return 0;
}

int lateAnswer() {
    std::atomic<int> started{0};

    std::thread never([&] {
        pthread_setname_np(pthread_self(), "never-answers");
        blockThreadSignal(SIG_BLOCK);
        ++started;
        for (;;) pause10ms();
    });
    std::thread late([&] {
        pthread_setname_np(pthread_self(), "late-answer");
        blockThreadSignal(SIG_BLOCK);
        ++started;
        // Unblocks once the second dump has signalled every thread: its owner's first wait follows the signals.
        waitFor([] { return waited.load() == 2 || count("live-stack-dump end") == 2; }, 30);
        blockThreadSignal(SIG_UNBLOCK);
        for (;;) pause10ms();
    });
    never.detach();
    late.detach();
    if (!waitFor([&] { return started.load() == 2; })) return fail("threads did not start");
    waiter.store(syscall(SYS_gettid));
    for (int dump = 1; dump <= 2; ++dump) { raised.store(dump); liveTestDumpStacks(); }
    return 0;
}

int nestedQuit() {
    std::atomic<long> tid{0};
    std::thread target([&] {
        pthread_setname_np(pthread_self(), "gated");
        tid.store(syscall(SYS_gettid));
        for (;;) pause10ms();
    });
    target.detach();
    if (!waitFor([&] { return tid.load() != 0; })) return fail("thread did not start");
    gated.store(tid.load());
    liveTestDumpStacks(); // The gated thread is still answering it when it ends.
    if (gatedCalls.load() != 1) return fail("the gated thread did not start answering the first dump");
    syscall(SYS_tgkill, getpid(), tid.load(), SIGQUIT);
    // SIGQUIT is pending on the thread while it answers, or (were the handlers to nest) has started a second dump.
    char status[64];
    std::snprintf(status, sizeof status, "/proc/self/task/%ld/status", tid.load());
    const unsigned long long quit = 1ULL << (SIGQUIT - 1);
    if (!waitFor([&] { return (statusField(status, "SigPnd:", 16) & quit) || count("live-stack-dump begin") == 2; }))
        return fail("SIGQUIT did not reach the gated thread");
    gateOpen.store(true);
    if (!waitFor([&] { return count("live-stack-dump end") == 2; })) return fail("the second dump did not end");
    liveTestDumpStacks();
    return 0;
}
}

extern "C" int __real_nanosleep(const timespec* interval, timespec* remaining);
// Records the dump in which the watched thread waits; a dump's owner first waits once it has signalled every thread.
extern "C" int __wrap_nanosleep(const timespec* interval, timespec* remaining) {
    if (waiter.load() != 0 && syscall(SYS_gettid) == waiter.load() && dumping.load()) waited.store(raised.load());
    return __real_nanosleep(interval, remaining);
}
extern "C" _Unwind_Reason_Code __real__Unwind_Backtrace(_Unwind_Trace_Fn trace, void* argument);
// The gated thread's first unwinding waits for the gate; its second faults.
extern "C" _Unwind_Reason_Code __wrap__Unwind_Backtrace(_Unwind_Trace_Fn trace, void* argument) {
    if (gated.load() != 0 && syscall(SYS_gettid) == gated.load()) {
        const int call = ++gatedCalls;
        if (call == 1) waitFor([] { return gateOpen.load(); });
        else if (call == 2) fault();
    }
    return __real__Unwind_Backtrace(trace, argument);
}

int main(int argc, char** argv) {
    if (argc != 2) return fail("usage: live_stack_dump_cases <full-guards|late-answer|nested-quit>");
    if (!installed.load()) return fail("LIVE_STACK_DUMP_PREFIX is not set");
    const int result = !std::strcmp(argv[1], "full-guards") ? fullGuards()
                       : !std::strcmp(argv[1], "late-answer") ? lateAnswer()
                       : !std::strcmp(argv[1], "nested-quit") ? nestedQuit()
                                                              : fail("unknown case");
    if (result == 0) { std::printf("done\n"); std::fflush(stdout); }
    std::_Exit(result); // The parked threads never end.
}
