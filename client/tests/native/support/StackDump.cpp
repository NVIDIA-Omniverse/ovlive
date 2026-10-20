// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Opt-in stack capture for native test drivers. With LIVE_STACK_DUMP_PREFIX
// set, SIGQUIT makes every thread append its backtrace to <prefix>.<pid>; the
// test runner sends it when a watchdog expires (docs/ovlive-testing.md). Only
// test executables link this file, never the plugin library or the samples.
#include <csignal>
#if defined(__linux__)
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <dirent.h>
#include <execinfo.h>
#include <fcntl.h>
#include <sys/syscall.h>
#include <unistd.h>
namespace {
constexpr int kFrames = 128;
char dumpPath[4096];            // <prefix>.<pid>, formatted before any handler runs.
std::atomic<bool> installed{false};
std::atomic<int> pending{0};    // Threads the current dump signalled that have not answered.
std::atomic<int> writing{0};    // Serializes thread blocks; the spin is bounded.
// Serializes whole dumps. A SIGQUIT handler is async-signal-safe only with
// lock-free atomics and nanosleep, so a second dump spins on this flag
// (at most 10 s) rather than taking a lock.
std::atomic<bool> dumping{false};
static_assert(std::atomic<int>::is_always_lock_free && std::atomic<bool>::is_always_lock_free,
              "signal handlers need lock-free atomics");
int threadSignal() { return SIGRTMIN + 3; }
void put(int fd, const char* text) { size_t n = 0; while (text[n]) ++n; while (n) { const auto w = write(fd, text, n); if (w <= 0) return; text += w; n -= size_t(w); } }
void putNumber(int fd, long value) { char digits[24]; int i = 23; digits[i] = 0; bool negative = value < 0; unsigned long v = negative ? 0UL - static_cast<unsigned long>(value) : static_cast<unsigned long>(value); do { digits[--i] = char('0' + v % 10); v /= 10; } while (v); if (negative) digits[--i] = '-'; put(fd, digits + i); }
void pause10ms() { timespec interval{0, 10000000}; nanosleep(&interval, nullptr); }
void putThreadName(int fd, long tid) { // /proc/self/task/<tid>/comm, so gRPC's thread names appear.
    char path[64] = "/proc/self/task/";
    size_t at = 16;
    char digits[24]; int i = 23; digits[i] = 0; unsigned long v = static_cast<unsigned long>(tid);
    do { digits[--i] = char('0' + v % 10); v /= 10; } while (v);
    for (const char* d = digits + i; *d;) path[at++] = *d++;
    for (const char* s = "/comm"; *s;) path[at++] = *s++;
    path[at] = 0;
    const int comm = open(path, O_RDONLY | O_CLOEXEC);
    if (comm < 0) return;
    char name[17];
    const auto n = read(comm, name, 16);
    close(comm);
    if (n <= 0) return;
    size_t length = size_t(n);
    if (name[length - 1] == '\n') --length;
    name[length] = 0;
    put(fd, name);
}
void dumpThisThread() {
    const int fd = open(dumpPath, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
    if (fd < 0) return;
    void* frames[kFrames];
    const int count = backtrace(frames, kFrames); // Primed at install; takes no allocation afterwards.
    for (int i = 0; i < 200 && writing.exchange(1); ++i) pause10ms(); // At most 2 s, then write anyway.
    const long tid = syscall(SYS_gettid);
    put(fd, "live-stack-thread tid="); putNumber(fd, tid); put(fd, " name="); putThreadName(fd, tid); put(fd, "\n");
    backtrace_symbols_fd(frames, count, fd);
    writing.store(0);
    close(fd);
}
// A late answer to an earlier dump can lower the count, never below zero.
void answered() { int value = pending.load(); while (value > 0 && !pending.compare_exchange_weak(value, value - 1)) {} }
void onThread(int, siginfo_t*, void*) { const int saved = errno; dumpThisThread(); answered(); errno = saved; }
void onQuit(int, siginfo_t*, void*) {
    const int saved = errno;
    // SIGQUIT is blocked on the thread running this handler, not on others: a
    // second process-directed SIGQUIT, or the in-process watchdog's raise(),
    // can start a second dump on another thread. It waits for the first, and
    // answers the first's thread signal meanwhile, since that signal is not
    // blocked here. Each dump counts only the threads it signals.
    bool owner = !dumping.exchange(true);
    for (int i = 0; !owner && i < 1000; ++i) { pause10ms(); owner = !dumping.exchange(true); }
    if (!owner) { errno = saved; return; }
    pending.store(0);
    const pid_t pid = getpid();
    const long self = syscall(SYS_gettid);
    int fd = open(dumpPath, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
    if (fd < 0) { dumping.store(false); errno = saved; return; }
    put(fd, "live-stack-dump begin pid="); putNumber(fd, pid); put(fd, "\n"); close(fd);
    // Signal every other thread; /proc/self/task is read with getdents64 (async-signal-safe).
    struct Entry { uint64_t inode; int64_t offset; unsigned short length; unsigned char type; char name[1]; }; // linux_dirent64
    alignas(8) char buffer[4096];
    const int directory = open("/proc/self/task", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    for (long size; directory >= 0 && (size = syscall(SYS_getdents64, directory, buffer, sizeof buffer)) > 0;) {
        for (long at = 0; at < size;) {
            const auto* entry = reinterpret_cast<const Entry*>(buffer + at);
            at += entry->length;
            const char* c = entry->name;
            if (*c < '0' || *c > '9') continue; // "." and ".."
            long tid = 0;
            for (; *c >= '0' && *c <= '9'; ++c) tid = tid * 10 + (*c - '0');
            if (tid == self) continue;
            pending.fetch_add(1);
            if (syscall(SYS_tgkill, pid, tid, threadSignal()) != 0) pending.fetch_sub(1);
        }
    }
    if (directory >= 0) close(directory);
    dumpThisThread();
    // A thread that never answers within 2 s, such as a sanitizer runtime thread
    // or one blocked in the dynamic loader, is reported as missing.
    for (int i = 0; i < 200 && pending.load() > 0; ++i) pause10ms();
    fd = open(dumpPath, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
    if (fd >= 0) { put(fd, "live-stack-dump end pid="); putNumber(fd, pid); put(fd, " missing="); putNumber(fd, pending.load()); put(fd, "\n"); close(fd); }
    dumping.store(false);
    errno = saved;
}
struct Installer {
    Installer() {
        const char* prefix = std::getenv("LIVE_STACK_DUMP_PREFIX");
        if (!prefix || !*prefix) return;
        if (std::snprintf(dumpPath, sizeof dumpPath, "%s.%ld", prefix, long(getpid())) >= int(sizeof dumpPath)) return;
        void* frames[1];
        (void)backtrace(frames, 1); // Loads libgcc's unwinder outside any handler.
        struct sigaction current {};
        if (sigaction(SIGQUIT, nullptr, &current) != 0 || current.sa_handler != SIG_DFL) return;
        if (sigaction(threadSignal(), nullptr, &current) != 0 || current.sa_handler != SIG_DFL) return;
        struct sigaction action {};
        sigemptyset(&action.sa_mask);
        action.sa_flags = SA_SIGINFO | SA_RESTART;
        action.sa_sigaction = onThread;
        if (sigaction(threadSignal(), &action, nullptr) != 0) return;
        action.sa_sigaction = onQuit;
        if (sigaction(SIGQUIT, &action, nullptr) == 0) installed.store(true);
    }
} installer;
}
// Dumps every thread's stack when the opt-in handler is installed; otherwise does nothing.
void liveTestDumpStacks() noexcept { if (installed.load()) std::raise(SIGQUIT); }
#else
#    if defined(_WIN32)
#        include <cstdio>
#        include <fcntl.h>
#        include <io.h>
namespace {
// The harness matches driver output byte for byte, so a Windows driver writes LF line ends as on Linux, not the
// CRLF of the CRT's text mode. Its stdout is unbuffered, so every record reaches the harness when written: the CRT
// flushes buffered stdout only in its own process detach, after ExitProcess has ended every other thread, and a driver
// whose gRPC threads are still exiting at that moment can end before the flush with exit code 0
// (docs/ovlive-testing.md#known-test-environment-issues; tests/harness/test_exit_flush.py).
const bool binaryStreams = _setmode(_fileno(stdout), _O_BINARY) != -1 && _setmode(_fileno(stderr), _O_BINARY) != -1 &&
                           std::setvbuf(stdout, nullptr, _IONBF, 0) == 0;
}
#    endif
void liveTestDumpStacks() noexcept {}
#endif
