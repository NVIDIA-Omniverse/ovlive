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
#include <csetjmp>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <dirent.h>
#include <execinfo.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <unwind.h>
namespace {
constexpr int kFrames = 128;
// A thread unwinds its own stack under a guard: a slot with its recovery point, its frames and an alternate signal
// stack. A fault in the unwinder, such as one following a wrong CFA through a frame whose unwind information is not
// accurate at the interrupted instruction, or one overflowing the thread's stack, returns to the recovery point, so
// the thread writes the frames it reached and the fault instead of ending the process partway through the dump.
constexpr int kGuards = 64;
constexpr size_t kGuardStack = 65536;
struct Guard {
    std::atomic<long> tid{0}; // The thread holding the guard; 0 when it is free.
    volatile sig_atomic_t unwinding;
    volatile sig_atomic_t fault; // The signal that ended the unwinding; 0 when it reached the end of the stack.
    sigjmp_buf recover;
    int count;
    void* frames[kFrames];
    alignas(16) char stack[kGuardStack];
};
Guard guards[kGuards];
// SIGTRAP is not contained: on x86-64 a handler that returns resumes after the trap instead of taking it again.
constexpr int kFaults[] = {SIGSEGV, SIGBUS, SIGILL, SIGFPE};
constexpr int kFaultCount = sizeof kFaults / sizeof *kFaults;
struct sigaction previousFault[kFaultCount]; // The dispositions a fault outside any unwinding goes back to.
char dumpPath[4096];            // <prefix>.<pid>, formatted before any handler runs.
std::atomic<bool> installed{false};
// The current dump in one word: its generation (high 32 bits), whether it is open (bit 31), and how many of its
// threads, the one that took SIGQUIT among them, have not written their block (low 31 bits). Each dump takes the next
// generation and sends it with its thread signal; a thread writes its block, and takes one from the count, only while
// the dump it answers is open, so an answer to an earlier dump, or one that comes after its dump's end line, counts
// toward no dump and writes nothing.
std::atomic<uint64_t> dumpState{0};
constexpr uint64_t kOpen = uint64_t(1) << 31;
constexpr uint64_t kCount = kOpen - 1;
std::atomic<int> writing{0};    // Serializes thread blocks and the end line; the spin is bounded.
// Serializes whole dumps. A SIGQUIT handler is async-signal-safe only with
// lock-free atomics and nanosleep, so a second dump spins on this flag
// (at most 10 s) rather than taking a lock.
std::atomic<bool> dumping{false};
static_assert(std::atomic<int>::is_always_lock_free && std::atomic<bool>::is_always_lock_free &&
              std::atomic<uint64_t>::is_always_lock_free && std::atomic<long>::is_always_lock_free,
              "signal handlers need lock-free atomics");
int threadSignal() { return SIGRTMIN + 3; }
void put(int fd, const char* text) { size_t n = 0; while (text[n]) ++n; while (n) { const auto w = write(fd, text, n); if (w <= 0) return; text += w; n -= size_t(w); } }
void putNumber(int fd, long value) { char digits[24]; int i = 23; digits[i] = 0; bool negative = value < 0; unsigned long v = negative ? 0UL - static_cast<unsigned long>(value) : static_cast<unsigned long>(value); do { digits[--i] = char('0' + v % 10); v /= 10; } while (v); if (negative) digits[--i] = '-'; put(fd, digits + i); }
void pause10ms() { timespec interval{0, 10000000}; nanosleep(&interval, nullptr); }
void lockWriting() { for (int i = 0; i < 200 && writing.exchange(1); ++i) pause10ms(); } // At most 2 s, then go on.
bool isOpen(uint64_t state, uint32_t generation) { return (state & kOpen) && uint32_t(state >> 32) == generation; }
// Takes this thread's answer from the count of dump `generation`; false when that dump is not open.
bool answer(uint32_t generation) {
    uint64_t state = dumpState.load();
    while (isOpen(state, generation) && (state & kCount))
        if (dumpState.compare_exchange_weak(state, state - 1)) return true;
    return false;
}
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
_Unwind_Reason_Code collect(_Unwind_Context* context, void* argument) {
    auto& guard = *static_cast<Guard*>(argument);
    const auto ip = _Unwind_GetIP(context);
    if (guard.count == kFrames || ip == 0) return _URC_END_OF_STACK; // The outermost frame reports address 0.
    guard.frames[guard.count++] = reinterpret_cast<void*>(ip);
    return _URC_NO_REASON;
}
void onFault(int signal, siginfo_t* info, void*) {
    const long tid = syscall(SYS_gettid);
    if (info->si_code > 0) // Raised by the faulting instruction, not sent with kill or tgkill.
        for (auto& guard : guards)
            if (guard.tid.load() == tid && guard.unwinding) { guard.fault = signal; siglongjmp(guard.recover, 1); }
    // Any other fault goes to the earlier disposition: a faulting instruction takes it when it runs again on return,
    // and a sent signal is raised again, pending until this handler returns.
    for (int i = 0; i < kFaultCount; ++i)
        if (kFaults[i] == signal) sigaction(signal, &previousFault[i], nullptr);
    if (info->si_code <= 0) raise(signal);
}
// Installed at the start of every dump, so a handler installed after an earlier dump does not disable containment.
void containFaults() {
    struct sigaction action {};
    sigemptyset(&action.sa_mask);
    action.sa_flags = SA_SIGINFO | SA_ONSTACK;
    action.sa_sigaction = onFault;
    for (int i = 0; i < kFaultCount; ++i) {
        struct sigaction current {};
        if (sigaction(kFaults[i], nullptr, &current) != 0) continue;
        if ((current.sa_flags & SA_SIGINFO) && current.sa_sigaction == onFault) continue;
        previousFault[i] = current;
        sigaction(kFaults[i], &action, nullptr);
    }
}
// A thread holds at most one guard, so a fault always returns to the recovery point of the unwinding it interrupted.
// The handlers cannot nest on one thread (the thread signal blocks SIGQUIT while it runs), and a thread that already
// holds a guard takes no second one.
Guard* acquireGuard(long tid) {
    for (auto& guard : guards)
        if (guard.tid.load() == tid) return nullptr;
    for (int i = 0; i < 200; ++i, pause10ms()) // At most 2 s while other threads hold every guard.
        for (auto& guard : guards) {
            long free = 0;
            if (guard.tid.compare_exchange_strong(free, tid)) return &guard;
        }
    return nullptr;
}
// Kept apart from any other local state, which a return through siglongjmp would leave indeterminate.
__attribute__((noinline)) void unwind(Guard* guard) {
    if (sigsetjmp(guard->recover, 1) == 0) {
        guard->unwinding = 1;
        _Unwind_Backtrace(collect, guard); // Primed at install; takes no allocation afterwards.
    }
    guard->unwinding = 0;
}
// Writes this thread's block into dump `generation` while that dump is open. A thread that gets no guard within 2 s,
// or whose dump ended first, writes nothing and stays in the dump's missing count.
void dumpThisThread(uint32_t generation) {
    const long tid = syscall(SYS_gettid);
    Guard* const guard = acquireGuard(tid);
    if (!guard) return;
    // The guard's stack serves a fault on a thread without an alternate stack of its own.
    stack_t own{}, previous{};
    own.ss_sp = guard->stack;
    own.ss_size = kGuardStack;
    const bool swapped =
        sigaltstack(nullptr, &previous) == 0 && (previous.ss_flags & SS_DISABLE) && sigaltstack(&own, nullptr) == 0;
    // A fault on a thread that blocks the signal ends the process whatever its handler, and ThreadSanitizer runs a
    // deferred handler with every signal blocked, so the unwinding unblocks the faults it contains.
    sigset_t faults, mask;
    sigemptyset(&faults);
    for (int signal : kFaults) sigaddset(&faults, signal);
    const bool unblocked = pthread_sigmask(SIG_UNBLOCK, &faults, &mask) == 0;
    guard->count = 0;
    guard->fault = 0;
    unwind(guard);
    if (unblocked) pthread_sigmask(SIG_SETMASK, &mask, nullptr);
    if (swapped) sigaltstack(&previous, nullptr);
    const int fd = open(dumpPath, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
    if (fd >= 0) {
        lockWriting();
        if (answer(generation)) {
            put(fd, "live-stack-thread tid="); putNumber(fd, tid); put(fd, " name="); putThreadName(fd, tid); put(fd, "\n");
            backtrace_symbols_fd(guard->frames, guard->count, fd);
            if (guard->fault) { put(fd, "live-stack-unwind-fault signal="); putNumber(fd, guard->fault); put(fd, "\n"); }
        }
        writing.store(0);
        close(fd);
    }
    guard->tid.store(0);
}
// Answers the dump whose generation the signal carries, if it is still open; any other signal is ignored.
void onThread(int, siginfo_t* info, void*) {
    const int saved = errno;
    if (info && info->si_code == SI_QUEUE && info->si_pid == getpid()) {
        const auto generation = static_cast<uint32_t>(info->si_value.sival_int);
        if (isOpen(dumpState.load(), generation)) dumpThisThread(generation);
    }
    errno = saved;
}
long signalThread(pid_t pid, long tid, uint32_t generation) {
    siginfo_t info{};
    info.si_signo = threadSignal();
    info.si_code = SI_QUEUE;
    info.si_pid = pid;
    info.si_uid = getuid();
    info.si_value.sival_int = static_cast<int>(generation);
    return syscall(SYS_rt_tgsigqueueinfo, pid, tid, threadSignal(), &info);
}
void onQuit(int, siginfo_t*, void*) {
    const int saved = errno;
    // SIGQUIT is blocked on the thread running this handler, not on others: a
    // second process-directed SIGQUIT, or the in-process watchdog's raise(),
    // can start a second dump on another thread. It waits for the first, and
    // answers the first's thread signal meanwhile, since that signal is not
    // blocked here.
    bool owner = !dumping.exchange(true);
    for (int i = 0; !owner && i < 1000; ++i) { pause10ms(); owner = !dumping.exchange(true); }
    if (!owner) { errno = saved; return; }
    const pid_t pid = getpid();
    const long self = syscall(SYS_gettid);
    int fd = open(dumpPath, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
    if (fd < 0) { dumping.store(false); errno = saved; return; }
    const uint32_t generation = uint32_t(dumpState.load() >> 32) + 1;
    dumpState.store((uint64_t(generation) << 32) | kOpen | 1); // This thread's block is the first one the dump counts.
    put(fd, "live-stack-dump begin pid="); putNumber(fd, pid); put(fd, "\n"); close(fd);
    containFaults();
    // This thread's block comes first, before any other thread could hold a lock the unwinder takes.
    dumpThisThread(generation);
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
            dumpState.fetch_add(1);
            if (signalThread(pid, tid, generation) != 0) dumpState.fetch_sub(1);
        }
    }
    if (directory >= 0) close(directory);
    // A thread that does not write its block within 2 s, such as a sanitizer
    // runtime thread or one blocked in the dynamic loader, is reported as missing.
    for (int i = 0; i < 200 && (dumpState.load() & kCount); ++i) pause10ms();
    fd = open(dumpPath, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
    lockWriting(); // Closing under the lock keeps every counted block before the end line.
    const uint64_t closed = dumpState.exchange(uint64_t(generation) << 32);
    if (fd >= 0) { put(fd, "live-stack-dump end pid="); putNumber(fd, pid); put(fd, " missing="); putNumber(fd, long(closed & kCount)); put(fd, "\n"); close(fd); }
    writing.store(0);
    dumping.store(false);
    errno = saved;
}
struct Installer {
    Installer() {
        const char* prefix = std::getenv("LIVE_STACK_DUMP_PREFIX");
        if (!prefix || !*prefix) return;
        if (std::snprintf(dumpPath, sizeof dumpPath, "%s.%ld", prefix, long(getpid())) >= int(sizeof dumpPath)) return;
        _Unwind_Backtrace(collect, &guards[0]); // Resolves and initializes the unwinder outside any handler.
        guards[0].count = 0;
        struct sigaction current {};
        if (sigaction(SIGQUIT, nullptr, &current) != 0 || current.sa_handler != SIG_DFL) return;
        if (sigaction(threadSignal(), nullptr, &current) != 0 || current.sa_handler != SIG_DFL) return;
        struct sigaction action {};
        sigemptyset(&action.sa_mask);
        sigaddset(&action.sa_mask, SIGQUIT); // A dump does not start on a thread that is answering one.
        action.sa_flags = SA_SIGINFO | SA_RESTART;
        action.sa_sigaction = onThread;
        if (sigaction(threadSignal(), &action, nullptr) != 0) return;
        sigemptyset(&action.sa_mask);
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
#        include <cstdlib>
#        include <exception>
#        include <fcntl.h>
#        include <io.h>
#        ifndef NOMINMAX
#            define NOMINMAX
#        endif
#        ifndef WIN32_LEAN_AND_MEAN
#            define WIN32_LEAN_AND_MEAN
#        endif
#        include <windows.h>
#        include <dbghelp.h>
#        if defined(_DEBUG)
#            include <crtdbg.h>
#        endif
namespace {
// Failure-only, bounded current-thread stack; PDBs remain beside the built drivers. Do not consult symbol servers.
void failureStack() noexcept {
    void* frames[32]{};
    const auto count = CaptureStackBackTrace(0, 32, frames, nullptr);
    const auto process = GetCurrentProcess();
    SymSetOptions(SYMOPT_LOAD_LINES | SYMOPT_FAIL_CRITICAL_ERRORS | SYMOPT_NO_PROMPTS | SYMOPT_IGNORE_NT_SYMPATH);
    const bool symbols = SymInitialize(process, ".", TRUE) != FALSE;
    std::fputs("native test failure stack:\n", stderr);
    for (USHORT index = 0; index < count; ++index) {
        alignas(SYMBOL_INFO) char buffer[sizeof(SYMBOL_INFO) + MAX_SYM_NAME]{};
        auto* symbol = reinterpret_cast<SYMBOL_INFO*>(buffer);
        symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
        symbol->MaxNameLen = MAX_SYM_NAME;
        DWORD64 displacement = 0;
        const auto address = reinterpret_cast<DWORD64>(frames[index]);
        std::fprintf(stderr, "  %p", frames[index]);
        if (symbols && SymFromAddr(process, address, &displacement, symbol))
            std::fprintf(stderr, " %s+0x%llx", symbol->Name, static_cast<unsigned long long>(displacement));
        IMAGEHLP_LINE64 line{};
        line.SizeOfStruct = sizeof(line);
        DWORD offset = 0;
        if (symbols && SymGetLineFromAddr64(process, address, &offset, &line))
            std::fprintf(stderr, " %s:%lu", line.FileName, line.LineNumber);
        std::fputc('\n', stderr);
    }
}
// The harness matches driver output byte for byte, so a Windows driver writes LF line ends as on Linux, not the
// CRLF of the CRT's text mode. Its stdout is unbuffered, so every record reaches the harness when written: the CRT
// flushes buffered stdout only in its own process detach, after ExitProcess has ended every other thread, and a driver
// whose gRPC threads are still exiting at that moment can end before the flush with exit code 0
// (docs/ovlive-testing.md#known-test-environment-issues; tests/harness/test_exit_flush.py).
const bool binaryStreams = _setmode(_fileno(stdout), _O_BINARY) != -1 && _setmode(_fileno(stderr), _O_BINARY) != -1 &&
    std::setvbuf(stdout, nullptr, _IONBF, 0) == 0;
const bool failureReports = [] {
#        if defined(_DEBUG)
    constexpr int types[] = {_CRT_WARN, _CRT_ERROR, _CRT_ASSERT};
    for (const int type : types) {
        _CrtSetReportMode(type, _CRTDBG_MODE_FILE);
        _CrtSetReportFile(type, _CRTDBG_FILE_STDERR);
    }
#        endif
    _set_error_mode(_OUT_TO_STDERR);
    _set_abort_behavior(0, _CALL_REPORTFAULT);
    std::set_terminate([] {
        std::fputs("native test terminated", stderr);
        if (const auto exception = std::current_exception()) {
            try {
                std::rethrow_exception(exception);
            } catch (const std::exception& error) { std::fprintf(stderr, ": %s", error.what()); } catch (...) {
                std::fputs(": non-standard exception", stderr);
            }
        }
        std::fputc('\n', stderr);
        failureStack();
        std::abort();
    });
    return true;
}();
}
#    endif
void liveTestDumpStacks() noexcept {}
#endif
