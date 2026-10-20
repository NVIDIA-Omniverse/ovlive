// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Standalone C-ABI consumer. No private runtime, USD or client-library APIs.
#include "DiagnosticLog.h"
#include "RecordWriter.h"
#include <atomic>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
#if defined(_WIN32)
#    include <windows.h>
#endif

namespace {
using Clock = std::chrono::steady_clock;
// Ends the process at once with `code`. On Windows std::_Exit still runs every DLL's detach, including the plugin's
// static destructors, which may wait on runtime threads a blocked callback holds; TerminateProcess runs none.
[[noreturn]] void exitNow(int code) {
#if defined(_WIN32)
    TerminateProcess(GetCurrentProcess(), UINT(code));
#endif
    std::_Exit(code);
}
using namespace omni::usd::live::sample;
template<class T> T record() {
    T value{};
    value.struct_size = sizeof(value);
    value.abi_version = OMNI_USD_LIVE_ABI_VERSION;
    return value;
}
uint32_t remaining(Clock::time_point deadline) {
    const auto now = Clock::now();
    return now >= deadline ? 0 :
                             uint32_t(std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count());
}
// stdout/stderr may share a pipe whose reader has stopped. A process watchdog
// covers main-thread output, including snapshot/error reporting and final stdio
// flush, without changing the caller's inherited descriptor flags. This is
// sample process policy, never a runtime/library shutdown mechanism.
class ExitDeadline {
    std::mutex mutex_;
    std::condition_variable changed_;
    Clock::time_point deadline_ = Clock::now() + std::chrono::seconds(5);
    bool stopped_ = false;
    std::thread thread_;
public:
    ExitDeadline()
        : thread_([this] {
              std::unique_lock<std::mutex> lock(mutex_);
              while (!stopped_) {
                  if (Clock::now() >= deadline_) exitNow(3);
                  changed_.wait_until(lock, deadline_);
              }
          }) {}
    void arm(uint32_t timeout) {
        std::lock_guard<std::mutex> lock(mutex_);
        deadline_ = Clock::now() + std::chrono::milliseconds(uint64_t(timeout) * 3 + 1000);
        changed_.notify_all();
    }
    ~ExitDeadline() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopped_ = true;
        }
        changed_.notify_all();
        thread_.join();
    }
};
struct Options {
    std::string authority, base, tokenEnv = "LIVE_SERVICES_TOKEN", caBundle, tlsServerName;
    uint32_t transportSecurity = OMNI_USD_LIVE_TRANSPORT_DEFAULT;
    LogFormat format = LogFormat::JsonLines;
    uint32_t timeout = 5000;
    bool debug = false, snapshot = false, help = false;
};
Options parse(int argc, char** argv) {
    Options result;
    std::vector<std::string_view> seen;
    for (int i = 1; i < argc; ++i) {
        const std::string_view name(argv[i]);
        for (const auto previous : seen)
            if (previous == name) throw std::invalid_argument("duplicate option");
        seen.push_back(name);
        if (name == "--help") {
            result.help = true;
            continue;
        }
        if (name == "--debug") {
            result.debug = true;
            continue;
        }
        if (name == "--snapshot") {
            result.snapshot = true;
            continue;
        }
        if (name != "--authority" && name != "--base-layer-key" && name != "--token-env" && name != "--format" &&
            name != "--timeout-ms" && name != "--ca-bundle" && name != "--tls-server-name" &&
            name != "--transport-security")
            throw std::invalid_argument("unknown option");
        if (++i == argc || !*argv[i]) throw std::invalid_argument("missing option value");
        const std::string_view value(argv[i]);
        if (name == "--authority") result.authority = value;
        else if (name == "--base-layer-key") result.base = value;
        else if (name == "--ca-bundle") result.caBundle = value;
        else if (name == "--tls-server-name") result.tlsServerName = value;
        else if (name == "--transport-security") {
            if (value == "default") result.transportSecurity = OMNI_USD_LIVE_TRANSPORT_DEFAULT;
            else if (value == "tls") result.transportSecurity = OMNI_USD_LIVE_TRANSPORT_TLS;
            else if (value == "plaintext") result.transportSecurity = OMNI_USD_LIVE_TRANSPORT_PLAINTEXT;
            else throw std::invalid_argument("invalid transport security");
        } else if (name == "--token-env") {
            // Environment names only; command-line tokens are never accepted.
            for (const char c : value)
                if (!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_'))
                    throw std::invalid_argument("invalid token environment name");
            result.tokenEnv = value;
        } else if (name == "--format") {
            if (value == "jsonl") result.format = LogFormat::JsonLines;
            else if (value == "text") result.format = LogFormat::Text;
            else throw std::invalid_argument("invalid log format");
        } else {
            uint32_t number = 0;
            const auto parsed = std::from_chars(value.data(), value.data() + value.size(), number);
            if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || !number || number > 60000)
                throw std::invalid_argument("timeout must be between 1 and 60000 milliseconds");
            result.timeout = number;
        }
    }
    if (result.help) {
        if (argc != 2) throw std::invalid_argument("help must be used alone");
    } else if (result.authority.empty() || result.base.empty())
        throw std::invalid_argument("authority and base layer key are required");
    return result;
}

struct State {
    OmniUsdLiveEndpoint endpoint = 0;
    OmniUsdLiveRequest request = 0;
    OmniUsdLiveCallbackHandle diagnostics = 0, statuses = 0;
    bool initialized = false;
    LogFormat format = LogFormat::JsonLines;
    std::atomic<uint32_t> listResult{OMNI_USD_LIVE_INTERNAL_ERROR};
    std::atomic<uint64_t> sessionCount{0}, statusCount{0}, terminalCount{0};
    std::atomic<OmniUsdLiveResult> sinkResult{OMNI_USD_LIVE_OK};
    std::mutex mutex;
    std::condition_variable changed;
    bool delivered = false;

    static void OMNI_USD_LIVE_CALL diagnostic(void* context, const OmniUsdLiveDiagnosticRecord* event) {
        auto& state = *static_cast<State*>(context);
        std::string line;
        auto result = event ? formatDiagnosticRecord(*event, state.format, line) :
                              OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
        if (result == OMNI_USD_LIVE_OK && !writeRecord(stdout, line)) result = OMNI_USD_LIVE_IO_ERROR;
        {
            std::lock_guard<std::mutex> lock(state.mutex);
            if (result != OMNI_USD_LIVE_OK) state.sinkResult.store(result);
            state.delivered = true;
        }
        state.changed.notify_all();
        // Registry failure accounting/disablement is separate from operational
        // status. Never recursively log a failing diagnostic sink.
        if (result != OMNI_USD_LIVE_OK) throw std::runtime_error("diagnostic output failed");
    }
    static void OMNI_USD_LIVE_CALL status(void* context, const OmniUsdLiveStatusEvent* event) {
        auto& state = *static_cast<State*>(context);
        {
            std::lock_guard<std::mutex> lock(state.mutex);
            state.statusCount.fetch_add(1);
            if (event && event->status && event->status->state == OMNI_USD_LIVE_STATE_FAILED)
                state.terminalCount.fetch_add(1);
        }
        state.changed.notify_all();
    }
    static void OMNI_USD_LIVE_CALL listed(void* context, OmniUsdLiveResult result, uint64_t count,
        const OmniUsdLiveSessionInfo* const*) {
        auto& state = *static_cast<State*>(context);
        // No borrowed payload escapes the callback. Names/base keys are not
        // added to diagnostics; a real UI may separately copy approved metadata.
        state.sessionCount.store(result == OMNI_USD_LIVE_OK ? count : 0);
        state.listResult.store(result);
    }
    OmniUsdLiveResult retire(Clock::time_point deadline) {
        auto result = OmniUsdLiveResult(OMNI_USD_LIVE_OK);
        if (request) {
            result = omniUsdLiveCancel(request, remaining(deadline));
            if (result != OMNI_USD_LIVE_OK) return result;
            result = omniUsdLiveReleaseRequest(request);
            if (result != OMNI_USD_LIVE_OK) return result;
            request = 0;
        }
        for (auto* handle : {&diagnostics, &statuses})
            if (*handle) {
                result = omniUsdLiveUnregisterCallback(*handle, remaining(deadline));
                if (result != OMNI_USD_LIVE_OK) return result;
                *handle = 0;
            }
        if (endpoint) {
            result = omniUsdLiveUnregisterEndpoint(endpoint, remaining(deadline));
            if (result != OMNI_USD_LIVE_OK) return result;
            endpoint = 0;
        }
        if (initialized) {
            result = omniUsdLiveShutdown(remaining(deadline));
            if (result != OMNI_USD_LIVE_OK) return result;
            initialized = false;
        }
        return result;
    }
};
void check(OmniUsdLiveResult result) {
    if (result != OMNI_USD_LIVE_OK) throw result;
}
void snapshot(OmniUsdLiveEndpoint endpoint) {
    auto options = record<OmniUsdLiveDiagnosticExportOptions>();
    options.max_records = 128;
    options.max_output_bytes = 256 * 1024;
    uint64_t size = 0;
    check(omniUsdLiveCopyDiagnostics(endpoint, &options, nullptr, &size));
    for (unsigned attempt = 0; attempt < 3; ++attempt) {
        if (!size || size > options.max_output_bytes) throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
        std::string document(size, '\0');
        const auto result = omniUsdLiveCopyDiagnostics(endpoint, &options, document.data(), &size);
        if (result == OMNI_USD_LIVE_BUFFER_TOO_SMALL) continue;
        check(result);
        if (!size || size > document.size() || document[size - 1] != '\0')
            throw OmniUsdLiveResult(OMNI_USD_LIVE_INTERNAL_ERROR);
        // Called only after diagnostic callback quiescence; one complete JSON
        // document cannot interleave with individual diagnostic records. The
        // newline replaces the document's terminating NUL.
        document[size - 1] = '\n';
        if (!writeRecord(stdout, std::string_view(document.data(), size)))
            throw OmniUsdLiveResult(OMNI_USD_LIVE_IO_ERROR);
        return;
    }
    throw OmniUsdLiveResult(OMNI_USD_LIVE_BUSY);
}
int run(const Options& options, const char* token) {
    auto state = std::make_unique<State>();
    state->format = options.format;
    auto outcome = OmniUsdLiveResult(OMNI_USD_LIVE_INTERNAL_ERROR);
    try {
        auto runtime = record<OmniUsdLiveRuntimeOptions>();
        runtime.request_timeout_ms = options.timeout;
        runtime.default_auth_wait_timeout_ms = options.timeout;
        runtime.diagnostic_min_severity = options.debug ? OMNI_USD_LIVE_SEVERITY_DEBUG : OMNI_USD_LIVE_SEVERITY_INFO;
        check(omniUsdLiveInitialize(&runtime));
        state->initialized = true;
        auto diagnostics = record<OmniUsdLiveDiagnosticOptions>();
        diagnostics.min_severity = runtime.diagnostic_min_severity;
        check(omniUsdLiveRegisterDiagnosticCallback(&diagnostics, state.get(), State::diagnostic, &state->diagnostics));
        check(omniUsdLiveRegisterStatusCallback(0, state.get(), State::status, &state->statuses));
        auto endpoint = record<OmniUsdLiveEndpointOptions>();
        endpoint.authority = options.authority.c_str();
        endpoint.access_token = token;
        std::string caPem;
        if (!options.caBundle.empty()) {
            std::ifstream input(options.caBundle, std::ios::binary);
            if (!input) throw std::invalid_argument("cannot read CA bundle");
            caPem.resize(1024 * 1024 + 1);
            input.read(caPem.data(), caPem.size());
            caPem.resize(input.gcount());
            if (input.bad() || caPem.size() > 1024 * 1024) throw std::invalid_argument("invalid CA bundle");
        }
        endpoint.ca_pem = caPem.c_str();
        endpoint.tls_server_name = options.tlsServerName.c_str();
        endpoint.transport_security = options.transportSecurity;
        check(omniUsdLiveRegisterEndpoint(&endpoint, &state->endpoint));
        const auto deadline = Clock::now() + std::chrono::milliseconds(options.timeout);
        outcome = omniUsdLiveListSessions(state->endpoint, options.base.c_str(), state.get(), State::listed,
            &state->request);
        if (outcome == OMNI_USD_LIVE_OK) {
            outcome = omniUsdLiveWait(state->request, remaining(deadline));
            if (outcome == OMNI_USD_LIVE_OK) outcome = state->listResult.load();
        }
        std::fprintf(stderr, "list_sessions result=%u count=%llu\n", outcome,
            static_cast<unsigned long long>(state->sessionCount.load()));
        auto status = record<OmniUsdLiveStatus>();
        const auto queried = omniUsdLiveGetEndpointStatus(state->endpoint, &status);
        std::fprintf(stderr, "endpoint_status query=%u result=%u state=%u incident=%llu\n", queried, status.result,
            status.state, static_cast<unsigned long long>(status.incident_id));
        const auto diagnosticDeadline = Clock::now() + std::chrono::milliseconds(options.timeout);
        {
            // Wait for at least the endpoint lifecycle event, not for all logs
            // to drain: callback unregistration cancels queued diagnostics.
            std::unique_lock<std::mutex> lock(state->mutex);
            if (!state->changed.wait_until(lock, diagnosticDeadline,
                    [&] { return state->delivered && state->statusCount.load(); }))
                throw OmniUsdLiveResult(OMNI_USD_LIVE_TIMEOUT);
        }
        check(omniUsdLiveUnregisterCallback(state->diagnostics, remaining(diagnosticDeadline)));
        state->diagnostics = 0;
        if (options.snapshot) snapshot(state->endpoint);
    } catch (OmniUsdLiveResult result) { outcome = result; } catch (...) {
        outcome = OMNI_USD_LIVE_INTERNAL_ERROR;
    }
    const auto retired = state->retire(Clock::now() + std::chrono::milliseconds(options.timeout));
    std::fprintf(stderr, "status_callbacks=%llu terminal_status_callbacks=%llu\n",
        static_cast<unsigned long long>(state->statusCount.load()),
        static_cast<unsigned long long>(state->terminalCount.load()));
    std::fprintf(stderr, "sample_result=%u diagnostic_result=%u cleanup_result=%u\n", outcome, state->sinkResult.load(),
        retired);
    if (retired != OMNI_USD_LIVE_OK) {
        // A callback may still own State, including blocked stdout I/O. A host
        // must retain/retry its owners. This finite read-only sample terminates
        // its process instead of destroying live callback captures or hanging.
        std::fflush(stderr);
        exitNow(3);
    }
    return outcome == OMNI_USD_LIVE_OK && state->sinkResult.load() == OMNI_USD_LIVE_OK ? 0 : 1;
}
}
int main(int argc, char** argv) {
#if defined(SIGPIPE)
    std::signal(SIGPIPE, SIG_IGN); // Treat a closed log pipe as an ordinary sink failure.
#endif
    ExitDeadline watchdog;
    int result = 2;
    try {
        const auto options = parse(argc, argv);
        watchdog.arm(options.timeout);
        if (options.help) {
            std::puts(
                "Usage: live_diagnostics_sample --authority HOST:PORT --base-layer-key KEY [--token-env NAME] [--ca-bundle PEM_FILE] [--tls-server-name DNS_NAME] [--transport-security default|tls|plaintext] [--format jsonl|text] [--debug] [--snapshot] [--timeout-ms 1..60000]\n"
                "Reads session metadata only. Token defaults to LIVE_SERVICES_TOKEN. Logs go to stdout; results to stderr. No files are written automatically.");
            result = 0;
        } else {
            const auto* token = std::getenv(options.tokenEnv.c_str());
            if (!token || !*token) throw std::invalid_argument("token environment variable is missing or empty");
            result = run(options, token);
        }
    } catch (const std::invalid_argument& error) {
        // Messages are fixed literals. Never echo command-line or token values.
        std::fprintf(stderr, "Invalid arguments: %s. Use --help.\n", error.what());
    } catch (...) { std::fputs("Sample initialization failed.\n", stderr); }
    // Records are flushed as they are written; this flush covers --help and the
    // standard error results. Nothing is left for the C runtime's flush at exit.
    const auto stdoutResult = std::fflush(stdout);
    const auto stderrResult = std::fflush(stderr);
    if (stdoutResult || stderrResult) result = 1;
    return result;
}
