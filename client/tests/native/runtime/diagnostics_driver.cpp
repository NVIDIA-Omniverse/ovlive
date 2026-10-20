// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "LiveUnary.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using namespace omni::usd::live;
namespace {
template <class T> T record()
{ T value{}; value.struct_size = sizeof(value); value.abi_version = OMNI_USD_LIVE_ABI_VERSION; return value; }

std::string copyDiagnostics(OmniUsdLiveEndpoint endpoint, const OmniUsdLiveDiagnosticExportOptions* options = nullptr)
{
    uint64_t size = 0;
    assert(omniUsdLiveCopyDiagnostics(endpoint, options, nullptr, &size) == OMNI_USD_LIVE_OK);
    for (unsigned attempt = 0; attempt < 4; ++attempt) {
        assert(size && size <= 8 * 1024 * 1024);
        std::vector<char> buffer(static_cast<size_t>(size), '\x5a');
        const auto result = omniUsdLiveCopyDiagnostics(endpoint, options, buffer.data(), &size);
        if (result == OMNI_USD_LIVE_BUFFER_TOO_SMALL) {
            assert(std::all_of(buffer.begin(), buffer.end(), [](char byte) { return byte == '\x5a'; }));
            continue;
        }
        assert(result == OMNI_USD_LIVE_OK && size && size <= buffer.size() && buffer[size - 1] == '\0');
        return std::string(buffer.data(), static_cast<size_t>(size - 1));
    }
    assert(false); return {};
}

struct PublicCalls {
    std::atomic<unsigned> calls{0};
    static void OMNI_USD_LIVE_CALL listed(void* user, OmniUsdLiveResult result, uint64_t,
                                         const OmniUsdLiveSessionInfo* const*)
    { assert(result == OMNI_USD_LIVE_OK); ++static_cast<PublicCalls*>(user)->calls; }
    static void OMNI_USD_LIVE_CALL stated(void* user, OmniUsdLiveResult result, const OmniUsdLiveLayerInfo* info)
    { assert(result == OMNI_USD_LIVE_OK && info); ++static_cast<PublicCalls*>(user)->calls; }
};

void finish(OmniUsdLiveRequest request, uint64_t attempts)
{
    assert(omniUsdLiveWait(request, 5000) == OMNI_USD_LIVE_OK);
    auto error = record<OmniUsdLiveRequestError>();
    assert(omniUsdLiveGetRequestError(request, &error) == OMNI_USD_LIVE_OK);
    assert(error.result == OMNI_USD_LIVE_OK && error.attempt_id == attempts);
    assert(omniUsdLiveReleaseRequest(request) == OMNI_USD_LIVE_OK);
}

struct HeldSinks {
    std::mutex mutex;
    std::condition_variable changed;
    unsigned entered = 0;
    bool release = false;
    static void OMNI_USD_LIVE_CALL callback(void* user, const OmniUsdLiveDiagnosticRecord* event)
    {
        auto& self = *static_cast<HeldSinks*>(user);
        assert(event && event->schema_version && event->runtime_instance_id && event->sequence);
        std::unique_lock<std::mutex> lock(self.mutex);
        ++self.entered; self.changed.notify_all();
        assert(self.changed.wait_for(lock, std::chrono::seconds(8), [&] { return self.release; }));
    }
};
struct StatusCalls {
    std::mutex mutex;
    std::condition_variable changed;
    bool ready = false;
    static void OMNI_USD_LIVE_CALL callback(void* user, const OmniUsdLiveStatusEvent* event)
    {
        auto& self = *static_cast<StatusCalls*>(user);
        assert(event && event->status);
        if (event->status->subject_kind == OMNI_USD_LIVE_SUBJECT_ENDPOINT &&
            event->status->state == OMNI_USD_LIVE_STATE_READY) {
            std::lock_guard<std::mutex> lock(self.mutex);
            self.ready = true; self.changed.notify_all();
        }
    }
};
}

int diagnosticsDriver(int argc, char** argv)
{
    assert(argc == 7);
    const std::string mode = argv[2];
    const bool blocked = mode == "diag-blocked";
    assert(blocked || mode == "diag-metrics");
    auto options = record<OmniUsdLiveRuntimeOptions>();
    options.retry_initial_ms = 5; options.retry_maximum_ms = 20; options.retry_max_attempts = 3;
    options.reconnect_initial_ms = 5; options.reconnect_maximum_ms = 20; options.reconnect_max_attempts = 3;
    options.request_timeout_ms = 3000;
    options.diagnostic_allowed_flags = OMNI_USD_LIVE_DIAGNOSTIC_RESOURCE_NAMES;
    options.diagnostic_min_severity = blocked ? OMNI_USD_LIVE_SEVERITY_TRACE : OMNI_USD_LIVE_SEVERITY_INFO;
    if (blocked) {
        options.max_diagnostic_records = 4; options.max_diagnostic_bytes = 8192;
        options.max_diagnostic_sink_records = 2;
    }
    assert(omniUsdLiveInitialize(&options) == OMNI_USD_LIVE_OK);
    HeldSinks held;
    std::array<OmniUsdLiveCallbackHandle, 2> sinks{};
    if (blocked) {
        auto filter = record<OmniUsdLiveDiagnosticOptions>();
        filter.min_severity = OMNI_USD_LIVE_SEVERITY_TRACE;
        for (auto& sink : sinks)
            assert(omniUsdLiveRegisterDiagnosticCallback(&filter, &held, HeldSinks::callback, &sink) == OMNI_USD_LIVE_OK);
    }
    auto endpointOptions = record<OmniUsdLiveEndpointOptions>();
    endpointOptions.authority = argv[1]; endpointOptions.access_token = argv[5];
    OmniUsdLiveEndpoint endpoint = 0;
    assert(omniUsdLiveRegisterEndpoint(&endpointOptions, &endpoint) == OMNI_USD_LIVE_OK);
    if (blocked) {
        std::unique_lock<std::mutex> lock(held.mutex);
        assert(held.changed.wait_for(lock, std::chrono::seconds(3), [&] { return held.entered == 2; }));
    }
    StatusCalls statuses;
    OmniUsdLiveCallbackHandle observer = 0;
    assert(omniUsdLiveRegisterStatusCallback(0, &statuses, StatusCalls::callback, &observer) == OMNI_USD_LIVE_OK);
    auto runtime = getRuntime();
    for (unsigned i = 0; i < (blocked ? 8u : 1u); ++i) {
        auto ping = submitUnary(*runtime, endpoint, rpc::PingRequest{}, UnaryCallback<rpc::PingResponse>{});
        assert(ping->wait(3000) == OMNI_USD_LIVE_OK);
        OmniUsdLiveResult result;
        assert(ping->result(&result) == OMNI_USD_LIVE_OK && result == OMNI_USD_LIVE_OK);
        assert(runtime->requests->release(ping->handle) == OMNI_USD_LIVE_OK);
    }
    PublicCalls calls;
    OmniUsdLiveRequest request = 0;
    assert(omniUsdLiveListSessions(endpoint, "base", &calls, PublicCalls::listed, &request) == OMNI_USD_LIVE_OK);
    finish(request, blocked ? 1 : 2);
    if (!blocked) {
        const auto uri = makeLayerUri(makeSessionUri(argv[1], argv[6]), "root.live");
        assert(omniUsdLiveStatLayer(uri.c_str(), &calls, PublicCalls::stated, &request) == OMNI_USD_LIVE_OK);
        finish(request, 2);
    }
    assert(calls.calls == (blocked ? 1u : 2u));
    {
        std::unique_lock<std::mutex> lock(statuses.mutex);
        assert(statuses.changed.wait_for(lock, std::chrono::seconds(3), [&] { return statuses.ready; }));
    }
    auto endpointStatus = record<OmniUsdLiveStatus>();
    assert(omniUsdLiveGetEndpointStatus(endpoint, &endpointStatus) == OMNI_USD_LIVE_OK);
    assert(endpointStatus.state == OMNI_USD_LIVE_STATE_READY);
    if (blocked) {
        std::lock_guard<std::mutex> lock(held.mutex);
        assert(!held.release && held.entered == 2); // Neither diagnostic worker helped the RPC/status work.
    }

    std::array<char, 32> sentinel; sentinel.fill('\x5a');
    uint64_t size = sentinel.size();
    assert(omniUsdLiveCopyDiagnostics(endpoint, nullptr, sentinel.data(), &size) == OMNI_USD_LIVE_BUFFER_TOO_SMALL);
    assert(size > sentinel.size() && std::all_of(sentinel.begin(), sentinel.end(), [](char byte) { return byte == '\x5a'; }));
    auto limited = record<OmniUsdLiveDiagnosticExportOptions>();
    limited.max_output_bytes = 16;
    size = sentinel.size();
    assert(omniUsdLiveCopyDiagnostics(endpoint, &limited, sentinel.data(), &size) == OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    assert(std::all_of(sentinel.begin(), sentinel.end(), [](char byte) { return byte == '\x5a'; }));
    size = 0;
    assert(omniUsdLiveCopyDiagnostics(endpoint + 1, nullptr, nullptr, &size) == OMNI_USD_LIVE_INVALID_HANDLE);
    const auto snapshot = copyDiagnostics(endpoint);
    auto noRecords = record<OmniUsdLiveDiagnosticExportOptions>();
    const auto withoutRecords = copyDiagnostics(endpoint, &noRecords);
    auto namedOptions = noRecords;
    namedOptions.flags = OMNI_USD_LIVE_DIAGNOSTIC_RESOURCE_NAMES; namedOptions.max_records = 32;
    const auto named = copyDiagnostics(endpoint, &namedOptions);
    for (const auto* value : {&snapshot, &withoutRecords, &named})
        assert(value->find(argv[5]) == std::string::npos);
    assert(snapshot.find(argv[1]) == std::string::npos && withoutRecords.find(argv[1]) == std::string::npos);

    if (blocked) {
        for (const auto sink : sinks)
            assert(omniUsdLiveUnregisterCallback(sink, 0) == OMNI_USD_LIVE_TIMEOUT);
    }
    assert(omniUsdLiveUnregisterCallback(observer, 3000) == OMNI_USD_LIVE_OK);
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    for (;;) {
        const auto result = omniUsdLiveUnregisterEndpoint(endpoint, 0);
        if (result == OMNI_USD_LIVE_OK) break;
        assert(result == OMNI_USD_LIVE_BUSY && std::chrono::steady_clock::now() < until);
        std::this_thread::yield();
    }
    if (blocked) {
        assert(omniUsdLiveShutdown(0) == OMNI_USD_LIVE_BUSY); // Only global diagnostic registrations remain.
        { std::lock_guard<std::mutex> lock(held.mutex); held.release = true; held.changed.notify_all(); }
        for (const auto sink : sinks) assert(omniUsdLiveUnregisterCallback(sink, 3000) == OMNI_USD_LIVE_OK);
    }
    runtime.reset();
    assert(omniUsdLiveShutdown(3000) == OMNI_USD_LIVE_OK);
    std::cout << "{\"snapshot\":" << snapshot << ",\"without_records\":" << withoutRecords
              << ",\"named\":" << named << "}" << std::endl;
    return 0;
}
