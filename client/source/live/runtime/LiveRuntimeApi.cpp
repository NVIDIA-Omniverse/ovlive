// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "LiveRuntime.h"
#include "LiveRequest.h"
#include "LiveAuth.h"
#include "LiveDrain.h"
#include "LiveStatus.h"
#include "LiveDiagnostics.h"
#include "LiveExceptionResult.h"

#include <cstring>

namespace {
using namespace omni::usd::live;

template<class Action> OmniUsdLiveResult withRequest(OmniUsdLiveRequest handle, Action&& action) {
    return resultBoundary([&]() -> OmniUsdLiveResult {
        auto runtime = getRuntime();
        if (!runtime) return OMNI_USD_LIVE_NOT_INITIALIZED;
        auto request = runtime->requests->find(handle);
        return request ? action(*request) : OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_HANDLE);
    });
}
} // namespace

extern "C" {
OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveWaitForPendingUpdates(uint32_t timeoutMs) {
    return resultBoundary([&]() -> OmniUsdLiveResult {
        auto runtime = getRuntime();
        if (!runtime) return OMNI_USD_LIVE_NOT_INITIALIZED;
        const auto started = std::chrono::steady_clock::now();
        const auto result = runtime->drains->wait(timeoutMs, !DispatchExecutor::isWorker());
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started).count();
        runtime->diagnostics->emit(result == OMNI_USD_LIVE_TIMEOUT ? OMNI_USD_LIVE_RECORD_DRAIN_TIMEOUT :
                                                                     OMNI_USD_LIVE_RECORD_DRAIN_COMPLETED,
            0, 0, 0, {{DiagnosticKey::Result, result}, {DiagnosticKey::DurationMs, static_cast<uint64_t>(elapsed)}}, {},
            result == OMNI_USD_LIVE_OK ? OMNI_USD_LIVE_SEVERITY_INFO : OMNI_USD_LIVE_SEVERITY_WARN);
        return result;
    });
}

OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveConfigureJitterReduction(OmniUsdLiveEndpoint endpoint,
    const OmniUsdLiveJitterParams* params) {
    return resultBoundary([&]() -> OmniUsdLiveResult {
        uint32_t size = 0;
        if (params) std::memcpy(&size, params, sizeof(size));
        if (!params || size < sizeof(*params)) return OMNI_USD_LIVE_INVALID_ARGUMENT;
        OmniUsdLiveJitterParams value{};
        std::memcpy(&value, params, sizeof(value));
        if ((value.abi_version >> 16) != (OMNI_USD_LIVE_ABI_VERSION >> 16)) return OMNI_USD_LIVE_UNSUPPORTED;
        value.struct_size = sizeof(value);
        value.abi_version = OMNI_USD_LIVE_ABI_VERSION;
        auto runtime = getRuntime();
        if (!runtime) return OMNI_USD_LIVE_NOT_INITIALIZED;
        if (endpoint == 0) {
            std::lock_guard<std::mutex> lock(runtime->mutex);
            if (runtime->closing) return OMNI_USD_LIVE_BUSY;
            runtime->defaultJitter = value;
        } else {
            auto found = runtime->findEndpoint(endpoint);
            if (!found) return OMNI_USD_LIVE_INVALID_HANDLE;
            std::lock_guard<std::mutex> lock(found->mutex);
            if (found->closed) return OMNI_USD_LIVE_INVALID_HANDLE;
            found->jitter = value;
        }
        // A timer may already be waiting on a holdback computed from the old
        // configuration. Wake after releasing the configuration locks.
        runtime->auth->wake();
        return OMNI_USD_LIVE_OK;
    });
}

OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveGetAuthState(OmniUsdLiveEndpoint endpoint,
    OmniUsdLiveAuthState* state) {
    return resultBoundary([&]() -> OmniUsdLiveResult {
        uint32_t size = 0;
        if (state) std::memcpy(&size, state, sizeof(size));
        if (!state || size < sizeof(*state)) return OMNI_USD_LIVE_INVALID_ARGUMENT;
        if ((state->abi_version >> 16) != (OMNI_USD_LIVE_ABI_VERSION >> 16)) return OMNI_USD_LIVE_UNSUPPORTED;
        auto runtime = getRuntime();
        return runtime ? runtime->auth->snapshot(endpoint, state) : OmniUsdLiveResult(OMNI_USD_LIVE_NOT_INITIALIZED);
    });
}
OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveSetAccessToken(OmniUsdLiveEndpoint endpoint, uint64_t generation,
    const char* token, uint64_t expiresAtMs, uint64_t* newGeneration) {
    if (newGeneration) *newGeneration = 0;
    return resultBoundary([&]() -> OmniUsdLiveResult {
        if (!newGeneration || !token) return OMNI_USD_LIVE_INVALID_ARGUMENT;
        auto runtime = getRuntime();
        return runtime ? runtime->auth->replace(endpoint, generation, token, expiresAtMs, newGeneration) :
                         OmniUsdLiveResult(OMNI_USD_LIVE_NOT_INITIALIZED);
    });
}
OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveRegisterAuthRequiredCallback(OmniUsdLiveEndpoint endpoint, void* user,
    OmniUsdLiveAuthRequiredCallback callback, OmniUsdLiveCallbackHandle* handle) {
    if (handle) *handle = 0;
    return resultBoundary([&]() -> OmniUsdLiveResult {
        if (!callback || !handle) return OMNI_USD_LIVE_INVALID_ARGUMENT;
        auto runtime = getRuntime();
        return runtime ? runtime->auth->subscribe(endpoint, user, callback, handle) :
                         OmniUsdLiveResult(OMNI_USD_LIVE_NOT_INITIALIZED);
    });
}
OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveFailAuthRefresh(OmniUsdLiveEndpoint endpoint, uint64_t challenge) {
    return resultBoundary([&]() -> OmniUsdLiveResult {
        auto runtime = getRuntime();
        return runtime ? runtime->auth->fail(endpoint, challenge) : OmniUsdLiveResult(OMNI_USD_LIVE_NOT_INITIALIZED);
    });
}
OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveUnregisterCallback(OmniUsdLiveCallbackHandle handle,
    uint32_t timeoutMs) {
    return resultBoundary([&]() -> OmniUsdLiveResult {
        auto runtime = getRuntime();
        if (!runtime) return OmniUsdLiveResult(OMNI_USD_LIVE_NOT_INITIALIZED);
        if ((handle >> 56) == 8) return runtime->diagnostics->unsubscribe(handle, timeoutMs);
        return (handle >> 56) == 5 ? runtime->statuses->unsubscribe(handle, timeoutMs) :
                                     runtime->auth->unsubscribe(handle, timeoutMs);
    });
}

OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveWait(OmniUsdLiveRequest request, uint32_t timeoutMs) {
    return withRequest(request, [&](Request& value) { return value.wait(timeoutMs); });
}

OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveCancel(OmniUsdLiveRequest request, uint32_t timeoutMs) {
    return withRequest(request, [&](Request& value) { return value.cancel(timeoutMs); });
}

OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveGetRequestStatus(OmniUsdLiveRequest request,
    OmniUsdLiveResult* result) {
    return withRequest(request, [&](Request& value) { return value.result(result); });
}

OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveGetRequestError(OmniUsdLiveRequest request,
    OmniUsdLiveRequestError* error) {
    return withRequest(request, [&](Request& value) { return value.error(error); });
}

OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveReleaseRequest(OmniUsdLiveRequest request) {
    return resultBoundary([&]() -> OmniUsdLiveResult {
        auto runtime = getRuntime();
        return runtime ? runtime->requests->release(request) : OmniUsdLiveResult(OMNI_USD_LIVE_NOT_INITIALIZED);
    });
}

OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveGetAbiVersion(uint32_t* version) {
    if (!version) return OMNI_USD_LIVE_INVALID_ARGUMENT;
    *version = OMNI_USD_LIVE_ABI_VERSION;
    return OMNI_USD_LIVE_OK;
}

OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveGetCapabilities(OmniUsdLiveCapabilities* capabilities) {
    uint32_t size = 0;
    if (capabilities) std::memcpy(&size, capabilities, sizeof(size));
    if (!capabilities || size < sizeof(*capabilities)) return OMNI_USD_LIVE_INVALID_ARGUMENT;
    uint32_t abi = 0;
    std::memcpy(&abi, reinterpret_cast<const char*>(capabilities) + sizeof(uint32_t), sizeof(abi));
    if ((abi >> 16) != (OMNI_USD_LIVE_ABI_VERSION >> 16)) return OMNI_USD_LIVE_UNSUPPORTED;
    // Static implementation/schema facts, independent of initialization or
    // endpoint state. These fields do not advertise platform/Kit qualification
    // or broaden the supported USD authoring operation matrix.
    OmniUsdLiveCapabilities result{};
    result.struct_size = size;
    result.abi_version = OMNI_USD_LIVE_ABI_VERSION;
    result.delegate_version = 2;
    result.diagnostic_schema_version = OMNI_USD_LIVE_DIAGNOSTIC_SCHEMA_VERSION;
    result.autonomous_progress = 1;
    std::memcpy(capabilities, &result, sizeof(result));
    return OMNI_USD_LIVE_OK;
}

OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveInitialize(const OmniUsdLiveRuntimeOptions* options) {
    return resultBoundary([&] { return initialize(options); });
}

OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveShutdown(uint32_t timeoutMs) {
    return resultBoundary([&] { return shutdown(timeoutMs); });
}

OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveRegisterEndpoint(const OmniUsdLiveEndpointOptions* options,
    OmniUsdLiveEndpoint* endpoint) {
    if (endpoint) *endpoint = 0;
    return resultBoundary([&]() -> OmniUsdLiveResult {
        if (!options || !endpoint) return OMNI_USD_LIVE_INVALID_ARGUMENT;
        auto runtime = getRuntime();
        if (!runtime) return OMNI_USD_LIVE_NOT_INITIALIZED;
        return runtime->registerEndpoint(EndpointOptions::copy(options, runtime->options), endpoint);
    });
}

OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveUnregisterEndpoint(OmniUsdLiveEndpoint endpoint, uint32_t timeoutMs) {
    // Dependents hold endpoint leases, so final release is BUSY at once while
    // any exists (Live.h); the endpoint owns no other internal work to drain,
    // so timeout_ms has nothing to bound.
    (void)timeoutMs;
    return resultBoundary([&]() -> OmniUsdLiveResult {
        auto runtime = getRuntime();
        return runtime ? runtime->unregisterEndpoint(endpoint) : OmniUsdLiveResult(OMNI_USD_LIVE_NOT_INITIALIZED);
    });
}

OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveGetEndpointStatus(OmniUsdLiveEndpoint endpoint,
    OmniUsdLiveStatus* status) {
    return resultBoundary([&]() -> OmniUsdLiveResult {
        uint32_t size = 0;
        if (status) std::memcpy(&size, status, sizeof(size));
        if (!status || !acceptsOutputSize<OmniUsdLiveStatus>(size)) return OMNI_USD_LIVE_INVALID_ARGUMENT;
        if ((status->abi_version >> 16) != (OMNI_USD_LIVE_ABI_VERSION >> 16)) return OMNI_USD_LIVE_UNSUPPORTED;
        auto runtime = getRuntime();
        if (!runtime) return OMNI_USD_LIVE_NOT_INITIALIZED;
        auto found = runtime->findEndpoint(endpoint);
        if (!found) return OMNI_USD_LIVE_INVALID_HANDLE;
        std::lock_guard<std::mutex> lock(found->mutex);
        if (found->closed) return OMNI_USD_LIVE_INVALID_HANDLE;
        return runtime->statuses->snapshot(found->status.subject_id, status);
    });
}

OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveGetLayerStatus(const char* uri, OmniUsdLiveStatus* status) {
    return resultBoundary([&]() -> OmniUsdLiveResult {
        uint32_t size = 0;
        if (status) std::memcpy(&size, status, sizeof(size));
        if (!uri || !status || !acceptsOutputSize<OmniUsdLiveStatus>(size)) return OMNI_USD_LIVE_INVALID_ARGUMENT;
        if ((status->abi_version >> 16) != (OMNI_USD_LIVE_ABI_VERSION >> 16)) return OMNI_USD_LIVE_UNSUPPORTED;
        const auto parsed = parseUri(uri);
        if (parsed.layerName.empty()) return OMNI_USD_LIVE_INVALID_ARGUMENT;
        auto runtime = getRuntime();
        return runtime ? runtime->statuses->layer(parsed.canonical, status) :
                         OmniUsdLiveResult(OMNI_USD_LIVE_NOT_INITIALIZED);
    });
}

OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveRegisterStatusCallback(OmniUsdLiveEndpoint endpoint, void* user,
    OmniUsdLiveStatusCallback callback, OmniUsdLiveCallbackHandle* output) {
    if (output) *output = 0;
    return resultBoundary([&]() -> OmniUsdLiveResult {
        if (!callback || !output) return OMNI_USD_LIVE_INVALID_ARGUMENT;
        auto runtime = getRuntime();
        if (!runtime) return OMNI_USD_LIVE_NOT_INITIALIZED;
        // Declare lease before lock: failures must retire ownership after unlock.
        std::shared_ptr<void> lease;
        if (endpoint) {
            auto dependency = runtime->acquireEndpoint(endpoint);
            if (!dependency) return runtime->closing ? OMNI_USD_LIVE_BUSY : OMNI_USD_LIVE_INVALID_HANDLE;
            lease = std::make_shared<EndpointLease>(std::move(dependency));
        }
        std::lock_guard<std::mutex> lock(runtime->mutex);
        if (runtime->closing) return OMNI_USD_LIVE_BUSY;
        *output = runtime->statuses->subscribe(endpoint, user, callback, lease);
        return OMNI_USD_LIVE_OK;
    });
}

OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveCreateStatusObserver(OmniUsdLiveEndpoint endpoint,
    OmniUsdLiveCallbackHandle* output) {
    if (output) *output = 0;
    return resultBoundary([&]() -> OmniUsdLiveResult {
        if (!output) return OMNI_USD_LIVE_INVALID_ARGUMENT;
        auto runtime = getRuntime();
        if (!runtime) return OMNI_USD_LIVE_NOT_INITIALIZED;
        std::shared_ptr<void> lease;
        if (endpoint) {
            auto dependency = runtime->acquireEndpoint(endpoint);
            if (!dependency) return runtime->closing ? OMNI_USD_LIVE_BUSY : OMNI_USD_LIVE_INVALID_HANDLE;
            lease = std::make_shared<EndpointLease>(std::move(dependency));
        }
        std::lock_guard<std::mutex> lock(runtime->mutex);
        if (runtime->closing) return OMNI_USD_LIVE_BUSY;
        *output = runtime->statuses->createObserver(endpoint, lease);
        return OMNI_USD_LIVE_OK;
    });
}

OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveReadStatus(OmniUsdLiveCallbackHandle observer, uint32_t timeoutMs,
    OmniUsdLiveStatusObservation* output, char* uri, uint64_t* uriSize) {
    return resultBoundary([&]() -> OmniUsdLiveResult {
        uint32_t size = 0, abiVersion = 0;
        if (output) std::memcpy(&size, output, sizeof(size));
        if (!output || !uriSize || !acceptsOutputSize<OmniUsdLiveStatusObservation>(size))
            return OMNI_USD_LIVE_INVALID_ARGUMENT;
        std::memcpy(&abiVersion, reinterpret_cast<const char*>(output) + sizeof(uint32_t), sizeof(abiVersion));
        if ((abiVersion >> 16) != (OMNI_USD_LIVE_ABI_VERSION >> 16)) return OMNI_USD_LIVE_UNSUPPORTED;
        auto runtime = getRuntime();
        return runtime ? runtime->statuses->read(observer, timeoutMs, output, uri, uriSize) :
                         OmniUsdLiveResult(OMNI_USD_LIVE_NOT_INITIALIZED);
    });
}

OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveAcknowledgeStatus(OmniUsdLiveCallbackHandle observer, uint64_t subject,
    uint64_t revision) {
    return resultBoundary([&]() -> OmniUsdLiveResult {
        if (!subject || !revision) return OMNI_USD_LIVE_INVALID_ARGUMENT;
        auto runtime = getRuntime();
        return runtime ? runtime->statuses->acknowledge(observer, subject, revision) :
                         OmniUsdLiveResult(OMNI_USD_LIVE_NOT_INITIALIZED);
    });
}

OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveGetResultString(OmniUsdLiveResult result, const char** text) {
    if (!text) return OMNI_USD_LIVE_INVALID_ARGUMENT;
    // Static strings survive runtime teardown and never contain transport detail.
    // Result 17 is reserved and maps to the unknown-result name.
    static const char* const names[] = {"OK", "INVALID_ARGUMENT", "INVALID_HANDLE", "NOT_INITIALIZED", "NOT_FOUND",
        "ALREADY_EXISTS", "UNAUTHENTICATED", "PERMISSION_DENIED", "UNAVAILABLE", "TIMEOUT", "CANCELLED",
        "MALFORMED_STREAM", "UNSUPPORTED", "RESOURCE_EXHAUSTED", "BUSY", "STALE_TOKEN_GENERATION", "OBJECT_REPLACED",
        "UNKNOWN_RESULT", "BUFFER_TOO_SMALL", "OUTCOME_UNKNOWN", "INTERNAL_ERROR", "IO_ERROR"};
    *text = result < sizeof(names) / sizeof(names[0]) ? names[result] : "UNKNOWN_RESULT";
    return OMNI_USD_LIVE_OK;
}
} // extern "C"
