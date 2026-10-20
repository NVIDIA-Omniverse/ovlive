// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "LiveSession.h"
#include "LiveRequest.h"
#include "LiveStatus.h"
#include "LiveExceptionResult.h"
#include <cstring>

namespace {
using namespace omni::usd::live;
template<class T> T versioned(const T* pointer) {
    uint32_t size = 0;
    if (pointer) std::memcpy(&size, pointer, sizeof(size));
    if (!pointer || size < sizeof(T)) throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
    T value;
    std::memcpy(&value, pointer, sizeof(value));
    if ((value.abi_version >> 16) != (OMNI_USD_LIVE_ABI_VERSION >> 16))
        throw OmniUsdLiveResult(OMNI_USD_LIVE_UNSUPPORTED);
    return value;
}
std::string boundedText(const char* text) {
    if (!text || !*text) throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
    size_t length = 0;
    while (text[length]) {
        if (length == kMaxUriBytes) throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
        ++length;
    }
    return {text, length};
}
}

extern "C" {
OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveJoinSession(const char* uri, const OmniUsdLiveJoinOptions* options,
    void* user, OmniUsdLiveSessionCallback callback, OmniUsdLiveSession* output) {
    if (output) *output = 0;
    return resultBoundary([&]() -> OmniUsdLiveResult {
        if (!output || !callback) return OMNI_USD_LIVE_INVALID_ARGUMENT;
        const auto settings = versioned(options);
        auto app = boundedText(settings.app_name);
        const auto parsed = parseUri(boundedText(uri));
        if (!parsed.layerName.empty()) return OMNI_USD_LIVE_INVALID_ARGUMENT;
        auto runtime = getRuntime();
        if (!runtime) return OMNI_USD_LIVE_NOT_INITIALIZED;
        *output = runtime->sessions->join(parsed, std::move(app), user, callback);
        return OMNI_USD_LIVE_OK;
    });
}
OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveLeaveSession(OmniUsdLiveSession session, uint32_t timeoutMs) {
    return resultBoundary([&]() -> OmniUsdLiveResult {
        auto runtime = getRuntime();
        return runtime ? runtime->sessions->leave(session, timeoutMs) :
                         OmniUsdLiveResult(OMNI_USD_LIVE_NOT_INITIALIZED);
    });
}
OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveGetSessionStatus(OmniUsdLiveSession session,
    OmniUsdLiveStatus* output) {
    return resultBoundary([&]() -> OmniUsdLiveResult {
        // Read only the ABI prefix, not the caller's uninitialized output body.
        uint32_t size = 0, abi = 0;
        if (output) std::memcpy(&size, output, sizeof(size));
        if (!output || !acceptsOutputSize<OmniUsdLiveStatus>(size)) return OMNI_USD_LIVE_INVALID_ARGUMENT;
        std::memcpy(&abi, reinterpret_cast<const char*>(output) + offsetof(OmniUsdLiveStatus, abi_version),
            sizeof(abi));
        if ((abi >> 16) != (OMNI_USD_LIVE_ABI_VERSION >> 16)) return OMNI_USD_LIVE_UNSUPPORTED;
        auto runtime = getRuntime();
        return runtime ? runtime->sessions->status(session, output) : OmniUsdLiveResult(OMNI_USD_LIVE_NOT_INITIALIZED);
    });
}
OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveSendMessage(OmniUsdLiveSession session, const void* bytes,
    uint64_t size, void* user, OmniUsdLiveResultCallback callback, OmniUsdLiveRequest* output) {
    if (output) *output = 0;
    return resultBoundary([&]() -> OmniUsdLiveResult {
        if (!output || !callback || (!bytes && size)) return OMNI_USD_LIVE_INVALID_ARGUMENT;
        auto runtime = getRuntime();
        if (!runtime) return OMNI_USD_LIVE_NOT_INITIALIZED;
        auto request = runtime->sessions->send(session, bytes, size, user, callback);
        *output = request->handle;
        return OMNI_USD_LIVE_OK;
    });
}
}
