// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <omni/usd/live/Live.h>
#include "LiveUri.h"

#include <cstring>
#include <new>
#include <stdexcept>

namespace {
using namespace omni::usd::live;

std::string_view input(const char* value) {
    if (!value) throw std::invalid_argument("null string");
    for (size_t i = 0; i <= kMaxUriBytes; ++i) {
        if (value[i] == '\0') return {value, i};
    }
    throw std::length_error("live identifier exceeds size limit");
}

template<class Action> OmniUsdLiveResult boundary(Action&& action) noexcept {
    try {
        return action();
    } catch (const std::invalid_argument&) { return OMNI_USD_LIVE_INVALID_ARGUMENT; } catch (const std::length_error&) {
        return OMNI_USD_LIVE_RESOURCE_EXHAUSTED;
    } catch (const std::bad_alloc&) { return OMNI_USD_LIVE_RESOURCE_EXHAUSTED; } catch (...) {
        return OMNI_USD_LIVE_INTERNAL_ERROR;
    }
}

OmniUsdLiveResult copyString(const std::string& text, char* buffer, uint64_t* size) {
    if (!size) return OMNI_USD_LIVE_INVALID_ARGUMENT;
    const uint64_t capacity = buffer ? *size : 0;
    *size = text.size() + 1;
    if (!buffer) return OMNI_USD_LIVE_OK;
    if (capacity < *size) return OMNI_USD_LIVE_BUFFER_TOO_SMALL;
    std::memcpy(buffer, text.c_str(), text.size() + 1);
    return OMNI_USD_LIVE_OK;
}
} // namespace

extern "C" {
OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveMakeSessionUri(const char* authority, const char* sessionId,
    char* buffer, uint64_t* size) {
    return boundary([&] { return copyString(makeSessionUri(input(authority), input(sessionId)), buffer, size); });
}

OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveMakeLayerUri(const char* sessionUri, const char* layerName,
    char* buffer, uint64_t* size) {
    return boundary([&] { return copyString(makeLayerUri(input(sessionUri), input(layerName)), buffer, size); });
}

OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveCombineWithBaseUri(const char* baseUri, const char* relativeLayerName,
    char* buffer, uint64_t* size) {
    return boundary(
        [&] { return copyString(combineWithBaseUri(input(baseUri), input(relativeLayerName)), buffer, size); });
}

OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveParseUri(const char* uri, OmniUsdLiveUriParts* parts, char* buffer,
    uint64_t* size) {
    return boundary([&]() -> OmniUsdLiveResult {
        if (!parts || !size || parts->struct_size < sizeof(OmniUsdLiveUriParts)) return OMNI_USD_LIVE_INVALID_ARGUMENT;
        if ((parts->abi_version >> 16) != (OMNI_USD_LIVE_ABI_VERSION >> 16)) return OMNI_USD_LIVE_UNSUPPORTED;
        const auto parsed = parseUri(input(uri));
        OmniUsdLiveUriParts output{};
        output.struct_size = parts->struct_size;
        output.abi_version = OMNI_USD_LIVE_ABI_VERSION;
        output.is_layer = parsed.layerName.empty() ? 0 : 1;
        std::string strings;
        const auto append = [&](const std::string& text) {
            const uint64_t offset = strings.size();
            strings += text;
            strings += '\0';
            return offset;
        };
        output.authority_offset = append(parsed.authority.canonical);
        output.session_id_offset = append(parsed.sessionId);
        output.layer_name_offset = append(parsed.layerName);
        output.canonical_uri_offset = strings.size();
        strings += parsed.canonical;
        const auto result = copyString(strings, buffer, size);
        if (result == OMNI_USD_LIVE_OK && buffer) std::memcpy(parts, &output, sizeof(output));
        return result;
    });
}
} // extern "C"
