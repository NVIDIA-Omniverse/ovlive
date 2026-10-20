// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <omni/usd/live/Live.h>

#include <array>
#include <cassert>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    const uint64_t seed = argc > 1 ? std::stoull(argv[1]) : 42;
    std::cout << "uri_sanitize master_seed=" << seed << std::endl;
    std::mt19937_64 random(seed);
    constexpr std::array<const char*, 5> fixtures{
        "live://host/session/root.live", "live://[::1]/session/nested/root.live",
        "live://example.test:8443/a%2Fb/%F0%9F%A6%89.live", "live://host/session", ""};
    for (size_t iteration = 0; iteration < 20000; ++iteration) {
        std::string uri = fixtures[random() % fixtures.size()];
        const size_t changes = random() % 8;
        for (size_t n = 0; n < changes; ++n) {
            if (uri.empty() || random() % 3 == 0)
                uri.insert(random() % (uri.size() + 1), 1, static_cast<char>(1 + random() % 255));
            else
                uri[random() % uri.size()] = static_cast<char>(1 + random() % 255);
        }
        OmniUsdLiveUriParts parts{};
        parts.struct_size = sizeof(parts);
        parts.abi_version = OMNI_USD_LIVE_ABI_VERSION;
        uint64_t size = 0;
        auto result = omniUsdLiveParseUri(uri.c_str(), &parts, nullptr, &size);
        if (result == OMNI_USD_LIVE_OK) {
            assert(size > 0 && size <= 4 * 65537);
            std::vector<char> buffer(static_cast<size_t>(size), 'z');
            uint64_t shortSize = size - 1;
            assert(omniUsdLiveParseUri(uri.c_str(), &parts, buffer.data(), &shortSize) == OMNI_USD_LIVE_BUFFER_TOO_SMALL);
            for (char value : buffer)
                assert(value == 'z');
            assert(omniUsdLiveParseUri(uri.c_str(), &parts, buffer.data(), &size) == OMNI_USD_LIVE_OK);
            for (uint64_t offset : {parts.authority_offset, parts.session_id_offset,
                                    parts.layer_name_offset, parts.canonical_uri_offset}) {
                assert(offset < size);
                assert(std::memchr(buffer.data() + offset, 0, static_cast<size_t>(size - offset)) != nullptr);
            }
            OmniUsdLiveUriParts canonicalParts{};
            canonicalParts.struct_size = sizeof(canonicalParts);
            canonicalParts.abi_version = OMNI_USD_LIVE_ABI_VERSION;
            uint64_t canonicalSize = 0;
            assert(omniUsdLiveParseUri(buffer.data() + parts.canonical_uri_offset,
                                      &canonicalParts, nullptr, &canonicalSize) == OMNI_USD_LIVE_OK);
            assert(canonicalSize == size);
        } else {
            assert(result == OMNI_USD_LIVE_INVALID_ARGUMENT || result == OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
        }
    }
    std::cout << "20000 native malformed-input histories passed" << std::endl;
}
