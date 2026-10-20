// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace omni::usd::live {

// Finite allocation bound, independent of the runtime's queue limits.
constexpr size_t kMaxUriBytes = 65536;

struct Authority {
    std::string host; // No brackets; IPv6 uses canonical hexadecimal notation.
    uint16_t port = 443;
    bool ipLiteral = false;
    bool loopback = false;
    std::string canonical;
};

struct LiveUri {
    Authority authority;
    std::string sessionId;
    std::string layerName; // Empty for a session URI; '/' separates nested names.
    std::string canonical;
};

// Throw invalid_argument for invalid input and length_error for bounded capacity.
// These are private C++ helpers. C exports catch every exception at the boundary.
Authority parseAuthority(std::string_view input);
LiveUri parseUri(std::string_view input);
std::string makeSessionUri(std::string_view authority, std::string_view sessionId);
std::string makeLayerUri(std::string_view sessionUri, std::string_view layerName);
std::string combineWithBaseUri(std::string_view baseUri, std::string_view relativeLayerName);

} // namespace omni::usd::live
