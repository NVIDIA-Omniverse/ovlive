// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "LiveUri.h"

#include <algorithm>
#include <array>
#include <stdexcept>

#if defined(_WIN32)
#    include <winsock2.h>
#    include <ws2tcpip.h>
#else
#    include <arpa/inet.h>
#endif

namespace omni::usd::live {
namespace {
[[noreturn]] void invalid() {
    // Never put potentially sensitive input in exceptions/diagnostic strings.
    throw std::invalid_argument("invalid live identifier");
}

void bound(size_t size) {
    if (size > kMaxUriBytes) throw std::length_error("live identifier exceeds size limit");
}

bool asciiDigit(char c) { return c >= '0' && c <= '9'; }

bool asciiLetter(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }

char lower(char c) { return c >= 'A' && c <= 'Z' ? static_cast<char>(c + ('a' - 'A')) : c; }

int hex(char c) {
    if (asciiDigit(c)) return c - '0';
    c = lower(c);
    return c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1;
}

bool unreserved(unsigned char c) {
    return asciiLetter(static_cast<char>(c)) || asciiDigit(static_cast<char>(c)) || c == '-' || c == '.' || c == '_' ||
        c == '~';
}

void validateText(std::string_view text) {
    bound(text.size());
    if (text.empty()) invalid();
    for (size_t i = 0; i < text.size();) {
        const auto first = static_cast<unsigned char>(text[i++]);
        uint32_t value = first;
        uint32_t minimum = 0;
        size_t trailing = 0;
        if (first >= 0x80) {
            if (first >= 0xC2 && first <= 0xDF) {
                value = first & 0x1F;
                trailing = 1;
                minimum = 0x80;
            } else if (first >= 0xE0 && first <= 0xEF) {
                value = first & 0x0F;
                trailing = 2;
                minimum = 0x800;
            } else if (first >= 0xF0 && first <= 0xF4) {
                value = first & 0x07;
                trailing = 3;
                minimum = 0x10000;
            } else {
                invalid();
            }
            if (trailing > text.size() - i) invalid();
            while (trailing--) {
                const auto next = static_cast<unsigned char>(text[i++]);
                if ((next & 0xC0) != 0x80) invalid();
                value = (value << 6) | (next & 0x3F);
            }
        }
        if (value < minimum || value > 0x10FFFF || (value >= 0xD800 && value <= 0xDFFF) || value <= 0x1F ||
            (value >= 0x7F && value <= 0x9F) || value == '\\')
            invalid();
    }
}

std::string encode(std::string_view text) {
    validateText(text);
    std::string result;
    constexpr char digits[] = "0123456789ABCDEF";
    for (unsigned char c : text) {
        if (unreserved(c)) {
            result += static_cast<char>(c);
        } else {
            result += '%';
            result += digits[c >> 4];
            result += digits[c & 15];
        }
        bound(result.size());
    }
    return result;
}

std::string decode(std::string_view text, bool session) {
    std::string result;
    for (size_t i = 0; i < text.size(); ++i) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        if (c == '%') {
            if (text.size() - i < 3 || hex(text[i + 1]) < 0 || hex(text[i + 2]) < 0) invalid();
            c = static_cast<unsigned char>((hex(text[i + 1]) << 4) | hex(text[i + 2]));
            i += 2;
            // Encoded slash is data in opaque session IDs. Layer nesting must
            // use literal separators so it cannot gain hidden path components.
            if (!session && c == '/') invalid();
        } else if (!unreserved(c)) {
            // Canonical URI bytes are ASCII. Builders accept decoded UTF-8.
            invalid();
        }
        result += static_cast<char>(c);
    }
    validateText(result);
    if (result == "." || result == "..") invalid();
    return result;
}

std::string encodedLayer(std::string_view layer) {
    validateText(layer);
    if (layer.size() < 5 || layer.substr(layer.size() - 5) != ".live") invalid();
    std::string result;
    size_t start = 0;
    for (;;) {
        const auto end = layer.find('/', start);
        const auto part = layer.substr(start, end == std::string_view::npos ? end : end - start);
        if (part.empty() || part == "." || part == "..") invalid();
        if (!result.empty()) result += '/';
        result += encode(part);
        if (end == std::string_view::npos) return result;
        start = end + 1;
    }
}

std::string ipv6Text(const std::array<unsigned char, 16>& bytes) {
    std::array<uint16_t, 8> words{};
    for (size_t i = 0; i < words.size(); ++i) words[i] = static_cast<uint16_t>((bytes[2 * i] << 8) | bytes[2 * i + 1]);
    size_t bestStart = words.size(), bestLength = 0;
    for (size_t i = 0; i < words.size();) {
        if (words[i] != 0) {
            ++i;
            continue;
        }
        const size_t start = i;
        while (i < words.size() && words[i] == 0) ++i;
        if (i - start > bestLength && i - start >= 2) {
            bestStart = start;
            bestLength = i - start;
        }
    }
    std::string result;
    constexpr char digits[] = "0123456789abcdef";
    for (size_t i = 0; i < words.size();) {
        if (i == bestStart) {
            result += "::";
            i += bestLength;
            continue;
        }
        if (!result.empty() && result.back() != ':') result += ':';
        uint16_t word = words[i++];
        bool writing = false;
        for (int shift = 12; shift >= 0; shift -= 4) {
            const auto digit = (word >> shift) & 15;
            if (digit || writing || shift == 0) {
                result += digits[digit];
                writing = true;
            }
        }
    }
    return result;
}

bool parseIp(int family, const std::string& text, void* bytes) {
#if defined(_WIN32)
    return InetPtonA(family, text.c_str(), bytes) == 1;
#else
    return inet_pton(family, text.c_str(), bytes) == 1;
#endif
}
} // namespace

Authority parseAuthority(std::string_view input) {
    bound(input.size());
    if (input.empty() || input.find_first_of("/@?#%\\ \t\r\n") != std::string_view::npos) invalid();
    for (unsigned char c : input) {
        if (c <= 0x20 || c >= 0x7F) invalid();
    }
    Authority result;
    std::string_view host, port;
    bool hasPort = false;
    bool ipv6 = input.front() == '[';
    if (ipv6) {
        const auto close = input.find(']');
        if (close == std::string_view::npos) invalid();
        host = input.substr(1, close - 1);
        if (close + 1 != input.size()) {
            if (input[close + 1] != ':') invalid();
            hasPort = true;
            port = input.substr(close + 2);
        }
    } else {
        const auto colon = input.find(':');
        host = input.substr(0, colon);
        if (colon != std::string_view::npos) {
            hasPort = true;
            port = input.substr(colon + 1);
        }
    }
    if (host.empty()) invalid();
    if (hasPort) {
        if (port.empty() || port.size() > 5) invalid();
        uint32_t value = 0;
        for (char c : port) {
            if (!asciiDigit(c)) invalid();
            value = value * 10 + static_cast<uint32_t>(c - '0');
        }
        if (value == 0 || value > 65535) invalid();
        result.port = static_cast<uint16_t>(value);
    }
    result.host = std::string(host);
    if (ipv6) {
        std::array<unsigned char, 16> bytes{};
        if (!parseIp(AF_INET6, result.host, bytes.data())) invalid();
        result.host = ipv6Text(bytes);
        result.ipLiteral = true;
        result.loopback = result.host == "::1";
    } else {
        std::transform(result.host.begin(), result.host.end(), result.host.begin(), lower);
        const bool numeric = std::all_of(result.host.begin(), result.host.end(),
            [](char c) { return asciiDigit(c) || c == '.'; });
        if (numeric) {
            std::array<unsigned char, 4> bytes{};
            if (!parseIp(AF_INET, result.host, bytes.data()))
                invalid(); // Also rejects old process-local numeric provider keys.
            // The authority uses canonical decimal octets on every platform.
            if (result.host !=
                std::to_string(bytes[0]) + '.' + std::to_string(bytes[1]) + '.' + std::to_string(bytes[2]) + '.' +
                    std::to_string(bytes[3]))
                invalid();
            result.ipLiteral = true;
            result.loopback = result.host == "127.0.0.1";
        } else {
            if (result.host.back() == '.') result.host.pop_back();
            if (result.host.empty() || result.host.size() > 253) invalid();
            size_t start = 0;
            for (size_t i = 0; i <= result.host.size(); ++i) {
                if (i == result.host.size() || result.host[i] == '.') {
                    if (i == start || i - start > 63 || result.host[start] == '-' || result.host[i - 1] == '-')
                        invalid();
                    start = i + 1;
                } else if (!asciiLetter(result.host[i]) && !asciiDigit(result.host[i]) && result.host[i] != '-') {
                    invalid();
                }
            }
            result.loopback = result.host == "localhost";
        }
    }
    result.canonical = ipv6 ? "[" + result.host + "]" : result.host;
    if (result.port != 443) result.canonical += ':' + std::to_string(result.port);
    return result;
}

std::string makeSessionUri(std::string_view authority, std::string_view sessionId) {
    if (sessionId == "." || sessionId == "..") invalid();
    auto result = "live://" + parseAuthority(authority).canonical + '/' + encode(sessionId);
    bound(result.size());
    return result;
}

LiveUri parseUri(std::string_view input) {
    bound(input.size());
    if (input.substr(0, 7) != "live://" || input.find_first_of("?#") != std::string_view::npos) invalid();
    const auto authorityEnd = input.find('/', 7);
    if (authorityEnd == std::string_view::npos) invalid();
    LiveUri result;
    result.authority = parseAuthority(input.substr(7, authorityEnd - 7));
    const auto sessionEnd = input.find('/', authorityEnd + 1);
    result.sessionId = decode(input.substr(authorityEnd + 1,
                                  sessionEnd == std::string_view::npos ? sessionEnd : sessionEnd - authorityEnd - 1),
        true);
    result.canonical = makeSessionUri(result.authority.canonical, result.sessionId);
    if (sessionEnd != std::string_view::npos) {
        size_t start = sessionEnd + 1;
        for (;;) {
            const auto end = input.find('/', start);
            if (!result.layerName.empty()) result.layerName += '/';
            result.layerName += decode(input.substr(start, end == std::string_view::npos ? end : end - start), false);
            if (end == std::string_view::npos) break;
            start = end + 1;
        }
        result.canonical += '/' + encodedLayer(result.layerName);
    }
    bound(result.canonical.size());
    return result;
}

std::string makeLayerUri(std::string_view sessionUri, std::string_view layerName) {
    const auto session = parseUri(sessionUri);
    if (!session.layerName.empty()) invalid();
    auto result = session.canonical + '/' + encodedLayer(layerName);
    bound(result.size());
    return result;
}

std::string combineWithBaseUri(std::string_view baseUri, std::string_view relativeLayerName) {
    auto base = parseUri(baseUri);
    // Validate the input before prefixing so leading slash/dot traversal cannot
    // escape the base layer's directory or the session component.
    encodedLayer(relativeLayerName);
    std::string layer;
    const auto separator = base.layerName.rfind('/');
    if (separator != std::string::npos) layer = base.layerName.substr(0, separator + 1);
    layer += relativeLayerName;
    return makeLayerUri(makeSessionUri(base.authority.canonical, base.sessionId), layer);
}
} // namespace omni::usd::live
