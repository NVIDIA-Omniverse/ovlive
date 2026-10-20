// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <string_view>
namespace omni::usd::live::wire::detail {
inline bool validNameUtf8(std::string_view text) {
    for (size_t i = 0; i < text.size();) {
        const auto a = static_cast<uint8_t>(text[i++]);
        if (!a) return false;
        if (a < 0x80) continue;
        unsigned count;
        uint32_t value, minimum;
        if (a >= 0xc2 && a <= 0xdf) {
            count = 1;
            value = a & 31;
            minimum = 0x80;
        } else if (a >= 0xe0 && a <= 0xef) {
            count = 2;
            value = a & 15;
            minimum = 0x800;
        } else if (a >= 0xf0 && a <= 0xf4) {
            count = 3;
            value = a & 7;
            minimum = 0x10000;
        } else return false;
        if (count > text.size() - i) return false;
        while (count--) {
            const auto b = static_cast<uint8_t>(text[i++]);
            if ((b & 0xc0) != 0x80) return false;
            value = (value << 6) | (b & 63);
        }
        if (value < minimum || value > 0x10ffff || (value >= 0xd800 && value <= 0xdfff)) return false;
    }
    return true;
}
} // namespace omni::usd::live::wire::detail
