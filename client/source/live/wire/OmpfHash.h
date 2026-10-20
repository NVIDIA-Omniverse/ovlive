// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <array>
#include <cstdint>
#include <string_view>
namespace omni::usd::live::wire::detail {
constexpr size_t hashBlockBytes = 1024 * 1024;
inline size_t blockHashScratch(size_t bytes) { return (bytes / hashBlockBytes + (bytes % hashBlockBytes != 0)) * 32; }
std::array<uint8_t, 32> blockHash(std::string_view);
} // namespace omni::usd::live::wire::detail
