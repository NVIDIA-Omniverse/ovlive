// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "OmpfHash.h"
#include "SHA256.h"
#include <algorithm>
#include <vector>
namespace omni::usd::live::wire::detail {
std::array<uint8_t, 32> blockHash(std::string_view bytes) {
    std::vector<uint8_t> hashes(blockHashScratch(bytes.size()));
    for (size_t offset = 0, index = 0; offset < bytes.size(); ++index) {
        const size_t size = std::min(hashBlockBytes, bytes.size() - offset);
        computeSHA256(bytes.data() + offset, size, hashes.data() + index * 32);
        offset += size;
    }
    std::array<uint8_t, 32> result{};
    computeSHA256(hashes.data(), hashes.size(), result.data());
    return result;
}
} // namespace omni::usd::live::wire::detail
