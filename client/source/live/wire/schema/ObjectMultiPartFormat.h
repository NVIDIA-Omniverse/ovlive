// SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

constexpr uint32_t fourcc(char const p[5])
{
    return (p[0] << 0) | (p[1] << 8) | (p[2] << 16) | (p[3] << 24);
}

constexpr uint32_t ObjectMultiPartFormatId = fourcc("OMPF");

struct ObjectMultiPartHeader
{
    uint32_t formatId;
    uint32_t numParts;
    uint64_t offsets[1];
};
