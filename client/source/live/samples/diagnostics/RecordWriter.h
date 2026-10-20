// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdio>
#include <string_view>

namespace omni::usd::live::sample {
// Writes one whole record and flushes it before returning, so no record waits for the C runtime's flush at process
// exit. On Windows that flush runs in the C runtime's DLL_PROCESS_DETACH, after ExitProcess has ended every other
// thread, and a process can end before it (docs/ovlive-samples.md#output).
inline bool writeRecord(std::FILE* stream, std::string_view record) noexcept {
    return std::fwrite(record.data(), 1, record.size(), stream) == record.size() && std::fflush(stream) == 0;
}
} // namespace omni::usd::live::sample
