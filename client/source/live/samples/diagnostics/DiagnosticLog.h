// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <omni/usd/live/Live.h>
#include <cstddef>
#include <string>

namespace omni::usd::live::sample {
enum class LogFormat { JsonLines, Text };
// Pure bounded formatting: no runtime calls or I/O. Output changes only on OK
// and contains exactly one final newline. Strings use reversible escaped-byte
// notation (high/control bytes as \u00XX), not an inferred UTF-8 conversion.
// Consumers interpret high-byte escapes as original bytes; current canonical
// event names/resource URIs are ASCII. This sample schema is separate from the
// runtime diagnostic snapshot's flat numeric fields.
// JSON fields map names to {kind,value}; integer values remain exact decimal
// numbers, booleans native JSON booleans. Text retains the numeric field kind.
// At most 16 fields, 64 bytes per nonempty event/key, 512 per string value, and
// 65536 output bytes including newline. These are logical bounds, not RSS bounds.
// Invalid records/options return INVALID_ARGUMENT, unsupported ABI major/schema
// returns UNSUPPORTED, bounded-size/allocation exhaustion RESOURCE_EXHAUSTED.
OmniUsdLiveResult formatDiagnosticRecord(const OmniUsdLiveDiagnosticRecord&, LogFormat, std::string& output,
    std::size_t maxBytes = 65536) noexcept;
} // namespace omni::usd::live::sample
