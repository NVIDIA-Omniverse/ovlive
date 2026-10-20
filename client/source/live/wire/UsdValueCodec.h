// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "OmpfEnvelope.h"
#include "OmpfEncode.h"
#include <pxr/base/vt/value.h>

namespace omni::usd::live::wire {
// Defaults sit above anything admission lets in (EnvelopeLimits); a value's
// elements and decoded bytes are linear in its encoded bytes.
struct ValueLimits {
    size_t maxEncodedBytes = size_t(16) << 30;
    size_t maxDecodedBytes = size_t(16) << 30;
    size_t maxElements = size_t(16) << 30;
    size_t maxStringBytes = size_t(4) << 30;
    size_t maxDepth = 64;
    size_t maxComparisonWork = size_t(16) << 30; // Conservative byte-work for opaque list equality.
};
// Version-zero little-endian, 64-bit-count USD values. Empty top-level bytes
// represent deletion; unknown type/version and nested empty values are errors.
// All bytes must be consumed. No scene or transport state is changed.
PXR_NS::VtValue decodeValue(std::string_view, const ValueLimits& = {});
struct EncodeLimits {
    size_t maxEncodedBytes = size_t(16) << 30;
    size_t maxElements = size_t(16) << 30;
    size_t maxStringBytes = size_t(4) << 30;
    size_t maxDepth = 64;
};
// Preflight the complete value before allocating output. Sources must remain
// immutable through both passes (the owning scene guard supplies this contract).
// Empty top-level values serialize as deletion. Nested empty/unsupported values
// and nonidentity payload offsets fail explicitly, without changing the input.
std::string encodeValue(const PXR_NS::VtValue&, const EncodeLimits& = {});
// The message of the Unsupported DecodeError for a nonidentity payload offset.
inline constexpr char kPayloadOffsetUnsupported[] = "USD payload offset is not representable in version zero";
struct DecodedValues {
    std::vector<PXR_NS::VtValue> fields;
    std::vector<PXR_NS::VtValue> samples;
};
struct ValueUsage {
    size_t encodedBytes = 0, decodedBytes = 0, elements = 0, comparisonWork = 0;
};
// One shared budget across all references (including external aliases). Returns
// only after every field/sample value is decoded; metadata remains in Envelope.
// Optional usage is written on success only and counts every reference.
DecodedValues decodeValues(const Envelope&, const ValueLimits& = {}, ValueUsage* usage = nullptr);
struct EncodedValues {
    std::vector<std::string> fields;
    std::vector<std::string> samples;
};
// Preflight the entire batch with one budget before allocating any output value.
EncodedValues encodeValues(const DecodedValues&, const EncodeLimits& = {});
// Shared-budget preflight without allocating encoded payload buffers. Used by
// retained-scene admission to bound the complete scene, not only its last delta.
// Optional element, string-byte and nesting-depth usage is written on success
// only; the depth is the least maxDepth the values fit.
size_t measureValues(const DecodedValues&, const EncodeLimits& = {}, size_t* elementsUsed = nullptr,
    size_t* stringsUsed = nullptr, size_t* depthUsed = nullptr);
// Typed outgoing bridge. Metadata field/sample counts must match values, and
// all metadata value views must be empty. Inputs stay immutable until return.
// Validate metadata, preflight/encode the complete value batch, then frame one
// owned payload. No caller-visible bytes or scene/transport effects on failure.
// Framing scratch limits cover its own temporary storage; this bridge also
// retains bounded metadata copies and encoded value buffers until return.
std::string encodeUsdEnvelope(const EnvelopeInput& metadata, const DecodedValues&, const EnvelopeEncodeLimits& = {},
    const EncodeLimits& = {});
} // namespace omni::usd::live::wire
