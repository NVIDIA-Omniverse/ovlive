// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "UsdValueCodec.h"
#include "OmpfEncodeInternal.h"
#include <algorithm>

namespace omni::usd::live::wire {
std::string encodeUsdEnvelope(const EnvelopeInput& metadata, const DecodedValues& values,
    const EnvelopeEncodeLimits& envelopeLimits, const EncodeLimits& valueLimits) {
    if (metadata.fields.size() != values.fields.size() || metadata.samples.size() != values.samples.size())
        throw DecodeError(ErrorKind::Malformed, "OMPF metadata/value count mismatch");
    // Validate before copying metadata or traversing USD values. Limits also
    // bound the complete field/sample scans below.
    detail::validateEnvelopeInput(metadata, envelopeLimits);
    for (const auto& field : metadata.fields)
        if (!field.value.empty()) throw DecodeError(ErrorKind::Malformed, "ambiguous OMPF field input");
    for (const auto& sample : metadata.samples)
        if (!sample.value.empty()) throw DecodeError(ErrorKind::Malformed, "ambiguous OMPF sample input");
    auto effective = valueLimits;
    effective.maxEncodedBytes = std::min(
        {effective.maxEncodedBytes, envelopeLimits.content.maxReferencedValueBytes, envelopeLimits.content.maxBytes});
    const auto encoded = encodeValues(values, effective);
    auto input = metadata;
    for (size_t i = 0; i < input.fields.size(); ++i) input.fields[i].value = encoded.fields[i];
    for (size_t i = 0; i < input.samples.size(); ++i) input.samples[i].value = encoded.samples[i];
    return encodeEnvelope(input, envelopeLimits);
}
} // namespace omni::usd::live::wire
