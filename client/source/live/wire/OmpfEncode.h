// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "OmpfEnvelope.h"

namespace omni::usd::live::wire {
struct EncodedField {
    uint64_t section = 0, setOrder = 0;
    std::string name;
    std::string_view value; // Already encoded USD value; empty means deletion.
};
struct EncodedSample {
    uint64_t section = 0;
    double time = 0;
    std::string_view value;
};
struct EnvelopeInput {
    bool isDiff = false;
    uint64_t baseVersion = 0;
    std::string sourceFormat;
    std::vector<Command> commands;
    std::vector<EncodedField> fields;
    std::vector<EncodedSample> samples;
};
struct EnvelopeEncodeLimits {
    EnvelopeLimits content;
    size_t maxBuilderBytes = size_t(8) << 30; // Live FlatBuffers storage, including growth overlap.
    size_t maxScratchBytes = size_t(4) << 30;
    size_t externalValueThreshold = 65535; // Writer policy, not a wire validity restriction.
};
// Framing encoder only: values must already be validated/encoded. Input and
// borrowed value bytes stay immutable until return. Returns one owned complete
// OMPF buffer or throws; it never publishes partial bytes or transport state.
std::string encodeEnvelope(const EnvelopeInput&, const EnvelopeEncodeLimits& = {});
} // namespace omni::usd::live::wire
