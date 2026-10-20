// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "OmpfEncode.h"

namespace omni::usd::live::wire::detail {
// Validate metadata and conservative resource bounds without constructing a
// FlatBuffer. Complete output sizing remains encodeEnvelope's responsibility.
void validateEnvelopeInput(const EnvelopeInput&, const EnvelopeEncodeLimits&);
}
