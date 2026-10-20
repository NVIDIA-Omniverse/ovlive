// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "DeltaEntries.h"
#include "WireScene.h"

namespace omni::usd::live {
struct SceneSeedLimits {
    WireSceneLimits scene;
    wire::EnvelopeEncodeLimits envelope;
    wire::EncodeLimits values;
    // The service's entry limits for the seed's one CreateLayer; a seed over
    // them throws DeltaEntryLimitExceeded before its values are encoded.
    DeltaEntryLimits entries;
    // A seed is one upload, bounded by max_update_bytes; these defaults sit above
    // what such an upload holds.
    size_t maxMetadataBytes = size_t(16) << 30;
    size_t maxWork = size_t(64) << 30;
};
struct SceneSeed {
    std::string bytes;
    WireScenePtr scene;
    size_t workUsed = 0;
};
// Compile a complete USD data source into a zero-base snapshot, then rebuild
// and compare its exact supported wire projection. Input must remain immutable
// and alive throughout the call; const alone does not provide synchronization.
// Spec identities start at root 1 and are deterministic within this new seed;
// they are not IDs for inserting into an existing incarnation. Section order
// carries child membership; sample records carry animation. Explicit empty
// membership/sample maps and nonfinite sample times have no persistent encoding.
// Resolved asset caches are not authored wire data. Unsupported/unrepresentable
// inputs and quotas fail before any result or transport effect is published.
// Limits bound logical metadata/work and codec stages, not SDK caches or RSS.
SceneSeed compileSceneSeed(const PXR_NS::SdfAbstractData&, std::string_view sourceFormat = {},
    const SceneSeedLimits& = {});
} // namespace omni::usd::live
