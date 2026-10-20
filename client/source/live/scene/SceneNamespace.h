// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "WireScene.h"
#include <functional>

namespace omni::usd::live {
struct NamespaceSpec {
    uint64_t id = 0;
    PXR_NS::SdfSpecType type = PXR_NS::SdfSpecTypeUnknown;
    PXR_NS::SdfPath path;
};
struct SceneNamespaceLimits {
    WireSceneLimits scene;
    size_t maxCommands = 65536;
    size_t maxMetadataBytes = 8u * 1024 * 1024;
    size_t maxWork = 512u * 1024 * 1024;
};
struct NamespacePlan {
    std::vector<wire::Command> commands;
    WireScenePtr candidate;
    size_t workUsed = 0;
};
// Reverse USD child policies. Callers must bound/validate paths first; mapper
// paths must already have canonical absolute targets (including nested targets).
PXR_NS::SdfPath namespaceParentPath(PXR_NS::SdfSpecType child, const PXR_NS::SdfPath&);
unsigned namespaceChildList(PXR_NS::SdfSpecType parent, PXR_NS::SdfSpecType child);
PXR_NS::TfToken namespaceChildField(PXR_NS::SdfSpecType parent, unsigned list);
PXR_NS::VtValue namespaceChildKey(PXR_NS::SdfSpecType child, const PXR_NS::SdfPath&);

// Complete desired identity/type/path graph, including {1,PseudoRoot,/}.
// Vector order is not a request to reorder siblings. Surviving IDs and fields
// remain attached to their identities; derived child fields follow structure.
// Only ordinary Create/Delete/Move commands are emitted. Type changes require
// explicit new identities. Scratch allocation is needed only to break blocked
// moves; its callback must supply globally fresh IDs >1. All base, desired and
// already allocated IDs are checked locally, including deleted IDs.
// Every intermediate and complete batch is validated against WireScene. Work
// charges include these repeated candidate builds; errors preserve the base.
// An optional exclusively owned shared budget is charged even on failure.
// This is standalone candidate planning, not concurrent namespace admission:
// recursive service moves/deletes can affect peer descendants absent from base.
NamespacePlan planSceneNamespace(const WireScenePtr& base, const std::vector<NamespaceSpec>& desired,
    const std::function<uint64_t()>& allocateScratch = {}, const SceneNamespaceLimits& limits = {},
    WireSceneWorkBudget* shared = nullptr);
// The desired graph as changes to the base's visible graph. Every visible base
// identity outside `placed` and `removed` keeps its type and path.
struct NamespaceChanges {
    // The desired type and path of each identity the edit creates or moves.
    std::vector<NamespaceSpec> placed;
    // Visible base identities absent from the desired graph.
    std::vector<uint64_t> removed;
};
// The plan planSceneNamespace returns for the complete desired graph that
// `changes` describes: the same commands, in the same order, and the same
// candidate. Each step reads and updates only the identities the changes
// name, the children of the base paths they vacate or replace, and the
// sections each command changes, so planning costs work in proportion to the
// edit, with a logarithmic factor in the scene. Validation covers the placed
// identities and the base children whose parent identity changes; the
// visible base graph is the service's and stays as it is.
NamespacePlan planSceneNamespaceChanges(const WireScenePtr& base, const NamespaceChanges& changes,
    const std::function<uint64_t()>& allocateScratch = {}, const SceneNamespaceLimits& limits = {},
    WireSceneWorkBudget* shared = nullptr);
} // namespace omni::usd::live
