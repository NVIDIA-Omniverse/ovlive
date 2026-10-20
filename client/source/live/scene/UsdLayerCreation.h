// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "SceneCreation.h"
#include "SceneAttachment.h"
#include <pxr/usd/sdf/layer.h>
#include <string_view>

namespace omni::usd::live {
// Private creation-to-host bridge behind host::Creation and the Export and
// CreateNew hooks. Every operation and final destruction must run on the
// captured serial USD host, outside enclosing SdfChangeBlocks. The owner must
// remain alive across calls/notices. It never retries Create. Explicit close
// or wrapper destruction retires the finite creation and actor even if a
// layer handle survives. The finite creation and actor share one public typed
// status subject; READY and authoring stay blocked until canonical host
// publication succeeds.
class UsdLayerCreation final {
public:
    using Limits = SceneJournal::Limits;
    // Reserve canonical admission and construct a private anonymous USD layer
    // before any mutation can issue. Caller explicitly calls start() after
    // this returns. Throws typed results; the seed is never installed as if it
    // were confirmed applied coverage.
    static std::shared_ptr<UsdLayerCreation> prepare(std::shared_ptr<Runtime>, OmniUsdLiveEndpoint, std::string session,
        std::string layer, std::string seedBytes, Limits = {});
    // Capture the configured host before accessing source data; compile exact
    // backing (without schema fallback fields) and resolve the registered URI.
    // With an admission deadline (the Export hook's), a closing backing of the
    // destination is waited for until then.
    static std::shared_ptr<UsdLayerCreation> prepare(std::shared_ptr<Runtime>, const PXR_NS::SdfLayer& source,
        const std::string& uri, Limits = {}, std::shared_ptr<const void> expectedContext = {},
        std::optional<std::chrono::steady_clock::time_point> admissionDeadline = {});
    // CreateNew borrows its canonical SDK target. No extra USD layer or strong
    // target reference is created; the workflow owns the attachment until
    // adoptIntoLayer().
    static std::shared_ptr<UsdLayerCreation> prepareInitializing(std::shared_ptr<Runtime>, PXR_NS::SdfLayer&,
        Limits = {});
    // Hook-only: prepare and install bootstrap while borrowing the target;
    // keep authoring/READY gated until the outer USD call has returned.
    OmniUsdLiveResult prepareInitializingAttachment(PXR_NS::SdfLayer&, uint32_t) noexcept;
    // Original-host queued continuation, strictly after the outer USD call.
    // A missing weak target fails opening; the workflow keeps its attachment
    // open until close or adoptIntoLayer().
    OmniUsdLiveResult finalizeInitializing(OmniUsdLiveResult completion = OMNI_USD_LIVE_OK) noexcept;
    ~UsdLayerCreation();
    UsdLayerCreation(const UsdLayerCreation&) = delete;
    UsdLayerCreation& operator=(const UsdLayerCreation&) = delete;
    // Observe the exact owned host target before consuming the finite attempt.
    // A host failure observed at a notice boundary may already have consumed it;
    // repeated/consumed start returns ALREADY_EXISTS, with result in the receipt.
    OmniUsdLiveResult start() noexcept;
    // Hook-only borrowing variant: never retains an unfinished target.
    OmniUsdLiveResult startInitializing(PXR_NS::SdfLayer&) noexcept;
    const std::shared_ptr<SceneCreation>& creation() const;
    std::shared_ptr<SceneAttachment> attachment() const;
    // Open once, only after confirmed creation, using its object and cursor
    // zero. Own the attachment before waiting. Timeout permits another wait;
    // initialization failure never permits a second open or loses provenance.
    OmniUsdLiveResult prepareAttachment(uint32_t timeoutMs) noexcept;
    // An acquired initializing-target workflow may finish its post-outer-call
    // transition here, including after a dropped finalization job.
    // Install bootstrap on the anonymous layer, then publish its canonical
    // identifier with USD's collision-checked registry update. Identifier
    // publication can query resolver metadata. Failure retains even a partly
    // installed/renamed layer and never retries creation or identifier setup.
    OmniUsdLiveResult activate() noexcept;
    // Null before successful host activation AND identifier publication. Subsequent failure does not
    // discard the installed layer; close preserves all retained owners.
    PXR_NS::SdfLayerRefPtr layer() const;
    OmniUsdLiveResult close() noexcept;
    // Original host, after successful activation and outside every workflow
    // call: hand the attachment to the layer binding without closing it and
    // drop the workflow's layer reference. The binding closes the attachment
    // when the last layer reference goes; the workflow keeps only its finite
    // creation, whose later close leaves the shared status subject unchanged.
    // Any other state returns its reason and changes nothing.
    OmniUsdLiveResult adoptIntoLayer() noexcept;
    // Thread-neutral after close() begins. No USD access/owner destruction;
    // wait on finite and actor retirement, polling on the original host.
    OmniUsdLiveResult waitForClose(uint32_t timeoutMs) noexcept;
    // Registry ownership transfer/cleanup requires the original serial host
    // and no reentrant workflow operation. Does not consult current config.
    void requireTransferHost() const;
    // Retain a bool-hook failure without closing the workflow; the host can
    // still Export the layer before close. Pending create settles normally;
    // confirmation cannot resume abandoned bootstrap. Finite rejected/unknown
    // results retain their own cause and raw transport error.
    OmniUsdLiveResult failHook(OmniUsdLiveResult) noexcept;
private:
    struct State;
    static std::shared_ptr<UsdLayerCreation> prepareSource(std::shared_ptr<Runtime>, const PXR_NS::SdfLayer*,
        const std::string&, Limits, std::shared_ptr<const void>, PXR_NS::SdfLayer*,
        std::optional<std::chrono::steady_clock::time_point> = {});
    OmniUsdLiveResult startOn(PXR_NS::SdfLayer*) noexcept;
    OmniUsdLiveResult prepareAttachmentOn(PXR_NS::SdfLayer*, uint32_t) noexcept;
    static std::shared_ptr<UsdLayerCreation> prepareOwned(std::unique_ptr<State>, std::shared_ptr<Runtime>,
        OmniUsdLiveEndpoint, std::string session, std::string layer, std::string seedBytes, Limits,
        PXR_NS::SdfLayer* = nullptr, std::optional<std::chrono::steady_clock::time_point> = {});
    explicit UsdLayerCreation(std::unique_ptr<State>);
    std::unique_ptr<State> state_;
};
} // namespace omni::usd::live
