// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <omni/usd/live/Live.h>
#include <pxr/usd/sdf/layer.h>
#include <functional>
#include <memory>
#include <string>

// MSVC requires a friend redeclaration of an exported function to repeat its DLL linkage; other compilers do not.
#if defined(_WIN32)
#    define OMNI_USD_LIVE_FRIEND_API OMNI_USD_LIVE_API
#else
#    define OMNI_USD_LIVE_FRIEND_API
#endif

// SDK-specific C++ host adapter. Consumers must use the plugin's exact USD,
// compiler/STL and build ABI. The portable C ABI remains in Live.h. Register the
// packaged USD resources and use the same canonical library for both APIs.
// USD type registration is process-lifetime: do not unload this library.
namespace omni::usd::live::host {
using PublicationDispatcher = std::function<void(std::function<void()>)>;
// True on the calling thread while the runtime is inside a PublicationDispatcher
// call for a layer's own publication job: the host-side step that installs an
// already admitted local intent before the runtime issues it. The runtime bounds
// those jobs to one outstanding per layer, so a dispatcher that meters host jobs
// may accept one without charging its quota. Flush jobs and peer publication
// jobs run outside this scope.
OMNI_USD_LIVE_API bool dispatchingLocalPublication() noexcept;

// Configure future opens. Both callbacks must be set or both empty. Empty clears
// configuration; it does not stop existing attachments or establish quiescence.
// Existing layers retain their original callbacks and captured host lifetime.
// FindOrOpen, CreateNew and every dispatched job must run on that same serial
// context. Flush jobs and publication jobs are both dispatched jobs under one rule:
// dispatch must enqueue without waiting, and a job must never run until the
// enclosing USD call, its synchronous notices and any external SdfChangeBlock have
// returned, including during nested event-loop pumps. Accepted jobs must run or
// release captures. Author a live layer only where isCurrent() returns true; the
// plugin assumes no particular thread, and the host defines the predicate (Kit:
// its main thread) and qualifies its writers against it.
// A mutation where isCurrent() is false, or from a handler for a notice that a
// publication job sent, is unsupported; the plugin detects it and moves
// the layer to FAILED. The host also synchronizes attached stages/Fabric;
// arbitrary bare Sdf access is unsafe. The executor must progress independently
// of a blocking save called where isCurrent() is false.
// SDK save-notice handlers must enqueue USD/host operations until the enclosing
// USD call returns. Capture identifiers or weak handles, never a strong owner
// of an initializing target; SDK notices may run under its layer registry lock.
OMNI_USD_LIVE_API OmniUsdLiveResult setPublicationContext(PublicationDispatcher dispatch,
    std::function<bool()> isCurrent) noexcept;

// Owned configuration for adapters that must establish host-callback quiescence.
// *output must be null. Both callbacks are required. Installation replaces the
// configuration for future opens; existing layers keep their original context.
class PublicationContext;
class Creation;
OMNI_USD_LIVE_API OmniUsdLiveResult createPublicationContext(PublicationDispatcher dispatch,
    std::function<bool()> isCurrent, PublicationContext** output) noexcept;
class PublicationContext final {
public:
    // Unpublishes this identity only. Does not wait for existing users; explicit
    // close is required before destroying host execution resources.
    OMNI_USD_LIVE_API ~PublicationContext();
    PublicationContext(const PublicationContext&) = delete;
    PublicationContext& operator=(const PublicationContext&) = delete;
    // Stops future opens using this identity, even when returning BUSY. Never
    // clears a replacement context. BUSY retains ownership for retry while
    // layers, creations or dispatched jobs still retain callbacks.
    // Keep the host executor progressing until OK. OK is idempotent and includes
    // capture destruction; no callbacks of this context can begin afterward.
    OMNI_USD_LIVE_API OmniUsdLiveResult close() noexcept;
    // Prepare only while this exact context is still configured for new work.
    // A replacement or close rejects without selecting another context.
    OMNI_USD_LIVE_API OmniUsdLiveResult prepareCreation(const PXR_NS::SdfLayer& seed, const char* liveUri,
        Creation** output) noexcept;
    // Same preparation, with bounded process-owned retention instead of a
    // caller-owned wrapper. See prepareRetainedCreation below.
    OMNI_USD_LIVE_API OmniUsdLiveResult prepareRetainedCreation(const PXR_NS::SdfLayer& seed, const char* liveUri,
        uint64_t* subjectId) noexcept;
private:
    struct Impl;
    explicit PublicationContext(Impl*) noexcept;
    Impl* const impl_;
    friend OMNI_USD_LIVE_FRIEND_API OmniUsdLiveResult createPublicationContext(PublicationDispatcher,
        std::function<bool()>, PublicationContext**) noexcept;
};

// Capture and complete this attachment's accepted work, without issuing
// create/replacement or reading mutable Sdf state where the host's isCurrent()
// predicate is false. Quiesce local authoring to use this as a save barrier;
// later admissions are outside the wait. Where isCurrent() holds the call polls
// (BUSY for pending work), even with a nonzero timeout. Elsewhere it waits within
// timeoutMs; zero polls. Timeout retains work.
// No callbacks are executed by the waiting thread. Failure/unknown work never
// reports success; inspect Live.h status. A FAILED layer is recovered with
// SdfLayer::Export followed by a fresh FindOrOpen.
OMNI_USD_LIVE_API OmniUsdLiveResult save(const PXR_NS::SdfLayer& liveLayer, uint32_t timeoutMs) noexcept;

enum class CreationPhase : uint32_t { Prepared, Creating, Confirmed, UnissuedFailure, Rejected, OutcomeUnknown };
struct CreationOutcome {
    CreationPhase phase = CreationPhase::Prepared;
    OmniUsdLiveResult result = OMNI_USD_LIVE_OK;
    OmniUsdLiveRequestError error{}; // Original finite request evidence, independent of diagnostics.
    uint64_t subjectId = 0; // Same layer status subject through bootstrap and activation.
    bool started = false, closing = false, quiescent = false;
};
class Creation;
// Freeze the complete seed on the configured serial host, outside change
// blocks. Caller pins and synchronizes the source through this call; afterward
// source mutation/destruction cannot change the prepared creation. An attached
// live layer is a valid seed and keeps its source format. *output must be null
// and is unchanged on error. Success reserves ownership/status but sends no RPC.
OMNI_USD_LIVE_API OmniUsdLiveResult prepareCreation(const PXR_NS::SdfLayer& seed, const char* liveUri,
    Creation** output) noexcept;

// Reserve one of 64 process-wide retained workflow slots BEFORE compiling the
// seed; sends no RPC. Success returns a stable subject ID and retains the owner
// independently of configuration/context lifetime. Output is unchanged on error.
// Implicit USD creation hooks use the same slots until activation succeeds; a
// host job after the outer USD call then hands the attachment to the layer and
// frees the slot. Failed or uncertain hooks stay retained. No slot is
// evicted. Acquire transfers ownership to a Creation before explicit start;
// enumeration finds the workflow again if subject delivery or caller ownership is lost.
OMNI_USD_LIVE_API OmniUsdLiveResult prepareRetainedCreation(const PXR_NS::SdfLayer& seed, const char* liveUri,
    uint64_t* subjectId) noexcept;
struct RetainedCreationInfo {
    uint64_t subjectId = 0;
    std::string liveUri;
};
// Thread-neutral ordered successor query; pass zero to begin. NOT_FOUND ends
// the scan and leaves output unchanged. This observes, not pins, a workflow;
// concurrent transfers can remove a result. Leased/retiring entries are visible.
OMNI_USD_LIVE_API OmniUsdLiveResult nextRetainedCreation(uint64_t afterSubject, RetainedCreationInfo* output) noexcept;
// Original host only, even after its context is replaced/destroyed. *output must
// be null. Transfers the sole retained workflow to a newly allocated wrapper;
// allocation failure, stale subject or BUSY leaves retention/output unchanged.
// A hook/finalization lease prevents acquisition until that operation returns.
OMNI_USD_LIVE_API OmniUsdLiveResult acquireRetainedCreation(uint64_t subjectId, Creation** output) noexcept;
// Explicit destructive cleanup on the original host. Export any needed content
// with SdfLayer::Export first: even BUSY begins cancellation. BUSY retains the
// slot for retry; OK removes it after native/USD owner destruction. Never waits
// for queued host work. Missing subjects return NOT_FOUND; no context rebinding.
OMNI_USD_LIVE_API OmniUsdLiveResult releaseRetainedCreation(uint64_t subjectId) noexcept;
class Creation final {
public:
    // Keep this owner for the entire created layer lifetime. Destruction
    // closes the workflow even if a copied layer handle survives. Destroy on
    // the original serial host, with no calls active.
    OMNI_USD_LIVE_API ~Creation();
    Creation(const Creation&) = delete;
    Creation& operator=(const Creation&) = delete;
    // Explicitly authorizes CreateLayer, replacing an existing destination.
    // One attempt only: repeated start returns ALREADY_EXISTS, never a retry.
    // The created layer is read-only until activation succeeds; USD refuses a
    // host edit of it before the edit reaches the layer's backing. A
    // replacement of the prepared host layer consumes the attempt before start,
    // and getOutcome then reports an unissued FAILED outcome.
    // OK means submitted, not confirmed. Runs on the original serial host.
    OMNI_USD_LIVE_API OmniUsdLiveResult start() noexcept;
    // Thread-neutral with this owner pinned. OK means finite request captures
    // retired; inspect getOutcome for execution result. Prepared returns BUSY.
    // Timeout does not cancel, retry, or discard work. No host jobs run here.
    OMNI_USD_LIVE_API OmniUsdLiveResult wait(uint32_t timeoutMs) noexcept;
    OMNI_USD_LIVE_API OmniUsdLiveResult getOutcome(CreationOutcome* output) const noexcept;
    // Original serial host only. Confirmed creation is required. Opens once at
    // the confirmed incarnation, then waits for native bootstrap without host
    // publication. Timeout retains that attachment and permits another wait.
    OMNI_USD_LIVE_API OmniUsdLiveResult prepareLayer(uint32_t timeoutMs) noexcept;
    // Install the confirmed content and canonical identifier, grant edit
    // permission and start live publication on the original serial host.
    // Failure retains seed, receipt and actual host; the host content stays
    // exportable with SdfLayer::Export. For an acquired CreateNew workflow whose
    // finalizer was dropped, this also installs its still-live target and
    // returns the original failure. Call only after the outer USD call, never
    // from an SDK notice.
    OMNI_USD_LIVE_API OmniUsdLiveResult activate() noexcept;
    // Original host only. BUSY before successful activation; output unchanged
    // on error. Returned references do not transfer workflow ownership.
    OMNI_USD_LIVE_API OmniUsdLiveResult getLayer(PXR_NS::SdfLayerRefPtr* output) const noexcept;
    // Original host only, poll-only cancellation/quiescence. BUSY means let
    // queued work retire and retry. Export any needed content with
    // SdfLayer::Export before close. Retains provenance until destruction.
    OMNI_USD_LIVE_API OmniUsdLiveResult close() noexcept;
    // After close begins, wait for finite and actor cleanup without releasing
    // USD ownership. Off-host uses one deadline; the original host and callback
    // workers only poll (BUSY). Call close on the host again to retire this owner.
    // Before close begins returns INVALID_ARGUMENT; timeout preserves ownership.
    OMNI_USD_LIVE_API OmniUsdLiveResult waitForClose(uint32_t timeoutMs) noexcept;
private:
    struct Impl;
    explicit Creation(Impl*) noexcept;
    Impl* const impl_;
    friend OMNI_USD_LIVE_FRIEND_API OmniUsdLiveResult prepareCreation(const PXR_NS::SdfLayer&, const char*,
        Creation**) noexcept;
    friend OMNI_USD_LIVE_FRIEND_API OmniUsdLiveResult acquireRetainedCreation(uint64_t, Creation**) noexcept;
    friend class PublicationContext;
    static OmniUsdLiveResult prepare(const PXR_NS::SdfLayer&, const char*, Creation**,
        std::shared_ptr<const void>) noexcept;
};
} // namespace omni::usd::live::host
