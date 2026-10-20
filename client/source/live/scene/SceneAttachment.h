// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "SceneJournal.h"
#include "LiveLayerActor.h"

namespace omni::usd::live {
// Validate framing, all values and the service-declared semantic range. Graph
// interpretation is staged in order by SceneJournal before applied advances.
// Readiness is separate: only the first complete response of an attempt is a
// boundary, under the service's first-response-is-current contract. Replay-only
// services must provide an explicit caught-up marker instead of that contract.
LayerActor::Decoded decodeSceneCoverage(std::string_view bytes, uint64_t sequence, bool firstResponse,
    const WireSceneLimits& = {});

// Marks the calling thread while the runtime is inside the publication
// dispatcher for a layer's own publication job; host::dispatchingLocalPublication()
// reports it. Scopes nest and restore the enclosing value.
class LocalPublicationScope final {
public:
    LocalPublicationScope() noexcept;
    ~LocalPublicationScope();
    LocalPublicationScope(const LocalPublicationScope&) = delete;
    LocalPublicationScope& operator=(const LocalPublicationScope&) = delete;
    static bool active() noexcept;
private:
    bool previous_;
};

// Private scene attachment. Owns canonical admission, native transport and
// immutable scene/intent publication. The live file format binds it to an
// SdfLayer; host edits, Save and the creation hooks reach it through that binding.
class SceneAttachment final {
public:
    struct Publication {
        // A bounded host queue. Enqueue failure throws before acceptance;
        // accepted jobs must run or release their complete captured owner.
        // The host must progress independently of blocking open/drain/close
        // callers, and invoke each job outside any enclosing SdfChangeBlock.
        std::function<void(std::function<void()>)> dispatch;
        // Runs in the host job, or on the opening caller during activation.
        // Receives the exact committed ordered-application owner or the
        // captured local-publication snapshot containing accepted work.
        // Return only after USD publication and synchronous notices finish.
        // A post-journal failure retains the actor work and candidate; it
        // cannot claim to roll back USD mutations or accepted local intent.
        std::function<void(SceneJournal::SnapshotPtr)> publish;
        std::shared_ptr<const std::atomic<OmniUsdLiveResult>> hostFailure;
        // The host edit buffer's unadmitted record count; see LayerActor::Options.
        std::shared_ptr<BufferedEditCount> bufferedEdits;
    };
    // Publication covers accepted local state before transport issuance and
    // ordered peer/ACK application. Admission remains distinct from host
    // publication, ACK and durability. No bare SdfLayer/UsdStage concurrency
    // guarantee is implied by enabling this private adapter.
    struct Options {
        std::string session, layer;
        uint64_t object = 0, resume = 0;
        WireScenePtr initial;
        SceneJournal::Limits limits;
        std::shared_ptr<LayerAdmission> admission;
        std::shared_ptr<LayerStatus> status;
        bool holdOpening = false; // Block READY/authoring until canonical host publication.
        bool discover = false;
        Publication publication;
        // Fresh initial read only: discovery or a confirmed nonzero object,
        // with resume zero and no preloaded authoritative scene.
        bool deferInitialPublication = false;
    };
    using Changed = std::function<void(const LayerProgress::Snapshot&)>;
    static std::shared_ptr<SceneAttachment> open(std::shared_ptr<Runtime>, OmniUsdLiveEndpoint, Options, Changed = {});
    // Reserve the canonical resource before issuing one object-zero read.
    // Return only after complete validated initial content has been applied;
    // errors never masquerade as empty layers. Initial-open deadline bounds
    // the wait; cleanup retains admission until workers quiesce. Worker calls
    // return BUSY. Changed starts after successful initial readiness and may
    // run before this function returns; failed opens never start host delivery.
    static std::shared_ptr<SceneAttachment> openExisting(std::shared_ptr<Runtime>, const std::string& uri,
        SceneJournal::Limits = {}, Changed = {}, Publication = {});
    // Validate initial content without posting bootstrap to a possibly blocked
    // host caller. Returns prepared, NOT ready: original actor work/drain remain
    // pending. Call activatePublication in the opening USD context, outside an
    // enclosing change block, before authoring. Later jobs use dispatch.
    static std::shared_ptr<SceneAttachment> prepareExisting(std::shared_ptr<Runtime>, const std::string& uri,
        Publication, SceneJournal::Limits = {}, Changed = {});
    // Wait for a deferred native bootstrap candidate, without running host
    // callbacks or advancing the actor's applied frontier. Caller owns this
    // attachment before waiting, including on timeout/failure. Timeout never
    // closes, reopens or retries creation. Its initial-open deadline remains
    // in force. Worker calls return BUSY; zero timeout polls.
    OmniUsdLiveResult waitForPrepared(uint32_t timeoutMs) noexcept;
    OmniUsdLiveResult activatePublication() noexcept;
    // Test seam: the initial-open deadline (publication activation and the
    // actor's initial application) counts as passed from this call.
    void expireActivationForTest() noexcept;
    // Test seam: the current read attempt's first-response deadline counts as
    // passed (LayerActor::expireReadForTest).
    void expireReadForTest() noexcept;
    // Test seam: a synchronous open calls hook(context) on its calling thread
    // right before it waits for the bootstrap, after every allocation of its
    // setup; null disables it. The hook must not allocate or throw.
    static void setBeforeOpenWaitForTest(void (*hook)(void*) noexcept, void* context) noexcept;
    ~SceneAttachment();
    uint64_t submitLocal(std::string_view bytes);
    // Admit a host-captured edit log compiled against `base`; see
    // SceneJournal::submitCaptured. Call at most once per captured log. An
    // update the runtime or journal limits reject however long it waits
    // throws CapturedRejection; an unsupported compile throws
    // UnsupportedOperation, and an update over the service's entry limits
    // throws DeltaEntryLimitExceeded.
    uint64_t submitCaptured(const WireScenePtr& base, const std::shared_ptr<const SdfOperationLog>& sealed,
        WireScenePtr* applied = nullptr);
    // Checks, without admitting it, whether submitCaptured of this log would
    // fail a limit however long it waits, throwing as submitCaptured does.
    void probeCaptured(const WireScenePtr& base, const std::shared_ptr<const SdfOperationLog>& sealed) const;
    // The raw log operation limit submitCaptured and probeCaptured compile
    // with: the journal's configured authoring limit, which no scene grows.
    size_t maxLogOperations() const noexcept { return journal_->limits().authoring.maxLogOperations; }
    // Whether admitted writes already fill the issuance window, so a flush
    // defers and its records coalesce with later host turns (design section 8).
    bool issuanceSaturated() const noexcept { return actor_->issuanceSaturated(); }
    // Test seam: the layer's chained-write issuance window.
    uint32_t chainWindowForTest() const noexcept { return actor_->chainWindowForTest(); }
    // Test seams: whether issuance holds for a re-sent unchained write's first
    // attempt with no write in flight, and expiring that hold.
    bool resendHeldForTest() const noexcept { return actor_->resendHeldForTest(); }
    void expireResendHoldForTest() noexcept { actor_->expireResendHoldForTest(); }
    // Thread-safe. Records the host turns one admitted update holds.
    void noteCoalescedTurns(uint64_t turns) noexcept { actor_->noteCoalescedTurns(turns); }
    // A charge of the host edit buffer to the endpoint's shared host buffer
    // budget; null once the runtime or the endpoint closes.
    std::unique_ptr<HostBufferCharge> hostBufferCharge() const { return actor_->hostBufferCharge(); }
    // The endpoint's shared host buffer budget, read without a charge.
    HostBufferCharge::Budget hostBufferBudget() const noexcept { return actor_->hostBufferBudget(); }
    SceneJournal::SnapshotPtr snapshot() const { return journal_->snapshot(); }
    LayerProgress::Snapshot progress() const { return actor_->snapshot(); }
    LayerActor::TransportSnapshot transportSnapshot() const { return actor_->transportSnapshot(); }
    uint64_t subjectId() const noexcept { return actor_->subjectId(); }
    OmniUsdLiveResult completeOpening() noexcept { return actor_->completeOpening(); }
    void failPublication(OmniUsdLiveResult result) noexcept;
    // Thread-safe notification of a latched host failure. Host boundaries
    // call it, and so does the file format's record sink when it rejects a
    // record made where the host predicate does not hold or inside the
    // layer's own publication notice; that caller touches no USD or
    // host-confined state.
    OmniUsdLiveResult observeHostFailure() noexcept { return actor_->observeHostFailure(); }
    // Thread-safe. The host edit buffer's count changed; status republishes it.
    void noteHostEdits() noexcept { actor_->noteHostEdits(); }
    // Thread-safe. Records flush_unsupported for a flush that failed the layer.
    void reportFlushUnsupported(UnsupportedOperation::Reason reason) noexcept {
        actor_->reportFlushUnsupported(static_cast<uint32_t>(reason));
    }
    std::vector<LayerProgress::Work> retained() const { return actor_->retained(); }
    // Captures this layer's existing work only. A host whose publication queue
    // is required for completion must poll (allowBlock=false); bounded waiting
    // belongs on an independently progressing caller before host Save.
    OmniUsdLiveResult save(uint32_t timeoutMs, bool allowBlock = true);
    OmniUsdLiveResult pollSave() noexcept { return actor_->pollSave(); }
    struct PublicationProgress {
        bool configured = false, hasAttempt = false, hasCompleted = false;
        uint64_t lastAttemptedRevision = 0, lastCompletedRevision = 0;
        // Current coordinator result, not per-RPC outcome evidence. Revisions
        // record host callback entry and successful return only: a callback
        // may mutate USD then throw, or return successfully before activation
        // fails its deadline. These fields do not establish actual host state.
        OmniUsdLiveResult result = OMNI_USD_LIVE_OK;
    };
    // Thread-safe. The host publication coordinator's current progress; a
    // layer without host publication reports configured=false. It samples
    // the coordinator only and never waits for an in-flight callback: a
    // sample taken during a callback reports that attempt with hasCompleted
    // and lastCompletedRevision unchanged.
    PublicationProgress publicationProgress() const;
    // No journal guard is held while waiting. A close preserves accepted work
    // until this owner and any snapshot owners have been released. `released`
    // states that the caller drops its last owner too (a USD layer's binding
    // on layer destruction, or the synchronous caller of a failed open, which
    // never publishes any owner): an open of the resource may then wait for
    // the canonical key instead of refusing at once.
    OmniUsdLiveResult close(uint32_t timeoutMs, bool released = false);
private:
    struct PublicationState;
    static std::shared_ptr<SceneAttachment> openExistingImpl(std::shared_ptr<Runtime>, const std::string&,
        SceneJournal::Limits, Changed, Publication, bool);
    void requireAuthoring() const;
    explicit SceneAttachment(std::shared_ptr<SceneJournal> journal) : journal_(std::move(journal)) {}
    const std::shared_ptr<SceneJournal> journal_;
    std::shared_ptr<LayerActor> actor_;
    std::shared_ptr<PublicationState> publication_;
};
} // namespace omni::usd::live
