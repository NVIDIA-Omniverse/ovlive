// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "LiveSdfFileFormat.h"
#include "UsdLayerCreation.h"
#include "UsdCreationRegistry.h"
#include "UsdLayerBinding.h"
#include "UsdCreationWorkflow.h"
#include "HostLayerState.h"
#include "FieldEditIntent.h"
#include "SceneSeed.h"
#include "LiveRuntime.h"
#include "LiveDispatch.h"
#include "LiveUri.h"
#include "LiveMetadata.h"
#include <pxr/base/tf/diagnostic.h>
#include <pxr/base/tf/enum.h>
#include <pxr/base/tf/notice.h>
#include <pxr/base/tf/errorMark.h>
#include <pxr/usd/sdf/changeBlock.h>
#include <pxr/usd/sdf/layer.h>
#include <pxr/usd/sdf/layerStateDelegate.h>
#include <pxr/usd/sdf/notice.h>
#include <pxr/usd/sdf/schema.h>
#include <optional>
#include <atomic>
#include <algorithm>
#include <iterator>
#include <limits>
#include <new>
#if defined(__SANITIZE_ADDRESS__)
#    define LIVE_LSAN 1
#elif defined(__has_feature)
#    if __has_feature(address_sanitizer)
#        define LIVE_LSAN 1
#    endif
#endif
#if defined(LIVE_LSAN)
#    include <sanitizer/lsan_interface.h>
#endif

// Private to the live file format (LiveSdfFileFormat.cpp) and its creation
// bridge (UsdLayerCreation.cpp): the live state of the layers the format
// instantiates, the host context both files share and CreateNew's queued
// finalizer.
namespace omni::usd::live::fileformat {
struct Host {
    live::UsdPublicationDispatcher dispatch;
    std::function<bool()> isCurrent;
};
// The host context that layers opened or created after this call capture.
std::shared_ptr<const Host> configuredHost();
// A dispatcher that retains `context` until every accepted job retires.
UsdPublicationDispatcher retainedDispatch(const std::shared_ptr<const Host>& context);
// The test progress observer that attachments opened after this call receive.
SceneAttachment::Changed progressObserver();
// The clock of Export and CreateNew creation deadlines.
std::chrono::steady_clock::time_point creationNow();
// Milliseconds left until `deadline` at `now`, rounded up and clamped to
// uint32_t; zero once the deadline passed.
inline uint32_t millisecondsLeft(std::chrono::steady_clock::time_point deadline,
    std::chrono::steady_clock::time_point now) noexcept {
    const auto duration = deadline - now;
    if (duration <= std::chrono::steady_clock::duration::zero()) return 0;
    return uint32_t(std::min<int64_t>(std::chrono::ceil<std::chrono::milliseconds>(duration).count(),
        std::numeric_limits<uint32_t>::max()));
}
// The opening deadline of an Export or CreateNew hook, on the creation clock.
struct CreationDeadline {
    std::chrono::steady_clock::time_point at;
    // Milliseconds left; throws TIMEOUT once the deadline passed.
    uint32_t remaining() const {
        const auto left = millisecondsLeft(at, creationNow());
        if (left == 0) throw OmniUsdLiveResult(OMNI_USD_LIVE_TIMEOUT);
        return left;
    }
};
// The result of the exception being handled, including layer rejections and
// wire decode failures. Call only inside a catch handler.
OmniUsdLiveResult creationException() noexcept;
// Posts a typed USD diagnostic for a failed file-format operation.
void report(OmniUsdLiveResult result) noexcept;
// Whether a live layer is dirty. USD may skip SaveToFile when this says clean.
// Its local clean bit cannot prove that asynchronous accepted intent has been
// confirmed, and recorded host edits await the flush that SaveToFile performs.
inline bool liveLayerDirty(const std::weak_ptr<live::UsdLayerBinding>& weak) noexcept {
    try {
        const auto binding = weak.lock();
        if (!binding || binding->hasPendingLocalEdits()) return true;
        const auto attachment = binding->attachment();
        return !attachment || attachment->pollSave() != OMNI_USD_LIVE_OK;
    } catch (...) { return true; }
}
// The layer's only state delegate, through which USD asks whether it is dirty.
inline void installDelegate(PXR_NS::SdfLayer& layer, const PXR_NS::SdfLayerStateDelegateBaseRefPtr& delegate) {
    layer.SetStateDelegate(delegate);
}
// A layer's flush job requests; shared with off-host flush requests.
struct FlushRequest {
    std::atomic<bool> queued{false}; // A flush job is queued.
    // The dispatcher refused the last flush job for capacity, and no flush
    // attempt and no accepted dispatch has followed.
    std::atomic<bool> refused{false};
};
// A live layer is a plain SdfLayer that USD's base factory
// (SdfFileFormat::_InstantiateNewLayer) constructs, so it needs no SdfLayer
// constructor, which Kit's Windows USD does not export. Its live state is this,
// the layer's state delegate: adopted once, owned only by the layer, and
// destroyed inside the layer's destruction, before the layer's TfRefBase and
// memory go.
class BoundLayer final : public PXR_NS::SdfSimpleLayerStateDelegate {
public:
    using DataAccessor = PXR_NS::SdfAbstractDataConstPtr (*)(const PXR_NS::SdfLayer&);
    using DataInstaller = void (*)(PXR_NS::SdfLayer&, const PXR_NS::TfRefPtr<live::HostSceneProjection>&);
    static BoundLayer* from(const PXR_NS::SdfLayer* layer) noexcept {
        if (!layer) return nullptr;
        try {
            auto* bound = dynamic_cast<BoundLayer*>(get_pointer(layer->GetStateDelegate()));
            return bound && bound->layer_.load() == layer ? bound : nullptr;
        } catch (...) { return nullptr; }
    }
    // The live state of a layer the base factory is about to construct. Every
    // fallible allocation happens here, before the layer exists, including the
    // weak remnant that GetStateDelegate hands out.
    static PXR_NS::TfRefPtr<BoundLayer> create(const PXR_NS::SdfFileFormatConstPtr& format, DataAccessor accessor,
        DataInstaller installer) {
        auto state = PXR_NS::TfCreateRefPtr(new BoundLayer(format, accessor, installer));
        const PXR_NS::TfWeakPtr<BoundLayer> remnant(state);
        (void)remnant;
        return state;
    }
    // Becomes `layer`'s state delegate, then binds the layer. Installing the
    // delegate first lets from() find a layer whose setup fails.
    void adopt(PXR_NS::SdfLayer& layer, const PXR_NS::TfRefPtr<BoundLayer>& self) noexcept {
        layer_.store(&layer);
        try {
            installDelegate(layer, self);
            bindState();
        } catch (const std::bad_alloc&) { setupResult = OMNI_USD_LIVE_RESOURCE_EXHAUSTED; } catch (...) {
            setupResult = OMNI_USD_LIVE_INTERNAL_ERROR;
        }
        if (setupResult != OMNI_USD_LIVE_OK && hostState) hostState->fault()->fail(setupResult);
    }
    ~BoundLayer() override { unbind(false); }
    PXR_NS::SdfLayer& layer() const noexcept { return *layer_.load(); }
    // InitData prewarms weak data bookkeeping. Adoption only retains the exact
    // initial owner, once; a later replacement can never reset it.
    void bindState() {
        const auto data = accessor(layer());
        const auto* initial = dynamic_cast<const live::HostSceneProjection*>(data.operator->());
        if (initial) {
            hostState.emplace(
                PXR_NS::TfRefPtr<live::HostSceneProjection>(const_cast<live::HostSceneProjection*>(initial)));
            armProjection(*hostState->authorized());
        } else setupResult = OMNI_USD_LIVE_UNSUPPORTED;
        binding->bind(&layer());
    }
    PXR_NS::SdfAbstractDataConstRefPtr currentData() const {
        return PXR_NS::SdfAbstractDataConstRefPtr(accessor(layer()).operator->());
    }
    void ensureSetup() {
        if (setupResult != OMNI_USD_LIVE_OK) throw setupResult;
        if (!notice) {
            try {
#if defined(LIVE_LSAN)
                // TfNotice::Register hands a new deliverer to Tf's notice
                // registry, which loses it when one of the registry's own
                // allocations throws before the deliverer is linked (a USD
                // ownership gap). LeakSanitizer ignores the allocations made
                // inside this USD call and treats them as roots, so a lost
                // deliverer is not reported and everything it references
                // stays reachable: its weak pointers reference this live state,
                // so a product leak of this state and what it owns on this
                // fault path is hidden too. The ASan scene lane's zero-report run of
                // the allocation sweep is what shows the layer is freed.
                const __lsan::ScopedDisabler usdOwned;
#endif
                notice = PXR_NS::TfNotice::Register(PXR_NS::TfCreateWeakPtr(this), &BoundLayer::changed,
                    PXR_NS::SdfLayerHandle(&layer()));
            } catch (const std::bad_alloc&) { setupResult = OMNI_USD_LIVE_RESOURCE_EXHAUSTED; } catch (...) {
                setupResult = OMNI_USD_LIVE_INTERNAL_ERROR;
            }
            if (!notice && setupResult == OMNI_USD_LIVE_OK) setupResult = OMNI_USD_LIVE_INTERNAL_ERROR;
            if (setupResult != OMNI_USD_LIVE_OK) {
                hostState->fault()->fail(setupResult);
                throw setupResult;
            }
        }
        observe();
    }
    void observe() const noexcept {
        if (!hostState) return;
        // An allocation failure latches RESOURCE_EXHAUSTED, as ensureSetup's does, and fails the layer (design
        // section 8).
        try {
            hostState->observe(currentData());
            armReplacement();
        } catch (const std::bad_alloc&) { hostState->fault()->fail(OMNI_USD_LIVE_RESOURCE_EXHAUSTED); } catch (...) {
            hostState->fault()->fail(OMNI_USD_LIVE_INTERNAL_ERROR);
        }
        if (hostState->fault()->result().load(std::memory_order_acquire) != OMNI_USD_LIVE_OK) {
            discardRecords();
            try {
                if (const auto attachment = binding->attachment()) (void)attachment->observeHostFailure();
                else if (creationWorkflow) {
                    // A prepared target is not the caller's original seed.
                    // Consume only an unissued finite attempt; a start winner
                    // keeps its real transport outcome and original seed.
                    (void)creationWorkflow->creation->failBeforeStart(
                        hostState->fault()->result().load(std::memory_order_acquire));
                }
            } catch (...) {} // Destructor/failed-setup paths have no attached actor.
        }
    }
    // Host-confined. A FAILED layer admits no record, and its host content
    // already holds every edit, so every projection that can back it (the
    // authorized one, an installation candidate and the current backing)
    // drops its records and retains no later one. The binding's buffered
    // count keeps growing (design section 13).
    void discardRecords() const noexcept {
        chargeBuffer(0);
        if (const auto& authorized = hostState->authorized()) authorized->discardLocalEdits();
        if (const auto& candidate = hostState->candidate()) candidate->discardLocalEdits();
        try {
            const auto data = currentData();
            if (const auto* backing = dynamic_cast<const live::HostSceneProjection*>(data.operator->()))
                const_cast<live::HostSceneProjection*>(backing)->discardLocalEdits();
        } catch (...) {}
    }
    enum class FlushOutcome { Idle, Admitted, Deferred, Failed };
    // Enqueue at most one flush job at a time; never touches the projection
    // itself. Thread-safe: an off-host save and an endpoint pending-byte
    // release also request flushes. A dispatcher refusal for capacity leaves
    // the buffer pending and the layer BLOCKED, and the next host boundary
    // ends the host turn (changed()); any other refusal fails the layer with
    // its typed cause, and the host content keeps the applied edit.
    // The job captures only the weak binding and pins the layer for its whole
    // body, as publication does; a layer already released or detached yields
    // no pin and the job returns without touching it.
    static void dispatchFlush(const std::shared_ptr<const Host>& host, const std::weak_ptr<live::UsdLayerBinding>& weak,
        const std::shared_ptr<FlushRequest>& request, const std::shared_ptr<live::HostFailureLatch>& fault,
        const std::shared_ptr<live::BufferedEditCount>& buffered) noexcept {
        if (!host || !request || request->queued.exchange(true)) return;
        try {
            live::DispatchExecutor::NonblockingScope nonblocking;
            // Cleared before the dispatch, so a refusal of this or a
            // concurrent dispatch is the last store and stays visible.
            request->refused.store(false);
            retainedDispatch(host)([weak, request] {
                const auto owner = weak.lock();
                if (!owner) return;
                PXR_NS::SdfLayerRefPtr pinned;
                try {
                    pinned = owner->pinLayer();
                } catch (...) { return; }
                if (!pinned) return;
                request->queued.store(false);
                if (auto* bound = BoundLayer::from(get_pointer(pinned))) bound->runFlushJob();
            });
        } catch (...) {
            const auto cause = creationException();
            // A dispatcher that refuses for capacity leaves the unissued buffer
            // pending (design section 8, transient class): the layer reports
            // BLOCKED and its next record, on-host or off-host save,
            // publication job or endpoint pending-byte release requests the
            // flush again, as does a mutation the buffer quota refuses.
            const bool deferred = cause == OMNI_USD_LIVE_RESOURCE_EXHAUSTED && buffered;
            // Refused is set before queued clears: a dispatch that then claims
            // the request clears it before its own attempt, so it never
            // outlives a dispatch that follows.
            if (deferred) request->refused.store(true);
            request->queued.store(false);
            if (deferred) {
                if (buffered->note(buffered->load(), cause)) {
                    try {
                        const auto owner = weak.lock();
                        const auto attachment = owner ? owner->attachment() : nullptr;
                        if (attachment) attachment->noteHostEdits();
                        else buffered->abandonWake();
                    } catch (...) { buffered->abandonWake(); }
                }
                return;
            }
            if (fault) fault->fail(cause);
        }
    }
    void scheduleFlush() noexcept {
        if (hostState) dispatchFlush(hostContext, binding, flushRequest, hostState->fault(), binding->bufferedEdits());
    }
    // Host-confined flush job body.
    void runFlushJob() noexcept {
        bool issued = false;
        const auto outcome = flushLocalEdits(&issued, true);
        try {
            installDeferredAfter(outcome, issued);
        } catch (...) {
            hostState->fault()->fail(creationException());
            observe();
        }
    }
    // Host-confined. Whether a publication job deferred installation and this
    // flush left the buffer empty without issuing an intent (an empty buffer,
    // or an empty or superseded envelope): no local publication follows, so
    // the deferred installation is the flusher's to run. A FAILED layer
    // installs nothing.
    bool deferredInstallDue(FlushOutcome outcome, bool issued) const noexcept {
        return installDeferred && !issued && (outcome == FlushOutcome::Idle || outcome == FlushOutcome::Admitted) &&
            failedLayerCause() == OMNI_USD_LIVE_OK;
    }
    // Host-confined, in a flush job only: installation runs outside any
    // enclosing change block, never inside a caller's SdfLayer::Save.
    void installDeferredAfter(FlushOutcome outcome, bool issued) {
        if (!deferredInstallDue(outcome, issued)) return;
        if (const auto attachment = binding->attachment()) install(attachment->snapshot()->visible);
    }
    // Whether a flush failure defers the host edit buffer instead of failing
    // the layer: BUSY before publication activates, CANCELLED while the
    // attachment closes, and a plain RESOURCE_EXHAUSTED. That last one is the
    // journal's pending-intent limit, a pending-byte, replay-byte or
    // replay-work charge that fails only beside other pending intents, the
    // endpoint's pending bytes (which other layers on the endpoint can hold),
    // or an allocation failure. A limit no wait relieves (one the intent
    // alone exceeds, or an exhausted journal revision or ordinal) arrives as
    // CapturedRejection and is definitive, as is every cause other than these
    // three results.
    static bool transientFlushFailure(OmniUsdLiveResult cause) noexcept {
        return cause == OMNI_USD_LIVE_BUSY || cause == OMNI_USD_LIVE_RESOURCE_EXHAUSTED ||
            cause == OMNI_USD_LIVE_CANCELLED;
    }
    // The cause of a layer that is already FAILED: the actor's terminal
    // result when its gate is terminal, else the host fault; OK while
    // issuance is still possible.
    static OmniUsdLiveResult failedLayerCause(const live::HostFailureLatch& fault,
        const live::SceneAttachment* attachment) noexcept {
        if (attachment) {
            try {
                const auto progress = attachment->progress();
                if (progress.gate == live::LayerProgress::Gate::Failed)
                    return progress.result != OMNI_USD_LIVE_OK ? progress.result :
                                                                 OmniUsdLiveResult(OMNI_USD_LIVE_INTERNAL_ERROR);
            } catch (...) { return OMNI_USD_LIVE_INTERNAL_ERROR; }
        }
        return fault.result().load(std::memory_order_acquire);
    }
    OmniUsdLiveResult failedLayerCause() const noexcept {
        std::shared_ptr<live::SceneAttachment> attachment;
        try {
            attachment = binding->attachment();
        } catch (...) {}
        return hostState ? failedLayerCause(*hostState->fault(), attachment.get()) :
                           OmniUsdLiveResult(OMNI_USD_LIVE_OK);
    }
    // Host-confined; never throws. Admit the authorized projection's host
    // edit buffer as one captured intent compiled against the base it was
    // applied to, then rebaseline the projection to that base with the
    // intent applied, which the compiler verified equals the host content.
    // The journal's visible scene is not used: it can include applied work
    // the layer has not yet installed. The buffer is read, never drained,
    // before admission; only the records one admission covers leave it.
    // - Failed: the buffer holds records and the layer is already FAILED
    //   (checked first), or this flush failed it with a definitive cause.
    //   The host fault latches the layer's cause, observe() drops the
    //   records while the buffered count keeps growing, and the host content
    //   keeps every edit.
    // - Deferred: a transient cause keeps every record for a later flush,
    //   which compiles the whole buffer, later records included, into one
    //   intent; status reports BLOCKED with that cause until an admission.
    //   Records recorded while this flush ran also defer, with no cause. An
    //   allocation failure latches allocationDeferred until the next attempt.
    // - Admitted: an admission left the buffer empty. `issued` reports
    //   whether it issued an intent; an empty or superseded envelope issues
    //   nothing, and the projection still rebaselines.
    // - Idle: the buffer was empty, on a healthy or FAILED layer.
    // An issued intent is never recompiled, merged or retried.
    // With `coalesce`, a flush while the layer's admitted writes fill its
    // issuance window defers with no cause and does no other work, so the
    // buffer coalesces later host turns into one update once the window has
    // room (design section 8); the layer requests that flush. A buffer of
    // several turns that one update cannot hold within a limit admits the
    // most whole turns that fit, found by a doubling and binary search of compiles that
    // admit nothing, and requests another flush for the rest, so merging
    // never fails a flush that admitting each turn alone would not.
    FlushOutcome flushLocalEdits(bool* issued = nullptr, bool coalesce = false) noexcept {
        if (issued) *issued = false;
        allocationDeferred = false;
        flushRequest->refused.store(false);
        if (!hostState) return FlushOutcome::Idle;
        // Every attempt ends the host turn, so the next mutation starts a new
        // one, which the buffer quota admits only below its limit.
        const struct TurnEnd {
            PXR_NS::TfRefPtr<live::HostSceneProjection> projection;
            ~TurnEnd() {
                if (projection) projection->endHostTurn();
            }
        } turnEnd{hostState->authorized()};
        const auto fault = hostState->fault();
        std::shared_ptr<live::SceneAttachment> attachment;
        try {
            attachment = binding->attachment();
        } catch (...) {} // CANCELLED after detach.
        const auto& projection = hostState->authorized();
        const bool pending = projection && projection->hasPendingLocalEdits();
        const auto failed = failedLayerCause(*fault, attachment.get());
        if (failed != OMNI_USD_LIVE_OK) {
            // A FAILED layer admits nothing. Its buffered count only grows
            // and exceeds the records it retains, which observe() drops.
            if (pending) {
                // Latch the cause on the host side too; observe() then drops
                // the records, and the host content keeps every edit.
                fault->fail(failed);
                observe();
                return FlushOutcome::Failed;
            }
            if (binding->bufferedEdits()->load() == 0) binding->notePendingLocalEdits(false);
            return FlushOutcome::Idle;
        }
        if (!pending) {
            // Cleared only here: a non-empty buffer keeps it set, also on a
            // failed layer, so an off-host save never drains without it.
            binding->notePendingLocalEdits(false);
            noteBuffered(0);
            return FlushOutcome::Idle;
        }
        // A replaced backing is observed and fails the layer at the next boundary.
        if (currentData().operator->() != projection.operator->() || !attachment) return FlushOutcome::Deferred;
        // An attempt that finds records no earlier attempt saw ends a host
        // turn: the buffer's record count is that turn's end. Failing to
        // record it is a transient allocation failure of this flush.
        const auto recorded = projection->pendingLocalEditCount();
        std::exception_ptr failure;
        try {
            if (recorded > (turnEnds.empty() ? 0 : turnEnds.back())) turnEnds.push_back(recorded);
        } catch (...) { failure = std::current_exception(); }
        if (!failure && coalesce && attachment->issuanceSaturated()) {
            // The layer requests a flush once its window has room; checked
            // again after the request is armed, so neither side misses it.
            binding->bufferedEdits()->awaitSlot();
            if (attachment->issuanceSaturated()) {
                noteBuffered(recorded);
                return FlushOutcome::Deferred;
            }
        }
        uint64_t admitted = 0;
        live::WireScenePtr applied;
        size_t count = 0, turns = std::max<size_t>(1, turnEnds.size());
        bool partial = false;
        const auto releases = binding->bufferedEdits()->releases();
        std::shared_ptr<const std::vector<live::SdfOperation>> operations;
        live::HostSceneProjection::LocalEdits edits;
        if (!failure) try {
                edits = projection->pendingLocalEdits();
                operations = edits.operations;
                count = edits.operations->size();
                admitted = attachment->submitCaptured(edits.base,
                    std::make_shared<const live::SdfOperationLog>(live::SdfOperationLog{edits.data, edits.operations}),
                    &applied);
            } catch (...) { failure = std::current_exception(); }
        if (failure && count && limitFailure(failure)) try {
                // The rare path: the turns exceed a limit together. A turn
                // ends at each change notice and at each flush attempt, so a
                // prefix may end at any of those ends.
                std::vector<size_t> ends;
                ends.reserve(turnEnds.size() + callEnds.size());
                std::merge(turnEnds.begin(), turnEnds.end(), callEnds.begin(), callEnds.end(),
                    std::back_inserter(ends));
                ends.erase(std::unique(ends.begin(), ends.end()), ends.end());
                ends.erase(std::upper_bound(ends.begin(), ends.end(), count), ends.end());
                if (ends.size() < 2 || ends.back() != count) std::rethrow_exception(failure);
                const auto prefix = [&](size_t whole) {
                    return std::make_shared<const live::SdfOperationLog>(live::SdfOperationLog{nullptr,
                        std::make_shared<const std::vector<live::SdfOperation>>(edits.operations->begin(),
                            edits.operations->begin() + std::ptrdiff_t(ends[whole - 1]))});
                };
                // Probes double from one turn until one exceeds the limit, then
                // bisect, so a search costs compilations in proportion to the
                // prefix that fits, never to a buffer of many updates.
                size_t fits = 0, over = ends.size();
                // The compiler refuses a log of more records than the
                // journal's log operation limit, which counts every write,
                // before any per-operation work, so a prefix over it never
                // fits and the search compiles none. Repeated writes
                // collapse, so a prefix within it may fit however many
                // records it holds, and only compiling decides.
                const auto withinLog = size_t(
                    std::upper_bound(ends.begin(), ends.end(), attachment->maxLogOperations()) - ends.begin());
                over = std::min(over, withinLog + 1);
                std::exception_ptr first;
                const auto probe = [&](size_t turns) {
                    try {
                        attachment->probeCaptured(edits.base, prefix(turns));
                        fits = turns;
                    } catch (...) {
                        if (!limitFailure(std::current_exception())) throw;
                        if (turns == 1) first = std::current_exception();
                        over = turns;
                    }
                };
                for (size_t turns = 1; turns < over && fits < turns; turns *= 2) probe(turns);
                while (over - fits > 1) probe(fits + (over - fits) / 2);
                if (!fits) {
                    // The first turn alone exceeds the limit: it fails as one turn does.
                    if (!first) attachment->probeCaptured(edits.base, prefix(1));
                    failure = first ? first : failure;
                } else {
                    const auto log = prefix(fits);
                    operations = log->operations;
                    admitted = attachment->submitCaptured(edits.base, log, &applied);
                    count = ends[fits - 1];
                    turns = std::max<size_t>(1,
                        size_t(std::upper_bound(turnEnds.begin(), turnEnds.end(), count) - turnEnds.begin()));
                    partial = true;
                    failure = nullptr;
                }
            } catch (...) { failure = std::current_exception(); }
        if (failure) try {
                std::rethrow_exception(failure);
            } catch (...) {
                // Classification point for an unissued buffer. A layer that failed
                // meanwhile reports its own cause and never defers.
                auto cause = failedLayerCause(*fault, attachment.get());
                std::optional<live::UnsupportedOperation> unsupported;
                if (cause == OMNI_USD_LIVE_OK) {
                    try {
                        throw;
                    } catch (const live::CapturedRejection& rejection) {
                        cause = rejection.result;
                    } catch (const live::UnsupportedOperation& error) {
                        cause = OMNI_USD_LIVE_UNSUPPORTED;
                        unsupported.emplace(error);
                    } catch (const live::DeltaEntryLimitExceeded& refused) {
                        // A compile limit: nothing was issued, and the host
                        // content keeps every edit for export.
                        reportEntryLimit(refused);
                        cause = OMNI_USD_LIVE_RESOURCE_EXHAUSTED;
                    } catch (...) {
                        cause = creationException();
                        if (transientFlushFailure(cause)) {
                            try {
                                throw;
                            } catch (const std::bad_alloc&) {
                                allocationDeferred = true;
                            } catch (const std::length_error&) { allocationDeferred = true; } catch (...) {
                            }
                            noteBuffered(projection->pendingLocalEditCount(), cause);
                            // A pending-byte release since the attempt started saw
                            // no deferral cause yet; retry for it.
                            if (binding->bufferedEdits()->releases() != releases) scheduleFlush();
                            return FlushOutcome::Deferred;
                        }
                    }
                }
                fault->fail(cause);
                observe();
                if (unsupported) reportUnsupported(*attachment, *unsupported, operations.get());
                return FlushOutcome::Failed;
            }
        try {
            if (!applied) throw OmniUsdLiveResult(OMNI_USD_LIVE_INTERNAL_ERROR);
            projection->acceptLocalEdits(count, std::move(applied));
            chargeBuffer(projection->pendingLocalEditBytes());
        } catch (...) {
            // Admission already covers these records; they are never admitted again.
            fault->fail(creationException());
            observe();
            return FlushOutcome::Failed;
        }
        if (issued) *issued = admitted != 0;
        // The USD calls and change blocks whose records the update holds.
        const auto calls = size_t(std::upper_bound(callEnds.begin(), callEnds.end(), count) - callEnds.begin());
        callEnds.erase(callEnds.begin(), callEnds.begin() + std::ptrdiff_t(calls));
        for (auto& end : callEnds) end -= std::min(end, count);
        if (admitted) attachment->noteCoalescedTurns(std::max(turns, calls));
        // The admitted turns leave the buffer; later turns' ends shift down.
        const auto kept = size_t(std::upper_bound(turnEnds.begin(), turnEnds.end(), count) - turnEnds.begin());
        turnEnds.erase(turnEnds.begin(), turnEnds.begin() + std::ptrdiff_t(kept));
        for (auto& end : turnEnds) end -= std::min(end, count);
        if (!partial) turnEnds.clear();
        // From here on, attachment work covers every admitted record.
        const bool remaining = projection->hasPendingLocalEdits();
        if (partial && remaining) scheduleFlush(); // The turns past the admitted prefix.
        binding->notePendingLocalEdits(remaining);
        noteBuffered(projection->pendingLocalEditCount());
        return remaining ? FlushOutcome::Deferred : FlushOutcome::Admitted;
    }
    // Host-confined; never throws. Name the cause of a flush that failed
    // UNSUPPORTED: a USD runtime error on the host names the layer and reason,
    // plus the path, field and value type when a single operation causes it,
    // and flush_unsupported records only the numeric reason (design section
    // 18 keeps names out of diagnostics).
    void reportUnsupported(live::SceneAttachment& attachment, const live::UnsupportedOperation& error,
        const std::vector<live::SdfOperation>* operations) const noexcept {
        using namespace PXR_NS; // TF_RUNTIME_ERROR names Tf symbols unqualified.
        try {
            auto text = "Live layer " + layer().GetIdentifier() + ": unsupported edit";
            if (operations && error.index() < operations->size()) {
                const auto& op = (*operations)[error.index()];
                const bool sample = op.kind == live::SdfOperationKind::SetSample ||
                    op.kind == live::SdfOperationKind::EraseSample;
                const auto& field = sample ? SdfDataTokens->TimeSamples : op.field;
                text += " at " + op.path.GetString();
                if (!field.IsEmpty()) text += " field " + field.GetString();
                if (!op.value.IsEmpty()) text += ", value type " + op.value.GetTypeName();
            }
            text += ": ";
            text += error.what();
            TF_RUNTIME_ERROR("%s (result=%u)", text.c_str(), unsigned(OMNI_USD_LIVE_UNSUPPORTED));
        } catch (...) {} // The layer's typed failure stands without its description.
        attachment.reportFlushUnsupported(error.reason());
    }
    // Host-confined; never throws. A USD runtime error on the host names the
    // layer and the counts of an update the service's entry limits refuse.
    void reportEntryLimit(const live::DeltaEntryLimitExceeded& refused) const noexcept {
        using namespace PXR_NS; // TF_RUNTIME_ERROR names Tf symbols unqualified.
        try {
            const auto text = "Live layer " + layer().GetIdentifier() +
                ": the update exceeds the Live Edit service's entry limits (" + refused.describe() +
                "); nothing was sent and the layer fails, and its content keeps the edits for SdfLayer::Export";
            TF_RUNTIME_ERROR("%s (result=%u)", text.c_str(), unsigned(OMNI_USD_LIVE_RESOURCE_EXHAUSTED));
        } catch (...) {} // The layer's typed status stands without its description.
    }
    // Host-confined. Install `visible`, the journal's visible scene, as the
    // layer's content; the only path that changes the content to journal
    // state. The caller has flushed the host edit buffer empty.
    // - The layer's first installation replaces the content with a fresh
    //   authorized projection and sends USD's whole-content replacement
    //   notice; before the layer's initialization completes USD sends none.
    //   A reset snapshot and a replacement of the layer's object are accepted
    //   only at bootstrap, and recovery reopens a layer, so each of them is a
    //   first installation.
    // - Every later installation reseats the authorized projection in place
    //   and sends per-path notices for the difference between its baseline,
    //   which the host content equals, and `visible`: removed, moved and
    //   created specs and changed fields (publishSceneDifference), inside
    //   one change block. Nothing is recorded into the host edit buffer.
    // - A `visible` with no difference from the baseline and the same source
    //   format, such as every publication of the layer's own admitted edits,
    //   whose verified candidate the flush rebaselined to, changes nothing:
    //   the projection keeps its baseline and content, which stay equal even
    //   where the comparison treats a representation change as equal, and no
    //   notice occurs (HostSceneProjection::keepsBaseline). An empty
    //   difference with a new source format reseats without a notice.
    // Comparing and installing cost work in proportion to the sections the
    // baseline and `visible` do not share, with a logarithmic factor in the
    // scene; USD's recomposition follows the changed paths only.
    void install(const live::WireScenePtr& visible) {
        auto& state = *hostState;
        const auto& authorized = state.authorized();
        const bool installed = authorized && !authorized->hasPendingLocalEdits() &&
            currentData().operator->() == authorized.operator->();
        if (installed && contentInstalled) {
            const auto baseline = authorized->baseline();
            const auto difference = baseline == visible ? live::WireSceneDifference{} :
                                                          live::diffProjectedScenes(*baseline, *visible);
            if (!authorized->keepsBaseline(visible, difference)) publishInPlace(visible, difference);
            installDeferred = false;
            return;
        }
        if (installed && live::matchesWireProjection(visible, *authorized)) {
            authorized->acceptLocalEdits(0, visible);
            installDeferred = false;
            contentInstalled = true;
            return;
        }
        auto candidate = state.prepare(visible);
        armProjection(*candidate);
        const PXR_NS::SdfAbstractDataConstPtr prepared(candidate);
        const auto admitted = state.begin(currentData(), candidate);
        if (admitted != live::HostLayerState::Admission::Accepted) throw state.admissionResult(admitted);
        try {
            {
                PXR_NS::SdfChangeBlock block;
                installer(layer(), candidate);
                // Record installation even when a later observer replaces
                // backing before our per-layer notice is delivered.
                state.observe(currentData());
            }
            if (!state.finish(currentData())) throw state.cause();
        } catch (...) {
            // Observation absorbs accessor failures, so abort always runs
            // and the original publication exception remains intact.
            observe();
            state.abort();
            observe();
            throw;
        }
        installDeferred = false;
        contentInstalled = true;
    }
    // Host-confined. Reseat the authorized projection, the layer's backing,
    // with `visible` and send `difference` as per-path notices through this
    // state delegate (publishSceneDifference), all in one change block while
    // installing() holds. The new content is prepared before any change; a
    // failure after the first notice fails the layer, whose notices are then
    // incomplete.
    void publishInPlace(const live::WireScenePtr& visible, const live::WireSceneDifference& difference) {
        auto& state = *hostState;
        const auto projection = state.authorized();
        auto prepared = projection->prepareReseat(visible, difference);
        const auto admitted = state.beginInPlace(currentData());
        if (admitted != live::HostLayerState::Admission::Accepted) throw state.admissionResult(admitted);
        try {
            {
                PXR_NS::SdfChangeBlock block;
                live::publishSceneDifference(*this, *projection, std::move(prepared), difference);
            }
            if (!state.finishInPlace(currentData())) throw state.cause();
        } catch (...) {
            observe();
            state.abort();
            observe();
            throw;
        }
    }
    // Host-confined, except for an unsupported off-host record, which fails
    // the layer and publishes its count from the recording thread.
    // Publish the buffer size and deferral cause to status. The actor is
    // woken only when no earlier wake still awaits its status publication;
    // records only store the count, so a host edit costs at most its change
    // notice's wake and its flush's.
    void noteBuffered(size_t pending, OmniUsdLiveResult blocked = OMNI_USD_LIVE_OK) const noexcept {
        const auto count = binding->bufferedEdits();
        // Records store the count without a wake, so an unchanged stored
        // count may still be unpublished.
        if (!binding->noteBufferedEdits(pending, blocked)) return;
        try {
            if (const auto attachment = binding->attachment()) {
                attachment->noteHostEdits();
                return;
            }
        } catch (...) {}
        count->abandonWake(); // No actor yet or any more; the next record asks again.
    }
    // Host-confined. A backing swap the plugin did not install (Clear or
    // another whole-content replacement installs fresh InitData) fails the
    // layer; arming the swapped-in projection once keeps its later host edits
    // counted, so a FAILED layer's buffered count keeps growing. A stock
    // SdfData backing has no record hook and cannot be armed.
    void armReplacement() const {
        const auto data = currentData();
        const auto* projection = dynamic_cast<const live::HostSceneProjection*>(data.operator->());
        if (!projection || (armedReplacement && armedReplacement.operator->() == projection)) return;
        const_cast<BoundLayer*>(this)->armProjection(const_cast<live::HostSceneProjection&>(*projection));
        armedReplacement = PXR_NS::SdfAbstractDataConstPtr(data);
    }
    // Every projection the layer backs itself with is armed with both sinks,
    // so the projection's no-sink default refusal never applies to it.
    void armProjection(live::HostSceneProjection& projection) {
        // Schedule only: the sinks never reenter the projection they report for.
        projection.setLocalEditSink([weak = PXR_NS::TfCreateWeakPtr(this)](size_t pending) {
            if (weak) get_pointer(weak)->recorded(pending);
        });
        projection.setQuotaSink([weak = PXR_NS::TfCreateWeakPtr(this)](
                                    size_t pending) { return !weak || get_pointer(weak)->refused(pending); });
        projection.setBudgetSink([weak = PXR_NS::TfCreateWeakPtr(this)](size_t pendingBytes) {
            return weak ? get_pointer(weak)->budget(pendingBytes) : live::HostEditBudget{};
        });
    }
    // Host-confined. Charge `bytes`, the authorized buffer's bytes, to the
    // endpoint's shared host buffer budget (design section 8, Quotas). Only
    // a non-empty buffer holds a charge, and with it an endpoint lease; the
    // charge follows the layer's current attachment. Without an attachment,
    // or when the charge cannot be created, the buffer is uncharged: the
    // endpoint's budget does not count it, and the layer's own quota,
    // HostEditBufferLimits::maxPendingBytes (1 GiB by default), bounds it.
    void chargeBuffer(size_t bytes) const noexcept {
        std::shared_ptr<live::SceneAttachment> attachment;
        if (bytes) try {
                attachment = binding->attachment();
            } catch (...) {}
        if (bufferCharge && (!attachment || chargedAttachment.lock() != attachment)) bufferCharge.reset();
        if (!attachment) return;
        if (!bufferCharge) try {
                bufferCharge = attachment->hostBufferCharge();
                chargedAttachment = attachment;
            } catch (...) { return; }
        if (bufferCharge) bufferCharge->set(bytes);
    }
    // The budget sink: a mutation would start a host turn on a healthy layer
    // whose buffer holds `pendingBytes`. On the host it charges them and
    // reports the endpoint's shared budget, which a layer without a charge,
    // such as one whose buffer is empty, reads through its attachment, so an
    // empty buffer is refused once other buffers exhaust the budget. Without
    // an attachment it reports nothing, and only the layer's own quota
    // applies. Elsewhere it reports nothing, and the record sink fails the
    // layer for the off-host record.
    live::HostEditBudget budget(size_t pendingBytes) const noexcept {
        if (!onHost() || !hostState || hostState->installing()) return {};
        chargeBuffer(pendingBytes);
        live::HostBufferCharge::Budget charged;
        if (bufferCharge) charged = bufferCharge->budget();
        else {
            std::shared_ptr<live::SceneAttachment> attachment;
            try {
                attachment = binding->attachment();
            } catch (...) {}
            if (!attachment) return {};
            charged = attachment->hostBufferBudget();
        }
        return {charged.exhausted, charged.charged, charged.limit};
    }
    // Whether the host predicate holds on the calling thread; a missing or
    // throwing predicate counts as false.
    bool onHost() const noexcept {
        try {
            return hostContext && hostContext->isCurrent();
        } catch (...) { return false; }
    }
    // Every record. A mutation where the host predicate does not hold, or
    // inside this layer's own installation (from a handler for a notice the
    // publication sent), is unsupported (design section 7): the layer fails
    // UNSUPPORTED, the edit stays in host content and in the buffer, and no
    // flush is requested. A record on a layer whose host fault is latched
    // adds one to the buffered count, which only grows once the layer is
    // FAILED, and requests no flush. Otherwise, every record requests a
    // flush, so a buffer a transient failure left pending flushes again with
    // the next record; scheduleFlush keeps at most one job queued. A record
    // keeps the current deferral cause. Detection costs one predicate call
    // per record, so the host predicate must be cheap.
    void recorded(size_t pending) noexcept {
        binding->notePendingLocalEdits(true);
        const auto count = binding->bufferedEdits();
        const bool violation = !onHost() || !hostState || hostState->installing();
        if (violation && hostState) hostState->fault()->fail(OMNI_USD_LIVE_UNSUPPORTED);
        if (violation ||
            (hostState && hostState->fault()->result().load(std::memory_order_acquire) != OMNI_USD_LIVE_OK)) {
            // Load-then-store, not an atomic increment: host records are
            // serialized, and only an unsupported off-host record can race
            // one, which at worst undercounts that already-FAILED layer.
            noteBuffered(count->load() + 1, count->blocked());
            // Wake the actor directly; host-confined observation waits for
            // the next host boundary.
            if (violation) try {
                    if (const auto attachment = binding->attachment()) (void)attachment->observeHostFailure();
                } catch (...) {}
            return;
        }
        // Records only store the count; the change notice that ends the USD
        // call or change block publishes it, one wake per host edit.
        count->store(pending, count->blocked());
        scheduleFlush();
    }
    // A mutation that would start a host turn while the buffer holds its
    // quota or the endpoint's shared host buffer budget is exhausted; returns
    // whether the projection refuses it before any change. A layer refused
    // for the shared budget may hold no records itself; another buffer's
    // release on the endpoint requests its flush, which clears the cause.
    // - Off the host, or inside this layer's own installation, the mutation
    //   is unsupported as in recorded(): the layer fails UNSUPPORTED, the
    //   actor is woken and the mutation is refused. Status is published only
    //   through the actor's failure, never from the recording thread.
    // - A layer already FAILED, by the actor or the host fault, never
    //   refuses: its edits apply and its count keeps growing.
    // - Otherwise the layer is BLOCKED with RESOURCE_EXHAUSTED at the
    //   buffered count until a flush admits. Nothing is recorded. A BUSY,
    //   CANCELLED, pending-intent or pending-byte deferral has its own retry
    //   events, so no flush is requested. An allocation failure and a
    //   dispatcher's capacity refusal of the flush job otherwise wait for the
    //   layer's next record, which the quota refuses, so the refusal requests
    //   a flush; scheduleFlush keeps at most one queued. A dispatcher refusal
    //   also retries at the next on-host or off-host save, publication job or
    //   endpoint pending-byte release.
    //   RESOURCE_EXHAUSTED replaces a BUSY or CANCELLED deferral cause in
    //   status until a flush attempt publishes its own transient cause or
    //   admits the buffer.
    bool refused(size_t pending) noexcept {
        if (!onHost() || !hostState || hostState->installing()) {
            if (hostState) hostState->fault()->fail(OMNI_USD_LIVE_UNSUPPORTED);
            try {
                if (const auto attachment = binding->attachment()) (void)attachment->observeHostFailure();
            } catch (...) {}
            return true;
        }
        if (failedLayerCause() != OMNI_USD_LIVE_OK) return false;
        noteBuffered(pending, OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
        if (allocationDeferred || flushRequest->refused.load()) scheduleFlush();
        return true;
    }
    void changed(const PXR_NS::SdfNotice::LayersDidChangeSentPerLayer&, const PXR_NS::SdfLayerHandle& sender) {
        // Notices from a layer whose destruction has begun, or whose delegate was replaced, are ignored.
        if (!layer_.load() || !sender || sender->GetCurrentCount() == 0) return;
        // A notice where the host predicate does not hold follows an
        // unsupported mutation that the record sink already failed; host-
        // confined state is not touched from here.
        if (!onHost()) {
            try {
                if (const auto attachment = binding->attachment()) (void)attachment->observeHostFailure();
            } catch (...) {}
            return;
        }
        observe();
        // The notice follows the whole USD call or change block and ends the
        // host turn, whether or not a flush job has run: the next mutation
        // starts a new turn, which the buffer quota admits only below its
        // limit (design section 8, Quotas). The records the call added are
        // one host edit, which the coalesced-turn metric counts, and the
        // notice publishes the buffered count they left.
        if (hostState)
            if (const auto& projection = hostState->authorized()) {
                projection->endHostTurn();
                const auto recorded = projection->pendingLocalEditCount();
                if (failedLayerCause() == OMNI_USD_LIVE_OK) chargeBuffer(projection->pendingLocalEditBytes());
                if (recorded > (callEnds.empty() ? 0 : callEnds.back())) {
                    try {
                        callEnds.push_back(recorded);
                    } catch (...) {} // The call counts with the next.
                    if (failedLayerCause() == OMNI_USD_LIVE_OK)
                        noteBuffered(recorded, binding->bufferedEdits()->blocked());
                }
            }
    }
    BoundLayer(const PXR_NS::SdfFileFormatConstPtr& format, DataAccessor accessor, DataInstaller installer)
        : format(PXR_NS::TfCreateRefPtrFromProtectedWeakPtr(format)),
          binding(std::make_shared<live::UsdLayerBinding>()), hostContext(configuredHost()), accessor(accessor),
          installer(installer), flushRequest(std::make_shared<FlushRequest>()) {}
    bool _IsDirty() override { return liveLayerDirty(binding); }
    void _MarkCurrentStateAsClean() override {}
    void _MarkCurrentStateAsDirty() override {}
    // USD calls this with the adopting layer, again with it when the layer's
    // identifier changes, and with null or another layer when the delegate is
    // replaced or the layer is destroyed.
    void _OnSetLayer(const PXR_NS::SdfLayerHandle& handle) override {
        SdfSimpleLayerStateDelegate::_OnSetLayer(handle);
        auto* current = layer_.load();
        if (!current || get_pointer(handle) == current) return;
        unbind(current->GetCurrentCount() != 0);
    }
    // Ends the layer's live state once, when the layer is destroyed or replaces
    // this delegate. A replacement fails a layer that is still alive: its host
    // edits stay applied, it keeps no record, and its attachment observes the
    // failure. Destruction touches no SdfLayer member.
    void unbind(bool replaced) noexcept {
        if (!layer_.load()) return;
        if (replaced && hostState) {
            hostState->fault()->fail(OMNI_USD_LIVE_UNSUPPORTED);
            discardRecords();
            try {
                if (const auto attachment = binding->attachment()) (void)attachment->observeHostFailure();
            } catch (...) {}
        }
        layer_.store(nullptr);
        if (initializingTarget) (void)binding->detachWithoutClosing();
        else binding->detach();
        PXR_NS::TfNotice::Revoke(notice);
    }
    // The layer's file format, retained through the layer's whole life.
    const PXR_NS::SdfFileFormatConstRefPtr format;
    std::atomic<PXR_NS::SdfLayer*> layer_{nullptr};
    const std::shared_ptr<live::UsdLayerBinding> binding;
    const std::shared_ptr<const Host> hostContext;
    const DataAccessor accessor;
    const DataInstaller installer;
    const std::shared_ptr<FlushRequest> flushRequest;
    // Host-confined: the buffer's record count at the end of each host turn
    // it holds, in order; a flush that cannot admit the whole buffer within a
    // limit admits a prefix of whole turns.
    std::vector<size_t> turnEnds;
    // Host-confined: the buffer's record count at the change notice that
    // ends each USD call or change block whose records it holds.
    std::vector<size_t> callEnds;
    // Whether a flush failure is a limit a smaller update may not reach.
    static bool limitFailure(const std::exception_ptr& failure) noexcept {
        try {
            std::rethrow_exception(failure);
        } catch (const live::CapturedRejection& rejection) {
            return rejection.result == OMNI_USD_LIVE_RESOURCE_EXHAUSTED;
        } catch (const live::DeltaEntryLimitExceeded&) { return true; } catch (...) {
            return false;
        }
    }
    // Host-confined. A publication job deferred installation over unflushed
    // host edits; the next publication or issuing-nothing flush installs.
    bool installDeferred = false;
    // Host-confined. An installation completed, so later ones are incremental.
    bool contentInstalled = false;
    // Host-confined. The last flush attempt deferred on an allocation failure.
    bool allocationDeferred = false;
    // Host-confined. The authorized buffer's charge to the endpoint's shared
    // host buffer budget, and the attachment it was created from.
    mutable std::unique_ptr<live::HostBufferCharge> bufferCharge;
    mutable std::weak_ptr<live::SceneAttachment> chargedAttachment;
    // The backing armReplacement last armed; weak, so a destroyed backing
    // never matches a later one at the same address.
    mutable PXR_NS::SdfAbstractDataConstPtr armedReplacement;
    mutable std::optional<live::HostLayerState> hostState;
    PXR_NS::TfNotice::Key notice;
    OmniUsdLiveResult setupResult = OMNI_USD_LIVE_OK;
    std::shared_ptr<live::UsdCreationWorkflow> creationWorkflow;
    // CreateNew's canonical target before finalization; its workflow owns the attachment.
    bool initializingTarget = false;
    bool readAttempted = false;
};
// CreateNew's queued finalizer, dispatched by SaveToFile; defined in
// UsdLayerCreation.cpp.
struct InitializingFinalization {
    live::UsdCreationLease lease;
    std::shared_ptr<live::SceneAttachment> attachment;
    std::atomic<bool> hookActive{true}, abandoned{false}, started{false}, completed{false};
    std::chrono::steady_clock::time_point deadline;
    ~InitializingFinalization();
    void run() noexcept;
};
// Export's successful activation: queue a host job that adopts the workflow
// into its layer after the outer Export returns. Defined in
// UsdLayerCreation.cpp. A job that runs inside the dispatch call or is never
// accepted leaves the workflow retained and reports why.
void queueExportAdoption(live::UsdCreationLease) noexcept;
} // namespace omni::usd::live::fileformat
