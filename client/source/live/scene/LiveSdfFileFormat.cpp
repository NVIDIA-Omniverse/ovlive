// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "LiveSdfFileFormat.h"
#include "BoundLayer.h"
#include "LiveExceptionResult.h"
using namespace omni::usd::live::fileformat;

namespace {
namespace live = omni::usd::live;
std::mutex dispatcherMutex;
std::shared_ptr<const Host> publicationHost;
std::mutex journalLimitsMutex;
std::optional<live::SceneJournal::Limits> journalLimitsForTest;
std::optional<live::DeltaEntryLimits> entryLimitsForTest; // Under journalLimitsMutex.
live::SceneJournal::Limits journalLimits() {
    std::lock_guard<std::mutex> lock(journalLimitsMutex);
    auto limits = journalLimitsForTest.value_or(live::SceneJournal::Limits{});
    if (entryLimitsForTest) limits.authoring.entries = *entryLimitsForTest;
    return limits;
}
// Export and CreateNew compile their seed and open their layer with the
// default journal limits and the service's entry limits.
live::SceneJournal::Limits creationLimits() {
    live::SceneJournal::Limits limits;
    std::lock_guard<std::mutex> lock(journalLimitsMutex);
    if (entryLimitsForTest) limits.authoring.entries = *entryLimitsForTest;
    return limits;
}
std::optional<size_t> hostEditQuotaForTest; // Under journalLimitsMutex.
live::HostSceneProjection::Limits hostEditLimits() {
    live::HostSceneProjection::Limits limits;
    std::lock_guard<std::mutex> lock(journalLimitsMutex);
    if (hostEditQuotaForTest) limits.maxPendingBytes = *hostEditQuotaForTest;
    return limits;
}
live::SceneAttachment::Changed progressObserverForTest; // Under journalLimitsMutex.
std::function<std::chrono::steady_clock::time_point()> creationClockForTest; // Under journalLimitsMutex.
}
namespace omni::usd::live::fileformat {
live::SceneAttachment::Changed progressObserver() {
    std::lock_guard<std::mutex> lock(journalLimitsMutex);
    return progressObserverForTest;
}
// The clock of Export and CreateNew creation deadlines. The test clock is
// copied under the mutex and called outside it.
std::chrono::steady_clock::time_point creationNow() {
    std::function<std::chrono::steady_clock::time_point()> clock;
    {
        std::lock_guard<std::mutex> lock(journalLimitsMutex);
        clock = creationClockForTest;
    }
    return clock ? clock() : std::chrono::steady_clock::now();
}
std::shared_ptr<const Host> configuredHost() {
    std::lock_guard<std::mutex> lock(dispatcherMutex);
    return publicationHost;
}

live::UsdPublicationDispatcher retainedDispatch(const std::shared_ptr<const Host>& context) {
    return [context](std::function<void()> job) {
        // Even weak jobs retain the original executor identity until
        // every accepted callable copy retires, including queue-then-throw.
        context->dispatch([context, job = std::move(job)] { job(); });
    };
}
OmniUsdLiveResult creationException() noexcept {
    try {
        throw;
    } catch (const live::LayerOperationRejected& rejected) {
        return rejected.result();
    } catch (const live::wire::DecodeError& error) {
        if (error.kind == live::wire::ErrorKind::ResourceLimit) return OMNI_USD_LIVE_RESOURCE_EXHAUSTED;
        if (error.kind == live::wire::ErrorKind::Unsupported) return OMNI_USD_LIVE_UNSUPPORTED;
        return OMNI_USD_LIVE_INVALID_ARGUMENT;
    } catch (...) { return live::currentExceptionResult(); }
}
}
namespace {
struct RescheduleFlush {
    BoundLayer& layer;
    ~RescheduleFlush() {
        const auto& projection = layer.hostState->authorized();
        if (projection && projection->hasPendingLocalEdits()) layer.scheduleFlush();
    }
};
OmniUsdLiveResult failedCause(const BoundLayer& layer) noexcept {
    const auto cause = layer.failedLayerCause();
    return cause != OMNI_USD_LIVE_OK ? cause : OmniUsdLiveResult(OMNI_USD_LIVE_INTERNAL_ERROR);
}
}
namespace omni::usd::live {
void prepareInitializingCreationTarget(PXR_NS::SdfLayer& layer,
    const std::shared_ptr<UsdCreationWorkflow>& workflow) try {
    auto* bound = BoundLayer::from(&layer);
    if (!bound || !workflow || !workflow->creation || !workflow->runtime)
        throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
    if (!bound->hostContext || !bound->hostContext->isCurrent() || DispatchExecutor::isWorker())
        throw OmniUsdLiveResult(OMNI_USD_LIVE_BUSY);
    if (bound->hostContext != configuredHost()) throw OmniUsdLiveResult(OMNI_USD_LIVE_NOT_INITIALIZED);
    if (bound->creationWorkflow || bound->initializingTarget || bound->binding->attachment())
        throw OmniUsdLiveResult(OMNI_USD_LIVE_ALREADY_EXISTS);
    const auto state = workflow->snapshot();
    const auto finite = workflow->creation->snapshot();
    if (state.activeCall || state.closing || state.openingAttempted || state.activated || state.attachment ||
        finite.phase != SceneCreation::Phase::Prepared || finite.started || finite.closing)
        throw OmniUsdLiveResult(OMNI_USD_LIVE_BUSY);
    const auto& uri = workflow->creation->admission()->canonicalUri();
    if (layer.IsAnonymous() || layer.GetIdentifier() != uri || layer.GetResolvedPath().GetPathString() != uri ||
        !layer.GetFileFormatArguments().empty())
        throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
    // The target must be untouched: its backing is the plugin's authorized
    // projection with no host failure.
    const auto data = PXR_NS::OmniUsdLiveNativeFileFormat::_GetLayerData(layer);
    bound->ensureSetup();
    if (!bound->hostState || data.operator->() != bound->hostState->authorized().operator->() ||
        bound->hostState->failure() != HostLayerState::Failure::None)
        throw bound->hostState ? bound->hostState->cause() : OmniUsdLiveResult(OMNI_USD_LIVE_UNSUPPORTED);
    bound->initializingTarget = true;
    bound->creationWorkflow = workflow;
} catch (...) { throw creationException(); }
SceneAttachment::Publication initializingCreationPublication(PXR_NS::SdfLayer& layer,
    const std::shared_ptr<UsdCreationWorkflow>& workflow) try {
    const auto* bound = BoundLayer::from(&layer);
    if (!bound || !workflow || bound->creationWorkflow != workflow || !bound->initializingTarget)
        throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
    const auto state = workflow->snapshot();
    if (state.closing || state.activated) throw OmniUsdLiveResult(OMNI_USD_LIVE_CANCELLED);
    return PXR_NS::OmniUsdLiveNativeFileFormat::_PublicationFor(&layer);
} catch (...) { throw creationException(); }
void bindInitializingCreationAttachment(PXR_NS::SdfLayer& layer, const std::shared_ptr<UsdCreationWorkflow>& workflow,
    const std::shared_ptr<SceneAttachment>& attachment) try {
    auto* bound = BoundLayer::from(&layer);
    if (!bound || !workflow || !attachment || bound->creationWorkflow != workflow || !bound->initializingTarget)
        throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
    if (!bound->hostContext || !bound->hostContext->isCurrent() || DispatchExecutor::isWorker())
        throw OmniUsdLiveResult(OMNI_USD_LIVE_BUSY);
    const auto state = workflow->snapshot();
    if (state.closing || state.activated || !state.openingAttempted || state.attachment != attachment ||
        attachment->subjectId() != workflow->creation->status()->id() ||
        workflow->creation->snapshot().phase != SceneCreation::Phase::Confirmed)
        throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
    PXR_NS::OmniUsdLiveNativeFileFormat::_Attach(&layer, attachment, false);
} catch (...) { throw creationException(); }
void setUsdPublicationDispatcher(UsdPublicationDispatcher dispatch, std::function<bool()> isCurrent) {
    if (bool(dispatch) != bool(isCurrent)) throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
    std::shared_ptr<const Host> replacement;
    if (dispatch) replacement = std::make_shared<Host>(Host{std::move(dispatch), std::move(isCurrent)});
    // Retire arbitrary host captures outside this mutex.
    {
        std::lock_guard<std::mutex> lock(dispatcherMutex);
        publicationHost.swap(replacement);
    }
}
std::shared_ptr<const void> createUsdPublicationContext(UsdPublicationDispatcher dispatch,
    std::function<bool()> isCurrent) {
    if (!dispatch || !isCurrent) throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
    auto configured = std::make_shared<Host>(Host{std::move(dispatch), std::move(isCurrent)});
    std::shared_ptr<const Host> replacement = configured;
    {
        std::lock_guard<std::mutex> lock(dispatcherMutex);
        publicationHost.swap(replacement);
    }
    return configured;
}
void unpublishUsdPublicationContext(const std::shared_ptr<const void>& context) noexcept {
    std::shared_ptr<const Host> retired;
    {
        std::lock_guard<std::mutex> lock(dispatcherMutex);
        if (context && publicationHost.get() == context.get()) retired = std::move(publicationHost);
    }
}
void setLiveLayerJournalLimitsForTest(std::optional<SceneJournal::Limits> limits) {
    std::lock_guard<std::mutex> lock(journalLimitsMutex);
    journalLimitsForTest = std::move(limits);
}
void setLiveLayerEntryLimitsForTest(std::optional<DeltaEntryLimits> limits) {
    std::lock_guard<std::mutex> lock(journalLimitsMutex);
    entryLimitsForTest = limits;
}
void setLiveLayerHostEditQuotaForTest(std::optional<size_t> maxPendingBytes) {
    std::lock_guard<std::mutex> lock(journalLimitsMutex);
    hostEditQuotaForTest = maxPendingBytes;
}
void setLiveLayerProgressObserverForTest(SceneAttachment::Changed observer) {
    std::lock_guard<std::mutex> lock(journalLimitsMutex);
    progressObserverForTest = std::move(observer);
}
void setLiveLayerCreationClockForTest(std::function<std::chrono::steady_clock::time_point()> clock) {
    std::lock_guard<std::mutex> lock(journalLimitsMutex);
    creationClockForTest = std::move(clock);
}
std::shared_ptr<SceneAttachment> attachedLiveScene(const PXR_NS::SdfLayer& layer) {
    const auto* bound = BoundLayer::from(&layer);
    if (!bound) throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
    // This lookup also serves off-host immutable readers.
    // Mutable backing observation belongs to serial host boundaries only.
    return bound->binding->attachment();
}
OmniUsdLiveResult saveAttachedLiveScene(const PXR_NS::SdfLayer& layer, uint32_t timeoutMs) {
    const auto* bound = BoundLayer::from(&layer);
    if (!bound) return OMNI_USD_LIVE_INVALID_ARGUMENT;
    const auto& context = bound->hostContext;
    if (!context) return OMNI_USD_LIVE_NOT_INITIALIZED;
    const auto attachment = bound->binding->attachment();
    if (!attachment) return OMNI_USD_LIVE_NOT_FOUND;
    const bool onHost = context->isCurrent();
    auto* mutableBound = const_cast<BoundLayer*>(bound);
    if (onHost) {
        // Admit recorded host edits first so the drain covers them; a save
        // never drains around unadmitted host edits.
        bound->observe();
        bool issued = false;
        const auto flushed = mutableBound->flushLocalEdits(&issued);
        if (flushed == BoundLayer::FlushOutcome::Failed) return failedCause(*bound);
        if (flushed == BoundLayer::FlushOutcome::Deferred) return OMNI_USD_LIVE_BUSY;
        // A save may run inside the host's change block: queue the install.
        if (mutableBound->deferredInstallDue(flushed, issued)) mutableBound->scheduleFlush();
    } else if (bound->binding->hasPendingLocalEdits()) {
        // Only the host can admit them; a drain here would omit those edits.
        // A failed layer reports its cause; its edits are never admitted.
        // Otherwise a flush job retries admission on the host.
        OmniUsdLiveResult failed = OMNI_USD_LIVE_OK;
        if (bound->hostState) failed = BoundLayer::failedLayerCause(*bound->hostState->fault(), attachment.get());
        if (failed != OMNI_USD_LIVE_OK) return failed;
        mutableBound->scheduleFlush();
        return OMNI_USD_LIVE_BUSY;
    }
    return attachment->save(timeoutMs, !onHost);
}
}
PXR_NAMESPACE_OPEN_SCOPE
enum class OmniUsdLiveFormatError { Failure };
TF_REGISTRY_FUNCTION(TfEnum) { TF_ADD_ENUM_NAME(OmniUsdLiveFormatError::Failure); }
TF_REGISTRY_FUNCTION(TfType) { SDF_DEFINE_FILE_FORMAT(OmniUsdLiveNativeFileFormat, SdfFileFormat); }
PXR_NAMESPACE_CLOSE_SCOPE
namespace omni::usd::live::fileformat {
void report(OmniUsdLiveResult result) noexcept {
    using namespace PXR_NS;
    try {
        TF_ERROR(TfDiagnosticInfo(result), TfEnum(OmniUsdLiveFormatError::Failure),
            "Live file format operation failed (result=%u)", unsigned(result));
    } catch (...) {}
}
}
PXR_NAMESPACE_OPEN_SCOPE
namespace {
// BUSY for a read off the host of a layer that has no attachment; the text
// keeps report()'s prefix and result field and adds the remedy.
void reportUnopenedOffHost() noexcept {
    try {
        TF_ERROR(TfDiagnosticInfo(OmniUsdLiveResult(OMNI_USD_LIVE_BUSY)), TfEnum(OmniUsdLiveFormatError::Failure),
            "Live file format operation failed (result=%u): live layers open only on their host; open live layers on "
            "the host before composing them",
            unsigned(OMNI_USD_LIVE_BUSY));
    } catch (...) {}
}
template<class Result, class Action> Result guarded(Action&& action) noexcept {
    try {
        return action();
    } catch (const live::LayerOperationRejected& rejected) { report(rejected.result()); } catch (...) {
        report(live::currentExceptionResult());
    }
    return {};
}
}
OmniUsdLiveNativeFileFormat::OmniUsdLiveNativeFileFormat()
    : SdfFileFormat(TfToken("live"), TfToken("2"), TfToken("usd"), "live") {}
SdfAbstractDataRefPtr OmniUsdLiveNativeFileFormat::InitData(const FileFormatArguments&) const {
    // Like layer construction, data initialization must produce a valid
    // object; allocation failure follows the SDK's exception boundary.
    SdfAbstractDataRefPtr data = TfCreateRefPtr(
        new live::HostSceneProjection(live::buildSceneCandidate({}, live::wire::encodeEnvelope({})),
            std::make_shared<live::HostFailureLatch>(), hostEditLimits()));
    const SdfAbstractDataConstPtr prepared(data); // Prewarm before SdfLayer's base exists.
    return data;
}
SdfAbstractDataRefPtr OmniUsdLiveNativeFileFormat::_InitDetachedData(const FileFormatArguments& args) const {
    auto data = SdfFileFormat::_InitDetachedData(args);
    const SdfAbstractDataConstPtr prepared(data); // Same no-allocation constructor retention boundary.
    return data; // Unsupported setup is rejected by guarded hooks, not constructor unwind.
}
bool OmniUsdLiveNativeFileFormat::CanRead(const std::string& path) const {
    try {
        const auto parsed = live::parseUri(path);
        return parsed.layerName.size() >= 5 && parsed.layerName.compare(parsed.layerName.size() - 5, 5, ".live") == 0;
    } catch (...) { return false; }
}
SdfLayer* OmniUsdLiveNativeFileFormat::_InstantiateNewLayer(const SdfFileFormatConstPtr& format,
    const std::string& identifier, const std::string& realPath, const ArAssetInfo& info,
    const FileFormatArguments& args) const {
    // USD dereferences this factory result without a null check. Semantic
    // rejection belongs in Read, before attachment admission; allocation or
    // constructor failures must not be translated into a null factory result.
    const BoundLayer::DataAccessor accessor = [](const SdfLayer& layer) { return _GetLayerData(layer); };
    const BoundLayer::DataInstaller installer = [](SdfLayer& layer, const TfRefPtr<live::HostSceneProjection>& data) {
        // The SDK takes a mutable base-typed reference. Never let it swap
        // the coordinator's retained candidate out from under validation.
        SdfAbstractDataRefPtr installed = data;
        const SdfAbstractDataConstPtr prepared(installed);
        SdfChangeBlock block;
        _SetLayerData(&layer, installed);
    };
    // The live state is allocated first, then USD's base factory constructs the
    // plain layer that adopts it as its state delegate (BoundLayer.h).
    const auto state = BoundLayer::create(format, accessor, installer);
    SdfLayer* const layer = SdfFileFormat::_InstantiateNewLayer(format, identifier, realPath, info, args);
    state->adopt(*layer, state);
    return layer;
}
live::SceneAttachment::Publication OmniUsdLiveNativeFileFormat::_PublicationFor(SdfLayer* layer) {
    auto* bound = BoundLayer::from(layer);
    if (!bound) throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
    const auto& openingHost = bound->hostContext;
    if (!openingHost) throw OmniUsdLiveResult(OMNI_USD_LIVE_NOT_INITIALIZED);
    if (!openingHost->isCurrent()) throw OmniUsdLiveResult(OMNI_USD_LIVE_BUSY);
    bound->ensureSetup();
    // ensureSetup's observation latches an allocation failure into the host
    // fault instead of throwing it. A latched fault fails the open here,
    // before it reserves an admission or issues its first RPC.
    const auto& hostState = *bound->hostState;
    if (hostState.failure() != live::HostLayerState::Failure::None ||
        hostState.fault()->result().load(std::memory_order_acquire) != OMNI_USD_LIVE_OK)
        throw hostState.cause();
    auto publication = live::SceneAttachment::Publication{};
    publication.hostFailure = std::shared_ptr<const std::atomic<OmniUsdLiveResult>>(bound->hostState->fault(),
        &bound->hostState->fault()->result());
    publication.bufferedEdits = bound->binding->bufferedEdits();
    // An endpoint pending-byte release requests a flush of a deferred buffer.
    publication.bufferedEdits->setRetry(
        [host = std::weak_ptr<const Host>(openingHost), weak = std::weak_ptr<live::UsdLayerBinding>(bound->binding),
            request = bound->flushRequest, fault = bound->hostState->fault()] {
        // A released binding has no buffer left to flush.
        const auto owner = weak.lock();
        if (!owner) return;
        BoundLayer::dispatchFlush(host.lock(), weak, request, fault, owner->bufferedEdits());
    });
    publication.dispatch = retainedDispatch(openingHost);
    publication.publish = [weak = std::weak_ptr<live::UsdLayerBinding>(bound->binding), openingHost](auto scene) {
        if (!openingHost->isCurrent()) throw OmniUsdLiveResult(OMNI_USD_LIVE_BUSY);
        const auto binding = weak.lock();
        if (!binding) throw OmniUsdLiveResult(OMNI_USD_LIVE_CANCELLED);
        binding->publish(std::move(scene), [](SdfLayer* target, auto snapshot) {
            auto* found = BoundLayer::from(target);
            if (!found) throw OmniUsdLiveResult(OMNI_USD_LIVE_CANCELLED);
            auto& bound = *found;
            bound.ensureSetup();
            using Outcome = BoundLayer::FlushOutcome;
            // Admit recorded host edits before installing so none is discarded;
            // while the issuance window is full they coalesce instead, and the
            // installation waits for the flush that admits them.
            const auto flushed = bound.flushLocalEdits(nullptr, true);
            // A FAILED layer installs nothing further.
            if (const auto cause = bound.failedLayerCause(); cause != OMNI_USD_LIVE_OK) throw cause;
            // A buffer this flush left non-empty requests another flush job
            // after this one; a pending-byte release that follows the job's
            // application retries through the endpoint's release fan-out.
            RescheduleFlush reschedule{bound};
            // Unflushed host edits own the layer content; never install over them.
            if (flushed == Outcome::Deferred) {
                bound.installDeferred = true;
                return;
            }
            // The job's snapshot predates any local admission made after it
            // was taken. Install the journal's current visible scene, which
            // adds those pending intents; installing only this snapshot would
            // drop the user's edits until a later install.
            live::WireScenePtr visible = snapshot->visible;
            if (const auto attachment = bound.binding->attachment())
                if (const auto current = attachment->snapshot(); current->localAdmitted > snapshot->localAdmitted)
                    visible = current->visible;
            bound.install(visible);
        });
    };
    return publication;
}
void OmniUsdLiveNativeFileFormat::_Attach(SdfLayer* layer, const std::shared_ptr<live::SceneAttachment>& attachment,
    bool opened) {
    auto* bound = BoundLayer::from(layer);
    if (!bound || !attachment) throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
    const auto& openingHost = bound->hostContext;
    if (!openingHost) throw OmniUsdLiveResult(OMNI_USD_LIVE_NOT_INITIALIZED);
    if (!openingHost->isCurrent()) throw OmniUsdLiveResult(OMNI_USD_LIVE_BUSY);
    // Only a creation layer attaches without having been opened; reject
    // before the layer gains an attachment.
    if (!opened && !bound->creationWorkflow) throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
    bound->ensureSetup();
    bound->binding->attachAttachment(attachment);
    bound->observe();
    // An opened layer is edited with plain USD once attached; a creation layer
    // gains edit permission at activation.
    if (opened) layer->SetPermissionToEdit(true);
    // Records made before attachment flush once it exists.
    if (bound->hostState && bound->hostState->authorized() && bound->hostState->authorized()->hasPendingLocalEdits())
        bound->scheduleFlush();
}
bool OmniUsdLiveNativeFileFormat::Read(SdfLayer* layer, const std::string& path, bool metadataOnly) const {
    return guarded<bool>([&] {
        auto* bound = BoundLayer::from(layer);
        if (!bound) throw OmniUsdLiveResult(OMNI_USD_LIVE_UNSUPPORTED);
        const auto& openingHost = bound->hostContext;
        if (!openingHost) throw OmniUsdLiveResult(OMNI_USD_LIVE_NOT_INITIALIZED);
        if (!openingHost->isCurrent()) {
            // A host opens each live layer before inserting it into a stage
            // (design section 1). An unattached layer read off the host is
            // most often a live URI composed before its host opened it, such
            // as a sublayer USD loads on a worker thread; a direct open or a
            // reload off the host takes this branch too. The diagnostic,
            // which USD's sublayer warning repeats, names the remedy for each.
            if (!bound->binding->attachment()) {
                reportUnopenedOffHost();
                return false;
            }
            throw OmniUsdLiveResult(OMNI_USD_LIVE_BUSY);
        }
        // SdfLayer::Import reads another asset into this layer through its own
        // format; replacing live content is unsupported (design section 8). Reject
        // before any mutation. Open and Reload read the layer's resolved path.
        if (path != layer->GetResolvedPath().GetPathString())
            throw live::LayerOperationRejected(OMNI_USD_LIVE_UNSUPPORTED);
        // A read of the layer's own path attaches only a layer that has never
        // read and is not a creation target: an Import of that path into a
        // layer whose earlier Read failed, or into an unactivated creation
        // layer, would otherwise replace its content with an attachment.
        if (!bound->binding->attachment() &&
            (bound->readAttempted || bound->creationWorkflow || bound->initializingTarget))
            throw live::LayerOperationRejected(OMNI_USD_LIVE_UNSUPPORTED);
        bound->readAttempted = true; // Never reinterpret a failed Read as first Save/create.
        if (metadataOnly || !layer->GetFileFormatArguments().empty())
            throw OmniUsdLiveResult(OMNI_USD_LIVE_UNSUPPORTED);
        // Forced reload cannot replace the attachment or discard accepted work.
        // The stream already supplies current content; re-opening is explicit.
        if (const auto existing = bound->binding->attachment()) {
            const auto state = existing->progress();
            if (state.result != OMNI_USD_LIVE_OK) throw state.result;
            return true;
        }
        auto publication = _PublicationFor(layer);
        auto attachment = live::SceneAttachment::prepareExisting(live::getRuntime(), path, std::move(publication),
            journalLimits(), progressObserver());
        _Attach(layer, attachment);
        const auto result = attachment->activatePublication();
        if (result != OMNI_USD_LIVE_OK) throw result;
        return true;
    });
}
bool OmniUsdLiveNativeFileFormat::_ReadDetached(SdfLayer*, const std::string&, bool) const {
    report(OMNI_USD_LIVE_UNSUPPORTED);
    return false;
}
bool OmniUsdLiveNativeFileFormat::WriteToFile(const SdfLayer& source, const std::string& path,
    const std::string& comment, const FileFormatArguments& args) const {
    return guarded<bool>([&] {
        if (!comment.empty() || !args.empty()) throw OmniUsdLiveResult(OMNI_USD_LIVE_UNSUPPORTED);
        const auto runtime = live::getRuntime();
        if (!runtime) throw OmniUsdLiveResult(OMNI_USD_LIVE_NOT_INITIALIZED);
        const CreationDeadline deadline{
            creationNow() + std::chrono::milliseconds(runtime->options.values.initial_open_timeout_ms)};
        live::MetadataDeadlineScope metadataBudget(deadline.at);
        auto retained = live::reserveUsdCreation();
        // A closing backing of the destination is waited for within the deadline.
        auto prepared = live::UsdLayerCreation::prepare(runtime, source, path, creationLimits(), {}, deadline.at);
        retained.install(prepared); // No Create can precede process-owned retention.
        auto& workflow = retained.workflow();
        // The hook's deadline also bounds every reissue of the creation.
        workflow.creation()->boundReissues(deadline.at);
        try {
            (void)deadline.remaining();
            auto result = workflow.start();
            if (result != OMNI_USD_LIVE_OK) throw result;
            result = workflow.creation()->wait(deadline.remaining());
            if (result != OMNI_USD_LIVE_OK) throw result;
            const auto receipt = workflow.creation()->snapshot();
            if (receipt.phase != live::SceneCreation::Phase::Confirmed)
                throw receipt.result == OMNI_USD_LIVE_OK ? OmniUsdLiveResult(OMNI_USD_LIVE_INTERNAL_ERROR) :
                                                           receipt.result;
            result = workflow.prepareAttachment(deadline.remaining());
            if (result != OMNI_USD_LIVE_OK) throw result;
            (void)deadline.remaining();
            // The deadline bounds every step before activation. Once
            // activate() returns OK the creation is confirmed and published,
            // and nothing after it fails the layer.
            result = workflow.activate();
            if (result != OMNI_USD_LIVE_OK) throw result;
        } catch (...) {
            const auto failure = creationException();
            (void)workflow.failHook(failure);
            throw failure;
        }
        // Every failure above retains the workflow. After activation a host
        // job hands the destination to its layer and frees the slot; no host
        // reference holds that layer, so it then closes.
        queueExportAdoption(std::move(retained));
        return true;
    });
}
bool OmniUsdLiveNativeFileFormat::SaveToFile(const SdfLayer& layer, const std::string& path, const std::string& comment,
    const FileFormatArguments& args) const {
    return guarded<bool>([&] {
        const auto* bound = BoundLayer::from(&layer);
        if (!bound || !args.empty()) throw OmniUsdLiveResult(OMNI_USD_LIVE_UNSUPPORTED);
        if (!bound->hostContext) throw OmniUsdLiveResult(OMNI_USD_LIVE_NOT_INITIALIZED);
        if (!bound->hostContext->isCurrent()) {
            // A FAILED layer reports its cause; its edits are never admitted.
            if (const auto failed = bound->failedLayerCause(); failed != OMNI_USD_LIVE_OK) throw failed;
            // Only the host admits recorded edits: a flush job retries admission there.
            if (bound->binding->attachment() && bound->binding->hasPendingLocalEdits())
                const_cast<BoundLayer*>(bound)->scheduleFlush();
            throw OmniUsdLiveResult(OMNI_USD_LIVE_BUSY);
        }
        // Save is only for this existing destination. Export/replacement must
        // never be smuggled through an existing attachment's successful drain.
        if (live::parseUri(path).canonical != live::parseUri(layer.GetResolvedPath().GetPathString()).canonical)
            throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
        const auto attachment = bound->binding->attachment();
        if (!attachment) {
            if (bound->readAttempted || bound->creationWorkflow || !comment.empty())
                throw OmniUsdLiveResult(OMNI_USD_LIVE_UNSUPPORTED);
            const auto runtime = live::getRuntime();
            if (!runtime) throw OmniUsdLiveResult(OMNI_USD_LIVE_NOT_INITIALIZED);
            const CreationDeadline deadline{
                creationNow() + std::chrono::milliseconds(runtime->options.values.initial_open_timeout_ms)};
            live::MetadataDeadlineScope metadataBudget(deadline.at);
            auto job = std::make_shared<InitializingFinalization>();
            job->deadline = deadline.at;
            job->lease = live::reserveUsdCreation();
            // USD's const hook borrows its mutable native target. Never retain
            // a strong target across this hook or register another USD layer.
            auto& target = const_cast<SdfLayer&>(layer);
            auto prepared = live::UsdLayerCreation::prepareInitializing(runtime, target, creationLimits());
            job->lease.install(prepared);
            auto& workflow = job->lease.workflow();
            // The hook's deadline also bounds every reissue of the creation.
            workflow.creation()->boundReissues(deadline.at);
            try {
                (void)deadline.remaining();
                auto result = workflow.startInitializing(target);
                if (result != OMNI_USD_LIVE_OK) throw result;
                result = workflow.creation()->wait(deadline.remaining());
                if (result != OMNI_USD_LIVE_OK) throw result;
                const auto receipt = workflow.creation()->snapshot();
                if (receipt.phase != live::SceneCreation::Phase::Confirmed)
                    throw receipt.result == OMNI_USD_LIVE_OK ? OmniUsdLiveResult(OMNI_USD_LIVE_INTERNAL_ERROR) :
                                                               receipt.result;
                result = workflow.prepareInitializingAttachment(target, deadline.remaining());
                if (result != OMNI_USD_LIVE_OK) throw result;
                job->attachment = workflow.attachment();
                (void)deadline.remaining();
                // Its captured lease excludes transfer through post-USD
                // finalization and final callable retirement. Inline dispatch
                // is rejected without promoting the unfinished target. Once
                // the host accepts the finalizer, only the finalizer applies
                // the opening deadline, before it completes opening.
                bound->hostContext->dispatch([job] { job->run(); });
                if (job->abandoned.load()) throw OmniUsdLiveResult(OMNI_USD_LIVE_BUSY);
                if (job.use_count() == 1) throw OmniUsdLiveResult(OMNI_USD_LIVE_CANCELLED);
                job->hookActive.store(false);
                return true;
            } catch (...) {
                const auto failure = creationException();
                job->abandoned.store(true);
                (void)workflow.failHook(failure);
                throw failure;
            }
        }
        // Admit recorded host edits first so the drain below covers them; a
        // save never drains around unadmitted host edits.
        auto* mutableBound = const_cast<BoundLayer*>(bound);
        bool issued = false;
        const auto flushed = mutableBound->flushLocalEdits(&issued);
        if (flushed == BoundLayer::FlushOutcome::Failed) throw failedCause(*bound);
        if (flushed == BoundLayer::FlushOutcome::Deferred) throw OmniUsdLiveResult(OMNI_USD_LIVE_BUSY);
        // A save may run inside the host's change block: queue the install.
        if (mutableBound->deferredInstallDue(flushed, issued)) mutableBound->scheduleFlush();
        // This host is also the publication executor. Never wait for its own
        // queued work, run callbacks here, or convert pending work into success.
        const auto result = attachment->save(0, false);
        if (result != OMNI_USD_LIVE_OK) throw result;
        return true;
    });
}
bool OmniUsdLiveNativeFileFormat::WriteToString(const SdfLayer& layer, std::string* text,
    const std::string& comment) const {
    return guarded<bool>([&] {
        const auto usda = SdfFileFormat::FindById(TfToken("usda"));
        return usda && usda->WriteToString(layer, text, comment);
    });
}
bool OmniUsdLiveNativeFileFormat::WriteToStream(const SdfSpecHandle& spec, std::ostream& out, size_t indent) const {
    return guarded<bool>([&] {
        const auto usda = SdfFileFormat::FindById(TfToken("usda"));
        return usda && usda->WriteToStream(spec, out, indent);
    });
}
PXR_NAMESPACE_CLOSE_SCOPE
