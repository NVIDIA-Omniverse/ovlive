// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "UsdLayerCreation.h"
#include "BoundLayer.h"

namespace omni::usd::live {
using namespace fileformat;
namespace {
struct CreationNoticeBlock {
    std::optional<PXR_NS::SdfChangeBlock> block{std::in_place};
    ~CreationNoticeBlock() {
        // SetIdentifier resolves metadata synchronously before it closes its
        // nested change block. Let those finite waits complete, but defer all
        // notices to this outer block and disallow host self-waits in them.
        // This also protects notice delivery during exception unwinding.
        live::DispatchExecutor::NonblockingScope nonblocking;
        block.reset();
    }
};
}
struct UsdLayerCreation::State {
    // Declared first so executor/runtime/provenance outlive layer destruction.
    std::shared_ptr<const Host> context;
    std::shared_ptr<Runtime> runtime;
    std::shared_ptr<SceneCreation> creation;
    std::shared_ptr<UsdCreationWorkflow> workflow;
    Limits limits;
    std::weak_ptr<UsdLayerBinding> targetBinding;
    bool initializing = false;
    PXR_NS::SdfLayerRefPtr layer;
    std::shared_ptr<SceneAttachment> attachment;
    bool openingAttempted = false, activated = false, closing = false, activeCall = false, openingAbandoned = false;
    std::atomic_flag startActive = ATOMIC_FLAG_INIT;
    OmniUsdLiveResult stop() noexcept;
    OmniUsdLiveResult openingFailure = OMNI_USD_LIVE_OK;
    struct Call {
        State& state;
        explicit Call(State& value) : state(value) {
            state.activeCall = true;
            if (state.workflow) state.workflow->begin();
        }
        ~Call() {
            state.activeCall = false;
            if (state.workflow)
                state.workflow->finish({state.attachment, state.openingAttempted, state.activated, state.closing, false,
                    state.openingAbandoned});
        }
    };
    void requireHost() const {
        if (!context || !context->isCurrent()) throw OmniUsdLiveResult(OMNI_USD_LIVE_BUSY);
    }
    OmniUsdLiveResult failOpening(OmniUsdLiveResult result) noexcept {
        if (openingFailure == OMNI_USD_LIVE_OK) openingFailure = result;
        if (attachment) attachment->failPublication(openingFailure);
        else (void)creation->status()->failOpening(openingFailure);
        return openingFailure;
    }
};
UsdLayerCreation::UsdLayerCreation(std::unique_ptr<State> state) : state_(std::move(state)) {}
std::shared_ptr<UsdLayerCreation> UsdLayerCreation::prepare(std::shared_ptr<Runtime> runtime,
    OmniUsdLiveEndpoint endpoint, std::string session, std::string layer, std::string seedBytes, Limits limits) try {
    auto state = std::make_unique<State>();
    state->context = configuredHost();
    if (!state->context) throw OmniUsdLiveResult(OMNI_USD_LIVE_NOT_INITIALIZED);
    state->requireHost();
    return prepareOwned(std::move(state), std::move(runtime), endpoint, std::move(session), std::move(layer),
        std::move(seedBytes), std::move(limits));
} catch (...) { throw creationException(); }
std::shared_ptr<UsdLayerCreation> UsdLayerCreation::prepare(std::shared_ptr<Runtime> runtime,
    const PXR_NS::SdfLayer& source, const std::string& uri, Limits limits, std::shared_ptr<const void> expectedContext,
    std::optional<std::chrono::steady_clock::time_point> admissionDeadline) {
    return prepareSource(std::move(runtime), &source, uri, std::move(limits), std::move(expectedContext), nullptr,
        admissionDeadline);
}
std::shared_ptr<UsdLayerCreation> UsdLayerCreation::prepareInitializing(std::shared_ptr<Runtime> runtime,
    PXR_NS::SdfLayer& target, Limits limits) {
    return prepareSource(std::move(runtime), &target, target.GetIdentifier(), std::move(limits), {}, &target);
}
std::shared_ptr<UsdLayerCreation> UsdLayerCreation::prepareSource(std::shared_ptr<Runtime> runtime,
    const PXR_NS::SdfLayer* source, const std::string& uri, Limits limits, std::shared_ptr<const void> expectedContext,
    PXR_NS::SdfLayer* target, std::optional<std::chrono::steady_clock::time_point> admissionDeadline) try {
    auto state = std::make_unique<State>();
    state->context = configuredHost();
    if (expectedContext && state->context != expectedContext) throw OmniUsdLiveResult(OMNI_USD_LIVE_NOT_INITIALIZED);
    if (!state->context || !runtime) throw OmniUsdLiveResult(OMNI_USD_LIVE_NOT_INITIALIZED);
    state->requireHost();
    if (DispatchExecutor::isWorker()) throw OmniUsdLiveResult(OMNI_USD_LIVE_BUSY);
    const auto parsed = parseUri(uri);
    OmniUsdLiveEndpoint endpoint = 0;
    {
        std::lock_guard<std::mutex> lock(runtime->mutex);
        if (runtime->closing) throw OmniUsdLiveResult(OMNI_USD_LIVE_BUSY);
        const auto found = runtime->authorities.find(parsed.authority.canonical);
        if (found == runtime->authorities.end()) throw OmniUsdLiveResult(OMNI_USD_LIVE_NOT_FOUND);
        endpoint = found->second;
    }
    if (parsed.layerName.size() < 5 || parsed.layerName.compare(parsed.layerName.size() - 5, 5, ".live"))
        throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
    if (const auto* bound = BoundLayer::from(source))
        if (!bound->hostContext || !bound->hostContext->isCurrent()) throw OmniUsdLiveResult(OMNI_USD_LIVE_BUSY);
    PXR_NS::TfErrorMark errors;
    try {
        const auto data = PXR_NS::OmniUsdLiveNativeFileFormat::_GetLayerData(*source);
        if (!data) throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
        std::string format;
        if (const auto* bound = BoundLayer::from(source); bound && bound->hostState)
            format = bound->hostState->provenance()->sourceFormat();
        else {
            const auto fileFormat = source->GetFileFormat();
            if (!fileFormat) throw OmniUsdLiveResult(OMNI_USD_LIVE_UNSUPPORTED);
            format = fileFormat->GetFormatId().GetString();
        }
        // A seed is one upload; its own work default sizes for that, beside the journal's replay work.
        SceneSeedLimits seedLimits;
        seedLimits.scene = limits.scene;
        seedLimits.entries = limits.authoring.entries;
        seedLimits.maxWork = std::max(seedLimits.maxWork, limits.maxReplayWork);
        const auto cap = std::min<uint64_t>(
            {runtime->options.values.max_update_bytes, runtime->options.values.max_pending_bytes_per_endpoint,
                limits.maxPendingBytes, seedLimits.envelope.content.maxBytes, std::numeric_limits<size_t>::max()});
        seedLimits.envelope.content.maxBytes = static_cast<size_t>(cap);
        auto compiled = compileSceneSeed(*data, format, seedLimits);
        if (!errors.IsClean()) throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
        compiled.scene.reset(); // Preparation validates owned wire with its own charged retention.
        return prepareOwned(std::move(state), std::move(runtime), endpoint, parsed.sessionId, parsed.layerName,
            std::move(compiled.bytes), std::move(limits), target, admissionDeadline);
    } catch (...) {
        errors.Clear();
        throw;
    }
} catch (const DeltaEntryLimitExceeded& refused) {
    // Nothing was sent. A USD runtime error names the destination and the
    // counts, as the service's DELTA_TOO_LARGE ErrorInfo would.
    try {
        using namespace PXR_NS; // TF_RUNTIME_ERROR names Tf symbols unqualified.
        const auto text = "Live layer creation of " + uri +
            ": the seed exceeds the Live Edit service's entry limits (" + refused.describe() + "); nothing was sent";
        TF_RUNTIME_ERROR("%s (result=%u)", text.c_str(), unsigned(OMNI_USD_LIVE_RESOURCE_EXHAUSTED));
    } catch (...) {} // The typed result stands without its description.
    throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
} catch (...) { throw creationException(); }
std::shared_ptr<UsdLayerCreation> UsdLayerCreation::prepareOwned(std::unique_ptr<State> state,
    std::shared_ptr<Runtime> runtime, OmniUsdLiveEndpoint endpoint, std::string session, std::string layer,
    std::string seedBytes, Limits limits, PXR_NS::SdfLayer* target,
    std::optional<std::chrono::steady_clock::time_point> admissionDeadline) {
    if (layer.size() < 5 || layer.compare(layer.size() - 5, 5, ".live"))
        throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
    state->runtime = runtime;
    state->limits = limits;
    state->creation = SceneCreation::prepare(std::move(runtime), endpoint, std::move(session), std::move(layer),
        std::move(seedBytes), limits, admissionDeadline);
    state->workflow = std::make_shared<UsdCreationWorkflow>(state->runtime, state->creation);
    // Allocate the returned owner before registering an anonymous USD layer.
    // The canonical identifier must not expose an empty successfully-opened
    // layer before bootstrap. No start(), FindOrOpen, Read or WriteToFile.
    auto result = std::shared_ptr<UsdLayerCreation>(new UsdLayerCreation(std::move(state)));
    auto& owned = *result->state_;
    if (target) {
        // This runs under USD's CreateNew registry lock. All preparation above
        // is native-only; never CreateAnonymous, Find or SetIdentifier here.
        prepareInitializingCreationTarget(*target, owned.workflow);
        target->SetPermissionToEdit(false); // A creation layer is read-only until activation.
        const auto* bound = BoundLayer::from(target);
        if (!bound) throw OmniUsdLiveResult(OMNI_USD_LIVE_INTERNAL_ERROR);
        owned.targetBinding = bound->binding;
        owned.initializing = true;
        return result;
    }
    PXR_NS::TfErrorMark errors;
    try {
        const auto format = PXR_NS::SdfFileFormat::FindById(PXR_NS::TfToken("live"));
        if (!format || !dynamic_cast<const PXR_NS::OmniUsdLiveNativeFileFormat*>(format.operator->()) ||
            !errors.IsClean())
            throw OmniUsdLiveResult(OMNI_USD_LIVE_UNSUPPORTED);
        owned.layer = PXR_NS::SdfLayer::CreateAnonymous("live-creation.live", format);
        if (!owned.layer || !errors.IsClean()) throw OmniUsdLiveResult(OMNI_USD_LIVE_INTERNAL_ERROR);
        auto* bound = BoundLayer::from(owned.layer.operator->());
        if (!bound || bound->hostContext != owned.context || !owned.layer->IsAnonymous() ||
            !owned.layer->GetResolvedPath().empty())
            throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
        bound->ensureSetup();
        owned.layer->SetPermissionToEdit(false); // A creation layer is read-only until activation.
        if (!errors.IsClean()) throw OmniUsdLiveResult(OMNI_USD_LIVE_INTERNAL_ERROR);
        bound->creationWorkflow = owned.workflow;
    } catch (...) {
        errors.Clear();
        throw;
    }
    return result;
}
UsdLayerCreation::~UsdLayerCreation() { (void)close(); }
void UsdLayerCreation::requireTransferHost() const {
    state_->requireHost();
    if (state_->activeCall || DispatchExecutor::isWorker()) throw OmniUsdLiveResult(OMNI_USD_LIVE_BUSY);
}
OmniUsdLiveResult UsdLayerCreation::failHook(OmniUsdLiveResult result) noexcept {
    try {
        requireTransferHost();
        if (result == OMNI_USD_LIVE_OK || !definedResult(result)) return OMNI_USD_LIVE_INVALID_ARGUMENT;
        auto& state = *state_;
        // Activation confirmed and published the creation; a hook failure
        // after it never fails the layer.
        if (state.activated) return OMNI_USD_LIVE_INVALID_ARGUMENT;
        const auto outcome = state.creation->snapshot();
        if (!state.attachment &&
            (outcome.phase == SceneCreation::Phase::Prepared || outcome.phase == SceneCreation::Phase::Creating ||
                outcome.phase == SceneCreation::Phase::Confirmed) &&
            result == OMNI_USD_LIVE_OUTCOME_UNKNOWN)
            return OMNI_USD_LIVE_INVALID_ARGUMENT;
        State::Call call(state);
        if (state.closing) return OMNI_USD_LIVE_CANCELLED;
        if (state.attachment) {
            (void)state.failOpening(result);
            return OMNI_USD_LIVE_OK;
        }
        if (state.openingFailure == OMNI_USD_LIVE_OK) state.openingFailure = result;
        if (outcome.phase == SceneCreation::Phase::Prepared) return state.creation->failBeforeStart(result);
        if (outcome.phase == SceneCreation::Phase::Creating || outcome.phase == SceneCreation::Phase::Confirmed) {
            state.openingAbandoned = true;
            state.creation->abandonReissues();
            return state.creation->status()->abandonOpening(result);
        }
        return OMNI_USD_LIVE_OK; // Keep original finite rejection/uncertainty.
    } catch (...) { return creationException(); }
}
OmniUsdLiveResult UsdLayerCreation::start() noexcept { return startOn(nullptr); }
OmniUsdLiveResult UsdLayerCreation::startInitializing(PXR_NS::SdfLayer& target) noexcept { return startOn(&target); }
OmniUsdLiveResult UsdLayerCreation::startOn(PXR_NS::SdfLayer* target) noexcept {
    // Acquire before foreign isCurrent: even a recursively invoked start must
    // return BUSY without touching USD ownership or consuming the attempt.
    struct Entry {
        std::atomic_flag& flag;
        const bool acquired;
        explicit Entry(std::atomic_flag& flag) : flag(flag), acquired(!flag.test_and_set(std::memory_order_acquire)) {}
        ~Entry() {
            if (acquired) flag.clear(std::memory_order_release);
        }
    } entry(state_->startActive);
    if (!entry.acquired) return OMNI_USD_LIVE_BUSY;
    try {
        requireTransferHost();
        State::Call call(*state_);
        // A consumed attempt needs no surviving USD target. In particular,
        // retained failed CreateNew workflows must never look retryable/BUSY.
        if (state_->creation->snapshot().started) return state_->creation->start();
        if (state_->closing) return OMNI_USD_LIVE_CANCELLED;
        if (state_->initializing) {
            if (!target) return OMNI_USD_LIVE_BUSY;
        } else {
            if (target) return OMNI_USD_LIVE_INVALID_ARGUMENT;
            target = state_->layer.operator->();
        }
        auto* bound = BoundLayer::from(target);
        if (!bound || bound->creationWorkflow != state_->workflow || bound->hostContext != state_->context ||
            (state_->initializing && bound->binding != state_->targetBinding.lock()))
            return OMNI_USD_LIVE_INVALID_ARGUMENT;
        bound->ensureSetup(); // Observe replacement even when its USD notice was deferred.
        return state_->creation->start();
    } catch (...) { return creationException(); }
}
const std::shared_ptr<SceneCreation>& UsdLayerCreation::creation() const {
    state_->requireHost();
    return state_->creation;
}
std::shared_ptr<SceneAttachment> UsdLayerCreation::attachment() const {
    state_->requireHost();
    return state_->attachment;
}
PXR_NS::SdfLayerRefPtr UsdLayerCreation::layer() const {
    state_->requireHost();
    return state_->activated ? state_->layer : PXR_NS::SdfLayerRefPtr{};
}
OmniUsdLiveResult UsdLayerCreation::prepareAttachment(uint32_t timeoutMs) noexcept {
    try {
        state_->requireHost();
        if (state_->initializing) return OMNI_USD_LIVE_BUSY;
        return prepareAttachmentOn(state_->layer.operator->(), timeoutMs);
    } catch (...) { return creationException(); }
}
OmniUsdLiveResult UsdLayerCreation::prepareInitializingAttachment(PXR_NS::SdfLayer& target,
    uint32_t timeoutMs) noexcept {
    try {
        state_->requireHost();
        const auto* bound = BoundLayer::from(&target);
        if (!state_->initializing || !bound || bound->creationWorkflow != state_->workflow ||
            bound->binding != state_->targetBinding.lock())
            return OMNI_USD_LIVE_INVALID_ARGUMENT;
        const auto result = prepareAttachmentOn(&target, timeoutMs);
        if (result != OMNI_USD_LIVE_OK) return result;
        State::Call call(*state_);
        const auto activated = state_->attachment->activatePublication();
        return activated == OMNI_USD_LIVE_OK ? activated : state_->failOpening(activated);
    } catch (...) { return creationException(); }
}
OmniUsdLiveResult UsdLayerCreation::prepareAttachmentOn(PXR_NS::SdfLayer* target, uint32_t timeoutMs) noexcept {
    auto& state = *state_;
    bool validHost = false;
    try {
        state.requireHost();
        if (state.activeCall) return OMNI_USD_LIVE_BUSY;
        State::Call call(state);
        validHost = true;
        if (state.closing) return OMNI_USD_LIVE_CANCELLED;
        if (state.openingFailure != OMNI_USD_LIVE_OK) return state.openingFailure;
        if (!state.openingAttempted) {
            const auto outcome = state.creation->snapshot();
            if (outcome.closing) return OMNI_USD_LIVE_CANCELLED;
            if (outcome.phase == SceneCreation::Phase::Prepared || outcome.phase == SceneCreation::Phase::Creating)
                return OMNI_USD_LIVE_BUSY;
            if (outcome.phase != SceneCreation::Phase::Confirmed) return outcome.result;
            if (!outcome.quiescent) return OMNI_USD_LIVE_BUSY;
            if (!outcome.object || outcome.sequence != 1) return OMNI_USD_LIVE_INTERNAL_ERROR;
            state.openingAttempted = true;
            SceneAttachment::Options options;
            const auto& admission = state.creation->admission();
            options.session = admission->session();
            options.layer = admission->layer();
            options.object = outcome.object;
            options.admission = admission;
            options.limits = state.limits;
            options.status = state.creation->status();
            options.holdOpening = true;
            options.deferInitialPublication = true;
            options.publication = PXR_NS::OmniUsdLiveNativeFileFormat::_PublicationFor(target);
            // Publish ownership before any setup/wait that can fail. The
            // one-shot admission can never be used to create a second actor.
            state.attachment = SceneAttachment::open(state.runtime, admission->endpoint(), std::move(options),
                progressObserver());
            // The workflow retains this attachment even if binding/identifier
            // setup subsequently fails.
            PXR_NS::OmniUsdLiveNativeFileFormat::_Attach(target, state.attachment, false);
        }
        return state.attachment->waitForPrepared(timeoutMs);
    } catch (...) {
        const auto result = creationException();
        if (validHost && state.openingAttempted) state.failOpening(result);
        return result;
    }
}
OmniUsdLiveResult UsdLayerCreation::activate() noexcept {
    auto& state = *state_;
    try {
        state.requireHost();
        if (state.activeCall) return OMNI_USD_LIVE_BUSY;
        if (state.initializing) return finalizeInitializing();
        State::Call call(state);
        if (state.closing) return OMNI_USD_LIVE_CANCELLED;
        if (state.openingFailure != OMNI_USD_LIVE_OK) return state.openingFailure;
        if (!state.attachment) return OMNI_USD_LIVE_BUSY;
        const auto result = state.attachment->activatePublication();
        if (result != OMNI_USD_LIVE_OK) {
            if (result != OMNI_USD_LIVE_BUSY) return state.failOpening(result);
            return result;
        }
        if (state.activated) return result;
        PXR_NS::TfErrorMark errors;
        try {
            const auto& uri = state.creation->admission()->canonicalUri();
            const auto existing = PXR_NS::SdfLayer::Find(uri);
            if (!errors.IsClean()) throw OmniUsdLiveResult(OMNI_USD_LIVE_INTERNAL_ERROR);
            if (existing && existing != state.layer) throw OmniUsdLiveResult(OMNI_USD_LIVE_ALREADY_EXISTS);
            // The SDK checks again under its layer-registry write lock before
            // changing identity, so this precheck is not a collision barrier.
            // SetIdentifier may diagnose/throw after renaming (e.g. resolver
            // metadata or notices); retain that exact layer on every failure.
            {
                CreationNoticeBlock notices;
                state.layer->SetIdentifier(uri);
            }
            if (!errors.IsClean() || state.layer->GetIdentifier() != uri ||
                state.layer->GetResolvedPath().GetPathString() != uri || state.layer->IsAnonymous())
                throw OmniUsdLiveResult(OMNI_USD_LIVE_INTERNAL_ERROR);
            const auto* bound = BoundLayer::from(state.layer.operator->());
            if (!bound) throw OmniUsdLiveResult(OMNI_USD_LIVE_INTERNAL_ERROR);
            bound->observe();
            const auto completed = state.attachment->completeOpening();
            if (completed != OMNI_USD_LIVE_OK) throw completed;
            state.layer->SetPermissionToEdit(true);
            state.activated = true;
            return OMNI_USD_LIVE_OK;
        } catch (...) {
            errors.Clear();
            return state.failOpening(creationException());
        }
    } catch (...) { return creationException(); }
}
OmniUsdLiveResult UsdLayerCreation::finalizeInitializing(OmniUsdLiveResult completion) noexcept {
    auto& state = *state_;
    bool validHost = false;
    try {
        state.requireHost();
        validHost = true;
        if (state.activeCall) return OMNI_USD_LIVE_BUSY;
        if (!state.initializing) return state.activated ? OMNI_USD_LIVE_OK : OMNI_USD_LIVE_INVALID_ARGUMENT;
        State::Call call(state);
        if (state.closing) return OMNI_USD_LIVE_CANCELLED;
        if (!state.attachment)
            return state.openingFailure != OMNI_USD_LIVE_OK ? state.openingFailure :
                                                              state.failOpening(OMNI_USD_LIVE_INTERNAL_ERROR);
        const auto binding = state.targetBinding.lock();
        if (!binding) return state.failOpening(OMNI_USD_LIVE_CANCELLED);
        // The dispatcher guarantees the outer USD call returned. A failed call
        // destroyed the target; never use Find as a readiness probe.
        auto layer = binding->pinLayer();
        auto* bound = BoundLayer::from(layer.operator->());
        const auto& uri = state.creation->admission()->canonicalUri();
        if (!bound || bound->creationWorkflow != state.workflow || !bound->initializingTarget || layer->IsAnonymous() ||
            layer->GetIdentifier() != uri || layer->GetResolvedPath().GetPathString() != uri)
            return state.failOpening(OMNI_USD_LIVE_INTERNAL_ERROR);
        // Retain the successfully initialized target even if opening fails below.
        state.layer = std::move(layer);
        bound->initializingTarget = false;
        state.initializing = false;
        bound->ensureSetup();
        if (state.openingFailure != OMNI_USD_LIVE_OK) return state.openingFailure;
        if (completion != OMNI_USD_LIVE_OK) return state.failOpening(completion);
        bound->observe();
        const auto completed = state.attachment->completeOpening();
        if (completed != OMNI_USD_LIVE_OK) return state.failOpening(completed);
        state.layer->SetPermissionToEdit(true);
        state.activated = true;
        return OMNI_USD_LIVE_OK;
    } catch (...) {
        const auto failure = creationException();
        return validHost ? state.failOpening(failure) : failure;
    }
}
OmniUsdLiveResult UsdLayerCreation::State::stop() noexcept {
    try {
        // Poll both even if one still has retained work. No host self-wait and
        // no owner is discarded merely because cancellation was requested.
        const auto creationResult = creation->close(0);
        const auto attachmentResult = attachment ? attachment->close(0) : OmniUsdLiveResult(OMNI_USD_LIVE_OK);
        if (creationResult != OMNI_USD_LIVE_OK && creationResult != OMNI_USD_LIVE_BUSY &&
            creationResult != OMNI_USD_LIVE_TIMEOUT)
            return creationResult;
        if (attachmentResult != OMNI_USD_LIVE_OK && attachmentResult != OMNI_USD_LIVE_BUSY &&
            attachmentResult != OMNI_USD_LIVE_TIMEOUT)
            return attachmentResult;
        return creationResult == OMNI_USD_LIVE_OK && attachmentResult == OMNI_USD_LIVE_OK ? OMNI_USD_LIVE_OK :
                                                                                            OMNI_USD_LIVE_BUSY;
    } catch (...) { return creationException(); }
}
OmniUsdLiveResult UsdLayerCreation::close() noexcept {
    auto& state = *state_;
    try {
        state.requireHost();
        if (state.activeCall) return OMNI_USD_LIVE_BUSY;
        State::Call call(state);
        state.closing = true;
        return state.stop();
    } catch (...) { return creationException(); }
}
OmniUsdLiveResult UsdLayerCreation::adoptIntoLayer() noexcept {
    auto& state = *state_;
    PXR_NS::SdfLayerRefPtr layer;
    try {
        requireTransferHost();
        if (state.closing) return OMNI_USD_LIVE_CANCELLED;
        if (!state.activated || state.initializing || !state.layer || !state.attachment)
            return OMNI_USD_LIVE_INVALID_ARGUMENT;
        // The binding already owns this attachment (_Attach at opening); it
        // closes it when the layer's last reference goes.
        auto* bound = BoundLayer::from(state.layer.operator->());
        if (!bound || bound->binding->attachment() != state.attachment) return OMNI_USD_LIVE_INVALID_ARGUMENT;
        State::Call call(state);
        state.attachment.reset();
        layer = std::move(state.layer);
    } catch (...) { return creationException(); }
    // Outside the workflow call: this may be the last reference, whose
    // destruction closes the attachment through the binding.
    layer.Reset();
    return OMNI_USD_LIVE_OK;
}
OmniUsdLiveResult UsdLayerCreation::waitForClose(uint32_t timeoutMs) noexcept {
    try {
        using Clock = std::chrono::steady_clock;
        const auto deadline = Clock::now() + std::chrono::milliseconds(timeoutMs);
        const auto snapshot = state_->workflow->snapshot();
        if (snapshot.activeCall) return OMNI_USD_LIVE_BUSY;
        if (!snapshot.closing) return OMNI_USD_LIVE_INVALID_ARGUMENT;
        const bool blocking = timeoutMs && !DispatchExecutor::isWorker() && !state_->context->isCurrent();
        auto remaining = [&] { return blocking ? millisecondsLeft(deadline, Clock::now()) : uint32_t(0); };
        auto result = state_->creation->close(remaining());
        if (result == OMNI_USD_LIVE_OK && snapshot.attachment) result = snapshot.attachment->close(remaining());
        if (!blocking && result == OMNI_USD_LIVE_TIMEOUT) result = OMNI_USD_LIVE_BUSY;
        if (result == OMNI_USD_LIVE_OK && blocking && Clock::now() >= deadline) result = OMNI_USD_LIVE_TIMEOUT;
        return result;
    } catch (...) { return creationException(); }
}
namespace fileformat {
InitializingFinalization::~InitializingFinalization() {
        // Closure retirement may occur off-host. Fail only the native actor;
        // the process slot keeps every USD owner on its original host.
    if (attachment && !completed.load() && !abandoned.load()) attachment->failPublication(OMNI_USD_LIVE_CANCELLED);
}
void InitializingFinalization::run() noexcept {
    if (abandoned.load() || started.exchange(true)) return;
    if (hookActive.load()) {
        abandoned.store(true);
        attachment->failPublication(OMNI_USD_LIVE_BUSY);
        return;
    }
    try {
        auto& workflow = lease.workflow();
        workflow.requireTransferHost();
            // Finalize the completed SDK target before reporting a delayed
            // finalizer's deadline failure.
        const auto result = workflow.finalizeInitializing(
            creationNow() >= deadline ? OMNI_USD_LIVE_TIMEOUT : OMNI_USD_LIVE_OK);
        if (result != OMNI_USD_LIVE_OK) report(result);
        else {
            // Hand the activated layer its attachment and free the slot; a
            // failed adoption keeps the workflow retained.
            if (const auto adopted = lease.retireAdopted(); adopted != OMNI_USD_LIVE_OK) report(adopted);
        }
    } catch (...) { attachment->failPublication(creationException()); }
    completed.store(true);
}
void queueExportAdoption(live::UsdCreationLease lease) noexcept {
    struct Adoption {
        live::UsdCreationLease lease;
        std::atomic<bool> hookActive{true}, settled{false};
        // A job the host drops without running leaves the workflow retained.
        ~Adoption() {
            if (!settled.load()) report(OMNI_USD_LIVE_CANCELLED);
        }
    };
    // Every path that leaves the workflow retained reports why, as the
    // CreateNew finalizer does.
    try {
        std::shared_ptr<const Host> context;
        {
            // The workflow still holds this layer; this is never its last reference.
            const auto layer = lease.workflow().layer();
            const auto* bound = BoundLayer::from(layer.operator->());
            if (!bound || !bound->hostContext) {
                report(OMNI_USD_LIVE_NOT_INITIALIZED);
                return;
            }
            context = bound->hostContext;
        }
        auto job = std::make_shared<Adoption>();
        job->lease = std::move(lease);
        // Dropping the workflow's layer inside the outer Export would destroy
        // an SdfLayer there, so an inline run keeps the workflow retained.
        try {
            context->dispatch([job] {
                job->settled.store(true);
                if (job->hookActive.load()) {
                    report(OMNI_USD_LIVE_BUSY);
                    return;
                }
                if (const auto adopted = job->lease.retireAdopted(); adopted != OMNI_USD_LIVE_OK) report(adopted);
            });
        } catch (...) {
            job->settled.store(true); // The handler below reports the dispatch failure.
            throw;
        }
        job->hookActive.store(false);
    } catch (...) { report(creationException()); }
}
}
} // namespace omni::usd::live
