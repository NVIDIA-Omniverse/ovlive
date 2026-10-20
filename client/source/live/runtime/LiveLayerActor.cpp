// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "LiveLayerActor.h"
#include "LiveDeadline.h"
#include "LiveStatus.h"
#include "LiveDiagnostics.h"
#include "LiveExceptionResult.h"
#include <openssl/sha.h>
#include <algorithm>
#include <cassert>
#include <iterator>
#include <type_traits>
#include <utility>

namespace omni::usd::live {
namespace {
std::atomic<void (*)(void*) noexcept> afterDrainRetired{nullptr};
std::atomic<void*> afterDrainRetiredContext{nullptr};
std::atomic<void (*)(void*) noexcept> beforeCloseQuiescent{nullptr};
std::atomic<void*> beforeCloseQuiescentContext{nullptr};
std::atomic<bool (*)(void*, size_t) noexcept> beforeSettleWrites{nullptr};
std::atomic<void*> beforeSettleWritesContext{nullptr};
uint64_t applicationElapsedMs(std::chrono::steady_clock::time_point start,
    std::chrono::steady_clock::time_point end) noexcept {
    if (end <= start) return 0;
    using Duration = std::chrono::steady_clock::duration;
    using Unsigned = std::make_unsigned_t<Duration::rep>;
    const auto ticks = Unsigned(end.time_since_epoch().count()) - Unsigned(start.time_since_epoch().count());
    return std::chrono::duration_cast<std::chrono::duration<uint64_t, std::milli>>(
        std::chrono::duration<Unsigned, Duration::period>(ticks))
        .count();
}
bool terminalGate(LayerProgress::Gate gate) { return gate == LayerProgress::Gate::Failed; }
// Whether the read subscription runs: a healthy layer, or one whose failure
// stopped issuance only (design section 13).
bool receiving(const LayerProgress::Snapshot& state) { return !terminalGate(state.gate) || state.readingAfterFailure; }
bool initialApplicationReady(const LayerProgress::Snapshot& state) {
    // Progress's latest flag proves validated catch-up coverage, not completed
    // scene/host publication. A retained resume already supplies applied > 0.
    return state.initialLatestPublished && state.applied > 0;
}
bool same(const LayerProgress::Snapshot& a, const LayerProgress::Snapshot& b) {
    return a.gate == b.gate && a.result == b.result && a.generation == b.generation && a.covered == b.covered &&
        a.applied == b.applied && a.readSequence == b.readSequence && a.issued == b.issued &&
        a.retained == b.retained && a.queued == b.queued && a.readStarted == b.readStarted &&
        a.initialLatestPublished == b.initialLatestPublished && a.pendingWrites == b.pendingWrites &&
        a.pendingBytes == b.pendingBytes && a.object == b.object && a.initialApplied == b.initialApplied &&
        a.localAdmitted == b.localAdmitted && a.localRetired == b.localRetired &&
        a.readingAfterFailure == b.readingAfterFailure && a.unplaced == b.unplaced;
}
}
struct LayerActor::ApplyCompletion::State {
    enum class Phase { Queued, Running, Finished };
    const std::weak_ptr<LayerActor> actor;
    // One local publication, or ordered work in application order.
    const std::vector<LayerProgress::Work> work;
    const std::vector<std::shared_ptr<const Effect>> effect;
    const bool local;
    std::mutex mutex;
    Phase phase = Phase::Queued;
    OmniUsdLiveResult result = OMNI_USD_LIVE_OK;
    bool retired = false;
    bool ran = false, timingRecorded = false;
    Clock::time_point admittedAt{}, runAt{}, bodyEndedAt{}, retiredAt{};
    Clock::time_point (*now)(void*) noexcept;
    void* clockContext;
    State(std::weak_ptr<LayerActor> actor, std::vector<LayerProgress::Work> work,
        std::vector<std::shared_ptr<const Effect>> effect, bool local, Clock::time_point (*clock)(void*) noexcept,
        void* context)
        : actor(std::move(actor)), work(std::move(work)), effect(std::move(effect)), local(local), now(clock),
          clockContext(context) {}
    void wake() noexcept {
        if (auto owner = actor.lock()) owner->wake();
    }
    bool claim() noexcept {
        std::lock_guard<std::mutex> lock(mutex);
        if (phase != Phase::Queued) return false;
        ran = true;
        runAt = now(clockContext);
        phase = Phase::Running;
        return true;
    }
    void bodyReturned() noexcept {
        const auto captured = now(clockContext);
        std::lock_guard<std::mutex> lock(mutex);
        bodyEndedAt = captured;
    }
    void finish(OmniUsdLiveResult value, bool reject = false) noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (phase == Phase::Finished && !reject) return;
            result = value;
            phase = Phase::Finished;
        }
        wake();
    }
    void cancelQueued() noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (phase != Phase::Queued) return;
            result = OMNI_USD_LIVE_CANCELLED;
            phase = Phase::Finished;
        }
        wake();
    }
};

LayerActor::ApplyCompletion::ApplyCompletion(std::shared_ptr<State> state) : state_(std::move(state)) {}
LayerActor::ApplyCompletion::~ApplyCompletion() {
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        // Abandoning a queued owner is never successful application.
        if (state_->phase != State::Phase::Finished) {
            state_->result = OMNI_USD_LIVE_CANCELLED;
            state_->phase = State::Phase::Finished;
        }
        state_->retiredAt = state_->now(state_->clockContext);
        state_->retired = true;
    }
    state_->wake();
}
void LayerActor::ApplyCompletion::complete(OmniUsdLiveResult result) noexcept { state_->finish(result); }
void LayerActor::ApplyCompletion::run(std::function<void()> action) noexcept {
    // A body may release the caller's last owner; keep final retirement after
    // the body and destruction of its owned action captures.
    const auto self = shared_from_this();
    DispatchExecutor::NonblockingScope scope;
    if (auto actor = state_->actor.lock()) (void)actor->observeHostFailure();
    if (!state_->claim()) {
        action = {};
        return;
    }
    OmniUsdLiveResult result = OMNI_USD_LIVE_OK;
    try {
        if (!action) throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
        action();
    } catch (OmniUsdLiveResult value) { result = value; } catch (const std::bad_alloc&) {
        result = OMNI_USD_LIVE_RESOURCE_EXHAUSTED;
    } catch (const std::length_error&) {
        result = OMNI_USD_LIVE_RESOURCE_EXHAUSTED;
    } catch (const std::invalid_argument&) { result = OMNI_USD_LIVE_INVALID_ARGUMENT; } catch (...) {
        result = OMNI_USD_LIVE_INTERNAL_ERROR;
    }
    state_->bodyReturned();
    action = {};
    complete(result);
}

LayerActor::LayerActor(std::shared_ptr<Runtime> runtime, EndpointLease endpoint, Options options, Callbacks callbacks)
    : runtime_(std::move(runtime)), admission_(std::move(options.admission)), statusOwner_(std::move(options.status)),
      endpoint_(std::move(endpoint)),
      progress_({endpoint_->handle, std::move(options.session), std::move(options.layer), endpoint_->writerId,
                    options.object},
          options.resume, std::move(options.initialState), runtime_->options.values.max_requests,
          runtime_->options.values.max_synthetic_gap, options.discover),
      updateHeaderBound_(
          layerUpdateHeaderBound(progress_.identity.session, progress_.identity.layer, endpoint_->writerId)),
      readHighWater_(
          std::max<size_t>(1, runtime_->options.values.max_requests - runtime_->options.values.max_requests / 8)),
      readLowWater_(runtime_->options.values.max_requests / 2), callbacks_(std::move(callbacks)),
      hostFailure_(std::move(options.hostFailure)), bufferedEdits_(std::move(options.bufferedEdits)),
      beforeAdmission_(options.beforeAdmissionForTest), admissionContext_(options.admissionContextForTest),
      afterReservation_(options.afterReservationForTest), reservationContext_(options.reservationContextForTest),
      drain_(runtime_->drains->makeState()), activity_(drain_->prepare()),
      timingTimeoutMs_(options.timingTimeoutMsForTest),
      applicationNow_(
          options.applicationNowForTest ? options.applicationNowForTest : [](void*) noexcept { return Clock::now(); }),
      applicationClockContext_(options.applicationClockContextForTest),
      openDeadline_(Clock::now() + std::chrono::milliseconds(runtime_->options.values.initial_open_timeout_ms)),
      authObservation_(*runtime_->diagnostics, AuthWaitKind::Layer), beforeAuthGate_(options.beforeAuthGateForTest),
      authGateContext_(options.authGateContextForTest), closingRetained_(options.closingRetainedForTest),
      closingRetainedContext_(options.closingRetainedContextForTest) {
    subject_ = statusOwner_->id();
    holdOpening_ = options.holdOpening;
    if (!callbacks_.decode ||
        int(bool(callbacks_.apply)) + int(bool(callbacks_.applyAsync)) + int(bool(callbacks_.applyBatchAsync)) != 1)
        throw std::invalid_argument("layer requires exactly one application callback");
    pendingApplied_.reserve(kMaxBatch);
    pendingDrains_.reserve(kMaxBatch);
    writes_.reserve(kMaxWindow);
    if (!timingTimeoutMs_) throw std::invalid_argument("zero timing probe test timeout");
    makeLayerUri(makeSessionUri(endpoint_->options.authority.canonical, progress_.identity.session),
        progress_.identity.layer);
    std::lock_guard<std::mutex> lock(endpoint_->mutex);
    if (bufferedEdits_) {
        auto& watchers = endpoint_->pendingByteWatchers;
        watchers.erase(
            std::remove_if(watchers.begin(), watchers.end(), [](const auto& watcher) { return watcher.expired(); }),
            watchers.end());
        watchers.push_back(bufferedEdits_);
        watchingPendingBytes_ = true;
    }
    authFailure_ = attachmentAuthFailure_ = endpoint_->authFailureSerial;
    waitingAuth_ = !runtime_->auth->usable(*endpoint_, wallTimeMs());
    authDeadline_ = Clock::now() + std::chrono::milliseconds(endpoint_->options.authWaitMs);
}

std::unique_ptr<HostBufferCharge> LayerActor::hostBufferCharge() const {
    auto lease = runtime_->acquireEndpoint(endpoint_->handle);
    if (!lease) return nullptr;
    return std::make_unique<HostBufferCharge>(std::move(lease),
        runtime_->options.values.max_pending_bytes_per_endpoint);
}
HostBufferCharge::Budget LayerActor::hostBufferBudget() const noexcept {
    return HostBufferCharge::budget(*endpoint_.operator->(), runtime_->options.values.max_pending_bytes_per_endpoint);
}

std::shared_ptr<LayerActor> LayerActor::open(std::shared_ptr<Runtime> runtime, OmniUsdLiveEndpoint endpoint,
    Options options, Callbacks callbacks) {
    if (!runtime) throw OmniUsdLiveResult(OMNI_USD_LIVE_NOT_INITIALIZED);
    if (!options.admission)
        options.admission = runtime->layerAdmissions->reserve(endpoint, options.session, options.layer);
    if (options.status) options.status->validate(runtime, options.admission);
    else options.status = LayerStatus::create(runtime, options.admission);
    const auto status = options.status;
    const auto eligible = status->canHandoff();
    if (eligible != OMNI_USD_LIVE_OK) throw eligible;
    // Invalid or duplicate attempts must not publish failure into the original
    // live subject. Only this attempt's successful canonical claim owns errors.
    runtime->layerAdmissions->claim(options.admission, endpoint, options.session, options.layer);
    std::shared_ptr<LayerActor> actor;
    bool reserved = false;
    try {
        auto lease = runtime->acquireEndpoint(endpoint);
        if (!lease) throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_HANDLE);
        actor = std::shared_ptr<LayerActor>(
            new LayerActor(runtime, std::move(lease), std::move(options), std::move(callbacks)));
        runtime->auth->reserve(actor);
        reserved = true;
        const auto handoff = status->handoff();
        if (handoff != OMNI_USD_LIVE_OK) throw handoff;
        actor->publishStatus();
        // The returned handle owns a separate count: its last release closes
        // the actor, and the scheduler's internal reference keeps it until it
        // stops. shared_from_this stays bound to the internal control block.
        // It is built before activation so that no throwing step follows it;
        // a handle released on the error path closes an unactivated actor.
        std::shared_ptr<LayerActor> handle(actor.get(), [internal = actor](LayerActor*) noexcept {
            try {
                (void)internal->close(0);
            } catch (...) {}
        });
        runtime->auth->activate(*actor);
        return handle;
    } catch (...) {
        const auto result = currentExceptionResult();
        if (reserved) runtime->auth->discard(*actor);
        (void)status->failOpening(result);
        throw result;
    }
}

LayerActor::~LayerActor() {
    // A close publishes status and ends any warning interval; this covers an
    // actor that never published after entering one.
    if (applyLagging_) runtime_->diagnostics->applyLagChanged(false);
}

void LayerActor::setAfterDrainRetiredForTest(void (*hook)(void*) noexcept, void* context) noexcept {
    // Release/acquire pairs preserve a hook publication with its context.
    if (!hook) afterDrainRetired.store(nullptr, std::memory_order_release);
    afterDrainRetiredContext.store(context, std::memory_order_release);
    if (hook) afterDrainRetired.store(hook, std::memory_order_release);
}

void LayerActor::setBeforeCloseQuiescentForTest(void (*hook)(void*) noexcept, void* context) noexcept {
    if (!hook) beforeCloseQuiescent.store(nullptr, std::memory_order_release);
    beforeCloseQuiescentContext.store(context, std::memory_order_release);
    if (hook) beforeCloseQuiescent.store(hook, std::memory_order_release);
}

void LayerActor::setBeforeSettleWritesForTest(bool (*hook)(void*, size_t) noexcept, void* context) noexcept {
    if (!hook) beforeSettleWrites.store(nullptr, std::memory_order_release);
    beforeSettleWritesContext.store(context, std::memory_order_release);
    if (hook) beforeSettleWrites.store(hook, std::memory_order_release);
}

void LayerActor::expireOpenForTest() noexcept {
    openExpiredForTest_.store(true, std::memory_order_release);
    wake();
}

void LayerActor::expireReadForTest() noexcept {
    readExpiredForTest_.store(true, std::memory_order_release);
    wake();
}

void LayerActor::resubscribeForTest() noexcept {
    resubscribeForTest_.store(true, std::memory_order_release);
    wake();
}

LayerActor::Clock::time_point LayerActor::issuanceResumes(Clock::time_point now) noexcept {
    const bool held = now < resendHold_ && !resendHoldExpiredForTest_.load(std::memory_order_acquire);
    // The worker settles writes before it asks, so a hold with nothing in
    // flight issues nothing in this pass.
    resendHeldForTest_.store(held && writes_.empty(), std::memory_order_release);
    return held ? std::max(resumeAt_, resendHold_) : resumeAt_;
}

std::string LayerActor::objectKey() const {
    auto key = progress_.identity.session;
    key.push_back('\0');
    key += progress_.identity.layer;
    key.push_back('\0');
    key += std::to_string(progress_.objectId());
    return key;
}

void LayerActor::expireResendHoldForTest() noexcept {
    resendHoldExpiredForTest_.store(true, std::memory_order_release);
    wake();
}

void LayerActor::wake() noexcept {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopped_) return;
        dirty_ = true;
    }
    runtime_->auth->wake(*this);
}

LayerActor::TransportSnapshot LayerActor::transportSnapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return {events_.size(), readCompletionPending_.load(std::memory_order_acquire), scheduled_,
        ping_ && ping_->wait(0) != OMNI_USD_LIVE_OK, readPaused_, dirty_,
        heldReadWindows_.load(std::memory_order_acquire)};
}

bool LayerActor::updateCanFit(size_t bytes) const noexcept {
    const auto& values = runtime_->options.values;
    // The layer's own standing read reservation is held beside every admission.
    return bytes <= values.max_update_bytes &&
        layerReadFrameLimit(values.max_update_bytes) + admissionCharge(bytes) <= values.max_pending_bytes_per_endpoint;
}

uint64_t LayerActor::admissionCharge(size_t bytes) const noexcept {
    return uint64_t(bytes) + 256 + layerUpdateCharge(updateHeaderBound_, bytes);
}

uint64_t LayerActor::submit(std::string_view bytes, std::shared_ptr<const void> completeIntent,
    LayerProgress::Publication publication, uint64_t retainedBytes) try {
    (void)observeHostFailure();
    if (!completeIntent) throw std::invalid_argument("missing complete intent");
    if (bytes.size() > runtime_->options.values.max_update_bytes)
        throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    auto effect = std::make_shared<Effect>();
    effect->authored.assign(bytes);
    effect->operationId = newOperationId();
    effect->intent = std::move(completeIntent);
    effect->drain = drain_->prepare();
    // Every issue of this intent sends these bytes, which a catch-up's replay
    // of the acknowledged write at its position repeats (design section 12).
    LayerProgress::ContentHash hash{};
    SHA256(reinterpret_cast<const unsigned char*>(bytes.data()), bytes.size(), hash.data());
    uint64_t id;
    if (beforeAdmission_) beforeAdmission_(admissionContext_);
    {
        std::lock_guard<std::mutex> lock(admissionMutex_);
        if (closing_) throw OmniUsdLiveResult(OMNI_USD_LIVE_CANCELLED);
        const auto before = progress_.snapshot();
        if (terminalGate(before.gate)) throw before.result;
        const auto hostResult = hostFailureResult();
        if (hostResult != OMNI_USD_LIVE_OK) throw hostResult;
        if (holdOpening_) throw OmniUsdLiveResult(OMNI_USD_LIVE_BUSY);
        const auto admitted = effect->drain->admit();
        if (admitted != OMNI_USD_LIVE_OK) throw admitted;
        // Reserve only once every other admission check has passed, so a
        // rejected admission holds no reservation. One the ledger then
        // refuses releases without reporting to waiting layers, unless the
        // endpoint refused another reservation while it was held.
        effect->reservation = std::make_shared<PendingBytes>(runtime_->acquireEndpoint(endpoint_->handle),
            admissionCharge(bytes.size()), runtime_->options.values.max_pending_bytes_per_endpoint);
        // The predecessor is fixed at admission: the previous admitted write,
        // or, for the actor's first, none when no layer of this endpoint has
        // written the object, which makes it the writer's first write there.
        // Otherwise the first write is unchained, since a closed layer's last
        // writes may still be settling.
        if (!chainStarted_) {
            auto key = objectKey();
            std::lock_guard<std::mutex> endpointLock(endpoint_->mutex);
            chainBaseEmpty_ = progress_.objectId() && !endpoint_->writtenObjects.count(key);
        }
        effect->chainable = chainStarted_ || chainBaseEmpty_;
        if (chainStarted_) effect->previousOperationId = lastAdmittedOperation_;
        // Allocated before admission, whose publication commit forbids later failures.
        auto operation = effect->operationId;
        try {
            if (afterReservation_) afterReservation_(reservationContext_);
            id = progress_.admit(effect, publication, retainedBytes ? retainedBytes : bytes.size(), &hash);
        } catch (...) {
            effect->reservation->releaseQuietly();
            throw;
        }
        lastAdmittedOperation_.swap(operation);
        chainStarted_ = true;
        publishStatus();
    }
    wake();
    return id;
} catch (...) {
    // Compilation/allocation may have raced a host fault. Never acquire this
    // gate recursively under admissionMutex_; publish status after unwinding.
    (void)observeHostFailure();
    throw;
}

namespace {
// When an incoming effect's jitter holdback ends. An effect of the current
// read generation is timed by the estimator, which fills `decision`.
std::chrono::steady_clock::time_point holdbackDue(const LayerActor::Effect& effect, uint64_t generation,
    std::chrono::steady_clock::time_point now, const OmniUsdLiveJitterParams& params, const JitterEstimator& jitter,
    JitterEstimator::Decision* decision) noexcept {
    if (effect.generation == generation) {
        const auto decided = jitter.decide(effect.incoming->header.server_timestamp_microseconds_since_epoch(),
            effect.received, now, params);
        if (decision) *decision = decided;
        return decided.due;
    }
    // A retained old-generation update must not be retimed against a new
    // backend's clock. Configuration can still shorten its frozen
    // eligibility and zero always disables holdback.
    JitterEstimator empty;
    auto cap = params;
    cap.delay_constant_ms = cap.delay_maximum_ms;
    return std::min(effect.eligible, empty.due(0, effect.received, now, cap));
}
// Incoming depth: queued frames plus unapplied peer coverage. Local intents
// are excluded; a paused read cannot fill the holes their acknowledgments
// wait on (design section 12, incoming pacing).
size_t unappliedPeers(const LayerProgress::Snapshot& state) noexcept {
    return state.retained - static_cast<size_t>(state.localAdmitted - state.localRetired);
}
} // namespace

void LayerActor::receive(uint64_t generation, std::shared_ptr<const LayerReadUpdate> update) {
    // A frame a worker has popped but not yet passed to acceptCoverage is in
    // neither count, so the depth undercounts by at most one in-flight frame
    // per worker; the pause then fires that many frames later, still below
    // max_requests.
    const auto peers = unappliedPeers(progress_.snapshot());
    const auto measured = applicationNow_(applicationClockContext_);
    size_t depth = 0;
    bool pause = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closing_ || stopped_) return;
        // Guards an invariant, not a limit: the read pauses at depth
        // readHighWater_ <= max_requests and a held read delivers no frame.
        if (events_.size() >= runtime_->options.values.max_requests)
            throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
        events_.push_back({generation, std::move(update), Clock::now(), measured});
        depth = events_.size() + peers;
        if (!readPaused_ && depth >= readHighWater_) {
            // readPaused_ and the flow's hold change together under mutex_
            // (lock order: actor mutex_, then the flow's), so relieveRead
            // never releases a hold that is not yet set. The hold precedes
            // this frame's delivery return, so the read parks after it.
            pause = readPaused_ = true;
            readFlow_->hold();
        }
    }
    if (pause) {
        runtime_->diagnostics->emit(OMNI_USD_LIVE_RECORD_READ_PAUSED, endpoint_->handle, subject_, 0,
            {{DiagnosticKey::Generation, generation}, {DiagnosticKey::QueueDepth, depth}});
    }
    wake();
}

void LayerActor::relieveRead() noexcept {
    const auto state = progress_.snapshot();
    const auto peers = unappliedPeers(state);
    size_t depth = 0;
    std::function<void()> wake;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!readPaused_) return;
        depth = events_.size() + peers;
        if (depth > readLowWater_) return;
        readPaused_ = false;
        wake = readFlow_->release();
    }
    if (wake) wake(); // Enqueues the parked read outside the actor lock.
    runtime_->diagnostics->emit(OMNI_USD_LIVE_RECORD_READ_RESUMED, endpoint_->handle, subject_, 0,
        {{DiagnosticKey::Generation, state.generation}, {DiagnosticKey::QueueDepth, depth}});
}

ScheduledStep LayerActor::advance(Clock::time_point) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopped_) return {true, {}};
    if (terminalFrozen_ && !closing_) return {false, Clock::time_point::max()};
    if (scheduled_) {
        dirty_ = true; // Includes auth/global wakes arriving during worker work.
        return {false, Clock::time_point::max()};
    }
    observedWake_ = wakeRevision();
    scheduled_ = true;
    dirty_ = false;
    activityAdmission_ = activity_ ? activity_->admit() : OmniUsdLiveResult(OMNI_USD_LIVE_OK);
    runtime_->requests->executor().enqueue(*this);
    return {false, Clock::time_point::max()};
}

void LayerActor::dispatch() noexcept {
    Clock::time_point next;
    std::shared_ptr<DrainTicket> following;
    try {
        // Reserve the next worker epoch outside the timer/CQ paths. Failure
        // latches a drain error, while transport cleanup can still run.
        if (!terminalGate(progress_.snapshot().gate) && !closing_) following = drain_->prepare();
        if (activityAdmission_ != OMNI_USD_LIVE_OK && !closing_ && !terminalGate(progress_.snapshot().gate)) {
            failReadStream(activityAdmission_);
            next = Clock::now();
        } else next = step();
    } catch (OmniUsdLiveResult result) {
        terminal(result);
        next = Clock::now();
    } catch (const std::bad_alloc&) {
        terminal(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
        next = Clock::now();
    } catch (const std::length_error&) {
        terminal(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
        next = Clock::now();
    } catch (...) {
        terminal(OMNI_USD_LIVE_INTERNAL_ERROR);
        next = Clock::now();
    }
    // External close/failure never touches the worker-owned jitter tracker.
    // Flush before publishing stopped/frozen state, including exception exits.
    if (closing_) finishJitterWait(JitterWaitResolution::Closed, Clock::now());
    else if (terminalGate(progress_.snapshot().gate)) finishJitterWait(JitterWaitResolution::Failed, Clock::now());
    const auto authState = progress_.snapshot();
    if (terminalGate(authState.gate)) authObservation_.finish(authState.result, Clock::now());
    else if (closing_) authObservation_.finish(OMNI_USD_LIVE_CANCELLED, Clock::now());
    if (!closing_) relieveRead();
    // A flush the host deferred while the issuance window was full runs once
    // the window has room, or once the layer cannot issue.
    if (bufferedEdits_ && (closing_ || terminalGate(progress_.snapshot().gate) || !issuanceSaturated()))
        bufferedEdits_->slotFreed();
    publishStatus(); // Includes auth changes made near the end of step().
    // Publish/changed callbacks have returned, or failure is already latched.
    // Work admitted before this worker epoch must itself cover those callbacks.
    if (pendingApplied_.empty() &&
        (!pendingDrains_.empty() || !absorbedOwners_.empty() || !failedAbsorbedOwners_.empty()))
        retireAppliedDrain();
    for (size_t i = 0; i < receivedDrainCount_; ++i) receivedDrains_[i].reset();
    receivedDrainCount_ = 0;
    std::shared_ptr<DrainTicket> finished;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        finished = std::move(activity_);
        activity_ = std::move(following);
    }
    if (finished) finished->complete();
    finished.reset();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        scheduled_ = false;
        // No transport, host task or Changed callback mutates the settled
        // state after this point. Auth/global wakes leave a terminal actor
        // frozen; explicit close still retires the actor itself.
        const auto settled = progress_.snapshot();
        // An external failure/close may race the earlier flush. Use this same
        // settled observation before freezing or notifying lifecycle waiters.
        if (closing_) finishJitterWait(JitterWaitResolution::Closed, Clock::now());
        else if (terminalGate(settled.gate)) finishJitterWait(JitterWaitResolution::Failed, Clock::now());
        if (terminalGate(settled.gate)) authObservation_.finish(settled.result, Clock::now());
        else if (closing_) authObservation_.finish(OMNI_USD_LIVE_CANCELLED, Clock::now());
        if (!closing_ && terminalGate(settled.gate) && !settled.readingAfterFailure && havePublished_ &&
            same(settled, published_) && !read_ && writes_.empty() && !ping_ && events_.empty() && !application_ &&
            pendingApplied_.empty() && pendingDrains_.empty() && !readEndDrain_ && !receivedDrainCount_ &&
            !resendPending_)
            terminalFrozen_ = true;
        if (stopped_ || terminalFrozen_) closed_.notify_all();
        // Publish while holding the ingress mutex so an event racing dispatch
        // cannot have its immediate wake overwritten by this future deadline.
        runtime_->auth->schedule(*this, dirty_ || stopped_ ? Clock::now() : next, observedWake_);
    }
}

void LayerActor::cancelRead() {
    std::shared_ptr<Request> request;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        request = read_;
    }
    if (request) request->cancel(0);
}
void LayerActor::cancelBoth() {
    std::shared_ptr<Request> read, ping;
    std::array<std::shared_ptr<Request>, kMaxWindow> writes;
    size_t count = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        read = read_;
        ping = ping_;
        for (const auto& write : writes_)
            if (count < writes.size()) writes[count++] = write.request;
    }
    if (read) read->cancel(0);
    for (size_t i = 0; i < count; ++i)
        if (writes[i]) writes[i]->cancel(0);
    if (ping) ping->cancel(0);
}
void LayerActor::terminal(OmniUsdLiveResult result, const OmniUsdLiveRequestError* evidence, bool keepReading) {
    keepReading = keepReading && bool(callbacks_.applyDetached);
    drain_->fail(result, keepReading);
    std::shared_ptr<ApplyCompletion::State> application;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        application = application_;
    }
    if (application) application->cancelQueued();
    {
        // Publish the cause and its transport evidence as one cached-state
        // transition against admission/close-side status capture.
        std::lock_guard<std::mutex> lock(statusMutex_);
        const auto before = progress_.snapshot();
        if (keepReading) progress_.stopIssuance(result);
        else progress_.fail(result);
        const auto after = progress_.snapshot();
        if (!terminalGate(before.gate) || before.result != after.result || statusError_.result != after.result) {
            // Exhausted refresh after a rejected call keeps that rejection's
            // gRPC status: it is the evidence of this UNAUTHENTICATED cause.
            const bool authCause = !evidence && after.result == OMNI_USD_LIVE_UNAUTHENTICATED && waitingAuth_.load() &&
                authEvidence_.load() && statusError_.result == OMNI_USD_LIVE_UNAUTHENTICATED;
            // Evidence describes `result`; it never decorates an earlier latched cause.
            if (!authCause) statusError_ = evidence && after.result == result ? *evidence : OmniUsdLiveRequestError{};
            statusError_.result = after.result;
        }
    }
    publishStatus();
    // Issued mutations settle; auth/read failure never replays them.
    if (!progress_.snapshot().readingAfterFailure) cancelRead();
}

bool LayerActor::authenticate(Clock::time_point now, Clock::time_point& next) {
    if (beforeAuthGate_) beforeAuthGate_(authGateContext_);
    // Same admission gate as external close/failPublication. Acquire before
    // endpoint policy so the observation cannot consume a terminal wait as OK
    // while blocked on the endpoint mutex, or start a new wait after close.
    std::lock_guard<std::mutex> admissionLock(admissionMutex_);
    std::lock_guard<std::mutex> lock(endpoint_->mutex);
    const auto state = progress_.snapshot();
    if (!receiving(state)) {
        authObservation_.finish(state.result, Clock::now());
        return false;
    }
    if (closing_) {
        authObservation_.finish(OMNI_USD_LIVE_CANCELLED, Clock::now());
        return false;
    }
    runtime_->auth->evaluate(*endpoint_, wallTimeMs(), now);
    if (endpoint_->authFailed || (waitingAuth_ && endpoint_->authFailureSerial != authFailure_))
        throw OmniUsdLiveResult(OMNI_USD_LIVE_UNAUTHENTICATED);
    if (!runtime_->auth->usable(*endpoint_, wallTimeMs())) {
        if (!waitingAuth_) {
            authEvidence_ = false;
            waitingAuth_ = true;
            authFailure_ = endpoint_->authFailureSerial;
            authDeadline_ = now + std::chrono::milliseconds(endpoint_->options.authWaitMs);
        }
        if (now >= authDeadline_) throw OmniUsdLiveResult(OMNI_USD_LIVE_UNAUTHENTICATED);
        if (!terminalGate(state.gate)) authObservation_.start(Clock::now());
        next = std::min(next, std::min(authDeadline_, endpoint_->authDeadline));
        return false;
    }
    authObservation_.finish(OMNI_USD_LIVE_OK, Clock::now());
    waitingAuth_ = false;
    authFailure_ = endpoint_->authFailureSerial;
    return true;
}

void LayerActor::observeReadReady() noexcept {
    if (readReadyReported_) return;
    readReadyReported_ = true;
    const auto state = progress_.snapshot();
    runtime_->diagnostics->emit(OMNI_USD_LIVE_RECORD_STREAM_RESUMED, endpoint_->handle, subject_, 0,
        {{DiagnosticKey::Operation, OMNI_USD_LIVE_OPERATION_READ}, {DiagnosticKey::Result, OMNI_USD_LIVE_OK},
            {DiagnosticKey::Request, readRequestId_}, {DiagnosticKey::Attempt, 1},
            {DiagnosticKey::Generation, readGeneration_}, {DiagnosticKey::ResumeSequence, readResume_},
            {DiagnosticKey::ReadSequence, state.readSequence}, {DiagnosticKey::AppliedSequence, state.applied}});
}

void LayerActor::recordReadDisconnect(OmniUsdLiveResult result, uint64_t requestId, uint64_t attemptId) noexcept {
    const auto state = progress_.snapshot();
    runtime_->diagnostics->emit(OMNI_USD_LIVE_RECORD_STREAM_DISCONNECTED, endpoint_->handle, subject_, 0,
        {{DiagnosticKey::Operation, OMNI_USD_LIVE_OPERATION_READ}, {DiagnosticKey::Result, result},
            {DiagnosticKey::Request, requestId}, {DiagnosticKey::Attempt, attemptId},
            {DiagnosticKey::Generation, readGeneration_}, {DiagnosticKey::ResumeSequence, readResume_},
            {DiagnosticKey::ReadSequence, state.readSequence}, {DiagnosticKey::AppliedSequence, state.applied}});
}

void LayerActor::failReadStream(OmniUsdLiveResult result) {
    // The read latches its own stream failure (for example a server sequence
    // regression, design section 12) on drain_ before its completion is delivered.
    // Attribute that latch to the read with correlated local evidence; any
    // other drain failure keeps its uncorrelated terminal path.
    bool attributed;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        attributed = readRequestId_ && readFailedGeneration_ == readGeneration_ && readFailure_ == result;
    }
    if (!attributed) throw result;
    recordReadDisconnect(result, readRequestId_, 1);
    auto evidence = OmniUsdLiveRequestError{};
    evidence.struct_size = sizeof(evidence);
    evidence.abi_version = OMNI_USD_LIVE_ABI_VERSION;
    evidence.operation = OMNI_USD_LIVE_OPERATION_READ;
    evidence.result = result;
    evidence.request_id = readRequestId_;
    evidence.attempt_id = 1;
    terminal(result, &evidence); // Also closes the read; its completion then records nothing.
}

void LayerActor::consume(const Event& event) {
    const auto current = progress_.snapshot();
    if (event.generation != current.generation) return;
    // The attempt's first complete response ends its first-response deadline;
    // a quiet stream after it is healthy, however long it stays open.
    if (event.update->firstResponse) readDeadline_ = Clock::time_point::max();
    // A read-side failure after FAILED keeps the first cause and its evidence.
    const bool failed = terminalGate(current.gate);
    const auto& update = *event.update;
    if (update.caughtUp) {
        std::lock_guard<std::mutex> lock(statusMutex_);
        try {
            if (progress_.caughtUp(event.generation, update.header.object_id(), update.header.sequence_num())
                    .accepted) {
                attempts_ = 0;
                // Only the attempt's first boundary is its catch-up target; a
                // later header-only frame is a live empty commit.
                if (!readReadyReported_) catchUpSequence_ = update.header.sequence_num();
                observeReadReady();
            }
        } catch (...) {
            if (!failed) {
                statusError_ = {};
                statusError_.result = progress_.snapshot().result;
            }
            throw;
        }
        return;
    }
    const auto decoded = callbacks_.decode(update);
    if (decoded.kind == LayerProgress::CoverageKind::Snapshot) {
        const auto state = progress_.snapshot();
        // A reset is bootstrap only, even when earlier retained coverage
        // has not yet reached scene application. Never clip it to the cursor.
        if (state.covered || state.readSequence) throw OmniUsdLiveResult(OMNI_USD_LIVE_MALFORMED_STREAM);
    }
    auto effect = std::make_shared<Effect>();
    effect->incoming = event.update;
    effect->drain = update.drain;
    effect->received = event.received;
    effect->measuredReceipt = event.measuredReceipt;
    effect->generation = event.generation;
    {
        std::lock_guard<std::mutex> lock(endpoint_->mutex);
        effect->eligible = jitter_.due(update.header.server_timestamp_microseconds_since_epoch(), event.received,
            Clock::now(), endpoint_->jitter);
    }
    LayerProgress::ContentHash hash{};
    SHA256(reinterpret_cast<const unsigned char*>(update.bytes.data()), update.bytes.size(), hash.data());
    std::lock_guard<std::mutex> lock(statusMutex_);
    try {
        progress_.acceptCoverage(event.generation, update.header.object_id(), decoded.first,
            update.header.sequence_num(), {effect, hash, decoded.kind});
        if (decoded.recoveryBoundary &&
            progress_.recoveryReady(event.generation, update.header.object_id(), update.header.sequence_num())
                .accepted) {
            attempts_ = 0;
            if (!readReadyReported_) catchUpSequence_ = update.header.sequence_num();
            observeReadReady();
        }
    } catch (...) {
        if (!failed) {
            statusError_ = {};
            statusError_.result = progress_.snapshot().result;
        }
        throw;
    }
}

void LayerActor::settleApplication() {
    std::shared_ptr<ApplyCompletion::State> application;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        application = application_;
    }
    if (!application) return;
    bool finished, retired;
    bool observe = false, ran = false;
    uint64_t lifetimeMs = 0, queueMs = 0, bodyMs = 0, postBodyMs = 0;
    OmniUsdLiveResult result;
    Clock::time_point retiredAt{};
    {
        std::lock_guard<std::mutex> lock(application->mutex);
        finished = application->phase == ApplyCompletion::State::Phase::Finished;
        retired = application->retired;
        result = application->result;
        if (retired && !application->timingRecorded) {
            application->timingRecorded = true;
            observe = true;
            ran = application->ran;
            retiredAt = application->retiredAt;
            lifetimeMs = applicationElapsedMs(application->admittedAt, application->retiredAt);
            if (ran) {
                queueMs = applicationElapsedMs(application->admittedAt, application->runAt);
                bodyMs = applicationElapsedMs(application->runAt, application->bodyEndedAt);
                postBodyMs = applicationElapsedMs(application->bodyEndedAt, application->retiredAt);
            }
        }
    }
    if (observe)
        runtime_->diagnostics->retiredApplication(application->local ? ApplicationKind::LocalAsync :
                                                                       ApplicationKind::OrderedAsync,
            result, lifetimeMs, ran, queueMs, bodyMs, postBodyMs);
    if (observe && result == OMNI_USD_LIVE_OK && !application->local) {
        // Every peer update the application installed measures from its own
        // frame's receipt to this one retirement (bounded by kMaxBatch).
        std::array<uint64_t, kMaxBatch> samples;
        size_t count = 0;
        for (const auto& effect : application->effect)
            if (effect->incoming && count < samples.size())
                samples[count++] = applicationElapsedMs(effect->measuredReceipt, retiredAt);
        if (count) runtime_->diagnostics->appliedPeerUpdates(samples.data(), count);
    }
    if (!finished) return;
    // After an issuance-only failure, an application the failure cancelled or
    // refused leaves its work to journal-only application (applyReady).
    if (result != OMNI_USD_LIVE_OK && !progress_.snapshot().readingAfterFailure) terminal(result);
    if (!retired) return;
    if (result == OMNI_USD_LIVE_OK) {
        if (application->local) application->effect.front()->localPublished_ = true;
        else {
            // Within the reserved capacity: no allocation.
            bool retiresLocal = false;
            for (size_t i = 0; i < application->work.size(); ++i) {
                const auto& work = application->work[i];
                pendingApplied_.push_back(work.id);
                pendingDrains_.push_back(application->effect[i]->drain);
                retiresLocal |= work.disposition != LayerProgress::Disposition::Peer || work.absorbedLocal != 0;
            }
            localRetirementPending_.store(retiresLocal, std::memory_order_release);
        }
    } else {
        // Failure has been latched before the original captured obligations
        // retire. Progress still owns the complete unapplied effects/intents.
        for (const auto& effect : application->effect)
            if (effect->drain) effect->drain->complete();
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        application_.reset();
        if (closing_ && admission_) admission_->pinToHost(false);
    }
}

void LayerActor::startApplication(std::vector<LayerProgress::Work> work,
    std::vector<std::shared_ptr<const Effect>> effect, bool local) {
    assert(!work.empty() && work.size() == effect.size() && work.size() <= kMaxBatch && (!local || work.size() == 1));
    auto application = std::make_shared<ApplyCompletion::State>(
        std::weak_ptr<LayerActor>(std::static_pointer_cast<LayerActor>(shared_from_this())), work, effect, local,
        applicationNow_, applicationClockContext_);
    auto completion = std::shared_ptr<ApplyCompletion>(new ApplyCompletion(application));
    {
        // External host-publication failure takes this same admission gate.
        // If installation wins, terminal() can see and cancel the queued
        // completion; if failure wins, no fresh application is published.
        std::lock_guard<std::mutex> admissionLock(admissionMutex_);
        std::lock_guard<std::mutex> lock(mutex_);
        if (closing_ || terminalGate(progress_.snapshot().gate)) return;
        const auto hostResult = hostFailureResult();
        if (hostResult != OMNI_USD_LIVE_OK) throw hostResult;
        assert(!application_);
        application->admittedAt = applicationNow_(applicationClockContext_);
        runtime_->diagnostics->admittedApplication(local ? ApplicationKind::LocalAsync : ApplicationKind::OrderedAsync);
        application_ = application;
    }
    if (local) finishJitterWait(JitterWaitResolution::Preempted, Clock::now());
    try {
        if (!local && callbacks_.applyBatchAsync)
            callbacks_.applyBatchAsync(std::move(work), std::move(effect), completion);
        else {
            const auto& callback = local ? callbacks_.localAsync : callbacks_.applyAsync;
            callback(std::move(work.front()), std::move(effect.front()), completion);
        }
    } catch (OmniUsdLiveResult result) {
        application->finish(result, true);
        throw;
    } catch (const std::bad_alloc&) {
        application->finish(OMNI_USD_LIVE_RESOURCE_EXHAUSTED, true);
        throw;
    } catch (const std::length_error&) {
        application->finish(OMNI_USD_LIVE_RESOURCE_EXHAUSTED, true);
        throw;
    } catch (...) {
        application->finish(OMNI_USD_LIVE_INTERNAL_ERROR, true);
        throw;
    }
}

bool LayerActor::localReady() {
    if (!callbacks_.localAsync) return true;
    const auto state = progress_.snapshot();
    if (closing_ || terminalGate(state.gate)) return false;
    if (!state.queued) return true;
    const auto work = progress_.retained();
    // Local IDs are allocated in FIFO admission order. An issued head is
    // already published; inspect the earliest still-queued local work.
    const auto item = std::find_if(work.begin(), work.end(),
        [](const auto& entry) { return entry.disposition == LayerProgress::Disposition::Queued; });
    if (item == work.end()) return true;
    const auto effect = std::static_pointer_cast<const Effect>(item->owner);
    if (effect->localPublished_) return true;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (application_) return false;
    }
    startApplication({*item}, {effect}, true);
    return false;
}

void LayerActor::finishJitterWait(JitterWaitResolution resolution, Clock::time_point observed) noexcept {
    if (const auto measurement = jitterWait_.finish(observed))
        runtime_->diagnostics->finishedJitterWait(resolution, *measurement);
}

bool LayerActor::applyReady(Clock::time_point& next) {
    bool occupied;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        occupied = bool(application_);
    }
    if (occupied) {
        finishJitterWait(JitterWaitResolution::Preempted, Clock::now());
        return false;
    }
    for (unsigned count = 0; count < 32 && !closing_; ++count) {
        const auto state = progress_.snapshot();
        if (!receiving(state) || state.applied == UINT64_MAX) return false;
        const bool batching = callbacks_.applyBatchAsync && !terminalGate(state.gate);
        auto run = progress_.applicable(batching ? kMaxBatch : 1);
        if (run.empty()) {
            finishJitterWait(JitterWaitResolution::Preempted, Clock::now());
            return false;
        }
        const auto item = run.begin();
        const auto effect = std::static_pointer_cast<const Effect>(item->owner);
        if (terminalGate(state.gate)) {
            // After an issuance-only FAILED the journal keeps receiving in
            // order, without holdback; nothing installs into the host layer.
            try {
                callbacks_.applyDetached(*item, *effect);
            } catch (...) {
                terminal(currentExceptionResult());
                return false;
            } // Closes the read; first cause stays.
            auto owners = progress_.applied(item->id);
            if (effect->drain) effect->drain->complete();
            completeAbsorbed(owners);
            publish();
            continue;
        }
        if (jitterWait_.active() && jitterWait_.work() != item->id)
            finishJitterWait(JitterWaitResolution::Preempted, Clock::now());
        OmniUsdLiveJitterParams params;
        {
            std::lock_guard<std::mutex> lock(endpoint_->mutex);
            params = endpoint_->jitter;
        }
        const auto now = Clock::now();
        if (effect->incoming && !effect->incoming->firstResponse) {
            JitterEstimator::Decision decision{effect->eligible};
            const auto due = holdbackDue(*effect, state.generation, now, params, jitter_, &decision);
            if (jitterWait_.active()) jitterWait_.observe(now, due);
            if (due > now) {
                if (!jitterWait_.active()) {
                    jitterWait_.begin(item->id, now, due);
                    runtime_->diagnostics->startedJitterWait(applicationElapsedMs(now, due),
                        effect->generation == state.generation ? &decision : nullptr);
                }
                next = std::min(next, due);
                return false;
            }
            finishJitterWait(JitterWaitResolution::Eligible, now);
        } else {
            finishJitterWait(JitterWaitResolution::Preempted, Clock::now());
        }
        if (callbacks_.applyAsync || callbacks_.applyBatchAsync) {
            std::vector<std::shared_ptr<const Effect>> effects{effect};
            size_t taken = 1;
            if (batching && item->disposition == LayerProgress::Disposition::Peer) {
                // Consecutive peer work whose holdback has elapsed joins the
                // head's application, within the batch's byte bound. A local
                // acknowledgment or a held peer ends the batch.
                effects.reserve(run.size());
                size_t bytes = 0;
                const size_t maxBytes = runtime_->options.values.max_update_bytes;
                for (; taken < run.size(); ++taken) {
                    const auto& work = run[taken];
                    if (work.disposition != LayerProgress::Disposition::Peer) break;
                    auto following = std::static_pointer_cast<const Effect>(work.owner);
                    const auto size = following->bytes().size();
                    if (size > maxBytes - bytes) break;
                    if (following->incoming && !following->incoming->firstResponse &&
                        holdbackDue(*following, state.generation, now, params, jitter_, nullptr) > now)
                        break;
                    bytes += size;
                    effects.push_back(std::move(following));
                }
            }
            run.resize(taken);
            startApplication(std::move(run), std::move(effects), false);
            return false; // Completion/retirement wake the actor; never poll.
        }
        {
            // Linearize synchronous callback admission against external
            // failure, but never invoke host code under the gate. An admitted
            // callback may finish after failure; this worker remains scheduled
            // through its application retirement, preventing the terminal
            // freeze from observing a partially completed callback.
            std::lock_guard<std::mutex> admissionLock(admissionMutex_);
            if (closing_ || terminalGate(progress_.snapshot().gate)) return false;
            const auto hostResult = hostFailureResult();
            if (hostResult != OMNI_USD_LIVE_OK) throw hostResult;
        }
        const auto admitted = applicationNow_(applicationClockContext_);
        runtime_->diagnostics->admittedApplication(ApplicationKind::OrderedSync);
        const auto started = applicationNow_(applicationClockContext_);
        const auto completed = [&](OmniUsdLiveResult result) noexcept {
            const auto ended = applicationNow_(applicationClockContext_);
            runtime_->diagnostics->retiredApplication(ApplicationKind::OrderedSync, result,
                applicationElapsedMs(admitted, ended), true, 0, applicationElapsedMs(started, ended), 0);
        };
        try {
            callbacks_.apply(*item, *effect);
        }
        // Match dispatch's translation while preserving its exception path.
        catch (OmniUsdLiveResult result) {
            completed(result);
            throw;
        } catch (const std::bad_alloc&) {
            completed(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
            throw;
        } catch (const std::length_error&) {
            completed(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
            throw;
        } catch (...) {
            completed(OMNI_USD_LIVE_INTERNAL_ERROR);
            throw;
        }
        completed(OMNI_USD_LIVE_OK);
        if (effect->incoming) {
            const uint64_t sample = applicationElapsedMs(effect->measuredReceipt,
                applicationNow_(applicationClockContext_));
            runtime_->diagnostics->appliedPeerUpdates(&sample, 1);
        }
        pendingApplied_.push_back(item->id);
        pendingDrains_.push_back(effect->drain);
        localRetirementPending_.store(item->disposition != LayerProgress::Disposition::Peer || item->absorbedLocal != 0,
            std::memory_order_release);
        settleApplied();
    }
    return true;
}

void LayerActor::pollPing(Clock::time_point now, Clock::time_point& next) {
    std::shared_ptr<Request> ping;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ping = ping_;
    }
    const auto state = progress_.snapshot();
    if (ping) {
        if (closing_ || terminalGate(state.gate) || pingGeneration_ != state.generation) ping->cancel(0);
        if (ping->wait(0) == OMNI_USD_LIVE_OK) {
            if (const auto capabilities =
                    pingOutcome_->completed.load(std::memory_order_acquire) ? pingOutcome_->capabilities : uint8_t(0)) {
                endpoint_->capabilities.store(capabilities, std::memory_order_release);
                publishWindow();
            }
            if (pingSamples_ && !closing_ && !terminalGate(state.gate) && pingGeneration_ == state.generation &&
                pingOutcome_->completed.load(std::memory_order_acquire) && pingOutcome_->serverUs &&
                pingOutcome_->received >= pingSent_) {
                const auto rtt =
                    std::chrono::duration_cast<std::chrono::microseconds>(pingOutcome_->received - pingSent_).count();
                jitter_.sample(uint64_t(rtt) / 2, pingOutcome_->serverUs, pingOutcome_->received);
            }
            const auto released = runtime_->requests->release(ping->handle);
            assert(released == OMNI_USD_LIVE_OK);
            (void)released;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                ping_.reset();
            }
            pingOutcome_.reset();
            ping.reset();
            nextPing_ = pingGeneration_ == state.generation ? now + std::chrono::seconds(10) : now;
        } else if (closing_ || terminalGate(state.gate) || pingGeneration_ != state.generation ||
            pingOutcome_->completed.load(std::memory_order_acquire))
            next = std::min(next, now + std::chrono::milliseconds(1));
    }
    if (ping || closing_ || terminalGate(state.gate) || !state.readStarted) return;
    if (now < nextPing_) {
        next = std::min(next, nextPing_);
        return;
    }
    {
        std::lock_guard<std::mutex> lock(endpoint_->mutex);
        // Timing is optional: it must not create an extra auth challenge, and
        // while deliberate smoothing is disabled a Ping is sent only until one
        // of the endpoint has reported its capabilities. One the token blocks
        // while they are unknown is sent once a usable token wakes the actor.
        const bool known = endpoint_->capabilities.load(std::memory_order_acquire) & Endpoint::kCapabilitiesKnown;
        if (!runtime_->auth->usable(*endpoint_, wallTimeMs())) {
            if (known) {
                nextPing_ = now + std::chrono::seconds(10);
                next = std::min(next, nextPing_);
            }
            return;
        }
        pingSamples_ = endpoint_->jitter.delay_maximum_ms != 0;
        if (!pingSamples_ && known) {
            nextPing_ = now + std::chrono::seconds(10);
            next = std::min(next, nextPing_);
            return;
        }
    }
    std::shared_ptr<PingOutcome> outcome;
    try {
        outcome = std::make_shared<PingOutcome>();
        auto receipt = std::make_shared<UnaryTimingReceipt>();
        const auto weak = std::weak_ptr<LayerActor>(std::static_pointer_cast<LayerActor>(shared_from_this()));
        rpc::PingRequest input;
        // Wire echo is a correlation value; RTT and scheduling use steady time.
        const auto stamp = uint64_t(
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch())
                .count());
        input.set_client_timestamp_microseconds_since_epoch(stamp);
        pingSent_ = Clock::now();
        pingGeneration_ = state.generation;
        ping = submitTimingPing(*runtime_, endpoint_->handle, input,
            UnaryCallback<rpc::PingResponse>{
                [weak, outcome, stamp, receipt](const auto& error, const auto& response, uint64_t) {
            const auto ready = receipt->ready.load(std::memory_order_acquire);
            outcome->received = ready ? receipt->completed : Clock::time_point{};
            if (ready && error.result == OMNI_USD_LIVE_OK &&
                response.client_timestamp_microseconds_since_epoch() == stamp)
                outcome->serverUs = response.server_timestamp_microseconds_since_epoch();
            if (error.result == OMNI_USD_LIVE_OK)
                outcome->capabilities = uint8_t(Endpoint::kCapabilitiesKnown |
                    (response.operation_ids() ? Endpoint::kOperationIds : 0) |
                    (response.chained_writes() ? Endpoint::kChainedWrites : 0));
            outcome->completed.store(true, std::memory_order_release);
            if (auto actor = weak.lock()) actor->wake();
        }},
            timingTimeoutMs_, receipt);
    } catch (...) {
        // Optional clock sampling must not reject an accepted scene edit when
        // the request registry is full or allocation/transport setup fails;
        // the endpoint keeps the capabilities it has.
        nextPing_ = now + std::chrono::seconds(10);
        next = std::min(next, nextPing_);
        return;
    }
    pingOutcome_ = std::move(outcome);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ping_ = std::move(ping);
    }
}

void LayerActor::publishStatus() noexcept {
    if (!subject_) return;
    // Synchronous admission and worker publication both capture while holding
    // this mutex; an older snapshot cannot overwrite newer pending counts.
    std::lock_guard<std::mutex> lock(statusMutex_);
    const auto state = progress_.snapshot();
    OmniUsdLiveStatus status{};
    status.subject_kind = OMNI_USD_LIVE_SUBJECT_LAYER;
    status.progress_known = 1;
    status.pending_write_count = state.pendingWrites;
    status.pending_bytes = state.pendingBytes;
    // A buffer that emptied since the last publication advances the revision
    // even when every field is unchanged (see BufferedEditCount). take()
    // consumes that report before publishActor(), which returns BUSY only
    // before handoff; open() hands off before the actor's first publication.
    const auto taken = bufferedEdits_ ? bufferedEdits_->take() : BufferedEditCount::Taken{};
    const auto buffered = taken.count;
    // The deferral cause holds whatever the count: a layer that the endpoint's
    // shared host buffer budget refuses is BLOCKED with an empty buffer.
    const auto blocked = bufferedEdits_ ? bufferedEdits_->blocked() : OmniUsdLiveResult(OMNI_USD_LIVE_OK);
    status.buffered_edit_count = buffered;
    status.has_unconfirmed_edits = state.pendingWrites != 0 || buffered != 0;
    status.result = state.result;
    status.operation = state.pendingWrites ? OMNI_USD_LIVE_OPERATION_UPDATE : OMNI_USD_LIVE_OPERATION_READ;
    bool transportCause = true; // Whether statusError_ can describe the reported cause.
    if (terminalGate(state.gate)) {
        status.state = OMNI_USD_LIVE_STATE_FAILED;
        status.recovery_action = OMNI_USD_LIVE_RECOVERY_PRESERVE_AND_REOPEN;
    } else if (waitingAuth_.load()) {
        status.state = OMNI_USD_LIVE_STATE_AUTH_WAIT;
        status.result = OMNI_USD_LIVE_UNAUTHENTICATED;
        status.operation = OMNI_USD_LIVE_OPERATION_AUTH;
        status.recovery_action = OMNI_USD_LIVE_RECOVERY_REFRESH_AUTH;
        transportCause = authEvidence_.load();
    } else if (blocked != OMNI_USD_LIVE_OK) {
        // A transient flush failure holds recorded host edits unadmitted. It
        // is a local cause: an earlier transport status never describes it.
        status.state = OMNI_USD_LIVE_STATE_BLOCKED;
        transportCause = false;
        status.result = blocked;
        status.operation = OMNI_USD_LIVE_OPERATION_UPDATE;
        status.recovery_action = OMNI_USD_LIVE_RECOVERY_RELIEVE_BACKPRESSURE;
    } else if (holdOpening_ || !initialApplicationReady(state)) {
        status.state = OMNI_USD_LIVE_STATE_OPENING;
        status.operation = OMNI_USD_LIVE_OPERATION_OPEN;
    } else if (state.gate == LayerProgress::Gate::Recovering) {
        status.state = OMNI_USD_LIVE_STATE_RECONNECTING;
        status.result = statusError_.result;
        status.recovery_action = OMNI_USD_LIVE_RECOVERY_WAIT_FOR_RETRY;
    } else status.state = OMNI_USD_LIVE_STATE_READY;
    if (transportCause && status.result != OMNI_USD_LIVE_OK && statusError_.result == status.result) {
        status.grpc_status_present = statusError_.grpc_status_present;
        status.grpc_status = statusError_.grpc_status;
        if (statusError_.operation) status.operation = statusError_.operation;
    }
    if (closing_) status.state = statusClosed_ ? OMNI_USD_LIVE_STATE_CLOSED : OMNI_USD_LIVE_STATE_CLOSING;
    status.read_sequence = state.readSequence;
    status.applied_sequence = state.applied;
    // Before publication: a status that shows the lag crossing a bound
    // follows the record that reports it. Only a READY layer, past its
    // initial application and outside a reconnect, is evaluated.
    evaluateApplyLag(state, !closing_ && !terminalGate(state.gate), status.state == OMNI_USD_LIVE_STATE_READY);
    (void)statusOwner_->publishActor(status, taken.emptied);
}

void LayerActor::evaluateApplyLag(const LayerProgress::Snapshot& state, bool active, bool ready) noexcept {
    if (active && !ready) return; // OPENING, RECONNECTING, AUTH_WAIT and BLOCKED hold any interval.
    // The read generation's catch-up content, up to its accepted target, is
    // one aggregate the host applies at once, not a host falling behind the
    // stream, so application lag counts only positions beyond that target.
    // Applied can lead the read: the read never delivers the layer's own writes.
    const uint64_t threshold = runtime_->options.values.apply_lag_warning_threshold;
    const uint64_t floor = std::max(state.applied, catchUpSequence_);
    const uint64_t lag = state.readSequence > floor ? state.readSequence - floor : 0;
    if (!applyLagging_) {
        if (!active || lag <= threshold) return;
        applyLagging_ = true;
        applyLagSince_ = Clock::now();
        runtime_->diagnostics->applyLagChanged(true);
        runtime_->diagnostics->emit(OMNI_USD_LIVE_RECORD_APPLY_LAG_ENTERED, endpoint_->handle, subject_, 0,
            {{DiagnosticKey::Generation, state.generation}, {DiagnosticKey::ReadSequence, state.readSequence},
                {DiagnosticKey::AppliedSequence, state.applied}});
        return;
    }
    // Hysteresis: the interval ends at half the threshold, so a lag that
    // hovers at the threshold reports one interval rather than many.
    if (active && lag > threshold / 2) return;
    applyLagging_ = false;
    runtime_->diagnostics->applyLagChanged(false);
    if (!active) return; // A failed or closing layer ends the interval without a record.
    runtime_->diagnostics->emit(OMNI_USD_LIVE_RECORD_APPLY_LAG_EXITED, endpoint_->handle, subject_, 0,
        {{DiagnosticKey::Generation, state.generation}, {DiagnosticKey::ReadSequence, state.readSequence},
            {DiagnosticKey::AppliedSequence, state.applied},
            {DiagnosticKey::DurationMs, applicationElapsedMs(applyLagSince_, Clock::now())}});
}

void LayerActor::noteHostEdits() noexcept {
    // A stopped or frozen terminal actor runs no worker step; the
    // FAILED layer's buffered count still reaches status.
    if (terminalGate(progress_.snapshot().gate) && !closing_) publishStatus();
    else wake();
}

void LayerActor::reportFlushUnsupported(uint32_t reason) noexcept {
    try {
        OmniUsdLiveStatus status{};
        status.struct_size = sizeof(status);
        status.abi_version = OMNI_USD_LIVE_ABI_VERSION;
        (void)runtime_->statuses->snapshot(subject_, &status);
        runtime_->diagnostics->emit(OMNI_USD_LIVE_RECORD_FLUSH_UNSUPPORTED, endpoint_->handle, subject_,
            status.incident_id,
            {{DiagnosticKey::Operation, OMNI_USD_LIVE_OPERATION_UPDATE},
                {DiagnosticKey::Result, OMNI_USD_LIVE_UNSUPPORTED}, {DiagnosticKey::Reason, reason}});
    } catch (...) { /* Diagnostics never change the layer's failure. */
    }
}
void LayerActor::reportDiscardedHostEdits() noexcept {
    // A closing actor admits nothing, so the shared buffer's records are never
    // issued; the CLOSED status keeps their count.
    const auto count = bufferedEdits_ ? bufferedEdits_->load() : 0;
    if (!count || !subject_) return;
    try {
        const auto state = progress_.snapshot();
        const auto result = terminalGate(state.gate) ? state.result : hostFailureResult();
        OmniUsdLiveStatus status{};
        status.struct_size = sizeof(status);
        status.abi_version = OMNI_USD_LIVE_ABI_VERSION;
        (void)runtime_->statuses->snapshot(subject_, &status);
        runtime_->diagnostics->emit(OMNI_USD_LIVE_RECORD_HOST_EDITS_DISCARDED, endpoint_->handle, subject_,
            status.incident_id, {{DiagnosticKey::Count, count}, {DiagnosticKey::Result, result}});
    } catch (...) { /* Diagnostics never change the close. */
    }
}
OmniUsdLiveResult LayerActor::hostFailureResult() const noexcept {
    return hostFailure_ ? hostFailure_->load(std::memory_order_acquire) : OmniUsdLiveResult(OMNI_USD_LIVE_OK);
}
OmniUsdLiveResult LayerActor::observeHostFailure() noexcept {
    const auto result = hostFailureResult();
    if (result == OMNI_USD_LIVE_OK) return result;
    try {
        std::lock_guard<std::mutex> lock(admissionMutex_);
        if (!terminalGate(progress_.snapshot().gate) && !closing_) {
            OmniUsdLiveRequestError evidence{};
            evidence.operation = OMNI_USD_LIVE_OPERATION_UPDATE;
            evidence.result = result;
            terminal(result, &evidence, true); // A host failure stops issuance; the journal keeps receiving.
        }
    } catch (...) {}
    wake();
    const auto after = progress_.snapshot();
    return terminalGate(after.gate) ? after.result : result;
}
OmniUsdLiveResult LayerActor::authoringResult() noexcept {
    (void)observeHostFailure();
    try {
        std::lock_guard<std::mutex> lock(admissionMutex_);
        const auto state = progress_.snapshot();
        if (terminalGate(state.gate)) return state.result;
        if (closing_) return OMNI_USD_LIVE_CANCELLED;
        const auto hostResult = hostFailureResult();
        if (hostResult != OMNI_USD_LIVE_OK) return hostResult;
        return holdOpening_ ? OMNI_USD_LIVE_BUSY : OMNI_USD_LIVE_OK;
    } catch (...) { return currentExceptionResult(); }
}

OmniUsdLiveResult LayerActor::completeOpening() noexcept {
    (void)observeHostFailure();
    try {
        {
            std::lock_guard<std::mutex> admissionLock(admissionMutex_);
            std::lock_guard<std::mutex> statusLock(statusMutex_);
            const auto state = progress_.snapshot();
            if (terminalGate(state.gate)) return state.result;
            if (closing_) return OMNI_USD_LIVE_CANCELLED;
            const auto hostResult = hostFailureResult();
            if (hostResult != OMNI_USD_LIVE_OK) throw hostResult;
            holdOpening_ = false;
        }
        publishStatus();
        wake();
        return authoringResult();
    } catch (...) {
        const auto result = currentExceptionResult();
        (void)observeHostFailure();
        return result;
    }
}

OmniUsdLiveResult LayerActor::failPublication(OmniUsdLiveResult result) noexcept {
    if (result == OMNI_USD_LIVE_OK || !definedResult(result)) return OMNI_USD_LIVE_INVALID_ARGUMENT;
    try {
        {
            std::lock_guard<std::mutex> lock(admissionMutex_);
            if (closing_) return OMNI_USD_LIVE_CANCELLED;
            holdOpening_ = true;
            OmniUsdLiveRequestError evidence{};
            evidence.operation = OMNI_USD_LIVE_OPERATION_OPEN;
            evidence.result = result;
            terminal(result, &evidence);
        }
        wake();
        return OMNI_USD_LIVE_OK;
    } catch (...) { return currentExceptionResult(); }
}

void LayerActor::publish() {
    publishStatus();
    const auto state = progress_.snapshot();
    if (!havePublished_ || !same(state, published_)) {
        published_ = state;
        havePublished_ = true;
        if (callbacks_.changed) {
            try {
                callbacks_.changed(state);
            } catch (...) {
                callbacks_.changed = {};
                throw;
            }
        }
    }
}

// Settles a finished application and publishes its effect before a later
// effect's callback starts.
void LayerActor::settleAppliedEffects() {
    (void)observeHostFailure();
    settleApplication();
    if (!pendingApplied_.empty()) settleApplied(); // Also finishes local bookkeeping during close.
}

void LayerActor::settleApplied() {
    // In application order. Each ID leaves once its bookkeeping holds, so a
    // throw from progress keeps only the unsettled ones.
    while (!pendingApplied_.empty()) {
        if (const auto result = adoptAbsorbed(progress_.applied(pendingApplied_.front())); result != OMNI_USD_LIVE_OK) {
            pendingApplied_.clear();
            throw result;
        }
        pendingApplied_.erase(pendingApplied_.begin());
    }
    // Publish these effects before starting a later application's callback.
    // A captured drain must not inherit later admissions.
    publish();
    retireAppliedDrain();
}

// A receiving layer re-evaluates its endpoint's credentials; a failed
// evaluation or a newer failure serial fails it UNAUTHENTICATED.
void LayerActor::checkReadAuthorization(Clock::time_point now) {
    if (!closing_ && receiving(progress_.snapshot())) {
        bool failed;
        {
            std::lock_guard<std::mutex> lock(endpoint_->mutex);
            runtime_->auth->evaluate(*endpoint_, wallTimeMs(), now);
            failed = endpoint_->authFailed || endpoint_->authFailureSerial != attachmentAuthFailure_;
        }
        if (failed) terminal(OMNI_USD_LIVE_UNAUTHENTICATED);
    }
}

// Consumes a bounded batch of queued frames; a closing or non-receiving
// layer drops them.
void LayerActor::consumeQueuedFrames() {
    // Drain only a bounded batch. Every frame retains its transport reservation.
    for (unsigned count = 0; count < 32; ++count) {
        Event event;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (events_.empty()) break;
            event = std::move(events_.front());
            events_.pop_front();
        }
        if (!closing_ && receiving(progress_.snapshot())) {
            assert(receivedDrainCount_ < receivedDrains_.size());
            receivedDrains_[receivedDrainCount_++] = event.update->drain;
            consume(event);
        }
    }
}

// Settles every finished write attempt: releases its request first, then
// settles a cancelled chained attempt, a chained write by the chain's failure
// table, and an unchained write by its single re-send after an unknown
// outcome or its recorded outcome. An unfinished attempt whose completion
// arrived, that this actor cancelled (a cancelled request delivers no
// completion callback) or that is closing asks for another step within 1 ms.
void LayerActor::settleWrites(Clock::time_point now, Clock::time_point& next) {
    if (const auto hook = beforeSettleWrites.load(std::memory_order_acquire)) {
        const auto context = beforeSettleWritesContext.load(std::memory_order_acquire);
        const auto completed = [&] {
            size_t count = 0;
            for (const auto& write : writes_) count += write.request->wait(0) == OMNI_USD_LIVE_OK;
            return count;
        };
        for (auto count = completed(); count && hook(context, count); count = completed()) {}
    }
    for (size_t index = 0; index < writes_.size();) {
        auto& entry = writes_[index];
        if (entry.request->wait(0) != OMNI_USD_LIVE_OK) {
            if (entry.outcome->completed.load(std::memory_order_acquire) || entry.cancelled || closing_)
                next = std::min(next, now + std::chrono::milliseconds(1));
            ++index;
            continue;
        }
        OmniUsdLiveResult result;
        entry.request->result(&result);
        OmniUsdLiveRequestError error{};
        error.struct_size = sizeof(error);
        error.abi_version = OMNI_USD_LIVE_ABI_VERSION;
        entry.request->error(&error);
        const auto& receipt = *entry.outcome->receipt;
        const auto sequence = receipt.sequence.load(std::memory_order_acquire);
        const auto retryDelayMs = receipt.retryDelayMs.load(std::memory_order_acquire);
        // A replica without operation-ID support answers UNSPECIFIED and may
        // have applied an attempt that follows an unknown one a second time.
        if (entry.repeated && result == OMNI_USD_LIVE_OK &&
            !acknowledgesResend(receipt.outcome.load(std::memory_order_relaxed))) {
            result = OMNI_USD_LIVE_OUTCOME_UNKNOWN;
            error.result = result;
        }
        // The outcome is captured, so the attempt's registry slot is free for
        // a re-send before one is submitted.
        const auto released = runtime_->requests->release(entry.request->handle);
        assert(released == OMNI_USD_LIVE_OK);
        (void)released;
        InFlight write;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            write = std::move(entry);
            writes_.erase(writes_.begin() + std::ptrdiff_t(index));
        }
        runtime_->diagnostics->layerWriteInFlight(false);
        write.request.reset();
        if (write.cancelled) {
            settleCancelled(std::move(write), result, error, sequence, now);
            continue;
        }
        if (write.chained) {
            if (result == OMNI_USD_LIVE_UNAVAILABLE && backOffRefusal(write, error, retryDelayMs, now)) continue;
            settleChained(std::move(write), result, error, sequence, now);
            continue;
        }
        // An issued re-send's completion wakes the actor; one awaiting a
        // token is issued by step() once authenticate() succeeds.
        if (result == OMNI_USD_LIVE_OUTCOME_UNKNOWN && resendWrite(now, error) != Resend::NotOffered) continue;
        // Only a refusal that applied nothing settles an update UNAVAILABLE.
        if (result == OMNI_USD_LIVE_UNAVAILABLE && backOffRefusal(write, error, retryDelayMs, now)) continue;
        settleAttempt(write.id, write.authFailure, result, error, sequence, now);
        if (result == OMNI_USD_LIVE_OK && !terminalGate(progress_.snapshot().gate))
            noteCommitted(write.ordinal, write.effect->operationId);
    }
}

// The failure table of a chained write (design section 13). A refusal or an
// unknown outcome breaks the chain: the chained attempts of later writes are
// cancelled, issuance holds until every other write in flight settles, and
// resolveBreak re-sends from the first write not known to have committed. A
// write that cannot be re-sent keeps its outcome.
void LayerActor::settleChained(InFlight write, OmniUsdLiveResult result, const OmniUsdLiveRequestError& error,
    uint64_t sequence, Clock::time_point now) {
    const auto& receipt = *write.outcome->receipt;
    if (result == OMNI_USD_LIVE_OK) {
        settleAttempt(write.id, write.authFailure, result, error, sequence, now);
        if (terminalGate(progress_.snapshot().gate)) return;
        noteCommitted(write.ordinal, write.effect->operationId);
        if (write.ordinal == std::exchange(probeOrdinal_, 0)) {
            // Every earlier write of the chain committed before the probe did.
            for (const auto& work : progress_.retained())
                if (work.localOrdinal && work.localOrdinal < write.ordinal && work.attempts &&
                    work.disposition == LayerProgress::Disposition::Queued)
                    progress_.confirmCommitted(work.id);
        }
        breakRounds_ = 0;
        debtRefusals_ = 0;
        serializing_ = false;
        // A window of acknowledged writes without a refusal widens the window.
        if (++windowStreak_ >= window_) {
            windowStreak_ = 0;
            if (window_ < kMaxWindow) changeWindow(window_ + 1);
        }
        return;
    }
    windowStreak_ = 0;
    // Whether an earlier attempt of the write had an unknown outcome.
    const auto uncertain = [&] {
        for (const auto& work : progress_.retained())
            if (work.id == write.id) return work.mayHaveCommitted;
        return false;
    };
    if (deadOrdinal_ && write.ordinal > deadOrdinal_) {
        // An earlier write of the chain was rejected, so this one, which
        // commits only after it, never committed; the rejection stays the cause.
        settleAttempt(write.id, write.authFailure,
            result == OMNI_USD_LIVE_OUTCOME_UNKNOWN && uncertain() ? OMNI_USD_LIVE_OUTCOME_UNKNOWN :
                                                                     OMNI_USD_LIVE_CANCELLED,
            error, 0, now);
        return;
    }
    if (!chainsWrites(*endpoint_) && receipt.chainUnsupported.load(std::memory_order_acquire)) {
        // A chained call on an endpoint that stopped chaining, refused by a
        // replica without chaining: the attempt applied nothing and is
        // settled, so its write is sent in FIFO order, chained again while
        // it or a later write is unsettled. Those re-sends end after
        // reconnect_max_attempts such refusals in a row: the unsettled
        // attempt may still commit, so the layer fails OUTCOME_UNKNOWN.
        if (!progress_.requeueWrite(write.id, false)) {
            settleAttempt(write.id, write.authFailure,
                uncertain() ? OMNI_USD_LIVE_OUTCOME_UNKNOWN : OMNI_USD_LIVE_CANCELLED, error, 0, now);
            return;
        }
        if (write.ordinal <= chainDebtOrdinal()) {
            if (++debtRefusals_ >= runtime_->options.values.reconnect_max_attempts) {
                terminal(OMNI_USD_LIVE_OUTCOME_UNKNOWN);
                return;
            }
            resumeAt_ = std::max(resumeAt_,
                now + std::chrono::milliseconds(reconnectWait(*runtime_, subject_, debtRefusals_)));
        }
        return;
    }
    if (receipt.chainRefused.load(std::memory_order_acquire)) {
        // ABORTED PREDECESSOR_NOT_COMMITTED: this attempt applied nothing, and
        // the chain committed through the write it names. The successors in
        // flight wait at the service for this write, so they are cancelled;
        // one already committing lands right after it, and its re-send is
        // answered ALREADY_COMMITTED.
        if (!progress_.requeueWrite(write.id, false)) {
            settleAttempt(write.id, write.authFailure,
                uncertain() ? OMNI_USD_LIVE_OUTCOME_UNKNOWN : OMNI_USD_LIVE_CANCELLED, error, 0, now);
            return;
        }
        namedOperations_.emplace_back(receipt.lastRecord, receipt.lastOperation);
        chainUnresolved_ = true;
        breakChain(write.ordinal, true);
        return;
    }
    if (receipt.chainUnsupported.load(std::memory_order_acquire)) {
        // A replica that does not know the chained header, as during a
        // rolling upgrade or a rollback, refused it before applying anything.
        // The endpoint's writes stay unchained: the window drops to one, and
        // writes are re-sent unchanged and unchained from the first one not
        // known to have committed (resolveBreak), each once no chained call
        // carrying it or an earlier write may still commit. The chained calls
        // of later writes stay in flight: each commits only after this write.
        endpoint_->chainingRefused.store(true, std::memory_order_release);
        publishWindow();
        if (!progress_.requeueWrite(write.id, false)) {
            settleAttempt(write.id, write.authFailure,
                uncertain() ? OMNI_USD_LIVE_OUTCOME_UNKNOWN : OMNI_USD_LIVE_CANCELLED, error, 0, now);
            return;
        }
        chainUnresolved_ = true;
        breakChain(write.ordinal, false);
        return;
    }
    if (result == OMNI_USD_LIVE_UNAUTHENTICATED) {
        // Rejected before execution: one reissue once a usable token is
        // installed (design section 13); successors cannot commit before it.
        // A write whose earlier attempt may have committed settles
        // OUTCOME_UNKNOWN instead (completeWrite), and its successors may
        // commit after that attempt.
        const bool mayHaveCommitted = uncertain();
        settleAttempt(write.id, write.authFailure, result, error, 0, now);
        if (!mayHaveCommitted && terminalGate(progress_.snapshot().gate)) deadOrdinal_ = write.ordinal;
        breakChain(write.ordinal, true);
        return;
    }
    // A re-send of a write whose earlier attempt may have committed that the
    // service rejects ALREADY_EXISTS, OBJECT_REPLACED or NOT_FOUND is final:
    // the operation ID, the object or the layer does not match.
    const bool mayHaveCommitted = uncertain();
    const bool finalResend = mayHaveCommitted &&
        (result == OMNI_USD_LIVE_ALREADY_EXISTS || result == OMNI_USD_LIVE_OBJECT_REPLACED ||
            result == OMNI_USD_LIVE_NOT_FOUND);
    if (result != OMNI_USD_LIVE_OUTCOME_UNKNOWN &&
        (finalResend || receipt.settledRefusal.load(std::memory_order_acquire))) {
        // A definitive rejection that settles the call, one with the
        // service's ErrorInfo, fails the layer with its result. The refusal
        // that reaches the reissue limit settles a write whose earlier
        // attempt may have committed OUTCOME_UNKNOWN instead (completeWrite).
        // Either way, successors may commit after such an earlier attempt,
        // so only a write without one ends the chain.
        settleAttempt(write.id, write.authFailure, result, error, 0, now);
        if (!mayHaveCommitted) deadOrdinal_ = write.ordinal;
        breakChain(write.ordinal, true);
        return;
    }
    // Outcome unknown, or a status that does not settle the call, such as a
    // plain RESOURCE_EXHAUSTED or PERMISSION_DENIED without the service's
    // ErrorInfo: the write is re-sent unchanged at once, with no_wait, and
    // its successors in flight are cancelled. Its re-send commits at most
    // once: the service answers ALREADY_COMMITTED while it is the latest
    // write and refuses it at once, naming the latest, after any other.
    if (closing_ || !resendsUnknownWrites(*endpoint_) || !progress_.requeueWrite(write.id, true)) {
        settleAttempt(write.id, write.authFailure, OMNI_USD_LIVE_OUTCOME_UNKNOWN, error, 0, now);
        return;
    }
    cancelledUncertain_.erase(std::remove(cancelledUncertain_.begin(), cancelledUncertain_.end(), write.id),
        cancelledUncertain_.end());
    chainDebt_.push_back(write.id);
    chainUnresolved_ = true;
    breakChain(write.ordinal, true);
}

// A chained attempt this actor cancelled because an earlier write of the
// chain did not commit. An acknowledgment settles its write; otherwise the
// write is queued again, still possibly committed unless a refusal settled
// the attempt (an error with the service's ErrorInfo, or INVALID_ARGUMENT
// without one), and is re-sent in chain order. A write
// the chain is already proven to have committed through is confirmed
// instead; any other write the attempt may have committed is unsettled
// (chainDebt_) and breaks the chain again, so that its re-send, with no_wait,
// locates it.
void LayerActor::settleCancelled(InFlight write, OmniUsdLiveResult result, const OmniUsdLiveRequestError& error,
    uint64_t sequence, Clock::time_point now) {
    const auto& receipt = *write.outcome->receipt;
    if (result == OMNI_USD_LIVE_OK) {
        settleAttempt(write.id, write.authFailure, result, error, sequence, now);
        if (!terminalGate(progress_.snapshot().gate)) noteCommitted(write.ordinal, write.effect->operationId);
        debtRefusals_ = 0;
        return;
    }
    // Only a refusal that settles the call proves it applied nothing; any
    // other ending, whatever its mapped result, may still commit later.
    const bool refused = receipt.chainRefused.load(std::memory_order_acquire);
    const bool appliedNothing = receipt.settledRefusal.load(std::memory_order_acquire);
    bool uncertain = false;
    for (const auto& work : progress_.retained())
        if (work.id == write.id) uncertain = work.mayHaveCommitted;
    if (deadOrdinal_ && write.ordinal > deadOrdinal_) {
        // An earlier write of the chain was rejected, so this one, which
        // commits only after it, never committed, whatever the cancelled
        // attempt's result; the rejection stays the cause.
        settleAttempt(write.id, write.authFailure, uncertain ? OMNI_USD_LIVE_OUTCOME_UNKNOWN : OMNI_USD_LIVE_CANCELLED,
            error, 0, now);
        return;
    }
    if (!progress_.requeueWrite(write.id, !appliedNothing)) {
        settleAttempt(write.id, write.authFailure,
            uncertain || !appliedNothing ? OMNI_USD_LIVE_OUTCOME_UNKNOWN : OMNI_USD_LIVE_CANCELLED, error, 0, now);
        return;
    }
    if (refused && chainBreak_) namedOperations_.emplace_back(receipt.lastRecord, receipt.lastOperation);
    if (provenPrefix_ && (write.ordinal < provenPrefix_ || (write.ordinal == provenPrefix_ && !provenNamed_))) {
        progress_.confirmCommitted(write.id);
        return;
    }
    if (appliedNothing) return;
    if (!uncertain) cancelledUncertain_.push_back(write.id);
    chainDebt_.push_back(write.id);
    breakChain(write.ordinal, chainsWrites(*endpoint_));
}

// Holds issuance until the other writes in flight settle; with
// `cancelSuccessors`, cancels the attempts of later writes of the chain,
// which cannot commit before the write at `ordinal` does and would otherwise
// wait at the service. The break does not wait for a cancelled attempt.
void LayerActor::breakChain(uint64_t ordinal, bool cancelSuccessors) {
    chainBreak_ = true;
    if (!cancelSuccessors) return;
    std::array<std::shared_ptr<Request>, kMaxWindow> cancel;
    size_t count = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& write : writes_)
            if (write.chained && write.ordinal > ordinal && !write.cancelled && count < cancel.size()) {
                write.cancelled = true;
                cancel[count++] = write.request;
            }
    }
    for (size_t i = 0; i < count; ++i) cancel[i]->cancel(0);
}

size_t LayerActor::cancelledInFlight() const noexcept {
    size_t count = 0;
    for (const auto& write : writes_) count += write.cancelled;
    return count;
}

// The highest local ordinal of an unsettled write (chainDebt_) that is still
// queued or in flight, or zero; settled writes leave chainDebt_ here.
uint64_t LayerActor::chainDebtOrdinal() {
    if (chainDebt_.empty()) return 0;
    const auto work = progress_.retained();
    uint64_t ordinal = 0;
    chainDebt_.erase(std::remove_if(chainDebt_.begin(), chainDebt_.end(),
                         [&](uint64_t id) {
        for (const auto& entry : work)
            if (entry.id == id &&
                (entry.disposition == LayerProgress::Disposition::Queued ||
                    entry.disposition == LayerProgress::Disposition::Issued)) {
                ordinal = std::max(ordinal, entry.localOrdinal);
                return false;
            }
        return true;
    }),
        chainDebt_.end());
    return ordinal;
}

// Whether a chained attempt of a write with local ordinal at most `ordinal`,
// cancelled or not, has not returned.
bool LayerActor::chainedInFlightThrough(uint64_t ordinal) const noexcept {
    for (const auto& write : writes_)
        if (write.chained && write.ordinal <= ordinal) return true;
    return false;
}

void LayerActor::noteCommitted(uint64_t ordinal, const std::string& operation) {
    if (committedOperations_.size() >= 64) committedOperations_.pop_front();
    committedOperations_.emplace_back(ordinal, operation);
}

// The local ordinal of the write a chain refusal names as the writer's latest:
// zero for an absent record when this actor's first write was the writer's
// first to the object. False when the writer never issued it: an operation
// this actor does not know or has admitted without issuing, a write without an
// operation ID, or an absent record of an object the writer wrote before.
bool LayerActor::lookupOperation(LastRecord record, const std::string& operation,
    const std::vector<LayerProgress::Work>& work, uint64_t& ordinal) const {
    if (record == LastRecord::Absent && chainBaseEmpty_) {
        ordinal = 0;
        return true;
    }
    if (record != LastRecord::Operation) return false;
    for (const auto& entry : work)
        if (entry.localOrdinal && entry.attempts &&
            std::static_pointer_cast<const Effect>(entry.owner)->operationId == operation) {
            ordinal = entry.localOrdinal;
            return true;
        }
    for (const auto& [committed, name] : committedOperations_)
        if (name == operation) {
            ordinal = committed;
            return true;
        }
    return false;
}

// Once every write in flight of a broken chain that may precede the re-sends
// has settled, apart from cancelled attempts: the chain committed through the
// latest acknowledged write or named operation, so each queued write at or
// below it committed (a read started after the proof delivers it) except the
// latest named one, whose re-send the service answers with its position. A refusal
// naming an operation shows that no later write had committed. Every later
// write is re-sent unchanged, from the first, alone and with no_wait until it
// is acknowledged; one whose outcome is unknown stays so until then, and the
// read stays closed meanwhile (design section 12). While writes whose
// attempts had unknown outcomes remain and no refusal located the chain, the
// latest of them is re-sent first, alone and with no_wait: the service
// answers it at once, ALREADY_COMMITTED, committed or refused naming the
// chain's latest write. A write the writer never issued,
// such as after the service restores a saved state, fails the layer
// OUTCOME_UNKNOWN. On an endpoint that stopped chaining, chained calls of
// writes after the first re-send stay in flight.
void LayerActor::resolveBreak(Clock::time_point now) {
    if (!chainBreak_ || resendPending_) return;
    const auto work = progress_.retained();
    uint64_t first = UINT64_MAX;
    for (const auto& entry : work)
        if (entry.localOrdinal && entry.disposition == LayerProgress::Disposition::Queued)
            first = std::min(first, entry.localOrdinal);
    const bool chaining = chainsWrites(*endpoint_);
    for (const auto& write : writes_)
        if (!write.cancelled && (chaining || write.ordinal <= first)) return;
    chainBreak_ = false;
    const auto state = progress_.snapshot();
    if (terminalGate(state.gate)) {
        namedOperations_.clear();
        chainUnresolved_ = false;
        return;
    }
    uint64_t prefix = state.localRetired;
    bool named = false;
    // A write re-sent alone was issued once the chain was proven committed
    // through `provenPrefix_`, so a refusal of it naming an earlier operation
    // means the service lost records it held, as after a restore of saved state.
    const auto proven = serializing_ ? provenPrefix_ : 0;
    for (const auto& entry : work)
        if (entry.localOrdinal && entry.disposition == LayerProgress::Disposition::Acknowledged)
            prefix = std::max(prefix, entry.localOrdinal);
    // A refusal naming the latest acknowledged write or a later operation
    // locates the chain: no later write had committed.
    bool located = false;
    for (const auto& [record, operation] : namedOperations_) {
        uint64_t ordinal = 0;
        if (!lookupOperation(record, operation, work, ordinal) || ordinal < proven) {
            namedOperations_.clear();
            chainUnresolved_ = false;
            terminal(OMNI_USD_LIVE_OUTCOME_UNKNOWN);
            return;
        }
        located |= ordinal >= prefix;
        if (ordinal > prefix) {
            prefix = ordinal;
            named = true;
        }
    }
    namedOperations_.clear();
    // Nor did any write after one whose every attempt applied nothing.
    uint64_t unapplied = UINT64_MAX;
    for (const auto& entry : work)
        if (entry.localOrdinal > prefix && entry.disposition == LayerProgress::Disposition::Queued && entry.attempts &&
            !entry.mayHaveCommitted)
            unapplied = std::min(unapplied, entry.localOrdinal);
    uint64_t resent = 0, probe = 0, probeOrdinal = 0;
    bool uncertain = false;
    for (const auto& entry : work) {
        if (!entry.localOrdinal || entry.disposition != LayerProgress::Disposition::Queued || !entry.attempts) continue;
        if (entry.localOrdinal < prefix || (entry.localOrdinal == prefix && !named))
            progress_.confirmCommitted(entry.id);
        else {
            ++resent;
            // A write after one whose every attempt applied nothing waited at
            // the service for it, so it applied nothing. A write after the
            // located point had not committed when the service answered, but
            // an attempt of it the service handed to its store may still
            // commit, so it stays unsettled until a re-send of it is
            // acknowledged.
            if (entry.localOrdinal > unapplied) {
                if (entry.mayHaveCommitted) progress_.clearUncertain(entry.id);
            } else if (entry.mayHaveCommitted) {
                uncertain = true;
                if (entry.localOrdinal > probeOrdinal &&
                    std::find(cancelledUncertain_.begin(), cancelledUncertain_.end(), entry.id) ==
                        cancelledUncertain_.end()) {
                    probe = entry.id;
                    probeOrdinal = entry.localOrdinal;
                }
            }
        }
    }
    probeId_ = located ? 0 : probe;
    cancelledUncertain_.erase(std::remove_if(cancelledUncertain_.begin(), cancelledUncertain_.end(),
                                  [&](uint64_t id) {
        return std::none_of(work.begin(), work.end(), [&](const auto& entry) { return entry.id == id; });
    }),
        cancelledUncertain_.end());
    // A refusal that locates the chain beyond what was proven is progress, so
    // its round re-sends at once, as the first round does.
    const bool advanced = located && prefix > provenPrefix_;
    provenPrefix_ = prefix;
    provenNamed_ = named;
    const bool unresolved = std::exchange(chainUnresolved_, false);
    if (terminalGate(progress_.snapshot().gate) || !resent) return;
    // Re-sending the window at once would let a successor reach the service
    // before its predecessor's re-send, and the first re-send carries no_wait
    // so that the service answers it at once.
    serializing_ = true;
    noWait_ = true;
    // A break that only refusals which applied nothing caused has run its
    // backoff and counts toward their own bound.
    if (!unresolved) return;
    if (advanced) breakRounds_ = 0;
    ++breakRounds_;
    runtime_->diagnostics->layerChainResend(resent);
    if (breakRounds_ > runtime_->options.values.reconnect_max_attempts) {
        terminal(uncertain ? OMNI_USD_LIVE_OUTCOME_UNKNOWN : OMNI_USD_LIVE_UNAVAILABLE);
        return;
    }
    const uint64_t delay = breakRounds_ > 1 ? reconnectWait(*runtime_, subject_, uint32_t(breakRounds_ - 1)) : 0;
    resumeAt_ = std::max(resumeAt_, now + std::chrono::milliseconds(delay));
    runtime_->diagnostics->emit(OMNI_USD_LIVE_RECORD_RETRY_SCHEDULED, endpoint_->handle, subject_, 0,
        {{DiagnosticKey::Operation, OMNI_USD_LIVE_OPERATION_UPDATE},
            {DiagnosticKey::Result, uint64_t(uncertain ? OMNI_USD_LIVE_OUTCOME_UNKNOWN : OMNI_USD_LIVE_OK)},
            {DiagnosticKey::Attempt, breakRounds_}, {DiagnosticKey::Count, resent},
            {DiagnosticKey::TargetLocalOrdinal, prefix}, {DiagnosticKey::Reason, 3u}, {DiagnosticKey::DelayMs, delay}});
}

// Records a write's settled outcome, arms the auth wait or fails the layer;
// its request is already released.
void LayerActor::settleAttempt(uint64_t id, uint64_t authFailure, OmniUsdLiveResult result,
    const OmniUsdLiveRequestError& error, uint64_t sequence, Clock::time_point now) {
    bool closeRead = false;
    {
        std::lock_guard<std::mutex> lock(statusMutex_);
        const auto before = progress_.snapshot();
        const auto settlement = progress_.completeWrite(id, result, sequence);
        auto settled = progress_.snapshot();
        if (settled.readingAfterFailure && !callbacks_.applyDetached) {
            // A definitive rejection keeps the read open only for a
            // journal-only application; without one it closes the read.
            progress_.fail(settled.result);
            settled = progress_.snapshot();
            closeRead = true;
        }
        if (settlement == LayerProgress::WriteSettlement::AwaitAuth) {
            // The rejected attempt never executed (design section 13). Wait for
            // a usable token generation; the queued write is reissued
            // once after authenticate() succeeds. Refresh exhaustion
            // or a failure serial advanced since issuance fails the
            // layer UNAUTHENTICATED with this rejection as evidence.
            statusError_ = error;
            statusError_.result = OMNI_USD_LIVE_UNAUTHENTICATED;
            authEvidence_ = true;
            waitingAuth_ = true;
            authFailure_ = authFailure;
            authDeadline_ = now + std::chrono::milliseconds(endpoint_->options.authWaitMs);
        } else if (terminalGate(settled.gate)) {
            drain_->fail(settled.result, settled.readingAfterFailure && callbacks_.applyDetached);
            if (!terminalGate(before.gate) || before.result != settled.result) {
                statusError_ = error;
                statusError_.result = settled.result;
            }
        }
    }
    if (closeRead) cancelRead();
    if (id == writeId_) {
        writeEffect_.reset();
        writeResent_ = false;
        resendPending_ = false;
        writeId_ = 0;
    }
}

// A re-send awaiting a token on a closing or failed layer is never issued; its
// write's first attempt keeps an unknown outcome.
void LayerActor::abandonResend(Clock::time_point now) {
    if (!resendPending_) return;
    auto error = resendEvidence_;
    error.result = OMNI_USD_LIVE_OUTCOME_UNKNOWN;
    settleAttempt(writeId_, authFailure_, OMNI_USD_LIVE_OUTCOME_UNKNOWN, error, 0, now);
}

// Settles an ended read: releases it, then reconnects, waits for credentials
// or fails the layer by the read's result.
void LayerActor::settleRead(std::shared_ptr<Request>& read, Clock::time_point now) {
    OmniUsdLiveResult result;
    read->result(&result);
    OmniUsdLiveRequestError readError{};
    readError.struct_size = sizeof(readError);
    readError.abi_version = OMNI_USD_LIVE_ABI_VERSION;
    read->error(&readError);
    // A read this actor cancelled at its first-response deadline ends TIMEOUT.
    if (std::exchange(readTimedOut_, false) && result == OMNI_USD_LIVE_CANCELLED)
        result = readError.result = OMNI_USD_LIVE_TIMEOUT;
    readDeadline_ = Clock::time_point::max();
    const auto released = runtime_->requests->release(read->handle);
    assert(released == OMNI_USD_LIVE_OK);
    (void)released;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        read_.reset();
        assert(receivedDrainCount_ < receivedDrains_.size());
        receivedDrains_[receivedDrainCount_++] = std::move(readEndDrain_);
    }
    read.reset();
    readCompletionPending_ = false;
    const auto ended = progress_.snapshot();
    if (!closing_ && receiving(ended)) {
        // A layer reading after an issuance-only failure reconnects like a
        // healthy one; its status keeps the failure and its evidence.
        recordReadDisconnect(result, readError.request_id, readError.attempt_id);
        {
            std::lock_guard<std::mutex> lock(statusMutex_);
            progress_.disconnect();
            if (!terminalGate(ended.gate)) statusError_ = readError;
            authEvidence_ = result == OMNI_USD_LIVE_UNAUTHENTICATED;
        }
        if (result == OMNI_USD_LIVE_UNAUTHENTICATED) {
            waitingAuth_ = true;
            authFailure_ = readAuthFailure_;
            authDeadline_ = now + std::chrono::milliseconds(endpoint_->options.authWaitMs);
            retryAt_ = now;
        } else if (result == OMNI_USD_LIVE_UNAVAILABLE || result == OMNI_USD_LIVE_TIMEOUT ||
            result == OMNI_USD_LIVE_CANCELLED ||
            (result == OMNI_USD_LIVE_INTERNAL_ERROR && readError.grpc_status_present &&
                readError.grpc_status == int32_t(grpc::StatusCode::INTERNAL))) {
            // A peer's HTTP/2 CANCEL resets this read, not the logical
            // subscription, and the service ends a read with INTERNAL during a
            // backend incident; a read is not a mutation. Explicit
            // close/terminal cancellation is already excluded above. Resume
            // from retained complete coverage through the ordinary bounded
            // recovery gate; writes settle separately. attempts_ counts the
            // reads started since the last recovery readiness; once they reach
            // reconnect_max_attempts, startRead fails the layer at once.
            if (std::exchange(settleReconnect_, false) || attempts_ >= runtime_->options.values.reconnect_max_attempts)
                retryAt_ = now;
            else {
                const auto delay = reconnectWait(*runtime_, subject_, attempts_ + 1);
                retryAt_ = now + std::chrono::milliseconds(delay);
                runtime_->diagnostics->emit(OMNI_USD_LIVE_RECORD_RETRY_SCHEDULED, endpoint_->handle, subject_, 0,
                    {{DiagnosticKey::Operation, OMNI_USD_LIVE_OPERATION_READ}, {DiagnosticKey::Result, result},
                        {DiagnosticKey::Request, readError.request_id}, {DiagnosticKey::Attempt, readError.attempt_id},
                        {DiagnosticKey::Generation, readGeneration_}, {DiagnosticKey::ResumeSequence, readResume_},
                        {DiagnosticKey::DelayMs, delay}});
            }
        } else terminal(result, &readError);
    }
}

// A closing actor stops once no read, write, ping, frame or application
// remains; otherwise it steps again within 1 ms.
LayerActor::Clock::time_point LayerActor::stepClosing(bool read, bool write, bool queuedFrames, Clock::time_point now,
    Clock::time_point next) {
    bool pingPending, applicationPending;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        pingPending = bool(ping_);
        applicationPending = bool(application_);
    }
    if (!read && !write && !pingPending && !queuedFrames && !applicationPending) {
        if (const auto hook = beforeCloseQuiescent.load(std::memory_order_acquire))
            hook(beforeCloseQuiescentContext.load(std::memory_order_acquire));
        callbacks_ = {}; // Worker retirement before announcing quiescence.
        std::lock_guard<std::mutex> lock(mutex_);
        stopped_ = true;
        statusClosed_ = true;
    } else {
        if (read || write || pingPending || queuedFrames) next = std::min(next, now + std::chrono::milliseconds(1));
        if (closingRetained_) closingRetained_(closingRetainedContext_);
    }
    return next;
}

// Subscribes the next read generation. False when the subscription awaits
// credentials; the caller then steps again at once.
bool LayerActor::startRead(const std::weak_ptr<LayerActor>& weak, Clock::time_point now,
    std::shared_ptr<Request>& read) {
    if (attempts_ >= runtime_->options.values.reconnect_max_attempts)
        throw OmniUsdLiveResult(OMNI_USD_LIVE_UNAVAILABLE);
    const auto cursor = progress_.startRead();
    readGeneration_ = cursor.generation;
    readResume_ = cursor.sequence;
    readRequestId_ = 0;
    readReadyReported_ = false;
    jitter_.reset();
    nextPing_ = now;
    LayerReadCallbacks callbacks;
    callbacks.update = [weak, generation = cursor.generation](auto update) {
        if (auto actor = weak.lock()) actor->receive(generation, std::move(update));
    };
    callbacks.failed = [weak, generation = cursor.generation](OmniUsdLiveResult result) {
        auto actor = weak.lock();
        if (!actor) return false;
        // Latch under mutex_ so advance() cannot observe this failure
        // before its attribution; only the winning latch is attributed.
        std::lock_guard<std::mutex> lock(actor->mutex_);
        if (actor->drain_->fail(result)) {
            actor->readFailedGeneration_ = generation;
            actor->readFailure_ = result;
        }
        return true;
    };
    callbacks.closed = [weak](const auto&, std::shared_ptr<DrainTicket> drain) {
        if (auto actor = weak.lock()) {
            {
                std::lock_guard<std::mutex> lock(actor->mutex_);
                actor->readEndDrain_ = std::move(drain);
            }
            actor->readCompletionPending_ = true;
            actor->wake();
        }
    };
    LayerReadOptions options;
    options.objectId = cursor.object;
    options.sequence = cursor.sequence;
    options.subjectId = subject_;
    options.streamGeneration = cursor.generation;
    options.drain = drain_;
    options.flow = readFlow_;
    readAuthFailure_ = authFailure_;
    readCompletionPending_ = false;
    try {
        read = subscribeLayerRead(*runtime_, endpoint_->handle, progress_.identity.session, progress_.identity.layer,
            options, std::move(callbacks));
    } catch (OmniUsdLiveResult result) {
        progress_.disconnect();
        if (result != OMNI_USD_LIVE_UNAUTHENTICATED) throw;
        authEvidence_ = false;
        waitingAuth_ = true;
        authDeadline_ = now + std::chrono::milliseconds(endpoint_->options.authWaitMs);
        return false;
    } catch (...) {
        progress_.disconnect();
        throw;
    }
    ++attempts_;
    // Each attempt has as long for its first complete response, the catch-up
    // target, as an initial open has for the layer's content (design section 12).
    readDeadline_ = now + std::chrono::milliseconds(runtime_->options.values.initial_open_timeout_ms);
    readRequestId_ = read->handle;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        read_ = read;
    }
    return true;
}

// The window the next chained write issues under: one write while an
// endpoint shows no operation-ID support or a broken chain re-sends its first
// unconfirmed write.
uint32_t LayerActor::chainWindow() const noexcept { return chainsWrites(*endpoint_) && !serializing_ ? window_ : 1; }

void LayerActor::changeWindow(uint32_t window) noexcept {
    const bool grew = window > window_;
    window_ = window;
    windowForTest_.store(window, std::memory_order_release);
    runtime_->diagnostics->layerWindowChanged(window, grew);
}

void LayerActor::publishWindow() noexcept { effectiveWindow_.store(chainWindow(), std::memory_order_release); }

// Admitted writes, in flight or queued, never exceed the window the next
// write issues under, one write on an endpoint that does not chain writes,
// and at most one admitted write waits to be issued: host turns admit no
// faster than the worker issues them (design section 13).
bool LayerActor::issuanceSaturated() const noexcept {
    try {
        const auto state = progress_.snapshot();
        return state.queued || state.issued + state.queued >= effectiveWindow_.load(std::memory_order_acquire);
    } catch (...) { return false; }
}

// Issues queued writes in FIFO order while the window allows: a chained write
// beside other chained writes, and an unchained one alone on an endpoint that
// chains writes. On one that stopped chaining, an unchained write waits until
// no chained call carrying it or an earlier write may still commit; chained
// calls of later writes may stay in flight. Cancelled attempts do not count
// toward the window, but while one is in flight only a broken chain's first
// re-send issues, so that writes keep their order. Each is first published
// locally when the host publishes local work. False when a write awaits
// credentials; the caller then steps again at once.
bool LayerActor::issueWrites(const std::weak_ptr<LayerActor>& weak, Clock::time_point now, Clock::time_point& next) {
    const auto hostResult = hostFailureResult();
    if (hostResult != OMNI_USD_LIVE_OK) throw hostResult;
    publishWindow();
    if (!resendHoldInherited_) {
        // A re-send by an earlier layer of this endpoint on the object holds
        // this layer's writes too: they carry the same writer ID.
        const auto key = objectKey();
        std::lock_guard<std::mutex> lock(endpoint_->mutex);
        const auto found = endpoint_->resendHolds.find(key);
        if (found != endpoint_->resendHolds.end() && found->second > resendHold_) resendHold_ = found->second;
        resendHoldInherited_ = true;
    }
    for (;;) {
        if (chainBreak_ || resendPending_) return true;
        if (now < issuanceResumes(now)) {
            next = std::min(next, issuanceResumes(now));
            return true;
        }
        for (const auto& write : writes_)
            if (!write.chained) return true;
        if (!localReady()) return true;
        const bool marked = chainsWrites(*endpoint_);
        const size_t cancelled = cancelledInFlight();
        if (marked && cancelled && !serializing_ && !probeId_) return true;
        const size_t window = marked ? (serializing_ ? 1 : window_) + cancelled : writes_.size() + 1;
        auto work = probeId_ ? progress_.takeWriteOutOfOrder(probeId_, window) : progress_.takeWrite(window);
        if (!work.id) return true;
        auto effect = std::static_pointer_cast<const Effect>(work.owner);
        // After the endpoint stopped chaining, a write at or below an
        // unsettled one is still sent chained, with no_wait, so that the
        // chain settles it; any write waits until every chained call of it
        // or an earlier write has returned.
        const bool debt = !marked && effect->chainable && work.localOrdinal <= chainDebtOrdinal();
        const bool chained = (marked && effect->chainable) || debt;
        if ((!chained && marked && !writes_.empty()) || (!marked && chainedInFlightThrough(work.localOrdinal))) {
            progress_.deferWrite(work.id);
            return true;
        }
        if (work.id == probeId_) {
            probeId_ = 0;
            probeOrdinal_ = work.localOrdinal;
        }
        InFlight write;
        try {
            write = submitWrite(weak, effect, work.id, work.localOrdinal, chained, debt || (chained && noWait_));
        } catch (OmniUsdLiveResult result) {
            progress_.deferWrite(work.id);
            if (result != OMNI_USD_LIVE_UNAUTHENTICATED) throw;
            authEvidence_ = false;
            waitingAuth_ = true;
            authDeadline_ = now + std::chrono::milliseconds(endpoint_->options.authWaitMs);
            return false;
        } catch (...) {
            progress_.deferWrite(work.id);
            throw;
        }
        noWait_ = false;
        write.repeated = work.mayHaveCommitted;
        if (!chained) {
            writeEffect_ = effect;
            writeId_ = work.id;
            writeOrdinal_ = work.localOrdinal;
            writeResent_ = false;
            writeIssuedAt_ = Clock::now();
        }
        track(std::move(write));
    }
}

// Submits one attempt of a local write with its bytes, operation ID and, when
// chained, its predecessor and no_wait; the attempt's completion wakes the
// actor.
LayerActor::InFlight LayerActor::submitWrite(const std::weak_ptr<LayerActor>& weak,
    std::shared_ptr<const Effect> effect, uint64_t id, uint64_t ordinal, bool chained, bool noWait) {
    const auto object = progress_.objectId();
    if (!objectWritten_) {
        // A later layer of this endpoint on the object cannot name this
        // write as its first write's predecessor. An actor writes one object.
        auto key = objectKey();
        std::lock_guard<std::mutex> lock(endpoint_->mutex);
        endpoint_->writtenObjects.insert(std::move(key));
        objectWritten_ = true;
    }
    auto outcome = std::make_shared<WriteOutcome>();
    InFlight write;
    write.request = submitLayerUpdate(*runtime_, endpoint_->handle, progress_.identity.session,
        progress_.identity.layer, object, 0, effect->authored,
        [weak, outcome](const auto&, const auto&, uint64_t) {
        outcome->completed.store(true, std::memory_order_release);
        if (auto actor = weak.lock()) actor->wake();
    }, 0, nullptr, outcome->receipt, {subject_, readGeneration_, id}, effect->reservation, effect->operationId,
        chained ? &effect->previousOperationId : nullptr, noWait);
    write.id = id;
    write.ordinal = ordinal;
    write.authFailure = authFailure_;
    write.outcome = std::move(outcome);
    write.effect = std::move(effect);
    write.chained = chained;
    return write;
}

void LayerActor::track(InFlight write) {
    const auto window = write.chained ? chainWindow() : 1;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        writes_.push_back(std::move(write)); // Within the capacity reserved at construction.
    }
    runtime_->diagnostics->layerWriteInFlight(true, window);
}

void LayerActor::noteCoalescedTurns(uint64_t turns) noexcept { runtime_->diagnostics->layerUpdateAdmitted(turns); }

// Offers the unchained write, whose attempt has completed with an unknown
// outcome and been released, its single re-send with the same bytes,
// operation ID and writer ID, on an endpoint whose replies have shown
// operation-ID support (design section 13). The service answers with the
// original commit if the first attempt committed. The service matches a
// re-send only before this writer's next write to the object commits, so it
// is not offered while a chained call of a later write, which stays in flight
// after the endpoint stops chaining, has not returned or ended unsettled
// (chainDebt_), since it may still commit. The re-send issues at once, but no
// later write issues until the first attempt's request_timeout_ms deadline
// plus kResendHoldMargin has passed: the service may still hold that attempt,
// and drops it by then. NotOffered leaves the write for the caller to settle
// as OUTCOME_UNKNOWN.
LayerActor::Resend LayerActor::resendWrite(Clock::time_point now, const OmniUsdLiveRequestError& error) {
    bool laterChained = chainDebtOrdinal() > writeOrdinal_;
    for (const auto& write : writes_) laterChained |= write.chained && write.ordinal > writeOrdinal_;
    if (closing_ || writeResent_ || !writeEffect_ || !resendsUnknownWrites(*endpoint_) || laterChained ||
        !progress_.resendWrite(writeId_))
        return Resend::NotOffered;
    // Every other write in flight is a chained call past the point where it may commit.
    assert(progress_.snapshot().issued == 1 + writes_.size());
    writeResent_ = true;
    resendEvidence_ = error;
    const auto hold = writeIssuedAt_ + std::chrono::milliseconds(runtime_->options.values.request_timeout_ms) +
        kResendHoldMargin;
    if (hold > resendHold_) resendHold_ = hold;
    resendHoldExpiredForTest_.store(false, std::memory_order_release);
    {
        // The endpoint keeps the hold for a later layer on the object; holds
        // that have passed are dropped.
        auto key = objectKey();
        std::lock_guard<std::mutex> lock(endpoint_->mutex);
        auto& holds = endpoint_->resendHolds;
        for (auto entry = holds.begin(); entry != holds.end();)
            entry = entry->second <= now ? holds.erase(entry) : std::next(entry);
        auto& held = holds[std::move(key)];
        if (resendHold_ > held) held = resendHold_;
    }
    // reason 4: the hold on later writes while the first attempt may still
    // be held by the service; delay_ms is its remaining length.
    const auto delay =
        resendHold_ > now ? uint64_t(std::chrono::ceil<std::chrono::milliseconds>(resendHold_ - now).count()) : 0u;
    runtime_->diagnostics->emit(OMNI_USD_LIVE_RECORD_RETRY_SCHEDULED, endpoint_->handle, subject_, 0,
        {{DiagnosticKey::Operation, OMNI_USD_LIVE_OPERATION_UPDATE},
            {DiagnosticKey::Result, OMNI_USD_LIVE_OUTCOME_UNKNOWN}, {DiagnosticKey::Request, error.request_id},
            {DiagnosticKey::Generation, readGeneration_}, {DiagnosticKey::Work, writeId_}, {DiagnosticKey::Reason, 4u},
            {DiagnosticKey::DelayMs, delay}});
    return issueResend(now);
}

// Submits the offered re-send. A token that is not usable defers it to
// step(), which issues it once authenticate() succeeds; it stays the write's
// only re-send, and the auth wait's expiry fails the layer OUTCOME_UNKNOWN.
// Any other submission failure leaves the outcome unknown.
LayerActor::Resend LayerActor::issueResend(Clock::time_point now) {
    const auto weak = std::weak_ptr<LayerActor>(std::static_pointer_cast<LayerActor>(shared_from_this()));
    InFlight write;
    try {
        write = submitWrite(weak, writeEffect_, writeId_, writeOrdinal_, false);
    } catch (OmniUsdLiveResult result) {
        if (result == OMNI_USD_LIVE_UNAUTHENTICATED) {
            if (!resendPending_) {
                resendPending_ = true;
                authEvidence_ = false;
                waitingAuth_ = true;
                authDeadline_ = now + std::chrono::milliseconds(endpoint_->options.authWaitMs);
            }
            return Resend::AwaitingAuth;
        }
        resendPending_ = false;
        return Resend::NotOffered;
    } catch (...) {
        resendPending_ = false;
        return Resend::NotOffered;
    }
    resendPending_ = false;
    write.repeated = true;
    track(std::move(write));
    return Resend::Issued;
}

// Keeps a write whose attempt the service refused UNAVAILABLE with nothing
// applied, and which has been released, for its reissue with the same bytes,
// operation ID and writer ID after a backoff (design section 13): the longer
// of the service's retry delay and the reconnect schedule's draw, capped at
// reconnect_maximum_ms, when it gave one, otherwise the reconnect schedule,
// from reconnectWait. The refusal is a capacity signal, so it halves the
// issuance window; a chained write's successors in flight wait at the service
// for it, so they are cancelled and re-sent after it. The write stays at its
// FIFO position and issuance resumes once the backoff passes. False leaves
// the write for the caller to settle UNAVAILABLE: a closing layer, an
// unchained write's re-send, or the refusal that reaches
// reconnect_max_attempts.
bool LayerActor::backOffRefusal(const InFlight& write, const OmniUsdLiveRequestError& error, uint64_t retryDelayMs,
    Clock::time_point now) {
    if (closing_ || (!write.chained && writeResent_)) return false;
    uint32_t refusals;
    {
        std::lock_guard<std::mutex> lock(statusMutex_);
        refusals = progress_.refuseWrite(write.id, runtime_->options.values.reconnect_max_attempts);
    }
    if (!refusals) return false;
    const bool hinted = retryDelayMs != kNoRetryDelay;
    const auto delay = reconnectWait(*runtime_, subject_, refusals,
        hinted ? std::optional<uint64_t>(retryDelayMs) : std::nullopt);
    resumeAt_ = std::max(resumeAt_, now + std::chrono::milliseconds(delay));
    if (window_ > 1) changeWindow(window_ / 2);
    windowStreak_ = 0;
    // reason is the delay's source: 1 the service's RetryInfo, 2 the
    // reconnect schedule.
    runtime_->diagnostics->emit(OMNI_USD_LIVE_RECORD_RETRY_SCHEDULED, endpoint_->handle, subject_, 0,
        {{DiagnosticKey::Operation, OMNI_USD_LIVE_OPERATION_UPDATE}, {DiagnosticKey::Result, error.result},
            {DiagnosticKey::Request, error.request_id}, {DiagnosticKey::Attempt, refusals},
            {DiagnosticKey::Generation, readGeneration_}, {DiagnosticKey::Work, write.id},
            {DiagnosticKey::Reason, hinted ? 1u : 2u}, {DiagnosticKey::DelayMs, delay}});
    if (write.chained) breakChain(write.ordinal, true);
    if (write.id == writeId_) {
        writeEffect_.reset();
        writeResent_ = false;
        resendPending_ = false;
        writeId_ = 0;
    }
    return true;
}

// A read open when the chain proved a write committed may have filtered that
// write's commit, and the service reports no position for it. Once settling
// ends, with no write in flight or of unknown outcome, the read is resubscribed
// from the coverage frontier: a read started after the proof delivers the
// commit in its catch-up, in service order with peers' commits (design
// section 12). That read's generation releases every such write, so one
// settle resubscribes once.
void LayerActor::resubscribeAfterSettle() {
    const auto state = progress_.snapshot();
    const bool forced = resubscribeForTest_.exchange(false, std::memory_order_acq_rel);
    if ((forced || (state.awaitingRead && !state.uncertain)) && writes_.empty() && !settleReconnect_ &&
        state.gate == LayerProgress::Gate::Ready) {
        settleReconnect_ = true;
        cancelRead();
    }
}

// A read attempt without its first complete response by its deadline is
// cancelled; it ends TIMEOUT and counts as a failed reconnect attempt (design
// section 12). A read this layer holds for incoming pacing is not stalled, so
// its window restarts while the hold lasts.
void LayerActor::checkReadDeadline(bool read, bool queuedFrames, Clock::time_point now, Clock::time_point& next) {
    if (!read || readDeadline_ == Clock::time_point::max()) {
        if (!queuedFrames) readExpiredForTest_.store(false, std::memory_order_release);
        return;
    }
    bool paused;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        paused = readPaused_;
    }
    const auto window = std::chrono::milliseconds(runtime_->options.values.initial_open_timeout_ms);
    if (paused) {
        if (now >= readDeadline_) heldReadWindows_.fetch_add(1, std::memory_order_acq_rel);
        readDeadline_ = now + window;
    } else if (!queuedFrames &&
        (readExpiredForTest_.exchange(false, std::memory_order_acq_rel) || now >= readDeadline_)) {
        readDeadline_ = Clock::time_point::max();
        readTimedOut_ = true;
        cancelRead();
    }
    next = std::min(next, readDeadline_);
}

LayerActor::Clock::time_point LayerActor::step() {
    auto next = Clock::time_point::max();
    const auto now = Clock::now();
    settleAppliedEffects();
    checkReadAuthorization(now);
    if (closing_) cancelBoth();
    if (!receiving(progress_.snapshot())) cancelRead();
    pollPing(now, next);
    consumeQueuedFrames();
    std::shared_ptr<Request> read;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        read = read_;
    }
    settleWrites(now, next);
    resolveBreak(now);
    if (read) resubscribeAfterSettle();
    bool queuedFrames;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        queuedFrames = !events_.empty();
    }
    if (queuedFrames) next = now;
    if (read && !queuedFrames && read->wait(0) == OMNI_USD_LIVE_OK) settleRead(read, now);
    if (!closing_) checkReadDeadline(bool(read), queuedFrames, now, next);
    // A read this actor cancelled has no completion callback; poll it until it ends.
    if (read && (readCompletionPending_.load(std::memory_order_acquire) || settleReconnect_ || readTimedOut_))
        next = std::min(next, now + std::chrono::milliseconds(1));
    // Offer the queued local head the shared host slot before processing the
    // next peer/ACK effect. This prevents a continuous peer stream starving
    // publication of admitted local work.
    if (!closing_) (void)localReady();
    if (!closing_ && applyReady(next)) next = now;
    publish();
    if (resendPending_ && (closing_ || !receiving(progress_.snapshot()) || terminalGate(progress_.snapshot().gate)))
        abandonResend(now);
    publishWindow();
    const auto state = progress_.snapshot();
    const bool inFlight = !writes_.empty();
    if (closing_) return stepClosing(bool(read), inFlight, queuedFrames, now, next);
    if (!receiving(state)) return read || inFlight ? std::min(next, now + std::chrono::milliseconds(1)) : next;
    // A re-send awaiting a token keeps the write in flight without a request.
    const bool writing = inFlight || resendPending_;
    const bool issuing = !terminalGate(state.gate);
    // Whether the window admits another write: none beside an unchained one,
    // and none while a refusal's backoff or a re-send hold runs.
    const bool due = now >= issuanceResumes(now);
    if (!due) next = std::min(next, issuanceResumes(now));
    size_t active = 0;
    bool unchainedInFlight = false;
    for (const auto& write : writes_) {
        active += !write.cancelled;
        unchainedInFlight |= !write.chained;
    }
    // Chained calls left in flight after the endpoint stopped chaining do not
    // fill the window; issueWrites decides when an unchained write may go.
    const bool room = !chainBreak_ && due && !unchainedInFlight &&
        (!chainsWrites(*endpoint_) || active < effectiveWindow_.load(std::memory_order_relaxed));
    if (!initialApplicationReady(state)) {
        const auto deadline = openExpiredForTest_.load(std::memory_order_acquire) ? now : openDeadline_;
        if (now >= deadline) throw OmniUsdLiveResult(expiredWait(now, deadline, waitingAuth_.load()));
        next = std::min(next, openDeadline_);
    }
    if ((!read && !writing) || (issuing && state.queued && room) || resendPending_) {
        if (!authenticate(now, next)) return next;
    }
    if (resendPending_) {
        // The re-send's state is worker-owned: a host failure or close only
        // latches its cause and wakes the worker, which abandons it here.
        if (closing_ || terminalGate(progress_.snapshot().gate)) {
            abandonResend(now);
            return now;
        }
        if (issueResend(now) == Resend::NotOffered) {
            auto error = resendEvidence_;
            error.result = OMNI_USD_LIVE_OUTCOME_UNKNOWN;
            settleAttempt(writeId_, authFailure_, OMNI_USD_LIVE_OUTCOME_UNKNOWN, error, 0, now);
        }
        return now;
    }
    const auto weak = std::weak_ptr<LayerActor>(std::static_pointer_cast<LayerActor>(shared_from_this()));
    if (!read && !writing && !state.uncertain && now >= retryAt_) { // Settle first (design section 12).
        if (!startRead(weak, now, read)) return now;
        next = std::min(next, now); // Start the generation's asynchronous Ping.
    } else if (!read && !writing && !state.uncertain) next = std::min(next, retryAt_);
    if (issuing && room && !closing_ && !readCompletionPending_.load(std::memory_order_acquire) &&
        initialApplicationReady(state) && progress_.snapshot().queued) {
        if (!issueWrites(weak, now, next)) return now;
    }
    return next;
}

void LayerActor::retireAppliedDrain() noexcept {
    // Every caller has published the effect, or latched the failure that
    // suppresses its publication. The clean save shortcut opens
    // before the tickets complete, so a save their completion wakes, and any
    // later save, observes the retired state.
    localRetirementPending_.store(false, std::memory_order_release);
    const bool retiring = !pendingDrains_.empty() || !absorbedOwners_.empty() || !failedAbsorbedOwners_.empty();
    for (auto& drain : pendingDrains_) {
        if (drain) drain->complete();
        drain.reset();
    }
    pendingDrains_.clear();
    completeAbsorbed(absorbedOwners_);
    completeAbsorbed(failedAbsorbedOwners_);
    if (retiring)
        if (const auto hook = afterDrainRetired.load(std::memory_order_acquire))
            hook(afterDrainRetiredContext.load(std::memory_order_acquire));
}

OmniUsdLiveResult LayerActor::adoptAbsorbed(std::vector<std::shared_ptr<const void>> owners) noexcept {
    if (absorbedOwners_.empty()) {
        absorbedOwners_.swap(owners);
        return OMNI_USD_LIVE_OK;
    }
    try {
        absorbedOwners_.insert(absorbedOwners_.end(), owners.begin(), owners.end());
    } catch (const std::bad_alloc&) {
        failedAbsorbedOwners_.swap(owners);
        return OMNI_USD_LIVE_RESOURCE_EXHAUSTED;
    } catch (const std::length_error&) {
        failedAbsorbedOwners_.swap(owners);
        return OMNI_USD_LIVE_RESOURCE_EXHAUSTED;
    } catch (...) {
        failedAbsorbedOwners_.swap(owners);
        return OMNI_USD_LIVE_INTERNAL_ERROR;
    }
    return OMNI_USD_LIVE_OK;
}

void LayerActor::completeAbsorbed(std::vector<std::shared_ptr<const void>>& owners) noexcept {
    // Each absorbed intent's original ticket retires with the coverage that
    // contained its effect, after that coverage's publication.
    for (auto& owner : owners) {
        auto drain = std::static_pointer_cast<const Effect>(owner)->drain;
        owner.reset();
        if (drain) drain->complete();
    }
    owners.clear();
}

OmniUsdLiveResult LayerActor::save(uint32_t timeoutMs, bool allowBlock) noexcept {
    const auto started = Clock::now();
    uint64_t target = 0;
    const auto result = saveImpl(timeoutMs, allowBlock, &target);
    const auto elapsed = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - started).count());
    // Observations cannot change the caller's result. They describe the
    // captured local boundary and current retirement/counts, not peer delivery
    // or atomic publication of a status/scene/diagnostic snapshot.
    try {
        const auto state = progress_.snapshot();
        OmniUsdLiveStatus status{};
        status.struct_size = sizeof(status);
        status.abi_version = OMNI_USD_LIVE_ABI_VERSION;
        (void)runtime_->statuses->snapshot(subject_, &status);
        const auto severity = result == OMNI_USD_LIVE_OK || result == OMNI_USD_LIVE_BUSY ?
            OMNI_USD_LIVE_SEVERITY_DEBUG :
            (result == OMNI_USD_LIVE_TIMEOUT || result == OMNI_USD_LIVE_CANCELLED ? OMNI_USD_LIVE_SEVERITY_WARN :
                                                                                    OMNI_USD_LIVE_SEVERITY_ERROR);
        runtime_->diagnostics->emit(OMNI_USD_LIVE_RECORD_SAVE_COMPLETED, endpoint_->handle, subject_,
            result == OMNI_USD_LIVE_OK ? 0 : status.incident_id,
            {{DiagnosticKey::Operation, OMNI_USD_LIVE_OPERATION_SAVE}, {DiagnosticKey::Result, result},
                {DiagnosticKey::Reason, state.result}, {DiagnosticKey::Generation, state.generation},
                {DiagnosticKey::TargetLocalOrdinal, target}, {DiagnosticKey::LocalRetired, state.localRetired},
                {DiagnosticKey::PendingWrites, state.pendingWrites}, {DiagnosticKey::PendingBytes, state.pendingBytes},
                {DiagnosticKey::ReadSequence, state.readSequence}, {DiagnosticKey::AppliedSequence, state.applied},
                {DiagnosticKey::DurationMs, elapsed}},
            {}, severity);
    } catch (...) { /* Diagnostics must not change save completion. */
    }
    return result;
}

OmniUsdLiveResult LayerActor::saveImpl(uint32_t timeoutMs, bool allowBlock, uint64_t* targetLocalOrdinal) noexcept {
    const auto deadline = Clock::now() + std::chrono::milliseconds(timeoutMs);
    try {
        (void)observeHostFailure();
        DrainRegistry::Capture captured;
        {
            std::lock_guard<std::mutex> lock(admissionMutex_);
            const auto state = progress_.snapshot();
            if (targetLocalOrdinal) *targetLocalOrdinal = state.localAdmitted;
            if (state.gate == LayerProgress::Gate::Failed) return state.result;
            if (closing_) return OMNI_USD_LIVE_CANCELLED;
            const auto hostResult = hostFailureResult();
            if (hostResult != OMNI_USD_LIVE_OK) throw hostResult;
            if (!state.pendingWrites && !localRetirementPending_.load(std::memory_order_acquire))
                return OMNI_USD_LIVE_OK;
            captured.targets.push_back({drain_, drain_->capture()});
        }
        const auto result = runtime_->drains->wait(captured, deadline, allowBlock && !DispatchExecutor::isWorker());
        return result == OMNI_USD_LIVE_OK ? observeHostFailure() : result;
    } catch (OmniUsdLiveResult result) {
        (void)observeHostFailure();
        return result;
    } catch (const std::bad_alloc&) { return OMNI_USD_LIVE_RESOURCE_EXHAUSTED; } catch (const std::length_error&) {
        return OMNI_USD_LIVE_RESOURCE_EXHAUSTED;
    } catch (...) { return OMNI_USD_LIVE_INTERNAL_ERROR; }
}

void LayerActor::markAdmissionRetiring() noexcept {
    if (admission_) admission_->markRetiring();
}
OmniUsdLiveResult LayerActor::close(uint32_t timeoutMs) {
    const auto until = Clock::now() + std::chrono::milliseconds(timeoutMs);
    bool first;
    {
        std::lock_guard<std::mutex> lock(admissionMutex_);
        first = !closing_.exchange(true);
    }
    {
        // A closing layer admits nothing, so it is not a pending-byte watcher:
        // an endpoint release requests no flush of its buffer, which the host
        // binding may outlive it with. Another actor sharing the buffer keeps
        // its own entry.
        std::lock_guard<std::mutex> lock(endpoint_->mutex);
        if (std::exchange(watchingPendingBytes_, false)) {
            auto& watchers = endpoint_->pendingByteWatchers;
            const auto own = std::find_if(watchers.begin(), watchers.end(),
                [&](const auto& watcher) { return watcher.lock() == bufferedEdits_; });
            if (own != watchers.end()) watchers.erase(own);
        }
    }
    std::shared_ptr<ApplyCompletion::State> application;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        application = application_;
        // The actor stops only after that application's host job retires;
        // until then an open of this resource on that host must not wait for
        // the key. Pin and unpin (settleApplication) both run under mutex_.
        if (first && application && admission_) admission_->pinToHost(true);
        closed_.notify_all();
    }
    if (application) application->cancelQueued();
    drain_->close();
    publishStatus();
    if (first) reportDiscardedHostEdits();
    cancelBoth();
    wake();
    std::unique_lock<std::mutex> lock(mutex_);
    if (stopped_ && !scheduled_) return OMNI_USD_LIVE_OK;
    if (DispatchExecutor::isWorker()) return OMNI_USD_LIVE_BUSY;
    return closed_.wait_until(lock, until, [&] { return stopped_ && !scheduled_; }) ? OMNI_USD_LIVE_OK :
                                                                                      OMNI_USD_LIVE_TIMEOUT;
}
} // namespace omni::usd::live
