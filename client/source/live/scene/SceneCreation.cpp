// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "SceneCreation.h"
#include "LiveAuth.h"
#include "LiveLayerWrite.h"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace omni::usd::live {
namespace {
using Clock = std::chrono::steady_clock;
OmniUsdLiveRequestError errorRecord(OmniUsdLiveResult result) noexcept {
    OmniUsdLiveRequestError error{};
    error.struct_size = sizeof(error);
    error.abi_version = OMNI_USD_LIVE_ABI_VERSION;
    error.operation = OMNI_USD_LIVE_OPERATION_CREATE;
    error.result = result;
    return error;
}
OmniUsdLiveResult decodeResult(const wire::DecodeError& error) noexcept {
    switch (error.kind) {
    case wire::ErrorKind::ResourceLimit: return OMNI_USD_LIVE_RESOURCE_EXHAUSTED;
    case wire::ErrorKind::Unsupported: return OMNI_USD_LIVE_UNSUPPORTED;
    case wire::ErrorKind::Malformed:
    case wire::ErrorKind::Integrity: return OMNI_USD_LIVE_MALFORMED_STREAM;
    }
    return OMNI_USD_LIVE_INTERNAL_ERROR;
}
void charge(size_t amount, size_t& remaining) {
    if (amount > remaining) throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    remaining -= amount;
}
} // namespace

namespace {
std::atomic<bool (*)(void*) noexcept> beforeResendForTest{nullptr};
std::atomic<void*> beforeResendContext{nullptr};
} // namespace

void SceneCreation::setBeforeResendForTest(bool (*hook)(void*) noexcept, void* context) noexcept {
    beforeResendContext.store(context, std::memory_order_release);
    beforeResendForTest.store(hook, std::memory_order_release);
}

struct SceneCreation::State final : ScheduledOperation, DispatchJob {
    // Scheduler retention is temporary: advance returns done after cleanup.
    // No callback captures this state or Runtime. Owner destruction wakes the
    // reserved operation even if start() was never called.
    const std::shared_ptr<Runtime> runtime;
    const std::shared_ptr<LayerAdmission> reservation;
    const std::shared_ptr<LayerStatus> status;
    const PendingBytes wireBudget, sceneBudget;
    const std::string wire;
    const WireScenePtr scene;
    const std::shared_ptr<LayerWriteReceipt> receipt = std::make_shared<LayerWriteReceipt>();
    // The creation's write identity; its single re-send after an unknown outcome repeats it with the same bytes.
    const std::string operationId = newOperationId();
    mutable std::mutex mutex;
    std::condition_variable changed;
    Snapshot outcome;
    std::shared_ptr<Request> request;
    // A submission is being prepared: by start(), or by dispatch() for the
    // re-send or a reissue. The scheduler retains this state meanwhile.
    bool starting = false;
    bool resent = false; // The re-send has been offered.
    // Consecutive attempts the service refused with nothing applied, and the
    // backoff before the next one's reissue (design section 13). A reissue
    // goes out only before openDeadline: the caller's initial-open deadline
    // from boundReissues(), or initial_open_timeout_ms counted from start().
    // An abandoned opening issues no reissue.
    uint32_t refusals = 0;
    bool backingOff = false, reissuing = false, abandoned = false;
    Clock::time_point reissueAt{}, openDeadline{};
    std::optional<Clock::time_point> callerDeadline;

    State(std::shared_ptr<Runtime> r, std::shared_ptr<LayerAdmission> admission, std::shared_ptr<LayerStatus> subject,
        PendingBytes wireCharge, PendingBytes sceneCharge, std::string bytes, WireScenePtr seed)
        : runtime(std::move(r)), reservation(std::move(admission)), status(std::move(subject)),
          wireBudget(std::move(wireCharge)), sceneBudget(std::move(sceneCharge)), wire(std::move(bytes)),
          scene(std::move(seed)) {
        outcome.error = errorRecord(OMNI_USD_LIVE_OK);
    }
    void publishStatus() noexcept { // mutex held; registry never calls host code inline.
        OmniUsdLiveStatus value{};
        value.subject_kind = OMNI_USD_LIVE_SUBJECT_LAYER;
        value.state = OMNI_USD_LIVE_STATE_OPENING;
        value.operation = OMNI_USD_LIVE_OPERATION_CREATE;
        // Creation is one finite obligation, not an actor work ID/frontier.
        // Confirmed/rejected/unissued receipts settle it; unknown retains it.
        const bool pending = outcome.phase == Phase::Creating || outcome.phase == Phase::OutcomeUnknown;
        value.pending_write_count = pending ? 1 : 0;
        value.pending_bytes = pending ? wire.size() : 0;
        value.has_unconfirmed_edits = pending;
        value.result = outcome.result;
        if (outcome.phase == Phase::OutcomeUnknown || outcome.phase == Phase::UnissuedFailure ||
            outcome.phase == Phase::Rejected)
            value.state = OMNI_USD_LIVE_STATE_FAILED;
        else if (outcome.closing)
            value.state = outcome.quiescent ? OMNI_USD_LIVE_STATE_CLOSED : OMNI_USD_LIVE_STATE_CLOSING;
        if (value.state == OMNI_USD_LIVE_STATE_FAILED)
            value.recovery_action = OMNI_USD_LIVE_RECOVERY_PRESERVE_AND_REOPEN;
        if (value.result != OMNI_USD_LIVE_OK && value.result == outcome.error.result) {
            value.grpc_status_present = outcome.error.grpc_status_present;
            value.grpc_status = outcome.error.grpc_status;
        }
        status->publishCreation(value, outcome.phase == Phase::Confirmed && outcome.quiescent && !outcome.closing);
    }
    void unissued(OmniUsdLiveResult result) noexcept { // mutex held
        outcome.phase = Phase::UnissuedFailure;
        outcome.result = result;
        outcome.error = errorRecord(result);
    }
    // The last refusal, which applied nothing, settles the creation: its
    // UNAVAILABLE record stays in outcome.error. mutex held.
    void refused() noexcept {
        outcome.phase = Phase::Rejected;
        outcome.result = outcome.error.result;
    }
    // Whether a reissue is barred at time `at`: the creation or the
    // runtime is closing, its opening was abandoned, or the deadline passed.
    // mutex held.
    bool reissueStopped(Clock::time_point at) const noexcept {
        return outcome.closing || abandoned || runtime->closing.load() || at >= openDeadline;
    }
    ScheduledStep advance(Clock::time_point now) noexcept override {
        std::shared_ptr<Request> active;
        bool cancel = false, reissue = false;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (outcome.quiescent) return {true, {}};
            if (outcome.phase == Phase::Prepared && !outcome.closing) return {};
            if (starting) return {false, now + std::chrono::milliseconds(1)};
            if (backingOff) {
                backingOff = false;
                // A stopped reissue does not go out; the refusal stands.
                if (reissueStopped(now)) refused();
                else if (now < reissueAt) {
                    backingOff = true;
                    return {false, reissueAt};
                } else reissue = starting = reissuing = true;
            }
            if (!reissue) {
                active = request;
                cancel = outcome.closing;
                if (!active) {
                    outcome.quiescent = true;
                    publishStatus();
                    changed.notify_all();
                    return {true, {}};
                }
            }
        }
        if (reissue) {
            // A worker prepares the reissue's upload off the timer thread.
            runtime->requests->executor().enqueue(*this);
            return {false, now + std::chrono::milliseconds(1)};
        }
        if (cancel) (void)active->cancel(0);
        if (active->wait(0) != OMNI_USD_LIVE_OK) return {false, now + std::chrono::milliseconds(1)};
        auto error = errorRecord(OMNI_USD_LIVE_INTERNAL_ERROR);
        if (active->error(&error) != OMNI_USD_LIVE_OK) return {false, now + std::chrono::milliseconds(1)};
        // Receipt survives ordinary callback suppression and is published
        // before request completion. Load sequence first as its publish marker.
        const auto sequence = receipt->sequence.load(std::memory_order_acquire);
        const auto object = receipt->object.load(std::memory_order_acquire);
        const auto proof = receipt->outcome.load(std::memory_order_relaxed);
        // A completed attempt with an unknown outcome is re-sent once on an
        // endpoint that has shown operation-ID support, and only an OK that
        // reports an outcome confirms the re-send (design section 13).
        const bool confirmed = object && sequence == 1 && (!resent || acknowledgesResend(proof));
        const bool resend = !confirmed && !resent && error.result == OMNI_USD_LIVE_OUTCOME_UNKNOWN && !cancel &&
            !runtime->closing.load() && resendsUnknownWrites(wireBudget.endpoint());
        const auto released = runtime->requests->release(active->handle);
        if (released != OMNI_USD_LIVE_OK && released != OMNI_USD_LIVE_INVALID_HANDLE)
            return {false, now + std::chrono::milliseconds(1)};
        std::shared_ptr<Request> retired;
        if (resend) {
            // Re-send once with the same bytes and operation ID (design
            // section 13); a worker prepares the upload off the timer thread.
            {
                std::lock_guard<std::mutex> lock(mutex);
                outcome.error = error;
                resent = true;
                starting = true;
                retired = std::move(request);
            }
            active.reset();
            retired.reset();
            runtime->requests->executor().enqueue(*this);
            return {false, now + std::chrono::milliseconds(1)};
        }
        // An attempt the service refused UNAVAILABLE with its ErrorInfo applied
        // nothing. It is issued again with the same bytes and operation ID (a
        // CreateLayer header has no writer ID) after section 13's backoff, the
        // longer of the reconnect draw and the refusal's RetryInfo delay
        // capped at reconnect_maximum_ms, while that backoff ends before the
        // initial-open deadline, until reconnect_max_attempts consecutive
        // refusals settle the creation UNAVAILABLE. A refused re-send instead
        // leaves the first attempt's outcome unknown.
        if (!confirmed && !resent && error.result == OMNI_USD_LIVE_UNAVAILABLE && !cancel && !runtime->closing.load()) {
            const auto hint = receipt->retryDelayMs.load(std::memory_order_acquire);
            const bool hinted = hint != kNoRetryDelay;
            bool backoff = false;
            uint32_t attempt;
            uint64_t delay;
            Clock::time_point at;
            {
                std::lock_guard<std::mutex> lock(mutex);
                attempt = refusals + 1;
                delay = reconnectWait(*runtime, status->id(), attempt,
                    hinted ? std::optional<uint64_t>(hint) : std::nullopt);
                at = now + std::chrono::milliseconds(delay);
                if (attempt < runtime->options.values.reconnect_max_attempts && !reissueStopped(at)) {
                    backoff = backingOff = true;
                    outcome.error = error;
                    refusals = attempt;
                    reissueAt = at;
                    retired = std::move(request);
                }
            }
            if (backoff) {
                // reason is the delay's source: 1 the service's RetryInfo, 2
                // the reconnect schedule.
                runtime->diagnostics->emit(OMNI_USD_LIVE_RECORD_RETRY_SCHEDULED, wireBudget.endpoint().handle,
                    status->id(), 0,
                    {{DiagnosticKey::Operation, OMNI_USD_LIVE_OPERATION_CREATE}, {DiagnosticKey::Result, error.result},
                        {DiagnosticKey::Request, error.request_id}, {DiagnosticKey::Attempt, attempt},
                        {DiagnosticKey::Generation, 0}, {DiagnosticKey::Work, 0},
                        {DiagnosticKey::Reason, hinted ? 1u : 2u}, {DiagnosticKey::DelayMs, delay}});
                active.reset();
                retired.reset(); // Endpoint/capture retirement outside owner locks.
                return {false, at};
            }
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            outcome.error = error;
            // Only a confirmed creation names an object; an unproven re-send's reply does not.
            outcome.object = confirmed ? object : 0;
            outcome.sequence = confirmed ? sequence : 0;
            outcome.result = error.result;
            if (confirmed) {
                outcome.phase = Phase::Confirmed;
                outcome.result = OMNI_USD_LIVE_OK;
            } else if (error.result == OMNI_USD_LIVE_OUTCOME_UNKNOWN || error.result == OMNI_USD_LIVE_OK ||
                (resent &&
                    (error.result == OMNI_USD_LIVE_UNAUTHENTICATED || error.result == OMNI_USD_LIVE_UNAVAILABLE))) {
                // An unacknowledged OK, or a re-send that was not evaluated or
                // that applied nothing, leaves the first attempt's outcome
                // unknown.
                outcome.phase = Phase::OutcomeUnknown;
                outcome.result = OMNI_USD_LIVE_OUTCOME_UNKNOWN;
            } else if (!error.attempt_id && error.result == OMNI_USD_LIVE_CANCELLED) {
                outcome.phase = Phase::UnissuedFailure;
            } else {
                // Only the request's classification establishes rejection;
                // raw gRPC evidence alone never implies retry safety.
                outcome.phase = Phase::Rejected;
            }
            retired = std::move(request);
        }
        active.reset();
        retired.reset(); // Endpoint/capture retirement outside owner locks.
        {
            std::lock_guard<std::mutex> lock(mutex);
            outcome.quiescent = true;
            publishStatus();
            changed.notify_all();
        }
        return {true, {}};
    }
    // Issues the re-send or the reissue that advance scheduled. A re-send that
    // cannot be issued leaves the first attempt's outcome unknown; a reissue
    // that cannot be issued, or that is stopped while it is queued, leaves
    // the refusal, which applied nothing, as the creation's result.
    void dispatch() noexcept override {
        std::shared_ptr<Request> next;
        bool reissue;
        {
            std::lock_guard<std::mutex> lock(mutex);
            reissue = reissuing;
        }
        const auto hook = beforeResendForTest.load(std::memory_order_acquire);
        const bool failed = hook && hook(beforeResendContext.load(std::memory_order_acquire));
        bool stopped;
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopped = reissue ? reissueStopped(Clock::now()) : outcome.closing;
        }
        // A creation closed while its re-send or reissue was queued issues
        // neither, as start() admits no attempt after close.
        if (!stopped && !failed) try {
                next = submitLayerCreate(*runtime, reservation->endpoint(), reservation->session(),
                    reservation->layer(), wire, {}, 0, nullptr, receipt, {status->id(), 0, 0}, operationId);
            } catch (...) {}
        {
            std::lock_guard<std::mutex> lock(mutex);
            // The snapshot keeps the first attempt's request handle; each later attempt's error record names its own.
            request = std::move(next);
            if (!request && reissue) refused();
            else if (!request) {
                outcome.phase = Phase::OutcomeUnknown;
                outcome.result = OMNI_USD_LIVE_OUTCOME_UNKNOWN;
            }
            starting = reissuing = false;
            publishStatus();
        }
        runtime->auth->wake(*this);
    }
};

std::shared_ptr<SceneCreation> SceneCreation::prepare(std::shared_ptr<Runtime> runtime, OmniUsdLiveEndpoint endpoint,
    std::string session, std::string layer, std::string bytes, SceneJournal::Limits limits,
    std::optional<std::chrono::steady_clock::time_point> admissionDeadline) {
    try {
        if (!runtime) throw OmniUsdLiveResult(OMNI_USD_LIVE_NOT_INITIALIZED);
        if (runtime->closing) throw OmniUsdLiveResult(OMNI_USD_LIVE_BUSY);
        if (bytes.size() > runtime->options.values.max_update_bytes || bytes.size() > limits.scene.envelope.maxBytes)
            throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
        auto reservation = runtime->layerAdmissions->reserve(endpoint, session, layer, admissionDeadline);
        const auto cap = static_cast<size_t>(std::min<uint64_t>(runtime->options.values.max_pending_bytes_per_endpoint,
            std::numeric_limits<size_t>::max()));
        const auto configured = std::min(cap, limits.maxPendingBytes);
        size_t available = configured;
        const auto original = available;
        charge(bytes.size(), available);
        charge(sizeof(State) + sizeof(SceneCreation) + 512, available);
        charge(reservation->canonicalUri().size(), available);
        charge(reservation->session().size(), available);
        charge(reservation->layer().size(), available);
        auto wireLease = runtime->acquireEndpoint(endpoint);
        if (!wireLease) throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_HANDLE);
        PendingBytes wireCharge(std::move(wireLease), original - available, cap);
        // Decoding, building and measuring the created scene cost work in
        // proportion to the upload, beside the configured replay work.
        constexpr size_t kWorkPerByte = 512;
        WireSceneWorkBudget work{
            bytes.size() > (std::numeric_limits<size_t>::max() - limits.maxReplayWork) / kWorkPerByte ?
                std::numeric_limits<size_t>::max() :
                limits.maxReplayWork + bytes.size() * kWorkPerByte};
        charge(bytes.size(), work.remaining);
        const auto envelope = wire::decodeEnvelope(bytes, limits.scene.envelope);
        if (envelope.isDiff && envelope.baseVersion) throw OmniUsdLiveResult(OMNI_USD_LIVE_UNSUPPORTED);
        // A seed over the service's entry limits is RESOURCE_EXHAUSTED before any RPC.
        checkDeltaEntries(envelope.commands, envelope.fields.size(), envelope.samples.size(), limits.authoring.entries);
        auto scene = buildSceneCandidate({}, bytes, limits.scene, &work);
        // As in the journal, the scene's charge has room for 32 charges of the
        // scene beside the configured pending bytes; the endpoint stays a cap.
        {
            const auto room = scene->retainedCharge() > std::numeric_limits<size_t>::max() / 32 ?
                std::numeric_limits<size_t>::max() :
                scene->retainedCharge() * 32;
            const auto grown = room > std::numeric_limits<size_t>::max() - limits.maxPendingBytes ?
                std::numeric_limits<size_t>::max() :
                limits.maxPendingBytes + room;
            available += std::min(cap, grown) - configured;
        }
        const auto retained = measureRetainedScene(*scene, limits.scene, available, work);
        auto sceneLease = runtime->acquireEndpoint(endpoint);
        if (!sceneLease) throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_HANDLE);
        PendingBytes sceneCharge(std::move(sceneLease), retained, cap);
        auto status = LayerStatus::create(runtime, reservation, true);
        auto state = std::make_shared<State>(runtime, std::move(reservation), std::move(status), std::move(wireCharge),
            std::move(sceneCharge), std::move(bytes), std::move(scene));
        // Outer owner allocation precedes registry publication, so an exception
        // cannot leave a prepared operation with no caller capable of closing it.
        auto owner = std::shared_ptr<SceneCreation>(new SceneCreation(state));
        runtime->auth->reserve(state);
        try {
            runtime->auth->activate(*state);
        } catch (...) {
            runtime->auth->discard(*state);
            throw;
        }
        return owner;
    } catch (const wire::DecodeError& error) { throw decodeResult(error); } catch (const std::bad_alloc&) {
        throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    } catch (const std::length_error&) {
        throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    } catch (const std::invalid_argument&) { throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT); }
}

SceneCreation::~SceneCreation() { (void)close(0); }

OmniUsdLiveResult SceneCreation::failBeforeStart(OmniUsdLiveResult result) noexcept {
    if (result == OMNI_USD_LIVE_OK || !definedResult(result) || result == OMNI_USD_LIVE_OUTCOME_UNKNOWN)
        return OMNI_USD_LIVE_INVALID_ARGUMENT;
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        if (state_->outcome.started) return OMNI_USD_LIVE_ALREADY_EXISTS;
        if (state_->outcome.closing) return OMNI_USD_LIVE_CANCELLED;
        state_->outcome.started = true;
        state_->unissued(result);
        state_->publishStatus();
    }
    state_->runtime->auth->wake(*state_);
    return OMNI_USD_LIVE_OK;
}

OmniUsdLiveResult SceneCreation::start() noexcept {
    OmniUsdLiveResult result = OMNI_USD_LIVE_OK;
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        if (state_->outcome.started) return OMNI_USD_LIVE_ALREADY_EXISTS;
        if (state_->outcome.closing) return OMNI_USD_LIVE_CANCELLED;
        state_->outcome.started = true;
        state_->outcome.phase = Phase::Creating;
        state_->openDeadline = state_->callerDeadline.value_or(
            Clock::now() + std::chrono::milliseconds(state_->runtime->options.values.initial_open_timeout_ms));
        state_->starting = true;
        state_->publishStatus();
    }
    // Preparing the transport's owned upload may allocate/copy. Keep it off
    // the timer thread and outside the mutex read by advance/close/snapshot.
    std::shared_ptr<Request> request;
    try {
        request = submitLayerCreate(*state_->runtime, state_->reservation->endpoint(), state_->reservation->session(),
            state_->reservation->layer(), state_->wire, {}, 0, nullptr, state_->receipt, {state_->status->id(), 0, 0},
            state_->operationId);
    } catch (OmniUsdLiveResult failure) { result = failure; } catch (const std::bad_alloc&) {
        result = OMNI_USD_LIVE_RESOURCE_EXHAUSTED;
    } catch (const std::length_error&) {
        result = OMNI_USD_LIVE_RESOURCE_EXHAUSTED;
    } catch (const std::invalid_argument&) { result = OMNI_USD_LIVE_INVALID_ARGUMENT; } catch (...) {
        result = OMNI_USD_LIVE_INTERNAL_ERROR;
    }
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        state_->request = std::move(request);
        if (state_->request) state_->outcome.request = state_->request->handle;
        if (result != OMNI_USD_LIVE_OK) state_->unissued(result);
        state_->starting = false;
        state_->publishStatus();
    }
    state_->runtime->auth->wake(*state_);
    return result;
}

void SceneCreation::boundReissues(Clock::time_point deadline) noexcept {
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        state_->callerDeadline = deadline;
        if (state_->outcome.started) state_->openDeadline = deadline;
    }
    state_->runtime->auth->wake(*state_);
}

void SceneCreation::abandonReissues() noexcept {
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        state_->abandoned = true;
    }
    state_->runtime->auth->wake(*state_);
}

OmniUsdLiveResult SceneCreation::wait(uint32_t timeoutMs) noexcept {
    const auto deadline = Clock::now() + std::chrono::milliseconds(timeoutMs);
    std::unique_lock<std::mutex> lock(state_->mutex);
    if (state_->outcome.quiescent) return OMNI_USD_LIVE_OK;
    if (state_->outcome.phase == Phase::Prepared || DispatchExecutor::isWorker()) return OMNI_USD_LIVE_BUSY;
    return state_->changed.wait_until(lock, deadline, [&] { return state_->outcome.quiescent; }) ?
        OMNI_USD_LIVE_OK :
        OMNI_USD_LIVE_TIMEOUT;
}

OmniUsdLiveResult SceneCreation::close(uint32_t timeoutMs) noexcept {
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        state_->outcome.closing = true;
        if (state_->outcome.phase == Phase::Prepared) state_->unissued(OMNI_USD_LIVE_CANCELLED);
        state_->publishStatus();
    }
    state_->runtime->auth->wake(*state_);
    return wait(timeoutMs);
}

SceneCreation::Snapshot SceneCreation::snapshot() const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->outcome;
}
const WireScenePtr& SceneCreation::seed() const noexcept { return state_->scene; }
const std::string& SceneCreation::bytes() const noexcept { return state_->wire; }
const std::shared_ptr<LayerAdmission>& SceneCreation::admission() const noexcept { return state_->reservation; }
const std::shared_ptr<LayerStatus>& SceneCreation::status() const noexcept { return state_->status; }
} // namespace omni::usd::live
