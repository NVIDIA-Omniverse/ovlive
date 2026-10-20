// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "LiveAuth.h"
#include "LiveLayerProgress.h"
#include "LiveLayerRead.h"
#include "LiveLayerWrite.h"
#include "LiveJitter.h"
#include "LiveLayerAdmission.h"
#include "LiveLayerStatus.h"
#include "LiveDiagnostics.h"
#include <algorithm>
#include <optional>
#include <array>

namespace omni::usd::live {
// Private native attachment for a known or initially discovered incarnation.
// Canonical admission precedes transport. Public USD binding, creation and
// OMPF interpretation stay with scene/file-format integration. No host pump.
class LayerActor final : public ScheduledOperation, public DispatchJob {
public:
    struct Options {
        std::string session, layer;
        uint64_t object = 0, resume = 0;
        std::shared_ptr<const void> initialState;
        // Optional immutable owner of a sticky OK -> typed host failure latch.
        // No USD objects/callbacks. Host mutation recording stores here, and
        // observeHostFailure (thread-safe) wakes status/cleanup: a safe host
        // boundary calls it, as does a rejected record from its own thread.
        std::shared_ptr<const std::atomic<OmniUsdLiveResult>> hostFailure;
        // Optional count of host edits that no flush has admitted, stored by
        // the host (see BufferedEditCount for off-host
        // writes). Status reports it and counts it as
        // unconfirmed edits; noteHostEdits republishes it.
        std::shared_ptr<BufferedEditCount> bufferedEdits;
        void (*beforeAdmissionForTest)(void*) noexcept = nullptr;
        // Private seam between a submission's byte reservation and its ledger
        // admission, under the admission lock; a throw refuses the admission.
        void (*afterReservationForTest)(void*) = nullptr;
        void* reservationContextForTest = nullptr;
        void* admissionContextForTest = nullptr;
        // Private fault-test seam; production attachments retain 1000 ms.
        uint32_t timingTimeoutMsForTest = 1000;
        // Private, pure/nonreentrant diagnostic clock; context outlives actor
        // and host completion owners. Does not change operational deadlines.
        std::chrono::steady_clock::time_point (*applicationNowForTest)(void*) noexcept = nullptr;
        void* applicationClockContextForTest = nullptr;
        // Private concurrency barrier before credential-gate locks, never host ABI.
        void (*beforeAuthGateForTest)(void*) noexcept = nullptr;
        void* authGateContextForTest = nullptr;
        // Private seam on the worker, without actor locks: a closing pass that
        // keeps the actor running because a read, write, ping, queued frame or
        // application owner remains. The context outlives the actor.
        void (*closingRetainedForTest)(void*) noexcept = nullptr;
        void* closingRetainedContextForTest = nullptr;
        // Discovery callers reserve canonical ownership before any RPC. A
        // supplied reservation is validated and claimed once by open().
        std::shared_ptr<LayerAdmission> admission;
        bool discover = false;
        std::shared_ptr<LayerStatus> status;
        bool holdOpening = false; // Host identity publication is a separate prerequisite.
    };
    struct Decoded {
        uint64_t first = 0;
        LayerProgress::CoverageKind kind = LayerProgress::CoverageKind::Update;
        bool recoveryBoundary = false;
    };
    struct Effect {
        // A local intent's admission reservation, which also backs the wire
        // buffers of its issued update; each issued request shares it.
        std::shared_ptr<PendingBytes> reservation;
        std::shared_ptr<const LayerReadUpdate> incoming;
        std::shared_ptr<const void> intent;
        std::string authored;
        std::string operationId; // A local intent's write identity; its re-send repeats it.
        // The operation ID of the writer's previous write to the object, fixed
        // at admission, and whether it is known: empty and known for the
        // writer's first write to the object. Every attempt of a chained write
        // sends it; an unknown predecessor issues the write unchained.
        std::string previousOperationId;
        bool chainable = false;
        std::chrono::steady_clock::time_point received{}, eligible{};
        // The frame's receipt on the diagnostic application clock, the start
        // of its receive-to-apply sample.
        std::chrono::steady_clock::time_point measuredReceipt{};
        uint64_t generation = 0;
        std::shared_ptr<DrainTicket> drain;
        std::string_view bytes() const {
            return incoming ? std::string_view(incoming->bytes) : std::string_view(authored);
        }
    private:
        friend class LayerActor;
        mutable bool localPublished_ = false; // Actor worker only; never host callbacks.
    };
    // One asynchronous ordered application. Reporting success is insufficient:
    // the actor also waits for this owner's final destruction. A queued Task
    // must declare its completion owner BEFORE every other owned capture, so
    // those captures retire first. Do not retain the Task from this owner.
    class ApplyCompletion final : public std::enable_shared_from_this<ApplyCompletion> {
    public:
        ~ApplyCompletion();
        ApplyCompletion(const ApplyCompletion&) = delete;
        ApplyCompletion& operator=(const ApplyCompletion&) = delete;
        // Claims once, skips work canceled before entry, marks host callbacks
        // nonblocking, and translates exceptions into the completion result.
        void run(std::function<void()> action) noexcept;
        void complete(OmniUsdLiveResult) noexcept;
    private:
        friend class LayerActor;
        struct State;
        explicit ApplyCompletion(std::shared_ptr<State>);
        const std::shared_ptr<State> state_;
    };
    struct Callbacks {
        // Complete payload validation and semantic range proof, not a guess
        // from header sequence gaps. Runs on a worker outside runtime locks.
        std::function<Decoded(const LayerReadUpdate&)> decode;
        // Applies one ordered effect (including a confirmed local effect).
        // Return only after its effects are present in retained layer state.
        std::function<void(const LayerProgress::Work&, const Effect&)> apply;
        std::function<void(const LayerProgress::Snapshot&)> changed;
        // Exactly one of apply/applyAsync is required. The asynchronous owner
        // receives owned inputs and may enqueue without blocking a worker.
        // At most one application is outstanding. Throwing after enqueue
        // rejects/cancels that application but still waits for owner retirement.
        std::function<void(LayerProgress::Work, std::shared_ptr<const Effect>, std::shared_ptr<ApplyCompletion>)>
            applyAsync;
        // Optional batched form of applyAsync, which it replaces: one
        // application receives a run of consecutive ordered peer work whose
        // holdback has elapsed, in order, at most kMaxBatch works and, beyond
        // the first, max_update_bytes of payload; other work arrives alone.
        // Success means every work's effect is in retained layer state.
        std::function<void(std::vector<LayerProgress::Work>, std::vector<std::shared_ptr<const Effect>>,
            std::shared_ptr<ApplyCompletion>)>
            applyBatchAsync;
        // Optional optimistic local publication after retained admission and
        // before write issuance. Shares the one outstanding application slot.
        // Success plus capture retirement enables issuance, not durability or
        // ordered application, and never completes the original drain ticket.
        std::function<void(LayerProgress::Work, std::shared_ptr<const Effect>, std::shared_ptr<ApplyCompletion>)>
            localAsync;
        // Optional synchronous journal-only application after an issuance-only
        // FAILED: the read stays open and ordered work keeps advancing the
        // journal while nothing installs into the host layer. It must accept
        // work an earlier application already added to the journal. Without
        // it, every failure closes the read.
        std::function<void(const LayerProgress::Work&, const Effect&)> applyDetached;
    };
    // The external owner's last release of the returned handle closes the
    // actor with close(0), so the endpoint lease, canonical key and read
    // subscription retire before the endpoint unregisters; the scheduler
    // keeps the actor until it stops. SceneAttachment's destructor closes
    // explicitly; the release then repeats an idempotent close.
    static std::shared_ptr<LayerActor> open(std::shared_ptr<Runtime> runtime, OmniUsdLiveEndpoint endpoint,
        Options options, Callbacks callbacks);
    ~LayerActor() override;
    // The most ordered works one batched application covers.
    static constexpr size_t kMaxBatch = 64;
    // The issuance window on an endpoint that chains writes: it starts at
    // kInitialWindow, grows by one after each window of acknowledged writes
    // without a refusal, halves on a capacity refusal and stays within
    // [1, kMaxWindow]. An endpoint without operation-ID support issues one
    // write at a time (design section 13).
    static constexpr uint32_t kInitialWindow = 8, kMaxWindow = 16;
    // The margin a re-sent unchained write's issuance hold adds to its first
    // attempt's request_timeout_ms: the service measures that attempt's
    // deadline from when it received the call, so the margin covers the
    // call's transit, and it equals the slack the default 35 s request
    // timeout keeps over the service's 30 s limit.
    static constexpr std::chrono::milliseconds kResendHoldMargin{5000};
    // Whether an admitted write waits to be issued or the layer has as many
    // admitted unsettled writes, in flight and queued, as its issuance window
    // allows: a flush then defers and its records coalesce with later host
    // turns. Any thread.
    bool issuanceSaturated() const noexcept;
    // Thread-safe. Records the host turns one admitted update holds.
    void noteCoalescedTurns(uint64_t turns) noexcept;
    // A charge of this layer's host edit buffer to its endpoint's shared host
    // buffer budget (design section 8, Quotas); null once the runtime or the
    // endpoint closes. Any thread; the charge itself is host-confined.
    std::unique_ptr<HostBufferCharge> hostBufferCharge() const;
    // Any thread. The endpoint's shared host buffer budget, read without a
    // charge, for a layer whose buffer holds none.
    HostBufferCharge::Budget hostBufferBudget() const noexcept;
    struct TransportSnapshot {
        size_t queuedReadFrames;
        bool readEnded, workerScheduled, timingPending, readPaused;
        bool wakePending; // A per-actor wake that no worker pass has consumed.
        // First-response deadlines that passed while the layer held the read
        // for incoming pacing, each restarting the attempt's window.
        uint64_t heldReadWindows;
    };
    TransportSnapshot transportSnapshot() const;
    // Test seam: the initial-open deadline counts as passed from the next step.
    void expireOpenForTest() noexcept;
    // Test seam: the current read attempt's first-response deadline counts as
    // passed at the next step, if that attempt still awaits its first complete
    // response; while the layer holds that read for incoming pacing the request
    // waits for the hold to end. Otherwise that step discards the request.
    void expireReadForTest() noexcept;
    // Test seam: the next step with an open read on a READY layer and no write
    // in flight resubscribes it at once, as a settled uncertain write does.
    void resubscribeForTest() noexcept;
    // Test seam: the chained-write issuance window, which a broken chain's
    // first re-send does not narrow. Any thread.
    uint32_t chainWindowForTest() const noexcept { return windowForTest_.load(std::memory_order_acquire); }
    // Test seam: whether, as of the actor's last pass, issuance holds for a
    // re-sent unchained write's first attempt (resendHold_) with no write
    // attempt in flight. Any thread.
    bool resendHeldForTest() const noexcept { return resendHeldForTest_.load(std::memory_order_acquire); }
    // Test seam: the current re-send hold counts as passed from the next step.
    void expireResendHoldForTest() noexcept;
    // Test seam: every actor worker calls hook(context) after it completes an
    // applied effect's drain tickets, before the rest of its pass; null disables
    // it. The hook must not allocate, throw or call into the actor.
    static void setAfterDrainRetiredForTest(void (*hook)(void*) noexcept, void* context) noexcept;
    // The test gate runs on a closing worker before it announces quiescence.
    // The hook must not throw or call into the actor and must tolerate null.
    static void setBeforeCloseQuiescentForTest(void (*hook)(void*) noexcept, void* context) noexcept;
    // Test seam: an actor worker that finds a completed write attempt calls
    // hook(context, completed) with the number of its attempts that have
    // completed before it settles any, and again while the hook returns true,
    // so that one pass can settle several completions in chain order. The
    // hook must not throw or call into the actor and must tolerate null.
    static void setBeforeSettleWritesForTest(bool (*hook)(void*, size_t) noexcept, void* context) noexcept;
    // Publication follows LayerProgress's nonthrowing commit contract. The
    // caller may hold its journal guard: lock order is journal -> admission ->
    // progress. All fallible preparation precedes commit; wake cannot throw.
    uint64_t submit(std::string_view bytes, std::shared_ptr<const void> completeIntent,
        LayerProgress::Publication publication = {nullptr, nullptr}, uint64_t retainedBytes = 0);
    // Whether an update of `bytes` fits the runtime's per-update and
    // per-endpoint pending limits beside the layer's own standing read
    // reservation and no other pending work. submit rejects
    // a nonfitting update with RESOURCE_EXHAUSTED however long it waits.
    bool updateCanFit(size_t bytes) const noexcept;
    // The pending bytes submit reserves on the endpoint for an update of
    // `bytes`: its retained intent and the wire buffers of its issuance, so
    // issuing an admitted intent never needs another reservation.
    uint64_t admissionCharge(size_t bytes) const noexcept;
    LayerProgress::Snapshot snapshot() const { return progress_.snapshot(); }
    std::vector<LayerProgress::Work> retained() const { return progress_.retained(); }
    // Captures this layer's admitted obligations only when local edits remain.
    // Clean saves ignore peer-only work. Later admissions cannot extend the
    // captured boundary; no mutation, cancellation or new RPC is performed.
    OmniUsdLiveResult save(uint32_t timeoutMs, bool allowBlock = true) noexcept;
    // Read-only dirty-state query; does not emit a save-operation diagnostic.
    OmniUsdLiveResult pollSave() noexcept { return saveImpl(0, false, nullptr); }
    // Fixed before open returns; binds only this actor's claimed reservation.
    uint64_t subjectId() const noexcept { return subject_; }
    // Whether the last auth evaluation found no usable credential. A caller
    // whose own wait on this actor expires classifies it with expiredWait.
    bool credentialStarved() const noexcept { return waitingAuth_.load(); }
    // Release the host-identifier publication gate. Initial application may
    // still be retiring; fresh status publication independently requires that
    // coverage before READY. Never waits or forces a second identifier update.
    OmniUsdLiveResult completeOpening() noexcept;
    OmniUsdLiveResult authoringResult() noexcept;
    OmniUsdLiveResult observeHostFailure() noexcept;
    // Thread-safe, never waits for the worker. Publishes the current buffered
    // edit count and deferral cause: through a worker wake, or directly on
    // the calling thread once the actor is terminal and its worker stopped.
    void noteHostEdits() noexcept;
    // Thread-safe. Emit flush_unsupported (numeric reason only) with this
    // subject and its status incident; the caller has latched the failure.
    void reportFlushUnsupported(uint32_t reason) noexcept;
    // Terminal failure, not close: preserve owners while ordinary worker
    // cleanup stops transport and queued application.
    OmniUsdLiveResult failPublication(OmniUsdLiveResult) noexcept;
    // Stops transport/callbacks but preserves progress and complete intents
    // until the attachment owner releases us.
    OmniUsdLiveResult close(uint32_t timeoutMs);
    // The owner releases this actor with its close: the canonical key goes
    // once the actor retires, so an open may wait for it (LayerAdmission).
    void markAdmissionRetiring() noexcept;
    ScheduledStep advance(std::chrono::steady_clock::time_point now) noexcept override;
private:
    using Clock = std::chrono::steady_clock;
    struct Event {
        uint64_t generation;
        std::shared_ptr<const LayerReadUpdate> update;
        Clock::time_point received, measuredReceipt;
    };
    struct PingOutcome {
        uint64_t serverUs = 0;
        // Endpoint::capabilities of an answered Ping, otherwise zero.
        uint8_t capabilities = 0;
        Clock::time_point received{};
        std::atomic<bool> completed{false};
    };
    struct WriteOutcome {
        std::shared_ptr<LayerWriteReceipt> receipt = std::make_shared<LayerWriteReceipt>();
        std::atomic<bool> completed{false};
    };
    // One attempt of a local write in flight. The vector and each request
    // pointer change under mutex_, on the worker only; the worker reads the
    // other members without it.
    struct InFlight {
        uint64_t id = 0, ordinal = 0, authFailure = 0;
        std::shared_ptr<Request> request;
        std::shared_ptr<WriteOutcome> outcome;
        std::shared_ptr<const Effect> effect;
        bool chained = false; // The attempt carries the write's predecessor.
        bool repeated = false; // An attempt after the write's first.
        bool cancelled = false; // Cancelled because an earlier write of the chain did not commit.
    };
    LayerActor(std::shared_ptr<Runtime> runtime, EndpointLease endpoint, Options options, Callbacks callbacks);
    std::shared_ptr<Runtime> runtime_;
    // Declared before all backing/transport members so their final retirement
    // precedes release of canonical ownership, including failed/closed actors.
    std::shared_ptr<LayerAdmission> admission_;
    std::shared_ptr<LayerStatus> statusOwner_;
    EndpointLease endpoint_;
    LayerProgress progress_;
    // Encoded UpdateLayer header bound for this identity and writer.
    const size_t updateHeaderBound_;
    // Incoming pacing (design section 12): the read parks at the high-water
    // mark of queued frames plus unapplied peer coverage and resumes at the
    // low-water mark.
    const std::shared_ptr<LayerReadFlow> readFlow_ = std::make_shared<LayerReadFlow>();
    const size_t readHighWater_, readLowWater_;
    bool readPaused_ = false; // Under mutex_.
    Callbacks callbacks_;
    const std::shared_ptr<const std::atomic<OmniUsdLiveResult>> hostFailure_;
    const std::shared_ptr<BufferedEditCount> bufferedEdits_;
    bool watchingPendingBytes_ = false; // This actor's entry in the endpoint's watchers, under its mutex.
    void (*const beforeAdmission_)(void*) noexcept;
    void* const admissionContext_;
    void (*const afterReservation_)(void*);
    void* const reservationContext_;
    OmniUsdLiveResult hostFailureResult() const noexcept;
    // First close only: emit host_edits_discarded (count and the FAILED cause
    // or OK) when the shared host edit buffer still holds records.
    void reportDiscardedHostEdits() noexcept;
    std::shared_ptr<DrainState> drain_;
    std::shared_ptr<DrainTicket> activity_, readEndDrain_;
    // Worker-owned: the applied application's work IDs awaiting progress
    // bookkeeping and its original drain tickets awaiting publication.
    // Reserved for kMaxBatch at construction, so settling never allocates.
    std::vector<uint64_t> pendingApplied_;
    std::vector<std::shared_ptr<DrainTicket>> pendingDrains_;
    // Worker-owned: owners of absorbed local intents that the last applied
    // coverage retired; their tickets complete with pendingDrains_.
    std::vector<std::shared_ptr<const void>> absorbedOwners_;
    // An adoption failure keeps incoming owners until the worker latches the failure.
    std::vector<std::shared_ptr<const void>> failedAbsorbedOwners_;
    std::shared_ptr<ApplyCompletion::State> application_; // Pointer guarded by mutex_.
    // Fixed worker batch bounds: no allocation after application publication.
    std::array<std::shared_ptr<DrainTicket>, 33> receivedDrains_{};
    size_t receivedDrainCount_ = 0;
    OmniUsdLiveResult activityAdmission_ = OMNI_USD_LIVE_OK;
    mutable std::mutex mutex_, admissionMutex_, statusMutex_;
    uint64_t subject_ = 0;
    OmniUsdLiveRequestError statusError_{}; // Guarded by statusMutex_.
    std::atomic<bool> statusClosed_{false};
    std::atomic<bool> holdOpening_{false};
    std::condition_variable closed_;
    std::deque<Event> events_;
    std::shared_ptr<Request> read_;
    std::vector<InFlight> writes_;
    std::shared_ptr<Request> ping_;
    std::shared_ptr<PingOutcome> pingOutcome_;
    Clock::time_point pingSent_{}, nextPing_{};
    uint64_t pingGeneration_ = 0;
    // Whether the in-flight Ping samples jitter, which only an endpoint with
    // smoothing enabled does. Writes never wait for a Ping: until one is
    // answered the endpoint has no capabilities.
    bool pingSamples_ = false;
    JitterEstimator jitter_;
    JitterWait jitterWait_; // Actor worker only; survives read-generation changes.
    void finishJitterWait(JitterWaitResolution resolution, Clock::time_point observed) noexcept;
    const uint32_t timingTimeoutMs_;
    Clock::time_point (*applicationNow_)(void*) noexcept;
    void* applicationClockContext_;
    // An unchained write in flight alone: its effect, whose bytes and
    // operation ID its single re-send repeats, and whether that re-send has
    // been offered (design section 13).
    std::shared_ptr<const Effect> writeEffect_;
    bool writeResent_ = false;
    uint64_t writeId_ = 0, writeOrdinal_ = 0;
    // The re-send waits for a usable token; no request is in flight meanwhile.
    bool resendPending_ = false;
    OmniUsdLiveRequestError resendEvidence_{}; // The unknown attempt the pending re-send follows.
    // When the unchained write's first attempt was submitted, at or after the
    // start of its request_timeout_ms deadline.
    Clock::time_point writeIssuedAt_{};
    // No later write issues before this instant once an unchained write is
    // re-sent: the service may hold its first attempt until that attempt's
    // deadline, and one that reads its start record only after the next write
    // commits can still apply (design section 13, The single operation-ID
    // re-send).
    Clock::time_point resendHold_{};
    // Whether issuance has taken over the endpoint's hold for this object,
    // which an earlier layer may have set (Endpoint::resendHolds).
    bool resendHoldInherited_ = false;
    std::atomic<bool> resendHeldForTest_{false}, resendHoldExpiredForTest_{false};
    uint64_t readGeneration_ = 0;
    // Admission-owned (admissionMutex_): the previous admitted write's
    // operation ID and whether the next admission knows its predecessor.
    std::string lastAdmittedOperation_;
    bool chainStarted_ = false;
    // Worker-owned chain state. A break cancels the chained attempts of later
    // writes and holds issuance until every other write in flight settles,
    // then resolves how far the chain committed from the acknowledgments and
    // the operations refusals named, and re-sends from the first write not
    // known to have committed, alone until it is acknowledged
    // (`serializing_`) and with no_wait (`noWait_`).
    uint32_t window_ = kInitialWindow, windowStreak_ = 0;
    std::atomic<uint32_t> effectiveWindow_{1}, windowForTest_{kInitialWindow};
    bool chainBreak_ = false, serializing_ = false, noWait_ = false;
    // The break needs the chain located: a refusal named an operation or an
    // attempt's outcome is unknown.
    bool chainUnresolved_ = false;
    std::vector<std::pair<LastRecord, std::string>> namedOperations_;
    uint64_t breakRounds_ = 0, deadOrdinal_ = 0, provenPrefix_ = 0;
    // Whether the last resolution re-sends the write at provenPrefix_, which
    // a refusal named, to learn its position.
    bool provenNamed_ = false;
    // Work IDs of writes a chained attempt of which ended unsettled: neither
    // acknowledged nor refused with the service's ErrorInfo or as a replica
    // without chaining. The service may still commit such an attempt, after
    // its deadline too, so on an endpoint that stopped chaining these writes
    // and every earlier one are finished through the chain before any write
    // is sent unchained. debtRefusals_ counts their consecutive re-sends a
    // replica without chaining refused.
    std::vector<uint64_t> chainDebt_;
    uint32_t debtRefusals_ = 0;
    // Work IDs of writes possibly committed only because a cancelled attempt
    // may have been committing: such a write committed only if the write that
    // broke the chain did, so it is never the probe resolveBreak re-sends.
    std::vector<uint64_t> cancelledUncertain_;
    // The latest write of unknown outcome that resolveBreak re-sends first,
    // and its ordinal while that re-send is in flight.
    uint64_t probeId_ = 0, probeOrdinal_ = 0;
    // Whether a read cancelled to deliver chain-proven commits reconnects at
    // once (resubscribeAfterSettle).
    bool settleReconnect_ = false;
    Clock::time_point resumeAt_{};
    // The instant issuance of new writes resumes: after a refusal's backoff
    // and any re-send hold. Publishes resendHeldForTest_.
    Clock::time_point issuanceResumes(Clock::time_point now) noexcept;
    // The session, layer name and object ID joined by NUL bytes: the key of
    // the endpoint's writtenObjects and resendHolds.
    std::string objectKey() const;
    // The chain's operations this actor has seen committed, newest last,
    // bounded; a refusal naming one of them locates the chain.
    std::deque<std::pair<uint64_t, std::string>> committedOperations_;
    bool chainBaseEmpty_ = false; // The actor's first write is the writer's first to the object.
    // Worker-owned: the endpoint's writtenObjects records this actor's object.
    bool objectWritten_ = false;
    uint64_t readRequestId_ = 0, readResume_ = 0; // Original attempt, not a changing frontier.
    bool readReadyReported_ = false; // Worker-owned; one accepted boundary per attempt.
    // Worker-owned: when the current read attempt ends TIMEOUT unless its first
    // complete response arrives first; max while no attempt awaits one. A
    // read this actor cancelled at that deadline is readTimedOut_.
    Clock::time_point readDeadline_ = Clock::time_point::max();
    bool readTimedOut_ = false;
    std::atomic<bool> readExpiredForTest_{false}, resubscribeForTest_{false};
    std::atomic<uint64_t> heldReadWindows_{0};
    uint64_t observedWake_ = 0;
    uint64_t attempts_ = 0, authFailure_ = 0, readAuthFailure_ = 0, attachmentAuthFailure_ = 0;
    std::atomic<bool> readCompletionPending_{false};
    // Guarded by mutex_: the read attempt whose own stream failure latched
    // drain_'s first failure. Set at most once and never cleared: drain_ never
    // unlatches, so no later read attempt can win and overwrite it, and
    // attribution also requires the current read generation.
    uint64_t readFailedGeneration_ = 0;
    OmniUsdLiveResult readFailure_ = OMNI_USD_LIVE_OK;
    std::atomic<bool> waitingAuth_{false};
    // statusError_ holds the transport rejection that started the current
    // auth wait; a locally detected wait carries no gRPC status.
    std::atomic<bool> authEvidence_{false};
    bool scheduled_ = false, dirty_ = true, stopped_ = false;
    // Guarded by mutex_. A terminal actor freezes once its issued work and
    // application callbacks retire; explicit close still retires it.
    bool terminalFrozen_ = false;
    std::atomic<bool> closing_{false};
    // Local pendingWrites retires before its Changed callback. The clean save
    // shortcut stays closed until that callback has run; retireAppliedDrain
    // opens it before it completes the original and absorbed tickets.
    std::atomic<bool> localRetirementPending_{false};
    Clock::time_point retryAt_{}, authDeadline_{}, openDeadline_;
    std::atomic<bool> openExpiredForTest_{false};
    OperationAuthWait authObservation_;
    void (*beforeAuthGate_)(void*) noexcept;
    void* authGateContext_;
    void (*closingRetained_)(void*) noexcept;
    void* closingRetainedContext_;
    bool havePublished_ = false;
    LayerProgress::Snapshot published_{};
    // Under statusMutex_: whether the layer is inside an apply-lag warning
    // interval (apply_lag_entered without its apply_lag_exited), and since when.
    bool applyLagging_ = false;
    Clock::time_point applyLagSince_{};
    // Under statusMutex_: the sequence of the current read attempt's catch-up
    // target, its first accepted boundary (the one observeReadReady reports),
    // below which pending application is catch-up. Later header-only frames
    // of the attempt are live empty commits and never move it.
    uint64_t catchUpSequence_ = 0;
    // Under statusMutex_: enters or leaves the apply-lag warning interval and
    // emits its record while the layer is READY. Another live state holds the
    // interval; an inactive (failed or closing) layer ends it silently.
    void evaluateApplyLag(const LayerProgress::Snapshot& state, bool active, bool ready) noexcept;
    void dispatch() noexcept override;
    void wake() noexcept;
    void receive(uint64_t generation, std::shared_ptr<const LayerReadUpdate> update);
    // Worker only: resumes a paused read once its depth reaches the low-water mark.
    void relieveRead() noexcept;
    Clock::time_point step();
    void settleAppliedEffects();
    void checkReadAuthorization(Clock::time_point now);
    void consumeQueuedFrames();
    void settleWrites(Clock::time_point now, Clock::time_point& next);
    void settleChained(InFlight write, OmniUsdLiveResult result, const OmniUsdLiveRequestError& error,
        uint64_t sequence, Clock::time_point now);
    void breakChain(uint64_t ordinal, bool cancelSuccessors);
    // Sets the issuance window and reports the change.
    void changeWindow(uint32_t window) noexcept;
    void resolveBreak(Clock::time_point now);
    void resubscribeAfterSettle();
    bool lookupOperation(LastRecord record, const std::string& operation, const std::vector<LayerProgress::Work>& work,
        uint64_t& ordinal) const;
    void settleCancelled(InFlight write, OmniUsdLiveResult result, const OmniUsdLiveRequestError& error,
        uint64_t sequence, Clock::time_point now);
    size_t cancelledInFlight() const noexcept;
    uint64_t chainDebtOrdinal();
    bool chainedInFlightThrough(uint64_t ordinal) const noexcept;
    uint32_t chainWindow() const noexcept;
    void publishWindow() noexcept;
    void settleRead(std::shared_ptr<Request>& read, Clock::time_point now);
    Clock::time_point stepClosing(bool read, bool write, bool queuedFrames, Clock::time_point now,
        Clock::time_point next);
    bool startRead(const std::weak_ptr<LayerActor>& weak, Clock::time_point now, std::shared_ptr<Request>& read);
    bool issueWrites(const std::weak_ptr<LayerActor>& weak, Clock::time_point now, Clock::time_point& next);
    InFlight submitWrite(const std::weak_ptr<LayerActor>& weak, std::shared_ptr<const Effect> effect, uint64_t id,
        uint64_t ordinal, bool chained, bool noWait = false);
    void track(InFlight write);
    enum class Resend { NotOffered, Issued, AwaitingAuth };
    Resend resendWrite(Clock::time_point now, const OmniUsdLiveRequestError& error);
    Resend issueResend(Clock::time_point now);
    void settleAttempt(uint64_t id, uint64_t authFailure, OmniUsdLiveResult result,
        const OmniUsdLiveRequestError& error, uint64_t sequence, Clock::time_point now);
    void noteCommitted(uint64_t ordinal, const std::string& operation);
    bool backOffRefusal(const InFlight& write, const OmniUsdLiveRequestError& error, uint64_t retryDelayMs,
        Clock::time_point now);
    void abandonResend(Clock::time_point now);
    void consume(const Event& event);
    OmniUsdLiveResult saveImpl(uint32_t timeoutMs, bool allowBlock, uint64_t* targetLocalOrdinal) noexcept;
    void observeReadReady() noexcept;
    void recordReadDisconnect(OmniUsdLiveResult result, uint64_t requestId, uint64_t attemptId) noexcept;
    void failReadStream(OmniUsdLiveResult result);
    bool authenticate(Clock::time_point now, Clock::time_point& next);
    void publish();
    void publishStatus() noexcept;
    bool applyReady(Clock::time_point& next);
    void settleApplication();
    void retireAppliedDrain() noexcept;
    // An append failure defers incoming owners until the worker latches its cause.
    OmniUsdLiveResult adoptAbsorbed(std::vector<std::shared_ptr<const void>> owners) noexcept;
    void completeAbsorbed(std::vector<std::shared_ptr<const void>>& owners) noexcept;
    bool localReady();
    void startApplication(std::vector<LayerProgress::Work>, std::vector<std::shared_ptr<const Effect>>, bool local);
    // Settles applied work's progress bookkeeping in order, then publishes.
    void settleApplied();
    void pollPing(Clock::time_point now, Clock::time_point& next);
    // `keepReading` stops issuance only (definitive rejection, host failure):
    // the read stays open and the journal keeps receiving (design section 13).
    void terminal(OmniUsdLiveResult result, const OmniUsdLiveRequestError* evidence = nullptr,
        bool keepReading = false);
    void cancelRead();
    void checkReadDeadline(bool read, bool queuedFrames, Clock::time_point now, Clock::time_point& next);
    void cancelBoth();
};
} // namespace omni::usd::live
