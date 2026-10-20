// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <omni/usd/live/Live.h>
#include "LiveDispatch.h"
#include "LiveStatus.h"
#include "LiveJitter.h"
#include <array>
#include <chrono>
#include <initializer_list>
#include <memory>
#include <string_view>
#include <vector>

namespace omni::usd::live {
// Numeric, catalog-owned keys only. No arbitrary strings, server details,
// authentication material or application payloads are accepted.
enum class DiagnosticKey : uint32_t {
    Operation,
    Result,
    GrpcStatus,
    Request,
    Attempt,
    Generation,
    AuthGeneration,
    Challenge,
    Reason,
    DelayMs,
    PendingBytes,
    PendingWrites,
    DurationMs,
    Count,
    State,
    Revision,
    ReadSequence,
    WriteSequence,
    AppliedSequence,
    Bytes,
    RetryDisposition,
    QueueDepth,
    SampleAgeMs,
    ResumeSequence,
    Work,
    Session,
    TargetLocalOrdinal,
    LocalRetired,
    AuthFailureCause,
    CallbackKind,
    Callback,
    StructCommands,
    SetFields,
    TimeSamples,
    Limit,
    CountKeys
};
// The live edit service's refusal of a layer write or creation over its entry
// limits, from the google.rpc.ErrorInfo of domain nvidia.omniverse.live.v1alpha
// its INVALID_ARGUMENT carries. reason is the rpc_failed record's reason: 1
// DELTA_TOO_LARGE, with the delta's structCommands, setFields and timeSamples,
// their sum in actual and max_entries in limit; 2 CHILDREN_LIST_TOO_LONG, with
// limit and actual. Zero is no such refusal.
struct ServiceLimitRefusal {
    uint32_t reason = 0;
    uint64_t structCommands = 0, setFields = 0, timeSamples = 0, limit = 0, actual = 0;
};
struct DiagnosticValue {
    DiagnosticKey key;
    uint64_t value;
};
struct DiagnosticEntry {
    uint32_t eventCode = 0, severity = 0, fieldCount = 0;
    uint64_t timestampMs = 0, elapsedMs = 0, runtimeInstanceId = 0, sequence = 0;
    OmniUsdLiveEndpoint endpoint = 0;
    uint64_t subjectId = 0, incidentId = 0;
    std::array<DiagnosticValue, 16> fields{};
    std::array<char, 513> resource{};
};
inline constexpr size_t kDiagnosticEventCount = OMNI_USD_LIVE_RECORD_APPLY_LAG_EXITED + 1;
enum class CallbackKind : uint32_t { Request = 0, Auth = 1, Status = 2, Session = 3 };
enum class AuthWaitResolution : uint32_t { Usable = 0, Failed = 1, Closed = 2 };
enum class AuthFailureCause : uint32_t { Provider = 0, Deadline = 1, RejectionLimit = 2, Internal = 3 };
enum class AuthWaitKind : uint32_t { Finite = 0, Layer = 1, Session = 2 };
struct DiagnosticOperationAuthMetrics {
    uint64_t current = 0, highWater = 0, started = 0, completed = 0;
    std::array<MetricDuration, 23> durationByResult{};
};
enum class ApplicationKind : uint32_t { LocalAsync = 0, OrderedAsync = 1, OrderedSync = 2 };
enum class ShutdownPhase : uint32_t {
    ReferenceRelease = 0,
    Admission = 1,
    CompletionQueue = 2,
    Sessions = 3,
    Authentication = 4,
    Status = 5,
    Requests = 6,
    Diagnostics = 7,
    Complete = 8
};
struct DiagnosticApplicationMetrics {
    uint64_t current = 0, highWater = 0, admitted = 0, retired = 0;
    MetricDuration queue, body, postBodyRetention;
    std::array<MetricDuration, 23> lifetimeByResult{};
};
struct DiagnosticJitterMetrics {
    uint64_t current = 0, highWater = 0, started = 0, completed = 0;
    uint64_t frozenStarts = 0, unsampledStarts = 0;
    MetricDuration requestedWait, targetHold, sampleAge;
    std::array<MetricDuration, 4> holdByResolution{}, postDueByResolution{};
};
// Layers whose applied sequence trails their read sequence past the warning
// threshold, and each peer update's time from its frame's receipt by the read
// to the retirement of the application that installed it. Every update in one
// batched application measures to the same retirement. Fixed storage only:
// the maximum and the histogram's fields each saturate independently.
struct DiagnosticApplyLagMetrics {
    uint64_t current = 0, highWater = 0, receiveToApplyMaxMs = 0;
    MetricDuration receiveToApply;
    void observe(uint64_t durationMs) noexcept {
        receiveToApply.observe(durationMs);
        receiveToApplyMaxMs = std::max(receiveToApplyMaxMs, durationMs);
    }
};
// Chained layer writes (design section 13): local writes in flight across
// every layer and the most at once, the largest issuance window a layer
// reached and how often windows grew and halved, chain breaks that re-sent
// writes and the writes they re-sent, and admitted updates with the host
// turns they hold. Fixed storage; counters saturate independently.
struct DiagnosticLayerWriteMetrics {
    uint64_t inFlight = 0, inFlightHighWater = 0, windowHighWater = 0, windowGrown = 0, windowHalved = 0;
    uint64_t chainResends = 0, resentWrites = 0, admittedUpdates = 0, coalescedTurns = 0, maxTurnsPerUpdate = 0;
};
struct DiagnosticCqMetrics {
    uint64_t submitted = 0, receivedOk = 0, receivedNotOk = 0, retired = 0, abandoned = 0;
    // Exact current counts; cumulative fields saturate. Retained includes
    // received tags until their derived owners/captures have been destroyed.
    uint64_t outstanding = 0, outstandingHighWater = 0, handling = 0;
    uint64_t retained = 0, retainedHighWater = 0;
};
struct DiagnosticSlotMetrics {
    uint64_t current = 0, highWater = 0, admitted = 0, released = 0, rejected = 0;
};
struct DiagnosticSubjectMetrics {
    uint64_t current = 0, highWater = 0, admitted = 0, detached = 0;
    // Public state values 1..9; slot zero is unused. Attached subjects only,
    // even when detached final observations still consume registry capacity.
    std::array<uint64_t, OMNI_USD_LIVE_STATE_CLOSED + 1> byState{};
};
struct DiagnosticReadinessMetrics {
    uint64_t initialWaiting = 0, recovering = 0;
    std::array<MetricDuration, 3> initialByResolution{}, recoveryByResolution{};
};
struct DiagnosticBackpressureMetrics {
    uint64_t current = 0, highWater = 0;
    std::array<MetricDuration, 3> durationByResolution{};
};
// A runtime aggregates multiple uint64 endpoint budgets. Keep the full sum so
// releasing a reservation after overflow restores the correct visible gauge.
struct DiagnosticByteGauge {
    uint64_t low = 0, high = 0;
    void add(uint64_t value) noexcept {
        const auto previous = low;
        low += value;
        high += low < previous;
    }
    void subtract(uint64_t value) noexcept {
        const bool borrow = low < value;
        low -= value;
        high -= borrow;
    }
    bool operator<(const DiagnosticByteGauge& other) const noexcept {
        return high < other.high || (high == other.high && low < other.low);
    }
};
struct DiagnosticByteMetrics {
    DiagnosticSlotMetrics reservations;
    DiagnosticByteGauge currentBytes, highWaterBytes;
    uint64_t admittedBytes = 0, releasedBytes = 0, rejectedBytes = 0;
};
struct DiagnosticMetrics {
    // Runtime aggregates even in an endpoint-filtered snapshot: no unbounded
    // endpoint/request/URI metric labels. Counters saturate at UINT64_MAX.
    uint64_t emitted = 0, suppressed = 0, warningSuppressed = 0, invalid = 0, historyOverwritten = 0;
    uint64_t historyDropped = 0, sinkDropped = 0, sinkFailures = 0, sinksDisabled = 0;
    uint64_t closedIncidentsEvicted = 0, rpcCompletions = 0, rpcAttempts = 0, rpcInFlight = 0;
    std::array<uint64_t, 17> rpcAttemptsByOperation{};
    std::array<uint64_t, kDiagnosticEventCount> events{};
    // Accepted, write-completed, received, discarded, rejected payload bytes.
    std::array<uint64_t, 5> messageBytes{};
    // Index16/22 are unknown operation/result buckets, not arbitrary labels.
    std::array<std::array<uint64_t, 23>, 17> rpc{};
    // Noncumulative buckets <=1,5,10,50,100,500,1000,5000 ms, then +Inf.
    std::array<uint64_t, 9> rpcDuration{};
    // Fixed operation/result dimensions, including the same unknown buckets as
    // rpc above. No subject/endpoint labels or allocation on observation.
    std::array<std::array<MetricDuration, 23>, 17> rpcDurationByOperationResult{};
    // Completed actor saves and public drain calls, including failed outcomes.
    // Independent of RPC metrics and of history/sink delivery or suppression.
    std::array<MetricDuration, 23> saveDurationByResult{}, drainDurationByResult{};
    // Whole matched-runtime shutdown calls classified by ending phase/result.
    std::array<std::array<MetricDuration, 23>, 9> shutdownDurationByPhaseResult{};
    uint64_t authBlockedEndpoints = 0, authBlockedEndpointsHighWater = 0, authStaleReplacements = 0;
    // Challenge reason 1..3; index0 groups unknown reasons. Failure/resolution
    // dimensions are the fixed private diagnostic enums above, not public results.
    std::array<uint64_t, 4> authChallengesByReason{}, authFailuresByCause{};
    std::array<MetricDuration, 3> authWaitByResolution{};
    std::array<DiagnosticOperationAuthMetrics, 3> operationAuthWaits;
    DiagnosticCqMetrics cqTags;
    std::array<DiagnosticSlotMetrics, 2> requestSlots; // Ordinary, auxiliary timing.
    DiagnosticByteMetrics pendingBytes;
    std::array<DiagnosticSubjectMetrics, 3> subjects; // Endpoint, session, layer.
    std::array<DiagnosticReadinessMetrics, 2> readiness; // Session, layer; endpoints excluded.
    std::array<DiagnosticBackpressureMetrics, 3> backpressure; // Endpoint, session, layer BLOCKED intervals.
    std::array<DiagnosticApplicationMetrics, 3> applications;
    DiagnosticJitterMetrics jitter;
    DiagnosticApplyLagMetrics applyLag;
    DiagnosticLayerWriteMetrics layerWrites;
};
struct DiagnosticIncident {
    OmniUsdLiveEndpoint endpoint = 0;
    OmniUsdLiveStatus status{};
    std::array<char, 513> resource{};
};
struct DiagnosticSnapshot {
    uint64_t runtimeInstanceId = 0, captureTimeMs = 0, firstSequence = 0, lastSequence = 0;
    bool truncated = false;
    DiagnosticMetrics metrics;
    DispatchMetrics diagnosticExecutor;
    std::vector<DiagnosticEntry> records;
    std::vector<DiagnosticIncident> closedIncidents;
};

// Independent of Runtime; owns two diagnostic workers. Construction/admission
// and snapshot allocate; emit/observeRpc/incident retention never allocate.
// Destroy outside dispatch workers after users stop entering this object.
class DiagnosticsRegistry final {
public:
    using Clock = std::chrono::steady_clock;
    // Optional private monotonic clock seam for deterministic suppression tests;
    // context outlives this object. Wait deadlines always use real steady time.
    DiagnosticsRegistry(const OmniUsdLiveRuntimeOptions& normalized, uint64_t runtimeInstanceId,
        Clock::time_point (*now)(void*) noexcept = nullptr, void* clockContext = nullptr);
    ~DiagnosticsRegistry();
    DiagnosticsRegistry(const DiagnosticsRegistry&) = delete;
    DiagnosticsRegistry& operator=(const DiagnosticsRegistry&) = delete;
    static const char* eventName(uint32_t code) noexcept;
    static uint32_t eventSeverity(uint32_t code) noexcept;
    static const char* fieldName(DiagnosticKey key) noexcept;
    static constexpr size_t recordStorageBytes() noexcept { return sizeof(DiagnosticEntry); }
    // Exact repeated WARN records coalesce within a 1000ms per-event window.
    // First occurrences and changed identities/fields are delivered. Suppressed
    // input increments counters and consumes sequence; the next changed/expired
    // warning of that code emits a count/reason summary first. Pending summaries
    // remain represented in counters even if no later warning arrives. ERROR,
    // INFO, DEBUG and RPC metrics are never rate limited.
    void emit(uint32_t code, OmniUsdLiveEndpoint endpoint = 0, uint64_t subject = 0, uint64_t incident = 0,
        std::initializer_list<DiagnosticValue> fields = {}, std::string_view canonicalResource = {},
        uint32_t severity = UINT32_MAX) noexcept;
    void startedRpc(uint32_t operation) noexcept;
    // Tag lifetime observations continue through CQ drain after beginShutdown.
    // No callback, allocation or history emission from these methods.
    void submittedCqTag() noexcept;
    void receivedCqTag(bool ok) noexcept;
    void returnedCqTag() noexcept;
    void destroyedCqTag(bool received) noexcept;
    void admittedRequestSlot(bool timing, bool admitted) noexcept;
    void releasedRequestSlot(bool timing) noexcept;
    void admittedPendingBytes(uint64_t amount, bool admitted) noexcept;
    void releasedPendingBytes(uint64_t amount) noexcept;
    // Called only for admitted/changed/detached status-hub publications.
    void subjectTransition(uint32_t kind, uint32_t previousState, uint32_t state, bool attached,
        const StatusTiming& timing) noexcept;
    void startAuthWait() noexcept;
    void endAuthWait(AuthWaitResolution resolution, uint64_t durationMs) noexcept;
    void startOperationAuthWait(AuthWaitKind kind) noexcept;
    void endOperationAuthWait(AuthWaitKind kind, OmniUsdLiveResult result, uint64_t durationMs) noexcept;
    // Private observation barrier; never advances operational clocks or state.
    bool waitForOperationAuthWaitsForTest(AuthWaitKind kind, uint64_t current, Clock::time_point deadline);
    void observeAuthFailure(AuthFailureCause cause) noexcept;
    void staleAuthReplacement() noexcept;
    void admittedApplication(ApplicationKind kind) noexcept;
    // Async run samples exist only after a successful run claim. Synchronous
    // callbacks have body samples but no host-queue or completion-owner tail.
    void retiredApplication(ApplicationKind kind, OmniUsdLiveResult result, uint64_t lifetimeMs, bool ran,
        uint64_t queueMs, uint64_t bodyMs, uint64_t postBodyMs) noexcept;
    // One receive-to-apply sample per peer update an ordered application
    // installed; a batch reports its updates under one lock.
    void appliedPeerUpdates(const uint64_t* receiveToApplyMs, size_t count) noexcept;
    // A layer entered (true) or left (false) its apply-lag warning interval.
    void applyLagChanged(bool lagging) noexcept;
    // A local write attempt entered (true), under issuance window `window`, or
    // left (false) flight.
    void layerWriteInFlight(bool entered, uint32_t window = 0) noexcept;
    // A layer's issuance window became `window`, growing or halving.
    void layerWindowChanged(uint32_t window, bool grew) noexcept;
    // A broken chain re-sends `writes` writes.
    void layerChainResend(uint64_t writes) noexcept;
    // An admitted update holds `turns` host turns.
    void layerUpdateAdmitted(uint64_t turns) noexcept;
    // Null decision means retained old-generation eligibility; do not borrow
    // the new generation's estimator age/target for that frozen decision.
    void startedJitterWait(uint64_t requestedMs, const JitterEstimator::Decision* decision) noexcept;
    void finishedJitterWait(JitterWaitResolution resolution, JitterWait::Measurement measurement) noexcept;
    void observeRpc(OmniUsdLiveEndpoint endpoint, uint32_t operation, OmniUsdLiveResult result, uint64_t durationMs,
        uint64_t request, uint64_t attempt, bool grpcPresent, uint32_t grpcCode, uint64_t subject = 0,
        uint64_t streamGeneration = 0, uint64_t authGeneration = 0, uint64_t work = 0, uint64_t session = 0,
        const ServiceLimitRefusal* refusal = nullptr) noexcept;
    // Records even after beginShutdown, without enqueueing any diagnostic sink.
    // Only TIMEOUT emits quiescence_failed; reference release/BUSY/success are metrics.
    void observeShutdown(ShutdownPhase phase, OmniUsdLiveResult result, uint64_t durationMs) noexcept;
    void retainClosedIncident(OmniUsdLiveEndpoint endpoint, std::string_view canonicalResource,
        const OmniUsdLiveStatus& status) noexcept;
    // History uses sizeof(DiagnosticEntry) per reserved slot; its capacity is
    // min(max_diagnostic_records, max_diagnostic_bytes / recordStorageBytes()).
    // History's minimum severity does not raise a sink's explicit filter.
    // Resource output requires runtime AND per-call permission. Invalid or
    // overlong resources are omitted and counted, never truncate identities.
    // Only live:// identities are accepted, never HTTP/base/signed URLs.
    // SERVER_DETAIL
    // is accepted as a flag but never enables arbitrary text in this component.
    DiagnosticSnapshot snapshot(OmniUsdLiveEndpoint endpoint, uint32_t maxRecords, uint32_t flags = 0,
        uint32_t maxIncidents = UINT32_MAX) const;
    // Options copied at admission. No replay of history. Sink overflow drops
    // newest, preserving queued FIFO; capacity excludes one in-flight record.
    // Three total callback exceptions disable
    // delivery and retire its lease, but handle remains until unsubscribe.
    OmniUsdLiveCallbackHandle subscribe(const OmniUsdLiveDiagnosticOptions& options, void* user,
        OmniUsdLiveDiagnosticCallback callback, std::shared_ptr<void> lifetimeLease = {});
    OmniUsdLiveResult unsubscribe(OmniUsdLiveCallbackHandle handle, uint32_t timeoutMs);
    bool hasSinks() const noexcept;
    void beginShutdown() noexcept;
    bool waitForShutdown(Clock::time_point deadline);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
// Serialized logical-operation owner only. This observational interval never
// drives operational auth flags/deadlines. Registry outlives the operation.
class OperationAuthWait {
public:
    using Clock = std::chrono::steady_clock;
    OperationAuthWait(DiagnosticsRegistry& diagnostics, AuthWaitKind kind) noexcept
        : diagnostics_(diagnostics), kind_(kind) {}
    OperationAuthWait(const OperationAuthWait&) = delete;
    OperationAuthWait& operator=(const OperationAuthWait&) = delete;
    ~OperationAuthWait();
    void start(Clock::time_point now) noexcept;
    void finish(OmniUsdLiveResult result, Clock::time_point now) noexcept;
private:
    DiagnosticsRegistry& diagnostics_;
    const AuthWaitKind kind_;
    Clock::time_point started_{};
    bool active_ = false;
};
} // namespace omni::usd::live
