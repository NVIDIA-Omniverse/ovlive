// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "LiveDiagnostics.h"
#include "LiveDispatch.h"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <mutex>

namespace omni::usd::live {
namespace {
constexpr const char* eventNames[] = {"invalid", "endpoint_registered", "endpoint_closed", "rpc_completed",
    "rpc_failed", "auth_challenge", "auth_replaced", "auth_failed", "stream_disconnected", "stream_resumed",
    "layer_ready", "object_replaced", "recovery_gate_closed", "recovery_gate_opened", "outcome_unknown",
    "backpressure_entered", "backpressure_exited", "save_completed", "drain_completed", "drain_timeout",
    "quiescence_failed", "retry_scheduled", "records_suppressed", "invalid", "message_accepted",
    "message_write_completed", "message_received", "message_discarded", "message_rejected", "callback_failed",
    "flush_unsupported", "host_edits_discarded", "read_paused", "read_resumed", "apply_lag_entered",
    "apply_lag_exited"};
static_assert(std::size(eventNames) == kDiagnosticEventCount);
constexpr uint32_t kReservedEventCode = 23; // Reserved: never emitted or subscribable.
constexpr const char* fieldNames[] = {"operation", "result", "grpc_status", "request_id", "attempt_id",
    "stream_generation", "auth_generation", "challenge_id", "reason", "delay_ms", "pending_bytes", "pending_writes",
    "duration_ms", "count", "state", "revision", "read_sequence", "write_sequence", "applied_sequence", "bytes",
    "retry_disposition", "queue_depth", "sample_age_ms", "resume_sequence", "work_id", "session_handle",
    "target_local_ordinal", "local_retired", "auth_failure_cause", "callback_kind", "callback_id", "struct_commands",
    "set_fields", "time_samples", "limit"};
constexpr uint32_t allowedFlags = OMNI_USD_LIVE_DIAGNOSTIC_RESOURCE_NAMES | OMNI_USD_LIVE_DIAGNOSTIC_SERVER_DETAIL;
std::atomic<uint64_t> nextSink{1};
void increment(uint64_t& value) noexcept {
    if (value != UINT64_MAX) ++value;
}
uint64_t wallMs() noexcept {
    const auto v = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch())
                       .count();
    return v < 0 ? 0 : static_cast<uint64_t>(v);
}
size_t operationIndex(uint32_t operation) noexcept { return operation < 16 ? operation : 16; }
size_t resultIndex(uint64_t result) noexcept { return result < 22 ? result : 22; }
bool sameWarning(const DiagnosticEntry& a, const DiagnosticEntry& b) noexcept {
    if (a.endpoint != b.endpoint || a.subjectId != b.subjectId || a.incidentId != b.incidentId ||
        a.fieldCount != b.fieldCount || a.resource != b.resource)
        return false;
    for (uint32_t i = 0; i < a.fieldCount; ++i)
        if (a.fields[i].key != b.fields[i].key || a.fields[i].value != b.fields[i].value) return false;
    return true;
}
uint64_t allocateHandle() {
    auto ordinal = nextSink.load(std::memory_order_relaxed);
    do {
        if (ordinal >= (UINT64_C(1) << 56)) throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    } while (!nextSink.compare_exchange_weak(ordinal, ordinal + 1, std::memory_order_relaxed));
    return (UINT64_C(8) << 56) | ordinal;
}
bool resourceValid(std::string_view value) noexcept {
    // Caller supplies a canonical resource identity, never a signed/base URL.
    // Reject controls/NUL; do not truncate names into a different identity.
    if (value.empty()) return true;
    if (value.size() > 512 || value.substr(0, 7) != "live://") return false;
    for (const unsigned char c : value)
        if (c <= 32 || c == 127 || c == '@' || c == '?' || c == '#' || c == '\\') return false;
    return true;
}
}

const char* DiagnosticsRegistry::eventName(uint32_t code) noexcept {
    return code < std::size(eventNames) ? eventNames[code] : eventNames[0];
}
uint32_t DiagnosticsRegistry::eventSeverity(uint32_t code) noexcept {
    switch (code) {
    case OMNI_USD_LIVE_RECORD_MESSAGE_ACCEPTED:
    case OMNI_USD_LIVE_RECORD_MESSAGE_WRITE_COMPLETED:
    case OMNI_USD_LIVE_RECORD_MESSAGE_RECEIVED:
    case OMNI_USD_LIVE_RECORD_RPC_COMPLETED: return OMNI_USD_LIVE_SEVERITY_DEBUG;
    case OMNI_USD_LIVE_RECORD_MESSAGE_DISCARDED:
    case OMNI_USD_LIVE_RECORD_HOST_EDITS_DISCARDED:
    case OMNI_USD_LIVE_RECORD_MESSAGE_REJECTED:
    case OMNI_USD_LIVE_RECORD_RPC_FAILED:
    case OMNI_USD_LIVE_RECORD_AUTH_CHALLENGE:
    case OMNI_USD_LIVE_RECORD_STREAM_DISCONNECTED:
    case OMNI_USD_LIVE_RECORD_BACKPRESSURE_ENTERED:
    case OMNI_USD_LIVE_RECORD_READ_PAUSED:
    case OMNI_USD_LIVE_RECORD_APPLY_LAG_ENTERED:
    case OMNI_USD_LIVE_RECORD_DRAIN_TIMEOUT:
    case OMNI_USD_LIVE_RECORD_RECORDS_SUPPRESSED: return OMNI_USD_LIVE_SEVERITY_WARN;
    case OMNI_USD_LIVE_RECORD_AUTH_FAILED:
    case OMNI_USD_LIVE_RECORD_OBJECT_REPLACED:
    case OMNI_USD_LIVE_RECORD_OUTCOME_UNKNOWN:
    case OMNI_USD_LIVE_RECORD_CALLBACK_FAILED:
    case OMNI_USD_LIVE_RECORD_FLUSH_UNSUPPORTED:
    case OMNI_USD_LIVE_RECORD_QUIESCENCE_FAILED: return OMNI_USD_LIVE_SEVERITY_ERROR;
    default: return OMNI_USD_LIVE_SEVERITY_INFO;
    }
}
const char* DiagnosticsRegistry::fieldName(DiagnosticKey key) noexcept {
    const auto index = static_cast<size_t>(key);
    return index < std::size(fieldNames) ? fieldNames[index] : nullptr;
}

struct DiagnosticsRegistry::Impl {
    struct Sink final : DispatchJob {
        Impl& hub;
        const uint64_t handle;
        const OmniUsdLiveEndpoint endpoint;
        const uint32_t severity, flags;
        const std::array<bool, kDiagnosticEventCount> events;
        void* user;
        OmniUsdLiveDiagnosticCallback callback;
        std::shared_ptr<void> lease;
        std::vector<DiagnosticEntry> ring;
        size_t head = 0, count = 0;
        unsigned failures = 0;
        bool scheduled = false, cancelled = false, quiescent = false;
        Sink(Impl& h, uint64_t id, const OmniUsdLiveDiagnosticOptions& options,
            std::array<bool, kDiagnosticEventCount> filter, void* data, OmniUsdLiveDiagnosticCallback cb,
            std::shared_ptr<void> lifetime)
            : hub(h), handle(id), endpoint(options.endpoint), severity(options.min_severity),
              flags(options.flags & h.flags), events(filter), user(data), callback(cb), lease(std::move(lifetime)),
              ring(h.sinkCapacity) {}
        void schedule() noexcept {
            if (!scheduled && !quiescent) {
                scheduled = true;
                hub.executor.enqueue(*this);
            }
        }
        void dispatch() noexcept override;
    };
    const uint64_t instance;
    const uint32_t flags, minimumSeverity;
    const size_t sinkCapacity;
    Clock::time_point (*const nowFunction)(void*) noexcept;
    void* const clockContext;
    Clock::time_point now() const noexcept { return nowFunction ? nowFunction(clockContext) : Clock::now(); }
    const Clock::time_point start;
    mutable std::mutex mutex;
    std::condition_variable changed;
    std::vector<DiagnosticEntry> history;
    size_t historyHead = 0, historyCount = 0;
    std::vector<DiagnosticIncident> incidents;
    size_t incidentHead = 0, incidentCount = 0;
    std::vector<std::shared_ptr<Sink>> sinks;
    DiagnosticMetrics metrics;
    size_t operationAuthTestWaiters = 0;
    struct Warning {
        DiagnosticEntry prior;
        Clock::time_point since{};
        uint64_t suppressed = 0;
        bool initialized = false;
    };
    std::array<Warning, kDiagnosticEventCount> warnings;
    uint64_t sequence = 0;
    bool stopping = false;
    // Declared last: explicit destructor waits before these threads are joined;
    // all ring/registry members remain alive until the executor is gone.
    DispatchExecutor executor{2};
    Impl(const OmniUsdLiveRuntimeOptions& o, uint64_t id, Clock::time_point (*clock)(void*) noexcept, void* context)
        : instance(id), flags(o.diagnostic_allowed_flags & allowedFlags), minimumSeverity(o.diagnostic_min_severity),
          sinkCapacity(o.max_diagnostic_sink_records), nowFunction(clock), clockContext(context), start(now()),
          history((o.flags & OMNI_USD_LIVE_RUNTIME_DISABLE_DIAGNOSTIC_HISTORY) ?
                  0 :
                  static_cast<size_t>(
                      std::min<uint64_t>(o.max_diagnostic_records, o.max_diagnostic_bytes / sizeof(DiagnosticEntry)))),
          incidents(o.max_closed_incidents), sinks(o.max_diagnostic_sinks) {
        if (minimumSeverity > OMNI_USD_LIVE_SEVERITY_ERROR) throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
    }
    void publish(DiagnosticEntry& entry, bool deliver = true) noexcept {
        // All mutation happens with mutex held; fixed rings and intrusive
        // dispatch reservations require no allocation in this path.
        increment(metrics.emitted);
        increment(metrics.events[entry.eventCode]);
        if (entry.eventCode == OMNI_USD_LIVE_RECORD_AUTH_CHALLENGE) {
            uint64_t reason = 0;
            for (uint32_t i = 0; i < entry.fieldCount; ++i)
                if (entry.fields[i].key == DiagnosticKey::Reason) reason = entry.fields[i].value;
            increment(metrics.authChallengesByReason[reason < 4 ? reason : 0]);
        }
        if (entry.eventCode == OMNI_USD_LIVE_RECORD_SAVE_COMPLETED ||
            entry.eventCode == OMNI_USD_LIVE_RECORD_DRAIN_COMPLETED ||
            entry.eventCode == OMNI_USD_LIVE_RECORD_DRAIN_TIMEOUT) {
            bool haveDuration = false, haveResult = false;
            uint64_t duration = 0, result = 0;
            for (uint32_t i = 0; i < entry.fieldCount; ++i) {
                const auto& field = entry.fields[i];
                if (field.key == DiagnosticKey::DurationMs) {
                    haveDuration = true;
                    duration = field.value;
                }
                if (field.key == DiagnosticKey::Result) {
                    haveResult = true;
                    result = field.value;
                }
            }
            // Other emitters may omit duration; never manufacture a zero sample.
            if (haveDuration && haveResult) {
                auto& rows = entry.eventCode == OMNI_USD_LIVE_RECORD_SAVE_COMPLETED ? metrics.saveDurationByResult :
                                                                                      metrics.drainDurationByResult;
                rows[resultIndex(result)].observe(duration);
            }
        }
        if (entry.eventCode >= OMNI_USD_LIVE_RECORD_MESSAGE_ACCEPTED &&
            entry.eventCode <= OMNI_USD_LIVE_RECORD_MESSAGE_REJECTED) {
            auto& bytes = metrics.messageBytes[entry.eventCode - OMNI_USD_LIVE_RECORD_MESSAGE_ACCEPTED];
            for (uint32_t i = 0; i < entry.fieldCount; ++i)
                if (entry.fields[i].key == DiagnosticKey::Bytes)
                    bytes += std::min(UINT64_MAX - bytes, entry.fields[i].value);
        }
        if (sequence == UINT64_MAX) {
            increment(metrics.historyDropped);
            return;
        }
        const auto observedNow = now();
        if (entry.severity == OMNI_USD_LIVE_SEVERITY_WARN &&
            entry.eventCode != OMNI_USD_LIVE_RECORD_RECORDS_SUPPRESSED) {
            auto& warning = warnings[entry.eventCode];
            if (warning.initialized && observedNow - warning.since < std::chrono::milliseconds(1000) &&
                sameWarning(warning.prior, entry)) {
                ++sequence;
                increment(warning.suppressed);
                increment(metrics.warningSuppressed);
                increment(metrics.suppressed);
                return;
            }
            if (warning.suppressed) {
                DiagnosticEntry summary;
                summary.eventCode = OMNI_USD_LIVE_RECORD_RECORDS_SUPPRESSED;
                summary.severity = OMNI_USD_LIVE_SEVERITY_WARN;
                summary.endpoint = warning.prior.endpoint;
                summary.subjectId = warning.prior.subjectId;
                summary.incidentId = warning.prior.incidentId;
                summary.fields[0] = {DiagnosticKey::Count, warning.suppressed};
                summary.fields[1] = {DiagnosticKey::Reason, entry.eventCode};
                summary.fieldCount = 2;
                publish(summary, deliver);
            }
            warning.prior = entry;
            warning.since = observedNow;
            warning.suppressed = 0;
            warning.initialized = true;
            if (sequence == UINT64_MAX) {
                increment(metrics.historyDropped);
                return;
            }
        }
        entry.sequence = ++sequence;
        entry.timestampMs = wallMs();
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(observedNow - start).count();
        entry.elapsedMs = elapsed < 0 ? 0 : static_cast<uint64_t>(elapsed);
        entry.runtimeInstanceId = instance;
        if (entry.severity < minimumSeverity) increment(metrics.suppressed);
        else if (history.empty()) increment(metrics.historyDropped);
        else {
            if (historyCount == history.size()) {
                historyHead = (historyHead + 1) % history.size();
                --historyCount;
                increment(metrics.historyOverwritten);
            }
            history[(historyHead + historyCount++) % history.size()] = entry;
        }
        if (!deliver) return;
        for (const auto& sink : sinks) {
            if (!sink || sink->cancelled || !sink->events[entry.eventCode] || entry.severity < sink->severity ||
                (sink->endpoint && sink->endpoint != entry.endpoint))
                continue;
            if (sink->count == sink->ring.size()) {
                increment(metrics.sinkDropped);
                continue;
            }
            sink->ring[(sink->head + sink->count++) % sink->ring.size()] = entry;
            sink->schedule();
        }
    }
};

void DiagnosticsRegistry::Impl::Sink::dispatch() noexcept {
    DiagnosticEntry entry;
    bool invoke = false;
    {
        std::lock_guard<std::mutex> lock(hub.mutex);
        if (!cancelled && count) {
            entry = ring[head];
            head = (head + 1) % ring.size();
            --count;
            invoke = true;
        }
    }
    bool failed = false;
    if (invoke) {
        std::array<OmniUsdLiveDiagnosticField, 16> fields{};
        std::array<const OmniUsdLiveDiagnosticField*, 16> pointers{};
        uint32_t n = 0;
        for (; n < entry.fieldCount; ++n) {
            auto& out = fields[n];
            out.struct_size = sizeof(out);
            out.abi_version = OMNI_USD_LIVE_ABI_VERSION;
            out.name = DiagnosticsRegistry::fieldName(entry.fields[n].key);
            out.kind = OMNI_USD_LIVE_FIELD_UINT64;
            out.unsigned_value = entry.fields[n].value;
            pointers[n] = &out;
        }
        if ((flags & OMNI_USD_LIVE_DIAGNOSTIC_RESOURCE_NAMES) && entry.resource[0] && n < fields.size()) {
            auto& out = fields[n];
            out.struct_size = sizeof(out);
            out.abi_version = OMNI_USD_LIVE_ABI_VERSION;
            out.name = "canonical_resource";
            out.kind = OMNI_USD_LIVE_FIELD_STRING;
            out.string_value = entry.resource.data();
            pointers[n] = &out;
            ++n;
        }
        OmniUsdLiveDiagnosticRecord record{};
        record.struct_size = sizeof(record);
        record.abi_version = OMNI_USD_LIVE_ABI_VERSION;
        record.schema_version = OMNI_USD_LIVE_DIAGNOSTIC_SCHEMA_VERSION;
        record.event_code = entry.eventCode;
        record.event_name = DiagnosticsRegistry::eventName(entry.eventCode);
        record.severity = entry.severity;
        record.timestamp_ms = entry.timestampMs;
        record.elapsed_ms = entry.elapsedMs;
        record.runtime_instance_id = entry.runtimeInstanceId;
        record.sequence = entry.sequence;
        record.endpoint = entry.endpoint;
        record.subject_id = entry.subjectId;
        record.incident_id = entry.incidentId;
        record.field_count = n;
        record.fields = pointers.data();
        try {
            callback(user, &record);
        } catch (...) { failed = true; }
    }
    std::shared_ptr<void> retired;
    {
        std::lock_guard<std::mutex> lock(hub.mutex);
        if (failed) {
            increment(hub.metrics.sinkFailures);
            if (++failures >= 3 && !cancelled) {
                cancelled = true;
                increment(hub.metrics.sinksDisabled);
            }
        }
        if (cancelled) {
            while (count) {
                --count;
                increment(hub.metrics.sinkDropped);
            }
            retired = std::move(lease);
        } else {
            if (count) hub.executor.enqueue(*this);
            else scheduled = false;
            hub.changed.notify_all();
            return;
        }
    }
    // Capture destruction can run host code; it stays on this diagnostic worker
    // outside every hub lock and is included in unregistration quiescence.
    retired.reset();
    {
        std::lock_guard<std::mutex> lock(hub.mutex);
        quiescent = true;
        scheduled = false;
        hub.changed.notify_all();
    }
}

DiagnosticsRegistry::DiagnosticsRegistry(const OmniUsdLiveRuntimeOptions& options, uint64_t instance,
    Clock::time_point (*now)(void*) noexcept, void* context)
    : impl_(std::make_unique<Impl>(options, instance, now, context)) {}
DiagnosticsRegistry::~DiagnosticsRegistry() {
    assert(!DispatchExecutor::isWorker());
    beginShutdown();
    (void)waitForShutdown(Clock::time_point::max());
}
void DiagnosticsRegistry::emit(uint32_t code, OmniUsdLiveEndpoint endpoint, uint64_t subject, uint64_t incident,
    std::initializer_list<DiagnosticValue> fields, std::string_view resource, uint32_t severity) noexcept {
    DiagnosticEntry entry;
    entry.eventCode = code;
    entry.endpoint = endpoint;
    entry.subjectId = subject;
    entry.incidentId = incident;
    entry.severity = severity == UINT32_MAX ? eventSeverity(code) : severity;
    bool valid = code > 0 && code < std::size(eventNames) && code != kReservedEventCode &&
        entry.severity <= OMNI_USD_LIVE_SEVERITY_ERROR && fields.size() <= 16u;
    std::array<bool, static_cast<size_t>(DiagnosticKey::CountKeys)> seen{};
    for (const auto& field : fields) {
        if (!valid) break;
        const auto key = static_cast<size_t>(field.key);
        if (key >= seen.size() || seen[key]) {
            valid = false;
            continue;
        }
        seen[key] = true;
        if (entry.fieldCount < entry.fields.size()) entry.fields[entry.fieldCount++] = field;
    }
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->stopping) return;
    if (!valid) {
        increment(impl_->metrics.invalid);
        return;
    }
    const bool resourceAccepted = resourceValid(resource) && (resource.empty() || fields.size() < 16);
    if (!resourceAccepted) increment(impl_->metrics.invalid);
    if ((impl_->flags & OMNI_USD_LIVE_DIAGNOSTIC_RESOURCE_NAMES) && resourceAccepted && !resource.empty())
        std::memcpy(entry.resource.data(), resource.data(), resource.size());
    impl_->publish(entry);
}
void DiagnosticsRegistry::observeShutdown(ShutdownPhase phase, OmniUsdLiveResult result, uint64_t durationMs) noexcept {
    const auto index = static_cast<size_t>(phase);
    assert(index < impl_->metrics.shutdownDurationByPhaseResult.size());
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->metrics.shutdownDurationByPhaseResult[index][resultIndex(result)].observe(durationMs);
    if (result != OMNI_USD_LIVE_TIMEOUT) return;
    DiagnosticEntry entry;
    entry.eventCode = OMNI_USD_LIVE_RECORD_QUIESCENCE_FAILED;
    entry.severity = OMNI_USD_LIVE_SEVERITY_ERROR;
    entry.fields[0] = {DiagnosticKey::Operation, OMNI_USD_LIVE_OPERATION_SHUTDOWN};
    entry.fields[1] = {DiagnosticKey::Result, result};
    entry.fields[2] = {DiagnosticKey::DurationMs, durationMs};
    entry.fields[3] = {DiagnosticKey::Reason, static_cast<uint32_t>(phase)};
    entry.fieldCount = 4;
    // Shutdown has already stopped sink delivery. Preserve bounded local
    // evidence without scheduling any callback or reviving an executor.
    impl_->publish(entry, false);
}
void DiagnosticsRegistry::subjectTransition(uint32_t kind, uint32_t previousState, uint32_t state, bool attached,
    const StatusTiming& timing) noexcept {
    assert(kind >= OMNI_USD_LIVE_SUBJECT_ENDPOINT && kind <= OMNI_USD_LIVE_SUBJECT_LAYER);
    assert(definedState(state));
    assert((!previousState || definedState(previousState)) && (attached || previousState));
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto& metric = impl_->metrics.subjects[kind - 1];
    if (previousState) {
        assert(metric.current && metric.byState[previousState]);
        --metric.byState[previousState];
    } else {
        assert(metric.current != UINT64_MAX);
        ++metric.current;
        increment(metric.admitted);
        metric.highWater = std::max(metric.highWater, metric.current);
    }
    if (attached) ++metric.byState[state];
    else {
        --metric.current;
        increment(metric.detached);
    }
    auto& pressure = impl_->metrics.backpressure[kind - 1];
    if (timing.backpressureStarted) {
        assert(pressure.current != UINT64_MAX);
        ++pressure.current;
        pressure.highWater = std::max(pressure.highWater, pressure.current);
    }
    if (timing.backpressureCompleted) {
        assert(pressure.current);
        --pressure.current;
        const auto resolution = static_cast<size_t>(timing.backpressureResolution);
        assert(resolution < pressure.durationByResolution.size());
        pressure.durationByResolution[resolution].observe(timing.backpressureMs);
    }
    if (kind != OMNI_USD_LIVE_SUBJECT_ENDPOINT) {
        auto& readiness = impl_->metrics.readiness[kind - OMNI_USD_LIVE_SUBJECT_SESSION];
        if (timing.initialStarted) ++readiness.initialWaiting;
        if (timing.recoveryStarted) ++readiness.recovering;
        const auto resolution = static_cast<size_t>(timing.resolution);
        assert(resolution < readiness.initialByResolution.size());
        if (timing.initialCompleted) {
            assert(readiness.initialWaiting);
            --readiness.initialWaiting;
            readiness.initialByResolution[resolution].observe(timing.initialMs);
        }
        if (timing.recoveryCompleted) {
            assert(readiness.recovering);
            --readiness.recovering;
            readiness.recoveryByResolution[resolution].observe(timing.recoveryMs);
        }
    }
}
void DiagnosticsRegistry::startedJitterWait(uint64_t requestedMs, const JitterEstimator::Decision* decision) noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto& metric = impl_->metrics.jitter;
    assert(metric.current != UINT64_MAX);
    ++metric.current;
    metric.highWater = std::max(metric.highWater, metric.current);
    increment(metric.started);
    metric.requestedWait.observe(requestedMs);
    if (decision) {
        metric.targetHold.observe(decision->targetHoldUs / 1000);
        if (decision->sampled) metric.sampleAge.observe(decision->sampleAgeUs / 1000);
    } else increment(metric.frozenStarts);
    if (!decision || !decision->sampled) increment(metric.unsampledStarts);
}
void DiagnosticsRegistry::finishedJitterWait(JitterWaitResolution resolution,
    JitterWait::Measurement measurement) noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto& metric = impl_->metrics.jitter;
    assert(metric.current);
    --metric.current;
    increment(metric.completed);
    const auto index = static_cast<size_t>(resolution);
    assert(index < metric.holdByResolution.size());
    metric.holdByResolution[index].observe(measurement.holdMs);
    metric.postDueByResolution[index].observe(measurement.postDueMs);
}
void DiagnosticsRegistry::appliedPeerUpdates(const uint64_t* receiveToApplyMs, size_t count) noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    for (size_t i = 0; i < count; ++i) impl_->metrics.applyLag.observe(receiveToApplyMs[i]);
}
void DiagnosticsRegistry::layerWriteInFlight(bool entered, uint32_t window) noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto& metric = impl_->metrics.layerWrites;
    if (entered) {
        assert(metric.inFlight != UINT64_MAX);
        ++metric.inFlight;
        metric.inFlightHighWater = std::max(metric.inFlightHighWater, metric.inFlight);
        metric.windowHighWater = std::max<uint64_t>(metric.windowHighWater, window);
    } else {
        assert(metric.inFlight);
        --metric.inFlight;
    }
}
void DiagnosticsRegistry::layerWindowChanged(uint32_t window, bool grew) noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto& metric = impl_->metrics.layerWrites;
    metric.windowHighWater = std::max<uint64_t>(metric.windowHighWater, window);
    increment(grew ? metric.windowGrown : metric.windowHalved);
}
void DiagnosticsRegistry::layerChainResend(uint64_t writes) noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto& metric = impl_->metrics.layerWrites;
    increment(metric.chainResends);
    metric.resentWrites = writes > UINT64_MAX - metric.resentWrites ? UINT64_MAX : metric.resentWrites + writes;
}
void DiagnosticsRegistry::layerUpdateAdmitted(uint64_t turns) noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto& metric = impl_->metrics.layerWrites;
    increment(metric.admittedUpdates);
    metric.coalescedTurns = turns > UINT64_MAX - metric.coalescedTurns ? UINT64_MAX : metric.coalescedTurns + turns;
    metric.maxTurnsPerUpdate = std::max(metric.maxTurnsPerUpdate, turns);
}
void DiagnosticsRegistry::applyLagChanged(bool lagging) noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto& metric = impl_->metrics.applyLag;
    if (lagging) {
        assert(metric.current != UINT64_MAX);
        ++metric.current;
        metric.highWater = std::max(metric.highWater, metric.current);
    } else {
        assert(metric.current);
        --metric.current;
    }
}
void DiagnosticsRegistry::admittedApplication(ApplicationKind kind) noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto& metric = impl_->metrics.applications[static_cast<size_t>(kind)];
    assert(metric.current != UINT64_MAX);
    ++metric.current;
    metric.highWater = std::max(metric.highWater, metric.current);
    increment(metric.admitted);
}
void DiagnosticsRegistry::retiredApplication(ApplicationKind kind, OmniUsdLiveResult result, uint64_t lifetimeMs,
    bool ran, uint64_t queueMs, uint64_t bodyMs, uint64_t postBodyMs) noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto& metric = impl_->metrics.applications[static_cast<size_t>(kind)];
    assert(metric.current);
    --metric.current;
    increment(metric.retired);
    // The reserved result shares the out-of-domain bucket; its own slot stays zero.
    const auto bucket = definedResult(result) ? size_t(result) : size_t(22);
    metric.lifetimeByResult[bucket].observe(lifetimeMs);
    if (ran) {
        metric.body.observe(bodyMs);
        if (kind != ApplicationKind::OrderedSync) {
            metric.queue.observe(queueMs);
            metric.postBodyRetention.observe(postBodyMs);
        }
    }
}
void DiagnosticsRegistry::admittedRequestSlot(bool timing, bool admitted) noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto& slots = impl_->metrics.requestSlots[timing ? 1 : 0];
    if (!admitted) {
        increment(slots.rejected);
        return;
    }
    increment(slots.admitted);
    assert(slots.current != UINT64_MAX);
    ++slots.current;
    slots.highWater = std::max(slots.highWater, slots.current);
}
void DiagnosticsRegistry::releasedRequestSlot(bool timing) noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto& slots = impl_->metrics.requestSlots[timing ? 1 : 0];
    assert(slots.current);
    --slots.current;
    increment(slots.released);
}
void DiagnosticsRegistry::admittedPendingBytes(uint64_t amount, bool admitted) noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto& bytes = impl_->metrics.pendingBytes;
    if (!admitted) {
        increment(bytes.reservations.rejected);
        bytes.rejectedBytes += std::min(UINT64_MAX - bytes.rejectedBytes, amount);
        return;
    }
    increment(bytes.reservations.admitted);
    assert(bytes.reservations.current != UINT64_MAX);
    ++bytes.reservations.current;
    bytes.reservations.highWater = std::max(bytes.reservations.highWater, bytes.reservations.current);
    bytes.admittedBytes += std::min(UINT64_MAX - bytes.admittedBytes, amount);
    bytes.currentBytes.add(amount);
    if (bytes.highWaterBytes < bytes.currentBytes) bytes.highWaterBytes = bytes.currentBytes;
}
void DiagnosticsRegistry::releasedPendingBytes(uint64_t amount) noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto& bytes = impl_->metrics.pendingBytes;
    assert(bytes.reservations.current && (bytes.currentBytes.high || bytes.currentBytes.low >= amount));
    --bytes.reservations.current;
    increment(bytes.reservations.released);
    bytes.releasedBytes += std::min(UINT64_MAX - bytes.releasedBytes, amount);
    bytes.currentBytes.subtract(amount);
}
void DiagnosticsRegistry::submittedCqTag() noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto& tags = impl_->metrics.cqTags;
    increment(tags.submitted);
    assert(tags.outstanding != UINT64_MAX && tags.retained != UINT64_MAX);
    ++tags.outstanding;
    ++tags.retained;
    tags.outstandingHighWater = std::max(tags.outstandingHighWater, tags.outstanding);
    tags.retainedHighWater = std::max(tags.retainedHighWater, tags.retained);
}
void DiagnosticsRegistry::receivedCqTag(bool ok) noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto& tags = impl_->metrics.cqTags;
    assert(tags.outstanding && tags.handling != UINT64_MAX);
    --tags.outstanding;
    ++tags.handling;
    increment(ok ? tags.receivedOk : tags.receivedNotOk);
}
void DiagnosticsRegistry::returnedCqTag() noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto& tags = impl_->metrics.cqTags;
    assert(tags.handling);
    --tags.handling;
}
void DiagnosticsRegistry::destroyedCqTag(bool received) noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto& tags = impl_->metrics.cqTags;
    assert(tags.retained);
    --tags.retained;
    increment(tags.retired);
    if (!received) {
        assert(tags.outstanding);
        --tags.outstanding;
        increment(tags.abandoned);
    }
}
void DiagnosticsRegistry::startedRpc(uint32_t operation) noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    increment(impl_->metrics.rpcAttempts);
    increment(impl_->metrics.rpcInFlight);
    increment(impl_->metrics.rpcAttemptsByOperation[operationIndex(operation)]);
}
void DiagnosticsRegistry::startAuthWait() noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    increment(impl_->metrics.authBlockedEndpoints);
    impl_->metrics.authBlockedEndpointsHighWater = std::max(impl_->metrics.authBlockedEndpointsHighWater,
        impl_->metrics.authBlockedEndpoints);
}
void DiagnosticsRegistry::endAuthWait(AuthWaitResolution resolution, uint64_t durationMs) noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto index = static_cast<size_t>(resolution);
    if (index >= impl_->metrics.authWaitByResolution.size()) {
        increment(impl_->metrics.invalid);
        return;
    }
    if (impl_->metrics.authBlockedEndpoints) --impl_->metrics.authBlockedEndpoints;
    impl_->metrics.authWaitByResolution[index].observe(durationMs);
}
void DiagnosticsRegistry::observeAuthFailure(AuthFailureCause cause) noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto index = static_cast<size_t>(cause);
    if (index >= impl_->metrics.authFailuresByCause.size()) {
        increment(impl_->metrics.invalid);
        return;
    }
    increment(impl_->metrics.authFailuresByCause[index]);
}
void DiagnosticsRegistry::startOperationAuthWait(AuthWaitKind kind) noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto index = static_cast<size_t>(kind);
    if (index >= impl_->metrics.operationAuthWaits.size()) {
        increment(impl_->metrics.invalid);
        return;
    }
    auto& metric = impl_->metrics.operationAuthWaits[index];
    ++metric.current;
    increment(metric.started);
    metric.highWater = std::max(metric.highWater, metric.current);
    if (impl_->operationAuthTestWaiters) impl_->changed.notify_all();
}
void DiagnosticsRegistry::endOperationAuthWait(AuthWaitKind kind, OmniUsdLiveResult result,
    uint64_t durationMs) noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto index = static_cast<size_t>(kind);
    if (index >= impl_->metrics.operationAuthWaits.size()) {
        increment(impl_->metrics.invalid);
        return;
    }
    auto& metric = impl_->metrics.operationAuthWaits[index];
    assert(metric.current);
    --metric.current;
    increment(metric.completed);
    metric.durationByResult[resultIndex(result)].observe(durationMs);
    if (impl_->operationAuthTestWaiters) impl_->changed.notify_all();
}
bool DiagnosticsRegistry::waitForOperationAuthWaitsForTest(AuthWaitKind kind, uint64_t current,
    Clock::time_point deadline) {
    std::unique_lock<std::mutex> lock(impl_->mutex);
    const auto index = static_cast<size_t>(kind);
    if (index >= impl_->metrics.operationAuthWaits.size()) return false;
    ++impl_->operationAuthTestWaiters;
    const bool reached = impl_->changed.wait_until(lock, deadline,
        [&] { return impl_->metrics.operationAuthWaits[index].current == current; });
    --impl_->operationAuthTestWaiters;
    return reached;
}
OperationAuthWait::~OperationAuthWait() {
    // Normal terminal paths resolve before publishing completion/stopped state.
    // Unexpected owner abandonment still retires the gauge, without callbacks.
    finish(OMNI_USD_LIVE_INTERNAL_ERROR, Clock::now());
}
void OperationAuthWait::start(Clock::time_point now) noexcept {
    if (active_) return;
    active_ = true;
    started_ = now;
    diagnostics_.startOperationAuthWait(kind_);
}
void OperationAuthWait::finish(OmniUsdLiveResult result, Clock::time_point now) noexcept {
    if (!active_) return;
    active_ = false;
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - started_).count();
    diagnostics_.endOperationAuthWait(kind_, result, static_cast<uint64_t>(std::max<int64_t>(0, elapsed)));
}
void DiagnosticsRegistry::staleAuthReplacement() noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    increment(impl_->metrics.authStaleReplacements);
}
void DiagnosticsRegistry::observeRpc(OmniUsdLiveEndpoint endpoint, uint32_t operation, OmniUsdLiveResult result,
    uint64_t durationMs, uint64_t request, uint64_t attempt, bool grpcPresent, uint32_t grpcCode, uint64_t subject,
    uint64_t streamGeneration, uint64_t authGeneration, uint64_t work, uint64_t session,
    const ServiceLimitRefusal* refusal) noexcept {
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        increment(impl_->metrics.rpcCompletions);
        if (impl_->metrics.rpcInFlight) --impl_->metrics.rpcInFlight;
        const auto op = operationIndex(operation), outcome = resultIndex(result);
        increment(impl_->metrics.rpc[op][outcome]);
        increment(impl_->metrics.rpcDuration[metricDurationBucket(durationMs)]);
        impl_->metrics.rpcDurationByOperationResult[op][outcome].observe(durationMs);
    }
    const auto code = result == OMNI_USD_LIVE_OK ? OMNI_USD_LIVE_RECORD_RPC_COMPLETED : OMNI_USD_LIVE_RECORD_RPC_FAILED;
    const auto severity = result == OMNI_USD_LIVE_OK ?
        OMNI_USD_LIVE_SEVERITY_DEBUG :
        (result == OMNI_USD_LIVE_UNAVAILABLE || result == OMNI_USD_LIVE_TIMEOUT || result == OMNI_USD_LIVE_CANCELLED ||
                    result == OMNI_USD_LIVE_UNAUTHENTICATED ?
                OMNI_USD_LIVE_SEVERITY_WARN :
                OMNI_USD_LIVE_SEVERITY_ERROR);
    // The service's entry-limit ErrorInfo adds its reason and counts.
    if (grpcPresent && refusal && refusal->reason == 1)
        emit(code, endpoint, subject, 0,
            {{DiagnosticKey::Operation, operation}, {DiagnosticKey::Result, result},
                {DiagnosticKey::DurationMs, durationMs}, {DiagnosticKey::Request, request},
                {DiagnosticKey::Attempt, attempt}, {DiagnosticKey::GrpcStatus, grpcCode},
                {DiagnosticKey::Generation, streamGeneration}, {DiagnosticKey::AuthGeneration, authGeneration},
                {DiagnosticKey::Work, work}, {DiagnosticKey::Session, session}, {DiagnosticKey::Reason, 1},
                {DiagnosticKey::StructCommands, refusal->structCommands},
                {DiagnosticKey::SetFields, refusal->setFields}, {DiagnosticKey::TimeSamples, refusal->timeSamples},
                {DiagnosticKey::Limit, refusal->limit}},
            {}, severity);
    else if (grpcPresent && refusal && refusal->reason == 2)
        emit(code, endpoint, subject, 0,
            {{DiagnosticKey::Operation, operation}, {DiagnosticKey::Result, result},
                {DiagnosticKey::DurationMs, durationMs}, {DiagnosticKey::Request, request},
                {DiagnosticKey::Attempt, attempt}, {DiagnosticKey::GrpcStatus, grpcCode},
                {DiagnosticKey::Generation, streamGeneration}, {DiagnosticKey::AuthGeneration, authGeneration},
                {DiagnosticKey::Work, work}, {DiagnosticKey::Session, session}, {DiagnosticKey::Reason, 2},
                {DiagnosticKey::Limit, refusal->limit}, {DiagnosticKey::Count, refusal->actual}},
            {}, severity);
    else if (grpcPresent)
        emit(code, endpoint, subject, 0,
            {{DiagnosticKey::Operation, operation}, {DiagnosticKey::Result, result},
                {DiagnosticKey::DurationMs, durationMs}, {DiagnosticKey::Request, request},
                {DiagnosticKey::Attempt, attempt}, {DiagnosticKey::GrpcStatus, grpcCode},
                {DiagnosticKey::Generation, streamGeneration}, {DiagnosticKey::AuthGeneration, authGeneration},
                {DiagnosticKey::Work, work}, {DiagnosticKey::Session, session}},
            {}, severity);
    else
        emit(code, endpoint, subject, 0,
            {{DiagnosticKey::Operation, operation}, {DiagnosticKey::Result, result},
                {DiagnosticKey::DurationMs, durationMs}, {DiagnosticKey::Request, request},
                {DiagnosticKey::Attempt, attempt}, {DiagnosticKey::Generation, streamGeneration},
                {DiagnosticKey::AuthGeneration, authGeneration}, {DiagnosticKey::Work, work},
                {DiagnosticKey::Session, session}},
            {}, severity);
}
void DiagnosticsRegistry::retainClosedIncident(OmniUsdLiveEndpoint endpoint, std::string_view resource,
    const OmniUsdLiveStatus& status) noexcept {
    if (!status.incident_id) return;
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->stopping) return;
    if (impl_->incidents.empty()) {
        increment(impl_->metrics.closedIncidentsEvicted);
        return;
    }
    size_t index = impl_->incidents.size();
    for (size_t i = 0; i < impl_->incidentCount; ++i) {
        const auto current = (impl_->incidentHead + i) % impl_->incidents.size();
        if (impl_->incidents[current].status.subject_id == status.subject_id) {
            index = current;
            break;
        }
    }
    if (index == impl_->incidents.size()) {
        if (impl_->incidentCount == impl_->incidents.size()) {
            impl_->incidentHead = (impl_->incidentHead + 1) % impl_->incidents.size();
            --impl_->incidentCount;
            increment(impl_->metrics.closedIncidentsEvicted);
        }
        index = (impl_->incidentHead + impl_->incidentCount++) % impl_->incidents.size();
    }
    auto& out = impl_->incidents[index];
    out = {};
    out.endpoint = endpoint;
    out.status = status;
    out.status.struct_size = sizeof(out.status);
    out.status.abi_version = OMNI_USD_LIVE_ABI_VERSION;
    if ((impl_->flags & OMNI_USD_LIVE_DIAGNOSTIC_RESOURCE_NAMES) && resourceValid(resource) && !resource.empty())
        std::memcpy(out.resource.data(), resource.data(), resource.size());
}
DiagnosticSnapshot DiagnosticsRegistry::snapshot(OmniUsdLiveEndpoint endpoint, uint32_t maxRecords, uint32_t flags,
    uint32_t maxIncidents) const {
    DiagnosticSnapshot out;
    out.diagnosticExecutor = impl_->executor.snapshot();
    std::lock_guard<std::mutex> lock(impl_->mutex);
    out.runtimeInstanceId = impl_->instance;
    out.captureTimeMs = wallMs();
    out.metrics = impl_->metrics;
    out.lastSequence = impl_->sequence;
    if (impl_->historyCount) out.firstSequence = impl_->history[impl_->historyHead].sequence;
    const bool names = (flags & impl_->flags & OMNI_USD_LIVE_DIAGNOSTIC_RESOURCE_NAMES) != 0;
    size_t matching = 0;
    for (size_t i = 0; i < impl_->historyCount; ++i) {
        const auto& entry = impl_->history[(impl_->historyHead + i) % impl_->history.size()];
        if (!endpoint || endpoint == entry.endpoint) ++matching;
    }
    const auto keep = std::min<size_t>(matching, maxRecords);
    out.records.reserve(keep);
    size_t skip = matching - keep;
    out.truncated = skip != 0;
    for (size_t i = 0; i < impl_->historyCount; ++i) {
        const auto& entry = impl_->history[(impl_->historyHead + i) % impl_->history.size()];
        if (endpoint && endpoint != entry.endpoint) continue;
        if (skip) {
            --skip;
            continue;
        }
        out.records.push_back(entry);
        if (!names) out.records.back().resource = {};
    }
    size_t matchingIncidents = 0;
    for (size_t i = 0; i < impl_->incidentCount; ++i) {
        const auto& incident = impl_->incidents[(impl_->incidentHead + i) % impl_->incidents.size()];
        if (!endpoint || endpoint == incident.endpoint) ++matchingIncidents;
    }
    const auto keepIncidents = std::min<size_t>(matchingIncidents, maxIncidents);
    size_t skipIncidents = matchingIncidents - keepIncidents;
    out.truncated = out.truncated || skipIncidents != 0;
    out.closedIncidents.reserve(keepIncidents);
    for (size_t i = 0; i < impl_->incidentCount; ++i) {
        const auto& incident = impl_->incidents[(impl_->incidentHead + i) % impl_->incidents.size()];
        if (endpoint && endpoint != incident.endpoint) continue;
        if (skipIncidents) {
            --skipIncidents;
            continue;
        }
        out.closedIncidents.push_back(incident);
        if (!names) out.closedIncidents.back().resource = {};
    }
    return out;
}
OmniUsdLiveCallbackHandle DiagnosticsRegistry::subscribe(const OmniUsdLiveDiagnosticOptions& options, void* user,
    OmniUsdLiveDiagnosticCallback callback, std::shared_ptr<void> lease) {
    if (!callback || options.min_severity > OMNI_USD_LIVE_SEVERITY_ERROR || (options.flags & ~allowedFlags) ||
        options.event_code_count >= kDiagnosticEventCount || (options.event_code_count && !options.event_codes))
        throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
    std::array<bool, kDiagnosticEventCount> events{};
    if (!options.event_code_count) events.fill(true);
    else
        for (uint32_t i = 0; i < options.event_code_count; ++i) {
            const auto code = options.event_codes[i];
            if (!code || code >= events.size() || code == kReservedEventCode)
                throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
            events[code] = true;
        }
    // Reserve all fallible storage outside the registry lock. On any admission
    // failure final lease/host owner destruction also happens outside that lock.
    auto sink = std::make_shared<Impl::Sink>(*impl_, allocateHandle(), options, events, user, callback,
        std::move(lease));
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->stopping) throw OmniUsdLiveResult(OMNI_USD_LIVE_BUSY);
    if (!impl_->sinkCapacity) throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    for (auto& slot : impl_->sinks)
        if (!slot) {
            slot = sink;
            return sink->handle;
        }
    throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
}
OmniUsdLiveResult DiagnosticsRegistry::unsubscribe(OmniUsdLiveCallbackHandle handle, uint32_t timeoutMs) {
    std::shared_ptr<Impl::Sink> retained;
    const auto deadline = Clock::now() + std::chrono::milliseconds(timeoutMs);
    std::unique_lock<std::mutex> lock(impl_->mutex);
    size_t index = impl_->sinks.size();
    for (size_t i = 0; i < impl_->sinks.size(); ++i)
        if (impl_->sinks[i] && impl_->sinks[i]->handle == handle) {
            index = i;
            break;
        }
    if (index == impl_->sinks.size()) return OMNI_USD_LIVE_INVALID_HANDLE;
    retained = impl_->sinks[index];
    retained->cancelled = true;
    retained->schedule();
    if (!retained->quiescent) {
        if (DispatchExecutor::isWorker()) return OMNI_USD_LIVE_BUSY;
        if (!impl_->changed.wait_until(lock, deadline, [&] { return retained->quiescent; }))
            return OMNI_USD_LIVE_TIMEOUT;
    }
    if (impl_->sinks[index] != retained) return OMNI_USD_LIVE_INVALID_HANDLE;
    impl_->sinks[index].reset(); // retained owner lives beyond the lock.
    return OMNI_USD_LIVE_OK;
}
bool DiagnosticsRegistry::hasSinks() const noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    for (const auto& sink : impl_->sinks)
        if (sink) return true;
    return false;
}
void DiagnosticsRegistry::beginShutdown() noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->stopping = true;
    for (const auto& sink : impl_->sinks)
        if (sink) {
            sink->cancelled = true;
            sink->schedule();
        }
}
bool DiagnosticsRegistry::waitForShutdown(Clock::time_point deadline) {
    std::unique_lock<std::mutex> lock(impl_->mutex);
    if (!impl_->stopping) return false;
    if (!impl_->changed.wait_until(lock, deadline, [&] {
        for (const auto& sink : impl_->sinks)
            if (sink && !sink->quiescent) return false;
        return true;
    }))
        return false;
    lock.unlock();
    impl_->executor.beginShutdown();
    return impl_->executor.waitForShutdown(deadline);
}
} // namespace omni::usd::live
