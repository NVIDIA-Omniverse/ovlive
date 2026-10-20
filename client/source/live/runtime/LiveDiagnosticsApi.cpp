// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "LiveDiagnostics.h"
#include "LiveBuildProvenance.h"
#include "LiveRuntime.h"
#include "LiveRequest.h"
#include "LiveStatus.h"
#include <algorithm>
#include <charconv>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace {
using namespace omni::usd::live;
constexpr uint64_t exportLimit = 16 * 1024 * 1024;
constexpr uint32_t knownFlags = OMNI_USD_LIVE_DIAGNOSTIC_RESOURCE_NAMES | OMNI_USD_LIVE_DIAGNOSTIC_SERVER_DETAIL;
struct OutputFull {};

// Every append checks the cap first. Construct the entire bounded document
// locally before touching caller memory; required size includes its terminator.
class Json {
public:
    explicit Json(size_t limit) : limit_(limit ? limit - 1 : 0) {}
    void raw(std::string_view value) {
        if (value.size() > limit_ - text.size()) throw OutputFull{};
        text.append(value);
    }
    void number(uint64_t value) {
        char bytes[32];
        const auto end = std::to_chars(bytes, bytes + sizeof(bytes), value);
        raw({bytes, static_cast<size_t>(end.ptr - bytes)});
    }
    void string(std::string_view value) {
        raw("\"");
        constexpr char hex[] = "0123456789abcdef";
        for (unsigned char c : value) {
            if (c == '"' || c == '\\') {
                char escaped[] = {'\\', static_cast<char>(c)};
                raw({escaped, 2});
            } else if (c < 32 || c >= 127) {
                // Resource identities are canonical ASCII URI strings. Escaping
                // other bytes keeps hostile diagnostic test input valid JSON.
                char escaped[] = {'\\', 'u', '0', '0', hex[c >> 4], hex[c & 15]};
                raw({escaped, 6});
            } else {
                char byte = static_cast<char>(c);
                raw({&byte, 1});
            }
        }
        raw("\"");
    }
    void key(std::string_view name) {
        string(name);
        raw(":");
    }
    void field(std::string_view name, uint64_t value) {
        raw(",");
        key(name);
        number(value);
    }
    void named(std::string_view name, std::string_view value) {
        raw(",");
        key(name);
        string(value);
    }
    std::string text;
private:
    size_t limit_;
};

template<class T> T optionsCopy(const T* input, T defaults) {
    if (!input) return defaults;
    uint32_t size;
    std::memcpy(&size, input, sizeof(size));
    if (size < sizeof(T)) throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
    T value;
    std::memcpy(&value, input, sizeof(value));
    if ((value.abi_version >> 16) != (OMNI_USD_LIVE_ABI_VERSION >> 16))
        throw OmniUsdLiveResult(OMNI_USD_LIVE_UNSUPPORTED);
    return value;
}
template<class Action> OmniUsdLiveResult boundary(Action&& action) noexcept {
    try {
        return action();
    } catch (OmniUsdLiveResult result) { return result; } catch (const OutputFull&) {
        return OMNI_USD_LIVE_RESOURCE_EXHAUSTED;
    } catch (const std::bad_alloc&) { return OMNI_USD_LIVE_RESOURCE_EXHAUSTED; } catch (const std::length_error&) {
        return OMNI_USD_LIVE_RESOURCE_EXHAUSTED;
    } catch (...) { return OMNI_USD_LIVE_INTERNAL_ERROR; }
}

void statusJson(Json& out, OmniUsdLiveEndpoint endpoint, OmniUsdLiveSession session, const OmniUsdLiveStatus& status,
    std::string_view resource) {
    out.raw("{\"endpoint\":");
    out.number(endpoint);
    out.field("session", session);
#define STATUS_FIELD(name) out.field(#name, status.name)
    STATUS_FIELD(subject_kind);
    STATUS_FIELD(state);
    STATUS_FIELD(subject_id);
    STATUS_FIELD(revision);
    STATUS_FIELD(changed_at_ms);
    STATUS_FIELD(result);
    STATUS_FIELD(operation);
    STATUS_FIELD(grpc_status_present);
    if (status.grpc_status_present) out.field("grpc_status", static_cast<uint32_t>(status.grpc_status));
    STATUS_FIELD(recovery_action);
    STATUS_FIELD(progress_known);
    STATUS_FIELD(incident_id);
    STATUS_FIELD(pending_write_count);
    STATUS_FIELD(pending_bytes);
    STATUS_FIELD(has_unconfirmed_edits);
    STATUS_FIELD(blocked_layers);
    STATUS_FIELD(failed_layers);
    STATUS_FIELD(blocked_sessions);
    STATUS_FIELD(failed_sessions);
    STATUS_FIELD(read_sequence);
    STATUS_FIELD(applied_sequence);
#undef STATUS_FIELD
    if (!resource.empty()) out.named("canonical_resource", resource);
    out.raw("}");
}
void durationBounds(Json& out) {
    out.raw("{\"bounds\":[");
    for (size_t i = 0; i < kMetricDurationBoundsMs.size(); ++i) {
        if (i) out.raw(",");
        out.number(kMetricDurationBoundsMs[i]);
    }
    out.raw("]");
}
void durationCounts(Json& out, const MetricDuration& metric) {
    out.field("count", metric.count);
    out.field("sum_ms", metric.sumMs);
    out.raw(",\"counts\":[");
    for (size_t i = 0; i < metric.counts.size(); ++i) {
        if (i) out.raw(",");
        out.number(metric.counts[i]);
    }
    out.raw("]");
}
void resultDurations(Json& out, const std::array<MetricDuration, 23>& metrics) {
    durationBounds(out);
    out.raw(",\"entries\":[");
    bool comma = false;
    for (size_t result = 0; result < metrics.size(); ++result) {
        if (!metrics[result].count) continue;
        if (comma) out.raw(",");
        comma = true;
        out.raw("{\"result\":");
        out.number(result);
        durationCounts(out, metrics[result]);
        out.raw("}");
    }
    out.raw("]}");
}
template<size_t N> void resolutionDurations(Json& out, const std::array<MetricDuration, N>& metrics) {
    durationBounds(out);
    out.raw(",\"entries\":[");
    bool comma = false;
    for (size_t resolution = 0; resolution < metrics.size(); ++resolution) {
        if (!metrics[resolution].count) continue;
        if (comma) out.raw(",");
        comma = true;
        out.raw("{\"resolution\":");
        out.number(resolution);
        durationCounts(out, metrics[resolution]);
        out.raw("}");
    }
    out.raw("]}");
}
void dispatchJson(Json& out, const DispatchMetrics& metrics) {
    out.raw("{\"enqueued\":");
    out.number(metrics.enqueued);
    out.field("started", metrics.started);
    out.field("completed", metrics.completed);
    out.field("queued", metrics.queued);
    out.field("running", metrics.running);
    out.field("queue_high_water", metrics.queueHighWater);
    out.field("configured_workers", metrics.configuredWorkers);
    out.field("live_workers", metrics.liveWorkers);
    out.field("oldest_queued_age_ms", metrics.oldestQueuedAgeMs);
    out.field("oldest_running_age_ms", metrics.oldestRunningAgeMs);
    out.raw(",\"queue_wait_ms\":");
    durationBounds(out);
    durationCounts(out, metrics.queueWait);
    out.raw("}");
    out.raw(",\"dispatch_duration_ms\":");
    durationBounds(out);
    durationCounts(out, metrics.dispatchDuration);
    out.raw("}}");
}
void slotJson(Json& out, const DiagnosticSlotMetrics& slots) {
    out.raw("{\"current\":");
    out.number(slots.current);
    out.field("high_water", slots.highWater);
    out.field("admitted", slots.admitted);
    out.field("released", slots.released);
    out.field("rejected", slots.rejected);
    out.raw("}");
}
void capacityJson(Json& out, const DiagnosticMetrics& metrics) {
    out.raw("{\"request_slots\":{\"ordinary\":");
    slotJson(out, metrics.requestSlots[0]);
    out.raw(",\"timing\":");
    slotJson(out, metrics.requestSlots[1]);
    const auto& bytes = metrics.pendingBytes;
    out.raw("},\"pending_bytes\":{\"reservations\":");
    slotJson(out, bytes.reservations);
    out.field("current_bytes", bytes.currentBytes.high ? UINT64_MAX : bytes.currentBytes.low);
    out.raw(",\"current_bytes_overflow\":");
    out.raw(bytes.currentBytes.high ? "true" : "false");
    out.field("high_water_bytes", bytes.highWaterBytes.high ? UINT64_MAX : bytes.highWaterBytes.low);
    out.raw(",\"high_water_bytes_overflow\":");
    out.raw(bytes.highWaterBytes.high ? "true" : "false");
    out.field("admitted_bytes", bytes.admittedBytes);
    out.field("released_bytes", bytes.releasedBytes);
    out.field("rejected_bytes", bytes.rejectedBytes);
    out.raw("}}");
}
void metricsJson(Json& out, const DiagnosticMetrics& metrics, const DispatchMetrics& operational,
    const DispatchMetrics& diagnostic) {
    out.raw("{\"scope\":\"runtime\"");
#define METRIC(name, member) out.field(name, metrics.member)
    METRIC("emitted", emitted);
    METRIC("suppressed", suppressed);
    METRIC("warning_suppressed", warningSuppressed);
    METRIC("invalid", invalid);
    METRIC("history_overwritten", historyOverwritten);
    METRIC("history_dropped", historyDropped);
    METRIC("sink_dropped", sinkDropped);
    METRIC("sink_failures", sinkFailures);
    METRIC("sinks_disabled", sinksDisabled);
    METRIC("closed_incidents_evicted", closedIncidentsEvicted);
    METRIC("rpc_attempts", rpcAttempts);
    METRIC("rpc_completions", rpcCompletions);
    METRIC("rpc_in_flight", rpcInFlight);
#undef METRIC
    out.raw(",\"rpc_by_operation_result\":[");
    bool comma = false;
    for (size_t op = 0; op < metrics.rpc.size(); ++op) {
        for (size_t result = 0; result < metrics.rpc[op].size(); ++result) {
            if (!metrics.rpc[op][result]) continue;
            if (comma) out.raw(",");
            comma = true;
            out.raw("{\"operation\":");
            out.number(op);
            out.field("result", result);
            out.field("count", metrics.rpc[op][result]);
            out.raw("}");
        }
    }
    out.raw("],\"rpc_attempts_by_operation\":[");
    comma = false;
    for (size_t op = 0; op < metrics.rpcAttemptsByOperation.size(); ++op) {
        if (!metrics.rpcAttemptsByOperation[op]) continue;
        if (comma) out.raw(",");
        comma = true;
        out.raw("{\"operation\":");
        out.number(op);
        out.field("count", metrics.rpcAttemptsByOperation[op]);
        out.raw("}");
    }
    out.raw("],\"events\":[");
    comma = false;
    for (size_t event = 0; event < metrics.events.size(); ++event) {
        if (!metrics.events[event]) continue;
        if (comma) out.raw(",");
        comma = true;
        out.raw("{\"event_code\":");
        out.number(event);
        out.field("count", metrics.events[event]);
        out.raw("}");
    }
    out.raw("],\"messages_by_event\":[");
    comma = false;
    for (size_t i = 0; i < metrics.messageBytes.size(); ++i) {
        const auto code = OMNI_USD_LIVE_RECORD_MESSAGE_ACCEPTED + i;
        if (!metrics.events[code]) continue;
        if (comma) out.raw(",");
        comma = true;
        out.raw("{\"event_code\":");
        out.number(code);
        out.field("count", metrics.events[code]);
        out.field("bytes", metrics.messageBytes[i]);
        out.raw("}");
    }
    out.raw("],\"rpc_duration_ms\":");
    durationBounds(out);
    out.raw(",\"counts\":[");
    for (size_t i = 0; i < metrics.rpcDuration.size(); ++i) {
        if (i) out.raw(",");
        out.number(metrics.rpcDuration[i]);
    }
    out.raw("]},\"rpc_duration_ms_by_operation_result\":");
    durationBounds(out);
    out.raw(",\"entries\":[");
    comma = false;
    for (size_t op = 0; op < metrics.rpcDurationByOperationResult.size(); ++op) {
        for (size_t result = 0; result < metrics.rpcDurationByOperationResult[op].size(); ++result) {
            const auto& metric = metrics.rpcDurationByOperationResult[op][result];
            if (!metric.count) continue;
            if (comma) out.raw(",");
            comma = true;
            out.raw("{\"operation\":");
            out.number(op);
            out.field("result", result);
            durationCounts(out, metric);
            out.raw("}");
        }
    }
    out.raw("]},\"save_duration_ms_by_result\":");
    resultDurations(out, metrics.saveDurationByResult);
    out.raw(",\"drain_duration_ms_by_result\":");
    resultDurations(out, metrics.drainDurationByResult);
    out.raw(",\"shutdown_duration_ms_by_phase_result\":");
    durationBounds(out);
    out.raw(",\"entries\":[");
    comma = false;
    for (size_t phase = 0; phase < metrics.shutdownDurationByPhaseResult.size(); ++phase) {
        for (size_t result = 0; result < metrics.shutdownDurationByPhaseResult[phase].size(); ++result) {
            const auto& metric = metrics.shutdownDurationByPhaseResult[phase][result];
            if (!metric.count) continue;
            if (comma) out.raw(",");
            comma = true;
            out.raw("{\"phase\":");
            out.number(phase);
            out.field("result", result);
            durationCounts(out, metric);
            out.raw("}");
        }
    }
    out.raw("]}");
    out.raw(",\"auth\":{\"blocked_endpoints\":");
    out.number(metrics.authBlockedEndpoints);
    out.field("blocked_endpoints_high_water", metrics.authBlockedEndpointsHighWater);
    out.field("stale_replacements", metrics.authStaleReplacements);
    out.raw(",\"challenges_by_reason\":[");
    comma = false;
    for (size_t reason = 0; reason < metrics.authChallengesByReason.size(); ++reason) {
        if (!metrics.authChallengesByReason[reason]) continue;
        if (comma) out.raw(",");
        comma = true;
        out.raw("{\"reason\":");
        out.number(reason);
        out.field("count", metrics.authChallengesByReason[reason]);
        out.raw("}");
    }
    out.raw("],\"failures_by_cause\":[");
    comma = false;
    for (size_t cause = 0; cause < metrics.authFailuresByCause.size(); ++cause) {
        if (!metrics.authFailuresByCause[cause]) continue;
        if (comma) out.raw(",");
        comma = true;
        out.raw("{\"cause\":");
        out.number(cause);
        out.field("count", metrics.authFailuresByCause[cause]);
        out.raw("}");
    }
    out.raw("],\"credential_wait_ms_by_resolution\":");
    durationBounds(out);
    out.raw(",\"entries\":[");
    comma = false;
    for (size_t resolution = 0; resolution < metrics.authWaitByResolution.size(); ++resolution) {
        const auto& metric = metrics.authWaitByResolution[resolution];
        if (!metric.count) continue;
        if (comma) out.raw(",");
        comma = true;
        out.raw("{\"resolution\":");
        out.number(resolution);
        durationCounts(out, metric);
        out.raw("}");
    }
    out.raw("]}},\"operation_auth_waits\":[");
    for (size_t kind = 0; kind < metrics.operationAuthWaits.size(); ++kind) {
        if (kind) out.raw(",");
        const auto& metric = metrics.operationAuthWaits[kind];
        out.raw("{\"kind\":");
        out.number(kind);
        out.field("current", metric.current);
        out.field("high_water", metric.highWater);
        out.field("started", metric.started);
        out.field("completed", metric.completed);
        out.raw(",\"duration_ms_by_result\":");
        resultDurations(out, metric.durationByResult);
        out.raw("}");
    }
    out.raw("],\"subjects\":[");
    for (size_t kind = 0; kind < metrics.subjects.size(); ++kind) {
        if (kind) out.raw(",");
        const auto& metric = metrics.subjects[kind];
        out.raw("{\"kind\":");
        out.number(kind + 1);
        out.field("current", metric.current);
        out.field("high_water", metric.highWater);
        out.field("admitted", metric.admitted);
        out.field("detached", metric.detached);
        out.raw(",\"states\":[");
        for (size_t state = 1; state < metric.byState.size(); ++state) {
            if (state != 1) out.raw(",");
            out.raw("{\"state\":");
            out.number(state);
            out.field("current", metric.byState[state]);
            out.raw("}");
        }
        out.raw("]}");
    }
    out.raw("],\"readiness\":[");
    for (size_t kind = 0; kind < metrics.readiness.size(); ++kind) {
        if (kind) out.raw(",");
        const auto& metric = metrics.readiness[kind];
        out.raw("{\"kind\":");
        out.number(kind + OMNI_USD_LIVE_SUBJECT_SESSION);
        out.field("initial_waiting", metric.initialWaiting);
        out.field("recovering", metric.recovering);
        out.raw(",\"initial_ms_by_resolution\":");
        resolutionDurations(out, metric.initialByResolution);
        out.raw(",\"recovery_ms_by_resolution\":");
        resolutionDurations(out, metric.recoveryByResolution);
        out.raw("}");
    }
    out.raw("],\"backpressure\":[");
    for (size_t kind = 0; kind < metrics.backpressure.size(); ++kind) {
        if (kind) out.raw(",");
        const auto& metric = metrics.backpressure[kind];
        out.raw("{\"kind\":");
        out.number(kind + 1);
        out.field("current", metric.current);
        out.field("high_water", metric.highWater);
        out.raw(",\"duration_ms_by_resolution\":");
        resolutionDurations(out, metric.durationByResolution);
        out.raw("}");
    }
    out.raw("],\"applications\":[");
    for (size_t kind = 0; kind < metrics.applications.size(); ++kind) {
        if (kind) out.raw(",");
        const auto& metric = metrics.applications[kind];
        out.raw("{\"kind\":");
        out.number(kind);
        out.field("current", metric.current);
        out.field("high_water", metric.highWater);
        out.field("admitted", metric.admitted);
        out.field("retired", metric.retired);
        out.raw(",\"queue_ms\":");
        durationBounds(out);
        durationCounts(out, metric.queue);
        out.raw("}");
        out.raw(",\"body_ms\":");
        durationBounds(out);
        durationCounts(out, metric.body);
        out.raw("}");
        out.raw(",\"post_body_retention_ms\":");
        durationBounds(out);
        durationCounts(out, metric.postBodyRetention);
        out.raw("}");
        out.raw(",\"lifetime_ms_by_result\":");
        resultDurations(out, metric.lifetimeByResult);
        out.raw("}");
    }
    out.raw("],\"jitter\":{\"current\":");
    out.number(metrics.jitter.current);
    out.field("high_water", metrics.jitter.highWater);
    out.field("started", metrics.jitter.started);
    out.field("completed", metrics.jitter.completed);
    out.field("frozen_generation_starts", metrics.jitter.frozenStarts);
    out.field("unsampled_starts", metrics.jitter.unsampledStarts);
    out.raw(",\"requested_wait_ms\":");
    durationBounds(out);
    durationCounts(out, metrics.jitter.requestedWait);
    out.raw("}");
    out.raw(",\"target_hold_ms\":");
    durationBounds(out);
    durationCounts(out, metrics.jitter.targetHold);
    out.raw("}");
    out.raw(",\"sample_age_ms\":");
    durationBounds(out);
    durationCounts(out, metrics.jitter.sampleAge);
    out.raw("}");
    out.raw(",\"observed_hold_ms_by_resolution\":");
    resolutionDurations(out, metrics.jitter.holdByResolution);
    out.raw(",\"post_due_observation_ms_by_resolution\":");
    resolutionDurations(out, metrics.jitter.postDueByResolution);
    out.raw("},\"apply_lag\":{\"current\":");
    out.number(metrics.applyLag.current);
    out.field("high_water", metrics.applyLag.highWater);
    out.field("receive_to_apply_max_ms", metrics.applyLag.receiveToApplyMaxMs);
    out.raw(",\"receive_to_apply_ms\":");
    durationBounds(out);
    durationCounts(out, metrics.applyLag.receiveToApply);
    out.raw("}");
    out.raw("},\"layer_writes\":{\"in_flight\":");
    out.number(metrics.layerWrites.inFlight);
    out.field("in_flight_high_water", metrics.layerWrites.inFlightHighWater);
    out.field("window_high_water", metrics.layerWrites.windowHighWater);
    out.field("window_grown", metrics.layerWrites.windowGrown);
    out.field("window_halved", metrics.layerWrites.windowHalved);
    out.field("chain_resends", metrics.layerWrites.chainResends);
    out.field("resent_writes", metrics.layerWrites.resentWrites);
    out.field("admitted_updates", metrics.layerWrites.admittedUpdates);
    out.field("coalesced_turns", metrics.layerWrites.coalescedTurns);
    out.field("max_turns_per_update", metrics.layerWrites.maxTurnsPerUpdate);
    out.raw("},\"capacity\":");
    capacityJson(out, metrics);
    out.raw(",\"cq_tags\":{\"submitted\":");
    out.number(metrics.cqTags.submitted);
    out.field("received_ok", metrics.cqTags.receivedOk);
    out.field("received_not_ok", metrics.cqTags.receivedNotOk);
    out.field("retired", metrics.cqTags.retired);
    out.field("abandoned", metrics.cqTags.abandoned);
    out.field("outstanding", metrics.cqTags.outstanding);
    out.field("outstanding_high_water", metrics.cqTags.outstandingHighWater);
    out.field("handling", metrics.cqTags.handling);
    out.field("retained", metrics.cqTags.retained);
    out.field("retained_high_water", metrics.cqTags.retainedHighWater);
    out.raw("},\"executors\":{\"operational\":");
    dispatchJson(out, operational);
    out.raw(",\"diagnostic\":");
    dispatchJson(out, diagnostic);
    out.raw("}}");
}
std::string document(const Runtime& runtime, const DiagnosticSnapshot& snapshot,
    const std::vector<StatusSnapshot>& statuses, size_t offset, size_t cap, OmniUsdLiveEndpoint filter,
    const DispatchMetrics& operational) {
    Json out(cap);
    out.raw("{\"schema_version\":");
    out.number(OMNI_USD_LIVE_DIAGNOSTIC_SCHEMA_VERSION);
    out.field("runtime_instance_id", snapshot.runtimeInstanceId);
    out.field("capture_time_ms", snapshot.captureTimeMs);
    out.field("endpoint_filter", filter);
    out.raw(",\"provenance\":{\"abi_version\":");
    out.number(OMNI_USD_LIVE_ABI_VERSION);
    out.named("proto", "nvidia/omniverse/live/v1alpha/live.proto");
    out.named("proto_sha256", LIVE_PROTO_SHA256);
    out.named("source_revision", LIVE_BUILD_REVISION);
    out.named("source_sha256", LIVE_BUILD_SOURCE_SHA256);
    out.named("compiler", LIVE_BUILD_COMPILER);
    out.named("artifact_scope", "native_runtime_component");
    out.raw("},\"effective_configuration\":{\"diagnostic_export_limit_bytes\":");
    out.number(exportLimit);
#define OPTION(name, unused) out.field(#name, runtime.options.values.name);
    LIVE_OPTION_DEFAULTS(OPTION)
#undef OPTION
    out.field("diagnostic_min_severity", runtime.options.values.diagnostic_min_severity);
    out.field("diagnostic_allowed_flags", runtime.options.values.diagnostic_allowed_flags);
    out.field("flags", runtime.options.values.flags);
    out.raw("},\"statuses\":[");
    for (size_t i = 0; i < statuses.size(); ++i) {
        if (i) out.raw(",");
        const auto& value = statuses[i];
        statusJson(out, value.endpoint, value.session, value.status, value.uri);
    }
    out.raw("],\"metrics\":");
    metricsJson(out, snapshot.metrics, operational, snapshot.diagnosticExecutor);
    out.raw(",\"closed_incidents\":[");
    for (size_t i = 0; i < snapshot.closedIncidents.size(); ++i) {
        if (i) out.raw(",");
        const auto& value = snapshot.closedIncidents[i];
        statusJson(out, value.endpoint, 0, value.status, value.resource.data());
    }
    out.raw("],\"history\":{\"first_sequence\":");
    out.number(snapshot.firstSequence);
    out.field("last_sequence", snapshot.lastSequence);
    out.raw(",\"truncated\":");
    out.raw(snapshot.truncated || offset ? "true" : "false");
    out.raw("},\"recent_records\":[");
    for (size_t i = offset; i < snapshot.records.size(); ++i) {
        if (i != offset) out.raw(",");
        const auto& value = snapshot.records[i];
        out.raw("{\"event_code\":");
        out.number(value.eventCode);
        out.named("event_name", DiagnosticsRegistry::eventName(value.eventCode));
        out.field("severity", value.severity);
        out.field("timestamp_ms", value.timestampMs);
        out.field("elapsed_ms", value.elapsedMs);
        out.field("runtime_instance_id", value.runtimeInstanceId);
        out.field("sequence", value.sequence);
        out.field("endpoint", value.endpoint);
        out.field("subject_id", value.subjectId);
        out.field("incident_id", value.incidentId);
        out.raw(",\"fields\":{");
        for (size_t f = 0; f < value.fieldCount; ++f) {
            if (f) out.raw(",");
            out.key(DiagnosticsRegistry::fieldName(value.fields[f].key));
            out.number(value.fields[f].value);
        }
        out.raw("}");
        if (value.resource[0]) out.named("canonical_resource", value.resource.data());
        out.raw("}");
    }
    out.raw("]}");
    return std::move(out.text);
}
}

extern "C" OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveRegisterDiagnosticCallback(
    const OmniUsdLiveDiagnosticOptions* input, void* user, OmniUsdLiveDiagnosticCallback callback,
    OmniUsdLiveCallbackHandle* output) {
    if (output) *output = 0;
    return boundary([&]() -> OmniUsdLiveResult {
        if (!callback || !output) return OMNI_USD_LIVE_INVALID_ARGUMENT;
        OmniUsdLiveDiagnosticOptions defaults{};
        defaults.struct_size = sizeof(defaults);
        defaults.abi_version = OMNI_USD_LIVE_ABI_VERSION;
        defaults.min_severity = OMNI_USD_LIVE_SEVERITY_INFO;
        auto options = optionsCopy(input, defaults);
        auto runtime = getRuntime();
        if (!runtime) return OMNI_USD_LIVE_NOT_INITIALIZED;
        std::shared_ptr<void> lease;
        if (options.endpoint) {
            auto dependency = runtime->acquireEndpoint(options.endpoint);
            if (!dependency) return runtime->closing ? OMNI_USD_LIVE_BUSY : OMNI_USD_LIVE_INVALID_HANDLE;
            lease = std::make_shared<EndpointLease>(std::move(dependency));
        }
        std::lock_guard<std::mutex> lock(runtime->mutex);
        if (runtime->closing) return OMNI_USD_LIVE_BUSY;
        *output = runtime->diagnostics->subscribe(options, user, callback, lease);
        return OMNI_USD_LIVE_OK;
    });
}

extern "C" OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveCopyDiagnostics(OmniUsdLiveEndpoint endpoint,
    const OmniUsdLiveDiagnosticExportOptions* input, char* buffer, uint64_t* size) {
    return boundary([&]() -> OmniUsdLiveResult {
        if (!size) return OMNI_USD_LIVE_INVALID_ARGUMENT;
        auto runtime = getRuntime();
        if (!runtime) return OMNI_USD_LIVE_NOT_INITIALIZED;
        OmniUsdLiveDiagnosticExportOptions defaults{};
        defaults.struct_size = sizeof(defaults);
        defaults.abi_version = OMNI_USD_LIVE_ABI_VERSION;
        defaults.max_records = runtime->options.values.max_diagnostic_records;
        auto options = optionsCopy(input, defaults);
        if (options.flags & ~knownFlags) return OMNI_USD_LIVE_INVALID_ARGUMENT;
        EndpointLease lease;
        if (endpoint) {
            lease = runtime->acquireEndpoint(endpoint);
            if (!lease) return OMNI_USD_LIVE_INVALID_HANDLE;
        }
        const auto cap = static_cast<size_t>(
            std::min(exportLimit, options.max_output_bytes ? options.max_output_bytes : exportLimit));
        const bool names = options.flags & runtime->options.values.diagnostic_allowed_flags &
            OMNI_USD_LIVE_DIAGNOSTIC_RESOURCE_NAMES;
        auto statuses = runtime->statuses->capture(endpoint, names, cap);
        auto snapshot = runtime->diagnostics->snapshot(endpoint,
            std::min(options.max_records, static_cast<uint32_t>(cap / sizeof(DiagnosticEntry))), options.flags,
            static_cast<uint32_t>(cap / sizeof(DiagnosticIncident)));
        const auto operational = runtime->requests->executor().snapshot();
        // Captures own bounded copies; no locks held during serialization.
        // Histories may change between sizing and copying calls; callers retry.
        size_t offset = 0;
        std::string json;
        for (;;) {
            try {
                json = document(*runtime, snapshot, statuses, offset, cap, endpoint, operational);
                break;
            } catch (const OutputFull&) {
                if (offset == snapshot.records.size()) throw;
                offset += (snapshot.records.size() - offset + 1) / 2;
            }
        }
        const auto capacity = buffer ? *size : 0;
        *size = json.size() + 1;
        if (!buffer) return OMNI_USD_LIVE_OK;
        if (capacity < *size) return OMNI_USD_LIVE_BUFFER_TOO_SMALL;
        std::memcpy(buffer, json.c_str(), static_cast<size_t>(*size));
        return OMNI_USD_LIVE_OK;
    });
}
