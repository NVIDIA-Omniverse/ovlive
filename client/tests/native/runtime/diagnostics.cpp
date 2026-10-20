// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "LiveDiagnostics.h"
#include "LiveDispatch.h"
#include "LiveRuntime.h"
#include <array>
#include <atomic>
#include <cassert>
#include <functional>
#include <future>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>

using namespace omni::usd::live;
namespace {
#include "LiveDiagnosticCatalogContract.h"
using Clock = DiagnosticsRegistry::Clock;
constexpr uint64_t instance = 998877;
constexpr OmniUsdLiveEndpoint endpoint = 11, otherEndpoint = 22;
constexpr uint32_t eventCode = OMNI_USD_LIVE_RECORD_ENDPOINT_REGISTERED;
const std::string resource = "live://diagnostics.example/session/root.live";
template <class T> T record()
{
    T result{}; result.struct_size = sizeof(result); result.abi_version = OMNI_USD_LIVE_ABI_VERSION;
    return result;
}
OmniUsdLiveRuntimeOptions options()
{
    auto result = record<OmniUsdLiveRuntimeOptions>();
    result.max_diagnostic_records = 32;
    result.max_diagnostic_bytes = 32 * sizeof(DiagnosticEntry);
    result.max_diagnostic_sink_records = 8;
    result.max_diagnostic_sinks = 4;
    result.max_closed_incidents = 2;
    result.diagnostic_min_severity = OMNI_USD_LIVE_SEVERITY_TRACE;
    return result;
}
OmniUsdLiveDiagnosticOptions sinkOptions(OmniUsdLiveEndpoint filter = 0)
{
    auto result = record<OmniUsdLiveDiagnosticOptions>();
    result.endpoint = filter;
    result.min_severity = OMNI_USD_LIVE_SEVERITY_TRACE;
    return result;
}
struct Field {
    uint32_t kind;
    uint64_t number;
    int64_t signedNumber;
    std::string text;
};
struct Seen {
    uint64_t sequence;
    uint32_t code, severity;
    OmniUsdLiveEndpoint endpoint;
    std::string name;
    std::map<std::string, Field> fields;
};
struct Sink {
    std::mutex mutex;
    std::condition_variable changed;
    std::vector<Seen> records;
    std::function<void(const Seen&)> action;
    std::atomic<unsigned> active{0};
    std::thread::id caller = std::this_thread::get_id();
    static void OMNI_USD_LIVE_CALL callback(void* user, const OmniUsdLiveDiagnosticRecord* input)
    {
        auto& self = *static_cast<Sink*>(user);
        assert(std::this_thread::get_id() != self.caller);
        assert(input && input->struct_size == sizeof(*input) && input->abi_version == OMNI_USD_LIVE_ABI_VERSION);
        assert(input->runtime_instance_id == instance && input->sequence && input->event_name);
        assert(input->field_count <= 16);
        assert(self.active.fetch_add(1) == 0);
        struct Active { std::atomic<unsigned>& count; ~Active() { assert(count.fetch_sub(1) == 1); } } active{self.active};
        Seen seen{input->sequence, input->event_code, input->severity, input->endpoint, input->event_name, {}};
        for (uint32_t i = 0; i < input->field_count; ++i) {
            const auto* field = input->fields[i];
            assert(field && field->name && field->struct_size == sizeof(*field));
            assert(field->abi_version == OMNI_USD_LIVE_ABI_VERSION);
            assert(seen.fields.emplace(field->name, Field{field->kind, field->unsigned_value,
                field->signed_value, field->string_value ? field->string_value : ""}).second);
        }
        {
            std::lock_guard<std::mutex> lock(self.mutex);
            if (!self.records.empty()) assert(self.records.back().sequence < seen.sequence);
            self.records.push_back(seen); self.changed.notify_all();
        }
        if (self.action) self.action(seen);
    }
    void wait(size_t count)
    {
        std::unique_lock<std::mutex> lock(mutex);
        assert(changed.wait_for(lock, std::chrono::seconds(3), [&] { return records.size() >= count; }));
    }
    std::vector<Seen> copy()
    {
        std::lock_guard<std::mutex> lock(mutex);
        return records;
    }
};
template <class Predicate> void eventually(Predicate predicate)
{
    const auto deadline = Clock::now() + std::chrono::seconds(3);
    while (!predicate()) { assert(Clock::now() < deadline); std::this_thread::yield(); }
}
void finish(DiagnosticsRegistry& registry)
{
    registry.beginShutdown();
    assert(registry.waitForShutdown(Clock::now() + std::chrono::seconds(3)));
}

void catalogRpcSeverityAndOptionalEvidence()
{
    auto o = options();
    o.max_diagnostic_records = 64;
    o.max_diagnostic_bytes = 64 * sizeof(DiagnosticEntry);
    DiagnosticsRegistry registry(o, instance);
    for (const auto& expected : catalog_rpc_cases) {
        const auto result = expected.result;
        for (const bool present : {false, true}) {
            registry.observeRpc(endpoint, OMNI_USD_LIVE_OPERATION_JOIN, result, 5,
                0, result + 1, present, 0, 42, 7, 8, 0, 99);
        }
    }
    const auto snapshot = registry.snapshot(0, 64);
    assert(snapshot.records.size() == 2 * catalog_rpc_cases.size());
    for (size_t i = 0; i < snapshot.records.size(); ++i) {
        const auto& entry = snapshot.records[i];
        const auto& expected = catalog_rpc_cases[i / 2];
        const auto result = expected.result;
        assert(entry.eventCode == (result ? OMNI_USD_LIVE_RECORD_RPC_FAILED : OMNI_USD_LIVE_RECORD_RPC_COMPLETED));
        assert(entry.severity == expected.severity);
        std::map<std::string, uint64_t> fields;
        for (uint32_t f = 0; f < entry.fieldCount; ++f)
            assert(fields.emplace(DiagnosticsRegistry::fieldName(entry.fields[f].key), entry.fields[f].value).second);
        assert(fields.size() == catalog_rpc_fields.size() + i % 2);
        for (const auto* name : catalog_rpc_fields) assert(fields.count(name) == 1);
        assert(fields.at("operation") == OMNI_USD_LIVE_OPERATION_JOIN && fields.at("result") == result);
        assert(fields.at("duration_ms") == 5 && fields.at("request_id") == 0);
        assert(fields.at("attempt_id") == result + 1 && fields.at("stream_generation") == 7);
        assert(fields.at("auth_generation") == 8 && fields.at("work_id") == 0 && fields.at("session_handle") == 99);
        assert(fields.count("grpc_status") == i % 2);
        if (i % 2) assert(fields.at("grpc_status") == 0);
    }
    finish(registry);
}

// A layer write or creation the service refused over its entry limits carries
// the ErrorInfo's reason and counts in its rpc_failed record.
void rpcServiceLimitFields()
{
    DiagnosticsRegistry registry(options(), instance);
    const ServiceLimitRefusal large{1, 1, 2, 3, 5, 6}, list{2, 0, 0, 0, 4, 9};
    registry.observeRpc(endpoint, OMNI_USD_LIVE_OPERATION_UPDATE, OMNI_USD_LIVE_INVALID_ARGUMENT, 5, 1, 1, true, 3, 42,
        7, 8, 9, 0, &large);
    registry.observeRpc(endpoint, OMNI_USD_LIVE_OPERATION_CREATE, OMNI_USD_LIVE_INVALID_ARGUMENT, 5, 2, 1, true, 3, 43,
        0, 8, 0, 0, &list);
    const auto snapshot = registry.snapshot(0, 8);
    assert(snapshot.records.size() == 2);
    std::array<std::map<std::string, uint64_t>, 2> fields;
    for (size_t i = 0; i < 2; ++i) {
        const auto& entry = snapshot.records[i];
        assert(entry.eventCode == OMNI_USD_LIVE_RECORD_RPC_FAILED);
        for (uint32_t f = 0; f < entry.fieldCount; ++f)
            assert(fields[i].emplace(DiagnosticsRegistry::fieldName(entry.fields[f].key), entry.fields[f].value).second);
        for (const auto* name : catalog_rpc_fields) assert(fields[i].count(name) == 1);
        assert(fields[i].at("grpc_status") == 3);
    }
    assert(fields[0].size() == catalog_rpc_fields.size() + 6);
    assert(fields[0].at("reason") == 1 && fields[0].at("struct_commands") == 1 && fields[0].at("set_fields") == 2 &&
        fields[0].at("time_samples") == 3 && fields[0].at("limit") == 5 && fields[0].count("count") == 0);
    assert(fields[1].size() == catalog_rpc_fields.size() + 4);
    assert(fields[1].at("reason") == 2 && fields[1].at("limit") == 4 && fields[1].at("count") == 9 &&
        fields[1].count("struct_commands") == 0);
    finish(registry);
}

struct DispatchClock {
    std::atomic<int64_t> milliseconds{0};
    static Clock::time_point now(void* context) noexcept {
        return Clock::time_point{} + std::chrono::milliseconds(
            static_cast<DispatchClock*>(context)->milliseconds.load());
    }
};
struct DispatchGate {
    std::promise<void> entered, release;
    std::future<void> ready = entered.get_future();
    std::shared_future<void> proceed = release.get_future().share();
    void waitEntered() { assert(ready.wait_for(std::chrono::seconds(3)) == std::future_status::ready); }
    void hold() {
        entered.set_value();
        assert(proceed.wait_for(std::chrono::seconds(3)) == std::future_status::ready);
    }
};

void executorMetricsTrackConcurrentRequeue()
{
    DispatchClock clock;
    DispatchExecutor executor(2, DispatchClock::now, &clock);
    struct Requeued final : DispatchJob {
        std::atomic<unsigned> calls{0};
        std::array<DispatchGate, 2> gates;
        void dispatch() noexcept override {
            const auto call = calls.fetch_add(1);
            assert(call < gates.size()); gates[call].hold();
        }
    };
    struct Queued final : DispatchJob {
        std::atomic<bool> called{false};
        void dispatch() noexcept override { assert(!called.exchange(true)); }
    };
    auto job = std::make_shared<Requeued>();
    auto queued = std::make_shared<Queued>();
    executor.enqueue(*job); job->gates[0].waitEntered();
    clock.milliseconds = 5;
    // The intrusive node is free after dequeue, even while dispatch is running.
    executor.enqueue(*job); job->gates[1].waitEntered();
    clock.milliseconds = 8;
    executor.enqueue(*queued);
    clock.milliseconds = 13;
    const auto held = executor.snapshot();
    assert(held.enqueued == 3 && held.started == 2 && held.completed == 0);
    assert(held.queued == 1 && held.running == 2 && held.queueHighWater == 1);
    assert(held.configuredWorkers == 2 && held.liveWorkers == 2);
    assert(held.oldestQueuedAgeMs == 5 && held.oldestRunningAgeMs == 13);
    assert(held.queueWait.count == 2 && held.queueWait.sumMs == 0 && held.queueWait.counts[0] == 2);
    assert(held.dispatchDuration.count == 0 && !queued->called);
    job->gates[0].release.set_value(); job->gates[1].release.set_value();
    executor.beginShutdown();
    assert(executor.waitForShutdown(Clock::now() + std::chrono::seconds(3)));
    const auto done = executor.snapshot();
    assert(queued->called && job->calls == 2);
    assert(done.enqueued == 3 && done.started == 3 && done.completed == 3);
    assert(done.queued == 0 && done.running == 0 && done.liveWorkers == 0);
    assert(done.oldestQueuedAgeMs == 0 && done.oldestRunningAgeMs == 0);
    assert(done.queueWait.count == 3 && done.queueWait.sumMs == 5);
    assert(done.queueWait.counts[0] == 2 && done.queueWait.counts[1] == 1);
    assert(done.dispatchDuration.count == 3 && done.dispatchDuration.sumMs == 21);
    assert(done.dispatchDuration.counts[0] == 1 && done.dispatchDuration.counts[2] == 1 &&
           done.dispatchDuration.counts[3] == 1);
}

void executorCompletionDoesNotImplyOwnerQuiescence()
{
    DispatchClock clock;
    DispatchExecutor executor(1, DispatchClock::now, &clock);
    DispatchGate dispatch, destructor;
    struct Owned final : DispatchJob {
        DispatchExecutor& executor;
        DispatchGate& dispatchGate;
        DispatchGate& destructorGate;
        Owned(DispatchExecutor& executor, DispatchGate& dispatch, DispatchGate& destructor)
            : executor(executor), dispatchGate(dispatch), destructorGate(destructor) {}
        void dispatch() noexcept override { dispatchGate.hold(); }
        ~Owned() override {
            // Destruction must remain outside the executor lock.
            const auto retired = executor.snapshot();
            assert(retired.completed == 1 && retired.running == 0);
            destructorGate.hold();
        }
    };
    auto job = std::make_shared<Owned>(executor, dispatch, destructor);
    executor.enqueue(*job); dispatch.waitEntered();
    job.reset(); // The worker owns the final reference from here on.
    clock.milliseconds = 7;
    dispatch.release.set_value(); destructor.waitEntered();
    clock.milliseconds = 37;
    executor.beginShutdown();
    const auto held = executor.snapshot();
    assert(held.queued == 0 && held.running == 0 && held.completed == 1 && held.liveWorkers == 1);
    assert(held.oldestRunningAgeMs == 0 && held.dispatchDuration.count == 1 && held.dispatchDuration.sumMs == 7);
    // This is a nonblocking state query: the explicit destructor gate proves
    // non-quiescence, never elapsed watchdog time.
    assert(!executor.waitForShutdown(Clock::now()));
    destructor.release.set_value();
    assert(executor.waitForShutdown(Clock::now() + std::chrono::seconds(3)));
    const auto done = executor.snapshot();
    assert(done.liveWorkers == 0 && done.dispatchDuration.sumMs == 7);
}

void cqTagMetricsFollowHandoffReceiptAndDestruction()
{
    auto config = options(); config.flags = OMNI_USD_LIVE_RUNTIME_DISABLE_DIAGNOSTIC_HISTORY;
    DiagnosticsRegistry registry(config, instance);
    DispatchGate handler, destructor;
    struct Tag final : CompletionTag {
        DiagnosticsRegistry& registry;
        DispatchGate* handler = nullptr;
        DispatchGate* destructor = nullptr;
        explicit Tag(DiagnosticsRegistry& registry) : registry(registry) {}
        void onComplete(bool, Clock::time_point) noexcept override {
            const auto tags = registry.snapshot(0, 0).metrics.cqTags;
            assert(tags.handling == 1); // Reentrant observation cannot deadlock.
            if (handler) handler->hold();
        }
        ~Tag() override {
            if (destructor) {
                const auto tags = registry.snapshot(0, 0).metrics.cqTags;
                assert(tags.handling == 0 && tags.retained == 1);
                destructor->hold();
            }
        }
    };
    // Reserved terminal tags destroyed before issuance must remain invisible.
    { auto reserved = std::make_unique<Tag>(registry); }
    assert(registry.snapshot(0, 0).metrics.cqTags.submitted == 0);
    auto first = std::make_unique<Tag>(registry), second = std::make_unique<Tag>(registry);
    first->handler = &handler; first->destructor = &destructor;
    std::unique_ptr<Tag> owned(submitTag(first, registry));
    std::unique_ptr<Tag> abandoned(submitTag(second, registry));
    assert(!first && !second);
    auto tags = registry.snapshot(0, 0).metrics.cqTags;
    assert(tags.submitted == 2 && tags.outstanding == 2 && tags.retained == 2);
    assert(tags.outstandingHighWater == 2 && tags.retainedHighWater == 2);
    // Accounting must continue while shutdown drains outstanding completions.
    registry.beginShutdown();
    auto receipt = std::async(std::launch::async, [&] { owned->complete(false); });
    handler.waitEntered();
    tags = registry.snapshot(0, 0).metrics.cqTags;
    assert(tags.outstanding == 1 && tags.handling == 1 && tags.retained == 2);
    assert(tags.receivedNotOk == 1 && tags.receivedOk == 0 && tags.retired == 0);
    handler.release.set_value();
    assert(receipt.wait_for(std::chrono::seconds(3)) == std::future_status::ready); receipt.get();
    tags = registry.snapshot(0, 0).metrics.cqTags;
    assert(tags.handling == 0 && tags.retained == 2 && tags.outstanding == 1);
    abandoned.reset();
    tags = registry.snapshot(0, 0).metrics.cqTags;
    assert(tags.abandoned == 1 && tags.retired == 1 && tags.outstanding == 0 && tags.retained == 1);
    auto retirement = std::async(std::launch::async, [&] { owned.reset(); });
    destructor.waitEntered();
    tags = registry.snapshot(0, 0).metrics.cqTags;
    assert(tags.retained == 1 && tags.retired == 1 && tags.handling == 0);
    destructor.release.set_value();
    assert(retirement.wait_for(std::chrono::seconds(3)) == std::future_status::ready); retirement.get();
    auto last = std::make_unique<Tag>(registry);
    std::unique_ptr<Tag> success(submitTag(last, registry));
    success->complete(true); success.reset();
    const auto snapshot = registry.snapshot(0, 100);
    tags = snapshot.metrics.cqTags;
    assert(snapshot.records.empty());
    assert(tags.submitted == 3 && tags.retired == 3 && tags.receivedOk == 1 && tags.receivedNotOk == 1);
    assert(tags.abandoned == 1 && tags.retained == 0 && tags.handling == 0 && tags.outstanding == 0);
    assert(tags.retainedHighWater == 2 && tags.outstandingHighWater == 2);
    finish(registry);
}

void historyCountAndByteLimits()
{
    {
        auto config = options(); config.max_diagnostic_records = 3;
        DiagnosticsRegistry registry(config, instance);
        for (unsigned i = 1; i <= 5; ++i)
            registry.emit(eventCode, endpoint, i, 0, {{DiagnosticKey::Count, i}});
        const auto snapshot = registry.snapshot(0, 100);
        assert(snapshot.runtimeInstanceId == instance && snapshot.records.size() == 3);
        assert(snapshot.metrics.emitted == 5 && snapshot.metrics.historyOverwritten == 2);
        for (size_t i = 0; i < 3; ++i) {
            const auto& entry = snapshot.records[i];
            assert(entry.sequence == i + 3 && entry.subjectId == i + 3);
            assert(entry.fields[0].key == DiagnosticKey::Count && entry.fields[0].value == i + 3);
        }
        assert(snapshot.firstSequence == 3 && snapshot.lastSequence == 5);
        const auto truncated = registry.snapshot(0, 1);
        assert(truncated.truncated && truncated.records.size() == 1 && truncated.records[0].sequence == 5);
        const auto noRecords = registry.snapshot(0, 0);
        assert(noRecords.records.empty() && noRecords.metrics.emitted == 5);
        finish(registry);
    }
    {
        auto config = options();
        config.max_diagnostic_records = 20;
        config.max_diagnostic_bytes = 2 * sizeof(DiagnosticEntry) + sizeof(DiagnosticEntry) / 2;
        DiagnosticsRegistry registry(config, instance);
        for (unsigned i = 0; i < 4; ++i) registry.emit(eventCode, endpoint);
        const auto snapshot = registry.snapshot(0, 100);
        assert(snapshot.records.size() == 2 && snapshot.records[0].sequence == 3 && snapshot.records[1].sequence == 4);
        assert(snapshot.records.size() * sizeof(DiagnosticEntry) <= config.max_diagnostic_bytes);
        assert(snapshot.metrics.historyOverwritten == 2);
        finish(registry);
    }
    {
        auto config = options(); config.max_diagnostic_bytes = sizeof(DiagnosticEntry) - 1;
        DiagnosticsRegistry registry(config, instance);
        registry.emit(eventCode, endpoint);
        const auto snapshot = registry.snapshot(0, 10);
        assert(snapshot.records.empty() && snapshot.metrics.historyDropped == 1 && snapshot.metrics.emitted == 1);
        finish(registry);
    }
}

void rpcMetricsWithoutHistoryOrDelivery()
{
    auto config = options();
    config.flags = OMNI_USD_LIVE_RUNTIME_DISABLE_DIAGNOSTIC_HISTORY;
    config.diagnostic_min_severity = OMNI_USD_LIVE_SEVERITY_ERROR;
    DiagnosticsRegistry registry(config, instance);
    // Boundary pairs independently pin the documented noncumulative buckets.
    const std::array<uint64_t, 17> durations{0, 1, 2, 5, 6, 10, 11, 50, 51, 100, 101, 500, 501, 1000, 1001, 5000, 5001};
    for (size_t i = 0; i < durations.size(); ++i) {
        registry.startedRpc(OMNI_USD_LIVE_OPERATION_PING);
        registry.observeRpc(endpoint + i, OMNI_USD_LIVE_OPERATION_PING, OMNI_USD_LIVE_OK, durations[i], i + 1, 1, true, 0);
    }
    registry.startedRpc(UINT32_MAX);
    registry.observeRpc(endpoint, UINT32_MAX, 123456, 5001, 99, 1, false, 0);
    auto snapshot = registry.snapshot(0, 100);
    assert(snapshot.records.empty());
    assert(snapshot.firstSequence == 0 && snapshot.lastSequence == 18);
    assert(snapshot.metrics.suppressed == 17 && snapshot.metrics.historyDropped == 1);
    assert(snapshot.metrics.events[OMNI_USD_LIVE_RECORD_RPC_COMPLETED] == 17 &&
           snapshot.metrics.events[OMNI_USD_LIVE_RECORD_RPC_FAILED] == 1);
    assert(snapshot.metrics.rpcAttempts == 18 && snapshot.metrics.rpcCompletions == 18 && !snapshot.metrics.rpcInFlight);
    assert(snapshot.metrics.rpcAttemptsByOperation[OMNI_USD_LIVE_OPERATION_PING] == 17 && snapshot.metrics.rpcAttemptsByOperation[16] == 1);
    assert(snapshot.metrics.rpc[OMNI_USD_LIVE_OPERATION_PING][OMNI_USD_LIVE_OK] == 17 && snapshot.metrics.rpc[16][22] == 1);
    const std::array<uint64_t, 9> expected{2, 2, 2, 2, 2, 2, 2, 2, 2};
    assert(snapshot.metrics.rpcDuration == expected);
    const auto& ping = snapshot.metrics.rpcDurationByOperationResult[OMNI_USD_LIVE_OPERATION_PING][OMNI_USD_LIVE_OK];
    assert(ping.count == 17 && ping.sumMs == 13340);
    assert((ping.counts == std::array<uint64_t, 9>{2, 2, 2, 2, 2, 2, 2, 2, 1}));
    const auto& unknown = snapshot.metrics.rpcDurationByOperationResult[16][22];
    assert(unknown.count == 1 && unknown.sumMs == 5001 && unknown.counts[8] == 1);
    // Filtering records cannot turn aggregate metrics into unbounded per-endpoint labels.
    const auto filtered = registry.snapshot(99999, 100);
    assert(filtered.metrics.rpc == snapshot.metrics.rpc && filtered.metrics.rpcDuration == expected);
    assert(filtered.metrics.rpcDurationByOperationResult[OMNI_USD_LIVE_OPERATION_PING][OMNI_USD_LIVE_OK].counts == ping.counts);
    registry.startedRpc(OMNI_USD_LIVE_OPERATION_PING);
    assert(registry.snapshot(0, 0).metrics.rpcInFlight == 1);
    registry.observeRpc(endpoint, OMNI_USD_LIVE_OPERATION_PING, OMNI_USD_LIVE_CANCELLED, 0, 100, 1, true, 1);
    snapshot = registry.snapshot(0, 0);
    assert(snapshot.metrics.rpcInFlight == 0);
    const auto& cancelled = snapshot.metrics.rpcDurationByOperationResult[OMNI_USD_LIVE_OPERATION_PING][OMNI_USD_LIVE_CANCELLED];
    assert(cancelled.count == 1 && cancelled.sumMs == 0 && cancelled.counts[0] == 1);
    for (auto duration : {UINT64_MAX, UINT64_C(1)}) {
        registry.startedRpc(UINT32_MAX);
        registry.observeRpc(endpoint, UINT32_MAX, UINT32_MAX, duration, 101, 1, false, 0);
    }
    const auto saturated = registry.snapshot(0, 0).metrics.rpcDurationByOperationResult[16][22];
    assert(saturated.count == 3 && saturated.sumMs == UINT64_MAX && saturated.counts[0] == 1 && saturated.counts[8] == 2);
    finish(registry);
}

void completionDurationsWithoutHistory()
{
    auto now = Clock::time_point{};
    auto clock = [](void* context) noexcept { return *static_cast<Clock::time_point*>(context); };
    auto config = options(); config.flags = OMNI_USD_LIVE_RUNTIME_DISABLE_DIAGNOSTIC_HISTORY;
    config.diagnostic_min_severity = OMNI_USD_LIVE_SEVERITY_ERROR;
    DiagnosticsRegistry registry(config, instance, clock, &now);
    auto emit = [&](uint32_t code, uint64_t result, uint64_t duration) {
        registry.emit(code, endpoint, 101, 0,
            {{DiagnosticKey::Result, result}, {DiagnosticKey::DurationMs, duration}});
    };
    for (auto duration : {0u, 1u, 2u, 5u, 6u, 10u, 11u, 50u, 51u, 100u, 101u, 500u, 501u, 1000u, 1001u, 5000u, 5001u})
        emit(OMNI_USD_LIVE_RECORD_SAVE_COMPLETED, OMNI_USD_LIVE_OK, duration);
    emit(OMNI_USD_LIVE_RECORD_SAVE_COMPLETED, OMNI_USD_LIVE_BUSY, 0);
    emit(OMNI_USD_LIVE_RECORD_SAVE_COMPLETED, UINT64_MAX, UINT64_MAX);
    emit(OMNI_USD_LIVE_RECORD_SAVE_COMPLETED, UINT64_MAX, 1);
    emit(OMNI_USD_LIVE_RECORD_DRAIN_COMPLETED, OMNI_USD_LIVE_OK, 0);
    emit(OMNI_USD_LIVE_RECORD_DRAIN_COMPLETED, OMNI_USD_LIVE_BUSY, 5);
    emit(OMNI_USD_LIVE_RECORD_DRAIN_TIMEOUT, OMNI_USD_LIVE_TIMEOUT, 10);
    emit(OMNI_USD_LIVE_RECORD_DRAIN_TIMEOUT, OMNI_USD_LIVE_TIMEOUT, 10);
    emit(OMNI_USD_LIVE_RECORD_DRAIN_COMPLETED, OMNI_USD_LIVE_OK, 0);
    emit(OMNI_USD_LIVE_RECORD_DRAIN_COMPLETED, OMNI_USD_LIVE_IO_ERROR, 11);
    emit(OMNI_USD_LIVE_RECORD_DRAIN_COMPLETED, UINT64_MAX, UINT64_MAX);
    emit(OMNI_USD_LIVE_RECORD_DRAIN_COMPLETED, UINT64_MAX, 1);
    // Missing duration/result is not a zero-valued sample, and other event
    // families cannot manufacture save/drain measurements.
    registry.emit(OMNI_USD_LIVE_RECORD_SAVE_COMPLETED, endpoint, 0, 0, {{DiagnosticKey::Result, OMNI_USD_LIVE_OK}});
    registry.emit(OMNI_USD_LIVE_RECORD_DRAIN_COMPLETED, endpoint, 0, 0, {{DiagnosticKey::DurationMs, 0}});
    registry.emit(OMNI_USD_LIVE_RECORD_DRAIN_COMPLETED, endpoint, 0, 0, {{DiagnosticKey::Result, OMNI_USD_LIVE_OK}});
    emit(OMNI_USD_LIVE_RECORD_RETRY_SCHEDULED, OMNI_USD_LIVE_OK, 5000);
    const auto snapshot = registry.snapshot(otherEndpoint, 100);
    assert(snapshot.records.empty() && snapshot.metrics.warningSuppressed == 1);
    const auto& saves = snapshot.metrics.saveDurationByResult;
    assert(saves[OMNI_USD_LIVE_OK].count == 17 && saves[OMNI_USD_LIVE_OK].sumMs == 13340);
    assert((saves[OMNI_USD_LIVE_OK].counts == std::array<uint64_t, 9>{2, 2, 2, 2, 2, 2, 2, 2, 1}));
    assert(saves[OMNI_USD_LIVE_BUSY].count == 1 && saves[OMNI_USD_LIVE_BUSY].sumMs == 0);
    assert(saves[22].count == 2 && saves[22].sumMs == UINT64_MAX);
    const auto& drains = snapshot.metrics.drainDurationByResult;
    assert(drains[OMNI_USD_LIVE_OK].count == 2 && drains[OMNI_USD_LIVE_OK].sumMs == 0 && drains[OMNI_USD_LIVE_OK].counts[0] == 2);
    assert(drains[OMNI_USD_LIVE_BUSY].count == 1 && drains[OMNI_USD_LIVE_BUSY].counts[1] == 1);
    assert(drains[OMNI_USD_LIVE_TIMEOUT].count == 2 && drains[OMNI_USD_LIVE_TIMEOUT].sumMs == 20 &&
           drains[OMNI_USD_LIVE_TIMEOUT].counts[2] == 2);
    assert(snapshot.metrics.rpcCompletions == 0 && snapshot.metrics.rpcAttempts == 0);
    assert(drains[OMNI_USD_LIVE_IO_ERROR].count == 1 && drains[OMNI_USD_LIVE_IO_ERROR].sumMs == 11 && drains[OMNI_USD_LIVE_IO_ERROR].counts[3] == 1);
    assert(drains[22].count == 2 && drains[22].sumMs == UINT64_MAX && drains[22].counts[0] == 1 && drains[22].counts[8] == 1);
    finish(registry);
}

// The receive-to-apply aggregate counts every sample in fixed storage: a
// batch reports each of its updates, the maximum and every histogram field
// saturate independently, and the lagging-layer gauge follows its intervals.
void applyLagMetricsCountEveryUpdate()
{
    DiagnosticApplyLagMetrics direct;
    for (auto duration : {0u, 1u, 2u, 5u, 6u, 10u, 11u, 50u, 51u, 100u, 101u, 500u, 501u, 1000u, 1001u, 5000u, 5001u})
        direct.observe(duration);
    assert(direct.receiveToApply.count == 17 && direct.receiveToApply.sumMs == 13340);
    assert((direct.receiveToApply.counts == std::array<uint64_t, 9>{2, 2, 2, 2, 2, 2, 2, 2, 1}));
    assert(direct.receiveToApplyMaxMs == 5001);
    direct.observe(3); // A smaller sample keeps the maximum.
    assert(direct.receiveToApplyMaxMs == 5001 && direct.receiveToApply.count == 18);
    direct.observe(UINT64_MAX);
    direct.observe(1);
    assert(direct.receiveToApply.sumMs == UINT64_MAX && direct.receiveToApplyMaxMs == UINT64_MAX);
    assert(direct.receiveToApply.count == 20 && direct.receiveToApply.counts[8] == 2);
    direct.receiveToApply.count = UINT64_MAX;
    direct.observe(0);
    assert(direct.receiveToApply.count == UINT64_MAX && direct.receiveToApply.counts[0] == 4);

    auto config = options(); config.flags = OMNI_USD_LIVE_RUNTIME_DISABLE_DIAGNOSTIC_HISTORY;
    config.diagnostic_min_severity = OMNI_USD_LIVE_SEVERITY_ERROR;
    DiagnosticsRegistry registry(config, instance);
    // One full batch of 64 updates that share a retirement, received over time.
    std::array<uint64_t, 64> batch{};
    for (size_t i = 0; i < batch.size(); ++i) batch[i] = 64 - i;
    registry.appliedPeerUpdates(batch.data(), batch.size());
    const uint64_t alone = 7000;
    registry.appliedPeerUpdates(&alone, 1);
    registry.appliedPeerUpdates(nullptr, 0);
    registry.applyLagChanged(true);
    registry.applyLagChanged(true);
    registry.applyLagChanged(false);
    // The records are counted though history is off and their severity filtered.
    registry.emit(OMNI_USD_LIVE_RECORD_APPLY_LAG_ENTERED, endpoint, 101, 0,
        {{DiagnosticKey::Generation, 1}, {DiagnosticKey::ReadSequence, 300}, {DiagnosticKey::AppliedSequence, 10}});
    registry.emit(OMNI_USD_LIVE_RECORD_APPLY_LAG_EXITED, endpoint, 101, 0,
        {{DiagnosticKey::Generation, 1}, {DiagnosticKey::ReadSequence, 300}, {DiagnosticKey::AppliedSequence, 300},
            {DiagnosticKey::DurationMs, 12}});
    const auto snapshot = registry.snapshot(0, 100);
    const auto& metric = snapshot.metrics.applyLag;
    assert(snapshot.records.empty());
    assert(metric.receiveToApply.count == 65 && metric.receiveToApply.sumMs == 64 * 65 / 2 + alone);
    // Buckets <=1, <=5, <=10, <=50, <=100 and >5000 ms.
    assert((metric.receiveToApply.counts == std::array<uint64_t, 9>{1, 4, 5, 40, 14, 0, 0, 0, 1}));
    assert(metric.receiveToApplyMaxMs == alone);
    assert(metric.current == 1 && metric.highWater == 2);
    assert(snapshot.metrics.events[OMNI_USD_LIVE_RECORD_APPLY_LAG_ENTERED] == 1);
    assert(snapshot.metrics.events[OMNI_USD_LIVE_RECORD_APPLY_LAG_EXITED] == 1);
    registry.applyLagChanged(false);
    assert(registry.snapshot(0, 0).metrics.applyLag.current == 0);
    finish(registry);
}

void shutdownObservationsNeverDispatch()
{
    for (const bool disabled : {false, true}) {
        auto config=options();
        if(disabled) config.flags=OMNI_USD_LIVE_RUNTIME_DISABLE_DIAGNOSTIC_HISTORY;
        DiagnosticsRegistry registry(config,instance);
        Sink sink;
        const auto handle=registry.subscribe(sinkOptions(),&sink,&Sink::callback);
        registry.observeShutdown(ShutdownPhase::ReferenceRelease,OMNI_USD_LIVE_OK,0);
        registry.observeShutdown(ShutdownPhase::Admission,OMNI_USD_LIVE_BUSY,5);
        registry.observeShutdown(ShutdownPhase::CompletionQueue,OMNI_USD_LIVE_TIMEOUT,11);
        // A following delivered record establishes the FIFO boundary without
        // waiting to see whether a forbidden callback eventually arrives.
        registry.emit(eventCode,endpoint);
        sink.wait(1);
        assert(registry.unsubscribe(handle,3000)==OMNI_USD_LIVE_OK);
        assert(sink.copy().size()==1 && sink.copy()[0].code==eventCode);
        finish(registry); // Executor is fully stopped before later observation.
        const auto stopped=registry.snapshot(0,0).diagnosticExecutor;
        for(const auto phase:{ShutdownPhase::Sessions,ShutdownPhase::Authentication,ShutdownPhase::Status,
                              ShutdownPhase::Requests,ShutdownPhase::Diagnostics})
            registry.observeShutdown(phase,OMNI_USD_LIVE_TIMEOUT,0);
        registry.observeShutdown(ShutdownPhase::Complete,OMNI_USD_LIVE_OK,50);
        registry.observeShutdown(ShutdownPhase::Diagnostics,UINT32_MAX,UINT64_MAX);
        registry.observeShutdown(ShutdownPhase::Diagnostics,UINT32_MAX,1);
        const auto snapshot=registry.snapshot(0,100);
        const auto& m=snapshot.metrics.shutdownDurationByPhaseResult;
        assert(m[0][OMNI_USD_LIVE_OK].count==1 && m[0][OMNI_USD_LIVE_OK].counts[0]==1);
        assert(m[1][OMNI_USD_LIVE_BUSY].sumMs==5 && m[1][OMNI_USD_LIVE_BUSY].counts[1]==1);
        assert(m[2][OMNI_USD_LIVE_TIMEOUT].sumMs==11 && m[2][OMNI_USD_LIVE_TIMEOUT].counts[3]==1);
        for(size_t phase=3;phase<=7;++phase) assert(m[phase][OMNI_USD_LIVE_TIMEOUT].count==1 && m[phase][OMNI_USD_LIVE_TIMEOUT].sumMs==0);
        assert(m[8][OMNI_USD_LIVE_OK].count==1 && m[8][OMNI_USD_LIVE_OK].sumMs==50);
        assert(m[7][22].count==2 && m[7][22].sumMs==UINT64_MAX && m[7][22].counts[0]==1 && m[7][22].counts[8]==1);
        assert(snapshot.metrics.events[OMNI_USD_LIVE_RECORD_QUIESCENCE_FAILED]==6);
        assert(snapshot.records.size()==(disabled ? 0 : 7));
        assert(snapshot.diagnosticExecutor.enqueued==stopped.enqueued &&
               snapshot.diagnosticExecutor.started==stopped.started &&
               snapshot.diagnosticExecutor.completed==stopped.completed &&
               snapshot.diagnosticExecutor.liveWorkers==0);
    }
}

void readinessMetricsWithVirtualTime()
{
    auto config = options(); config.flags = OMNI_USD_LIVE_RUNTIME_DISABLE_DIAGNOSTIC_HISTORY;
    DiagnosticsRegistry diagnostics(config, instance);
    DispatchExecutor executor(1);
    Clock::time_point now{}; // Zero is a valid start, not an inactive sentinel.
    const auto observe = [](void* context, OmniUsdLiveEndpoint, OmniUsdLiveSession, std::string_view,
                            const OmniUsdLiveStatus* previous, const OmniUsdLiveStatus& current,
                            bool attached, const StatusTiming& timing) noexcept {
        static_cast<DiagnosticsRegistry*>(context)->subjectTransition(current.subject_kind,
            previous ? previous->state : 0, current.state, attached, timing);
    };
    StatusRegistry hub(executor, 2, 0, observe, &diagnostics,
        [](void* clock) noexcept { return *static_cast<Clock::time_point*>(clock); }, &now);
    const auto at = [&](uint64_t ms) { now = Clock::time_point{} + std::chrono::milliseconds(ms); };
    const auto metrics = [&] { return diagnostics.snapshot(otherEndpoint, 0).metrics.readiness; };
    auto endpointStatus = record<OmniUsdLiveStatus>();
    endpointStatus.subject_kind = OMNI_USD_LIVE_SUBJECT_ENDPOINT;
    endpointStatus.state = OMNI_USD_LIVE_STATE_OPENING;
    const auto ep = hub.add(endpoint, 0, "", endpointStatus);
    endpointStatus.state = OMNI_USD_LIVE_STATE_READY;
    assert(hub.publish(ep, endpointStatus) == OMNI_USD_LIVE_OK);
    auto value = record<OmniUsdLiveStatus>();
    value.subject_kind = OMNI_USD_LIVE_SUBJECT_LAYER; value.state = OMNI_USD_LIVE_STATE_OPENING;
    auto layer = hub.add(endpoint, 0, resource, value);
    assert(metrics()[1].initialWaiting == 1 && !metrics()[0].initialWaiting);
    try { hub.add(endpoint, 0, resource + "-extra", value); assert(false); }
    catch (OmniUsdLiveResult result) { assert(result == OMNI_USD_LIVE_RESOURCE_EXHAUSTED); }
    assert(metrics()[1].initialWaiting == 1);
    const auto publish = [&](uint64_t id, uint64_t ms, uint32_t state, uint32_t result = OMNI_USD_LIVE_OK) {
        at(ms); value.state = state; value.result = result;
        assert(hub.publish(id, value) == OMNI_USD_LIVE_OK);
    };
    publish(layer, 5, OMNI_USD_LIVE_STATE_AUTH_WAIT, OMNI_USD_LIVE_UNAUTHENTICATED);
    publish(layer, 7, OMNI_USD_LIVE_STATE_RECONNECTING, OMNI_USD_LIVE_UNAVAILABLE);
    publish(layer, 11, OMNI_USD_LIVE_STATE_BLOCKED);
    assert(metrics()[1].initialWaiting == 1 && !metrics()[1].recovering);
    publish(layer, 17, OMNI_USD_LIVE_STATE_READY);
    auto observed = metrics()[1];
    assert(!observed.initialWaiting && !observed.recovering);
    assert(observed.initialByResolution[0].count == 1 && observed.initialByResolution[0].sumMs == 17);
    assert(observed.initialByResolution[0].counts[3] == 1);
    value.pending_write_count = 1; // Same-state semantic changes do not restart readiness.
    publish(layer, 25, OMNI_USD_LIVE_STATE_READY);
    publish(layer, 30, OMNI_USD_LIVE_STATE_AUTH_WAIT, OMNI_USD_LIVE_UNAUTHENTICATED);
    publish(layer, 40, OMNI_USD_LIVE_STATE_RECONNECTING, OMNI_USD_LIVE_UNAVAILABLE);
    publish(layer, 45, OMNI_USD_LIVE_STATE_BLOCKED);
    assert(metrics()[1].recovering == 1 && !metrics()[1].recoveryByResolution[0].count);
    publish(layer, 60, OMNI_USD_LIVE_STATE_READY);
    observed = metrics()[1];
    assert(!observed.recovering && observed.initialByResolution[0].count == 1);
    assert(observed.recoveryByResolution[0].count == 1 && observed.recoveryByResolution[0].sumMs == 30);
    publish(layer, 70, OMNI_USD_LIVE_STATE_RECONNECTING, OMNI_USD_LIVE_UNAVAILABLE);
    publish(layer, 80, OMNI_USD_LIVE_STATE_FAILED, OMNI_USD_LIVE_OUTCOME_UNKNOWN);
    publish(layer, 90, OMNI_USD_LIVE_STATE_READY); // Latched failure cannot manufacture success.
    publish(layer, 100, OMNI_USD_LIVE_STATE_CLOSING);
    at(101); assert(hub.remove(layer) == OMNI_USD_LIVE_OK);
    observed = metrics()[1];
    assert(!observed.recovering && observed.recoveryByResolution[1].count == 1);
    assert(observed.recoveryByResolution[1].sumMs == 10 && observed.recoveryByResolution[1].counts[2] == 1);
    assert(!observed.recoveryByResolution[2].count);

    at(110); value = record<OmniUsdLiveStatus>();
    value.subject_kind = OMNI_USD_LIVE_SUBJECT_LAYER; value.state = OMNI_USD_LIVE_STATE_OPENING;
    layer = hub.add(endpoint, 0, resource, value);
    publish(layer, 120, OMNI_USD_LIVE_STATE_FAILED, OMNI_USD_LIVE_TIMEOUT);
    assert(hub.remove(layer) == OMNI_USD_LIVE_OK);
    at(130); value.state = OMNI_USD_LIVE_STATE_OPENING; value.result = OMNI_USD_LIVE_OK;
    layer = hub.add(endpoint, 0, resource, value);
    publish(layer, 150, OMNI_USD_LIVE_STATE_CLOSING);
    publish(layer, 160, OMNI_USD_LIVE_STATE_CLOSED);
    assert(hub.remove(layer) == OMNI_USD_LIVE_OK);
    observed = metrics()[1];
    assert(!observed.initialWaiting && observed.initialByResolution[1].count == 1);
    assert(observed.initialByResolution[1].sumMs == 10);
    assert(observed.initialByResolution[2].count == 1 && observed.initialByResolution[2].sumMs == 20);

    // Initial READY/terminal/CLOSING admissions resolve immediately. Direct
    // READY->terminal has no automatic recovery episode; removal never repeats it.
    at(200); value.subject_kind = OMNI_USD_LIVE_SUBJECT_SESSION; value.state = OMNI_USD_LIVE_STATE_READY;
    auto session = hub.add(endpoint, 73, "", value);
    publish(session, 205, OMNI_USD_LIVE_STATE_RECONNECTING, OMNI_USD_LIVE_UNAVAILABLE);
    at(212); assert(hub.remove(session) == OMNI_USD_LIVE_OK);
    at(220); value.state = OMNI_USD_LIVE_STATE_READY; value.result = OMNI_USD_LIVE_OK;
    session = hub.add(endpoint, 73, "", value);
    publish(session, 230, OMNI_USD_LIVE_STATE_FAILED, OMNI_USD_LIVE_PERMISSION_DENIED);
    assert(hub.remove(session) == OMNI_USD_LIVE_OK);
    at(240); session = hub.add(endpoint, 73, "", value);
    assert(hub.remove(session) == OMNI_USD_LIVE_OK);
    value.state = OMNI_USD_LIVE_STATE_CLOSING; value.result = OMNI_USD_LIVE_OK;
    session = hub.add(endpoint, 73, "", value);
    assert(hub.remove(session) == OMNI_USD_LIVE_OK);
    observed = metrics()[0];
    assert(!observed.initialWaiting && !observed.recovering);
    assert(observed.initialByResolution[0].count == 2 && observed.initialByResolution[0].sumMs == 0);
    assert(observed.initialByResolution[0].counts[0] == 2);
    assert(observed.initialByResolution[1].count == 1 && !observed.initialByResolution[1].sumMs);
    assert(observed.initialByResolution[2].count == 1 && !observed.initialByResolution[2].sumMs);
    assert(!observed.recoveryByResolution[0].count && !observed.recoveryByResolution[1].count);
    assert(observed.recoveryByResolution[2].count == 1 && observed.recoveryByResolution[2].sumMs == 7);
    assert(hub.remove(ep) == OMNI_USD_LIVE_OK);
    const auto snapshot = diagnostics.snapshot(0, 100);
    assert(snapshot.records.empty());
    for (const auto& kind : snapshot.metrics.subjects) assert(!kind.current);
    finish(diagnostics);
}

void backpressureMetricsWithVirtualTime()
{
    auto config = options(); config.flags = OMNI_USD_LIVE_RUNTIME_DISABLE_DIAGNOSTIC_HISTORY;
    config.diagnostic_min_severity = OMNI_USD_LIVE_SEVERITY_ERROR;
    DiagnosticsRegistry diagnostics(config, instance);
    DispatchExecutor executor(1);
    Clock::time_point now{};
    const auto observe = [](void* context, OmniUsdLiveEndpoint, OmniUsdLiveSession, std::string_view,
                            const OmniUsdLiveStatus* previous, const OmniUsdLiveStatus& current,
                            bool attached, const StatusTiming& timing) noexcept {
        static_cast<DiagnosticsRegistry*>(context)->subjectTransition(current.subject_kind,
            previous ? previous->state : 0, current.state, attached, timing);
    };
    StatusRegistry hub(executor, 3, 0, observe, &diagnostics,
        [](void* clock) noexcept { return *static_cast<Clock::time_point*>(clock); }, &now);
    const auto at = [&](uint64_t ms) { now = Clock::time_point{} + std::chrono::milliseconds(ms); };
    const auto metrics = [&] { return diagnostics.snapshot(otherEndpoint, 0).metrics.backpressure; };
    auto value = record<OmniUsdLiveStatus>();
    value.subject_kind = OMNI_USD_LIVE_SUBJECT_ENDPOINT; value.state = OMNI_USD_LIVE_STATE_READY;
    const auto ep = hub.add(endpoint, 0, "", value);
    value.subject_kind = OMNI_USD_LIVE_SUBJECT_LAYER; value.state = OMNI_USD_LIVE_STATE_BLOCKED;
    const auto a = hub.add(endpoint, 0, resource, value); // Time zero is an active interval.
    at(5); const auto b = hub.add(endpoint, 0, resource + "-b", value);
    assert(metrics()[2].current == 2 && metrics()[2].highWater == 2 && !metrics()[0].current);
    try { hub.add(endpoint, 0, resource + "-extra", value); assert(false); }
    catch (OmniUsdLiveResult result) { assert(result == OMNI_USD_LIVE_RESOURCE_EXHAUSTED); }
    at(7); value.pending_bytes = 17;
    assert(hub.publish(a, value) == OMNI_USD_LIVE_OK); // Changed counts must not restart.
    assert(hub.publish(a, value) == OMNI_USD_LIVE_OK); // Identical status is neutral.
    const auto publish = [&](uint64_t id, uint64_t ms, uint32_t state, uint32_t result = OMNI_USD_LIVE_OK) {
        at(ms); value.state = state; value.result = result;
        assert(hub.publish(id, value) == OMNI_USD_LIVE_OK);
    };
    publish(a, 10, OMNI_USD_LIVE_STATE_AUTH_WAIT, OMNI_USD_LIVE_UNAUTHENTICATED);
    auto observed = metrics()[2];
    assert(observed.current == 1 && observed.durationByResolution[0].count == 1);
    assert(observed.durationByResolution[0].sumMs == 10 && observed.durationByResolution[0].counts[2] == 1);
    publish(a, 12, OMNI_USD_LIVE_STATE_READY); // Leaving BLOCKED does not promise readiness.
    publish(a, 20, OMNI_USD_LIVE_STATE_BLOCKED);
    publish(a, 25, OMNI_USD_LIVE_STATE_FAILED, OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    publish(a, 27, OMNI_USD_LIVE_STATE_BLOCKED); // Latched failure cannot open another interval.
    at(30); assert(hub.remove(a) == OMNI_USD_LIVE_OK);
    publish(b, 35, OMNI_USD_LIVE_STATE_CLOSING);
    publish(b, 40, OMNI_USD_LIVE_STATE_CLOSED);
    assert(hub.remove(b) == OMNI_USD_LIVE_OK);
    assert(hub.remove(b) == OMNI_USD_LIVE_INVALID_HANDLE);
    observed = metrics()[2];
    assert(!observed.current && observed.highWater == 2);
    assert(observed.durationByResolution[1].count == 1 && observed.durationByResolution[1].sumMs == 5);
    assert(observed.durationByResolution[2].count == 1 && observed.durationByResolution[2].sumMs == 30);
    at(50); value = record<OmniUsdLiveStatus>();
    value.subject_kind = OMNI_USD_LIVE_SUBJECT_LAYER; value.state = OMNI_USD_LIVE_STATE_BLOCKED;
    auto layer = hub.add(endpoint, 0, resource, value);
    assert(hub.remove(layer) == OMNI_USD_LIVE_OK); // Direct removal, zero-duration sample.
    observed = metrics()[2];
    assert(observed.durationByResolution[2].count == 2 && observed.durationByResolution[2].sumMs == 30);
    assert(observed.durationByResolution[2].counts[0] == 1);
    at(70); value.subject_kind = OMNI_USD_LIVE_SUBJECT_SESSION;
    const auto session = hub.add(endpoint, 73, "", value);
    publish(session, 77, OMNI_USD_LIVE_STATE_AUTH_WAIT);
    assert(hub.remove(session) == OMNI_USD_LIVE_OK);
    assert(!metrics()[1].current && metrics()[1].highWater == 1);
    assert(metrics()[1].durationByResolution[0].count == 1 && metrics()[1].durationByResolution[0].sumMs == 7);
    at(80); value.subject_kind = OMNI_USD_LIVE_SUBJECT_LAYER; value.state = OMNI_USD_LIVE_STATE_BLOCKED;
    layer = hub.add(endpoint, 0, resource, value);
    value.recovery_action = OMNI_USD_LIVE_RECOVERY_PRESERVE_AND_REOPEN;
    publish(layer, 88, OMNI_USD_LIVE_STATE_CLOSING, OMNI_USD_LIVE_OUTCOME_UNKNOWN);
    assert(hub.remove(layer) == OMNI_USD_LIVE_OK); // Failure takes priority over simultaneous close.
    observed = metrics()[2];
    assert(!observed.current && observed.durationByResolution[1].count == 2 &&
           observed.durationByResolution[1].sumMs == 13 && observed.durationByResolution[2].count == 2);
    at(90); value = record<OmniUsdLiveStatus>();
    value.subject_kind = OMNI_USD_LIVE_SUBJECT_LAYER; value.state = OMNI_USD_LIVE_STATE_BLOCKED;
    layer = hub.add(endpoint, 0, resource, value);
    value.recovery_action = OMNI_USD_LIVE_RECOVERY_PRESERVE_AND_REOPEN;
    publish(layer, 95, OMNI_USD_LIVE_STATE_BLOCKED, OMNI_USD_LIVE_OUTCOME_UNKNOWN);
    observed = metrics()[2]; // A terminal cause can end BLOCKED without changing the state value.
    assert(!observed.current && observed.durationByResolution[1].count == 3 && observed.durationByResolution[1].sumMs == 18);
    assert(hub.remove(layer) == OMNI_USD_LIVE_OK);
    layer = hub.add(endpoint, 0, resource, value); // Already terminal BLOCKED never opens an interval.
    assert(!metrics()[2].current);
    assert(hub.remove(layer) == OMNI_USD_LIVE_OK);
    assert(metrics()[2].durationByResolution[1].count == 3);
    value = record<OmniUsdLiveStatus>(); value.subject_kind = OMNI_USD_LIVE_SUBJECT_ENDPOINT;
    publish(ep, 100, OMNI_USD_LIVE_STATE_BLOCKED);
    at(112); assert(hub.remove(ep) == OMNI_USD_LIVE_OK);
    assert(!metrics()[0].current && metrics()[0].highWater == 1);
    assert(metrics()[0].durationByResolution[2].count == 1 && metrics()[0].durationByResolution[2].sumMs == 12);
    assert(diagnostics.snapshot(0, 100).records.empty());
    finish(diagnostics);
}

void authMetricsWithoutHistory()
{
    auto now = Clock::time_point{};
    auto clock = [](void* context) noexcept { return *static_cast<Clock::time_point*>(context); };
    auto config = options(); config.flags = OMNI_USD_LIVE_RUNTIME_DISABLE_DIAGNOSTIC_HISTORY;
    config.diagnostic_min_severity = OMNI_USD_LIVE_SEVERITY_ERROR;
    DiagnosticsRegistry registry(config, instance, clock, &now);
    for (auto reason : {UINT64_C(1), UINT64_C(1), UINT64_C(2), UINT64_C(3), UINT64_MAX})
        registry.emit(OMNI_USD_LIVE_RECORD_AUTH_CHALLENGE, endpoint, 0, 0, {{DiagnosticKey::Reason, reason}});
    for (auto cause : {AuthFailureCause::Provider, AuthFailureCause::Deadline, AuthFailureCause::RejectionLimit, AuthFailureCause::Internal})
        registry.observeAuthFailure(cause);
    registry.staleAuthReplacement(); registry.staleAuthReplacement();
    registry.startAuthWait(); registry.startAuthWait();
    assert(registry.snapshot(0, 0).metrics.authBlockedEndpoints == 2);
    registry.endAuthWait(AuthWaitResolution::Usable, 0);
    registry.endAuthWait(AuthWaitResolution::Failed, UINT64_MAX);
    registry.startAuthWait(); registry.endAuthWait(AuthWaitResolution::Failed, 1);
    registry.startAuthWait(); registry.endAuthWait(AuthWaitResolution::Closed, 6);
    const auto snapshot = registry.snapshot(otherEndpoint, 100);
    const auto& metrics = snapshot.metrics;
    assert(snapshot.records.empty() && metrics.warningSuppressed == 1);
    assert((metrics.authChallengesByReason == std::array<uint64_t, 4>{1, 2, 1, 1}));
    assert((metrics.authFailuresByCause == std::array<uint64_t, 4>{1, 1, 1, 1}));
    assert(metrics.authStaleReplacements == 2 && metrics.authBlockedEndpoints == 0 && metrics.authBlockedEndpointsHighWater == 2);
    assert(metrics.authWaitByResolution[0].count == 1 && metrics.authWaitByResolution[0].counts[0] == 1);
    assert(metrics.authWaitByResolution[1].count == 2 && metrics.authWaitByResolution[1].sumMs == UINT64_MAX);
    assert(metrics.authWaitByResolution[1].counts[0] == 1 && metrics.authWaitByResolution[1].counts[8] == 1);
    assert(metrics.authWaitByResolution[2].count == 1 && metrics.authWaitByResolution[2].sumMs == 6 &&
           metrics.authWaitByResolution[2].counts[2] == 1);
    finish(registry);
}

void sinkFiltersAndResourcePermissions()
{
    for (bool allowed : {false, true}) for (bool requested : {false, true}) {
        auto config = options();
        config.diagnostic_allowed_flags = allowed ? OMNI_USD_LIVE_DIAGNOSTIC_RESOURCE_NAMES : 0;
        DiagnosticsRegistry registry(config, instance);
        registry.emit(OMNI_USD_LIVE_RECORD_RPC_FAILED, endpoint); // Subscription must not replay history.
        Sink sink;
        uint32_t selected = OMNI_USD_LIVE_RECORD_RPC_FAILED;
        auto subscription = sinkOptions(endpoint);
        subscription.min_severity = OMNI_USD_LIVE_SEVERITY_WARN;
        subscription.flags = requested ? OMNI_USD_LIVE_DIAGNOSTIC_RESOURCE_NAMES : 0;
        subscription.event_code_count = 1; subscription.event_codes = &selected;
        const auto handle = registry.subscribe(subscription, &sink, Sink::callback);
        selected = OMNI_USD_LIVE_RECORD_AUTH_FAILED; // Admission owns filter values.
        registry.emit(OMNI_USD_LIVE_RECORD_RPC_FAILED, otherEndpoint);
        registry.emit(OMNI_USD_LIVE_RECORD_RPC_FAILED, endpoint, 0, 0, {}, {}, OMNI_USD_LIVE_SEVERITY_INFO);
        registry.emit(OMNI_USD_LIVE_RECORD_AUTH_FAILED, endpoint);
        std::string authored = resource;
        registry.emit(OMNI_USD_LIVE_RECORD_RPC_FAILED, endpoint, 123, 456,
            {{DiagnosticKey::Operation, OMNI_USD_LIVE_OPERATION_UPDATE}, {DiagnosticKey::GrpcStatus, 14}}, authored,
            OMNI_USD_LIVE_SEVERITY_ERROR);
        authored.assign("mutated-caller-buffer");
        sink.wait(1);
        assert(registry.unsubscribe(handle, 3000) == OMNI_USD_LIVE_OK);
        const auto records = sink.copy();
        assert(records.size() == 1 && records[0].code == OMNI_USD_LIVE_RECORD_RPC_FAILED && records[0].endpoint == endpoint);
        assert(records[0].severity == OMNI_USD_LIVE_SEVERITY_ERROR);
        assert(records[0].fields.at("operation").number == OMNI_USD_LIVE_OPERATION_UPDATE);
        assert(records[0].fields.at("grpc_status").number == 14);
        const auto name = records[0].fields.find("canonical_resource");
        assert((name != records[0].fields.end()) == (allowed && requested));
        if (name != records[0].fields.end()) assert(name->second.kind == OMNI_USD_LIVE_FIELD_STRING && name->second.text == resource);
        const auto plain = registry.snapshot(endpoint, 100);
        for (const auto& entry : plain.records) assert(!entry.resource[0]);
        const auto opted = registry.snapshot(endpoint, 100, OMNI_USD_LIVE_DIAGNOSTIC_RESOURCE_NAMES);
        assert(!opted.records.empty());
        assert(std::string(opted.records.back().resource.data()) == (allowed ? resource : ""));
        finish(registry);
    }
}

void invalidResourcesAndCatalogFieldsNeverEscape()
{
    auto config = options();
    config.diagnostic_allowed_flags = OMNI_USD_LIVE_DIAGNOSTIC_RESOURCE_NAMES | OMNI_USD_LIVE_DIAGNOSTIC_SERVER_DETAIL;
    DiagnosticsRegistry registry(config, instance);
    Sink sink;
    auto subscription = sinkOptions(); subscription.flags = config.diagnostic_allowed_flags;
    const auto handle = registry.subscribe(subscription, &sink, Sink::callback);
    const std::vector<std::string> invalid{
        "https://signed.example/path?token=secret-canary",
        "live://name:secret-canary@host/session/layer",
        resource + "?sig=secret-canary", resource + "#secret-canary",
        resource + "\nsecret-canary", resource + "\\secret-canary",
        "live://host/session/" + std::string(513, 'x')};
    for (size_t i = 0; i < invalid.size(); ++i)
        registry.emit(eventCode, endpoint, 0, 0, {{DiagnosticKey::Count, i}}, invalid[i]);
    sink.wait(invalid.size());
    assert(registry.unsubscribe(handle, 3000) == OMNI_USD_LIVE_OK);
    for (const auto& entry : sink.copy()) {
        assert(entry.fields.size() == 1 && entry.fields.count("count") == 1);
        for (const auto& field : entry.fields) assert(field.second.text.empty());
    }
    auto snapshot = registry.snapshot(0, 100, subscription.flags);
    assert(snapshot.metrics.invalid == invalid.size() && snapshot.records.size() == invalid.size());
    for (const auto& entry : snapshot.records) assert(!entry.resource[0] && entry.fieldCount == 1);
    registry.emit(eventCode, endpoint, 0, 0, {{DiagnosticKey::Count, 1}, {DiagnosticKey::Count, 2}});
    registry.emit(eventCode, endpoint, 0, 0, {{DiagnosticKey::CountKeys, 1}});
    registry.emit(0, endpoint);
    snapshot = registry.snapshot(0, 100, subscription.flags);
    assert(snapshot.metrics.invalid == invalid.size() + 3 && snapshot.metrics.emitted == invalid.size());
    assert(snapshot.records.size() == invalid.size());
    finish(registry);
}

void reservedEventCodeIsNeverEmittedOrSubscribable()
{
    // Record 23 is reserved (design section 16): it keeps codes 24-29 stable and is never emitted or selectable.
    DiagnosticsRegistry registry(options(), instance);
    Sink sink;
    uint32_t selected = 23;
    auto subscription = sinkOptions();
    subscription.event_code_count = 1; subscription.event_codes = &selected;
    bool invalid = false;
    try { (void)registry.subscribe(subscription, &sink, Sink::callback); }
    catch (OmniUsdLiveResult result) { invalid = result == OMNI_USD_LIVE_INVALID_ARGUMENT; }
    assert(invalid);
    assert(std::string_view(DiagnosticsRegistry::eventName(23)) == "invalid");
    registry.emit(23, endpoint);
    const auto snapshot = registry.snapshot(0, 100);
    assert(snapshot.metrics.events[23] == 0 && snapshot.metrics.emitted == 0 && snapshot.metrics.invalid == 1);
    assert(snapshot.records.empty());
    finish(registry);
}
void boundedOverflowDoesNotBlockOperationalWork()
{
    auto config = options(); config.max_diagnostic_sinks = 2; config.max_diagnostic_sink_records = 2;
    DiagnosticsRegistry registry(config, instance);
    Sink first, second;
    std::promise<void> firstEntered, secondEntered, resume;
    auto readyA = firstEntered.get_future(), readyB = secondEntered.get_future();
    auto gate = resume.get_future().share();
    first.action = [&](const Seen& seen) { if (seen.sequence == 1) { firstEntered.set_value(); gate.wait(); } };
    second.action = [&](const Seen&) { secondEntered.set_value(); gate.wait(); };
    const auto a = registry.subscribe(sinkOptions(endpoint), &first, Sink::callback);
    const auto b = registry.subscribe(sinkOptions(otherEndpoint), &second, Sink::callback);
    Sink rejected;
    bool full = false;
    try { registry.subscribe(sinkOptions(), &rejected, Sink::callback); }
    catch (OmniUsdLiveResult result) { full = result == OMNI_USD_LIVE_RESOURCE_EXHAUSTED; }
    assert(full);
    registry.emit(eventCode, endpoint); registry.emit(eventCode, otherEndpoint);
    assert(readyA.wait_for(std::chrono::seconds(3)) == std::future_status::ready);
    assert(readyB.wait_for(std::chrono::seconds(3)) == std::future_status::ready);
    for (unsigned i = 0; i < 4; ++i) registry.emit(eventCode, endpoint);
    const auto blocked = registry.snapshot(0, 100);
    assert(blocked.metrics.sinkDropped == 2 && blocked.metrics.emitted == 6 && blocked.records.size() == 6);
    struct Work final : DispatchJob {
        std::promise<void> completed;
        void dispatch() noexcept override { completed.set_value(); }
    };
    DispatchExecutor operational(1);
    auto work = std::make_shared<Work>();
    auto complete = work->completed.get_future();
    operational.enqueue(*work);
    assert(complete.wait_for(std::chrono::seconds(3)) == std::future_status::ready);
    resume.set_value();
    first.wait(3); second.wait(1);
    assert(registry.unsubscribe(a, 3000) == OMNI_USD_LIVE_OK && registry.unsubscribe(b, 3000) == OMNI_USD_LIVE_OK);
    const auto records = first.copy();
    assert(records.size() == 3 && records[0].sequence == 1 && records[1].sequence == 3 && records[2].sequence == 4);
    assert(second.copy().size() == 1);
    assert(rejected.copy().empty());
    operational.beginShutdown();
    assert(operational.waitForShutdown(Clock::now() + std::chrono::seconds(3)));
    finish(registry);
}

void throwingSinkDisablesWithoutRecursiveRecords()
{
    DiagnosticsRegistry registry(options(), instance);
    Sink sink;
    sink.action = [](const Seen& seen) {
        if (seen.sequence % 2) throw std::runtime_error("sink-secret-canary-must-not-be-recorded");
    };
    auto lease = std::make_shared<int>(3);
    std::weak_ptr<int> leaseLifetime = lease;
    const auto handle = registry.subscribe(sinkOptions(), &sink, Sink::callback, lease);
    lease.reset();
    // Successes between throws must not reset the three-total-failures limit.
    for (unsigned i = 1; i <= 5; ++i) {
        registry.emit(eventCode, endpoint);
        sink.wait(i);
        eventually([&] { return registry.snapshot(0, 0).metrics.sinkFailures == (i + 1) / 2; });
    }
    eventually([&] { return registry.snapshot(0, 0).metrics.sinksDisabled == 1; });
    eventually([&] { return leaseLifetime.expired(); });
    assert(registry.hasSinks()); // Disabled handle remains explicit ownership.
    registry.emit(eventCode, endpoint);
    assert(registry.unsubscribe(handle, 3000) == OMNI_USD_LIVE_OK);
    assert(sink.copy().size() == 5);
    const auto snapshot = registry.snapshot(0, 100);
    assert(snapshot.metrics.emitted == 6 && snapshot.metrics.sinkFailures == 3 && snapshot.metrics.sinksDisabled == 1);
    assert(snapshot.records.size() == 6 && snapshot.records.back().sequence == 6);
    for (const auto& entry : snapshot.records) assert(entry.eventCode == eventCode && entry.fieldCount == 0 && !entry.resource[0]);
    assert(registry.unsubscribe(handle, 0) == OMNI_USD_LIVE_INVALID_HANDLE);
    finish(registry);
}

void selfUnsubscribeAndLeaseRetirement()
{
    DiagnosticsRegistry registry(options(), instance);
    Sink sink;
    OmniUsdLiveCallbackHandle handle = 0;
    std::promise<void> entered, resume;
    auto enteredFuture = entered.get_future();
    auto gate = resume.get_future().share();
    sink.action = [&](const Seen&) {
        assert(registry.unsubscribe(handle, 1000) == OMNI_USD_LIVE_BUSY);
        entered.set_value(); gate.wait();
    };
    handle = registry.subscribe(sinkOptions(), &sink, Sink::callback);
    registry.emit(eventCode, endpoint);
    assert(enteredFuture.wait_for(std::chrono::seconds(3)) == std::future_status::ready);
    assert(registry.unsubscribe(handle, 0) == OMNI_USD_LIVE_TIMEOUT);
    registry.emit(eventCode, endpoint);
    resume.set_value();
    assert(registry.unsubscribe(handle, 3000) == OMNI_USD_LIVE_OK && sink.copy().size() == 1);
    assert(!registry.hasSinks());

    struct Barrier { std::promise<void> entered, resume; std::shared_future<void> gate = resume.get_future().share(); };
    struct Lease {
        DiagnosticsRegistry* registry;
        std::shared_ptr<Barrier> barrier;
        ~Lease() {
            assert(registry->snapshot(0, 0).runtimeInstanceId == instance);
            barrier->entered.set_value(); barrier->gate.wait();
        }
    };
    auto barrier = std::make_shared<Barrier>();
    auto retiring = barrier->entered.get_future();
    auto lease = std::shared_ptr<Lease>(new Lease{&registry, barrier});
    Sink idle;
    const auto retainedHandle = registry.subscribe(sinkOptions(), &idle, Sink::callback, lease);
    lease.reset();
    auto removed = std::async(std::launch::async, [&] { return registry.unsubscribe(retainedHandle, 3000); });
    assert(retiring.wait_for(std::chrono::seconds(3)) == std::future_status::ready);
    assert(removed.wait_for(std::chrono::seconds(0)) == std::future_status::timeout);
    assert(registry.unsubscribe(retainedHandle, 0) == OMNI_USD_LIVE_TIMEOUT);
    assert(registry.snapshot(0, 0).runtimeInstanceId == instance);
    barrier->resume.set_value();
    assert(removed.wait_for(std::chrono::seconds(3)) == std::future_status::ready && removed.get() == OMNI_USD_LIVE_OK);
    assert(!registry.hasSinks());
    finish(registry);
}

void boundedIncidentAndEndpointChurn()
{
    auto config = options(); config.max_diagnostic_records = 4;
    config.diagnostic_allowed_flags = OMNI_USD_LIVE_DIAGNOSTIC_RESOURCE_NAMES;
    DiagnosticsRegistry registry(config, instance);
    for (unsigned i = 0; i < 128; ++i)
        registry.emit(eventCode, 1000 + i, i + 1, 0, {}, resource + std::to_string(i));
    const auto history = registry.snapshot(0, 100);
    assert(history.records.size() == 4 && history.metrics.emitted == 128 && history.metrics.events[eventCode] == 128);
    assert(history.metrics.historyOverwritten == 124);
    assert(registry.snapshot(1000, 100).records.empty());
    assert(registry.snapshot(1000, 0).metrics.emitted == 128);
    auto closed = record<OmniUsdLiveStatus>();
    closed.subject_kind = OMNI_USD_LIVE_SUBJECT_LAYER; closed.state = OMNI_USD_LIVE_STATE_CLOSED;
    closed.result = OMNI_USD_LIVE_OUTCOME_UNKNOWN; closed.has_unconfirmed_edits = 1;
    closed.subject_id = 101; closed.incident_id = 501;
    registry.retainClosedIncident(endpoint, resource, closed);
    closed.subject_id = 102; closed.incident_id = 502;
    registry.retainClosedIncident(otherEndpoint, resource + "-2", closed);
    closed.subject_id = 101; closed.incident_id = 501; closed.pending_write_count = 7;
    registry.retainClosedIncident(endpoint, resource, closed);
    auto incidents = registry.snapshot(0, 0, OMNI_USD_LIVE_DIAGNOSTIC_RESOURCE_NAMES);
    assert(incidents.closedIncidents.size() == 2 && incidents.metrics.closedIncidentsEvicted == 0);
    bool replaced = false;
    for (const auto& item : incidents.closedIncidents)
        if (item.status.subject_id == 101) { replaced = true; assert(item.status.pending_write_count == 7); }
    assert(replaced);
    closed.subject_id = 103; closed.incident_id = 503;
    registry.retainClosedIncident(endpoint, resource + "-3", closed);
    incidents = registry.snapshot(0, 0);
    assert(incidents.closedIncidents.size() == 2 && incidents.metrics.closedIncidentsEvicted == 1);
    for (const auto& item : incidents.closedIncidents) assert(item.status.subject_id != 101 && !item.resource[0]);
    const auto filtered = registry.snapshot(otherEndpoint, 0, OMNI_USD_LIVE_DIAGNOSTIC_RESOURCE_NAMES);
    assert(filtered.closedIncidents.size() == 1 && filtered.closedIncidents[0].status.subject_id == 102);
    assert(std::string(filtered.closedIncidents[0].resource.data()) == resource + "-2");
    assert(filtered.metrics.closedIncidentsEvicted == 1);
    const auto capped = registry.snapshot(0, 0, 0, 1);
    assert(capped.truncated && capped.closedIncidents.size() == 1 && capped.closedIncidents[0].status.subject_id == 103);
    finish(registry);
}

void messageMetricsIgnoreHistoryFilteringAndSaturate()
{
    auto now = Clock::time_point{};
    auto clock = [](void* context) noexcept { return *static_cast<Clock::time_point*>(context); };
    auto config = options(); config.flags = OMNI_USD_LIVE_RUNTIME_DISABLE_DIAGNOSTIC_HISTORY;
    config.diagnostic_min_severity = OMNI_USD_LIVE_SEVERITY_ERROR;
    DiagnosticsRegistry registry(config, instance, clock, &now);
    Sink sink;
    auto subscription = sinkOptions();
    uint32_t selected = OMNI_USD_LIVE_RECORD_MESSAGE_REJECTED;
    subscription.event_code_count = 1; subscription.event_codes = &selected;
    const auto handle = registry.subscribe(subscription, &sink, Sink::callback);
    for (uint32_t code = OMNI_USD_LIVE_RECORD_MESSAGE_ACCEPTED; code <= OMNI_USD_LIVE_RECORD_MESSAGE_REJECTED; ++code) {
        registry.emit(code, endpoint, 101, 0, {{DiagnosticKey::Bytes, UINT64_MAX - 2}});
        registry.emit(code, endpoint, 101, 0, {{DiagnosticKey::Bytes, 4}});
        registry.emit(code, endpoint, 101, 0, {{DiagnosticKey::Bytes, 4}});
        registry.emit(code, endpoint, 101, 0, {{DiagnosticKey::Bytes, 0}});
    }
    // WARN duplicates coalesce on a fixed clock; filtered successes and all
    // zero-byte messages still count. No fake payload allocation is needed.
    sink.wait(3); // First, changed, changed zero-byte; summary filtered by code.
    assert(registry.unsubscribe(handle, 3000) == OMNI_USD_LIVE_OK);
    const auto snapshot = registry.snapshot(0, 100);
    assert(snapshot.records.empty() && snapshot.metrics.warningSuppressed == 2);
    for (size_t i = 0; i < snapshot.metrics.messageBytes.size(); ++i) {
        assert(snapshot.metrics.messageBytes[i] == UINT64_MAX);
        assert(snapshot.metrics.events[OMNI_USD_LIVE_RECORD_MESSAGE_ACCEPTED + i] == 4);
    }
    assert(snapshot.metrics.rpcAttempts == 0 && snapshot.metrics.rpcCompletions == 0);
    const auto delivered = sink.copy();
    assert(delivered.size() == 3);
    for (const auto& record : delivered)
        assert(record.code == OMNI_USD_LIVE_RECORD_MESSAGE_REJECTED);
    finish(registry);
}

void deterministicWarningSuppressionAndSummaries()
{
    auto now = Clock::time_point{} + std::chrono::seconds(50);
    auto clock = [](void* context) noexcept { return *static_cast<Clock::time_point*>(context); };
    DiagnosticsRegistry registry(options(), instance, clock, &now);
    Sink sink;
    const auto handle = registry.subscribe(sinkOptions(), &sink, Sink::callback);
    constexpr uint32_t warning = OMNI_USD_LIVE_RECORD_STREAM_DISCONNECTED;
    auto emit = [&](uint64_t count) {
        registry.emit(warning, endpoint, 101, 0, {{DiagnosticKey::Count, count}});
    };
    emit(1); emit(1); emit(1);
    auto snapshot = registry.snapshot(0, 100);
    assert(snapshot.metrics.emitted == 3 && snapshot.metrics.events[warning] == 3);
    assert(snapshot.metrics.warningSuppressed == 2 && snapshot.metrics.suppressed == 2);
    assert(snapshot.records.size() == 1 && snapshot.records[0].sequence == 1 && snapshot.lastSequence == 3);
    emit(2); // A differing value flushes a summary before the new warning.
    snapshot = registry.snapshot(0, 100);
    assert(snapshot.records.size() == 3 && snapshot.records[1].sequence == 4 && snapshot.records[2].sequence == 5);
    const auto& summary = snapshot.records[1];
    assert(summary.eventCode == OMNI_USD_LIVE_RECORD_RECORDS_SUPPRESSED && summary.endpoint == endpoint && summary.subjectId == 101);
    bool count = false, reason = false;
    for (uint32_t i = 0; i < summary.fieldCount; ++i) {
        if (summary.fields[i].key == DiagnosticKey::Count) { count = true; assert(summary.fields[i].value == 2); }
        if (summary.fields[i].key == DiagnosticKey::Reason) { reason = true; assert(summary.fields[i].value == warning); }
    }
    assert(count && reason);
    emit(2); // Still within the new window.
    now += std::chrono::milliseconds(1000);
    emit(2); // Exact window boundary is eligible; its pending summary comes first.
    snapshot = registry.snapshot(0, 100);
    assert(snapshot.records.size() == 5 && snapshot.records[3].sequence == 7 && snapshot.records[4].sequence == 8);
    assert(snapshot.records[3].eventCode == OMNI_USD_LIVE_RECORD_RECORDS_SUPPRESSED);
    assert(snapshot.metrics.emitted == 8 && snapshot.metrics.events[warning] == 6);
    assert(snapshot.metrics.events[OMNI_USD_LIVE_RECORD_RECORDS_SUPPRESSED] == 2 && snapshot.metrics.warningSuppressed == 3);
    registry.emit(warning, otherEndpoint, 101, 0, {{DiagnosticKey::Count, 2}}); // Distinct identity is never suppressed.
    registry.emit(warning, endpoint, 101, 0, {}, {}, OMNI_USD_LIVE_SEVERITY_ERROR);
    registry.emit(warning, endpoint, 101, 0, {}, {}, OMNI_USD_LIVE_SEVERITY_ERROR);
    sink.wait(8);
    assert(registry.unsubscribe(handle, 3000) == OMNI_USD_LIVE_OK);
    const auto delivered = sink.copy();
    const std::array<uint64_t, 8> sequences{1, 4, 5, 7, 8, 9, 10, 11};
    assert(delivered.size() == sequences.size());
    for (size_t i = 0; i < sequences.size(); ++i) assert(delivered[i].sequence == sequences[i]);
    snapshot = registry.snapshot(0, 100);
    assert(snapshot.metrics.warningSuppressed == 3 && snapshot.lastSequence == 11);
    finish(registry);
}
} // namespace

void operationAuthWaitMetrics()
{
    auto config = options(); config.flags = OMNI_USD_LIVE_RUNTIME_DISABLE_DIAGNOSTIC_HISTORY;
    DiagnosticsRegistry registry(config, instance);
    const auto start = Clock::time_point{} + std::chrono::seconds(10);
    for (size_t kind = 0; kind < 3; ++kind) {
        const auto category = static_cast<AuthWaitKind>(kind);
        OperationAuthWait first(registry, category), second(registry, category);
        first.finish(OMNI_USD_LIVE_OK, start); // No synthetic wait before a park.
        first.start(start); first.start(start + std::chrono::milliseconds(7));
        second.start(start + std::chrono::milliseconds(5));
        auto metric = registry.snapshot(otherEndpoint, 0).metrics.operationAuthWaits[kind];
        assert(metric.current == 2 && metric.highWater == 2 && metric.started == 2 && !metric.completed);
        first.finish(OMNI_USD_LIVE_CANCELLED, start + std::chrono::milliseconds(11));
        first.finish(OMNI_USD_LIVE_OK, start + std::chrono::milliseconds(12));
        metric = registry.snapshot(0, 0).metrics.operationAuthWaits[kind];
        assert(metric.current == 1 && metric.completed == 1);
        assert(metric.durationByResult[OMNI_USD_LIVE_CANCELLED].count == 1);
        assert(metric.durationByResult[OMNI_USD_LIVE_CANCELLED].sumMs == 11);
        second.finish(OMNI_USD_LIVE_OK, start + std::chrono::milliseconds(15));
        // A later rejection is a distinct episode; a genuine zero is retained.
        first.start(start + std::chrono::milliseconds(20));
        first.finish(OMNI_USD_LIVE_UNAUTHENTICATED, start + std::chrono::milliseconds(20));
        registry.startOperationAuthWait(category);
        registry.endOperationAuthWait(category, OMNI_USD_LIVE_TIMEOUT, UINT64_MAX);
        registry.startOperationAuthWait(category);
        registry.endOperationAuthWait(category, OMNI_USD_LIVE_TIMEOUT, 1);
        registry.startOperationAuthWait(category);
        registry.endOperationAuthWait(category, static_cast<OmniUsdLiveResult>(99), 5);
        metric = registry.snapshot(0, 0).metrics.operationAuthWaits[kind];
        assert(metric.current == 0 && metric.highWater == 2 && metric.started == 6 && metric.completed == 6);
        assert(metric.durationByResult[OMNI_USD_LIVE_OK].count == 1 && metric.durationByResult[OMNI_USD_LIVE_OK].sumMs == 10);
        assert(metric.durationByResult[OMNI_USD_LIVE_UNAUTHENTICATED].counts[0] == 1);
        assert(metric.durationByResult[OMNI_USD_LIVE_TIMEOUT].count == 2);
        assert(metric.durationByResult[OMNI_USD_LIVE_TIMEOUT].sumMs == UINT64_MAX);
        assert(metric.durationByResult[22].count == 1 && metric.durationByResult[22].sumMs == 5);
    }
    registry.beginShutdown();
    { OperationAuthWait abandoned(registry, AuthWaitKind::Finite); abandoned.start(Clock::now()); }
    const auto snapshot = registry.snapshot(0, 0);
    assert(!snapshot.metrics.operationAuthWaits[0].current);
    assert(snapshot.metrics.operationAuthWaits[0].durationByResult[OMNI_USD_LIVE_INTERNAL_ERROR].count == 1);
    assert(snapshot.records.empty() && !snapshot.metrics.emitted);
    finish(registry);
}

void diagnosticsRegistryTests()
{
    diagnosticCatalogContract();
    catalogRpcSeverityAndOptionalEvidence();
    rpcServiceLimitFields();
    executorMetricsTrackConcurrentRequeue();
    executorCompletionDoesNotImplyOwnerQuiescence();
    cqTagMetricsFollowHandoffReceiptAndDestruction();
    historyCountAndByteLimits();
    rpcMetricsWithoutHistoryOrDelivery();
    completionDurationsWithoutHistory();
    applyLagMetricsCountEveryUpdate();
    shutdownObservationsNeverDispatch();
    authMetricsWithoutHistory();
    operationAuthWaitMetrics();
    readinessMetricsWithVirtualTime();
    backpressureMetricsWithVirtualTime();
    sinkFiltersAndResourcePermissions();
    invalidResourcesAndCatalogFieldsNeverEscape();
    reservedEventCodeIsNeverEmittedOrSubscribable();
    boundedOverflowDoesNotBlockOperationalWork();
    throwingSinkDisablesWithoutRecursiveRecords();
    selfUnsubscribeAndLeaseRetirement();
    boundedIncidentAndEndpointChurn();
    deterministicWarningSuppressionAndSummaries();
    messageMetricsIgnoreHistoryFilteringAndSaturate();
    std::cout << "diagnostics registry: bounded history/sinks, metrics, filters, failures, and quiescence passed\n";
}
