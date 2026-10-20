// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "LiveDiagnostics.h"
#include "LiveAuth.h"
#include "LiveRuntime.h"
#include "LiveRequest.h"
#include "LiveStatus.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <future>
#include <iostream>
#include <map>
#include <string>
#include <stdexcept>
#include <thread>
#include <vector>
#if defined(__unix__)
#include <sys/mman.h>
#include <unistd.h>
#endif

using namespace omni::usd::live;
namespace {
template <class T> T record()
{
    T value{}; value.struct_size = sizeof(value); value.abi_version = OMNI_USD_LIVE_ABI_VERSION;
    return value;
}
constexpr const char* tokenCanary = "synthetic-DIAGNOSTIC-TOKEN-CANARY-729";
// Optional test-only corpus. All calls are on the diagnostics API test owner.
// Each frame embeds the exact bytes returned by the public copy API.
class SnapshotCorpus {
    std::ofstream output;
    size_t count = 0, bytes = 0;
public:
    const char* scenario = nullptr;
    SnapshotCorpus()
    {
        if (const auto* path = std::getenv("LIVE_TEST_DIAGNOSTIC_CORPUS")) {
            output.open(path, std::ios::binary | std::ios::trunc);
            assert(output.is_open() && output.good());
        }
    }
    void append(const std::string& snapshot)
    {
        if (!output.is_open()) return;
        assert(scenario);
        const auto frame = std::string("{\"scenario\":\"") + scenario + "\",\"snapshot\":" + snapshot + "}\n";
        bytes += frame.size();
        assert(++count <= 1024 && bytes <= 64 * 1024 * 1024 - 128);
        output << frame;
        output.flush();
        assert(output.good());
    }
    void finish()
    {
        if (!output.is_open()) return;
        output << "{\"complete\":true,\"count\":" << count << "}\n";
        output.close();
        assert(!output.fail());
    }
};
SnapshotCorpus& corpus() { static SnapshotCorpus value; return value; }
struct Fixture {
    std::shared_ptr<Runtime> runtime;
    OmniUsdLiveEndpoint a = 0, b = 0;
    explicit Fixture(bool names = false, bool disabled = false, uint64_t byteLimit = 0, uint32_t subjectLimit = 0)
    {
        auto options = record<OmniUsdLiveRuntimeOptions>();
        options.diagnostic_allowed_flags = names ? OMNI_USD_LIVE_DIAGNOSTIC_RESOURCE_NAMES : 0;
        options.flags = disabled ? OMNI_USD_LIVE_RUNTIME_DISABLE_DIAGNOSTIC_HISTORY : 0;
        options.max_diagnostic_records = 32;
        options.max_pending_bytes_per_endpoint = byteLimit;
        options.max_status_subjects = subjectLimit;
        assert(omniUsdLiveInitialize(&options) == OMNI_USD_LIVE_OK);
        auto endpointOptions = record<OmniUsdLiveEndpointOptions>();
        endpointOptions.authority = "127.0.0.1:34341"; endpointOptions.access_token = tokenCanary;
        assert(omniUsdLiveRegisterEndpoint(&endpointOptions, &a) == OMNI_USD_LIVE_OK);
        endpointOptions.authority = "127.0.0.1:34342";
        assert(omniUsdLiveRegisterEndpoint(&endpointOptions, &b) == OMNI_USD_LIVE_OK);
        runtime = getRuntime();
    }
    ~Fixture()
    {
        assert(omniUsdLiveUnregisterEndpoint(a, 3000) == OMNI_USD_LIVE_OK);
        assert(omniUsdLiveUnregisterEndpoint(b, 3000) == OMNI_USD_LIVE_OK);
        runtime.reset();
        assert(omniUsdLiveShutdown(3000) == OMNI_USD_LIVE_OK);
    }
};
std::string copy(OmniUsdLiveEndpoint endpoint = 0, const OmniUsdLiveDiagnosticExportOptions* options = nullptr)
{
    for (unsigned attempt = 0; attempt < 4; ++attempt) {
        uint64_t required = UINT64_MAX;
        assert(omniUsdLiveCopyDiagnostics(endpoint, options, nullptr, &required) == OMNI_USD_LIVE_OK);
        assert(required > 1 && required <= 16 * 1024 * 1024);
        std::vector<char> bytes(static_cast<size_t>(required) + 16, '~');
        auto size = required;
        const auto result = omniUsdLiveCopyDiagnostics(endpoint, options, bytes.data(), &size);
        if (result == OMNI_USD_LIVE_BUFFER_TOO_SMALL) {
            assert(std::all_of(bytes.begin(), bytes.end(), [](char c) { return c == '~'; }));
            continue;
        }
        assert(result == OMNI_USD_LIVE_OK && size > 1 && size <= required);
        assert(bytes[size - 1] == 0 && std::strlen(bytes.data()) == size - 1);
        assert(std::all_of(bytes.begin() + static_cast<size_t>(size), bytes.end(), [](char c) { return c == '~'; }));
        auto snapshot = std::string(bytes.data(), static_cast<size_t>(size - 1));
        corpus().append(snapshot);
        return snapshot;
    }
    assert(false && "local diagnostic export kept growing during bounded size-query retries");
    return {};
}
void contains(const std::string& text, const std::string& expected)
{
    if (text.find(expected) == std::string::npos)
        std::cerr << "diagnostics ABI missing expected text: " << expected << "\nsynthetic export: " << text << '\n';
    assert(text.find(expected) != std::string::npos);
}
void excludes(const std::string& text, const std::string& unexpected)
{ assert(text.find(unexpected) == std::string::npos); }
std::string section(const std::string& json, const char* first, const char* next)
{
    const auto begin = json.find(first);
    assert(begin != std::string::npos);
    const auto end = json.find(next, begin + std::strlen(first));
    assert(end != std::string::npos);
    return json.substr(begin, end - begin);
}
std::string subject(uint64_t id) { return "\"subject_id\":" + std::to_string(id) + ","; }
void OMNI_USD_LIVE_CALL count(void* user, const OmniUsdLiveDiagnosticRecord*)
{ ++*static_cast<std::atomic<unsigned>*>(user); }

void executorMetricsExportWithoutHistory()
{
    Fixture fixture(false, true);
    struct Held final : DispatchJob {
        std::promise<void> entered, release;
        std::future<void> ready = entered.get_future();
        std::shared_future<void> proceed = release.get_future().share();
        void dispatch() noexcept override {
            entered.set_value();
            assert(proceed.wait_for(std::chrono::seconds(3)) == std::future_status::ready);
        }
        static void OMNI_USD_LIVE_CALL callback(void* user, const OmniUsdLiveDiagnosticRecord*) {
            static_cast<Held*>(user)->dispatch();
        }
        void wait() { assert(ready.wait_for(std::chrono::seconds(3)) == std::future_status::ready); }
    };
    auto operational = std::make_shared<Held>();
    Held diagnostic;
    fixture.runtime->requests->executor().enqueue(*operational);
    operational->wait();
    OmniUsdLiveCallbackHandle sink = 0;
    assert(omniUsdLiveRegisterDiagnosticCallback(nullptr, &diagnostic, Held::callback, &sink) == OMNI_USD_LIVE_OK);
    fixture.runtime->diagnostics->emit(OMNI_USD_LIVE_RECORD_ENDPOINT_REGISTERED, fixture.a);
    diagnostic.wait();
    // Both executors have actual held work. Endpoint filtering and disabled
    // history must leave these runtime-wide metrics available.
    const auto json = copy(fixture.b);
    contains(json, "\"recent_records\":[]");
    const auto operations = section(json, "\"operational\":{", ",\"diagnostic\":{");
    const auto diagnostics = section(json, "\"diagnostic\":{", ",\"closed_incidents\":");
    for (const auto* metrics : {&operations, &diagnostics}) {
        contains(*metrics, "\"running\":1,");
        contains(*metrics, "\"queued\":0,");
        contains(*metrics, "\"configured_workers\":2,");
        contains(*metrics, "\"live_workers\":2,");
        contains(*metrics, "\"queue_wait_ms\":{\"bounds\":[1,5,10,50,100,500,1000,5000],\"count\":1,");
        contains(*metrics, "\"dispatch_duration_ms\":{\"bounds\":[1,5,10,50,100,500,1000,5000],\"count\":0,");
    }
    operational->release.set_value(); diagnostic.release.set_value();
    assert(omniUsdLiveUnregisterCallback(sink, 3000) == OMNI_USD_LIVE_OK);
}

void callbackFailuresPreserveOutcomes()
{
    for (const bool disabled : {false, true}) {
        Fixture fixture(false, disabled);
        struct HeldSink {
            std::promise<void> entered, release;
            std::shared_future<void> gate = release.get_future().share();
            static void OMNI_USD_LIVE_CALL callback(void* data, const OmniUsdLiveDiagnosticRecord*) {
                auto& self = *static_cast<HeldSink*>(data);
                self.entered.set_value(); self.gate.wait();
            }
        } held;
        auto sinkOptions = record<OmniUsdLiveDiagnosticOptions>();
        const uint32_t selected = OMNI_USD_LIVE_RECORD_CALLBACK_FAILED;
        sinkOptions.event_codes = &selected; sinkOptions.event_code_count = 1;
        OmniUsdLiveCallbackHandle sink = 0;
        assert(omniUsdLiveRegisterDiagnosticCallback(&sinkOptions, &held, HeldSink::callback, &sink) == OMNI_USD_LIVE_OK);
        auto request = fixture.runtime->requests->create(fixture.a, OMNI_USD_LIVE_OPERATION_UPDATE, {});
        auto error = record<OmniUsdLiveRequestError>();
        error.result = OMNI_USD_LIVE_OUTCOME_UNKNOWN; error.attempt_id = 17; error.incident_id = 19;
        request->finish(error, [] { throw std::runtime_error(tokenCanary); });
        assert(omniUsdLiveWait(request->handle, 3000) == OMNI_USD_LIVE_OK);
        assert(held.entered.get_future().wait_for(std::chrono::seconds(3)) == std::future_status::ready);
        auto actual = record<OmniUsdLiveRequestError>();
        assert(omniUsdLiveGetRequestError(request->handle, &actual) == OMNI_USD_LIVE_OK);
        assert(actual.result == error.result && actual.attempt_id == 17 && actual.incident_id == 19);
        assert(request->callbackThrew());

        struct ThrowingAuth {
            std::mutex mutex; std::condition_variable changed;
            std::vector<OmniUsdLiveAuthRequired> events;
            static void OMNI_USD_LIVE_CALL callback(void* data, const OmniUsdLiveAuthRequired* event) {
                auto& self = *static_cast<ThrowingAuth*>(data);
                { std::lock_guard<std::mutex> lock(self.mutex); self.events.push_back(*event); self.changed.notify_all(); }
                throw std::runtime_error(tokenCanary);
            }
            void wait(size_t count) {
                std::unique_lock<std::mutex> lock(mutex);
                assert(changed.wait_for(lock, std::chrono::seconds(3), [&] { return events.size() == count; }));
            }
        } auth;
        OmniUsdLiveCallbackHandle authHandle = 0;
        assert(omniUsdLiveRegisterAuthRequiredCallback(fixture.a, &auth, ThrowingAuth::callback, &authHandle) == OMNI_USD_LIVE_OK);
        auto endpoint = fixture.runtime->findEndpoint(fixture.a);
        fixture.runtime->auth->completed(*endpoint, 1, grpc::StatusCode::UNAUTHENTICATED);
        auth.wait(1);
        uint64_t generation = 0;
        assert(omniUsdLiveSetAccessToken(fixture.a, 1, tokenCanary, 0, &generation) == OMNI_USD_LIVE_OK && generation == 2);
        fixture.runtime->auth->completed(*endpoint, 2, grpc::StatusCode::UNAUTHENTICATED);
        auth.wait(2);
        assert(omniUsdLiveUnregisterCallback(authHandle, 3000) == OMNI_USD_LIVE_OK);
        assert(auth.events[0].challenge_id != auth.events[1].challenge_id);
        auto authState = record<OmniUsdLiveAuthState>();
        assert(omniUsdLiveGetAuthState(fixture.a, &authState) == OMNI_USD_LIVE_OK);
        assert(authState.state == OMNI_USD_LIVE_STATE_AUTH_WAIT && authState.generation == 2);

        struct ThrowingStatus {
            std::promise<void> entered;
            uint64_t subject = 0;
            OmniUsdLiveStatus delivered{};
            static void OMNI_USD_LIVE_CALL callback(void* data, const OmniUsdLiveStatusEvent* event) {
                auto& self = *static_cast<ThrowingStatus*>(data);
                if (event->status->subject_id != self.subject) return;
                self.delivered = *event->status; self.entered.set_value();
                throw std::runtime_error(tokenCanary);
            }
        } status;
        const char* layerUri = "live://127.0.0.1:34342/callback/root.live";
        auto layer = record<OmniUsdLiveStatus>();
        layer.subject_kind = OMNI_USD_LIVE_SUBJECT_LAYER; layer.state = OMNI_USD_LIVE_STATE_READY;
        status.subject = fixture.runtime->statuses->add(fixture.b, 31337, layerUri, layer);
        layer.state = OMNI_USD_LIVE_STATE_FAILED; layer.result = OMNI_USD_LIVE_OUTCOME_UNKNOWN;
        layer.recovery_action = OMNI_USD_LIVE_RECOVERY_PRESERVE_AND_REOPEN;
        layer.pending_write_count = 2; layer.pending_bytes = 71; layer.has_unconfirmed_edits = 1;
        assert(fixture.runtime->statuses->publish(status.subject, layer) == OMNI_USD_LIVE_OK);
        OmniUsdLiveCallbackHandle statusHandle = 0;
        assert(omniUsdLiveRegisterStatusCallback(fixture.b, &status, ThrowingStatus::callback, &statusHandle) == OMNI_USD_LIVE_OK);
        assert(status.entered.get_future().wait_for(std::chrono::seconds(3)) == std::future_status::ready);
        assert(omniUsdLiveUnregisterCallback(statusHandle, 3000) == OMNI_USD_LIVE_OK);
        auto current = record<OmniUsdLiveStatus>();
        assert(omniUsdLiveGetLayerStatus(layerUri, &current) == OMNI_USD_LIVE_OK && current.incident_id);
        assert(current.revision == status.delivered.revision && current.result == status.delivered.result &&
               current.state == status.delivered.state && current.incident_id == status.delivered.incident_id);
        assert(current.result == layer.result && current.pending_write_count == 2 && current.pending_bytes == 71 &&
               current.has_unconfirmed_edits && current.recovery_action == OMNI_USD_LIVE_RECOVERY_PRESERVE_AND_REOPEN);
        const auto snapshot = fixture.runtime->diagnostics->snapshot(0, 32);
        assert(snapshot.metrics.events[selected] == 4 && snapshot.metrics.sinkFailures == 0);
        if (disabled) assert(snapshot.records.empty());
        else {
            std::array<size_t, 3> kinds{};
            for (const auto& entry : snapshot.records) {
                if (entry.eventCode != selected) continue;
                std::map<DiagnosticKey, uint64_t> fields;
                for (uint32_t i = 0; i < entry.fieldCount; ++i) fields.emplace(entry.fields[i].key, entry.fields[i].value);
                const auto kind = fields.at(DiagnosticKey::CallbackKind);
                assert(kind < kinds.size() && entry.severity == OMNI_USD_LIVE_SEVERITY_ERROR);
                ++kinds[kind];
                if (kind == 0) {
                    assert(entry.endpoint == fixture.a && !entry.subjectId && entry.incidentId == 19);
                    assert(fields.at(DiagnosticKey::Request) == request->handle && fields.at(DiagnosticKey::Attempt) == 17);
                    assert(fields.at(DiagnosticKey::Operation) == OMNI_USD_LIVE_OPERATION_UPDATE);
                } else if (kind == 1) {
                    assert(entry.endpoint == fixture.a && entry.subjectId == fixture.a);
                    assert(fields.at(DiagnosticKey::Callback) == authHandle);
                    const auto& delivered = auth.events[kinds[1] - 1];
                    assert(fields.at(DiagnosticKey::Challenge) == delivered.challenge_id &&
                           fields.at(DiagnosticKey::AuthGeneration) == delivered.rejected_generation);
                } else {
                    assert(entry.endpoint == fixture.b && entry.subjectId == current.subject_id &&
                           entry.incidentId == current.incident_id);
                    assert(fields.at(DiagnosticKey::Callback) == statusHandle &&
                           fields.at(DiagnosticKey::Revision) == current.revision &&
                           fields.at(DiagnosticKey::Session) == 31337);
                }
            }
            assert((kinds == std::array<size_t, 3>{1, 2, 1}));
        }
        const auto json = copy(fixture.b); // Runtime metrics survive endpoint filtering.
        contains(json, "\"event_code\":29,\"count\":4"); excludes(json, tokenCanary);
        // Begin cancellation while the diagnostic callback is still held, so
        // queued records cannot enter it after release. No timeout proves state.
        assert(omniUsdLiveUnregisterCallback(sink, 0) == OMNI_USD_LIVE_TIMEOUT);
        held.release.set_value();
        assert(omniUsdLiveUnregisterCallback(sink, 3000) == OMNI_USD_LIVE_OK);
        assert(fixture.runtime->statuses->remove(status.subject) == OMNI_USD_LIVE_OK);
        assert(omniUsdLiveReleaseRequest(request->handle) == OMNI_USD_LIVE_OK);
        request.reset(); endpoint.reset();
    }
}

void subjectMetricsTrackAttachedLatchedStates()
{
    Fixture fixture(false, true, 0, 4); // Two endpoints and two child slots.
    auto& hub = *fixture.runtime->statuses;
    const auto metrics = [&] { return fixture.runtime->diagnostics->snapshot(0, 0).metrics.subjects; };
    auto initial = record<OmniUsdLiveStatus>();
    initial.subject_kind = OMNI_USD_LIVE_SUBJECT_LAYER;
    initial.state = OMNI_USD_LIVE_STATE_OPENING;
    const auto observer = hub.createObserver(0);
    const auto a = hub.add(fixture.a, 0, "live://opaque/s/alpha.live", initial);
    const auto b = hub.add(fixture.b, 0, "live://opaque/s/beta.live", initial);
    auto values = metrics();
    assert(values[0].current == 2 && values[0].admitted == 2 && !values[0].detached);
    assert(values[2].current == 2 && values[2].highWater == 2 && values[2].admitted == 2);
    assert(values[2].byState[OMNI_USD_LIVE_STATE_OPENING] == 2);
    const auto rejectedAdd = [&](const char* uri, OmniUsdLiveResult expected) {
        try { hub.add(fixture.a, 0, uri, initial); assert(false); }
        catch (OmniUsdLiveResult result) { assert(result == expected); }
        const auto value = metrics()[2];
        assert(value.current == 2 && value.admitted == 2 && !value.detached);
    };
    rejectedAdd("live://opaque/s/alpha.live", OMNI_USD_LIVE_ALREADY_EXISTS);
    rejectedAdd("live://opaque/s/third.live", OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    auto invalid = initial; invalid.state = 0;
    assert(hub.publish(a, invalid) == OMNI_USD_LIVE_INVALID_ARGUMENT);
    for (const auto state : {OMNI_USD_LIVE_STATE_READY, OMNI_USD_LIVE_STATE_RECONNECTING,
                            OMNI_USD_LIVE_STATE_AUTH_WAIT, OMNI_USD_LIVE_STATE_BLOCKED}) {
        auto value = initial; value.state = state;
        assert(hub.publish(a, value) == OMNI_USD_LIVE_OK);
        assert(hub.publish(a, value) == OMNI_USD_LIVE_OK); // Identical publication is neutral.
        const auto actual = metrics()[2];
        assert(actual.current == 2 && actual.admitted == 2 && actual.highWater == 2);
        for (uint32_t s = 1; s <= OMNI_USD_LIVE_STATE_CLOSED; ++s)
            assert(actual.byState[s] == uint64_t(s == OMNI_USD_LIVE_STATE_OPENING) + uint64_t(s == state));
    }
    auto failure = initial;
    failure.state = OMNI_USD_LIVE_STATE_FAILED;
    failure.result = OMNI_USD_LIVE_OUTCOME_UNKNOWN;
    failure.recovery_action = OMNI_USD_LIVE_RECOVERY_PRESERVE_AND_REOPEN;
    assert(hub.publish(a, failure) == OMNI_USD_LIVE_OK);
    auto stale = initial; stale.state = OMNI_USD_LIVE_STATE_READY;
    assert(hub.publish(a, stale) == OMNI_USD_LIVE_OK);
    failure.state = OMNI_USD_LIVE_STATE_FAILED; failure.result = OMNI_USD_LIVE_MALFORMED_STREAM;
    assert(hub.publish(b, failure) == OMNI_USD_LIVE_OK);
    values = metrics();
    assert(values[2].byState[OMNI_USD_LIVE_STATE_FAILED] == 2 && !values[2].byState[OMNI_USD_LIVE_STATE_READY]);
    const auto exported = copy(fixture.b); // Metrics include both endpoints, even without history.
    contains(exported, "\"recent_records\":[]");
    contains(exported, "\"kind\":3,\"current\":2,\"high_water\":2,\"admitted\":2,\"detached\":0");
    contains(exported, "\"state\":7,\"current\":2");
    auto closed = initial; closed.state = OMNI_USD_LIVE_STATE_CLOSING;
    assert(hub.publish(a, closed) == OMNI_USD_LIVE_OK);
    assert(metrics()[2].current == 2 && metrics()[2].byState[OMNI_USD_LIVE_STATE_CLOSING] == 1);
    closed.state = OMNI_USD_LIVE_STATE_CLOSED;
    assert(hub.publish(a, closed) == OMNI_USD_LIVE_OK);
    assert(metrics()[2].current == 2 && metrics()[2].byState[OMNI_USD_LIVE_STATE_CLOSED] == 1);
    assert(hub.remove(a) == OMNI_USD_LIVE_OK);
    assert(hub.remove(a) == OMNI_USD_LIVE_INVALID_HANDLE);
    assert(hub.remove(b) == OMNI_USD_LIVE_OK);
    values = metrics();
    assert(!values[2].current && values[2].admitted == 2 && values[2].detached == 2);
    for (auto current : values[2].byState) assert(!current);
    // Final pull notifications still own the two registry slots after detach.
    try { hub.add(fixture.a, 0, "live://opaque/s/third.live", initial); assert(false); }
    catch (OmniUsdLiveResult result) { assert(result == OMNI_USD_LIVE_RESOURCE_EXHAUSTED); }
    assert(hub.unsubscribe(observer, 3000) == OMNI_USD_LIVE_OK);
    const auto next = hub.add(fixture.a, 0, "live://opaque/s/alpha.live", initial);
    initial.subject_kind = OMNI_USD_LIVE_SUBJECT_SESSION;
    const auto session = hub.add(fixture.a, 73, "", initial);
    auto ready = initial; ready.state = OMNI_USD_LIVE_STATE_READY;
    assert(hub.publish(session, ready) == OMNI_USD_LIVE_OK);
    assert(metrics()[1].current == 1 && metrics()[1].byState[OMNI_USD_LIVE_STATE_READY] == 1);
    assert(hub.remove(session) == OMNI_USD_LIVE_OK);
    assert(hub.remove(next) == OMNI_USD_LIVE_OK);
    values = metrics();
    assert(!values[1].current && values[1].highWater == 1 && values[1].admitted == 1 && values[1].detached == 1);
    assert(!values[2].current && values[2].highWater == 2 && values[2].admitted == 3 && values[2].detached == 3);
    auto endpointOptions = record<OmniUsdLiveEndpointOptions>();
    endpointOptions.authority = "127.0.0.1:34343"; endpointOptions.access_token = tokenCanary;
    OmniUsdLiveEndpoint extra = 0, alias = 0;
    assert(omniUsdLiveRegisterEndpoint(&endpointOptions, &extra) == OMNI_USD_LIVE_OK);
    assert(omniUsdLiveRegisterEndpoint(&endpointOptions, &alias) == OMNI_USD_LIVE_OK && alias == extra);
    assert(metrics()[0].current == 3 && metrics()[0].admitted == 3);
    assert(omniUsdLiveUnregisterEndpoint(alias, 3000) == OMNI_USD_LIVE_OK);
    assert(metrics()[0].current == 3 && !metrics()[0].detached);
    assert(omniUsdLiveUnregisterEndpoint(extra, 3000) == OMNI_USD_LIVE_OK);
    values = metrics();
    assert(values[0].current == 2 && values[0].highWater == 3 && values[0].admitted == 3 && values[0].detached == 1);
    for (const auto& kind : values) {
        uint64_t total = 0;
        for (auto current : kind.byState) total += current;
        assert(total == kind.current && kind.admitted - kind.detached == kind.current);
    }
}

void backpressureExitRecordsAndExport()
{
    for (bool disabled : {false, true}) {
        Fixture fixture(false, disabled);
        auto& hub = *fixture.runtime->statuses;
        const std::array<uint32_t, 5> exits = {OMNI_USD_LIVE_STATE_AUTH_WAIT, OMNI_USD_LIVE_STATE_FAILED,
            OMNI_USD_LIVE_STATE_CLOSING, OMNI_USD_LIVE_STATE_CLOSED, 0};
        for (size_t i = 0; i < exits.size(); ++i) {
            auto value = record<OmniUsdLiveStatus>();
            value.subject_kind = OMNI_USD_LIVE_SUBJECT_LAYER; value.state = OMNI_USD_LIVE_STATE_BLOCKED;
            const auto id = hub.add(fixture.a, 0, "live://opaque/session/blocked.live", value);
            value.pending_bytes = 1;
            assert(hub.publish(id, value) == OMNI_USD_LIVE_OK);
            const auto active = section(copy(fixture.b), "\"backpressure\":[", "],\"applications\":");
            contains(active, "\"kind\":3,\"current\":1,\"high_water\":1");
            if (exits[i]) {
                value.state = exits[i];
                value.result = exits[i] == OMNI_USD_LIVE_STATE_FAILED ? OMNI_USD_LIVE_RESOURCE_EXHAUSTED : OMNI_USD_LIVE_OK;
                assert(hub.publish(id, value) == OMNI_USD_LIVE_OK);
            }
            assert(hub.remove(id) == OMNI_USD_LIVE_OK);
            const auto snapshot = fixture.runtime->diagnostics->snapshot(0, 32);
            assert(snapshot.metrics.events[OMNI_USD_LIVE_RECORD_BACKPRESSURE_ENTERED] == i + 1);
            assert(snapshot.metrics.events[OMNI_USD_LIVE_RECORD_BACKPRESSURE_EXITED] == i + 1);
            if (disabled) assert(snapshot.records.empty());
            else {
                unsigned entered = 0, exited = 0;
                for (const auto& event : snapshot.records) {
                    if (event.subjectId != id) continue;
                    if (event.eventCode == OMNI_USD_LIVE_RECORD_BACKPRESSURE_ENTERED) ++entered;
                    if (event.eventCode != OMNI_USD_LIVE_RECORD_BACKPRESSURE_EXITED) continue;
                    ++exited;
                    bool duration = false, reason = false;
                    for (uint32_t f = 0; f < event.fieldCount; ++f) {
                        const auto& field = event.fields[f];
                        duration |= field.key == DiagnosticKey::DurationMs;
                        if (field.key == DiagnosticKey::Reason) {
                            reason = true; assert(field.value == (i < 2 ? i : 2));
                        }
                    }
                    assert(duration && reason);
                }
                assert(entered == 1 && exited == 1);
                std::array<uint64_t, 3> counts{}, sums{};
                for (const auto& event : snapshot.records) {
                    if (event.eventCode != OMNI_USD_LIVE_RECORD_BACKPRESSURE_EXITED) continue;
                    uint64_t reason = UINT64_MAX, duration = 0;
                    for (uint32_t f = 0; f < event.fieldCount; ++f) {
                        const auto& field = event.fields[f];
                        if (field.key == DiagnosticKey::Reason) reason = field.value;
                        if (field.key == DiagnosticKey::DurationMs) duration = field.value;
                    }
                    assert(reason < counts.size()); ++counts[reason]; sums[reason] += duration;
                }
                for (size_t reason = 0; reason < counts.size(); ++reason) {
                    const auto& metric = snapshot.metrics.backpressure[2].durationByResolution[reason];
                    assert(counts[reason] == metric.count && sums[reason] == metric.sumMs);
                }
            }
        }
        const auto json = copy(fixture.b); // Runtime metrics survive endpoint history filtering.
        const auto pressure = section(json, "\"backpressure\":[", "],\"applications\":");
        contains(pressure, "\"kind\":1,\"current\":0,\"high_water\":0");
        contains(pressure, "\"kind\":2,\"current\":0,\"high_water\":0");
        contains(pressure, "\"kind\":3,\"current\":0,\"high_water\":1");
        contains(pressure, "\"duration_ms_by_resolution\":{\"bounds\":[1,5,10,50,100,500,1000,5000]");
        contains(pressure, "\"resolution\":0,\"count\":1,");
        contains(pressure, "\"resolution\":1,\"count\":1,");
        contains(pressure, "\"resolution\":2,\"count\":3,");
        excludes(json, "blocked.live"); excludes(json, tokenCanary);
    }
}

void applicationMetricsExportWithoutHistory()
{
    Fixture fixture(false, true);
    auto& diagnostics = *fixture.runtime->diagnostics;
    diagnostics.admittedApplication(ApplicationKind::LocalAsync);
    diagnostics.admittedApplication(ApplicationKind::OrderedAsync);
    diagnostics.admittedApplication(ApplicationKind::OrderedAsync);
    diagnostics.admittedApplication(ApplicationKind::OrderedSync);
    auto active = section(copy(fixture.b), "\"applications\":[", "],\"jitter\":");
    contains(active, "\"kind\":1,\"current\":2,\"high_water\":2,\"admitted\":2,\"retired\":0");
    diagnostics.retiredApplication(ApplicationKind::LocalAsync, OMNI_USD_LIVE_CANCELLED, 50, false, 0, 0, 0);
    diagnostics.retiredApplication(ApplicationKind::OrderedAsync, OMNI_USD_LIVE_OK, 41, true, 13, 11, 17);
    diagnostics.retiredApplication(ApplicationKind::OrderedAsync, OMNI_USD_LIVE_OK, UINT64_MAX, true, 0, UINT64_MAX, 0);
    diagnostics.retiredApplication(ApplicationKind::OrderedSync, UINT32_MAX, 5, true, 0, 3, 0);
    // The reserved result joins the out-of-domain bucket; its own slot stays zero.
    diagnostics.admittedApplication(ApplicationKind::LocalAsync);
    diagnostics.retiredApplication(ApplicationKind::LocalAsync, kReservedResult, 9, false, 0, 0, 0);
    const auto metrics = diagnostics.snapshot(0, 0).metrics.applications;
    assert(metrics[0].lifetimeByResult[22].count == 1 && metrics[0].lifetimeByResult[22].sumMs == 9);
    assert(!metrics[0].lifetimeByResult[kReservedResult].count);
    assert(!metrics[0].queue.count && !metrics[0].body.count && !metrics[0].postBodyRetention.count);
    assert(metrics[0].lifetimeByResult[OMNI_USD_LIVE_CANCELLED].count == 1);
    assert(metrics[1].lifetimeByResult[0].count == 2 && metrics[1].lifetimeByResult[0].sumMs == UINT64_MAX);
    assert(metrics[1].queue.count == 2 && metrics[1].queue.sumMs == 13 && metrics[1].queue.counts[0] == 1);
    assert(metrics[1].body.count == 2 && metrics[1].body.sumMs == UINT64_MAX && metrics[1].body.counts[8] == 1);
    assert(metrics[2].body.count == 1 && !metrics[2].queue.count && !metrics[2].postBodyRetention.count);
    const auto json = copy(fixture.b);
    contains(json, "\"recent_records\":[]");
    const auto applications = section(json, "\"applications\":[", "],\"jitter\":");
    contains(applications, "\"kind\":1,\"current\":0,\"high_water\":2,\"admitted\":2,\"retired\":2");
    contains(applications, "\"body_ms\":{\"bounds\":[1,5,10,50,100,500,1000,5000],\"count\":2,\"sum_ms\":18446744073709551615");
    contains(applications, "\"result\":10,\"count\":1,\"sum_ms\":50");
    contains(applications, "\"result\":22,\"count\":1,\"sum_ms\":5");
    contains(applications, "\"result\":22,\"count\":1,\"sum_ms\":9");
    for (const auto& metric : metrics) assert(!metric.current && metric.admitted == metric.retired);
}

void jitterMetricsExportWithoutHistory()
{
    Fixture fixture(false, true);
    auto& diagnostics = *fixture.runtime->diagnostics;
    JitterEstimator::Decision measured{}; measured.targetHoldUs = 15000; measured.sampleAgeUs = 5000; measured.sampled = true;
    JitterEstimator::Decision fallback{}; fallback.targetHoldUs = 7000;
    diagnostics.startedJitterWait(13, &measured);
    diagnostics.startedJitterWait(6, &fallback);
    diagnostics.startedJitterWait(2, nullptr);
    diagnostics.startedJitterWait(3, nullptr);
    const auto active = section(copy(fixture.b), "\"jitter\":{", "},\"capacity\":");
    contains(active, "\"current\":4,\"high_water\":4,\"started\":4,\"completed\":0");
    const std::array<JitterWait::Measurement, 4> times = {{{10, 3}, {2, 0}, {6, 1}, {3, 0}}};
    for (size_t i = 0; i < times.size(); ++i)
        diagnostics.finishedJitterWait(static_cast<JitterWaitResolution>(i), times[i]);
    const auto metric = diagnostics.snapshot(0, 0).metrics.jitter;
    assert(!metric.current && metric.completed == 4 && metric.highWater == 4);
    assert(metric.frozenStarts == 2 && metric.unsampledStarts == 3);
    assert(metric.requestedWait.count == 4 && metric.requestedWait.sumMs == 24);
    assert(metric.targetHold.count == 2 && metric.targetHold.sumMs == 22);
    assert(metric.sampleAge.count == 1 && metric.sampleAge.sumMs == 5);
    for (size_t i = 0; i < times.size(); ++i) {
        assert(metric.holdByResolution[i].count == 1 && metric.holdByResolution[i].sumMs == times[i].holdMs);
        assert(metric.postDueByResolution[i].count == 1 && metric.postDueByResolution[i].sumMs == times[i].postDueMs);
    }
    const auto json = copy(fixture.b);
    contains(json, "\"recent_records\":[]");
    const auto exported = section(json, "\"jitter\":{", "},\"capacity\":");
    contains(exported, "\"current\":0,\"high_water\":4,\"started\":4,\"completed\":4");
    contains(exported, "\"frozen_generation_starts\":2,\"unsampled_starts\":3");
    contains(exported, "\"sample_age_ms\":{\"bounds\":[1,5,10,50,100,500,1000,5000],\"count\":1,\"sum_ms\":5");
    contains(exported, "\"observed_hold_ms_by_resolution\":{");
    contains(exported, "\"post_due_observation_ms_by_resolution\":{");
    contains(exported, "\"resolution\":3,\"count\":1,\"sum_ms\":3");
    contains(exported, "\"resolution\":3,\"count\":1,\"sum_ms\":0");
}

void operationAuthMetricsExportWithoutHistory()
{
    Fixture fixture(false, true);
    const auto start = std::chrono::steady_clock::time_point{};
    for (size_t kind = 0; kind < 3; ++kind) {
        OperationAuthWait wait(*fixture.runtime->diagnostics, static_cast<AuthWaitKind>(kind));
        wait.start(start);
        const auto active = copy(fixture.b);
        contains(active, "\"kind\":" + std::to_string(kind) + ",\"current\":1,\"high_water\":1,\"started\":1,\"completed\":0");
        wait.finish(OMNI_USD_LIVE_CANCELLED, start + std::chrono::milliseconds(5));
    }
    const auto json = copy(fixture.b);
    const auto rows = section(json, "\"operation_auth_waits\":[", "],\"subjects\":");
    for (size_t kind = 0; kind < 3; ++kind) {
        const auto marker = "\"kind\":" + std::to_string(kind);
        const auto begin = rows.find(marker);
        assert(begin != std::string::npos);
        const auto end = rows.find("\"kind\":", begin + marker.size());
        const auto row = rows.substr(begin, end == std::string::npos ? end : end - begin);
        contains(row, marker + ",\"current\":0,\"high_water\":1,\"started\":1,\"completed\":1");
        contains(row, "\"result\":10,\"count\":1,\"sum_ms\":5");
    }
    contains(json, "\"recent_records\":[]");
}

void capacityMetricsExportAndOverflowRecovery()
{
    Fixture fixture(false, true, UINT64_MAX);
    // Synthetic reservations exercise arithmetic without allocating payloads.
    auto large = std::make_unique<PendingBytes>(fixture.runtime->acquireEndpoint(fixture.a), UINT64_MAX, UINT64_MAX);
    auto small = std::make_unique<PendingBytes>(fixture.runtime->acquireEndpoint(fixture.b), 5, UINT64_MAX);
    auto request = fixture.runtime->requests->create(fixture.a, OMNI_USD_LIVE_OPERATION_STAT, {});
    auto snapshot = copy(fixture.b); // Both byte/slot metrics retain runtime scope.
    contains(snapshot, "\"recent_records\":[]");
    contains(snapshot, "\"request_slots\":{\"ordinary\":{\"current\":1,\"high_water\":1,\"admitted\":1,\"released\":0,\"rejected\":0}");
    contains(snapshot, "\"current_bytes\":18446744073709551615,\"current_bytes_overflow\":true");
    contains(snapshot, "\"high_water_bytes\":18446744073709551615,\"high_water_bytes_overflow\":true");
    large.reset();
    snapshot = copy();
    contains(snapshot, "\"current_bytes\":5,\"current_bytes_overflow\":false");
    contains(snapshot, "\"high_water_bytes_overflow\":true");
    large = std::make_unique<PendingBytes>(fixture.runtime->acquireEndpoint(fixture.a), UINT64_MAX, UINT64_MAX);
    small.reset();
    snapshot = copy();
    contains(snapshot, "\"current_bytes\":18446744073709551615,\"current_bytes_overflow\":false");
    large.reset();
    OmniUsdLiveRequestError error{}; error.result = OMNI_USD_LIVE_OK;
    request->finish(error, {});
    assert(omniUsdLiveReleaseRequest(request->handle) == OMNI_USD_LIVE_OK);
    snapshot = copy();
    contains(snapshot, "\"request_slots\":{\"ordinary\":{\"current\":0,\"high_water\":1,\"admitted\":1,\"released\":1,\"rejected\":0}");
    contains(snapshot, "\"reservations\":{\"current\":0,\"high_water\":2,\"admitted\":3,\"released\":3,\"rejected\":0}");
    contains(snapshot, "\"current_bytes\":0,\"current_bytes_overflow\":false");
    contains(snapshot, "\"admitted_bytes\":18446744073709551615,\"released_bytes\":18446744073709551615");
    excludes(snapshot, tokenCanary);
}

void optionValidationAndDefaultBufferContract()
{
    uint64_t size = 71;
    assert(omniUsdLiveCopyDiagnostics(0, nullptr, nullptr, &size) == OMNI_USD_LIVE_NOT_INITIALIZED && size == 71);
    std::atomic<unsigned> callbacks{0};
    OmniUsdLiveCallbackHandle handle = 99;
    assert(omniUsdLiveRegisterDiagnosticCallback(nullptr, &callbacks, count, &handle) == OMNI_USD_LIVE_NOT_INITIALIZED && !handle);
    Fixture fixture;
    assert(omniUsdLiveCopyDiagnostics(0, nullptr, nullptr, nullptr) == OMNI_USD_LIVE_INVALID_ARGUMENT);
    assert(omniUsdLiveRegisterDiagnosticCallback(nullptr, &callbacks, nullptr, &handle) == OMNI_USD_LIVE_INVALID_ARGUMENT && !handle);
    assert(omniUsdLiveRegisterDiagnosticCallback(nullptr, &callbacks, count, nullptr) == OMNI_USD_LIVE_INVALID_ARGUMENT);
    auto subscription = record<OmniUsdLiveDiagnosticOptions>();
    subscription.abi_version ^= UINT32_C(0x10000);
    assert(omniUsdLiveRegisterDiagnosticCallback(&subscription, &callbacks, count, &handle) == OMNI_USD_LIVE_UNSUPPORTED && !handle);
    subscription = record<OmniUsdLiveDiagnosticOptions>(); subscription.flags = UINT32_C(0x80000000);
    assert(omniUsdLiveRegisterDiagnosticCallback(&subscription, &callbacks, count, &handle) == OMNI_USD_LIVE_INVALID_ARGUMENT && !handle);
    subscription.flags = 0; subscription.min_severity = OMNI_USD_LIVE_SEVERITY_ERROR + 1;
    assert(omniUsdLiveRegisterDiagnosticCallback(&subscription, &callbacks, count, &handle) == OMNI_USD_LIVE_INVALID_ARGUMENT && !handle);
    subscription.min_severity = 0; subscription.event_code_count = 1;
    assert(omniUsdLiveRegisterDiagnosticCallback(&subscription, &callbacks, count, &handle) == OMNI_USD_LIVE_INVALID_ARGUMENT && !handle);
    subscription.event_code_count = 0; subscription.endpoint = UINT64_MAX;
    assert(omniUsdLiveRegisterDiagnosticCallback(&subscription, &callbacks, count, &handle) == OMNI_USD_LIVE_INVALID_HANDLE && !handle);
    assert(omniUsdLiveUnregisterCallback(UINT64_MAX, 0) == OMNI_USD_LIVE_INVALID_HANDLE);

    auto exporting = record<OmniUsdLiveDiagnosticExportOptions>();
    exporting.abi_version ^= UINT32_C(0x10000);
    assert(omniUsdLiveCopyDiagnostics(0, &exporting, nullptr, &size) == OMNI_USD_LIVE_UNSUPPORTED);
    exporting = record<OmniUsdLiveDiagnosticExportOptions>(); exporting.flags = UINT32_C(0x80000000);
    assert(omniUsdLiveCopyDiagnostics(0, &exporting, nullptr, &size) == OMNI_USD_LIVE_INVALID_ARGUMENT);
    assert(omniUsdLiveCopyDiagnostics(UINT64_MAX, nullptr, nullptr, &size) == OMNI_USD_LIVE_INVALID_HANDLE);

    alignas(8) uint32_t shortSize = sizeof(uint32_t);
    auto checkShort = [&](uint32_t* bytes) {
        assert(omniUsdLiveRegisterDiagnosticCallback(reinterpret_cast<const OmniUsdLiveDiagnosticOptions*>(bytes),
            &callbacks, count, &handle) == OMNI_USD_LIVE_INVALID_ARGUMENT && !handle);
        assert(omniUsdLiveCopyDiagnostics(0, reinterpret_cast<const OmniUsdLiveDiagnosticExportOptions*>(bytes),
            nullptr, &size) == OMNI_USD_LIVE_INVALID_ARGUMENT);
    };
    checkShort(&shortSize);
#if defined(__unix__)
    const auto page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    auto* memory = static_cast<char*>(mmap(nullptr, 2 * page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    assert(memory != MAP_FAILED && mprotect(memory + page, page, PROT_NONE) == 0);
    auto* boundary = reinterpret_cast<uint32_t*>(memory + page - sizeof(uint32_t));
    *boundary = sizeof(uint32_t);
    checkShort(boundary); // Only struct_size is readable; even abi_version would fault.
    assert(munmap(memory, 2 * page) == 0);
#endif
    assert(omniUsdLiveRegisterDiagnosticCallback(nullptr, &callbacks, count, &handle) == OMNI_USD_LIVE_OK && handle);
    fixture.runtime->diagnostics->emit(OMNI_USD_LIVE_RECORD_LAYER_READY, fixture.a);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (callbacks.load() == 0) { assert(std::chrono::steady_clock::now() < deadline); std::this_thread::yield(); }
    assert(omniUsdLiveUnregisterCallback(handle, 3000) == OMNI_USD_LIVE_OK);
    assert(omniUsdLiveUnregisterCallback(handle, 0) == OMNI_USD_LIVE_INVALID_HANDLE);
    const auto json = copy();
    assert(json.front() == '{' && json.back() == '}');
    contains(json, "\"schema_version\":" + std::to_string(OMNI_USD_LIVE_DIAGNOSTIC_SCHEMA_VERSION));
    contains(json, "\"provenance\":{"); contains(json, "\"metrics\":{"); contains(json, "\"statuses\":[{");
    contains(json, "\"rpc_attempts\":0"); contains(json, "\"rpc_completions\":0");
    excludes(json, tokenCanary);

    std::array<char, 64> output; output.fill('~');
    size = 1;
    assert(omniUsdLiveCopyDiagnostics(0, nullptr, output.data(), &size) == OMNI_USD_LIVE_BUFFER_TOO_SMALL && size > 1);
    assert(std::all_of(output.begin(), output.end(), [](char c) { return c == '~'; }));
    exporting = record<OmniUsdLiveDiagnosticExportOptions>(); exporting.max_output_bytes = 1;
    size = output.size();
    assert(omniUsdLiveCopyDiagnostics(0, &exporting, output.data(), &size) == OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    assert(std::all_of(output.begin(), output.end(), [](char c) { return c == '~'; }));
    exporting.max_output_bytes = 0; exporting.max_records = 0;
    const auto noHistory = copy(0, &exporting);
    contains(noHistory, "\"recent_records\":[]"); contains(noHistory, "\"statuses\":[{");
    contains(noHistory, "\"metrics\":{");
}

void durationMetricsExportWithoutHistory()
{
    Fixture fixture(false, true);
    auto& diagnostics = *fixture.runtime->diagnostics;
    diagnostics.startedRpc(OMNI_USD_LIVE_OPERATION_PING);
    diagnostics.observeRpc(fixture.a, OMNI_USD_LIVE_OPERATION_PING, OMNI_USD_LIVE_OK, 5, 101, 1, true, 0);
    diagnostics.startedRpc(OMNI_USD_LIVE_OPERATION_PING);
    diagnostics.observeRpc(fixture.a, OMNI_USD_LIVE_OPERATION_PING, OMNI_USD_LIVE_CANCELLED, 6, 102, 1, true, 1);
    diagnostics.emit(OMNI_USD_LIVE_RECORD_SAVE_COMPLETED, fixture.a, 101, 0,
        {{DiagnosticKey::Result, OMNI_USD_LIVE_OK}, {DiagnosticKey::DurationMs, 0}});
    diagnostics.emit(OMNI_USD_LIVE_RECORD_DRAIN_TIMEOUT, 0, 0, 0,
        {{DiagnosticKey::Result, OMNI_USD_LIVE_TIMEOUT}, {DiagnosticKey::DurationMs, 5001}});
    assert(omniUsdLiveWaitForPendingUpdates(0) == OMNI_USD_LIVE_OK);
    diagnostics.emit(OMNI_USD_LIVE_RECORD_DRAIN_COMPLETED, fixture.a, 101, 0,
        {{DiagnosticKey::Result, OMNI_USD_LIVE_OK}, {DiagnosticKey::DurationMs, 0}});
    diagnostics.emit(OMNI_USD_LIVE_RECORD_DRAIN_COMPLETED, fixture.a, 101, 0,
        {{DiagnosticKey::Result, OMNI_USD_LIVE_IO_ERROR}, {DiagnosticKey::DurationMs, 11}});
    diagnostics.observeShutdown(ShutdownPhase::Admission,OMNI_USD_LIVE_BUSY,0);
    diagnostics.observeShutdown(ShutdownPhase::Requests,OMNI_USD_LIVE_TIMEOUT,11);
    const auto json = copy(fixture.b); // All metrics stay runtime-wide.
    contains(json, "\"recent_records\":[]");
    const std::string bounds = "{\"bounds\":[1,5,10,50,100,500,1000,5000],\"entries\":[";
    contains(json, "\"rpc_duration_ms_by_operation_result\":" + bounds +
        "{\"operation\":" + std::to_string(OMNI_USD_LIVE_OPERATION_PING) +
        ",\"result\":0,\"count\":1,\"sum_ms\":5,\"counts\":[0,1,0,0,0,0,0,0,0]}," +
        "{\"operation\":" + std::to_string(OMNI_USD_LIVE_OPERATION_PING) +
        ",\"result\":" + std::to_string(OMNI_USD_LIVE_CANCELLED) +
        ",\"count\":1,\"sum_ms\":6,\"counts\":[0,0,1,0,0,0,0,0,0]}]}");
    contains(json, "\"save_duration_ms_by_result\":" + bounds +
        "{\"result\":0,\"count\":1,\"sum_ms\":0,\"counts\":[1,0,0,0,0,0,0,0,0]}]}");
    const auto drains = section(json, "\"drain_duration_ms_by_result\":", ",\"shutdown_duration_ms_by_phase_result\":");
    contains(drains, "\"result\":0,\"count\":2"); // No elapsed-time assertion on the real call.
    contains(drains, "\"result\":" + std::to_string(OMNI_USD_LIVE_TIMEOUT) +
        ",\"count\":1,\"sum_ms\":5001,\"counts\":[0,0,0,0,0,0,0,0,1]");
    contains(drains, "{\"result\":21,\"count\":1,\"sum_ms\":11,\"counts\":[0,0,0,1,0,0,0,0,0]}]}");
    contains(json, "\"shutdown_duration_ms_by_phase_result\":" + bounds +
        "{\"phase\":1,\"result\":14,\"count\":1,\"sum_ms\":0,\"counts\":[1,0,0,0,0,0,0,0,0]}," +
        "{\"phase\":6,\"result\":9,\"count\":1,\"sum_ms\":11,\"counts\":[0,0,0,1,0,0,0,0,0]}]}");
    excludes(json, tokenCanary);
}

void filteredStatusesResourcesAndClosedIncidents()
{
    for (bool names : {false, true}) for (bool historyDisabled : {false, true}) {
        Fixture fixture(names, historyDisabled);
        std::string callerUri = makeLayerUri(makeSessionUri("127.0.0.1:34341", "diagnostic-session-alpha"), "unique-alpha.live");
        const auto ownedUri = callerUri;
        const auto otherUri = makeLayerUri(makeSessionUri("127.0.0.1:34342", "diagnostic-session-beta"), "unique-beta.live");
        auto initial = record<OmniUsdLiveStatus>();
        initial.subject_kind = OMNI_USD_LIVE_SUBJECT_LAYER; initial.state = OMNI_USD_LIVE_STATE_OPENING;
        initial.progress_known = 1;
        const auto alpha = fixture.runtime->statuses->add(fixture.a, 0, callerUri, initial);
        callerUri.assign("mutated-caller-uri");
        const auto beta = fixture.runtime->statuses->add(fixture.b, 0, otherUri, initial);
        auto failed = initial;
        failed.state = OMNI_USD_LIVE_STATE_FAILED; failed.result = OMNI_USD_LIVE_OUTCOME_UNKNOWN;
        failed.operation = OMNI_USD_LIVE_OPERATION_UPDATE;
        failed.pending_write_count = 2; failed.pending_bytes = 71; failed.has_unconfirmed_edits = 1;
        failed.recovery_action = OMNI_USD_LIVE_RECOVERY_PRESERVE_AND_REOPEN;
        assert(fixture.runtime->statuses->publish(alpha, failed) == OMNI_USD_LIVE_OK);
        auto known = record<OmniUsdLiveStatus>();
        assert(omniUsdLiveGetLayerStatus(ownedUri.c_str(), &known) == OMNI_USD_LIVE_OK && known.incident_id);
        const auto incidentId = known.incident_id;
        const auto plain = copy();
        contains(section(plain, "\"statuses\":[", "],\"metrics\":"), subject(alpha));
        contains(section(plain, "\"statuses\":[", "],\"metrics\":"), subject(beta));
        excludes(plain, "canonical_resource"); excludes(plain, "unique-alpha.live"); excludes(plain, "unique-beta.live");
        excludes(plain, tokenCanary); excludes(plain, "mutated-caller-uri");
        contains(plain, "\"rpc_attempts\":0"); contains(plain, "\"pending_write_count\":2");
        if (historyDisabled) contains(plain, "\"recent_records\":[]");
        else contains(plain, "\"event_name\":\"outcome_unknown\"");
        auto exportOptions = record<OmniUsdLiveDiagnosticExportOptions>();
        exportOptions.max_records = 32;
        exportOptions.flags = OMNI_USD_LIVE_DIAGNOSTIC_RESOURCE_NAMES | OMNI_USD_LIVE_DIAGNOSTIC_SERVER_DETAIL;
        const auto selected = copy(fixture.a, &exportOptions);
        contains(selected, "\"endpoint_filter\":" + std::to_string(fixture.a));
        contains(selected, subject(alpha)); excludes(selected, subject(beta));
        excludes(selected, "\"endpoint\":" + std::to_string(fixture.b) + ",");
        if (names) contains(selected, ownedUri);
        else excludes(selected, "canonical_resource");
        excludes(selected, otherUri); excludes(selected, tokenCanary);

        assert(fixture.runtime->statuses->remove(alpha) == OMNI_USD_LIVE_OK);
        assert(omniUsdLiveGetLayerStatus(ownedUri.c_str(), &known) == OMNI_USD_LIVE_NOT_FOUND);
        const auto closed = copy(fixture.a, &exportOptions);
        excludes(section(closed, "\"statuses\":[", "],\"metrics\":"), subject(alpha));
        const auto incidents = section(closed, "\"closed_incidents\":[", "],\"history\":");
        contains(incidents, subject(alpha)); contains(incidents, "\"state\":" + std::to_string(OMNI_USD_LIVE_STATE_CLOSED));
        contains(incidents, "\"result\":" + std::to_string(OMNI_USD_LIVE_OUTCOME_UNKNOWN));
        contains(incidents, "\"incident_id\":" + std::to_string(incidentId));
        contains(incidents, "\"pending_write_count\":2"); contains(incidents, "\"has_unconfirmed_edits\":1");
        if (names) contains(incidents, ownedUri);
        else excludes(incidents, "canonical_resource");
        const auto closedRedacted = copy(fixture.a);
        excludes(closedRedacted, "canonical_resource"); excludes(closedRedacted, ownedUri);
        excludes(copy(fixture.b, &exportOptions), subject(alpha));
        assert(fixture.runtime->statuses->remove(beta) == OMNI_USD_LIVE_OK);
    }
}
} // namespace

void diagnosticsApiTests()
{
    const auto run = [](const char* scenario, auto action) {
        corpus().scenario = scenario;
        action();
    };
    run("options", optionValidationAndDefaultBufferContract);
    run("executors", executorMetricsExportWithoutHistory);
    run("callback-failures", callbackFailuresPreserveOutcomes);
    run("subjects", subjectMetricsTrackAttachedLatchedStates);
    run("backpressure", backpressureExitRecordsAndExport);
    run("applications", applicationMetricsExportWithoutHistory);
    run("jitter", jitterMetricsExportWithoutHistory);
    run("auth-waits", operationAuthMetricsExportWithoutHistory);
    run("capacity", capacityMetricsExportAndOverflowRecovery);
    run("durations", durationMetricsExportWithoutHistory);
    run("status-history", filteredStatusesResourcesAndClosedIncidents);
    corpus().finish();
    std::cout << "diagnostics ABI: options, bounded copies, filtering, resources, and closed incidents passed\n";
}
