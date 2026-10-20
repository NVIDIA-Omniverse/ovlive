// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "LiveUnary.h"
#include "LiveLayerWrite.h"
#include "LiveAuth.h"
#include "LiveMetadata.h"
#include "../support/test_tls.h"
#include <cassert>
#include <fstream>
#include <future>
#include <iostream>
#include <iterator>
#include <vector>

using namespace omni::usd::live;

template <class T> T record()
{
    T value{};
    value.struct_size = sizeof(value);
    value.abi_version = OMNI_USD_LIVE_ABI_VERSION;
    return value;
}

struct Refresh {
    std::mutex mutex;
    std::condition_variable changed;
    unsigned calls = 0;
    uint64_t generation = 0;
    static void OMNI_USD_LIVE_CALL callback(void* user, const OmniUsdLiveAuthRequired* event)
    {
        auto& self = *static_cast<Refresh*>(user);
        assert(event->reason == OMNI_USD_LIVE_AUTH_REJECTED && event->grpc_status_present && event->grpc_status == 16);
        assert(event->rejected_generation == 1);
        uint64_t generation = 0;
        assert(omniUsdLiveSetAccessToken(event->endpoint, event->rejected_generation, "synthetic-writer", 0, &generation) == OMNI_USD_LIVE_OK);
        std::lock_guard<std::mutex> lock(self.mutex);
        ++self.calls;
        self.generation = generation;
        self.changed.notify_all();
    }
};

// This path exercises C API submissions and callback views. The auth-timeout
// fixture alone advances the private controller clock after the real auth
// callback proves the wait exists; it never waits out a wall-clock deadline.
struct PublicResult {
    std::thread::id caller = std::this_thread::get_id();
    unsigned callbacks = 0;
    OmniUsdLiveResult result = OMNI_USD_LIVE_INTERNAL_ERROR;
    std::string id, uri, name, creator;
    uint64_t count = 0, created = 0, modified = 0;
    void observe(OmniUsdLiveResult value)
    {
        assert(std::this_thread::get_id() != caller);
        ++callbacks;
        result = value;
    }
    static void OMNI_USD_LIVE_CALL createdCallback(void* user, OmniUsdLiveResult result, const OmniUsdLiveCreatedSession* info)
    {
        auto& self = *static_cast<PublicResult*>(user);
        self.observe(result);
        assert(bool(info) == (result == OMNI_USD_LIVE_OK));
        if (info) {
            assert(info->struct_size == sizeof(*info) && info->abi_version == OMNI_USD_LIVE_ABI_VERSION);
            self.id = info->session_id; self.uri = info->uri;
            assert(!self.id.empty() && !self.uri.empty());
        }
    }
    static void OMNI_USD_LIVE_CALL listCallback(void* user, OmniUsdLiveResult result, uint64_t count, const OmniUsdLiveSessionInfo* const* infos)
    {
        auto& self = *static_cast<PublicResult*>(user);
        self.observe(result);
        assert(bool(infos) == bool(count));
        assert(result == OMNI_USD_LIVE_OK || !count);
        self.count = count;
        for (uint64_t i = 0; i < count; ++i) {
            assert(infos[i]->struct_size == sizeof(*infos[i]) && infos[i]->abi_version == OMNI_USD_LIVE_ABI_VERSION);
            if (!self.id.empty()) {
                assert(self.id == infos[i]->session_id && self.uri == infos[i]->uri);
            }
            self.name = infos[i]->name; self.creator = infos[i]->created_by;
            self.created = infos[i]->created_time_ms;
        }
    }
    static void OMNI_USD_LIVE_CALL statCallback(void* user, OmniUsdLiveResult result, const OmniUsdLiveLayerInfo* info)
    {
        auto& self = *static_cast<PublicResult*>(user);
        self.observe(result);
        assert(bool(info) == (result == OMNI_USD_LIVE_OK));
        if (info) {
            assert(info->struct_size == sizeof(*info) && info->abi_version == OMNI_USD_LIVE_ABI_VERSION);
            self.created = info->created_time_ms; self.modified = info->modified_time_ms;
        }
    }
    static void OMNI_USD_LIVE_CALL resultCallback(void* user, OmniUsdLiveResult result)
    { static_cast<PublicResult*>(user)->observe(result); }
};
struct PublicAuth {
    bool fail = false, expire = false;
    std::atomic<unsigned> calls{0};
    static void OMNI_USD_LIVE_CALL callback(void* user, const OmniUsdLiveAuthRequired* event)
    {
        auto& self = *static_cast<PublicAuth*>(user);
        if (self.expire) {
            auto runtime = getRuntime();
            auto endpoint = runtime->findEndpoint(event->endpoint);
            assert(endpoint);
            std::lock_guard<std::mutex> lock(endpoint->mutex);
            assert(endpoint->authWaitActive && !endpoint->authFailed);
            runtime->auth->evaluate(*endpoint, wallTimeMs(), endpoint->authDeadline);
            assert(endpoint->authFailed);
            ++self.calls;
            return;
        }
        if (self.fail)
            assert(omniUsdLiveFailAuthRefresh(event->endpoint, event->challenge_id) == OMNI_USD_LIVE_OK);
        uint64_t generation = 0;
        assert(omniUsdLiveSetAccessToken(event->endpoint, event->rejected_generation, "synthetic-writer", 0, &generation) == OMNI_USD_LIVE_OK);
        ++self.calls;
    }
};
std::string publicDiagnostics()
{
    uint64_t size = 0;
    assert(omniUsdLiveCopyDiagnostics(0, nullptr, nullptr, &size) == OMNI_USD_LIVE_OK);
    for (unsigned attempt = 0; attempt < 4; ++attempt) {
        assert(size && size <= 8 * 1024 * 1024);
        std::vector<char> buffer(static_cast<size_t>(size));
        const auto result = omniUsdLiveCopyDiagnostics(0, nullptr, buffer.data(), &size);
        if (result == OMNI_USD_LIVE_BUFFER_TOO_SMALL) continue;
        assert(result == OMNI_USD_LIVE_OK && size && size <= buffer.size() && buffer[size - 1] == 0);
        return std::string(buffer.data(), static_cast<size_t>(size - 1));
    }
    assert(false); return {};
}
int publicFinite(int argc, char** argv)
{
    const std::string mode = argv[2];
    const auto expected = static_cast<OmniUsdLiveResult>(std::stoul(argv[3]));
    const int expectedGrpc = std::stoi(argv[4]);
    auto options = record<OmniUsdLiveRuntimeOptions>();
    options.request_timeout_ms = mode == "public-deadline-starved" ? 1 : mode.rfind("public-deadline-", 0) == 0 ? 500 : 3000;
    options.default_auth_wait_timeout_ms = 30000;
    options.retry_initial_ms = 20; options.retry_maximum_ms = 50; options.retry_max_attempts = 3;
    // Finite requests follow retry_*; a finite retry bounded by these reconnect options would stop after one attempt.
    options.reconnect_initial_ms = options.reconnect_maximum_ms = 60000; options.reconnect_max_attempts = 1;
    if (mode == "public-finite-budget") {
        // The endpoint budget holds a full update but no finite response allowance.
        options.max_update_bytes = 1024; options.max_pending_bytes_per_endpoint = 1024;
    }
    assert(omniUsdLiveInitialize(&options) == OMNI_USD_LIVE_OK);
    auto endpointOptions = record<OmniUsdLiveEndpointOptions>();
    endpointOptions.authority = argv[1]; endpointOptions.access_token = argv[5];
    OmniUsdLiveEndpoint endpoint = 0;
    assert(omniUsdLiveRegisterEndpoint(&endpointOptions, &endpoint) == OMNI_USD_LIVE_OK);
    PublicResult seen;
    OmniUsdLiveRequest request = 0;
    auto finish = [&] {
        assert(omniUsdLiveWait(request, 5000) == OMNI_USD_LIVE_OK);
        auto error = record<OmniUsdLiveRequestError>();
        assert(omniUsdLiveGetRequestError(request, &error) == OMNI_USD_LIVE_OK);
        OmniUsdLiveResult result = OMNI_USD_LIVE_INTERNAL_ERROR;
        assert(omniUsdLiveGetRequestStatus(request, &result) == OMNI_USD_LIVE_OK && result == error.result);
        assert(error.request_id == request);
        assert(omniUsdLiveReleaseRequest(request) == OMNI_USD_LIVE_OK);
        request = 0;
        return error;
    };
    const bool create = mode.find("create") != std::string::npos || mode == "public-roundtrip";
    if (create) {
        std::string key = "public/base.usda", name = "Náme / not a path";
        assert(omniUsdLiveCreateSession(endpoint, key.c_str(), name.c_str(), &seen, PublicResult::createdCallback, &request) == OMNI_USD_LIVE_OK);
        key.assign("changed"); name.assign("changed");
        if (mode == "public-roundtrip" || mode == "public-create-list-failure") {
            assert(finish().result == OMNI_USD_LIVE_OK);
            assert(omniUsdLiveListSessions(endpoint, "public/base.usda", &seen, PublicResult::listCallback, &request) == OMNI_USD_LIVE_OK);
            if (mode == "public-roundtrip") {
                assert(finish().result == OMNI_USD_LIVE_OK);
                assert(seen.count == 1 && seen.name == "Náme / not a path" && seen.creator == "writer");
                assert(omniUsdLiveDeleteSession(seen.uri.c_str(), &seen, PublicResult::resultCallback, &request) == OMNI_USD_LIVE_OK);
            }
        }
    } else if (mode == "public-stat" || mode == "public-delete-layer") {
        assert(argc == 7);
        char session[4096], layer[4096]; uint64_t size = sizeof(session);
        assert(omniUsdLiveMakeSessionUri(argv[1], argv[6], session, &size) == OMNI_USD_LIVE_OK);
        size = sizeof(layer);
        assert(omniUsdLiveMakeLayerUri(session, "root.live", layer, &size) == OMNI_USD_LIVE_OK);
        if (mode == "public-stat")
            assert(omniUsdLiveStatLayer(layer, &seen, PublicResult::statCallback, &request) == OMNI_USD_LIVE_OK);
        else
            assert(omniUsdLiveDeleteLayer(layer, &seen, PublicResult::resultCallback, &request) == OMNI_USD_LIVE_OK);
    } else {
        assert(omniUsdLiveListSessions(endpoint, "public/base.usda", &seen, PublicResult::listCallback, &request) == OMNI_USD_LIVE_OK);
    }
    PublicAuth auth; auth.fail = mode == "public-auth-fail";
    auth.expire = mode == "public-auth-timeout";
    if (mode == "public-auth-waiters") {
        auto runtime = getRuntime();
        PublicResult cancelledResult;
        OmniUsdLiveRequest cancelledRequest = 0;
        assert(omniUsdLiveListSessions(endpoint, "public/base.usda", &cancelledResult,
            PublicResult::listCallback, &cancelledRequest) == OMNI_USD_LIVE_OK);
        assert(runtime->diagnostics->waitForOperationAuthWaitsForTest(AuthWaitKind::Finite, 2,
            std::chrono::steady_clock::now() + std::chrono::seconds(3)));
        assert(runtime->diagnostics->snapshot(0, 0).metrics.authBlockedEndpoints == 1);
        assert(omniUsdLiveCancel(cancelledRequest, 3000) == OMNI_USD_LIVE_OK);
        assert(omniUsdLiveReleaseRequest(cancelledRequest) == OMNI_USD_LIVE_OK);
        assert(cancelledResult.callbacks == 0);
        const auto parked = runtime->diagnostics->snapshot(0, 0).metrics.operationAuthWaits[0];
        assert(parked.current == 1 && parked.started == 2 && parked.completed == 1);
        assert(parked.durationByResult[OMNI_USD_LIVE_CANCELLED].count == 1);
        uint64_t generation = 0;
        assert(omniUsdLiveSetAccessToken(endpoint, 0, "synthetic-writer", 0, &generation) == OMNI_USD_LIVE_OK);
    }
    // Establish the actual finite-operation gate before delivering the host
    // response; an endpoint challenge alone does not prove a request parked.
    if (mode == "public-auth-missing" || mode == "public-auth-rejected" || auth.fail || auth.expire)
        assert(getRuntime()->diagnostics->waitForOperationAuthWaitsForTest(AuthWaitKind::Finite, 1,
            std::chrono::steady_clock::now() + std::chrono::seconds(3)));
    OmniUsdLiveCallbackHandle authCallback = 0;
    if (mode == "public-auth-missing" || mode == "public-auth-rejected" || auth.fail || auth.expire)
        assert(omniUsdLiveRegisterAuthRequiredCallback(endpoint, &auth, PublicAuth::callback, &authCallback) == OMNI_USD_LIVE_OK);
    const bool cancel = mode == "public-cancel-create" || mode == "public-cancel-missing";
    if (cancel) {
        if (mode == "public-cancel-create") {
            std::string command; std::getline(std::cin, command);
            assert(command == "cancel");
        }
        const auto result = omniUsdLiveCancel(request, 0);
        assert(result == OMNI_USD_LIVE_OK || result == OMNI_USD_LIVE_TIMEOUT);
        if (mode == "public-cancel-missing") {
            uint64_t generation = 0;
            assert(omniUsdLiveSetAccessToken(endpoint, 0, "synthetic-writer", 0, &generation) == OMNI_USD_LIVE_OK);
        }
    }
    const auto error = finish();
    assert(error.result == expected);
    assert((error.grpc_status_present ? static_cast<int>(error.grpc_status) : -1) == expectedGrpc);
    assert(cancel ? seen.callbacks == 0 : seen.result == expected);
    if (authCallback) {
        assert(omniUsdLiveUnregisterCallback(authCallback, 3000) == OMNI_USD_LIVE_OK);
        assert(auth.calls == 1);
    }
    const auto rpcsStarted = getRuntime()->diagnostics->snapshot(0, 0).metrics.rpcAttempts;
    assert(omniUsdLiveUnregisterEndpoint(endpoint, 0) == OMNI_USD_LIVE_OK);
    const auto diagnostics = mode.rfind("public-auth-", 0) == 0 ? publicDiagnostics() : "null";
    assert(omniUsdLiveShutdown(3000) == OMNI_USD_LIVE_OK);
    std::cout << "{\"result\":" << error.result << ",\"grpc_status\":" << expectedGrpc
              << ",\"callbacks\":" << seen.callbacks << ",\"attempts\":" << error.attempt_id
              << ",\"count\":" << seen.count << ",\"created_ms\":" << seen.created
              << ",\"modified_ms\":" << seen.modified << ",\"rpcs_started\":" << rpcsStarted
              << ",\"diagnostics\":" << diagnostics << "}" << std::endl;
    return 0;
}

std::string patternedBytes(size_t size, unsigned salt)
{
    std::string bytes(size, '\0');
    for (size_t index = 0; index < size; ++index)
        bytes[index] = static_cast<char>((index * 31 + (index >> 16) * 13 + salt) & 255);
    return bytes;
}
int layerWriteDriver(int argc, char** argv)
{
    assert(argc == 7);
    const std::string mode = argv[2], session = argv[6];
    const auto expected = static_cast<OmniUsdLiveResult>(std::stoul(argv[3]));
    const int expectedGrpc = std::stoi(argv[4]);
    auto options = record<OmniUsdLiveRuntimeOptions>();
    options.max_update_bytes = mode == "layer-exact-limit" ? 8 : 6 * 1024 * 1024;
    options.max_pending_bytes_per_endpoint = 8 * 1024 * 1024;
    assert(omniUsdLiveInitialize(&options) == OMNI_USD_LIVE_OK);
    auto epOptions = record<OmniUsdLiveEndpointOptions>();
    epOptions.authority = argv[1]; epOptions.access_token = argv[5];
    OmniUsdLiveEndpoint endpoint = 0;
    assert(omniUsdLiveRegisterEndpoint(&epOptions, &endpoint) == OMNI_USD_LIVE_OK);
    auto runtime = getRuntime();
    auto retained = runtime->findEndpoint(endpoint);
    const auto caller = std::this_thread::get_id();
    const auto writerId = retained->writerId;
    unsigned callbacks = 0;
    uint64_t responseId = 0, responseSequence = 0, callbackRequest = 0;
    OmniUsdLiveResult callbackResult = OMNI_USD_LIVE_INTERNAL_ERROR;
    auto observe = [&](const OmniUsdLiveRequestError& error, uint64_t generation) {
        assert(std::this_thread::get_id() != caller && generation == 1);
        assert(error.attempt_id == 1 && error.request_id);
        assert(error.struct_size == sizeof(error) && error.abi_version == OMNI_USD_LIVE_ABI_VERSION);
        callbackRequest = error.request_id;
        callbackResult = error.result;
        ++callbacks;
    };
    std::promise<void> callbackEntered, callbackRelease;
    auto callbackGate = callbackRelease.get_future().share();
    auto callbackStarted = callbackEntered.get_future();
    UnaryCallback<rpc::CreateLayerResponse> created = [&](const auto& error, const auto& reply, uint64_t generation) {
        observe(error, generation);
        if (error.result == OMNI_USD_LIVE_OK) responseId = reply.object_id();
        if (mode == "layer-held-callback") {
            callbackEntered.set_value();
            callbackGate.wait();
        }
    };
    UnaryCallback<rpc::UpdateLayerResponse> updated = [&](const auto& error, const auto& reply, uint64_t generation) {
        observe(error, generation);
        if (error.result == OMNI_USD_LIVE_OK) responseSequence = reply.sequence_num();
    };
    auto pendingBytes = [&] {
        std::lock_guard<std::mutex> lock(retained->mutex);
        return retained->pendingTransportBytes;
    };
    auto finish = [&](std::shared_ptr<Request>& request) {
        assert(request->wait(6000) == OMNI_USD_LIVE_OK);
        assert(pendingBytes() == 0); // Wait itself proves buffer retirement.
        auto error = record<OmniUsdLiveRequestError>();
        assert(request->error(&error) == OMNI_USD_LIVE_OK);
        if (callbacks) assert(callbackRequest == request->handle);
        assert(runtime->requests->release(request->handle) == OMNI_USD_LIVE_OK);
        request.reset();
        assert(pendingBytes() == 0); // Quiescence includes capture/buffer retirement.
        return error;
    };
    std::shared_ptr<Request> request;
    OmniUsdLiveRequestError outcome{};
    if (mode == "layer-local-reject") {
        auto tooLarge = patternedBytes(options.max_update_bytes + 1, 1);
        try {
            request = submitLayerCreate(*runtime, endpoint, session, "root.live", tooLarge, created);
            assert(false);
        } catch (OmniUsdLiveResult result) { assert(result == OMNI_USD_LIVE_RESOURCE_EXHAUSTED); }
        try {
            request = submitLayerUpdate(*runtime, endpoint, session, "root.live", 0, 0, "x", updated);
            assert(false);
        } catch (const std::invalid_argument&) {}
        try {
            request = submitLayerCreate(*runtime, endpoint, session, "../root.live", "x", created);
            assert(false);
        } catch (const std::invalid_argument&) {}
        assert(!callbacks && !pendingBytes());
        outcome.result = expected;
    } else if (mode == "layer-missing-auth") {
        try {
            request = submitLayerCreate(*runtime, endpoint, session, "root.live", "x", created);
            assert(false);
        } catch (OmniUsdLiveResult result) { assert(result == OMNI_USD_LIVE_UNAUTHENTICATED); }
        assert(!callbacks && !pendingBytes());
        outcome.result = expected;
    } else {
        // Connection readiness is proven before short-deadline scenarios.
        if (mode.find("deadline") != std::string::npos) {
            rpc::PingRequest ping;
            auto warmup = submitUnary(*runtime, endpoint, ping, UnaryCallback<rpc::PingResponse>{});
            assert(warmup->wait(5000) == OMNI_USD_LIVE_OK);
            OmniUsdLiveResult result;
            assert(warmup->result(&result) == OMNI_USD_LIVE_OK && result == OMNI_USD_LIVE_OK);
            assert(runtime->requests->release(warmup->handle) == OMNI_USD_LIVE_OK);
        }
        auto bytes = patternedBytes(mode == "layer-empty" ? 0 : mode == "layer-exact-limit" ? 8 : 2 * kLayerChunkBytes + 37, 7);
        const bool update = mode.find("update") != std::string::npos;
        const auto timeout = mode.find("deadline") != std::string::npos ? 500 : 5000;
        if (update)
            request = submitLayerUpdate(*runtime, endpoint, session, "root.live", 1, 123456789, bytes, updated, timeout);
        else
            request = submitLayerCreate(*runtime, endpoint, session, "root.live", bytes, created, timeout);
        bytes.assign("caller replaced its input");
        if (mode == "layer-held-callback") {
            assert(callbackStarted.wait_for(std::chrono::seconds(3)) == std::future_status::ready);
            assert(request->wait(0) == OMNI_USD_LIVE_TIMEOUT);
            assert(pendingBytes() > 2 * kLayerChunkBytes);
            auto second = patternedBytes(2 * kLayerChunkBytes + 37, 8);
            try {
                auto rejected = submitLayerCreate(*runtime, endpoint, session, "rejected.live", second, created);
                assert(false);
            } catch (OmniUsdLiveResult result) { assert(result == OMNI_USD_LIVE_RESOURCE_EXHAUSTED); }
            callbackRelease.set_value();
        }
        const bool cancel = mode.find("cancel") != std::string::npos || mode == "layer-capacity";
        if (cancel) {
            std::string command; std::getline(std::cin, command);
            assert(command == "cancel"); // Reference fault barrier, not a sleep.
            if (mode == "layer-capacity") {
                assert(pendingBytes() > 2 * kLayerChunkBytes);
                auto second = patternedBytes(2 * kLayerChunkBytes + 37, 8);
                try {
                    auto rejected = submitLayerCreate(*runtime, endpoint, session, "rejected.live", second, created);
                    assert(false);
                } catch (OmniUsdLiveResult result) { assert(result == OMNI_USD_LIVE_RESOURCE_EXHAUSTED); }
            }
            assert(request->cancel(5000) == OMNI_USD_LIVE_OK);
            assert(!callbacks);
        }
        outcome = finish(request);
        if (!cancel) assert(callbackResult == outcome.result);
        if (mode == "layer-roundtrip") {
            assert(outcome.result == OMNI_USD_LIVE_OK && responseId == 1);
            auto delta = patternedBytes(kLayerChunkBytes + 19, 23);
            request = submitLayerUpdate(*runtime, endpoint, session, "root.live", responseId, 123456789, delta, updated);
            delta.assign("caller released its update");
            outcome = finish(request);
            assert(responseSequence == 2 && callbacks == 2);
        }
        if (mode == "layer-capacity") {
            request = submitLayerCreate(*runtime, endpoint, session, "after.live", "after-cancel", created);
            assert(finish(request).result == OMNI_USD_LIVE_OK && callbacks == 1);
        }
    }
    assert(outcome.result == expected);
    assert((outcome.grpc_status_present ? static_cast<int>(outcome.grpc_status) : -1) == expectedGrpc);
    retained.reset();
    assert(omniUsdLiveUnregisterEndpoint(endpoint, 0) == OMNI_USD_LIVE_OK);
    runtime.reset();
    assert(omniUsdLiveShutdown(5000) == OMNI_USD_LIVE_OK);
    std::cout << "{\"result\":" << outcome.result << ",\"grpc_status\":" << expectedGrpc
              << ",\"callbacks\":" << callbacks << ",\"object_id\":" << responseId
              << ",\"sequence\":" << responseSequence << ",\"writer\":\"" << writerId << "\"}" << std::endl;
    return 0;
}

int layerReadDriver(int argc, char** argv);
int layerActorDriver(int argc, char** argv);
int diagnosticsDriver(int argc, char** argv);
int sessionDriver(int argc, char** argv);

int timingReceiptDriver(int argc, char** argv)
{
    assert(argc == 6);
    auto options = record<OmniUsdLiveRuntimeOptions>();
    options.callback_worker_count = 1;
    assert(omniUsdLiveInitialize(&options) == OMNI_USD_LIVE_OK);
    auto endpointOptions = record<OmniUsdLiveEndpointOptions>();
    endpointOptions.authority = argv[1]; endpointOptions.access_token = argv[5];
    OmniUsdLiveEndpoint endpoint = 0;
    assert(omniUsdLiveRegisterEndpoint(&endpointOptions, &endpoint) == OMNI_USD_LIVE_OK);
    auto runtime = getRuntime();
    struct BlockWorker final : DispatchJob {
        std::promise<void> entered, release;
        std::shared_future<void> gate = release.get_future().share();
        void dispatch() noexcept override { entered.set_value(); gate.wait(); }
    };
    auto blocker = std::make_shared<BlockWorker>();
    auto entered = blocker->entered.get_future();
    runtime->requests->executor().enqueue(*blocker);
    assert(entered.wait_for(std::chrono::seconds(3)) == std::future_status::ready);

    using Clock = std::chrono::steady_clock;
    auto receipt = std::make_shared<UnaryTimingReceipt>();
    std::atomic<unsigned> callbacks{0};
    Clock::time_point observedCompletion{}, released{}, delivered{};
    const auto caller = std::this_thread::get_id();
    const auto submitted = Clock::now();
    rpc::PingRequest ping;
    ping.set_client_timestamp_microseconds_since_epoch(123456789);
    auto request = submitTimingPing(*runtime, endpoint, ping,
        [&](const OmniUsdLiveRequestError& error, const rpc::PingResponse& response, uint64_t generation) {
            assert(std::this_thread::get_id() != caller);
            assert(error.result == OMNI_USD_LIVE_OK && error.grpc_status_present && error.grpc_status == 0);
            assert(generation == 1 && response.client_timestamp_microseconds_since_epoch() == 123456789);
            assert(response.server_timestamp_microseconds_since_epoch() > 0);
            assert(receipt->ready.load(std::memory_order_acquire));
            assert(receipt->completed == observedCompletion);
            delivered = Clock::now();
            assert(delivered >= released && released >= observedCompletion);
            ++callbacks;
        }, 3000, receipt);
    const auto deadline = Clock::now() + std::chrono::seconds(4);
    while (!receipt->ready.load(std::memory_order_acquire)) {
        assert(Clock::now() < deadline);
        std::this_thread::yield();
    }
    // Receipt publication cannot depend on dispatch: its only worker remains
    // inside the barrier. Read the immutable CQ timestamp before releasing it.
    observedCompletion = receipt->completed;
    assert(observedCompletion >= submitted && observedCompletion <= Clock::now());
    assert(callbacks.load() == 0 && request->wait(0) == OMNI_USD_LIVE_TIMEOUT);
    released = Clock::now();
    blocker->release.set_value();
    assert(request->wait(4000) == OMNI_USD_LIVE_OK);
    assert(callbacks.load() == 1 && receipt->completed == observedCompletion && delivered >= released);
    assert(runtime->requests->release(request->handle) == OMNI_USD_LIVE_OK);
    request.reset();
    blocker.reset();
    assert(omniUsdLiveUnregisterEndpoint(endpoint, 0) == OMNI_USD_LIVE_OK);
    runtime.reset();
    assert(omniUsdLiveShutdown(3000) == OMNI_USD_LIVE_OK);
    std::cout << "{\"result\":0,\"grpc_status\":0,\"callbacks\":1,"
                 "\"receipt_before_release\":true,\"callback_after_release\":true}" << std::endl;
    return 0;
}

// A metadata request whose inherited deadline has already passed classifies
// credential starvation before it returns (design section 14).
int metadataStarved(int argc, char** argv)
{
    assert(argc == 7);
    const auto expected = static_cast<OmniUsdLiveResult>(std::stoul(argv[3]));
    auto options = record<OmniUsdLiveRuntimeOptions>();
    options.default_auth_wait_timeout_ms = 30000;
    assert(omniUsdLiveInitialize(&options) == OMNI_USD_LIVE_OK);
    auto endpointOptions = record<OmniUsdLiveEndpointOptions>();
    endpointOptions.authority = argv[1]; endpointOptions.access_token = argv[5];
    OmniUsdLiveEndpoint endpoint = 0;
    assert(omniUsdLiveRegisterEndpoint(&endpointOptions, &endpoint) == OMNI_USD_LIVE_OK);
    char session[4096], layer[4096]; uint64_t size = sizeof(session);
    assert(omniUsdLiveMakeSessionUri(argv[1], argv[6], session, &size) == OMNI_USD_LIVE_OK);
    size = sizeof(layer);
    assert(omniUsdLiveMakeLayerUri(session, "root.live", layer, &size) == OMNI_USD_LIVE_OK);
    OmniUsdLiveLayerInfo info{};
    OmniUsdLiveResult result;
    {
        MetadataDeadlineScope expired(std::chrono::steady_clock::now()); // Already passed.
        result = statLayerMetadata(layer, info, nullptr);
    }
    assert(result == expected);
    assert(omniUsdLiveUnregisterEndpoint(endpoint, 3000) == OMNI_USD_LIVE_OK);
    assert(omniUsdLiveShutdown(3000) == OMNI_USD_LIVE_OK);
    std::cout << "{\"result\":" << result << ",\"grpc_status\":-1}" << std::endl;
    return 0;
}

int main(int argc, char** argv)
{
    // Only synthetic credentials are used by this local test driver. Production
    // hosts provide credentials through the API, never command-line arguments.
    assert(argc >= 6);
    if (std::string(argv[2]) == "metadata-starved-expired-scope")
        return metadataStarved(argc, argv);
    if (std::string(argv[2]).rfind("join-", 0) == 0 || std::string(argv[2]).rfind("send-", 0) == 0)
        return sessionDriver(argc, argv);
    if (std::string(argv[2]).rfind("diag-", 0) == 0)
        return diagnosticsDriver(argc, argv);
    if (std::string(argv[2]) == "timing-receipt")
        return timingReceiptDriver(argc, argv);
    if (std::string(argv[2]).rfind("actor-", 0) == 0)
        return layerActorDriver(argc, argv);
    if (std::string(argv[2]).rfind("read-", 0) == 0)
        return layerReadDriver(argc, argv);
    if (std::string(argv[2]).rfind("layer-", 0) == 0)
        return layerWriteDriver(argc, argv);
    if (std::string(argv[2]).rfind("public-", 0) == 0)
        return publicFinite(argc, argv);
    const std::string mode = argv[2];
    const auto expected = static_cast<OmniUsdLiveResult>(std::stoul(argv[3]));
    const int expectedGrpc = std::stoi(argv[4]);
    auto runtimeOptions = record<OmniUsdLiveRuntimeOptions>();
    if (mode == "capacity")
        runtimeOptions.max_requests = 1;
    assert(omniUsdLiveInitialize(&runtimeOptions) == OMNI_USD_LIVE_OK);
    const TestTls tls;
    auto options = record<OmniUsdLiveEndpointOptions>();
    options.authority = argv[1];
    options.access_token = argv[5];
    if (mode != "tls-ping")
        tls.apply(options);
    if (mode == "expired-auth")
        options.expires_at_ms = 1;
    std::string ca;
    if (mode == "tls-ping") {
        assert(argc == 8);
        std::ifstream input(argv[6]);
        ca.assign(std::istreambuf_iterator<char>(input), {});
        assert(!ca.empty());
        options.transport_security = OMNI_USD_LIVE_TRANSPORT_TLS;
        options.tls_server_name = argv[7];
        options.ca_pem = ca.c_str();
    }
    OmniUsdLiveEndpoint endpoint = 0;
    assert(omniUsdLiveRegisterEndpoint(&options, &endpoint) == OMNI_USD_LIVE_OK);
    auto runtime = getRuntime();
    Refresh refresh;
    OmniUsdLiveCallbackHandle authCallback = 0;
    if (mode == "auth-rotate")
        assert(omniUsdLiveRegisterAuthRequiredCallback(endpoint, &refresh, Refresh::callback, &authCallback) == OMNI_USD_LIVE_OK);
    const auto caller = std::this_thread::get_id();
    uint32_t callbacks = 0;
    OmniUsdLiveRequestError callbackError{};
    auto observe = [&](const OmniUsdLiveRequestError& error, uint64_t generation) {
        assert(std::this_thread::get_id() != caller);
        assert(generation == (mode == "auth-rotate" && callbacks ? 2U : 1U));
        assert(error.attempt_id == 1 && error.request_id);
        assert(error.detail[0] == '\0');
        callbackError = error;
        ++callbacks;
    };
    std::shared_ptr<Request> request;
    auto release = [&] {
        assert(request->wait(5000) == OMNI_USD_LIVE_OK);
        assert(runtime->requests->release(request->handle) == OMNI_USD_LIVE_OK);
        request.reset();
    };
    rpc::PingRequest ping;
    ping.set_client_timestamp_microseconds_since_epoch(123456789);
    auto pingCallback = [&](const OmniUsdLiveRequestError& error, const rpc::PingResponse& response, uint64_t generation) {
        observe(error, generation);
        if (error.result == OMNI_USD_LIVE_OK) {
            assert(response.client_timestamp_microseconds_since_epoch() == 123456789);
            assert(response.server_timestamp_microseconds_since_epoch() > 0);
        }
    };
    const bool cancel = mode == "cancel-create" || mode == "cancel-ping";
    const bool deadline = mode == "deadline-create" || mode == "deadline-ping";
    if (mode == "missing-auth" || mode == "expired-auth") {
        bool rejected = false;
        try {
            request = submitUnary(*runtime, endpoint, ping, pingCallback);
        } catch (OmniUsdLiveResult result) {
            rejected = result == OMNI_USD_LIVE_UNAUTHENTICATED;
        }
        assert(rejected && !request && callbacks == 0);
        assert(omniUsdLiveUnregisterEndpoint(endpoint, 0) == OMNI_USD_LIVE_OK);
        runtime.reset();
        assert(omniUsdLiveShutdown(5000) == OMNI_USD_LIVE_OK);
        std::cout << "{\"result\":6,\"grpc_status\":-1,\"callbacks\":0}" << std::endl;
        return 0;
    }
    if (cancel || deadline) {
        // Explicit native readiness before the short client-deadline scenario.
        request = submitUnary(*runtime, endpoint, ping, pingCallback);
        release();
        assert(callbackError.result == OMNI_USD_LIVE_OK);
        callbacks = 0;
    }
    if (mode == "auth-rotate") {
        rpc::CreateSessionRequest create;
        create.set_path("native/base.usda");
        create.set_name("must-not-retry-after-refresh");
        request = submitUnary(*runtime, endpoint, create,
            [&](const OmniUsdLiveRequestError& error, const rpc::CreateSessionResponse&, uint64_t generation) {
                observe(error, generation);
            });
        release();
        // Rejected before execution: definitive, and never resent after refresh.
        assert(callbackError.result == OMNI_USD_LIVE_UNAUTHENTICATED && callbackError.grpc_status == 16);
        {
            std::unique_lock<std::mutex> lock(refresh.mutex);
            assert(refresh.changed.wait_for(lock, std::chrono::seconds(3), [&] { return refresh.calls == 1; }));
            assert(refresh.generation == 2);
        }
        auto snapshot = record<OmniUsdLiveAuthState>();
        assert(omniUsdLiveGetAuthState(endpoint, &snapshot) == OMNI_USD_LIVE_OK);
        assert(snapshot.generation == 2 && !snapshot.challenge_id);
        request = submitUnary(*runtime, endpoint, ping, pingCallback);
    } else if (mode == "concurrent") {
        std::atomic<uint32_t> seen{0};
        std::vector<std::shared_ptr<Request>> requests(128);
        std::vector<uint64_t> callbackIds(128);
        std::vector<std::thread> writers;
        for (uint32_t writer = 0; writer < 8; ++writer) {
            writers.emplace_back([&, writer] {
                for (uint32_t i = writer; i < requests.size(); i += 8) {
                    rpc::PingRequest distinct;
                    distinct.set_client_timestamp_microseconds_since_epoch(123456789 + i);
                    requests[i] = submitUnary(*runtime, endpoint, distinct,
                        [&, i](const OmniUsdLiveRequestError& error, const rpc::PingResponse& response, uint64_t generation) {
                            assert(generation == 1 && error.result == OMNI_USD_LIVE_OK);
                            assert(response.client_timestamp_microseconds_since_epoch() == 123456789 + i);
                            callbackIds[i] = error.request_id;
                            ++seen;
                        });
                }
            });
        }
        for (auto& writer : writers)
            writer.join();
        for (uint32_t i = 0; i < requests.size(); ++i) {
            auto& pending = requests[i];
            assert(pending->wait(5000) == OMNI_USD_LIVE_OK);
            assert(callbackIds[i] == pending->handle);
            OmniUsdLiveResult result = OMNI_USD_LIVE_INTERNAL_ERROR;
            assert(pending->result(&result) == OMNI_USD_LIVE_OK && result == OMNI_USD_LIVE_OK);
            if (&pending != &requests.back())
                assert(runtime->requests->release(pending->handle) == OMNI_USD_LIVE_OK);
        }
        request = std::move(requests.back());
        callbacks = seen;
        assert(callbacks == 128);
        callbackError.result = OMNI_USD_LIVE_OK;
        callbackError.grpc_status = 0;
    } else if (mode == "capacity") {
        request = submitUnary(*runtime, endpoint, ping, pingCallback);
        assert(request->wait(5000) == OMNI_USD_LIVE_OK);
        rpc::CreateSessionRequest create;
        create.set_path("native/base.usda");
        create.set_name("must-not-execute");
        bool rejected = false;
        try {
            submitUnary(*runtime, endpoint, create, {});
        } catch (OmniUsdLiveResult result) {
            rejected = result == OMNI_USD_LIVE_RESOURCE_EXHAUSTED;
        }
        assert(rejected);
    } else if (mode == "roundtrip") {
        request = submitUnary(*runtime, endpoint, ping, pingCallback);
        release();
        assert(callbackError.result == OMNI_USD_LIVE_OK);
        std::string identifier;
        rpc::CreateSessionRequest create;
        create.set_path("native/base.usda");
        create.set_name("native-created");
        request = submitUnary(*runtime, endpoint, create,
            [&](const OmniUsdLiveRequestError& error, const rpc::CreateSessionResponse& response, uint64_t generation) {
                observe(error, generation);
                identifier = response.session_id();
            });
        create.set_name("caller-mutated-input"); // Async transport owns its wire request.
        release();
        assert(callbackError.result == OMNI_USD_LIVE_OK && !identifier.empty());
        rpc::ListSessionsRequest list;
        list.set_path("native/base.usda");
        request = submitUnary(*runtime, endpoint, list,
            [&](const OmniUsdLiveRequestError& error, const rpc::ListSessionsResponse& response, uint64_t generation) {
                observe(error, generation);
                assert(response.sessions_size() == 1);
                assert(response.sessions(0).session_id() == identifier);
                assert(response.sessions(0).name() == "native-created");
                assert(response.sessions(0).created_by_user_name() == "writer");
                assert(response.sessions(0).created_at() > 0);
            });
        release();
        assert(callbackError.result == OMNI_USD_LIVE_OK);
        rpc::DeleteSessionRequest remove;
        remove.set_session_id(identifier);
        request = submitUnary(*runtime, endpoint, remove,
            [&](const OmniUsdLiveRequestError& error, const rpc::DeleteSessionResponse&, uint64_t generation) {
                observe(error, generation);
            });
    } else if (mode.find("create") != std::string::npos) {
        rpc::CreateSessionRequest create;
        create.set_path("native/base.usda");
        create.set_name("native-created");
        request = submitUnary(*runtime, endpoint, create,
            [&](const OmniUsdLiveRequestError& error, const rpc::CreateSessionResponse&, uint64_t generation) {
                observe(error, generation);
            }, deadline ? 500 : 5000);
    } else if (mode == "stat" || mode == "delete-layer") {
        assert(argc == 7);
        if (mode == "stat") {
            rpc::StatLayerRequest stat;
            stat.set_session_id(argv[6]);
            stat.set_layer_name("root.live");
            request = submitUnary(*runtime, endpoint, stat,
                [&](const OmniUsdLiveRequestError& error, const rpc::StatLayerResponse& response, uint64_t generation) {
                    observe(error, generation);
                    if (error.result == OMNI_USD_LIVE_OK) {
                        assert(response.created_by_user_name() == "writer");
                        assert(response.created_at() > 0 && response.last_modified_at() > 0);
                    }
                });
        } else {
            rpc::DeleteLayerRequest remove;
            remove.set_session_id(argv[6]);
            remove.set_layer_name("root.live");
            request = submitUnary(*runtime, endpoint, remove,
                [&](const OmniUsdLiveRequestError& error, const rpc::DeleteLayerResponse&, uint64_t generation) {
                    observe(error, generation);
                });
        }
    } else {
        request = submitUnary(*runtime, endpoint, ping, pingCallback, deadline ? 500 : 5000);
    }
    std::cout << "ISSUED" << std::endl;
    if (cancel) {
        std::string command;
        std::getline(std::cin, command); // Fixture sends only after its fault barrier was reached.
        assert(command == "cancel");
        assert(request->cancel(5000) == OMNI_USD_LIVE_OK);
        assert(callbacks == 0);
    }
    assert(request->wait(6000) == OMNI_USD_LIVE_OK);
    OmniUsdLiveResult result = OMNI_USD_LIVE_INTERNAL_ERROR;
    assert(request->result(&result) == OMNI_USD_LIVE_OK && result == expected);
    auto error = record<OmniUsdLiveRequestError>();
    assert(request->error(&error) == OMNI_USD_LIVE_OK);
    assert(error.result == expected && error.grpc_status_present && error.grpc_status == expectedGrpc);
    assert(error.attempt_id == 1 && error.detail[0] == '\0');
    if (!cancel) {
        assert(callbacks == (mode == "roundtrip" ? 4 : mode == "concurrent" ? 128 : mode == "auth-rotate" ? 2 : 1));
        assert(callbackError.result == result && callbackError.grpc_status == expectedGrpc);
    }
    assert(omniUsdLiveUnregisterEndpoint(endpoint, 0) == OMNI_USD_LIVE_BUSY);
    // Late cancellation must not rewrite an authoritative completion.
    assert(request->cancel(0) == OMNI_USD_LIVE_OK);
    OmniUsdLiveResult afterCancel = OMNI_USD_LIVE_INTERNAL_ERROR;
    assert(request->result(&afterCancel) == OMNI_USD_LIVE_OK && afterCancel == result);
    release();
    if (authCallback) {
        assert(omniUsdLiveUnregisterCallback(authCallback, 3000) == OMNI_USD_LIVE_OK);
        assert(refresh.calls == 1);
    }
    assert(omniUsdLiveUnregisterEndpoint(endpoint, 0) == OMNI_USD_LIVE_OK);
    runtime.reset();
    assert(omniUsdLiveShutdown(5000) == OMNI_USD_LIVE_OK);
    std::cout << "{\"result\":" << result << ",\"grpc_status\":" << expectedGrpc
              << ",\"callbacks\":" << callbacks << "}" << std::endl;
}
