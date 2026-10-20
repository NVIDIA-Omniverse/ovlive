// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "LiveLayerRead.h"
#include "LiveDiagnostics.h"
#include <openssl/sha.h>
#include <cassert>
#include <future>
#include <iostream>

using namespace omni::usd::live;
namespace {
template <class T> T record()
{
    T value{}; value.struct_size = sizeof(value); value.abi_version = OMNI_USD_LIVE_ABI_VERSION; return value;
}
struct Observation {
    uint64_t object, sequence, size;
    bool caught;
    std::string digest;
};
struct Receiver {
    std::mutex mutex;
    std::condition_variable changed;
    std::vector<Observation> observations;
    std::vector<std::shared_ptr<const LayerReadUpdate>> retained;
    unsigned closed = 0;
    bool release = false;
    void wait(size_t count)
    {
        std::unique_lock<std::mutex> lock(mutex);
        assert(changed.wait_for(lock, std::chrono::seconds(4), [&] { return observations.size() >= count || closed; }));
        assert(observations.size() >= count);
    }
};
std::string digest(std::string_view bytes)
{
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(bytes.data()), bytes.size(), hash);
    std::string result;
    const char* digits = "0123456789abcdef";
    for (auto value : hash) { result += digits[value >> 4]; result += digits[value & 15]; }
    return result;
}
}
int layerReadDriver(int argc, char** argv)
{
    assert(argc == 7);
    const std::string mode = argv[2];
    const auto expected = static_cast<OmniUsdLiveResult>(std::stoul(argv[3]));
    const int expectedGrpc = std::stoi(argv[4]);
    auto options = record<OmniUsdLiveRuntimeOptions>();
    options.max_update_bytes = 6 * 1024 * 1024;
    // read-adopt-large: an endpoint budget above the write limit, for a read update larger than any upload.
    options.max_pending_bytes_per_endpoint = (mode == "read-retain" ? 12 : mode == "read-adopt-large" ? 16 : 8) * 1024 * 1024;
    assert(omniUsdLiveInitialize(&options) == OMNI_USD_LIVE_OK);
    auto ep = record<OmniUsdLiveEndpointOptions>(); ep.authority = argv[1]; ep.access_token = argv[5];
    OmniUsdLiveEndpoint endpoint = 0;
    assert(omniUsdLiveRegisterEndpoint(&ep, &endpoint) == OMNI_USD_LIVE_OK);
    auto runtime = getRuntime();
    auto retainedEndpoint = runtime->findEndpoint(endpoint);
    Receiver seen;
    const auto caller = std::this_thread::get_id();
    const bool retain = mode == "read-retain" || mode == "read-backpressure";
    std::promise<std::shared_ptr<Request>> admitted;
    auto readyRequest = admitted.get_future().share();
    LayerReadCallbacks callbacks;
    callbacks.update = [&](std::shared_ptr<const LayerReadUpdate> update) {
        assert(std::this_thread::get_id() != caller);
        assert(update->caughtUp == update->bytes.empty());
        assert(update->bytes.size() == update->header.total_size());
        auto hash = digest(update->bytes);
        {
            std::unique_lock<std::mutex> lock(seen.mutex);
            assert(seen.observations.size() < 16);
            seen.observations.push_back({update->header.object_id(), update->header.sequence_num(), update->bytes.size(), update->caughtUp, hash});
            if (retain) seen.retained.push_back(update);
            seen.changed.notify_all();
            if (mode == "read-block") seen.changed.wait(lock, [&] { return seen.release; });
        }
        if (mode == "read-throw") throw std::runtime_error("synthetic callback failure");
        if (mode == "read-self-cancel") assert(readyRequest.get()->cancel(0) == OMNI_USD_LIVE_BUSY);
    };
    callbacks.closed = [&](const OmniUsdLiveRequestError& error, std::shared_ptr<DrainTicket>) {
        assert(std::this_thread::get_id() != caller);
        if (error.result != expected) std::cerr << "read result=" << error.result << " raw=" << error.grpc_status << " expected=" << expected << std::endl;
        assert(error.result == expected && error.request_id && error.attempt_id == 1);
        std::lock_guard<std::mutex> lock(seen.mutex);
        ++seen.closed; seen.changed.notify_all();
    };
    LayerReadOptions settings; settings.timeoutMs = mode == "read-deadline" ? 1000 : 5000;
    const bool adopt = mode == "read-adopt" || mode == "read-adopt-large";
    if (mode == "read-caughtup" || mode == "read-expected-object" || adopt) {
        settings.objectId = 1; settings.sequence = 1;
        settings.permitInitialReplacement = adopt;
    }
    auto request = subscribeLayerRead(*runtime, endpoint, argv[6], "root.live", settings, std::move(callbacks));
    admitted.set_value(request);
    if (mode == "read-cancel-before") {
        std::string command; std::getline(std::cin, command); assert(command == "cancel");
        assert(request->cancel(5000) == OMNI_USD_LIVE_OK);
    } else if (mode == "read-roundtrip" || mode == "read-retain" || mode == "read-block" || mode == "read-caughtup" || adopt) {
        // The reference service ends catch-up with one empty caught-up header
        // and then holds the stream, so waiting for that terminator fixes the
        // delivered set before cancellation.
        seen.wait(mode == "read-roundtrip" || mode == "read-retain" ? 3 : 1);
        if (mode == "read-block") {
            assert(request->cancel(0) == OMNI_USD_LIVE_TIMEOUT);
            assert(request->wait(0) == OMNI_USD_LIVE_TIMEOUT);
            // The held consumer must not block CQ progress or another worker.
            rpc::PingRequest ping;
            auto other = submitUnary(*runtime, endpoint, ping, UnaryCallback<rpc::PingResponse>{});
            assert(other->wait(3000) == OMNI_USD_LIVE_OK);
            OmniUsdLiveResult result;
            assert(other->result(&result) == OMNI_USD_LIVE_OK && result == OMNI_USD_LIVE_OK);
            assert(runtime->requests->release(other->handle) == OMNI_USD_LIVE_OK);
            std::lock_guard<std::mutex> lock(seen.mutex); seen.release = true; seen.changed.notify_all();
        } else
            assert(request->cancel(5000) == OMNI_USD_LIVE_OK);
    }
    assert(request->wait(6000) == OMNI_USD_LIVE_OK);
    auto error = record<OmniUsdLiveRequestError>();
    assert(request->error(&error) == OMNI_USD_LIVE_OK && error.result == expected);
    assert(error.grpc_status_present && error.grpc_status == expectedGrpc && error.attempt_id == 1);
    assert(runtime->requests->release(request->handle) == OMNI_USD_LIVE_OK);
    request.reset();
    // The shared future deliberately held an independent request reference.
    // Release it before testing endpoint retention by content alone.
    readyRequest = {};
    admitted = {};
    {
        std::lock_guard<std::mutex> lock(retainedEndpoint->mutex);
        assert(retain ? retainedEndpoint->pendingTransportBytes > 0 : retainedEndpoint->pendingTransportBytes == 0);
    }
    if (retain) {
        const auto bytes = runtime->diagnostics->snapshot(0, 0).metrics.pendingBytes;
        assert(bytes.reservations.current == seen.retained.size() && bytes.currentBytes.low > 0 && bytes.currentBytes.high == 0);
        assert(runtime->diagnostics->snapshot(0, 0).metrics.requestSlots[0].current == 0);
        assert(omniUsdLiveUnregisterEndpoint(endpoint, 0) == OMNI_USD_LIVE_BUSY);
        assert(seen.retained.size() == seen.observations.size());
        for (size_t i = 0; i < seen.retained.size(); ++i) {
            const auto& update = *seen.retained[i];
            const auto& observed = seen.observations[i];
            assert(update.header.object_id() == observed.object);
            assert(update.header.sequence_num() == observed.sequence);
            assert(update.header.total_size() == observed.size && update.bytes.size() == observed.size);
            assert(update.caughtUp == observed.caught && digest(update.bytes) == observed.digest);
        }
        if (mode == "read-retain") {
            assert(seen.retained.size() == 3);
            assert(!seen.observations[0].caught && !seen.observations[1].caught && seen.observations[2].caught);
            assert(seen.observations[0].digest != seen.observations[1].digest);
        }
        seen.retained.clear();
        const auto retired = runtime->diagnostics->snapshot(0, 0).metrics.pendingBytes;
        assert(retired.reservations.current == 0 && retired.currentBytes.low == 0 && retired.currentBytes.high == 0);
        std::lock_guard<std::mutex> lock(retainedEndpoint->mutex);
        assert(retainedEndpoint->pendingTransportBytes == 0);
    }
    retainedEndpoint.reset();
    assert(omniUsdLiveUnregisterEndpoint(endpoint, 0) == OMNI_USD_LIVE_OK);
    runtime.reset();
    assert(omniUsdLiveShutdown(5000) == OMNI_USD_LIVE_OK);
    std::cout << "{\"result\":" << error.result << ",\"grpc_status\":" << error.grpc_status
              << ",\"closed\":" << seen.closed << ",\"updates\":[";
    for (size_t i = 0; i < seen.observations.size(); ++i) {
        const auto& item = seen.observations[i];
        if (i) std::cout << ',';
        std::cout << "{\"object\":" << item.object << ",\"sequence\":" << item.sequence
                  << ",\"size\":" << item.size << ",\"caught\":" << (item.caught ? "true" : "false")
                  << ",\"sha256\":\"" << item.digest << "\"}";
    }
    std::cout << "]}" << std::endl;
    return 0;
}
