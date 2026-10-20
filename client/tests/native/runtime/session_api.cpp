// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "LiveRequest.h"
#include "LiveRuntime.h"
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>

void liveTestFailAllocationsAfter(size_t) noexcept;
void liveTestRestoreAllocations() noexcept;
using namespace omni::usd::live;
namespace {
template<class T> T record()
{ T out{}; out.struct_size = sizeof(out); out.abi_version = OMNI_USD_LIVE_ABI_VERSION; return out; }
void OMNI_USD_LIVE_CALL joined(void* user, const OmniUsdLiveSessionEvent*)
{ ++*static_cast<std::atomic<unsigned>*>(user); }
void OMNI_USD_LIVE_CALL accepted(void* user, OmniUsdLiveResult result)
{ assert(result == OMNI_USD_LIVE_OK); ++*static_cast<std::atomic<unsigned>*>(user); }
void done(OmniUsdLiveRequest request)
{
    assert(omniUsdLiveWait(request, 3000) == OMNI_USD_LIVE_OK);
    auto error = record<OmniUsdLiveRequestError>();
    assert(omniUsdLiveGetRequestError(request, &error) == OMNI_USD_LIVE_OK && error.result == OMNI_USD_LIVE_OK);
    assert(error.operation == OMNI_USD_LIVE_OPERATION_MESSAGE && error.attempt_id == 0 && !error.grpc_status_present);
    assert(omniUsdLiveReleaseRequest(request) == OMNI_USD_LIVE_OK);
}
void validationAndOfflineQueueOwnership()
{
    std::atomic<unsigned> events{0}, completions{0};
    auto options = record<OmniUsdLiveJoinOptions>(); options.app_name = "native-contract";
    const auto uri = makeSessionUri("127.0.0.1:35491", "offline-contract");
    OmniUsdLiveSession session = 99;
    assert(omniUsdLiveJoinSession(uri.c_str(), &options, &events, joined, &session) == OMNI_USD_LIVE_NOT_INITIALIZED && !session);
    assert(omniUsdLiveLeaveSession(0, 0) == OMNI_USD_LIVE_NOT_INITIALIZED);
    auto config = record<OmniUsdLiveRuntimeOptions>();
    config.max_update_bytes = 4;
    config.max_pending_bytes_per_endpoint = 8;
    config.max_pending_messages_per_session = 2;
    assert(omniUsdLiveInitialize(&config) == OMNI_USD_LIVE_OK);
    auto endpointOptions = record<OmniUsdLiveEndpointOptions>(); endpointOptions.authority = "127.0.0.1:35491";
    OmniUsdLiveEndpoint endpoint = 0;
    assert(omniUsdLiveRegisterEndpoint(&endpointOptions, &endpoint) == OMNI_USD_LIVE_OK);
    assert(omniUsdLiveJoinSession(uri.c_str(), nullptr, &events, joined, &session) == OMNI_USD_LIVE_INVALID_ARGUMENT && !session);
    assert(omniUsdLiveJoinSession(uri.c_str(), &options, &events, nullptr, &session) == OMNI_USD_LIVE_INVALID_ARGUMENT && !session);
    assert(omniUsdLiveJoinSession(nullptr, &options, &events, joined, &session) == OMNI_USD_LIVE_INVALID_ARGUMENT && !session);
    const auto layer = makeLayerUri(uri, "root.live");
    assert(omniUsdLiveJoinSession(layer.c_str(), &options, &events, joined, &session) == OMNI_USD_LIVE_INVALID_ARGUMENT);
    options.struct_size = sizeof(uint32_t);
    assert(omniUsdLiveJoinSession(uri.c_str(), &options, &events, joined, &session) == OMNI_USD_LIVE_INVALID_ARGUMENT);
    options.struct_size = sizeof(options); options.abi_version = UINT32_C(0x20000);
    assert(omniUsdLiveJoinSession(uri.c_str(), &options, &events, joined, &session) == OMNI_USD_LIVE_UNSUPPORTED);
    options.abi_version = OMNI_USD_LIVE_ABI_VERSION; options.app_name = "";
    assert(omniUsdLiveJoinSession(uri.c_str(), &options, &events, joined, &session) == OMNI_USD_LIVE_INVALID_ARGUMENT);
    options.app_name = "\xC0\x80";
    assert(omniUsdLiveJoinSession(uri.c_str(), &options, &events, joined, &session) == OMNI_USD_LIVE_INVALID_ARGUMENT);
    const std::string tooLong(kMaxUriBytes + 1, 'x'); options.app_name = tooLong.c_str();
    assert(omniUsdLiveJoinSession(uri.c_str(), &options, &events, joined, &session) == OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    options.app_name = "native-contract";
    const auto missing = makeSessionUri("127.0.0.1:35492", "missing-endpoint");
    assert(omniUsdLiveJoinSession(missing.c_str(), &options, &events, joined, &session) == OMNI_USD_LIVE_NOT_FOUND);

    assert(omniUsdLiveJoinSession(uri.c_str(), &options, &events, joined, &session) == OMNI_USD_LIVE_OK && session);
    assert((session >> 56) == 2);
    OmniUsdLiveSession other = 0;
    assert(omniUsdLiveJoinSession(uri.c_str(), &options, &events, joined, &other) == OMNI_USD_LIVE_OK && other != session);
    assert(omniUsdLiveUnregisterEndpoint(endpoint, 0) == OMNI_USD_LIVE_BUSY);
    auto status = record<OmniUsdLiveStatus>();
    assert(omniUsdLiveGetSessionStatus(session, &status) == OMNI_USD_LIVE_OK);
    assert(status.subject_kind == OMNI_USD_LIVE_SUBJECT_SESSION && status.subject_id);
    const auto prior = status;
    assert(omniUsdLiveGetSessionStatus(endpoint, &status) == OMNI_USD_LIVE_INVALID_HANDLE);
    assert(std::memcmp(&status, &prior, sizeof(status)) == 0);
    status.struct_size = sizeof(uint32_t);
    assert(omniUsdLiveGetSessionStatus(session, &status) == OMNI_USD_LIVE_INVALID_ARGUMENT);
    status = record<OmniUsdLiveStatus>(); status.abi_version = UINT32_C(0x20000);
    assert(omniUsdLiveGetSessionStatus(session, &status) == OMNI_USD_LIVE_UNSUPPORTED);
    assert(omniUsdLiveGetSessionStatus(session, nullptr) == OMNI_USD_LIVE_INVALID_ARGUMENT);
    {
        // A caller built against the 1.0 layout passes the prefix before
        // buffered_edit_count and receives exactly that prefix; a size that
        // splits the appended field is rejected before any write.
        constexpr uint32_t prefix = offsetof(OmniUsdLiveStatus, buffered_edit_count);
        struct OlderStatus { unsigned char bytes[prefix]; uint64_t tail; } older;
        const uint32_t abi = OMNI_USD_LIVE_ABI_VERSION;
        const auto prepare = [&](uint32_t size) {
            std::memset(&older, 0xAB, sizeof(older));
            std::memcpy(older.bytes, &size, sizeof(size)); std::memcpy(older.bytes + 4, &abi, sizeof(abi));
        };
        prepare(prefix);
        assert(omniUsdLiveGetSessionStatus(session, reinterpret_cast<OmniUsdLiveStatus*>(&older)) == OMNI_USD_LIVE_OK);
        assert(older.tail == UINT64_C(0xABABABABABABABAB));
        uint64_t subject = 0; uint32_t size = 0;
        std::memcpy(&size, older.bytes, sizeof(size));
        std::memcpy(&subject, older.bytes + offsetof(OmniUsdLiveStatus, subject_id), sizeof(subject));
        assert(size == prefix && subject == prior.subject_id);
        prepare(prefix + 4);
        const auto before = older;
        assert(omniUsdLiveGetSessionStatus(session, reinterpret_cast<OmniUsdLiveStatus*>(&older)) ==
               OMNI_USD_LIVE_INVALID_ARGUMENT);
        assert(std::memcmp(&older, &before, sizeof(older)) == 0);
    }

    OmniUsdLiveRequest request = 99;
    assert(omniUsdLiveSendMessage(session, nullptr, 1, &completions, accepted, &request) == OMNI_USD_LIVE_INVALID_ARGUMENT && !request);
    assert(omniUsdLiveSendMessage(endpoint, nullptr, 0, &completions, accepted, &request) == OMNI_USD_LIVE_INVALID_HANDLE && !request);
    assert(omniUsdLiveSendMessage(session, reinterpret_cast<const void*>(uintptr_t(1)), UINT64_MAX,
        &completions, accepted, &request) == OMNI_USD_LIVE_RESOURCE_EXHAUSTED && !request);
    std::array<unsigned char, 4> payload{0, 0xFF, 1, 2};
    assert(omniUsdLiveSendMessage(session, payload.data(), payload.size(), &completions, accepted, &request) == OMNI_USD_LIVE_OK);
    done(request); // Acceptance finishes without any token or transport attempt.
    assert(omniUsdLiveSendMessage(session, payload.data(), payload.size(), &completions, accepted, &request) == OMNI_USD_LIVE_OK);
    done(request);
    assert(omniUsdLiveSendMessage(session, nullptr, 0, &completions, accepted, &request) == OMNI_USD_LIVE_RESOURCE_EXHAUSTED && !request);
    assert(omniUsdLiveSendMessage(other, payload.data(), 1, &completions, accepted, &request) == OMNI_USD_LIVE_RESOURCE_EXHAUSTED && !request);
    assert(completions == 2 && events == 0);
    assert(omniUsdLiveLeaveSession(session, 3000) == OMNI_USD_LIVE_OK);
    assert(omniUsdLiveLeaveSession(session, 0) == OMNI_USD_LIVE_INVALID_HANDLE);
    status = record<OmniUsdLiveStatus>();
    assert(omniUsdLiveGetSessionStatus(session, &status) == OMNI_USD_LIVE_INVALID_HANDLE);
    assert(omniUsdLiveSendMessage(other, payload.data(), 1, &completions, accepted, &request) == OMNI_USD_LIVE_OK);
    done(request);
    assert(omniUsdLiveLeaveSession(other, 3000) == OMNI_USD_LIVE_OK);
    assert(completions == 3 && events == 0);
    assert(omniUsdLiveUnregisterEndpoint(endpoint, 3000) == OMNI_USD_LIVE_OK);
    assert(omniUsdLiveShutdown(3000) == OMNI_USD_LIVE_OK);
}
void admissionAllocationRollback()
{
    assert(omniUsdLiveInitialize(nullptr) == OMNI_USD_LIVE_OK);
    auto options = record<OmniUsdLiveEndpointOptions>(); options.authority = "127.0.0.1:35493";
    OmniUsdLiveEndpoint endpoint = 0;
    assert(omniUsdLiveRegisterEndpoint(&options, &endpoint) == OMNI_USD_LIVE_OK);
    const auto uri = makeSessionUri(options.authority, "allocation-rollback");
    auto join = record<OmniUsdLiveJoinOptions>(); join.app_name = "allocation-contract";
    std::atomic<unsigned> events{0}, receipts{0};
    unsigned rejectedJoins = 0, rejectedSends = 0;
    OmniUsdLiveSession session = 0;
    for (size_t permit = 0; permit < 128; ++permit) {
        liveTestFailAllocationsAfter(permit);
        const auto result = omniUsdLiveJoinSession(uri.c_str(), &join, &events, joined, &session);
        liveTestRestoreAllocations();
        if (result == OMNI_USD_LIVE_OK) break;
        assert(result == OMNI_USD_LIVE_RESOURCE_EXHAUSTED && !session);
        ++rejectedJoins;
        auto runtime = getRuntime();
        auto retained = runtime->findEndpoint(endpoint);
        std::lock_guard<std::mutex> lock(retained->mutex);
        assert(!retained->dependents && !retained->pendingTransportBytes);
    }
    assert(session && rejectedJoins >= 4);
    std::array<char, 64> payload{};
    OmniUsdLiveRequest request = 0;
    for (size_t permit = 0; permit < 128; ++permit) {
        liveTestFailAllocationsAfter(permit);
        const auto result = omniUsdLiveSendMessage(session, payload.data(), payload.size(), &receipts, accepted, &request);
        liveTestRestoreAllocations();
        if (result == OMNI_USD_LIVE_OK) break;
        assert(result == OMNI_USD_LIVE_RESOURCE_EXHAUSTED && !request);
        ++rejectedSends;
        auto status = record<OmniUsdLiveStatus>();
        assert(omniUsdLiveGetSessionStatus(session, &status) == OMNI_USD_LIVE_OK);
        assert(!status.pending_write_count && !status.pending_bytes);
        auto retained = getRuntime()->findEndpoint(endpoint);
        std::lock_guard<std::mutex> lock(retained->mutex);
        assert(!retained->pendingTransportBytes);
    }
    assert(request && rejectedSends >= 3);
    done(request);
    assert(omniUsdLiveLeaveSession(session, 3000) == OMNI_USD_LIVE_OK);
    assert(omniUsdLiveUnregisterEndpoint(endpoint, 3000) == OMNI_USD_LIVE_OK);
    assert(omniUsdLiveShutdown(3000) == OMNI_USD_LIVE_OK);
    assert(events == 0 && receipts == 1);
}
// Holds the first request completion's worker after dispatch returns, while
// the worker still retains the request, until the test releases it.
struct HeldCompletion {
    std::mutex mutex;
    std::condition_variable changed;
    bool held = false, released = false;
    static void hook(void* context, DispatchJob& job) noexcept
    {
        if (!dynamic_cast<Request*>(&job)) return;
        auto& self = *static_cast<HeldCompletion*>(context);
        std::unique_lock<std::mutex> lock(self.mutex);
        if (self.held) return;
        self.held = true;
        self.changed.notify_all();
        // Failure-only watchdog: the test releases the worker explicitly.
        if (!self.changed.wait_for(lock, std::chrono::seconds(30), [&] { return self.released; })) std::abort();
    }
};
void releasedRequestLeavesNoEndpointDependent()
{
    // One callback worker is held; the other runs the session's close.
    auto config = record<OmniUsdLiveRuntimeOptions>();
    config.callback_worker_count = 2;
    assert(omniUsdLiveInitialize(&config) == OMNI_USD_LIVE_OK);
    auto options = record<OmniUsdLiveEndpointOptions>(); options.authority = "127.0.0.1:35494";
    OmniUsdLiveEndpoint endpoint = 0;
    assert(omniUsdLiveRegisterEndpoint(&options, &endpoint) == OMNI_USD_LIVE_OK);
    const auto uri = makeSessionUri(options.authority, "released-request");
    auto join = record<OmniUsdLiveJoinOptions>(); join.app_name = "released-request";
    std::atomic<unsigned> events{0}, receipts{0};
    OmniUsdLiveSession session = 0;
    assert(omniUsdLiveJoinSession(uri.c_str(), &join, &events, joined, &session) == OMNI_USD_LIVE_OK);
    HeldCompletion hold;
    getRuntime()->requests->executor().setAfterDispatchForTest(HeldCompletion::hook, &hold);
    std::array<unsigned char, 4> payload{1, 2, 3, 4};
    OmniUsdLiveRequest request = 0;
    assert(omniUsdLiveSendMessage(session, payload.data(), payload.size(), &receipts, accepted, &request) == OMNI_USD_LIVE_OK);
    {
        std::unique_lock<std::mutex> lock(hold.mutex);
        assert(hold.changed.wait_for(lock, std::chrono::seconds(30), [&] { return hold.held; }));
    }
    // The worker that delivered the completion still retains the request.
    done(request);
    assert(omniUsdLiveLeaveSession(session, 3000) == OMNI_USD_LIVE_OK);
    assert(omniUsdLiveUnregisterEndpoint(endpoint, 3000) == OMNI_USD_LIVE_OK);
    getRuntime()->requests->executor().setAfterDispatchForTest(nullptr, nullptr);
    {
        std::lock_guard<std::mutex> lock(hold.mutex);
        hold.released = true;
        hold.changed.notify_all();
    }
    assert(omniUsdLiveShutdown(3000) == OMNI_USD_LIVE_OK);
    assert(events == 0 && receipts == 1);
}
}
void sessionApiTests()
{
    validationAndOfflineQueueOwnership();
    admissionAllocationRollback();
    releasedRequestLeavesNoEndpointDependent();
    std::cout << "session ABI: validation, parked joins, local acceptance, quota ownership and released-request endpoint dependents passed\n";
}
