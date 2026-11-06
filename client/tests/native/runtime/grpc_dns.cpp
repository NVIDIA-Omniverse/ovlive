// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// LoopbackDnsServer.h includes winsock2.h, which must precede any windows.h.
#include "../support/LoopbackDnsServer.h"

#include "LiveRuntime.h"
#include "nvidia/omniverse/live/v1alpha/live.grpc.pb.h"
#include <grpcpp/grpcpp.h>

#include <atomic>
#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

using namespace omni::usd::live;
namespace {
namespace api = nvidia::omniverse::live::v1alpha;

class PingService final : public api::LiveService::Service {
public:
    std::atomic<uint32_t> pings{0};
    grpc::Status Ping(grpc::ServerContext*, const api::PingRequest*, api::PingResponse*) override {
        ++pings;
        return grpc::Status::OK;
    }
};
} // namespace

// An endpoint's channel resolves a hostname through a UDP DNS reply. Numeric addresses, "localhost" and hosts-file
// names never reach gRPC's c-ares socket code, so every other native test leaves it unexercised; here a reserved
// .test name resolves through a loopback DNS server that the channel target's authority names. The case guards the
// statically linked gRPC and c-ares pair: gRPC's Windows c-ares socket wrapper before 1.69.0 aborts with "Check
// failed: *from_len <= recv_from_source_addr_len_" on the first UDP reply from c-ares 1.34 (grpc/grpc#37969).
void grpcDnsTests() {
    constexpr const char* hostname = "grpc-dns.test";
    LoopbackDnsServer dns(hostname);
    assert(dns.start());

    PingService service;
    int serverPort = 0;
    grpc::ServerBuilder builder;
    builder.RegisterService(&service);
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &serverPort);
    std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
    assert(server && serverPort);

    assert(omniUsdLiveInitialize(nullptr) == OMNI_USD_LIVE_OK);
    {
        auto runtime = getRuntime();
        EndpointOptions options;
        options.authority.host = hostname;
        options.authority.port = static_cast<uint16_t>(serverPort);
        // The trailing dot makes the name fully qualified, so c-ares sends it as is instead of first trying the
        // machine's DNS search suffixes.
        options.authority.canonical = "dns://127.0.0.1:" + std::to_string(dns.port()) + "/" + hostname + ".:" +
            std::to_string(serverPort);
        options.secure = false;
        OmniUsdLiveEndpoint handle = 0;
        assert(runtime->registerEndpoint(options, &handle) == OMNI_USD_LIVE_OK);
        auto endpoint = runtime->findEndpoint(handle);
        assert(endpoint && endpoint->channel && endpoint->stub);

        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));
        context.set_wait_for_ready(true);
        api::PingRequest request;
        api::PingResponse response;
        const grpc::Status status = endpoint->stub->Ping(&context, request, &response);
        assert(status.ok());
        assert(service.pings >= 1);
        dns.stop();
        assert(dns.addressReplies() >= 1);
        assert(dns.errors() == 0);

        endpoint.reset();
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        for (;;) {
            const auto result = omniUsdLiveUnregisterEndpoint(handle, 0);
            if (result == OMNI_USD_LIVE_OK) break;
            assert(result == OMNI_USD_LIVE_BUSY && std::chrono::steady_clock::now() < deadline);
            std::this_thread::yield();
        }
    }
    assert(omniUsdLiveShutdown(3000) == OMNI_USD_LIVE_OK);
    server->Shutdown();
    std::cout << "grpc dns: an endpoint channel resolved " << hostname << " through a loopback UDP DNS reply\n";
}
