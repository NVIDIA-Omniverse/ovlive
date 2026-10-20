// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "LiveUnary.h"
#include "LiveDrain.h"

namespace omni::usd::live {
// Owner-controlled read pacing. While held, a read that has delivered a
// complete frame (or started) issues no next Read until the wake release()
// returns runs; HTTP/2 flow control then holds the service. Cancelling a
// parked read finishes it.
class LayerReadFlow final {
public:
    void hold() noexcept;
    // Clears the hold and returns the parked read's wake for the caller to
    // run after releasing its own locks; the flow runs nothing itself.
    std::function<void()> release() noexcept;
    // True: parked; the caller issues no Read until wake runs.
    bool park(std::function<void()> wake);
private:
    std::mutex mutex_;
    bool held_ = false;
    std::function<void()> parked_;
};
struct LayerReadOptions {
    uint64_t objectId = 0;
    uint64_t sequence = 0;
    // Local diagnostic correlation only; never serialized into the request.
    uint64_t subjectId = 0, streamGeneration = 0;
    // Only a new logical attachment may adopt a replacement initial object.
    // Reconnects of an attached incarnation leave this false.
    bool permitInitialReplacement = false;
    uint32_t timeoutMs = 0; // Zero keeps this stream attempt open until stopped.
    std::shared_ptr<DrainState> drain; // Optional private attachment pipeline.
    std::shared_ptr<LayerReadFlow> flow; // Optional owner pacing.
};

// Immutable shared delivery. Consumers may retain it for ordered application;
// its endpoint byte reservation then outlives transport/request cancellation.
// Sequence is received metadata, never proof of contiguous/applied coverage.
struct LayerReadUpdate {
    PendingBytes budget;
    rpc::SubscribeReadLayerHeader header;
    std::string bytes;
    bool caughtUp = false;
    // Local attempt provenance, not a server-supplied latest flag. Readiness
    // requires the selected service contract as well as validated coverage.
    bool firstResponse = false;
    std::shared_ptr<DrainTicket> drain; // Complete-frame ticket follows application.
    LayerReadUpdate(PendingBytes reservation, const rpc::SubscribeReadLayerHeader& metadata)
        : budget(std::move(reservation)), header(metadata), caughtUp(metadata.total_size() == 0) {}
};
struct LayerReadCallbacks {
    std::function<void(std::shared_ptr<const LayerReadUpdate>)> update;
    std::function<void(const OmniUsdLiveRequestError&, std::shared_ptr<DrainTicket>)> closed;
    // Optional: this stream's own validation or resource failure. Returns
    // true when the owner latched the stream's drain itself, so it can record
    // under its own lock whether this failure won the latch; false leaves
    // latching to the read. Invoked on a transport worker; it must not throw
    // or block.
    std::function<bool(OmniUsdLiveResult)> failed;
};
class LayerReadTransport {
public:
    virtual ~LayerReadTransport() = default;
    virtual std::unique_ptr<grpc::ClientAsyncReaderInterface<rpc::SubscribeReadLayerResponse>> prepare(
        grpc::ClientContext*, const rpc::SubscribeReadLayerRequest&, grpc::CompletionQueue*) = 0;
};

// One native subscription attempt; no automatic reconnect or invented gap
// coverage. The layer owner supplies the safe resume cursor and recovery gate.
// Header/chunk assembly and serialized callbacks run on a worker, never the CQ.
// On clean unexpected EOF, reports UNAVAILABLE; successful truncation is
// MALFORMED_STREAM. Non-OK interruption discards only the partial assembly.
std::shared_ptr<Request> subscribeLayerRead(Runtime& runtime, OmniUsdLiveEndpoint endpoint, std::string_view sessionId,
    std::string_view layerName, LayerReadOptions options, LayerReadCallbacks callbacks,
    LayerReadTransport* transport = nullptr);
} // namespace omni::usd::live
