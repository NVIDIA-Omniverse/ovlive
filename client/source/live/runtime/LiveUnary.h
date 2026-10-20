// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "LiveRequest.h"
#include <optional>

namespace omni::usd::live {
namespace rpc = nvidia::omniverse::live::v1alpha;
OmniUsdLiveResult classifyUnary(grpc::StatusCode code, bool mutation) noexcept;
// A transport failure (ok=false) at Finish never reaches a gRPC status: the
// service may have committed a mutation it never acknowledged, so that case
// is OUTCOME_UNKNOWN; a read has nothing to have committed.
OmniUsdLiveResult classifyTransportFailure(bool mutation) noexcept;
// Runtime entropy prevents synchronized clients. Tests supply a recorded seed
// string directly to reproduce the same bounded delay sequence.
uint64_t retryDelay(std::string_view seed, uint64_t request, uint64_t attempt, uint32_t initialMs,
    uint32_t maximumMs) noexcept;
// The backoff before reconnect attempt `attempt` (from 1) of a layer read or
// session join: uniform in [0, 2 × min(maximumMs, initialMs × 2^(attempt−1))),
// so its mean is the nominal step and its longest wait twice maximumMs. The
// draw is a deterministic hash of seed, subject and attempt, so the subjects
// of one runtime spread and a recorded seed reproduces every wait.
uint64_t reconnectDelay(std::string_view seed, uint64_t subject, uint64_t attempt, uint32_t initialMs,
    uint32_t maximumMs) noexcept;
// The backoff before reissue `attempt` (from 1) of a write or creation the
// service refused with a retry delay of `hintMs`: the longer of the hint and the attempt's own
// reconnectDelay draw, capped at maximumMs. The service randomizes its retry
// delay, so nothing is added on top of the hint.
uint64_t hintedReconnectDelay(std::string_view seed, uint64_t subject, uint64_t attempt, uint64_t hintMs,
    uint32_t initialMs, uint32_t maximumMs) noexcept;
// The wait before a layer read's or session join's reconnect attempt, or a
// layer write's or creation's reissue after a refusal that applied nothing,
// from the runtime's nonce and reconnect options. Every reconnect and reissue
// path schedules through this function and nothing else, so it is the single
// place where the service's retry delay, `hintMs`, replaces the computed
// backoff.
uint64_t reconnectWait(const Runtime& runtime, uint64_t subject, uint64_t attempt,
    std::optional<uint64_t> hintMs = std::nullopt) noexcept;

// Private single-attempt transport primitive, not the public submission policy.
// The future auth/retry controller must wait for usable credentials before
// calling this seam and decide whether a read/rejected mutation is retryable.
// Issued ambiguous mutations are never retried here. Response/error references
// are callback-scoped; callbacks and capture cleanup run on request workers.
template<class Response>
using UnaryCallback = std::function<void(const OmniUsdLiveRequestError&, const Response&, uint64_t tokenGeneration)>;

#define LIVE_DECLARE_UNARY(Name) \
    std::shared_ptr<Request> submitUnary(Runtime& runtime, OmniUsdLiveEndpoint endpoint, \
        const rpc::Name##Request& input, UnaryCallback<rpc::Name##Response> callback, uint32_t timeoutMs = 0);
LIVE_DECLARE_UNARY(Ping)
LIVE_DECLARE_UNARY(ListSessions)
LIVE_DECLARE_UNARY(CreateSession)
LIVE_DECLARE_UNARY(DeleteSession)
LIVE_DECLARE_UNARY(StatLayer)
LIVE_DECLARE_UNARY(DeleteLayer)
#undef LIVE_DECLARE_UNARY
// Uses the runtime's single auxiliary timing slot, never ordinary request
// capacity. Competing probes fail admission and may skip this sampling period.
struct UnaryTimingReceipt {
    std::chrono::steady_clock::time_point completed{};
    std::atomic<bool> ready{false};
};
std::shared_ptr<Request> submitTimingPing(Runtime& runtime, OmniUsdLiveEndpoint endpoint, const rpc::PingRequest&,
    UnaryCallback<rpc::PingResponse>, uint32_t timeoutMs, std::shared_ptr<UnaryTimingReceipt> receipt = {});
} // namespace omni::usd::live
