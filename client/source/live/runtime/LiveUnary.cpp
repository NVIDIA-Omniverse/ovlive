// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "LiveDeadline.h"
#include "LiveUnary.h"
#include "LiveAuth.h"
#include "LiveDiagnostics.h"
#include <stdexcept>

namespace omni::usd::live {
namespace {
// A deterministic 64-bit draw for one (seed, key, attempt) triple.
uint64_t backoffDraw(std::string_view seed, uint64_t key, uint64_t attempt) noexcept {
    uint64_t mix = UINT64_C(14695981039346656037);
    for (unsigned char byte : seed) {
        mix ^= byte;
        mix *= UINT64_C(1099511628211);
    }
    mix ^= key ^ (attempt * UINT64_C(0x9e3779b97f4a7c15));
    mix ^= mix >> 30;
    mix *= UINT64_C(0xbf58476d1ce4e5b9);
    mix ^= mix >> 27;
    mix *= UINT64_C(0x94d049bb133111eb);
    mix ^= mix >> 31;
    return mix;
}
// The nominal step of an attempt counted from 1: initialMs doubled per
// attempt and capped at maximumMs.
uint64_t backoffStep(uint64_t attempt, uint32_t initialMs, uint32_t maximumMs) noexcept {
    uint64_t delay = std::min(initialMs, maximumMs);
    for (uint64_t i = 1; i < attempt && delay < maximumMs && delay; ++i)
        delay = std::min<uint64_t>(delay * 2, maximumMs);
    return delay;
}
} // namespace

uint64_t retryDelay(std::string_view seed, uint64_t request, uint64_t attempt, uint32_t initialMs,
    uint32_t maximumMs) noexcept {
    const auto mix = backoffDraw(seed, request, attempt);
    const auto delay = backoffStep(attempt, initialMs, maximumMs);
    return delay / 2 + mix % (delay - delay / 2 + 1);
}

uint64_t reconnectDelay(std::string_view seed, uint64_t subject, uint64_t attempt, uint32_t initialMs,
    uint32_t maximumMs) noexcept {
    const auto window = 2 * backoffStep(attempt, initialMs, maximumMs);
    return window ? backoffDraw(seed, subject, attempt) % window : 0;
}

uint64_t hintedReconnectDelay(std::string_view seed, uint64_t subject, uint64_t attempt, uint64_t hintMs,
    uint32_t initialMs, uint32_t maximumMs) noexcept {
    const auto own = reconnectDelay(seed, subject, attempt, initialMs, maximumMs);
    return std::min<uint64_t>(std::max(hintMs, own), maximumMs);
}

uint64_t reconnectWait(const Runtime& runtime, uint64_t subject, uint64_t attempt,
    std::optional<uint64_t> hintMs) noexcept {
    const auto& values = runtime.options.values;
    if (hintMs)
        return hintedReconnectDelay(runtime.instanceNonce, subject, attempt, *hintMs, values.reconnect_initial_ms,
            values.reconnect_maximum_ms);
    return reconnectDelay(runtime.instanceNonce, subject, attempt, values.reconnect_initial_ms,
        values.reconnect_maximum_ms);
}

OmniUsdLiveResult classifyUnary(grpc::StatusCode code, bool mutation) noexcept {
    // A definitive rejection proves the service did not apply the attempt it
    // evaluated (design section 13); the caller keeps the gRPC code as
    // evidence. The service authenticates a request and checks write
    // permission before it commits (design section 20), so UNAUTHENTICATED and
    // PERMISSION_DENIED are pre-execution, as are its request validation
    // (INVALID_ARGUMENT, OUT_OF_RANGE), missing layer (NOT_FOUND), replaced
    // object (FAILED_PRECONDITION), quota (RESOURCE_EXHAUSTED) and existing
    // name (ALREADY_EXISTS) rejections. Every other code may follow a commit
    // and is OUTCOME_UNKNOWN. The reference service's privileged ledger must
    // never influence this decision.
    if (mutation) {
        switch (code) {
        case grpc::StatusCode::OK: return OMNI_USD_LIVE_OK;
        case grpc::StatusCode::INVALID_ARGUMENT:
        case grpc::StatusCode::OUT_OF_RANGE: return OMNI_USD_LIVE_INVALID_ARGUMENT;
        case grpc::StatusCode::NOT_FOUND: return OMNI_USD_LIVE_NOT_FOUND;
        case grpc::StatusCode::FAILED_PRECONDITION: return OMNI_USD_LIVE_OBJECT_REPLACED;
        case grpc::StatusCode::PERMISSION_DENIED: return OMNI_USD_LIVE_PERMISSION_DENIED;
        case grpc::StatusCode::UNAUTHENTICATED: return OMNI_USD_LIVE_UNAUTHENTICATED;
        case grpc::StatusCode::RESOURCE_EXHAUSTED: return OMNI_USD_LIVE_RESOURCE_EXHAUSTED;
        case grpc::StatusCode::ALREADY_EXISTS: return OMNI_USD_LIVE_ALREADY_EXISTS;
        default: return OMNI_USD_LIVE_OUTCOME_UNKNOWN;
        }
    }
    switch (code) {
    case grpc::StatusCode::OK: return OMNI_USD_LIVE_OK;
    case grpc::StatusCode::INVALID_ARGUMENT: return OMNI_USD_LIVE_INVALID_ARGUMENT;
    case grpc::StatusCode::OUT_OF_RANGE: return OMNI_USD_LIVE_INVALID_ARGUMENT;
    case grpc::StatusCode::NOT_FOUND: return OMNI_USD_LIVE_NOT_FOUND;
    case grpc::StatusCode::ALREADY_EXISTS: return OMNI_USD_LIVE_ALREADY_EXISTS;
    case grpc::StatusCode::PERMISSION_DENIED: return OMNI_USD_LIVE_PERMISSION_DENIED;
    case grpc::StatusCode::UNAUTHENTICATED: return OMNI_USD_LIVE_UNAUTHENTICATED;
    case grpc::StatusCode::UNIMPLEMENTED: return OMNI_USD_LIVE_UNSUPPORTED;
    default: break;
    }
    switch (code) {
    case grpc::StatusCode::CANCELLED: return OMNI_USD_LIVE_CANCELLED;
    case grpc::StatusCode::DEADLINE_EXCEEDED: return OMNI_USD_LIVE_TIMEOUT;
    case grpc::StatusCode::RESOURCE_EXHAUSTED: return OMNI_USD_LIVE_RESOURCE_EXHAUSTED;
    case grpc::StatusCode::UNAVAILABLE: return OMNI_USD_LIVE_UNAVAILABLE;
    default: return OMNI_USD_LIVE_INTERNAL_ERROR;
    }
}

OmniUsdLiveResult classifyTransportFailure(bool mutation) noexcept {
    return mutation ? OMNI_USD_LIVE_OUTCOME_UNKNOWN : OMNI_USD_LIVE_INTERNAL_ERROR;
}

namespace {
template<class Response> struct Payload {
    Response response;
    OmniUsdLiveRequestError error{};
    uint64_t tokenGeneration = 0;
    UnaryCallback<Response> callback;
};

template<class Response> struct Call {
    grpc::ClientContext context;
    grpc::Status status;
    std::unique_ptr<grpc::ClientAsyncResponseReader<Response>> reader;
    std::shared_ptr<Request> request;
    std::shared_ptr<Payload<Response>> payload;
    std::function<void()> delivery;
    bool mutation = false;
    AuthRegistry* auth = nullptr;
    DiagnosticsRegistry* diagnostics = nullptr;
    std::chrono::steady_clock::time_point started;
    EndpointLease endpoint;
    std::shared_ptr<UnaryTimingReceipt> timingReceipt;
};

template<class Response> struct Tag final : CompletionTag {
    std::shared_ptr<Call<Response>> call;
    explicit Tag(std::shared_ptr<Call<Response>> value) : call(std::move(value)) {}
    void onComplete(bool ok, Clock::time_point received) noexcept override {
        const auto completed = received;
        if (call->timingReceipt) {
            call->timingReceipt->completed = completed;
            call->timingReceipt->ready.store(true, std::memory_order_release);
        }
        auto& error = call->payload->error;
        error.struct_size = sizeof(error);
        error.abi_version = OMNI_USD_LIVE_ABI_VERSION;
        error.operation = call->request->operation;
        error.request_id = call->request->handle;
        error.attempt_id = 1; // This primitive issues exactly one attempt.
        error.grpc_status_present = ok ? 1 : 0;
        error.grpc_status = ok ? call->status.error_code() : 0;
        error.result = ok ? classifyUnary(call->status.error_code(), call->mutation) :
                            classifyTransportFailure(call->mutation);
        call->diagnostics->observeRpc(call->endpoint->handle, error.operation, error.result,
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::milliseconds>(completed - call->started).count()),
            error.request_id, 1, error.grpc_status_present, error.grpc_status, 0, 0, call->payload->tokenGeneration);
        const auto outcome = error;
        if (ok) call->auth->completed(*call->endpoint, call->payload->tokenGeneration, call->status.error_code());
        // Transfer the only payload/callback owner to the worker BEFORE enqueue.
        // Otherwise a fast worker could report quiescence while a CQ reference
        // still owns host captures, destroying those captures on the CQ later.
        call->payload.reset();
        // The endpoint's last use on this call is above. The request's own
        // lease covers the dependency until its handle is released, and the
        // gRPC context keeps the channel alive for the reader's destruction.
        { EndpointLease retired = std::move(call->endpoint); }
        call->request->finish(outcome, std::move(call->delivery));
    }
};

template<class Input, class Response, class Prepare>
std::shared_ptr<Request> submit(Runtime& runtime, OmniUsdLiveEndpoint endpoint, const Input& input,
    UnaryCallback<Response> callback, uint32_t timeoutMs, uint32_t operation, bool mutation, Prepare prepare,
    bool timing = false, std::shared_ptr<UnaryTimingReceipt> receipt = {}) {
    // Hold admission against final unregister while preparing the transport.
    auto lease = runtime.acquireEndpoint(endpoint);
    if (!lease) throw OmniUsdLiveResult(runtime.closing ? OMNI_USD_LIVE_BUSY : OMNI_USD_LIVE_INVALID_HANDLE);
    auto call = std::make_shared<Call<Response>>();
    call->mutation = mutation;
    call->timingReceipt = std::move(receipt);
    call->auth = runtime.auth.get();
    call->diagnostics = runtime.diagnostics.get();
    uint64_t tokenGeneration = 0;
    {
        std::lock_guard<std::mutex> lock(lease->mutex);
        runtime.auth->evaluate(*lease, wallTimeMs(), std::chrono::steady_clock::now());
        if (!runtime.auth->usable(*lease, wallTimeMs())) throw OmniUsdLiveResult(OMNI_USD_LIVE_UNAUTHENTICATED);
        call->context.AddMetadata("authorization", "Bearer " + lease->token);
        tokenGeneration = lease->tokenGeneration;
    }
    call->context.set_deadline(
        rpcDeadlineAfter(std::chrono::milliseconds(timeoutMs ? timeoutMs : runtime.options.values.request_timeout_ms)));
    call->context.set_wait_for_ready(false);
    call->payload = std::make_shared<Payload<Response>>();
    call->payload->tokenGeneration = tokenGeneration;
    call->payload->callback = std::move(callback);
    call->delivery = [payload = call->payload] {
        if (payload->callback) payload->callback(payload->error, payload->response, payload->tokenGeneration);
    };
    auto tag = std::make_unique<Tag<Response>>(call);
    // Admit before preparing: gRPC serializes the input into the call when it
    // prepares the reader, and a prepared reader discarded without StartCall
    // never releases that buffer, so no refusal may follow preparation.
    std::weak_ptr<Call<Response>> weak = call;
    auto request = runtime.requests->create(endpoint, operation, [weak] {
        if (auto active = weak.lock()) active->context.TryCancel();
    }, timing);
    try {
        // Prepare serializes/copies the input but issues no network operation.
        call->reader = prepare(*lease->stub, &call->context, input, &runtime.queue());
    } catch (...) {
        call->reader.reset(); // A throwing preparation is refused like a missing reader.
    }
    if (!call->reader) {
        // Release the admitted slot; the caller never receives this handle.
        OmniUsdLiveRequestError refused{};
        refused.result = OMNI_USD_LIVE_RESOURCE_EXHAUSTED;
        request->finish(refused, {});
        runtime.requests->release(request->handle);
        throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    }
    call->request = request;
    call->endpoint = std::move(lease);
    // Every fallible step precedes issuance: a failure between admission and
    // issuance releases what it admitted and issues nothing. Generated gRPC
    // StartCall/Finish register the preallocated completion tag with the CQ.
    call->started = std::chrono::steady_clock::now();
    call->diagnostics->startedRpc(operation);
    call->reader->StartCall();
    // Read before submitTag: its release is the last access before completion
    // can reset the payload on the poller.
    auto* response = &call->payload->response;
    auto* status = &call->status;
    call->reader->Finish(response, status, submitTag(tag, *runtime.diagnostics));
    return request;
}
} // namespace

#define LIVE_DEFINE_UNARY(Name, Operation, Mutation) \
std::shared_ptr<Request> submitUnary(Runtime& runtime, OmniUsdLiveEndpoint endpoint, \
    const rpc::Name##Request& input, UnaryCallback<rpc::Name##Response> callback, uint32_t timeoutMs) \
{ \
    return submit(runtime, endpoint, input, std::move(callback), timeoutMs, Operation, Mutation, \
        [](rpc::LiveService::Stub& stub, grpc::ClientContext* context, \
           const rpc::Name##Request& message, grpc::CompletionQueue* queue) { \
            return stub.PrepareAsync##Name(context, message, queue); \
        }); \
}
LIVE_DEFINE_UNARY(Ping, OMNI_USD_LIVE_OPERATION_PING, false)
LIVE_DEFINE_UNARY(ListSessions, OMNI_USD_LIVE_OPERATION_LIST, false)
LIVE_DEFINE_UNARY(CreateSession, OMNI_USD_LIVE_OPERATION_CREATE, true)
LIVE_DEFINE_UNARY(DeleteSession, OMNI_USD_LIVE_OPERATION_DELETE, true)
LIVE_DEFINE_UNARY(StatLayer, OMNI_USD_LIVE_OPERATION_STAT, false)
LIVE_DEFINE_UNARY(DeleteLayer, OMNI_USD_LIVE_OPERATION_DELETE, true)
#undef LIVE_DEFINE_UNARY
std::shared_ptr<Request> submitTimingPing(Runtime& runtime, OmniUsdLiveEndpoint endpoint, const rpc::PingRequest& input,
    UnaryCallback<rpc::PingResponse> callback, uint32_t timeoutMs, std::shared_ptr<UnaryTimingReceipt> receipt) {
    return submit(runtime, endpoint, input, std::move(callback), timeoutMs, OMNI_USD_LIVE_OPERATION_PING, false,
        [](rpc::LiveService::Stub& stub, grpc::ClientContext* context, const rpc::PingRequest& message,
            grpc::CompletionQueue* queue) { return stub.PrepareAsyncPing(context, message, queue); },
        true, std::move(receipt));
}
} // namespace omni::usd::live
