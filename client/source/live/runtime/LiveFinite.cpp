// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "LiveDeadline.h"
#include "LiveAuth.h"
#include "LiveUnary.h"
#include "LiveDiagnostics.h"
#include "LiveMetadata.h"
#include "LiveExceptionResult.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <optional>
#include <unordered_set>

namespace omni::usd::live {
namespace {
using Clock = std::chrono::steady_clock;
thread_local Clock::time_point metadataDeadline = Clock::time_point::max();

void textValid(std::string_view value, bool empty = false) {
    if (value.size() > kMaxUriBytes) throw std::length_error("metadata text capacity");
    if (!empty && value.empty()) throw std::invalid_argument("empty metadata text");
    for (size_t i = 0; i < value.size();) {
        uint32_t point = static_cast<unsigned char>(value[i++]);
        uint32_t minimum = 0;
        unsigned extra = 0;
        if (point >= 0x80) {
            if (point >= 0xc2 && point <= 0xdf) {
                point &= 0x1f;
                extra = 1;
                minimum = 0x80;
            } else if (point >= 0xe0 && point <= 0xef) {
                point &= 0x0f;
                extra = 2;
                minimum = 0x800;
            } else if (point >= 0xf0 && point <= 0xf4) {
                point &= 7;
                extra = 3;
                minimum = 0x10000;
            } else throw std::invalid_argument("invalid UTF-8 metadata");
            if (extra > value.size() - i) throw std::invalid_argument("truncated UTF-8 metadata");
            while (extra--) {
                const auto next = static_cast<unsigned char>(value[i++]);
                if ((next & 0xc0) != 0x80) throw std::invalid_argument("invalid UTF-8 metadata");
                point = (point << 6) | (next & 0x3f);
            }
        }
        if (!point || point < minimum || point > 0x10ffff || (point >= 0xd800 && point <= 0xdfff))
            throw std::invalid_argument("invalid metadata scalar");
    }
}
std::string copyText(const char* value) {
    if (!value) throw std::invalid_argument("missing text");
    size_t length = 0;
    while (value[length]) {
        if (length == kMaxUriBytes) throw std::length_error("text capacity");
        ++length;
    }
    textValid({value, length});
    return {value, length};
}

struct CreatedView {
    using Input = rpc::CreateSessionRequest;
    using Response = rpc::CreateSessionResponse;
    static constexpr uint32_t operation = OMNI_USD_LIVE_OPERATION_CREATE;
    static constexpr bool mutation = true;
    Response response;
    std::string authority, uri;
    OmniUsdLiveCreatedSession info{};
    void* user = nullptr;
    OmniUsdLiveCreateSessionCallback callback = nullptr;
    void prepare() {
        uri = makeSessionUri(authority, response.session_id());
        info = {sizeof(info), OMNI_USD_LIVE_ABI_VERSION, response.session_id().c_str(), uri.c_str()};
    }
    static uint64_t responseAllowance(uint32_t) noexcept { return kCreateSessionResponseBytes; }
    void deliver(OmniUsdLiveResult result) { callback(user, result, result == OMNI_USD_LIVE_OK ? &info : nullptr); }
    static auto reader(rpc::LiveService::Stub& stub, grpc::ClientContext* context, const Input& input,
        grpc::CompletionQueue* queue) {
        return stub.PrepareAsyncCreateSession(context, input, queue);
    }
};
struct ListView {
    using Input = rpc::ListSessionsRequest;
    using Response = rpc::ListSessionsResponse;
    static constexpr uint32_t operation = OMNI_USD_LIVE_OPERATION_LIST;
    static constexpr bool mutation = false;
    Response response;
    std::string authority;
    std::vector<std::string> uris;
    std::vector<OmniUsdLiveSessionInfo> infos;
    std::vector<const OmniUsdLiveSessionInfo*> pointers;
    void* user = nullptr;
    OmniUsdLiveListSessionsCallback callback = nullptr;
    static uint64_t responseAllowance(uint32_t maxUpdateBytes) noexcept { return finiteResponseLimit(maxUpdateBytes); }
    void prepare() {
        const auto count = static_cast<size_t>(response.sessions_size());
        uris.resize(count);
        infos.resize(count);
        pointers.resize(count);
        std::unordered_set<std::string_view> ids;
        for (size_t i = 0; i < count; ++i) {
            const auto& session = response.sessions(static_cast<int>(i));
            textValid(session.name(), true);
            textValid(session.created_by_user_name(), true);
            if (!ids.insert(session.session_id()).second) throw std::invalid_argument("duplicate session identity");
            uris[i] = makeSessionUri(authority, session.session_id());
            infos[i] = {sizeof(OmniUsdLiveSessionInfo), OMNI_USD_LIVE_ABI_VERSION, session.session_id().c_str(),
                uris[i].c_str(), session.name().c_str(), session.created_by_user_name().c_str(),
                session.created_at() / 1000};
            pointers[i] = &infos[i];
        }
    }
    void deliver(OmniUsdLiveResult result) {
        callback(user, result, result == OMNI_USD_LIVE_OK ? pointers.size() : 0,
            result == OMNI_USD_LIVE_OK && !pointers.empty() ? pointers.data() : nullptr);
    }
    static auto reader(rpc::LiveService::Stub& stub, grpc::ClientContext* context, const Input& input,
        grpc::CompletionQueue* queue) {
        return stub.PrepareAsyncListSessions(context, input, queue);
    }
};
struct StatView {
    using Input = rpc::StatLayerRequest;
    using Response = rpc::StatLayerResponse;
    static constexpr uint32_t operation = OMNI_USD_LIVE_OPERATION_STAT;
    static constexpr bool mutation = false;
    Response response;
    OmniUsdLiveLayerInfo info{};
    void* user = nullptr;
    OmniUsdLiveStatLayerCallback callback = nullptr;
    std::function<void(OmniUsdLiveResult, const OmniUsdLiveLayerInfo*, uint64_t)> ownedCallback;
    static uint64_t responseAllowance(uint32_t) noexcept { return kStatLayerResponseBytes; }
    void prepare() {
        // These are independent service wall-clock observations, not sequence
        // numbers. Clock corrections or backend skew may put modification
        // before creation; the protocol promises no ordering relationship.
        // Preserve the evidence without rejecting or clamping a valid stat.
        info = {sizeof(info), OMNI_USD_LIVE_ABI_VERSION, response.created_at() / 1000,
            response.last_modified_at() / 1000};
    }
    void deliver(OmniUsdLiveResult result) {
        const auto* value = result == OMNI_USD_LIVE_OK ? &info : nullptr;
        if (ownedCallback) ownedCallback(result, value, result == OMNI_USD_LIVE_OK ? response.last_modified_at() : 0);
        else callback(user, result, value);
    }
    static auto reader(rpc::LiveService::Stub& stub, grpc::ClientContext* context, const Input& input,
        grpc::CompletionQueue* queue) {
        return stub.PrepareAsyncStatLayer(context, input, queue);
    }
};
template<class RequestType, class ResponseType> struct DeleteView {
    using Input = RequestType;
    using Response = ResponseType;
    static constexpr uint32_t operation = OMNI_USD_LIVE_OPERATION_DELETE;
    static constexpr bool mutation = true;
    Response response;
    void* user = nullptr;
    OmniUsdLiveResultCallback callback = nullptr;
    static uint64_t responseAllowance(uint32_t) noexcept { return kFixedResponseBytes; }
    void prepare() {}
    void deliver(OmniUsdLiveResult result) { callback(user, result); }
    static auto reader(rpc::LiveService::Stub& stub, grpc::ClientContext* context, const Input& input,
        grpc::CompletionQueue* queue) {
        if constexpr (std::is_same_v<Input, rpc::DeleteSessionRequest>)
            return stub.PrepareAsyncDeleteSession(context, input, queue);
        else return stub.PrepareAsyncDeleteLayer(context, input, queue);
    }
};
using DeleteSessionView = DeleteView<rpc::DeleteSessionRequest, rpc::DeleteSessionResponse>;
using DeleteLayerView = DeleteView<rpc::DeleteLayerRequest, rpc::DeleteLayerResponse>;

template<class View> class Finite final : public ScheduledOperation, public std::enable_shared_from_this<Finite<View>> {
    struct Delivery {
        // The response allowance. Declared first so it outlives the response
        // buffers it accounts for, until the callback and its captures retire.
        std::optional<PendingBytes> reservation;
        View view;
        OmniUsdLiveResult result = OMNI_USD_LIVE_INTERNAL_ERROR;
    };
    struct Call {
        grpc::ClientContext context;
        grpc::Status status;
        typename View::Response response;
        std::unique_ptr<grpc::ClientAsyncResponseReader<typename View::Response>> reader;
        uint64_t generation = 0;
        uint64_t authFailure = 0;
        uint64_t attempt = 0;
        Clock::time_point started{}, completed{};
        bool issued = false, observed = false; // Scheduler-owned, never callback state.
        std::atomic<bool> done{false};
        bool ok = false;
    };
    struct Tag final : CompletionTag {
        std::shared_ptr<Finite> owner;
        std::shared_ptr<Call> call;
        Tag(std::shared_ptr<Finite> value, std::shared_ptr<Call> attempt)
            : owner(std::move(value)), call(std::move(attempt)) {}
        void onComplete(bool ok, Clock::time_point received) noexcept override {
            call->completed = received;
            call->ok = ok;
            if (ok) owner->runtime_.auth->completed(*owner->endpoint_, call->generation, call->status.error_code());
            call->done.store(true, std::memory_order_release);
            owner->runtime_.auth->wake(*owner);
        }
    };
public:
    Finite(Runtime& runtime, EndpointLease endpoint, typename View::Input input, View view, bool retire = false,
        Clock::time_point deadline = Clock::time_point::max())
        : runtime_(runtime), endpoint_(std::move(endpoint)), input_(std::move(input)),
          delivery_(std::make_shared<Delivery>()),
          deadline_(
              std::min(deadline, Clock::now() + std::chrono::milliseconds(runtime.options.values.request_timeout_ms))),
          retire_(retire), authObservation_(*runtime.diagnostics, AuthWaitKind::Finite) {
        delivery_->view = std::move(view);
        callback_ = [delivery = delivery_] { delivery->view.deliver(delivery->result); };
        std::lock_guard<std::mutex> lock(endpoint_->mutex);
        waitingAuth_ = !runtime_.auth->usable(*endpoint_, wallTimeMs());
        authFailure_ = endpoint_->authFailureSerial;
        authDeadline_ = Clock::now() + std::chrono::milliseconds(endpoint_->options.authWaitMs);
    }
    std::shared_ptr<Request> request;
    // Whether the last auth evaluation found no usable credential. A caller
    // whose own wait on this request expires classifies it with expiredWait.
    bool credentialStarved() {
        std::lock_guard<std::mutex> lock(mutex_);
        return waitingAuth_;
    }
    void cancel() {
        // The caller's cancellation deadline includes this hook. Never wait
        // for response preparation; the scheduler owns transport cancellation.
        cancelled_.store(true, std::memory_order_release);
        runtime_.auth->wake(*this);
    }
    ScheduledStep advance(Clock::time_point now) noexcept override {
        std::lock_guard<std::mutex> lock(mutex_);
        try {
            return step(now);
        } catch (const std::bad_alloc&) {
            return finishLocal(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
        } catch (const std::length_error&) {
            return finishLocal(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
        } catch (OmniUsdLiveResult result) { return finishLocal(result); } catch (...) {
            return finishLocal(OMNI_USD_LIVE_INTERNAL_ERROR);
        }
    }
private:
    Runtime& runtime_;
    EndpointLease endpoint_;
    typename View::Input input_;
    std::shared_ptr<Delivery> delivery_;
    std::function<void()> callback_;
    std::mutex mutex_;
    std::shared_ptr<Call> call_;
    Clock::time_point deadline_, nextAttempt_{}, authDeadline_{};
    OmniUsdLiveRequestError last_{};
    uint64_t authFailure_ = 0, attempts_ = 0;
    bool waitingAuth_ = false;
    const bool retire_;
    bool finished_ = false;
    std::atomic<bool> cancelled_{false};
    OperationAuthWait authObservation_;
    void observeAttempt(const OmniUsdLiveRequestError& error) noexcept {
        if (!call_ || !call_->issued || call_->observed || !call_->done.load(std::memory_order_acquire)) return;
        call_->observed = true;
        runtime_.diagnostics->observeRpc(endpoint_->handle, View::operation, error.result,
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::milliseconds>(call_->completed - call_->started).count()),
            request->handle, call_->attempt, error.grpc_status_present, error.grpc_status, 0, 0, call_->generation);
    }
    ScheduledStep finish(OmniUsdLiveRequestError error) {
        // No issued attempt is outstanding: its Tag::onComplete, the only
        // endpoint use off the scheduler, published done before this step.
        assert(!call_ || !call_->issued || call_->done.load(std::memory_order_acquire));
        authObservation_.finish(error.result, Clock::now());
        observeAttempt(error);
        error.attempt_id = attempts_;
        delivery_->result = error.result;
        delivery_.reset(); // Transfer all callback state to the executor before publishing.
        call_.reset();
        // The endpoint's last use is above, and a finished step never reads it.
        // The request's own lease covers the dependency until handle release.
        // Callers never hold the endpoint mutex here: releasing takes it.
        { EndpointLease retired = std::move(endpoint_); }
        request->finish(error, std::move(callback_));
        finished_ = true;
        return {!retire_, retire_ ? Clock::now() : Clock::time_point{}};
    }
    ScheduledStep finishLocal(OmniUsdLiveResult result) {
        // A reservation for a request that never issued backed a failed admission.
        if (!attempts_ && delivery_ && delivery_->reservation) delivery_->reservation->releaseQuietly();
        OmniUsdLiveRequestError error{};
        error.result = result;
        return finish(error);
    }
    ScheduledStep step(Clock::time_point now) {
        if (finished_) {
            // Internal resolver requests have no caller-owned public handle.
            // Keep their registry slot until callback captures are quiescent,
            // including suppressed delivery after a synchronous timeout.
            OmniUsdLiveResult outcome = OMNI_USD_LIVE_INTERNAL_ERROR;
            if (request->result(&outcome) == OMNI_USD_LIVE_BUSY) return {false, now + std::chrono::milliseconds(1)};
            runtime_.requests->release(request->handle);
            return {true, {}};
        }
        if (call_) {
            if (!call_->done.load(std::memory_order_acquire)) {
                if (cancelled_) call_->context.TryCancel();
                // gRPC owns the already-installed transport deadline. Calling
                // TryCancel for expiry would race it and report CANCELLED for
                // a timed-out read. Await the authoritative completion, also
                // allowing an already successful mutation to remain successful.
                return {false, now >= deadline_ ? Clock::time_point::max() : deadline_};
            }
            last_ = {};
            last_.grpc_status_present = call_->ok;
            last_.grpc_status = call_->ok ? call_->status.error_code() : 0;
            last_.result = call_->ok ?
                classifyUnary(call_->status.error_code(), View::mutation) :
                OmniUsdLiveResult(View::mutation ? OMNI_USD_LIVE_OUTCOME_UNKNOWN : OMNI_USD_LIVE_INTERNAL_ERROR);
            if (last_.result == OMNI_USD_LIVE_OK) {
                delivery_->view.response.Swap(&call_->response);
                try {
                    delivery_->view.prepare();
                } catch (const std::bad_alloc&) {
                    last_.result = View::mutation ? OMNI_USD_LIVE_OUTCOME_UNKNOWN : OMNI_USD_LIVE_RESOURCE_EXHAUSTED;
                } catch (const std::length_error&) {
                    last_.result = View::mutation ? OMNI_USD_LIVE_OUTCOME_UNKNOWN : OMNI_USD_LIVE_RESOURCE_EXHAUSTED;
                } catch (...) {
                    last_.result = View::mutation ? OMNI_USD_LIVE_OUTCOME_UNKNOWN : OMNI_USD_LIVE_MALFORMED_STREAM;
                }
                return finish(last_);
            }
            observeAttempt(last_); // Record this failed transport before auth/backoff replaces its owner.
            if (View::mutation || cancelled_ ||
                (last_.result != OMNI_USD_LIVE_UNAVAILABLE && last_.result != OMNI_USD_LIVE_UNAUTHENTICATED))
                return finish(last_);
            if (now >= deadline_) return finishLocal(OMNI_USD_LIVE_TIMEOUT);
            if (attempts_ >= runtime_.options.values.retry_max_attempts) return finish(last_);
            const bool auth = last_.result == OMNI_USD_LIVE_UNAUTHENTICATED;
            const auto failedBeforeAttempt = call_->authFailure;
            const auto failedGeneration = call_->generation;
            call_.reset();
            uint64_t delay = 0;
            if (auth) {
                waitingAuth_ = true;
                std::lock_guard<std::mutex> lock(endpoint_->mutex);
                authFailure_ = failedBeforeAttempt;
                authDeadline_ = now + std::chrono::milliseconds(endpoint_->options.authWaitMs);
                nextAttempt_ = now;
            } else {
                delay = retryDelay(runtime_.instanceNonce, request->handle, attempts_,
                    runtime_.options.values.retry_initial_ms, runtime_.options.values.retry_maximum_ms);
                nextAttempt_ = now + std::chrono::milliseconds(delay);
            }
            // This is a retry decision, not an issued RPC. Auth wait duration
            // is separate from the zero backoff used for credential refresh.
            runtime_.diagnostics->emit(OMNI_USD_LIVE_RECORD_RETRY_SCHEDULED, endpoint_->handle, 0, 0,
                {{DiagnosticKey::Operation, View::operation}, {DiagnosticKey::Request, request->handle},
                    {DiagnosticKey::Attempt, attempts_ + 1}, {DiagnosticKey::AuthGeneration, failedGeneration},
                    {DiagnosticKey::Reason, last_.result}, {DiagnosticKey::DelayMs, delay}});
        }
        if (cancelled_) return finishLocal(OMNI_USD_LIVE_CANCELLED);
        if (now >= deadline_) return finishLocal(expiredWait(now, deadline_, waitingAuth_));
        if (now < nextAttempt_) return {false, std::min(deadline_, nextAttempt_)};
        auto candidate = std::make_shared<Call>();
        bool unauthenticated = false; // finish releases the lease, so it runs after the endpoint lock.
        {
            std::lock_guard<std::mutex> lock(endpoint_->mutex);
            runtime_.auth->evaluate(*endpoint_, wallTimeMs(), now);
            if (endpoint_->authFailed || (waitingAuth_ && endpoint_->authFailureSerial != authFailure_)) {
                unauthenticated = true;
            } else if (!runtime_.auth->usable(*endpoint_, wallTimeMs())) {
                if (!waitingAuth_) {
                    waitingAuth_ = true;
                    authFailure_ = endpoint_->authFailureSerial;
                    authDeadline_ = now + std::chrono::milliseconds(endpoint_->options.authWaitMs);
                }
                if (now < authDeadline_) {
                    authObservation_.start(Clock::now());
                    return {false, std::min({deadline_, authDeadline_, endpoint_->authDeadline})};
                }
                unauthenticated = true;
            } else {
                authObservation_.finish(OMNI_USD_LIVE_OK, Clock::now());
                waitingAuth_ = false;
                candidate->generation = endpoint_->tokenGeneration;
                candidate->authFailure = endpoint_->authFailureSerial;
                candidate->context.AddMetadata("authorization", "Bearer " + endpoint_->token);
            }
        }
        if (unauthenticated) return finishLocal(OMNI_USD_LIVE_UNAUTHENTICATED);
        candidate->context.set_deadline(
            rpcDeadlineAfter(std::chrono::duration_cast<std::chrono::milliseconds>(deadline_ - Clock::now())));
        candidate->context.set_wait_for_ready(false);
        // Every local refusal precedes View::reader: gRPC serializes the input
        // into the call when it prepares the reader, and a prepared reader
        // discarded without StartCall never releases that buffer.
        if (Clock::now() >= deadline_) return finishLocal(OMNI_USD_LIVE_TIMEOUT);
        // Reserve once, before the first issuance; retries reuse the allowance.
        // A refusal propagates to advance, which finishes locally without an RPC.
        if (!delivery_->reservation)
            delivery_->reservation.emplace(runtime_.acquireEndpoint(endpoint_->handle),
                View::responseAllowance(runtime_.options.values.max_update_bytes),
                runtime_.options.values.max_pending_bytes_per_endpoint);
        auto tag = std::make_unique<Tag>(this->shared_from_this(), candidate);
        candidate->reader = View::reader(*endpoint_->stub, &candidate->context, input_, &runtime_.queue());
        if (!candidate->reader) return finishLocal(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
        call_ = std::move(candidate);
        // Preparation stays outside issueIfNotCancelled because that callback
        // runs under the request mutex, which cancellation and completion wait
        // on; serializing there would hold it across allocation. A cancellation
        // that lands between preparation and issuance therefore discards a
        // prepared reader, and gRPC keeps its serialized request buffer.
        const bool issued = request->issueIfNotCancelled([&] {
            ++attempts_;
            call_->attempt = attempts_;
            call_->issued = true;
            call_->started = Clock::now();
            runtime_.diagnostics->startedRpc(View::operation);
            call_->reader->StartCall();
            // Read before submitTag: its release is the last access before completion.
            auto* response = &call_->response;
            auto* status = &call_->status;
            call_->reader->Finish(response, status, submitTag(tag, *runtime_.diagnostics));
        });
        if (!issued) return finishLocal(OMNI_USD_LIVE_CANCELLED);
        return {false, deadline_};
    }
};

template<class View>
std::shared_ptr<Finite<View>> start(Runtime& runtime, EndpointLease lease, typename View::Input input, View view,
    bool retire = false, Clock::time_point deadline = Clock::time_point::max()) {
    const auto endpoint = lease->handle;
    auto operation = std::make_shared<Finite<View>>(runtime, std::move(lease), std::move(input), std::move(view),
        retire, deadline);
    runtime.auth->reserve(operation);
    try {
        std::weak_ptr<Finite<View>> weak = operation;
        operation->request = runtime.requests->create(endpoint, View::operation, [weak] {
            if (auto pending = weak.lock()) pending->cancel();
        });
    } catch (...) {
        runtime.auth->discard(*operation);
        throw;
    }
    runtime.auth->activate(*operation);
    return operation;
}
template<class View>
OmniUsdLiveRequest submit(Runtime& runtime, EndpointLease lease, typename View::Input input, View view) {
    return start(runtime, std::move(lease), std::move(input), std::move(view))->request->handle;
}
template<class Action>
OmniUsdLiveResult boundary(OmniUsdLiveRequest* output, bool validCallback, Action action) noexcept {
    if (output) *output = 0;
    if (!output || !validCallback) return OMNI_USD_LIVE_INVALID_ARGUMENT;
    try {
        auto runtime = getRuntime();
        if (!runtime) return OMNI_USD_LIVE_NOT_INITIALIZED;
        *output = action(*runtime);
        return OMNI_USD_LIVE_OK;
    } catch (...) { return currentExceptionResult(); }
}
EndpointLease acquire(Runtime& runtime, OmniUsdLiveEndpoint endpoint) {
    auto lease = runtime.acquireEndpoint(endpoint);
    if (!lease) throw OmniUsdLiveResult(runtime.closing ? OMNI_USD_LIVE_BUSY : OMNI_USD_LIVE_INVALID_HANDLE);
    return lease;
}
EndpointLease resolve(Runtime& runtime, const LiveUri& uri) {
    OmniUsdLiveEndpoint endpoint = 0;
    {
        std::lock_guard<std::mutex> lock(runtime.mutex);
        const auto found = runtime.authorities.find(uri.authority.canonical);
        if (found == runtime.authorities.end()) throw OmniUsdLiveResult(OMNI_USD_LIVE_NOT_FOUND);
        endpoint = found->second;
    }
    return acquire(runtime, endpoint);
}
// A metadata deadline found expired before the request starts: UNAUTHENTICATED
// when the endpoint has no usable credential (design section 14).
OmniUsdLiveResult expiredBeforeStart(Runtime& runtime, const Endpoint& endpoint, Clock::time_point deadline) noexcept {
    std::lock_guard<std::mutex> lock(endpoint.mutex);
    return expiredWait(deadline, deadline, !runtime.auth->usable(endpoint, wallTimeMs()));
}
} // namespace

MetadataDeadlineScope::MetadataDeadlineScope(TimePoint deadline) noexcept : previous_(metadataDeadline) {
    metadataDeadline = std::min(metadataDeadline, deadline);
}
MetadataDeadlineScope::~MetadataDeadlineScope() { metadataDeadline = previous_; }
MetadataDeadlineScope::TimePoint MetadataDeadlineScope::limit(TimePoint deadline) noexcept {
    return std::min(deadline, metadataDeadline);
}
OmniUsdLiveResult statLayerMetadata(const std::string& uri, OmniUsdLiveLayerInfo& output,
    uint64_t* modifiedTimeUs) noexcept {
    try {
        auto runtime = getRuntime();
        if (!runtime) return OMNI_USD_LIVE_NOT_INITIALIZED;
        if (DispatchExecutor::isWorker()) return OMNI_USD_LIVE_BUSY;
        const auto deadline = MetadataDeadlineScope::limit(
            Clock::now() + std::chrono::milliseconds(runtime->options.values.initial_open_timeout_ms));
        const auto parsed = parseUri(uri);
        if (parsed.layerName.empty()) return OMNI_USD_LIVE_INVALID_ARGUMENT;
        auto lease = resolve(*runtime, parsed);
        if (Clock::now() >= deadline) return expiredBeforeStart(*runtime, *lease, deadline);
        rpc::StatLayerRequest input;
        input.set_session_id(parsed.sessionId);
        input.set_layer_name(parsed.layerName);
        struct Result {
            OmniUsdLiveLayerInfo info{};
            OmniUsdLiveResult result = OMNI_USD_LIVE_INTERNAL_ERROR;
            uint64_t modifiedTimeUs = 0;
        };
        auto delivery = std::make_shared<Result>();
        StatView view;
        view.ownedCallback = [delivery](OmniUsdLiveResult result, const OmniUsdLiveLayerInfo* info,
                                 uint64_t modifiedUs) {
            delivery->result = result;
            if (info) delivery->info = *info;
            delivery->modifiedTimeUs = modifiedUs;
        };
        if (Clock::now() >= deadline) return expiredBeforeStart(*runtime, *lease, deadline);
        // Held through classification: the operation keeps its starvation
        // flag after it finishes and the scheduler drops it.
        const auto operation = start(*runtime, std::move(lease), std::move(input), std::move(view), true, deadline);
        const auto request = operation->request;
        // Round down: preparation cannot extend the caller's absolute budget.
        const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
        auto waited = request->wait(remaining > 0 ? static_cast<uint32_t>(remaining) : 0);
        // The rounded-down wait can end just before the operation's deadline;
        // a completion in that gap delivers its own result.
        if (waited == OMNI_USD_LIVE_TIMEOUT) waited = request->wait(0);
        if (waited != OMNI_USD_LIVE_OK) {
            // This wait ends at the operation deadline, so it classifies expiry
            // as the operation's own deadline check does.
            if (waited == OMNI_USD_LIVE_TIMEOUT)
                waited = expiredWait(deadline, deadline, operation->credentialStarved());
            request->cancel(0);
            return waited;
        }
        // wait synchronizes with delivery and destruction of its owned capture.
        if (delivery->result == OMNI_USD_LIVE_OK) {
            output = delivery->info;
            if (modifiedTimeUs) *modifiedTimeUs = delivery->modifiedTimeUs;
        }
        return delivery->result;
    } catch (...) { return currentExceptionResult(); }
}
} // namespace omni::usd::live

using namespace omni::usd::live;
extern "C" {
OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveCreateSession(OmniUsdLiveEndpoint endpoint, const char* key,
    const char* name, void* user, OmniUsdLiveCreateSessionCallback callback, OmniUsdLiveRequest* output) {
    return boundary(output, callback != nullptr, [&](Runtime& runtime) {
        rpc::CreateSessionRequest input;
        input.set_path(copyText(key));
        input.set_name(copyText(name));
        auto lease = acquire(runtime, endpoint);
        CreatedView view;
        view.authority = lease->options.authority.canonical;
        view.user = user;
        view.callback = callback;
        return submit(runtime, std::move(lease), std::move(input), std::move(view));
    });
}
OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveListSessions(OmniUsdLiveEndpoint endpoint, const char* key, void* user,
    OmniUsdLiveListSessionsCallback callback, OmniUsdLiveRequest* output) {
    return boundary(output, callback != nullptr, [&](Runtime& runtime) {
        rpc::ListSessionsRequest input;
        input.set_path(copyText(key));
        auto lease = acquire(runtime, endpoint);
        ListView view;
        view.authority = lease->options.authority.canonical;
        view.user = user;
        view.callback = callback;
        return submit(runtime, std::move(lease), std::move(input), std::move(view));
    });
}
OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveDeleteSession(const char* uri, void* user,
    OmniUsdLiveResultCallback callback, OmniUsdLiveRequest* output) {
    return boundary(output, callback != nullptr, [&](Runtime& runtime) {
        auto parsed = parseUri(copyText(uri));
        if (!parsed.layerName.empty()) throw std::invalid_argument("expected session URI");
        rpc::DeleteSessionRequest input;
        input.set_session_id(parsed.sessionId);
        DeleteSessionView view;
        view.user = user;
        view.callback = callback;
        return submit(runtime, resolve(runtime, parsed), std::move(input), std::move(view));
    });
}
OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveStatLayer(const char* uri, void* user,
    OmniUsdLiveStatLayerCallback callback, OmniUsdLiveRequest* output) {
    return boundary(output, callback != nullptr, [&](Runtime& runtime) {
        auto parsed = parseUri(copyText(uri));
        if (parsed.layerName.empty()) throw std::invalid_argument("expected layer URI");
        rpc::StatLayerRequest input;
        input.set_session_id(parsed.sessionId);
        input.set_layer_name(parsed.layerName);
        StatView view;
        view.user = user;
        view.callback = callback;
        return submit(runtime, resolve(runtime, parsed), std::move(input), std::move(view));
    });
}
OmniUsdLiveResult OMNI_USD_LIVE_CALL omniUsdLiveDeleteLayer(const char* uri, void* user,
    OmniUsdLiveResultCallback callback, OmniUsdLiveRequest* output) {
    return boundary(output, callback != nullptr, [&](Runtime& runtime) {
        auto parsed = parseUri(copyText(uri));
        if (parsed.layerName.empty()) throw std::invalid_argument("expected layer URI");
        rpc::DeleteLayerRequest input;
        input.set_session_id(parsed.sessionId);
        input.set_layer_name(parsed.layerName);
        DeleteLayerView view;
        view.user = user;
        view.callback = callback;
        return submit(runtime, resolve(runtime, parsed), std::move(input), std::move(view));
    });
}
} // extern "C"
