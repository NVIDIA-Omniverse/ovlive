// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "LiveDeadline.h"
#include "LiveLayerRead.h"
#include "LiveAuth.h"
#include "LiveDiagnostics.h"
#include <cassert>
#include <stdexcept>

namespace omni::usd::live {
void LayerReadFlow::hold() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    held_ = true;
}
std::function<void()> LayerReadFlow::release() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    held_ = false;
    auto wake = std::move(parked_);
    parked_ = nullptr;
    return wake;
}
bool LayerReadFlow::park(std::function<void()> wake) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!held_) return false;
    parked_ = std::move(wake);
    return true;
}
namespace {
struct Payload {
    PendingBytes slot; // Last destroyed: covers retained protobuf receive storage.
    // Retain completion through callback-capture and receive-buffer cleanup.
    std::shared_ptr<DrainTicket> drain;
    rpc::SubscribeReadLayerResponse frame;
    LayerReadCallbacks callbacks;
    OmniUsdLiveRequestError error{};
    explicit Payload(PendingBytes reservation) : slot(std::move(reservation)) {}
};
class ReadCall final : public DispatchJob {
public:
    using Clock = std::chrono::steady_clock;
    enum class Phase { Start, Read, Finish, Resume };
    struct Tag final : CompletionTag {
        std::shared_ptr<ReadCall> owner;
        Phase phase;
        std::shared_ptr<DrainTicket> drain;
        explicit Tag(Phase value) : phase(value) {}
        void onComplete(bool ok, Clock::time_point received) noexcept override {
            const auto receivedAt = received;
            const auto admitted = drain ? drain->admit() : OmniUsdLiveResult(OMNI_USD_LIVE_OK);
            owner->ready(phase, ok, std::move(drain), admitted, receivedAt);
        }
    };
    Runtime& runtime;
    LayerReadOptions options;
    grpc::ClientContext context;
    grpc::Status status;
    std::unique_ptr<grpc::ClientAsyncReaderInterface<rpc::SubscribeReadLayerResponse>> reader;
    std::shared_ptr<Request> request;
    std::shared_ptr<Payload> payload;
    std::function<void()> delivery;
    std::shared_ptr<ReadCall> keepAlive;
    std::unique_ptr<Tag> start = std::make_unique<Tag>(Phase::Start);
    std::unique_ptr<Tag> finish = std::make_unique<Tag>(Phase::Finish);
    uint64_t generation = 0;
    Clock::time_point started{}, completed{};
    ReadCall(Runtime& value, LayerReadOptions settings)
        : runtime(value), options(settings), object_(settings.objectId), sequence_(settings.sequence) {}
    Tag* handoff(std::unique_ptr<Tag>& tag) {
        tag->owner = std::static_pointer_cast<ReadCall>(shared_from_this());
        return submitTag(tag, *runtime.diagnostics);
    }
    void retire(OmniUsdLiveRequestError error) noexcept {
        error.struct_size = sizeof(error);
        error.abi_version = OMNI_USD_LIVE_ABI_VERSION;
        error.operation = OMNI_USD_LIVE_OPERATION_READ;
        error.request_id = request->handle;
        error.attempt_id = issued_ ? 1 : 0;
        if (issued_)
            runtime.diagnostics->observeRpc(payload->slot.endpoint().handle, error.operation, error.result,
                static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(completed - started).count()),
                error.request_id, 1, error.grpc_status_present, error.grpc_status, options.subjectId,
                options.streamGeneration, generation);
        pending_.reset();
        payload->error = error;
        payload.reset(); // Sole callback/buffer owner transferred before enqueue.
        request->finish(error, std::move(delivery));
        keepAlive.reset(); // Dispatcher/current tag still retains this object.
    }
    bool issued_ = false;
private:
    std::mutex mutex_;
    bool scheduled_ = false, eventReady_ = false;
    Phase event_ = Phase::Start;
    bool eventOk_ = false;
    std::shared_ptr<DrainTicket> eventDrain_;
    OmniUsdLiveResult eventAdmission_ = OMNI_USD_LIVE_OK;
    uint64_t object_, sequence_;
    bool firstHeader_ = true;
    bool parked_ = false; // Guarded by mutex_: no Read or Finish is outstanding.
    std::shared_ptr<LayerReadUpdate> pending_;
    size_t offset_ = 0;
    OmniUsdLiveResult localFailure_ = OMNI_USD_LIVE_OK;

    void ready(Phase phase, bool ok, std::shared_ptr<DrainTicket> drain, OmniUsdLiveResult admitted,
        Clock::time_point received) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        assert(!eventReady_);
        event_ = phase;
        eventOk_ = ok;
        eventReady_ = true;
        if (phase == Phase::Finish) completed = received;
        eventDrain_ = std::move(drain);
        eventAdmission_ = admitted;
        if (!scheduled_) {
            scheduled_ = true;
            runtime.requests->executor().enqueue(*this);
        }
    }
    void dispatch() noexcept override {
        Phase phase;
        bool ok;
        std::shared_ptr<DrainTicket> drain;
        OmniUsdLiveResult admitted;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            assert(scheduled_ && eventReady_);
            phase = event_;
            ok = eventOk_;
            eventReady_ = false;
            drain = std::move(eventDrain_);
            admitted = eventAdmission_;
        }
        try {
            // Rejection reflects a state already closed/failed (or an
            // admission overflow already latched by the tracker). Retiring a
            // future CQ event must not turn a clean close into cancellation.
            if (phase != Phase::Finish && admitted != OMNI_USD_LIVE_OK) fail(admitted, false);
            else step(phase, ok, drain);
        } catch (OmniUsdLiveResult result) { fail(result); } catch (const std::bad_alloc&) {
            fail(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
        } catch (const std::length_error&) { fail(OMNI_USD_LIVE_RESOURCE_EXHAUSTED); } catch (...) {
            fail(OMNI_USD_LIVE_INTERNAL_ERROR);
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            // A completion can arrive before this dispatch finishes. Publish
            // the next job only after all processing/state access above ends.
            if (eventReady_) runtime.requests->executor().enqueue(*this);
            else scheduled_ = false;
        }
    }
    void finishTransport() { reader->Finish(&status, handoff(finish)); }
    // Parks the call while its owner holds the flow. The request check and the
    // flow registration run under mutex_, and unpark takes only mutex_, so a
    // cancellation either prevents parking or wakes the parked call.
    bool park() {
        if (!options.flow) return false;
        std::weak_ptr<ReadCall> weak = std::static_pointer_cast<ReadCall>(shared_from_this());
        std::lock_guard<std::mutex> lock(mutex_);
        if (!request->issueIfNotCancelled([] {})) return false;
        if (!options.flow->park([weak] {
            if (auto call = weak.lock()) call->unpark();
        }))
            return false;
        parked_ = true;
        return true;
    }
public:
    void unpark() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!parked_) return;
        parked_ = false;
        assert(!eventReady_); // A parked call has no Read or Finish outstanding.
        event_ = Phase::Resume;
        eventOk_ = true;
        eventReady_ = true;
        eventDrain_.reset();
        eventAdmission_ = OMNI_USD_LIVE_OK;
        if (!scheduled_) {
            scheduled_ = true;
            runtime.requests->executor().enqueue(*this);
        }
    }
private:
    void fail(OmniUsdLiveResult result, bool latch = true) noexcept {
        localFailure_ = result;
        if (latch && payload && payload->callbacks.failed && payload->callbacks.failed(result)) latch = false;
        if (latch && options.drain) options.drain->fail(result);
        pending_.reset();
        context.TryCancel();
        finishTransport();
    }
    void readNext() {
        auto tag = std::make_unique<Tag>(Phase::Read);
        if (options.drain) tag->drain = options.drain->prepare();
        payload->frame.Clear();
        // Loaded before handoff: its release is the last access before completion.
        auto* frame = &payload->frame;
        if (!request->issueIfNotCancelled([&] { reader->Read(frame, handoff(tag)); })) finishTransport();
    }
    void deliver(const std::shared_ptr<DrainTicket>& drain) {
        auto complete = std::move(pending_);
        complete->drain = drain;
        sequence_ = complete->header.sequence_num();
        runtime.auth->completed(payload->slot.endpoint(), generation, grpc::StatusCode::OK);
        // This admission point linearizes callback start against cancellation.
        // Host/consumer code is invoked only after releasing the request mutex.
        if (request->issueIfNotCancelled([] {})) payload->callbacks.update(std::move(complete));
    }
    void acceptFrame(const std::shared_ptr<DrainTicket>& drain) {
        auto& frame = payload->frame;
        if (frame.has_header()) {
            const auto& header = frame.header();
            if (pending_ || !header.object_id() || !header.sequence_num())
                throw OmniUsdLiveResult(OMNI_USD_LIVE_MALFORMED_STREAM);
            if (object_ && header.object_id() != object_) {
                if (!firstHeader_ || !options.permitInitialReplacement)
                    throw OmniUsdLiveResult(OMNI_USD_LIVE_OBJECT_REPLACED);
                sequence_ = 0;
            }
            // A header below the client cursor, including a caught-up header, is a server
            // sequence regression: the layer fails; intents stay retained.
            if (header.sequence_num() < sequence_ || (header.total_size() && header.sequence_num() == sequence_))
                throw OmniUsdLiveResult(OMNI_USD_LIVE_MALFORMED_STREAM);
            // A read update is bounded by the endpoint's pending bytes, not the write limit: a catch-up snapshot
            // of a large layer exceeds any single upload.
            if (header.total_size() > runtime.options.values.max_pending_bytes_per_endpoint)
                throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
            const bool firstResponse = firstHeader_;
            object_ = header.object_id();
            firstHeader_ = false;
            auto lease = runtime.acquireEndpoint(payload->slot.endpoint().handle);
            PendingBytes budget(std::move(lease), header.total_size() + header.ByteSizeLong() + 64,
                runtime.options.values.max_pending_bytes_per_endpoint);
            pending_ = std::make_shared<LayerReadUpdate>(std::move(budget), header);
            pending_->firstResponse = firstResponse;
            pending_->bytes.resize(static_cast<size_t>(header.total_size()));
            offset_ = 0;
            if (!header.total_size()) deliver(drain); // Empty caught-up is a distinct signal, never a delta.
        } else if (frame.has_chunk()) {
            if (!pending_) throw OmniUsdLiveResult(OMNI_USD_LIVE_MALFORMED_STREAM);
            const auto& bytes = frame.chunk().data();
            if (bytes.size() > pending_->bytes.size() - offset_)
                throw OmniUsdLiveResult(OMNI_USD_LIVE_MALFORMED_STREAM);
            pending_->bytes.replace(offset_, bytes.size(), bytes);
            offset_ += bytes.size();
            if (offset_ == pending_->bytes.size()) deliver(drain);
        } else throw OmniUsdLiveResult(OMNI_USD_LIVE_MALFORMED_STREAM);
    }
    void step(Phase phase, bool ok, const std::shared_ptr<DrainTicket>& drain) {
        if (phase == Phase::Finish) {
            payload->drain = drain;
            OmniUsdLiveRequestError error{};
            error.grpc_status_present = ok;
            error.grpc_status = ok ? status.error_code() : 0;
            error.result = localFailure_ != OMNI_USD_LIVE_OK ? localFailure_ :
                !ok                                          ? OmniUsdLiveResult(OMNI_USD_LIVE_INTERNAL_ERROR) :
                status.ok() ? OmniUsdLiveResult(pending_ ? OMNI_USD_LIVE_MALFORMED_STREAM : OMNI_USD_LIVE_UNAVAILABLE) :
                              classifyUnary(status.error_code(), false);
            if (ok) runtime.auth->completed(payload->slot.endpoint(), generation, status.error_code());
            retire(error);
        } else if (!ok) {
            // The received EOF/start failure is finite queued work. Carry its
            // ticket through Finish and classification, not a future read.
            if (drain) finish->drain = drain;
            finishTransport(); // A non-OK interruption is not successful truncation.
        } else {
            if (phase == Phase::Read) acceptFrame(drain);
            if (park()) return;
            readNext();
        }
    }
};
} // namespace

std::shared_ptr<Request> subscribeLayerRead(Runtime& runtime, OmniUsdLiveEndpoint endpoint, std::string_view sessionId,
    std::string_view layerName, LayerReadOptions options, LayerReadCallbacks callbacks, LayerReadTransport* transport) {
    if (!callbacks.update || (!options.objectId && options.sequence))
        throw std::invalid_argument("invalid layer read options");
    auto lease = runtime.acquireEndpoint(endpoint);
    if (!lease) throw OmniUsdLiveResult(runtime.closing ? OMNI_USD_LIVE_BUSY : OMNI_USD_LIVE_INVALID_HANDLE);
    makeLayerUri(makeSessionUri(lease->options.authority.canonical, sessionId), layerName);
    rpc::SubscribeReadLayerRequest input;
    input.set_session_id(std::string(sessionId));
    input.set_layer_name(std::string(layerName));
    input.set_object_id(options.objectId);
    input.set_sequence_num(options.sequence);
    input.set_client_id(lease->writerId);
    auto call = std::make_shared<ReadCall>(runtime, options);
    if (options.drain) {
        call->start->drain = options.drain->prepare();
        call->finish->drain = options.drain->prepare();
    }
    {
        std::lock_guard<std::mutex> lock(lease->mutex);
        runtime.auth->evaluate(*lease, wallTimeMs(), std::chrono::steady_clock::now());
        if (!runtime.auth->usable(*lease, wallTimeMs())) throw OmniUsdLiveResult(OMNI_USD_LIVE_UNAUTHENTICATED);
        call->context.AddMetadata("authorization", "Bearer " + lease->token);
        call->generation = lease->tokenGeneration;
    }
    auto payload = std::make_shared<Payload>(
        PendingBytes(std::move(lease), layerReadFrameLimit(runtime.options.values.max_update_bytes),
            runtime.options.values.max_pending_bytes_per_endpoint));
    payload->callbacks = std::move(callbacks);
    call->payload = payload;
    call->delivery = [payload] {
        if (payload->callbacks.closed) payload->callbacks.closed(payload->error, payload->drain);
    };
    if (options.timeoutMs) call->context.set_deadline(rpcDeadlineAfter(std::chrono::milliseconds(options.timeoutMs)));
    call->context.set_wait_for_ready(false);
    call->reader = transport ?
        transport->prepare(&call->context, input, &runtime.queue()) :
        payload->slot.endpoint().stub->PrepareAsyncSubscribeReadLayer(&call->context, input, &runtime.queue());
    if (!call->reader) throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    std::weak_ptr<ReadCall> weak = call;
    call->request = runtime.requests->create(endpoint, OMNI_USD_LIVE_OPERATION_READ, [weak] {
        if (auto active = weak.lock()) {
            active->context.TryCancel();
            active->unpark();
        }
    });
    payload.reset();
    call->keepAlive = call;
    if (!call->request->issueIfNotCancelled([&] {
        call->issued_ = true;
        call->started = ReadCall::Clock::now();
        runtime.diagnostics->startedRpc(OMNI_USD_LIVE_OPERATION_READ);
        call->reader->StartCall(call->handoff(call->start));
    })) {
        OmniUsdLiveRequestError error{};
        error.result = OMNI_USD_LIVE_CANCELLED;
        call->retire(error);
    }
    return call->request;
}
} // namespace omni::usd::live
