// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "LiveLayerRead.h"
#include "LiveDiagnostics.h"
#include <grpcpp/alarm.h>
#include <array>
#include <cassert>
#include <future>

using namespace omni::usd::live;
namespace {
void readDeadlinePreparation(Runtime& runtime, OmniUsdLiveEndpoint endpoint)
{
    class DeadlineTransport final : public LayerReadTransport {
    public:
        uint32_t timeoutMs = 0;
        bool reached = false;
        std::unique_ptr<grpc::ClientAsyncReaderInterface<rpc::SubscribeReadLayerResponse>> prepare(
            grpc::ClientContext* context, const rpc::SubscribeReadLayerRequest&, grpc::CompletionQueue*) override
        {
            reached = true;
            const auto deadline = context->raw_deadline();
            if (timeoutMs) {
                assert(deadline.clock_type == GPR_CLOCK_MONOTONIC);
                assert(gpr_time_cmp(deadline, gpr_inf_future(GPR_CLOCK_MONOTONIC)) < 0);
            } else {
                assert(gpr_time_cmp(deadline, gpr_inf_future(deadline.clock_type)) == 0);
            }
            return {}; // Explicitly reject preparation; no RPC or expiry wait.
        }
    };
    for (const uint32_t timeout : {0u, 1u, UINT32_MAX}) {
        DeadlineTransport transport;
        transport.timeoutMs = timeout;
        LayerReadOptions options;
        options.timeoutMs = timeout;
        LayerReadCallbacks callbacks;
        callbacks.update = [](auto) { assert(false); };
        try {
            (void)subscribeLayerRead(runtime, endpoint, "scripted", "root.live", options,
                                     std::move(callbacks), &transport);
            assert(false);
        } catch (OmniUsdLiveResult result) {
            assert(result == OMNI_USD_LIVE_RESOURCE_EXHAUSTED && transport.reached);
        }
        const auto metrics = runtime.diagnostics->snapshot(0, 0).metrics;
        assert(metrics.pendingBytes.reservations.current == 0 && metrics.cqTags.submitted == 0);
    }
}
struct Script {
    std::promise<void> partialConsumed;
    std::promise<void*> pendingTag;
    grpc::CompletionQueue* queue = nullptr;
    grpc::Alarm cancellation;
    unsigned reads = 0;
    void completeCancelledRead()
    {
        auto* tag = pendingTag.get_future().get();
        cancellation.Set(queue, std::chrono::system_clock::now() + std::chrono::hours(1), tag);
        cancellation.Cancel(); // Deterministic false CQ event after caller cancellation.
    }
};
class Reader final : public grpc::ClientAsyncReaderInterface<rpc::SubscribeReadLayerResponse> {
public:
    explicit Reader(std::shared_ptr<Script> script) : script_(std::move(script)) {}
    void StartCall(void* tag) override { signal(tag); }
    void ReadInitialMetadata(void*) override { assert(false); }
    void Read(rpc::SubscribeReadLayerResponse* output, void* tag) override
    {
        const auto index = script_->reads++;
        if (index == 0) {
            auto* header = output->mutable_header();
            header->set_object_id(1); header->set_sequence_num(1); header->set_total_size(5);
            signal(tag);
        } else if (index == 1) {
            output->mutable_chunk()->set_data("ab");
            signal(tag);
        } else {
            assert(index == 2);
            script_->pendingTag.set_value(tag);
            // Requesting the next frame proves that the production worker has
            // accepted the header and appended the preceding partial chunk.
            script_->partialConsumed.set_value();
        }
    }
    void Finish(grpc::Status* status, void* tag) override
    {
        *status = grpc::Status(grpc::StatusCode::CANCELLED, "");
        signal(tag);
    }
private:
    std::shared_ptr<Script> script_;
    std::array<grpc::Alarm, 4> alarms_;
    size_t next_ = 0;
    void signal(void* tag)
    {
        assert(next_ < alarms_.size());
        alarms_[next_++].Set(script_->queue, std::chrono::system_clock::now(), tag);
    }
};
class Transport final : public LayerReadTransport {
public:
    std::shared_ptr<Script> script = std::make_shared<Script>();
    std::unique_ptr<grpc::ClientAsyncReaderInterface<rpc::SubscribeReadLayerResponse>> prepare(
        grpc::ClientContext*, const rpc::SubscribeReadLayerRequest& input, grpc::CompletionQueue* queue) override
    {
        assert(input.object_id() == 0 && input.sequence_num() == 0 && !input.client_id().empty());
        script->queue = queue;
        return std::make_unique<Reader>(script);
    }
};

// A real CQ event forwards to the production tag, then announces that its
// receipt-side work has completed. Holding the sole dispatch worker separates
// CQ admission from assembly/callback processing without relying on sleeps.
struct FrameSlot : std::enable_shared_from_this<FrameSlot> {
    rpc::SubscribeReadLayerResponse* output = nullptr;
    void* tag = nullptr;
    grpc::CompletionQueue* queue = nullptr;
    grpc::Alarm alarm;
    std::promise<void> observed;
    std::shared_future<void> received = observed.get_future().share();
    struct Forward final : CompletionTag {
        std::unique_ptr<CompletionTag> actual;
        std::shared_ptr<FrameSlot> slot;
        void onComplete(bool ok, Clock::time_point) noexcept override
        {
            actual->complete(ok);
            slot->observed.set_value();
        }
    };
    void send(const rpc::SubscribeReadLayerResponse& frame = {}, bool ok = true)
    {
        assert(tag);
        if (output) *output = frame;
        auto forward = std::make_unique<Forward>();
        forward->actual.reset(static_cast<CompletionTag*>(tag));
        forward->slot = shared_from_this();
        tag = nullptr;
        LIVE_TSAN_RELEASE(forward.get());
        alarm.Set(queue, std::chrono::system_clock::now() + (ok ? std::chrono::seconds(0) : std::chrono::hours(1)),
                  forward.release());
        if (!ok) alarm.Cancel();
        assert(received.wait_for(std::chrono::seconds(3)) == std::future_status::ready);
    }
};
struct DrainScript {
    grpc::CompletionQueue* queue = nullptr;
    grpc::Status finishStatus{grpc::StatusCode::CANCELLED, ""};
    std::mutex mutex;
    std::condition_variable changed;
    std::vector<std::shared_ptr<FrameSlot>> reads;
    std::shared_ptr<FrameSlot> finish;
    std::shared_ptr<FrameSlot> awaitRead(size_t index)
    {
        std::unique_lock<std::mutex> lock(mutex);
        assert(changed.wait_for(lock, std::chrono::seconds(3), [&] { return reads.size() > index; }));
        return reads[index];
    }
    std::shared_ptr<FrameSlot> awaitFinish()
    {
        std::unique_lock<std::mutex> lock(mutex);
        assert(changed.wait_for(lock, std::chrono::seconds(3), [&] { return bool(finish); }));
        return finish;
    }
};
class DrainReader final : public grpc::ClientAsyncReaderInterface<rpc::SubscribeReadLayerResponse> {
public:
    explicit DrainReader(std::shared_ptr<DrainScript> script) : script_(std::move(script)) {}
    void StartCall(void* tag) override { start_.Set(script_->queue, std::chrono::system_clock::now(), tag); }
    void ReadInitialMetadata(void*) override { assert(false); }
    void Read(rpc::SubscribeReadLayerResponse* output, void* tag) override
    {
        auto slot = std::make_shared<FrameSlot>();
        slot->output = output; slot->tag = tag; slot->queue = script_->queue;
        std::lock_guard<std::mutex> lock(script_->mutex);
        script_->reads.push_back(std::move(slot)); script_->changed.notify_all();
    }
    void Finish(grpc::Status* status, void* tag) override
    {
        *status = script_->finishStatus;
        auto slot = std::make_shared<FrameSlot>();
        slot->tag = tag; slot->queue = script_->queue;
        std::lock_guard<std::mutex> lock(script_->mutex);
        assert(!script_->finish);
        script_->finish = std::move(slot); script_->changed.notify_all();
    }
private:
    std::shared_ptr<DrainScript> script_;
    grpc::Alarm start_;
};
class DrainTransport final : public LayerReadTransport {
public:
    std::shared_ptr<DrainScript> script = std::make_shared<DrainScript>();
    std::unique_ptr<grpc::ClientAsyncReaderInterface<rpc::SubscribeReadLayerResponse>> prepare(
        grpc::ClientContext*, const rpc::SubscribeReadLayerRequest&, grpc::CompletionQueue* queue) override
    {
        script->queue = queue;
        return std::make_unique<DrainReader>(script);
    }
};
struct HeldWorker final : DispatchJob {
    std::promise<void> entered, resume;
    std::shared_future<void> gate = resume.get_future().share();
    void dispatch() noexcept override { entered.set_value(); gate.wait(); }
    static std::shared_ptr<HeldWorker> hold(Runtime& runtime)
    {
        auto job = std::make_shared<HeldWorker>();
        auto entered = job->entered.get_future();
        runtime.requests->executor().enqueue(*job);
        assert(entered.wait_for(std::chrono::seconds(3)) == std::future_status::ready);
        return job;
    }
};
rpc::SubscribeReadLayerResponse header(uint64_t sequence, uint64_t size)
{
    rpc::SubscribeReadLayerResponse frame;
    auto* value = frame.mutable_header();
    value->set_object_id(1); value->set_sequence_num(sequence); value->set_total_size(size);
    return frame;
}
rpc::SubscribeReadLayerResponse chunk(const char* bytes)
{
    rpc::SubscribeReadLayerResponse frame;
    frame.mutable_chunk()->set_data(bytes);
    return frame;
}
void stopRead(Runtime& runtime, std::shared_ptr<Request>& request,
              const std::shared_ptr<DrainScript>& script, size_t pendingRead)
{
    assert(request->cancel(0) == OMNI_USD_LIVE_TIMEOUT);
    script->awaitRead(pendingRead)->send({}, false);
    script->awaitFinish()->send();
    assert(request->wait(3000) == OMNI_USD_LIVE_OK);
    assert(runtime.requests->release(request->handle) == OMNI_USD_LIVE_OK);
    request.reset();
}

void readDrainHandoffs(Runtime& runtime, OmniUsdLiveEndpoint endpoint)
{
    DrainTransport transport;
    LayerReadOptions options;
    options.drain = runtime.drains->makeState();
    std::shared_ptr<const LayerReadUpdate> retained;
    std::promise<void> callbackEntered, callbackResume;
    auto entered = callbackEntered.get_future();
    auto resume = callbackResume.get_future().share();
    unsigned updates = 0;
    LayerReadCallbacks callbacks;
    callbacks.update = [&](auto update) {
        assert(update->drain);
        retained = std::move(update);
        if (++updates == 1) {
            assert(omniUsdLiveWaitForPendingUpdates(0) == OMNI_USD_LIVE_BUSY);
            callbackEntered.set_value();
            resume.wait();
        }
    };
    auto request = subscribeLayerRead(runtime, endpoint, "scripted", "root.live", options,
                                      std::move(callbacks), &transport);
    auto first = transport.script->awaitRead(0);
    assert(omniUsdLiveWaitForPendingUpdates(2000) == OMNI_USD_LIVE_OK); // An issued future Read is excluded.
    auto worker = HeldWorker::hold(runtime);
    const auto beforeForward = runtime.diagnostics->snapshot(0, 0).metrics.cqTags;
    first->send(header(1, 5));
    const auto afterForward = runtime.diagnostics->snapshot(0, 0).metrics.cqTags;
    assert(afterForward.submitted == beforeForward.submitted);
    assert(afterForward.receivedOk == beforeForward.receivedOk + 1);
    assert(afterForward.outstanding + 1 == beforeForward.outstanding);
    const auto headerCapture = runtime.drains->capture();
    assert(runtime.drains->wait(headerCapture, DrainRegistry::Clock::now()) == OMNI_USD_LIVE_TIMEOUT);
    worker->resume.set_value();
    auto partial = transport.script->awaitRead(1);
    assert(runtime.drains->wait(headerCapture, DrainRegistry::Clock::now() + std::chrono::seconds(3)) == OMNI_USD_LIVE_OK);
    assert(!retained && updates == 0);

    worker = HeldWorker::hold(runtime);
    partial->send(chunk("ab"));
    const auto partialCapture = runtime.drains->capture();
    assert(runtime.drains->wait(partialCapture, DrainRegistry::Clock::now()) == OMNI_USD_LIVE_TIMEOUT);
    worker->resume.set_value();
    auto final = transport.script->awaitRead(2);
    assert(runtime.drains->wait(partialCapture, DrainRegistry::Clock::now() + std::chrono::seconds(3)) == OMNI_USD_LIVE_OK);
    assert(!retained && updates == 0); // Partial assembly does not wait for future chunks.

    worker = HeldWorker::hold(runtime);
    final->send(chunk("cde"));
    const auto completeCapture = runtime.drains->capture();
    assert(runtime.drains->wait(completeCapture, DrainRegistry::Clock::now()) == OMNI_USD_LIVE_TIMEOUT);
    worker->resume.set_value();
    assert(entered.wait_for(std::chrono::seconds(3)) == std::future_status::ready);
    assert(retained && retained->bytes == "abcde" && retained->header.sequence_num() == 1);
    assert(runtime.drains->wait(completeCapture, DrainRegistry::Clock::now()) == OMNI_USD_LIVE_TIMEOUT);
    callbackResume.set_value();
    auto caughtUp = transport.script->awaitRead(3);
    assert(omniUsdLiveWaitForPendingUpdates(0) == OMNI_USD_LIVE_TIMEOUT);
    retained->drain->complete();
    assert(runtime.drains->wait(completeCapture, DrainRegistry::Clock::now()) == OMNI_USD_LIVE_OK);
    assert(retained->bytes == "abcde"); // Explicit retirement need not release scene content.
    retained.reset();

    worker = HeldWorker::hold(runtime);
    caughtUp->send(header(1, 0));
    const auto caughtCapture = runtime.drains->capture();
    assert(runtime.drains->wait(caughtCapture, DrainRegistry::Clock::now()) == OMNI_USD_LIVE_TIMEOUT);
    worker->resume.set_value();
    transport.script->awaitRead(4);
    assert(updates == 2 && retained && retained->caughtUp && retained->bytes.empty());
    assert(runtime.drains->wait(caughtCapture, DrainRegistry::Clock::now()) == OMNI_USD_LIVE_TIMEOUT);
    retained.reset(); // Last consumer destruction completes the delivered ticket.
    assert(runtime.drains->wait(caughtCapture, DrainRegistry::Clock::now() + std::chrono::seconds(3)) == OMNI_USD_LIVE_OK);
    stopRead(runtime, request, transport.script, 4);
    assert(runtime.drains->wait(completeCapture, DrainRegistry::Clock::now()) == OMNI_USD_LIVE_OK);
}

void malformedFrameDrainFailure(Runtime& runtime, OmniUsdLiveEndpoint endpoint)
{
    DrainTransport transport;
    LayerReadOptions options;
    options.drain = runtime.drains->makeState();
    unsigned updates = 0, closed = 0;
    LayerReadCallbacks callbacks;
    callbacks.update = [&](auto) { ++updates; };
    callbacks.closed = [&](const auto& error, auto) { assert(error.result == OMNI_USD_LIVE_MALFORMED_STREAM); ++closed; };
    auto request = subscribeLayerRead(runtime, endpoint, "scripted", "root.live", options,
                                      std::move(callbacks), &transport);
    auto frame = transport.script->awaitRead(0);
    auto worker = HeldWorker::hold(runtime);
    frame->send(chunk("missing-header"));
    const auto captured = runtime.drains->capture();
    assert(runtime.drains->wait(captured, DrainRegistry::Clock::now()) == OMNI_USD_LIVE_TIMEOUT);
    worker->resume.set_value();
    auto finish = transport.script->awaitFinish();
    // Local validation failure must be latched before retiring the frame;
    // successful transport teardown cannot disguise it as an empty drain.
    assert(runtime.drains->wait(captured, DrainRegistry::Clock::now()) == OMNI_USD_LIVE_MALFORMED_STREAM);
    finish->send();
    assert(request->wait(3000) == OMNI_USD_LIVE_OK && updates == 0 && closed == 1);
    assert(runtime.requests->release(request->handle) == OMNI_USD_LIVE_OK);
    request.reset();
    assert(runtime.drains->wait(captured, DrainRegistry::Clock::now()) == OMNI_USD_LIVE_MALFORMED_STREAM);
}

void finishDrainHandoff(Runtime& runtime, OmniUsdLiveEndpoint endpoint)
{
    DrainTransport transport;
    transport.script->finishStatus = grpc::Status::OK;
    LayerReadOptions options;
    options.drain = runtime.drains->makeState();
    std::shared_ptr<DrainTicket> retained;
    std::promise<void> entered, resume;
    auto callbackEntered = entered.get_future();
    auto gate = resume.get_future().share();
    LayerReadCallbacks callbacks;
    callbacks.update = [](auto) { assert(false); };
    callbacks.closed = [&](const auto& error, auto ticket) {
        assert(error.result == OMNI_USD_LIVE_UNAVAILABLE && ticket);
        retained = std::move(ticket);
        assert(omniUsdLiveWaitForPendingUpdates(0) == OMNI_USD_LIVE_BUSY);
        entered.set_value();
        gate.wait();
    };
    auto request = subscribeLayerRead(runtime, endpoint, "scripted", "root.live", options,
                                      std::move(callbacks), &transport);
    auto first = transport.script->awaitRead(0);
    auto worker = HeldWorker::hold(runtime);
    first->send({}, false); // EOF before any content.
    const auto eofCapture = runtime.drains->capture();
    assert(runtime.drains->wait(eofCapture, DrainRegistry::Clock::now()) == OMNI_USD_LIVE_TIMEOUT);
    worker->resume.set_value();
    auto finish = transport.script->awaitFinish();
    assert(runtime.drains->wait(eofCapture, DrainRegistry::Clock::now()) == OMNI_USD_LIVE_TIMEOUT);
    worker = HeldWorker::hold(runtime);
    finish->send();
    const auto captured = runtime.drains->capture();
    assert(runtime.drains->wait(captured, DrainRegistry::Clock::now()) == OMNI_USD_LIVE_TIMEOUT);
    worker->resume.set_value();
    assert(callbackEntered.wait_for(std::chrono::seconds(3)) == std::future_status::ready);
    assert(retained && runtime.drains->wait(captured, DrainRegistry::Clock::now()) == OMNI_USD_LIVE_TIMEOUT);
    assert(runtime.drains->wait(eofCapture, DrainRegistry::Clock::now()) == OMNI_USD_LIVE_TIMEOUT);
    resume.set_value();
    assert(request->wait(3000) == OMNI_USD_LIVE_OK);
    // Delivery quiescence does not retire a terminal event retained by an actor
    // for subsequent retry/failure classification on its own worker.
    assert(runtime.drains->wait(captured, DrainRegistry::Clock::now()) == OMNI_USD_LIVE_TIMEOUT);
    retained->complete();
    assert(runtime.drains->wait(captured, DrainRegistry::Clock::now()) == OMNI_USD_LIVE_OK);
    assert(runtime.drains->wait(eofCapture, DrainRegistry::Clock::now()) == OMNI_USD_LIVE_OK);
    assert(runtime.requests->release(request->handle) == OMNI_USD_LIVE_OK);
    request.reset(); retained.reset();
}

void finishDrainIncludesCallbackCaptureDestruction(Runtime& runtime, OmniUsdLiveEndpoint endpoint)
{
    struct CleanupBarrier {
        std::promise<void> entered, resume;
        std::shared_future<void> gate = resume.get_future().share();
    };
    struct HeldCapture {
        std::shared_ptr<CleanupBarrier> barrier;
        explicit HeldCapture(std::shared_ptr<CleanupBarrier> value) : barrier(std::move(value)) {}
        ~HeldCapture()
        {
            barrier->entered.set_value();
            barrier->gate.wait();
        }
    };
    DrainTransport transport;
    transport.script->finishStatus = grpc::Status::OK;
    LayerReadOptions options;
    options.drain = runtime.drains->makeState();
    auto barrier = std::make_shared<CleanupBarrier>();
    auto cleanupEntered = barrier->entered.get_future();
    auto captureOwner = std::make_shared<HeldCapture>(barrier);
    unsigned closed = 0;
    LayerReadCallbacks callbacks;
    callbacks.update = [](auto) { assert(false); };
    callbacks.closed = [captureOwner, &closed](const auto& error, auto ticket) {
        assert(error.result == OMNI_USD_LIVE_UNAVAILABLE && ticket && captureOwner);
        ++closed;
        // The consumer neither retains nor explicitly completes this ticket.
        // Payload cleanup must keep it alive through destruction of this capture.
    };
    auto request = subscribeLayerRead(runtime, endpoint, "scripted", "root.live", options,
                                      std::move(callbacks), &transport);
    captureOwner.reset();
    transport.script->awaitRead(0)->send({}, false);
    auto finish = transport.script->awaitFinish();
    auto worker = HeldWorker::hold(runtime);
    finish->send();
    const auto captured = runtime.drains->capture();
    assert(runtime.drains->wait(captured, DrainRegistry::Clock::now()) == OMNI_USD_LIVE_TIMEOUT);
    worker->resume.set_value();
    assert(cleanupEntered.wait_for(std::chrono::seconds(3)) == std::future_status::ready);
    assert(closed == 1); // The callback body returned; its capture is still retiring.
    assert(request->wait(0) == OMNI_USD_LIVE_TIMEOUT);
    assert(runtime.drains->wait(captured, DrainRegistry::Clock::now()) == OMNI_USD_LIVE_TIMEOUT);
    barrier->resume.set_value();
    assert(request->wait(3000) == OMNI_USD_LIVE_OK);
    assert(runtime.drains->wait(captured, DrainRegistry::Clock::now() + std::chrono::seconds(3)) == OMNI_USD_LIVE_OK);
    assert(runtime.requests->release(request->handle) == OMNI_USD_LIVE_OK);
    request.reset();
}

void awaitDrainOwnerRetirement(Runtime& runtime)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (!runtime.drains->capture().targets.empty()) {
        assert(std::chrono::steady_clock::now() < deadline); std::this_thread::yield();
    }
}
}
void testLayerReadLifecycle()
{
    assert(omniUsdLiveWaitForPendingUpdates(0) == OMNI_USD_LIVE_NOT_INITIALIZED);
    OmniUsdLiveRuntimeOptions runtimeOptions{};
    runtimeOptions.struct_size = sizeof(runtimeOptions); runtimeOptions.abi_version = OMNI_USD_LIVE_ABI_VERSION;
    runtimeOptions.callback_worker_count = 1;
    assert(omniUsdLiveInitialize(&runtimeOptions) == OMNI_USD_LIVE_OK);
    OmniUsdLiveEndpointOptions options{};
    options.struct_size = sizeof(options); options.abi_version = OMNI_USD_LIVE_ABI_VERSION;
    options.authority = "read-script.example.invalid"; options.access_token = "synthetic-script";
    OmniUsdLiveEndpoint endpoint = 0;
    assert(omniUsdLiveRegisterEndpoint(&options, &endpoint) == OMNI_USD_LIVE_OK);
    auto runtime = getRuntime();
    readDeadlinePreparation(*runtime, endpoint);
    Transport transport;
    auto partial = transport.script->partialConsumed.get_future();
    unsigned updates = 0, closed = 0;
    LayerReadCallbacks callbacks;
    callbacks.update = [&](auto) { ++updates; };
    callbacks.closed = [&](const auto&, auto) { ++closed; };
    auto request = subscribeLayerRead(*runtime, endpoint, "scripted", "root.live", {}, std::move(callbacks), &transport);
    assert(partial.wait_for(std::chrono::seconds(3)) == std::future_status::ready);
    auto tags = runtime->diagnostics->snapshot(0, 0).metrics.cqTags;
    assert(tags.submitted == 4 && tags.receivedOk == 3 && tags.outstanding == 1);
    auto retained = runtime->findEndpoint(endpoint);
    {
        std::lock_guard<std::mutex> lock(retained->mutex);
        assert(retained->pendingTransportBytes > layerReadFrameLimit(runtime->options.values.max_update_bytes));
    }
    assert(request->cancel(0) == OMNI_USD_LIVE_TIMEOUT);
    transport.script->completeCancelledRead();
    assert(request->wait(3000) == OMNI_USD_LIVE_OK);
    assert(!updates && !closed && transport.script->reads == 3);
    OmniUsdLiveResult result;
    assert(request->result(&result) == OMNI_USD_LIVE_OK && result == OMNI_USD_LIVE_CANCELLED);
    tags = runtime->diagnostics->snapshot(0, 0).metrics.cqTags;
    assert(tags.submitted == 5 && tags.receivedOk == 4 && tags.receivedNotOk == 1 && tags.outstanding == 0);
    {
        std::lock_guard<std::mutex> lock(retained->mutex);
        assert(retained->pendingTransportBytes == 0);
    }
    assert(runtime->requests->release(request->handle) == OMNI_USD_LIVE_OK);
    request.reset(); retained.reset();
    readDrainHandoffs(*runtime, endpoint);
    awaitDrainOwnerRetirement(*runtime);
    finishDrainHandoff(*runtime, endpoint);
    awaitDrainOwnerRetirement(*runtime);
    finishDrainIncludesCallbackCaptureDestruction(*runtime, endpoint);
    awaitDrainOwnerRetirement(*runtime);
    malformedFrameDrainFailure(*runtime, endpoint);
    assert(omniUsdLiveUnregisterEndpoint(endpoint, 0) == OMNI_USD_LIVE_OK);
    assert(omniUsdLiveShutdown(3000) == OMNI_USD_LIVE_OK);
    tags = runtime->diagnostics->snapshot(0, 0).metrics.cqTags;
    assert(tags.submitted == tags.retired && tags.outstanding == 0 && tags.retained == 0 && tags.handling == 0);
    assert(tags.abandoned == 0);
    runtime.reset();
}
