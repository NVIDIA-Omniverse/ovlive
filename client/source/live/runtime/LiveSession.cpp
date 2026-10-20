// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "LiveSession.h"
#include "LiveSessionPeers.h"
#include "LiveAuth.h"
#include "LiveDeadline.h"
#include "LiveDiagnostics.h"
#include "LiveLayerWrite.h"
#include "LiveRequest.h"
#include "LiveStatus.h"
#include "LiveUnary.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <unordered_map>

namespace omni::usd::live {
namespace {
std::atomic<uint64_t> nextSession{1};
uint64_t allocateSession() {
    auto n = nextSession.load();
    do {
        if (n >= (UINT64_C(1) << 56)) throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    } while (!nextSession.compare_exchange_weak(n, n + 1));
    return (UINT64_C(2) << 56) | n;
}
using Clock = std::chrono::steady_clock;
class Session final : public ScheduledOperation, public DispatchJob {
    enum Phase : unsigned { Start = 1, Read = 2, Write = 4, Finish = 8 };
    struct Message {
        PendingBytes reservation;
        rpc::JoinSessionRequest frame;
        uint64_t size, request = 0;
        Message(PendingBytes bytes, uint64_t n) : reservation(std::move(bytes)), size(n) {}
    };
    struct Attempt {
        PendingBytes receiveReservation;
        grpc::ClientContext context;
        grpc::Status status;
        std::unique_ptr<grpc::ClientAsyncReaderWriter<rpc::JoinSessionRequest, rpc::JoinSessionResponse>> stream;
        rpc::JoinSessionRequest header, keepaliveFrame;
        rpc::JoinSessionResponse incoming;
        SessionKeepalive keepalive;
        std::unique_ptr<CompletionTag> finishTag;
        std::shared_ptr<Message> active;
        uint64_t generation = 0, authGeneration = 0, authFailure = 0;
        Clock::time_point started{}, completed{}, written{}; // written: CQ receipt of the last write.
        unsigned pending = 0, successful = 0; // Protected by session mutex.
        bool startDone = false, reading = false, writing = false, finishing = false, finished = false;
        bool broken = false, headerWritten = false, connected = false, issued = false, finishOk = false;
        explicit Attempt(PendingBytes bytes) : receiveReservation(std::move(bytes)) {}
    };
    struct Tag final : CompletionTag {
        std::shared_ptr<Session> owner;
        std::shared_ptr<Attempt> attempt;
        Phase phase;
        explicit Tag(Phase p) : phase(p) {}
        ~Tag() override {
            attempt.reset();
            if (owner) owner->tagRetired();
        }
        void onComplete(bool ok, Clock::time_point received) noexcept override {
            owner->completed(attempt, phase, ok, received);
        }
    };
public:
    Runtime& runtime;
    const uint64_t handle, endpointHandle;
    const std::string uri, sessionId, app;
    uint64_t subject = 0;
    Session(Runtime& r, EndpointLease lease, uint64_t id, const LiveUri& parsed, std::string application, void* data,
        OmniUsdLiveSessionCallback function, void (*afterPrepare)(void*, uint64_t) noexcept = nullptr,
        void* afterPrepareContext = nullptr)
        : runtime(r), handle(id), endpointHandle(lease->handle), uri(parsed.canonical), sessionId(parsed.sessionId),
          app(std::move(application)), endpoint_(std::move(lease)), user_(data), callback_(function),
          messages_(r.options.values.max_pending_messages_per_session),
          peers_({r.options.values.max_pending_messages_per_session, r.options.values.max_pending_bytes_per_endpoint,
                     kMaxUriBytes},
              r.options.values.peer_reconciliation_grace_ms),
          openDeadline_(Clock::now() + std::chrono::milliseconds(r.options.values.initial_open_timeout_ms)),
          authObservation_(*r.diagnostics, AuthWaitKind::Session), afterPrepare_(afterPrepare),
          afterPrepareContext_(afterPrepareContext) {
        std::lock_guard<std::mutex> lock(endpoint_->mutex);
        // Starvation is known before the first step, so an open deadline that passes first classifies it (design section 14).
        authWait_ = !r.auth->usable(*endpoint_, wallTimeMs());
        authFailure_ = endpoint_->authFailureSerial;
        authDeadline_ = Clock::now() + std::chrono::milliseconds(endpoint_->options.authWaitMs);
    }
    ~Session() override {
        if (subject) runtime.statuses->remove(subject);
    }
    void activateStatus() {
        OmniUsdLiveStatus status{};
        status.struct_size = sizeof(status);
        status.abi_version = OMNI_USD_LIVE_ABI_VERSION;
        status.subject_kind = OMNI_USD_LIVE_SUBJECT_SESSION;
        status.state = authWait_ ? OMNI_USD_LIVE_STATE_AUTH_WAIT : OMNI_USD_LIVE_STATE_OPENING;
        status.operation = authWait_ ? OMNI_USD_LIVE_OPERATION_AUTH : OMNI_USD_LIVE_OPERATION_JOIN;
        status.result = authWait_ ? OMNI_USD_LIVE_UNAUTHENTICATED : OMNI_USD_LIVE_OK;
        status.recovery_action = authWait_ ? OMNI_USD_LIVE_RECOVERY_REFRESH_AUTH : OMNI_USD_LIVE_RECOVERY_NONE;
        status.progress_known = 1;
        subject = runtime.statuses->add(endpointHandle, handle, uri, status);
    }
    std::shared_ptr<Request> send(const void* bytes, uint64_t size, void* user, OmniUsdLiveResultCallback cb) try {
        if ((!bytes && size) || !cb) throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
        if (size > runtime.options.values.max_update_bytes || size > kMaxSessionMessageBytes)
            throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
        std::function<void()> completion = [user, cb] { cb(user, OMNI_USD_LIVE_OK); };
        // A closed or full session refuses before any preparation or copy.
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto refused = admissionLocked();
            if (refused != OMNI_USD_LIVE_OK) throw refused;
        }
        // Preparation runs outside the session mutex: it takes runtime and
        // endpoint locks, copies the payload and creates the request.
        auto lease = runtime.acquireEndpoint(endpointHandle);
        if (!lease) throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_HANDLE);
        auto prepared = std::make_shared<Message>(
            PendingBytes(std::move(lease), size, runtime.options.values.max_pending_bytes_per_endpoint), size);
        // This pinned protobuf's mutable oneof accessor sets its case
        // before allocating the child. A throwing allocation would leave
        // an invalid pointer for frame destruction. Prepare the heap child
        // completely, then use same-arena (both null) ownership transfer.
        auto payload = std::make_unique<rpc::SessionMessageRequest>();
        if (size) payload->set_data(bytes, static_cast<size_t>(size));
        prepared->frame.set_allocated_message(payload.get());
        (void)payload.release();
        auto request = runtime.requests->create(endpointHandle, OMNI_USD_LIVE_OPERATION_MESSAGE, {});
        prepared->request = request->handle;
        if (afterPrepare_) afterPrepare_(afterPrepareContext_, request->handle);
        // Acceptance commits under the mutex, so FIFO order is commit order
        // and the capacity bound holds against concurrent senders and Leave.
        // Nothing in the commit allocates or throws.
        OmniUsdLiveResult refused;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            refused = admissionLocked();
            if (refused == OMNI_USD_LIVE_OK) {
                messages_[(head_ + count_++) % messages_.size()] = prepared;
                pendingBytes_ += size;
                messageRecordLocked(OMNI_USD_LIVE_RECORD_MESSAGE_ACCEPTED, OMNI_USD_LIVE_OK, size, prepared->request);
                publishLocked();
            }
        }
        if (refused != OMNI_USD_LIVE_OK) {
            // Release the prepared slot and bytes; the caller never receives
            // this handle and no callback is scheduled.
            OmniUsdLiveRequestError error{};
            error.result = refused;
            request->finish(error, {});
            (void)runtime.requests->release(request->handle);
            prepared->reservation.releaseQuietly();
            throw refused;
        }
        OmniUsdLiveRequestError error{};
        error.result = OMNI_USD_LIVE_OK;
        request->finish(error, std::move(completion));
        wake();
        return request;
    } catch (OmniUsdLiveResult result) {
        messageRejected(size, result);
        throw;
    } catch (const std::invalid_argument&) {
        messageRejected(size, OMNI_USD_LIVE_INVALID_ARGUMENT);
        throw;
    } catch (const std::bad_alloc&) {
        messageRejected(size, OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
        throw;
    } catch (const std::length_error&) {
        messageRejected(size, OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
        throw;
    } catch (...) {
        messageRejected(size, OMNI_USD_LIVE_INTERNAL_ERROR);
        throw;
    }
    OmniUsdLiveResult close(uint32_t timeout) {
        const auto deadline = Clock::now() + std::chrono::milliseconds(timeout);
        std::shared_ptr<Attempt> call;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closing_ = true;
            call = call_;
            publishLocked();
        }
        if (call) call->context.TryCancel();
        wake();
        std::unique_lock<std::mutex> lock(mutex_);
        if (!(stopped_ && !scheduled_ && !tags_)) {
            if (DispatchExecutor::isWorker()) return OMNI_USD_LIVE_BUSY;
            if (!closed_.wait_until(lock, deadline, [&] { return stopped_ && !scheduled_ && !tags_; }))
                return OMNI_USD_LIVE_TIMEOUT;
        }
        return OMNI_USD_LIVE_OK;
    }
    void detach() {
        EndpointLease retired;
        uint64_t removed = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            assert(stopped_ && !scheduled_ && !tags_);
            retired = std::move(endpoint_);
            removed = subject;
            subject = 0;
        }
        if (removed) runtime.statuses->remove(removed);
    }
    void expirePeersForTest() noexcept {
        expirePeers_.store(true, std::memory_order_release);
        wake();
    }
    void expireKeepaliveForTest() noexcept {
        expireKeepalive_.store(true, std::memory_order_release);
        wake();
    }
    OmniUsdLiveResult snapshot(OmniUsdLiveStatus* out) {
        std::lock_guard<std::mutex> lock(mutex_);
        return subject ? runtime.statuses->snapshot(subject, out) : OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_HANDLE);
    }
    ScheduledStep advance(Clock::time_point) noexcept override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopped_ && !scheduled_ && !tags_) return {true, {}};
        if (scheduled_) {
            dirty_ = true;
            return {false, Clock::time_point::max()};
        }
        observedWake_ = wakeRevision();
        scheduled_ = true;
        dirty_ = false;
        runtime.requests->executor().enqueue(*this);
        return {false, Clock::time_point::max()};
    }
private:
    EndpointLease endpoint_;
    void* user_;
    OmniUsdLiveSessionCallback callback_;
    mutable std::mutex mutex_;
    std::condition_variable closed_;
    std::vector<std::shared_ptr<Message>> messages_;
    size_t head_ = 0, count_ = 0, tags_ = 0;
    uint64_t pendingBytes_ = 0;
    SessionPeers peers_;
    std::shared_ptr<Attempt> call_;
    std::string connection_;
    uint64_t generation_ = 0, attempts_ = 0, authFailure_ = 0, observedWake_ = 0;
    bool scheduled_ = false, dirty_ = true, stopped_ = false, closing_ = false;
    bool ready_ = false, opened_ = false, authWait_ = false, callbackDisabled_ = false, failureDelivered_ = false;
    OmniUsdLiveResult failure_ = OMNI_USD_LIVE_OK, transient_ = OMNI_USD_LIVE_OK;
    bool grpcPresent_ = false;
    std::atomic<bool> expirePeers_{false}, expireKeepalive_{false};
    int32_t grpcCode_ = 0;
    uint64_t grpcGeneration_ = 0;
    Clock::time_point openDeadline_, retryAt_{}, authDeadline_;
    OperationAuthWait authObservation_;
    void (*const afterPrepare_)(void*, uint64_t) noexcept;
    void* const afterPrepareContext_;
    std::shared_ptr<Session> self() { return std::static_pointer_cast<Session>(shared_from_this()); }
    void wake() noexcept { runtime.auth->wake(*this); }
    std::unique_ptr<Tag> prepareTag(const std::shared_ptr<Attempt>& call, Phase phase) {
        auto tag = std::make_unique<Tag>(phase);
        armTag(*tag, call);
        return tag;
    }
    void armTag(Tag& tag, const std::shared_ptr<Attempt>& call) {
        tag.owner = self();
        tag.attempt = call;
        std::lock_guard<std::mutex> lock(mutex_);
        ++tags_;
    }
    void tagRetired() noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            assert(tags_);
            --tags_;
        }
        wake();
    }
    void completed(const std::shared_ptr<Attempt>& call, Phase phase, bool ok, Clock::time_point received) noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            call->pending |= phase;
            if (ok) call->successful |= phase;
            if (phase == Finish) call->completed = received;
            if (phase == Write) call->written = received;
        }
        wake();
    }
    // Caller holds mutex_; numeric emission never invokes host code. Local
    // receipt/queue events use zero transport generations. A copied receipt ID
    // is only correlation and does not keep a released Request alive.
    void messageRecordLocked(uint32_t code, OmniUsdLiveResult result, uint64_t bytes, uint64_t request = 0,
        const Attempt* call = nullptr) noexcept {
        runtime.diagnostics->emit(code, endpointHandle, subject, 0,
            {{DiagnosticKey::Operation, OMNI_USD_LIVE_OPERATION_MESSAGE}, {DiagnosticKey::Result, result},
                {DiagnosticKey::Session, handle}, {DiagnosticKey::Request, request},
                {DiagnosticKey::Generation, call ? call->generation : 0},
                {DiagnosticKey::AuthGeneration, call ? call->authGeneration : 0}, {DiagnosticKey::Bytes, bytes},
                {DiagnosticKey::QueueDepth, count_}, {DiagnosticKey::PendingBytes, pendingBytes_}});
    }
    // Caller holds mutex_. A terminal failure, then Leave, then a full FIFO.
    OmniUsdLiveResult admissionLocked() const noexcept {
        if (failure_ != OMNI_USD_LIVE_OK) return failure_;
        if (closing_ || stopped_) return OMNI_USD_LIVE_CANCELLED;
        if (count_ >= messages_.size()) return OMNI_USD_LIVE_RESOURCE_EXHAUSTED;
        return OMNI_USD_LIVE_OK;
    }
    void messageRejected(uint64_t bytes, OmniUsdLiveResult result) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        messageRecordLocked(OMNI_USD_LIVE_RECORD_MESSAGE_REJECTED, result, bytes);
    }
    void publishLocked() noexcept {
        if (!subject) return;
        OmniUsdLiveStatus status{};
        status.struct_size = sizeof(status);
        status.abi_version = OMNI_USD_LIVE_ABI_VERSION;
        status.subject_kind = OMNI_USD_LIVE_SUBJECT_SESSION;
        status.operation = OMNI_USD_LIVE_OPERATION_JOIN;
        status.result = failure_;
        status.progress_known = 1;
        status.pending_write_count = count_;
        status.pending_bytes = pendingBytes_;
        status.grpc_status_present = grpcPresent_;
        status.grpc_status = grpcCode_;
        if (failure_ != OMNI_USD_LIVE_OK) {
            status.state = OMNI_USD_LIVE_STATE_FAILED;
            status.recovery_action = OMNI_USD_LIVE_RECOVERY_RECONFIGURE;
        } else if (authWait_) {
            status.state = OMNI_USD_LIVE_STATE_AUTH_WAIT;
            status.result = OMNI_USD_LIVE_UNAUTHENTICATED;
            status.operation = OMNI_USD_LIVE_OPERATION_AUTH;
            status.recovery_action = OMNI_USD_LIVE_RECOVERY_REFRESH_AUTH;
        } else if (ready_) status.state = OMNI_USD_LIVE_STATE_READY;
        else if (opened_) {
            status.state = OMNI_USD_LIVE_STATE_RECONNECTING;
            status.result = transient_;
            status.recovery_action = OMNI_USD_LIVE_RECOVERY_WAIT_FOR_RETRY;
        } else status.state = OMNI_USD_LIVE_STATE_OPENING;
        if (closing_) status.state = stopped_ ? OMNI_USD_LIVE_STATE_CLOSED : OMNI_USD_LIVE_STATE_CLOSING;
        runtime.statuses->publish(subject, status);
    }
    void publish() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        publishLocked();
    }
    void fail(OmniUsdLiveResult result, bool transportEvidence = false) noexcept {
        std::shared_ptr<Attempt> call;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (failure_ == OMNI_USD_LIVE_OK) {
                failure_ = result;
                if (!transportEvidence) {
                    grpcPresent_ = false;
                    grpcCode_ = 0;
                }
            }
            ready_ = false;
            call = call_;
            publishLocked();
        }
        if (call) call->context.TryCancel();
    }
    void deliver(uint32_t kind, OmniUsdLiveResult result, const std::string& id, const std::string* user = nullptr,
        const std::string* appName = nullptr, const std::string* bytes = nullptr) {
        OmniUsdLiveSessionCallback callback;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (closing_ || callbackDisabled_) return;
            callback = callback_;
        }
        if (!callback) return;
        OmniUsdLiveSessionEvent event{};
        event.struct_size = sizeof(event);
        event.abi_version = OMNI_USD_LIVE_ABI_VERSION;
        event.kind = kind;
        event.result = result;
        event.connection_id = id.empty() ? nullptr : id.c_str();
        event.user_name = user ? user->c_str() : nullptr;
        event.app_name = appName ? appName->c_str() : nullptr;
        event.bytes = bytes ? bytes->data() : nullptr;
        event.size = bytes ? bytes->size() : 0;
        try {
            callback(user_, &event);
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                callbackDisabled_ = true;
            }
            runtime.diagnostics->emit(OMNI_USD_LIVE_RECORD_CALLBACK_FAILED, endpointHandle, subject, 0,
                {{DiagnosticKey::CallbackKind, static_cast<uint32_t>(CallbackKind::Session)},
                    {DiagnosticKey::Session, handle}, {DiagnosticKey::Reason, kind}});
            fail(OMNI_USD_LIVE_INTERNAL_ERROR);
        }
    }
    void deliverPeers(const std::vector<SessionPeers::Event>& events) {
        for (const auto& event : events)
            deliver(event.kind, OMNI_USD_LIVE_OK, event.peer.connectionId,
                event.kind == OMNI_USD_LIVE_EVENT_JOINED ? &event.peer.userName : nullptr,
                event.kind == OMNI_USD_LIVE_EVENT_JOINED ? &event.peer.appName : nullptr);
    }
    void notifyFailure() {
        OmniUsdLiveResult result;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (failure_ == OMNI_USD_LIVE_OK || failureDelivered_ || closing_ || callbackDisabled_) return;
            result = failure_;
            failureDelivered_ = true;
        }
        deliver(OMNI_USD_LIVE_EVENT_CONNECTION, result, connection_);
    }
    void consume(const std::shared_ptr<Attempt>& call) {
        const auto& frame = call->incoming;
        if (!call->connected) {
            if (!frame.has_connected()) throw OmniUsdLiveResult(OMNI_USD_LIVE_MALFORMED_STREAM);
            const auto& id = frame.connected().connection_id();
            SessionPeers::validateText(id, false, kMaxUriBytes);
            std::string owned = id;
            peers_.beginConnection(call->generation, Clock::now());
            connection_.swap(owned);
            call->connected = true;
            call->keepalive.connected(frame.connected().keepalive_interval_ms());
            {
                std::lock_guard<std::mutex> lock(mutex_);
                // Leave can win after step() inspected the completion batch.
                // Do not announce readiness after its CLOSING publication.
                if (closing_ || failure_ != OMNI_USD_LIVE_OK) return;
                ready_ = opened_ = true;
                authWait_ = false;
                attempts_ = 0;
                runtime.diagnostics->emit(OMNI_USD_LIVE_RECORD_STREAM_RESUMED, endpointHandle, subject, 0,
                    {{DiagnosticKey::Operation, OMNI_USD_LIVE_OPERATION_JOIN},
                        {DiagnosticKey::Result, OMNI_USD_LIVE_OK}, {DiagnosticKey::Session, handle},
                        {DiagnosticKey::Attempt, call->generation}, {DiagnosticKey::Generation, call->generation},
                        {DiagnosticKey::AuthGeneration, call->authGeneration}});
                transient_ = OMNI_USD_LIVE_OK;
                grpcPresent_ = false;
                publishLocked();
            }
            runtime.auth->completed(*endpoint_, call->authGeneration, grpc::StatusCode::OK);
            deliver(OMNI_USD_LIVE_EVENT_CONNECTION, OMNI_USD_LIVE_OK, connection_);
        } else if (frame.has_peer_joined()) {
            const auto& peer = frame.peer_joined();
            if (peer.connection_id() != connection_)
                deliverPeers(
                    peers_.joined(call->generation, {peer.connection_id(), peer.user_name(), peer.user_application()}));
        } else if (frame.has_peer_left()) {
            if (frame.peer_left().connection_id() != connection_)
                deliverPeers(peers_.left(call->generation, frame.peer_left().connection_id()));
        } else if (frame.has_peer_message()) {
            const auto& message = frame.peer_message();
            SessionPeers::validateText(message.connection_id(), false, kMaxUriBytes);
            if (message.data().size() > runtime.options.values.max_update_bytes)
                throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                messageRecordLocked(OMNI_USD_LIVE_RECORD_MESSAGE_RECEIVED, OMNI_USD_LIVE_OK, message.data().size(), 0,
                    call.get());
            }
            deliver(OMNI_USD_LIVE_EVENT_MESSAGE, OMNI_USD_LIVE_OK, message.connection_id(), nullptr, nullptr,
                &message.data());
        } else throw OmniUsdLiveResult(OMNI_USD_LIVE_MALFORMED_STREAM);
    }
    bool authenticate(Clock::time_point now, Clock::time_point& next) {
        bool usable, failed, rejected;
        uint64_t serial, generation;
        {
            std::lock_guard<std::mutex> lock(endpoint_->mutex);
            runtime.auth->evaluate(*endpoint_, wallTimeMs(), now);
            usable = runtime.auth->usable(*endpoint_, wallTimeMs());
            failed = endpoint_->authFailed;
            serial = endpoint_->authFailureSerial;
            rejected = endpoint_->generationRejected;
            generation = endpoint_->tokenGeneration;
        }
        if (failed || serial != authFailure_) {
            fail(OMNI_USD_LIVE_UNAUTHENTICATED);
            return false;
        }
        if (!usable) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (closing_ || failure_ != OMNI_USD_LIVE_OK) return false;
            if (!authWait_) authDeadline_ = now + std::chrono::milliseconds(endpoint_->options.authWaitMs);
            authWait_ = true;
            ready_ = false;
            if (!rejected || !grpcPresent_ || grpcCode_ != grpc::StatusCode::UNAUTHENTICATED ||
                grpcGeneration_ != generation) {
                grpcPresent_ = false;
                grpcCode_ = 0;
            }
            if (now >= authDeadline_) {
                failure_ = OMNI_USD_LIVE_UNAUTHENTICATED;
                grpcPresent_ = false;
                grpcCode_ = 0;
                publishLocked();
            } else {
                authObservation_.start(Clock::now());
                next = std::min(next, authDeadline_);
            }
            return false;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (closing_ || failure_ != OMNI_USD_LIVE_OK) return false;
            authObservation_.finish(OMNI_USD_LIVE_OK, Clock::now());
            authWait_ = false;
        }
        return true;
    }
    void startAttempt() {
        const auto limit = wireLimit(runtime.options.values.max_update_bytes);
        auto lease = runtime.acquireEndpoint(endpointHandle);
        if (!lease) throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_HANDLE);
        auto call = std::make_shared<Attempt>(
            PendingBytes(std::move(lease), limit, runtime.options.values.max_pending_bytes_per_endpoint));
        // Unarmed tag has no owners/cycle. Cleanup after issuance cannot fail
        // because allocating its terminal CQ node was done before admission.
        call->finishTag = std::make_unique<Tag>(Finish);
        call->generation = generation_ + 1;
        if (!call->generation) throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
        {
            std::lock_guard<std::mutex> lock(endpoint_->mutex);
            if (!runtime.auth->usable(*endpoint_, wallTimeMs())) return;
            call->authGeneration = endpoint_->tokenGeneration;
            call->authFailure = endpoint_->authFailureSerial;
            call->context.AddMetadata("authorization", "Bearer " + endpoint_->token);
        }
        call->context.set_wait_for_ready(false);
        auto header = std::make_unique<rpc::JoinSessionHeader>();
        header->set_session_id(sessionId);
        header->set_user_application(app);
        // Every join opts in; only a Connected that reports an interval starts keepalives.
        header->set_client_keepalive(true);
        call->header.set_allocated_header(header.get());
        (void)header.release();
        auto keepalive = std::make_unique<rpc::JoinSessionKeepalive>();
        call->keepaliveFrame.set_allocated_keepalive(keepalive.get());
        (void)keepalive.release();
        call->stream = endpoint_->stub->PrepareAsyncJoinSession(&call->context, &runtime.queue());
        if (!call->stream) throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
        auto tag = prepareTag(call, Start);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (closing_ || failure_ != OMNI_USD_LIVE_OK) return;
            call_ = call;
            generation_ = call->generation;
            ++attempts_;
            call->started = Clock::now();
            call->issued = true;
            runtime.diagnostics->startedRpc(OMNI_USD_LIVE_OPERATION_JOIN);
            call->stream->StartCall(submitTag(tag, *runtime.diagnostics));
        }
    }
    void read(const std::shared_ptr<Attempt>& call) {
        auto tag = prepareTag(call, Read);
        call->incoming.Clear();
        call->reading = true;
        call->stream->Read(&call->incoming, submitTag(tag, *runtime.diagnostics));
    }
    // Writes the header, else the FIFO head, else a due keepalive. A
    // keepalive takes no request, FIFO slot or pending bytes, and it is never
    // issued while another write is in flight or a message is queued.
    void write(const std::shared_ptr<Attempt>& call, Clock::time_point now) {
        std::shared_ptr<Message> message;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (closing_ || failure_ != OMNI_USD_LIVE_OK) return;
            if (call->headerWritten && call->connected && count_) message = messages_[head_];
        }
        bool keepalive = false;
        if (call->headerWritten && !message) {
            if (!call->connected) return;
            const bool expired = expireKeepalive_.load(std::memory_order_acquire);
            if (!call->keepalive.due(expired ? Clock::time_point::max() : now, false, false)) return;
            if (expired) expireKeepalive_.store(false, std::memory_order_release);
            keepalive = true;
        }
        auto tag = prepareTag(call, Write);
        call->active = std::move(message);
        call->writing = true;
        // Chosen before submitTag: its release is the last access before completion.
        const auto& frame = call->active ? call->active->frame : keepalive ? call->keepaliveFrame : call->header;
        call->stream->Write(frame, submitTag(tag, *runtime.diagnostics));
    }
    void finishTransport(const std::shared_ptr<Attempt>& call) {
        auto tag = std::unique_ptr<Tag>(static_cast<Tag*>(call->finishTag.release()));
        assert(tag);
        armTag(*tag, call);
        call->finishing = true;
        call->stream->Finish(&call->status, submitTag(tag, *runtime.diagnostics));
    }
    void settle(const std::shared_ptr<Attempt>& call) {
        auto result = call->finishOk ? classifyUnary(call->status.error_code(), false) :
                                       OmniUsdLiveResult(OMNI_USD_LIVE_INTERNAL_ERROR);
        if (result == OMNI_USD_LIVE_OK) result = OMNI_USD_LIVE_UNAVAILABLE;
        if (call->finishOk) runtime.auth->completed(*endpoint_, call->authGeneration, call->status.error_code());
        runtime.diagnostics->observeRpc(endpointHandle, OMNI_USD_LIVE_OPERATION_JOIN, result,
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::milliseconds>(call->completed - call->started).count()),
            0, call->generation, call->finishOk, call->finishOk ? call->status.error_code() : 0, subject,
            call->generation, call->authGeneration, 0, handle);
        peers_.disconnected(call->generation);
        bool closed, terminal;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            call_.reset();
            ready_ = false;
            closed = closing_;
            terminal = failure_ != OMNI_USD_LIVE_OK;
            if (!terminal) {
                grpcPresent_ = call->finishOk;
                grpcCode_ = call->finishOk ? call->status.error_code() : 0;
                grpcGeneration_ = call->authGeneration;
                transient_ = result;
            }
        }
        // Failed write bytes remain at the FIFO head. Replaying a session
        // message is explicitly nondurable and may duplicate peer delivery.
        if (closed || terminal) return;
        // The service ends a join whose client sent nothing for three keepalive
        // intervals UNAVAILABLE with its ErrorInfo; the record names it with
        // reason 1, and the join rejoins as on any UNAVAILABLE.
        const bool keepaliveTimeout = call->finishOk && call->status.error_code() == grpc::StatusCode::UNAVAILABLE &&
            parseServiceErrorDetails(call->status.error_details()).clientKeepaliveTimeout;
        if (keepaliveTimeout)
            runtime.diagnostics->emit(OMNI_USD_LIVE_RECORD_STREAM_DISCONNECTED, endpointHandle, subject, 0,
                {{DiagnosticKey::Operation, OMNI_USD_LIVE_OPERATION_JOIN}, {DiagnosticKey::Result, result},
                    {DiagnosticKey::Session, handle}, {DiagnosticKey::Attempt, call->generation},
                    {DiagnosticKey::Generation, call->generation},
                    {DiagnosticKey::AuthGeneration, call->authGeneration}, {DiagnosticKey::Reason, 1}});
        else
            runtime.diagnostics->emit(OMNI_USD_LIVE_RECORD_STREAM_DISCONNECTED, endpointHandle, subject, 0,
                {{DiagnosticKey::Operation, OMNI_USD_LIVE_OPERATION_JOIN}, {DiagnosticKey::Result, result},
                    {DiagnosticKey::Session, handle}, {DiagnosticKey::Attempt, call->generation},
                    {DiagnosticKey::Generation, call->generation},
                    {DiagnosticKey::AuthGeneration, call->authGeneration}});
        if (result == OMNI_USD_LIVE_UNAUTHENTICATED) {
            std::lock_guard<std::mutex> lock(mutex_);
            authWait_ = true;
            authFailure_ = call->authFailure;
            authDeadline_ = Clock::now() + std::chrono::milliseconds(endpoint_->options.authWaitMs);
        } else if (result == OMNI_USD_LIVE_UNAVAILABLE || result == OMNI_USD_LIVE_TIMEOUT ||
            result == OMNI_USD_LIVE_CANCELLED) {
            // Remote HTTP/2 CANCEL ends an attempt. Explicit Leave and local
            // terminal cancellation are excluded above. Message replay remains
            // nondurable and may duplicate delivery, as for other reconnects.
            // attempts_ counts the joins started since the last validated
            // Connection; the next one is attempt attempts_ + 1.
            if (attempts_ >= runtime.options.values.reconnect_max_attempts) {
                // Exhausted attempts fail the session UNAVAILABLE, as they
                // fail a layer; the last attempt's own result stays in its
                // stream_disconnected record.
                fail(OMNI_USD_LIVE_UNAVAILABLE, true);
            } else {
                const auto delay = reconnectWait(runtime, handle, attempts_ + 1);
                retryAt_ = Clock::now() + std::chrono::milliseconds(delay);
                runtime.diagnostics->emit(OMNI_USD_LIVE_RECORD_RETRY_SCHEDULED, endpointHandle, subject, 0,
                    {{DiagnosticKey::Operation, OMNI_USD_LIVE_OPERATION_JOIN}, {DiagnosticKey::Result, result},
                        {DiagnosticKey::Session, handle}, {DiagnosticKey::Attempt, call->generation},
                        {DiagnosticKey::Generation, call->generation},
                        {DiagnosticKey::AuthGeneration, call->authGeneration}, {DiagnosticKey::Reason, result},
                        {DiagnosticKey::DelayMs, delay}});
            }
        } else fail(result, true);
        bool failed;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            failed = failure_ != OMNI_USD_LIVE_OK;
        }
        if (!failed) deliver(OMNI_USD_LIVE_EVENT_CONNECTION, result, connection_);
    }
    Clock::time_point step() {
        const auto now = Clock::now();
        auto next = Clock::time_point::max();
        std::shared_ptr<Attempt> call;
        bool stopping;
        bool authFailed;
        {
            std::lock_guard<std::mutex> lock(endpoint_->mutex);
            authFailed = endpoint_->authFailed || endpoint_->authFailureSerial != authFailure_;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (authFailed && !closing_ && failure_ == OMNI_USD_LIVE_OK) {
                failure_ = OMNI_USD_LIVE_UNAUTHENTICATED;
                grpcPresent_ = false;
                grpcCode_ = 0;
            }
            if (!opened_ && !closing_ && failure_ == OMNI_USD_LIVE_OK && now >= openDeadline_) {
                failure_ = expiredWait(now, openDeadline_, authWait_);
                grpcPresent_ = false;
                grpcCode_ = 0;
            }
            stopping = closing_ || failure_ != OMNI_USD_LIVE_OK;
            call = call_;
            if (!opened_ && !stopping) next = openDeadline_;
        }
        if (call) {
            unsigned pending, ok;
            Clock::time_point written;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                pending = call->pending;
                ok = call->successful;
                written = call->written;
                call->pending = call->successful = 0;
            }
            if (pending & Start) {
                call->startDone = true;
                if (!(ok & Start)) call->broken = true;
            }
            if (pending & Write) {
                call->writing = false;
                // Every completed write, keepalives included, restarts the keepalive interval.
                if (ok & Write) call->keepalive.writeCompleted(written);
                if (!(ok & Write)) call->broken = true;
                else if (!call->headerWritten) call->headerWritten = true;
                else if (call->active) {
                    std::shared_ptr<Message> retired;
                    {
                        std::lock_guard<std::mutex> lock(mutex_);
                        assert(count_ && messages_[head_] == call->active);
                        retired = std::move(messages_[head_]);
                        head_ = (head_ + 1) % messages_.size();
                        --count_;
                        pendingBytes_ -= retired->size;
                        messageRecordLocked(OMNI_USD_LIVE_RECORD_MESSAGE_WRITE_COMPLETED, OMNI_USD_LIVE_OK,
                            retired->size, retired->request, call.get());
                    }
                }
                call->active.reset();
            }
            if (pending & Read) {
                call->reading = false;
                if (!(ok & Read)) call->broken = true;
                else if (!stopping) consume(call);
            }
            if (pending & Finish) {
                call->finished = true;
                call->finishOk = (ok & Finish) != 0;
            }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                stopping = closing_ || failure_ != OMNI_USD_LIVE_OK;
            }
            // Remote EOF/failed Write already end the transport. Cancellation
            // here could replace the service's permission/auth status with
            // CANCELLED before Finish observes it. Local stop alone cancels.
            if (stopping || call->broken) peers_.disconnected(call->generation);
            if (stopping) call->context.TryCancel();
            if (call->finished) {
                size_t tags;
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    tags = tags_;
                }
                if (!tags) {
                    settle(call);
                    call.reset();
                }
            } else if (call->startDone && !call->finishing) {
                if (stopping || call->broken) {
                    if (!call->reading && !call->writing) finishTransport(call);
                } else {
                    if (!call->reading) read(call);
                    if (!call->writing) write(call, now);
                    // An idle connection wakes when its next keepalive is due; a write in flight wakes it on completion.
                    if (!call->writing && call->connected) next = std::min(next, call->keepalive.deadline());
                }
            }
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping = closing_ || failure_ != OMNI_USD_LIVE_OK;
        }
        if (stopping) {
            if (!call) {
                notifyFailure();
                // Message payloads and peer metadata are destroyed only on this
                // worker. No user callbacks can start after the close marker.
                for (;;) {
                    std::shared_ptr<Message> retired;
                    {
                        std::lock_guard<std::mutex> lock(mutex_);
                        if (!count_) break;
                        retired = std::move(messages_[head_]);
                        head_ = (head_ + 1) % messages_.size();
                        --count_;
                        pendingBytes_ -= retired->size;
                        messageRecordLocked(OMNI_USD_LIVE_RECORD_MESSAGE_DISCARDED,
                            failure_ == OMNI_USD_LIVE_OK ? OmniUsdLiveResult(OMNI_USD_LIVE_CANCELLED) : failure_,
                            retired->size, retired->request);
                    }
                }
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    callback_ = nullptr;
                    stopped_ = true;
                    publishLocked();
                }
            }
            return next;
        }
        if (call && call->connected && !call->broken && !call->finishing) {
            deliverPeers(peers_.expire(call->generation,
                expirePeers_.load(std::memory_order_acquire) ? Clock::time_point::max() : now));
            next = std::min(next, peers_.nextDeadline());
        }
        if (!call) {
            if (!authenticate(now, next)) {
                std::lock_guard<std::mutex> lock(mutex_);
                return failure_ != OMNI_USD_LIVE_OK ? now : next;
            }
            if (now < retryAt_) next = std::min(next, retryAt_);
            else startAttempt();
        }
        return next;
    }
    void dispatch() noexcept override {
        auto next = Clock::time_point::max();
        try {
            next = step();
        } catch (OmniUsdLiveResult result) {
            fail(result);
            next = Clock::now();
        } catch (const std::bad_alloc&) {
            fail(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
            next = Clock::now();
        } catch (const std::length_error&) {
            fail(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
            next = Clock::now();
        } catch (...) {
            fail(OMNI_USD_LIVE_INTERNAL_ERROR);
            next = Clock::now();
        }
        notifyFailure();
        publish();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (failure_ != OMNI_USD_LIVE_OK) authObservation_.finish(failure_, Clock::now());
            else if (closing_) authObservation_.finish(OMNI_USD_LIVE_CANCELLED, Clock::now());
            scheduled_ = false;
            runtime.auth->schedule(*this, dirty_ || stopped_ ? Clock::now() : next, observedWake_);
            closed_.notify_all();
        }
    }
};
}

struct SessionRegistry::Impl {
    Runtime& runtime;
    mutable std::mutex mutex;
    std::unordered_map<uint64_t, std::shared_ptr<Session>> sessions;
    bool stopping = false;
    void (*afterPrepare)(void*, uint64_t) noexcept = nullptr; // Under mutex.
    void* afterPrepareContext = nullptr;
    explicit Impl(Runtime& r) : runtime(r) {}
    std::shared_ptr<Session> find(uint64_t id) {
        std::lock_guard<std::mutex> lock(mutex);
        const auto found = sessions.find(id);
        return found == sessions.end() ? nullptr : found->second;
    }
};
SessionRegistry::SessionRegistry(Runtime& runtime) : impl_(std::make_unique<Impl>(runtime)) {}
SessionRegistry::~SessionRegistry() {
    beginShutdown();
    (void)waitForShutdown(Clock::time_point::max());
}
OmniUsdLiveSession SessionRegistry::join(const LiveUri& uri, std::string app, void* user,
    OmniUsdLiveSessionCallback callback) {
    if (!callback || !uri.layerName.empty()) throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
    try {
        SessionPeers::validateText(app, false, kMaxUriBytes);
    } catch (OmniUsdLiveResult result) {
        throw OmniUsdLiveResult(
            result == OMNI_USD_LIVE_MALFORMED_STREAM ? OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT) : result);
    }
    uint64_t endpoint;
    {
        std::lock_guard<std::mutex> lock(impl_->runtime.mutex);
        if (impl_->runtime.closing) throw OmniUsdLiveResult(OMNI_USD_LIVE_BUSY);
        const auto found = impl_->runtime.authorities.find(uri.authority.canonical);
        if (found == impl_->runtime.authorities.end()) throw OmniUsdLiveResult(OMNI_USD_LIVE_NOT_FOUND);
        endpoint = found->second;
    }
    auto lease = impl_->runtime.acquireEndpoint(endpoint);
    if (!lease) throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_HANDLE);
    const auto id = allocateSession();
    void (*afterPrepare)(void*, uint64_t) noexcept;
    void* afterPrepareContext;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        afterPrepare = impl_->afterPrepare;
        afterPrepareContext = impl_->afterPrepareContext;
    }
    auto session = std::make_shared<Session>(impl_->runtime, std::move(lease), id, uri, std::move(app), user, callback,
        afterPrepare, afterPrepareContext);
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (impl_->stopping) throw OmniUsdLiveResult(OMNI_USD_LIVE_BUSY);
        if (impl_->sessions.size() >= impl_->runtime.options.values.max_status_subjects)
            throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
        impl_->sessions.emplace(id, session);
        try {
            session->activateStatus();
            impl_->runtime.auth->reserve(session);
        } catch (...) {
            impl_->sessions.erase(id);
            throw;
        }
        impl_->runtime.auth->activate(*session);
    }
    return id;
}
OmniUsdLiveResult SessionRegistry::leave(OmniUsdLiveSession id, uint32_t timeout) {
    auto session = impl_->find(id);
    if (!session) return OMNI_USD_LIVE_INVALID_HANDLE;
    const auto result = session->close(timeout);
    if (result != OMNI_USD_LIVE_OK) return result;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        const auto found = impl_->sessions.find(id);
        if (found == impl_->sessions.end() || found->second != session) return OMNI_USD_LIVE_INVALID_HANDLE;
        impl_->sessions.erase(found);
    }
    session->detach();
    return OMNI_USD_LIVE_OK;
}
OmniUsdLiveResult SessionRegistry::status(OmniUsdLiveSession id, OmniUsdLiveStatus* out) {
    auto session = impl_->find(id);
    return session ? session->snapshot(out) : OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_HANDLE);
}
std::shared_ptr<Request> SessionRegistry::send(OmniUsdLiveSession id, const void* bytes, uint64_t size, void* user,
    OmniUsdLiveResultCallback callback) {
    auto session = impl_->find(id);
    if (!session) throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_HANDLE);
    return session->send(bytes, size, user, callback);
}
void SessionRegistry::expirePeersForTest(OmniUsdLiveSession id) noexcept {
    if (auto session = impl_->find(id)) session->expirePeersForTest();
}
void SessionRegistry::expireKeepaliveForTest(OmniUsdLiveSession id) noexcept {
    if (auto session = impl_->find(id)) session->expireKeepaliveForTest();
}
void SessionRegistry::setAfterPrepareForTest(void (*hook)(void*, uint64_t) noexcept, void* context) noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->afterPrepare = hook;
    impl_->afterPrepareContext = context;
}
bool SessionRegistry::hasSessions() const noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return !impl_->sessions.empty();
}
void SessionRegistry::beginShutdown() noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->stopping = true;
    for (const auto& entry : impl_->sessions) (void)entry.second->close(0);
}
bool SessionRegistry::waitForShutdown(Clock::time_point deadline) {
    for (;;) {
        uint64_t id;
        {
            std::lock_guard<std::mutex> lock(impl_->mutex);
            if (impl_->sessions.empty()) return true;
            id = impl_->sessions.begin()->first;
        }
        const auto now = Clock::now();
        const auto remaining =
            deadline <= now ? 0 : std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
        const auto wait = static_cast<uint32_t>(std::min<uint64_t>(remaining, UINT32_MAX));
        if (leave(id, wait) != OMNI_USD_LIVE_OK) return false;
    }
}
} // namespace omni::usd::live
