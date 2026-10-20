// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "LiveDeadline.h"
#include "LiveLayerWrite.h"
#include "LiveAuth.h"
#include "LiveDiagnostics.h"
#include <algorithm>
#include <array>
#include <initializer_list>
#include <random>
#include <stdexcept>
#include <type_traits>

namespace omni::usd::live {
namespace {
template<class Input, class Response> struct Payload {
    // Declared first so reservation is released after buffers and host captures.
    // A caller-supplied reservation may be shared with its other owners.
    std::shared_ptr<PendingBytes> budget;
    std::vector<Input> messages;
    Response response;
    OmniUsdLiveRequestError error{};
    uint64_t generation = 0;
    LayerWriteDiagnostics diagnostics;
    UnaryCallback<Response> callback;
    explicit Payload(std::shared_ptr<PendingBytes> bytes) : budget(std::move(bytes)) {}
};

template<class Input, class Response> struct Call : std::enable_shared_from_this<Call<Input, Response>> {
    enum class Phase { Start, Write, HalfClose, Finish };
    struct Tag final : CompletionTag {
        std::shared_ptr<Call> owner; // Assigned only when handed to the CQ.
        Phase phase;
        explicit Tag(Phase value) : phase(value) {}
        void onComplete(bool ok, Clock::time_point received) noexcept override { owner->complete(phase, ok, received); }
    };
    grpc::ClientContext context;
    grpc::Status status;
    std::unique_ptr<grpc::ClientAsyncWriterInterface<Input>> writer;
    std::shared_ptr<Request> request;
    std::shared_ptr<Payload<Input, Response>> payload;
    std::function<void()> delivery;
    AuthRegistry* auth = nullptr;
    DiagnosticsRegistry* diagnostics = nullptr;
    std::chrono::steady_clock::time_point started;
    std::shared_ptr<LayerWriteReceipt> receipt;
    std::vector<std::unique_ptr<Tag>> writes;
    std::unique_ptr<Tag> start, halfClose, finish;
    size_t nextMessage = 0;
    bool uploadComplete = false;
    bool identified = false; // The header carries an operation ID.
    bool chained = false;    // The request sends a chained header.

    Tag* handoff(std::unique_ptr<Tag>& tag) noexcept {
        tag->owner = this->shared_from_this();
        return submitTag(tag, *diagnostics);
    }
    void complete(Phase phase, bool ok, std::chrono::steady_clock::time_point received) noexcept {
        if (phase == Phase::Finish) {
            const auto completed = received;
            auto& error = payload->error;
            error.operation = request->operation;
            error.request_id = request->handle;
            error.attempt_id = 1;
            error.grpc_status_present = ok;
            error.grpc_status = ok ? status.error_code() : 0;
            error.result = ok ? classifyUnary(status.error_code(), true) :
                                OmniUsdLiveResult(OMNI_USD_LIVE_OUTCOME_UNKNOWN);
            if (error.result == OMNI_USD_LIVE_OK) {
                bool valid;
                if constexpr (std::is_same_v<Response, rpc::CreateLayerResponse>)
                    valid = payload->response.object_id() != 0;
                else valid = payload->response.sequence_num() > 1;
                if (!valid || !uploadComplete) error.result = OMNI_USD_LIVE_OUTCOME_UNKNOWN;
            }
            // The service's entry-limit refusal names its counts in an ErrorInfo
            // detail, which only the rpc_failed record carries.
            ServiceLimitRefusal refusal;
            if (ok && status.error_code() == grpc::StatusCode::INVALID_ARGUMENT)
                refusal = parseServiceLimitRefusal(status.error_details());
            // A chained update's refusal names the writer's last committed
            // operation, which locates the chain (design section 13).
            if (ok && receipt && status.error_code() == grpc::StatusCode::ABORTED &&
                parsePredecessorRefusal(status.error_details(), receipt->lastOperation, receipt->lastRecord))
                receipt->chainRefused.store(true, std::memory_order_release);
            // A replica that does not know the chained header refuses it
            // INVALID_ARGUMENT, with no ErrorInfo of the service's domain,
            // before reading any chunk: the attempt applied nothing.
            if (ok && receipt && chained && refusedChainedHeader(status.error_code(), status.error_details()))
                receipt->chainUnsupported.store(true, std::memory_order_release);
            // Only such a refusal or one with the service's ErrorInfo settles
            // a chained call; any other ending may still commit later.
            if (ok && receipt && chained && settlesChainedCall(status.error_code(), status.error_details()))
                receipt->settledRefusal.store(true, std::memory_order_release);
            // An update or creation the service refused with nothing applied
            // is a definitive UNAVAILABLE, not an unknown outcome; its receipt
            // carries the service's retry delay.
            if (ok && status.error_code() == grpc::StatusCode::UNAVAILABLE) {
                const auto details = parseServiceErrorDetails(status.error_details());
                if (refusalAppliedNothing(status.error_code(), details)) {
                    error.result = OMNI_USD_LIVE_UNAVAILABLE;
                    if (receipt) receipt->retryDelayMs.store(details.retryDelayMs, std::memory_order_release);
                }
            }
            const auto outcome = payload->response.operation_outcome();
            // An upgraded replica reports an outcome on every OK to a request
            // with an operation ID; UNSPECIFIED marks one that ignored it.
            if (error.result == OMNI_USD_LIVE_OK && identified)
                payload->budget->endpoint().operationIdsIgnored.store(!acknowledgesResend(uint32_t(outcome)),
                    std::memory_order_release);
            if (receipt && error.result == OMNI_USD_LIVE_OK) {
                receipt->outcome.store(uint32_t(outcome), std::memory_order_relaxed);
                if constexpr (std::is_same_v<Response, rpc::CreateLayerResponse>) {
                    receipt->object.store(payload->response.object_id(), std::memory_order_release);
                    receipt->sequence.store(1, std::memory_order_release);
                } else receipt->sequence.store(payload->response.sequence_num(), std::memory_order_release);
            }
            diagnostics->observeRpc(payload->budget->endpoint().handle, error.operation, error.result,
                static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(completed - started).count()),
                error.request_id, 1, error.grpc_status_present, error.grpc_status, payload->diagnostics.subjectId,
                payload->diagnostics.streamGeneration, payload->generation, payload->diagnostics.workId, 0, &refusal);
            if (ok) auth->completed(payload->budget->endpoint(), payload->generation, status.error_code());
            retire(error);
        } else if (!ok || phase == Phase::HalfClose) {
            uploadComplete = phase == Phase::HalfClose && ok;
            // A failed Start/Write may represent auth rejection, cancellation,
            // or an early server failure. Only Finish supplies the raw status.
            writer->Finish(&status, handoff(finish));
        } else if (nextMessage < payload->messages.size()) {
            const auto index = nextMessage++;
            // Loaded before handoff: its release is the last access before completion.
            const auto& message = payload->messages[index];
            writer->Write(message, handoff(writes[index]));
        } else {
            writer->WritesDone(handoff(halfClose));
        }
    }
    void retire(OmniUsdLiveRequestError error) noexcept {
        error.struct_size = sizeof(error);
        error.abi_version = OMNI_USD_LIVE_ABI_VERSION;
        payload->error = error;
        // Last payload/capture owner goes to the callback worker before enqueue.
        // Request quiescence then also proves charged wire buffers were retired.
        payload.reset();
        request->finish(error, std::move(delivery));
    }
};

// The update header a request header carries: a chained header wraps it.
template<class Header> const auto& baseHeader(const Header& header) {
    if constexpr (std::is_same_v<Header, rpc::ChainedUpdateLayerHeader>) return header.header();
    else return header;
}

template<class Input, class Response, class Header>
std::shared_ptr<Request> submit(Runtime& runtime, EndpointLease lease, Header header, std::string_view bytes,
    UnaryCallback<Response> callback, uint32_t timeoutMs, uint32_t operation, LayerWriteTransport* transport,
    std::shared_ptr<LayerWriteReceipt> receipt, LayerWriteDiagnostics diagnostics,
    std::shared_ptr<PendingBytes> reservation = nullptr) {
    if (bytes.size() > runtime.options.values.max_update_bytes)
        throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    // Validate identity with the same private URI codec used by the layer owner.
    const auto& base = baseHeader(header);
    makeLayerUri(makeSessionUri(lease->options.authority.canonical, base.session_id()), base.layer_name());
    const size_t chunks = bytes.size() / kLayerChunkBytes + (bytes.size() % kLayerChunkBytes != 0);
    const uint64_t charged = layerUpdateCharge(header.ByteSizeLong(), bytes.size());
    if (reservation && (&reservation->endpoint() != &*lease || reservation->amount() < charged))
        throw std::invalid_argument("reservation does not cover the update");
    auto payload = std::make_shared<Payload<Input, Response>>(reservation ?
            std::move(reservation) :
            std::make_shared<PendingBytes>(std::move(lease), charged,
                runtime.options.values.max_pending_bytes_per_endpoint));
    auto call = std::make_shared<Call<Input, Response>>();
    call->identified = !base.operation_id().empty();
    call->chained = std::is_same_v<Header, rpc::ChainedUpdateLayerHeader>;
    payload->diagnostics = diagnostics;
    call->auth = runtime.auth.get();
    call->diagnostics = runtime.diagnostics.get();
    call->receipt = std::move(receipt);
    {
        auto& endpoint = payload->budget->endpoint();
        std::lock_guard<std::mutex> lock(endpoint.mutex);
        runtime.auth->evaluate(endpoint, wallTimeMs(), std::chrono::steady_clock::now());
        if (!runtime.auth->usable(endpoint, wallTimeMs())) throw OmniUsdLiveResult(OMNI_USD_LIVE_UNAUTHENTICATED);
        call->context.AddMetadata("authorization", "Bearer " + endpoint.token);
        payload->generation = endpoint.tokenGeneration;
    }
    payload->callback = std::move(callback);
    payload->messages.resize(chunks + 1);
    // The pinned generated mutable_* oneof setter publishes its discriminator
    // before allocating the child. Construct/populate separately so allocation
    // failure cannot leave the parent tagged with an uninitialized pointer.
    // Both parent and child are heap messages (no arenas), making adoption a
    // nonallocating pointer transfer in the generated set_allocated_* path.
    auto ownedHeader = std::make_unique<Header>(std::move(header));
    if constexpr (std::is_same_v<Header, rpc::ChainedUpdateLayerHeader>)
        payload->messages.front().set_allocated_chained_header(ownedHeader.get());
    else payload->messages.front().set_allocated_header(ownedHeader.get());
    (void)ownedHeader.release();
    using Chunk = std::decay_t<decltype(payload->messages.front().chunk())>;
    for (size_t index = 0, offset = 0; index < chunks; ++index) {
        const auto count = std::min(kLayerChunkBytes, bytes.size() - offset);
        auto chunk = std::make_unique<Chunk>();
        chunk->set_data(bytes.data() + offset, count);
        payload->messages[index + 1].set_allocated_chunk(chunk.get());
        (void)chunk.release();
        offset += count;
    }
    call->payload = payload;
    call->delivery = [payload] {
        if (payload->callback) payload->callback(payload->error, payload->response, payload->generation);
    };
    using Phase = typename Call<Input, Response>::Phase;
    using Tag = typename Call<Input, Response>::Tag;
    call->start = std::make_unique<Tag>(Phase::Start);
    call->halfClose = std::make_unique<Tag>(Phase::HalfClose);
    call->finish = std::make_unique<Tag>(Phase::Finish);
    call->writes.reserve(chunks + 1);
    for (size_t index = 0; index <= chunks; ++index) call->writes.push_back(std::make_unique<Tag>(Phase::Write));
    call->context.set_deadline(
        rpcDeadlineAfter(std::chrono::milliseconds(timeoutMs ? timeoutMs : runtime.options.values.request_timeout_ms)));
    call->context.set_wait_for_ready(false);
    auto& endpoint = payload->budget->endpoint();
    if constexpr (std::is_same_v<Input, rpc::CreateLayerRequest>)
        call->writer = transport ?
            transport->create(&call->context, &payload->response, &runtime.queue()) :
            endpoint.stub->PrepareAsyncCreateLayer(&call->context, &payload->response, &runtime.queue());
    else
        call->writer = transport ?
            transport->update(&call->context, &payload->response, &runtime.queue()) :
            endpoint.stub->PrepareAsyncUpdateLayer(&call->context, &payload->response, &runtime.queue());
    if (!call->writer) throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    std::weak_ptr<Call<Input, Response>> weak = call;
    call->request = runtime.requests->create(endpoint.handle, operation, [weak] {
        if (auto active = weak.lock()) active->context.TryCancel();
    });
    // All owned buffers and CQ nodes are allocated before transport issuance.
    // Remove the submitter's payload owner before a fast completion can retire.
    payload.reset();
    if (!call->request->issueIfNotCancelled([&] {
        call->started = std::chrono::steady_clock::now();
        call->diagnostics->startedRpc(operation);
        call->writer->StartCall(call->handoff(call->start));
    })) {
        OmniUsdLiveRequestError error{};
        error.result = OMNI_USD_LIVE_CANCELLED;
        call->retire(error);
    }
    return call->request;
}
EndpointLease acquire(Runtime& runtime, OmniUsdLiveEndpoint endpoint) {
    auto lease = runtime.acquireEndpoint(endpoint);
    if (!lease) throw OmniUsdLiveResult(runtime.closing ? OMNI_USD_LIVE_BUSY : OMNI_USD_LIVE_INVALID_HANDLE);
    return lease;
}
} // namespace

std::shared_ptr<Request> submitLayerCreate(Runtime& runtime, OmniUsdLiveEndpoint endpoint, std::string_view sessionId,
    std::string_view layerName, std::string_view bytes, UnaryCallback<rpc::CreateLayerResponse> callback,
    uint32_t timeoutMs, LayerWriteTransport* transport, std::shared_ptr<LayerWriteReceipt> receipt,
    LayerWriteDiagnostics diagnostics, std::string_view operationId) {
    if (sessionId.size() > kMaxUriBytes || layerName.size() > kMaxUriBytes)
        throw std::length_error("layer identity capacity");
    if (operationId.size() > kMaxOperationIdBytes) throw std::length_error("operation identity capacity");
    auto lease = acquire(runtime, endpoint);
    rpc::CreateLayerHeader header;
    header.set_session_id(std::string(sessionId));
    header.set_layer_name(std::string(layerName));
    header.set_operation_id(std::string(operationId));
    return submit<rpc::CreateLayerRequest, rpc::CreateLayerResponse>(runtime, std::move(lease), std::move(header),
        bytes, std::move(callback), timeoutMs, OMNI_USD_LIVE_OPERATION_CREATE, transport, std::move(receipt),
        diagnostics);
}
std::shared_ptr<Request> submitLayerUpdate(Runtime& runtime, OmniUsdLiveEndpoint endpoint, std::string_view sessionId,
    std::string_view layerName, uint64_t objectId, uint64_t estimatedServerTimeUs, std::string_view bytes,
    UnaryCallback<rpc::UpdateLayerResponse> callback, uint32_t timeoutMs, LayerWriteTransport* transport,
    std::shared_ptr<LayerWriteReceipt> receipt, LayerWriteDiagnostics diagnostics,
    std::shared_ptr<PendingBytes> reservation, std::string_view operationId, const std::string* previousOperationId,
    bool noWait) {
    if (!objectId) throw std::invalid_argument("update requires an existing object incarnation");
    if (previousOperationId && (previousOperationId->size() > kMaxOperationIdBytes || operationId.empty()))
        throw std::invalid_argument("chained update identity");
    if (sessionId.size() > kMaxUriBytes || layerName.size() > kMaxUriBytes)
        throw std::length_error("layer identity capacity");
    if (operationId.size() > kMaxOperationIdBytes) throw std::length_error("operation identity capacity");
    auto lease = acquire(runtime, endpoint);
    rpc::UpdateLayerHeader header;
    header.set_session_id(std::string(sessionId));
    header.set_layer_name(std::string(layerName));
    header.set_object_id(objectId);
    header.set_timestamp_microseconds_since_epoch(estimatedServerTimeUs);
    header.set_client_id(lease->writerId); // Immutable endpoint-lifetime identity.
    header.set_operation_id(std::string(operationId));
    if (previousOperationId) {
        rpc::ChainedUpdateLayerHeader chained;
        *chained.mutable_header() = std::move(header);
        chained.set_previous_operation_id(*previousOperationId);
        chained.set_no_wait(noWait);
        return submit<rpc::UpdateLayerRequest, rpc::UpdateLayerResponse>(runtime, std::move(lease), std::move(chained),
            bytes, std::move(callback), timeoutMs, OMNI_USD_LIVE_OPERATION_UPDATE, transport, std::move(receipt),
            diagnostics, std::move(reservation));
    }
    return submit<rpc::UpdateLayerRequest, rpc::UpdateLayerResponse>(runtime, std::move(lease), std::move(header),
        bytes, std::move(callback), timeoutMs, OMNI_USD_LIVE_OPERATION_UPDATE, transport, std::move(receipt),
        diagnostics, std::move(reservation));
}
namespace {
// A bounded reader of the length-delimited fields of a protobuf message, the
// only wire type the google.rpc fields it reads have. Varint and fixed-width
// fields are skipped, as protobuf skips a field of another wire type than its
// declared one, and groups or anything truncated end the message as malformed.
class ProtoFields {
public:
    explicit ProtoFields(std::string_view bytes) noexcept : rest_(bytes) {}
    // The next length-delimited field, or false at the end or on malformed
    // input (then failed()).
    bool next(uint32_t& number, std::string_view& bytes) noexcept {
        uint64_t value = 0;
        bool isVarint = false;
        return next(number, bytes, value, false, isVarint);
    }
    // The next varint field, with its value; length-delimited fields are
    // skipped as next skips varints.
    bool nextVarint(uint32_t& number, uint64_t& value) noexcept {
        std::string_view bytes;
        bool isVarint = false;
        while (next(number, bytes, value, true, isVarint))
            if (isVarint) return true;
        return false;
    }
    bool failed() const noexcept { return failed_; }
private:
    // The next length-delimited field in `bytes`, or with `varints` also the
    // next varint field in `value`; `isVarint` tells them apart.
    bool next(uint32_t& number, std::string_view& bytes, uint64_t& value, bool varints, bool& isVarint) noexcept {
        while (!rest_.empty()) {
            uint64_t key = 0;
            if (!varint(key) || (key >> 3) == 0 || (key >> 3) > UINT32_MAX) return fail();
            switch (key & 7) {
            case 0:
                if (!varint(value)) return fail();
                if (varints) {
                    number = uint32_t(key >> 3);
                    isVarint = true;
                    return true;
                }
                continue;
            case 1:
                if (!skip(8)) return fail();
                continue;
            case 2: {
                uint64_t size = 0;
                if (!varint(size) || size > rest_.size()) return fail();
                number = uint32_t(key >> 3);
                bytes = rest_.substr(0, size_t(size));
                rest_.remove_prefix(size_t(size));
                isVarint = false;
                return true;
            }
            case 5:
                if (!skip(4)) return fail();
                continue;
            default: return fail();
            }
        }
        return false;
    }
    bool varint(uint64_t& value) noexcept {
        value = 0;
        for (unsigned shift = 0; shift < 64 && !rest_.empty(); shift += 7) {
            const auto byte = uint8_t(rest_.front());
            rest_.remove_prefix(1);
            value |= uint64_t(byte & 0x7f) << shift;
            if (!(byte & 0x80)) return true;
        }
        return false;
    }
    bool skip(size_t size) noexcept {
        if (size > rest_.size()) return false;
        rest_.remove_prefix(size);
        return true;
    }
    bool fail() noexcept {
        failed_ = true;
        return false;
    }
    std::string_view rest_;
    bool failed_ = false;
};
// A decimal uint64 of 1 to 20 digits, without sign or overflow.
bool decimal(std::string_view text, uint64_t& value) noexcept {
    if (text.empty() || text.size() > 20) return false;
    value = 0;
    for (const char c : text) {
        if (c < '0' || c > '9') return false;
        const uint64_t digit = uint64_t(c - '0');
        if (value > (UINT64_MAX - digit) / 10) return false;
        value = value * 10 + digit;
    }
    return true;
}
// One google.rpc.ErrorInfo: reason 1, domain 2, metadata map entries 3.
ServiceLimitRefusal errorInfoRefusal(std::string_view info) noexcept {
    std::string_view reason, domain;
    std::array<std::pair<std::string_view, std::string_view>, 6> wanted{{{"struct_commands", {}}, {"set_fields", {}},
        {"time_samples", {}}, {"max_entries", {}}, {"limit", {}}, {"actual", {}}}};
    std::array<bool, 6> present{};
    ProtoFields fields(info);
    uint32_t number = 0;
    std::string_view bytes;
    while (fields.next(number, bytes)) {
        if (number == 1) reason = bytes;
        else if (number == 2) domain = bytes;
        else if (number == 3) {
            std::string_view key, text;
            ProtoFields entry(bytes);
            uint32_t part = 0;
            std::string_view partBytes;
            while (entry.next(part, partBytes)) {
                if (part == 1) key = partBytes;
                else if (part == 2) text = partBytes;
            }
            if (entry.failed()) return {};
            for (size_t i = 0; i < wanted.size(); ++i)
                if (key == wanted[i].first) {
                    wanted[i].second = text;
                    present[i] = true;
                }
        }
    }
    if (fields.failed() || domain != "nvidia.omniverse.live.v1alpha") return {};
    std::array<uint64_t, 6> counts{};
    const auto counted = [&](std::initializer_list<size_t> keys) {
        for (const auto key : keys)
            if (!present[key] || !decimal(wanted[key].second, counts[key])) return false;
        return true;
    };
    ServiceLimitRefusal result;
    if (reason == "DELTA_TOO_LARGE" && counted({0, 1, 2, 3})) {
        result.structCommands = counts[0];
        result.setFields = counts[1];
        result.timeSamples = counts[2];
        result.limit = counts[3];
        const auto sum = counts[0] + counts[1];
        if (sum < counts[0] || sum + counts[2] < sum) return {};
        result.actual = sum + counts[2];
        result.reason = 1;
    } else if (reason == "CHILDREN_LIST_TOO_LONG" && counted({4, 5})) {
        result.limit = counts[4];
        result.actual = counts[5];
        result.reason = 2;
    }
    return result;
}
} // namespace

bool parsePredecessorRefusal(std::string_view statusDetails, std::string& lastOperation, LastRecord& lastRecord) {
    // google.rpc.Status: details 3, each a google.protobuf.Any with type_url 1
    // and value 2; ErrorInfo: reason 1, domain 2, metadata map entries 3.
    ProtoFields status(statusDetails);
    uint32_t number = 0;
    std::string_view bytes;
    while (status.next(number, bytes)) {
        if (number != 3) continue;
        std::string_view type, payload;
        ProtoFields any(bytes);
        uint32_t part = 0;
        std::string_view partBytes;
        while (any.next(part, partBytes)) {
            if (part == 1) type = partBytes;
            else if (part == 2) payload = partBytes;
        }
        constexpr std::string_view errorInfo = "/google.rpc.ErrorInfo";
        if (any.failed()) return false;
        if (type.size() <= errorInfo.size() || type.substr(type.size() - errorInfo.size()) != errorInfo) continue;
        std::string_view reason, domain, last, record;
        bool recorded = false;
        ProtoFields info(payload);
        while (info.next(part, partBytes)) {
            if (part == 1) reason = partBytes;
            else if (part == 2) domain = partBytes;
            else if (part == 3) {
                std::string_view key, text;
                ProtoFields entry(partBytes);
                uint32_t field = 0;
                std::string_view fieldBytes;
                while (entry.next(field, fieldBytes)) {
                    if (field == 1) key = fieldBytes;
                    else if (field == 2) text = fieldBytes;
                }
                if (entry.failed()) return false;
                if (key == "last_operation_id") last = text;
                else if (key == "last_record") {
                    record = text;
                    recorded = true;
                }
            }
        }
        if (info.failed() || domain != "nvidia.omniverse.live.v1alpha" || reason != "PREDECESSOR_NOT_COMMITTED")
            continue;
        LastRecord kind;
        if (!recorded) return false;
        if (record == "operation") kind = LastRecord::Operation;
        else if (record == "absent") kind = LastRecord::Absent;
        else if (record == "no_operation_id") kind = LastRecord::NoOperationId;
        else return false;
        if ((kind == LastRecord::Operation) == last.empty() || last.size() > kMaxOperationIdBytes || status.failed())
            return false;
        // Every later byte must still be a well-formed status.
        while (status.next(number, bytes)) {}
        if (status.failed()) return false;
        lastOperation.assign(last);
        lastRecord = kind;
        return true;
    }
    return false;
}

ServiceLimitRefusal parseServiceLimitRefusal(std::string_view statusDetails) noexcept {
    // google.rpc.Status: details 3, each a google.protobuf.Any with type_url 1 and value 2.
    ProtoFields status(statusDetails);
    uint32_t number = 0;
    std::string_view bytes;
    ServiceLimitRefusal found;
    while (status.next(number, bytes)) {
        if (number != 3) continue;
        std::string_view type, payload;
        ProtoFields any(bytes);
        uint32_t part = 0;
        std::string_view partBytes;
        while (any.next(part, partBytes)) {
            if (part == 1) type = partBytes;
            else if (part == 2) payload = partBytes;
        }
        constexpr std::string_view errorInfo = "/google.rpc.ErrorInfo";
        if (any.failed()) return {};
        if (type.size() > errorInfo.size() && type.substr(type.size() - errorInfo.size()) == errorInfo && !found.reason)
            found = errorInfoRefusal(payload);
    }
    return status.failed() ? ServiceLimitRefusal{} : found;
}
namespace {
// Whether `type` is a google.protobuf.Any type URL of the message `name`.
bool anyType(std::string_view type, std::string_view name) noexcept {
    return type.size() > name.size() && type.substr(type.size() - name.size()) == name &&
        type[type.size() - name.size() - 1] == '/';
}
// Whether `info`, one google.rpc.ErrorInfo (reason 1, domain 2, metadata map
// entries 3), is well formed with the service's domain and, with
// `requireReason`, a reason; when it is, `reasonOut` holds its reason.
bool serviceErrorInfo(std::string_view info, std::string_view& reasonOut, bool requireReason = true) noexcept {
    std::string_view reason, domain;
    ProtoFields fields(info);
    uint32_t number = 0;
    std::string_view bytes;
    while (fields.next(number, bytes)) {
        if (number == 1) reason = bytes;
        else if (number == 2) domain = bytes;
        else if (number == 3) {
            ProtoFields entry(bytes);
            uint32_t part = 0;
            std::string_view partBytes;
            while (entry.next(part, partBytes)) {}
            if (entry.failed()) return false;
        }
    }
    if (fields.failed() || (requireReason && reason.empty()) || domain != "nvidia.omniverse.live.v1alpha") return false;
    reasonOut = reason;
    return true;
}
// The retry_delay of one google.rpc.RetryInfo (field 1), a
// google.protobuf.Duration (seconds 1, nanos 2), in milliseconds rounded up;
// kNoRetryDelay when it is absent, malformed, negative or outside the
// Duration range.
uint64_t retryInfoDelayMs(std::string_view info) noexcept {
    constexpr int64_t maxSeconds = INT64_C(315576000000);
    ProtoFields fields(info);
    uint32_t number = 0;
    std::string_view bytes, delay;
    bool present = false;
    while (fields.next(number, bytes))
        if (number == 1) {
            delay = bytes;
            present = true;
        }
    if (fields.failed() || !present) return kNoRetryDelay;
    // The last occurrence of a scalar field wins, as protobuf parses it.
    int64_t seconds = 0;
    int32_t nanos = 0;
    ProtoFields duration(delay);
    uint64_t value = 0;
    while (duration.nextVarint(number, value)) {
        if (number == 1) seconds = int64_t(value);
        else if (number == 2) nanos = int32_t(uint32_t(value));
    }
    if (duration.failed() || seconds < 0 || seconds > maxSeconds || nanos < 0 || nanos > 999999999)
        return kNoRetryDelay;
    // At most 315,576,000,001,000 ms, far below kNoRetryDelay.
    return uint64_t(seconds) * 1000 + (uint64_t(nanos) + 999999) / 1000000;
}
} // namespace

ServiceErrorDetails parseServiceErrorDetails(std::string_view statusDetails) noexcept {
    // google.rpc.Status: details 3, each a google.protobuf.Any with type_url 1 and value 2.
    ProtoFields status(statusDetails);
    uint32_t number = 0;
    std::string_view bytes;
    ServiceErrorDetails found;
    while (status.next(number, bytes)) {
        if (number != 3) continue;
        std::string_view type, payload;
        ProtoFields any(bytes);
        uint32_t part = 0;
        std::string_view partBytes;
        while (any.next(part, partBytes)) {
            if (part == 1) type = partBytes;
            else if (part == 2) payload = partBytes;
        }
        if (any.failed()) return {};
        std::string_view reason;
        if (anyType(type, "google.rpc.ErrorInfo")) {
            if (serviceErrorInfo(payload, reason)) {
                found.errorInfo = true;
                found.clientKeepaliveTimeout = found.clientKeepaliveTimeout || reason == "CLIENT_KEEPALIVE_TIMEOUT";
            }
        } else if (anyType(type, "google.rpc.RetryInfo") && found.retryDelayMs == kNoRetryDelay)
            found.retryDelayMs = retryInfoDelayMs(payload);
    }
    return status.failed() ? ServiceErrorDetails{} : found;
}
namespace {
// Whether `statusDetails`, a serialized google.rpc.Status, holds a
// well-formed google.rpc.ErrorInfo of the service's domain, with or without a
// reason, before any malformed detail.
bool carriesServiceErrorInfo(std::string_view statusDetails) noexcept {
    // google.rpc.Status: details 3, each a google.protobuf.Any with type_url 1 and value 2.
    ProtoFields status(statusDetails);
    uint32_t number = 0;
    std::string_view bytes;
    while (status.next(number, bytes)) {
        if (number != 3) continue;
        std::string_view type, payload;
        ProtoFields any(bytes);
        uint32_t part = 0;
        std::string_view partBytes;
        while (any.next(part, partBytes)) {
            if (part == 1) type = partBytes;
            else if (part == 2) payload = partBytes;
        }
        if (any.failed()) return false;
        std::string_view reason;
        if (anyType(type, "google.rpc.ErrorInfo") && serviceErrorInfo(payload, reason, false)) return true;
    }
    return false;
}
} // namespace

bool refusedChainedHeader(grpc::StatusCode code, std::string_view statusDetails) noexcept {
    return code == grpc::StatusCode::INVALID_ARGUMENT && !carriesServiceErrorInfo(statusDetails);
}
bool settlesChainedCall(grpc::StatusCode code, std::string_view statusDetails) noexcept {
    return code != grpc::StatusCode::OK &&
        (code == grpc::StatusCode::INVALID_ARGUMENT || carriesServiceErrorInfo(statusDetails));
}
std::string newOperationId() {
    // A per-thread generator seeded from the system's random source: IDs need
    // to be distinct per writer and layer, not unpredictable.
    thread_local std::mt19937_64 random = [] {
        std::random_device device;
        std::seed_seq seed{device(), device(), device(), device(), device(), device(), device(), device()};
        return std::mt19937_64(seed);
    }();
    std::array<uint8_t, 16> bytes;
    for (size_t i = 0; i < bytes.size(); i += 8) {
        const auto bits = random();
        for (size_t j = 0; j < 8; ++j) bytes[i + j] = uint8_t(bits >> (8 * j));
    }
    bytes[6] = uint8_t((bytes[6] & 0x0f) | 0x40); // Version 4.
    bytes[8] = uint8_t((bytes[8] & 0x3f) | 0x80); // RFC 4122 variant.
    constexpr char digits[] = "0123456789abcdef";
    std::string id;
    id.reserve(36);
    for (size_t i = 0; i < bytes.size(); ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) id += '-';
        id += digits[bytes[i] >> 4];
        id += digits[bytes[i] & 15];
    }
    return id;
}
size_t layerUpdateHeaderBound(std::string_view sessionId, std::string_view layerName, std::string_view writerId) {
    rpc::ChainedUpdateLayerHeader chained;
    auto& header = *chained.mutable_header();
    header.set_session_id(std::string(sessionId));
    header.set_layer_name(std::string(layerName));
    header.set_client_id(std::string(writerId));
    header.set_operation_id(std::string(kMaxOperationIdBytes, '0'));
    chained.set_previous_operation_id(std::string(kMaxOperationIdBytes, '0'));
    chained.set_no_wait(true);
    // The widest varint encodings of any object ID and timestamp.
    header.set_object_id(UINT64_MAX);
    header.set_timestamp_microseconds_since_epoch(UINT64_MAX);
    return chained.ByteSizeLong();
}
// Payload plus a conservative wire-envelope allowance per message. This is
// wire storage accounting, not a claim about allocator/SDK resident bytes.
uint64_t layerUpdateCharge(size_t headerBytes, size_t bytes) noexcept {
    const size_t chunks = bytes / kLayerChunkBytes + (bytes % kLayerChunkBytes != 0);
    return uint64_t(bytes) + headerBytes + 64 * (chunks + 1);
}
} // namespace omni::usd::live
