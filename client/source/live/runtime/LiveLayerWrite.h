// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "LiveDiagnostics.h"
#include "LiveUnary.h"

namespace omni::usd::live {

// No retry delay: a refusal carried no valid google.rpc.RetryInfo.
constexpr uint64_t kNoRetryDelay = UINT64_MAX;
// What a chained update's refusal names as the writer's latest write to the
// object (its ErrorInfo metadata last_record): no record, a record holding an
// operation ID, or a latest write without an operation ID.
enum class LastRecord : uint8_t { Absent, Operation, NoOperationId };
// Internal evidence survives cancellation of ordinary callback delivery.
// object, sequence and outcome are written only for a validated successful
// complete upload, and retryDelayMs only for an update or creation refused
// with nothing applied, before Request is completed. This is not a host
// callback or permission to replay a mutation. outcome is the response's
// rpc::OperationOutcome: whether the request applied the write or matched an
// earlier commit of its operation ID. retryDelayMs is the refusal's RetryInfo
// delay, or kNoRetryDelay.
struct LayerWriteReceipt {
    std::atomic<uint64_t> object{0}, sequence{0};
    std::atomic<uint32_t> outcome{0};
    // A chained update's refusal: ABORTED with the service's ErrorInfo
    // PREDECESSOR_NOT_COMMITTED. lastRecord and lastOperation, what the
    // writer's latest write to the object left and its operation ID (empty
    // unless lastRecord is Operation), are written before chainRefused is
    // released.
    std::string lastOperation;
    LastRecord lastRecord = LastRecord::Absent;
    std::atomic<bool> chainRefused{false};
    // A chained update refused INVALID_ARGUMENT without an ErrorInfo of the
    // service's domain, by a replica that does not know the chained header;
    // it applied nothing.
    std::atomic<bool> chainUnsupported{false};
    // A chained update's refusal that settles the call, so it applied nothing
    // and never commits later: an error with an ErrorInfo of the service's
    // domain, or INVALID_ARGUMENT without one. Any other ending of a chained
    // call apart from OK may still commit later (design section 13).
    std::atomic<bool> settledRefusal{false};
    std::atomic<uint64_t> retryDelayMs{kNoRetryDelay};
};

// The protocol bound on an operation ID: 1-64 bytes of A-Z a-z 0-9 - . _ ~.
constexpr size_t kMaxOperationIdBytes = 64;
// A new random UUIDv4, the identity of one write and of its re-send.
std::string newOperationId();

// Whether a layer write or creation whose outcome is unknown is re-sent once
// on this endpoint: only when its Ping reports operation-ID support and no
// later reply showed a replica ignoring an operation ID (design section 13).
// A server without it applies a re-send as a new write.
inline bool resendsUnknownWrites(const Endpoint& endpoint) noexcept {
    return (endpoint.capabilities.load(std::memory_order_acquire) & Endpoint::kOperationIds) &&
        !endpoint.operationIdsIgnored.load(std::memory_order_acquire);
}
// Whether layer writes on this endpoint are chained: its Ping reports chained
// writes and operation-ID support, and no replica has refused a chained
// header (design section 13).
inline bool chainsWrites(const Endpoint& endpoint) noexcept {
    return resendsUnknownWrites(endpoint) &&
        (endpoint.capabilities.load(std::memory_order_acquire) & Endpoint::kChainedWrites) &&
        !endpoint.chainingRefused.load(std::memory_order_acquire);
}
// Whether an OK reply with this rpc::OperationOutcome acknowledges a re-send.
// UNSPECIFIED comes from a replica without operation-ID support, which may
// have applied the re-send as a second write.
inline bool acknowledgesResend(uint32_t outcome) noexcept {
    return outcome == rpc::OPERATION_OUTCOME_APPLIED || outcome == rpc::OPERATION_OUTCOME_ALREADY_COMMITTED;
}

// The service's entry-limit refusal in `statusDetails`, a serialized
// google.rpc.Status from a rejection's grpc-status-details-bin. Reason zero
// unless an ErrorInfo detail of the service's domain names DELTA_TOO_LARGE or
// CHILDREN_LIST_TOO_LONG with every count as a decimal uint64; anything
// malformed or incomplete is no refusal. Reads nothing past `statusDetails`.
ServiceLimitRefusal parseServiceLimitRefusal(std::string_view statusDetails) noexcept;
// The service's refusal of a chained update in `statusDetails`: true, with
// the ErrorInfo's last_record and last_operation_id metadata, when an
// ErrorInfo detail of the service's domain names PREDECESSOR_NOT_COMMITTED
// and its metadata is consistent: last_record is absent, operation or
// no_operation_id, and last_operation_id, 1 to kMaxOperationIdBytes bytes, is
// present exactly for operation (empty or missing otherwise). Anything else
// is no such refusal and leaves both outputs unchanged.
bool parsePredecessorRefusal(std::string_view statusDetails, std::string& lastOperation, LastRecord& lastRecord);
// Whether a chained update's rejection `code` and `statusDetails` show a
// replica that does not know the chained header: INVALID_ARGUMENT without a
// well-formed google.rpc.ErrorInfo of domain nvidia.omniverse.live.v1alpha.
// A chaining replica marks every INVALID_ARGUMENT it gives a chained call
// with one, whatever its reason, so that refusal is definitive (design
// section 13).
bool refusedChainedHeader(grpc::StatusCode code, std::string_view statusDetails) noexcept;
// Whether a chained update's rejection `code` and `statusDetails` settle the
// call: an error with a well-formed google.rpc.ErrorInfo of domain
// nvidia.omniverse.live.v1alpha, or INVALID_ARGUMENT without one. Any other
// status, a plain UNAVAILABLE or RESOURCE_EXHAUSTED included, may follow a
// commit the service makes later (design section 13).
bool settlesChainedCall(grpc::StatusCode code, std::string_view statusDetails) noexcept;

// The live edit service's error details in `statusDetails`, a serialized
// google.rpc.Status from a rejection's grpc-status-details-bin. errorInfo is
// whether a google.rpc.ErrorInfo detail is complete: well formed, with a
// reason and the domain nvidia.omniverse.live.v1alpha. retryDelayMs is the
// delay of the first valid google.rpc.RetryInfo detail, rounded up to whole
// milliseconds, or kNoRetryDelay. clientKeepaliveTimeout is whether such an
// ErrorInfo names CLIENT_KEEPALIVE_TIMEOUT, the reason the service ends a join
// whose client sent nothing for three keepalive intervals. A malformed status
// has none of them. Reads nothing past `statusDetails`.
struct ServiceErrorDetails {
    bool errorInfo = false;
    bool clientKeepaliveTimeout = false;
    uint64_t retryDelayMs = kNoRetryDelay;
};
ServiceErrorDetails parseServiceErrorDetails(std::string_view statusDetails) noexcept;
// Whether a refused UpdateLayer or CreateLayer attempt provably applied
// nothing, so that its write or creation may be issued again unchanged
// (design section 13). The service
// attaches its ErrorInfo to every UNAVAILABLE that applied nothing; any other
// UNAVAILABLE may follow a commit. This is the only place that decides it.
inline bool refusalAppliedNothing(grpc::StatusCode code, const ServiceErrorDetails& details) noexcept {
    return code == grpc::StatusCode::UNAVAILABLE && details.errorInfo;
}

// Local provenance captured at issuance, never serialized on the wire.
// streamGeneration is the actor's read generation when the update was issued.
// Creation has a layer subject but no read generation or retained actor work ID.
struct LayerWriteDiagnostics {
    uint64_t subjectId = 0, streamGeneration = 0, workId = 0;
};

// Private deterministic transport seam. Production callers leave this null;
// tests drive real CQ tags through scripted write/half-close failure sequences.
// The factory is used only synchronously before admission, never on a CQ thread.
class LayerWriteTransport {
public:
    virtual ~LayerWriteTransport() = default;
    virtual std::unique_ptr<grpc::ClientAsyncWriterInterface<rpc::CreateLayerRequest>> create(grpc::ClientContext*,
        rpc::CreateLayerResponse*, grpc::CompletionQueue*) = 0;
    virtual std::unique_ptr<grpc::ClientAsyncWriterInterface<rpc::UpdateLayerRequest>> update(grpc::ClientContext*,
        rpc::UpdateLayerResponse*, grpc::CompletionQueue*) = 0;
};

// Private transport for the in-plugin layer owner. Copies inputs before
// returning; exactly one header and at most 2 MiB per subsequent message.
// These primitives require usable credentials and never retry issued mutations.
// The layer owner must enforce FIFO/incarnation/recovery and retain authored
// intent after OUTCOME_UNKNOWN; request release is not recovery/discard consent.
// A nonempty operationId is sent in the header; only an owner re-sending the
// same bytes under the same ID after an unknown outcome may reuse it.
std::shared_ptr<Request> submitLayerCreate(Runtime& runtime, OmniUsdLiveEndpoint endpoint, std::string_view sessionId,
    std::string_view layerName, std::string_view bytes, UnaryCallback<rpc::CreateLayerResponse> callback,
    uint32_t timeoutMs = 0, LayerWriteTransport* transport = nullptr,
    std::shared_ptr<LayerWriteReceipt> receipt = nullptr, LayerWriteDiagnostics diagnostics = {},
    std::string_view operationId = {});
// A supplied reservation on the same endpoint, at least layerUpdateCharge of
// the submission, backs its wire buffers instead of a new reservation, so the
// submission cannot be refused for pending bytes. The request shares it. A
// non-null previousOperationId chains the update: the service commits it only
// after that operation, the writer's previous write to the object, or for an
// empty value only as the writer's first write to the object. A chained
// update waits at the service for its predecessor unless `noWait`, which has
// the service commit it, answer ALREADY_COMMITTED or refuse it at once.
std::shared_ptr<Request> submitLayerUpdate(Runtime& runtime, OmniUsdLiveEndpoint endpoint, std::string_view sessionId,
    std::string_view layerName, uint64_t objectId, uint64_t estimatedServerTimeUs, std::string_view bytes,
    UnaryCallback<rpc::UpdateLayerResponse> callback, uint32_t timeoutMs = 0, LayerWriteTransport* transport = nullptr,
    std::shared_ptr<LayerWriteReceipt> receipt = nullptr, LayerWriteDiagnostics diagnostics = {},
    std::shared_ptr<PendingBytes> reservation = nullptr, std::string_view operationId = {},
    const std::string* previousOperationId = nullptr, bool noWait = false);
// An upper bound on the encoded UpdateLayer header of this identity and
// writer, for any object ID, timestamp, operation ID and predecessor.
size_t layerUpdateHeaderBound(std::string_view sessionId, std::string_view layerName, std::string_view writerId);
// The pending bytes submitLayerUpdate charges for `bytes` under a header of
// `headerBytes` encoded bytes.
uint64_t layerUpdateCharge(size_t headerBytes, size_t bytes) noexcept;
} // namespace omni::usd::live
