// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "LiveLayerWrite.h"
#include "LiveDiagnostics.h"
#include <grpcpp/alarm.h>
#include <array>
#include <cassert>
#include <future>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace omni::usd::live;
void liveTestFailAllocationsAfter(size_t) noexcept;
void liveTestRestoreAllocations() noexcept;
namespace {
// Independent scripted transport drives production CQ state transitions. It
// deliberately returns a valid-looking OK ack after an incomplete upload.
// Alarm cancellation produces a deterministic false CQ completion (not a race
// against a socket's buffering or the server application's receive schedule).
enum class Failure { None, Start, Header, Chunk, HalfClose };
struct Script {
    Failure failure;
    std::vector<unsigned> events;
    std::string operation; // The operation ID every header carries; the reply reports it committed earlier.
    bool legacy = false; // The reply omits the outcome, as a replica without operation-ID support does.
    explicit Script(Failure value, std::string id = {}) : failure(value), operation(std::move(id)) { events.reserve(8); }
};
template <class Input, class Response>
class Writer final : public grpc::ClientAsyncWriterInterface<Input> {
public:
    Writer(std::shared_ptr<Script> script, Response* response, grpc::CompletionQueue* queue)
        : script_(std::move(script)), response_(response), queue_(queue) {}
    void StartCall(void* tag) override
    {
        script_->events.push_back(0);
        signal(tag, script_->failure != Failure::Start);
    }
    void ReadInitialMetadata(void*) override { assert(false); }
    void Write(const Input& message, void* tag) override
    {
        const auto index = writes_++;
        script_->events.push_back(index + 1);
        if (!index) {
            assert(message.has_header());
            assert(message.header().session_id() == "scripted-session" && message.header().layer_name() == "root.live");
            assert(message.header().operation_id() == script_->operation);
            if constexpr (std::is_same_v<Input, rpc::UpdateLayerRequest>) {
                assert(message.header().object_id() == 13 && !message.header().client_id().empty());
                assert(message.header().timestamp_microseconds_since_epoch() == 42);
            }
        } else {
            assert(message.has_chunk());
            const auto& data = message.chunk().data();
            assert(data.size() == (index == 1 ? kLayerChunkBytes : 13));
            assert(data.find_first_not_of(index == 1 ? 'a' : 'b') == std::string::npos);
        }
        signal(tag, !((!index && script_->failure == Failure::Header) ||
                      (index && script_->failure == Failure::Chunk)));
    }
    void Write(const Input&, grpc::WriteOptions, void*) override { assert(false); }
    void WritesDone(void* tag) override
    {
        assert(writes_ == 3);
        script_->events.push_back(4);
        signal(tag, script_->failure != Failure::HalfClose);
    }
    void Finish(grpc::Status* status, void* tag) override
    {
        script_->events.push_back(5);
        *status = grpc::Status::OK;
        if constexpr (std::is_same_v<Response, rpc::CreateLayerResponse>) response_->set_object_id(11);
        else response_->set_sequence_num(7);
        if (!script_->operation.empty() && !script_->legacy) response_->set_operation_outcome(rpc::OPERATION_OUTCOME_ALREADY_COMMITTED);
        signal(tag, true);
    }
private:
    std::shared_ptr<Script> script_;
    Response* response_;
    grpc::CompletionQueue* queue_;
    std::array<grpc::Alarm, 8> alarms_;
    size_t nextAlarm_ = 0;
    unsigned writes_ = 0;
    void signal(void* tag, bool ok)
    {
        assert(nextAlarm_ < alarms_.size());
        auto& alarm = alarms_[nextAlarm_++];
        alarm.Set(queue_, std::chrono::system_clock::now() + (ok ? std::chrono::hours(0) : std::chrono::hours(1)), tag);
        if (!ok) alarm.Cancel();
    }
};
class Transport final : public LayerWriteTransport {
public:
    std::shared_ptr<Script> script;
    explicit Transport(Failure failure, std::string operation = {})
        : script(std::make_shared<Script>(failure, std::move(operation))) {}
    std::unique_ptr<grpc::ClientAsyncWriterInterface<rpc::CreateLayerRequest>> create(
        grpc::ClientContext* context, rpc::CreateLayerResponse* response, grpc::CompletionQueue* queue) override
    {
        assert(context->raw_deadline().clock_type == GPR_CLOCK_MONOTONIC);
        return std::make_unique<Writer<rpc::CreateLayerRequest, rpc::CreateLayerResponse>>(script, response, queue);
    }
    std::unique_ptr<grpc::ClientAsyncWriterInterface<rpc::UpdateLayerRequest>> update(
        grpc::ClientContext* context, rpc::UpdateLayerResponse* response, grpc::CompletionQueue* queue) override
    {
        assert(context->raw_deadline().clock_type == GPR_CLOCK_MONOTONIC);
        return std::make_unique<Writer<rpc::UpdateLayerRequest, rpc::UpdateLayerResponse>>(script, response, queue);
    }
};
// Stop caller-thread allocation injection at the transport factory boundary.
// Every header/chunk/tag has already been prepared, but no RPC was issued.
// CQ and gRPC alarm internals are not the allocation-failure target here.
class PreparationTransport final : public LayerWriteTransport {
public:
    Transport transport{Failure::None};
    bool reached = false;
    std::unique_ptr<grpc::ClientAsyncWriterInterface<rpc::CreateLayerRequest>> create(
        grpc::ClientContext* context, rpc::CreateLayerResponse* response, grpc::CompletionQueue* queue) override {
        liveTestRestoreAllocations(); reached = true;
        return transport.create(context,response,queue);
    }
    std::unique_ptr<grpc::ClientAsyncWriterInterface<rpc::UpdateLayerRequest>> update(
        grpc::ClientContext* context, rpc::UpdateLayerResponse* response, grpc::CompletionQueue* queue) override {
        liveTestRestoreAllocations(); reached = true;
        return transport.update(context,response,queue);
    }
};
void allocationRollback(Runtime& runtime, OmniUsdLiveEndpoint endpoint, const std::string& bytes) {
    for (bool update : {false,true}) {
        bool succeeded = false;
        size_t failures = 0;
        for (size_t permit = 0; permit < 256; ++permit) {
            PreparationTransport transport;
            auto receipt = std::make_shared<LayerWriteReceipt>();
            std::shared_ptr<Request> request;
            const auto before = runtime.diagnostics->snapshot(0, 0).metrics.cqTags.submitted;
            liveTestFailAllocationsAfter(permit);
            try {
                request = update ?
                    submitLayerUpdate(runtime, endpoint, "scripted-session", "root.live", 13, 42, bytes, {}, 0, &transport, receipt) :
                    submitLayerCreate(runtime, endpoint, "scripted-session", "root.live", bytes, {}, 0, &transport, receipt);
                liveTestRestoreAllocations();
            } catch (const std::bad_alloc&) {
                liveTestRestoreAllocations(); ++failures;
                assert(!request && !transport.reached && transport.transport.script->events.empty());
                assert(runtime.diagnostics->snapshot(0, 0).metrics.cqTags.submitted == before);
                assert(receipt->object.load() == 0 && receipt->sequence.load() == 0);
                const auto pending = runtime.diagnostics->snapshot(0, 0).metrics.pendingBytes;
                assert(pending.reservations.current == 0 && pending.currentBytes.low == 0 && pending.currentBytes.high == 0);
                auto retained = runtime.findEndpoint(endpoint);
                std::lock_guard<std::mutex> lock(retained->mutex);
                assert(retained->dependents == 0 && retained->pendingTransportBytes == 0);
                continue;
            } catch (...) { liveTestRestoreAllocations(); throw; }
            assert(request && transport.reached);
            assert(request->wait(3000) == OMNI_USD_LIVE_OK);
            OmniUsdLiveResult result;
            assert(request->result(&result) == OMNI_USD_LIVE_OK && result == OMNI_USD_LIVE_OK);
            assert(receipt->sequence.load() == (update ? 7 : 1));
            assert(receipt->object.load() == (update ? 0 : 11));
            assert((transport.transport.script->events == std::vector<unsigned>{0,1,2,3,4,5}));
            assert(runtime.requests->release(request->handle) == OMNI_USD_LIVE_OK);
            request.reset();
            const auto until = std::chrono::steady_clock::now()+std::chrono::seconds(3);
            for (;;) {
                bool retired;
                {
                    auto retained = runtime.findEndpoint(endpoint);
                    std::lock_guard<std::mutex> lock(retained->mutex);
                    retired = !retained->dependents && !retained->pendingTransportBytes;
                }
                if (retired) break;
                assert(std::chrono::steady_clock::now() < until);
                std::this_thread::yield();
            }
            succeeded = true; break;
        }
        assert(succeeded && failures >= 8);
    }
}
// Protobuf wire encoding of the google.rpc messages a rejection carries in
// grpc-status-details-bin; the reference service encodes them with protobuf.
std::string varint(uint64_t value) {
    std::string out;
    do { out.push_back(char((value & 0x7f) | (value > 0x7f ? 0x80 : 0))); value >>= 7; } while (value);
    return out;
}
std::string bytesField(uint32_t number, const std::string& value) {
    return varint(uint64_t(number) << 3 | 2) + varint(value.size()) + value;
}
std::string errorInfo(const std::string& reason, const std::string& domain,
    const std::vector<std::pair<std::string, std::string>>& metadata) {
    auto info = bytesField(1, reason) + bytesField(2, domain);
    for (const auto& [key, value] : metadata) info += bytesField(3, bytesField(1, key) + bytesField(2, value));
    return info;
}
std::string statusDetails(const std::string& info, const std::string& type = "type.googleapis.com/google.rpc.ErrorInfo") {
    const auto any = bytesField(1, type) + bytesField(2, info);
    // code INVALID_ARGUMENT, a message, an unrelated detail first, then the ErrorInfo.
    return varint(1 << 3) + varint(3) + bytesField(2, "delta too large") +
        bytesField(3, bytesField(1, "type.googleapis.com/google.rpc.DebugInfo") + bytesField(2, "x")) + bytesField(3, any);
}
// The service's entry-limit ErrorInfo is recognized only whole: its domain,
// a known reason and every count as a decimal number.
void serviceLimitRefusals() {
    constexpr const char* domain = "nvidia.omniverse.live.v1alpha";
    const auto large = parseServiceLimitRefusal(statusDetails(errorInfo("DELTA_TOO_LARGE", domain,
        {{"struct_commands", "1"}, {"set_fields", "99998"}, {"time_samples", "1"}, {"max_entries", "99999"}})));
    assert(large.reason == 1 && large.structCommands == 1 && large.setFields == 99998 && large.timeSamples == 1 &&
        large.limit == 99999 && large.actual == 100000);
    const auto list = parseServiceLimitRefusal(statusDetails(errorInfo("CHILDREN_LIST_TOO_LONG", domain,
        {{"actual", "100001"}, {"limit", "100000"}})));
    assert(list.reason == 2 && list.limit == 100000 && list.actual == 100001);
    for (const auto& unrecognized : {std::string(),
             statusDetails(errorInfo("DELTA_TOO_LARGE", "other.domain",
                 {{"struct_commands", "1"}, {"set_fields", "1"}, {"time_samples", "1"}, {"max_entries", "2"}})),
             statusDetails(errorInfo("OTHER_REASON", domain, {{"limit", "1"}, {"actual", "2"}})),
             statusDetails(errorInfo("CHILDREN_LIST_TOO_LONG", domain, {{"limit", "1"}})),
             statusDetails(errorInfo("CHILDREN_LIST_TOO_LONG", domain, {{"limit", "1"}, {"actual", "-2"}})),
             statusDetails(errorInfo("CHILDREN_LIST_TOO_LONG", domain, {{"limit", "1"}, {"actual", "18446744073709551616"}})),
             statusDetails(errorInfo("CHILDREN_LIST_TOO_LONG", domain, {{"limit", "1"}, {"actual", "2"}}),
                 "type.googleapis.com/google.rpc.BadRequest"),
             statusDetails(errorInfo("CHILDREN_LIST_TOO_LONG", domain, {{"limit", "1"}, {"actual", "2"}})).substr(0, 40)}) {
        assert(parseServiceLimitRefusal(unrecognized).reason == 0);
    }
    // A known field number with another wire type is skipped as protobuf
    // skips it, never read as the bytes of an earlier unknown field: here an
    // unknown string field holds the value and the known field is a varint.
    const auto wrongType = [](uint32_t number) { return varint(uint64_t(number) << 3) + varint(7); };
    const auto children = errorInfo("CHILDREN_LIST_TOO_LONG", domain, {{"limit", "1"}, {"actual", "2"}});
    const std::string anyType = "type.googleapis.com/google.rpc.ErrorInfo";
    for (const auto& stale : {
             statusDetails(bytesField(1, "CHILDREN_LIST_TOO_LONG") + bytesField(4, domain) + wrongType(2) +
                 bytesField(3, bytesField(1, "limit") + bytesField(2, "1")) +
                 bytesField(3, bytesField(1, "actual") + bytesField(2, "2"))),
             statusDetails(bytesField(1, "CHILDREN_LIST_TOO_LONG") + bytesField(2, domain) +
                 bytesField(3, bytesField(1, "limit") + bytesField(2, "1")) +
                 bytesField(3, bytesField(1, "actual") + bytesField(4, "2") + wrongType(2))),
             varint(1 << 3) + varint(3) + bytesField(3, bytesField(1, anyType) + bytesField(4, children) + wrongType(2))}) {
        assert(parseServiceLimitRefusal(stale).reason == 0);
    }
    // A complete ErrorInfo stays recognized beside fields of other wire types.
    const auto mixed = parseServiceLimitRefusal(statusDetails(wrongType(1) + children + wrongType(5)));
    assert(mixed.reason == 2 && mixed.limit == 1 && mixed.actual == 2);
}
// A chained update's refusal names the writer's latest write in the service's
// ErrorInfo: last_record says what it left, and last_operation_id holds its
// operation ID exactly when last_record is `operation`.
void predecessorRefusals() {
    constexpr const char* domain = "nvidia.omniverse.live.v1alpha";
    const auto refusal = [&](std::vector<std::pair<std::string, std::string>> metadata) {
        return statusDetails(errorInfo("PREDECESSOR_NOT_COMMITTED", domain, metadata));
    };
    std::string last = "unchanged";
    auto record = LastRecord::Absent;
    assert(parsePredecessorRefusal(refusal({{"last_operation_id", "op-7"}, {"last_record", "operation"}}), last,
               record) &&
        last == "op-7" && record == LastRecord::Operation);
    // No record: absent before any write or after a lost record, or a latest
    // write without an operation ID; last_operation_id is empty or missing.
    assert(parsePredecessorRefusal(refusal({{"last_operation_id", ""}, {"last_record", "absent"}}), last, record) &&
        last.empty() && record == LastRecord::Absent);
    last = "x";
    assert(parsePredecessorRefusal(refusal({{"last_record", "no_operation_id"}}), last, record) && last.empty() &&
        record == LastRecord::NoOperationId);
    assert(parsePredecessorRefusal(refusal({{"last_record", "absent"}}), last, record) && last.empty() &&
        record == LastRecord::Absent);
    for (const auto& other : {std::string(),
             statusDetails(errorInfo("PREDECESSOR_NOT_COMMITTED", "other.domain",
                 {{"last_operation_id", "a"}, {"last_record", "operation"}})),
             statusDetails(errorInfo("OTHER_REASON", domain, {{"last_operation_id", "a"}, {"last_record", "operation"}})),
             refusal({{"last_operation_id", std::string(65, 'a')}, {"last_record", "operation"}}),
             // last_record is required and must agree with last_operation_id.
             refusal({{"last_operation_id", "a"}}), refusal({}), refusal({{"last_operation_id", ""}}),
             refusal({{"last_record", "operation"}}), refusal({{"last_operation_id", ""}, {"last_record", "operation"}}),
             refusal({{"last_operation_id", "a"}, {"last_record", "absent"}}),
             refusal({{"last_operation_id", "a"}, {"last_record", "no_operation_id"}}),
             refusal({{"last_operation_id", "a"}, {"last_record", "other"}}),
             statusDetails(errorInfo("PREDECESSOR_NOT_COMMITTED", domain, {{"last_record", "absent"}}),
                 "type.googleapis.com/google.rpc.BadRequest"),
             refusal({{"last_operation_id", "a"}, {"last_record", "operation"}}).substr(0, 40)}) {
        last = "kept";
        record = LastRecord::NoOperationId;
        assert(!parsePredecessorRefusal(other, last, record) && last == "kept");
    }
}
// A chained call refused INVALID_ARGUMENT without an ErrorInfo of the
// service's domain came from a replica that does not know the chained header;
// a chaining replica marks every INVALID_ARGUMENT it gives a chained call
// with one, INVALID_UPDATE or another reason, and that refusal is definitive.
void chainedHeaderRefusals() {
    constexpr const char* domain = "nvidia.omniverse.live.v1alpha";
    const auto invalid = [](const std::string& details) {
        return varint(1 << 3) + varint(3) + bytesField(2, "invalid") + details;
    };
    const auto info = [](const std::string& reason, const std::string& domainName) {
        return bytesField(3, bytesField(1, "type.googleapis.com/google.rpc.ErrorInfo") +
                                 bytesField(2, errorInfo(reason, domainName, {})));
    };
    for (const auto& unknown : {std::string(), invalid(""), invalid(info("INVALID_UPDATE", "other.domain")),
             invalid(bytesField(3, bytesField(1, "type.googleapis.com/google.rpc.BadRequest") +
                 bytesField(2, errorInfo("INVALID_UPDATE", domain, {})))),
             invalid(info("INVALID_UPDATE", domain)).substr(0, 30)})
        assert(refusedChainedHeader(grpc::StatusCode::INVALID_ARGUMENT, unknown));
    for (const auto& marked : {invalid(info("INVALID_UPDATE", domain)), invalid(info("", domain)),
             invalid(info("DELTA_TOO_LARGE", domain)), invalid(info("X", "other.domain") + info("INVALID_UPDATE", domain))})
        assert(!refusedChainedHeader(grpc::StatusCode::INVALID_ARGUMENT, marked));
    for (const auto code : {grpc::StatusCode::OK, grpc::StatusCode::UNAVAILABLE, grpc::StatusCode::ABORTED,
             grpc::StatusCode::OUT_OF_RANGE, grpc::StatusCode::FAILED_PRECONDITION})
        assert(!refusedChainedHeader(code, std::string()));
    // Only an error with the service's ErrorInfo or an INVALID_ARGUMENT
    // without one settles a chained call; any other status may follow a
    // commit the service makes later.
    const auto status = [](int code, const std::string& details) {
        return varint(1 << 3) + varint(uint64_t(code)) + bytesField(2, "refused") + details;
    };
    for (const auto& [code, details] : std::vector<std::pair<grpc::StatusCode, std::string>>{
             {grpc::StatusCode::INVALID_ARGUMENT, std::string()},
             {grpc::StatusCode::INVALID_ARGUMENT, invalid(info("INVALID_UPDATE", domain))},
             {grpc::StatusCode::RESOURCE_EXHAUSTED, status(8, info("UPLOAD_ADMISSION_SATURATED", domain))},
             {grpc::StatusCode::UNAVAILABLE, status(14, info("", domain))},
             {grpc::StatusCode::ABORTED, status(10, info("PREDECESSOR_NOT_COMMITTED", domain))}})
        assert(settlesChainedCall(code, details));
    for (const auto& [code, details] : std::vector<std::pair<grpc::StatusCode, std::string>>{
             {grpc::StatusCode::OK, std::string()}, {grpc::StatusCode::RESOURCE_EXHAUSTED, std::string()},
             {grpc::StatusCode::RESOURCE_EXHAUSTED, status(8, info("UPLOAD_ADMISSION_SATURATED", "other.domain"))},
             {grpc::StatusCode::UNAVAILABLE, std::string()}, {grpc::StatusCode::CANCELLED, std::string()},
             {grpc::StatusCode::DEADLINE_EXCEEDED, std::string()},
             {grpc::StatusCode::PERMISSION_DENIED, status(7, info("X", domain)).substr(0, 30)},
             {grpc::StatusCode::UNAUTHENTICATED, std::string()}})
        assert(!settlesChainedCall(code, details));
}
// A google.protobuf.Duration of seconds 1 and nanos 2, each a varint field;
// a negative value is a sign-extended 64-bit varint, as protobuf encodes it.
std::string duration(int64_t seconds, int32_t nanos) {
    return (seconds ? varint(1 << 3) + varint(uint64_t(seconds)) : std::string()) +
        (nanos ? varint(2 << 3) + varint(uint64_t(int64_t(nanos))) : std::string());
}
std::string anyDetail(const std::string& type, const std::string& value) {
    return bytesField(3, bytesField(1, "type.googleapis.com/google.rpc." + type) + bytesField(2, value));
}
std::string unavailable(const std::string& details) {
    return varint(1 << 3) + varint(14) + bytesField(2, "service unavailable") + details;
}
// An UNAVAILABLE applied nothing only with a complete ErrorInfo of the
// service's domain; a RetryInfo delay is read alone, rounded up to whole
// milliseconds, and only when it is a valid nonnegative Duration.
void serviceErrorDetails() {
    constexpr const char* domain = "nvidia.omniverse.live.v1alpha";
    const auto info = anyDetail("ErrorInfo", errorInfo("UPLOAD_NOT_ADMITTED", domain, {{"key", "value"}}));
    const auto retry = [](int64_t seconds, int32_t nanos) {
        return anyDetail("RetryInfo", bytesField(1, duration(seconds, nanos)));
    };
    auto parsed = parseServiceErrorDetails(unavailable(info + retry(1, 500000000)));
    assert(parsed.errorInfo && parsed.retryDelayMs == 1500);
    assert(refusalAppliedNothing(grpc::StatusCode::UNAVAILABLE, parsed));
    for (const auto code : {grpc::StatusCode::INTERNAL, grpc::StatusCode::RESOURCE_EXHAUSTED,
             grpc::StatusCode::DEADLINE_EXCEEDED, grpc::StatusCode::OK})
        assert(!refusalAppliedNothing(code, parsed));
    // Either detail stands alone, in any order beside unrelated details.
    parsed = parseServiceErrorDetails(unavailable(anyDetail("DebugInfo", "x") + retry(0, 1) + info));
    assert(parsed.errorInfo && parsed.retryDelayMs == 1);
    parsed = parseServiceErrorDetails(unavailable(info));
    assert(parsed.errorInfo && parsed.retryDelayMs == kNoRetryDelay);
    parsed = parseServiceErrorDetails(unavailable(retry(2, 0)));
    assert(!parsed.errorInfo && parsed.retryDelayMs == 2000);
    assert(!refusalAppliedNothing(grpc::StatusCode::UNAVAILABLE, parsed));
    // An empty Duration is zero and a RetryInfo without one is none; the
    // largest Duration is kept exactly.
    assert(parseServiceErrorDetails(unavailable(retry(0, 0))).retryDelayMs == 0);
    assert(parseServiceErrorDetails(unavailable(anyDetail("RetryInfo", ""))).retryDelayMs == kNoRetryDelay);
    assert(parseServiceErrorDetails(unavailable(retry(315576000000, 999999999))).retryDelayMs == 315576000001000);
    // The first valid RetryInfo counts; an invalid one alone is none.
    assert(parseServiceErrorDetails(unavailable(retry(3, 0) + retry(4, 0))).retryDelayMs == 3000);
    for (const auto& invalid : {retry(-1, 0), retry(0, -1), retry(315576000001, 0), retry(0, 1000000000),
             anyDetail("RetryInfo", bytesField(1, varint(1 << 3)))})
        assert(parseServiceErrorDetails(unavailable(info + invalid)).retryDelayMs == kNoRetryDelay);
    // Only a well-formed ErrorInfo with a reason and the service's domain is complete.
    for (const auto& incomplete : {std::string(), unavailable(""),
             unavailable(anyDetail("ErrorInfo", errorInfo("UPLOAD_NOT_ADMITTED", "other.domain", {}))),
             unavailable(anyDetail("ErrorInfo", errorInfo("", domain, {}))),
             unavailable(anyDetail("ErrorInfo", bytesField(1, "UPLOAD_NOT_ADMITTED"))),
             unavailable(anyDetail("ErrorInfo", errorInfo("UPLOAD_NOT_ADMITTED", domain, {}) + bytesField(3, "\x0a"))),
             unavailable(anyDetail("BadRequest", errorInfo("UPLOAD_NOT_ADMITTED", domain, {}))),
             unavailable(bytesField(3, bytesField(1, "google.rpc.ErrorInfo") +
                 bytesField(2, errorInfo("UPLOAD_NOT_ADMITTED", domain, {})))),
             unavailable(info).substr(0, unavailable(info).size() - 3), unavailable(info) + "\x1a"})
        assert(!parseServiceErrorDetails(incomplete).errorInfo);
    // A malformed status has neither detail.
    parsed = parseServiceErrorDetails(unavailable(info + retry(1, 0)) + "\x1a\x05");
    assert(!parsed.errorInfo && parsed.retryDelayMs == kNoRetryDelay);
    // A keepalive timeout is a complete ErrorInfo of the service's domain whose reason is CLIENT_KEEPALIVE_TIMEOUT.
    const auto keepalive = anyDetail("ErrorInfo", errorInfo("CLIENT_KEEPALIVE_TIMEOUT", domain, {}));
    assert(!parseServiceErrorDetails(unavailable(info)).clientKeepaliveTimeout);
    parsed = parseServiceErrorDetails(unavailable(info + keepalive));
    assert(parsed.errorInfo && parsed.clientKeepaliveTimeout);
    for (const auto& other : {unavailable(anyDetail("ErrorInfo", errorInfo("CLIENT_KEEPALIVE_TIMEOUT", "other.domain", {}))),
             unavailable(anyDetail("RetryInfo", errorInfo("CLIENT_KEEPALIVE_TIMEOUT", domain, {}))),
             unavailable(anyDetail("ErrorInfo", errorInfo("CLIENT_KEEPALIVE_TIMEOUT_", domain, {}))),
             unavailable(keepalive) + "\x1a"})
        assert(!parseServiceErrorDetails(other).clientKeepaliveTimeout);
}
}
void testLayerWriteLifecycle()
{
    serviceLimitRefusals();
    predecessorRefusals();
    chainedHeaderRefusals();
    serviceErrorDetails();
    // One UUIDv4 per write: 36 bytes of the operation-ID alphabet, version 4 and the RFC 4122 variant.
    std::set<std::string> minted;
    for (unsigned i = 0; i < 64; ++i) {
        const auto id = newOperationId();
        assert(id.size() == 36 && id.size() <= kMaxOperationIdBytes);
        for (size_t at = 0; at < id.size(); ++at)
            assert((at == 8 || at == 13 || at == 18 || at == 23) ? id[at] == '-' :
                std::string_view("0123456789abcdef").find(id[at]) != std::string_view::npos);
        assert(id[14] == '4' && std::string_view("89ab").find(id[19]) != std::string_view::npos);
        minted.insert(id);
    }
    assert(minted.size() == 64);
    OmniUsdLiveRuntimeOptions configuration{};
    configuration.struct_size = sizeof(configuration); configuration.abi_version = OMNI_USD_LIVE_ABI_VERSION;
    configuration.diagnostic_min_severity = OMNI_USD_LIVE_SEVERITY_DEBUG;
    assert(omniUsdLiveInitialize(&configuration) == OMNI_USD_LIVE_OK);
    OmniUsdLiveEndpointOptions options{};
    options.struct_size = sizeof(options); options.abi_version = OMNI_USD_LIVE_ABI_VERSION;
    options.authority = "scripted.example.invalid"; options.access_token = "synthetic-script";
    OmniUsdLiveEndpoint endpoint = 0;
    assert(omniUsdLiveRegisterEndpoint(&options, &endpoint) == OMNI_USD_LIVE_OK);
    auto runtime = getRuntime();
    const std::string bytes = std::string(kLayerChunkBytes, 'a') + std::string(13, 'b');
    allocationRollback(*runtime, endpoint, bytes);
    // Until a Ping reports operation-ID support no unknown outcome on this
    // endpoint is re-sent. Only APPLIED or ALREADY_COMMITTED acknowledges a re-send.
    const auto registered = runtime->findEndpoint(endpoint);
    assert(!registered->operationIdsIgnored.load() && !resendsUnknownWrites(*registered) && !chainsWrites(*registered));
    assert(!acknowledgesResend(rpc::OPERATION_OUTCOME_UNSPECIFIED) &&
           acknowledgesResend(rpc::OPERATION_OUTCOME_APPLIED) &&
           acknowledgesResend(rpc::OPERATION_OUTCOME_ALREADY_COMMITTED));
    uint64_t expectedTags = 12, expectedFalseReceipts = 0; // Two successful six-event scripts above.
    const auto caller = std::this_thread::get_id();
    for (bool update : {false, true}) {
        for (auto failure : {Failure::None, Failure::Start, Failure::Header, Failure::Chunk, Failure::HalfClose}) {
            // Every write carries an operation ID; only a validated success records its reported outcome.
            const std::string operation = update ? "update-" + std::to_string(int(failure)) : "create-a.b_c~0";
            Transport transport(failure, operation);
            unsigned callbacks = 0;
            uint64_t callbackId = 0;
            const auto expected = failure == Failure::None ? OMNI_USD_LIVE_OK : OMNI_USD_LIVE_OUTCOME_UNKNOWN;
            auto callback = [&](const OmniUsdLiveRequestError& error, const auto&, uint64_t generation) {
                assert(std::this_thread::get_id() != caller && generation == 1);
                assert(error.result == expected && error.grpc_status_present && error.grpc_status == 0);
                assert(error.attempt_id == 1);
                callbackId = error.request_id;
                ++callbacks;
            };
            auto receipt = std::make_shared<LayerWriteReceipt>();
            const LayerWriteDiagnostics correlation = update ? LayerWriteDiagnostics{71, 29, 37} : LayerWriteDiagnostics{73, 0, 0};
            auto request = update ?
                submitLayerUpdate(*runtime, endpoint, "scripted-session", "root.live", 13, 42, bytes, callback, 0, &transport, receipt, correlation, nullptr, operation) :
                submitLayerCreate(*runtime, endpoint, "scripted-session", "root.live", bytes, callback, 0, &transport, receipt, correlation, operation);
            assert(request->wait(3000) == OMNI_USD_LIVE_OK);
            assert(callbacks == 1 && callbackId == request->handle);
            assert(receipt->sequence.load() == (expected == OMNI_USD_LIVE_OK ? (update ? 7 : 1) : 0));
            assert(receipt->object.load() == (expected == OMNI_USD_LIVE_OK && !update ? 11 : 0));
            assert(receipt->outcome.load() == (expected == OMNI_USD_LIVE_OK ? rpc::OPERATION_OUTCOME_ALREADY_COMMITTED : rpc::OPERATION_OUTCOME_UNSPECIFIED));
            // A reply reporting an outcome grants no capability, which only a Ping reports.
            assert(!registered->operationIdsIgnored.load() && !resendsUnknownWrites(*registered));
            const auto diagnostics = runtime->diagnostics->snapshot(endpoint, 128);
            size_t matches = 0;
            for (const auto& entry : diagnostics.records) {
                if (entry.eventCode != OMNI_USD_LIVE_RECORD_RPC_COMPLETED && entry.eventCode != OMNI_USD_LIVE_RECORD_RPC_FAILED)
                    continue;
                const auto field = [&](DiagnosticKey key) {
                    for (uint32_t i = 0; i < entry.fieldCount; ++i)
                        if (entry.fields[i].key == key) return entry.fields[i].value;
                    assert(false && "missing RPC correlation field"); return uint64_t(0);
                };
                if (field(DiagnosticKey::Request) != request->handle) continue;
                ++matches;
                assert(entry.eventCode == (expected == OMNI_USD_LIVE_OK ? OMNI_USD_LIVE_RECORD_RPC_COMPLETED :
                                           OMNI_USD_LIVE_RECORD_RPC_FAILED));
                assert(entry.subjectId == correlation.subjectId);
                assert(field(DiagnosticKey::Generation) == correlation.streamGeneration);
                assert(field(DiagnosticKey::Work) == correlation.workId);
                assert(field(DiagnosticKey::AuthGeneration) == 1 && field(DiagnosticKey::Attempt) == 1);
                assert(field(DiagnosticKey::Operation) == (update ? OMNI_USD_LIVE_OPERATION_UPDATE : OMNI_USD_LIVE_OPERATION_CREATE));
                assert(field(DiagnosticKey::Result) == expected && field(DiagnosticKey::GrpcStatus) == 0);
            }
            assert(matches == 1);
            std::vector<unsigned> expectedEvents = failure == Failure::Start ? std::vector<unsigned>{0, 5} :
                failure == Failure::Header ? std::vector<unsigned>{0, 1, 5} :
                failure == Failure::Chunk ? std::vector<unsigned>{0, 1, 2, 5} : std::vector<unsigned>{0, 1, 2, 3, 4, 5};
            assert(transport.script->events == expectedEvents);
            expectedTags += expectedEvents.size();
            expectedFalseReceipts += failure == Failure::None ? 0 : 1;
            const auto tags = runtime->diagnostics->snapshot(0, 0).metrics.cqTags;
            assert(tags.submitted == expectedTags && tags.receivedNotOk == expectedFalseReceipts);
            assert(tags.receivedOk + tags.receivedNotOk == expectedTags && tags.outstanding == 0 && tags.abandoned == 0);
            {
                auto retained = runtime->findEndpoint(endpoint);
                std::lock_guard<std::mutex> lock(retained->mutex);
                assert(retained->pendingTransportBytes == 0);
            }
            assert(runtime->requests->release(request->handle) == OMNI_USD_LIVE_OK);
        }
    }
    // An OK reporting UNSPECIFIED to a request with an operation ID comes from a
    // replica that ignored it and withdraws the capability the Ping reported;
    // the next reported outcome restores it.
    registered->capabilities.store(Endpoint::kCapabilitiesKnown | Endpoint::kOperationIds | Endpoint::kChainedWrites);
    assert(resendsUnknownWrites(*registered) && chainsWrites(*registered));
    for (bool legacy : {true, false}) {
        Transport transport(Failure::None, "op-replica");
        transport.script->legacy = legacy;
        auto receipt = std::make_shared<LayerWriteReceipt>();
        auto request = submitLayerUpdate(*runtime, endpoint, "scripted-session", "root.live", 13, 42, bytes, {}, 0,
            &transport, receipt, {}, nullptr, "op-replica");
        assert(request->wait(3000) == OMNI_USD_LIVE_OK);
        OmniUsdLiveResult result;
        assert(request->result(&result) == OMNI_USD_LIVE_OK && result == OMNI_USD_LIVE_OK);
        assert(receipt->outcome.load() == (legacy ? rpc::OPERATION_OUTCOME_UNSPECIFIED :
                                                    rpc::OPERATION_OUTCOME_ALREADY_COMMITTED));
        assert(registered->operationIdsIgnored.load() == legacy && resendsUnknownWrites(*registered) == !legacy &&
               chainsWrites(*registered) == !legacy);
        expectedTags += transport.script->events.size();
        assert(runtime->requests->release(request->handle) == OMNI_USD_LIVE_OK);
    }
    // Hold both callback workers while CQ completes a successful mutation.
    // Cancellation suppresses delivery but cannot erase the validated receipt.
    for (bool update : {false, true}) {
        std::promise<void> release;
        auto go = release.get_future().share();
        std::array<std::promise<void>, 2> entered;
        std::array<std::shared_ptr<Request>, 2> blockers;
        for (size_t i = 0; i < 2; ++i) {
            blockers[i] = runtime->requests->create(endpoint, OMNI_USD_LIVE_OPERATION_PING, {});
            OmniUsdLiveRequestError error{}; error.result = OMNI_USD_LIVE_OK;
            blockers[i]->finish(error, [&, i, go] { entered[i].set_value(); go.wait(); });
        }
        for (auto& barrier : entered) assert(barrier.get_future().wait_for(std::chrono::seconds(3)) == std::future_status::ready);
        Transport transport(Failure::None);
        auto receipt = std::make_shared<LayerWriteReceipt>();
        unsigned callbacks = 0;
        auto callback = [&](const auto&, const auto&, uint64_t) { ++callbacks; };
        auto request = update ?
            submitLayerUpdate(*runtime, endpoint, "scripted-session", "root.live", 13, 42, bytes, callback, 0, &transport, receipt) :
            submitLayerCreate(*runtime, endpoint, "scripted-session", "root.live", bytes, callback, 0, &transport, receipt);
        const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        while (!receipt->sequence.load(std::memory_order_acquire)) {
            assert(std::chrono::steady_clock::now() < until); std::this_thread::yield();
        }
        assert(request->cancel(0) == OMNI_USD_LIVE_TIMEOUT);
        const auto charged = runtime->diagnostics->snapshot(0, 0).metrics.pendingBytes;
        assert(charged.reservations.current == 1 && charged.currentBytes.low > bytes.size() && charged.currentBytes.high == 0);
        release.set_value();
        assert(request->wait(3000) == OMNI_USD_LIVE_OK && callbacks == 0);
        const auto retired = runtime->diagnostics->snapshot(0, 0).metrics.pendingBytes;
        assert(retired.reservations.current == 0 && retired.currentBytes.low == 0 && retired.currentBytes.high == 0);
        expectedTags += transport.script->events.size();
        OmniUsdLiveResult result;
        assert(request->result(&result) == OMNI_USD_LIVE_OK && result == OMNI_USD_LIVE_OK);
        assert(receipt->sequence.load() == (update ? 7 : 1));
        assert(runtime->requests->release(request->handle) == OMNI_USD_LIVE_OK);
        for (auto& blocker : blockers) {
            assert(blocker->wait(3000) == OMNI_USD_LIVE_OK);
            assert(runtime->requests->release(blocker->handle) == OMNI_USD_LIVE_OK);
        }
    }
    assert(omniUsdLiveUnregisterEndpoint(endpoint, 0) == OMNI_USD_LIVE_OK);
    assert(omniUsdLiveShutdown(3000) == OMNI_USD_LIVE_OK);
    const auto tags = runtime->diagnostics->snapshot(0, 0).metrics.cqTags;
    assert(tags.submitted == expectedTags && tags.retired == expectedTags);
    assert(tags.receivedNotOk == expectedFalseReceipts && tags.receivedOk + tags.receivedNotOk == expectedTags);
    assert(tags.outstanding == 0 && tags.handling == 0 && tags.retained == 0 && tags.abandoned == 0);
    runtime.reset();
}
