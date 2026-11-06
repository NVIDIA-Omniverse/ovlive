// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <omni/usd/live/Live.h>
#include "LiveUri.h"
#include "nvidia/omniverse/live/v1alpha/live.grpc.pb.h"
#include <grpcpp/grpcpp.h>
#include <algorithm>
#include <atomic>
#include <climits>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <map>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "LiveTsan.h"

namespace omni::usd::live {
class RequestRegistry;
class AuthRegistry;
class AuthCallback;
class DrainRegistry;
class StatusRegistry;
enum class ShutdownPhase : uint32_t;
class DiagnosticsRegistry;
class SessionRegistry;
class LayerAdmissionRegistry;
inline constexpr size_t kLayerChunkBytes = 2 * 1024 * 1024;
// The live edit service decodes at most 4 MiB per stream message; a session
// message keeps 64 KiB of that for the request's framing and identifiers.
inline constexpr uint64_t kMaxSessionMessageBytes = 4 * 1024 * 1024 - 64 * 1024;
inline uint32_t layerReadFrameLimit(uint32_t maxUpdateBytes) {
    return (maxUpdateBytes < kLayerChunkBytes ? maxUpdateBytes : uint32_t(kLayerChunkBytes)) + 64;
}
// Channel message limit. Payload admission excludes protobuf framing. Small
// payload quotas must still allow an identity header; a full-quota session
// message also carries its bounded sender identity.
inline uint64_t wireLimit(uint32_t maxUpdateBytes) noexcept {
    return std::min<uint64_t>(INT_MAX,
        std::max<uint64_t>(uint64_t(maxUpdateBytes) + kMaxUriBytes + 64, 3 * kMaxUriBytes + 1024));
}
// Each finite method reserves its own response allowance from the endpoint's
// pending-byte budget before issuance, and the channel caps that method's
// response at the same allowance:
//   ListSessions                finiteResponseLimit(max_update_bytes)
//   CreateSession               kMaxUriBytes + 1024 (one session identifier)
//   StatLayer                   4096 (creator name plus two uint64s; a creator
//                               name longer than about 4 KB fails the stat
//                               with RESOURCE_EXHAUSTED)
//   DeleteSession, DeleteLayer  1024 (empty responses)
// Ping is capped at 1024 and does not reserve: a PingResponse is two uint64s,
// and the timing ping's unary path holds no pending-byte reservation.
inline uint64_t finiteResponseLimit(uint32_t maxUpdateBytes) noexcept {
    return std::min<uint64_t>(wireLimit(maxUpdateBytes), std::max<uint64_t>(3 * kMaxUriBytes + 1024, 1u << 20));
}
inline constexpr uint64_t kCreateSessionResponseBytes = kMaxUriBytes + 1024;
inline constexpr uint64_t kStatLayerResponseBytes = 4096;
inline constexpr uint64_t kFixedResponseBytes = 1024;

// Every nonzero default is explicit. Zero input means this default except the
// separately handled severity/flags. All queue/diagnostic budgets are finite.
// request_timeout_ms exceeds the Live Edit service's 30 s request timeout, so
// a layer write's deadline outlasts the service's own internal retries.
#define LIVE_OPTION_DEFAULTS(X) \
    X(default_auth_wait_timeout_ms, 30000) \
    X(max_rejected_token_generations, 2) \
    X(auth_expiry_lead_ms, 30000) \
    X(request_timeout_ms, 35000) \
    X(initial_open_timeout_ms, 30000) \
    X(retry_initial_ms, 100) \
    X(retry_maximum_ms, 10000) \
    X(retry_max_attempts, 8) \
    X(peer_reconciliation_grace_ms, 10000) \
    X(max_update_bytes, 104857600) \
    X(max_pending_bytes_per_endpoint, 1073741824) \
    X(max_pending_messages_per_session, 1024) \
    X(max_synthetic_gap, 65536) \
    X(max_status_subjects, 4096) \
    X(max_status_observers, 64) \
    X(max_diagnostic_records, 2048) \
    X(max_diagnostic_bytes, 8388608) \
    X(max_diagnostic_sink_records, 512) \
    X(max_diagnostic_sinks, 8) \
    X(max_closed_incidents, 128) \
    X(max_requests, 4096) \
    X(callback_worker_count, 2) \
    X(max_auth_callbacks, 64) \
    X(apply_lag_warning_threshold, 256) \
    X(reconnect_initial_ms, 1000) \
    X(reconnect_maximum_ms, 60000) \
    X(reconnect_max_attempts, 10) \
    X(max_calls_per_endpoint, 1024)

struct RuntimeOptions {
    OmniUsdLiveRuntimeOptions values{};
    OmniUsdLiveJitterParams jitter{};
    static RuntimeOptions copy(const OmniUsdLiveRuntimeOptions* input);
    bool operator==(const RuntimeOptions& other) const;
};

// The steady-clock instant at which a token installed at (wallNowMs, now) with
// wall-clock expiry expiresAtMs ends; unset when the token has no expiry.
inline std::chrono::steady_clock::time_point steadyExpiry(uint64_t expiresAtMs, uint64_t wallNowMs,
    std::chrono::steady_clock::time_point now) noexcept {
    using Clock = std::chrono::steady_clock;
    if (!expiresAtMs) return {};
    const uint64_t lifetime = expiresAtMs > wallNowMs ? expiresAtMs - wallNowMs : 0;
    const auto headroom = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::time_point::max() - now).count();
    return lifetime >= static_cast<uint64_t>(headroom) ? Clock::time_point::max() :
                                                         now + std::chrono::milliseconds(lifetime);
}
struct EndpointOptions {
    Authority authority;
    std::string tlsServerName;
    std::string caPem;
    std::string initialToken;
    uint64_t expiresAtMs = 0;
    uint32_t authWaitMs = 0;
    bool secure = true;
    static EndpointOptions copy(const OmniUsdLiveEndpointOptions* input, const RuntimeOptions& runtime);
    bool sameConfiguration(const EndpointOptions& other) const;
};

class CallBudget;
class StreamReservation;
// One open gRPC call on an endpoint, from before its preparation until its
// completion is observed, or until the call is abandoned unissued. A call slot
// also holds one unit of the budget's call share; a stream slot rides on its
// owner's StreamReservation. Destruction releases it.
class CallSlot {
public:
    CallSlot() = default;
    CallSlot(const CallSlot&) = delete;
    CallSlot& operator=(const CallSlot&) = delete;
    CallSlot(CallSlot&& other) noexcept;
    CallSlot& operator=(CallSlot&& other) noexcept;
    ~CallSlot();
    explicit operator bool() const noexcept { return bool(budget_); }
    void reset() noexcept;
private:
    friend class CallBudget;
    friend class StreamReservation;
    std::shared_ptr<CallBudget> budget_;
    bool call_ = false;
};
// One unit of the budget's stream share, which a layer holds until it closes
// or fails with no call open and a session until it leaves, across every
// reconnect, so that a reconnect never competes for a unit. Its owner opens
// one stream call at a time on it.
class StreamReservation {
public:
    StreamReservation(const StreamReservation&) = delete;
    StreamReservation& operator=(const StreamReservation&) = delete;
    ~StreamReservation();
    CallSlot open() noexcept;
private:
    friend class CallBudget;
    explicit StreamReservation(std::shared_ptr<CallBudget> budget) noexcept : budget_(std::move(budget)) {}
    std::shared_ptr<CallBudget> budget_;
};
// A queue position for a call that waits for a free call unit. The budget
// hands a released unit to the oldest waiter and runs its wake outside every
// lock; destruction leaves the queue and returns a unit handed over but not taken.
class CallWaiter : public std::enable_shared_from_this<CallWaiter> {
public:
    CallWaiter(const CallWaiter&) = delete;
    CallWaiter& operator=(const CallWaiter&) = delete;
    ~CallWaiter();
private:
    friend class CallBudget;
    CallWaiter(std::shared_ptr<CallBudget> budget, std::function<void()> wake)
        : budget_(std::move(budget)), wake_(std::move(wake)) {}
    std::shared_ptr<CallBudget> budget_;
    std::function<void()> wake_;
    uint64_t granted_ = 0; // Units handed over, under the budget's mutex.
    bool queued_ = false;  // Under the budget's mutex.
};
// An endpoint's concurrent gRPC calls, which its one channel carries on one
// connection, bounded by max_calls_per_endpoint so that gRPC never queues a
// call behind a server or proxy stream limit at or above the budget (design
// section 4). The budget is split in two fixed shares, so neither kind of
// call can starve the other.
// The call share, max(2, limit / 8), serves short calls: layer writes,
// finite requests and Pings, each holding a unit while it is open. Writes,
// finite requests and the Ping that learns an endpoint's capabilities wait in
// FIFO order for a unit (acquire), so issuance windows shrink under
// contention; other calls are refused when none is free (tryAcquire). The
// stream share, the rest, serves layer reads and session joins, each holding
// a unit while its layer or session is live, across reconnects, with a
// layer's creation riding on its layer's unit; one past it is refused at once
// with RESOURCE_EXHAUSTED and never waits. Calls open on the endpoint
// therefore never exceed the limit. The mutex is a leaf: nothing runs under it.
class CallBudget final : public std::enable_shared_from_this<CallBudget> {
public:
    CallBudget(uint64_t limit, DiagnosticsRegistry& diagnostics, OmniUsdLiveEndpoint endpoint) noexcept;
    CallBudget(const CallBudget&) = delete;
    CallBudget& operator=(const CallBudget&) = delete;
    static uint64_t callShare(uint64_t limit) noexcept { return std::max<uint64_t>(2, limit / 8); }
    uint64_t limit() const noexcept { return limit_; }
    // A unit of the stream share for a layer or session whose first call is
    // `operation`; null, with a call_budget_exhausted record, when it is full.
    // Allocation failure throws and reserves nothing.
    std::shared_ptr<StreamReservation> reserveStream(uint32_t operation);
    // Whether the stream share has a free unit, without reserving it; false,
    // with the refusal's call_budget_exhausted record, when it is full. A
    // layer open checks it before its first RPC, which precedes its reservation.
    bool streamAvailable(uint32_t operation) noexcept;
    // A queue position whose wake runs, outside every lock, when the budget
    // hands it a unit.
    std::shared_ptr<CallWaiter> waiter(std::function<void()> wake);
    // A call unit for `waiter`: one handed to it, else a free one when no
    // other waiter is ahead. An empty slot queues the waiter, if it is not
    // queued, behind every other.
    CallSlot acquire(CallWaiter& waiter, uint32_t operation) noexcept;
    // A free call unit when no waiter is queued; empty otherwise, with a
    // call_budget_exhausted record when `report`. Never queues.
    CallSlot tryAcquire(uint32_t operation, bool report = true) noexcept;
    // Units handed to `waiter` and not taken go to the next waiter or the share.
    void returnGranted(CallWaiter& waiter) noexcept;
    // A call_budget_exhausted record for `operation` with `reason`.
    void report(uint32_t operation, uint32_t reason) noexcept;
    // Reasons of call_budget_exhausted records.
    static constexpr uint32_t kStreamRefused = 1, kCallQueued = 2, kCallRefused = 3, kWaitExpired = 4;
    struct Counts {
        uint64_t open = 0, peak = 0, streams = 0, calls = 0, waiting = 0;
    };
    // Calls open at the time of the call and at most so far, stream and call
    // units held, and waiters queued.
    Counts counts() const noexcept;
private:
    friend class CallSlot;
    friend class StreamReservation;
    friend class CallWaiter;
    void opened() noexcept; // Mutex held.
    void closed(bool call) noexcept;
    void releaseStream() noexcept;
    // Hands up to `units` call units to queued waiters, and returns the rest to the share.
    void releaseCalls(uint64_t units) noexcept;
    void pruneExpired() noexcept; // Mutex held.
    void emit(uint32_t operation, uint32_t reason, const Counts& counts) noexcept;
    const uint64_t limit_, calls_, streams_;
    DiagnosticsRegistry& diagnostics_;
    const OmniUsdLiveEndpoint endpoint_;
    mutable std::mutex mutex_;
    uint64_t open_ = 0, peak_ = 0, streamUnits_ = 0, callUnits_ = 0;
    // Waiters, oldest first. A waiter that is gone prunes its own entry, so a
    // free unit never stays behind an entry nobody waits on.
    std::deque<std::weak_ptr<CallWaiter>> waiters_;
    bool saturated_ = false; // A waiter queued since the call share last had a free unit.
};

class BufferedEditCount;
struct Endpoint {
    mutable std::mutex mutex;
    EndpointOptions options;
    OmniUsdLiveEndpoint handle;
    uint64_t owners = 1;
    uint64_t dependents = 0;
    uint64_t pendingTransportBytes = 0; // Charged immutable wire payloads, under mutex.
    // Layers' host edit buffers, under mutex. Every pending-byte release on
    // this endpoint reports to each, so a flush a shortfall deferred retries.
    std::vector<std::weak_ptr<BufferedEditCount>> pendingByteWatchers;
    uint64_t pendingByteRefusals = 0; // Reservations refused for lack of bytes, under mutex.
    // Every layer's host edit buffer bytes on this endpoint, under mutex: a
    // budget of max_pending_bytes_per_endpoint separate from the transport
    // bytes above, which never count it (design section 8, Quotas).
    uint64_t hostBufferBytes = 0;
    // A live reservation's lease prevents final unregister/shutdown. A closed
    // endpoint may outlive Runtime, but admits no reservation.
    DiagnosticsRegistry& diagnostics;
    bool closed = false;
    std::string writerId;
    std::string token;
    uint64_t tokenGeneration = 0;
    uint64_t expiresAtMs = 0;
    // The same lifetime on the steady clock, measured when the token was
    // installed; unset without an expiry. A token is unusable when either says so.
    std::chrono::steady_clock::time_point expiresAtSteady{};
    OmniUsdLiveAuthRequired challenge{};
    bool authFailed = false;
    uint64_t authFailureSerial = 0; // Retains failure across a later token replacement.
    bool generationRejected = false;
    uint32_t rejectedGenerations = 0;
    std::chrono::steady_clock::time_point authDeadline{};
    bool authWaitActive = false; // Observed credential-blocked interval; mutex held.
    std::chrono::steady_clock::time_point authWaitStarted{};
    // Strong observer ownership is removed explicitly by UnregisterCallback.
    // A temporary weak lock under this mutex must never destroy an endpoint lease.
    std::vector<std::shared_ptr<AuthCallback>> authObservers;
    OmniUsdLiveStatus status{};
    OmniUsdLiveJitterParams jitter{};
    // The capabilities the latest answered Ping reported, with
    // kCapabilitiesKnown set once any Ping is answered.
    static constexpr uint8_t kOperationIds = 1, kChainedWrites = 2, kCapabilitiesKnown = 4;
    std::atomic<uint8_t> capabilities{0};
    // The latest validated OK to a layer write or creation that carried an
    // operation ID reported UNSPECIFIED: a replica ignored the operation ID.
    std::atomic<bool> operationIdsIgnored{false};
    // A replica refused a chained header; the endpoint's writes stay unchained.
    std::atomic<bool> chainingRefused{false};
    // Session, layer name and object ID, joined by NUL bytes, of every object
    // a layer of this endpoint has issued an update to, under mutex. Another
    // layer's writes to one of them precede the next layer's, so its first
    // write cannot name its predecessor in the writer's chain.
    std::set<std::string> writtenObjects;
    // The same keys for objects one of whose unchained writes a layer of this
    // endpoint re-sent, each with the instant before which no later write of
    // the endpoint's writer to the object issues, under mutex. The service
    // keys its operation records by that writer ID, so the hold outlives the
    // layer that set it (design section 13, The single operation-ID re-send).
    std::map<std::string, std::chrono::steady_clock::time_point> resendHolds;
    std::shared_ptr<grpc::Channel> channel;
    std::unique_ptr<nvidia::omniverse::live::v1alpha::LiveService::Stub> stub;
    // Every call this endpoint's channel carries holds a slot of this budget.
    const std::shared_ptr<CallBudget> calls;

    Endpoint(EndpointOptions options, OmniUsdLiveEndpoint handle, std::string writerId, OmniUsdLiveJitterParams jitter,
        uint32_t maxUpdateBytes, uint64_t maxCalls, DiagnosticsRegistry& diagnostics);
};

// A layer/request/session/callback must acquire a dependency before starting
// work. Its lease survives every CQ tag/callback that can touch the endpoint.
class EndpointLease {
public:
    EndpointLease() = default;
    explicit EndpointLease(std::shared_ptr<Endpoint> endpoint);
    EndpointLease(const EndpointLease&) = delete;
    EndpointLease& operator=(const EndpointLease&) = delete;
    EndpointLease(EndpointLease&& other) noexcept = default;
    EndpointLease& operator=(EndpointLease&& other) noexcept;
    ~EndpointLease();
    Endpoint* operator->() const { return endpoint_.get(); }
    Endpoint& operator*() const { return *endpoint_; }
    explicit operator bool() const { return bool(endpoint_); }
private:
    std::shared_ptr<Endpoint> endpoint_;
    void release() noexcept;
};

// Reserve before copying payloads. A reservation retains endpoint ownership and
// must outlive the buffers it accounts for, including suppressed callbacks.
class PendingBytes {
public:
    PendingBytes(EndpointLease endpoint, uint64_t amount, uint64_t limit);
    PendingBytes(const PendingBytes&) = delete;
    PendingBytes& operator=(const PendingBytes&) = delete;
    PendingBytes(PendingBytes&& other) noexcept = default;
    PendingBytes& operator=(PendingBytes&& other) = delete;
    ~PendingBytes();
    Endpoint& endpoint() const { return *endpoint_; }
    uint64_t amount() const noexcept { return amount_; }
    // Release without reporting to the endpoint's watchers: the reservation
    // backed an admission that failed. It still reports when the endpoint
    // refused another reservation while this one was held, since that
    // refusal may have waited on these bytes.
    void releaseQuietly() noexcept { quiet_ = true; }
    // While one is active on a thread, releases on that thread report to
    // watchers only when the outermost scope ends. Wraps admission paths that
    // hold runtime or journal locks, so no host code runs under them.
    class DeferredReleaseNotices {
    public:
        DeferredReleaseNotices() noexcept;
        ~DeferredReleaseNotices();
        DeferredReleaseNotices(const DeferredReleaseNotices&) = delete;
        DeferredReleaseNotices& operator=(const DeferredReleaseNotices&) = delete;
    };
private:
    EndpointLease endpoint_;
    uint64_t amount_;
    uint64_t refusals_ = 0; // The endpoint's refusal count when reserved.
    bool quiet_ = false;
};

// One layer's host edit buffer bytes, charged to its endpoint's shared host
// buffer budget (design section 8, Quotas). Host-confined, like the buffer it
// charges. Charging always succeeds, so a started host turn records whole;
// budget() reports whether every buffer on the endpoint together reaches the
// limit, which refuses a new host turn. A decrease reports to the endpoint's
// pending-byte watchers, as a transport release does, so a layer a full budget
// left BLOCKED flushes again. Destruction releases the charge.
class HostBufferCharge {
public:
    HostBufferCharge(EndpointLease endpoint, uint64_t limit);
    HostBufferCharge(const HostBufferCharge&) = delete;
    HostBufferCharge& operator=(const HostBufferCharge&) = delete;
    ~HostBufferCharge();
    // Charge exactly `amount` bytes for this buffer.
    void set(uint64_t amount) noexcept;
    uint64_t amount() const noexcept { return amount_; }
    struct Budget {
        bool exhausted = false;
        uint64_t charged = 0, limit = 0;
    };
    Budget budget() const noexcept;
    // The shared budget of `endpoint` with `limit`, read without a charge,
    // as a buffer that holds no bytes reads it.
    static Budget budget(Endpoint& endpoint, uint64_t limit) noexcept;
private:
    EndpointLease endpoint_;
    uint64_t limit_;
    uint64_t amount_ = 0;
};

// Tags are heap-owned by the CQ after successful submission. Register only at
// handoff, not during fallible preparation. Diagnostics outlives every tag.
// Handlers only update internal state/enqueue work; host callbacks run elsewhere.
// onComplete must not destroy its tag; the CQ owner retires it after return.
struct CompletionTag {
    using Clock = std::chrono::steady_clock;
    CompletionTag() = default;
    CompletionTag(const CompletionTag&) = delete;
    CompletionTag& operator=(const CompletionTag&) = delete;
    virtual ~CompletionTag();
    void submitted(DiagnosticsRegistry& diagnostics) noexcept;
    void complete(bool ok) noexcept;
private:
    DiagnosticsRegistry* diagnostics_ = nullptr;
    bool received_ = false, handling_ = false;
    virtual void onComplete(bool ok, Clock::time_point received) noexcept = 0;
};

template<class Tag> Tag* submitTag(std::unique_ptr<Tag>& tag, DiagnosticsRegistry& diagnostics) noexcept {
    tag->submitted(diagnostics);
    // Everything written to the tag before this point happens before its completion on the poller.
    LIVE_TSAN_RELEASE(tag.get());
    return tag.release();
}

class Runtime {
public:
    explicit Runtime(RuntimeOptions options);
    ~Runtime();
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;
    const RuntimeOptions options;
    const std::string instanceNonce;
    std::atomic<bool> closing{false};
    std::mutex mutex;
    std::unordered_map<OmniUsdLiveEndpoint, std::shared_ptr<Endpoint>> endpoints;
    std::unordered_map<std::string, OmniUsdLiveEndpoint> authorities;
    OmniUsdLiveJitterParams defaultJitter;
    // Outlives every producer, including request and authentication teardown.
    std::unique_ptr<DiagnosticsRegistry> diagnostics;
    std::unique_ptr<RequestRegistry> requests;
    std::unique_ptr<AuthRegistry> auth;
    std::unique_ptr<DrainRegistry> drains;
    std::unique_ptr<StatusRegistry> statuses;
    std::unique_ptr<SessionRegistry> sessions;
    std::unique_ptr<LayerAdmissionRegistry> layerAdmissions;

    OmniUsdLiveResult registerEndpoint(EndpointOptions options, OmniUsdLiveEndpoint* output);
    OmniUsdLiveResult unregisterEndpoint(OmniUsdLiveEndpoint handle);
    std::shared_ptr<Endpoint> findEndpoint(OmniUsdLiveEndpoint handle);
    EndpointLease acquireEndpoint(OmniUsdLiveEndpoint handle);
    void beginShutdown(); // Registry must be empty; caller synchronizes admission.
    bool waitForShutdown(uint32_t timeoutMs, ShutdownPhase* endingPhase = nullptr);
    grpc::CompletionQueue& queue() { return queue_; }
    // Test seam: the poller calls hook(context, tag) after each tag's
    // onComplete returns, while it still owns the tag. Both fields publish
    // and load with release/acquire, in hook-then-context order for the
    // poller and context-then-hook order for an arm, so a hook a racing poll
    // observes is never paired with a context that arm predates; the poll
    // may still pair a not-yet-updated hook with the new context, so hooks
    // must tolerate a context newer than expected, including null. Null
    // disables it.
    void setAfterCompleteForTest(void (*hook)(void*, CompletionTag&) noexcept, void* context) noexcept;
private:
    grpc::CompletionQueue queue_;
    std::atomic<void (*)(void*, CompletionTag&) noexcept> afterComplete_{nullptr};
    std::atomic<void*> afterCompleteContext_{nullptr};
    std::thread poller_;
    std::mutex exitMutex_;
    std::condition_variable exited_;
    bool pollerExited_ = false;
    void poll() noexcept;
};

// One runtime per final plugin, shared by all C/Python/USD callers. These helpers
// are private, never exposed through the export map or installed headers.
std::shared_ptr<Runtime> getRuntime();
OmniUsdLiveResult initialize(const OmniUsdLiveRuntimeOptions* options);
OmniUsdLiveResult shutdown(uint32_t timeoutMs);
uint64_t wallTimeMs();

} // namespace omni::usd::live
