// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "LiveRuntime.h"
#include "LiveDrain.h"
#include "LiveStatus.h"
#include "LiveDiagnostics.h"
#include "LiveSession.h"
#include "LiveLayerAdmission.h"
#include "LiveRequest.h"
#include "LiveAuth.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <climits>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>

namespace omni::usd::live {
namespace {
std::mutex globalMutex;
std::shared_ptr<Runtime> globalRuntime;
uint64_t runtimeOwners = 0;
std::atomic<uint64_t> nextEndpoint{1}; // Never reset across runtime lifetimes.

void diagnosticStatusCallbackFailure(void* context, OmniUsdLiveCallbackHandle callback, OmniUsdLiveEndpoint endpoint,
    OmniUsdLiveSession session, const OmniUsdLiveStatus& value) noexcept {
    static_cast<DiagnosticsRegistry*>(context)->emit(OMNI_USD_LIVE_RECORD_CALLBACK_FAILED, endpoint, value.subject_id,
        value.incident_id,
        {{DiagnosticKey::CallbackKind, static_cast<uint32_t>(CallbackKind::Status)},
            {DiagnosticKey::Callback, callback}, {DiagnosticKey::Session, session},
            {DiagnosticKey::Revision, value.revision}});
}

void diagnosticTransition(void* context, OmniUsdLiveEndpoint endpoint, OmniUsdLiveSession, std::string_view uri,
    const OmniUsdLiveStatus* previous, const OmniUsdLiveStatus& value, bool attached,
    const StatusTiming& timing) noexcept {
    auto& diagnostics = *static_cast<DiagnosticsRegistry*>(context);
    diagnostics.subjectTransition(value.subject_kind, previous ? previous->state : 0, value.state, attached, timing);
    const auto record = [&](uint32_t code) {
        diagnostics.emit(code, endpoint, value.subject_id, value.incident_id,
            {{DiagnosticKey::State, value.state}, {DiagnosticKey::Revision, value.revision},
                {DiagnosticKey::Result, value.result}, {DiagnosticKey::Operation, value.operation},
                {DiagnosticKey::PendingBytes, value.pending_bytes},
                {DiagnosticKey::PendingWrites, value.pending_write_count}},
            uri);
    };
    if (!previous && value.subject_kind == OMNI_USD_LIVE_SUBJECT_ENDPOINT)
        record(OMNI_USD_LIVE_RECORD_ENDPOINT_REGISTERED);
    if (timing.backpressureStarted) record(OMNI_USD_LIVE_RECORD_BACKPRESSURE_ENTERED);
    if (timing.backpressureCompleted)
        diagnostics.emit(OMNI_USD_LIVE_RECORD_BACKPRESSURE_EXITED, endpoint, value.subject_id, value.incident_id,
            {{DiagnosticKey::State, value.state}, {DiagnosticKey::Revision, value.revision},
                {DiagnosticKey::Result, value.result}, {DiagnosticKey::Operation, value.operation},
                {DiagnosticKey::PendingBytes, value.pending_bytes},
                {DiagnosticKey::PendingWrites, value.pending_write_count},
                {DiagnosticKey::DurationMs, timing.backpressureMs},
                {DiagnosticKey::Reason, static_cast<uint32_t>(timing.backpressureResolution)}},
            uri);
    if (value.state == OMNI_USD_LIVE_STATE_CLOSED) {
        if (!previous || previous->state != value.state) {
            if (value.subject_kind == OMNI_USD_LIVE_SUBJECT_ENDPOINT) record(OMNI_USD_LIVE_RECORD_ENDPOINT_CLOSED);
        }
        if (value.incident_id) diagnostics.retainClosedIncident(endpoint, uri, value);
        return;
    }
    if (!previous || previous->result != value.result) {
        if (value.result == OMNI_USD_LIVE_OUTCOME_UNKNOWN) record(OMNI_USD_LIVE_RECORD_OUTCOME_UNKNOWN);
        if (value.result == OMNI_USD_LIVE_OBJECT_REPLACED) record(OMNI_USD_LIVE_RECORD_OBJECT_REPLACED);
    }
    if (previous && previous->state == value.state) return;
    if (value.state == OMNI_USD_LIVE_STATE_RECONNECTING) {
        // Layer/session owners record actual attempts with local correlation
        // evidence. A status transition is a separate recovery-gate event;
        // it must not duplicate or stand in for transport lifecycle evidence.
        if (value.subject_kind != OMNI_USD_LIVE_SUBJECT_LAYER && value.subject_kind != OMNI_USD_LIVE_SUBJECT_SESSION)
            record(OMNI_USD_LIVE_RECORD_STREAM_DISCONNECTED);
        record(OMNI_USD_LIVE_RECORD_RECOVERY_GATE_CLOSED);
    } else if (value.state == OMNI_USD_LIVE_STATE_READY) {
        if (previous && previous->state == OMNI_USD_LIVE_STATE_RECONNECTING) {
            if (value.subject_kind != OMNI_USD_LIVE_SUBJECT_LAYER &&
                value.subject_kind != OMNI_USD_LIVE_SUBJECT_SESSION)
                record(OMNI_USD_LIVE_RECORD_STREAM_RESUMED);
            record(OMNI_USD_LIVE_RECORD_RECOVERY_GATE_OPENED);
        }
        if (value.subject_kind == OMNI_USD_LIVE_SUBJECT_LAYER) record(OMNI_USD_LIVE_RECORD_LAYER_READY);
    }
}

std::string nonce() {
    std::random_device random;
    constexpr char digits[] = "0123456789abcdef";
    std::string value;
    for (int i = 0; i < 4; ++i) {
        uint32_t bits = random();
        for (int shift = 28; shift >= 0; shift -= 4) value += digits[(bits >> shift) & 15];
    }
    return value;
}

bool sameJitter(const OmniUsdLiveJitterParams& first, const OmniUsdLiveJitterParams& second) {
    return first.delay_constant_ms == second.delay_constant_ms && first.delay_multiple == second.delay_multiple &&
        first.delay_maximum_ms == second.delay_maximum_ms;
}

// Input sizes: the complete record, a larger same-major record, or exactly a
// 1.0 prefix that ends before a group of appended fields, whose missing fields
// read as zero; a size that splits a field or a group is rejected.
template<class T> constexpr bool acceptedPrefixSize(size_t) noexcept { return false; }
template<> constexpr bool acceptedPrefixSize<OmniUsdLiveRuntimeOptions>(size_t size) noexcept {
    return size == offsetof(OmniUsdLiveRuntimeOptions, apply_lag_warning_threshold) ||
        size == offsetof(OmniUsdLiveRuntimeOptions, reconnect_initial_ms);
}
static_assert(sizeof(OmniUsdLiveRuntimeOptions) == 160,
    "extend the accepted runtime option input sizes for an appended field");

template<class T> void validateVersion(const T* record) {
    // A short caller allocation may contain only struct_size. Do not read the
    // next field until the advertised extent has been accepted.
    uint32_t size = 0;
    std::memcpy(&size, record, sizeof(size));
    if (size < sizeof(T) && !acceptedPrefixSize<T>(size)) throw std::invalid_argument("short ABI structure");
    if ((record->abi_version >> 16) != (OMNI_USD_LIVE_ABI_VERSION >> 16))
        throw OmniUsdLiveResult(OMNI_USD_LIVE_UNSUPPORTED);
}

std::string copyString(const char* value, size_t limit) {
    if (!value) return {};
    for (size_t i = 0; i <= limit; ++i) {
        if (!value[i]) return {value, i};
    }
    throw std::length_error("endpoint configuration string exceeds limit");
}
} // namespace

uint64_t wallTimeMs() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
            .count());
}

void CompletionTag::submitted(DiagnosticsRegistry& diagnostics) noexcept {
    assert(!diagnostics_ && !received_);
    diagnostics_ = &diagnostics;
    diagnostics_->submittedCqTag();
}
void CompletionTag::complete(bool ok) noexcept {
    assert(!received_);
    const auto received = Clock::now(); // Exclude metric-lock wait from RPC/Ping timing.
    received_ = true;
    handling_ = true;
    if (diagnostics_) diagnostics_->receivedCqTag(ok);
    onComplete(ok, received);
    handling_ = false;
    if (diagnostics_) diagnostics_->returnedCqTag();
}
CompletionTag::~CompletionTag() {
    assert(!handling_);
    if (diagnostics_) diagnostics_->destroyedCqTag(received_);
}

PendingBytes::PendingBytes(EndpointLease endpoint, uint64_t amount, uint64_t limit)
    : endpoint_(std::move(endpoint)), amount_(amount) {
    if (!endpoint_) throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_HANDLE);
    std::lock_guard<std::mutex> lock(endpoint_->mutex);
    if (endpoint_->pendingTransportBytes > limit || amount > limit - endpoint_->pendingTransportBytes) {
        ++endpoint_->pendingByteRefusals;
        endpoint_->diagnostics.admittedPendingBytes(amount, false);
        throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    }
    endpoint_->pendingTransportBytes += amount;
    refusals_ = endpoint_->pendingByteRefusals;
    endpoint_->diagnostics.admittedPendingBytes(amount, true);
}
namespace {
thread_local unsigned deferredReleaseDepth = 0;
thread_local std::vector<std::shared_ptr<BufferedEditCount>>* deferredReleaseWatchers = nullptr;
}
PendingBytes::DeferredReleaseNotices::DeferredReleaseNotices() noexcept { ++deferredReleaseDepth; }
PendingBytes::DeferredReleaseNotices::~DeferredReleaseNotices() {
    if (--deferredReleaseDepth) return;
    std::unique_ptr<std::vector<std::shared_ptr<BufferedEditCount>>> watchers(deferredReleaseWatchers);
    deferredReleaseWatchers = nullptr;
    if (watchers)
        for (const auto& watcher : *watchers) watcher->released();
}
namespace {
// Under the endpoint mutex: the live pending-byte watchers, pruning expired ones.
std::vector<std::shared_ptr<BufferedEditCount>> liveWatchers(Endpoint& endpoint) noexcept {
    std::vector<std::shared_ptr<BufferedEditCount>> watchers;
    try {
        auto& list = endpoint.pendingByteWatchers;
        for (auto it = list.begin(); it != list.end();) {
            if (auto watcher = it->lock()) {
                watchers.push_back(std::move(watcher));
                ++it;
            } else it = list.erase(it);
        }
    } catch (...) {} // Allocation failure skips this retry event; the next release reports again.
    return watchers;
}
// Outside the endpoint mutex: report a release to `watchers`, or defer the
// report to the outermost DeferredReleaseNotices scope on this thread.
void reportRelease(std::vector<std::shared_ptr<BufferedEditCount>>&& watchers) noexcept {
    if (deferredReleaseDepth) {
        try {
            if (!deferredReleaseWatchers)
                deferredReleaseWatchers = new std::vector<std::shared_ptr<BufferedEditCount>>();
            for (auto& watcher : watchers) deferredReleaseWatchers->push_back(std::move(watcher));
        } catch (...) {} // Allocation failure skips this retry event; the next release reports again.
        return;
    }
    // Outside the endpoint mutex: a blocked watcher requests a host flush.
    for (const auto& watcher : watchers) watcher->released();
}
}
HostBufferCharge::HostBufferCharge(EndpointLease endpoint, uint64_t limit)
    : endpoint_(std::move(endpoint)), limit_(limit) {
    if (!endpoint_) throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_HANDLE);
}
HostBufferCharge::~HostBufferCharge() { set(0); }
void HostBufferCharge::set(uint64_t amount) noexcept {
    if (amount == amount_) return;
    std::vector<std::shared_ptr<BufferedEditCount>> watchers;
    {
        std::lock_guard<std::mutex> lock(endpoint_->mutex);
        endpoint_->hostBufferBytes = endpoint_->hostBufferBytes - amount_ + amount;
        if (amount < amount_) watchers = liveWatchers(*endpoint_);
        amount_ = amount;
    }
    reportRelease(std::move(watchers));
}
HostBufferCharge::Budget HostBufferCharge::budget() const noexcept { return budget(*endpoint_.operator->(), limit_); }
HostBufferCharge::Budget HostBufferCharge::budget(Endpoint& endpoint, uint64_t limit) noexcept {
    std::lock_guard<std::mutex> lock(endpoint.mutex);
    return {endpoint.hostBufferBytes >= limit, endpoint.hostBufferBytes, limit};
}
PendingBytes::~PendingBytes() {
    if (!endpoint_) return;
    std::vector<std::shared_ptr<BufferedEditCount>> watchers;
    {
        std::lock_guard<std::mutex> lock(endpoint_->mutex);
        endpoint_->pendingTransportBytes -= amount_;
        endpoint_->diagnostics.releasedPendingBytes(amount_);
        if (quiet_ && endpoint_->pendingByteRefusals == refusals_) return;
        watchers = liveWatchers(*endpoint_);
    }
    reportRelease(std::move(watchers));
}

RuntimeOptions RuntimeOptions::copy(const OmniUsdLiveRuntimeOptions* supplied) {
    RuntimeOptions result;
    // Read only the caller's extent; an accepted prefix leaves later fields zero.
    OmniUsdLiveRuntimeOptions extent{};
    const OmniUsdLiveRuntimeOptions* input = nullptr;
    if (supplied) {
        validateVersion(supplied);
        uint32_t size = 0;
        std::memcpy(&size, supplied, sizeof(size));
        std::memcpy(&extent, supplied, std::min<size_t>(size, sizeof(extent)));
        input = &extent;
    }
    result.values.struct_size = sizeof(result.values);
    result.values.abi_version = OMNI_USD_LIVE_ABI_VERSION;
#define COPY_DEFAULT(name, defaultValue) result.values.name = input && input->name ? input->name : defaultValue;
    LIVE_OPTION_DEFAULTS(COPY_DEFAULT)
#undef COPY_DEFAULT
    result.values.diagnostic_min_severity = input ? input->diagnostic_min_severity :
                                                    uint32_t(OMNI_USD_LIVE_SEVERITY_INFO);
    result.values.diagnostic_allowed_flags = input ? input->diagnostic_allowed_flags : 0;
    result.values.flags = input ? input->flags : 0;
    result.jitter = {sizeof(OmniUsdLiveJitterParams), OMNI_USD_LIVE_ABI_VERSION, 10, 2, 1000};
    if (input && input->default_jitter) {
        validateVersion(input->default_jitter);
        result.jitter.delay_constant_ms = input->default_jitter->delay_constant_ms;
        result.jitter.delay_multiple = input->default_jitter->delay_multiple;
        result.jitter.delay_maximum_ms = input->default_jitter->delay_maximum_ms;
    }
    if (result.values.max_update_bytes > INT_MAX || result.values.callback_worker_count > 64 ||
        result.values.max_pending_bytes_per_endpoint < result.values.max_update_bytes ||
        result.values.retry_initial_ms > result.values.retry_maximum_ms ||
        result.values.reconnect_initial_ms > result.values.reconnect_maximum_ms ||
        result.values.diagnostic_min_severity > OMNI_USD_LIVE_SEVERITY_ERROR ||
        (result.values.diagnostic_allowed_flags &
            ~uint32_t(OMNI_USD_LIVE_DIAGNOSTIC_RESOURCE_NAMES | OMNI_USD_LIVE_DIAGNOSTIC_SERVER_DETAIL)) ||
        (result.values.flags & ~uint32_t(OMNI_USD_LIVE_RUNTIME_DISABLE_DIAGNOSTIC_HISTORY)))
        throw std::invalid_argument("invalid runtime limits");
    return result;
}

bool RuntimeOptions::operator==(const RuntimeOptions& other) const {
#define COMPARE(name, defaultValue) if (values.name != other.values.name) return false;
    LIVE_OPTION_DEFAULTS(COMPARE)
#undef COMPARE
    return values.diagnostic_min_severity == other.values.diagnostic_min_severity &&
        values.diagnostic_allowed_flags == other.values.diagnostic_allowed_flags &&
        values.flags == other.values.flags && sameJitter(jitter, other.jitter);
}

EndpointOptions EndpointOptions::copy(const OmniUsdLiveEndpointOptions* input, const RuntimeOptions& runtime) {
    validateVersion(input);
    EndpointOptions result;
    result.authority = parseAuthority(copyString(input->authority, kMaxUriBytes));
    result.tlsServerName = copyString(input->tls_server_name, 253);
    result.caPem = copyString(input->ca_pem, 1024 * 1024);
    result.initialToken = copyString(input->access_token, 65536);
    for (unsigned char c : result.initialToken) {
        if (c <= 0x20 || c >= 0x7F) throw std::invalid_argument("invalid bearer token");
    }
    result.expiresAtMs = input->expires_at_ms;
    result.authWaitMs = input->auth_wait_timeout_ms ? input->auth_wait_timeout_ms :
                                                      runtime.values.default_auth_wait_timeout_ms;
    if (input->transport_security > OMNI_USD_LIVE_TRANSPORT_PLAINTEXT)
        throw std::invalid_argument("unknown transport security");
    result.secure = input->transport_security == OMNI_USD_LIVE_TRANSPORT_TLS ||
        (input->transport_security == OMNI_USD_LIVE_TRANSPORT_DEFAULT && !result.authority.loopback);
    if (!result.tlsServerName.empty()) {
        auto name = parseAuthority(result.tlsServerName);
        if (name.ipLiteral || result.tlsServerName.find(':') != std::string::npos)
            throw std::invalid_argument("TLS server name must be a DNS name");
        result.tlsServerName = name.host;
        if (!result.authority.ipLiteral && name.host != result.authority.host)
            throw std::invalid_argument("DNS TLS name must match dialed host");
    }
    if (result.secure && result.authority.ipLiteral && result.tlsServerName.empty())
        throw std::invalid_argument("IP TLS endpoint requires a DNS server name");
    if (!result.authority.ipLiteral) result.tlsServerName = result.authority.host;
    return result;
}

bool EndpointOptions::sameConfiguration(const EndpointOptions& other) const {
    // Token/expiry are initial values only. Duplicate registration cannot rotate
    // credentials or undo a newer refresh generation.
    return authority.canonical == other.authority.canonical && tlsServerName == other.tlsServerName &&
        caPem == other.caPem && secure == other.secure && authWaitMs == other.authWaitMs;
}

Endpoint::Endpoint(EndpointOptions supplied, OmniUsdLiveEndpoint value, std::string id,
    OmniUsdLiveJitterParams jitterValue, uint32_t maxUpdateBytes, DiagnosticsRegistry& metrics)
    : options(std::move(supplied)), handle(value), diagnostics(metrics), writerId(std::move(id)),
      token(std::move(options.initialToken)), tokenGeneration(token.empty() ? 0 : 1), expiresAtMs(options.expiresAtMs),
      expiresAtSteady(steadyExpiry(options.expiresAtMs, wallTimeMs(), std::chrono::steady_clock::now())),
      jitter(jitterValue) {
    status.struct_size = sizeof(status);
    status.abi_version = OMNI_USD_LIVE_ABI_VERSION;
    status.subject_kind = OMNI_USD_LIVE_SUBJECT_ENDPOINT;
    status.subject_id = handle;
    status.revision = 1;
    status.changed_at_ms = wallTimeMs();
    status.state = token.empty() ? OMNI_USD_LIVE_STATE_AUTH_WAIT : OMNI_USD_LIVE_STATE_OPENING;
    status.result = token.empty() ? OMNI_USD_LIVE_UNAUTHENTICATED : OMNI_USD_LIVE_OK;
    status.operation = OMNI_USD_LIVE_OPERATION_OPEN;
    status.recovery_action = token.empty() ? OMNI_USD_LIVE_RECOVERY_REFRESH_AUTH : OMNI_USD_LIVE_RECOVERY_NONE;
    status.progress_known = 1;
    grpc::ChannelArguments arguments;
    arguments.SetUserAgentPrefix("omni-usd-live/1");
    arguments.SetInt(GRPC_ARG_KEEPALIVE_TIME_MS, 10000);
    arguments.SetInt(GRPC_ARG_KEEPALIVE_TIMEOUT_MS, 10000);
    arguments.SetInt(GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS, 1);
    arguments.SetInt(GRPC_ARG_ENABLE_RETRIES, 0);
    // Read-layer frames and finite responses have smaller per-method caps.
    const auto messageLimit = static_cast<int>(wireLimit(maxUpdateBytes));
    arguments.SetMaxReceiveMessageSize(messageLimit);
    arguments.SetMaxSendMessageSize(messageLimit);
    // Bound a single read frame independently of a potentially much larger
    // assembled update, and each finite response at its reserved allowance.
    // Methods not listed retain the channel message limit.
    const auto methodLimit = [](std::initializer_list<const char*> methods, uint64_t bytes) {
        std::string names;
        for (const auto* method : methods)
            names += std::string(names.empty() ? "" : ",") +
                "{\"service\":\"nvidia.omniverse.live.v1alpha.LiveService\",\"method\":\"" + method + "\"}";
        return "{\"name\":[" + names + "],\"maxResponseMessageBytes\":" + std::to_string(bytes) + "}";
    };
    arguments.SetInt(GRPC_ARG_SERVICE_CONFIG_DISABLE_RESOLUTION, 1);
    arguments.SetServiceConfigJSON("{\"methodConfig\":[" +
        methodLimit({"SubscribeReadLayer"}, layerReadFrameLimit(maxUpdateBytes)) + "," +
        methodLimit({"ListSessions"}, finiteResponseLimit(maxUpdateBytes)) + "," +
        methodLimit({"CreateSession"}, kCreateSessionResponseBytes) + "," +
        methodLimit({"StatLayer"}, kStatLayerResponseBytes) + "," +
        methodLimit({"DeleteSession", "DeleteLayer", "Ping"}, kFixedResponseBytes) + "]}");
    std::shared_ptr<grpc::ChannelCredentials> credentials;
    if (options.secure) {
        grpc::SslCredentialsOptions tls;
        tls.pem_root_certs = options.caPem;
        if (options.authority.ipLiteral) arguments.SetSslTargetNameOverride(options.tlsServerName);
        credentials = grpc::SslCredentials(tls);
    } else {
        credentials = grpc::InsecureChannelCredentials();
    }
    channel = grpc::CreateCustomChannel(options.authority.canonical, credentials, arguments);
    stub = nvidia::omniverse::live::v1alpha::LiveService::NewStub(channel);
}

EndpointLease::EndpointLease(std::shared_ptr<Endpoint> endpoint) : endpoint_(std::move(endpoint)) {
    if (endpoint_) {
        // The guard's mutex must outlive its unlock, even if this was the last
        // reference to an already-closed endpoint.
        auto retained = endpoint_;
        std::lock_guard<std::mutex> lock(retained->mutex);
        if (retained->closed) {
            endpoint_.reset();
        } else {
            if (retained->dependents == std::numeric_limits<uint64_t>::max())
                throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
            ++retained->dependents;
        }
    }
}

void EndpointLease::release() noexcept {
    if (endpoint_) {
        // Keep the object alive until after unlocking its mutex.
        auto endpoint = std::move(endpoint_);
        std::lock_guard<std::mutex> lock(endpoint->mutex);
        --endpoint->dependents;
    }
}

EndpointLease& EndpointLease::operator=(EndpointLease&& other) noexcept {
    if (this != &other) {
        release();
        endpoint_ = std::move(other.endpoint_);
    }
    return *this;
}
EndpointLease::~EndpointLease() { release(); }

Runtime::Runtime(RuntimeOptions configuration)
    : options(std::move(configuration)), instanceNonce(nonce()), defaultJitter(options.jitter) {
    // Start only after every member, including the exit condition, exists.
    try {
        auto id = std::stoull(instanceNonce.substr(0, 16), nullptr, 16);
        diagnostics = std::make_unique<DiagnosticsRegistry>(options.values, id ? id : 1);
        drains = std::make_unique<DrainRegistry>(options.values.max_status_subjects);
        requests = std::make_unique<RequestRegistry>(*this, options.values.max_requests,
            options.values.callback_worker_count);
        statuses = std::make_unique<StatusRegistry>(requests->executor(), options.values.max_status_subjects,
            options.values.max_status_observers, diagnosticTransition, diagnostics.get(), nullptr, nullptr,
            diagnosticStatusCallbackFailure);
        auth = std::make_unique<AuthRegistry>(*this);
        sessions = std::make_unique<SessionRegistry>(*this);
        layerAdmissions = std::make_unique<LayerAdmissionRegistry>(*this, options.values.max_status_subjects);
        poller_ = std::thread([this] { poll(); });
    } catch (...) {
        // This runtime has not been published and has no submitted work. gRPC
        // still requires an empty queue to be shut down and drained before its
        // destructor, including when thread creation/allocation failed.
        queue_.Shutdown();
        void* tag = nullptr;
        bool ok = false;
        while (queue_.Next(&tag, &ok)) {}
        throw;
    }
}

Runtime::~Runtime() {
    beginShutdown();
    if (poller_.joinable()) poller_.join();
}

void Runtime::poll() noexcept {
    void* tag = nullptr;
    bool ok = false;
    while (queue_.Next(&tag, &ok)) {
        LIVE_TSAN_ACQUIRE(tag);
        std::unique_ptr<CompletionTag> owned(static_cast<CompletionTag*>(tag));
        // Acquire pairs with the release stores in setAfterCompleteForTest: a
        // hook observed here sees the context store that preceded its own.
        const auto afterComplete = afterComplete_.load(std::memory_order_acquire);
        void* const afterCompleteContext = afterComplete ? afterCompleteContext_.load(std::memory_order_acquire) :
                                                           nullptr;
        owned->complete(ok);
        if (afterComplete) afterComplete(afterCompleteContext, *owned);
    }
    {
        std::lock_guard<std::mutex> lock(exitMutex_);
        pollerExited_ = true;
    }
    exited_.notify_all();
}

void Runtime::setAfterCompleteForTest(void (*hook)(void*, CompletionTag&) noexcept, void* context) noexcept {
    if (!hook) afterComplete_.store(nullptr, std::memory_order_release);
    afterCompleteContext_.store(context, std::memory_order_release);
    if (hook) afterComplete_.store(hook, std::memory_order_release);
}

void Runtime::beginShutdown() {
    if (!closing.exchange(true)) {
        sessions->beginShutdown();
        diagnostics->beginShutdown();
        statuses->beginShutdown();
        auth->beginShutdown();
        requests->beginShutdown();
        queue_.Shutdown();
    }
}

bool Runtime::waitForShutdown(uint32_t timeoutMs, ShutdownPhase* endingPhase) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
    const auto phase = [&](ShutdownPhase value) {
        if (endingPhase) *endingPhase = value;
    };
    phase(ShutdownPhase::CompletionQueue);
    std::unique_lock<std::mutex> lock(exitMutex_);
    if (!exited_.wait_until(lock, deadline, [&] { return pollerExited_; })) return false;
    lock.unlock();
    phase(ShutdownPhase::Sessions);
    if (!sessions->waitForShutdown(deadline)) return false;
    phase(ShutdownPhase::Authentication);
    if (!auth->waitForShutdown(deadline)) return false;
    phase(ShutdownPhase::Status);
    if (!statuses->waitForShutdown(deadline)) return false;
    phase(ShutdownPhase::Requests);
    if (!requests->waitForShutdown(deadline)) return false;
    phase(ShutdownPhase::Diagnostics);
    if (!diagnostics->waitForShutdown(deadline)) return false;
    phase(ShutdownPhase::Complete);
    return true;
}

OmniUsdLiveResult Runtime::registerEndpoint(EndpointOptions configuration, OmniUsdLiveEndpoint* output) {
    OmniUsdLiveJitterParams jitter;
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (closing) return OMNI_USD_LIVE_BUSY;
        jitter = defaultJitter;
    }
    uint64_t ordinal = nextEndpoint.load();
    do {
        if (ordinal >= (UINT64_C(1) << 56)) return OMNI_USD_LIVE_RESOURCE_EXHAUSTED;
    } while (!nextEndpoint.compare_exchange_weak(ordinal, ordinal + 1));
    const auto handle = (UINT64_C(1) << 56) | ordinal;
    auto candidate = std::make_shared<Endpoint>(std::move(configuration), handle,
        instanceNonce + ":" + std::to_string(ordinal), jitter, options.values.max_update_bytes, *diagnostics);
    std::lock_guard<std::mutex> lock(mutex);
    if (closing) return OMNI_USD_LIVE_BUSY;
    auto previous = authorities.find(candidate->options.authority.canonical);
    if (previous != authorities.end()) {
        auto existing = endpoints.at(previous->second);
        std::lock_guard<std::mutex> endpointLock(existing->mutex);
        if (!existing->options.sameConfiguration(candidate->options)) return OMNI_USD_LIVE_ALREADY_EXISTS;
        if (existing->owners == std::numeric_limits<uint64_t>::max()) return OMNI_USD_LIVE_RESOURCE_EXHAUSTED;
        ++existing->owners;
        *output = existing->handle;
        return OMNI_USD_LIVE_OK;
    }
    if (endpoints.size() >= options.values.max_status_subjects) return OMNI_USD_LIVE_RESOURCE_EXHAUSTED;
    // Roll back the first insertion if allocation of the second fails.
    endpoints.emplace(handle, candidate);
    try {
        authorities.emplace(candidate->options.authority.canonical, handle);
        statuses->add(handle, 0, {}, candidate->status);
    } catch (...) {
        authorities.erase(candidate->options.authority.canonical);
        endpoints.erase(handle);
        throw;
    }
    *output = handle;
    auth->wake();
    return OMNI_USD_LIVE_OK;
}

OmniUsdLiveResult Runtime::unregisterEndpoint(OmniUsdLiveEndpoint handle) {
    std::shared_ptr<Endpoint> retained;
    {
        std::lock_guard<std::mutex> lock(mutex);
        const auto found = endpoints.find(handle);
        if (found == endpoints.end()) return OMNI_USD_LIVE_INVALID_HANDLE;
        retained = found->second;
        std::lock_guard<std::mutex> endpointLock(retained->mutex);
        if (retained->owners > 1) {
            --retained->owners;
            return OMNI_USD_LIVE_OK;
        }
        if (retained->dependents) return OMNI_USD_LIVE_BUSY;
        retained->closed = true;
        auth->closed(*retained, std::chrono::steady_clock::now());
        retained->status.state = OMNI_USD_LIVE_STATE_CLOSED;
        ++retained->status.revision;
        retained->status.changed_at_ms = wallTimeMs();
        statuses->remove(retained->status.subject_id);
        authorities.erase(retained->options.authority.canonical);
        endpoints.erase(found);
    }
    return OMNI_USD_LIVE_OK;
}

std::shared_ptr<Endpoint> Runtime::findEndpoint(OmniUsdLiveEndpoint handle) {
    std::lock_guard<std::mutex> lock(mutex);
    const auto found = endpoints.find(handle);
    return found == endpoints.end() ? nullptr : found->second;
}

EndpointLease Runtime::acquireEndpoint(OmniUsdLiveEndpoint handle) {
    std::lock_guard<std::mutex> lock(mutex);
    if (closing) return {};
    const auto found = endpoints.find(handle);
    return found == endpoints.end() ? EndpointLease{} : EndpointLease(found->second);
}

std::shared_ptr<Runtime> getRuntime() {
    std::lock_guard<std::mutex> lock(globalMutex);
    return globalRuntime;
}

OmniUsdLiveResult initialize(const OmniUsdLiveRuntimeOptions* input) {
    auto options = RuntimeOptions::copy(input);
    std::lock_guard<std::mutex> lock(globalMutex);
    if (globalRuntime) {
        if (globalRuntime->closing) return OMNI_USD_LIVE_BUSY;
        if (!(globalRuntime->options == options)) return OMNI_USD_LIVE_INVALID_ARGUMENT;
        if (runtimeOwners == std::numeric_limits<uint64_t>::max()) return OMNI_USD_LIVE_RESOURCE_EXHAUSTED;
        ++runtimeOwners;
        return OMNI_USD_LIVE_OK;
    }
    globalRuntime = std::make_shared<Runtime>(std::move(options));
    runtimeOwners = 1;
    return OMNI_USD_LIVE_OK;
}

OmniUsdLiveResult shutdown(uint32_t timeoutMs) {
    const auto started = std::chrono::steady_clock::now();
    std::shared_ptr<Runtime> runtime;
    auto phase = ShutdownPhase::ReferenceRelease;
    OmniUsdLiveResult result = OMNI_USD_LIVE_OK;
    bool quiescing = false;
    {
        std::lock_guard<std::mutex> lock(globalMutex);
        if (!globalRuntime) return OMNI_USD_LIVE_NOT_INITIALIZED;
        runtime = globalRuntime;
        if (runtimeOwners > 1) {
            --runtimeOwners;
        } else {
            std::lock_guard<std::mutex> registryLock(runtime->mutex);
            if (!runtime->endpoints.empty() || runtime->sessions->hasSessions() || runtime->statuses->hasObservers() ||
                runtime->diagnostics->hasSinks()) {
                phase = ShutdownPhase::Admission;
                result = OMNI_USD_LIVE_BUSY;
            } else {
                runtime->beginShutdown();
                quiescing = true;
            }
        }
    }
    if (quiescing) {
        if (!runtime->waitForShutdown(timeoutMs, &phase)) result = OMNI_USD_LIVE_TIMEOUT;
        else {
            std::lock_guard<std::mutex> lock(globalMutex);
            if (globalRuntime == runtime) {
                globalRuntime.reset();
                runtimeOwners = 0;
            }
        }
    }
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started).count();
    // The pin was selected under globalMutex. Never attribute a concurrent
    // shutdown/reinitialize race to a newly installed runtime.
    runtime->diagnostics->observeShutdown(phase, result, static_cast<uint64_t>(elapsed));
    return result;
}
} // namespace omni::usd::live
