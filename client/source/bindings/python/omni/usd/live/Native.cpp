// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// This module links the one public plugin DSO. No private runtime/USD implementation
// is linked here, and C callbacks never access Python or acquire its GIL.
#include "Host.h"
#include <omni/usd/live/Live.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#if defined(_WIN32)
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    include <windows.h>
#else
#    include <dlfcn.h>
#endif
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace py = pybind11;
namespace {
using Result = OmniUsdLiveResult;
namespace code {
constexpr Result BUFFER_TOO_SMALL = OMNI_USD_LIVE_BUFFER_TOO_SMALL;
constexpr Result BUSY = OMNI_USD_LIVE_BUSY;
constexpr Result INTERNAL_ERROR = OMNI_USD_LIVE_INTERNAL_ERROR;
constexpr Result INVALID_HANDLE = OMNI_USD_LIVE_INVALID_HANDLE;
constexpr Result OK = OMNI_USD_LIVE_OK;
constexpr Result RESOURCE_EXHAUSTED = OMNI_USD_LIVE_RESOURCE_EXHAUSTED;
constexpr Result TIMEOUT = OMNI_USD_LIVE_TIMEOUT;
constexpr Result CANCELLED = OMNI_USD_LIVE_CANCELLED;
constexpr Result OUTCOME_UNKNOWN = OMNI_USD_LIVE_OUTCOME_UNKNOWN;
constexpr Result INVALID_ARGUMENT = OMNI_USD_LIVE_INVALID_ARGUMENT;
}
constexpr std::size_t OwnerLimit = 8192;
constexpr uint64_t ReplyByteLimit = 16 * 1024 * 1024;
constexpr uint64_t TotalReplyByteLimit = 64 * 1024 * 1024;
constexpr uint64_t ReplyRecordLimit = 65536;
std::atomic<uint64_t> retainedReplyBytes{0};
template<class T> T initRecord() {
    T value{};
    value.struct_size = sizeof(T);
    value.abi_version = OMNI_USD_LIVE_ABI_VERSION;
    return value;
}
void text(const std::string& value) {
    if (value.find('\0') != std::string::npos) throw py::value_error("Embedded NUL is not permitted");
}
void keys(const py::dict& input, std::initializer_list<const char*> allowed) {
    for (auto item : input) {
        const auto name = py::cast<std::string>(item.first);
        bool found = false;
        for (const auto* key : allowed)
            if (name == key) {
                found = true;
                break;
            }
        if (!found) throw py::value_error("Unknown option: " + name);
    }
}
template<class T> void option(const py::dict& input, const char* name, T& out) {
    if (input.contains(name)) out = py::cast<T>(input[name]);
}
std::string stringOption(const py::dict& input, const char* name) {
    auto result = input.contains(name) ? py::cast<std::string>(input[name]) : std::string{};
    text(result);
    return result;
}

struct Cleanup {
    std::atomic<bool> abandoned{false};
    std::size_t slot = OwnerLimit;
    virtual ~Cleanup() = default;
    virtual bool poll() noexcept = 0;
};
class Owners {
public:
    static Owners& get() {
        static Owners* value = new Owners;
        return *value;
    }
    bool add(const std::shared_ptr<Cleanup>& state) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (std::size_t i = 0; i != states_.size(); ++i)
            if (!states_[i]) {
                state->slot = i;
                states_[i] = state;
                return true;
            }
        return false;
    }
    void retire(const std::shared_ptr<Cleanup>& state) noexcept {
        std::shared_ptr<Cleanup> displaced;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (state->slot < states_.size() && states_[state->slot] == state)
                displaced = std::move(states_[state->slot]);
        }
        cv_.notify_all();
    }
    void abandon(const std::shared_ptr<Cleanup>& state) noexcept { requestCleanup(state.get()); }
    void requestCleanup(Cleanup* state) noexcept {
        {
            // Pair the predicate transition with wait's mutex: a notification
            // between a false predicate and the atomic wait must not be lost.
            std::lock_guard<std::mutex> lock(mutex_);
            if (state->slot < states_.size() && states_[state->slot].get() == state)
                state->abandoned.store(true, std::memory_order_release);
        }
        cv_.notify_all();
    }
    Result drain(uint32_t timeout) noexcept {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout);
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            bool pending = false;
            for (const auto& state : states_)
                if (state && state->abandoned.load(std::memory_order_acquire)) {
                    pending = true;
                    break;
                }
            if (!pending) return code::OK;
            if (std::chrono::steady_clock::now() >= deadline) return code::TIMEOUT;
            cv_.wait_until(lock, deadline);
        }
    }
private:
    Owners() {
        std::thread([this] { run(); }).detach();
    }
    void run() noexcept {
        std::array<std::shared_ptr<Cleanup>, OwnerLimit> pending{};
        for (;;) {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                const auto abandoned = [this] {
                    for (const auto& state : states_)
                        if (state && state->abandoned.load(std::memory_order_acquire)) return true;
                    return false;
                };
                if (abandoned()) cv_.wait_for(lock, std::chrono::milliseconds(10));
                else cv_.wait(lock, abandoned);
                for (std::size_t i = 0; i != states_.size(); ++i)
                    if (states_[i] && states_[i]->abandoned.load(std::memory_order_acquire)) pending[i] = states_[i];
            }
            for (auto& state : pending)
                if (state) {
                    if (state->poll()) retire(state);
                    state.reset(); // No capture destruction while registry mutex is held.
                }
        }
    }
    std::mutex mutex_;
    std::condition_variable cv_;
    std::array<std::shared_ptr<Cleanup>, OwnerLimit> states_{};
};

struct RefState final : Cleanup {
    enum class Kind { Runtime, Endpoint };
    explicit RefState(Kind k) : kind(k) {}
    Kind kind;
    std::mutex mutex;
    uint64_t handle = 0;
    bool owned = false;
    Result close(uint32_t timeout) noexcept {
        std::lock_guard<std::mutex> lock(mutex);
        if (!owned) return code::INVALID_HANDLE;
        const auto result = kind == Kind::Runtime ? omniUsdLiveShutdown(timeout) :
                                                    omniUsdLiveUnregisterEndpoint(handle, timeout);
        if (result == code::OK) {
            owned = false;
            handle = 0;
        }
        return result;
    }
    bool poll() noexcept override {
        const auto result = close(0);
        return result == code::OK || result == code::INVALID_HANDLE;
    }
};
class Runtime {
public:
    explicit Runtime(std::shared_ptr<RefState> value) : state(std::move(value)) {}
    ~Runtime() { Owners::get().abandon(state); }
    Result close(uint32_t timeout) {
        Result result;
        {
            py::gil_scoped_release release;
            result = state->close(timeout);
        }
        if (result == code::OK) Owners::get().retire(state);
        return result;
    }
    Result cleanup(uint32_t timeout) {
        py::gil_scoped_release release;
        return Owners::get().drain(timeout);
    }
    std::shared_ptr<RefState> state;
};
class Endpoint {
public:
    explicit Endpoint(std::shared_ptr<RefState> value) : state(std::move(value)) {}
    ~Endpoint() { Owners::get().abandon(state); }
    uint64_t handle() const {
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(state->mutex);
        return state->handle;
    }
    Result close(uint32_t timeout) {
        Result result;
        {
            py::gil_scoped_release release;
            result = state->close(timeout);
        }
        if (result == code::OK) Owners::get().retire(state);
        return result;
    }
    std::shared_ptr<RefState> state;
};

struct SessionInfo {
    std::string id, uri, name, creator;
    uint64_t created = 0;
};
struct RequestState final : Cleanup {
    enum class Kind { List, Create, Stat, ResultOnly };
    explicit RequestState(Kind value) : kind(value) {}
    ~RequestState() override { retainedReplyBytes.fetch_sub(payloadCharge, std::memory_order_relaxed); }
    Kind kind;
    std::mutex mutex, payloadMutex;
    uint64_t handle = 0;
    bool delivered = false;
    Result delivery = code::BUSY;
    std::vector<SessionInfo> sessions;
    std::string id, uri;
    uint64_t created = 0, modified = 0;
    uint64_t payloadCharge = 0;
    Result outcome(Result native) {
        if (native != code::OK) return native;
        std::lock_guard<std::mutex> lock(payloadMutex);
        if (delivered) return delivery;
        if (kind == Kind::Create) return code::OUTCOME_UNKNOWN;
        if (kind == Kind::List || kind == Kind::Stat) return code::CANCELLED;
        return native;
    }
    Result release() noexcept {
        std::lock_guard<std::mutex> lock(mutex);
        if (!handle) return code::INVALID_HANDLE;
        const auto result = omniUsdLiveReleaseRequest(handle);
        if (result == code::OK) handle = 0;
        return result;
    }
    bool poll() noexcept override {
        const auto result = release();
        return result == code::OK || result == code::INVALID_HANDLE;
    }
};
const char* nonnull(const char* input) { return input ? input : ""; }
void countText(uint64_t& bytes, const char* input) {
    const auto count = input ? strnlen(input, ReplyByteLimit + 1) : 0;
    if (count >= ReplyByteLimit || bytes > ReplyByteLimit - count - 1) throw std::bad_alloc();
    bytes += count + 1;
}
void reserveBytes(uint64_t bytes) {
    if (bytes > TotalReplyByteLimit) throw std::bad_alloc();
    auto total = retainedReplyBytes.load(std::memory_order_relaxed);
    for (;;) {
        if (total > TotalReplyByteLimit - bytes) throw std::bad_alloc();
        if (retainedReplyBytes.compare_exchange_weak(total, total + bytes, std::memory_order_relaxed)) break;
    }
}
void reserveReply(RequestState& state, uint64_t bytes) {
    if (bytes > ReplyByteLimit) throw std::bad_alloc();
    reserveBytes(bytes);
    state.payloadCharge = bytes;
}
template<class Fill> void deliver(void* raw, Result result, Fill&& fill) noexcept {
    auto& state = *static_cast<RequestState*>(raw);
    std::lock_guard<std::mutex> lock(state.payloadMutex);
    if (state.delivered) return;
    try {
        if (result == code::OK) fill(state);
        state.delivery = result;
    } catch (const std::bad_alloc&) {
        state.delivery = state.kind == RequestState::Kind::Create ? code::OUTCOME_UNKNOWN : code::RESOURCE_EXHAUSTED;
    } catch (...) {
        state.delivery = state.kind == RequestState::Kind::Create ? code::OUTCOME_UNKNOWN : code::INTERNAL_ERROR;
    }
    if (state.delivery != code::OK) {
        std::vector<SessionInfo>().swap(state.sessions);
        std::string().swap(state.id);
        std::string().swap(state.uri);
        retainedReplyBytes.fetch_sub(state.payloadCharge, std::memory_order_relaxed);
        state.payloadCharge = 0;
    }
    state.delivered = true;
}
void OMNI_USD_LIVE_CALL listCallback(void* user, Result result, uint64_t count,
    const OmniUsdLiveSessionInfo* const* items) noexcept {
    deliver(user, result, [&](RequestState& state) {
        if (count > ReplyRecordLimit) throw std::bad_alloc();
        if (count && !items) throw std::runtime_error("Invalid session payload");
        uint64_t bytes = count * sizeof(SessionInfo);
        for (uint64_t i = 0; i != count; ++i) {
            if (!items[i]) throw std::runtime_error("Invalid session record");
            countText(bytes, items[i]->session_id);
            countText(bytes, items[i]->uri);
            countText(bytes, items[i]->name);
            countText(bytes, items[i]->created_by);
        }
        reserveReply(state, bytes);
        std::vector<SessionInfo> copied;
        copied.reserve(static_cast<std::size_t>(count));
        for (uint64_t i = 0; i != count; ++i) {
            if (!items[i]) throw std::runtime_error("Invalid session record");
            const auto& value = *items[i];
            copied.push_back({nonnull(value.session_id), nonnull(value.uri), nonnull(value.name),
                nonnull(value.created_by), value.created_time_ms});
        }
        state.sessions = std::move(copied);
    });
}
void OMNI_USD_LIVE_CALL createCallback(void* user, Result result, const OmniUsdLiveCreatedSession* value) noexcept {
    deliver(user, result, [&](RequestState& state) {
        if (!value) throw std::runtime_error("Missing create result");
        uint64_t bytes = 0;
        countText(bytes, value->session_id);
        countText(bytes, value->uri);
        reserveReply(state, bytes);
        state.id = nonnull(value->session_id);
        state.uri = nonnull(value->uri);
    });
}
void OMNI_USD_LIVE_CALL statCallback(void* user, Result result, const OmniUsdLiveLayerInfo* value) noexcept {
    deliver(user, result, [&](RequestState& state) {
        if (!value) throw std::runtime_error("Missing stat result");
        state.created = value->created_time_ms;
        state.modified = value->modified_time_ms;
    });
}
void OMNI_USD_LIVE_CALL resultCallback(void* user, Result result) noexcept {
    deliver(user, result, [](RequestState&) {});
}

class Request {
public:
    explicit Request(std::shared_ptr<RequestState> value) : state(std::move(value)) {}
    ~Request() { Owners::get().abandon(state); }
    uint64_t handle() const {
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(state->mutex);
        return state->handle;
    }
    Result wait(uint32_t timeout) {
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(state->mutex);
        return state->handle ? omniUsdLiveWait(state->handle, timeout) : code::INVALID_HANDLE;
    }
    Result cancel(uint32_t timeout) {
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(state->mutex);
        return state->handle ? omniUsdLiveCancel(state->handle, timeout) : code::INVALID_HANDLE;
    }
    Result release() {
        Result result;
        {
            py::gil_scoped_release release;
            result = state->release();
        }
        if (result == code::OK) Owners::get().retire(state);
        return result;
    }
    py::tuple status() const {
        Result query, result = code::BUSY;
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(state->mutex);
            query = state->handle ? omniUsdLiveGetRequestStatus(state->handle, &result) : code::INVALID_HANDLE;
        }
        return py::make_tuple(query, query == code::OK ? state->outcome(result) : result);
    }
    py::tuple nativeStatus() const {
        Result query, result = code::BUSY;
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(state->mutex);
            query = state->handle ? omniUsdLiveGetRequestStatus(state->handle, &result) : code::INVALID_HANDLE;
        }
        return py::make_tuple(query, result);
    }
    py::tuple error() const {
        auto value = initRecord<OmniUsdLiveRequestError>();
        Result result;
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(state->mutex);
            result = state->handle ? omniUsdLiveGetRequestError(state->handle, &value) : code::INVALID_HANDLE;
        }
        py::dict out;
        if (result == code::OK) {
            const auto nativeResult = value.result;
            value.result = state->outcome(value.result);
            if (nativeResult != value.result)
                std::strcpy(value.detail, "Python binding could not retain or deliver the successful native reply");
#define ERROR_FIELD(name) out[#name] = value.name
            ERROR_FIELD(result);
            ERROR_FIELD(operation);
            ERROR_FIELD(grpc_status_present);
            ERROR_FIELD(grpc_status);
            ERROR_FIELD(request_id);
            ERROR_FIELD(attempt_id);
            ERROR_FIELD(incident_id);
            out["detail"] = value.detail;
#undef ERROR_FIELD
        }
        return py::make_tuple(result, out);
    }
    py::tuple payload() const {
        // Wait polls actual native callback quiescence: cancellation may suppress
        // delivery entirely. It is never represented by a callback-only flag.
        Result query, terminal = code::BUSY;
        bool released = false;
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(state->mutex);
            released = state->handle == 0; // Only a successful quiescent release clears it.
            query = state->handle ? omniUsdLiveWait(state->handle, 0) : code::INVALID_HANDLE;
            if (query == code::OK) query = omniUsdLiveGetRequestStatus(state->handle, &terminal);
        }
        if (query == code::OK) terminal = state->outcome(terminal);
        if (query != code::OK && !released)
            return py::make_tuple(query == code::TIMEOUT ? code::BUSY : query, py::none());
        // Native Wait OK (or our successful Release) proves callback and capture
        // quiescence. The once-only payload is immutable thereafter and this
        // Request pins it. Publish through its mutex, then convert without any
        // native lock: Python allocations can run reentrant GC finalizers.
        { std::lock_guard<std::mutex> published(state->payloadMutex); }
        if (!state->delivered) return py::make_tuple(query == code::OK ? terminal : query, py::none());
        if (state->delivery != code::OK) return py::make_tuple(state->delivery, py::none());
        if (state->kind == RequestState::Kind::List) {
            py::list out;
            for (const auto& value : state->sessions) {
                py::dict item;
                item["session_id"] = value.id;
                item["uri"] = value.uri;
                item["name"] = value.name;
                item["created_by"] = value.creator;
                item["created_time_ms"] = value.created;
                out.append(item);
            }
            return py::make_tuple(state->delivery, out);
        }
        py::dict out;
        if (state->kind == RequestState::Kind::Create) {
            out["session_id"] = state->id;
            out["uri"] = state->uri;
        } else if (state->kind == RequestState::Kind::Stat) {
            out["created_time_ms"] = state->created;
            out["modified_time_ms"] = state->modified;
        } else return py::make_tuple(state->delivery, py::none());
        return py::make_tuple(state->delivery, out);
    }
    std::shared_ptr<RequestState> state;
};

template<class Submit> py::tuple submit(RequestState::Kind kind, Submit&& function) {
    auto& owners = Owners::get();
    auto state = std::make_shared<RequestState>(kind);
    auto request = std::make_shared<Request>(state); // All owners exist before submission.
    auto success = py::make_tuple(code::OK, request); // No Python allocations after native acceptance.
    if (!owners.add(state)) return py::make_tuple(code::RESOURCE_EXHAUSTED, py::none());
    Result result;
    {
        py::gil_scoped_release release;
        result = function(state.get(), &state->handle);
    }
    if (result != code::OK) {
        Owners::get().retire(state);
        return py::make_tuple(result, py::none());
    }
    return success;
}

py::dict statusDict(const OmniUsdLiveStatus& value) {
    py::dict out;
#define STATUS_FIELD(name) out[#name] = value.name
    STATUS_FIELD(subject_kind);
    STATUS_FIELD(state);
    STATUS_FIELD(subject_id);
    STATUS_FIELD(revision);
    STATUS_FIELD(changed_at_ms);
    STATUS_FIELD(result);
    STATUS_FIELD(operation);
    STATUS_FIELD(grpc_status_present);
    STATUS_FIELD(grpc_status);
    STATUS_FIELD(recovery_action);
    STATUS_FIELD(progress_known);
    STATUS_FIELD(incident_id);
    STATUS_FIELD(pending_write_count);
    STATUS_FIELD(pending_bytes);
    STATUS_FIELD(has_unconfirmed_edits);
    STATUS_FIELD(blocked_layers);
    STATUS_FIELD(failed_layers);
    STATUS_FIELD(blocked_sessions);
    STATUS_FIELD(failed_sessions);
    STATUS_FIELD(buffered_edit_count);
    STATUS_FIELD(read_sequence);
    STATUS_FIELD(applied_sequence);
#undef STATUS_FIELD
    return out;
}
template<class Function> py::tuple statusQuery(Function&& function) {
    auto value = initRecord<OmniUsdLiveStatus>();
    Result result;
    {
        py::gil_scoped_release release;
        result = function(&value);
    }
    return py::make_tuple(result, result == code::OK ? statusDict(value) : py::dict{});
}

struct OwnedSessionEvent {
    uint32_t kind = OMNI_USD_LIVE_EVENT_CONNECTION;
    Result result = code::OK;
    std::string connection, user, app, bytes;
    bool connectionPresent = false, userPresent = false, appPresent = false;
    uint64_t charge = 0;
};
struct SessionState final : Cleanup {
    std::mutex lifeMutex, eventMutex;
    std::condition_variable events;
    uint64_t handle = 0;
    bool owned = false;
    bool cleanupRetired = false;
    std::atomic<bool> admitted{false}, stop{false}, interrupted{false};
    std::atomic<bool> discardRequested{false};
    std::atomic<bool> consumerActive{false};
    std::atomic<Result> delivery{code::OK};
    std::vector<OwnedSessionEvent> ring;
    std::size_t head = 0, count = 0;
    uint64_t maxBytes = 0, queuedBytes = 0, ringCharge = 0;
    bool terminalPending = false, terminalDelivered = false, nativeStopped = false;
    ~SessionState() override { retainedReplyBytes.fetch_sub(ringCharge + queuedBytes, std::memory_order_relaxed); }
    void initialize(uint32_t capacity, uint64_t bytes) {
        const auto charge = static_cast<uint64_t>(capacity) * sizeof(OwnedSessionEvent);
        reserveBytes(charge);
        ringCharge = charge;
        ring.resize(capacity);
        maxBytes = bytes;
    }
    void interrupt() noexcept {
        {
            std::lock_guard<std::mutex> lock(eventMutex);
            interrupted.store(true, std::memory_order_release);
            stop.store(true, std::memory_order_release);
        }
        events.notify_all();
    }
    void clearQueue() noexcept {
        // Caller owns the consumer gate, so no Python conversion refers to a slot.
        std::lock_guard<std::mutex> lock(eventMutex);
        std::vector<OwnedSessionEvent>().swap(ring);
        retainedReplyBytes.fetch_sub(ringCharge + queuedBytes, std::memory_order_relaxed);
        ringCharge = queuedBytes = 0;
        head = count = 0;
        terminalPending = false;
    }
    void discard() noexcept {
        discardRequested.store(true, std::memory_order_release);
        interrupt();
        clearIfIdle();
    }
    void clearIfIdle() noexcept {
        if (acquireConsumer()) {
            clearQueue();
            releaseConsumer();
        }
        // An in-flight conversion clears on exit; never invalidate its slot.
    }
    bool acquireConsumer() noexcept {
        bool expected = false;
        return consumerActive.compare_exchange_strong(expected, true, std::memory_order_acquire);
    }
    void releaseConsumer() noexcept { consumerActive.store(false, std::memory_order_release); }
    Result leave(uint32_t timeout, bool cleanup = false) noexcept {
        {
            std::lock_guard<std::mutex> lock(eventMutex);
            stop.store(true, std::memory_order_release);
            nativeStopped = true;
        }
        events.notify_all();
        std::lock_guard<std::mutex> lock(lifeMutex);
        if (!owned) {
            // A bridge failure may already have quiesced the native reference.
            // Let its still-live Python owner acknowledge that cleanup once.
            if (!cleanup && cleanupRetired) {
                cleanupRetired = false;
                return code::OK;
            }
            return code::INVALID_HANDLE;
        }
        const auto result = omniUsdLiveLeaveSession(handle, timeout);
        if (result == code::OK) {
            handle = 0;
            owned = false;
            cleanupRetired = cleanup;
        }
        return result;
    }
    bool poll() noexcept override {
        // Failure callbacks can precede Join's return/output-handle publication.
        if (!admitted.load(std::memory_order_acquire)) return false;
        const auto result = leave(0, true);
        return result == code::OK || result == code::INVALID_HANDLE;
    }
    void failLocked(Result result) noexcept {
        if (delivery.load(std::memory_order_relaxed) != code::OK) return;
        delivery.store(result, std::memory_order_release);
        stop.store(true, std::memory_order_release);
        terminalPending = true; // An inline reserved slot, independent of ring capacity.
    }
    void receive(const OmniUsdLiveSessionEvent* value) noexcept {
        bool failed = false;
        {
            std::lock_guard<std::mutex> lock(eventMutex);
            if (stop.load(std::memory_order_acquire)) return;
            uint64_t reserved = 0;
            try {
                if (!value) throw std::runtime_error("Missing session event");
                if (count == ring.size() || value->size > maxBytes) throw std::bad_alloc();
                uint64_t charge = value->size;
                countText(charge, value->connection_id);
                countText(charge, value->user_name);
                countText(charge, value->app_name);
                if (charge > maxBytes || queuedBytes > maxBytes - charge) throw std::bad_alloc();
                if (value->size && !value->bytes) throw std::runtime_error("Missing message bytes");
                reserveBytes(charge);
                reserved = charge;
                OwnedSessionEvent copied;
                copied.kind = value->kind;
                copied.result = value->result;
                copied.connectionPresent = value->connection_id != nullptr;
                copied.userPresent = value->user_name != nullptr;
                copied.appPresent = value->app_name != nullptr;
                copied.connection = nonnull(value->connection_id);
                copied.user = nonnull(value->user_name);
                copied.app = nonnull(value->app_name);
                if (value->size)
                    copied.bytes.assign(static_cast<const char*>(value->bytes), static_cast<std::size_t>(value->size));
                copied.charge = charge;
                ring[(head + count) % ring.size()] = std::move(copied);
                ++count;
                queuedBytes += charge;
                reserved = 0;
            } catch (const std::bad_alloc&) {
                failLocked(code::RESOURCE_EXHAUSTED);
                failed = true;
            } catch (...) {
                failLocked(code::INTERNAL_ERROR);
                failed = true;
            }
            if (reserved) retainedReplyBytes.fetch_sub(reserved, std::memory_order_relaxed);
        }
        events.notify_all();
        if (failed) Owners::get().requestCleanup(this); // No native Leave or Python call on this callback stack.
    }
};
void OMNI_USD_LIVE_CALL sessionCallback(void* user, const OmniUsdLiveSessionEvent* value) noexcept {
    static_cast<SessionState*>(user)->receive(value);
}
py::dict sessionEventDict(const OwnedSessionEvent& value) {
    py::dict out;
    out["kind"] = value.kind;
    out["result"] = value.result;
    out["connection_id"] = value.connectionPresent ? py::cast(value.connection) : py::none();
    out["user_name"] = value.userPresent ? py::cast(value.user) : py::none();
    out["app_name"] = value.appPresent ? py::cast(value.app) : py::none();
    out["bytes"] = py::bytes(value.bytes);
    return out;
}
class Session {
public:
    explicit Session(std::shared_ptr<SessionState> value) : state(std::move(value)) {}
    ~Session() { Owners::get().abandon(state); }
    uint64_t handle() const {
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(state->lifeMutex);
        return state->handle;
    }
    Result leave(uint32_t timeout) {
        Result result;
        {
            py::gil_scoped_release release;
            result = state->leave(timeout);
            if (result == code::OK) state->discard();
        }
        if (result == code::OK) Owners::get().retire(state);
        return result;
    }
    void interrupt() {
        py::gil_scoped_release release;
        state->interrupt();
    }
    Result deliveryStatus() const noexcept { return state->delivery.load(std::memory_order_acquire); }
    py::tuple status() const {
        return statusQuery([&](OmniUsdLiveStatus* value) -> Result {
            std::lock_guard<std::mutex> lock(state->lifeMutex);
            return state->owned ? omniUsdLiveGetSessionStatus(state->handle, value) : code::INVALID_HANDLE;
        });
    }
    py::tuple next(uint32_t timeout) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout);
        if (!state->acquireConsumer()) return py::make_tuple(code::BUSY, py::none());
        struct Finish {
            SessionState& state;
            ~Finish() {
                py::gil_scoped_release release;
                // Unlock before checking the deferred-discard flag so an
                // explicit closer cannot miss our last active conversion.
                state.releaseConsumer();
                if (state.discardRequested.load(std::memory_order_acquire)) state.clearIfIdle();
            }
        } finish{*state};
        std::unique_lock<std::mutex> lock;
        bool canceled = false, queued = false, terminal = false, closed = false;
        std::size_t index = 0;
        {
            py::gil_scoped_release release;
            lock = std::unique_lock<std::mutex>(state->eventMutex);
            state->events.wait_until(lock, deadline, [&] {
                return state->count || state->terminalPending || state->nativeStopped ||
                    state->interrupted.load(std::memory_order_acquire);
            });
            canceled = state->interrupted.load(std::memory_order_acquire);
            queued = state->count != 0;
            terminal = state->terminalPending;
            closed = state->nativeStopped || state->terminalDelivered;
            index = state->head;
            lock.unlock();
        }
        if (canceled) return py::make_tuple(code::CANCELLED, py::none());
        if (queued) {
            auto& event = state->ring[index];
            // Materialize the whole Python return before committing consumption.
            // Only this consumer can pop; callbacks append to different slots.
            // No event mutex is held while Python allocates or may run GC.
            auto result = py::make_tuple(code::OK, sessionEventDict(event));
            {
                py::gil_scoped_release release;
                std::lock_guard<std::mutex> commit(state->eventMutex);
                const auto charge = event.charge;
                event = OwnedSessionEvent{};
                state->head = (state->head + 1) % state->ring.size();
                --state->count;
                state->queuedBytes -= charge;
                retainedReplyBytes.fetch_sub(charge, std::memory_order_relaxed);
            }
            return result;
        }
        if (terminal) {
            OwnedSessionEvent failure;
            failure.result = deliveryStatus();
            auto result = py::make_tuple(code::OK, sessionEventDict(failure));
            {
                py::gil_scoped_release release;
                std::lock_guard<std::mutex> commit(state->eventMutex);
                state->terminalPending = false;
                state->terminalDelivered = true;
            }
            return result;
        }
        return py::make_tuple(closed ? code::CANCELLED : code::TIMEOUT, py::none());
    }
    std::shared_ptr<SessionState> state;
};
py::tuple joinSession(const std::string& uri, const std::string& app, uint32_t capacity, uint64_t bytes) {
    text(uri);
    text(app);
    if (!capacity || capacity > 4096 || !bytes || bytes > ReplyByteLimit)
        return py::make_tuple(code::INVALID_ARGUMENT, py::none());
    auto& owners = Owners::get();
    auto state = std::make_shared<SessionState>();
    try {
        state->initialize(capacity, bytes);
    } catch (const std::bad_alloc&) { return py::make_tuple(code::RESOURCE_EXHAUSTED, py::none()); }
    auto session = std::make_shared<Session>(state);
    auto success = py::make_tuple(code::OK, session);
    if (!owners.add(state)) return py::make_tuple(code::RESOURCE_EXHAUSTED, py::none());
    auto options = initRecord<OmniUsdLiveJoinOptions>();
    options.app_name = app.c_str();
    Result result;
    {
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(state->lifeMutex);
        result = omniUsdLiveJoinSession(uri.c_str(), &options, state.get(), sessionCallback, &state->handle);
        state->owned = result == code::OK;
        state->admitted.store(true, std::memory_order_release);
    }
    if (result != code::OK) {
        owners.retire(state);
        return py::make_tuple(result, py::none());
    }
    return success;
}

// Python observers use owned C++ records only; no C callback enters Python.
struct OwnedDiagnosticField {
    char name[128]{}, text[513]{};
    uint32_t kind = 0;
    uint64_t unsignedValue = 0;
    int64_t signedValue = 0;
};
struct OwnedDiagnostic {
    OmniUsdLiveDiagnosticRecord record{};
    char name[128]{};
    std::array<OwnedDiagnosticField, 16> fields{};
    uint64_t charge = 0;
};
template<std::size_t N> bool boundedText(char (&out)[N], const char* input) noexcept {
    if (!input) {
        out[0] = '\0';
        return true;
    }
    const auto length = strnlen(input, N);
    if (length == N) return false;
    std::memcpy(out, input, length + 1);
    return true;
}
struct SubscriptionState final : Cleanup {
    enum class Kind { Auth, Status, Diagnostic };
    explicit SubscriptionState(Kind value) : kind(value) {}
    ~SubscriptionState() override { retainedReplyBytes.fetch_sub(ringCharge, std::memory_order_relaxed); }
    Kind kind;
    std::mutex lifeMutex, eventMutex;
    std::condition_variable events;
    uint64_t handle = 0;
    uint64_t endpoint = 0; // Borrowed identity retained by the C registration lease.
    std::atomic<bool> interrupted{false}, consuming{false};
    std::atomic<Result> delivery{code::OK};
    std::atomic<uint64_t> dropped{0};
    OmniUsdLiveAuthRequired auth{};
    uint64_t authVersion = 0;
    bool authPending = false;
    std::vector<OwnedDiagnostic> ring;
    std::size_t head = 0, count = 0;
    uint64_t ringCharge = 0, queuedBytes = 0, maxBytes = 0;
    void initialize(uint32_t records, uint64_t bytes) {
        const auto charge = static_cast<uint64_t>(records) * sizeof(OwnedDiagnostic);
        reserveBytes(charge);
        ringCharge = charge;
        ring.resize(records);
        maxBytes = bytes;
    }
    uint64_t currentHandle() {
        std::lock_guard<std::mutex> lock(lifeMutex);
        return handle;
    }
    void interrupt() noexcept {
        {
            std::lock_guard<std::mutex> lock(eventMutex);
            interrupted.store(true, std::memory_order_release);
        }
        events.notify_all();
    }
    Result close(uint32_t timeout) noexcept {
        interrupt();
        std::lock_guard<std::mutex> lock(lifeMutex);
        if (!handle) return code::INVALID_HANDLE;
        const auto result = omniUsdLiveUnregisterCallback(handle, timeout);
        if (result == code::OK) {
            handle = 0;
            // Consumers materialize stack snapshots; no ring slot is borrowed.
            std::lock_guard<std::mutex> eventLock(eventMutex);
            std::vector<OwnedDiagnostic>().swap(ring);
            retainedReplyBytes.fetch_sub(ringCharge, std::memory_order_relaxed);
            ringCharge = queuedBytes = 0;
            count = head = 0;
            authPending = false;
        }
        return result;
    }
    bool poll() noexcept override {
        const auto result = close(0);
        return result == code::OK || result == code::INVALID_HANDLE;
    }
    void lose() noexcept {
        auto value = dropped.load(std::memory_order_relaxed);
        while (value != UINT64_MAX && !dropped.compare_exchange_weak(value, value + 1, std::memory_order_relaxed)) {}
    }
    void fail(Result value) noexcept {
        {
            std::lock_guard<std::mutex> lock(eventMutex);
            Result expected = code::OK;
            delivery.compare_exchange_strong(expected, value, std::memory_order_release);
        }
        events.notify_all();
    }
};
void OMNI_USD_LIVE_CALL authObserver(void* user, const OmniUsdLiveAuthRequired* value) noexcept {
    auto& state = *static_cast<SubscriptionState*>(user);
    if (!value) {
        state.fail(code::INTERNAL_ERROR);
        return;
    }
    {
        std::lock_guard<std::mutex> lock(state.eventMutex);
        if (state.interrupted.load(std::memory_order_acquire)) return;
        state.auth = *value;
        ++state.authVersion;
        state.authPending = true;
    }
    state.events.notify_all();
}
void OMNI_USD_LIVE_CALL diagnosticObserver(void* user, const OmniUsdLiveDiagnosticRecord* value) noexcept {
    auto& state = *static_cast<SubscriptionState*>(user);
    OwnedDiagnostic copied;
    if (!value || value->field_count > copied.fields.size() || (value->field_count && !value->fields) ||
        !boundedText(copied.name, value->event_name)) {
        state.lose();
        state.fail(code::INTERNAL_ERROR);
        return;
    }
    copied.record = *value;
    copied.record.event_name = nullptr;
    copied.record.fields = nullptr;
    copied.charge = sizeof(OmniUsdLiveDiagnosticRecord) + std::strlen(copied.name) + 1;
    for (uint32_t i = 0; i < value->field_count; ++i) {
        const auto* field = value->fields[i];
        auto& owned = copied.fields[i];
        if (!field || field->kind < OMNI_USD_LIVE_FIELD_UINT64 || field->kind > OMNI_USD_LIVE_FIELD_BOOLEAN ||
            (field->kind == OMNI_USD_LIVE_FIELD_BOOLEAN && field->unsigned_value > 1) ||
            !boundedText(owned.name, field->name) ||
            (field->kind == OMNI_USD_LIVE_FIELD_STRING && !boundedText(owned.text, field->string_value))) {
            state.lose();
            state.fail(code::INTERNAL_ERROR);
            return;
        }
        owned.kind = field->kind;
        owned.unsignedValue = field->unsigned_value;
        owned.signedValue = field->signed_value;
        copied.charge += sizeof(OmniUsdLiveDiagnosticField) + std::strlen(owned.name) + 1 + std::strlen(owned.text) + 1;
    }
    {
        std::lock_guard<std::mutex> lock(state.eventMutex);
        if (state.interrupted.load(std::memory_order_acquire)) return;
        if (state.count == state.ring.size() || copied.charge > state.maxBytes ||
            state.queuedBytes > state.maxBytes - copied.charge) {
            state.lose();
            return;
        }
        state.ring[(state.head + state.count) % state.ring.size()] = copied;
        ++state.count;
        state.queuedBytes += copied.charge;
    }
    state.events.notify_all();
}
py::dict authDict(const OmniUsdLiveAuthRequired& value) {
    py::dict out;
#define AUTH_FIELD(name) out[#name] = value.name
    AUTH_FIELD(endpoint);
    AUTH_FIELD(rejected_generation);
    AUTH_FIELD(challenge_id);
    AUTH_FIELD(reason);
    AUTH_FIELD(grpc_status_present);
    AUTH_FIELD(grpc_status);
#undef AUTH_FIELD
    return out;
}
py::dict diagnosticDict(const OwnedDiagnostic& value) {
    py::dict out;
#define DIAG_FIELD(name) out[#name] = value.record.name
    DIAG_FIELD(schema_version);
    DIAG_FIELD(event_code);
    DIAG_FIELD(severity);
    DIAG_FIELD(timestamp_ms);
    DIAG_FIELD(elapsed_ms);
    DIAG_FIELD(runtime_instance_id);
    DIAG_FIELD(sequence);
    DIAG_FIELD(endpoint);
    DIAG_FIELD(subject_id);
    DIAG_FIELD(incident_id);
#undef DIAG_FIELD
    out["event_name"] = value.name;
    py::list fields;
    for (uint32_t i = 0; i < value.record.field_count; ++i) {
        const auto& field = value.fields[i];
        py::dict item;
        item["name"] = field.name;
        item["kind"] = field.kind;
        if (field.kind == OMNI_USD_LIVE_FIELD_STRING) item["value"] = py::cast(std::string(field.text));
        else if (field.kind == OMNI_USD_LIVE_FIELD_INT64) item["value"] = py::cast(field.signedValue);
        else if (field.kind == OMNI_USD_LIVE_FIELD_BOOLEAN) item["value"] = py::bool_(field.unsignedValue != 0);
        else item["value"] = py::cast(field.unsignedValue);
        fields.append(item);
    }
    out["fields"] = fields;
    return out;
}
class Subscription {
public:
    explicit Subscription(std::shared_ptr<SubscriptionState> value) : state(std::move(value)) {}
    ~Subscription() { Owners::get().abandon(state); }
    uint64_t handle() {
        py::gil_scoped_release release;
        return state->currentHandle();
    }
    Result close(uint32_t timeout) {
        Result result;
        {
            py::gil_scoped_release release;
            result = state->close(timeout);
        }
        if (result == code::OK) Owners::get().retire(state);
        return result;
    }
    void interrupt() { state->interrupt(); }
    Result deliveryStatus() const noexcept { return state->delivery.load(std::memory_order_acquire); }
    uint64_t dropped() const noexcept { return state->dropped.load(std::memory_order_acquire); }
    bool authCurrent(uint64_t generation, uint64_t challenge) const {
        auto current = initRecord<OmniUsdLiveAuthState>();
        py::gil_scoped_release release;
        return challenge && omniUsdLiveGetAuthState(state->endpoint, &current) == code::OK &&
            current.generation == generation && current.challenge_id == challenge;
    }
    Result ack(uint64_t subject, uint64_t revision) {
        py::gil_scoped_release release;
        const auto handle = state->currentHandle();
        return handle ? omniUsdLiveAcknowledgeStatus(handle, subject, revision) : code::INVALID_HANDLE;
    }
    py::tuple next(uint32_t timeout) {
        bool expected = false;
        if (!state->consuming.compare_exchange_strong(expected, true, std::memory_order_acquire))
            return py::make_tuple(code::BUSY, py::none());
        struct Guard {
            SubscriptionState& state;
            ~Guard() { state.consuming.store(false, std::memory_order_release); }
        } guard{*state};
        if (state->interrupted.load(std::memory_order_acquire)) return py::make_tuple(code::CANCELLED, py::none());
        if (state->kind == SubscriptionState::Kind::Status) {
            auto value = initRecord<OmniUsdLiveStatusObservation>();
            uint64_t size = 0;
            Result result;
            uint64_t handle;
            {
                py::gil_scoped_release release;
                handle = state->currentHandle();
                result = handle ? omniUsdLiveReadStatus(handle, timeout, &value, nullptr, &size) : code::INVALID_HANDLE;
            }
            if (result != code::OK) return py::make_tuple(result, py::none());
            if (!size || size > 1024 * 1024) return py::make_tuple(code::RESOURCE_EXHAUSTED, py::none());
            reserveBytes(size);
            struct Charge {
                uint64_t size;
                ~Charge() { retainedReplyBytes.fetch_sub(size, std::memory_order_relaxed); }
            } charge{size};
            std::vector<char> uri(static_cast<std::size_t>(size));
            {
                py::gil_scoped_release release;
                result = omniUsdLiveReadStatus(handle, 0, &value, uri.data(), &size);
            }
            if (result != code::OK) return py::make_tuple(result, py::none());
            if (!size || size > uri.size() || uri[size - 1] != '\0')
                return py::make_tuple(code::INTERNAL_ERROR, py::none());
            py::dict out;
            out["endpoint"] = value.endpoint;
            out["session"] = value.session;
            out["uri"] = std::string(uri.data(), static_cast<std::size_t>(size - 1));
            out["status"] = statusDict(value.status);
            return py::make_tuple(code::OK, out); // Retained in C until explicit matching ACK.
        }
        OmniUsdLiveAuthRequired auth{};
        uint64_t version = 0;
        OwnedDiagnostic diagnostic;
        bool pending = false;
        Result failure;
        {
            py::gil_scoped_release release;
            std::unique_lock<std::mutex> lock(state->eventMutex);
            state->events.wait_for(lock, std::chrono::milliseconds(timeout), [&] {
                return state->interrupted.load(std::memory_order_acquire) || state->authPending || state->count ||
                    state->delivery.load(std::memory_order_acquire) != code::OK;
            });
            if (state->kind == SubscriptionState::Kind::Auth) {
                pending = state->authPending;
                auth = state->auth;
                version = state->authVersion;
            } else if (state->count) {
                pending = true;
                diagnostic = state->ring[state->head];
            }
            failure = state->delivery.load(std::memory_order_acquire);
        }
        if (state->interrupted.load(std::memory_order_acquire)) return py::make_tuple(code::CANCELLED, py::none());
        if (!pending) return py::make_tuple(failure == code::OK ? code::TIMEOUT : failure, py::none());
        if (state->kind == SubscriptionState::Kind::Auth) {
            auto current = initRecord<OmniUsdLiveAuthState>();
            Result result;
            {
                py::gil_scoped_release release;
                result = omniUsdLiveGetAuthState(auth.endpoint, &current);
            }
            const bool valid = result == code::OK && current.challenge_id == auth.challenge_id &&
                current.generation == auth.rejected_generation && auth.challenge_id != 0;
            py::tuple out = valid ? py::make_tuple(code::OK, authDict(auth)) :
                                    py::make_tuple(code::TIMEOUT, py::none());
            {
                py::gil_scoped_release release;
                std::lock_guard<std::mutex> lock(state->eventMutex);
                if (version == state->authVersion) state->authPending = false;
            }
            return out;
        }
        auto out = py::make_tuple(code::OK, diagnosticDict(diagnostic));
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(state->eventMutex);
            if (state->count && state->ring[state->head].record.sequence == diagnostic.record.sequence) {
                state->queuedBytes -= state->ring[state->head].charge;
                state->ring[state->head] = OwnedDiagnostic{};
                state->head = (state->head + 1) % state->ring.size();
                --state->count;
            }
        }
        return out;
    }
    std::shared_ptr<SubscriptionState> state;
};
py::tuple registerSubscription(SubscriptionState::Kind kind, const std::shared_ptr<Endpoint>& endpoint,
    const py::dict& input, uint32_t records, uint64_t bytes) {
    if (kind == SubscriptionState::Kind::Auth && !endpoint) return py::make_tuple(code::INVALID_ARGUMENT, py::none());
    if (!records || records > 4096 || !bytes || bytes > ReplyByteLimit)
        return py::make_tuple(code::INVALID_ARGUMENT, py::none());
    keys(input, {"min_severity", "flags", "event_codes"});
    auto options = initRecord<OmniUsdLiveDiagnosticOptions>();
    option(input, "min_severity", options.min_severity);
    option(input, "flags", options.flags);
    if (input.contains("event_codes") && py::len(input["event_codes"]) > 256)
        return py::make_tuple(code::RESOURCE_EXHAUSTED, py::none());
    const auto codes = input.contains("event_codes") ? py::cast<std::vector<uint32_t>>(input["event_codes"]) :
                                                       std::vector<uint32_t>{};
    if (codes.size() > 256) return py::make_tuple(code::RESOURCE_EXHAUSTED, py::none());
    options.event_codes = codes.data();
    options.event_code_count = static_cast<uint32_t>(codes.size());
    auto state = std::make_shared<SubscriptionState>(kind);
    if (kind == SubscriptionState::Kind::Diagnostic) {
        try {
            state->initialize(records, bytes);
        } catch (const std::bad_alloc&) { return py::make_tuple(code::RESOURCE_EXHAUSTED, py::none()); }
    }
    auto owner = std::make_shared<Subscription>(state);
    auto success = py::make_tuple(code::OK, owner);
    if (!Owners::get().add(state)) return py::make_tuple(code::RESOURCE_EXHAUSTED, py::none());
    Result result;
    {
        py::gil_scoped_release release;
        std::unique_lock<std::mutex> endpointLock;
        if (endpoint) endpointLock = std::unique_lock<std::mutex>(endpoint->state->mutex);
        if (endpoint && !endpoint->state->owned) result = code::INVALID_HANDLE;
        else {
            const auto filter = endpoint ? endpoint->state->handle : 0;
            state->endpoint = filter;
            if (kind == SubscriptionState::Kind::Auth)
                result = omniUsdLiveRegisterAuthRequiredCallback(filter, state.get(), authObserver, &state->handle);
            else if (kind == SubscriptionState::Kind::Status)
                result = omniUsdLiveCreateStatusObserver(filter, &state->handle);
            else {
                options.endpoint = filter;
                result = omniUsdLiveRegisterDiagnosticCallback(&options, state.get(), diagnosticObserver,
                    &state->handle);
            }
        }
    }
    if (result != code::OK) {
        Owners::get().retire(state);
        return py::make_tuple(result, py::none());
    }
    return success;
}

OmniUsdLiveJitterParams jitter(const py::dict& input) {
    keys(input, {"delay_constant_ms", "delay_multiple", "delay_maximum_ms"});
    auto value = initRecord<OmniUsdLiveJitterParams>();
    option(input, "delay_constant_ms", value.delay_constant_ms);
    option(input, "delay_multiple", value.delay_multiple);
    option(input, "delay_maximum_ms", value.delay_maximum_ms);
    return value;
}

py::tuple initialize(const py::dict& input) {
    auto& owners = Owners::get();
    auto value = initRecord<OmniUsdLiveRuntimeOptions>();
    keys(input,
        {"default_auth_wait_timeout_ms", "max_rejected_token_generations", "auth_expiry_lead_ms", "request_timeout_ms",
            "initial_open_timeout_ms", "retry_initial_ms", "retry_maximum_ms", "retry_max_attempts",
            "peer_reconciliation_grace_ms", "max_update_bytes", "max_pending_bytes_per_endpoint",
            "max_pending_messages_per_session", "max_synthetic_gap", "max_status_subjects", "max_status_observers",
            "max_diagnostic_records", "max_diagnostic_bytes", "max_diagnostic_sink_records", "max_diagnostic_sinks",
            "max_closed_incidents", "diagnostic_min_severity", "diagnostic_allowed_flags", "flags", "default_jitter",
            "max_requests", "callback_worker_count", "max_auth_callbacks", "apply_lag_warning_threshold",
            "reconnect_initial_ms", "reconnect_maximum_ms", "reconnect_max_attempts"});
#define RUNTIME_FIELD(name) option(input, #name, value.name)
    RUNTIME_FIELD(default_auth_wait_timeout_ms);
    RUNTIME_FIELD(max_rejected_token_generations);
    RUNTIME_FIELD(auth_expiry_lead_ms);
    RUNTIME_FIELD(request_timeout_ms);
    RUNTIME_FIELD(initial_open_timeout_ms);
    RUNTIME_FIELD(retry_initial_ms);
    RUNTIME_FIELD(retry_maximum_ms);
    RUNTIME_FIELD(retry_max_attempts);
    RUNTIME_FIELD(peer_reconciliation_grace_ms);
    RUNTIME_FIELD(max_update_bytes);
    RUNTIME_FIELD(max_pending_bytes_per_endpoint);
    RUNTIME_FIELD(max_pending_messages_per_session);
    RUNTIME_FIELD(max_synthetic_gap);
    RUNTIME_FIELD(max_status_subjects);
    RUNTIME_FIELD(max_status_observers);
    RUNTIME_FIELD(max_diagnostic_records);
    RUNTIME_FIELD(max_diagnostic_bytes);
    RUNTIME_FIELD(max_diagnostic_sink_records);
    RUNTIME_FIELD(max_diagnostic_sinks);
    RUNTIME_FIELD(max_closed_incidents);
    RUNTIME_FIELD(diagnostic_min_severity);
    RUNTIME_FIELD(diagnostic_allowed_flags);
    RUNTIME_FIELD(flags);
    RUNTIME_FIELD(max_requests);
    RUNTIME_FIELD(callback_worker_count);
    RUNTIME_FIELD(max_auth_callbacks);
    RUNTIME_FIELD(apply_lag_warning_threshold);
    RUNTIME_FIELD(reconnect_initial_ms);
    RUNTIME_FIELD(reconnect_maximum_ms);
    RUNTIME_FIELD(reconnect_max_attempts);
#undef RUNTIME_FIELD
    auto delay = initRecord<OmniUsdLiveJitterParams>();
    if (input.contains("default_jitter") && !input["default_jitter"].is_none()) {
        delay = jitter(py::cast<py::dict>(input["default_jitter"]));
        value.default_jitter = &delay;
    }
    auto state = std::make_shared<RefState>(RefState::Kind::Runtime);
    auto owner = std::make_shared<Runtime>(state);
    auto success = py::make_tuple(code::OK, owner);
    if (!owners.add(state)) return py::make_tuple(code::RESOURCE_EXHAUSTED, py::none());
    const bool defaults = input.empty();
    Result result;
    {
        py::gil_scoped_release release;
        result = omniUsdLiveInitialize(defaults ? nullptr : &value);
    }
    if (result != code::OK) {
        Owners::get().retire(state);
        return py::make_tuple(result, py::none());
    }
    state->owned = true;
    return success;
}
py::tuple registerEndpoint(const py::dict& input) {
    auto& owners = Owners::get();
    keys(input,
        {"authority", "tls_server_name", "ca_pem", "access_token", "expires_at_ms", "transport_security",
            "auth_wait_timeout_ms"});
    const auto authority = stringOption(input, "authority"), server = stringOption(input, "tls_server_name");
    const auto ca = stringOption(input, "ca_pem"), token = stringOption(input, "access_token");
    auto value = initRecord<OmniUsdLiveEndpointOptions>();
    value.authority = authority.c_str();
    value.tls_server_name = server.c_str();
    value.ca_pem = ca.c_str();
    value.access_token = token.c_str();
    option(input, "expires_at_ms", value.expires_at_ms);
    option(input, "transport_security", value.transport_security);
    option(input, "auth_wait_timeout_ms", value.auth_wait_timeout_ms);
    auto state = std::make_shared<RefState>(RefState::Kind::Endpoint);
    auto owner = std::make_shared<Endpoint>(state);
    auto success = py::make_tuple(code::OK, owner);
    if (!owners.add(state)) return py::make_tuple(code::RESOURCE_EXHAUSTED, py::none());
    Result result;
    {
        py::gil_scoped_release release;
        result = omniUsdLiveRegisterEndpoint(&value, &state->handle);
    }
    if (result != code::OK) {
        Owners::get().retire(state);
        return py::make_tuple(result, py::none());
    }
    state->owned = true;
    return success;
}

template<class Query> py::tuple buffer(Query&& query, uint64_t limit) {
    uint64_t size = 0;
    Result result;
    {
        py::gil_scoped_release release;
        result = query(nullptr, &size);
    }
    if (result != code::OK) return py::make_tuple(result, std::string{});
    // Live diagnostics can grow between size query and copy. Retry boundedly.
    for (unsigned attempt = 0; attempt != 4; ++attempt) {
        if (!size || size > limit) return py::make_tuple(code::RESOURCE_EXHAUSTED, std::string{});
        std::vector<char> output(static_cast<std::size_t>(size));
        {
            py::gil_scoped_release release;
            result = query(output.data(), &size);
        }
        if (result == code::BUFFER_TOO_SMALL) continue;
        if (result != code::OK) return py::make_tuple(result, std::string{});
        if (!size || size > output.size() || output[size - 1] != '\0')
            return py::make_tuple(code::INTERNAL_ERROR, std::string{});
        return py::make_tuple(result, std::string(output.data(), static_cast<std::size_t>(size - 1)));
    }
    return py::make_tuple(code::BUSY, std::string{});
}
// Pins the plugin library and this extension module for the life of the process, so a callback running on a native
// thread never outlives the code it runs.
void pinCode() {
    static int anchor;
    for (const void* address :
        {reinterpret_cast<const void*>(&omniUsdLiveGetAbiVersion), static_cast<const void*>(&anchor)}) {
#if defined(_WIN32)
        HMODULE module = nullptr;
        if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
                static_cast<LPCWSTR>(address), &module))
            throw std::runtime_error("Could not pin native live module/plugin for callback lifetime");
#else
        Dl_info info{};
        if (!dladdr(address, &info) || !info.dli_fname ||
            !dlopen(info.dli_fname, RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE))
            throw std::runtime_error("Could not pin native live module/plugin for callback lifetime");
#endif
    }
}
} // namespace

static void bindLive(py::module_& module);
static py::module_::module_def liveModuleDefinition;
// Pinned pybind11 2.11 initializes its GILState-backed internals before the
// PYBIND11_MODULE body. Reject subinterpreters before that initialization.
PYBIND11_PLUGIN_IMPL(_live) {
    PYBIND11_CHECK_PYTHON_VERSION
    if (PyThreadState_Get()->interp != PyInterpreterState_Main()) {
        PyErr_SetString(PyExc_ImportError, "omni.usd.live supports the main Python interpreter only");
        return nullptr;
    }
    try {
        PYBIND11_ENSURE_INTERNALS_READY
        auto module = py::module_::create_extension_module("_live", nullptr, &liveModuleDefinition);
        bindLive(module);
        return module.ptr();
    }
    PYBIND11_CATCH_INIT_EXCEPTIONS
}

static void bindLive(py::module_& module) {
    pinCode();
    bindLiveHost(module);
    module.attr("ABI_VERSION") = OMNI_USD_LIVE_ABI_VERSION;
    module.attr("OWNER_LIMIT") = OwnerLimit;
    module.attr("REPLY_BYTE_LIMIT") = ReplyByteLimit;
    module.attr("TOTAL_REPLY_BYTE_LIMIT") = TotalReplyByteLimit;
    module.attr("REPLY_RECORD_LIMIT") = ReplyRecordLimit;
    py::class_<Runtime, std::shared_ptr<Runtime>>(module, "Runtime")
        .def("close", &Runtime::close, py::arg("timeout_ms") = 0)
        .def("cleanup", &Runtime::cleanup, py::arg("timeout_ms") = 0);
    py::class_<Endpoint, std::shared_ptr<Endpoint>>(module, "Endpoint")
        .def_property_readonly("handle", &Endpoint::handle)
        .def("close", &Endpoint::close, py::arg("timeout_ms") = 0)
        .def("auth_state",
            [](const Endpoint& endpoint) {
        auto value = initRecord<OmniUsdLiveAuthState>();
        Result result;
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(endpoint.state->mutex);
            result = endpoint.state->owned ? omniUsdLiveGetAuthState(endpoint.state->handle, &value) :
                                             code::INVALID_HANDLE;
        }
        py::dict out;
        if (result == code::OK) {
            out["generation"] = value.generation;
            out["expires_at_ms"] = value.expires_at_ms;
            out["challenge_id"] = value.challenge_id;
            out["state"] = value.state;
            out["reason"] = value.reason;
        }
        return py::make_tuple(result, out);
    })
        .def("set_access_token",
            [](const Endpoint& endpoint, uint64_t expected, const std::string& token, uint64_t expires) {
        text(token);
        uint64_t generation = 0;
        Result result;
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(endpoint.state->mutex);
            result = endpoint.state->owned ?
                omniUsdLiveSetAccessToken(endpoint.state->handle, expected, token.c_str(), expires, &generation) :
                code::INVALID_HANDLE;
        }
        return py::make_tuple(result, generation);
    }, py::arg("expected_generation"), py::arg("token"), py::arg("expires_at_ms") = 0)
        .def("fail_auth_refresh", [](const Endpoint& endpoint, uint64_t challenge) -> Result {
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(endpoint.state->mutex);
        return endpoint.state->owned ? omniUsdLiveFailAuthRefresh(endpoint.state->handle, challenge) :
                                       code::INVALID_HANDLE;
    }).def("status", [](const Endpoint& endpoint) {
        return statusQuery([&](OmniUsdLiveStatus* value) -> Result {
            std::lock_guard<std::mutex> lock(endpoint.state->mutex);
            return endpoint.state->owned ? omniUsdLiveGetEndpointStatus(endpoint.state->handle, value) :
                                           code::INVALID_HANDLE;
        });
    });
    py::class_<Request, std::shared_ptr<Request>>(module, "Request")
        .def_property_readonly("handle", &Request::handle)
        .def("wait", &Request::wait, py::arg("timeout_ms") = 0)
        .def("cancel", &Request::cancel, py::arg("timeout_ms") = 0)
        .def("release", &Request::release)
        .def("status", &Request::status)
        .def("native_status", &Request::nativeStatus)
        .def("error", &Request::error)
        .def("payload", &Request::payload);
    py::class_<Session, std::shared_ptr<Session>>(module, "Session")
        .def_property_readonly("handle", &Session::handle)
        .def("next_event", &Session::next, py::arg("timeout_ms") = 0)
        .def("leave", &Session::leave, py::arg("timeout_ms") = 0)
        .def("interrupt_wait", &Session::interrupt)
        .def("status", &Session::status)
        .def("delivery_status", &Session::deliveryStatus);
    py::class_<Subscription, std::shared_ptr<Subscription>>(module, "Subscription")
        .def_property_readonly("handle", &Subscription::handle)
        .def("next_event", &Subscription::next, py::arg("timeout_ms") = 0)
        .def("acknowledge", &Subscription::ack)
        .def("auth_current", &Subscription::authCurrent)
        .def("close", &Subscription::close, py::arg("timeout_ms") = 0)
        .def("interrupt_wait", &Subscription::interrupt)
        .def("delivery_status", &Subscription::deliveryStatus)
        .def_property_readonly("dropped_records", &Subscription::dropped);
    module.def("register_auth_required_callback", [](const std::shared_ptr<Endpoint>& endpoint) {
        return registerSubscription(SubscriptionState::Kind::Auth, endpoint, py::dict{}, 1, 1);
    });
    module.def("register_status_callback", [](const std::shared_ptr<Endpoint>& endpoint) {
        return registerSubscription(SubscriptionState::Kind::Status, endpoint, py::dict{}, 1, 1);
    }, py::arg("endpoint") = nullptr);
    module.def("register_diagnostic_callback",
        [](const std::shared_ptr<Endpoint>& endpoint, const py::dict& input, uint32_t records, uint64_t bytes) {
        return registerSubscription(SubscriptionState::Kind::Diagnostic, endpoint, input, records, bytes);
    }, py::arg("endpoint") = nullptr, py::arg("options") = py::dict{}, py::arg("max_records") = 256,
        py::arg("max_bytes") = 1024 * 1024);
    module.def("join_session", &joinSession, py::arg("uri"), py::arg("app_name"), py::arg("max_events") = 256,
        py::arg("max_bytes") = 1024 * 1024);
    module.def("send_message", [](const Session& session, const py::bytes& bytes) {
        char* data = nullptr;
        Py_ssize_t size = 0;
        if (PyBytes_AsStringAndSize(bytes.ptr(), &data, &size) != 0) throw py::error_already_set();
        // Immutable Python bytes stay pinned by the call; C copies before return.
        return submit(RequestState::Kind::ResultOnly, [&](void* user, uint64_t* request) -> Result {
            std::lock_guard<std::mutex> lock(session.state->lifeMutex);
            const auto failure = session.state->delivery.load(std::memory_order_acquire);
            if (failure != code::OK) return failure;
            if (!session.state->owned) return code::INVALID_HANDLE;
            if (session.state->stop.load(std::memory_order_acquire)) return code::CANCELLED;
            return omniUsdLiveSendMessage(session.state->handle, data, static_cast<uint64_t>(size), user,
                resultCallback, request);
        });
    });
    module.def("initialize", &initialize, py::arg("options") = py::dict{});
    module.def("register_endpoint", &registerEndpoint);
    module.def("get_abi_version", [] {
        uint32_t value = 0;
        auto result = omniUsdLiveGetAbiVersion(&value);
        return py::make_tuple(result, value);
    });
    module.def("get_capabilities", [] {
        auto value = initRecord<OmniUsdLiveCapabilities>();
        auto result = omniUsdLiveGetCapabilities(&value);
        py::dict out;
        if (result == code::OK) {
            out["delegate_version"] = value.delegate_version;
            out["diagnostic_schema_version"] = value.diagnostic_schema_version;
            out["autonomous_progress"] = value.autonomous_progress;
        }
        return py::make_tuple(result, out);
    });
    module.def("get_result_string", [](Result code) {
        const char* value = nullptr;
        auto result = omniUsdLiveGetResultString(code, &value);
        return py::make_tuple(result, nonnull(value));
    });
    module.def("list_sessions", [](const Endpoint& endpoint, const std::string& key) {
        text(key);
        return submit(RequestState::Kind::List, [&](void* user, uint64_t* request) -> Result {
            std::lock_guard<std::mutex> lock(endpoint.state->mutex);
            return endpoint.state->owned ?
                omniUsdLiveListSessions(endpoint.state->handle, key.c_str(), user, listCallback, request) :
                code::INVALID_HANDLE;
        });
    });
    module.def("create_session", [](const Endpoint& endpoint, const std::string& key, const std::string& name) {
        text(key);
        text(name);
        return submit(RequestState::Kind::Create, [&](void* user, uint64_t* request) -> Result {
            std::lock_guard<std::mutex> lock(endpoint.state->mutex);
            return endpoint.state->owned ? omniUsdLiveCreateSession(endpoint.state->handle, key.c_str(), name.c_str(),
                                               user, createCallback, request) :
                                           code::INVALID_HANDLE;
        });
    });
    module.def("delete_session", [](const std::string& uri) {
        text(uri);
        return submit(RequestState::Kind::ResultOnly, [&](void* user, uint64_t* request) {
            return omniUsdLiveDeleteSession(uri.c_str(), user, resultCallback, request);
        });
    });
    module.def("stat_layer", [](const std::string& uri) {
        text(uri);
        return submit(RequestState::Kind::Stat, [&](void* user, uint64_t* request) {
            return omniUsdLiveStatLayer(uri.c_str(), user, statCallback, request);
        });
    });
    module.def("delete_layer", [](const std::string& uri) {
        text(uri);
        return submit(RequestState::Kind::ResultOnly, [&](void* user, uint64_t* request) {
            return omniUsdLiveDeleteLayer(uri.c_str(), user, resultCallback, request);
        });
    });
    module.def("make_session_uri", [](const std::string& authority, const std::string& id) {
        text(authority);
        text(id);
        return buffer([&](char* out, uint64_t* size) {
            return omniUsdLiveMakeSessionUri(authority.c_str(), id.c_str(), out, size);
        }, 1024 * 1024);
    });
    module.def("make_layer_uri", [](const std::string& base, const std::string& name) {
        text(base);
        text(name);
        return buffer([&](char* out, uint64_t* size) {
            return omniUsdLiveMakeLayerUri(base.c_str(), name.c_str(), out, size);
        }, 1024 * 1024);
    });
    module.def("combine_with_base_uri", [](const std::string& base, const std::string& name) {
        text(base);
        text(name);
        return buffer([&](char* out, uint64_t* size) {
            return omniUsdLiveCombineWithBaseUri(base.c_str(), name.c_str(), out, size);
        }, 1024 * 1024);
    });
    module.def("parse_uri", [](const std::string& uri) {
        text(uri);
        auto parts = initRecord<OmniUsdLiveUriParts>();
        uint64_t size = 0;
        Result result;
        {
            py::gil_scoped_release release;
            result = omniUsdLiveParseUri(uri.c_str(), &parts, nullptr, &size);
        }
        py::dict out;
        if (result != code::OK) return py::make_tuple(result, out);
        if (!size || size > 1024 * 1024) return py::make_tuple(code::RESOURCE_EXHAUSTED, out);
        std::vector<char> bytes(static_cast<std::size_t>(size));
        {
            py::gil_scoped_release release;
            result = omniUsdLiveParseUri(uri.c_str(), &parts, bytes.data(), &size);
        }
        if (result == code::OK) {
            out["authority"] = bytes.data() + parts.authority_offset;
            out["session_id"] = bytes.data() + parts.session_id_offset;
            out["layer_name"] = bytes.data() + parts.layer_name_offset;
            out["canonical_uri"] = bytes.data() + parts.canonical_uri_offset;
            out["is_layer"] = bool(parts.is_layer);
        }
        return py::make_tuple(result, out);
    });
    module.def("get_layer_status", [](const std::string& uri) {
        text(uri);
        return statusQuery([&](OmniUsdLiveStatus* out) { return omniUsdLiveGetLayerStatus(uri.c_str(), out); });
    });
    module.def("get_session_status", [](uint64_t session) {
        return statusQuery([&](OmniUsdLiveStatus* out) { return omniUsdLiveGetSessionStatus(session, out); });
    });
    module.def("configure_jitter_reduction", [](const Endpoint& endpoint, const py::dict& input) {
        auto value = jitter(input);
        py::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(endpoint.state->mutex);
        return endpoint.state->owned ? omniUsdLiveConfigureJitterReduction(endpoint.state->handle, &value) :
                                       code::INVALID_HANDLE;
    });
    module.def("wait_for_pending_updates", [](uint32_t timeout) {
        py::gil_scoped_release release;
        return omniUsdLiveWaitForPendingUpdates(timeout);
    }, py::arg("timeout_ms") = 0);
    module.def("copy_diagnostics", [](const std::shared_ptr<Endpoint>& endpoint, const py::dict& input) {
        keys(input, {"flags", "max_records", "max_output_bytes"});
        auto value = initRecord<OmniUsdLiveDiagnosticExportOptions>();
        // History never exceeds max_diagnostic_records, so an omitted max_records returns every retained
        // record, as NULL C options do; an explicit zero omits records.
        value.max_records = UINT32_MAX;
        option(input, "flags", value.flags);
        option(input, "max_records", value.max_records);
        option(input, "max_output_bytes", value.max_output_bytes);
        return buffer([&](char* out, uint64_t* size) -> Result {
            if (!endpoint) return omniUsdLiveCopyDiagnostics(0, &value, out, size);
            std::lock_guard<std::mutex> lock(endpoint->state->mutex);
            return endpoint->state->owned ? omniUsdLiveCopyDiagnostics(endpoint->state->handle, &value, out, size) :
                                            code::INVALID_HANDLE;
        }, 16 * 1024 * 1024);
    }, py::arg("endpoint") = nullptr, py::arg("options") = py::dict{});
}
