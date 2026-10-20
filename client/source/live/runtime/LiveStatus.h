// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "LiveDispatch.h"
#include <omni/usd/live/Live.h>
#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace omni::usd::live {
// Reserved values (design section 16): no operation returns result 17 and no status
// publishes state 6; neither number is reassigned.
constexpr OmniUsdLiveResult kReservedResult = 17u;
constexpr uint32_t kReservedState = 6u;
constexpr bool definedResult(OmniUsdLiveResult result) noexcept {
    return result <= OMNI_USD_LIVE_IO_ERROR && result != kReservedResult;
}
constexpr bool definedState(uint32_t state) noexcept {
    return state >= OMNI_USD_LIVE_STATE_OPENING && state <= OMNI_USD_LIVE_STATE_CLOSED && state != kReservedState;
}

// Status output sizes. A caller passes the complete record, a larger
// same-major record, or exactly a 1.0 prefix that ends before an appended
// field; a size that splits a field is rejected. Outputs write
// min(struct_size, sizeof(T)) bytes.
template<class T> constexpr size_t minimumOutputSize() noexcept { return sizeof(T); }
template<class T> constexpr size_t appendedOutputSize() noexcept { return sizeof(T); }
template<> constexpr size_t minimumOutputSize<OmniUsdLiveStatus>() noexcept {
    return offsetof(OmniUsdLiveStatus, buffered_edit_count);
}
template<> constexpr size_t appendedOutputSize<OmniUsdLiveStatus>() noexcept {
    return offsetof(OmniUsdLiveStatus, read_sequence);
}
template<> constexpr size_t minimumOutputSize<OmniUsdLiveStatusObservation>() noexcept {
    return offsetof(OmniUsdLiveStatusObservation, status) + offsetof(OmniUsdLiveStatus, buffered_edit_count);
}
template<> constexpr size_t appendedOutputSize<OmniUsdLiveStatusObservation>() noexcept {
    return offsetof(OmniUsdLiveStatusObservation, status) + offsetof(OmniUsdLiveStatus, read_sequence);
}
// Accepted prefixes per record: the 1.0 layout before buffered_edit_count,
// and the layout before read_sequence. The next field appended to either
// record must extend acceptsOutputSize with the then-previous size as well.
static_assert(sizeof(OmniUsdLiveStatus) == 136, "extend the accepted status output sizes for an appended field");
static_assert(sizeof(OmniUsdLiveStatusObservation) == 160,
    "extend the accepted observation output sizes for an appended field");
template<class T> constexpr bool acceptsOutputSize(uint32_t size) noexcept {
    return size >= sizeof(T) || size == minimumOutputSize<T>() || size == appendedOutputSize<T>();
}

// Count of host edit buffer records that no flush has admitted, and the
// transient cause of the last deferred flush while it holds them, shared by
// the host's recording and flush paths and the layer actor's status
// publication. Both values are independent atomics: the host context writes
// them, and so does a record the host predicate rejects, from its own
// thread, as that record fails the layer. On a FAILED layer a reader can
// therefore observe a count and cause from different writes. A burst of
// records asks for one actor wake: note() requests a wake only when no
// requested publication is still to read the count, and take() clears that
// request before it reads. Every nonzero count noted since the last take()
// is remembered until the next take(): a take() that finds it with a count
// of zero reports that the buffer emptied, and its publication advances the
// status revision even when no status field changed. A flush that clears the
// buffer before the woken publication reads it, and issues nothing, therefore
// still publishes a later revision with a zero count to every host that
// could have observed the nonzero one. Ordering (all seq_cst): note() stores
// the count, then sets the bit, then requests the wake; take() clears the
// wake, then consumes the bit, then loads the count. A take() that consumes a
// note()'s bit therefore reads that note()'s count or a later one, never an
// earlier zero; a bit set after a take()'s exchange survives it, and that
// note()'s wake request follows the take()'s clear, so a later take() reads it.
class BufferedEditCount final {
public:
    // Any thread; touches no USD or host-confined state. Stores the count
    // and the deferral cause (OK while no transient failure holds the
    // buffer); true when the caller must wake the actor.
    bool note(uint64_t count, OmniUsdLiveResult blocked = OMNI_USD_LIVE_OK) noexcept {
        blocked_.store(blocked, std::memory_order_seq_cst);
        count_.store(count, std::memory_order_seq_cst);
        if (count) nonzero_.store(true, std::memory_order_seq_cst);
        return !wakePending_.exchange(true, std::memory_order_seq_cst);
    }
    // Any thread. Stores the count and cause without asking for a wake: the
    // host notes them again, with a wake, when the host turn ends.
    void store(uint64_t count, OmniUsdLiveResult blocked = OMNI_USD_LIVE_OK) noexcept {
        blocked_.store(blocked, std::memory_order_seq_cst);
        count_.store(count, std::memory_order_seq_cst);
        if (count) nonzero_.store(true, std::memory_order_seq_cst);
    }
    struct Taken {
        uint64_t count = 0;
        // A nonzero count was noted since the previous take() and the buffer
        // is empty again: the publication must advance the revision.
        bool emptied = false;
    };
    // Status publication. Reads a count no older than any note() whose wake
    // request this call consumes.
    Taken take() noexcept {
        wakePending_.exchange(false, std::memory_order_seq_cst);
        const bool nonzero = nonzero_.exchange(false, std::memory_order_seq_cst);
        const auto count = count_.load(std::memory_order_seq_cst);
        return {count, nonzero && count == 0};
    }
    // The caller of a note() that returned true, when no actor could be woken.
    void abandonWake() noexcept { wakePending_.store(false, std::memory_order_seq_cst); }
    uint64_t load() const noexcept { return count_.load(std::memory_order_acquire); }
    OmniUsdLiveResult blocked() const noexcept { return blocked_.load(std::memory_order_seq_cst); }
    // The host's flush request, run on the releasing thread; it only enqueues.
    void setRetry(std::function<void()> retry) {
        std::lock_guard<std::mutex> lock(retryMutex_);
        retry_ = std::move(retry);
    }
    // Any thread, outside endpoint locks: a pending-byte reservation on the
    // endpoint was released. Counts the release, then requests a host flush
    // when a deferred flush holds the buffer. A host that stores its
    // deferral cause and then sees an unchanged release count is certain a
    // later release observes that cause.
    void released() noexcept {
        releases_.fetch_add(1, std::memory_order_seq_cst);
        if (blocked() == OMNI_USD_LIVE_OK) return;
        std::function<void()> retry;
        try {
            std::lock_guard<std::mutex> lock(retryMutex_);
            retry = retry_;
        } catch (...) { return; }
        try {
            if (retry) retry();
        } catch (...) {}
    }
    uint64_t releases() const noexcept { return releases_.load(std::memory_order_seq_cst); }
    // The host deferred a flush because the layer's issuance window is full
    // and checks the window again after this call; a layer whose window then
    // gains room calls slotFreed, which requests the flush once. Any thread,
    // outside endpoint locks.
    void awaitSlot() noexcept { awaitingSlot_.store(true, std::memory_order_seq_cst); }
    void slotFreed() noexcept {
        if (!awaitingSlot_.load(std::memory_order_seq_cst) || !awaitingSlot_.exchange(false)) return;
        std::function<void()> retry;
        try {
            std::lock_guard<std::mutex> lock(retryMutex_);
            retry = retry_;
        } catch (...) { return; }
        try {
            if (retry) retry();
        } catch (...) {}
    }
private:
    std::atomic<bool> awaitingSlot_{false};
    std::atomic<uint64_t> releases_{0};
    std::mutex retryMutex_;
    std::function<void()> retry_;
    std::atomic<uint64_t> count_{0};
    std::atomic<OmniUsdLiveResult> blocked_{OMNI_USD_LIVE_OK};
    std::atomic<bool> wakePending_{false};
    std::atomic<bool> nonzero_{false};
};

struct StatusSnapshot {
    OmniUsdLiveEndpoint endpoint;
    OmniUsdLiveSession session;
    std::string uri;
    OmniUsdLiveStatus status;
};
enum class ReadinessResolution : uint32_t { Ready = 0, Failed = 1, Closed = 2 };
enum class BackpressureResolution : uint32_t { LeftBlocked = 0, Failed = 1, Closed = 2 };
struct StatusTiming {
    bool initialStarted = false, initialCompleted = false;
    bool recoveryStarted = false, recoveryCompleted = false;
    ReadinessResolution resolution = ReadinessResolution::Ready;
    uint64_t initialMs = 0, recoveryMs = 0;
    bool backpressureStarted = false, backpressureCompleted = false;
    BackpressureResolution backpressureResolution = BackpressureResolution::LeftBlocked;
    uint64_t backpressureMs = 0;
};

// Bounded local bookkeeping only: no Runtime, network, or scene access. The
// executor must outlive this registry. Destroy outside callback workers;
// destruction cancels registrations and waits for callback/capture retirement.
// Queries/publish may run from callbacks, including callbacks of this registry.
class StatusRegistry final {
public:
    using Clock = std::chrono::steady_clock;
    // Internal bounded bookkeeping hook, under the registry lock. Never call
    // back into this registry, host code, or Runtime from the hook.
    // previous==nullptr marks admission; attached==false marks detachment,
    // including removal of an already CLOSED subject retained by observers.
    using TransitionObserver = void (*)(void*, OmniUsdLiveEndpoint, OmniUsdLiveSession, std::string_view,
        const OmniUsdLiveStatus*, const OmniUsdLiveStatus&, bool attached, const StatusTiming&) noexcept;
    // Internal diagnostic hook, outside the hub lock after disabling a throwing
    // observer. Shares the transition context; no host code or status mutation.
    using CallbackFailureObserver = void (*)(void*, OmniUsdLiveCallbackHandle, OmniUsdLiveEndpoint, OmniUsdLiveSession,
        const OmniUsdLiveStatus&) noexcept;
    // Private monotonic clock for status interval metrics; invoked under the
    // hub lock, noexcept/nonreentrant. Operational wait deadlines stay real.
    StatusRegistry(DispatchExecutor& executor, size_t maxSubjects, size_t maxObservers,
        TransitionObserver observer = nullptr, void* context = nullptr,
        Clock::time_point (*now)(void*) noexcept = nullptr, void* clockContext = nullptr,
        CallbackFailureObserver callbackFailure = nullptr);
    ~StatusRegistry();
    StatusRegistry(const StatusRegistry&) = delete;
    StatusRegistry& operator=(const StatusRegistry&) = delete;

    // Admission allocates before publication and may throw OmniUsdLiveResult,
    // bad_alloc, or length_error. Layer URI uniqueness is checked verbatim;
    // callers provide canonical identities. Endpoint subjects use their handle
    // as initial.subject_id; all other IDs come from process-global kind 6.
    uint64_t add(OmniUsdLiveEndpoint endpoint, OmniUsdLiveSession session, std::string uri,
        const OmniUsdLiveStatus& initial);
    // The hub owns ID, kind, revision, change time and endpoint child counts.
    // Terminal cause/incident and pending edits survive stale transient updates.
    // OUTCOME_UNKNOWN may strengthen an earlier cause, retaining its incident.
    // Authoritative terminal/closing progress with zero pending writes can
    // clear unconfirmed edits after actual settlement. Identical semantic
    // publications are no-ops; detachment still has a distinct final revision.
    // `advance` publishes a new revision even when no field changed.
    OmniUsdLiveResult publish(uint64_t subject, const OmniUsdLiveStatus& status, bool advance = false) noexcept;
    OmniUsdLiveResult snapshot(uint64_t subject, OmniUsdLiveStatus* output) const noexcept;
    OmniUsdLiveResult layer(std::string_view canonicalUri, OmniUsdLiveStatus* output) const noexcept;
    std::vector<StatusSnapshot> capture(OmniUsdLiveEndpoint filter, bool includeResources, uint64_t byteBudget) const;
    // Detaches query visibility immediately. Existing observers retain CLOSED
    // until delivered or explicitly unregistered, including while callbacks run.
    OmniUsdLiveResult remove(uint64_t subject) noexcept;

    // Endpoint zero observes all subjects. Atomic registration supplies current
    // active snapshots then monotonic changes; intermediate revisions coalesce.
    // One fixed slot per subject/observer is reserved at admission. Undelivered
    // CLOSED subjects retain global capacity, so slow observers reject churn
    // visibly instead of losing final failure notifications. New subscriptions
    // do not receive already detached subjects. Callback handles use kind 5.
    OmniUsdLiveCallbackHandle subscribe(OmniUsdLiveEndpoint endpointFilter, void* user,
        OmniUsdLiveStatusCallback callback, std::shared_ptr<void> lifetimeLease = {});
    // Pull registrations retain one immutable selected observation until ACK;
    // pending per-subject slots continue coalescing independently. Only native
    // copying occurs under the hub lock; reads never execute host callbacks.
    OmniUsdLiveCallbackHandle createObserver(OmniUsdLiveEndpoint endpointFilter,
        std::shared_ptr<void> lifetimeLease = {});
    OmniUsdLiveResult read(OmniUsdLiveCallbackHandle observer, uint32_t timeoutMs, OmniUsdLiveStatusObservation* output,
        char* uri, uint64_t* uriSize);
    OmniUsdLiveResult acknowledge(OmniUsdLiveCallbackHandle observer, uint64_t subject, uint64_t revision);
    OmniUsdLiveResult unsubscribe(OmniUsdLiveCallbackHandle handle, uint32_t timeoutMs);
    bool hasObservers() const noexcept;
    void beginShutdown() noexcept;
    bool waitForShutdown(Clock::time_point deadline);
private:
    OmniUsdLiveCallbackHandle subscribeImpl(OmniUsdLiveEndpoint, void*, OmniUsdLiveStatusCallback,
        std::shared_ptr<void>);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace omni::usd::live
