// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "LiveStatus.h"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

namespace omni::usd::live {
namespace {
std::atomic<uint64_t> nextSubject{1}, nextObserver{1}, nextIncident{1};
uint64_t allocate(std::atomic<uint64_t>& counter, uint64_t kind) noexcept {
    auto value = counter.load(std::memory_order_relaxed);
    do {
        if (value >= (UINT64_C(1) << 56)) return 0;
    } while (!counter.compare_exchange_weak(value, value + 1, std::memory_order_relaxed));
    return (kind << 56) | value;
}
uint64_t wallMs() noexcept {
    const auto value = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch())
                           .count();
    return value < 0 ? 0 : static_cast<uint64_t>(value);
}
bool terminal(uint32_t state) noexcept { return state == OMNI_USD_LIVE_STATE_FAILED; }
bool terminalCause(const OmniUsdLiveStatus& status) noexcept {
    return terminal(status.state) ||
        (status.result != OMNI_USD_LIVE_OK && status.recovery_action == OMNI_USD_LIVE_RECOVERY_PRESERVE_AND_REOPEN);
}
bool closing(uint32_t state) noexcept {
    return state == OMNI_USD_LIVE_STATE_CLOSING || state == OMNI_USD_LIVE_STATE_CLOSED;
}
bool blocked(uint32_t state) noexcept {
    return state == OMNI_USD_LIVE_STATE_AUTH_WAIT || state == OMNI_USD_LIVE_STATE_BLOCKED;
}
uint64_t elapsedMs(StatusRegistry::Clock::time_point start, StatusRegistry::Clock::time_point end) noexcept {
    if (end <= start) return 0;
    using Duration = StatusRegistry::Clock::duration;
    using Unsigned = std::make_unsigned_t<Duration::rep>;
    const auto ticks = Unsigned(end.time_since_epoch().count()) - Unsigned(start.time_since_epoch().count());
    return std::chrono::duration_cast<std::chrono::duration<uint64_t, std::milli>>(
        std::chrono::duration<Unsigned, Duration::period>(ticks))
        .count();
}
bool sameSemantic(const OmniUsdLiveStatus& a, const OmniUsdLiveStatus& b) noexcept {
    return a.subject_kind == b.subject_kind && a.subject_id == b.subject_id && a.state == b.state &&
        a.result == b.result && a.operation == b.operation && a.grpc_status_present == b.grpc_status_present &&
        a.grpc_status == b.grpc_status && a.recovery_action == b.recovery_action &&
        a.progress_known == b.progress_known && a.incident_id == b.incident_id &&
        a.pending_write_count == b.pending_write_count && a.pending_bytes == b.pending_bytes &&
        a.has_unconfirmed_edits == b.has_unconfirmed_edits && a.blocked_layers == b.blocked_layers &&
        a.failed_layers == b.failed_layers && a.blocked_sessions == b.blocked_sessions &&
        a.failed_sessions == b.failed_sessions && a.buffered_edit_count == b.buffered_edit_count;
}
template<class T> OmniUsdLiveResult outputVersion(T* output, uint32_t& size) noexcept {
    size = 0;
    if (output) std::memcpy(&size, output, sizeof(size));
    if (!output || !acceptsOutputSize<T>(size)) return OMNI_USD_LIVE_INVALID_ARGUMENT;
    uint32_t abi;
    std::memcpy(&abi, reinterpret_cast<const char*>(output) + offsetof(T, abi_version), sizeof(abi));
    return (abi >> 16) == (OMNI_USD_LIVE_ABI_VERSION >> 16) ? OmniUsdLiveResult(OMNI_USD_LIVE_OK) :
                                                              OmniUsdLiveResult(OMNI_USD_LIVE_UNSUPPORTED);
}
}

struct StatusRegistry::Impl {
    struct Subject {
        OmniUsdLiveEndpoint endpoint;
        OmniUsdLiveSession session;
        std::string uri;
        OmniUsdLiveStatus status;
        bool attached = true;
        bool initialPending = false, wasReady = false, recoveryPending = false;
        Clock::time_point initialStart{}, recoveryStart{};
        bool backpressurePending = false;
        Clock::time_point backpressureStart{};
    };
    struct Slot {
        std::shared_ptr<Subject> subject;
        OmniUsdLiveStatus status{};
        bool pending = false;
    };
    struct Observer final : DispatchJob {
        Impl& hub;
        const OmniUsdLiveCallbackHandle handle;
        const OmniUsdLiveEndpoint filter;
        const bool pull;
        void* user;
        OmniUsdLiveStatusCallback callback;
        std::shared_ptr<void> lease;
        std::vector<Slot> slots;
        Slot outstanding;
        size_t outstandingIndex = 0;
        size_t cursor = 0;
        bool scheduled = false, cancelled = false, quiescent = false;
        Observer(Impl& value, uint64_t id, OmniUsdLiveEndpoint endpoint, void* data, OmniUsdLiveStatusCallback function,
            std::shared_ptr<void> lifetime)
            : hub(value), handle(id), filter(endpoint), pull(function == nullptr), user(data), callback(function),
              lease(std::move(lifetime)), slots(value.subjects.size()) {}
        bool matches(const Subject& subject) const noexcept { return !filter || filter == subject.endpoint; }
        bool pending() const noexcept {
            for (const auto& slot : slots)
                if (slot.pending) return true;
            return false;
        }
        void schedule() noexcept {
            if (pull && !cancelled) {
                hub.changed.notify_all();
                return;
            }
            if (!scheduled && !quiescent) {
                scheduled = true;
                hub.executor.enqueue(*this);
            }
        }
        void dispatch() noexcept override;
    };
    DispatchExecutor& executor;
    TransitionObserver transition;
    CallbackFailureObserver callbackFailure;
    void* transitionContext;
    Clock::time_point (*now)(void*) noexcept;
    void* clockContext;
    mutable std::mutex mutex;
    std::condition_variable changed;
    std::vector<std::shared_ptr<Subject>> subjects;
    std::vector<std::shared_ptr<Observer>> observers;
    bool stopping = false;
    Impl(DispatchExecutor& value, size_t subjectCount, size_t observerCount, TransitionObserver observer, void* context,
        Clock::time_point (*clock)(void*) noexcept, void* clockData, CallbackFailureObserver failure)
        : executor(value), transition(observer), callbackFailure(failure), transitionContext(context),
          now(clock ? clock : [](void*) noexcept { return Clock::now(); }), clockContext(clockData),
          subjects(subjectCount), observers(observerCount) {
        if (subjectCount > UINT32_MAX) throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    }
    void record(Subject& subject, const OmniUsdLiveStatus* previous) noexcept {
        StatusTiming timing;
        const auto captured = now(clockContext);
        const bool failed = terminalCause(subject.status);
        const bool closed = !subject.attached || closing(subject.status.state);
        const bool pressured = !failed && !closed && subject.status.state == OMNI_USD_LIVE_STATE_BLOCKED;
        if (pressured && !subject.backpressurePending) {
            subject.backpressurePending = true;
            subject.backpressureStart = captured;
            timing.backpressureStarted = true;
        }
        if (subject.backpressurePending && !pressured) {
            subject.backpressurePending = false;
            timing.backpressureCompleted = true;
            timing.backpressureMs = elapsedMs(subject.backpressureStart, captured);
            timing.backpressureResolution = failed ? BackpressureResolution::Failed :
                closed                             ? BackpressureResolution::Closed :
                                                     BackpressureResolution::LeftBlocked;
        }
        if (subject.status.subject_kind != OMNI_USD_LIVE_SUBJECT_ENDPOINT) {
            if (!previous) {
                subject.initialPending = true;
                subject.initialStart = captured;
                timing.initialStarted = true;
            }
            const bool ready = !failed && !closed && subject.status.state == OMNI_USD_LIVE_STATE_READY;
            const bool resolved = failed || closed || ready;
            timing.resolution = failed ? ReadinessResolution::Failed :
                closed                 ? ReadinessResolution::Closed :
                                         ReadinessResolution::Ready;
            if (subject.initialPending && resolved) {
                subject.initialPending = false;
                subject.wasReady = ready;
                timing.initialCompleted = true;
                timing.initialMs = elapsedMs(subject.initialStart, captured);
            }
            if (subject.wasReady && !resolved && !subject.recoveryPending) {
                subject.recoveryPending = true;
                subject.recoveryStart = captured;
                timing.recoveryStarted = true;
            }
            if (subject.recoveryPending && resolved) {
                subject.recoveryPending = false;
                timing.recoveryCompleted = true;
                timing.recoveryMs = elapsedMs(subject.recoveryStart, captured);
            }
        }
        if (transition)
            transition(transitionContext, subject.endpoint, subject.session, subject.uri, previous, subject.status,
                subject.attached, timing);
    }
    size_t find(uint64_t id) const noexcept {
        for (size_t i = 0; i < subjects.size(); ++i)
            if (subjects[i] && subjects[i]->attached && subjects[i]->status.subject_id == id) return i;
        return subjects.size();
    }
    // Caller holds mutex; all slots were reserved during observer admission.
    void notify(size_t index) noexcept {
        const auto& subject = subjects[index];
        for (const auto& observer : observers) {
            if (!observer || observer->cancelled || !observer->matches(*subject)) continue;
            auto& slot = observer->slots[index];
            assert(!slot.subject || slot.subject == subject);
            slot.subject = subject;
            slot.status = subject->status;
            slot.pending = true;
            observer->schedule();
        }
    }
    void counts(OmniUsdLiveEndpoint endpoint, OmniUsdLiveStatus& status) const noexcept {
        status.blocked_layers = status.failed_layers = status.blocked_sessions = status.failed_sessions = 0;
        for (const auto& subject : subjects) {
            if (!subject || !subject->attached || subject->endpoint != endpoint) continue;
            const auto& child = subject->status;
            if (child.subject_kind == OMNI_USD_LIVE_SUBJECT_LAYER) {
                status.blocked_layers += blocked(child.state);
                status.failed_layers += terminal(child.state);
            } else if (child.subject_kind == OMNI_USD_LIVE_SUBJECT_SESSION) {
                status.blocked_sessions += blocked(child.state);
                status.failed_sessions += terminal(child.state);
            }
        }
    }
    void aggregate(OmniUsdLiveEndpoint endpoint) noexcept {
        for (size_t i = 0; i < subjects.size(); ++i) {
            const auto& subject = subjects[i];
            if (!subject || !subject->attached || subject->endpoint != endpoint ||
                subject->status.subject_kind != OMNI_USD_LIVE_SUBJECT_ENDPOINT)
                continue;
            auto value = subject->status;
            counts(endpoint, value);
            const auto& previous = subject->status;
            if (value.blocked_layers == previous.blocked_layers && value.failed_layers == previous.failed_layers &&
                value.blocked_sessions == previous.blocked_sessions &&
                value.failed_sessions == previous.failed_sessions)
                continue;
            if (value.revision != UINT64_MAX) ++value.revision;
            value.changed_at_ms = std::max(value.changed_at_ms, wallMs());
            subject->status = value;
            notify(i);
        }
    }
    // The returned owner must be destroyed after the caller drops mutex.
    std::shared_ptr<Subject> collectOne() noexcept {
        for (auto& subject : subjects) {
            if (!subject || subject->attached) continue;
            bool retained = false;
            for (const auto& observer : observers) {
                if (!observer) continue;
                for (const auto& slot : observer->slots)
                    if (slot.subject == subject) {
                        retained = true;
                        break;
                    }
                if (retained) break;
            }
            if (!retained) return std::move(subject);
        }
        return {};
    }
    void collect() noexcept {
        for (;;) {
            std::shared_ptr<Subject> retired;
            {
                std::lock_guard<std::mutex> lock(mutex);
                retired = collectOne();
            }
            if (!retired) return;
        }
    }
    enum class Update { Changed, Unchanged, Exhausted };
    static Update update(Subject& subject, const OmniUsdLiveStatus& incoming, bool force = false) noexcept {
        const auto& previous = subject.status;
        auto value = incoming;
        value.struct_size = sizeof(value);
        value.abi_version = OMNI_USD_LIVE_ABI_VERSION;
        value.subject_id = previous.subject_id;
        value.subject_kind = previous.subject_kind;
        const bool latched = terminalCause(previous) || previous.incident_id;
        if (latched) {
            const bool stronger = incoming.result == OMNI_USD_LIVE_OUTCOME_UNKNOWN &&
                previous.result != OMNI_USD_LIVE_OUTCOME_UNKNOWN;
            if (stronger) {
                if (!closing(value.state)) value.state = OMNI_USD_LIVE_STATE_FAILED;
            } else {
                if (!closing(value.state)) value.state = previous.state;
                value.result = previous.result;
                value.operation = previous.operation;
                value.grpc_status_present = previous.grpc_status_present;
                value.grpc_status = previous.grpc_status;
                value.recovery_action = previous.recovery_action;
            }
            value.incident_id = previous.incident_id;
        }
        if (closing(previous.state)) {
            value.state = previous.state == OMNI_USD_LIVE_STATE_CLOSED ?
                previous.state :
                (incoming.state == OMNI_USD_LIVE_STATE_CLOSED ? incoming.state : previous.state);
        }
        const bool authoritative = terminal(incoming.state) || closing(incoming.state);
        if ((latched || closing(previous.state)) && !authoritative) {
            // A stale READY producer cannot make stopped work appear flushed.
            value.progress_known = previous.progress_known;
            value.pending_write_count = previous.pending_write_count;
            value.pending_bytes = previous.pending_bytes;
            value.buffered_edit_count = previous.buffered_edit_count;
            value.has_unconfirmed_edits = previous.has_unconfirmed_edits;
        } else if (latched || closing(previous.state) || authoritative) {
            // A held application can finish during close. Its authoritative
            // zero counts may clear the edit flag without erasing the incident.
            if (!(authoritative && incoming.progress_known && !incoming.pending_write_count &&
                    !incoming.buffered_edit_count))
                value.has_unconfirmed_edits |= previous.has_unconfirmed_edits;
        }
        if (terminalCause(value) && !value.incident_id) {
            value.incident_id = allocate(nextIncident, 7);
            if (!value.incident_id) return Update::Exhausted;
        }
        // Endpoint aggregate counts are owned by the hub, not cached auth or
        // connection publishers. The caller recomputes them before update.
        // The progress sequences refresh in place: alone they never advance
        // the revision or wake an observer, so a stream of peer updates costs
        // no status notification per update.
        if (sameSemantic(value, previous) && !force) {
            subject.status.read_sequence = value.read_sequence;
            subject.status.applied_sequence = value.applied_sequence;
            return Update::Unchanged;
        }
        if (previous.revision == UINT64_MAX) return Update::Exhausted;
        value.revision = previous.revision + 1;
        value.changed_at_ms = std::max(previous.changed_at_ms, wallMs());
        subject.status = value;
        return Update::Changed;
    }
};

void StatusRegistry::Impl::Observer::dispatch() noexcept {
    std::shared_ptr<Subject> subject, retiredSubject, retiredOutstanding;
    std::shared_ptr<void> retiredLease;
    std::vector<Slot> retiredSlots;
    OmniUsdLiveStatus value{};
    OmniUsdLiveStatusCallback function = nullptr;
    void* data = nullptr;
    size_t index = 0;
    {
        std::lock_guard<std::mutex> lock(hub.mutex);
        if (!cancelled) {
            for (size_t offset = 0; offset < slots.size(); ++offset) {
                index = (cursor + offset) % slots.size();
                auto& slot = slots[index];
                if (!slot.pending) continue;
                subject = slot.subject;
                value = slot.status;
                slot.pending = false;
                function = callback;
                data = user;
                cursor = (index + 1) % slots.size();
                break;
            }
        }
    }
    if (function) {
        const OmniUsdLiveStatusEvent event{sizeof(OmniUsdLiveStatusEvent), OMNI_USD_LIVE_ABI_VERSION, subject->endpoint,
            subject->session, subject->uri.c_str(), &value};
        try {
            function(data, &event);
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(hub.mutex);
                cancelled = true;
            }
            if (hub.callbackFailure)
                hub.callbackFailure(hub.transitionContext, handle, subject->endpoint, subject->session, value);
        }
    }
    bool retiring = false;
    {
        std::lock_guard<std::mutex> lock(hub.mutex);
        if (cancelled) {
            retiredLease = std::move(lease);
            retiredSlots.swap(slots);
            retiredOutstanding = std::move(outstanding.subject);
            outstanding.pending = false;
            callback = nullptr;
            user = nullptr;
            retiring = true;
        } else {
            if (subject && value.state == OMNI_USD_LIVE_STATE_CLOSED && !subject->attached) {
                auto& slot = slots[index];
                assert(slot.subject == subject);
                if (!slot.pending) retiredSubject = std::move(slot.subject);
            }
        }
    }
    // In particular the endpoint lease may reenter endpoint/runtime/status
    // bookkeeping. Quiescence is not visible until its cleanup has returned.
    retiredLease.reset();
    retiredSlots.clear();
    retiredSubject.reset();
    retiredOutstanding.reset();
    subject.reset();
    hub.collect();
    {
        std::lock_guard<std::mutex> lock(hub.mutex);
        scheduled = false;
        if (retiring) {
            quiescent = true;
            hub.changed.notify_all();
        } else if (cancelled || pending()) schedule();
        // No hub access after publishing another dispatch. A different worker
        // may retire it and release the registry's final observer owner.
    }
}

StatusRegistry::StatusRegistry(DispatchExecutor& executor, size_t maxSubjects, size_t maxObservers,
    TransitionObserver observer, void* context, Clock::time_point (*now)(void*) noexcept, void* clockContext,
    CallbackFailureObserver callbackFailure)
    : impl_(std::make_unique<Impl>(executor, maxSubjects, maxObservers, observer, context, now, clockContext,
          callbackFailure)) {}
StatusRegistry::~StatusRegistry() {
    assert(!DispatchExecutor::isWorker());
    beginShutdown();
    waitForShutdown(Clock::time_point::max());
}

uint64_t StatusRegistry::add(OmniUsdLiveEndpoint endpoint, OmniUsdLiveSession session, std::string uri,
    const OmniUsdLiveStatus& initial) {
    if (initial.subject_kind < OMNI_USD_LIVE_SUBJECT_ENDPOINT || initial.subject_kind > OMNI_USD_LIVE_SUBJECT_LAYER ||
        !definedState(initial.state) || initial.state == OMNI_USD_LIVE_STATE_CLOSED ||
        (initial.subject_kind == OMNI_USD_LIVE_SUBJECT_LAYER && uri.empty()))
        throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
    const auto id = initial.subject_kind == OMNI_USD_LIVE_SUBJECT_ENDPOINT && initial.subject_id ?
        initial.subject_id :
        allocate(nextSubject, 6);
    if (!id) throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    if (initial.subject_kind == OMNI_USD_LIVE_SUBJECT_ENDPOINT && initial.subject_id && initial.subject_id != endpoint)
        throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
    auto value = initial;
    value.subject_id = id;
    value.revision = 1;
    value.changed_at_ms = wallMs();
    value.struct_size = sizeof(value);
    value.abi_version = OMNI_USD_LIVE_ABI_VERSION;
    if (terminalCause(value) && !value.incident_id) {
        value.incident_id = allocate(nextIncident, 7);
        if (!value.incident_id) throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    }
    auto subject = std::make_shared<Impl::Subject>(Impl::Subject{endpoint, session, std::move(uri), value});
    std::shared_ptr<Impl::Subject> retired;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (impl_->stopping) throw OmniUsdLiveResult(OMNI_USD_LIVE_BUSY);
        for (const auto& existing : impl_->subjects) {
            if (!existing) continue;
            if (existing->status.subject_id == id ||
                (existing->attached && initial.subject_kind == OMNI_USD_LIVE_SUBJECT_LAYER &&
                    existing->status.subject_kind == OMNI_USD_LIVE_SUBJECT_LAYER && existing->uri == subject->uri))
                throw OmniUsdLiveResult(OMNI_USD_LIVE_ALREADY_EXISTS);
        }
        auto slot = std::find(impl_->subjects.begin(), impl_->subjects.end(), nullptr);
        if (slot == impl_->subjects.end()) {
            retired = impl_->collectOne();
            slot = std::find(impl_->subjects.begin(), impl_->subjects.end(), nullptr);
        }
        if (slot == impl_->subjects.end()) throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
        *slot = subject;
        if (value.subject_kind == OMNI_USD_LIVE_SUBJECT_ENDPOINT) impl_->counts(endpoint, subject->status);
        impl_->record(*subject, nullptr);
        impl_->notify(static_cast<size_t>(slot - impl_->subjects.begin()));
        if (value.subject_kind != OMNI_USD_LIVE_SUBJECT_ENDPOINT) impl_->aggregate(endpoint);
    }
    return id;
}

OmniUsdLiveResult StatusRegistry::publish(uint64_t id, const OmniUsdLiveStatus& status, bool advance) noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto index = impl_->find(id);
    if (index == impl_->subjects.size()) return OMNI_USD_LIVE_INVALID_HANDLE;
    if (!definedState(status.state)) return OMNI_USD_LIVE_INVALID_ARGUMENT;
    auto& subject = *impl_->subjects[index];
    auto incoming = status;
    if (subject.status.subject_kind == OMNI_USD_LIVE_SUBJECT_ENDPOINT) impl_->counts(subject.endpoint, incoming);
    const auto previous = subject.status;
    const auto updated = Impl::update(subject, incoming, advance);
    if (updated == Impl::Update::Exhausted) return OMNI_USD_LIVE_RESOURCE_EXHAUSTED;
    if (updated == Impl::Update::Unchanged) return OMNI_USD_LIVE_OK;
    impl_->record(subject, &previous);
    impl_->notify(index);
    if (subject.status.subject_kind != OMNI_USD_LIVE_SUBJECT_ENDPOINT) impl_->aggregate(subject.endpoint);
    return OMNI_USD_LIVE_OK;
}

OmniUsdLiveResult StatusRegistry::snapshot(uint64_t id, OmniUsdLiveStatus* output) const noexcept {
    uint32_t size;
    const auto valid = outputVersion(output, size);
    if (valid != OMNI_USD_LIVE_OK) return valid;
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto index = impl_->find(id);
    if (index == impl_->subjects.size()) return OMNI_USD_LIVE_INVALID_HANDLE;
    auto value = impl_->subjects[index]->status;
    value.struct_size = size;
    std::memcpy(output, &value, std::min<size_t>(size, sizeof(value)));
    return OMNI_USD_LIVE_OK;
}

OmniUsdLiveResult StatusRegistry::layer(std::string_view uri, OmniUsdLiveStatus* output) const noexcept {
    uint32_t size;
    const auto valid = outputVersion(output, size);
    if (valid != OMNI_USD_LIVE_OK) return valid;
    std::lock_guard<std::mutex> lock(impl_->mutex);
    for (const auto& subject : impl_->subjects) {
        if (!subject || !subject->attached || subject->status.subject_kind != OMNI_USD_LIVE_SUBJECT_LAYER ||
            subject->uri != uri)
            continue;
        auto value = subject->status;
        value.struct_size = size;
        std::memcpy(output, &value, std::min<size_t>(size, sizeof(value)));
        return OMNI_USD_LIVE_OK;
    }
    return OMNI_USD_LIVE_NOT_FOUND;
}

std::vector<StatusSnapshot> StatusRegistry::capture(OmniUsdLiveEndpoint filter, bool resources, uint64_t budget) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    size_t count = 0;
    for (const auto& subject : impl_->subjects) {
        if (!subject || !subject->attached || (filter && subject->endpoint != filter)) continue;
        const auto bytes = sizeof(StatusSnapshot) + (resources ? subject->uri.size() : 0);
        if (bytes > budget) throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
        budget -= bytes;
        ++count;
    }
    std::vector<StatusSnapshot> result;
    result.reserve(count);
    for (const auto& subject : impl_->subjects) {
        if (!subject || !subject->attached || (filter && subject->endpoint != filter)) continue;
        result.push_back(
            {subject->endpoint, subject->session, resources ? subject->uri : std::string{}, subject->status});
    }
    return result;
}

OmniUsdLiveResult StatusRegistry::remove(uint64_t id) noexcept {
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        const auto index = impl_->find(id);
        if (index == impl_->subjects.size()) return OMNI_USD_LIVE_INVALID_HANDLE;
        auto& subject = *impl_->subjects[index];
        const auto previous = subject.status;
        auto closed = subject.status;
        closed.state = OMNI_USD_LIVE_STATE_CLOSED;
        if (Impl::update(subject, closed, true) == Impl::Update::Exhausted) return OMNI_USD_LIVE_RESOURCE_EXHAUSTED;
        subject.attached = false;
        impl_->record(subject, &previous);
        impl_->notify(index);
        if (subject.status.subject_kind != OMNI_USD_LIVE_SUBJECT_ENDPOINT) impl_->aggregate(subject.endpoint);
    }
    impl_->collect();
    return OMNI_USD_LIVE_OK;
}

OmniUsdLiveCallbackHandle StatusRegistry::subscribe(OmniUsdLiveEndpoint filter, void* user,
    OmniUsdLiveStatusCallback callback, std::shared_ptr<void> lease) {
    if (!callback) throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
    return subscribeImpl(filter, user, callback, std::move(lease));
}

OmniUsdLiveCallbackHandle StatusRegistry::createObserver(OmniUsdLiveEndpoint filter, std::shared_ptr<void> lease) {
    return subscribeImpl(filter, nullptr, nullptr, std::move(lease));
}

OmniUsdLiveCallbackHandle StatusRegistry::subscribeImpl(OmniUsdLiveEndpoint filter, void* user,
    OmniUsdLiveStatusCallback callback, std::shared_ptr<void> lease) {
    const auto handle = allocate(nextObserver, 5);
    if (!handle) throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    auto observer = std::make_shared<Impl::Observer>(*impl_, handle, filter, user, callback, std::move(lease));
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (impl_->stopping) throw OmniUsdLiveResult(OMNI_USD_LIVE_BUSY);
        const auto slot = std::find(impl_->observers.begin(), impl_->observers.end(), nullptr);
        if (slot == impl_->observers.end()) throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
        for (size_t i = 0; i < impl_->subjects.size(); ++i) {
            const auto& subject = impl_->subjects[i];
            if (subject && subject->attached && observer->matches(*subject))
                observer->slots[i] = {subject, subject->status, true};
        }
        *slot = observer;
        if (observer->pending()) observer->schedule();
    }
    return handle;
}

OmniUsdLiveResult StatusRegistry::read(OmniUsdLiveCallbackHandle handle, uint32_t timeoutMs,
    OmniUsdLiveStatusObservation* output, char* uri, uint64_t* uriSize) {
    uint32_t outputSize;
    const auto valid = outputVersion(output, outputSize);
    if (valid != OMNI_USD_LIVE_OK) return valid;
    if (!uriSize) return OMNI_USD_LIVE_INVALID_ARGUMENT;
    const auto deadline = Clock::now() + std::chrono::milliseconds(timeoutMs);
    // Keep this strong owner outside lock's lifetime. Cancellation dispatch may
    // retire registry ownership while a reader is waking or returning.
    std::shared_ptr<Impl::Observer> observer;
    std::unique_lock<std::mutex> lock(impl_->mutex);
    for (const auto& item : impl_->observers)
        if (item && item->handle == handle) {
            observer = item;
            break;
        }
    if (!observer) return OMNI_USD_LIVE_INVALID_HANDLE;
    if (!observer->pull) return OMNI_USD_LIVE_UNSUPPORTED;
    const auto ready = [&] { return observer->cancelled || observer->outstanding.pending || observer->pending(); };
    if (!ready()) {
        if (timeoutMs && DispatchExecutor::isWorker()) return OMNI_USD_LIVE_BUSY;
        if (!impl_->changed.wait_until(lock, deadline, ready)) return OMNI_USD_LIVE_TIMEOUT;
    }
    if (observer->cancelled) return OMNI_USD_LIVE_CANCELLED;
    if (!observer->outstanding.pending) {
        for (size_t offset = 0; offset < observer->slots.size(); ++offset) {
            const auto index = (observer->cursor + offset) % observer->slots.size();
            auto& slot = observer->slots[index];
            if (!slot.pending) continue;
            observer->outstanding = slot;
            observer->outstandingIndex = index;
            slot.pending = false;
            observer->cursor = (index + 1) % observer->slots.size();
            break;
        }
    }
    assert(observer->outstanding.pending && observer->outstanding.subject);
    const auto& current = observer->outstanding;
    const auto required = static_cast<uint64_t>(current.subject->uri.size()) + 1;
    if (!uri) {
        *uriSize = required;
        return OMNI_USD_LIVE_OK;
    }
    const auto capacity = *uriSize;
    if (capacity < required) {
        *uriSize = required;
        return OMNI_USD_LIVE_BUFFER_TOO_SMALL;
    }
    OmniUsdLiveStatusObservation value{outputSize, OMNI_USD_LIVE_ABI_VERSION, current.subject->endpoint,
        current.subject->session, current.status};
    const auto copied = std::min<size_t>(outputSize, sizeof(value));
    value.status.struct_size = static_cast<uint32_t>(copied - offsetof(OmniUsdLiveStatusObservation, status));
    // Copying and cancellation/ACK serialize on this native mutex. No host code
    // or Python executes here; both output buffers are complete before unlock.
    std::memcpy(uri, current.subject->uri.c_str(), static_cast<size_t>(required));
    std::memcpy(output, &value, copied);
    *uriSize = required;
    return OMNI_USD_LIVE_OK;
}

OmniUsdLiveResult StatusRegistry::acknowledge(OmniUsdLiveCallbackHandle handle, uint64_t subject, uint64_t revision) {
    std::shared_ptr<Impl::Subject> retiredOutstanding, retiredSlot;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        const auto found = std::find_if(impl_->observers.begin(), impl_->observers.end(),
            [&](const auto& observer) { return observer && observer->handle == handle; });
        if (found == impl_->observers.end()) return OMNI_USD_LIVE_INVALID_HANDLE;
        auto& observer = **found;
        if (!observer.pull) return OMNI_USD_LIVE_UNSUPPORTED;
        if (observer.cancelled) return OMNI_USD_LIVE_CANCELLED;
        const auto& current = observer.outstanding;
        if (!current.pending || current.status.subject_id != subject || current.status.revision != revision)
            return OMNI_USD_LIVE_INVALID_ARGUMENT;
        auto& slot = observer.slots[observer.outstandingIndex];
        if (!slot.pending && current.status.state == OMNI_USD_LIVE_STATE_CLOSED && !current.subject->attached)
            retiredSlot = std::move(slot.subject);
        retiredOutstanding = std::move(observer.outstanding.subject);
        observer.outstanding.pending = false;
        impl_->changed.notify_all();
    }
    retiredOutstanding.reset();
    retiredSlot.reset();
    impl_->collect();
    return OMNI_USD_LIVE_OK;
}

OmniUsdLiveResult StatusRegistry::unsubscribe(OmniUsdLiveCallbackHandle handle, uint32_t timeoutMs) {
    const auto deadline = Clock::now() + std::chrono::milliseconds(timeoutMs);
    std::shared_ptr<Impl::Observer> observer;
    std::unique_lock<std::mutex> lock(impl_->mutex);
    for (const auto& item : impl_->observers)
        if (item && item->handle == handle) {
            observer = item;
            break;
        }
    if (!observer) return OMNI_USD_LIVE_INVALID_HANDLE;
    observer->cancelled = true;
    impl_->changed.notify_all();
    observer->schedule();
    if (!observer->quiescent && DispatchExecutor::isWorker()) return OMNI_USD_LIVE_BUSY;
    if (!impl_->changed.wait_until(lock, deadline, [&] { return observer->quiescent; })) return OMNI_USD_LIVE_TIMEOUT;
    const auto slot = std::find(impl_->observers.begin(), impl_->observers.end(), observer);
    if (slot == impl_->observers.end()) return OMNI_USD_LIVE_INVALID_HANDLE;
    slot->reset(); // The local owner outlives lock's destructor.
    lock.unlock();
    return OMNI_USD_LIVE_OK;
}

bool StatusRegistry::hasObservers() const noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    for (const auto& observer : impl_->observers)
        if (observer) return true;
    return false;
}

void StatusRegistry::beginShutdown() noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->stopping = true;
    for (const auto& observer : impl_->observers)
        if (observer) {
            observer->cancelled = true;
            observer->schedule();
        }
    impl_->changed.notify_all();
}

bool StatusRegistry::waitForShutdown(Clock::time_point deadline) {
    std::unique_lock<std::mutex> lock(impl_->mutex);
    const auto done = [&] {
        for (const auto& observer : impl_->observers)
            if (observer && !observer->quiescent) return false;
        return true;
    };
    if (DispatchExecutor::isWorker()) return done();
    return impl_->changed.wait_until(lock, deadline, done);
}
} // namespace omni::usd::live
