// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "Host.h"
#include <omni/usd/live/LiveUsd.h>
#include <pxr/base/tf/makePyConstructor.h>
#include <pxr/external/boost/python/extract.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace py = pybind11;
namespace bp = PXR_BOOST_PYTHON_NAMESPACE;
namespace host = omni::usd::live::host;
using namespace PXR_NS;
namespace {
using Result = OmniUsdLiveResult;
constexpr Result OK = OMNI_USD_LIVE_OK, BUSY = OMNI_USD_LIVE_BUSY, INVALID = OMNI_USD_LIVE_INVALID_ARGUMENT,
                 CLOSED = OMNI_USD_LIVE_INVALID_HANDLE, CANCELLED = OMNI_USD_LIVE_CANCELLED,
                 LIMIT = OMNI_USD_LIVE_RESOURCE_EXHAUSTED;
struct State;
struct CreationState {
    host::Creation* creation = nullptr;
    bool active = false;
};
struct State {
    explicit State(std::size_t limit) : creator(std::this_thread::get_id()), queue(limit), creations(limit) {}
    const std::thread::id creator;
    std::mutex mutex;
    std::condition_variable ready;
    std::vector<std::function<void()>> queue;
    // A layer's own publication jobs: bounded by the runtime to one per layer,
    // not charged to `outstanding`, and consumed ahead of the ring, so their
    // order relative to ring jobs is not preserved. Close returns BUSY while
    // any is queued, and abandonment discards them. An exempt task in flight
    // keeps the native Host pinned, so native close returns BUSY until its
    // captures retire.
    std::deque<std::function<void()>> exempt;
    std::vector<std::shared_ptr<CreationState>> creations;
    std::size_t head = 0, count = 0, outstanding = 0;
    bool abandoned = false, closed = false, interrupted = false;
    std::atomic<Result> delivery{OK};
    bool current() const noexcept { return creator == std::this_thread::get_id(); }
    void failure(Result result) noexcept {
        Result expected = OK;
        delivery.compare_exchange_strong(expected, result);
    }
    void enqueue(std::function<void()> job) {
        Result rejected = OK;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (abandoned || closed) rejected = CANCELLED;
            else if (host::dispatchingLocalPublication()) {
                try {
                    exempt.push_back(std::move(job));
                } catch (const std::bad_alloc&) { rejected = LIMIT; }
            } else if (outstanding == queue.size()) rejected = LIMIT;
            else {
                queue[(head + count) % queue.size()].swap(job);
                ++count;
                ++outstanding;
            }
        }
        if (rejected != OK) {
            failure(rejected);
            throw rejected;
        }
        ready.notify_one();
    }
};
// Bounded native-only retention: abnormal shutdown can lose its creator thread.
// Keep those creation slots alive rather than deleting mutable USD off-thread.
// Normal creation close frees the slot; no Python refs live here.
struct States {
    std::mutex mutex;
    std::array<std::shared_ptr<State>, 256> slots{};
    static States& get() {
        static auto* states = new States;
        return *states;
    }
    bool add(const std::shared_ptr<State>& state) {
        std::lock_guard<std::mutex> lock(mutex);
        for (auto& slot : slots)
            if (!slot) {
                slot = state;
                return true;
            }
        return false;
    }
    void retire(const std::shared_ptr<State>& state) noexcept {
        std::shared_ptr<State> removed;
        {
            std::lock_guard<std::mutex> lock(mutex);
            for (auto& slot : slots)
                if (slot == state) {
                    removed = std::move(slot);
                    break;
                }
        }
    }
};
void maybeRetire(const std::shared_ptr<State>& state) noexcept {
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (!state->abandoned && !state->closed) return;
        for (const auto& creation : state->creations)
            if (creation) return;
    }
    States::get().retire(state);
}
SdfLayerRefPtr layerInput(py::handle value) {
    py::module_::import("pxr.Sdf");
    try {
        bp::extract<SdfLayerHandle> extract(value.ptr());
        if (!extract.check()) return {};
        // RefPtr extraction transfers Python ownership; weak extraction followed
        // by an independent pin leaves the caller's ownership intact.
        const auto weak = extract();
        return weak ? SdfLayerRefPtr(weak) : SdfLayerRefPtr{};
    } catch (const bp::error_already_set&) { throw py::error_already_set(); }
}
py::object layerOutput(const SdfLayerRefPtr& layer) {
    try {
        TfPyRefPtrFactory<>::apply<SdfLayerRefPtr>::type convert;
        PyObject* output = convert(layer);
        if (!output) throw py::error_already_set();
        return py::reinterpret_steal<py::object>(output);
    } catch (const bp::error_already_set&) { throw py::error_already_set(); }
}
class Task {
public:
    ~Task() { release(); }
    Result run() {
        bool expected = false;
        if (!running.compare_exchange_strong(expected, true)) return BUSY;
        struct RunningGuard {
            std::atomic<bool>& flag;
            ~RunningGuard() { flag.store(false); }
        } runningGuard{running};
        if (!state) return CLOSED;
        if (!state->current()) {
            state->failure(BUSY);
            return BUSY;
        }
        if (used) return CLOSED;
        used = true;
        Result result = OK;
        {
            py::gil_scoped_release gil;
            bool abandoned;
            {
                std::lock_guard<std::mutex> lock(state->mutex);
                abandoned = state->abandoned;
            }
            if (abandoned) result = CANCELLED;
            else {
                try {
                    job();
                } catch (Result error) { result = error; } catch (const std::bad_alloc&) {
                    result = LIMIT;
                } catch (...) { result = OMNI_USD_LIVE_INTERNAL_ERROR; }
            }
            if (result != OK) state->failure(result);
            release();
        }
        return result;
    }
    void release() noexcept {
        if (!state) return;
        // Completion owners can wake/close the actor. Destroy outside the lock,
        // and keep the quota charged until those captures are fully retired.
        job = {};
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            if (charged) {
                --state->outstanding;
                charged = false;
            }
        }
        state->ready.notify_all();
        state.reset();
    }
    std::shared_ptr<State> state;
    std::function<void()> job;
    bool charged = false, used = false;
    std::atomic<bool> running{false};
};
class Creation {
public:
    Creation(std::shared_ptr<State> state, std::shared_ptr<CreationState> slot)
        : state(std::move(state)), slot(std::move(slot)) {}
    // Slots, not Python finalizers, own mutable USD and creation provenance.
    // Only an explicit successful creator-thread close retires this owner.
    struct Operation {
        std::shared_ptr<State> state;
        std::shared_ptr<CreationState> slot;
        host::Creation* owner = nullptr;
        Result result = OK;
        Operation(const Creation& creation, bool hostOnly, bool allowStopped = false)
            : state(creation.state), slot(creation.slot) {
            if (hostOnly && !state->current()) {
                result = BUSY;
                return;
            }
            std::lock_guard<std::mutex> lock(state->mutex);
            if (slot->active) result = BUSY;
            else if (!slot->creation) result = CLOSED;
            else if (!allowStopped && (state->abandoned || state->closed)) result = CANCELLED;
            else {
                owner = slot->creation;
                slot->active = true;
            }
        }
        ~Operation() {
            if (!owner) return;
            {
                std::lock_guard<std::mutex> lock(state->mutex);
                slot->active = false;
            }
            state->ready.notify_all();
        }
    };
    template<class F> Result call(F action, bool hostOnly = true, bool allowStopped = false) {
        Operation operation(*this, hostOnly, allowStopped);
        if (operation.result != OK) return operation.result;
        py::gil_scoped_release gil;
        return action(*operation.owner);
    }
    Result waitClosed(uint32_t timeout) {
        Operation operation(*this, false, true);
        if (operation.result == CLOSED) return OK;
        if (operation.result != OK) return operation.result;
        py::gil_scoped_release gil;
        return operation.owner->waitForClose(timeout);
    }
    Result start() {
        return call([](auto& owner) { return owner.start(); });
    }
    Result wait(uint32_t timeout) {
        return call([&](auto& owner) { return owner.wait(timeout); }, false, true);
    }
    Result prepare(uint32_t timeout) {
        return call([&](auto& owner) { return owner.prepareLayer(timeout); });
    }
    Result activate() {
        return call([](auto& owner) { return owner.activate(); });
    }
    py::tuple layer() {
        Operation operation(*this, true, true);
        if (operation.result != OK) return py::make_tuple(operation.result, py::none());
        SdfLayerRefPtr layer;
        Result result;
        {
            py::gil_scoped_release gil;
            result = operation.owner->getLayer(&layer);
        }
        return result == OK ? py::make_tuple(OK, layerOutput(layer)) : py::make_tuple(result, py::none());
    }
    py::tuple outcome() {
        Operation operation(*this, false, true);
        if (operation.result != OK) return py::make_tuple(operation.result, py::none());
        host::CreationOutcome value;
        Result result;
        {
            py::gil_scoped_release gil;
            result = operation.owner->getOutcome(&value);
        }
        if (result != OK) return py::make_tuple(result, py::none());
        py::dict error, out;
#define CREATION_ERROR(field) error[#field] = value.error.field
        CREATION_ERROR(result);
        CREATION_ERROR(operation);
        CREATION_ERROR(grpc_status_present);
        CREATION_ERROR(grpc_status);
        CREATION_ERROR(request_id);
        CREATION_ERROR(attempt_id);
        CREATION_ERROR(incident_id);
#undef CREATION_ERROR
        error["detail"] = value.error.detail;
        out["phase"] = static_cast<uint32_t>(value.phase);
        out["result"] = value.result;
        out["error"] = error;
        out["subject_id"] = value.subjectId;
        out["started"] = value.started;
        out["closing"] = value.closing;
        out["quiescent"] = value.quiescent;
        return py::make_tuple(OK, out);
    }
    Result close() {
        Operation operation(*this, true, true);
        if (operation.result == CLOSED) return OK;
        if (operation.result != OK) return operation.result;
        py::gil_scoped_release gil;
        const auto result = operation.owner->close();
        if (result != OK) return result;
        delete operation.owner; // Slot stays active through all destructor/notice callbacks.
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            slot->creation = nullptr;
            for (auto& entry : state->creations)
                if (entry == slot) {
                    entry.reset();
                    break;
                }
        }
        maybeRetire(state);
        return OK;
    }
    std::shared_ptr<State> state;
    std::shared_ptr<CreationState> slot;
};
class Context {
public:
    explicit Context(std::shared_ptr<State> value) : state(std::move(value)) {}
    ~Context() { abandon(); }
    Result close() {
        py::gil_scoped_release gil;
        Result result;
        std::shared_ptr<host::PublicationContext> retained;
        {
            std::lock_guard<std::mutex> life(lifeMutex);
            retained = native;
        }
        result = retained ? retained->close() : OK;
        if (result != OK) return result;
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            for (const auto& creation : state->creations)
                if (creation) return BUSY;
            if (state->outstanding || !state->exempt.empty()) return BUSY;
            state->closed = true;
        }
        state->ready.notify_all();
        maybeRetire(state);
        return OK;
    }
    void abandon() noexcept {
        std::vector<std::function<void()>> discarded;
        std::deque<std::function<void()>> exempt;
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            state->abandoned = true;
            state->outstanding -= state->count;
            state->count = state->head = 0;
            discarded.swap(state->queue);
            exempt.swap(state->exempt);
        }
        // Destruction only unpublishes this identity; existing callbacks retain
        // C++ state and reject future dispatch without touching Python.
        std::shared_ptr<host::PublicationContext> retired;
        {
            std::lock_guard<std::mutex> life(lifeMutex);
            retired = std::move(native);
        }
        if (retired) retired->close(); // Unpublish immediately, even if an in-flight call pins it.
        retired.reset();
        discarded.clear();
        exempt.clear();
        // Only an explicit creator-thread close retires a creation after dispatch abandonment.
        // Never close/drop a creation implicitly: it may hold an uncertain
        // mutation or actual-host layer state.
        state->ready.notify_all();
        maybeRetire(state);
    }
    void interrupt() {
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            state->interrupted = true;
        }
        state->ready.notify_all();
    }
    Result deliveryStatus() const noexcept { return state->delivery.load(); }
    py::tuple next(uint32_t timeout) {
        // Preallocate every Python owner before removing native queue work.
        auto task = std::make_shared<Task>();
        auto success = py::make_tuple(OK, task);
        Result result = OMNI_USD_LIVE_TIMEOUT;
        {
            py::gil_scoped_release gil;
            std::unique_lock<std::mutex> lock(state->mutex);
            state->ready.wait_for(lock, std::chrono::milliseconds(timeout), [&] {
                return state->count || !state->exempt.empty() || state->abandoned || state->closed ||
                    state->interrupted;
            });
            if (state->abandoned || state->closed || state->interrupted) {
                state->interrupted = false;
                result = CANCELLED;
            } else if (!state->exempt.empty()) {
                task->state = state;
                task->job = std::move(state->exempt.front());
                state->exempt.pop_front();
                result = OK;
            } else if (state->count) {
                task->state = state;
                task->charged = true;
                task->job.swap(state->queue[state->head]);
                state->head = (state->head + 1) % state->queue.size();
                --state->count;
                result = OK;
            }
        }
        return result == OK ? success : py::make_tuple(result, py::none());
    }
    py::tuple allocateCreation() {
        if (!state->current()) return py::make_tuple(BUSY, py::none());
        auto owner = std::make_shared<Creation>(state, std::make_shared<CreationState>());
        return py::make_tuple(OK, owner);
    }
    Result prepareCreation(const std::shared_ptr<Creation>& owner, py::object input, py::str uri) {
        if (!owner || owner->state != state) return INVALID;
        if (!state->current()) return BUSY;
        if (PyUnicode_GetLength(uri.ptr()) > 65536) return LIMIT;
        Py_ssize_t length = 0;
        const char* text = PyUnicode_AsUTF8AndSize(uri.ptr(), &length);
        if (!text) throw py::error_already_set();
        if (length > 65536) return LIMIT;
        if (std::memchr(text, 0, static_cast<size_t>(length))) return INVALID;
        auto layer = layerInput(input);
        if (!layer) return INVALID;
        const auto& slot = owner->slot;
        Result result = OK;
        bool reserved = false;
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            if (slot->active) result = BUSY;
            else if (slot->creation) result = OMNI_USD_LIVE_ALREADY_EXISTS;
            else if (state->closed || state->abandoned) result = CLOSED;
            else
                for (auto& entry : state->creations)
                    if (!entry) {
                        entry = slot;
                        slot->active = true;
                        reserved = true;
                        break;
                    }
            if (result == OK && !reserved) result = LIMIT;
        }
        if (result != OK) return result;
        std::shared_ptr<host::PublicationContext> context;
        {
            std::lock_guard<std::mutex> life(lifeMutex);
            context = native;
        }
        host::Creation* admitted = nullptr;
        {
            py::gil_scoped_release gil;
            result = !context ? CANCELLED : context->prepareCreation(*layer, text, &admitted);
        }
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            if (state->abandoned || state->closed) result = CANCELLED;
        }
        // No RPC has issued. A reentrant abandonment may safely retire this
        // unstarted preparation on its creator, outside every slot/life lock.
        if (result != OK) {
            py::gil_scoped_release gil;
            delete admitted;
            admitted = nullptr;
        }
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            slot->creation = admitted;
            slot->active = false;
            if (result != OK)
                for (auto& entry : state->creations)
                    if (entry == slot) {
                        entry.reset();
                        break;
                    }
        }
        state->ready.notify_all();
        maybeRetire(state);
        return result;
    }
    py::tuple prepareRetainedCreation(py::object input, py::str uri) {
        if (!state->current()) return py::make_tuple(BUSY, uint64_t(0));
        if (PyUnicode_GetLength(uri.ptr()) > 65536) return py::make_tuple(LIMIT, uint64_t(0));
        Py_ssize_t length = 0;
        const char* text = PyUnicode_AsUTF8AndSize(uri.ptr(), &length);
        if (!text) throw py::error_already_set();
        if (length > 65536) return py::make_tuple(LIMIT, uint64_t(0));
        if (std::memchr(text, 0, static_cast<size_t>(length))) return py::make_tuple(INVALID, uint64_t(0));
        bool stopped;
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            stopped = state->abandoned || state->closed;
        }
        if (stopped) return py::make_tuple(CLOSED, uint64_t(0));
        auto layer = layerInput(input);
        if (!layer) return py::make_tuple(INVALID, uint64_t(0));
        std::shared_ptr<host::PublicationContext> context;
        {
            std::lock_guard<std::mutex> life(lifeMutex);
            context = native;
        }
        uint64_t subject = 0;
        Result result;
        {
            py::gil_scoped_release gil;
            result = !context ? CANCELLED : context->prepareRetainedCreation(*layer, text, &subject);
        }
        // The process registry already owns the workflow. Python allocation or
        // delivery failure cannot strand it; global enumeration remains valid.
        return py::make_tuple(result, subject);
    }
    Result acquireRetainedCreation(const std::shared_ptr<Creation>& owner, uint64_t subject) {
        if (!owner || owner->state != state) return INVALID;
        if (!state->current()) return BUSY;
        const auto& slot = owner->slot;
        Result result = OK;
        bool reserved = false;
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            if (slot->active) result = BUSY;
            else if (slot->creation) result = OMNI_USD_LIVE_ALREADY_EXISTS;
            else if (state->closed || state->abandoned) result = CLOSED;
            else
                for (auto& entry : state->creations)
                    if (!entry) {
                        entry = slot;
                        slot->active = true;
                        reserved = true;
                        break;
                    }
            if (result == OK && !reserved) result = LIMIT;
        }
        if (result != OK) return result;
        host::Creation* admitted = nullptr;
        {
            py::gil_scoped_release gil;
            result = host::acquireRetainedCreation(subject, &admitted);
        }
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            // Acquisition can transfer already committed or uncertain work.
            // Even concurrent abandonment MUST retain the transferred owner.
            slot->creation = admitted;
            slot->active = false;
            if (!admitted)
                for (auto& entry : state->creations)
                    if (entry == slot) {
                        entry.reset();
                        break;
                    }
        }
        state->ready.notify_all();
        maybeRetire(state);
        return result;
    }
    std::shared_ptr<State> state;
    std::mutex lifeMutex;
    std::shared_ptr<host::PublicationContext> native;
};
py::tuple createContext(uint32_t limit) {
    if (!limit || limit > 256) return py::make_tuple(INVALID, py::none());
    auto state = std::make_shared<State>(limit);
    auto owner = std::make_shared<Context>(state);
    auto success = py::make_tuple(OK, owner);
    if (!States::get().add(state)) return py::make_tuple(LIMIT, py::none());
    host::PublicationContext* admitted = nullptr;
    Result result;
    {
        py::gil_scoped_release gil;
        result = host::createPublicationContext([state](std::function<void()> job) { state->enqueue(std::move(job)); },
            [state] { return state->current(); }, &admitted);
    }
    owner->native.reset(admitted);
    if (result != OK) {
        States::get().retire(state);
        return py::make_tuple(result, py::none());
    }
    return success;
}
}
void bindLiveHost(py::module_& module) {
    py::class_<Task, std::shared_ptr<Task>>(module, "PublicationTask").def("run", &Task::run);
    py::class_<Creation, std::shared_ptr<Creation>>(module, "Creation")
        .def("start", &Creation::start)
        .def("wait", &Creation::wait)
        .def("outcome", &Creation::outcome)
        .def("prepare_layer", &Creation::prepare)
        .def("activate", &Creation::activate)
        .def("layer", &Creation::layer)
        .def("close", &Creation::close)
        .def("wait_closed", &Creation::waitClosed);
    py::class_<Context, std::shared_ptr<Context>>(module, "PublicationContext")
        .def("next_task", &Context::next, py::arg("timeout_ms") = 0)
        .def("allocate_creation", &Context::allocateCreation)
        .def("prepare_creation", &Context::prepareCreation)
        .def("prepare_retained_creation", &Context::prepareRetainedCreation)
        .def("acquire_retained_creation", &Context::acquireRetainedCreation)
        .def("close", &Context::close)
        .def("abandon", &Context::abandon)
        .def("interrupt_wait", &Context::interrupt)
        .def("delivery_status", &Context::deliveryStatus);
    module.def("create_publication_context", &createContext, py::arg("limit") = 64);
    module.def("next_retained_creation", [](uint64_t after) {
        host::RetainedCreationInfo value;
        Result result;
        {
            py::gil_scoped_release gil;
            result = host::nextRetainedCreation(after, &value);
        }
        if (result != OK) return py::make_tuple(result, py::none());
        py::dict info;
        info["subject_id"] = value.subjectId;
        info["live_uri"] = value.liveUri;
        return py::make_tuple(OK, info);
    }, py::arg("after_subject") = 0);
    module.def("release_retained_creation", [](uint64_t subject) {
        py::gil_scoped_release gil;
        return host::releaseRetainedCreation(subject);
    });
    module.def("save_layer", [](py::object input, uint32_t timeout) {
        auto layer = layerInput(input);
        if (!layer) return INVALID;
        py::gil_scoped_release gil;
        return host::save(*layer, timeout);
    }, py::arg("layer"), py::arg("timeout_ms") = 0);
}
