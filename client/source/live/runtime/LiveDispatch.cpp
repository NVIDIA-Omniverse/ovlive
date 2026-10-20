// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "LiveDispatch.h"
#include <cassert>

namespace omni::usd::live {
namespace {
thread_local bool worker = false;
void increment(uint64_t& value) noexcept {
    if (value != UINT64_MAX) ++value;
}
uint64_t elapsedMs(DispatchExecutor::Clock::time_point start, DispatchExecutor::Clock::time_point end) noexcept {
    if (end <= start) return 0;
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
}
}
bool DispatchExecutor::isWorker() noexcept { return worker; }
DispatchExecutor::NonblockingScope::NonblockingScope() noexcept : previous_(worker) { worker = true; }
DispatchExecutor::NonblockingScope::~NonblockingScope() { worker = previous_; }
DispatchExecutor::DispatchExecutor(uint32_t count, Clock::time_point (*clock)(void*) noexcept, void* context)
    : nowFunction_(clock), clockContext_(context), workerStates_(count) {
    metrics_.configuredWorkers = count;
    workers_.reserve(count);
    try {
        for (uint32_t i = 0; i < count; ++i) {
            std::lock_guard<std::mutex> lock(mutex_);
            workers_.emplace_back([this, i] { work(i); });
            ++activeWorkers_;
        }
    } catch (...) {
        beginShutdown();
        for (auto& thread : workers_) thread.join();
        throw;
    }
}
DispatchExecutor::~DispatchExecutor() {
    beginShutdown();
    for (auto& thread : workers_) thread.join();
}
void DispatchExecutor::enqueue(DispatchJob& job) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    assert(!stopping_ && !job.linked_);
    job.next_ = nullptr;
    job.linked_ = true;
    job.enqueuedAt_ = now();
    if (last_) last_->next_ = &job;
    else first_ = &job;
    last_ = &job;
    increment(metrics_.enqueued);
    assert(metrics_.queued != UINT64_MAX);
    ++metrics_.queued;
    metrics_.queueHighWater = std::max(metrics_.queueHighWater, metrics_.queued);
    ready_.notify_one();
}
void DispatchExecutor::work(size_t index) noexcept {
    worker = true;
    for (;;) {
        std::shared_ptr<DispatchJob> job;
        void (*afterDispatch)(void*, DispatchJob&) noexcept = nullptr;
        void* afterDispatchContext = nullptr;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            ready_.wait(lock, [&] { return stopping_ || first_; });
            if (!first_) {
                --activeWorkers_;
                stopped_.notify_all();
                return;
            }
            job = first_->shared_from_this();
            first_ = first_->next_;
            if (!first_) last_ = nullptr;
            job->next_ = nullptr;
            job->linked_ = false;
            auto& state = workerStates_[index];
            state.running = true;
            state.started = now();
            assert(metrics_.queued);
            --metrics_.queued;
            increment(metrics_.started);
            ++metrics_.running;
            metrics_.queueWait.observe(elapsedMs(job->enqueuedAt_, state.started));
            afterDispatch = afterDispatch_;
            afterDispatchContext = afterDispatchContext_;
        }
        job->dispatch();
        const auto finished = now(); // Excludes accounting-lock wait and owner destruction.
        if (afterDispatch) afterDispatch(afterDispatchContext, *job);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto& state = workerStates_[index];
            metrics_.dispatchDuration.observe(elapsedMs(state.started, finished));
            state.running = false;
            assert(metrics_.running);
            --metrics_.running;
            increment(metrics_.completed);
        }
    }
}
DispatchMetrics DispatchExecutor::snapshot() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto result = metrics_;
    result.liveWorkers = activeWorkers_;
    const auto captured = now();
    if (first_) result.oldestQueuedAgeMs = elapsedMs(first_->enqueuedAt_, captured);
    for (const auto& state : workerStates_)
        if (state.running)
            result.oldestRunningAgeMs = std::max(result.oldestRunningAgeMs, elapsedMs(state.started, captured));
    return result;
}
void DispatchExecutor::setAfterDispatchForTest(void (*hook)(void*, DispatchJob&) noexcept, void* context) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    afterDispatch_ = hook;
    afterDispatchContext_ = context;
}
void DispatchExecutor::beginShutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
    ready_.notify_all();
}
bool DispatchExecutor::waitForShutdown(std::chrono::steady_clock::time_point deadline) {
    std::unique_lock<std::mutex> lock(mutex_);
    return stopped_.wait_until(lock, deadline, [&] { return !activeWorkers_; });
}
} // namespace omni::usd::live
