// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "LiveMetricDuration.h"
#include <chrono>
#include <cstdint>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace omni::usd::live {
class DispatchExecutor;
// Jobs reserve their queue node at admission. An owner must retain the job
// until queued/running work is quiescent; publication never allocates a node.
class DispatchJob : public std::enable_shared_from_this<DispatchJob> {
public:
    virtual ~DispatchJob() = default;
private:
    friend class DispatchExecutor;
    DispatchJob* next_ = nullptr;
    bool linked_ = false;
    std::chrono::steady_clock::time_point enqueuedAt_{};
    virtual void dispatch() noexcept = 0;
};

struct DispatchMetrics {
    uint64_t enqueued = 0, started = 0, completed = 0;
    uint64_t queued = 0, running = 0, queueHighWater = 0;
    uint64_t oldestQueuedAgeMs = 0, oldestRunningAgeMs = 0;
    uint32_t configuredWorkers = 0, liveWorkers = 0;
    MetricDuration queueWait, dispatchDuration;
};

class DispatchExecutor final {
public:
    using Clock = std::chrono::steady_clock;
    // Optional private clock must be thread-safe/noexcept/non-reentrant and outlive this
    // executor. Shutdown waits retain real steady-clock failure deadlines.
    explicit DispatchExecutor(uint32_t workers, Clock::time_point (*now)(void*) noexcept = nullptr,
        void* clockContext = nullptr);
    ~DispatchExecutor();
    void enqueue(DispatchJob& job) noexcept;
    void beginShutdown();
    bool waitForShutdown(std::chrono::steady_clock::time_point deadline);
    DispatchMetrics snapshot() const noexcept;
    static bool isWorker() noexcept;
    // Test seam: a worker calls hook after each job's dispatch returns, while it
    // still retains the job, before its completion accounting. It captures the
    // hook when the job starts, so a job started before a reset still calls the
    // previous hook.
    void setAfterDispatchForTest(void (*hook)(void*, DispatchJob&) noexcept, void* context) noexcept;
    // Mark an external host callback as unable to block on runtime work. This
    // prevents a notice/publication callback waiting for its own retirement.
    class NonblockingScope final {
    public:
        NonblockingScope() noexcept;
        ~NonblockingScope();
        NonblockingScope(const NonblockingScope&) = delete;
        NonblockingScope& operator=(const NonblockingScope&) = delete;
    private:
        bool previous_;
    };
private:
    mutable std::mutex mutex_;
    std::condition_variable ready_, stopped_;
    DispatchJob* first_ = nullptr;
    DispatchJob* last_ = nullptr;
    bool stopping_ = false;
    uint32_t activeWorkers_ = 0;
    void (*afterDispatch_)(void*, DispatchJob&) noexcept = nullptr;
    void* afterDispatchContext_ = nullptr;
    Clock::time_point (*const nowFunction_)(void*) noexcept;
    void* const clockContext_;
    Clock::time_point now() const noexcept { return nowFunction_ ? nowFunction_(clockContext_) : Clock::now(); }
    struct WorkerState {
        bool running = false;
        Clock::time_point started{};
    };
    std::vector<WorkerState> workerStates_;
    DispatchMetrics metrics_;
    std::vector<std::thread> workers_;
    void work(size_t index) noexcept;
};
} // namespace omni::usd::live
