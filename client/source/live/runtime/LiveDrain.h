// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <omni/usd/live/Live.h>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace omni::usd::live {
struct DrainSignal;
class DrainTicket;

// A captured cutoff includes admitted tickets only. Completing a ticket means
// its complete application/callback obligation has retired; callers must fail
// or close the state before retiring abandoned work. This tracker never runs
// callbacks, cancels work, or advances a transport/application pipeline.
class DrainState final : public std::enable_shared_from_this<DrainState> {
public:
    struct Fence {
        uint64_t cutoff = 0;
        OmniUsdLiveResult baselineResult = OMNI_USD_LIVE_OK;
    };
    std::shared_ptr<DrainTicket> prepare(); // Allocates an inactive ticket.
    Fence capture() const;
    OmniUsdLiveResult poll(const Fence& fence) const;
    // Fails every current obligation and later capture. With `admitting`,
    // new tickets still admit (a read that outlives issuance keeps retiring
    // its frames) until an ordinary failure or close; a failure without it
    // ends admission for good. Returns whether this call latched the first
    // failure result (a later OUTCOME_UNKNOWN still overrides it).
    bool fail(OmniUsdLiveResult result, bool admitting = false) noexcept;
    void close() noexcept;
private:
    friend class DrainRegistry;
    friend class DrainTicket;
    struct Probe {
        OmniUsdLiveResult result;
        bool pending;
    };
    Probe probe(const Fence& fence) const;
    explicit DrainState(std::shared_ptr<DrainSignal> signal);
    std::shared_ptr<DrainSignal> signal_;
    mutable std::mutex mutex_;
    DrainTicket* first_ = nullptr;
    uint64_t lastOrdinal_ = 0;
    uint64_t failedFrom_ = 0; // Earliest pending obligation abandoned by failure.
    OmniUsdLiveResult result_ = OMNI_USD_LIVE_OK;
    bool admitting_ = false; // Admission continues after an issuance-only failure.
    bool closed_ = false;
};

// The shared ticket retains its state; the state owns only intrusive pointers.
// Admission and completion allocate nothing and are serialized with capture.
// Admission is idempotent, including after successful completion; failed
// admission is permanently rejected. Completing an unadmitted ticket prevents
// subsequent admission. Destruction completes an admitted ticket.
class DrainTicket final {
public:
    ~DrainTicket();
    DrainTicket(const DrainTicket&) = delete;
    DrainTicket& operator=(const DrainTicket&) = delete;
    OmniUsdLiveResult admit() noexcept;
    void complete() noexcept;
private:
    friend class DrainState;
    explicit DrainTicket(std::shared_ptr<DrainState> state);
    enum class Phase { Prepared, Pending, Retired };
    std::shared_ptr<DrainState> state_;
    DrainTicket* previous_ = nullptr;
    DrainTicket* next_ = nullptr;
    uint64_t ordinal_ = 0;
    Phase phase_ = Phase::Prepared;
    OmniUsdLiveResult admissionResult_ = OMNI_USD_LIVE_CANCELLED;
};

// Registry ownership is weak. A capture retains the registered live states,
// with a per-state cutoff taken after releasing the registry lock; it is not
// an atomic snapshot across all states. Later admissions never extend it.
// Callers bound pending ticket counts through their existing admission quotas.
class DrainRegistry final {
public:
    using Clock = std::chrono::steady_clock;
    struct Target {
        std::shared_ptr<DrainState> state;
        DrainState::Fence fence;
    };
    struct Capture {
        std::vector<Target> targets;
    };
    explicit DrainRegistry(size_t maxStates);
    std::shared_ptr<DrainState> makeState();
    Capture capture();
    // All targets share one absolute deadline. An already failed state is an
    // error even if empty at capture. A successful captured completion wins
    // over a failure introduced by later work. Existing failures are sticky;
    // OUTCOME_UNKNOWN overrides other failures. No target state is mutated.
    OmniUsdLiveResult wait(const Capture& captured, Clock::time_point deadline, bool allowBlock = true) const;
    // The timeout starts before capture. Zero polls; allowBlock=false returns
    // BUSY for pending work, including timeout zero (worker self-wait policy).
    OmniUsdLiveResult wait(uint32_t timeoutMs, bool allowBlock = true);
private:
    const size_t maxStates_;
    std::shared_ptr<DrainSignal> signal_;
    std::mutex mutex_;
    std::vector<std::weak_ptr<DrainState>> states_;
};
} // namespace omni::usd::live
