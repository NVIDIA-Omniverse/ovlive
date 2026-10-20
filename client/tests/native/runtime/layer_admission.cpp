// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "LiveLayerAdmission.h"
#include "LiveUri.h"
#include "LiveDispatch.h"
#include "LiveDiagnostics.h"
#include "LiveStatus.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <future>
#include <functional>
#include <iostream>
#include <mutex>
#include <new>
#include <stdexcept>
#include <thread>
#include <vector>

void liveTestFailAllocationsAfter(size_t) noexcept;
void liveTestRestoreAllocations() noexcept;
using namespace omni::usd::live;
namespace {
template<class T> T record() {
    T out{}; out.struct_size = sizeof(out); out.abi_version = OMNI_USD_LIVE_ABI_VERSION; return out;
}
template<class Action> void rejects(OmniUsdLiveResult expected, Action action) {
    bool rejected = false;
    try { action(); } catch (OmniUsdLiveResult result) { assert(result == expected); rejected = true; }
    assert(rejected);
}
struct Fixture {
    std::shared_ptr<Runtime> runtime;
    OmniUsdLiveEndpoint endpoint = 0, other = 0;
    Fixture() {
        auto settings=record<OmniUsdLiveRuntimeOptions>();
        settings.diagnostic_min_severity=OMNI_USD_LIVE_SEVERITY_INFO;
        assert(omniUsdLiveInitialize(&settings) == OMNI_USD_LIVE_OK);
        runtime = getRuntime();
        auto options = record<OmniUsdLiveEndpointOptions>(); options.authority = "LOCALHOST:35495";
        assert(omniUsdLiveRegisterEndpoint(&options, &endpoint) == OMNI_USD_LIVE_OK);
        options.authority = "127.0.0.1:35496";
        assert(omniUsdLiveRegisterEndpoint(&options, &other) == OMNI_USD_LIVE_OK);
    }
    uint64_t dependencies(OmniUsdLiveEndpoint id) {
        auto owner = runtime->findEndpoint(id);
        std::lock_guard<std::mutex> lock(owner->mutex); return owner->dependents;
    }
    ~Fixture() {
        assert(omniUsdLiveUnregisterEndpoint(endpoint, 3000) == OMNI_USD_LIVE_OK);
        assert(omniUsdLiveUnregisterEndpoint(other, 3000) == OMNI_USD_LIVE_OK);
        runtime.reset();
        assert(omniUsdLiveShutdown(3000) == OMNI_USD_LIVE_OK);
    }
};
void concurrentDuplicateAdmission() {
    Fixture f;
    LayerAdmissionRegistry registry(*f.runtime, 1);
    std::promise<void> release;
    auto gate = release.get_future().share();
    std::atomic<unsigned> ready{0}, won{0}, duplicates{0};
    std::array<std::shared_ptr<LayerAdmission>, 16> held;
    std::vector<std::thread> threads;
    for (size_t i = 0; i < held.size(); ++i) threads.emplace_back([&, i] {
        ++ready; gate.wait();
        try {
            held[i] = registry.reserve(f.endpoint, "opaque/session", "nested/root.live");
            ++won;
        } catch (OmniUsdLiveResult result) { assert(result == OMNI_USD_LIVE_ALREADY_EXISTS); ++duplicates; }
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (ready != held.size()) { assert(std::chrono::steady_clock::now() < deadline); std::this_thread::yield(); }
    release.set_value();
    for (auto& thread : threads) thread.join();
    // The winner stays held across every losing attempt; scheduling cannot turn
    // this into sixteen sequential successful reservations.
    assert(won == 1 && duplicates == 15 && f.dependencies(f.endpoint) == 1);
    rejects(OMNI_USD_LIVE_ALREADY_EXISTS, [&] { registry.reserve(f.endpoint, "opaque/session", "nested/root.live"); });
    assert(omniUsdLiveUnregisterEndpoint(f.endpoint, 0) == OMNI_USD_LIVE_BUSY);
    for (auto& owner : held) owner.reset();
    assert(f.dependencies(f.endpoint) == 0);
    auto next = registry.reserve(f.endpoint, "opaque/session", "nested/root.live");
    assert(next->canonicalUri() == "live://localhost:35495/opaque%2Fsession/nested/root.live");
}
void capacityClaimsAndValidation() {
    Fixture f;
    LayerAdmissionRegistry registry(*f.runtime, 2);
    auto first = registry.reserve(f.endpoint, "s @?#% \xC3\xA9", "nested/name #.live");
    assert(first->endpoint() == f.endpoint && first->session() == "s @?#% \xC3\xA9" && first->layer() == "nested/name #.live");
    const auto parsed = parseUri(first->canonicalUri());
    assert(parsed.sessionId == first->session() && parsed.layerName == first->layer());
    auto duplicateLease = first;
    registry.validate(first, f.endpoint, first->session(), first->layer());
    registry.claim(first, f.endpoint, first->session(), first->layer());
    rejects(OMNI_USD_LIVE_ALREADY_EXISTS, [&] { registry.claim(duplicateLease, f.endpoint, first->session(), first->layer()); });
    rejects(OMNI_USD_LIVE_INVALID_ARGUMENT, [&] { registry.validate({}, f.endpoint, "s", "root.live"); });
    rejects(OMNI_USD_LIVE_INVALID_ARGUMENT, [&] { registry.validate(first, f.other, first->session(), first->layer()); });
    rejects(OMNI_USD_LIVE_INVALID_ARGUMENT, [&] { registry.claim(first, f.endpoint, "other", first->layer()); });
    rejects(OMNI_USD_LIVE_INVALID_ARGUMENT, [&] { registry.validate(first, f.endpoint, first->session(), "other.live"); });
    LayerAdmissionRegistry independent(*f.runtime, 2);
    rejects(OMNI_USD_LIVE_INVALID_ARGUMENT, [&] { independent.validate(first, f.endpoint, first->session(), first->layer()); });
    auto anotherRuntime = std::make_shared<Runtime>(RuntimeOptions::copy(nullptr));
    rejects(OMNI_USD_LIVE_INVALID_ARGUMENT, [&] {
        anotherRuntime->layerAdmissions->validate(first, f.endpoint, first->session(), first->layer());
    });
    anotherRuntime.reset();
    auto second = registry.reserve(f.other, "s @?#% \xC3\xA9", "nested/name #.live");
    rejects(OMNI_USD_LIVE_RESOURCE_EXHAUSTED, [&] { registry.reserve(f.endpoint, "new", "root.live"); });
    assert(f.dependencies(f.endpoint) == 1 && f.dependencies(f.other) == 1);
    first.reset();
    rejects(OMNI_USD_LIVE_ALREADY_EXISTS, [&] { registry.reserve(f.endpoint, "s @?#% \xC3\xA9", "nested/name #.live"); });
    duplicateLease.reset(); second.reset();
    assert(f.dependencies(f.endpoint) == 0 && f.dependencies(f.other) == 0);
    for (unsigned i = 0; i < 64; ++i) {
        auto reused = registry.reserve(f.endpoint, std::to_string(i), "root.live");
        assert(f.dependencies(f.endpoint) == 1);
    }
    assert(f.dependencies(f.endpoint) == 0);
    LayerAdmissionRegistry empty(*f.runtime, 0);
    rejects(OMNI_USD_LIVE_RESOURCE_EXHAUSTED, [&] { empty.reserve(f.endpoint, "s", "root.live"); });
    rejects(OMNI_USD_LIVE_INVALID_HANDLE, [&] { registry.reserve(UINT64_MAX, "s", "root.live"); });
    bool invalid = false;
    try { registry.reserve(f.endpoint, "s", "../escape.live"); }
    catch (const std::invalid_argument&) { invalid = true; }
    assert(invalid && f.dependencies(f.endpoint) == 0);
}
void independentRegistryStateLifetime() {
    Fixture f;
    auto registry = std::make_unique<LayerAdmissionRegistry>(*f.runtime, 1);
    auto retained = registry->reserve(f.endpoint, "retained", "root.live");
    registry.reset();
    assert(retained->canonicalUri() == "live://localhost:35495/retained/root.live");
    assert(f.dependencies(f.endpoint) == 1);
    assert(omniUsdLiveUnregisterEndpoint(f.endpoint, 0) == OMNI_USD_LIVE_BUSY);
    retained.reset();
    assert(f.dependencies(f.endpoint) == 0);
}
void allocationRollback() {
    Fixture f;
    LayerAdmissionRegistry registry(*f.runtime, 1);
    const std::string session(96, 's'), layer = std::string(96, 'l') + ".live";
    std::shared_ptr<LayerAdmission> accepted;
    unsigned failures = 0;
    for (size_t permit = 0; permit < 128; ++permit) {
        bool rejected = false;
        liveTestFailAllocationsAfter(permit);
        try { accepted = registry.reserve(f.endpoint, session, layer); }
        catch (const std::bad_alloc&) { rejected = true; }
        liveTestRestoreAllocations();
        if (!rejected) break;
        ++failures;
        assert(!accepted && f.dependencies(f.endpoint) == 0);
        // A failed publication cannot leave a live slot or a phantom key.
        auto control = registry.reserve(f.endpoint, "control", "root.live");
        assert(control && f.dependencies(f.endpoint) == 1);
    }
    assert(accepted && failures >= 4 && f.dependencies(f.endpoint) == 1);
    // Claim is an allocation-free ownership boundary.
    liveTestFailAllocationsAfter(0);
    registry.claim(accepted, f.endpoint, session, layer);
    liveTestRestoreAllocations();
    rejects(OMNI_USD_LIVE_ALREADY_EXISTS, [&] { registry.claim(accepted, f.endpoint, session, layer); });
    accepted.reset();
    assert(f.dependencies(f.endpoint) == 0);
}
// A reserve with a deadline waits only for a retiring holder; any other
// holder, or a retiring one that outlives the deadline, is ALREADY_EXISTS.
void retiringHolderRelease() {
    using Clock = std::chrono::steady_clock;
    Fixture f;
    LayerAdmissionRegistry registry(*f.runtime, 2);
    const auto never = [] { return Clock::now() + std::chrono::hours(1); };
    auto held = registry.reserve(f.endpoint, "s", "root.live");
    // Not retiring: refused at once, so an hour's deadline never elapses.
    rejects(OMNI_USD_LIVE_ALREADY_EXISTS, [&] { registry.reserve(f.endpoint, "s", "root.live", never()); });
    assert(registry.waiting() == 0);
    held->markRetiring();
    held->markRetiring(); // One-way and idempotent.
    // Without a deadline a retiring holder still refuses at once.
    rejects(OMNI_USD_LIVE_ALREADY_EXISTS, [&] { registry.reserve(f.endpoint, "s", "root.live"); });
    auto other = registry.reserve(f.endpoint, "s", "other.live"); // Other keys are unaffected.
    auto contender = std::async(std::launch::async, [&] { return registry.reserve(f.endpoint, "s", "root.live", never()); });
    const auto barrier = Clock::now() + std::chrono::seconds(5);
    while (registry.waiting() != 1) { assert(Clock::now() < barrier); std::this_thread::yield(); }
    held.reset(); // The last owner's release wakes the waiting reserve.
    assert(contender.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
    auto next = contender.get();
    assert(next && next->canonicalUri() == "live://localhost:35495/s/root.live" && registry.waiting() == 0);
    // The reused slot is not retiring until its new holder says so.
    rejects(OMNI_USD_LIVE_ALREADY_EXISTS, [&] { registry.reserve(f.endpoint, "s", "root.live", never()); });
    // A retiring holder pinned by a queued host job refuses at once, and a
    // waiting reserve that sees the pin refuses too.
    next->markRetiring();
    next->pinToHost(true);
    rejects(OMNI_USD_LIVE_ALREADY_EXISTS, [&] { registry.reserve(f.endpoint, "s", "root.live", never()); });
    next->pinToHost(false);
    auto pinned = std::async(std::launch::async, [&] {
        try { registry.reserve(f.endpoint, "s", "root.live", never()); } catch (OmniUsdLiveResult result) { return result; }
        return OmniUsdLiveResult(OMNI_USD_LIVE_OK);
    });
    const auto pinBarrier = Clock::now() + std::chrono::seconds(5);
    while (registry.waiting() != 1) { assert(Clock::now() < pinBarrier); std::this_thread::yield(); }
    next->pinToHost(true);
    assert(pinned.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
    assert(pinned.get() == OMNI_USD_LIVE_ALREADY_EXISTS && registry.waiting() == 0);
    next->pinToHost(false);
    // A retiring holder that is never released: refused at the deadline.
    rejects(OMNI_USD_LIVE_ALREADY_EXISTS, [&] {
        registry.reserve(f.endpoint, "s", "root.live", Clock::now() + std::chrono::milliseconds(20));
    });
    assert(registry.waiting() == 0 && f.dependencies(f.endpoint) == 2);
    next.reset(); other.reset();
    assert(f.dependencies(f.endpoint) == 0);
}
}
void layerAdmissionTests() {
    concurrentDuplicateAdmission();
    retiringHolderRelease();
    capacityClaimsAndValidation();
    independentRegistryStateLifetime();
    allocationRollback();
    std::cout << "layer admission: 16 contenders, retiring holders, canonical ownership, one-shot claims and allocation rollback passed\n";
}
