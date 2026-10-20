// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "LiveLayerProgress.h"
#include "LiveLayerActor.h"
#include "LiveUnary.h"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <new>

// Faults are restricted to this test's calling thread and disarmed before
// assertions/cleanup. Production worker allocations are unaffected.
namespace {
thread_local size_t allocationCountdown = std::numeric_limits<size_t>::max();
}
// Shared by other foundation components; affects only the calling test thread.
void liveTestFailAllocationsAfter(size_t count) noexcept { allocationCountdown = count; }
void liveTestRestoreAllocations() noexcept { allocationCountdown = std::numeric_limits<size_t>::max(); }
// Keep the replacement allocation pair out of line. In optimized builds GCC
// otherwise inlines free() into a delete expression and diagnoses a spurious
// mismatched-new-delete against our malloc-backed replacement operator new.
#if defined(_MSC_VER)
#define LIVE_TEST_ALLOC_NOINLINE __declspec(noinline)
#else
#define LIVE_TEST_ALLOC_NOINLINE __attribute__((noinline))
#endif
LIVE_TEST_ALLOC_NOINLINE void* operator new(size_t size)
{
    if (allocationCountdown != std::numeric_limits<size_t>::max()) {
        if (!allocationCountdown) throw std::bad_alloc();
        --allocationCountdown;
    }
    if (auto* result = std::malloc(size ? size : 1)) return result;
    throw std::bad_alloc();
}
LIVE_TEST_ALLOC_NOINLINE void* operator new[](size_t size) { return ::operator new(size); }
LIVE_TEST_ALLOC_NOINLINE void operator delete(void* value) noexcept { std::free(value); }
LIVE_TEST_ALLOC_NOINLINE void operator delete[](void* value) noexcept { std::free(value); }
LIVE_TEST_ALLOC_NOINLINE void operator delete(void* value, size_t) noexcept { std::free(value); }
LIVE_TEST_ALLOC_NOINLINE void operator delete[](void* value, size_t) noexcept { std::free(value); }
#undef LIVE_TEST_ALLOC_NOINLINE

using namespace omni::usd::live;
namespace {
using Gate = LayerProgress::Gate;
using State = LayerProgress::Disposition;
std::shared_ptr<const void> value(uint64_t number = 1)
{ return std::make_shared<const uint64_t>(number); }
LayerProgress::Coverage proof(std::shared_ptr<const void> owner,
                             LayerProgress::CoverageKind kind = LayerProgress::CoverageKind::Update)
{
    // Synthetic immutable operation identity, independent of controller IDs.
    // Real actors must use SHA-256 over complete validated bytes.
    auto number = *std::static_pointer_cast<const uint64_t>(owner);
    LayerProgress::ContentHash hash{};
    for (unsigned i = 0; i < 8; ++i) hash[i] = static_cast<uint8_t>(number >> (8 * i));
    return {std::move(owner), hash, kind};
}
LayerIdentity key(uint64_t endpoint = 1, std::string layer = "root.live")
{ return {endpoint, "session", std::move(layer), "writer", 1}; }
template <class F> void error(OmniUsdLiveResult expected, F fn)
{
    bool caught = false;
    try { fn(); }
    catch (OmniUsdLiveResult result) { assert(result == expected); caught = true; }
    assert(caught);
}
void open(LayerProgress& state, uint64_t cursor)
{
    auto read = state.startRead();
    assert(read.sequence == cursor && read.object == 1);
    auto ready = state.caughtUp(read.generation, read.object, cursor);
    assert(ready.accepted && ready.initialLatest);
}

struct PublicationState {
    uint64_t id = 0, commits = 0;
    std::shared_ptr<const void> visible, candidate;
    static void commit(void* context, uint64_t id) noexcept
    {
        auto& state = *static_cast<PublicationState*>(context);
        state.id = id; ++state.commits;
        state.visible.swap(state.candidate); // Displaced owner remains caller-owned.
    }
    LayerProgress::Publication hook() { return {commit, this}; }
};

void publicationFailuresAndRetirement()
{
    LayerProgress state(key(), 10, value(), 512, 100); open(state, 10);
    PublicationState publication;
    bool retired = false;
    publication.visible = std::shared_ptr<const void>(new int(0), [&](const void* ptr) {
        assert(state.snapshot().retained > 0); // Safe only outside progress lock.
        delete static_cast<const int*>(ptr); retired = true;
    });
    publication.candidate = value(1);
    const auto intent = value(2);
    allocationCountdown = 0;
    bool failed = false;
    try { state.admit(intent, publication.hook()); }
    catch (const std::bad_alloc&) { failed = true; }
    allocationCountdown = std::numeric_limits<size_t>::max();
    assert(failed && !publication.commits && !publication.id && !retired);
    assert(!state.snapshot().retained && !state.snapshot().queued);
    assert(state.snapshot().localAdmitted==0 && state.snapshot().localRetired==0 && state.snapshot().initialApplied==10);
    assert(state.admit(intent, publication.hook()) == 1);
    assert(publication.id == 1 && publication.commits == 1 && !retired);
    publication.candidate.reset(); assert(retired);

    // Permit the map allocation, fail a later deque block/map allocation when
    // FIFO growth needs it. Do not assume a particular deque block size.
    unsigned rolledBack = 0;
    for (uint64_t expected = 2; expected <= 256; ++expected) {
        publication.candidate = value(expected);
        const auto commits = publication.commits;
        allocationCountdown = 1;
        failed = false;
        uint64_t accepted = 0;
        try { accepted = state.admit(intent, publication.hook()); }
        catch (const std::bad_alloc&) { failed = true; }
        allocationCountdown = std::numeric_limits<size_t>::max();
        if (failed) {
            ++rolledBack;
            assert(publication.commits == commits && publication.id == expected - 1);
            assert(state.snapshot().retained == expected - 1 && state.snapshot().queued == expected - 1);
            assert(state.snapshot().localAdmitted==expected-1 && state.snapshot().localRetired==0);
            accepted = state.admit(intent, publication.hook());
        }
        assert(accepted == expected && publication.id == expected && publication.commits == commits + 1);
        assert(state.snapshot().localAdmitted==expected && state.retained().back().localOrdinal==expected);
        assert(*std::static_pointer_cast<const uint64_t>(publication.visible) == expected);
    }
    assert(rolledBack);
    LayerProgress full(key(), 10, value(), 1, 100);
    full.admit(intent);
    auto commits = publication.commits;
    error(OMNI_USD_LIVE_RESOURCE_EXHAUSTED, [&] { full.admit(intent, publication.hook()); });
    full.fail(OMNI_USD_LIVE_CANCELLED);
    error(OMNI_USD_LIVE_CANCELLED, [&] { full.admit(intent, publication.hook()); });
    assert(publication.commits == commits);
}

struct PausedPublication {
    std::atomic<bool> entered{false}, release{false};
    uint64_t published = 0;
    bool forbidLaterAllocations = false;
    static void commit(void* context, uint64_t id) noexcept
    {
        auto& state = *static_cast<PausedPublication*>(context);
        // Test-only barrier exposes the locked publication interval. Actual
        // journal callbacks only assign IDs/swap preallocated pointers.
        state.entered.store(true, std::memory_order_release);
        while (!state.release.load(std::memory_order_acquire)) std::this_thread::yield();
        state.published = id;
        if (state.forbidLaterAllocations) allocationCountdown = 0;
    }
};
void publicationPrecedesIssuance()
{
    LayerProgress state(key(), 10, value(), 8, 100); open(state, 10);
    PausedPublication publication;
    uint64_t admitted = 0;
    std::thread author([&] { admitted = state.admit(value(), {PausedPublication::commit, &publication}); });
    while (!publication.entered.load(std::memory_order_acquire)) std::this_thread::yield();
    std::atomic<bool> attempted{false}, taken{false};
    std::thread writer([&] {
        attempted.store(true, std::memory_order_release);
        const auto work = state.takeWrite();
        assert(work.id && publication.published == work.id);
        taken.store(true, std::memory_order_release);
    });
    while (!attempted.load(std::memory_order_acquire)) std::this_thread::yield();
    assert(!taken.load(std::memory_order_acquire));
    publication.release.store(true, std::memory_order_release);
    author.join(); writer.join();
    assert(admitted == 1 && publication.published == admitted && taken);
}

void actorPublicationAndClose()
{
    assert(omniUsdLiveInitialize(nullptr) == OMNI_USD_LIVE_OK);
    auto runtime = getRuntime();
    OmniUsdLiveEndpointOptions options{};
    options.struct_size = sizeof(options); options.abi_version = OMNI_USD_LIVE_ABI_VERSION;
    options.authority = "localhost";
    OmniUsdLiveEndpoint endpoint = 0;
    assert(omniUsdLiveRegisterEndpoint(&options, &endpoint) == OMNI_USD_LIVE_OK);
    LayerActor::Options layer; layer.session = "publication"; layer.layer = "root.live"; layer.object = 1;
    LayerActor::Callbacks callbacks;
    callbacks.decode = [](const LayerReadUpdate&) { assert(false); return LayerActor::Decoded{}; };
    callbacks.apply = [](const auto&, const auto&) { assert(false); };
    // No token is supplied: the actor stays in auth wait, without network I/O.
    auto actor = LayerActor::open(runtime, endpoint, std::move(layer), std::move(callbacks));
    PausedPublication publication;
    publication.forbidLaterAllocations = true;
    uint64_t admitted = 0;
    std::thread author([&] {
        admitted = actor->submit("edit", value(), {PausedPublication::commit, &publication});
        allocationCountdown = std::numeric_limits<size_t>::max();
    });
    while (!publication.entered.load(std::memory_order_acquire)) std::this_thread::yield();
    std::atomic<bool> attempted{false}, closed{false};
    std::thread closer([&] {
        attempted.store(true, std::memory_order_release);
        const auto result = actor->close(0);
        assert(result == OMNI_USD_LIVE_OK || result == OMNI_USD_LIVE_TIMEOUT);
        assert(publication.published != 0);
        closed.store(true, std::memory_order_release);
    });
    while (!attempted.load(std::memory_order_acquire)) std::this_thread::yield();
    assert(!closed.load(std::memory_order_acquire));
    publication.release.store(true, std::memory_order_release);
    author.join(); closer.join();
    assert(admitted == publication.published && admitted == 1);
    PublicationState rejected;
    error(OMNI_USD_LIVE_CANCELLED, [&] { actor->submit("later", value(), rejected.hook()); });
    assert(!rejected.commits);
    assert(actor->close(2000) == OMNI_USD_LIVE_OK);
    assert(actor->retained().size() == 1 && actor->retained()[0].id == admitted);
    actor.reset();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    for (;;) {
        const auto result = omniUsdLiveUnregisterEndpoint(endpoint, 0);
        if (result == OMNI_USD_LIVE_OK) break;
        assert(result == OMNI_USD_LIVE_BUSY && std::chrono::steady_clock::now() < deadline);
        std::this_thread::yield();
    }
    assert(omniUsdLiveShutdown(2000) == OMNI_USD_LIVE_OK);
    runtime.reset();
}

void peerHoleAndRetention()
{
    LayerProgress state(key(), 10, value(), 16, 100);
    open(state, 10);
    auto intent = value(120);
    std::weak_ptr<const void> retainedIntent = intent;
    const auto local = state.admit(intent); intent.reset();
    auto issued = state.takeWrite(); assert(issued.id == local); issued = {};
    state.completeWrite(local, OMNI_USD_LIVE_OK, 12);
    assert(state.snapshot().covered == 10 && state.snapshot().applied == 10);
    assert(!retainedIntent.expired());
    const auto unsent = state.admit(value(130));
    const auto generation = state.disconnect();
    assert(!state.takeWrite().id);
    auto resumed = state.startRead(); assert(resumed.sequence == 10);
    assert(!state.caughtUp(generation - 1, 1, 12).accepted);
    auto peer = value(110); std::weak_ptr<const void> retainedPeer = peer;
    const auto remote = state.acceptCoverage(generation, 1, 11, 11, proof(peer)); peer.reset();
    assert(state.snapshot().covered == 12 && state.snapshot().applied == 10);
    const auto ready = state.recoveryReady(generation, 1, 12);
    assert(ready.accepted && !ready.initialLatest);
    assert(state.takeWrite().id == unsent);
    state.deferWrite(unsent);
    // An applied later effect cannot free the intent while an earlier effect
    // still has to enter the captured layer state.
    state.applied(local);
    assert(state.snapshot().applied == 10 && !retainedIntent.expired());
    state.applied(remote);
    assert(state.snapshot().applied == 12);
    assert(retainedIntent.expired() && retainedPeer.expired());
    assert(state.snapshot().retained == 1);
}

void gateIsolationAndUnknown()
{
    LayerProgress a(key(), 10, value(), 256, 100), b(key(2), 10, value(), 256, 100);
    open(a, 10); open(b, 10);
    const auto first = a.admit(value());
    assert(a.takeWrite().id == first);
    a.disconnect();
    // Continuous authoring adds unsent work, not in-flight work. It cannot
    // prevent the reader from starting after the single issued call settles.
    for (unsigned i = 0; i < 100; ++i) a.admit(value(i));
    error(OMNI_USD_LIVE_BUSY, [&] { a.startRead(); });
    auto other = b.admit(value()); assert(b.takeWrite().id == other);
    b.completeWrite(other, OMNI_USD_LIVE_OK, 11);
    assert(b.snapshot().covered == 11);
    a.completeWrite(first, OMNI_USD_LIVE_OK, 11);
    auto read = a.startRead(); assert(read.sequence == 11);
    assert(!a.takeWrite().id);
    auto ready = a.caughtUp(read.generation, 1, 11);
    assert(ready.accepted && !ready.initialLatest);
    auto next = a.takeWrite(); assert(next.id == first + 1);
    a.completeWrite(next.id, OMNI_USD_LIVE_OUTCOME_UNKNOWN);
    const auto status = a.snapshot();
    assert(status.gate == Gate::Failed && status.result == OMNI_USD_LIVE_OUTCOME_UNKNOWN &&
           status.retained == 101 && status.queued == 99);
    assert(!a.takeWrite().id);
    error(OMNI_USD_LIVE_OUTCOME_UNKNOWN, [&] { a.startRead(); });
    error(OMNI_USD_LIVE_OUTCOME_UNKNOWN, [&] { a.admit(value()); });
    auto work = a.retained();
    assert(work[0].disposition == State::Acknowledged && work[1].disposition == State::Unknown);
    for (const auto& entry : work) assert(entry.owner);
    a.fail(OMNI_USD_LIVE_UNAVAILABLE);
    assert(a.snapshot().gate == Gate::Failed && a.snapshot().result == OMNI_USD_LIVE_OUTCOME_UNKNOWN);
    // A later uncertain write outcome replaces an earlier terminal result.
    LayerProgress c(key(3), 10, value(), 16, 1000);
    open(c, 10);
    const auto pending = c.admit(value(1)); assert(c.takeWrite().id == pending);
    c.fail(OMNI_USD_LIVE_MALFORMED_STREAM);
    assert(c.snapshot().gate == Gate::Failed && c.snapshot().result == OMNI_USD_LIVE_MALFORMED_STREAM);
    c.completeWrite(pending, OMNI_USD_LIVE_OUTCOME_UNKNOWN);
    assert(c.snapshot().gate == Gate::Failed && c.snapshot().result == OMNI_USD_LIVE_OUTCOME_UNKNOWN);
}

void malformedAndBounds()
{
    // A hole wider than the synthetic gap bound fails like an oversized aggregate.
    LayerProgress gap(key(), 10, value(), 8, 4);
    auto read = gap.startRead();
    error(OMNI_USD_LIVE_RESOURCE_EXHAUSTED, [&] { gap.acceptCoverage(read.generation, 1, 16, 16, proof(value())); });
    assert(gap.snapshot().covered == 10);
    LayerProgress replaced(key(), 10, value(), 8, 4);
    auto old = replaced.startRead(); replaced.disconnect(); auto fresh = replaced.startRead();
    assert(!replaced.acceptCoverage(old.generation, 2, 11, 11, proof(value())));
    error(OMNI_USD_LIVE_OBJECT_REPLACED, [&] { replaced.acceptCoverage(fresh.generation, 2, 11, 11, proof(value())); });
    LayerProgress bound(key(), 10, value(), 1, 4);
    open(bound, 10);
    auto id = bound.admit(value());
    error(OMNI_USD_LIVE_RESOURCE_EXHAUSTED, [&] { bound.admit(value()); });
    assert(bound.snapshot().retained == 1 && bound.takeWrite().id == id);
    bound.completeWrite(id, OMNI_USD_LIVE_OK, 11); bound.applied(id);
    assert(bound.admit(value()) > id);
    LayerProgress aggregate(key(), 10, value(), 8, 4);
    read = aggregate.startRead();
    auto complete = aggregate.acceptCoverage(read.generation, 1, 11, 15, proof(value(), LayerProgress::CoverageKind::Aggregate));
    assert(aggregate.snapshot().covered == 15 && aggregate.snapshot().applied == 10);
    auto ready = aggregate.recoveryReady(read.generation, 1, 15);
    assert(ready.accepted && ready.initialLatest);
    assert(!aggregate.caughtUp(read.generation, 1, 15).initialLatest);
    aggregate.applied(complete); assert(aggregate.snapshot().applied == 15);
    error(OMNI_USD_LIVE_RESOURCE_EXHAUSTED, [&] {
        aggregate.acceptCoverage(read.generation, 1, 16, std::numeric_limits<uint64_t>::max(), proof(value(), LayerProgress::CoverageKind::Aggregate));
    });
    assert(aggregate.snapshot().covered == 15);
    LayerProgress maximum(key(), std::numeric_limits<uint64_t>::max() - 1, value(), 2, 0);
    read = maximum.startRead();
    id = maximum.acceptCoverage(read.generation, 1, std::numeric_limits<uint64_t>::max(),
                                std::numeric_limits<uint64_t>::max(), proof(value()));
    maximum.applied(id);
    assert(maximum.snapshot().covered == std::numeric_limits<uint64_t>::max());
}

// gRPC delivers a read in order, so a position the read skips is an empty commit, except one an issued write may
// hold: the read never carries this writer's own commits.
void emptyPositions()
{
    LayerProgress update(key(), 10, value(), 8, 100);
    auto read = update.startRead();
    const auto next = update.acceptCoverage(read.generation, 1, 12, 12, proof(value()));
    assert(update.snapshot().covered == 12 && update.snapshot().applied == 11 && update.snapshot().retained == 1);
    assert(update.caughtUp(read.generation, 1, 12).accepted);
    update.applied(next); assert(update.snapshot().applied == 12 && update.snapshot().retained == 0);
    // An empty header above coverage, a caught-up header or an empty commit, covers up to its position.
    LayerProgress header(key(), 10, value(), 8, 100);
    read = header.startRead();
    const auto ready = header.caughtUp(read.generation, 1, 13);
    assert(ready.accepted && ready.initialLatest);
    assert(header.snapshot().covered == 13 && header.snapshot().applied == 13);
    assert(header.caughtUp(read.generation, 1, 15).accepted && header.snapshot().covered == 15);
    header.disconnect(); assert(header.startRead().sequence == 15);
    // A hole below an issued write's acknowledgment may be that write; it fills once the write settles.
    LayerProgress issued(key(), 10, value(), 8, 100);
    open(issued, 10);
    auto own = issued.admit(value(2)); assert(issued.takeWrite().id == own);
    const auto peer = issued.acceptCoverage(1, 1, 12, 12, proof(value(3)));
    assert(issued.snapshot().covered == 10 && issued.snapshot().applied == 10);
    issued.completeWrite(own, OMNI_USD_LIVE_OK, 11);
    assert(issued.snapshot().result == OMNI_USD_LIVE_OK && issued.snapshot().covered == 12);
    issued.applied(own); issued.applied(peer); assert(issued.snapshot().applied == 12);
    LayerProgress later(key(), 10, value(), 8, 100);
    open(later, 10);
    own = later.admit(value(2)); assert(later.takeWrite().id == own);
    later.acceptCoverage(1, 1, 12, 12, proof(value(3)));
    later.completeWrite(own, OMNI_USD_LIVE_OK, 13);
    assert(later.snapshot().result == OMNI_USD_LIVE_OK && later.snapshot().covered == 13 && later.snapshot().applied == 11);
}

// The ordered run of unapplied work starts at the applied frontier and follows each work across the empty
// commits after it; it stops at the first position no work holds and at the requested count.
void applicableRun()
{
    LayerProgress state(key(), 10, value(), 16, 100);
    open(state, 10);
    assert(state.applicable(8).empty());
    const auto first = state.acceptCoverage(1, 1, 11, 11, proof(value(11)));
    const auto second = state.acceptCoverage(1, 1, 13, 13, proof(value(13))); // 12 is an empty commit.
    const auto own = state.admit(value(2)); assert(state.takeWrite().id == own);
    state.completeWrite(own, OMNI_USD_LIVE_OK, 14);
    const auto third = state.acceptCoverage(1, 1, 15, 15, proof(value(15)));
    auto run = state.applicable(8);
    assert(run.size() == 4 && run[0].id == first && run[1].id == second && run[2].id == own && run[3].id == third);
    assert(run[2].disposition == State::Acknowledged && run[1].first == 13);
    run = state.applicable(2);
    assert(run.size() == 2 && run[1].id == second);
    assert(state.applicable(0).empty());
    // Listing changes nothing; application advances the run's start.
    assert(state.snapshot().applied == 10 && state.snapshot().retained == 4);
    state.applied(first);
    run = state.applicable(8);
    assert(state.snapshot().applied == 12 && run.size() == 3 && run[0].id == second);
}

void snapshotBootstrap()
{
    LayerProgress state(key(), 0, {}, 8, 4);
    const auto read = state.startRead();
    assert(read.sequence == 0);
    const auto queued = state.admit(value(999999));
    const auto initial = state.acceptCoverage(read.generation, 1, 1, 1000000,
        proof(value(1000000), LayerProgress::CoverageKind::Snapshot));
    assert(initial && state.snapshot().covered == 1000000 && state.snapshot().applied == 0);
    assert(state.retained().back().kind == LayerProgress::CoverageKind::Snapshot);
    assert(!state.takeWrite().id);
    const auto ready = state.recoveryReady(read.generation, 1, 1000000);
    assert(ready.accepted && ready.initialLatest);
    assert(state.takeWrite().id == queued);
    state.completeWrite(queued, OMNI_USD_LIVE_OK, 1000001);
    state.applied(queued); assert(state.snapshot().applied == 0);
    state.applied(initial); assert(state.snapshot().applied == 1000001 && state.snapshot().retained == 0);
    state.disconnect(); auto resumed = state.startRead(); assert(resumed.sequence == 1000001);
    error(OMNI_USD_LIVE_MALFORMED_STREAM, [&] {
        state.acceptCoverage(resumed.generation, 1, 1000002, 1000003,
            proof(value(1000003), LayerProgress::CoverageKind::Snapshot));
    });
}

void acknowledgementOrder()
{
    LayerProgress state(key(), 10, value(), 8, 100);
    open(state, 10);
    auto id = state.admit(value()); assert(state.takeWrite().id == id);
    state.completeWrite(id, OMNI_USD_LIVE_OK, 12);
    id = state.admit(value()); assert(state.takeWrite().id == id);
    state.completeWrite(id, OMNI_USD_LIVE_OK, 11);
    assert(state.snapshot().gate == Gate::Failed && state.snapshot().result == OMNI_USD_LIVE_OUTCOME_UNKNOWN &&
           state.snapshot().covered == 10);
    LayerProgress raced(key(), 10, value(), 8, 100);
    open(raced, 10);
    id = raced.admit(value()); assert(raced.takeWrite().id == id);
    // Peer13 can arrive before the earlier self12 acknowledgment. Only the
    // high-water captured BEFORE issuance is a lower bound for that ACK.
    raced.acceptCoverage(1, 1, 13, 13, proof(value()));
    raced.completeWrite(id, OMNI_USD_LIVE_OK, 12);
    // The read delivered 13 in order, so 11, which no write holds, is an empty commit.
    assert(raced.snapshot().gate == Gate::Ready && raced.snapshot().covered == 13);
    raced.disconnect(); auto read = raced.startRead(); assert(read.sequence == 13);
    // A late unknown write must strengthen an already latched read failure.
    LayerProgress late(key(), 10, value(), 8, 100); open(late, 10);
    id = late.admit(value()); assert(late.takeWrite().id == id);
    late.fail(OMNI_USD_LIVE_PERMISSION_DENIED);
    late.completeWrite(id, OMNI_USD_LIVE_OUTCOME_UNKNOWN);
    assert(late.snapshot().gate == Gate::Failed && late.snapshot().result == OMNI_USD_LIVE_OUTCOME_UNKNOWN);
}

void concurrentAdmissionAndReentrantRetirement()
{
    LayerProgress state(key(), 10, value(), 32, 100);
    open(state, 10);
    std::atomic<unsigned> waiting{0}, accepted{0}, refused{0};
    std::atomic<bool> go{false};
    std::vector<std::thread> writers;
    std::array<std::vector<std::pair<uint64_t, uint64_t>>, 8> admissions;
    for (unsigned i = 0; i < 8; ++i) writers.emplace_back([&, i] {
        ++waiting;
        while (!go.load()) std::this_thread::yield();
        for (unsigned j = 0; j < 8; ++j) {
            try {
                const auto datum = i * 8 + j;
                auto id = state.admit(value(datum));
                admissions[i].emplace_back(id, datum); ++accepted;
            }
            catch (OmniUsdLiveResult result) { assert(result == OMNI_USD_LIVE_RESOURCE_EXHAUSTED); ++refused; }
        }
    });
    while (waiting.load() != 8) std::this_thread::yield();
    go = true;
    for (auto& writer : writers) writer.join();
    assert(accepted == 32 && refused == 32);
    auto retained = state.retained();
    std::map<uint64_t, uint64_t> expected;
    for (const auto& writer : admissions)
        for (const auto& admission : writer) assert(expected.emplace(admission).second);
    std::map<uint64_t, uint64_t> actual;
    for (const auto& work : retained)
        actual.emplace(work.id, *std::static_pointer_cast<const uint64_t>(work.owner));
    assert(actual == expected && expected.size() == 32);
    for (unsigned i = 0; i < 32; ++i) {
        auto work = state.takeWrite(); assert(work.id == i + 1);
        assert(*std::static_pointer_cast<const uint64_t>(work.owner) == expected.at(work.id));
        assert(!state.takeWrite().id);
        state.completeWrite(work.id, OMNI_USD_LIVE_OK, 11 + i);
        state.applied(work.id);
    }
    assert(state.snapshot().retained == 0 && state.snapshot().applied == 42);
    // Destruction may legitimately query progress. It must run outside locks.
    std::atomic<bool> destroyed{false};
    std::shared_ptr<const void> owner(new int(1), [&](const void* ptr) {
        assert(state.snapshot().retained == 0);
        delete static_cast<const int*>(ptr); destroyed = true;
    });
    auto id = state.admit(owner); owner.reset();
    assert(state.takeWrite().id == id);
    state.completeWrite(id, OMNI_USD_LIVE_OK, 43); state.applied(id);
    assert(destroyed);
}

void singleWriteAndOverlap()
{
    LayerProgress state(key(), 10, value(), 16, 100); open(state, 10);
    auto first = state.admit(value(1)), second = state.admit(value(2));
    assert(state.takeWrite().id == first && !state.takeWrite().id);
    std::vector<std::thread> contenders;
    for (unsigned i = 0; i < 8; ++i) contenders.emplace_back([&] { assert(!state.takeWrite().id); });
    for (auto& contender : contenders) contender.join();
    state.deferWrite(first);
    assert(state.takeWrite().id == first && !state.takeWrite().id);
    state.completeWrite(first, OMNI_USD_LIVE_OK, 11);
    assert(state.takeWrite().id == second);
    state.deferWrite(second);
    // With no write issued, a position the read skipped is an empty commit, so the read resumes above it.
    const auto remote = state.acceptCoverage(1, 1, 13, 13, proof(value(13)));
    assert(state.snapshot().covered == 13);
    state.disconnect(); auto read = state.startRead(); assert(read.sequence == 13);
    state.applied(remote); assert(state.snapshot().applied == 10);
    state.applied(first); assert(state.snapshot().applied == 13);
    auto later = state.acceptCoverage(read.generation, 1, 15, 15, proof(value(15)));
    state.disconnect(); read = state.startRead(); assert(read.sequence == 15);
    state.applied(later); assert(state.snapshot().applied == 15);
    assert(state.snapshot().retained == 1); // Only the queued second intent.
    LayerProgress conflict(key(), 10, value(), 16, 100); open(conflict, 10);
    conflict.acceptCoverage(1, 1, 12, 12, proof(value(12)));
    conflict.disconnect(); read = conflict.startRead();
    error(OMNI_USD_LIVE_MALFORMED_STREAM, [&] {
        conflict.acceptCoverage(read.generation, 1, 12, 12, proof(value(999)));
    });
    assert(conflict.snapshot().retained == 1);
    // An aggregate that starts on an acknowledged position overlaps it.
    LayerProgress aggregate(key(), 10, value(), 16, 100); open(aggregate, 10);
    const auto own = aggregate.admit(value(11)); assert(aggregate.takeWrite().id == own);
    aggregate.completeWrite(own, OMNI_USD_LIVE_OK, 12);
    aggregate.disconnect(); read = aggregate.startRead(); assert(read.sequence == 10);
    error(OMNI_USD_LIVE_UNSUPPORTED, [&] {
        aggregate.acceptCoverage(read.generation, 1, 12, 14, proof(value(1214), LayerProgress::CoverageKind::Aggregate));
    });
    assert(aggregate.snapshot().gate == Gate::Failed && aggregate.snapshot().result == OMNI_USD_LIVE_UNSUPPORTED &&
           aggregate.snapshot().retained == 1);
}

void aggregateAbsorbsAcknowledgedIntents()
{
    // Coverage 10, an unseen peer write 11 and an acknowledged self-write 12
    // (design section 12): the resume aggregate 11..12 retires the self-write.
    LayerProgress state(key(), 10, value(), 16, 100); open(state, 10);
    auto intent = value(120); std::weak_ptr<const void> retainedIntent = intent;
    const auto local = state.admit(intent); intent.reset();
    assert(state.takeWrite().id == local);
    state.completeWrite(local, OMNI_USD_LIVE_OK, 12);
    const auto generation = state.disconnect(); auto read = state.startRead(); assert(read.sequence == 10);
    const auto aggregate = state.acceptCoverage(generation, 1, 11, 12,
        proof(value(1112), LayerProgress::CoverageKind::Aggregate));
    assert(aggregate && state.snapshot().covered == 12 && state.snapshot().gate != Gate::Failed);
    for (const auto& work : state.retained()) {
        if (work.id == local) assert(work.absorbedBy == aggregate && !work.applied);
        if (work.id == aggregate) assert(work.absorbedLocal == 1 && work.absorbedThrough == 1);
    }
    auto owners = state.applied(aggregate);
    const auto after = state.snapshot();
    assert(owners.size() == 1 && after.applied == 12 && after.retained == 0 && after.localRetired == 1 &&
           after.pendingWrites == 0 && after.pendingBytes == 0);
    owners.clear(); assert(retainedIntent.expired());
    // The re-sent frame can arrive while the acknowledgment's own application
    // is in flight, after the peer before it applied: still a replay.
    LayerProgress racing(key(5), 10, value(), 16, 100); open(racing, 10);
    const auto sent = proof(value(12)).fingerprint;
    auto raced = value(12); std::weak_ptr<const void> racedIntent = raced;
    const auto flight = racing.admit(raced, {nullptr, nullptr}, 0, &sent); raced.reset();
    assert(racing.takeWrite().id == flight); racing.completeWrite(flight, OMNI_USD_LIVE_OK, 12);
    auto again = racing.disconnect(); read = racing.startRead();
    racing.applied(racing.acceptCoverage(again, 1, 11, 11, proof(value(11))));
    const auto replayed = racing.acceptCoverage(again, 1, 12, 12, proof(value(12)));
    auto released = racing.applied(flight);
    const auto raceState = racing.snapshot();
    assert(raceState.retained == 0 && raceState.applied == 12 && raceState.localRetired == 1 &&
           raceState.pendingWrites == 0);
    assert(!replayed && released.empty() && racedIntent.expired());
    // A per-commit catch-up re-sends the layer's own write: equal bytes are a
    // replay, and the acknowledgment retires through its own application.
    LayerProgress echo(key(2), 10, value(), 16, 100); open(echo, 10);
    auto echoed = value(12); std::weak_ptr<const void> echoedIntent = echoed;
    const auto own = echo.admit(echoed, {nullptr, nullptr}, 0, &sent); echoed.reset();
    assert(echo.takeWrite().id == own); echo.completeWrite(own, OMNI_USD_LIVE_OK, 12);
    again = echo.disconnect(); read = echo.startRead();
    const auto peer = echo.acceptCoverage(again, 1, 11, 11, proof(value(11)));
    assert(!echo.acceptCoverage(again, 1, 12, 12, proof(value(12))) && echo.snapshot().gate != Gate::Failed);
    echo.applied(peer); assert(echo.applied(own).empty());
    assert(echo.snapshot().retained == 0 && echo.snapshot().localRetired == 1 && echoedIntent.expired());
    // Different bytes at the acknowledged position prove nothing: stop, keep work.
    LayerProgress differ(key(3), 10, value(), 16, 100); open(differ, 10);
    const auto mine = differ.admit(value(12), {nullptr, nullptr}, 0, &sent);
    assert(differ.takeWrite().id == mine); differ.completeWrite(mine, OMNI_USD_LIVE_OK, 12);
    again = differ.disconnect(); read = differ.startRead();
    differ.acceptCoverage(again, 1, 11, 11, proof(value(11)));
    error(OMNI_USD_LIVE_UNSUPPORTED, [&] { differ.acceptCoverage(again, 1, 12, 12, proof(value(999))); });
    assert(differ.snapshot().gate == Gate::Failed && differ.snapshot().retained == 2);
    // An unfingerprinted acknowledgment is never retired by a singleton.
    LayerProgress bare(key(4), 10, value(), 16, 100); open(bare, 10);
    const auto plain = bare.admit(value(12));
    assert(bare.takeWrite().id == plain); bare.completeWrite(plain, OMNI_USD_LIVE_OK, 12);
    again = bare.disconnect(); read = bare.startRead();
    bare.acceptCoverage(again, 1, 11, 11, proof(value(11)));
    error(OMNI_USD_LIVE_UNSUPPORTED, [&] { bare.acceptCoverage(again, 1, 12, 12, proof(value(12))); });
}

void aggregateStartingOnAcknowledgmentStops()
{
    // Coverage 10, own write acknowledged at 12, peer 11 accepted: an
    // aggregate 12..13 starts on the acknowledged position, so absorbing the
    // acknowledgment would leave the aggregate behind the acknowledgment's own
    // application. It stops UNSUPPORTED with every accepted record retained.
    for (const bool reconnect : {false, true}) {
        LayerProgress state(key(6), 10, value(), 16, 100); open(state, 10);
        const auto own = state.admit(value(12));
        assert(state.takeWrite().id == own); state.completeWrite(own, OMNI_USD_LIVE_OK, 12);
        auto generation = state.snapshot().generation;
        if (reconnect) {
            generation = state.disconnect();
            assert(state.startRead().sequence == 10);
        }
        const auto peer = state.acceptCoverage(generation, 1, 11, 11, proof(value(11)));
        assert(peer);
        error(OMNI_USD_LIVE_UNSUPPORTED, [&] {
            state.acceptCoverage(generation, 1, 12, 13, proof(value(1213), LayerProgress::CoverageKind::Aggregate));
        });
        auto stopped = state.snapshot();
        assert(stopped.gate == Gate::Failed && stopped.result == OMNI_USD_LIVE_UNSUPPORTED && stopped.retained == 2 &&
               stopped.pendingWrites == 1 && stopped.localRetired == 0);
        for (const auto& work : state.retained())
            if (work.id == own) assert(!work.absorbedBy && !work.applied && work.first == 12);
        // Accepted coverage still applies in order; nothing applies past 11.
        assert(state.applied(peer).empty());
        stopped = state.snapshot();
        assert(stopped.applied == 11 && stopped.retained == 1 && stopped.pendingWrites == 1);
    }
    // An acknowledgment already absorbed by an earlier aggregate is not
    // absorbed again: a later aggregate over it stops, and both stay retained.
    LayerProgress absorbed(key(7), 10, value(), 16, 100); open(absorbed, 10);
    const auto ack = absorbed.admit(value(12));
    assert(absorbed.takeWrite().id == ack); absorbed.completeWrite(ack, OMNI_USD_LIVE_OK, 12);
    auto generation = absorbed.snapshot().generation;
    const auto first = absorbed.acceptCoverage(generation, 1, 11, 12,
        proof(value(1112), LayerProgress::CoverageKind::Aggregate));
    assert(first);
    error(OMNI_USD_LIVE_UNSUPPORTED, [&] {
        absorbed.acceptCoverage(generation, 1, 12, 13, proof(value(1213), LayerProgress::CoverageKind::Aggregate));
    });
    assert(absorbed.snapshot().gate == Gate::Failed && absorbed.snapshot().retained == 2);
    for (const auto& work : absorbed.retained())
        if (work.id == ack) assert(work.absorbedBy == first && !work.applied);
    // An acknowledgment whose application already ran is never absorbed: while
    // a coverage hole keeps it retained, an aggregate over it stops.
    LayerProgress early(key(8), 10, value(), 16, 100); open(early, 10);
    const auto ran = early.admit(value(12));
    assert(early.takeWrite().id == ran); early.completeWrite(ran, OMNI_USD_LIVE_OK, 12);
    assert(early.applied(ran).empty() && early.snapshot().applied == 10 && early.snapshot().retained == 1);
    generation = early.snapshot().generation;
    error(OMNI_USD_LIVE_UNSUPPORTED, [&] {
        early.acceptCoverage(generation, 1, 11, 13, proof(value(1113), LayerProgress::CoverageKind::Aggregate));
    });
    assert(early.snapshot().gate == Gate::Failed && early.snapshot().retained == 1);
    // Once the acknowledgment has applied and retired, coverage ending at or
    // below the applied frontier is a replay, and coverage starting at or below
    // it with a later end straddles the frontier and stops.
    LayerProgress retired(key(9), 10, value(), 16, 100); open(retired, 10);
    const auto done = retired.admit(value(12));
    assert(retired.takeWrite().id == done); retired.completeWrite(done, OMNI_USD_LIVE_OK, 12);
    generation = retired.snapshot().generation;
    retired.applied(retired.acceptCoverage(generation, 1, 11, 11, proof(value(11))));
    retired.applied(done);
    assert(retired.snapshot().applied == 12 && retired.snapshot().retained == 0);
    assert(!retired.acceptCoverage(generation, 1, 11, 12, proof(value(1112), LayerProgress::CoverageKind::Aggregate)));
    assert(retired.snapshot().gate != Gate::Failed);
    error(OMNI_USD_LIVE_UNSUPPORTED, [&] {
        retired.acceptCoverage(generation, 1, 12, 13, proof(value(1213), LayerProgress::CoverageKind::Aggregate));
    });
    assert(retired.snapshot().gate == Gate::Failed && retired.snapshot().applied == 12);
}

// A coalescing service can deliver an aggregate containing an own write's
// commit before that write's acknowledgment: the acknowledgment strictly
// inside it is absorbed, as in the opposite arrival order.
void aggregateBeforeAcknowledgment()
{
    // Coverage 10, an in-flight write (floor 10), peer 11 and the write at 12.
    for (const bool okFirst : {false, true}) {
        LayerProgress state(key(10), 10, value(), 16, 100); open(state, 10);
        auto intent = value(120); std::weak_ptr<const void> retainedIntent = intent;
        const auto local = state.admit(intent); intent.reset();
        assert(state.takeWrite().id == local);
        const auto generation = state.snapshot().generation;
        uint64_t aggregate = 0;
        if (okFirst) state.completeWrite(local, OMNI_USD_LIVE_OK, 12);
        aggregate = state.acceptCoverage(generation, 1, 11, 12, proof(value(1112), LayerProgress::CoverageKind::Aggregate));
        assert(aggregate);
        if (!okFirst) {
            // The aggregate may contain the write's commit: it waits for the acknowledgment.
            assert(state.applicable(4).empty());
            state.completeWrite(local, OMNI_USD_LIVE_OK, 12);
        }
        auto settled = state.snapshot();
        assert(settled.gate == Gate::Ready && settled.result == OMNI_USD_LIVE_OK && settled.covered == 12 &&
               settled.issued == 0);
        for (const auto& work : state.retained()) {
            if (work.id == local)
                assert(work.disposition == State::Acknowledged && work.first == 12 && work.absorbedBy == aggregate &&
                       !work.applied);
            if (work.id == aggregate) assert(work.absorbedLocal == 1 && work.absorbedThrough == 1);
        }
        const auto run = state.applicable(4);
        assert(run.size() == 1 && run.front().id == aggregate);
        auto owners = state.applied(aggregate);
        settled = state.snapshot();
        assert(owners.size() == 1 && settled.applied == 12 && settled.retained == 0 && settled.localRetired == 1 &&
               settled.pendingWrites == 0);
        owners.clear(); assert(retainedIntent.expired());
    }
    // Two writes in flight, acknowledged in reverse order inside one aggregate:
    // both are absorbed and retire, in local order, with its application.
    LayerProgress window(key(11), 10, value(), 16, 100); open(window, 10);
    const auto a = window.admit(value(1)), b = window.admit(value(2));
    assert(window.takeWrite(2).id == a && window.takeWrite(2).id == b);
    auto generation = window.snapshot().generation;
    const auto both = window.acceptCoverage(generation, 1, 11, 13,
        proof(value(1113), LayerProgress::CoverageKind::Aggregate));
    window.completeWrite(b, OMNI_USD_LIVE_OK, 13);
    assert(window.applicable(4).empty()); // a may still be inside it.
    window.completeWrite(a, OMNI_USD_LIVE_OK, 12);
    assert(window.snapshot().gate == Gate::Ready && window.snapshot().covered == 13);
    for (const auto& work : window.retained())
        if (work.id == both) assert(work.absorbedLocal == 2 && work.absorbedThrough == 2);
        else assert(work.absorbedBy == both);
    assert(window.applicable(4).size() == 1 && window.applied(both).size() == 2);
    assert(window.snapshot().retained == 0 && window.snapshot().localRetired == 2 && window.snapshot().applied == 13);
    // An acknowledgment above the aggregate is the write's own work: the
    // aggregate applies once the write settles, and absorbs nothing.
    LayerProgress above(key(12), 10, value(), 16, 100); open(above, 10);
    const auto later = above.admit(value(13));
    assert(above.takeWrite().id == later);
    generation = above.snapshot().generation;
    const auto below = above.acceptCoverage(generation, 1, 11, 12,
        proof(value(1112), LayerProgress::CoverageKind::Aggregate));
    assert(above.applicable(4).empty());
    above.completeWrite(later, OMNI_USD_LIVE_OK, 13);
    auto run = above.applicable(4);
    assert(above.snapshot().gate == Gate::Ready && run.size() == 2 && run[0].id == below && run[1].id == later &&
           !run[0].absorbedLocal && !run[1].absorbedBy);
    assert(above.applied(below).empty() && above.applied(later).empty());
    assert(above.snapshot().retained == 0 && above.snapshot().localRetired == 1);
    // A write taken after the aggregate arrived cannot be inside it.
    LayerProgress after(key(13), 10, value(), 16, 100); open(after, 10);
    generation = after.snapshot().generation;
    const auto early = after.acceptCoverage(generation, 1, 11, 12,
        proof(value(1112), LayerProgress::CoverageKind::Aggregate));
    const auto fresh = after.admit(value(13));
    assert(after.takeWrite().id == fresh);
    run = after.applicable(4);
    assert(run.size() == 1 && run.front().id == early);
    // An OK at an aggregate's first position, which the opposite arrival order
    // stops as an overlap, or at a single update's position collides.
    for (const auto kind : {LayerProgress::CoverageKind::Aggregate, LayerProgress::CoverageKind::Update}) {
        LayerProgress collide(key(14), 10, value(), 16, 100); open(collide, 10);
        const auto own = collide.admit(value(12));
        assert(collide.takeWrite().id == own);
        generation = collide.snapshot().generation;
        const auto last = kind == LayerProgress::CoverageKind::Aggregate ? 13 : 12;
        assert(collide.acceptCoverage(generation, 1, 12, last, proof(value(1213), kind)));
        collide.completeWrite(own, OMNI_USD_LIVE_OK, 12);
        const auto failed = collide.snapshot();
        assert(failed.gate == Gate::Failed && failed.result == OMNI_USD_LIVE_OUTCOME_UNKNOWN && failed.retained == 2);
        for (const auto& work : collide.retained()) assert(!work.absorbedBy && !work.absorbedLocal);
    }
}

// Mixed acknowledgment order inside one window: A and C acknowledged before
// the aggregate that contains them, B after it. All three are absorbed and
// retire, in local order, with the aggregate's application.
void aggregateBetweenWindowedAcknowledgments()
{
    LayerProgress state(key(15), 10, value(), 16, 100); open(state, 10);
    const auto a = state.admit(value(1)), b = state.admit(value(2)), c = state.admit(value(3));
    assert(state.takeWrite(3).id == a && state.takeWrite(3).id == b && state.takeWrite(3).id == c);
    state.completeWrite(a, OMNI_USD_LIVE_OK, 12);
    state.completeWrite(c, OMNI_USD_LIVE_OK, 14);
    const auto aggregate = state.acceptCoverage(state.snapshot().generation, 1, 11, 14,
        proof(value(1114), LayerProgress::CoverageKind::Aggregate));
    assert(aggregate && state.snapshot().gate == Gate::Ready);
    assert(state.applicable(4).empty()); // b may still be inside it.
    state.completeWrite(b, OMNI_USD_LIVE_OK, 13);
    for (const auto& work : state.retained())
        if (work.id == aggregate) assert(work.absorbedLocal == 3 && work.absorbedThrough == 3);
        else assert(work.absorbedBy == aggregate);
    assert(state.applicable(4).size() == 1 && state.applied(aggregate).size() == 3);
    const auto settled = state.snapshot();
    assert(settled.gate == Gate::Ready && settled.retained == 0 && settled.localRetired == 3 && settled.applied == 14);
}

// A queued write whose attempt may have committed holds an aggregate back
// like an issued one: its re-send's acknowledgment inside the aggregate is
// absorbed. A singleton aggregate has no position after its first one and
// holds nothing back.
void aggregateHoldBounds()
{
    LayerProgress state(key(17), 10, value(), 16, 100); open(state, 10);
    const auto write = state.admit(value(12));
    assert(state.takeWrite().id == write);
    const auto aggregate = state.acceptCoverage(state.snapshot().generation, 1, 11, 12,
        proof(value(1112), LayerProgress::CoverageKind::Aggregate));
    assert(state.requeueWrite(write, true) && state.applicable(4).empty());
    assert(state.takeWrite().id == write && state.applicable(4).empty());
    state.completeWrite(write, OMNI_USD_LIVE_OK, 12);
    for (const auto& work : state.retained())
        if (work.id == write) assert(work.absorbedBy == aggregate);
    assert(state.applicable(4).size() == 1 && state.applied(aggregate).size() == 1);
    auto settled = state.snapshot();
    assert(settled.gate == Gate::Ready && settled.retained == 0 && settled.localRetired == 1 && settled.applied == 12);

    LayerProgress single(key(18), 10, value(), 16, 100); open(single, 10);
    const auto later = single.admit(value(12));
    assert(single.takeWrite().id == later);
    const auto peer = single.acceptCoverage(single.snapshot().generation, 1, 11, 11,
        proof(value(1111), LayerProgress::CoverageKind::Aggregate));
    const auto run = single.applicable(4);
    assert(run.size() == 1 && run.front().id == peer);
    single.applied(peer);
    single.completeWrite(later, OMNI_USD_LIVE_OK, 12);
    single.applied(later);
    settled = single.snapshot();
    assert(settled.gate == Gate::Ready && settled.retained == 0 && settled.localRetired == 1 && settled.applied == 12);
}

// Every interleaving of one to three windowed writes' replies with the read's
// frames: peers below a live aggregate as single updates, the aggregate,
// which starts on a peer's position, then the peers above it. The writes
// commit in order at the remaining positions; no interleaving is a real
// collision, so every one settles and retires everything, whether work
// applies after each event or only at the end. Exhaustive, so deterministic
// without a seed.
void aggregateInterleavings()
{
    size_t cases = 0, failures = 0;
    for (unsigned writes = 1; writes <= 3; ++writes)
        for (unsigned peers = 1; peers <= 3; ++peers) {
            const unsigned total = writes + peers;
            // Own positions: a subset of 11..10+total of size `writes`.
            for (unsigned mask = 0; mask < (1u << total); ++mask) {
                std::vector<uint64_t> own, peer;
                for (unsigned bit = 0; bit < total; ++bit)
                    (mask >> bit & 1 ? own : peer).push_back(11 + bit);
                if (own.size() != writes) continue;
                for (const auto start : peer)
                    for (uint64_t last = start; last <= 10 + total; ++last) {
                        std::vector<std::pair<uint64_t, uint64_t>> frames;
                        for (const auto position : peer)
                            if (position < start) frames.emplace_back(position, position);
                        const auto aggregateFrame = frames.size();
                        frames.emplace_back(start, last);
                        for (const auto position : peer)
                            if (position > last) frames.emplace_back(position, position);
                        // Events: 0..writes-1 are replies, `writes` is the next read frame.
                        std::vector<unsigned> events(writes + frames.size());
                        for (unsigned i = 0; i < events.size(); ++i) events[i] = std::min(i, writes);
                        do {
                            for (const bool eager : {false, true}) {
                                ++cases;
                                LayerProgress state(key(16), 10, value(), 64, 100); open(state, 10);
                                std::vector<uint64_t> ids;
                                for (unsigned i = 0; i < writes; ++i) ids.push_back(state.admit(value(i + 1)));
                                for (unsigned i = 0; i < writes; ++i) assert(state.takeWrite(writes).id == ids[i]);
                                const auto apply = [&] {
                                    for (auto run = state.applicable(64); !run.empty(); run = state.applicable(64))
                                        for (const auto& work : run) state.applied(work.id);
                                };
                                std::string trace;
                                bool failed = false;
                                size_t frame = 0;
                                try {
                                    for (const auto event : events) {
                                        if (event < writes) {
                                            trace += " ack" + std::to_string(event) + ":" + std::to_string(own[event]);
                                            state.completeWrite(ids[event], OMNI_USD_LIVE_OK, own[event]);
                                        } else {
                                            const auto kind = frame == aggregateFrame ?
                                                LayerProgress::CoverageKind::Aggregate :
                                                LayerProgress::CoverageKind::Update;
                                            const auto [first, through] = frames[frame++];
                                            trace += " rd" + std::to_string(first) + ":" + std::to_string(through);
                                            assert(state.acceptCoverage(state.snapshot().generation, 1, first, through,
                                                proof(value(1000 * first + through), kind)));
                                        }
                                        if (state.snapshot().gate != Gate::Ready) failed = true;
                                        if (eager) apply();
                                    }
                                    apply();
                                } catch (OmniUsdLiveResult) { failed = true; }
                                const auto settled = state.snapshot();
                                failed |= settled.gate != Gate::Ready || settled.applied != 10 + total ||
                                    settled.retained != 0 || settled.localRetired != writes ||
                                    settled.pendingWrites != 0;
                                if (failed && ++failures <= 8)
                                    std::cerr << "aggregate interleaving failed: gate=" << int(settled.gate)
                                              << " result=" << settled.result << " applied=" << settled.applied
                                              << " retained=" << settled.retained << " eager=" << eager << trace
                                              << std::endl;
                            }
                            // Replies are distinct events and frames identical ones, so the
                            // permutations of the multiset keep the frames in read order.
                        } while (std::next_permutation(events.begin(), events.end()));
                    }
            }
        }
    std::cout << "aggregate interleavings: " << cases << " cases, " << failures << " failures" << std::endl;
    assert(!failures);
}

void generatedCoverageHistories()
{
    const auto* input = std::getenv("LIVE_TEST_SEED");
    const uint64_t seed = input ? std::stoull(input) : 42;
    std::cout << "layer_progress_seed=" << seed << std::endl; // Before generated work.
    std::mt19937_64 random(seed);
    for (unsigned history = 0; history < 200; ++history) {
        LayerProgress state(key(), 1, value(), 256, 100);
        open(state, 1);
        std::vector<uint64_t> ids;
        std::map<uint64_t, uint64_t> sequenceById;
        std::deque<uint64_t> pendingPeers;
        std::set<uint64_t> independentlyRetained{1};
        uint64_t generation = 1;
        auto check = [&] {
            uint64_t expected = 1;
            while (independentlyRetained.count(expected + 1)) ++expected;
            assert(state.snapshot().covered == expected);
            return expected;
        };
        auto receivePeer = [&] {
            const auto sequence = pendingPeers.front(); pendingPeers.pop_front();
            auto id = state.acceptCoverage(generation, 1, sequence, sequence, proof(value(sequence)));
            ids.push_back(id); sequenceById.emplace(id, sequence);
            independentlyRetained.insert(sequence); check();
        };
        // Delay peer callbacks while later local acknowledgments arrive. The
        // oracle tracks actual retained positions, never the maximum ACK.
        for (uint64_t sequence = 2; sequence <= 65; ++sequence) {
            if (random() % 2) {
                auto id = state.admit(value(sequence)); assert(state.takeWrite().id == id);
                state.completeWrite(id, OMNI_USD_LIVE_OK, sequence);
                ids.push_back(id); sequenceById.emplace(id, sequence);
                independentlyRetained.insert(sequence); check();
            } else pendingPeers.push_back(sequence);
            if (!pendingPeers.empty() && random() % 3 == 0) receivePeer();
            if (random() % 5 == 0) {
                generation = state.disconnect(); auto read = state.startRead();
                assert(read.sequence == check());
                assert(!state.takeWrite().id);
                while (!pendingPeers.empty()) receivePeer();
                auto ready = state.caughtUp(generation, 1, sequence);
                assert(ready.accepted && !ready.initialLatest);
            }
        }
        while (!pendingPeers.empty()) receivePeer();
        auto retained = state.retained();
        assert(retained.size() == 64);
        std::set<uint64_t> observed;
        for (const auto& work : retained) {
            const auto datum = *std::static_pointer_cast<const uint64_t>(work.owner);
            assert(datum == work.first && datum == work.last);
            observed.insert(datum);
        }
        independentlyRetained.erase(1); assert(observed == independentlyRetained);
        // Fisher-Yates over the engine's own output, which the standard fixes; the library shuffle's draws are implementation-defined.
        for (size_t remaining = ids.size(); remaining > 1; --remaining)
            std::swap(ids[remaining - 1], ids[random() % remaining]);
        std::set<uint64_t> appliedPositions{1};
        for (auto id : ids) {
            state.applied(id); appliedPositions.insert(sequenceById.at(id));
            uint64_t expectedApplied = 1;
            while (appliedPositions.count(expectedApplied + 1)) ++expectedApplied;
            assert(state.snapshot().applied == expectedApplied);
            assert(state.snapshot().retained == 65 - expectedApplied);
        }
        assert(state.snapshot().applied == 65 && state.snapshot().retained == 0);
    }
}
}

void retainedStatusAccounting()
{
    LayerProgress state(key(), 1, value(), 10, 100);
    state.startRead(); state.caughtUp(1, 1, 1);
    const auto first = state.admit(value(), {}, 17);
    const auto second = state.admit(value(), {}, 29);
    assert(state.snapshot().pendingWrites == 2 && state.snapshot().pendingBytes == 46);
    assert(state.takeWrite().id == first);
    state.completeWrite(first, OMNI_USD_LIVE_OK, 3);
    assert(state.snapshot().pendingWrites == 2 && state.snapshot().pendingBytes == 46);
    const auto peer = state.acceptCoverage(1, 1, 2, 2, proof(value(2)));
    assert(state.snapshot().pendingWrites == 2 && state.snapshot().pendingBytes == 46);
    state.applied(first); // Out-of-order application retains ownership/counters.
    assert(state.snapshot().pendingWrites == 2);
    state.applied(peer);
    assert(state.snapshot().pendingWrites == 1 && state.snapshot().pendingBytes == 29);
    assert(state.takeWrite().id == second);
    state.completeWrite(second, OMNI_USD_LIVE_OUTCOME_UNKNOWN);
    state.fail(OMNI_USD_LIVE_UNAUTHENTICATED);
    assert(state.snapshot().pendingWrites == 1 && state.snapshot().pendingBytes == 29);
    assert(state.snapshot().result == OMNI_USD_LIVE_OUTCOME_UNKNOWN);
    LayerProgress bounded(key(), 1, value(), 10, 100);
    assert(bounded.admit(value(), {}, UINT64_MAX) == 1);
    error(OMNI_USD_LIVE_RESOURCE_EXHAUSTED, [&] { bounded.admit(value(), {}, 1); });
    assert(bounded.snapshot().pendingWrites == 1 && bounded.snapshot().pendingBytes == UINT64_MAX);
}

namespace {
void admittedLedger(const LayerProgress& state,const std::vector<uint64_t>& accepted,size_t retired,uint64_t initial) {
    const auto snapshot=state.snapshot();
    assert(snapshot.initialApplied==initial && snapshot.localAdmitted==accepted.size() && snapshot.localRetired==retired);
    std::vector<uint64_t> remaining;
    for (const auto& work:state.retained()) {
        if (work.disposition==State::Peer) { assert(work.localOrdinal==0); continue; }
        assert(work.localOrdinal>retired && work.localOrdinal<=accepted.size());
        assert(work.id==accepted[work.localOrdinal-1]);
        remaining.push_back(work.id);
    }
    assert(remaining==std::vector<uint64_t>(accepted.begin()+retired,accepted.end()));
    assert(snapshot.pendingWrites==accepted.size()-retired);
}
void localAdmissionHistory() {
    LayerProgress state(key(),10,value(),8,100); open(state,10);
    std::vector<uint64_t> accepted;
    admittedLedger(state,accepted,0,10);
    accepted.push_back(state.admit(value(101)));
    const auto peer11=state.acceptCoverage(1,1,11,11,proof(value(11)));
    accepted.push_back(state.admit(value(102)));
    auto write=state.takeWrite(); assert(write.id==accepted[0] && write.localOrdinal==1);
    state.deferWrite(write.id); admittedLedger(state,accepted,0,10);
    write=state.takeWrite(); assert(write.id==accepted[0] && write.localOrdinal==1);
    const auto peer13=state.acceptCoverage(1,1,13,13,proof(value(13)));
    accepted.push_back(state.admit(value(103)));
    assert(accepted[1]>accepted[0]+1 && accepted[2]>accepted[1]+1); // Peer IDs interleave; local ordinals do not.
    admittedLedger(state,accepted,0,10);
    state.completeWrite(write.id,OMNI_USD_LIVE_OK,12);
    write=state.takeWrite(); assert(write.id==accepted[1] && write.localOrdinal==2);
    state.completeWrite(write.id,OMNI_USD_LIVE_OK,14);
    write=state.takeWrite(); assert(write.id==accepted[2] && write.localOrdinal==3);
    state.completeWrite(write.id,OMNI_USD_LIVE_OK,15);
    admittedLedger(state,accepted,0,10); // ACK alone retires nothing.
    state.applied(accepted[2]); state.applied(accepted[1]); state.applied(accepted[0]);
    assert(state.snapshot().applied==10); admittedLedger(state,accepted,0,10);
    state.applied(peer11);
    assert(state.snapshot().applied==12); admittedLedger(state,accepted,1,10);
    state.disconnect(); const auto cursor=state.startRead();
    assert(cursor.sequence==15); assert(state.caughtUp(cursor.generation,1,15).accepted);
    admittedLedger(state,accepted,1,10); // Original resume survives reconnect, not reset to15.
    state.applied(peer13);
    assert(state.snapshot().applied==15 && state.snapshot().retained==0);
    admittedLedger(state,accepted,3,10);

    // Cumulative counters outlive a tiny retained suffix. Every round has a
    // new peer ID, then local admission, so IDs cannot substitute for ordinals.
    LayerProgress bounded(key(),5,value(),2,100); open(bounded,5);
    std::vector<uint64_t> history;
    for (uint64_t round=0;round<64;++round) {
        const uint64_t sequence=6+2*round;
        const auto peer=bounded.acceptCoverage(1,1,sequence,sequence,proof(value(sequence)));
        history.push_back(bounded.admit(value(1000+round)));
        admittedLedger(bounded,history,round,5);
        error(OMNI_USD_LIVE_RESOURCE_EXHAUSTED,[&] { bounded.admit(value(999)); });
        admittedLedger(bounded,history,round,5);
        const auto pending=bounded.takeWrite();
        assert(pending.id==history.back() && pending.localOrdinal==round+1);
        bounded.completeWrite(pending.id,OMNI_USD_LIVE_OK,sequence+1);
        bounded.applied(pending.id); admittedLedger(bounded,history,round,5);
        bounded.applied(peer); admittedLedger(bounded,history,round+1,5);
        assert(bounded.snapshot().retained==0);
    }
    assert(bounded.snapshot().localAdmitted==64 && bounded.snapshot().localRetired==64);

    for (const auto outcome:{OmniUsdLiveResult(OMNI_USD_LIVE_OUTCOME_UNKNOWN),OmniUsdLiveResult(OMNI_USD_LIVE_PERMISSION_DENIED)}) {
        LayerProgress stopped(key(),10,value(),4,100); open(stopped,10);
        const auto peer=stopped.acceptCoverage(1,1,11,11,proof(value(11))); stopped.applied(peer);
        const std::vector<uint64_t> ids{stopped.admit(value(1)),stopped.admit(value(2))};
        const auto issued=stopped.takeWrite(); assert(issued.id==ids.front() && issued.localOrdinal==1);
        stopped.completeWrite(issued.id,outcome);
        const auto kept=stopped.retained();
        assert(kept.front().disposition==(outcome==OMNI_USD_LIVE_OUTCOME_UNKNOWN?State::Unknown:State::Rejected));
        admittedLedger(stopped,ids,0,10);
        error(outcome,[&] { stopped.admit(value(3)); });
        admittedLedger(stopped,ids,0,10);
    }
    LayerProgress overflow(key(),10,value(),4,100);
    const std::vector<uint64_t> ids{overflow.admit(value(),{},UINT64_MAX)};
    error(OMNI_USD_LIVE_RESOURCE_EXHAUSTED,[&] { overflow.admit(value(),{},1); });
    admittedLedger(overflow,ids,0,10);

    bool retirementSucceeded=false;
    unsigned retirementFailures=0;
    for (size_t allowance=0;allowance<8 && !retirementSucceeded;++allowance) {
        LayerProgress retiring(key(),10,value(),4,100); open(retiring,10);
        const auto peer=retiring.acceptCoverage(1,1,11,11,proof(value(11)));
        const std::vector<uint64_t> acceptedIds{retiring.admit(value(12))};
        assert(retiring.takeWrite().id==acceptedIds.front());
        retiring.completeWrite(acceptedIds.front(),OMNI_USD_LIVE_OK,12);
        retiring.applied(acceptedIds.front());
        const auto before=retiring.retained();
        liveTestFailAllocationsAfter(allowance);
        try { retiring.applied(peer); retirementSucceeded=true; }
        catch (const std::bad_alloc&) { ++retirementFailures; }
        liveTestRestoreAllocations();
        if (retirementSucceeded) admittedLedger(retiring,acceptedIds,1,10);
        else {
            admittedLedger(retiring,acceptedIds,0,10); assert(retiring.snapshot().applied==10);
            const auto after=retiring.retained(); assert(after.size()==before.size());
            for (size_t i=0;i<before.size();++i)
                assert(after[i].id==before[i].id && after[i].localOrdinal==before[i].localOrdinal &&
                       after[i].owner==before[i].owner && after[i].applied==before[i].applied);
        }
    }
    assert(retirementSucceeded && retirementFailures);
}
void mutationRejectionsAndReissue()
{
    // Definitive rejections keep their typed result; every other failure may
    // have committed and is uncertain. Transport failure (ok=false) is mapped
    // to OUTCOME_UNKNOWN by each mutation completion before classification.
    const std::pair<grpc::StatusCode, OmniUsdLiveResult> definitive[] = {
        {grpc::StatusCode::INVALID_ARGUMENT, OMNI_USD_LIVE_INVALID_ARGUMENT},
        {grpc::StatusCode::OUT_OF_RANGE, OMNI_USD_LIVE_INVALID_ARGUMENT},
        {grpc::StatusCode::NOT_FOUND, OMNI_USD_LIVE_NOT_FOUND},
        {grpc::StatusCode::FAILED_PRECONDITION, OMNI_USD_LIVE_OBJECT_REPLACED},
        {grpc::StatusCode::PERMISSION_DENIED, OMNI_USD_LIVE_PERMISSION_DENIED},
        {grpc::StatusCode::UNAUTHENTICATED, OMNI_USD_LIVE_UNAUTHENTICATED},
        {grpc::StatusCode::RESOURCE_EXHAUSTED, OMNI_USD_LIVE_RESOURCE_EXHAUSTED},
        {grpc::StatusCode::ALREADY_EXISTS, OMNI_USD_LIVE_ALREADY_EXISTS}};
    assert(classifyUnary(grpc::StatusCode::OK, true) == OMNI_USD_LIVE_OK);
    for (const auto& [code, result] : definitive) assert(classifyUnary(code, true) == result);
    for (auto code : {grpc::StatusCode::UNAVAILABLE, grpc::StatusCode::DEADLINE_EXCEEDED, grpc::StatusCode::INTERNAL,
                      grpc::StatusCode::CANCELLED, grpc::StatusCode::UNKNOWN, grpc::StatusCode::ABORTED,
                      grpc::StatusCode::UNIMPLEMENTED, grpc::StatusCode::DATA_LOSS})
        assert(classifyUnary(code, true) == OMNI_USD_LIVE_OUTCOME_UNKNOWN);
    // A transport failure on Finish (ok=false) never reaches classifyUnary; a
    // mutation in that state is OUTCOME_UNKNOWN, a read is INTERNAL_ERROR.
    assert(classifyTransportFailure(true) == OMNI_USD_LIVE_OUTCOME_UNKNOWN);
    assert(classifyTransportFailure(false) == OMNI_USD_LIVE_INTERNAL_ERROR);
    // A definitive rejection retires the issued intent as Rejected and stops
    // issuance; the intent and every later queued intent stay retained.
    for (const auto& [code, result] : definitive) {
        if (result == OMNI_USD_LIVE_UNAUTHENTICATED) continue;
        LayerProgress p(key(), 10, value(), 16, 1000); open(p, 10);
        const auto id = p.admit(value(1)); p.admit(value(2));
        assert(p.takeWrite().id == id);
        assert(p.completeWrite(id, result) == LayerProgress::WriteSettlement::Settled);
        const auto state = p.snapshot();
        assert(state.gate == Gate::Failed && state.result == result && !state.issued && state.queued == 1 &&
               state.pendingWrites == 2 && state.retained == 2);
        assert(p.retained().front().disposition == State::Rejected && !p.takeWrite().id);
        error(result, [&] { p.admit(value(3)); });
    }
    // UNAUTHENTICATED: the first rejection keeps the intent at the FIFO head
    // for exactly one reissue; the second fails the layer.
    LayerProgress p(key(), 10, value(), 16, 1000); open(p, 10);
    const auto id = p.admit(value(1)); const auto later = p.admit(value(2));
    assert(p.takeWrite().id == id);
    assert(p.completeWrite(id, OMNI_USD_LIVE_UNAUTHENTICATED) == LayerProgress::WriteSettlement::AwaitAuth);
    auto state = p.snapshot();
    assert(state.gate == Gate::Ready && state.result == OMNI_USD_LIVE_OK && state.queued == 2 &&
           state.pendingWrites == 2 && !state.issued);
    assert(p.retained().front().disposition == State::Queued && p.retained().front().authReissued);
    auto reissued = p.takeWrite(); // The same intent, exactly once more.
    assert(reissued.id == id && reissued.authReissued);
    assert(p.completeWrite(id, OMNI_USD_LIVE_UNAUTHENTICATED) == LayerProgress::WriteSettlement::Settled);
    state = p.snapshot();
    assert(state.gate == Gate::Failed && state.result == OMNI_USD_LIVE_UNAUTHENTICATED && state.queued == 1);
    assert(p.retained().front().disposition == State::Rejected && !p.takeWrite().id);
    assert(p.retained().back().id == later && p.retained().back().disposition == State::Queued &&
           !p.retained().back().authReissued);
    // A successful reissue leaves later intents with their own single reissue.
    LayerProgress q(key(2), 10, value(), 16, 1000); open(q, 10);
    const auto first = q.admit(value(1)); const auto second = q.admit(value(2));
    assert(q.takeWrite().id == first);
    assert(q.completeWrite(first, OMNI_USD_LIVE_UNAUTHENTICATED) == LayerProgress::WriteSettlement::AwaitAuth);
    assert(q.takeWrite().id == first);
    assert(q.completeWrite(first, OMNI_USD_LIVE_OK, 11) == LayerProgress::WriteSettlement::Settled);
    assert(q.takeWrite().id == second);
    assert(q.completeWrite(second, OMNI_USD_LIVE_UNAUTHENTICATED) == LayerProgress::WriteSettlement::AwaitAuth);
    assert(q.snapshot().gate == Gate::Ready && q.snapshot().covered == 11);
    // A layer that already failed retires an UNAUTHENTICATED completion as
    // rejected, keeping its first cause; no reissue is offered.
    LayerProgress r(key(3), 10, value(), 16, 1000); open(r, 10);
    const auto pending = r.admit(value(1));
    assert(r.takeWrite().id == pending);
    r.fail(OMNI_USD_LIVE_MALFORMED_STREAM);
    assert(r.completeWrite(pending, OMNI_USD_LIVE_UNAUTHENTICATED) == LayerProgress::WriteSettlement::Settled);
    assert(r.snapshot().result == OMNI_USD_LIVE_MALFORMED_STREAM && r.retained().front().disposition == State::Rejected);
}
// An unknown outcome of a write that carries an operation ID keeps it in flight for one re-send of the same bytes
// with that ID (design section 13). A position the read skips meanwhile is not an empty commit, and the re-send's
// ALREADY_COMMITTED reply at that position is accepted although the read has moved past it.
void operationIdResend()
{
    LayerProgress p(key(), 10, value(), 16, 1000); open(p, 10);
    const auto id = p.admit(value(1)); const auto later = p.admit(value(2));
    assert(p.takeWrite().id == id);
    assert(p.resendWrite(id));
    auto state = p.snapshot();
    assert(state.gate == Gate::Ready && state.result == OMNI_USD_LIVE_OK && state.issued == 1 && state.queued == 1);
    assert(p.retained().front().disposition == State::Issued && p.retained().front().resent);
    assert(!p.takeWrite().id);
    assert(!p.resendWrite(id)); // Exactly one re-send.
    // The read never carries this writer's commit at 11; it delivers peer work at 12 and catches up there.
    const auto peer = p.acceptCoverage(1, 1, 12, 12, proof(value(3)));
    assert(p.caughtUp(1, 1, 12).accepted);
    assert(p.snapshot().covered == 10 && p.snapshot().applied == 10);
    assert(p.completeWrite(id, OMNI_USD_LIVE_OK, 11) == LayerProgress::WriteSettlement::Settled);
    state = p.snapshot();
    assert(state.gate == Gate::Ready && state.result == OMNI_USD_LIVE_OK && state.covered == 12 && !state.issued);
    p.applied(id); p.applied(peer);
    assert(p.snapshot().applied == 12 && p.takeWrite().id == later);
    // A re-send whose own outcome is unknown, or that is rejected UNAUTHENTICATED and so earns no further issue,
    // leaves the first attempt's outcome unknown.
    for (const auto result : {OMNI_USD_LIVE_OUTCOME_UNKNOWN, OMNI_USD_LIVE_UNAUTHENTICATED}) {
        LayerProgress u(key(2), 10, value(), 16, 1000); open(u, 10);
        const auto uncertain = u.admit(value(1)); u.admit(value(2));
        assert(u.takeWrite().id == uncertain && u.resendWrite(uncertain));
        assert(u.completeWrite(uncertain, OmniUsdLiveResult(result)) == LayerProgress::WriteSettlement::Settled);
        state = u.snapshot();
        assert(state.gate == Gate::Failed && state.result == OMNI_USD_LIVE_OUTCOME_UNKNOWN && !state.readStarted &&
               !state.readingAfterFailure && state.queued == 1 && state.pendingWrites == 2);
        assert(u.retained().front().disposition == State::Unknown && !u.takeWrite().id);
    }
    // A definitive rejection of the re-send is final with its typed result: ALREADY_EXISTS (the ID was committed
    // with other content), OBJECT_REPLACED, NOT_FOUND and the others. It says nothing of the first attempt, which
    // may have committed at the position the read skipped, so the read closes as for an unknown outcome: that
    // position never fills as an empty commit, and no applied frontier passes it.
    for (const auto result : {OMNI_USD_LIVE_ALREADY_EXISTS, OMNI_USD_LIVE_OBJECT_REPLACED, OMNI_USD_LIVE_NOT_FOUND,
             OMNI_USD_LIVE_PERMISSION_DENIED, OMNI_USD_LIVE_INVALID_ARGUMENT, OMNI_USD_LIVE_RESOURCE_EXHAUSTED}) {
        LayerProgress d(key(5), 10, value(), 16, 1000); open(d, 10);
        const auto rejected = d.admit(value(1)); d.admit(value(2));
        assert(d.takeWrite().id == rejected && d.resendWrite(rejected));
        // The first attempt committed at 11, unseen; the read delivers a peer's commit at 12.
        d.acceptCoverage(1, 1, 12, 12, proof(value(3)));
        assert(d.completeWrite(rejected, OmniUsdLiveResult(result)) == LayerProgress::WriteSettlement::Settled);
        state = d.snapshot();
        assert(state.gate == Gate::Failed && state.result == result && !state.readStarted &&
               !state.readingAfterFailure && state.queued == 1 && state.pendingWrites == 2);
        assert(state.covered == 10 && state.applied == 10 && d.applicable(8).empty());
        assert(d.retained().front().disposition == State::Rejected && !d.takeWrite().id);
        assert(!d.acceptCoverage(1, 1, 13, 13, proof(value(4))) && !d.caughtUp(1, 1, 13).accepted);
        assert(d.snapshot().covered == 10);
        error(result, [&] { d.startRead(); });
    }
    // So does a definitive rejection of a write requeued after an attempt of unknown outcome, as a chained write's
    // re-send is.
    for (const auto result : {OMNI_USD_LIVE_PERMISSION_DENIED, OMNI_USD_LIVE_NOT_FOUND}) {
        LayerProgress m(key(6), 10, value(), 16, 1000); open(m, 10);
        const auto uncertain = m.admit(value(1));
        assert(m.takeWrite().id == uncertain && m.requeueWrite(uncertain, true));
        assert(m.takeWrite().id == uncertain);
        m.acceptCoverage(1, 1, 12, 12, proof(value(3)));
        assert(m.completeWrite(uncertain, OmniUsdLiveResult(result)) == LayerProgress::WriteSettlement::Settled);
        state = m.snapshot();
        assert(state.gate == Gate::Failed && state.result == result && !state.readStarted &&
               !state.readingAfterFailure && state.covered == 10 && state.applied == 10);
        assert(m.retained().front().disposition == State::Rejected && m.applicable(8).empty());
    }
    // A layer already reading after an issuance-only failure keeps that cause and closes its read.
    LayerProgress h(key(7), 10, value(), 16, 1000); open(h, 10);
    const auto resent = h.admit(value(1));
    assert(h.takeWrite().id == resent && h.resendWrite(resent));
    h.stopIssuance(OMNI_USD_LIVE_UNSUPPORTED);
    h.acceptCoverage(1, 1, 12, 12, proof(value(3)));
    assert(h.snapshot().readingAfterFailure && h.snapshot().covered == 10);
    assert(h.completeWrite(resent, OMNI_USD_LIVE_NOT_FOUND) == LayerProgress::WriteSettlement::Settled);
    state = h.snapshot();
    assert(state.result == OMNI_USD_LIVE_UNSUPPORTED && !state.readStarted && !state.readingAfterFailure &&
           state.covered == 10 && h.applicable(8).empty());
    // A first attempt's definitive rejection leaves no earlier attempt of unknown outcome: the read stays open, as
    // design section 13 keeps it after a definitive rejection, and the position it skipped fills as an empty commit.
    LayerProgress k(key(8), 10, value(), 16, 1000); open(k, 10);
    const auto once = k.admit(value(1));
    assert(k.takeWrite().id == once);
    const auto peer12 = k.acceptCoverage(1, 1, 12, 12, proof(value(3)));
    assert(k.completeWrite(once, OMNI_USD_LIVE_PERMISSION_DENIED) == LayerProgress::WriteSettlement::Settled);
    state = k.snapshot();
    assert(state.result == OMNI_USD_LIVE_PERMISSION_DENIED && state.readingAfterFailure && state.covered == 12);
    const auto run = k.applicable(8);
    assert(run.size() == 1 && run[0].id == peer12);
    // The re-send's acknowledgment keeps every other check: peer coverage already holding its position collides.
    LayerProgress c(key(3), 10, value(), 16, 1000); open(c, 10);
    const auto collided = c.admit(value(1));
    assert(c.takeWrite().id == collided && c.resendWrite(collided));
    c.acceptCoverage(1, 1, 11, 11, proof(value(3)));
    c.completeWrite(collided, OMNI_USD_LIVE_OK, 11);
    assert(c.snapshot().gate == Gate::Failed && c.snapshot().result == OMNI_USD_LIVE_OUTCOME_UNKNOWN);
    // A failed layer never re-sends, and only the write in flight can be re-sent.
    LayerProgress f(key(4), 10, value(), 16, 1000); open(f, 10);
    const auto stopped = f.admit(value(1)); const auto queued = f.admit(value(2));
    assert(f.takeWrite().id == stopped);
    bool refused = false;
    try { f.resendWrite(queued); } catch (const std::invalid_argument&) { refused = true; }
    assert(refused);
    f.fail(OMNI_USD_LIVE_MALFORMED_STREAM);
    assert(!f.resendWrite(stopped) && !f.retained().front().resent);
    f.completeWrite(stopped, OMNI_USD_LIVE_OUTCOME_UNKNOWN);
    assert(f.snapshot().result == OMNI_USD_LIVE_OUTCOME_UNKNOWN);
}

// A refusal that applied nothing keeps the intent at the FIFO head for a
// reissue of the same bytes, up to the attempt bound; the bound's refusal
// fails the layer UNAVAILABLE as a definitive rejection (design section 13).
void nothingAppliedReissue()
{
    LayerProgress p(key(), 10, value(), 16, 1000); open(p, 10);
    const auto id = p.admit(value(1)); const auto later = p.admit(value(2));
    for (uint32_t refusal = 1; refusal < 3; ++refusal) {
        assert(p.takeWrite().id == id);
        assert(p.refuseWrite(id, 3) == refusal);
        const auto state = p.snapshot();
        assert(state.gate == Gate::Ready && state.result == OMNI_USD_LIVE_OK && !state.issued && state.queued == 2);
        assert(p.retained().front().disposition == State::Queued && p.retained().front().refusals == refusal);
    }
    // While the reissue waits, no write holds a position: the read's skipped
    // positions are empty commits.
    p.acceptCoverage(1, 1, 12, 12, proof(value(3)));
    assert(p.snapshot().covered == 12);
    assert(p.takeWrite().id == id && !p.refuseWrite(id, 3)); // The third refusal reaches the bound.
    assert(p.completeWrite(id, OMNI_USD_LIVE_UNAVAILABLE) == LayerProgress::WriteSettlement::Settled);
    auto state = p.snapshot();
    assert(state.gate == Gate::Failed && state.result == OMNI_USD_LIVE_UNAVAILABLE && state.readingAfterFailure &&
           state.queued == 1 && state.pendingWrites == 2);
    assert(p.retained().front().disposition == State::Rejected && !p.takeWrite().id);
    // A reissue that commits settles as any attempt; the next intent starts its own count.
    LayerProgress q(key(2), 10, value(), 16, 1000); open(q, 10);
    const auto first = q.admit(value(1)); const auto second = q.admit(value(2));
    assert(q.takeWrite().id == first && q.refuseWrite(first, 3) == 1);
    assert(q.takeWrite().id == first);
    assert(q.completeWrite(first, OMNI_USD_LIVE_OK, 11) == LayerProgress::WriteSettlement::Settled);
    assert(q.takeWrite().id == second && q.refuseWrite(second, 3) == 1);
    // An UNAUTHENTICATED reissue between refusals restarts the consecutive count.
    assert(q.takeWrite().id == second && q.refuseWrite(second, 3) == 2);
    assert(q.takeWrite().id == second);
    assert(q.completeWrite(second, OMNI_USD_LIVE_UNAUTHENTICATED) == LayerProgress::WriteSettlement::AwaitAuth);
    assert(q.takeWrite().id == second && q.refuseWrite(second, 3) == 1);
    // A re-sent write is not reissued: the re-send applied nothing, but its
    // first attempt's outcome stays unknown.
    LayerProgress u(key(3), 10, value(), 16, 1000); open(u, 10);
    const auto uncertain = u.admit(value(1));
    assert(u.takeWrite().id == uncertain && u.resendWrite(uncertain) && !u.refuseWrite(uncertain, 3));
    assert(u.completeWrite(uncertain, OMNI_USD_LIVE_UNAVAILABLE) == LayerProgress::WriteSettlement::Settled);
    assert(u.snapshot().gate == Gate::Failed && u.snapshot().result == OMNI_USD_LIVE_OUTCOME_UNKNOWN);
    assert(u.retained().front().disposition == State::Unknown);
    // A stopped layer offers no reissue, a bound of one fails at the first
    // refusal, and only the write in flight can be refused.
    LayerProgress f(key(4), 10, value(), 16, 1000); open(f, 10);
    const auto stopped = f.admit(value(1)); const auto queued = f.admit(value(2));
    assert(f.takeWrite().id == stopped && !f.refuseWrite(stopped, 1));
    bool refused = false;
    try { f.refuseWrite(queued, 3); } catch (const std::invalid_argument&) { refused = true; }
    assert(refused);
    f.fail(OMNI_USD_LIVE_MALFORMED_STREAM);
    assert(!f.refuseWrite(stopped, 3));
    // A position the read skips while the write is issued may be that write's;
    // the refusal proves it is not, so it fills as an empty commit with no
    // further read, and the peer update after it applies during the backoff.
    LayerProgress h(key(5), 10, value(), 16, 1000); open(h, 10);
    const auto held = h.admit(value(1));
    assert(h.takeWrite().id == held);
    const auto peer = h.acceptCoverage(1, 1, 12, 12, proof(value(3)));
    assert(h.snapshot().covered == 10 && h.applicable(8).empty());
    assert(h.refuseWrite(held, 3) == 1);
    auto quiet = h.snapshot();
    assert(quiet.gate == Gate::Ready && quiet.covered == 12 && quiet.applied == 11 && !quiet.issued);
    const auto run = h.applicable(8);
    assert(run.size() == 1 && run[0].id == peer);
    h.applied(peer); assert(h.snapshot().applied == 12 && h.retained().front().id == held);
    // A skipped hole wider than the gap bound fails the layer, as when a
    // settled write leaves it; the refusal then earns no reissue.
    LayerProgress w(key(6), 10, value(), 16, 4); open(w, 10);
    const auto wide = w.admit(value(1));
    assert(w.takeWrite().id == wide);
    w.acceptCoverage(1, 1, 20, 20, proof(value(3)));
    assert(!w.refuseWrite(wide, 3));
    quiet = w.snapshot();
    assert(quiet.gate == Gate::Failed && quiet.result == OMNI_USD_LIVE_RESOURCE_EXHAUSTED && quiet.issued == 1);
    assert(w.completeWrite(wide, OMNI_USD_LIVE_UNAVAILABLE) == LayerProgress::WriteSettlement::Settled);
    assert(w.snapshot().result == OMNI_USD_LIVE_RESOURCE_EXHAUSTED && w.retained().front().refusals == 0);
    (void)later;
}

// An UNAUTHENTICATED attempt never executed, so while its reissue waits for a
// token the write holds no position: the positions the read skipped while it
// was issued fill as empty commits, as after a refusal that applied nothing.
void authWaitFillsHoles()
{
    // The read stays quiet through the auth wait, and the peer update after
    // the skipped position applies without waiting for another read.
    LayerProgress p(key(), 10, value(), 16, 1000); open(p, 10);
    const auto id = p.admit(value(1));
    assert(p.takeWrite().id == id);
    const auto peer = p.acceptCoverage(1, 1, 12, 12, proof(value(3)));
    assert(p.snapshot().covered == 10 && p.applicable(8).empty());
    assert(p.completeWrite(id, OMNI_USD_LIVE_UNAUTHENTICATED) == LayerProgress::WriteSettlement::AwaitAuth);
    auto state = p.snapshot();
    assert(state.gate == Gate::Ready && state.result == OMNI_USD_LIVE_OK && state.covered == 12 &&
           state.applied == 11 && !state.issued && state.queued == 1);
    auto run = p.applicable(8);
    assert(run.size() == 1 && run[0].id == peer);
    p.applied(peer); assert(p.snapshot().applied == 12);
    // The reissue settles above the filled position.
    assert(p.takeWrite().id == id);
    assert(p.completeWrite(id, OMNI_USD_LIVE_OK, 13) == LayerProgress::WriteSettlement::Settled);
    assert(p.snapshot().gate == Gate::Ready && p.snapshot().covered == 13);
    // A skipped hole wider than the gap bound fails the layer RESOURCE_EXHAUSTED; the write then settles as rejected
    // and earns no reissue.
    LayerProgress w(key(2), 10, value(), 16, 4); open(w, 10);
    const auto wide = w.admit(value(1));
    assert(w.takeWrite().id == wide);
    w.acceptCoverage(1, 1, 20, 20, proof(value(3)));
    assert(w.completeWrite(wide, OMNI_USD_LIVE_UNAUTHENTICATED) == LayerProgress::WriteSettlement::Settled);
    state = w.snapshot();
    assert(state.gate == Gate::Failed && state.result == OMNI_USD_LIVE_RESOURCE_EXHAUSTED && !state.readStarted &&
           !state.readingAfterFailure && !state.issued && state.covered == 10 && state.pendingWrites == 1);
    assert(w.retained().front().disposition == State::Rejected && !w.retained().front().authReissued);
    // A write an earlier attempt of which may have committed keeps holding the positions above its floor.
    LayerProgress u(key(3), 10, value(), 16, 1000); open(u, 10);
    const auto uncertain = u.admit(value(1));
    assert(u.takeWrite().id == uncertain && u.requeueWrite(uncertain, true));
    assert(u.takeWrite().id == uncertain);
    u.acceptCoverage(1, 1, 12, 12, proof(value(3)));
    assert(u.completeWrite(uncertain, OMNI_USD_LIVE_UNAUTHENTICATED) == LayerProgress::WriteSettlement::AwaitAuth);
    assert(u.snapshot().covered == 10 && u.snapshot().uncertain == 1 && u.applicable(8).empty());
    // So does a later write still in flight: the chain commits it above the rejected one's floor.
    LayerProgress l(key(4), 10, value(), 16, 1000); open(l, 10);
    const auto head = l.admit(value(1)); const auto next = l.admit(value(2));
    assert(l.takeWrite(2).id == head && l.takeWrite(2).id == next);
    l.acceptCoverage(1, 1, 12, 12, proof(value(3)));
    assert(l.completeWrite(head, OMNI_USD_LIVE_UNAUTHENTICATED) == LayerProgress::WriteSettlement::AwaitAuth);
    assert(l.snapshot().covered == 10 && l.snapshot().issued == 1);
    // A taken write deferred before it reaches transport, as when no usable token is installed, holds no position.
    LayerProgress d(key(5), 10, value(), 16, 1000); open(d, 10);
    const auto deferred = d.admit(value(1));
    assert(d.takeWrite().id == deferred);
    d.acceptCoverage(1, 1, 12, 12, proof(value(3)));
    assert(d.snapshot().covered == 10);
    d.deferWrite(deferred);
    state = d.snapshot();
    assert(state.gate == Gate::Ready && state.covered == 12 && state.applied == 11 && !state.issued &&
           d.retained().front().disposition == State::Queued && !d.retained().front().attempts);
    LayerProgress dw(key(6), 10, value(), 16, 4); open(dw, 10);
    const auto taken = dw.admit(value(1));
    assert(dw.takeWrite().id == taken);
    dw.acceptCoverage(1, 1, 20, 20, proof(value(3)));
    dw.deferWrite(taken);
    assert(dw.snapshot().gate == Gate::Failed && dw.snapshot().result == OMNI_USD_LIVE_RESOURCE_EXHAUSTED &&
           dw.retained().front().disposition == State::Queued);
    // Regression reproduction: a header-only frame names an empty commit at 11, or the read skips 11, and a peer
    // update follows at 12 while the write is issued; the write is rejected UNAUTHENTICATED and the read reconnects
    // during the auth wait. The read resumes from 12, and the layer is ready and healthy.
    for (const bool named : {true, false}) {
        LayerProgress q(key(7), 10, value(), 16, 1000); open(q, 10);
        const auto write = q.admit(value(1));
        assert(q.takeWrite().id == write);
        if (named) assert(q.caughtUp(1, 1, 11).accepted);
        const auto update = q.acceptCoverage(1, 1, 12, 12, proof(value(3)));
        assert(q.completeWrite(write, OMNI_USD_LIVE_UNAUTHENTICATED) == LayerProgress::WriteSettlement::AwaitAuth);
        assert(q.snapshot().covered == 12);
        const auto generation = q.disconnect();
        const auto cursor = q.startRead();
        assert(cursor.generation == generation && cursor.sequence == 12);
        assert(q.caughtUp(generation, 1, 12).accepted);
        state = q.snapshot();
        assert(state.gate == Gate::Ready && state.result == OMNI_USD_LIVE_OK && state.covered == 12);
        q.applied(update);
        assert(q.takeWrite().id == write);
        assert(q.completeWrite(write, OMNI_USD_LIVE_OK, 13) == LayerProgress::WriteSettlement::Settled);
        assert(q.snapshot().covered == 13 && q.snapshot().applied == 12);
    }
}

void issuanceStopKeepsReading()
{
    // A definitive rejection or a host failure stops issuance only: pending
    // work stays retained, the read stays open, and received peer work keeps
    // advancing the applied frontier.
    LayerProgress p(key(), 10, value(), 16, 1000); open(p, 10);
    const auto held = p.admit(value(1));
    p.stopIssuance(OMNI_USD_LIVE_UNSUPPORTED);
    auto state = p.snapshot();
    assert(state.gate == Gate::Failed && state.result == OMNI_USD_LIVE_UNSUPPORTED && state.readStarted &&
           state.readingAfterFailure && state.pendingWrites == 1 && state.queued == 1 && state.retained == 1);
    assert(!p.takeWrite().id && p.retained().front().id == held);
    error(OMNI_USD_LIVE_UNSUPPORTED, [&] { p.admit(value(2)); });
    const auto generation = state.generation;
    const auto peer = p.acceptCoverage(generation, 1, 11, 11, proof(value(11)));
    assert(peer); p.applied(peer);
    state = p.snapshot();
    assert(state.applied == 11 && state.covered == 11 && state.retained == 1 && state.pendingWrites == 1 &&
           state.gate == Gate::Failed && state.result == OMNI_USD_LIVE_UNSUPPORTED);
    // A caught-up header advances the read, never the gate.
    const auto caught = p.caughtUp(generation, 1, 11);
    assert(caught.accepted && !caught.initialLatest && p.snapshot().gate == Gate::Failed && !p.takeWrite().id);
    // The read reconnects like a healthy one, resuming from covered work.
    const auto next = p.disconnect();
    assert(next == generation + 1 && !p.snapshot().readStarted && p.snapshot().readingAfterFailure &&
           p.snapshot().gate == Gate::Failed);
    assert(!p.acceptCoverage(generation, 1, 12, 12, proof(value(12)))); // The old stream is stale.
    const auto resumed = p.startRead();
    assert(resumed.generation == next && resumed.sequence == 11 && p.snapshot().readStarted);
    const auto later = p.acceptCoverage(next, 1, 12, 12, proof(value(12)));
    assert(later); p.applied(later);
    assert(p.snapshot().applied == 12);
    // Another issuance failure keeps the first cause and the read.
    p.stopIssuance(OMNI_USD_LIVE_INVALID_ARGUMENT);
    state = p.snapshot();
    assert(state.result == OMNI_USD_LIVE_UNSUPPORTED && state.readStarted && state.readingAfterFailure);
    // A read-side failure closes the read and freezes the frontier; the first cause stays.
    p.fail(OMNI_USD_LIVE_MALFORMED_STREAM);
    state = p.snapshot();
    assert(state.result == OMNI_USD_LIVE_UNSUPPORTED && !state.readStarted && !state.readingAfterFailure);
    assert(!p.acceptCoverage(next, 1, 13, 13, proof(value(13))) && p.snapshot().applied == 12);
    error(OMNI_USD_LIVE_UNSUPPORTED, [&] { p.startRead(); });
    assert(p.disconnect() == next);
    p.stopIssuance(OMNI_USD_LIVE_INVALID_ARGUMENT); // Never reopens a closed read.
    assert(!p.snapshot().readStarted && !p.snapshot().readingAfterFailure);

    // A definitive write rejection keeps reading; an unknown outcome closes the read.
    LayerProgress rejected(key(2), 10, value(), 16, 1000); open(rejected, 10);
    const auto refused = rejected.admit(value(1)); assert(rejected.takeWrite().id == refused);
    rejected.completeWrite(refused, OMNI_USD_LIVE_INVALID_ARGUMENT);
    assert(rejected.snapshot().readStarted && rejected.snapshot().readingAfterFailure &&
           rejected.retained().front().disposition == State::Rejected);
    LayerProgress unknown(key(3), 10, value(), 16, 1000); open(unknown, 10);
    const auto uncertain = unknown.admit(value(1)); assert(unknown.takeWrite().id == uncertain);
    unknown.completeWrite(uncertain, OMNI_USD_LIVE_OUTCOME_UNKNOWN);
    assert(!unknown.snapshot().readStarted && !unknown.snapshot().readingAfterFailure);
    // A write in flight at a host failure: its unknown outcome still closes the read.
    LayerProgress host(key(4), 10, value(), 16, 1000); open(host, 10);
    const auto inflight = host.admit(value(1)); assert(host.takeWrite().id == inflight);
    host.stopIssuance(OMNI_USD_LIVE_UNSUPPORTED);
    assert(host.snapshot().readingAfterFailure && host.snapshot().issued == 1);
    host.completeWrite(inflight, OMNI_USD_LIVE_OUTCOME_UNKNOWN);
    state = host.snapshot();
    assert(state.result == OMNI_USD_LIVE_OUTCOME_UNKNOWN && !state.readStarted && !state.readingAfterFailure);
}

// Several writes are in flight at once (chained writes, design section 13): they issue in FIFO order up to the
// window, their acknowledgments settle in any order, and every acknowledgment check holds across the window.
void windowedWrites()
{
    LayerProgress p(key(), 10, value(), 32, 1000); open(p, 10);
    const auto a = p.admit(value(1)), b = p.admit(value(2)), c = p.admit(value(3)), d = p.admit(value(4));
    assert(p.takeWrite(3).id == a && p.takeWrite(3).id == b && p.takeWrite(3).id == c && !p.takeWrite(3).id);
    auto state = p.snapshot();
    assert(state.issued == 3 && state.queued == 1 && state.pendingWrites == 4);
    error(OMNI_USD_LIVE_BUSY, [&] { p.startRead(); }); // A reconnect waits for every write in flight.
    // Acknowledgments arrive out of order; positions increase along the FIFO.
    p.completeWrite(b, OMNI_USD_LIVE_OK, 12);
    assert(p.snapshot().covered == 10 && p.snapshot().issued == 2);
    p.completeWrite(c, OMNI_USD_LIVE_OK, 13);
    p.completeWrite(a, OMNI_USD_LIVE_OK, 11);
    state = p.snapshot();
    assert(state.covered == 13 && state.issued == 0 && state.queued == 1 && state.gate == Gate::Ready);
    assert(p.takeWrite(3).id == d);
    p.completeWrite(d, OMNI_USD_LIVE_OK, 14);
    for (const auto id : {a, b, c, d}) p.applied(id);
    assert(p.snapshot().applied == 14 && p.snapshot().retained == 0);

    // An acknowledgment out of chain order collides: the chain commits in FIFO order.
    LayerProgress inverted(key(2), 10, value(), 32, 1000); open(inverted, 10);
    const auto first = inverted.admit(value(1)), second = inverted.admit(value(2));
    assert(inverted.takeWrite(2).id == first && inverted.takeWrite(2).id == second);
    inverted.completeWrite(second, OMNI_USD_LIVE_OK, 11);
    inverted.completeWrite(first, OMNI_USD_LIVE_OK, 12);
    assert(inverted.snapshot().gate == Gate::Failed && inverted.snapshot().result == OMNI_USD_LIVE_OUTCOME_UNKNOWN);

    // Each write keeps the floor of its own first issue; another write's or a peer's position collides.
    LayerProgress floors(key(3), 10, value(), 32, 1000); open(floors, 10);
    const auto early = floors.admit(value(1));
    assert(floors.takeWrite(4).id == early);
    const auto peer = floors.acceptCoverage(1, 1, 12, 12, proof(value(12)));
    const auto late = floors.admit(value(2));
    assert(floors.takeWrite(4).id == late);
    assert(floors.retained().back().floor == 12 && floors.retained().front().floor == 10);
    floors.completeWrite(late, OMNI_USD_LIVE_OK, 13);
    floors.completeWrite(early, OMNI_USD_LIVE_OK, 11);
    assert(floors.snapshot().gate == Gate::Ready && floors.snapshot().covered == 13);
    floors.applied(early); floors.applied(peer); floors.applied(late);
    assert(floors.snapshot().applied == 13);
    LayerProgress below(key(4), 10, value(), 32, 1000); open(below, 10);
    below.acceptCoverage(1, 1, 12, 12, proof(value(12)));
    const auto stale = below.admit(value(1));
    assert(below.takeWrite(4).id == stale);
    below.completeWrite(stale, OMNI_USD_LIVE_OK, 11); // At or below its floor: never this write.
    assert(below.snapshot().result == OMNI_USD_LIVE_OUTCOME_UNKNOWN);
    LayerProgress twice(key(5), 10, value(), 32, 1000); open(twice, 10);
    const auto one = twice.admit(value(1)), two = twice.admit(value(2));
    assert(twice.takeWrite(4).id == one && twice.takeWrite(4).id == two);
    twice.completeWrite(one, OMNI_USD_LIVE_OK, 11);
    twice.completeWrite(two, OMNI_USD_LIVE_OK, 11); // Another local write's position.
    assert(twice.snapshot().result == OMNI_USD_LIVE_OUTCOME_UNKNOWN);
}

// A position the read skips is an empty commit unless an unsettled write may hold it: above that write's floor and
// the last acknowledged earlier write, and below the next acknowledged later write. Other holes fill while writes
// are in flight.
void windowedHoles()
{
    LayerProgress p(key(), 10, value(), 32, 1000); open(p, 10);
    const auto a = p.admit(value(1)), b = p.admit(value(2));
    assert(p.takeWrite(2).id == a && p.takeWrite(2).id == b); // Both floors are 10.
    p.acceptCoverage(1, 1, 14, 14, proof(value(14)));
    assert(p.snapshot().covered == 10);
    p.completeWrite(b, OMNI_USD_LIVE_OK, 12); // 13 cannot be a's, so it is an empty commit.
    assert(p.snapshot().covered == 10);
    p.completeWrite(a, OMNI_USD_LIVE_OK, 11);
    assert(p.snapshot().covered == 14);
    const auto c = p.admit(value(3)), d = p.admit(value(4));
    assert(p.takeWrite(2).id == c); // Floor 14.
    p.acceptCoverage(1, 1, 16, 16, proof(value(16))); // 15 may be c's.
    assert(p.takeWrite(2).id == d); // Floor 16.
    p.acceptCoverage(1, 1, 19, 19, proof(value(19)));
    assert(p.snapshot().covered == 14);
    p.completeWrite(c, OMNI_USD_LIVE_OK, 15);
    assert(p.snapshot().covered == 16); // 17 and 18 may still be d's.
    p.completeWrite(d, OMNI_USD_LIVE_OK, 18);
    assert(p.snapshot().covered == 19);
    // A hole below every unsettled write's range fills while they are in flight.
    LayerProgress q(key(2), 10, value(), 32, 1000); open(q, 10);
    const auto e = q.admit(value(1));
    assert(q.takeWrite(2).id == e); // Floor 10.
    q.acceptCoverage(1, 1, 13, 13, proof(value(13))); // 11 and 12 may be e's.
    const auto f = q.admit(value(2));
    assert(q.takeWrite(2).id == f); // Floor 13.
    q.completeWrite(e, OMNI_USD_LIVE_OK, 11);
    assert(q.snapshot().covered == 13 && q.snapshot().issued == 1);
    // A caught-up header above coverage while a write may hold the gap is not a malformed stream.
    assert(q.caughtUp(1, 1, 15).accepted && q.snapshot().covered == 13);
    q.completeWrite(f, OMNI_USD_LIVE_OK, 14);
    assert(q.snapshot().covered == 15);
}

// A write returned to the FIFO issues again first and keeps its first floor. Only one whose attempt's outcome is
// unknown holds the positions it may have committed at; a refused attempt applied nothing.
void requeuedWrites()
{
    LayerProgress p(key(), 10, value(), 32, 1000); open(p, 10);
    const auto a = p.admit(value(1)), b = p.admit(value(2));
    assert(p.takeWrite(4).id == a && p.takeWrite(4).id == b);
    assert(p.requeueWrite(a, false));
    auto state = p.snapshot();
    assert(state.issued == 1 && state.queued == 1);
    p.acceptCoverage(1, 1, 12, 12, proof(value(12))); // b may hold 11; a's attempt applied nothing.
    assert(p.snapshot().covered == 10);
    assert(p.requeueWrite(b, true));
    assert(p.snapshot().covered == 10 && p.snapshot().issued == 0 && p.snapshot().queued == 2);
    assert(p.takeWrite(1).id == a && !p.takeWrite(1).id);
    assert(p.retained().front().attempts == 2 && p.retained().front().floor == 10);
    p.completeWrite(a, OMNI_USD_LIVE_OK, 13);
    assert(p.snapshot().covered == 13); // b's position is above a's, so 11 is an empty commit.
    assert(p.takeWrite(1).id == b && p.retained()[1].floor == 10);
    p.completeWrite(b, OMNI_USD_LIVE_OK, 14);
    assert(p.snapshot().covered == 14 && p.snapshot().gate == Gate::Ready);
    // A stopped layer requeues nothing; the caller settles the write.
    LayerProgress s(key(2), 10, value(), 32, 1000); open(s, 10);
    const auto x = s.admit(value(1));
    assert(s.takeWrite(4).id == x);
    s.stopIssuance(OMNI_USD_LIVE_UNSUPPORTED);
    assert(!s.requeueWrite(x, true) && s.snapshot().issued == 1);
    // A failure while a queued write's earlier attempt has an unknown outcome leaves that outcome unknown, which
    // closes the read.
    LayerProgress v(key(5), 10, value(), 32, 1000); open(v, 10);
    const auto maybe = v.admit(value(1)), refusedHere = v.admit(value(2));
    assert(v.takeWrite(4).id == maybe && v.takeWrite(4).id == refusedHere);
    assert(v.requeueWrite(maybe, true) && v.requeueWrite(refusedHere, false));
    v.stopIssuance(OMNI_USD_LIVE_UNSUPPORTED);
    state = v.snapshot();
    assert(state.result == OMNI_USD_LIVE_OUTCOME_UNKNOWN && !state.readingAfterFailure && !state.readStarted &&
           state.queued == 1 && state.pendingWrites == 2);
    assert(v.retained()[0].disposition == State::Unknown && v.retained()[1].disposition == State::Queued);
    // A write rejected UNAUTHENTICATED while a later write is in flight waits for its reissue at the FIFO head.
    LayerProgress u(key(3), 10, value(), 32, 1000); open(u, 10);
    const auto head = u.admit(value(1)), next = u.admit(value(2));
    assert(u.takeWrite(2).id == head && u.takeWrite(2).id == next);
    assert(u.completeWrite(head, OMNI_USD_LIVE_UNAUTHENTICATED) == LayerProgress::WriteSettlement::AwaitAuth);
    assert(u.snapshot().issued == 1 && u.snapshot().queued == 1);
    assert(u.requeueWrite(next, false));
    assert(u.takeWrite(2).id == head && u.takeWrite(2).id == next);
    // A definitive rejection fails the layer; a later write that cannot commit keeps that cause.
    LayerProgress r(key(4), 10, value(), 32, 1000); open(r, 10);
    const auto bad = r.admit(value(1)), after = r.admit(value(2));
    assert(r.takeWrite(2).id == bad && r.takeWrite(2).id == after);
    r.completeWrite(bad, OMNI_USD_LIVE_INVALID_ARGUMENT);
    r.completeWrite(after, OMNI_USD_LIVE_CANCELLED);
    state = r.snapshot();
    assert(state.result == OMNI_USD_LIVE_INVALID_ARGUMENT && state.readingAfterFailure && state.issued == 0 &&
           state.pendingWrites == 2);
}

// A write the chain proves committed but whose reply was lost has no reported position, and none is inferred. A read
// open at the proof may have filtered its commit, so the layer waits for a read started after it: that read delivers
// the commit as ordinary coverage, and once it passes the position before the next local write with a position, the
// write retires with that write's application.
void committedRelease()
{
    // The open read delivers a peer update at 11 and filters A's commit at 12; B is acknowledged at 13.
    LayerProgress p(key(), 10, value(), 32, 1000); open(p, 10);
    const auto a = p.admit(value(1)), b = p.admit(value(2));
    assert(p.takeWrite(2).id == a && p.takeWrite(2).id == b);
    assert(p.requeueWrite(a, true));
    p.completeWrite(b, OMNI_USD_LIVE_OK, 13);
    p.confirmCommitted(a);
    auto state = p.snapshot();
    assert(state.gate == Gate::Ready && state.unplaced == 1 && state.uncertain == 0 && state.awaitingRead == 1);
    const auto peer = p.acceptCoverage(1, 1, 11, 11, proof(value(11)));
    assert(peer && p.retained().front().disposition == State::Committed && p.retained().front().first == 0);
    // The read is older than the proof, so it releases nothing.
    assert(p.snapshot().covered == 11 && !p.retained().front().absorbedBy);
    auto run = p.applicable(4);
    assert(run.size() == 1 && run.front().id == peer);
    p.applied(peer);
    // The resubscribed read's catch-up delivers A's commit as an ordinary update and repeats B's as its replay.
    const auto generation = p.disconnect();
    auto read = p.startRead();
    assert(read.generation == generation && read.sequence == 11 && p.snapshot().awaitingRead == 0);
    const auto own = p.acceptCoverage(generation, 1, 12, 12, proof(value(101)));
    assert(own);
    auto work = p.retained();
    assert(work.front().id == a && work.front().absorbedBy == b);
    const auto carrier = std::find_if(work.begin(), work.end(), [&](const auto& item) { return item.id == b; });
    assert(carrier->absorbedLocal == 1 && carrier->absorbedThrough == work.front().localOrdinal);
    assert(p.recoveryReady(generation, 1, 12).accepted);
    run = p.applicable(4);
    assert(run.size() == 2 && run[0].id == own && run[1].id == b && run[1].absorbedLocal == 1);
    p.applied(own);
    p.applied(b);
    state = p.snapshot();
    assert(state.applied == 13 && state.retained == 0 && state.localRetired == 2 && state.unplaced == 0);

    // A committed write without a later local write that has a position keeps the read closed, and the next write
    // issues with it closed to settle it.
    LayerProgress q(key(2), 10, value(), 32, 1000); open(q, 10);
    const auto c = q.admit(value(1)), d = q.admit(value(2));
    assert(q.takeWrite(2).id == c && q.takeWrite(2).id == d);
    q.disconnect();
    assert(q.requeueWrite(c, false) && q.requeueWrite(d, false));
    q.confirmCommitted(c);
    assert(q.snapshot().uncertain == 1);
    error(OMNI_USD_LIVE_BUSY, [&] { q.startRead(); });
    assert(q.takeWrite(1).id == d);
    q.completeWrite(d, OMNI_USD_LIVE_OK, 12);
    assert(q.snapshot().uncertain == 0 && q.startRead().sequence == 10);

    // Work after the position before B waits: positions covered by an older read let B become next before the
    // resumed read reaches its bound, and B retires A only once that read passes it.
    LayerProgress r(key(3), 10, value(), 32, 1000); open(r, 10);
    const auto e = r.admit(value(1)), f = r.admit(value(2));
    assert(r.takeWrite(2).id == e && r.takeWrite(2).id == f);
    const auto p11 = r.acceptCoverage(1, 1, 11, 11, proof(value(11)));
    const auto p13 = r.acceptCoverage(1, 1, 13, 13, proof(value(13)));
    assert(r.requeueWrite(e, true));
    r.completeWrite(f, OMNI_USD_LIVE_OK, 14);
    r.confirmCommitted(e);
    r.applied(p11);
    const auto resumed = r.disconnect();
    assert(r.startRead().sequence == 11);
    const auto p12 = r.acceptCoverage(resumed, 1, 12, 12, proof(value(101)));
    run = r.applicable(4);
    assert(run.size() == 2 && run[0].id == p12 && run[1].id == p13);
    r.applied(p12);
    r.applied(p13);
    assert(r.applicable(4).empty() && r.snapshot().applied == 13);
    assert(!r.acceptCoverage(resumed, 1, 13, 13, proof(value(13))));
    run = r.applicable(4);
    assert(run.size() == 1 && run[0].id == f && run[0].absorbedLocal == 1);
    r.applied(f);
    assert(r.snapshot().localRetired == 2 && r.snapshot().retained == 0);

    // An aggregate of the resumed read that absorbs B holds A's commit too.
    LayerProgress s(key(4), 10, value(), 32, 1000); open(s, 10);
    const auto g = s.admit(value(1)), h = s.admit(value(2)), i = s.admit(value(3));
    assert(s.takeWrite(3).id == g && s.takeWrite(3).id == h && s.takeWrite(3).id == i);
    s.completeWrite(g, OMNI_USD_LIVE_OK, 12);
    assert(s.requeueWrite(h, true));
    s.completeWrite(i, OMNI_USD_LIVE_OK, 14);
    s.confirmCommitted(h);
    const auto again = s.disconnect();
    assert(s.startRead().sequence == 10);
    const auto aggregate =
        s.acceptCoverage(again, 1, 11, 14, proof(value(1114), LayerProgress::CoverageKind::Aggregate));
    work = s.retained();
    assert(work[0].absorbedBy == aggregate && work[1].absorbedBy == aggregate && work[2].absorbedBy == aggregate);
    assert(s.recoveryReady(again, 1, 14).accepted);
    s.applied(aggregate);
    assert(s.snapshot().applied == 14 && s.snapshot().retained == 0 && s.snapshot().localRetired == 3);

    // A net-empty catch-up passes A and B: B retires without applying, and A retires with it.
    LayerProgress n(key(5), 10, value(), 32, 1000); open(n, 10);
    const auto j = n.admit(value(1)), k = n.admit(value(2));
    assert(n.takeWrite(2).id == j && n.takeWrite(2).id == k);
    n.disconnect();
    assert(n.requeueWrite(j, true) && n.requeueWrite(k, true));
    assert(n.takeWriteOutOfOrder(k, 1).id == k);
    n.completeWrite(k, OMNI_USD_LIVE_OK, 12);
    n.confirmCommitted(j);
    read = n.startRead();
    assert(read.sequence == 10);
    assert(n.caughtUp(read.generation, 1, 12).accepted);
    run = n.applicable(4);
    assert(run.size() == 1 && run[0].id == k && run[0].netted && run[0].absorbedLocal == 1);
    n.applied(k);
    assert(n.snapshot().applied == 12 && n.snapshot().retained == 0 && n.snapshot().localRetired == 2);
}

// A connection failure can lose both a chained write's reply and its read. The write is queued again with an
// unknown outcome, and the read stays closed while it is: the write issues to settle, and the read resumes only once
// its re-send is acknowledged, so the catch-up holds the write's commit at the position the reply named.
void uncertainReplay()
{
    const auto ha = proof(value(101)).fingerprint;
    // The re-send is answered ALREADY_COMMITTED at 11, so the read resumes from 11.
    LayerProgress p(key(), 10, value(), 32, 1000); open(p, 10);
    const auto a = p.admit(value(1), {nullptr, nullptr}, 0, &ha);
    assert(p.takeWrite(8).id == a);
    p.disconnect();
    assert(p.requeueWrite(a, true));
    error(OMNI_USD_LIVE_BUSY, [&] { p.startRead(); });
    auto state = p.snapshot();
    assert(state.gate == Gate::Recovering && state.uncertain == 1 && !state.readStarted);
    assert(p.takeWrite(1).id == a);
    error(OMNI_USD_LIVE_BUSY, [&] { p.startRead(); });
    p.completeWrite(a, OMNI_USD_LIVE_OK, 11);
    // A write whose outcome is known waits for the read.
    const auto later = p.admit(value(2));
    assert(!p.takeWrite(1).id);
    auto read = p.startRead();
    assert(read.sequence == 11);
    const auto peer = p.acceptCoverage(read.generation, 1, 12, 12, proof(value(12)));
    assert(p.recoveryReady(read.generation, 1, 12).accepted && p.snapshot().covered == 12);
    assert(p.takeWrite(1).id == later);
    p.completeWrite(later, OMNI_USD_LIVE_OK, 13);
    p.applied(a); p.applied(peer); p.applied(later);
    assert(p.snapshot().applied == 13 && p.snapshot().retained == 0 && p.snapshot().localRetired == 2);
    // A peer committed 11 while the connection was down, so the read resumes from 10 and its catch-up repeats the
    // write at the position its re-send was acknowledged at: a replay.
    LayerProgress q(key(2), 10, value(), 32, 1000); open(q, 10);
    const auto qa = q.admit(value(1), {nullptr, nullptr}, 0, &ha);
    assert(q.takeWrite(8).id == qa);
    q.disconnect();
    assert(q.requeueWrite(qa, true));
    assert(q.takeWrite(1).id == qa);
    q.completeWrite(qa, OMNI_USD_LIVE_OK, 12);
    read = q.startRead();
    assert(read.sequence == 10);
    const auto qpeer = q.acceptCoverage(read.generation, 1, 11, 11, proof(value(11)));
    assert(!q.acceptCoverage(read.generation, 1, 12, 12, proof(value(101))));
    assert(q.recoveryReady(read.generation, 1, 12).accepted && q.snapshot().covered == 12);
    q.applied(qpeer); q.applied(qa);
    assert(q.snapshot().applied == 12 && q.snapshot().retained == 0 && q.snapshot().localRetired == 1);
    // A refusal of the re-send leaves the outcome unknown: an attempt the service handed to its store may still
    // commit. The read stays closed until a re-send is acknowledged.
    LayerProgress n(key(3), 10, value(), 32, 1000); open(n, 10);
    const auto na = n.admit(value(1), {nullptr, nullptr}, 0, &ha), nb = n.admit(value(2));
    assert(n.takeWrite(8).id == na && n.takeWrite(8).id == nb);
    n.disconnect();
    assert(n.requeueWrite(na, true) && n.requeueWrite(nb, true));
    assert(n.takeWriteOutOfOrder(nb, 1).id == nb);
    assert(n.requeueWrite(nb, false) && n.snapshot().uncertain == 2);
    error(OMNI_USD_LIVE_BUSY, [&] { n.startRead(); });
    assert(n.takeWrite(1).id == na);
    n.completeWrite(na, OMNI_USD_LIVE_OK, 11);
    assert(n.takeWrite(1).id == nb);
    n.completeWrite(nb, OMNI_USD_LIVE_OK, 12);
    assert(n.startRead().sequence == 12);
    // A write after one whose every attempt applied nothing applied nothing either; once cleared, it holds no read.
    LayerProgress c(key(4), 10, value(), 32, 1000); open(c, 10);
    const auto ca = c.admit(value(1)), cb = c.admit(value(2));
    assert(c.takeWrite(8).id == ca && c.takeWrite(8).id == cb);
    c.disconnect();
    assert(c.requeueWrite(ca, false) && c.requeueWrite(cb, true));
    c.clearUncertain(cb);
    assert(!c.snapshot().uncertain && c.startRead().sequence == 10);
}

// The reconnect scenario matrix: every ordering of a lost reply, a lost read and a resumed read's catch-up that
// review found, run against one controller and driven as the layer actor drives it (design sections 12 and 13). A
// scripted service commits this writer's chain in order, answers each no_wait re-send, and delivers each read: live
// it carries peers' commits and header-only frames for empty commits, never this writer's; a catch-up repeats every
// commit after the cursor, as single updates, one aggregate, or, when the commits net to nothing, a header-only
// frame. A case passes when the layer ends as expected and, for a layer that stays READY, the applied history
// reproduces the service's: every commit is applied exactly once, as the update holding it or as a local intent with
// the same bytes, nothing a net-empty catch-up passed applies, and every local write retires.
namespace matrix {
enum class Delivery { Singletons, Aggregate, NetEmpty };
struct Commit {
    unsigned writer = 0; // Zero for a peer; otherwise the local write's ordinal.
    uint64_t bytes = 0;
};
struct Reply {
    bool ok = false;
    uint64_t position = 0;
    unsigned named = 0; // A refusal: the ordinal of the writer's latest committed write, zero for none.
};
struct Service {
    uint64_t latest = 10;
    unsigned last = 0;
    std::map<uint64_t, Commit> commits; // A position without an entry is an empty commit.
    std::map<unsigned, uint64_t> own;
    void peer(uint64_t bytes) { commits[++latest] = {0, bytes}; }
    void empty() { ++latest; }
    // An attempt of local write `ordinal` commits only right after its predecessor.
    bool commit(unsigned ordinal)
    {
        if (own.count(ordinal) || last + 1 != ordinal) return false;
        commits[++latest] = {ordinal, 100 + ordinal};
        own[ordinal] = latest;
        last = ordinal;
        return true;
    }
    // A no_wait re-send commits, answers ALREADY_COMMITTED while the write is the writer's latest, or is refused
    // naming the latest.
    Reply resend(unsigned ordinal)
    {
        if (last == ordinal || commit(ordinal)) return {true, own.at(ordinal), 0};
        return {false, 0, last};
    }
};
uint64_t bytesOf(const LayerProgress::ContentHash& hash)
{
    uint64_t number = 0;
    for (unsigned i = 0; i < 8; ++i) number |= uint64_t(hash[i]) << (8 * i);
    return number;
}
struct Applied {
    uint64_t first, last, bytes;
    bool local, aggregate, netted;
    std::vector<unsigned> absorbed; // Local writes an aggregate holds.
};
struct Harness {
    LayerProgress p{key(), 10, value(), 64, 1000};
    Service service;
    Delivery delivery;
    std::vector<uint64_t> ids; // By local ordinal - 1.
    uint64_t generation = 1;
    bool reading = true, catchingUp = false;
    std::vector<uint64_t> pending; // Live commits while the read catches up.
    std::vector<std::pair<uint64_t, uint64_t>> quiet; // Ranges a header-only catch-up covered.
    unsigned proven = 0, named = 0; // Writes below `proven` committed; `named` awaits its position.
    bool located = false;
    std::function<void()> afterReadStart;
    std::function<void(unsigned)> afterCommit; // After a re-send's commit step, before its reply.
    std::function<void(unsigned, Reply&)> lie; // Replaces the service's answer.

    Harness(unsigned writes, Delivery mode) : delivery(mode)
    {
        open(p, 10);
        for (unsigned ordinal = 1; ordinal <= writes; ++ordinal) {
            const auto hash = proof(value(100 + ordinal)).fingerprint;
            ids.push_back(p.admit(value(100 + ordinal), {nullptr, nullptr}, 0, &hash));
        }
        for (const auto id : ids) assert(p.takeWrite(8).id == id); // Every floor is 10.
    }
    uint64_t id(unsigned ordinal) const { return ids.at(ordinal - 1); }
    void deliver(uint64_t position)
    {
        const auto found = service.commits.find(position);
        if (found == service.commits.end()) p.caughtUp(generation, 1, position);
        else if (!found->second.writer)
            p.acceptCoverage(generation, 1, position, position, proof(value(found->second.bytes)));
    }
    void live()
    {
        if (!reading) return;
        if (catchingUp) pending.push_back(service.latest);
        else deliver(service.latest);
    }
    void peer(uint64_t bytes) { service.peer(bytes); live(); }
    void empty() { service.empty(); live(); }
    // An attempt of the write commits, if its predecessor is the writer's latest; the read filters it.
    void commit(unsigned ordinal)
    {
        if (service.commit(ordinal)) live();
    }
    void reply(unsigned ordinal)
    {
        p.completeWrite(id(ordinal), OMNI_USD_LIVE_OK, service.own.at(ordinal));
        proven = std::max(proven, ordinal);
    }
    void dropRead()
    {
        generation = p.disconnect();
        reading = false;
    }
    // A connection failure ends the read and loses the replies of every write in flight.
    void drop()
    {
        dropRead();
        for (const auto& work : p.retained())
            if (work.localOrdinal && work.disposition == State::Issued) assert(p.requeueWrite(work.id, true));
    }
    // The chain committed every write below `proven`.
    bool locate()
    {
        bool confirmed = false;
        for (const auto& work : p.retained())
            if (work.localOrdinal && work.disposition == State::Queued && work.attempts && work.localOrdinal < proven &&
                p.snapshot().gate != Gate::Failed) {
                p.confirmCommitted(work.id);
                confirmed = true;
            }
        return confirmed;
    }
    // One step of the writer's recovery, as the actor's chain resolution takes it: a write the service named is
    // re-sent for its position; until a reply locates the chain, the latest write of unknown outcome is; then the
    // writes are re-sent in order. False when there is nothing to do or the controller issues nothing.
    bool resend()
    {
        if (p.snapshot().gate == Gate::Failed) return false;
        const bool confirmed = locate();
        uint64_t first = 0, target = 0;
        for (const auto& work : p.retained()) {
            if (!work.localOrdinal || work.disposition != State::Queued) continue;
            if (!first) first = work.id;
            if (work.localOrdinal == named || (!named && !located && work.mayHaveCommitted)) target = work.id;
        }
        if (!target) target = first;
        if (!target) return confirmed;
        const auto taken = target == first ? p.takeWrite(1) : p.takeWriteOutOfOrder(target, 1);
        if (!taken.id) return confirmed;
        const auto ordinal = unsigned(taken.localOrdinal);
        auto answer = service.resend(ordinal);
        if (lie) lie(ordinal, answer);
        if (afterCommit) afterCommit(ordinal);
        located = true;
        if (answer.ok) {
            p.completeWrite(taken.id, OMNI_USD_LIVE_OK, answer.position);
            proven = std::max(proven, ordinal);
            if (named == ordinal) named = 0;
        } else {
            p.requeueWrite(taken.id, false);
            named = answer.named;
            proven = std::max(proven, answer.named);
        }
        return true;
    }
    void settle()
    {
        for (unsigned steps = 0; steps < 32 && resend(); ++steps) {}
    }
    void catchUp(uint64_t cursor, uint64_t through)
    {
        if (through <= cursor) {
            p.caughtUp(generation, 1, cursor);
            return;
        }
        if (delivery == Delivery::NetEmpty) {
            quiet.emplace_back(cursor + 1, through);
            p.caughtUp(generation, 1, through);
        } else if (delivery == Delivery::Aggregate) {
            p.acceptCoverage(generation, 1, cursor + 1, through,
                proof(value(1000000 + cursor * 1000 + through), LayerProgress::CoverageKind::Aggregate));
            p.recoveryReady(generation, 1, through);
        } else {
            for (auto position = cursor + 1; position <= through; ++position) {
                const auto found = service.commits.find(position);
                if (found == service.commits.end()) p.caughtUp(generation, 1, position);
                else p.acceptCoverage(generation, 1, position, position, proof(value(found->second.bytes)));
            }
            if (service.commits.count(through)) p.recoveryReady(generation, 1, through);
        }
    }
    // Resumes the read once no write is in flight, as the actor does; while the controller refuses to read, the
    // writes it holds are settled first. The resumed read's catch-up holds the commits at subscription.
    void recover()
    {
        locate();
        LayerProgress::ReadCursor cursor{};
        for (unsigned steps = 0;; ++steps) {
            try {
                cursor = p.startRead();
                break;
            } catch (OmniUsdLiveResult result) {
                if (result != OMNI_USD_LIVE_BUSY || steps >= 32 || !resend()) throw;
            }
        }
        generation = cursor.generation;
        reading = catchingUp = true;
        const auto through = service.latest;
        if (afterReadStart) afterReadStart();
        catchUp(cursor.sequence, through);
        catchingUp = false;
        for (const auto position : std::exchange(pending, {})) deliver(position);
        settle();
        resubscribe();
    }
    // The actor resubscribes from the coverage frontier when a write the chain proved committed while the read was
    // open waits for a read started after the proof, and no write is in flight or of unknown outcome.
    void resubscribe()
    {
        const auto state = p.snapshot();
        if (state.awaitingRead && !state.uncertain && !state.issued && state.gate == Gate::Ready) {
            dropRead();
            recover();
        }
    }
    // Applies every ready work in order, recording what each application applies.
    std::vector<Applied> applyAll()
    {
        std::vector<Applied> trace;
        for (auto run = p.applicable(1); !run.empty(); run = p.applicable(1)) {
            const auto& work = run.front();
            Applied entry{work.first, work.last, bytesOf(work.fingerprint), work.disposition != State::Peer,
                work.kind == LayerProgress::CoverageKind::Aggregate, work.netted, {}};
            if (entry.local) entry.bytes = 100 + work.localOrdinal;
            for (const auto& other : p.retained())
                if (other.absorbedBy == work.id && other.localOrdinal)
                    entry.absorbed.push_back(unsigned(other.localOrdinal));
            trace.push_back(std::move(entry));
            p.applied(work.id);
        }
        return trace;
    }
    // Empty when the applied history reproduces the service's; otherwise the first discrepancy.
    std::string reproduces(unsigned writes)
    {
        const auto trace = applyAll();
        const auto state = p.snapshot();
        if (state.applied != service.latest || state.retained || state.localRetired != writes)
            return "applied " + std::to_string(state.applied) + " of " + std::to_string(service.latest) + ", retained " +
                std::to_string(state.retained) + ", retired " + std::to_string(state.localRetired);
        std::map<uint64_t, unsigned> covered;
        const auto inQuiet = [&](uint64_t first, uint64_t last) {
            return std::any_of(quiet.begin(), quiet.end(),
                [&](const auto& range) { return range.first <= first && last <= range.second; });
        };
        for (const auto& entry : trace) {
            const std::string kind = entry.local ? "local" : "peer";
            if (entry.netted) {
                if (!inQuiet(entry.first, entry.last))
                    return kind + " work at " + std::to_string(entry.first) +
                        " retired without applying outside a net-empty catch-up";
                continue;
            }
            // A net-empty catch-up header nets every position it passes, so nothing there applies.
            for (auto position = entry.first; position <= entry.last; ++position)
                if (inQuiet(position, position))
                    return kind + " application at " + std::to_string(position) + " inside a net-empty catch-up";
            for (auto position = entry.first; position <= entry.last; ++position) ++covered[position];
            if (entry.aggregate) {
                for (const auto ordinal : entry.absorbed)
                    if (service.own.at(ordinal) < entry.first || service.own.at(ordinal) > entry.last)
                        return "write " + std::to_string(ordinal) + " retired with an aggregate not holding it";
                continue;
            }
            const auto found = service.commits.find(entry.first);
            if (found == service.commits.end() || found->second.bytes != entry.bytes)
                return std::string(entry.local ? "local" : "peer") + " application at " + std::to_string(entry.first) +
                    " is not the commit there";
        }
        for (const auto& [first, last] : quiet)
            for (auto position = first; position <= last; ++position) ++covered[position];
        for (auto position = uint64_t(11); position <= service.latest; ++position)
            if (covered[position] > 1 || (service.commits.count(position) && covered[position] != 1))
                return "position " + std::to_string(position) + " applied " + std::to_string(covered[position]) +
                    " times";
        return {};
    }
};
struct Case {
    const char* name;
    unsigned writes;
    Delivery delivery;
    OmniUsdLiveResult expected; // OK: the layer stays READY and reproduces the service's history.
    std::function<void(Harness&)> script;
};
void run()
{
    const auto retainedPeers = [](Harness& h) {
        h.commit(1);
        h.peer(12);
        h.peer(13);
        h.commit(2);
        h.reply(2);
        assert(h.p.requeueWrite(h.id(1), true));
        h.settle();
        h.resubscribe();
    };
    const std::vector<Case> cases{
        // A write commits; the connection then loses its reply and the read.
        {"lost reply then read loss", 1, Delivery::Singletons, OMNI_USD_LIVE_OK,
            [](Harness& h) {
        h.commit(1);
        h.drop();
        h.recover();
    }},
        // A write's attempt commits only after the resumed read opened, so that read filters it.
        {"stale attempt commits after the resumed read", 1, Delivery::Singletons, OMNI_USD_LIVE_OK,
            [](Harness& h) {
        h.drop();
        h.afterReadStart = [&h] {
            h.commit(1);
            h.peer(12);
        };
        h.recover();
    }},
        // A commits before the reconnect and is repeated by catch-up; B's stale attempt commits after the resumed
        // read opens; C's probe commits, and a peer update reaches the reader before C's reply.
        {"A replayed, B filtered, C probe with a peer update first", 3, Delivery::Singletons, OMNI_USD_LIVE_OK,
            [](Harness& h) {
        h.commit(1);
        h.drop();
        h.afterReadStart = [&h] { h.commit(2); };
        h.afterCommit = [&h](unsigned ordinal) {
            if (ordinal == 3) h.peer(14);
        };
        h.recover();
    }},
        // A peer commits 11 and the write 12 while the connection is down; the catch-up is the aggregate 11..12.
        {"aggregate containing the lost write", 1, Delivery::Aggregate, OMNI_USD_LIVE_OK,
            [](Harness& h) {
        h.drop();
        h.peer(11);
        h.commit(1);
        h.recover();
    }},
        // A is acknowledged at 12 and C at 14; B's reply is lost and the chain proves it committed; the catch-up is
        // the aggregate 11..14.
        {"aggregate 11..14 with a chain-proven write between two acknowledged ones", 3, Delivery::Aggregate,
            OMNI_USD_LIVE_OK, [](Harness& h) {
        h.dropRead();
        h.peer(11);
        h.commit(1);
        h.reply(1);
        h.commit(2);
        assert(h.p.requeueWrite(h.id(2), true));
        h.commit(3);
        h.reply(3);
        h.recover();
    }},
        // An edit and its undo commit; the connection loses both replies and the read; the catch-up nets to nothing,
        // so the service sends only the caught-up header above the cursor.
        {"net-empty catch-up header above the cursor", 2, Delivery::NetEmpty, OMNI_USD_LIVE_OK,
            [](Harness& h) {
        h.commit(1);
        h.commit(2);
        h.drop();
        h.recover();
    }},
        // A commits before the reconnect, and B's stale attempt would commit after the read opens; the chain settles
        // first, so A, B and C are acknowledged at 11, 12 and 13 before the read resumes from 13.
        {"placement resubscription after an uncertain catch-up", 3, Delivery::Aggregate, OMNI_USD_LIVE_OK,
            [](Harness& h) {
        h.commit(1);
        h.drop();
        h.afterReadStart = [&h] { h.commit(2); };
        h.recover();
    }},
        // With the read open, A commits at 11 and a peer's 12 is delivered; a peer's 13 is still on its way when B
        // is acknowledged at 14 and A's reply is lost. The chain proves A committed, so the read is resubscribed from
        // the frontier, and its catch-up aggregate 11..14 holds A's commit and the retained peer update at 12.
        {"placement resubscription over retained peer coverage", 2, Delivery::Aggregate, OMNI_USD_LIVE_OK,
            [](Harness& h) {
        h.commit(1);
        h.peer(12);
        h.service.peer(13);
        h.commit(2);
        h.reply(2);
        assert(h.p.requeueWrite(h.id(1), true));
        h.settle();
        h.resubscribe();
    }},
        // While the write's outcome is unknown, a peer commits identical field assignments at 11; the write's
        // re-send commits at 12.
        {"peer commits identical bytes while the write is uncertain", 1, Delivery::Singletons, OMNI_USD_LIVE_OK,
            [](Harness& h) {
        h.drop();
        h.peer(101);
        h.recover();
    }},
        // Both replies and the read are lost; A then commits at 11, 12 is an empty commit and B commits at 13, and
        // the catch-up delivers A's update, a header-only frame and B's update.
        {"empty position between replayed writes", 2, Delivery::Singletons, OMNI_USD_LIVE_OK,
            [](Harness& h) {
        h.drop();
        h.commit(1);
        h.empty();
        h.commit(2);
        h.recover();
    }},
        // With the read open, A commits at 11, a header-only frame announces the empty commit 12 and B commits at
        // 13; both replies and the read are then lost.
        {"empty commit announced live between uncertain writes", 2, Delivery::Singletons, OMNI_USD_LIVE_OK,
            [](Harness& h) {
        h.commit(1);
        h.empty();
        h.commit(2);
        h.drop();
        h.recover();
    }},
        // A peer's update arrives on the open read at 11; the uncertain write's re-send is then acknowledged at 11.
        {"contradictory acknowledgment", 1, Delivery::Singletons, OMNI_USD_LIVE_OUTCOME_UNKNOWN,
            [](Harness& h) {
        assert(h.p.requeueWrite(h.id(1), true));
        h.peer(11);
        h.lie = [](unsigned, Reply& answer) { answer = {true, 11, 0}; };
        h.settle();
    }},
        // A peer commits A's bytes at 11 before A commits at 12 and B at 13; C's probe proves both. The catch-up
        // applies 11, 12 and 13 as delivered, in service order, and A and B retire with C.
        {"chain-proven run around a peer's identical edit", 3, Delivery::Singletons, OMNI_USD_LIVE_OK,
            [](Harness& h) {
        h.drop();
        h.peer(101);
        h.commit(1);
        h.commit(2);
        h.recover();
    }},
        // A peer commits B's bytes at 11 before A commits at 12 and B at 13; every reply is lost, and C's probe
        // commits at 14 and proves A and B. The catch-up applies 11 as the peer's update and A and B at their own
        // positions, in service order.
        {"peer commits a later write's bytes ahead of the chain", 3, Delivery::Singletons, OMNI_USD_LIVE_OK,
            [](Harness& h) {
        h.drop();
        h.peer(102);
        h.commit(1);
        h.commit(2);
        h.recover();
    }},
        // With the read open, a peer commits A's bytes at 11 and another update at 12; A commits at 13 and B at 14.
        // B is acknowledged and A's reply is lost; the chain proves A committed before the read delivers 11 and 12.
        // The open read filtered A's commit, so the read is resubscribed, and its catch-up applies A's commit at 13
        // as delivered.
        {"peer commits identical bytes on a read that stays open", 2, Delivery::Singletons, OMNI_USD_LIVE_OK,
            [](Harness& h) {
        h.service.peer(101);
        h.service.peer(12);
        h.commit(1);
        h.commit(2);
        h.reply(2);
        assert(h.p.requeueWrite(h.id(1), true));
        h.settle();
        h.deliver(11);
        h.deliver(12);
        h.resubscribe();
        h.peer(15);
    }},
        // Both replies and the read are lost; A commits at 11, a peer undoes A's edit at 12 and B commits at 13. B's
        // probe proves A, and the catch-up applies A's commit and the peer's undo in order, so the peer's value wins.
        {"peer undoes a chain-proven write's edit before the catch-up", 2, Delivery::Singletons, OMNI_USD_LIVE_OK,
            [](Harness& h) {
        h.drop();
        h.commit(1);
        h.peer(201);
        h.commit(2);
        h.recover();
    }},
        // As above, with B at 12 and the peer undoing both at 13: the catch-up nets to nothing, so B retires without
        // applying and A retires with it.
        {"peer undoes both writes before a net-empty catch-up", 2, Delivery::NetEmpty, OMNI_USD_LIVE_OK,
            [](Harness& h) {
        h.drop();
        h.commit(1);
        h.commit(2);
        h.peer(301);
        h.recover();
    }},
        // The reply and the read are lost; a peer commits at 11 and the write at 12, and the write's re-send is
        // acknowledged at 12. The read resumes from 10 and its catch-up nets to nothing, so the write retires
        // without applying.
        {"acknowledged write inside a net-empty catch-up", 1, Delivery::NetEmpty, OMNI_USD_LIVE_OK,
            [](Harness& h) {
        h.drop();
        h.peer(11);
        h.commit(1);
        h.recover();
    }},
        // With the read open, A commits at 11 and peers' 12 and 13 are delivered, retained behind A's position; B
        // commits at 14 and is acknowledged, and A's reply is lost. The chain proves A committed, so the read is
        // resubscribed from 10. The commits 11..14 net to nothing (B restores the field 12 set and deletes the prim 13
        // created), so the catch-up is the header at 14: the retained peer updates and B retire without applying.
        {"net-empty catch-up over retained peer coverage", 2, Delivery::NetEmpty, OMNI_USD_LIVE_OK, retainedPeers},
        // The same ordering with a catch-up of single updates, which repeat A's commit and the retained peer updates.
        {"singleton catch-up over retained peer coverage", 2, Delivery::Singletons, OMNI_USD_LIVE_OK, retainedPeers},
    };
    unsigned failures = 0;
    for (const auto& item : cases) {
        Harness harness(item.writes, item.delivery);
        std::string outcome;
        try {
            item.script(harness);
        } catch (OmniUsdLiveResult result) {
            outcome = "threw result " + std::to_string(result);
        } catch (const std::exception& error) {
            outcome = std::string("threw ") + error.what();
        }
        // A failed layer issues and reads nothing more, so its result is the outcome however the script ended.
        const auto state = harness.p.snapshot();
        if (state.gate == Gate::Failed) outcome = "failed with result " + std::to_string(state.result);
        else if (outcome.empty() && state.gate != Gate::Ready) outcome = "not ready";
        if (outcome.empty()) {
            try {
                outcome = harness.reproduces(item.writes);
            } catch (const std::exception& error) {
                outcome = std::string("application threw ") + error.what();
            }
            if (outcome.empty()) outcome = "ready, history reproduced";
            else outcome = "ready, " + outcome;
        }
        const auto expected = item.expected == OMNI_USD_LIVE_OK ? std::string("ready, history reproduced") :
                                                                  "failed with result " + std::to_string(item.expected);
        const bool pass = outcome == expected;
        failures += !pass;
        std::cout << "reconnect matrix: " << (pass ? "pass" : "FAIL") << ": " << item.name << ": " << outcome
                  << (pass ? "" : " (expected " + expected + ")") << std::endl;
    }
    assert(!failures);
}
} // namespace matrix
}

void testLayerProgress()
{
    auto unknown = key(); unknown.object = 0;
    LayerProgress discovery(unknown, 0, {}, 10, 100, true);
    error(OMNI_USD_LIVE_BUSY, [&] { discovery.admit(value()); });
    auto cursor = discovery.startRead();
    assert(!cursor.object && !cursor.sequence && !discovery.objectId());
    auto initial = proof(value(99), LayerProgress::CoverageKind::Snapshot);
    allocationCountdown = 0;
    bool exhausted = false;
    try { discovery.acceptCoverage(cursor.generation, 17, 1, 1234, initial); }
    catch (const std::bad_alloc&) { exhausted = true; }
    allocationCountdown = std::numeric_limits<size_t>::max();
    assert(exhausted && !discovery.objectId() && !discovery.snapshot().retained);
    const auto id = discovery.acceptCoverage(cursor.generation, 17, 1, 1234, initial);
    assert(discovery.objectId() == 17 && discovery.snapshot().object == 17);
    assert(discovery.retained().front().localOrdinal==0 && discovery.snapshot().localAdmitted==0 &&
           discovery.snapshot().localRetired==0 && discovery.snapshot().initialApplied==0);
    assert(discovery.recoveryReady(cursor.generation, 17, 1234).accepted);
    discovery.applied(id);
    discovery.disconnect(); cursor = discovery.startRead();
    assert(cursor.object == 17 && cursor.sequence == 1234);
    assert(discovery.snapshot().initialApplied==0 && discovery.snapshot().localAdmitted==0 && discovery.snapshot().localRetired==0);
    error(OMNI_USD_LIVE_OBJECT_REPLACED, [&] { discovery.caughtUp(cursor.generation, 18, 1234); });
    assert(discovery.objectId() == 17 && discovery.snapshot().applied == 1234);
    LayerProgress empty(unknown, 0, {}, 10, 100, true);
    cursor = empty.startRead();
    error(OMNI_USD_LIVE_MALFORMED_STREAM, [&] { empty.caughtUp(cursor.generation, 7, 1); });
    assert(!empty.objectId());
    LayerProgress gap(unknown, 0, {}, 10, 100, true);
    cursor = gap.startRead();
    error(OMNI_USD_LIVE_MALFORMED_STREAM, [&] { gap.acceptCoverage(cursor.generation, 7, 2, 2, proof(value())); });
    assert(!gap.objectId());
    LayerProgress stale(unknown, 0, {}, 10, 100, true);
    cursor = stale.startRead(); stale.disconnect();
    assert(!stale.acceptCoverage(cursor.generation, 7, 1, 1, proof(value())) && !stale.objectId());
    cursor = stale.startRead();
    assert(stale.acceptCoverage(cursor.generation, 8, 1, 1, proof(value())));
    assert(stale.objectId() == 8);
    retainedStatusAccounting();
    localAdmissionHistory();
    publicationFailuresAndRetirement();
    publicationPrecedesIssuance();
    actorPublicationAndClose();
    peerHoleAndRetention();
    gateIsolationAndUnknown();
    mutationRejectionsAndReissue();
    operationIdResend();
    nothingAppliedReissue();
    authWaitFillsHoles();
    issuanceStopKeepsReading();
    malformedAndBounds();
    emptyPositions();
    applicableRun();
    snapshotBootstrap();
    acknowledgementOrder();
    concurrentAdmissionAndReentrantRetirement();
    singleWriteAndOverlap();
    aggregateAbsorbsAcknowledgedIntents();
    aggregateStartingOnAcknowledgmentStops();
    aggregateBeforeAcknowledgment();
    aggregateBetweenWindowedAcknowledgments();
    aggregateHoldBounds();
    aggregateInterleavings();
    generatedCoverageHistories();
    windowedWrites();
    windowedHoles();
    requeuedWrites();
    committedRelease();
    uncertainReplay();
    matrix::run();
}
