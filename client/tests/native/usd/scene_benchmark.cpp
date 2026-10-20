// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "SdfSceneData.h"
#include <pxr/usd/sdf/data.h>
#include <pxr/usd/sdf/schema.h>
#include <pxr/base/tf/errorMark.h>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <thread>
#include <chrono>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(_WIN32)
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    include <windows.h>
#    include <psapi.h>
#else
#    include <sys/resource.h>
#endif

using namespace PXR_NS;
using omni::usd::live::SdfSceneData;
namespace {
using Clock = std::chrono::steady_clock;
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
uint64_t ns(Clock::duration value) { return std::chrono::duration_cast<std::chrono::nanoseconds>(value).count(); }
#if defined(_WIN32)
uint64_t cpuNs() {
    FILETIME created, exited, kernel, user;
    require(GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user) != 0, "GetProcessTimes failed");
    const auto ticks = [](const FILETIME& time) { return (uint64_t(time.dwHighDateTime) << 32) | time.dwLowDateTime; };
    return (ticks(kernel) + ticks(user)) * 100; // FILETIME counts 100 ns intervals.
}
uint64_t peakRssKib() {
    PROCESS_MEMORY_COUNTERS counters{}; counters.cb = sizeof(counters);
    require(K32GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters)) != 0, "GetProcessMemoryInfo failed");
    return uint64_t(counters.PeakWorkingSetSize) / 1024;
}
#else
uint64_t cpuNs() {
    rusage usage{}; require(getrusage(RUSAGE_SELF, &usage) == 0, "getrusage failed");
    return (uint64_t(usage.ru_utime.tv_sec) + usage.ru_stime.tv_sec) * 1000000000 +
           (uint64_t(usage.ru_utime.tv_usec) + usage.ru_stime.tv_usec) * 1000;
}
uint64_t peakRssKib() {
    rusage usage{}; require(getrusage(RUSAGE_SELF, &usage) == 0, "getrusage failed");
    return uint64_t(usage.ru_maxrss); // Linux reports kibibytes.
}
#endif
using Record = std::pair<SdfSpecType, std::map<TfToken, VtValue>>;
using State = std::map<SdfPath, Record>;
State snapshot(const SdfAbstractData& data) {
    struct Visitor : SdfAbstractDataSpecVisitor {
        State state;
        bool VisitSpec(const SdfAbstractData& data, const SdfPath& path) override {
            Record record{data.GetSpecType(path), {}};
            for (const auto& key : data.List(path)) record.second.emplace(key, data.Get(path, key));
            require(state.emplace(path, std::move(record)).second, "duplicate spec in traversal");
            return true;
        }
        void Done(const SdfAbstractData&) override {}
    } visitor;
    data.VisitSpecs(&visitor); return visitor.state;
}
constexpr int pairMask = 0x5a5a5a;
const TfToken revisionKey("bench_revision"), payloadKey("bench_payload");
struct Op { size_t index; VtValue value; double time; int ordinal; };
uint64_t apply(SdfAbstractData& data, const std::string& kind, const Op& op,
               const std::vector<SdfPath>& paths, const std::vector<SdfPath>& moved) {
    static const TfToken key("bench_value");
    const auto& path = paths[op.index];
    if (kind == "field_read") {
        const auto value = data.Get(path, key);
        require(value.IsHolding<int>(), "field read lost its value");
        return static_cast<uint64_t>(value.UncheckedGet<int>());
    }
    if (kind == "mixed_pair") {
        data.Set(path, revisionKey, VtValue(op.ordinal));
        data.Set(path, payloadKey, VtValue(op.ordinal ^ pairMask));
    } else if (kind == "time_sample") data.SetTimeSample(path, op.time, op.value);
    else if (kind == "move_pair") {
        data.MoveSpec(path, moved[op.index]);
        require(!data.HasSpec(path) && data.GetSpecType(moved[op.index]) == SdfSpecTypeAttribute,
                "move did not complete");
        data.MoveSpec(moved[op.index], path);
        require(!data.HasSpec(moved[op.index]) && data.GetSpecType(path) == SdfSpecTypeAttribute,
                "return move did not complete");
    } else data.Set(path, key, op.value);
    return 0;
}
uint64_t percentile(const std::vector<uint64_t>& sorted, size_t numerator) {
    return sorted[(sorted.size() * numerator + 99) / 100 - 1];
}
struct ReaderMeasure {
    std::vector<uint64_t> latency;
    uint64_t elapsed = 0, freeWhilePending = 0, maxRevision = 0;
};
struct MixedMeasure {
    uint64_t elapsed = 0, cpu = 0;
    std::vector<ReaderMeasure> readers;
};
MixedMeasure mixed(SdfSceneData& data, const std::vector<Op>& edits,
                   const std::vector<SdfPath>& paths, const std::vector<SdfPath>& moved,
                   size_t count, size_t readerCount, std::vector<uint64_t>& writerLatency) {
    constexpr size_t epochs = 4;
    require(count >= epochs && count % epochs == 0, "mixed work must divide into four epochs");
    struct Gate {
        std::mutex mutex; std::condition_variable changed;
        size_t ready = 0, epoch = 0, witnesses = 0, finished = 0;
        enum Stage { Parked, Witness, Free } stage = Parked;
        std::exception_ptr failure;
        void fail(std::exception_ptr error) {
            std::lock_guard<std::mutex> lock(mutex);
            if (!failure) failure = error;
            changed.notify_all();
        }
    } gate;
    std::atomic<size_t> published{0};
    MixedMeasure result; result.readers.resize(readerCount);
    for (auto& reader : result.readers) reader.latency.resize(count);
    std::vector<std::thread> threads; threads.reserve(readerCount);
    auto wait = [&](auto predicate) {
        std::unique_lock<std::mutex> lock(gate.mutex);
        gate.changed.wait(lock, [&] { return gate.failure || predicate(); });
        if (gate.failure) std::rethrow_exception(gate.failure);
    };
    const size_t batch = count / epochs;
    try {
        for (size_t id = 0; id < readerCount; ++id) threads.emplace_back([&, id] {
            try {
                TfErrorMark errors;
                auto& measurement = result.readers[id];
                std::vector<int> last(paths.size());
                auto read = [&](size_t ordinal, size_t pathIndex, int witness, size_t end) {
                    const auto before = Clock::now();
                    {
                        auto guard = data.AcquireReadLock();
                        const auto revision = data.Get(paths[pathIndex], revisionKey);
                        const auto payload = data.Get(paths[pathIndex], payloadKey);
                        require(revision.IsHolding<int>() && payload.IsHolding<int>(), "mixed read lost pair fields");
                        const int observed = revision.UncheckedGet<int>();
                        require(observed >= last[pathIndex] && observed <= int(count), "mixed revision moved backwards/outside history");
                        require(payload.UncheckedGet<int>() == (observed ^ pairMask), "mixed read observed a torn pair");
                        if (witness) require(observed == witness, "mixed epoch witness missed exact revision");
                        else if (published.load(std::memory_order_acquire) < end) ++measurement.freeWhilePending;
                        last[pathIndex] = observed;
                        measurement.maxRevision = std::max(measurement.maxRevision, uint64_t(observed));
                    }
                    measurement.latency[ordinal] = ns(Clock::now() - before);
                };
                {
                    std::lock_guard<std::mutex> lock(gate.mutex); ++gate.ready; gate.changed.notify_all();
                }
                Clock::time_point started;
                for (size_t epoch = 0; epoch < epochs; ++epoch) {
                    const size_t begin = epoch * batch, end = begin + batch;
                    wait([&] { return gate.epoch == epoch && gate.stage == Gate::Witness; });
                    if (!epoch) started = Clock::now();
                    read(begin, edits[begin].index, int(begin + 1), end);
                    { std::lock_guard<std::mutex> lock(gate.mutex); ++gate.witnesses; gate.changed.notify_all(); }
                    wait([&] { return gate.epoch == epoch && gate.stage == Gate::Free; });
                    for (size_t i = begin + 1; i < end; ++i)
                        read(i, edits[(i + id * 31) % count].index, 0, end);
                    require(errors.IsClean(), "USD error in mixed reader");
                    if (epoch + 1 == epochs) measurement.elapsed = ns(Clock::now() - started);
                    { std::lock_guard<std::mutex> lock(gate.mutex); ++gate.finished; gate.changed.notify_all(); }
                }
            } catch (...) { gate.fail(std::current_exception()); }
        });
        wait([&] { return gate.ready == readerCount; });
        const auto cpuStart = cpuNs(); const auto started = Clock::now();
        auto write = [&](size_t i) {
            const auto before = Clock::now();
            {
                auto guard = data.AcquireWriteLock();
                (void)apply(data, "mixed_pair", edits[i], paths, moved);
                // Read while holding the same data guard: this counter denotes
                // completed transactions, never measured mutex contention.
                published.store(i + 1, std::memory_order_release);
            }
            writerLatency[i] = ns(Clock::now() - before);
        };
        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            const size_t begin = epoch * batch, end = begin + batch;
            write(begin);
            {
                std::lock_guard<std::mutex> lock(gate.mutex);
                gate.epoch = epoch; gate.witnesses = gate.finished = 0;
                gate.stage = Gate::Witness; gate.changed.notify_all();
            }
            wait([&] { return gate.witnesses == readerCount; });
            { std::lock_guard<std::mutex> lock(gate.mutex); gate.stage = Gate::Free; gate.changed.notify_all(); }
            for (size_t i = begin + 1; i < end; ++i) write(i);
            wait([&] { return gate.finished == readerCount; });
        }
        result.elapsed = ns(Clock::now() - started); result.cpu = cpuNs() - cpuStart;
    } catch (...) {
        gate.fail(std::current_exception());
        for (auto& thread : threads) thread.join();
        std::rethrow_exception(gate.failure);
    }
    for (auto& thread : threads) thread.join();
    if (gate.failure) std::rethrow_exception(gate.failure);
    for (auto& reader : result.readers) std::sort(reader.latency.begin(), reader.latency.end());
    return result;
}
uint64_t number(const char* text, uint64_t maximum) {
    const std::string input(text); size_t end = 0;
    require(!input.empty() && input.find_first_not_of("0123456789") == std::string::npos, "unsigned integer required");
    const auto value = std::stoull(input, &end);
    require(end == input.size() && value <= maximum, "argument exceeds bounded workload limit");
    return value;
}
}
int main(int argc, char** argv) try {
    require(argc == 8, "usage: driver kind seed specs operations array_elements warmup_operations readers");
    const std::string kind(argv[1]);
    require(kind == "field_set" || kind == "array_set" || kind == "field_read" ||
            kind == "time_sample" || kind == "move_pair" || kind == "mixed_pair", "unknown workload kind");
    const auto seed = number(argv[2], UINT64_MAX), specs = number(argv[3], 100000);
    const auto operations = number(argv[4], 1000000), elements = number(argv[5], 65536);
    const auto warmup = number(argv[6], 1000000), readers = number(argv[7], 8);
    require((kind == "mixed_pair") == (readers != 0), "reader count does not match workload");
    if (readers) require(operations % 4 == 0 && warmup % 4 == 0, "mixed counts require four complete epochs");
    require(specs && operations && warmup && warmup <= operations, "positive bounded work and warmup required");
    require((kind == "array_set") == (elements != 0), "array size does not match workload");
    require(elements == 0 || operations <= 16777216 / elements, "prepared values exceed 64 MiB");
    std::cout << "seed=" << seed << " kind=" << kind << std::endl;
    TfErrorMark errors;
    std::vector<SdfPath> paths, moved;
    for (size_t i = 0; i < specs; ++i) {
        paths.emplace_back("/P.v" + std::to_string(i));
        moved.emplace_back("/P.m" + std::to_string(i));
    }
    auto initial = TfCreateRefPtr(new SdfData);
    initial->CreateSpec(SdfPath::AbsoluteRootPath(), SdfSpecTypePseudoRoot);
    initial->CreateSpec(SdfPath("/P"), SdfSpecTypePrim);
    for (size_t i = 0; i < specs; ++i) {
        initial->CreateSpec(paths[i], SdfSpecTypeAttribute);
        initial->Set(paths[i], TfToken("bench_value"), VtValue(int(i)));
        if (readers) {
            initial->Set(paths[i], revisionKey, VtValue(0));
            initial->Set(paths[i], payloadKey, VtValue(pairMask));
        }
    }
    auto history = [&] {
        std::mt19937_64 random(seed);
        std::vector<Op> edits; edits.reserve(operations);
        for (size_t i = 0; i < operations; ++i) {
            const size_t index = random() % specs;
            VtValue value(int(random() % 1000000));
            if (elements) { VtIntArray values(elements, int(i)); value = VtValue(std::move(values)); }
            edits.push_back({index, std::move(value), double(random() % 32), int(i + 1)});
        }
        return edits;
    };
    const auto edits = history();
    // Every child process warms the implementation before its own trial. The
    // measured candidate starts from the declared initial scene afterwards.
    {
        auto warm = TfCreateRefPtr(new SdfSceneData); warm->CopyFrom(initial);
        if (readers) {
            std::vector<uint64_t> ignored(warmup);
            (void)mixed(*warm, edits, paths, moved, warmup, readers, ignored);
        } else for (size_t i = 0; i < warmup; ++i) (void)apply(*warm, kind, edits[i], paths, moved);
    }
    auto actual = TfCreateRefPtr(new SdfSceneData); actual->CopyFrom(initial);
    std::vector<uint64_t> latency(operations);
    std::vector<uint64_t> timerPairs(1024);
    for (auto& duration : timerPairs) { const auto before = Clock::now(); duration = ns(Clock::now() - before); }
    std::sort(timerPairs.begin(), timerPairs.end());
    uint64_t checksum = 0;
    MixedMeasure measurement;
    if (readers) measurement = mixed(*actual, edits, paths, moved, operations, readers, latency);
    else {
        const auto cpuStart = cpuNs(); const auto started = Clock::now();
        for (size_t i = 0; i < operations; ++i) {
            const auto before = Clock::now();
            checksum += apply(*actual, kind, edits[i], paths, moved);
            latency[i] = ns(Clock::now() - before);
        }
        measurement.elapsed = ns(Clock::now() - started); measurement.cpu = cpuNs() - cpuStart;
    }
    const auto elapsed = measurement.elapsed, cpu = measurement.cpu;
    const uint64_t peakRss = peakRssKib();
    // Stock SdfData is a correctness oracle only, never a speed reference.
    // Verification and oracle work are outside timing.
    auto expected = TfCreateRefPtr(new SdfData); expected->CopyFrom(initial);
    uint64_t expectedChecksum = 0;
    // Regenerate rather than sharing COW input owners with the candidate: an
    // accidental mutation of its original inputs must not poison the oracle.
    const auto referenceEdits = history();
    for (const auto& edit : referenceEdits) expectedChecksum += apply(*expected, kind, edit, paths, moved);
    require(checksum == expectedChecksum && snapshot(*actual) == snapshot(*expected), "stock USD oracle mismatch");
    require(errors.IsClean(), "USD reported an error");
    const auto cleanupStart = Clock::now(); actual = nullptr;
    const auto cleanup = ns(Clock::now() - cleanupStart);
    std::sort(latency.begin(), latency.end());
    std::cout << "{\"schema_version\":1,\"boundary\":\"sdf_storage\",\"kind\":\"" << kind
              << "\",\"seed\":" << seed << ",\"specs\":" << specs << ",\"operations\":" << operations
              << ",\"array_elements\":" << elements << ",\"warmup_operations\":" << warmup
              << ",\"elapsed_ns\":" << elapsed << ",\"cpu_ns\":" << cpu
              << ",\"latency_p50_ns\":" << percentile(latency, 50) << ",\"latency_p95_ns\":" << percentile(latency, 95)
              << ",\"latency_p99_ns\":" << percentile(latency, 99) << ",\"process_peak_rss_before_oracle_kib\":" << peakRss
              << ",\"timer_pair_p50_ns\":" << timerPairs[timerPairs.size() / 2 - 1]
              << ",\"candidate_cleanup_ns\":" << cleanup << ",\"readers\":" << readers
              << ",\"reader_progress\":[";
    for (size_t id = 0; id < measurement.readers.size(); ++id) {
        const auto& reader = measurement.readers[id];
        if (id) std::cout << ',';
        std::cout << "{\"index\":" << id << ",\"operations\":" << operations << ",\"witness_reads\":4"
                  << ",\"free_reads\":" << operations - 4 << ",\"free_reads_while_writer_pending\":" << reader.freeWhilePending
                  << ",\"max_revision\":" << reader.maxRevision << ",\"elapsed_ns\":" << reader.elapsed
                  << ",\"latency_p50_ns\":" << percentile(reader.latency, 50)
                  << ",\"latency_p95_ns\":" << percentile(reader.latency, 95)
                  << ",\"latency_p99_ns\":" << percentile(reader.latency, 99) << '}';
    }
    std::cout << "],\"correctness\":\"passed\"}" << std::endl;
    return 0;
} catch (const std::exception& error) {
    std::cerr << "benchmark failed: " << error.what() << std::endl; return 1;
}
