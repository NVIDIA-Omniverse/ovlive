// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Frozen per-case work budgets. Each case runs production scene code once and
// reads the scene work meter around it: the elements it visits or copies, the
// index comparisons it makes and the atomic updates host storage and its
// projection take. The counts depend only on the source, the seeded workload
// and the toolchain/STL checking mode, never on the machine or its load, so the same budgets
// hold on every runner of a toolchain and a case fails only when the code
// does more work.
#include "footprint_work.h"
#include "SdfSceneData.h"
#include <pxr/usd/sdf/data.h>
#include <pxr/usd/sdf/schema.h>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE
using namespace omni::usd::live;
using namespace omni::usd::live::test::footprint;
namespace {
// The frozen meter counts of each case, measured on the stock-25.11 profile.
// A case's budget for each count is its frozen value plus a tenth, rounded up:
// the counts are exact, so the headroom is not for noise but lets an
// incidental extra visit pass, while a change that does a tenth more work
// than the frozen code fails. To freeze a change that does more or less work
// on purpose, copy the counts the test prints into this table and state the
// reason in the pull request.
struct Frozen {
    const char* id;
    uint64_t work, comparisons, atomics;
};
constexpr Frozen kFrozen[] = {
    {"storage.scalar_small", 8192, 42245, 0},
    {"storage.scalar_large_scene", 8192, 98308, 0},
    {"storage.array_install", 2048, 10549, 0},
    {"storage.time_samples", 32512, 115376, 0},
    {"storage.namespace_move_pairs", 125165, 232164, 0},
    {"storage.field_reads", 16384, 196623, 0},
    {"storage.mixed_pair_2_readers", 73728, 346128, 0},
    {"storage.mixed_pair_4_readers", 122880, 576880, 0},
    {"turn.field_x4", 636, 2104, 128},
    {"turn.samples", 194, 1138, 58},
    {"turn.create", 522, 6410, 90},
    {"turn.rename", 882, 9343, 70},
    {"turn.reparent", 1412, 83899, 94},
    {"turn.reorder", 1407, 164193, 42},
    {"turn.delete", 1434, 64991, 70},
    {"publication.peer_field", 72, 155, 18},
    {"publication.peer_namespace_x3", 573, 4604, 110},
    {"turn.multi_prim_150_writes", 7162, 73016, 2118},
    {"turn.multi_prim_3000_rewrites", 15712, 112916, 36318},
    {"turn.wide_spec_8192_fields", 65497, 74, 32},
    {"publication.wide_peer_write_8192_fields", 8263, 49284, 24},
    {"publication.wide_peer_change_8192_fields", 16442, 32786, 18},
    {"install.wide_8192_fields", 8198, 16393, 20},
    {"install.wide_over_8192_fields", 16393, 16394, 20},
    {"flush.capture", 0, 0, 18},
    {"read.field", 12000, 167765, 120000},
    {"read.typed_field", 12000, 167765, 168000},
    {"read.spec_type", 0, 167765, 96000},
    {"read.time_sample", 24000, 167765, 120000},
};
// The comparison counts that differ under MSVC, measured on windows-x86_64
// (both USD profiles count the same). The host reparent, reorder and subtree
// delete turns visit and copy the same elements and take the same atomic
// updates there, but make a different number of index comparisons. Each
// replaces the case's frozen comparisons above; the other cases retain their
// shared budgets. Checked MSVC algorithms also validate a
// comparator's reverse ordering: the sibling-sort comparator looks up scene
// sections, so these extra calls count index comparisons. Keep separate
// checked counts without relaxing the release budgets, element or atomic counts.
struct FrozenComparisons {
    const char* id;
    uint64_t comparisons, checkedComparisons;
};
constexpr FrozenComparisons kFrozenMsvc[] = {
    {"turn.reparent", 69409, 78509},
    {"turn.reorder", 148233, 169905},
    {"turn.delete", 72411, 85431},
};
#if defined(_MSC_VER) && _ITERATOR_DEBUG_LEVEL == 2
constexpr bool kMsvcChecked = true;
#else
constexpr bool kMsvcChecked = false;
#endif
#ifdef _MSC_VER
constexpr bool kMsvc = true;
#else
constexpr bool kMsvc = false;
#endif
uint64_t budget(uint64_t frozen) { return frozen + (frozen + 9) / 10; }

// The storage benchmark workloads of tests/benchmarks/scene-workloads.json
// (the same kinds, spec counts, operation counts and array sizes, with the
// benchmark driver's seeded history), run once on SdfSceneData. Mixed cases
// run the writer's transactions and each reader's reads in turn on this
// thread; the counts of a read do not depend on the threads around it.
struct StorageCase {
    const char* id;
    const char* kind;
    size_t specs, operations, elements, readers;
};
constexpr StorageCase kStorageCases[] = {
    {"scalar_small", "field_set", 32, 8192, 0, 0},
    {"scalar_large_scene", "field_set", 4096, 8192, 0, 0},
    {"array_install", "array_set", 32, 2048, 4096, 0},
    {"time_samples", "time_sample", 128, 8192, 0, 0},
    {"namespace_move_pairs", "move_pair", 128, 4096, 0, 0},
    {"field_reads", "field_read", 4096, 16384, 0, 0},
    {"mixed_pair_2_readers", "mixed_pair", 128, 8192, 0, 2},
    {"mixed_pair_4_readers", "mixed_pair", 128, 8192, 0, 4},
};
Cost storageCase(const StorageCase& workload, uint64_t seed) {
    const std::string kind(workload.kind);
    static const TfToken value("bench_value"), revision("bench_revision"), payload("bench_payload");
    constexpr int pairMask = 0x5a5a5a;
    std::vector<SdfPath> paths, moved;
    auto initial = TfCreateRefPtr(new SdfData);
    initial->CreateSpec(SdfPath::AbsoluteRootPath(), SdfSpecTypePseudoRoot);
    initial->CreateSpec(SdfPath("/P"), SdfSpecTypePrim);
    for (size_t i = 0; i < workload.specs; ++i) {
        paths.emplace_back("/P.v" + std::to_string(i));
        moved.emplace_back("/P.m" + std::to_string(i));
        initial->CreateSpec(paths[i], SdfSpecTypeAttribute);
        initial->Set(paths[i], value, VtValue(int(i)));
        if (workload.readers) {
            initial->Set(paths[i], revision, VtValue(0));
            initial->Set(paths[i], payload, VtValue(pairMask));
        }
    }
    struct Op {
        size_t index;
        VtValue value;
        double time;
    };
    std::mt19937_64 random(seed);
    std::vector<Op> history;
    for (size_t i = 0; i < workload.operations; ++i) {
        const size_t index = random() % workload.specs;
        VtValue next(int(random() % 1000000));
        if (workload.elements) next = VtValue(VtIntArray(workload.elements, int(i)));
        history.push_back({index, std::move(next), double(random() % 32)});
    }
    auto data = TfCreateRefPtr(new SdfSceneData);
    data->CopyFrom(initial);
    uint64_t checksum = 0;
    const auto cost = metered([&] {
        for (size_t i = 0; i < history.size(); ++i) {
            const auto& op = history[i];
            const auto& path = paths[op.index];
            if (kind == "field_read") checksum += uint64_t(data->Get(path, value).Get<int>());
            else if (kind == "time_sample") data->SetTimeSample(path, op.time, op.value);
            else if (kind == "move_pair") {
                data->MoveSpec(path, moved[op.index]);
                data->MoveSpec(moved[op.index], path);
            } else if (kind == "mixed_pair") {
                {
                    auto guard = data->AcquireWriteLock();
                    data->Set(path, revision, VtValue(int(i + 1)));
                    data->Set(path, payload, VtValue(int(i + 1) ^ pairMask));
                }
                for (size_t reader = 0; reader < workload.readers; ++reader) {
                    auto guard = data->AcquireReadLock();
                    const auto& read = paths[history[(i + reader * 31) % history.size()].index];
                    checksum += uint64_t(data->Get(read, revision).Get<int>() ^ data->Get(read, payload).Get<int>());
                }
            } else data->Set(path, value, op.value);
        }
    });
    require(kind != "field_read" || checksum != 0, "the field reads read nothing");
    require(kind != "move_pair" || data->GetSpecType(paths.front()) == SdfSpecTypeAttribute,
        "the move pairs did not return their specs");
    return cost;
}

// The layers of these cases hold 4,000 unedited prims (see footprint::scene).
constexpr size_t kLayerPrims = 4000;
// A warm-up turn rebaselines the projection on a compiled candidate, as every
// later turn of a live layer finds it.
void warmUp(Layer& layer) {
    layer.turn([&](SdfAbstractData& data) { data.Set(SdfPath("/E.x"), SdfFieldKeys->Default, VtValue(-1.0)); });
}
// The capture each flush takes of the host storage, the copy of the whole
// scene it compiles against, after a host edit of one field.
Cost flushCapture() {
    Layer layer(scene(kLayerPrims));
    warmUp(layer);
    layer.projection->Set(SdfPath("/E.x"), SdfFieldKeys->Default, VtValue(-2.0));
    HostSceneProjection::LocalEdits edits;
    const auto cost = metered([&] { edits = layer.projection->pendingLocalEdits(); });
    require(edits.data && edits.data->Get(SdfPath("/E.x"), SdfFieldKeys->Default) == VtValue(-2.0) &&
            edits.operations->size() == 1,
        "the flush capture differs from the host storage");
    return cost;
}
// The reads USD composition makes of a live layer's data, through its host
// projection, once per attribute of every unedited prim of the layer: a
// field's value, a typed value as SdfLayer reads one, a spec's type and a time
// sample.
std::vector<std::pair<std::string, Cost>> compositionReads() {
    Layer layer(scene(kLayerPrims));
    warmUp(layer);
    const auto& projection = *layer.projection;
    std::vector<SdfPath> attributes;
    for (size_t i = 0; i < kLayerPrims; ++i)
        for (const char* attribute : {"a", "b", "c"})
            attributes.emplace_back("/U" + std::to_string(i) + "." + attribute);
    std::vector<std::pair<std::string, Cost>> result;
    size_t found = 0;
    result.emplace_back("read.field", metered([&] {
        for (const auto& path : attributes) found += !projection.Get(path, SdfFieldKeys->Default).IsEmpty();
    }));
    result.emplace_back("read.typed_field", metered([&] {
        double value = 0;
        for (const auto& path : attributes) {
            SdfAbstractDataTypedValue<double> typed(&value);
            found += projection.Has(path, SdfFieldKeys->Default, &typed);
        }
    }));
    result.emplace_back("read.spec_type", metered([&] {
        for (const auto& path : attributes) found += projection.GetSpecType(path) == SdfSpecTypeAttribute;
    }));
    result.emplace_back("read.time_sample", metered([&] {
        VtValue sample;
        for (const auto& path : attributes) found += projection.QueryTimeSample(path, 1.0, &sample);
    }));
    require(found == 4 * attributes.size(), "a composition read missed its attribute");
    return result;
}

std::vector<std::pair<std::string, Cost>> measureCases(uint64_t seed) {
    std::vector<std::pair<std::string, Cost>> cases;
    for (const auto& workload : kStorageCases)
        cases.emplace_back(std::string("storage.") + workload.id, storageCase(workload, seed));
    const auto work = measure(kLayerPrims);
    for (const auto& [id, cost] : std::initializer_list<std::pair<const char*, Cost>>{
             {"turn.field_x4", work.field}, {"turn.samples", work.samples}, {"turn.create", work.create},
             {"turn.rename", work.rename}, {"turn.reparent", work.reparent}, {"turn.reorder", work.reorder},
             {"turn.delete", work.erase}, {"publication.peer_field", work.peer},
             {"publication.peer_namespace_x3", work.structure}})
        cases.emplace_back(id, cost);
    cases.emplace_back("turn.multi_prim_150_writes", multiPrimTurn(kLayerPrims, 50, 1));
    cases.emplace_back("turn.multi_prim_3000_rewrites", multiPrimTurn(kLayerPrims, 50, 20));
    cases.emplace_back("turn.wide_spec_8192_fields", wideSpecTurn(8192));
    const auto peer = widePeerUpdates(8192);
    cases.emplace_back("publication.wide_peer_write_8192_fields", peer.first);
    cases.emplace_back("publication.wide_peer_change_8192_fields", peer.second);
    const auto install = wideInstall(8192);
    cases.emplace_back("install.wide_8192_fields", install.first);
    cases.emplace_back("install.wide_over_8192_fields", install.second);
    cases.emplace_back("flush.capture", flushCapture());
    for (auto& read : compositionReads()) cases.push_back(std::move(read));
    return cases;
}
} // namespace

// Fails when a case exceeds its budget in any count, or when the measured
// cases and the frozen table differ.
void workBudgetTests() {
    // The production configuration: the whole-scene derivation and compile
    // checks are test-only and would count their own work.
    setWireSceneDerivationCheck(false);
    setFieldEditCompileCheck(false);
    const auto cases = measureCases(42);
    setWireSceneDerivationCheck(true);
    setFieldEditCompileCheck(true);
    std::map<std::string, Frozen> frozen;
    for (const auto& entry : kFrozen) frozen.emplace(entry.id, entry);
    for (const auto& entry : kFrozenMsvc) {
        const auto found = frozen.find(entry.id);
        if (found == frozen.end()) throw std::runtime_error(std::string("no frozen case ") + entry.id);
        if (kMsvc) found->second.comparisons = kMsvcChecked ? entry.checkedComparisons : entry.comparisons;
    }
    std::ostringstream failures;
    for (const auto& [id, cost] : cases) {
        std::cout << "work_budget " << id << " work=" << cost.work << " comparisons=" << cost.comparisons
                  << " atomics=" << cost.atomics;
        const auto found = frozen.find(id);
        if (found == frozen.end()) {
            std::cout << " (no frozen budget)" << std::endl;
            failures << "\n  " << id << " has no frozen budget";
            continue;
        }
        const auto entry = found->second;
        frozen.erase(found);
        std::cout << " budgets " << budget(entry.work) << '/' << budget(entry.comparisons) << '/'
                  << budget(entry.atomics) << std::endl;
        for (const auto& [name, measured, limit] : {std::tuple{"work", cost.work, entry.work},
                 std::tuple{"comparisons", cost.comparisons, entry.comparisons},
                 std::tuple{"atomics", cost.atomics, entry.atomics}})
            if (measured > budget(limit))
                failures << "\n  " << id << ' ' << name << ' ' << measured << " exceeds its budget " << budget(limit)
                         << " (frozen " << limit << ')';
    }
    for (const auto& [id, entry] : frozen) failures << "\n  " << id << " is frozen but not measured";
    if (!failures.str().empty()) throw std::runtime_error("work budgets exceeded:" + failures.str());
    std::cout << "work budgets hold for " << cases.size() << " cases" << std::endl;
}
