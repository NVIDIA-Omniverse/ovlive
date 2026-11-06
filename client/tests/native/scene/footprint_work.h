// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// The production steps of one live layer's host turns and peer updates, and
// the scene work meter's readings around them, for the footprint and work
// budget tests.
#pragma once
#include "SceneJournal.h"
#include "SceneWork.h"
#include "captured_log.h"
#include <SHA256.h>
#include <pxr/usd/sdf/schema.h>
#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace omni::usd::live::test::footprint {
using namespace PXR_NS;
using namespace omni::usd::live::wire;
inline void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error("footprint work: " + message);
}
// The scene work meter's three counts on the calling thread.
struct Cost {
    uint64_t work = 0, comparisons = 0, atomics = 0;
    static Cost now() { return {sceneWork::total(), sceneWork::comparisons(), sceneWork::atomics()}; }
    Cost operator-(const Cost& other) const {
        return {work - other.work, comparisons - other.comparisons, atomics - other.atomics};
    }
    Cost& operator+=(const Cost& other) {
        work += other.work;
        comparisons += other.comparisons;
        atomics += other.atomics;
        return *this;
    }
};
// The meter's counts for `run`.
template<class Run> Cost metered(Run&& run) {
    const auto start = Cost::now();
    run();
    return Cost::now() - start;
}
inline Command create(uint64_t id, uint64_t parent, const std::string& name, SdfSpecType type) {
    Command result{};
    result.kind = Command::Kind::Create;
    result.section = id;
    result.parent = parent;
    result.name = name;
    result.sectionType = uint8_t(type);
    return result;
}
// Children of /R, the parent whose order a structural turn changes.
constexpr size_t kReorderChildren = 64;
// `prims` unedited prims /U<i>, each with attributes a, b and c holding a
// default and two time samples, beside the edited prim /E with attributes x
// and y and the prim /R with kReorderChildren child prims /R/K<i>:
// 4 * prims + kReorderChildren + 5 specs.
inline WireScenePtr scene(size_t prims) {
    EnvelopeInput metadata;
    DecodedValues values;
    uint64_t next = 2;
    const auto prim = [&](const std::string& name, const std::array<const char*, 3>& attributes) {
        const auto id = next++;
        metadata.commands.push_back(create(id, 1, name, SdfSpecTypePrim));
        for (const auto* attribute : attributes) {
            if (!attribute) continue;
            const auto child = next++;
            metadata.commands.push_back(create(child, id, attribute, SdfSpecTypeAttribute));
            metadata.fields.push_back({child, 0, "default", {}});
            values.fields.push_back(VtValue(double(child)));
            metadata.samples.push_back({child, 0.0, {}});
            values.samples.push_back(VtValue(1.0));
            metadata.samples.push_back({child, 1.0, {}});
            values.samples.push_back(VtValue(2.0));
        }
    };
    prim("E", {"x", "y", nullptr});
    const auto reordered = next++;
    metadata.commands.push_back(create(reordered, 1, "R", SdfSpecTypePrim));
    for (size_t i = 0; i < kReorderChildren; ++i)
        metadata.commands.push_back(create(next++, reordered, "K" + std::to_string(i), SdfSpecTypePrim));
    for (size_t i = 0; i < prims; ++i) prim("U" + std::to_string(i), {"a", "b", "c"});
    WireSceneLimits limits;
    limits.maxWork = size_t(1) << 40;
    return buildSceneCandidate({}, encodeUsdEnvelope(metadata, values), limits);
}
struct Effect {
    std::shared_ptr<const SceneJournal::Intent> intent;
};
// Section IDs drawn from a seeded generator, so the scene index positions
// of the specs a turn creates, and the meter's counts of the turn, are the same
// in every run; production draws them from the system entropy source.
inline std::shared_ptr<SectionIdAllocator> seededIds() {
    return std::make_shared<SectionIdAllocator>(size_t(1) << 26, 64,
        [random = std::make_shared<std::mt19937_64>(0x6964730000000001ull)] { return (*random)(); });
}
// One layer's journal, actor progress and host projection, as a live layer
// holds them.
struct Layer {
    WireScenePtr initial;
    SceneJournal journal;
    LayerProgress progress;
    LayerProgress::ReadCursor read;
    PXR_NS::TfRefPtr<HostSceneProjection> projection;
    uint64_t sequence = 10;
    explicit Layer(WireScenePtr scene)
        : initial(std::move(scene)), journal(initial, 10, {}, seededIds()), progress({1, "session", "layer", "writer", 1}, 10, initial, 128, 1000),
          projection(recordingBacking(initial)) {
        read = progress.startRead();
        require(progress.caughtUp(read.generation, read.object, 10).accepted, "progress did not become ready");
    }
    SceneJournal::Submit submitter() {
        return [this](std::string_view, std::shared_ptr<const SceneJournal::Intent> intent,
                   LayerProgress::Publication publication) {
            const auto retainedBytes = intent->retainedBytes;
            return progress.admit(std::make_shared<const Effect>(Effect{std::move(intent)}), publication, retainedBytes);
        };
    }
    LayerProgress::Work retained(uint64_t id) const {
        for (const auto& work : progress.retained())
            if (work.id == id) return work;
        throw std::runtime_error("footprint work: missing retained progress work");
    }
    // The publication job: compare the visible scene with the host's baseline
    // and reseat the projection when they differ.
    void publish() {
        const auto visible = journal.snapshot()->visible;
        const auto& baseline = projection->baseline();
        const auto difference =
            baseline == visible ? WireSceneDifference{} : diffProjectedScenes(*baseline, *visible);
        if (projection->keepsBaseline(visible, difference)) return;
        projection->reseat(projection->prepareReseat(visible, difference));
    }
    // One host turn that authors `author` and publishes its acknowledgment.
    void turn(const std::function<void(SdfAbstractData&)>& author) {
        author(*projection);
        const auto edits = projection->pendingLocalEdits();
        const auto sealed = std::make_shared<const SdfOperationLog>(SdfOperationLog{edits.data, edits.operations});
        WireScenePtr applied;
        const auto id = journal.submitCaptured(edits.base, sealed, submitter(), &applied);
        require(id != 0 && applied, "the turn admitted nothing");
        projection->acceptLocalEdits(edits.operations->size(), applied);
        require(progress.takeWrite().id == id, "the turn was not issued");
        progress.completeWrite(id, OMNI_USD_LIVE_OK, ++sequence);
        const auto work = retained(id);
        const auto intent = std::static_pointer_cast<const Effect>(work.owner)->intent;
        journal.apply(work, intent->bytes, intent);
        progress.applied(id);
        publish();
    }
    // A peer's field update to `path`'s default, applied and published.
    void peer(uint64_t section, double value) {
        EnvelopeInput metadata;
        metadata.fields = {{section, 0, "default", {}}};
        peer(encodeUsdEnvelope(metadata, DecodedValues{{VtValue(value)}, {}}));
    }
    // A peer's update, applied and published.
    void peer(const std::string& bytes) {
        auto owner = std::make_shared<const std::string>(bytes);
        LayerProgress::ContentHash hash{};
        computeSHA256(bytes.data(), bytes.size(), hash.data());
        const auto id = progress.acceptCoverage(read.generation, read.object, sequence + 1, sequence + 1, {owner, hash});
        require(id != 0, "peer coverage was not retained");
        ++sequence;
        journal.apply(retained(id), bytes);
        progress.applied(id);
        publish();
    }
};
// The meter's counts for the turns and publications of a layer of `prims`
// unedited prims (see scene): four field turns, one sample turn, a peer's
// field update, a peer's create, rename and delete of a child prim, and host
// turns that create, rename, reparent, reorder and delete a prim, each through
// capture, compile, admission, acknowledgment and publication.
struct Work {
    Cost field, samples, peer, structure;
    // Host structural turns.
    Cost create, rename, reparent, reorder, erase;
};
Work measure(size_t prims);
// One host turn that sets the default of attributes a, b and c of `prims`
// unedited prims /U<i> of a layer of `layerPrims`, as one Kit frame of a
// multi-prim edit authors them, `repeats` times over in the turn.
Cost multiPrimTurn(size_t layerPrims, size_t prims, size_t repeats);
// One host turn that writes one field of the prim /P, which holds `count`
// fields, through capture, compile and its verification of /P against the
// replay and the captured storage, admission, acknowledgment and publication.
Cost wideSpecTurn(size_t count);
// Two published peer updates: one that creates the prim /Q with `count`
// fields, which the reseat writes as a new spec, and one that changes every
// one of them, which it applies as changed fields.
std::pair<Cost, Cost> widePeerUpdates(size_t count);
// Copies of a scene whose /Q holds `count` fields into host storage, as a full
// install does: into empty storage, and over storage that already holds /Q
// with other values.
std::pair<Cost, Cost> wideInstall(size_t count);
// Building and copying a spec with `count` fields: a full wire build, an
// update that adds the fields to an existing section, and a copy from
// foreign data.
struct FieldWork {
    Cost build, update, copy;
};
FieldWork fieldWork(size_t count);
} // namespace omni::usd::live::test::footprint
