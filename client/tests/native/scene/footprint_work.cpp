// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// A host turn's scene work follows its footprint: the specs and fields it
// touches, never the unedited rest of the layer. Each case runs the production
// steps of one turn (capture, compile and admission, rebaseline, the
// acknowledgment's application and the publication's comparison and reseat)
// on layers that differ only in unedited prims, and compares the scene work
// meter, not wall time.
#include "PersistentMap.h"
#include "SceneJournal.h"
#include "SceneWork.h"
#include "SdfSceneData.h"
#include "captured_log.h"
#include <SHA256.h>
#include <pxr/usd/sdf/data.h>
#include <pxr/usd/sdf/schema.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <string>

PXR_NAMESPACE_USING_DIRECTIVE
using namespace omni::usd::live;
using namespace omni::usd::live::wire;
using namespace omni::usd::live::test;
namespace {
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error("footprint work: " + message);
}
// Seeded edits of copies of one map agree with independent std::map copies and
// never change another copy; the difference of two copies reports exactly the
// keys one side lacks or whose values differ, each with both sides' values.
void persistentMapSeeds(uint64_t seed) {
    using Map = PersistentMap<int, int>;
    std::mt19937_64 random(seed ^ 0x70657273697374ull);
    std::vector<Map> maps(4);
    std::vector<std::map<int, int>> oracles(4);
    const auto agree = [&](size_t i) {
        require(maps[i].size() == oracles[i].size(), "map size differs from its oracle");
        auto expected = oracles[i].begin();
        for (const auto& [key, value] : maps[i]) {
            require(expected != oracles[i].end() && key == expected->first && value == expected->second,
                "map iteration differs from its oracle");
            ++expected;
        }
        require(expected == oracles[i].end(), "map iteration ended early");
        const int probe = int(random() % 600);
        const auto bound = maps[i].lower_bound(probe);
        const auto oracle = oracles[i].lower_bound(probe);
        require((bound == maps[i].end()) == (oracle == oracles[i].end()) &&
                (oracle == oracles[i].end() || (bound.key() == oracle->first && bound.value() == oracle->second)),
            "lower bound differs from its oracle");
    };
    const auto differ = [&](size_t a, size_t b) {
        std::map<int, std::pair<const int*, const int*>> reported;
        Map::difference(maps[a], maps[b], [&](int key, const int* before, const int* after) {
            require(reported.emplace(key, std::make_pair(before, after)).second && (before || after),
                "difference reported a key twice or with neither side");
        });
        std::set<int> keys;
        for (const auto& item : oracles[a]) keys.insert(item.first);
        for (const auto& item : oracles[b]) keys.insert(item.first);
        for (const auto key : keys) {
            const auto x = oracles[a].find(key), y = oracles[b].find(key);
            const bool same = x != oracles[a].end() && y != oracles[b].end() && x->second == y->second;
            const auto found = reported.find(key);
            if (!same) require(found != reported.end(), "difference omitted a changed key");
            if (found == reported.end()) continue;
            const auto [before, after] = found->second;
            require((before ? x != oracles[a].end() && *before == x->second : x == oracles[a].end()) &&
                    (after ? y != oracles[b].end() && *after == y->second : y == oracles[b].end()),
                "difference reported wrong values");
        }
        require(reported.size() <= keys.size(), "difference reported an absent key");
    };
    for (int step = 0; step < 40000; ++step) {
        const size_t i = random() % maps.size();
        const int key = int(random() % 512);
        switch (random() % 6) {
        case 0: {
            const auto j = random() % maps.size();
            maps[i] = maps[j];
            oracles[i] = oracles[j];
            break;
        }
        case 1: require(maps[i].insert(key, step) == oracles[i].emplace(key, step).second, "insert differs"); break;
        case 2:
            maps[i].assign(key, step);
            oracles[i][key] = step;
            break;
        case 3: {
            auto* value = maps[i].edit(key);
            const auto found = oracles[i].find(key);
            require((value != nullptr) == (found != oracles[i].end()), "edit presence differs");
            if (value) *value = found->second = -step;
            break;
        }
        default: require(maps[i].erase(key) == (oracles[i].erase(key) == 1), "erase differs"); break;
        }
        if (step % 499 == 0) {
            for (size_t m = 0; m < maps.size(); ++m) agree(m);
            differ(random() % maps.size(), random() % maps.size());
            // Against an empty map every key is on one side only.
            maps.emplace_back();
            oracles.emplace_back();
            differ(random() % (maps.size() - 1), maps.size() - 1);
            differ(maps.size() - 1, random() % (maps.size() - 1));
            maps.pop_back();
            oracles.pop_back();
        }
    }
    // An edit of a copy of a large map copies a logarithmic number of nodes,
    // and the copies' difference visits only those.
    Map large;
    for (int key = 0; key < 100000; ++key) large.insert(key, key);
    auto copy = large;
    auto start = sceneWork::total();
    *copy.edit(4242) = -1;
    copy.erase(77);
    copy.insert(100001, 1);
    require(sceneWork::total() - start < 160, "an edit of a large map's copy copied more than its paths");
    start = sceneWork::total();
    size_t changed = 0;
    Map::difference(large, copy, [&](int, const int*, const int*) { ++changed; });
    require(changed == 3 && sceneWork::total() - start < 400, "the difference of two copies visited shared subtrees");
}
Command create(uint64_t id, uint64_t parent, const std::string& name, SdfSpecType type) {
    Command result{};
    result.kind = Command::Kind::Create;
    result.section = id;
    result.parent = parent;
    result.name = name;
    result.sectionType = uint8_t(type);
    return result;
}
// Upper bounds of a field write's index comparisons per level of the layer's
// index, through capture, compile, admission, acknowledgment and publication:
// a turn that writes each attribute once, and one that rewrites them.
constexpr double kSingleWriteComparisons = 40, kRepeatedWriteComparisons = 4;
// Seeded field edits of one spec of scene data agree with an oracle that
// keeps the order fields were first set, through erasures and re-sets of
// records small and large, and through copies that share the record.
void specFieldSeeds(uint64_t seed) {
    std::mt19937_64 random(seed ^ 0x6669656c6473ull);
    const SdfPath path("/P");
    for (const size_t names : {size_t(4), size_t(40), size_t(400)}) {
        auto data = TfCreateRefPtr(new SdfSceneData);
        data->CreateSpec(path, SdfSpecTypePrim);
        std::vector<std::pair<TfToken, int>> oracle;
        const auto agree = [&](const SdfSceneData& scene, const std::vector<std::pair<TfToken, int>>& expected) {
            const auto listed = scene.List(path);
            require(listed.size() == expected.size(), "a spec lists a different number of fields");
            for (size_t i = 0; i < expected.size(); ++i)
                require(listed[i] == expected[i].first && scene.Get(path, expected[i].first) == VtValue(expected[i].second),
                    "a spec's fields differ from their first-set order or values");
        };
        for (int step = 0; step < 6000; ++step) {
            const TfToken name("f" + std::to_string(random() % names));
            auto found = std::find_if(oracle.begin(), oracle.end(), [&](const auto& item) { return item.first == name; });
            switch (random() % 5) {
            case 0:
            case 1:
                data->Erase(path, name);
                if (found != oracle.end()) oracle.erase(found);
                break;
            case 2:
                data->Set(path, name, VtValue());
                if (found != oracle.end()) oracle.erase(found);
                break;
            default:
                data->Set(path, name, VtValue(step));
                if (found != oracle.end()) found->second = step;
                else oracle.emplace_back(name, step);
                break;
            }
            require(data->Has(path, name, static_cast<VtValue*>(nullptr)) ==
                    (std::find_if(oracle.begin(), oracle.end(), [&](const auto& item) { return item.first == name; }) !=
                        oracle.end()),
                "a spec's field presence differs from its oracle");
            if (step % 97 == 0) {
                agree(*data, oracle);
                // An edit of a copy changes neither the original nor its order.
                auto copy = TfCreateRefPtr(new SdfSceneData);
                copy->CopyFrom(data);
                copy->Set(path, TfToken("copied"), VtValue(-1));
                if (!oracle.empty()) copy->Erase(path, oracle.front().first);
                agree(*data, oracle);
            }
        }
        agree(*data, oracle);
    }
}
// Building and copying a spec with `count` fields: a full wire build, an
// update that adds the fields to an existing section, and a copy from
// foreign data.
struct FieldWork {
    uint64_t build = 0, update = 0, copy = 0;
};
FieldWork fieldWork(size_t count) {
    EnvelopeInput structure;
    structure.commands.push_back(create(2, 1, "P", SdfSpecTypePrim));
    EnvelopeInput metadata = structure;
    DecodedValues values;
    for (size_t i = 0; i < count; ++i) {
        metadata.fields.push_back({2, 0, "f" + std::to_string(i), {}});
        values.fields.push_back(VtValue(int(i)));
    }
    WireSceneLimits limits;
    limits.maxWork = size_t(1) << 40;
    FieldWork result;
    auto start = sceneWork::total();
    const auto built = buildSceneCandidate({}, encodeUsdEnvelope(metadata, values), limits);
    result.build = sceneWork::total() - start;
    const auto empty = buildSceneCandidate({}, encodeUsdEnvelope(structure, {}), limits);
    metadata.commands.clear();
    start = sceneWork::total();
    const auto updated = buildSceneCandidate(empty, encodeUsdEnvelope(metadata, values), limits);
    result.update = sceneWork::total() - start;
    const SdfPath path("/P");
    require(built->data()->List(path).size() == count && updated->data()->List(path).size() == count &&
            updated->data()->Get(path, TfToken("f" + std::to_string(count - 1))) == VtValue(int(count - 1)),
        "a spec with many fields lost some");
    auto foreign = TfCreateRefPtr(new SdfData);
    foreign->CopyFrom(built->data());
    auto copy = TfCreateRefPtr(new SdfSceneData);
    start = sceneWork::total();
    copy->CopyFrom(foreign);
    result.copy = sceneWork::total() - start;
    require(copy->List(path).size() == count, "a copy of a spec with many fields lost some");
    return result;
}
// Each per-field operation on a spec costs at most a logarithmic factor of
// its field count, so building or copying a spec with F fields stays within a
// constant times F log F, however many fields one section carries.
void fieldsOfOneSpec() {
    const size_t small = 512, large = 8192;
    const auto a = fieldWork(small), b = fieldWork(large);
    std::cout << "footprint_work fields=" << small << " build=" << a.build << " update=" << a.update
              << " copy=" << a.copy << "; fields=" << large << " build=" << b.build << " update=" << b.update
              << " copy=" << b.copy << std::endl;
    const auto linear = [&](uint64_t x, uint64_t y, const char* what) {
        require(y <= 2 * (large / small) * x + 1024,
            std::string(what) + " work grows faster than its spec's field count: " + std::to_string(x) + " at " +
                std::to_string(small) + " fields, " + std::to_string(y) + " at " + std::to_string(large) + " fields");
    };
    linear(a.build, b.build, "a wire build of one spec's fields");
    linear(a.update, b.update, "an update of one section's fields");
    linear(a.copy, b.copy, "a foreign copy of one spec's fields");
}
// Children of /R, the parent whose order a structural turn changes.
constexpr size_t kReorderChildren = 64;
// `prims` unedited prims /U<i>, each with attributes a, b and c holding a
// default and two time samples, beside the edited prim /E with attributes x
// and y and the prim /R with kReorderChildren child prims /R/K<i>:
// 4 * prims + kReorderChildren + 5 specs.
WireScenePtr scene(size_t prims) {
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
        : initial(std::move(scene)), journal(initial, 10, {}), progress({1, "session", "layer", "writer", 1}, 10, initial, 128, 1000),
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
struct Work {
    uint64_t field = 0, samples = 0, peer = 0, structure = 0;
    // Host structural turns.
    uint64_t create = 0, rename = 0, reparent = 0, reorder = 0, erase = 0;
};
// The prim children `parent` lists.
std::vector<TfToken> primChildren(const SdfAbstractData& data, const SdfPath& parent) {
    const auto value = data.Get(parent, SdfChildrenKeys->PrimChildren);
    return value.IsHolding<std::vector<TfToken>>() ? value.UncheckedGet<std::vector<TfToken>>() :
                                                     std::vector<TfToken>{};
}
// Authors `parent`'s prim children as stock Sdf does: an empty list erases the field.
void setPrimChildren(SdfAbstractData& data, const SdfPath& parent, const std::vector<TfToken>& children) {
    if (children.empty()) data.Erase(parent, SdfChildrenKeys->PrimChildren);
    else data.Set(parent, SdfChildrenKeys->PrimChildren, VtValue(children));
}
// Moves the prim `from`, with its attribute z, to `to`, as stock Sdf records a
// namespace edit: each spec moves, then both parents' child lists change.
void movePrim(SdfAbstractData& data, const SdfPath& from, const SdfPath& to) {
    data.MoveSpec(from, to);
    data.MoveSpec(from.AppendProperty(TfToken("z")), to.AppendProperty(TfToken("z")));
    auto source = primChildren(data, from.GetParentPath());
    source.erase(std::find(source.begin(), source.end(), from.GetNameToken()));
    if (from.GetParentPath() == to.GetParentPath()) {
        source.push_back(to.GetNameToken());
        setPrimChildren(data, from.GetParentPath(), source);
        return;
    }
    setPrimChildren(data, from.GetParentPath(), source);
    auto target = primChildren(data, to.GetParentPath());
    target.push_back(to.GetNameToken());
    setPrimChildren(data, to.GetParentPath(), target);
}
Command command(Command::Kind kind, uint64_t id, uint64_t parent, const std::string& name, SdfSpecType type,
    uint64_t oldParent = 0) {
    Command result{};
    result.kind = kind;
    result.section = id;
    result.parent = parent;
    result.oldParent = oldParent;
    result.name = name;
    result.sectionType = uint8_t(type);
    return result;
}
// Every spec of `data` with its type and fields.
std::map<SdfPath, std::pair<SdfSpecType, std::map<TfToken, VtValue>>> inspect(const SdfAbstractData& data) {
    struct Visitor : SdfAbstractDataSpecVisitor {
        std::map<SdfPath, std::pair<SdfSpecType, std::map<TfToken, VtValue>>> specs;
        bool VisitSpec(const SdfAbstractData& source, const SdfPath& path) override {
            auto& spec = specs[path];
            spec.first = source.GetSpecType(path);
            for (const auto& field : source.List(path)) spec.second.emplace(field, source.Get(path, field));
            return true;
        }
        void Done(const SdfAbstractData&) override {}
    } visitor;
    data.VisitSpecs(&visitor);
    return visitor.specs;
}
Work measure(size_t prims) {
    Layer layer(scene(prims));
    const SdfPath x("/E.x");
    // A warm-up turn rebaselines the projection on a compiled candidate, as
    // every later turn of a live layer finds it.
    layer.turn([&](SdfAbstractData& data) { data.Set(x, SdfFieldKeys->Default, VtValue(-1.0)); });
    Work result;
    auto start = sceneWork::total();
    for (int i = 0; i < 4; ++i)
        layer.turn([&](SdfAbstractData& data) { data.Set(x, SdfFieldKeys->Default, VtValue(double(i))); });
    result.field = sceneWork::total() - start;
    start = sceneWork::total();
    layer.turn([&](SdfAbstractData& data) {
        data.SetTimeSample(x, 2.0, VtValue(3.0));
        data.EraseTimeSample(x, 0.0);
        data.Set(SdfPath("/E"), SdfFieldKeys->Documentation, VtValue(std::string("edited")));
    });
    result.samples = sceneWork::total() - start;
    start = sceneWork::total();
    layer.peer(4, 9.0); // /E.y
    result.peer = sceneWork::total() - start;
    // A peer creates a child of /E with an attribute, renames it and deletes
    // it: each publishes its namespace change. The peer's IDs follow the
    // fixture's, so every layer size measures the same three changes, and the
    // projection is checked between them, outside the measured work.
    const uint64_t child = 4 * prims + kReorderChildren + 6, attribute = child + 1;
    require(!layer.initial->containsSection(child) && !layer.initial->containsSection(attribute),
        "the peer's section IDs are already live");
    const auto structure = [&](std::vector<Command> commands) {
        EnvelopeInput metadata;
        metadata.commands = std::move(commands);
        const auto bytes = encodeEnvelope(metadata);
        const auto begin = sceneWork::total();
        layer.peer(bytes);
        result.structure += sceneWork::total() - begin;
    };
    const auto has = [&](const char* path) { return layer.projection->HasSpec(SdfPath(path)); };
    structure({command(Command::Kind::Create, child, 2, "Child", SdfSpecTypePrim),
        command(Command::Kind::Create, attribute, child, "z", SdfSpecTypeAttribute)});
    require(has("/E/Child") && has("/E/Child.z"), "the peer's creation was not published");
    structure({command(Command::Kind::Move, child, 2, "Renamed", SdfSpecTypeUnknown, 2)});
    require(has("/E/Renamed") && has("/E/Renamed.z") && !has("/E/Child") && !has("/E/Child.z"),
        "the peer's rename was not published");
    structure({command(Command::Kind::Delete, child, 2, "", SdfSpecTypeUnknown)});
    require(!has("/E/Renamed") && !has("/E/Renamed.z") && !has("/E/Child"), "the peer's deletion was not published");
    // Host structural turns: create /E/C with an attribute, rename it to
    // /E/D, reparent it to /R/D, reverse /R's children and delete /R/D's
    // subtree. Each is one turn through capture, compile, admission,
    // acknowledgment and publication.
    const auto structural = [&](uint64_t& slot, const std::function<void(SdfAbstractData&)>& author) {
        const auto before = sceneWork::total();
        layer.turn(author);
        slot = sceneWork::total() - before;
    };
    structural(result.create, [](SdfAbstractData& data) {
        const SdfPath child("/E/C"), attribute("/E/C.z");
        data.CreateSpec(child, SdfSpecTypePrim);
        auto children = primChildren(data, SdfPath("/E"));
        children.emplace_back("C");
        setPrimChildren(data, SdfPath("/E"), children);
        data.CreateSpec(attribute, SdfSpecTypeAttribute);
        data.Set(child, SdfChildrenKeys->PropertyChildren, VtValue(std::vector<TfToken>{TfToken("z")}));
        data.Set(attribute, SdfFieldKeys->Default, VtValue(5.0));
    });
    structural(result.rename, [](SdfAbstractData& data) { movePrim(data, SdfPath("/E/C"), SdfPath("/E/D")); });
    structural(result.reparent, [](SdfAbstractData& data) { movePrim(data, SdfPath("/E/D"), SdfPath("/R/D")); });
    structural(result.reorder, [](SdfAbstractData& data) {
        auto children = primChildren(data, SdfPath("/R"));
        std::reverse(children.begin(), children.end());
        setPrimChildren(data, SdfPath("/R"), children);
    });
    structural(result.erase, [](SdfAbstractData& data) {
        data.EraseSpec(SdfPath("/R/D.z"));
        data.EraseSpec(SdfPath("/R/D"));
        auto children = primChildren(data, SdfPath("/R"));
        children.erase(std::find(children.begin(), children.end(), TfToken("D")));
        setPrimChildren(data, SdfPath("/R"), children);
    });
    auto reversed = primChildren(*layer.projection, SdfPath("/R"));
    std::reverse(reversed.begin(), reversed.end());
    require(reversed.size() == kReorderChildren && reversed.front() == TfToken("K0") &&
            !layer.projection->HasSpec(SdfPath("/R/D")) && !layer.projection->HasSpec(SdfPath("/E/C")) &&
            !layer.projection->HasSpec(SdfPath("/E/D")) && primChildren(*layer.projection, SdfPath("/E")).empty(),
        "the host's structural turns were not published");
    VtValue sample;
    require(layer.projection->Get(x, SdfFieldKeys->Default) == VtValue(3.0) &&
            layer.projection->ListTimeSamplesForPath(x) == std::set<double>{1.0, 2.0} &&
            layer.projection->QueryTimeSample(x, 2.0, &sample) && sample == VtValue(3.0) &&
            layer.projection->Get(SdfPath("/E"), SdfFieldKeys->Documentation) == VtValue(std::string("edited")) &&
            layer.projection->Get(SdfPath("/E.y"), SdfFieldKeys->Default) == VtValue(9.0) &&
            !layer.projection->hasPendingLocalEdits(),
        "the host content differs from the published turns");
    require(inspect(*layer.projection) == inspect(*layer.journal.snapshot()->visible->data()),
        "the host content differs from the journal's visible scene");
    return result;
}
// The key comparisons of index lookups and edits per field write: one host
// turn that sets the default of attributes a, b and c of `prims` unedited prims
// /U<i>, as one Kit frame of a multi-prim edit authors them, through capture,
// compile, admission, acknowledgment and publication, `repeats` times over in
// the turn.
double comparisonsPerWrite(size_t layerPrims, size_t prims, size_t repeats) {
    Layer layer(scene(layerPrims));
    // A warm-up turn rebaselines the projection on a compiled candidate.
    layer.turn([&](SdfAbstractData& data) { data.Set(SdfPath("/E.x"), SdfFieldKeys->Default, VtValue(-1.0)); });
    const auto start = sceneWork::comparisons();
    layer.turn([&](SdfAbstractData& data) {
        for (size_t repeat = 0; repeat < repeats; ++repeat)
            for (size_t i = 0; i < prims; ++i)
                for (const char* attribute : {"a", "b", "c"})
                    data.Set(SdfPath("/U" + std::to_string(i) + "." + attribute), SdfFieldKeys->Default,
                        VtValue(double(repeat)));
    });
    const auto writes = double(prims * 3 * repeats);
    require(layer.projection->Get(SdfPath("/U0.a"), SdfFieldKeys->Default) == VtValue(double(repeats - 1)) &&
            !layer.projection->hasPendingLocalEdits(),
        "the multi-prim turn was not published");
    return double(sceneWork::comparisons() - start) / writes;
}
// The scene work of one host turn that writes one field of the prim /P, which
// holds `count` fields, through capture, compile and its verification of /P
// against the replay and the captured storage, admission, acknowledgment and
// publication.
uint64_t wideSpecTurn(size_t count) {
    EnvelopeInput metadata;
    DecodedValues values;
    metadata.commands.push_back(create(2, 1, "P", SdfSpecTypePrim));
    for (size_t i = 0; i < count; ++i) {
        metadata.fields.push_back({2, 0, "f" + std::to_string(i), {}});
        values.fields.push_back(VtValue(int(i)));
    }
    WireSceneLimits limits;
    limits.maxWork = size_t(1) << 40;
    Layer layer(buildSceneCandidate({}, encodeUsdEnvelope(metadata, values), limits));
    const SdfPath path("/P");
    const TfToken last("f" + std::to_string(count - 1));
    // A warm-up turn rebaselines the projection on a compiled candidate.
    layer.turn([&](SdfAbstractData& data) { data.Set(path, last, VtValue(-1)); });
    const auto start = sceneWork::total();
    layer.turn([&](SdfAbstractData& data) { data.Set(path, last, VtValue(-2)); });
    const auto work = sceneWork::total() - start;
    require(layer.projection->Get(path, last) == VtValue(-2) && layer.projection->List(path).size() == count &&
            !layer.projection->hasPendingLocalEdits(),
        "the wide spec's turn was not published");
    return work;
}
// A turn that writes one field of a spec with F fields verifies that spec
// with work linear in F, however many fields it carries.
void wideSpecVerification() {
    const size_t small = 512, large = 8192;
    const auto a = wideSpecTurn(small), b = wideSpecTurn(large);
    std::cout << "footprint_work wide spec turn fields=" << small << " work=" << a << "; fields=" << large
              << " work=" << b << std::endl;
    require(b <= 2 * (large / small) * a + 1024,
        "a wide spec's turn work grows faster than its field count: " + std::to_string(a) + " at " +
            std::to_string(small) + " fields, " + std::to_string(b) + " at " + std::to_string(large) + " fields");
}
} // namespace
// The work of field, sample and structural host turns and of a peer's field
// and namespace publications on a 4,000-prim layer stays within twice that on
// a 50-prim layer: the unedited specs contribute at most a logarithmic factor.
void footprintWorkTests(uint64_t seed) {
    persistentMapSeeds(seed);
    specFieldSeeds(seed);
    fieldsOfOneSpec();
    // The whole-scene derivation and compile checks would count their own work.
    setWireSceneDerivationCheck(false);
    setFieldEditCompileCheck(false);
    wideSpecVerification();
    const std::array<size_t, 3> sizes{50, 1000, 4000};
    std::array<Work, 3> work;
    for (size_t i = 0; i < sizes.size(); ++i) {
        work[i] = measure(sizes[i]);
        std::cout << "footprint_work prims=" << sizes[i] << " field_turns=" << work[i].field
                  << " sample_turn=" << work[i].samples << " peer_publication=" << work[i].peer
                  << " peer_structure=" << work[i].structure << " create=" << work[i].create
                  << " rename=" << work[i].rename << " reparent=" << work[i].reparent
                  << " reorder=" << work[i].reorder << " delete=" << work[i].erase << std::endl;
    }
    const auto bounded = [&](uint64_t small, uint64_t large, const char* what) {
        require(large <= 2 * small + 64,
            std::string(what) + " work grows with unedited layer size: " + std::to_string(small) + " at " +
                std::to_string(sizes.front()) + " prims, " + std::to_string(large) + " at " +
                std::to_string(sizes.back()) + " prims");
    };
    bounded(work.front().field, work.back().field, "field turn");
    bounded(work.front().samples, work.back().samples, "sample turn");
    bounded(work.front().peer, work.back().peer, "peer publication");
    bounded(work.front().structure, work.back().structure, "peer namespace change");
    bounded(work.front().create, work.back().create, "host create turn");
    bounded(work.front().rename, work.back().rename, "host rename turn");
    bounded(work.front().reparent, work.back().reparent, "host reparent turn");
    bounded(work.front().reorder, work.back().reorder, "host reorder turn");
    bounded(work.front().erase, work.back().erase, "host subtree delete turn");
    // A field write costs a bounded number of index comparisons per level of
    // the layer's index, through every step of its turn: a 50-prim turn of
    // single writes, and one that rewrites those attributes 20 times.
    const auto levels = std::log2(double(4 * sizes.back() + kReorderChildren + 5));
    const auto single = comparisonsPerWrite(sizes.back(), 50, 1) / levels;
    const auto repeated = comparisonsPerWrite(sizes.back(), 50, 20) / levels;
    std::cout << "footprint_work comparisons per write and index level: single=" << single
              << " repeated=" << repeated << std::endl;
    require(single <= kSingleWriteComparisons,
        "a multi-prim field turn compares " + std::to_string(single) + " keys per write and index level");
    require(repeated <= kRepeatedWriteComparisons,
        "a repeated multi-prim field turn compares " + std::to_string(repeated) + " keys per write and index level");
    setWireSceneDerivationCheck(true);
    setFieldEditCompileCheck(true);
    std::cout << "footprint work follows the turn's footprint" << std::endl;
}
