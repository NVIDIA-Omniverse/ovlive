// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// A host turn's scene work follows its footprint: the specs and fields it
// touches, never the unedited rest of the layer. Each case runs the production
// steps of one turn (capture, compile and admission, rebaseline, the
// acknowledgment's application and the publication's comparison and reseat)
// on layers that differ only in unedited prims, and compares the scene work
// meter, not wall time.
#include "footprint_work.h"
#include "PersistentMap.h"
#include "SdfSceneData.h"
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
#include <tuple>

PXR_NAMESPACE_USING_DIRECTIVE
using namespace omni::usd::live;
using namespace omni::usd::live::wire;
using namespace omni::usd::live::test;
namespace omni::usd::live::test::footprint {
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
    WireScenePtr built, updated;
    result.build = metered([&] { built = buildSceneCandidate({}, encodeUsdEnvelope(metadata, values), limits); });
    const auto empty = buildSceneCandidate({}, encodeUsdEnvelope(structure, {}), limits);
    metadata.commands.clear();
    result.update = metered([&] { updated = buildSceneCandidate(empty, encodeUsdEnvelope(metadata, values), limits); });
    const SdfPath path("/P");
    require(built->data()->List(path).size() == count && updated->data()->List(path).size() == count &&
            updated->data()->Get(path, TfToken("f" + std::to_string(count - 1))) == VtValue(int(count - 1)),
        "a spec with many fields lost some");
    auto foreign = TfCreateRefPtr(new SdfData);
    foreign->CopyFrom(built->data());
    auto copy = TfCreateRefPtr(new SdfSceneData);
    result.copy = metered([&] { copy->CopyFrom(foreign); });
    require(copy->List(path).size() == count, "a copy of a spec with many fields lost some");
    return result;
}
// Each per-field operation on a spec costs at most a logarithmic factor of
// its field count, so building or copying a spec with F fields stays within a
// constant times F log F, however many fields one section carries.
void fieldsOfOneSpec() {
    const size_t small = 512, large = 8192;
    const auto a = fieldWork(small), b = fieldWork(large);
    std::cout << "footprint_work fields=" << small << " build=" << a.build.work << " update=" << a.update.work
              << " copy=" << a.copy.work << "; fields=" << large << " build=" << b.build.work
              << " update=" << b.update.work << " copy=" << b.copy.work << std::endl;
    const auto linear = [&](uint64_t x, uint64_t y, const char* what) {
        require(y <= 2 * (large / small) * x + 1024,
            std::string(what) + " work grows faster than its spec's field count: " + std::to_string(x) + " at " +
                std::to_string(small) + " fields, " + std::to_string(y) + " at " + std::to_string(large) + " fields");
    };
    linear(a.build.work, b.build.work, "a wire build of one spec's fields");
    linear(a.update.work, b.update.work, "an update of one section's fields");
    linear(a.copy.work, b.copy.work, "a foreign copy of one spec's fields");
}
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
    result.field = metered([&] {
        for (int i = 0; i < 4; ++i)
            layer.turn([&](SdfAbstractData& data) { data.Set(x, SdfFieldKeys->Default, VtValue(double(i))); });
    });
    result.samples = metered([&] {
        layer.turn([&](SdfAbstractData& data) {
            data.SetTimeSample(x, 2.0, VtValue(3.0));
            data.EraseTimeSample(x, 0.0);
            data.Set(SdfPath("/E"), SdfFieldKeys->Documentation, VtValue(std::string("edited")));
        });
    });
    result.peer = metered([&] { layer.peer(4, 9.0); }); // /E.y
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
        result.structure += metered([&] { layer.peer(bytes); });
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
    const auto structural = [&](Cost& slot, const std::function<void(SdfAbstractData&)>& author) {
        slot = metered([&] { layer.turn(author); });
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
Cost multiPrimTurn(size_t layerPrims, size_t prims, size_t repeats) {
    Layer layer(scene(layerPrims));
    // A warm-up turn rebaselines the projection on a compiled candidate.
    layer.turn([&](SdfAbstractData& data) { data.Set(SdfPath("/E.x"), SdfFieldKeys->Default, VtValue(-1.0)); });
    const auto cost = metered([&] {
        layer.turn([&](SdfAbstractData& data) {
            for (size_t repeat = 0; repeat < repeats; ++repeat)
                for (size_t i = 0; i < prims; ++i)
                    for (const char* attribute : {"a", "b", "c"})
                        data.Set(SdfPath("/U" + std::to_string(i) + "." + attribute), SdfFieldKeys->Default,
                            VtValue(double(repeat)));
        });
    });
    require(layer.projection->Get(SdfPath("/U0.a"), SdfFieldKeys->Default) == VtValue(double(repeats - 1)) &&
            !layer.projection->hasPendingLocalEdits(),
        "the multi-prim turn was not published");
    return cost;
}
// The key comparisons of index lookups and edits per field write of a
// multiPrimTurn.
double comparisonsPerWrite(size_t layerPrims, size_t prims, size_t repeats) {
    return double(multiPrimTurn(layerPrims, prims, repeats).comparisons) / double(prims * 3 * repeats);
}
Cost wideSpecTurn(size_t count) {
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
    const auto work = metered([&] { layer.turn([&](SdfAbstractData& data) { data.Set(path, last, VtValue(-2)); }); });
    require(layer.projection->Get(path, last) == VtValue(-2) && layer.projection->List(path).size() == count &&
            !layer.projection->hasPendingLocalEdits(),
        "the wide spec's turn was not published");
    return work;
}
// A turn that writes one field of a spec with F fields verifies that spec
// with work linear in F, however many fields it carries.
void wideSpecVerification() {
    const size_t small = 512, large = 8192;
    const auto a = wideSpecTurn(small).work, b = wideSpecTurn(large).work;
    std::cout << "footprint_work wide spec turn fields=" << small << " work=" << a << "; fields=" << large
              << " work=" << b << std::endl;
    require(b <= 2 * (large / small) * a + 1024,
        "a wide spec's turn work grows faster than its field count: " + std::to_string(a) + " at " +
            std::to_string(small) + " fields, " + std::to_string(b) + " at " + std::to_string(large) + " fields");
}
std::pair<Cost, Cost> widePeerUpdates(size_t count) {
    EnvelopeInput metadata;
    metadata.commands.push_back(create(2, 1, "P", SdfSpecTypePrim));
    Layer layer(buildSceneCandidate({}, encodeEnvelope(metadata)));
    const auto update = [&](bool created, int offset) {
        EnvelopeInput fields;
        DecodedValues values;
        if (created) fields.commands.push_back(create(3, 1, "Q", SdfSpecTypePrim));
        for (size_t i = 0; i < count; ++i) {
            fields.fields.push_back({3, 0, "f" + std::to_string(i), {}});
            values.fields.push_back(VtValue(int(i) + offset));
        }
        return encodeUsdEnvelope(fields, values);
    };
    const auto created = update(true, 0), changed = update(false, 1);
    const auto write = metered([&] { layer.peer(created); });
    const auto change = metered([&] { layer.peer(changed); });
    const SdfPath path("/Q");
    require(layer.projection->List(path).size() == count &&
            layer.projection->Get(path, TfToken("f" + std::to_string(count - 1))) == VtValue(int(count)),
        "the wide peer updates were not published");
    return {write, change};
}
std::pair<Cost, Cost> wideInstall(size_t count) {
    EnvelopeInput metadata;
    DecodedValues values;
    metadata.commands.push_back(create(2, 1, "Q", SdfSpecTypePrim));
    for (size_t i = 0; i < count; ++i) {
        metadata.fields.push_back({2, 0, "f" + std::to_string(i), {}});
        values.fields.push_back(VtValue(int(i)));
    }
    WireSceneLimits limits;
    limits.maxWork = size_t(1) << 40;
    const auto scene = buildSceneCandidate({}, encodeUsdEnvelope(metadata, values), limits);
    const auto storage = PXR_NS::TfCreateRefPtr(new HostSceneStorage);
    const auto empty = metered([&] { storage->CopyFrom(scene->data()); });
    for (size_t i = 0; i < count; i += 2)
        storage->Set(SdfPath("/Q"), TfToken("f" + std::to_string(i)), VtValue(-1));
    const auto over = metered([&] { storage->CopyFrom(scene->data()); });
    require(storage->List(SdfPath("/Q")).size() == count &&
            storage->Get(SdfPath("/Q"), TfToken("f0")) == VtValue(0),
        "the wide install was not copied");
    return {empty, over};
}
// Publishing a peer update that writes or changes the F fields of one spec,
// and installing a scene with such a spec, cost work linear in F.
void widePeerPublication() {
    const size_t small = 512, large = 8192;
    const auto works = [](const std::pair<Cost, Cost>& costs) {
        return std::pair{costs.first.work, costs.second.work};
    };
    const auto a = works(widePeerUpdates(small)), b = works(widePeerUpdates(large));
    std::cout << "footprint_work wide peer update fields=" << small << " write=" << a.first << " change=" << a.second
              << "; fields=" << large << " write=" << b.first << " change=" << b.second << std::endl;
    const auto c = works(wideInstall(small)), d = works(wideInstall(large));
    std::cout << "footprint_work wide install fields=" << small << " empty=" << c.first << " over=" << c.second
              << "; fields=" << large << " empty=" << d.first << " over=" << d.second << std::endl;
    for (const auto& [x, y, name] : {std::tuple{a.first, b.first, "write"}, std::tuple{a.second, b.second, "change"},
             std::tuple{c.first, d.first, "install"}, std::tuple{c.second, d.second, "install over"}})
        require(y <= 2 * (large / small) * x + 1024,
            std::string("a wide spec's peer ") + name + " work grows faster than its field count: " +
                std::to_string(x) + " at " + std::to_string(small) + " fields, " + std::to_string(y) + " at " +
                std::to_string(large) + " fields");
}
} // namespace omni::usd::live::test::footprint
using namespace omni::usd::live::test::footprint;
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
    widePeerPublication();
    const std::array<size_t, 3> sizes{50, 1000, 4000};
    std::array<Work, 3> work;
    for (size_t i = 0; i < sizes.size(); ++i) {
        work[i] = measure(sizes[i]);
        std::cout << "footprint_work prims=" << sizes[i] << " field_turns=" << work[i].field.work
                  << " sample_turn=" << work[i].samples.work << " peer_publication=" << work[i].peer.work
                  << " peer_structure=" << work[i].structure.work << " create=" << work[i].create.work
                  << " rename=" << work[i].rename.work << " reparent=" << work[i].reparent.work
                  << " reorder=" << work[i].reorder.work << " delete=" << work[i].erase.work << std::endl;
    }
    const auto bounded = [&](const Cost& smallCost, const Cost& largeCost, const char* what) {
        const auto small = smallCost.work, large = largeCost.work;
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
