// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "WireScene.h"
#include "SceneWork.h"
#include <pxr/base/vt/dictionary.h>
#include <pxr/usd/sdf/listOp.h>
#include <pxr/usd/sdf/schema.h>
#include <algorithm>
#include <array>
#include <future>
#include <chrono>
#include <iostream>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <tuple>

PXR_NAMESPACE_USING_DIRECTIVE
using namespace omni::usd::live;
using namespace omni::usd::live::wire;
namespace {
std::string context;
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(context+": "+message);
}
struct Record {
    SdfSpecType type;
    std::map<TfToken,VtValue> fields;
    bool operator==(const Record& other) const { return type == other.type && fields == other.fields; }
};
using State = std::map<SdfPath,Record>;
State inspect(const WireScenePtr& scene) {
    struct Visitor : SdfAbstractDataSpecVisitor {
        State state;
        bool VisitSpec(const SdfAbstractData& data, const SdfPath& path) override {
            Record record{data.GetSpecType(path),{}};
            for (const auto& key : data.List(path)) record.fields.emplace(key,data.Get(path,key));
            state.emplace(path,std::move(record)); return true;
        }
        void Done(const SdfAbstractData&) override {}
    } visitor;
    scene->data()->VisitSpecs(&visitor); return visitor.state;
}
Command create(uint64_t id, uint64_t parent, const std::string& name, SdfSpecType type = SdfSpecTypePrim) {
    Command c{}; c.kind = Command::Kind::Create; c.section = id; c.parent = parent; c.name = name; c.sectionType = uint8_t(type); return c;
}
Command diff(uint64_t id, uint64_t parent, const std::string& name, uint32_t order, SdfSpecType type = SdfSpecTypePrim) {
    auto c = create(id,parent,name,type); c.kind = Command::Kind::DiffChange; c.sectionOrder = order; return c;
}
Command move(uint64_t id, uint64_t oldParent, uint64_t parent, const std::string& name) {
    Command c{}; c.kind = Command::Kind::Move; c.section = id; c.oldParent = oldParent; c.parent = parent; c.name = name; return c;
}
Command erase(uint64_t id, uint64_t parent = 0, bool isDiff = false) {
    Command c{}; c.kind = isDiff ? Command::Kind::DiffDelete : Command::Kind::Delete; c.section = id; c.parent = parent; return c;
}
Command reorder(uint64_t parent, std::vector<uint64_t> ids, uint8_t list = 0) {
    Command c{}; c.kind = Command::Kind::Reorder; c.section = parent; c.childList = list; c.children = std::move(ids); return c;
}
struct Batch {
    EnvelopeInput metadata;
    DecodedValues values;
    void field(uint64_t id, const TfToken& key, const VtValue& value, uint64_t order = 0) {
        metadata.fields.push_back({id,order,key.GetString(),{}}); values.fields.push_back(value);
    }
    void sample(uint64_t id, double time, const VtValue& value) {
        metadata.samples.push_back({id,time,{}}); values.samples.push_back(value);
    }
    std::string bytes() const { return encodeUsdEnvelope(metadata,values); }
};
WireScenePtr candidate(const WireScenePtr& previous, const Batch& batch, const WireSceneLimits& limits = {}) {
    return buildSceneCandidate(previous,batch.bytes(),limits);
}
WireScenePtr delivered(const WireScenePtr& previous, const Batch& batch) {
    return buildSceneCandidate(previous,batch.bytes(),{},nullptr,WireSceneBatchPolicy::Delivered);
}
VtValue tokens(std::initializer_list<const char*> names) {
    std::vector<TfToken> result; for (const auto* name : names) result.emplace_back(name); return VtValue(result);
}
VtValue paths(std::initializer_list<const char*> names) {
    SdfPathVector result; for (const auto* name : names) result.emplace_back(name); return VtValue(result);
}
void field(const WireScenePtr& scene, const char* path, const TfToken& key, const VtValue& expected) {
    const SdfPath p(path); check(scene->data()->Has(p,key),"expected field absent");
    check(scene->data()->Get(p,key) == expected,"authored/derived field differs from independent expectation");
}
void absent(const WireScenePtr& scene, const char* path, const TfToken& key) {
    check(!scene->data()->Has(SdfPath(path),key),"erased field remains present");
}
void indexed(const WireScenePtr& scene, uint64_t id, const std::string& path, SdfSpecType type) {
    check(scene->pathForSection(id) == SdfPath(path),"ID-to-path index differs");
    check(scene->sectionForPath(SdfPath(path)) == id,"path-to-ID index differs");
    check(scene->data()->HasSpec(SdfPath(path)) && scene->data()->GetSpecType(SdfPath(path)) == type,"Sdf spec/type differs");
}
void missing(const WireScenePtr& scene, uint64_t id, const char* path) {
    check(scene->pathForSection(id).IsEmpty() && scene->sectionForPath(SdfPath(path)) == 0 &&
        !scene->data()->HasSpec(SdfPath(path)),"deleted section/path remains indexed");
}
void gone(const WireScenePtr& scene, uint64_t id) {
    check(scene->pathForSection(id).IsEmpty(),"removed section remains indexed");
}
void rejectedBytes(const WireScenePtr& previous, const std::string& bytes, std::optional<ErrorKind> kind = {}, const WireSceneLimits& limits = {}) {
    const auto before = inspect(previous); const auto count = previous->sectionCount();
    const auto format = previous->sourceFormat();
    WireScenePtr published = previous;
    bool rejected = false;
    try { published = buildSceneCandidate(previous,bytes,limits); }
    catch (const DecodeError& error) { rejected = true; if (kind) check(error.kind == *kind,"wrong scene rejection kind"); }
    check(rejected,"invalid/over-budget candidate was accepted");
    check(published == previous && inspect(previous) == before && previous->sectionCount() == count && previous->sourceFormat() == format,
        "rejected candidate changed prior owner/data");
    for (const auto& [path,record] : before) {
        const auto id = previous->sectionForPath(path);
        check(id && previous->pathForSection(id) == path,"failed candidate corrupted prior identity indexes");
    }
}
void rejected(const WireScenePtr& previous, const Batch& batch, std::optional<ErrorKind> kind = {}, const WireSceneLimits& limits = {}) {
    rejectedBytes(previous,batch.bytes(),kind,limits);
}
void topology() {
    context = "all child policies and subtree identity";
    Batch batch;
    batch.metadata.commands = {create(2,1,"P"),create(3,2,"a",SdfSpecTypeAttribute),
        create(4,3,"/Target.a",SdfSpecTypeConnection),create(5,3,"expression",SdfSpecTypeExpression),
        create(6,3,"Target.a",SdfSpecTypeMapper),create(7,6,"arg",SdfSpecTypeMapperArg),
        create(8,2,"rel",SdfSpecTypeRelationship),create(9,8,"/Target",SdfSpecTypeRelationshipTarget),
        create(10,9,"weight",SdfSpecTypeAttribute),create(11,8,"expression",SdfSpecTypeExpression),
        create(12,2,"v",SdfSpecTypeVariantSet),create(13,12,"x",SdfSpecTypeVariant),
        create(14,13,"Child"),create(15,13,"inside",SdfSpecTypeAttribute),create(16,13,"rel",SdfSpecTypeRelationship),
        create(17,13,"nested",SdfSpecTypeVariantSet),create(18,17,"y",SdfSpecTypeVariant),create(19,2,"Sub")};
    batch.field(10,TfToken("default"),VtValue(17)); batch.field(18,TfToken("comment"),VtValue(std::string("nested")));
    const auto scene = candidate({},batch); const auto before = inspect(scene);
    const auto verify = [](const WireScenePtr& scene, const std::string& p) {
        indexed(scene,1,"/",SdfSpecTypePseudoRoot); indexed(scene,2,p,SdfSpecTypePrim);
        const std::vector<std::tuple<uint64_t,std::string,SdfSpecType>> expected = {
            {3,p+".a",SdfSpecTypeAttribute},{4,p+".a[/Target.a]",SdfSpecTypeConnection},
            {5,p+".a.expression",SdfSpecTypeExpression},{6,p+".a.mapper[/P/Target.a]",SdfSpecTypeMapper},
            {7,p+".a.mapper[/P/Target.a].arg",SdfSpecTypeMapperArg},{8,p+".rel",SdfSpecTypeRelationship},
            {9,p+".rel[/Target]",SdfSpecTypeRelationshipTarget},{10,p+".rel[/Target].weight",SdfSpecTypeAttribute},
            {11,p+".rel.expression",SdfSpecTypeExpression},{12,p+"{v=}",SdfSpecTypeVariantSet},{13,p+"{v=x}",SdfSpecTypeVariant},
            {14,p+"{v=x}Child",SdfSpecTypePrim},{15,p+"{v=x}.inside",SdfSpecTypeAttribute},{16,p+"{v=x}.rel",SdfSpecTypeRelationship},
            {17,p+"{v=x}{nested=}",SdfSpecTypeVariantSet},{18,p+"{v=x}{nested=y}",SdfSpecTypeVariant},{19,p+"/Sub",SdfSpecTypePrim}};
        for (const auto& [id,path,type] : expected) indexed(scene,id,path,type);
        check(scene->sectionCount() == 19,"wrong all-policy section count");
    };
    verify(scene,"/P");
    const std::array<uint32_t,19> storedOrders{{0,0,0,0,1,2,0,1,0,0,1,2,0,0,1,2,3,0,3}};
    size_t graphCount = 0;
    check(scene->visitSections(19,[&](const WireSectionView& section) {
        check(section.id == ++graphCount,"section visitor omitted/reordered a stored ID");
        check(section.order == storedOrders[section.id-1],"section visitor changed cross-type stored order");
        if (section.id == 1) {
            check(section.parent == 0 && section.type == SdfSpecTypePseudoRoot && section.name.empty() &&
                  section.path == SdfPath::AbsoluteRootPath(),"section visitor changed root identity");
        } else {
            const auto& authored = batch.metadata.commands[section.id-2];
            const auto name = section.id == 6 ? std::string("/P/Target.a") : authored.name;
            check(section.parent == authored.parent && section.type == SdfSpecType(authored.sectionType) &&
                  section.name == name && section.path == scene->pathForSection(section.id),
                  "section visitor lost parent/type/stored name/path");
        }
        return true;
    }) && graphCount == 19,"section visitor did not finish all spec types");
    field(scene,"/",SdfChildrenKeys->PrimChildren,tokens({"P"}));
    field(scene,"/P",SdfChildrenKeys->PrimChildren,tokens({"Sub"}));
    field(scene,"/P",SdfChildrenKeys->PropertyChildren,tokens({"a","rel"}));
    field(scene,"/P",SdfChildrenKeys->VariantSetChildren,tokens({"v"}));
    field(scene,"/P.a",SdfChildrenKeys->ConnectionChildren,paths({"/Target.a"}));
    field(scene,"/P.a",SdfChildrenKeys->ExpressionChildren,tokens({"expression"}));
    field(scene,"/P.a",SdfChildrenKeys->MapperChildren,paths({"/P/Target.a"}));
    field(scene,"/P.a.mapper[/P/Target.a]",SdfChildrenKeys->MapperArgChildren,tokens({"arg"}));
    field(scene,"/P.rel",SdfChildrenKeys->RelationshipTargetChildren,paths({"/Target"}));
    field(scene,"/P.rel",SdfChildrenKeys->ExpressionChildren,tokens({"expression"}));
    field(scene,"/P.rel[/Target]",SdfChildrenKeys->PropertyChildren,tokens({"weight"}));
    field(scene,"/P{v=}",SdfChildrenKeys->VariantChildren,tokens({"x"}));
    field(scene,"/P{v=x}",SdfChildrenKeys->PrimChildren,tokens({"Child"}));
    field(scene,"/P{v=x}",SdfChildrenKeys->PropertyChildren,tokens({"inside","rel"}));
    field(scene,"/P{v=x}",SdfChildrenKeys->VariantSetChildren,tokens({"nested"}));
    field(scene,"/P{v=x}{nested=}",SdfChildrenKeys->VariantChildren,tokens({"y"}));
    Batch rename; rename.metadata.commands = {move(2,1,1,"Q")};
    const auto moved = candidate(scene,rename); verify(moved,"/Q");
    field(moved,"/Q.a",SdfChildrenKeys->MapperChildren,paths({"/P/Target.a"}));
    field(moved,"/Q.rel[/Target].weight",TfToken("default"),VtValue(17));
    field(moved,"/Q{v=x}{nested=y}",TfToken("comment"),VtValue(std::string("nested")));
    check(inspect(scene) == before && moved->sectionForPath(SdfPath("/P")) == 0,"move mutated old owner or left old paths");
    Batch remove; remove.metadata.commands = {erase(12,2)};
    const auto deleted = candidate(scene,remove);
    for (uint64_t id = 12; id <= 18; ++id) check(deleted->pathForSection(id).IsEmpty(),"variant graph descendant survived normal delete");
    check(deleted->sectionCount() == 12,"variant subtree deletion removed wrong graph members");
    absent(deleted,"/P",SdfChildrenKeys->VariantSetChildren);
    indexed(deleted,19,"/P/Sub",SdfSpecTypePrim);
}
void sectionEnumeration() {
    context = "bounded complete section enumeration";
    Batch batch; batch.metadata.isDiff = true;
    batch.metadata.commands = {diff(90,1,"Later",UINT32_MAX),diff(42,1,"Earlier",7),
        diff(77,42,"value",19,SdfSpecTypeAttribute),diff(12,42,"Nested",3)};
    const auto scene = candidate({},batch);
    using Item = std::tuple<uint64_t,uint64_t,SdfSpecType,std::string,uint32_t,std::string>;
    // A diff's order is a sibling index, so each section's order is its position among its parent's children.
    const std::vector<Item> expected = {
        {1,0,SdfSpecTypePseudoRoot,"",0,"/"},
        {12,42,SdfSpecTypePrim,"Nested",0,"/Earlier/Nested"},
        {42,1,SdfSpecTypePrim,"Earlier",0,"/Earlier"},
        {77,42,SdfSpecTypeAttribute,"value",1,"/Earlier.value"},
        {90,1,SdfSpecTypePrim,"Later",1,"/Later"}};
    const auto read = [](const WireScenePtr& owner) {
        std::vector<Item> result;
        check(owner->visitSections(5,[&](const WireSectionView& section) {
            result.emplace_back(section.id,section.parent,section.type,std::string(section.name),
                                section.order,section.path.GetString());
            return true;
        }),"complete enumeration stopped");
        return result;
    };
    check(read(scene) == expected,"enumeration normalized identities/order or used traversal order");
    size_t visited = 0; bool rejected = false;
    try { scene->visitSections(4,[&](const WireSectionView&) { ++visited; return true; }); }
    catch (const DecodeError& error) { rejected = error.kind == ErrorKind::ResourceLimit; }
    check(rejected && visited == 0,"section limit failed after partial callback delivery");
    check(!scene->visitSections(5,[&](const WireSectionView&) { ++visited; return false; }) && visited == 1,
          "visitor stop was ignored");
    rejected = false;
    try { scene->visitSections(5,{}); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected,"empty section visitor accepted");
    rejected = false;
    try { scene->visitSections(5,[](const WireSectionView&)->bool { throw 17; }); }
    catch (int value) { rejected = value == 17; }
    check(rejected && read(scene) == expected,"visitor exception changed original scene");
    Batch moved; moved.metadata.commands = {move(42,1,1,"Renamed")};
    const auto next = candidate(scene,moved);
    check(next->pathForSection(77) == SdfPath("/Renamed.value") && read(scene) == expected,
          "later candidate changed archived original graph");
}
void deletionDiffAndOrdering() {
    context = "normal/diff deletion, reset and ordering";
    Batch initial; initial.metadata.commands = {create(2,1,"A"),create(3,2,"Child"),create(4,3,"Grand"),create(5,1,"B")};
    initial.field(4,TfToken("value"),VtValue(91));
    const auto old = candidate({},initial); const auto oldState = inspect(old);
    Batch normal; normal.metadata.commands = {erase(2,1)};
    const auto removed = candidate(old,normal);
    missing(removed,2,"/A"); missing(removed,3,"/A/Child"); missing(removed,4,"/A/Child/Grand");
    field(removed,"/",SdfChildrenKeys->PrimChildren,tokens({"B"}));
    Batch delta; delta.metadata.isDiff = true; delta.metadata.baseVersion = 10; delta.metadata.sourceFormat = "usda";
    delta.metadata.commands = {erase(2,0,true),diff(3,5,"Saved",7)};
    const auto saved = candidate(old,delta);
    missing(saved,2,"/A"); indexed(saved,3,"/B/Saved",SdfSpecTypePrim); indexed(saved,4,"/B/Saved/Grand",SdfSpecTypePrim);
    field(saved,"/B/Saved/Grand",TfToken("value"),VtValue(91)); check(saved->sourceFormat() == "usda","diff sourceFormat lost");
    Batch reset; reset.metadata.isDiff = true; reset.metadata.sourceFormat = "usdc";
    const auto empty = candidate(saved,reset);
    check(empty->sectionCount() == 1 && inspect(empty) == State{{SdfPath("/"),{SdfSpecTypePseudoRoot,{}}}},"base-zero reset retained old graph/fields");
    check(empty->sourceFormat() == "usdc" && inspect(old) == oldState,"reset format/old owner changed");
    Batch root; root.metadata.isDiff = true; root.metadata.commands = {diff(1,0,"",0,SdfSpecTypePseudoRoot)};
    check(candidate(saved,root)->sectionCount() == 1,"explicit reset root created duplicate spec");
    Batch siblings; siblings.metadata.commands = {create(2,1,"A"),create(3,1,"B"),create(4,1,"C"),create(5,1,"D")};
    const auto ordered = candidate({},siblings);
    Batch complete; complete.metadata.commands = {create(6,1,"E"),reorder(1,{2,5,4,3,6})};
    field(candidate(ordered,complete),"/",SdfChildrenKeys->PrimChildren,tokens({"A","D","C","B","E"}));
    Batch order; order.metadata.isDiff = true; order.metadata.baseVersion = 1;
    order.metadata.commands = {diff(5,1,"D",40),diff(2,1,"A",30),diff(4,1,"C",20),diff(3,1,"B",10)};
    field(candidate(ordered,order),"/",SdfChildrenKeys->PrimChildren,tokens({"B","C","A","D"}));
    Batch swap; swap.metadata.isDiff = true; swap.metadata.baseVersion = 1;
    swap.metadata.commands = {diff(2,1,"B",0),diff(3,1,"A",1)};
    const auto swapped = candidate(ordered,swap);
    indexed(swapped,2,"/B",SdfSpecTypePrim); indexed(swapped,3,"/A",SdfSpecTypePrim);
}
void wholeListReorder() {
    context = "incoming reorder orders the parent's children as USD does";
    Batch start; start.metadata.commands = {create(2,1,"P"),create(7,1,"Q"),create(3,2,"A"),create(4,2,"B"),
        create(5,2,"C"),create(6,2,"D"),create(8,2,"attr",SdfSpecTypeAttribute),create(9,4,"Child")};
    start.field(4,TfToken("value"),VtValue(4));
    const auto base = candidate({},start); const auto before = inspect(base);
    check(base->pathCount() == 9,"path count differs from projected sections");
    // The list spans every child list of the parent; the child-list ID is ignored.
    Batch full; full.metadata.commands = {reorder(2,{6,5,8,4,3})};
    const auto reversed = candidate(base,full);
    field(reversed,"/P",SdfChildrenKeys->PrimChildren,tokens({"D","C","B","A"}));
    field(reversed,"/P",SdfChildrenKeys->PropertyChildren,tokens({"attr"}));
    // A partial list orders the children it names as USD's SdfApplyListOrdering does: each named child, in list
    // order, moves to the end with the unnamed siblings of its child list that follow it, and unnamed children
    // before the first named one stay in front. No child leaves the scene.
    Batch subset; subset.metadata.commands = {reorder(2,{5,3})};
    const auto partial = candidate(base,subset);
    field(partial,"/P",SdfChildrenKeys->PrimChildren,tokens({"C","D","A","B"}));
    field(partial,"/P",SdfChildrenKeys->PropertyChildren,tokens({"attr"}));
    indexed(partial,4,"/P/B",SdfSpecTypePrim); indexed(partial,9,"/P/B/Child",SdfSpecTypePrim);
    field(partial,"/P/B",TfToken("value"),VtValue(4));
    check(partial->sectionCount() == base->sectionCount() && partial->pathCount() == base->pathCount(),
          "a partial reorder dropped or hid a child");
    Batch leading; leading.metadata.commands = {reorder(2,{6,4})};
    field(candidate(base,leading),"/P",SdfChildrenKeys->PrimChildren,tokens({"A","D","B","C"}));
    // An empty list names nothing and changes nothing.
    Batch none; none.metadata.commands = {reorder(2,{})};
    check(inspect(candidate(base,none)) == inspect(base),"an empty reorder changed the scene");
    // Local content never names a section that is not a current child. A delivered list that does repeats
    // a commit the scene holds; the current children it names take its order.
    Batch nonmember; nonmember.metadata.commands = {reorder(2,{3,99})}; rejected(base,nonmember,ErrorKind::Malformed);
    auto reordered = [&](std::vector<Command> commands) {
        Batch batch; batch.metadata.commands = std::move(commands); return delivered(base,batch); };
    for (const auto& scene : {reordered({reorder(2,{3,99})}), reordered({reorder(2,{3,7})})})
        check(inspect(scene) == inspect(base),"a reorder naming only the first child changed the scene");
    check(inspect(reordered({reorder(99,{3})})) == inspect(base),"reorder of a missing parent changed the scene");
    { const auto stale = reordered({move(4,2,7,"B"),erase(5,2),reorder(2,{6,5,4,3,8})});
      field(stale,"/P",SdfChildrenKeys->PrimChildren,tokens({"D","A"}));
      field(stale,"/P",SdfChildrenKeys->PropertyChildren,tokens({"attr"})); indexed(stale,4,"/Q/B",SdfSpecTypePrim); }
    Batch emptyMissing; emptyMissing.metadata.commands = {reorder(99,{})};
    check(inspect(candidate(base,emptyMissing)) == inspect(base),"empty reorder of a missing parent changed the scene");
    // An unchanged order leaves an absent child field absent.
    Batch absentField; absentField.metadata.commands = {create(2,1,"P")};
    const auto lone = candidate({},absentField);
    Batch same; same.metadata.commands = {reorder(1,{2})};
    check(inspect(candidate(lone,same)) == inspect(lone),"unchanged reorder rewrote the child field");
    check(inspect(base) == before,"reorder mutated retained original owner");
}
// One update holding reorder(parent, ids), where ids may repeat or name any section: the encoder refuses both, so
// distinct placeholder IDs are encoded and each is then replaced in place by the ID it stands for.
std::string reorderBytes(uint64_t parent, const std::vector<uint64_t>& ids, uint8_t list = 0) {
    std::vector<uint64_t> placeholders;
    for (size_t i = 0; i < ids.size(); ++i) placeholders.push_back(0x5a17c0de00000000ull + i);
    Batch batch; batch.metadata.commands = {reorder(parent,placeholders,list)};
    auto bytes = batch.bytes();
    const auto little = [](uint64_t value) {
        std::string result(8,'\0');
        for (size_t i = 0; i < 8; ++i) result[i] = char((value >> (8*i)) & 0xff);
        return result;
    };
    for (size_t i = 0; i < ids.size(); ++i) {
        const auto from = little(placeholders[i]); const auto at = bytes.find(from);
        check(at != std::string::npos && bytes.find(from,at+1) == std::string::npos,"reorder placeholder is not unique");
        bytes.replace(at,8,little(ids[i]));
    }
    return bytes;
}
WireScenePtr reordered(const WireScenePtr& previous, const std::string& bytes, WireSceneBatchPolicy policy) {
    return buildSceneCandidate(previous,bytes,{},nullptr,policy);
}
// Every delivered or replayed Reorder equals USD's SdfApplyListOrdering applied to each child list on its own, with
// the command's whole list as the order: USD ignores the IDs that are not members of that list (other child lists,
// non-children, the parent itself, the root, unknown IDs) and every repeat after an ID's first occurrence. Seeded
// parents interleave prims, attributes, relationships and variant sets; seeded lists are empty, partial or full and
// hold repeats and non-members. The order of the positions a child list holds among the parent's children is kept.
void usdListOrderingDifferential(uint64_t seed) {
    context = "per-list reorder equals SdfApplyListOrdering on each child list";
    std::mt19937_64 random(seed ^ 0x6f726465726c7374ull);
    const SdfSpecType kinds[] = {SdfSpecTypePrim, SdfSpecTypeAttribute, SdfSpecTypeRelationship, SdfSpecTypeVariantSet};
    const auto listOf = [](SdfSpecType type) {
        return type == SdfSpecTypePrim ? 0u : type == SdfSpecTypeVariantSet ? 2u : 1u;
    };
    // A Fisher-Yates shuffle drawing only the engine's fixed output.
    const auto shuffle = [&](std::vector<uint64_t>& items) {
        for (size_t i = items.size(); i > 1; --i) std::swap(items[i-1],items[random() % i]);
    };
    const TfToken fields[] = {SdfChildrenKeys->PrimChildren, SdfChildrenKeys->PropertyChildren,
        SdfChildrenKeys->VariantSetChildren};
    const auto oracle = [&](const WireScenePtr& base, const std::map<uint64_t,SdfSpecType>& types,
                            const std::vector<uint64_t>& ids) {
        auto mixed = base->listedChildren(2);
        std::vector<std::string> order;
        for (auto id : ids) order.push_back(std::to_string(id));
        for (unsigned list = 0; list < 3; ++list) {
            std::vector<size_t> at; std::vector<std::string> members;
            for (size_t i = 0; i < mixed.size(); ++i)
                if (listOf(types.at(mixed[i])) == list) { at.push_back(i); members.push_back(std::to_string(mixed[i])); }
            SdfApplyListOrdering(&members,order);
            for (size_t k = 0; k < at.size(); ++k) mixed[at[k]] = std::stoull(members[k]);
        }
        return mixed;
    };
    const auto agrees = [&](const WireScenePtr& scene, const std::map<uint64_t,SdfSpecType>& types,
                            const std::vector<uint64_t>& expected) {
        if (scene->listedChildren(2) != expected) return false;
        for (unsigned list = 0; list < 3; ++list) {
            std::vector<TfToken> names;
            for (auto id : expected) if (listOf(types.at(id)) == list) names.emplace_back("c" + std::to_string(id));
            const auto value = scene->data()->Get(SdfPath("/P"),fields[list]);
            if (names.empty() ? !value.IsEmpty() : value != VtValue(names)) return false;
        }
        return true;
    };
    // The issue's example: children [A, x, B, y] ordered [B, A] give [B, x, A, y].
    {
        Batch start; start.metadata.commands = {create(2,1,"P"),create(10,2,"c10"),create(11,2,"c11",SdfSpecTypeAttribute),
            create(12,2,"c12"),create(13,2,"c13",SdfSpecTypeAttribute)};
        const auto base = candidate({},start);
        const std::map<uint64_t,SdfSpecType> types{{10,SdfSpecTypePrim},{11,SdfSpecTypeAttribute},{12,SdfSpecTypePrim},
            {13,SdfSpecTypeAttribute}};
        const std::vector<uint64_t> ids{12,10};
        check(oracle(base,types,ids) == std::vector<uint64_t>{12,11,10,13},"the oracle differs from the issue's example");
        check(agrees(reordered(base,reorderBytes(2,ids),WireSceneBatchPolicy::Delivered),types,{12,11,10,13}),
              "the issue's example differs from USD's per-list ordering");
    }
    size_t cases = 0, repeats = 0, foreign = 0, full = 0, moved = 0;
    for (int round = 0; round < 6000; ++round) {
        Batch start; start.metadata.commands = {create(2,1,"P"),create(3,1,"Q"),create(4,3,"Elsewhere")};
        std::map<uint64_t,SdfSpecType> types;
        std::vector<uint64_t> children;
        const size_t count = random() % 10;
        for (size_t i = 0; i < count; ++i) {
            const uint64_t id = 10 + i; const auto type = kinds[random() % 4];
            start.metadata.commands.push_back(create(id,2,"c" + std::to_string(id),type));
            types[id] = type; children.push_back(id);
        }
        auto base = candidate({},start);
        if (count && random() % 2) {
            // A full reorder first, so the kinds' positions are not the creation order.
            auto shuffled = children; shuffle(shuffled);
            base = reordered(base,reorderBytes(2,shuffled),WireSceneBatchPolicy::Delivered);
        }
        std::vector<uint64_t> ids;
        if (count && random() % 4 == 0) {
            ids = children; shuffle(ids); ++full;
        }
        const std::vector<uint64_t> others{0,1,2,3,4,999};
        const size_t extra = random() % 12;
        for (size_t i = 0; i < extra; ++i) {
            const bool member = count && random() % 4 != 0;
            const auto id = member ? children[random() % count] : others[random() % others.size()];
            ids.insert(ids.begin() + ptrdiff_t(random() % (ids.size() + 1)),id);
        }
        const std::set<uint64_t> distinct(ids.begin(),ids.end());
        repeats += distinct.size() != ids.size();
        foreign += std::any_of(ids.begin(),ids.end(),[&](uint64_t id) { return !types.count(id); });
        const auto policy = round % 2 ? WireSceneBatchPolicy::Delivered : WireSceneBatchPolicy::PendingReplay;
        const auto expected = oracle(base,types,ids);
        moved += expected != base->listedChildren(2);
        const auto scene = reordered(base,reorderBytes(2,ids,uint8_t(random() % 3)),policy);
        if (!agrees(scene,types,expected)) {
            std::ostringstream text; text << "round " << round << " differs from USD's per-list ordering: ids";
            for (auto id : ids) text << ' ' << id;
            check(false,text.str().c_str());
        }
        ++cases;
    }
    std::cout << "usd_list_ordering_differential seed=" << seed << " cases=" << cases << " repeats=" << repeats
              << " non_members=" << foreign << " full=" << full << " moved=" << moved << std::endl;
    check(repeats > 1000 && foreign > 1000 && full > 500 && moved > 2000,"differential cases exercised too few shapes");
}
void perListReorder() {
    context = "incoming reorder orders each child list it names on its own";
    // P's children interleave prims A and B, properties x and y, and variant set v.
    Batch start; start.metadata.commands = {create(2,1,"P"),create(3,2,"A"),create(4,2,"x",SdfSpecTypeAttribute),
        create(5,2,"B"),create(6,2,"y",SdfSpecTypeRelationship),create(7,2,"v",SdfSpecTypeVariantSet)};
    const auto base = candidate({},start); const auto before = inspect(base);
    auto lists = [&](const WireScenePtr& scene, VtValue prims, VtValue properties, std::vector<uint64_t> listed) {
        field(scene,"/P",SdfChildrenKeys->PrimChildren,prims);
        field(scene,"/P",SdfChildrenKeys->PropertyChildren,properties);
        field(scene,"/P",SdfChildrenKeys->VariantSetChildren,tokens({"v"}));
        check(scene->listedChildren(2) == listed,"a child list left the positions it holds among the parent's children");
        check(scene->sectionCount() == base->sectionCount() && scene->pathCount() == base->pathCount(),
              "a reorder dropped or hid a child");
    };
    // A prim order moves only prims, within the positions prims hold; the properties keep their order, as USD's
    // primOrder leaves properties alone.
    Batch prims; prims.metadata.commands = {reorder(2,{5,3})};
    const auto primOrdered = candidate(base,prims);
    lists(primOrdered,tokens({"B","A"}),tokens({"x","y"}),{5,4,3,6,7});
    // The property mirror: propertyOrder leaves the prims alone. The child-list ID plays no part.
    Batch properties; properties.metadata.commands = {reorder(2,{6,4},1)};
    lists(delivered(base,properties),tokens({"A","B"}),tokens({"y","x"}),{3,6,5,4,7});
    Batch unlabelled; unlabelled.metadata.commands = {reorder(2,{6,4},0)};
    lists(candidate(base,unlabelled),tokens({"A","B"}),tokens({"y","x"}),{3,6,5,4,7});
    // The interleaving survives a reorder, so later partial reorders still order each list on its own.
    Batch propertiesAfter; propertiesAfter.metadata.commands = {reorder(2,{6,4},1)};
    const auto bothOrdered = candidate(primOrdered,propertiesAfter);
    lists(bothOrdered,tokens({"B","A"}),tokens({"y","x"}),{5,6,3,4,7});
    Batch primsBack; primsBack.metadata.commands = {reorder(2,{3})};
    lists(candidate(bothOrdered,primsBack),tokens({"B","A"}),tokens({"y","x"}),{5,6,3,4,7});
    Batch primsSwap; primsSwap.metadata.commands = {reorder(2,{3,5})};
    lists(candidate(bothOrdered,primsSwap),tokens({"A","B"}),tokens({"y","x"}),{3,6,5,4,7});
    // A full list naming both kinds orders each list as the full list does, whatever its interleaving; each kind
    // keeps its positions.
    for (auto ids : {std::vector<uint64_t>{5,3,6,4,7}, std::vector<uint64_t>{6,4,7,5,3}, std::vector<uint64_t>{7,6,5,4,3}}) {
        Batch full; full.metadata.commands = {reorder(2,ids)};
        lists(candidate(base,full),tokens({"B","A"}),tokens({"y","x"}),{5,6,3,4,7});
    }
    // A list naming some members of both kinds orders each named list by its own named members.
    Batch mixed; mixed.metadata.commands = {reorder(2,{6,5,3})};
    lists(candidate(base,mixed),tokens({"B","A"}),tokens({"x","y"}),{5,4,3,6,7});
    // A delivered or replayed list applies exactly as USD does: a repeat after a child's first occurrence, and IDs
    // that are not current children, are ignored. Local content stays strict.
    for (const auto policy : {WireSceneBatchPolicy::Delivered, WireSceneBatchPolicy::PendingReplay}) {
        lists(reordered(base,reorderBytes(2,{5,3,5,99,2,1}),policy),tokens({"B","A"}),tokens({"x","y"}),{5,4,3,6,7});
        lists(reordered(base,reorderBytes(2,{3,5,3}),policy),tokens({"A","B"}),tokens({"x","y"}),{3,4,5,6,7});
        lists(reordered(base,reorderBytes(2,{6,4,6,5,3,4}),policy),tokens({"B","A"}),tokens({"y","x"}),{5,6,3,4,7});
    }
    rejectedBytes(base,reorderBytes(2,{5,3,5}),ErrorKind::Malformed);
    rejectedBytes(base,reorderBytes(2,{5,3,1}),ErrorKind::Malformed);
    check(inspect(base) == before,"reorder mutated retained original owner");
}
void duplicateSiblingProjection() {
    context = "duplicate sibling names from a same-parent rename";
    Batch start; start.metadata.commands = {create(2,1,"P"),create(3,2,"A"),create(4,2,"B"),create(5,4,"Child")};
    start.field(4,TfToken("value"),VtValue(4)); const auto base = candidate({},start);
    Batch rename; rename.metadata.commands = {move(4,2,2,"A")}; // Equal parents: rename B onto the occupied name A.
    const auto shadowed = candidate(base,rename);
    indexed(shadowed,3,"/P/A",SdfSpecTypePrim);
    check(shadowed->pathForSection(4).IsEmpty() && shadowed->pathForSection(5).IsEmpty(),"shadowed sibling or descendant has a path");
    check(!shadowed->data()->Has(SdfPath("/P/A"),TfToken("value")) && !shadowed->data()->HasSpec(SdfPath("/P/A/Child")),
          "shadowed sibling's fields or subtree became visible");
    field(shadowed,"/P",SdfChildrenKeys->PrimChildren,tokens({"A"}));
    check(shadowed->sectionCount() == 5 && shadowed->pathCount() == 3,"shadowed sections were dropped or projected");
    Batch removeOwner; removeOwner.metadata.commands = {erase(3,2)};
    const auto revealed = candidate(shadowed,removeOwner);
    indexed(revealed,4,"/P/A",SdfSpecTypePrim); indexed(revealed,5,"/P/A/Child",SdfSpecTypePrim);
    field(revealed,"/P/A",TfToken("value"),VtValue(4)); field(revealed,"/P",SdfChildrenKeys->PrimChildren,tokens({"A"}));
    Batch separate; separate.metadata.commands = {move(3,2,2,"Z")};
    const auto separated = candidate(shadowed,separate);
    indexed(separated,3,"/P/Z",SdfSpecTypePrim); indexed(separated,4,"/P/A",SdfSpecTypePrim);
    field(separated,"/P",SdfChildrenKeys->PrimChildren,tokens({"Z","A"}));
    // A Create at the shared name replaces the lowest identity, which reveals the other.
    Batch replace; replace.metadata.commands = {create(6,2,"A")};
    const auto replaced = candidate(shadowed,replace);
    gone(replaced,3); indexed(replaced,4,"/P/A",SdfSpecTypePrim);
    check(replaced->pathForSection(6).IsEmpty() && replaced->sectionCount() == 5,"second duplicate was projected");
}
void fieldsAndSamples() {
    context = "field precedence, notifications and sample phases";
    Batch batch; batch.metadata.commands = {create(2,1,"P"),create(3,2,"a",SdfSpecTypeAttribute)};
    batch.field(2,TfToken("value"),VtValue(11),9);
    batch.field(2,TfToken("other"),VtValue(std::string("kept")),2);
    batch.field(2,TfToken("value"),VtValue(22),2);
    batch.field(2,TfToken("erased"),VtValue(7),5);
    batch.field(2,TfToken("erased"),VtValue(),6);
    batch.field(2,TfToken("blocked"),VtValue(SdfValueBlock()),7);
    batch.field(3,SdfFieldKeys->TimeSamples,VtValue(SdfTimeSampleMap{{1,VtValue(1)},{2,VtValue(2)}}),0);
    batch.sample(3,1,VtValue(10)); batch.sample(3,2,VtValue()); batch.sample(3,3,VtValue(30)); batch.sample(3,3,VtValue(31));
    const auto scene = candidate({},batch);
    field(scene,"/P",TfToken("value"),VtValue(22)); field(scene,"/P",TfToken("other"),VtValue(std::string("kept")));
    absent(scene,"/P",TfToken("erased")); field(scene,"/P",TfToken("blocked"),VtValue(SdfValueBlock()));
    field(scene,"/P.a",SdfFieldKeys->TimeSamples,VtValue(SdfTimeSampleMap{{1,VtValue(10)},{3,VtValue(31)}}));
    check(scene->data()->ListTimeSamplesForPath(SdfPath("/P.a")) == std::set<double>{1,3},"sample index differs from replacement/override oracle");
    const std::vector<size_t> records{6,1,2,3,4,5,0};
    const auto& changes = scene->fieldChanges(); check(changes.size() == records.size(),"notification records dropped or samples mixed in");
    for (size_t i = 0; i < records.size(); ++i) {
        const auto r = records[i]; const auto& expected = batch.metadata.fields[r];
        check(changes[i].record == r && changes[i].section == expected.section && changes[i].setOrder == expected.setOrder &&
            changes[i].field == TfToken(expected.name),"notification order/identity differs");
    }
    Batch emptySamples;
    emptySamples.field(3,SdfFieldKeys->TimeSamples,VtValue(SdfTimeSampleMap{}));
    emptySamples.sample(3,99,VtValue());
    const auto explicitEmpty = candidate(scene,emptySamples);
    field(explicitEmpty,"/P.a",SdfFieldKeys->TimeSamples,VtValue(SdfTimeSampleMap{}));
    check(explicitEmpty->data()->ListTimeSamplesForPath(SdfPath("/P.a")).empty(),"empty authored sample map retained samples");
    Batch missingSample; missingSample.sample(3,100,VtValue());
    const auto stillEmpty = candidate(explicitEmpty,missingSample);
    field(stillEmpty,"/P.a",SdfFieldKeys->TimeSamples,VtValue(SdfTimeSampleMap{}));
    Batch eraseSamples; eraseSamples.field(3,SdfFieldKeys->TimeSamples,VtValue()); eraseSamples.sample(3,100,VtValue());
    absent(candidate(stillEmpty,eraseSamples),"/P.a",SdfFieldKeys->TimeSamples);
}
void authoredChildren() {
    context = "effective explicit child fields";
    Batch start; start.metadata.commands = {create(2,1,"P"),create(5,1,"Empty")};
    start.field(5,SdfChildrenKeys->PrimChildren,tokens({}));
    const auto previous = candidate({},start); const auto before = inspect(previous);
    Batch batch; batch.metadata.commands = {create(3,2,"A"),create(4,2,"B")};
    batch.field(2,SdfChildrenKeys->PrimChildren,tokens({"A"}));
    batch.field(2,SdfChildrenKeys->PrimChildren,tokens({}));
    batch.field(2,SdfChildrenKeys->PrimChildren,VtValue());
    batch.field(2,SdfChildrenKeys->PrimChildren,tokens({"B","A"}));
    const auto scene = candidate(previous,batch);
    field(scene,"/P",SdfChildrenKeys->PrimChildren,tokens({"B","A"}));
    field(scene,"/Empty",SdfChildrenKeys->PrimChildren,tokens({}));
    check(inspect(previous) == before,"valid candidate changed previous owner");
    Batch add; add.metadata.commands = {create(6,2,"C")};
    field(candidate(scene,add),"/P",SdfChildrenKeys->PrimChildren,tokens({"B","A","C"}));
    for (const auto& invalid : {tokens({"A"}),tokens({"A","A"}),tokens({}),VtValue()}) {
        auto conflict = batch; conflict.values.fields.back() = invalid; rejected(previous,conflict);
    }
    auto badType = batch; badType.values.fields.front() = VtValue(7); rejected(previous,badType,ErrorKind::Malformed);
    Batch eraseEmpty; eraseEmpty.field(5,SdfChildrenKeys->PrimChildren,VtValue());
    const auto absentEmpty = candidate(scene,eraseEmpty); absent(absentEmpty,"/Empty",SdfChildrenKeys->PrimChildren);
    Batch noop; check(inspect(candidate(scene,noop)) == inspect(scene),"unrelated batch erased explicit-empty child field");
}
void serviceConflictRules() {
    context = "service conflict rules for missing targets and occupied names";
    Batch start; start.metadata.commands = {create(2,1,"P"),create(3,2,"A"),create(4,1,"Q"),create(5,4,"B")};
    start.field(3,TfToken("value"),VtValue(77)); const auto base = candidate({},start);
    auto apply = [&](std::vector<Command> commands) { Batch batch; batch.metadata.commands = std::move(commands); return candidate(base,batch); };
    // Field and sample records for a missing section are orphan data: accepted, never visible.
    Batch orphan; orphan.field(99,TfToken("value"),VtValue(1)); orphan.sample(98,1.0,VtValue(2.0));
    orphan.field(3,TfToken("value"),VtValue(78));
    const auto orphaned = candidate(base,orphan);
    field(orphaned,"/P/A",TfToken("value"),VtValue(78));
    check(orphaned->sectionCount() == base->sectionCount(),"orphan data created a section");
    // Data for a section deleted earlier in the same batch is orphan data too.
    Batch sameBatch; sameBatch.metadata.commands = {erase(3,2)};
    sameBatch.field(3,TfToken("value"),VtValue(79)); sameBatch.sample(3,2.0,VtValue(3.0));
    sameBatch.field(2,TfToken("kept"),VtValue(1),4);
    const auto dropped = candidate(base,sameBatch);
    missing(dropped,3,"/P/A"); field(dropped,"/P",TfToken("kept"),VtValue(1));
    check(dropped->fieldChanges().size() == 1 && dropped->fieldChanges()[0].section == 2 && dropped->fieldChanges()[0].record == 1,
          "orphan record reported as a field change");
    // An orphan timeSamples value is still type-checked.
    Batch badOrphan; badOrphan.metadata.commands = {erase(3,2)}; badOrphan.field(3,SdfFieldKeys->TimeSamples,VtValue(7));
    rejected(base,badOrphan,ErrorKind::Malformed);
    // Structural commands whose target is missing are skipped.
    check(inspect(apply({erase(99,1)})) == inspect(base),"delete of a missing section changed the scene");
    check(inspect(apply({move(99,1,2,"M")})) == inspect(base),"move of a missing section changed the scene");
    check(inspect(apply({create(6,99,"C")})) == inspect(base),"create under a missing parent changed the scene");
    // A Delete's carried parent is bookkeeping, not a precondition.
    missing(apply({erase(3,4)}),3,"/P/A");
    // Create at an occupied (parent, name) deletes the occupant's subtree first.
    const auto replaced = apply({create(6,1,"Q")});
    indexed(replaced,6,"/Q",SdfSpecTypePrim); gone(replaced,4); gone(replaced,5);
    // Local content never creates a live identity. A delivered Create of one repeats a commit the scene
    // holds and changes nothing. A retired identity is created again, even in the same batch.
    { Batch live; live.metadata.commands = {create(2,1,"Again")}; rejected(base,live,ErrorKind::Malformed);
      check(inspect(delivered(base,live)) == inspect(base),"repeated create changed the scene");
      Batch moved; moved.metadata.commands = {create(3,4,"A")};
      check(inspect(delivered(base,moved)) == inspect(base),"repeated create of a moved section changed the scene"); }
    indexed(apply({erase(3,2),create(3,1,"Reused")}),3,"/Reused",SdfSpecTypePrim);
    // Move: a missing new parent or a cycle deletes the moved subtree.
    missing(apply({move(3,2,99,"A")}),3,"/P/A");
    missing(apply({move(2,1,3,"Cycle")}),2,"/P");
    // Move onto another occupied name deletes the occupant, then moves.
    { const auto moved = apply({move(5,4,2,"A")}); indexed(moved,5,"/P/A",SdfSpecTypePrim); gone(moved,3); }
    // Move onto the name of the mover's own ancestor deletes the mover.
    { Batch nested; nested.metadata.commands = {create(6,5,"Deep")}; const auto deep = candidate(base,nested);
      Batch up; up.metadata.commands = {move(6,5,4,"B")}; const auto ancestor = candidate(deep,up);
      gone(ancestor,6); indexed(ancestor,5,"/Q/B",SdfSpecTypePrim); }
    // Equal old and new parents rename under the stored parent.
    indexed(apply({move(3,4,4,"Renamed")}),3,"/P/Renamed",SdfSpecTypePrim);
    // A captured Move replayed after a peer moved its section from P to Q: the command's
    // parents select the rule, which then acts on the stored parent.
    { Batch peer; peer.metadata.commands = {move(3,2,4,"A")}; const auto movedByPeer = candidate(base,peer);
      auto replay = [&](std::vector<Command> commands) { Batch batch; batch.metadata.commands = std::move(commands); return candidate(movedByPeer,batch); };
      // Move(S, old=P, new=P) is a rename under the stored parent Q, without a sibling-name check.
      { const auto renamed = replay({move(3,2,2,"N2")});
        indexed(renamed,3,"/Q/N2",SdfSpecTypePrim); check(!renamed->data()->HasSpec(SdfPath("/P/N2")),"rename used the command parent"); field(renamed,"/Q",SdfChildrenKeys->PrimChildren,tokens({"B","N2"})); }
      { const auto duplicate = replay({move(3,2,2,"B")});
        indexed(duplicate,3,"/Q/B",SdfSpecTypePrim); check(duplicate->pathForSection(5).IsEmpty(),"higher duplicate identity was projected");
        check(duplicate->sectionCount() == base->sectionCount(),"same-parent rename deleted the occupant"); }
      // Move(S, old=P, new=Q) takes the cross-parent path: occupancy applies at Q, and S keeps its slot there.
      { const auto moved = replay({move(3,2,4,"A2")});
        indexed(moved,3,"/Q/A2",SdfSpecTypePrim); field(moved,"/Q",SdfChildrenKeys->PrimChildren,tokens({"B","A2"})); }
      { const auto occupied = replay({move(3,2,4,"B")});
        indexed(occupied,3,"/Q/B",SdfSpecTypePrim); gone(occupied,5); field(occupied,"/Q",SdfChildrenKeys->PrimChildren,tokens({"B"})); }
      // Move(S, old=P, new=R) removes S only from P's list, which then omits it:
      // Q's list still names S, so deleting Q deletes S, as the service's cascade does.
      { const auto stale = replay({create(6,1,"R"),move(3,2,6,"A")});
        indexed(stale,3,"/R/A",SdfSpecTypePrim); field(stale,"/Q",SdfChildrenKeys->PrimChildren,tokens({"B"}));
        field(stale,"/R",SdfChildrenKeys->PrimChildren,tokens({"A"}));
        Batch dropQ; dropQ.metadata.commands = {erase(4,1)}; const auto dropped = candidate(stale,dropQ);
        gone(dropped,3); gone(dropped,4); gone(dropped,5); indexed(dropped,6,"/R",SdfSpecTypePrim);
        absent(dropped,"/R",SdfChildrenKeys->PrimChildren);
        // A stale listing under a parent that survives leaves S reachable through its stored parent.
        Batch dropB; dropB.metadata.commands = {erase(5,4)}; const auto kept = candidate(stale,dropB);
        indexed(kept,3,"/R/A",SdfSpecTypePrim); gone(kept,5);
        // A Reorder of Q replaces its whole list, so a later delete of Q keeps S.
        Batch relist; relist.metadata.commands = {reorder(4,{5}),erase(4,1)}; const auto relisted = candidate(stale,relist);
        indexed(relisted,3,"/R/A",SdfSpecTypePrim); gone(relisted,4);
        // A diff retracting Q takes Q's list with it: S stays under R and no listing names Q.
        Batch retractQ; retractQ.metadata.isDiff = true; retractQ.metadata.baseVersion = 1;
        retractQ.metadata.commands = {erase(4,0,true),erase(5,0,true)}; const auto retracted = candidate(stale,retractQ);
        indexed(retracted,3,"/R/A",SdfSpecTypePrim); gone(retracted,4); gone(retracted,5); }
      // Within one batch, a Move that drops a listing is seen by a later Delete of that parent.
      { const auto stale = replay({create(6,1,"R"),move(3,2,6,"A"),create(7,1,"X")});
        Batch leave; leave.metadata.commands = {move(3,4,7,"A"),erase(4,1)}; const auto left = candidate(stale,leave);
        indexed(left,3,"/X/A",SdfSpecTypePrim); gone(left,4);
        Batch plain; plain.metadata.commands = {create(6,1,"R"),move(3,4,6,"A"),erase(4,1)};
        indexed(candidate(movedByPeer,plain),3,"/R/A",SdfSpecTypePrim); }
      // With the carried old parent current, the move leaves no stale listing.
      { const auto fresh = replay({create(6,1,"R"),move(3,4,6,"A")});
        Batch dropQ; dropQ.metadata.commands = {erase(4,1)}; indexed(candidate(fresh,dropQ),3,"/R/A",SdfSpecTypePrim); } }
    // A ghost retraction of a section the client never held is a no-op.
    Batch ghost; ghost.metadata.isDiff = true; ghost.metadata.baseVersion = 1; ghost.metadata.commands = {erase(99,0,true)};
    check(inspect(candidate(base,ghost)) == inspect(base),"missing diff deletion changed the scene");
}
void conflictsAndLimits() {
    context = "atomic graph conflicts and scene quotas";
    Batch start; start.metadata.commands = {create(2,1,"P"),create(3,2,"A"),create(4,1,"Q")};
    start.field(3,TfToken("value"),VtValue(77)); const auto previous = candidate({},start);
    Batch diffConflict; diffConflict.metadata.isDiff = true; diffConflict.metadata.baseVersion = 1;
    diffConflict.metadata.commands = {diff(2,3,"P",0),diff(3,2,"A",0)}; rejected(previous,diffConflict,ErrorKind::Unsupported);
    // Restated siblings are inserted at their indices in command order, so a later one at an equal index precedes.
    diffConflict.metadata.commands = {diff(4,1,"Q",0),diff(2,1,"P",0)};
    field(candidate(previous,diffConflict),"/",SdfChildrenKeys->PrimChildren,tokens({"P","Q"}));
    rejectedBytes(previous,"not an envelope",ErrorKind::Malformed);
    Batch edit; edit.field(3,TfToken("value"),VtValue(99)); const auto bytes = edit.bytes();
    auto limit = WireSceneLimits{}; limit.envelope.maxBytes = bytes.size()-1; rejected(previous,edit,ErrorKind::ResourceLimit,limit);
    limit = {}; limit.values.maxEncodedBytes = 1; rejected(previous,edit,ErrorKind::ResourceLimit,limit);
    limit = {}; limit.maxSections = 3; rejected(previous,edit,ErrorKind::ResourceLimit,limit);
    limit = {}; limit.maxFields = 1; rejected(previous,edit,ErrorKind::ResourceLimit,limit);
    limit = {}; limit.maxMetadataBytes = 1; rejected(previous,edit,ErrorKind::ResourceLimit,limit);
    limit = {}; limit.maxPathBytes = 1; rejected(previous,edit,ErrorKind::ResourceLimit,limit);
    limit = {}; limit.maxPathDepth = 1; rejected(previous,edit,ErrorKind::ResourceLimit,limit);
    // A rename is charged its new paths only, not beside the old ones it replaces.
    { limit = {}; limit.maxPathBytes = 200;
      Batch named; named.metadata.commands = {create(2,1,std::string(60,'a'))};
      Batch renamed; renamed.metadata.commands = {move(2,1,1,std::string(60,'b'))};
      indexed(candidate(candidate({},named,limit),renamed,limit),2,"/"+std::string(60,'b'),SdfSpecTypePrim); }
    // Every path an update derives again is charged its new bytes only, whatever its level: renaming an ancestor
    // and its descendant together, or an ancestor that shortens its subtree beside a sibling that grows, fits a
    // limit its final scene fits, as the same scene built from nothing does.
    { limit = {}; limit.maxPathBytes = 1000;
      const auto renamedTogether = [&](const Batch& initial, const Batch& renamed, const Batch& fresh) {
          const auto after = candidate(candidate({},initial,limit),renamed,limit);
          check(inspect(after) == inspect(candidate({},fresh,limit)),"a rename fitting the path limit differs from its scene built fresh");
          return after;
      };
      const std::string long300(300,'p'), long400(400,'c');
      Batch nested; nested.metadata.commands = {create(2,1,"P"),create(3,2,long400)};
      Batch both; both.metadata.commands = {move(2,1,1,long300),move(3,2,2,"C")};
      Batch nestedFresh; nestedFresh.metadata.commands = {create(2,1,long300),create(3,2,"C")};
      const auto after = renamedTogether(nested,both,nestedFresh);
      indexed(after,2,"/"+long300,SdfSpecTypePrim); indexed(after,3,"/"+long300+"/C",SdfSpecTypePrim);
      const std::string long150(150,'x'), long320(320,'b');
      Batch siblings; siblings.metadata.commands = {create(2,1,std::string(150,'a')),create(3,2,long150),create(4,1,"B")};
      Batch exchange; exchange.metadata.commands = {move(2,1,1,"A"),move(4,1,1,long320)};
      Batch siblingsFresh; siblingsFresh.metadata.commands = {create(2,1,"A"),create(3,2,long150),create(4,1,long320)};
      indexed(renamedTogether(siblings,exchange,siblingsFresh),3,"/A/"+long150,SdfSpecTypePrim); }
    // Every update may do work proportional to the scene and its records; maxWork bounds the rest, such as path bytes.
    { Batch longName; longName.metadata.commands = {create(5,2,std::string(65536,'n').c_str())};
      limit = {}; limit.maxWork = 1; rejected(previous,longName,ErrorKind::ResourceLimit,limit);
      check(candidate(previous,longName)->containsSection(5),"long name rejected under the default work limit"); }
    limit = {}; limit.retainedValues.maxEncodedBytes = 1; rejected(previous,edit,ErrorKind::ResourceLimit,limit);
    field(previous,"/P/A",TfToken("value"),VtValue(77));
    // A candidate built on its predecessor holds the predecessor's retained values to the retained-value limits,
    // even when its update touches no valued section.
    { VtDictionary inner; inner["leaf"] = VtValue(1); VtDictionary outer; outer["inner"] = VtValue(inner);
      Batch valued; valued.field(2,TfToken("nested"),VtValue(outer)); valued.field(3,TfToken("text"),VtValue(std::string(64,'s')));
      const auto heavy = candidate(previous,valued);
      Batch none; Batch unvalued; unvalued.metadata.commands = {create(9,1,"Z")};
      for (const auto* update : {&none,&unvalued}) {
          check(candidate(heavy,*update)->containsSection(3),"an update of a valued scene failed under default limits");
          limit = {}; limit.retainedValues.maxElements = 1; rejected(heavy,*update,ErrorKind::ResourceLimit,limit);
          limit = {}; limit.retainedValues.maxStringBytes = 63; rejected(heavy,*update,ErrorKind::ResourceLimit,limit);
          limit = {}; limit.retainedValues.maxDepth = 2; rejected(heavy,*update,ErrorKind::ResourceLimit,limit);
      }
      // Replacing the nested value leaves the scene within the nesting limit it exceeded.
      Batch flatten; flatten.field(2,TfToken("nested"),VtValue(1)); const auto flat = candidate(heavy,flatten);
      limit = {}; limit.retainedValues.maxDepth = 2;
      check(candidate(flat,none,limit)->containsSection(3),"a scene's nesting depth kept a replaced value's"); }
}
// Deleting a parent costs work in proportion to its subtree and its parent's
// children list, whether a Delete cascades over the subtree or a resume diff
// retracts each row: four times the children cost at most six times the
// scene work, where shifting each removed row out of its siblings' list one
// at a time would cost sixteen.
void wideDeletion() {
    context = "wide parent deletion work";
    const auto measure = [](size_t children, bool retract) {
        Batch build; build.metadata.commands = {create(2,1,"P"),create(3,1,"Q")};
        for (size_t i = 0; i < children; ++i) build.metadata.commands.push_back(create(4+i,2,"C"+std::to_string(i)));
        const auto previous = candidate({},build);
        Batch removal;
        if (retract) {
            removal.metadata.isDiff = true; removal.metadata.baseVersion = 1;
            removal.metadata.commands.push_back(erase(2,0,true));
            for (size_t i = 0; i < children; ++i) removal.metadata.commands.push_back(erase(4+i,0,true));
        } else removal.metadata.commands = {erase(2,1)};
        const auto bytes = removal.bytes();
        // The whole-scene derivation check would count its own work.
        setWireSceneDerivationCheck(false);
        const auto start = sceneWork::total();
        const auto removed = buildSceneCandidate(previous,bytes,{},nullptr,WireSceneBatchPolicy::Delivered);
        const auto work = sceneWork::total() - start;
        setWireSceneDerivationCheck(true);
        (void)buildSceneCandidate(previous,bytes,{},nullptr,WireSceneBatchPolicy::Delivered);
        check(removed->sectionCount() == 2 && !removed->containsSection(2) && !removed->containsSection(3+children),
            "the deletion left a row of the subtree");
        missing(removed,2,"/P"); indexed(removed,3,"/Q",SdfSpecTypePrim);
        field(removed,"/",SdfChildrenKeys->PrimChildren,tokens({"Q"}));
        return work;
    };
    for (const bool retract : {false,true}) {
        const auto small = measure(1000,retract), large = measure(4000,retract);
        std::cout << "wide_deletion " << (retract ? "retracted" : "cascaded") << " children=1000 work=" << small
                  << " children=4000 work=" << large << std::endl;
        check(large <= 6 * small,"deleting a parent's children costs more than linear work");
    }
}
// Creating a parent's children in one update costs work in proportion to
// them, in whatever order their IDs arrive: production IDs are random, and
// four times the creates cost at most six times the scene work, where
// inserting each into its siblings' list one at a time would cost sixteen.
void shuffledCreates(uint64_t seed) {
    context = "shuffled-ID creation work";
    std::mt19937_64 random(seed);
    const auto measure = [&](size_t children) {
        Batch parent; parent.metadata.commands = {create(2,1,"P")};
        const auto previous = candidate({},parent);
        std::vector<uint64_t> ids(children);
        for (size_t i = 0; i < children; ++i) ids[i] = 3+i;
        // Fisher-Yates over the engine's fixed output, so a seed names one order.
        for (size_t i = children; i > 1; --i) std::swap(ids[i-1],ids[random() % i]);
        Batch creation;
        for (const auto id : ids) creation.metadata.commands.push_back(create(id,2,"C"+std::to_string(id)));
        const auto bytes = creation.bytes();
        // The whole-scene derivation check would count its own work.
        setWireSceneDerivationCheck(false);
        const auto start = sceneWork::total();
        const auto created = buildSceneCandidate(previous,bytes,{});
        const auto work = sceneWork::total() - start;
        setWireSceneDerivationCheck(true);
        (void)buildSceneCandidate(previous,bytes,{});
        check(created->sectionCount() == children+2,"the creation left out a child");
        for (const auto id : {ids.front(),ids.back()})
            indexed(created,id,"/P/C"+std::to_string(id),SdfSpecTypePrim);
        check(created->listedChildren(2) == ids,"the children's order differs from their creation order");
        return work;
    };
    const auto small = measure(1000), large = measure(4000);
    std::cout << "shuffled_creates children=1000 work=" << small << " children=4000 work=" << large << std::endl;
    check(large <= 6 * small,"creating a parent's children costs more than linear work");
}
void variantNamesAndNestedTargets() {
    context = "variant names and canonical nested mapper targets";
    Batch variants;
    variants.metadata.commands = {create(2,1,"P"),create(3,2,"v",SdfSpecTypeVariantSet),
        create(4,3,"1",SdfSpecTypeVariant),create(5,3,"high-low",SdfSpecTypeVariant),
        create(6,3,"é",SdfSpecTypeVariant),create(7,2,"a",SdfSpecTypeAttribute),
        create(8,7,"/T.rel[../U]",SdfSpecTypeMapper)};
    const auto scene = candidate({},variants);
    indexed(scene,4,"/P{v=1}",SdfSpecTypeVariant);
    indexed(scene,5,"/P{v=high-low}",SdfSpecTypeVariant);
    indexed(scene,6,"/P{v=é}",SdfSpecTypeVariant);
    field(scene,"/P{v=}",SdfChildrenKeys->VariantChildren,tokens({"1","high-low","é"}));
    indexed(scene,8,"/P.a.mapper[/T.rel[/U]]",SdfSpecTypeMapper);
    field(scene,"/P.a",SdfChildrenKeys->MapperChildren,paths({"/T.rel[/U]"}));
    Batch rename; rename.metadata.commands = {move(2,1,1,"Q")};
    const auto moved = candidate(scene,rename);
    indexed(moved,8,"/Q.a.mapper[/T.rel[/U]]",SdfSpecTypeMapper);
    field(moved,"/Q.a",SdfChildrenKeys->MapperChildren,paths({"/T.rel[/U]"}));
    indexed(scene,8,"/P.a.mapper[/T.rel[/U]]",SdfSpecTypeMapper);
    Batch invalid; invalid.metadata.commands = {create(9,7,"/T.rel[../../U]",SdfSpecTypeMapper)};
    rejected(scene,invalid);

    std::string target = "/Leaf";
    for (size_t i = 0; i < 12; ++i) target = "/T.rel["+target+"]";
    Batch deep; deep.metadata.commands = {create(2,1,"P"),create(3,2,"a",SdfSpecTypeAttribute),
        create(4,3,target,SdfSpecTypeMapper)};
    const auto nested = candidate({},deep);
    indexed(nested,4,"/P.a.mapper["+target+"]",SdfSpecTypeMapper);
    // Existing graph ancestry is shallow. Narrowing the new candidate's limit
    // must still inspect the retained target's recursive path structure.
    WireSceneLimits limits; limits.maxPathDepth = 6;
    rejected(nested,Batch{},ErrorKind::ResourceLimit,limits);
}
void lateQuotaFailuresAndOwnership() {
    context = "late candidate quotas and borrowed data ownership";
    Batch initial; initial.metadata.commands = {create(2,1,"P")}; initial.field(2,TfToken("value"),VtValue(17));
    const auto previous = candidate({},initial); const auto expected = inspect(previous);
    WireSceneLimits limits; limits.maxFields = 2; // Root primChildren plus P.value.
    check(inspect(candidate(previous,Batch{},limits)) == expected,"prior scene does not fit late-field quota fixture");
    Batch additional; additional.field(2,TfToken("another"),VtValue(18));
    rejected(previous,additional,ErrorKind::ResourceLimit,limits);
    limits = {}; limits.retainedValues.maxEncodedBytes = 128;
    check(inspect(candidate(previous,Batch{},limits)) == expected,"prior scene does not fit retained-value quota fixture");
    Batch larger; larger.field(2,TfToken("value"),VtValue(std::string(512,'x')));
    rejected(previous,larger,ErrorKind::ResourceLimit,limits);
    field(previous,"/P",TfToken("value"),VtValue(17));

    auto owner = candidate({},initial);
    auto retained = owner;
    const SdfAbstractDataConstPtr borrowed = owner->data();
    owner.reset();
    check(bool(borrowed) && borrowed->HasSpec(SdfPath("/P")) &&
        borrowed->Get(SdfPath("/P"),TfToken("value")) == VtValue(17),"snapshot owner did not retain borrowed data view");
    retained.reset();
    check(!borrowed,"borrowed USD data view outlived its last snapshot owner");
}
struct Prim { std::string name; std::optional<int> value; };
State model(const std::map<uint64_t,Prim>& prims, const std::vector<uint64_t>& order) {
    State result{{SdfPath("/"),{SdfSpecTypePseudoRoot,{}}}};
    std::vector<TfToken> names;
    for (auto id : order) names.emplace_back(prims.at(id).name);
    if (!names.empty()) result.at(SdfPath("/")).fields[SdfChildrenKeys->PrimChildren] = VtValue(names);
    for (const auto& [id,prim] : prims) {
        Record record{SdfSpecTypePrim,{}};
        if (prim.value) record.fields[TfToken("value")] = VtValue(*prim.value);
        result.emplace(SdfPath("/"+prim.name),std::move(record));
    }
    return result;
}
void histories(uint64_t seed) {
    std::mt19937_64 random(seed);
    std::map<uint64_t,Prim> prims; std::vector<uint64_t> order; uint64_t next = 2;
    WireScenePtr scene = candidate({},Batch{});
    std::vector<std::pair<WireScenePtr,State>> retained;
    uint64_t fingerprint = UINT64_C(14695981039346656037);
    for (size_t step = 0; step < 256; ++step) {
        context = "wire scene seed "+std::to_string(seed)+" step "+std::to_string(step);
        Batch batch; const auto action = random()%5;
        if (order.empty() || action == 0) {
            const uint64_t id = next++; const std::string name = "N"+std::to_string(id);
            batch.metadata.commands = {create(id,1,name)}; prims.emplace(id,Prim{name,{}}); order.push_back(id);
        } else {
            const size_t selected = random()%order.size(); const uint64_t id = order[selected];
            if (action == 1) {
                const std::string name = "R"+std::to_string(step);
                batch.metadata.commands = {move(id,1,1,name)}; prims.at(id).name = name;
            } else if (action == 2) {
                batch.metadata.commands = {erase(id,1)}; prims.erase(id); order.erase(order.begin()+selected);
            } else if (action == 3 || order.size() < 2) {
                const int v = int(random()%20001)-10000;
                batch.field(id,TfToken("value"),VtValue(v+1),7);
                const bool eraseValue = random()%4 == 0;
                batch.field(id,TfToken("value"),eraseValue ? VtValue() : VtValue(v),2);
                prims.at(id).value = eraseValue ? std::optional<int>{} : std::optional<int>{v};
            } else {
                size_t other = (selected+1+random()%(order.size()-1))%order.size();
                const size_t a = std::min(selected,other), b = std::max(selected,other);
                std::swap(order[a],order[b]); batch.metadata.commands = {reorder(1,order)};
            }
        }
        const auto old = scene; const auto before = inspect(old);
        scene = candidate(old,batch); const auto expected = model(prims,order);
        check(inspect(scene) == expected,"seeded scene differs from independent map/order model");
        check(inspect(old) == before,"seeded candidate mutated predecessor");
        check(scene->sectionCount() == prims.size()+1,"seeded section count differs");
        for (const auto& [id,prim] : prims) indexed(scene,id,"/"+prim.name,SdfSpecTypePrim);
        for (uint64_t id = 2; id < next; ++id) if (!prims.count(id)) check(scene->pathForSection(id).IsEmpty(),"deleted history ID revived");
        for (const auto& [path,record] : expected) for (unsigned char c : path.GetString()) { fingerprint ^= c; fingerprint *= UINT64_C(1099511628211); }
        if (step%16 == 0) retained.emplace_back(scene,expected);
    }
    for (const auto& [owner,state] : retained) check(inspect(owner) == state,"retained historical owner changed");
    std::cout << "wire scene seed=" << seed << " steps=256 fingerprint=" << fingerprint << std::endl;
}
// Independent oracle: each sample applies to the map the earlier samples of its
// update left. Erasing an absent key changes nothing, including an authored
// empty map's presence; a change that empties the map removes the field.
using SampleField = std::optional<SdfTimeSampleMap>;
void applySample(SampleField& field, double time, const VtValue& value) {
    auto samples = field ? *field : SdfTimeSampleMap{};
    if (value.IsEmpty()) { if (!samples.erase(time)) return; }
    else samples[time] = value;
    if (samples.empty()) field.reset(); else field = std::move(samples);
}
void expectSamples(const WireScenePtr& scene, const char* path, const SampleField& expected) {
    if (expected) field(scene,path,SdfFieldKeys->TimeSamples,VtValue(*expected));
    else absent(scene,path,SdfFieldKeys->TimeSamples);
}
// /P with attributes a (3) {1:1,2:2}, b (4) an authored empty map, c (5) without samples and d (6) {5:5}.
WireScenePtr sampleBase() {
    Batch batch; batch.metadata.commands = {create(2,1,"P"),create(3,2,"a",SdfSpecTypeAttribute),create(4,2,"b",SdfSpecTypeAttribute),
        create(5,2,"c",SdfSpecTypeAttribute),create(6,2,"d",SdfSpecTypeAttribute)};
    batch.field(3,SdfFieldKeys->TimeSamples,VtValue(SdfTimeSampleMap{{1,VtValue(1)},{2,VtValue(2)}}));
    batch.field(4,SdfFieldKeys->TimeSamples,VtValue(SdfTimeSampleMap{}));
    batch.field(6,SdfFieldKeys->TimeSamples,VtValue(SdfTimeSampleMap{{5,VtValue(5)}}));
    return candidate({},batch);
}
void sampleApplication() {
    context = "per-section sample application";
    const auto base = sampleBase();
    const auto apply = [&](std::initializer_list<std::tuple<uint64_t,double,VtValue>> samples) {
        Batch batch; for (const auto& [id,time,value] : samples) batch.sample(id,time,value);
        return candidate(base,batch);
    };
    const SdfTimeSampleMap a{{1,VtValue(1)},{2,VtValue(2)}};
    // A set and an erase of one key in one update: the later record wins.
    auto scene = apply({{3,7,VtValue(70)},{3,7,VtValue()},{3,1,VtValue()},{3,1,VtValue(11)}});
    expectSamples(scene,"/P.a",SdfTimeSampleMap{{1,VtValue(11)},{2,VtValue(2)}});
    // Erasing an absent key keeps an authored empty map and never creates a field.
    scene = apply({{4,9,VtValue()},{5,9,VtValue()}});
    expectSamples(scene,"/P.b",SdfTimeSampleMap{}); expectSamples(scene,"/P.c",{});
    // A set then an erase on an authored empty map empties it, which removes the field.
    expectSamples(apply({{4,9,VtValue(90)},{4,9,VtValue()}}),"/P.b",{});
    // Emptying a map removes the field; a later set recreates it, and a later absent-key erase changes nothing.
    scene = apply({{6,5,VtValue()},{6,5,VtValue()}});
    expectSamples(scene,"/P.d",{});
    expectSamples(apply({{6,5,VtValue()},{6,8,VtValue(80)}}),"/P.d",SdfTimeSampleMap{{8,VtValue(80)}});
    // Interleaved sections apply independently; an orphan sample for a missing section is skipped.
    scene = apply({{3,3,VtValue(30)},{99,1,VtValue(1)},{6,6,VtValue(60)},{3,1,VtValue()},{99,2,VtValue()},{6,5,VtValue()}});
    expectSamples(scene,"/P.a",SdfTimeSampleMap{{2,VtValue(2)},{3,VtValue(30)}});
    expectSamples(scene,"/P.b",SdfTimeSampleMap{}); expectSamples(scene,"/P.c",{});
    expectSamples(scene,"/P.d",SdfTimeSampleMap{{6,VtValue(60)}});
    check(!scene->containsSection(99) && scene->sectionCount() == base->sectionCount(),"orphan sample created a section");
    // Samples after a same-update timeSamples field apply to the field's value.
    Batch replaced; replaced.field(3,SdfFieldKeys->TimeSamples,VtValue(SdfTimeSampleMap{})); replaced.sample(3,4,VtValue());
    replaced.field(6,SdfFieldKeys->TimeSamples,VtValue()); replaced.sample(6,5,VtValue());
    scene = candidate(base,replaced);
    expectSamples(scene,"/P.a",SdfTimeSampleMap{}); expectSamples(scene,"/P.d",{});
    expectSamples(scene,"/P.b",SdfTimeSampleMap{}); expectSamples(scene,"/P.c",{});
}
// Seeded updates of random sample records over the sampleBase states, compared with the per-record oracle.
void sampleHistories(uint64_t seed) {
    std::mt19937_64 random(seed ^ UINT64_C(0x5a3e9e1c));
    const auto base = sampleBase();
    const std::array<const char*,4> paths{"/P.a","/P.b","/P.c","/P.d"};
    for (size_t step = 0; step < 256; ++step) {
        context = "sample application seed "+std::to_string(seed)+" step "+std::to_string(step);
        std::array<SampleField,4> expected{SdfTimeSampleMap{{1,VtValue(1)},{2,VtValue(2)}},SdfTimeSampleMap{},std::nullopt,
            SdfTimeSampleMap{{5,VtValue(5)}}};
        Batch batch;
        for (size_t i = 0, count = random()%16; i < count; ++i) {
            const uint64_t id = 3+random()%5; // Section 7 is missing.
            const double time = double(1+random()%6);
            const auto value = random()%2 ? VtValue() : VtValue(int(random()%100));
            batch.sample(id,time,value);
            if (id < 7) applySample(expected[id-3],time,value);
        }
        const auto scene = candidate(base,batch);
        for (size_t i = 0; i < paths.size(); ++i) expectSamples(scene,paths[i],expected[i]);
        check(!scene->containsSection(7),"orphan sample created a section");
    }
}
// One attribute carries the service's per-delta entry limit of samples within the default work limit.
void largeSingleAttributeSamples() {
    context = "one attribute with 99,999 samples";
    constexpr int count = 99999;
    Batch batch; batch.metadata.commands = {create(2,1,"P"),create(3,2,"a",SdfSpecTypeAttribute)};
    for (int i = 0; i < count; ++i) batch.sample(3,double(i),VtValue(i));
    const auto scene = candidate({},batch);
    const auto value = scene->data()->Get(SdfPath("/P.a"),SdfFieldKeys->TimeSamples);
    check(value.IsHolding<SdfTimeSampleMap>() && value.UncheckedGet<SdfTimeSampleMap>().size() == size_t(count),
        "single-attribute samples dropped");
    check(value.UncheckedGet<SdfTimeSampleMap>().rbegin()->second == VtValue(count-1),"last sample value differs");
    // A later update rewrites every key of the retained map and erases the odd ones.
    Batch rewrite;
    for (int i = 0; i < count; ++i) rewrite.sample(3,double(i),i%2 ? VtValue() : VtValue(-i));
    const auto next = candidate(scene,rewrite);
    const auto rewritten = next->data()->Get(SdfPath("/P.a"),SdfFieldKeys->TimeSamples).UncheckedGet<SdfTimeSampleMap>();
    check(rewritten.size() == size_t(count/2+1) && rewritten.at(count-1) == VtValue(1-count),"rewritten samples differ");
    // Copying a retained map is charged per entry, beyond the per-record allowance.
    Batch one; one.sample(3,0.5,VtValue(5));
    auto limit = WireSceneLimits{}; limit.maxWork = 1; rejected(scene,one,ErrorKind::ResourceLimit,limit);
    check(candidate(scene,one)->data()->GetNumTimeSamplesForPath(SdfPath("/P.a")) == size_t(count+1),"one-sample update differs");
}
void concurrentOldOwner() {
    context = "immutable owner concurrent candidate construction";
    Batch initial; initial.metadata.commands = {create(2,1,"P"),create(3,2,"Child")}; initial.field(3,TfToken("value"),VtValue(17));
    const auto old = candidate({},initial); const auto expected = inspect(old);
    std::promise<void> ready, start; auto gate = start.get_future().share();
    auto reader = std::async(std::launch::async,[old,expected,&ready,gate] {
        ready.set_value(); gate.wait();
        for (size_t i = 0; i < 512; ++i) {
            if (inspect(old) != expected || old->sectionForPath(SdfPath("/P/Child")) != 3 || old->pathForSection(3) != SdfPath("/P/Child"))
                throw std::runtime_error("old scene owner changed during concurrent build");
        }
    });
    const auto readiness = ready.get_future().wait_for(std::chrono::seconds(2));
    start.set_value(); check(readiness == std::future_status::ready,"old-owner reader did not start");
    auto newest = old;
    for (int i = 0; i < 64; ++i) { Batch edit; edit.field(3,TfToken("value"),VtValue(i)); newest = candidate(old,edit); }
    check(reader.wait_for(std::chrono::seconds(5)) == std::future_status::ready,"old-owner reader did not finish"); reader.get();
    field(newest,"/P/Child",TfToken("value"),VtValue(63)); check(inspect(old) == expected,"concurrent build mutated retained old owner");
}
} // namespace
int hostAllocationProbe(const std::string&,const std::string&,long);
void hostProjectionTests();
void hostProjectionAllocationTests();
void hostLayerStateTests();
int main(int argc, char** argv) {
    try {
        if(argc==5 && std::string(argv[1])=="--allocation-probe")
            return hostAllocationProbe(argv[2],argv[3],std::stol(argv[4]));
        const uint64_t seed = argc > 1 ? std::stoull(argv[1]) : 42;
        // Every build checks its incremental derivation against a whole-scene one.
        setWireSceneDerivationCheck(true);
        std::cout << "master_seed=" << seed << " scenario=wire-scene" << std::endl;
        topology(); sectionEnumeration(); deletionDiffAndOrdering(); wholeListReorder(); perListReorder(); usdListOrderingDifferential(seed); fieldsAndSamples(); authoredChildren(); serviceConflictRules(); duplicateSiblingProjection(); conflictsAndLimits(); wideDeletion(); shuffledCreates(seed);
        variantNamesAndNestedTargets(); lateQuotaFailuresAndOwnership(); histories(seed); sampleApplication(); sampleHistories(seed); largeSingleAttributeSamples(); concurrentOldOwner(); context = "host projection"; hostProjectionTests(); hostProjectionAllocationTests();
        context = "host layer state"; hostLayerStateTests();
        std::cout << "independent wire scene graph/fields/ordering/retention contracts passed" << std::endl; return 0;
    } catch (const std::exception& error) { std::cerr << "wire scene test failed (" << context << "): " << error.what() << std::endl; return 1; }
}
