// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "SceneNamespace.h"
#include "UnsupportedOperation.h"
#include <pxr/usd/sdf/schema.h>
#include <algorithm>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <tuple>

using namespace PXR_NS;
using namespace omni::usd::live;
using namespace omni::usd::live::wire;
namespace {
std::string namespaceContext;
void requireNamespace(bool condition,const char* message) {
    if (!condition) throw std::runtime_error(namespaceContext+": "+message);
}
struct Record {
    SdfSpecType type;
    std::map<TfToken,VtValue> fields;
    bool operator==(const Record& other) const { return type == other.type && fields == other.fields; }
};
using State = std::map<SdfPath,Record>;
State inspect(const WireScenePtr& scene) {
    struct Visitor : SdfAbstractDataSpecVisitor {
        State result;
        bool VisitSpec(const SdfAbstractData& data,const SdfPath& path) override {
            Record record{data.GetSpecType(path),{}};
            for (const auto& field : data.List(path)) record.fields.emplace(field,data.Get(path,field));
            requireNamespace(result.emplace(path,std::move(record)).second,"duplicate final path"); return true;
        }
        void Done(const SdfAbstractData&) override {}
    } visitor;
    scene->data()->VisitSpecs(&visitor); return visitor.result;
}
Command create(uint64_t id,uint64_t parent,const char* name,SdfSpecType type = SdfSpecTypePrim) {
    Command command{}; command.kind = Command::Kind::Create; command.section = id; command.parent = parent;
    command.name = name; command.sectionType = uint8_t(type); return command;
}
WireScenePtr baseScene(const std::vector<Command>& commands) {
    EnvelopeInput metadata; metadata.commands = commands; DecodedValues values;
    for (const auto& command : commands) {
        metadata.fields.push_back({command.section,command.section,"marker",{}});
        values.fields.push_back(VtValue(int(command.section*10)));
    }
    return buildSceneCandidate({},encodeUsdEnvelope(metadata,values));
}
NamespaceSpec spec(uint64_t id,const char* path,SdfSpecType type = SdfSpecTypePrim) {
    return {id,type,SdfPath(path)};
}
NamespaceSpec root() { return spec(1,"/",SdfSpecTypePseudoRoot); }
void identities(const WireScenePtr& scene,const std::vector<NamespaceSpec>& expected,const std::set<uint64_t>& retained) {
    const auto state = inspect(scene);
    requireNamespace(scene->sectionCount() == expected.size() && state.size() == expected.size(),"final graph includes scratch/missing specs");
    std::set<uint64_t> ids;
    for (const auto& item : expected) {
        requireNamespace(ids.insert(item.id).second,"duplicate expected fixture ID");
        requireNamespace(scene->pathForSection(item.id) == item.path && scene->sectionForPath(item.path) == item.id &&
                         state.at(item.path).type == item.type,"final ID/path/type differs from explicit oracle");
        const auto value = state.at(item.path).fields.find(TfToken("marker"));
        if (retained.count(item.id))
            requireNamespace(value != state.at(item.path).fields.end() && value->second == VtValue(int(item.id*10)),
                             "retained identity lost or inherited another spec's fields");
        else requireNamespace(value == state.at(item.path).fields.end(),"new identity inherited deleted occupant's fields");
    }
}
void childTokens(const WireScenePtr& scene,const char* owner,const TfToken& field,std::initializer_list<const char*> expected,bool ordered = true) {
    std::vector<TfToken> tokens; for (const auto* name : expected) tokens.emplace_back(name);
    const VtValue value = scene->data()->Get(SdfPath(owner),field);
    if (ordered) requireNamespace(value == VtValue(tokens),"ordered token child membership differs");
    else {
        requireNamespace(value.IsHolding<std::vector<TfToken>>(),"child membership has wrong type");
        auto actual = value.Get<std::vector<TfToken>>();
        std::sort(actual.begin(),actual.end()); std::sort(tokens.begin(),tokens.end());
        requireNamespace(actual == tokens,"token child membership differs");
    }
}
void childPaths(const WireScenePtr& scene,const char* owner,const TfToken& field,std::initializer_list<const char*> expected) {
    std::vector<SdfPath> paths; for (const auto* path : expected) paths.emplace_back(path);
    const VtValue value = scene->data()->Get(SdfPath(owner),field);
    requireNamespace(value == VtValue(paths),"ordered path child membership differs");
}
struct CheckedPlan {
    NamespacePlan plan;
    std::vector<uint64_t> scratch;
};
CheckedPlan good(const WireScenePtr& base,const std::vector<NamespaceSpec>& desired,const std::set<uint64_t>& retained,uint64_t scratchBase = 10000) {
    const auto before = inspect(base);
    std::vector<uint64_t> scratch;
    const auto allocate = [&] { const auto id = scratchBase+scratch.size(); scratch.push_back(id); return id; };
    auto plan = planSceneNamespace(base,desired,allocate);
    requireNamespace(plan.candidate && plan.workUsed > 0,"planner omitted candidate/work accounting");
    identities(plan.candidate,desired,retained);
    requireNamespace(inspect(base) == before,"planner mutated retained base");
    std::set<uint64_t> created;
    for (const auto& command : plan.commands) {
        requireNamespace(command.kind == Command::Kind::Create || command.kind == Command::Kind::Delete || command.kind == Command::Kind::Move,
                         "namespace plan contains unsupported reorder/diff commands");
        if (command.kind == Command::Kind::Create) requireNamespace(created.insert(command.section).second,"planner reused a create ID");
    }
    for (auto id : scratch) requireNamespace(plan.candidate->pathForSection(id).IsEmpty(),"temporary scratch ID survived final plan");
    // Replay the complete command list in one ordinary batch. The independent
    // identity/value assertions above remain the oracle, not this consistency check.
    EnvelopeInput metadata; metadata.commands = plan.commands;
    const auto replayed = buildSceneCandidate(base,encodeEnvelope(metadata));
    requireNamespace(inspect(replayed) == inspect(plan.candidate),"returned commands/candidate disagree");
    return {std::move(plan),std::move(scratch)};
}
void reject(const WireScenePtr& base,const std::vector<NamespaceSpec>& desired,ErrorKind kind,
            const std::function<uint64_t()>& allocate = {},const SceneNamespaceLimits& limits = {},WireSceneWorkBudget* work = nullptr) {
    const auto before = inspect(base);
    bool rejected = false;
    try { (void)planSceneNamespace(base,desired,allocate,limits,work); }
    catch (const DecodeError& error) { requireNamespace(error.kind == kind,"wrong namespace rejection kind"); rejected = true; }
    requireNamespace(rejected && inspect(base) == before,"failed namespace plan accepted or changed retained base");
}
// The plan fails with UnsupportedOperation naming `reason` and no single operation.
void rejectReason(const WireScenePtr& base,const std::vector<NamespaceSpec>& desired,UnsupportedOperation::Reason reason) {
    reject(base,desired,ErrorKind::Unsupported);
    bool named = false;
    try { (void)planSceneNamespace(base,desired); }
    catch (const UnsupportedOperation& error) {
        requireNamespace(error.reason() == reason && error.index() == UnsupportedOperation::noOperation,
                         "unsupported namespace plan named the wrong reason");
        named = true;
    }
    requireNamespace(named,"unsupported namespace plan did not name its reason");
}
void createsDeletesAndReplacement() {
    namespaceContext = "namespace create/delete/replacement";
    const auto empty = baseScene({});
    const std::vector<NamespaceSpec> created{root(),spec(17,"/P"),spec(91,"/P.a",SdfSpecTypeAttribute),spec(203,"/Q")};
    const auto addition = good(empty,created,{});
    requireNamespace(addition.scratch.empty(),"acyclic creates allocated scratch identity");
    childTokens(addition.plan.candidate,"/",SdfChildrenKeys->PrimChildren,{"P","Q"},false);
    childTokens(addition.plan.candidate,"/P",SdfChildrenKeys->PropertyChildren,{"a"});
    const auto base = baseScene({create(2,1,"A"),create(3,1,"B"),create(4,2,"Child")});
    const auto cleared = good(base,{root()},{});
    requireNamespace(!cleared.plan.candidate->data()->Has(SdfPath("/"),SdfChildrenKeys->PrimChildren),"delete-all retained stale child field");
    requireNamespace(cleared.plan.candidate->pathForSection(2).IsEmpty() && cleared.plan.candidate->pathForSection(4).IsEmpty(),
                     "deleted subtree ID survived");
    const std::vector<NamespaceSpec> replaced{root(),spec(3,"/A"),spec(20,"/B"),spec(21,"/B/New")};
    const auto result = good(base,replaced,{3});
    requireNamespace(result.plan.candidate->pathForSection(2).IsEmpty() && result.plan.candidate->pathForSection(4).IsEmpty(),
                     "deleted identity reused by occupied path");
    childTokens(result.plan.candidate,"/B",SdfChildrenKeys->PrimChildren,{"New"});
}
void movingSubtrees() {
    namespaceContext = "namespace subtree and escape moves";
    const auto base = baseScene({create(2,1,"A"),create(3,2,"Child"),create(4,3,"a",SdfSpecTypeAttribute),create(5,1,"B")});
    const std::vector<NamespaceSpec> moved{root(),spec(2,"/B/A"),spec(3,"/B/A/Child"),spec(4,"/B/A/Child.a",SdfSpecTypeAttribute),spec(5,"/B")};
    const auto plan = good(base,moved,{2,3,4,5});
    childTokens(plan.plan.candidate,"/",SdfChildrenKeys->PrimChildren,{"B"});
    childTokens(plan.plan.candidate,"/B",SdfChildrenKeys->PrimChildren,{"A"});
    childTokens(plan.plan.candidate,"/B/A",SdfChildrenKeys->PrimChildren,{"Child"});
    childTokens(plan.plan.candidate,"/B/A/Child",SdfChildrenKeys->PropertyChildren,{"a"});
    const std::vector<NamespaceSpec> escaped{root(),spec(3,"/Keep"),spec(4,"/Keep.a",SdfSpecTypeAttribute),spec(5,"/B")};
    const auto escape = good(base,escaped,{3,4,5});
    requireNamespace(escape.plan.candidate->pathForSection(2).IsEmpty(),"deleted parent survives escaped child");
    childTokens(escape.plan.candidate,"/Keep",SdfChildrenKeys->PropertyChildren,{"a"});
    // A child becomes the new ancestor, while its old parent keeps its ID/data.
    const std::vector<NamespaceSpec> reversed{root(),spec(2,"/Child/A"),spec(3,"/Child"),spec(4,"/Child.a",SdfSpecTypeAttribute),spec(5,"/B")};
    const auto reverse = good(base,reversed,{2,3,4,5});
    childTokens(reverse.plan.candidate,"/Child",SdfChildrenKeys->PrimChildren,{"A"});
    childTokens(reverse.plan.candidate,"/Child",SdfChildrenKeys->PropertyChildren,{"a"});
}
void maximalRecursiveDeletes() {
    namespaceContext = "maximal delete independent of descendant IDs";
    const auto base = baseScene({create(100,1,"P"),create(2,100,"Child"),create(3,2,"Grand"),
                                create(200,1,"Q"),create(4,200,"Outside"),create(5,4,"Leaf")});
    const auto original = inspect(base);
    const auto removal = good(base,{root(),spec(200,"/Q"),spec(4,"/Q/Outside"),spec(5,"/Q/Outside/Leaf")},{200,4,5});
    requireNamespace(removal.plan.commands.size() == 1 && removal.scratch.empty(),
                     "recursive delete enumerated descendants or allocated scratch");
    const auto& command = removal.plan.commands.front();
    requireNamespace(command.kind == Command::Kind::Delete && command.section == 100 && command.parent == 1,
                     "recursive delete did not target highest disappearing ancestor");
    EnvelopeInput outgoing; outgoing.commands = removal.plan.commands;
    const auto acceptedBytes = encodeEnvelope(outgoing);
    auto move = [](uint64_t id,uint64_t oldParent,uint64_t parent,const char* name) {
        Command result{}; result.kind = Command::Kind::Move; result.section = id;
        result.oldParent = oldParent; result.parent = parent; result.name = name; return result;
    };
    auto replay = [&](const std::vector<Command>& peer, const std::vector<NamespaceSpec>& expected,
                      const std::set<uint64_t>& retained) {
        EnvelopeInput input; input.commands = peer;
        const auto current = buildSceneCandidate(base,encodeEnvelope(input));
        const auto before = inspect(current);
        // Apply the exact already-planned bytes to the service-time graph. The
        // expected IDs/paths below are independent of the planner projection.
        const auto result = buildSceneCandidate(current,acceptedBytes);
        identities(result,expected,retained);
        requireNamespace(inspect(current) == before && inspect(base) == original,
                         "latest-subtree application mutated a retained peer/base scene");
        return result;
    };
    namespaceContext = "maximal delete leaves peer-moved-out subtree alive";
    const auto escaped = replay({move(2,100,200,"Escaped")},
        {root(),spec(200,"/Q"),spec(4,"/Q/Outside"),spec(5,"/Q/Outside/Leaf"),
         spec(2,"/Q/Escaped"),spec(3,"/Q/Escaped/Grand")},{200,4,5,2,3});
    childTokens(escaped,"/Q",SdfChildrenKeys->PrimChildren,{"Outside","Escaped"});
    childTokens(escaped,"/Q/Escaped",SdfChildrenKeys->PrimChildren,{"Grand"});
    namespaceContext = "maximal delete includes peer-moved-in subtree";
    const auto entered = replay({move(4,200,100,"Entered")},{root(),spec(200,"/Q")},{200});
    requireNamespace(!entered->data()->Has(SdfPath("/Q"),SdfChildrenKeys->PrimChildren),
                     "moved-in subtree left stale outside child membership");
    namespaceContext = "maximal delete includes unseen child and grandchild";
    replay({create(6,100,"New"),create(7,6,"Grand")},
        {root(),spec(200,"/Q"),spec(4,"/Q/Outside"),spec(5,"/Q/Outside/Leaf")},{200,4,5});

    namespaceContext = "maximal delete waits for desired survivor before deleting siblings";
    const auto withSurvivor = baseScene({create(100,1,"P"),create(2,100,"Keep"),create(3,2,"Grand"),
                                       create(4,100,"Discard"),create(200,1,"Q")});
    const auto escape = good(withSurvivor,
        {root(),spec(200,"/Q"),spec(2,"/Q/Keep"),spec(3,"/Q/Keep/Grand")},{200,2,3});
    requireNamespace(escape.scratch.empty() && escape.plan.commands.size() == 2,
                     "survivor escape emitted redundant descendant deletion");
    const auto& first = escape.plan.commands[0]; const auto& last = escape.plan.commands[1];
    requireNamespace(first.kind == Command::Kind::Move && first.section == 2 && first.oldParent == 100 &&
                     first.parent == 200 && first.name == "Keep" && last.kind == Command::Kind::Delete &&
                     last.section == 100 && last.parent == 1,"desired survivor did not escape before maximal deletion");
    EnvelopeInput peer; peer.commands = {move(4,100,200,"Outside"),create(6,2,"New"),create(7,100,"DiscardNew")};
    const auto peerScene = buildSceneCandidate(withSurvivor,encodeEnvelope(peer));
    const auto peerBefore = inspect(peerScene);
    outgoing.commands = escape.plan.commands;
    const auto final = buildSceneCandidate(peerScene,encodeEnvelope(outgoing));
    identities(final,{root(),spec(200,"/Q"),spec(4,"/Q/Outside"),spec(2,"/Q/Keep"),
                      spec(3,"/Q/Keep/Grand"),spec(6,"/Q/Keep/New")},{200,4,2,3});
    childTokens(final,"/Q/Keep",SdfChildrenKeys->PrimChildren,{"Grand","New"});
    requireNamespace(inspect(peerScene) == peerBefore,"survivor application mutated the peer scene");
}
void replacedParentIdentity() {
    namespaceContext = "unchanged child path with replaced parent identity";
    const auto base = baseScene({create(2,1,"P"),create(3,2,"A"),create(5,3,"a",SdfSpecTypeAttribute),
                                create(6,2,"root",SdfSpecTypeAttribute)});
    const std::vector<NamespaceSpec> desired{root(),spec(4,"/P"),spec(3,"/P/A"),
        spec(5,"/P/A.a",SdfSpecTypeAttribute),spec(6,"/P.root",SdfSpecTypeAttribute)};
    const auto result = good(base,desired,{3,5,6});
    requireNamespace(result.plan.candidate->pathForSection(2).IsEmpty(),"replaced parent ID survived");
    childTokens(result.plan.candidate,"/P",SdfChildrenKeys->PrimChildren,{"A"});
    childTokens(result.plan.candidate,"/P",SdfChildrenKeys->PropertyChildren,{"root"});
    childTokens(result.plan.candidate,"/P/A",SdfChildrenKeys->PropertyChildren,{"a"});
    requireNamespace(!result.scratch.empty(),"parent replacement did not exercise survivor parking");

    namespaceContext = "unchanged connection path with replaced attribute identity";
    const auto property = baseScene({create(2,1,"P"),create(3,2,"a",SdfSpecTypeAttribute),
                                    create(4,3,"/Target.x",SdfSpecTypeConnection)});
    const std::vector<NamespaceSpec> propertyDesired{root(),spec(2,"/P"),spec(5,"/P.a",SdfSpecTypeAttribute),
        spec(4,"/P.a[/Target.x]",SdfSpecTypeConnection)};
    const auto changed = good(property,propertyDesired,{2,4});
    requireNamespace(changed.plan.candidate->pathForSection(3).IsEmpty(),"replaced attribute ID survived");
    childTokens(changed.plan.candidate,"/P",SdfChildrenKeys->PropertyChildren,{"a"});
    childPaths(changed.plan.candidate,"/P.a",SdfChildrenKeys->ConnectionChildren,{"/Target.x"});
}
void swapsAndAllocator() {
    namespaceContext = "namespace swaps and scratch ownership";
    const auto base = baseScene({create(2,1,"A"),create(3,1,"B"),create(4,2,"a",SdfSpecTypeAttribute),create(5,3,"b",SdfSpecTypeAttribute)});
    const std::vector<NamespaceSpec> desired{root(),spec(2,"/B"),spec(3,"/A"),spec(4,"/B.a",SdfSpecTypeAttribute),spec(5,"/A.b",SdfSpecTypeAttribute)};
    const auto first = good(base,desired,{2,3,4,5});
    requireNamespace(!first.scratch.empty(),"swap fixture did not exercise allocator");
    childTokens(first.plan.candidate,"/B",SdfChildrenKeys->PropertyChildren,{"a"});
    childTokens(first.plan.candidate,"/A",SdfChildrenKeys->PropertyChildren,{"b"});
    const auto again = good(base,desired,{2,3,4,5});
    EnvelopeInput a,b; a.commands = first.plan.commands; b.commands = again.plan.commands;
    requireNamespace(first.scratch == again.scratch && encodeEnvelope(a) == encodeEnvelope(b),"deterministic allocator produced nondeterministic command identities");
    rejectReason(base,desired,UnsupportedOperation::Reason::MoveCycle);
    for (uint64_t collision : {uint64_t(0),uint64_t(1),uint64_t(2),uint64_t(5)})
        reject(base,desired,ErrorKind::Malformed,[collision] { return collision; });
    const auto three = baseScene({create(2,1,"A"),create(3,1,"B"),create(4,1,"C")});
    good(three,{root(),spec(2,"/B"),spec(3,"/C"),spec(4,"/A")},{2,3,4});
}
std::vector<NamespaceSpec> topology(const std::string& p) {
    const std::vector<std::tuple<uint64_t,const char*,SdfSpecType>> suffixes{
        {2,"",SdfSpecTypePrim},{3,".a",SdfSpecTypeAttribute},{4,".a[/Target.a]",SdfSpecTypeConnection},
        {5,".a.expression",SdfSpecTypeExpression},{6,".a.mapper[/T.rel[/U]]",SdfSpecTypeMapper},
        {7,".a.mapper[/T.rel[/U]].arg",SdfSpecTypeMapperArg},{8,".rel",SdfSpecTypeRelationship},
        {9,".rel[/Target]",SdfSpecTypeRelationshipTarget},{10,".rel[/Target].weight",SdfSpecTypeAttribute},
        {11,".rel.expression",SdfSpecTypeExpression},{12,"{v=}",SdfSpecTypeVariantSet},{13,"{v=x}",SdfSpecTypeVariant},
        {14,"{v=x}Child",SdfSpecTypePrim},{15,"{v=x}.inside",SdfSpecTypeAttribute},{16,"{v=x}.rel",SdfSpecTypeRelationship},
        {17,"{v=x}{nested=}",SdfSpecTypeVariantSet},{18,"{v=x}{nested=y}",SdfSpecTypeVariant},{19,"/Sub",SdfSpecTypePrim}};
    std::vector<NamespaceSpec> result{root()};
    for (const auto& [id,suffix,type] : suffixes) result.push_back({id,type,SdfPath(p+suffix)});
    return result;
}
void verifyTopologyChildren(const WireScenePtr& scene,const std::string& prefix,bool ordered = true) {
    const auto at = [&](const char* suffix) { return prefix+suffix; };
    childTokens(scene,"/",SdfChildrenKeys->PrimChildren,{prefix.c_str()+1},ordered);
    childTokens(scene,prefix.c_str(),SdfChildrenKeys->PrimChildren,{"Sub"},ordered);
    childTokens(scene,prefix.c_str(),SdfChildrenKeys->PropertyChildren,{"a","rel"},ordered);
    childTokens(scene,prefix.c_str(),SdfChildrenKeys->VariantSetChildren,{"v"},ordered);
    childPaths(scene,at(".a").c_str(),SdfChildrenKeys->ConnectionChildren,{"/Target.a"});
    childTokens(scene,at(".a").c_str(),SdfChildrenKeys->ExpressionChildren,{"expression"},ordered);
    childPaths(scene,at(".a").c_str(),SdfChildrenKeys->MapperChildren,{"/T.rel[/U]"});
    childTokens(scene,at(".a.mapper[/T.rel[/U]]").c_str(),SdfChildrenKeys->MapperArgChildren,{"arg"},ordered);
    childPaths(scene,at(".rel").c_str(),SdfChildrenKeys->RelationshipTargetChildren,{"/Target"});
    childTokens(scene,at(".rel").c_str(),SdfChildrenKeys->ExpressionChildren,{"expression"},ordered);
    childTokens(scene,at(".rel[/Target]").c_str(),SdfChildrenKeys->PropertyChildren,{"weight"},ordered);
    childTokens(scene,at("{v=}").c_str(),SdfChildrenKeys->VariantChildren,{"x"},ordered);
    childTokens(scene,at("{v=x}").c_str(),SdfChildrenKeys->PrimChildren,{"Child"},ordered);
    childTokens(scene,at("{v=x}").c_str(),SdfChildrenKeys->PropertyChildren,{"inside","rel"},ordered);
    childTokens(scene,at("{v=x}").c_str(),SdfChildrenKeys->VariantSetChildren,{"nested"},ordered);
    childTokens(scene,at("{v=x}{nested=}").c_str(),SdfChildrenKeys->VariantChildren,{"y"},ordered);
}
void allChildPolicies() {
    namespaceContext = "all USD namespace child policies";
    auto initial = topology("/P");
    const auto empty = baseScene({});
    const auto made = good(empty,initial,{});
    verifyTopologyChildren(made.plan.candidate,"/P",false);
    const auto base = baseScene({create(2,1,"P"),create(3,2,"a",SdfSpecTypeAttribute),
        create(4,3,"/Target.a",SdfSpecTypeConnection),create(5,3,"expression",SdfSpecTypeExpression),
        create(6,3,"/T.rel[../U]",SdfSpecTypeMapper),create(7,6,"arg",SdfSpecTypeMapperArg),
        create(8,2,"rel",SdfSpecTypeRelationship),create(9,8,"/Target",SdfSpecTypeRelationshipTarget),
        create(10,9,"weight",SdfSpecTypeAttribute),create(11,8,"expression",SdfSpecTypeExpression),
        create(12,2,"v",SdfSpecTypeVariantSet),create(13,12,"x",SdfSpecTypeVariant),create(14,13,"Child"),
        create(15,13,"inside",SdfSpecTypeAttribute),create(16,13,"rel",SdfSpecTypeRelationship),
        create(17,13,"nested",SdfSpecTypeVariantSet),create(18,17,"y",SdfSpecTypeVariant),create(19,2,"Sub")});
    std::set<uint64_t> retained; for (uint64_t id = 2; id <= 19; ++id) retained.insert(id);
    identities(base,initial,retained); verifyTopologyChildren(base,"/P");
    const auto same = good(base,initial,retained);
    requireNamespace(same.plan.commands.empty() && same.scratch.empty() && inspect(same.plan.candidate) == inspect(base),
                     "unchanged namespace changed fields/order or allocated IDs");
    auto moved = topology("/Q"); std::reverse(moved.begin(),moved.end());
    const auto changed = good(base,moved,retained);
    verifyTopologyChildren(changed.plan.candidate,"/Q");
    requireNamespace(changed.plan.candidate->pathForSection(6) == SdfPath("/Q.a.mapper[/T.rel[/U]]"),
                     "ancestor move rebound canonical nested mapper target");
    auto renamedVariant = initial;
    for (auto& item : renamedVariant) {
        auto path = item.path.GetString(); const auto position = path.find("{v=x}");
        if (position != std::string::npos) { path.replace(position,5,"{v=high-low}"); item.path = SdfPath(path); }
    }
    const auto variant = good(base,renamedVariant,retained);
    childTokens(variant.plan.candidate,"/P{v=}",SdfChildrenKeys->VariantChildren,{"high-low"});
    childTokens(variant.plan.candidate,"/P{v=high-low}",SdfChildrenKeys->VariantSetChildren,{"nested"});
}
void namespaceHistories(uint64_t seed) {
    std::cout << "namespace_seed=" << seed << " steps=64" << std::endl;
    std::mt19937_64 random(seed);
    struct Prim { uint64_t parent; std::string name; };
    // The oracle mutates IDs, parent IDs and names directly; it never obtains
    // expected paths from a plan, a candidate or a wire decoder.
    std::map<uint64_t,Prim> model{{2,{1,"Left"}},{3,{1,"Right"}},
        {10,{2,"N10"}},{11,{2,"N11"}},{12,{3,"N12"}},{13,{3,"N13"}}};
    std::set<uint64_t> marked{2,3,10,11,12,13};
    auto scene = baseScene({create(2,1,"Left"),create(3,1,"Right"),
        create(10,2,"N10"),create(11,2,"N11"),create(12,3,"N12"),create(13,3,"N13")});
    uint64_t next = 20, fingerprint = UINT64_C(14695981039346656037);
    std::vector<std::pair<WireScenePtr,State>> retained;
    for (unsigned step = 0; step < 64; ++step) {
        namespaceContext = "namespace seed "+std::to_string(seed)+" step "+std::to_string(step);
        std::vector<uint64_t> leaves;
        for (const auto& [id,prim] : model) if (id >= 10) leaves.push_back(id);
        const auto action = random()%5;
        if (leaves.empty() || (action == 0 && leaves.size() < 10)) {
            const auto id = next++;
            model.emplace(id,Prim{1+random()%3,"N"+std::to_string(id)});
        } else {
            const auto selected = size_t(random()%leaves.size()); const auto id = leaves[selected];
            if (action == 1) { model.erase(id); marked.erase(id); }
            else if (action == 2) model.at(id).parent = 1+random()%3;
            else if (action == 3 && leaves.size() > 1) {
                const auto other = leaves[(selected+1+random()%(leaves.size()-1))%leaves.size()];
                std::swap(model.at(id).name,model.at(other).name);
            } else model.at(id).name = "R"+std::to_string(step);
        }
        std::vector<NamespaceSpec> expected{root()};
        std::map<uint64_t,std::string> paths{{1,"/"},{2,"/Left"},{3,"/Right"}};
        std::map<uint64_t,std::vector<TfToken>> children;
        for (const auto& [id,prim] : model) {
            const auto path = prim.parent == 1 ? "/"+prim.name : paths.at(prim.parent)+"/"+prim.name;
            paths[id] = path; expected.push_back({id,SdfSpecTypePrim,SdfPath(path)});
            children[prim.parent].emplace_back(prim.name);
        }
        auto plan = good(scene,expected,marked,100000+uint64_t(step)*1000);
        scene = std::move(plan.plan.candidate);
        for (const auto& [id,path] : paths) {
            if (id != 1 && !model.count(id)) continue;
            const auto found = children.find(id);
            const VtValue value = scene->data()->Get(SdfPath(path),SdfChildrenKeys->PrimChildren);
            if (found == children.end()) requireNamespace(value.IsEmpty(),"history retained absent child membership");
            else {
                requireNamespace(value.IsHolding<std::vector<TfToken>>(),"history child membership type differs");
                auto actual = value.Get<std::vector<TfToken>>(), wanted = found->second;
                std::sort(actual.begin(),actual.end()); std::sort(wanted.begin(),wanted.end());
                requireNamespace(actual == wanted,"history child membership differs from parent-ID oracle");
            }
        }
        for (const auto& [id,prim] : model) {
            for (unsigned byte = 0; byte < 8; ++byte) { fingerprint ^= uint8_t(id >> (8*byte)); fingerprint *= UINT64_C(1099511628211); }
            for (unsigned char c : paths.at(id)) { fingerprint ^= c; fingerprint *= UINT64_C(1099511628211); }
            fingerprint ^= uint8_t(marked.count(id)); fingerprint *= UINT64_C(1099511628211);
        }
        if (step%8 == 0) retained.emplace_back(scene,inspect(scene));
    }
    for (const auto& [owner,state] : retained) requireNamespace(inspect(owner) == state,"history changed a retained prior owner");
    std::cout << "namespace_seed=" << seed << " steps=64 fingerprint=" << fingerprint << std::endl;
}
// The section limit bounds the desired graph's size on both planners: on a
// base whose path count equals maxSections, moves and a delete paired with a
// create keep the count and plan identically, and an unpaired create is a
// resource-limit rejection on both.
void changesAtSectionLimit() {
    namespaceContext = "namespace changes at the section limit";
    const auto base = baseScene({create(2,1,"A"),create(3,2,"B"),create(4,1,"C")});
    SceneNamespaceLimits limits; limits.scene.maxSections = base->pathCount();
    const auto same = [&](const std::vector<NamespaceSpec>& desired,const NamespaceChanges& changes) {
        const auto whole = planSceneNamespace(base,desired,{},limits);
        const auto local = planSceneNamespaceChanges(base,changes,{},limits);
        EnvelopeInput wholeCommands, localCommands;
        wholeCommands.commands = whole.commands; localCommands.commands = local.commands;
        requireNamespace(encodeEnvelope(wholeCommands) == encodeEnvelope(localCommands) &&
                         inspect(whole.candidate) == inspect(local.candidate),"footprint plan differs from the whole-scene plan");
        identities(local.candidate,desired,{2,3,4});
    };
    same({root(),spec(2,"/R"),spec(3,"/R/B"),spec(4,"/C")},{{spec(2,"/R"),spec(3,"/R/B")},{}});
    same({root(),spec(2,"/A"),spec(3,"/C/B"),spec(4,"/C")},{{spec(3,"/C/B")},{}});
    same({root(),spec(2,"/A"),spec(3,"/A/B"),spec(5,"/D")},{{spec(5,"/D")},{4}});
    reject(base,{root(),spec(2,"/A"),spec(3,"/A/B"),spec(4,"/C"),spec(5,"/D")},ErrorKind::ResourceLimit,{},limits);
    bool rejected = false;
    try { (void)planSceneNamespaceChanges(base,{{spec(5,"/D")},{}},{},limits); }
    catch (const DecodeError& error) { rejected = error.kind == ErrorKind::ResourceLimit; }
    requireNamespace(rejected,"footprint plan accepted a create past the section limit");
}
void invalidAndLimits() {
    namespaceContext = "namespace invalid graphs and bounded failure";
    const auto base = baseScene({create(2,1,"A"),create(3,1,"B"),create(4,2,"a",SdfSpecTypeAttribute)});
    const std::vector<NamespaceSpec> unchanged{root(),spec(2,"/A"),spec(3,"/B"),spec(4,"/A.a",SdfSpecTypeAttribute)};
    const auto noOpBefore = inspect(base);
    SceneNamespaceLimits noOpLimits;
    const size_t supplied = noOpLimits.maxWork+4096;
    WireSceneWorkBudget shared{supplied};
    const auto noOp = planSceneNamespace(base,unchanged,{},noOpLimits,&shared);
    requireNamespace(noOp.commands.empty() && inspect(noOp.candidate) == noOpBefore && inspect(base) == noOpBefore,
                     "successful no-op changed retained fields or namespace");
    requireNamespace(noOp.workUsed > 0 && noOp.workUsed <= supplied && shared.remaining == supplied-noOp.workUsed,
                     "shared namespace budget debit differs from reported work");
    noOpLimits.scene.maxFields = 0;
    reject(base,unchanged,ErrorKind::ResourceLimit,{},noOpLimits);
    // Even an unchanged namespace must validate existing authored fields under
    // this call's scene limits. Its measured work cannot be omitted either.
    noOpLimits = {};
    WireSceneWorkBudget oneShort{noOp.workUsed-1};
    reject(base,unchanged,ErrorKind::ResourceLimit,{},noOpLimits,&oneShort);
    requireNamespace(oneShort.remaining < noOp.workUsed-1,"failed namespace planning did not charge any shared work");
    noOpLimits.maxWork = noOp.workUsed-1;
    reject(base,unchanged,ErrorKind::ResourceLimit,{},noOpLimits);
    auto invalid = unchanged; invalid[3].type = SdfSpecTypeRelationship;
    rejectReason(base,invalid,UnsupportedOperation::Reason::IdentityTypeChange);
    invalid = unchanged; invalid[3].path = SdfPath("/Missing.a");
    reject(base,invalid,ErrorKind::Malformed);
    invalid = unchanged; invalid.push_back(spec(20,"/A.a.expression"));
    reject(base,invalid,ErrorKind::Malformed);
    invalid = unchanged; invalid.push_back(spec(2,"/Other"));
    reject(base,invalid,ErrorKind::Malformed);
    invalid = unchanged; invalid.push_back(spec(20,"/A"));
    reject(base,invalid,ErrorKind::Malformed);
    invalid = unchanged; invalid.erase(invalid.begin());
    reject(base,invalid,ErrorKind::Malformed);
    invalid = unchanged; invalid[0].id = 7;
    reject(base,invalid,ErrorKind::Malformed);
    invalid = unchanged; invalid.push_back(spec(20,"/A.a.mapper[Target.a]",SdfSpecTypeMapper));
    rejectReason(base,invalid,UnsupportedOperation::Reason::IdentityRealization);
    const std::vector<NamespaceSpec> swap{root(),spec(2,"/B"),spec(3,"/A"),spec(4,"/B.a",SdfSpecTypeAttribute)};
    const auto allocate = [] { return uint64_t(10000); };
    SceneNamespaceLimits limits; limits.maxCommands = 1;
    reject(base,swap,ErrorKind::ResourceLimit,allocate,limits);
    limits = {}; limits.maxMetadataBytes = 1;
    reject(base,swap,ErrorKind::ResourceLimit,allocate,limits);
    limits = {}; limits.maxWork = 1;
    reject(base,swap,ErrorKind::ResourceLimit,allocate,limits);
    limits = {}; limits.scene.maxSections = 2;
    reject(base,swap,ErrorKind::ResourceLimit,allocate,limits);
    WireSceneWorkBudget exhausted{1};
    reject(base,swap,ErrorKind::ResourceLimit,allocate,{},&exhausted);
    // A scratch identity cannot reuse a removed ID or a not-yet-created ID.
    const auto withDeleted = baseScene({create(2,1,"A"),create(3,1,"B"),create(50,1,"Deleted")});
    const std::vector<NamespaceSpec> withNew{root(),spec(2,"/B"),spec(3,"/A"),spec(60,"/New")};
    reject(withDeleted,withNew,ErrorKind::Malformed,[] { return uint64_t(50); });
    reject(withDeleted,withNew,ErrorKind::Malformed,[] { return uint64_t(60); });
    const auto before = inspect(base); bool threw = false;
    try { (void)planSceneNamespace(base,swap,[]()->uint64_t { throw std::bad_alloc(); }); }
    catch (const std::bad_alloc&) { threw = true; }
    requireNamespace(threw && inspect(base) == before,"allocator exception changed retained scene");
}
} // namespace

void runSceneNamespaceTests(uint64_t seed) {
    createsDeletesAndReplacement(); movingSubtrees(); maximalRecursiveDeletes(); replacedParentIdentity(); swapsAndAllocator(); allChildPolicies(); invalidAndLimits(); changesAtSectionLimit(); namespaceHistories(seed);
    std::cout << "independent namespace identity, child policy, scratch and quota contracts passed" << std::endl;
}
