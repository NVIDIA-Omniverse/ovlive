// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "SceneSeed.h"
#include "FieldEditIntent.h"
#include "captured_log.h"
#include <pxr/base/tf/errorMark.h>
#include <pxr/base/vt/array.h>
#include <pxr/base/vt/dictionary.h>
#include <pxr/usd/sdf/assetPath.h>
#include <pxr/usd/sdf/data.h>
#include <pxr/usd/sdf/fileFormat.h>
#include <pxr/usd/sdf/layer.h>
#include <pxr/usd/sdf/primSpec.h>
#include <pxr/usd/sdf/schema.h>
#include <algorithm>
#include <cstring>
#include <map>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace PXR_NS;
using namespace omni::usd::live;
namespace {
void check(bool value,const char* message) { if(!value) throw std::runtime_error(message); }
class Access final : public SdfFileFormat {
public:
    static SdfAbstractDataConstPtr data(const SdfLayer& layer) { return _GetLayerData(layer); }
};
struct Spec {
    SdfSpecType type;
    std::map<TfToken,std::string> fields;
    bool operator==(const Spec& other) const { return type==other.type && fields==other.fields; }
};
using Contents=std::map<SdfPath,Spec>;
Contents contents(const SdfAbstractData& source) {
    struct Visitor final : SdfAbstractDataSpecVisitor {
        Contents result;
        bool VisitSpec(const SdfAbstractData& source,const SdfPath& path) override {
            Spec spec{source.GetSpecType(path),{}};
            for(const auto& key:source.List(path))
                check(spec.fields.emplace(key,wire::encodeValue(source.Get(path,key))).second,"seed oracle duplicate field");
            check(result.emplace(path,std::move(spec)).second,"seed oracle duplicate spec"); return true;
        }
        void Done(const SdfAbstractData&) override {}
    } visitor;
    source.VisitSpecs(&visitor); return visitor.result;
}
template<class F> void rejected(F action,wire::ErrorKind kind) {
    bool failed=false;
    try { action(); } catch(const wire::DecodeError& error) {
        check(error.kind==kind,"seed compiler rejection category differs"); failed=true;
    }
    check(failed,"seed compiler accepted invalid or over-limit source");
}
VtValue tokens(std::initializer_list<const char*> names) {
    std::vector<TfToken> result; for(auto name:names) result.emplace_back(name); return VtValue(result);
}
VtValue paths(std::initializer_list<const char*> names) {
    std::vector<SdfPath> result; for(auto name:names) result.emplace_back(name); return VtValue(result);
}
void field(SdfData& data,const char* path,const TfToken& name,const VtValue& value) {
    data.Set(SdfPath(path),name,value);
}
SdfDataRefPtr rootData() {
    auto data=TfCreateRefPtr(new SdfData); data->CreateSpec(SdfPath("/"),SdfSpecTypePseudoRoot); return data;
}
void verify(const SdfAbstractData& source,const SceneSeed& seed,std::string_view format={}) {
    const auto expected=contents(source);
    check(seed.scene && seed.workUsed && contents(*seed.scene->data())==expected,
          "seed compiler changed exact authored spec/field projection");
    const auto envelope=wire::decodeEnvelope(seed.bytes);
    check(envelope.isDiff && envelope.baseVersion==0 && envelope.sourceFormat==format &&
          envelope.commands.size()==expected.size(),
          "seed is not one complete zero-base snapshot with canonical samples");
    for(const auto& command:envelope.commands)
        check(command.kind==wire::Command::Kind::DiffChange,"seed emitted incremental structural command");
    uint64_t id=0;
    for(const auto& [path,spec]:expected) {
        (void)spec;
        check(seed.scene->sectionForPath(path)==++id,"seed IDs are not deterministic SdfPath order");
    }
    check(seed.scene->sectionForPath(SdfPath("/"))==1 && seed.scene->sourceFormat()==format,
          "seed lost root identity or source format");
    const auto rebuilt=buildSceneCandidate({},seed.bytes);
    check(contents(*rebuilt->data())==expected,"owned seed bytes do not rebuild exact source");
}
void stockUsdAndOwnership() {
    const char* text=R"USD(#usda 1.0
(
    customLayerData = { int revision = 7 }
    defaultPrim = "P"
    subLayers = [@./base.usda@ (offset = 2; scale = 0.5)]
)
def Xform "P" (
    prepend references = @./ref.usda@</Root> (offset = 3; scale = 2)
    prepend payload = @./payload.usda@</Payload>
    variants = { string look = "red" }
    prepend variantSets = "look"
) {
    custom asset texture = @./texture.tx@
    custom int[] numbers = [1, -2, 3]
    custom double animated = -0
    double animated.timeSamples = { 1: 1.25, 2: 2.5 }
    double driven.connect = </Other.output>
    custom rel targets = [</Other>, </P/Z>]
    def Scope "Z" {}
    def Scope "A" {}
    variantSet "look" = {
        "red" {
            custom string label = "warm"
        }
        "blue" {
            custom string label = "cool"
        }
    }
}
def Scope "Other" {
    double output = 9
}
)USD";
    SceneSeed seed; Contents expected;
    {
        auto layer=SdfLayer::CreateAnonymous("seed-source.usda");
        check(layer && layer->ImportFromString(text),"independent stock USD seed fixture failed");
        layer->SetRootPrimOrder({TfToken("Other"),TfToken("P")});
        auto prim=layer->GetPrimAtPath(SdfPath("/P"));
        check(bool(prim),"stock seed prim missing");
        prim->SetNameChildrenOrder({TfToken("A"),TfToken("Z")});
        prim->SetPropertyOrder({TfToken("targets"),TfToken("animated"),TfToken("texture")});
        auto source=Access::data(*layer); expected=contents(*source);
        seed=compileSceneSeed(*source,"usda"); verify(*source,seed,"usda");
        check(contents(*source)==expected,"seed compiler mutated original stock layer");
        layer->SetField(SdfPath("/P.animated"),SdfFieldKeys->Default,VtValue(77.0));
        check(contents(*seed.scene->data())==expected,"seed retained mutable source backing");
    }
    check(contents(*seed.scene->data())==expected && contents(*buildSceneCandidate({},seed.bytes)->data())==expected,
          "seed lost contents after all original USD owners retired");
}
struct Node { const char* path; SdfSpecType type; const char* parent; const char* name; };
const Node nodes[]={
    {"/",SdfSpecTypePseudoRoot,"",""}, {"/P",SdfSpecTypePrim,"/","P"},
    {"/P/Z",SdfSpecTypePrim,"/P","Z"}, {"/P/A",SdfSpecTypePrim,"/P","A"},
    {"/P.a",SdfSpecTypeAttribute,"/P","a"},
    {"/P.a[../Target.a]",SdfSpecTypeConnection,"/P.a","../Target.a"},
    {"/P.a.expression",SdfSpecTypeExpression,"/P.a","expression"},
    {"/P.a.mapper[/P/Target.a]",SdfSpecTypeMapper,"/P.a","/P/Target.a"},
    {"/P.a.mapper[/P/Target.a].arg",SdfSpecTypeMapperArg,"/P.a.mapper[/P/Target.a]","arg"},
    {"/P.rel",SdfSpecTypeRelationship,"/P","rel"},
    {"/P.rel[../Target]",SdfSpecTypeRelationshipTarget,"/P.rel","../Target"},
    {"/P.rel[../Target].weight",SdfSpecTypeAttribute,"/P.rel[../Target]","weight"},
    {"/P.rel.expression",SdfSpecTypeExpression,"/P.rel","expression"},
    {"/P{look=}",SdfSpecTypeVariantSet,"/P","look"},
    {"/P{look=red}",SdfSpecTypeVariant,"/P{look=}","red"},
    {"/P{look=red}Inside",SdfSpecTypePrim,"/P{look=red}","Inside"},
    {"/P{look=red}.inside",SdfSpecTypeAttribute,"/P{look=red}","inside"},
    {"/P{look=red}{detail=}",SdfSpecTypeVariantSet,"/P{look=red}","detail"},
    {"/P{look=red}{detail=high}",SdfSpecTypeVariant,"/P{look=red}{detail=}","high"}
};
SdfDataRefPtr allPolicies() {
    auto data=TfCreateRefPtr(new SdfData);
    for(const auto& node:nodes) data->CreateSpec(SdfPath(node.path),node.type);
    field(*data,"/",SdfChildrenKeys->PrimChildren,tokens({"P"}));
    field(*data,"/P",SdfChildrenKeys->PrimChildren,tokens({"Z","A"}));
    field(*data,"/P",SdfChildrenKeys->PropertyChildren,tokens({"rel","a"}));
    field(*data,"/P",SdfChildrenKeys->VariantSetChildren,tokens({"look"}));
    field(*data,"/P.a",SdfChildrenKeys->ConnectionChildren,paths({"../Target.a"}));
    field(*data,"/P.a",SdfChildrenKeys->ExpressionChildren,tokens({"expression"}));
    field(*data,"/P.a",SdfChildrenKeys->MapperChildren,paths({"/P/Target.a"}));
    field(*data,"/P.a.mapper[/P/Target.a]",SdfChildrenKeys->MapperArgChildren,tokens({"arg"}));
    field(*data,"/P.rel",SdfChildrenKeys->RelationshipTargetChildren,paths({"../Target"}));
    field(*data,"/P.rel",SdfChildrenKeys->ExpressionChildren,tokens({"expression"}));
    field(*data,"/P.rel[../Target]",SdfChildrenKeys->PropertyChildren,tokens({"weight"}));
    field(*data,"/P{look=}",SdfChildrenKeys->VariantChildren,tokens({"red"}));
    field(*data,"/P{look=red}",SdfChildrenKeys->PrimChildren,tokens({"Inside"}));
    field(*data,"/P{look=red}",SdfChildrenKeys->PropertyChildren,tokens({"inside"}));
    field(*data,"/P{look=red}",SdfChildrenKeys->VariantSetChildren,tokens({"detail"}));
    field(*data,"/P{look=red}{detail=}",SdfChildrenKeys->VariantChildren,tokens({"high"}));
    field(*data,"/P.rel[../Target].weight",SdfFieldKeys->Default,VtValue(17));
    return data;
}
class Enumeration final : public SdfData {
public:
    std::vector<SdfPath> paths;
    bool reverseFields=false,duplicateField=false,emptyField=false;
    SdfPath forbiddenList;
    bool Has(const SdfPath& path,const TfToken& name,VtValue* value) const override {
        if(emptyField && path==SdfPath("/") && name==TfToken("presentEmpty")) {
            if(value) *value=VtValue();
            return true;
        }
        return SdfData::Has(path,name,value);
    }
    std::vector<TfToken> List(const SdfPath& path) const override {
        check(path!=forbiddenList,"seed enumerated fields before rejecting over-depth spec");
        auto result=SdfData::List(path);
        if(reverseFields) std::reverse(result.begin(),result.end());
        if(path==SdfPath("/")) {
            if(duplicateField && !result.empty()) result.push_back(result.front());
            if(emptyField) result.emplace_back("presentEmpty");
        }
        return result;
    }
protected:
    void _VisitSpecs(SdfAbstractDataSpecVisitor* visitor) const override {
        for(const auto& path:paths) if(!visitor->VisitSpec(*this,path)) break;
    }
};
void childPoliciesAndEnumeration() {
    auto source=allPolicies(); const auto before=contents(*source);
    auto seed=compileSceneSeed(*source); verify(*source,seed);
    std::map<SdfPath,std::pair<uint64_t,std::string>> relationships;
    seed.scene->visitSections(64,[&](const WireSectionView& view) {
        relationships.emplace(view.path,std::make_pair(view.parent,std::string(view.name))); return true;
    });
    for(const auto& node:nodes) {
        const auto& actual=relationships.at(SdfPath(node.path));
        check(actual.first==(*node.parent?seed.scene->sectionForPath(SdfPath(node.parent)):0) && actual.second==node.name,
              "seed confused variant ancestry, relative target or mapper identity");
    }
    auto shuffled=TfCreateRefPtr(new Enumeration); shuffled->CopyFrom(source);
    for(const auto& [path,spec]:before) { (void)spec; shuffled->paths.push_back(path); }
    std::reverse(shuffled->paths.begin(),shuffled->paths.end()); shuffled->reverseFields=true;
    check(compileSceneSeed(*shuffled).bytes==seed.bytes,"seed depends on source VisitSpecs/List enumeration order");
    check(contents(*source)==before,"all-policy seed mutated original fields");
}
void typedPresenceAndBits() {
    const auto layer=SdfLayer::CreateAnonymous(".usda");
    check(layer && layer->ImportFromString("#usda 1.0\ndef \"P\" {\n double a.timeSamples = {}\n}\n"),
          "stock USD rejected explicit empty animation");
    const auto imported=contents(*Access::data(*layer));
    rejected([&] { (void)compileSceneSeed(*Access::data(*layer)); },wire::ErrorKind::Unsupported);
    check(contents(*Access::data(*layer))==imported,"empty animation rejection changed imported USD");
    auto source=rootData();
    field(*source,"/",SdfChildrenKeys->PrimChildren,tokens({}));
    rejected([&] { (void)compileSceneSeed(*source); },wire::ErrorKind::Unsupported);
    source->Erase(SdfPath("/"),SdfChildrenKeys->PrimChildren);
    field(*source,"/",TfToken("emptyArray"),VtValue(VtIntArray{}));
    field(*source,"/",TfToken("emptyDictionary"),VtValue(VtDictionary{}));
    field(*source,"/",SdfFieldKeys->TimeSamples,VtValue(SdfTimeSampleMap{}));
    rejected([&] { (void)compileSceneSeed(*source); },wire::ErrorKind::Unsupported);
    source->Erase(SdfPath("/"),SdfFieldKeys->TimeSamples);
    field(*source,"/",TfToken("negativeZero"),VtValue(-0.0));
    uint64_t nanBits=UINT64_C(0x7ff8000000000042); double nan;
    std::memcpy(&nan,&nanBits,sizeof(nan)); field(*source,"/",TfToken("nan"),VtValue(nan));
    SdfTimeSampleMap times; times.emplace(-0.0,VtValue(-0.0)); times.emplace(2.0,VtValue(nan));
    field(*source,"/",TfToken("sampleMap"),VtValue(times));
    const SdfAssetPath asset("relative.tx","/cached/relative.tx");
    field(*source,"/",TfToken("asset"),VtValue(asset));
    const auto before=contents(*source); auto seed=compileSceneSeed(*source); verify(*source,seed);
    check(source->Get(SdfPath("/"),TfToken("asset"))==VtValue(asset),"seed altered source resolved asset cache");
    const auto envelope=wire::decodeEnvelope(seed.bytes);
    for(const auto& value:envelope.fields)
        check(value.value.bytes()==before.at(SdfPath("/")).fields.at(TfToken(value.name)),
              "seed field bytes lost typed empty, NaN payload or signed-zero bits");
    check(!seed.scene->data()->Has(SdfPath("/"),TfToken("absent")),"seed invented absent field");

}
void invalidSources() {
    using wire::ErrorKind;
    auto empty=TfCreateRefPtr(new SdfData);
    rejected([&] { (void)compileSceneSeed(*empty); },ErrorKind::Malformed);
    auto data=rootData(); data->CreateSpec(SdfPath("/Missing/Child"),SdfSpecTypePrim);
    rejected([&] { (void)compileSceneSeed(*data); },ErrorKind::Malformed);
    data=rootData(); data->CreateSpec(SdfPath("/P"),SdfSpecTypeAttribute);
    rejected([&] { (void)compileSceneSeed(*data); },ErrorKind::Malformed);
    data=rootData(); data->CreateSpec(SdfPath("/P"),SdfSpecTypePrim);
    rejected([&] { (void)compileSceneSeed(*data); },ErrorKind::Unsupported);
    field(*data,"/",SdfChildrenKeys->PrimChildren,tokens({"Wrong"}));
    const auto inconsistent=contents(*data);
    rejected([&] { (void)compileSceneSeed(*data); },ErrorKind::Unsupported);
    check(contents(*data)==inconsistent,"rejected child membership changed original source");
    field(*data,"/",SdfChildrenKeys->PrimChildren,tokens({"P","P"}));
    rejected([&] { (void)compileSceneSeed(*data); },ErrorKind::Unsupported);
    auto listed=TfCreateRefPtr(new Enumeration); listed->CreateSpec(SdfPath("/"),SdfSpecTypePseudoRoot);
    listed->paths={SdfPath("/"),SdfPath("/")};
    rejected([&] { (void)compileSceneSeed(*listed); },ErrorKind::Malformed);
    listed->paths.resize(1); listed->emptyField=true;
    rejected([&] { (void)compileSceneSeed(*listed); },ErrorKind::Unsupported);
    listed->emptyField=false; listed->duplicateField=true;
    field(*listed,"/",TfToken("counter"),VtValue(1));
    rejected([&] { (void)compileSceneSeed(*listed); },ErrorKind::Malformed);
    data=rootData(); field(*data,"/",TfToken("unsupportedShort"),VtValue(short(7)));
    rejected([&] { (void)compileSceneSeed(*data); },ErrorKind::Unsupported);
    data=allPolicies();
    data->MoveSpec(SdfPath("/P.a.mapper[/P/Target.a].arg"),SdfPath("/P.a.mapper[Target.a].arg"));
    data->MoveSpec(SdfPath("/P.a.mapper[/P/Target.a]"),SdfPath("/P.a.mapper[Target.a]"));
    field(*data,"/P.a",SdfChildrenKeys->MapperChildren,paths({"Target.a"}));
    rejected([&] { (void)compileSceneSeed(*data); },ErrorKind::Unsupported);
}
void quotasAndAtomicity() {
    auto source=allPolicies(); const auto before=contents(*source); const auto good=compileSceneSeed(*source);
    SceneSeedLimits exact; exact.maxWork=good.workUsed;
    check(compileSceneSeed(*source,{},exact).bytes==good.bytes,"exact seed work limit rejected");
    for(unsigned index=0;index<12;++index) {
        SceneSeedLimits limits;
        switch(index) {
        case 0: limits.maxWork=good.workUsed-1; break;
        case 1: limits.maxMetadataBytes=0; break;
        case 2: limits.scene.maxSections=before.size()-1; break;
        case 3: limits.scene.maxFields=1; break;
        case 4: limits.scene.maxPathDepth=1; break;
        case 5: limits.envelope.content.maxBytes=good.bytes.size()-1; break;
        case 6: limits.values.maxElements=1; break;
        case 7: limits.values.maxStringBytes=1; break;
        case 8: limits.scene.values.maxEncodedBytes=1; break;
        case 9: limits.scene.retainedValues.maxEncodedBytes=1; break;
        case 10: limits.envelope.content.maxReferencedValueBytes=1; break;
        case 11: limits.envelope.content.maxStringBytes=1; break;
        }
        auto published=good;
        rejected([&] { published=compileSceneSeed(*source,{},limits); },wire::ErrorKind::ResourceLimit);
        check(published.bytes==good.bytes && published.scene==good.scene && contents(*source)==before,
              "failed seed compile changed caller output or original source");
    }
    // Every scalar independently fits; their aggregate must spend one budget.
    auto many=rootData();
    for(unsigned i=0;i<64;++i) field(*many,"/",TfToken("value"+std::to_string(i)),VtValue(int(i)));
    SceneSeedLimits limits; limits.values.maxEncodedBytes=32;
    auto single=rootData(); field(*single,"/",TfToken("value0"),VtValue(0));
    verify(*single,compileSceneSeed(*single,{},limits));
    rejected([&] { (void)compileSceneSeed(*many,{},limits); },wire::ErrorKind::ResourceLimit);
    limits={}; limits.values.maxElements=8;
    verify(*single,compileSceneSeed(*single,{},limits));
    rejected([&] { (void)compileSceneSeed(*many,{},limits); },wire::ErrorKind::ResourceLimit);
    // Flat namespace depth must be checked before enumerating that spec's
    // fields. Nested target depth in the value codec is a separate limit.
    auto deep=TfCreateRefPtr(new Enumeration);
    deep->CreateSpec(SdfPath("/"),SdfSpecTypePseudoRoot);
    for(const char* path:{"/A","/A/B","/A/B/C"}) deep->CreateSpec(SdfPath(path),SdfSpecTypePrim);
    field(*deep,"/",SdfChildrenKeys->PrimChildren,tokens({"A"}));
    field(*deep,"/A",SdfChildrenKeys->PrimChildren,tokens({"B"}));
    field(*deep,"/A/B",SdfChildrenKeys->PrimChildren,tokens({"C"}));
    deep->paths={SdfPath("/"),SdfPath("/A/B/C"),SdfPath("/A"),SdfPath("/A/B")};
    deep->forbiddenList=SdfPath("/A/B/C");
    limits={}; limits.scene.maxPathDepth=1;
    rejected([&] { (void)compileSceneSeed(*deep,{},limits); },wire::ErrorKind::ResourceLimit);
}
// This bounded oracle stores opaque field and sample rows independently, as
// both service implementations do. It has no Sdf projection or field synthesis.
// Only the commands exercised here are implemented; other commands fail.
struct PersistentRows {
    std::map<uint64_t,wire::Command> sections;
    std::map<uint64_t,std::vector<uint64_t>> children;
    std::map<std::pair<uint64_t,std::string>,std::string> fields;
    std::map<std::pair<uint64_t,double>,std::string> samples;
    void erase(uint64_t id) {
        const auto descendants=children[id];
        for (auto child:descendants) erase(child);
        auto& siblings=children[sections.at(id).parent];
        siblings.erase(std::remove(siblings.begin(),siblings.end(),id),siblings.end());
        children.erase(id); sections.erase(id);
        for (auto it=fields.begin();it!=fields.end();) it=it->first.first==id?fields.erase(it):std::next(it);
        for (auto it=samples.begin();it!=samples.end();) it=it->first.first==id?samples.erase(it):std::next(it);
    }
    void apply(std::string_view bytes) {
        const auto input=wire::decodeEnvelope(bytes);
        for (auto command:input.commands) {
            if (command.kind==wire::Command::Kind::Delete) { erase(command.section); continue; }
            if (command.kind==wire::Command::Kind::Move) {
                check(command.oldParent==command.parent && sections.at(command.section).parent==command.parent,
                      "persistence oracle move outside rename coverage");
                sections.at(command.section).name=command.name; continue;
            }
            check(command.kind==wire::Command::Kind::Create || command.kind==wire::Command::Kind::DiffChange,
                  "persistence oracle command outside bounded coverage");
            if (command.kind==wire::Command::Kind::Create) command.sectionOrder=0;
            check(sections.emplace(command.section,command).second,"persistence oracle requires fresh identities");
            auto& siblings=children[command.parent];
            const auto position=std::find_if(siblings.begin(),siblings.end(),[&](auto id) {
                return sections.at(id).sectionOrder>command.sectionOrder;
            });
            siblings.insert(position,command.section);
        }
        for (const auto& field:input.fields) fields[{field.section,field.name}]=std::string(field.value.bytes());
        for (const auto& sample:input.samples) samples[{sample.section,sample.time}]=std::string(sample.value.bytes());
    }
    void visit(uint64_t parent,wire::EnvelopeInput& snapshot) const {
        const auto found=children.find(parent); if (found==children.end()) return;
        uint32_t order=0;
        for (auto id:found->second) {
            auto command=sections.at(id); command.kind=wire::Command::Kind::DiffChange; command.sectionOrder=order++;
            snapshot.commands.push_back(command); visit(id,snapshot);
        }
    }
    std::string snapshot() const {
        wire::EnvelopeInput result; result.isDiff=true; visit(0,result);
        for (const auto& [key,value]:fields) result.fields.push_back({key.first,0,key.second,value});
        for (const auto& [key,value]:samples) result.samples.push_back({key.first,key.second,value});
        return wire::encodeEnvelope(result);
    }
};
void persistentSnapshots() {
    auto source=rootData();
    source->CreateSpec(SdfPath("/P"),SdfSpecTypePrim);
    field(*source,"/",SdfChildrenKeys->PrimChildren,tokens({"P"}));
    source->CreateSpec(SdfPath("/P.a"),SdfSpecTypeAttribute);
    field(*source,"/P",SdfChildrenKeys->PropertyChildren,tokens({"a"}));
    field(*source,"/P.a",SdfFieldKeys->TimeSamples,VtValue(SdfTimeSampleMap{{0,VtValue(10)}}));
    const auto seed=compileSceneSeed(*source);
    PersistentRows service; service.apply(seed.bytes);
    auto scene=seed.scene;
    uint64_t nextId=1000;
    const auto ids=std::make_shared<SectionIdAllocator>(4096,64,[&] { return nextId++; });
    const auto compare=[&] {
        const auto fresh=buildSceneCandidate({},service.snapshot());
        check(contents(*fresh->data())==contents(*scene->data()),"stored rows changed fresh snapshot contents");
    };
    compare();
    const auto commit=[&](auto author) {
        const auto log=test::capturedLog(scene,author);
        const auto intent=compileFieldEdits(scene,log,{},ids);
        scene=buildSceneCandidate(scene,intent->bytes());
        service.apply(intent->bytes()); compare();
    };
    commit([](auto& data) { data.SetTimeSample(SdfPath("/P.a"),1,VtValue(11)); });
    commit([](auto& data) { data.Set(SdfPath("/P.a"),SdfFieldKeys->TimeSamples,VtValue(SdfTimeSampleMap{{1,VtValue(22)},{2,VtValue(33)}})); });
    commit([](auto& data) { data.EraseTimeSample(SdfPath("/P.a"),1); });
    commit([](auto& data) { data.Erase(SdfPath("/P.a"),SdfFieldKeys->TimeSamples); });
    commit([](auto& data) { data.SetTimeSample(SdfPath("/P.a"),3,VtValue(SdfValueBlock{})); });
    commit([](auto& data) {
        data.CreateSpec(SdfPath("/P.b"),SdfSpecTypeAttribute);
        data.Set(SdfPath("/P"),SdfChildrenKeys->PropertyChildren,tokens({"a","b"}));
    });
    commit([](auto& data) {
        data.EraseSpec(SdfPath("/P.a"));
        data.Set(SdfPath("/P"),SdfChildrenKeys->PropertyChildren,tokens({"b"}));
    });
    commit([](auto& data) {
        data.MoveSpec(SdfPath("/P.b"),SdfPath("/P.c"));
        data.Set(SdfPath("/P"),SdfChildrenKeys->PropertyChildren,tokens({"c"}));
    });
    commit([](auto& data) {
        data.EraseSpec(SdfPath("/P.c")); data.Erase(SdfPath("/P"),SdfChildrenKeys->PropertyChildren);
    });
    commit([](auto& data) {
        data.CreateSpec(SdfPath("/Q"),SdfSpecTypePrim);
        data.Set(SdfPath("/"),SdfChildrenKeys->PrimChildren,tokens({"P","Q"}));
    });
    commit([](auto& data) {
        data.EraseSpec(SdfPath("/P"));
        data.Set(SdfPath("/"),SdfChildrenKeys->PrimChildren,tokens({"Q"}));
    });
    // A whole-map replacement removes observed keys only. Unseen peer keys
    // survive because the service has per-sample last-writer-wins storage.
    const auto log=test::capturedLog(seed.scene,[](auto& data) {
        data.Set(SdfPath("/P.a"),SdfFieldKeys->TimeSamples,VtValue(SdfTimeSampleMap{{4,VtValue(44)}}));
    });
    const auto intent=compileFieldEdits(seed.scene,log);
    wire::EnvelopeInput peer; peer.samples={{seed.scene->sectionForPath(SdfPath("/P.a")),9,{}}};
    const auto peerBytes=wire::encodeUsdEnvelope(peer,wire::DecodedValues{{},{VtValue(99)}});
    const auto merged=buildSceneCandidate(buildSceneCandidate(seed.scene,peerBytes),intent->bytes());
    check(merged->data()->Get(SdfPath("/P.a"),SdfFieldKeys->TimeSamples)==
          VtValue(SdfTimeSampleMap{{4,VtValue(44)},{9,VtValue(99)}}),"map replacement discarded unseen peer sample");
    FieldEditLimits editLimits; editLimits.envelope.content.maxRecords=1;
    rejected([&] { (void)compileFieldEdits(seed.scene,log,editLimits); },wire::ErrorKind::ResourceLimit);
    auto many=TfCreateRefPtr(new SdfData); many->CopyFrom(source);
    SdfTimeSampleMap manySamples;
    for (int i=0;i<16;++i) manySamples.emplace(i,VtValue(i));
    many->Set(SdfPath("/P.a"),SdfFieldKeys->TimeSamples,VtValue(manySamples));
    SceneSeedLimits seedLimits; seedLimits.envelope.content.maxRecords=10;
    rejected([&] { (void)compileSceneSeed(*many,{},seedLimits); },wire::ErrorKind::ResourceLimit);
    const auto infiniteLog=test::capturedLog(seed.scene,[](auto& data) {
        data.SetTimeSample(SdfPath("/P.a"),std::numeric_limits<double>::infinity(),VtValue(1));
    });
    rejected([&] { (void)compileFieldEdits(seed.scene,infiniteLog); },wire::ErrorKind::Unsupported);
    many->Set(SdfPath("/P.a"),SdfFieldKeys->TimeSamples,
              VtValue(SdfTimeSampleMap{{std::numeric_limits<double>::infinity(),VtValue(1)}}));
    rejected([&] { (void)compileSceneSeed(*many); },wire::ErrorKind::Unsupported);
    // The oracle also checks every supported child policy's authored order.
    const auto policies=allPolicies(); const auto policySeed=compileSceneSeed(*policies);
    PersistentRows policyService; policyService.apply(policySeed.bytes);
    check(contents(*buildSceneCandidate({},policyService.snapshot())->data())==contents(*policies),
          "stored seed section order changed a child policy");
}
// A seed's sections, fields and samples count together against the service's
// entry limit, exactly at it and one past it; child membership fields are
// section order, not entries.
void seedEntryLimits() {
    const SceneSeedLimits defaults;
    check(defaults.entries.maxEntries==99999 && defaults.entries.maxChildrenListEntries==100000,
          "seed entry limit defaults differ from the service's");
    auto source=rootData();
    source->CreateSpec(SdfPath("/P"),SdfSpecTypePrim);
    field(*source,"/",SdfChildrenKeys->PrimChildren,tokens({"P"}));
    source->CreateSpec(SdfPath("/P.a"),SdfSpecTypeAttribute);
    field(*source,"/P",SdfChildrenKeys->PropertyChildren,tokens({"a"}));
    field(*source,"/",TfToken("counter"),VtValue(7));
    field(*source,"/P.a",SdfFieldKeys->TimeSamples,VtValue(SdfTimeSampleMap{{0,VtValue(10)},{1,VtValue(11)}}));
    // Three sections, one field and two samples: six entries.
    SceneSeedLimits exact; exact.entries.maxEntries=6;
    verify(*source,compileSceneSeed(*source,{},exact));
    SceneSeedLimits over; over.entries.maxEntries=5;
    const auto before=contents(*source);
    bool refused=false;
    try { (void)compileSceneSeed(*source,{},over); }
    catch(const DeltaEntryLimitExceeded& error) {
        const auto& c=error.counts();
        refused=error.kind==wire::ErrorKind::ResourceLimit && error.reason()==DeltaEntryLimitExceeded::Reason::DeltaTooLarge &&
            c.structCommands==3 && c.setFields==1 && c.timeSamples==2 && c.limit==5 && c.actual==6;
    }
    check(refused && contents(*source)==before,"a seed one entry past the limit was not refused with its counts");
}

} // namespace
void sceneSeedTests() {
    TfErrorMark errors;
    persistentSnapshots(); stockUsdAndOwnership(); childPoliciesAndEnumeration(); typedPresenceAndBits(); invalidSources(); quotasAndAtomicity();
    seedEntryLimits();
    check(errors.IsClean(),"scene seed compiler leaked USD diagnostics");
}
