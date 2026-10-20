// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "FieldEditIntent.h"
#include "HostSceneProjection.h"
#include "UsdValueCodec.h"
#include <pxr/base/tf/notice.h>
#include <pxr/base/tf/weakBase.h>
#include <pxr/base/tf/weakPtr.h>
#include <pxr/usd/sdf/changeBlock.h>
#include <pxr/usd/sdf/fileFormat.h>
#include <pxr/usd/sdf/layer.h>
#include <pxr/usd/sdf/notice.h>
#include <pxr/usd/sdf/schema.h>
#include <pxr/usd/sdf/spec.h>
#include <pxr/usd/usd/attribute.h>
#include <pxr/usd/usd/notice.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usd/stage.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <set>
#include <stdexcept>

PXR_NAMESPACE_USING_DIRECTIVE
using namespace omni::usd::live;
using namespace omni::usd::live::wire;
namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(std::string("USD streaming publication: ")+message);
}
WireScenePtr scene(int revision, bool gone, bool moved, bool relationship) {
    EnvelopeInput input;
    DecodedValues values;
    auto create = [&](uint64_t id,uint64_t parent,const char* name,SdfSpecType type) {
        Command command{}; command.kind=Command::Kind::Create;
        command.section=id; command.parent=parent; command.name=name; command.sectionType=type;
        input.commands.push_back(std::move(command));
        if (type==SdfSpecTypePrim) {
            input.fields.push_back({id,0,"specifier",{}});
            values.fields.emplace_back(SdfSpecifierDef);
        }
    };
    create(2,1,"P",SdfSpecTypePrim);
    if (gone) create(revision==1 ? 3 : 7,1,"Gone",SdfSpecTypePrim);
    create(4,1,moved ? "Moved" : "Mover",SdfSpecTypePrim);
    create(relationship ? 6 : 5,2,"prop",relationship ? SdfSpecTypeRelationship : SdfSpecTypeAttribute);
    if (!relationship) {
        input.fields.push_back({5,0,"typeName",{}}); values.fields.emplace_back(TfToken("int"));
    }
    input.fields.push_back({1,0,"revision",{}}); values.fields.emplace_back(revision);
    return buildSceneCandidate({},encodeUsdEnvelope(input,values));
}
// One section of a complete scene: its identity, parent, name, type and fields.
struct Section {
    uint64_t id, parent;
    std::string name;
    SdfSpecType type;
    std::vector<std::pair<std::string,VtValue>> fields;
};
Section prim(uint64_t id,uint64_t parent,const char* name) {
    return {id,parent,name,SdfSpecTypePrim,{{"specifier",VtValue(SdfSpecifierDef)}}};
}
Section attribute(uint64_t id,uint64_t parent,const char* name,double value) {
    return {id,parent,name,SdfSpecTypeAttribute,{{"typeName",VtValue(TfToken("double"))},{"default",VtValue(value)}}};
}
// A complete scene of `sections`, parents first, plus a root revision field
// and any other `root` fields.
WireScenePtr build(const std::vector<Section>& sections,int revision,
                   const std::vector<std::pair<std::string,VtValue>>& root={}) {
    EnvelopeInput input;
    DecodedValues values;
    for (const auto& section:sections) {
        Command command{}; command.kind=Command::Kind::Create;
        command.section=section.id; command.parent=section.parent; command.name=section.name;
        command.sectionType=uint8_t(section.type);
        input.commands.push_back(std::move(command));
        for (const auto& [name,value]:section.fields) {
            input.fields.push_back({section.id,0,name,{}}); values.fields.push_back(value);
        }
    }
    input.fields.push_back({1,0,"revision",{}}); values.fields.emplace_back(revision);
    for (const auto& [name,value]:root) { input.fields.push_back({1,0,name,{}}); values.fields.push_back(value); }
    return buildSceneCandidate({},encodeUsdEnvelope(input,values));
}
class PublicationFormat final : public SdfFileFormat {
    const SdfAbstractDataRefPtr initial_;
public:
    explicit PublicationFormat(SdfAbstractDataRefPtr initial)
        : SdfFileFormat(TfToken("stream_publication_probe"),TfToken("1"),TfToken("usd"),
                        "stream_publication_probe"), initial_(std::move(initial)) {}
    SdfAbstractDataRefPtr InitData(const FileFormatArguments&) const override { return initial_; }
    bool CanRead(const std::string&) const override { return false; }
    bool Read(SdfLayer*,const std::string&,bool) const override { return false; }
    static void publish(SdfLayer* layer,SdfAbstractDataRefPtr data) { _SetLayerData(layer,data); }
    static const SdfAbstractData* backing(const SdfLayer& layer) { return _GetLayerData(layer).operator->(); }
protected:
    // Anonymous creation asks for detached initialization. This test expressly
    // installs streaming data to exercise the publication path, not file I/O.
    SdfAbstractDataRefPtr _InitDetachedData(const FileFormatArguments& args) const override { return InitData(args); }
};
// A live-style layer: the format's streaming projection of `initial`.
struct ProbeLayer {
    std::shared_ptr<HostFailureLatch> fault=std::make_shared<HostFailureLatch>();
    TfRefPtr<HostSceneProjection> projection;
    SdfFileFormatRefPtr format;
    SdfLayerRefPtr layer;
    explicit ProbeLayer(const WireScenePtr& initial)
        : projection(TfCreateRefPtr(new HostSceneProjection(initial,fault))),
          format(TfCreateRefPtr(new PublicationFormat(projection))),
          layer(SdfLayer::CreateAnonymous("incremental.stream_publication_probe",format)) {
        require(layer && PublicationFormat::backing(*layer)==projection.operator->(),"initial data was not adopted");
    }
    // A later publication of `next`, as BoundLayer::install makes it: none
    // where the projection keeps its baseline, else notices and the reseat
    // in one change block, as BoundLayer::publishInPlace sends them.
    WireSceneDifference publish(const WireScenePtr& next) {
        return publish(next,diffProjectedScenes(*projection->baseline(),*next));
    }
    WireSceneDifference publish(const WireScenePtr& next,const WireSceneDifference& difference) {
        if (projection->keepsBaseline(next,difference)) return difference;
        auto prepared=projection->prepareReseat(next,difference);
        SdfChangeBlock block;
        publishSceneDifference(*layer->GetStateDelegate(),*projection,std::move(prepared),difference);
        return difference;
    }
    void unchanged(const char* message) const {
        require(projection->pendingLocalEditCount()==0 && fault->result().load()==OMNI_USD_LIVE_OK &&
                PublicationFormat::backing(*layer)==projection.operator->(),message);
    }
};
struct Observer final : TfWeakBase {
    size_t replacements=0, changes=0, serial=0;
    bool replacementFlag=true, serialIncreasing=true;
    SdfChangeList last;
    TfNotice::Key replaceKey, changeKey;
    explicit Observer(const SdfLayerHandle& layer) {
        replaceKey=TfNotice::Register(TfCreateWeakPtr(this),&Observer::replaced,layer);
        changeKey=TfNotice::Register(TfCreateWeakPtr(this),&Observer::changed,layer);
    }
    ~Observer() { TfNotice::Revoke(replaceKey); TfNotice::Revoke(changeKey); }
    void replaced(const SdfNotice::LayerDidReplaceContent&,const SdfLayerHandle&) { ++replacements; }
    void changed(const SdfNotice::LayersDidChangeSentPerLayer& notice,const SdfLayerHandle& layer) {
        ++changes; serialIncreasing &= notice.GetSerialNumber()>serial; serial=notice.GetSerialNumber();
        bool found=false;
        for (const auto& item:notice.GetChangeListVec()) if (item.first==layer) {
            last=item.second;
            for (const auto& entry:item.second.GetEntryList())
                found |= entry.first==SdfPath::AbsoluteRootPath() && entry.second.flags.didReplaceContent;
        }
        replacementFlag &= found;
    }
    const SdfChangeList::Entry* entry(const char* path) const {
        for (const auto& item:last.GetEntryList()) if (item.first==SdfPath(path)) return &item.second;
        return nullptr;
    }
    std::set<SdfPath> paths() const {
        std::set<SdfPath> result;
        for (const auto& item:last.GetEntryList()) result.insert(item.first);
        return result;
    }
};
// UsdNotice::ObjectsChanged of one stage, accumulated until taken.
struct StageObserver final : TfWeakBase {
    std::set<SdfPath> resynced, infoOnly;
    size_t notices=0;
    TfNotice::Key key;
    explicit StageObserver(const UsdStageRefPtr& stage) {
        key=TfNotice::Register(TfCreateWeakPtr(this),&StageObserver::changed,UsdStageWeakPtr(stage));
    }
    ~StageObserver() { TfNotice::Revoke(key); }
    void changed(const UsdNotice::ObjectsChanged& notice,const UsdStageWeakPtr&) {
        ++notices;
        for (const auto& path:notice.GetResyncedPaths()) resynced.insert(path);
        for (const auto& path:notice.GetChangedInfoOnlyPaths()) infoOnly.insert(path);
    }
    void clear() { resynced.clear(); infoOnly.clear(); notices=0; }
};
int revision(const SdfAbstractData& data) {
    const auto value=data.Get(SdfPath::AbsoluteRootPath(),TfToken("revision"));
    require(value.IsHolding<int>(),"snapshot revision missing"); return value.UncheckedGet<int>();
}
std::vector<Section> stageScene(double value) {
    return {prim(2,1,"World"),attribute(3,2,"a",value),attribute(4,2,"b",0.0),prim(5,2,"Child"),prim(6,1,"Other")};
}

// The first installation and any reset replace the whole content, as the
// pinned USD Sdf replacement path does for streaming data. It does not
// qualify concurrent unguarded layer access or a live file-format subscription.
void replacementPublication() {
    const auto first=scene(1,true,false,false);
    const auto second=scene(2,false,true,true);
    const auto third=scene(3,true,true,true);
    const auto oldFault=std::make_shared<HostFailureLatch>(),newFault=std::make_shared<HostFailureLatch>(),
        thirdFault=std::make_shared<HostFailureLatch>();
    const auto oldData=TfCreateRefPtr(new HostSceneProjection(first,oldFault));
    const SdfFileFormatRefPtr format=TfCreateRefPtr(new PublicationFormat(oldData));
    const auto layer=SdfLayer::CreateAnonymous("publication.stream_publication_probe",format);
    require(layer && PublicationFormat::backing(*layer)==oldData.operator->(),"initial data was not adopted");
    require(layer->StreamsData() && !layer->IsDetached(),"streaming wrapper flags lost");
    const SdfSpec retainedGone=layer->GetObjectAtPath(SdfPath("/Gone")).GetSpec();
    const SdfSpec retainedMover=layer->GetObjectAtPath(SdfPath("/Mover")).GetSpec();
    const SdfSpec retainedProperty=layer->GetObjectAtPath(SdfPath("/P.prop")).GetSpec();
    Observer observer{SdfLayerHandle(layer)};

    const auto newData=TfCreateRefPtr(new HostSceneProjection(second,newFault));
    // Adoption, not a setter fallback: neither backing recorded or faulted a mutation.
    PublicationFormat::publish(layer.operator->(),newData);
    require(oldData->pendingLocalEditCount()==0 && newData->pendingLocalEditCount()==0 &&
            oldFault->result().load()==OMNI_USD_LIVE_OK && newFault->result().load()==OMNI_USD_LIVE_OK,
            "replacement wrote through a backing instead of adopting it");
    require(PublicationFormat::backing(*layer)==newData.operator->(),"same-type replacement did not adopt owner");
    require(observer.replacements==1 && observer.changes==1 && observer.replacementFlag,
            "replacement did not emit ordinary USD replacement/change notices");
    require(layer->GetField(SdfPath::AbsoluteRootPath(),TfToken("revision"))==VtValue(2),"layer did not expose new snapshot");
    require(revision(*oldData)==1 && oldData->HasSpec(SdfPath("/Gone")) &&
            oldData->HasSpec(SdfPath("/Mover")) && oldData->GetSpecType(SdfPath("/P.prop"))==SdfSpecTypeAttribute,
            "replacement mutated retained old data");
    require(retainedGone.IsDormant() && retainedMover.IsDormant(),"removed namespace handles remain active");
    require(retainedMover.GetPath()==SdfPath("/Mover") && layer->HasSpec(SdfPath("/Moved")),
            "coarse replacement incorrectly promised identity-preserving move");
    require(!retainedProperty.IsDormant() && retainedProperty.GetSpecType()==SdfSpecTypeRelationship,
            "same-path generic handle did not observe current spec type");

    const auto thirdData=TfCreateRefPtr(new HostSceneProjection(third,thirdFault));
    {
        SdfChangeBlock block;
        PublicationFormat::publish(layer.operator->(),thirdData);
        require(observer.replacements==1 && observer.changes==1,"notices escaped enclosing change block");
    }
    require(thirdData->pendingLocalEditCount()==0 && thirdFault->result().load()==OMNI_USD_LIVE_OK,
            "replacement wrote through a backing instead of adopting it");
    require(observer.replacements==2 && observer.changes==2 && observer.replacementFlag && observer.serialIncreasing,
            "deferred replacement notification or global serial is incorrect");
    // Sdf handles identify paths, not service section IDs: Gone's new section 7
    // reactivates the generic handle that section 3 was associated with.
    require(!retainedGone.IsDormant() && retainedGone.GetPath()==SdfPath("/Gone") &&
            third->sectionForPath(SdfPath("/Gone"))!=first->sectionForPath(SdfPath("/Gone")),
            "recreated-path handle semantics differ from pinned USD");
    require(PublicationFormat::backing(*layer)==thirdData.operator->() && revision(*newData)==2 && revision(*oldData)==1,
            "later publication corrupted earlier retained snapshots");
}

// An incremental publication keeps the backing, records nothing and sends one
// change notice whose entries are exactly the visible difference.
void incrementalPublication() {
    const auto first=build({prim(2,1,"P"),attribute(5,2,"a",1.0),prim(3,1,"Gone"),prim(4,1,"Mover"),
                            prim(8,1,"Q"),prim(9,8,"C")},1);
    const auto second=build({prim(2,1,"P"),attribute(5,2,"a",2.0),prim(4,1,"Moved"),prim(8,1,"Q"),prim(9,2,"C"),
                             prim(10,1,"New"),attribute(11,10,"b",0.0)},2);
    require(diffProjectedScenes(*first,*build({prim(2,1,"P"),attribute(5,2,"a",1.0),prim(3,1,"Gone"),
                prim(4,1,"Mover"),prim(8,1,"Q"),prim(9,8,"C")},1)).empty(),
            "separately built equal scenes differ");
    ProbeLayer probe(first);
    const auto& layer=probe.layer;
    const SdfSpec mover=layer->GetObjectAtPath(SdfPath("/Mover")).GetSpec();
    const SdfSpec gone=layer->GetObjectAtPath(SdfPath("/Gone")).GetSpec();
    Observer observer{SdfLayerHandle(layer)};
    const auto difference=probe.publish(second);
    probe.unchanged("incremental publication recorded, faulted or replaced the backing");
    require(probe.projection->baseline()==second && revision(*probe.projection)==2 &&
            layer->GetField(SdfPath("/P.a"),SdfFieldKeys->Default)==VtValue(2.0) && !layer->HasSpec(SdfPath("/Gone")) &&
            layer->HasSpec(SdfPath("/Moved")) && layer->HasSpec(SdfPath("/P/C")) && layer->HasSpec(SdfPath("/New.b")),
            "incremental publication did not install the new scene");
    require(difference.removed==std::vector<SdfPath>{SdfPath("/Gone")} && difference.moved.size()==2 &&
            difference.created.size()==2,"scene difference is not the visible change");
    require(observer.replacements==0 && observer.changes==1,"incremental publication did not send one change notice");
    require(observer.paths()==std::set<SdfPath>{SdfPath("/"),SdfPath("/P.a"),SdfPath("/Gone"),SdfPath("/Moved"),
                SdfPath("/P/C"),SdfPath("/New"),SdfPath("/New.b"),SdfPath("/Q/C")},
            "incremental notice paths are not the visible difference");
    const auto* root=observer.entry("/");
    require(root && !root->flags.didReplaceContent && root->HasInfoChange(TfToken("revision")),
            "root entry is a replacement or misses its field change");
    const auto* field=observer.entry("/P.a");
    require(field && field->infoChanged.size()==1 && field->HasInfoChange(SdfFieldKeys->Default) &&
            field->FindInfoChange(SdfFieldKeys->Default)->second.first==VtValue(1.0) &&
            field->FindInfoChange(SdfFieldKeys->Default)->second.second==VtValue(2.0) &&
            !field->flags.didAddProperty && !field->flags.didRemoveProperty,
            "a field edit is not an info-only change with its old and new value");
    const auto* removed=observer.entry("/Gone");
    require(removed && removed->flags.didRemoveNonInertPrim && removed->infoChanged.empty(),"a delete is not a removal");
    const auto* renamed=observer.entry("/Moved");
    require(renamed && renamed->flags.didRename && renamed->oldPath==SdfPath("/Mover"),"a rename is not a move");
    const auto* reparented=observer.entry("/P/C");
    require(reparented && reparented->flags.didAddNonInertPrim && reparented->oldPath==SdfPath("/Q/C"),
            "a reparent is not a move");
    const auto* created=observer.entry("/New");
    const auto* property=observer.entry("/New.b");
    require(created && created->flags.didAddNonInertPrim && property && property->flags.didAddProperty,
            "a create is not a significant addition");
    // A moved spec's handle follows its identity; a removed one goes dormant.
    require(!mover.IsDormant() && mover.GetPath()==SdfPath("/Moved") && gone.IsDormant(),
            "incremental move or removal left the wrong handle state");

    // A publication whose visible content did not change sends nothing.
    require(diffProjectedScenes(*second,*second).empty() &&
            diffProjectedScenes(*second,*build({prim(2,1,"P"),attribute(5,2,"a",2.0),prim(4,1,"Moved"),prim(8,1,"Q"),
                prim(9,2,"C"),prim(10,1,"New"),attribute(11,10,"b",0.0)},2)).empty(),
            "an unchanged visible scene has a difference");

    // A host edit after the publication records as usual.
    layer->SetField(SdfPath("/P.a"),SdfFieldKeys->Default,VtValue(5.0));
    require(probe.projection->pendingLocalEditCount()==1,"announcement scope outlived the publication");
}

// Retained handles to a moved subtree's root and descendants follow the move,
// and handles to a removed subtree go dormant, as for a host's own edits.
void subtreeHandles() {
    const auto child=[](uint64_t parent) {
        return std::vector<Section>{prim(3,parent,"Child"),attribute(4,3,"x",1.0)};
    };
    auto scene=[&](const char* root,uint64_t parent,std::vector<Section> extra) {
        std::vector<Section> sections{prim(10,1,"P"),prim(2,parent,root)};
        for (auto& section:child(2)) sections.push_back(section);
        for (auto& section:extra) sections.push_back(std::move(section));
        return build(sections,1);
    };
    ProbeLayer probe(scene("A",1,{}));
    const auto& layer=probe.layer;
    const SdfSpec root=layer->GetObjectAtPath(SdfPath("/A")).GetSpec();
    const SdfSpec nested=layer->GetObjectAtPath(SdfPath("/A/Child")).GetSpec();
    const SdfSpec property=layer->GetObjectAtPath(SdfPath("/A/Child.x")).GetSpec();
    const auto at=[&](const char* a,const char* b,const char* c) {
        return !root.IsDormant() && !nested.IsDormant() && !property.IsDormant() && root.GetPath()==SdfPath(a) &&
            nested.GetPath()==SdfPath(b) && property.GetPath()==SdfPath(c);
    };
    probe.publish(scene("B",1,{}));
    require(at("/B","/B/Child","/B/Child.x"),"a rename left a retained descendant handle behind");
    probe.publish(scene("B",10,{}));
    require(at("/P/B","/P/B/Child","/P/B/Child.x"),"a reparent left a retained descendant handle behind");
    // Unrelated specs at the old paths never revive the moved handles.
    probe.publish(scene("B",10,{prim(20,1,"A"),prim(21,20,"Child"),attribute(22,21,"x",9.0)}));
    require(at("/P/B","/P/B/Child","/P/B/Child.x") &&
            property.GetLayer()->GetField(property.GetPath(),SdfFieldKeys->Default)==VtValue(1.0),
            "a moved handle revived against unrelated content");
    probe.publish(build({prim(10,1,"P"),prim(20,1,"A"),prim(21,20,"Child"),attribute(22,21,"x",9.0)},1));
    require(root.IsDormant() && nested.IsDormant() && property.IsDormant(),"a removed subtree's handles stay active");
    probe.unchanged("subtree publications recorded, faulted or replaced the backing");
}

// A path change that a single move cannot state is a removal and a creation.
void namespaceFallbacks() {
    using Paths=std::vector<SdfPath>;
    const auto first=build({prim(2,1,"A"),prim(3,1,"B"),prim(8,1,"Q"),prim(9,8,"C"),prim(10,1,"P"),
                            attribute(11,10,"x",1.0)},1);
    // A and B swap names; Q is renamed while its child C moves to P; P.x
    // becomes a relationship under the same identity.
    const auto second=build({prim(2,1,"B"),prim(3,1,"A"),prim(8,1,"R"),prim(10,1,"P"),prim(9,10,"C"),
                             {11,10,"x",SdfSpecTypeRelationship,{}}},1);
    const auto difference=diffProjectedScenes(*first,*second);
    require(difference.removed==Paths{SdfPath("/Q/C"),SdfPath("/P.x"),SdfPath("/B"),SdfPath("/A")},
            "fallback removals are not the old paths, deepest first");
    require(difference.moved==std::vector<std::pair<SdfPath,SdfPath>>{{SdfPath("/Q"),SdfPath("/R")}},
            "only the unambiguous rename is a move");
    std::vector<SdfPath> created;
    for (const auto& item:difference.created) created.push_back(item.first);
    require(std::set<SdfPath>(created.begin(),created.end())==std::set<SdfPath>{SdfPath("/A"),SdfPath("/B"),
                SdfPath("/P/C"),SdfPath("/P.x")} && std::is_sorted(created.begin(),created.end()),
            "fallback creations are not the new paths in namespace order");
    ProbeLayer probe(first);
    Observer observer{SdfLayerHandle(probe.layer)};
    probe.publish(second);
    probe.unchanged("fallback publication recorded, faulted or replaced the backing");
    require(observer.changes==1 && observer.replacements==0 && probe.layer->HasSpec(SdfPath("/P/C")) &&
            probe.layer->GetSpecType(SdfPath("/P.x"))==SdfSpecTypeRelationship && probe.layer->HasSpec(SdfPath("/R")) &&
            !probe.layer->HasSpec(SdfPath("/Q")),
            "fallback publication did not install the new scene");
    // Sdf keeps a removal and a later addition of one path as separate entries.
    bool removedA=false, addedA=false;
    for (const auto& [path,entry]:observer.last.GetEntryList()) if (path==SdfPath("/A")) {
        removedA |= entry.flags.didRemoveNonInertPrim;
        addedA |= entry.flags.didAddNonInertPrim;
    }
    require(removedA && addedA,"a swapped name is not a removal and a creation");
}

// Floating-point content compares by representation: a visible change of only
// the sign of a zero, in a value or a time-sample key, is a field change, so
// the host content stays exactly its baseline's projection and a later host
// edit compiles against it. A set of a time-sample key keeps an equivalent
// key's bits, so a writer whose own pending sample replays onto a peer's
// equivalent key sees that key change sign once the peer's erase arrives.
void signedZeroPublication() {
    const SdfPath samplesPath("/P.value");
    const auto samples=[](std::initializer_list<std::pair<double,int>> items) {
        SdfTimeSampleMap map;
        for (const auto& [time,value]:items) map[time]=VtValue(value);
        return build({prim(2,1,"P"),{3,2,"value",SdfSpecTypeAttribute,
                      {{"typeName",VtValue(TfToken("int"))},{"timeSamples",VtValue(map)}}}},1);
    };
    const auto keySign=[&](const HostSceneProjection& data) {
        const auto value=data.Get(samplesPath,SdfFieldKeys->TimeSamples);
        require(value.IsHolding<SdfTimeSampleMap>() && value.UncheckedGet<SdfTimeSampleMap>().size()==1,
                "time samples missing");
        return std::signbit(value.UncheckedGet<SdfTimeSampleMap>().begin()->first);
    };
    require(diffProjectedScenes(*samples({{0.0,3}}),*samples({{-0.0,3}})).fields.size()==1 &&
            diffProjectedScenes(*samples({{-0.0,3}}),*samples({{0.0,3}})).fields.size()==1,
            "a time-sample key's sign change is no difference");
    require(diffProjectedScenes(*build({prim(2,1,"P"),attribute(3,2,"a",0.0)},1),
                                *build({prim(2,1,"P"),attribute(3,2,"a",-0.0)},1)).fields.size()==1,
            "a value's sign change of zero is no difference");
    require(diffProjectedScenes(*samples({{-0.0,3}}),*samples({{-0.0,3}})).empty() &&
            diffProjectedScenes(*build({prim(2,1,"P"),attribute(3,2,"a",std::nan(""))},1),
                                *build({prim(2,1,"P"),attribute(3,2,"a",std::nan(""))},1)).empty(),
            "an unchanged signed zero or NaN is a difference");

    ProbeLayer probe(samples({{0.0,3},{2.0,3}}));
    probe.publish(samples({{0.0,3}}));
    require(!keySign(*probe.projection),"positive zero key not installed");
    Observer observer{SdfLayerHandle(probe.layer)};
    const auto difference=probe.publish(samples({{-0.0,3}}));
    probe.unchanged("sign publication recorded, faulted or replaced the backing");
    require(difference.fields.size()==1 && observer.changes==1 && observer.entry("/P.value"),
            "a key's sign change sent no notice");
    require(keySign(*probe.projection) && matchesWireProjection(probe.projection->baseline(),*probe.projection),
            "host content differs from its baseline after a key's sign change");

    // A later host edit of the same attribute compiles against that baseline.
    probe.layer->SetTimeSample(samplesPath,1.0,VtValue(3));
    const auto edits=probe.projection->pendingLocalEdits();
    require(edits.operations->size()==1,"host sample edit not recorded");
    const auto log=std::make_shared<const SdfOperationLog>(SdfOperationLog{edits.data,edits.operations});
    try {
        require(!compileFieldEdits(edits.base,log)->empty(),"host sample edit compiled empty");
    } catch (const DecodeError& error) {
        throw std::runtime_error(std::string("USD streaming publication: host sample edit rejected: ")+error.what());
    }
}

// A value that VtValue equality calls unchanged but whose projection changes,
// here a sublayer offset within USD's comparison tolerance: the difference
// compares offsets by representation, so the change reaches the host content.
// A publication whose difference is empty, as one the comparison misses would
// be, keeps the baseline and content without a notice, and a later nonempty
// publication differs from that kept baseline. Either way the content equals
// its baseline and a later host edit compiles against it.
void inexactEqualityPublication() {
    const auto offsets=[](double offset) {
        return VtValue(std::vector<SdfLayerOffset>{SdfLayerOffset(offset)});
    };
    const auto nearOffset=std::nextafter(1.0,2.0);
    require(SdfLayerOffset(1.0)==SdfLayerOffset(nearOffset),"USD layer offset equality is exact");
    const auto scene=[&](double offset,double value) {
        return build({prim(2,1,"P"),attribute(3,2,"a",value)},1,{{"subLayerOffsets",offsets(offset)}});
    };
    const auto difference=diffProjectedScenes(*scene(1.0,1.0),*scene(nearOffset,1.0));
    require(difference.fields.size()==1 && difference.fields[0].field==SdfFieldKeys->SubLayerOffsets,
            "an offset's representation change is no difference");
    // 0: the offset alone changes; 1: a field changes beside it; 2: the
    // difference is forced empty, then a field changes.
    for (const int mode:{0,1,2}) {
        const auto first=scene(1.0,1.0);
        ProbeLayer probe(first);
        Observer observer{SdfLayerHandle(probe.layer)};
        const auto next=scene(nearOffset,mode==1 ? 2.0 : 1.0);
        if (mode==2) {
            probe.publish(next,WireSceneDifference{});
            probe.unchanged("kept baseline recorded, faulted or replaced the backing");
            require(observer.changes==0 && probe.projection->baseline()==first &&
                    matchesWireProjection(first,*probe.projection),
                    "an empty difference did not keep the baseline and its content");
            const auto later=scene(nearOffset,3.0);
            const auto kept=probe.publish(later);
            require(kept.fields.size()==2 && observer.changes==1,
                    "a later publication did not differ from the kept baseline");
            require(probe.projection->baseline()==later && matchesWireProjection(later,*probe.projection),
                    "host content differs from its baseline after a publication from the kept baseline");
        } else {
            probe.publish(next);
            require(observer.changes==1 && probe.projection->baseline()==next &&
                    matchesWireProjection(next,*probe.projection),
                    "host content differs from its baseline after an offset publication");
        }
        probe.unchanged("offset publication recorded, faulted or replaced the backing");
        require(observer.replacements==0,"offset publication replaced the content");
        probe.layer->SetField(SdfPath("/P.a"),SdfFieldKeys->Default,VtValue(5.0));
        const auto edits=probe.projection->pendingLocalEdits();
        const auto log=std::make_shared<const SdfOperationLog>(SdfOperationLog{edits.data,edits.operations});
        try {
            require(!compileFieldEdits(edits.base,log)->empty(),"host field edit compiled empty");
        } catch (const DecodeError& error) {
            throw std::runtime_error(std::string("USD streaming publication: host field edit rejected after offset "
                                                 "publication mode ")+std::to_string(mode)+": "+error.what());
        }
    }
    // A new source format with no field difference, here from two empty
    // base-zero reset snapshots, reseats the projection, which then reports
    // the new format, without a notice.
    const auto reset=[](const char* format) {
        EnvelopeInput input;
        input.isDiff=true; input.sourceFormat=format;
        return buildSceneCandidate({},encodeUsdEnvelope(input,DecodedValues{}));
    };
    ProbeLayer probe(reset("usda"));
    Observer observer{SdfLayerHandle(probe.layer)};
    const auto formatted=reset("usdc");
    require(probe.publish(formatted).empty() && probe.projection->baseline()==formatted && observer.changes==0 &&
            matchesWireProjection(formatted,*probe.projection),"a new source format was not adopted silently");
    probe.unchanged("source format publication recorded, faulted or replaced the backing");
}

// A UsdStage of the layer recomposes only the changed paths.
void stagePublication() {
    ProbeLayer probe(build(stageScene(1.0),1));
    // The probe format has no plugin registration, so Pcp cannot open it by
    // identifier as a sublayer; the stage's root layer stack holds it as its
    // root instead. The scene-reference lane composes a live layer as a sublayer.
    const auto stage=UsdStage::Open(probe.layer);
    require(stage && bool(stage->GetPrimAtPath(SdfPath("/World/Child"))),"stage did not compose the probe layer");
    StageObserver observer(stage);

    // A one-attribute peer edit is an info-only change of that attribute.
    probe.publish(build(stageScene(2.0),1));
    require(observer.notices==1 && observer.resynced.empty() &&
            observer.infoOnly==std::set<SdfPath>{SdfPath("/World.a")},"an attribute edit resynced or missed its path");
    double value=0;
    require(stage->GetAttributeAtPath(SdfPath("/World.a")).Get(&value) && value==2.0,"stage reads a stale value");

    // A create resyncs only the new prim.
    observer.clear();
    auto created=stageScene(2.0);
    created.push_back(prim(7,2,"Added"));
    probe.publish(build(created,1));
    require(observer.notices==1 && observer.resynced==std::set<SdfPath>{SdfPath("/World/Added")} &&
            observer.infoOnly.empty(),"a create resynced more than its path");
    require(bool(stage->GetPrimAtPath(SdfPath("/World/Added"))),"stage misses the created prim");

    // A rename resyncs the old and the new path only.
    observer.clear();
    auto renamed=created;
    renamed[4]=prim(6,1,"Renamed");
    probe.publish(build(renamed,1));
    require(observer.resynced==std::set<SdfPath>{SdfPath("/Other"),SdfPath("/Renamed")} && observer.infoOnly.empty(),
            "a rename resynced more than its paths");

    // A whole-content replacement, as a first installation sends, resyncs the
    // whole layer.
    observer.clear();
    PublicationFormat::publish(probe.layer.operator->(),
        TfCreateRefPtr(new HostSceneProjection(build(renamed,2),std::make_shared<HostFailureLatch>())));
    require(observer.resynced==std::set<SdfPath>{SdfPath::AbsoluteRootPath()},"a replacement did not resync the root");
}
} // namespace

void usdPublicationTests() {
    replacementPublication();
    incrementalPublication();
    subtreeHandles();
    namespaceFallbacks();
    signedZeroPublication();
    inexactEqualityPublication();
    stagePublication();
}

// Host time of one peer edit of one attribute, published to a UsdStage of the
// layer, by whole-content replacement and by per-path notices.
// A measurement only; no test runs it.
int measureUsdPublicationNotices() {
    using Clock=std::chrono::steady_clock;
    for (const size_t prims:{size_t(500),size_t(2500),size_t(10000)}) {
        std::vector<Section> sections;
        for (size_t i=0; i<prims; ++i) {
            const auto id=2+2*i;
            const auto name="P"+std::to_string(i);
            sections.push_back(prim(id,1,name.c_str()));
            sections.push_back(attribute(id+1,id,"a",0.0));
        }
        auto edited=sections;
        const auto base=build(sections,1);
        constexpr int rounds=20;
        double coarse=0, fine=0;
        for (int mode=0; mode<2; ++mode) {
            ProbeLayer probe(base);
            const auto stage=UsdStage::Open(probe.layer);
            for (const auto prim:stage->Traverse()) (void)prim.GetAttribute(TfToken("a"));
            Clock::duration total{};
            for (int round=1; round<=rounds; ++round) {
                edited[1].fields[1].second=VtValue(double(round+mode*rounds));
                const auto next=build(edited,1);
                const auto started=Clock::now();
                if (mode==0) {
                    // The replacement path: a fresh projection installed whole.
                    const auto data=TfCreateRefPtr(new HostSceneProjection(next,probe.fault));
                    SdfChangeBlock block;
                    PublicationFormat::publish(probe.layer.operator->(),data);
                } else probe.publish(next);
                total+=Clock::now()-started;
            }
            (mode==0 ? coarse : fine)=std::chrono::duration<double,std::milli>(total).count()/rounds;
        }
        std::cout << "{\"specs\":" << 2*prims+1 << ",\"coarse_ms\":" << coarse << ",\"fine_ms\":" << fine << "}" << std::endl;
    }
    return 0;
}
