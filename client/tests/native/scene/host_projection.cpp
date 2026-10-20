// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "HostSceneProjection.h"
#include "HostSceneStorage.h"
#include "HostLayerState.h"
#include "UsdValueCodec.h"
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/tf/errorMark.h>
#include <pxr/base/tf/notice.h>
#include <pxr/base/tf/weakBase.h>
#include <pxr/base/tf/weakPtr.h>
#include <pxr/base/vt/array.h>
#include <pxr/base/vt/dictionary.h>
#include <pxr/usd/sdf/attributeSpec.h>
#include <pxr/usd/sdf/changeBlock.h>
#include <pxr/usd/sdf/fileFormat.h>
#include <pxr/usd/sdf/layer.h>
#include <pxr/usd/sdf/listOp.h>
#include <pxr/usd/sdf/namespaceEdit.h>
#include <pxr/usd/sdf/notice.h>
#include <pxr/usd/sdf/payload.h>
#include <pxr/usd/sdf/primSpec.h>
#include <pxr/usd/sdf/reference.h>
#include <pxr/usd/sdf/schema.h>
#include <iostream>
#include <limits>
#include <string>
#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <map>
#include <stdexcept>

PXR_NAMESPACE_USING_DIRECTIVE
using namespace omni::usd::live;
using namespace omni::usd::live::wire;
namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(std::string("host projection: ") + message);
}
WireScenePtr baseline() {
    EnvelopeInput input;
    DecodedValues values;
    auto create = [&](uint64_t id, uint64_t parent, const char* name, SdfSpecType type) {
        Command c{}; c.kind = Command::Kind::Create; c.section = id; c.parent = parent;
        c.name = name; c.sectionType = type; input.commands.push_back(c);
    };
    auto field = [&](uint64_t id, const char* name, VtValue value) {
        input.fields.push_back({id, 0, name, {}}); values.fields.push_back(std::move(value));
    };
    create(2, 1, "P", SdfSpecTypePrim);
    create(3, 2, "Child", SdfSpecTypePrim);
    create(4, 3, "value", SdfSpecTypeAttribute);
    field(2, "specifier", VtValue(SdfSpecifierDef));
    field(3, "specifier", VtValue(SdfSpecifierDef));
    field(4, "typeName", VtValue(TfToken("double")));
    field(4, "default", VtValue(7.0));
    return buildSceneCandidate({}, encodeUsdEnvelope(input, values));
}
using Records = std::map<SdfPath, std::pair<SdfSpecType, std::map<TfToken, VtValue>>>;
Records inspect(const SdfAbstractData& source) {
    struct Visitor final : SdfAbstractDataSpecVisitor {
        const SdfAbstractData* expected;
        Records records;
        unsigned done = 0;
        explicit Visitor(const SdfAbstractData& data) : expected(&data) {}
        bool VisitSpec(const SdfAbstractData& data, const SdfPath& path) override {
            require(&data == expected, "visitor exposed a different backing object");
            auto& record = records[path]; record.first = data.GetSpecType(path);
            for (const auto& field : data.List(path)) record.second[field] = data.Get(path, field);
            return true;
        }
        void Done(const SdfAbstractData& data) override {
            require(&data == expected, "Done exposed a different backing object"); ++done;
        }
    } visitor(source);
    source.VisitSpecs(&visitor);
    require(visitor.done == 1, "visitor completion differs");
    return visitor.records;
}
class Format final : public SdfFileFormat {
    const WireScenePtr empty_;
    const std::shared_ptr<HostFailureLatch> fault_;
public:
    explicit Format(std::shared_ptr<HostFailureLatch> fault)
        : SdfFileFormat(TfToken("host_projection_probe"), TfToken("1"), TfToken("usd"), "host_projection_probe"),
          empty_(buildSceneCandidate({}, encodeEnvelope({}))), fault_(std::move(fault)) {}
    SdfAbstractDataRefPtr InitData(const FileFormatArguments&) const override {
        // USD also calls this for TransferContent/Clear: never return the
        // existing layer backing or a prepopulated scene.
        return TfCreateRefPtr(new HostSceneProjection(empty_, fault_));
    }
    static void publish(SdfLayer* layer, SdfAbstractDataRefPtr data) { _SetLayerData(layer, data); }
    bool CanRead(const std::string&) const override { return false; }
    bool Read(SdfLayer*, const std::string&, bool) const override { return false; }
    static SdfAbstractDataConstPtr backing(const SdfLayer& layer) { return _GetLayerData(layer); }
protected:
    SdfAbstractDataRefPtr _InitDetachedData(const FileFormatArguments& args) const override { return InitData(args); }
};
struct Notices final : TfWeakBase {
    const std::shared_ptr<HostFailureLatch> fault;
    const TfRefPtr<HostSceneProjection> projection;
    TfNotice::Key key;
    unsigned delivered = 0;
    Notices(const SdfLayerHandle& layer, std::shared_ptr<HostFailureLatch> value, TfRefPtr<HostSceneProjection> recorder)
        : fault(std::move(value)), projection(std::move(recorder)) {
        key = TfNotice::Register(TfCreateWeakPtr(this), &Notices::changed, layer);
    }
    ~Notices() { TfNotice::Revoke(key); }
    // Plain edits are recorded before USD notifies; an edit that cannot be
    // recorded latches the host failure first.
    void changed(const SdfNotice::LayersDidChangeSentPerLayer&, const SdfLayerHandle&) {
        require(projection->hasPendingLocalEdits() || fault->result().load() != OMNI_USD_LIVE_OK,
                "notice preceded mutation record or latch");
        ++delivered;
    }
};
void rawMutators() {
    const auto source = baseline();
    const auto original = inspect(*source->data());
    using Recorded = SdfOperationKind;
    // Plain mutations record one operation without latching.
    auto check = [&](Recorded expected, auto edit) {
        auto fault = std::make_shared<HostFailureLatch>();
        const auto actual = TfCreateRefPtr(new HostSceneProjection(source, fault));
        const auto stock = TfCreateRefPtr(new SdfData); stock->CopyFrom(source->data());
        const auto owned = TfCreateRefPtr(new HostSceneStorage); owned->CopyFrom(source->data());
        require(fault->result().load() == OMNI_USD_LIVE_OK, "preparation latched host mutation");
        edit(*actual); edit(*stock); edit(*owned);
        require(fault->result().load() == OMNI_USD_LIVE_OK, "recorded mutation latched a fault");
        const auto edits = actual->pendingLocalEdits();
        require(edits.operations->size() == 1 && edits.operations->front().kind == expected,
                "wrong recorded operation kind");
        require(inspect(*actual) == inspect(*stock), "forwarded data differs from stock USD");
        require(inspect(*owned) == inspect(*stock), "owned storage differs from stock USD");
        require(actual->baseline() == source && inspect(*source->data()) == original, "baseline changed");
    };
    // Whole-content CopyFrom is rejected with a typed std::exception before any
    // storage change: nothing is recorded, latched or forwarded.
    auto rejectsCopy = [&](auto copy) {
        auto fault = std::make_shared<HostFailureLatch>();
        const auto actual = TfCreateRefPtr(new HostSceneProjection(source, fault));
        actual->Set(SdfPath("/P"), SdfFieldKeys->Documentation, VtValue(std::string("host")));
        const auto before = inspect(*actual);
        TfErrorMark errors;
        OmniUsdLiveResult rejected = OMNI_USD_LIVE_OK;
        try { copy(*actual); }
        catch (const std::exception& error) {
            const auto* typed = dynamic_cast<const LayerOperationRejected*>(&error);
            rejected = typed ? typed->result() : OMNI_USD_LIVE_INTERNAL_ERROR;
        }
        require(rejected == OMNI_USD_LIVE_UNSUPPORTED, "CopyFrom was not rejected with a typed UNSUPPORTED exception");
        require(!errors.IsClean(), "CopyFrom rejection posted no USD diagnostic");
        errors.Clear();
        require(inspect(*actual) == before && actual->pendingLocalEditCount() == 1, "rejected CopyFrom changed storage or records");
        require(fault->result().load() == OMNI_USD_LIVE_OK, "rejected CopyFrom latched a fault");
        require(actual->baseline() == source && inspect(*source->data()) == original, "baseline changed");
    };
    const SdfPath prim("/P"), attribute("/P/Child.value");
    const TfToken doc("documentation"), custom("customData"), key("nested:value");
    check(Recorded::CreateSpec, [](auto& d) { d.CreateSpec(SdfPath("/New"), SdfSpecTypePrim); });
    check(Recorded::EraseSpec, [&](auto& d) { d.EraseSpec(attribute); });
    check(Recorded::MoveSpec, [&](auto& d) { d.MoveSpec(attribute, SdfPath("/P/Child.renamed")); });
    check(Recorded::SetField, [&](auto& d) { d.Set(prim, doc, VtValue(std::string("changed"))); });
    check(Recorded::SetField, [&](auto& d) {
        const std::string text("boxed"); const SdfAbstractDataConstTypedValue<std::string> value(&text);
        d.Set(prim, doc, value);
    });
    check(Recorded::SetField, [&](auto& d) {
        const VtValue empty; const SdfAbstractDataConstTypedValue<VtValue> value(&empty);
        d.Set(prim, doc, value);
    });
    check(Recorded::EraseField, [&](auto& d) { d.Erase(attribute, SdfFieldKeys->Default); });
    check(Recorded::SetDictKey, [&](auto& d) { d.SetDictValueByKey(prim, custom, key, VtValue(19)); });
    check(Recorded::SetDictKey, [&](auto& d) {
        const int number = 23; const SdfAbstractDataConstTypedValue<int> value(&number);
        d.SetDictValueByKey(prim, custom, key, value);
    });
    check(Recorded::EraseDictKey, [&](auto& d) { d.EraseDictValueByKey(prim, custom, key); });
    check(Recorded::SetSample, [&](auto& d) { d.SetTimeSample(attribute, 1.5, VtValue(11.0)); });
    check(Recorded::EraseSample, [&](auto& d) { d.EraseTimeSample(attribute, 1.5); });
    rejectsCopy([&](auto& d) { d.CopyFrom(source->data()); });
    rejectsCopy([&](auto& d) { d.CopyFrom(SdfAbstractDataConstPtr(&d)); });
    rejectsCopy([&](auto& d) { d.CopyFrom({}); });
    // A no-op still records the attempt, rather than claiming any retained edit.
    check(Recorded::EraseField, [&](auto& d) { d.Erase(prim, TfToken("absentField")); });
    auto shared = std::make_shared<HostFailureLatch>();
    const auto first = TfCreateRefPtr(new HostSceneProjection(source, shared));
    const auto second = TfCreateRefPtr(new HostSceneProjection(source, shared));
    first->Set(prim, doc, VtValue(std::string("first")));
    second->EraseSpec(attribute);
    require(shared->result().load() == OMNI_USD_LIVE_OK, "plain projection mutations latched the shared fault");
    require(first->hasPendingLocalEdits() && second->hasPendingLocalEdits() &&
            first->pendingLocalEditCount() == 1 && second->pendingLocalEditCount() == 1,
            "projections share an operation buffer");
    require(first->HasSpec(attribute) && !second->HasSpec(attribute), "projections alias mutable storage");
    const auto captured = first->capture();
    const auto preserved = inspect(*captured);
    first->Set(prim, doc, VtValue(std::string("later")));
    require(inspect(*captured) == preserved, "capture follows later edits");
    require(inspect(*source->data()) == original, "retained source changed after capture");
}
void ownedStorageCallbacks() {
    const auto source=baseline();
    const auto owned=TfCreateRefPtr(new HostSceneStorage); owned->CopyFrom(source->data());
    const auto stock=TfCreateRefPtr(new SdfData); stock->CopyFrom(source->data());
    const SdfPath path("/P"); const TfToken field("documentation");
    for(const auto& data : {SdfAbstractDataRefPtr(owned),SdfAbstractDataRefPtr(stock)}) {
        data->Set(path,field,VtValue(std::string("old")));
        data->CreateSpec(path,SdfSpecTypePrim); // Preserve fields of existing spec.
        data->Set(path,TfToken("orderedLast"),VtValue(2));
        data->Set(path,field,VtValue(std::string("new"))); // Preserve field position.
        data->Erase(path,TfToken("orderedLast")); data->Set(path,TfToken("orderedLast"),VtValue(3));
        data->Set(path,TfToken("timeSamples"),VtValue(std::string("not a sample map")));
        require(data->ListTimeSamplesForPath(path).empty(),"malformed sample map was interpreted");
        data->SetTimeSample(path,1,VtValue(4.0));
        data->CopyFrom(source->data());
    }
    require(inspect(*owned)==inspect(*stock) && owned->List(path)==stock->List(path),"owned overlay/order parity failed");
    unsigned blocked=0;
    auto attempt=[&] {
        try { owned->EraseSpec(path); }
        catch(OmniUsdLiveResult result) { require(result==OMNI_USD_LIVE_BUSY,"wrong reentry result"); ++blocked; }
    };
    struct Input final : SdfAbstractDataConstValue {
        std::function<void()> callback;
        bool succeeds=true;
        Input(std::function<void()> fn) : SdfAbstractDataConstValue(nullptr,typeid(void),false,typeid(void)),callback(std::move(fn)) {}
        bool GetValue(VtValue* output) const override { callback(); *output=VtValue(std::string("input")); return succeeds; }
        bool IsEqual(const VtValue&) const override { return false; }
    } input(attempt);
    owned->Set(path,field,input);
    require(blocked==1 && owned->Get(path,field)==VtValue(std::string("input")),"input callback invalidated mutation");
    input.succeeds=false;
    const auto prior=inspect(*owned); bool rejected=false;
    try { owned->Set(path,field,input); }
    catch(OmniUsdLiveResult result) { require(result==OMNI_USD_LIVE_UNSUPPORTED,"wrong failed extraction result"); rejected=true; }
    require(rejected && inspect(*owned)==prior,"failed foreign extraction partially committed");
    struct Output final : SdfAbstractDataValue {
        std::function<void()> callback; VtValue received;
        Output(std::function<void()> fn) : SdfAbstractDataValue(nullptr,typeid(void)),callback(std::move(fn)) {}
        bool _StoreVtValue(const VtValue& value) override { callback(); received=value; return true; }
        bool _StoreVtValue(VtValue&& value) override { return _StoreVtValue(value); }
    } output(attempt);
    require(owned->Has(path,field,&output) && output.received==VtValue(std::string("input")),"output callback lost pinned value");
    struct Visitor final : SdfAbstractDataSpecVisitor {
        std::function<void()> callback;
        explicit Visitor(std::function<void()> fn):callback(std::move(fn)) {}
        bool VisitSpec(const SdfAbstractData&,const SdfPath&) override { callback(); return false; }
        void Done(const SdfAbstractData&) override {}
    } visitor(attempt);
    owned->VisitSpecs(&visitor);
    require(blocked==4 && inspect(*owned)==prior,"visitor callback changed live traversal");
    struct Foreign final : SdfData {
        std::function<void()> callback;
        VtValue Get(const SdfPath& p,const TfToken& f) const override { callback(); return SdfData::Get(p,f); }
    };
    const auto foreign=TfCreateRefPtr(new Foreign); foreign->CopyFrom(source->data()); foreign->callback=attempt;
    owned->CopyFrom(foreign);
    require(blocked>4 && inspect(*owned)==prior,"foreign overlay invalidated destination");
    foreign->callback=[] { throw std::runtime_error("foreign source failure"); };
    rejected=false; try { owned->CopyFrom(foreign); } catch(const std::runtime_error&) { rejected=true; }
    require(rejected && inspect(*owned)==prior,"throwing foreign source partially committed");
    owned->Set(path,field,VtValue(std::string("after failure")));
    require(owned->Get(path,field)==VtValue(std::string("after failure")),"callback guard survived unwind");
    output.callback=[] { throw std::runtime_error("output callback"); };
    visitor.callback=[] { throw std::runtime_error("visitor callback"); };
    for(bool visit:{false,true}) {
        rejected=false;
        try { if(visit) owned->VisitSpecs(&visitor); else owned->Has(path,field,&output); }
        catch(const std::runtime_error&) { rejected=true; }
        require(rejected,"callback exception disappeared");
        owned->Set(path,field,VtValue(std::string("after callback")));
        require(owned->Get(path,field)==VtValue(std::string("after callback")),"throwing callback left guard active");
    }
    // Empty dictionary paths follow stock behavior, including diagnostics.
    for(bool erase:{false,true}) {
        bool stockClean=false,ownedClean=false;
        for(const auto& data:{SdfAbstractDataRefPtr(stock),SdfAbstractDataRefPtr(owned)}) {
            TfErrorMark errors;
            if(erase) data->EraseDictValueByKey(path,TfToken("customData"),TfToken());
            else data->SetDictValueByKey(path,TfToken("customData"),TfToken(),VtValue(2));
            (data==stock ? stockClean : ownedClean)=errors.IsClean(); errors.Clear();
        }
        require(stockClean==ownedClean && stock->Get(path,TfToken("customData"))==owned->Get(path,TfToken("customData")),
                "empty dictionary path differs from stock");
    }
    // Immutable captures can be visited concurrently. Force the first visitor
    // to retire while the second remains active; restoring a prior bool is wrong.
    std::mutex mutex; std::condition_variable changed;
    unsigned entered=0,released=0;
    auto traverse=[&](unsigned bit) {
        Visitor pending([&] {
            std::unique_lock<std::mutex> lock(mutex); entered|=bit; changed.notify_all();
            changed.wait(lock,[&] { return (released&bit)!=0; });
        });
        owned->VisitSpecs(&pending);
    };
    std::thread first(traverse,1);
    { std::unique_lock<std::mutex> lock(mutex); changed.wait(lock,[&] { return entered==1; }); }
    std::thread second(traverse,2);
    { std::unique_lock<std::mutex> lock(mutex); changed.wait(lock,[&] { return entered==3; }); released=1; changed.notify_all(); }
    first.join();
    { std::lock_guard<std::mutex> lock(mutex); released=3; changed.notify_all(); }
    second.join();
    owned->Set(path,field,VtValue(std::string("after both readers")));
    require(owned->Get(path,field)==VtValue(std::string("after both readers")),"frozen traversal left stale reentry accounting");
}
void ownedStorageRetirement() {
    struct CallbackState { bool armed=false,called=false,coherent=false,failed=false; std::function<void()> action; };
    struct Payload {
        std::shared_ptr<CallbackState> state;
        ~Payload() { if(state && state->armed) { state->called=true; try { state->action(); } catch(...) { state->failed=true; } } }
    };
    const SdfPath path("/P"),after("/AfterRetirement"); const TfToken field("retiring");
    for(unsigned operation=0;operation!=3;++operation) {
        const auto owned=TfCreateRefPtr(new HostSceneStorage);
        owned->CreateSpec(path,SdfSpecTypePrim);
        const auto replacement=TfCreateRefPtr(new HostSceneStorage); replacement->CreateSpec(path,SdfSpecTypePrim);
        replacement->Set(path,field,VtValue(std::string("replacement")));
        const auto state=std::make_shared<CallbackState>();
        struct Disarm { std::shared_ptr<CallbackState> state; ~Disarm() { state->armed=false; } } disarm{state};
        auto payload=std::make_shared<Payload>(); payload->state=state;
        owned->Set(path,field,VtValue(payload)); payload.reset();
        state->action=[&,operation] {
            state->coherent=operation==1 ? !owned->HasSpec(path) : owned->Get(path,field)==VtValue(std::string("replacement"));
            // This replaces the whole State while the outer mutator is returning.
            owned->CopyFrom(replacement);
            owned->CreateSpec(after,SdfSpecTypePrim);
        };
        state->armed=true;
        if(operation==0) owned->Set(path,field,VtValue(std::string("replacement")));
        else if(operation==1) owned->EraseSpec(path);
        else owned->CopyFrom(replacement);
        state->armed=false;
        require(state->called && state->coherent && !state->failed && owned->HasSpec(after),
                "retired value ran under a guard or before complete state publication");
    }
    // Failed foreign extraction can create a value with its own destructor.
    const auto owned=TfCreateRefPtr(new HostSceneStorage); owned->CreateSpec(path,SdfSpecTypePrim);
    const auto state=std::make_shared<CallbackState>();
    struct Disarm { std::shared_ptr<CallbackState> state; ~Disarm() { state->armed=false; } } disarm{state};
    state->action=[&] { state->coherent=!owned->Has(path,field); owned->CreateSpec(after,SdfSpecTypePrim); };
    struct FailingInput final : SdfAbstractDataConstValue {
        std::shared_ptr<CallbackState> state;
        explicit FailingInput(std::shared_ptr<CallbackState> s):SdfAbstractDataConstValue(nullptr,typeid(void),false,typeid(void)),state(std::move(s)) {}
        bool GetValue(VtValue* output) const override {
            auto payload=std::make_shared<Payload>(); payload->state=state; *output=VtValue(payload);
            throw std::runtime_error("failed extraction after constructing value");
        }
        bool IsEqual(const VtValue&) const override { return false; }
    } input(state);
    bool caught=false; state->armed=true;
    try { owned->Set(path,field,input); } catch(const std::runtime_error&) { caught=true; }
    state->armed=false;
    require(caught && state->called && state->coherent && !state->failed && owned->HasSpec(after),
            "failed foreign value retired before guard unwind");
}
void callbackOwnerLifetimes() {
    const auto source=baseline(); const SdfPath path("/P"); const TfToken field("documentation");
    for(bool projection:{false,true}) for(bool output:{false,true}) {
        SdfAbstractDataRefPtr owner=projection ? SdfAbstractDataRefPtr(TfCreateRefPtr(new HostSceneProjection(source,std::make_shared<HostFailureLatch>()))) :
            SdfAbstractDataRefPtr(TfCreateRefPtr(new HostSceneStorage));
        if(!projection) owner->CopyFrom(source->data());
        owner->Set(path,field,VtValue(std::string("retained")));
        const SdfAbstractDataConstPtr weak(owner);
        bool called=false;
        auto drop=[&] { owner.Reset(); called=true; require(bool(weak),"callback destroyed active receiver"); };
        struct Input final : SdfAbstractDataConstValue {
            std::function<void()> fn;
            explicit Input(std::function<void()> f):SdfAbstractDataConstValue(nullptr,typeid(void),false,typeid(void)),fn(std::move(f)) {}
            bool GetValue(VtValue* out) const override { fn(); *out=VtValue(std::string("input")); return true; }
            bool IsEqual(const VtValue&) const override { return false; }
        } input(drop);
        struct Output final : SdfAbstractDataValue {
            std::function<void()> fn; VtValue received;
            explicit Output(std::function<void()> f):SdfAbstractDataValue(nullptr,typeid(void)),fn(std::move(f)) {}
            bool _StoreVtValue(const VtValue& v) override { fn(); received=v; return true; }
            bool _StoreVtValue(VtValue&& v) override { return _StoreVtValue(v); }
        } result(drop);
        auto* raw=owner.operator->();
        if(output) require(raw->Has(path,field,&result) && result.received==VtValue(std::string("retained")),"last-owner query lost value");
        else raw->Set(path,field,input);
        require(called && !weak,"receiver pin leaked beyond virtual call");
    }
    const auto projection=TfCreateRefPtr(new HostSceneProjection(source,std::make_shared<HostFailureLatch>()));
    const auto before=inspect(*projection);
    struct Foreign final : SdfData {
        std::function<void()> callback;
        VtValue Get(const SdfPath& p,const TfToken& f) const override { callback(); return SdfData::Get(p,f); }
    };
    auto foreign=TfCreateRefPtr(new Foreign); foreign->CopyFrom(source->data());
    unsigned callbacks=0; foreign->callback=[&] { ++callbacks; };
    // The rejection precedes any source traversal.
    TfErrorMark errors; bool rejected=false;
    try { projection->CopyFrom(foreign); } catch(const LayerOperationRejected&) { rejected=true; }
    errors.Clear();
    require(rejected && callbacks==0 && inspect(*projection)==before,"projection CopyFrom read its source or changed storage");
    // VisitSpecs has a nonvirtual SDK epilogue that calls Done after _VisitSpecs.
    // The invoking boundary must own the receiver through the complete call.
    SdfAbstractDataConstRefPtr traversal=projection->capture();
    const SdfAbstractDataConstPtr weakTraversal(traversal); bool done=false;
    struct Visitor final : SdfAbstractDataSpecVisitor {
        std::function<void()> visit,finish;
        bool VisitSpec(const SdfAbstractData&,const SdfPath&) override { visit(); return false; }
        void Done(const SdfAbstractData&) override { finish(); }
    } visitor;
    visitor.visit=[&] { traversal.Reset(); require(bool(weakTraversal),"traversal owner disappeared during callback"); };
    visitor.finish=[&] { require(bool(weakTraversal),"traversal owner disappeared before SDK Done"); done=true; };
    auto visitOwned=[](SdfAbstractDataConstRefPtr owner,SdfAbstractDataSpecVisitor* target) { owner->VisitSpecs(target); };
    visitOwned(traversal,&visitor);
    require(done && !weakTraversal,"whole-call traversal pin leaked");
}
// USD 25.11 streams a projection-backed layer: Clear adopts fresh InitData
// without touching any mutator, so only the layer owner can detect it, while
// TransferContent first calls CopyFrom on fresh InitData, which rejects it
// before the layer changes. Production's shared format cannot assume this
// probe's per-layer InitData factory and tracks replacement at the layer owner.
OmniUsdLiveResult rejection(const std::function<void()>& operation) {
    TfErrorMark errors;
    OmniUsdLiveResult result = OMNI_USD_LIVE_OK;
    try { operation(); }
    catch (const std::exception& error) {
        const auto* typed = dynamic_cast<const LayerOperationRejected*>(&error);
        result = typed ? typed->result() : OMNI_USD_LIVE_INTERNAL_ERROR;
    }
    errors.Clear();
    return result;
}
void replacementControls() {
    const auto source = baseline();
    const auto original = inspect(*source->data());
    for (bool clear : {true, false}) {
        auto fault = std::make_shared<HostFailureLatch>();
        const auto initial = TfCreateRefPtr(new HostSceneProjection(source, fault));
        const SdfFileFormatRefPtr format = TfCreateRefPtr(new Format(fault));
        const auto layer = SdfLayer::CreateAnonymous("replacement.host_projection_probe", format);
        require(bool(layer), "replacement control layer not created");
        Format::publish(layer.operator->(), initial);
        const auto expected = SdfLayer::CreateAnonymous("replacement-control.usda");
        require(fault->result().load() == OMNI_USD_LIVE_OK, "fresh replacement control already latched");
        if (!clear) {
            const auto replacement = SdfLayer::CreateAnonymous("replacement-source.usda");
            require(replacement->ImportFromString("#usda 1.0\ndef \"Replacement\" {}\n"), "replacement control seed failed");
            require(rejection([&] { layer->TransferContent(replacement); }) == OMNI_USD_LIVE_UNSUPPORTED,
                    "TransferContent was not rejected with a typed UNSUPPORTED exception");
            require(Format::backing(*layer).operator->() == initial.operator->() && inspect(*initial) == original,
                    "rejected TransferContent replaced or changed the backing");
            require(fault->result().load() == OMNI_USD_LIVE_OK,
                    "rejected TransferContent latched a fault");
            continue;
        }
        layer->Clear();
        const auto* current = dynamic_cast<const HostSceneProjection*>(Format::backing(*layer).operator->());
        require(current && current != initial.operator->() && current->fault() == fault,
                "Clear did not adopt independent backing with shared owner");
        require(inspect(*current) == inspect(*Format::backing(*expected)), "Clear content differs from stock USD");
        require(inspect(*initial) == original && inspect(*source->data()) == original,
                "Clear changed retained host or service baseline");
        require(fault->result().load() == OMNI_USD_LIVE_OK, "Clear reached a mutation hook");
    }
}
void publicLayerOperations() {
    const auto source = baseline();
    const auto original = inspect(*source->data());
    auto fault = std::make_shared<HostFailureLatch>();
    const auto projection = TfCreateRefPtr(new HostSceneProjection(source, fault));
    // Format outlives layer; this probe does not install a production handler.
    const SdfFileFormatRefPtr format = TfCreateRefPtr(new Format(fault));
    const auto actual = SdfLayer::CreateAnonymous("host.host_projection_probe", format);
    const auto expected = SdfLayer::CreateAnonymous("stock.usda");
    require(bool(actual), "projection layer not created");
    Format::publish(actual.operator->(), projection);
    require(Format::backing(*actual).operator->() == projection.operator->(), "projection not adopted");
    require(expected->ImportFromString("#usda 1.0\ndef \"P\"\n{\n    def \"Child\"\n    {\n        double value = 7\n    }\n}\n"), "stock seed failed");
    require(inspect(*Format::backing(*expected)) == original, "independent stock seed differs");
    actual->SetPermissionToEdit(false);
    // Public permission override is the known crash trigger. No app guard.
    actual->SetPermissionToEdit(true);
    Notices notices(actual, fault, projection);
    const SdfSpec retained = actual->GetObjectAtPath(SdfPath("/P/Child.value")).GetSpec();
    auto both = [&](const char* operation, auto edit) {
        edit(actual); edit(expected);
        const auto actualRecords = inspect(*Format::backing(*actual));
        const auto expectedRecords = inspect(*Format::backing(*expected));
        if (actualRecords != expectedRecords) {
            std::cerr << "public USD mismatch after " << operation << '\n';
            for (const auto& [path, record] : actualRecords) {
                const auto found = expectedRecords.find(path);
                if (found != expectedRecords.end() && found->second == record) continue;
                std::cerr << "actual " << path << " type=" << record.first << '\n';
                for (const auto& [field, value] : record.second) std::cerr << "  " << field << " = " << value << '\n';
            }
            for (const auto& [path, record] : expectedRecords) {
                const auto found = actualRecords.find(path);
                if (found != actualRecords.end() && found->second == record) continue;
                std::cerr << "expected " << path << " type=" << record.first << '\n';
                for (const auto& [field, value] : record.second) std::cerr << "  " << field << " = " << value << '\n';
            }
            require(false, "public USD content differs");
        }
        require(inspect(*source->data()) == original, "public edit changed immutable source");
    };
    both("compound authoring", [](const auto& layer) {
        SdfChangeBlock block;
        auto prim = SdfCreatePrimInLayer(layer, SdfPath("/P/New/Grandchild"));
        auto attr = SdfAttributeSpec::New(prim, "x", SdfValueTypeNames->Double);
        attr->SetDefaultValue(VtValue(29.0));
        layer->SetFieldDictValueByKey(SdfPath("/P"), SdfFieldKeys->CustomData, TfToken("nested:value"), VtValue(17));
        layer->SetTimeSample(SdfPath("/P/Child.value"), 2.0, VtValue(13.0));
    });
    require(notices.delivered == 1, "compound edit did not complete one outer notice boundary");
    both("subtree move", [](const auto& layer) {
        SdfBatchNamespaceEdit edits;
        edits.Add(SdfPath("/P"), SdfPath("/Moved"));
        require(layer->Apply(edits), "subtree namespace move rejected");
    });
    require(!retained.IsDormant() && retained.GetPath() == SdfPath("/Moved/Child.value"), "retained handle disagrees with move");
    require(projection->HasSpec(retained.GetPath()), "moved retained handle has no backing");
    both("subtree removal", [](const auto& layer) {
        layer->EraseTimeSample(SdfPath("/Moved/Child.value"), 2.0);
        layer->EraseFieldDictValueByKey(SdfPath("/Moved"), SdfFieldKeys->CustomData, TfToken("nested:value"));
        SdfBatchNamespaceEdit edits;
        edits.Add(SdfPath("/Moved/New"), SdfPath());
        require(layer->Apply(edits), "subtree removal rejected");
    });
    const auto captured = projection->capture();
    const auto before = inspect(*captured);
    const auto replacement = SdfLayer::CreateAnonymous("replacement.usda");
    require(replacement->ImportFromString("#usda 1.0\ndef \"Replacement\" {}\n"), "replacement seed failed");
    const auto delivered = notices.delivered;
    require(rejection([&] { actual->TransferContent(replacement); }) == OMNI_USD_LIVE_UNSUPPORTED,
            "TransferContent was not rejected with a typed UNSUPPORTED exception");
    require(Format::backing(*actual).operator->() == projection.operator->() && inspect(*projection) == before,
            "rejected TransferContent replaced or changed the backing");
    require(inspect(*captured) == before, "rejected TransferContent altered retained capture");
    require(!retained.IsDormant() && retained.GetPath() == SdfPath("/Moved/Child.value"),
            "rejected TransferContent invalidated a live spec handle");
    require(fault->result().load() == OMNI_USD_LIVE_OK && notices.delivered == delivered && delivered >= 3,
            "rejected TransferContent latched a fault or sent a notice");
}
void directAuthoringRecordsOperations() {
    auto fault = std::make_shared<HostFailureLatch>();
    const auto base = buildSceneCandidate({}, encodeEnvelope({}));
    auto projection = TfCreateRefPtr(new HostSceneProjection(base, fault));
    const auto root = SdfPath::AbsoluteRootPath();
    std::vector<size_t> recorded;
    projection->setLocalEditSink([&](size_t pending) { recorded.push_back(pending); });
    projection->Set(root, TfToken("a"), VtValue(1));
    projection->Set(root, TfToken("b"), VtValue(2));
    require(recorded == std::vector<size_t>{1, 2}, "sink sees every record with the buffer size");
    require(fault->result().load() == OMNI_USD_LIVE_OK, "plain mutations do not fail the layer");
    const auto edits = projection->pendingLocalEdits();
    require(edits.operations->size() == 2 && projection->pendingLocalEditCount() == 2, "peek keeps the buffer");
    require((*edits.operations)[0].field == TfToken("a") && edits.data->Get(root, TfToken("b")) == VtValue(2) &&
            edits.base == base, "peek returns exact records, capture and base");
    projection->Set(root, TfToken("c"), VtValue(3));
    projection->acceptLocalEdits(edits.operations->size(), base);
    require(projection->pendingLocalEditCount() == 1 &&
            (*projection->pendingLocalEdits().operations)[0].field == TfToken("c"), "accept drops only the peeked records");
    bool overrun = false;
    try { projection->acceptLocalEdits(2, base); } catch (const std::logic_error&) { overrun = true; }
    require(overrun && projection->pendingLocalEditCount() == 1, "accepting more records than buffered was allowed");
    projection->setLocalEditSink([](size_t) { throw std::runtime_error("sink"); });
    projection->Set(root, TfToken("d"), VtValue(4));
    require(fault->result().load() == OMNI_USD_LIVE_INTERNAL_ERROR && projection->Get(root, TfToken("d")) == VtValue(4),
            "a throwing sink escaped the mutator or lost the edit");
    HostFailureLatch first;
    first.fail(OMNI_USD_LIVE_UNSUPPORTED); first.fail(OMNI_USD_LIVE_INTERNAL_ERROR); first.fail(OMNI_USD_LIVE_OK);
    require(first.result().load() == OMNI_USD_LIVE_UNSUPPORTED, "the first host failure cause is not latched");
}
// A quota of `records` buffered root int field records.
HostSceneProjection::Limits recordQuota(size_t records) {
    SdfOperation record(SdfOperationKind::SetField);
    record.path = SdfPath::AbsoluteRootPath();
    record.field = TfToken("a");
    record.value = VtValue(1);
    return HostSceneProjection::Limits{records * hostEditRecordBytes(record)};
}
// The buffer's bytes are its records' hostEditRecordBytes: values charge their
// heap bytes, admission and discard release them, and the budget sink applies
// the quota at every host turn start, never inside a started turn.
// Time sample maps, list operations, token, path and string vectors, asset
// paths and string-bearing arrays charge their nodes, items and characters,
// and a larger value of each kind charges more.
void recordBytesOfLargeKinds() {
    SdfOperation empty(SdfOperationKind::SetField);
    empty.value = VtValue(1);
    const auto record = hostEditRecordBytes(empty);
    const auto charge = [&](VtValue value) {
        SdfOperation operation = empty;
        operation.value = std::move(value);
        return hostEditRecordBytes(operation) - record;
    };
    SdfTimeSampleMap scalars, arrays;
    for (int i = 0; i < 100; ++i) {
        scalars[i] = VtValue(double(i));
        arrays[i] = VtValue(VtArray<float>(10));
    }
    require(charge(VtValue(scalars)) == 100 * 64 && charge(VtValue(arrays)) == 100 * (64 + 10 * 16),
            "a time sample map does not charge its samples");
    TfTokenVector tokens(20);
    for (size_t i = 0; i < tokens.size(); ++i) tokens[i] = TfToken("child" + std::to_string(i));
    require(charge(VtValue(tokens)) == 20 * 16, "a children-order token vector does not charge its items");
    require(charge(VtValue(SdfPathVector(5, SdfPath("/A")))) == 5 * 16, "a path vector does not charge its items");
    require(charge(VtValue(std::vector<std::string>{"abc", "de"})) == 2 * sizeof(std::string) + 5,
            "a string vector does not charge its characters");
    SdfTokenListOp tokenOp;
    tokenOp.SetExplicitItems(tokens);
    SdfPathListOp pathOp;
    // List operations drop duplicate items, so each list holds distinct paths.
    pathOp.SetPrependedItems({SdfPath("/A"), SdfPath("/B"), SdfPath("/C"), SdfPath("/D"), SdfPath("/E")});
    pathOp.SetDeletedItems({SdfPath("/F"), SdfPath("/G"), SdfPath("/H")});
    SdfStringListOp stringOp;
    stringOp.SetAppendedItems({std::string(100, 's')});
    SdfReferenceListOp referenceOp;
    referenceOp.SetPrependedItems({SdfReference("a.usd")});
    SdfPayloadListOp payloadOp;
    payloadOp.SetPrependedItems({SdfPayload("pay.usd")});
    require(charge(VtValue(tokenOp)) == 20 * 16 && charge(VtValue(pathOp)) == 8 * 16 &&
            charge(VtValue(stringOp)) == sizeof(std::string) + 100 &&
            charge(VtValue(referenceOp)) == std::max<size_t>(16, sizeof(SdfReference)) + 5 &&
            charge(VtValue(payloadOp)) == std::max<size_t>(16, sizeof(SdfPayload)) + 7,
            "a list operation does not charge its items");
    require(charge(VtValue(SdfAssetPath("x/y.usd"))) == 7, "an asset path does not charge its characters");
    require(charge(VtValue(VtArray<std::string>{"ab", "cde"})) == 2 * std::max<size_t>(16, sizeof(std::string)) + 5 &&
            charge(VtValue(VtArray<SdfAssetPath>{SdfAssetPath("a.usd")})) ==
                std::max<size_t>(16, sizeof(SdfAssetPath)) + 5,
            "a string-bearing array does not charge its characters");
    // Larger values of each kind charge more.
    scalars[100] = VtValue(1.0);
    tokens.push_back(TfToken("more"));
    tokenOp.SetExplicitItems(tokens);
    require(charge(VtValue(scalars)) > 100 * 64 && charge(VtValue(tokenOp)) > 20 * 16 &&
            charge(VtValue(VtArray<std::string>{"ab", "cdef"})) > charge(VtValue(VtArray<std::string>{"ab", "cde"})),
            "a larger value does not charge more");
}
// A buffer for which the budget sink reports no shared budget, as for a
// layer without an attachment, is bounded by the layer's own quota, which is
// finite by default.
void unchargedBufferKeepsItsQuota() {
    require(HostSceneProjection::Limits{}.maxPendingBytes == size_t(1) << 30, "the default layer quota is not 1 GiB");
    const auto base = buildSceneCandidate({}, encodeEnvelope({}));
    const auto root = SdfPath::AbsoluteRootPath();
    auto projection = TfCreateRefPtr(new HostSceneProjection(base, std::make_shared<HostFailureLatch>(), recordQuota(2)));
    size_t consulted = 0, refusals = 0;
    projection->setBudgetSink([&](size_t) { ++consulted; return HostEditBudget{}; });
    projection->setQuotaSink([&](size_t) { ++refusals; return true; });
    projection->Set(root, TfToken("a"), VtValue(1));
    projection->endHostTurn();
    projection->Set(root, TfToken("b"), VtValue(2));
    projection->endHostTurn();
    TfErrorMark errors;
    bool refused = false;
    try { projection->Set(root, TfToken("c"), VtValue(3)); }
    catch (const LayerOperationRejected& error) { refused = error.result() == OMNI_USD_LIVE_RESOURCE_EXHAUSTED; }
    bool named = false;
    for (const auto& error : errors) named = named || error.GetCommentary().find("layer's quota") != std::string::npos;
    errors.Clear();
    require(refused && named && consulted == 2 && refusals == 1 && projection->pendingLocalEditCount() == 2 &&
            projection->Get(root, TfToken("c")).IsEmpty(), "an uncharged buffer passed the layer's own quota");
}
void bufferBytesAndBudget() {
    const auto root = SdfPath::AbsoluteRootPath();
    SdfOperation integer(SdfOperationKind::SetField);
    integer.value = VtValue(1);
    SdfOperation text = integer, array = integer, matrices = integer;
    text.value = VtValue(std::string(1000, 'x'));
    array.value = VtValue(VtArray<double>(1000));
    matrices.value = VtValue(VtArray<GfMatrix4d>(1000));
    const auto record = hostEditRecordBytes(integer);
    require(record >= sizeof(SdfOperation) && hostEditRecordBytes(text) == record + 1000 &&
            hostEditRecordBytes(array) == record + 16000 &&
            hostEditRecordBytes(matrices) == record + 1000 * sizeof(GfMatrix4d),
            "a record does not charge its value's heap bytes");
    const auto base = buildSceneCandidate({}, encodeEnvelope({}));
    auto fault = std::make_shared<HostFailureLatch>();
    auto projection = TfCreateRefPtr(new HostSceneProjection(base, fault));
    require(projection->limits().maxPendingBytes == size_t(1) << 30,
            "a layer's own quota is not 1 GiB by default");
    std::vector<size_t> budgets;
    size_t refusals = 0;
    HostEditBudget reported{false, 0, 100};
    projection->setBudgetSink([&](size_t bytes) { budgets.push_back(bytes); return reported; });
    projection->setQuotaSink([&](size_t) { ++refusals; return true; });
    projection->Set(root, TfToken("a"), VtValue(1));
    projection->Set(root, TfToken("t"), VtValue(std::string(1000, 'x')));
    require(projection->pendingLocalEditBytes() == 2 * record + 1000 && budgets == std::vector<size_t>{0},
            "a started turn consulted the budget again or miscounted its bytes");
    projection->endHostTurn();
    // An exhausted budget refuses the next turn before any change.
    reported = {true, 100, 100};
    TfErrorMark errors;
    bool refused = false;
    try { projection->Set(root, TfToken("b"), VtValue(2)); }
    catch (const LayerOperationRejected& error) { refused = error.result() == OMNI_USD_LIVE_RESOURCE_EXHAUSTED; }
    require(refused && refusals == 1 && projection->Get(root, TfToken("b")).IsEmpty() &&
            projection->pendingLocalEditCount() == 2 && budgets.size() == 2 && budgets[1] == 2 * record + 1000,
            "an exhausted budget did not refuse a new turn before any change");
    require(!errors.IsClean(), "a budget refusal posted no USD runtime error");
    errors.Clear();
    // Admission releases the admitted records' bytes.
    projection->acceptLocalEdits(1, base);
    require(projection->pendingLocalEditBytes() == record + 1000, "an admission kept the admitted bytes");
    reported = {false, 0, 100};
    projection->Set(root, TfToken("b"), VtValue(2));
    require(projection->pendingLocalEditBytes() == 2 * record + 1000, "a turn below the budget did not record");
    projection->endHostTurn();
    // A throwing budget sink fails the layer and refuses with that cause.
    projection->setBudgetSink([](size_t) -> HostEditBudget { throw std::runtime_error("sink"); });
    refused = false;
    try { projection->Set(root, TfToken("c"), VtValue(3)); }
    catch (const LayerOperationRejected& error) { refused = error.result() == OMNI_USD_LIVE_INTERNAL_ERROR; }
    require(refused && fault->result().load() == OMNI_USD_LIVE_INTERNAL_ERROR &&
            projection->Get(root, TfToken("c")).IsEmpty(), "a throwing budget sink applied the edit");
    errors.Clear();
    projection->discardLocalEdits();
    require(projection->pendingLocalEditBytes() == 0, "a discard kept buffered bytes");
}
void directAuthoringQuota() {
    auto fault = std::make_shared<HostFailureLatch>();
    const auto base = buildSceneCandidate({}, encodeEnvelope({}));
    auto projection = TfCreateRefPtr(new HostSceneProjection(base, fault, recordQuota(2)));
    require(projection->limits().maxPendingBytes == recordQuota(2).maxPendingBytes, "projection lost its quota");
    const auto root = SdfPath::AbsoluteRootPath();
    std::vector<size_t> refused;
    projection->setQuotaSink([&](size_t pending) { refused.push_back(pending); return true; });
    // A started turn records past the quota: one USD call never applies partially.
    projection->Set(root, TfToken("a"), VtValue(1));
    projection->Set(root, TfToken("b"), VtValue(2));
    projection->Set(root, TfToken("c"), VtValue(3));
    require(projection->pendingLocalEditCount() == 3 && refused.empty(), "a started turn was refused");
    projection->endHostTurn();
    // A new turn at the quota is refused before any change.
    TfErrorMark errors;
    auto rejected = [&](auto mutate, OmniUsdLiveResult expected = OMNI_USD_LIVE_RESOURCE_EXHAUSTED) {
        try { mutate(); }
        catch (const LayerOperationRejected& error) { return error.result() == expected; }
        return false;
    };
    const auto before = inspect(*projection);
    const int boxed = 7;
    const SdfAbstractDataConstTypedValue<int> boxedValue(&boxed);
    require(rejected([&] { projection->Set(root, TfToken("d"), VtValue(4)); }), "a new turn at the quota was not refused");
    require(rejected([&] { projection->CreateSpec(SdfPath("/Q"), SdfSpecTypePrim); }), "a refused mutation opened a turn");
    require(rejected([&] { projection->Set(root, TfToken("d"), boxedValue); }),
            "an abstract-value Set at the quota was not refused");
    require(rejected([&] { projection->SetDictValueByKey(root, SdfFieldKeys->CustomData, TfToken("d"), boxedValue); }),
            "an abstract-value SetDictValueByKey at the quota was not refused");
    require(inspect(*projection) == before && projection->pendingLocalEditCount() == 3,
            "a refused mutation changed storage or records");
    require(refused == std::vector<size_t>{3, 3, 3, 3}, "the quota sink missed a refusal or its buffer size");
    require(fault->result().load() == OMNI_USD_LIVE_OK, "a quota refusal failed the layer");
    require(!errors.IsClean(), "a quota refusal posted no USD diagnostic");
    errors.Clear();
    // Without a quota sink a new turn at the quota is refused.
    auto unsunk = TfCreateRefPtr(new HostSceneProjection(base, std::make_shared<HostFailureLatch>(),
                                                          recordQuota(1)));
    unsunk->Set(root, TfToken("a"), VtValue(1));
    unsunk->endHostTurn();
    require(rejected([&] { unsunk->Set(root, TfToken("b"), VtValue(2)); }) && unsunk->pendingLocalEditCount() == 1 &&
            unsunk->Get(root, TfToken("b")).IsEmpty() && unsunk->fault()->result().load() == OMNI_USD_LIVE_OK,
            "a projection without a quota sink did not refuse a new turn at the quota");
    errors.Clear();
    // A throwing quota sink fails the layer; the mutation is still refused,
    // with the cause the layer latched.
    auto throwing = TfCreateRefPtr(new HostSceneProjection(base, std::make_shared<HostFailureLatch>(),
                                                            recordQuota(1)));
    throwing->Set(root, TfToken("a"), VtValue(1));
    throwing->endHostTurn();
    throwing->setQuotaSink([](size_t) -> bool { throw std::runtime_error("sink"); });
    require(rejected([&] { throwing->Set(root, TfToken("b"), VtValue(2)); }, OMNI_USD_LIVE_INTERNAL_ERROR) &&
            throwing->fault()->result().load() == OMNI_USD_LIVE_INTERNAL_ERROR &&
            throwing->pendingLocalEditCount() == 1 && throwing->Get(root, TfToken("b")).IsEmpty(),
            "a throwing quota sink escaped the mutator, applied the edit or kept the layer healthy");
    errors.Clear();
    // A sink that fails the layer while refusing (an off-host mutation)
    // refuses with the cause the layer latched.
    auto unsupported = TfCreateRefPtr(new HostSceneProjection(base, std::make_shared<HostFailureLatch>(),
                                                               recordQuota(1)));
    unsupported->Set(root, TfToken("a"), VtValue(1));
    unsupported->endHostTurn();
    unsupported->setQuotaSink([&](size_t) { unsupported->fault()->fail(OMNI_USD_LIVE_UNSUPPORTED); return true; });
    require(rejected([&] { unsupported->Set(root, TfToken("b"), VtValue(2)); }, OMNI_USD_LIVE_UNSUPPORTED) &&
            unsupported->pendingLocalEditCount() == 1 && unsupported->Get(root, TfToken("b")).IsEmpty(),
            "a refusal that failed the layer reported another cause or applied the edit");
    errors.Clear();
    // The sink decides: a layer its owner already failed keeps recording.
    auto exempt = TfCreateRefPtr(new HostSceneProjection(base, std::make_shared<HostFailureLatch>(),
                                                          recordQuota(1)));
    exempt->Set(root, TfToken("a"), VtValue(1));
    exempt->endHostTurn();
    size_t consulted = 0;
    exempt->setQuotaSink([&](size_t) { ++consulted; return false; });
    exempt->Set(root, TfToken("b"), VtValue(2));
    require(consulted == 1 && exempt->pendingLocalEditCount() == 2 && exempt->Get(root, TfToken("b")) == VtValue(2),
            "a quota sink that declined the refusal lost the edit");
    // An admission below the quota lets the next turn record.
    projection->acceptLocalEdits(2, base);
    projection->Set(root, TfToken("d"), VtValue(4));
    projection->Set(root, TfToken("e"), VtValue(5));
    require(projection->pendingLocalEditCount() == 3 && projection->Get(root, TfToken("d")) == VtValue(4),
            "an admission did not reopen the quota");
    projection->endHostTurn();
    // A FAILED layer only records; the quota never refuses its edits.
    fault->fail(OMNI_USD_LIVE_UNSUPPORTED);
    projection->Set(root, TfToken("f"), VtValue(6));
    require(projection->pendingLocalEditCount() == 4 && refused.size() == 4, "a failed layer refused an edit");
    // Projections a layer prepares inherit its quota.
    const auto adopted = TfCreateRefPtr(new HostSceneProjection(base, std::make_shared<HostFailureLatch>(),
                                                                recordQuota(2)));
    HostLayerState state(adopted);
    require(state.prepare(base)->limits().maxPendingBytes == recordQuota(2).maxPendingBytes, "a prepared projection lost the layer's quota");
    bool zero = false;
    try { (void)TfCreateRefPtr(new HostSceneProjection(base, std::make_shared<HostFailureLatch>(), HostSceneProjection::Limits{0})); }
    catch (const std::invalid_argument&) { zero = true; }
    require(zero, "a zero quota was accepted");
}
void failedLayerDiscardsRecords() {
    const auto base = buildSceneCandidate({}, encodeEnvelope({}));
    const auto root = SdfPath::AbsoluteRootPath();
    auto fault = std::make_shared<HostFailureLatch>();
    auto projection = TfCreateRefPtr(new HostSceneProjection(base, fault, recordQuota(2)));
    size_t notified = 0, consulted = 0;
    projection->setLocalEditSink([&](size_t) { ++notified; });
    projection->setQuotaSink([&](size_t) { ++consulted; return true; });
    projection->Set(root, TfToken("a"), VtValue(1));
    projection->Set(root, TfToken("b"), VtValue(2));
    projection->endHostTurn();
    require(projection->pendingLocalEditCount() == 2 && !projection->discardsLocalEdits(), "fixture did not fill the buffer");
    fault->fail(OMNI_USD_LIVE_UNSUPPORTED);
    projection->discardLocalEdits();
    require(!projection->hasPendingLocalEdits() && projection->discardsLocalEdits(), "discard kept a record");
    // Every later turn applies and notifies, at any value size, but retains
    // nothing and never meets the quota.
    const VtValue large(std::string(65536, 'x'));
    for (int turn = 0; turn < 64; ++turn) {
        projection->Set(root, TfToken("large"), large);
        projection->Set(root, TfToken("turn"), VtValue(turn));
        projection->endHostTurn();
    }
    require(projection->pendingLocalEditCount() == 0 && consulted == 0 && notified == 2 + 128,
            "a discarding projection retained a record, consulted the quota or skipped the sink");
    require(projection->Get(root, TfToken("turn")) == VtValue(63) && projection->Get(root, TfToken("large")) == large &&
            projection->Get(root, TfToken("a")) == VtValue(1), "a discarding projection lost host content");
    // A storage edit that throws removes no record it never appended.
    {
        TfErrorMark errors;
        bool typed = false;
        try {
            projection->SetTimeSample(SdfPath("/P.x"), std::numeric_limits<double>::quiet_NaN(), VtValue(1.0));
        } catch (const std::exception& error) {
            const auto* rejected = dynamic_cast<const LayerOperationRejected*>(&error);
            typed = rejected && rejected->result() == OMNI_USD_LIVE_INVALID_ARGUMENT;
        }
        require(typed, "NaN time sample is a typed INVALID_ARGUMENT rejection");
        require(!errors.IsClean(), "NaN time sample posts a USD runtime error");
        errors.Clear();
        require(projection->pendingLocalEditCount() == 0, "rejected NaN sample leaves no record");
    }
    projection->discardLocalEdits();
    require(projection->pendingLocalEditCount() == 0 && projection->discardsLocalEdits(), "a second discard changed the projection");
}
// A snapshot taken by a foreign abstract value while its write extracts it
// holds the storage as it was before that write.
void snapshotDuringForeignExtraction() {
    const auto root = SdfPath::AbsoluteRootPath();
    const TfToken field("documentation");
    struct Input final : SdfAbstractDataConstValue {
        std::function<void()> callback;
        explicit Input(std::function<void()> fn)
            : SdfAbstractDataConstValue(nullptr, typeid(void), false, typeid(void)), callback(std::move(fn)) {}
        bool GetValue(VtValue* output) const override { callback(); *output = VtValue(std::string("after")); return true; }
        bool IsEqual(const VtValue&) const override { return false; }
    };
    const auto storage = TfCreateRefPtr(new HostSceneStorage);
    storage->CreateSpec(root, SdfSpecTypePseudoRoot);
    storage->Set(root, field, VtValue(std::string("before")));
    TfRefPtr<HostSceneStorage> cloned;
    Input cloning([&] { if (!cloned) cloned = storage->clone(); });
    storage->Set(root, field, cloning);
    require(cloned && cloned->Get(root, field) == VtValue(std::string("before")) &&
            storage->Get(root, field) == VtValue(std::string("after")),
            "a clone taken during a foreign extraction observed the write");
    const auto base = buildSceneCandidate({}, encodeEnvelope({}));
    auto projection = TfCreateRefPtr(new HostSceneProjection(base, std::make_shared<HostFailureLatch>()));
    projection->Set(root, field, VtValue(std::string("before")));
    SdfAbstractDataConstRefPtr captured;
    Input capturing([&] { if (!captured) captured = projection->capture(); });
    projection->Set(root, field, capturing);
    require(captured && captured->Get(root, field) == VtValue(std::string("before")) &&
            projection->Get(root, field) == VtValue(std::string("after")),
            "a capture taken during a foreign extraction observed the write");
}
// The abstract-value overloads of a discarding projection apply, notify and
// retain nothing, like every forwarded mutation.
void discardingAbstractValueEdits() {
    const auto base = buildSceneCandidate({}, encodeEnvelope({}));
    const auto root = SdfPath::AbsoluteRootPath();
    auto fault = std::make_shared<HostFailureLatch>();
    auto projection = TfCreateRefPtr(new HostSceneProjection(base, fault, recordQuota(1)));
    size_t notified = 0, consulted = 0;
    projection->setLocalEditSink([&](size_t) { ++notified; });
    projection->setQuotaSink([&](size_t) { ++consulted; return true; });
    fault->fail(OMNI_USD_LIVE_UNSUPPORTED);
    projection->discardLocalEdits();
    const int number = 5; const SdfAbstractDataConstTypedValue<int> boxed(&number);
    projection->Set(root, TfToken("boxed"), boxed);
    projection->SetDictValueByKey(root, TfToken("customData"), TfToken("k"), boxed);
    const auto dictionary = projection->Get(root, TfToken("customData"));
    require(projection->pendingLocalEditCount() == 0 && consulted == 0 && notified == 2,
            "a discarding abstract-value edit retained a record, consulted the quota or skipped the sink");
    require(projection->Get(root, TfToken("boxed")) == VtValue(5) && dictionary.IsHolding<VtDictionary>() &&
            dictionary.UncheckedGet<VtDictionary>().GetValueAtPath("k") &&
            *dictionary.UncheckedGet<VtDictionary>().GetValueAtPath("k") == VtValue(5),
            "a discarding abstract-value edit lost host content");
}
void directAuthoringRejectsNestedMutation() {
    const auto base = buildSceneCandidate({}, encodeEnvelope({}));
    const auto root = SdfPath::AbsoluteRootPath();
    // An abstract value whose extraction attempts a nested mutation of the
    // projection it is written to. The nested call is rejected before any
    // change, so the outer call's record is the one it completes or removes.
    struct Input final : SdfAbstractDataConstValue {
        std::function<void()> callback;
        explicit Input(std::function<void()> fn)
            : SdfAbstractDataConstValue(nullptr, typeid(void), false, typeid(void)), callback(std::move(fn)) {}
        bool GetValue(VtValue* output) const override { callback(); *output = VtValue(std::string("outer")); return true; }
        bool IsEqual(const VtValue&) const override { return false; }
    };
    for (bool dictionary : {false, true}) {
        auto fault = std::make_shared<HostFailureLatch>();
        auto projection = TfCreateRefPtr(new HostSceneProjection(base, fault));
        projection->Set(root, TfToken("prior"), VtValue(1));
        unsigned busy = 0;
        Input input([&] {
            try { projection->Set(root, TfToken("nested"), VtValue(2)); }
            catch (const LayerOperationRejected& error) { require(error.result() == OMNI_USD_LIVE_BUSY, "wrong nested mutation result"); ++busy; }
        });
        if (dictionary) projection->SetDictValueByKey(root, SdfFieldKeys->CustomData, TfToken("outer"), input);
        else projection->Set(root, TfToken("outer"), input);
        const auto edits = projection->pendingLocalEdits();
        const auto& last = edits.operations->back();
        // Storage and the record each extract the value once, and each
        // extraction's nested mutation is rejected.
        require(busy == 2 && fault->result().load() == OMNI_USD_LIVE_OK, "a nested mutation was not rejected");
        require(edits.operations->size() == 2 && projection->Get(root, TfToken("nested")).IsEmpty() &&
                last.kind == (dictionary ? SdfOperationKind::SetDictKey : SdfOperationKind::SetField) &&
                (dictionary ? last.key : last.field) == TfToken("outer") && last.value == VtValue(std::string("outer")),
                "a nested mutation changed storage or displaced the outer record");
        // A nested mutation that escapes the storage edit's extraction fails the
        // outer call before storage changes and removes only the outer record.
        Input escaping([&] { projection->Set(root, TfToken("nested"), VtValue(2)); });
        bool escaped = false;
        try {
            if (dictionary) projection->SetDictValueByKey(root, SdfFieldKeys->CustomData, TfToken("escaping"), escaping);
            else projection->Set(root, TfToken("escaping"), escaping);
        } catch (const LayerOperationRejected& error) { escaped = error.result() == OMNI_USD_LIVE_BUSY; }
        require(escaped && projection->pendingLocalEditCount() == 2 &&
                (*projection->pendingLocalEdits().operations)[0].field == TfToken("prior") &&
                projection->Get(root, TfToken("nested")).IsEmpty() && projection->Get(root, TfToken("escaping")).IsEmpty() &&
                fault->result().load() == OMNI_USD_LIVE_OK,
                "an escaping nested mutation removed the wrong record or changed storage");
    }
}

void concurrentReadersLeaveNoNestingState() {
    // USD composition reads a layer's backing from worker threads. Readers that
    // overlap and retire out of order leave no nesting state behind, so the
    // next host mutation is not misread as nested in a forwarded call.
    const SdfPath path("/P");
    const TfToken field("documentation");
    const auto projection = TfCreateRefPtr(new HostSceneProjection(baseline(), std::make_shared<HostFailureLatch>()));
    struct Visitor final : SdfAbstractDataSpecVisitor {
        std::function<void()> callback;
        explicit Visitor(std::function<void()> fn) : callback(std::move(fn)) {}
        bool VisitSpec(const SdfAbstractData&, const SdfPath&) override { callback(); return false; }
        void Done(const SdfAbstractData&) override {}
    };
    std::mutex mutex; std::condition_variable changed;
    unsigned entered = 0, released = 0;
    auto traverse = [&](unsigned bit) {
        Visitor pending([&] {
            std::unique_lock<std::mutex> lock(mutex); entered |= bit; changed.notify_all();
            changed.wait(lock, [&] { return (released & bit) != 0; });
        });
        projection->VisitSpecs(&pending);
    };
    // The first reader retires while the second is still inside its call.
    std::thread first(traverse, 1);
    { std::unique_lock<std::mutex> lock(mutex); changed.wait(lock, [&] { return entered == 1; }); }
    std::thread second(traverse, 2);
    { std::unique_lock<std::mutex> lock(mutex); changed.wait(lock, [&] { return entered == 3; }); released = 1; changed.notify_all(); }
    first.join();
    { std::lock_guard<std::mutex> lock(mutex); released = 3; changed.notify_all(); }
    second.join();
    bool busy = false;
    try { projection->Set(path, field, VtValue(std::string("after both readers"))); }
    catch (const LayerOperationRejected& error) { busy = error.result() == OMNI_USD_LIVE_BUSY; }
    require(!busy && projection->Get(path, field) == VtValue(std::string("after both readers")) &&
            projection->pendingLocalEditCount() == 1,
            "overlapping readers left stale nesting state that refused a host mutation");
}
// A foreign value whose extraction fails is rejected before storage changes,
// as a typed UNSUPPORTED exception with a USD runtime error and no record.
void abstractValueExtractionFailureIsTyped() {
    const SdfPath path("/P");
    const TfToken field("documentation");
    struct Input final : SdfAbstractDataConstValue {
        Input() : SdfAbstractDataConstValue(nullptr, typeid(void), false, typeid(void)) {}
        bool GetValue(VtValue* output) const override { *output = VtValue(std::string("input")); return false; }
        bool IsEqual(const VtValue&) const override { return false; }
    } input;
    for (bool dictionary : {false, true}) {
        auto fault = std::make_shared<HostFailureLatch>();
        const auto actual = TfCreateRefPtr(new HostSceneProjection(baseline(), fault));
        const auto before = inspect(*actual);
        TfErrorMark errors;
        OmniUsdLiveResult rejected = OMNI_USD_LIVE_OK;
        try {
            if (dictionary) actual->SetDictValueByKey(path, field, TfToken("k"), input);
            else actual->Set(path, field, input);
        } catch (const std::exception& error) {
            const auto* typed = dynamic_cast<const LayerOperationRejected*>(&error);
            rejected = typed ? typed->result() : OMNI_USD_LIVE_INTERNAL_ERROR;
        }
        require(rejected == OMNI_USD_LIVE_UNSUPPORTED, "failed foreign extraction was not a typed UNSUPPORTED rejection");
        require(!errors.IsClean(), "failed foreign extraction posted no USD runtime error");
        errors.Clear();
        require(inspect(*actual) == before && actual->pendingLocalEditCount() == 0,
                "failed foreign extraction changed storage or kept a record");
        require(fault->result().load() == OMNI_USD_LIVE_OK, "failed foreign extraction latched a fault");
    }
}
// A reader that enters storage after the projection's nesting check (here,
// while the quota sink runs) holds the storage write guard, so storage refuses
// the mutation. The refusal is a typed BUSY rejection with a USD runtime error.
void mutationRefusedByStorageGuardIsTyped() {
    auto fault = std::make_shared<HostFailureLatch>();
    const auto projection =
        TfCreateRefPtr(new HostSceneProjection(baseline(), fault, recordQuota(1)));
    projection->Set(SdfPath("/P"), TfToken("documentation"), VtValue(std::string("fill")));
    projection->endHostTurn();
    const auto before = inspect(*projection);
    struct Visitor final : SdfAbstractDataSpecVisitor {
        std::function<void()> callback;
        explicit Visitor(std::function<void()> fn) : callback(std::move(fn)) {}
        bool VisitSpec(const SdfAbstractData&, const SdfPath&) override { callback(); return false; }
        void Done(const SdfAbstractData&) override {}
    };
    std::mutex mutex; std::condition_variable changed;
    bool start = false, entered = false, release = false;
    std::thread reader([&] {
        {
            std::unique_lock<std::mutex> lock(mutex);
            changed.wait(lock, [&] { return start; });
            if (release) return;
        }
        Visitor pending([&] {
            std::unique_lock<std::mutex> lock(mutex); entered = true; changed.notify_all();
            changed.wait(lock, [&] { return release; });
        });
        try { projection->VisitSpecs(&pending); } catch (...) {}
        std::lock_guard<std::mutex> lock(mutex); entered = true; changed.notify_all();
    });
    // Every exit path releases the reader and joins it.
    struct Join {
        std::thread& thread; std::mutex& mutex; std::condition_variable& changed; bool& start; bool& release;
        ~Join() {
            { std::lock_guard<std::mutex> lock(mutex); start = release = true; changed.notify_all(); }
            thread.join();
        }
    } join{reader, mutex, changed, start, release};
    size_t consulted = 0;
    projection->setQuotaSink([&](size_t) {
        ++consulted;
        std::unique_lock<std::mutex> lock(mutex); start = true; changed.notify_all();
        changed.wait(lock, [&] { return entered; });
        return false;
    });
    TfErrorMark errors;
    OmniUsdLiveResult rejected = OMNI_USD_LIVE_OK;
    try { projection->CreateSpec(SdfPath("/Q"), SdfSpecTypePrim); }
    catch (const std::exception& error) {
        const auto* typed = dynamic_cast<const LayerOperationRejected*>(&error);
        rejected = typed ? typed->result() : OMNI_USD_LIVE_INTERNAL_ERROR;
    } catch (...) { rejected = OMNI_USD_LIVE_INTERNAL_ERROR; }
    const bool posted = !errors.IsClean();
    errors.Clear();
    { std::lock_guard<std::mutex> lock(mutex); release = true; changed.notify_all(); }
    require(consulted == 1, "the mutation did not consult the quota sink");
    require(rejected == OMNI_USD_LIVE_BUSY, "a mutation refused by the storage guard was not a typed BUSY rejection");
    require(posted, "a mutation refused by the storage guard posted no USD runtime error");
    require(inspect(*projection) == before && projection->pendingLocalEditCount() == 1,
            "a mutation refused by the storage guard changed storage or kept its record");
    require(fault->result().load() == OMNI_USD_LIVE_OK, "a mutation refused by the storage guard latched a fault");
}
}
void hostProjectionTests() {
    TfErrorMark mark;
    rawMutators(); ownedStorageCallbacks(); ownedStorageRetirement(); callbackOwnerLifetimes(); replacementControls(); publicLayerOperations();
    directAuthoringRecordsOperations();
    directAuthoringQuota();
    bufferBytesAndBudget();
    recordBytesOfLargeKinds();
    unchargedBufferKeepsItsQuota();
    failedLayerDiscardsRecords();
    discardingAbstractValueEdits();
    snapshotDuringForeignExtraction();
    directAuthoringRejectsNestedMutation();
    concurrentReadersLeaveNoNestingState();
    abstractValueExtractionFailureIsTyped();
    mutationRefusedByStorageGuardIsTyped();
    require(mark.IsClean(), "unexpected USD diagnostic");
    std::cout << "host projection: stock mutation/namespace/capture isolation passed (not production containment)\n";
}
