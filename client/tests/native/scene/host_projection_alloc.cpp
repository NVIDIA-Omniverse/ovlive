// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "HostSceneProjection.h"
#include "HostSceneStorage.h"
#include "HostLayerState.h"
#include "data_equality.h"
#include "UsdValueCodec.h"
#include <pxr/base/tf/errorMark.h>
#include <pxr/base/vt/dictionary.h>
#include <pxr/usd/sdf/fileFormat.h>
#include <pxr/usd/sdf/layer.h>
#include <pxr/usd/sdf/namespaceEdit.h>
#include <pxr/usd/sdf/primSpec.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usd/attribute.h>
#include <iomanip>
#include <functional>
#if defined(_WIN32)
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    include <windows.h>
#else
#    include <dlfcn.h>
#endif
#include <pxr/usd/sdf/schema.h>
#include <cstdlib>
#include <iostream>
#include <new>
#include <stdexcept>

// One-shot, calling-thread-only failures. Other wire-scene tests and SDK worker
// threads never enable this switch. Only ordinary C++ new allocations observed
// by this executable are covered, not all allocators inside the external SDK.
namespace {
thread_local long failAllocation = -1;
thread_local bool allocationFailed = false;
}
// ThreadSanitizer builds keep the standard allocator: the tsan profile's
// called_from_lib suppressions ignore allocations only when the prebuilt USD
// libraries call the allocator directly, and a replacement defined in this
// executable would become the caller of every USD allocation.
#if defined(__SANITIZE_THREAD__)
#define LIVE_TEST_ALLOCATOR 0
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define LIVE_TEST_ALLOCATOR 0
#endif
#endif
#ifndef LIVE_TEST_ALLOCATOR
#define LIVE_TEST_ALLOCATOR 1
#endif
#if LIVE_TEST_ALLOCATOR
void* operator new(std::size_t bytes) {
    if (failAllocation >= 0 && failAllocation-- == 0) {
        failAllocation = -1; // Unwinding/diagnostics may allocate normally.
        allocationFailed = true;
        throw std::bad_alloc();
    }
    if (auto* value = std::malloc(bytes ? bytes : 1)) return value;
    throw std::bad_alloc();
}
void* operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete(void* value) noexcept { std::free(value); }
void operator delete[](void* value) noexcept { std::free(value); }
void operator delete(void* value, std::size_t) noexcept { std::free(value); }
void operator delete[](void* value, std::size_t) noexcept { std::free(value); }
#endif

PXR_NAMESPACE_USING_DIRECTIVE
using namespace omni::usd::live;
using namespace omni::usd::live::wire;
namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(std::string("host projection allocation: ") + message);
}
// The file name of the loaded library that holds `address`, or empty when none does.
std::string loadedLibrary(const void* address) {
#if defined(_WIN32)
    HMODULE module=nullptr;
    char name[MAX_PATH]{};
    if(!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           static_cast<LPCWSTR>(address),&module))
        return {};
    const DWORD length=GetModuleFileNameA(module,name,MAX_PATH);
    return length && length<MAX_PATH ? std::string(name,length) : std::string{};
#else
    Dl_info loaded{};
    return dladdr(address,&loaded) && loaded.dli_fname ? std::string(loaded.dli_fname) : std::string{};
#endif
}
struct Fail {
    explicit Fail(long index) { allocationFailed = false; failAllocation = index; }
    ~Fail() { failAllocation = -1; }
};
WireScenePtr seed() {
    EnvelopeInput input;
    DecodedValues values;
    for (uint64_t index = 2; index != 10; ++index) {
        Command c{}; c.kind = Command::Kind::Create; c.section = index; c.parent = 1;
        c.name = "Prim" + std::to_string(index); c.sectionType = SdfSpecTypePrim;
        input.commands.push_back(c);
        input.fields.push_back({index, 0, "specifier", {}}); values.fields.emplace_back(SdfSpecifierDef);
        input.fields.push_back({index, 0, "documentation", {}}); values.fields.emplace_back(std::string(128, char('a' + index)));
    }
    return buildSceneCandidate({}, encodeUsdEnvelope(input, values));
}
}
void hostProjectionAllocationTests() {
#if !LIVE_TEST_ALLOCATOR
    // Every case injects through the replacement allocator; the thread
    // sanitizer build has none, and these single-threaded cases add no
    // concurrency for it to check.
    std::cout << "host projection allocation: skipped without the test allocator\n";
    return;
#endif
    // Standalone owned-storage proof before changing production projection data.
    // Keep every stock SDK object outside the injected/destruction scope.
    const auto storageSeed=seed();
    const auto input=TfCreateRefPtr(new HostSceneStorage); input->CopyFrom(storageSeed->data());
    const SdfPath storagePath("/Prim2"); const TfToken storageField("documentation");
    input->Set(storagePath,storageField,VtValue(std::string(2048,'n')));
    input->SetDictValueByKey(storagePath,TfToken("customData"),TfToken("nested:value"),VtValue(std::string(256,'v')));
    input->SetTimeSample(storagePath,1.0,VtValue(7.0));
    input->SetTimeSample(storagePath,2.0,VtValue(9.0));
    const auto inputSnapshot=copyData(input);
    const auto overlayOracle=TfCreateRefPtr(new SdfData); overlayOracle->CopyFrom(storageSeed->data());
    overlayOracle->CreateSpec(SdfPath("/DestinationOnly"),SdfSpecTypePrim);
    overlayOracle->Set(storagePath,TfToken("destinationOnly"),VtValue(17));
    overlayOracle->CopyFrom(input);
    const auto overlayExpected=copyData(overlayOracle);
    unsigned copyFailures=0; bool copied=false;
    for(long index=0;index<4096;++index) {
        auto owned=TfCreateRefPtr(new HostSceneStorage); owned->CopyFrom(storageSeed->data());
        owned->CreateSpec(SdfPath("/DestinationOnly"),SdfSpecTypePrim);
        owned->Set(storagePath,TfToken("destinationOnly"),VtValue(17));
        const auto before=copyData(owned);
        bool caught=false;
        try { Fail fail(index); owned->CopyFrom(input); }
        catch(const std::bad_alloc&) { caught=true; }
        require(sameData(input,inputSnapshot),"owned overlay changed its source");
        if(caught) {
            require(allocationFailed && sameData(owned,before),"owned overlay partially committed");
            ++copyFailures;
        } else {
            require(!allocationFailed && sameData(owned,overlayExpected),
                    "owned overlay lost source or destination-only content");
            copied=true;
        }
        {
            // Both failed and completed candidates must retire without new.
            Fail fail(0); owned.Reset(); if(allocationFailed) std::abort();
        }
        if(copied) break;
    }
    require(copied && copyFailures>2,"owned overlay allocation sweep incomplete");
    {
        // No source retains these payloads: this is final-value destruction,
        // not just dropping another reference to values held by the input.
        auto standalone=TfCreateRefPtr(new HostSceneStorage);
        standalone->CreateSpec(storagePath,SdfSpecTypePrim);
        standalone->Set(storagePath,storageField,VtValue(std::string(8192,'s')));
        standalone->SetDictValueByKey(storagePath,TfToken("customData"),TfToken("nested:value"),VtValue(std::string(4096,'d')));
        standalone->SetTimeSample(storagePath,1.0,VtValue(std::string(4096,'t')));
        standalone->Set(storagePath,TfToken("tokens"),VtValue(std::vector<TfToken>{TfToken("first"),TfToken("second")}));
        Fail fail(0); standalone.Reset(); if(allocationFailed) std::abort();
    }
    std::cout << "owned storage overlay allocation failures=" << copyFailures
              << " synchronous retirement completed (ordinary C++ new only)\n";
    const VtValue mutationValue(std::string(2048,'m'));
    const SdfAbstractDataConstTypedValue<VtValue> boxedMutation(&mutationValue);
    const TfToken custom("customData"),nested("nested:value"),ownedNewField("newField");
    const SdfPath moved("/Moved"),created("/Created");
    const std::vector<std::pair<const char*,std::function<void(SdfAbstractData&)>>> mutations={
        {"set",[&](auto& d) { d.Set(storagePath,storageField,mutationValue); }},
        {"new-field",[&](auto& d) { d.Set(storagePath,ownedNewField,mutationValue); }},
        {"abstract-input",[&](auto& d) { d.Set(storagePath,storageField,boxedMutation); }},
        {"erase-field",[&](auto& d) { d.Erase(storagePath,storageField); }},
        {"dictionary",[&](auto& d) { d.SetDictValueByKey(storagePath,custom,nested,mutationValue); }},
        {"erase-dictionary",[&](auto& d) { d.EraseDictValueByKey(storagePath,custom,nested); }},
        {"sample",[&](auto& d) { d.SetTimeSample(storagePath,3.0,mutationValue); }},
        {"erase-sample",[&](auto& d) { d.EraseTimeSample(storagePath,1.0); }},
        {"existing-spec",[&](auto& d) { d.CreateSpec(storagePath,SdfSpecTypePrim); }},
        {"new-spec",[&](auto& d) { d.CreateSpec(created,SdfSpecTypePrim); }},
        {"move",[&](auto& d) { d.MoveSpec(storagePath,moved); }},
        {"erase-spec",[&](auto& d) { d.EraseSpec(storagePath); }}
    };
    for(const auto& operation:mutations) {
        const auto oracle=TfCreateRefPtr(new SdfData); oracle->CopyFrom(input); operation.second(*oracle);
        const auto expected=copyData(oracle);
        unsigned failures=0; bool complete=false;
        for(long index=0;index<4096;++index) {
            auto owned=TfCreateRefPtr(new HostSceneStorage); owned->CopyFrom(input);
            bool caught=false;
            try { Fail fail(index); operation.second(*owned); } catch(const std::bad_alloc&) { caught=true; }
            if(caught) {
                require(allocationFailed && sameData(owned,inputSnapshot),"owned mutator partially committed"); ++failures;
            } else {
                require(!allocationFailed && sameData(owned,expected),"owned mutator differs from stock after success"); complete=true;
            }
            { Fail fail(0); owned.Reset(); if(allocationFailed) std::abort(); }
            if(complete) break;
        }
        require(complete,"owned mutator allocation sweep incomplete");
        std::cout << "owned storage " << operation.first << " allocation failures=" << failures << '\n';
    }
    const auto baseline = seed();
    const auto fault = std::make_shared<HostFailureLatch>();
    const auto stable = TfCreateRefPtr(new HostSceneProjection(baseline, fault));
    const auto path = SdfPath("/Prim2");
    const auto key = SdfFieldKeys->Documentation;
    const auto original = baseline->data()->Get(path, key);
    auto captured = stable->capture(); // Prepare lazy SDK bookkeeping before the sweep.
    const auto previous = captured;
    const auto baselineCopy = TfCreateRefPtr(new SdfData);
    baselineCopy->CopyFrom(baseline->data());
    unsigned constructionFailures = 0, captureFailures = 0, mutationFailures = 0;
    bool constructionComplete = false, captureComplete = false, mutationComplete = false;
    for (long index = 0; index != 1024; ++index) {
        TfRefPtr<HostSceneProjection> candidate;
        bool caught = false;
        try { Fail fail(index); candidate = TfCreateRefPtr(new HostSceneProjection(baseline, fault)); }
        catch (const std::bad_alloc&) { caught = true; }
        require(fault->result().load() == OMNI_USD_LIVE_OK, "construction altered mutation latch");
        require(baseline->data()->Get(path, key) == original && baseline->sectionCount() == 9,
                "failed construction changed immutable baseline");
        if (caught) {
            require(allocationFailed && !candidate, "partially constructed owner escaped");
            ++constructionFailures;
        } else {
            require(!allocationFailed && candidate && candidate->Get(path, key) == original, "construction swallowed allocation failure");
            constructionComplete = true; break;
        }
    }
    for (long index = 0; index != 1024; ++index) {
        bool caught = false;
        try { Fail fail(index); captured = stable->capture(); }
        catch (const std::bad_alloc&) { caught = true; }
        require(stable->Get(path, key) == original && baseline->data()->Get(path, key) == original,
                "capture changed retained host or baseline");
        require(fault->result().load() == OMNI_USD_LIVE_OK, "capture latched host mutation");
        if (caught) {
            require(allocationFailed && captured == previous, "failed capture replaced caller output");
            ++captureFailures;
        } else {
            require(!allocationFailed && captured != previous && captured->Get(path, key) == original,
                    "capture swallowed allocation failure or reused mutable owner");
            captureComplete = true; break;
        }
    }
    require(constructionComplete && captureComplete && constructionFailures > 2 && captureFailures > 1,
            "allocation sweep did not reach copying failures and successful completion");
    // A forwarded write that throws in storage leaves it unchanged and records
    // nothing. A plain write's record is appended before storage changes, so
    // it never applies unrecorded. An abstract-value write that applies but
    // cannot be re-extracted fails the layer with a typed cause. Modes: plain
    // Set, abstract-value Set, abstract-value SetDictValueByKey. This does not
    // assert safety of throwing through a public SdfLayer compound operation.
    const TfToken newField("hostAllocationProbe");
    const std::string replacementText(512, 'z');
    const VtValue replacement(replacementText);
    const SdfAbstractDataConstTypedValue<std::string> boxedReplacement(&replacementText);
    unsigned unrecorded[3] = {0, 0, 0};
    for (int mode : {0, 1, 2}) {
        mutationComplete = false;
        for (long index = 0; index != 1024; ++index) {
            auto mutationFault = std::make_shared<HostFailureLatch>();
            const auto candidate = TfCreateRefPtr(new HostSceneProjection(baseline, mutationFault));
            bool caught = false;
            try {
                Fail fail(index);
                if (mode == 2) candidate->SetDictValueByKey(path, SdfFieldKeys->CustomData, TfToken("probe"), boxedReplacement);
                else if (mode == 1) candidate->Set(path, newField, boxedReplacement);
                else candidate->Set(path, newField, replacement);
            }
            catch (const std::bad_alloc&) { caught = true; }
            const auto dictionary = candidate->Get(path, SdfFieldKeys->CustomData);
            const auto keyed = [&] {
                if (!dictionary.IsHolding<VtDictionary>()) return false;
                const auto& entries = dictionary.UncheckedGet<VtDictionary>();
                const auto entry = entries.find("probe");
                return entry != entries.end() && entry->second == replacement;
            };
            const bool applied = mode == 2 ? keyed() : candidate->Get(path, newField) == replacement;
            const auto cause = mutationFault->result().load();
            const bool failed = cause != OMNI_USD_LIVE_OK;
            require(!failed || cause == OMNI_USD_LIVE_RESOURCE_EXHAUSTED, "unrecordable write latched an untyped cause");
            require(candidate->hasPendingLocalEdits() ? applied && !failed : !caught || applied == failed,
                    "forwarded write lost an applied edit without failing the layer");
            require(baseline->data()->Equals(baselineCopy), "forwarded write changed immutable baseline");
            // Copy and traverse every surviving spec/field after injection is off.
            const auto recovered = candidate->capture();
            require(recovered->Equals(candidate), "failed write backing cannot be captured exactly");
            if (caught) {
                require(allocationFailed, "write failed without injection");
                ++mutationFailures;
                if (applied) ++unrecorded[mode];
            } else {
                require(!allocationFailed && applied, "write swallowed allocation failure or lost value");
                mutationComplete = true; break;
            }
        }
        require(mutationComplete, "forwarded write sweep did not complete");
    }
    // The abstract-value overloads re-extract after storage applies the edit.
    require(unrecorded[1] > 0 && unrecorded[2] > 0, "applied-but-unrecorded abstract write failure not reached");
    require(unrecorded[0] == 0, "a plain write applied without its record");
    require(mutationComplete && mutationFailures > 0, "forwarded write allocation failure not reached");
    HostLayerState state(baseline);
    const auto authorized = state.authorized();
    unsigned publicationPreparationFailures = 0;
    bool prepared = false;
    TfRefPtr<HostSceneProjection> pending;
    for (long index = 0; index != 1024; ++index) {
        bool caught = false;
        try { Fail fail(index); pending = state.prepare(baseline); }
        catch (const std::bad_alloc&) { caught = true; }
        require(state.authorized() == authorized && state.observed() == authorized && !state.installing() &&
                state.failure() == HostLayerState::Failure::None && !state.publicationFailed() &&
                state.fault()->result().load() == OMNI_USD_LIVE_OK,
                "failed publication preparation changed layer state");
        if (caught) {
            require(allocationFailed && !pending, "partial publication candidate escaped");
            ++publicationPreparationFailures;
        } else {
            require(!allocationFailed && pending && pending != authorized, "publication preparation swallowed failure");
            prepared = true; break;
        }
    }
    require(prepared && publicationPreparationFailures > 2, "publication preparation sweep incomplete");
    // Prepared-owner transitions must not allocate, even when the next
    // allocation would fail. No USD mutation or notice is performed here.
    {
        Fail fail(0);
        if (state.begin(authorized, pending) != HostLayerState::Admission::Accepted ||
            !state.observe(pending) || !state.finish(pending) || allocationFailed) std::abort();
    }
    HostLayerState active(baseline);
    const auto activeOld = active.authorized(), admitted = active.prepare(baseline);
    require(active.begin(activeOld, admitted) == HostLayerState::Admission::Accepted && active.observe(admitted),
            "active preparation control did not install candidate");
    unsigned activePreparationFailures = 0;
    bool activePrepared = false;
    for (long index = 0; index != 1024; ++index) {
        TfRefPtr<HostSceneProjection> extra;
        bool caught = false;
        try { Fail fail(index); extra = active.prepare(baseline); }
        catch (const std::bad_alloc&) { caught = true; }
        require(active.authorized() == activeOld && active.candidate() == admitted && active.observed() == admitted &&
                active.failure() == HostLayerState::Failure::None && !active.publicationFailed(),
                "candidate preparation changed active installation");
        if (caught) {
            require(allocationFailed && !extra, "partial active candidate escaped");
            ++activePreparationFailures;
        } else {
            require(!allocationFailed && extra && extra != admitted, "active preparation swallowed failure");
            activePrepared = true; break;
        }
    }
    require(activePrepared && activePreparationFailures > 2, "active preparation sweep incomplete");
    require(!active.observe(activeOld) && active.failure() == HostLayerState::Failure::BackingReplaced,
            "preparation forgot already-observed candidate and allowed rollback");
    // The control latch itself must work even when the next allocation fails.
    {
        Fail fail(0);
        fault->fail(OMNI_USD_LIVE_UNSUPPORTED);
        fault->fail(OMNI_USD_LIVE_INTERNAL_ERROR);
        if (fault->result().load() != OMNI_USD_LIVE_UNSUPPORTED || allocationFailed) std::abort();
    }
    std::cout << "host projection allocation failures: preparation=" << constructionFailures
              << " capture=" << captureFailures << " mutation=" << mutationFailures << " unrecorded=" << unrecorded[1] << "/" << unrecorded[2]
              << " publication-preparation=" << publicationPreparationFailures
              << " active-preparation=" << activePreparationFailures
              << " (observed ordinary C++ new only)\n";
}


// Isolated diagnostic entry only, never called by the ordinary component pass.
// A supervisor records every allocation index and child exit before continuing.
int hostAllocationProbe(const std::string& backing, const std::string& operation, long index) {
    require(backing=="stock" || backing=="projection", "unknown diagnostic backing");
    require(operation=="move" || operation=="create" || operation=="field" || operation=="dictionary" || operation=="sample",
            "unknown diagnostic operation");
    require(index>=-1 && index<4096,"invalid allocation index");
    require(LIVE_TEST_ALLOCATOR || index==-1,"allocation injection needs the test allocator");
    auto baseline=seed();
    EnvelopeInput extra; DecodedValues values;
    Command child{}; child.kind=Command::Kind::Create; child.section=10; child.parent=2;
    child.name="Child"; child.sectionType=SdfSpecTypePrim; extra.commands.push_back(child);
    Command attribute{}; attribute.kind=Command::Kind::Create; attribute.section=11; attribute.parent=10;
    attribute.name="value"; attribute.sectionType=SdfSpecTypeAttribute; extra.commands.push_back(attribute);
    extra.fields.push_back({10,0,"specifier",{}}); values.fields.emplace_back(SdfSpecifierDef);
    extra.fields.push_back({11,0,"typeName",{}}); values.fields.emplace_back(TfToken("double"));
    extra.fields.push_back({11,0,"default",{}}); values.fields.emplace_back(7.0);
    baseline=buildSceneCandidate(baseline,encodeUsdEnvelope(extra,values));
    const auto original=copyData(baseline->data());
    auto fault=std::make_shared<HostFailureLatch>();
    SdfAbstractDataRefPtr data;
    if(backing=="stock") { data=TfCreateRefPtr(new SdfData); data->CopyFrom(baseline->data()); }
    else data=TfCreateRefPtr(new HostSceneProjection(baseline,fault));
    struct Access : SdfFileFormat {
        static void install(SdfLayer* layer,SdfAbstractDataRefPtr data) { _SetLayerData(layer,data); }
    };
    const auto layer=SdfLayer::CreateAnonymous("allocation-probe.usda");
    require(bool(layer),"diagnostic layer unavailable");
    Access::install(layer.operator->(),data);
    const auto stage=UsdStage::Open(layer);
    require(bool(stage),"diagnostic stage unavailable");
    const SdfPath path("/Prim2"),childPath("/Prim2/Child"),samplePath("/Prim2/Child.value");
    const auto retained=layer->GetPrimAtPath(path);
    const auto retainedChild=layer->GetPrimAtPath(childPath);
    const auto retainedStage=stage->GetPrimAtPath(childPath);
    require(retained && retainedChild && retainedStage,"diagnostic retained handles unavailable");
    SdfBatchNamespaceEdit edit; edit.Add(path,SdfPath("/Moved"));
    const VtValue fieldValue(std::string(512,'z')),sampleValue(11.0);
    const TfToken field("documentation"),dict("customData"),key("nested:value");
    const auto function=static_cast<SdfLayerRefPtr (*)(const std::string&,const SdfLayer::FileFormatArguments&)>(&SdfLayer::FindOrOpen);
    const std::string library=loadedLibrary(reinterpret_cast<const void*>(function));
    require(!library.empty(),"loaded SDK unavailable");
    std::cout << std::boolalpha << "{\"phase\":\"armed\",\"backing\":" << std::quoted(backing)
              << ",\"operation\":" << std::quoted(operation) << ",\"index\":" << index
              << ",\"usd_version\":" << PXR_VERSION << ",\"sdf_library\":" << std::quoted(library) << "}" << std::endl;
    TfErrorMark errors;
    bool caught=false,returned=false;
    try {
        Fail fail(index);
        if(operation=="move") returned=layer->Apply(edit);
        else if(operation=="create") returned=bool(SdfPrimSpec::New(retained,"Added",SdfSpecifierDef));
        else if(operation=="field") { layer->SetField(path,field,fieldValue); returned=true; }
        else if(operation=="dictionary") { layer->SetFieldDictValueByKey(path,dict,key,fieldValue); returned=true; }
        else { layer->SetTimeSample(samplePath,1.0,sampleValue); returned=true; }
    } catch(const std::bad_alloc&) { caught=true; }
    const bool injected=allocationFailed;
    std::cout << "{\"phase\":\"returned\",\"injected\":" << injected << ",\"caught_bad_alloc\":" << caught
              << ",\"returned_success\":" << returned << ",\"failed\":"
              << (fault->result().load()!=OMNI_USD_LIVE_OK) << "}" << std::endl;
    // Injection is off from here on. Inspect backing and USD identity separately; either
    // traversal or final teardown can still expose corruption in this child.
    const auto actual=copyData(data);
    const auto dataPath=retained->GetPath(),dataChild=retainedChild->GetPath();
    const bool coherent=data->HasSpec(dataPath) && data->HasSpec(dataChild) &&
        layer->GetPrimAtPath(dataPath)==retained && layer->GetPrimAtPath(dataChild)==retainedChild;
    const bool baselineUnchanged=sameData(baseline->data(),original);
    bool stageCoherent=true;
    struct Visit : SdfAbstractDataSpecVisitor {
        const UsdStageRefPtr& stage; bool& coherent;
        Visit(const UsdStageRefPtr& s,bool& c):stage(s),coherent(c) {}
        bool VisitSpec(const SdfAbstractData& d,const SdfPath& p) override {
            if(d.GetSpecType(p)==SdfSpecTypePrim && !stage->GetPrimAtPath(p)) coherent=false;
            return true;
        }
        void Done(const SdfAbstractData&) override {}
    } visit(stage,stageCoherent);
    data->VisitSpecs(&visit);
    for(const auto& prim:stage->Traverse()) if(!data->HasSpec(prim.GetPath())) stageCoherent=false;
    bool valuesCoherent=true;
    if(operation=="field") {
        const auto authored=data->Get(path,field);
        // USD resolves absent documentation to an empty string.
        valuesCoherent=VtValue(stage->GetPrimAtPath(path).GetDocumentation())==
            (authored.IsEmpty()?VtValue(std::string()):authored);
    }
    else if(operation=="dictionary") valuesCoherent=stage->GetPrimAtPath(path).GetCustomDataByKey(key)==data->GetDictValueByKey(path,dict,key);
    else if(operation=="sample") {
        VtValue expected; if(!data->QueryTimeSample(samplePath,1.0,&expected)) expected=data->Get(samplePath,SdfFieldKeys->Default);
        double value=0;
        valuesCoherent=stage->GetAttributeAtPath(samplePath).Get(&value,UsdTimeCode(1.0)) && VtValue(value)==expected;
    } else {
        // Namespace edits must also preserve the existing attribute's value.
        const auto valuePath=dataChild.AppendProperty(TfToken("value"));
        const auto attribute=stage->GetAttributeAtPath(valuePath);
        double value=0;
        valuesCoherent=attribute && attribute.Get(&value) && value==7.0 &&
            data->Get(valuePath,SdfFieldKeys->Default)==VtValue(value);
    }
    bool effect=false;
    if(operation=="move") effect=!data->HasSpec(path) && !data->HasSpec(childPath) &&
        data->HasSpec(SdfPath("/Moved")) && data->HasSpec(SdfPath("/Moved/Child.value")) &&
        dataPath==SdfPath("/Moved") && dataChild==SdfPath("/Moved/Child");
    else if(operation=="create") effect=data->GetSpecType(SdfPath("/Prim2/Added"))==SdfSpecTypePrim &&
        data->Get(SdfPath("/Prim2/Added"),SdfFieldKeys->Specifier)==VtValue(SdfSpecifierDef);
    else if(operation=="field") effect=data->Get(path,field)==fieldValue;
    else if(operation=="dictionary") effect=data->GetDictValueByKey(path,dict,key)==fieldValue;
    else { VtValue value; effect=data->QueryTimeSample(samplePath,1.0,&value) && value==sampleValue; }
    std::cout << "{\"phase\":\"observed\",\"baseline_unchanged\":" << baselineUnchanged
              << ",\"expected_effect\":" << effect << ",\"data_unchanged\":" << sameData(actual,original) << ",\"retained_coherent\":" << coherent
              << ",\"stage_namespace_coherent\":" << stageCoherent << ",\"stage_values_coherent\":" << valuesCoherent << ",\"sdk_diagnostics\":" << !errors.IsClean()
              << ",\"retained_path\":" << std::quoted(dataPath.GetString())
              << ",\"retained_child_path\":" << std::quoted(dataChild.GetString()) << "}" << std::endl;
    errors.Clear();
    return 0;
}
