// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "FieldEditIntent.h"
#include "HostSceneStorage.h"
#include "captured_log.h"
#include <pxr/base/vt/array.h>
#include <pxr/base/vt/dictionary.h>
#include <pxr/usd/sdf/data.h>
#include <pxr/usd/sdf/schema.h>
#include <pxr/usd/sdf/attributeSpec.h>
#include <pxr/usd/sdf/changeBlock.h>
#include <pxr/usd/sdf/fileFormat.h>
#include <pxr/usd/sdf/layer.h>
#include <pxr/usd/sdf/namespaceEdit.h>
#include <pxr/usd/sdf/payload.h>
#include <pxr/usd/sdf/primSpec.h>
#include <algorithm>
#include <array>
#include <optional>
#include <random>
#include <set>
#include <iostream>
#include <map>
#include <stdexcept>

using namespace PXR_NS;
using namespace omni::usd::live;
using namespace omni::usd::live::wire;
using namespace omni::usd::live::test;
namespace {
std::string intentContext;
void checkIntent(bool condition,const char* message) { if (!condition) throw std::runtime_error(intentContext+": "+message); }
struct Record {
    SdfSpecType type;
    std::map<TfToken,VtValue> fields;
    bool operator==(const Record& other) const { return type == other.type && fields == other.fields; }
};
using State = std::map<SdfPath,Record>;
State inspect(const SdfAbstractData& source) {
    struct Visitor : SdfAbstractDataSpecVisitor {
        State result;
        bool VisitSpec(const SdfAbstractData& source,const SdfPath& path) override {
            Record record{source.GetSpecType(path),{}};
            for (const auto& field : source.List(path)) record.fields.emplace(field,source.Get(path,field));
            checkIntent(result.emplace(path,std::move(record)).second,"duplicate oracle path"); return true;
        }
        void Done(const SdfAbstractData&) override {}
    } visitor;
    source.VisitSpecs(&visitor); return visitor.result;
}
// Whether a log can skip namespace planning: no structural operation and no namespace child field.
bool fieldOnly(const std::vector<SdfOperation>& operations) {
    static const std::set<TfToken> children{SdfChildrenKeys->PrimChildren,SdfChildrenKeys->PropertyChildren,
        SdfChildrenKeys->VariantSetChildren,SdfChildrenKeys->VariantChildren,SdfChildrenKeys->ConnectionChildren,
        SdfChildrenKeys->RelationshipTargetChildren,SdfChildrenKeys->MapperChildren,SdfChildrenKeys->MapperArgChildren,
        SdfChildrenKeys->ExpressionChildren};
    for (const auto& op : operations)
        if (op.kind == SdfOperationKind::CreateSpec || op.kind == SdfOperationKind::EraseSpec ||
            op.kind == SdfOperationKind::MoveSpec || children.count(op.field))
            return false;
    return true;
}
// The fast path skips planning exactly for field-only logs, and planning every log compiles the same bytes.
void samePlanEitherWay(const WireScenePtr& base,const std::shared_ptr<const SdfOperationLog>& sealed,
                       const FieldEditIntentPtr& fast) {
    FieldEditLimits always; always.footprint = false;
    const auto slow = compileFieldEdits(base,sealed,always);
    checkIntent(fast->planned() == !fieldOnly(*sealed->operations) && slow->planned(),
                "namespace planning ran for a field-only log or was skipped for another");
    checkIntent(fast->bytes() == slow->bytes() && fast->retainedBytes() == slow->retainedBytes() &&
                fast->empty() == slow->empty() && fast->concurrentStructureSupported() == slow->concurrentStructureSupported(),
                "the field-only fast path changed the compiled intent");
}
Command create(uint64_t id,uint64_t parent,const char* name,SdfSpecType type) {
    Command result{}; result.kind = Command::Kind::Create; result.section = id; result.parent = parent;
    result.name = name; result.sectionType = uint8_t(type); return result;
}
WireScenePtr initialScene() {
    EnvelopeInput metadata;
    metadata.commands = {create(17,1,"P",SdfSpecTypePrim),create(91,17,"a",SdfSpecTypeAttribute),create(203,17,"b",SdfSpecTypeAttribute)};
    metadata.fields = {{91,0,"default",{}},{203,1,"default",{}},{17,2,"customData",{}}};
    metadata.samples = {{91,0,{}},{203,0,{}}};
    VtDictionary dictionary; dictionary["base"] = VtValue(1);
    DecodedValues values{{VtValue(0),VtValue(0),VtValue(dictionary)}, {VtValue(10),VtValue(20)}};
    return buildSceneCandidate({},encodeUsdEnvelope(metadata,values));
}
struct Fixture {
    WireScenePtr base = initialScene();
    TfRefPtr<HostSceneProjection> backing = recordingBacking(base);
    SdfDataRefPtr oracle = TfCreateRefPtr(new SdfData);
    const State before = inspect(*base->data());
    Fixture() { oracle->CopyFrom(base->data()); }
    template<class F> void both(F mutation) {
        mutation(*backing); mutation(*oracle);
        if (inspect(*backing) != inspect(*oracle)) {
            std::cerr << "backing data after operation " << backing->pendingLocalEditCount() << ":\n";
            backing->WriteToStream(std::cerr);
            std::cerr << "stock oracle data:\n"; oracle->WriteToStream(std::cerr);
            checkIntent(false,"backing mutation differs from stock oracle");
        }
    }
    FieldEditIntentPtr compile() {
        const auto sealed = recordedLog(*backing); const auto count = sealed->operations->size();
        auto result = compileFieldEdits(base,sealed);
        samePlanEitherWay(base,sealed,result);
        checkIntent(result && result->operations() == sealed->operations,"compiler did not retain the original log");
        checkIntent(sealed->operations->size() == count && inspect(*sealed->data) == inspect(*oracle) && inspect(*base->data()) == before,
                    "compiler changed retained original/base");
        const auto candidate = buildSceneCandidate(base,result->bytes());
        checkIntent(inspect(*candidate->data()) == inspect(*oracle),"compiled operations differ from independent stock mutations");
        checkIntent(candidate->sectionForPath(SdfPath("/P")) == 17 && candidate->sectionForPath(SdfPath("/P.a")) == 91 &&
                    candidate->sectionForPath(SdfPath("/P.b")) == 203,"field compiler changed base section identity");
        return result;
    }
};
void ordinary(const Envelope& envelope) {
    checkIntent(!envelope.isDiff && envelope.commands.empty() && envelope.sourceFormat.empty(),"field intent emitted structural/diff metadata");
}
void orderedFields() {
    Fixture f;
    for (int value : {1,0,0}) f.both([&](auto& data) { data.Set(SdfPath("/P.a"),SdfFieldKeys->Default,VtValue(value)); });
    f.both([](auto& data) { data.Set(SdfPath("/P.b"),SdfFieldKeys->Default,VtValue(-7)); });
    const auto intent = f.compile(); const auto envelope = decodeEnvelope(intent->bytes()); ordinary(envelope);
    // /P.a's writes collapse to one record of its last value, which equals
    // the base's and is still emitted.
    checkIntent(envelope.fields.size() == 2 && envelope.samples.empty(),"repeated field writes did not collapse");
    for (size_t i = 0; i < 2; ++i) {
        const auto& field = envelope.fields[i];
        checkIntent(field.section == (i == 1 ? 203 : 91) && field.name == "default" && field.setOrder == i+3,
                    "field IDs/order differ from the last write of each field");
        const int expected = i == 0 ? 0 : -7;
        const auto raw = field.value.bytes();
        checkIntent(raw.size() == 6 && uint8_t(raw[0]) == 3 && raw[1] == 0,"int32 type bytes differ");
        for (unsigned byte = 0; byte < 4; ++byte)
            checkIntent(uint8_t(raw[2+byte]) == uint8_t(uint32_t(expected) >> (byte*8)),"independent int32 bytes differ");
    }
    const auto savedBytes = intent->bytes(); const auto owner = intent->operations();
    f.backing = nullptr; f.base.reset();
    checkIntent(intent->bytes() == savedBytes && owner->size() == 4 &&
                owner->at(0).value == VtValue(1),"compiled intent lost retained bytes/log after backing retirement");
}
void interleavedSamples() {
    Fixture f;
    f.both([](auto& data) { data.SetTimeSample(SdfPath("/P.a"),1,VtValue(11)); });
    f.both([](auto& data) { data.SetTimeSample(SdfPath("/P.b"),9,VtValue(99)); });
    const SdfTimeSampleMap first{{2,VtValue(22)}};
    f.both([&](auto& data) { data.Set(SdfPath("/P.a"),SdfFieldKeys->TimeSamples,VtValue(first)); });
    f.both([](auto& data) { data.SetTimeSample(SdfPath("/P.a"),3,VtValue(33)); });
    const SdfTimeSampleMap last{{4,VtValue(44)},{5,VtValue(55)}};
    f.both([&](auto& data) { data.Set(SdfPath("/P.a"),SdfFieldKeys->TimeSamples,VtValue(last)); });
    f.both([](auto& data) { data.EraseTimeSample(SdfPath("/P.a"),4); });
    f.both([](auto& data) { data.SetTimeSample(SdfPath("/P.a"),6,VtValue(SdfValueBlock{})); });
    const auto intent = f.compile(); const auto envelope = decodeEnvelope(intent->bytes()); ordinary(envelope);
    checkIntent(intent->operations()->size() == 7,"sample normalization changed original operation history");
    const auto& original = *intent->operations();
    checkIntent(original[0].kind == SdfOperationKind::SetSample && original[0].time == 1 && original[0].value == VtValue(11) &&
                original[3].kind == SdfOperationKind::SetSample && original[3].time == 3 && original[3].value == VtValue(33),
                "suppressed samples were erased or rewritten in retained original log");
    checkIntent(envelope.fields.empty(),"time samples emitted as ordinary fields");
    checkIntent(envelope.samples.size() == 8,"expanded sample records differ");
    const auto values = decodeValues(envelope);
    checkIntent(values.samples.back() == VtValue(SdfValueBlock{}),"sample block collapsed into deletion");
}
void emptyMapsAndErasure() {
    for (bool eraseMap : {false,true}) {
        Fixture f;
        f.both([](auto& data) { data.SetTimeSample(SdfPath("/P.a"),1,VtValue(11)); });
        f.both([&](auto& data) {
            if (eraseMap) data.Erase(SdfPath("/P.a"),SdfFieldKeys->TimeSamples);
            else data.Set(SdfPath("/P.a"),SdfFieldKeys->TimeSamples,VtValue(SdfTimeSampleMap{}));
        });
        if (eraseMap) f.both([](auto& data) { data.EraseTimeSample(SdfPath("/P.a"),999); });
        else {
            // Pinned stock SdfData drops an explicitly authored empty map when
            // erasing an absent sample. The scene contract intentionally keeps
            // field presence for this no-op. Assert the deviation explicitly,
            // then author the contract's expected empty map in the stock model.
            f.backing->EraseTimeSample(SdfPath("/P.a"),999);
            f.oracle->EraseTimeSample(SdfPath("/P.a"),999);
            checkIntent(!f.oracle->Has(SdfPath("/P.a"),SdfFieldKeys->TimeSamples),
                        "pinned stock missing-sample/empty-map diagnostic oracle changed");
            checkIntent(f.backing->Has(SdfPath("/P.a"),SdfFieldKeys->TimeSamples),
                        "backing lost explicitly authored empty-map presence");
            f.oracle->Set(SdfPath("/P.a"),SdfFieldKeys->TimeSamples,VtValue(SdfTimeSampleMap{}));
            checkIntent(inspect(*f.backing) == inspect(*f.oracle),"empty-map presence correction differs");
        }
        if (!eraseMap) {
            bool unsupported = false;
            try { (void)f.compile(); } catch (const DecodeError& error) { unsupported = error.kind == ErrorKind::Unsupported; }
            checkIntent(unsupported,"explicit empty sample map admitted");
            checkIntent(f.backing->Has(SdfPath("/P.a"),SdfFieldKeys->TimeSamples),"rejection changed host field");
            continue;
        }
        const auto intent = f.compile(); const auto envelope = decodeEnvelope(intent->bytes());
        checkIntent(envelope.fields.empty() && envelope.samples.size() == 3 &&
                    envelope.samples.back().time == 999 && envelope.samples.back().value.empty(),"erased map normalization differs");
        const auto actual = buildSceneCandidate(f.base,intent->bytes());
        checkIntent(!actual->data()->Has(SdfPath("/P.a"),SdfFieldKeys->TimeSamples),"map erasure changed presence");
    }
    Fixture f;
    f.both([](auto& data) { data.Set(SdfPath("/P.a"),SdfFieldKeys->Default,VtValue(SdfValueBlock{})); });
    f.both([](auto& data) { data.Erase(SdfPath("/P.b"),SdfFieldKeys->Default); });
    const auto envelope = decodeEnvelope(f.compile()->bytes());
    checkIntent(envelope.fields.size() == 2 && !envelope.fields[0].value.empty() && envelope.fields[1].value.empty(),
                "ValueBlock collapsed into field deletion");
}
void dictionarySemantics() {
    Fixture f; std::vector<VtValue> dictionaries;
    auto capture = [&] { dictionaries.push_back(f.oracle->Get(SdfPath("/P"),SdfFieldKeys->CustomData)); };
    f.both([](auto& data) { data.SetDictValueByKey(SdfPath("/P"),SdfFieldKeys->CustomData,TfToken("nested:k"),VtValue(2)); }); capture();
    f.both([](auto& data) { data.EraseDictValueByKey(SdfPath("/P"),SdfFieldKeys->CustomData,TfToken("base")); }); capture();
    f.both([](auto& data) { data.SetDictValueByKey(SdfPath("/P"),SdfFieldKeys->CustomData,TfToken("nested:k"),VtValue(3)); }); capture();
    const auto intent = f.compile(); const auto envelope = decodeEnvelope(intent->bytes());
    // Dictionary-key edits are whole-field replacements, so they collapse to
    // one record of the field's last value.
    checkIntent(envelope.fields.size() == 1 && envelope.samples.empty() &&
                decodeValues(envelope).fields == std::vector<VtValue>{dictionaries.back()},
                "dictionary operations did not compile to one whole-field value");
    checkIntent(envelope.fields[0].section == 17 && envelope.fields[0].name == "customData" && envelope.fields[0].setOrder == 3,
                "dictionary wire field identity/order differs");
    checkIntent(intent->operations()->size() == 3,"the intent did not retain every dictionary operation");
    VtDictionary peer; peer["base"] = VtValue(1); peer["peerOnly"] = VtValue(99);
    EnvelopeInput metadata; metadata.fields = {{17,0,"customData",{}}};
    const auto peerScene = buildSceneCandidate(f.base,encodeUsdEnvelope(metadata,DecodedValues{{VtValue(peer)},{}}));
    const auto rebased = buildSceneCandidate(peerScene,intent->bytes());
    checkIntent(rebased->data()->Get(SdfPath("/P"),SdfFieldKeys->CustomData) == dictionaries.back(),
                "dictionary field semantics incorrectly merged peer keys");

    Fixture sample;
    sample.both([](auto& data) { data.SetTimeSample(SdfPath("/P.a"),1,VtValue(11)); });
    const auto sampleIntent = sample.compile();
    metadata = {}; metadata.samples = {{91,8,{}}};
    const auto peerSamples = buildSceneCandidate(sample.base,encodeUsdEnvelope(metadata,DecodedValues{{},{VtValue(88)}}));
    const auto after = buildSceneCandidate(peerSamples,sampleIntent->bytes());
    checkIntent(after->data()->Get(SdfPath("/P.a"),SdfFieldKeys->TimeSamples) ==
                VtValue(SdfTimeSampleMap{{0,VtValue(10)},{1,VtValue(11)},{8,VtValue(88)}}),"per-sample wire edit replaced peer map");
}
void emptyDictionaryKeySemantics() {
    for (unsigned mode = 0; mode < 3; ++mode) {
        VtDictionary keep; keep["keep"] = VtValue(9);
        EnvelopeInput metadata; metadata.fields = {{17,0,"customData",{}}};
        const VtValue initial = mode == 0 ? VtValue() : mode == 1 ? VtValue(9) : VtValue(keep);
        const auto base = buildSceneCandidate(initialScene(),encodeUsdEnvelope(metadata,DecodedValues{{initial},{}}));
        const auto before = inspect(*base->data());
        auto backing = recordingBacking(base);
        auto oracle = TfCreateRefPtr(new SdfData); oracle->CopyFrom(base->data());
        for (SdfAbstractData* data : {static_cast<SdfAbstractData*>(backing.operator->()),
                                      static_cast<SdfAbstractData*>(oracle.operator->())}) {
            data->SetDictValueByKey(SdfPath("/P"),SdfFieldKeys->CustomData,TfToken(),VtValue(7));
            data->EraseDictValueByKey(SdfPath("/P"),SdfFieldKeys->CustomData,TfToken());
        }
        const auto original = recordedLog(*backing); const auto intent = compileFieldEdits(base,original);
        const auto candidate = buildSceneCandidate(base,intent->bytes());
        checkIntent(inspect(*candidate->data()) == inspect(*oracle) && inspect(*base->data()) == before,
                    "compiled empty-key semantics differ from stock or changed retained base");
        const auto envelope = decodeEnvelope(intent->bytes()); const auto values = decodeValues(envelope);
        // The set and the erase are whole-field replacements of one field,
        // which collapse to the erase's.
        const VtValue afterErase = mode == 2 ? VtValue(keep) : VtValue();
        checkIntent(values.fields == std::vector<VtValue>{afterErase} && envelope.fields.size() == 1 &&
                    envelope.fields[0].section == 17 && envelope.fields[0].setOrder == 2,
                    "empty-key operations did not compile to the field's last whole-field value");
        checkIntent(intent->operations() == original->operations && original->operations->size() == 2 &&
                    original->operations->at(0).kind == SdfOperationKind::SetDictKey &&
                    original->operations->at(1).kind == SdfOperationKind::EraseDictKey &&
                    original->operations->at(0).key.IsEmpty() && original->operations->at(1).key.IsEmpty(),
                    "compiler discarded empty-key original operations");
    }
}
void typedAndExternal() {
    Fixture f;
    int value = 7; const SdfAbstractDataConstTypedValue<int> typed(&value);
    f.both([&](auto& data) { data.Set(SdfPath("/P.a"),SdfFieldKeys->Default,typed); });
    value = 99;
    VtIntArray array{1,-2,3};
    f.both([&](auto& data) { data.Set(SdfPath("/P"),TfToken("array"),VtValue(array)); });
    const std::string external(70000,'x');
    f.both([&](auto& data) { data.Set(SdfPath("/P"),TfToken("payload"),VtValue(external)); });
    f.both([&](auto& data) { data.SetTimeSample(SdfPath("/P.b"),5,VtValue(external+"sample")); });
    const auto intent = f.compile(); const auto& bytes = intent->bytes();
    checkIntent(bytes.size() > external.size()*2 && bytes.substr(0,4) == "OMPF","missing multipart payload");
    uint32_t parts = 0;
    for (unsigned i = 0; i < 4; ++i) parts |= uint32_t(uint8_t(bytes[4+i])) << (8*i);
    checkIntent(parts == 3,"large field/sample were not independent external parts");
    const auto envelope = decodeEnvelope(bytes); const auto decoded = decodeValues(envelope);
    checkIntent(decoded.fields == std::vector<VtValue>{VtValue(7),VtValue(array),VtValue(external)} &&
                decoded.samples == std::vector<VtValue>{VtValue(external+"sample")},"typed/external data differs");
    checkIntent(intent->retainedBytes() >= bytes.size() && intent->workUsed() > 0,"missing logical retention/work accounting");
}
void reject(const WireScenePtr& base,const std::shared_ptr<const SdfOperationLog>& original,
            ErrorKind kind,const FieldEditLimits& limits = {}) {
    const auto baseBefore = inspect(*base->data()), dataBefore = inspect(*original->data);
    const auto operationsOwner = original->operations;
    const auto operations = *original->operations;
    std::string published = "unpublished sentinel"; bool rejected = false;
    try { published = compileFieldEdits(base,original,limits)->bytes(); }
    catch (const DecodeError& error) { checkIntent(error.kind == kind,"wrong compiler rejection kind"); rejected = true; }
    checkIntent(rejected && published == "unpublished sentinel","unsupported/limited compile published bytes");
    checkIntent(inspect(*base->data()) == baseBefore && inspect(*original->data) == dataBefore && original->operations == operationsOwner &&
                original->operations->size() == operations.size(),"failed compiler changed retained base/data/log");
    for (size_t i = 0; i < operations.size(); ++i) {
        const auto& a = operations[i]; const auto& b = original->operations->at(i);
        checkIntent(a.kind == b.kind && a.path == b.path && a.destination == b.destination && a.specType == b.specType &&
                    a.field == b.field && a.key == b.key && a.time == b.time && a.value == b.value && a.replacement == b.replacement,
                    "failed compiler changed operation metadata/owner");
    }
}
// The compile fails with UnsupportedOperation naming operation `index`.
void unsupportedAt(const WireScenePtr& base,const std::shared_ptr<const SdfOperationLog>& original,size_t index,
                   UnsupportedOperation::Reason reason) {
    bool named = false;
    try { (void)compileFieldEdits(base,original); }
    catch (const UnsupportedOperation& error) {
        checkIntent(error.kind == ErrorKind::Unsupported && error.index() == index && error.reason() == reason,
                    "unsupported compile named the wrong operation or reason");
        named = true;
    }
    checkIntent(named,"unsupported compile did not name its operation");
}
size_t firstOperation(const SdfOperationLog& log,SdfOperationKind kind) {
    const auto& operations = *log.operations;
    for (size_t i = 0; i < operations.size(); ++i) if (operations[i].kind == kind) return i;
    checkIntent(false,"recorded log lacks the expected operation"); return 0;
}
void unsupportedAndLimits() {
    for (unsigned which = 0; which < 2; ++which) {
        Fixture f;
        f.backing->Set(SdfPath("/P.a"),SdfFieldKeys->Default,VtValue(7));
        if (which == 0) {
            // The backing retypes the existing spec and records the Create.
            f.backing->CreateSpec(SdfPath("/P.a"),SdfSpecTypeRelationship);
            reject(f.base,recordedLog(*f.backing),ErrorKind::Unsupported);
            const auto log = recordedLog(*f.backing);
            unsupportedAt(f.base,log,firstOperation(*log,SdfOperationKind::CreateSpec),
                          UnsupportedOperation::Reason::IdentityTypeChange);
            continue;
        }
        // The host backing refuses CopyFrom before mutation, so its record
        // is appended to the recorded log.
        SdfOperation copy(SdfOperationKind::CopyFrom);
        copy.replacement = f.oracle;
        reject(f.base,withOperation(*recordedLog(*f.backing),std::move(copy)),ErrorKind::Unsupported);
    }
    {
        // A whole time-sample field must hold a sample map; storage classifies
        // any other value as malformed during replay.
        Fixture f;
        SdfOperation samples(SdfOperationKind::SetField);
        samples.path = SdfPath("/P.a"); samples.field = SdfFieldKeys->TimeSamples; samples.value = VtValue(1);
        reject(f.base,withOperation(*recordedLog(*f.backing),std::move(samples)),ErrorKind::Malformed);
    }
    Fixture f;
    f.both([](auto& data) { data.Set(SdfPath("/P.a"),SdfFieldKeys->Default,VtValue(1)); });
    f.both([](auto& data) { data.Set(SdfPath("/P.b"),SdfFieldKeys->Default,VtValue(2)); });
    (void)f.compile();
    auto unsupported = std::make_shared<SdfOperationLog>(*recordedLog(*f.backing));
    SdfOperation dictionarySample(SdfOperationKind::SetDictKey);
    dictionarySample.path = SdfPath("/P.a"); dictionarySample.field = SdfFieldKeys->TimeSamples;
    dictionarySample.key = TfToken("1"); dictionarySample.value = VtValue(9);
    unsupported->operations = std::make_shared<const std::vector<SdfOperation>>(
        std::vector<SdfOperation>{dictionarySample});
    reject(f.base,unsupported,ErrorKind::Unsupported);
    // After supported operations, the time-sample dictionary-key edit is the
    // named operation.
    const auto recorded = recordedLog(*f.backing);
    unsupportedAt(f.base,withOperation(*recorded,dictionarySample),recorded->operations->size(),
                  UnsupportedOperation::Reason::TimeSampleDictionaryKey);
    {
        // A mapper whose target is relative cannot become a wire identity.
        Fixture mapper;
        mapper.both([](auto& data) { data.CreateSpec(SdfPath("/P.a.mapper[T.rel]"),SdfSpecTypeMapper); });
        unsupportedAt(mapper.base,recordedLog(*mapper.backing),UnsupportedOperation::noOperation,
                      UnsupportedOperation::Reason::IdentityRealization);
    }
    {
        // A mixed batch names its first unencodable operation, not the first
        // operation or the last one.
        Fixture mixed;
        mixed.both([](auto& data) { data.Set(SdfPath("/P.a"),SdfFieldKeys->Default,VtValue(5)); });
        mixed.both([](auto& data) { data.Set(SdfPath("/P"),TfToken("vector"),VtValue(std::vector<int>{1,2,3})); });
        const SdfPayload offset("model.usda",SdfPath("/Root"),SdfLayerOffset(3,1));
        mixed.both([&](auto& data) { data.Set(SdfPath("/P"),TfToken("offset"),VtValue(offset)); });
        unsupportedAt(mixed.base,recordedLog(*mixed.backing),1,UnsupportedOperation::Reason::ValueType);
        Fixture payload;
        payload.both([](auto& data) { data.Set(SdfPath("/P.a"),SdfFieldKeys->Default,VtValue(5)); });
        payload.both([&](auto& data) { data.Set(SdfPath("/P"),TfToken("offset"),VtValue(offset)); });
        unsupportedAt(payload.base,recordedLog(*payload.backing),1,UnsupportedOperation::Reason::PayloadOffset);
        // A dictionary-key value is preflighted before the batch measure.
        Fixture key;
        key.both([](auto& data) { data.Set(SdfPath("/P.a"),SdfFieldKeys->Default,VtValue(5)); });
        key.both([](auto& data) {
            data.SetDictValueByKey(SdfPath("/P"),SdfFieldKeys->CustomData,TfToken("vector"),VtValue(std::vector<int>{1}));
        });
        unsupportedAt(key.base,recordedLog(*key.backing),1,UnsupportedOperation::Reason::ValueType);
    }
    FieldEditLimits limits; limits.maxOperations = 1;
    reject(f.base,recordedLog(*f.backing),ErrorKind::ResourceLimit,limits);
    limits = {}; limits.maxWork = 1;
    reject(f.base,recordedLog(*f.backing),ErrorKind::ResourceLimit,limits);
    limits = {}; limits.envelope.content.maxBytes = 1;
    reject(f.base,recordedLog(*f.backing),ErrorKind::ResourceLimit,limits);
    limits = {}; limits.values.maxEncodedBytes = 1;
    reject(f.base,recordedLog(*f.backing),ErrorKind::ResourceLimit,limits);
    limits = {}; limits.maxRetainedBytes = 1;
    reject(f.base,recordedLog(*f.backing),ErrorKind::ResourceLimit,limits);

    // This large original operation vanishes from normalized wire bytes and
    // final scene. Its owner still matters to retention quotas and recovery.
    Fixture large;
    const std::string overwritten(2*1024*1024,'z');
    large.both([&](auto& data) { data.SetTimeSample(SdfPath("/P.a"),1,VtValue(overwritten)); });
    large.both([](auto& data) { data.Erase(SdfPath("/P.a"),SdfFieldKeys->TimeSamples); });
    const auto intent = large.compile();
    checkIntent(intent->bytes().size() < 4096 && intent->retainedBytes() >= overwritten.size() &&
                intent->operations()->at(0).value == VtValue(overwritten),"normalization lost superseded retained payload");
    limits = {}; limits.maxRetainedBytes = 256*1024;
    // Establish this limit accommodates the same small base and ordinary edits.
    checkIntent(bool(compileFieldEdits(f.base,recordedLog(*f.backing),limits)),"retention fixture limit rejects small baseline");
    reject(large.base,recordedLog(*large.backing),ErrorKind::ResourceLimit,limits);
}
void dictionaryDepthPreflight() {
    struct CountedSource final : SdfData {
        mutable size_t reads = 0;
        VtValue Get(const SdfPath& path,const TfToken& field) const override {
            ++reads; return SdfData::Get(path,field);
        }
    };
    const auto base = initialScene();
    const State before = inspect(*base->data());
    FieldEditLimits limits;
    limits.values.maxDepth = limits.scene.retainedValues.maxDepth = limits.scene.values.maxDepth = 5;
    auto run = [&](const char* key,const VtValue& value,bool accepted) {
        auto source = TfCreateRefPtr(new CountedSource); source->CopyFrom(base->data());
        source->Erase(SdfPath("/P"),SdfFieldKeys->CustomData);
        const auto expectedSource = inspect(*source);
        auto sealed = std::make_shared<SdfOperationLog>(); sealed->data = source;
        SdfOperation set(SdfOperationKind::SetDictKey);
        set.path = SdfPath("/P"); set.field = SdfFieldKeys->CustomData; set.key = TfToken(key); set.value = value;
        SdfOperation erase(SdfOperationKind::EraseField);
        erase.path = set.path; erase.field = set.field;
        sealed->operations = std::make_shared<const std::vector<SdfOperation>>(
            std::vector<SdfOperation>{set,erase});
        source->reads = 0;
        FieldEditIntentPtr intent; bool resource = false;
        try { intent = compileFieldEdits(base,sealed,limits); }
        catch (const DecodeError& error) {
            resource = error.kind == ErrorKind::ResourceLimit;
            if (accepted) std::cerr << "expected valid dictionary key=" << key << " reads=" << source->reads << " error=" << error.what() << '\n';
        }
        if (accepted) {
            checkIntent(intent && !resource && source->reads > 0,"valid exact dictionary depth failed or prepass control never traversed source");
            const auto final = buildSceneCandidate(base,intent->bytes());
            checkIntent(!final->data()->Has(SdfPath("/P"),SdfFieldKeys->CustomData),"final shallow dictionary fixture differs");
        } else {
            // This counted snapshot is visited before the compiler constructs
            // mutable replay. A later encoder rejection must not pass this test.
            checkIntent(resource && !intent && source->reads == 0,"dictionary depth rejected only after replay preparation");
        }
        checkIntent(inspect(*base->data()) == before && inspect(*source) == expectedSource && sealed->operations->size() == 2 &&
                    sealed->operations->at(0).key == TfToken(key) && sealed->operations->at(0).value == value,
                    "dictionary preflight changed retained inputs");
    };
    // Scalar depth1 plus two codec levels per key component: depth5 is exact.
    run("a:b",VtValue(1),true);
    run(":",VtValue(1),true); // Two empty components are a valid depth5 key.
    run("::",VtValue(1),false); // Three empty components require depth7.
    // SdfAbstractData's key path keeps empty split components: this decorated
    // key is deeper than a:b and must be bounded before invoking the SDK.
    run(":a::b:",VtValue(1),false);
    run("a:b:c",VtValue(1),false);
    VtDictionary nested; nested["leaf"] = VtValue(2); // Standalone depth3.
    run("a",VtValue(nested),true);
    run("a:b",VtValue(nested),false); // Combined depth7, individually bounded inputs.
    VtDictionary deeper; deeper["nested"] = VtValue(nested); // Standalone depth5.
    run("",VtValue(deeper),true); // A truly empty key adds zero path components.
    run("a:b:c",VtValue(),false); // Erase-equivalent Set still traverses the key path.
}
// A host capture of `log` changed by `edit`: a forged final snapshot on the
// production path, where the capture is host storage.
template<class Edit>
std::shared_ptr<const SdfOperationLog> forgedCapture(const SdfOperationLog& log,Edit edit) {
    const auto* host = dynamic_cast<const HostSceneStorage*>(log.data.operator->());
    checkIntent(host != nullptr,"a recorded log's capture is not host storage");
    auto copy = host->clone(); edit(*copy);
    auto forged = std::make_shared<SdfOperationLog>(log); forged->data = copy; return forged;
}
// The compiler refuses `log` as Malformed, by the check whose message holds `check`.
void refusedAs(const WireScenePtr& base,const std::shared_ptr<const SdfOperationLog>& log,const char* check) {
    reject(base,log,ErrorKind::Malformed);
    std::string message;
    try { (void)compileFieldEdits(base,log); } catch (const DecodeError& error) { message = error.what(); }
    if (message.find(check) == std::string::npos) std::cerr << "refused with: " << message << '\n';
    checkIntent(message.find(check) != std::string::npos,"a forged capture was refused by another check");
}
void operationAuthority() {
    Fixture f;
    f.both([](auto& data) { data.Set(SdfPath("/P.a"),SdfFieldKeys->Default,VtValue(7)); });
    const auto original = recordedLog(*f.backing);
    // A field-only log over host storage takes the production path: no
    // planning, verified at the specs it addresses.
    checkIntent(!compileFieldEdits(f.base,original)->planned(),"a field-only host log was planned");
    // A forged final snapshot must not override the original captured edit,
    // nor may it silently pass final-equivalence verification. The log still
    // drives lowering; the immutable final snapshot must also agree, so an
    // omitted capture cannot silently publish different contents.
    refusedAs(f.base,forgedCapture(*original,[](HostSceneStorage& data) {
        data.Set(SdfPath("/P.a"),SdfFieldKeys->Default,VtValue(123)); }),"changed authored values");
    // A snapshot of another storage kind is checked against every spec.
    auto detached = std::make_shared<SdfOperationLog>(*original);
    auto different = TfCreateRefPtr(new SdfData); different->CopyFrom(f.base->data());
    different->Set(SdfPath("/P.a"),SdfFieldKeys->Default,VtValue(123));
    detached->data = different;
    reject(f.base,detached,ErrorKind::Malformed);
}
// On the production path, a host capture that holds a spec the log does not
// address, lacks one, or whose addressed spec lacks or adds a field, is
// refused.
void hostCaptureDivergence() {
    Fixture f;
    f.both([](auto& data) { data.Set(SdfPath("/P.a"),SdfFieldKeys->Default,VtValue(7)); });
    const auto original = recordedLog(*f.backing);
    checkIntent(!compileFieldEdits(f.base,original)->planned(),"a field-only host log was planned");
    refusedAs(f.base,forgedCapture(*original,[](HostSceneStorage& data) {
        data.CreateSpec(SdfPath("/Extra"),SdfSpecTypePrim); }),"added specs");
    refusedAs(f.base,forgedCapture(*original,[](HostSceneStorage& data) {
        data.EraseSpec(SdfPath("/P.b")); }),"added specs");
    refusedAs(f.base,forgedCapture(*original,[](HostSceneStorage& data) {
        data.Erase(SdfPath("/P.a"),SdfFieldKeys->Default); }),"changed field presence");
    refusedAs(f.base,forgedCapture(*original,[](HostSceneStorage& data) {
        data.Set(SdfPath("/P.a"),SdfFieldKeys->Documentation,VtValue(std::string("forged"))); }),"changed field presence");
}
// Host storage ignores value writes to absent specs and moves from an absent
// source or onto an existing destination, yet the host projection records
// them. The compiler skips exactly those records; the final snapshot check
// still rejects a log whose snapshot shows such an edit applied.
void hostNoOpRecords() {
    const auto base = initialScene();
    using Kind = SdfOperationKind;
    auto op = [](Kind kind,const char* path) { SdfOperation result(kind); result.path = SdfPath(path); return result; };
    std::vector<SdfOperation> operations;
    auto set = op(Kind::SetField,"/Missing"); set.field = SdfFieldKeys->Default; set.value = VtValue(5); operations.push_back(set);
    auto key = op(Kind::SetDictKey,"/Missing"); key.field = SdfFieldKeys->CustomData; key.key = TfToken("k"); key.value = VtValue(1);
    operations.push_back(key);
    auto sample = op(Kind::SetSample,"/Missing"); sample.time = 1; sample.value = VtValue(3); operations.push_back(sample);
    auto absent = op(Kind::MoveSpec,"/Missing"); absent.destination = SdfPath("/Other"); operations.push_back(absent);
    auto occupied = op(Kind::MoveSpec,"/P.a"); occupied.destination = SdfPath("/P.b"); operations.push_back(occupied);
    auto self = op(Kind::MoveSpec,"/P"); self.destination = SdfPath("/P"); operations.push_back(self);
    auto real = op(Kind::SetField,"/P.a"); real.field = SdfFieldKeys->Default; real.value = VtValue(9); operations.push_back(real);
    auto expected = TfCreateRefPtr(new SdfData); expected->CopyFrom(base->data());
    expected->Set(SdfPath("/P.a"),SdfFieldKeys->Default,VtValue(9));
    auto sealed = std::make_shared<SdfOperationLog>();
    sealed->operations = std::make_shared<const std::vector<SdfOperation>>(operations);
    sealed->data = expected;
    WireScenePtr appliedScene;
    const auto intent = compileFieldEdits(base,sealed,{},{},&appliedScene);
    const auto candidate = buildSceneCandidate(base,intent->bytes());
    checkIntent(inspect(*candidate->data()) == inspect(*expected),"host no-op records changed the compiled scene");
    checkIntent(appliedScene && inspect(*appliedScene->data()) == inspect(*expected) &&
                appliedScene->sectionForPath(SdfPath("/P.a")) == candidate->sectionForPath(SdfPath("/P.a")),
                "compiler did not report the verified applied scene");
    checkIntent(intent->concurrentStructureSupported() && intent->unsupportedStructure() == UnsupportedOperation::Reason{},
                "skipped no-op moves made the intent structural");
    // A snapshot claiming the ignored write applied is still inconsistent.
    auto applied = TfCreateRefPtr(new SdfData); applied->CopyFrom(expected);
    applied->CreateSpec(SdfPath("/Missing"),SdfSpecTypePrim);
    applied->Set(SdfPath("/Missing"),SdfFieldKeys->Default,VtValue(5));
    auto forged = std::make_shared<SdfOperationLog>(*sealed); forged->data = applied;
    reject(base,forged,ErrorKind::Malformed);
}
class IntentLayerFormat final : public SdfFileFormat {
    const SdfAbstractDataRefPtr data_;
public:
    explicit IntentLayerFormat(SdfAbstractDataRefPtr data)
        : SdfFileFormat(TfToken("structural_intent_test"),TfToken("1"),TfToken("usd"),"intenttest"), data_(std::move(data)) {}
    SdfAbstractDataRefPtr InitData(const FileFormatArguments&) const override { return data_; }
    bool CanRead(const std::string&) const override { return false; }
    bool Read(SdfLayer*,const std::string&,bool) const override { return false; }
protected:
    SdfAbstractDataRefPtr _InitDetachedData(const FileFormatArguments& args) const override { return InitData(args); }
};
std::shared_ptr<SectionIdAllocator> sequentialIds(uint64_t first = 1000) {
    return std::make_shared<SectionIdAllocator>(4096,64,[next=first]() mutable { return next++; });
}
FieldEditIntentPtr structuralCompile(const WireScenePtr& base,const TfRefPtr<HostSceneProjection>& backing,
                                    const SdfDataRefPtr& oracle,const std::shared_ptr<SectionIdAllocator>& ids) {
    const auto before = inspect(*base->data());
    const auto sealed = recordedLog(*backing); const auto operations = sealed->operations;
    const auto count = operations->size();
    checkIntent(inspect(*sealed->data) == inspect(*oracle),"structural backing differs from independent stock operations");
    const auto intent = compileFieldEdits(base,sealed,{},ids);
    checkIntent(intent->planned(),"a structural edit skipped namespace planning");
    checkIntent(intent->operations() == operations && sealed->operations == operations &&
                operations->size() == count && inspect(*base->data()) == before,"structural compile changed original ownership/state");
    const auto envelope = decodeEnvelope(intent->bytes());
    checkIntent(!envelope.isDiff && envelope.sourceFormat.empty(),"structural authoring emitted snapshot/diff");
    const auto candidate = buildSceneCandidate(base,intent->bytes());
    if (inspect(*candidate->data()) != inspect(*oracle)) {
        std::cerr << "compiled structural data:\n"; candidate->data()->WriteToStream(std::cerr);
        std::cerr << "stock structural data:\n"; oracle->WriteToStream(std::cerr);
        checkIntent(false,"structural candidate differs from independent stock scene");
    }
    return intent;
}
// Whole writes that a later whole write of the same field replaces, beside
// dictionary-key edits that read the field between them, compile to the stock
// result: an edit that reads a field sees every write before it.
void rewrittenFieldsKeepTheirReads() {
    Fixture f;
    const SdfPath prim("/P"), attribute("/P.a");
    const auto dictionary = [](const char* key, int value) {
        VtDictionary result;
        result[key] = VtValue(value);
        return VtValue(result);
    };
    f.both([&](auto& data) { data.Set(prim,SdfFieldKeys->CustomData,dictionary("x",1)); });
    f.both([&](auto& data) { data.SetDictValueByKey(prim,SdfFieldKeys->CustomData,TfToken("v"),VtValue(5)); });
    f.both([&](auto& data) { data.Set(prim,SdfFieldKeys->CustomData,dictionary("w",4)); });
    f.both([&](auto& data) { data.Erase(prim,SdfFieldKeys->CustomData); });
    f.both([&](auto& data) { data.Set(prim,SdfFieldKeys->CustomData,dictionary("y",2)); });
    f.both([&](auto& data) { data.SetDictValueByKey(prim,SdfFieldKeys->CustomData,TfToken("z"),VtValue(3)); });
    f.both([&](auto& data) { data.Set(prim,SdfFieldKeys->Documentation,VtValue(std::string("first"))); });
    f.both([&](auto& data) { data.Set(attribute,SdfFieldKeys->Default,VtValue(1)); });
    f.both([&](auto& data) { data.Set(attribute,SdfFieldKeys->Default,VtValue(2)); });
    f.both([&](auto& data) { data.Erase(prim,SdfFieldKeys->Documentation); });
    f.both([&](auto& data) { data.Set(attribute,SdfFieldKeys->Default,VtValue(3)); });
    const auto envelope = decodeEnvelope(f.compile()->bytes());
    ordinary(envelope);
    checkIntent(envelope.fields.size() == 3,"rewritten fields did not collapse to one record each");
}
// One field written many times emits one record carrying its last value, and
// the operation and entry limits count emitted records, not buffered writes.
void repeatedFieldWritesCollapse() {
    Fixture f;
    constexpr int writes = 1000;
    for (int i = 1; i <= writes; ++i)
        f.both([&](auto& data) { data.Set(SdfPath("/P.a"),SdfFieldKeys->Default,VtValue(i)); });
    const auto intent = f.compile(); const auto envelope = decodeEnvelope(intent->bytes()); ordinary(envelope);
    checkIntent(envelope.fields.size() == 1 && envelope.samples.empty() && envelope.fields[0].section == 91 &&
                envelope.fields[0].name == "default" && envelope.fields[0].setOrder == writes &&
                decodeValues(envelope).fields == std::vector<VtValue>{VtValue(writes)},
                "repeated writes of one field did not compile to one record of the last value");
    checkIntent(intent->operations()->size() == size_t(writes) && intent->operations()->front().value == VtValue(1) &&
                intent->operations()->back().value == VtValue(writes),"the intent did not retain every original write");
    // 150 attributes each written 500 times: 75,000 writes, past the default
    // operation limit, compile to 150 records within every limit.
    constexpr int attributes = 150, rounds = 500;
    EnvelopeInput metadata; metadata.commands = {create(17,1,"P",SdfSpecTypePrim)};
    for (int i = 0; i < attributes; ++i)
        metadata.commands.push_back(create(1000+i,17,("a"+std::to_string(i)).c_str(),SdfSpecTypeAttribute));
    const auto base = buildSceneCandidate({},encodeEnvelope(metadata));
    auto operations = std::make_shared<std::vector<SdfOperation>>();
    operations->reserve(size_t(attributes) * rounds);
    const auto data = TfCreateRefPtr(new SdfData); data->CopyFrom(base->data());
    for (int round = 0; round < rounds; ++round)
        for (int i = 0; i < attributes; ++i) {
            SdfOperation op(SdfOperationKind::SetField); op.path = SdfPath("/P.a"+std::to_string(i));
            op.field = SdfFieldKeys->Default; op.value = VtValue(round * attributes + i);
            data->Set(op.path,op.field,op.value);
            operations->push_back(std::move(op));
        }
    checkIntent(operations->size() > FieldEditLimits{}.maxOperations,"the fixture does not cross the operation limit");
    const auto log = std::make_shared<const SdfOperationLog>(SdfOperationLog{data,operations});
    FieldEditLimits exact; exact.maxOperations = attributes; exact.entries.maxEntries = attributes;
    for (const auto& limits : {FieldEditLimits{},exact}) {
        const auto frame = compileFieldEdits(base,log,limits);
        const auto compiled = decodeEnvelope(frame->bytes());
        checkIntent(compiled.fields.size() == size_t(attributes) && frame->operations() == log->operations,
                    "a frame of repeated writes did not compile to one record per field");
        checkIntent(inspect(*buildSceneCandidate(base,frame->bytes())->data()) == inspect(*data),
                    "the collapsed frame differs from its writes applied in order");
    }
    FieldEditLimits operationLimit; operationLimit.maxOperations = attributes - 1;
    bool limited = false;
    try { (void)compileFieldEdits(base,log,operationLimit); }
    catch (const DecodeError& error) { limited = error.kind == ErrorKind::ResourceLimit; }
    checkIntent(limited,"a frame of more distinct fields than the operation limit compiled");
    FieldEditLimits entryLimit; entryLimit.entries.maxEntries = attributes - 1;
    bool refused = false;
    try { (void)compileFieldEdits(base,log,entryLimit); }
    catch (const DeltaEntryLimitExceeded& error) { refused = error.counts().setFields == uint64_t(attributes); }
    checkIntent(refused,"the entry limit did not count one set field per distinct field");
}
// Repeated writes to one sample key, one dictionary key and one metadata field
// each emit one record of the last value; the intent keeps every write.
void repeatedSampleAndMetadataWritesCollapse() {
    Fixture f;
    constexpr int writes = 100;
    for (int i = 1; i <= writes; ++i) {
        f.both([&](auto& data) { data.SetTimeSample(SdfPath("/P.a"),1,VtValue(i)); });
        f.both([&](auto& data) { data.SetDictValueByKey(SdfPath("/P"),SdfFieldKeys->CustomData,TfToken("k"),VtValue(i)); });
        f.both([&](auto& data) { data.Set(SdfPath("/P"),SdfFieldKeys->Documentation,VtValue(std::to_string(i))); });
    }
    const auto intent = f.compile(); const auto envelope = decodeEnvelope(intent->bytes()); ordinary(envelope);
    const auto values = decodeValues(envelope);
    checkIntent(envelope.samples.size() == 1 && envelope.samples[0].section == 91 && envelope.samples[0].time == 1 &&
                values.samples == std::vector<VtValue>{VtValue(writes)},"repeated sample writes did not collapse");
    checkIntent(envelope.fields.size() == 2,"repeated dictionary-key and metadata writes did not collapse");
    for (size_t i = 0; i < 2; ++i) {
        const auto& field = envelope.fields[i];
        const bool dictionary = field.name == "customData";
        checkIntent(field.section == 17 && (dictionary || field.name == "documentation") &&
                    field.setOrder == size_t(3 * writes - (dictionary ? 1 : 0)) &&
                    values.fields[i] == f.oracle->Get(SdfPath("/P"),TfToken(field.name)),
                    "a collapsed record lost its field's last value or write position");
    }
    checkIntent(intent->operations()->size() == size_t(3 * writes),"the intent did not retain every original write");
}
// Repeated whole-map writes count once when every generated sample record
// collapses; adding a sample key keeps its write in the operation count.
void repeatedWholeSampleMapsCollapse() {
    Fixture f;
    constexpr int writes = 100;
    for (int i = 1; i <= writes; ++i)
        f.both([&](auto& data) {
            data.Set(SdfPath("/P.a"),SdfDataTokens->TimeSamples,
                     VtValue(SdfTimeSampleMap{{0,VtValue(i)},{1,VtValue(i + 1)}}));
        });
    const auto log = recordedLog(*f.backing);
    FieldEditLimits limits; limits.maxOperations = 1;
    const auto intent = compileFieldEdits(f.base,log,limits);
    const auto envelope = decodeEnvelope(intent->bytes());
    checkIntent(envelope.samples.size() == 2 && envelope.fields.empty() &&
                intent->operations()->size() == size_t(writes),
                "whole-map rewrites did not count once or lost original operations");
    checkIntent(inspect(*buildSceneCandidate(f.base,intent->bytes())->data()) == inspect(*f.oracle),
                "collapsed whole-map rewrites differ from stock USD");
    f.both([](auto& data) {
        data.Set(SdfPath("/P.a"),SdfDataTokens->TimeSamples,
                 VtValue(SdfTimeSampleMap{{0,VtValue(100)},{1,VtValue(101)},{2,VtValue(102)}}));
    });
    const auto extended = recordedLog(*f.backing);
    bool limited = false;
    try { (void)compileFieldEdits(f.base,extended,limits); }
    catch (const DecodeError& error) { limited = error.kind == ErrorKind::ResourceLimit; }
    checkIntent(limited,"a whole-map write adding a sample key did not count as another operation");
}
// Writes to one field collapse across moves of the identity they address.
// Writes to an erased identity are dropped, and writes to the identity
// created at the same path collapse onto that new identity.
void fieldWritesAcrossStructure() {
    EnvelopeInput initial; initial.commands={create(2,1,"A",SdfSpecTypePrim),create(3,1,"B",SdfSpecTypePrim)};
    initial.fields={{2,0,"marker",{}},{3,1,"marker",{}}};
    const auto base=buildSceneCandidate({},encodeUsdEnvelope(initial,DecodedValues{{VtValue(2),VtValue(3)},{}}));
    auto backing=recordingBacking(base);
    auto oracle=TfCreateRefPtr(new SdfData); oracle->CopyFrom(base->data());
    const TfToken marker("marker");
    for (SdfAbstractData* data : {static_cast<SdfAbstractData*>(backing.operator->()),static_cast<SdfAbstractData*>(oracle.operator->())}) {
        data->Set(SdfPath("/A"),marker,VtValue(10));
        data->Set(SdfPath("/A"),marker,VtValue(11));
        data->MoveSpec(SdfPath("/A"),SdfPath("/C"));
        data->Set(SdfPath("/C"),marker,VtValue(12));
        data->Set(SdfPath("/C"),marker,VtValue(13));
        data->Set(SdfPath("/B"),marker,VtValue(20));
        data->EraseSpec(SdfPath("/B"));
        data->CreateSpec(SdfPath("/B"),SdfSpecTypePrim);
        data->Set(SdfPath("/B"),marker,VtValue(21));
        data->Set(SdfPath("/B"),marker,VtValue(22));
        data->Set(SdfPath::AbsoluteRootPath(),SdfChildrenKeys->PrimChildren,
                  VtValue(std::vector<TfToken>{TfToken("C"),TfToken("B")}));
    }
    const auto intent = structuralCompile(base,backing,oracle,sequentialIds());
    checkIntent(intent->operations()->size() == 11,"the intent did not retain every original operation");
    const auto candidate = buildSceneCandidate(base,intent->bytes());
    const auto created = candidate->sectionForPath(SdfPath("/B"));
    checkIntent(candidate->sectionForPath(SdfPath("/C")) == 2 && created > 3,"a write rebound a structural identity");
    const auto envelope = decodeEnvelope(intent->bytes()); const auto values = decodeValues(envelope);
    std::vector<std::pair<uint64_t,VtValue>> markers;
    for (size_t i = 0; i < envelope.fields.size(); ++i)
        if (envelope.fields[i].name == "marker") markers.emplace_back(envelope.fields[i].section,values.fields[i]);
    const std::vector<std::pair<uint64_t,VtValue>> expected{{2,VtValue(13)},{created,VtValue(22)}};
    checkIntent(markers == expected,"field writes did not collapse across a move or collapsed onto another identity");
}
void publicStructuralDraft() {
    // Public SdfLayer sample authoring requires declared attribute types; the
    // lower-level field fixture intentionally does not need schema metadata.
    EnvelopeInput typed;
    typed.fields = {{91,0,"typeName",{}},{203,1,"typeName",{}}};
    const auto base = buildSceneCandidate(initialScene(),encodeUsdEnvelope(typed,
        DecodedValues{{VtValue(TfToken("int")),VtValue(TfToken("int"))},{}}));
    const auto before = inspect(*base->data());
    auto backing = recordingBacking(base);
    auto oracle = TfCreateRefPtr(new SdfData); oracle->CopyFrom(base->data());
    const SdfFileFormatRefPtr capturedFormat = TfCreateRefPtr(new IntentLayerFormat(backing));
    const SdfFileFormatRefPtr stockFormat = TfCreateRefPtr(new IntentLayerFormat(oracle));
    auto captured = SdfLayer::CreateAnonymous("captured.intenttest",capturedFormat);
    auto stock = SdfLayer::CreateAnonymous("stock.intenttest",stockFormat);
    checkIntent(captured && stock && backing->pendingLocalEditCount() == 0,"public structural fixture changed backing at initialization");
    for (const auto& layer : {captured,stock}) {
        layer->SetField(SdfPath("/P.a"),SdfFieldKeys->Default,VtValue(81));
        const auto q = SdfPrimSpec::New(layer,"Q",SdfSpecifierDef);
        const auto leaf = SdfPrimSpec::New(q,"Leaf",SdfSpecifierDef);
        const auto property = SdfAttributeSpec::New(leaf,"answer",SdfValueTypeNames->Int);
        checkIntent(q && leaf && property,"public prim/attribute creation failed");
        property->SetDefaultValue(VtValue(42));
        layer->SetTimeSample(property->GetPath(),1,VtValue(43));
        SdfBatchNamespaceEdit move;
        move.Add(SdfPath("/P"),SdfPath("/Moved"));
        checkIntent(layer->Apply(move),"public parent namespace edit failed");
        SdfBatchNamespaceEdit rename;
        rename.Add(SdfPath("/Moved.a"),SdfPath("/Moved.renamed"));
        checkIntent(layer->Apply(rename),"public attribute namespace edit failed");
        layer->SetField(SdfPath("/Moved.renamed"),SdfFieldKeys->Default,VtValue(82));
        layer->SetTimeSample(SdfPath("/Moved.renamed"),2,VtValue(83));
        SdfBatchNamespaceEdit remove;
        remove.Add(SdfPath("/Moved.b"),SdfPath());
        checkIntent(layer->Apply(remove),"public property deletion failed");
    }
    const auto intent = structuralCompile(base,backing,oracle,sequentialIds());
    const auto candidate = buildSceneCandidate(base,intent->bytes());
    checkIntent(candidate->sectionForPath(SdfPath("/Moved")) == 17 &&
                candidate->sectionForPath(SdfPath("/Moved.renamed")) == 91 && candidate->pathForSection(203).IsEmpty(),
                "public namespace changes lost existing identities");
    std::set<uint64_t> identities{1,17,91};
    for (const char* path : {"/Q","/Q/Leaf","/Q/Leaf.answer"}) {
        const auto id = candidate->sectionForPath(SdfPath(path));
        checkIntent(id > 1 && id != 203 && identities.insert(id).second,"created public specs reused a retained/deleted identity");
    }
    checkIntent(candidate->sectionCount() == 6,"scratch/temporary specs leaked into public namespace result");
    bool created=false,moved=false,erased=false;
    for (const auto& operation : *intent->operations()) {
        created |= operation.kind == SdfOperationKind::CreateSpec;
        moved |= operation.kind == SdfOperationKind::MoveSpec;
        erased |= operation.kind == SdfOperationKind::EraseSpec;
    }
    checkIntent(created && moved && erased && inspect(*base->data()) == before,"public calls did not capture distinct structural operations");
}
// A captured log may create, edit and remove a spec in one host job. Its
// records addressing removed identities are dropped without clearing
// concurrentStructureSupported, whether the identity is fresh or pre-existing;
// only the removal compiles.
void capturedFreshIdentityRemoval() {
    const auto base = initialScene();
    const auto before = inspect(*base->data());
    auto backing = recordingBacking(base);
    auto oracle = TfCreateRefPtr(new SdfData); oracle->CopyFrom(base->data());
    const SdfFileFormatRefPtr capturedFormat = TfCreateRefPtr(new IntentLayerFormat(backing));
    const SdfFileFormatRefPtr stockFormat = TfCreateRefPtr(new IntentLayerFormat(oracle));
    auto captured = SdfLayer::CreateAnonymous("fresh.intenttest",capturedFormat);
    auto stock = SdfLayer::CreateAnonymous("fresh-stock.intenttest",stockFormat);
    for (const auto& layer : {captured,stock}) {
        SdfChangeBlock block;
        const auto prim = SdfPrimSpec::New(layer,"New",SdfSpecifierDef,"Scope");
        const auto value = SdfAttributeSpec::New(prim,"value",SdfValueTypeNames->Int);
        checkIntent(prim && value,"fresh prim/attribute creation failed");
        value->SetDefaultValue(VtValue(5));
        layer->RemoveRootPrim(prim);
    }
    const auto sealed = recordedLog(*backing);
    bool created = false, erased = false, set = false;
    for (const auto& operation : *sealed->operations) {
        created |= operation.kind == SdfOperationKind::CreateSpec;
        erased |= operation.kind == SdfOperationKind::EraseSpec;
        set |= operation.kind == SdfOperationKind::SetField && operation.path == SdfPath("/New.value");
    }
    checkIntent(created && erased && set && inspect(*oracle) == before,"fixture did not create, set and erase a fresh spec");
    const auto intent = compileFieldEdits(base,sealed,{},sequentialIds());
    const auto candidate = buildSceneCandidate(base,intent->bytes());
    checkIntent(inspect(*candidate->data()) == before && candidate->sectionCount() == base->sectionCount(),
                "fresh create-then-remove changed the compiled scene");
    const auto envelope = decodeEnvelope(intent->bytes());
    checkIntent(envelope.commands.empty() && envelope.fields.empty() && envelope.samples.empty(),
                "fresh create-then-remove emitted wire records");
    checkIntent(intent->concurrentStructureSupported(),"captured fresh identity removal vetoed concurrent structure");

    auto existing = recordingBacking(base);
    auto expected = TfCreateRefPtr(new SdfData); expected->CopyFrom(base->data());
    for (SdfAbstractData* data : {static_cast<SdfAbstractData*>(existing.operator->()),static_cast<SdfAbstractData*>(expected.operator->())}) {
        data->Set(SdfPath("/P.b"),SdfFieldKeys->Default,VtValue(9));
        data->EraseSpec(SdfPath("/P.b"));
        data->Set(SdfPath("/P"),SdfChildrenKeys->PropertyChildren,VtValue(std::vector<TfToken>{TfToken("a")}));
    }
    const auto removed = recordedLog(*existing);
    checkIntent(inspect(*removed->data) == inspect(*expected),"pre-existing removal fixture differs from stock");
    const auto result = compileFieldEdits(base,removed,{},sequentialIds());
    checkIntent(inspect(*buildSceneCandidate(base,result->bytes())->data()) == inspect(*expected),
                "pre-existing removal compiled to a different scene");
    const auto dropped = decodeEnvelope(result->bytes());
    checkIntent(result->concurrentStructureSupported(),"captured edit to a removed pre-existing identity was vetoed");
    checkIntent(dropped.commands.size() == 1 && dropped.commands[0].kind == Command::Kind::Delete &&
                dropped.commands[0].section == 203 && dropped.fields.empty() && dropped.samples.empty(),
                "captured records addressing a removed identity were emitted");
}
// Captured logs drop records inside a subtree they delete, but move-cycle
// scratch Creates and child-membership overrides stay unsupported.
void capturedStructuralVeto() {
    EnvelopeInput start;
    start.commands = {create(2,1,"P",SdfSpecTypePrim),create(3,2,"C",SdfSpecTypePrim),create(4,3,"A",SdfSpecTypePrim),
                      create(5,3,"B",SdfSpecTypePrim),create(6,1,"Q",SdfSpecTypePrim),create(7,1,"Outside",SdfSpecTypePrim)};
    const auto base = buildSceneCandidate({},encodeEnvelope(start));
    const auto tokens = [](std::initializer_list<const char*> names) {
        std::vector<TfToken> result; for (auto name : names) result.emplace_back(name); return VtValue(result);
    };
    auto subtree = recordingBacking(base);
    subtree->Set(SdfPath("/P/C"),TfToken("marker"),VtValue(1));
    subtree->Set(SdfPath("/P/C"),SdfChildrenKeys->PrimChildren,tokens({"B","A"}));
    subtree->MoveSpec(SdfPath("/Outside"),SdfPath("/P/C/A/Inside"));
    subtree->Set(SdfPath("/P/C/A/Inside"),TfToken("marker"),VtValue(2));
    subtree->MoveSpec(SdfPath("/P/C/B"),SdfPath("/Kept"));
    subtree->Set(SdfPath("/Kept"),TfToken("marker"),VtValue(3));
    for (const char* path : {"/P/C/A/Inside","/P/C/A","/P/C","/P"}) subtree->EraseSpec(SdfPath(path));
    subtree->Set(SdfPath::AbsoluteRootPath(),SdfChildrenKeys->PrimChildren,tokens({"Q","Kept"}));
    const auto sealed = recordedLog(*subtree);
    // The Deletes of P's subtree and of the moved-in Outside compile; B moved
    // out before the delete survives with its Move and field record.
    const auto captured = compileFieldEdits(base,sealed,{},sequentialIds());
    const auto deleted = decodeEnvelope(captured->bytes());
    std::set<uint64_t> deletedIds; bool keptMove = false;
    for (const auto& command : deleted.commands) {
        if (command.kind == Command::Kind::Delete) deletedIds.insert(command.section);
        keptMove |= command.kind == Command::Kind::Move && command.section == 5 && command.parent == 1 && command.name == "Kept";
    }
    checkIntent(captured->concurrentStructureSupported() && deleted.commands.size() == 3 && keptMove &&
                deletedIds == std::set<uint64_t>{2,7} && deleted.fields.size() == 1 && deleted.fields[0].section == 5 &&
                deleted.fields[0].name == "marker" && deleted.samples.empty(),
                "captured edits inside a deleted subtree were emitted or vetoed, or the moved-out survivor lost its records");
    checkIntent(buildSceneCandidate(base,captured->bytes())->sectionForPath(SdfPath("/Kept")) == 5,
                "moved-out survivor lost its identity");
    auto swap = recordingBacking(base);
    swap->MoveSpec(SdfPath("/Q"),SdfPath("/Temporary"));
    swap->MoveSpec(SdfPath("/Outside"),SdfPath("/Q"));
    swap->MoveSpec(SdfPath("/Temporary"),SdfPath("/Outside"));
    const auto cycle = compileFieldEdits(base,recordedLog(*swap),{},sequentialIds());
    const auto swapped = buildSceneCandidate(base,cycle->bytes());
    bool scratch = false;
    for (const auto& command : decodeEnvelope(cycle->bytes()).commands)
        scratch |= command.kind == Command::Kind::Create && swapped->pathForSection(command.section).IsEmpty();
    checkIntent(scratch && !cycle->concurrentStructureSupported(),"captured move-cycle scratch Create lost the veto");
    checkIntent(cycle->unsupportedStructure() == UnsupportedOperation::Reason::MoveCycle,
                "captured move-cycle veto names the wrong reason");
    auto emptied = recordingBacking(base);
    for (const char* path : {"/P/C/A","/P/C/B"}) emptied->EraseSpec(SdfPath(path));
    emptied->Set(SdfPath("/P/C"),SdfChildrenKeys->PrimChildren,tokens({}));
    bool rejected = false;
    try { (void)compileFieldEdits(base,recordedLog(*emptied),{},sequentialIds()); }
    catch (const UnsupportedOperation& error) { rejected = error.reason() == UnsupportedOperation::Reason::MultiUpdate; }
    checkIntent(rejected,"captured child-membership override admitted");
}
void replacementPreservesSurvivingChildren() {
    auto base = initialScene();
    EnvelopeInput children; children.commands = {create(303,17,"Child",SdfSpecTypePrim)};
    children.fields = {{303,0,"marker",{}}};
    base = buildSceneCandidate(base,encodeUsdEnvelope(children,DecodedValues{{VtValue(3030)},{}}));
    auto backing = recordingBacking(base);
    auto oracle = TfCreateRefPtr(new SdfData); oracle->CopyFrom(base->data());
    for (SdfAbstractData* data : {static_cast<SdfAbstractData*>(backing.operator->()),static_cast<SdfAbstractData*>(oracle.operator->())}) {
        data->EraseSpec(SdfPath("/P")); // One-spec operation leaves temporary orphans.
        checkIntent(data->HasSpec(SdfPath("/P.a")) && data->HasSpec(SdfPath("/P/Child")),"fixture erase unexpectedly removed subtree");
        data->CreateSpec(SdfPath("/P"),SdfSpecTypePrim);
        data->Set(SdfPath("/P"),SdfChildrenKeys->PropertyChildren,VtValue(std::vector<TfToken>{TfToken("a"),TfToken("b")}));
        data->Set(SdfPath("/P"),SdfChildrenKeys->PrimChildren,VtValue(std::vector<TfToken>{TfToken("Child")}));
        data->Set(SdfPath("/P"),TfToken("newValue"),VtValue(7));
    }
    const auto intent = structuralCompile(base,backing,oracle,sequentialIds());
    const auto candidate = buildSceneCandidate(base,intent->bytes());
    const auto replacement = candidate->sectionForPath(SdfPath("/P"));
    checkIntent(replacement > 1 && replacement != 17 && replacement != 91 && replacement != 203 && replacement != 303 &&
                candidate->pathForSection(17).IsEmpty(),"erase/create reused old parent identity");
    checkIntent(candidate->sectionForPath(SdfPath("/P.a")) == 91 && candidate->sectionForPath(SdfPath("/P.b")) == 203 &&
                candidate->sectionForPath(SdfPath("/P/Child")) == 303 && candidate->sectionCount() == 5,
                "replacement lost surviving children or retained scratch");
    checkIntent(!candidate->data()->Has(SdfPath("/P"),SdfFieldKeys->CustomData) &&
                candidate->data()->Get(SdfPath("/P/Child"),TfToken("marker")) == VtValue(3030),
                "replacement inherited old fields or erased child fields");
}
void movedSamplesAndDeletedValues() {
    const auto base = initialScene();
    auto backing = recordingBacking(base);
    auto oracle = TfCreateRefPtr(new SdfData); oracle->CopyFrom(base->data());
    for (SdfAbstractData* data : {static_cast<SdfAbstractData*>(backing.operator->()),static_cast<SdfAbstractData*>(oracle.operator->())}) {
        data->SetTimeSample(SdfPath("/P.a"),1,VtValue(11));
        data->MoveSpec(SdfPath("/P.a"),SdfPath("/P.tmp"));
        data->SetTimeSample(SdfPath("/P.tmp"),2,VtValue(12));
        data->MoveSpec(SdfPath("/P.b"),SdfPath("/P.a"));
        data->Set(SdfPath("/P.a"),SdfFieldKeys->TimeSamples,VtValue(SdfTimeSampleMap{{4,VtValue(24)}}));
        data->SetTimeSample(SdfPath("/P.a"),5,VtValue(25));
        data->MoveSpec(SdfPath("/P.tmp"),SdfPath("/P.b"));
        data->SetTimeSample(SdfPath("/P.b"),3,VtValue(13));
        data->Set(SdfPath("/P.b"),SdfFieldKeys->Default,VtValue(91));
        data->CreateSpec(SdfPath("/P.transient"),SdfSpecTypeAttribute);
        data->Set(SdfPath("/P.transient"),SdfFieldKeys->Default,VtValue(std::string(70000,'z')));
        data->EraseSpec(SdfPath("/P.transient"));
        data->Set(SdfPath("/P"),SdfChildrenKeys->PropertyChildren,VtValue(std::vector<TfToken>{TfToken("b"),TfToken("a")}));
    }
    const auto intent = structuralCompile(base,backing,oracle,sequentialIds());
    const auto candidate = buildSceneCandidate(base,intent->bytes());
    checkIntent(candidate->sectionForPath(SdfPath("/P.a")) == 203 && candidate->sectionForPath(SdfPath("/P.b")) == 91 &&
                candidate->sectionCount() == 4,"swap lost identities or preserved temporary spec");
    checkIntent(candidate->data()->Get(SdfPath("/P.b"),SdfFieldKeys->TimeSamples) ==
                VtValue(SdfTimeSampleMap{{0,VtValue(10)},{1,VtValue(11)},{2,VtValue(12)},{3,VtValue(13)}}) &&
                candidate->data()->Get(SdfPath("/P.a"),SdfFieldKeys->TimeSamples) ==
                VtValue(SdfTimeSampleMap{{4,VtValue(24)},{5,VtValue(25)}}),"sample/map operations rebound by final path rather than original identity");
    const auto envelope = decodeEnvelope(intent->bytes());
    for (const auto& sample : envelope.samples) {
        checkIntent(sample.section == (sample.time == 0 || sample.time >= 4 ? 203 : 91),"wire sample section lost move-time binding");
    }
    checkIntent(intent->operations()->size() == 13 && intent->operations()->at(10).value == VtValue(std::string(70000,'z')) &&
                intent->retainedBytes() >= 70000,"deleted transient payload/log was discarded from retained intent");
}
void structuralChildPolicies() {
    intentContext = "structural captured child policies";
    const auto empty = buildSceneCandidate({},encodeUsdEnvelope(EnvelopeInput{},DecodedValues{}));
    auto backing = recordingBacking(empty);
    auto oracle = TfCreateRefPtr(new SdfData); oracle->CopyFrom(empty->data());
    const std::vector<std::pair<const char*,SdfSpecType>> specs{
        {"",SdfSpecTypePrim},{".a",SdfSpecTypeAttribute},{".a[/Target.a]",SdfSpecTypeConnection},
        {".a.expression",SdfSpecTypeExpression},{".a.mapper[/T.rel[/U]]",SdfSpecTypeMapper},
        {".a.mapper[/T.rel[/U]].arg",SdfSpecTypeMapperArg},{".rel",SdfSpecTypeRelationship},
        {".rel[/Target]",SdfSpecTypeRelationshipTarget},{".rel[/Target].weight",SdfSpecTypeAttribute},
        {".rel.expression",SdfSpecTypeExpression},{"{v=}",SdfSpecTypeVariantSet},{"{v=x}",SdfSpecTypeVariant},
        {"{v=x}Child",SdfSpecTypePrim},{"{v=x}.inside",SdfSpecTypeAttribute},
        {"{v=x}{nested=}",SdfSpecTypeVariantSet},{"{v=x}{nested=y}",SdfSpecTypeVariant}};
    const auto tokens = [](std::initializer_list<const char*> names) {
        std::vector<TfToken> result; for (auto name:names) result.emplace_back(name); return VtValue(result);
    };
    const auto paths = [](std::initializer_list<const char*> names) {
        SdfPathVector result; for (auto name:names) result.emplace_back(name); return VtValue(result);
    };
    const auto populate = [&](auto& data) {
        for (size_t i=0;i<specs.size();++i) {
            const SdfPath path(std::string("/P")+specs[i].first);
            data.CreateSpec(path,specs[i].second);
            data.Set(path,TfToken("marker"),VtValue(int(i+10)));
        }
        data.Set(SdfPath("/"),SdfChildrenKeys->PrimChildren,tokens({"P"}));
        data.Set(SdfPath("/P"),SdfChildrenKeys->PropertyChildren,tokens({"rel","a"}));
        data.Set(SdfPath("/P"),SdfChildrenKeys->VariantSetChildren,tokens({"v"}));
        data.Set(SdfPath("/P.a"),SdfChildrenKeys->ConnectionChildren,paths({"/Target.a"}));
        data.Set(SdfPath("/P.a"),SdfChildrenKeys->ExpressionChildren,tokens({"expression"}));
        data.Set(SdfPath("/P.a"),SdfChildrenKeys->MapperChildren,paths({"/T.rel[/U]"}));
        data.Set(SdfPath("/P.a.mapper[/T.rel[/U]]"),SdfChildrenKeys->MapperArgChildren,tokens({"arg"}));
        data.Set(SdfPath("/P.rel"),SdfChildrenKeys->RelationshipTargetChildren,paths({"/Target"}));
        data.Set(SdfPath("/P.rel"),SdfChildrenKeys->ExpressionChildren,tokens({"expression"}));
        data.Set(SdfPath("/P.rel[/Target]"),SdfChildrenKeys->PropertyChildren,tokens({"weight"}));
        data.Set(SdfPath("/P{v=}"),SdfChildrenKeys->VariantChildren,tokens({"x"}));
        data.Set(SdfPath("/P{v=x}"),SdfChildrenKeys->PrimChildren,tokens({"Child"}));
        data.Set(SdfPath("/P{v=x}"),SdfChildrenKeys->PropertyChildren,tokens({"inside"}));
        data.Set(SdfPath("/P{v=x}"),SdfChildrenKeys->VariantSetChildren,tokens({"nested"}));
        data.Set(SdfPath("/P{v=x}{nested=}"),SdfChildrenKeys->VariantChildren,tokens({"y"}));
    };
    populate(*backing); populate(*oracle);
    const auto ids = sequentialIds();
    const auto first = structuralCompile(empty,backing,oracle,ids);
    const auto made = buildSceneCandidate(empty,first->bytes());
    std::vector<uint64_t> originalIds;
    for (const auto& spec:specs) originalIds.push_back(made->sectionForPath(SdfPath(std::string("/P")+spec.first)));
    checkIntent(std::set<uint64_t>(originalIds.begin(),originalIds.end()).size()==specs.size(),"child policy creation reused identities");
    backing = recordingBacking(made);
    // SdfAbstractData moves one spec at a time; move children first so this
    // explicitly exercises captured temporary orphan paths as public USD does.
    const auto move = [&](auto& data) {
        for (auto i=specs.rbegin();i!=specs.rend();++i)
            data.MoveSpec(SdfPath(std::string("/P")+i->first),SdfPath(std::string("/Q")+i->first));
        data.Set(SdfPath("/"),SdfChildrenKeys->PrimChildren,tokens({"Q"}));
        data.CreateSpec(SdfPath("/Q"),SdfSpecTypePrim); // Same type retains identity.
    };
    move(*backing); move(*oracle);
    const auto second = structuralCompile(made,backing,oracle,ids);
    const auto moved = buildSceneCandidate(made,second->bytes());
    for (size_t i=0;i<specs.size();++i) {
        const SdfPath path(std::string("/Q")+specs[i].first);
        checkIntent(moved->sectionForPath(path)==originalIds[i] && moved->data()->GetSpecType(path)==specs[i].second &&
                    moved->data()->Get(path,TfToken("marker"))==VtValue(int(i+10)),"moved child policy lost identity/type/field");
    }
    checkIntent(moved->sectionCount()==specs.size()+1,"child policy normalization leaked scratch specs");
}
void structuralLatestSubtreeReplay() {
    for (bool move : {false,true}) {
        intentContext = move?"standalone latest-subtree move":"standalone latest-subtree delete";
        EnvelopeInput initial;
        initial.commands={create(2,1,"P",SdfSpecTypePrim),create(3,2,"Child",SdfSpecTypePrim),create(4,1,"Other",SdfSpecTypePrim)};
        initial.fields={{3,1,"marker",{}}};
        const auto base=buildSceneCandidate({},encodeUsdEnvelope(initial,DecodedValues{{VtValue(11)},{}}));
        auto backing=recordingBacking(base);
        auto oracle=TfCreateRefPtr(new SdfData); oracle->CopyFrom(base->data());
        const auto mutate=[&](auto& data) {
            if (move) {
                data.MoveSpec(SdfPath("/P/Child"),SdfPath("/Q/Child"));
                data.MoveSpec(SdfPath("/P"),SdfPath("/Q"));
            } else {
                data.EraseSpec(SdfPath("/P/Child")); data.EraseSpec(SdfPath("/P"));
            }
            data.Set(SdfPath("/"),SdfChildrenKeys->PrimChildren,VtValue(move?
                std::vector<TfToken>{TfToken("Q"),TfToken("Other")}:std::vector<TfToken>{TfToken("Other")}));
        };
        mutate(*backing); mutate(*oracle);
        const auto intent=structuralCompile(base,backing,oracle,sequentialIds());
        checkIntent(intent->concurrentStructureSupported(),"ordinary subtree intent unexpectedly requires service extension");
        const auto bytes=intent->bytes();
        const auto original=intent->operations(); const auto before=inspect(*base->data());
        for (unsigned variant=0;variant<4;++variant) {
            EnvelopeInput peer; DecodedValues values;
            if (variant==0) { peer.fields={{3,1,"marker",{}}}; values.fields={VtValue(99)}; }
            else if (variant==1 || variant==2) peer.commands={create(300,variant==1?1:2,"Peer",SdfSpecTypePrim)};
            else { Command command{}; command.kind=Command::Kind::Move; command.section=3;
                command.parent=1; command.oldParent=2; command.name="Escaped"; peer.commands={command}; }
            const auto current=buildSceneCandidate(base,encodeUsdEnvelope(peer,values));
            const auto saved=inspect(*current->data());
            const auto applied=buildSceneCandidate(current,intent->bytes());
            checkIntent(inspect(*base->data())==before && inspect(*current->data())==saved && intent->operations()==original,
                        "latest-subtree replay changed retained input owners/state");
            checkIntent(intent->bytes()==bytes,"latest-subtree replay rewrote accepted bytes");
            if (variant==0 && move)
                checkIntent(applied->data()->Get(SdfPath("/Q/Child"),TfToken("marker"))==VtValue(99),
                            "observed peer field lost across recursive move");
            if (variant==1)
                checkIntent(applied->sectionForPath(SdfPath("/Peer"))==300 && applied->sectionForPath(SdfPath("/Other"))==4,
                            "unrelated peer namespace lost across normalization");
            if (variant==2)
                checkIntent(move ? applied->sectionForPath(SdfPath("/Q/Peer"))==300 : applied->pathForSection(300).IsEmpty(),
                            "latest subtree did not include new peer child");
            if (variant==3)
                checkIntent(applied->sectionForPath(SdfPath("/Escaped"))==3 &&
                            applied->data()->Get(SdfPath("/Escaped"),TfToken("marker"))==VtValue(11),
                            "latest subtree changed a peer-moved-out child");
        }
    }
}
// A log over the log operation limit fails before any per-operation work, so
// no section ID is drawn for its Create; a log within it whose collapsed
// writes fit the operation limit compiles.
void logOperationLimit() {
    checkIntent(FieldEditLimits{}.maxLogOperations > FieldEditLimits{}.maxOperations,
                "the log operation limit leaves collapse no room");
    const auto base = initialScene();
    auto backing = recordingBacking(base);
    backing->CreateSpec(SdfPath("/New"),SdfSpecTypePrim);
    backing->Set(SdfPath::AbsoluteRootPath(),SdfChildrenKeys->PrimChildren,VtValue(std::vector<TfToken>{TfToken("P"),TfToken("New")}));
    constexpr int writes = 10;
    for (int i = 1; i <= writes; ++i) backing->Set(SdfPath("/P.a"),SdfFieldKeys->Default,VtValue(i));
    const auto sealed = recordedLog(*backing); const auto size = sealed->operations->size();
    unsigned calls = 0;
    const auto counted = std::make_shared<SectionIdAllocator>(4096,64,[&] { return uint64_t(5000) + calls++; });
    FieldEditLimits limits; limits.maxLogOperations = size - 1; limits.maxOperations = size - (writes - 1);
    std::string message;
    try { (void)compileFieldEdits(base,sealed,limits,counted); }
    catch (const DecodeError& error) { if (error.kind == ErrorKind::ResourceLimit) message = error.what(); }
    checkIntent(message.find("log operation limit") != std::string::npos && calls == 0,
                "a log over the log operation limit was not refused before any work");
    reject(base,sealed,ErrorKind::ResourceLimit,limits);
    limits.maxLogOperations = size;
    const auto intent = compileFieldEdits(base,sealed,limits,counted);
    checkIntent(calls > 0 && intent->operations() == sealed->operations &&
                buildSceneCandidate(base,intent->bytes())->sectionForPath(SdfPath("/New")) != 0,
                "a log within the log operation limit whose collapsed writes fit did not compile");
    limits.maxOperations = size - writes;
    reject(base,sealed,ErrorKind::ResourceLimit,limits);
}
void structuralAllocatorAndLimits() {
    const auto base = initialScene();
    auto backing = recordingBacking(base);
    backing->CreateSpec(SdfPath("/New"),SdfSpecTypePrim);
    backing->Set(SdfPath::AbsoluteRootPath(),SdfChildrenKeys->PrimChildren,VtValue(std::vector<TfToken>{TfToken("P"),TfToken("New")}));
    const auto sealed = recordedLog(*backing); const auto before = inspect(*base->data());
    unsigned calls=0;
    const auto colliding = std::make_shared<SectionIdAllocator>(4096,4,[&] { ++calls; return uint64_t(17); });
    bool rejected=false;
    try { (void)compileFieldEdits(base,sealed,{},colliding); }
    catch(const DecodeError& error) { rejected=error.kind==ErrorKind::ResourceLimit; }
    checkIntent(rejected && calls==4 && inspect(*base->data())==before,"allocator collision loop unbounded or changed base");
    FieldEditLimits limits; limits.maxStructuralCommands=0;
    reject(base,sealed,ErrorKind::ResourceLimit,limits);
    limits={}; limits.maxSectionIds=1;
    reject(base,sealed,ErrorKind::ResourceLimit,limits);
    std::vector<uint64_t> generated;
    const auto ids=std::make_shared<SectionIdAllocator>(4096,64,[&] {
        const auto id=uint64_t(1000)+generated.size(); generated.push_back(id); return id;
    });
    // A final snapshot mismatch occurs after structural preparation allocates
    // IDs. Rejection preserves those consumed identities in allocator history.
    auto mismatched=std::make_shared<SdfOperationLog>(*sealed);
    auto wrongSnapshot=TfCreateRefPtr(new SdfData); wrongSnapshot->CopyFrom(base->data()); mismatched->data=wrongSnapshot;
    bool malformed=false;
    try { (void)compileFieldEdits(base,mismatched,{},ids); }
    catch (const DecodeError& error) { malformed=error.kind==ErrorKind::Malformed; }
    checkIntent(malformed && !generated.empty() && inspect(*base->data())==before,"late preparation rejection did not exercise allocator preservation");
    const auto burned=generated;
    const auto one=compileFieldEdits(base,sealed,{},ids);
    const auto two=compileFieldEdits(base,sealed,{},ids);
    const auto first=buildSceneCandidate(base,one->bytes())->sectionForPath(SdfPath("/New"));
    const auto second=buildSceneCandidate(base,two->bytes())->sectionForPath(SdfPath("/New"));
    checkIntent(first!=second && first>1 && second>1 && std::find(burned.begin(),burned.end(),first)==burned.end() &&
                std::find(burned.begin(),burned.end(),second)==burned.end(),"independent preparations reused new or burned section identity");
}

// Seeded public namespace edits of nested prims (create, subtree delete,
// rename, reparent, reorder, sibling swap and attribute values), authored
// through SdfLayer on a recording backing and on an independent stock layer.
// Each turn compiles to exactly the stock result; with the compile check on,
// the footprint compilation also matches the whole-scene one. Every seed runs
// 24 turns and continues, up to a bound, until a turn compiles a move cycle,
// so each seed covers structural turns and a scratch-identity plan.
void nestedStructuralSeeds(uint64_t seed) {
    std::mt19937_64 random(seed ^ 0x6e65737465647374ull);
    EnvelopeInput initial;
    initial.commands = {create(2,1,"A",SdfSpecTypePrim),create(3,2,"B",SdfSpecTypePrim),create(4,1,"C",SdfSpecTypePrim),
                        create(5,3,"v",SdfSpecTypeAttribute)};
    initial.fields = {{5,0,"typeName",{}},{5,1,"default",{}},{2,2,"specifier",{}},{3,3,"specifier",{}},{4,4,"specifier",{}}};
    auto base = buildSceneCandidate({},encodeUsdEnvelope(initial,DecodedValues{{VtValue(TfToken("double")),VtValue(1.0),
        VtValue(SdfSpecifierDef),VtValue(SdfSpecifierDef),VtValue(SdfSpecifierDef)},{}}));
    const auto ids = sequentialIds(20000);
    unsigned fresh = 0, compiled = 0, cycles = 0;
    for (unsigned step = 0; step < 24 || (cycles == 0 && step < 512); ++step) {
        intentContext = "nested structural seed=" + std::to_string(seed) + " step=" + std::to_string(step);
        auto backing = recordingBacking(base);
        auto oracle = TfCreateRefPtr(new SdfData); oracle->CopyFrom(base->data());
        const SdfFileFormatRefPtr capturedFormat = TfCreateRefPtr(new IntentLayerFormat(backing));
        const SdfFileFormatRefPtr stockFormat = TfCreateRefPtr(new IntentLayerFormat(oracle));
        auto captured = SdfLayer::CreateAnonymous("captured.intenttest",capturedFormat);
        auto stock = SdfLayer::CreateAnonymous("stock.intenttest",stockFormat);
        std::vector<SdfPath> prims;
        for (const auto& [path, record] : inspect(*base->data()))
            if (record.type == SdfSpecTypePrim) prims.push_back(path);
        const auto pick = [&]() { return prims.empty() ? SdfPath::AbsoluteRootPath() : prims[random() % prims.size()]; };
        bool structural = false;
        const auto apply = [&](const SdfBatchNamespaceEdit& edit) {
            SdfNamespaceEditDetailVector why;
            if (stock->CanApply(edit) != SdfNamespaceEditDetail::Okay) return;
            checkIntent(captured->Apply(edit) && stock->Apply(edit),"seeded namespace edit failed");
            structural = true;
        };
        for (unsigned count = 1 + unsigned(random() % 3), k = 0; k < count; ++k) {
            switch (const auto choice = random() % 10; choice < 3 ? 0 : choice - 2) {
            case 0: {
                const auto parent = random() % 4 ? pick() : SdfPath::AbsoluteRootPath();
                if (!parent.IsAbsoluteRootPath() && !stock->GetPrimAtPath(parent)) break; // Moved or deleted this turn.
                const auto name = "N" + std::to_string(fresh++);
                const bool attribute = random() % 2;
                const double value = double(random() % 100);
                for (const auto& layer : {captured,stock}) {
                    const auto prim = parent.IsAbsoluteRootPath() ? SdfPrimSpec::New(layer,name,SdfSpecifierDef) :
                        SdfPrimSpec::New(layer->GetPrimAtPath(parent),name,SdfSpecifierDef);
                    checkIntent(bool(prim),"seeded prim creation failed");
                    if (attribute) SdfAttributeSpec::New(prim,"v",SdfValueTypeNames->Double)->SetDefaultValue(VtValue(value));
                }
                structural = true;
                break;
            }
            case 1: {
                if (prims.empty()) break;
                SdfBatchNamespaceEdit edit; edit.Add(pick(),SdfPath()); apply(edit); break;
            }
            case 2: {
                if (prims.empty()) break;
                const auto from = pick();
                SdfBatchNamespaceEdit edit; edit.Add(from,from.ReplaceName(TfToken("R" + std::to_string(fresh++))));
                apply(edit); break;
            }
            case 3: {
                if (prims.empty()) break;
                const auto from = pick(), parent = random() % 3 ? pick() : SdfPath::AbsoluteRootPath();
                SdfBatchNamespaceEdit edit; edit.Add(from,parent.AppendChild(TfToken("M" + std::to_string(fresh++))));
                apply(edit); break;
            }
            case 4: {
                if (prims.empty()) break;
                const auto child = pick();
                SdfBatchNamespaceEdit edit; edit.Add(SdfNamespaceEdit::Reorder(child,int(random() % 4)));
                apply(edit); break;
            }
            case 5: {
                // Swap two siblings through a temporary name: a move cycle.
                const auto first = pick();
                if (first.IsAbsoluteRootPath()) break;
                const auto parent = first.GetParentPath();
                const auto siblings = stock->GetField(parent,SdfChildrenKeys->PrimChildren).GetWithDefault<std::vector<TfToken>>();
                if (siblings.size() < 2) break;
                const auto second = parent.AppendChild(siblings[random() % siblings.size()]);
                if (second == first) break;
                const auto temporary = parent.AppendChild(TfToken("T" + std::to_string(fresh++)));
                for (const auto& [from, to] : {std::pair{first,temporary},std::pair{second,first},std::pair{temporary,second}}) {
                    SdfBatchNamespaceEdit edit; edit.Add(from,to); apply(edit);
                }
                break;
            }
            default: {
                const auto prim = pick();
                const auto attribute = prim.AppendProperty(TfToken("v"));
                if (!stock->HasSpec(attribute)) break;
                const VtValue value(double(random() % 100));
                captured->SetField(attribute,SdfFieldKeys->Default,value); stock->SetField(attribute,SdfFieldKeys->Default,value);
                break;
            }
            }
        }
        if (!structural || fieldOnly(*recordedLog(*backing)->operations)) continue;
        const auto intent = structuralCompile(base,backing,oracle,ids);
        ++compiled;
        cycles += intent->unsupportedStructure() == UnsupportedOperation::Reason::MoveCycle;
        base = buildSceneCandidate(base,intent->bytes());
    }
    std::cerr << "nested_structural_turns=" << compiled << " move_cycles=" << cycles << '\n';
    intentContext = "nested structural seed=" + std::to_string(seed);
    checkIntent(compiled > 0 && cycles > 0,"seeded nested history compiled no structural turn or no move cycle");
}
void structuralSeedHistory(uint64_t seed) {
    std::mt19937_64 random(seed);
    uint64_t fingerprint=1469598103934665603ULL;
    auto mix=[&](uint64_t value) { fingerprint^=value; fingerprint*=1099511628211ULL; };
    EnvelopeInput initial; initial.commands={create(2,1,"A",SdfSpecTypePrim),create(3,1,"B",SdfSpecTypePrim)};
    initial.fields={{2,0,"marker",{}},{3,1,"marker",{}}};
    auto base=buildSceneCandidate({},encodeUsdEnvelope(initial,DecodedValues{{VtValue(2),VtValue(3)},{}}));
    struct Model { uint64_t id; int marker; };
    std::map<std::string,Model> model{{"A",{2,2}},{"B",{3,3}}};
    std::vector<std::string> order{"A","B"}; std::set<uint64_t> ever{1,2,3};
    const auto ids=sequentialIds(10000);
    auto oracle=TfCreateRefPtr(new SdfData); oracle->CopyFrom(base->data());
    std::cerr << "structural_intent_seed=" << seed << '\n';
    for(unsigned step=0;step<32;++step) {
        intentContext="structural seed="+std::to_string(seed)+" step="+std::to_string(step);
        auto backing=recordingBacking(base);
        const unsigned action=order.empty()?0:unsigned(random()%6);
        const size_t selected=order.empty()?0:size_t(random()%order.size());
        const std::string name=order.empty()?"":order[selected], fresh="N"+std::to_string(step);
        const int value=int(random()%2001)-1000; mix(action); mix(selected); mix(uint32_t(value));
        auto both=[&](auto mutation) { mutation(*backing); mutation(*oracle); };
        if(action==0) {
            both([&](auto& data) { data.CreateSpec(SdfPath("/"+fresh),SdfSpecTypePrim); data.Set(SdfPath("/"+fresh),TfToken("marker"),VtValue(value)); });
            model[fresh]={0,value}; order.push_back(fresh);
        } else if(action==1) {
            both([&](auto& data) { data.Set(SdfPath("/"+name),TfToken("marker"),VtValue(value)); }); model.at(name).marker=value;
        } else if(action==2) {
            both([&](auto& data) { data.MoveSpec(SdfPath("/"+name),SdfPath("/"+fresh)); });
            model[fresh]=model.at(name); model.erase(name); order[selected]=fresh;
        } else if(action==3) {
            both([&](auto& data) { data.EraseSpec(SdfPath("/"+name)); });
            model.erase(name); order.erase(order.begin()+selected);
        } else if(action==4) {
            both([&](auto& data) { data.EraseSpec(SdfPath("/"+name)); data.CreateSpec(SdfPath("/"+name),SdfSpecTypePrim);
                                  data.Set(SdfPath("/"+name),TfToken("marker"),VtValue(value)); });
            model[name]={0,value};
        } else if(order.size()>1) {
            const size_t other=(selected+1)%order.size(); const auto otherName=order[other];
            both([&](auto& data) { data.MoveSpec(SdfPath("/"+name),SdfPath("/Temporary"));
                                  data.MoveSpec(SdfPath("/"+otherName),SdfPath("/"+name));
                                  data.MoveSpec(SdfPath("/Temporary"),SdfPath("/"+otherName)); });
            std::swap(model.at(name),model.at(otherName));
            std::swap(order[selected],order[other]);
        }
        // Stock Sdf erases a child list whose last child goes rather than
        // authoring it empty; an explicitly empty list is a refused membership
        // override (capturedStructuralVeto).
        std::vector<TfToken> children; for(const auto& item:order) children.emplace_back(item);
        const VtValue childValue=children.empty()?VtValue():VtValue(children);
        both([&](auto& data) {
            if(children.empty()) data.Erase(SdfPath::AbsoluteRootPath(),SdfChildrenKeys->PrimChildren);
            else data.Set(SdfPath::AbsoluteRootPath(),SdfChildrenKeys->PrimChildren,childValue);
        });
        const auto intent=structuralCompile(base,backing,oracle,ids);
        auto candidate=buildSceneCandidate(base,intent->bytes());
        checkIntent(candidate->sectionCount()==model.size()+1,"seeded structural graph retained scratch or lost specs");
        for(auto& item:model) {
            const SdfPath path("/"+item.first); const auto actual=candidate->sectionForPath(path);
            if(!item.second.id) { checkIntent(actual>1 && ever.insert(actual).second,"seeded create reused historical ID"); item.second.id=actual; }
            else checkIntent(actual==item.second.id,"seeded mutation rebound surviving identity");
            checkIntent(candidate->data()->GetSpecType(path)==SdfSpecTypePrim &&
                        candidate->data()->Get(path,TfToken("marker"))==VtValue(item.second.marker),"seeded explicit path/value oracle differs");
            mix(actual); mix(uint32_t(item.second.marker));
        }
        checkIntent(candidate->data()->Get(SdfPath::AbsoluteRootPath(),SdfChildrenKeys->PrimChildren)==childValue,
                    "seeded child order differs from independently maintained order");
        base=std::move(candidate);
    }
    std::cerr << "structural_intent_fingerprint=" << fingerprint << '\n';
}

} // namespace

// A service may store a Reorder verbatim as the parent's whole children list,
// so a changed order in one child list names every child of the parent.
void parentWideReorderEmission() {
    auto base = initialScene();
    EnvelopeInput children; children.commands = {create(303,17,"A",SdfSpecTypePrim),create(304,17,"B",SdfSpecTypePrim)};
    base = buildSceneCandidate(base,encodeEnvelope(children));
    auto backing = recordingBacking(base);
    backing->Set(SdfPath("/P"),SdfChildrenKeys->PrimChildren,VtValue(std::vector<TfToken>{TfToken("B"),TfToken("A")}));
    const auto intent = compileFieldEdits(base,recordedLog(*backing),{},sequentialIds());
    checkIntent(intent->planned(),"a child-order edit skipped namespace planning");
    const auto envelope = decodeEnvelope(intent->bytes());
    checkIntent(envelope.commands.size() == 1 && envelope.commands[0].kind == Command::Kind::Reorder &&
                envelope.commands[0].section == 17 && envelope.commands[0].children == std::vector<uint64_t>{304,303,91,203},
                "reorder did not list the parent's complete children");
    const auto candidate = buildSceneCandidate(base,intent->bytes());
    checkIntent(candidate->sectionForPath(SdfPath("/P.a")) == 91 && candidate->sectionForPath(SdfPath("/P.b")) == 203 &&
                candidate->data()->Get(SdfPath("/P"),SdfChildrenKeys->PrimChildren) ==
                    VtValue(std::vector<TfToken>{TfToken("B"),TfToken("A")}),
                "prim reorder hid the parent's properties or lost the order");
    auto append = recordingBacking(base);
    append->CreateSpec(SdfPath("/P/C"),SdfSpecTypePrim);
    append->Set(SdfPath("/P"),SdfChildrenKeys->PrimChildren,
                VtValue(std::vector<TfToken>{TfToken("A"),TfToken("B"),TfToken("C")}));
    const auto appended = decodeEnvelope(compileFieldEdits(base,recordedLog(*append),{},sequentialIds())->bytes());
    checkIntent(appended.commands.size() == 1 && appended.commands[0].kind == Command::Kind::Create,
                "structural append emitted a reorder");
    // A peer rename leaves 304 as a shadowed duplicate of A; it owns no spec but stays listed.
    EnvelopeInput peer; Command rename{}; rename.kind = Command::Kind::Move; rename.section = 304;
    rename.oldParent = 17; rename.parent = 17; rename.name = "A";
    peer.commands = {rename,create(305,17,"C",SdfSpecTypePrim)};
    const auto shadowed = buildSceneCandidate(base,encodeEnvelope(peer));
    checkIntent(shadowed->pathForSection(304).IsEmpty(),"fixture did not shadow the duplicate sibling");
    auto swap = recordingBacking(shadowed);
    swap->Set(SdfPath("/P"),SdfChildrenKeys->PrimChildren,VtValue(std::vector<TfToken>{TfToken("C"),TfToken("A")}));
    const auto kept = compileFieldEdits(shadowed,recordedLog(*swap),{},sequentialIds());
    const auto listed = decodeEnvelope(kept->bytes());
    checkIntent(listed.commands.size() == 1 && listed.commands[0].kind == Command::Kind::Reorder &&
                listed.commands[0].children == std::vector<uint64_t>{305,303,91,304,203},
                "reorder omitted the shadowed sibling or moved it from its stored position");
    const auto reordered = buildSceneCandidate(shadowed,kept->bytes());
    EnvelopeInput removeOwner; Command erase{}; erase.kind = Command::Kind::Delete; erase.section = 303; erase.parent = 17;
    removeOwner.commands = {erase};
    const auto revealed = buildSceneCandidate(reordered,encodeEnvelope(removeOwner));
    checkIntent(revealed->sectionForPath(SdfPath("/P/A")) == 304 &&
                revealed->data()->Get(SdfPath("/P"),SdfChildrenKeys->PrimChildren) ==
                    VtValue(std::vector<TfToken>{TfToken("C"),TfToken("A")}),
                "shadowed sibling was not revealed after its owner's delete");
}
// A changed order in a non-prim list still names every child of the parent;
// restating the current order emits nothing.
void completeReorderEmission() {
    auto base = initialScene();
    EnvelopeInput children; children.commands = {create(303,17,"A",SdfSpecTypePrim),create(304,17,"B",SdfSpecTypePrim)};
    base = buildSceneCandidate(base,encodeEnvelope(children));
    auto properties = recordingBacking(base);
    properties->Set(SdfPath("/P"),SdfChildrenKeys->PropertyChildren,VtValue(std::vector<TfToken>{TfToken("b"),TfToken("a")}));
    const auto intent = compileFieldEdits(base,recordedLog(*properties),{},sequentialIds());
    const auto envelope = decodeEnvelope(intent->bytes());
    checkIntent(envelope.commands.size() == 1 && envelope.commands[0].kind == Command::Kind::Reorder &&
                envelope.commands[0].section == 17 && envelope.commands[0].children == std::vector<uint64_t>{303,304,203,91},
                "property reorder did not list the parent's complete children");
    auto same = recordingBacking(base);
    same->Set(SdfPath("/P"),SdfChildrenKeys->PrimChildren,VtValue(std::vector<TfToken>{TfToken("A"),TfToken("B")}));
    const auto unchanged = decodeEnvelope(compileFieldEdits(base,recordedLog(*same),{},sequentialIds())->bytes());
    checkIntent(unchanged.commands.empty() && unchanged.fields.empty(),"an unchanged child order emitted a reorder");
}
// The refusal compiling `log` against `base` under `entries`, or none when it compiles.
std::optional<DeltaEntryLimitExceeded> entryRefusal(const WireScenePtr& base,const std::shared_ptr<const SdfOperationLog>& log,
                                                    const DeltaEntryLimits& entries) {
    FieldEditLimits limits; limits.entries = entries;
    try { (void)compileFieldEdits(base,log,limits,sequentialIds()); }
    catch (const DeltaEntryLimitExceeded& refused) {
        checkIntent(refused.kind == ErrorKind::ResourceLimit,"an entry refusal is not a resource limit");
        return refused;
    } catch (const DecodeError& error) { checkIntent(false,error.what()); }
    return std::nullopt;
}
void sameCounts(const DeltaEntryLimitExceeded& refused,DeltaEntryLimitExceeded::Reason reason,
                std::array<uint64_t,5> expected) {
    const auto& c = refused.counts();
    checkIntent(refused.reason() == reason &&
                std::array<uint64_t,5>{c.structCommands,c.setFields,c.timeSamples,c.limit,c.actual} == expected,
                "entry refusal reported other counts than the service's ErrorInfo would");
}
SdfTimeSampleMap integerSamples(int first,int count) {
    SdfTimeSampleMap result;
    for (int i = 0; i < count; ++i) result.emplace(double(first+i),VtValue(i));
    return result;
}
// Struct commands, set fields and time samples count together against one
// limit, exactly at it and one past it, with the counts the service reports.
void deltaEntryLimits() {
    const FieldEditLimits defaults;
    checkIntent(defaults.entries.maxEntries == 99999 && defaults.entries.maxChildrenListEntries == 100000,
                "entry limit defaults differ from the service's");
    const auto base = initialScene();
    const auto log = capturedLog(base,[](SdfAbstractData& data) {
        data.CreateSpec(SdfPath("/X"),SdfSpecTypePrim);
        data.Set(SdfPath::AbsoluteRootPath(),SdfChildrenKeys->PrimChildren,
                 VtValue(std::vector<TfToken>{TfToken("P"),TfToken("X")}));
        data.Set(SdfPath("/X"),SdfFieldKeys->Specifier,VtValue(SdfSpecifierDef));
        data.Set(SdfPath("/P.a"),SdfFieldKeys->Default,VtValue(5));
        // Key 0 stays, so the whole map is one sample record per key.
        data.Set(SdfPath("/P.a"),SdfFieldKeys->TimeSamples,VtValue(integerSamples(0,2)));
    });
    checkIntent(!entryRefusal(base,log,{5,100000}),"an update exactly at the entry limit was refused");
    const auto compiled = decodeEnvelope(compileFieldEdits(base,log,{},sequentialIds())->bytes());
    checkIntent(compiled.commands.size() == 1 && compiled.fields.size() == 2 && compiled.samples.size() == 2,
                "fixture compiled other entries than it counts");
    const auto refused = entryRefusal(base,log,{4,100000});
    checkIntent(bool(refused),"an update one entry past the limit was not refused");
    sameCounts(*refused,DeltaEntryLimitExceeded::Reason::DeltaTooLarge,{1,2,2,4,5});
}
// A whole-map timeSamples write is one sample record per key plus one erasure
// per observed key it drops; the service limit applies at its default. The
// maps spread over 1,000 attributes, since one update's samples of a single
// attribute cost work quadratic in their count.
void wholeMapSampleEntryLimit() {
    constexpr int attributes = 1000;
    EnvelopeInput metadata;
    metadata.commands = {create(17,1,"P",SdfSpecTypePrim)};
    for (int i = 0; i < attributes; ++i)
        metadata.commands.push_back(create(1000+i,17,("a"+std::to_string(i)).c_str(),SdfSpecTypeAttribute));
    metadata.samples = {{1000,0,{}}}; // /P.a0 holds the sample key 0.
    const auto base = buildSceneCandidate({},encodeUsdEnvelope(metadata,DecodedValues{{},{VtValue(-1)}}));
    // Every attribute gets keys 0..99 from `first`, the last `lastCount` keys.
    const auto maps = [&](int first,int lastCount) {
        return capturedLog(base,[&](SdfAbstractData& data) {
            for (int i = 0; i < attributes; ++i)
                data.Set(SdfPath("/P.a"+std::to_string(i)),SdfFieldKeys->TimeSamples,
                         VtValue(integerSamples(i ? 0 : first,i+1 == attributes ? lastCount : 100)));
        });
    };
    checkIntent(!entryRefusal(base,maps(0,99),{}),"99,999 sample records were refused at the default limit");
    const auto refused = entryRefusal(base,maps(0,100),{});
    checkIntent(bool(refused),"100,000 sample records were not refused at the default limit");
    sameCounts(*refused,DeltaEntryLimitExceeded::Reason::DeltaTooLarge,{0,0,100000,99999,100000});
    // Dropping the observed key 0 of /P.a0 adds its erasure to 99,999 written keys.
    const auto erased = entryRefusal(base,maps(1,99),{});
    checkIntent(bool(erased),"a map write's erasure was not counted");
    sameCounts(*erased,DeltaEntryLimitExceeded::Reason::DeltaTooLarge,{0,0,100000,99999,100000});
}
// A Reorder is one struct command; the children list it names, every child of
// the parent across its lists, has its own limit outside the entry count.
void childrenListLimit() {
    auto base = initialScene();
    EnvelopeInput children;
    children.commands = {create(303,17,"A",SdfSpecTypePrim),create(304,17,"B",SdfSpecTypePrim),create(305,17,"C",SdfSpecTypePrim)};
    base = buildSceneCandidate(base,encodeEnvelope(children));
    const auto log = capturedLog(base,[](SdfAbstractData& data) {
        data.Set(SdfPath("/P"),SdfChildrenKeys->PrimChildren,
                 VtValue(std::vector<TfToken>{TfToken("C"),TfToken("B"),TfToken("A")}));
    });
    checkIntent(!entryRefusal(base,log,{1,5}),"a children list exactly at its limit was refused");
    const auto refused = entryRefusal(base,log,{1,4});
    checkIntent(bool(refused),"a children list one past its limit was not refused");
    sameCounts(*refused,DeltaEntryLimitExceeded::Reason::ChildrenListTooLong,{1,0,0,4,5});
    const auto total = entryRefusal(base,log,{0,4});
    checkIntent(bool(total),"the entry limit was not checked before the children list limit");
    sameCounts(*total,DeltaEntryLimitExceeded::Reason::DeltaTooLarge,{1,0,0,0,1});
}
// Content equality over the wire projection: identical content matches and a
// changed value type does not.
void wireProjectionEquality() {
    const auto base = initialScene();
    auto same = TfCreateRefPtr(new SdfData); same->CopyFrom(base->data());
    checkIntent(matchesWireProjection(base,*same),"identical content did not match");
    same->Set(SdfPath("/P.a"),SdfFieldKeys->Default,VtValue(-0.0));
    checkIntent(!matchesWireProjection(base,*same),"changed content matched");
    same->CopyFrom(base->data()); same->CreateSpec(SdfPath("/Q"),SdfSpecTypePrim);
    checkIntent(!matchesWireProjection(base,*same),"added spec matched");
    checkIntent(!matchesWireProjection({},*same),"missing scene matched");
}
void signedZeroSampleKeys() {
    // A sample key keeps its bits until an erase of an equivalent time; the next set after an erase authors new bits.
    const auto keyBits = [](const WireScenePtr& scene) {
        const auto map = scene->data()->Get(SdfPath("/P.a"),SdfFieldKeys->TimeSamples).UncheckedGet<SdfTimeSampleMap>();
        std::vector<std::pair<bool,int>> keys;
        for (const auto& [time, value] : map) keys.emplace_back(std::signbit(time), value.UncheckedGet<int>());
        return keys;
    };
    { Fixture f; // Base key +0; a whole map authored at -0 replaces the key.
      f.both([](auto& data) { data.Set(SdfPath("/P.a"),SdfFieldKeys->TimeSamples,VtValue(SdfTimeSampleMap{{-0.0,VtValue(1027)}})); });
      checkIntent(keyBits(buildSceneCandidate(f.base,f.compile()->bytes())) == std::vector<std::pair<bool,int>>{{true,1027}},
                  "whole map at an equivalent time kept the base key bits"); }
    { Fixture f; // An ordinary set at -0 keeps the base key +0.
      f.both([](auto& data) { data.SetTimeSample(SdfPath("/P.a"),-0.0,VtValue(5)); });
      checkIntent(keyBits(buildSceneCandidate(f.base,f.compile()->bytes())) == std::vector<std::pair<bool,int>>{{false,5}},
                  "ordinary set replaced the base key bits"); }
    { Fixture f; // Set, erase, then set at an equivalent time: the last set's bits.
      f.both([](auto& data) { data.SetTimeSample(SdfPath("/P.a"),0.0,VtValue(6)); });
      f.both([](auto& data) { data.EraseTimeSample(SdfPath("/P.a"),0.0); });
      f.both([](auto& data) { data.SetTimeSample(SdfPath("/P.a"),-0.0,VtValue(7)); });
      checkIntent(keyBits(buildSceneCandidate(f.base,f.compile()->bytes())) == std::vector<std::pair<bool,int>>{{true,7}},
                  "set after an erase kept the erased key bits"); }
    { Fixture f; // Erase after the last set: no key.
      f.both([](auto& data) { data.SetTimeSample(SdfPath("/P.a"),-0.0,VtValue(8)); });
      f.both([](auto& data) { data.EraseTimeSample(SdfPath("/P.a"),0.0); });
      checkIntent(!buildSceneCandidate(f.base,f.compile()->bytes())->data()->Has(SdfPath("/P.a"),SdfFieldKeys->TimeSamples),
                  "erase after a set left a key"); }
}
// Seeded field-only logs (values, field erasure, samples, dictionary keys) over a multi-prim scene compile the same
// bytes with and without namespace planning.
void fieldOnlyFastPathSeeds(uint64_t seed) {
    EnvelopeInput metadata; DecodedValues values;
    std::vector<SdfPath> attributes;
    for (uint64_t i = 0; i < 8; ++i) {
        const auto prim = 1000 + 10 * i;
        metadata.commands.push_back(create(prim,1,("Q" + std::to_string(i)).c_str(),SdfSpecTypePrim));
        for (uint64_t j = 0; j < 3; ++j) {
            const std::string name = "x" + std::to_string(j);
            metadata.commands.push_back(create(prim + 1 + j,prim,name.c_str(),SdfSpecTypeAttribute));
            metadata.fields.push_back({prim + 1 + j,0,"default",{}}); values.fields.push_back(VtValue(int(j)));
            attributes.push_back(SdfPath("/Q" + std::to_string(i) + "." + name));
        }
    }
    const auto base = buildSceneCandidate({},encodeUsdEnvelope(metadata,values));
    std::mt19937_64 random(seed ^ 0x6661737470617468ull);
    for (int round = 0; round < 64; ++round) {
        const auto sealed = capturedLog(base,[&](SdfAbstractData& data) {
            const int count = 1 + int(random() % 6);
            for (int k = 0; k < count; ++k) {
                const auto& path = attributes[random() % attributes.size()];
                const auto prim = path.GetPrimPath();
                switch (random() % 7) {
                case 0: data.Set(path,SdfFieldKeys->Default,VtValue(int(random() % 100))); break;
                case 1: data.Erase(path,SdfFieldKeys->Default); break;
                case 2: data.SetTimeSample(path,double(random() % 5),VtValue(int(random() % 100))); break;
                case 3: data.EraseTimeSample(path,double(random() % 5)); break;
                case 4: data.SetDictValueByKey(prim,SdfFieldKeys->CustomData,TfToken("k" + std::to_string(random() % 3)),
                                               VtValue(int(random() % 100))); break;
                case 5: data.EraseDictValueByKey(prim,SdfFieldKeys->CustomData,TfToken("k" + std::to_string(random() % 3))); break;
                default: data.Set(prim,SdfFieldKeys->Documentation,VtValue(std::string(1 + random() % 8,'d'))); break;
                }
            }
        });
        const auto fast = compileFieldEdits(base,sealed);
        checkIntent(!fast->planned(),"a seeded field-only log planned its namespace");
        samePlanEitherWay(base,sealed,fast);
    }
}
// Seeded runs of sample writes interleaved with whole-map and other field writes on the same attributes compile
// to the independent stock oracle's result.
void sampleRunSeeds(uint64_t seed) {
    std::mt19937_64 random(seed ^ 0x73616d706c657275ull);
    const std::array<SdfPath,2> paths{SdfPath("/P.a"),SdfPath("/P.b")};
    for (int round = 0; round < 64; ++round) {
        intentContext = "sample runs seed " + std::to_string(seed) + " round " + std::to_string(round);
        Fixture f;
        for (int k = 0, count = 1 + int(random() % 24); k < count; ++k) {
            const auto& path = paths[random() % paths.size()];
            const double time = double(random() % 4);
            const int value = int(random() % 100);
            switch (random() % 8) {
            case 0: f.both([&](auto& data) { data.Set(path,SdfFieldKeys->TimeSamples,VtValue(SdfTimeSampleMap{{time,VtValue(value)}})); }); break;
            case 1: f.both([&](auto& data) { data.Erase(path,SdfFieldKeys->TimeSamples); }); break;
            case 2: f.both([&](auto& data) { data.Set(path,SdfFieldKeys->Default,VtValue(value)); }); break;
            case 3: case 4: f.both([&](auto& data) { data.EraseTimeSample(path,time); }); break;
            default: f.both([&](auto& data) { data.SetTimeSample(path,time,VtValue(value)); }); break;
            }
        }
        (void)f.compile();
    }
}
// A log of the default operation limit's sample writes to one attribute compiles within the default work limits.
void largeSingleAttributeSampleLog() {
    const auto base = initialScene();
    const int count = int(FieldEditLimits{}.maxOperations) - 1;
    const SdfPath attribute("/P.a");
    auto operations = std::make_shared<std::vector<SdfOperation>>();
    operations->reserve(count + 1);
    SdfTimeSampleMap expected;
    for (int i = 1; i <= count; ++i) {
        SdfOperation op(SdfOperationKind::SetSample); op.path = attribute; op.time = i; op.value = VtValue(i);
        operations->push_back(std::move(op)); expected[i] = VtValue(i);
    }
    SdfOperation erase(SdfOperationKind::EraseSample); erase.path = attribute; erase.time = 0; // The base sample.
    operations->push_back(std::move(erase));
    const auto data = TfCreateRefPtr(new SdfData); data->CopyFrom(base->data());
    data->Set(attribute,SdfFieldKeys->TimeSamples,VtValue(expected));
    const auto log = std::make_shared<const SdfOperationLog>(SdfOperationLog{data,operations});
    const auto intent = compileFieldEdits(base,log);
    checkIntent(decodeEnvelope(intent->bytes()).samples.size() == size_t(count+1),"compiled sample records differ from the log");
    const auto candidate = buildSceneCandidate(base,intent->bytes());
    checkIntent(candidate->data()->Get(attribute,SdfFieldKeys->TimeSamples) == VtValue(expected),"compiled samples differ from the log");
}
void runFieldEditIntentTests(uint64_t seed) {
    sampleRunSeeds(seed);
    intentContext = "one attribute with 65,536 sample writes"; largeSingleAttributeSampleLog();
    intentContext = "field-only fast path seeds"; fieldOnlyFastPathSeeds(seed);
    intentContext = "ordered fields"; orderedFields();
    intentContext = "repeated field writes collapse"; repeatedFieldWritesCollapse();
    intentContext = "rewritten fields keep their reads"; rewrittenFieldsKeepTheirReads();
    intentContext = "repeated sample and metadata writes collapse"; repeatedSampleAndMetadataWritesCollapse();
    intentContext = "repeated whole sample maps collapse"; repeatedWholeSampleMapsCollapse();
    intentContext = "field writes across structure"; fieldWritesAcrossStructure();
    intentContext = "interleaved samples/maps"; interleavedSamples();
    intentContext = "signed zero sample keys"; signedZeroSampleKeys();
    intentContext = "empty maps and erasure"; emptyMapsAndErasure();
    intentContext = "dictionary semantics"; dictionarySemantics();
    intentContext = "empty dictionary keys"; emptyDictionaryKeySemantics();
    intentContext = "typed and external"; typedAndExternal();
    intentContext = "unsupported and limits"; unsupportedAndLimits();
    intentContext = "dictionary depth preflight"; dictionaryDepthPreflight();
    intentContext = "operation authority"; operationAuthority();
    intentContext = "host capture divergence"; hostCaptureDivergence();
    intentContext = "host no-op records"; hostNoOpRecords();
    intentContext = "public structural Sdf operations"; publicStructuralDraft();
    intentContext = "captured fresh identity removal"; capturedFreshIdentityRemoval();
    intentContext = "captured structural veto"; capturedStructuralVeto();
    intentContext = "parent-wide reorder emission"; parentWideReorderEmission();
    intentContext = "complete reorder emission"; completeReorderEmission();
    intentContext = "replacement parent with surviving children"; replacementPreservesSurvivingChildren();
    intentContext = "moved samples and deleted values"; movedSamplesAndDeletedValues();
    intentContext = "wire projection equality"; wireProjectionEquality();
    intentContext = "delta entry limits"; deltaEntryLimits();
    intentContext = "whole-map sample entry limit"; wholeMapSampleEntryLimit();
    intentContext = "children list limit"; childrenListLimit();
    structuralChildPolicies();
    structuralLatestSubtreeReplay();
    intentContext = "structural allocator and limits"; structuralAllocatorAndLimits();
    intentContext = "log operation limit"; logOperationLimit();
    structuralSeedHistory(seed);
    nestedStructuralSeeds(seed);
    std::cout << "field edit intent ordering, independent stock oracle and retained ownership passed" << std::endl;
}
