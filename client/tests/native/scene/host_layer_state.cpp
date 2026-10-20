// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "HostLayerState.h"
#include "UsdValueCodec.h"
#include <pxr/base/tf/errorMark.h>
#include <pxr/base/tf/notice.h>
#include <pxr/base/tf/weakBase.h>
#include <pxr/base/tf/weakPtr.h>
#include <pxr/usd/sdf/changeBlock.h>
#include <pxr/usd/sdf/fileFormat.h>
#include <pxr/usd/sdf/layer.h>
#include <pxr/usd/sdf/notice.h>
#include <pxr/usd/sdf/schema.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usd/stage.h>
#include <functional>
#include <iostream>
#include <set>
#include <stdexcept>

PXR_NAMESPACE_USING_DIRECTIVE
using namespace omni::usd::live;
using namespace omni::usd::live::wire;
namespace {
using Failure = HostLayerState::Failure;
using Admission = HostLayerState::Admission;
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(std::string("host layer state: ") + message);
}
WireScenePtr scene(const char* name) {
    EnvelopeInput input;
    Command create{}; create.kind = Command::Kind::Create;
    create.section = 2; create.parent = 1; create.name = name; create.sectionType = SdfSpecTypePrim;
    input.commands.push_back(create);
    input.fields.push_back({2, 0, "specifier", {}});
    DecodedValues values; values.fields.emplace_back(SdfSpecifierDef);
    return buildSceneCandidate({}, encodeUsdEnvelope(input, values));
}
// Shared across layers, like the production format. InitData has no layer
// identity and must not share a layer's fault or data owner with another layer.
class Format final : public SdfFileFormat {
    const WireScenePtr empty_ = buildSceneCandidate({}, encodeEnvelope({}));
public:
    Format() : SdfFileFormat(TfToken("host_state_probe"), TfToken("1"), TfToken("usd"), "host_state_probe") {}
    SdfAbstractDataRefPtr InitData(const FileFormatArguments&) const override {
        return TfCreateRefPtr(new HostSceneProjection(empty_, std::make_shared<HostFailureLatch>()));
    }
    bool CanRead(const std::string&) const override { return false; }
    bool Read(SdfLayer*, const std::string&, bool) const override { return false; }
    bool ReadFromString(SdfLayer* layer, const std::string& value) const override {
        return SdfFileFormat::FindById(TfToken("usda"))->ReadFromString(layer, value);
    }
    static void publish(SdfLayer* layer, SdfAbstractDataRefPtr data) { _SetLayerData(layer, data); }
    static SdfAbstractDataConstRefPtr backing(const SdfLayer& layer) {
        // Caller owns the layer and its serial host access throughout promotion.
        return TfCreateRefPtrFromProtectedWeakPtr(_GetLayerData(layer));
    }
protected:
    SdfAbstractDataRefPtr _InitDetachedData(const FileFormatArguments& args) const override { return InitData(args); }
};
struct Observer final : TfWeakBase {
    HostLayerState& state;
    TfNotice::Key key;
    std::function<void()> action;
    unsigned calls = 0;
    Observer(const SdfLayerHandle& layer, HostLayerState& state) : state(state) {
        key = TfNotice::Register(TfCreateWeakPtr(this), &Observer::changed, layer);
    }
    ~Observer() { TfNotice::Revoke(key); }
    void changed(const SdfNotice::LayersDidChangeSentPerLayer&, const SdfLayerHandle& layer) {
        ++calls;
        state.observe(Format::backing(*layer));
        auto run = std::move(action); // One-shot before permitting nested notices.
        if (run) run();
    }
};
void stateTransitions() {
    const auto baseline = scene("Old"), next = scene("Next");
    HostLayerState state(baseline), other(baseline);
    const auto old = state.authorized();
    const auto candidate = state.prepare(next);
    require(state.begin(old, other.authorized()) == Admission::InvalidCandidate,
            "cross-layer candidate accepted");
    require(state.begin(old, old) == Admission::InvalidCandidate, "same owner accepted as a new publication");
    require(state.failure() == Failure::None && !state.installing(), "invalid candidate changed state");
    require(state.begin(old, candidate) == Admission::Accepted, "valid publication rejected");
    require(state.observe(old) && state.observe(old), "old owner rejected before installation");
    require(state.begin(old, state.prepare(next)) == Admission::Busy, "nested publication accepted");
    require(state.observe(candidate) && state.observe(candidate), "candidate rejected during installation");
    require(state.finish(candidate) && state.authorized() == candidate && !state.installing(),
            "successful publication did not advance exact owner");
    require(state.authorized()->baseline() == next && old->baseline() == baseline,
            "publication changed retained provenance");
    require(!state.finish(candidate) && !state.abort(), "finished publication accepted another completion");

    for (bool installed : {false, true}) {
        HostLayerState failed(baseline);
        const auto original = failed.authorized(), pending = failed.prepare(next);
        require(failed.begin(original, pending) == Admission::Accepted, "failure control begin");
        if (installed) require(failed.observe(pending), "failure control install");
        require(failed.abort() && failed.publicationFailed() && failed.failure() == Failure::PublicationFailed,
                "aborted installation returned to stable");
        require(failed.authorized() == original && failed.candidate() == pending,
                "failed installation discarded provenance owners");
        require(!failed.finish(installed ? SdfAbstractDataConstRefPtr(pending) : SdfAbstractDataConstRefPtr(original)),
                "failed installation later succeeded");
    }
    HostLayerState missing(baseline);
    const auto pending = missing.prepare(next);
    require(missing.begin(missing.authorized(), pending) == Admission::Accepted, "missing install begin");
    require(!missing.finish(missing.authorized()) && missing.failure() == Failure::PublicationFailed,
            "uninstalled candidate reported success");
    HostLayerState rollback(baseline);
    const auto original = rollback.authorized(), replacement = rollback.prepare(next);
    require(rollback.begin(original, replacement) == Admission::Accepted && rollback.observe(replacement),
            "rollback control begin");
    require(!rollback.observe(original) && rollback.failure() == Failure::BackingReplaced,
            "observed candidate-to-old rollback accepted");
    require(!rollback.finish(replacement) && rollback.authorized() == original && rollback.publicationFailed(),
            "faulted rollback advanced provenance");
    HostLayerState mutated(baseline);
    const auto changed = mutated.prepare(next);
    require(mutated.begin(mutated.authorized(), changed) == Admission::Accepted, "mutation abort begin");
    changed->Set(SdfPath("/Next"), SdfFieldKeys->Documentation, VtValue(std::string("unadmitted")));
    require(changed->hasPendingLocalEdits() && mutated.abort() && mutated.failure() == Failure::PublicationFailed &&
            mutated.publicationFailed(), "a recorded host edit was reported as a host failure");
    HostLayerState copied(baseline);
    const auto replaced = copied.prepare(next);
    require(copied.begin(copied.authorized(), replaced) == Admission::Accepted, "copy abort begin");
    replaced->fault()->fail(OMNI_USD_LIVE_UNSUPPORTED);
    require(copied.abort() && copied.failure() == Failure::HostMutation && copied.publicationFailed(),
            "aborting publication obscured the host failure");
    HostLayerState substituted(baseline);
    const auto admitted = substituted.prepare(next), unadmitted = substituted.prepare(next);
    const auto retained = substituted.authorized();
    require(admitted != unadmitted && admitted->fault() == unadmitted->fault(), "same-owner control invalid");
    require(substituted.begin(retained, admitted) == Admission::Accepted, "exact candidate control begin");
    require(!substituted.finish(unadmitted) && substituted.failure() == Failure::BackingReplaced &&
            substituted.authorized() == retained && substituted.candidate() == admitted &&
            substituted.observed() == unadmitted, "same-latch candidate substitution accepted");
}
// An incremental publication keeps the authorized projection as the backing.
void inPlaceTransitions() {
    const auto baseline = scene("Old"), next = scene("Next");
    HostLayerState state(baseline);
    const auto authorized = state.authorized();
    require(state.beginInPlace(authorized) == Admission::Accepted && state.installing(), "in-place publication rejected");
    require(state.beginInPlace(authorized) == Admission::Busy &&
            state.begin(authorized, state.prepare(next)) == Admission::Busy, "nested publication accepted");
    authorized->reseat(authorized->prepareReseat(next, diffProjectedScenes(*baseline, *next)));
    require(state.finishInPlace(authorized) && !state.installing() && state.authorized() == authorized &&
            authorized->baseline() == next && authorized->HasSpec(SdfPath("/Next")) &&
            !authorized->HasSpec(SdfPath("/Old")) && state.failure() == Failure::None,
            "in-place publication did not keep the reseated owner");
    require(!state.finishInPlace(authorized) && !state.abort(), "finished publication accepted another completion");
    HostLayerState other(baseline);
    require(state.beginInPlace(other.authorized()) == Admission::Faulted && state.failure() == Failure::BackingReplaced,
            "a foreign backing admitted an in-place publication");
    HostLayerState swapped(baseline);
    require(swapped.beginInPlace(swapped.authorized()) == Admission::Accepted, "swap control begin");
    require(!swapped.finishInPlace(other.authorized()) && swapped.publicationFailed() &&
            swapped.failure() == Failure::BackingReplaced, "a swapped backing finished an in-place publication");
    HostLayerState aborted(baseline);
    require(aborted.beginInPlace(aborted.authorized()) == Admission::Accepted && aborted.abort() &&
            aborted.failure() == Failure::PublicationFailed && !aborted.installing(), "in-place abort did not fail");
}
void initialAdoption() {
    const auto baseline=scene("Original");
    const auto first=TfCreateRefPtr(new HostSceneProjection(baseline,std::make_shared<HostFailureLatch>()));
    const auto second=TfCreateRefPtr(new HostSceneProjection(baseline,std::make_shared<HostFailureLatch>()));
    HostLayerState state(first), other(second);
    require(state.authorized()==first && state.observed()==first && state.provenance()==baseline,
            "initial adoption replaced exact backing or provenance");
    first->Set(SdfPath("/Original"),SdfFieldKeys->Documentation,VtValue(std::string("host")));
    require(state.observe(first) && first->hasPendingLocalEdits() && !second->hasPendingLocalEdits(),
            "a recorded host edit faulted its layer or reached another initial owner");
    first->fault()->fail(OMNI_USD_LIVE_UNSUPPORTED);
    require(!state.observe(first) && other.observe(second) && other.failure()==Failure::None,
            "one layer's mutation contaminated another initial owner");
    HostLayerState adoptedFault(first);
    require(adoptedFault.failure()==Failure::HostMutation && adoptedFault.authorized()==first,
            "initial adoption cleared an existing host fault");
    HostLayerState publication(second);
    const auto next=publication.prepare(scene("Next"));
    require(publication.begin(second,next)==Admission::Accepted && publication.observe(next),"candidate adoption setup");
    require(publication.provenance()==next->baseline(),"installed candidate provenance missing before completion");
    next->fault()->fail(OMNI_USD_LIVE_UNSUPPORTED);
    require(!publication.observe(next) && publication.provenance()==next->baseline(),
            "candidate mutation erased installed service provenance");
    const auto foreign=TfCreateRefPtr(new HostSceneProjection(scene("Foreign"),std::make_shared<HostFailureLatch>()));
    require(!publication.finish(foreign) && publication.provenance()==next->baseline(),
            "foreign replacement substituted publication provenance");
}
void ownerRetirement() {
    struct ReenterData final : SdfData {
        std::function<void()> retire;
        ~ReenterData() override { if (retire) retire(); }
    };
    HostLayerState state(scene("Old"));
    const auto pending = state.prepare(scene("Next"));
    const SdfAbstractDataConstRefPtr replacement = TfCreateRefPtr(new SdfData);
    const SdfAbstractDataConstRefPtr reentrant = TfCreateRefPtr(new SdfData);
    bool retired = false, coherent = false, denied = false;
    auto foreign = TfCreateRefPtr(new ReenterData);
    foreign->retire = [&] {
        retired = true;
        coherent = state.failure() == Failure::BackingReplaced && state.observed() == replacement;
        denied = state.begin(replacement, pending) == Admission::Faulted;
        state.observe(reentrant);
    };
    require(!state.observe(foreign), "foreign backing authorized");
    foreign.Reset(); // The state owns the only foreign reference from here on.
    require(!retired, "foreign backing was not retained");
    require(!state.observe(replacement), "replacement cleared permanent fault");
    require(retired && coherent && denied && state.observed() == reentrant,
            "owner retirement exposed incomplete state or lost reentrant observation");
    require(!state.observe(state.observed()) && state.observed() == reentrant,
            "aliased observation lost its owner");
}
void replacements(const SdfFileFormatRefPtr& format) {
    const auto baseline = scene("Old");
    for (unsigned operation = 0; operation != 3; ++operation) {
        HostLayerState state(baseline);
        const auto layer = SdfLayer::CreateAnonymous("replace.host_state_probe", format);
        Format::publish(layer.operator->(), state.authorized());
        Observer observer(layer, state);
        const auto stock = SdfLayer::CreateAnonymous("stock.usda");
        require(stock->ImportFromString("#usda 1.0\ndef \"New\" {}\n"), "replacement stock parse");
        if (operation == 1) {
            // CopyFrom on the fresh InitData projection rejects TransferContent
            // before the layer's backing changes.
            TfErrorMark errors; bool rejected = false;
            try { layer->TransferContent(stock); }
            catch (const LayerOperationRejected& error) { rejected = error.result() == OMNI_USD_LIVE_UNSUPPORTED; }
            errors.Clear();
            require(rejected && Format::backing(*layer) == state.authorized() && state.observe(Format::backing(*layer)) &&
                    state.failure() == Failure::None && state.fault()->result().load() == OMNI_USD_LIVE_OK,
                    "TransferContent was not rejected before replacing the backing");
            continue;
        }
        if (operation == 0) { layer->Clear(); stock->Clear(); }
        if (operation == 2) require(layer->ImportFromString("#usda 1.0\ndef \"New\" {}\n"), "import failed");
        const auto actual = Format::backing(*layer);
        require(observer.calls && !state.observe(actual) && state.failure() == Failure::BackingReplaced,
                "fresh whole-content replacement not detected");
        const auto expected = TfCreateRefPtr(new SdfData); expected->CopyFrom(Format::backing(*stock));
        require(actual->Equals(expected), "replacement differs from independently authored USD");
        require(state.observed() == actual && state.authorized()->baseline() == baseline &&
                state.authorized()->HasSpec(SdfPath("/Old")), "replacement discarded original provenance");
        require(!state.authorized()->hasPendingLocalEdits(),
                "shared format routed replacement's mutation to original layer projection");
        require(state.begin(actual, state.prepare(scene("Later"))) == Admission::Faulted,
                "faulted layer admitted publication");
    }
}
void stockReentryControl() {
    // No custom backing participates in the operation that triggers reentry.
    // This distinguishes an SDK notice-ordering limit from our storage logic.
    const auto layer = SdfLayer::CreateAnonymous("stock-reentry.usda");
    const auto source = SdfLayer::CreateAnonymous("stock-replacement.usda");
    require(layer->ImportFromString("#usda 1.0\ndef \"Old\" {}\n") &&
            source->ImportFromString("#usda 1.0\ndef \"Next\" {}\n"), "stock control parse failed");
    const auto stage = UsdStage::Open(layer);
    require(stage && stage->GetPrimAtPath(SdfPath("/Old")), "stock control stage failed");
    HostLayerState observationOnly(scene("Old"));
    Observer observer(layer, observationOnly);
    observer.action = [&] {
        layer->SetField(SdfPath("/Next"), SdfFieldKeys->Documentation, VtValue(std::string("stock edit")));
    };
    TfErrorMark errors;
    layer->TransferContent(source);
    for (const auto& error : errors) {
        require(error.GetCommentary().find("Detected usd threading violation.") == 0,
                "unexpected stock reentry diagnostic");
        std::cout << "stock SDK reentry control: " << error.GetCommentary() << std::endl;
    }
    errors.Clear();
    const auto actual = Format::backing(*layer);
    require(actual->Get(SdfPath("/Next"), SdfFieldKeys->Documentation) == VtValue(std::string("stock edit")),
            "stock control lost direct edit");
}
void reentrantPublication(const SdfFileFormatRefPtr& format) {
    const auto baseline = scene("Old"), next = scene("Next");
    for (unsigned operation = 0; operation != 4; ++operation) {
        std::cout << "host layer state publication case=" << operation << std::endl;
        HostLayerState state(baseline), other(baseline);
        const auto layer = SdfLayer::CreateAnonymous("notice.host_state_probe", format);
        const auto peer = SdfLayer::CreateAnonymous("peer.host_state_probe", format);
        const auto old = state.authorized();
        Format::publish(layer.operator->(), old);
        Format::publish(peer.operator->(), other.authorized());
        const auto stage = UsdStage::Open(layer), peerStage = UsdStage::Open(peer);
        require(stage && peerStage && stage->GetPrimAtPath(SdfPath("/Old")) &&
                peerStage->GetPrimAtPath(SdfPath("/Old")), "initial stage composition failed");
        Observer observer(layer, state);
        const auto candidate = state.prepare(next);
        if (operation == 1) observer.action = [&] {
            layer->SetField(SdfPath("/Next"), SdfFieldKeys->Documentation, VtValue(std::string("host edit")));
        };
        if (operation == 2) observer.action = [&] { layer->Clear(); };
        if (operation == 3) observer.action = [&] { Format::publish(layer.operator->(), old); };
        require(state.begin(Format::backing(*layer), candidate) == Admission::Accepted, "public begin failed");
        TfErrorMark publicationErrors;
        { SdfChangeBlock block; Format::publish(layer.operator->(), candidate); }
        const auto actual = Format::backing(*layer);
        const bool finished = state.finish(actual);
        require(observer.calls != 0, "publication notices not observed");
        require(other.observe(Format::backing(*peer)) && other.failure() == Failure::None,
                "shared-format neighbor contaminated");
        std::set<SdfPath> composed;
        for (const auto& prim : stage->Traverse()) composed.insert(prim.GetPath());
        const std::set<SdfPath> expectedPaths = operation == 2 ? std::set<SdfPath>{} :
            std::set<SdfPath>{SdfPath(operation == 3 ? "/Old" : "/Next")};
        for (const auto& error : publicationErrors) {
            const auto& text = error.GetCommentary();
            require(operation != 0 && text.find("Detected usd threading violation.") == 0,
                    "unexpected reentrant USD diagnostic");
            std::cout << "observed SDK reentry diagnostic: " << text << std::endl;
        }
        publicationErrors.Clear();
        // HostLayerState alone records a reentrant edit; the file format fails
        // the layer for it (scene_file_format notice-author). Whole-content
        // replacement (cases 2 and 3) faults here.
        // Case 1 edits inside the publication notice. Stock USD reports a
        // threading violation and keeps the stale /Old prim composed; this test
        // does not recompose that stage, so only the layer contents are asserted.
        require(operation == 1 || composed == expectedPaths, "composed stage disagrees with actual host replacement");
        require(peerStage->GetPrimAtPath(SdfPath("/Old")) && !peerStage->GetPrimAtPath(SdfPath("/Next")),
                "shared-format neighbor stage changed");
        if (operation <= 1) {
            require(finished && state.authorized() == candidate && state.failure() == Failure::None,
                    "ordinary publication or recorded reentrant edit rejected");
            if (operation == 1) {
                require(candidate->hasPendingLocalEdits() &&
                        candidate->fault()->result().load() == OMNI_USD_LIVE_OK,
                        "reentrant host edit was not recorded on the installed candidate");
                const auto captured = candidate->capture();
                require(captured->Get(SdfPath("/Next"), SdfFieldKeys->Documentation) == VtValue(std::string("host edit")),
                        "actual host edit not capturable");
                require(layer->GetField(SdfPath("/Next"), SdfFieldKeys->Documentation) == VtValue(std::string("host edit")),
                        "layer lost reentrant host metadata");
                require(!next->data()->Has(SdfPath("/Next"), SdfFieldKeys->Documentation),
                        "host edit changed accepted service baseline");
            }
        } else {
            require(!finished && state.authorized() == old && state.candidate() == candidate && state.publicationFailed(),
                    "reentrant replacement authorized or discarded provenance");
            require(state.failure() == Failure::BackingReplaced, "wrong reentrant failure reason");
            require(state.observed() == Format::backing(*layer), "observer missed actual host backing");
        }
    }
}
}
// Replacement-generation exhaustion reports RESOURCE_EXHAUSTED whether the
// publication or a host observer detects it first, and a later publication
// abort cannot replace that cause.
void publicationFailureCauses() {
    const auto baseline = scene("Old");
    for (unsigned order = 0; order != 3; ++order) {
        HostLayerState state(baseline, {1});
        const auto foreign = TfCreateRefPtr(new SdfData); foreign->CopyFrom(baseline->data());
        const auto candidate = state.prepare(scene("Next"));
        if (order == 0) {
            require(state.begin(foreign, candidate) == Admission::Faulted, "exhausted publication admitted");
            require(state.admissionResult(Admission::Faulted) == OMNI_USD_LIVE_RESOURCE_EXHAUSTED,
                    "publication-first exhaustion reported another cause");
        } else if (order == 1) {
            require(!state.observe(foreign), "observer missed exhaustion");
            require(state.begin(foreign, candidate) == Admission::Faulted &&
                    state.admissionResult(Admission::Faulted) == OMNI_USD_LIVE_RESOURCE_EXHAUSTED,
                    "observer-first exhaustion reported another cause");
        } else {
            require(state.begin(state.authorized(), candidate) == Admission::Accepted, "publication control begin");
            require(!state.finish(foreign), "exhausted installation finished");
            state.abort();
            require(state.cause() == OMNI_USD_LIVE_RESOURCE_EXHAUSTED,
                    "abort replaced an exhaustion detected during installation");
        }
        require(state.failure() == Failure::GenerationExhausted &&
                state.fault()->result().load() == OMNI_USD_LIVE_RESOURCE_EXHAUSTED &&
                state.cause() == OMNI_USD_LIVE_RESOURCE_EXHAUSTED, "latched exhaustion cause diverged");
    }
    HostLayerState healthy(baseline);
    require(healthy.admissionResult(Admission::Accepted) == OMNI_USD_LIVE_OK &&
            healthy.admissionResult(Admission::Busy) == OMNI_USD_LIVE_BUSY &&
            healthy.admissionResult(Admission::InvalidCandidate) == OMNI_USD_LIVE_INTERNAL_ERROR &&
            healthy.cause() == OMNI_USD_LIVE_UNSUPPORTED, "admission result mapping");
    healthy.fault()->fail(OMNI_USD_LIVE_UNSUPPORTED);
    require(!healthy.observe(healthy.authorized()) && healthy.cause() == OMNI_USD_LIVE_UNSUPPORTED &&
            healthy.admissionResult(Admission::Faulted) == OMNI_USD_LIVE_UNSUPPORTED,
            "host mutation cause changed");
}
void hostLayerStateTests() {
    initialAdoption();
    TfErrorMark errors;
    stateTransitions(); inPlaceTransitions(); ownerRetirement();
    const SdfFileFormatRefPtr format = TfCreateRefPtr(new Format);
    replacements(format); stockReentryControl(); reentrantPublication(format);
    publicationFailureCauses();
    require(errors.IsClean(), "unexpected USD diagnostic");
    std::cout << "host layer state: exact publication/replacement/reentrant ownership passed (not production containment)\n";
}
