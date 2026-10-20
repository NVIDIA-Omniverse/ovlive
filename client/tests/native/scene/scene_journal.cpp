// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <random>
#include <sstream>
#include "SceneJournal.h"
#include "SceneSeed.h"
#include "captured_log.h"
#include <SHA256.h>
#include <pxr/usd/sdf/schema.h>
#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <future>
#include <iostream>
#include <initializer_list>
#include <map>
#include <stdexcept>
#include <thread>

PXR_NAMESPACE_USING_DIRECTIVE
using namespace omni::usd::live;
using namespace omni::usd::live::wire;
using namespace omni::usd::live::test;
namespace {
std::string context;
void check(bool condition, const char* message) { if (!condition) throw std::runtime_error(context+": "+message); }
std::string edit(int value, const char* key = "value") {
    EnvelopeInput metadata; metadata.fields = {{2,0,key,{}}};
    return encodeUsdEnvelope(metadata,DecodedValues{{VtValue(value)},{}});
}
std::string createChild() {
    EnvelopeInput metadata; Command c{}; c.kind = Command::Kind::Create; c.section = 3; c.parent = 2;
    c.name = "Child"; c.sectionType = SdfSpecTypePrim; metadata.commands = {c}; return encodeEnvelope(metadata);
}
std::string deleteParent() {
    EnvelopeInput metadata; Command c{}; c.kind = Command::Kind::Delete; c.section = 2; c.parent = 1;
    metadata.commands = {c}; return encodeEnvelope(metadata);
}
WireScenePtr initialScene() {
    EnvelopeInput metadata; Command c{}; c.kind = Command::Kind::Create; c.section = 2; c.parent = 1;
    c.name = "P"; c.sectionType = SdfSpecTypePrim; metadata.commands = {c}; metadata.fields = {{2,0,"value",{}}};
    return buildSceneCandidate({},encodeUsdEnvelope(metadata,DecodedValues{{VtValue(0)},{}}));
}
int number(const WireScenePtr& scene, const char* key = "value") {
    const VtValue value = scene->data()->Get(SdfPath("/P"),TfToken(key));
    check(value.IsHolding<int>(),"expected integer field absent/wrong type"); return value.Get<int>();
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
        bool VisitSpec(const SdfAbstractData& data, const SdfPath& path) override {
            Record entry{data.GetSpecType(path),{}};
            for (const auto& field : data.List(path)) entry.fields.emplace(field,data.Get(path,field));
            result.emplace(path,std::move(entry)); return true;
        }
        void Done(const SdfAbstractData&) override {}
    } visitor;
    scene->data()->VisitSpecs(&visitor); return visitor.result;
}
struct Saved {
    SceneJournal::SnapshotPtr pointer;
    State authoritative, visible;
    std::vector<uint64_t> ids;
    std::vector<std::string> bytes;
    std::vector<std::shared_ptr<const SceneJournal::Intent>> owners;
    uint64_t applied;
    uint64_t initialApplied, localAdmitted, localRetired;
    size_t pendingBytes;
    OmniUsdLiveResult fault;
    explicit Saved(const SceneJournal& journal) : pointer(journal.snapshot()), authoritative(inspect(pointer->authoritative)), visible(inspect(pointer->visible)),
        applied(pointer->applied), initialApplied(pointer->initialApplied), localAdmitted(pointer->localAdmitted),
        localRetired(pointer->localRetired), pendingBytes(pointer->pendingBytes), fault(pointer->submissionFault) {
        for (const auto& entry : pointer->pending) { ids.push_back(entry.id); bytes.push_back(entry.intent->bytes); owners.push_back(entry.intent); }
    }
    void unchanged(const SceneJournal& journal) const {
        const auto now = journal.snapshot();
        check(now == pointer && inspect(now->authoritative) == authoritative && inspect(now->visible) == visible,"rejection changed journal snapshot/scene");
        check(now->pending.size() == ids.size() && now->applied == applied && now->pendingBytes == pendingBytes && now->submissionFault == fault,
            "rejection changed retained count/frontier/byte accounting/fault");
        check(now->initialApplied==initialApplied && now->localAdmitted==localAdmitted && now->localRetired==localRetired,
              "rejection changed admission/retirement history");
        for (size_t i = 0; i < ids.size(); ++i) check(now->pending[i].id == ids[i] && now->pending[i].intent->bytes == bytes[i] &&
            now->pending[i].intent == owners[i] && !now->pending[i].intent.owner_before(owners[i]) && !owners[i].owner_before(now->pending[i].intent),
            "rejection changed intent identity/content");
    }
};
template<class F> void result(OmniUsdLiveResult expected, F&& action) {
    bool caught = false;
    try { action(); } catch (OmniUsdLiveResult value) { check(value == expected,"wrong journal result"); caught = true; }
    check(caught,"journal operation did not reject");
}
// A captured structural shape the service cannot take in one update is
// rejected with its reason and no single operation.
template<class F> void unsupportedShape(UnsupportedOperation::Reason reason, F&& action) {
    bool named = false;
    try { action(); } catch (const UnsupportedOperation& error) {
        check(error.kind == ErrorKind::Unsupported && error.index() == UnsupportedOperation::noOperation &&
              error.reason() == reason,"structural rejection named the wrong reason");
        named = true;
    }
    check(named,"structural rejection did not name its reason");
}
struct Effect { std::shared_ptr<const SceneJournal::Intent> intent; };
struct Fixture {
    WireScenePtr initial;
    SceneJournal journal;
    LayerProgress progress;
    LayerProgress::ReadCursor read;
    size_t submits = 0;
    explicit Fixture(SceneJournal::Limits limits = {}, size_t capacity = 128, WireScenePtr source = {},
                     std::shared_ptr<SectionIdAllocator> ids = {})
        : initial(source ? std::move(source) : initialScene()), journal(initial,10,limits,std::move(ids)),
          progress({1,"session","layer","writer",1},10,initial,capacity,1000) {
        read = progress.startRead();
        check(progress.caughtUp(read.generation,read.object,10).accepted,"progress did not become ready");
    }
    SceneJournal::Submit submitter() {
        return [this](std::string_view bytes, std::shared_ptr<const SceneJournal::Intent> intent, LayerProgress::Publication publication) {
            ++submits; check(intent && intent->bytes == bytes,"journal did not retain original submitted bytes");
            // The real actor retains an Effect owner distinct from its Intent.
            const auto retainedBytes=intent->retainedBytes;
            return progress.admit(std::make_shared<const Effect>(Effect{std::move(intent)}),publication,retainedBytes);
        };
    }
    uint64_t local(const std::string& bytes) { return journal.submitLocal(bytes,submitter()); }
    LayerProgress::Work retained(uint64_t id) const {
        for (const auto& work : progress.retained()) if (work.id == id) return work;
        throw std::runtime_error("missing retained progress work");
    }
    LayerProgress::Work ack(uint64_t id, uint64_t sequence) {
        const auto issued = progress.takeWrite(); check(issued.id == id,"wrong issued FIFO head");
        progress.completeWrite(id,OMNI_USD_LIVE_OK,sequence);
        auto work = retained(id); check(work.disposition == LayerProgress::Disposition::Acknowledged,"write did not become acknowledged"); return work;
    }
    LayerProgress::Work peer(const std::string& bytes, uint64_t sequence) {
        auto owner = std::make_shared<const std::string>(bytes);
        LayerProgress::ContentHash hash{}; computeSHA256(bytes.data(),bytes.size(),hash.data());
        const auto id = progress.acceptCoverage(read.generation,read.object,sequence,sequence,{owner,hash});
        check(id != 0,"new peer coverage was not retained"); return retained(id);
    }
    static std::shared_ptr<const SceneJournal::Intent> intent(const LayerProgress::Work& work) {
        return std::static_pointer_cast<const Effect>(work.owner)->intent;
    }
    void applyLocal(const LayerProgress::Work& work) {
        const auto owner = intent(work); journal.apply(work,owner->bytes,owner); progress.applied(work.id);
    }
    void applyPeer(const LayerProgress::Work& work, const std::string& bytes) { journal.apply(work,bytes); progress.applied(work.id); }
};
void admissionHistory() {
    context = "local admission history and original dependency revisions";
    Fixture f;
    auto peer=f.peer(edit(7,"peer"),11); f.applyPeer(peer,edit(7,"peer"));
    const auto firstBase=f.journal.snapshot();
    const auto first=f.local(edit(1)); const auto secondBase=f.journal.snapshot();
    const auto second=f.local(edit(2)); const auto both=f.journal.snapshot();
    check(first==2 && second==3 && f.retained(first).localOrdinal==1 && f.retained(second).localOrdinal==2,
          "peer work consumed a local ordinal");
    check(both->initialApplied==10 && both->localAdmitted==2 && !both->localRetired,
          "journal admission checkpoint differs from accepted history");
    const auto a=both->pending[0].intent, b=both->pending[1].intent;
    check(a->baseRevision==firstBase->revision && a->localOrdinal==1 && !a->dependencyCutoff &&
          b->baseRevision==secondBase->revision && b->localOrdinal==2 && b->dependencyCutoff==1 &&
          a->retainedBytes==a->bytes.size() && b->retainedBytes==b->bytes.size(),
          "raw intent lost its base revision or dependency cutoff, or was charged for a scene");
    auto ack=f.ack(first,12); auto wrong=ack; wrong.localOrdinal=2;
    const Saved before(f.journal);
    result(OMNI_USD_LIVE_INTERNAL_ERROR,[&] { f.journal.apply(wrong,a->bytes,a); }); before.unchanged(f.journal);
    f.journal.apply(ack,a->bytes,a);
    check(f.journal.snapshot()->localRetired==1 && f.progress.snapshot().localRetired==0 &&
          f.progress.snapshot().pendingWrites==2 && f.journal.snapshot()->pending.size()==1,
          "journal application was confused with completed host/actor application");
    f.progress.applied(first);
    peer=f.peer(edit(9,"peer"),13); f.applyPeer(peer,edit(9,"peer"));
    check(f.journal.snapshot()->localAdmitted==2 && f.journal.snapshot()->localRetired==1 &&
          b->baseRevision==secondBase->revision && b->bytes==edit(2) && number(f.journal.snapshot()->visible)==2 &&
          number(f.journal.snapshot()->visible,"peer")==9,
          "peer replay rewrote original intent dependencies or bytes");
    f.applyLocal(f.ack(second,14));
    check(f.progress.snapshot().localRetired==2 && f.journal.snapshot()->localRetired==2 &&
          f.journal.snapshot()->pending.empty() && f.progress.snapshot().initialApplied==10,
          "retired prefix lost historical accepted edits");
    const auto original=f.journal.snapshot();
    const auto log=capturedLog(original->visible,[](SdfAbstractData& data) {
        data.Set(SdfPath("/P"),TfToken("value"),VtValue(3));
    });
    const auto third=f.journal.submitCaptured(original->visible,log,f.submitter());
    const auto authored=f.journal.snapshot()->pending.front().intent;
    check(authored->localOrdinal==3 && authored->dependencyCutoff==2 && authored->baseRevision==original->revision &&
          authored->authored->operations()==log->operations,
          "authored base revision and ordinal disagree with admitted history");
    check(f.progress.takeWrite().id==third,"third local work not issued");
    f.progress.completeWrite(third,OMNI_USD_LIVE_OUTCOME_UNKNOWN);
    const Saved uncertain(f.journal);
    result(OMNI_USD_LIVE_OUTCOME_UNKNOWN,[&] { f.local(edit(4)); }); uncertain.unchanged(f.journal);
    check(f.progress.snapshot().localAdmitted==3 && f.progress.snapshot().localRetired==2 &&
          f.retained(third).localOrdinal==3,"uncertain edit disappeared from admission suffix");

    EnvelopeInput large; large.fields={{2,0,"large",{}}};
    const auto heavy=buildSceneCandidate(initialScene(),encodeUsdEnvelope(large,{{VtValue(std::string(65536,'x'))},{}}));
    // A raw edit pins no base, so it is charged its bytes whatever the layer's size, and the configured quota
    // bounds what repeated edits retain.
    SceneJournal::Limits limits; limits.maxPendingBytes=4096;
    Fixture bounded(limits,4096,heavy);
    bounded.local(edit(1));
    check(bounded.journal.snapshot()->pendingBytes == edit(1).size(),"a raw edit was charged for the layer");
    bool rejected=false; size_t admitted=1;
    for (; admitted < 256 && !rejected; ++admitted) {
        const Saved before(bounded.journal);
        try { bounded.local(edit(int(admitted))); continue; }
        catch(const DecodeError& e) { rejected=e.kind==ErrorKind::ResourceLimit; }
        catch(OmniUsdLiveResult e) { rejected=e==OMNI_USD_LIVE_RESOURCE_EXHAUSTED; }
        before.unchanged(bounded.journal);
    }
    check(rejected && admitted < 256 && bounded.progress.snapshot().localAdmitted == admitted - 1 &&
          bounded.journal.snapshot()->pendingBytes <= limits.maxPendingBytes &&
          bounded.journal.snapshot()->pendingBytes + edit(int(admitted - 1)).size() > limits.maxPendingBytes,
          "raw edits never exhausted the configured pending quota");
}
void authoredAdmission() {
    context = "captured admission against its authored base";
    Fixture f;
    const auto base = f.journal.snapshot()->visible;
    const auto log = capturedLog(base,[](SdfAbstractData& data) {
        data.Set(SdfPath("/P"),TfToken("value"),VtValue(1));
        data.Set(SdfPath("/P"),TfToken("value"),VtValue(0));
    });
    const Saved before(f.journal);
    bool failed = false;
    try { f.journal.submitCaptured(base,log,[](auto,auto,auto)->uint64_t { throw std::bad_alloc(); }); }
    catch (const std::bad_alloc&) { failed = true; }
    check(failed && !f.progress.snapshot().localAdmitted,"failed captured admission became accepted");
    before.unchanged(f.journal);
    const auto id = f.journal.submitCaptured(base,log,f.submitter());
    check(id == 1 && f.submits == 1,"captured acceptance identity differs");
    const auto now = f.journal.snapshot();
    const auto intent = now->pending.at(0).intent;
    check(intent->authored && intent->authored->operations() == log->operations && log->operations->size() == 2 &&
          intent->baseRevision == before.pointer->revision,
          "accepted captured log lost its original operations or base revision");
    check(now->pendingBytes == intent->retainedBytes && now->pendingBytes > intent->bytes.size(),
          "authoring retention charged only wire bytes");
    check(number(now->visible) == 0 && now->applied == 10,"ABA authoring changed authoritative frontier");
    f.applyLocal(f.ack(id,11));
    check(f.journal.snapshot()->pending.empty() && !f.journal.snapshot()->pendingBytes,
          "authored confirmation did not retire complete retention charge");

    SceneJournal::Limits limits; limits.maxPendingBytes = intent->bytes.size();
    Fixture bounded(limits);
    const auto boundedBase = bounded.journal.snapshot()->visible;
    // The quota holds 32 charges of the scene beside its configured amount; an edit whose own value exceeds both is
    // rejected.
    const auto large = capturedLog(boundedBase,[](SdfAbstractData& data) {
        data.Set(SdfPath("/P"),TfToken("value"),VtValue(std::string(1 << 20,'x')));
    });
    const Saved boundedBefore(bounded.journal);
    bool resource = false;
    try { bounded.journal.submitCaptured(boundedBase,large,bounded.submitter()); }
    catch (const CapturedRejection& rejection) { resource = rejection.result == OMNI_USD_LIVE_RESOURCE_EXHAUSTED; }
    check(resource,"captured retention limit did not reject preparation");
    boundedBefore.unchanged(bounded.journal);
    check(bounded.submits == 0 && !bounded.progress.snapshot().localAdmitted,"retained captured log bypassed pending budget");
}
// A committed peer delete of a child that a pending local Reorder lists, or a
// peer move of it out of the parent, is applied; the replayed Reorder skips the
// child and orders the current children it names, as a service that orders
// lists per child list commits it, and the rest of the intent stays visible. A
// service that stores lists verbatim rejects the local intent instead: that is
// a definitive rejection of the write, not a refusal of the peer's committed
// update, and the authoritative scene stays the service's.
void staleReorderReplay() {
    for (const bool move : {false,true}) {
    context = std::string("pending reorder after a peer ") + (move ? "move" : "delete") + " of a listed child";
    const auto command = [](Command::Kind kind, uint64_t id, uint64_t parent, const char* name) {
        Command c{}; c.kind = kind; c.section = id; c.parent = parent; c.name = name;
        if (kind == Command::Kind::Create) c.sectionType = SdfSpecTypePrim;
        return c;
    };
    EnvelopeInput setup; setup.commands = {command(Command::Kind::Create,3,2,"X"),command(Command::Kind::Create,4,2,"Y"),
        command(Command::Kind::Create,5,2,"Z")};
    Fixture f({},128,buildSceneCandidate(initialScene(),encodeEnvelope(setup)));
    const auto children = [](const WireScenePtr& scene, std::initializer_list<const char*> names) {
        std::vector<TfToken> expected; for (const auto* name : names) expected.emplace_back(name);
        return scene->data()->Get(SdfPath("/P"),SdfChildrenKeys->PrimChildren) == VtValue(expected);
    };
    EnvelopeInput local; Command order{}; order.kind = Command::Kind::Reorder; order.section = 2; order.children = {5,4,3};
    local.commands = {order}; local.fields = {{2,0,"stamp",{}}};
    const auto bytes = encodeUsdEnvelope(local,DecodedValues{{VtValue(5)},{}}); const auto id = f.local(bytes);
    check(children(f.journal.snapshot()->visible,{"Z","Y","X"}) && number(f.journal.snapshot()->visible,"stamp") == 5,
          "pending reorder not visible");
    EnvelopeInput peer;
    if (move) { auto out = command(Command::Kind::Move,4,1,"Escaped"); out.oldParent = 2; peer.commands = {out}; }
    else peer.commands = {command(Command::Kind::Delete,4,2,"")};
    const auto peerBytes = encodeEnvelope(peer); f.applyPeer(f.peer(peerBytes,11),peerBytes);
    const auto rebased = f.journal.snapshot();
    check(rebased->applied == 11 && children(rebased->authoritative,{"X","Z"}) &&
          (move ? rebased->authoritative->pathForSection(4) == SdfPath("/Escaped") : rebased->authoritative->pathForSection(4).IsEmpty()) &&
          rebased->pending.size() == 1 && rebased->pending[0].id == id && rebased->pending[0].intent->bytes == bytes &&
          rebased->pending[0].replayed,"committed peer update was refused or the stale intent changed");
    // The replay skips the removed child and orders the rest; the intent's field stays visible.
    check(children(rebased->visible,{"Z","X"}) && number(rebased->visible,"stamp") == 5 &&
          rebased->visible->pathForSection(4) == rebased->authoritative->pathForSection(4),
          "replayed reorder did not skip the removed child and order the rest");
    // Applying the same bytes as the service's commit gives the predicted scene.
    const auto committed = buildSceneCandidate(rebased->authoritative,bytes,{},nullptr,WireSceneBatchPolicy::Delivered);
    check(inspect(committed) == inspect(rebased->visible),"prediction differs from the per-list service's commit");
    // A later intent replays onto the predicted state.
    const auto later = f.local(edit(8,"later"));
    check(number(f.journal.snapshot()->visible,"later") == 8 && number(f.journal.snapshot()->visible,"stamp") == 5 &&
          children(f.journal.snapshot()->visible,{"Z","X"}),"later intent did not replay onto the predicted state");
    (void)later;
    // A verbatim service rejects the write: the rejection is definitive and stops issuance, the intent stays
    // pending and the authoritative scene stays the service's.
    check(f.progress.takeWrite().id == id,"stale reorder was not issued");
    f.progress.completeWrite(id,OMNI_USD_LIVE_INVALID_ARGUMENT);
    check(f.retained(id).disposition == LayerProgress::Disposition::Rejected,"service rejection was not definitive");
    check(f.progress.snapshot().result == OMNI_USD_LIVE_INVALID_ARGUMENT && !f.progress.takeWrite().id,
          "a rejected stale reorder did not stop issuance with INVALID_ARGUMENT");
    check(inspect(f.journal.snapshot()->authoritative) == inspect(rebased->authoritative) &&
          f.journal.snapshot()->pending.size() == 2,"the rejection changed the journal");
    // Local content stays strict: the same list is malformed.
    bool rejected = false;
    try { (void)buildSceneCandidate(rebased->authoritative,bytes); }
    catch (const DecodeError& error) { rejected = error.kind == ErrorKind::Malformed; }
    check(rejected,"local reorder naming a removed child was accepted");
    }
    // A newly admitted raw intent is validated strictly.
    EnvelopeInput setup; Command x{}; x.kind = Command::Kind::Create; x.section = 4; x.parent = 2; x.name = "Y";
    x.sectionType = SdfSpecTypePrim; setup.commands = {x};
    Fixture strict({},128,buildSceneCandidate(initialScene(),encodeEnvelope(setup)));
    const Saved untouched(strict.journal);
    for (const auto& ids : {std::vector<uint64_t>{4,99},std::vector<uint64_t>{4,4}}) {
        EnvelopeInput nonchild; Command wrong{}; wrong.kind = Command::Kind::Reorder; wrong.section = 2; wrong.children = ids;
        nonchild.commands = {wrong};
        bool rejected = false;
        try { strict.local(encodeEnvelope(nonchild)); }
        catch (const DecodeError& error) { rejected = error.kind == ErrorKind::Malformed; }
        check(rejected && !strict.submits,"raw reorder naming a non-child or a repeat was not rejected as malformed");
    }
    untouched.unchanged(strict.journal);
}
// A committed peer Create of a section ID that a pending local Create also
// creates is applied; the service rejects the local Create of the live ID, so
// the prediction drops the whole intent, and a later peer Delete that retires
// the ID makes the local Create visible again. The service's rejection, not
// the peer's committed update, decides the layer's outcome.
void liveIdentityCreateReplay() {
    context = "pending create after a peer create of the same identity";
    const auto create = [](uint64_t id, const char* name) {
        Command c{}; c.kind = Command::Kind::Create; c.section = id; c.parent = 2; c.name = name;
        c.sectionType = SdfSpecTypePrim; return c;
    };
    Fixture f;
    EnvelopeInput local; local.commands = {create(9,"New")}; local.fields = {{9,0,"marker",{}}};
    const auto bytes = encodeUsdEnvelope(local,DecodedValues{{VtValue(73)},{}}); const auto id = f.local(bytes);
    check(f.journal.snapshot()->visible->sectionForPath(SdfPath("/P/New")) == 9,"pending create not visible");
    EnvelopeInput peer; peer.commands = {create(9,"Collision")};
    const auto peerBytes = encodeEnvelope(peer); f.applyPeer(f.peer(peerBytes,11),peerBytes);
    const auto rebased = f.journal.snapshot();
    check(rebased->applied == 11 && rebased->authoritative->sectionForPath(SdfPath("/P/Collision")) == 9 &&
          rebased->pending.size() == 1 && rebased->pending[0].id == id && rebased->pending[0].intent->bytes == bytes,
          "committed peer create was refused or the colliding intent changed");
    check(rebased->visible->sectionForPath(SdfPath("/P/New")) == 0 && !rebased->visible->data()->Has(SdfPath("/P/Collision"),TfToken("marker")) &&
          inspect(rebased->visible) == inspect(rebased->authoritative),"predicted-rejected create partly applied");
    // A peer Delete retires the ID; the service accepts a Create of a retired ID.
    EnvelopeInput retire; Command erase{}; erase.kind = Command::Kind::Delete; erase.section = 9; erase.parent = 2;
    retire.commands = {erase};
    const auto retireBytes = encodeEnvelope(retire); f.applyPeer(f.peer(retireBytes,12),retireBytes);
    const auto retired = f.journal.snapshot()->visible;
    check(retired->sectionForPath(SdfPath("/P/New")) == 9 &&
          retired->data()->Get(SdfPath("/P/New"),TfToken("marker")) == VtValue(73),
          "pending create of a retired identity is not visible");
    check(f.progress.takeWrite().id == id,"colliding create was not issued");
    f.progress.completeWrite(id,OMNI_USD_LIVE_INVALID_ARGUMENT);
    check(f.retained(id).disposition == LayerProgress::Disposition::Rejected,"service rejection was not definitive");
    // Local content stays strict: the same Create of a live ID is malformed.
    bool rejected = false;
    try { (void)buildSceneCandidate(rebased->authoritative,bytes); }
    catch (const DecodeError& error) { rejected = error.kind == ErrorKind::Malformed; }
    check(rejected,"local create of a live identity was accepted");
}
void authoredCreateAdmission() {
    const auto create = [](uint64_t id,uint64_t parent,const char* name) {
        Command c{}; c.kind=Command::Kind::Create; c.section=id; c.parent=parent;
        c.name=name; c.sectionType=SdfSpecTypePrim; return c;
    };
    const auto add = [](SdfAbstractData& data,const char* path,int value) {
        const SdfPath target(path), parent=target.GetParentPath();
        auto children=data.Get(parent,SdfChildrenKeys->PrimChildren);
        auto names=children.IsEmpty()?std::vector<TfToken>{}:children.Get<std::vector<TfToken>>();
        names.push_back(target.GetNameToken());
        data.CreateSpec(target,SdfSpecTypePrim);
        data.Set(target,TfToken("marker"),VtValue(value));
        data.Set(parent,SdfChildrenKeys->PrimChildren,VtValue(names));
    };
    // Accepted bytes must retain their parent and new identity when peers move
    // that parent or insert/reorder siblings. A peer delete of the parent
    // supersedes the Create; a peer occupant of its name is replaced by it. A
    // peer reuse of its identity applies and predicts the service's rejection
    // of the Create, which stays pending with every owner.
    for(unsigned scenario=0;scenario<5;++scenario) {
        context="captured Create peer scenario "+std::to_string(scenario);
        size_t generated=0;
        auto ids=std::make_shared<SectionIdAllocator>(4096,64,[&] { return uint64_t(1000)+generated++; });
        Fixture f({},128,{},ids); const Saved base(f.journal);
        const auto original=capturedLog(base.pointer->visible,[&](SdfAbstractData& data) { add(data,"/P/New",7); });
        const auto id=f.journal.submitCaptured(base.pointer->visible,original,f.submitter());
        const Saved admitted(f.journal);
        const auto intent=admitted.owners.front(); const auto bytes=intent->bytes;
        const auto compiled=decodeEnvelope(bytes);
        check(generated==1 && compiled.commands.size()==1 && compiled.commands[0].section==1000 &&
              compiled.commands[0].parent==2 && compiled.commands[0].kind==Command::Kind::Create &&
              compiled.fields.size()==1 && compiled.fields[0].name=="marker",
              "Create emitted a membership override, changed identity or allocated twice");
        check(id && intent->authored && intent->authored->operations()==original->operations &&
              intent->baseRevision==base.pointer->revision && f.submits==1,"Create lost original owners or admission identity");
        EnvelopeInput peer;
        if(scenario==0) {
            Command move{}; move.kind=Command::Kind::Move; move.section=2;
            move.oldParent=1; move.parent=1; move.name="Moved"; peer.commands={move};
        } else if(scenario==1) peer.commands={create(3,2,"Peer")};
        else if(scenario==2) {
            Command erase{}; erase.kind=Command::Kind::Delete; erase.section=2; erase.parent=1;
            peer.commands={erase};
        } else peer.commands={create(scenario==3?3:1000,2,scenario==3?"New":"Other")};
        const auto peerBytes=encodeEnvelope(peer); const auto work=f.peer(peerBytes,11);
        if(scenario==4) {
            f.applyPeer(work,peerBytes);
            const auto rebased=f.journal.snapshot();
            check(rebased->applied==11 && rebased->authoritative->sectionForPath(SdfPath("/P/Other"))==1000 &&
                  inspect(rebased->visible)==inspect(rebased->authoritative) && rebased->pending.size()==1 &&
                  rebased->pending[0].intent==intent && intent->bytes==bytes && intent->authored->operations()==original->operations &&
                  f.progress.snapshot().applied==11 && f.progress.retained().size()==1 && generated==1,
                  "Create identity reuse refused the peer, applied the Create or discarded local work");
            continue;
        }
        f.applyPeer(work,peerBytes);
        const auto rebased=f.journal.snapshot();
        const SdfPath newPath(scenario==0?"/Moved/New":"/P/New");
        if(scenario==2) check(rebased->visible->sectionForPath(newPath)==0 && rebased->visible->sectionForPath(SdfPath("/P"))==0 &&
              rebased->pending.size()==1 && rebased->pending[0].intent==intent && intent->bytes==bytes && generated==1,
              "superseded Create stayed visible or changed accepted bytes/identity");
        else check(rebased->visible->sectionForPath(newPath)==1000 &&
              rebased->visible->data()->Get(newPath,TfToken("marker"))==VtValue(7) &&
              rebased->pending[0].intent==intent && intent->bytes==bytes && generated==1,
              "Create rebase changed accepted bytes/identity or lost authored value");
        if(scenario==1) check(rebased->visible->data()->Get(SdfPath("/P"),SdfChildrenKeys->PrimChildren)==
              VtValue(std::vector<TfToken>{TfToken("Peer"),TfToken("New")}),"Create lost peer sibling/order");
        const auto expected=inspect(rebased->visible);
        f.applyLocal(f.ack(id,12));
        const auto confirmed=f.journal.snapshot();
        check(inspect(confirmed->authoritative)==expected && inspect(confirmed->visible)==expected &&
              confirmed->pending.empty() && confirmed->localRetired==1 && confirmed->applied==12 &&
              !confirmed->pendingBytes && generated==1,"Create ACK replay rejected own identity or lost work");
        check(inspect(base.pointer->visible)==base.visible && inspect(admitted.pointer->visible)==admitted.visible,
              "Create publication mutated pinned old snapshots");
    }
    {
        // Admission-time supersession: the peer delete of the parent applies
        // between capture and submit. The Create is accepted, never visible,
        // and retires on its own acknowledgement.
        context="captured Create admitted after a peer delete of its parent";
        size_t generated=0;
        auto ids=std::make_shared<SectionIdAllocator>(4096,64,[&] { return uint64_t(1000)+generated++; });
        Fixture f({},128,{},ids);
        const auto captureBase=f.journal.snapshot()->visible;
        const auto sealed=capturedLog(captureBase,[&](SdfAbstractData& data) { add(data,"/P/New",7); });
        const auto peerBytes=deleteParent(); f.applyPeer(f.peer(peerBytes,11),peerBytes);
        WireScenePtr applied;
        const auto id=f.journal.submitCaptured(captureBase,sealed,f.submitter(),&applied);
        const auto admitted=f.journal.snapshot();
        check(applied && applied->data()->Get(SdfPath("/P/New"),TfToken("marker"))==VtValue(7),
              "captured Create lost the host content it was authored on");
        check(id && f.submits==1 && generated==1 && admitted->pending.size()==1 &&
              admitted->pending[0].intent->authored &&
              admitted->visible->sectionForPath(SdfPath("/P/New"))==0 &&
              admitted->visible->sectionForPath(SdfPath("/P"))==0,
              "Create superseded before submit was rejected or became visible");
        const auto expected=inspect(admitted->visible);
        f.applyLocal(f.ack(id,12));
        const auto confirmed=f.journal.snapshot();
        check(confirmed->pending.empty() && confirmed->localRetired==1 && confirmed->applied==12 &&
              inspect(confirmed->authoritative)==expected && inspect(confirmed->visible)==expected,
              "superseded Create did not retire on its own acknowledgement");
    }
    {
        context="dependent captured creates and erase/recreate preserve distinct identities";
        size_t generated=0;
        auto ids=std::make_shared<SectionIdAllocator>(4096,64,[&] {
            // The first entropy result collides with the initial scene.
            return generated++==0?uint64_t(2):uint64_t(999)+generated;
        });
        Fixture f({},128,{},ids);
        // Each log is authored on, and admitted against, the current visible scene.
        const auto admit=[&](const auto& author) {
            const auto base=f.journal.snapshot()->visible;
            return f.journal.submitCaptured(base,capturedLog(base,author),f.submitter());
        };
        const auto a=admit([&](SdfAbstractData& data) { add(data,"/P/New",7); });
        const auto parent=f.journal.snapshot()->visible->sectionForPath(SdfPath("/P/New"));
        const auto b=admit([&](SdfAbstractData& data) { add(data,"/P/New/Child",8); });
        const auto pending=f.journal.snapshot();
        const auto child=pending->visible->sectionForPath(SdfPath("/P/New/Child"));
        check(parent>2 && child>parent && generated==3 && pending->pending[1].intent->dependencyCutoff==1,
              "dependent Create reused an observed ID or lost its dependency");
        // A same-type repeated Create keeps the identity; its envelope is empty,
        // so the captured log admits nothing.
        const auto c=admit([](SdfAbstractData& data) { data.CreateSpec(SdfPath("/P/New/Child"),SdfSpecTypePrim); });
        check(!c && generated==3 && f.journal.snapshot()==pending &&
              f.journal.snapshot()->visible->sectionForPath(SdfPath("/P/New/Child"))==child,
              "same-type repeated Create allocated a new identity or admitted work");
        f.applyLocal(f.ack(a,11)); f.applyLocal(f.ack(b,12));
        check(inspect(f.journal.snapshot()->authoritative)==inspect(pending->visible),"dependent ACKs lost graph");
        const auto d=admit([&](SdfAbstractData& data) {
            data.EraseSpec(SdfPath("/P/New/Child"));
            data.Erase(SdfPath("/P/New"),SdfChildrenKeys->PrimChildren);
            add(data,"/P/New/Child",9);
        });
        const auto replaced=f.journal.snapshot()->visible;
        check(replaced->sectionForPath(SdfPath("/P/New/Child"))!=child && generated==4,
              "erase/recreate reused a retired identity");
        f.applyLocal(f.ack(d,13));
        check(inspect(f.journal.snapshot()->authoritative)==inspect(replaced) &&
              f.journal.snapshot()->localRetired==3,"replacement ACK lost new identity/value");
    }
}
void authoredStructuralAdmission() {
    const auto tokens = [](std::initializer_list<const char*> names) {
        std::vector<TfToken> result;
        for (const auto* name : names) result.emplace_back(name);
        return VtValue(result);
    };
    const auto create = [](uint64_t id,uint64_t parent,const char* name) {
        Command result{}; result.kind=Command::Kind::Create; result.section=id;
        result.parent=parent; result.name=name; result.sectionType=SdfSpecTypePrim; return result;
    };
    const auto originalLog = [](const std::shared_ptr<const SdfOperationLog>& original,
                                const std::vector<SdfOperation>& expected) {
        check(original->operations->size()==expected.size(),"retained structural log lost operations");
        for(size_t i=0;i<expected.size();++i) {
            const auto& a=original->operations->at(i); const auto& b=expected[i];
            check(a.kind==b.kind && a.path==b.path && a.destination==b.destination &&
                  a.specType==b.specType && a.field==b.field && a.key==b.key &&
                  a.time==b.time && a.value==b.value && a.replacement==b.replacement,
                  "retained structural operation changed");
        }
    };
    using Graph = std::map<uint64_t,std::pair<uint64_t,uint32_t>>;
    const auto expect = [](const WireScenePtr& scene,const State& state,
                           const std::map<SdfPath,uint64_t>& identities,const Graph& graph) {
        check(inspect(scene)==state && scene->sectionCount()==state.size(),"exact structural scene differs");
        check(identities.size()==state.size() && graph.size()==state.size(),"incomplete structural test oracle");
        for(const auto& entry:identities)
            check(scene->sectionForPath(entry.first)==entry.second && scene->pathForSection(entry.second)==entry.first,
                  "structural replay lost or retargeted section identity");
        size_t visited=0;
        scene->visitSections(32,[&](const WireSectionView& section) {
            const auto found=graph.find(section.id);
            check(found!=graph.end() && section.parent==found->second.first && section.order==found->second.second,
                  "structural replay changed parent or sparse sibling slot");
            ++visited; return true;
        });
        check(visited==graph.size(),"structural graph traversal incomplete");
    };
    for(bool move:{false,true}) for(bool escape:{false,true}) {
        context=std::string("admitted recursive ")+(move?"move":"delete")+(escape?" preserves peer move-out":" includes peer insertion");
        EnvelopeInput initial;
        initial.commands={create(2,1,"P"),create(3,2,"Child"),create(4,1,"Q")};
        initial.fields={{2,0,"value",{}},{3,1,"marker",{}},{4,2,"marker",{}}};
        const auto source=buildSceneCandidate({},encodeUsdEnvelope(initial,{{VtValue(7),VtValue(11),VtValue(44)},{}}));
        size_t generated=0;
        auto ids=std::make_shared<SectionIdAllocator>(4096,64,[&] { return uint64_t(1000)+generated++; });
        Fixture f({},128,source,ids);
        const Saved base(f.journal);
        const auto original=capturedLog(base.pointer->visible,[&](SdfAbstractData& data) {
            if(move) {
                // Sdf operations are one-spec; move descendants explicitly in the
                // captured log before moving their temporarily disconnected parent.
                data.MoveSpec(SdfPath("/P/Child"),SdfPath("/Moved/Child"));
                data.MoveSpec(SdfPath("/P"),SdfPath("/Moved"));
                data.Set(SdfPath("/"),SdfChildrenKeys->PrimChildren,tokens({"Moved","Q"}));
            } else {
                data.EraseSpec(SdfPath("/P/Child"));
                data.EraseSpec(SdfPath("/P"));
                data.Set(SdfPath("/"),SdfChildrenKeys->PrimChildren,tokens({"Q"}));
            }
        });
        const auto operations=*original->operations;
        const auto id=f.journal.submitCaptured(base.pointer->visible,original,f.submitter());
        const auto admitted=f.journal.snapshot();
        check(id && f.submits==1 && !generated && admitted->pending.size()==1,
              "move/delete admission failed or allocated an unnecessary section identity");
        const auto intent=admitted->pending.front().intent;
        const auto bytes=intent->bytes;
        const auto envelope=decodeEnvelope(bytes);
        check(envelope.commands.size()==1 && envelope.commands.front().section==2 &&
              envelope.commands.front().kind==(move?Command::Kind::Move:Command::Kind::Delete) &&
              envelope.fields.empty() && envelope.samples.empty(),
              "captured subtree emitted descendant targets, scratch Create or complete child-field overrides");
        check(intent->authored && intent->authored->operations()==original->operations && intent->localOrdinal==1 &&
              intent->dependencyCutoff==0 && intent->baseRevision==base.pointer->revision,
              "accepted structural intent lost original ownership or dependency evidence");
        const auto rootChildren=move?tokens({"Moved","Q"}):tokens({"Q"});
        State optimistic{{SdfPath("/"),{SdfSpecTypePseudoRoot,{{SdfChildrenKeys->PrimChildren,rootChildren}}}},
            {SdfPath("/Q"),{SdfSpecTypePrim,{{TfToken("marker"),VtValue(44)}}}}};
        std::map<SdfPath,uint64_t> optimisticIds{{SdfPath("/"),1},{SdfPath("/Q"),4}};
        Graph optimisticGraph{{1,{0,0}},{4,{1,1}}};
        if(move) {
            optimistic.emplace(SdfPath("/Moved"),Record{SdfSpecTypePrim,{{TfToken("value"),VtValue(7)},
                {SdfChildrenKeys->PrimChildren,tokens({"Child"})}}});
            optimistic.emplace(SdfPath("/Moved/Child"),Record{SdfSpecTypePrim,{{TfToken("marker"),VtValue(11)}}});
            optimisticIds.emplace(SdfPath("/Moved"),2); optimisticIds.emplace(SdfPath("/Moved/Child"),3);
            optimisticGraph.emplace(2,std::make_pair(1,0)); optimisticGraph.emplace(3,std::make_pair(2,0));
        }
        expect(admitted->visible,optimistic,optimisticIds,optimisticGraph);
        check(inspect(admitted->authoritative)==base.authoritative && admitted->applied==10,
              "optimistic structural admission advanced authoritative state");
        EnvelopeInput remote; DecodedValues values;
        if(escape) {
            Command escaped{}; escaped.kind=Command::Kind::Move; escaped.section=3; escaped.oldParent=2;
            escaped.parent=4; escaped.name="Escaped"; remote.commands={escaped};
        } else {
            remote.commands={create(30,2,"Peer")}; remote.fields={{30,0,"marker",{}}}; values.fields={VtValue(99)};
        }
        const auto peerBytes=encodeUsdEnvelope(remote,values);
        f.applyPeer(f.peer(peerBytes,11),peerBytes);
        const auto replayed=f.journal.snapshot();
        State authoritative{{SdfPath("/"),{SdfSpecTypePseudoRoot,{{SdfChildrenKeys->PrimChildren,tokens({"P","Q"})}}}},
            {SdfPath("/P"),{SdfSpecTypePrim,{{TfToken("value"),VtValue(7)}}}},
            {SdfPath("/Q"),{SdfSpecTypePrim,{{TfToken("marker"),VtValue(44)}}}}};
        std::map<SdfPath,uint64_t> authoritativeIds{{SdfPath("/"),1},{SdfPath("/P"),2},{SdfPath("/Q"),4}};
        Graph authoritativeGraph{{1,{0,0}},{2,{1,0}},{4,{1,1}}};
        State expected=optimistic;
        auto expectedIds=optimisticIds; auto expectedGraph=optimisticGraph;
        if(escape) {
            authoritative.at(SdfPath("/Q")).fields[SdfChildrenKeys->PrimChildren]=tokens({"Escaped"});
            authoritative.emplace(SdfPath("/Q/Escaped"),Record{SdfSpecTypePrim,{{TfToken("marker"),VtValue(11)}}});
            authoritativeIds.emplace(SdfPath("/Q/Escaped"),3); authoritativeGraph.emplace(3,std::make_pair(4,0));
            expected.at(SdfPath("/Q")).fields[SdfChildrenKeys->PrimChildren]=tokens({"Escaped"});
            expected.emplace(SdfPath("/Q/Escaped"),Record{SdfSpecTypePrim,{{TfToken("marker"),VtValue(11)}}});
            if(move) {
                expected.at(SdfPath("/Moved")).fields.erase(SdfChildrenKeys->PrimChildren);
                expected.erase(SdfPath("/Moved/Child")); expectedIds.erase(SdfPath("/Moved/Child")); expectedGraph.erase(3);
            }
            expectedIds.emplace(SdfPath("/Q/Escaped"),3); expectedGraph.emplace(3,std::make_pair(4,0));
        } else {
            authoritative.at(SdfPath("/P")).fields[SdfChildrenKeys->PrimChildren]=tokens({"Child","Peer"});
            authoritative.emplace(SdfPath("/P/Child"),Record{SdfSpecTypePrim,{{TfToken("marker"),VtValue(11)}}});
            authoritative.emplace(SdfPath("/P/Peer"),Record{SdfSpecTypePrim,{{TfToken("marker"),VtValue(99)}}});
            authoritativeIds.emplace(SdfPath("/P/Child"),3); authoritativeIds.emplace(SdfPath("/P/Peer"),30);
            authoritativeGraph.emplace(3,std::make_pair(2,0)); authoritativeGraph.emplace(30,std::make_pair(2,1));
            if(move) {
                expected.at(SdfPath("/Moved")).fields[SdfChildrenKeys->PrimChildren]=tokens({"Child","Peer"});
                expected.emplace(SdfPath("/Moved/Peer"),Record{SdfSpecTypePrim,{{TfToken("marker"),VtValue(99)}}});
                expectedIds.emplace(SdfPath("/Moved/Peer"),30); expectedGraph.emplace(30,std::make_pair(2,1));
            }
        }
        expect(replayed->authoritative,authoritative,authoritativeIds,authoritativeGraph);
        expect(replayed->visible,expected,expectedIds,expectedGraph);
        check(replayed->applied==11 && replayed->pending.size()==1 && replayed->pending.front().id==id &&
              replayed->pending.front().intent==intent && intent->bytes==bytes &&
              intent->authored->operations()==original->operations,
              "peer replay rewrote or retired original structural work");
        originalLog(original,operations);
        f.applyLocal(f.ack(id,12));
        const auto confirmed=f.journal.snapshot();
        expect(confirmed->authoritative,expected,expectedIds,expectedGraph);
        expect(confirmed->visible,expected,expectedIds,expectedGraph);
        check(confirmed->pending.empty() && !confirmed->pendingBytes && confirmed->applied==12 &&
              confirmed->localAdmitted==1 && confirmed->localRetired==1 && f.progress.snapshot().localRetired==1,
              "structural ACK did not retire exactly the admitted original work");
        originalLog(original,operations);
        expect(admitted->visible,optimistic,optimisticIds,optimisticGraph);
        check(inspect(base.pointer->visible)==base.visible && inspect(base.pointer->authoritative)==base.authoritative,
              "later structural publication mutated original base owners");
    }
    // A plain final-child deletion can erase the field. Explicit empty is a
    // complete membership assertion and stays outside this admission subset.
    {
        context="last-child structural deletion preserves absent field";
        Fixture f; const auto base=f.journal.snapshot()->visible;
        const auto log=capturedLog(base,[](SdfAbstractData& data) {
            data.EraseSpec(SdfPath("/P"));
            data.Erase(SdfPath("/"),SdfChildrenKeys->PrimChildren);
        });
        const auto id=f.journal.submitCaptured(base,log,f.submitter());
        check(id && !f.journal.snapshot()->visible->data()->Has(SdfPath("/"),SdfChildrenKeys->PrimChildren),
              "last-child deletion did not preserve absent child field");
        f.applyLocal(f.ack(id,11));
        check(f.journal.snapshot()->authoritative->sectionCount()==1 && f.journal.snapshot()->pending.empty(),
              "plain last-child deletion did not confirm");
    }
    for(unsigned variant=1;variant<3;++variant) {
        context=variant==1?"explicit-empty structural override rejected":"scratch namespace swap rejected";
        size_t generated=0;
        auto ids=std::make_shared<SectionIdAllocator>(4096,64,[&] { return uint64_t(1000)+generated++; });
        EnvelopeInput extra; extra.commands={create(4,1,"Q")};
        const auto source=variant==2?buildSceneCandidate(initialScene(),encodeEnvelope(extra)):initialScene();
        Fixture f({},128,source,ids); const auto base=f.journal.snapshot()->visible;
        const auto original=capturedLog(base,[&](SdfAbstractData& data) {
            if(variant==1) {
                data.EraseSpec(SdfPath("/P"));
                data.Set(SdfPath("/"),SdfChildrenKeys->PrimChildren,tokens({}));
            } else {
                data.MoveSpec(SdfPath("/P"),SdfPath("/Temporary"));
                data.MoveSpec(SdfPath("/Q"),SdfPath("/P"));
                data.MoveSpec(SdfPath("/Temporary"),SdfPath("/Q"));
                data.Set(SdfPath("/"),SdfChildrenKeys->PrimChildren,tokens({"Q","P"}));
            }
        });
        const auto operations=*original->operations;
        const Saved before(f.journal);
        unsupportedShape(variant==1?UnsupportedOperation::Reason::MultiUpdate:UnsupportedOperation::Reason::MoveCycle,
                         [&] { (void)f.journal.submitCaptured(base,original,f.submitter()); });
        before.unchanged(f.journal);
        check(!f.submits && !f.progress.snapshot().localAdmitted && !f.progress.snapshot().retained,
              "structural rejection submitted work");
        originalLog(original,operations);
        if(variant!=2) check(!generated,"non-scratch rejection allocated section IDs");
    }
    // A captured log is already applied in the host layer. Records addressing
    // an identity it removes are dropped and only the removal compiles.
    const auto removals=[](const std::string& bytes,const std::set<uint64_t>& expected) {
        const auto envelope=decodeEnvelope(bytes); std::set<uint64_t> deleted;
        for(const auto& command:envelope.commands) if(command.kind==Command::Kind::Delete) deleted.insert(command.section);
        return envelope.commands.size()==expected.size() && deleted==expected && envelope.fields.empty() && envelope.samples.empty();
    };
    for(bool sample:{false,true}) {
        context=sample?"sample on removed descendant is dropped":"field on removed descendant is dropped";
        EnvelopeInput initial; auto child=create(3,2,sample?"sample":"Child");
        if(sample) child.sectionType=SdfSpecTypeAttribute;
        initial.commands={child}; DecodedValues values;
        if(sample) {
            initial.fields={{3,0,SdfFieldKeys->TypeName.GetString(),{}}};
            values.fields={VtValue(TfToken("double"))};
        }
        const auto source=buildSceneCandidate(initialScene(),encodeUsdEnvelope(initial,values));
        Fixture f({},128,source); const auto base=f.journal.snapshot()->visible;
        const SdfPath childPath(sample?"/P.sample":"/P/Child");
        const auto original=capturedLog(base,[&](SdfAbstractData& data) {
            if(sample) data.SetTimeSample(childPath,1.25,VtValue(9.5));
            else data.Set(childPath,TfToken("marker"),VtValue(9));
            data.EraseSpec(childPath);
            data.EraseSpec(SdfPath("/P"));
            data.Erase(SdfPath("/"),SdfChildrenKeys->PrimChildren);
        });
        const auto operations=*original->operations;
        const auto id=f.journal.submitCaptured(base,original,f.submitter());
        const auto admitted=f.journal.snapshot();
        check(id && f.submits==1 && admitted->pending.size()==1 && removals(admitted->pending[0].intent->bytes,{2}) &&
              admitted->visible->sectionForPath(SdfPath("/P"))==0 && admitted->visible->sectionCount()==1,
              "removed-descendant value was emitted or the removal was not admitted");
        originalLog(original,operations);
        f.applyLocal(f.ack(id,11));
        check(f.journal.snapshot()->pending.empty() && f.journal.snapshot()->authoritative->sectionCount()==1,
              "removed-descendant removal did not confirm");
    }
    for(bool removedMove:{false,true}) {
        context=removedMove?"move into removed subtree is dropped":"child order on removed descendant is dropped";
        EnvelopeInput initial; initial.commands={create(3,2,"C")};
        if(removedMove) initial.commands.push_back(create(4,1,"Outside"));
        else {
            initial.commands.push_back(create(4,3,"A"));
            initial.commands.push_back(create(5,3,"B"));
        }
        const auto source=buildSceneCandidate(initialScene(),encodeEnvelope(initial));
        Fixture f({},128,source); const auto base=f.journal.snapshot()->visible;
        const auto original=capturedLog(base,[&](SdfAbstractData& data) {
            if(removedMove) {
                // The move binds ID4 into C before the subtree removal; only
                // the Deletes of P and of the moved-in ID4 compile.
                data.MoveSpec(SdfPath("/Outside"),SdfPath("/P/C/Inside"));
                data.EraseSpec(SdfPath("/P/C/Inside"));
            } else {
                data.Set(SdfPath("/P/C"),SdfChildrenKeys->PrimChildren,tokens({"B","A"}));
                data.EraseSpec(SdfPath("/P/C/A"));
                data.EraseSpec(SdfPath("/P/C/B"));
            }
            data.EraseSpec(SdfPath("/P/C"));
            data.EraseSpec(SdfPath("/P"));
            data.Erase(SdfPath("/"),SdfChildrenKeys->PrimChildren);
        });
        const auto operations=*original->operations;
        const auto id=f.journal.submitCaptured(base,original,f.submitter());
        const auto admitted=f.journal.snapshot();
        const std::set<uint64_t> deleted=removedMove?std::set<uint64_t>{2,4}:std::set<uint64_t>{2};
        check(id && f.submits==1 && admitted->pending.size()==1 && removals(admitted->pending[0].intent->bytes,deleted) &&
              admitted->visible->sectionCount()==1,
              "removed-structure records were emitted or the removal was not admitted");
        originalLog(original,operations);
        f.applyLocal(f.ack(id,11));
        check(f.journal.snapshot()->pending.empty() && f.journal.snapshot()->authoritative->sectionCount()==1,
              "removed-structure removal did not confirm");
    }
}
void capturedRejectionClassification() {
    context = "captured admission marks limits no wait relieves";
    const auto edit = [](const Fixture& f, int value) {
        auto log = recordingBacking(f.initial);
        log->Set(SdfPath("/P"),TfToken("value"),VtValue(value));
        return recordedLog(*log);
    };
    // An intent larger than the whole replay quota never fits.
    SceneJournal::Limits tiny; tiny.maxReplayBytes = 16;
    Fixture f(tiny); const Saved before(f.journal);
    bool definitive = false;
    try { (void)f.journal.submitCaptured(f.initial,edit(f,1),f.submitter()); }
    catch (const CapturedRejection& rejection) { definitive = rejection.result == OMNI_USD_LIVE_RESOURCE_EXHAUSTED; }
    check(definitive && f.submits == 0,"oversized captured intent was not a definitive rejection");
    before.unchanged(f.journal);
    // The same limit keeps its plain result for an uncaptured submission.
    bool plain = false;
    try { (void)f.local(std::string(64,'\0')); }
    catch (OmniUsdLiveResult result) { plain = result == OMNI_USD_LIVE_RESOURCE_EXHAUSTED; }
    catch (...) {}
    check(plain,"uncaptured oversized submission changed its rejection");
    // The pending-intent limit clears as pending work settles: plain result.
    SceneJournal::Limits one; one.maxPending = 1;
    Fixture g(one);
    check(g.journal.submitCaptured(g.initial,edit(g,2),g.submitter()) != 0,"first captured intent was rejected");
    bool transient = false;
    try { (void)g.journal.submitCaptured(g.initial,edit(g,3),g.submitter()); }
    catch (OmniUsdLiveResult result) { transient = result == OMNI_USD_LIVE_RESOURCE_EXHAUSTED; }
    check(transient && g.submits == 1,"pending-intent limit was not a transient rejection");
    // A compile limit that depends only on the buffer never clears.
    auto twoRecords = [](const Fixture& h) {
        auto log = recordingBacking(h.initial);
        log->Set(SdfPath("/P"),TfToken("value"),VtValue(4));
        log->Set(SdfPath("/P"),TfToken("other"),VtValue(5));
        return recordedLog(*log);
    };
    SceneJournal::Limits operations; operations.authoring.maxOperations = 1;
    Fixture o(operations); const Saved unchangedOperations(o.journal);
    definitive = false;
    try { (void)o.journal.submitCaptured(o.initial,twoRecords(o),o.submitter()); }
    catch (const CapturedRejection& rejection) { definitive = rejection.result == OMNI_USD_LIVE_RESOURCE_EXHAUSTED; }
    catch (...) {}
    check(definitive && o.submits == 0,"captured operation limit was not a definitive rejection");
    unchangedOperations.unchanged(o.journal);
    // The service's entry limit is definitive and names its counts for the
    // flush's runtime error.
    SceneJournal::Limits entries; entries.authoring.entries.maxEntries = 1;
    Fixture e(entries); const Saved unchangedEntries(e.journal);
    bool named = false;
    try { (void)e.journal.submitCaptured(e.initial,twoRecords(e),e.submitter()); }
    catch (const DeltaEntryLimitExceeded& refused) {
        named = refused.reason() == DeltaEntryLimitExceeded::Reason::DeltaTooLarge && refused.counts().setFields == 2 &&
            refused.counts().limit == 1 && refused.counts().actual == 2;
    } catch (...) {}
    check(named && e.submits == 0,"captured entry limit did not name its counts or was admitted");
    unchangedEntries.unchanged(e.journal);
    check(e.journal.submitCaptured(e.initial,edit(e,9),e.submitter()) != 0,"an update at the entry limit was refused");
    // Retained bytes that fit alone but not beside other pending work clear as it settles.
    Fixture probe;
    (void)probe.journal.submitCaptured(probe.initial,edit(probe,6),probe.submitter());
    const auto charged = probe.journal.snapshot()->pendingBytes;
    // The quota also holds 32 charges of the scene, so several intents fit before the rest of it fills.
    SceneJournal::Limits shared; shared.maxPendingBytes = charged + charged / 2;
    Fixture r(shared,1024);
    check(r.journal.submitCaptured(r.initial,edit(r,7),r.submitter()) != 0,"first captured intent exceeded its own quota");
    transient = false; size_t admitted = 1;
    for (int value = 8; value < 1024 && !transient; ++value) {
        try { (void)r.journal.submitCaptured(r.initial,edit(r,value),r.submitter()); ++admitted; }
        catch (OmniUsdLiveResult result) { transient = result == OMNI_USD_LIVE_RESOURCE_EXHAUSTED; }
        catch (...) { break; }
    }
    check(transient && r.submits == admitted,"pending retained-byte quota was not a transient rejection");
}
void capturedFlushOrdering() {
    context = "captured admission checks emptiness, supersession and lone limits first";
    const auto set = [](const WireScenePtr& base, std::initializer_list<std::pair<const char*,int>> values) {
        auto log = recordingBacking(base);
        for (const auto& [path,value] : values) {
            const SdfPath target(path);
            log->Set(target.GetPrimPath(),TfToken(target.IsPropertyPath() ? target.GetName() : "value"),VtValue(value));
        }
        return recordedLog(*log);
    };
    // An empty envelope admits nothing even while the pending-intent limit is full.
    SceneJournal::Limits one; one.maxPending = 1;
    Fixture f(one);
    check(f.journal.submitCaptured(f.initial,set(f.initial,{{"/P",2}}),f.submitter()) != 0,"first captured intent was rejected");
    const Saved full(f.journal);
    auto fresh = recordingBacking(f.initial);
    fresh->CreateSpec(SdfPath("/Temp"),SdfSpecTypePrim); fresh->EraseSpec(SdfPath("/Temp"));
    WireScenePtr same;
    try { check(f.journal.submitCaptured(f.initial,recordedLog(*fresh),f.submitter(),&same) == 0,"empty captured envelope was admitted"); }
    catch (OmniUsdLiveResult) { check(false,"empty captured envelope reported the pending-intent limit"); }
    check(same && inspect(same) == inspect(f.initial) && f.submits == 1,"empty captured envelope did not rebaseline alone");
    full.unchanged(f.journal);
    // Every record addresses a section a committed peer delete removed: the
    // service would store it as invisible orphan data, so nothing is issued.
    Fixture g;
    const auto removal = deleteParent(); g.applyPeer(g.peer(removal,11),removal);
    const Saved deleted(g.journal);
    WireScenePtr applied;
    check(g.journal.submitCaptured(g.initial,set(g.initial,{{"/P",9}}),g.submitter(),&applied) == 0 && !g.submits,
          "superseded captured intent was issued");
    check(applied && number(applied) == 9,"superseded captured intent did not report its verified candidate");
    deleted.unchanged(g.journal);
    // A captured intent with any live target is issued whole.
    auto mixed = recordingBacking(g.initial);
    mixed->Set(SdfPath("/P"),TfToken("value"),VtValue(10));
    mixed->Set(SdfPath::AbsoluteRootPath(),TfToken("stamp"),VtValue(11));
    check(g.journal.submitCaptured(g.initial,recordedLog(*mixed),g.submitter()) != 0 && g.submits == 1,
          "partly superseded captured intent was not issued");
    check(!g.journal.snapshot()->visible->data()->Has(SdfPath("/P"),TfToken("value")) &&
          g.journal.snapshot()->visible->data()->Get(SdfPath::AbsoluteRootPath(),TfToken("stamp")) == VtValue(11),
          "partly superseded captured intent replayed its superseded record or lost its live one");
    // Replay work that a lone captured intent exceeds never clears. Scan the
    // replay-work limits between journal construction and admission. The
    // quota also holds 32 charges of the scene, so the intent's own value
    // carries work beyond them.
    enum class Outcome { Unbuilt, Admitted, Definitive, Plain };
    const auto heavy = [](const WireScenePtr& base) {
        auto log = recordingBacking(base);
        log->Set(SdfPath("/P"),TfToken("value"),VtValue(std::string(64 * 1024,'x')));
        return recordedLog(*log);
    };
    const auto outcome = [&](size_t work) {
        SceneJournal::Limits limits; limits.maxReplayWork = work;
        std::unique_ptr<Fixture> h;
        try { h = std::make_unique<Fixture>(limits); } catch (...) { return Outcome::Unbuilt; }
        try { (void)h->journal.submitCaptured(h->initial,heavy(h->initial),h->submitter()); }
        catch (const CapturedRejection& rejection) {
            check(rejection.result == OMNI_USD_LIVE_RESOURCE_EXHAUSTED && !h->submits,"wrong lone work rejection");
            return Outcome::Definitive;
        }
        catch (OmniUsdLiveResult) { return Outcome::Plain; }
        catch (const DecodeError&) { return Outcome::Plain; }
        return Outcome::Admitted;
    };
    size_t admitted = 1;
    while (outcome(admitted) != Outcome::Admitted) { check(admitted < (size_t(1) << 40),"no replay work admits"); admitted *= 2; }
    size_t low = admitted / 2, high = admitted; // Smallest admitting limit lies in (low, high].
    while (high - low > 1) { const auto middle = low + (high - low) / 2; (outcome(middle) == Outcome::Admitted ? high : low) = middle; }
    size_t definitive = 0;
    for (size_t work = high - 1; work > 0 && high - work <= 4096; --work) {
        const auto observed = outcome(work);
        if (observed == Outcome::Unbuilt) break;
        check(observed != Outcome::Plain,"lone captured intent over the replay work limit stayed transient");
        definitive += observed == Outcome::Definitive;
    }
    check(definitive > 0,"no replay work limit rejected a lone captured intent");
    // A captured Reorder whose listed child a peer removed after the host
    // captured the new order is local content naming a non-child, which is
    // malformed (INVALID_ARGUMENT) and definitive.
    EnvelopeInput setup; Command x{}, y{};
    x.kind = y.kind = Command::Kind::Create; x.parent = y.parent = 2; x.sectionType = y.sectionType = SdfSpecTypePrim;
    x.section = 3; x.name = "X"; y.section = 4; y.name = "Y"; setup.commands = {x,y};
    Fixture r({},128,buildSceneCandidate(initialScene(),encodeEnvelope(setup)));
    auto order = recordingBacking(r.initial);
    order->Set(SdfPath("/P"),SdfChildrenKeys->PrimChildren,VtValue(std::vector<TfToken>{TfToken("Y"),TfToken("X")}));
    const auto reorder = recordedLog(*order);
    check(!decodeEnvelope(compileFieldEdits(r.initial,reorder,{},{})->bytes()).commands.empty(),
          "captured child order compiled without a Reorder");
    EnvelopeInput peer; Command removeX{}; removeX.kind = Command::Kind::Delete; removeX.section = 3; removeX.parent = 2;
    peer.commands = {removeX};
    const auto peerBytes = encodeEnvelope(peer); r.applyPeer(r.peer(peerBytes,11),peerBytes);
    const Saved stale(r.journal);
    bool rejected = false;
    try { (void)r.journal.submitCaptured(r.initial,reorder,r.submitter()); }
    catch (const DecodeError& error) { rejected = error.kind == ErrorKind::Malformed; }
    check(rejected && !r.submits,"stale captured Reorder was not a malformed (INVALID_ARGUMENT) rejection");
    stale.unchanged(r.journal);
}
void capturedShapesAndEmptyFlushes() {
    context = "captured shapes are admitted and empty flushes issue nothing";
    const auto tokens = [](std::initializer_list<const char*> names) {
        std::vector<TfToken> result;
        for (const auto* name : names) result.emplace_back(name);
        return VtValue(result);
    };
    const auto create = [](uint64_t id,uint64_t parent,const char* name) {
        Command result{}; result.kind=Command::Kind::Create; result.section=id;
        result.parent=parent; result.name=name; result.sectionType=SdfSpecTypePrim; return result;
    };
    Fixture f;
    // Editing a pre-existing section and then deleting it compiles to the Delete alone.
    auto removed = recordingBacking(f.initial);
    removed->Set(SdfPath("/P"),TfToken("value"),VtValue(9));
    removed->EraseSpec(SdfPath("/P"));
    removed->Erase(SdfPath::AbsoluteRootPath(),SdfChildrenKeys->PrimChildren);
    WireScenePtr applied;
    const auto id = f.journal.submitCaptured(f.initial,recordedLog(*removed),f.submitter(),&applied);
    check(id != 0 && f.submits == 1 && applied && applied->sectionForPath(SdfPath("/P")) == 0,"captured delete-after-edit was vetoed");
    const auto envelope = decodeEnvelope(f.journal.snapshot()->pending.back().intent->bytes);
    check(envelope.commands.size() == 1 && envelope.commands[0].kind == Command::Kind::Delete && envelope.fields.empty(),
          "records inside a deleted subtree were emitted");
    // A buffer whose compiled envelope is empty admits nothing.
    Fixture g; const Saved before(g.journal);
    auto fresh = recordingBacking(g.initial);
    fresh->CreateSpec(SdfPath("/Temp"),SdfSpecTypePrim); fresh->EraseSpec(SdfPath("/Temp"));
    WireScenePtr same;
    check(g.journal.submitCaptured(g.initial,recordedLog(*fresh),g.submitter(),&same) == 0 && g.submits == 0 && same &&
          inspect(same) == inspect(g.initial),"empty captured envelope was admitted");
    before.unchanged(g.journal);
    check(!g.progress.snapshot().localAdmitted && !g.progress.snapshot().retained,"empty captured envelope reached the actor");
    // Move-cycle scratch Creates and child-membership overrides stay unsupported for captured logs.
    for (bool scratch : {false,true}) {
        EnvelopeInput extra; extra.commands = {create(4,1,"Q")};
        Fixture h({},128,scratch ? buildSceneCandidate(initialScene(),encodeEnvelope(extra)) : initialScene());
        const Saved unchanged(h.journal);
        auto log = recordingBacking(h.initial);
        if (scratch) {
            log->MoveSpec(SdfPath("/P"),SdfPath("/Temporary"));
            log->MoveSpec(SdfPath("/Q"),SdfPath("/P"));
            log->MoveSpec(SdfPath("/Temporary"),SdfPath("/Q"));
            log->Set(SdfPath::AbsoluteRootPath(),SdfChildrenKeys->PrimChildren,tokens({"Q","P"}));
        } else {
            log->EraseSpec(SdfPath("/P"));
            log->Set(SdfPath::AbsoluteRootPath(),SdfChildrenKeys->PrimChildren,tokens({}));
        }
        const auto sealed = recordedLog(*log);
        if (scratch) {
            const auto cycle = compileFieldEdits(h.initial,sealed,{},{});
            const auto swapped = buildSceneCandidate(h.initial,cycle->bytes());
            bool generated = false;
            for (const auto& command : decodeEnvelope(cycle->bytes()).commands)
                generated |= command.kind == Command::Kind::Create && swapped->pathForSection(command.section).IsEmpty();
            check(generated,"swap fixture compiled without a scratch Create");
        }
        unsupportedShape(scratch ? UnsupportedOperation::Reason::MoveCycle : UnsupportedOperation::Reason::MultiUpdate,
                         [&] { (void)h.journal.submitCaptured(h.initial,sealed,h.submitter()); });
        unchanged.unchanged(h.journal);
        check(!h.submits,"unsupported captured shape was submitted");
    }
}
void authoredPublication() {
    context = "authored ID publication before submitter completion";
    for (bool throws : {false,true}) {
        Fixture f;
        const auto base = f.journal.snapshot()->visible;
        const auto backing = recordingBacking(base);
        backing->Set(SdfPath("/P"),TfToken("value"),VtValue(7));
        const auto original = recordedLog(*backing);
        const auto real = f.submitter();
        bool entered = false;
        const auto id = f.journal.submitCaptured(base,original,[&](auto bytes,auto owner,auto publication)->uint64_t {
            entered = true;
            // The pending.front().id checks below prove the journal published
            // the committed ID despite this submitter's wrong return or throw.
            const auto accepted = real(bytes,std::move(owner),publication);
            if (throws) throw std::runtime_error("authored submitter failed after publication");
            return accepted+1;
        });
        check(entered,"captured submitter was not invoked");
        const auto state = f.journal.snapshot();
        check(id == 1 && f.submits == 1 && state->pending.size() == 1 &&
              state->pending.front().id == id && state->pending.front().intent->authored &&
              state->pendingBytes == state->pending.front().intent->retainedBytes && number(state->visible) == 7 &&
              state->submissionFault == OMNI_USD_LIVE_INTERNAL_ERROR && f.progress.retained().size() == 1,
              "postpublication authoring fault lost accepted ID/state/owner");
        const Saved accepted(f.journal);
        backing->Set(SdfPath("/P"),TfToken("value"),VtValue(99));
        check(state->pending.front().intent->authored->operations() == original->operations &&
              original->data->Get(SdfPath("/P"),TfToken("value")) == VtValue(7),
              "later host edit after submitter fault changed the accepted log");
        accepted.unchanged(f.journal);
        auto work = f.ack(id,11);
        result(OMNI_USD_LIVE_INTERNAL_ERROR,[&] { f.applyLocal(work); });
        accepted.unchanged(f.journal);
    }
}
void authoredSharedWork() {
    context = "shared compilation and journal replay work";
    // Enough pending work that replay exceeds the journal's room for the scene.
    constexpr unsigned backlog = 256;
    const auto source = initialScene();
    const auto set42 = [](SdfAbstractData& data) { data.Set(SdfPath("/P"),TfToken("value"),VtValue(42)); };
    Fixture measured({},1024,source);
    for (unsigned i = 0; i < backlog; ++i) measured.local(edit(int(i+1)));
    const auto prior = measured.journal.snapshot();
    const auto compiled = compileFieldEdits(prior->visible,capturedLog(prior->visible,set42));
    const size_t generous = SceneJournal::Limits{}.maxReplayWork;
    // Admission replays every pending intent and the new one in one build.
    auto replayAll = [&](const SceneJournal::SnapshotPtr& state,const std::string& extra,WireSceneWorkBudget& work) {
        std::vector<std::string_view> intents;
        for (const auto& pending : state->pending) intents.push_back(pending.intent->bytes);
        intents.push_back(extra);
        return buildReplayedScene(state->authoritative,intents,{},&work,true);
    };
    WireSceneWorkBudget replay{generous};
    const auto standalone = replayAll(prior,compiled->bytes(),replay);
    const size_t replayUsed = generous-replay.remaining;
    check(replayUsed > 0 && compiled->workUsed() > 0 && compiled->workUsed() <= generous-replayUsed,
          "invalid independent compile/replay work measurements");
    check(number(standalone) == 42,"independent replay baseline failed");
    // The existing backlog gives the compiler ample conservative preflight
    // headroom. Each phase fits alone, but their sum exceeds this limit by one.
    // The journal adds room for the scene to the configured limit.
    const size_t headroom = measured.journal.effectiveLimits(*prior->visible).maxReplayWork - generous;
    check(compiled->workUsed()+replayUsed-1 > headroom,"fixture work does not exceed the scene headroom");
    SceneJournal::Limits limits;
    limits.maxReplayWork = compiled->workUsed()+replayUsed-1-headroom;
    Fixture bounded(limits,1024,source);
    for (unsigned i = 0; i < backlog; ++i) bounded.local(edit(int(i+1)));
    const auto boundedBase = bounded.journal.snapshot()->visible;
    const auto log = capturedLog(boundedBase,set42);
    const size_t effective = limits.maxReplayWork + headroom;
    auto compilerLimits = limits.authoring;
    compilerLimits.scene = limits.scene; compilerLimits.maxWork = effective;
    const auto alone = compileFieldEdits(boundedBase,log,compilerLimits);
    WireSceneWorkBudget replayAlone{effective};
    (void)replayAll(bounded.journal.snapshot(),alone->bytes(),replayAlone);
    check(alone->workUsed() == compiled->workUsed() && effective-replayAlone.remaining == replayUsed &&
          alone->workUsed() > effective-replayUsed,
          "fixture does not isolate combined compile/replay exhaustion");
    const Saved before(bounded.journal);
    const auto submittedBefore = bounded.submits;
    const auto retainedBefore = bounded.progress.retained().size();
    bool rejected = false;
    // With pending work beside it the intent is not lone, so replay reports
    // exhaustion as a plain result or DecodeError, never CapturedRejection.
    try { bounded.journal.submitCaptured(boundedBase,log,bounded.submitter()); }
    catch (const DecodeError& error) { rejected = error.kind == ErrorKind::ResourceLimit; }
    catch (OmniUsdLiveResult error) { rejected = error == OMNI_USD_LIVE_RESOURCE_EXHAUSTED; }
    check(rejected && bounded.submits == submittedBefore &&
          bounded.progress.retained().size() == retainedBefore,
          "journal reset work budget between authored compilation and replay");
    before.unchanged(bounded.journal);
}
void admissionAndIds() {
    context = "journal admission atomicity and IDs";
    Fixture f;
    const Saved before(f.journal); const auto bytes = edit(1);
    bool allocation = false;
    try { f.journal.submitLocal(bytes,[](auto,auto,auto)->uint64_t { throw std::bad_alloc(); }); }
    catch (const std::bad_alloc&) { allocation = true; }
    check(allocation,"prepublication exception not propagated"); before.unchanged(f.journal);
    bool malformed = false;
    try { f.local("invalid envelope"); } catch (const DecodeError&) { malformed = true; }
    check(malformed && f.submits == 0 && f.progress.retained().empty(),"invalid local preparation reached admission");
    before.unchanged(f.journal);
    const auto id = f.local(bytes); check(id == 1,"rejected preparation consumed an ID");
    const auto current = f.journal.snapshot();
    check(current->applied == 10 && number(current->authoritative) == 0 && number(current->visible) == 1 &&
        current->pending.size() == 1 && current->pending[0].id == id && current->pending[0].intent->bytes == bytes &&
        current->pendingBytes == current->pending[0].intent->retainedBytes && current->pendingBytes==bytes.size(),
        "visible state and complete retained admission were not published together");
    check(Fixture::intent(f.retained(id)) == current->pending[0].intent,"actor/progress retained a different intent owner");

    Fixture full({},1); const auto first = full.local(edit(1)); const Saved accepted(full.journal);
    result(OMNI_USD_LIVE_RESOURCE_EXHAUSTED,[&] { full.local(edit(2)); }); accepted.unchanged(full.journal);
    auto acknowledged = full.ack(first,11); full.applyLocal(acknowledged);
    check(full.local(edit(2)) == 2,"failed progress admission consumed the next ID");
}
void acknowledgementsAndABA() {
    context = "ACK retention and A-B-A operation identity";
    Fixture f; const auto toB = edit(1), toA = edit(0);
    const auto first = f.local(toB), second = f.local(toA);
    auto admitted = f.journal.snapshot();
    check(first == 1 && second == 2 && admitted->pending.size() == 2 && number(admitted->visible) == 0 &&
        admitted->pending[0].intent != admitted->pending[1].intent && admitted->pending[0].intent->bytes == toB &&
        admitted->pending[1].intent->bytes == toA,"A-B-A edits collapsed by value equality");
    std::weak_ptr<const SceneJournal::Intent> firstOwner = admitted->pending[0].intent;
    auto firstWork = f.ack(first,11);
    check(f.journal.snapshot() == admitted && !firstOwner.expired(),"ACK alone retired or applied local work");
    // Exact pointer AND shared ownership must match. A different control block
    // with the same raw address is not the admitted retained owner.
    const auto owner = Fixture::intent(firstWork);
    const Saved before(f.journal);
    auto fake = std::shared_ptr<const void>(owner.get(),[](const void*) {});
    result(OMNI_USD_LIVE_INTERNAL_ERROR,[&] { f.journal.apply(firstWork,toB,fake); }); before.unchanged(f.journal);
    result(OMNI_USD_LIVE_INTERNAL_ERROR,[&] { f.journal.apply(firstWork,toA,owner); }); before.unchanged(f.journal);
    auto wrong = firstWork; wrong.id = second;
    result(OMNI_USD_LIVE_INTERNAL_ERROR,[&] { f.journal.apply(wrong,toB,owner); }); before.unchanged(f.journal);
    auto gap = firstWork; gap.first = gap.last = 12;
    result(OMNI_USD_LIVE_INTERNAL_ERROR,[&] { f.journal.apply(gap,toB,owner); }); before.unchanged(f.journal);
    f.applyLocal(firstWork);
    const auto intermediate = f.journal.snapshot();
    check(intermediate->applied == 11 && number(intermediate->authoritative) == 1 && number(intermediate->visible) == 0 &&
        intermediate->pending.size() == 1 && intermediate->pending[0].id == second &&
        intermediate->pending[0].intent == admitted->pending[1].intent && intermediate->pendingBytes == admitted->pending[1].intent->retainedBytes,
        "ordered local application retired wrong intent or failed to replay newer edit");
    auto secondWork = f.ack(second,12); f.applyLocal(secondWork);
    const auto final = f.journal.snapshot();
    check(final->applied == 12 && number(final->authoritative) == 0 && number(final->visible) == 0 &&
        final->pending.empty() && final->pendingBytes == 0 && f.progress.snapshot().retained == 0,"final local intent did not retire");
    check(admitted->pending.size() == 2 && number(admitted->authoritative) == 0 && !firstOwner.expired(),"retained snapshot lost historical intent");
}
void peerReplayAndFailure() {
    context = "peer replay and atomic replay failure";
    Fixture f; const auto bytes = edit(1); const auto id = f.local(bytes);
    const auto admitted = f.journal.snapshot(); auto peer = f.peer(bytes,11); f.applyPeer(peer,bytes);
    const auto replayed = f.journal.snapshot();
    check(replayed->applied == 11 && number(replayed->authoritative) == 1 && number(replayed->visible) == 1 &&
        replayed->pending.size() == 1 && replayed->pending[0].id == id && replayed->pending[0].intent == admitted->pending[0].intent,
        "peer operation/value equality retired local intent");
    auto own = f.ack(id,12); f.applyLocal(own);
    check(f.journal.snapshot()->pending.empty(),"confirmed local replay did not retire intent");

    // A pending create whose parent a committed peer delete removed is superseded:
    // replay drops it from the visible scene, and its own acknowledgement retires it.
    Fixture superseded; const auto local = createChild(); const auto pending = superseded.local(local);
    const auto deletion = deleteParent(); auto remote = superseded.peer(deletion,11); superseded.applyPeer(remote,deletion);
    const auto dropped = superseded.journal.snapshot();
    check(dropped->visible->sectionForPath(SdfPath("/P/Child")) == 0 && dropped->visible->sectionForPath(SdfPath("/P")) == 0 &&
          dropped->pending.size() == 1 && dropped->pending[0].id == pending,"superseded pending edit stayed visible or was dropped");
    auto acknowledged = superseded.ack(pending,12); superseded.applyLocal(acknowledged);
    const auto retired = superseded.journal.snapshot();
    check(retired->pending.empty() && retired->applied == 12 && retired->authoritative->sectionForPath(SdfPath("/P/Child")) == 0,
          "superseded edit did not retire on its own acknowledgement");
    Fixture malformedPeer; const Saved before(malformedPeer.journal); auto bad = malformedPeer.peer("bad envelope",11);
    result(OMNI_USD_LIVE_MALFORMED_STREAM,[&] { malformedPeer.journal.apply(bad,"bad envelope"); }); before.unchanged(malformedPeer.journal);
}
void aggregateRetiresAbsorbedIntents() {
    context = "aggregate coverage retires the acknowledged intents it covers";
    Fixture f; const auto own = edit(1); const auto id = f.local(own);
    f.ack(id, 12);                                              // Hole at 11.
    f.progress.disconnect(); f.read = f.progress.startRead();  // Resumes from 10.
    const auto combined = edit(2);                              // The service's state at 12.
    auto owner = std::make_shared<const std::string>(combined);
    LayerProgress::ContentHash hash{}; computeSHA256(combined.data(), combined.size(), hash.data());
    const auto aggregateId = f.progress.acceptCoverage(f.read.generation, f.read.object, 11, 12,
        {owner, hash, LayerProgress::CoverageKind::Aggregate});
    const auto work = f.retained(aggregateId);
    check(work.absorbedLocal == 1 && work.absorbedThrough == 1, "aggregate did not absorb the acknowledgment");
    auto forged = work; forged.absorbedThrough = 2;
    const Saved before(f.journal);
    result(OMNI_USD_LIVE_INTERNAL_ERROR, [&] { f.journal.apply(forged, combined); }); before.unchanged(f.journal);
    f.applyPeer(work, combined);
    const auto state = f.journal.snapshot();
    check(state->applied == 12 && state->pending.empty() && state->pendingBytes == 0 && state->localRetired == 1 &&
          number(state->authoritative) == 2 && number(state->visible) == 2, "absorbed intent did not retire with the aggregate");
    const auto progress = f.progress.snapshot();
    check(progress.retained == 0 && progress.localRetired == 1 && progress.pendingWrites == 0,
          "progress kept absorbed work");
}
// A write the chain proves committed has no reported position: a read started after the proof delivers its commit
// as an ordinary update, and its intent stays pending, its value visible, until the next local write's application
// retires it first. An acknowledged write or retained peer update a net-empty catch-up passed retires without
// applying its bytes.
void committedAndNettedRetirement() {
    context = "chain-proven and net-empty retirement";
    Fixture f; const auto a = f.local(edit(1)); const auto b = f.local(edit(2,"other"));
    check(f.progress.takeWrite(2).id == a && f.progress.takeWrite(2).id == b,"writes did not issue");
    check(f.progress.requeueWrite(a,true),"write was not requeued");
    f.progress.completeWrite(b,OMNI_USD_LIVE_OK,14);
    f.progress.confirmCommitted(a);
    f.progress.disconnect(); f.read = f.progress.startRead(); // Resumes from 10.
    auto peer = f.peer(edit(5,"peer"),11); f.applyPeer(peer,edit(5,"peer"));
    auto own = f.peer(edit(1),12); f.applyPeer(own,edit(1)); // A's commit, delivered as an ordinary update.
    auto undo = f.peer(edit(7),13); f.applyPeer(undo,edit(7));
    const auto overlaid = f.journal.snapshot();
    check(number(overlaid->authoritative) == 7 && number(overlaid->visible) == 1 && overlaid->pending.size() == 2,
          "a chain-proven write's pending value did not stay visible until its proving write applies");
    const auto carrier = f.retained(b);
    check(carrier.absorbedLocal == 1 && carrier.absorbedThrough == 1 && f.retained(a).absorbedBy == b,
          "the chain-proven write does not retire with its proving write");
    auto alone = carrier; alone.absorbedLocal = 0; alone.absorbedThrough = 0;
    const auto owner = Fixture::intent(carrier);
    const Saved before(f.journal);
    result(OMNI_USD_LIVE_INTERNAL_ERROR,[&] { f.journal.apply(alone,owner->bytes,owner); }); before.unchanged(f.journal);
    f.applyLocal(carrier);
    const auto retired = f.journal.snapshot();
    check(retired->applied == 14 && retired->pending.empty() && retired->localRetired == 2 &&
          number(retired->authoritative) == 7 && number(retired->visible) == 7 && number(retired->visible,"other") == 2,
          "the proving write did not retire the chain-proven intent before its own");
    check(f.progress.snapshot().retained == 0 && f.progress.snapshot().localRetired == 2,"progress kept retired work");

    Fixture g; const auto c = g.local(edit(3));
    g.ack(c,12); // Hole at 11.
    g.progress.disconnect(); g.read = g.progress.startRead();
    check(g.progress.caughtUp(g.read.generation,g.read.object,12).accepted,"net-empty catch-up was not accepted");
    const auto netted = g.retained(c);
    check(netted.netted && netted.first == 12,"the net-empty catch-up did not pass the acknowledged write");
    auto forged = g.peer(edit(4,"peer"),13); forged.netted = true; forged.localOrdinal = 1;
    const Saved unchanged(g.journal);
    result(OMNI_USD_LIVE_INTERNAL_ERROR,[&] { g.journal.apply(forged,edit(4,"peer")); }); unchanged.unchanged(g.journal);
    forged.localOrdinal = 0;
    result(OMNI_USD_LIVE_INTERNAL_ERROR,[&] { g.journal.apply(forged,edit(4,"peer"),Fixture::intent(netted)); });
    unchanged.unchanged(g.journal);
    g.journal.applyEmpty(11); g.applyLocal(netted);
    const auto quiet = g.journal.snapshot();
    check(quiet->applied == 12 && quiet->pending.empty() && quiet->localRetired == 1 &&
          number(quiet->authoritative) == number(g.initial) && number(quiet->visible) == number(g.initial),
          "a write a net-empty catch-up passed applied its bytes");

    // With the read open, A commits at 11, unseen, and peers' 12 and 13 wait behind A's position; B is acknowledged
    // at 14 and the chain proves A. The resubscribed read's catch-up nets 11..14 to nothing, so the retained peer
    // updates retire without applying, as B does, and A retires with B.
    Fixture h; const auto ha = h.local(edit(3)); const auto hb = h.local(edit(4,"other"));
    check(h.progress.takeWrite(2).id == ha && h.progress.takeWrite(2).id == hb,"writes did not issue");
    auto set = h.peer(edit(5),12); auto create = h.peer(createChild(),13);
    h.progress.completeWrite(hb,OMNI_USD_LIVE_OK,14);
    check(h.progress.requeueWrite(ha,true),"write was not requeued");
    h.progress.confirmCommitted(ha);
    h.progress.disconnect(); h.read = h.progress.startRead();
    check(h.read.sequence == 10,"the resubscribed read did not resume below the retained peer coverage");
    check(h.progress.caughtUp(h.read.generation,h.read.object,14).accepted,"net-empty catch-up was not accepted");
    set = h.retained(set.id); create = h.retained(create.id); const auto proving = h.retained(hb);
    check(set.netted && create.netted && proving.netted && proving.absorbedLocal == 1 && h.retained(ha).absorbedBy == hb,
          "the net-empty catch-up did not pass the retained peer coverage and the proving write");
    h.journal.applyEmpty(11); h.applyPeer(set,edit(5));
    const auto passed = h.journal.snapshot();
    check(passed->applied == 12 && inspect(passed->authoritative) == inspect(h.initial) &&
          passed->pending.size() == 2,"a retained peer update a net-empty catch-up passed applied its bytes");
    h.applyPeer(create,createChild()); h.applyLocal(proving);
    const auto settled = h.journal.snapshot();
    check(settled->applied == 14 && settled->pending.empty() && settled->localRetired == 2 &&
          inspect(settled->authoritative) == inspect(h.initial) && inspect(settled->visible) == inspect(h.initial),
          "the net-empty catch-up over retained peer coverage did not leave the service's state");
    check(h.progress.snapshot().retained == 0 && h.progress.snapshot().localRetired == 2,"progress kept netted work");
}
// Consecutive ordered work applied as one transaction equals applying each work alone: the empty commits between
// them advance the frontier, absorbed and acknowledged intents retire in order, the pending intents replay once
// onto the last authoritative scene and one snapshot publishes. A failing work publishes nothing.
// Replaying pending intents in one build equals replaying each in turn, over seeded sequences that place a
// predicted rejection (a Create of a live identity) or a Reorder naming a non-child, which the replay skips, at
// every position, beside field edits, fresh Creates, Moves, Deletes and valid Reorders; with a strict tail, the
// last intent is an ordinary update and a Create of a live identity or a Reorder naming a non-child fails both ways.
void onePassReplay(uint64_t seed) {
    context = "one-pass replay of pending intents";
    const auto command = [](Command::Kind kind, uint64_t section, uint64_t parent, const std::string& name = "") {
        Command c{}; c.kind = kind; c.section = section; c.parent = parent; c.name = name;
        if (kind == Command::Kind::Create) c.sectionType = SdfSpecTypePrim;
        return c;
    };
    const auto structural = [](std::vector<Command> commands) {
        EnvelopeInput m; m.commands = std::move(commands); return encodeEnvelope(m);
    };
    EnvelopeInput children; children.commands = {command(Command::Kind::Create,3,2,"A"),command(Command::Kind::Create,4,2,"B")};
    const auto base = buildSceneCandidate(initialScene(),encodeEnvelope(children));
    std::mt19937_64 random(seed ^ 0x7265706c6179ull);
    uint64_t fresh = 100;
    const auto valid = [&]() -> std::string {
        switch (random() % 6) {
        case 0: return edit(int(random() % 100),"replayed");
        case 1: { const auto id = fresh++; return structural({command(Command::Kind::Create,id,2,"N" + std::to_string(id))}); }
        case 2: { Command reorder = command(Command::Kind::Reorder,2,0); reorder.children = {4,3}; return structural({reorder}); }
        case 3: return structural({command(Command::Kind::Delete,4,2)});
        case 4: { auto move = command(Command::Kind::Move,3,4,"Moved"); move.oldParent = 2; return structural({move}); }
        default: return edit(int(random() % 100),"value");
        }
    };
    const auto rejected = [&]() -> std::string {
        if (random() % 2) return structural({command(Command::Kind::Create,3,2,"Again")});
        Command reorder = command(Command::Kind::Reorder,2,0); reorder.children = {3,999}; return structural({reorder});
    };
    const auto describe = [](const WireScenePtr& scene) {
        // The graph, hidden rows included, and the last intent's field changes; inspect compares the Sdf data.
        std::ostringstream text;
        scene->visitSections(1u << 20,[&](const WireSectionView& view) {
            text << '|' << view.id << ',' << view.parent << ',' << view.name << ',' << view.order << ',' << view.path;
            return true;
        });
        for (const auto& change : scene->fieldChanges()) text << '#' << change.section << ',' << change.record << ',' << change.field;
        return text.str();
    };
    size_t compared = 0, rejections = 0;
    for (int round = 0; round < 48; ++round) {
        const size_t count = 1 + random() % 8;
        for (size_t at = 0; at < count; ++at) {
            std::vector<std::string> intents;
            for (size_t i = 0; i < count; ++i) intents.push_back(i == at ? rejected() : valid());
            std::vector<std::string_view> views(intents.begin(),intents.end());
            for (const bool strict : {false,true}) {
                WireScenePtr sequential = base; bool sequentialFailed = false;
                try {
                    for (size_t i = 0; i < intents.size(); ++i) {
                        const auto next = buildSceneCandidate(sequential,intents[i],{},nullptr,
                            strict && i + 1 == intents.size() ? WireSceneBatchPolicy::OrdinaryUpdate : WireSceneBatchPolicy::PendingReplay);
                        rejections += !strict && next == sequential;
                        sequential = next;
                    }
                } catch (const DecodeError&) { sequentialFailed = true; }
                WireScenePtr built; bool builtFailed = false;
                try { built = buildReplayedScene(base,views,{},nullptr,strict); } catch (const DecodeError&) { builtFailed = true; }
                check(sequentialFailed == builtFailed,"one-pass replay failed differently from replaying each intent");
                if (sequentialFailed) continue;
                check(inspect(built) == inspect(sequential) && describe(built) == describe(sequential),
                      "one-pass replay differs from replaying each intent");
                ++compared;
            }
        }
    }
    check(compared > 100 && rejections > 50,"seeded replay sequences exercised too few predicted rejections");
    // Field and metadata limits hold on every intermediate scene, as they do when each intent is replayed in turn:
    // a pending intent that exceeds a limit fails the replay even when a later intent erases back under it.
    const auto replayEachFails = [&](const std::vector<std::string>& intents, const WireSceneLimits& limits) {
        WireScenePtr scene = base;
        try {
            for (const auto& intent : intents)
                scene = buildSceneCandidate(scene,intent,limits,nullptr,WireSceneBatchPolicy::PendingReplay);
        } catch (const DecodeError& error) { return error.kind == ErrorKind::ResourceLimit; }
        return false;
    };
    const auto replayOnceFails = [&](const std::vector<std::string>& intents, const WireSceneLimits& limits) {
        std::vector<std::string_view> views(intents.begin(),intents.end());
        try { (void)buildReplayedScene(base,views,limits); }
        catch (const DecodeError& error) { return error.kind == ErrorKind::ResourceLimit; }
        return false;
    };
    size_t baseFields = 0, baseMetadata = base->sourceFormat().size();
    base->visitSections(1u << 20,[&](const WireSectionView& view) {
        baseMetadata += view.name.size();
        if (!view.path.IsEmpty())
            for (const auto& name : base->data()->List(view.path)) { ++baseFields; baseMetadata += name.GetString().size(); }
        return true;
    });
    const auto fieldsOn = [](const std::vector<std::pair<std::string,int>>& set) {
        EnvelopeInput m; DecodedValues v;
        for (const auto& [name, value] : set) {
            m.fields.push_back({2,uint64_t(m.fields.size()),name,{}});
            v.fields.push_back(value < 0 ? VtValue() : VtValue(value));
        }
        return encodeUsdEnvelope(m,v);
    };
    const std::vector<std::string> overThenBack{fieldsOn({{"extraA",1},{"extraB",2},{"extraC",3}}),
                                                fieldsOn({{"extraA",-1},{"extraB",-1},{"extraC",-1}})};
    WireSceneLimits fieldLimit; fieldLimit.maxFields = baseFields + 2;
    check(replayEachFails(overThenBack,fieldLimit),"per-intent replay did not refuse an intermediate field excess");
    check(replayOnceFails(overThenBack,fieldLimit),"one-pass replay accepted an intermediate field excess");
    fieldLimit.maxFields = baseFields + 3;
    check(!replayEachFails(overThenBack,fieldLimit) && !replayOnceFails(overThenBack,fieldLimit),
          "a replay within the field limit was refused");
    const std::string longName(64,'m');
    const std::vector<std::string> longThenBack{fieldsOn({{longName,1}}),fieldsOn({{longName,-1}})};
    WireSceneLimits metadataLimit; metadataLimit.maxMetadataBytes = baseMetadata + 63;
    check(replayEachFails(longThenBack,metadataLimit),"per-intent replay did not refuse an intermediate metadata excess");
    check(replayOnceFails(longThenBack,metadataLimit),"one-pass replay accepted an intermediate metadata excess");
    metadataLimit.maxMetadataBytes = baseMetadata + 64;
    check(!replayEachFails(longThenBack,metadataLimit) && !replayOnceFails(longThenBack,metadataLimit),
          "a replay within the metadata limit was refused");
    // No pending intent, or only rejected ones, leaves the base scene itself.
    check(buildReplayedScene(base,{}) == base,"an empty replay built a new scene");
    const auto again = structural({command(Command::Kind::Create,3,2,"Again")});
    check(buildReplayedScene(base,{again,again}) == base,"a fully rejected replay built a new scene");
}
void batchedApplication() {
    context = "batched application";
    const auto peerA = edit(1,"peerA"), peerB = edit(2,"peerB"), peerC = edit(3,"value"), own = edit(9,"own");
    const auto populate = [&](Fixture& f) {
        const auto pending = f.local(own);
        std::vector<LayerProgress::Work> works{f.peer(peerA,11),f.peer(peerB,13)}; // 12 is an empty commit.
        auto ack = f.ack(pending,14); works.push_back(ack);
        works.push_back(f.peer(peerC,15));
        return works;
    };
    const auto bytesOf = [&](const LayerProgress::Work& work) {
        return work.disposition == LayerProgress::Disposition::Peer ? *std::static_pointer_cast<const std::string>(work.owner) :
                                                                      Fixture::intent(work)->bytes;
    };
    Fixture alone; const auto aloneWork = populate(alone);
    for (const auto& work : aloneWork) {
        alone.journal.applyEmpty(work.first - 1);
        if (work.disposition == LayerProgress::Disposition::Peer) alone.journal.apply(work,bytesOf(work));
        else alone.journal.apply(work,bytesOf(work),Fixture::intent(work));
    }
    Fixture batched; const auto batchedWork = populate(batched);
    const auto before = batched.journal.snapshot();
    std::vector<std::string> payloads;
    for (const auto& work : batchedWork) payloads.push_back(bytesOf(work));
    std::vector<SceneJournal::Application> batch;
    for (size_t i = 0; i < batchedWork.size(); ++i) {
        const auto& work = batchedWork[i];
        batch.push_back({work,payloads[i],work.disposition == LayerProgress::Disposition::Peer ? nullptr : Fixture::intent(work)});
    }
    batched.journal.applyEmpty(10);
    const auto committed = batched.journal.apply(batch);
    const auto expected = alone.journal.snapshot();
    check(committed == batched.journal.snapshot() && committed->revision == before->revision + 1,
          "batched application did not publish one snapshot");
    check(committed->applied == 15 && expected->applied == 15 && committed->pending.empty() && committed->localRetired == 1 &&
          committed->pendingBytes == 0 && inspect(committed->authoritative) == inspect(expected->authoritative) &&
          inspect(committed->visible) == inspect(expected->visible) && number(committed->visible,"own") == 9,
          "batched application differs from applying each work alone");

    // A pending intent replays once onto the batch's last authoritative scene.
    Fixture replayed; replayed.local(edit(7,"own"));
    std::vector<LayerProgress::Work> peers{replayed.peer(peerA,11),replayed.peer(peerB,12)};
    const auto replayedBatch = replayed.journal.apply({{peers[0],peerA,nullptr},{peers[1],peerB,nullptr}});
    check(replayedBatch->pending.size() == 1 && number(replayedBatch->visible,"own") == 7 &&
          number(replayedBatch->visible,"peerB") == 2 && number(replayedBatch->authoritative,"peerA") == 1,
          "batched peers lost the pending intent's replay");

    // One build applies a batch whose later updates move, reorder and delete what earlier ones created,
    // reorder siblings through an explicit child field and write to a deleted section: the scenes, including the
    // last update's field changes, equal applying each update alone.
    context = "batched structural application";
    const auto command = [](Command::Kind kind, uint64_t section, uint64_t parent, const char* name = "") {
        Command c{}; c.kind = kind; c.section = section; c.parent = parent; c.name = name;
        if (kind == Command::Kind::Create) c.sectionType = SdfSpecTypePrim;
        return c;
    };
    std::vector<std::string> structural;
    {
        EnvelopeInput m; m.commands = {command(Command::Kind::Create,3,2,"A"),command(Command::Kind::Create,4,2,"B"),
                                       command(Command::Kind::Create,5,3,"Leaf")};
        m.fields = {{3,0,"value",{}},{5,1,"value",{}}};
        structural.push_back(encodeUsdEnvelope(m,DecodedValues{{VtValue(30),VtValue(50)},{}}));
    }
    {
        EnvelopeInput m; auto reorder = command(Command::Kind::Reorder,2,0); reorder.children = {4,3};
        m.commands = {reorder}; structural.push_back(encodeEnvelope(m));
    }
    {
        EnvelopeInput m; auto move = command(Command::Kind::Move,5,4,"Moved"); move.oldParent = 3;
        m.commands = {move,command(Command::Kind::Create,6,2,"C")};
        m.fields = {{2,0,"primChildren",{}}};
        structural.push_back(encodeUsdEnvelope(m,DecodedValues{{VtValue(std::vector<TfToken>{TfToken("C"),TfToken("A"),TfToken("B")})},{}}));
    }
    {
        EnvelopeInput m; m.commands = {command(Command::Kind::Create,7,2,"D"),command(Command::Kind::Delete,3,2)};
        m.fields = {{3,0,"value",{}},{5,1,"value",{}},{7,2,"value",{}}};
        structural.push_back(encodeUsdEnvelope(m,DecodedValues{{VtValue(31),VtValue(51),VtValue(70)},{}}));
    }
    Fixture each, once;
    each.local(edit(9,"own")); once.local(edit(9,"own"));
    std::vector<LayerProgress::Work> eachWork, onceWork;
    for (size_t i = 0; i < structural.size(); ++i) {
        eachWork.push_back(each.peer(structural[i],11 + i));
        onceWork.push_back(once.peer(structural[i],11 + i));
    }
    for (size_t i = 0; i < structural.size(); ++i) each.journal.apply(eachWork[i],structural[i]);
    std::vector<SceneJournal::Application> sequence;
    for (size_t i = 0; i < structural.size(); ++i) sequence.push_back({onceWork[i],structural[i],nullptr});
    const auto built = once.journal.apply(sequence);
    const auto separate = each.journal.snapshot();
    check(built->applied == 14 && separate->applied == 14,"structural batch did not reach its last position");
    check(inspect(built->authoritative) == inspect(separate->authoritative) &&
          inspect(built->visible) == inspect(separate->visible),"one structural build differs from building each update");
    check(built->authoritative->listedChildren(2) == separate->authoritative->listedChildren(2) &&
          built->authoritative->listedChildren(2) == std::vector<uint64_t>({6,4,7}) &&
          built->authoritative->sectionForPath(SdfPath("/P/B/Moved")) == 5 && !built->authoritative->containsSection(3),
          "one structural build lost a move, reorder or delete");
    const auto& builtChanges = built->authoritative->fieldChanges();
    const auto& separateChanges = separate->authoritative->fieldChanges();
    check(builtChanges.size() == separateChanges.size() && builtChanges.size() == 2,"field changes differ");
    for (size_t i = 0; i < builtChanges.size(); ++i)
        check(builtChanges[i].section == separateChanges[i].section && builtChanges[i].record == separateChanges[i].record &&
              builtChanges[i].field == separateChanges[i].field,"field change records differ");
    check(number(built->visible,"own") == 9 && built->pending.size() == 1,"structural batch lost the pending intent");

    // An aggregate diff first, in the middle and last of a batch, followed by structural ordinary updates: it
    // places a new section at a sibling index, restates a section at its current index and retracts a subtree.
    context = "batched diff application";
    const auto diffCommand = [](Command::Kind kind, uint64_t section, uint64_t parent, const char* name, uint32_t order) {
        Command c{}; c.kind = kind; c.section = section; c.parent = parent; c.name = name; c.sectionOrder = order;
        if (kind == Command::Kind::DiffChange) c.sectionType = SdfSpecTypePrim;
        return c;
    };
    std::string aggregate;
    {
        EnvelopeInput m; m.isDiff = true; m.baseVersion = 10;
        m.commands = {diffCommand(Command::Kind::DiffChange,8,2,"E",1),diffCommand(Command::Kind::DiffChange,4,2,"B",0),
                      diffCommand(Command::Kind::DiffDelete,3,0,"",0),diffCommand(Command::Kind::DiffDelete,5,0,"",0)};
        m.fields = {{8,0,"value",{}},{4,1,"value",{}}};
        aggregate = encodeUsdEnvelope(m,DecodedValues{{VtValue(80),VtValue(40)},{}});
    }
    std::vector<std::string> following;
    {
        EnvelopeInput m; m.commands = {command(Command::Kind::Create,9,2,"F"),command(Command::Kind::Create,10,8,"G")};
        following.push_back(encodeEnvelope(m));
    }
    {
        EnvelopeInput m; auto reorder = command(Command::Kind::Reorder,2,0); reorder.children = {9,4};
        auto move = command(Command::Kind::Move,10,9,"H"); move.oldParent = 8;
        m.commands = {reorder,move,command(Command::Kind::Delete,8,2)};
        m.fields = {{10,0,"value",{}}};
        following.push_back(encodeUsdEnvelope(m,DecodedValues{{VtValue(100)},{}}));
    }
    for (size_t at = 0; at <= 2; ++at) {
        std::vector<std::string> updates;
        if (at) updates.push_back(structural[0]);
        if (at == 2) updates.push_back(structural[1]);
        updates.push_back(aggregate);
        updates.insert(updates.end(),following.begin(),following.end());
        if (at == 2) updates.push_back(aggregate); // Last, after the structural updates.
        Fixture one, all;
        std::vector<LayerProgress::Work> oneWork, allWork;
        for (size_t i = 0; i < updates.size(); ++i) {
            oneWork.push_back(one.peer(updates[i],11 + i));
            allWork.push_back(all.peer(updates[i],11 + i));
        }
        for (size_t i = 0; i < updates.size(); ++i) one.journal.apply(oneWork[i],updates[i]);
        std::vector<SceneJournal::Application> diffBatch;
        for (size_t i = 0; i < updates.size(); ++i) diffBatch.push_back({allWork[i],updates[i],nullptr});
        const auto combined = all.journal.apply(diffBatch);
        const auto sequential = one.journal.snapshot();
        check(inspect(combined->authoritative) == inspect(sequential->authoritative) &&
              combined->authoritative->listedChildren(2) == sequential->authoritative->listedChildren(2) &&
              combined->authoritative->sectionCount() == sequential->authoritative->sectionCount(),
              "a batch holding a diff differs from building each update");
        check(combined->authoritative->sectionForPath(SdfPath("/P/F/H")) == 10 &&
              !combined->authoritative->containsSection(3) && !combined->authoritative->containsSection(5),
              "a batch holding a diff lost its placement or retraction");
    }

    // Out-of-order or failing work leaves the journal unchanged.
    Fixture ordered; std::vector<LayerProgress::Work> works{ordered.peer(peerA,11),ordered.peer(peerB,12)};
    const Saved unchanged(ordered.journal);
    result(OMNI_USD_LIVE_INTERNAL_ERROR,[&] { ordered.journal.apply({{works[1],peerB,nullptr},{works[0],peerA,nullptr}}); });
    unchanged.unchanged(ordered.journal);
    result(OMNI_USD_LIVE_INTERNAL_ERROR,[&] { ordered.journal.apply({{works[0],peerA,nullptr},{works[0],peerA,nullptr}}); });
    unchanged.unchanged(ordered.journal);
    result(OMNI_USD_LIVE_INTERNAL_ERROR,[&] { ordered.journal.apply(std::vector<SceneJournal::Application>{}); });
    unchanged.unchanged(ordered.journal);
    result(OMNI_USD_LIVE_MALFORMED_STREAM,[&] { ordered.journal.apply({{works[0],peerA,nullptr},{works[1],"bad envelope",nullptr}}); });
    unchanged.unchanged(ordered.journal);
}
void operationPreservation() {
    context = "sample/map and dependent-create operation preservation";
    EnvelopeInput attribute;
    Command c{}; c.kind = Command::Kind::Create; c.section = 3; c.parent = 2; c.name = "a"; c.sectionType = SdfSpecTypeAttribute;
    attribute.commands = {c}; attribute.fields = {{3,0,"timeSamples",{}}};
    const auto source = buildSceneCandidate(initialScene(),encodeUsdEnvelope(attribute,
        DecodedValues{{VtValue(SdfTimeSampleMap{{1,VtValue(1)}})},{}}));
    EnvelopeInput peerMetadata; peerMetadata.samples = {{3,3,{}}};
    const auto peerBytes = encodeUsdEnvelope(peerMetadata,DecodedValues{{},{VtValue(33)}});
    for (bool wholeMap : {false,true}) {
        Fixture f({},128,source);
        EnvelopeInput localMetadata; DecodedValues localValues;
        if (wholeMap) {
            localMetadata.fields = {{3,0,"timeSamples",{}}};
            localValues.fields = {VtValue(SdfTimeSampleMap{{2,VtValue(22)}})};
        } else { localMetadata.samples = {{3,2,{}}}; localValues.samples = {VtValue(22)}; }
        const auto bytes = encodeUsdEnvelope(localMetadata,localValues);
        const auto id = f.local(bytes); auto peer = f.peer(peerBytes,11); f.applyPeer(peer,peerBytes);
        const auto rebased = f.journal.snapshot();
        const SdfTimeSampleMap authoritative{{1,VtValue(1)},{3,VtValue(33)}};
        const SdfTimeSampleMap expected = wholeMap ? SdfTimeSampleMap{{2,VtValue(22)}} :
            SdfTimeSampleMap{{1,VtValue(1)},{2,VtValue(22)},{3,VtValue(33)}};
        check(rebased->authoritative->data()->Get(SdfPath("/P.a"),SdfFieldKeys->TimeSamples) == VtValue(authoritative) &&
            rebased->visible->data()->Get(SdfPath("/P.a"),SdfFieldKeys->TimeSamples) == VtValue(expected),
            "replay conflated SetTimeSample with whole-map replacement");
        check(rebased->pending.size() == 1 && rebased->pending[0].id == id && rebased->pending[0].intent->bytes == bytes,
            "sample/map replay rewrote original operation bytes");
        auto own = f.ack(id,12); f.applyLocal(own);
        check(f.journal.snapshot()->pending.empty() &&
            f.journal.snapshot()->authoritative->data()->Get(SdfPath("/P.a"),SdfFieldKeys->TimeSamples) == VtValue(expected),
            "confirmed sample/map operation changed meaning");
    }
    Fixture dependent;
    const auto create = dependent.local(createChild());
    EnvelopeInput editMetadata; editMetadata.fields = {{3,0,"value",{}}};
    const auto editBytes = encodeUsdEnvelope(editMetadata,DecodedValues{{VtValue(42)},{}});
    const auto editId = dependent.local(editBytes);
    const auto remoteBytes = edit(7,"peer"); auto peer = dependent.peer(remoteBytes,11); dependent.applyPeer(peer,remoteBytes);
    const auto replayed = dependent.journal.snapshot();
    check(replayed->authoritative->sectionForPath(SdfPath("/P/Child")) == 0 &&
        replayed->visible->sectionForPath(SdfPath("/P/Child")) == 3 &&
        replayed->visible->data()->Get(SdfPath("/P/Child"),TfToken("value")) == VtValue(42) &&
        replayed->pending.size() == 2 && replayed->pending[0].id == create && replayed->pending[1].id == editId,
        "dependent create/edit replay lost order, identity or value");
    auto first = dependent.ack(create,12); dependent.applyLocal(first);
    auto second = dependent.ack(editId,13); dependent.applyLocal(second);
    check(dependent.journal.snapshot()->pending.empty() && dependent.journal.snapshot()->applied == 13,
        "dependent create/edit did not retire in order");
}
void ownershipRetirement() {
    context = "exact intent lifetime and unconfirmed work rejection";
    Fixture f; const auto bytes = edit(1); const auto id = f.local(bytes);
    auto retained = f.journal.snapshot();
    std::weak_ptr<const SceneJournal::Intent> weak = retained->pending[0].intent;
    auto queued = f.retained(id); queued.first = queued.last = 11;
    const Saved before(f.journal);
    result(OMNI_USD_LIVE_INTERNAL_ERROR,[&] { f.journal.apply(queued,bytes,Fixture::intent(queued)); });
    auto unknown = queued; unknown.disposition = LayerProgress::Disposition::Unknown;
    result(OMNI_USD_LIVE_INTERNAL_ERROR,[&] { f.journal.apply(unknown,bytes,Fixture::intent(unknown)); }); before.unchanged(f.journal);
    auto confirmed = f.ack(id,11); f.applyLocal(confirmed);
    check(!weak.expired(),"intent died while retained snapshot still owns it");
    // Saved deliberately retains the original snapshot too; a separate scope
    // below tests expiry after every publication/progress/callback owner ends.
    std::weak_ptr<const SceneJournal::Intent> retired;
    {
        Fixture isolated; const auto accepted = isolated.local(bytes);
        retired = isolated.journal.snapshot()->pending[0].intent;
        auto work = isolated.ack(accepted,11); isolated.applyLocal(work);
        check(!retired.expired(),"effect owner did not retain completed intent");
        work.owner.reset();
        check(retired.expired(),"retired intent leaked after last progress/effect/snapshot owner");
    }
    Fixture uncertain;
    const auto uncertainId = uncertain.local(bytes); const Saved original(uncertain.journal);
    auto issued = uncertain.progress.takeWrite(); check(issued.id == uncertainId,"unknown-outcome fixture did not issue");
    issued.first = issued.last = 11; // Isolate disposition rejection from missing-coverage rejection.
    result(OMNI_USD_LIVE_INTERNAL_ERROR,[&] { uncertain.journal.apply(issued,bytes,Fixture::intent(issued)); });
    original.unchanged(uncertain.journal);
    uncertain.progress.completeWrite(uncertainId,OMNI_USD_LIVE_OUTCOME_UNKNOWN);
    auto unknownWork = uncertain.retained(uncertainId); unknownWork.first = unknownWork.last = 11;
    result(OMNI_USD_LIVE_INTERNAL_ERROR,[&] { uncertain.journal.apply(unknownWork,bytes,Fixture::intent(unknownWork)); });
    uncertain.progress.disconnect(); uncertain.progress.fail(OMNI_USD_LIVE_INTERNAL_ERROR);
    original.unchanged(uncertain.journal);
    result(OMNI_USD_LIVE_OUTCOME_UNKNOWN,[&] { uncertain.local(edit(2)); }); original.unchanged(uncertain.journal);
    check(uncertain.progress.snapshot().retained == 1 && uncertain.progress.snapshot().applied == 10 &&
        Fixture::intent(uncertain.retained(uncertainId)) == original.pointer->pending[0].intent,
        "uncertain progress outcome discarded or advanced retained intent");
}
// A journal validates a supplied initial scene against its own limits,
// retained values included.
void initialSceneLimits() {
    context = "initial scene limits";
    VtDictionary inner; inner["leaf"] = VtValue(1); VtDictionary outer; outer["inner"] = VtValue(inner);
    EnvelopeInput metadata; metadata.fields = {{2,0,"nested",{}},{2,0,"text",{}}};
    const auto valued = buildSceneCandidate(initialScene(),encodeUsdEnvelope(metadata,
        DecodedValues{{VtValue(outer),VtValue(std::string(64,'s'))},{}}));
    const auto before = inspect(valued);
    SceneJournal accepted(valued,10,{});
    check(accepted.snapshot()->authoritative == valued,"a journal replaced a supplied initial scene within its limits");
    const auto refused = [&](const char* what, auto narrow) {
        SceneJournal::Limits limits; narrow(limits.scene.retainedValues);
        bool limited = false;
        try { SceneJournal journal(valued,10,limits); }
        catch (const DecodeError& error) { limited = error.kind == ErrorKind::ResourceLimit; }
        check(limited && inspect(valued) == before,what);
    };
    refused("a journal accepted an initial scene over its element limit",[](EncodeLimits& l) { l.maxElements = 2; });
    refused("a journal accepted an initial scene over its string-byte limit",[](EncodeLimits& l) { l.maxStringBytes = 63; });
    refused("a journal accepted an initial scene over its nesting limit",[](EncodeLimits& l) { l.maxDepth = 2; });
}
void quotasAndSharedWork() {
    context = "aggregate journal quotas and shared scene work";
    const auto one = edit(1), two = edit(2), peerBytes = edit(7,"peer");
    SceneJournal::Limits limits; limits.maxPending = 1;
    Fixture count(limits); count.local(one); const Saved full(count.journal);
    result(OMNI_USD_LIVE_RESOURCE_EXHAUSTED,[&] { count.local(two); }); full.unchanged(count.journal);
    check(count.submits == 1,"pending count failure reached submitter");
    limits = {}; limits.maxPendingBytes = 16384;
    Fixture bytes(limits); bool byteExhausted=false;
    for(size_t i=0;i<1000;++i) {
        const Saved before(bytes.journal); const auto submits=bytes.submits;
        try { bytes.local(one); }
        catch(const DecodeError& e) {
            check(e.kind==ErrorKind::ResourceLimit,"wrong raw-base capacity rejection"); byteExhausted=true;
        } catch(OmniUsdLiveResult e) {
            check(e==OMNI_USD_LIVE_RESOURCE_EXHAUSTED,"wrong journal byte rejection"); byteExhausted=true;
        }
        if(byteExhausted) { before.unchanged(bytes.journal); check(bytes.submits==submits,"byte rejection reached admission"); break; }
    }
    // The configured quota bounds what raw edits retain.
    check(byteExhausted && bytes.submits>0 && bytes.journal.snapshot()->pendingBytes<=limits.maxPendingBytes &&
          bytes.journal.snapshot()->pendingBytes+one.size()>limits.maxPendingBytes,
          "aggregate retained bytes did not bound raw admissions");
    limits = {}; limits.maxReplayBytes = one.size()+two.size();
    Fixture replay(limits); replay.local(one); replay.local(two); const Saved retained(replay.journal);
    auto remote = replay.peer(peerBytes,11);
    result(OMNI_USD_LIVE_RESOURCE_EXHAUSTED,[&] { replay.journal.apply(remote,peerBytes); }); retained.unchanged(replay.journal);
    check(replay.progress.snapshot().applied == 10,"aggregate replay-byte failure advanced applied frontier");
    Fixture generous; generous.local(one); generous.local(two); auto accepted = generous.peer(peerBytes,11); generous.applyPeer(accepted,peerBytes);
    check(number(generous.journal.snapshot()->visible) == 2 && number(generous.journal.snapshot()->visible,"peer") == 7,
        "generous replay baseline did not merge peer and pending edits");
    const auto initial = initialScene(); const auto old = inspect(initial);
    WireSceneWorkBudget budget{1000000};
    const auto a = buildSceneCandidate(initial,one,{},&budget); const auto afterFirst = budget.remaining;
    const auto b = buildSceneCandidate(initial,two,{},&budget);
    check(afterFirst < 1000000 && budget.remaining < afterFirst && number(a) == 1 && number(b) == 2,"scene candidates did not share work budget");
    budget.remaining = 1;
    bool failed = false;
    try { (void)buildSceneCandidate(initial,one,{},&budget); }
    catch (const DecodeError& error) { check(error.kind == ErrorKind::ResourceLimit,"wrong shared scene budget error"); failed = true; }
    check(failed && inspect(initial) == old,"shared work exhaustion changed prior scene");
    WireSceneWorkBudget finite{8192};
    std::vector<WireScenePtr> retainedOwners;
    bool exhausted = false;
    for (int i = 1; i <= 128; ++i) {
        const auto before = finite.remaining;
        try {
            retainedOwners.push_back(buildSceneCandidate(initial,edit(i),{},&finite));
            check(finite.remaining < before,"successful shared transaction did not consume work");
        } catch (const DecodeError& error) {
            check(error.kind == ErrorKind::ResourceLimit,"wrong naturally exhausted shared-work error"); exhausted = true; break;
        }
    }
    check(exhausted && !retainedOwners.empty() && inspect(initial) == old,"finite shared work did not bound successive valid candidates");
    for (size_t i = 0; i < retainedOwners.size(); ++i) check(number(retainedOwners[i]) == int(i+1),"shared-work failure changed earlier owner");
    limits = {}; limits.maxReplayWork = 8192;
    Fixture workLimited(limits, 2048);
    bool replayExhausted = false; size_t acceptedCount = 0;
    // The quota also holds 32 charges of the scene, so exhaustion takes more pending edits than 8192 alone.
    for (int i = 1; i <= 1024; ++i) {
        const Saved before(workLimited.journal); const auto calls = workLimited.submits;
        try { check(workLimited.local(edit(i)) == uint64_t(i),"work-limited admission ID differs"); ++acceptedCount; }
        catch (const DecodeError& error) {
            check(error.kind == ErrorKind::ResourceLimit,"wrong local replay-work error");
            before.unchanged(workLimited.journal); check(workLimited.submits == calls,"replay-work failure reached submitter");
            replayExhausted = true; break;
        }
    }
    check(replayExhausted && acceptedCount > 0 && workLimited.progress.snapshot().retained == acceptedCount,
        "aggregate replay work was reset for each pending operation");

    // Isolate retained-scene work from incoming-wire work: the no-op has no
    // command/field/sample metadata and no USD values to decode. Even ALL eight
    // attempts (and all36 journal replay inputs below) consume under1/16 of the
    // budget in input bytes. A candidate shares its previous scene's sections
    // and storage and measures only the sections its update changes, so a no-op
    // on a large retained value never revisits it and is charged no more than
    // its input and records.
    constexpr size_t payloadSize = 32768, attempts = 8, retainedBudget = payloadSize*3;
    EnvelopeInput largeMetadata; largeMetadata.fields = {{2,0,"payload",{}}};
    const auto large = buildSceneCandidate(initial,encodeUsdEnvelope(largeMetadata,
        DecodedValues{{VtValue(std::string(payloadSize,'x'))},{}}));
    const auto largeBefore = inspect(large);
    const auto noop = encodeEnvelope({});
    check(noop.size()*attempts*(attempts+1)/2 < retainedBudget/16,"retained-work fixture does not separate input costs");
    WireSceneWorkBudget retainedWork{retainedBudget};
    std::vector<WireScenePtr> largeOwners;
    for (size_t i = 0; i < attempts; ++i) largeOwners.push_back(buildSceneCandidate(large,noop,{},&retainedWork));
    check(retainedBudget - retainedWork.remaining < retainedBudget/16 && inspect(large) == largeBefore,
        "a no-op candidate revisited its retained scene");
    for (const auto& owner : largeOwners) check(inspect(owner) == largeBefore,"retained-work rejection mutated an earlier candidate");
    // One build replays every pending intent, so a journal traverses and charges the large retained scene once
    // per replay, not once per pending intent: far more no-op intents fit than the seven payloads of configured
    // work that building each intent in turn exhausted.
    limits = {}; limits.maxReplayWork = payloadSize*7;
    Fixture largeJournal(limits,1024,large);
    for (size_t i = 1; i <= 64; ++i)
        check(largeJournal.local(noop) == i,"one-pass replay charged the retained scene once per pending intent");
    check(largeJournal.progress.snapshot().retained == 64 && largeJournal.journal.snapshot()->pending.size() == 64 &&
          inspect(largeJournal.journal.snapshot()->visible) == largeBefore,"pending no-op intents changed the large scene");
}
void brokenSubmitters() {
    context = "post-publication submitter fault latch";
    for (bool throws : {false,true}) {
        Fixture f;
        const auto real = f.submitter();
        const auto id = f.journal.submitLocal(edit(1),[&](auto bytes,auto owner,auto publication)->uint64_t {
            const auto accepted = real(bytes,std::move(owner),publication);
            if (throws) throw std::runtime_error("broken submitter after publication");
            return accepted+1;
        });
        const auto accepted = f.journal.snapshot();
        check(id == 1 && accepted->submissionFault == OMNI_USD_LIVE_INTERNAL_ERROR && accepted->pending.size() == 1 &&
            accepted->pending[0].id == id && number(accepted->visible) == 1 && f.progress.retained().size() == 1,
            "post-publication fault was reported as rejection or lost accepted edit");
        const Saved before(f.journal);
        result(OMNI_USD_LIVE_INTERNAL_ERROR,[&] { f.local(edit(2)); }); before.unchanged(f.journal);
        auto work = f.ack(id,11);
        result(OMNI_USD_LIVE_INTERNAL_ERROR,[&] { f.applyLocal(work); }); before.unchanged(f.journal);
        check(f.progress.snapshot().applied == 10 && f.submits == 1,"fault latch allowed further mutation/application");
    }
}
void concurrentSnapshots() {
    context = "concurrent immutable journal snapshots";
    Fixture f;
    constexpr size_t revisions = 24;
    std::array<std::promise<void>,revisions> published;
    std::array<std::promise<void>,revisions+1> observed;
    std::vector<std::future<void>> observations;
    for (auto& promise : observed) observations.push_back(promise.get_future());
    auto reader = std::async(std::launch::async,[&] {
        std::vector<SceneJournal::SnapshotPtr> retained;
        const auto coherent = [](const SceneJournal::SnapshotPtr& snapshot) {
            if (snapshot->applied != 10 || number(snapshot->authoritative) != 0 || number(snapshot->visible) != int(snapshot->pending.size()))
                throw std::runtime_error("reader observed mixed visible/pending publication");
            size_t bytes = 0;
            for (size_t j = 0; j < snapshot->pending.size(); ++j) {
                if (snapshot->pending[j].id != j+1 || !snapshot->pending[j].intent)
                    throw std::runtime_error("reader saw unassigned/incorrect admission ID");
                bytes += snapshot->pending[j].intent->retainedBytes;
                if(snapshot->pending[j].intent->localOrdinal!=j+1 || snapshot->pending[j].intent->dependencyCutoff!=j)
                    throw std::runtime_error("reader saw torn local admission provenance");
            }
            if (bytes != snapshot->pendingBytes) throw std::runtime_error("reader saw torn intent-byte accounting");
            if(snapshot->initialApplied!=10 || snapshot->localAdmitted!=snapshot->pending.size() || snapshot->localRetired)
                throw std::runtime_error("reader saw torn local admission ledger");
        };
        for (size_t revision = 0; revision <= revisions; ++revision) {
            if (revision) {
                auto publication = published[revision-1].get_future();
                const auto deadline = std::chrono::steady_clock::now()+std::chrono::seconds(5);
                while (publication.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
                    // Continue probing coherent publication while the writer
                    // prepares/admit-swaps, rather than only reading afterward.
                    coherent(f.journal.snapshot());
                    if (std::chrono::steady_clock::now() >= deadline)
                        throw std::runtime_error("writer did not publish expected journal revision");
                    std::this_thread::yield();
                }
            }
            auto snapshot = f.journal.snapshot();
            coherent(snapshot);
            if (snapshot->pending.size() != revision) throw std::runtime_error("reader skipped expected publication revision");
            retained.push_back(std::move(snapshot));
            for (size_t earlier = 0; earlier < retained.size(); ++earlier) {
                const auto& prior = retained[earlier];
                if (prior->pending.size() != earlier || number(prior->visible) != int(earlier) ||
                    number(prior->authoritative) != 0 || prior->applied != 10)
                    throw std::runtime_error("retained journal snapshot changed after later publication");
            }
            observed[revision].set_value();
        }
    });
    auto awaitObservation = [&](size_t revision) {
        if (observations[revision].wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
            if (reader.wait_for(std::chrono::seconds(0)) == std::future_status::ready) reader.get();
            throw std::runtime_error("reader did not observe required publication revision");
        }
    };
    awaitObservation(0);
    for (size_t i = 1; i <= revisions; ++i) {
        check(f.local(edit(int(i))) == i,"concurrent journal admission ID differs");
        published[i-1].set_value();
        // The next publication cannot overtake the reader. It must observe all
        //25 distinct owners, and keep earlier owners across each new build.
        awaitObservation(i);
    }
    check(reader.wait_for(std::chrono::seconds(10)) == std::future_status::ready,"snapshot reader did not finish"); reader.get();
    check(f.journal.snapshot()->pending.size() == 24 && number(f.journal.snapshot()->visible) == 24,"final local snapshot differs");
}
// Admitting one write, applying one acknowledgment and applying one peer update each cost work in proportion to
// that write or update and the scene, never to the number of pending writes: the journal's deterministic
// per-transaction work counter stays flat across backlogs of 100, 1,000 and 4,000 pending writes, and draining
// a backlog costs work linear in its length.
void backlogScaling() {
    context = "per-write work across pending backlogs";
    struct Costs { size_t admission, acknowledgment, peer, drain; };
    const auto measure = [](size_t backlog) {
        Fixture f({},backlog + 8);
        for (size_t i = 0; i < backlog; ++i) f.local(edit(int(i % 1000),i % 2 ? "value" : "other"));
        Costs costs{f.journal.snapshot()->work,0,0,0};        check(f.journal.snapshot()->pending.size() == backlog,"backlog was not admitted");
        auto first = f.ack(1,11); f.applyLocal(first);
        costs.acknowledgment = f.journal.snapshot()->work;
        const auto peerBytes = edit(5,"peer");
        auto peer = f.peer(peerBytes,12); f.applyPeer(peer,peerBytes);
        costs.peer = f.journal.snapshot()->work;
        uint64_t sequence = 13;
        for (uint64_t id = 2; id <= backlog; ++id) {
            auto work = f.ack(id,sequence++); f.applyLocal(work);
            costs.drain += f.journal.snapshot()->work;
        }
        const auto drained = f.journal.snapshot();
        check(drained->pending.empty() && number(drained->visible) == int((backlog - 1) % 1000) &&
              number(drained->visible,"peer") == 5,"drained backlog lost an edit");
        return costs;
    };
    const auto small = measure(100), medium = measure(1000), large = measure(4000);
    std::cout << "backlog_work admission=" << small.admission << ',' << medium.admission << ',' << large.admission
              << " acknowledgment=" << small.acknowledgment << ',' << medium.acknowledgment << ',' << large.acknowledgment
              << " peer=" << small.peer << ',' << medium.peer << ',' << large.peer
              << " drain=" << small.drain << ',' << medium.drain << ',' << large.drain << std::endl;
    // A generous factor: per-transaction work may vary with bookkeeping, but a replay of every pending write
    // multiplies it by the backlog (40 from 100 to 4,000 writes).
    constexpr size_t factor = 4;
    check(large.admission <= factor * small.admission && medium.admission <= factor * small.admission,
          "admission work grows with the pending backlog");
    check(large.acknowledgment <= factor * small.acknowledgment + factor &&
          medium.acknowledgment <= factor * small.acknowledgment + factor,
          "acknowledgment work grows with the pending backlog");
    check(large.peer <= factor * small.peer && medium.peer <= factor * small.peer,
          "peer application work grows with the pending backlog");
    // Linear drains scale by the backlog ratio (4 from 1,000 to 4,000 writes); quadratic ones by its square.
    check(large.drain <= 2 * 4 * medium.drain && medium.drain <= 2 * 10 * small.drain,
          "draining a backlog costs work superlinear in its length");
}
// Drives a journal with ordered work directly, as the actor hands it over: peers, acknowledgments of the leading
// intent and aggregates that absorb it, at consecutive positions.
struct Driver {
    SceneJournal journal;
    uint64_t position = 10, nextId = 1;
    explicit Driver(WireScenePtr initial, SceneJournal::Limits limits = {}) : journal(std::move(initial),10,std::move(limits)) {}
    uint64_t admit(const std::string& bytes) {
        return journal.submitLocal(bytes,[this](std::string_view, std::shared_ptr<const SceneJournal::Intent>,
                                                LayerProgress::Publication publication) {
            const auto id = nextId++; publication.commit(publication.context,id); return id;
        });
    }
    LayerProgress::Work peer(uint64_t at, uint64_t absorbed = 0) const {
        LayerProgress::Work work; work.id = 1000000 + at; work.first = work.last = at;
        work.disposition = LayerProgress::Disposition::Peer;
        if (absorbed) { work.absorbedLocal = absorbed; work.absorbedThrough = journal.snapshot()->localRetired + absorbed; }
        return work;
    }
    LayerProgress::Work ack(const SceneJournal::Pending& entry, uint64_t at) const {
        LayerProgress::Work work; work.id = entry.id; work.first = work.last = at;
        work.disposition = LayerProgress::Disposition::Acknowledged; work.localOrdinal = entry.intent->localOrdinal;
        return work;
    }
};
// An admission's replay-work quota holds each pending intent's step of a replay of every pending intent (design
// section 8). Admissions that apply each intent to the previous visible scene record exactly the work of the
// matching step of that replay, which continues from the step before it: in a scene of many sections, a step
// derives paths and child groups only first and after an intent that reassigns sibling orders, and saves a copy
// of the scene before a Create or a Reorder.
void replayWorkIsReplaySteps() {
    context = "pending replay work against a full replay";
    const auto create = [](uint64_t section, uint64_t parent, const std::string& name) {
        Command c{}; c.kind = Command::Kind::Create; c.section = section; c.parent = parent; c.name = name;
        c.sectionType = SdfSpecTypePrim; return c;
    };
    EnvelopeInput shape;
    shape.commands = {create(3,1,"Q"),create(4,3,"a"),create(5,3,"b")};
    for (uint64_t i = 0; i < 500; ++i) shape.commands.push_back(create(10 + i,2,"c" + std::to_string(i)));
    Driver d(buildSceneCandidate(initialScene(),encodeEnvelope(shape)));
    const auto structural = [](const Command& command) { EnvelopeInput m; m.commands = {command}; return encodeEnvelope(m); };
    Command reorder{}; reorder.kind = Command::Kind::Reorder; reorder.section = 3; reorder.children = {5,4};
    const auto childOrder = [] {
        EnvelopeInput m; m.fields = {{3,0,"primChildren",{}}};
        return encodeUsdEnvelope(m,DecodedValues{{VtValue(std::vector<TfToken>{TfToken("a"),TfToken("b")})},{}});
    }();
    const std::vector<std::string> intents{edit(1),edit(2,"other"),structural(create(900,2,"n900")),edit(3),
        structural(reorder),edit(4),childOrder,edit(5),structural(create(901,2,"n901")),edit(6)};
    for (const auto& intent : intents) {
        (void)d.admit(intent);
        const auto state = d.journal.snapshot();
        check(state->derivation == SceneJournal::Derivation::Extended,"an admission replayed every pending intent");
        std::vector<std::string_view> replay;
        for (const auto& entry : state->pending) replay.push_back(entry.intent->bytes);
        WireSceneWorkBudget budget{std::numeric_limits<size_t>::max()};
        std::vector<WireReplayStep> steps;
        (void)buildReplayedScene(state->authoritative,replay,{},&budget,false,&steps);
        size_t total = 0;
        for (size_t i = 0; i < steps.size(); ++i) {
            check(state->pending[i].replayWork == steps[i].work,"a pending intent's replay work differs from its replay step");
            total += steps[i].work;
        }
        check(state->replayWork == total,"the pending replay work differs from the replay's steps");
    }
}
// The visible scene's identity graph, hidden rows included; inspect compares the Sdf data.
std::string graphOf(const WireScenePtr& scene) {
    std::ostringstream text;
    scene->visitSections(1u << 20,[&](const WireSectionView& view) {
        text << '|' << view.id << ',' << view.parent << ',' << view.name << ',' << view.order << ',' << view.path;
        return true;
    });
    return text.str();
}
// The visible scene is the replay of the pending intents onto the authoritative scene (design section 8),
// however the journal derives it. Each entry's replayed flag is the replay's prediction for it, and the replay
// bounds are at least the field count and metadata bytes of every scene the replay passes through: the Retained
// decision and the overlay's limit check rely on both.
void visibleIsReplay(const SceneJournal& journal, const char* message) {
    const auto state = journal.snapshot();
    std::vector<std::string_view> intents;
    for (const auto& entry : state->pending) intents.push_back(entry.intent->bytes);
    std::vector<WireReplayStep> steps;
    const auto expected = buildReplayedScene(state->authoritative,intents,{},nullptr,false,&steps);
    check(inspect(state->visible) == inspect(expected) && graphOf(state->visible) == graphOf(expected),message);
    check(steps.size() == state->pending.size(),"the replay did not report a step per pending intent");
    size_t fields = 0, metadata = 0;
    for (size_t i = 0; i < steps.size(); ++i) {
        check(state->pending[i].replayed == steps[i].applied,"a pending entry's replayed flag differs from the replay");
        fields = std::max(fields,steps[i].fields); metadata = std::max(metadata,steps[i].metadataBytes);
    }
    check(state->replayFields >= fields && state->replayMetadataBytes >= metadata,
          "the replay bounds are below a scene the replay passes through");
    if (state->pending.empty()) check(state->replayFields == 0 && state->replayMetadataBytes == 0,"no pending intent left replay bounds");
    size_t wire = 0, retained = 0;
    for (const auto& entry : state->pending) { wire += entry.intent->bytes.size(); retained += entry.intent->retainedBytes; }
    check(state->pendingWireBytes == wire && state->pendingBytes == retained,"pending byte totals differ from the pending intents");
    if (state->pending.empty()) check(state->visible == state->authoritative,"no pending intent left a distinct visible scene");
}
// Seeded histories interleave admissions of field, sample, whole-map and structural intents with peer updates
// (field and sample records that pending intents do or do not also write, records of absent sections, explicit
// child fields and structural commands), acknowledgments, aggregates that absorb an acknowledgment, malformed
// and identity-stealing peer updates, and a rejected or unknown write that stops acknowledgments while peer
// work keeps applying. After every transaction the visible scene equals replaying the pending intents onto the
// authoritative scene, every derivation is exercised, and a failed transaction publishes nothing.
void incrementalReplay(uint64_t seed) {
    context = "incremental visible scene against a full replay";
    const auto command = [](Command::Kind kind, uint64_t section, uint64_t parent, const std::string& name = "",
                            SdfSpecType type = SdfSpecTypePrim) {
        Command c{}; c.kind = kind; c.section = section; c.parent = parent; c.name = name;
        if (kind == Command::Kind::Create) c.sectionType = type;
        return c;
    };
    EnvelopeInput shape;
    shape.commands = {command(Command::Kind::Create,3,2,"A"),command(Command::Kind::Create,4,2,"B"),
                      command(Command::Kind::Create,5,3,"x",SdfSpecTypeAttribute)};
    shape.fields = {{5,0,"timeSamples",{}}};
    const auto base = buildSceneCandidate(initialScene(),encodeUsdEnvelope(shape,
        DecodedValues{{VtValue(SdfTimeSampleMap{{1,VtValue(1)},{2,VtValue(2)}})},{}}));
    std::mt19937_64 random(seed ^ 0x696e6372656dull);
    std::map<SceneJournal::Derivation,size_t> derived;
    size_t failures = 0, stalls = 0, compared = 0;
    for (int round = 0; round < 40; ++round) {
        Driver d(base);
        uint64_t fresh = 100 + uint64_t(round) * 1000;
        std::vector<uint64_t> created; // Identities pending intents create.
        bool stalled = false;
        const auto pick = [&](std::initializer_list<uint64_t> fixed) {
            std::vector<uint64_t> choices(fixed);
            if (!created.empty() && random() % 3 == 0) choices.push_back(created[random() % created.size()]);
            return choices[random() % choices.size()];
        };
        const auto fieldNames = std::array<const char*,3>{"value","other","peer"};
        const auto fieldEdit = [&](bool samplesToo) {
            EnvelopeInput m; DecodedValues v;
            const size_t count = 1 + random() % 3;
            for (size_t i = 0; i < count; ++i) {
                const auto erase = random() % 5 == 0;
                m.fields.push_back({pick({2,3,4,999}),uint64_t(i),fieldNames[random() % 3],{}});
                v.fields.push_back(erase ? VtValue() : VtValue(int(random() % 50)));
            }
            if (samplesToo) {
                if (random() % 2) {
                    m.fields.push_back({5,uint64_t(count),"timeSamples",{}});
                    v.fields.push_back(VtValue(SdfTimeSampleMap{{double(random() % 4),VtValue(int(random() % 9))}}));
                } else {
                    m.samples.push_back({5,double(random() % 4),{}});
                    v.samples.push_back(random() % 4 == 0 ? VtValue() : VtValue(int(random() % 9)));
                }
            }
            return encodeUsdEnvelope(m,v);
        };
        const auto structural = [&](bool local) -> std::string {
            EnvelopeInput m;
            switch (random() % 5) {
            case 0: case 1: {
                const auto id = fresh++;
                if (local) created.push_back(id);
                m.commands = {command(Command::Kind::Create,id,pick({2,3,4}),"N" + std::to_string(id))};
                break;
            }
            case 2: m.commands = {command(Command::Kind::Delete,pick({3,4}),2)}; break;
            case 3: { auto move = command(Command::Kind::Move,4,3,"Moved"); move.oldParent = 2; m.commands = {move}; break; }
            default: { auto reorder = command(Command::Kind::Reorder,2,0); reorder.children = {4,3}; m.commands = {reorder}; }
            }
            return encodeEnvelope(m);
        };
        const auto peerUpdate = [&]() -> std::string {
            switch (random() % 8) {
            case 0: return structural(false);
            case 1: {
                // A peer takes an identity a pending intent creates, which predicts that intent's rejection.
                if (created.empty()) return fieldEdit(false);
                EnvelopeInput m; m.commands = {command(Command::Kind::Create,created[random() % created.size()],2,"Taken" + std::to_string(fresh++))};
                return encodeEnvelope(m);
            }
            case 2: {
                // An explicit child field restating the authoritative order.
                const auto scene = d.journal.snapshot()->authoritative;
                const auto value = scene->data()->Get(SdfPath("/P"),SdfChildrenKeys->PrimChildren);
                if (!value.IsHolding<std::vector<TfToken>>()) return fieldEdit(true);
                EnvelopeInput m; m.fields = {{2,0,"primChildren",{}}};
                return encodeUsdEnvelope(m,DecodedValues{{value},{}});
            }
            default: return fieldEdit(random() % 2 == 0);
            }
        };
        for (int step = 0; step < 48; ++step) {
            const auto before = d.journal.snapshot();
            const Saved saved(d.journal);
            const auto choice = random() % 10;
            bool ok = true;
            try {
                if (choice < 4 && !stalled) {
                    (void)d.admit(random() % 3 ? fieldEdit(random() % 2 == 0) : structural(true));
                } else if (choice < 7) {
                    // Peer updates, then, when acknowledgments flow, the leading intent's acknowledgment.
                    std::vector<std::string> payloads;
                    const size_t peers = choice == 6 ? 0 : random() % 3;
                    for (size_t i = 0; i < peers; ++i) payloads.push_back(peerUpdate());
                    if (choice != 6 && random() % 16 == 0) payloads.push_back("bad envelope");
                    std::vector<LayerProgress::Work> works;
                    std::vector<SceneJournal::Application> batch;
                    for (size_t i = 0; i < payloads.size(); ++i) works.push_back(d.peer(d.position + 1 + i));
                    const bool acknowledge = !stalled && !before->pending.empty() && (payloads.empty() || random() % 2);
                    if (acknowledge) works.push_back(d.ack(before->pending.front(),d.position + 1 + payloads.size()));
                    // Occasionally a peer after the acknowledgment, which the replay derives.
                    if (acknowledge && choice != 6 && random() % 4 == 0) { payloads.push_back(fieldEdit(false)); works.push_back(d.peer(d.position + 2 + payloads.size() - 1)); }
                    if (works.empty()) continue;
                    size_t next = 0;
                    for (size_t i = 0; i < works.size(); ++i) {
                        if (works[i].disposition == LayerProgress::Disposition::Peer) batch.push_back({works[i],payloads[next++],nullptr});
                        else batch.push_back({works[i],before->pending.front().intent->bytes,before->pending.front().intent});
                    }
                    d.journal.apply(batch);
                    d.position += works.size();
                } else if (choice < 8 && !stalled && !before->pending.empty()) {
                    // An aggregate absorbing the leading acknowledged intent: the service's state holds its effect.
                    const auto& head = before->pending.front();
                    d.journal.apply(d.peer(d.position + 1,1),head.intent->bytes);
                    d.position += 1;
                } else if (choice < 9 && !stalled && !before->pending.empty() && random() % 3 == 0) {
                    // A definitive rejection or unknown outcome: no acknowledgment arrives, pending work stays and
                    // peer work keeps applying.
                    stalled = true; ++stalls; continue;
                } else {
                    const auto bytes = peerUpdate();
                    d.journal.apply(d.peer(d.position + 1),bytes);
                    d.position += 1;
                }
            } catch (const DecodeError&) { ok = false; } catch (OmniUsdLiveResult) { ok = false; }
            if (!ok) { ++failures; saved.unchanged(d.journal); continue; }
            const auto after = d.journal.snapshot();
            if (after == before) continue;
            ++derived[after->derivation];
            if (after->derivation == SceneJournal::Derivation::Retained)
                check(after->visible == before->visible,"a retained derivation built a new visible scene");
            visibleIsReplay(d.journal,"the derived visible scene differs from replaying the pending intents");
            ++compared;
        }
    }
    using D = SceneJournal::Derivation;
    std::cout << "incremental_replay compared=" << compared << " failures=" << failures << " stalls=" << stalls
              << " authoritative=" << derived[D::Authoritative] << " replayed=" << derived[D::Replayed]
              << " extended=" << derived[D::Extended] << " retained=" << derived[D::Retained]
              << " overlaid=" << derived[D::Overlaid] << std::endl;
    check(compared > 1000 && failures > 10 && stalls > 5,"seeded histories exercised too few transactions");
    for (const auto kind : {D::Authoritative,D::Replayed,D::Extended,D::Retained,D::Overlaid})
        check(derived[kind] > 10,"seeded histories exercised a derivation too rarely");
}
// Own pending edits hide peer updates until confirmed: a peer write of a field a pending intent also writes stays
// hidden, a peer write of another field shows at once, and after the own write is confirmed the ordered result
// shows. Field and metadata limits hold on every scene the replay passes through, whichever derivation applies.
void ownEditsHidePeerUpdates() {
    context = "own pending edits hide peer updates";
    Driver d(initialScene());
    d.admit(edit(1)); d.admit(edit(2,"other"));
    const auto hidden = edit(7), shown = edit(5,"peer");
    d.journal.apply(d.peer(11),hidden);
    auto state = d.journal.snapshot();
    check(state->derivation == SceneJournal::Derivation::Overlaid && number(state->visible) == 1 &&
          number(state->authoritative) == 7,"a peer write of a pending field showed before confirmation");
    d.journal.apply(d.peer(12),shown);
    state = d.journal.snapshot();
    check(state->derivation == SceneJournal::Derivation::Overlaid && number(state->visible,"peer") == 5,
          "a peer write of a field no pending intent writes stayed hidden");
    visibleIsReplay(d.journal,"overlaid scene differs from the replay");
    const auto ownFirst = state->pending.front();
    d.journal.apply(d.ack(ownFirst,13),ownFirst.intent->bytes,ownFirst.intent);
    state = d.journal.snapshot();
    check(state->derivation == SceneJournal::Derivation::Retained && number(state->authoritative) == 1 &&
          number(state->visible) == 1 && state->pending.size() == 1,"confirmation changed the visible scene");
    // With no pending write of the field left, a peer write shows.
    d.journal.apply(d.peer(14),edit(8));
    state = d.journal.snapshot();
    check(number(state->visible) == 8 && number(state->visible,"other") == 2,"a confirmed field hid a later peer write");
    visibleIsReplay(d.journal,"overlaid scene differs from the replay after confirmation");
    const auto ownSecond = state->pending.front();
    d.journal.apply(d.ack(ownSecond,15),ownSecond.intent->bytes,ownSecond.intent);
    state = d.journal.snapshot();
    check(state->derivation == SceneJournal::Derivation::Authoritative && state->visible == state->authoritative &&
          number(state->visible,"other") == 2,"the last confirmation left a distinct visible scene");

    context = "field limits on every replayed scene";
    // A pending intent adds three fields that a later one erases; a peer adding one field elsewhere pushes the
    // scene after the first intent over a limit that the final scene stays under. When the erasing intent also
    // writes the peer's field, the peer write stays hidden in the visible scene yet still lies in every scene the
    // replay passes through before that intent.
    const auto three = [] {
        EnvelopeInput m; m.fields = {{2,0,"extraA",{}},{2,1,"extraB",{}},{2,2,"extraC",{}}};
        return encodeUsdEnvelope(m,DecodedValues{{VtValue(1),VtValue(2),VtValue(3)},{}});
    }();
    const auto erase = [](bool share) {
        EnvelopeInput m; m.fields = {{2,0,"extraA",{}},{2,1,"extraB",{}},{2,2,"extraC",{}}};
        DecodedValues v{{VtValue(),VtValue(),VtValue()},{}};
        if (share) { m.fields.push_back({2,3,"peer",{}}); v.fields.push_back(VtValue(9)); }
        return encodeUsdEnvelope(m,v);
    };
    for (const bool share : {false,true}) {
        const auto second = erase(share);
        Driver probe(initialScene()); probe.admit(three); probe.admit(second);
        const auto peak = probe.journal.snapshot()->replayFields;
        for (const bool room : {false,true}) {
            SceneJournal::Limits limits; limits.scene.maxFields = peak + (room ? 1 : 0);
            Driver bounded(initialScene(),limits); bounded.admit(three); bounded.admit(second);
            const Saved before(bounded.journal);
            const auto peer = edit(4,"peer");
            std::vector<std::string_view> replay{three,second};
            bool replayFails = false;
            try { (void)buildReplayedScene(buildSceneCandidate(bounded.journal.snapshot()->authoritative,peer),replay,limits.scene); }
            catch (const DecodeError& error) { replayFails = error.kind == ErrorKind::ResourceLimit; }
            check(replayFails == !room,"fixture does not place the limit at the intermediate scene");
            if (room) {
                bounded.journal.apply(bounded.peer(11),peer);
                const auto state = bounded.journal.snapshot();
                check(state->derivation == SceneJournal::Derivation::Overlaid,"a peer write within limits replayed");
                check(number(state->visible,"peer") == (share ? 9 : 4),"the peer write's visibility differs from the replay");
                visibleIsReplay(bounded.journal,"overlaid scene within limits differs from the replay");
            } else {
                result(OMNI_USD_LIVE_RESOURCE_EXHAUSTED,[&] { bounded.journal.apply(bounded.peer(11),peer); });
                before.unchanged(bounded.journal);
            }
        }
    }
}
} // namespace
// Default limits hold a layer larger than any single upload: more sections, fields and value elements than a
// fixed per-layer ceiling of 262,144 sections or 1,048,576 elements allows. A peer update applies to it and a
// local edit compiles and admits against it.
void largeLayer() {
    context = "large layer under default limits";
    constexpr uint64_t sections = 300000;
    // A layer arrives as a snapshot diff, as the service's catch-up sends it.
    EnvelopeInput metadata; DecodedValues values;
    metadata.isDiff = true;
    metadata.commands.reserve(sections + 1);
    Command p{}; p.kind = Command::Kind::DiffChange; p.section = 2; p.parent = 1; p.name = "P"; p.sectionType = SdfSpecTypePrim;
    metadata.commands.push_back(p); metadata.fields.push_back({2,0,"value",{}}); values.fields.push_back(VtValue(0));
    const VtArray<GfVec3f> points(4, GfVec3f(1.0f, 2.0f, 3.0f));
    for (uint64_t i = 0; i < sections; ++i) {
        Command c{}; c.kind = Command::Kind::DiffChange; c.section = 3 + i; c.parent = 1;
        c.name = "C" + std::to_string(i); c.sectionType = SdfSpecTypePrim; c.sectionOrder = uint32_t(1 + i);
        metadata.commands.push_back(std::move(c));
        metadata.fields.push_back({3 + i,0,"points",{}}); values.fields.push_back(VtValue(points));
    }
    const auto scene = buildSceneCandidate({},encodeUsdEnvelope(metadata,values));
    check(scene->sectionCount() == sections + 2,"large layer lost sections");
    context = "large layer: creation seed";
    // One CreateLayer holds at most the service's entry limit, which the
    // layer's sections and fields exceed; with that limit lifted, the seed
    // compiler holds the layer whole.
    bool refused = false;
    try { (void)compileSceneSeed(*scene->data()); }
    catch (const DeltaEntryLimitExceeded& error) { refused = error.counts().actual == 2 * sections + 3; }
    check(refused,"a seed over the service's entry limit was not refused");
    SceneSeedLimits unlimited; unlimited.entries.maxEntries = std::numeric_limits<size_t>::max();
    const auto seed = compileSceneSeed(*scene->data(),{},unlimited);
    check(seed.scene->sectionCount() == sections + 2,"creation seed lost sections of the large layer");
    context = "large layer: journal"; Fixture f({},128,scene);
    context = "large layer: peer update"; const auto peer = f.peer(edit(7,"peer"),11); f.applyPeer(peer,edit(7,"peer"));
    context = "large layer: local edit";
    check(number(f.journal.snapshot()->authoritative,"peer") == 7,"peer update did not apply to the large layer");
    f.local(edit(5));
    check(number(f.journal.snapshot()->visible) == 5,"local edit did not admit against the large layer");
    context = "large layer: captured edits";
    const auto base = f.journal.snapshot()->visible;
    const auto log = capturedLog(base,[](SdfAbstractData& data) {
        data.Set(SdfPath("/P"),TfToken("value"),VtValue(6));
        data.CreateSpec(SdfPath("/New"),SdfSpecTypePrim);
        auto children = data.Get(SdfPath::AbsoluteRootPath(),SdfChildrenKeys->PrimChildren).UncheckedGet<std::vector<TfToken>>();
        children.push_back(TfToken("New"));
        data.Set(SdfPath::AbsoluteRootPath(),SdfChildrenKeys->PrimChildren,VtValue(children));
        data.Set(SdfPath("/C7"),TfToken("points"),VtValue(VtArray<GfVec3f>(8,GfVec3f(4.0f))));
    });
    check(f.journal.submitCaptured(base,log,f.submitter()) != 0,"captured edits did not admit against the large layer");
    const auto visible = f.journal.snapshot()->visible;
    check(number(visible) == 6 && visible->data()->HasSpec(SdfPath("/New")),"captured edits did not apply to the large layer");
}
// A bulk create into a layer of thousands of specs admits under default limits: planning and building the
// update cost work in proportion to the batch plus the layer, not their product.
// N pending small writes on a large layer charge O(N x write), not O(N x layer), and pin no scene of their own:
// each captured flush rebaselines like the host, and a superseded base or sealed host copy is released at once.
// The pending-byte limit still refuses a write whose own retained data is large.
void pendingChargeIsPerWrite() {
    context = "pending charge per write";
    EnvelopeInput metadata; DecodedValues values; metadata.isDiff = true;
    Command p{}; p.kind = Command::Kind::DiffChange; p.section = 2; p.parent = 1; p.name = "P"; p.sectionType = SdfSpecTypePrim;
    metadata.commands.push_back(p); metadata.fields.push_back({2,0,"value",{}}); values.fields.push_back(VtValue(0));
    uint64_t next = 3;
    for (uint64_t i = 0; i < 2000; ++i) {
        const auto prim = next++;
        Command c{}; c.kind = Command::Kind::DiffChange; c.section = prim; c.parent = 1;
        c.name = "C" + std::to_string(i); c.sectionType = SdfSpecTypePrim; c.sectionOrder = uint32_t(1 + i);
        metadata.commands.push_back(std::move(c));
        for (const char* name : {"a","b","c"}) {
            Command a{}; a.kind = Command::Kind::DiffChange; a.section = next++; a.parent = prim; a.name = name;
            a.sectionType = SdfSpecTypeAttribute; metadata.commands.push_back(std::move(a));
            metadata.fields.push_back({a.section,0,"default",{}}); values.fields.push_back(VtValue(1.0f));
        }
    }
    const auto layer = buildSceneCandidate({},encodeUsdEnvelope(metadata,values));
    const auto layerCharge = layer->retainedCharge();
    Fixture f({},128,layer);
    constexpr int writes = 32;
    WireScenePtr base = layer;
    std::vector<std::weak_ptr<const WireScene>> bases;
    std::vector<std::weak_ptr<const SdfOperationLog>> logs;
    for (int i = 0; i < writes; ++i) {
        const auto log = capturedLog(base,[i](SdfAbstractData& data) {
            data.Set(SdfPath("/C" + std::to_string(i) + ".a"),SdfFieldKeys->Default,VtValue(float(i)));
        });
        WireScenePtr applied;
        check(f.journal.submitCaptured(base,log,f.submitter(),&applied) != 0 && applied,"small write was not admitted");
        bases.push_back(base); logs.push_back(log);
        base = applied; // The host's next flush compiles against the verified candidate.
    }
    const auto state = f.journal.snapshot();
    check(state->pending.size() == size_t(writes),"small writes were not all pending");
    for (const auto& entry : state->pending)
        check(entry.intent->retainedBytes < 4096,"a one-field write was charged for the layer");
    check(state->pendingBytes < size_t(writes) * 4096 && state->pendingBytes < layerCharge,
          "pending writes were charged per layer copy");
    check(f.progress.snapshot().pendingBytes == state->pendingBytes,"progress and journal pending bytes differ");
    // Only the layer itself (still the fixture's authoritative scene) stays alive; no pending write pins a base
    // scene or the host's sealed copy.
    for (size_t i = 1; i < bases.size(); ++i) check(bases[i].expired(),"a pending write pinned its base scene");
    for (const auto& log : logs) check(log.expired(),"a pending write pinned the host's sealed copy");
    check(number(f.journal.snapshot()->visible) == 0 &&
          f.journal.snapshot()->visible->data()->Get(SdfPath("/C31.a"),SdfFieldKeys->Default) == VtValue(31.0f),
          "pending writes did not replay");

    context = "pending charge limit";
    SceneJournal::Limits small; small.maxPendingBytes = 64 * 1024;
    Fixture bounded(small);
    WireScenePtr current = bounded.journal.snapshot()->visible;
    const auto write = [&](size_t size) {
        const auto log = capturedLog(current,[size](SdfAbstractData& data) {
            data.Set(SdfPath("/P"),TfToken("blob"),VtValue(std::string(size,'x')));
        });
        WireScenePtr applied;
        const auto id = bounded.journal.submitCaptured(current,log,bounded.submitter(),&applied);
        current = applied;
        return id;
    };
    bool definitive = false;
    try { write(100 * 1024); } catch (const CapturedRejection& rejection) {
        definitive = rejection.result == OMNI_USD_LIVE_RESOURCE_EXHAUSTED;
    }
    check(definitive,"a write larger than the pending-byte limit was admitted");
    current = bounded.journal.snapshot()->visible;
    size_t admitted = 0;
    bool exhausted = false;
    for (int i = 0; i < 8 && !exhausted; ++i) {
        try { write(16 * 1024 + size_t(i)); ++admitted; } catch (OmniUsdLiveResult value) {
            exhausted = value == OMNI_USD_LIVE_RESOURCE_EXHAUSTED;
        }
    }
    check(exhausted && admitted >= 1 && admitted <= 3 && bounded.journal.snapshot()->pendingBytes <= small.maxPendingBytes,
          "pending-byte limit did not bound large retained writes");
}
void bulkCreateIntoLargerLayer() {
    context = "bulk create into a larger layer";
    constexpr uint64_t prims = 2000;
    EnvelopeInput metadata; DecodedValues values;
    metadata.isDiff = true;
    Command p{}; p.kind = Command::Kind::DiffChange; p.section = 2; p.parent = 1; p.name = "P"; p.sectionType = SdfSpecTypePrim;
    metadata.commands.push_back(p); metadata.fields.push_back({2,0,"value",{}}); values.fields.push_back(VtValue(0));
    uint64_t next = 3;
    for (uint64_t i = 0; i < prims; ++i) {
        const auto prim = next++;
        Command c{}; c.kind = Command::Kind::DiffChange; c.section = prim; c.parent = 1;
        c.name = "C" + std::to_string(i); c.sectionType = SdfSpecTypePrim; c.sectionOrder = uint32_t(1 + i);
        metadata.commands.push_back(std::move(c));
        for (const char* name : {"a","b","c"}) {
            Command a{}; a.kind = Command::Kind::DiffChange; a.section = next++; a.parent = prim; a.name = name;
            a.sectionType = SdfSpecTypeAttribute; metadata.commands.push_back(std::move(a));
            metadata.fields.push_back({a.section,0,"default",{}}); values.fields.push_back(VtValue(1.0f));
        }
    }
    Fixture f({},128,buildSceneCandidate({},encodeUsdEnvelope(metadata,values)));
    const auto base = f.journal.snapshot()->visible;
    const auto log = capturedLog(base,[](SdfAbstractData& data) {
        auto children = data.Get(SdfPath::AbsoluteRootPath(),SdfChildrenKeys->PrimChildren).UncheckedGet<std::vector<TfToken>>();
        for (int i = 0; i < 100; ++i) {
            const auto name = "N" + std::to_string(i);
            const SdfPath prim("/" + name);
            data.CreateSpec(prim,SdfSpecTypePrim);
            std::vector<TfToken> properties;
            for (const char* attribute : {"x","y","z","w"}) {
                data.CreateSpec(prim.AppendProperty(TfToken(attribute)),SdfSpecTypeAttribute);
                data.Set(prim.AppendProperty(TfToken(attribute)),SdfFieldKeys->Default,VtValue(2.0f));
                properties.emplace_back(attribute);
            }
            data.Set(prim,SdfChildrenKeys->PropertyChildren,VtValue(properties));
            children.emplace_back(name);
        }
        data.Set(SdfPath::AbsoluteRootPath(),SdfChildrenKeys->PrimChildren,VtValue(children));
    });
    // One delivered update of 100 Creates into a 2,800-spec layer applies under default limits.
    context = "bulk create delivered into a larger layer";
    {
        EnvelopeInput small; small.isDiff = true;
        for (uint64_t i = 0; i < 2800; ++i) {
            Command c{}; c.kind = Command::Kind::DiffChange; c.section = 10 + i; c.parent = 1;
            c.name = "S" + std::to_string(i); c.sectionType = SdfSpecTypePrim; c.sectionOrder = uint32_t(i);
            small.commands.push_back(std::move(c));
        }
        const auto layer = buildSceneCandidate({},encodeEnvelope(small));
        EnvelopeInput creates;
        for (uint64_t i = 0; i < 100; ++i) {
            Command c{}; c.kind = Command::Kind::Create; c.section = 100000 + i; c.parent = 1;
            c.name = "D" + std::to_string(i); c.sectionType = SdfSpecTypePrim;
            creates.commands.push_back(std::move(c));
        }
        const auto applied = buildSceneCandidate(layer,encodeEnvelope(creates),{},nullptr,WireSceneBatchPolicy::Delivered);
        check(applied->sectionCount() == 2901 && applied->data()->HasSpec(SdfPath("/D99")),
              "delivered bulk create did not apply to the larger layer");
    }
    context = "bulk create into a larger layer";
    check(f.journal.submitCaptured(base,log,f.submitter()) != 0,"bulk create did not admit against the larger layer");
    check(f.journal.snapshot()->visible->data()->HasSpec(SdfPath("/N99.w")),"bulk create did not apply");
}
void runFieldEditIntentTests(uint64_t seed);
void runSceneNamespaceTests(uint64_t seed);
void usdPublicationTests();
int measureUsdPublicationNotices();
void sceneSeedTests();
void footprintWorkTests(uint64_t seed);
int main(int argc, char** argv) {
    try {
        if (argc > 1 && std::string(argv[1]) == "--measure-usd-notices") return measureUsdPublicationNotices();
        const uint64_t seed = argc > 1 ? std::stoull(argv[1]) : 42;
        std::cout << "master_seed=" << seed << " scenario=scene-journal" << std::endl;
        // Every build checks its incremental derivation against a whole-scene
        // one, and every compilation its footprint against the whole scene.
        setWireSceneDerivationCheck(true);
        setFieldEditCompileCheck(true);
        backlogScaling(); replayWorkIsReplaySteps(); ownEditsHidePeerUpdates(); incrementalReplay(seed);
        admissionAndIds(); admissionHistory(); acknowledgementsAndABA(); peerReplayAndFailure(); aggregateRetiresAbsorbedIntents(); committedAndNettedRetirement(); batchedApplication(); onePassReplay(seed); operationPreservation(); ownershipRetirement();
        initialSceneLimits(); quotasAndSharedWork(); brokenSubmitters(); concurrentSnapshots(); authoredAdmission(); authoredCreateAdmission(); staleReorderReplay(); liveIdentityCreateReplay(); authoredStructuralAdmission(); capturedShapesAndEmptyFlushes(); capturedRejectionClassification(); capturedFlushOrdering(); authoredPublication(); authoredSharedWork(); runFieldEditIntentTests(seed); runSceneNamespaceTests(seed);
        largeLayer();
        bulkCreateIntoLargerLayer();
        pendingChargeIsPerWrite();
        usdPublicationTests();
        context="creation seed compilation"; sceneSeedTests();
        context="footprint work"; footprintWorkTests(seed);
        std::cout << "scene journal admission/ACK/replay/quota/retention contracts passed" << std::endl; return 0;
    } catch (OmniUsdLiveResult result) { std::cerr << "scene journal failed (" << context << "): result=" << result << std::endl; return 1;
    } catch (const std::exception& error) { std::cerr << "scene journal failed (" << context << "): " << error.what() << std::endl; return 1; }
}
