// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "LiveLayerStatus.h"
#include "LiveLayerAdmission.h"
#include "LiveLayerActor.h"
#include "LiveDiagnostics.h"
#include "LiveRuntime.h"
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <new>
#include <vector>

void liveTestFailAllocationsAfter(size_t) noexcept;
void liveTestRestoreAllocations() noexcept;

using namespace omni::usd::live;
namespace {
template<class T> T record() {
    T value{}; value.struct_size=sizeof(value); value.abi_version=OMNI_USD_LIVE_ABI_VERSION; return value;
}
template<class Action> void rejects(OmniUsdLiveResult expected, Action action) {
    bool rejected=false;
    try { action(); } catch(OmniUsdLiveResult result) { assert(result==expected); rejected=true; }
    assert(rejected);
}
OmniUsdLiveStatus value(uint32_t state=OMNI_USD_LIVE_STATE_OPENING) {
    auto result=record<OmniUsdLiveStatus>();
    result.subject_kind=OMNI_USD_LIVE_SUBJECT_LAYER;
    result.state=state; result.operation=OMNI_USD_LIVE_OPERATION_CREATE;
    return result;
}
struct Fixture {
    std::shared_ptr<Runtime> runtime;
    OmniUsdLiveEndpoint endpoint=0;
    explicit Fixture(uint32_t subjects=8,const char* token="synthetic-no-rpc") {
        auto options=record<OmniUsdLiveRuntimeOptions>();
        options.max_status_subjects=subjects;
        options.flags=OMNI_USD_LIVE_RUNTIME_DISABLE_DIAGNOSTIC_HISTORY;
        assert(omniUsdLiveInitialize(&options)==OMNI_USD_LIVE_OK);
        runtime=getRuntime();
        auto endpointOptions=record<OmniUsdLiveEndpointOptions>();
        endpointOptions.authority="status-creation.invalid:35499";
        endpointOptions.access_token=token;
        assert(omniUsdLiveRegisterEndpoint(&endpointOptions,&endpoint)==OMNI_USD_LIVE_OK);
    }
    auto admission(const char* name="root.live") {
        return runtime->layerAdmissions->reserve(endpoint,"opaque/session",name);
    }
    OmniUsdLiveStatus query(const std::shared_ptr<LayerAdmission>& admission) {
        auto status=record<OmniUsdLiveStatus>();
        assert(omniUsdLiveGetLayerStatus(admission->canonicalUri().c_str(),&status)==OMNI_USD_LIVE_OK);
        return status;
    }
    void noRpcOrDiagnosticHistory() {
        const auto snapshot=runtime->diagnostics->snapshot(0,128);
        assert(snapshot.metrics.rpcAttempts==0 && snapshot.metrics.rpcCompletions==0 && snapshot.metrics.rpcInFlight==0);
        assert(snapshot.records.empty());
    }
    ~Fixture() {
        noRpcOrDiagnosticHistory();
        if(endpoint) assert(omniUsdLiveUnregisterEndpoint(endpoint,3000)==OMNI_USD_LIVE_OK);
        runtime.reset();
        assert(omniUsdLiveShutdown(3000)==OMNI_USD_LIVE_OK);
    }
};
struct Observer {
    std::mutex mutex;
    std::condition_variable changed;
    std::vector<OmniUsdLiveStatus> events;
    static void OMNI_USD_LIVE_CALL callback(void* opaque,const OmniUsdLiveStatusEvent* event) {
        auto& self=*static_cast<Observer*>(opaque);
        assert(event && event->status);
        std::lock_guard<std::mutex> lock(self.mutex);
        for(const auto& previous:self.events)
            if(previous.subject_id==event->status->subject_id) assert(previous.revision<event->status->revision);
        self.events.push_back(*event->status); self.changed.notify_all();
    }
    OmniUsdLiveStatus wait(uint64_t subject,uint32_t state) {
        std::unique_lock<std::mutex> lock(mutex);
        const auto found=[&] {
            for(const auto& event:events) if(event.subject_id==subject && event.state==state) return true;
            return false;
        };
        assert(changed.wait_for(lock,std::chrono::seconds(3),found));
        for(const auto& event:events) if(event.subject_id==subject && event.state==state) return event;
        assert(false); return {};
    }
    OmniUsdLiveStatus waitRevision(uint64_t subject,uint64_t revision) {
        std::unique_lock<std::mutex> lock(mutex);
        const auto found=[&] {
            for(const auto& event:events)
                if(event.subject_id==subject && event.revision>=revision) return true;
            return false;
        };
        assert(changed.wait_for(lock,std::chrono::seconds(3),found));
        for(const auto& event:events)
            if(event.subject_id==subject && event.revision>=revision) return event;
        assert(false); return {};
    }
};
OmniUsdLiveStatus creating() {
    auto pending=value();
    pending.pending_write_count=1; pending.pending_bytes=913; pending.has_unconfirmed_edits=1;
    return pending;
}
void abandonedOpeningConfirmationOrders() {
    // Both serializations of the completion/abandonment race must settle to
    // the same public cause. No deadline expiry or scheduler ordering is used.
    for(bool confirmFirst:{false,true}) {
        Fixture f;
        auto admission=f.admission();
        auto status=LayerStatus::create(f.runtime,admission,true);
        Observer observer; OmniUsdLiveCallbackHandle callback=0;
        assert(omniUsdLiveRegisterStatusCallback(f.endpoint,&observer,Observer::callback,&callback)==OMNI_USD_LIVE_OK);
        assert(status->publishCreation(creating())==OMNI_USD_LIVE_OK);
        const auto pending=f.query(admission);
        assert(pending.pending_write_count==1 && pending.pending_bytes==913 && pending.has_unconfirmed_edits);
        if(confirmFirst) {
            assert(status->publishCreation(value(),true)==OMNI_USD_LIVE_OK);
            const auto confirmed=f.query(admission);
            assert(confirmed.state==OMNI_USD_LIVE_STATE_OPENING && confirmed.result==OMNI_USD_LIVE_OK &&
                   !confirmed.pending_write_count && !confirmed.pending_bytes && !confirmed.has_unconfirmed_edits);
        }
        assert(status->abandonOpening(OMNI_USD_LIVE_TIMEOUT)==OMNI_USD_LIVE_OK);
        if(!confirmFirst) {
            const auto abandoned=f.query(admission);
            assert(abandoned.subject_id==status->id() && abandoned.state==OMNI_USD_LIVE_STATE_OPENING &&
                   abandoned.result==OMNI_USD_LIVE_TIMEOUT && abandoned.operation==OMNI_USD_LIVE_OPERATION_CREATE &&
                   abandoned.pending_write_count==1 && abandoned.pending_bytes==913 && abandoned.has_unconfirmed_edits &&
                   !abandoned.progress_known && !abandoned.incident_id && !abandoned.recovery_action &&
                   !abandoned.grpc_status_present && !abandoned.grpc_status);
            const auto delivered=observer.waitRevision(status->id(),abandoned.revision);
            assert(delivered.state==abandoned.state && delivered.result==abandoned.result &&
                   delivered.pending_write_count==1 && delivered.pending_bytes==913 && !delivered.incident_id);
            assert(status->canHandoff()==OMNI_USD_LIVE_BUSY && status->handoff()==OMNI_USD_LIVE_BUSY);
            assert(status->publishActor(value(OMNI_USD_LIVE_STATE_READY))==OMNI_USD_LIVE_BUSY);
            // A repeated local failure or refreshed in-flight finite status
            // cannot replace the first abandonment cause or forget its bytes.
            assert(status->abandonOpening(OMNI_USD_LIVE_INTERNAL_ERROR)==OMNI_USD_LIVE_OK);
            assert(status->publishCreation(creating())==OMNI_USD_LIVE_OK);
            const auto repeated=f.query(admission);
            assert(repeated.result==OMNI_USD_LIVE_TIMEOUT && repeated.pending_write_count==1 &&
                   repeated.pending_bytes==913 && repeated.has_unconfirmed_edits && !repeated.incident_id);
            assert(status->publishCreation(value(),true)==OMNI_USD_LIVE_OK);
        }
        const auto failed=f.query(admission);
        assert(failed.subject_id==status->id() && failed.state==OMNI_USD_LIVE_STATE_FAILED &&
               failed.result==OMNI_USD_LIVE_TIMEOUT && failed.operation==OMNI_USD_LIVE_OPERATION_OPEN &&
               failed.incident_id && failed.recovery_action==OMNI_USD_LIVE_RECOVERY_PRESERVE_AND_REOPEN &&
               !failed.pending_write_count && !failed.pending_bytes && !failed.has_unconfirmed_edits &&
               !failed.progress_known && !failed.grpc_status_present && !failed.grpc_status);
        const auto delivered=observer.waitRevision(status->id(),failed.revision);
        assert(delivered.state==failed.state && delivered.result==failed.result &&
               delivered.incident_id==failed.incident_id && !delivered.has_unconfirmed_edits);
        assert(status->canHandoff()==OMNI_USD_LIVE_TIMEOUT && status->handoff()==OMNI_USD_LIVE_TIMEOUT);
        assert(status->abandonOpening(OMNI_USD_LIVE_INTERNAL_ERROR)==OMNI_USD_LIVE_OK);
        assert(status->publishCreation(value(),true)==OMNI_USD_LIVE_OK);
        assert(status->publishCreation(value(OMNI_USD_LIVE_STATE_READY),true)==OMNI_USD_LIVE_INVALID_ARGUMENT);
        const auto repeated=f.query(admission);
        assert(repeated.revision==failed.revision && repeated.incident_id==failed.incident_id &&
               repeated.state==OMNI_USD_LIVE_STATE_FAILED && !repeated.pending_write_count &&
               !repeated.pending_bytes && !repeated.has_unconfirmed_edits);
        assert(omniUsdLiveUnregisterCallback(callback,3000)==OMNI_USD_LIVE_OK);
        f.noRpcOrDiagnosticHistory();
    }
}
void abandonmentPreservesFiniteDisposition() {
    // The first case is the independent Creating -> Rejected control: a
    // definitive rejection retires the edit flag without actor progress.
    for(unsigned scenario=0;scenario!=3;++scenario) {
        const bool abandon=scenario!=0, unknown=scenario==2;
        Fixture f;
        auto admission=f.admission();
        auto status=LayerStatus::create(f.runtime,admission,true);
        assert(status->publishCreation(creating())==OMNI_USD_LIVE_OK);
        if(abandon) assert(status->abandonOpening(OMNI_USD_LIVE_TIMEOUT)==OMNI_USD_LIVE_OK);
        auto completion=value(OMNI_USD_LIVE_STATE_FAILED);
        completion.result=unknown?OMNI_USD_LIVE_OUTCOME_UNKNOWN:OMNI_USD_LIVE_PERMISSION_DENIED;
        completion.grpc_status_present=1; completion.grpc_status=unknown?14:7;
        completion.recovery_action=OMNI_USD_LIVE_RECOVERY_PRESERVE_AND_REOPEN;
        if(unknown) {
            completion.pending_write_count=1; completion.pending_bytes=913; completion.has_unconfirmed_edits=1;
        }
        assert(status->publishCreation(completion)==OMNI_USD_LIVE_OK);
        const auto failed=f.query(admission);
        assert(failed.state==completion.state && failed.result==completion.result &&
               failed.operation==OMNI_USD_LIVE_OPERATION_CREATE && failed.grpc_status_present==1 &&
               failed.grpc_status==completion.grpc_status && failed.incident_id &&
               failed.recovery_action==OMNI_USD_LIVE_RECOVERY_PRESERVE_AND_REOPEN && !failed.progress_known &&
               failed.pending_write_count==(unknown?1u:0u) && failed.pending_bytes==(unknown?913u:0u) &&
               failed.has_unconfirmed_edits==unsigned(unknown));
        assert(status->handoff()==completion.result);
        assert(status->abandonOpening(OMNI_USD_LIVE_INTERNAL_ERROR)==OMNI_USD_LIVE_OK);
        assert(status->publishCreation(value(),true)==OMNI_USD_LIVE_OK);
        assert(status->publishActor(value(OMNI_USD_LIVE_STATE_READY))==OMNI_USD_LIVE_BUSY);
        const auto repeated=f.query(admission);
        assert(repeated.revision==failed.revision && repeated.result==completion.result &&
               repeated.operation==OMNI_USD_LIVE_OPERATION_CREATE && repeated.incident_id==failed.incident_id &&
               repeated.grpc_status_present && repeated.grpc_status==completion.grpc_status &&
               repeated.pending_write_count==failed.pending_write_count && repeated.pending_bytes==failed.pending_bytes &&
               repeated.has_unconfirmed_edits==failed.has_unconfirmed_edits);
        f.noRpcOrDiagnosticHistory();
    }
}
void sharedSubjectAndFailureRetirement() {
    Fixture f;
    auto admission=f.admission();
    Observer observer;
    OmniUsdLiveCallbackHandle callback=0;
    assert(omniUsdLiveRegisterStatusCallback(f.endpoint,&observer,Observer::callback,&callback)==OMNI_USD_LIVE_OK);
    auto creation=LayerStatus::create(f.runtime,admission,true);
    const auto id=creation->id();
    auto initial=f.query(admission);
    assert(initial.subject_id==id && initial.state==OMNI_USD_LIVE_STATE_OPENING &&
           initial.operation==OMNI_USD_LIVE_OPERATION_CREATE && !initial.progress_known);
    observer.wait(id,OMNI_USD_LIVE_STATE_OPENING);
    f.noRpcOrDiagnosticHistory(); // Subject and callback exist before any RPC.
    assert(creation->handoff()==OMNI_USD_LIVE_BUSY);
    assert(creation->publishActor(value())==OMNI_USD_LIVE_BUSY);
    assert(creation->publishCreation(value(OMNI_USD_LIVE_STATE_READY),true)==OMNI_USD_LIVE_INVALID_ARGUMENT);
    assert(f.query(admission).revision==initial.revision);
    assert(creation->handoff()==OMNI_USD_LIVE_BUSY); // Rejected READY cannot confirm.
    assert(creation->publishCreation(value(),true)==OMNI_USD_LIVE_OK);
    auto actor=creation;
    actor->validate(f.runtime,admission);
    assert(actor->handoff()==OMNI_USD_LIVE_OK);
    assert(actor->handoff()==OMNI_USD_LIVE_ALREADY_EXISTS);
    assert(f.query(admission).subject_id==id);
    auto ready=value(OMNI_USD_LIVE_STATE_READY);
    ready.operation=OMNI_USD_LIVE_OPERATION_UPDATE; ready.progress_known=1;
    ready.pending_write_count=3; ready.pending_bytes=777; ready.has_unconfirmed_edits=1;
    ready.grpc_status_present=1; ready.grpc_status=14;
    assert(actor->publishActor(ready)==OMNI_USD_LIVE_OK);
    observer.wait(id,OMNI_USD_LIVE_STATE_READY);
    auto current=f.query(admission);
    for(const auto state:{OMNI_USD_LIVE_STATE_CLOSING,OMNI_USD_LIVE_STATE_CLOSED,
                          OMNI_USD_LIVE_STATE_FAILED}) {
        auto stale=value(state); stale.result=OMNI_USD_LIVE_OUTCOME_UNKNOWN;
        assert(creation->publishCreation(stale)==OMNI_USD_LIVE_OK);
        assert(f.query(admission).revision==current.revision);
    }
    assert(actor->failOpening(OMNI_USD_LIVE_MALFORMED_STREAM)==OMNI_USD_LIVE_OK);
    const auto failed=f.query(admission);
    assert(failed.subject_id==id && failed.state==OMNI_USD_LIVE_STATE_FAILED &&
           failed.result==OMNI_USD_LIVE_MALFORMED_STREAM && failed.operation==OMNI_USD_LIVE_OPERATION_OPEN);
    assert(failed.pending_write_count==3 && failed.pending_bytes==777 && failed.has_unconfirmed_edits);
    assert(failed.incident_id && !failed.grpc_status_present);
    const auto observed=observer.wait(id,OMNI_USD_LIVE_STATE_FAILED);
    assert(observed.pending_write_count==3 && observed.pending_bytes==777 && observed.incident_id==failed.incident_id);
    auto endpoint=record<OmniUsdLiveStatus>();
    assert(omniUsdLiveGetEndpointStatus(f.endpoint,&endpoint)==OMNI_USD_LIVE_OK && endpoint.failed_layers==1);
    auto stale=ready; stale.pending_write_count=0; stale.pending_bytes=0; stale.has_unconfirmed_edits=0;
    assert(actor->publishActor(stale)==OMNI_USD_LIVE_OK);
    current=f.query(admission);
    assert(current.result==failed.result && current.incident_id==failed.incident_id &&
           current.pending_write_count==3 && current.pending_bytes==777 && current.has_unconfirmed_edits);
    actor.reset(); // Actor retirement alone must not hide creation provenance.
    assert(f.query(admission).subject_id==id);
    creation.reset();
    auto missing=record<OmniUsdLiveStatus>();
    assert(omniUsdLiveGetLayerStatus(admission->canonicalUri().c_str(),&missing)==OMNI_USD_LIVE_NOT_FOUND);
    const auto closed=observer.wait(id,OMNI_USD_LIVE_STATE_CLOSED);
    assert(closed.result==failed.result && closed.incident_id==failed.incident_id && closed.pending_write_count==3);
    assert(omniUsdLiveGetEndpointStatus(f.endpoint,&endpoint)==OMNI_USD_LIVE_OK && !endpoint.failed_layers);
    assert(omniUsdLiveUnregisterCallback(callback,3000)==OMNI_USD_LIVE_OK);
}
void exactRuntimeAndReservationValidation() {
    Fixture f;
    auto admission=f.admission();
    auto status=LayerStatus::create(f.runtime,admission,true);
    status->validate(f.runtime,admission);
    auto second=f.admission("second.live");
    rejects(OMNI_USD_LIVE_INVALID_ARGUMENT,[&] { status->validate(f.runtime,second); });
    rejects(OMNI_USD_LIVE_INVALID_ARGUMENT,[&] { status->validate({},admission); });
    rejects(OMNI_USD_LIVE_INVALID_ARGUMENT,[&] { status->validate(f.runtime,{}); });
    // Same pointer under a different control block is not the retained token.
    auto borrowed=std::shared_ptr<LayerAdmission>(admission.get(),[](LayerAdmission*) {});
    rejects(OMNI_USD_LIVE_INVALID_ARGUMENT,[&] { status->validate(f.runtime,borrowed); });
    auto borrowedRuntime=std::shared_ptr<Runtime>(f.runtime.get(),[](Runtime*) {});
    rejects(OMNI_USD_LIVE_INVALID_ARGUMENT,[&] { status->validate(borrowedRuntime,admission); });
    auto other=std::make_shared<Runtime>(RuntimeOptions::copy(nullptr));
    rejects(OMNI_USD_LIVE_INVALID_ARGUMENT,[&] { status->validate(other,admission); });
    rejects(OMNI_USD_LIVE_INVALID_ARGUMENT,[&] { LayerStatus::create(other,admission,true); });
    other.reset();
    assert(status->handoff()==OMNI_USD_LIVE_BUSY);
    assert(status->publishCreation(value(),true)==OMNI_USD_LIVE_OK);
    assert(status->handoff()==OMNI_USD_LIVE_OK); // Rejections did not consume authority.
}
void unknownAndClosingCannotHandoff() {
    Fixture f;
    auto admission=f.admission();
    auto status=LayerStatus::create(f.runtime,admission,true);
    auto unknown=value(OMNI_USD_LIVE_STATE_FAILED);
    unknown.result=OMNI_USD_LIVE_OUTCOME_UNKNOWN; unknown.grpc_status_present=1; unknown.grpc_status=14;
    unknown.pending_write_count=1; unknown.pending_bytes=321; unknown.has_unconfirmed_edits=1;
    assert(status->publishCreation(unknown)==OMNI_USD_LIVE_OK);
    const auto failed=f.query(admission);
    assert(failed.result==OMNI_USD_LIVE_OUTCOME_UNKNOWN && failed.incident_id);
    assert(status->handoff()==OMNI_USD_LIVE_OUTCOME_UNKNOWN);
    assert(status->publishCreation(value(),true)==OMNI_USD_LIVE_OK);
    assert(status->handoff()==OMNI_USD_LIVE_OUTCOME_UNKNOWN);
    assert(status->failOpening(OMNI_USD_LIVE_INTERNAL_ERROR)==OMNI_USD_LIVE_OK);
    auto current=f.query(admission);
    assert(current.result==OMNI_USD_LIVE_OUTCOME_UNKNOWN && current.incident_id==failed.incident_id &&
           current.grpc_status_present && current.grpc_status==14 && current.pending_write_count==1);
    current.state=OMNI_USD_LIVE_STATE_CLOSING;
    assert(status->publishCreation(current)==OMNI_USD_LIVE_OK);
    assert(f.query(admission).result==OMNI_USD_LIVE_OUTCOME_UNKNOWN);
    auto cleanAdmission=f.admission("clean.live");
    auto clean=LayerStatus::create(f.runtime,cleanAdmission,true);
    assert(clean->publishCreation(value(),true)==OMNI_USD_LIVE_OK);
    assert(clean->publishCreation(value(OMNI_USD_LIVE_STATE_CLOSING))==OMNI_USD_LIVE_OK);
    assert(clean->handoff()==OMNI_USD_LIVE_CANCELLED);
}
void prematureActorOpenPreservesAdmission() {
    Fixture f;
    auto admission=f.admission();
    auto status=LayerStatus::create(f.runtime,admission,true);
    const auto before=f.query(admission);
    LayerActor::Options options;
    options.session=admission->session(); options.layer=admission->layer();
    options.admission=admission; options.status=status;
    rejects(OMNI_USD_LIVE_BUSY,[&] { LayerActor::open(f.runtime,f.endpoint,options,{}); });
    const auto after=f.query(admission);
    assert(after.subject_id==before.subject_id && after.revision==before.revision &&
           after.state==OMNI_USD_LIVE_STATE_OPENING && after.result==OMNI_USD_LIVE_OK);
    // The eligibility failure must precede the one-shot canonical claim. A
    // subsequent explicit claim is the independent witness, without issuing
    // transport or relying on the invalid callback configuration reaching ctor.
    f.runtime->layerAdmissions->claim(admission,f.endpoint,admission->session(),admission->layer());
    f.noRpcOrDiagnosticHistory();
}
void handoffDoesNotConsumeAnotherSubjectSlot() {
    Fixture f(2); // Endpoint plus exactly one layer subject.
    auto admission=f.admission();
    auto creation=LayerStatus::create(f.runtime,admission,true);
    const auto id=creation->id();
    auto next=f.admission("next.live");
    rejects(OMNI_USD_LIVE_RESOURCE_EXHAUSTED,[&] { LayerStatus::create(f.runtime,next,true); });
    assert(creation->publishCreation(value(),true)==OMNI_USD_LIVE_OK);
    auto actor=creation;
    assert(actor->handoff()==OMNI_USD_LIVE_OK);
    assert(actor->publishActor(value(OMNI_USD_LIVE_STATE_READY))==OMNI_USD_LIVE_OK);
    assert(f.query(admission).subject_id==id);
    creation.reset();
    rejects(OMNI_USD_LIVE_RESOURCE_EXHAUSTED,[&] { LayerStatus::create(f.runtime,next,true); });
    actor.reset();
    auto fresh=LayerStatus::create(f.runtime,next); // Ordinary non-create path also shares owner API.
    assert(fresh->id()!=id && fresh->handoff()==OMNI_USD_LIVE_OK);
    assert(fresh->publishActor(value(OMNI_USD_LIVE_STATE_READY))==OMNI_USD_LIVE_OK);
    assert(f.query(next).state==OMNI_USD_LIVE_STATE_READY);
}
// Allocation failure anywhere in LayerActor::open, before its first RPC,
// leaves no admission, status subject, endpoint dependent or RPC behind
// (design section 8). The countdown is sticky, so failure paths must not
// allocate. The endpoint has no credential, so the one success parks in
// AUTH_WAIT and issues nothing.
void actorOpenAllocationSweep() {
    size_t failures=0;
    for(size_t permit=0;;++permit) {
        assert(permit<4096);
        Fixture f(8,"");
        auto admission=f.admission();
        LayerActor::Options options;
        options.session=admission->session(); options.layer=admission->layer(); options.admission=admission;
        options.object=1; options.resume=1; options.initialState=std::make_shared<int>(0);
        LayerActor::Callbacks callbacks;
        callbacks.decode=[](const LayerReadUpdate&) { return LayerActor::Decoded{}; };
        callbacks.apply=[](const LayerProgress::Work&, const LayerActor::Effect&) {};
        std::shared_ptr<LayerActor> actor; bool failed=false;
        liveTestFailAllocationsAfter(permit);
        try { actor=LayerActor::open(f.runtime,f.endpoint,std::move(options),std::move(callbacks)); }
        catch(const std::bad_alloc&) { failed=true; }
        catch(OmniUsdLiveResult result) { assert(result==OMNI_USD_LIVE_RESOURCE_EXHAUSTED); failed=true; }
        liveTestRestoreAllocations();
        if(!failed) {
            assert(actor && actor->close(3000)==OMNI_USD_LIVE_OK);
            actor.reset(); admission.reset();
            f.noRpcOrDiagnosticHistory();
            // The actor's scheduler drops its endpoint lease after close
            // returns; final unregister is BUSY until then.
            const auto until=std::chrono::steady_clock::now()+std::chrono::seconds(3);
            for(;;) {
                const auto result=omniUsdLiveUnregisterEndpoint(f.endpoint,0);
                if(result==OMNI_USD_LIVE_OK) break;
                assert(result==OMNI_USD_LIVE_BUSY && std::chrono::steady_clock::now()<until);
                std::this_thread::yield();
            }
            f.endpoint=0;
            break; // ~Fixture: the runtime shuts down.
        }
        ++failures;
        auto status=record<OmniUsdLiveStatus>();
        const auto found=omniUsdLiveGetLayerStatus(admission->canonicalUri().c_str(),&status);
        assert(found==OMNI_USD_LIVE_NOT_FOUND ||
               (status.state==OMNI_USD_LIVE_STATE_FAILED && status.result==OMNI_USD_LIVE_RESOURCE_EXHAUSTED));
        const auto uri=admission->canonicalUri();
        admission.reset();
        // A FAILED subject lasts only as long as the admission that holds it.
        assert(omniUsdLiveGetLayerStatus(uri.c_str(),&status)==OMNI_USD_LIVE_NOT_FOUND);
        (void)f.admission(); // A retained reservation would throw ALREADY_EXISTS.
        f.noRpcOrDiagnosticHistory();
    }
    assert(failures>=4);
    std::cout << "actor-open allocation sweep: " << failures << " failures\n";
}
}
void layerStatusTests() {
    abandonedOpeningConfirmationOrders();
    abandonmentPreservesFiniteDisposition();
    sharedSubjectAndFailureRetirement();
    exactRuntimeAndReservationValidation();
    unknownAndClosingCannotHandoff();
    prematureActorOpenPreservesAdmission();
    handoffDoesNotConsumeAnotherSubjectSlot();
    actorOpenAllocationSweep();
    std::cout << "layer status: shared creation/actor subject, exact handoff, failure retention and final retirement passed"
                 ", actor-open allocation failures release everything\n";
}
