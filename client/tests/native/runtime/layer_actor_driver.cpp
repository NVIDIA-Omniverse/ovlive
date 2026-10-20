// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "LiveLayerActor.h"
#include "LiveDrain.h"
#include "../support/test_tls.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <future>

using namespace omni::usd::live;
std::string publicDiagnostics();
namespace {
template <class T> T record()
{ T value{}; value.struct_size = sizeof(value); value.abi_version = OMNI_USD_LIVE_ABI_VERSION; return value; }
struct Seen {
    std::mutex mutex;
    std::condition_variable changed;
    LayerProgress::Snapshot state{};
    std::vector<std::pair<uint64_t, std::string>> applied;
    std::vector<LayerProgress::Snapshot> history;
    unsigned auth = 0;
    bool entered = false, release = false;
    template <class Predicate> void wait(Predicate predicate)
    {
        std::unique_lock<std::mutex> lock(mutex);
        assert(changed.wait_for(lock, std::chrono::seconds(6), predicate));
    }
};
// Counts closing worker passes that keep the actor running. A pass counted
// after close() proves the worker saw closing with the owner under test held.
struct ClosingPasses {
    std::mutex mutex;
    std::condition_variable changed;
    size_t count = 0;
    void install(LayerActor::Options& settings)
    {
        settings.closingRetainedContextForTest = this;
        settings.closingRetainedForTest = [](void* context) noexcept {
            auto& passes = *static_cast<ClosingPasses*>(context);
            std::lock_guard<std::mutex> lock(passes.mutex);
            ++passes.count; passes.changed.notify_all();
        };
    }
    size_t current() { std::lock_guard<std::mutex> lock(mutex); return count; }
    void waitBeyond(size_t seen)
    {
        std::unique_lock<std::mutex> lock(mutex);
        assert(changed.wait_for(lock, std::chrono::seconds(6), [&] { return count > seen; }));
    }
};
struct AsyncRetirement {
    std::mutex mutex;
    std::condition_variable changed;
    bool entered=false, release=false;
};
struct HeldAsyncCapture {
    std::shared_ptr<AsyncRetirement> gate;
    ~HeldAsyncCapture() {
        std::unique_lock<std::mutex> lock(gate->mutex);
        gate->entered=true; gate->changed.notify_all();
        assert(gate->changed.wait_for(lock,std::chrono::seconds(6),[&] { return gate->release; }));
    }
};
struct AsyncApplication {
    // Host tasks retire captures before their completion owner. This ordering
    // is the explicit async API contract, not reliance on lambda capture order.
    std::shared_ptr<LayerActor::ApplyCompletion> completion;
    std::shared_ptr<const LayerActor::Effect> effect;
    LayerProgress::Work work;
    std::unique_ptr<HeldAsyncCapture> capture;
};
struct StatusSeen {
    std::mutex mutex;
    std::condition_variable changed;
    struct Event { OmniUsdLiveEndpoint endpoint; std::string uri; OmniUsdLiveStatus status; };
    std::vector<Event> events;
    std::thread::id caller;
    template <class Predicate> void wait(Predicate predicate, std::chrono::seconds watchdog = std::chrono::seconds(3))
    {
        std::unique_lock<std::mutex> lock(mutex);
        assert(changed.wait_for(lock, watchdog, predicate));
    }
    void receive(const OmniUsdLiveStatusEvent& event)
    {
        assert(std::this_thread::get_id() != caller);
        assert(event.status && event.uri);
        const auto& status = *event.status;
        assert(status.struct_size == sizeof(status) && status.abi_version == OMNI_USD_LIVE_ABI_VERSION);
        assert(status.subject_id && status.revision && status.changed_at_ms);
        std::lock_guard<std::mutex> lock(mutex);
        assert(events.size() < 256);
        const auto previous = std::find_if(events.rbegin(), events.rend(), [&](const auto& value) {
            return value.status.subject_id == status.subject_id;
        });
        if (previous != events.rend()) assert(previous->status.revision < status.revision);
        events.push_back({event.endpoint, event.uri, status}); // Own borrowed callback data.
        changed.notify_all();
    }
};
bool failed(const LayerProgress::Snapshot& state)
{ return state.gate == LayerProgress::Gate::Failed; }

// A ledger refusal after the byte reservation releases it quietly, unless the
// endpoint refused another reservation while it was held.
struct ReleaseContention {
    std::shared_ptr<Runtime> runtime;
    OmniUsdLiveEndpoint endpoint=0;
    bool contend=false;
};
int actorReleaseNoticesDriver(int argc,char** argv) {
    assert(argc==7);
    auto options=record<OmniUsdLiveRuntimeOptions>();
    options.flags=OMNI_USD_LIVE_RUNTIME_DISABLE_DIAGNOSTIC_HISTORY;
    assert(omniUsdLiveInitialize(&options)==OMNI_USD_LIVE_OK);
    auto ep=record<OmniUsdLiveEndpointOptions>(); ep.authority=argv[1]; ep.access_token=argv[5];
    OmniUsdLiveEndpoint endpoint=0;
    assert(omniUsdLiveRegisterEndpoint(&ep,&endpoint)==OMNI_USD_LIVE_OK);
    auto runtime=getRuntime();
    ReleaseContention contention{runtime,endpoint,false};
    Seen seen;
    LayerActor::Callbacks callbacks;
    callbacks.decode=[](const LayerReadUpdate& update) {
        LayerActor::Decoded result; result.first=update.header.sequence_num(); return result;
    };
    callbacks.apply=[](const auto&,const auto&) {};
    callbacks.changed=[&](const auto& state) {
        std::lock_guard<std::mutex> lock(seen.mutex); seen.state=state; seen.changed.notify_all();
    };
    LayerActor::Options settings; settings.session=argv[6]; settings.layer="root.live";
    settings.object=settings.resume=1; settings.initialState=std::make_shared<const std::string>("opaque-base");
    settings.afterReservationForTest=[](void* context) {
        auto& state=*static_cast<ReleaseContention*>(context);
        if(state.contend) {
            // Another reservation on the endpoint is refused while this one is held.
            bool refused=false;
            try { PendingBytes other(state.runtime->acquireEndpoint(state.endpoint),UINT64_MAX,1); }
            catch(OmniUsdLiveResult result) { refused=result==OMNI_USD_LIVE_RESOURCE_EXHAUSTED; }
            assert(refused);
        }
        throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED); // The ledger refuses the admission.
    };
    settings.reservationContextForTest=&contention;
    // This layer's own host edit buffer, held by a transient deferral.
    std::atomic<unsigned> ownRetries{0};
    auto own=std::make_shared<BufferedEditCount>();
    own->setRetry([&ownRetries] { ++ownRetries; });
    (void)own->note(1,OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    settings.bufferedEdits=own;
    auto actor=LayerActor::open(runtime,endpoint,std::move(settings),std::move(callbacks));
    seen.wait([&] { return seen.state.gate==LayerProgress::Gate::Ready || failed(seen.state); });
    assert(!failed(seen.state));
    // A waiting layer on the same endpoint whose flush a transient cause deferred.
    std::atomic<unsigned> retries{0};
    auto watcher=std::make_shared<BufferedEditCount>();
    watcher->setRetry([&retries] { ++retries; });
    (void)watcher->note(1,OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    { const auto owner=runtime->findEndpoint(endpoint); std::lock_guard<std::mutex> lock(owner->mutex);
      owner->pendingByteWatchers.push_back(watcher); }
    const auto original=std::make_shared<const std::string>("refused-intent");
    auto refusedSubmit=[&] {
        bool refused=false;
        try { (void)actor->submit(*original,original); }
        catch(OmniUsdLiveResult result) { refused=result==OMNI_USD_LIVE_RESOURCE_EXHAUSTED; }
        assert(refused && actor->snapshot().localAdmitted==0);
    };
    refusedSubmit(); // No contention: a quiet release.
    const auto quietReleases=watcher->releases(); const unsigned quietRetries=retries;
    contention.contend=true;
    refusedSubmit(); // The refusal in between makes the release report.
    const auto contendedReleases=watcher->releases()-quietReleases; const unsigned contendedRetries=retries-quietRetries;
    assert(actor->close(5000)==OMNI_USD_LIVE_OK); actor.reset();
    // A closed layer stops watching the endpoint: a later release requests no
    // flush for its buffer, which the host binding still owns.
    const auto ownReleases=own->releases(); const unsigned ownRetriesAtClose=ownRetries;
    { PendingBytes later(runtime->acquireEndpoint(endpoint),64,256); }
    assert(own->releases()==ownReleases && ownRetries==ownRetriesAtClose);
    { const auto owner=runtime->findEndpoint(endpoint); std::lock_guard<std::mutex> lock(owner->mutex);
      for(const auto& entry:owner->pendingByteWatchers) assert(entry.lock()!=own); }
    const auto until=std::chrono::steady_clock::now()+std::chrono::seconds(5);
    for(;;) {
        const auto result=omniUsdLiveUnregisterEndpoint(endpoint,0);
        if(result==OMNI_USD_LIVE_OK) break;
        assert(result==OMNI_USD_LIVE_BUSY && std::chrono::steady_clock::now()<until); std::this_thread::yield();
    }
    watcher.reset(); contention.runtime.reset();
    runtime.reset(); assert(omniUsdLiveShutdown(5000)==OMNI_USD_LIVE_OK);
    std::cout << "{\"result\":0,\"quiet_releases\":" << quietReleases << ",\"quiet_retries\":" << quietRetries
              << ",\"contended_releases\":" << contendedReleases << ",\"contended_retries\":" << contendedRetries << '}' << std::endl;
    return 0;
}
// An admitted intent's issuance cannot be refused for endpoint pending bytes.
// The first intent is acknowledged, its write retired and its application
// held; a second intent that its issuance could not fit beside the first is
// refused at admission (transient) instead of being admitted and failing the
// layer at issue. Once the first retires, the second is admitted and issued.
int actorIssueReservationDriver(int argc,char** argv) {
    assert(argc==7);
    auto options=record<OmniUsdLiveRuntimeOptions>();
    options.flags=OMNI_USD_LIVE_RUNTIME_DISABLE_DIAGNOSTIC_HISTORY;
    // A 4160-byte standing read reservation plus one admitted 1000-byte
    // intent fits; a second one's admission and issuance together do not.
    options.max_update_bytes=4096; options.max_pending_bytes_per_endpoint=7000;
    assert(omniUsdLiveInitialize(&options)==OMNI_USD_LIVE_OK);
    auto ep=record<OmniUsdLiveEndpointOptions>(); ep.authority=argv[1]; ep.access_token=argv[5];
    OmniUsdLiveEndpoint endpoint=0;
    assert(omniUsdLiveRegisterEndpoint(&ep,&endpoint)==OMNI_USD_LIVE_OK);
    auto runtime=getRuntime();
    auto retained=runtime->findEndpoint(endpoint);
    auto pending=[&] { std::lock_guard<std::mutex> lock(retained->mutex); return retained->pendingTransportBytes; };
    auto until=[&](auto predicate) {
        // Watchdog only: every awaited value is established by the steps above it.
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
        while(!predicate()) { assert(std::chrono::steady_clock::now()<deadline); std::this_thread::yield(); }
    };
    Seen seen;
    std::shared_ptr<LayerActor::ApplyCompletion> held;
    unsigned localApplications=0;
    LayerActor::Callbacks callbacks;
    callbacks.decode=[](const LayerReadUpdate& update) {
        LayerActor::Decoded result; result.first=update.header.sequence_num(); return result;
    };
    callbacks.applyAsync=[&](LayerProgress::Work,std::shared_ptr<const LayerActor::Effect> effect,
                             std::shared_ptr<LayerActor::ApplyCompletion> completion) {
        assert(effect && !effect->incoming && effect->intent);
        {
            std::lock_guard<std::mutex> lock(seen.mutex);
            if(++localApplications==1) { held=std::move(completion); seen.entered=true; seen.changed.notify_all(); return; }
        }
        completion->run([] {});
    };
    callbacks.changed=[&](const auto& state) {
        std::lock_guard<std::mutex> lock(seen.mutex); seen.state=state; seen.changed.notify_all();
    };
    LayerActor::Options settings; settings.session=argv[6]; settings.layer="root.live";
    settings.object=settings.resume=1; settings.initialState=std::make_shared<const std::string>("opaque-base");
    auto actor=LayerActor::open(runtime,endpoint,std::move(settings),std::move(callbacks));
    seen.wait([&] { return seen.state.gate==LayerProgress::Gate::Ready || failed(seen.state); });
    assert(!failed(seen.state));
    const uint64_t read=layerReadFrameLimit(options.max_update_bytes);
    // A buffer fits alone only beside the layer's own standing read: find the
    // largest one that does, and check the next would fit without that read.
    size_t largest=0;
    while(largest<options.max_update_bytes && actor->updateCanFit(largest+1)) ++largest;
    assert(actor->updateCanFit(largest) && read+actor->admissionCharge(largest)<=options.max_pending_bytes_per_endpoint);
    const bool fitIncludesRead=!actor->updateCanFit(largest+1) &&
        actor->admissionCharge(largest+1)<=options.max_pending_bytes_per_endpoint;
    until([&] { return pending()==read; }); // Only the standing read reservation.
    const auto first=std::make_shared<const std::string>(1000,'a');
    (void)actor->submit(*first,first);
    // Acknowledged, application held, and its write's buffers retired.
    seen.wait([&] { return seen.entered && seen.state.issued==0; });
    until([&] { return pending()==read+actor->admissionCharge(first->size()); });
    const auto second=std::make_shared<const std::string>(1000,'b');
    bool refused=false;
    try { (void)actor->submit(*second,second); }
    catch(OmniUsdLiveResult result) { refused=result==OMNI_USD_LIVE_RESOURCE_EXHAUSTED; }
    if(!refused) {
        // The second intent was admitted: record what its issuance did.
        seen.wait([&] { return failed(seen.state) || seen.state.issued!=0; });
    }
    const auto afterRefusal=actor->snapshot();
    bool issued=!failed(afterRefusal);
    if(refused && issued) {
        { std::lock_guard<std::mutex> lock(seen.mutex); held->run([] {}); held.reset(); }
        seen.wait([&] { return seen.state.localRetired==1 || failed(seen.state); });
        until([&] { return pending()==read; });
        (void)actor->submit(*second,second);
        seen.wait([&] { return seen.state.localRetired==2 || failed(seen.state); });
        issued=!failed(seen.state);
    } else {
        std::lock_guard<std::mutex> lock(seen.mutex); if(held) held->run([] {}); held.reset();
    }
    const auto final=actor->snapshot();
    assert(actor->close(5000)==OMNI_USD_LIVE_OK); actor.reset();
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
    for(;;) {
        const auto result=omniUsdLiveUnregisterEndpoint(endpoint,0);
        if(result==OMNI_USD_LIVE_OK) break;
        assert(result==OMNI_USD_LIVE_BUSY && std::chrono::steady_clock::now()<deadline); std::this_thread::yield();
    }
    retained.reset(); runtime.reset(); assert(omniUsdLiveShutdown(5000)==OMNI_USD_LIVE_OK);
    std::cout << "{\"result\":0,\"refused\":" << (refused?"true":"false")
              << ",\"failed_after_refusal\":" << (failed(afterRefusal)?"true":"false")
              << ",\"failure\":" << afterRefusal.result << ",\"issued\":" << (issued?"true":"false")
              << ",\"retired\":" << final.localRetired << ",\"fit_includes_read\":" << (fitIncludesRead?"true":"false")
              << '}' << std::endl;
    return 0;
}
// A host record notes a nonzero buffered count and requests the actor wake;
// the host flush then empties the buffer and issues nothing before the actor
// publishes, so the publication reads a count of zero and every other status
// field is unchanged. The layer still publishes a newer revision with a zero
// count, which a host waiting after its record observes.
int actorBufferEmptiedDriver(int argc,char** argv) {
    assert(argc==7);
    auto options=record<OmniUsdLiveRuntimeOptions>();
    options.flags=OMNI_USD_LIVE_RUNTIME_DISABLE_DIAGNOSTIC_HISTORY;
    assert(omniUsdLiveInitialize(&options)==OMNI_USD_LIVE_OK);
    auto ep=record<OmniUsdLiveEndpointOptions>(); ep.authority=argv[1]; ep.access_token=argv[5];
    OmniUsdLiveEndpoint endpoint=0;
    assert(omniUsdLiveRegisterEndpoint(&ep,&endpoint)==OMNI_USD_LIVE_OK);
    auto runtime=getRuntime();
    StatusSeen statuses; statuses.caller=std::this_thread::get_id();
    OmniUsdLiveCallbackHandle statusObserver=0;
    assert(omniUsdLiveRegisterStatusCallback(0,&statuses,
        [](void* user,const OmniUsdLiveStatusEvent* event) {
            assert(event); static_cast<StatusSeen*>(user)->receive(*event);
        },&statusObserver)==OMNI_USD_LIVE_OK);
    Seen seen;
    LayerActor::Callbacks callbacks;
    callbacks.decode=[](const LayerReadUpdate& update) {
        LayerActor::Decoded result; result.first=update.header.sequence_num(); return result;
    };
    callbacks.apply=[](const auto&,const auto&) {};
    callbacks.changed=[&](const auto& state) {
        std::lock_guard<std::mutex> lock(seen.mutex); seen.state=state; seen.changed.notify_all();
    };
    LayerActor::Options settings; settings.session=argv[6]; settings.layer="root.live";
    settings.object=settings.resume=1; settings.initialState=std::make_shared<const std::string>("opaque-base");
    auto buffered=std::make_shared<BufferedEditCount>();
    settings.bufferedEdits=buffered;
    auto actor=LayerActor::open(runtime,endpoint,std::move(settings),std::move(callbacks));
    const auto subject=actor->subjectId();
    seen.wait([&] { return seen.state.gate==LayerProgress::Gate::Ready || failed(seen.state); });
    assert(!failed(seen.state));
    // Under statuses.mutex, as StatusSeen::wait calls its predicate.
    auto latest=[&](OmniUsdLiveStatus& status) {
        for(auto event=statuses.events.rbegin(); event!=statuses.events.rend(); ++event)
            if(event->status.subject_id==subject) { status=event->status; return true; }
        return false;
    };
    // The layer's settled READY status, delivered before the record.
    statuses.wait([&] {
        return std::any_of(statuses.events.begin(),statuses.events.end(),[&](const auto& event) {
            return event.status.subject_id==subject && event.status.state==OMNI_USD_LIVE_STATE_READY;
        });
    });
    auto before=record<OmniUsdLiveStatus>();
    assert(runtime->statuses->snapshot(subject,&before)==OMNI_USD_LIVE_OK);
    assert(before.state==OMNI_USD_LIVE_STATE_READY && before.buffered_edit_count==0 && !before.pending_write_count);
    assert(buffered->note(1));   // The record requests the actor wake.
    assert(!buffered->note(0));  // The flush empties the buffer first; its wake is coalesced.
    actor->noteHostEdits();      // The record's wake reaches the actor.
    std::cout << "awaiting the emptied-buffer status after revision " << before.revision << std::endl;
    auto after=record<OmniUsdLiveStatus>();
    statuses.wait([&] { return latest(after) && after.revision>before.revision; });
    assert(after.state==OMNI_USD_LIVE_STATE_READY && after.result==OMNI_USD_LIVE_OK && after.buffered_edit_count==0 &&
           !after.pending_write_count && !after.has_unconfirmed_edits);
    const auto admitted=actor->snapshot().localAdmitted;
    assert(omniUsdLiveUnregisterCallback(statusObserver,5000)==OMNI_USD_LIVE_OK);
    assert(actor->close(5000)==OMNI_USD_LIVE_OK); actor.reset();
    const auto until=std::chrono::steady_clock::now()+std::chrono::seconds(5);
    for(;;) {
        const auto result=omniUsdLiveUnregisterEndpoint(endpoint,0);
        if(result==OMNI_USD_LIVE_OK) break;
        assert(result==OMNI_USD_LIVE_BUSY && std::chrono::steady_clock::now()<until); std::this_thread::yield();
    }
    buffered.reset();
    runtime.reset(); assert(omniUsdLiveShutdown(5000)==OMNI_USD_LIVE_OK);
    std::cout << "{\"result\":0,\"advanced\":" << (after.revision>before.revision?"true":"false")
              << ",\"buffered\":" << after.buffered_edit_count << ",\"admitted\":" << admitted << '}' << std::endl;
    return 0;
}
int actorDropWithoutCloseDriver(int argc,char** argv) {
    assert(argc==7);
    auto options=record<OmniUsdLiveRuntimeOptions>();
    options.flags=OMNI_USD_LIVE_RUNTIME_DISABLE_DIAGNOSTIC_HISTORY;
    assert(omniUsdLiveInitialize(&options)==OMNI_USD_LIVE_OK);
    auto ep=record<OmniUsdLiveEndpointOptions>(); ep.authority=argv[1]; ep.access_token=argv[5];
    OmniUsdLiveEndpoint endpoint=0;
    assert(omniUsdLiveRegisterEndpoint(&ep,&endpoint)==OMNI_USD_LIVE_OK);
    auto runtime=getRuntime();
    StatusSeen statuses; statuses.caller=std::this_thread::get_id();
    OmniUsdLiveCallbackHandle statusObserver=0;
    assert(omniUsdLiveRegisterStatusCallback(0,&statuses,
        [](void* user,const OmniUsdLiveStatusEvent* event) {
            assert(event); static_cast<StatusSeen*>(user)->receive(*event);
        },&statusObserver)==OMNI_USD_LIVE_OK);
    LayerActor::Callbacks callbacks;
    callbacks.decode=[](const LayerReadUpdate& update) {
        LayerActor::Decoded result; result.first=update.header.sequence_num(); return result;
    };
    callbacks.apply=[](const auto&,const auto&) {};
    LayerActor::Options settings; settings.session=argv[6]; settings.layer="root.live";
    settings.object=settings.resume=1; settings.initialState=std::make_shared<const std::string>("opaque-base");
    const auto uri=makeLayerUri(makeSessionUri(argv[1],argv[6]),"root.live");
    auto actor=LayerActor::open(runtime,endpoint,std::move(settings),std::move(callbacks));
    const auto subject=actor->subjectId();
    statuses.wait([&] {
        return std::any_of(statuses.events.begin(),statuses.events.end(),[&](const auto& event) {
            return event.status.subject_id==subject && event.status.state==OMNI_USD_LIVE_STATE_READY;
        });
    });
    // The owner's last release, with no close(), retires the actor.
    actor.reset();
    std::cout << "actor-dropped" << std::endl;
    std::string command; assert(std::getline(std::cin,command) && command=="continue");
    const auto until=std::chrono::steady_clock::now()+std::chrono::seconds(3);
    for(;;) {
        const auto result=omniUsdLiveUnregisterEndpoint(endpoint,0);
        if(result==OMNI_USD_LIVE_OK) break;
        assert(result==OMNI_USD_LIVE_BUSY && std::chrono::steady_clock::now()<until); std::this_thread::yield();
    }
    statuses.wait([&] {
        return std::any_of(statuses.events.begin(),statuses.events.end(),[&](const auto& event) {
            return event.status.subject_id==subject && event.status.state==OMNI_USD_LIVE_STATE_CLOSED;
        });
    });
    auto detached=record<OmniUsdLiveStatus>();
    assert(omniUsdLiveGetLayerStatus(uri.c_str(),&detached)==OMNI_USD_LIVE_NOT_FOUND);
    assert(omniUsdLiveUnregisterCallback(statusObserver,3000)==OMNI_USD_LIVE_OK);
    runtime.reset(); assert(omniUsdLiveShutdown(3000)==OMNI_USD_LIVE_OK);
    std::cout << "{\"result\":0,\"closed\":true}" << std::endl;
    return 0;
}
int actorHostFaultDriver(int argc,char** argv) {
    assert(argc==7);
    const std::string mode=argv[2];
    const bool queued=mode=="actor-host-fault-queued";
    const bool opening=mode=="actor-host-fault-opening";
    const bool sync=mode=="actor-host-fault-sync";
    const bool retained=queued || sync;
    const bool holdWorker=!queued && !sync;
    auto options=record<OmniUsdLiveRuntimeOptions>();
    options.flags=OMNI_USD_LIVE_RUNTIME_DISABLE_DIAGNOSTIC_HISTORY;
    assert(omniUsdLiveInitialize(&options)==OMNI_USD_LIVE_OK);
    auto ep=record<OmniUsdLiveEndpointOptions>(); ep.authority=argv[1]; ep.access_token=argv[5];
    OmniUsdLiveEndpoint endpoint=0;
    assert(omniUsdLiveRegisterEndpoint(&ep,&endpoint)==OMNI_USD_LIVE_OK);
    auto runtime=getRuntime();
    auto fault=std::make_shared<std::atomic<OmniUsdLiveResult>>(OMNI_USD_LIVE_OK);
    Seen seen, admission;
    StatusSeen statuses; statuses.caller=std::this_thread::get_id();
    OmniUsdLiveCallbackHandle statusObserver=0;
    assert(omniUsdLiveRegisterStatusCallback(0,&statuses,
        [](void* user,const OmniUsdLiveStatusEvent* event) {
            assert(event); static_cast<StatusSeen*>(user)->receive(*event);
        },&statusObserver)==OMNI_USD_LIVE_OK);
    std::shared_ptr<LayerActor::ApplyCompletion> held;
    LayerActor::Callbacks callbacks;
    callbacks.decode=[&](const LayerReadUpdate& update) {
        // Fault after the worker entry check and before synchronous application.
        if(sync) fault->store(OMNI_USD_LIVE_INTERNAL_ERROR,std::memory_order_release);
        LayerActor::Decoded result; result.first=update.header.sequence_num(); return result;
    };
    callbacks.apply=[](const auto&,const auto&) { assert(false && "host-fault fixture unexpectedly applied data"); };
    callbacks.changed=[&](const auto& state) {
        std::unique_lock<std::mutex> lock(seen.mutex); seen.state=state; seen.changed.notify_all();
        if(holdWorker && state.gate==LayerProgress::Gate::Ready && !seen.entered) {
            // Isolate the public admission/opening checks: an independently
            // waking worker must not terminalize the actor before they run.
            seen.entered=true; seen.changed.notify_all();
            assert(seen.changed.wait_for(lock,std::chrono::seconds(6),[&] { return seen.release; }));
        }
    };
    if(queued) callbacks.localAsync=[&](auto,auto,std::shared_ptr<LayerActor::ApplyCompletion> completion) {
        std::unique_lock<std::mutex> lock(seen.mutex); held=std::move(completion);
        seen.entered=true; seen.changed.notify_all();
        // Keep worker observation from masking ApplyCompletion::run's gate.
        assert(seen.changed.wait_for(lock,std::chrono::seconds(6),[&] { return seen.release; }));
    };
    LayerActor::Options settings; settings.session=argv[6]; settings.layer="root.live";
    settings.object=settings.resume=1; settings.initialState=std::make_shared<const std::string>("opaque-base");
    settings.hostFailure=fault; settings.holdOpening=opening;
    if(sync) { settings.resume=0; settings.initialState.reset(); }
    if(!queued && !opening && !sync) {
        settings.beforeAdmissionForTest=[](void* context) noexcept {
            auto& gate=*static_cast<Seen*>(context); std::unique_lock<std::mutex> lock(gate.mutex);
            gate.entered=true; gate.changed.notify_all();
            assert(gate.changed.wait_for(lock,std::chrono::seconds(6),[&] { return gate.release; }));
        };
        settings.admissionContextForTest=&admission;
    }
    auto actor=LayerActor::open(runtime,endpoint,std::move(settings),std::move(callbacks));
    seen.wait([&] { return (!sync && seen.state.gate==LayerProgress::Gate::Ready &&
                                   (!holdWorker || seen.entered)) || failed(seen.state); });
    if(!sync) assert(!failed(actor->snapshot()) && actor->pollSave()==OMNI_USD_LIVE_OK);
    const auto original=std::make_shared<const std::string>("preserved-host-fault-intent");
    uint64_t id=0;
    // A waiting layer on the same endpoint whose flush a transient cause deferred.
    std::atomic<unsigned> retries{0};
    auto watcher=std::make_shared<BufferedEditCount>();
    watcher->setRetry([&retries] { ++retries; });
    if(opening) {
        (void)watcher->note(1,OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
        { const auto owner=runtime->findEndpoint(endpoint); std::lock_guard<std::mutex> lock(owner->mutex);
          owner->pendingByteWatchers.push_back(watcher); }
        // Admission is BUSY until opening completes. The rejection reserves no
        // endpoint bytes, so it reports no release to waiting layers.
        const auto admittedBefore=runtime->diagnostics->snapshot(0,0).metrics.pendingBytes.reservations.admitted;
        bool busy=false;
        try { (void)actor->submit(*original,original); } catch(OmniUsdLiveResult result) { busy=result==OMNI_USD_LIVE_BUSY; }
        assert(busy && actor->snapshot().localAdmitted==0);
        assert(runtime->diagnostics->snapshot(0,0).metrics.pendingBytes.reservations.admitted==admittedBefore);
        assert(watcher->releases()==0 && retries==0);
        fault->store(OMNI_USD_LIVE_INTERNAL_ERROR,std::memory_order_release);
        assert(actor->completeOpening()==OMNI_USD_LIVE_INTERNAL_ERROR);
    } else if(sync) {
        assert(failed(actor->snapshot()));
    } else if(queued) {
        id=actor->submit(*original,original); assert(id);
        seen.wait([&] { return seen.entered; });
        fault->store(OMNI_USD_LIVE_INTERNAL_ERROR,std::memory_order_release);
        std::shared_ptr<LayerActor::ApplyCompletion> completion;
        { std::lock_guard<std::mutex> lock(seen.mutex); completion=std::move(held); }
        bool ran=false; completion->run([&] { ran=true; }); completion.reset();
        assert(!ran);
        { std::lock_guard<std::mutex> lock(seen.mutex); seen.release=true; seen.changed.notify_all(); }
        assert(actor->save(5000)==OMNI_USD_LIVE_INTERNAL_ERROR);
    } else {
        auto submission=std::async(std::launch::async,[&] {
            try { (void)actor->submit(*original,original); return OmniUsdLiveResult(OMNI_USD_LIVE_OK); }
            catch(OmniUsdLiveResult result) { return result; }
        });
        admission.wait([&] { return admission.entered; });
        fault->store(OMNI_USD_LIVE_INTERNAL_ERROR,std::memory_order_release);
        { std::lock_guard<std::mutex> lock(admission.mutex); admission.release=true; admission.changed.notify_all(); }
        assert(submission.get()==OMNI_USD_LIVE_INTERNAL_ERROR);
        assert(actor->snapshot().localAdmitted==0);
    }
    if(holdWorker) {
        std::lock_guard<std::mutex> lock(seen.mutex); seen.release=true; seen.changed.notify_all();
    }
    assert(actor->authoringResult()==OMNI_USD_LIVE_INTERNAL_ERROR && actor->pollSave()==OMNI_USD_LIVE_INTERNAL_ERROR);
    seen.wait([&] { return failed(seen.state); });
    statuses.wait([&] {
        return std::any_of(statuses.events.begin(),statuses.events.end(),[&](const auto& event) {
            return event.status.subject_id==actor->subjectId() &&
                   event.status.state==OMNI_USD_LIVE_STATE_FAILED &&
                   event.status.result==OMNI_USD_LIVE_INTERNAL_ERROR;
        });
    });
    assert(omniUsdLiveUnregisterCallback(statusObserver,5000)==OMNI_USD_LIVE_OK);
    if(opening) {
        // Status callbacks may coalesce; aggregate transition counters cannot
        // hide a transient READY followed immediately by terminal failure.
        const auto readiness=runtime->diagnostics->snapshot(0,0).metrics.readiness[1];
        assert(readiness.initialByResolution[size_t(ReadinessResolution::Ready)].count==0 &&
               readiness.initialByResolution[size_t(ReadinessResolution::Failed)].count==1);
        for(const auto& event:statuses.events)
            if(event.status.subject_id==actor->subjectId()) assert(event.status.state!=OMNI_USD_LIVE_STATE_READY);
    }
    // Close waits for queued and running application ownership and keeps the
    // terminal progress, retained intents and the failure's status fields.
    assert(actor->close(5000)==OMNI_USD_LIVE_OK);
    {
        const auto progress=actor->snapshot();
        const auto work=actor->retained();
        assert(progress.result==OMNI_USD_LIVE_INTERNAL_ERROR && progress.applied==uint64_t(!sync) &&
               progress.issued==0 && work.size()==size_t(retained));
        auto status=record<OmniUsdLiveStatus>();
        assert(runtime->statuses->snapshot(actor->subjectId(),&status)==OMNI_USD_LIVE_OK);
        assert(status.result==OMNI_USD_LIVE_INTERNAL_ERROR && status.state==OMNI_USD_LIVE_STATE_CLOSED);
        // A host failure observed at an authoring boundary reports an UPDATE cause.
        if(!sync) assert(status.operation==OMNI_USD_LIVE_OPERATION_UPDATE);
        if(queued) {
            assert(work.front().id==id && work.front().disposition==LayerProgress::Disposition::Queued);
            const auto effect=std::static_pointer_cast<const LayerActor::Effect>(work.front().owner);
            assert(effect->intent==original && effect->authored==*original);
        }
        if(sync) {
            assert(work.front().disposition==LayerProgress::Disposition::Peer);
            const auto effect=std::static_pointer_cast<const LayerActor::Effect>(work.front().owner);
            assert(effect->incoming && effect->incoming->bytes=="opaque-base");
        }
    }
    actor.reset();
    const auto until=std::chrono::steady_clock::now()+std::chrono::seconds(5);
    for(;;) {
        const auto result=omniUsdLiveUnregisterEndpoint(endpoint,0);
        if(result==OMNI_USD_LIVE_OK) break;
        assert(result==OMNI_USD_LIVE_BUSY && std::chrono::steady_clock::now()<until); std::this_thread::yield();
    }
    watcher.reset();
    runtime.reset(); assert(omniUsdLiveShutdown(5000)==OMNI_USD_LIVE_OK);
    std::cout << "{\"result\":20,\"retained\":" << unsigned(retained) << ",\"issued\":0}" << std::endl;
    return 0;
}
int actorFloodDriver(int argc, char** argv)
{
    assert(argc == 8 && std::string(argv[2]) == "actor-flood-bytes");
    const unsigned peers = 9;
    auto options = record<OmniUsdLiveRuntimeOptions>();
    options.max_requests = 64;
    options.max_update_bytes = 1024;
    // A 1088-byte standing read, three local intents of 7 bytes each
    // reserving about 680 bytes (retained intent plus issuance), and nine
    // held 1024-byte peer frames fit; a tenth frame does not. Nine peers stay
    // below the read's high-water mark of 56.
    options.max_pending_bytes_per_endpoint = 12848;
    assert(omniUsdLiveInitialize(&options) == OMNI_USD_LIVE_OK);
    auto endpointOptions = record<OmniUsdLiveEndpointOptions>();
    endpointOptions.authority = argv[1]; endpointOptions.access_token = argv[5];
    OmniUsdLiveEndpoint endpoint = 0, other = 0;
    assert(omniUsdLiveRegisterEndpoint(&endpointOptions, &endpoint) == OMNI_USD_LIVE_OK);
    endpointOptions.authority = argv[7];
    assert(omniUsdLiveRegisterEndpoint(&endpointOptions, &other) == OMNI_USD_LIVE_OK && other != endpoint);
    auto jitter = record<OmniUsdLiveJitterParams>();
    assert(omniUsdLiveConfigureJitterReduction(endpoint, &jitter) == OMNI_USD_LIVE_OK);
    auto runtime = getRuntime();
    Seen seen;
    std::shared_ptr<AsyncApplication> held;
    LayerActor::Callbacks callbacks;
    callbacks.decode = [&](const LayerReadUpdate& update) {
        LayerActor::Decoded decoded; decoded.first = update.header.sequence_num(); return decoded;
    };
    callbacks.apply = [](const auto&, const auto&) { assert(false && "publication escaped the held local task"); };
    callbacks.localAsync = [&](LayerProgress::Work work, std::shared_ptr<const LayerActor::Effect> effect,
                               std::shared_ptr<LayerActor::ApplyCompletion> completion) {
        std::lock_guard<std::mutex> lock(seen.mutex);
        assert(!held && !seen.entered && work.disposition == LayerProgress::Disposition::Queued);
        held = std::make_shared<AsyncApplication>();
        held->completion = std::move(completion); held->effect = std::move(effect); held->work = std::move(work);
        seen.entered = true; seen.changed.notify_all();
    };
    callbacks.changed = [&](const LayerProgress::Snapshot& state) {
        std::lock_guard<std::mutex> lock(seen.mutex);
        seen.state = state; seen.changed.notify_all();
    };
    ClosingPasses closing;
    LayerActor::Options settings;
    settings.session = argv[6]; settings.layer = "root.live"; settings.object = settings.resume = 1;
    settings.initialState = std::make_shared<const std::string>("opaque-base");
    closing.install(settings);
    auto actor = LayerActor::open(runtime, endpoint, std::move(settings), std::move(callbacks));
    seen.wait([&] { return seen.state.gate == LayerProgress::Gate::Ready || failed(seen.state); });
    assert(!failed(actor->snapshot()));
    std::vector<uint64_t> ids;
    std::vector<std::shared_ptr<const std::string>> originals;
    for (unsigned i = 0; i < 3; ++i) {
        originals.push_back(std::make_shared<const std::string>("local-" + std::to_string(i)));
        ids.push_back(actor->submit(*originals.back(), originals.back()));
    }
    seen.wait([&] { return seen.entered; });
    const auto pingOther = [&] {
        rpc::PingRequest ping;
        auto request = submitUnary(*runtime, other, ping, UnaryCallback<rpc::PingResponse>{});
        assert(request->wait(3000) == OMNI_USD_LIVE_OK);
        OmniUsdLiveResult result = OMNI_USD_LIVE_INTERNAL_ERROR;
        assert(request->result(&result) == OMNI_USD_LIVE_OK && result == OMNI_USD_LIVE_OK);
        assert(runtime->requests->release(request->handle) == OMNI_USD_LIVE_OK);
    };
    const auto barrier = [](const std::string& marker) {
        std::cout << marker << std::endl;
        std::string command; assert(std::getline(std::cin, command) && command == "continue");
    };
    pingOther(); // A held host task must not occupy an operational worker.
    barrier("actor-flood-ready");
    for (unsigned sequence = 2; sequence <= peers + 1; ++sequence) {
        seen.wait([&] { return seen.state.covered >= sequence || failed(seen.state); });
        const auto state = actor->snapshot();
        assert(!failed(state) && state.covered == sequence && state.applied == 1);
        assert(state.retained == 3 + sequence - 1 && state.queued == 3 && !state.issued);
        const auto metrics = runtime->diagnostics->snapshot(0, 0).metrics;
        assert(!metrics.pendingBytes.currentBytes.high &&
               metrics.pendingBytes.currentBytes.low <= options.max_pending_bytes_per_endpoint);
        barrier("actor-flood-retained-" + std::to_string(sequence));
    }
    seen.wait([&] { return failed(seen.state); });
    assert(actor->snapshot().result == OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    barrier("actor-flood-failed"); // Reference service confirms cancellation before release.
    pingOther(); // Failure and retained owners must not stall another endpoint.
    std::shared_ptr<AsyncApplication> task;
    { std::lock_guard<std::mutex> lock(seen.mutex); task = std::move(held); }
    bool ran = false;
    task->completion->run([&] { ran = true; });
    assert(!ran); // Terminal exhaustion cancelled the queued host action.
    // The held application owner keeps the terminal actor from stopping.
    const auto passes = closing.current();
    assert(actor->close(0) == OMNI_USD_LIVE_TIMEOUT);
    closing.waitBeyond(passes);
    assert(actor->close(0) == OMNI_USD_LIVE_TIMEOUT);
    task.reset(); // Retire effect/work captures before the completion owner.
    // Close keeps progress and every retained intent.
    assert(actor->close(5000) == OMNI_USD_LIVE_OK);
    const auto state = actor->snapshot();
    assert(state.result == OMNI_USD_LIVE_RESOURCE_EXHAUSTED && state.applied == 1 && state.covered == peers + 1);
    assert(state.retained == peers + 3 && state.queued == 3 && !state.issued && !state.readStarted);
    assert(state.localAdmitted == 3 && !state.localRetired && state.pendingWrites == 3 && state.pendingBytes == 21);
    auto status = record<OmniUsdLiveStatus>();
    assert(runtime->statuses->snapshot(actor->subjectId(), &status) == OMNI_USD_LIVE_OK);
    assert(status.state == OMNI_USD_LIVE_STATE_CLOSED && status.result == OMNI_USD_LIVE_RESOURCE_EXHAUSTED &&
           status.incident_id && status.pending_write_count == 3 && status.pending_bytes == 21 &&
           status.has_unconfirmed_edits);
    unsigned locals = 0, retainedPeers = 0;
    for (const auto& work : actor->retained()) {
        const auto effect = std::static_pointer_cast<const LayerActor::Effect>(work.owner);
        assert(effect && !work.applied);
        if (work.disposition == LayerProgress::Disposition::Queued) {
            assert(locals < ids.size() && work.id == ids[locals] && work.localOrdinal == locals + 1);
            assert(effect->intent == originals[locals] && effect->bytes() == *originals[locals] && !effect->incoming);
            assert(!work.first && !work.last && work.retainedBytes == originals[locals]->size());
            ++locals;
        } else {
            ++retainedPeers;
            assert(work.disposition == LayerProgress::Disposition::Peer && work.first == retainedPeers + 1 &&
                   work.last == work.first && effect->incoming && !effect->intent);
            assert(effect->bytes() == std::string(1024, char('a' + work.first % 26)));
        }
    }
    assert(locals == 3 && retainedPeers == peers);
    const auto frozen = runtime->diagnostics->snapshot(0, 0).metrics;
    assert(!frozen.requestSlots[0].current && !frozen.requestSlots[1].current);
    assert(!frozen.pendingBytes.highWaterBytes.high &&
           frozen.pendingBytes.highWaterBytes.low <= options.max_pending_bytes_per_endpoint);
    assert(frozen.pendingBytes.reservations.rejected == 1);
    // CQ owners can briefly retain the receive-frame reservation beyond actor
    // close. Exact zero is checked after final endpoint-owner retirement.
    assert(frozen.pendingBytes.reservations.current >= peers + 3);
    const auto retainedBytes = frozen.pendingBytes.currentBytes.low;
    assert(retainedBytes >= peers * 1024 + 21);
    actor.reset(); originals.clear();
    const auto unregister = [](OmniUsdLiveEndpoint handle) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        for (;;) {
            // Closed actors can still have a scheduler owner; BUSY does not
            // consume endpoint ownership. Observe actual retirement, not delay.
            const auto result = omniUsdLiveUnregisterEndpoint(handle, 0);
            if (result == OMNI_USD_LIVE_OK) return;
            assert(result == OMNI_USD_LIVE_BUSY && std::chrono::steady_clock::now() < deadline);
            std::this_thread::yield();
        }
    };
    unregister(endpoint); unregister(other);
    const auto final = runtime->diagnostics->snapshot(0, 0).metrics;
    assert(!final.pendingBytes.currentBytes.high && !final.pendingBytes.currentBytes.low &&
           !final.pendingBytes.reservations.current && !final.requestSlots[0].current && !final.requestSlots[1].current);
    assert(final.pendingBytes.reservations.admitted == final.pendingBytes.reservations.released);
    runtime.reset(); assert(omniUsdLiveShutdown(5000) == OMNI_USD_LIVE_OK);
    std::cout << "{\"result\":13,\"local_count\":3,\"local_bytes\":21,\"peers\":" << peers
              << ",\"covered\":" << peers + 1 << ",\"applied\":1,\"retained_bytes\":" << retainedBytes
              << ",\"peak_bytes\":" << frozen.pendingBytes.highWaterBytes.low
              << ",\"budget\":" << options.max_pending_bytes_per_endpoint
              << ",\"byte_rejections\":" << frozen.pendingBytes.reservations.rejected
              << ",\"final_bytes\":0,\"final_slots\":0}" << std::endl;
    return 0;
}
// A host that publishes slower than the service delivers pauses the layer's
// read at its incoming high-water mark and resumes it at the low-water mark;
// the layer stays READY. Held: the first application, or the decoder (-ingress).
// -close closes the actor while its read is parked: cancellation finishes it.
// -service-end: the service ends the stream while the read is parked; the layer
// sees the end only once the read resumes, then reconnects from its coverage
// frontier. -terminal: a host failure while parked fails the layer, which
// keeps its retained work and cancels the parked read.
int actorReadPauseDriver(int argc, char** argv)
{
    assert(argc == 7 && std::string(argv[6]) == "scripted");
    const bool ingress = std::string(argv[2]) == "actor-read-pause-ingress";
    const bool closeParked = std::string(argv[2]) == "actor-read-pause-close";
    const bool serviceEnd = std::string(argv[2]) == "actor-read-pause-service-end";
    const bool terminal = std::string(argv[2]) == "actor-read-pause-terminal";
    auto fault = std::make_shared<std::atomic<OmniUsdLiveResult>>(OMNI_USD_LIVE_OK);
    constexpr uint64_t last = 37; // One caught-up header, then 36 peer updates.
    auto options = record<OmniUsdLiveRuntimeOptions>();
    options.max_requests = 32; // High-water mark 28, low-water mark 16.
    options.reconnect_initial_ms = 5; options.reconnect_maximum_ms = 20; // A service end reconnects promptly.
    options.max_update_bytes = 1024;
    options.max_pending_bytes_per_endpoint = 131072;
    assert(omniUsdLiveInitialize(&options) == OMNI_USD_LIVE_OK);
    auto endpointOptions = record<OmniUsdLiveEndpointOptions>();
    endpointOptions.authority = argv[1]; endpointOptions.access_token = argv[5];
    OmniUsdLiveEndpoint endpoint = 0;
    assert(omniUsdLiveRegisterEndpoint(&endpointOptions, &endpoint) == OMNI_USD_LIVE_OK);
    auto jitter = record<OmniUsdLiveJitterParams>();
    assert(omniUsdLiveConfigureJitterReduction(endpoint, &jitter) == OMNI_USD_LIVE_OK);
    auto runtime = getRuntime();
    Seen seen, decoder;
    std::shared_ptr<AsyncApplication> held;
    LayerActor::Callbacks callbacks;
    callbacks.decode = [&](const LayerReadUpdate& update) {
        if (ingress && update.header.sequence_num() == 2) {
            std::unique_lock<std::mutex> lock(decoder.mutex);
            decoder.entered = true; decoder.changed.notify_all();
            assert(decoder.changed.wait_for(lock, std::chrono::seconds(6), [&] { return decoder.release; }));
        }
        LayerActor::Decoded decoded; decoded.first = update.header.sequence_num(); return decoded;
    };
    callbacks.applyAsync = [&](LayerProgress::Work work, std::shared_ptr<const LayerActor::Effect> effect,
                               std::shared_ptr<LayerActor::ApplyCompletion> completion) {
        assert(work.disposition == LayerProgress::Disposition::Peer && effect->incoming);
        if (!ingress && work.last == 2) {
            std::lock_guard<std::mutex> lock(seen.mutex);
            assert(!held);
            held = std::make_shared<AsyncApplication>();
            held->completion = std::move(completion); held->effect = std::move(effect); held->work = std::move(work);
            seen.entered = true; seen.changed.notify_all();
            return;
        }
        completion->run([&] {
            std::lock_guard<std::mutex> lock(seen.mutex);
            seen.applied.emplace_back(work.last, effect->bytes());
            seen.changed.notify_all();
        });
    };
    callbacks.changed = [&](const LayerProgress::Snapshot& state) {
        std::lock_guard<std::mutex> lock(seen.mutex);
        seen.state = state; seen.changed.notify_all();
    };
    LayerActor::Options settings;
    settings.session = argv[6]; settings.layer = "root.live"; settings.object = settings.resume = 1;
    settings.initialState = std::make_shared<const std::string>("opaque-base");
    settings.hostFailure = fault;
    auto actor = LayerActor::open(runtime, endpoint, std::move(settings), std::move(callbacks));
    seen.wait([&] { return seen.state.gate == LayerProgress::Gate::Ready || failed(seen.state); });
    assert(!failed(actor->snapshot()));
    const auto barrier = [](const std::string& marker) {
        std::cout << marker << std::endl;
        std::string command; assert(std::getline(std::cin, command) && command == "continue");
    };
    const auto unregister = [&] {
        const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        for (;;) {
            // A closed actor can still have a scheduler owner; observe retirement.
            const auto result = omniUsdLiveUnregisterEndpoint(endpoint, 0);
            if (result == OMNI_USD_LIVE_OK) return;
            assert(result == OMNI_USD_LIVE_BUSY && std::chrono::steady_clock::now() < until);
            std::this_thread::yield();
        }
    };
    const auto statusState = [&] {
        auto status = record<OmniUsdLiveStatus>();
        assert(runtime->statuses->snapshot(actor->subjectId(), &status) == OMNI_USD_LIVE_OK);
        return status.state;
    };
    barrier("actor-pause-ready");
    // Failure-only deadline: the service has sent the whole burst.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (!actor->transportSnapshot().readPaused) {
        const auto state = actor->snapshot();
        if (failed(state) || std::chrono::steady_clock::now() >= deadline) {
            std::cout << "{\"unpaused_result\":" << unsigned(state.result) << ",\"covered\":" << state.covered
                      << "}" << std::endl;
            assert(false && "the read did not pause at its high-water mark");
        }
        std::this_thread::yield();
    }
    const auto paused = actor->snapshot();
    assert(!failed(paused) && paused.gate == LayerProgress::Gate::Ready && paused.covered < last);
    assert(statusState() == OMNI_USD_LIVE_STATE_READY);
    std::cout << "{\"paused_covered\":" << paused.covered << "}" << std::endl;
    barrier("actor-read-paused");
    if (closeParked) {
        assert(actor->transportSnapshot().readPaused);
        // The held application owner keeps the actor running; close cancels
        // the parked read, and only the owner's retirement remains.
        assert(actor->close(0) == OMNI_USD_LIVE_TIMEOUT);
        std::shared_ptr<AsyncApplication> task;
        { std::lock_guard<std::mutex> lock(seen.mutex); task = std::move(held); }
        assert(task);
        task->completion->run([] {});
        task.reset(); // Retire effect/work captures before the completion owner.
        assert(actor->close(5000) == OMNI_USD_LIVE_OK);
        const auto closed = actor->snapshot();
        // Coverage can pass the pause snapshot by frames already queued; the
        // parked read receives no more.
        assert(!closed.readStarted);
        assert(closed.covered >= paused.covered && closed.covered < last);
        actor.reset();
        const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        for (;;) {
            const auto result = omniUsdLiveUnregisterEndpoint(endpoint, 0);
            if (result == OMNI_USD_LIVE_OK) break;
            assert(result == OMNI_USD_LIVE_BUSY && std::chrono::steady_clock::now() < until);
            std::this_thread::yield();
        }
        const auto diagnostics = publicDiagnostics();
        runtime.reset(); assert(omniUsdLiveShutdown(5000) == OMNI_USD_LIVE_OK);
        std::cout << "{\"closed\":1,\"covered\":" << closed.covered << ",\"diagnostics\":" << diagnostics << "}"
                  << std::endl;
        return 0;
    }
    if (terminal) {
        // The host fails while the read is parked; its held application never runs.
        fault->store(OMNI_USD_LIVE_INTERNAL_ERROR, std::memory_order_release);
        std::shared_ptr<AsyncApplication> task;
        { std::lock_guard<std::mutex> lock(seen.mutex); task = std::move(held); }
        assert(task);
        bool ran = false;
        task->completion->run([&] { ran = true; });
        task.reset(); // Retire effect/work captures before the completion owner.
        seen.wait([&] { return failed(seen.state); });
        const auto stopped = actor->snapshot();
        assert(!ran && stopped.result == OMNI_USD_LIVE_INTERNAL_ERROR && !stopped.readStarted);
        // Accepted coverage stays retained; only the base is applied.
        assert(stopped.applied == 1 && stopped.covered >= paused.covered && stopped.covered < last);
        assert(stopped.retained == stopped.covered - 1);
        for (const auto& work : actor->retained()) {
            const auto effect = std::static_pointer_cast<const LayerActor::Effect>(work.owner);
            assert(work.disposition == LayerProgress::Disposition::Peer && work.first == work.last && !work.applied);
            assert(effect->bytes() == std::string(1024, char('a' + work.first % 26)));
        }
        assert(statusState() == OMNI_USD_LIVE_STATE_FAILED);
        barrier("actor-read-failed"); // The service confirms the parked read's cancellation before close.
        assert(actor->close(5000) == OMNI_USD_LIVE_OK);
        actor.reset();
        unregister();
        const auto diagnostics = publicDiagnostics();
        runtime.reset(); assert(omniUsdLiveShutdown(5000) == OMNI_USD_LIVE_OK);
        std::cout << "{\"terminal\":1,\"result\":" << unsigned(stopped.result) << ",\"covered\":" << stopped.covered
                  << ",\"retained\":" << stopped.retained << ",\"diagnostics\":" << diagnostics << "}" << std::endl;
        return 0;
    }
    if (serviceEnd) {
        // The service ended the stream while the read was parked: nothing
        // changes until the read resumes, and no reconnect starts.
        const auto parked = actor->snapshot();
        assert(actor->transportSnapshot().readPaused && parked.readStarted && !failed(parked));
        assert(parked.generation == paused.generation);
        assert(statusState() == OMNI_USD_LIVE_STATE_READY);
    }
    if (ingress) {
        std::lock_guard<std::mutex> lock(decoder.mutex); decoder.release = true; decoder.changed.notify_all();
    } else {
        std::shared_ptr<AsyncApplication> task;
        { std::lock_guard<std::mutex> lock(seen.mutex); task = std::move(held); }
        assert(task);
        task->completion->run([&] {
            std::lock_guard<std::mutex> lock(seen.mutex);
            seen.applied.emplace_back(task->work.last, task->effect->bytes());
        });
        task.reset(); // Retire effect/work captures before the completion owner.
    }
    seen.wait([&] { return seen.applied.size() == last - 1 || failed(seen.state); });
    seen.wait([&] { return seen.state.applied == last || failed(seen.state); });
    const auto state = actor->snapshot();
    assert(!failed(state) && state.gate == LayerProgress::Gate::Ready && state.covered == last);
    assert(!actor->transportSnapshot().readPaused);
    // A service-ended read reconnects once, from the frontier the old stream covered.
    assert(serviceEnd ? state.generation == paused.generation + 1 : state.generation == paused.generation);
    assert(statusState() == OMNI_USD_LIVE_STATE_READY);
    {
        std::lock_guard<std::mutex> lock(seen.mutex);
        for (uint64_t sequence = 2; sequence <= last; ++sequence)
            assert(seen.applied[sequence - 2].first == sequence &&
                   seen.applied[sequence - 2].second == std::string(1024, char('a' + sequence % 26)));
    }
    assert(actor->close(5000) == OMNI_USD_LIVE_OK);
    actor.reset();
    unregister();
    const auto diagnostics = publicDiagnostics();
    runtime.reset(); assert(omniUsdLiveShutdown(5000) == OMNI_USD_LIVE_OK);
    std::cout << "{\"applied\":" << state.applied << ",\"covered\":" << state.covered
              << ",\"generations\":" << state.generation - paused.generation + 1
              << ",\"diagnostics\":" << diagnostics << "}" << std::endl;
    return 0;
}
// Server-side rejection of an accepted token (design sections 13 and 14): a
// read reconnect, an UpdateLayer issue, or a held Ping carrying a replaced
// generation. argv[5] is the initial token, argv[6] the session and argv[7..]
// the replacement tokens, one per challenge. The auth callback installs them,
// except that a read's first challenge is answered by the driver once the
// layer reports AUTH_WAIT.
int actorServerAuthDriver(int argc, char** argv)
{
    assert(argc >= 7);
    const std::string mode = argv[2];
    const bool read = mode == "actor-auth-server-read", update = mode == "actor-auth-server-update";
    const bool stale = mode == "actor-auth-stale-generation";
    assert(read || update || stale);
    const auto expected = static_cast<OmniUsdLiveResult>(std::stoul(argv[3]));
    auto options = record<OmniUsdLiveRuntimeOptions>();
    options.retry_initial_ms = 5; options.retry_maximum_ms = 20; options.retry_max_attempts = 3;
    // Reads follow the reconnect options, whose attempt bound differs from retry_max_attempts.
    options.reconnect_initial_ms = 5; options.reconnect_maximum_ms = 20; options.reconnect_max_attempts = 4;
    options.default_auth_wait_timeout_ms = 30000; options.initial_open_timeout_ms = 4000;
    assert(omniUsdLiveInitialize(&options) == OMNI_USD_LIVE_OK);
    const TestTls tls;
    auto endpointOptions = record<OmniUsdLiveEndpointOptions>();
    endpointOptions.authority = argv[1]; endpointOptions.access_token = argv[5];
    tls.apply(endpointOptions);
    OmniUsdLiveEndpoint endpoint = 0;
    assert(omniUsdLiveRegisterEndpoint(&endpointOptions, &endpoint) == OMNI_USD_LIVE_OK);
    // Without timing Pings, once the Ping that learns the endpoint's
    // capabilities is answered (awaited below), the next new stream is the
    // read reconnect or the update.
    if (!stale) {
        auto jitter = record<OmniUsdLiveJitterParams>();
        assert(omniUsdLiveConfigureJitterReduction(endpoint, &jitter) == OMNI_USD_LIVE_OK);
    }
    auto runtime = getRuntime();
    struct Challenges {
        std::mutex mutex; std::condition_variable changed;
        std::vector<OmniUsdLiveAuthRequired> events; std::vector<OmniUsdLiveResult> answers;
        std::vector<std::string> tokens; size_t next = 0;
        bool deferFirst = false; // The driver answers the first challenge itself.
    } challenges;
    challenges.deferFirst = read;
    for (int i = 7; i < argc; ++i) challenges.tokens.emplace_back(argv[i]);
    OmniUsdLiveCallbackHandle observer = 0;
    assert(omniUsdLiveRegisterAuthRequiredCallback(endpoint, &challenges,
        [](void* user, const OmniUsdLiveAuthRequired* event) {
            auto& self = *static_cast<Challenges*>(user);
            std::string token;
            {
                std::lock_guard<std::mutex> lock(self.mutex);
                self.events.push_back(*event);
                if (self.deferFirst && self.events.size() == 1) self.next = 1;
                else if (self.next < self.tokens.size()) token = self.tokens[self.next++];
                self.changed.notify_all();
            }
            if (token.empty()) return; // No replacement: the wait ends by its own bound.
            uint64_t generation = 0;
            const auto answer = omniUsdLiveSetAccessToken(event->endpoint, event->rejected_generation, token.c_str(),
                                                          0, &generation);
            std::lock_guard<std::mutex> lock(self.mutex);
            self.answers.push_back(answer); self.changed.notify_all();
        }, &observer) == OMNI_USD_LIVE_OK);
    StatusSeen statuses; statuses.caller = std::this_thread::get_id();
    OmniUsdLiveCallbackHandle statusObserver = 0;
    assert(omniUsdLiveRegisterStatusCallback(endpoint, &statuses, [](void* user, const OmniUsdLiveStatusEvent* event) {
        assert(event); static_cast<StatusSeen*>(user)->receive(*event);
    }, &statusObserver) == OMNI_USD_LIVE_OK);
    Seen seen;
    LayerActor::Callbacks callbacks;
    callbacks.decode = [](const LayerReadUpdate& update) {
        LayerActor::Decoded decoded; decoded.first = update.header.sequence_num(); return decoded;
    };
    callbacks.apply = [&](const LayerProgress::Work& work, const LayerActor::Effect& effect) {
        std::lock_guard<std::mutex> lock(seen.mutex);
        seen.applied.emplace_back(work.last, effect.bytes()); seen.changed.notify_all();
    };
    callbacks.changed = [&](const LayerProgress::Snapshot& state) {
        std::lock_guard<std::mutex> lock(seen.mutex); seen.state = state; seen.changed.notify_all();
    };
    LayerActor::Options settings;
    settings.session = argv[6]; settings.layer = "root.live"; settings.object = settings.resume = 1;
    settings.initialState = std::make_shared<const std::string>("opaque-base");
    auto actor = LayerActor::open(runtime, endpoint, std::move(settings), std::move(callbacks));
    seen.wait([&] { return seen.state.gate == LayerProgress::Gate::Ready || failed(seen.state); });
    assert(!failed(actor->snapshot()));
    const auto ready = actor->snapshot();
    const auto barrier = [](const std::string& marker) {
        std::cout << marker << std::endl;
        std::string command; assert(std::getline(std::cin, command) && command == "continue");
    };
    if (!stale) {
        const auto registered = runtime->findEndpoint(endpoint);
        const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(6);
        while (!(registered->capabilities.load() & Endpoint::kCapabilitiesKnown)) {
            assert(std::chrono::steady_clock::now() < until);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    barrier("actor-server-auth-ready"); // The fixture expires, revokes, arms or resets.
    if (update || stale) {
        const std::string bytes = "local-0";
        assert(actor->submit(bytes, std::make_shared<const std::string>(bytes)));
    }
    if (read) { // Answered once the layer reports AUTH_WAIT, so that state is always published.
        OmniUsdLiveAuthRequired first{};
        {
            std::unique_lock<std::mutex> lock(challenges.mutex);
            assert(challenges.changed.wait_for(lock, std::chrono::seconds(6), [&] { return !challenges.events.empty(); }));
            first = challenges.events.front();
        }
        statuses.wait([&] {
            for (const auto& event : statuses.events)
                if (event.status.subject_id == actor->subjectId() && event.status.state == OMNI_USD_LIVE_STATE_AUTH_WAIT)
                    return true;
            return false;
        });
        uint64_t generation = 0;
        const auto answer = omniUsdLiveSetAccessToken(endpoint, first.rejected_generation, challenges.tokens[0].c_str(),
                                                      0, &generation);
        std::lock_guard<std::mutex> lock(challenges.mutex);
        challenges.answers.push_back(answer); challenges.changed.notify_all();
    }
    // Failure-only deadline: a recovered layer is READY on a newer read generation
    // (read) or has applied its own write (update, stale); a failed one is terminal.
    seen.wait([&] {
        if (failed(seen.state)) return true;
        return read ? seen.state.gate == LayerProgress::Gate::Ready && seen.state.generation > ready.generation
                    : seen.state.applied >= 2;
    });
    if (stale) {
        assert(!failed(actor->snapshot()));
        barrier("actor-server-auth-committed"); // The fixture releases the held Ping.
        // The rejected Ping's handler, which runs the auth completion, has returned.
        const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        for (;;) {
            const auto metrics = runtime->diagnostics->snapshot(0, 0).metrics;
            if (metrics.rpc[OMNI_USD_LIVE_OPERATION_PING][OMNI_USD_LIVE_UNAUTHENTICATED] == 1 && !metrics.cqTags.handling)
                break;
            assert(std::chrono::steady_clock::now() < until); std::this_thread::yield();
        }
    }
    const auto state = actor->snapshot();
    assert(expected == OMNI_USD_LIVE_OK ? !failed(state) : state.result == expected);
    auto layer = record<OmniUsdLiveStatus>();
    assert(runtime->statuses->snapshot(actor->subjectId(), &layer) == OMNI_USD_LIVE_OK);
    auto auth = record<OmniUsdLiveAuthState>();
    assert(omniUsdLiveGetAuthState(endpoint, &auth) == OMNI_USD_LIVE_OK);
    // Status callbacks run on a callback worker after the actor publishes, so
    // the sample waits until the snapshot's revision has been delivered.
    statuses.wait([&] {
        for (const auto& event : statuses.events)
            if (event.status.subject_id == actor->subjectId() && event.status.revision >= layer.revision) return true;
        return false;
    });
    std::vector<uint32_t> states; // The layer's published states, consecutive repeats collapsed.
    {
        std::lock_guard<std::mutex> lock(statuses.mutex);
        for (const auto& event : statuses.events)
            if (event.status.subject_id == actor->subjectId() && (states.empty() || states.back() != event.status.state))
                states.push_back(event.status.state);
    }
    const auto diagnostics = publicDiagnostics();
    assert(omniUsdLiveUnregisterCallback(observer, 3000) == OMNI_USD_LIVE_OK);
    assert(omniUsdLiveUnregisterCallback(statusObserver, 3000) == OMNI_USD_LIVE_OK);
    assert(actor->close(5000) == OMNI_USD_LIVE_OK);
    actor.reset();
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    for (;;) { // A closed actor can still have a scheduler owner; observe retirement.
        const auto result = omniUsdLiveUnregisterEndpoint(endpoint, 0);
        if (result == OMNI_USD_LIVE_OK) break;
        assert(result == OMNI_USD_LIVE_BUSY && std::chrono::steady_clock::now() < until);
        std::this_thread::yield();
    }
    runtime.reset(); assert(omniUsdLiveShutdown(5000) == OMNI_USD_LIVE_OK);
    std::cout << "{\"result\":" << state.result << ",\"applied\":" << state.applied << ",\"generation\":"
              << auth.generation << ",\"pending\":" << layer.pending_write_count << ",\"challenges\":[";
    for (size_t i = 0; i < challenges.events.size(); ++i) { // [reason, rejected generation, gRPC present, gRPC code]
        const auto& event = challenges.events[i];
        std::cout << (i ? "," : "") << '[' << event.reason << ',' << event.rejected_generation << ','
                  << event.grpc_status_present << ',' << event.grpc_status << ']';
    }
    std::cout << "],\"answers\":[";
    for (size_t i = 0; i < challenges.answers.size(); ++i) std::cout << (i ? "," : "") << challenges.answers[i];
    std::cout << "],\"states\":[";
    for (size_t i = 0; i < states.size(); ++i) std::cout << (i ? "," : "") << states[i];
    std::cout << "],\"diagnostics\":" << diagnostics << '}' << std::endl;
    return 0;
}
// First-response deadline of a read attempt (design section 12), on its real
// clock: initial_open_timeout_ms is 1000 ms. -stall: the service ends the first
// read and accepts every reconnect without answering it; each attempt ends
// TIMEOUT at its deadline and counts, so the third fails the layer UNAVAILABLE.
// -paced: the read is held for incoming pacing behind a held application when
// the layer resubscribes; the new attempt's deadline passes while it is held
// and restarts, and the attempt resumes once the application is released.
int actorReadDeadlineDriver(int argc, char** argv)
{
    assert(argc == 7 && std::string(argv[6]) == "scripted");
    const bool paced = std::string(argv[2]) == "actor-read-deadline-paced";
    constexpr uint64_t last = 37; // One caught-up header, then 36 peer updates (-paced).
    auto options = record<OmniUsdLiveRuntimeOptions>();
    options.initial_open_timeout_ms = 1000;
    options.max_requests = 32; // High-water mark 28, low-water mark 16.
    options.reconnect_initial_ms = 5; options.reconnect_maximum_ms = 20; options.reconnect_max_attempts = 3;
    options.max_update_bytes = 1024;
    options.max_pending_bytes_per_endpoint = 131072;
    assert(omniUsdLiveInitialize(&options) == OMNI_USD_LIVE_OK);
    auto endpointOptions = record<OmniUsdLiveEndpointOptions>();
    endpointOptions.authority = argv[1]; endpointOptions.access_token = argv[5];
    OmniUsdLiveEndpoint endpoint = 0;
    assert(omniUsdLiveRegisterEndpoint(&endpointOptions, &endpoint) == OMNI_USD_LIVE_OK);
    auto jitter = record<OmniUsdLiveJitterParams>();
    assert(omniUsdLiveConfigureJitterReduction(endpoint, &jitter) == OMNI_USD_LIVE_OK);
    auto runtime = getRuntime();
    Seen seen;
    std::shared_ptr<AsyncApplication> held;
    LayerActor::Callbacks callbacks;
    callbacks.decode = [](const LayerReadUpdate& update) {
        LayerActor::Decoded decoded; decoded.first = update.header.sequence_num(); return decoded;
    };
    callbacks.applyAsync = [&](LayerProgress::Work work, std::shared_ptr<const LayerActor::Effect> effect,
                               std::shared_ptr<LayerActor::ApplyCompletion> completion) {
        if (work.last == 2) {
            std::lock_guard<std::mutex> lock(seen.mutex);
            assert(!held);
            held = std::make_shared<AsyncApplication>();
            held->completion = std::move(completion); held->effect = std::move(effect); held->work = std::move(work);
            seen.entered = true; seen.changed.notify_all();
            return;
        }
        completion->run([&] {
            std::lock_guard<std::mutex> lock(seen.mutex);
            seen.applied.emplace_back(work.last, effect->bytes());
            seen.changed.notify_all();
        });
    };
    callbacks.changed = [&](const LayerProgress::Snapshot& state) {
        std::lock_guard<std::mutex> lock(seen.mutex);
        seen.state = state; seen.changed.notify_all();
    };
    LayerActor::Options settings;
    settings.session = argv[6]; settings.layer = "root.live"; settings.object = settings.resume = 1;
    settings.initialState = std::make_shared<const std::string>("opaque-base");
    auto actor = LayerActor::open(runtime, endpoint, std::move(settings), std::move(callbacks));
    seen.wait([&] { return seen.state.gate == LayerProgress::Gate::Ready || failed(seen.state); });
    assert(!failed(actor->snapshot()));
    const auto barrier = [](const std::string& marker) {
        std::cout << marker << std::endl;
        std::string command; assert(std::getline(std::cin, command) && command == "continue");
    };
    // Failure-only watchdog: three deadlines and their backoffs fit well inside it.
    const auto watchdog = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    const auto observe = [&](const auto& predicate) {
        std::unique_lock<std::mutex> lock(seen.mutex);
        assert(seen.changed.wait_until(lock, watchdog, [&] { return predicate() || failed(seen.state); }));
    };
    const auto ready = actor->snapshot();
    barrier("actor-deadline-ready");
    if (!paced) {
        observe([] { return false; }); // Only the failure ends the wait.
    } else {
        // The service has sent the burst; the held application pauses the read.
        while (!actor->transportSnapshot().readPaused) {
            assert(!failed(actor->snapshot()) && std::chrono::steady_clock::now() < watchdog);
            std::this_thread::yield();
        }
        actor->resubscribeForTest();
        // The new attempt starts parked behind the hold, and its deadline passes.
        while (actor->transportSnapshot().heldReadWindows == 0) {
            assert(!failed(actor->snapshot()) && std::chrono::steady_clock::now() < watchdog);
            std::this_thread::yield();
        }
        const auto parked = actor->snapshot();
        assert(parked.generation == ready.generation + 1 && parked.gate == LayerProgress::Gate::Recovering);
        assert(actor->transportSnapshot().readPaused);
        std::shared_ptr<AsyncApplication> task;
        {
            std::lock_guard<std::mutex> lock(seen.mutex);
            task = std::move(held);
        }
        assert(task);
        task->completion->run([&] {
            std::lock_guard<std::mutex> lock(seen.mutex);
            seen.applied.emplace_back(task->work.last, task->effect->bytes());
        });
        task.reset(); // Retire effect/work captures before the completion owner.
        observe([&] { return seen.state.applied == last && seen.state.gate == LayerProgress::Gate::Ready; });
    }
    const auto state = actor->snapshot();
    const auto windows = actor->transportSnapshot().heldReadWindows;
    assert(actor->close(5000) == OMNI_USD_LIVE_OK);
    actor.reset();
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    for (;;) {
        const auto result = omniUsdLiveUnregisterEndpoint(endpoint, 0);
        if (result == OMNI_USD_LIVE_OK) break;
        assert(result == OMNI_USD_LIVE_BUSY && std::chrono::steady_clock::now() < until);
        std::this_thread::yield();
    }
    const auto diagnostics = publicDiagnostics();
    runtime.reset(); assert(omniUsdLiveShutdown(5000) == OMNI_USD_LIVE_OK);
    std::cout << "{\"result\":" << unsigned(state.result) << ",\"failed\":" << failed(state)
              << ",\"applied\":" << state.applied << ",\"covered\":" << state.covered
              << ",\"generations\":" << state.generation - ready.generation + 1 << ",\"held_windows\":" << windows
              << ",\"diagnostics\":" << diagnostics << "}" << std::endl;
    return 0;
}
}
int layerActorDriver(int argc, char** argv)
{
    if (argc > 2 && std::string(argv[2]).rfind("actor-host-fault-", 0) == 0) return actorHostFaultDriver(argc,argv);
    if (argc > 2 && std::string(argv[2]) == "actor-release-notices") return actorReleaseNoticesDriver(argc,argv);
    if (argc > 2 && std::string(argv[2]) == "actor-buffer-emptied") return actorBufferEmptiedDriver(argc,argv);
    if (argc > 2 && std::string(argv[2]) == "actor-drop-without-close") return actorDropWithoutCloseDriver(argc,argv);
    if (argc > 2 && std::string(argv[2]) == "actor-issue-reservation") return actorIssueReservationDriver(argc,argv);
    if (argc > 2 && std::string(argv[2]).rfind("actor-flood-", 0) == 0) return actorFloodDriver(argc, argv);
    if (argc > 2 && std::string(argv[2]).rfind("actor-read-pause", 0) == 0)
        return actorReadPauseDriver(argc, argv);
    if (argc > 2 && std::string(argv[2]).rfind("actor-read-deadline-", 0) == 0)
        return actorReadDeadlineDriver(argc, argv);
    if (argc > 2 && (std::string(argv[2]).rfind("actor-auth-server-", 0) == 0 || std::string(argv[2]) == "actor-auth-stale-generation"))
        return actorServerAuthDriver(argc, argv);
    assert(argc == 7);
    const std::string mode = argv[2];
    if (mode == "actor-auth-wait-open-deadline") {
        // The 1 ms initial-open deadline precedes the 30000 ms auth wait, and the
        // actor records starvation when it is constructed: an open starved of a
        // credential still ends with the auth error.
        auto options = record<OmniUsdLiveRuntimeOptions>();
        options.default_auth_wait_timeout_ms = 30000; options.initial_open_timeout_ms = 1;
        assert(omniUsdLiveInitialize(&options) == OMNI_USD_LIVE_OK);
        auto ep = record<OmniUsdLiveEndpointOptions>(); ep.authority = argv[1]; ep.access_token = "";
        OmniUsdLiveEndpoint endpoint = 0;
        assert(omniUsdLiveRegisterEndpoint(&ep, &endpoint) == OMNI_USD_LIVE_OK);
        auto runtime = getRuntime();
        LayerActor::Options settings; settings.session = argv[6]; settings.layer = "root.live";
        settings.object = 1; settings.resume = 1; settings.initialState = std::make_shared<int>(0);
        struct Terminal {
            std::mutex mutex;
            std::condition_variable changed;
            bool failed = false;
            OmniUsdLiveResult result = OMNI_USD_LIVE_OK;
        };
        auto terminal = std::make_shared<Terminal>();
        LayerActor::Callbacks callbacks;
        callbacks.decode = [](const LayerReadUpdate&) { return LayerActor::Decoded{}; };
        callbacks.apply = [](const LayerProgress::Work&, const LayerActor::Effect&) {};
        callbacks.changed = [terminal](const LayerProgress::Snapshot& state) {
            if (!failed(state)) return;
            std::lock_guard<std::mutex> lock(terminal->mutex);
            if (!terminal->failed) { terminal->failed = true; terminal->result = state.result; }
            terminal->changed.notify_all();
        };
        auto actor = LayerActor::open(runtime, endpoint, std::move(settings), std::move(callbacks));
        OmniUsdLiveResult result;
        {
            // Failure watchdog only; the result, not the duration, is asserted.
            std::unique_lock<std::mutex> lock(terminal->mutex);
            assert(terminal->changed.wait_for(lock, std::chrono::seconds(20), [&] { return terminal->failed; }));
            result = terminal->result;
        }
        assert(actor->snapshot().result == result);
        assert(actor->close(3000) == OMNI_USD_LIVE_OK);
        actor.reset();
        const auto retiredBy = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        for (;;) {
            const auto unregistered = omniUsdLiveUnregisterEndpoint(endpoint, 0);
            if (unregistered == OMNI_USD_LIVE_OK) break;
            assert(unregistered == OMNI_USD_LIVE_BUSY && std::chrono::steady_clock::now() < retiredBy);
            std::this_thread::yield();
        }
        runtime.reset();
        assert(omniUsdLiveShutdown(3000) == OMNI_USD_LIVE_OK);
        std::cout << "{\"result\":" << result << "}\n";
        return 0;
    }
    if (mode == "actor-auth-wait-close" || mode == "actor-auth-wait-fail-close") {
        auto options = record<OmniUsdLiveRuntimeOptions>();
        options.default_auth_wait_timeout_ms = options.initial_open_timeout_ms = 30000;
        assert(omniUsdLiveInitialize(&options) == OMNI_USD_LIVE_OK);
        auto ep = record<OmniUsdLiveEndpointOptions>(); ep.authority = argv[1]; ep.access_token = "";
        OmniUsdLiveEndpoint endpoint = 0;
        assert(omniUsdLiveRegisterEndpoint(&ep, &endpoint) == OMNI_USD_LIVE_OK);
        auto runtime = getRuntime();
        LayerActor::Options settings; settings.session = argv[6]; settings.layer = "root.live";
        settings.object = 1; settings.resume = 1; settings.initialState = std::make_shared<int>(0);
        struct Gate {
            std::mutex mutex;
            std::condition_variable changed;
            size_t visits = 0;
            bool entered = false, release = false;
        } gate;
        if (mode == "actor-auth-wait-fail-close") {
            settings.authGateContextForTest = &gate;
            settings.beforeAuthGateForTest = [](void* context) noexcept {
                auto& gate = *static_cast<Gate*>(context);
                std::unique_lock<std::mutex> lock(gate.mutex);
                if (++gate.visits == 1 || gate.release) return;
                gate.entered = true; gate.changed.notify_all();
                assert(gate.changed.wait_for(lock, std::chrono::seconds(3), [&] { return gate.release; }));
            };
        }
        LayerActor::Callbacks callbacks;
        callbacks.decode = [](const LayerReadUpdate&) { return LayerActor::Decoded{}; };
        callbacks.apply = [](const LayerProgress::Work&, const LayerActor::Effect&) {};
        auto actor = LayerActor::open(runtime, endpoint, std::move(settings), std::move(callbacks));
        assert(runtime->diagnostics->waitForOperationAuthWaitsForTest(AuthWaitKind::Layer, 1,
            std::chrono::steady_clock::now() + std::chrono::seconds(3)));
        if (mode == "actor-auth-wait-fail-close") {
            uint64_t generation = 0;
            assert(omniUsdLiveSetAccessToken(endpoint, 0, "synthetic-writer", 0, &generation) == OMNI_USD_LIVE_OK);
            {
                std::unique_lock<std::mutex> lock(gate.mutex);
                assert(gate.changed.wait_for(lock, std::chrono::seconds(3), [&] { return gate.entered; }));
            }
            assert(actor->failPublication(OMNI_USD_LIVE_UNAVAILABLE) == OMNI_USD_LIVE_OK);
            assert(actor->close(0) == OMNI_USD_LIVE_TIMEOUT);
            { std::lock_guard<std::mutex> lock(gate.mutex); gate.release = true; gate.changed.notify_all(); }
        }
        assert(actor->close(3000) == OMNI_USD_LIVE_OK);
        assert(!runtime->diagnostics->snapshot(0, 0).metrics.operationAuthWaits[1].current);
        actor.reset();
        const auto retiredBy = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        for (;;) {
            const auto result = omniUsdLiveUnregisterEndpoint(endpoint, 0);
            if (result == OMNI_USD_LIVE_OK) break;
            assert(result == OMNI_USD_LIVE_BUSY && std::chrono::steady_clock::now() < retiredBy);
            std::this_thread::yield();
        }
        const auto diagnostics = publicDiagnostics();
        runtime.reset();
        assert(omniUsdLiveShutdown(3000) == OMNI_USD_LIVE_OK);
        std::cout << "{\"diagnostics\":" << diagnostics << "}\n";
        return 0;
    }
    const bool authExpiry = mode=="actor-auth-expiry-idle" || mode=="actor-auth-expiry-held";
    const bool asyncTest = mode.rfind("actor-async-",0)==0;
    const bool applicationTimingTest = mode == "actor-async-fail" || mode == "actor-async-cancel" ||
        mode == "actor-async-queue-throw" || mode == "actor-async-close" || mode == "actor-async-run-throw";
    const bool syncTimingTest = mode == "actor-sync-invalid";
    std::atomic<uint64_t> applicationMs{0};
    const bool uncertain = mode=="actor-uncertain" || mode=="actor-capture-changed";
    const bool captureOwner = mode=="actor-async-capture";
    const bool localTest = mode.rfind("actor-local-",0)==0;
    const bool jitterTest = mode == "actor-jitter-release" || mode == "actor-jitter-close" ||
                            mode == "actor-jitter-budget" || mode == "actor-drain-capture" ||
                            mode == "actor-jitter-fail" || mode == "actor-jitter-preempt";
    const auto expected = static_cast<OmniUsdLiveResult>(std::stoul(argv[3]));
    auto options = record<OmniUsdLiveRuntimeOptions>();
    options.retry_initial_ms = 5; options.retry_maximum_ms = 20; options.retry_max_attempts = 3;
    // Reads follow the reconnect options, whose attempt bound differs from retry_max_attempts.
    options.reconnect_initial_ms = 5; options.reconnect_maximum_ms = 20; options.reconnect_max_attempts = 4;
    const bool authReissue = mode == "actor-auth-refresh" || mode == "actor-auth-reissue-fail";
    // Reissue modes refresh on the callback worker or fail refresh explicitly;
    // the auth wait never races that as a wall-clock deadline.
    options.default_auth_wait_timeout_ms = authExpiry || authReissue ? 30000 : 200; options.initial_open_timeout_ms = 4000;
    if (mode == "actor-jitter-budget") options.max_requests = 2;
    assert(omniUsdLiveInitialize(&options) == OMNI_USD_LIVE_OK);
    auto endpointOptions = record<OmniUsdLiveEndpointOptions>();
    endpointOptions.authority = argv[1]; endpointOptions.access_token = argv[5];
    OmniUsdLiveEndpoint endpoint = 0;
    assert(omniUsdLiveRegisterEndpoint(&endpointOptions, &endpoint) == OMNI_USD_LIVE_OK);
    if (jitterTest) {
        auto jitter = record<OmniUsdLiveJitterParams>();
        jitter.delay_constant_ms = jitter.delay_maximum_ms = 30000;
        assert(omniUsdLiveConfigureJitterReduction(endpoint, &jitter) == OMNI_USD_LIVE_OK);
    } else if (mode == "actor-drain-clean-close" || asyncTest || localTest || authExpiry) {
        // No optional Ping completion should race the idle-close precondition.
        auto jitter = record<OmniUsdLiveJitterParams>();
        assert(omniUsdLiveConfigureJitterReduction(endpoint, &jitter) == OMNI_USD_LIVE_OK);
    }
    auto runtime = getRuntime();
    Seen seen;
    StatusSeen statuses;
    statuses.caller = std::this_thread::get_id();
    OmniUsdLiveCallbackHandle statusObserver = 0;
    // No diagnostic sink is installed: typed status alone drives these checks.
    assert(omniUsdLiveRegisterStatusCallback(0, &statuses,
        [](void* user, const OmniUsdLiveStatusEvent* event) {
            assert(event); static_cast<StatusSeen*>(user)->receive(*event);
        }, &statusObserver) == OMNI_USD_LIVE_OK);
    statuses.wait([&] { return !statuses.events.empty(); });
    {
        std::lock_guard<std::mutex> lock(statuses.mutex);
        assert(statuses.events.front().endpoint == endpoint &&
               statuses.events.front().status.subject_kind == OMNI_USD_LIVE_SUBJECT_ENDPOINT);
    }
    struct Auth { Seen* seen; std::string mode; } auth{&seen, mode};
    OmniUsdLiveCallbackHandle observer = 0;
    assert(omniUsdLiveRegisterAuthRequiredCallback(endpoint, &auth,
        [](void* user, const OmniUsdLiveAuthRequired* event) {
            auto& auth = *static_cast<Auth*>(user);
            {
                std::lock_guard<std::mutex> lock(auth.seen->mutex);
                ++auth.seen->auth; auth.seen->changed.notify_all();
            }
            if (auth.mode == "actor-auth-refresh") {
                uint64_t generation;
                assert(omniUsdLiveSetAccessToken(event->endpoint, event->rejected_generation,
                    "synthetic-writer", 0, &generation) == OMNI_USD_LIVE_OK);
            }
        }, &observer) == OMNI_USD_LIVE_OK);
    std::shared_ptr<AsyncApplication> asyncApplication;
    const auto asyncRetirement=std::make_shared<AsyncRetirement>();
    size_t asyncCalls=0;
    std::atomic<LayerActor*> saveActor{nullptr}; // Strong main-thread owner survives callback quiescence.
    LayerActor::Callbacks callbacks;
    const auto caller = std::this_thread::get_id();
    callbacks.decode = [&](const LayerReadUpdate& update) {
        if (mode == "actor-backlog" && update.header.sequence_num() == 2) {
            assert(omniUsdLiveWaitForPendingUpdates(1000) == OMNI_USD_LIVE_BUSY);
            std::unique_lock<std::mutex> lock(seen.mutex);
            seen.entered = true; seen.changed.notify_all();
            seen.changed.wait(lock, [&] { return seen.release; });
        }
        LayerActor::Decoded decoded;
        decoded.first = update.header.sequence_num();
        if ((mode == "actor-bootstrap" || jitterTest) && decoded.first == 1) {
            decoded.kind = LayerProgress::CoverageKind::Snapshot;
            decoded.recoveryBoundary = true;
        }
        return decoded;
    };
    callbacks.apply = [&](const LayerProgress::Work& work, const LayerActor::Effect& effect) {
        assert(std::this_thread::get_id() != caller);
        if (syncTimingTest) { applicationMs = 11; throw std::invalid_argument("synthetic host failure"); }
        std::unique_lock<std::mutex> lock(seen.mutex);
        seen.applied.emplace_back(work.last, effect.bytes());
        if (!effect.incoming) assert(effect.intent);
        if (mode == "actor-held" || (mode == "actor-drain-capture" && work.last == 3)) {
            assert(omniUsdLiveWaitForPendingUpdates(1000) == OMNI_USD_LIVE_BUSY);
            seen.entered = true; seen.changed.notify_all();
            seen.changed.wait(lock, [&] { return seen.release; });
        }
    };
    if (asyncTest) {
        callbacks.apply={};
        callbacks.applyAsync = [&](LayerProgress::Work work,std::shared_ptr<const LayerActor::Effect> effect,
                                   std::shared_ptr<LayerActor::ApplyCompletion> completion) {
            assert(std::this_thread::get_id()!=caller && completion && effect && !effect->incoming && effect->intent);
            auto task=std::make_shared<AsyncApplication>();
            task->completion=std::move(completion); task->effect=std::move(effect); task->work=std::move(work);
            bool first;
            {
                std::lock_guard<std::mutex> lock(seen.mutex);
                first=++asyncCalls==1;
                assert(first || (mode=="actor-async-release" && !asyncApplication));
                if (first) {
                    if (mode=="actor-async-release" || captureOwner) {
                        task->capture=std::make_unique<HeldAsyncCapture>(); task->capture->gate=asyncRetirement;
                    }
                    asyncApplication=task; seen.entered=true; seen.changed.notify_all();
                }
            }
            if (first && mode == "actor-async-run-throw") {
                applicationMs = 13;
                task->completion->run([&] { applicationMs = 24; });
                throw OmniUsdLiveResult(OMNI_USD_LIVE_INTERNAL_ERROR);
            }
            if (first && (mode=="actor-async-queue-throw" || captureOwner)) throw OmniUsdLiveResult(OMNI_USD_LIVE_INTERNAL_ERROR);
            if (!first) task->completion->run([&] {
                assert(omniUsdLiveWaitForPendingUpdates(1000)==OMNI_USD_LIVE_BUSY);
                std::lock_guard<std::mutex> lock(seen.mutex);
                seen.applied.emplace_back(task->work.last,task->effect->bytes());
            });
        };
    }
    if (localTest) {
        if (mode=="actor-local-shared") {
            callbacks.apply={};
            callbacks.applyAsync=[&](LayerProgress::Work work,std::shared_ptr<const LayerActor::Effect> effect,
                                     std::shared_ptr<LayerActor::ApplyCompletion> completion) {
                completion->run([&] {
                    std::lock_guard<std::mutex> lock(seen.mutex);
                    assert(asyncCalls>=1 && asyncCalls<3); ++asyncCalls;
                    seen.applied.emplace_back(work.last,effect->bytes());
                });
            };
        }
        callbacks.localAsync = [&](LayerProgress::Work work,std::shared_ptr<const LayerActor::Effect> effect,
                                   std::shared_ptr<LayerActor::ApplyCompletion> completion) {
            assert(std::this_thread::get_id()!=caller && completion && effect && !effect->incoming && effect->intent);
            assert(work.disposition==LayerProgress::Disposition::Queued);
            auto task=std::make_shared<AsyncApplication>();
            task->completion=std::move(completion); task->effect=std::move(effect); task->work=std::move(work);
            task->capture=std::make_unique<HeldAsyncCapture>(); task->capture->gate=asyncRetirement;
            std::lock_guard<std::mutex> lock(seen.mutex);
            assert(!asyncApplication && ++asyncCalls==1);
            asyncApplication=std::move(task); seen.entered=true; seen.changed.notify_all();
        };
    }
    if (mode == "actor-jitter-preempt") {
        callbacks.localAsync = [&](auto, auto, std::shared_ptr<LayerActor::ApplyCompletion> completion) {
            const auto metric = runtime->diagnostics->snapshot(0, 0).metrics.jitter;
            assert(!metric.current && metric.started == 1 && metric.completed == 1);
            assert(metric.holdByResolution[static_cast<size_t>(JitterWaitResolution::Preempted)].count == 1);
            completion->run([] {});
        };
    }
    callbacks.changed = [&](const auto& state) {
        assert(std::this_thread::get_id() != caller);
        std::unique_lock<std::mutex> lock(seen.mutex);
        seen.state = state; seen.history.push_back(state); seen.changed.notify_all();
        if (mode=="actor-capture-changed" && failed(state) && !seen.entered) {
            seen.entered=true; seen.changed.notify_all();
            assert(seen.changed.wait_for(lock,std::chrono::seconds(6),[&] { return seen.release; }));
        }
        if ((mode == "actor-auth-fail" || mode == "actor-auth-expiry-held") &&
            state.gate == LayerProgress::Gate::Ready && !seen.entered) {
            seen.entered = true; seen.changed.notify_all();
            seen.changed.wait(lock, [&] { return seen.release; });
        }
        if (mode == "actor-drain-changed" && state.applied == 4 && !seen.entered) {
            assert(state.retained == 0 && state.pendingWrites == 0);
            assert(omniUsdLiveWaitForPendingUpdates(1000) == OMNI_USD_LIVE_BUSY);
            const auto owner = saveActor.load(std::memory_order_acquire);
            assert(owner && owner->save(1000) == OMNI_USD_LIVE_BUSY);
            seen.entered = true; seen.changed.notify_all();
            seen.changed.wait(lock, [&] { return seen.release; });
        }
    };
    ClosingPasses closing;
    LayerActor::Options settings;
    settings.session = argv[6]; settings.layer = "root.live"; settings.object = 1;
    closing.install(settings);
    if (applicationTimingTest || syncTimingTest) {
        settings.applicationNowForTest = [](void* context) noexcept {
            return std::chrono::steady_clock::time_point{} +
                std::chrono::milliseconds(static_cast<std::atomic<uint64_t>*>(context)->load());
        };
        settings.applicationClockContextForTest = &applicationMs;
    }
    const bool sequenceRegression = mode == "actor-sequence-regression";
    // The regression case resumes ahead of the scripted service's latest sequence.
    settings.resume = mode == "actor-bootstrap" || jitterTest ? 0 : sequenceRegression ? 3 : 1;
    settings.initialState = std::make_shared<const std::string>("opaque-base");
    // Longer than the entire Python driver deadline: held timing calls can
    // retire only through explicit release/cancellation during these cases.
    if (mode == "actor-jitter-budget" || mode == "actor-jitter-reconnect")
        settings.timingTimeoutMsForTest = 60000;
    const auto duplicateSettings = settings;
    auto actor = LayerActor::open(runtime, endpoint, std::move(settings), std::move(callbacks));
    saveActor.store(actor.get(), std::memory_order_release);
    const auto uri = makeLayerUri(makeSessionUri(argv[1], argv[6]), "root.live");
    auto layerStatus = [&] {
        auto status = record<OmniUsdLiveStatus>();
        assert(omniUsdLiveGetLayerStatus(uri.c_str(), &status) == OMNI_USD_LIVE_OK);
        assert(status.subject_kind == OMNI_USD_LIVE_SUBJECT_LAYER && status.progress_known);
        return status;
    };
    auto endpointStatus = [&] {
        auto status = record<OmniUsdLiveStatus>();
        assert(omniUsdLiveGetEndpointStatus(endpoint, &status) == OMNI_USD_LIVE_OK);
        return status;
    };
    const auto subject = layerStatus().subject_id;
    assert(subject);
    bool duplicateRejected = false;
    if (mode == "actor-roundtrip") {
        LayerActor::Callbacks duplicateCallbacks;
        duplicateCallbacks.decode = [](const auto&) { return LayerActor::Decoded{}; };
        duplicateCallbacks.apply = [](const auto&, const auto&) {};
        try {
            (void)LayerActor::open(runtime, endpoint, duplicateSettings, std::move(duplicateCallbacks));
            assert(false);
        } catch (OmniUsdLiveResult result) {
            assert(result == OMNI_USD_LIVE_ALREADY_EXISTS); duplicateRejected = true;
        }
        assert(layerStatus().subject_id == subject);
    }
    const bool localCause = mode == "actor-latched-local-cause";
    const bool rejectionClosesRead = mode == "actor-rejection-closes-read";
    const bool queuedInitially = mode == "actor-prequeue" || uncertain || authReissue || localCause ||
        rejectionClosesRead || sequenceRegression;
    std::vector<uint64_t> submitted;
    unsigned immediateChecks = 0;
    uint64_t incident = 0, acknowledgedBytes = 0;
    auto author = [&] {
        for (unsigned i = 0; i < (localTest || syncTimingTest || (asyncTest && mode!="actor-async-release") ? 1u : 3u); ++i) {
            std::string bytes = "local-" + std::to_string(i);
            const auto id = actor->submit(bytes, std::make_shared<const std::string>(bytes));
            assert(id); submitted.push_back(id);
            if (uncertain || mode == "actor-backlog") {
                // The reference holds the first write response, or this worker
                // is held in decode: no intent can retire across this query.
                const auto status = layerStatus();
                assert(status.pending_write_count == i + 1 && status.pending_bytes == (i + 1) * bytes.size());
                assert(status.has_unconfirmed_edits); ++immediateChecks;
            }
        }
    };
    if (queuedInitially) author();
    if (uncertain) std::cout << "actor-admissions-complete" << std::endl;
    if (sequenceRegression) { // The fixture holds the scripted read until every intent is queued.
        std::cout << "actor-regression-queued" << std::endl;
        std::string input; assert(bool(std::getline(std::cin, input)) && input == "continue");
    }
    if (localTest) {
        const auto checkpoint=[](const char* marker) {
            std::cout << marker << std::endl;
            std::string input; assert(bool(std::getline(std::cin,input)) && input=="continue");
        };
        const auto queued=[&] {
            const auto state=actor->snapshot(); const auto retained=actor->retained();
            assert(state.applied==1 && !state.issued && state.queued==1 && state.pendingWrites==1);
            const auto local=std::find_if(retained.begin(),retained.end(),[&](const auto& work) { return work.id==submitted.front(); });
            assert(local!=retained.end() && local->disposition==LayerProgress::Disposition::Queued);
            if (mode!="actor-local-shared") assert(retained.size()==1);
            const auto effect=std::static_pointer_cast<const LayerActor::Effect>(local->owner);
            assert(effect->authored=="local-0" && *std::static_pointer_cast<const std::string>(effect->intent)=="local-0");
        };
        seen.wait([&] { return seen.state.gate==LayerProgress::Gate::Ready || failed(seen.state); });
        assert(!failed(actor->snapshot())); author();
        seen.wait([&] { return seen.entered && seen.state.pendingWrites==1; });
        const auto idleUntil=std::chrono::steady_clock::now()+std::chrono::seconds(3);
        while (actor->transportSnapshot().workerScheduled) {
            assert(std::chrono::steady_clock::now()<idleUntil); std::this_thread::yield();
        }
        std::shared_ptr<AsyncApplication> task;
        { std::lock_guard<std::mutex> lock(seen.mutex); task=std::move(asyncApplication); assert(asyncCalls==1 && seen.applied.empty()); }
        queued();
        assert(omniUsdLiveWaitForPendingUpdates(25)==OMNI_USD_LIVE_TIMEOUT);
        checkpoint("local-publication-held");
        if (mode=="actor-local-shared") {
            const auto until=std::chrono::steady_clock::now()+std::chrono::seconds(3);
            while (actor->snapshot().covered<2) { assert(std::chrono::steady_clock::now()<until); std::this_thread::yield(); }
            assert(actor->snapshot().applied==1);
            std::lock_guard<std::mutex> lock(seen.mutex); assert(asyncCalls==1 && seen.applied.empty());
        }
        bool ran=false;
        if (mode=="actor-local-cancel") assert(actor->close(0)==OMNI_USD_LIVE_TIMEOUT);
        task->completion->run([&] {
            ran=true;
            assert(DispatchExecutor::isWorker());
            assert(omniUsdLiveWaitForPendingUpdates(1000)==OMNI_USD_LIVE_BUSY);
            if (mode=="actor-local-fail") throw OmniUsdLiveResult(OMNI_USD_LIVE_INTERNAL_ERROR);
        });
        assert(ran==(mode!="actor-local-cancel"));
        queued();
        std::thread retire([owned=std::move(task)]() mutable { owned.reset(); });
        {
            std::unique_lock<std::mutex> lock(asyncRetirement->mutex);
            assert(asyncRetirement->changed.wait_for(lock,std::chrono::seconds(3),[&] { return asyncRetirement->entered; }));
        }
        queued();
        if (mode=="actor-local-release" || mode=="actor-local-shared") {
            assert(omniUsdLiveWaitForPendingUpdates(25)==OMNI_USD_LIVE_TIMEOUT);
            std::lock_guard<std::mutex> lock(seen.mutex); assert(asyncCalls==1 && seen.applied.empty());
        }
        checkpoint("local-publication-capture-held");
        { std::lock_guard<std::mutex> lock(asyncRetirement->mutex); asyncRetirement->release=true; asyncRetirement->changed.notify_all(); }
        retire.join();
        if (mode=="actor-local-release") {
            const auto issuedUntil=std::chrono::steady_clock::now()+std::chrono::seconds(3);
            while (!actor->snapshot().issued && !failed(actor->snapshot())) {
                assert(std::chrono::steady_clock::now()<issuedUntil); std::this_thread::yield();
            }
            assert(!failed(actor->snapshot()) && actor->snapshot().applied==1);
            assert(omniUsdLiveWaitForPendingUpdates(25)==OMNI_USD_LIVE_TIMEOUT);
            checkpoint("local-publication-awaiting-ack");
            seen.wait([&] { return seen.state.applied==2 || failed(seen.state); });
            assert(actor->snapshot().applied==2 && !actor->snapshot().retained);
            assert(omniUsdLiveWaitForPendingUpdates(5000)==OMNI_USD_LIVE_OK);
        } else if (mode=="actor-local-shared") {
            seen.wait([&] { return seen.state.applied==3 || failed(seen.state); });
            assert(actor->snapshot().applied==3 && !actor->snapshot().retained);
            assert(omniUsdLiveWaitForPendingUpdates(5000)==OMNI_USD_LIVE_OK);
            std::lock_guard<std::mutex> lock(seen.mutex); assert(asyncCalls==3);
        } else {
            seen.wait([&] { return failed(seen.state); });
            queued(); assert(actor->snapshot().result==expected);
            assert(omniUsdLiveWaitForPendingUpdates(0)==expected);
        }
    } else if (syncTimingTest) {
        seen.wait([&] { return seen.state.gate == LayerProgress::Gate::Ready || failed(seen.state); });
        assert(!failed(actor->snapshot())); author();
        seen.wait([&] { return failed(seen.state); });
        assert(actor->snapshot().result == OMNI_USD_LIVE_INTERNAL_ERROR && actor->snapshot().applied == 1);
    } else if (asyncTest) {
        seen.wait([&] { return seen.state.gate==LayerProgress::Gate::Ready || failed(seen.state); });
        assert(!failed(actor->snapshot()));
        author();
        seen.wait([&] { return seen.entered; });
        std::shared_ptr<AsyncApplication> task;
        { std::lock_guard<std::mutex> lock(seen.mutex); task=std::move(asyncApplication); assert(asyncCalls==1); }
        assert(task && task->work.last==2 && task->effect->bytes()=="local-0" &&
               *std::static_pointer_cast<const std::string>(task->effect->intent)=="local-0");
        assert(actor->snapshot().applied==1);
        if (mode=="actor-async-queue-throw" || mode=="actor-async-run-throw" || captureOwner) {
            seen.wait([&] { return failed(seen.state); });
            assert(omniUsdLiveWaitForPendingUpdates(0)==OMNI_USD_LIVE_INTERNAL_ERROR);
        } else {
            assert(omniUsdLiveWaitForPendingUpdates(0)==OMNI_USD_LIVE_TIMEOUT);
            assert(omniUsdLiveWaitForPendingUpdates(25)==OMNI_USD_LIVE_TIMEOUT);
        }
        rpc::PingRequest ping;
        auto request=submitUnary(*runtime,endpoint,ping,UnaryCallback<rpc::PingResponse>{});
        assert(request->wait(2000)==OMNI_USD_LIVE_OK);
        assert(runtime->requests->release(request->handle)==OMNI_USD_LIVE_OK);
        bool ran=false;
        if (applicationTimingTest) {
            const auto metric = runtime->diagnostics->snapshot(0, 0).metrics.applications[1];
            assert(metric.current == 1 && metric.admitted == 1 && !metric.retired && !metric.body.count);
            if (mode != "actor-async-run-throw") applicationMs = 13;
        }
        if (mode=="actor-async-cancel" || mode=="actor-async-queue-throw" || mode=="actor-async-run-throw" || captureOwner) {
            if (mode=="actor-async-cancel") assert(actor->close(0)==OMNI_USD_LIVE_TIMEOUT);
            task->completion->run([&] { ran=true; });
            assert(!ran && actor->snapshot().applied==1);
            // Check rejection suppressed queued work before initiating close;
            // cancellation itself must not make this regression pass.
            if (mode=="actor-async-queue-throw") assert(actor->close(0)==OMNI_USD_LIVE_TIMEOUT);
        } else {
            task->completion->run([&] {
                ran=true;
                if (applicationTimingTest) applicationMs = 24;
                assert(DispatchExecutor::isWorker()); // Host-thread self-wait scope.
                assert(omniUsdLiveWaitForPendingUpdates(1000)==OMNI_USD_LIVE_BUSY);
                if (mode=="actor-async-close") assert(actor->close(1000)==OMNI_USD_LIVE_BUSY);
                if (mode=="actor-async-fail") throw OmniUsdLiveResult(OMNI_USD_LIVE_INTERNAL_ERROR);
                std::lock_guard<std::mutex> lock(seen.mutex);
                seen.applied.emplace_back(task->work.last,task->effect->bytes());
            });
            assert(ran && !DispatchExecutor::isWorker());
            // A second report cannot change a successfully claimed result.
            task->completion->complete(OMNI_USD_LIVE_INTERNAL_ERROR);
            bool repeated=false; task->completion->run([&] { repeated=true; });
            assert(!repeated && actor->snapshot().applied==1);
        }
        if (mode=="actor-async-close" || mode=="actor-async-cancel") {
            // The held application owner keeps the closing actor running.
            const auto passes = closing.current();
            assert(actor->close(0)==OMNI_USD_LIVE_TIMEOUT);
            closing.waitBeyond(passes);
            assert(actor->close(0)==OMNI_USD_LIVE_TIMEOUT);
            assert(omniUsdLiveWaitForPendingUpdates(0)==OMNI_USD_LIVE_CANCELLED);
        } else if (mode=="actor-async-release" || captureOwner) {
            assert(omniUsdLiveWaitForPendingUpdates(25)==(captureOwner?OMNI_USD_LIVE_INTERNAL_ERROR:OMNI_USD_LIVE_TIMEOUT));
            // Hold a real host task capture destructor after publication ran.
            // Neither progress nor drain may retire until this cleanup ends.
            std::thread retire([owned=std::move(task)]() mutable { owned.reset(); });
            {
                std::unique_lock<std::mutex> lock(asyncRetirement->mutex);
                assert(asyncRetirement->changed.wait_for(lock,std::chrono::seconds(3),[&] { return asyncRetirement->entered; }));
            }
            assert(actor->snapshot().applied==1);
            { std::lock_guard<std::mutex> lock(seen.mutex); assert(asyncCalls==1); }
            assert(omniUsdLiveWaitForPendingUpdates(25)==(captureOwner?OMNI_USD_LIVE_INTERNAL_ERROR:OMNI_USD_LIVE_TIMEOUT));
            // Close cannot complete while the host task capture is retiring.
            if (captureOwner) {
                const auto passes = closing.current();
                assert(actor->close(0)==OMNI_USD_LIVE_TIMEOUT);
                closing.waitBeyond(passes);
                assert(actor->close(0)==OMNI_USD_LIVE_TIMEOUT);
            }
            { std::lock_guard<std::mutex> lock(asyncRetirement->mutex); asyncRetirement->release=true; asyncRetirement->changed.notify_all(); }
            retire.join();
        }
        if (applicationTimingTest) {
            const auto metric = runtime->diagnostics->snapshot(0, 0).metrics.applications[1];
            assert(metric.current == 1 && !metric.retired); // Finished body is not owner retirement.
            applicationMs = 41;
        }
        task.reset();
        if (mode=="actor-async-fail" || mode=="actor-async-cancel" || mode=="actor-async-queue-throw" ||
            mode=="actor-async-run-throw" || captureOwner) {
            seen.wait([&] { return failed(seen.state); });
            assert(actor->snapshot().result==expected && actor->snapshot().applied==1);
            assert(omniUsdLiveWaitForPendingUpdates(0)==expected);
        } else if (mode=="actor-async-release") {
            seen.wait([&] { return seen.state.applied==4 || failed(seen.state); });
            assert(actor->snapshot().applied==4 && !actor->snapshot().retained);
            assert(omniUsdLiveWaitForPendingUpdates(5000)==OMNI_USD_LIVE_OK);
            std::lock_guard<std::mutex> lock(seen.mutex); assert(asyncCalls==3);
        }
    } else if (mode == "actor-jitter-reconnect") {
        // The second read's catch-up response is gated by the reference service
        // on a new automatic Ping. The old generation's Ping remains held and
        // is cancelled, suppressing its callback; no authoring or host wake is
        // available to rescue a stranded cancellation/clock-sampling timer.
        seen.wait([&] { return (seen.state.generation == 2 && seen.state.gate == LayerProgress::Gate::Ready) ||
                               failed(seen.state); });
        const auto state = actor->snapshot();
        assert(!failed(state) && state.generation == 2 && state.covered == 1 && state.applied == 1 && !state.retained);
        std::cout << "actor-new-generation-ping" << std::endl;
    } else if (mode == "actor-reconnect-reset") {
        // Reads 1, 5 and 9 catch up; 1 and 5 then end, and the three reads
        // after each fail. Readiness restarts the count, so each failure
        // sequence waits in the windows of attempts 1 to 4 again.
        seen.wait([&] { return (seen.state.generation == 9 && seen.state.gate == LayerProgress::Gate::Ready) ||
                               failed(seen.state); });
        const auto state = actor->snapshot();
        assert(!failed(state) && state.generation == 9 && state.covered == 1 && state.applied == 1);
        const auto history = runtime->diagnostics->snapshot(endpoint, 1024);
        std::vector<uint64_t> delays;
        for (const auto& entry : history.records) {
            if (entry.subjectId != subject || entry.eventCode != OMNI_USD_LIVE_RECORD_RETRY_SCHEDULED) continue;
            for (uint32_t i = 0; i < entry.fieldCount; ++i)
                if (entry.fields[i].key == DiagnosticKey::DelayMs) delays.push_back(entry.fields[i].value);
        }
        // [0, 2 x min(20, 5 x 2^(n-1))) for attempts 1 to 4, twice.
        const uint64_t windows[] = {10, 20, 40, 40, 10, 20, 40, 40};
        assert(delays.size() == std::size(windows));
        for (size_t i = 0; i < delays.size(); ++i) assert(delays[i] < windows[i]);
    } else if (mode == "actor-drain-clean-close") {
        seen.wait([&] { return seen.state.gate == LayerProgress::Gate::Ready || failed(seen.state); });
        assert(!failed(actor->snapshot()));
        // Idle is transient while the scheduler can admit another actor epoch.
        // Hold admission through close; already admitted workers and Read owners
        // can finish without the scheduler mutex being held.
        struct SchedulerGate final : ScheduledOperation {
            std::mutex mutex;
            std::condition_variable changed;
            bool entered = false, released = false;
            ScheduledStep advance(std::chrono::steady_clock::time_point) noexcept override
            {
                std::unique_lock<std::mutex> lock(mutex);
                entered = true; changed.notify_all();
                assert(changed.wait_for(lock, std::chrono::seconds(6), [&] { return released; }));
                return {true, {}};
            }
        };
        const auto gate = std::make_shared<SchedulerGate>();
        runtime->auth->reserve(gate);
        runtime->auth->activate(*gate);
        {
            std::unique_lock<std::mutex> lock(gate->mutex);
            assert(gate->changed.wait_for(lock, std::chrono::seconds(6), [&] { return gate->entered; }));
        }
        const auto idleDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        for (;;) {
            const auto transport = actor->transportSnapshot();
            if (!transport.workerScheduled && !transport.queuedReadFrames && !transport.timingPending) break;
            assert(std::chrono::steady_clock::now() < idleDeadline); std::this_thread::yield();
        }
        // Worker-idle alone does not retire the Read worker's local drain owner.
        assert(omniUsdLiveWaitForPendingUpdates(5000) == OMNI_USD_LIVE_OK);
        const auto state = actor->snapshot();
        assert(state.applied == 1 && state.covered == 1 && state.retained == 0 && state.queued == 0 && !state.issued);
        assert(omniUsdLiveWaitForPendingUpdates(0) == OMNI_USD_LIVE_OK);
        assert(actor->close(0) == OMNI_USD_LIVE_TIMEOUT);
        {
            std::lock_guard<std::mutex> lock(gate->mutex);
            gate->released = true; gate->changed.notify_all();
        }
        assert(actor->close(5000) == OMNI_USD_LIVE_OK);
        // The cancelled future Read never joined the drained capture. Its
        // rejected EOF ticket must not poison this still-owned empty state.
        assert(omniUsdLiveWaitForPendingUpdates(0) == OMNI_USD_LIVE_OK);
    } else if (jitterTest) {
        // The reference sends a real initial payload then a peer delta. Their
        // timestamps are zero, fixing fallback holdback independently of Ping.
        // Readiness and bootstrap application must beat the 30-second holdback.
        seen.wait([&] { return (seen.state.initialLatestPublished && seen.state.covered == 2 &&
                               seen.state.applied == 1) || failed(seen.state); });
        assert(actor->snapshot().gate == LayerProgress::Gate::Ready);
        assert(actor->snapshot().covered == 2 && actor->snapshot().applied == 1);
        // Jitter deliberately retains peer sequence 2, but this layer has no
        // authored obligations. A clean save must not wait for peer holdback.
        assert(actor->snapshot().pendingWrites == 0);
        assert(actor->save(0, false) == OMNI_USD_LIVE_OK);
        DrainRegistry::Capture earlierCapture;
        if (mode == "actor-drain-capture") {
            const auto idleDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
            while (actor->transportSnapshot().workerScheduled) {
                assert(std::chrono::steady_clock::now() < idleDeadline); std::this_thread::yield();
            }
            // Freeze the peer's obligation before admitting the later local
            // edit. Its held apply callback must not extend this capture.
            earlierCapture = runtime->drains->capture();
        }
        const std::string local = "local-after-peer";
        const auto localId = actor->submit(local, std::make_shared<const std::string>(local));
        submitted.push_back(localId);
        {
            const auto status = layerStatus();
            assert(status.pending_write_count == 1 && status.pending_bytes == local.size() && status.has_unconfirmed_edits);
            ++immediateChecks;
        }
        seen.wait([&] { return (seen.state.covered == 3 && seen.state.issued == 0) || failed(seen.state); });
        auto state = actor->snapshot();
        assert(!failed(state) && state.covered == 3 && state.applied == 1 && state.retained == 2);
        const auto waiting = runtime->diagnostics->snapshot(0, 0).metrics.jitter;
        const auto starts = mode == "actor-jitter-preempt" ? 2u : 1u;
        assert(waiting.current == 1 && waiting.highWater == 1 && waiting.started == starts && waiting.completed == starts - 1);
        assert(waiting.requestedWait.count == starts && waiting.requestedWait.sumMs <= starts * 30000);
        assert(waiting.targetHold.count == starts && waiting.targetHold.sumMs == starts * 30000);
        assert(waiting.unsampledStarts == starts && !waiting.sampleAge.count && !waiting.frozenStarts);
        {
            // A peer frame and an acknowledged local edit remain retained.
            // Only the local intent contributes to pending authoring counters.
            const auto status = layerStatus();
            assert(status.state == OMNI_USD_LIVE_STATE_READY && status.result == OMNI_USD_LIVE_OK);
            assert(status.pending_write_count == 1 && status.pending_bytes == local.size() && status.has_unconfirmed_edits);
            acknowledgedBytes = status.pending_bytes;
            assert(endpointStatus().failed_layers == 0 && endpointStatus().blocked_layers == 0);
        }
        auto verifyRetained = [&] {
            const auto work = actor->retained();
            assert(work.size() == 2);
            const auto peer = std::find_if(work.begin(), work.end(), [](const auto& item) { return item.first == 2; });
            const auto own = std::find_if(work.begin(), work.end(), [&](const auto& item) { return item.id == localId; });
            assert(peer != work.end() && own != work.end());
            assert(peer->disposition == LayerProgress::Disposition::Peer && !peer->applied);
            assert(own->disposition == LayerProgress::Disposition::Acknowledged && own->first == 3 && !own->applied);
            const auto peerEffect = std::static_pointer_cast<const LayerActor::Effect>(peer->owner);
            const auto ownEffect = std::static_pointer_cast<const LayerActor::Effect>(own->owner);
            assert(peerEffect->bytes() == "peer-delayed" && peerEffect->incoming && !peerEffect->incoming->firstResponse);
            assert(ownEffect->bytes() == local && *std::static_pointer_cast<const std::string>(ownEffect->intent) == local);
        };
        verifyRetained();
        if (mode == "actor-jitter-budget") {
            assert(actor->transportSnapshot().timingPending);
            std::cout << "actor-write-settled-with-held-ping" << std::endl;
        }
        auto ping = [&] {
            auto request = submitUnary(*runtime, endpoint, rpc::PingRequest{}, UnaryCallback<rpc::PingResponse>{});
            assert(request->wait(2000) == OMNI_USD_LIVE_OK);
            OmniUsdLiveResult result;
            assert(request->result(&result) == OMNI_USD_LIVE_OK && result == OMNI_USD_LIVE_OK);
            assert(runtime->requests->release(request->handle) == OMNI_USD_LIVE_OK);
        };
        ping(); // Delayed layer application must not sleep on the shared CQ.
        {
            std::lock_guard<std::mutex> lock(seen.mutex);
            assert(seen.applied.size() == 1 && seen.applied.front().first == 1 &&
                   seen.applied.front().second == "opaque-base");
        }
        assert(actor->snapshot().applied == 1); // The acknowledged write cannot pass the peer.
        assert(omniUsdLiveWaitForPendingUpdates(0) == OMNI_USD_LIVE_TIMEOUT);
        verifyRetained(); // Waiting does not cancel or discard delayed work.
        if (mode == "actor-jitter-fail") {
            assert(actor->failPublication(OMNI_USD_LIVE_INTERNAL_ERROR) == OMNI_USD_LIVE_OK);
            seen.wait([&] { return failed(seen.state); });
            // Close keeps both retained works.
            assert(actor->close(5000) == OMNI_USD_LIVE_OK);
            assert(actor->snapshot().applied == 1 && actor->retained().size() == 2);
            assert(!runtime->diagnostics->snapshot(0, 0).metrics.jitter.current);
            verifyRetained();
        } else if (mode != "actor-jitter-close") {
            auto jitter = record<OmniUsdLiveJitterParams>(); // max=0 must wake a scheduled 30s delay.
            assert(omniUsdLiveConfigureJitterReduction(endpoint, &jitter) == OMNI_USD_LIVE_OK);
            if (mode == "actor-drain-capture") {
                seen.wait([&] { return seen.entered; });
                assert(actor->snapshot().applied == 2 && actor->snapshot().retained == 1);
                assert(runtime->drains->wait(earlierCapture,
                    std::chrono::steady_clock::now() + std::chrono::seconds(2)) == OMNI_USD_LIVE_OK);
                assert(omniUsdLiveWaitForPendingUpdates(0) == OMNI_USD_LIVE_TIMEOUT);
                { std::lock_guard<std::mutex> lock(seen.mutex); seen.release = true; seen.changed.notify_all(); }
            }
            seen.wait([&] { return seen.state.applied == 3 || failed(seen.state); });
            assert(actor->snapshot().applied == 3 && actor->snapshot().retained == 0);
            assert(omniUsdLiveWaitForPendingUpdates(5000) == OMNI_USD_LIVE_OK);
        } else {
            assert(actor->close(3000) == OMNI_USD_LIVE_OK);
            assert(omniUsdLiveWaitForPendingUpdates(0) == OMNI_USD_LIVE_CANCELLED);
            assert(actor->snapshot().applied == 1 && actor->snapshot().covered == 3);
            verifyRetained();
            size_t closedCallbacks;
            { std::lock_guard<std::mutex> lock(seen.mutex); closedCallbacks = seen.history.size(); }
            auto jitter = record<OmniUsdLiveJitterParams>();
            assert(omniUsdLiveConfigureJitterReduction(endpoint, &jitter) == OMNI_USD_LIVE_OK);
            ping(); // Exercise a scheduler wake and CQ after close has established quiescence.
            {
                std::lock_guard<std::mutex> lock(seen.mutex);
                assert(seen.history.size() == closedCallbacks && seen.applied.size() == 1);
            }
            verifyRetained();
        }
    } else if (mode == "actor-backlog") {
        seen.wait([&] { return seen.entered; });
        assert(actor->snapshot().gate == LayerProgress::Gate::Ready);
        assert(actor->snapshot().covered == 1 && actor->snapshot().retained == 0);
        assert(omniUsdLiveWaitForPendingUpdates(0) == OMNI_USD_LIVE_TIMEOUT);
        assert(omniUsdLiveWaitForPendingUpdates(25) == OMNI_USD_LIVE_TIMEOUT);
        author();
        const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        for (;;) {
            const auto transport = actor->transportSnapshot();
            if (transport.readEnded && transport.queuedReadFrames >= 32) break;
            assert(std::chrono::steady_clock::now() < until); std::this_thread::yield();
        }
        { std::lock_guard<std::mutex> lock(seen.mutex); seen.release = true; seen.changed.notify_all(); }
        seen.wait([&] { return seen.state.applied == 37 || failed(seen.state); });
        assert(actor->snapshot().applied == 37);
        assert(omniUsdLiveWaitForPendingUpdates(5000) == OMNI_USD_LIVE_OK);
    } else if (mode == "actor-terminal") {
        seen.wait([&] { return failed(seen.state); });
    } else if (sequenceRegression) {
        // A caught-up header below the resume cursor is a server sequence
        // regression: FAILED MALFORMED_STREAM, queued intents retained, one
        // correlated stream_disconnected record, no retry.
        seen.wait([&] { return failed(seen.state); });
        const auto state = actor->snapshot();
        assert(state.gate == LayerProgress::Gate::Failed && state.result == OMNI_USD_LIVE_MALFORMED_STREAM);
        assert(state.applied == 3 && state.readSequence == 3 && !state.issued);
        const auto retained = actor->retained();
        assert(retained.size() == 3);
        for (size_t i = 0; i < retained.size(); ++i) {
            assert(retained[i].id == submitted[i] && retained[i].disposition == LayerProgress::Disposition::Queued);
            const auto effect = std::static_pointer_cast<const LayerActor::Effect>(retained[i].owner);
            assert(effect->authored == "local-" + std::to_string(i) &&
                   *std::static_pointer_cast<const std::string>(effect->intent) == effect->authored);
        }
        const auto status = layerStatus();
        assert(status.state == OMNI_USD_LIVE_STATE_FAILED && status.result == OMNI_USD_LIVE_MALFORMED_STREAM);
        assert(status.pending_write_count == 3 && status.has_unconfirmed_edits);
        assert(status.operation == OMNI_USD_LIVE_OPERATION_READ && status.incident_id);
        incident = status.incident_id;
        const auto history = runtime->diagnostics->snapshot(endpoint, 1024);
        auto field = [](const DiagnosticEntry& entry, DiagnosticKey key) {
            for (uint32_t i = 0; i < entry.fieldCount; ++i)
                if (entry.fields[i].key == key) return entry.fields[i].value;
            assert(false); return uint64_t(0);
        };
        unsigned disconnects = 0;
        for (const auto& entry : history.records) {
            assert(entry.subjectId != subject || entry.eventCode != OMNI_USD_LIVE_RECORD_RETRY_SCHEDULED);
            if (entry.subjectId != subject || entry.eventCode != OMNI_USD_LIVE_RECORD_STREAM_DISCONNECTED) continue;
            ++disconnects;
            assert(entry.endpoint == endpoint && !entry.resource[0]);
            assert(field(entry, DiagnosticKey::Operation) == OMNI_USD_LIVE_OPERATION_READ &&
                   field(entry, DiagnosticKey::Result) == OMNI_USD_LIVE_MALFORMED_STREAM &&
                   field(entry, DiagnosticKey::Generation) == 1 &&
                   field(entry, DiagnosticKey::ResumeSequence) == 3 &&
                   field(entry, DiagnosticKey::ReadSequence) == 3 &&
                   field(entry, DiagnosticKey::AppliedSequence) == 3 &&
                   field(entry, DiagnosticKey::Request) && field(entry, DiagnosticKey::Attempt));
        }
        assert(disconnects == 1);
    } else if (uncertain) {
        seen.wait([&] { return failed(seen.state); });
        if (mode=="actor-capture-changed") {
            seen.wait([&] { return seen.entered; });
            // Close cannot complete while the terminal Changed callback runs:
            // the worker is inside this pass, so it stays scheduled and cannot
            // reach its closing step until the callback returns.
            assert(actor->transportSnapshot().workerScheduled);
            assert(actor->close(0)==OMNI_USD_LIVE_TIMEOUT);
            assert(actor->transportSnapshot().workerScheduled);
            { std::lock_guard<std::mutex> lock(seen.mutex); seen.release=true; seen.changed.notify_all(); }
        }
        assert(actor->snapshot().gate == LayerProgress::Gate::Failed &&
               actor->snapshot().result == OMNI_USD_LIVE_OUTCOME_UNKNOWN);
        assert(omniUsdLiveWaitForPendingUpdates(5000) == OMNI_USD_LIVE_OUTCOME_UNKNOWN);
        const auto retained = actor->retained();
        assert(retained.size() == 3 && retained.front().disposition == LayerProgress::Disposition::Unknown);
        const auto status = layerStatus();
        assert(status.result == OMNI_USD_LIVE_OUTCOME_UNKNOWN);
        if (mode=="actor-uncertain") assert(status.state == OMNI_USD_LIVE_STATE_FAILED);
        assert(status.pending_write_count == 3 && status.pending_bytes == 21 && status.has_unconfirmed_edits);
        assert(status.incident_id && status.recovery_action == OMNI_USD_LIVE_RECOVERY_PRESERVE_AND_REOPEN);
        incident = status.incident_id;
        if (mode=="actor-uncertain") {
            // A closing layer does not count as a failed child, so this mode,
            // not actor-capture-changed, pins the open failed incident.
            const auto aggregate = endpointStatus();
            assert(aggregate.failed_layers == 1 && aggregate.blocked_layers == 0);
            // Unrelated successful transport/auth activity cannot resolve this incident.
            auto authState = record<OmniUsdLiveAuthState>();
            assert(omniUsdLiveGetAuthState(endpoint, &authState) == OMNI_USD_LIVE_OK);
            uint64_t generation;
            assert(omniUsdLiveSetAccessToken(endpoint, authState.generation, "synthetic-writer", 0, &generation) == OMNI_USD_LIVE_OK);
            auto ping = submitUnary(*runtime, endpoint, rpc::PingRequest{}, UnaryCallback<rpc::PingResponse>{});
            assert(ping->wait(2000) == OMNI_USD_LIVE_OK);
            assert(runtime->requests->release(ping->handle) == OMNI_USD_LIVE_OK);
            assert(layerStatus().incident_id == incident && layerStatus().result == OMNI_USD_LIVE_OUTCOME_UNKNOWN);
        }
        for (unsigned i = 0; i < 3; ++i) {
            auto effect = std::static_pointer_cast<const LayerActor::Effect>(retained[i].owner);
            assert(effect->authored == "local-" + std::to_string(i));
            assert(*std::static_pointer_cast<const std::string>(effect->intent) == effect->authored);
        }
    } else if (localCause) {
        // A local INVALID_ARGUMENT latches while the write is held at the
        // service; the write's later server INVALID_ARGUMENT is not evidence
        // of that local cause and attaches no gRPC status.
        // The harness closes stdin once the service holds the issued write.
        std::string barrier; (void)std::getline(std::cin, barrier);
        assert(actor->snapshot().issued);
        assert(actor->failPublication(OMNI_USD_LIVE_INVALID_ARGUMENT) == OMNI_USD_LIVE_OK);
        auto status = layerStatus();
        assert(status.state == OMNI_USD_LIVE_STATE_FAILED && status.result == OMNI_USD_LIVE_INVALID_ARGUMENT &&
               !status.grpc_status_present);
        std::cout << "actor-local-cause-latched" << std::endl;
        seen.wait([&] { return failed(seen.state) && !seen.state.issued; });
        const auto retained = actor->retained();
        assert(retained.size() == 3 && retained.front().disposition == LayerProgress::Disposition::Rejected);
        status = layerStatus();
        assert(status.state == OMNI_USD_LIVE_STATE_FAILED && status.result == OMNI_USD_LIVE_INVALID_ARGUMENT &&
               !status.grpc_status_present && status.operation == OMNI_USD_LIVE_OPERATION_OPEN);
    } else if (rejectionClosesRead) {
        // This actor has no journal-only application, so a definitive
        // rejection closes the read with the issuance instead of keeping it
        // open to apply ordered work nothing can install.
        seen.wait([&] { return failed(seen.state) && !seen.state.issued; });
        const auto state = actor->snapshot();
        assert(state.gate == LayerProgress::Gate::Failed && state.result == OMNI_USD_LIVE_INVALID_ARGUMENT);
        assert(!state.readingAfterFailure && !state.readStarted);
        const auto retained = actor->retained();
        assert(retained.size() == 3 && retained.front().disposition == LayerProgress::Disposition::Rejected);
        const auto status = layerStatus();
        assert(status.state == OMNI_USD_LIVE_STATE_FAILED && status.result == OMNI_USD_LIVE_INVALID_ARGUMENT &&
               status.grpc_status_present && status.grpc_status == grpc::StatusCode::INVALID_ARGUMENT &&
               status.operation == OMNI_USD_LIVE_OPERATION_UPDATE && status.pending_write_count == 3);
    } else if (authReissue) {
        // UpdateLayer rejected UNAUTHENTICATED: the intent stays pending in
        // AUTH_WAIT and is reissued at most once after a token replacement.
        if (mode == "actor-auth-reissue-fail") {
            seen.wait([&] { return seen.auth > 0; });
            OmniUsdLiveStatus status{};
            statuses.wait([&] {
                const auto event = std::find_if(statuses.events.begin(), statuses.events.end(), [&](const auto& value) {
                    return value.status.subject_id == subject && value.status.state == OMNI_USD_LIVE_STATE_AUTH_WAIT;
                });
                if (event != statuses.events.end()) status = event->status;
                return event != statuses.events.end();
            }, std::chrono::seconds(5));
            assert(layerStatus().state == OMNI_USD_LIVE_STATE_AUTH_WAIT); // Refresh is held until FailAuthRefresh.
            assert(status.result == OMNI_USD_LIVE_UNAUTHENTICATED && status.grpc_status_present &&
                   status.grpc_status == grpc::StatusCode::UNAUTHENTICATED &&
                   status.operation == OMNI_USD_LIVE_OPERATION_UPDATE &&
                   status.recovery_action == OMNI_USD_LIVE_RECOVERY_REFRESH_AUTH &&
                   status.pending_write_count == 3 && status.has_unconfirmed_edits);
            const auto waiting = actor->retained();
            assert(waiting.size() == 3 && waiting.front().id == submitted.front() && waiting.front().authReissued &&
                   waiting.front().disposition == LayerProgress::Disposition::Queued);
            auto state = record<OmniUsdLiveAuthState>();
            assert(omniUsdLiveGetAuthState(endpoint, &state) == OMNI_USD_LIVE_OK && state.challenge_id);
            assert(omniUsdLiveFailAuthRefresh(endpoint, state.challenge_id) == OMNI_USD_LIVE_OK);
        }
        if (expected == OMNI_USD_LIVE_OK) {
            seen.wait([&] { return seen.state.applied == 4 || failed(seen.state); });
            assert(actor->snapshot().applied == 4 && actor->snapshot().retained == 0);
        } else {
            seen.wait([&] { return failed(seen.state); });
            const auto state = actor->snapshot();
            assert(state.result == expected && state.applied == 1 && state.pendingWrites == 3);
            const auto retained = actor->retained();
            assert(retained.size() == 3);
            for (size_t i = 0; i < retained.size(); ++i) {
                assert(retained[i].id == submitted[i] && retained[i].authReissued == !i);
                // Refresh exhaustion never reissues; a second rejection retires it.
                assert(retained[i].disposition == (i || mode == "actor-auth-reissue-fail" ?
                    LayerProgress::Disposition::Queued : LayerProgress::Disposition::Rejected));
            }
            const auto status = layerStatus();
            assert(status.state == OMNI_USD_LIVE_STATE_FAILED && status.result == expected &&
                   status.grpc_status_present && status.grpc_status == grpc::StatusCode::UNAUTHENTICATED &&
                   status.operation == OMNI_USD_LIVE_OPERATION_UPDATE && status.pending_write_count == 3);
        }
    } else {
        seen.wait([&] { return seen.state.gate == LayerProgress::Gate::Ready || failed(seen.state); });
        assert(!failed(actor->snapshot()));
        if (mode == "actor-auth-fail" || mode == "actor-auth-timeout" || authExpiry) {
            if(authExpiry) {
                if(mode=="actor-auth-expiry-held") seen.wait([&] { return seen.entered; });
                auto state=record<OmniUsdLiveAuthState>();
                assert(omniUsdLiveGetAuthState(endpoint,&state)==OMNI_USD_LIVE_OK);
                uint64_t generation=0;
                assert(omniUsdLiveSetAccessToken(endpoint,state.generation,"synthetic-writer",UINT64_MAX,
                                                 &generation)==OMNI_USD_LIVE_OK);
                const auto retained=runtime->findEndpoint(endpoint);
                uint64_t serial=0;
                {
                    std::lock_guard<std::mutex> lock(retained->mutex);
                    // Supply expiry directly under the controller's clock seam.
                    // Every real timer scan also sees this token as expired.
                    retained->expiresAtMs=1;
                    serial=retained->authFailureSerial;
                    runtime->auth->evaluate(*retained,1,std::chrono::steady_clock::now());
                    assert(!retained->authFailed && retained->challenge.challenge_id);
                }
                seen.wait([&] { return seen.auth>0; });
                if(mode=="actor-auth-expiry-idle") {
                    // Replacement wakes the actor: establish retirement AFTER
                    // that wake and challenge delivery, not before replacement.
                    const auto watchdog=std::chrono::steady_clock::now()+std::chrono::seconds(3);
                    while(actor->transportSnapshot().workerScheduled) {
                        assert(std::chrono::steady_clock::now()<watchdog); std::this_thread::yield();
                    }
                }
                {
                    std::lock_guard<std::mutex> lock(retained->mutex);
                    assert(!retained->authFailed && retained->authFailureSerial==serial &&
                           retained->authDeadline!=std::chrono::steady_clock::time_point{});
                    runtime->auth->evaluate(*retained,1,retained->authDeadline);
                    assert(retained->authFailed && retained->authFailureSerial==serial+1);
                }
                if(mode=="actor-auth-expiry-held") {
                    std::lock_guard<std::mutex> lock(seen.mutex); seen.release=true; seen.changed.notify_all();
                }
            } else {
                // actor-auth-timeout lets a real token expire on purpose: it is the auth-watchdog probe for test-intermittent, and no correctness test runs it.
                if (mode == "actor-auth-timeout") {
                    const auto idleUntil = std::chrono::steady_clock::now() + std::chrono::seconds(3);
                    while (actor->transportSnapshot().workerScheduled) {
                        assert(std::chrono::steady_clock::now() < idleUntil); std::this_thread::yield();
                    }
                }
                auto state = record<OmniUsdLiveAuthState>();
                assert(omniUsdLiveGetAuthState(endpoint, &state) == OMNI_USD_LIVE_OK);
                uint64_t generation;
                assert(omniUsdLiveSetAccessToken(endpoint, state.generation,
                    mode == "actor-auth-timeout" ? "synthetic-writer" : "", mode == "actor-auth-timeout" ? wallTimeMs() + 50 : 0,
                    &generation) == OMNI_USD_LIVE_OK);
                if (mode == "actor-auth-fail") {
                    seen.wait([&] { return seen.auth > 0 && seen.entered; });
                    assert(omniUsdLiveGetAuthState(endpoint, &state) == OMNI_USD_LIVE_OK && state.challenge_id);
                    assert(omniUsdLiveFailAuthRefresh(endpoint, state.challenge_id) == OMNI_USD_LIVE_OK);
                    assert(omniUsdLiveSetAccessToken(endpoint, generation, "synthetic-writer", 0, &generation) == OMNI_USD_LIVE_OK);
                    { std::lock_guard<std::mutex> lock(seen.mutex); seen.release = true; seen.changed.notify_all(); }
                }
            }
            bool terminalObserved=false;
            {
                std::unique_lock<std::mutex> lock(seen.mutex);
                terminalObserved=seen.changed.wait_for(lock,std::chrono::seconds(6),[&] { return failed(seen.state); });
            }
            if(!terminalObserved) {
                const auto progress=actor->snapshot();
                const auto transport=actor->transportSnapshot();
                const auto retained=runtime->findEndpoint(endpoint);
                std::lock_guard<std::mutex> lock(retained->mutex);
                std::cerr << "auth-terminal-watchdog mode=" << mode << " wall=" << wallTimeMs()
                    << " expiry=" << retained->expiresAtMs << " auth_failed=" << retained->authFailed
                    << " auth_failure_serial=" << retained->authFailureSerial
                    << " auth_deadline_remaining_ms=" << std::chrono::duration_cast<std::chrono::milliseconds>(
                        retained->authDeadline-std::chrono::steady_clock::now()).count()
                    << " gate=" << unsigned(progress.gate) << " result=" << progress.result
                    << " worker_scheduled=" << transport.workerScheduled << " read_ended=" << transport.readEnded
                    << " timing_pending=" << transport.timingPending << std::endl;
            }
            assert(terminalObserved);
            if(authExpiry) {
                // The active read retires without explicit close: the ordinary
                // request slot drains before close is initiated.
                const auto readDeadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
                while(runtime->diagnostics->snapshot(0,0).metrics.requestSlots[0].current) {
                    assert(std::chrono::steady_clock::now()<readDeadline); std::this_thread::yield();
                }
                // Close keeps the terminal progress.
                assert(actor->close(5000)==OMNI_USD_LIVE_OK);
                const auto progress=actor->snapshot();
                assert(progress.result==OMNI_USD_LIVE_UNAUTHENTICATED &&
                       progress.applied==1 && actor->retained().empty());
            }
        } else {
            if (!queuedInitially) author();
            if (mode == "actor-held") {
                seen.wait([&] { return seen.entered; });
                assert(omniUsdLiveWaitForPendingUpdates(0) == OMNI_USD_LIVE_TIMEOUT);
                assert(omniUsdLiveWaitForPendingUpdates(25) == OMNI_USD_LIVE_TIMEOUT);
                assert(actor->close(0) == OMNI_USD_LIVE_TIMEOUT);
                assert(omniUsdLiveWaitForPendingUpdates(0) == OMNI_USD_LIVE_CANCELLED);
                rpc::PingRequest ping;
                auto request = submitUnary(*runtime, endpoint, ping, UnaryCallback<rpc::PingResponse>{});
                assert(request->wait(2000) == OMNI_USD_LIVE_OK);
                assert(runtime->requests->release(request->handle) == OMNI_USD_LIVE_OK);
                { std::lock_guard<std::mutex> lock(seen.mutex); seen.release = true; seen.changed.notify_all(); }
            } else if (mode == "actor-drain-changed") {
                seen.wait([&] { return seen.entered; });
                assert(actor->snapshot().applied == 4 && actor->snapshot().retained == 0);
                assert(actor->snapshot().pendingWrites == 0);
                assert(actor->save(0, false) == OMNI_USD_LIVE_BUSY);
                assert(actor->save(25) == OMNI_USD_LIVE_TIMEOUT);
                assert(omniUsdLiveWaitForPendingUpdates(0) == OMNI_USD_LIVE_TIMEOUT);
                assert(omniUsdLiveWaitForPendingUpdates(25) == OMNI_USD_LIVE_TIMEOUT);
                { std::lock_guard<std::mutex> lock(seen.mutex); seen.release = true; seen.changed.notify_all(); }
                assert(actor->save(5000) == OMNI_USD_LIVE_OK);
                assert(omniUsdLiveWaitForPendingUpdates(5000) == OMNI_USD_LIVE_OK);
            } else {
                if (mode == "actor-recover-issued") {
                    seen.wait([&] { return seen.state.generation >= 2 && seen.state.issued &&
                        seen.state.gate == LayerProgress::Gate::Recovering; });
                    assert(actor->snapshot().queued == 2);
                    std::cout << "actor-recovery-paused" << std::endl;
                }
                seen.wait([&] { return seen.state.applied == 4 || failed(seen.state); });
                assert(actor->snapshot().applied == 4 && actor->snapshot().retained == 0);
            }
        }
    }
    if (failed(actor->snapshot())) {
        const auto status = layerStatus();
        assert(status.result == expected && status.incident_id);
        if (mode=="actor-async-cancel" || mode=="actor-async-queue-throw" || mode=="actor-local-cancel" ||
            mode=="actor-jitter-fail" || authExpiry || captureOwner || mode=="actor-capture-changed") {
            // These cases already initiated close, while holding the host task
            // or to observe the settled terminal actor. Closing subjects retain
            // failure evidence and work, but cease to count as active failed
            // children in endpoint aggregates.
            assert(status.state==OMNI_USD_LIVE_STATE_CLOSING || status.state==OMNI_USD_LIVE_STATE_CLOSED);
            const unsigned pending=authExpiry ? 0 : mode=="actor-capture-changed" ? 3 : 1;
            assert(status.pending_write_count==pending && bool(status.has_unconfirmed_edits)==bool(pending));
            assert(endpointStatus().failed_layers==0);
        } else assert(endpointStatus().failed_layers==1);
        if (!incident) incident = status.incident_id;
    }
    const bool terminalClose = uncertain || captureOwner || mode=="actor-terminal";
    std::vector<LayerProgress::Work> beforeClose;
    if (terminalClose) {
        // captureOwner and actor-capture-changed initiated close while a host
        // owner was held; the others are still FAILED.
        const auto state=layerStatus().state;
        if (captureOwner || mode=="actor-capture-changed")
            assert(state==OMNI_USD_LIVE_STATE_CLOSING || state==OMNI_USD_LIVE_STATE_CLOSED);
        else assert(state==OMNI_USD_LIVE_STATE_FAILED);
        beforeClose=actor->retained();
    }
    assert(actor->close(5000) == OMNI_USD_LIVE_OK);
    if (terminalClose) {
        // Close waits for the terminal actor to settle and keeps its progress
        // and every retained work record unchanged.
        const auto progress=actor->snapshot();
        assert(progress.result==expected && progress.applied==1);
        const auto after=actor->retained();
        assert(after.size()==beforeClose.size());
        for (size_t i=0;i<beforeClose.size();++i) {
            const auto& work=after[i];
            const auto& before=beforeClose[i];
            assert(work.id==submitted[i] && work.id==before.id && work.owner==before.owner && work.first==before.first &&
                   work.last==before.last && work.disposition==before.disposition && !work.applied &&
                   work.fingerprint==before.fingerprint && work.kind==before.kind &&
                   work.retainedBytes==before.retainedBytes);
            const auto effect=std::static_pointer_cast<const LayerActor::Effect>(work.owner);
            assert(effect->authored=="local-"+std::to_string(i) && !effect->incoming &&
                   *std::static_pointer_cast<const std::string>(effect->intent)==effect->authored);
        }
        beforeClose.clear(); // Work owners must retire before endpoint quiescence below.
    }
    const auto applicationMetrics = runtime->diagnostics->snapshot(0, 0).metrics.applications;
    const auto jitterMetrics = runtime->diagnostics->snapshot(0, 0).metrics.jitter;
    assert(!jitterMetrics.current && jitterMetrics.started == jitterMetrics.completed);
    if (jitterTest) {
        const bool preempted = mode == "actor-jitter-preempt";
        assert(jitterMetrics.started == (preempted ? 2u : 1u));
        const auto resolution = mode == "actor-jitter-close" ? JitterWaitResolution::Closed :
            mode == "actor-jitter-fail" ? JitterWaitResolution::Failed : JitterWaitResolution::Eligible;
        for (size_t i = 0; i < jitterMetrics.holdByResolution.size(); ++i) {
            const auto count = unsigned(i == static_cast<size_t>(resolution)) +
                unsigned(preempted && i == static_cast<size_t>(JitterWaitResolution::Preempted));
            assert(jitterMetrics.holdByResolution[i].count == count);
            assert(jitterMetrics.postDueByResolution[i].count == count);
        }
    }
    for (const auto& metric : applicationMetrics) assert(!metric.current && metric.admitted == metric.retired);
    if (syncTimingTest) {
        const auto& metric = applicationMetrics[2];
        assert(metric.admitted == 1 && metric.retired == 1 && metric.body.count == 1 && metric.body.sumMs == 11);
        assert(!metric.queue.count && !metric.postBodyRetention.count);
        assert(metric.lifetimeByResult[OMNI_USD_LIVE_INTERNAL_ERROR].count == 1 &&
               metric.lifetimeByResult[OMNI_USD_LIVE_INTERNAL_ERROR].sumMs == 11);
        assert(!metric.lifetimeByResult[OMNI_USD_LIVE_INVALID_ARGUMENT].count);
    }
    if (applicationTimingTest) {
        const auto& metric = applicationMetrics[1];
        const bool ran = mode == "actor-async-fail" || mode == "actor-async-close" || mode == "actor-async-run-throw";
        const auto result = mode == "actor-async-close" ? OmniUsdLiveResult(OMNI_USD_LIVE_OK) : expected;
        assert(metric.admitted == 1 && metric.retired == 1 && metric.highWater == 1);
        assert(metric.lifetimeByResult[result].count == 1 && metric.lifetimeByResult[result].sumMs == 41);
        if (mode == "actor-async-run-throw") assert(!metric.lifetimeByResult[OMNI_USD_LIVE_OK].count);
        assert(metric.queue.count == unsigned(ran) && metric.body.count == unsigned(ran) &&
               metric.postBodyRetention.count == unsigned(ran));
        if (ran) assert(metric.queue.sumMs == 13 && metric.body.sumMs == 11 && metric.postBodyRetention.sumMs == 17);
    }
    if (uncertain)
        assert(omniUsdLiveWaitForPendingUpdates(0) == OMNI_USD_LIVE_OUTCOME_UNKNOWN);
    auto final = actor->snapshot();
    assert(final.result == expected);
    const auto closedStatus = layerStatus();
    assert(closedStatus.subject_id == subject && closedStatus.state == OMNI_USD_LIVE_STATE_CLOSED);
    assert(closedStatus.result == expected && closedStatus.incident_id == incident);
    assert(closedStatus.pending_write_count == final.pendingWrites && closedStatus.pending_bytes == final.pendingBytes);
    assert(bool(closedStatus.has_unconfirmed_edits) == bool(final.pendingWrites));
    statuses.wait([&] {
        return std::any_of(statuses.events.begin(), statuses.events.end(), [&](const auto& event) {
            return event.status.subject_id == subject && event.status.state == OMNI_USD_LIVE_STATE_CLOSED;
        });
    });
    {
        std::lock_guard<std::mutex> lock(statuses.mutex);
        const auto event = std::find_if(statuses.events.rbegin(), statuses.events.rend(), [&](const auto& value) {
            return value.status.subject_id == subject;
        });
        assert(event != statuses.events.rend() && event->uri == uri && event->endpoint == endpoint);
        assert(event->status.revision == closedStatus.revision && event->status.incident_id == incident);
        assert(event->status.result == expected && event->status.pending_write_count == final.pendingWrites &&
               event->status.pending_bytes == final.pendingBytes);
    }
    const auto preserved = actor->retained().size();
    if (uncertain || mode == "actor-held") {
        const auto retained = actor->retained();
        const size_t offset = mode == "actor-held" ? 1 : 0;
        assert(retained.size() == submitted.size() - offset);
        for (size_t i = 0; i < retained.size(); ++i) {
            assert(retained[i].id == submitted[offset + i]);
            assert(retained[i].disposition == (uncertain && !i ?
                LayerProgress::Disposition::Unknown : LayerProgress::Disposition::Queued));
            auto effect = std::static_pointer_cast<const LayerActor::Effect>(retained[i].owner);
            assert(effect->authored == "local-" + std::to_string(offset + i));
            assert(*std::static_pointer_cast<const std::string>(effect->intent) == effect->authored);
        }
    }
    if (asyncTest) {
        const auto retained=actor->retained();
        const bool retain=mode=="actor-async-cancel" || mode=="actor-async-fail" || mode=="actor-async-queue-throw" ||
            mode=="actor-async-run-throw" || captureOwner;
        assert(final.applied==(retain?1u:mode=="actor-async-release"?4u:2u));
        assert(retained.size()==(retain?1u:0u));
        if (retain) {
            assert(retained.front().id==submitted.front() &&
                   retained.front().disposition==LayerProgress::Disposition::Acknowledged);
            const auto effect=std::static_pointer_cast<const LayerActor::Effect>(retained.front().owner);
            assert(effect->authored=="local-0" && *std::static_pointer_cast<const std::string>(effect->intent)=="local-0");
        }
    }
    if (localTest) {
        const auto retained=actor->retained(); const bool success=mode=="actor-local-release" || mode=="actor-local-shared";
        assert(final.applied==(success?(mode=="actor-local-shared"?3u:2u):1u) && retained.size()==(success?0u:1u));
        if (!success) {
            assert(retained.front().id==submitted.front() && retained.front().disposition==LayerProgress::Disposition::Queued);
            const auto effect=std::static_pointer_cast<const LayerActor::Effect>(retained.front().owner);
            assert(effect->authored=="local-0" && *std::static_pointer_cast<const std::string>(effect->intent)=="local-0");
        }
    }
    actor.reset();
    assert(omniUsdLiveUnregisterCallback(observer, 3000) == OMNI_USD_LIVE_OK);
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (omniUsdLiveUnregisterEndpoint(endpoint, 0) != OMNI_USD_LIVE_OK) {
        assert(std::chrono::steady_clock::now() < until); std::this_thread::yield();
    }
    statuses.wait([&] {
        return std::any_of(statuses.events.begin(), statuses.events.end(), [&](const auto& event) {
            return event.endpoint == endpoint && event.status.subject_kind == OMNI_USD_LIVE_SUBJECT_ENDPOINT &&
                   event.status.state == OMNI_USD_LIVE_STATE_CLOSED;
        });
    });
    auto detached = record<OmniUsdLiveStatus>();
    assert(omniUsdLiveGetLayerStatus(uri.c_str(), &detached) == OMNI_USD_LIVE_NOT_FOUND);
    assert(omniUsdLiveShutdown(0) == OMNI_USD_LIVE_BUSY); // Global observer outlives endpoints.
    assert(omniUsdLiveUnregisterCallback(statusObserver, 3000) == OMNI_USD_LIVE_OK);
    runtime.reset(); assert(omniUsdLiveShutdown(3000) == OMNI_USD_LIVE_OK);
    // A logical actor can span multiple attempts; its snapshot has no single raw status.
    std::cout << "{\"result\":" << final.result << ",\"grpc_status\":0,\"grpc_status_present\":false,\"generation\":" << final.generation
              << ",\"applied\":" << final.applied << ",\"retained\":" << preserved << ",\"auth\":" << seen.auth
              << ",\"status_events\":" << statuses.events.size() << ",\"immediate_status_checks\":" << immediateChecks
              << ",\"duplicate_rejected\":" << (duplicateRejected ? "true" : "false")
              << ",\"acknowledged_pending_bytes\":" << acknowledgedBytes
              << ",\"closed_status\":{\"state\":" << closedStatus.state << ",\"result\":" << closedStatus.result
              << ",\"incident\":" << closedStatus.incident_id << ",\"pending\":" << closedStatus.pending_write_count
              << ",\"bytes\":" << closedStatus.pending_bytes << ",\"unconfirmed\":"
              << (closedStatus.has_unconfirmed_edits ? "true" : "false") << '}'
              << ",\"effects\":[";
    for (size_t i = 0; i < seen.applied.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << "{\"sequence\":" << seen.applied[i].first << ",\"bytes\":\"" << seen.applied[i].second << "\"}";
    }
    std::cout << "]}" << std::endl;
    return 0;
}
