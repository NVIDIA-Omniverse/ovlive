// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// This consumer deliberately includes no private implementation headers.
#include <omni/usd/live/Live.h>
#include <omni/usd/live/LiveUsd.h>
#include "../support/test_tls.h"
#include <pxr/base/plug/registry.h>
#include <pxr/base/tf/errorMark.h>
#include <pxr/base/work/threadLimits.h>
#include <pxr/usd/ar/resolvedPath.h>
#include <pxr/usd/ar/resolver.h>
#include <pxr/usd/sdf/changeBlock.h>
#include <pxr/usd/sdf/fileFormat.h>
#include <pxr/usd/sdf/data.h>
#include <pxr/usd/sdf/layer.h>
#include <pxr/usd/sdf/attributeSpec.h>
#include <pxr/usd/sdf/namespaceEdit.h>
#include <pxr/usd/sdf/primSpec.h>
#include <pxr/usd/sdf/schema.h>
#include <pxr/usd/sdf/types.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usd/stage.h>
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <limits>
#include <iostream>
#include <iomanip>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE
namespace hostApi=omni::usd::live::host;
namespace {
using Clock=std::chrono::steady_clock;
using Fields=std::map<std::string,int>;
void check(bool condition,const char* message) { if(!condition) throw std::runtime_error(message); }
template<class T> T record() { T value{}; value.struct_size=sizeof(value); value.abi_version=OMNI_USD_LIVE_ABI_VERSION; return value; }
bool identifier(const std::string& text) {
    if(text.empty()) return false;
    for(char c:text) if(!(c=='_' || (c>='a' && c<='z') || (c>='A' && c<='Z') || (c>='0' && c<='9'))) return false;
    return true;
}
// Root integer fields; the root's prim membership is reported by prims().
Fields fields(const SdfLayer& layer) {
    Fields result;
    for(const auto& key:layer.ListFields(SdfPath::AbsoluteRootPath())) {
        if(key==SdfChildrenKeys->PrimChildren) continue;
        const auto value=layer.GetField(SdfPath::AbsoluteRootPath(),key);
        check(identifier(key.GetString()) && value.IsHolding<int>(),"public fixture expected root integer field");
        result.emplace(key.GetString(),value.UncheckedGet<int>());
    }
    return result;
}
// Root integer fields of a reopened USDA export. The text format has no
// schema for the fixture's root fields, so it parses each one back as an
// SdfUnregisteredValue holding the decimal text.
Fields exportedFields(const SdfLayer& layer) {
    Fields result;
    for(const auto& key:layer.ListFields(SdfPath::AbsoluteRootPath())) {
        if(key==SdfChildrenKeys->PrimChildren) continue;
        const auto value=layer.GetField(SdfPath::AbsoluteRootPath(),key);
        check(identifier(key.GetString()) && value.IsHolding<SdfUnregisteredValue>(),"export expected unregistered root field");
        const auto& inner=value.UncheckedGet<SdfUnregisteredValue>().GetValue();
        check(inner.IsHolding<std::string>(),"export expected textual root field");
        const auto& text=inner.UncheckedGet<std::string>();
        int number=0; const auto parsed=std::from_chars(text.data(),text.data()+text.size(),number);
        check(!text.empty() && parsed.ec==std::errc{} && parsed.ptr==text.data()+text.size(),"export expected root integer field");
        result.emplace(key.GetString(),number);
    }
    return result;
}
// Prim paths plus each prim's integer `value` attribute default.
struct Prims {
    std::vector<std::string> paths;
    Fields values;
    bool operator==(const Prims& other) const { return paths==other.paths && values==other.values; }
};
const TfToken& valueName() { static const TfToken name("value"); return name; }
Prims prims(SdfLayer& layer) {
    Prims result;
    layer.Traverse(SdfPath::AbsoluteRootPath(),[&](const SdfPath& path) {
        if(!path.IsPrimPath()) return;
        result.paths.push_back(path.GetString());
        const auto value=layer.GetField(path.AppendProperty(valueName()),SdfFieldKeys->Default);
        if(value.IsHolding<int>()) result.values.emplace(path.GetString(),value.UncheckedGet<int>());
    });
    std::sort(result.paths.begin(),result.paths.end());
    return result;
}
void json(const Fields& fields) {
    std::cout << '{'; bool first=true;
    for(const auto& item:fields) { if(!first) std::cout << ','; first=false; std::cout << '"' << item.first << "\":" << item.second; }
    std::cout << '}';
}
template<class F> std::string uriText(F action) {
    uint64_t size=0; check(action(nullptr,&size)==OMNI_USD_LIVE_OK && size,"public URI size query failed");
    std::string result(size,'\0');
    check(action(result.data(),&size)==OMNI_USD_LIVE_OK && size==result.size(),"public URI fill failed");
    result.pop_back(); return result;
}
class Host {
    std::mutex mutex_;
    std::condition_variable changed_;
    std::deque<std::function<void()>> jobs_;
    std::function<void()> heldPublication_;
    bool throwAfterQueue_=false, holdPublication_=false, holdDispatch_=false;
    std::function<void()> heldDispatch_;
    bool stop_=false;
    uint64_t publicationRevision_=0;
    struct Gate { std::promise<void> parked, done; std::promise<std::function<void()>> action; };
    std::shared_ptr<Gate> gate_;
    std::thread worker_;
public:
    SdfLayerRefPtr layer; // All access is through call() on the serial host.
    std::unique_ptr<hostApi::Creation> creation;
    SdfLayerRefPtr created;
    Host():worker_([this] {
        for(;;) {
            std::function<void()> job;
            { std::unique_lock<std::mutex> lock(mutex_); changed_.wait(lock,[&] { return stop_ || !jobs_.empty(); });
              if(stop_ && jobs_.empty()) break;
              job=std::move(jobs_.front()); jobs_.pop_front(); }
            job(); job={};
        }
        heldPublication_={}; heldDispatch_={}; creation.reset(); created.Reset(); layer.Reset();
    }) {}
    ~Host() { { std::lock_guard<std::mutex> lock(mutex_); stop_=true; changed_.notify_all(); } worker_.join(); }
    bool current() const { return std::this_thread::get_id()==worker_.get_id(); }
    void publish(std::function<void()> job) {
        std::function<void()> counted=[this,job=std::move(job)]() mutable {
            job(); job={};
            std::lock_guard<std::mutex> lock(mutex_);
            ++publicationRevision_; changed_.notify_all();
        };
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if(holdDispatch_) { holdDispatch_=false; heldDispatch_=std::move(counted); return; }
        }
        dispatch(std::move(counted));
    }
    // Withhold the next job the publication context dispatches, such as the
    // flush job a direct edit schedules, until releaseDispatch queues it.
    void holdDispatch() {
        std::lock_guard<std::mutex> lock(mutex_);
        check(!holdDispatch_ && !heldDispatch_,"dispatch hold already armed");
        holdDispatch_=true;
    }
    bool dispatchHeld() { std::lock_guard<std::mutex> lock(mutex_); return bool(heldDispatch_); }
    void releaseDispatch() {
        std::lock_guard<std::mutex> lock(mutex_);
        check(bool(heldDispatch_),"no withheld dispatch to release");
        jobs_.push_back(std::move(heldDispatch_)); heldDispatch_={}; changed_.notify_all();
    }
    uint64_t publicationRevision() {
        std::lock_guard<std::mutex> lock(mutex_); return publicationRevision_;
    }
    void awaitPublication(uint64_t previous,Clock::time_point deadline) {
        std::unique_lock<std::mutex> lock(mutex_);
        check(changed_.wait_until(lock,deadline,[&] { return publicationRevision_!=previous; }),
              "public scene convergence timed out waiting for publication");
    }
    void dispatch(std::function<void()> job) {
        std::lock_guard<std::mutex> lock(mutex_); check(!stop_ && jobs_.size()<64,"public consumer host queue exhausted");
        jobs_.push_back(std::move(job)); changed_.notify_all();
    }
    void dispatchPublication(std::function<void()> job) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if(throwAfterQueue_ || holdPublication_) {
                check(!heldPublication_,"publication fixture already holds a queued closure");
                heldPublication_=std::move(job); holdPublication_=false;
                if(throwAfterQueue_) {
                    throwAfterQueue_=false;
                    throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
                }
                return;
            }
        }
        dispatch(std::move(job));
    }
    void queueThenThrowPublication() {
        std::lock_guard<std::mutex> lock(mutex_);
        check(!heldPublication_ && !throwAfterQueue_,"publication rejection already armed");
        throwAfterQueue_=true;
    }
    void holdPublication() {
        std::lock_guard<std::mutex> lock(mutex_);
        check(!heldPublication_ && !throwAfterQueue_ && !holdPublication_,"publication hold already armed");
        holdPublication_=true;
    }
    bool publicationHeld() { std::lock_guard<std::mutex> lock(mutex_); return bool(heldPublication_); }
    void runHeldPublication() {
        check(current(),"late publication must run on original host");
        std::function<void()> job;
        { std::lock_guard<std::mutex> lock(mutex_);
          check(bool(heldPublication_),"no rejected publication closure retained"); job=std::move(heldPublication_); }
        job(); // Its final capture owner retires on this host before the FIFO barrier.
    }
    void dropHeldPublication() {
        check(current(),"publication captures must retire on original host");
        std::function<void()> job;
        { std::lock_guard<std::mutex> lock(mutex_);
          check(bool(heldPublication_),"no held publication closure to drop"); job=std::move(heldPublication_); }
        job={}; // Deliberately never invoke the accepted finalization closure.
    }
    // Park the worker in a gate job; runParked later hands the gate one action
    // to run on the host ahead of every job queued behind the gate.
    void park() {
        auto gate=std::make_shared<Gate>();
        auto parked=gate->parked.get_future();
        auto action=gate->action.get_future().share();
        dispatch([gate,action] { gate->parked.set_value(); action.get()(); gate->done.set_value(); });
        check(parked.wait_for(std::chrono::seconds(5))==std::future_status::ready,"host gate did not park");
        std::lock_guard<std::mutex> lock(mutex_); gate_=gate;
    }
    // Until at least `count` jobs wait behind the parked gate.
    void awaitQueued(size_t count) {
        std::unique_lock<std::mutex> lock(mutex_);
        check(changed_.wait_for(lock,std::chrono::seconds(5),[&] { return jobs_.size()>=count; }),
              "no job queued behind the parked host");
    }
    // `guard` is a failure-only deadline for the action.
    template<class F> void runParked(F action,std::chrono::milliseconds guard=std::chrono::seconds(8)) {
        std::shared_ptr<Gate> gate;
        { std::lock_guard<std::mutex> lock(mutex_); gate=std::move(gate_); }
        check(bool(gate),"host is not parked");
        auto done=gate->done.get_future();
        gate->action.set_value(std::function<void()>(std::move(action)));
        check(done.wait_for(guard)==std::future_status::ready,"parked host action timed out");
    }
    template<class F> auto call(F action)->decltype(action()) {
        auto task=std::make_shared<std::packaged_task<decltype(action())()>>(std::move(action));
        auto future=task->get_future(); dispatch([task] { (*task)(); });
        check(future.wait_for(std::chrono::seconds(8))==std::future_status::ready,"public host call timed out");
        return future.get();
    }
    Fields read() { return call([&] { check(bool(layer),"public layer already dropped"); return fields(*layer); }); }
    Prims readPrims() { return call([&] { check(bool(layer),"public layer already dropped"); return prims(*layer); }); }
    Fields facadeChecks() {
        return call([&] {
            TfErrorMark errors;
            const auto original=fields(*layer);
            const auto ordinary=SdfLayer::CreateAnonymous("public-facade-sentinel.usda");
            check(bool(ordinary),"facade sentinel creation failed");
            check(hostApi::save(*ordinary,0)==OMNI_USD_LIVE_INVALID_ARGUMENT,"non-live save succeeded");
            check(layer->PermissionToEdit(),"attached live layer is not editable");
            hostApi::PublicationContext* probeRaw=nullptr;
            check(hostApi::createPublicationContext([this](auto job) { publish(std::move(job)); },
                  [this] { return current(); },&probeRaw)==OMNI_USD_LIVE_OK && probeRaw,
                  "seed probe context failed");
            std::unique_ptr<hostApi::PublicationContext> probeContext(probeRaw);
            const auto uri=layer->GetIdentifier();
            auto wrongHost=std::async(std::launch::async,[&] {
                hostApi::Creation* absent=nullptr;
                uint64_t subject=UINT64_MAX;
                check(hostApi::prepareCreation(*ordinary,uri.c_str(),&absent)==OMNI_USD_LIVE_BUSY && !absent &&
                      probeContext->prepareCreation(*ordinary,uri.c_str(),&absent)==OMNI_USD_LIVE_BUSY && !absent &&
                      hostApi::prepareRetainedCreation(*ordinary,uri.c_str(),&subject)==OMNI_USD_LIVE_BUSY && subject==UINT64_MAX &&
                      probeContext->prepareRetainedCreation(*ordinary,uri.c_str(),&subject)==OMNI_USD_LIVE_BUSY && subject==UINT64_MAX,
                      "off-host seed preparation bypassed host admission or changed outputs");
            });
            wrongHost.get();
            check(hostApi::setPublicationContext([this](auto job) { publish(std::move(job)); },
                  [this] { return current(); })==OMNI_USD_LIVE_OK,"seed context replacement failed");
            hostApi::Creation* unexpected=nullptr;
            uint64_t subject=UINT64_MAX;
            check(probeContext->prepareCreation(*ordinary,uri.c_str(),&unexpected)==OMNI_USD_LIVE_NOT_INITIALIZED && !unexpected &&
                  probeContext->prepareRetainedCreation(*ordinary,uri.c_str(),&subject)==OMNI_USD_LIVE_NOT_INITIALIZED &&
                  subject==UINT64_MAX,"seed preparation rebound a replaced context or changed outputs");
            check(probeContext->close()==OMNI_USD_LIVE_OK,"rejected seed leaked temporary context");
            check(fields(*layer)==original && errors.IsClean(),"facade checks changed the live layer or emitted SDK diagnostics");
            return original;
        });
    }
    void drop() { call([&] { creation.reset(); created.Reset(); layer.Reset(); }); }
};
struct Runtime {
    OmniUsdLiveEndpoint endpoint=0;
    bool initialized=false;
    void close() {
        if(endpoint) {
            const auto deadline=Clock::now()+std::chrono::seconds(5);
            for(;;) {
                const auto result=omniUsdLiveUnregisterEndpoint(endpoint,0);
                if(result==OMNI_USD_LIVE_OK) break;
                check(result==OMNI_USD_LIVE_BUSY && Clock::now()<deadline,"public plugin leaked endpoint owners");
                std::this_thread::yield();
            }
            endpoint=0;
        }
        if(initialized) { check(omniUsdLiveShutdown(5000)==OMNI_USD_LIVE_OK,"public plugin shutdown failed"); initialized=false; }
    }
};
class StatusObserver {
    std::mutex mutex_;
    std::condition_variable changed_;
    OmniUsdLiveCallbackHandle handle_=0;
    uint64_t subject_=0;
    OmniUsdLiveStatus latest_{};
public:
    void open(OmniUsdLiveEndpoint endpoint,uint64_t subject) {
        check(!handle_ && subject,"invalid status observer registration");
        subject_=subject;
        check(omniUsdLiveRegisterStatusCallback(endpoint,this,
            [](void* user,const OmniUsdLiveStatusEvent* event) {
                auto& self=*static_cast<StatusObserver*>(user);
                if(!event || !event->status || event->status->subject_id!=self.subject_) return;
                std::lock_guard<std::mutex> lock(self.mutex_);
                if(event->status->revision>self.latest_.revision) self.latest_=*event->status;
                self.changed_.notify_all();
            },&handle_)==OMNI_USD_LIVE_OK,"public status observer registration failed");
    }
    OmniUsdLiveStatus wait(uint32_t expected) {
        check(handle_ && expected>=OMNI_USD_LIVE_STATE_OPENING && expected<=OMNI_USD_LIVE_STATE_CLOSED,
              "invalid status wait");
        std::unique_lock<std::mutex> lock(mutex_);
        check(changed_.wait_for(lock,std::chrono::seconds(5),[&] {
            return latest_.state==expected || latest_.state==OMNI_USD_LIVE_STATE_FAILED;
        }),"public status callback wait timed out");
        check(latest_.state==expected,"public status reached unexpected terminal failure");
        return latest_;
    }
    void close() {
        if(!handle_) return;
        check(omniUsdLiveUnregisterCallback(handle_,5000)==OMNI_USD_LIVE_OK,
              "public status observer failed to retire");
        handle_=0;
    }
    ~StatusObserver() {
        try { close(); }
        catch(...) { std::cerr << "status callback cleanup failed\n"; std::_Exit(2); }
    }
};
std::string readFile(const std::filesystem::path& path) {
    std::ifstream file(path,std::ios::binary); check(bool(file),"public export file missing");
    return {std::istreambuf_iterator<char>(file),std::istreambuf_iterator<char>()};
}
void contextOwnershipChecks() {
    hostApi::PublicationContext* owner = nullptr;
    check(hostApi::createPublicationContext({}, {}, &owner) == OMNI_USD_LIVE_INVALID_ARGUMENT && !owner,
          "empty owned context accepted");
    OmniUsdLiveResult reentered = OMNI_USD_LIVE_INTERNAL_ERROR;
    struct Capture {
        hostApi::PublicationContext*& owner;
        OmniUsdLiveResult& result;
        Capture(hostApi::PublicationContext*& value, OmniUsdLiveResult& output) : owner(value), result(output) {}
        ~Capture() { result = owner->close(); }
    };
    auto capture = std::make_shared<Capture>(owner, reentered);
    check(hostApi::createPublicationContext([capture](auto job) { job(); }, [capture] { return true; }, &owner) == OMNI_USD_LIVE_OK,
          "owned context admission failed");
    capture.reset();
    check(owner->close() == OMNI_USD_LIVE_OK && reentered == OMNI_USD_LIVE_BUSY,
          "context close did not retire captures outside locks");
    check(owner->close() == OMNI_USD_LIVE_OK, "context close is not idempotent");
    delete owner;
}
// The runtime's initial-open deadline and a failure-only guard below it. A
// reopen that waited for the closing layer instead of refusing would wait on
// the host job queued behind its own gate until the open deadline, so the
// guard fails the parked reopen before that wait could end in a refusal.
constexpr uint32_t kInitialOpenTimeoutMs=4000;
constexpr std::chrono::milliseconds kParkedReopenGuard(kInitialOpenTimeoutMs*3/4);
// Test-only credentials for the soak: LIVE_TEST_TOKEN_PREFIX names
// generation-numbered tokens (<prefix>-<generation>) that expire
// LIVE_TEST_TOKEN_TTL_MS after installation; the callback answers every
// challenge with the next generation's token, as a host would.
struct SoakAuth {
    std::string prefix; uint64_t ttlMs=0; OmniUsdLiveCallbackHandle handle=0;
    std::mutex mutex; std::vector<std::array<uint64_t,6>> events; // challenge, reason, rejected, grpc present, grpc, result
    SoakAuth() {
        if(const char* value=std::getenv("LIVE_TEST_TOKEN_PREFIX")) prefix=value;
        if(const char* value=std::getenv("LIVE_TEST_TOKEN_TTL_MS")) ttlMs=std::stoull(value);
    }
    bool enabled() const { return !prefix.empty(); }
    std::string token(uint64_t generation) const { return prefix+"-"+std::to_string(generation); }
    uint64_t expiry() const {
        if(!ttlMs) return 0;
        return uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count())+ttlMs;
    }
    // The answer holds the event lock across the replacement, so a report
    // taken after any RPC the new token enabled already lists this answer.
    static void OMNI_USD_LIVE_CALL answer(void* user,const OmniUsdLiveAuthRequired* event) {
        auto& self=*static_cast<SoakAuth*>(user);
        const auto next=self.token(event->rejected_generation+1);
        uint64_t generation=0;
        std::lock_guard<std::mutex> lock(self.mutex);
        const auto result=omniUsdLiveSetAccessToken(event->endpoint,event->rejected_generation,next.c_str(),
                                                    self.expiry(),&generation);
        self.events.push_back({event->challenge_id,event->reason,event->rejected_generation,
                               event->grpc_status_present,uint64_t(uint32_t(event->grpc_status)),uint64_t(result)});
    }
    void install(OmniUsdLiveEndpoint endpoint) {
        if(enabled()) check(omniUsdLiveRegisterAuthRequiredCallback(endpoint,this,answer,&handle)==OMNI_USD_LIVE_OK,
                            "soak auth callback registration failed");
    }
    void close() {
        if(handle) { check(omniUsdLiveUnregisterCallback(handle,3000)==OMNI_USD_LIVE_OK,"soak auth callback retained"); handle=0; }
    }
    // The endpoint's credential state and the callback answers since the previous report.
    void report(OmniUsdLiveEndpoint endpoint) {
        auto state=record<OmniUsdLiveAuthState>();
        check(omniUsdLiveGetAuthState(endpoint,&state)==OMNI_USD_LIVE_OK,"public auth state query failed");
        std::vector<std::array<uint64_t,6>> answered;
        { std::lock_guard<std::mutex> lock(mutex); answered.swap(events); }
        std::cout << "{\"type\":\"auth-events\",\"generation\":" << state.generation
                  << ",\"expires_at_ms\":" << state.expires_at_ms << ",\"events\":[";
        for(size_t i=0;i<answered.size();++i) {
            std::cout << (i ? ",[" : "[");
            for(size_t j=0;j<answered[i].size();++j) std::cout << (j ? "," : "") << answered[i][j];
            std::cout << ']';
        }
        std::cout << "]}" << std::endl;
    }
};
int run(int argc,char** argv) {
    check(argc==5,"public plugin driver expects authority session seed plugInfo");
    std::cerr << "public shared-plugin consumer seed=" << argv[3] << '\n';
    check(std::filesystem::is_regular_file(argv[4]),"packaged plugin metadata missing");
    check(!PlugRegistry::GetInstance().RegisterPlugins(argv[4]).empty(),"packaged plugin metadata not registered");
    contextOwnershipChecks();
    Host host; Runtime runtime;
    SoakAuth auth;
    struct Cleanup {
        Host& host; Runtime& runtime; SoakAuth& auth;
        ~Cleanup() { try { host.drop(); hostApi::setPublicationContext({},{}); auth.close(); runtime.close(); }
                     catch(...) { std::cerr << "public consumer cleanup failed\n"; std::_Exit(2); } }
    } cleanup{host,runtime,auth};
    auto options=record<OmniUsdLiveRuntimeOptions>(); options.initial_open_timeout_ms=kInitialOpenTimeoutMs;
    options.max_update_bytes=64*1024; // Also exercises seed compiler preallocation limits.
    // Rotations reconnect within a 100 ms first window; the production defaults would add about a second each.
    options.reconnect_initial_ms=50; options.reconnect_maximum_ms=5000;
    check(omniUsdLiveInitialize(&options)==OMNI_USD_LIVE_OK,"public initialize failed"); runtime.initialized=true;
    const TestTls tls;
    const auto initial=auth.token(1);
    auto endpoint=record<OmniUsdLiveEndpointOptions>(); endpoint.authority=argv[1];
    endpoint.access_token=auth.enabled() ? initial.c_str() : "synthetic-writer";
    endpoint.expires_at_ms=auth.enabled() ? auth.expiry() : 0; tls.apply(endpoint);
    check(omniUsdLiveRegisterEndpoint(&endpoint,&runtime.endpoint)==OMNI_USD_LIVE_OK,"public endpoint registration failed");
    auth.install(runtime.endpoint);
    hostApi::PublicationContext* previousRaw = nullptr;
    check(hostApi::createPublicationContext([](auto job) { job(); }, [] { return false; }, &previousRaw) == OMNI_USD_LIVE_OK,
          "previous context admission failed");
    std::unique_ptr<hostApi::PublicationContext> previous(previousRaw);
    hostApi::PublicationContext* contextRaw = nullptr;
    check(hostApi::createPublicationContext([&](auto job) { host.publish(std::move(job)); },[&] { return host.current(); }, &contextRaw)==OMNI_USD_LIVE_OK,
          "public owned host context configuration failed");
    std::unique_ptr<hostApi::PublicationContext> context(contextRaw);
    check(previous->close() == OMNI_USD_LIVE_OK, "replaced context did not close");
    previous.reset(); // Its close/destructor must not clear the new context.

    const auto session=uriText([&](char* out,uint64_t* size) { return omniUsdLiveMakeSessionUri(argv[1],argv[2],out,size); });
    const auto uri=uriText([&](char* out,uint64_t* size) { return omniUsdLiveMakeLayerUri(session.c_str(),"root.live",out,size); });
    const auto creationUri=uriText([&](char* out,uint64_t* size) { return omniUsdLiveMakeLayerUri(session.c_str(),"created.live",out,size); });
    std::unique_ptr<hostApi::PublicationContext> creationContext;
    uint64_t retainedSubject=0;
    host.call([&] {
        TfErrorMark errors; host.layer=SdfLayer::FindOrOpen(uri);
        check(host.layer && errors.IsClean() && host.layer->PermissionToEdit(),"packaged file format open failed or returned a read-only layer");
        check(host.layer->GetFileFormat()->GetFormatId()==TfToken("live"),"public consumer selected wrong file format");
        auto alias=uri; alias.replace(alias.rfind("root.live"),9,"%72oot.live");
        check(SdfLayer::FindOrOpen(alias)==host.layer && errors.IsClean(),"public canonical alias created second layer");
    });
    check(context->close() == OMNI_USD_LIVE_BUSY, "attached layer failed to retain original context");
    auto status=[&] {
        auto value=record<OmniUsdLiveStatus>();
        check(omniUsdLiveGetLayerStatus(uri.c_str(),&value)==OMNI_USD_LIVE_OK && value.subject_id,"public status query failed");
        return value;
    };
    { StatusObserver initial; initial.open(runtime.endpoint,status().subject_id); initial.wait(OMNI_USD_LIVE_STATE_READY); }
    auto report=[&](const char* type) {
        const auto values=host.read(); const auto value=status();
        std::cout << "{\"type\":\"" << type << "\",\"result\":" << value.result << ",\"state\":" << value.state
                  << ",\"subject\":\"" << value.subject_id << "\",\"pending\":" << value.pending_write_count
                  << ",\"pending_bytes\":" << value.pending_bytes << ",\"fields\":";
        json(values); std::cout << '}' << std::endl;
    };
    report("ready");
    auto creationReport=[&](OmniUsdLiveResult result) {
        const auto* owner=host.call([&] { return host.creation.get(); });
        check(owner,"missing public creation");
        hostApi::CreationOutcome outcome;
        check(owner->getOutcome(&outcome)==OMNI_USD_LIVE_OK,"public creation outcome unavailable off-host");
        const auto values=host.call([&] {
            SdfLayerRefPtr value=host.layer;
            const auto got=host.creation->getLayer(&value);
            if(got==OMNI_USD_LIVE_OK) { host.created=value; return fields(*value); }
            check(got==OMNI_USD_LIVE_BUSY && value==host.layer,"creation getLayer changed error output");
            return Fields{};
        });
        auto current=record<OmniUsdLiveStatus>();
        check(omniUsdLiveGetLayerStatus(creationUri.c_str(),&current)==OMNI_USD_LIVE_OK &&
              current.subject_id==outcome.subjectId,"creation status identity changed");
        std::cout << "{\"type\":\"creation\",\"result\":" << result << ",\"phase\":" << unsigned(outcome.phase)
                  << ",\"outcome\":" << outcome.result << ",\"raw_result\":" << outcome.error.result
                  << ",\"subject\":" << outcome.subjectId << ",\"started\":" << outcome.started
                  << ",\"closing\":" << outcome.closing << ",\"request\":" << outcome.error.request_id
                  << ",\"attempt\":" << outcome.error.attempt_id << ",\"request_incident\":" << outcome.error.incident_id
                  << ",\"operation\":" << outcome.error.operation << ",\"grpc_present\":" << outcome.error.grpc_status_present
                  << ",\"grpc_status\":" << outcome.error.grpc_status
                  << ",\"quiescent\":" << outcome.quiescent << ",\"state\":" << current.state
                  << ",\"incident\":" << current.incident_id << ",\"pending\":" << current.pending_write_count
                  << ",\"returned_layer_alive\":" << host.call([&] { return bool(host.created); })
                  << ",\"fields\":"; json(values); std::cout << '}' << std::endl;
    };
    auto retainedReport=[&](OmniUsdLiveResult result) {
        hostApi::RetainedCreationInfo info;
        info.subjectId=UINT64_MAX; info.liveUri="unchanged-enumeration";
        const auto found=hostApi::nextRetainedCreation(0,&info);
        check(found==OMNI_USD_LIVE_OK || found==OMNI_USD_LIVE_NOT_FOUND,"retained creation enumeration failed");
        if(found==OMNI_USD_LIVE_OK) {
            check(info.subjectId==retainedSubject && info.liveUri==creationUri,"retained canonical identity changed");
            hostApi::RetainedCreationInfo end=info;
            check(hostApi::nextRetainedCreation(info.subjectId,&end)==OMNI_USD_LIVE_NOT_FOUND &&
                  end.subjectId==info.subjectId && end.liveUri==info.liveUri,
                  "retained ordered successor changed output at end");
        } else check(info.subjectId==UINT64_MAX && info.liveUri=="unchanged-enumeration",
                     "missing retained enumeration changed output");
        std::cout << "{\"type\":\"retained-creation\",\"result\":" << result
                  << ",\"subject\":" << retainedSubject << ",\"listed\":" << (found==OMNI_USD_LIVE_OK)
                  << ",\"uri\":" << std::quoted(found==OMNI_USD_LIVE_OK?info.liveUri:std::string{}) << '}' << std::endl;
    };
    std::map<std::filesystem::path,std::string> exports;
    StatusObserver observer; // Retires before Cleanup and its native Host target.
    std::string line;
    while(std::getline(std::cin,line)) {
        std::istringstream input(line); std::string command,extra; input>>command;
        if(command=="creation-usd-new" || command=="creation-usd-new-drop" ||
           command=="creation-usd-new-reject-dispatch" || command=="creation-usd-new-hold-finalizer" ||
           command=="creation-usd-new-save") {
            check(!creationContext && !retainedSubject && !host.call([&] { return bool(host.creation) || bool(host.created); }),
                  "CreateNew fixture already owns a workflow");
            hostApi::PublicationContext* raw=nullptr;
            check(hostApi::createPublicationContext([&](auto job) { host.dispatchPublication(std::move(job)); },
                [&] { return host.current(); },&raw)==OMNI_USD_LIVE_OK,"CreateNew host context failed");
            creationContext.reset(raw);
            bool diagnosed=false, dropped=false, readOnlyRefused=false;
            uint32_t immediateState=0;
            const auto success=host.call([&] {
                TfErrorMark errors;
                if(command=="creation-usd-new-reject-dispatch") host.queueThenThrowPublication();
                if(command=="creation-usd-new-hold-finalizer") host.holdPublication();
                bool created=false;
                if(command=="creation-usd-new-save") {
                    host.created=SdfLayer::New(SdfFileFormat::FindById(TfToken("live")),creationUri);
                    check(bool(host.created),"actual Sdf New did not return a target");
                    created=host.created->Save(); // Default Save must not skip pristine live creation.
                } else {
                    host.created=SdfLayer::CreateNew(creationUri);
                    created=bool(host.created);
                }
                diagnosed=!errors.IsClean();
                if(created) check(!diagnosed,"successful CreateNew emitted USD diagnostics");
                errors.Clear();
                hostApi::RetainedCreationInfo info;
                check(hostApi::nextRetainedCreation(0,&info)==OMNI_USD_LIVE_OK && info.liveUri==creationUri,
                      "actual CreateNew did not retain its canonical workflow");
                retainedSubject=info.subjectId;
                auto current=record<OmniUsdLiveStatus>();
                check(omniUsdLiveGetLayerStatus(creationUri.c_str(),&current)==OMNI_USD_LIVE_OK &&
                      current.subject_id==retainedSubject,"CreateNew lost pre-finalization status identity");
                immediateState=current.state;
                if(created) {
                    check(current.state==OMNI_USD_LIVE_STATE_OPENING &&
                          host.created->GetIdentifier()==creationUri &&
                          host.created->GetResolvedPath().GetPathString()==creationUri &&
                          host.created->GetFileFormat()->GetFormatId()==TfToken("live") &&
                          !host.created->PermissionToEdit() && fields(*host.created).empty(),
                          "CreateNew exposed premature readiness, an editable unactivated target, wrong identity or nonempty seed");
                    // USD refuses an edit of the unactivated target before it reaches the backing.
                    TfErrorMark refused;
                    host.created->SetField(SdfPath::AbsoluteRootPath(),TfToken("unactivated"),VtValue(1));
                    bool coded=false;
                    for(const auto& error:refused)
                        coded=coded || error.GetCommentary().find("not editable")!=std::string::npos;
                    refused.Clear();
                    auto after=record<OmniUsdLiveStatus>();
                    readOnlyRefused=coded && fields(*host.created).empty() &&
                                    omniUsdLiveGetLayerStatus(creationUri.c_str(),&after)==OMNI_USD_LIVE_OK &&
                                    after.state==OMNI_USD_LIVE_STATE_OPENING && after.buffered_edit_count==0 &&
                                    after.pending_write_count==0;
                    check(readOnlyRefused,"unactivated CreateNew target accepted an edit");
                    hostApi::Creation* absent=nullptr;
                    check(hostApi::acquireRetainedCreation(retainedSubject,&absent)==OMNI_USD_LIVE_BUSY && !absent,
                          "CreateNew workflow transferable before queued finalization retires");
                    if(command=="creation-usd-new-drop") {
                        const SdfLayerHandle weak=host.created;
                        host.created.Reset();
                        check(!weak,"queued CreateNew finalizer pinned the target layer");
                        dropped=true;
                    }
                }
                return created;
            });
            // Serial queue order proves outer USD initialization has returned
            // and its normally enqueued finalizer/captures have retired.
            host.call([] {});
            const auto subject=retainedSubject;
            auto current=record<OmniUsdLiveStatus>();
            check(omniUsdLiveGetLayerStatus(creationUri.c_str(),&current)==OMNI_USD_LIVE_OK &&
                  current.subject_id==subject,"CreateNew finalization changed subject");
            // A finalized target owns its workflow's attachment and frees the
            // retention slot; any other outcome stays retained.
            hostApi::RetainedCreationInfo after;
            const bool listed=hostApi::nextRetainedCreation(0,&after)==OMNI_USD_LIVE_OK;
            check(!listed || after.subjectId==subject,"CreateNew finalization replaced its retained subject");
            if(!listed) retainedSubject=0;
            // The ordinary finalizer has run: activation grants edit permission.
            // A held finalizer leaves the target read-only.
            bool editable=false;
            if(success && !dropped) {
                editable=host.call([&] { return host.created->PermissionToEdit(); });
                if(command=="creation-usd-new")
                    check(current.state==OMNI_USD_LIVE_STATE_READY && editable,"activated creation layer is not editable");
                if(command=="creation-usd-new-hold-finalizer")
                    check(host.publicationHeld() && !editable,"creation layer is editable before activation");
            }
            std::cout << "{\"type\":\"usd-create-new\",\"hook_ok\":" << success
                      << ",\"read_only_refused\":" << readOnlyRefused << ",\"editable\":" << editable
                      << ",\"diagnosed\":" << diagnosed << ",\"subject\":" << subject << ",\"listed\":" << listed
                      << ",\"immediate_state\":" << immediateState << ",\"state\":" << current.state
                      << ",\"outcome\":" << current.result << ",\"pending\":" << current.pending_write_count
                      << ",\"returned_layer_alive\":" << host.call([&] { return bool(host.created); })
                      << ",\"dropped\":" << dropped << ",\"held_finalizer\":" << host.publicationHeld() << '}' << std::endl;
        } else if(command=="usd-new-import-twice") {
            // SdfLayer::New gives an unread, unattached live layer. Importing
            // its own (existing) path reaches the format's Read hook like an
            // open; a failed first read leaves the layer unattached, and the
            // second Import reports the format's rejection code.
            check(!creationContext && !host.call([&] { return bool(host.created); }),"New fixture already owns a layer");
            hostApi::PublicationContext* raw=nullptr;
            check(hostApi::createPublicationContext([&](auto job) { host.dispatchPublication(std::move(job)); },
                [&] { return host.current(); },&raw)==OMNI_USD_LIVE_OK,"New host context failed");
            creationContext.reset(raw);
            const std::string prefix="Live file format operation failed (result=";
            auto attempt=[&](bool& imported) {
                TfErrorMark errors;
                imported=host.created->Import(creationUri);
                long code=-1;
                for(const auto& error:errors) {
                    const auto text=error.GetCommentary();
                    if(const auto at=text.find(prefix); at!=std::string::npos) code=std::stol(text.substr(at+prefix.size()));
                }
                errors.Clear();
                return code;
            };
            bool first=true,second=true; long firstCode=-1,secondCode=-1;
            host.call([&] {
                host.created=SdfLayer::New(SdfFileFormat::FindById(TfToken("live")),creationUri);
                check(bool(host.created),"actual Sdf New did not return a target");
                firstCode=attempt(first);
                secondCode=attempt(second);
                host.created.Reset();
            });
            creationContext.reset();
            std::cout << "{\"type\":\"usd-new-import\",\"first\":" << first << ",\"first_result\":" << firstCode
                      << ",\"second\":" << second << ",\"second_result\":" << secondCode << '}' << std::endl;
        } else if(command=="creation-new-repeat-save") {
            bool diagnosed=false;
            const auto success=host.call([&] {
                check(bool(host.created),"repeat initial Save requires a surviving target");
                TfErrorMark errors;
                const auto result=host.created->Save(true);
                diagnosed=!errors.IsClean(); errors.Clear();
                return result;
            });
            std::cout << "{\"type\":\"repeated-initial-save\",\"hook_ok\":" << success
                      << ",\"diagnosed\":" << diagnosed << '}' << std::endl;
        } else if(command=="creation-new-reset-target") {
            host.call([&] {
                check(bool(host.created),"target reset requires a surviving target");
                const SdfLayerHandle weak=host.created; host.created.Reset();
                check(!weak,"failed initial Save workflow pinned its target");
            });
            if(host.call([&] { return bool(host.creation); })) creationReport(OMNI_USD_LIVE_OK);
            else retainedReport(OMNI_USD_LIVE_OK);
        } else if(command=="creation-new-late-finalize" || command=="creation-new-drop-finalizer") {
            host.call([&] {
                hostApi::Creation* absent=nullptr;
                check(hostApi::acquireRetainedCreation(retainedSubject,&absent)==OMNI_USD_LIVE_BUSY && !absent,
                      "queued rejected finalizer released its retention lease early");
                if(command=="creation-new-drop-finalizer") {
                    check(bool(host.created),"dropped finalizer fixture lost the live returned target");
                    host.dropHeldPublication();
                    check(host.created->GetIdentifier()==creationUri && fields(*host.created).empty(),
                          "dropping finalizer changed the returned target");
                } else host.runHeldPublication();
            });
            host.call([] {});
            check(!host.publicationHeld(),"late CreateNew finalizer did not retire");
            retainedReport(OMNI_USD_LIVE_OK);
        } else if(command=="creation-usd-export" || command=="creation-usd-export-comment" ||
           command=="creation-usd-export-args" || command=="creation-usd-export-find" ||
           command=="creation-usd-export-reopen") {
            // creation-usd-export waits for a successful destination to close.
            // -find opens the destination in the Export's own host turn and
            // keeps it; -reopen opens it in the next host turn, while the
            // adopted destination's actor may still be closing.
            const bool find=command=="creation-usd-export-find", reopen=command=="creation-usd-export-reopen";
            int counter=0; check(bool(input>>counter),"missing USD Export seed counter");
            check(!host.call([&] { return bool(host.creation) || ((find || reopen) && bool(host.created)); }),
                  "Export fixture already owns a workflow or created layer");
            if(!creationContext) {
                hostApi::PublicationContext* raw=nullptr;
                check(hostApi::createPublicationContext([&](auto job) { host.dispatch(std::move(job)); },
                    [&] { return host.current(); },&raw)==OMNI_USD_LIVE_OK,"USD Export host context failed");
                creationContext.reset(raw);
            }
            bool diagnosed=false, listedInside=false;
            uint64_t subject=0;
            auto current=record<OmniUsdLiveStatus>();
            StatusObserver destination;
            const auto success=host.call([&] {
                TfErrorMark errors;
                auto seed=SdfLayer::CreateAnonymous("actual-usd-export.usda");
                check(bool(seed),"USD Export seed allocation failed");
                seed->SetField(SdfPath::AbsoluteRootPath(),TfToken("counter"),VtValue(counter));
                const auto identifier=seed->GetIdentifier();
                SdfLayer::FileFormatArguments arguments;
                if(command=="creation-usd-export-args") arguments.emplace("unsupportedFixtureOption","1");
                const auto exported=seed->Export(creationUri,
                    command=="creation-usd-export-comment"?"unsupported fixture comment":"",arguments);
                diagnosed=!errors.IsClean();
                if(exported) check(!diagnosed,"successful actual USD Export emitted an error");
                errors.Clear(); // Expected bool failure must not poison later SDK checks.
                check(seed->GetIdentifier()==identifier && fields(*seed)==Fields{{"counter",counter}},
                      "USD Export changed its authored source");
                seed->SetField(SdfPath::AbsoluteRootPath(),TfToken("counter"),VtValue(99));
                seed.Reset(); // Only the hook's workflow can preserve the original seed.
                // Still inside the outer host turn: the workflow is retained,
                // including after success until its adoption job runs.
                hostApi::RetainedCreationInfo info;
                const auto listed=hostApi::nextRetainedCreation(0,&info);
                check(listed==OMNI_USD_LIVE_OK || listed==OMNI_USD_LIVE_NOT_FOUND,"USD Export retained enumeration failed");
                check(!exported || listed==OMNI_USD_LIVE_OK,"successful USD Export did not retain its workflow");
                listedInside=listed==OMNI_USD_LIVE_OK;
                if(listedInside) {
                    check(info.liveUri==creationUri && (!retainedSubject || retainedSubject==info.subjectId),
                          "USD Export replaced retained canonical subject");
                    subject=info.subjectId;
                    hostApi::RetainedCreationInfo end;
                    check(hostApi::nextRetainedCreation(subject,&end)==OMNI_USD_LIVE_NOT_FOUND,
                          "USD Export unexpectedly retained a second workflow");
                }
                const auto queried=omniUsdLiveGetLayerStatus(creationUri.c_str(),&current);
                check(!listedInside || (queried==OMNI_USD_LIVE_OK && current.subject_id==subject),
                      "USD Export retained/public status identities disagree");
                if(!listedInside) current=record<OmniUsdLiveStatus>();
                if(exported) {
                    destination.open(runtime.endpoint,subject); // Before the adoption job can run.
                    if(find) {
                        host.created=SdfLayer::FindOrOpen(creationUri);
                        check(host.created && host.created==SdfLayer::Find(creationUri) && errors.IsClean() &&
                              host.created->PermissionToEdit(),"Export destination was not found in its own host turn");
                    }
                }
                return exported;
            });
            // Serial queue order: the adoption job a successful Export queued
            // has run. A failed Export stays retained.
            host.call([] {});
            hostApi::RetainedCreationInfo after;
            const bool listed=hostApi::nextRetainedCreation(0,&after)==OMNI_USD_LIVE_OK;
            check(listed==(listedInside && !success) && (!listed || after.subjectId==subject),
                  "USD Export retention after its outer call disagrees with its outcome");
            retainedSubject=listed ? subject : 0;
            const std::string prefix="Live file format operation failed (result=";
            auto open=[&](long& code) {
                return host.call([&] {
                    TfErrorMark errors;
                    host.created=SdfLayer::FindOrOpen(creationUri);
                    for(const auto& error:errors) {
                        const auto text=error.GetCommentary();
                        if(const auto at=text.find(prefix); at!=std::string::npos) code=std::stol(text.substr(at+prefix.size()));
                    }
                    errors.Clear();
                    return bool(host.created);
                });
            };
            bool closed=false, firstOpened=false;
            long firstResult=0;
            Fields opened;
            if(success && find) {
                // The host reference keeps the adopted destination open.
                const auto value=destination.wait(OMNI_USD_LIVE_STATE_READY);
                check(value.subject_id==subject,"found Export destination changed subject");
                opened=host.call([&] { return fields(*host.created); });
            } else if(success && reopen) {
                // The open waits for the closing destination to release its key.
                firstOpened=open(firstResult);
                check(firstOpened,"reopen after the Export's host turn failed");
                auto value=record<OmniUsdLiveStatus>();
                check(omniUsdLiveGetLayerStatus(creationUri.c_str(),&value)==OMNI_USD_LIVE_OK,"reopened layer lost its status");
                StatusObserver ready; ready.open(runtime.endpoint,value.subject_id); ready.wait(OMNI_USD_LIVE_STATE_READY);
                opened=host.call([&] { return fields(*host.created); });
            } else if(success) {
                destination.wait(OMNI_USD_LIVE_STATE_CLOSED); // No host reference holds the destination.
                closed=true;
            }
            destination.close();
            std::cout << "{\"type\":\"usd-export\",\"hook_ok\":" << success
                      << ",\"diagnosed\":" << diagnosed << ",\"listed\":" << listed
                      << ",\"subject\":" << subject << ",\"state\":" << current.state
                      << ",\"outcome\":" << current.result << ",\"incident\":" << current.incident_id
                      << ",\"pending\":" << current.pending_write_count << ",\"closed\":" << closed
                      << ",\"first_opened\":" << firstOpened << ",\"first_result\":" << firstResult << ",\"fields\":";
            json(opened); std::cout << '}' << std::endl;
        } else if(command=="creation-usd-export-drop-adoption") {
            // The host drops the adoption job a successful Export queued
            // without running it: the job reports CANCELLED and the workflow
            // stays retained.
            int counter=0; check(bool(input>>counter),"missing USD Export seed counter");
            check(!creationContext && !retainedSubject && !host.call([&] { return bool(host.creation) || bool(host.created); }),
                  "dropped adoption fixture already owns a workflow");
            hostApi::PublicationContext* raw=nullptr;
            check(hostApi::createPublicationContext([&](auto job) { host.dispatchPublication(std::move(job)); },
                [&] { return host.current(); },&raw)==OMNI_USD_LIVE_OK,"dropped adoption host context failed");
            creationContext.reset(raw);
            const std::string prefix="Live file format operation failed (result=";
            bool exported=false, held=false;
            long dropped=0;
            host.call([&] {
                TfErrorMark errors;
                host.holdPublication();
                auto seed=SdfLayer::CreateAnonymous("dropped-adoption.usda");
                check(bool(seed),"dropped adoption seed allocation failed");
                seed->SetField(SdfPath::AbsoluteRootPath(),TfToken("counter"),VtValue(counter));
                exported=seed->Export(creationUri);
                check(errors.IsClean(),"dropped adoption Export emitted an error");
                held=host.publicationHeld();
                if(held) host.dropHeldPublication();
                for(const auto& error:errors) {
                    const auto text=error.GetCommentary();
                    if(const auto at=text.find(prefix); at!=std::string::npos) dropped=std::stol(text.substr(at+prefix.size()));
                }
                errors.Clear();
            });
            hostApi::RetainedCreationInfo info;
            const bool listed=hostApi::nextRetainedCreation(0,&info)==OMNI_USD_LIVE_OK && info.liveUri==creationUri;
            if(listed) retainedSubject=info.subjectId;
            std::cout << "{\"type\":\"usd-export-drop-adoption\",\"hook_ok\":" << exported << ",\"held\":" << held
                      << ",\"dropped_result\":" << dropped << ",\"listed\":" << listed << '}' << std::endl;
        } else if(command=="creation-hold-host") {
            host.park();
            std::cout << "{\"type\":\"host-parked\"}" << std::endl;
        } else if(command=="creation-reopen-parked") {
            // A publication for the kept created layer waits behind the parked
            // host. Drop that layer and reopen its URI in the gate, ahead of
            // the queued publication: the closing actor cannot stop before
            // that host job retires, so the reopen is refused at once. After
            // the gate the old layer closes and a second reopen succeeds.
            host.awaitQueued(1);
            auto old=record<OmniUsdLiveStatus>();
            check(omniUsdLiveGetLayerStatus(creationUri.c_str(),&old)==OMNI_USD_LIVE_OK,"parked reopen lost the created status");
            StatusObserver previous;
            previous.open(runtime.endpoint,old.subject_id);
            const std::string prefix="Live file format operation failed (result=";
            long code=0; bool opened=false;
            host.runParked([&] {
                check(bool(host.created),"parked reopen requires the created layer");
                const SdfLayerHandle old=host.created;
                host.created.Reset();
                check(!old,"parked reopen fixture still pins the created layer");
                TfErrorMark errors;
                host.created=SdfLayer::FindOrOpen(creationUri);
                for(const auto& error:errors) {
                    const auto text=error.GetCommentary();
                    if(const auto at=text.find(prefix); at!=std::string::npos) code=std::stol(text.substr(at+prefix.size()));
                }
                errors.Clear();
                opened=bool(host.created);
            },kParkedReopenGuard);
            // The gate returned, so the queued publication runs and retires;
            // the old layer then closes and a reopen succeeds.
            bool reopened=false;
            if(!opened) {
                previous.wait(OMNI_USD_LIVE_STATE_CLOSED);
                reopened=host.call([&] {
                    TfErrorMark errors;
                    host.created=SdfLayer::FindOrOpen(creationUri);
                    check(errors.IsClean(),"reopen after the old layer closed emitted diagnostics");
                    return bool(host.created);
                });
            }
            previous.close();
            auto current=record<OmniUsdLiveStatus>();
            if(opened || reopened) {
                check(omniUsdLiveGetLayerStatus(creationUri.c_str(),&current)==OMNI_USD_LIVE_OK,"reopened layer lost its status");
                StatusObserver ready; ready.open(runtime.endpoint,current.subject_id); current=ready.wait(OMNI_USD_LIVE_STATE_READY);
            }
            std::cout << "{\"type\":\"parked-reopen\",\"opened\":" << opened << ",\"result\":" << code
                      << ",\"reopened\":" << reopened << ",\"state\":" << current.state << ",\"fields\":";
            json(host.call([&] { return host.created ? fields(*host.created) : Fields{}; })); std::cout << '}' << std::endl;
        } else if(command=="creation-open") {
            // SdfLayer::FindOrOpen the created URI on the host, then wait for
            // READY with a failure-only deadline.
            check(!host.call([&] { return bool(host.creation) || bool(host.created); }),"creation open fixture already owns a layer");
            const std::string prefix="Live file format operation failed (result=";
            long code=0;
            const auto opened=host.call([&] {
                TfErrorMark errors;
                host.created=SdfLayer::FindOrOpen(creationUri);
                for(const auto& error:errors) {
                    const auto text=error.GetCommentary();
                    if(const auto at=text.find(prefix); at!=std::string::npos) code=std::stol(text.substr(at+prefix.size()));
                }
                errors.Clear();
                return bool(host.created);
            });
            auto current=record<OmniUsdLiveStatus>();
            if(opened) {
                check(omniUsdLiveGetLayerStatus(creationUri.c_str(),&current)==OMNI_USD_LIVE_OK,"opened creation lost its status");
                StatusObserver ready; ready.open(runtime.endpoint,current.subject_id); current=ready.wait(OMNI_USD_LIVE_STATE_READY);
            }
            std::cout << "{\"type\":\"created-open\",\"opened\":" << opened << ",\"result\":" << code
                      << ",\"state\":" << current.state << ",\"subject\":" << current.subject_id << ",\"fields\":";
            json(host.call([&] { return host.created ? fields(*host.created) : Fields{}; })); std::cout << '}' << std::endl;
        } else if(command=="creation-edit") {
            // Author the actual CreateNew target with plain Sdf; an editable
            // layer then drains the admitted edit off-host.
            std::string key; int value=0;
            check(bool(input>>key>>value) && identifier(key),"invalid creation edit");
            bool diagnosed=false;
            const auto authored=host.call([&] {
                check(bool(host.created),"creation edit requires the CreateNew target");
                TfErrorMark errors;
                { SdfChangeBlock changes; host.created->SetField(SdfPath::AbsoluteRootPath(),TfToken(key),VtValue(value)); }
                diagnosed=!errors.IsClean(); errors.Clear();
                return std::make_pair(host.created,host.created->PermissionToEdit());
            });
            const auto& layer=authored.first; const bool editable=authored.second;
            host.call([] {}); // Serial order: the flush job the edit queued has admitted it.
            const int saved=editable ? int(hostApi::save(*layer,5000)) : -1;
            std::cout << "{\"type\":\"creation-edit\",\"editable\":" << editable << ",\"diagnosed\":" << diagnosed
                      << ",\"saved\":" << saved << ",\"fields\":";
            json(host.call([&] { return fields(*layer); })); std::cout << '}' << std::endl;
        } else if(command=="creation-sdf-save") {
            // An acquired activated creation's layer, or the host's created layer.
            const bool owned=host.call([&] { return bool(host.creation); });
            const auto layer=host.call([&] {
                SdfLayerRefPtr value=host.created;
                if(owned)
                    check(host.creation->getLayer(&value)==OMNI_USD_LIVE_OK && value,
                          "actual Sdf Save requires acquired activated creation");
                check(bool(value),"actual Sdf Save requires a created layer");
                return value;
            });
            const auto saved=hostApi::save(*layer,5000); // Off-host bounded retirement barrier.
            check(saved==OMNI_USD_LIVE_OK,"created layer did not finish pending publication before Sdf Save");
            host.call([&] {
                TfErrorMark errors;
                const auto before=fields(*layer);
                const auto identifier=layer->GetIdentifier();
                check(layer->Save() && layer->Save(true),"default or forced Sdf Save failed");
                check(layer->Export(creationUri),"same-path Sdf Export did not use existing SaveToFile");
                check(fields(*layer)==before && layer->GetIdentifier()==identifier && errors.IsClean(),
                      "existing-layer save/export changed content, identity or emitted diagnostics");
            });
            if(owned) creationReport(saved);
            else {
                auto current=record<OmniUsdLiveStatus>();
                check(omniUsdLiveGetLayerStatus(creationUri.c_str(),&current)==OMNI_USD_LIVE_OK,"created layer status query failed");
                std::cout << "{\"type\":\"created-save\",\"result\":" << saved << ",\"subject\":" << current.subject_id
                          << ",\"state\":" << current.state << ",\"fields\":";
                json(host.call([&] { return fields(*layer); })); std::cout << '}' << std::endl;
            }
        } else if(command=="creation-retained-prepare" || command=="creation-retained-prepare-global" ||
                  command=="creation-retained-prepare-live" || command=="creation-retained-prepare-live-global") {
            check(!creationContext && !retainedSubject && !host.call([&] { return bool(host.creation); }),
                  "duplicate retained creation fixture preparation");
            hostApi::PublicationContext* rawContext=nullptr;
            check(hostApi::createPublicationContext([&](auto job) { host.dispatch(std::move(job)); },
                [&] { return host.current(); },&rawContext)==OMNI_USD_LIVE_OK,"retained host context failed");
            creationContext.reset(rawContext);
            host.call([&] {
                TfErrorMark errors;
                auto seed=SdfLayer::CreateAnonymous("retained-public-creation.usda");
                check(bool(seed),"retained seed allocation failed");
                seed->SetField(SdfPath::AbsoluteRootPath(),TfToken("counter"),VtValue(17));
                uint64_t untouched=UINT64_MAX;
                check(hostApi::prepareRetainedCreation(*seed,nullptr,&untouched)==OMNI_USD_LIVE_INVALID_ARGUMENT &&
                      untouched==UINT64_MAX &&
                      creationContext->prepareRetainedCreation(*seed,creationUri.c_str(),nullptr)==OMNI_USD_LIVE_INVALID_ARGUMENT,
                      "invalid retained preparation changed output");
                auto wrongHost=std::async(std::launch::async,[&] {
                    uint64_t unchanged=UINT64_MAX;
                    check(hostApi::prepareRetainedCreation(*seed,creationUri.c_str(),&unchanged)==OMNI_USD_LIVE_BUSY &&
                          unchanged==UINT64_MAX &&
                          creationContext->prepareRetainedCreation(*seed,creationUri.c_str(),&unchanged)==OMNI_USD_LIVE_BUSY &&
                          unchanged==UINT64_MAX,"wrong host prepared retained seed or changed subject output");
                });
                check(wrongHost.wait_for(std::chrono::seconds(5))==std::future_status::ready,
                      "wrong-host retained preparation blocked");
                wrongHost.get();
                const bool live=command=="creation-retained-prepare-live" || command=="creation-retained-prepare-live-global";
                const bool global=command=="creation-retained-prepare-global" || command=="creation-retained-prepare-live-global";
                const SdfLayer& source=live ? *host.layer : *seed;
                const auto result=global ?
                    hostApi::prepareRetainedCreation(source,creationUri.c_str(),&retainedSubject) :
                    creationContext->prepareRetainedCreation(source,creationUri.c_str(),&retainedSubject);
                check(result==OMNI_USD_LIVE_OK && retainedSubject,"retained creation preparation failed");
                seed->SetField(SdfPath::AbsoluteRootPath(),TfToken("counter"),VtValue(99));
                check(hostApi::prepareRetainedCreation(*seed,creationUri.c_str(),&untouched)==OMNI_USD_LIVE_ALREADY_EXISTS &&
                      untouched==UINT64_MAX,"duplicate retained destination replaced original subject");
                hostApi::Creation* absent=nullptr;
                check(hostApi::prepareCreation(*seed,creationUri.c_str(),&absent)==OMNI_USD_LIVE_ALREADY_EXISTS && !absent,
                      "explicit creation replaced retained destination");
                seed.Reset(); // The registry is the sole creation workflow owner from here on.
                check(errors.IsClean(),"retained creation fixture emitted USD diagnostics");
            });
            auto state=record<OmniUsdLiveStatus>();
            check(omniUsdLiveGetLayerStatus(creationUri.c_str(),&state)==OMNI_USD_LIVE_OK &&
                  state.subject_id==retainedSubject && state.state==OMNI_USD_LIVE_STATE_OPENING &&
                  !state.pending_write_count,"retained preparation lost public pre-RPC status");
            retainedReport(OMNI_USD_LIVE_OK);
        } else if(command=="creation-retained-checks") {
            check(retainedSubject,"retained checks require a subject");
            hostApi::Creation* absent=nullptr;
            check(hostApi::acquireRetainedCreation(retainedSubject,&absent)==OMNI_USD_LIVE_BUSY && !absent &&
                  hostApi::releaseRetainedCreation(retainedSubject)==OMNI_USD_LIVE_BUSY,
                  "wrong host acquired or cancelled retained workflow");
            check(hostApi::acquireRetainedCreation(UINT64_MAX,&absent)==OMNI_USD_LIVE_NOT_FOUND && !absent &&
                  hostApi::releaseRetainedCreation(UINT64_MAX)==OMNI_USD_LIVE_NOT_FOUND &&
                  hostApi::nextRetainedCreation(0,nullptr)==OMNI_USD_LIVE_INVALID_ARGUMENT &&
                  hostApi::acquireRetainedCreation(retainedSubject,nullptr)==OMNI_USD_LIVE_INVALID_ARGUMENT,
                  "invalid retained lookup accepted or changed owner output");
            retainedReport(OMNI_USD_LIVE_OK);
        } else if(command=="creation-retained-replace-context") {
            check(bool(creationContext),"retained context already destroyed");
            check(hostApi::setPublicationContext([](auto) { throw std::runtime_error("used replacement dispatcher"); },
                [] { throw std::runtime_error("used replacement host predicate"); return false; })==OMNI_USD_LIVE_OK,
                "retained context replacement failed");
            host.call([&] {
                auto seed=SdfLayer::CreateAnonymous("retained-context-sentinel.usda");
                uint64_t unchanged=UINT64_MAX;
                check(creationContext->prepareRetainedCreation(*seed,creationUri.c_str(),&unchanged)==OMNI_USD_LIVE_NOT_INITIALIZED &&
                      unchanged==UINT64_MAX,"replaced context rebound retained preparation");
            });
            creationContext.reset();
            retainedReport(OMNI_USD_LIVE_OK);
        } else if(command=="creation-retained-acquire") {
            host.call([&] {
                hostApi::Creation* raw=nullptr;
                check(hostApi::acquireRetainedCreation(retainedSubject,&raw)==OMNI_USD_LIVE_OK && raw,
                      "original host could not acquire retained creation");
                host.creation.reset(raw);
                if(host.created) {
                    SdfLayerRefPtr same;
                    const auto layerResult=host.creation->getLayer(&same);
                    if(layerResult==OMNI_USD_LIVE_OK)
                        check(same==host.created,"acquired creation replaced actual CreateNew target identity");
                    else {
                        auto failed=record<OmniUsdLiveStatus>();
                        check(layerResult==OMNI_USD_LIVE_BUSY && !same &&
                              omniUsdLiveGetLayerStatus(creationUri.c_str(),&failed)==OMNI_USD_LIVE_OK &&
                              failed.state==OMNI_USD_LIVE_STATE_FAILED &&
                              failed.result!=OMNI_USD_LIVE_OK,
                              "acquired successful CreateNew target was not finalized");
                    }
                }
                check(hostApi::acquireRetainedCreation(retainedSubject,&raw)==OMNI_USD_LIVE_INVALID_ARGUMENT && raw==host.creation.get(),
                      "retained acquire overwrote a nonnull owner");
                hostApi::Creation* missing=nullptr;
                check(hostApi::acquireRetainedCreation(retainedSubject,&missing)==OMNI_USD_LIVE_NOT_FOUND && !missing &&
                      hostApi::releaseRetainedCreation(retainedSubject)==OMNI_USD_LIVE_NOT_FOUND,
                      "retained transfer left a second registry owner");
            });
            hostApi::RetainedCreationInfo end;
            check(hostApi::nextRetainedCreation(0,&end)==OMNI_USD_LIVE_NOT_FOUND,"transferred entry remains discoverable");
            creationReport(OMNI_USD_LIVE_OK);
        } else if(command=="creation-retained-release") {
            retainedReport(host.call([&] { return hostApi::releaseRetainedCreation(retainedSubject); }));
        } else if(command=="creation-retained-list") retainedReport(OMNI_USD_LIVE_OK);
        else if(command=="creation-prepare" || command=="creation-prepare-live") {
            check(!creationContext,"duplicate creation fixture preparation");
            hostApi::PublicationContext* rawContext=nullptr;
            check(hostApi::createPublicationContext([&](auto job) { host.dispatch(std::move(job)); },
                [&] { return host.current(); },&rawContext)==OMNI_USD_LIVE_OK,"creation host context failed");
            creationContext.reset(rawContext);
            host.call([&] {
                TfErrorMark errors;
                auto seed=SdfLayer::CreateAnonymous("public-creation.usda");
                check(bool(seed),"ordinary seed failed");
                seed->SetField(SdfPath::AbsoluteRootPath(),TfToken("counter"),VtValue(17));
                hostApi::Creation* owner=nullptr;
                const std::string oversizedUri(65537,'x');
                check(hostApi::prepareCreation(*seed,oversizedUri.c_str(),&owner)==OMNI_USD_LIVE_RESOURCE_EXHAUSTED && !owner,
                      "creation URI limit not enforced before preparation");
                seed->SetField(SdfPath::AbsoluteRootPath(),TfToken("counter"),VtValue(short(7)));
                check(hostApi::prepareCreation(*seed,creationUri.c_str(),&owner)==OMNI_USD_LIVE_UNSUPPORTED && !owner,
                      "unsupported seed value lost classified result");
                seed->SetField(SdfPath::AbsoluteRootPath(),TfToken("counter"),VtValue(std::string(65536,'x')));
                check(hostApi::prepareCreation(*seed,creationUri.c_str(),&owner)==OMNI_USD_LIVE_RESOURCE_EXHAUSTED && !owner,
                      "seed ignored runtime update limit");
                seed->SetField(SdfPath::AbsoluteRootPath(),TfToken("counter"),VtValue(17));
                check(hostApi::prepareCreation(*seed,nullptr,&owner)==OMNI_USD_LIVE_INVALID_ARGUMENT && !owner,
                      "null creation URI changed owner");
                check(hostApi::prepareCreation(*seed,creationUri.c_str(),nullptr)==OMNI_USD_LIVE_INVALID_ARGUMENT,
                      "null creation output accepted");
                const bool live=command=="creation-prepare-live";
                check(hostApi::prepareCreation(live ? *host.layer : *seed,creationUri.c_str(),&owner)==OMNI_USD_LIVE_OK && owner,
                      "public creation prepare failed");
                host.creation.reset(owner);
                check(hostApi::prepareCreation(*seed,creationUri.c_str(),&owner)==OMNI_USD_LIVE_INVALID_ARGUMENT && owner==host.creation.get(),
                      "creation replaced existing output");
                check(owner->getOutcome(nullptr)==OMNI_USD_LIVE_INVALID_ARGUMENT &&
                      owner->getLayer(nullptr)==OMNI_USD_LIVE_INVALID_ARGUMENT,"creation accepted null outputs");
                auto other=std::async(std::launch::async,[&] {
                    hostApi::Creation* absent=nullptr;
                    check(hostApi::prepareCreation(*seed,creationUri.c_str(),&absent)==OMNI_USD_LIVE_BUSY && !absent,
                          "wrong-host prepare succeeded");
                    SdfLayerRefPtr sentinel=seed;
                    check(owner->start()==OMNI_USD_LIVE_BUSY && owner->prepareLayer(0)==OMNI_USD_LIVE_BUSY &&
                          owner->activate()==OMNI_USD_LIVE_BUSY && owner->close()==OMNI_USD_LIVE_BUSY &&
                          owner->getLayer(&sentinel)==OMNI_USD_LIVE_BUSY && sentinel==seed,
                          "wrong-host creation method succeeded or changed output");
                });
                other.get();
                seed->SetField(SdfPath::AbsoluteRootPath(),TfToken("counter"),VtValue(99));
                seed.Reset(); // No seed layer owner survives prepare.
                check(errors.IsClean(),"creation preparation emitted USD diagnostics");
            });
            check(creationContext->close()==OMNI_USD_LIVE_BUSY,"prepared creation did not retain original host context");
            creationReport(OMNI_USD_LIVE_OK);
        } else if(command=="creation-start") {
            creationReport(host.call([&] { return host.creation->start(); }));
        } else if(command=="creation-wait") {
            uint32_t timeout=0; check(bool(input>>timeout),"missing creation wait deadline");
            auto* owner=host.call([&] { return host.creation.get(); });
            check(owner,"missing creation owner"); creationReport(owner->wait(timeout));
        } else if(command=="creation-layer") {
            creationReport(host.call([&] { return host.creation->prepareLayer(5000); }));
        } else if(command=="creation-activate") {
            creationReport(host.call([&] { return host.creation->activate(); }));
        } else if(command=="creation-close") {
            creationReport(host.call([&] { return host.creation->close(); }));
        } else if(command=="state") report("state");
        else if(command=="diagnostics" || command=="diagnostics-history") {
            auto options=record<OmniUsdLiveDiagnosticExportOptions>();
            options.max_output_bytes=256*1024; // No history; only bounded status and metrics.
            if(command=="diagnostics-history") options.max_records=128;
            std::string bytes(options.max_output_bytes,'\0'); uint64_t size=bytes.size();
            check(omniUsdLiveCopyDiagnostics(runtime.endpoint,&options,bytes.data(),&size)==OMNI_USD_LIVE_OK &&
                  size>1 && size<=bytes.size() && bytes[size-1]=='\0',"public diagnostic snapshot failed");
            bytes.resize(size-1);
            std::cout << "{\"type\":\"diagnostics\",\"snapshot\":" << bytes << '}' << std::endl;
        }
        else if(command=="auth-events") auth.report(runtime.endpoint);
        else if(command=="observe-status" || command=="await-status") {
            uint32_t expected=OMNI_USD_LIVE_STATE_READY;
            if(command=="observe-status") observer.open(runtime.endpoint,status().subject_id);
            else check(bool(input>>expected),"missing expected public status");
            const auto value=observer.wait(expected);
            std::cout << "{\"type\":\"status\",\"subject\":\"" << value.subject_id << "\",\"state\":" << value.state
                      << ",\"revision\":" << value.revision << ",\"incident\":" << value.incident_id
                      << ",\"pending\":" << value.pending_write_count << '}' << std::endl;
        }
        else if(command=="cached-open") {
            // FindOrOpen of the attached root layer, by its URI and by a
            // percent-encoded alias, returns the same layer.
            host.call([&] {
                TfErrorMark errors;
                auto alias=uri; alias.replace(alias.rfind("root.live"),9,"%72oot.live");
                check(SdfLayer::FindOrOpen(uri)==host.layer && SdfLayer::FindOrOpen(alias)==host.layer && errors.IsClean(),
                      "cached open of the attached layer failed");
            });
            report("cached-open");
        } else if(command=="sdf-save-timestamp") {
            // Author and retire the edit, then force an Sdf Save so it reaches
            // the file format, and ask the resolver for the layer's
            // modification time in the same host call.
            std::string key; int value=0;
            check(bool(input>>key>>value) && identifier(key),"invalid direct field edit before timestamp");
            host.call([&] { SdfChangeBlock changes; host.layer->SetField(SdfPath::AbsoluteRootPath(),TfToken(key),VtValue(value)); });
            host.call([] {}); // Fence: the flush job the change block queued has run.
            check(hostApi::save(*host.call([&] { return host.layer; }),5000)==OMNI_USD_LIVE_OK,"attached layer edit did not retire");
            const auto modified=host.call([&] {
                TfErrorMark errors;
                check(host.layer->Save(true) && errors.IsClean(),"Sdf Save of the attached layer failed");
                const auto time=ArGetResolver().GetModificationTimestamp(uri,ArResolvedPath(uri));
                check(time.IsValid() && errors.IsClean(),"attached layer modification timestamp failed");
                return std::llround(time.GetTime()*1000000.0);
            });
            std::cout << "{\"type\":\"save-timestamp\",\"modified_us\":" << modified << '}' << std::endl;
        } else if(command=="reopen-root") {
            // Release the root layer in one host turn and reopen its URI in a
            // later turn, then wait for the new subject's readiness.
            const auto previous=status().subject_id;
            host.call([&] {
                const SdfLayerHandle old=host.layer;
                host.layer.Reset();
                check(!old,"root reopen fixture still pins the released layer");
            });
            // The released layer retired the startup context, which was closed
            // while attached; the reopen needs a configured one.
            if(!creationContext) {
                hostApi::PublicationContext* raw=nullptr;
                check(hostApi::createPublicationContext([&](auto job) { host.publish(std::move(job)); },
                    [&] { return host.current(); },&raw)==OMNI_USD_LIVE_OK,"reopen host context failed");
                creationContext.reset(raw);
            }
            host.call([&] {
                TfErrorMark errors;
                host.layer=SdfLayer::FindOrOpen(uri);
                check(host.layer && errors.IsClean(),"reopen of the released root layer failed");
            });
            // The reopen's FindOrOpen has returned, so the new subject exists; the callback's initial snapshot reports a READY reached before registration.
            const auto reopened=status();
            check(reopened.subject_id!=previous,"reopen kept the released subject");
            { StatusObserver ready; ready.open(runtime.endpoint,reopened.subject_id); ready.wait(OMNI_USD_LIVE_STATE_READY); }
            report("reopened");
        }
        else if(command=="await-sequences") {
            // Polls the layer status until it reports exactly these read and
            // applied sequences; they refresh without a status notification.
            // An applied position's effect is installed in the host layer.
            uint64_t read=0, applied=0; check(bool(input>>read>>applied),"missing expected status sequences");
            const auto deadline=Clock::now()+std::chrono::seconds(10);
            for(;;) {
                const auto value=status();
                if(value.read_sequence==read && value.applied_sequence==applied) break;
                check(Clock::now()<deadline,"layer status did not reach the expected sequences");
                std::this_thread::yield();
            }
            report("sequences");
        }
        else if(command=="facade-checks") {
            const auto value=host.facadeChecks();
            std::cout << "{\"type\":\"facade-checked\",\"fields\":"; json(value); std::cout << '}' << std::endl;
        }
        else if(command=="usd-set" || command=="usd-set-admit" || command=="usd-set-many") {
            // Plain Sdf authoring of root int fields in one change block. The
            // reply follows a host fence, so the flush job the block queued
            // has run; `pending` is the layer's admitted, not yet applied
            // intent count, which admission publishes synchronously. While the
            // issuance window is full that flush coalesces the block with later
            // ones; `usd-set-admit` then saves with no wait in the same host
            // call, which admits the block at once as its own intent, so a
            // campaign can bind one intent to each command.
            size_t count=0; std::vector<std::pair<std::string,int>> edits;
            const bool admit=command=="usd-set-admit";
            if(command!="usd-set-many") {
                check(bool(input>>count) && count<=64,"invalid direct edit count");
                for(size_t i=0;i<count;++i) { std::string key; int value=0;
                    check(bool(input>>key>>value) && identifier(key),"invalid direct field edit"); edits.emplace_back(key,value); }
            } else {
                // Fields many0..many<N-1> with values 1..N.
                check(bool(input>>count) && count>0 && count<=4096,"invalid direct edit count");
                for(size_t i=0;i<count;++i) edits.emplace_back("many"+std::to_string(i),static_cast<int>(i)+1);
            }
            host.call([&] {
                check(host.layer->PermissionToEdit(),"live layer must be editable for direct authoring");
                {
                    SdfChangeBlock changes;
                    for(const auto& [key,value]:edits) host.layer->SetField(SdfPath::AbsoluteRootPath(),TfToken(key),VtValue(value));
                }
                if(admit) {
                    const auto saved=hostApi::save(*host.layer,0);
                    check(saved==OMNI_USD_LIVE_OK || saved==OMNI_USD_LIVE_TIMEOUT || saved==OMNI_USD_LIVE_BUSY,
                          ("admitting save failed: "+std::to_string(int(saved))).c_str());
                }
            });
            host.call([] {}); // Fence: the flush job the change block queued has run.
            const auto value=status();
            std::cout << "{\"type\":\"authored\",\"count\":" << count << ",\"result\":0,\"status_result\":" << value.result
                      << ",\"pending\":" << value.pending_write_count << '}' << std::endl;
        } else if(command=="usd-set-save" || command=="usd-set-sdf-save") {
            // Author, then save, inside one host call: the flush job queued by
            // the edit cannot run first, so only the save path admits the edit.
            std::string key; int value=0;
            check(bool(input>>key>>value) && identifier(key),"invalid direct field edit before save");
            const auto outcome=host.call([&] {
                check(host.layer->PermissionToEdit(),"live layer must be editable for direct authoring");
                { SdfChangeBlock changes; host.layer->SetField(SdfPath::AbsoluteRootPath(),TfToken(key),VtValue(value)); }
                // The host is busy in this call, so an off-host save sees the pending edit.
                const auto offHost=std::async(std::launch::async,[&] { return hostApi::save(*host.layer,0); }).get();
                int result=0;
                if(command=="usd-set-save") result=hostApi::save(*host.layer,5000);
                else { TfErrorMark errors; result=host.layer->Save() ? 0 : -1; errors.Clear(); }
                return std::make_tuple(result,offHost,status().pending_write_count);
            });
            std::cout << "{\"type\":\"saved\",\"result\":" << std::get<0>(outcome) << ",\"offhost\":" << std::get<1>(outcome)
                      << ",\"pending\":" << std::get<2>(outcome) << '}' << std::endl;
        } else if(command=="usd-stage-set") {
            // Compose a UsdStage over the live layer with at least two USD work
            // threads, so composition reads the layer's backing from worker
            // threads, then author one root field on the live layer in a change
            // block while the stage is open. An anonymous session layer defines
            // `prims` root prims with four children each; composing every one
            // queries the live layer in the same layer stack. A limit below two
            // is raised to two for this command and reported as forced. USD
            // caps the raise at the process's physical concurrency and
            // PXR_WORK_THREAD_LIMIT overrides it, so the reply reports the
            // limit in effect during composition.
            // A refusal of the edit (the plugin's typed rejection is a
            // std::runtime_error) is reported with its message, not thrown.
            size_t prims=0; std::string key; int value=0;
            check(bool(input>>prims>>key>>value) && prims>0 && prims<=1024 && identifier(key),"invalid stage edit");
            const unsigned defaultLimit=WorkGetConcurrencyLimit();
            const bool forced=defaultLimit<2;
            if(forced) WorkSetConcurrencyLimit(2);
            struct Outcome { int result=0; std::string refused; size_t composed=0; unsigned concurrency=0; };
            // The queued job owns copies of the command's inputs.
            const auto outcome=host.call([&host,prims,key,value] {
                check(host.layer->PermissionToEdit(),"live layer must be editable for direct authoring");
                Outcome out;
                out.concurrency=WorkGetConcurrencyLimit();
                const auto session=SdfLayer::CreateAnonymous("threaded-composition-session");
                {
                    SdfChangeBlock changes;
                    for(size_t i=0;i<prims;++i) {
                        const auto prim=SdfPrimSpec::New(session->GetPseudoRoot(),"Compose"+std::to_string(i),SdfSpecifierDef,"Scope");
                        check(bool(prim),"session prim creation failed");
                        for(int child=0;child<4;++child)
                            check(bool(SdfPrimSpec::New(prim,"Child"+std::to_string(child),SdfSpecifierDef,"Scope")),
                                  "session child creation failed");
                    }
                }
                const auto stage=UsdStage::Open(host.layer,session,UsdStage::LoadAll);
                check(bool(stage) && stage->GetRootLayer()==host.layer,"threaded composition stage open failed");
                for(const auto& prim:stage->Traverse()) { (void)prim; ++out.composed; }
                try { SdfChangeBlock changes; host.layer->SetField(SdfPath::AbsoluteRootPath(),TfToken(key),VtValue(value)); }
                catch(const std::exception& refused) {
                    out.result=-1;
                    for(const char c:std::string(refused.what())) out.refused+=(c=='"' || c=='\\') ? '\'' : c;
                }
                return out;
            });
            if(forced) WorkSetConcurrencyLimit(defaultLimit);
            host.call([] {}); // Fence: the flush job the change block queued has run.
            const auto current=status();
            std::cout << "{\"type\":\"stage-authored\",\"result\":" << outcome.result << ",\"refused\":\"" << outcome.refused
                      << "\",\"composed\":" << outcome.composed << ",\"concurrency\":" << outcome.concurrency
                      << ",\"forced\":" << (forced ? "true" : "false") << ",\"status_result\":" << current.result
                      << ",\"state\":" << current.state << ",\"pending\":" << current.pending_write_count << '}' << std::endl;
        } else if(command=="usd-prim") {
            // Plain Sdf namespace authoring, one change block per command.
            std::string op,name,other; int value=0;
            check(bool(input>>op>>name) && identifier(name),"invalid direct prim edit");
            if(op=="define" || op=="set" || op=="create-remove") check(bool(input>>value),"missing direct prim value");
            else if(op=="rename") check(bool(input>>other) && identifier(other),"missing direct prim rename target");
            else check(op=="remove" || op=="create-remove","unknown direct prim operation");
            host.call([&] {
                auto& layer=host.layer;
                check(layer->PermissionToEdit(),"live layer must be editable for direct authoring");
                const SdfPath path("/"+name);
                auto define=[&] {
                    const auto prim=SdfCreatePrimInLayer(layer,path);
                    check(bool(prim),"direct prim creation failed");
                    prim->SetSpecifier(SdfSpecifierDef); prim->SetTypeName("Scope");
                    const auto attribute=SdfAttributeSpec::New(prim,valueName(),SdfValueTypeNames->Int);
                    check(bool(attribute),"direct attribute creation failed");
                    attribute->SetDefaultValue(VtValue(value));
                    return prim;
                };
                SdfChangeBlock changes;
                if(op=="define") (void)define();
                else if(op=="create-remove") layer->RemoveRootPrim(define());
                else if(op=="set") {
                    check(bool(layer->GetAttributeAtPath(path.AppendProperty(valueName()))),"direct prim value target missing");
                    layer->SetField(path.AppendProperty(valueName()),SdfFieldKeys->Default,VtValue(value));
                } else if(op=="remove") {
                    const auto prim=layer->GetPrimAtPath(path);
                    check(bool(prim),"direct prim removal target missing");
                    layer->RemoveRootPrim(prim);
                } else {
                    SdfBatchNamespaceEdit edit; edit.Add(path,SdfPath("/"+other));
                    check(layer->Apply(edit),"direct prim rename failed");
                }
            });
            std::cout << "{\"type\":\"authored\",\"result\":0}" << std::endl;
        } else if(command=="prims" || command=="wait-prims") {
            Prims expected; bool waiting=command=="wait-prims";
            if(waiting) {
                size_t count=0; check(bool(input>>count) && count<=1024,"invalid expected prim count");
                for(size_t i=0;i<count;++i) { std::string path; int value=0;
                    check(bool(input>>path>>value) && !expected.values.count(path),"invalid expected prim");
                    expected.paths.push_back(path); expected.values.emplace(path,value); }
                std::sort(expected.paths.begin(),expected.paths.end());
            }
            const auto deadline=Clock::now()+std::chrono::seconds(5);
            auto current=host.readPrims();
            while(waiting && !(current==expected)) {
                // Capture before the host query so a racing publication is not missed.
                const auto revision=host.publicationRevision();
                current=host.readPrims();
                if(current==expected) break;
                check(status().result==OMNI_USD_LIVE_OK,"public exact prim convergence failed");
                host.awaitPublication(revision,deadline);
            }
            std::cout << "{\"type\":\"prims\",\"result\":" << status().result << ",\"paths\":[";
            for(size_t i=0;i<current.paths.size();++i) std::cout << (i ? "," : "") << '"' << current.paths[i] << '"';
            std::cout << "],\"values\":"; json(current.values); std::cout << '}' << std::endl;
        } else if(command=="hold-dispatch") {
            host.holdDispatch(); std::cout << "{\"type\":\"armed\",\"result\":0}" << std::endl;
        } else if(command=="release-dispatch") {
            check(host.dispatchHeld(),"no dispatch was withheld");
            host.releaseDispatch(); host.call([] {}); // Fence: the released job has run.
            report("state");
        } else if(command=="save") {
            uint32_t timeout=0; check(bool(input>>timeout),"missing public save deadline");
            const auto protectedLayer=host.call([&] { return host.layer; });
            const auto result=hostApi::save(*protectedLayer,timeout);
            std::cout << "{\"type\":\"saved\",\"result\":" << result << '}' << std::endl;
        } else if(command=="wait") {
            size_t count=0; check(bool(input>>count) && count<=1024,"invalid public expected field count"); Fields expected;
            for(size_t i=0;i<count;++i) { std::string key; int value; check(bool(input>>key>>value) && identifier(key) && expected.emplace(key,value).second,"invalid expected field"); }
            const auto deadline=Clock::now()+std::chrono::seconds(5);
            for(;;) {
                // Capture before the host query so a publication racing the
                // read cannot be missed. Query jobs do not advance this event.
                const auto revision=host.publicationRevision();
                if(host.read()==expected) break;
                check(status().result==OMNI_USD_LIVE_OK,"public exact scene convergence failed");
                host.awaitPublication(revision,deadline);
            }
            report("state");
        } else if(command=="reset-context") {
            check(hostApi::setPublicationContext({},{})==OMNI_USD_LIVE_OK,"public context reset failed"); report("context-reset");
        } else if(command=="failed") {
            { StatusObserver terminal; terminal.open(runtime.endpoint,status().subject_id); terminal.wait(OMNI_USD_LIVE_STATE_FAILED); }
            report("state");
        } else if(command=="export") {
            std::string path; check(bool(input>>path),"missing public export path");
            const auto before=status(); const auto value=host.read();
            // SdfLayer::Export of the failed live layer to an ordinary local
            // file is the whole recovery artifact (design section 13).
            const auto exported=host.call([&] {
                check(bool(host.layer),"public layer already dropped");
                if(!host.layer->Export(path)) return std::optional<Fields>{};
                const auto file=SdfLayer::OpenAsAnonymous(path);
                check(bool(file),"exported live layer did not open");
                return std::optional<Fields>{exportedFields(*file)};
            });
            if(exported) exports.insert_or_assign(path,readFile(path));
            const auto after=status();
            check(after.subject_id==before.subject_id && after.state==before.state && after.result==before.result &&
                  after.pending_write_count==before.pending_write_count && after.pending_bytes==before.pending_bytes &&
                  host.read()==value,"SdfLayer::Export changed the failed layer or its retained work");
            std::cout << "{\"type\":\"exported\",\"result\":" << (exported ? 0 : 1) << ",\"fields\":";
            json(exported ? *exported : Fields{}); std::cout << '}' << std::endl;
        } else if(command=="close") {
            observer.close();
            const auto values=host.read(); host.drop();
            check(hostApi::setPublicationContext({},{})==OMNI_USD_LIVE_OK,"public final context reset failed");
            auth.close(); runtime.close();
            const auto contextDeadline = Clock::now() + std::chrono::seconds(5);
            while (context->close() == OMNI_USD_LIVE_BUSY) {
                check(Clock::now() < contextDeadline, "host context did not retire after layer close");
                std::this_thread::yield();
            }
            check(context->close() == OMNI_USD_LIVE_OK, "host context final close failed");
            for(const auto& entry:exports) check(readFile(entry.first)==entry.second,"exported bytes changed after all public owners retired");
            std::cout << "{\"type\":\"closed\",\"fields\":"; json(values); std::cout << '}' << std::endl; return 0;
        } else throw std::runtime_error("unknown public plugin command");
        check(!(input>>extra),"unexpected public plugin command argument");
    }
    throw std::runtime_error("public plugin input ended without close");
}
}
int main(int argc,char** argv) {
    try { return run(argc,argv); }
    catch(const std::exception& error) { std::cerr << "public shared-plugin consumer failed: " << error.what() << '\n'; return 1; }
}
