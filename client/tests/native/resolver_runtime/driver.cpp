// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <omni/usd/live/Live.h>
#include <pxr/base/plug/registry.h>
#include <pxr/base/tf/errorMark.h>
#include <pxr/usd/ar/resolver.h>
#include <pxr/usd/ar/resolvedPath.h>
#include <pxr/usd/ar/timestamp.h>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <future>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <thread>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE
namespace {
using Clock=std::chrono::steady_clock;
void check(bool valid,const char* reason) { if(!valid) throw std::runtime_error(reason); }
template<class T> T record() { T value{}; value.struct_size=sizeof(value); value.abi_version=OMNI_USD_LIVE_ABI_VERSION; return value; }
template<class Operation> std::string uri(Operation operation) {
    uint64_t size=0; check(operation(nullptr,&size)==OMNI_USD_LIVE_OK,"URI size query failed");
    std::vector<char> bytes(size); check(operation(bytes.data(),&size)==OMNI_USD_LIVE_OK,"URI construction failed");
    return bytes.data();
}
// 0 when the resolver declines quietly (empty identifier, no diagnostic);
// otherwise the typed result of its single diagnostic.
template<class Operation> int identifierOutcome(Operation operation) {
    TfErrorMark mark;
    check(operation().empty(),"rejected or declined path produced an identifier");
    if(mark.IsClean()) return 0;
    int outcome=-1;
    for(const auto& error:mark) {
        const auto* typed=error.GetInfo<OmniUsdLiveResult>();
        check(typed && outcome==-1,"untyped or repeated resolver diagnostic"); outcome=int(*typed);
    }
    mark.Clear(); return outcome;
}
void expectedError(TfErrorMark& mark,OmniUsdLiveResult result) {
    bool found=false;
    for(const auto& error:mark) {
        const auto* typed=error.GetInfo<OmniUsdLiveResult>();
        check(typed && *typed==result,"unexpected/untyped resolver diagnostic"); found=true;
        check(error.GetCommentary().find("synthetic-writer")==std::string::npos,"credential in USD diagnostic");
        check(error.GetCommentary().find("live://")==std::string::npos,"resource in USD diagnostic");
    }
    mark.Clear(); check(found,"resolver omitted typed failure diagnostic");
}
struct RuntimeOwner {
    bool initialized=false; OmniUsdLiveEndpoint endpoint=0;
    void close() {
        if(endpoint) {
            const auto deadline=Clock::now()+std::chrono::seconds(5);
            for(;;) {
                const auto result=omniUsdLiveUnregisterEndpoint(endpoint,0);
                if(result==OMNI_USD_LIVE_OK) break;
                check(result==OMNI_USD_LIVE_BUSY && Clock::now()<deadline,"metadata retained endpoint after cleanup");
                std::this_thread::yield();
            }
            endpoint=0;
        }
        if(initialized) { check(omniUsdLiveShutdown(5000)==OMNI_USD_LIVE_OK,"runtime cleanup failed"); initialized=false; }
    }
    ~RuntimeOwner() { try { close(); } catch(...) { std::cerr<<"resolver runtime cleanup failed\n"; std::_Exit(2); } }
};
struct Worker {
    std::string path; std::promise<void> done; std::atomic<unsigned> calls{0};
    static void OMNI_USD_LIVE_CALL callback(void* user,const OmniUsdLiveDiagnosticRecord*) {
        auto& self=*static_cast<Worker*>(user);
        if(self.calls.fetch_add(1)) return;
        try {
            TfErrorMark mark;
            check(ArGetResolver().Resolve(self.path).empty(),"callback worker performed blocking metadata resolve");
            expectedError(mark,OMNI_USD_LIVE_BUSY); self.done.set_value();
        } catch(...) { self.done.set_exception(std::current_exception()); }
    }
};
int run(int argc,char** argv) {
    check(argc==5,"usage: resolver_driver authority plugInfo mode session");
    const std::string authority=argv[1], mode=argv[3], session=argv[4];
    PlugRegistry::GetInstance().RegisterPlugins(argv[2]);
    auto& resolver=ArGetResolver();
    const auto sessionUri=uri([&](char* out,uint64_t* size){return omniUsdLiveMakeSessionUri(authority.c_str(),session.c_str(),out,size);});
    const auto layerUri=uri([&](char* out,uint64_t* size){return omniUsdLiveMakeLayerUri(sessionUri.c_str(),"root.live",out,size);});
    const auto newUri=uri([&](char* out,uint64_t* size){return omniUsdLiveMakeLayerUri(sessionUri.c_str(),"new.live",out,size);});
    auto alias=layerUri; alias.replace(alias.rfind("root.live"),9,"%72oot.live");
    RuntimeOwner owner;
    auto options=record<OmniUsdLiveRuntimeOptions>(); options.initial_open_timeout_ms=mode=="auth-deadline"?1:mode=="timeout-retry"?250:2000;
    // auth-deadline: a 1 ms metadata deadline precedes the auth wait; starvation is classified before every deadline check.
    options.default_auth_wait_timeout_ms=mode=="auth-deadline"?30000:100; options.retry_initial_ms=5; options.retry_maximum_ms=20; options.retry_max_attempts=3;
    options.reconnect_initial_ms=5; options.reconnect_maximum_ms=20; options.reconnect_max_attempts=3;
    check(omniUsdLiveInitialize(&options)==OMNI_USD_LIVE_OK,"runtime initialize failed"); owner.initialized=true;
    Worker worker; worker.path=layerUri; auto workerDone=worker.done.get_future(); OmniUsdLiveCallbackHandle callback=0;
    if(mode=="worker") {
        auto diagnostic=record<OmniUsdLiveDiagnosticOptions>();
        const uint32_t code=OMNI_USD_LIVE_RECORD_ENDPOINT_REGISTERED;
        diagnostic.event_codes=&code; diagnostic.event_code_count=1;
        check(omniUsdLiveRegisterDiagnosticCallback(&diagnostic,&worker,Worker::callback,&callback)==OMNI_USD_LIVE_OK,"worker observer registration failed");
    }
    if(mode!="unregistered") {
        auto endpoint=record<OmniUsdLiveEndpointOptions>(); endpoint.authority=authority.c_str();
        endpoint.access_token=mode=="auth" || mode=="auth-deadline"?"":"synthetic-writer";
        check(omniUsdLiveRegisterEndpoint(&endpoint,&owner.endpoint)==OMNI_USD_LIVE_OK,"endpoint registration failed");
    }
    std::string canonical=resolver.CreateIdentifier(alias,ArResolvedPath()),relative,resolved;
    check(canonical==layerUri,"production URI handler did not canonicalize alias");
    double timestamp=0; bool timestampValid=false; std::vector<double> timestamps;
    std::vector<std::pair<std::string,int>> outcomes;
    if(mode=="metadata") {
        const auto anchor=uri([&](char* out,uint64_t* size){return omniUsdLiveMakeLayerUri(sessionUri.c_str(),"nested/root.live",out,size);});
        relative=resolver.CreateIdentifier("child.live",ArResolvedPath(anchor));
        check(relative==sessionUri+"/nested/child.live","nested live anchor lost directory/session");
        const auto creation=resolver.CreateIdentifierForNewAsset("new.live",ArResolvedPath(anchor));
        check(creation==sessionUri+"/nested/new.live","new relative identifier differs");
        resolved=resolver.Resolve(alias).GetPathString(); check(resolved==layerUri,"alias metadata resolution failed");
        check(resolver.Resolve(canonical).GetPathString()==layerUri,"canonical metadata resolution failed");
        const auto stamp=resolver.GetModificationTimestamp(canonical,ArResolvedPath(resolved));
        check(stamp.IsValid(),"valid Stat lost timestamp"); timestampValid=true; timestamp=stamp.GetTime();
        std::string reason;
        check(resolver.CanWriteAssetToPath(ArResolvedPath(newUri),&reason),"registered new live path refused custom writer");
        { TfErrorMark mark; check(!resolver.OpenAsset(ArResolvedPath(canonical)),"resolver invented generic byte asset"); expectedError(mark,OMNI_USD_LIVE_UNSUPPORTED); }
        { TfErrorMark mark; check(!resolver.OpenAssetForWrite(ArResolvedPath(newUri),ArResolver::WriteMode::Replace),"resolver invented generic writable asset"); expectedError(mark,OMNI_USD_LIVE_UNSUPPORTED); }
    } else if(mode=="relative-assets") {
        const auto anchor=uri([&](char* out,uint64_t* size){return omniUsdLiveMakeLayerUri(sessionUri.c_str(),"nested/root.live",out,size);});
        // A texture, payload or USD file authored relative to a live layer is an
        // ordinary asset: the live resolver declines it without a USD error.
        for(const char* asset:{"textures/wood.png","../base.usd","payload.usda","nested/model.usdc"}) {
            TfErrorMark mark;
            check(resolver.CreateIdentifier(asset,ArResolvedPath(anchor)).empty(),"ordinary relative asset became a live identifier");
            check(resolver.CreateIdentifierForNewAsset(asset,ArResolvedPath(anchor)).empty(),"ordinary new asset became a live identifier");
            check(mark.IsClean(),"ordinary relative asset posted a USD error");
        }
        relative=resolver.CreateIdentifier("child.live",ArResolvedPath(anchor));
        check(relative==sessionUri+"/nested/child.live","live child lost its anchor");
        // Design 4.3: only the exact, case-sensitive `.live` suffix selects a
        // live identifier; absolute and other-scheme `.live` paths and
        // over-length paths reject with a diagnostic.
        const std::string overLength=std::string(65536,'a')+".live";
        const std::vector<std::pair<std::string,OmniUsdLiveResult>> anchored{
            {"/abs/tex.png",OMNI_USD_LIVE_OK},{"file:x.png",OMNI_USD_LIVE_OK},{"x.LIVE",OMNI_USD_LIVE_OK},
            {"/abs/x.live",OMNI_USD_LIVE_INVALID_ARGUMENT},{"a:b.live",OMNI_USD_LIVE_INVALID_ARGUMENT},
            {overLength,OMNI_USD_LIVE_RESOURCE_EXHAUSTED}};
        for(const auto& [path,expected]:anchored) {
            const auto label=path.size()>64?std::string("over-length.live"):path;
            const auto outcome=identifierOutcome([&]{return resolver.CreateIdentifier(path,ArResolvedPath(anchor));});
            check(outcome==int(expected),"anchored path outcome differs from design section 6");
            check(identifierOutcome([&]{return resolver.CreateIdentifierForNewAsset(path,ArResolvedPath(anchor));})==outcome,
                  "new-asset path outcome differs from identifier outcome");
            outcomes.emplace_back(label,outcome);
        }
        // USD dispatches a scheme-less path with an empty anchor to the primary
        // resolver, so the live resolver's empty-anchor rule is driven directly.
        const auto liveType=TfType::FindByName("OmniUsdLiveResolver");
        check(!liveType.IsUnknown(),"live resolver type is not registered");
        const auto direct=ArCreateResolver(liveType); check(bool(direct),"live resolver could not be constructed");
        const auto emptyAnchor=identifierOutcome([&]{return direct->CreateIdentifier("child.live",ArResolvedPath());});
        check(emptyAnchor==int(OMNI_USD_LIVE_INVALID_ARGUMENT),"relative live path with empty anchor did not reject");
        outcomes.emplace_back("empty-anchor:child.live",emptyAnchor);
    } else if(mode=="timestamps") {
        for(unsigned i=0;i<3;++i) {
            const auto stamp=resolver.GetModificationTimestamp(canonical,ArResolvedPath(canonical));
            check(stamp.IsValid(),"successful timestamp including epoch became invalid");
            timestamps.push_back(stamp.GetTime());
        }
    } else if(mode=="new") {
        check(resolver.CreateIdentifierForNewAsset(newUri,ArResolvedPath())==newUri,"new identifier failed");
        resolved=resolver.ResolveForNewAsset(newUri).GetPathString(); check(resolved==newUri,"registered new path did not resolve");
        std::string reason; check(resolver.CanWriteAssetToPath(ArResolvedPath(newUri),&reason),"custom live creation refused");
    } else if(mode=="retry") {
        resolved=resolver.Resolve(layerUri).GetPathString(); check(resolved==layerUri,"safe metadata retry failed");
    } else if(mode=="timeout-retry") {
        for(unsigned round=0; round<3; ++round) {
            const auto start=Clock::now();
            { TfErrorMark mark; check(resolver.Resolve(layerUri).empty(),"held Stat returned an asset"); expectedError(mark,OMNI_USD_LIVE_TIMEOUT); }
            check(Clock::now()-start<std::chrono::seconds(3),"metadata deadline did not bound resolver wait");
            std::cout<<"{\"type\":\"timeout\",\"round\":"<<round<<"}"<<std::endl;
            std::string command; check(bool(std::getline(std::cin,command)) && command=="continue","missing late-completion release");
            resolved=resolver.Resolve(layerUri).GetPathString(); check(resolved==layerUri,"metadata retry after timeout failed");
            std::cout<<"{\"type\":\"retry\",\"round\":"<<round<<"}"<<std::endl;
        }
    } else if(mode=="worker") {
        check(workerDone.wait_for(std::chrono::seconds(5))==std::future_status::ready,"worker metadata probe not delivered"); workerDone.get();
        check(omniUsdLiveUnregisterCallback(callback,3000)==OMNI_USD_LIVE_OK,"worker observer cleanup failed"); callback=0;
        check(worker.calls==1,"worker probe duplicated");
    } else {
        const auto expected=mode=="auth" || mode=="auth-deadline"?OMNI_USD_LIVE_UNAUTHENTICATED:mode=="permission"?OMNI_USD_LIVE_PERMISSION_DENIED:OMNI_USD_LIVE_NOT_FOUND;
        { TfErrorMark mark; check(resolver.Resolve(layerUri).empty(),"failed metadata returned a live asset"); expectedError(mark,expected); }
        if(mode=="unregistered") {
            TfErrorMark mark; check(resolver.ResolveForNewAsset(newUri).empty(),"unregistered new path resolved"); expectedError(mark,OMNI_USD_LIVE_NOT_FOUND);
        }
    }
    owner.close();
    std::cout<<std::setprecision(17)<<"{\"type\":\"result\",\"canonical\":"<<std::quoted(canonical)
             <<",\"relative\":"<<std::quoted(relative)<<",\"resolved\":"<<std::quoted(resolved)
             <<",\"timestamp_valid\":"<<(timestampValid?"true":"false")<<",\"timestamp\":"<<timestamp<<",\"timestamps\":[";
    for(size_t i=0;i<timestamps.size();++i) { if(i) std::cout<<','; std::cout<<timestamps[i]; }
    std::cout<<"],\"outcomes\":{";
    for(size_t i=0;i<outcomes.size();++i) { if(i) std::cout<<','; std::cout<<std::quoted(outcomes[i].first)<<':'<<outcomes[i].second; }
    // The result is the driver's only record, so a failed write fails the driver.
    std::cout<<"}}"<<std::endl;
    return std::cout.good() ? 0 : 1;
}
}
int main(int argc,char** argv) {
    try { return run(argc,argv); }
    catch(const std::exception& error) { std::cerr<<"resolver driver: "<<error.what()<<'\n'; }
    return 1;
}
