// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "LiveRuntime.h"
#include "LiveDiagnostics.h"
#include "LiveAuth.h"
#include "LiveSession.h"
#include "../support/test_tls.h"
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <future>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

std::string publicDiagnostics();
void liveTestDumpStacks() noexcept; // tests/native/support/StackDump.cpp

namespace {
using namespace omni::usd::live;
template<class T> T sessionRecord() { T v{}; v.struct_size=sizeof(v); v.abi_version=OMNI_USD_LIVE_ABI_VERSION; return v; }
struct Event { unsigned kind; OmniUsdLiveResult result; std::string id,user,app,bytes; };
struct Events {
    std::mutex mutex; std::condition_variable cv; std::vector<Event> values;
    OmniUsdLiveSession session=0; bool hold=false, entered=false, release=false, selfLeave=false, throwAfterHold=false;
    OmniUsdLiveResult selfResult=OMNI_USD_LIVE_INTERNAL_ERROR;
    OmniUsdLiveEndpoint revokeEndpoint=0; bool revokeOnError=false, revoked=false;
    static void OMNI_USD_LIVE_CALL callback(void* user,const OmniUsdLiveSessionEvent* value) {
        auto& s=*static_cast<Events*>(user);
        assert(value && value->struct_size==sizeof(*value) && value->abi_version==OMNI_USD_LIVE_ABI_VERSION);
        Event copy{value->kind,value->result,value->connection_id?value->connection_id:"",
                   value->user_name?value->user_name:"",value->app_name?value->app_name:"",{}};
        if(value->size) { assert(value->bytes); copy.bytes.assign(static_cast<const char*>(value->bytes),value->size); }
        std::unique_lock<std::mutex> lock(s.mutex);
        s.values.push_back(std::move(copy));
        if(s.revokeOnError && value->kind==OMNI_USD_LIVE_EVENT_CONNECTION && value->result==OMNI_USD_LIVE_UNAVAILABLE) {
            // Revoke before returning from this session's serialized callback:
            // the session's worker runs this callback inside its step, so it
            // cannot issue the rejoin until the revocation has happened.
            assert(!s.revoked); const auto endpoint=s.revokeEndpoint;
            lock.unlock(); uint64_t generation=0;
            assert(omniUsdLiveSetAccessToken(endpoint,1,"",0,&generation)==OMNI_USD_LIVE_OK && generation==2);
            lock.lock(); s.revoked=true;
        }
        if(s.hold && value->kind==OMNI_USD_LIVE_EVENT_CONNECTION && value->result==OMNI_USD_LIVE_OK) {
            s.entered=true; s.cv.notify_all();
            assert(s.cv.wait_for(lock,std::chrono::seconds(5),[&]{return s.release;}));
            if(s.throwAfterHold) throw std::runtime_error("synthetic-session-callback-secret");
            if(s.selfLeave) { const auto handle=s.session; lock.unlock(); const auto result=omniUsdLiveLeaveSession(handle,0); lock.lock(); s.selfResult=result; }
        }
        s.cv.notify_all();
    }
    template<class Predicate> void wait(Predicate predicate) {
        std::unique_lock<std::mutex> lock(mutex);
        assert(cv.wait_for(lock,std::chrono::seconds(6),[&]{return predicate(values);}));
    }
    size_t count(unsigned kind) const { size_t n=0; for(const auto& e:values) if(e.kind==kind) ++n; return n; }
};
void OMNI_USD_LIVE_CALL ignoredReceipt(void*,OmniUsdLiveResult) {}
struct SessionRefresh {
    std::atomic<unsigned> calls{0};
    std::atomic<uint64_t> generation{0};
    static void OMNI_USD_LIVE_CALL callback(void* user,const OmniUsdLiveAuthRequired* event) {
        auto& self=*static_cast<SessionRefresh*>(user);
        assert(event->reason==OMNI_USD_LIVE_AUTH_MISSING && !event->grpc_status_present);
        uint64_t replacement=0;
        assert(omniUsdLiveSetAccessToken(event->endpoint,event->rejected_generation,"synthetic-writer",0,&replacement)==OMNI_USD_LIVE_OK);
        assert(replacement==event->rejected_generation+1);
        self.generation.store(replacement); ++self.calls;
    }
};
// Records a REJECTED challenge; the driver answers it after queuing a message.
struct ExpiryRefresh {
    std::mutex mutex; std::condition_variable cv; std::vector<OmniUsdLiveAuthRequired> events;
    static void OMNI_USD_LIVE_CALL callback(void* user,const OmniUsdLiveAuthRequired* event) {
        auto& self=*static_cast<ExpiryRefresh*>(user);
        std::lock_guard<std::mutex> lock(self.mutex); self.events.push_back(*event); self.cv.notify_all();
    }
    OmniUsdLiveAuthRequired wait() {
        std::unique_lock<std::mutex> lock(mutex);
        assert(cv.wait_for(lock,std::chrono::seconds(5),[&]{return !events.empty();}));
        return events.front();
    }
};
struct Sent { std::mutex mutex; std::condition_variable cv; unsigned calls=0; std::vector<OmniUsdLiveResult> results;
    static void OMNI_USD_LIVE_CALL callback(void* user,OmniUsdLiveResult result) {
        auto& s=*static_cast<Sent*>(user); std::lock_guard<std::mutex> lock(s.mutex); ++s.calls; s.results.push_back(result); s.cv.notify_all();
    }
    void wait(unsigned count) { std::unique_lock<std::mutex> lock(mutex); assert(cv.wait_for(lock,std::chrono::seconds(5),[&]{return calls==count;})); for(auto r:results) assert(r==OMNI_USD_LIVE_OK); }
};
// Aborts the driver if the guarded scope outlives its limit, so a deadlock
// fails the case instead of hanging the lane. Success never waits on it.
class Watchdog {
public:
    Watchdog(const char* label,std::chrono::seconds limit) : thread_([this,label,limit] {
        std::unique_lock<std::mutex> lock(mutex_);
        if(!cv_.wait_for(lock,limit,[&]{return done_;})) { std::cout<<"watchdog-expired: "<<label<<std::endl; liveTestDumpStacks(); std::abort(); }
    }) {}
    ~Watchdog() { { std::lock_guard<std::mutex> lock(mutex_); done_=true; } cv_.notify_all(); thread_.join(); }
private:
    std::mutex mutex_; std::condition_variable cv_; bool done_=false; std::thread thread_;
};
// Runs in Session::send between preparation and commit on the sending thread.
struct Prepared {
    OmniUsdLiveSession session=0; bool leave=false; unsigned calls=0; OmniUsdLiveRequest request=0;
    OmniUsdLiveResult status=OMNI_USD_LIVE_INTERNAL_ERROR, left=OMNI_USD_LIVE_INTERNAL_ERROR;
    static void hook(void* user,uint64_t request) noexcept {
        auto& self=*static_cast<Prepared*>(user); ++self.calls; self.request=request;
        Watchdog watchdog(self.leave?"leave during send preparation":"session status during send preparation",std::chrono::seconds(10));
        if(self.leave) {
            const auto session=self.session;
            auto left=std::async(std::launch::async,[session]{return omniUsdLiveLeaveSession(session,3000);});
            self.left=left.get();
        } else {
            auto status=sessionRecord<OmniUsdLiveStatus>();
            self.status=omniUsdLiveGetSessionStatus(self.session,&status);
        }
    }
};
std::string hex(const std::string& bytes) { std::ostringstream out; out<<std::hex<<std::setfill('0'); for(unsigned char c:bytes) out<<std::setw(2)<<unsigned(c); return out.str(); }
void printEvents(const Events& events) {
    std::cout<<'['; bool first=true; for(const auto& e:events.values) {
        if(!first) std::cout<<',';
        first=false;
        // Fixtures use ASCII identifiers; bytes are always hex encoded.
        std::cout<<"{\"kind\":"<<e.kind<<",\"result\":"<<e.result<<",\"id\":"<<std::quoted(e.id)
                 <<",\"user\":"<<std::quoted(e.user)<<",\"app\":"<<std::quoted(e.app)<<",\"hex\":"<<std::quoted(hex(e.bytes))<<'}';
    } std::cout<<']';
}
}
int sessionDriver(int argc,char** argv) {
    assert(argc==7); const std::string mode=argv[2]; const auto expected=static_cast<OmniUsdLiveResult>(std::stoul(argv[3]));
    auto options=sessionRecord<OmniUsdLiveRuntimeOptions>();
    options.diagnostic_min_severity=OMNI_USD_LIVE_SEVERITY_DEBUG;
    if(mode=="join-discard-no-history") options.flags=OMNI_USD_LIVE_RUNTIME_DISABLE_DIAGNOSTIC_HISTORY;
    options.retry_initial_ms=20; options.retry_maximum_ms=50; options.retry_max_attempts=3;
    // Joins follow the reconnect options, whose attempt bound differs from retry_max_attempts.
    options.reconnect_initial_ms=20; options.reconnect_maximum_ms=50;
    options.reconnect_max_attempts=4; options.peer_reconciliation_grace_ms=30000; options.default_auth_wait_timeout_ms=30000;
    options.initial_open_timeout_ms=2000; options.max_pending_messages_per_session=mode=="join-capacity"?2:16;
    // The session records credential starvation at construction, so a deadline that passes before its first step still ends UNAUTHENTICATED.
    if(mode=="join-auth-deadline") options.initial_open_timeout_ms=1;
    if(mode=="join-auth-wait-close") options.initial_open_timeout_ms=options.default_auth_wait_timeout_ms=30000;
    assert(omniUsdLiveInitialize(&options)==OMNI_USD_LIVE_OK);
    Prepared prepared; prepared.leave=mode=="send-leave-during-prepare";
    if(mode=="send-status-under-prepare" || prepared.leave)
        omni::usd::live::getRuntime()->sessions->setAfterPrepareForTest(Prepared::hook,&prepared);
    const TestTls tls;
    auto ep=sessionRecord<OmniUsdLiveEndpointOptions>(); ep.authority=argv[1]; ep.access_token=argv[5]; tls.apply(ep);
    OmniUsdLiveEndpoint endpoint=0; assert(omniUsdLiveRegisterEndpoint(&ep,&endpoint)==OMNI_USD_LIVE_OK);
    const auto uri=makeSessionUri(argv[1],argv[6]);
    Events a,b; a.hold=mode=="join-held" || mode=="join-self" || mode=="join-discard" || mode=="join-discard-no-history"; a.selfLeave=mode=="join-self";
    a.revokeOnError=mode=="join-revoke-after-unavailable"; a.revokeEndpoint=endpoint;
    if(mode=="join-callback-throw") a.hold=a.throwAfterHold=true;
    auto join=sessionRecord<OmniUsdLiveJoinOptions>(); join.app_name="native-alpha";
    OmniUsdLiveSession first=0; assert(omniUsdLiveJoinSession(uri.c_str(),&join,&a,Events::callback,&first)==OMNI_USD_LIVE_OK && first);
    { std::lock_guard<std::mutex> lock(a.mutex); a.session=first; }
    prepared.session=first;
    if(mode=="join-auth-timeout") {
        auto runtime=omni::usd::live::getRuntime();
        assert(runtime->diagnostics->waitForOperationAuthWaitsForTest(omni::usd::live::AuthWaitKind::Session,1,
            std::chrono::steady_clock::now()+std::chrono::seconds(3)));
        {
            auto owned=runtime->findEndpoint(endpoint);
            std::lock_guard<std::mutex> lock(owned->mutex);
            runtime->auth->evaluate(*owned,omni::usd::live::wallTimeMs(),owned->authDeadline);
        }
        runtime->auth->wake();
    }
    if(mode=="join-auth-wait-close") {
        auto runtime=omni::usd::live::getRuntime();
        assert(runtime->diagnostics->waitForOperationAuthWaitsForTest(omni::usd::live::AuthWaitKind::Session,1,
            std::chrono::steady_clock::now()+std::chrono::seconds(3)));
        assert(omniUsdLiveLeaveSession(first,3000)==OMNI_USD_LIVE_OK);
        assert(!runtime->diagnostics->snapshot(0,0).metrics.operationAuthWaits[2].current);
        assert(omniUsdLiveUnregisterEndpoint(endpoint,3000)==OMNI_USD_LIVE_OK);
        const auto diagnostics=publicDiagnostics(); runtime.reset();
        assert(omniUsdLiveShutdown(3000)==OMNI_USD_LIVE_OK);
        std::cout<<"{\"diagnostics\":"<<diagnostics<<"}\n";
        return 0;
    }
    SessionRefresh refresh; OmniUsdLiveCallbackHandle authHandle=0;
    if(mode=="join-auth-refresh") {
        assert(omni::usd::live::getRuntime()->diagnostics->waitForOperationAuthWaitsForTest(
            omni::usd::live::AuthWaitKind::Session,1,std::chrono::steady_clock::now()+std::chrono::seconds(2)));
        assert(omniUsdLiveRegisterAuthRequiredCallback(endpoint,&refresh,SessionRefresh::callback,&authHandle)==OMNI_USD_LIVE_OK);
    }
    Sent sent; unsigned sentCount=0; std::vector<OmniUsdLiveRequest> requests, receipts;
    auto send=[&](std::string payload,bool releaseImmediately=false) {
        OmniUsdLiveRequest request=0;
        assert(omniUsdLiveSendMessage(first,payload.data(),payload.size(),releaseImmediately?nullptr:&sent,releaseImmediately?ignoredReceipt:Sent::callback,&request)==OMNI_USD_LIVE_OK && request);
        receipts.push_back(request);
        payload.assign(payload.size(),'X');
        if(releaseImmediately) {
            const auto released=omniUsdLiveReleaseRequest(request);
            assert(released==OMNI_USD_LIVE_OK || released==OMNI_USD_LIVE_BUSY);
            if(released==OMNI_USD_LIVE_BUSY) { assert(omniUsdLiveCancel(request,3000)==OMNI_USD_LIVE_OK); assert(omniUsdLiveReleaseRequest(request)==OMNI_USD_LIVE_OK); }
        } else { ++sentCount; requests.push_back(request); }
    };
    OmniUsdLiveSession second=0; OmniUsdLiveResult result=OMNI_USD_LIVE_OK; unsigned finalState=0; bool left=false;
    auto finalStatus=sessionRecord<OmniUsdLiveStatus>();
    if(mode=="join-callback-throw") {
        a.wait([&](const auto&){return a.entered;});
        send(std::string("discard\0bytes",13)); sent.wait(sentCount);
        { std::lock_guard<std::mutex> lock(a.mutex); a.release=true; a.cv.notify_all(); }
        const auto watchdog=std::chrono::steady_clock::now()+std::chrono::seconds(3);
        for(;;) {
            auto status=sessionRecord<OmniUsdLiveStatus>();
            assert(omniUsdLiveGetSessionStatus(first,&status)==OMNI_USD_LIVE_OK);
            if(status.state==OMNI_USD_LIVE_STATE_FAILED) {
                result=status.result; finalState=status.state;
                assert(result==OMNI_USD_LIVE_INTERNAL_ERROR && !status.grpc_status_present); break;
            }
            assert(std::chrono::steady_clock::now()<watchdog); std::this_thread::yield();
        }
    } else if(mode=="join-held" || mode=="join-self" || mode=="join-discard" || mode=="join-discard-no-history") {
        a.wait([&](const auto&){return a.entered;});
        if(mode=="join-discard" || mode=="join-discard-no-history") {
            // Holding the Connected callback prevents this session worker from
            // issuing writes. Receipts still complete on independent workers.
            send("",true); send(std::string("discard\0bytes",13)); sent.wait(sentCount);
            assert(omniUsdLiveLeaveSession(first,0)==OMNI_USD_LIVE_TIMEOUT);
        }
        if(mode=="join-held") assert(omniUsdLiveLeaveSession(first,0)==OMNI_USD_LIVE_TIMEOUT);
        { std::lock_guard<std::mutex> lock(a.mutex); a.release=true; a.cv.notify_all(); }
        if(mode=="join-self") a.wait([&](const auto&){return a.selfResult==OMNI_USD_LIVE_BUSY;});
    } else if(mode=="join-fail-queued") {
        send("",true); send(std::string("discard\0bytes",13)); sent.wait(sentCount);
        std::cout<<"session-queued\n"<<std::flush;
        a.wait([&](const auto& v){for(const auto& e:v) if(e.kind==OMNI_USD_LIVE_EVENT_CONNECTION && e.result==expected) return true; return false;});
        result=expected;
    } else if(mode=="join-revoke-after-unavailable") {
        a.wait([&](const auto&){return a.revoked;});
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(2);
        for(;;) {
            auto status=sessionRecord<OmniUsdLiveStatus>();
            assert(omniUsdLiveGetSessionStatus(first,&status)==OMNI_USD_LIVE_OK);
            if(status.state==OMNI_USD_LIVE_STATE_AUTH_WAIT) {
                assert(!status.grpc_status_present);
                result=status.result; finalState=status.state; break;
            }
            assert(std::chrono::steady_clock::now()<deadline); std::this_thread::yield();
        }
        assert(result==expected);
    } else if(mode=="join-auth-refresh") {
        a.wait([&](const auto& v){for(const auto& e:v) if(e.kind==OMNI_USD_LIVE_EVENT_CONNECTION && e.result==OMNI_USD_LIVE_OK) return true; return false;});
        assert(omniUsdLiveUnregisterCallback(authHandle,3000)==OMNI_USD_LIVE_OK); authHandle=0;
        assert(refresh.calls==1 && refresh.generation==1);
        auto status=sessionRecord<OmniUsdLiveStatus>();
        assert(omniUsdLiveGetSessionStatus(first,&status)==OMNI_USD_LIVE_OK && status.state==OMNI_USD_LIVE_STATE_READY);
    } else if(mode=="join-auth-server-expiry") {
        a.wait([&](const auto& v){for(const auto& e:v) if(e.kind==OMNI_USD_LIVE_EVENT_CONNECTION && e.result==OMNI_USD_LIVE_OK) return true; return false;});
        ExpiryRefresh expiry;
        assert(omniUsdLiveRegisterAuthRequiredCallback(endpoint,&expiry,ExpiryRefresh::callback,&authHandle)==OMNI_USD_LIVE_OK);
        std::cout<<"session-expiry-ready\n"<<std::flush; // The fixture expires the token and resets the stream.
        const auto challenge=expiry.wait();
        assert(challenge.reason==OMNI_USD_LIVE_AUTH_REJECTED && challenge.rejected_generation==1 &&
               challenge.grpc_status_present && challenge.grpc_status==16);
        send("queued-during-auth"); sent.wait(sentCount); // A receipt is local acceptance.
        uint64_t replacement=0;
        assert(omniUsdLiveSetAccessToken(endpoint,1,"session-t2",0,&replacement)==OMNI_USD_LIVE_OK && replacement==2);
        a.wait([&](const auto& v){
            std::vector<std::string> ids;
            for(const auto& e:v) if(e.kind==OMNI_USD_LIVE_EVENT_CONNECTION && e.result==OMNI_USD_LIVE_OK) ids.push_back(e.id);
            return ids.size()==2 && ids[0]!=ids[1];
        });
        // Leave discards unsent messages and cancels the stream, so the fixture
        // answers "continue" only after the service records the queued frame.
        std::string command; assert(std::getline(std::cin,command) && command=="continue");
        // The callback state lives in this scope, so its registration ends here.
        assert(omniUsdLiveUnregisterCallback(authHandle,3000)==OMNI_USD_LIVE_OK); authHandle=0;
    } else if(mode=="join-happy") {
        a.wait([&](const auto&){return a.count(OMNI_USD_LIVE_EVENT_CONNECTION)>=1;});
        join.app_name="native-beta";
        assert(omniUsdLiveJoinSession(uri.c_str(),&join,&b,Events::callback,&second)==OMNI_USD_LIVE_OK && second!=first);
        a.wait([&](const auto&){return a.count(OMNI_USD_LIVE_EVENT_JOINED)==1;});
        b.wait([&](const auto&){return b.count(OMNI_USD_LIVE_EVENT_JOINED)==1;});
        send(std::string("A\0\xff",3),true); send("second"); send("third"); sent.wait(sentCount);
        b.wait([&](const auto&){return b.count(OMNI_USD_LIVE_EVENT_MESSAGE)==3;});
        // The service decodes at most 4 MiB per stream message, so a larger session message is refused locally
        // rather than lost on the wire.
        const std::string oversized(omni::usd::live::kMaxSessionMessageBytes+1,'x');
        OmniUsdLiveRequest refusedLarge=0;
        assert(omniUsdLiveSendMessage(first,oversized.data(),oversized.size(),nullptr,ignoredReceipt,&refusedLarge)==
               OMNI_USD_LIVE_RESOURCE_EXHAUSTED && !refusedLarge);
    } else if((mode=="join-reconnect" || mode=="join-reconnect-eof" || mode=="join-reconnect-reset")) {
        a.wait([&](const auto& v){for(const auto& e:v) if(e.bytes=="rotate") return true; return false;});
        std::cout<<"session-rotate-ready\n"<<std::flush;
        a.wait([&](const auto& v){for(const auto& e:v) if(e.kind==OMNI_USD_LIVE_EVENT_CONNECTION && e.result!=OMNI_USD_LIVE_OK) return true; return false;});
        send(std::string("queued\0one",10),true); send("queued-two"); sent.wait(sentCount);
        std::cout<<"session-queued\n"<<std::flush;
        a.wait([&](const auto& v){for(const auto& e:v) if(e.bytes=="resumed") return true; return false;});
        // The service re-announces every surviving peer before "resumed", so an expired grace departs only the absent peer.
        omni::usd::live::getRuntime()->sessions->expirePeersForTest(first);
        a.wait([&](const auto& v){bool left=mode=="join-reconnect-reset"; for(const auto& e:v) left|=e.kind==OMNI_USD_LIVE_EVENT_LEFT && e.id=="peer-gone"; return left;});
    } else if(mode=="join-keepalive-queued" || mode=="join-keepalive-older") {
        a.wait([&](const auto& v){for(const auto& e:v) if(e.kind==OMNI_USD_LIVE_EVENT_CONNECTION && e.result==OMNI_USD_LIVE_OK) return true; return false;});
        // Accepted messages are queued or in flight when the interval is expired, so the keepalive goes after them.
        if(mode=="join-keepalive-queued") { send("first"); send("second"); sent.wait(sentCount); }
        omni::usd::live::getRuntime()->sessions->expireKeepaliveForTest(first);
        // A service that returned no interval gets no keepalive, only this message.
        if(mode=="join-keepalive-older") { send("after-expiry"); sent.wait(sentCount); }
        std::cout<<"session-keepalive-expired\n"<<std::flush;
        // Leave discards unsent frames and cancels the stream, so the fixture
        // answers "continue" only after the service records the expected frame.
        std::string command; assert(std::getline(std::cin,command) && command=="continue");
        auto status=sessionRecord<OmniUsdLiveStatus>();
        assert(omniUsdLiveGetSessionStatus(first,&status)==OMNI_USD_LIVE_OK && status.state==OMNI_USD_LIVE_STATE_READY);
    } else if(mode=="join-keepalive-host-blocked") {
        a.wait([&](const auto& v){for(const auto& e:v) if(e.kind==OMNI_USD_LIVE_EVENT_CONNECTION && e.result==OMNI_USD_LIVE_OK) return true; return false;});
        std::cout<<"session-keepalive-ready\n"<<std::flush;
        // The host thread blocks here and calls nothing; the session's own
        // worker sends the keepalives the fixture waits for.
        std::string command; assert(std::getline(std::cin,command) && command=="continue");
        auto status=sessionRecord<OmniUsdLiveStatus>();
        assert(omniUsdLiveGetSessionStatus(first,&status)==OMNI_USD_LIVE_OK && status.state==OMNI_USD_LIVE_STATE_READY);
    } else if(mode=="join-keepalive-timeout") {
        a.wait([&](const auto& v){
            bool connected=false, joined=false;
            for(const auto& e:v) { connected|=e.kind==OMNI_USD_LIVE_EVENT_CONNECTION && e.result==OMNI_USD_LIVE_OK; joined|=e.kind==OMNI_USD_LIVE_EVENT_JOINED; }
            return connected && joined;
        });
        std::cout<<"session-keepalive-timeout-ready\n"<<std::flush; // The fixture lapses this join's keepalive.
        a.wait([&](const auto& v){
            std::vector<std::string> ids;
            for(const auto& e:v) if(e.kind==OMNI_USD_LIVE_EVENT_CONNECTION && e.result==OMNI_USD_LIVE_OK) ids.push_back(e.id);
            return ids.size()==2 && ids[0]!=ids[1];
        });
    } else if(mode=="send-status-under-prepare") {
        a.wait([&](const auto& v){for(const auto& e:v) if(e.kind==OMNI_USD_LIVE_EVENT_CONNECTION && e.result==OMNI_USD_LIVE_OK) return true; return false;});
        send("prepared"); sent.wait(sentCount);
        assert(prepared.calls==1 && prepared.status==OMNI_USD_LIVE_OK && prepared.request==requests.back());
    } else if(mode=="send-leave-during-prepare") {
        a.wait([&](const auto& v){for(const auto& e:v) if(e.kind==OMNI_USD_LIVE_EVENT_CONNECTION && e.result==OMNI_USD_LIVE_OK) return true; return false;});
        assert(omniUsdLiveGetSessionStatus(first,&finalStatus)==OMNI_USD_LIVE_OK);
        OmniUsdLiveRequest refused=99; const std::string payload="cancelled";
        assert(omniUsdLiveSendMessage(first,payload.data(),payload.size(),&sent,Sent::callback,&refused)==OMNI_USD_LIVE_CANCELLED && !refused);
        assert(prepared.calls==1 && prepared.left==OMNI_USD_LIVE_OK && prepared.request);
        // The prepared request was released without reaching the caller.
        OmniUsdLiveResult receipt=OMNI_USD_LIVE_OK;
        assert(omniUsdLiveGetRequestStatus(prepared.request,&receipt)==OMNI_USD_LIVE_INVALID_HANDLE);
        left=true;
    } else if(mode=="join-capacity") {
        // Service withholds Connected until admission completes, so queued
        // messages cannot leave the bounded local pending queue.
        send("one"); send("two"); sent.wait(sentCount);
        OmniUsdLiveRequest rejected=99; const std::string third="three";
        assert(omniUsdLiveSendMessage(first,third.data(),third.size(),nullptr,ignoredReceipt,&rejected)==OMNI_USD_LIVE_RESOURCE_EXHAUSTED && !rejected);
        std::cout<<"session-queued\n"<<std::flush;
        a.wait([&](const auto& v){for(const auto& e:v) if(e.bytes=="received-two") return true; return false;});
    } else {
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(6);
        for(;;) { auto status=sessionRecord<OmniUsdLiveStatus>(); assert(omniUsdLiveGetSessionStatus(first,&status)==OMNI_USD_LIVE_OK);
            if(status.state==OMNI_USD_LIVE_STATE_FAILED) { result=status.result; finalState=status.state; break; }
            assert(std::chrono::steady_clock::now()<deadline); std::this_thread::yield(); }
        assert(result==expected);
        // A terminal status alone does not prove that the host receives its
        // failure notification. Wait before Leave can suppress queued events.
        a.wait([&](const auto& v){
            for(const auto& e:v) if(e.kind==OMNI_USD_LIVE_EVENT_CONNECTION && e.result==expected) return true;
            return false;
        });
    }
    for(auto request:requests) assert(omniUsdLiveWait(request,3000)==OMNI_USD_LIVE_OK);
    if(!left) {
        assert(omniUsdLiveGetSessionStatus(first,&finalStatus)==OMNI_USD_LIVE_OK);
        assert(omniUsdLiveLeaveSession(first,3000)==OMNI_USD_LIVE_OK);
    }
    auto invalid=sessionRecord<OmniUsdLiveStatus>(); assert(omniUsdLiveGetSessionStatus(first,&invalid)==OMNI_USD_LIVE_INVALID_HANDLE);
    auto receiverStatus=finalStatus;
    if(second) {
        assert(omniUsdLiveGetSessionStatus(second,&receiverStatus)==OMNI_USD_LIVE_OK);
        assert(omniUsdLiveLeaveSession(second,3000)==OMNI_USD_LIVE_OK);
    }
    // Receipts remain local acceptance, including after terminal/Leave discard.
    for(auto request:requests) {
        OmniUsdLiveResult receipt=OMNI_USD_LIVE_INTERNAL_ERROR;
        assert(omniUsdLiveGetRequestStatus(request,&receipt)==OMNI_USD_LIVE_OK && receipt==OMNI_USD_LIVE_OK);
        auto detail=sessionRecord<OmniUsdLiveRequestError>();
        assert(omniUsdLiveGetRequestError(request,&detail)==OMNI_USD_LIVE_OK && detail.result==OMNI_USD_LIVE_OK &&
               !detail.grpc_status_present && detail.attempt_id==0);
        assert(omniUsdLiveReleaseRequest(request)==OMNI_USD_LIVE_OK);
    }
    if(authHandle) assert(omniUsdLiveUnregisterCallback(authHandle,3000)==OMNI_USD_LIVE_OK);
    auto exportOptions=sessionRecord<OmniUsdLiveDiagnosticExportOptions>();
    exportOptions.max_records=128; exportOptions.max_output_bytes=32768;
    std::vector<char> diagnostics(exportOptions.max_output_bytes); uint64_t diagnosticSize=diagnostics.size();
    assert(omniUsdLiveCopyDiagnostics(endpoint,&exportOptions,diagnostics.data(),&diagnosticSize)==OMNI_USD_LIVE_OK &&
           diagnosticSize && diagnosticSize<=diagnostics.size());
    assert(omniUsdLiveUnregisterEndpoint(endpoint,3000)==OMNI_USD_LIVE_OK); assert(omniUsdLiveShutdown(3000)==OMNI_USD_LIVE_OK);
    std::cout<<"{\"send_requests\":[";
    for(size_t i=0;i<receipts.size();++i) { if(i) std::cout<<','; std::cout<<receipts[i]; }
    std::cout<<"],\"receiver_handle\":"<<(second?second:first)<<",\"receiver_subject\":"<<receiverStatus.subject_id;
    std::cout<<",\"result\":"<<result<<",\"grpc_status_present\":"<<(finalStatus.grpc_status_present?"true":"false")<<",\"grpc_status\":"<<finalStatus.grpc_status<<",\"state\":"<<finalState<<",\"send_callbacks\":"<<sent.calls<<",\"auth_callbacks\":"<<refresh.calls.load()<<",\"session_handle\":"<<first<<",\"subject\":"<<finalStatus.subject_id<<",\"diagnostics\":"<<diagnostics.data()<<",\"a\":"; printEvents(a); std::cout<<",\"b\":"; printEvents(b); std::cout<<"}\n";
    return 0;
}
