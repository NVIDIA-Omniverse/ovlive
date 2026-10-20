// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "LiveSdfFileFormat.h"
#include "BoundLayer.h"
#include "FieldEditIntent.h"
#include "HostSceneProjection.h"
#include "host_admission.h"
#include "LiveLayerActor.h"
#include "LiveRuntime.h"
#include "LiveUri.h"
#include "UsdCreationRegistry.h"
#include <pxr/base/plug/registry.h>
#include <pxr/base/tf/diagnosticMgr.h>
#include <pxr/base/tf/errorMark.h>
#include <pxr/usd/pcp/errors.h>
#include <pxr/usd/sdf/attributeSpec.h>
#include <pxr/usd/sdf/changeBlock.h>
#include <pxr/usd/sdf/data.h>
#include <pxr/usd/sdf/layer.h>
#include <pxr/usd/sdf/notice.h>
#include <pxr/usd/sdf/payload.h>
#include <pxr/usd/sdf/primSpec.h>
#include <pxr/usd/sdf/schema.h>
#include <pxr/usd/sdf/types.h>
#include <pxr/usd/usd/notice.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/stage.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <future>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <memory>
#include <mutex>
#include <new>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>

// One-shot allocation failure on the calling thread. Only the host worker
// arms it, for one allocation of one plugin-dispatched job or one sweep call;
// other threads and every other job allocate normally.
namespace {
thread_local long allocationCountdown=-1; // One-shot: fail allocation N (0-based) on this thread.
thread_local bool allocationFailed=false;
thread_local bool reachedOpenWait=false;
// Registered while an open sweep runs: a synchronous open stops injection
// where it starts waiting for the bootstrap, before USD installs content.
void disarmAtOpenWait(void*) noexcept { allocationCountdown=-1; reachedOpenWait=true; }
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
// Out of line, so GCC does not pair an inlined free() with a delete
// expression and diagnose a spurious mismatched-new-delete.
#if defined(_MSC_VER)
#define LIVE_TEST_ALLOC_NOINLINE __declspec(noinline)
#else
#define LIVE_TEST_ALLOC_NOINLINE __attribute__((noinline))
#endif
LIVE_TEST_ALLOC_NOINLINE void* operator new(std::size_t bytes) {
    if(allocationCountdown>=0 && allocationCountdown--==0) { allocationFailed=true; throw std::bad_alloc(); }
    if(auto* value=std::malloc(bytes ? bytes : 1)) return value;
    throw std::bad_alloc();
}
LIVE_TEST_ALLOC_NOINLINE void* operator new[](std::size_t bytes) { return ::operator new(bytes); }
LIVE_TEST_ALLOC_NOINLINE void operator delete(void* value) noexcept { std::free(value); }
LIVE_TEST_ALLOC_NOINLINE void operator delete[](void* value) noexcept { std::free(value); }
LIVE_TEST_ALLOC_NOINLINE void operator delete(void* value,std::size_t) noexcept { std::free(value); }
LIVE_TEST_ALLOC_NOINLINE void operator delete[](void* value,std::size_t) noexcept { std::free(value); }
#undef LIVE_TEST_ALLOC_NOINLINE
#endif

PXR_NAMESPACE_USING_DIRECTIVE
using namespace omni::usd::live;
namespace {
using Clock=std::chrono::steady_clock;
using Fields=std::map<std::string,int>;
void check(bool condition,const char* message) { if (!condition) throw std::runtime_error(message); }
template<class T> T record() { T value{}; value.struct_size=sizeof(value); value.abi_version=OMNI_USD_LIVE_ABI_VERSION; return value; }
// Waits for a status event of `subject` carrying `result` and, when nonzero,
// `state`; registration delivers the current snapshot first.
class StatusBoundary {
    std::mutex mutex_; std::condition_variable changed_;
    uint64_t subject_; OmniUsdLiveResult result_; uint32_t state_; bool reached_=false;
    OmniUsdLiveCallbackHandle callback_=0;
    static void OMNI_USD_LIVE_CALL receive(void* user,const OmniUsdLiveStatusEvent* event) {
        auto& self=*static_cast<StatusBoundary*>(user);
        if (!event || !event->status || event->status->subject_id!=self.subject_ || event->status->result!=self.result_ ||
            (self.state_ && event->status->state!=self.state_)) return;
        std::lock_guard<std::mutex> lock(self.mutex_); self.reached_=true; self.changed_.notify_all();
    }
public:
    StatusBoundary(OmniUsdLiveEndpoint endpoint,uint64_t subject,OmniUsdLiveResult result,uint32_t state=0)
        :subject_(subject),result_(result),state_(state) {
        check(omniUsdLiveRegisterStatusCallback(endpoint,this,receive,&callback_)==OMNI_USD_LIVE_OK,"status boundary registration failed");
    }
    ~StatusBoundary() {
        if (omniUsdLiveUnregisterCallback(callback_,5000)!=OMNI_USD_LIVE_OK) std::_Exit(2);
    }
    void wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        check(changed_.wait_for(lock,std::chrono::seconds(5),[&] { return reached_; }),"status boundary did not arrive");
    }
};
// Waits for a status event of `subject` carrying `expected` buffered edits;
// registration delivers the current snapshot first.
class BufferedBoundary {
    std::mutex mutex_; std::condition_variable changed_;
    uint64_t subject_, expected_; bool reached_=false;
    OmniUsdLiveCallbackHandle callback_=0;
    static void OMNI_USD_LIVE_CALL receive(void* user,const OmniUsdLiveStatusEvent* event) {
        auto& self=*static_cast<BufferedBoundary*>(user);
        if (!event || !event->status || event->status->subject_id!=self.subject_ ||
            event->status->buffered_edit_count!=self.expected_) return;
        std::lock_guard<std::mutex> lock(self.mutex_); self.reached_=true; self.changed_.notify_all();
    }
public:
    BufferedBoundary(OmniUsdLiveEndpoint endpoint,uint64_t subject,uint64_t expected):subject_(subject),expected_(expected) {
        check(omniUsdLiveRegisterStatusCallback(endpoint,this,receive,&callback_)==OMNI_USD_LIVE_OK,"buffered boundary registration failed");
    }
    ~BufferedBoundary() {
        if (omniUsdLiveUnregisterCallback(callback_,5000)!=OMNI_USD_LIVE_OK) std::_Exit(2);
    }
    void wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        check(changed_.wait_for(lock,std::chrono::seconds(5),[&] { return reached_; }),"buffered edit count did not arrive");
    }
};
// Captures the CLOSED status event of `subject`; register it before the
// release that closes the layer.
class ClosedBoundary {
    std::mutex mutex_; std::condition_variable changed_;
    uint64_t subject_; bool reached_=false; OmniUsdLiveStatus closed_{};
    OmniUsdLiveCallbackHandle callback_=0;
    static void OMNI_USD_LIVE_CALL receive(void* user,const OmniUsdLiveStatusEvent* event) {
        auto& self=*static_cast<ClosedBoundary*>(user);
        if (!event || !event->status || event->status->subject_id!=self.subject_ ||
            event->status->state!=OMNI_USD_LIVE_STATE_CLOSED) return;
        std::lock_guard<std::mutex> lock(self.mutex_); self.closed_=*event->status; self.reached_=true; self.changed_.notify_all();
    }
public:
    ClosedBoundary(OmniUsdLiveEndpoint endpoint,uint64_t subject):subject_(subject) {
        check(omniUsdLiveRegisterStatusCallback(endpoint,this,receive,&callback_)==OMNI_USD_LIVE_OK,"closed boundary registration failed");
    }
    ~ClosedBoundary() {
        if (omniUsdLiveUnregisterCallback(callback_,5000)!=OMNI_USD_LIVE_OK) std::_Exit(2);
    }
    OmniUsdLiveStatus wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        check(changed_.wait_for(lock,std::chrono::seconds(5),[&] { return reached_; }),"closed status did not arrive");
        return closed_;
    }
};
bool identifier(const std::string& text) {
    if (text.empty()) return false;
    for (const char c:text) if (!(c=='_' || (c>='a' && c<='z') || (c>='A' && c<='Z') || (c>='0' && c<='9'))) return false;
    return true;
}
void expectedError(TfErrorMark& mark,OmniUsdLiveResult expected) {
    bool found=false;
    for (const auto& error:mark) {
        if (const auto* result=error.GetInfo<OmniUsdLiveResult>()) {
            check(*result==expected,"wrong typed native format rejection"); found=true;
            check(error.GetCommentary().find("synthetic-writer")==std::string::npos,"credential in native format diagnostic");
        }
    }
    mark.Clear(); check(found,"native format omitted typed rejection");
}
// A root value whose copy runs a test hook on the copying thread. VtValue
// stores it locally, so every VtValue copy, including the flush's copies after
// it takes the host edit buffer, invokes the hook. The wire codec rejects it.
std::function<void()> flushProbeHook;
struct FlushProbe {
    int id=0;
    FlushProbe()=default;
    FlushProbe(const FlushProbe& other) noexcept : id(other.id) { if(flushProbeHook) flushProbeHook(); }
    FlushProbe& operator=(const FlushProbe&)=default;
    bool operator==(const FlushProbe& other) const { return id==other.id; }
};
Fields fields(const SdfLayer& layer) {
    Fields result;
    for (const auto& key:layer.ListFields(SdfPath::AbsoluteRootPath())) {
        const auto value=layer.GetField(SdfPath::AbsoluteRootPath(),key);
        // Host-only unencodable probes; never reported.
        if (value.IsHolding<FlushProbe>() || value.IsHolding<SdfPayload>()) continue;
        if (key==SdfChildrenKeys->PrimChildren) continue; // Structure of a temporary root prim.
        check(identifier(key.GetString()) && value.IsHolding<int>(),"native format fixture expects identifier/int root fields");
        result.emplace(key.GetString(),value.UncheckedGet<int>());
    }
    return result;
}
// Collects the commentary of every Tf error posted while it exists and no
// TfErrorMark is active on the posting thread, such as the runtime error a
// failing flush posts on the host from the flush path, which runs in a flush
// job and also inside a save or a publication.
class ErrorCapture final : public TfDiagnosticMgr::Delegate {
    std::mutex mutex_; std::vector<std::string> errors_;
public:
    ErrorCapture() { TfDiagnosticMgr::GetInstance().AddDelegate(this); }
    ~ErrorCapture() override { TfDiagnosticMgr::GetInstance().RemoveDelegate(this); }
    void IssueError(const TfError& error) override { std::lock_guard<std::mutex> lock(mutex_); errors_.push_back(error.GetCommentary()); }
    void IssueFatalError(const TfCallContext&,const std::string&) override {}
    void IssueStatus(const TfStatus&) override {}
    void IssueWarning(const TfWarning&) override {}
    std::vector<std::string> errors() { std::lock_guard<std::mutex> lock(mutex_); return errors_; }
};
std::string jsonString(const std::string& text) {
    std::string result="\"";
    for (const unsigned char c:text) {
        if (c=='"' || c=='\\') { result+='\\'; result+=char(c); }
        else if (c<0x20) { char escaped[8]; std::snprintf(escaped,sizeof(escaped),"\\u%04x",c); result+=escaped; }
        else result+=char(c);
    }
    return result+'"';
}
void jsonFields(const Fields& values) {
    std::cout << '{'; bool first=true;
    for (const auto& entry:values) { if (!first) std::cout << ','; first=false; std::cout << '"' << entry.first << "\":" << entry.second; }
    std::cout << '}';
}
Fields fields(const WireScene& scene) {
    Fields result;
    for(const auto& key:scene.data()->List(SdfPath::AbsoluteRootPath())) {
        const auto value=scene.data()->Get(SdfPath::AbsoluteRootPath(),key);
        check(identifier(key.GetString()) && value.IsHolding<int>(),"journal scene expects identifier/int fields");
        result.emplace(key.GetString(),value.UncheckedGet<int>());
    }
    return result;
}
// A layer's backing data, read through the file format's protected accessor.
class FormatBacking : public SdfFileFormat {
public:
    static const SdfAbstractData* data(const SdfLayer& layer) { return _GetLayerData(layer).operator->(); }
};
// Actual USD layer reads, opening, reload and publication share this queue.
// Holding one accepted publication never blocks control jobs on its worker.
// Signaled by the production progress observer seam on every attachment
// progress change and after every host job. `until` re-evaluates `ready`
// after each signal; its deadline is a failure-only watchdog.
class ProgressEvents {
    std::mutex mutex_; std::condition_variable changed_; uint64_t revision_=0;
public:
    void signal() { std::lock_guard<std::mutex> lock(mutex_); ++revision_; changed_.notify_all(); }
    template<class Ready> void until(Ready ready,const char* message) {
        const auto deadline=Clock::now()+std::chrono::seconds(5);
        for (;;) {
            uint64_t seen=0;
            { std::lock_guard<std::mutex> lock(mutex_); seen=revision_; }
            if (ready()) return;
            std::unique_lock<std::mutex> lock(mutex_);
            check(changed_.wait_until(lock,deadline,[&] { return revision_!=seen; }),message);
        }
    }
};
ProgressEvents& progressEvents() { static ProgressEvents events; return events; }
class Host final {
    std::mutex mutex_;
    std::condition_variable changed_;
    std::deque<std::function<void()>> jobs_;
    std::function<void()> held_;
    bool stop_=false, holdNext_=false, rejectNext_=false;
    bool rejectContext_=false; // Access only on worker; forces one production callback rejection.
    bool failAllocationNext_=false; // The next dispatched job fails its first allocation once.
    bool refuseDispatch_=false; // Under mutex_: every dispatch is refused for capacity.
    bool allocationFaulted_=false; // Host thread only: whether that job's allocation failed.
    std::thread worker_;
    SdfLayerRefPtr layer_; // Accessed/released only on the host thread.
    SdfLayerRefPtr second_; // Another live layer on the same endpoint; host thread only.
    // Counts the layer's whole-content replacements, one per first
    // installation, and the change notices it sends.
    struct Replacements final : TfWeakBase {
        std::atomic<unsigned> count{0}, changes{0}; TfNotice::Key key, changeKey;
        void replaced(const SdfNotice::LayerDidReplaceContent&, const SdfLayerHandle&) { ++count; }
        void changed(const SdfNotice::LayersDidChangeSentPerLayer&, const SdfLayerHandle&) { ++changes; }
    };
    Replacements replacements_; // Host thread only.
    // A stage whose anonymous root layer sublayers the live layer, and the
    // paths its ObjectsChanged notices named since they were last taken.
    struct StageNotices final : TfWeakBase {
        std::set<SdfPath> resynced, info; unsigned notices=0; TfNotice::Key key;
        void changed(const UsdNotice::ObjectsChanged& notice, const UsdStageWeakPtr&) {
            ++notices;
            for(const auto& path:notice.GetResyncedPaths()) resynced.insert(path);
            for(const auto& path:notice.GetChangedInfoOnlyPaths()) info.insert(path);
        }
    };
    UsdStageRefPtr stage_; // Host thread only.
    StageNotices stageNotices_; // Host thread only.
    // Once armed, authors one root field from inside the next change notice
    // the layer sends, such as a publication's installation notice.
    struct NoticeAuthor final : TfWeakBase {
        SdfLayerHandle layer; TfToken key; int value=0; bool armed=false; TfNotice::Key registration;
        void changed(const SdfNotice::LayersDidChangeSentPerLayer&, const SdfLayerHandle&) {
            if(!armed) return;
            armed=false; layer->SetField(SdfPath::AbsoluteRootPath(),key,VtValue(value));
        }
    };
    NoticeAuthor noticeAuthor_; // Host thread only.
    SdfLayerRefPtr created_; // The last CreateNew target or Export destination; host thread only.
    // While set, every job dispatched from the host thread crosses the fake
    // creation deadline and is counted. Under mutex_.
    std::shared_ptr<std::atomic<bool>> crossOnHostDispatch_;
    unsigned hostDispatches_=0; // Under mutex_.
public:
    Host():worker_([this] {
        for (;;) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                changed_.wait(lock,[&] { return stop_ || !jobs_.empty(); });
                if (stop_ && jobs_.empty()) break;
                job=std::move(jobs_.front()); jobs_.pop_front();
            }
            job(); job={}; progressEvents().signal();
        }
        TfNotice::Revoke(stageNotices_.key); stage_.Reset();
        created_.Reset(); second_.Reset(); layer_.Reset();
    }) {}
    ~Host() {
        releaseHeld();
        { std::lock_guard<std::mutex> lock(mutex_); stop_=true; changed_.notify_all(); }
        worker_.join();
    }
    bool current() const { return std::this_thread::get_id()==worker_.get_id() && !rejectContext_; }
    void enqueue(std::function<void()> job) {
        std::lock_guard<std::mutex> lock(mutex_);
        check(!stop_ && jobs_.size()<16,"native format host queue exhausted");
        jobs_.push_back(std::move(job)); changed_.notify_all();
    }
    void dispatch(std::function<void()> job) {
        std::lock_guard<std::mutex> lock(mutex_);
        check(!stop_ && jobs_.size()<16,"native format publication queue exhausted");
        if(refuseDispatch_) throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
        if(crossOnHostDispatch_ && std::this_thread::get_id()==worker_.get_id()) {
            crossOnHostDispatch_->store(true); ++hostDispatches_;
        }
        if(rejectNext_) {
            rejectNext_=false;
            job=[this,original=std::move(job)]() mutable {
                rejectContext_=true;
                try { original(); } catch(...) { rejectContext_=false; throw; }
                rejectContext_=false;
            };
        }
        if(failAllocationNext_) {
            failAllocationNext_=false;
            job=[this,original=std::move(job)]() mutable {
                allocationFailed=false; allocationCountdown=0;
                try { original(); } catch(...) { allocationCountdown=-1; throw; }
                allocationCountdown=-1; allocationFaulted_=std::exchange(allocationFailed,false);
            };
        }
        if (holdNext_) { check(!held_,"multiple withheld publications"); holdNext_=false; held_=std::move(job); }
        else jobs_.push_back(std::move(job));
        changed_.notify_all();
    }
    // `watchdog` is a failure-only bound on the host job.
    template<class F> auto call(F action,std::chrono::seconds watchdog=std::chrono::seconds(8))->decltype(action()) {
        using Result=decltype(action());
        auto task=std::make_shared<std::packaged_task<Result()>>(std::move(action));
        auto future=task->get_future(); enqueue([task] { (*task)(); });
        check(future.wait_for(watchdog)==std::future_status::ready,"native format host call timed out");
        return future.get();
    }
    void arm() { std::lock_guard<std::mutex> lock(mutex_); check(!held_ && !holdNext_,"publication already held"); holdNext_=true; }
    bool armed() { std::lock_guard<std::mutex> lock(mutex_); return holdNext_; }
    void refuseDispatch(bool refuse) { std::lock_guard<std::mutex> lock(mutex_); refuseDispatch_=refuse; }
    void rejectNext() { std::lock_guard<std::mutex> lock(mutex_); check(!rejectNext_,"publication rejection already armed"); rejectNext_=true; }
    void waitHeld() {
        std::unique_lock<std::mutex> lock(mutex_);
        check(changed_.wait_for(lock,std::chrono::seconds(5),[&] { return bool(held_); }),"native format publication was not queued");
    }
    void releaseHeld() {
        std::lock_guard<std::mutex> lock(mutex_);
        holdNext_=false;
        if (held_) { jobs_.push_back(std::move(held_)); held_={}; changed_.notify_all(); }
    }
    // Host thread only: run the withheld job before any job queued by the
    // current one, such as a flush scheduled by a direct host mutation.
    void releaseHeldFirst() {
        std::lock_guard<std::mutex> lock(mutex_);
        check(bool(held_),"no withheld publication to release first");
        jobs_.push_front(std::move(held_)); held_={}; changed_.notify_all();
    }
    // Host thread only: run `job` immediately after the next queued job.
    void enqueueSecond(std::function<void()> job) {
        std::lock_guard<std::mutex> lock(mutex_);
        check(!jobs_.empty() && jobs_.size()<16,"no queued job to follow");
        jobs_.insert(std::next(jobs_.begin()),std::move(job)); changed_.notify_all();
    }
    void rejectedOpens(const std::string& uri) {
        call([&] {
            const auto rules=SdfLayer::GetDetachedLayerRules();
            SdfLayer::SetDetachedLayerRules(SdfLayer::DetachedLayerRules().IncludeAll());
            TfErrorMark detachedErrors;
            const auto detached=SdfLayer::FindOrOpen(uri);
            SdfLayer::SetDetachedLayerRules(rules);
            check(!detached,"production format silently opened a detached live layer");
            expectedError(detachedErrors,OMNI_USD_LIVE_UNSUPPORTED);
            TfErrorMark argumentErrors;
            const auto arguments=SdfLayer::FindOrOpen(uri,SdfLayer::FileFormatArguments{{"unsupported","value"}});
            check(!arguments,"unsupported file-format arguments acquired another live backing");
            expectedError(argumentErrors,OMNI_USD_LIVE_UNSUPPORTED);
        });
    }
    void rejectedReads(const std::string& uri,const std::shared_ptr<SceneAttachment>& attachment) {
        const auto retained=call([&] {
            TfErrorMark errors;
            check(!layer_->GetFileFormat()->Read(layer_.operator->(),uri,true),"metadata-only native read unexpectedly succeeded");
            expectedError(errors,OMNI_USD_LIVE_UNSUPPORTED);
            check(attachedLiveScene(*layer_)==attachment,"rejected metadata read changed backing");
            return layer_;
        });
        // No service edit has been sent yet. This direct Read call exercises
        // the production context rejection rather than a USD cache hit.
        TfErrorMark errors;
        check(!retained->GetFileFormat()->Read(retained.operator->(),uri,false),"wrong-context native read unexpectedly succeeded");
        expectedError(errors,OMNI_USD_LIVE_BUSY);
    }
    std::shared_ptr<SceneAttachment> open(const std::string& uri) {
        return call([&] {
            TfErrorMark errors;
            layer_=SdfLayer::FindOrOpen(uri);
            check(layer_ && errors.IsClean(),"production live FindOrOpen failed");
            check(layer_->GetFileFormat()->GetFormatId()==TfToken("live") && layer_->PermissionToEdit(),
                  "FindOrOpen selected wrong format or read-only live backing");
            auto alias=uri; alias.replace(alias.rfind("root.live"),9,"%72oot.live");
            const auto aliased=SdfLayer::FindOrOpen(alias);
            check(aliased==layer_ && errors.IsClean(),"canonical alias created another SdfLayer");
            const auto attachment=attachedLiveScene(*layer_);
            check(attachment && attachedLiveScene(*aliased)==attachment,"aliases acquired different backing attachments");
            replacements_.key=TfNotice::Register(TfCreateWeakPtr(&replacements_),&Replacements::replaced,SdfLayerHandle(layer_));
            replacements_.changeKey=TfNotice::Register(TfCreateWeakPtr(&replacements_),&Replacements::changed,
                                                       SdfLayerHandle(layer_));
            return attachment;
        });
    }
    Fields read() { return call([&] { check(bool(layer_),"native format layer already retired"); return fields(*layer_); }); }
    bool dirty() { return call([&] { return layer_ && layer_->IsDirty(); }); }
    void save(OmniUsdLiveResult expected) {
        call([&] {
            TfErrorMark errors;
            const auto attachment=attachedLiveScene(*layer_);
            const bool saved=layer_->Save(); // Exercise USD's default clean-state fast path too.
            check(saved==(expected==OMNI_USD_LIVE_OK),"default Sdf Save returned wrong success state");
            if (expected==OMNI_USD_LIVE_OK) {
                check(errors.IsClean(),"successful Sdf Save emitted diagnostics");
                check(layer_->Save(true) && errors.IsClean(),"forced Sdf Save failed or emitted diagnostics");
                check(attachedLiveScene(*layer_)==attachment,"forced Sdf Save replaced the backing attachment");
            }
            else {
                expectedError(errors,expected);
                check(layer_->IsDirty(),"failed Sdf Save marked pending/failed layer clean");
            }
        });
    }
    void reload(const std::shared_ptr<SceneAttachment>& attachment) {
        call([&] {
            TfErrorMark errors;
            check(layer_->Reload(true) && errors.IsClean(),"forced native live reload failed");
            check(attachedLiveScene(*layer_)==attachment,"forced reload replaced canonical backing state");
        });
    }
    // Plain Sdf authoring of one root field in one change block. Returns the
    // ID of the one local intent that change block's flush admitted, or 0
    // when it admitted none.
    uint64_t edit(const std::shared_ptr<SceneAttachment>& attachment,const std::string& key,int value) {
        return test::authorAdmittedIntent([this](std::function<void()> job) { enqueue(std::move(job)); },attachment,[&] {
            check(attachedLiveScene(*layer_)==attachment,"edit escaped production backing");
            SdfChangeBlock changes;
            layer_->SetField(SdfPath::AbsoluteRootPath(),TfToken(key),VtValue(value));
        }).id;
    }
    // Author root fields with plain Sdf calls inside one change block. The
    // flush runs as a later host job, so no local intent is admitted here.
    // BeforeHeld: the withheld publication runs before that flush job, and
    // the returned fields are read in the job right after the publication.
    // BetweenFlushAndHeld: the first field's flush job runs, then the second
    // field is authored, then the withheld publication runs.
    // HoldLocalPublication: the edit is authored and publication is armed in
    // the same host job, so the next dispatched job, the local publication
    // of the flush's admission, is withheld.
    // FlushBeforeHeld: the edit's flush job runs before the withheld
    // publication, and the returned fields are read in the job right after it.
    enum class DirectOrder { Plain, BeforeHeld, BetweenFlushAndHeld, HoldLocalPublication, FlushBeforeHeld };
    std::shared_ptr<Fields> directSet(const std::shared_ptr<SceneAttachment>& attachment,const Fields& values,
                                      DirectOrder order) {
        auto installed=std::make_shared<Fields>();
        auto author=[this](const Fields& fields) {
            SdfChangeBlock changes;
            for(const auto& field:fields)
                layer_->SetField(SdfPath::AbsoluteRootPath(),TfToken(field.first),VtValue(field.second));
        };
        call([&] {
            check(attachedLiveScene(*layer_)==attachment,"direct edit escaped production backing");
            check(layer_->PermissionToEdit(),"attached live layer is not editable");
            const auto admitted=attachment->snapshot()->localAdmitted;
            if(order==DirectOrder::BetweenFlushAndHeld) {
                check(values.size()==2,"flush ordering needs exactly two fields");
                author({*values.begin()});
                check(attachment->snapshot()->localAdmitted==admitted,"direct edit admitted intent inside the host mutation");
                enqueue([this,author,second=*std::next(values.begin())] { author({second}); });
                releaseHeld();
                return;
            }
            author(values);
            check(attachment->snapshot()->localAdmitted==admitted,"direct edit admitted intent inside the host mutation");
            if(order==DirectOrder::BeforeHeld) {
                releaseHeldFirst();
                enqueueSecond([this,installed] { *installed=fields(*layer_); });
            }
            if(order==DirectOrder::HoldLocalPublication) arm();
            if(order==DirectOrder::FlushBeforeHeld) {
                releaseHeld();
                enqueue([this,installed] { *installed=fields(*layer_); });
            }
        });
        return installed;
    }
    // Author one root field in its own host job, as a host that catches
    // std::exception around Sdf calls does. Reports the typed result of a
    // rejection thrown before the layer content changed (OK when the edit
    // applied), the change notices the layer sent for the call and whether
    // the layer reports dirty afterwards.
    struct TryOutcome { OmniUsdLiveResult refused=OMNI_USD_LIVE_OK; unsigned notices=0; bool dirty=false; };
    TryOutcome directSetTry(const std::string& key,int value) { return call([&] { return authorTry(key,value); }); }
    // Author `frames` one-field turns in one host job, each followed by the
    // flush its flush job runs, as a host frame does. The job holds the host,
    // so no local publication, which issuance waits for, runs before it
    // returns: an admitted write stays waiting to be issued throughout.
    // Returns the intents the frames admitted.
    uint64_t busyFrames(unsigned frames,const std::string& prefix,const std::shared_ptr<SceneAttachment>& attachment) {
        return call([&] {
            check(layer_->PermissionToEdit(),"attached live layer is not editable");
            auto* bound=fileformat::BoundLayer::from(layer_.operator->());
            check(bound!=nullptr,"live layer is not bound");
            const auto before=attachment->snapshot()->localAdmitted;
            for(unsigned frame=0;frame<frames;++frame) {
                { SdfChangeBlock changes; layer_->SetField(SdfPath::AbsoluteRootPath(),TfToken(prefix+std::to_string(frame)),VtValue(int(frame))); }
                bool issued=false;
                check(bound->flushLocalEdits(&issued,true)!=fileformat::BoundLayer::FlushOutcome::Failed,"frame flush failed");
            }
            return attachment->snapshot()->localAdmitted-before;
        });
    }
    // Author `turns` change blocks of `fields` root fields each inside one
    // host job, so no flush job runs between them, as a host that authors
    // faster than it runs dispatched jobs does. Field `f` of turn `t` is
    // `prefix` + f, or with `distinct` and t >= `repeated` `prefix` + t + "_" + f,
    // or with `numbered` `prefix` + (t * `fields` + f), set to t. Reports the
    // blocks the buffer quota refused, the first of them (or `turns`), and the
    // records and bytes the buffer then holds.
    struct TurnsOutcome { unsigned refused=0,firstRefused=0; size_t buffered=0,bufferedBytes=0; bool endpointBound=false; };
    TurnsOutcome directTurns(unsigned turns,unsigned fields,const std::string& prefix,bool distinct,unsigned repeated=0,
                             bool numbered=false) {
        return call([&] { return authorTurns(layer_,turns,fields,prefix,distinct,repeated,numbered); },
                    std::chrono::seconds(240));
    }
    // directTurns on the second layer, without distinct or numbered fields.
    TurnsOutcome secondTurns(unsigned turns,unsigned fields,const std::string& prefix) {
        return call([&] { check(bool(second_),"no second live layer"); return authorTurns(second_,turns,fields,prefix,false); });
    }
    // In one host job, author `turns` blocks of `fields` fields on the first
    // layer as directTurns does, then one block of `secondFields` fields on
    // the second layer. Reports both outcomes, the deferral cause the second
    // layer's buffered count then holds, whether its content gained any of
    // the block's fields and the second layer's public status (`secondUri`).
    // The job holds the host, so the first layer's flush job, which releases
    // the endpoint budget, runs only after it returns; the status is polled
    // until it reports BLOCKED or 5 seconds pass, and the last one read is
    // reported.
    struct SharedOutcome {
        TurnsOutcome first,second; OmniUsdLiveResult secondBlocked=OMNI_USD_LIVE_OK; bool secondChanged=false;
        uint32_t secondState=0; OmniUsdLiveResult secondResult=OMNI_USD_LIVE_OK;
    };
    SharedOutcome sharedBudgetTurns(unsigned turns,unsigned fields,const std::string& prefix,unsigned secondFields,
                                    const std::string& secondPrefix,const std::string& secondUri) {
        return call([&] {
            check(bool(second_),"no second live layer");
            SharedOutcome outcome;
            outcome.first=authorTurns(layer_,turns,fields,prefix,false);
            outcome.second=authorTurns(second_,1,secondFields,secondPrefix,false);
            const auto* bound=fileformat::BoundLayer::from(second_.operator->());
            check(bound!=nullptr,"second live layer is not bound");
            outcome.secondBlocked=bound->binding->bufferedEdits()->blocked();
            for(unsigned field=0;field<secondFields;++field)
                if(second_->HasField(SdfPath::AbsoluteRootPath(),TfToken(secondPrefix+std::to_string(field))))
                    outcome.secondChanged=true;
            auto status=record<OmniUsdLiveStatus>();
            const auto until=Clock::now()+std::chrono::seconds(5);
            for (;;) {
                check(omniUsdLiveGetLayerStatus(secondUri.c_str(),&status)==OMNI_USD_LIVE_OK,"second layer status query failed");
                if (status.state==OMNI_USD_LIVE_STATE_BLOCKED || Clock::now()>=until) break;
                std::this_thread::yield();
            }
            outcome.secondState=status.state; outcome.secondResult=status.result;
            return outcome;
        });
    }
    // The second layer's buffered edit count, which any thread may read.
    std::shared_ptr<BufferedEditCount> secondBuffered() {
        return call([&] {
            check(bool(second_),"no second live layer");
            const auto* bound=fileformat::BoundLayer::from(second_.operator->());
            check(bound!=nullptr,"second live layer is not bound");
            return bound->binding->bufferedEdits();
        });
    }
    // Park the host worker in a job, then return. The job waits until
    // `failing` has failed and then authors as directSetTry does, so no host
    // job queued after the park, such as a flush the failure requests, runs
    // before the edit. The future yields the outcome.
    std::shared_future<TryOutcome> parkTryOnFailure(const std::string& key,int value,
                                                    const std::shared_ptr<SceneAttachment>& failing) {
        const auto parked=std::make_shared<std::promise<void>>();
        auto parkedFuture=parked->get_future();
        const auto outcome=std::make_shared<std::promise<TryOutcome>>();
        auto result=outcome->get_future().share();
        enqueue([this,parked,outcome,key,value,failing] {
            parked->set_value();
            try {
                progressEvents().until([&] { return failing->progress().result!=OMNI_USD_LIVE_OK; },
                                       "native format failure did not arrive");
                outcome->set_value(authorTry(key,value));
            } catch(...) { outcome->set_exception(std::current_exception()); }
        });
        check(parkedFuture.wait_for(std::chrono::seconds(5))==std::future_status::ready,"host gate did not park");
        return result;
    }
private:
    // Host thread only: the turns directTurns describes, on `layer`.
    TurnsOutcome authorTurns(const SdfLayerRefPtr& layer,unsigned turns,unsigned fields,const std::string& prefix,
                             bool distinct,unsigned repeated=0,bool numbered=false) {
        check(layer->PermissionToEdit(),"attached live layer is not editable");
        TurnsOutcome outcome; outcome.firstRefused=turns;
        TfErrorMark errors;
        for(unsigned turn=0;turn<turns;++turn) {
            try {
                SdfChangeBlock changes;
                const auto stem=distinct && turn>=repeated ? prefix+std::to_string(turn)+"_" : prefix;
                for(unsigned field=0;field<fields;++field) {
                    const auto name=numbered ? prefix+std::to_string(turn*fields+field) : stem+std::to_string(field);
                    layer->SetField(SdfPath::AbsoluteRootPath(),TfToken(name),VtValue(int(turn)));
                }
            } catch(const LayerOperationRejected& rejected) {
                check(rejected.result()==OMNI_USD_LIVE_RESOURCE_EXHAUSTED,"turn refused for another cause");
                if(!outcome.refused) outcome.firstRefused=turn;
                ++outcome.refused;
            }
        }
        // Whether a refusal's USD runtime error names the endpoint's shared budget.
        for(const auto& error:errors)
            if(error.GetCommentary().find("max_pending_bytes_per_endpoint")!=std::string::npos) outcome.endpointBound=true;
        errors.Clear();
        const auto* backing=dynamic_cast<const HostSceneProjection*>(FormatBacking::data(*layer));
        check(backing!=nullptr,"live layer has no host projection backing");
        outcome.buffered=backing->pendingLocalEditCount(); outcome.bufferedBytes=backing->pendingLocalEditBytes();
        return outcome;
    }
    // Host thread only.
    TryOutcome authorTry(const std::string& key,int value) {
        struct Notices final : TfWeakBase {
            unsigned count=0;
            void changed(const SdfNotice::LayersDidChangeSentPerLayer&, const SdfLayerHandle&) { ++count; }
        };
        check(layer_->PermissionToEdit(),"attached live layer is not editable");
        Notices notices;
        auto registration=TfNotice::Register(TfCreateWeakPtr(&notices),&Notices::changed,SdfLayerHandle(layer_));
        TryOutcome outcome;
        TfErrorMark errors;
        try { SdfChangeBlock changes; layer_->SetField(SdfPath::AbsoluteRootPath(),TfToken(key),VtValue(value)); }
        catch(const LayerOperationRejected& rejected) { outcome.refused=rejected.result(); }
        errors.Clear();
        TfNotice::Revoke(registration);
        outcome.notices=notices.count; outcome.dirty=layer_->IsDirty();
        return outcome;
    }
public:
    // Author a probe root field. While the flush job runs, every probe copy
    // performs an off-host save and records its result.
    std::vector<OmniUsdLiveResult> directSetProbe(const std::shared_ptr<SceneAttachment>& attachment) {
        std::vector<OmniUsdLiveResult> saves;
        const auto protectedLayer=call([&] {
            check(attachedLiveScene(*layer_)==attachment && layer_->PermissionToEdit(),"probe fixture is not editable");
            { SdfChangeBlock changes; layer_->SetField(SdfPath::AbsoluteRootPath(),TfToken("probe"),VtValue(FlushProbe{})); }
            check(layer_->GetField(SdfPath::AbsoluteRootPath(),TfToken("probe")).IsHolding<FlushProbe>(),"probe field was not stored");
            flushProbeHook=[&saves,layer=layer_] {
                auto save=std::async(std::launch::async,[&] { return saveAttachedLiveScene(*layer,0); });
                check(save.wait_for(std::chrono::seconds(5))==std::future_status::ready,"off-host save blocked on the host");
                saves.push_back(save.get());
            };
            return layer_;
        });
        call([] {}); // FIFO barrier: the queued flush job has run.
        call([] { flushProbeHook={}; });
        saves.push_back(saveAttachedLiveScene(*protectedLayer,0)); // Off host, after the flush.
        call([&] { const SdfLayerHandle weak(layer_); layer_.Reset(); });
        return saves;
    }
    // Author root fields in one host job whose first record dispatches the
    // flush job; that job's first allocation fails once. Returns whether it
    // failed, read in a host job queued after the flush job.
    bool directSetAllocationFault(const std::shared_ptr<SceneAttachment>& attachment,const Fields& values) {
        call([&] {
            check(attachedLiveScene(*layer_)==attachment && layer_->PermissionToEdit(),"allocation fault fixture is not editable");
            { std::lock_guard<std::mutex> lock(mutex_); check(!failAllocationNext_,"allocation fault already armed"); failAllocationNext_=true; }
            {
                SdfChangeBlock changes;
                for(const auto& field:values) layer_->SetField(SdfPath::AbsoluteRootPath(),TfToken(field.first),VtValue(field.second));
            }
            std::lock_guard<std::mutex> lock(mutex_);
            check(!failAllocationNext_,"direct edit dispatched no flush job");
        });
        return call([&] { return std::exchange(allocationFaulted_,false); });
    }
    // Author one root field, then release the last layer reference in the
    // same host job, so the flush job it enqueued runs after the layer died.
    void directSetDrop(const std::shared_ptr<SceneAttachment>& attachment,const std::string& key,int value) {
        call([&] {
            check(attachedLiveScene(*layer_)==attachment && layer_->PermissionToEdit(),"direct drop fixture is not editable");
            { SdfChangeBlock changes; layer_->SetField(SdfPath::AbsoluteRootPath(),TfToken(key),VtValue(value)); }
            const SdfLayerHandle weak(layer_);
            layer_.Reset(); check(!weak,"last production layer reference did not retire");
        });
    }
    // Open or release another live layer of the same session and endpoint.
    void openSecond(const std::string& uri) {
        call([&] {
            TfErrorMark errors;
            second_=SdfLayer::FindOrOpen(uri);
            check(second_ && errors.IsClean() && attachedLiveScene(*second_),"second live layer open failed");
        });
    }
    void dropSecond() {
        call([&] {
            const SdfLayerHandle weak(second_);
            second_.Reset(); check(!weak,"second live layer reference did not retire");
        });
    }
    // Define a root prim with plain Sdf.
    void createRootPrim(const std::string& name) {
        call([&] {
            SdfChangeBlock changes;
            const auto prim=SdfCreatePrimInLayer(layer_,SdfPath("/"+name));
            check(bool(prim),"temporary root prim creation failed");
            prim->SetSpecifier(SdfSpecifierDef);
        });
    }
    // In one host job, so no queued flush job runs first: remove a root prim
    // with plain Sdf, run SdfLayer::Save, then read the layer's fields.
    std::pair<bool,Fields> removeAndSave(const std::string& name) {
        return call([&] {
            {
                SdfChangeBlock changes;
                const auto prim=layer_->GetPrimAtPath(SdfPath("/"+name));
                check(bool(prim),"temporary root prim is missing");
                layer_->RemoveRootPrim(prim);
            }
            TfErrorMark errors;
            const bool saved=layer_->Save();
            errors.Clear();
            return std::make_pair(saved,fields(*layer_));
        });
    }
    // In one host turn with plain Sdf: create a root prim with an int
    // attribute `v` whose whole timeSamples map holds `count` keys.
    void animateRootPrim(const std::string& name,int count) {
        call([&] {
            SdfChangeBlock changes;
            const auto prim=SdfCreatePrimInLayer(layer_,SdfPath("/"+name));
            check(bool(prim),"animated root prim creation failed");
            prim->SetSpecifier(SdfSpecifierDef);
            const auto attribute=SdfAttributeSpec::New(prim,"v",SdfValueTypeNames->Int);
            check(bool(attribute),"animated attribute creation failed");
            SdfTimeSampleMap samples;
            for(int i=0;i<count;++i) samples.emplace(double(i),VtValue(i));
            layer_->SetField(attribute->GetPath(),SdfFieldKeys->TimeSamples,VtValue(samples));
        });
    }
    // With plain Sdf, one host turn each: create a root prim, then append
    // `count` child prims c0, c1, ... An append alone emits no Reorder.
    void createParent(const std::string& name,int count) {
        call([&] {
            SdfChangeBlock changes;
            const auto prim=SdfCreatePrimInLayer(layer_,SdfPath("/"+name));
            check(bool(prim),"parent prim creation failed");
            prim->SetSpecifier(SdfSpecifierDef);
        });
        for(int i=0;i<count;++i)
            call([&] {
                const auto prim=layer_->GetPrimAtPath(SdfPath("/"+name));
                check(bool(prim) && bool(SdfPrimSpec::New(prim,"c"+std::to_string(i),SdfSpecifierDef)),
                      "child prim creation failed");
            });
    }
    // In one host turn with plain Sdf: reverse a root prim's child order.
    void reverseChildren(const std::string& name) {
        call([&] {
            const SdfPath path("/"+name);
            auto order=layer_->GetFieldAs<std::vector<TfToken>>(path,SdfChildrenKeys->PrimChildren);
            check(order.size()>1,"reversed prim has fewer than two children");
            std::reverse(order.begin(),order.end());
            layer_->SetField(path,SdfChildrenKeys->PrimChildren,VtValue(order));
        });
    }
    // SdfLayer::Export an anonymous seed with `count` int root fields f0, f1,
    // ... to `uri`; reports whether it succeeded and the Tf errors it posted.
    std::pair<bool,std::vector<std::string>> exportFields(const std::string& uri,int count) {
        return call([&] {
            const auto seed=SdfLayer::CreateAnonymous("fields-export.usda");
            check(bool(seed),"export seed allocation failed");
            for(int i=0;i<count;++i) seed->SetField(SdfPath::AbsoluteRootPath(),TfToken("f"+std::to_string(i)),VtValue(i));
            TfErrorMark errors;
            const bool exported=seed->Export(uri);
            std::vector<std::string> posted;
            for(const auto& error:errors) posted.push_back(error.GetCommentary());
            errors.Clear();
            return std::make_pair(exported,posted);
        });
    }
    bool hasPrim(const std::string& name) { return bool(layer_->GetPrimAtPath(SdfPath("/"+name))); }
    // Author an unencodable root value with plain Sdf; its flush fails the layer.
    void authorUnencodable(const std::string& key) {
        check(layer_ && layer_->PermissionToEdit(),"unencodable fixture is not editable");
        SdfChangeBlock changes; layer_->SetField(SdfPath::AbsoluteRootPath(),TfToken(key),VtValue(FlushProbe{}));
    }
    // Author a root payload whose layer offset version zero cannot represent.
    void authorPayloadOffset(const std::string& key) {
        check(layer_ && layer_->PermissionToEdit(),"payload offset fixture is not editable");
        SdfChangeBlock changes;
        layer_->SetField(SdfPath::AbsoluteRootPath(),TfToken(key),
                         VtValue(SdfPayload("model.usda",SdfPath("/Root"),SdfLayerOffset(3,1))));
    }
    std::string layerIdentifier() { return call([&] { return layer_->GetIdentifier(); }); }
    unsigned replacements() { return call([&] { return replacements_.count.load(); }); }
    unsigned changes() { return call([&] { return replacements_.changes.load(); }); }
    // Open a stage whose anonymous root layer sublayers the live layer.
    size_t openStage() {
        return call([&] {
            check(!stage_,"stage already open");
            const auto root=SdfLayer::CreateAnonymous("stage-root.usda");
            root->InsertSubLayerPath(layer_->GetIdentifier());
            stage_=UsdStage::Open(root);
            check(stage_ && stage_->GetLayerStack(false).size()==2,"stage did not compose the live layer");
            stageNotices_.key=TfNotice::Register(TfCreateWeakPtr(&stageNotices_),&StageNotices::changed,
                                                 UsdStageWeakPtr(stage_));
            return stage_->GetLayerStack(false).size();
        });
    }
    // The stage's notices since the last take, then cleared.
    struct TakenNotices { std::set<SdfPath> resynced, info; unsigned notices=0; };
    TakenNotices takeStageNotices() {
        return call([&] {
            check(bool(stage_),"no stage open");
            TakenNotices taken; taken.resynced.swap(stageNotices_.resynced); taken.info.swap(stageNotices_.info);
            taken.notices=std::exchange(stageNotices_.notices,0u);
            return taken;
        });
    }
    // Whether the stage composes a prim at `path`.
    bool stageHasPrim(const std::string& path) { return call([&] { return bool(stage_->GetPrimAtPath(SdfPath(path))); }); }
    void armNoticeAuthor(const std::string& key,int value) {
        call([&] {
            noticeAuthor_.layer=layer_; noticeAuthor_.key=TfToken(key); noticeAuthor_.value=value; noticeAuthor_.armed=true;
            noticeAuthor_.registration=TfNotice::Register(TfCreateWeakPtr(&noticeAuthor_),&NoticeAuthor::changed,
                                                          SdfLayerHandle(layer_));
        });
    }
    // Author one root field on the calling thread, which is not the host
    // context, while a gate job parks the host worker, so no host job runs
    // during the foreign mutation. Returns the typed result of a rejection
    // thrown before the layer content changed, or OK when the edit applied.
    OmniUsdLiveResult directSetOffHost(const std::string& key,int value) {
        const auto layer=call([&] { return layer_; });
        // The gate job owns `parked`, and every exit releases the worker, so
        // a failed park check leaves no dangling reference or held worker.
        const auto parked=std::make_shared<std::promise<void>>();
        auto parkedFuture=parked->get_future();
        std::promise<void> release;
        auto released=release.get_future().share();
        enqueue([parked,released] { parked->set_value(); released.wait(); });
        auto result=OmniUsdLiveResult(OMNI_USD_LIVE_OK);
        try {
            check(parkedFuture.wait_for(std::chrono::seconds(5))==std::future_status::ready,"host gate did not park");
            TfErrorMark errors;
            try { layer->SetField(SdfPath::AbsoluteRootPath(),TfToken(key),VtValue(value)); }
            catch(const LayerOperationRejected& rejected) { result=rejected.result(); }
            errors.Clear();
        } catch(...) { release.set_value(); throw; }
        release.set_value();
        return result;
    }
    // Run SdfLayer::Save off the host while a gate job parks the host, so the
    // format's off-host path sees the layer's buffered edits unchanged.
    void saveOffHost(OmniUsdLiveResult expected) {
        const auto layer=call([&] { return layer_; });
        const auto parked=std::make_shared<std::promise<void>>();
        auto parkedFuture=parked->get_future();
        std::promise<void> release;
        auto released=release.get_future().share();
        enqueue([parked,released] { parked->set_value(); released.wait(); });
        try {
            check(parkedFuture.wait_for(std::chrono::seconds(5))==std::future_status::ready,"host gate did not park");
            TfErrorMark errors;
            check(!layer->Save(),"off-host Sdf Save succeeded");
            expectedError(errors,expected);
        } catch(...) { release.set_value(); throw; }
        release.set_value();
    }
    // Save the attached layer through the off-host production path.
    OmniUsdLiveResult saveAttachedOffHost(uint32_t timeoutMs) {
        const auto layer=call([&] { return layer_; });
        return saveAttachedLiveScene(*layer,timeoutMs);
    }
    // Forced SdfLayer::Save on the host; true when it succeeds without diagnostics.
    bool forcedSaveOnHost() {
        return call([&] { TfErrorMark errors; const bool saved=layer_->Save(true); return saved && errors.IsClean(); });
    }
    // SdfLayer::Export the live layer to an ordinary file, as FAILED recovery
    // does, and read the exported file's root fields.
    Fields exportLayer(const std::string& path) {
        return call([&] {
            check(bool(layer_) && layer_->Export(path),"live layer export failed");
            const auto exported=SdfLayer::OpenAsAnonymous(path);
            check(bool(exported),"exported live layer did not open");
            return fields(*exported);
        });
    }
    struct LiveCreation {
        bool succeeded=false, crossed=false, diagnosed=false, alive=false, editable=false;
        unsigned hostDispatches=0; Fields fields;
    };
    // SdfLayer::Export an anonymous seed to `uri` and keep no reference to the
    // destination, so its adoption job drops the last one. A seam in that job,
    // after adoption and before the slot is freed, records whether the
    // destination layer is still alive (its close has not started) and whether
    // the slot is still listed; a FIFO fence then runs the job to completion.
    struct UnreferencedExport { bool succeeded=false, adopted=false, aliveAtAdoption=false, listedAtAdoption=false, retainedAfter=true; };
    UnreferencedExport exportUnreferenced(const std::string& uri,int counter) {
        struct Probe {
            std::string uri; bool adopted=false, alive=false, listed=false;
            static void hook(void* context) noexcept {
                if(!context) return; // The job can pair this hook with a cleared context.
                auto& self=*static_cast<Probe*>(context);
                self.adopted=true;
                self.alive=bool(SdfLayer::Find(self.uri));
                RetainedUsdCreationInfo info;
                self.listed=nextRetainedUsdCreation(0,info);
            }
        };
        // Static: the seam may still read it while it is disarmed.
        static Probe probe;
        probe=Probe{}; probe.uri=uri;
        UsdCreationLease::setAfterAdoptionForTest(Probe::hook,&probe);
        struct Disarm { ~Disarm() { UsdCreationLease::setAfterAdoptionForTest(nullptr,nullptr); } } disarm;
        UnreferencedExport outcome;
        outcome.succeeded=call([&] {
            const auto seed=SdfLayer::CreateAnonymous("unreferenced-export.usda");
            check(bool(seed),"export seed allocation failed");
            seed->SetField(SdfPath::AbsoluteRootPath(),TfToken("counter"),VtValue(counter));
            TfErrorMark errors;
            const bool exported=seed->Export(uri);
            return exported && errors.IsClean();
        });
        call([] {}); // FIFO fence: the adoption job the Export queued has run.
        outcome.adopted=probe.adopted; outcome.aliveAtAdoption=probe.alive; outcome.listedAtAdoption=probe.listed;
        RetainedUsdCreationInfo info;
        outcome.retainedAfter=nextRetainedUsdCreation(0,info);
        return outcome;
    }
    // SdfLayer::Export an anonymous seed to `uri` with the creation clock
    // frozen at its start until the destination's identifier notice, which
    // activation sends after SetIdentifier; from then on the clock reads one
    // hour later. Reports the destination layer, which exists once the
    // creation activated, and keeps it: the adoption job the hook queued
    // hands it to its binding after this host job.
    LiveCreation exportCrossingAtIdentifier(const std::string& uri,int counter) {
        return call([&] {
            struct Crossing final : TfWeakBase {
                std::string uri; std::atomic<bool> crossed{false};
                void renamed(const SdfNotice::LayerIdentifierDidChange& notice) {
                    if(notice.GetNewIdentifier()==uri) crossed.store(true);
                }
            } crossing;
            crossing.uri=uri;
            const auto start=Clock::now();
            setLiveLayerCreationClockForTest([&crossing,start] {
                return crossing.crossed.load() ? start+std::chrono::hours(1) : start;
            });
            struct Restore {
                TfNotice::Key key;
                ~Restore() { TfNotice::Revoke(key); setLiveLayerCreationClockForTest({}); }
            } restore{TfNotice::Register(TfCreateWeakPtr(&crossing),&Crossing::renamed)};
            LiveCreation outcome;
            {
                const auto seed=SdfLayer::CreateAnonymous("deadline-export.usda");
                check(bool(seed),"export seed allocation failed");
                seed->SetField(SdfPath::AbsoluteRootPath(),TfToken("counter"),VtValue(counter));
                TfErrorMark errors;
                outcome.succeeded=seed->Export(uri);
                outcome.diagnosed=!errors.IsClean(); errors.Clear();
            }
            outcome.crossed=crossing.crossed.load();
            created_=SdfLayer::Find(uri);
            if(created_) {
                outcome.alive=true; outcome.editable=created_->PermissionToEdit(); outcome.fields=fields(*created_);
            }
            return outcome;
        });
    }
    // SdfLayer::CreateNew `uri` with the creation clock frozen at its start
    // until the host accepts a job dispatched from the host thread during the
    // call (the hook's finalizer); from then on the clock reads one hour
    // later. A FIFO fence runs the finalizer before the target is inspected.
    LiveCreation createNewCrossingAtDispatch(const std::string& uri) {
        const auto crossed=std::make_shared<std::atomic<bool>>(false);
        const auto start=Clock::now();
        setLiveLayerCreationClockForTest([crossed,start] { return crossed->load() ? start+std::chrono::hours(1) : start; });
        struct Restore { ~Restore() { setLiveLayerCreationClockForTest({}); } } restore;
        LiveCreation outcome;
        call([&] {
            { std::lock_guard<std::mutex> lock(mutex_); crossOnHostDispatch_=crossed; hostDispatches_=0; }
            TfErrorMark errors;
            created_=SdfLayer::CreateNew(uri);
            outcome.succeeded=bool(created_);
            outcome.diagnosed=!errors.IsClean(); errors.Clear();
            std::lock_guard<std::mutex> lock(mutex_);
            crossOnHostDispatch_.reset(); outcome.hostDispatches=hostDispatches_;
        });
        call([&] { // Fence: the finalizer the hook queued has run.
            outcome.alive=bool(created_);
            if(created_) { outcome.editable=created_->PermissionToEdit(); outcome.fields=fields(*created_); }
        });
        outcome.crossed=crossed->load();
        return outcome;
    }
    // Take every retained creation workflow on the host and close it, wait
    // for its closure off the host, then release it and any created layer
    // on the host. Returns how many it retired.
    size_t retireCreations() {
        auto owned=call([&] {
            std::vector<std::shared_ptr<UsdLayerCreation>> taken;
            RetainedUsdCreationInfo info;
            for(uint64_t after=0; nextRetainedUsdCreation(after,info); after=info.subjectId) {
                auto lease=acquireUsdCreation(info.subjectId);
                taken.push_back(lease.take());
                const auto closed=taken.back()->close();
                check(closed==OMNI_USD_LIVE_OK || closed==OMNI_USD_LIVE_BUSY,"retained creation close failed");
            }
            return taken;
        });
        for(const auto& creation:owned) check(creation->waitForClose(5000)==OMNI_USD_LIVE_OK,"retained creation did not close");
        const auto count=owned.size();
        call([&] { owned.clear(); created_.Reset(); });
        return count;
    }
    // The records the layer's current backing projection retains.
    size_t retainedRecords() {
        return call([&] {
            const auto* backing=dynamic_cast<const HostSceneProjection*>(FormatBacking::data(*layer_));
            check(backing!=nullptr,"live layer has no host projection backing");
            return backing->pendingLocalEditCount();
        });
    }
    // Whole-content replacement attempts on the live layer, made the way a
    // host that catches std::exception around Sdf calls makes them.
    struct Replacement {
        unsigned transfer=0; bool typed=false;     // TransferContent's rejection
        bool imported=true; unsigned importResult=0; // Import and its typed format diagnostic
        bool importedString=true;
    };
    Replacement replaceContent(const std::string& importPath) {
        return call([&] {
            const auto source=SdfLayer::CreateAnonymous("replacement.usda");
            check(source && source->ImportFromString("#usda 1.0\ndef \"Replacement\" {}\n"),"replacement seed failed");
            check(source->Export(importPath),"replacement import source export failed");
            Replacement outcome;
            TfErrorMark errors;
            try { layer_->TransferContent(source); }
            catch(const std::exception& error) {
                const auto* rejected=dynamic_cast<const LayerOperationRejected*>(&error);
                outcome.typed=rejected!=nullptr;
                outcome.transfer=rejected ? rejected->result() : OMNI_USD_LIVE_INTERNAL_ERROR;
            }
            errors.Clear();
            outcome.imported=layer_->Import(importPath);
            // A typed live-format diagnostic shows the import reached this format's Read hook.
            for(const auto& error:errors)
                if(const auto* result=error.GetInfo<OmniUsdLiveResult>()) outcome.importResult=*result;
            errors.Clear();
            outcome.importedString=layer_->ImportFromString("#usda 1.0\ndef \"Other\" {}\n");
            errors.Clear();
            return outcome;
        });
    }
    // Clear the layer, then author any given root fields in the same host job,
    // before any job queued by Clear's notice runs.
    void clearLayer(const Fields& values) {
        call([&] {
            { TfErrorMark errors; layer_->Clear(); errors.Clear(); }
            for(const auto& field:values) layer_->SetField(SdfPath::AbsoluteRootPath(),TfToken(field.first),VtValue(field.second));
        });
    }
    void drop() {
        call([&] {
            TfNotice::Revoke(replacements_.key); TfNotice::Revoke(replacements_.changeKey);
            TfNotice::Revoke(noticeAuthor_.registration);
            TfNotice::Revoke(stageNotices_.key); stage_.Reset();
            const SdfLayerHandle weak(layer_);
            layer_.Reset(); check(!weak,"last production layer reference did not retire");
        });
    }
};
struct RuntimeOwner {
    bool initialized=false;
    OmniUsdLiveEndpoint endpoint=0;
    std::shared_ptr<SceneAttachment> attachment;
    void cleanup() {
        if (attachment) { check(attachment->close(5000)==OMNI_USD_LIVE_OK,"native format actor close timed out"); attachment.reset(); }
        if (endpoint) {
            const auto until=Clock::now()+std::chrono::seconds(5);
            for (;;) {
                const auto result=omniUsdLiveUnregisterEndpoint(endpoint,0);
                if (result==OMNI_USD_LIVE_OK) break;
                check(result==OMNI_USD_LIVE_BUSY && Clock::now()<until,"native format endpoint lease leaked");
                std::this_thread::yield();
            }
            endpoint=0;
        }
        if (initialized) { check(omniUsdLiveShutdown(5000)==OMNI_USD_LIVE_OK,"native format runtime shutdown timed out"); initialized=false; }
    }
};
// The members of a turns outcome, as JSON object members.
std::string turnsJson(const Host::TurnsOutcome& outcome) {
    return "\"refused\":"+std::to_string(outcome.refused)+",\"first_refused\":"+std::to_string(outcome.firstRefused)+
           ",\"buffered\":"+std::to_string(outcome.buffered)+",\"buffered_bytes\":"+std::to_string(outcome.bufferedBytes)+
           ",\"endpoint_bound\":"+(outcome.endpointBound ? "true" : "false");
}
void emit(const char* type,const std::shared_ptr<SceneAttachment>& attachment,const Fields& actual,bool dropped=false,bool dirty=false) {
    const auto state=attachment->progress(); const auto journal=attachment->snapshot();
    std::cout << "{\"type\":\"" << type << "\",\"result\":" << state.result << ",\"applied\":" << state.applied
              << ",\"covered\":" << state.covered << ",\"journal_applied\":" << journal->applied
              << ",\"retained\":" << state.retained << ",\"object\":" << state.object
              << ",\"read_started\":" << (state.readStarted?"true":"false")
              << ",\"reading_after_failure\":" << (state.readingAfterFailure?"true":"false")
              << ",\"dirty\":" << (dirty?"true":"false") << ",\"dropped\":" << (dropped?"true":"false") << ",\"fields\":";
    jsonFields(actual); std::cout << ",\"pending_ids\":[";
    for (size_t i=0;i<journal->pending.size();++i) { if (i) std::cout << ','; std::cout << journal->pending[i].id; }
    std::cout << "],\"pending_hex\":[";
    for (size_t i=0;i<journal->pending.size();++i) {
        if (i) std::cout << ',';
        std::cout << '"';
        for (const unsigned char c:journal->pending[i].intent->bytes)
            std::cout << "0123456789abcdef"[c>>4] << "0123456789abcdef"[c&15];
        std::cout << '"';
    }
    std::cout << "]}" << std::endl;
}
}
int fileFormatMain(int argc,char** argv) {
    // Optional arguments: `limits=UPDATE,ENDPOINT` sets the runtime's
    // per-update and per-endpoint pending byte limits; `--max-pending=N` sets
    // the journal's pending-intent limit, `--max-buffered=N` the host edit
    // buffer quota for the layer this driver opens, in the bytes of N records
    // of one root int field, and `--lag-warning=N` the
    // runtime's apply_lag_warning_threshold; `--no-holdback` disables the
    // endpoint's jitter holdback, so every covered peer update is ready for
    // application at once; `--max-requests=N` sets the runtime's request
    // registry capacity; `--max-delta-entries=N` and
    // `--max-children-list-entries=N` lower the service's entry limits for the
    // layers this driver opens and creates; `--max-log-operations=N` sets the
    // journal's field edit compiler raw log operation limit.
    check(argc>=5 && argc<=15,"file-format driver expects authority session seed --file-format [limits=UPDATE,ENDPOINT] [--max-pending=N] [--max-buffered=N] [--lag-warning=N] [--no-holdback] [--max-requests=N] [--max-delta-entries=N] [--max-children-list-entries=N] [--max-log-operations=N] [--request-timeout=MS]");
    uint32_t updateLimit=0, maxRequests=0, requestTimeout=0; uint64_t endpointLimit=0, lagWarning=0;
    bool noHoldback=false; std::optional<DeltaEntryLimits> entries; std::optional<SceneJournal::Limits> journalLimits;
    for(int index=5;index<argc;++index) {
        const std::string option=argv[index];
        if(option=="--no-holdback") {
            noHoldback=true;
            continue;
        }
        if(option.rfind("--max-delta-entries=",0)==0) {
            if(!entries) entries.emplace();
            entries->maxEntries=std::stoul(option.substr(20));
            check(entries->maxEntries>0,"invalid file-format driver entry limit");
            continue;
        }
        if(option.rfind("--max-children-list-entries=",0)==0) {
            if(!entries) entries.emplace();
            entries->maxChildrenListEntries=std::stoul(option.substr(28));
            check(entries->maxChildrenListEntries>0,"invalid file-format driver children list limit");
            continue;
        }
        if(option.rfind("--lag-warning=",0)==0) {
            lagWarning=std::stoull(option.substr(14));
            check(lagWarning>0,"invalid file-format driver lag warning threshold");
            continue;
        }
        if(option.rfind("--request-timeout=",0)==0) {
            requestTimeout=static_cast<uint32_t>(std::stoul(option.substr(18)));
            check(requestTimeout>0,"invalid file-format driver request timeout");
            continue;
        }
        if(option.rfind("--max-requests=",0)==0) {
            maxRequests=static_cast<uint32_t>(std::stoul(option.substr(15)));
            check(maxRequests>0,"invalid file-format driver request capacity");
            continue;
        }
        if(option.rfind("--max-buffered=",0)==0) {
            const auto quota=std::stoull(option.substr(15));
            check(quota>0,"invalid file-format driver host edit quota");
            // N records of one root int field, the record every direct-set authors.
            SdfOperation record(SdfOperationKind::SetField);
            record.path=SdfPath::AbsoluteRootPath(); record.field=TfToken("quota"); record.value=VtValue(0);
            setLiveLayerHostEditQuotaForTest(static_cast<size_t>(quota)*hostEditRecordBytes(record));
            continue;
        }
        if(option.rfind("--max-pending=",0)==0) {
            if(!journalLimits) journalLimits.emplace();
            journalLimits->maxPending=std::stoul(option.substr(14));
            check(journalLimits->maxPending>0,"invalid file-format driver pending-intent limit");
            continue;
        }
        if(option.rfind("--max-log-operations=",0)==0) {
            if(!journalLimits) journalLimits.emplace();
            journalLimits->authoring.maxLogOperations=std::stoul(option.substr(21));
            check(journalLimits->authoring.maxLogOperations>0,"invalid file-format driver log operation limit");
            continue;
        }
        const std::string prefix="limits=";
        const auto comma=option.find(',');
        check(option.rfind(prefix,0)==0 && comma!=std::string::npos,"invalid file-format driver limits");
        updateLimit=static_cast<uint32_t>(std::stoul(option.substr(prefix.size(),comma-prefix.size())));
        endpointLimit=std::stoull(option.substr(comma+1));
    }
    if(journalLimits) setLiveLayerJournalLimitsForTest(journalLimits);
    std::cerr << "native file-format seed=" << argv[3] << std::endl;
    const auto metadata=std::filesystem::absolute(argv[0]).parent_path()/"native-file-format"/"plugInfo.json";
    check(std::filesystem::is_regular_file(metadata),"native file-format plugin metadata missing");
    PlugRegistry::GetInstance().RegisterPlugins(metadata.string()); // Before any Ar/Sdf entrypoint.
    Host host; RuntimeOwner owner;
    struct Cleanup {
        Host& host; RuntimeOwner& owner;
        ~Cleanup() {
            try {
                host.releaseHeld(); host.drop(); owner.cleanup(); setUsdPublicationDispatcher({});
                setLiveLayerProgressObserverForTest({});
                setLiveLayerCreationClockForTest({});
                setLiveLayerEntryLimitsForTest({});
            } catch (...) { std::cerr << "native file-format cleanup failed\n"; std::_Exit(2); }
        }
    } cleanup{host,owner};
    auto options=record<OmniUsdLiveRuntimeOptions>(); options.initial_open_timeout_ms=4000;
    options.reconnect_initial_ms=5; options.reconnect_maximum_ms=20; // Reconnects within milliseconds.
    options.diagnostic_min_severity=OMNI_USD_LIVE_SEVERITY_DEBUG;
    if(updateLimit) { options.max_update_bytes=updateLimit; options.max_pending_bytes_per_endpoint=endpointLimit; }
    options.apply_lag_warning_threshold=lagWarning;
    if(maxRequests) options.max_requests=maxRequests;
    if(requestTimeout) options.request_timeout_ms=requestTimeout;
    check(omniUsdLiveInitialize(&options)==OMNI_USD_LIVE_OK,"native format runtime initialize failed"); owner.initialized=true;
    auto endpoint=record<OmniUsdLiveEndpointOptions>(); endpoint.authority=argv[1]; endpoint.access_token="synthetic-writer";
    check(omniUsdLiveRegisterEndpoint(&endpoint,&owner.endpoint)==OMNI_USD_LIVE_OK,"native format endpoint registration failed");
    if(updateLimit || noHoldback || maxRequests) {
        // Exact pending-byte and request ledgers and batch counts do not test
        // jitter. Disable holdback and timing Pings, whose response-allowance
        // release would otherwise wake a blocked flush.
        auto jitter=record<OmniUsdLiveJitterParams>();
        check(omniUsdLiveConfigureJitterReduction(owner.endpoint,&jitter)==OMNI_USD_LIVE_OK,
              "native format timing configuration failed");
    }
    setUsdPublicationDispatcher([&host](auto job) { host.dispatch(std::move(job)); },[&host] { return host.current(); });
    const auto uri=makeLayerUri(makeSessionUri(argv[1],argv[2]),"root.live");
    // Attachments the file format opens deliver progress changes to
    // progressEvents through the test observer seam.
    setLiveLayerProgressObserverForTest([](const auto&) { progressEvents().signal(); });
    if(entries) setLiveLayerEntryLimitsForTest(entries);
    host.rejectedOpens(uri);
    owner.attachment=host.open(uri);
    host.rejectedReads(uri,owner.attachment);
    // FindOrOpen has published the initial USD data, but the actor still has
    // to observe completion-owner retirement. Latch that initial sequence
    // outside the host queue so a startup snapshot cannot race bookkeeping.
    const auto bootstrap=owner.attachment->snapshot()->applied;
    check(bootstrap>0,"native format opening omitted initial snapshot");
    progressEvents().until([&] {
        const auto progress=owner.attachment->progress();
        check(progress.result==OMNI_USD_LIVE_OK,"native format initial application did not retire");
        return progress.applied>=bootstrap;
    },"native format initial application did not retire");
    Fields last=host.read(); bool dropped=false;
    std::shared_future<Host::TryOutcome> parkedTry; // From park-try-on-failure until parked-try.
    std::string secondUri; // The second live layer's URI, from open-second until drop-second.
    auto subject=[&] {
        auto status=record<OmniUsdLiveStatus>();
        check(omniUsdLiveGetLayerStatus(uri.c_str(),&status)==OMNI_USD_LIVE_OK && status.subject_id,"missing native layer subject");
        return status.subject_id;
    };
    auto report=[&](const char* type,const Fields& values) {
        emit(type,owner.attachment,values,dropped,!dropped && host.dirty());
    };
    {
        // Authoring starts once a Ping has reported the endpoint's
        // capabilities, so whether writes chain and re-send does not race it.
        const auto registered=getRuntime()->findEndpoint(owner.endpoint);
        const auto until=std::chrono::steady_clock::now()+std::chrono::seconds(10);
        while (!(registered->capabilities.load()&Endpoint::kCapabilitiesKnown)) {
            check(std::chrono::steady_clock::now()<until,"endpoint capabilities did not arrive");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    report("ready",last);
    std::string line;
    while (std::getline(std::cin,line)) {
        std::istringstream input(line); std::string command,extra; input>>command;
        if (command=="set") {
            std::string key; int value; check(bool(input>>key>>value) && identifier(key),"invalid native format field edit");
            // An armed hold would withhold this edit's flush job, so no intent
            // could be admitted before the fence.
            check(!host.armed(),"set refuses to author while a publication hold is armed");
            const auto id=host.edit(owner.attachment,key,value);
            std::cout << "{\"type\":\"accepted\",\"result\":0,\"id\":" << id << '}' << std::endl;
        } else if (command=="direct-set" || command=="direct-set-before-held" || command=="direct-set-between-flush-and-held" ||
                   command=="direct-set-hold-local-publication" || command=="direct-set-flush-before-held") {
            Fields values; std::string key; int value;
            while(input>>key>>value) { check(identifier(key),"invalid direct field name"); values[key]=value; }
            check(!values.empty() && input.eof(),"invalid direct field edits");
            using Order=Host::DirectOrder;
            const auto order=command=="direct-set" ? Order::Plain :
                             command=="direct-set-before-held" ? Order::BeforeHeld :
                             command=="direct-set-between-flush-and-held" ? Order::BetweenFlushAndHeld :
                             command=="direct-set-hold-local-publication" ? Order::HoldLocalPublication : Order::FlushBeforeHeld;
            if(order!=Order::Plain && order!=Order::HoldLocalPublication) host.waitHeld();
            const auto installed=host.directSet(owner.attachment,values,order);
            if(order==Order::HoldLocalPublication) host.waitHeld();
            const auto current=host.read(); // Queued last: every job above has run.
            std::cout << "{\"type\":\"direct\",\"result\":" << owner.attachment->progress().result << ",\"fields\":";
            jsonFields(current); std::cout << ",\"installed\":"; jsonFields(*installed); std::cout << '}' << std::endl;
        } else if (command=="direct-set-alloc-fault") {
            Fields values; std::string key; int value;
            while(input>>key>>value) { check(identifier(key),"invalid direct field name"); values[key]=value; }
            check(!values.empty() && input.eof(),"invalid direct field edits");
            const bool faulted=host.directSetAllocationFault(owner.attachment,values);
            last=host.read();
            std::cout << "{\"type\":\"alloc-fault\",\"result\":" << owner.attachment->progress().result
                      << ",\"injection\":" << (LIVE_TEST_ALLOCATOR?"true":"false")
                      << ",\"faulted\":" << (faulted?"true":"false") << ",\"fields\":";
            jsonFields(last); std::cout << '}' << std::endl;
        } else if (command=="park-try-on-failure") {
            std::string key; int value=0; check(bool(input>>key>>value) && identifier(key),"invalid direct field edit");
            check(!parkedTry.valid(),"a parked edit is already pending");
            parkedTry=host.parkTryOnFailure(key,value,owner.attachment);
            std::cout << "{\"type\":\"parked\"}" << std::endl;
        } else if (command=="direct-set-try" || command=="parked-try") {
            Host::TryOutcome outcome;
            if(command=="direct-set-try") {
                std::string key; int value=0; check(bool(input>>key>>value) && identifier(key),"invalid direct field edit");
                outcome=host.directSetTry(key,value);
            } else {
                check(parkedTry.valid() && parkedTry.wait_for(std::chrono::seconds(10))==std::future_status::ready,
                      "parked edit did not run");
                outcome=parkedTry.get(); parkedTry={};
            }
            last=host.read(); // Queued last: the flush job the edit queued has run.
            std::cout << "{\"type\":\"direct-try\",\"refused\":" << outcome.refused << ",\"notices\":" << outcome.notices
                      << ",\"dirty\":" << (outcome.dirty?"true":"false") << ",\"fields\":";
            jsonFields(last); std::cout << '}' << std::endl;
        } else if (command=="applied") {
            uint64_t sequence=0; check(bool(input>>sequence),"missing applied sequence");
            progressEvents().until([&] { return owner.attachment->progress().applied>=sequence; },
                                   "native format application did not reach the sequence");
            host.call([] {}); // Fence: the publication job for that sequence has run.
            last=host.read(); report("applied",last);
        } else if (command=="covered") {
            // The coverage frontier: received and retained, whether or not applied.
            uint64_t sequence=0; check(bool(input>>sequence),"missing covered sequence");
            progressEvents().until([&] { return owner.attachment->progress().covered>=sequence; },
                                   "native format coverage did not reach the sequence");
            const auto progress=owner.attachment->progress();
            std::cout << "{\"type\":\"covered\",\"covered\":" << progress.covered << ",\"applied\":" << progress.applied
                      << '}' << std::endl;
        } else if (command=="journal") {
            // The journal frontier, which keeps advancing after an issuance-only FAILED.
            uint64_t sequence=0; check(bool(input>>sequence),"missing journal sequence");
            progressEvents().until([&] { return owner.attachment->snapshot()->applied>=sequence; },
                                   "native format journal did not reach the sequence");
            host.call([] {}); // Fence: any publication job queued for that sequence has run.
            last=host.read(); report("journal",last);
        } else if (command=="alloc-sweep-open") {
            // Fail each allocation, in turn, of the Read hook up to its
            // bootstrap wait and then of SdfLayer::New of the live format.
            // Each failure reports one cause and leaves nothing attached; each
            // sweep ends at its first uninjected call (the control). An
            // allocation failure inside USD's own layer constructor can leave
            // USD's layer registry holding a dead entry for the identifier, and
            // a later layer of that identifier aborts inside USD, so the New
            // sweep continues under a fresh identifier after every throw and
            // its control opens `control`, another service layer.
            std::string layerName, controlName; long max=0;
            check(bool(input>>layerName>>controlName>>max) && max>=0 && input.eof(),"invalid allocation sweep");
            const auto sessionUri=makeSessionUri(argv[1],argv[2]);
            const auto format=SdfFileFormat::FindById(TfToken("live"));
            check(bool(format),"live file format is not registered");
            // Off host: no admission or status subject of `uri` remains. A
            // failed open marks its admission retiring once its backing is
            // quiescent, so reserve(deadline) waits for that retirement
            // instead of needing its own poll loop here.
            const auto retired=[&](const std::string& uri) {
                const auto parsed=parseUri(uri);
                (void)getRuntime()->layerAdmissions->reserve(owner.endpoint,parsed.sessionId,parsed.layerName,
                    Clock::now()+std::chrono::seconds(5));
            };
            const auto noSubject=[](const std::string& uri) {
                auto status=record<OmniUsdLiveStatus>();
                check(omniUsdLiveGetLayerStatus(uri.c_str(),&status)==OMNI_USD_LIVE_NOT_FOUND,"failed layer setup left a status subject");
            };
            // The single typed failure that `source` (the file format or the
            // live resolver) reported.
            const auto codeOf=[](TfErrorMark& errors,const char* source="Live file format operation failed") {
                const std::string marker=" (result=";
                long code=-1; size_t count=0;
                for(const auto& error:errors) {
                    ++count;
                    const auto text=error.GetCommentary();
                    const auto at=text.find(marker);
                    check(text.find(source)!=std::string::npos && at!=std::string::npos,"swept open reported an unexpected error");
                    code=std::stol(text.substr(at+marker.size()));
                }
                errors.Clear();
                check(count==1,"swept open did not report exactly one error");
                return code;
            };
            // `opened`: the Read started the layer's open and reached its
            // bootstrap wait, so the open may have issued its first RPC.
            struct Outcome { std::string uri; SdfLayerRefPtr layer; bool fired=false, threw=false, ok=false, attached=false,
                             editable=false, editableBefore=false, opened=false; long code=-1, resolverCode=-1;
                             uint64_t subject=0; };
            // Host thread: a Read of a new layer, under any armed countdown,
            // which it disarms; the caller records the permission beforehand.
            const auto read=[](Outcome& outcome,TfErrorMark& errors,const auto& codeOf) {
                reachedOpenWait=false;
                outcome.ok=outcome.layer->GetFileFormat()->Read(outcome.layer.operator->(),
                    outcome.layer->GetResolvedPath().GetPathString(),false);
                outcome.opened=reachedOpenWait;
                allocationCountdown=-1; outcome.fired=std::exchange(allocationFailed,false) || outcome.fired;
                if(!outcome.ok) outcome.code=codeOf(errors);
                else check(errors.IsClean(),"attached swept layer reported an error");
                const auto attachment=attachedLiveScene(*outcome.layer);
                outcome.attached=bool(attachment); outcome.editable=outcome.layer->PermissionToEdit();
                if(attachment) outcome.subject=attachment->subjectId();
            };
            // Off host: an attached layer reaches READY.
            const auto ready=[&](const Outcome& outcome) {
                check(outcome.ok && outcome.attached && outcome.editable && outcome.subject,"swept layer did not attach");
                StatusBoundary boundary(owner.endpoint,outcome.subject,OMNI_USD_LIVE_OK,OMNI_USD_LIVE_STATE_READY);
                boundary.wait();
            };
            const auto release=[&](Outcome& outcome) {
                host.call([&] {
                    const SdfLayerHandle weak(outcome.layer);
                    outcome.layer.Reset(); check(!weak,"swept layer reference did not retire");
                });
                retired(outcome.uri);
            };
            const auto unattached=[](const Outcome& outcome) {
                check(!outcome.attached && outcome.editable==outcome.editableBefore,
                      "failed open attached the layer or changed its permission");
            };
            // An allocation failure before the open's bootstrap wait fails
            // the Read before the open starts, so it issues no RPC.
            const auto unopened=[](const Outcome& outcome) {
                check(!outcome.opened,"a Read failed by an earlier allocation failure still opened the layer");
            };
            std::map<long,unsigned> newCodes, readCodes;
            std::map<long,unsigned> absorbedCodes; // The resolver's reported cause per absorbed New; -1 for none.
            unsigned control=0, newSites=0, threw=0, refused=0, setup=0, newAbsorbed=0, unbacked=0, readSites=0, readFailed=0, readAbsorbed=0;
            bool newComplete=false, readComplete=false;
            // One character per index: F failed, A absorbed, C control (Read);
            // R refused, T threw, S setup failure, A absorbed, U absorbed under
            // a fresh identifier, C control (New).
            std::string readMap, newMap;
            SceneAttachment::setBeforeOpenWaitForTest(disarmAtOpenWait,nullptr);
            for(long index=0;index<=max && !readComplete;++index) {
                Outcome outcome; outcome.uri=makeLayerUri(sessionUri,layerName);
                host.call([&] {
                    outcome.layer=SdfLayer::New(format,outcome.uri);
                    check(bool(outcome.layer),"live SdfLayer::New returned no layer");
                    TfErrorMark errors;
                    outcome.editableBefore=outcome.layer->PermissionToEdit();
                    allocationFailed=false; allocationCountdown=LIVE_TEST_ALLOCATOR ? index : -1;
                    read(outcome,errors,codeOf);
                    if(!outcome.fired) check(outcome.ok && outcome.opened,"uninjected Read did not reach its bootstrap wait");
                });
                if(!outcome.fired) {
                    ready(outcome); release(outcome); ++control; readComplete=true; readMap+='C'; break;
                }
                ++readSites;
                if(outcome.ok) { ready(outcome); ++readAbsorbed; readMap+='A'; }
                else { unattached(outcome); unopened(outcome); ++readFailed; ++readCodes[outcome.code]; readMap+='F'; }
                release(outcome);
            }
            std::string current=layerName; // A service layer until the first throw.
            for(long index=0;index<=max && readComplete && !newComplete;++index) {
                Outcome outcome; outcome.uri=makeLayerUri(sessionUri,current);
                host.call([&] {
                    TfErrorMark errors;
                    allocationFailed=false; allocationCountdown=LIVE_TEST_ALLOCATOR ? index : -1;
                    try { outcome.layer=SdfLayer::New(format,outcome.uri); } catch(const std::bad_alloc&) { outcome.threw=true; }
                    allocationCountdown=-1; outcome.fired=std::exchange(allocationFailed,false);
                    if(outcome.threw || !outcome.layer) {
                        // The live resolver refuses the new asset with one typed error.
                        if(!outcome.threw) outcome.code=codeOf(errors,"Live resolver ");
                        check(outcome.fired && !SdfLayer::Find(outcome.uri),"failed layer construction left a layer");
                        return;
                    }
                    // A New that absorbed its allocation failure may still have
                    // reported the resolver's typed cause.
                    if(outcome.fired && !errors.IsClean()) outcome.resolverCode=codeOf(errors,"Live resolver ");
                    errors.Clear();
                    if(!outcome.fired && current!=layerName) return; // The control opens `control` instead.
                    // A fired allocation absorbed by New is not the Read's.
                    const bool fired=outcome.fired; outcome.fired=false;
                    outcome.editableBefore=outcome.layer->PermissionToEdit();
                    read(outcome,errors,codeOf);
                    outcome.fired=fired;
                });
                if(!outcome.fired) {
                    if(current!=layerName) {
                        // The fresh identifier names no service layer.
                        release(outcome);
                        outcome=Outcome{}; outcome.uri=makeLayerUri(sessionUri,controlName);
                        host.call([&] {
                            outcome.layer=SdfLayer::New(format,outcome.uri);
                            check(bool(outcome.layer),"live SdfLayer::New of the control layer failed");
                            TfErrorMark errors;
                            outcome.editableBefore=outcome.layer->PermissionToEdit();
                            read(outcome,errors,codeOf);
                        });
                    }
                    ready(outcome); release(outcome); ++control; newComplete=true; newMap+='C'; break;
                }
                ++newSites;
                if(outcome.threw) {
                    ++threw; newMap+='T'; retired(outcome.uri); noSubject(outcome.uri);
                    current="sweep-"+std::to_string(index)+".live";
                    continue;
                }
                if(!outcome.layer) {
                    ++refused; ++newCodes[outcome.code]; newMap+='R'; retired(outcome.uri); noSubject(outcome.uri);
                    continue;
                }
                if(outcome.ok) {
                    check(current==layerName,"a fresh identifier attached");
                    ready(outcome); ++newAbsorbed; newMap+='A'; ++absorbedCodes[outcome.resolverCode];
                } else if(current!=layerName && outcome.code==OMNI_USD_LIVE_NOT_FOUND) {
                    // New absorbed the failure; the open reached the service,
                    // which has no layer of the fresh identifier.
                    unattached(outcome); ++unbacked; newMap+='U'; ++absorbedCodes[outcome.resolverCode];
                } else {
                    // Setup failures precede the open's admission and first RPC.
                    unattached(outcome); unopened(outcome); noSubject(outcome.uri); ++setup; ++newCodes[outcome.code];
                    newMap+='S';
                }
                release(outcome);
                noSubject(outcome.uri);
            }
            SceneAttachment::setBeforeOpenWaitForTest(nullptr,nullptr);
            const auto codes=[](const std::map<long,unsigned>& values) {
                std::cout << '{';
                bool first=true;
                for(const auto& [code,count]:values) {
                    std::cout << (first ? "" : ",") << '"' << code << "\":" << count; first=false;
                }
                std::cout << '}';
            };
            std::cout << "{\"type\":\"alloc-sweep\",\"injection\":" << (LIVE_TEST_ALLOCATOR?"true":"false")
                      << ",\"complete\":" << (newComplete && readComplete ? "true" : "false") << ",\"control\":" << control
                      << ",\"new\":{\"sites\":" << newSites << ",\"threw\":" << threw << ",\"refused\":" << refused << ",\"setup\":" << setup
                      << ",\"absorbed\":" << newAbsorbed << ",\"unbacked\":" << unbacked << ",\"absorbed_codes\":";
            codes(absorbedCodes);
            std::cout << ",\"map\":\"" << newMap << "\",\"codes\":";
            codes(newCodes);
            std::cout << "},\"read\":{\"sites\":" << readSites << ",\"failed\":" << readFailed
                      << ",\"absorbed\":" << readAbsorbed << ",\"map\":\"" << readMap << "\",\"codes\":";
            codes(readCodes);
            std::cout << "}}" << std::endl;
        } else if (command=="compose-unopened") {
            // Off the host, on this driver thread: a direct FindOrOpen of an
            // unopened live URI, then a stage whose root layer sublayers two
            // unopened live URIs. Each open is refused before it issues an RPC,
            // and its diagnostic says to open live layers on the host first.
            std::string firstName,secondName;
            check(bool(input>>firstName>>secondName),"missing unopened layer names");
            const auto sessionUri=makeSessionUri(argv[1],argv[2]);
            const auto first=makeLayerUri(sessionUri,firstName),second=makeLayerUri(sessionUri,secondName);
            const std::string guidance="open live layers on the host before composing them";
            const auto guided=[&](const std::string& text) { return text.find(guidance)!=std::string::npos; };
            TfErrorMark errors;
            const bool directOpened=bool(SdfLayer::FindOrOpen(first));
            long directResult=-1; size_t directErrors=0; bool directGuided=false;
            for(const auto& error:errors) {
                ++directErrors;
                if(const auto* result=error.GetInfo<OmniUsdLiveResult>()) directResult=long(*result);
                directGuided=directGuided || guided(error.GetCommentary());
            }
            errors.Clear();
            size_t sublayerFailures=0,sublayerGuided=0,liveLoaded=0;
            {
                const auto root=SdfLayer::CreateAnonymous("compose-unopened.usda");
                root->SetSubLayerPaths({first,second});
                const auto stage=UsdStage::Open(root);
                check(bool(stage),"stage over unopened live sublayers did not open");
                for(const auto& error:stage->GetCompositionErrors()) {
                    if(!error || error->errorType!=PcpErrorType_InvalidSublayerPath) continue;
                    ++sublayerFailures;
                    if(guided(error->ToString())) ++sublayerGuided;
                }
                for(const auto& layer:stage->GetLayerStack())
                    if(layer->GetFileFormat()->GetFormatId()==TfToken("live")) ++liveLoaded;
            }
            errors.Clear();
            const bool registered=bool(SdfLayer::Find(first)) || bool(SdfLayer::Find(second));
            std::cout << "{\"type\":\"composed-unopened\",\"direct_opened\":" << directOpened
                      << ",\"direct_result\":" << directResult << ",\"direct_errors\":" << directErrors
                      << ",\"direct_guided\":" << directGuided << ",\"sublayer_failures\":" << sublayerFailures
                      << ",\"sublayer_guided\":" << sublayerGuided << ",\"live_loaded\":" << liveLoaded
                      << ",\"registered\":" << registered << '}' << std::endl;
        } else if (command=="open-second" || command=="drop-second") {
            if (command=="open-second") {
                std::string name; check(bool(input>>name),"missing second layer name");
                secondUri=makeLayerUri(makeSessionUri(argv[1],argv[2]),name);
                host.openSecond(secondUri);
            } else { host.dropSecond(); secondUri.clear(); }
            last=host.read(); report(command=="open-second" ? "second-opened" : "second-dropped",last);
        } else if (command=="await-standing-reads") {
            // Until the endpoint's pending bytes equal exactly `count` standing
            // read reservations. This observes the byte total only: a read may
            // still reserve a later frame, such as its caught-up header.
            unsigned count=0; check(bool(input>>count),"missing standing read count");
            const auto runtime=getRuntime(); const auto endpoint=runtime->findEndpoint(owner.endpoint);
            check(runtime && endpoint,"file-format endpoint is not registered");
            const uint64_t expected=uint64_t(count)*layerReadFrameLimit(runtime->options.values.max_update_bytes);
            auto held=[&] { std::lock_guard<std::mutex> lock(endpoint->mutex); return endpoint->pendingTransportBytes; };
            const auto until=Clock::now()+std::chrono::seconds(5);
            while (held()!=expected) {
                check(Clock::now()<until,"endpoint pending bytes did not settle to the standing reads"); std::this_thread::yield();
            }
            last=host.read(); report("standing-reads",last);
        } else if (command=="token-revoke" || command=="token-set" || command=="token-fail") {
            // Revoke the endpoint's token, install the writer's token again, or
            // fail the open refresh challenge, against the current generation.
            auto auth=record<OmniUsdLiveAuthState>();
            auto result=omniUsdLiveGetAuthState(owner.endpoint,&auth);
            uint64_t generation=0;
            if(result==OMNI_USD_LIVE_OK && command=="token-fail")
                result=omniUsdLiveFailAuthRefresh(owner.endpoint,auth.challenge_id);
            else if(result==OMNI_USD_LIVE_OK)
                result=omniUsdLiveSetAccessToken(owner.endpoint,auth.generation,
                    command=="token-set" ? "synthetic-writer" : "",0,&generation);
            std::cout << "{\"type\":\"token\",\"result\":" << result << ",\"challenge\":" << auth.challenge_id << '}'
                      << std::endl;
        } else if (command=="await-state") {
            uint32_t state=0; unsigned result=0; check(bool(input>>state>>result),"missing expected status state/result");
            { StatusBoundary boundary(owner.endpoint,subject(),static_cast<OmniUsdLiveResult>(result),state); boundary.wait(); }
            last=host.read(); report("state",last);
        } else if (command=="direct-create-prim") {
            std::string name; check(bool(input>>name) && identifier(name),"invalid temporary prim name");
            host.createRootPrim(name);
            host.call([] {}); // Fence: the flush job has run.
            std::cout << "{\"type\":\"prim-created\",\"result\":" << owner.attachment->progress().result << '}' << std::endl;
        } else if (command=="direct-remove-save") {
            std::string name; check(bool(input>>name) && identifier(name),"invalid temporary prim name");
            const auto [saved,values]=host.removeAndSave(name); last=values;
            std::cout << "{\"type\":\"remove-saved\",\"saved\":" << (saved?"true":"false") << ",\"fields\":";
            jsonFields(values); std::cout << '}' << std::endl;
        } else if (command=="direct-animate" || command=="direct-parent" || command=="direct-reverse") {
            // One host turn of structural Sdf authoring; reports the Tf errors
            // its flush posts and the layer's status after that flush.
            std::string name; int count=0;
            check(bool(input>>name) && identifier(name),"invalid structural prim name");
            if(command=="direct-animate" || command=="direct-parent") check(bool(input>>count) && count>0,"invalid count");
            std::vector<std::string> errors;
            {
                ErrorCapture capture;
                if(command=="direct-animate") host.animateRootPrim(name,count);
                else if(command=="direct-parent") host.createParent(name,count);
                else host.reverseChildren(name);
                host.call([] {}); // Fence: the flush job has run.
                errors=capture.errors();
            }
            auto status=record<OmniUsdLiveStatus>();
            check(omniUsdLiveGetLayerStatus(uri.c_str(),&status)==OMNI_USD_LIVE_OK,"layer status query failed");
            std::cout << "{\"type\":\"structure\",\"result\":" << owner.attachment->progress().result
                      << ",\"state\":" << status.state << ",\"status_result\":" << status.result
                      << ",\"pending\":" << status.pending_write_count << ",\"buffered\":" << status.buffered_edit_count
                      << ",\"exists\":" << (host.call([&] { return bool(host.hasPrim(name)); })?"true":"false")
                      << ",\"errors\":[";
            for (size_t i=0;i<errors.size();++i) std::cout << (i?",":"") << jsonString(errors[i]);
            std::cout << "]}" << std::endl;
        } else if (command=="export-live-fields") {
            std::string stem; int count=0;
            check(bool(input>>stem>>count) && identifier(stem) && count>=0,"invalid field export");
            const auto [exported,errors]=host.exportFields(makeLayerUri(makeSessionUri(argv[1],argv[2]),stem+".live"),count);
            std::cout << "{\"type\":\"fields-export\",\"succeeded\":" << (exported?"true":"false") << ",\"errors\":[";
            for (size_t i=0;i<errors.size();++i) std::cout << (i?",":"") << jsonString(errors[i]);
            std::cout << "]}" << std::endl;
        } else if (command=="replacements") {
            std::cout << "{\"type\":\"replacements\",\"count\":" << host.replacements()
                      << ",\"changes\":" << host.changes() << '}' << std::endl;
        } else if (command=="stage-open") {
            std::cout << "{\"type\":\"stage\",\"layers\":" << host.openStage() << '}' << std::endl;
        } else if (command=="stage-notices") {
            const auto taken=host.takeStageNotices();
            const auto paths=[](const std::set<SdfPath>& values) {
                std::cout << '[';
                bool first=true;
                for(const auto& path:values) { std::cout << (first?"":",") << '"' << path.GetString() << '"'; first=false; }
                std::cout << ']';
            };
            std::cout << "{\"type\":\"stage-notices\",\"notices\":" << taken.notices << ",\"resynced\":";
            paths(taken.resynced); std::cout << ",\"info\":"; paths(taken.info); std::cout << '}' << std::endl;
        } else if (command=="stage-prim") {
            std::string path; check(bool(input>>path) && SdfPath::IsValidPathString(path),"invalid stage prim path");
            std::cout << "{\"type\":\"stage-prim\",\"exists\":" << (host.stageHasPrim(path)?"true":"false") << '}'
                      << std::endl;
        } else if (command=="notice-author" || command=="direct-set-offhost") {
            std::string key; int value=0; check(bool(input>>key>>value) && identifier(key),"invalid notice or off-host field edit");
            if(command=="notice-author") host.armNoticeAuthor(key,value);
            else check(host.directSetOffHost(key,value)==OMNI_USD_LIVE_OK,"off-host field edit was refused");
            host.call([] {}); // Fence: any job the edit queued has run.
            last=host.read(); report(command=="notice-author" ? "notice-armed" : "offhost",last);
        } else if (command=="direct-set-offhost-try") {
            std::string key; int value=0; check(bool(input>>key>>value) && identifier(key),"invalid off-host field edit");
            const auto refused=host.directSetOffHost(key,value);
            host.call([] {}); // Fence: any job the edit queued has run.
            last=host.read();
            std::cout << "{\"type\":\"offhost-try\",\"refused\":" << refused << ",\"fields\":";
            jsonFields(last); std::cout << '}' << std::endl;
        } else if (command=="export-usd") {
            std::string path; check(bool(input>>path),"missing USD export path");
            const auto exportedFields=host.exportLayer(path);
            std::cout << "{\"type\":\"exported\",\"fields\":"; jsonFields(exportedFields); std::cout << '}' << std::endl;
        } else if (command=="replace-content") {
            std::string path; check(bool(input>>path),"missing replacement import source path");
            const auto before=host.read(); const auto replaced=host.replacements();
            const auto outcome=host.replaceContent(path);
            host.call([] {}); // Fence: any job the attempts queued has run.
            const bool unchanged=host.read()==before;
            std::cout << "{\"type\":\"replace-content\",\"transfer\":" << outcome.transfer
                      << ",\"typed\":" << (outcome.typed?"true":"false") << ",\"import\":" << (outcome.imported?"true":"false")
                      << ",\"import_result\":" << outcome.importResult
                      << ",\"import_string\":" << (outcome.importedString?"true":"false")
                      << ",\"unchanged\":" << (unchanged?"true":"false") << ",\"replacements\":" << host.replacements()-replaced
                      << ",\"result\":" << owner.attachment->progress().result << '}' << std::endl;
        } else if (command=="clear") {
            Fields values; std::string key; int value;
            while(input>>key>>value) { check(identifier(key),"invalid clear field name"); values[key]=value; }
            check(input.eof(),"invalid clear field edits");
            host.clearLayer(values);
            host.call([] {}); // Fence: any job the clear queued has run.
            last=host.read(); report("cleared",last);
        } else if (command=="direct-set-unencodable" || command=="direct-set-payload-offset") {
            // Reports the Tf errors the failing flush posts on the host.
            std::string key; check(bool(input>>key) && identifier(key),"invalid unencodable field name");
            std::vector<std::string> errors;
            {
                ErrorCapture capture;
                if (command=="direct-set-unencodable") host.call([&] { host.authorUnencodable(key); });
                else host.call([&] { host.authorPayloadOffset(key); });
                host.call([] {}); // Fence: the flush job has run.
                errors=capture.errors();
            }
            last=host.read();
            std::cout << "{\"type\":\"unencodable\",\"identifier\":" << jsonString(host.layerIdentifier())
                      << ",\"result\":" << owner.attachment->progress().result << ",\"fields\":";
            jsonFields(last); std::cout << ",\"errors\":[";
            for (size_t i=0;i<errors.size();++i) std::cout << (i?",":"") << jsonString(errors[i]);
            std::cout << "]}" << std::endl;
        } else if (command=="save") {
            unsigned expected=0; check(bool(input>>expected),"missing expected save result");
            host.save(static_cast<OmniUsdLiveResult>(expected)); report("saved",host.read());
        } else if (command=="save-offhost") {
            // The expected result defaults to BUSY.
            unsigned expected=0; if(!(input>>expected)) expected=OMNI_USD_LIVE_BUSY;
            host.saveOffHost(static_cast<OmniUsdLiveResult>(expected));
            host.call([] {}); // Fence: any job the save queued has run.
            report("saved-offhost",host.read());
        } else if (command=="save-held-retirement") {
            // Author one edit, save it off the host, then hold the actor worker
            // right after it completes the edit's drain tickets and run a
            // forced Sdf Save on the host while the worker is still held.
            std::string key; int value=0; check(bool(input>>key>>value) && identifier(key),"invalid held-retirement edit");
            struct RetirementHold {
                std::mutex mutex; std::condition_variable changed; bool held=false, released=false;
                static void hook(void* context) noexcept {
                    if(!context) return; // A worker racing the disarm may load a cleared context.
                    auto& self=*static_cast<RetirementHold*>(context);
                    std::unique_lock<std::mutex> lock(self.mutex);
                    if(self.held) return;
                    self.held=true; self.changed.notify_all();
                    // Failure-only watchdog: the driver releases the worker explicitly.
                    if(!self.changed.wait_for(lock,std::chrono::seconds(30),[&] { return self.released; })) std::abort();
                }
            };
            // Static: a worker that loaded the hook before disarming may still read it.
            static RetirementHold hold;
            { std::lock_guard<std::mutex> lock(hold.mutex); hold.held=false; hold.released=false; }
            struct Disarm { ~Disarm() {
                LayerActor::setAfterDrainRetiredForTest(nullptr,nullptr);
                std::lock_guard<std::mutex> lock(hold.mutex); hold.released=true; hold.changed.notify_all();
            } };
            LayerActor::setAfterDrainRetiredForTest(RetirementHold::hook,&hold);
            bool onHost=false; OmniUsdLiveResult offHost=OMNI_USD_LIVE_INTERNAL_ERROR;
            {
                Disarm disarm;
                host.edit(owner.attachment,key,value);
                offHost=host.saveAttachedOffHost(5000);
                {
                    std::unique_lock<std::mutex> lock(hold.mutex);
                    check(hold.changed.wait_for(lock,std::chrono::seconds(10),[&] { return hold.held; }),
                          "actor worker did not reach the retired-drain seam");
                }
                onHost=host.forcedSaveOnHost();
            }
            last=host.read();
            std::cout << "{\"type\":\"held-retirement-save\",\"offhost\":" << offHost
                      << ",\"onhost\":" << (onHost?"true":"false") << ",\"fields\":";
            jsonFields(last); std::cout << '}' << std::endl;
        } else if (command=="save-wait") {
            uint32_t timeout=0; check(bool(input>>timeout),"missing save timeout");
            const auto result=owner.attachment->save(timeout);
            std::cout << "{\"type\":\"save-result\",\"result\":" << result << '}' << std::endl;
        } else if (command=="diagnostics") {
            unsigned expected=0; check(bool(input>>expected),"missing expected diagnostic status result");
            { StatusBoundary boundary(owner.endpoint,subject(),static_cast<OmniUsdLiveResult>(expected)); boundary.wait(); }
            auto config=record<OmniUsdLiveDiagnosticExportOptions>(); config.max_records=128; config.max_output_bytes=65536;
            std::vector<char> output(config.max_output_bytes); uint64_t size=output.size();
            check(omniUsdLiveCopyDiagnostics(owner.endpoint,&config,output.data(),&size)==OMNI_USD_LIVE_OK &&
                  size && size<=output.size(),"bounded file-format diagnostic snapshot failed");
            std::cout << "{\"type\":\"diagnostics\",\"subject\":" << subject() << ",\"diagnostics\":" << output.data() << '}' << std::endl;
        } else if (command=="direct-turns" || command=="direct-distinct-turns") {
            unsigned turns=0,fields=0; std::string prefix,mode;
            check(bool(input>>turns>>fields>>prefix) && identifier(prefix+"0"),"invalid direct turns");
            const bool numbered=command=="direct-distinct-turns";
            if(!numbered) input>>mode;
            check(mode.empty() || mode=="distinct" || mode=="mixed","invalid direct turns mode");
            unsigned repeated=0;
            if(mode=="mixed") check(bool(input>>repeated) && repeated<turns,"invalid repeated prefix");
            std::cout << "{\"type\":\"turns\"," << turnsJson(host.directTurns(turns,fields,prefix,!mode.empty(),repeated,numbered))
                      << '}' << std::endl;
        } else if (command=="second-turns") {
            unsigned turns=0,fields=0; std::string prefix;
            check(bool(input>>turns>>fields>>prefix) && identifier(prefix+"0"),"invalid second turns");
            std::cout << "{\"type\":\"turns\"," << turnsJson(host.secondTurns(turns,fields,prefix)) << '}' << std::endl;
        } else if (command=="shared-budget-turns") {
            // In one host job: turns on this layer, then one turn on the second
            // layer, then the second layer's public status.
            unsigned turns=0,fields=0,secondFields=0; std::string prefix,secondPrefix;
            check(bool(input>>turns>>fields>>prefix>>secondFields>>secondPrefix) && identifier(prefix+"0") &&
                  identifier(secondPrefix+"0"),"invalid shared budget turns");
            check(!secondUri.empty(),"no second live layer");
            const auto outcome=host.sharedBudgetTurns(turns,fields,prefix,secondFields,secondPrefix,secondUri);
            std::cout << "{\"type\":\"shared-budget\",\"first\":{" << turnsJson(outcome.first) << "},\"second\":{"
                      << turnsJson(outcome.second) << ",\"blocked\":" << unsigned(outcome.secondBlocked)
                      << ",\"changed\":" << (outcome.secondChanged ? "true" : "false")
                      << ",\"state\":" << outcome.secondState << ",\"status_result\":" << unsigned(outcome.secondResult)
                      << "}}" << std::endl;
        } else if (command=="await-second-ready") {
            // Until the second layer's buffered count holds no deferral cause and
            // no records, then until its status reports READY with OK.
            check(!secondUri.empty(),"no second live layer");
            const auto count=host.secondBuffered();
            const auto until=Clock::now()+std::chrono::seconds(10);
            while (count->blocked()!=OMNI_USD_LIVE_OK || count->load()!=0) {
                check(Clock::now()<until,"second layer deferral cause did not clear"); std::this_thread::yield();
            }
            auto status=record<OmniUsdLiveStatus>();
            for (;;) {
                check(omniUsdLiveGetLayerStatus(secondUri.c_str(),&status)==OMNI_USD_LIVE_OK,"second layer status query failed");
                if (status.state==OMNI_USD_LIVE_STATE_READY && status.result==OMNI_USD_LIVE_OK && status.buffered_edit_count==0) break;
                check(Clock::now()<until,"second layer status did not return to READY"); std::this_thread::yield();
            }
            std::cout << "{\"type\":\"second-status\",\"state\":" << status.state << ",\"status_result\":" << status.result
                      << ",\"buffered\":" << status.buffered_edit_count << '}' << std::endl;
        } else if (command=="busy-frames") {
            unsigned frames=0; std::string prefix;
            check(bool(input>>frames>>prefix) && frames>0 && identifier(prefix+"0"),"invalid busy frames");
            const auto admitted=host.busyFrames(frames,prefix,owner.attachment);
            std::cout << "{\"type\":\"frames\",\"admitted\":" << admitted << '}' << std::endl;
        } else if (command=="issued") {
            // Waits until no admitted write waits to be issued: the worker has
            // issued every one, so the next flush admits if the window has room.
            const auto until=std::chrono::steady_clock::now()+std::chrono::seconds(10);
            while (owner.attachment->progress().queued) {
                check(std::chrono::steady_clock::now()<until,"admitted writes were not issued");
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            std::cout << "{\"type\":\"issued\"}" << std::endl;
        } else if (command=="await-window-halved") {
            // Waits until the layer's writes have halved the issuance window
            // `count` times, as a refusal that applied nothing does.
            uint64_t count=0; check(bool(input>>count),"missing window halving count");
            const auto until=std::chrono::steady_clock::now()+std::chrono::seconds(10);
            while (getRuntime()->diagnostics->snapshot(owner.endpoint,0).metrics.layerWrites.windowHalved<count) {
                check(std::chrono::steady_clock::now()<until,"the issuance window was not halved");
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            std::cout << "{\"type\":\"window-halved\"}" << std::endl;
        } else if (command=="await-resend-hold") {
            // Waits until a worker pass ends with issuance held for a re-sent
            // unchained write's first attempt and no write attempt in flight.
            const auto until=std::chrono::steady_clock::now()+std::chrono::seconds(10);
            while (!owner.attachment->resendHeldForTest()) {
                check(std::chrono::steady_clock::now()<until,"no re-send hold was reached");
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            std::cout << "{\"type\":\"resend-hold\",\"queued\":" << owner.attachment->progress().queued << '}'
                      << std::endl;
        } else if (command=="reopen") {
            // Closes the layer's actor and releases the layer, then opens it
            // again: a new actor under the same endpoint and writer ID.
            check(owner.attachment->close(5000)==OMNI_USD_LIVE_OK,"native format close did not quiesce");
            host.drop(); owner.attachment.reset();
            owner.attachment=host.open(uri);
            const auto reopened=owner.attachment->snapshot()->applied;
            progressEvents().until([&] {
                const auto progress=owner.attachment->progress();
                check(progress.result==OMNI_USD_LIVE_OK,"reopened layer initial application did not retire");
                return progress.applied>=reopened;
            },"reopened layer initial application did not retire");
            last=host.read(); report("reopened",last);
        } else if (command=="expire-resend-hold") {
            owner.attachment->expireResendHoldForTest();
            std::cout << "{\"type\":\"resend-hold-expired\"}" << std::endl;
        } else if (command=="window") {
            // The layer's chained-write issuance window, apart from the one
            // write a broken chain's first re-send issues under.
            std::cout << "{\"type\":\"window\",\"window\":" << owner.attachment->chainWindowForTest() << '}'
                      << std::endl;
        } else if (command=="operations") {
            // The layer's retained local writes in chain order: each one's
            // local ordinal, attempts that reached transport and operation ID.
            std::cout << "{\"type\":\"operations\",\"writes\":[";
            bool first=true;
            for (const auto& work : owner.attachment->retained()) {
                if (!work.localOrdinal) continue;
                const auto& effect=*std::static_pointer_cast<const LayerActor::Effect>(work.owner);
                std::cout << (first?"":",") << "{\"ordinal\":" << work.localOrdinal << ",\"attempts\":" << work.attempts
                          << ",\"operation_id\":\"" << effect.operationId << "\"}";
                first=false;
            }
            std::cout << "]}" << std::endl;
        } else if (command=="settle-together") {
            // Holds the next actor worker pass that finds a completed write
            // attempt until `count` of its attempts have completed, so that one
            // pass settles them in chain order.
            size_t count=0; check(bool(input>>count) && count>0,"invalid settle count");
            struct SettleTogether {
                std::atomic<size_t> target{0};
                std::atomic<std::chrono::steady_clock::time_point> until{};
                static bool hook(void* context, size_t completed) noexcept {
                    if(!context) return false; // A worker racing the disarm may load a cleared context.
                    auto& self=*static_cast<SettleTogether*>(context);
                    const auto target=self.target.load();
                    if(!target) return false;
                    if(completed>=target) { self.target=0; return false; }
                    // Failure-only watchdog: the test scripts the completions this pass waits for.
                    if(std::chrono::steady_clock::now()>self.until.load()) std::abort();
                    std::this_thread::yield();
                    return true;
                }
            };
            // Static: a worker that loaded the hook may still read it.
            static SettleTogether settle;
            settle.until=std::chrono::steady_clock::now()+std::chrono::seconds(30);
            settle.target=count;
            LayerActor::setBeforeSettleWritesForTest(SettleTogether::hook,&settle);
            std::cout << "{\"type\":\"settle-together\"}" << std::endl;
        } else if (command=="compiles") {
            host.call([] {}); // Fence: host jobs queued before this command, flushes included, have run.
            std::cout << "{\"type\":\"compiles\",\"count\":" << fieldEditCompilationsForTest() << '}' << std::endl;
        } else if (command=="release-held") { host.releaseHeld(); report("released",host.read()); }
        else if (command=="reject-next") { host.rejectNext(); report("armed",host.read()); }
        else if (command=="refuse-dispatch") {
            // `refuse-dispatch 1` refuses every host job dispatch with
            // RESOURCE_EXHAUSTED, as a full metering dispatcher does, until
            // `refuse-dispatch 0`.
            unsigned refuse=0; check(bool(input>>refuse) && refuse<=1,"invalid dispatch refusal");
            host.refuseDispatch(refuse!=0); report("armed",host.read());
        }
        else if (command=="capture-host") {
            // The host layer, journal and publication progress of a FAILED
            // layer, read after every queued host job has run.
            host.call([] {});
            const auto hostFields=host.read();
            const auto progress=owner.attachment->progress(); const auto journal=owner.attachment->snapshot();
            const auto publication=owner.attachment->publicationProgress();
            std::cout << "{\"type\":\"host-state\",\"result\":" << progress.result
                      << ",\"applied\":" << progress.applied
                      << ",\"journal_applied\":" << journal->applied
                      << ",\"last_attempted_revision\":" << publication.lastAttemptedRevision
                      << ",\"last_completed_revision\":" << publication.lastCompletedRevision
                      << ",\"publication_result\":" << publication.result << ",\"host_fields\":";
            jsonFields(hostFields); std::cout << ",\"journal_fields\":";
            jsonFields(fields(*journal->visible)); std::cout << '}' << std::endl;
        }
        else if (command=="arm") { host.arm(); report("armed",host.read()); }
        else if (command=="held") {
            host.waitHeld(); check(omniUsdLiveWaitForPendingUpdates(0)==OMNI_USD_LIVE_TIMEOUT,"held format publication drained");
            report("held",host.read());
        } else if (command=="direct-set-probe") {
            last=host.read();
            const auto saves=host.directSetProbe(owner.attachment); dropped=true;
            std::cout << "{\"type\":\"probe\",\"result\":" << owner.attachment->progress().result
                      << ",\"admitted\":" << owner.attachment->snapshot()->localAdmitted << ",\"saves\":[";
            for(size_t i=0;i<saves.size();++i) std::cout << (i ? "," : "") << saves[i];
            std::cout << "]}" << std::endl;
        } else if (command=="direct-set-drop") {
            std::string key; int value=0; check(bool(input>>key>>value) && identifier(key),"invalid direct drop field edit");
            last=host.read();
            const auto before=owner.attachment->snapshot();
            const auto id=subject();
            ClosedBoundary closedStatus(owner.endpoint,id);
            host.directSetDrop(owner.attachment,key,value); dropped=true;
            host.call([] {}); // FIFO barrier: the queued flush job has run.
            check(owner.attachment->snapshot()->localAdmitted==before->localAdmitted,
                  "flush job admitted an edit from a released layer");
            check(owner.attachment->close(5000)==OMNI_USD_LIVE_OK,"released layer attachment did not retire");
            const auto closed=closedStatus.wait();
            auto config=record<OmniUsdLiveDiagnosticExportOptions>(); config.max_records=128; config.max_output_bytes=65536;
            std::vector<char> output(config.max_output_bytes); uint64_t size=output.size();
            check(omniUsdLiveCopyDiagnostics(owner.endpoint,&config,output.data(),&size)==OMNI_USD_LIVE_OK &&
                  size && size<=output.size(),"bounded released-layer diagnostic snapshot failed");
            const auto journal=owner.attachment->snapshot();
            std::cout << "{\"type\":\"dropped\",\"result\":" << owner.attachment->progress().result
                      << ",\"dropped\":true,\"pending_ids\":[";
            for (size_t i=0;i<journal->pending.size();++i) std::cout << (i?",":"") << journal->pending[i].id;
            std::cout << "],\"fields\":";
            jsonFields(last);
            std::cout << ",\"subject\":" << id << ",\"closed\":{\"state\":" << closed.state
                      << ",\"result\":" << closed.result << ",\"incident\":" << closed.incident_id
                      << ",\"buffered\":" << closed.buffered_edit_count
                      << ",\"unconfirmed\":" << closed.has_unconfirmed_edits
                      << "},\"diagnostics\":" << output.data() << '}' << std::endl;
        } else if (command=="drop-queued") {
            host.waitHeld(); last=host.read();
            const auto before=owner.attachment->snapshot(); const auto applied=owner.attachment->progress().applied;
            host.drop(); dropped=true; host.releaseHeld();
            check(owner.attachment->close(5000)==OMNI_USD_LIVE_OK,"last-layer queued publication did not retire");
            check(owner.attachment->snapshot()==before && owner.attachment->progress().applied==applied,
                  "detached layer queued job published or advanced application");
            check(owner.attachment->progress().retained>0,"detached layer discarded queued peer receipt");
            report("dropped",last);
        } else if (command=="status" || command=="await-buffered") {
            uint64_t expected=0;
            if (command=="await-buffered") {
                check(bool(input>>expected),"missing expected buffered edit count");
                BufferedBoundary boundary(owner.endpoint,subject(),expected); boundary.wait();
            }
            auto status=record<OmniUsdLiveStatus>();
            check(omniUsdLiveGetLayerStatus(uri.c_str(),&status)==OMNI_USD_LIVE_OK,"layer status query failed");
            std::cout << "{\"type\":\"status\",\"state\":" << status.state << ",\"status_result\":" << status.result
                      << ",\"grpc_present\":" << status.grpc_status_present << ",\"grpc\":" << status.grpc_status
                      << ",\"operation\":" << status.operation
                      << ",\"pending\":" << status.pending_write_count << ",\"buffered\":" << status.buffered_edit_count
                      << ",\"unconfirmed\":" << status.has_unconfirmed_edits << ",\"revision\":" << status.revision
                      << ",\"read_sequence\":" << status.read_sequence
                      << ",\"applied_sequence\":" << status.applied_sequence << ",\"fields\":";
            jsonFields(host.read()); std::cout << '}' << std::endl;
        } else if (command=="await-sequences") {
            // Polls the layer status until it reports exactly these read and
            // applied sequences; they refresh without a status notification.
            uint64_t read=0, applied=0; check(bool(input>>read>>applied),"missing expected status sequences");
            auto status=record<OmniUsdLiveStatus>();
            const auto deadline=Clock::now()+std::chrono::seconds(10);
            for (;;) {
                check(omniUsdLiveGetLayerStatus(uri.c_str(),&status)==OMNI_USD_LIVE_OK,"layer status query failed");
                if (status.read_sequence==read && status.applied_sequence==applied) break;
                check(Clock::now()<deadline,"layer status did not reach the expected sequences");
                std::this_thread::yield();
            }
            std::cout << "{\"type\":\"sequences\",\"state\":" << status.state << ",\"status_result\":" << status.result
                      << ",\"revision\":" << status.revision << ",\"read_sequence\":" << status.read_sequence
                      << ",\"applied_sequence\":" << status.applied_sequence << '}' << std::endl;
        } else if (command=="status-prefix") {
            // A caller built against the 1.0 layout passes the shorter size and receives only that prefix.
            std::array<unsigned char,sizeof(OmniUsdLiveStatus)+8> buffer; buffer.fill(0xAB);
            const uint32_t size=offsetof(OmniUsdLiveStatus,buffered_edit_count), abi=OMNI_USD_LIVE_ABI_VERSION;
            std::memcpy(buffer.data(),&size,sizeof(size)); std::memcpy(buffer.data()+sizeof(size),&abi,sizeof(abi));
            const auto result=omniUsdLiveGetLayerStatus(uri.c_str(),reinterpret_cast<OmniUsdLiveStatus*>(buffer.data()));
            bool untouched=true; for(size_t i=size;i<buffer.size();++i) untouched&=buffer[i]==0xAB;
            std::cout << "{\"type\":\"status-prefix\",\"result\":" << result << ",\"untouched\":" << (untouched?"true":"false") << '}' << std::endl;
        } else if (command=="state" || command=="reload" || command=="wait" || command=="failed") {
            if (command=="failed") {
                progressEvents().until([&] { return owner.attachment->progress().result!=OMNI_USD_LIVE_OK; },
                                       "native format failure did not arrive");
            }
            if (command=="reload") host.reload(owner.attachment);
            if (command=="wait") {
                uint64_t sequence=0; check(bool(input>>sequence),"missing native format wait sequence");
                progressEvents().until([&] {
                    const auto progress=owner.attachment->progress();
                    check(progress.result==OMNI_USD_LIVE_OK,"native format application did not settle");
                    return progress.applied>=sequence;
                },"native format application did not settle");
                check(omniUsdLiveWaitForPendingUpdates(1000)==OMNI_USD_LIVE_OK,"native format applied state did not drain");
            }
            last=host.read(); report("state",last);
        } else if (command=="export-live-deadline" || command=="create-new-deadline") {
            std::string stem; int counter=0;
            check(bool(input>>stem) && identifier(stem),"invalid created layer name");
            if(command=="export-live-deadline") check(bool(input>>counter),"missing export seed counter");
            const auto target=makeLayerUri(makeSessionUri(argv[1],argv[2]),stem+".live");
            const auto outcome=command=="export-live-deadline" ? host.exportCrossingAtIdentifier(target,counter)
                                                                 : host.createNewCrossingAtDispatch(target);
            std::cout << "{\"type\":\"live-creation\",\"succeeded\":" << (outcome.succeeded?"true":"false")
                      << ",\"crossed\":" << (outcome.crossed?"true":"false")
                      << ",\"diagnosed\":" << (outcome.diagnosed?"true":"false")
                      << ",\"alive\":" << (outcome.alive?"true":"false")
                      << ",\"editable\":" << (outcome.editable?"true":"false")
                      << ",\"host_dispatches\":" << outcome.hostDispatches << ",\"fields\":";
            jsonFields(outcome.fields); std::cout << '}' << std::endl;
        } else if (command=="export-live-unreferenced") {
            std::string stem; int counter=0;
            check(bool(input>>stem>>counter) && identifier(stem),"invalid unreferenced export");
            const auto outcome=host.exportUnreferenced(makeLayerUri(makeSessionUri(argv[1],argv[2]),stem+".live"),counter);
            std::cout << "{\"type\":\"unreferenced-export\",\"succeeded\":" << (outcome.succeeded?"true":"false")
                      << ",\"adopted\":" << (outcome.adopted?"true":"false")
                      << ",\"alive_at_adoption\":" << (outcome.aliveAtAdoption?"true":"false")
                      << ",\"listed_at_adoption\":" << (outcome.listedAtAdoption?"true":"false")
                      << ",\"retained_after\":" << (outcome.retainedAfter?"true":"false") << '}' << std::endl;
        } else if (command=="await-layer") {
            // Until the named layer's status reports STATE and RESULT;
            // registration delivers the current snapshot first.
            std::string stem; uint32_t state=0; unsigned result=0;
            check(bool(input>>stem>>state>>result) && identifier(stem),"invalid awaited layer state");
            const auto target=makeLayerUri(makeSessionUri(argv[1],argv[2]),stem+".live");
            auto status=record<OmniUsdLiveStatus>();
            check(omniUsdLiveGetLayerStatus(target.c_str(),&status)==OMNI_USD_LIVE_OK && status.subject_id,
                  "missing created layer subject");
            { StatusBoundary boundary(owner.endpoint,status.subject_id,static_cast<OmniUsdLiveResult>(result),state); boundary.wait(); }
            check(omniUsdLiveGetLayerStatus(target.c_str(),&status)==OMNI_USD_LIVE_OK,"created layer status query failed");
            std::cout << "{\"type\":\"layer-state\",\"state\":" << status.state << ",\"status_result\":" << status.result
                      << ",\"buffered\":" << status.buffered_edit_count << '}' << std::endl;
        } else if (command=="retire-creations") {
            std::cout << "{\"type\":\"retired\",\"retired\":" << host.retireCreations() << '}' << std::endl;
        } else if (command=="retained-records") {
            std::cout << "{\"type\":\"retained\",\"records\":" << host.retainedRecords() << '}' << std::endl;
        } else if (command=="close") {
            check(owner.attachment->close(5000)==OMNI_USD_LIVE_OK,"native format close did not quiesce");
            if (!dropped) { last=host.read(); host.drop(); dropped=true; }
            // Preserve bounded data for the final response, then retire every
            // endpoint owner before advertising successful process completion.
            report("closed",last);
            return 0; // Cleanup performs endpoint/shutdown verification.
        } else throw std::runtime_error("unknown native file-format command");
        check(!(input>>extra),"unexpected native file-format argument");
    }
    throw std::runtime_error("native file-format driver input ended before close");
}
