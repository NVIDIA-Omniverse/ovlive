// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "SceneAttachment.h"
#include "SceneCreation.h"
#include "UsdLayerCreation.h"
#include "UsdCreationRegistry.h"
#include "LiveSdfFileFormat.h"
#include "BoundLayer.h"
#include "UsdCreationWorkflow.h"
#include "HostSceneProjection.h"
#include "LiveDrain.h"
#include "LiveUri.h"
#include "host_admission.h"
#include "graph_value_json.h"
#include "../support/test_tls.h"
#include <pxr/base/plug/registry.h>
#include <pxr/base/tf/notice.h>
#include <pxr/base/tf/errorMark.h>
#include <pxr/usd/sdf/changeBlock.h>
#include <pxr/usd/sdf/copyUtils.h>
#include <pxr/usd/sdf/attributeSpec.h>
#include <pxr/usd/sdf/primSpec.h>
#include <pxr/usd/sdf/relationshipSpec.h>
#include <pxr/usd/sdf/listOp.h>
#include <pxr/usd/sdf/variantSetSpec.h>
#include <pxr/usd/sdf/variantSpec.h>
#include <pxr/usd/sdf/data.h>
#include <pxr/usd/sdf/fileFormat.h>
#include <pxr/usd/sdf/layer.h>
#include <pxr/usd/sdf/namespaceEdit.h>
#include <pxr/usd/sdf/notice.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cmath>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <cstdlib>
#include <deque>
#include <future>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <utility>

PXR_NAMESPACE_USING_DIRECTIVE
using namespace omni::usd::live;
using namespace omni::usd::live::wire;
namespace {
using Clock = std::chrono::steady_clock;
using Fields = std::map<std::string,int>;
void check(bool yes, const char* message) { if (!yes) throw std::runtime_error(message); }
template<class T> T record() { T result{}; result.struct_size = sizeof(result); result.abi_version = OMNI_USD_LIVE_ABI_VERSION; return result; }
bool identifier(const std::string& value) {
    if (value.empty()) return false;
    auto letter = [](char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_'; };
    if (!letter(value[0])) return false;
    for (char c : value) if (!letter(c) && !(c >= '0' && c <= '9')) return false;
    return true;
}
bool terminal(const LayerProgress::Snapshot& state) {
    return state.gate == LayerProgress::Gate::Failed;
}
Fields fields(const WireScenePtr& scene) {
    check(bool(scene),"missing immutable scene owner");
    check(scene->sectionCount() == 1 && scene->sectionForPath(SdfPath::AbsoluteRootPath()) == 1,
          "reference driver expects exactly the root section");
    Fields result;
    for (const auto& name : scene->data()->List(SdfPath::AbsoluteRootPath())) {
        check(identifier(name.GetString()),"unsafe JSON field identifier");
        const VtValue value = scene->data()->Get(SdfPath::AbsoluteRootPath(),name);
        check(value.IsHolding<int>(),"non-int root field outside driver oracle");
        check(result.emplace(name.GetString(),value.Get<int>()).second,"duplicate Sdf root field");
    }
    return result;
}
struct Saved {
    SceneJournal::SnapshotPtr owner;
    Fields authoritative, visible;
    std::vector<uint64_t> ids;
    std::vector<std::string> bytes;
    std::vector<FieldEditIntentPtr> authored;
    size_t authoredCount = 0;
    size_t pendingBytes;
    uint64_t applied;
    explicit Saved(SceneJournal::SnapshotPtr value) : owner(std::move(value)),
        authoritative(fields(owner->authoritative)), visible(fields(owner->visible)),
        pendingBytes(owner->pendingBytes), applied(owner->applied) {
        size_t sum = 0; uint64_t previous = 0;
        for (const auto& pending : owner->pending) {
            check(pending.id > previous && bool(pending.intent),"invalid pending ID/owner");
            previous = pending.id;
            check(pending.intent->retainedBytes <= std::numeric_limits<size_t>::max()-sum,"pending byte overflow");
            sum += pending.intent->retainedBytes;
            ids.push_back(pending.id); bytes.push_back(pending.intent->bytes);
            authored.push_back(pending.intent->authored);
            if (pending.intent->authored) {
                const auto& source = pending.intent->authored;
                check(bool(source->operations()),"accepted captured log lost its original operations");
                check(!source->operations()->empty() && source->bytes() == pending.intent->bytes,
                      "accepted captured log lost operations or changed compiled bytes");
                check(pending.intent->retainedBytes > pending.intent->bytes.size(),"authored retention not accounted");
                ++authoredCount;
            }
        }
        check(sum == pendingBytes,"incoherent pending byte accounting");
    }
    void verify() const {
        check(fields(owner->authoritative) == authoritative && fields(owner->visible) == visible,
              "retained immutable scene changed");
        check(owner->applied == applied && owner->pendingBytes == pendingBytes && owner->pending.size() == ids.size(),
              "retained snapshot metadata changed");
        for (size_t i = 0; i < ids.size(); ++i)
            check(owner->pending[i].id == ids[i] && owner->pending[i].intent && owner->pending[i].intent->bytes == bytes[i] &&
                  owner->pending[i].intent->authored == authored[i],
                  "retained intent changed");
    }
};
// All failures in the background reader are observed by command/wait paths.
// Retain multiple earlier owners while current transport publishes new scenes.
class Reader {
    std::atomic<bool> stop_{false};
    std::atomic<uint64_t> passes_{0};
    std::promise<void> completed_;
    std::future<void> done_ = completed_.get_future();
    std::thread thread_;
public:
    explicit Reader(std::shared_ptr<SceneAttachment> attachment) : thread_([this,attachment=std::move(attachment)] {
        try {
            std::deque<Saved> saved;
            while (!stop_.load()) {
                auto current = attachment->snapshot();
                if (saved.empty() || saved.back().owner != current) {
                    saved.emplace_back(std::move(current));
                    if (saved.size() > 16) saved.pop_front();
                }
                for (const auto& item : saved) item.verify();
                ++passes_;
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            completed_.set_value();
        } catch (...) { completed_.set_exception(std::current_exception()); }
    }) {}
    void healthy() {
        if (done_.valid() && done_.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
            done_.get();
            check(stop_.load(),"reader stopped unexpectedly");
        }
    }
    void observe() {
        const auto before = passes_.load();
        const auto until = Clock::now()+std::chrono::seconds(3);
        while (passes_.load() == before) {
            healthy(); check(Clock::now() < until,"snapshot reader observation timed out");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        healthy();
    }
    void finish() {
        stop_ = true;
        if (thread_.joinable()) {
            if (done_.valid() && done_.wait_for(std::chrono::seconds(3)) != std::future_status::ready) {
                std::cerr << "snapshot reader failed to stop" << std::endl; std::_Exit(2);
            }
            thread_.join();
        }
        if (done_.valid()) done_.get();
    }
    ~Reader() { try { finish(); } catch (...) { std::cerr << "snapshot reader cleanup failed" << std::endl; std::_Exit(2); } }
};
void jsonFields(const Fields& value) {
    std::cout << '{'; bool first = true;
    for (const auto& item : value) { if (!first) std::cout << ','; first = false; std::cout << '"' << item.first << "\":" << item.second; }
    std::cout << '}';
}
void emit(const std::shared_ptr<SceneAttachment>& attachment, Reader& reader, const std::string& uri) {
    reader.observe();
    const auto progress = attachment->progress();
    const Saved saved(attachment->snapshot());
    if (terminal(progress)) {
        // Failure freezes admission/application; independently sum original
        // journal owners and compare the public cached status, including USD
        // edit logs/base ownership rather than only their compiled wire bytes.
        check(progress.pendingWrites == saved.ids.size() && progress.pendingBytes == saved.pendingBytes,
              "actor status omitted retained journal intent ownership");
        auto status = record<OmniUsdLiveStatus>();
        const auto until = Clock::now() + std::chrono::seconds(3);
        do {
            check(omniUsdLiveGetLayerStatus(uri.c_str(), &status) == OMNI_USD_LIVE_OK, "local status query failed");
            if (status.result == progress.result) break;
            check(Clock::now() < until, "terminal status publication timed out"); std::this_thread::yield();
        } while (true);
        check(status.pending_write_count == saved.ids.size() && status.pending_bytes == saved.pendingBytes &&
              bool(status.has_unconfirmed_edits) == !saved.ids.empty(), "public status lost complete intent accounting");
    }
    std::cout << "{\"type\":\"state\",\"result\":" << progress.result << ",\"gate\":" << int(progress.gate)
              << ",\"generation\":" << progress.generation << ",\"covered\":" << progress.covered
              << ",\"applied\":" << saved.applied << ",\"pending_ids\":[";
    for (size_t i = 0; i < saved.ids.size(); ++i) { if (i) std::cout << ','; std::cout << saved.ids[i]; }
    std::cout << "],\"authoritative\":"; jsonFields(saved.authoritative);
    std::cout << ",\"visible\":"; jsonFields(saved.visible);
    std::cout << ",\"retained\":" << progress.retained << ",\"authored_pending\":" << saved.authoredCount << '}' << std::endl;
}
template<class Predicate> void waitFor(Reader& reader, Predicate predicate) {
    const auto until = Clock::now()+std::chrono::seconds(10);
    while (!predicate()) {
        reader.healthy(); check(Clock::now() < until,"attachment wait timed out");
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    reader.healthy();
}
struct RuntimeOwner {
    bool initialized = false;
    OmniUsdLiveEndpoint endpoint = 0;
    std::shared_ptr<Runtime> runtime;
    std::shared_ptr<SceneAttachment> attachment;
    void cleanup() {
        if (attachment) { check(attachment->close(5000) == OMNI_USD_LIVE_OK,"attachment close timed out"); attachment.reset(); }
        if (endpoint) {
            const auto until = Clock::now()+std::chrono::seconds(5);
            for (;;) {
                const auto result = omniUsdLiveUnregisterEndpoint(endpoint,0);
                if (result == OMNI_USD_LIVE_OK) break;
                check(result == OMNI_USD_LIVE_BUSY && Clock::now() < until,"endpoint cleanup failed");
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            endpoint = 0;
        }
        runtime.reset();
        if (initialized) { check(omniUsdLiveShutdown(5000) == OMNI_USD_LIVE_OK,"runtime shutdown timed out"); initialized = false; }
    }
    ~RuntimeOwner() { try { cleanup(); } catch (...) { std::cerr << "runtime cleanup failed" << std::endl; std::_Exit(2); } }
};
std::string edit(const std::string& key, const VtValue& value) {
    EnvelopeInput metadata; metadata.fields = {{1,0,key,{}}};
    return encodeUsdEnvelope(metadata,DecodedValues{{value},{}});
}
class TestLayerAccess : public SdfFileFormat {
public:
    static void install(SdfLayer* layer, const WireScenePtr& scene,
                        const std::shared_ptr<UsdCreationWorkflow>& workflow) {
        auto snapshot=std::make_shared<SceneJournal::Snapshot>(); snapshot->visible=scene;
        // Inject a service publication through the production ownership
        // transaction. Arbitrary replacement must never authorize provenance.
        initializingCreationPublication(*layer,workflow).publish(std::move(snapshot));
        check(data(*layer)!=nullptr,"installed target backing unavailable");
    }
    static const SdfAbstractData* data(const SdfLayer& layer) { return _GetLayerData(layer).operator->(); }
};
void coverage() {
    auto expect = [](std::string_view bytes, uint64_t sequence, OmniUsdLiveResult expected, const WireSceneLimits& limits = {}) {
        bool rejected = false;
        try { (void)decodeSceneCoverage(bytes,sequence,false,limits); }
        catch (OmniUsdLiveResult error) { check(error == expected,"wrong coverage error"); rejected = true; }
        check(rejected,"invalid coverage accepted");
    };
    EnvelopeInput metadata; metadata.baseVersion = 999;
    const auto ordinary = encodeEnvelope(metadata);
    for (bool first : {false,true}) {
        const auto decoded = decodeSceneCoverage(ordinary,7,first);
        check(decoded.first == 7 && decoded.kind == LayerProgress::CoverageKind::Update && decoded.recoveryBoundary == first,
              "ordinary nonzero base expanded coverage or changed readiness");
    }
    metadata.isDiff = true; metadata.baseVersion = 0;
    const auto snapshot = encodeEnvelope(metadata);
    for (bool first : {false,true}) {
        const auto decoded = decodeSceneCoverage(snapshot,7,first);
        check(decoded.first == 1 && decoded.kind == LayerProgress::CoverageKind::Snapshot && decoded.recoveryBoundary == first,
              "snapshot range/readiness differs");
    }
    metadata.baseVersion = 3;
    const auto aggregate = encodeEnvelope(metadata);
    for (bool first : {false,true}) {
        const auto decoded = decodeSceneCoverage(aggregate,7,first);
        check(decoded.first == 4 && decoded.kind == LayerProgress::CoverageKind::Aggregate && decoded.recoveryBoundary == first,
              "aggregate range/readiness differs");
    }
    expect(ordinary,0,OMNI_USD_LIVE_MALFORMED_STREAM);
    expect(aggregate,3,OMNI_USD_LIVE_MALFORMED_STREAM);
    expect(aggregate,2,OMNI_USD_LIVE_MALFORMED_STREAM);
    metadata.baseVersion = UINT64_MAX;
    expect(encodeEnvelope(metadata),UINT64_MAX,OMNI_USD_LIVE_MALFORMED_STREAM);
    metadata.baseVersion = UINT64_MAX-1;
    const auto last = decodeSceneCoverage(encodeEnvelope(metadata),UINT64_MAX,false);
    check(last.first == UINT64_MAX && last.kind == LayerProgress::CoverageKind::Aggregate,"last valid aggregate overflow");
    expect("malformed",7,OMNI_USD_LIVE_MALFORMED_STREAM);
    metadata = {}; const std::string truncated("\x03\0",2);
    metadata.fields = {{1,0,"value",truncated}};
    expect(encodeEnvelope(metadata),7,OMNI_USD_LIVE_MALFORMED_STREAM);
    WireSceneLimits limits; limits.envelope.maxBytes = ordinary.size()-1;
    expect(ordinary,7,OMNI_USD_LIVE_RESOURCE_EXHAUSTED,limits);
    limits = {}; limits.values.maxDecodedBytes = 0;
    expect(edit("value",VtValue(7)),7,OMNI_USD_LIVE_RESOURCE_EXHAUSTED,limits);
    std::cout << "{\"type\":\"coverage\",\"result\":0}" << std::endl;
}
// Discovery/admission commands keep attachment ownership explicit: no reader
// thread or hidden observer may accidentally keep the claim alive after release.
int discovery(int argc, char** argv) {
    check(argc == 5,"usage: scene_transport authority session seed --discovery[-no-auth|-auth-deadline]");
    const auto seed = std::stoull(argv[3]);
    std::cerr << "scene discovery seed=" << seed << std::endl;
    RuntimeOwner owner;
    auto options = record<OmniUsdLiveRuntimeOptions>();
    options.retry_initial_ms = 5; options.retry_maximum_ms = 20; options.retry_max_attempts = 3;
    options.reconnect_initial_ms = 5; options.reconnect_maximum_ms = 20; options.reconnect_max_attempts = 3;
    // --discovery-auth-deadline: the initial-open deadline precedes the auth wait.
    const bool authDeadline = std::string(argv[4]) == "--discovery-auth-deadline";
    options.default_auth_wait_timeout_ms = authDeadline ? 30000 : 200;
    options.initial_open_timeout_ms = authDeadline ? 1 : 4000;
    check(omniUsdLiveInitialize(&options) == OMNI_USD_LIVE_OK,"discovery initialize failed"); owner.initialized = true;
    auto endpoint = record<OmniUsdLiveEndpointOptions>(); endpoint.authority = argv[1];
    endpoint.access_token = std::string(argv[4]) == "--discovery-no-auth" || authDeadline ? "" : "synthetic-writer";
    check(omniUsdLiveRegisterEndpoint(&endpoint,&owner.endpoint) == OMNI_USD_LIVE_OK,"discovery endpoint failed");
    owner.runtime = getRuntime();
    const auto uri = makeLayerUri(makeSessionUri(argv[1],argv[2]),"root.live");
    auto alias = uri;
    const auto leaf = alias.rfind("root.live"); check(leaf != std::string::npos,"missing canonical leaf");
    alias.replace(leaf,9,"%72oot.live");
    std::future<std::shared_ptr<SceneAttachment>> pending;
    bool retryRetiredClaim = false;
    struct RetirementHold {
        std::mutex mutex;
        std::condition_variable changed;
        bool entered = false, released = false;
        static void hook(void* context) noexcept {
            if (!context) return;
            auto& hold = *static_cast<RetirementHold*>(context);
            std::unique_lock<std::mutex> lock(hold.mutex);
            if (hold.entered) return;
            hold.entered = true;
            hold.changed.notify_all();
            if (!hold.changed.wait_for(lock,std::chrono::seconds(10),[&] { return hold.released; })) std::abort();
        }
    };
    static RetirementHold retirementHold;
    auto awaitRemoval = [&] {
        const auto deadline = Clock::now()+std::chrono::seconds(5);
        for (;;) {
            auto status = record<OmniUsdLiveStatus>();
            const auto result = omniUsdLiveGetLayerStatus(uri.c_str(),&status);
            if (result == OMNI_USD_LIVE_NOT_FOUND) return;
            check(result == OMNI_USD_LIVE_OK && Clock::now() < deadline,"retired scene status did not disappear");
            std::this_thread::yield();
        }
    };
    auto reopenWith = [&](auto open) {
        const auto deadline = Clock::now()+std::chrono::seconds(5);
        for (;;) {
            try { return open(); }
            catch (OmniUsdLiveResult result) {
                // A failed/closed actor removes its status immediately before
                // its final admission owner retires. Retry ONLY this known
                // cleanup interval; duplicate commands never use this path.
                if (!retryRetiredClaim || result != OMNI_USD_LIVE_ALREADY_EXISTS) throw;
                check(Clock::now() < deadline,"quiescent scene admission did not retire");
                std::this_thread::yield();
            }
        }
    };
    auto reopen = [&] { return reopenWith([&] { return SceneAttachment::openExisting(owner.runtime,uri); }); };
    auto state = [&](const char* type) {
        check(bool(owner.attachment),"discovery returned no attachment");
        const auto progress = owner.attachment->progress();
        const auto scene = owner.attachment->snapshot();
        std::cout << "{\"type\":\"" << type << "\",\"result\":" << progress.result
                  << ",\"object\":" << progress.object << ",\"applied\":" << scene->applied
                  << ",\"covered\":" << progress.covered << ",\"gate\":" << int(progress.gate)
                  << ",\"authoritative\":"; jsonFields(fields(scene->authoritative));
        std::cout << ",\"visible\":"; jsonFields(fields(scene->visible)); std::cout << '}' << std::endl;
    };
    auto accept = [&](auto action, bool waitForRemoval = true) {
        check(!owner.attachment,"open would replace retained owner");
        try {
            owner.attachment = action(); retryRetiredClaim = false;
            const auto progress = owner.attachment->progress();
            check(progress.object && progress.initialLatestPublished && progress.applied > 0 &&
                  owner.attachment->snapshot()->applied > 0,
                  "openExisting returned before initial scene application");
            state("opened");
        } catch (OmniUsdLiveResult result) {
            check(!owner.attachment,"failed open published an asset");
            if (waitForRemoval) awaitRemoval();
            retryRetiredClaim = true;
            std::cout << "{\"type\":\"rejected\",\"result\":" << result << '}' << std::endl;
        }
    };
    std::cout << "{\"type\":\"discovery-ready\"}" << std::endl;
    std::string line;
    while (std::getline(std::cin,line)) {
        std::istringstream command(line); std::string verb, extra; command >> verb;
        if (verb == "hold-failed-retirement") {
            check(!owner.attachment && !pending.valid(),"retirement hold requires no owner");
            { std::lock_guard<std::mutex> lock(retirementHold.mutex);
              retirementHold.entered = retirementHold.released = false; }
            LayerActor::setBeforeCloseQuiescentForTest(RetirementHold::hook,&retirementHold);
            std::cout << "{\"type\":\"retirement-held\"}" << std::endl;
        } else if (verb == "begin-open") {
            check(!owner.attachment && !pending.valid(),"duplicate pending open");
            pending = std::async(std::launch::async,[runtime=owner.runtime,uri] { return SceneAttachment::openExisting(runtime,uri); });
            std::cout << "{\"type\":\"opening\"}" << std::endl;
        } else if (verb == "prepare") {
            // The deferred open the USD file format performs. Only the
            // auth-deadline fixture uses it: no publication job is ever queued.
            check(!pending.valid(),"pending open not collected");
            SceneAttachment::Publication publication;
            publication.dispatch = [](std::function<void()> job) { job(); };
            publication.publish = [](SceneJournal::SnapshotPtr) {};
            accept([&] { return reopenWith([&] { return SceneAttachment::prepareExisting(owner.runtime,uri,publication); }); });
        } else if (verb == "finish-open" || verb == "finish-open-immediate") {
            check(pending.valid(),"no pending open");
            accept([&] { return pending.get(); },verb != "finish-open-immediate");
        } else if (verb == "open") {
            check(!pending.valid(),"pending open not collected");
            accept(reopen);
        } else if (verb == "duplicate") {
            try {
                auto duplicate = SceneAttachment::openExisting(owner.runtime,alias);
                check(duplicate->close(5000) == OMNI_USD_LIVE_OK,"unexpected duplicate cleanup failed");
                throw std::runtime_error("canonical alias admitted a second scene owner");
            } catch (OmniUsdLiveResult result) {
                std::cout << "{\"type\":\"rejected\",\"result\":" << result << '}' << std::endl;
            }
        } else if (verb == "retry-after-failure") {
            // The failed backing stays at the retirement gate until the open
            // itself is observed waiting on its admission.
            check(!owner.attachment && !pending.valid(),"retry-after-failure requires no held owner");
            const auto deadline = Clock::now()+std::chrono::seconds(5);
            {
                std::unique_lock<std::mutex> lock(retirementHold.mutex);
                check(retirementHold.changed.wait_until(lock,deadline,[&] { return retirementHold.entered; }),
                      "failed backing did not reach retirement gate");
            }
            auto retry = std::async(std::launch::async,[runtime=owner.runtime,uri] {
                return SceneAttachment::openExisting(runtime,uri);
            });
            while (owner.runtime->layerAdmissions->waiting() != 1) {
                check(Clock::now() < deadline,"retry did not wait for failed admission");
                std::this_thread::yield();
            }
            check(retry.wait_for(std::chrono::milliseconds(0)) == std::future_status::timeout,
                  "retry returned while failed admission was held");
            LayerActor::setBeforeCloseQuiescentForTest(nullptr,nullptr);
            { std::lock_guard<std::mutex> lock(retirementHold.mutex); retirementHold.released = true;
              retirementHold.changed.notify_all(); }
            check(retry.wait_until(deadline) == std::future_status::ready,"retry exceeded its deadline");
            accept([&] { return retry.get(); },false);
            check(owner.attachment && Clock::now() < deadline,"retry did not open within its deadline");
        } else if (verb == "pinned-refusal") {
            check(!owner.attachment && !pending.valid(),"pinned-refusal requires no held owner");
            const auto parsed = parseUri(uri);
            const auto deadline = Clock::now()+std::chrono::seconds(5);
            auto reopened = owner.runtime->layerAdmissions->reserve(owner.endpoint,parsed.sessionId,parsed.layerName,deadline);
            reopened->markRetiring();
            reopened->pinToHost(true);
            bool pinnedRefused = false;
            try {
                (void)owner.runtime->layerAdmissions->reserve(owner.endpoint,parsed.sessionId,parsed.layerName,
                    Clock::now()+std::chrono::milliseconds(200));
            } catch (OmniUsdLiveResult result) { pinnedRefused = result == OMNI_USD_LIVE_ALREADY_EXISTS; }
            check(owner.runtime->layerAdmissions->waiting() == 0,"host-pinned admission caused a wait");
            reopened->pinToHost(false);
            reopened.reset(); retryRetiredClaim = true;
            std::cout << "{\"type\":\"pinned-refusal\",\"refused\":"
                      << (pinnedRefused?"true":"false") << '}' << std::endl;
        } else if (verb == "close-held") {
            check(owner.attachment && owner.attachment->close(5000) == OMNI_USD_LIVE_OK,"held owner close failed");
            state("closed-held");
        } else if (verb == "release") {
            check(bool(owner.attachment),"no owner to release");
            check(owner.attachment->close(5000) == OMNI_USD_LIVE_OK,"release close failed");
            owner.attachment.reset(); awaitRemoval(); retryRetiredClaim = true;
            std::cout << "{\"type\":\"released\"}" << std::endl;
        } else if (verb == "set") {
            std::string key; int value;
            check(bool(owner.attachment) && bool(command >> key >> value) && identifier(key),"invalid discovery edit");
            const auto id = owner.attachment->submitLocal(edit(key,VtValue(value)));
            std::cout << "{\"type\":\"accepted\",\"id\":" << id << "}" << std::endl;
        } else if (verb == "state" || verb == "wait") {
            check(bool(owner.attachment),"no scene state");
            if (verb == "wait") {
                uint64_t target; check(bool(command >> target),"missing discovery sequence");
                const auto deadline = Clock::now()+std::chrono::seconds(5);
                for (;;) {
                    const auto progress = owner.attachment->progress();
                    if (terminal(progress) || (progress.applied >= target && owner.attachment->snapshot()->pending.empty())) break;
                    check(Clock::now() < deadline,"discovery scene wait timed out"); std::this_thread::yield();
                }
            }
            state("state");
        } else if (verb == "exit") {
            check(!pending.valid(),"uncollected pending discovery");
            owner.cleanup(); std::cout << "{\"type\":\"exited\"}" << std::endl; return 0;
        } else throw std::runtime_error("unknown discovery command");
        check(!(command >> extra),"unexpected discovery argument");
    }
    if (pending.valid()) { try { owner.attachment = pending.get(); } catch (OmniUsdLiveResult) {} }
    owner.cleanup(); return 0;
}
// Graph mode deliberately uses complete typed projections, independently compared
// by the Python oracle. This does not authorize captured structural admission.
constexpr std::size_t GraphJsonLimit = 16u*1024*1024;
struct GraphJson {
    std::string text;
    void add(std::string_view value) {
        check(value.size()<=GraphJsonLimit-text.size(),"graph probe JSON limit"); text.append(value);
    }
    template<class T> void number(T value) {
        char bytes[64]; const auto encoded=std::to_chars(bytes,bytes+sizeof(bytes),value);
        check(encoded.ec==std::errc{},"graph probe numeric encoding failed"); add({bytes,static_cast<std::size_t>(encoded.ptr-bytes)});
    }
    template<class Bytes> void hex(const Bytes& bytes) {
        constexpr char digits[]="0123456789abcdef";
        for(unsigned char byte:bytes) { const char pair[]={digits[byte>>4],digits[byte&15]}; add({pair,2}); }
    }
    void quote(std::string_view value) {
        constexpr char hex[]="0123456789abcdef";
        const auto escaped = [&](uint32_t unit) {
            const char bytes[]={'\\','u',hex[(unit>>12)&15],hex[(unit>>8)&15],hex[(unit>>4)&15],hex[unit&15]};
            add({bytes,6});
        };
        add("\"");
        for(std::size_t i=0;i<value.size();) {
            const auto byte=static_cast<unsigned char>(value[i++]);
            if(byte=='"') add("\\\"");
            else if(byte=='\\') add("\\\\");
            else if(byte<32 || byte==127) escaped(byte);
            else if(byte<128) { const char character=static_cast<char>(byte); add({&character,1}); }
            else {
                unsigned count; uint32_t scalar, minimum;
                if(byte>=0xc2 && byte<=0xdf) { count=1; scalar=byte&31; minimum=0x80; }
                else if(byte>=0xe0 && byte<=0xef) { count=2; scalar=byte&15; minimum=0x800; }
                else if(byte>=0xf0 && byte<=0xf4) { count=3; scalar=byte&7; minimum=0x10000; }
                else throw std::runtime_error("invalid UTF-8 in graph fixture");
                check(count<=value.size()-i,"truncated UTF-8 in graph fixture");
                while(count--) {
                    const auto next=static_cast<unsigned char>(value[i++]);
                    check((next&0xc0)==0x80,"invalid UTF-8 continuation in graph fixture");
                    scalar=(scalar<<6)|(next&63);
                }
                check(scalar>=minimum && scalar<=0x10ffff && !(scalar>=0xd800 && scalar<=0xdfff),
                      "invalid UTF-8 scalar in graph fixture");
                if(scalar<=0xffff) escaped(scalar);
                else { scalar-=0x10000; escaped(0xd800+(scalar>>10)); escaped(0xdc00+(scalar&1023)); }
            }
        }
        add("\"");
    }
    void value(const VtValue& value) {
        if(value.IsHolding<SdfTimeSampleMap>()) {
            add("{\"type\":\"time_samples\",\"value\":["); bool first=true;
            for(const auto& [time,sample]:value.UncheckedGet<SdfTimeSampleMap>()) {
                if(!first) add(",");
                first=false;
                uint64_t bits; std::memcpy(&bits,&time,sizeof(bits));
                add("{\"time_bits\":"); number(bits); add(",\"value\":"); this->value(sample); add("}");
            }
            add("]}");
        } else if(value.IsHolding<int>()) {
            add("{\"type\":\"int32\",\"value\":"); number(value.UncheckedGet<int>()); add("}");
        }
        else if(value.IsHolding<double>()) {
            add("{\"type\":\"double\",\"value\":");
            const auto scalar=value.UncheckedGet<double>();
            check(std::isfinite(scalar),"graph fixture contains unsupported nonfinite double");
            const auto begin=text.size(); number(scalar);
            if(text.find_first_of(".eE",begin)==std::string::npos) add(".0");
            add("}");
        } else if(value.IsHolding<SdfSpecifier>()) {
            add("{\"type\":\"specifier\",\"value\":");
            number(static_cast<uint32_t>(value.UncheckedGet<SdfSpecifier>())); add("}");
        } else if(value.IsHolding<bool>()) {
            add("{\"type\":\"bool\",\"value\":"); add(value.UncheckedGet<bool>()?"true":"false"); add("}");
        } else if(value.IsHolding<TfToken>()) {
            add("{\"type\":\"token\",\"value\":"); quote(value.UncheckedGet<TfToken>().GetString()); add("}");
        } else if(value.IsHolding<SdfVariability>()) {
            add("{\"type\":\"variability\",\"value\":"); number(static_cast<uint32_t>(value.UncheckedGet<SdfVariability>())); add("}");
        } else if(value.IsHolding<SdfPathVector>()) {
            add("{\"type\":\"paths\",\"value\":["); bool first=true;
            for(const auto& path:value.UncheckedGet<SdfPathVector>()) {
                if(!first) add(",");
                first=false; quote(path.GetString());
            }
            add("]}");
        } else if(value.IsHolding<SdfPathListOp>()) {
            const auto& op=value.UncheckedGet<SdfPathListOp>();
            add("{\"type\":\"path_op\",\"value\":{\"explicit\":"); add(op.IsExplicit()?"true":"false");
            const std::pair<const char*,const SdfPathVector*> buckets[] = {
                {"explicit_items",&op.GetExplicitItems()}, {"added",&op.GetAddedItems()},
                {"prepended",&op.GetPrependedItems()}, {"appended",&op.GetAppendedItems()},
                {"deleted",&op.GetDeletedItems()}, {"ordered",&op.GetOrderedItems()}};
            for(const auto& [name, paths]:buckets) {
                add(","); quote(name); add(":["); bool first=true;
                for(const auto& path:*paths) { if(!first) add(","); first=false; quote(path.GetString()); }
                add("]");
            }
            add("}}");
        } else if(value.IsHolding<std::vector<TfToken>>()) {
            add("{\"type\":\"tokens\",\"value\":["); bool first=true;
            for(const auto& token:value.UncheckedGet<std::vector<TfToken>>()) {
                if(!first) add(",");
                first=false; quote(token.GetString());
            }
            add("]}");
        } else if(!graph_values::project(*this,value))
            throw std::runtime_error("graph fixture contains unsupported field type: " + value.GetTypeName());
    }
};
template<class Sections>
std::string graphProjection(const Sections& sections,const SdfAbstractDataConstRefPtr& data) {
    GraphJson out; out.add("[");
    bool first=true;
    for(const auto& section:sections) {
        if(!first) out.add(",");
        first=false;
        out.add("{\"id\":"); out.number(section.id);
        out.add(",\"parent\":"); out.number(section.parent);
        out.add(",\"type\":"); out.number(static_cast<uint32_t>(section.type));
        out.add(",\"name\":"); out.quote(section.name);
        out.add(",\"order\":"); out.number(section.order);
        out.add(",\"path\":"); out.quote(section.path.GetString());
        out.add(",\"fields\":{");
        auto fields=data->List(section.path);
        std::sort(fields.begin(),fields.end(),[](const auto& a,const auto& b) { return a.GetString()<b.GetString(); });
        bool firstField=true;
        for(const auto& field:fields) {
            if(!firstField) out.add(",");
            firstField=false;
            out.quote(field.GetString()); out.add(":"); out.value(data->Get(section.path,field));
        }
        out.add("}}");
    }
    out.add("]"); return std::move(out.text);
}
struct GraphSection {
    uint64_t id = 0, parent = 0;
    SdfSpecType type = SdfSpecTypeUnknown;
    std::string name;
    uint32_t order = 0;
    SdfPath path;
};
std::string graphProjection(const WireScenePtr& scene) {
    if (!scene) return "[]";
    std::vector<GraphSection> sections;
    sections.reserve(scene->sectionCount());
    scene->visitSections(16384,[&](const auto& section) {
        sections.push_back({section.id,section.parent,section.type,std::string(section.name),section.order,section.path}); return true;
    });
    std::sort(sections.begin(),sections.end(),[](const auto& a,const auto& b) { return a.id<b.id; });
    return graphProjection(sections,scene->data());
}
struct GraphSaved {
    SceneJournal::SnapshotPtr owner;
    std::string authoritative, visible;
    uint64_t applied, revision;
    size_t pendingBytes;
    std::vector<uint64_t> ids;
    std::vector<std::shared_ptr<const SceneJournal::Intent>> intents;
    std::vector<std::string> bytes;
    explicit GraphSaved(SceneJournal::SnapshotPtr value) : owner(std::move(value)),
        authoritative(graphProjection(owner->authoritative)), visible(graphProjection(owner->visible)),
        applied(owner->applied), revision(owner->revision), pendingBytes(owner->pendingBytes) {
        for (const auto& pending : owner->pending) {
            check(bool(pending.intent), "graph pending owner missing");
            ids.push_back(pending.id); intents.push_back(pending.intent); bytes.push_back(pending.intent->bytes);
        }
    }
    void verify() const {
        check(graphProjection(owner->authoritative) == authoritative && graphProjection(owner->visible) == visible,
              "retained graph snapshot changed");
        check(owner->applied == applied && owner->revision == revision && owner->pendingBytes == pendingBytes &&
              owner->pending.size() == ids.size(), "retained graph metadata changed");
        for (size_t i = 0; i < ids.size(); ++i)
            check(owner->pending[i].id == ids[i] && owner->pending[i].intent == intents[i] &&
                  owner->pending[i].intent->bytes == bytes[i], "retained graph intent changed");
    }
};
// Changed callbacks and reader requests advance a generation. Observe it before
// checking predicates so a notification between the check and wait is retained.
class GraphEvents {
    std::mutex mutex_;
    std::condition_variable changed_;
    uint64_t revision_ = 0;
public:
    void signal() {
        std::lock_guard<std::mutex> lock(mutex_); ++revision_; changed_.notify_all();
    }
    uint64_t revision() {
        std::lock_guard<std::mutex> lock(mutex_); return revision_;
    }
    bool wait(uint64_t previous, Clock::time_point deadline = Clock::time_point::max()) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (deadline == Clock::time_point::max()) {
            changed_.wait(lock,[&] { return revision_ != previous; }); return true;
        }
        return changed_.wait_until(lock,deadline,[&] { return revision_ != previous; });
    }
};
class GraphReader {
    std::atomic<bool> stop_{false};
    std::shared_ptr<GraphEvents> events_;
    std::mutex mutex_;
    std::condition_variable changed_;
    uint64_t passes_ = 0;
    bool finished_ = false;
    std::promise<void> completed_;
    std::future<void> done_ = completed_.get_future();
    std::thread thread_;
public:
    GraphReader(std::shared_ptr<SceneAttachment> attachment,std::shared_ptr<GraphEvents> events)
        : events_(std::move(events)), thread_([this, attachment=std::move(attachment)] {
        try {
            std::deque<GraphSaved> saved;
            while (!stop_.load()) {
                const auto observed = events_->revision();
                for (const auto& item : saved) item.verify();
                auto current = attachment->snapshot();
                if (saved.empty() || saved.back().owner != current) {
                    if (saved.size() == 16) saved.pop_front();
                    saved.emplace_back(std::move(current));
                }
                {
                    std::lock_guard<std::mutex> lock(mutex_); ++passes_; changed_.notify_all();
                }
                if (!stop_.load()) events_->wait(observed);
            }
            completed_.set_value();
        } catch (...) { completed_.set_exception(std::current_exception()); }
        {
            std::lock_guard<std::mutex> lock(mutex_); finished_ = true; changed_.notify_all();
        }
        events_->signal();
    }) {}
    void healthy() {
        if (done_.valid() && done_.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
            done_.get(); check(stop_.load(), "graph reader stopped unexpectedly");
        }
    }
    void observe() {
        std::unique_lock<std::mutex> lock(mutex_);
        const auto before = passes_;
        events_->signal();
        check(changed_.wait_for(lock,std::chrono::seconds(3),[&] { return passes_ > before || finished_; }),
              "graph reader observation timed out");
        lock.unlock(); healthy();
    }
    void finish() {
        stop_ = true; events_->signal();
        if (thread_.joinable()) {
            if (done_.valid() && done_.wait_for(std::chrono::seconds(3)) != std::future_status::ready) {
                std::cerr << "graph reader failed to stop" << std::endl; std::_Exit(2);
            }
            thread_.join();
        }
        if (done_.valid()) done_.get();
    }
    ~GraphReader() { try { finish(); } catch (...) { std::cerr << "graph reader cleanup failed" << std::endl; std::_Exit(2); } }
};
bool graphCommand(std::string& line) {
    line.clear();
    char byte;
    while (std::cin.get(byte)) {
        if (byte == '\n') return true;
        check(line.size() < 4096, "graph command limit");
        line.push_back(byte);
    }
    check(std::cin.eof(), "graph command read failed");
    return !line.empty();
}
template<class Layer>
std::string hostGraphProjection(const Layer& layer) {
    GraphJson out; out.add("[");
    if (layer) {
        std::vector<SdfPath> paths;
        // USD 25.11 declares this read-only traversal non-const; this cast only traverses.
        const_cast<SdfLayer*>(layer.operator->())->Traverse(SdfPath::AbsoluteRootPath(),[&](const SdfPath& path) {
            check(paths.size() < 16384, "host graph fixture section limit"); paths.push_back(path);
        });
        std::sort(paths.begin(),paths.end(),[](const auto& a,const auto& b) { return a.GetString()<b.GetString(); });
        bool first=true;
        for (const auto& path:paths) {
            if (!first) out.add(",");
            first=false;
            out.add("{\"path\":"); out.quote(path.GetString()); out.add(",\"type\":");
            out.number(static_cast<uint32_t>(layer->GetSpecType(path))); out.add(",\"fields\":{");
            auto fields=layer->ListFields(path);
            std::sort(fields.begin(),fields.end(),[](const auto& a,const auto& b) { return a.GetString()<b.GetString(); });
            bool firstField=true;
            for (const auto& field:fields) {
                if (!firstField) out.add(",");
                firstField=false;
                out.quote(field.GetString()); out.add(":"); out.value(layer->GetField(path,field));
            }
            out.add("}}");
        }
    }
    out.add("]"); return std::move(out.text);
}
// Offline reader for a file SdfLayer::Export wrote; no live plugin, runtime or
// service is involved. Prints the same host projection the live report uses.
int usdGraph(const std::string& path) {
    const auto layer=SdfLayer::OpenAsAnonymous(path);
    check(bool(layer),"exported USD layer did not open");
    std::cout << "{\"host\":" << hostGraphProjection(layer) << '}' << std::endl;
    return 0;
}
class GraphStockFormat final : public SdfFileFormat {
    const SdfDataRefPtr data_;
public:
    explicit GraphStockFormat(const WireScenePtr& base)
        : SdfFileFormat(TfToken("graph_stock_test"),TfToken("1"),TfToken("usd"),"graphstock"),
          data_(TfCreateRefPtr(new SdfData)) { data_->CopyFrom(base->data()); }
    SdfAbstractDataRefPtr InitData(const FileFormatArguments&) const override { return data_; }
    bool CanRead(const std::string&) const override { return false; }
    bool Read(SdfLayer*,const std::string&,bool) const override { return false; }
protected:
    SdfAbstractDataRefPtr _InitDetachedData(const FileFormatArguments& args) const override { return InitData(args); }
};
void publicGraphCreate(const SdfLayerRefPtr& layer,const SdfPath& parent,const std::string& name,int value) {
    SdfChangeBlock block;
    auto prim=parent.IsAbsoluteRootPath()?SdfPrimSpec::New(layer,name,SdfSpecifierDef,"Scope"):
        SdfPrimSpec::New(layer->GetPrimAtPath(parent),name,SdfSpecifierDef,"Scope");
    check(bool(prim),"public prim creation failed");
    auto child=SdfPrimSpec::New(prim,"Child",SdfSpecifierDef);
    check(bool(child),"public nested prim creation failed");
    auto attribute=SdfAttributeSpec::New(child,"amount",SdfValueTypeNames->Int,SdfVariabilityVarying,false);
    check(bool(attribute),"public attribute creation failed");
    attribute->SetDefaultValue(VtValue(value));
    check(bool(SdfRelationshipSpec::New(prim,"link",true,SdfVariabilityUniform)),"public relationship creation failed");
}
void publicGraphVariants(const SdfLayerRefPtr& layer,const SdfPath& parent,const std::string& name,int value) {
    SdfChangeBlock block;
    auto prim=parent.IsAbsoluteRootPath()?SdfPrimSpec::New(layer,name,SdfSpecifierDef,"Scope"):
        SdfPrimSpec::New(layer->GetPrimAtPath(parent),name,SdfSpecifierDef,"Scope");
    check(bool(prim),"public variant owner creation failed");
    auto set=SdfVariantSetSpec::New(prim,"look");
    check(bool(set),"public variant set creation failed");
    check(bool(SdfVariantSpec::New(set,"1")),"public numeric variant creation failed");
    auto variant=SdfVariantSpec::New(set,"high-low");
    check(bool(variant),"public hyphenated variant creation failed");
    auto child=SdfPrimSpec::New(variant->GetPrimSpec(),"Child",SdfSpecifierDef);
    check(bool(child),"public variant prim child creation failed");
    auto attribute=SdfAttributeSpec::New(variant->GetPrimSpec(),"amount",SdfValueTypeNames->Int);
    check(bool(attribute),"public variant attribute creation failed");
    attribute->SetDefaultValue(VtValue(value));
    auto nestedSet=SdfVariantSetSpec::New(variant,"detail");
    check(bool(nestedSet),"public nested variant set creation failed");
    auto nested=SdfVariantSpec::New(nestedSet,"2");
    check(bool(nested),"public nested variant creation failed");
    check(bool(SdfRelationshipSpec::New(nested->GetPrimSpec(),"link",true,SdfVariabilityUniform)),
          "public nested variant relationship creation failed");
}
void publicGraphTargets(const SdfLayerRefPtr& layer,const SdfPath& parent,const std::string& name,int value) {
    SdfChangeBlock block;
    auto prim=parent.IsAbsoluteRootPath()?SdfPrimSpec::New(layer,name,SdfSpecifierDef,"Scope"):
        SdfPrimSpec::New(layer->GetPrimAtPath(parent),name,SdfSpecifierDef,"Scope");
    check(bool(prim),"public target owner creation failed");
    auto attribute=SdfAttributeSpec::New(prim,"amount",SdfValueTypeNames->Int);
    check(bool(attribute),"public connection owner creation failed");
    attribute->SetDefaultValue(VtValue(value));
    // Public proxies canonicalize these paths relative to the owning prim.
    // The referenced targets deliberately need not exist in this layer.
    attribute->GetConnectionPathList().GetExplicitItems()=SdfPathVector{SdfPath("Input.x"),SdfPath("/External.y")};
    auto relation=SdfRelationshipSpec::New(prim,"link",true,SdfVariabilityUniform);
    check(bool(relation),"public target relationship creation failed");
    relation->GetTargetPathList().GetExplicitItems()=SdfPathVector{SdfPath("Target"),SdfPath("/Elsewhere")};
}
class CreationHost : public TfWeakBase {
    struct State {
        std::mutex mutex;
        std::condition_variable queued;
        std::deque<std::function<void()>> jobs;
        const std::thread::id thread=std::this_thread::get_id();
        bool reject=false;
        uint64_t retiringSubject=0;
        bool retirementObserved=false, retirementBusy=false, retirementVisible=false;
        std::function<void()> currentProbe; // Original host only; consumed before callback entry.
    };
    struct RetirementProbe {
        std::shared_ptr<State> state;
        ~RetirementProbe() {
            if(!state->retiringSubject) return;
            state->retirementObserved=true;
            try {
                RetainedUsdCreationInfo info;
                state->retirementVisible=nextRetainedUsdCreation(0,info) && info.subjectId==state->retiringSubject;
                try { (void)acquireUsdCreation(state->retiringSubject); }
                catch(OmniUsdLiveResult error) { state->retirementBusy=error==OMNI_USD_LIVE_BUSY; }
            } catch(...) {}
        }
    };
    const std::shared_ptr<State> state_=std::make_shared<State>();
    TfNotice::Key identifierNotice_;
    size_t identifierNotices_=0;
    std::weak_ptr<SceneAttachment> attachment_;
    std::function<void()> heldJob_;
    void identifierChanged(const SdfNotice::LayerIdentifierDidChange& notice) {
        if (notice.GetNewIdentifier().compare(0,7,"live://")) return;
        check(omniUsdLiveWaitForPendingUpdates(100)==OMNI_USD_LIVE_BUSY,
              "identifier notice could block its own host publication");
        auto status=record<OmniUsdLiveStatus>();
        check(omniUsdLiveGetLayerStatus(notice.GetNewIdentifier().c_str(),&status)==OMNI_USD_LIVE_OK &&
              status.state==OMNI_USD_LIVE_STATE_OPENING,"identifier notice exposed premature public readiness");
        const auto attachment=attachment_.lock();
        check(attachment && status.subject_id==attachment->subjectId(),"identifier notice lost shared subject");
        bool submitBusy=false;
        try { (void)attachment->submitLocal(encodeEnvelope(EnvelopeInput{})); }
        catch(OmniUsdLiveResult result) { submitBusy=result==OMNI_USD_LIVE_BUSY; }
        check(submitBusy,"identifier notice admitted authoring before canonical activation completed");
        ++identifierNotices_;
    }
public:
    explicit CreationHost(bool enabled) {
        auto retirement=std::make_shared<RetirementProbe>(); retirement->state=state_;
        if (enabled) identifierNotice_=TfNotice::Register(TfCreateWeakPtr(this),&CreationHost::identifierChanged);
        if (enabled) setUsdPublicationDispatcher([state=state_,retirement](auto job) {
            std::lock_guard<std::mutex> lock(state->mutex);
            if (state->reject || state->jobs.size()>=32) throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
            state->jobs.push_back(std::move(job));
            state->queued.notify_all();
        },[state=state_,retirement] {
            if(std::this_thread::get_id()!=state->thread) return false;
            auto probe=std::move(state->currentProbe); if(probe) probe();
            return true;
        });
    }
    ~CreationHost() {
        TfNotice::Revoke(identifierNotice_);
        setUsdPublicationDispatcher({},{});
        std::deque<std::function<void()>> retired;
        { std::lock_guard<std::mutex> lock(state_->mutex); retired.swap(state_->jobs); }
    }
    void pump() {
        for (size_t i=0;i<32;++i) {
            std::function<void()> job;
            { std::lock_guard<std::mutex> lock(state_->mutex);
              if (state_->jobs.empty()) return;
              job=std::move(state_->jobs.front()); state_->jobs.pop_front(); }
            job();
        }
    }
    void reject() { std::lock_guard<std::mutex> lock(state_->mutex); state_->reject=true; }
    void waitQueued() {
        std::unique_lock<std::mutex> lock(state_->mutex);
        check(state_->queued.wait_for(lock,std::chrono::seconds(5),[&] { return !state_->jobs.empty(); }),
              "host dispatch did not reach queued-event barrier");
    }
    void runHeld() {
        check(!heldJob_,"host fixture already holds a closure");
        { std::lock_guard<std::mutex> lock(state_->mutex);
          check(state_->jobs.size()==1,"host fixture requires exactly one queued closure");
          heldJob_=std::move(state_->jobs.front()); state_->jobs.pop_front(); }
        heldJob_();
    }
    void dropHeld() { check(bool(heldJob_),"missing held host closure"); heldJob_={}; }
    bool held() const { return bool(heldJob_); }
    size_t identifierNotices() const { return identifierNotices_; }
    size_t queued() const { std::lock_guard<std::mutex> lock(state_->mutex); return state_->jobs.size(); }
    void attachment(const std::shared_ptr<SceneAttachment>& value) { attachment_=value; }
    void expectRegistryRetirement(uint64_t subject) { state_->retiringSubject=subject; }
    void verifyRegistryRetirement() {
        check(state_->retirementObserved && state_->retirementVisible && state_->retirementBusy,
              "final host closure destruction did not observe a retained BUSY registry entry");
    }
    void onCurrent(std::function<void()> callback) { state_->currentProbe=std::move(callback); }

    void replace() {
        setUsdPublicationDispatcher([](auto) { throw std::runtime_error("creation used replacement host dispatcher"); },
                                   [] { throw std::runtime_error("creation used replacement host context"); return false; });
    }
};
// Replace a live layer's whole content the way a host still can. Clear swaps
// in fresh backing (TransferContent is rejected before mutation); per-spec
// authoring then copies the stock layer's root fields and root prims.
void replaceByClear(const SdfLayerHandle& layer,const SdfLayerHandle& stock) {
    SdfChangeBlock block;
    layer->Clear();
    const auto root=SdfPath::AbsoluteRootPath();
    for(const auto& field:stock->ListFields(root))
        if(field!=SdfChildrenKeys->PrimChildren) layer->SetField(root,field,stock->GetField(root,field));
    for(const auto& prim:stock->GetRootPrims())
        if(!SdfCopySpec(stock,prim->GetPath(),layer,prim->GetPath())) throw std::runtime_error("replacement spec copy failed");
}
class CreationStatusObserver {
    const std::string uri_;
    mutable std::mutex mutex_;
    std::condition_variable changed_;
    std::array<OmniUsdLiveStatus,64> events_{};
    size_t count_=0;
    bool failed_=false;
    OmniUsdLiveCallbackHandle handle_=0;
    static void OMNI_USD_LIVE_CALL receive(void* user,const OmniUsdLiveStatusEvent* event) noexcept {
        auto& self=*static_cast<CreationStatusObserver*>(user);
        try {
            if(!event || !event->status || event->status->subject_kind!=OMNI_USD_LIVE_SUBJECT_LAYER || !event->uri || self.uri_!=event->uri) return;
            auto queried=record<OmniUsdLiveStatus>(),endpoint=record<OmniUsdLiveStatus>();
            const auto result=omniUsdLiveGetLayerStatus(self.uri_.c_str(),&queried);
            const bool valid=(result==OMNI_USD_LIVE_OK && queried.subject_id==event->status->subject_id &&
                              queried.revision>=event->status->revision) ||
                             (event->status->state==OMNI_USD_LIVE_STATE_CLOSED && result==OMNI_USD_LIVE_NOT_FOUND);
            check(valid && omniUsdLiveGetEndpointStatus(event->endpoint,&endpoint)==OMNI_USD_LIVE_OK,
                  "creation status callback reentrant queries failed");
            std::lock_guard<std::mutex> lock(self.mutex_);
            check(self.count_<self.events_.size(),"creation status fixture event capacity");
            if(self.count_) check(self.events_[self.count_-1].subject_id==event->status->subject_id &&
                                 self.events_[self.count_-1].revision<event->status->revision,
                                 "creation status handoff changed subject or reversed revision");
            self.events_[self.count_++]=*event->status; self.changed_.notify_all();
        } catch(...) {
            std::lock_guard<std::mutex> lock(self.mutex_); self.failed_=true; self.changed_.notify_all();
        }
    }
public:
    explicit CreationStatusObserver(std::string uri):uri_(std::move(uri)) {
        check(omniUsdLiveRegisterStatusCallback(0,this,receive,&handle_)==OMNI_USD_LIVE_OK,"creation status subscription failed");
    }
    ~CreationStatusObserver() { if(handle_) (void)omniUsdLiveUnregisterCallback(handle_,5000); }
    void close() {
        check(omniUsdLiveUnregisterCallback(handle_,5000)==OMNI_USD_LIVE_OK,"creation status subscription did not retire");
        handle_=0;
    }
    OmniUsdLiveStatus latest() const {
        std::lock_guard<std::mutex> lock(mutex_); check(!failed_,"creation status callback failed");
        return count_?events_[count_-1]:OmniUsdLiveStatus{};
    }
    // Waits for a delivered status event, not for elapsed time; the bound
    // is a failure-only watchdog.
    void waitUntil(const std::function<bool(const OmniUsdLiveStatus&)>& predicate) {
        std::unique_lock<std::mutex> lock(mutex_);
        check(changed_.wait_for(lock,std::chrono::seconds(5),[&] {
            return failed_ || (count_ && predicate(events_[count_-1]));
        }) && !failed_,"creation status callback did not reach the expected state");
    }
    void wait(uint64_t revision) {
        std::unique_lock<std::mutex> lock(mutex_);
        check(changed_.wait_for(lock,std::chrono::seconds(5),[&] {
            return failed_ || (count_ && events_[count_-1].revision>=revision);
        }) && !failed_,"creation status callback did not reach queried revision");
    }
};
int creationInteractive(int argc, char** argv) {
    check(argc == 6 || (argc == 7 && std::string(argv[6]) == "--diagnostics"),
          "usage: scene_transport --create AUTHORITY SESSION LAYER SEED_PATH [--diagnostics]");
    const auto size = std::filesystem::file_size(argv[5]);
    check(size && size <= 65536, "creation fixture seed size limit");
    std::ifstream input(argv[5],std::ios::binary);
    std::string bytes(static_cast<size_t>(size),'\0');
    check(bool(input.read(bytes.data(),static_cast<std::streamsize>(bytes.size()))), "creation fixture seed read failed");
    RuntimeOwner owner;
    const bool targetMode=std::string(argv[1])=="--create-target";
    const bool hostMode=std::string(argv[1])=="--create-host" || targetMode;
    CreationHost host(hostMode);
    auto options = record<OmniUsdLiveRuntimeOptions>();
    options.initial_open_timeout_ms = 4000;
    options.request_timeout_ms = 6000;
    options.retry_initial_ms = 5; options.retry_maximum_ms = 20;
    options.reconnect_initial_ms = 5; options.reconnect_maximum_ms = 20;
    if (argc == 6) options.flags=OMNI_USD_LIVE_RUNTIME_DISABLE_DIAGNOSTIC_HISTORY;
    else options.diagnostic_min_severity=OMNI_USD_LIVE_SEVERITY_DEBUG;
    options.max_status_subjects=2; // One endpoint plus one shared creation/attachment subject.
    check(omniUsdLiveInitialize(&options) == OMNI_USD_LIVE_OK, "creation initialize failed"); owner.initialized = true;
    auto endpoint = record<OmniUsdLiveEndpointOptions>(); endpoint.authority = argv[2]; endpoint.access_token = "synthetic-writer";
    check(omniUsdLiveRegisterEndpoint(&endpoint,&owner.endpoint) == OMNI_USD_LIVE_OK, "creation endpoint failed");
    // Creation receipt/export ledgers do not test jitter. Disable optional
    // timing probes so a scheduled Ping cannot race the exact no-RPC checks.
    auto jitter=record<OmniUsdLiveJitterParams>();
    check(omniUsdLiveConfigureJitterReduction(owner.endpoint,&jitter)==OMNI_USD_LIVE_OK,
          "creation fixture timing configuration failed");
    owner.runtime = getRuntime();
    const auto canonical=makeLayerUri(makeSessionUri(argv[2],argv[3]),argv[4]);
    CreationStatusObserver observed(canonical);
    auto usdCreation=hostMode && !targetMode ? UsdLayerCreation::prepare(owner.runtime,owner.endpoint,argv[3],argv[4],bytes) :
                              std::shared_ptr<UsdLayerCreation>{};
    auto creation=usdCreation ? usdCreation->creation() : SceneCreation::prepare(owner.runtime,owner.endpoint,argv[3],argv[4],bytes);
    std::shared_ptr<UsdCreationWorkflow> targetWorkflow;
    SdfLayerRefPtr target;
    SdfLayerHandle privateHost;
    if(targetMode) {
        const auto format=SdfFileFormat::FindById(TfToken("live"));
        check(bool(format),"production format missing for initializing target");
        target=format->NewLayer(format,canonical,canonical,ArAssetInfo{},{});
        check(bool(target),"production initializing target allocation failed");
        privateHost=target;
        targetWorkflow=std::make_shared<UsdCreationWorkflow>(owner.runtime,creation);
        prepareInitializingCreationTarget(*target,targetWorkflow);
        EnvelopeInput metadata;
        metadata.fields={{creation->seed()->sectionForPath(SdfPath("/P")),0,"counter",{}}};
        const auto installed=buildSceneCandidate(creation->seed(),encodeUsdEnvelope(metadata,DecodedValues{{VtValue(5)}, {}}));
        TestLayerAccess::install(target.operator->(),installed,targetWorkflow);
    } else if(hostMode) {
        for(const auto& loaded:SdfLayer::GetLoadedLayers())
            if(loaded->IsAnonymous() && loaded->GetIdentifier().find("live-creation.live")!=std::string::npos) {
                check(!privateHost,"ambiguous private creation fixture layer"); privateHost=loaded;
            }
        check(bool(privateHost),"private creation fixture layer missing");
    }
    SdfLayerRefPtr occupied;
    const auto originalSeed = creation->seed();
    const auto originalProjection = graphProjection(originalSeed);
    bool attemptedAttach = false;
    OmniUsdLiveResult attachResult = OMNI_USD_LIVE_BUSY;
    std::string diagnosticSnapshot="null";
    auto phaseName = [](SceneCreation::Phase phase) {
        switch (phase) {
        case SceneCreation::Phase::Prepared: return "prepared";
        case SceneCreation::Phase::Creating: return "creating";
        case SceneCreation::Phase::Confirmed: return "confirmed";
        case SceneCreation::Phase::UnissuedFailure: return "unissued";
        case SceneCreation::Phase::Rejected: return "rejected";
        case SceneCreation::Phase::OutcomeUnknown: return "unknown";
        }
        throw std::runtime_error("unknown creation phase");
    };
    auto report = [&](OmniUsdLiveResult result) {
        const auto state = creation->snapshot();
        check(creation->seed() == originalSeed && creation->bytes() == bytes &&
              graphProjection(originalSeed) == originalProjection, "creation lost original seed or wire");
        const auto scene = owner.attachment ? owner.attachment->snapshot() : SceneJournal::SnapshotPtr{};
        const auto progress = owner.attachment ? owner.attachment->progress() : LayerProgress::Snapshot{};
        GraphJson sourceFormats;
        sourceFormats.add("{\"seed\":"); sourceFormats.quote(originalSeed->sourceFormat());
        sourceFormats.add(",\"authoritative\":");
        sourceFormats.quote(scene && scene->authoritative ? scene->authoritative->sourceFormat() : "");
        sourceFormats.add(",\"visible\":");
        sourceFormats.quote(scene && scene->visible ? scene->visible->sourceFormat() : "");
        sourceFormats.add("}");
        const auto layer=usdCreation ? usdCreation->layer() : SdfLayerRefPtr{};
        // Find may invoke resolver Stat even for a registry observation. Scan
        // already-loaded handles so reports cannot manufacture service RPCs.
        SdfLayerHandle registered;
        if (hostMode) for (const auto& loaded:SdfLayer::GetLoadedLayers())
            if (loaded->GetIdentifier()==creation->admission()->canonicalUri()) registered=loaded;
        if (layer) check(registered==layer,"activated creation missing from canonical USD registry");
        const auto hostProjection=hostGraphProjection(layer);
        GraphJson privateEvidence;
        privateEvidence.add("{\"identifier\":");
        const auto actualHost=TfCreateRefPtrFromProtectedWeakPtr(privateHost);
        privateEvidence.quote(actualHost?actualHost->GetIdentifier():"");
        privateEvidence.add(",\"resolved_path\":");
        privateEvidence.quote(actualHost?actualHost->GetResolvedPath().GetPathString():"");
        privateEvidence.add(",\"scene\":");
        if(actualHost) {
            const auto* mutableHost=dynamic_cast<const HostSceneProjection*>(TestLayerAccess::data(*actualHost));
            if(mutableHost) {
                privateEvidence.add(graphProjection(mutableHost->baseline()));
                privateEvidence.add(",\"source_format\":"); privateEvidence.quote(mutableHost->baseline()->sourceFormat());
            } else {
                check(targetMode,"private creation host backing is not a host projection");
                privateEvidence.add("[],\"source_format\":\"\"");
            }
        } else privateEvidence.add("[],\"source_format\":\"\"");
        privateEvidence.add("}");
        const auto registeredProjection=hostGraphProjection(TfCreateRefPtrFromProtectedWeakPtr(registered));
        auto status=record<OmniUsdLiveStatus>();
        check(omniUsdLiveGetLayerStatus(canonical.c_str(),&status)==OMNI_USD_LIVE_OK,"creation public status missing");
        const auto delivered=observed.latest();
        GraphJson pending; pending.add("["); bool firstIntent=true;
        if(scene) for(const auto& item:scene->pending) {
            const auto& intent=item.intent;
            if(!firstIntent) pending.add(",");
            firstIntent=false;
            pending.add("{\"id\":"); pending.number(item.id);
            pending.add(",\"local_ordinal\":"); pending.number(intent->localOrdinal);
            pending.add(",\"base_revision\":"); pending.number(intent->baseRevision);
            pending.add(",\"dependency_cutoff\":"); pending.number(intent->dependencyCutoff);
            pending.add(",\"retained_bytes\":"); pending.number(intent->retainedBytes);
            pending.add(",\"bytes\":\"");
            pending.hex(intent->bytes);
            pending.add("\"}");
        }
        pending.add("]");
        // Peer work the actor retains without applying it, in admission order.
        GraphJson peers; peers.add("["); bool firstPeer=true;
        if(owner.attachment) for(const auto& work:owner.attachment->retained()) {
            if(work.disposition!=LayerProgress::Disposition::Peer) continue;
            const auto effect=std::static_pointer_cast<const LayerActor::Effect>(work.owner);
            check(effect && effect->incoming,"retained creation peer lost its original incoming owner");
            if(!firstPeer) peers.add(",");
            firstPeer=false;
            peers.add("{\"id\":"); peers.number(work.id); peers.add(",\"first\":"); peers.number(work.first);
            peers.add(",\"last\":"); peers.number(work.last); peers.add(",\"applied\":");
            peers.add(work.applied?"true":"false"); peers.add(",\"bytes\":\"");
            peers.hex(effect->bytes());
            peers.add("\"}");
        }
        peers.add("]");
        std::cout << std::boolalpha << "{\"result\":" << result << ",\"phase\":\"" << phaseName(state.phase)
                  << "\",\"create_result\":" << state.result << ",\"object\":" << state.object
                  << ",\"sequence\":" << state.sequence << ",\"request\":" << state.request
                  << ",\"started\":" << state.started << ",\"closing\":" << state.closing
                  << ",\"quiescent\":" << state.quiescent << ",\"seed\":" << originalProjection << ",\"bytes\":\"";
        constexpr char hex[] = "0123456789abcdef";
        for (unsigned char byte : bytes) std::cout << hex[byte >> 4] << hex[byte & 15];
        std::cout << "\",\"attached\":" << (hostMode ? bool(layer) : bool(owner.attachment && attachResult == OMNI_USD_LIVE_OK))
                  << ",\"source_formats\":" << sourceFormats.text
                  << ",\"host_mode\":" << hostMode << ",\"layer_exposed\":" << bool(layer)
                  << ",\"target_mode\":" << targetMode << ",\"target_alive\":" << bool(privateHost)
                  << ",\"registered\":" << bool(registered)
                  << ",\"registered_host\":" << registeredProjection
                  << ",\"private_host\":" << privateEvidence.text
                  << ",\"identifier_notices\":" << host.identifierNotices()
                  << ",\"host\":" << hostProjection << ",\"journal_applied\":" << (scene ? scene->applied : 0)
                  << ",\"host_queued\":" << host.queued() << ",\"covered\":" << progress.covered
                  << ",\"host_held\":" << host.held()
                  << ",\"retained\":" << progress.retained << ",\"retained_peers\":" << peers.text
                  << ",\"pending_intents\":" << pending.text
                  << ",\"journal_revision\":" << (scene?scene->revision:0)
                  << ",\"status_state\":" << status.state
                  << ",\"subject\":" << status.subject_id << ",\"revision\":" << status.revision
                  << ",\"status_result\":" << status.result << ",\"incident\":" << status.incident_id
                  << ",\"observed_subject\":" << delivered.subject_id << ",\"observed_revision\":" << delivered.revision
                  << ",\"observed_state\":" << delivered.state << ",\"observed_result\":" << delivered.result
                  << ",\"observed_incident\":" << delivered.incident_id
                  << ",\"status_pending\":" << status.pending_write_count << ",\"status_bytes\":" << status.pending_bytes
                  << ",\"status_unconfirmed\":" << status.has_unconfirmed_edits
                  << ",\"status_operation\":" << status.operation
                  << ",\"status_grpc_present\":" << status.grpc_status_present << ",\"status_grpc\":" << status.grpc_status
                  << ",\"status_progress_known\":" << status.progress_known
                  << ",\"diagnostics\":" << diagnosticSnapshot
                  << ",\"attach_result\":" << (owner.attachment && terminal(progress) ? progress.result : attachResult)
                  << ",\"applied\":" << progress.applied
                  << ",\"attach_object\":" << progress.object
                  << ",\"authoritative\":" << (scene ? graphProjection(scene->authoritative) : "[]")
                  << ",\"visible\":" << (scene ? graphProjection(scene->visible) : "[]") << '}' << std::endl;
        check(bool(std::cout), "creation output failed");
    };
    report(OMNI_USD_LIVE_OK);
    std::string line;
    while (graphCommand(line)) {
        std::istringstream command(line); std::string verb, extra;
        check(bool(command >> verb), "missing creation command");
        uint32_t timeout = 0;
        if (verb == "wait" || verb == "close" || verb == "host-prepare" || verb == "registry-host-prepare" ||
            verb == "reissue-deadline") {
            std::string value; check(bool(command >> value), "missing creation timeout");
            const auto parsed = std::from_chars(value.data(),value.data()+value.size(),timeout);
            check(parsed.ec == std::errc{} && parsed.ptr == value.data()+value.size() && timeout <= 5000,
                  "invalid creation timeout");
        }
        std::string exportPath;
        if (verb == "host-export") {
            check(bool(command >> std::quoted(exportPath)) && !exportPath.empty() && exportPath.size() <= 4096,
                  "invalid creation export path");
        }
        check(!(command >> extra), "unexpected creation command argument");
        OmniUsdLiveResult result = OMNI_USD_LIVE_OK;
        if (verb == "registry-retain") {
            check(usdCreation && !creation->snapshot().started,"registry fixture must begin prepared");
            std::array<UsdCreationLease,kRetainedUsdCreationCapacity> slots;
            for(auto& slot:slots) slot=reserveUsdCreation();
            bool full=false;
            try { (void)reserveUsdCreation(); }
            catch(OmniUsdLiveResult error) { full=error==OMNI_USD_LIVE_RESOURCE_EXHAUSTED; }
            check(full,"retention registry did not enforce its process bound");
            RetainedUsdCreationInfo sentinel{123,"unchanged"};
            check(!nextRetainedUsdCreation(0,sentinel) && sentinel.subjectId==123 && sentinel.liveUri=="unchanged",
                  "uninstalled reservation exposed identity or modified enumeration output");
            slots[0].install(usdCreation);
            check(!usdCreation,"retention install did not consume source owner");
            const auto subject=creation->status()->id();
            RetainedUsdCreationInfo info;
            check(nextRetainedUsdCreation(0,info) && info.subjectId==subject && info.liveUri==canonical,
                  "leased workflow was not discoverable by exact identity");
            bool busy=false;
            try { (void)acquireUsdCreation(subject); }
            catch(OmniUsdLiveResult error) { busy=error==OMNI_USD_LIVE_BUSY; }
            check(busy,"active hook lease allowed recursive ownership transfer");
            // Throw after ownership installation. Stack unwinding may release
            // reservations/lease flags, but cannot discard the installed owner.
            struct SimulatedHookExit {};
            try {
                auto active=std::move(slots[0]);
                throw SimulatedHookExit{};
            } catch(const SimulatedHookExit&) {}
            auto retained=acquireUsdCreation(subject);
            check(retained.workflow().creation()==creation && bool(privateHost),
                  "hook unwinding destroyed exact retained workflow or USD layer");
            auto offhost=std::async(std::launch::async,[&] {
                bool wrongHost=false;
                try { (void)retained.take(); }
                catch(OmniUsdLiveResult error) { wrongHost=error==OMNI_USD_LIVE_BUSY; }
                check(wrongHost && retained.close()==OMNI_USD_LIVE_BUSY,
                      "off-host retention transfer/cleanup succeeded");
            });
            offhost.get();
            check(!creation->snapshot().closing,"wrong-host cleanup cancelled prepared work");
        }
        else if (verb == "hook-fail-timeout") {
            UsdCreationLease lease;
            auto* workflow=usdCreation.get();
            if(!workflow) { lease=acquireUsdCreation(creation->status()->id()); workflow=&lease.workflow(); }
            const auto before=creation->snapshot();
            check(workflow->failHook(OMNI_USD_LIVE_OUTCOME_UNKNOWN)==OMNI_USD_LIVE_INVALID_ARGUMENT &&
                  // Result 17 is reserved.
                  workflow->failHook(OmniUsdLiveResult(17))==OMNI_USD_LIVE_INVALID_ARGUMENT &&
                  creation->snapshot().started==before.started,
                  "invalid local hook failure mutated finite admission");
            result=workflow->failHook(OMNI_USD_LIVE_TIMEOUT);
        }
        else if (verb == "registry-start") {
            // Model a bool hook throwing after submission. The process root
            // must still retain the workflow and its original provenance.
            struct SimulatedHookExit {};
            try {
                auto active=acquireUsdCreation(creation->status()->id());
                result=active.workflow().start();
                check(result==OMNI_USD_LIVE_OK,"retained creation start failed");
                throw SimulatedHookExit{};
            } catch(const SimulatedHookExit&) {}
            RetainedUsdCreationInfo info;
            check(nextRetainedUsdCreation(0,info) && info.subjectId==creation->status()->id() && bool(privateHost),
                  "post-issuance unwind discarded workflow");
        }
        else if (verb == "registry-take") {
            check(!usdCreation,"test already owns transferred workflow");
            usdCreation=acquireUsdCreation(creation->status()->id()).take();
            check(usdCreation->creation()==creation,"registry transferred a different workflow");
            bool gone=false;
            try { (void)acquireUsdCreation(creation->status()->id()); }
            catch(OmniUsdLiveResult error) { gone=error==OMNI_USD_LIVE_NOT_FOUND; }
            check(gone,"transferred workflow remained registry-owned");
        }
        else if (verb == "registry-close") {
            check(!usdCreation,"registry cleanup fixture must be exclusively retained");
            check(creation->close(5000)==OMNI_USD_LIVE_OK,"finite cleanup did not reach quiescence event");
            const auto subject=creation->status()->id();
            host.expectRegistryRetirement(subject);
            host.replace(); // The occupied slot is the only Host lifetime root from here on.
            result=acquireUsdCreation(subject).close();
            check(result==OMNI_USD_LIVE_OK && !privateHost,"registry cleanup retained USD target");
            host.verifyRegistryRetirement();
            RetainedUsdCreationInfo info{123,"unchanged"};
            check(!nextRetainedUsdCreation(0,info) && info.subjectId==123 && info.liveUri=="unchanged",
                  "retired registry entry remained discoverable or changed error output");
        }
        else if (verb == "registry-cancel-held") {
            check(!usdCreation && owner.attachment,"retained actor required for busy cleanup");
            host.waitQueued(); // Peer publication has reached the host boundary.
            const auto subject=creation->status()->id();
            host.expectRegistryRetirement(subject);
            host.replace();
            {
                auto lease=acquireUsdCreation(subject);
                check(std::async(std::launch::async,[&] { return lease.workflow().waitForClose(0); }).get()==
                      OMNI_USD_LIVE_INVALID_ARGUMENT,"wait-for-close initiated cancellation before explicit close");
                check(lease.close()==OMNI_USD_LIVE_BUSY,"queued publication did not hold cleanup BUSY");
                check(lease.workflow().waitForClose(5000)==OMNI_USD_LIVE_BUSY,
                      "original-host retirement wait did not poll while publication was queued");
                RetainedUsdCreationInfo info;
                check(nextRetainedUsdCreation(0,info) && info.subjectId==subject && bool(privateHost),
                      "BUSY cleanup discarded retained subject/host");
                // Cancellation cannot release the slot merely because the
                // queued callback runs: its last accepted copy still owns work.
                host.runHeld();
                check(lease.close()==OMNI_USD_LIVE_BUSY,"executed but retained callback allowed early cleanup");
            }
            // The failed cleanup lease is gone, but its slot remains occupied.
            check(acquireUsdCreation(subject).close()==OMNI_USD_LIVE_BUSY,"BUSY cleanup lost retained owner on retry");
            {
                auto lease=acquireUsdCreation(subject);
                check(lease.workflow().waitForClose(5000)==OMNI_USD_LIVE_BUSY,
                      "retained completed callback allowed early retirement");
                auto retired=std::async(std::launch::async,[&] { return lease.workflow().waitForClose(5000); });
                host.dropHeld();
                check(retired.get()==OMNI_USD_LIVE_OK,"released callback did not signal workflow retirement");
                check(lease.workflow().waitForClose(5000)==OMNI_USD_LIVE_OK,
                      "original-host retired workflow did not poll successfully");
            }
            owner.attachment.reset();
            result=acquireUsdCreation(subject).close();
            check(result==OMNI_USD_LIVE_OK && !privateHost,"retained busy cleanup did not finish");
            host.verifyRegistryRetirement();
            RetainedUsdCreationInfo info;
            check(!nextRetainedUsdCreation(0,info),"successful cleanup left an occupied slot");
        }
        else if(verb=="start-reentrant") {
            check(usdCreation && !targetMode,"start predicate fixture requires owned creation");
            bool entered=false;
            host.onCurrent([&] {
                entered=true;
                check(usdCreation->start()==OMNI_USD_LIVE_BUSY && !creation->snapshot().started,
                      "reentrant host predicate consumed creation admission");
            });
            result=usdCreation->start();
            check(entered,"start did not exercise original host predicate");
        }
        else if (verb == "start") result = usdCreation ? usdCreation->start() : creation->start();
        else if (verb == "wait") result = creation->wait(timeout);
        else if (verb == "close") result = creation->close(timeout);
        else if (verb == "mark-operation-ids") {
            // The creation is this endpoint's first request, so no Ping has
            // reported operation-ID support yet; mark it as a layer's would.
            owner.runtime->findEndpoint(owner.endpoint)->capabilities.store(
                Endpoint::kCapabilitiesKnown | Endpoint::kOperationIds);
        } else if (verb == "reissue-deadline") {
            // The caller's initial-open deadline, as an Export or first Save
            // hook bounds its creation's reissues, `timeout` ms after this command.
            creation->boundReissues(std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout));
        } else if (verb == "reissue-expire") {
            // The hook runs on the dispatch worker with a reissue queued and
            // moves the deadline to that moment, as a worker that runs the
            // reissue after its deadline sees it.
            static SceneCreation* expiring = nullptr;
            expiring = creation.get();
            SceneCreation::setBeforeResendForTest(+[](void*) noexcept {
                expiring->boundReissues(std::chrono::steady_clock::now());
                return false;
            }, nullptr);
        } else if (verb == "resend-close" || verb == "resend-fail") {
            // The hook runs on the dispatch worker with the re-send queued:
            // close the creation then, or fail the re-send's submission.
            static SceneCreation* closing = nullptr;
            closing = creation.get();
            SceneCreation::setBeforeResendForTest(verb == "resend-close" ?
                    +[](void*) noexcept { (void)closing->close(0); return false; } :
                    +[](void*) noexcept { return true; },
                nullptr);
        } else if (verb == "workflow-close") {
            if(targetWorkflow) {
                auto snapshot=targetWorkflow->snapshot(); snapshot.closing=true; targetWorkflow->finish(std::move(snapshot));
                if(owner.attachment) result=owner.attachment->close(5000);
                if(result==OMNI_USD_LIVE_OK) result=creation->close(5000);
            } else {
                check(bool(usdCreation),"workflow close requires host creation"); result=usdCreation->close();
            }
        }
        else if(verb=="host-export") {
            check(usdCreation && !targetMode && privateHost,"host export requires the private creation layer");
            // SdfLayer::Export of the creation's private host layer to an
            // ordinary file; the reopened file is projected like a live report.
            const auto layer=TfCreateRefPtrFromProtectedWeakPtr(privateHost);
            check(layer->Export(exportPath),"creation host export failed");
            std::cout << "{\"result\":0,\"host\":" << hostGraphProjection(SdfLayer::OpenAsAnonymous(exportPath)) << '}'
                      << std::endl;
            check(bool(std::cout),"creation output failed");
            continue;
        }
        else if(verb=="host-replace" || verb=="host-replace-deferred-start") {
            check(usdCreation && !targetMode && privateHost,"host replacement requires owned preparation");
            const auto layer=TfCreateRefPtrFromProtectedWeakPtr(privateHost);
            const auto stock=SdfLayer::CreateAnonymous("host-replacement.usda");
            check(stock->ImportFromString("#usda 1.0\ndef \"HostOnly\" {}\n"),"host replacement seed parse");
            stock->SetField(SdfPath::AbsoluteRootPath(),TfToken("hostWitness"),VtValue(777));
            check(!layer->PermissionToEdit(),"prepared creation layer is editable before activation");
            layer->SetPermissionToEdit(true); // A host override of the read-only creation layer.
            SdfChangeBlock block;
            replaceByClear(layer,stock);
            if(verb=="host-replace-deferred-start") {
                check(creation->snapshot().phase==SceneCreation::Phase::Prepared,
                      "replacement was observed before explicit start boundary");
                result=usdCreation->start();
            }
        }
        else if(verb=="target-destroy") {
            check(targetMode && bool(target),"missing live initializing target");
            target.Reset();
            check(!privateHost,"creation workflow retained the initializing layer");
        }
        else if(verb=="target-attach") {
            check(bool(target) && targetWorkflow && !owner.attachment &&
                  creation->snapshot().phase==SceneCreation::Phase::Confirmed,"invalid initializing actor fixture");
            struct Applied {
                std::mutex mutex;
                std::condition_variable changed;
                bool complete=false;
            };
            const auto applied=std::make_shared<Applied>();
            const auto* installed=dynamic_cast<const HostSceneProjection*>(TestLayerAccess::data(*target));
            check(installed && installed->baseline(),"initializing actor fixture lost distinct original host snapshot");
            const auto distinctHost=installed->baseline();
            SceneAttachment::Options configuration;
            configuration.session=argv[3]; configuration.layer=argv[4];
            configuration.object=creation->snapshot().object;
            configuration.admission=creation->admission(); configuration.status=creation->status();
            configuration.holdOpening=true; // Unfinished target is not public host readiness.
            configuration.publication=initializingCreationPublication(*target,targetWorkflow);
            configuration.deferInitialPublication=true;
            owner.attachment=SceneAttachment::open(owner.runtime,owner.endpoint,std::move(configuration),
                [applied](const LayerProgress::Snapshot& progress) {
                    std::lock_guard<std::mutex> lock(applied->mutex);
                    if(progress.applied || terminal(progress)) { applied->complete=true; applied->changed.notify_all(); }
                });
            auto snapshot=targetWorkflow->snapshot(); snapshot.openingAttempted=true; snapshot.attachment=owner.attachment;
            targetWorkflow->finish(std::move(snapshot));
            bindInitializingCreationAttachment(*target,targetWorkflow,owner.attachment);
            check(owner.attachment->waitForPrepared(5000)==OMNI_USD_LIVE_OK,
                  "initializing actor bootstrap did not become prepared");
            check(owner.attachment->activatePublication()==OMNI_USD_LIVE_OK,
                  "initializing actor bootstrap publication failed");
            std::unique_lock<std::mutex> lock(applied->mutex);
            check(applied->changed.wait_for(lock,std::chrono::seconds(5),[&] { return applied->complete; }),
                  "initializing actor never delivered applied or terminal event");
            check(owner.attachment->progress().applied==1 && !terminal(owner.attachment->progress()),
                  "initializing actor did not apply confirmed seed");
            lock.unlock();
            const auto* published=dynamic_cast<const HostSceneProjection*>(TestLayerAccess::data(*target));
            check(published && published->baseline()==owner.attachment->snapshot()->visible,
                  "production callback did not install the exact bootstrap snapshot");
            // Independently retain the actual last installed host contents,
            // even when they differ from the completed publication revision.
            TestLayerAccess::install(target.operator->(),distinctHost,targetWorkflow);
        }
        else if(verb=="target-fail-actor") {
            check(targetWorkflow && owner.attachment && !target,"failure requires destroyed attached target");
            owner.attachment->failPublication(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
            observed.waitUntil([](const OmniUsdLiveStatus& status) { return status.state==OMNI_USD_LIVE_STATE_FAILED; });
            const auto progress=owner.attachment->progress();
            check(terminal(progress) && progress.result==OMNI_USD_LIVE_RESOURCE_EXHAUSTED,
                  "destroyed target actor did not fail without closing");
        }
        else if (verb == "observed-status") {
            auto status=record<OmniUsdLiveStatus>();
            check(omniUsdLiveGetLayerStatus(canonical.c_str(),&status)==OMNI_USD_LIVE_OK,"creation status wait query failed");
            observed.wait(status.revision);
        }
        else if (verb == "diagnostics") {
            auto config=record<OmniUsdLiveDiagnosticExportOptions>(); config.max_records=64; config.max_output_bytes=65536;
            std::array<char,65536> output{}; uint64_t size=output.size();
            result=omniUsdLiveCopyDiagnostics(owner.endpoint,&config,output.data(),&size);
            check(result==OMNI_USD_LIVE_OK && size && size<=output.size(),"bounded creation diagnostic snapshot failed");
            diagnosticSnapshot=output.data();
        }
        else if(verb=="publication-state") {
            check(bool(owner.attachment),"creation publication state requires attachment");
            result=owner.attachment->waitForPrepared(0);
        }
        else if (verb == "host-prepare" || verb == "registry-host-prepare") {
            UsdCreationLease lease;
            auto* workflow=usdCreation.get();
            if(verb=="registry-host-prepare") { lease=acquireUsdCreation(creation->status()->id()); workflow=&lease.workflow(); }
            check(workflow,"host prepare requires owned USD creation");
            result=workflow->prepareAttachment(timeout);
            owner.attachment=workflow->attachment(); attachResult=result;
            host.attachment(owner.attachment);
        } else if (verb == "activate" || verb == "registry-activate") {
            UsdCreationLease lease;
            auto* workflow=usdCreation.get();
            if(verb=="registry-activate") { lease=acquireUsdCreation(creation->status()->id()); workflow=&lease.workflow(); }
            check(workflow,"activation requires owned USD creation");
            // Independent pending host work makes a reentrant notice drain
            // exercise the no-self-wait rule even after bootstrap retires.
            const auto drain=owner.runtime->drains->makeState();
            const auto ticket=drain->prepare();
            check(ticket->admit()==OMNI_USD_LIVE_OK,"identifier notice drain admission failed");
            result=workflow->activate(); attachResult=result;
            ticket->complete();
        } else if (verb == "pump") host.pump();
        else if (verb == "fail-publication") {
            check(bool(owner.attachment) && host.queued()==1,"external failure fixture requires one retained host job");
            owner.attachment->failPublication(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
        }
        else if (verb == "replace-host") host.replace();
        else if (verb == "host-reject-dispatch") host.reject();
        else if (verb == "occupy") {
            check(bool(usdCreation) && !occupied,"invalid canonical occupancy fixture");
            TfErrorMark errors;
            occupied=SdfLayer::New(SdfFileFormat::FindById(TfToken("live")),creation->admission()->canonicalUri());
            check(bool(occupied) && errors.IsClean(),"canonical occupancy fixture failed");
            occupied->SetPermissionToEdit(false);
        }
        else if (verb == "duplicate") {
            try {
                const auto duplicate = SceneCreation::prepare(owner.runtime,owner.endpoint,argv[3],argv[4],bytes);
                throw std::runtime_error("duplicate creation reservation admitted");
            } catch (OmniUsdLiveResult failure) { result = failure; }
        } else if (verb == "attach") {
            const auto state = creation->snapshot();
            if (attemptedAttach) result = attachResult;
            else if (state.closing) result = OMNI_USD_LIVE_CANCELLED;
            else if (state.phase != SceneCreation::Phase::Confirmed)
                result = state.phase == SceneCreation::Phase::OutcomeUnknown ? OMNI_USD_LIVE_OUTCOME_UNKNOWN : OMNI_USD_LIVE_BUSY;
            else {
                attemptedAttach = true;
                try {
                    SceneAttachment::Options configuration;
                    configuration.session = argv[3]; configuration.layer = argv[4];
                    configuration.object = state.object; configuration.resume = 0;
                    configuration.admission = creation->admission();
                    configuration.status = creation->status();
                    // Confirmed Create is not applied read coverage. Start
                    // with an empty journal so ordinary sequence1 is valid.
                    owner.attachment = SceneAttachment::open(owner.runtime,owner.endpoint,std::move(configuration));
                    const auto until = Clock::now()+std::chrono::seconds(5);
                    for (;;) {
                        const auto progress = owner.attachment->progress();
                        if (terminal(progress)) { result = progress.result; break; }
                        if (progress.gate == LayerProgress::Gate::Ready && progress.initialLatestPublished && progress.applied) break;
                        if (Clock::now() >= until) { result = OMNI_USD_LIVE_TIMEOUT; break; }
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    }
                } catch (OmniUsdLiveResult failure) { result = failure; }
                attachResult = result;
            }
        } else if (verb != "report" && verb != "quit") throw std::runtime_error("unknown creation command");
        if (verb == "quit") {
            if (usdCreation) {
                const auto until=Clock::now()+std::chrono::seconds(5);
                for (;;) {
                    const auto closed=usdCreation->close();
                    if (closed==OMNI_USD_LIVE_OK) break;
                    check(closed==OMNI_USD_LIVE_BUSY && Clock::now()<until,"owned USD creation close did not quiesce");
                    host.pump(); std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }
            if (owner.attachment) check(owner.attachment->close(5000) == OMNI_USD_LIVE_OK, "creation attachment close failed");
            result = creation->close(5000);
        }
        report(result);
        if (verb == "quit") break;
    }
    check(creation->close(5000) == OMNI_USD_LIVE_OK, "creation cleanup did not quiesce");
    observed.close(); target.Reset(); targetWorkflow.reset();
    usdCreation.reset(); creation.reset(); owner.cleanup(); return 0;
}
// Optional production file-format host for the --graph-host transport tests. USD opening,
// reads, publication and destruction stay on one independently progressing queue.
class GraphHost {
    std::mutex mutex_;
    std::condition_variable changed_;
    std::deque<std::function<void()>> jobs_;
    bool stop_ = false;
    std::shared_ptr<GraphEvents> events_;
    OmniUsdLiveCallbackHandle callback_ = 0;
    SdfLayerRefPtr layer_;
    std::shared_ptr<SceneAttachment> attachment_;
    std::function<void()> currentProbe_;
    std::thread worker_;
public:
    explicit GraphHost(std::shared_ptr<GraphEvents> events) : events_(std::move(events)), worker_([this] {
        for (;;) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                changed_.wait(lock,[&] { return stop_ || !jobs_.empty(); });
                if (stop_ && jobs_.empty()) return;
                job = std::move(jobs_.front()); jobs_.pop_front();
            }
            job(); job = {}; events_->signal();
        }
    }) {}
    void dispatch(std::function<void()> job) {
        std::lock_guard<std::mutex> lock(mutex_);
        check(!stop_ && jobs_.size() < 32,"graph host queue exhausted");
        jobs_.push_back(std::move(job)); changed_.notify_all();
    }
    template<class F> auto call(F action) -> decltype(action()) {
        auto task = std::make_shared<std::packaged_task<decltype(action())()>>(std::move(action));
        auto result = task->get_future(); dispatch([task] { (*task)(); });
        check(result.wait_for(std::chrono::seconds(8)) == std::future_status::ready,"graph host call did not finish");
        return result.get();
    }
    std::shared_ptr<SceneAttachment> open(OmniUsdLiveEndpoint endpoint,const std::string& uri) {
        check(omniUsdLiveRegisterStatusCallback(endpoint,events_.get(),
            [](void* user,const OmniUsdLiveStatusEvent*) { static_cast<GraphEvents*>(user)->signal(); },
            &callback_) == OMNI_USD_LIVE_OK,"graph host status registration failed");
        setUsdPublicationDispatcher([this](auto job) { dispatch(std::move(job)); },
                                   [this] {
                                       if(std::this_thread::get_id()!=worker_.get_id()) return false;
                                       auto probe=std::move(currentProbe_); if(probe) probe();
                                       return true;
                                   });
        // Queued jobs capture by value: a job that runs after the caller's
        // watchdog expired touches no caller frame.
        return call([this,uri] {
            TfErrorMark errors;
            layer_ = SdfLayer::FindOrOpen(uri);
            check(layer_ && errors.IsClean() && layer_->GetFileFormat()->GetFormatId() == TfToken("live"),
                  "graph host did not open production live format");
            const auto attachment = attachedLiveScene(*layer_);
            check(bool(attachment),"graph host has no live attachment");
            attachment_ = attachment;
            return attachment;
        });
    }
    std::string projection() { return call([this] { return hostGraphProjection(layer_); }); }
    // Author the live layer inside one host job through the shared admission
    // helper, which fences on the flush that job queued. Returns the ID of the
    // one intent that flush admitted, or the layer's failure cause with ID 0.
    // `write(layer, backing)` receives the live layer and its installed host
    // backing. A healthy layer whose flush admitted nothing fails the driver:
    // the campaign oracles bind one intent per command. While the issuance
    // window is full a flush job coalesces instead, so the job then admits the
    // edit at once, as a save does.
    // The queued job owns `write` and everything it captures, so a job that
    // runs after the caller's watchdog expired touches no caller frame.
    template<class Write> std::pair<OmniUsdLiveResult,uint64_t> author(Write write) {
        const auto admission=test::authorAdmittedIntent([this](std::function<void()> job) { dispatch(std::move(job)); },
            attachment_,[this,write=std::move(write)] {
                const auto* backing=dynamic_cast<const HostSceneProjection*>(TestLayerAccess::data(*layer_));
                check(backing,"graph host live layer has no host backing");
                write(layer_,*const_cast<HostSceneProjection*>(backing));
                if(attachment_->issuanceSaturated()) {
                    auto* bound=fileformat::BoundLayer::from(layer_.operator->());
                    check(bound,"graph host live layer is not bound");
                    bool issued=false;
                    const auto flushed=bound->flushLocalEdits(&issued);
                    if(bound->deferredInstallDue(flushed,issued)) bound->scheduleFlush();
                }
            });
        if(admission.result!=OMNI_USD_LIVE_OK) return {admission.result,0};
        check(admission.id!=0,"a graph authoring command must admit exactly one intent");
        return {OMNI_USD_LIVE_OK,admission.id};
    }
    bool exportLayer(const std::string& path) { return call([&] { return bool(layer_) && layer_->Export(path); }); }
    void close() {
        // Also covers EOF and exception unwinding: keep this dispatcher and its
        // worker alive until every admitted publication has retired.
        const auto attachment = call([this] {
            return attachment_ ? attachment_ : (layer_ ? attachedLiveScene(*layer_) : std::shared_ptr<SceneAttachment>{});
        });
        if (attachment) check(attachment->close(5000) == OMNI_USD_LIVE_OK,"graph host attachment did not quiesce");
        call([this] { layer_.Reset(); attachment_.reset(); });
        setUsdPublicationDispatcher({},{});
        if (callback_) {
            check(omniUsdLiveUnregisterCallback(callback_,5000) == OMNI_USD_LIVE_OK,"graph status callback did not retire");
            callback_ = 0;
        }
    }
    ~GraphHost() {
        try { close(); }
        catch (...) { std::cerr << "graph host cleanup failed" << std::endl; std::_Exit(2); }
        { std::lock_guard<std::mutex> lock(mutex_); stop_ = true; changed_.notify_all(); }
        worker_.join();
    }
};

int graphInteractive(int argc, char** argv) {
    check(argc == 5, "usage: scene_transport --graph AUTHORITY SESSION LAYER");
    RuntimeOwner owner;
    auto options = record<OmniUsdLiveRuntimeOptions>();
    options.retry_initial_ms = 5; options.retry_maximum_ms = 20; options.retry_max_attempts = 3;
    options.reconnect_initial_ms = 5; options.reconnect_maximum_ms = 20; options.reconnect_max_attempts = 3;
    options.default_auth_wait_timeout_ms = 200; options.initial_open_timeout_ms = 4000;
    check(omniUsdLiveInitialize(&options) == OMNI_USD_LIVE_OK, "graph initialize failed"); owner.initialized = true;
    const TestTls tls;
    auto endpoint = record<OmniUsdLiveEndpointOptions>(); endpoint.authority = argv[2]; endpoint.access_token = "synthetic-writer";
    tls.apply(endpoint);
    check(omniUsdLiveRegisterEndpoint(&endpoint, &owner.endpoint) == OMNI_USD_LIVE_OK, "graph endpoint failed");
    owner.runtime = getRuntime();
    const auto uri = makeLayerUri(makeSessionUri(argv[2], argv[3]), argv[4]);
    GraphJson writerIdentity;
    writerIdentity.quote(owner.runtime->findEndpoint(owner.endpoint)->writerId);
    SceneJournal::Limits limits;
    limits.scene.maxSections = 16384; limits.scene.maxFields = 65536;
    limits.scene.envelope.maxBytes = 16u*1024*1024;
    auto events = std::make_shared<GraphEvents>();
    std::unique_ptr<GraphHost> host;
    if (std::string(argv[1]) == "--graph-host") {
        host = std::make_unique<GraphHost>(events);
        // The file format opens this attachment, so its progress changes reach
        // GraphEvents through the test observer seam, as the Changed callback
        // below does for --graph. GraphHost signals after every host job.
        setLiveLayerProgressObserverForTest([events](const auto&) { events->signal(); });
        owner.attachment = host->open(owner.endpoint,uri);
        setLiveLayerProgressObserverForTest({});
    } else owner.attachment = SceneAttachment::openExisting(owner.runtime, uri, limits,
        [events](const auto&) { events->signal(); });
    GraphReader reader(owner.attachment,events);
    std::deque<GraphSaved> saved;
    auto report = [&](OmniUsdLiveResult result, uint64_t accepted = 0, bool diagnostics = false) {
        reader.observe();
        for (const auto& previous : saved) previous.verify();
        const auto snapshot = owner.attachment->snapshot();
        if (saved.empty() || saved.back().owner != snapshot) {
            if (saved.size() == 16) saved.pop_front();
            saved.emplace_back(snapshot);
        }
        const auto progress = owner.attachment->progress();
        auto status = record<OmniUsdLiveStatus>();
        const auto until = Clock::now()+std::chrono::seconds(3);
        do {
            const auto observed = events->revision();
            check(omniUsdLiveGetLayerStatus(uri.c_str(), &status) == OMNI_USD_LIVE_OK, "graph status query failed");
            if (!terminal(progress) || status.result == progress.result) break;
            reader.healthy();
            check(events->wait(observed,until), "graph terminal status publication timed out");
        } while (true);
        check(status.subject_id == owner.attachment->subjectId(), "graph status subject changed");
        const auto& current = saved.back();
        std::cout << "{\"result\":" << result << ",\"gate\":" << int(progress.gate)
                  << ",\"object\":" << progress.object << ",\"applied\":" << progress.applied
                  << ",\"journal_applied\":" << current.applied << ",\"covered\":" << progress.covered
                  << ",\"pending\":" << current.ids.size() << ",\"generation\":" << progress.generation
                  << ",\"progress_result\":" << progress.result << ",\"subject\":" << status.subject_id
                  << ",\"status_state\":" << status.state << ",\"status_result\":" << status.result
                  << ",\"status_grpc_present\":" << status.grpc_status_present << ",\"status_grpc\":" << status.grpc_status
                  << ",\"incident\":" << status.incident_id << ",\"accepted_id\":" << accepted
                  << ",\"writer\":" << writerIdentity.text << ",\"local_admitted\":" << progress.localAdmitted
                  << ",\"local_retired\":" << progress.localRetired
                  << ",\"pending_intents\":[";
        constexpr char hex[] = "0123456789abcdef";
        for (size_t i = 0; i < current.ids.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << "{\"id\":" << current.ids[i] << ",\"bytes\":\"";
            for (unsigned char byte : current.bytes[i]) std::cout << hex[byte >> 4] << hex[byte & 15];
            std::cout << "\"}";
        }
        std::cout << "],\"intent_provenance\":[";
        for(size_t i=0;i<current.intents.size();++i) {
            if(i) std::cout << ',';
            const auto& intent=*current.intents[i];
            std::cout << "{\"id\":" << current.ids[i] << ",\"local_ordinal\":" << intent.localOrdinal
                      << ",\"base_revision\":" << intent.baseRevision
                      << ",\"dependency_cutoff\":" << intent.dependencyCutoff << '}';
        }
        std::cout << "],\"authoritative\":" << current.authoritative << ",\"visible\":" << current.visible;
        if (host) std::cout << ",\"host\":" << host->projection();
        if (diagnostics) {
            auto config = record<OmniUsdLiveDiagnosticExportOptions>();
            config.max_records = 128; config.max_output_bytes = 262144;
            std::vector<char> output(config.max_output_bytes); uint64_t size = output.size();
            check(omniUsdLiveCopyDiagnostics(owner.endpoint,&config,output.data(),&size) == OMNI_USD_LIVE_OK &&
                  size && size <= output.size(), "bounded graph diagnostic snapshot failed");
            std::cout << ",\"diagnostics\":" << output.data();
        }
        std::cout << '}' << std::endl;
        check(bool(std::cout), "graph output failed");
    };
    report(OMNI_USD_LIVE_OK);
    // Graph authoring verbs run in one --graph-host job on the production live
    // layer. Namespace (move, erase) and public constructor (usd-*) verbs use
    // SdfLayer. Field (set, create) and sample verbs call the layer's installed
    // host backing, the same SdfAbstractData calls SdfLayer forwards: SdfLayer
    // drops missing-key erases and equal-value writes that the campaign
    // generators deliberately issue, and the sample fixture has no typeName for
    // SdfLayer::SetTimeSample. Record, flush, admission and apply are the same
    // runtime path either way; plugin-reference covers SdfLayer's filtering.
    std::string line;
    while (graphCommand(line)) {
        reader.healthy();
        std::istringstream command(line); std::string verb, extra;
        check(bool(command >> verb), "missing graph command");
        OmniUsdLiveResult result = OMNI_USD_LIVE_OK;
        uint64_t accepted = 0;
        if (verb == "report" || verb == "close" || verb == "diagnostics" || verb == "close-diagnostics") {
            check(!(command >> extra), "unexpected graph command argument");
            if (verb == "close" || verb == "close-diagnostics") result = owner.attachment->close(5000);
        } else if (verb == "export") {
            std::string path;
            check(host && bool(command >> std::quoted(path)) && !path.empty() && !(command >> extra),
                  "invalid graph export");
            // SdfLayer::Export of the failed live layer (design section 13); it leaves
            // the attachment's retained work untouched.
            const auto before = owner.attachment->snapshot();
            result = host->exportLayer(path) ? OMNI_USD_LIVE_OK : OMNI_USD_LIVE_IO_ERROR;
            check(owner.attachment->snapshot()->pending.size() == before->pending.size(),
                  "export changed retained graph work");
        } else if (verb == "wait" || verb == "journal-applied") {
            std::string value;
            check(bool(command >> value) && !(command >> extra), "invalid graph wait");
            uint64_t sequence = 0;
            const auto parsed = std::from_chars(value.data(), value.data()+value.size(), sequence);
            check(parsed.ec == std::errc{} && parsed.ptr == value.data()+value.size(), "invalid graph sequence");
            const auto until = Clock::now()+std::chrono::seconds(5);
            for (;;) {
                const auto observed = events->revision();
                const auto progress = owner.attachment->progress();
                const auto state = owner.attachment->snapshot();
                // Journal publication signals through the host job; actor
                // progress signals through Changed or, with --graph-host, the
                // progress observer seam. journal-applied deliberately leaves
                // pending writes intact, and it observes the journal of a layer
                // that a definitive rejection failed, which keeps applying peer
                // updates (design section 13).
                if (verb == "journal-applied" && state->applied >= sequence) break;
                if (terminal(progress) && verb == "wait") { result = progress.result; break; }
                if (state->applied >= sequence && progress.applied >= sequence && state->pending.empty()) break;
                reader.healthy();
                if (!events->wait(observed,until)) { result = OMNI_USD_LIVE_TIMEOUT; break; }
            }
        } else if (verb == "acknowledged") {
            std::string value;
            check(bool(command >> value) && !(command >> extra), "invalid graph acknowledged");
            uint64_t id = 0;
            const auto parsed = std::from_chars(value.data(), value.data()+value.size(), id);
            check(parsed.ec == std::errc{} && parsed.ptr == value.data()+value.size() && id, "invalid graph work ID");
            const auto until = Clock::now()+std::chrono::seconds(5);
            for (;;) {
                const auto observed = events->revision();
                const auto progress = owner.attachment->progress();
                if (terminal(progress)) { result = progress.result; break; }
                const auto work = owner.attachment->retained();
                if (std::any_of(work.begin(), work.end(), [&](const auto& item) {
                        return item.id == id && item.disposition == LayerProgress::Disposition::Acknowledged;
                    })) break;
                reader.healthy();
                if (!events->wait(observed,until)) { result = OMNI_USD_LIVE_TIMEOUT; break; }
            }
        } else if (verb == "move" || verb == "erase") {
            std::string sectionText, destination; uint64_t section = 0;
            check(bool(command >> sectionText), "missing graph structural identity");
            if (verb == "move") check(bool(command >> destination), "missing graph move destination");
            check(!(command >> extra), "unexpected graph structural argument");
            const auto parsed = std::from_chars(sectionText.data(), sectionText.data()+sectionText.size(), section);
            check(parsed.ec == std::errc{} && parsed.ptr == sectionText.data()+sectionText.size() && section > 1,
                  "invalid graph structural identity");
            check(bool(host), "graph authoring requires --graph-host");
            const auto attachment = owner.attachment;
            try {
                std::tie(result, accepted) = host->author([=](const SdfLayerRefPtr& layer, HostSceneProjection&) {
                    const auto path = attachment->snapshot()->visible->pathForSection(section);
                    if (path.IsEmpty()) throw OmniUsdLiveResult(OMNI_USD_LIVE_NOT_FOUND);
                    SdfBatchNamespaceEdit change;
                    change.Add(path, verb == "move" ? SdfPath(destination) : SdfPath());
                    if (!layer->Apply(change)) throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
                });
            } catch (OmniUsdLiveResult failure) { result = failure; }
        } else if (verb == "create" || verb == "usd-create" || verb == "usd-variants" || verb == "usd-targets") {
            std::string parentText, name; uint64_t parent = 0; int64_t number = 0;
            check(bool(command >> parentText >> name >> number) && !(command >> extra) && identifier(name) &&
                  number >= INT32_MIN && number <= INT32_MAX, "invalid graph create");
            const auto parsed = std::from_chars(parentText.data(), parentText.data()+parentText.size(), parent);
            check(parsed.ec == std::errc{} && parsed.ptr == parentText.data()+parentText.size() && parent,
                  "invalid graph create parent");
            check(bool(host), "graph authoring requires --graph-host");
            const auto attachment = owner.attachment;
            try {
                if(verb=="usd-create" || verb=="usd-variants" || verb=="usd-targets") {
                    std::tie(result, accepted) = host->author([=](const SdfLayerRefPtr& layer, HostSceneProjection&) {
                        const auto base = attachment->snapshot()->visible;
                        const auto path = base->pathForSection(parent);
                        if (path.IsEmpty()) throw OmniUsdLiveResult(OMNI_USD_LIVE_NOT_FOUND);
                        // The oracle starts with stock SdfData and executes the same
                        // public constructors independently of the live layer.
                        const SdfFileFormatRefPtr format=TfCreateRefPtr(new GraphStockFormat(base));
                        const auto stock=SdfLayer::CreateAnonymous("stock.graphstock",format);
                        check(bool(stock),"stock public graph fixture failed");
                        const auto construct=verb=="usd-targets"?publicGraphTargets:verb=="usd-variants"?publicGraphVariants:publicGraphCreate;
                        construct(stock,path,name,int(number));
                        construct(layer,path,name,int(number));
                        check(hostGraphProjection(layer)==hostGraphProjection(stock),
                              "host public authoring differs from exact stock fields/presence");
                    });
                } else {
                    std::tie(result, accepted) = host->author([=](const SdfLayerRefPtr&, HostSceneProjection& data) {
                        const auto path = attachment->snapshot()->visible->pathForSection(parent);
                        if (path.IsEmpty()) throw OmniUsdLiveResult(OMNI_USD_LIVE_NOT_FOUND);
                        const auto target = path.AppendChild(TfToken(name));
                        if (data.HasSpec(target)) throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
                        const auto children = data.Get(path,SdfChildrenKeys->PrimChildren);
                        auto names = children.IsEmpty()?std::vector<TfToken>{}:children.Get<std::vector<TfToken>>();
                        names.emplace_back(name);
                        data.CreateSpec(target,SdfSpecTypePrim);
                        data.Set(target,TfToken("marker"),VtValue(int(number)));
                        data.Set(path,SdfChildrenKeys->PrimChildren,VtValue(names));
                    });
                }
            } catch (OmniUsdLiveResult failure) { result = failure; }
        } else if (verb == "sample" || verb == "sample-map" || verb == "erase-sample") {
            uint64_t section=0,bits=0; int64_t number=0;
            check(bool(command >> section >> bits) && section,"invalid graph sample identity/time");
            if(verb!="erase-sample") check(bool(command >> number) && number>=INT32_MIN && number<=INT32_MAX,
                                           "invalid graph sample value");
            check(!(command >> extra),"extra graph sample arguments");
            double time; std::memcpy(&time,&bits,sizeof(time)); check(!std::isnan(time),"NaN graph sample time");
            check(bool(host), "graph authoring requires --graph-host");
            const auto attachment = owner.attachment;
            try {
                std::tie(result, accepted) = host->author([=](const SdfLayerRefPtr&, HostSceneProjection& data) {
                    const auto path=attachment->snapshot()->visible->pathForSection(section);
                    if(path.IsEmpty()) throw OmniUsdLiveResult(OMNI_USD_LIVE_NOT_FOUND);
                    if(verb=="sample-map") data.Set(path,SdfFieldKeys->TimeSamples,VtValue(SdfTimeSampleMap{{time,VtValue(int(number))}}));
                    else if(verb=="erase-sample") data.EraseTimeSample(path,time);
                    else data.SetTimeSample(path,time,VtValue(int(number)));
                });
            } catch(OmniUsdLiveResult failure) { result=failure; }
        } else if (verb == "reverse-children") {
            // Reverses a section's prim children: one Reorder naming all of them.
            uint64_t section = 0;
            check(bool(command >> section) && section && !(command >> extra), "invalid graph reverse-children");
            check(bool(host), "graph authoring requires --graph-host");
            const auto attachment = owner.attachment;
            try {
                std::tie(result, accepted) = host->author([=](const SdfLayerRefPtr&, HostSceneProjection& data) {
                    const auto path = attachment->snapshot()->visible->pathForSection(section);
                    if (path.IsEmpty()) throw OmniUsdLiveResult(OMNI_USD_LIVE_NOT_FOUND);
                    auto names = data.Get(path, SdfChildrenKeys->PrimChildren).Get<std::vector<TfToken>>();
                    std::reverse(names.begin(), names.end());
                    data.Set(path, SdfChildrenKeys->PrimChildren, VtValue(names));
                });
            } catch (OmniUsdLiveResult failure) { result = failure; }
        } else if (verb == "set-value" || verb == "erase-field") {
            // set-value authors one presence value family (graph_value_json.h); erase-field removes a field.
            uint64_t section = 0; std::string field;
            check(bool(command >> section >> field) && section && identifier(field), "invalid graph typed field");
            VtValue value;
            if (verb == "set-value") value = graph_values::parse(command);
            check(!(command >> extra), "unexpected graph typed field argument");
            check(bool(host), "graph authoring requires --graph-host");
            const auto attachment = owner.attachment;
            try {
                std::tie(result, accepted) = host->author([=](const SdfLayerRefPtr&, HostSceneProjection& data) {
                    const auto path = attachment->snapshot()->visible->pathForSection(section);
                    if (path.IsEmpty()) throw OmniUsdLiveResult(OMNI_USD_LIVE_NOT_FOUND);
                    if (verb == "set-value") data.Set(path, TfToken(field), value);
                    else data.Erase(path, TfToken(field));
                });
            } catch (OmniUsdLiveResult failure) { result = failure; }
        } else if (verb == "set") {
            std::string sectionText, field; int64_t number = 0; uint64_t section = 0;
            check(bool(command >> sectionText >> field >> number) && !(command >> extra) && identifier(field) &&
                  number >= INT32_MIN && number <= INT32_MAX, "invalid graph set");
            const auto parsed = std::from_chars(sectionText.data(), sectionText.data()+sectionText.size(), section);
            check(parsed.ec == std::errc{} && parsed.ptr == sectionText.data()+sectionText.size() && section,
                  "invalid graph section identity");
            check(bool(host), "graph authoring requires --graph-host");
            const auto attachment = owner.attachment;
            try {
                std::tie(result, accepted) = host->author([=](const SdfLayerRefPtr&, HostSceneProjection& data) {
                    const auto path = attachment->snapshot()->visible->pathForSection(section);
                    if (path.IsEmpty()) throw OmniUsdLiveResult(OMNI_USD_LIVE_NOT_FOUND);
                    data.Set(path, TfToken(field), VtValue(int(number)));
                });
            } catch (OmniUsdLiveResult failure) { result = failure; }
        } else throw std::runtime_error("unknown graph command");
        report(result, accepted, verb == "diagnostics" || verb == "close-diagnostics");
        if (verb == "close" || verb == "close-diagnostics") break;
    }
    reader.finish(); host.reset(); owner.cleanup();
    for (const auto& previous : saved) previous.verify();
    return 0;
}

int interactive(int argc, char** argv) {
    check(argc == 4,"usage: scene_transport authority session seed | --coverage");
    size_t parsed = 0; const auto seed = std::stoull(argv[3],&parsed);
    check(parsed == std::string(argv[3]).size(),"invalid seed");
    std::cerr << "scene transport seed=" << seed << std::endl;
    RuntimeOwner owner;
    auto options = record<OmniUsdLiveRuntimeOptions>();
    options.retry_initial_ms = 5; options.retry_maximum_ms = 20; options.retry_max_attempts = 3;
    options.reconnect_initial_ms = 5; options.reconnect_maximum_ms = 20; options.reconnect_max_attempts = 3;
    options.default_auth_wait_timeout_ms = 200; options.initial_open_timeout_ms = 4000;
    check(omniUsdLiveInitialize(&options) == OMNI_USD_LIVE_OK,"runtime initialization failed"); owner.initialized = true;
    auto endpoint = record<OmniUsdLiveEndpointOptions>(); endpoint.authority = argv[1]; endpoint.access_token = "synthetic-writer";
    check(omniUsdLiveRegisterEndpoint(&endpoint,&owner.endpoint) == OMNI_USD_LIVE_OK,"endpoint registration failed");
    owner.runtime = getRuntime();
    const auto layerUri = makeLayerUri(makeSessionUri(argv[1], argv[2]), "root.live");
    SceneAttachment::Options settings; settings.session = argv[2]; settings.layer = "root.live"; settings.object = 1;
    owner.attachment = SceneAttachment::open(owner.runtime,owner.endpoint,std::move(settings));
    Reader reader(owner.attachment);
    waitFor(reader,[&] {
        const auto progress = owner.attachment->progress();
        return terminal(progress) || (progress.gate == LayerProgress::Gate::Ready && progress.applied > 0 &&
                                    owner.attachment->snapshot()->applied > 0);
    });
    check(!terminal(owner.attachment->progress()),"attachment failed before readiness");
    {
        // Authoring starts once a Ping has reported the endpoint's
        // capabilities, so whether writes chain and re-send does not race it.
        const auto registered=owner.runtime->findEndpoint(owner.endpoint);
        const auto until=std::chrono::steady_clock::now()+std::chrono::seconds(10);
        while (!(registered->capabilities.load()&Endpoint::kCapabilitiesKnown)) {
            check(std::chrono::steady_clock::now()<until,"endpoint capabilities did not arrive");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    reader.observe(); std::cout << "{\"type\":\"ready\"}" << std::endl;
    std::string line;
    while (std::getline(std::cin,line)) {
        reader.healthy();
        std::istringstream command(line); std::string verb, key, extra; command >> verb;
        if (verb == "set" || verb == "erase") {
            check(bool(command >> key) && identifier(key),"invalid field key");
            VtValue value;
            if (verb == "set") {
                int64_t number; check(bool(command >> number) && number >= INT32_MIN && number <= INT32_MAX,"invalid int32 value");
                value = VtValue(int(number));
            }
            check(!(command >> extra),"unexpected command argument");
            try {
                const auto id = owner.attachment->submitLocal(edit(key,value));
                std::cout << "{\"type\":\"accepted\",\"id\":" << id << ",\"result\":0}" << std::endl;
            } catch (OmniUsdLiveResult result) {
                std::cout << "{\"type\":\"rejected\",\"result\":" << result << '}' << std::endl;
            }
        } else if (verb == "wait" || verb == "generation" || verb == "ack") {
            std::string number; check(bool(command >> number) && !number.empty() && number[0] != '-',"missing wait sequence");
            size_t used = 0; const auto sequence = std::stoull(number,&used);
            check(used == number.size() && !(command >> extra),"invalid wait sequence");
            check(verb != "ack" || sequence != 0,"ack requires a nonzero retained ID");
            waitFor(reader,[&] {
                const auto progress = owner.attachment->progress(); const auto state = owner.attachment->snapshot();
                if (terminal(progress)) return true;
                if (verb == "ack") {
                    for (const auto& work : owner.attachment->retained())
                        if (work.id == sequence && work.disposition == LayerProgress::Disposition::Acknowledged) return true;
                    return false;
                }
                if (verb == "generation") return progress.generation >= sequence && progress.gate == LayerProgress::Gate::Ready;
                return progress.applied >= sequence && state->applied >= sequence && state->pending.empty();
            });
            emit(owner.attachment,reader,layerUri);
        } else if (verb == "expire-read") {
            // The current read attempt's first-response deadline passes at the actor's next step.
            check(!(command >> extra),"unexpected command argument");
            owner.attachment->expireReadForTest();
            std::cout << "{\"type\":\"read-expired\"}" << std::endl;
        } else if (verb == "idle") {
            // The actor's worker is not running and no wake-driven pass is
            // outstanding, so no issuance decision is outstanding.
            check(!(command >> extra),"unexpected command argument");
            const auto deadline = Clock::now()+std::chrono::seconds(3);
            for (;;) {
                const auto transport = owner.attachment->transportSnapshot();
                if (!transport.workerScheduled && !transport.wakePending) break;
                check(Clock::now() < deadline,"scene actor did not go idle"); std::this_thread::yield();
            }
            const auto progress = owner.attachment->progress();
            std::cout << "{\"type\":\"idle\",\"gate\":" << int(progress.gate) << ",\"issued\":" << progress.issued
                      << ",\"queued\":" << progress.queued << '}' << std::endl;
        } else if (verb == "state" || verb == "failed" || verb == "close") {
            check(!(command >> extra),"unexpected command argument");
            if (verb == "failed") waitFor(reader,[&] { return terminal(owner.attachment->progress()); });
            if (verb == "close") check(owner.attachment->close(5000) == OMNI_USD_LIVE_OK,"close command timed out");
            emit(owner.attachment,reader,layerUri);
            if (verb == "close") break;
        } else throw std::runtime_error("unknown scene driver command");
    }
    reader.finish(); owner.cleanup(); return 0;
}
} // namespace
int scenePublicationDriver(int argc, char** argv);
int fileFormatMain(int argc, char** argv);
int main(int argc, char** argv) {
    try {
        // Every scene build checks its incremental derivation against a whole-scene
        // one, and every compilation its footprint against the whole scene.
        setWireSceneDerivationCheck(true);
        setFieldEditCompileCheck(true);
        const auto metadata=std::filesystem::absolute(argv[0]).parent_path()/"native-file-format"/"plugInfo.json";
        check(std::filesystem::is_regular_file(metadata),"native component plugin metadata missing");
        PlugRegistry::GetInstance().RegisterPlugins(metadata.string());
        if ((argc == 6 || argc == 7) && (std::string(argv[1]) == "--create" || std::string(argv[1]) == "--create-host" ||
                          std::string(argv[1]) == "--create-target"))
            return creationInteractive(argc, argv);
        if (argc == 5 && (std::string(argv[1]) == "--graph" || std::string(argv[1]) == "--graph-host"))
            return graphInteractive(argc, argv);
        if (argc == 3 && std::string(argv[1]) == "--usd-graph") return usdGraph(argv[2]);
        if (argc == 2 && std::string(argv[1]) == "--coverage") { coverage(); return 0; }
        if (argc >= 5 && argc <= 13 && std::string(argv[4]) == "--file-format") return fileFormatMain(argc, argv);
        if (argc == 5 && (std::string(argv[4]) == "--publication" ||
                         std::string(argv[4]) == "--prepared-publication" ||
                         std::string(argv[4]) == "--prepared-timeout")) return scenePublicationDriver(argc, argv);
        if (argc == 5 && (std::string(argv[4]) == "--discovery" || std::string(argv[4]) == "--discovery-no-auth" ||
                          std::string(argv[4]) == "--discovery-auth-deadline"))
            return discovery(argc,argv);
        return interactive(argc,argv);
    } catch (OmniUsdLiveResult result) { std::cerr << "scene driver result=" << result << std::endl; }
      catch (const std::exception& error) { std::cerr << "scene driver: " << error.what() << std::endl; }
    return 1;
}
