// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "SceneAttachment.h"
#include "LiveDiagnostics.h"
#include "HostSceneProjection.h"
#include "UsdValueCodec.h"
#include <pxr/base/tf/notice.h>
#include <pxr/usd/sdf/changeBlock.h>
#include <pxr/usd/sdf/fileFormat.h>
#include <pxr/usd/sdf/layer.h>
#include <pxr/usd/sdf/notice.h>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <future>
#include <iostream>
#include <map>
#include <mutex>
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
using Fields = std::map<std::string, int>;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
template<class T> T record() { T value{}; value.struct_size = sizeof(value); value.abi_version = OMNI_USD_LIVE_ABI_VERSION; return value; }
Fields values(const SdfLayer& layer) {
    Fields result;
    for (const auto& key : layer.ListFields(SdfPath::AbsoluteRootPath())) {
        const auto value = layer.GetField(SdfPath::AbsoluteRootPath(), key);
        require(value.IsHolding<int>(), "publication fixture expected root int fields");
        result.emplace(key.GetString(), value.UncheckedGet<int>());
    }
    return result;
}
void publicationJson(const SceneAttachment::PublicationProgress& p, const Fields& hostFields) {
    std::cout << ",\"publication\":{\"configured\":" << (p.configured ? "true" : "false")
              << ",\"has_attempt\":" << (p.hasAttempt ? "true" : "false")
              << ",\"has_completed\":" << (p.hasCompleted ? "true" : "false")
              << ",\"last_attempted_revision\":" << p.lastAttemptedRevision
              << ",\"last_completed_revision\":" << p.lastCompletedRevision
              << ",\"result\":" << p.result << "},\"actual_host_fields\":{";
    bool first = true;
    for (const auto& [key, value] : hostFields) {
        if (!first) std::cout << ',';
        first = false; std::cout << '"' << key << "\":" << value;
    }
    std::cout << '}';
}
class Format final : public SdfFileFormat {
    SdfAbstractDataRefPtr initial_;
public:
    explicit Format(WireScenePtr scene)
        : SdfFileFormat(TfToken("async_publication_probe"), TfToken("1"), TfToken("usd"), "async_publication_probe"),
          initial_(TfCreateRefPtr(new HostSceneProjection(std::move(scene), std::make_shared<HostFailureLatch>()))) {}
    SdfAbstractDataRefPtr InitData(const FileFormatArguments&) const override { return initial_; }
    bool CanRead(const std::string&) const override { return false; }
    bool Read(SdfLayer*, const std::string&, bool) const override { return false; }
    static void publish(SdfLayer* layer, WireScenePtr scene) {
        SdfAbstractDataRefPtr data = TfCreateRefPtr(new HostSceneProjection(std::move(scene), std::make_shared<HostFailureLatch>()));
        _SetLayerData(layer, data);
    }
protected:
    SdfAbstractDataRefPtr _InitDetachedData(const FileFormatArguments& args) const override { return InitData(args); }
};
// An independently progressing bounded host executor. All access to the test
// SdfLayer (including destruction) stays on this thread, outside external USD
// change blocks. This is not a Kit executor or production live layer binding.
class Host final : public TfWeakBase {
    std::mutex mutex_;
    std::condition_variable changed_;
    std::deque<std::function<void()>> jobs_;
    bool stop_ = false, armed_ = false, noticeHeld_ = false, releaseNotice_ = false;
    bool jobHeld_ = false, releaseJob_ = false, holdRetirement_ = false, failNext_ = false, failBeforeNext_ = false;
    bool holdBootstrap_ = false, bootstrapEntered_ = false, releaseBootstrap_ = false;
    std::exception_ptr failure_;
    Fields fields_;
    uint64_t revision_ = 0, sequence_ = 0, notices_ = 0, dispatches_ = 0;
    uint64_t attemptedRevision_ = 0, completedRevision_ = 0;
    bool attempted_ = false, completed_ = false;
    const std::thread::id openingThread_ = std::this_thread::get_id();
    const bool allowOpening_;
    // Only the host thread accesses these USD owners. The format outlives layer.
    SdfFileFormatRefPtr format_;
    SdfLayerRefPtr layer_;
    TfNotice::Key notice_;
    std::thread worker_;
    void healthy() { if (failure_) std::rethrow_exception(failure_); }
    template<class Predicate> void wait(std::unique_lock<std::mutex>& lock, Predicate condition) {
        require(changed_.wait_for(lock, std::chrono::seconds(5), [&] { return failure_ || condition(); }), "host barrier timed out");
        healthy();
    }
    void onChange(const SdfNotice::LayersDidChangeSentPerLayer&, const SdfLayerHandle&) {
        require(omniUsdLiveWaitForPendingUpdates(100) == OMNI_USD_LIVE_BUSY,
                "USD notice was allowed to wait for its own publication");
        std::unique_lock<std::mutex> lock(mutex_);
        ++notices_;
        if (armed_) {
            armed_ = false; noticeHeld_ = true; holdRetirement_ = true; changed_.notify_all();
            wait(lock, [&] { return releaseNotice_ || stop_; });
        }
    }
    void work() noexcept {
        try {
            for (;;) {
                std::function<void()> job;
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    changed_.wait(lock, [&] { return stop_ || !jobs_.empty(); });
                    if (stop_ && jobs_.empty()) break;
                    job = std::move(jobs_.front()); jobs_.pop_front();
                }
                job();
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    if (holdRetirement_) {
                        holdRetirement_ = false; jobHeld_ = true; changed_.notify_all();
                        wait(lock, [&] { return releaseJob_ || stop_; });
                    }
                }
                job = {}; // The actor must wait for this capture retirement.
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex_); failure_ = std::current_exception(); changed_.notify_all();
        }
        TfNotice::Revoke(notice_); layer_.Reset(); format_.Reset();
    }
public:
    explicit Host(bool allowOpening = false) : allowOpening_(allowOpening), worker_([this] { work(); }) {}
    ~Host() {
        { std::lock_guard<std::mutex> lock(mutex_); stop_ = true; changed_.notify_all(); }
        worker_.join();
    }
    SceneAttachment::Publication callbacks() {
        return {
            [this](std::function<void()> job) {
                std::lock_guard<std::mutex> lock(mutex_); healthy();
                require(!stop_ && jobs_.size() < 1, "host publication queue overflow");
                jobs_.push_back(std::move(job)); ++dispatches_; changed_.notify_all();
            },
            [this](SceneJournal::SnapshotPtr snapshot) {
                require(std::this_thread::get_id() == worker_.get_id() ||
                        (allowOpening_ && !layer_ && std::this_thread::get_id() == openingThread_),
                        "USD publication escaped host context");
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    attempted_ = true; attemptedRevision_ = snapshot->revision;
                    if (failBeforeNext_) {
                        failBeforeNext_ = false;
                        throw OmniUsdLiveResult(OMNI_USD_LIVE_INTERNAL_ERROR);
                    }
                }
                if (!layer_) {
                    format_ = TfCreateRefPtr(new Format(snapshot->visible));
                    layer_ = SdfLayer::CreateAnonymous("scene.async_publication_probe", format_);
                    require(bool(layer_), "host layer creation failed");
                    notice_ = TfNotice::Register(TfCreateWeakPtr(this), &Host::onChange, SdfLayerHandle(layer_));
                } else {
                    SdfChangeBlock block;
                    Format::publish(layer_.operator->(), snapshot->visible);
                } // All synchronous USD notices have returned here.
                const auto fields = values(*layer_);
                std::unique_lock<std::mutex> lock(mutex_);
                require(snapshot->revision > revision_, "host received a stale journal revision");
                revision_ = snapshot->revision; sequence_ = snapshot->applied; fields_ = fields;
                if (holdBootstrap_) {
                    holdBootstrap_ = false; bootstrapEntered_ = true; changed_.notify_all();
                    wait(lock, [&] { return releaseBootstrap_ || stop_; });
                }
                if (failNext_) { failNext_ = false; throw OmniUsdLiveResult(OMNI_USD_LIVE_INTERNAL_ERROR); }
                completed_ = true; completedRevision_ = snapshot->revision;
            }
        };
    }
    void arm() {
        std::lock_guard<std::mutex> lock(mutex_); healthy();
        armed_ = true; noticeHeld_ = jobHeld_ = releaseNotice_ = releaseJob_ = false;
    }
    void held() { std::unique_lock<std::mutex> lock(mutex_); wait(lock, [&] { return noticeHeld_; }); }
    void releaseNotice() {
        std::unique_lock<std::mutex> lock(mutex_); releaseNotice_ = true; changed_.notify_all();
        wait(lock, [&] { return jobHeld_; });
    }
    void releaseJob() { std::lock_guard<std::mutex> lock(mutex_); releaseJob_ = true; changed_.notify_all(); }
    void releaseAll() { std::lock_guard<std::mutex> lock(mutex_); releaseNotice_ = releaseJob_ = releaseBootstrap_ = true; armed_ = false; changed_.notify_all(); }
    void armBootstrap() { std::lock_guard<std::mutex> lock(mutex_); holdBootstrap_ = true; }
    void bootstrapEntered() { std::unique_lock<std::mutex> lock(mutex_); wait(lock, [&] { return bootstrapEntered_; }); }
    void releaseBootstrap() { std::lock_guard<std::mutex> lock(mutex_); releaseBootstrap_ = true; changed_.notify_all(); }
    void failNext() { std::lock_guard<std::mutex> lock(mutex_); failNext_ = true; }
    void failBeforeNext() { std::lock_guard<std::mutex> lock(mutex_); failBeforeNext_ = true; }
    // A distinct host task reads actual USD after every job already queued,
    // then the attachment's publication progress is checked against this
    // host's independent callback evidence. Neither journal-visible data nor
    // callback bookkeeping is the oracle for the installed host contents.
    std::pair<SceneAttachment::PublicationProgress, Fields> publicationFields(const SceneAttachment& attachment) {
        auto promise = std::make_shared<std::promise<Fields>>();
        auto result = promise->get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_); healthy();
            require(!stop_, "host read after host shutdown");
            jobs_.push_back([this, promise] {
                try { promise->set_value(layer_ ? values(*layer_) : Fields{}); }
                catch (...) { promise->set_exception(std::current_exception()); }
            });
            changed_.notify_all();
        }
        require(result.wait_for(std::chrono::seconds(5)) == std::future_status::ready,
                "actual host scene read timed out");
        auto fields = result.get();
        const auto p = attachment.publicationProgress();
        std::lock_guard<std::mutex> lock(mutex_); healthy();
        require(p.configured && p.hasAttempt == attempted_ && p.hasCompleted == completed_ &&
                p.lastAttemptedRevision == attemptedRevision_ && p.lastCompletedRevision == completedRevision_,
                "publication progress differs from independent callback evidence");
        return {p, std::move(fields)};
    }
    void state(const std::shared_ptr<SceneAttachment>& attachment, const char* type) {
        const auto progress = attachment->progress(); const auto journal = attachment->snapshot();
        std::lock_guard<std::mutex> lock(mutex_); healthy();
        std::cout << "{\"type\":\"" << type << "\",\"result\":" << progress.result
                  << ",\"applied\":" << progress.applied << ",\"covered\":" << progress.covered
                  << ",\"journal_applied\":" << journal->applied
                  << ",\"published\":" << sequence_ << ",\"revision\":" << revision_
                  << ",\"retained\":" << progress.retained << ",\"notices\":" << notices_
                  << ",\"dispatches\":" << dispatches_ << ",\"fields\":{";
        bool first = true;
        for (const auto& entry : fields_) { if (!first) std::cout << ','; first = false; std::cout << '"' << entry.first << "\":" << entry.second; }
        std::cout << "}}" << std::endl;
    }
};
struct Owner {
    bool initialized = false;
    OmniUsdLiveEndpoint endpoint = 0;
    OmniUsdLiveCallbackHandle readinessObserver = 0;
    std::shared_ptr<SceneAttachment> attachment;
    void waitReady(const std::string& uri) {
        require(!readinessObserver && omniUsdLiveCreateStatusObserver(endpoint, &readinessObserver) == OMNI_USD_LIVE_OK,
                "readiness observer registration failed");
        std::vector<char> resource(uri.size() + 1);
        const auto deadline = Clock::now() + std::chrono::seconds(3);
        for (;;) {
            const auto now = Clock::now();
            require(now < deadline, "actor did not publish readiness after host activation");
            const auto remaining = std::max<int64_t>(1,
                std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count());
            auto observation = record<OmniUsdLiveStatusObservation>();
            uint64_t size = resource.size();
            require(omniUsdLiveReadStatus(readinessObserver, uint32_t(remaining), &observation, resource.data(), &size)
                    == OMNI_USD_LIVE_OK, "actor readiness observation failed");
            require(omniUsdLiveAcknowledgeStatus(readinessObserver, observation.status.subject_id,
                    observation.status.revision) == OMNI_USD_LIVE_OK, "readiness observation ACK failed");
            if (observation.status.subject_kind != OMNI_USD_LIVE_SUBJECT_LAYER) continue;
            require(uri == resource.data(), "unexpected readiness subject");
            if (observation.status.state == OMNI_USD_LIVE_STATE_READY) break;
            require(observation.status.state != OMNI_USD_LIVE_STATE_FAILED &&
                    observation.status.state != OMNI_USD_LIVE_STATE_CLOSING &&
                    observation.status.state != OMNI_USD_LIVE_STATE_CLOSED, "activation never became ready");
        }
        require(omniUsdLiveUnregisterCallback(readinessObserver, 3000) == OMNI_USD_LIVE_OK,
                "readiness observer cleanup failed");
        readinessObserver = 0;
    }
    void close() {
        if (readinessObserver) {
            require(omniUsdLiveUnregisterCallback(readinessObserver, 5000) == OMNI_USD_LIVE_OK,
                    "readiness observer cleanup failed");
            readinessObserver = 0;
        }
        if (attachment) { require(attachment->close(5000) == OMNI_USD_LIVE_OK, "async attachment close failed"); attachment.reset(); }
        if (endpoint) {
            const auto until = Clock::now() + std::chrono::seconds(5);
            for (;;) {
                const auto result = omniUsdLiveUnregisterEndpoint(endpoint, 0);
                if (result == OMNI_USD_LIVE_OK) break;
                require(result == OMNI_USD_LIVE_BUSY && Clock::now() < until, "async endpoint did not retire");
                std::this_thread::yield();
            }
            endpoint = 0;
        }
        if (initialized) { require(omniUsdLiveShutdown(5000) == OMNI_USD_LIVE_OK, "async runtime shutdown failed"); initialized = false; }
    }
    ~Owner() { try { close(); } catch (...) { std::cerr << "async publication cleanup failed\n"; std::_Exit(2); } }
};
}
int scenePublicationDriver(int argc, char** argv) {
    require(argc == 5, "publication driver arguments missing");
    std::cerr << "scene publication seed=" << argv[3] << std::endl;
    Owner owner;
    const bool prepared = std::string(argv[4]) != "--publication";
    Host host(prepared);
    struct Cleanup {
        Owner& owner; Host& host;
        ~Cleanup() {
            host.releaseAll();
            try { owner.close(); } catch (...) { std::cerr << "async publication cleanup failed\n"; std::_Exit(2); }
        }
    } cleanup{owner, host}; // Quiesce callbacks before destroying their host.
    auto options = record<OmniUsdLiveRuntimeOptions>();
    // --prepared-timeout expires the initial-open deadline through a test seam
    // at a held point, never by elapsed time.
    const bool expireByTest = std::string(argv[4]) == "--prepared-timeout";
    options.initial_open_timeout_ms = 4000;
    options.reconnect_initial_ms = 5; options.reconnect_maximum_ms = 20; // Reconnects within milliseconds.
    require(omniUsdLiveInitialize(&options) == OMNI_USD_LIVE_OK, "async runtime initialize failed"); owner.initialized = true;
    auto endpoint = record<OmniUsdLiveEndpointOptions>(); endpoint.authority = argv[1]; endpoint.access_token = "synthetic-writer";
    require(omniUsdLiveRegisterEndpoint(&endpoint, &owner.endpoint) == OMNI_USD_LIVE_OK, "async endpoint registration failed");
    const auto uri = makeLayerUri(makeSessionUri(argv[1], argv[2]), "root.live");
    owner.attachment = prepared ? SceneAttachment::prepareExisting(getRuntime(), uri, host.callbacks()) :
        SceneAttachment::openExisting(getRuntime(), uri, {}, {}, host.callbacks());
    uint64_t bootstrapId = 0;
    std::weak_ptr<const void> bootstrapOwner;
    std::string bootstrapBytes;
    if (prepared) {
        const auto readiness = getRuntime()->diagnostics->snapshot(0, 0).metrics.readiness[1];
        require(readiness.initialWaiting == 1 && !readiness.initialByResolution[0].count,
                "prepared bootstrap completed the initial-readiness metric before host publication");
        const auto retained = owner.attachment->retained();
        require(retained.size() == 1 && retained.front().kind == LayerProgress::CoverageKind::Snapshot,
                "prepared fixture lacks one retained bootstrap receipt");
        bootstrapId = retained.front().id; bootstrapOwner = retained.front().owner;
        bootstrapBytes = std::string(std::static_pointer_cast<const LayerActor::Effect>(retained.front().owner)->bytes());
        require(!owner.attachment->progress().applied, "prepared bootstrap advanced actor applied frontier");
        auto status = record<OmniUsdLiveStatus>();
        require(omniUsdLiveGetLayerStatus(uri.c_str(), &status) == OMNI_USD_LIVE_OK &&
                status.state == OMNI_USD_LIVE_STATE_OPENING,
                "prepared bootstrap incorrectly advertised public readiness");
        require(omniUsdLiveWaitForPendingUpdates(0) == OMNI_USD_LIVE_TIMEOUT,
                "prepared bootstrap retired its original drain obligation");
        bool blocked = false;
        try { (void)owner.attachment->submitLocal(encodeEnvelope(EnvelopeInput{})); }
        catch (OmniUsdLiveResult result) { blocked = result == OMNI_USD_LIVE_BUSY; }
        require(blocked, "prepared bootstrap admitted authoring before activation");
    }
    host.state(owner.attachment, "ready");
    bool timedOutDuringBootstrap = false;
    bool activationWaitChecked = false;
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream input(line); std::string command, extra; input >> command;
        if (command == "retained") {
            uint64_t sequence=0; std::string hex;
            require(prepared && bool(input >> sequence >> hex) && sequence>1 && hex.size()%2==0 && hex.size()<=131072,
                    "invalid prepared-peer barrier");
            const auto nibble=[](char c)->unsigned {
                if (c>='0' && c<='9') return unsigned(c-'0');
                if (c>='a' && c<='f') return unsigned(c-'a'+10);
                throw std::runtime_error("invalid peer payload hex");
            };
            std::string expected; expected.reserve(hex.size()/2);
            for (size_t i=0;i<hex.size();i+=2) expected.push_back(char((nibble(hex[i])<<4)|nibble(hex[i+1])));
            const auto until=Clock::now()+std::chrono::seconds(5);
            while (owner.attachment->progress().covered<sequence) {
                require(owner.attachment->progress().result==OMNI_USD_LIVE_OK && Clock::now()<until,
                        "peer receipt did not reach prepared attachment");
                std::this_thread::yield();
            }
            require(owner.attachment->progress().applied==0,"peer barrier ran after bootstrap activation");
            const auto retained=owner.attachment->retained();
            const auto peer=std::find_if(retained.begin(),retained.end(),[&](const auto& work) {
                return work.first==sequence && work.last==sequence;
            });
            require(retained.size()==2 && peer!=retained.end() && !peer->applied &&
                    peer->disposition==LayerProgress::Disposition::Peer && peer->kind==LayerProgress::CoverageKind::Update,
                    "prepared attachment did not retain the exact peer receipt");
            const auto effect=std::static_pointer_cast<const LayerActor::Effect>(peer->owner);
            require(effect->incoming && effect->incoming->header.sequence_num()==sequence && effect->bytes()==expected,
                    "retained peer bytes differ from independent reference payload");
            const auto bootstrap=std::find_if(retained.begin(),retained.end(),[&](const auto& work) { return work.id==bootstrapId; });
            require(bootstrap!=retained.end() && !bootstrap->applied && bootstrap->owner==bootstrapOwner.lock() &&
                    std::static_pointer_cast<const LayerActor::Effect>(bootstrap->owner)->bytes()==bootstrapBytes,
                    "queued peer replaced the original bootstrap receipt");
            host.state(owner.attachment,"retained");
        }
        else if (command == "activate-close" || command == "activate-timeout") {
            host.armBootstrap();
            auto interrupt = std::async(std::launch::async, [&] {
                host.bootstrapEntered();
                if (command == "activate-close")
                    require(owner.attachment->close(0) == OMNI_USD_LIVE_TIMEOUT, "close escaped active bootstrap publication");
                else {
                    // The bootstrap is held in the host: the deadline passes
                    // while its publication is running. Failure-only bound
                    // below the ordinary deadline, so only the seam reaches it.
                    owner.attachment->expireActivationForTest();
                    const auto until = Clock::now() + std::chrono::seconds(3);
                    while (owner.attachment->progress().result != OMNI_USD_LIVE_TIMEOUT) {
                        require(Clock::now() < until, "bootstrap publication did not reach its deadline");
                        std::this_thread::yield();
                    }
                }
                host.releaseBootstrap();
            });
            const auto result = owner.attachment->activatePublication();
            interrupt.get();
            require(result == (command == "activate-close" ? OMNI_USD_LIVE_CANCELLED : OMNI_USD_LIVE_TIMEOUT),
                    "interrupted activation incorrectly reported success");
            timedOutDuringBootstrap = command == "activate-timeout";
            SceneAttachment::PublicationProgress timedOut;
            Fields actualHost;
            if (timedOutDuringBootstrap) {
                std::tie(timedOut, actualHost) = host.publicationFields(*owner.attachment);
                require(timedOut.hasCompleted && timedOut.hasAttempt &&
                        timedOut.lastAttemptedRevision == timedOut.lastCompletedRevision &&
                        timedOut.lastCompletedRevision == owner.attachment->snapshot()->revision &&
                        timedOut.result == OMNI_USD_LIVE_TIMEOUT &&
                        owner.attachment->progress().result == OMNI_USD_LIVE_TIMEOUT &&
                        owner.attachment->progress().applied == 0,
                        "successful callback return was confused with timely bootstrap application");
            }
            std::cout << "{\"type\":\"activated\",\"result\":" << result;
            if (timedOutDuringBootstrap) publicationJson(timedOut, actualHost);
            std::cout << '}' << std::endl;
        }
        else if (command == "activate") {
            std::future<void> waiterCheck;
            if (!activationWaitChecked && std::string(argv[4]) == "--prepared-publication") {
                activationWaitChecked = true;
                host.armBootstrap();
                waiterCheck = std::async(std::launch::async, [&] {
                    try {
                        host.bootstrapEntered(); // Publication is running, not merely staged.
                        const auto readiness = getRuntime()->diagnostics->snapshot(0, 0).metrics.readiness[1];
                        require(readiness.initialWaiting == 1 && !readiness.initialByResolution[0].count,
                                "held host publication prematurely completed readiness timing");
                        std::promise<void> entered;
                        auto entry = entered.get_future();
                        auto waiter = std::async(std::launch::async, [&] {
                            require(owner.attachment->waitForPrepared(0) == OMNI_USD_LIVE_TIMEOUT,
                                    "activation waiter did not observe held publication");
                            entered.set_value();
                            return owner.attachment->waitForPrepared(3000);
                        });
                        const bool began = entry.wait_for(std::chrono::seconds(1)) == std::future_status::ready;
                        // The host gate is closed: prepared is still false at the last held instant.
                        require(owner.attachment->waitForPrepared(0) == OMNI_USD_LIVE_TIMEOUT,
                                "activation completed through the held host publication gate");
                        host.releaseBootstrap();
                        const bool woke = waiter.wait_for(std::chrono::seconds(1)) == std::future_status::ready;
                        const auto result = waiter.get(); // Bounded even on the regression path.
                        require(began, "activation waiter did not start while the host publication gate held");
                        require(woke && result == OMNI_USD_LIVE_OK,
                                "prepared waiter missed successful activation notification");
                    } catch (...) { host.releaseBootstrap(); throw; }
                });
            }
            const auto result = owner.attachment->activatePublication();
            if (waiterCheck.valid()) waiterCheck.get();
            if (result == OMNI_USD_LIVE_OK) {
                owner.waitReady(uri); // Activation precedes asynchronous actor/status publication.
                const auto readiness = getRuntime()->diagnostics->snapshot(0, 0).metrics.readiness[1];
                require(!readiness.initialWaiting && readiness.initialByResolution[0].count == 1,
                        "successful host activation did not complete readiness timing exactly once");
            }
            std::cout << "{\"type\":\"activated\",\"result\":" << result << '}' << std::endl;
        }
        else if (command == "state") host.state(owner.attachment, "state");
        else if (command == "arm") { host.arm(); host.state(owner.attachment, "armed"); }
        else if (command == "held" || command == "release-notice") {
            if (command == "held") host.held(); else host.releaseNotice();
            require(omniUsdLiveWaitForPendingUpdates(20) == OMNI_USD_LIVE_TIMEOUT,
                    "drain escaped a held USD notice or host job capture");
            host.state(owner.attachment, command.c_str());
        } else if (command == "release-job") { host.releaseJob(); host.state(owner.attachment, "released"); }
        else if (command == "wait" || command == "failed") {
            uint64_t target = 0; if (command == "wait") require(bool(input >> target), "missing publication target");
            // An unactivated prepared bootstrap never applies; the seam passes
            // its deadline, within a failure-only bound below the ordinary one.
            if (command == "failed" && expireByTest) owner.attachment->expireActivationForTest();
            const auto until = Clock::now() + std::chrono::seconds(command == "failed" && expireByTest ? 3 : 5);
            for (;;) {
                const auto state = owner.attachment->progress();
                if (command == "failed" ? state.result != OMNI_USD_LIVE_OK : state.applied >= target) break;
                require(Clock::now() < until, "async application did not settle"); std::this_thread::yield();
            }
            if (command == "wait") require(omniUsdLiveWaitForPendingUpdates(1000) == OMNI_USD_LIVE_OK, "published work did not drain");
            host.state(owner.attachment, "state");
        } else if (command == "set") {
            std::string key; int value; require(bool(input >> key >> value), "invalid local publication edit");
            EnvelopeInput metadata; metadata.fields = {{1, 0, key, {}}};
            require(owner.attachment->submitLocal(encodeUsdEnvelope(metadata, DecodedValues{{VtValue(value)}, {}})) != 0,
                    "local publication edit not admitted");
            host.state(owner.attachment, "accepted");
        } else if (command == "fail-next") { host.failNext(); host.state(owner.attachment, "armed"); }
        else if (command == "fail-before-next") { host.failBeforeNext(); host.state(owner.attachment, "armed"); }
        else if (command == "close") {
            require(owner.attachment->close(5000) == OMNI_USD_LIVE_OK, "async close did not quiesce");
            if (timedOutDuringBootstrap) {
                const auto progress = owner.attachment->progress();
                const auto retained = owner.attachment->retained();
                require(progress.result == OMNI_USD_LIVE_TIMEOUT && progress.applied == 0 && retained.size() == 1,
                        "timed-out running bootstrap falsely retired original actor work");
                require(owner.attachment->snapshot()->applied == 1 && retained.front().id == bootstrapId &&
                        !retained.front().applied && retained.front().owner == bootstrapOwner.lock() &&
                        std::static_pointer_cast<const LayerActor::Effect>(retained.front().owner)->bytes() == bootstrapBytes,
                        "timed-out running bootstrap lost candidate or original receipt bytes/owner");
            }
            host.state(owner.attachment, "closed");
            owner.close(); return 0;
        } else throw std::runtime_error("unknown publication command");
        require(!(input >> extra), "unexpected publication command argument");
    }
    owner.close(); return 0;
}
