// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "SceneAttachment.h"
#include "LiveLayerAdmission.h"
#include "LiveDeadline.h"
#include "LiveExceptionResult.h"
#include <algorithm>
#include <atomic>
#include <condition_variable>

namespace omni::usd::live {
namespace {
thread_local bool localPublication = false;
}
LocalPublicationScope::LocalPublicationScope() noexcept : previous_(localPublication) { localPublication = true; }
LocalPublicationScope::~LocalPublicationScope() { localPublication = previous_; }
bool LocalPublicationScope::active() noexcept { return localPublication; }
namespace {
std::atomic<void (*)(void*) noexcept> beforeOpenWait{nullptr};
std::atomic<void*> beforeOpenWaitContext{nullptr};
OmniUsdLiveResult decodeResult(const wire::DecodeError& error, bool incoming) {
    if (error.kind == wire::ErrorKind::ResourceLimit) return OMNI_USD_LIVE_RESOURCE_EXHAUSTED;
    if (error.kind == wire::ErrorKind::Unsupported) return OMNI_USD_LIVE_UNSUPPORTED;
    return incoming ? OMNI_USD_LIVE_MALFORMED_STREAM : OMNI_USD_LIVE_INVALID_ARGUMENT;
}
struct PublicationTask {
    // Reverse member destruction retires every scene/host capture before the
    // completion owner releases the actor's original drain/close obligation.
    std::shared_ptr<LayerActor::ApplyCompletion> completion;
    std::shared_ptr<SceneJournal> journal;
    // One local publication, or ordered work in application order.
    std::vector<LayerProgress::Work> work;
    std::vector<std::shared_ptr<const LayerActor::Effect>> effect;
    std::function<void(SceneJournal::SnapshotPtr)> publish;
    SceneJournal::SnapshotPtr prepared;
    PublicationTask(std::shared_ptr<LayerActor::ApplyCompletion> completion, std::shared_ptr<SceneJournal> journal,
        std::vector<LayerProgress::Work> work, std::vector<std::shared_ptr<const LayerActor::Effect>> effect,
        std::function<void(SceneJournal::SnapshotPtr)> publish)
        : completion(std::move(completion)), journal(std::move(journal)), work(std::move(work)),
          effect(std::move(effect)), publish(std::move(publish)) {}
    ~PublicationTask() {
        DispatchExecutor::NonblockingScope scope;
        publish = {};
        prepared.reset();
        effect.clear();
        work.clear();
        journal.reset();
        completion.reset();
    }
    // Applies the ordered work to the journal as one transaction.
    SceneJournal::SnapshotPtr apply() const {
        std::vector<SceneJournal::Application> batch;
        batch.reserve(work.size());
        for (size_t i = 0; i < work.size(); ++i) batch.push_back({work[i], effect[i]->bytes(), effect[i]->intent});
        journal->applyEmpty(work.front().first - 1);
        return journal->apply(batch);
    }
    OmniUsdLiveResult run() noexcept {
        auto result = OmniUsdLiveResult(OMNI_USD_LIVE_CANCELLED);
        try {
            completion->run([this, &result] {
                try {
                    const auto committed = prepared ? prepared : apply();
                    publish(committed);
                    result = OMNI_USD_LIVE_OK;
                } catch (...) {
                    result = currentExceptionResult();
                    throw result;
                }
            });
        } catch (...) {
            result = currentExceptionResult();
            completion->complete(result);
        }
        return result;
    }
};
}

struct SceneAttachment::PublicationState : std::enable_shared_from_this<PublicationState> {
    enum class Phase { Waiting, Prepared, Activating, Active, Failed, Closed };
    const Publication callbacks;
    const bool deferred;
    mutable std::mutex mutex;
    std::condition_variable changed;
    Phase phase;
    OmniUsdLiveResult result = OMNI_USD_LIVE_OK;
    std::shared_ptr<PublicationTask> bootstrap;
    std::chrono::steady_clock::time_point deadline;
    uint64_t publishedRevision = 0;
    bool published = false, publishing = false;
    uint64_t attemptedRevision = 0;
    bool attempted = false;

    PublicationState(Publication callbacks, bool deferred, std::chrono::steady_clock::time_point deadline)
        : callbacks(std::move(callbacks)), deferred(deferred), phase(deferred ? Phase::Waiting : Phase::Active),
          deadline(deadline) {}
    bool preparing() const {
        std::lock_guard<std::mutex> lock(mutex);
        return phase == Phase::Waiting;
    }
    bool prepared() const {
        std::lock_guard<std::mutex> lock(mutex);
        return phase == Phase::Prepared;
    }
    void requireAuthoring() const {
        std::lock_guard<std::mutex> lock(mutex);
        if (phase == Phase::Failed) throw result;
        if (phase == Phase::Closed) throw OmniUsdLiveResult(OMNI_USD_LIVE_CANCELLED);
        if (phase != Phase::Active) throw OmniUsdLiveResult(OMNI_USD_LIVE_BUSY);
    }
    void stage(std::shared_ptr<PublicationTask> task) {
        std::lock_guard<std::mutex> lock(mutex);
        if (phase != Phase::Waiting) throw OmniUsdLiveResult(OMNI_USD_LIVE_CANCELLED);
        bootstrap = std::move(task);
        phase = Phase::Prepared;
        changed.notify_all();
    }
    void publish(SceneJournal::SnapshotPtr scene) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (phase == Phase::Closed) throw OmniUsdLiveResult(OMNI_USD_LIVE_CANCELLED);
            if (phase == Phase::Failed) throw result;
            if (phase != Phase::Active && phase != Phase::Activating) throw OmniUsdLiveResult(OMNI_USD_LIVE_BUSY);
            if (publishing) throw OmniUsdLiveResult(OMNI_USD_LIVE_BUSY);
            if (published && scene->revision <= publishedRevision) return;
            publishing = true;
            attemptedRevision = scene->revision;
            attempted = true;
        }
        try {
            callbacks.publish(scene);
        } catch (...) {
            const auto error = currentExceptionResult();
            std::lock_guard<std::mutex> lock(mutex);
            publishing = false;
            if (phase != Phase::Closed) {
                phase = Phase::Failed;
                result = error;
            }
            throw error;
        }
        std::lock_guard<std::mutex> lock(mutex);
        // Record a successful callback return even if a concurrent terminal
        // transition or activation deadline prevents successful application.
        publishedRevision = scene->revision;
        published = true;
        publishing = false;
        // A worker may have failed activation while this host callback was
        // running. Preserve that result on the original receipt even though
        // USD already contains the candidate. Closing an otherwise successful
        // running callback still permits its applied bookkeeping to complete.
        if (phase == Phase::Failed || (phase == Phase::Closed && result != OMNI_USD_LIVE_CANCELLED)) throw result;
        if (phase == Phase::Activating && std::chrono::steady_clock::now() >= deadline) {
            phase = Phase::Failed;
            result = OMNI_USD_LIVE_TIMEOUT;
            // USD may already contain the candidate. Fail the original receipt
            // rather than claiming timely activation or rolling anything back.
            throw OmniUsdLiveResult(OMNI_USD_LIVE_TIMEOUT);
        }
    }
    std::function<void(SceneJournal::SnapshotPtr)> publisher() {
        return [weak = std::weak_ptr<PublicationState>(shared_from_this())](auto scene) {
            const auto owner = weak.lock();
            if (!owner) throw OmniUsdLiveResult(OMNI_USD_LIVE_CANCELLED);
            owner->publish(std::move(scene));
        };
    }
    void stop(OmniUsdLiveResult error, bool closing = false) {
        std::shared_ptr<PublicationTask> abandoned;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (phase != Phase::Closed) {
                if (phase != Phase::Failed) result = error;
                phase = closing ? Phase::Closed : Phase::Failed;
            }
            abandoned = std::move(bootstrap);
            changed.notify_all();
        }
        // Actor callbacks retain this coordinator; explicitly abandon dormant
        // handoff ownership so close/failure cannot wait for its own captures.
        if (abandoned) abandoned->completion->complete(error);
    }
    OmniUsdLiveResult activate() noexcept {
        std::shared_ptr<PublicationTask> task;
        try {
            // The deadline is read once, under the mutex that expireActivationForTest() writes it under.
            bool expired = false;
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (!deferred) return OMNI_USD_LIVE_INVALID_ARGUMENT;
                if (phase == Phase::Active) return OMNI_USD_LIVE_OK;
                if (phase == Phase::Closed) return OMNI_USD_LIVE_CANCELLED;
                if (phase == Phase::Failed) return result;
                if (phase != Phase::Prepared) return OMNI_USD_LIVE_BUSY;
                task = std::move(bootstrap);
                expired = std::chrono::steady_clock::now() >= deadline;
                if (expired) {
                    phase = Phase::Failed;
                    result = OMNI_USD_LIVE_TIMEOUT;
                } else phase = Phase::Activating;
            }
            if (expired) {
                task->completion->complete(OMNI_USD_LIVE_TIMEOUT);
                stop(OMNI_USD_LIVE_TIMEOUT);
                return OMNI_USD_LIVE_TIMEOUT;
            }
            const auto outcome = task->run();
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (phase == Phase::Activating) {
                    phase = outcome == OMNI_USD_LIVE_OK ? Phase::Active : Phase::Failed;
                    result = outcome;
                }
                changed.notify_all();
                if (phase == Phase::Closed) return OMNI_USD_LIVE_CANCELLED;
                if (phase == Phase::Failed) return result;
            }
            return outcome; // Task teardown precedes receipt retirement.
        } catch (...) {
            const auto error = currentExceptionResult();
            if (task) task->completion->complete(error);
            stop(error);
            return error;
        }
    }
};
LayerActor::Decoded decodeSceneCoverage(std::string_view bytes, uint64_t sequence, bool firstResponse,
    const WireSceneLimits& limits) {
    if (!sequence) throw OmniUsdLiveResult(OMNI_USD_LIVE_MALFORMED_STREAM);
    try {
        const auto envelope = wire::decodeEnvelope(bytes, limits.envelope);
        (void)wire::decodeValues(envelope, limits.values);
        LayerActor::Decoded decoded;
        decoded.first = sequence;
        if (envelope.isDiff) {
            if (envelope.baseVersion >= sequence) throw OmniUsdLiveResult(OMNI_USD_LIVE_MALFORMED_STREAM);
            decoded.first = envelope.baseVersion + 1; // Above comparison rules out overflow.
            decoded.kind = envelope.baseVersion ? LayerProgress::CoverageKind::Aggregate :
                                                  LayerProgress::CoverageKind::Snapshot;
        }
        decoded.recoveryBoundary = firstResponse;
        return decoded;
    } catch (const wire::DecodeError& error) { throw decodeResult(error, true); }
}

std::shared_ptr<SceneAttachment> SceneAttachment::open(std::shared_ptr<Runtime> runtime, OmniUsdLiveEndpoint endpoint,
    Options options, Changed changed) {
    if (!runtime) throw OmniUsdLiveResult(OMNI_USD_LIVE_NOT_INITIALIZED);
    if (bool(options.publication.dispatch) != bool(options.publication.publish))
        throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
    if (options.deferInitialPublication &&
        (!options.publication.dispatch || options.resume || options.initial ||
            (options.discover ? options.object != 0 : options.object == 0)))
        throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
    auto journal = std::make_shared<SceneJournal>(std::move(options.initial), options.resume, options.limits);
    auto attachment = std::shared_ptr<SceneAttachment>(new SceneAttachment(journal));
    LayerActor::Options actorOptions;
    actorOptions.session = std::move(options.session);
    actorOptions.layer = std::move(options.layer);
    actorOptions.object = options.object;
    actorOptions.resume = options.resume;
    actorOptions.admission = std::move(options.admission);
    actorOptions.discover = options.discover;
    actorOptions.status = std::move(options.status);
    actorOptions.holdOpening = options.holdOpening;
    actorOptions.hostFailure = options.publication.hostFailure;
    actorOptions.bufferedEdits = options.publication.bufferedEdits;
    actorOptions.initialState = journal->snapshot();
    LayerActor::Callbacks callbacks;
    callbacks.decode = [limits = options.limits.scene](const LayerReadUpdate& update) {
        return decodeSceneCoverage(update.bytes, update.header.sequence_num(), update.firstResponse, limits);
    };
    if (options.publication.dispatch) {
        auto publication = std::make_shared<PublicationState>(std::move(options.publication),
            options.deferInitialPublication,
            std::chrono::steady_clock::now() +
                std::chrono::milliseconds(runtime->options.values.initial_open_timeout_ms));
        attachment->publication_ = publication;
        // One publication installs a batch of ordered work (design section 12).
        callbacks.applyBatchAsync = [journal, publication](std::vector<LayerProgress::Work> work,
                                        std::vector<std::shared_ptr<const LayerActor::Effect>> effect,
                                        std::shared_ptr<LayerActor::ApplyCompletion> completion) {
            auto task = std::make_shared<PublicationTask>(std::move(completion), journal, std::move(work),
                std::move(effect), publication->publisher());
            if (publication->preparing()) {
                // Only native candidate preparation runs here. The original
                // host completion remains queued and owns the bootstrap ticket.
                task->prepared = task->apply();
                publication->stage(std::move(task));
            } else publication->callbacks.dispatch([task = std::move(task)] { task->run(); });
        };
        callbacks.localAsync = [journal, publication](LayerProgress::Work work,
                                   std::shared_ptr<const LayerActor::Effect> effect,
                                   std::shared_ptr<LayerActor::ApplyCompletion> completion) {
            const auto id = work.id;
            const auto intent = effect->intent;
            auto task = std::make_shared<PublicationTask>(std::move(completion), journal,
                std::vector<LayerProgress::Work>{std::move(work)},
                std::vector<std::shared_ptr<const LayerActor::Effect>>{std::move(effect)}, publication->publisher());
            task->prepared = journal->snapshot();
            const auto& pending = task->prepared->pending;
            const auto found = std::find_if(pending.begin(), pending.end(), [&](const auto& entry) {
                return entry.id == id && entry.intent.get() == intent.get() && !entry.intent.owner_before(intent) &&
                    !intent.owner_before(entry.intent);
            });
            if (found == pending.end()) throw OmniUsdLiveResult(OMNI_USD_LIVE_INTERNAL_ERROR);
            // One current owner can include later accepted edits. Their own
            // original work/tickets still require individual publication/ACK.
            LocalPublicationScope local;
            publication->callbacks.dispatch([task = std::move(task)] { task->run(); });
        };
    } else {
        callbacks.apply = [journal](const LayerProgress::Work& work, const LayerActor::Effect& effect) {
            journal->applyEmpty(work.first - 1);
            journal->apply(work, effect.bytes(), effect.intent);
        };
    }
    // After an issuance-only FAILED the journal keeps receiving; nothing
    // installs into the host layer. A publication that failed after its own
    // journal application already holds that work.
    callbacks.applyDetached = [journal](const LayerProgress::Work& work, const LayerActor::Effect& effect) {
        if (journal->snapshot()->applied >= work.last) return;
        journal->applyEmpty(work.first - 1);
        journal->apply(work, effect.bytes(), effect.intent);
    };
    callbacks.changed = [journal, publication = attachment->publication_, changed = std::move(changed)](
                            const auto& state) {
        if (publication && (state.gate == LayerProgress::Gate::Failed)) publication->stop(state.result);
        journal->applyEmpty(state.applied);
        if (changed) changed(state);
    };
    // Captures own only the journal, never the attachment/actor: no owner cycle.
    // Callbacks may start before open returns and do not depend on actor_ assignment.
    attachment->actor_ = LayerActor::open(std::move(runtime), endpoint, std::move(actorOptions), std::move(callbacks));
    return attachment;
}

std::shared_ptr<SceneAttachment> SceneAttachment::openExisting(std::shared_ptr<Runtime> runtime, const std::string& uri,
    SceneJournal::Limits limits, Changed changed, Publication publication) {
    return openExistingImpl(std::move(runtime), uri, std::move(limits), std::move(changed), std::move(publication),
        false);
}

std::shared_ptr<SceneAttachment> SceneAttachment::prepareExisting(std::shared_ptr<Runtime> runtime,
    const std::string& uri, Publication publication, SceneJournal::Limits limits, Changed changed) {
    return openExistingImpl(std::move(runtime), uri, std::move(limits), std::move(changed), std::move(publication),
        true);
}

std::shared_ptr<SceneAttachment> SceneAttachment::openExistingImpl(std::shared_ptr<Runtime> runtime,
    const std::string& uri, SceneJournal::Limits limits, Changed changed, Publication publication, bool deferred) {
    if (!runtime) throw OmniUsdLiveResult(OMNI_USD_LIVE_NOT_INITIALIZED);
    if (DispatchExecutor::isWorker()) throw OmniUsdLiveResult(OMNI_USD_LIVE_BUSY);
    const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::milliseconds(runtime->options.values.initial_open_timeout_ms);
    const auto parsed = parseUri(uri);
    if (parsed.layerName.size() < 5 || parsed.layerName.compare(parsed.layerName.size() - 5, 5, ".live"))
        throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
    OmniUsdLiveEndpoint endpoint = 0;
    {
        std::lock_guard<std::mutex> lock(runtime->mutex);
        if (runtime->closing) throw OmniUsdLiveResult(OMNI_USD_LIVE_BUSY);
        const auto found = runtime->authorities.find(parsed.authority.canonical);
        if (found == runtime->authorities.end()) throw OmniUsdLiveResult(OMNI_USD_LIVE_NOT_FOUND);
        endpoint = found->second;
    }
    Options options;
    // A closing backing of this resource is waited for within the open deadline.
    options.admission = runtime->layerAdmissions->reserve(endpoint, parsed.sessionId, parsed.layerName, deadline);
    options.session = parsed.sessionId;
    options.layer = parsed.layerName;
    options.limits = std::move(limits);
    options.discover = true;
    options.publication = std::move(publication);
    options.deferInitialPublication = deferred;
    struct Opening {
        std::mutex mutex;
        std::condition_variable changed;
        bool ready = false, failed = false, abandoned = false;
        OmniUsdLiveResult result = OMNI_USD_LIVE_OK;
        SceneAttachment::Changed callback;
    };
    auto opening = std::make_shared<Opening>();
    opening->callback = std::move(changed);
    auto attachment = open(runtime, endpoint, std::move(options), [opening](const auto& state) {
        bool deliver;
        {
            std::lock_guard<std::mutex> lock(opening->mutex);
            if (!opening->ready && !opening->failed && !opening->abandoned) {
                // Capture bootstrap completion once. Later peer arrivals must
                // not keep extending the synchronous open's work boundary.
                opening->ready = state.gate == LayerProgress::Gate::Ready && state.object &&
                    state.initialLatestPublished && state.applied > 0;
                opening->failed = state.gate == LayerProgress::Gate::Failed;
                opening->result = state.result;
            }
            deliver = opening->ready && !opening->abandoned;
            opening->changed.notify_all();
        }
        if (deliver && opening->callback) opening->callback(state);
    });
    if (attachment->publication_) {
        std::lock_guard<std::mutex> stateLock(attachment->publication_->mutex);
        attachment->publication_->deadline = deadline;
    }
    if (const auto hook = beforeOpenWait.load(std::memory_order_acquire))
        hook(beforeOpenWaitContext.load(std::memory_order_acquire));
    if (deferred) {
        const auto& state = attachment->publication_;
        std::unique_lock<std::mutex> stateLock(state->mutex);
        const bool done = state->changed.wait_until(stateLock, deadline, [&] {
            return state->phase == PublicationState::Phase::Prepared ||
                state->phase == PublicationState::Phase::Failed || state->phase == PublicationState::Phase::Closed;
        });
        if (!done || state->phase != PublicationState::Phase::Prepared) {
            const auto error = done ?
                state->result :
                expiredWait(std::chrono::steady_clock::now(), deadline, attachment->actor_->credentialStarved());
            stateLock.unlock();
            {
                std::lock_guard<std::mutex> guard(opening->mutex);
                opening->abandoned = true;
            }
            // This owner is the only one ever holding this failed attachment,
            // so its close marks the admission retiring at once: an immediate
            // retry then waits for it instead of failing with ALREADY_EXISTS.
            (void)attachment->close(0, true);
            throw error;
        }
        return attachment;
    }
    std::unique_lock<std::mutex> lock(opening->mutex);
    const bool finished = opening->changed.wait_until(lock, deadline,
        [&] { return opening->ready || opening->failed; });
    if (!finished || opening->failed) {
        const auto result = finished ?
            opening->result :
            expiredWait(std::chrono::steady_clock::now(), deadline, attachment->actor_->credentialStarved());
        opening->abandoned = true;
        lock.unlock();
        // This owner is the only one ever holding this failed attachment, so
        // its close marks the admission retiring at once: an immediate retry
        // then waits for it instead of failing with ALREADY_EXISTS.
        (void)attachment->close(0, true);
        throw result;
    }
    return attachment;
}

SceneAttachment::~SceneAttachment() {
    if (actor_) (void)close(0); // Scheduler retains in-flight cleanup; never wait on its worker.
}

OmniUsdLiveResult SceneAttachment::waitForPrepared(uint32_t timeoutMs) noexcept {
    if (!publication_ || !publication_->deferred) return OMNI_USD_LIVE_INVALID_ARGUMENT;
    if (DispatchExecutor::isWorker()) return OMNI_USD_LIVE_BUSY;
    try {
        const auto& state = publication_;
        std::unique_lock<std::mutex> lock(state->mutex);
        const auto limit = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
        const auto settled = [&] {
            return state->phase == PublicationState::Phase::Prepared ||
                state->phase == PublicationState::Phase::Active || state->phase == PublicationState::Phase::Failed ||
                state->phase == PublicationState::Phase::Closed;
        };
        // The activation deadline is re-read on every wake: it can only move earlier.
        while (!settled()) {
            const auto deadline = std::min(state->deadline, limit);
            if (std::chrono::steady_clock::now() >= deadline) return OMNI_USD_LIVE_TIMEOUT;
            state->changed.wait_until(lock, deadline);
        }
        if (state->phase == PublicationState::Phase::Failed || state->phase == PublicationState::Phase::Closed)
            return state->result;
        if (state->phase != PublicationState::Phase::Active && std::chrono::steady_clock::now() >= state->deadline)
            return OMNI_USD_LIVE_TIMEOUT;
        lock.unlock();
        const auto progress = actor_->snapshot();
        if (progress.gate == LayerProgress::Gate::Failed) {
            state->stop(progress.result);
            return progress.result;
        }
        return OMNI_USD_LIVE_OK;
    } catch (...) { return currentExceptionResult(); }
}

void SceneAttachment::setBeforeOpenWaitForTest(void (*hook)(void*) noexcept, void* context) noexcept {
    beforeOpenWaitContext.store(context, std::memory_order_release);
    beforeOpenWait.store(hook, std::memory_order_release);
}

void SceneAttachment::expireActivationForTest() noexcept {
    if (publication_) {
        std::lock_guard<std::mutex> lock(publication_->mutex);
        publication_->deadline = std::min(publication_->deadline, std::chrono::steady_clock::now());
        publication_->changed.notify_all();
    }
    if (actor_) actor_->expireOpenForTest();
}

void SceneAttachment::expireReadForTest() noexcept {
    if (actor_) actor_->expireReadForTest();
}

OmniUsdLiveResult SceneAttachment::activatePublication() noexcept {
    if (!publication_) return OMNI_USD_LIVE_INVALID_ARGUMENT;
    const auto result = publication_->activate();
    if (result != OMNI_USD_LIVE_OK) return result;
    // Terminal progress can precede Changed delivery to the coordinator.
    // Do not return success merely because that callback has not run yet.
    const auto state = actor_->snapshot();
    if (state.gate == LayerProgress::Gate::Failed) {
        publication_->stop(state.result);
        return state.result;
    }
    return OMNI_USD_LIVE_OK;
}

void SceneAttachment::failPublication(OmniUsdLiveResult result) noexcept {
    actor_->failPublication(result);
    if (publication_) publication_->stop(result, false);
}

OmniUsdLiveResult SceneAttachment::close(uint32_t timeoutMs, bool released) {
    if (publication_) publication_->stop(OMNI_USD_LIVE_CANCELLED, true);
    if (released) actor_->markAdmissionRetiring();
    return actor_->close(timeoutMs);
}

OmniUsdLiveResult SceneAttachment::save(uint32_t timeoutMs, bool allowBlock) {
    return actor_->save(timeoutMs, allowBlock);
}

SceneAttachment::PublicationProgress SceneAttachment::publicationProgress() const {
    if (!publication_) return {};
    std::lock_guard<std::mutex> lock(publication_->mutex);
    return {true, publication_->attempted, publication_->published, publication_->attemptedRevision,
        publication_->publishedRevision, publication_->result};
}

void SceneAttachment::requireAuthoring() const {
    if (publication_) publication_->requireAuthoring();
    const auto result = actor_->authoringResult();
    if (result != OMNI_USD_LIVE_OK) throw result;
}
uint64_t SceneAttachment::submitLocal(std::string_view bytes) {
    // Journal and actor locks are held while an admission unwinds; report
    // any reservation release to waiting layers after they are released.
    PendingBytes::DeferredReleaseNotices notices;
    try {
        requireAuthoring();
        return journal_->submitLocal(bytes, [this](auto original, auto intent, auto publication) {
            const auto retainedBytes = intent->retainedBytes;
            return actor_->submit(original, std::move(intent), publication, retainedBytes);
        });
    } catch (const wire::DecodeError& error) { throw decodeResult(error, false); }
}

void SceneAttachment::probeCaptured(const WireScenePtr& base,
    const std::shared_ptr<const SdfOperationLog>& sealed) const {
    try {
        if (!actor_->updateCanFit(journal_->probeCaptured(base, sealed)))
            throw CapturedRejection{OMNI_USD_LIVE_RESOURCE_EXHAUSTED};
    } catch (const UnsupportedOperation&) { throw; } catch (const DeltaEntryLimitExceeded&) {
        throw;
    } catch (const wire::DecodeError& error) { throw decodeResult(error, false); }
}

uint64_t SceneAttachment::submitCaptured(const WireScenePtr& base, const std::shared_ptr<const SdfOperationLog>& sealed,
    WireScenePtr* applied) {
    // Journal and actor locks are held while an admission unwinds; report
    // any reservation release to waiting layers after they are released.
    PendingBytes::DeferredReleaseNotices notices;
    try {
        requireAuthoring();
        return journal_->submitCaptured(base, sealed, [this](auto original, auto intent, auto publication) {
            if (!actor_->updateCanFit(original.size())) throw CapturedRejection{OMNI_USD_LIVE_RESOURCE_EXHAUSTED};
            const auto retainedBytes = intent->retainedBytes;
            return actor_->submit(original, std::move(intent), publication, retainedBytes);
        }, applied);
    } catch (const UnsupportedOperation&) {
        throw; // The flush names the operation; its result is UNSUPPORTED.
    } catch (const DeltaEntryLimitExceeded&) {
        throw; // The flush names the counts; its result is RESOURCE_EXHAUSTED.
    } catch (const wire::DecodeError& error) { throw decodeResult(error, false); }
}
} // namespace omni::usd::live
