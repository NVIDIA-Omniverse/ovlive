// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "SceneJournal.h"
#include "OmpfEncode.h"
#include "UsdValueWriter.h"
#include <pxr/usd/sdf/schema.h>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace omni::usd::live {
namespace {
void charge(size_t amount, size_t& remaining) {
    if (amount > remaining) throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    remaining -= amount;
}
void healthy(const SceneJournal::Snapshot& state) {
    if (state.submissionFault != OMNI_USD_LIVE_OK) throw state.submissionFault;
}
// A retained scene's graph and data representations, measured conservatively,
// including compound value elements, without allocating an encoded scene. SDK
// caches/List scratch are not RSS bounds.
size_t rawBaseCharge(const WireScene& scene, const WireSceneLimits& limits, size_t available,
    WireSceneWorkBudget& work) {
    // Encoded bytes already count a scalar's storage; the per-element charge
    // covers container and value bookkeeping.
    constexpr size_t elementCost = 16;
    const auto original = available;
    wire::detail::EncodeBudget values(limits.retainedValues);
    wire::detail::Writer measure(values);
    auto spend = [&](size_t amount) {
        for (unsigned copy = 0; copy != 2; ++copy) {
            charge(amount, available);
            charge(amount, work.remaining);
        }
    };
    auto value = [&](const PXR_NS::VtValue& item) {
        // Two representation charges, each comprising encoded bytes and
        // per-element bookkeeping, must fit before traversing this value.
        const auto remaining = std::min(available, work.remaining);
        values.encoded = std::min(values.encoded, remaining / 4);
        values.elements = std::min(values.elements, remaining / (4 * elementCost));
        values.strings = std::min(values.strings, values.encoded);
        const auto before = measure.size(), elements = values.elements;
        measure.top(item);
        spend(measure.size() - before);
        spend((elements - values.elements) * elementCost);
    };
    spend(scene.sourceFormat().size());
    size_t fields = limits.maxFields, metadata = limits.maxMetadataBytes, paths = limits.maxPathBytes;
    charge(scene.sourceFormat().size(), metadata);
    scene.visitSections(limits.maxSections, [&](const WireSectionView& section) {
        spend(256);
        charge(section.name.size(), metadata);
        spend(section.name.size());
        const auto valueDepth = values.maxDepth;
        values.maxDepth = limits.maxPathDepth;
        value(PXR_NS::VtValue(section.path));
        values.maxDepth = valueDepth;
        charge(section.path.GetString().size(), paths);
        const auto names = scene.data()->List(section.path);
        charge(names.size(), fields);
        for (const auto& name : names) {
            spend(128);
            charge(name.GetString().size(), metadata);
            value(PXR_NS::VtValue(name));
            value(scene.data()->Get(section.path, name));
        }
        return true;
    });
    size_t changes = limits.envelope.maxRecords;
    charge(scene.fieldChanges().size(), changes);
    for (const auto& change : scene.fieldChanges()) {
        spend(sizeof(WireFieldChange));
        value(PXR_NS::VtValue(change.field));
    }
    return original - available;
}
size_t saturatingAdd(size_t a, size_t b) noexcept {
    return b > std::numeric_limits<size_t>::max() - a ? std::numeric_limits<size_t>::max() : a + b;
}
bool childField(std::string_view name) {
    for (const auto& token : PXR_NS::SdfChildrenKeys->allTokens)
        if (token.GetString() == name) return true;
    return false;
}
// Replays every pending intent tolerantly (a predicted service rejection
// applies none of the intent) onto the authoritative scene in one build
// (buildReplayedScene), and records what each replay did in its entry and
// the snapshot's replay totals.
void replayPending(SceneJournal::Snapshot& state, const SceneJournal::Limits& limits, WireSceneWorkBudget& work) {
    std::vector<std::string_view> intents;
    intents.reserve(state.pending.size());
    for (const auto& entry : state.pending) intents.push_back(entry.intent->bytes);
    std::vector<WireReplayStep> steps;
    state.visible = buildReplayedScene(state.authoritative, intents, limits.scene, &work, false, &steps);
    state.replayWork = state.replayFields = state.replayMetadataBytes = 0;
    for (size_t i = 0; i < steps.size(); ++i) {
        state.pending[i].replayed = steps[i].applied;
        state.pending[i].replayWork = steps[i].work;
        state.replayWork = saturatingAdd(state.replayWork, steps[i].work);
        state.replayFields = std::max(state.replayFields, steps[i].fields);
        state.replayMetadataBytes = std::max(state.replayMetadataBytes, steps[i].metadataBytes);
    }
    state.replayRederives = !steps.empty() && (steps.back().rederives || !steps.back().derived);
    state.derivation = SceneJournal::Derivation::Replayed;
}
// Conservatively retain every referenced section identity, including temporary
// IDs deleted within this same batch. History may grow on rejected preparation;
// scene/frontier publication remains independent and atomic.
void observeIds(SectionIdAllocator& ids, std::string_view bytes, const WireSceneLimits& limits,
    WireSceneWorkBudget& work) {
    charge(bytes.size(), work.remaining);
    auto framing = limits.envelope;
    framing.maxRecords = uint32_t(std::min<size_t>(framing.maxRecords, work.remaining / 128));
    framing.maxStringBytes = std::min(framing.maxStringBytes, work.remaining / 2);
    if (!framing.maxRecords) throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    const auto envelope = wire::decodeEnvelope(bytes, framing);
    for (const auto& command : envelope.commands) {
        charge(128, work.remaining);
        charge(command.name.size(), work.remaining);
        ids.observe(command.section);
        ids.observe(command.parent);
        ids.observe(command.oldParent);
        for (const auto id : command.children) {
            charge(128, work.remaining);
            ids.observe(id);
        }
    }
    for (const auto& field : envelope.fields) {
        charge(128, work.remaining);
        ids.observe(field.section);
    }
    for (const auto& sample : envelope.samples) {
        charge(128, work.remaining);
        ids.observe(sample.section);
    }
}
void observeScene(SectionIdAllocator& ids, const WireScenePtr& scene, WireSceneWorkBudget& work) {
    struct Visitor final : PXR_NS::SdfAbstractDataSpecVisitor {
        SectionIdAllocator& ids;
        const WireScenePtr& scene;
        WireSceneWorkBudget& work;
        Visitor(SectionIdAllocator& i, const WireScenePtr& s, WireSceneWorkBudget& w) : ids(i), scene(s), work(w) {}
        bool VisitSpec(const PXR_NS::SdfAbstractData&, const PXR_NS::SdfPath& path) override {
            charge(128, work.remaining);
            ids.observe(scene->sectionForPath(path));
            return true;
        }
        void Done(const PXR_NS::SdfAbstractData&) override {}
    } visitor(ids, scene, work);
    scene->data()->VisitSpecs(&visitor);
}
// Whether the service would store every record of a compiled captured intent
// as orphan data: it carries only field and sample records, and each addresses
// a section absent from `scene` because a committed peer delete removed it.
bool superseded(std::string_view bytes, const WireScene& scene) {
    const auto envelope = wire::decodeEnvelope(bytes);
    if (!envelope.commands.empty() || (envelope.fields.empty() && envelope.samples.empty())) return false;
    for (const auto& field : envelope.fields)
        if (scene.containsSection(field.section)) return false;
    for (const auto& sample : envelope.samples)
        if (scene.containsSection(sample.section)) return false;
    return true;
}
OmniUsdLiveResult applicationError(const wire::DecodeError& error, bool rebasing) {
    if (error.kind == wire::ErrorKind::ResourceLimit) return OMNI_USD_LIVE_RESOURCE_EXHAUSTED;
    if (rebasing || error.kind == wire::ErrorKind::Unsupported) return OMNI_USD_LIVE_UNSUPPORTED;
    return OMNI_USD_LIVE_MALFORMED_STREAM;
}
} // namespace

size_t measureRetainedScene(const WireScene& scene, const WireSceneLimits& limits, size_t available,
    WireSceneWorkBudget& work) {
    return rawBaseCharge(scene, limits, available, work);
}

std::vector<SceneJournal::Footprint> SceneJournal::footprintOf(std::string_view bytes) const {
    const auto envelope = wire::decodeEnvelope(bytes, limits_.scene.envelope);
    std::vector<Footprint> keys;
    keys.reserve(envelope.fields.size() + envelope.samples.size());
    for (const auto& field : envelope.fields) keys.push_back({field.section, field.name, false});
    for (const auto& sample : envelope.samples) keys.push_back({sample.section, {}, true});
    const FootprintOrder order;
    std::sort(keys.begin(), keys.end(), order);
    keys.erase(std::unique(keys.begin(), keys.end(),
                   [&](const Footprint& a, const Footprint& b) { return !order(a, b) && !order(b, a); }),
        keys.end());
    return keys;
}

SceneJournal::FootprintIndex SceneJournal::index(const std::vector<Pending>& pending) const {
    FootprintIndex result;
    for (const auto& entry : pending)
        if (entry.replayed)
            for (auto& key : footprintOf(entry.intent->bytes)) ++result[std::move(key)];
    return result;
}

void SceneJournal::retire(FootprintIndex& index, const Footprint* begin, const Footprint* end) noexcept {
    for (; begin != end; ++begin) {
        const auto found = index.find(*begin);
        if (found != index.end() && --found->second == 0) index.erase(found);
    }
}

bool SceneJournal::overlay(const Snapshot& previous, Snapshot& prepared, const std::vector<std::string_view>& peers,
    const WireSceneLimits& limits, WireSceneWorkBudget& work) const {
    // A peer update without structural commands, child fields or a source
    // format changes no section's existence, parent, name, order or children,
    // so it changes no pending intent's replay except where that intent
    // writes the peer's targets. The replay applies the peer's records to the
    // authoritative scene before every pending intent, so each record shows
    // in the replayed scene unless:
    // - its section is absent from the authoritative scene, where it is
    //   orphan data, even when a pending intent creates that identity
    //   (identities are never reused, design section 9, so a pending Create
    //   of a section the authoritative scene holds is a predicted rejection);
    // - a pending intent the visible scene applies writes the same field
    //   later, or replaces the section's whole time-sample map (own pending
    //   edits hide peer updates, design section 8).
    // A pending sample record merges with the peer's samples key by key,
    // which skipping no record reproduces, so such a batch replays instead.
    const auto& timeSamples = PXR_NS::SdfFieldKeys->TimeSamples.GetString();
    // An Orphan record reaches no scene; a Hidden record is absent from the
    // visible scene but lies in every replayed scene before the pending
    // intent that hides it.
    enum class Fate { Apply, Hidden, Orphan, Replay };
    const auto fate = [&](uint64_t section, std::string_view field, bool sample) {
        if (!previous.authoritative->containsSection(section)) return Fate::Orphan;
        const auto holds = [&](std::string_view name, bool records) {
            return footprint_.find(FootprintView{section, name, records}) != footprint_.end();
        };
        if (sample || field == timeSamples) {
            if (holds(timeSamples, false)) return Fate::Hidden;
            return holds({}, true) ? Fate::Replay : Fate::Apply;
        }
        return holds(field, false) ? Fate::Hidden : Fate::Apply;
    };
    // Each record that applies or is hidden adds at most one field and its
    // name to every scene the replay passes through, so the previous bounds
    // plus those bound the new replay's scenes, which the field and metadata
    // limits must hold.
    size_t fields = 0, metadata = 0;
    for (const auto bytes : peers) {
        const auto envelope = wire::decodeEnvelope(bytes, limits.envelope);
        if (envelope.isDiff || !envelope.commands.empty() || !envelope.sourceFormat.empty()) return false;
        for (const auto& field : envelope.fields) {
            if (childField(field.name)) return false;
            const auto outcome = fate(field.section, field.name, false);
            if (outcome == Fate::Replay) return false;
            if (outcome != Fate::Orphan) {
                fields = saturatingAdd(fields, 1);
                metadata = saturatingAdd(metadata, field.name.size());
            }
        }
        for (const auto& sample : envelope.samples) {
            const auto outcome = fate(sample.section, {}, true);
            if (outcome == Fate::Replay) return false;
            if (outcome != Fate::Orphan) {
                fields = saturatingAdd(fields, 1);
                metadata = saturatingAdd(metadata, timeSamples.size());
            }
        }
    }
    fields = saturatingAdd(previous.replayFields, fields);
    metadata = saturatingAdd(previous.replayMetadataBytes, metadata);
    if (fields > limits.maxFields || metadata > limits.maxMetadataBytes) return false;
    const std::function<bool(const WireRecordKey&)> skip = [&](const WireRecordKey& key) {
        const auto outcome = fate(key.section, key.field, key.sample);
        return outcome == Fate::Hidden || outcome == Fate::Orphan;
    };
    auto attempt = work;
    WireScenePtr visible;
    try {
        visible = buildOverlaidScene(previous.visible, peers, limits, &attempt, skip);
    } catch (const wire::DecodeError&) { return false; } // The replay decides the outcome.
    prepared.visible = std::move(visible);
    prepared.replayFields = fields;
    prepared.replayMetadataBytes = metadata;
    prepared.derivation = Derivation::Overlaid;
    work = attempt;
    return true;
}

SceneJournal::Limits SceneJournal::effectiveLimits(const WireScene& scene) const {
    // Compiling, admitting and replaying an edit each do work in proportion to
    // the scene, never retaining it. Room for kSceneCopies charges of the
    // scene beside the configured work lets a large layer take as many edits
    // as a small one; the pending-byte and retained-byte quotas bound what
    // intents retain and stay as configured.
    constexpr size_t kSceneCopies = 32;
    const auto charge = scene.retainedCharge();
    const auto extra = charge > std::numeric_limits<size_t>::max() / kSceneCopies ? std::numeric_limits<size_t>::max() :
                                                                                    charge * kSceneCopies;
    const auto grow = [extra](size_t& value) {
        value = extra > std::numeric_limits<size_t>::max() - value ? std::numeric_limits<size_t>::max() : value + extra;
    };
    auto result = limits_;
    grow(result.maxReplayWork);
    grow(result.authoring.maxWork);
    return result;
}

SceneJournal::SceneJournal(WireScenePtr initial, uint64_t applied, Limits limits,
    std::shared_ptr<SectionIdAllocator> ids)
    : limits_(std::move(limits)),
      ids_(ids ?
              std::move(ids) :
              std::make_shared<SectionIdAllocator>(limits_.authoring.maxSectionIds, limits_.authoring.maxIdAttempts)) {
    if (!limits_.maxPending || !limits_.maxPendingBytes || !limits_.maxReplayBytes || !limits_.maxReplayWork)
        throw std::invalid_argument("invalid scene journal limits");
    if (applied && !initial) throw std::invalid_argument("resume requires retained scene");
    WireSceneWorkBudget work{limits_.maxReplayWork};
    // Validate a supplied scene against this journal's limits, too. The old
    // immutable owner remains usable if construction rejects it.
    auto checked = buildSceneCandidate(initial, wire::encodeEnvelope({}), limits_.scene, &work);
    auto state = std::make_shared<Snapshot>();
    state->authoritative = initial ? std::move(initial) : std::move(checked);
    observeScene(*ids_, state->authoritative, work);
    state->visible = state->authoritative;
    state->applied = applied;
    state->initialApplied = applied;
    state_ = std::move(state);
}

SceneJournal::SnapshotPtr SceneJournal::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_;
}

uint64_t SceneJournal::submitLocal(std::string_view bytes, const Submit& submitter) {
    return submit(bytes, submitter, {}, {});
}

uint64_t SceneJournal::submitCaptured(const WireScenePtr& base, const std::shared_ptr<const SdfOperationLog>& sealed,
    const Submit& submitter, WireScenePtr* applied) {
    if (!base || !sealed || !sealed->operations) throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
    if (sealed->operations->empty()) return 0;
    return submit({}, submitter, base, sealed, applied);
}

size_t SceneJournal::probeCaptured(const WireScenePtr& base,
    const std::shared_ptr<const SdfOperationLog>& sealed) const {
    if (!base || !sealed || !sealed->operations) throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
    if (sealed->operations->empty()) return 0;
    const auto previous = snapshot();
    healthy(*previous);
    const auto limits = effectiveLimits(*previous->visible);
    auto compile = limits.authoring;
    compile.scene = limits.scene;
    compile.maxWork = std::min(compile.maxWork, limits.maxReplayWork);
    compile.maxRetainedBytes = std::min(compile.maxRetainedBytes, limits.maxPendingBytes);
    FieldEditIntentPtr authored;
    try {
        // A probe's own allocator: identities it would create are not the
        // admission's, and only their count and encoding matter here.
        authored = compileFieldEdits(base, sealed, compile, std::make_shared<SectionIdAllocator>());
    } catch (const wire::DecodeError& error) {
        if (error.kind == wire::ErrorKind::ResourceLimit) throw CapturedRejection{OMNI_USD_LIVE_RESOURCE_EXHAUSTED};
        throw;
    }
    if (authored->empty()) return 0;
    const auto size = authored->bytes().size();
    if (size > limits.scene.envelope.maxBytes || size > limits.maxReplayBytes || size > limits.maxPendingBytes ||
        authored->retainedBytes() > limits.maxPendingBytes - size)
        throw CapturedRejection{OMNI_USD_LIVE_RESOURCE_EXHAUSTED};
    return size;
}

uint64_t SceneJournal::submit(std::string_view bytes, const Submit& submitter, const WireScenePtr& explicitBase,
    const std::shared_ptr<const SdfOperationLog>& captured, WireScenePtr* applied) {
    if (!submitter) throw std::invalid_argument("missing scene submitter");
    // Owners precede guards so displaced snapshots retire after all journal
    // guards. Snapshot readers only need mutex_, not the preparation guard.
    SnapshotPtr previous, publication;
    std::shared_ptr<Snapshot> prepared;
    std::shared_ptr<const FieldEditIntent> authored;
    std::shared_ptr<const Intent> retained;
    WireScenePtr compiled;
    std::unique_lock<std::mutex> transaction(transactionMutex_);
    previous = snapshot();
    healthy(*previous);
    const auto limits = effectiveLimits(*previous->visible);
    // A captured log reports limits that waiting cannot relieve as definitive.
    const auto definitive = [&captured] {
        if (captured) throw CapturedRejection{OMNI_USD_LIVE_RESOURCE_EXHAUSTED};
        throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    };
    if (previous->revision == std::numeric_limits<uint64_t>::max() ||
        previous->localAdmitted == std::numeric_limits<uint64_t>::max())
        definitive();
    // A captured log checks this limit after compilation: an empty or
    // superseded log admits nothing and never waits for pending work.
    const auto pendingFull = [&] {
        if (previous->pending.size() >= limits.maxPending) throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    };
    if (!captured) pendingFull();
    size_t compileWork = 0;
    // A captured log compiles against the base its host edits were applied
    // to; replay below carries it onto the current scene like any pending intent.
    const auto& compileBase = captured ? explicitBase : previous->visible;
    if (captured) {
        auto compile = limits.authoring;
        compile.scene = limits.scene;
        compile.maxWork = std::min(compile.maxWork, limits.maxReplayWork);
        // A captured log compiles against the whole pending-byte quota, so
        // every compile limit it trips depends on the log alone and no wait
        // relieves it. Its share beside other pending work is charged below.
        compile.maxRetainedBytes = std::min(compile.maxRetainedBytes, limits.maxPendingBytes);
        try {
            authored = compileFieldEdits(compileBase, captured, compile, ids_, applied ? &compiled : nullptr);
        } catch (const DeltaEntryLimitExceeded&) {
            // Definitive like any compile limit; the flush names its counts.
            throw;
        } catch (const wire::DecodeError& error) {
            if (error.kind == wire::ErrorKind::ResourceLimit) definitive();
            throw;
        }
        if (!authored->concurrentStructureSupported()) throw UnsupportedOperation(authored->unsupportedStructure());
        if (authored->empty() || superseded(authored->bytes(), *previous->visible)) {
            // Host content equals the base, or the service would store every
            // record as orphan data; nothing is admitted or issued.
            if (applied) *applied = std::move(compiled);
            return 0;
        }
        bytes = authored->bytes();
        compileWork = authored->workUsed();
    }
    if (captured) pendingFull();
    // Reject impossible transactions before allocating the retained payload or
    // replaying older intents. Replay performs complete framing/value checks.
    // An intent that exceeds a quota even with no other pending work never fits.
    if (bytes.size() > limits.scene.envelope.maxBytes || bytes.size() > limits.maxReplayBytes ||
        bytes.size() > limits.maxPendingBytes ||
        (authored && authored->retainedBytes() > limits.maxPendingBytes - bytes.size()))
        definitive();
    size_t pendingSpace = limits.maxPendingBytes;
    charge(previous->pendingBytes, pendingSpace);
    charge(bytes.size(), pendingSpace);
    if (authored) charge(authored->retainedBytes(), pendingSpace);
    WireSceneWorkBudget work{limits.maxReplayWork};
    // Replay work a captured intent exceeds with no other pending intent
    // depends on the intent alone; no wait relieves it.
    const bool lone = captured && previous->pending.empty();
    try {
        charge(compileWork, work.remaining);
    } catch (OmniUsdLiveResult) {
        if (lone) definitive();
        throw;
    }
    size_t replaySpace = limits.maxReplayBytes;
    charge(previous->pendingWireBytes, replaySpace);
    charge(bytes.size(), replaySpace);
    prepared = std::make_shared<Snapshot>(*previous);
    prepared->revision = previous->revision + 1;
    prepared->localAdmitted = previous->localAdmitted + 1;
    retained = std::make_shared<Intent>(std::string(bytes), std::move(authored), prepared->localAdmitted,
        previous->revision, previous->localAdmitted);
    prepared->pending.push_back({0, retained});
    prepared->pendingBytes = limits.maxPendingBytes - pendingSpace;
    prepared->pendingWireBytes = limits.maxReplayBytes - replaySpace;
    std::vector<Footprint> footprint;
    try {
        // The visible scene is the replay of the pending intents, so the new
        // intent applies to it as the last step of that replay would. The
        // budget still holds the replay of every pending intent: each one's
        // step, beside this step, whose derivation of the scene's child
        // groups stands for the one a replay does for the first intent that
        // reads them.
        charge(previous->replayWork, work.remaining);
        observeIds(*ids_, retained->bytes, limits.scene, work);
        std::vector<WireReplayStep> steps;
        prepared->visible = buildReplayedScene(previous->visible, {retained->bytes}, limits.scene, &work, true, &steps);
        const auto& step = steps.front();
        // In a replay of every pending intent this step continues from the
        // previous intent's: it derives child groups from current paths only
        // when the previous step leaves them to derive again, and otherwise
        // keeps them derived.
        const bool continues = !previous->pending.empty() && !previous->replayRederives;
        auto& entry = prepared->pending.back();
        entry.replayWork = step.work - (continues ? step.derivedWork : 0);
        prepared->replayWork = saturatingAdd(previous->replayWork, entry.replayWork);
        prepared->replayFields = std::max(previous->replayFields, step.fields);
        prepared->replayMetadataBytes = std::max(previous->replayMetadataBytes, step.metadataBytes);
        prepared->replayRederives = step.rederives || !(step.derived || continues);
        prepared->derivation = Derivation::Extended;
        footprint = footprintOf(retained->bytes);
    } catch (OmniUsdLiveResult result) {
        if (lone && result == OMNI_USD_LIVE_RESOURCE_EXHAUSTED) definitive();
        throw;
    } catch (const wire::DecodeError& error) {
        if (lone && error.kind == wire::ErrorKind::ResourceLimit) definitive();
        throw;
    }
    prepared->work = limits.maxReplayWork - work.remaining - previous->replayWork;
    // The index gains the intent before the submitter publishes it, since
    // nothing may fail afterward; an admission that fails before publication
    // removes it again.
    size_t indexed = 0;
    try {
        for (const auto& key : footprint) {
            ++footprint_[key];
            ++indexed;
        }
    } catch (...) {
        retire(footprint_, footprint.data(), footprint.data() + indexed);
        throw;
    }
    publication = prepared;
    struct Commit {
        SnapshotPtr* state;
        SnapshotPtr* replacement;
        Snapshot* prepared;
        uint64_t accepted = 0;
        static void publish(void* opaque, uint64_t id) noexcept {
            auto& self = *static_cast<Commit*>(opaque);
            // A broken private submitter must never publish twice or admit ID0.
            if (!id || self.accepted) std::terminate();
            self.prepared->pending.back().id = id;
            self.accepted = id;
            self.state->swap(*self.replacement);
        }
    } commit{&state_, &publication, prepared.get()};
    std::unique_lock<std::mutex> guard(mutex_);
    try {
        const auto id = submitter(prepared->pending.back().intent->bytes, prepared->pending.back().intent,
            {Commit::publish, &commit});
        if (!commit.accepted) throw std::logic_error("submitter omitted scene publication");
        if (id != commit.accepted) prepared->submissionFault = OMNI_USD_LIVE_INTERNAL_ERROR;
    } catch (...) {
        if (!commit.accepted) {
            retire(footprint_, footprint);
            throw;
        }
        // The edit is already accepted. Returning a rejection here would let
        // callers resubmit it. Retain it, return its ID and stop future work.
        prepared->submissionFault = OMNI_USD_LIVE_INTERNAL_ERROR;
    }
    if (applied) *applied = std::move(compiled);
    return commit.accepted;
}

SceneJournal::SnapshotPtr SceneJournal::apply(const LayerProgress::Work& work, std::string_view bytes,
    std::shared_ptr<const void> localIntent) {
    return apply(std::vector<Application>{{work, bytes, std::move(localIntent)}});
}

SceneJournal::SnapshotPtr SceneJournal::apply(const std::vector<Application>& batch) {
    SnapshotPtr previous, publication;
    std::shared_ptr<Snapshot> prepared;
    // The index the published snapshot replaces footprint_ with, when its
    // derivation replays every pending intent, and the intents it retires.
    FootprintIndex replayed;
    std::vector<std::shared_ptr<const Intent>> retired;
    std::unique_lock<std::mutex> transaction(transactionMutex_);
    previous = snapshot();
    healthy(*previous);
    if (previous->revision == std::numeric_limits<uint64_t>::max())
        throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    if (batch.empty()) throw OmniUsdLiveResult(OMNI_USD_LIVE_INTERNAL_ERROR);
    prepared = std::make_shared<Snapshot>(*previous);
    using Disposition = LayerProgress::Disposition;
    // Each work is charged as if it applied alone, against the limits of the
    // scene before the batch; the derivation of the visible scene shares the
    // last work's budgets, scaled to the batch's final scene. One build
    // applies every work's delta to the authoritative scene.
    const auto limits = effectiveLimits(*previous->authoritative);
    size_t replayBytes = 0;
    std::vector<WireSceneWorkBudget> budgets;
    budgets.reserve(batch.size());
    std::vector<WireSceneDelivery> deliveries;
    deliveries.reserve(batch.size());
    // The visible scene follows from the previous one when every intent the
    // batch confirms is one the previous visible scene applies, every peer
    // update precedes the batch's acknowledgments and none absorbs one.
    bool follows = true, confirmed = false;
    std::vector<std::string_view> peers;
    bool rebasing = false;
    try {
        for (size_t index = 0; index < batch.size(); ++index) {
            const auto& [work, bytes, localIntent] = batch[index];
            // The positions between consecutive work are empty commits.
            if (index && work.first > prepared->applied + 1) prepared->applied = work.first - 1;
            if (!work.id || work.applied || prepared->applied == std::numeric_limits<uint64_t>::max() ||
                work.first != prepared->applied + 1 || work.last < work.first ||
                (work.disposition != Disposition::Peer && work.disposition != Disposition::Acknowledged))
                throw OmniUsdLiveResult(OMNI_USD_LIVE_INTERNAL_ERROR);
            const bool local = work.disposition == Disposition::Acknowledged;
            const auto absorbed = static_cast<size_t>(work.absorbedLocal);
            if (work.absorbedLocal) {
                // Peer coverage that contains acknowledged local writes, and a
                // local acknowledgment that committed writes before it retire
                // with, retires exactly those intents, the leading pending
                // ordinals (design section 12).
                if (work.absorbedLocal + (local ? 1 : 0) > prepared->pending.size() ||
                    prepared->localRetired > UINT64_MAX - work.absorbedLocal ||
                    prepared->localRetired + work.absorbedLocal != work.absorbedThrough)
                    throw OmniUsdLiveResult(OMNI_USD_LIVE_INTERNAL_ERROR);
                for (size_t i = 0; i < absorbed; ++i)
                    if (prepared->pending[i].intent->localOrdinal != prepared->localRetired + 1 + i)
                        throw OmniUsdLiveResult(OMNI_USD_LIVE_INTERNAL_ERROR);
            }
            if (local) {
                if (work.last != work.first || prepared->pending.size() <= absorbed)
                    throw OmniUsdLiveResult(OMNI_USD_LIVE_INTERNAL_ERROR);
                const auto& first = prepared->pending[absorbed];
                if (first.id != work.id || !localIntent || first.intent.get() != localIntent.get() ||
                    first.intent.owner_before(localIntent) || localIntent.owner_before(first.intent) ||
                    first.intent->bytes != bytes || prepared->localRetired >= UINT64_MAX - absorbed ||
                    first.intent->localOrdinal != prepared->localRetired + 1 + absorbed ||
                    work.localOrdinal != first.intent->localOrdinal)
                    throw OmniUsdLiveResult(OMNI_USD_LIVE_INTERNAL_ERROR);
                // An intent retiring without its own application, or one a
                // net-empty catch-up passed, leaves a visible scene the
                // previous one does not lead to.
                follows = follows && first.replayed && !absorbed && !work.netted;
                confirmed = true;
            } else if (localIntent || work.localOrdinal) {
                throw OmniUsdLiveResult(OMNI_USD_LIVE_INTERNAL_ERROR);
            } else {
                follows = follows && !confirmed && !work.absorbedLocal;
                if (!work.netted) peers.push_back(bytes);
            }
            replayBytes = limits.maxReplayBytes;
            // A net-empty catch-up passed this work: the service's net result
            // already accounts for it, so it retires without its bytes, as do
            // the intents it retires.
            if (!work.netted) {
                auto& budget = budgets.emplace_back(WireSceneWorkBudget{limits.maxReplayWork});
                charge(bytes.size(), replayBytes);
                observeIds(*ids_, bytes, limits.scene, budget);
                deliveries.push_back({bytes, &budget});
            }
            const auto leave = [&](size_t count) {
                for (size_t i = 0; i < count; ++i) {
                    const auto& entry = prepared->pending[i];
                    prepared->pendingBytes -= entry.intent->retainedBytes;
                    prepared->pendingWireBytes -= entry.intent->bytes.size();
                    prepared->replayWork -= std::min(prepared->replayWork, entry.replayWork);
                    retired.push_back(entry.intent);
                }
                prepared->pending.erase(prepared->pending.begin(), prepared->pending.begin() + ptrdiff_t(count));
                prepared->localRetired += count;
            };
            leave(local ? absorbed + 1 : absorbed);
            prepared->applied = work.last;
        }
        if (!deliveries.empty())
            prepared->authoritative = buildDeliveredScene(prepared->authoritative, deliveries, limits.scene);
        // The derivation is budgeted against the batch's final scene, less
        // the work its last update spent, as that update's own application
        // would.
        const auto finalLimits = effectiveLimits(*prepared->authoritative);
        const auto spent = budgets.empty() ? 0 : limits.maxReplayWork - budgets.back().remaining;
        const WireSceneWorkBudget allowed{finalLimits.maxReplayWork > spent ? finalLimits.maxReplayWork - spent : 0};
        WireSceneWorkBudget replayWork = allowed;
        rebasing = true;
        charge(prepared->pendingWireBytes, replayBytes);
        if (prepared->pending.empty()) {
            prepared->visible = prepared->authoritative;
            prepared->replayWork = prepared->replayFields = prepared->replayMetadataBytes = 0;
            prepared->replayRederives = false;
            prepared->derivation = Derivation::Authoritative;
        } else if (follows && peers.empty()) {
            // Applying a confirmed intent to the authoritative scene is the
            // first step of the previous visible scene's replay, so the
            // remaining intents replay to that same scene.
            prepared->visible = previous->visible;
            prepared->derivation = Derivation::Retained;
        } else if (!follows || !overlay(*previous, *prepared, peers, finalLimits.scene, replayWork)) {
            replayWork = allowed;
            replayPending(*prepared, finalLimits, replayWork);
            replayed = index(prepared->pending);
        }
        prepared->work = allowed.remaining - replayWork.remaining;
        for (const auto& budget : budgets) prepared->work += limits.maxReplayWork - budget.remaining;
    } catch (const wire::DecodeError& error) { throw applicationError(error, rebasing); }
    // The intents leaving a visible scene that follows from the previous one
    // leave the index with this publication.
    std::vector<Footprint> leaving;
    if (prepared->derivation == Derivation::Retained || prepared->derivation == Derivation::Overlaid)
        for (const auto& intent : retired) {
            auto keys = footprintOf(intent->bytes);
            leaving.insert(leaving.end(), std::make_move_iterator(keys.begin()), std::make_move_iterator(keys.end()));
        }
    prepared->revision = previous->revision + 1;
    publication = prepared;
    std::lock_guard<std::mutex> guard(mutex_);
    state_.swap(publication);
    switch (prepared->derivation) {
    case Derivation::Authoritative: footprint_.clear(); break;
    case Derivation::Replayed: footprint_.swap(replayed); break;
    default: retire(footprint_, leaving.data(), leaving.data() + leaving.size());
    }
    return prepared;
}

void SceneJournal::applyEmpty(uint64_t through) noexcept {
    std::shared_ptr<const Snapshot> publication;
    std::unique_lock<std::mutex> transaction(transactionMutex_);
    const auto previous = snapshot();
    if (through <= previous->applied || previous->revision == std::numeric_limits<uint64_t>::max()) return;
    std::shared_ptr<Snapshot> prepared;
    try {
        prepared = std::make_shared<Snapshot>(*previous);
    } catch (...) { return; } // The position lags until the next applied work carries it.
    prepared->applied = through;
    prepared->revision = previous->revision + 1;
    prepared->work = 0;
    prepared->derivation = Derivation::None;
    publication = prepared;
    std::lock_guard<std::mutex> guard(mutex_);
    state_.swap(publication);
}
} // namespace omni::usd::live
