// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "SceneJournal.h"
#include "OmpfEncode.h"
#include "UsdValueWriter.h"
#include <pxr/usd/sdf/schema.h>
#include "SceneNamespace.h"
#include <algorithm>
#include <atomic>
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
// Whether `path` of `scene` is a prim whose only fields, beside child
// fields, are an `over` specifier, if any: an implicit container's shape.
bool implicitShape(const WireScene& scene, const PXR_NS::SdfPath& path, WireSceneWorkBudget& work) {
    const auto& data = *scene.data();
    if (data.GetSpecType(path) != PXR_NS::SdfSpecTypePrim) return false;
    for (const auto& name : data.List(path)) {
        charge(128, work.remaining);
        if (childField(name.GetString())) continue;
        if (name != PXR_NS::SdfFieldKeys->Specifier) return false;
        const auto value = data.Get(path, name);
        if (!value.IsHolding<PXR_NS::SdfSpecifier>() ||
            value.UncheckedGet<PXR_NS::SdfSpecifier>() != PXR_NS::SdfSpecifierOver)
            return false;
    }
    return true;
}
// Builds a restore intent's update against `visible`: a Create of each
// restored section with a fresh identity, appended to its parent's
// children, and its fields and samples, with no Reorder. Its only records
// of a section it does not create are an authored copy's prim fields on an
// implicit container that holds the copy's path. Content restores to the
// path it held in its source's scene, under the section that holds that
// path in the scene it plans against.
class RestorePlanner {
public:
    // What one source restored: how many sections it created or adopted,
    // its round, the implicit containers it created, and whether it, or
    // which of its roots, waits for a section at its destination.
    struct Outcome {
        size_t written = 0;
        uint32_t round = 0;
        std::vector<uint64_t> containers;
        // Every section the source creates, authored or implicit, and every
        // section it adopts.
        std::vector<uint64_t> sections;
        // The sections of the planned scene the source adopts: implicit
        // containers its authored copies write their prim fields onto.
        std::vector<uint64_t> adopted;
        bool wait = false;
        std::set<uint64_t> waiting;
    };
    RestorePlanner(const WireScenePtr& visible, SectionIdAllocator& ids, const std::function<bool(uint64_t)>& ours,
        const std::function<uint32_t(uint64_t)>& roundOf, WireSceneWorkBudget& work)
        : visible_(visible), ids_(ids), ours_(ours), roundOf_(roundOf), work_(work) {}
    Outcome add(const SceneJournal::RestoreSource& source) {
        Outcome outcome;
        outcome_ = &outcome;
        adoptedRoots_ = &source.adopted;
        copied_ = source.round;
        const auto commands = metadata_.commands.size();
        const auto adoptions = adoptions_;
        const auto& before = *source.before;
        if (!at(source.path)) {
            // The section that replaced it is gone: the restore waits for
            // one, unless only an authored prim's adoption of it would restore.
            outcome.wait = !(source.authored && source.roots.empty());
        } else if (source.whole) {
            // Another writer's records can have authored the container.
            if (holdsContainer(source.path) && !implicitShape(before, source.path, work_))
                adopt(before, source.replaced, source.path);
            merge(before, source.path);
        } else if (source.authored && holdsContainer(source.path) && !implicitShape(before, source.path, work_)) {
            // An implicit container replaced this journal's authored prim:
            // the prim adopts it, and the prim's children merge into it.
            adopt(before, source.replaced, source.path);
            merge(before, source.path);
        } else {
            for (const auto root : source.roots) {
                charge(128, work_.remaining);
                const auto path = before.pathForSection(root);
                if (path.IsEmpty()) continue;
                // A root waits while an authored section above it is absent:
                // that section's creator restores it, and the root then
                // attaches under it.
                if (!ensure(before, namespaceParentPath(before.data()->GetSpecType(path), path), source.path))
                    outcome.waiting.insert(root);
                else if (!source.adopted.count(root)) place(before, root, path);
                else if (!at(path)) create(before, root, path);
                else if (holdsContainer(path) && !implicitShape(before, path, work_)) adopt(before, root, path);
                // An adopted root whose path an authored section holds is
                // dropped, as the first section at a path wins; its pending
                // children are roots of their own.
            }
        }
        outcome.written = metadata_.commands.size() - commands + adoptions_ - adoptions;
        outcome.round = copied_ + 1;
        outcome_ = nullptr;
        adoptedRoots_ = nullptr;
        return outcome;
    }
    bool empty() const { return metadata_.commands.empty() && metadata_.fields.empty(); }
    // Whether this plan created or adopted a section at or below `path`.
    bool createdWithin(const PXR_NS::SdfPath& path) const {
        for (const auto& entry : created_) {
            charge(64, work_.remaining);
            if (entry.first.HasPrefix(path)) return true;
        }
        for (const auto& adopted : adopted_) {
            charge(64, work_.remaining);
            if (adopted.HasPrefix(path)) return true;
        }
        return false;
    }
    // Whether a section this plan created as an implicit container still
    // is one: no authored copy adopted it.
    bool container(uint64_t id) const { return !uncontained_.count(id); }
    std::string encode(const SceneJournal::Limits& limits) {
        checkDeltaEntries(metadata_.commands, metadata_.fields.size(), metadata_.samples.size(),
            limits.authoring.entries);
        wire::EnvelopeEncodeLimits framing = limits.authoring.envelope;
        framing.content = limits.scene.envelope;
        return wire::encodeUsdEnvelope(metadata_, values_, framing, limits.authoring.values);
    }
private:
    uint64_t at(const PXR_NS::SdfPath& path) const {
        if (const auto found = created_.find(path); found != created_.end()) return found->second;
        return visible_->sectionForPath(path);
    }
    // Whether a section holds `path`, creating an implicit container, or a
    // section the source's pending restore adopted, from `source` where none
    // does; false at `stop` or at another authored section.
    bool ensure(const WireScene& source, const PXR_NS::SdfPath& path, const PXR_NS::SdfPath& stop) {
        charge(128, work_.remaining);
        if (at(path)) return true;
        if (path == stop || path.IsAbsoluteRootPath()) return false;
        const auto id = source.sectionForPath(path);
        if (!id || !(ours_(id) || (adoptedRoots_ && adoptedRoots_->count(id)) || implicitShape(source, path, work_)))
            return false;
        if (!ensure(source, namespaceParentPath(source.data()->GetSpecType(path), path), stop)) return false;
        create(source, id, path);
        return true;
    }
    // Restores the section `child` of `source` at `path`: a copy with its
    // subtree, or, where a section holds the path and either it or the copy
    // is an implicit container, the copy's children merged into it. An
    // authored copy writes its own prim fields onto an implicit container
    // there, which so becomes the authored prim, and an implicit copy's own
    // fields are dropped. An authored copy at a path an authored section
    // holds is dropped with its subtree, as the first section at a path
    // wins, unless it is this journal's implicit container that another
    // writer's records authored, whose children merge.
    void place(const WireScene& source, uint64_t child, const PXR_NS::SdfPath& path) {
        if (at(path)) {
            const bool container = holdsContainer(path);
            const bool implicit = implicitShape(source, path, work_);
            if (container && !implicit) adopt(source, child, path);
            if (ours_(child) || implicit || container) merge(source, path);
            return;
        }
        create(source, child, path);
        merge(source, path);
    }
    // Whether the section at `path` is an implicit container: one this plan
    // copied from an implicit container and no authored copy adopted, or one
    // of that shape in the scene it plans against that this plan has not
    // adopted.
    bool holdsContainer(const PXR_NS::SdfPath& path) {
        if (created_.count(path)) return containers_.count(path) != 0;
        if (adopted_.count(path)) return false;
        return implicitShape(*visible_, path, work_);
    }
    // Writes the prim fields of the authored copy `copied` of `source` at
    // `path` onto the implicit container that holds the path, which becomes
    // the authored prim.
    void adopt(const WireScene& source, uint64_t copied, const PXR_NS::SdfPath& path) {
        const auto id = at(path);
        copied_ = std::max(copied_, roundOf_(copied));
        copyFields(source, path, id);
        if (created_.count(path)) {
            containers_.erase(path);
            uncontained_.insert(id);
        } else {
            adopted_.insert(path);
            outcome_->adopted.push_back(id);
        }
        outcome_->sections.push_back(id);
        ++adoptions_;
    }
    void merge(const WireScene& source, const PXR_NS::SdfPath& path) {
        for (const auto child : source.listedChildren(source.sectionForPath(path))) {
            charge(256, work_.remaining);
            const auto childPath = source.pathForSection(child);
            if (childPath.IsEmpty()) continue; // A shadowed sibling has no content of its own.
            place(source, child, childPath);
        }
    }
    void create(const WireScene& source, uint64_t copied, const PXR_NS::SdfPath& path) {
        const auto type = source.data()->GetSpecType(path);
        const auto parent = at(namespaceParentPath(type, path));
        if (!parent) throw OmniUsdLiveResult(OMNI_USD_LIVE_INTERNAL_ERROR);
        copied_ = std::max(copied_, roundOf_(copied));
        const auto id = ids_.allocate(&work_, [&](uint64_t candidate) { return visible_->containsSection(candidate); });
        wire::Command command{wire::Command::Kind::Create};
        command.section = id;
        command.parent = parent;
        command.sectionType = uint8_t(type);
        const auto key = namespaceChildKey(type, path);
        command.name = key.IsHolding<PXR_NS::TfToken>() ? key.UncheckedGet<PXR_NS::TfToken>().GetString() :
                                                          key.UncheckedGet<PXR_NS::SdfPath>().GetString();
        charge(sizeof(wire::Command) + command.name.size(), work_.remaining);
        metadata_.commands.push_back(std::move(command));
        created_.emplace(path, id);
        outcome_->sections.push_back(id);
        copyFields(source, path, id);
        if (type == PXR_NS::SdfSpecTypePrim && implicitShape(source, path, work_)) {
            outcome_->containers.push_back(id);
            containers_.insert(path);
        }
    }
    // Writes the fields and samples of `path` in `source` as records of the
    // section `id`; a field this plan already writes of the section takes
    // the new value in its record.
    void copyFields(const WireScene& source, const PXR_NS::SdfPath& path, uint64_t id) {
        const auto& data = *source.data();
        for (const auto& name : data.List(path)) {
            charge(128 + name.GetString().size(), work_.remaining);
            if (childField(name.GetString())) continue; // Structure carries child fields.
            auto value = data.Get(path, name);
            if (name == PXR_NS::SdfFieldKeys->TimeSamples && value.IsHolding<PXR_NS::SdfTimeSampleMap>()) {
                for (const auto& [time, sample] : value.UncheckedGet<PXR_NS::SdfTimeSampleMap>()) {
                    charge(128, work_.remaining);
                    metadata_.samples.push_back({id, time, {}});
                    values_.samples.push_back(sample);
                }
                continue;
            }
            const auto [entry, added] = fieldAt_.emplace(std::make_pair(id, name), values_.fields.size());
            if (!added) {
                values_.fields[entry->second] = std::move(value);
                continue;
            }
            metadata_.fields.push_back({id, ++order_, name.GetString(), {}});
            values_.fields.push_back(std::move(value));
        }
    }
    const WireScenePtr& visible_;
    SectionIdAllocator& ids_;
    const std::function<bool(uint64_t)>& ours_;
    const std::function<uint32_t(uint64_t)>& roundOf_;
    WireSceneWorkBudget& work_;
    Outcome* outcome_ = nullptr;
    uint32_t copied_ = 0;
    wire::EnvelopeInput metadata_;
    wire::DecodedValues values_;
    std::map<PXR_NS::SdfPath, uint64_t> created_;
    std::set<PXR_NS::SdfPath> containers_;
    // The paths of planned-scene sections this plan adopted, the implicit
    // containers it created that an authored copy adopted, and the index of
    // each field record it writes.
    std::set<PXR_NS::SdfPath> adopted_;
    std::set<uint64_t> uncontained_;
    std::map<std::pair<uint64_t, PXR_NS::TfToken>, size_t> fieldAt_;
    // The current source's roots that its pending restore adopted.
    const std::set<uint64_t>* adoptedRoots_ = nullptr;
    size_t adoptions_ = 0;
    uint64_t order_ = 0;
};
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

std::vector<uint64_t> SceneJournal::createsOf(std::string_view bytes,
    std::vector<std::pair<uint64_t, uint64_t>>* attachments) const {
    const auto envelope = wire::decodeEnvelope(bytes, limits_.scene.envelope);
    std::vector<uint64_t> ids;
    for (const auto& command : envelope.commands)
        if (command.kind == wire::Command::Kind::Create) ids.push_back(command.section);
    if (attachments) {
        const std::set<uint64_t> own(ids.begin(), ids.end());
        for (const auto& command : envelope.commands)
            if (command.kind == wire::Command::Kind::Create && !own.count(command.parent))
                attachments->emplace_back(command.parent, command.section);
    }
    return ids;
}

namespace {
std::atomic<uint32_t> restoreRounds{SceneJournal::kMaxRestoreRounds};
std::atomic<uint32_t> refusedRestores{0};
}

void SceneJournal::setRestoreRoundsForTest(uint32_t rounds) noexcept {
    restoreRounds.store(rounds, std::memory_order_relaxed);
}

void SceneJournal::refuseRestoresForTest(uint32_t count) noexcept { refusedRestores.store(count); }

bool SceneJournal::takeRestoreRefusalForTest() noexcept {
    for (auto count = refusedRestores.load(); count;)
        if (refusedRestores.compare_exchange_weak(count, count - 1)) return true;
    return false;
}

void SceneJournal::restoreWith(Submit restorer) {
    auto shared = restorer ? std::make_shared<const Submit>(std::move(restorer)) : nullptr;
    std::lock_guard<std::mutex> transaction(transactionMutex_);
    restorer_ = std::move(shared);
}

void SceneJournal::observeReplaced(std::function<void(const ReplacedSection&)> observer) {
    auto shared = std::make_shared<const std::function<void(const ReplacedSection&)>>(std::move(observer));
    std::lock_guard<std::mutex> transaction(transactionMutex_);
    replaced_ = std::move(shared);
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
    WireReplayStep compiledStep;
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
            authored = compileFieldEdits(compileBase, captured, compile, ids_, &compiled, &compiledStep);
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
    const auto made = std::make_shared<Intent>(std::string(bytes), std::move(authored), prepared->localAdmitted,
        previous->revision, previous->localAdmitted);
    retained = made;
    prepared->pending.push_back({0, retained});
    prepared->pendingBytes = limits.maxPendingBytes - pendingSpace;
    prepared->pendingWireBytes = limits.maxReplayBytes - replaySpace;
    std::vector<Footprint> footprint;
    std::vector<uint64_t> creates;
    try {
        // The visible scene is the replay of the pending intents, so the new
        // intent applies to it as the last step of that replay would. The
        // budget still holds the replay of every pending intent: each one's
        // step, beside this step, whose derivation of the scene's child
        // groups stands for the one a replay does for the first intent that
        // reads them.
        charge(previous->replayWork, work.remaining);
        observeIds(*ids_, retained->bytes, limits.scene, work);
        WireReplayStep step;
        if (compiled && compileBase == previous->visible) {
            // The compiler built that step from this very scene, with these
            // limits: its candidate is the extended visible scene, and its
            // build is the step, which the budget is charged as a replay.
            charge(compiledStep.work, work.remaining);
            prepared->visible = compiled;
            step = compiledStep;
        } else {
            std::vector<WireReplayStep> steps;
            prepared->visible = buildReplayedScene(previous->visible, {retained->bytes}, limits.scene, &work, true,
                &steps);
            step = steps.front();
        }
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
        creates = createsOf(retained->bytes, &made->attachments);
        made->creates = creates;
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
    // So do the set of created sections and the implicit containers: the
    // intent's own join them, and a container it writes a record of leaves.
    size_t indexed = 0;
    std::vector<uint64_t> added, marked;
    std::vector<decltype(containers_)::node_type> cleared;
    const auto unindex = [&] {
        retire(footprint_, footprint.data(), footprint.data() + indexed);
        for (const auto id : added) created_.erase(id);
        for (const auto id : marked) containers_.erase(id);
        for (auto& node : cleared) containers_.insert(std::move(node));
    };
    try {
        for (const auto& key : footprint) {
            ++footprint_[key];
            ++indexed;
        }
        added.reserve(creates.size());
        for (const auto id : creates)
            if (created_.insert(id).second) added.push_back(id);
        cleared.reserve(footprint.size());
        for (const auto& key : footprint)
            if (const auto found = containers_.find(key.section); found != containers_.end())
                cleared.push_back(containers_.extract(found));
        if (retained->authored) {
            const auto& containers = retained->authored->implicitContainers();
            marked.reserve(containers.size());
            for (const auto id : containers)
                if (containers_.emplace(id, 0).second) marked.push_back(id);
        }
    } catch (...) {
        unindex();
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
            unindex();
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
    std::shared_ptr<const void> localIntent, bool restore) {
    return apply(std::vector<Application>{{work, bytes, std::move(localIntent)}}, restore);
}

SceneJournal::SnapshotPtr SceneJournal::apply(const std::vector<Application>& batch, bool restore) {
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
    // The sections each work's Creates replace, recorded for peer work only.
    std::vector<std::vector<WireReplacedSection>> replacements(batch.size());
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
                deliveries.push_back({bytes, &budget, local ? nullptr : &replacements[index]});
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
    const auto rounds = restoreRounds.load(std::memory_order_relaxed);
    const auto restorer = restorer_;
    const bool restoring = restore && restorer && *restorer && rounds;
    std::vector<WireReplacedSection> replacedAll;
    for (const auto& list : replacements) replacedAll.insert(replacedAll.end(), list.begin(), list.end());
    if (restoring && batch.size() > 1 && !replacedAll.empty()) {
        const bool attached = std::any_of(previous->pending.begin(), previous->pending.end(),
            [](const Pending& entry) { return !entry.intent->attachments.empty() || !entry.intent->adopted.empty(); });
        // Every section this journal created restores when an implicit
        // container replaces it, as its own containers do whatever replaces them.
        const bool relevant = attached ||
            std::any_of(replacedAll.begin(), replacedAll.end(),
                [&](const WireReplacedSection& replaced) { return created_.count(replaced.section) != 0; });
        if (relevant) {
            // A restore copies the visible scene before its replacing update,
            // which holds the restores of every earlier work: apply the works
            // as separate transactions would.
            transaction.unlock();
            SnapshotPtr last;
            for (const auto& application : batch) {
                applyEmpty(application.work.first - 1);
                last = apply(std::vector<Application>{application}, restore);
            }
            return last;
        }
    }
    // The content the replacements remove that this journal restores, with
    // the restores that wait.
    Restoration restoration;
    if (restoring) try {
            auto sources = restoreSources(*previous, replacedAll);
            sources.insert(sources.end(), waiting_.begin(), waiting_.end());
            if (!sources.empty()) restoration = planRestore(*previous, *prepared, std::move(sources), rounds);
        } catch (const wire::DecodeError& error) { throw applicationError(error, false); }
    // The replaced sections this journal created, found before publication
    // so nothing after it allocates; they leave the set with it, so no later
    // delivery of the same commit reports them again.
    std::vector<ReplacedSection> reports;
    std::vector<std::set<uint64_t>::iterator> reported;
    for (size_t index = 0; index < batch.size(); ++index)
        for (const auto& replaced : replacements[index]) {
            const auto found = created_.find(replaced.section);
            if (found == created_.end() || std::find(reported.begin(), reported.end(), found) != reported.end())
                continue;
            reported.push_back(found);
            reports.push_back({replaced.section, replaced.path, batch[index].work.last,
                restoration.intent && restoration.restored.count(replaced.section) != 0});
        }
    const auto observer = replaced_;
    prepared->revision = previous->revision + 1;
    // The index of the pending intents the prepared visible scene applies,
    // which a restore intent joins.
    FootprintIndex next;
    if (restoration.intent) switch (prepared->derivation) {
        case Derivation::Authoritative: break;
        case Derivation::Replayed: next = replayed; break;
        default: next = footprint_; retire(next, leaving.data(), leaving.data() + leaving.size());
        }
    bool joined = false;
    if (restoring) {
        try {
            joined = publishRestore(publication, prepared, restoration, std::move(next), restorer, true);
        } catch (const wire::DecodeError& error) { throw applicationError(error, false); }
        if (!joined)
            for (auto& report : reports) report.restored = false;
    } else {
        publication = prepared;
        std::lock_guard<std::mutex> guard(mutex_);
        state_.swap(publication);
    }
    if (!joined) {
        // No restore joined: the index follows the derivation.
        switch (prepared->derivation) {
        case Derivation::Authoritative: footprint_.clear(); break;
        case Derivation::Replayed: footprint_.swap(replayed); break;
        default: retire(footprint_, leaving.data(), leaving.data() + leaving.size());
        }
    }
    for (const auto found : reported) {
        containers_.erase(*found);
        restored_.erase(*found);
        created_.erase(found);
    }
    forgetAbsent();
    // The observer runs outside every journal guard, so host code it reaches
    // never holds the journal's lock.
    transaction.unlock();
    if (observer && *observer)
        for (const auto& report : reports) {
            try {
                (*observer)(report);
            } catch (...) {} // A report never changes the published application.
        }
    return prepared;
}

void SceneJournal::retryRestore() {
    SnapshotPtr previous, publication;
    std::shared_ptr<Snapshot> prepared;
    std::unique_lock<std::mutex> transaction(transactionMutex_);
    const auto rounds = restoreRounds.load(std::memory_order_relaxed);
    const auto restorer = restorer_;
    if (waiting_.empty() || !restorer || !*restorer || !rounds) return;
    previous = snapshot();
    healthy(*previous);
    if (previous->revision == std::numeric_limits<uint64_t>::max()) return;
    prepared = std::make_shared<Snapshot>(*previous);
    prepared->revision = previous->revision + 1;
    prepared->work = 0;
    Restoration restoration;
    try {
        restoration = planRestore(*previous, *prepared, waiting_, rounds);
    } catch (const wire::DecodeError& error) { throw applicationError(error, false); }
    try {
        publishRestore(publication, prepared, restoration, footprint_, restorer, false);
    } catch (const wire::DecodeError& error) { throw applicationError(error, false); }
}

std::vector<SceneJournal::RestoreSource> SceneJournal::restoreSources(const Snapshot& previous,
    const std::vector<WireReplacedSection>& replaced) const {
    std::vector<RestoreSource> sources;
    const auto& before = previous.visible;
    for (const auto& section : replaced) {
        RestoreSource source;
        source.before = before;
        source.replaced = section.section;
        source.path = before->pathForSection(section.section);
        if (source.path.IsEmpty()) continue; // A pending intent of this client removed it.
        if (const auto mark = containers_.find(section.section); mark != containers_.end()) {
            // This journal's implicit container: restore every child.
            source.whole = true;
            source.round = mark->second;
        } else {
            // A section this journal created: an authored prim, restored when
            // an implicit container replaced it.
            source.authored = created_.count(section.section) != 0;
            // The subtrees pending intents create under it, whose Creates the
            // service skips, since their parent is gone, and the sections at
            // or under it whose prim fields a pending restore writes, whose
            // records it skips once they are gone.
            for (const auto& entry : previous.pending) {
                for (const auto& attachment : entry.intent->attachments) {
                    const auto path = before->pathForSection(attachment.second);
                    if (!path.IsEmpty() && path != source.path && path.HasPrefix(source.path))
                        source.roots.insert(attachment.second);
                }
                for (const auto id : entry.intent->adopted) {
                    const auto path = before->pathForSection(id);
                    if (!path.IsEmpty() && path.HasPrefix(source.path)) {
                        source.roots.insert(id);
                        source.adopted.insert(id);
                    }
                }
            }
            if (source.roots.empty() && !source.authored) continue;
        }
        sources.push_back(std::move(source));
    }
    return sources;
}

SceneJournal::Restoration SceneJournal::planRestore(const Snapshot& previous, const Snapshot& prepared,
    std::vector<RestoreSource> sources, uint32_t rounds) const {
    Restoration result;
    const auto limits = effectiveLimits(*prepared.visible);
    WireSceneWorkBudget work{limits.maxReplayWork};
    // The restore round of each section a copy can hold: this journal's
    // implicit containers and every section an admitted restore created,
    // pending or committed, each with its own source's round.
    std::map<uint64_t, uint32_t> sectionRounds(restored_.begin(), restored_.end());
    for (const auto& [id, round] : containers_) {
        auto& known = sectionRounds[id];
        known = std::max(known, round);
    }
    const std::function<bool(uint64_t)> ours = [&](uint64_t id) { return containers_.count(id) != 0; };
    const std::function<uint32_t(uint64_t)> roundOf = [&](uint64_t id) {
        const auto found = sectionRounds.find(id);
        return found == sectionRounds.end() ? 0u : found->second;
    };
    RestorePlanner planner(prepared.visible, *ids_, ours, roundOf, work);
    uint32_t intentRound = 0;
    std::vector<uint64_t> adopted;
    // The sources a whole restore covers; each counts as restored when the
    // plan creates content within its path.
    std::vector<const RestoreSource*> covered;
    for (auto& source : sources) {
        charge(256, work.remaining);
        // A whole restore from the same scene that holds this one's subtree
        // already copies it.
        if (std::any_of(sources.begin(), sources.end(), [&](const RestoreSource& other) {
            return other.whole && other.before == source.before && other.path != source.path &&
                source.path.HasPrefix(other.path);
        })) {
            covered.push_back(&source);
            continue;
        }
        auto outcome = planner.add(source);
        if (outcome.written) {
            // A subtree restored more rounds than the bound fails the layer;
            // its host content keeps it.
            if (outcome.round > rounds) throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
            intentRound = std::max(intentRound, outcome.round);
            result.restored.insert(source.replaced);
            for (const auto id : outcome.containers) result.containers.emplace_back(id, outcome.round);
            for (const auto id : outcome.sections) result.sections.emplace_back(id, outcome.round);
            adopted.insert(adopted.end(), outcome.adopted.begin(), outcome.adopted.end());
            auto planned = source;
            for (const auto root : outcome.waiting) {
                planned.roots.erase(root);
                planned.adopted.erase(root);
            }
            result.planned.push_back(std::move(planned));
        }
        if (outcome.wait) {
            result.waiting.push_back(std::move(source));
        } else if (!outcome.waiting.empty()) {
            source.roots = std::move(outcome.waiting);
            for (auto root = source.adopted.begin(); root != source.adopted.end();)
                root = source.roots.count(*root) ? std::next(root) : source.adopted.erase(root);
            result.waiting.push_back(std::move(source));
        }
    }
    for (const auto* source : covered)
        if (planner.createdWithin(source->path)) result.restored.insert(source->replaced);
    // A container an authored copy adopted later in the plan is none.
    result.containers.erase(std::remove_if(result.containers.begin(), result.containers.end(),
                                [&](const auto& entry) { return !planner.container(entry.first); }),
        result.containers.end());
    if (planner.empty()) return result;
    result.intent = std::make_shared<Intent>(planner.encode(limits), nullptr, prepared.localAdmitted + 1,
        previous.revision, previous.localAdmitted);
    result.intent->creates = createsOf(result.intent->bytes, &result.intent->attachments);
    result.intent->adopted = std::move(adopted);
    result.intent->round = intentRound;
    return result;
}

bool SceneJournal::admitRestore(Snapshot& prepared, const std::shared_ptr<Intent>& intent) const {
    const auto limits = effectiveLimits(*prepared.visible);
    const auto size = intent->bytes.size();
    // A restore larger than any admission allows never fits.
    if (size > limits.scene.envelope.maxBytes || size > limits.maxReplayBytes ||
        intent->retainedBytes > limits.maxPendingBytes ||
        prepared.localAdmitted == std::numeric_limits<uint64_t>::max())
        throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    if (prepared.pending.size() >= limits.maxPending ||
        prepared.pendingBytes > limits.maxPendingBytes - intent->retainedBytes ||
        prepared.pendingWireBytes > limits.maxReplayBytes - size)
        return false;
    WireSceneWorkBudget work{limits.maxReplayWork};
    try {
        charge(prepared.replayWork, work.remaining);
    } catch (OmniUsdLiveResult) { return false; }
    // The restore intent applies to the visible scene as the last step of
    // the pending intents' replay, as an admission's does.
    std::vector<WireReplayStep> steps;
    WireScenePtr visible;
    try {
        visible = buildReplayedScene(prepared.visible, {intent->bytes}, limits.scene, &work, true, &steps);
    } catch (const wire::DecodeError& error) {
        if (error.kind != wire::ErrorKind::ResourceLimit) throw;
        // Beside the pending intents' replay work the restore may not fit
        // where it fits alone, and their retirement relieves that; a
        // restore that does not fit alone never fits.
        WireSceneWorkBudget alone{limits.maxReplayWork};
        (void)buildReplayedScene(prepared.visible, {intent->bytes}, limits.scene, &alone, true);
        return false;
    }
    const auto& step = steps.front();
    const bool continues = !prepared.pending.empty() && !prepared.replayRederives;
    Pending entry{0, intent};
    entry.replayWork = step.work - (continues ? step.derivedWork : 0);
    prepared.pending.push_back(std::move(entry));
    prepared.visible = std::move(visible);
    prepared.localAdmitted += 1;
    prepared.pendingBytes += intent->retainedBytes;
    prepared.pendingWireBytes += size;
    prepared.replayWork = saturatingAdd(prepared.replayWork, prepared.pending.back().replayWork);
    prepared.replayFields = std::max(prepared.replayFields, step.fields);
    prepared.replayMetadataBytes = std::max(prepared.replayMetadataBytes, step.metadataBytes);
    prepared.replayRederives = step.rederives || !(step.derived || continues);
    prepared.derivation = Derivation::Extended;
    prepared.work = saturatingAdd(prepared.work, step.work);
    return true;
}

bool SceneJournal::publishRestore(SnapshotPtr& publication, std::shared_ptr<Snapshot>& prepared,
    Restoration& restoration, FootprintIndex index, const std::shared_ptr<const Submit>& restorer, bool publishAlone) {
    // The restores planning left waiting wait whatever the admission does.
    if (restoration.waiting.size() > kMaxWaitingRestores) throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    bool admitted = false;
    if (restoration.intent) {
        auto candidate = std::make_shared<Snapshot>(*prepared);
        if (admitRestore(*candidate, restoration.intent)) {
            // The restore intent's created sections and containers join with
            // its admission, which publishes the candidate, and a container of
            // this journal it adopts leaves, as one an admission writes a
            // record of does.
            std::vector<uint64_t> added, marked;
            std::vector<uint64_t> rounded;
            std::vector<decltype(containers_)::node_type> cleared;
            const auto unindex = [&] {
                for (const auto id : added) created_.erase(id);
                for (const auto id : marked) containers_.erase(id);
                for (const auto id : rounded) restored_.erase(id);
                for (auto& node : cleared) containers_.insert(std::move(node));
            };
            try {
                for (const auto id : restoration.intent->creates)
                    if (created_.insert(id).second) added.push_back(id);
                cleared.reserve(restoration.intent->adopted.size());
                for (const auto id : restoration.intent->adopted)
                    if (const auto found = containers_.find(id); found != containers_.end())
                        cleared.push_back(containers_.extract(found));
                for (const auto& [id, round] : restoration.containers)
                    if (containers_.emplace(id, round).second) marked.push_back(id);
                for (const auto& [id, round] : restoration.sections)
                    if (restored_.emplace(id, round).second) rounded.push_back(id);
            } catch (...) {
                unindex();
                throw;
            }
            struct Commit {
                SnapshotPtr* state;
                SnapshotPtr* replacement;
                Snapshot* prepared;
                uint64_t accepted = 0;
                static void publish(void* opaque, uint64_t id) noexcept {
                    auto& self = *static_cast<Commit*>(opaque);
                    if (!id || self.accepted) std::terminate();
                    self.prepared->pending.back().id = id;
                    self.accepted = id;
                    self.state->swap(*self.replacement);
                }
            };
            publication = candidate;
            Commit commit{&state_, &publication, candidate.get()};
            std::unique_lock<std::mutex> guard(mutex_);
            try {
                const auto id = (*restorer)(restoration.intent->bytes, restoration.intent, {Commit::publish, &commit});
                if (!commit.accepted) throw std::logic_error("restorer omitted scene publication");
                if (id != commit.accepted) candidate->submissionFault = OMNI_USD_LIVE_INTERNAL_ERROR;
            } catch (...) {
                if (commit.accepted) {
                    // Accepted: retain the restore intent and stop future
                    // work, as an admission does.
                    candidate->submissionFault = OMNI_USD_LIVE_INTERNAL_ERROR;
                } else {
                    unindex();
                    publication = nullptr;
                    try {
                        throw;
                    } catch (const CapturedRejection& rejection) {
                        throw OmniUsdLiveResult(rejection.result); // Never fits.
                    } catch (OmniUsdLiveResult result) {
                        // Capacity the endpoint, the issuance window or a
                        // closing layer holds frees later: the restore waits.
                        if (result != OMNI_USD_LIVE_RESOURCE_EXHAUSTED && result != OMNI_USD_LIVE_BUSY &&
                            result != OMNI_USD_LIVE_CANCELLED)
                            throw;
                    }
                }
            }
            admitted = commit.accepted != 0;
            if (admitted) {
                for (auto& key : footprintOf(restoration.intent->bytes)) ++index[std::move(key)];
                footprint_.swap(index);
                prepared = std::move(candidate);
            }
        }
    }
    if (admitted) {
        waiting_ = std::move(restoration.waiting);
        return true;
    }
    // Refused, the planned restores wait too, each source once: a source
    // whose other roots wait already takes back its planned roots.
    auto waiting = std::move(restoration.waiting);
    if (restoration.intent)
        for (const auto& planned : restoration.planned) {
            const auto same = std::find_if(waiting.begin(), waiting.end(), [&](const RestoreSource& source) {
                return source.before == planned.before && source.replaced == planned.replaced && !source.whole;
            });
            if (same != waiting.end() && !planned.whole) {
                same->roots.insert(planned.roots.begin(), planned.roots.end());
                same->adopted.insert(planned.adopted.begin(), planned.adopted.end());
            } else waiting.push_back(planned);
        }
    if (waiting.size() > kMaxWaitingRestores) throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    if (publishAlone) {
        publication = prepared;
        std::lock_guard<std::mutex> guard(mutex_);
        state_.swap(publication);
    }
    waiting_ = std::move(waiting);
    return false;
}

void SceneJournal::forgetAbsent() noexcept {
    const auto tracked = [&] { return created_.size() + containers_.size() + restored_.size(); };
    if (tracked() < kMinSweep) return;
    const auto state = snapshot();
    if (tracked() < sweepAt_) {
        // The scene shrinks: each set holds one entry a section, so a set
        // with more than twice the sections that can hold one loses more
        // than half of its entries to the sweep. A waiting source can hold
        // any number of sections, so none is swept for this while one waits.
        if (!waiting_.empty()) return;
        const auto most = std::max({created_.size(), containers_.size(), restored_.size()});
        auto sections = state->authoritative->sectionCount();
        if (most <= 2 * sections) return;
        for (const auto& entry : state->pending) sections += entry.intent->creates.size();
        if (most <= 2 * sections) return;
    }
    std::vector<uint64_t> creates;
    try {
        for (const auto& entry : state->pending)
            creates.insert(creates.end(), entry.intent->creates.begin(), entry.intent->creates.end());
    } catch (...) { return; } // The next application sweeps.
    std::sort(creates.begin(), creates.end());
    const auto held = [&](uint64_t id) {
        if (state->authoritative->containsSection(id) || std::binary_search(creates.begin(), creates.end(), id))
            return true;
        return std::any_of(waiting_.begin(), waiting_.end(),
            [&](const RestoreSource& source) { return source.before->containsSection(id); });
    };
    for (auto it = created_.begin(); it != created_.end();) it = held(*it) ? std::next(it) : created_.erase(it);
    for (auto it = containers_.begin(); it != containers_.end();)
        it = held(it->first) ? std::next(it) : containers_.erase(it);
    for (auto it = restored_.begin(); it != restored_.end();)
        it = held(it->first) ? std::next(it) : restored_.erase(it);
    sweepAt_ = std::max(kMinSweep, 2 * tracked());
}

SceneJournal::Tracked SceneJournal::trackedForTest() {
    std::lock_guard<std::mutex> transaction(transactionMutex_);
    return {created_.size(), containers_.size(), restored_.size()};
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
