// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "WireScene.h"
#include <pxr/usd/sdf/listOp.h>
#include "SdfSceneData.h"
#include "PersistentMap.h"
#include <pxr/usd/sdf/childrenPolicies.h>
#include <pxr/usd/sdf/schema.h>
#include <pxr/usd/sdf/timeCode.h>
#include <pxr/usd/sdf/layerOffset.h>
#include <pxr/usd/sdf/payload.h>
#include <pxr/usd/sdf/reference.h>
#include <pxr/usd/sdf/types.h>
#include <pxr/base/gf/half.h>
#include <pxr/base/gf/matrix2d.h>
#include <pxr/base/gf/matrix3d.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/quatd.h>
#include <pxr/base/gf/quatf.h>
#include <pxr/base/gf/quath.h>
#include <pxr/base/gf/range1d.h>
#include <pxr/base/gf/range1f.h>
#include <pxr/base/gf/range2d.h>
#include <pxr/base/gf/range2f.h>
#include <pxr/base/gf/range3d.h>
#include <pxr/base/gf/range3f.h>
#include <pxr/base/gf/vec2d.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec2h.h>
#include <pxr/base/gf/vec3d.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/gf/vec3h.h>
#include <pxr/base/gf/vec4d.h>
#include <pxr/base/gf/vec4f.h>
#include <pxr/base/gf/vec4h.h>
#include <pxr/base/vt/array.h>
#include <pxr/base/vt/dictionary.h>
#include <algorithm>
#include <atomic>
#include <stdexcept>
#include <cmath>
#include <cstring>
#include <optional>
#include <type_traits>
#include <limits>
#include <map>
#include <set>

using namespace PXR_NS;
namespace omni::usd::live {
namespace {
using wire::DecodeError;
using wire::ErrorKind;
using wire::Command;
using Group = std::pair<uint64_t, unsigned>;
struct PredictedRejection {}; // A pending intent the service rejects as a whole.
std::atomic<bool> derivationCheck{false};
[[noreturn]] void malformed(const char* text) { throw DecodeError(ErrorKind::Malformed, text); }
[[noreturn]] void conflict(const char* text) { throw DecodeError(ErrorKind::Unsupported, text); }
[[noreturn]] void limited(const char* text) { throw DecodeError(ErrorKind::ResourceLimit, text); }
void spend(size_t n, size_t& left, const char* message) {
    if (n > left) limited(message);
    left -= n;
}
TfToken groupField(SdfSpecType parent, unsigned list) {
    if (parent == SdfSpecTypePseudoRoot) return list == 0 ? SdfChildrenKeys->PrimChildren : TfToken();
    if (parent == SdfSpecTypePrim || parent == SdfSpecTypeVariant) {
        if (list == 0) return SdfChildrenKeys->PrimChildren;
        if (list == 1) return SdfChildrenKeys->PropertyChildren;
        if (list == 2) return SdfChildrenKeys->VariantSetChildren;
    }
    if (parent == SdfSpecTypeVariantSet) return list == 0 ? SdfChildrenKeys->VariantChildren : TfToken();
    if (parent == SdfSpecTypeAttribute) {
        if (list == 0) return SdfChildrenKeys->ConnectionChildren;
        if (list == 1) return SdfChildrenKeys->ExpressionChildren;
        if (list == 2) return SdfChildrenKeys->MapperChildren;
    }
    if (parent == SdfSpecTypeRelationship) {
        if (list == 0) return SdfChildrenKeys->RelationshipTargetChildren;
        if (list == 1) return SdfChildrenKeys->ExpressionChildren;
    }
    if (parent == SdfSpecTypeRelationshipTarget) return list == 0 ? SdfChildrenKeys->PropertyChildren : TfToken();
    if (parent == SdfSpecTypeMapper) return list == 0 ? SdfChildrenKeys->MapperArgChildren : TfToken();
    return {};
}
unsigned childList(SdfSpecType parent, SdfSpecType child) {
    if ((parent == SdfSpecTypePseudoRoot || parent == SdfSpecTypePrim || parent == SdfSpecTypeVariant) &&
        child == SdfSpecTypePrim)
        return 0;
    if ((parent == SdfSpecTypePrim || parent == SdfSpecTypeVariant) &&
        (child == SdfSpecTypeAttribute || child == SdfSpecTypeRelationship))
        return 1;
    if ((parent == SdfSpecTypePrim || parent == SdfSpecTypeVariant) && child == SdfSpecTypeVariantSet) return 2;
    if (parent == SdfSpecTypeVariantSet && child == SdfSpecTypeVariant) return 0;
    if (parent == SdfSpecTypeAttribute) {
        if (child == SdfSpecTypeConnection) return 0;
        if (child == SdfSpecTypeExpression) return 1;
        if (child == SdfSpecTypeMapper) return 2;
    }
    if (parent == SdfSpecTypeRelationship) {
        if (child == SdfSpecTypeRelationshipTarget) return 0;
        if (child == SdfSpecTypeExpression) return 1;
    }
    if (parent == SdfSpecTypeRelationshipTarget && child == SdfSpecTypeAttribute) return 0;
    if (parent == SdfSpecTypeMapper && child == SdfSpecTypeMapperArg) return 0;
    malformed("incompatible wire scene parent/type");
}
bool pathField(const TfToken& field) {
    return field == SdfChildrenKeys->ConnectionChildren || field == SdfChildrenKeys->RelationshipTargetChildren ||
        field == SdfChildrenKeys->MapperChildren;
}
bool childField(const TfToken& field) {
    for (auto type : {SdfSpecTypePseudoRoot, SdfSpecTypePrim, SdfSpecTypeVariantSet, SdfSpecTypeAttribute,
             SdfSpecTypeRelationship, SdfSpecTypeRelationshipTarget, SdfSpecTypeMapper})
        for (unsigned list = 0; list < 3; ++list)
            if (field == groupField(type, list) && !field.IsEmpty()) return true;
    return false;
}
TfToken tokenKey(SdfSpecType type, const SdfPath& path) {
    if (type == SdfSpecTypeVariant) return TfToken(path.GetVariantSelection().second);
    if (type == SdfSpecTypeVariantSet) return TfToken(path.GetVariantSelection().first);
    return path.GetNameToken();
}
SdfPath pathKey(SdfSpecType type, const SdfPath& path) {
    return type == SdfSpecTypeMapper ? Sdf_MapperChildPolicy::GetFieldValue(path) : path.GetTargetPath();
}
void sampleMap(const VtValue& value) {
    if (value.IsEmpty()) return;
    if (!value.IsHolding<SdfTimeSampleMap>()) malformed("timeSamples field must be a map");
    for (const auto& [time, sample] : value.UncheckedGet<SdfTimeSampleMap>())
        if (std::isnan(time) || sample.IsEmpty()) malformed("invalid wire scene time sample");
}
}
struct WireScene::State {
    struct Section {
        uint64_t parent = 0;
        SdfSpecType type = SdfSpecTypeUnknown;
        std::string name;
        uint32_t order = 0;
        SdfPath path; // Derived path, kept for sections outside the visible scene too.
        size_t depth = 0;
        std::map<TfToken, VtValue> fields;
        bool unlisted = false; // An incoming Reorder of the parent omitted this child.
        bool shadowed = false; // A lower sibling ID owns the same derived path.
        bool hidden = false; // Unlisted, shadowed or below such a section: not projected.
        bool stranded = false; // The parent is missing: the derived path is the last one it had.
        // Other parents whose service children lists still name this section:
        // a Move removes it only from its carried old parent's list.
        std::vector<uint64_t> stale;
        // What the retained-value measurement counts for this section's field
        // values: encoded bytes, elements, string bytes and nesting depth.
        size_t encodedBytes = 0, elements = 0, stringBytes = 0, valueDepth = 0;
    };
    // Successive scenes share these maps' structure and their unchanged
    // sections, so a scene costs memory and build work in proportion to what
    // changed since the scene it was built on.
    PersistentMap<uint64_t, Section> sections;
    PersistentMap<SdfPath, uint64_t> paths;
    // Every row other than the root under its stored parent's ID, listed or
    // not, and every row under each parent its stale listings name, both in
    // ascending ID order. A parent's entry changes with its children only.
    using Index = PersistentMap<uint64_t, std::vector<uint64_t>>;
    Index children, staleListings;
    SdfAbstractDataRefPtr dataOwner;
    SdfAbstractDataConstPtr data;
    std::string sourceFormat;
    std::vector<WireFieldChange> changes;
    size_t fields = 0, metadataBytes = 0, pathBytes = 0, encodedBytes = 0, elements = 0, stringBytes = 0;
    // No derived path is deeper, in path nodes; removing the deepest subtree
    // leaves it as it was.
    size_t maxDepth = 0;
    // The number of sections whose values nest to each depth above zero.
    std::map<size_t, size_t> valueDepths;
    size_t valueDepth() const { return valueDepths.empty() ? 0 : valueDepths.rbegin()->first; }
    // The state a scene built on this one starts from: every member but the
    // publication records and Sdf storage, which belong to this scene alone.
    std::unique_ptr<State> successor() const {
        auto next = std::make_unique<State>();
        next->sections = sections;
        next->paths = paths;
        next->children = children;
        next->staleListings = staleListings;
        next->sourceFormat = sourceFormat;
        next->fields = fields;
        next->metadataBytes = metadataBytes;
        next->pathBytes = pathBytes;
        next->encodedBytes = encodedBytes;
        next->elements = elements;
        next->stringBytes = stringBytes;
        next->maxDepth = maxDepth;
        next->valueDepths = valueDepths;
        return next;
    }
};
WireScene::WireScene(std::unique_ptr<State> state) : state_(std::move(state)) {}
WireScene::~WireScene() = default;
size_t WireScene::retainedCharge() const {
    // The per-element and per-section costs match the journal's retained-scene charge.
    const auto& s = *state_;
    const long double charge = (long double)s.encodedBytes + 16.0L * s.elements + s.metadataBytes + s.pathBytes +
        256.0L * s.sections.size() + 64.0L * s.fields;
    return charge >= (long double)std::numeric_limits<size_t>::max() ? std::numeric_limits<size_t>::max() :
                                                                       size_t(charge);
}
const SdfAbstractDataConstPtr& WireScene::data() const { return state_->data; }
SdfPath WireScene::pathForSection(uint64_t id) const {
    const auto* section = state_->sections.find(id);
    return !section || section->hidden ? SdfPath() : section->path;
}
uint64_t WireScene::sectionForPath(const SdfPath& path) const {
    const auto* id = state_->paths.find(path);
    return id ? *id : 0;
}
bool WireScene::containsSection(uint64_t id) const { return state_->sections.find(id) != nullptr; }
size_t WireScene::sectionCount() const { return state_->sections.size(); }
size_t WireScene::pathCount() const { return state_->paths.size(); }
std::vector<uint64_t> WireScene::listedChildren(uint64_t parent) const {
    std::vector<std::pair<uint32_t, uint64_t>> ordered;
    if (const auto* children = state_->children.find(parent)) {
        sceneWork::count(children->size());
        for (const auto id : *children)
            if (const auto& section = *state_->sections.find(id); !section.unlisted)
                ordered.emplace_back(section.order, id);
    }
    std::sort(ordered.begin(), ordered.end());
    std::vector<uint64_t> result;
    result.reserve(ordered.size());
    for (const auto& item : ordered) result.push_back(item.second);
    return result;
}
bool WireScene::visitSections(size_t maxSections, const std::function<bool(const WireSectionView&)>& visitor) const {
    if (!visitor) throw std::invalid_argument("missing wire section visitor");
    if (state_->sections.size() > maxSections) limited("wire section enumeration limit exceeded");
    static const SdfPath hidden;
    for (const auto& [id, section] : state_->sections) {
        const WireSectionView view{id, section.parent, section.type, section.name, section.order,
            section.hidden ? hidden : section.path};
        if (!visitor(view)) return false;
    }
    return true;
}
const std::string& WireScene::sourceFormat() const { return state_->sourceFormat; }
const std::vector<WireFieldChange>& WireScene::fieldChanges() const { return state_->changes; }

class WireSceneBuilder {
    using Section = WireScene::State::Section;
    using Index = WireScene::State::Index;
    static constexpr size_t kWorkPerItem = 64;
public:
    WireSceneBuilder(const WireScenePtr& previous, const WireSceneLimits& limits, bool pendingReplay, bool delivered)
        : previous_(previous), limits_(limits), work_(limits.maxWork), pendingReplay_(pendingReplay),
          delivered_(delivered) {
        if (!limits.maxSections || !limits.maxPathDepth || limits.maxPathDepth > 256)
            throw std::invalid_argument("invalid wire scene limits");
    }
    // Applies one update to the working scene, as a candidate built on the
    // scene so far would, without materializing Sdf storage; `shared` is that
    // update's transaction budget. Updates apply in order, and finish
    // materializes the result once.
    void apply(const wire::Envelope& envelope, const wire::DecodedValues& values, WireSceneWorkBudget* shared) {
        envelope_ = &envelope;
        values_ = &values;
        shared_ = shared;
        const bool fresh = envelope.isDiff && envelope.baseVersion == 0;
        // Admitting the update's records and deriving the paths of the
        // sections it moves cost work linear in their counts and the children
        // lists they change, up to a sort's logarithmic factor. That much is
        // allowed on top of maxWork; a shared budget is charged what the
        // update does, and its owner sizes it for the scene.
        size_t items = envelope.commands.size() + envelope.fields.size() + envelope.samples.size();
        if (state_) items += state_->sections.size() + fieldCount_;
        else if (previous_) items += previous_->state_->sections.size() + previous_->state_->fields;
        const size_t allowance = items > std::numeric_limits<size_t>::max() / kWorkPerItem ?
            std::numeric_limits<size_t>::max() :
            items * kWorkPerItem;
        work_ = allowance > std::numeric_limits<size_t>::max() - limits_.maxWork ? std::numeric_limits<size_t>::max() :
                                                                                   limits_.maxWork + allowance;
        touched_.clear();
        retyped_.clear();
        if (state_ && !fresh) {
            // The scene so far, with current paths and children indexes.
        } else if (previous_ && !state_ && !fresh) {
            // The previous scene's sections, indexes and totals, shared: the
            // update copies only what it changes.
            const auto& old = *previous_->state_;
            if (old.sections.size() > limits_.maxSections || old.fields > limits_.maxFields ||
                old.metadataBytes > limits_.maxMetadataBytes || old.pathBytes > limits_.maxPathBytes ||
                old.encodedBytes > limits_.retainedValues.maxEncodedBytes ||
                old.elements > limits_.retainedValues.maxElements ||
                old.stringBytes > limits_.retainedValues.maxStringBytes ||
                old.valueDepth() > limits_.retainedValues.maxDepth || old.maxDepth > limits_.maxPathDepth)
                limited("previous wire scene exceeds limits");
            state_ = old.successor();
            fieldCount_ = old.fields;
            sectionMetadata_ = old.metadataBytes - old.sourceFormat.size();
            incremental_ = true;
        } else {
            state_ = std::make_unique<WireScene::State>();
            Section root;
            root.type = SdfSpecTypePseudoRoot;
            root.path = SdfPath::AbsoluteRootPath();
            insert(1, std::move(root));
            state_->paths.insert(SdfPath::AbsoluteRootPath(), 1);
            state_->pathBytes = 1;
            fieldCount_ = 0;
            sectionMetadata_ = 0;
            incremental_ = false;
        }
        state_->changes.clear();
        slots_.clear();
        explicitChildren_.clear();
        if (!envelope.sourceFormat.empty()) state_->sourceFormat = envelope.sourceFormat;
        affected_.clear();
        if (envelope.isDiff) diff();
        else {
            for (const auto& command : envelope.commands) normal(command);
            derive();
        }
        synchronizeChildren();
        fields();
        samples();
        validateFinalChildren();
        // The working scene's exact field and metadata totals, checked as a
        // materialized candidate checks them, so every intermediate scene of a
        // batch or replay stays within the limits. Only the sections this
        // update changed contribute anything new.
        size_t fields = fieldCount_, metadata = sectionMetadata_;
        for (const auto& [id, before] : touched_) {
            const auto* now = state_->sections.find(id);
            work(1 + (now ? now->fields.size() : 0));
            if (now) {
                const auto after = contribution(*now);
                fields = saturatingAdd(fields, after.fields);
                metadata = saturatingAdd(metadata, after.metadata);
            }
            fields -= std::min(fields, before.fields);
            metadata -= std::min(metadata, before.metadata);
        }
        const auto total = saturatingAdd(metadata, state_->sourceFormat.size());
        if (fields > limits_.maxFields) limited("wire scene field limit exceeded");
        if (total > limits_.maxMetadataBytes) limited("wire scene metadata limit exceeded");
        fieldCount_ = fields;
        sectionMetadata_ = metadata;
        flush();
        if (derivationCheck.load(std::memory_order_relaxed)) checkDerivation();
    }
    // Applies one retained local intent to the working scene as a PendingReplay
    // candidate built on the scene so far would, or, when `strict`, as an
    // OrdinaryUpdate candidate. An intent the service predictably rejects as a
    // whole leaves the working scene as it was and returns false. Only a Create
    // can be rejected, so only an intent holding one saves the working state
    // first; the saved state shares the working scene's structure.
    bool replayIntent(const wire::Envelope& envelope, const wire::DecodedValues& values, WireSceneWorkBudget* shared,
        bool strict) {
        pendingReplay_ = !strict && (state_ || previous_);
        const bool rejectable = pendingReplay_ &&
            std::any_of(envelope.commands.begin(), envelope.commands.end(),
                [](const auto& command) { return command.kind == Command::Kind::Create; });
        if (!rejectable) {
            apply(envelope, values, shared);
            return true;
        }
        std::unique_ptr<WireScene::State> saved;
        const auto fields = fieldCount_, metadata = sectionMetadata_;
        const bool incremental = incremental_;
        if (state_) saved = std::make_unique<WireScene::State>(*state_);
        try {
            apply(envelope, values, shared);
            return true;
        } catch (const PredictedRejection&) {
            // Without a saved copy no update has applied, and the next starts
            // from the previous scene again.
            state_ = std::move(saved);
            fieldCount_ = fields;
            sectionMetadata_ = metadata;
            incremental_ = incremental;
            slots_.clear();
            batch_.reset();
            deferred_ = {};
            dirty_.clear();
            dirtyParents_.clear();
            orderParents_.clear();
            return false;
        }
    }
    // The working scene, materialized; the previous scene when no update applied.
    WireScenePtr finish() {
        if (!state_) return previous_;
        materialize();
        return WireScenePtr(new WireScene(std::move(state_)));
    }
    // The working scene's field count and metadata bytes.
    size_t fieldCount() const { return state_ ? fieldCount_ : previous_ ? previous_->state_->fields : 0; }
    size_t metadataBytes() const {
        return state_ ? saturatingAdd(sectionMetadata_, state_->sourceFormat.size()) :
            previous_ ? previous_->state_->metadataBytes :
                        0;
    }
    // Every update leaves the working scene's paths, child groups and
    // children lists current, deriving only what it changes, so no intent
    // derives them from current paths and none leaves them to derive again
    // (WireReplayStep).
    size_t derivedWork() const { return 0; }
    bool derived() const { return true; }
    bool rederives() const { return false; }
    // Field and sample records the selector names apply nothing; they are
    // still decoded and charged.
    void skipRecords(const std::function<bool(const WireRecordKey&)>* skip) { skip_ = skip; }
private:
    // A section's share of the scene's field count and metadata bytes.
    struct Contribution {
        size_t fields = 0, metadata = 0;
    };
    static Contribution contribution(const Section& section) {
        Contribution result{section.fields.size(), section.name.size()};
        for (const auto& [name, value] : section.fields) {
            (void)value;
            result.metadata += name.GetString().size();
        }
        return result;
    }
    // The path bytes the scene charges for a row: none for the root, which is
    // charged once, and none for a stranded row, whose retained path is not
    // derived.
    static size_t pathCharge(uint64_t id, const Section& row) {
        return id == 1 || row.stranded ? 0 : row.path.GetString().size();
    }
    void work(size_t n = 1) {
        spend(n, work_, "wire scene work limit exceeded");
        if (shared_) spend(n, shared_->remaining, "wire scene transaction work limit exceeded");
    }
    const Section* lookup(uint64_t id) const { return state_->sections.find(id); }
    const Section& section(uint64_t id) const {
        const auto* found = state_->sections.find(id);
        if (!found) conflict("missing wire scene section");
        return *found;
    }
    // The rows `index` holds under `key`, in ascending ID order. Outside a
    // batch, the key's deferred changes are merged in first.
    const std::vector<uint64_t>& rows(Index& index, uint64_t key) {
        if (!batch_) {
            auto& deferred = changes(index);
            if (const auto found = deferred.find(key); found != deferred.end()) {
                mergeKey(index, key, found->second);
                deferred.erase(found);
            }
        }
        return held(index, key);
    }
    static const std::vector<uint64_t>& held(const Index& index, uint64_t key) {
        static const std::vector<uint64_t> none;
        const auto* found = index.find(key);
        return found ? *found : none;
    }
    const std::vector<uint64_t>& childrenOf(uint64_t parent) { return rows(state_->children, parent); }
    // Index changes, by index, key and row: whether the row ends under the
    // key. Inserting or removing rows one at a time would shift a list once
    // per row, so a run of changes to one list, such as creating a parent's
    // children in random ID order or removing a wide parent's subtree, costs
    // that list's length per row. Outside a batch, changes are deferred and
    // merged into a key's list when it is read and at the end of each update;
    // inside one, they are gathered for applyBatch. Either way each merge is
    // linear in the list and its changes.
    using Changes = std::map<uint64_t, std::map<uint64_t, bool>>;
    struct Batch {
        Changes children, listings;
    };
    std::optional<Batch> batch_;
    Batch deferred_;
    Changes& changes(const Index& index) {
        auto& target = batch_ ? *batch_ : deferred_;
        return &index == &state_->children ? target.children : target.listings;
    }
    void addTo(Index& index, uint64_t key, uint64_t id) { changes(index)[key][id] = true; }
    void removeFrom(Index& index, uint64_t key, uint64_t id) { changes(index)[key][id] = false; }
    // Merges every deferred change into the indexes.
    void flush() {
        auto deferred = std::move(deferred_);
        deferred_ = {};
        merge(state_->children, deferred.children);
        merge(state_->staleListings, deferred.listings);
    }
    // Opens a batch unless one is open; true when this call opened it. While
    // it is open the indexes still list removed rows and lack added ones, so
    // only reads whose extra or missing rows are already marked dirty, or
    // are gone, may read them.
    bool openBatch() {
        if (batch_) return false;
        flush();
        batch_.emplace();
        return true;
    }
    void applyBatch() {
        const auto changes = std::move(*batch_);
        batch_.reset();
        merge(state_->children, changes.children);
        merge(state_->staleListings, changes.listings);
    }
    void merge(Index& index, const Changes& changes) {
        for (const auto& [key, changed] : changes) mergeKey(index, key, changed);
    }
    void mergeKey(Index& index, uint64_t key, const std::map<uint64_t, bool>& changed) {
        const auto& old = held(index, key);
        work(old.size() + changed.size());
        sceneWork::count(old.size() + changed.size());
        std::vector<uint64_t> merged;
        merged.reserve(old.size() + changed.size());
        auto change = changed.begin();
        for (const auto id : old) {
            for (; change != changed.end() && change->first < id; ++change)
                if (change->second) merged.push_back(change->first);
            if (change == changed.end() || change->first != id) merged.push_back(id);
            else if ((change++)->second) merged.push_back(id);
        }
        for (; change != changed.end(); ++change)
            if (change->second) merged.push_back(change->first);
        if (merged.empty()) index.erase(key);
        else index.assign(key, std::move(merged));
    }
    // Every change to a section goes through touch, which records it for the
    // update's totals and the scene's materialization. Changes of a section's
    // parent, name, type, listing or stale listings go through insert, erase
    // and restructure, which also keep the children indexes and mark what
    // derive must derive again; edit changes nothing structural.
    void touch(uint64_t id) { touch(id, nullptr, false); }
    // `current`, when `known`, is the section's current row, or null for none.
    void touch(uint64_t id, const Section* current, bool known) {
        if (!touched_.count(id)) {
            const auto* before = known ? current : lookup(id);
            touched_.emplace(id, before ? contribution(*before) : Contribution{});
        }
        built_.insert(id);
    }
    Section& edit(uint64_t id) {
        touch(id);
        return owned(id);
    }
    // The section's current row `current`, made this scene's own to change.
    Section& edit(uint64_t id, const Section& current) {
        touch(id, &current, true);
        return owned(id);
    }
    Section& owned(uint64_t id) {
        auto* found = state_->sections.edit(id);
        if (!found) conflict("missing wire scene section");
        return *found;
    }
    void insert(uint64_t id, Section row) {
        touch(id);
        const auto parent = row.parent;
        const auto stale = row.stale;
        state_->sections.insert(id, std::move(row));
        if (id == 1) return;
        addTo(state_->children, parent, id);
        for (const auto listed : stale) addTo(state_->staleListings, listed, id);
        dirty_.insert(id);
        dirtyParents_.insert(parent);
        orderParents_.insert(parent);
        // Rows stranded under this ID derive from it again.
        for (const auto child : childrenOf(id)) dirty_.insert(child);
    }
    void erase(uint64_t id) {
        touch(id);
        const auto& row = section(id);
        const auto parent = row.parent;
        const auto stale = row.stale;
        if (!row.hidden && !row.path.IsEmpty()) {
            const auto* held = state_->paths.find(row.path);
            if (held && *held == id) state_->paths.erase(row.path);
        }
        state_->pathBytes -= std::min(state_->pathBytes, pathCharge(id, row));
        state_->sections.erase(id);
        removeFrom(state_->children, parent, id);
        for (const auto listed : stale) removeFrom(state_->staleListings, listed, id);
        // A sibling it shadowed takes its path; its children are stranded.
        dirtyParents_.insert(parent);
        for (const auto child : childrenOf(id)) dirty_.insert(child);
    }
    // Applies `change` to the section's stored parent, name, type, order,
    // listing and stale listings, keeping the indexes.
    template<class Change> void restructure(uint64_t id, Change&& change) {
        const auto& was = section(id);
        const auto parent = was.parent;
        const auto order = was.order;
        const auto type = was.type;
        const bool unlisted = was.unlisted;
        const auto name = was.name;
        const auto stale = was.stale;
        auto& row = edit(id);
        change(row);
        const auto now = row.parent;
        const bool moved = now != parent || row.name != name || row.type != type || row.unlisted != unlisted;
        const bool reordered = row.order != order;
        const bool retyped = row.type != type;
        const auto listings = row.stale;
        if (id == 1) return;
        if (now != parent) {
            removeFrom(state_->children, parent, id);
            addTo(state_->children, now, id);
        }
        if (listings != stale) {
            for (const auto listed : stale)
                if (std::find(listings.begin(), listings.end(), listed) == listings.end())
                    removeFrom(state_->staleListings, listed, id);
            for (const auto listed : listings)
                if (std::find(stale.begin(), stale.end(), listed) == stale.end())
                    addTo(state_->staleListings, listed, id);
        }
        if (moved) {
            dirty_.insert(id);
            dirtyParents_.insert(parent);
            dirtyParents_.insert(now);
            orderParents_.insert(now);
        }
        if (reordered) orderParents_.insert(now);
        if (retyped) {
            retyped_.insert(id);
            for (const auto child : childrenOf(id)) dirty_.insert(child);
        }
    }
    Group group(uint64_t id) const {
        const auto& child = section(id);
        return {child.parent, childList(section(child.parent).type, child.type)};
    }
    // A stranded row belongs to no parent's child fields.
    void mark(uint64_t id) {
        if (id != 1 && lookup(section(id).parent)) affected_.insert(group(id));
    }
    // The parent's listed, unshadowed children in `list`, in sibling order.
    std::vector<uint64_t> groupOf(Group key) {
        std::vector<uint64_t> ids;
        const auto* parent = lookup(key.first);
        if (!parent) return ids;
        const auto type = parent->type;
        for (const auto id : childrenOf(key.first)) {
            work();
            const auto& child = section(id);
            if (!child.unlisted && !child.shadowed && childList(type, child.type) == key.second) ids.push_back(id);
        }
        work(ids.size());
        std::sort(ids.begin(), ids.end(), [&](auto a, auto b) {
            work();
            return section(a).order < section(b).order;
        });
        return ids;
    }
    // Each parent's children by name and its next free order, built on first
    // use from its children and kept by Create; a Delete, Move or Reorder
    // drops them.
    struct Slots {
        std::map<std::string, std::set<uint64_t>> occupants;
        uint64_t nextOrder = 0; // Past the largest child order; 2^32 means exhausted.
    };
    std::map<uint64_t, Slots> slots_;
    Slots& slots(uint64_t parent) {
        if (const auto found = slots_.find(parent); found != slots_.end()) return found->second;
        Slots built;
        for (const auto id : childrenOf(parent)) {
            const auto& child = section(id);
            work(1 + child.name.size());
            built.occupants[child.name].insert(id);
            built.nextOrder = std::max<uint64_t>(built.nextOrder, uint64_t(child.order) + 1);
        }
        return slots_.emplace(parent, std::move(built)).first->second;
    }
    // The service resolves a (parent, name) slot to its lowest live identity.
    uint64_t occupant(uint64_t parent, const std::string& name, uint64_t except) {
        work(name.size() + 1);
        const auto& index = slots(parent).occupants;
        const auto found = index.find(name);
        if (found == index.end()) return 0;
        for (auto id : found->second) {
            work();
            if (id != except) return id;
        }
        return 0;
    }
    // Whether `id` is `ancestor` or lies in its current subtree.
    bool descends(uint64_t id, uint64_t ancestor) {
        for (size_t steps = 0; steps <= state_->sections.size(); ++steps) {
            work();
            if (id == ancestor) return true;
            if (id == 1) return false;
            const auto* found = lookup(id);
            if (!found) return false;
            id = found->parent;
        }
        conflict("cyclic wire scene graph");
    }
    // Unlists each of `rows` that a deleted or replaced children list of
    // `parents` names.
    void unlistFrom(const std::vector<uint64_t>& rows, const std::set<uint64_t>& parents) {
        const bool opened = openBatch();
        for (const auto id : rows) {
            work();
            const auto* row = lookup(id);
            if (!row ||
                std::none_of(row->stale.begin(), row->stale.end(), [&](auto p) { return parents.count(p) != 0; }))
                continue;
            restructure(id, [&](Section& edited) {
                auto& stale = edited.stale;
                stale.erase(std::remove_if(stale.begin(), stale.end(), [&](auto p) { return parents.count(p) != 0; }),
                    stale.end());
            });
        }
        if (opened) applyBatch();
    }
    // The rows that list any of `parents` stale.
    std::vector<uint64_t> staleListed(const std::set<uint64_t>& parents) {
        std::vector<uint64_t> result;
        for (const auto parent : parents) {
            const auto& listed = rows(state_->staleListings, parent);
            result.insert(result.end(), listed.begin(), listed.end());
        }
        return result;
    }
    // Delete a section and every section its service children list names,
    // recursively, including peer additions and stale listings: the listed
    // subtree of design section 10. Like the service's walk over stored
    // children lists, it leaves an unlisted descendant, with its subtree, as
    // a stranded row whose parent is missing.
    void cascade(uint64_t top) {
        slots_.clear();
        std::vector<uint64_t> dead{top};
        std::set<uint64_t> doomed{top};
        auto take = [&](uint64_t id) {
            work();
            if (doomed.insert(id).second) dead.push_back(id);
        };
        for (size_t i = 0; i < dead.size(); ++i) {
            for (auto id : childrenOf(dead[i]))
                if (!section(id).unlisted) take(id);
            for (auto id : rows(state_->staleListings, dead[i])) take(id);
        }
        const auto listed = staleListed(doomed);
        for (auto id : dead) mark(id); // A stale-listed section leaves its surviving parent's child field.
        // Erase reads a dead row's children only to mark them dirty; the
        // dead among them are erased themselves.
        openBatch();
        for (auto id : dead) erase(id);
        unlistFrom(listed, doomed);
        applyBatch();
    }
    uint32_t nextOrder(uint64_t parent) {
        work();
        const auto next = slots(parent).nextOrder;
        if (next > UINT32_MAX) conflict("wire scene sibling order exhausted");
        return uint32_t(next);
    }
    // A child's derived path, its depth in path nodes and its stored name,
    // which a mapper's normalized target replaces.
    struct Derived {
        SdfPath path;
        size_t depth = 0;
        std::string name;
    };
    // Count syntax work and nested targets before entering SDK parsers. The
    // expansion bound charges each possible target anchor before normalization.
    Derived childPath(const SdfPath& parentPath, SdfSpecType parentType, size_t parentDepth, const Section& child,
        size_t& bytesLeft) {
        Derived derived;
        derived.name = child.name;
        const auto& name = child.name;
        work(name.size());
        auto pathNodes = [&](const std::string& text) {
            size_t nodes = 1, nesting = 0;
            for (char c : text) {
                if (c == '[') {
                    if (++nesting > limits_.maxPathDepth) limited("wire scene target depth limit exceeded");
                }
                if (c == ']' && nesting) --nesting;
                if (c == '/' || c == '.' || c == '[' || c == '{') {
                    if (++nodes > limits_.maxPathDepth) limited("wire scene path node limit exceeded");
                }
            }
            if (parentDepth >= limits_.maxPathDepth || nodes > limits_.maxPathDepth - parentDepth)
                limited("wire scene path depth limit exceeded");
            return nodes;
        };
        size_t nodes = pathNodes(name);
        const size_t parentBytes = parentPath.GetString().size();
        auto checkExpansion = [&](size_t nameBytes, size_t count) {
            size_t allowance = bytesLeft;
            spend(nameBytes, allowance, "wire scene path byte limit exceeded");
            spend(32, allowance, "wire scene path byte limit exceeded");
            spend(parentBytes, allowance, "wire scene path byte limit exceeded");
            const size_t unit = bytesLeft - allowance;
            if (unit > bytesLeft / (count + 1)) limited("wire scene expanded path limit exceeded");
        };
        checkExpansion(name.size(), nodes);
        // Check a conservative bound before SDK operations, then charge actual
        // retained path bytes. SDK cache/allocator memory is a separate budget.
        SdfPath result;
        const TfToken token(name);
        switch (child.type) {
        case SdfSpecTypePrim:
            if (!SdfSchema::IsValidIdentifier(name)) malformed("invalid wire prim name");
            result = parentPath.AppendChild(token);
            break;
        case SdfSpecTypeAttribute:
        case SdfSpecTypeRelationship:
            if (!SdfSchema::IsValidNamespacedIdentifier(name)) malformed("invalid wire property name");
            result = parentType == SdfSpecTypeRelationshipTarget ? parentPath.AppendRelationalAttribute(token) :
                                                                   parentPath.AppendProperty(token);
            break;
        case SdfSpecTypeVariantSet:
            if (!SdfSchema::IsValidIdentifier(name)) malformed("invalid wire variant set name");
            result = parentPath.AppendVariantSelection(token, TfToken());
            break;
        case SdfSpecTypeVariant:
            if (!SdfSchema::IsValidVariantIdentifier(name)) malformed("invalid wire variant name");
            result = parentPath.GetParentPath().AppendVariantSelection(parentPath.GetVariantSelection().first, name);
            break;
        case SdfSpecTypeExpression:
            if (name != "expression") malformed("noncanonical wire expression name");
            result = parentPath.AppendExpression();
            break;
        case SdfSpecTypeMapperArg:
            if (!SdfSchema::IsValidIdentifier(name)) malformed("invalid wire mapper argument name");
            result = parentPath.AppendMapperArg(token);
            break;
        case SdfSpecTypeConnection:
        case SdfSpecTypeRelationshipTarget:
        case SdfSpecTypeMapper: {
            if (name.empty() || !SdfPath::IsValidPathString(name)) malformed("invalid wire target path");
            SdfPath target(name);
            if (child.type == SdfSpecTypeMapper) {
                target = target.MakeAbsolutePath(parentPath.GetPrimPath());
                if (target.IsEmpty()) malformed("invalid normalized mapper target");
                // A subsequent ancestor move changes the mapper spec path,
                // not the target already authored by this operation.
                const auto& canonical = target.GetString();
                if (canonical.size() > limits_.maxMetadataBytes) limited("wire scene metadata limit exceeded");
                work(canonical.size());
                nodes = pathNodes(canonical);
                checkExpansion(canonical.size(), nodes);
                derived.name = canonical;
                result = parentPath.AppendMapper(target);
            } else result = parentPath.AppendTarget(target);
            break;
        }
        default: malformed("unsupported wire child type");
        }
        if (result.IsEmpty()) malformed("invalid wire child path");
        const auto& text = result.GetString();
        work(text.size());
        spend(text.size(), bytesLeft, "wire scene path byte limit exceeded");
        derived.path = result;
        derived.depth = parentDepth + nodes;
        return derived;
    }
    // Sections outside the visible scene keep their derived paths and child
    // fields but own no path index entry and no Sdf spec. A child an incoming
    // Reorder omitted is unlisted; among listed siblings sharing a derived path
    // the lowest ID owns it and the others are shadowed. Both hide their subtrees.
    // An unlisted row whose parent was deleted is stranded: it keeps its last
    // derived path and roots a hidden subtree until a Move gives it a parent.
    //
    // Derives again the rows that changed structurally and the children of
    // every row whose path, depth or visibility changes, parents before
    // children, and resolves shadowing among the children of each parent
    // involved. A section whose derivation is unchanged is left as it is,
    // shared with the scene it came from, so the work follows the changed
    // rows, their subtrees and their siblings.
    void derive() {
        if (dirty_.empty() && dirtyParents_.empty()) {
            checkOrders();
            return;
        }
        std::map<uint64_t, std::set<uint64_t>> pending; // Parent to the children to derive.
        for (const auto id : dirty_) {
            const auto* row = lookup(id);
            if (row && id != 1) pending[row->parent].insert(id);
        }
        for (const auto parent : dirtyParents_) pending[parent];
        dirty_.clear();
        dirtyParents_.clear();
        std::map<size_t, std::set<uint64_t>> levels; // Parents by depth below the root or a stranded row.
        for (const auto& [parent, ids] : pending) {
            (void)ids;
            levels[ancestry(parent)].insert(parent);
        }
        // Each changed row's subtree derives again unless the row's
        // derivation leaves it as it was, so the subtree's current paths are
        // refunded until each row is derived again or found unchanged: no row
        // is charged an old path the update replaces, at any level.
        refunds_.clear();
        refunded_ = 0;
        for (const auto& [parent, ids] : pending)
            for (const auto id : ids) refund(id);
        const auto scheduled = [&](uint64_t parent, uint64_t id) {
            const auto found = pending.find(parent);
            return found != pending.end() && found->second.count(id) != 0;
        };
        std::vector<PathChange> changes;
        while (!levels.empty()) {
            const auto level = levels.begin();
            const auto depth = level->first;
            const auto parents = std::move(level->second);
            levels.erase(level);
            for (const auto parent : parents) {
                const auto found = pending.find(parent);
                if (found == pending.end()) continue;
                const auto ids = std::move(found->second);
                pending.erase(found);
                std::set<uint64_t> descended;
                resolve(parent, ids, changes, [&](uint64_t changed) {
                    // Its children derive from its new path or visibility.
                    descended.insert(changed);
                    const auto& children = childrenOf(changed);
                    if (children.empty()) return;
                    pending[changed].insert(children.begin(), children.end());
                    levels[depth + 1].insert(changed);
                });
                for (const auto id : ids) settle(id, descended.count(id) != 0, scheduled);
            }
        }
        refunds_.clear();
        refunded_ = 0;
        // Every row leaves its old path before any takes a new one, so rows
        // that exchange paths never collide.
        for (const auto& change : changes)
            if (change.wasVisible && !change.was.IsEmpty()) {
                const auto* held = state_->paths.find(change.was);
                if (held && *held == change.id) state_->paths.erase(change.was);
            }
        for (const auto& change : changes) {
            if (!change.visible) continue;
            const auto* held = state_->paths.find(change.path);
            if (held && *held != change.id) conflict("wire scene path collision");
            if (!held) state_->paths.insert(change.path, change.id);
        }
        if (state_->pathBytes > limits_.maxPathBytes) limited("wire scene path byte limit exceeded");
        checkOrders();
    }
    // A derived row's change of path or visibility.
    struct PathChange {
        uint64_t id;
        SdfPath was, path;
        bool wasVisible, visible;
    };
    // The current path charges of rows derive may derive again and has not
    // yet, by row, and their sum; resolve leaves them out of the charged bytes.
    std::map<uint64_t, size_t> refunds_;
    size_t refunded_ = 0;
    // Refunds `id` and every row in its subtree not refunded already.
    void refund(uint64_t id) {
        std::vector<uint64_t> stack{id};
        while (!stack.empty()) {
            const auto next = stack.back();
            stack.pop_back();
            work();
            const auto* row = lookup(next);
            if (!row || !refunds_.emplace(next, pathCharge(next, *row)).second) continue;
            refunded_ += refunds_.at(next);
            const auto& children = childrenOf(next);
            stack.insert(stack.end(), children.begin(), children.end());
        }
    }
    // Charges `id` again once it is derived. Unless its children derive
    // again, its subtree keeps its paths and is charged again too, but for
    // the rows `scheduled` still derives again and their subtrees.
    template<class Scheduled> void settle(uint64_t id, bool descended, const Scheduled& scheduled) {
        std::vector<uint64_t> stack{id};
        while (!stack.empty()) {
            const auto next = stack.back();
            stack.pop_back();
            const auto found = refunds_.find(next);
            if (found == refunds_.end()) continue;
            work();
            refunded_ -= found->second;
            refunds_.erase(found);
            if (next == id && descended) continue;
            for (const auto child : childrenOf(next))
                if (!scheduled(next, child)) stack.push_back(child);
        }
    }
    // The number of ancestors above `id` up to the root or a missing parent.
    size_t ancestry(uint64_t id) {
        size_t depth = 0;
        for (size_t steps = 0;; ++steps) {
            work();
            if (steps > state_->sections.size()) conflict("cyclic or disconnected wire scene graph");
            if (id == 1) return depth;
            const auto* row = lookup(id);
            if (!row) return depth;
            id = row->parent;
            ++depth;
        }
    }
    // Derives `ids`, children of `parent`, again, and the shadowing of all of
    // `parent`'s children; calls `changed` with each row whose path, depth or
    // visibility changes.
    template<class Changed>
    void resolve(uint64_t parent, const std::set<uint64_t>& ids, std::vector<PathChange>& changes, Changed&& changed) {
        const auto* owner = lookup(parent);
        if (!owner) {
            // Nothing reaches a stranded row from the root; its subtree derives
            // hidden paths from the row's last derived path, which is not
            // charged to the path byte limit (the independent model charges
            // it), so pathBytes undercounts a scene with stranded rows by their
            // own path bytes.
            for (const auto id : ids) {
                work();
                const auto& row = section(id);
                if (!row.unlisted || row.path.IsEmpty()) conflict("cyclic or disconnected wire scene graph");
                update(id, {row.path, row.depth, row.name}, false, true, true, changes, changed);
            }
            return;
        }
        const auto parentPath = owner->path;
        const auto parentType = owner->type;
        const auto parentDepth = owner->depth;
        const bool parentHidden = owner->hidden;
        // The rows derived again, here or later in this derivation, give back
        // their current charges first, so each is charged its new path only.
        size_t charged = state_->pathBytes - std::min(state_->pathBytes, refunded_);
        for (const auto id : ids)
            if (!refunds_.count(id)) charged -= std::min(charged, pathCharge(id, section(id)));
        size_t bytesLeft = limits_.maxPathBytes > charged ? limits_.maxPathBytes - charged : 0;
        std::map<uint64_t, Derived> derived;
        for (const auto id : ids) {
            work();
            const auto& row = section(id);
            for (const auto listed : row.stale) {
                work();
                if (listed == row.parent || !lookup(listed)) conflict("inconsistent wire stale listing");
            }
            (void)group(id);
            derived.emplace(id, childPath(parentPath, parentType, parentDepth, row, bytesLeft));
        }
        // Ascending ID: the lowest listed sibling claims its derived path.
        const auto siblings = childrenOf(parent);
        std::map<SdfPath, uint64_t> claimed;
        for (const auto id : siblings) {
            work();
            const auto& row = section(id);
            if (row.unlisted) continue;
            const auto found = derived.find(id);
            claimed.emplace(found == derived.end() ? row.path : found->second.path, id);
        }
        for (const auto id : siblings) {
            work();
            const auto& row = section(id);
            const auto found = derived.find(id);
            const auto& path = found == derived.end() ? row.path : found->second.path;
            const bool shadowed = !row.unlisted && claimed.at(path) != id;
            const bool hidden = parentHidden || row.unlisted || shadowed;
            if (found == derived.end() && shadowed == row.shadowed && hidden == row.hidden && !row.stranded) continue;
            update(id, found == derived.end() ? Derived{row.path, row.depth, row.name} : std::move(found->second),
                shadowed, hidden, false, changes, changed);
        }
    }
    template<class Changed>
    void update(uint64_t id, Derived derived, bool shadowed, bool hidden, bool stranded,
        std::vector<PathChange>& changes, Changed&& changed) {
        const auto& row = section(id);
        const bool moved = derived.path != row.path;
        if (moved || shadowed != row.shadowed) mark(id);
        state_->maxDepth = std::max(state_->maxDepth, derived.depth);
        if (!moved && shadowed == row.shadowed && hidden == row.hidden && derived.depth == row.depth &&
            derived.name == row.name && stranded == row.stranded)
            return;
        const bool descendants = moved || hidden != row.hidden || derived.depth != row.depth;
        changes.push_back({id, row.path, derived.path, !row.hidden, !hidden});
        state_->pathBytes -= std::min(state_->pathBytes, pathCharge(id, row));
        auto& edited = edit(id);
        edited.path = std::move(derived.path);
        edited.depth = derived.depth;
        edited.name = std::move(derived.name);
        edited.shadowed = shadowed;
        edited.hidden = hidden;
        edited.stranded = stranded;
        state_->pathBytes += pathCharge(id, edited);
        if (descendants) changed(id);
    }
    // Listed siblings never share an order.
    void checkOrders() {
        for (const auto parent : orderParents_) {
            if (!lookup(parent)) continue;
            std::set<uint32_t> orders;
            for (const auto id : childrenOf(parent)) {
                work();
                const auto& child = section(id);
                if (!child.unlisted && !orders.insert(child.order).second) conflict("ambiguous wire sibling order");
            }
        }
        orderParents_.clear();
    }
    // Test hook: derives the whole working scene from its sections alone and
    // fails unless the stored derivation, indexes and path bytes equal it,
    // every stale listing names an existing parent other than the row's
    // own, listed siblings never share an order and no row is deeper than
    // maxDepth.
    void checkDerivation() {
        const auto fail = [](const char* what) {
            throw std::logic_error(std::string("wire scene derivation: ") + what);
        };
        Index children, listings;
        for (const auto& [id, row] : state_->sections) {
            if (id == 1) continue;
            auto* list = children.edit(row.parent);
            if (list) list->push_back(id);
            else children.insert(row.parent, {id});
            if (row.depth > state_->maxDepth) fail("a row is deeper than the maximum depth");
            for (const auto listed : row.stale) {
                if (listed == row.parent || !state_->sections.find(listed)) fail("a stale listing is inconsistent");
                if (auto* stale = listings.edit(listed)) stale->push_back(id);
                else listings.insert(listed, {id});
            }
        }
        const auto same = [](const Index& a, const Index& b) {
            if (a.size() != b.size()) return false;
            for (const auto& [key, ids] : a) {
                const auto* other = b.find(key);
                if (!other || *other != ids) return false;
            }
            return true;
        };
        if (!same(children, state_->children)) fail("children index differs");
        if (!same(listings, state_->staleListings)) fail("stale listing index differs");
        PersistentMap<SdfPath, uint64_t> paths;
        paths.insert(SdfPath::AbsoluteRootPath(), 1);
        size_t bytes = 1, rowsSeen = 1;
        std::vector<uint64_t> queue{1};
        for (const auto& [id, row] : state_->sections)
            if (id != 1 && !state_->sections.find(row.parent)) {
                if (!row.hidden || row.shadowed || !row.stranded) fail("stranded row is visible");
                queue.push_back(id);
            }
        rowsSeen = queue.size();
        for (size_t cursor = 0; cursor < queue.size(); ++cursor) {
            const auto& parent = *state_->sections.find(queue[cursor]);
            const auto* list = children.find(queue[cursor]);
            if (!list) continue;
            std::set<SdfPath> claimed;
            std::set<uint32_t> orders;
            for (const auto id : *list) {
                const auto& row = *state_->sections.find(id);
                if (!row.unlisted && !orders.insert(row.order).second) fail("listed siblings share an order");
                size_t unlimited = std::numeric_limits<size_t>::max();
                const auto derived = childPathUncharged(parent, row, unlimited);
                const bool shadowed = !row.unlisted && !claimed.insert(derived.path).second;
                const bool hidden = parent.hidden || row.unlisted || shadowed;
                if (derived.path != row.path || derived.depth != row.depth || derived.name != row.name ||
                    shadowed != row.shadowed || hidden != row.hidden || row.stranded)
                    fail("a row's derived path or visibility differs");
                if (!hidden) paths.insert(row.path, id);
                bytes += row.path.GetString().size();
                queue.push_back(id);
                ++rowsSeen;
            }
        }
        if (rowsSeen != state_->sections.size()) fail("a row is unreachable");
        if (paths.size() != state_->paths.size()) fail("path index size differs");
        for (const auto& [path, id] : paths) {
            const auto* held = state_->paths.find(path);
            if (!held || *held != id) fail("path index differs");
        }
        if (bytes != state_->pathBytes) fail("path bytes differ");
    }
    // childPath without charging this build's work budgets.
    Derived childPathUncharged(const Section& parent, const Section& child, size_t& bytesLeft) {
        const auto work = work_;
        const auto shared = shared_;
        shared_ = nullptr;
        work_ = std::numeric_limits<size_t>::max();
        auto result = childPath(parent.path, parent.type, parent.depth, child, bytesLeft);
        work_ = work;
        shared_ = shared;
        return result;
    }
    void normal(const Command& c) {
        work();
        switch (c.kind) {
        case Command::Kind::Create: {
            if (c.section <= 1 || c.parent == c.section) malformed("invalid wire create identity");
            if (lookup(c.section)) {
                // A retained local intent replayed after a peer created the same identity predicts the
                // service's all-or-nothing rejection of a Create of a live ID. A delivered repeat changes
                // nothing; the later repeated commits restore any move or rename the section has had since.
                if (pendingReplay_) throw PredictedRejection{};
                if (!delivered_) malformed("duplicate wire section identity");
                break;
            }
            if (!lookup(c.parent)) break; // Create under a missing parent is skipped.
            if (const auto replaced = occupant(c.parent, c.name, c.section)) cascade(replaced);
            if (state_->sections.size() >= limits_.maxSections) limited("wire scene section limit exceeded");
            Section child;
            child.parent = c.parent;
            child.type = SdfSpecType(c.sectionType);
            child.name = c.name;
            child.order = nextOrder(c.parent);
            auto& slot = slots(c.parent);
            slot.occupants[child.name].insert(c.section);
            slot.nextOrder = std::max<uint64_t>(slot.nextOrder, uint64_t(child.order) + 1);
            insert(c.section, std::move(child));
            mark(c.section);
            break;
        }
        case Command::Kind::Delete: {
            if (c.section <= 1) malformed("invalid wire delete identity");
            // The carried parent is bookkeeping; a missing section is skipped.
            if (lookup(c.section)) cascade(c.section);
            break;
        }
        case Command::Kind::Move: {
            if (c.section <= 1 || c.parent == c.section) malformed("invalid wire move identity");
            if (!lookup(c.section)) break; // A Move of a missing section is skipped.
            if (c.oldParent == c.parent) { // Equal command parents rename under the stored parent.
                slots_.clear();
                mark(c.section);
                restructure(c.section, [&](Section& row) { row.name = c.name; });
                mark(c.section);
                break;
            }
            if (!lookup(c.parent) || descends(c.parent, c.section)) {
                cascade(c.section);
                break;
            }
            const auto replaced = occupant(c.parent, c.name, c.section);
            if (replaced && descends(c.section, replaced)) {
                cascade(c.section);
                break;
            }
            if (replaced) cascade(replaced);
            mark(c.section);
            const auto& current = section(c.section);
            // The service removes the section from its carried old parent's
            // children list only; a list that does not name it is unchanged,
            // so a stale carried parent leaves the stored parent's listing in
            // place. The new parent's list gains the section unless it names it.
            const bool append = current.parent != c.parent || current.unlisted;
            const auto order = append ? nextOrder(c.parent) : current.order;
            restructure(c.section, [&](Section& moved) {
                auto unlist = [&](uint64_t parent) {
                    const auto it = std::find(moved.stale.begin(), moved.stale.end(), parent);
                    if (it == moved.stale.end()) return false;
                    moved.stale.erase(it);
                    return true;
                };
                if (c.oldParent != moved.parent) (void)unlist(c.oldParent);
                if (moved.parent != c.parent) {
                    if (!moved.unlisted && c.oldParent != moved.parent) moved.stale.push_back(moved.parent);
                    (void)unlist(c.parent);
                }
                if (append) {
                    moved.order = order;
                    moved.unlisted = false;
                }
                moved.parent = c.parent;
                moved.name = c.name;
            });
            slots_.clear(); // Parent, name or order changed.
            mark(c.section);
            break;
        }
        case Command::Kind::Reorder: {
            // Local content names only current children, each once. A
            // delivered list, and a retained local intent replayed after a
            // peer removed or moved a listed child, is applied exactly as USD
            // applies an order: IDs that are not current children of the
            // section are ignored, and so is every repeat after an ID's first
            // occurrence. A service that orders the list per child list
            // commits a stale local intent that way; a service that stores
            // lists verbatim rejects it instead, and that definitive rejection
            // fails the layer (design section 10). A delivered list naming a
            // non-child is either such a commit or the repeat of a commit this
            // scene holds, whose later repeated commits restore the rest.
            const bool strict = !delivered_ && !pendingReplay_;
            std::set<uint64_t> named;
            std::vector<uint64_t> order;
            for (auto id : c.children) {
                work();
                const auto* found = lookup(id);
                if (!named.insert(id).second && strict) malformed("duplicate wire reorder member");
                if (!found || found->parent != c.section) {
                    if (strict) malformed("wire reorder names a section that is not a current child");
                    continue;
                }
                order.push_back(id);
            }
            // A missing parent's list names only stranded rows, which stay
            // unreachable, so the client applies nothing. The service stores
            // the list under the missing ID; that differs only if a later
            // Create reuses the ID, which the allocator never does for an ID
            // it has observed.
            if (!lookup(c.section)) break;
            slots_.clear(); // Orders change.
            derive(); // Reads derived groups.
            std::map<Group, std::vector<uint64_t>> before;
            for (unsigned list = 0; list < 3; ++list)
                before.emplace(Group(c.section, list), groupOf({c.section, list}));
            auto siblings = childrenOf(c.section);
            work(siblings.size());
            std::sort(siblings.begin(), siblings.end(), [&](auto a, auto b) {
                work();
                return section(a).order < section(b).order;
            });
            // USD orders each child list on its own, as Pcp applies primOrder
            // to a prim's children and propertyOrder to its properties, and so
            // does the service; the child-list ID plays no part. The parent's
            // children are split by child list, and each list the Reorder
            // names is passed to SdfApplyListOrdering with that list's named
            // members in list order, repeats included, so the result is
            // exactly USD's: the first occurrence of each named child, in
            // list order, moves to the end with the unnamed siblings of its
            // list that follow it, and unnamed children before the first named
            // one stay in front. IDs of other lists and non-children are not
            // members of the list, which USD ignores, so leaving them out
            // changes nothing. The result fills the positions that list holds
            // among the parent's children, so a list the Reorder does not name
            // keeps its order and positions, and no child leaves the scene.
            const auto parentType = section(c.section).type;
            std::map<unsigned, std::vector<size_t>> positions;
            for (size_t i = 0; i < siblings.size(); ++i) {
                work();
                positions[childList(parentType, section(siblings[i]).type)].push_back(i);
            }
            std::map<unsigned, std::vector<std::string>> orders;
            for (auto id : order) {
                work(20);
                orders[childList(parentType, section(id).type)].push_back(std::to_string(id));
            }
            auto placed = siblings;
            for (const auto& [list, listOrder] : orders) {
                const auto found = positions.find(list);
                if (found == positions.end()) continue; // Unreachable: every named ID is a sibling.
                const auto& at = found->second;
                std::vector<std::string> members;
                members.reserve(at.size());
                for (auto i : at) {
                    work(20);
                    members.push_back(std::to_string(siblings[i]));
                }
                SdfApplyListOrdering(&members, listOrder);
                work(members.size());
                for (size_t k = 0; k < at.size(); ++k) placed[at[k]] = std::stoull(members[k]);
            }
            work(placed.size());
            for (size_t i = 0; i < placed.size(); ++i) {
                const auto& child = section(placed[i]);
                if (child.order != uint32_t(i) || child.unlisted)
                    restructure(placed[i], [&](Section& row) {
                        row.order = uint32_t(i);
                        row.unlisted = false;
                    });
            }
            // Sections a stale Move left listed under the parent are not its
            // children and no list names them; the reorder drops their listings.
            const std::set<uint64_t> parent{c.section};
            unlistFrom(staleListed(parent), parent);
            derive();
            for (unsigned list = 0; list < 3; ++list) {
                const Group key(c.section, list);
                if (before.at(key) != groupOf(key)) affected_.insert(key);
            }
            return;
        }
        default: malformed("diff command in ordinary wire scene update");
        }
    }
    void diff() {
        std::set<uint64_t> changed;
        for (const auto& c : envelope_->commands) {
            work();
            if (!changed.insert(c.section).second) conflict("repeated diff section identity");
            if (lookup(c.section)) mark(c.section);
        }
        // Insert, erase and restructure read a row's children only to mark
        // them dirty; a row the batch adds there or removes is restated,
        // and so dirty, or erased.
        openBatch();
        for (const auto& c : envelope_->commands) {
            work();
            if (c.kind == Command::Kind::DiffDelete) {
                if (c.section == 1) malformed("diff cannot delete the pseudoroot");
                if (lookup(c.section)) erase(c.section); // A retraction of a section this client never held is a no-op.
            } else if (c.kind == Command::Kind::DiffChange) {
                if (!lookup(c.section)) {
                    if (state_->sections.size() >= limits_.maxSections) limited("wire scene section limit exceeded");
                    Section row;
                    row.parent = c.parent;
                    row.name = c.name;
                    row.type = SdfSpecType(c.sectionType);
                    row.order = c.sectionOrder;
                    insert(c.section, std::move(row));
                    continue;
                }
                restructure(c.section, [&](Section& child) {
                    child.parent = c.parent;
                    child.name = c.name;
                    child.unlisted = false;
                    child.stale.erase(std::remove(child.stale.begin(), child.stale.end(), c.parent), child.stale.end());
                    child.type = SdfSpecType(c.sectionType);
                    child.order = c.sectionOrder;
                });
            } else malformed("ordinary command in diff wire scene update");
        }
        applyBatch();
        // A retracted section's children list is gone with it.
        std::set<uint64_t> retracted;
        for (const auto& c : envelope_->commands)
            if (c.kind == Command::Kind::DiffDelete) retracted.insert(c.section);
        if (!retracted.empty()) unlistFrom(staleListed(retracted), retracted);
        place();
        derive();
        for (auto id : changed)
            if (lookup(id)) mark(id);
    }
    // A diff's section order is the section's index in its parent's current
    // children list, and the service restates only changed sections, so an
    // unchanged sibling keeps an index from an older list. Each restated
    // section is inserted at its index among the parent's other listed
    // children, in index order, and those children are renumbered from zero.
    void place() {
        std::map<uint64_t, std::vector<std::pair<uint32_t, uint64_t>>> placed;
        std::set<uint64_t> restated;
        for (const auto& c : envelope_->commands) {
            work();
            if (c.kind == Command::Kind::DiffChange && c.section != 1) {
                placed[c.parent].emplace_back(c.sectionOrder, c.section);
                restated.insert(c.section);
            }
        }
        for (auto& [parent, entries] : placed) {
            std::vector<std::pair<uint32_t, uint64_t>> others;
            for (const auto id : childrenOf(parent)) {
                work();
                const auto& child = section(id);
                if (!child.unlisted && !restated.count(id)) others.emplace_back(child.order, id);
            }
            work(others.size() + entries.size());
            std::sort(others.begin(), others.end());
            std::vector<uint64_t> children;
            children.reserve(others.size() + entries.size());
            for (const auto& entry : others) children.push_back(entry.second);
            std::stable_sort(entries.begin(), entries.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });
            for (const auto& [order, id] : entries) {
                const auto at = std::min<size_t>(order, children.size());
                work(children.size() - at + 1); // Elements the insertion shifts.
                children.insert(children.begin() + ptrdiff_t(at), id);
            }
            for (size_t i = 0; i < children.size(); ++i)
                if (section(children[i]).order != uint32_t(i))
                    restructure(children[i], [&](Section& row) { row.order = uint32_t(i); });
        }
    }
    void synchronizeChildren() {
        for (const auto& key : affected_) {
            work();
            const auto* parent = lookup(key.first);
            if (!parent) continue;
            const auto field = groupField(parent->type, key.second);
            if (field.IsEmpty()) continue;
            const auto group = groupOf(key);
            parent = lookup(key.first);
            const auto existing = parent->fields.find(field);
            if (group.empty()) {
                if (existing != parent->fields.end()) edit(key.first).fields.erase(field);
            } else if (pathField(field)) {
                std::vector<SdfPath> paths;
                paths.reserve(group.size());
                for (auto id : group) {
                    work();
                    const auto& child = section(id);
                    paths.push_back(pathKey(child.type, child.path));
                }
                if (existing == parent->fields.end() || !existing->second.IsHolding<std::vector<SdfPath>>() ||
                    existing->second.UncheckedGet<std::vector<SdfPath>>() != paths)
                    edit(key.first).fields[field] = VtValue::Take(paths);
            } else {
                std::vector<TfToken> names;
                names.reserve(group.size());
                for (auto id : group) {
                    work();
                    const auto& child = section(id);
                    names.push_back(tokenKey(child.type, child.path));
                }
                if (existing == parent->fields.end() || !existing->second.IsHolding<std::vector<TfToken>>() ||
                    existing->second.UncheckedGet<std::vector<TfToken>>() != names)
                    edit(key.first).fields[field] = VtValue::Take(names);
            }
        }
    }
    unsigned fieldList(const Section& parent, const TfToken& field) {
        for (unsigned list = 0; list < 3; ++list)
            if (groupField(parent.type, list) == field) return list;
        malformed("child field incompatible with section type");
    }
    void fields() {
        state_->changes.reserve(envelope_->fields.size());
        for (size_t i = 0; i < envelope_->fields.size(); ++i) {
            work();
            const auto& field = envelope_->fields[i];
            const auto& value = values_->fields[i];
            const TfToken name(field.name);
            // A skipped record changes nothing; the build that applied it to
            // the authoritative scene validated it.
            if (skip_ && (*skip_)(WireRecordKey{field.section, field.name, false})) continue;
            if (name == SdfFieldKeys->TimeSamples) sampleMap(value);
            const auto* found = lookup(field.section);
            if (!found) continue; // Orphan data for a missing section never becomes visible.
            if (childField(name)) {
                (void)fieldList(*found, name);
                if (!value.IsEmpty() &&
                    (pathField(name) ? !value.IsHolding<std::vector<SdfPath>>() :
                                       !value.IsHolding<std::vector<TfToken>>()))
                    malformed("invalid child field value type");
                explicitChildren_.insert({field.section, fieldList(*found, name)});
            }
            if (name == SdfFieldKeys->TimeSamples && value.IsHolding<SdfTimeSampleMap>())
                work(value.UncheckedGet<SdfTimeSampleMap>().size());
            auto& owner = edit(field.section, *found);
            if (value.IsEmpty()) owner.fields.erase(name);
            else owner.fields[name] = value;
            state_->changes.push_back({field.section, field.setOrder, i, name});
        }
        std::sort(state_->changes.begin(), state_->changes.end(), [&](const auto& a, const auto& b) {
            work();
            return a.setOrder != b.setOrder ? a.setOrder < b.setOrder : a.record < b.record;
        });
    }
    // Each sample applies to the map the update's earlier samples left. A
    // section's map is copied once, on its first sample, and written back once:
    // erasing an absent key changes nothing, so an authored empty map stays and
    // an absent field stays absent, while a changed map that ends empty removes
    // the field. Work is one unit per sample plus one per copied entry.
    void samples() {
        struct Working {
            SdfTimeSampleMap samples;
            bool changed = false;
        };
        std::map<uint64_t, Working> working;
        for (size_t i = 0; i < envelope_->samples.size(); ++i) {
            work();
            const auto& sample = envelope_->samples[i];
            const auto& value = values_->samples[i];
            if (skip_ && (*skip_)(WireRecordKey{sample.section, {}, true})) continue;
            auto entry = working.find(sample.section);
            if (entry == working.end()) {
                const auto* found = lookup(sample.section);
                if (!found) continue; // Orphan sample for a missing section.
                Working copy;
                const auto existing = found->fields.find(SdfFieldKeys->TimeSamples);
                if (existing != found->fields.end()) {
                    work(existing->second.UncheckedGet<SdfTimeSampleMap>().size());
                    copy.samples = existing->second.UncheckedGet<SdfTimeSampleMap>();
                }
                entry = working.emplace(sample.section, std::move(copy)).first;
            }
            auto& current = entry->second;
            if (value.IsEmpty()) current.changed |= current.samples.erase(sample.time) != 0;
            else {
                current.samples[sample.time] = value;
                current.changed = true;
            }
        }
        for (auto& [id, current] : working) {
            if (!current.changed) continue;
            auto& fields = edit(id).fields;
            if (current.samples.empty()) fields.erase(SdfFieldKeys->TimeSamples);
            else fields[SdfFieldKeys->TimeSamples] = VtValue::Take(current.samples);
        }
    }
    template<class T> void validateGroup(Group key, const std::vector<T>& authored) {
        const auto ids = groupOf(key);
        if (authored.size() != ids.size()) conflict("child field differs from final graph");
        std::map<T, uint64_t> members;
        std::vector<uint32_t> slots;
        slots.reserve(ids.size());
        for (auto id : ids) {
            work();
            const auto& child = section(id);
            T member;
            if constexpr (std::is_same_v<T, SdfPath>) member = pathKey(child.type, child.path);
            else member = tokenKey(child.type, child.path);
            if (!members.emplace(member, id).second) conflict("ambiguous child field members");
            slots.push_back(child.order);
        }
        std::sort(slots.begin(), slots.end());
        for (size_t i = 0; i < authored.size(); ++i) {
            work();
            auto member = members.find(authored[i]);
            if (member == members.end()) conflict("duplicate or unknown child field member");
            if (section(member->second).order != slots[i]) edit(member->second).order = slots[i];
            members.erase(member);
        }
    }
    void validateFinalChildren() {
        // Validate effective explicit values only; earlier records may describe
        // an intermediate namespace state that later records replace.
        for (const auto& key : explicitChildren_) {
            const auto& parent = section(key.first);
            const auto field = groupField(parent.type, key.second);
            const auto found = parent.fields.find(field);
            if (found == parent.fields.end()) {
                if (!groupOf(key).empty()) conflict("erased nonempty child field");
            } else if (pathField(field)) {
                const auto authored = found->second.UncheckedGet<std::vector<SdfPath>>();
                validateGroup(key, authored);
            } else {
                const auto authored = found->second.UncheckedGet<std::vector<TfToken>>();
                validateGroup(key, authored);
            }
        }
        // A type-changing diff may expose a retained child field that is no
        // longer legal. Never silently drop it when rebuilding the scene. Only
        // a diff changes a section's type, and every other update checks each
        // child field it writes.
        for (const auto id : retyped_) {
            const auto* child = lookup(id);
            if (!child) continue;
            for (const auto& [field, value] : child->fields) {
                (void)value;
                work();
                if (childField(field)) (void)fieldList(*child, field);
            }
        }
    }
    // The section's field values as the retained-value measurement counts
    // them, within what the limits leave beside the rest of the scene.
    void measure(Section& child, const wire::EncodeLimits& room) {
        wire::DecodedValues retained;
        retained.fields.reserve(child.fields.size());
        for (const auto& [name, value] : child.fields) {
            (void)name;
            retained.fields.push_back(value);
        }
        size_t elements = 0, strings = 0, depth = 0;
        child.encodedBytes = wire::measureValues(retained, room, &elements, &strings, &depth);
        child.elements = elements;
        child.stringBytes = strings;
        child.valueDepth = depth;
    }
    // Adds the measured section to the scene's retained-value totals.
    void addTotals(const Section& child) {
        state_->encodedBytes += child.encodedBytes;
        state_->elements += child.elements;
        state_->stringBytes += child.stringBytes;
        if (child.valueDepth) ++state_->valueDepths[child.valueDepth];
    }
    // Removes a section the scene's totals hold.
    void removeTotals(const Section& child) {
        state_->encodedBytes -= std::min(state_->encodedBytes, child.encodedBytes);
        state_->elements -= std::min(state_->elements, child.elements);
        state_->stringBytes -= std::min(state_->stringBytes, child.stringBytes);
        const auto depth = state_->valueDepths.find(child.valueDepth);
        if (depth != state_->valueDepths.end() && !--depth->second) state_->valueDepths.erase(depth);
    }
    // The retained-value limits less what the rest of the scene uses, within
    // what a shared budget leaves for the measurement itself.
    wire::EncodeLimits room() const {
        auto encodingLimits = limits_.retainedValues;
        const auto left = [](size_t limit, size_t used) { return limit > used ? limit - used : 0; };
        encodingLimits.maxEncodedBytes = left(encodingLimits.maxEncodedBytes, state_->encodedBytes);
        encodingLimits.maxElements = left(encodingLimits.maxElements, state_->elements);
        encodingLimits.maxStringBytes = left(encodingLimits.maxStringBytes, state_->stringBytes);
        if (shared_) {
            encodingLimits.maxEncodedBytes = std::min(encodingLimits.maxEncodedBytes, shared_->remaining / 2);
            encodingLimits.maxElements = std::min(encodingLimits.maxElements, shared_->remaining / 2);
        }
        return encodingLimits;
    }
    // Measures and stores the sections the build changed and the Sdf storage
    // of the visible ones. Storage built on the previous scene shares that
    // scene's storage and replaces the specs of the changed sections only.
    void materialize() {
        state_->fields = fieldCount_;
        state_->metadataBytes = saturatingAdd(sectionMetadata_, state_->sourceFormat.size());
        auto data = TfCreateRefPtr(new SdfSceneData);
        const auto charge = [&](const Section& child) {
            if (!shared_) return;
            spend(child.encodedBytes, shared_->remaining, "wire scene transaction value limit exceeded");
            spend(child.elements, shared_->remaining, "wire scene transaction value limit exceeded");
        };
        if (incremental_) {
            const auto& old = previous_->state_->sections;
            data->CopyFrom(previous_->state_->data);
            // The sections the build changed, each with whether the scene still
            // holds it. Each is charged in the second loop in this order, so a
            // section is measured with the budget the sections before it leave.
            std::vector<std::pair<uint64_t, bool>> changed;
            changed.reserve(built_.size());
            for (const auto id : built_) {
                const auto* before = old.find(id);
                const auto* after = lookup(id);
                if (before == after) continue; // Shared: unchanged.
                changed.emplace_back(id, after != nullptr);
                if (before) {
                    removeTotals(*before);
                    // A spec that stays projected at its path is replaced in
                    // place below; no other section can take that path.
                    const bool kept = after && !after->hidden && after->path == before->path;
                    if (!before->hidden && !kept) data->EraseSpec(before->path);
                }
            }
            for (const auto& [id, held] : changed) {
                work();
                if (!held) continue;
                auto& child = owned(id);
                work(child.fields.size());
                measure(child, room());
                addTotals(child);
                charge(child);
                if (child.hidden) continue;
                data->PutSpec(child.path, child.type, child.fields);
            }
        } else {
            state_->encodedBytes = state_->elements = state_->stringBytes = 0;
            state_->valueDepths.clear();
            std::vector<uint64_t> ids;
            ids.reserve(state_->sections.size());
            for (const auto& [id, child] : state_->sections) {
                (void)child;
                ids.push_back(id);
            }
            for (const auto id : ids) {
                work();
                auto& child = edit(id);
                work(child.fields.size());
                measure(child, room());
                addTotals(child);
                charge(child);
            }
            for (const auto& [id, child] : state_->sections) {
                (void)id;
                if (!child.hidden) data->PutSpec(child.path, child.type, child.fields);
            }
        }
        state_->dataOwner = data;
        state_->data = data;
    }
    const WireScenePtr previous_;
    const wire::Envelope* envelope_ = nullptr;
    const wire::DecodedValues* values_ = nullptr;
    const WireSceneLimits& limits_;
    size_t work_;
    // The working scene started from the previous scene's sections, so its
    // storage and measurements change only for the sections in built_.
    bool incremental_ = false;
    // The working scene's fields and section metadata bytes after the last
    // update, which each later update is charged as a build on the scene so
    // far would be.
    size_t fieldCount_ = 0, sectionMetadata_ = 0;
    static size_t saturatingAdd(size_t a, size_t b) noexcept {
        return b > std::numeric_limits<size_t>::max() - a ? std::numeric_limits<size_t>::max() : a + b;
    }
    WireSceneWorkBudget* shared_ = nullptr;
    const std::function<bool(const WireRecordKey&)>* skip_ = nullptr;
    bool pendingReplay_;
    const bool delivered_;
    std::unique_ptr<WireScene::State> state_;
    // Sections the current update changed, with their contributions before it.
    std::map<uint64_t, Contribution> touched_;
    // Sections any update of this build changed.
    std::set<uint64_t> built_;
    // Rows whose derived path, depth or visibility may change; parents whose
    // children's shadowing must be resolved again; parents whose listed
    // children's orders must be checked; sections whose type changed.
    std::set<uint64_t> dirty_, dirtyParents_, orderParents_, retyped_;
    std::set<Group> affected_, explicitChildren_;
};
namespace {
struct Decoded {
    wire::Envelope envelope;
    wire::DecodedValues values;
};
// Decodes and validates one update's envelope and values, charging its input,
// records and decoding to `shared`.
Decoded decodeUpdate(std::string_view bytes, const WireSceneLimits& limits, WireSceneWorkBudget* shared,
    WireSceneBatchPolicy policy) {
    auto framing = limits.envelope;
    if (shared) {
        spend(bytes.size(), shared->remaining, "wire scene transaction input limit exceeded");
        framing.maxStringBytes = std::min(framing.maxStringBytes, shared->remaining / 2);
        framing.maxRecords = uint32_t(std::min<size_t>(framing.maxRecords, shared->remaining / 2));
        // The parser requires a positive configured record limit even for an
        // empty envelope; this is an exhausted transaction, not invalid config.
        if (!framing.maxRecords) limited("wire scene transaction metadata limit exceeded");
    }
    Decoded decoded{wire::decodeEnvelope(bytes, framing), {}};
    const auto& envelope = decoded.envelope;
    if ((policy == WireSceneBatchPolicy::OrdinaryUpdate || policy == WireSceneBatchPolicy::PendingReplay) &&
        envelope.isDiff)
        throw wire::DecodeError(wire::ErrorKind::Unsupported, "local scene intent must be an ordinary update");
    if (shared) {
        for (const auto& command : envelope.commands) {
            spend(1 + command.children.size(), shared->remaining, "wire scene transaction metadata limit exceeded");
            spend(command.name.size(), shared->remaining, "wire scene transaction metadata limit exceeded");
        }
        for (const auto& field : envelope.fields) {
            spend(1, shared->remaining, "wire scene transaction metadata limit exceeded");
            spend(field.name.size(), shared->remaining, "wire scene transaction metadata limit exceeded");
        }
        spend(envelope.samples.size(), shared->remaining, "wire scene transaction metadata limit exceeded");
        spend(envelope.sourceFormat.size(), shared->remaining, "wire scene transaction metadata limit exceeded");
    }
    auto valueLimits = limits.values;
    if (shared) {
        const auto allowance = shared->remaining / 4;
        valueLimits.maxEncodedBytes = std::min(valueLimits.maxEncodedBytes, allowance);
        valueLimits.maxDecodedBytes = std::min(valueLimits.maxDecodedBytes, allowance);
        valueLimits.maxElements = std::min(valueLimits.maxElements, allowance);
        valueLimits.maxComparisonWork = std::min(valueLimits.maxComparisonWork, allowance);
    }
    wire::ValueUsage usage;
    decoded.values = wire::decodeValues(envelope, valueLimits, &usage);
    if (shared)
        for (auto charge : {usage.encodedBytes, usage.decodedBytes, usage.elements, usage.comparisonWork})
            spend(charge, shared->remaining, "wire scene transaction decoding limit exceeded");
    return decoded;
}
} // namespace

namespace {
template<class T> bool sameBits(const T* a, const T* b, size_t count) {
    static_assert(std::is_trivially_copyable_v<T>);
    return count == 0 || std::memcmp(a, b, sizeof(T) * count) == 0;
}
// Set when `a` holds T or VtArray<T>; `b` holds the same type.
template<class T> std::optional<bool> sameFloating(const VtValue& a, const VtValue& b) {
    if (a.IsHolding<T>()) return sameBits(&a.UncheckedGet<T>(), &b.UncheckedGet<T>(), 1);
    if (a.IsHolding<VtArray<T>>()) {
        const auto& x = a.UncheckedGet<VtArray<T>>();
        const auto& y = b.UncheckedGet<VtArray<T>>();
        return x.size() == y.size() && (x.IsIdentical(y) || sameBits(x.cdata(), y.cdata(), x.size()));
    }
    return std::nullopt;
}
template<class... T> std::optional<bool> sameFloatingAny(const VtValue& a, const VtValue& b) {
    std::optional<bool> result;
    (void)((result = sameFloating<T>(a, b)).has_value() || ...);
    return result;
}
bool sameProjectedValue(const VtValue& a, const VtValue& b);
// USD compares layer offsets within a tolerance; the projection keeps their bits.
bool sameItem(const SdfLayerOffset& a, const SdfLayerOffset& b) {
    const double x[] = {a.GetOffset(), a.GetScale()}, y[] = {b.GetOffset(), b.GetScale()};
    return sameBits(x, y, 2);
}
bool sameItem(const SdfReference& a, const SdfReference& b) {
    return a == b && sameItem(a.GetLayerOffset(), b.GetLayerOffset()) &&
        sameProjectedValue(VtValue(a.GetCustomData()), VtValue(b.GetCustomData()));
}
bool sameItem(const SdfPayload& a, const SdfPayload& b) {
    return a == b && sameItem(a.GetLayerOffset(), b.GetLayerOffset());
}
bool sameItem(const SdfUnregisteredValue& a, const SdfUnregisteredValue& b) {
    return sameProjectedValue(a.GetValue(), b.GetValue());
}
template<class T> bool sameItems(const std::vector<T>& a, const std::vector<T>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (!sameItem(a[i], b[i])) return false;
    return true;
}
template<class T> bool sameItem(const SdfListOp<T>& a, const SdfListOp<T>& b) {
    return a.IsExplicit() == b.IsExplicit() && sameItems(a.GetExplicitItems(), b.GetExplicitItems()) &&
        sameItems(a.GetAddedItems(), b.GetAddedItems()) && sameItems(a.GetPrependedItems(), b.GetPrependedItems()) &&
        sameItems(a.GetAppendedItems(), b.GetAppendedItems()) && sameItems(a.GetDeletedItems(), b.GetDeletedItems()) &&
        sameItems(a.GetOrderedItems(), b.GetOrderedItems());
}
// Set when `a` holds T, a vector of T, or a list operation of T; `b` holds the same type.
template<class T> std::optional<bool> sameComposite(const VtValue& a, const VtValue& b) {
    if (a.IsHolding<T>()) return sameItem(a.UncheckedGet<T>(), b.UncheckedGet<T>());
    if (a.IsHolding<std::vector<T>>())
        return sameItems(a.UncheckedGet<std::vector<T>>(), b.UncheckedGet<std::vector<T>>());
    if (a.IsHolding<SdfListOp<T>>()) return sameItem(a.UncheckedGet<SdfListOp<T>>(), b.UncheckedGet<SdfListOp<T>>());
    return std::nullopt;
}
// Value equality of the wire projection. Floating-point content compares by
// representation: of scalars, arrays, double vectors, dictionaries,
// time-sample maps including their keys, layer offsets, references, payloads,
// their vectors and list operations, and unregistered values. A signed zero
// differs from zero, whose encodings differ, and a NaN equals the same NaN.
// Other values compare with VtValue equality.
bool sameProjectedValue(const VtValue& a, const VtValue& b) {
    if (a.GetType() != b.GetType()) return false;
    if (a.IsHolding<SdfTimeSampleMap>()) {
        const auto& x = a.UncheckedGet<SdfTimeSampleMap>();
        const auto& y = b.UncheckedGet<SdfTimeSampleMap>();
        if (x.size() != y.size()) return false;
        for (auto i = x.begin(), j = y.begin(); i != x.end(); ++i, ++j)
            if (!sameBits(&i->first, &j->first, 1) || !sameProjectedValue(i->second, j->second)) return false;
        return true;
    }
    if (a.IsHolding<VtDictionary>()) {
        const auto& x = a.UncheckedGet<VtDictionary>();
        const auto& y = b.UncheckedGet<VtDictionary>();
        if (x.size() != y.size()) return false;
        for (auto i = x.begin(), j = y.begin(); i != x.end(); ++i, ++j)
            if (i->first != j->first || !sameProjectedValue(i->second, j->second)) return false;
        return true;
    }
    if (a.IsHolding<std::vector<double>>()) {
        const auto& x = a.UncheckedGet<std::vector<double>>();
        const auto& y = b.UncheckedGet<std::vector<double>>();
        return x.size() == y.size() && sameBits(x.data(), y.data(), x.size());
    }
    if (const auto same = sameFloatingAny<GfHalf, float, double, SdfTimeCode, GfMatrix2d, GfMatrix3d, GfMatrix4d,
            GfQuatd, GfQuatf, GfQuath, GfVec2d, GfVec2f, GfVec2h, GfVec3d, GfVec3f, GfVec3h, GfVec4d, GfVec4f, GfVec4h,
            GfRange1d, GfRange1f, GfRange2d, GfRange2f, GfRange3d, GfRange3f>(a, b))
        return *same;
    if (const auto same = sameComposite<SdfLayerOffset>(a, b)) return *same;
    if (const auto same = sameComposite<SdfReference>(a, b)) return *same;
    if (const auto same = sameComposite<SdfPayload>(a, b)) return *same;
    if (const auto same = sameComposite<SdfUnregisteredValue>(a, b)) return *same;
    return a == b;
}
bool sameProjectedFields(const std::map<TfToken, VtValue>& x, const std::map<TfToken, VtValue>& y) {
    if (x.size() != y.size()) return false;
    for (auto i = x.begin(), j = y.begin(); i != x.end(); ++i, ++j)
        if (i->first != j->first || !sameProjectedValue(i->second, j->second)) return false;
    return true;
}
} // namespace

void setWireSceneDerivationCheck(bool enabled) noexcept { derivationCheck.store(enabled, std::memory_order_relaxed); }
WireSceneDifference diffProjectedScenes(const WireScene& before, const WireScene& after) {
    WireSceneDifference difference;
    if (&before == &after) return difference;
    using Section = WireScene::State::Section;
    const auto& old = *before.state_;
    const auto& next = *after.state_;
    // Sections leaving or entering the projection, and those it keeps with
    // their type, by ID. Sections the two scenes share are equal and skipped.
    struct Kept {
        uint64_t id;
        const Section *before, *after;
    };
    std::set<uint64_t> removed, created;
    std::vector<Kept> kept;
    decltype(old.sections)::difference(old.sections, next.sections,
        [&](uint64_t id, const Section* x, const Section* y) {
        if (x && x->hidden) x = nullptr;
        if (y && y->hidden) y = nullptr;
        if (x && y && x->type == y->type) kept.push_back({id, x, y});
        else {
            if (x) removed.insert(id);
            if (y) created.insert(id);
        }
    });
    // A path whose spec is the same section in both scenes.
    const auto stable = [&](const SdfPath& path) {
        const auto* o = old.paths.find(path);
        const auto* n = next.paths.find(path);
        return o && n && *o == *n;
    };
    const auto movable = [](const SdfPath& from, const SdfPath& to) {
        return (from.IsPrimPath() && to.IsPrimPath()) || (from.IsPrimPropertyPath() && to.IsPrimPropertyPath());
    };
    std::vector<std::pair<const Section*, const Section*>> changed;
    changed.reserve(kept.size());
    for (const auto& [id, x, y] : kept) {
        if (x->path != y->path) {
            difference.erased.push_back(x->path);
            difference.written.push_back(y->path);
        }
        // A section with its parent and name follows any path change of its parent.
        if (x->path != y->path && (x->parent != y->parent || x->name != y->name)) {
            if (movable(x->path, y->path) && stable(x->path.GetParentPath()) && stable(y->path.GetParentPath()) &&
                !next.paths.find(x->path) && !old.paths.find(y->path))
                difference.moved.emplace_back(x->path, y->path);
            else {
                // Recorded as a removal and a creation below.
                difference.erased.pop_back();
                difference.written.pop_back();
                removed.insert(id);
                created.insert(id);
                continue;
            }
        }
        if (!sameProjectedFields(x->fields, y->fields)) changed.emplace_back(x, y);
    }
    for (const auto id : removed) {
        const auto& section = *old.sections.find(id);
        difference.erased.push_back(section.path);
        if (!removed.count(section.parent)) difference.removed.push_back(section.path);
    }
    std::sort(difference.removed.begin(), difference.removed.end(),
        [](const SdfPath& l, const SdfPath& r) { return r < l; });
    for (const auto id : created) {
        const auto& section = *next.sections.find(id);
        difference.created.emplace_back(section.path, section.type);
        difference.written.push_back(section.path);
    }
    std::sort(difference.created.begin(), difference.created.end());
    for (const auto& [x, y] : changed) {
        const bool rewritten = x->path != y->path;
        auto f = x->fields.begin(), g = y->fields.begin();
        while (f != x->fields.end() || g != y->fields.end()) {
            if (g == y->fields.end() || (f != x->fields.end() && f->first < g->first)) {
                difference.fields.push_back({y->path, f->first, f->second, VtValue(), rewritten});
                ++f;
            } else if (f == x->fields.end() || g->first < f->first) {
                difference.fields.push_back({y->path, g->first, VtValue(), g->second, rewritten});
                ++g;
            } else {
                if (!sameProjectedValue(f->second, g->second))
                    difference.fields.push_back({y->path, f->first, f->second, g->second, rewritten});
                ++f;
                ++g;
            }
        }
    }
    return difference;
}

WireScenePtr buildSceneCandidate(const WireScenePtr& previous, std::string_view bytes, const WireSceneLimits& limits,
    WireSceneWorkBudget* shared, WireSceneBatchPolicy policy) {
    const auto decoded = decodeUpdate(bytes, limits, shared, policy);
    const bool replay = policy == WireSceneBatchPolicy::PendingReplay && previous;
    WireSceneBuilder builder(previous, limits, replay, policy == WireSceneBatchPolicy::Delivered);
    try {
        builder.apply(decoded.envelope, decoded.values, shared);
        return builder.finish();
    } catch (const PredictedRejection&) {
        if (!replay) throw;
        return previous;
    }
}

WireScenePtr buildDeliveredScene(const WireScenePtr& previous, const std::vector<WireSceneDelivery>& updates,
    const WireSceneLimits& limits) {
    if (updates.empty()) throw std::invalid_argument("no delivered wire scene updates");
    WireSceneBuilder builder(previous, limits, false, true);
    for (const auto& update : updates) {
        const auto decoded = decodeUpdate(update.bytes, limits, update.work, WireSceneBatchPolicy::Delivered);
        builder.apply(decoded.envelope, decoded.values, update.work);
    }
    return builder.finish();
}

WireScenePtr buildReplayedScene(const WireScenePtr& previous, const std::vector<std::string_view>& intents,
    const WireSceneLimits& limits, WireSceneWorkBudget* work, bool strictTail, std::vector<WireReplayStep>* steps) {
    WireSceneBuilder builder(previous, limits, false, false);
    if (steps) {
        steps->clear();
        steps->reserve(intents.size());
    }
    for (size_t i = 0; i < intents.size(); ++i) {
        const bool strict = strictTail && i + 1 == intents.size();
        const auto before = work ? work->remaining : 0;
        const auto decoded = decodeUpdate(intents[i], limits, work,
            strict ? WireSceneBatchPolicy::OrdinaryUpdate : WireSceneBatchPolicy::PendingReplay);
        const bool applied = builder.replayIntent(decoded.envelope, decoded.values, work, strict);
        if (steps)
            steps->push_back({applied, work ? before - work->remaining : 0, builder.derivedWork(), builder.derived(),
                builder.rederives(), builder.fieldCount(), builder.metadataBytes()});
    }
    return builder.finish();
}

WireScenePtr buildOverlaidScene(const WireScenePtr& previous, const std::vector<std::string_view>& updates,
    const WireSceneLimits& limits, WireSceneWorkBudget* work, const std::function<bool(const WireRecordKey&)>& skip) {
    if (!previous || updates.empty() || !skip) throw std::invalid_argument("invalid wire scene overlay");
    WireSceneBuilder builder(previous, limits, false, true);
    builder.skipRecords(&skip);
    for (const auto& update : updates) {
        const auto decoded = decodeUpdate(update, limits, work, WireSceneBatchPolicy::Delivered);
        if (decoded.envelope.isDiff || !decoded.envelope.commands.empty())
            throw std::invalid_argument("structural wire scene overlay");
        builder.apply(decoded.envelope, decoded.values, work);
    }
    return builder.finish();
}
} // namespace omni::usd::live
