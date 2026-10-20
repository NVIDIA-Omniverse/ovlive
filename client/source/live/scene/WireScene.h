// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "UsdValueCodec.h"
#include <pxr/usd/sdf/abstractData.h>
#include <functional>
#include <memory>
#include <string_view>
#include <vector>

namespace omni::usd::live {
struct WireSceneLimits {
    wire::EnvelopeLimits envelope;
    wire::ValueLimits values;
    // A layer is as large as the service holds, so its retained size is bounded
    // by memory, not by these defaults.
    wire::EncodeLimits retainedValues; // Logical encoded size/work of the complete retained scene.
    size_t maxSections = size_t(1) << 32;
    size_t maxFields = size_t(1) << 34;
    size_t maxMetadataBytes = size_t(16) << 30;
    size_t maxPathBytes = size_t(64) << 30;
    size_t maxPathDepth = 256;
    // Logical graph/field/path work per update beyond the work proportional to
    // the previous scene and the update's records, which every update is allowed.
    size_t maxWork = 16u * 1024 * 1024;
};
class WireScene;
// Optional transaction-wide logical work/allocation charge shared by successive
// candidates. Caller owns exclusive access; failed calls may consume budget.
struct WireSceneWorkBudget {
    size_t remaining;
};
// PendingReplay is OrdinaryUpdate for a retained local intent replayed onto a
// newer authoritative scene: when a Create names an identity a committed peer
// update has since created, the service rejects the whole update, so the
// candidate is `previous` unchanged and none of the intent's commands, fields
// or samples apply. When a Reorder names a child that a committed peer update
// has since removed or moved, the Reorder ignores it, as USD's list ordering
// ignores a name that is not in the list, and orders the current children it
// names, as a service that orders lists per child list commits it; a service
// that stores lists verbatim rejects the update, and that rejection fails the
// layer. The peer update itself still applies.
// Delivered is Any for an update the service delivered. The service never
// commits a Create of a live identity, so a delivered one repeats a commit the
// scene holds: a catch-up without a proven stable cut acks its pin and
// delivers again the commits its snapshot folded in. The Create changes
// nothing. A delivered Reorder, like a replayed one, is exactly USD's
// SdfApplyListOrdering on each child list: it ignores non-children and every
// repeat after an ID's first occurrence.
enum class WireSceneBatchPolicy { Any, OrdinaryUpdate, PendingReplay, Delivered };
using WireScenePtr = std::shared_ptr<const WireScene>;
struct WireFieldChange {
    uint64_t section = 0, setOrder = 0;
    size_t record = 0;
    PXR_NS::TfToken field;
};
struct WireSectionView {
    uint64_t id, parent;
    PXR_NS::SdfSpecType type;
    std::string_view name;
    uint32_t order;
    const PXR_NS::SdfPath& path;
};
// The difference between two projected scenes of one layer, matched by section
// identity: the Sdf spec and field changes that turn the first scene's
// projection into the second's. A section that is hidden counts as absent.
struct WireSceneDifference {
    struct Field {
        PXR_NS::SdfPath path; // In the second scene.
        PXR_NS::TfToken field;
        PXR_NS::VtValue before, after; // Empty where the field is absent.
        bool rewritten = false; // The spec changed path, so `written` holds it.
    };
    // Specs absent from the second projection, at their first-scene paths,
    // deepest first. A spec whose parent is also removed is covered by the
    // parent's entry.
    std::vector<PXR_NS::SdfPath> removed;
    // (first path, second path) of a kept spec of the same type that changed
    // parent or name, where the section at each of the two parent paths is the
    // same in both scenes and neither path holds a spec in the other scene.
    // Any other path change of a kept spec is a removal and a creation; a
    // descendant that only follows its parent has no entry.
    std::vector<std::pair<PXR_NS::SdfPath, PXR_NS::SdfPath>> moved;
    // Specs new to the second projection, parents first.
    std::vector<std::pair<PXR_NS::SdfPath, PXR_NS::SdfSpecType>> created;
    // Changed fields of every spec in both projections that is neither
    // removed nor created, in section and field order.
    std::vector<Field> fields;
    // The storage edit: every first-scene path whose spec leaves the
    // projection or changes path, and every second-scene path whose spec
    // enters it or changes path; `fields` that are not rewritten cover the
    // other specs.
    std::vector<PXR_NS::SdfPath> erased, written;
    bool empty() const noexcept {
        return removed.empty() && moved.empty() && created.empty() && fields.empty() && erased.empty() &&
            written.empty();
    }
};
// Visits only the sections the two scenes do not share: a scene built on
// another shares its unchanged sections and index structure, so comparing a
// scene with one built on it costs work in proportion to the sections the
// build changed, with a logarithmic factor in the scene; unrelated scenes cost
// a pass over both. Values compare as their wire
// projection does: floating-point content, including time-sample keys and
// layer offsets, by representation, so a signed zero differs from zero and an
// offset within USD's comparison tolerance differs, and shared array storage
// by identity first; other values with VtValue equality.
WireSceneDifference diffProjectedScenes(const WireScene& before, const WireScene& after);
// Test hook, off by default: while enabled, every update a scene build applies
// also derives the whole working scene from its sections and throws
// std::logic_error unless the incremental derivation, children indexes and
// path index equal it. Production code never enables it.
void setWireSceneDerivationCheck(bool enabled) noexcept;
// Immutable candidate, including section identity and exact Sdf storage. Holding
// one owner supplies a coherent read view; later candidates never mutate it.
// No transport frontier, intent retirement or public-layer publication occurs
// here. The attachment must publish this together with its intent ledger.
class WireScene final {
public:
    ~WireScene();
    // USD's ConstPtr is weak. Retain this WireScene owner while using the view.
    const PXR_NS::SdfAbstractDataConstPtr& data() const;
    PXR_NS::SdfPath pathForSection(uint64_t) const;
    uint64_t sectionForPath(const PXR_NS::SdfPath&) const;
    // Whether the graph holds this section, visible or not.
    bool containsSection(uint64_t) const;
    size_t sectionCount() const;
    // A logical estimate of one retained representation of this scene: encoded
    // values with per-element bookkeeping, names, paths and per-section and
    // per-field overhead. Budgets that pin or copy a scene scale with it.
    size_t retainedCharge() const;
    // Sections that own a projected path, root included: the Sdf spec count.
    size_t pathCount() const;
    // The parent's listed children across all child lists in stored order,
    // including shadowed duplicates and excluding children a Reorder omitted.
    // Visits the parent's children only, each counting toward the scene work
    // meter. An unknown parent has none.
    std::vector<uint64_t> listedChildren(uint64_t parent) const;
    // Complete immutable graph in ascending section-ID order, including the
    // root. Names/orders are the stored wire identities, not reconstructed
    // from Sdf fields. A section outside the visible scene (a shadowed
    // duplicate sibling, a child an incoming Reorder omitted, or a descendant
    // of either) reports an empty path. Precheck maxSections before invoking any callback;
    // false stops traversal. No graph/name/path copies are allocated here.
    // View references remain valid only while this scene owner is retained;
    // the view object itself is borrowed for the callback. Exceptions from a
    // visitor propagate without changing the scene. Empty visitor is invalid.
    bool visitSections(size_t maxSections, const std::function<bool(const WireSectionView&)>& visitor) const;
    const std::string& sourceFormat() const;
    const std::vector<WireFieldChange>& fieldChanges() const;
private:
    friend class WireSceneBuilder;
    friend WireSceneDifference diffProjectedScenes(const WireScene&, const WireScene&);
    struct State;
    explicit WireScene(std::unique_ptr<State>);
    std::unique_ptr<State> state_;
};
// Decode/validate the complete wire batch, stage graph and scene, then return
// one complete immutable owner. A null previous scene starts with a pseudoroot.
// Commands follow the service's conflict rules (design section 20): missing
// targets are skipped, orphan field and sample data is dropped, occupied names
// resolve by deleting the occupant or the mover. A Reorder orders each child
// list it names on its own, exactly as USD's SdfApplyListOrdering does, within
// the positions that list holds among the parent's children, and keeps every
// child.
// Among siblings sharing a derived path the lowest ID owns it; the others stay
// in the graph without projection. DecodeError(Unsupported) marks
// a graph the client cannot represent; callers retain the batch and intents.
// Standard allocation exceptions may propagate as well, with the same previous
// state preservation. Limits bound logical storage/work, not SDK caches or RSS.
// The result shares `previous`'s unchanged sections, path index and Sdf
// storage, and measures and stores only the sections the update changes. An
// update without structural commands derives nothing, so it costs work in
// proportion to its records and the sections they address, with a
// logarithmic factor in the scene. A structural command derives the paths of
// the sections it moves and their subtrees again and resolves the shadowing
// among their siblings, so it also costs those subtrees and the children
// lists of the parents it changes.
WireScenePtr buildSceneCandidate(const WireScenePtr& previous, std::string_view bytes, const WireSceneLimits& = {},
    WireSceneWorkBudget* = nullptr, WireSceneBatchPolicy = WireSceneBatchPolicy::Any);
// One delivered update and the transaction budget its own build is charged.
struct WireSceneDelivery {
    std::string_view bytes;
    WireSceneWorkBudget* work = nullptr;
};
// The scene that successive Delivered candidates of `updates` in order would
// produce, building it once: each update is decoded, validated and charged as
// its own candidate would be, while the scene-wide measurement and Sdf storage
// happen once, charged to the last update's budget. A failing update returns
// nothing.
// The scene that successive candidates of `intents` in order would produce,
// each a PendingReplay candidate on the scene so far and, when `strictTail`,
// the last an OrdinaryUpdate candidate, building it once: an intent the
// service predictably rejects leaves the scene as it was, every intent is
// decoded, validated and charged to `work`, and the scene-wide measurement
// and Sdf storage happen once. With no intent, or none applied, the result
// is `previous`. A failing intent returns nothing.
// With `steps`, one entry per intent reports what its replay did.
struct WireReplayStep {
    // False when the intent is a predicted service rejection, which leaves
    // the scene as it was.
    bool applied = false;
    // The work its decoding and application charged to `work`; the scene's
    // single measurement and Sdf storage are charged to no intent.
    size_t work = 0;
    // The part of `work` that derived the child groups of a scene with
    // current paths for a command or child field that reads them, whether
    // the working scene's child groups are derived after the intent, and
    // whether the next intent derives them again when it reads them. Every
    // update keeps paths, child groups and children lists current, deriving
    // only what it changes, so `derivedWork` is zero, `derived` true and
    // `rederives` false: no step's work depends on the steps before it.
    size_t derivedWork = 0;
    bool derived = false;
    bool rederives = false;
    // The working scene's field count and metadata bytes (source format,
    // section names and field names) after the intent, which the field and
    // metadata limits bound.
    size_t fields = 0, metadataBytes = 0;
};
WireScenePtr buildReplayedScene(const WireScenePtr& previous, const std::vector<std::string_view>& intents,
    const WireSceneLimits& = {}, WireSceneWorkBudget* work = nullptr, bool strictTail = false,
    std::vector<WireReplayStep>* steps = nullptr);
WireScenePtr buildDeliveredScene(const WireScenePtr& previous, const std::vector<WireSceneDelivery>& updates,
    const WireSceneLimits& = {});
// A field or sample record of an update: its section and, for a field
// record, its field name; a sample record names no field.
struct WireRecordKey {
    uint64_t section;
    std::string_view field;
    bool sample;
};
// The scene that successive Delivered candidates of `updates` in order would
// produce from `previous`, except that every field and sample record `skip`
// selects is dropped; each update is decoded, validated and charged to
// `work` as its own candidate would be, and the scene is measured and stored
// once. Every update must be an ordinary update without structural commands,
// so the graph, paths and child fields stay those of `previous`. A failing
// update returns nothing.
WireScenePtr buildOverlaidScene(const WireScenePtr& previous, const std::vector<std::string_view>& updates,
    const WireSceneLimits&, WireSceneWorkBudget* work, const std::function<bool(const WireRecordKey&)>& skip);
} // namespace omni::usd::live
