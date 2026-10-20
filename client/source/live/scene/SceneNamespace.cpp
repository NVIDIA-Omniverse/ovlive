// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "SceneNamespace.h"
#include "UnsupportedOperation.h"
#include <pxr/usd/sdf/childrenPolicies.h>
#include <pxr/usd/sdf/schema.h>
#include <algorithm>
#include <map>
#include <optional>
#include <set>

using namespace PXR_NS;
namespace omni::usd::live {
namespace {
[[noreturn]] void reject(wire::ErrorKind kind, const char* message) { throw wire::DecodeError(kind, message); }
[[noreturn]] void malformed(const char* message) { reject(wire::ErrorKind::Malformed, message); }
// Every unsupported plan names its reason; no single operation causes it.
[[noreturn]] void unsupported(UnsupportedOperation::Reason reason) { throw UnsupportedOperation(reason); }
void spend(size_t amount, size_t& remaining) {
    if (amount > remaining) reject(wire::ErrorKind::ResourceLimit, "namespace planning budget exceeded");
    remaining -= amount;
}
}
SdfPath namespaceParentPath(SdfSpecType child, const SdfPath& path) {
    if (child == SdfSpecTypePseudoRoot) return {};
    return child == SdfSpecTypeVariant ? Sdf_VariantChildPolicy::GetParentPath(path) : path.GetParentPath();
}
unsigned namespaceChildList(SdfSpecType parent, SdfSpecType child) {
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
    malformed("incompatible namespace parent and child types");
}
TfToken namespaceChildField(SdfSpecType parent, unsigned list) {
    if (parent == SdfSpecTypePseudoRoot) return list == 0 ? SdfChildrenKeys->PrimChildren : TfToken();
    if (parent == SdfSpecTypePrim || parent == SdfSpecTypeVariant) {
        if (list == 0) return SdfChildrenKeys->PrimChildren;
        if (list == 1) return SdfChildrenKeys->PropertyChildren;
        if (list == 2) return SdfChildrenKeys->VariantSetChildren;
    }
    if (parent == SdfSpecTypeVariantSet && list == 0) return SdfChildrenKeys->VariantChildren;
    if (parent == SdfSpecTypeAttribute) {
        if (list == 0) return SdfChildrenKeys->ConnectionChildren;
        if (list == 1) return SdfChildrenKeys->ExpressionChildren;
        if (list == 2) return SdfChildrenKeys->MapperChildren;
    }
    if (parent == SdfSpecTypeRelationship) {
        if (list == 0) return SdfChildrenKeys->RelationshipTargetChildren;
        if (list == 1) return SdfChildrenKeys->ExpressionChildren;
    }
    if (parent == SdfSpecTypeRelationshipTarget && list == 0) return SdfChildrenKeys->PropertyChildren;
    if (parent == SdfSpecTypeMapper && list == 0) return SdfChildrenKeys->MapperArgChildren;
    return {};
}
VtValue namespaceChildKey(SdfSpecType child, const SdfPath& path) {
    if (child == SdfSpecTypeConnection || child == SdfSpecTypeRelationshipTarget || child == SdfSpecTypeMapper)
        return VtValue(path.GetTargetPath());
    if (child == SdfSpecTypeVariant) return VtValue(TfToken(path.GetVariantSelection().second));
    if (child == SdfSpecTypeVariantSet) return VtValue(TfToken(path.GetVariantSelection().first));
    if (child == SdfSpecTypeExpression) return VtValue(TfToken("expression"));
    if (child == SdfSpecTypePseudoRoot) malformed("pseudoroot has no namespace child key");
    return VtValue(path.GetNameToken());
}
namespace {
struct Meter {
    WireSceneWorkBudget work;
    WireSceneWorkBudget* shared;
    size_t initial;
    Meter(size_t limit, WireSceneWorkBudget* owner)
        : work{owner ? std::min(limit, owner->remaining) : limit}, shared(owner), initial(work.remaining) {}
    ~Meter() {
        if (shared) shared->remaining -= initial - work.remaining;
    }
    void charge(size_t amount = 1) { spend(amount, work.remaining); }
};
struct Node {
    NamespaceSpec spec;
    uint64_t parent = 0;
    std::string name;
};
using Graph = std::map<uint64_t, Node>;

// The validation, metering and encoding both planners share.
class PlanCore {
protected:
    const WireScenePtr& base_;
    const SceneNamespaceLimits& limits_;
    const std::function<uint64_t()>& allocate_;
    Meter meter_;
    size_t metadata_;
    std::set<uint64_t> used_, staged_;
    NamespacePlan plan_;
    bool built_ = true; // The candidate reflects every planned command.

    PlanCore(const WireScenePtr& base, const SceneNamespaceLimits& limits, const std::function<uint64_t()>& allocate,
        WireSceneWorkBudget* shared)
        : base_(base), limits_(limits), allocate_(allocate), meter_(limits.maxWork, shared),
          metadata_(limits.maxMetadataBytes) {}
    void nodes(const SdfPath& path, size_t& remaining) {
        for (const auto& ancestor : path.GetAncestorsRange()) {
            meter_.charge();
            if (!ancestor.IsAbsoluteRootPath()) spend(1, remaining);
            if (ancestor.IsTargetPath() || ancestor.IsMapperPath()) nodes(ancestor.GetTargetPath(), remaining);
        }
    }
    void path(const SdfPath& value) {
        if (value.IsEmpty() || !value.IsAbsolutePath()) malformed("namespace spec requires an absolute nonempty path");
        size_t remaining = limits_.scene.maxPathDepth;
        nodes(value, remaining); // Bounds recursive targets before SDK cached text.
        auto encoding = limits_.scene.retainedValues;
        encoding.maxDepth = limits_.scene.maxPathDepth;
        encoding.maxEncodedBytes = std::min(
            {encoding.maxEncodedBytes, limits_.scene.maxPathBytes, meter_.work.remaining / 2});
        encoding.maxStringBytes = std::min(encoding.maxStringBytes, encoding.maxEncodedBytes);
        encoding.maxElements = std::min(encoding.maxElements, meter_.work.remaining / 2);
        size_t elements = 0;
        const auto bytes = wire::measureValues(wire::DecodedValues{{VtValue(value)}, {}}, encoding, &elements);
        meter_.charge(bytes);
        meter_.charge(elements);
    }
    bool absoluteTargets(const SdfPath& target) {
        if (!target.IsAbsolutePath()) return false;
        for (const auto& ancestor : target.GetAncestorsRange()) {
            meter_.charge();
            if ((ancestor.IsTargetPath() || ancestor.IsMapperPath()) && !absoluteTargets(ancestor.GetTargetPath()))
                return false;
        }
        return true;
    }
    Node node(const NamespaceSpec& spec) {
        meter_.charge(sizeof(Node));
        path(spec.path);
        bool shape = false;
        switch (spec.type) {
        case SdfSpecTypePseudoRoot: shape = spec.id == 1 && spec.path.IsAbsoluteRootPath(); break;
        case SdfSpecTypePrim: shape = spec.path.IsPrimPath() && !spec.path.IsAbsoluteRootPath(); break;
        case SdfSpecTypeAttribute:
            shape = spec.path.IsPrimPropertyPath() || spec.path.IsRelationalAttributePath();
            break;
        case SdfSpecTypeRelationship: shape = spec.path.IsPrimPropertyPath(); break;
        case SdfSpecTypeConnection:
        case SdfSpecTypeRelationshipTarget: shape = spec.path.IsTargetPath(); break;
        case SdfSpecTypeExpression: shape = spec.path.IsExpressionPath(); break;
        case SdfSpecTypeMapper: shape = spec.path.IsMapperPath(); break;
        case SdfSpecTypeMapperArg: shape = spec.path.IsMapperArgPath(); break;
        case SdfSpecTypeVariantSet:
            shape = spec.path.IsPrimVariantSelectionPath() && spec.path.GetVariantSelection().second.empty();
            break;
        case SdfSpecTypeVariant:
            shape = spec.path.IsPrimVariantSelectionPath() && !spec.path.GetVariantSelection().second.empty();
            break;
        default: break;
        }
        if (!shape || !spec.id || (spec.id == 1 && spec.type != SdfSpecTypePseudoRoot))
            malformed("invalid namespace identity/type/path");
        Node result{spec, 0, {}};
        if (spec.id == 1) return result;
        if (spec.type == SdfSpecTypeMapper && !absoluteTargets(spec.path.GetTargetPath()))
            unsupported(UnsupportedOperation::Reason::IdentityRealization); // Mapper target is not canonical absolute.
        const auto key = namespaceChildKey(spec.type, spec.path);
        const auto& text = key.IsHolding<TfToken>() ? key.UncheckedGet<TfToken>().GetString() :
                                                      key.UncheckedGet<SdfPath>().GetString();
        meter_.charge(text.size());
        result.name = text;
        const bool ordinary = spec.type == SdfSpecTypePrim || spec.type == SdfSpecTypeVariantSet ||
            spec.type == SdfSpecTypeMapperArg;
        if ((ordinary && !SdfSchema::IsValidIdentifier(text)) ||
            ((spec.type == SdfSpecTypeAttribute || spec.type == SdfSpecTypeRelationship) &&
                !SdfSchema::IsValidNamespacedIdentifier(text)) ||
            (spec.type == SdfSpecTypeVariant && !SdfSchema::IsValidVariantIdentifier(text)))
            malformed("invalid namespace child name");
        return result;
    }
    std::string encode(const wire::EnvelopeInput& input) {
        // Structural-only tables: reserve a conservative envelope size before
        // FlatBuffers allocates its initial buffer, grows overlapping storage,
        // builds offset/tag vectors, and creates the final owned output.
        size_t allowance = meter_.work.remaining;
        spend(1024, allowance);
        for (const auto& command : input.commands) {
            spend(256, allowance);
            spend(command.name.size(), allowance);
        }
        const auto upper = meter_.work.remaining - allowance;
        if (upper > meter_.work.remaining / 5)
            reject(wire::ErrorKind::ResourceLimit, "namespace encoding work limit exceeded");
        meter_.charge(upper * 5);
        wire::EnvelopeEncodeLimits encoding;
        encoding.content = limits_.scene.envelope;
        encoding.content.maxBytes = std::min(encoding.content.maxBytes, upper);
        encoding.content.maxStringBytes = std::min(encoding.content.maxStringBytes, limits_.maxMetadataBytes);
        encoding.maxBuilderBytes = std::min(encoding.maxBuilderBytes, upper * 3);
        encoding.maxScratchBytes = std::min(encoding.maxScratchBytes, upper);
        return wire::encodeEnvelope(input, encoding);
    }
    void admit(const wire::Command& command) {
        if (plan_.commands.size() >= limits_.maxCommands || plan_.commands.size() >= limits_.scene.envelope.maxRecords)
            reject(wire::ErrorKind::ResourceLimit, "namespace command limit exceeded");
        meter_.charge(sizeof(wire::Command));
        meter_.charge(command.name.size());
        spend(command.name.size(), metadata_);
    }
};

class Planner : PlanCore {
    Graph desired_, current_;
    std::map<SdfPath, uint64_t> desiredPaths_, currentPaths_;

    void link(Graph& graph, const std::map<SdfPath, uint64_t>& paths) {
        const auto root = graph.find(1);
        if (root == graph.end() || root->second.spec.type != SdfSpecTypePseudoRoot ||
            !root->second.spec.path.IsAbsoluteRootPath())
            malformed("complete namespace requires pseudoroot identity 1");
        for (auto& [id, child] : graph) {
            meter_.charge();
            if (id == 1) continue;
            const auto parent = paths.find(namespaceParentPath(child.spec.type, child.spec.path));
            if (parent == paths.end()) malformed("dangling namespace parent");
            child.parent = parent->second;
            (void)namespaceChildList(graph.at(child.parent).spec.type, child.spec.type);
            // Every reverse policy removes a path component (Variant changes
            // its selected value to empty), so this also proves acyclicity.
        }
    }
    void refresh() {
        current_.clear();
        currentPaths_.clear();
        struct Visitor final : SdfAbstractDataSpecVisitor {
            Planner& owner;
            explicit Visitor(Planner& p) : owner(p) {}
            bool VisitSpec(const SdfAbstractData& data, const SdfPath& path) override {
                const auto id = owner.plan_.candidate->sectionForPath(path);
                auto child = owner.node({id, data.GetSpecType(path), path});
                owner.current_.emplace(id, std::move(child));
                owner.currentPaths_.emplace(path, id);
                return true;
            }
            void Done(const SdfAbstractData&) override {}
        } visitor(*this);
        plan_.candidate->data()->VisitSpecs(&visitor);
        link(current_, currentPaths_);
    }
    bool below(uint64_t id, uint64_t ancestor) {
        while (id) {
            meter_.charge();
            if (id == ancestor) return true;
            id = current_.at(id).parent;
        }
        return false;
    }
    // Bring the candidate up to every planned command: Creates of desired
    // identities update the graph without a build, so one build covers them.
    void materialize() {
        if (built_) return;
        wire::EnvelopeInput input;
        input.commands = plan_.commands;
        plan_.candidate = buildSceneCandidate(base_, encode(input), limits_.scene, &meter_.work,
            WireSceneBatchPolicy::OrdinaryUpdate);
        built_ = true;
    }
    void apply(wire::Command command) {
        admit(command);
        materialize();
        wire::EnvelopeInput input;
        input.commands.push_back(command);
        const auto bytes = encode(input);
        auto next = buildSceneCandidate(plan_.candidate, bytes, limits_.scene, &meter_.work,
            WireSceneBatchPolicy::OrdinaryUpdate);
        plan_.commands.push_back(std::move(command));
        plan_.candidate = std::move(next);
        refresh();
    }
    // A desired identity created under its parent at its desired path, which
    // advance() checks is free, lands exactly as its desired node: record it
    // without building the scene. The final whole-batch build verifies it.
    void createDesired(uint64_t id, const Node& target) {
        meter_.charge(target.name.size());
        wire::Command c{wire::Command::Kind::Create};
        c.section = id;
        c.parent = target.parent;
        c.sectionType = uint8_t(target.spec.type);
        c.name = target.name;
        admit(c);
        meter_.charge(sizeof(Node));
        current_.emplace(id, target);
        currentPaths_.emplace(target.spec.path, id);
        plan_.commands.push_back(std::move(c));
        built_ = false;
    }
    void create(uint64_t id, uint64_t parent, SdfSpecType type, const std::string& name) {
        meter_.charge(name.size());
        wire::Command c{wire::Command::Kind::Create};
        c.section = id;
        c.parent = parent;
        c.sectionType = uint8_t(type);
        c.name = name;
        apply(std::move(c));
    }
    void move(uint64_t id, uint64_t parent, const std::string& name) {
        meter_.charge(name.size());
        wire::Command c{wire::Command::Kind::Move};
        c.section = id;
        c.parent = parent;
        c.oldParent = current_.at(id).parent;
        c.name = name;
        apply(std::move(c));
    }
    bool advance() {
        // Delete only a whole subtree with no desired survivor. A descendant
        // that escapes a deleted ancestor must first move with its original ID.
        // Select the highest disappearing ancestor, independently of ID order.
        // Enumerating obsolete descendants would give them separate destructive
        // targets even if a peer moved them outside that ancestor before service
        // execution. Wait for desired survivors to escape instead of deleting
        // their disappearing siblings while the ancestor is still blocked.
        for (const auto& [id, child] : current_) {
            meter_.charge();
            if (desired_.count(id)) continue;
            bool disappearingAncestor = false;
            for (auto parent = child.parent; parent; parent = current_.at(parent).parent) {
                meter_.charge();
                if (!desired_.count(parent)) {
                    disappearingAncestor = true;
                    break;
                }
            }
            if (disappearingAncestor) continue;
            bool survivor = false;
            for (const auto& [other, desired] : desired_) {
                (void)desired;
                meter_.charge();
                if (current_.count(other) && below(other, id)) {
                    survivor = true;
                    break;
                }
            }
            if (survivor) continue;
            wire::Command c{wire::Command::Kind::Delete};
            c.section = id;
            c.parent = current_.at(id).parent;
            apply(std::move(c));
            return true;
        }
        for (const auto& [id, target] : desired_) {
            meter_.charge();
            if (id == 1) continue;
            const auto found = current_.find(id);
            if (found != current_.end() && found->second.spec.path == target.spec.path &&
                found->second.parent == target.parent)
                continue;
            const auto parent = current_.find(target.parent);
            if (parent == current_.end() || parent->second.spec.path != desired_.at(target.parent).spec.path ||
                currentPaths_.count(target.spec.path))
                continue;
            if (found == current_.end()) createDesired(id, target);
            else {
                if (below(target.parent, id)) continue;
                move(id, target.parent, target.name);
            }
            return true;
        }
        return false;
    }
    uint64_t fresh() {
        meter_.charge(256);
        if (!allocate_) unsupported(UnsupportedOperation::Reason::MoveCycle); // No scratch identity allocator.
        const auto id = allocate_();
        if (id <= 1 || !used_.insert(id).second)
            malformed("namespace scratch allocator returned a reused or invalid identity");
        return id;
    }
    void stage(uint64_t id) {
        meter_.charge(sizeof(Node));
        meter_.charge(current_.at(id).name.size());
        const auto child = current_.at(id); // Own metadata across candidate replacement.
        uint64_t parent;
        for (;;) {
            parent = fresh();
            const std::string name = "__live_namespace_" + std::to_string(parent);
            const auto path = SdfPath::AbsoluteRootPath().AppendChild(TfToken(name));
            if (currentPaths_.count(path) || desiredPaths_.count(path)) continue;
            create(parent, 1, SdfSpecTypePrim, name);
            break;
        }
        auto add = [&](SdfSpecType type, const char* name) {
            const auto next = fresh();
            create(next, parent, type, name);
            parent = next;
        };
        switch (child.spec.type) {
        case SdfSpecTypePrim:
        case SdfSpecTypeAttribute:
        case SdfSpecTypeRelationship:
        case SdfSpecTypeVariantSet: break;
        case SdfSpecTypeVariant: add(SdfSpecTypeVariantSet, "set"); break;
        case SdfSpecTypeConnection:
        case SdfSpecTypeExpression:
        case SdfSpecTypeMapper: add(SdfSpecTypeAttribute, "value"); break;
        case SdfSpecTypeRelationshipTarget: add(SdfSpecTypeRelationship, "relationship"); break;
        case SdfSpecTypeMapperArg:
            add(SdfSpecTypeAttribute, "value");
            add(SdfSpecTypeMapper, "/__live_namespace_target");
            break;
        default: unsupported(UnsupportedOperation::Reason::IdentityRealization); // No scratch parent shape.
        }
        move(id, parent, child.name);
        staged_.insert(id);
    }
    bool finished() {
        if (current_.size() != desired_.size()) return false;
        for (const auto& [id, target] : desired_) {
            meter_.charge();
            const auto it = current_.find(id);
            if (it == current_.end() || it->second.spec.path != target.spec.path || it->second.parent != target.parent)
                return false;
        }
        return true;
    }
public:
    Planner(const WireScenePtr& base, const SceneNamespaceLimits& limits, const std::function<uint64_t()>& allocate,
        WireSceneWorkBudget* shared)
        : PlanCore(base, limits, allocate, shared) {}
    NamespacePlan run(const std::vector<NamespaceSpec>& desired) {
        if (!base_) malformed("namespace planning requires a base scene");
        if (!limits_.scene.maxPathDepth || limits_.scene.maxPathDepth > 256)
            throw std::invalid_argument("invalid namespace path depth limit");
        if (desired.size() > limits_.scene.maxSections || base_->sectionCount() > limits_.scene.maxSections)
            reject(wire::ErrorKind::ResourceLimit, "namespace section limit exceeded");
        for (const auto& spec : desired) {
            auto child = node(spec);
            spend(child.name.size(), metadata_);
            spend(spec.path.GetString().size(), metadata_);
            if (!desired_.emplace(spec.id, std::move(child)).second ||
                !desiredPaths_.emplace(spec.path, spec.id).second)
                malformed("duplicate namespace identity or path");
            used_.insert(spec.id);
        }
        link(desired_, desiredPaths_);
        plan_.candidate = base_;
        refresh();
        for (const auto& [id, child] : current_) {
            used_.insert(id);
            const auto target = desired_.find(id);
            if (target != desired_.end() && target->second.spec.type != child.spec.type)
                unsupported(UnsupportedOperation::Reason::IdentityTypeChange);
        }
        while (!finished()) {
            if (advance()) continue;
            uint64_t blocked = 0;
            for (const auto& [id, target] : desired_) {
                meter_.charge();
                const auto it = current_.find(id);
                if (id != 1 && it != current_.end() &&
                    (it->second.spec.path != target.spec.path || it->second.parent != target.parent) &&
                    !staged_.count(id)) {
                    blocked = id;
                    break;
                }
            }
            if (!blocked) unsupported(UnsupportedOperation::Reason::IdentityRealization);
            stage(blocked);
        }
        // Validate the actual single outgoing batch too, including no-op plans
        // under today's limits. Intermediate candidates synchronize child
        // fields after each individual step rather than once for the batch.
        for (const auto& command : plan_.commands) {
            meter_.charge(sizeof(wire::Command));
            meter_.charge(command.name.size());
        }
        wire::EnvelopeInput input;
        input.commands = plan_.commands;
        const auto bytes = encode(input);
        auto validated = buildSceneCandidate(base_, bytes, limits_.scene, &meter_.work,
            WireSceneBatchPolicy::OrdinaryUpdate);
        // A successful no-op can retain the original immutable owner after its
        // full revalidation; it still consumes all validation work above.
        plan_.candidate = plan_.commands.empty() ? base_ : std::move(validated);
        refresh();
        if (!finished())
            reject(wire::ErrorKind::Integrity, "namespace batch projection differs from desired identities");
        plan_.workUsed = meter_.initial - meter_.work.remaining;
        return std::move(plan_);
    }
};
// Plans what Planner plans for the complete desired graph that a
// NamespaceChanges describes, reading the base, the desired graph and each
// candidate through their indexes instead of copying whole graphs. Planner's
// loops act only on current identities outside the desired graph and on
// desired identities, other than the root, that are absent or at another path
// or parent; this planner keeps exactly those two sets, in the same ID order,
// and reclassifies only the identities each new candidate changes, so both
// emit the same commands.
class ChangePlanner : PlanCore {
    std::map<uint64_t, Node> placed_;
    std::map<SdfPath, uint64_t> placedPaths_;
    std::set<uint64_t> removed_;
    // Desired Creates recorded since the candidate was last built.
    std::map<uint64_t, Node> pending_;
    std::map<SdfPath, uint64_t> pendingPaths_;
    // Current identities outside the desired graph, and desired identities
    // other than the root that are absent or at another path or parent.
    std::set<uint64_t> extra_, unfinished_;

    bool visible(const WireScenePtr& scene, uint64_t id) const { return !scene->pathForSection(id).IsEmpty(); }
    // A spec's child key text; desired specs are validated by node().
    std::string key(const NamespaceSpec& spec) {
        const auto value = namespaceChildKey(spec.type, spec.path);
        auto text = value.IsHolding<TfToken>() ? value.UncheckedGet<TfToken>().GetString() :
                                                 value.UncheckedGet<SdfPath>().GetString();
        meter_.charge(text.size());
        return text;
    }
    uint64_t currentAt(const SdfPath& path) {
        meter_.charge();
        if (const auto found = pendingPaths_.find(path); found != pendingPaths_.end()) return found->second;
        return plan_.candidate->sectionForPath(path);
    }
    std::optional<Node> current(uint64_t id) {
        meter_.charge();
        if (const auto found = pending_.find(id); found != pending_.end()) return found->second;
        auto path = plan_.candidate->pathForSection(id);
        if (path.IsEmpty()) return std::nullopt;
        Node result{{id, plan_.candidate->data()->GetSpecType(path), std::move(path)}, 0, {}};
        if (id != 1) {
            result.parent = currentAt(namespaceParentPath(result.spec.type, result.spec.path));
            result.name = key(result.spec);
        }
        return result;
    }
    Node currentNode(uint64_t id) {
        auto result = current(id);
        if (!result) malformed("missing current namespace identity");
        return std::move(*result);
    }
    bool desired(uint64_t id) const { return placed_.count(id) || (!removed_.count(id) && visible(base_, id)); }
    uint64_t desiredAt(const SdfPath& path) {
        meter_.charge();
        if (const auto found = placedPaths_.find(path); found != placedPaths_.end()) return found->second;
        const auto id = base_->sectionForPath(path);
        return id && !removed_.count(id) && !placed_.count(id) ? id : 0;
    }
    // The desired node of `id`; a visible base identity the changes do not
    // name keeps its base type and path under the desired owner of its
    // parent path.
    std::optional<Node> target(uint64_t id) {
        meter_.charge();
        if (const auto found = placed_.find(id); found != placed_.end()) return found->second;
        if (removed_.count(id)) return std::nullopt;
        auto path = base_->pathForSection(id);
        if (path.IsEmpty()) return std::nullopt;
        Node result{{id, base_->data()->GetSpecType(path), std::move(path)}, 0, {}};
        if (id != 1) {
            result.parent = desiredAt(namespaceParentPath(result.spec.type, result.spec.path));
            result.name = key(result.spec);
        }
        return result;
    }
    Node targetNode(uint64_t id) {
        auto result = target(id);
        if (!result) malformed("missing desired namespace identity");
        return std::move(*result);
    }
    // A desired node's parent exists and may hold it.
    void link(const Node& child) {
        if (!child.parent) malformed("dangling namespace parent");
        (void)namespaceChildList(targetNode(child.parent).spec.type, child.spec.type);
    }
    void classify(uint64_t id) {
        meter_.charge();
        const auto now = current(id);
        if (now && !desired(id)) extra_.insert(id);
        else extra_.erase(id);
        const auto goal = id == 1 ? std::nullopt : target(id);
        if (goal && !(now && now->spec.path == goal->spec.path && now->parent == goal->parent)) unfinished_.insert(id);
        else unfinished_.erase(id);
    }
    // The identities whose current node can differ between `before` and
    // `after`: every section entering, leaving or changing path, and the
    // children of a path whose owner changes, which keep their paths under a
    // new parent.
    std::set<uint64_t> changed(const WireScene& before, const WireScenePtr& after) {
        const auto difference = diffProjectedScenes(before, *after);
        std::set<uint64_t> ids;
        for (const auto& path : difference.erased) {
            meter_.charge();
            ids.insert(before.sectionForPath(path));
        }
        for (const auto& path : difference.written) {
            meter_.charge();
            const auto id = after->sectionForPath(path);
            ids.insert(id);
            if (id == before.sectionForPath(path)) continue;
            for (const auto child : after->listedChildren(id)) {
                meter_.charge();
                ids.insert(child);
            }
        }
        ids.erase(0);
        return ids;
    }
    void reseat(WireScenePtr next) {
        const auto before = std::move(plan_.candidate);
        plan_.candidate = std::move(next);
        pending_.clear();
        pendingPaths_.clear();
        for (const auto id : changed(*before, plan_.candidate)) classify(id);
    }
    bool below(uint64_t id, uint64_t ancestor) {
        while (id) {
            meter_.charge();
            if (id == ancestor) return true;
            id = currentNode(id).parent;
        }
        return false;
    }
    void materialize() {
        if (built_) return;
        wire::EnvelopeInput input;
        input.commands = plan_.commands;
        reseat(buildSceneCandidate(base_, encode(input), limits_.scene, &meter_.work,
            WireSceneBatchPolicy::OrdinaryUpdate));
        built_ = true;
    }
    void apply(wire::Command command) {
        admit(command);
        materialize();
        wire::EnvelopeInput input;
        input.commands.push_back(command);
        const auto bytes = encode(input);
        auto next = buildSceneCandidate(plan_.candidate, bytes, limits_.scene, &meter_.work,
            WireSceneBatchPolicy::OrdinaryUpdate);
        plan_.commands.push_back(std::move(command));
        reseat(std::move(next));
    }
    void createDesired(uint64_t id, const Node& target) {
        meter_.charge(target.name.size());
        wire::Command c{wire::Command::Kind::Create};
        c.section = id;
        c.parent = target.parent;
        c.sectionType = uint8_t(target.spec.type);
        c.name = target.name;
        admit(c);
        meter_.charge(sizeof(Node));
        pending_.emplace(id, target);
        pendingPaths_.emplace(target.spec.path, id);
        plan_.commands.push_back(std::move(c));
        built_ = false;
        classify(id);
    }
    void create(uint64_t id, uint64_t parent, SdfSpecType type, const std::string& name) {
        meter_.charge(name.size());
        wire::Command c{wire::Command::Kind::Create};
        c.section = id;
        c.parent = parent;
        c.sectionType = uint8_t(type);
        c.name = name;
        apply(std::move(c));
    }
    void move(uint64_t id, uint64_t parent, const std::string& name) {
        meter_.charge(name.size());
        wire::Command c{wire::Command::Kind::Move};
        c.section = id;
        c.parent = parent;
        c.oldParent = currentNode(id).parent;
        c.name = name;
        apply(std::move(c));
    }
    bool advance() {
        // Planner::advance's first loop over current identities outside the
        // desired graph. A desired identity below one of them is a survivor;
        // walking up from it, the first identity whose current parent is not
        // desired is itself desired and unfinished, so looking for an
        // unfinished one below is the same test.
        for (const auto id : extra_) {
            meter_.charge();
            const auto child = currentNode(id);
            bool disappearingAncestor = false;
            for (auto parent = child.parent; parent; parent = currentNode(parent).parent) {
                meter_.charge();
                if (!desired(parent)) {
                    disappearingAncestor = true;
                    break;
                }
            }
            if (disappearingAncestor) continue;
            bool survivor = false;
            for (const auto other : unfinished_) {
                meter_.charge();
                if (current(other) && below(other, id)) {
                    survivor = true;
                    break;
                }
            }
            if (survivor) continue;
            wire::Command c{wire::Command::Kind::Delete};
            c.section = id;
            c.parent = child.parent;
            apply(std::move(c));
            return true;
        }
        for (const auto id : unfinished_) {
            meter_.charge();
            const auto goal = targetNode(id);
            const auto found = current(id);
            const auto parent = current(goal.parent);
            if (!parent || parent->spec.path != targetNode(goal.parent).spec.path || currentAt(goal.spec.path))
                continue;
            if (!found) createDesired(id, goal);
            else {
                if (below(goal.parent, id)) continue;
                move(id, goal.parent, goal.name);
            }
            return true;
        }
        return false;
    }
    uint64_t fresh() {
        meter_.charge(256);
        if (!allocate_) unsupported(UnsupportedOperation::Reason::MoveCycle); // No scratch identity allocator.
        const auto id = allocate_();
        if (id <= 1 || placed_.count(id) || visible(base_, id) || !used_.insert(id).second)
            malformed("namespace scratch allocator returned a reused or invalid identity");
        return id;
    }
    void stage(uint64_t id) {
        const auto child = currentNode(id); // Own metadata across candidate replacement.
        meter_.charge(sizeof(Node));
        meter_.charge(child.name.size());
        uint64_t parent;
        for (;;) {
            parent = fresh();
            const std::string name = "__live_namespace_" + std::to_string(parent);
            const auto path = SdfPath::AbsoluteRootPath().AppendChild(TfToken(name));
            if (currentAt(path) || desiredAt(path)) continue;
            create(parent, 1, SdfSpecTypePrim, name);
            break;
        }
        auto add = [&](SdfSpecType type, const char* name) {
            const auto next = fresh();
            create(next, parent, type, name);
            parent = next;
        };
        switch (child.spec.type) {
        case SdfSpecTypePrim:
        case SdfSpecTypeAttribute:
        case SdfSpecTypeRelationship:
        case SdfSpecTypeVariantSet: break;
        case SdfSpecTypeVariant: add(SdfSpecTypeVariantSet, "set"); break;
        case SdfSpecTypeConnection:
        case SdfSpecTypeExpression:
        case SdfSpecTypeMapper: add(SdfSpecTypeAttribute, "value"); break;
        case SdfSpecTypeRelationshipTarget: add(SdfSpecTypeRelationship, "relationship"); break;
        case SdfSpecTypeMapperArg:
            add(SdfSpecTypeAttribute, "value");
            add(SdfSpecTypeMapper, "/__live_namespace_target");
            break;
        default: unsupported(UnsupportedOperation::Reason::IdentityRealization); // No scratch parent shape.
        }
        move(id, parent, child.name);
        staged_.insert(id);
    }
    bool finished() const { return extra_.empty() && unfinished_.empty(); }
public:
    ChangePlanner(const WireScenePtr& base, const SceneNamespaceLimits& limits,
        const std::function<uint64_t()>& allocate, WireSceneWorkBudget* shared)
        : PlanCore(base, limits, allocate, shared) {}
    NamespacePlan run(const NamespaceChanges& changes) {
        if (!base_) malformed("namespace planning requires a base scene");
        if (!limits_.scene.maxPathDepth || limits_.scene.maxPathDepth > 256)
            throw std::invalid_argument("invalid namespace path depth limit");
        // The desired graph holds every placed identity, so more placed
        // identities than maxSections exceed the limit before any is read.
        if (base_->sectionCount() > limits_.scene.maxSections || changes.placed.size() > limits_.scene.maxSections)
            reject(wire::ErrorKind::ResourceLimit, "namespace section limit exceeded");
        size_t created = 0;
        for (const auto& spec : changes.placed) {
            created += !visible(base_, spec.id);
            auto child = node(spec);
            spend(child.name.size(), metadata_);
            spend(spec.path.GetString().size(), metadata_);
            if (!placed_.emplace(spec.id, std::move(child)).second || !placedPaths_.emplace(spec.path, spec.id).second)
                malformed("duplicate namespace identity or path");
            used_.insert(spec.id);
        }
        for (const auto id : changes.removed) {
            meter_.charge();
            if (id == 1 || placed_.count(id) || !visible(base_, id) || !removed_.insert(id).second)
                malformed("invalid removed namespace identity");
        }
        // The desired graph's size: the visible base less its removed
        // identities, which are visible and unplaced, plus the placed
        // identities the base does not show. planSceneNamespace bounds the same
        // count.
        if (created >
            limits_.scene.maxSections - std::min(limits_.scene.maxSections, base_->pathCount() - removed_.size()))
            reject(wire::ErrorKind::ResourceLimit, "namespace section limit exceeded");
        for (const auto& [path, id] : placedPaths_) {
            const auto other = base_->sectionForPath(path);
            if (other && other != id && !removed_.count(other) && !placed_.count(other))
                malformed("duplicate namespace identity or path");
        }
        const auto root = target(1);
        if (!root || root->spec.type != SdfSpecTypePseudoRoot || !root->spec.path.IsAbsoluteRootPath())
            malformed("complete namespace requires pseudoroot identity 1");
        // Every reverse policy removes a path component, so linking each
        // placed identity to the owner of its parent path proves acyclicity.
        for (auto& [id, child] : placed_) {
            meter_.charge();
            if (id == 1) continue;
            child.parent = desiredAt(namespaceParentPath(child.spec.type, child.spec.path));
            link(child);
        }
        // The identities whose node may differ from the base's: the placed and
        // removed ones, and the base children of each path they vacate or
        // replace, which stay at their paths under the path's desired owner.
        std::set<uint64_t> initial;
        std::set<SdfPath> vacated;
        for (const auto id : removed_) {
            initial.insert(id);
            vacated.insert(base_->pathForSection(id));
        }
        for (const auto& [id, child] : placed_) {
            initial.insert(id);
            const auto from = base_->pathForSection(id);
            if (!from.IsEmpty() && from != child.spec.path) vacated.insert(from);
        }
        for (const auto& path : vacated) {
            for (const auto child : base_->listedChildren(base_->sectionForPath(path))) {
                meter_.charge();
                if (placed_.count(child) || removed_.count(child)) continue;
                const auto goal = target(child);
                if (!goal) continue; // Not projected.
                link(*goal);
                initial.insert(child);
            }
        }
        for (const auto& [id, child] : placed_) {
            meter_.charge();
            const auto path = base_->pathForSection(id);
            if (!path.IsEmpty() && base_->data()->GetSpecType(path) != child.spec.type)
                unsupported(UnsupportedOperation::Reason::IdentityTypeChange);
        }
        plan_.candidate = base_;
        for (const auto id : initial) classify(id);
        while (!finished()) {
            if (advance()) continue;
            uint64_t blocked = 0;
            for (const auto id : unfinished_) {
                meter_.charge();
                if (current(id) && !staged_.count(id)) {
                    blocked = id;
                    break;
                }
            }
            if (!blocked) unsupported(UnsupportedOperation::Reason::IdentityRealization);
            stage(blocked);
        }
        for (const auto& command : plan_.commands) {
            meter_.charge(sizeof(wire::Command));
            meter_.charge(command.name.size());
        }
        wire::EnvelopeInput input;
        input.commands = plan_.commands;
        const auto bytes = encode(input);
        auto validated = buildSceneCandidate(base_, bytes, limits_.scene, &meter_.work,
            WireSceneBatchPolicy::OrdinaryUpdate);
        // The batch reaches the desired graph if every identity whose node it
        // changes, and every one the changes name, is classified finished.
        auto ids = changed(*base_, validated);
        ids.insert(initial.begin(), initial.end());
        plan_.candidate = plan_.commands.empty() ? base_ : std::move(validated);
        pending_.clear();
        pendingPaths_.clear();
        extra_.clear();
        unfinished_.clear();
        for (const auto id : ids) classify(id);
        if (!finished())
            reject(wire::ErrorKind::Integrity, "namespace batch projection differs from desired identities");
        plan_.workUsed = meter_.initial - meter_.work.remaining;
        return std::move(plan_);
    }
};
}
NamespacePlan planSceneNamespace(const WireScenePtr& base, const std::vector<NamespaceSpec>& desired,
    const std::function<uint64_t()>& allocateScratch, const SceneNamespaceLimits& limits, WireSceneWorkBudget* shared) {
    return Planner(base, limits, allocateScratch, shared).run(desired);
}
NamespacePlan planSceneNamespaceChanges(const WireScenePtr& base, const NamespaceChanges& changes,
    const std::function<uint64_t()>& allocateScratch, const SceneNamespaceLimits& limits, WireSceneWorkBudget* shared) {
    return ChangePlanner(base, limits, allocateScratch, shared).run(changes);
}
} // namespace omni::usd::live
