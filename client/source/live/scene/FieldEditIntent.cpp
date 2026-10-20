// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "FieldEditIntent.h"
#include "HostSceneStorage.h"
#include "SceneNamespace.h"
#include "SceneWork.h"
#include "SdfSceneData.h"
#include <pxr/usd/sdf/schema.h>
#include <algorithm>
#include <atomic>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>

using namespace PXR_NS;
namespace omni::usd::live {
namespace {
using Kind = SdfOperationKind;
using Operation = SdfOperation;
// Encoded bytes already count a scalar's storage; the per-element charge covers container and value bookkeeping.
constexpr size_t elementCost = 16;
[[noreturn]] void fail(wire::ErrorKind kind, const char* message) { throw wire::DecodeError(kind, message); }
void spend(size_t amount, size_t& remaining) {
    if (amount > remaining) fail(wire::ErrorKind::ResourceLimit, "field intent budget exceeded");
    remaining -= amount;
}
void product(size_t count, size_t size, size_t& remaining) {
    if (size && count > remaining / size) fail(wire::ErrorKind::ResourceLimit, "field intent record budget exceeded");
    remaining -= count * size;
}
struct Budget {
    WireSceneWorkBudget work;
    size_t retained;
    explicit Budget(const FieldEditLimits& limits) : work{limits.maxWork}, retained(limits.maxRetainedBytes) {}
};
// Values' encoded bytes and elements, which a measurement charges.
struct Measurement {
    size_t bytes = 0, elements = 0;
};
// `limits` capped by what the remaining work budget can pay for, so that a
// measurement under them never charges more than the budget holds.
wire::EncodeLimits affordable(wire::EncodeLimits limits, const Budget& budget) {
    limits.maxEncodedBytes = std::min(limits.maxEncodedBytes, budget.work.remaining / 2);
    limits.maxStringBytes = std::min(limits.maxStringBytes, limits.maxEncodedBytes);
    limits.maxElements = std::min(limits.maxElements, budget.work.remaining / (2 * elementCost));
    return limits;
}
Measurement measurement(const wire::DecodedValues& values, const wire::EncodeLimits& limits, const Budget& budget) {
    Measurement result;
    result.bytes = wire::measureValues(values, affordable(limits, budget), &result.elements);
    return result;
}
// Charges `measured` to the work budget and returns the amount charged.
size_t charge(const Measurement& measured, Budget& budget) {
    const auto before = budget.work.remaining;
    spend(measured.bytes, budget.work.remaining);
    product(measured.elements, elementCost, budget.work.remaining);
    return before - budget.work.remaining;
}
// Charge traversal and a conservative logical representation before making a
// mutable copy or performing an operation that rewrites a compound value.
size_t measure(const wire::DecodedValues& values, const wire::EncodeLimits& limits, Budget& budget) {
    return charge(measurement(values, limits, budget), budget);
}
size_t measure(const VtValue& value, const wire::EncodeLimits& limits, Budget& budget) {
    return measure(wire::DecodedValues{{value}, {}}, limits, budget);
}
using PathMeasurements = std::unordered_map<SdfPath, Measurement, SdfPath::Hash>;
// The charge of retaining `path`, measured once per compilation: a path the
// log names again is charged what measure() charges for it, under the same
// limits, without encoding it again.
size_t pathCost(const SdfPath& path, const FieldEditLimits& limits, Budget& budget, PathMeasurements& measured) {
    if (path.IsEmpty()) fail(wire::ErrorKind::Malformed, "empty field intent path");
    auto encoding = limits.scene.retainedValues;
    encoding.maxDepth = limits.scene.maxPathDepth;
    encoding.maxStringBytes = std::min(encoding.maxStringBytes, limits.scene.maxPathBytes);
    // The path passed every limit that does not depend on the remaining
    // budget when it was first measured; these are the ones that do.
    const auto room = affordable(encoding, budget);
    const auto known = measured.find(path);
    if (known != measured.end() && known->second.bytes <= room.maxEncodedBytes &&
        known->second.elements <= room.maxElements)
        return charge(known->second, budget);
    const auto cost = measurement(wire::DecodedValues{{VtValue(path)}, {}}, encoding, budget);
    measured[path] = cost;
    return charge(cost, budget);
}
// Rethrow an Unsupported value-encoding failure of operation `index` with its
// reason; every other error propagates unchanged.
[[noreturn]] void unsupportedValue(const wire::DecodeError& error, size_t index) {
    if (error.kind != wire::ErrorKind::Unsupported) throw;
    throw UnsupportedOperation(std::strcmp(error.what(), wire::kPayloadOffsetUnsupported) == 0 ?
            UnsupportedOperation::Reason::PayloadOffset :
            UnsupportedOperation::Reason::ValueType,
        index);
}
bool sampleKind(Kind kind) { return kind == Kind::SetSample || kind == Kind::EraseSample; }
bool dictionaryKind(Kind kind) { return kind == Kind::SetDictKey || kind == Kind::EraseDictKey; }
bool settingKind(Kind kind) { return kind == Kind::SetField || kind == Kind::SetDictKey || kind == Kind::SetSample; }
wire::EncodeLimits dictionaryLimits(const FieldEditLimits& limits) {
    auto result = limits.values;
    result.maxDepth = std::min(
        {result.maxDepth, limits.scene.retainedValues.maxDepth, limits.scene.values.maxDepth, size_t(256)});
    result.maxElements = std::min(
        {result.maxElements, limits.scene.retainedValues.maxElements, limits.scene.values.maxElements});
    result.maxEncodedBytes = std::min(
        {result.maxEncodedBytes, limits.scene.retainedValues.maxEncodedBytes, limits.scene.values.maxEncodedBytes});
    result.maxStringBytes = std::min(
        {result.maxStringBytes, limits.scene.retainedValues.maxStringBytes, limits.scene.values.maxStringBytes});
    return result;
}
void preflightDictionaryKey(const Operation& op, const FieldEditLimits& limits, Budget& budget) {
    auto encoding = dictionaryLimits(limits);
    if (!encoding.maxDepth) fail(wire::ErrorKind::ResourceLimit, "dictionary intent depth limit exceeded");
    // Pinned USD 25.11's VtDictionary string Set/EraseValueAtPath call
    // TfStringSplit(":") (verified in the SDK binary), despite dictionary.h's
    // TfStringTokenize wording. Leading/trailing/repeated separators preserve
    // empty keys, each of which consumes another dictionary/VtValue level.
    // Count borrowed bytes without constructing a vector or dictionaries.
    const auto& text = op.key.GetString();
    // Empty input supplies no path elements; USD still applies its normal
    // dictionary field-presence rules (for example, authoring an empty map).
    size_t components = text.empty() ? 0 : 1, componentBytes = 0;
    const auto maxComponents = (encoding.maxDepth - 1) / 2;
    if (components > maxComponents)
        fail(wire::ErrorKind::ResourceLimit, "dictionary intent key nesting limit exceeded");
    for (char c : text) {
        if (c == ':') {
            if (components == maxComponents)
                fail(wire::ErrorKind::ResourceLimit, "dictionary intent key nesting limit exceeded");
            ++components;
        } else ++componentBytes; // The caller already charged the complete string.
    }
    // Each key component adds a dictionary and VtValue wrapper to the value's
    // codec depth; its key string is another element. Charge expansion before
    // invoking USD even if a later operation erases the resulting dictionary.
    encoding.maxDepth -= 2 * components;
    product(components, 3, encoding.maxElements);
    product(components, 14, encoding.maxEncodedBytes); // Dictionary tag/count + key length.
    spend(componentBytes, encoding.maxEncodedBytes);
    spend(componentBytes, encoding.maxStringBytes);
    product(components, 3 * elementCost + 14, budget.work.remaining);
    spend(componentBytes, budget.work.remaining);
    // Combining the prefix and inserted value is essential: independently
    // valid key and value depths can exceed the limit when composed by USD.
    (void)measure(op.value, encoding, budget);
}
// Host storage ignores value writes to absent specs and moves from an absent
// source or onto another existing spec, while a host-captured log still records
// them. Replay skips such records; the final snapshot check still covers them.
// `held` caches whether the replay store holds a spec at a path since the last
// structural record.
bool ignoredByHostStorage(const SdfAbstractData& replay, const Operation& op,
    std::unordered_map<SdfPath, bool, SdfPath::Hash>& held) {
    switch (op.kind) {
    case Kind::SetField:
    case Kind::SetDictKey:
    case Kind::SetSample: {
        if (op.value.IsEmpty()) return false;
        const auto [entry, added] = held.emplace(op.path, false);
        if (added) entry->second = replay.HasSpec(op.path);
        return !entry->second;
    }
    case Kind::MoveSpec:
        return !replay.HasSpec(op.path) || (op.path != op.destination && replay.HasSpec(op.destination));
    default: return false;
    }
}
// A section's field, the key of its collapsed record.
struct FieldKey {
    uint64_t section;
    TfToken field;
    bool operator==(const FieldKey& other) const { return section == other.section && field == other.field; }
};
struct FieldKeyHash {
    size_t operator()(const FieldKey& key) const {
        return std::hash<uint64_t>{}(key.section) ^ (TfToken::HashFunctor{}(key.field) * 0x9e3779b97f4a7c15ull);
    }
};
// Apply one record to the compiler's replay store. The earlier checks reject
// malformed records and skip the value writes and moves that host storage
// ignores; erase and empty-value records on absent specs replay as no-ops.
// Storage may still classify a value as std::invalid_argument.
void replay(SdfAbstractData& data, const Operation& op) {
    switch (op.kind) {
    case Kind::CreateSpec: data.CreateSpec(op.path, op.specType); break;
    case Kind::EraseSpec: data.EraseSpec(op.path); break;
    case Kind::MoveSpec: data.MoveSpec(op.path, op.destination); break;
    case Kind::SetField:
        if (op.value.IsEmpty()) data.Erase(op.path, op.field);
        else data.Set(op.path, op.field, op.value);
        break;
    case Kind::EraseField: data.Erase(op.path, op.field); break;
    case Kind::SetDictKey: data.SetDictValueByKey(op.path, op.field, op.key, op.value); break;
    case Kind::EraseDictKey: data.EraseDictValueByKey(op.path, op.field, op.key); break;
    case Kind::SetSample: data.SetTimeSample(op.path, op.time, op.value); break;
    case Kind::EraseSample: data.EraseTimeSample(op.path, op.time); break;
    default: fail(wire::ErrorKind::Unsupported, "unsupported intent operation");
    }
}
bool structuralKind(Kind kind) { return kind == Kind::CreateSpec || kind == Kind::EraseSpec || kind == Kind::MoveSpec; }
// For each operation, whether replaying it into the replay store is
// unobservable: a field write or erasure, other than of timeSamples, whose
// field the next operation on that path and field writes or erases whole
// again before any structural operation. Nothing reads the field between
// the two, so the store ends the same without it.
std::vector<bool> supersededWrites(const std::vector<Operation>& operations) {
    struct Key {
        SdfPath path;
        TfToken field;
        bool operator==(const Key& other) const { return path == other.path && field == other.field; }
    };
    struct KeyHash {
        size_t operator()(const Key& key) const {
            return SdfPath::Hash{}(key.path) ^ (TfToken::HashFunctor{}(key.field) * 0x9e3779b97f4a7c15ull);
        }
    };
    std::vector<bool> result(operations.size(), false);
    // The fields whose next operation writes or erases them whole.
    std::unordered_set<Key, KeyHash> rewritten;
    for (size_t i = operations.size(); i-- > 0;) {
        const auto& op = operations[i];
        if (structuralKind(op.kind)) {
            rewritten.clear();
            continue;
        }
        const bool whole = (op.kind == Kind::SetField || op.kind == Kind::EraseField) &&
            op.field != SdfDataTokens->TimeSamples;
        Key key{op.path, sampleKind(op.kind) ? SdfDataTokens->TimeSamples : op.field};
        if (!whole) {
            rewritten.erase(key);
            continue;
        }
        result[i] = !rewritten.insert(std::move(key)).second;
    }
    return result;
}
// `index` is the operation's log position.
void supported(const Operation& op, size_t index) {
    if (op.kind == Kind::CopyFrom)
        fail(wire::ErrorKind::Unsupported, "whole-scene CopyFrom replacement is not implemented");
    if (structuralKind(op.kind)) {
        if (op.replacement || !op.field.IsEmpty() || !op.key.IsEmpty() || !op.value.IsEmpty() || op.time != 0 ||
            (op.kind == Kind::MoveSpec ? op.destination.IsEmpty() : !op.destination.IsEmpty()) ||
            (op.kind == Kind::CreateSpec ? (op.specType <= SdfSpecTypeUnknown || op.specType >= SdfNumSpecTypes) :
                                           op.specType != SdfSpecTypeUnknown))
            fail(wire::ErrorKind::Malformed, "inconsistent structural intent operation record");
        return;
    }
    switch (op.kind) {
    case Kind::SetField:
    case Kind::EraseField:
    case Kind::SetDictKey:
    case Kind::EraseDictKey:
    case Kind::SetSample:
    case Kind::EraseSample: break;
    default: fail(wire::ErrorKind::Malformed, "invalid intent operation kind");
    }
    if (!op.destination.IsEmpty() || op.specType != SdfSpecTypeUnknown || op.replacement ||
        (!dictionaryKind(op.kind) && !op.key.IsEmpty()) ||
        (sampleKind(op.kind) ? !op.field.IsEmpty() : op.field.IsEmpty()) || (!sampleKind(op.kind) && op.time != 0) ||
        (!settingKind(op.kind) && !op.value.IsEmpty()))
        fail(wire::ErrorKind::Malformed, "inconsistent field intent operation record");
    if (dictionaryKind(op.kind) && op.field == SdfDataTokens->TimeSamples)
        throw UnsupportedOperation(UnsupportedOperation::Reason::TimeSampleDictionaryKey, index);
    if (sampleKind(op.kind) && std::isnan(op.time)) fail(wire::ErrorKind::Malformed, "NaN sample intent time");
}
wire::EnvelopeEncodeLimits framingLimits(const FieldEditLimits& limits, const Budget& budget) {
    auto result = limits.envelope;
    auto& c = result.content;
    const auto& s = limits.scene.envelope;
    c.maxBytes = std::min({c.maxBytes, s.maxBytes, budget.work.remaining, budget.retained});
    c.maxParts = std::min(c.maxParts, s.maxParts);
    c.maxRecords = std::min(c.maxRecords, s.maxRecords);
    c.maxStringBytes = std::min({c.maxStringBytes, s.maxStringBytes, budget.work.remaining});
    c.maxReferencedValueBytes = std::min({c.maxReferencedValueBytes, s.maxReferencedValueBytes, budget.work.remaining});
    result.maxBuilderBytes = std::min(result.maxBuilderBytes, budget.work.remaining);
    result.maxScratchBytes = std::min(result.maxScratchBytes, budget.work.remaining);
    return result;
}
// A spec's type and fields, read with one lookup from the storages that keep
// each spec as one record, and field by field from any other storage. The
// scene work meter counts every field a read copies.
struct SpecFields {
    SdfSpecType type = SdfSpecTypeUnknown;
    std::vector<std::pair<TfToken, VtValue>> fields;
};
bool readFields(const SdfAbstractData& data, const SdfPath& path, SpecFields& spec) {
    if (const auto* scene = dynamic_cast<const SdfSceneData*>(&data))
        return scene->ReadSpec(path, &spec.type, &spec.fields);
    if (const auto* host = dynamic_cast<const HostSceneStorage*>(&data))
        return host->ReadSpec(path, &spec.type, &spec.fields);
    if (!data.HasSpec(path)) return false;
    spec.type = data.GetSpecType(path);
    spec.fields.clear();
    for (const auto& name : data.List(path)) spec.fields.emplace_back(name, data.Get(path, name));
    return true;
}
bool readSpec(const SdfAbstractData& data, const SdfPath& path, SpecFields& spec) {
    if (!readFields(data, path, spec)) return false;
    sceneWork::count(spec.fields.size());
    return true;
}
// A candidate's spec at a path, read once and indexed by field name once, so
// that verifying it against each storage looks every stored field up in
// constant time. `present` tells whether the candidate holds the spec.
struct CandidateSpec {
    SpecFields spec;
    bool present = false;
    std::unordered_map<TfToken, const VtValue*, TfToken::HashFunctor> fields;
};
void readCandidate(const SdfAbstractData& data, const SdfPath& path, CandidateSpec& candidate) {
    candidate.fields.clear();
    candidate.present = readSpec(data, path, candidate.spec);
    if (!candidate.present) return;
    sceneWork::count(candidate.spec.fields.size());
    candidate.fields.reserve(candidate.spec.fields.size());
    for (const auto& [name, value] : candidate.spec.fields) candidate.fields.emplace(name, &value);
}
// Compare the actual wire projection at `path`, avoiding recursive VtValue
// equality and its cache/asset-resolution semantics. Both projections use
// bounded all-type codec preflight; no difference in field presence or
// sample-map shape is lost. A path neither holds matches. The work is linear
// in the spec's fields: each stored field is one indexed candidate lookup.
void verifySpec(const CandidateSpec& candidate, const SdfAbstractData& data, const SdfPath& path,
    const FieldEditLimits& limits, Budget& budget, wire::ErrorKind mismatch) {
    spend(256, budget.work.remaining);
    SpecFields stored;
    if (!readSpec(data, path, stored)) {
        if (candidate.present) fail(mismatch, "field intent projection added specs");
        return;
    }
    if (!candidate.present || candidate.spec.type != stored.type) fail(mismatch, "field intent changed the spec graph");
    if (candidate.spec.fields.size() != stored.fields.size())
        fail(mismatch, "field intent projection changed field presence");
    sceneWork::count(stored.fields.size());
    for (const auto& [name, storedValue] : stored.fields) {
        spend(128, budget.work.remaining);
        spend(name.GetString().size(), budget.work.remaining);
        const auto found = candidate.fields.find(name);
        if (found == candidate.fields.end()) fail(mismatch, "field intent projection lost a field");
        const wire::DecodedValues pair{{storedValue, *found->second}, {}};
        // Each member obeys its own retained-value limits; the pair
        // itself shares the remaining transaction work budget.
        auto encoding = limits.scene.retainedValues;
        encoding.maxEncodedBytes = std::min(encoding.maxEncodedBytes, budget.work.remaining / 4);
        encoding.maxElements = std::min(encoding.maxElements, budget.work.remaining / (4 * elementCost));
        const auto cost = measure(pair, encoding, budget);
        spend(cost, budget.work.remaining); // Emission/comparison pass.
        const auto bytes = wire::encodeValues(pair, encoding);
        if (bytes.fields[0] != bytes.fields[1]) fail(mismatch, "field intent wire phases changed authored values");
    }
}
void verifySpec(const SdfAbstractData& candidate, const SdfAbstractData& data, const SdfPath& path,
    const FieldEditLimits& limits, Budget& budget, wire::ErrorKind mismatch) {
    CandidateSpec spec;
    readCandidate(candidate, path, spec);
    verifySpec(spec, data, path, limits, budget, mismatch);
}
// Every spec of `replay` against the candidate's projection.
void verifyProjection(const WireScenePtr& candidate, const SdfAbstractData& replay, const FieldEditLimits& limits,
    Budget& budget, wire::ErrorKind mismatch = wire::ErrorKind::Integrity) {
    struct Visitor final : SdfAbstractDataSpecVisitor {
        const SdfAbstractData& candidate;
        const FieldEditLimits& limits;
        Budget& budget;
        size_t count = 0;
        const wire::ErrorKind mismatch;
        Visitor(const SdfAbstractData& c, const FieldEditLimits& l, Budget& b, wire::ErrorKind m)
            : candidate(c), limits(l), budget(b), mismatch(m) {}
        bool VisitSpec(const SdfAbstractData& data, const SdfPath& path) override {
            ++count;
            if (!candidate.HasSpec(path)) fail(mismatch, "field intent changed the spec graph");
            verifySpec(candidate, data, path, limits, budget, mismatch);
            return true;
        }
        void Done(const SdfAbstractData&) override {}
    } visitor(*candidate->data(), limits, budget, mismatch);
    replay.VisitSpecs(&visitor);
    if (visitor.count != candidate->pathCount()) fail(mismatch, "field intent projection added specs");
}
using IdentityMap = std::map<SdfPath, NamespaceSpec>;
IdentityMap identities(const WireScenePtr& base, const FieldEditLimits& limits, Budget& budget,
    SectionIdAllocator& allocator) {
    struct Visitor final : SdfAbstractDataSpecVisitor {
        const WireScenePtr& base;
        const FieldEditLimits& limits;
        Budget& budget;
        SectionIdAllocator& allocator;
        IdentityMap result;
        Visitor(const WireScenePtr& b, const FieldEditLimits& l, Budget& w, SectionIdAllocator& a)
            : base(b), limits(l), budget(w), allocator(a) {}
        bool VisitSpec(const SdfAbstractData& data, const SdfPath& path) override {
            spend(sizeof(NamespaceSpec) + 128, budget.work.remaining);
            if (result.size() >= limits.scene.maxSections) fail(wire::ErrorKind::ResourceLimit, "intent section limit");
            const auto id = base->sectionForPath(path);
            allocator.observe(id);
            result.emplace(path, NamespaceSpec{id, data.GetSpecType(path), path});
            return true;
        }
        void Done(const SdfAbstractData&) override {}
    } visitor(base, limits, budget, allocator);
    base->data()->VisitSpecs(&visitor);
    return std::move(visitor.result);
}
int childList(SdfSpecType type, const TfToken& field) {
    for (unsigned list = 0; list < 3; ++list)
        if (!field.IsEmpty() && namespaceChildField(type, list) == field) return int(list);
    return -1;
}
// Whether `field` is a namespace child field of any spec type.
bool anyChildField(const TfToken& field) {
    // The distinct child fields of every spec type, gathered once.
    static const std::vector<TfToken> fields = [] {
        std::vector<TfToken> result;
        for (int type = 0; type < SdfNumSpecTypes; ++type)
            for (unsigned list = 0; list < 3; ++list) {
                const auto name = namespaceChildField(SdfSpecType(type), list);
                if (!name.IsEmpty() && std::find(result.begin(), result.end(), name) == result.end())
                    result.push_back(name);
            }
        return result;
    }();
    return !field.IsEmpty() && std::find(fields.begin(), fields.end(), field) != fields.end();
}
bool pathChildren(const TfToken& field) {
    return field == SdfChildrenKeys->ConnectionChildren || field == SdfChildrenKeys->RelationshipTargetChildren ||
        field == SdfChildrenKeys->MapperChildren;
}
void childValue(const TfToken& field, const VtValue& value) {
    if (!value.IsEmpty() &&
        (pathChildren(field) ? !value.IsHolding<std::vector<SdfPath>>() : !value.IsHolding<std::vector<TfToken>>()))
        fail(wire::ErrorKind::Malformed, "invalid authored child vector type");
}
// Complete membership is checked against the final graph. A service may store
// a Reorder verbatim as the parent's whole children list across every child
// list, so a changed order in any list emits one Reorder naming all of the
// parent's final children, each once; structural append alone emits none. A
// reader, and a service that orders lists per child list, orders each child
// list on its own, as USD does, so the one list orders each of them.
template<class Key>
std::vector<uint64_t> orderedChildren(const VtValue& value, const std::map<Key, uint64_t>& members, Budget& budget) {
    if (value.IsEmpty()) {
        if (!members.empty()) fail(wire::ErrorKind::Malformed, "missing authored child membership");
        return {};
    }
    if (!value.IsHolding<std::vector<Key>>())
        fail(wire::ErrorKind::Malformed, "invalid authored child membership type");
    const auto& keys = value.UncheckedGet<std::vector<Key>>();
    if (keys.size() != members.size()) fail(wire::ErrorKind::Malformed, "authored child membership differs from graph");
    product(keys.size(), sizeof(uint64_t) + 64, budget.work.remaining);
    std::vector<uint64_t> result;
    result.reserve(keys.size());
    std::set<uint64_t> seen;
    for (const auto& key : keys) {
        spend(1, budget.work.remaining);
        const auto found = members.find(key);
        if (found == members.end() || !seen.insert(found->second).second)
            fail(wire::ErrorKind::Malformed, "duplicate or nonmember authored child");
        result.push_back(found->second);
    }
    return result;
}
// A parent's final children in each of its child lists, by child key.
struct Members {
    std::map<TfToken, uint64_t> tokens;
    std::map<SdfPath, uint64_t> paths;
};
using ChildLists = std::array<Members, 3>;
void addMember(ChildLists& lists, SdfSpecType parent, const NamespaceSpec& child, Budget& budget) {
    spend(256, budget.work.remaining);
    const auto list = namespaceChildList(parent, child.type);
    const auto key = namespaceChildKey(child.type, child.path);
    auto& members = lists[list];
    const bool unique = key.IsHolding<TfToken>() ?
        members.tokens.emplace(key.UncheckedGet<TfToken>(), child.id).second :
        members.paths.emplace(key.UncheckedGet<SdfPath>(), child.id).second;
    if (!unique) fail(wire::ErrorKind::Malformed, "ambiguous final child membership");
}
// Appends the Reorder that gives `parent`, at `path`, the child order the
// replay store authors, when it differs from the planned candidate's.
void normalizeParent(const SdfPath& path, const NamespaceSpec& parent, const ChildLists& lists,
    const WireScenePtr& planned, const SdfAbstractData& replay, wire::EnvelopeInput& metadata,
    const FieldEditLimits& limits, Budget& budget) {
    wire::Command reorder{wire::Command::Kind::Reorder};
    reorder.section = parent.id;
    bool reordered = false;
    for (unsigned list = 0; list < 3; ++list) {
        spend(128, budget.work.remaining);
        const auto field = namespaceChildField(parent.type, list);
        if (field.IsEmpty()) continue;
        const auto desired = replay.Get(path, field), current = planned->data()->Get(path, field);
        childValue(field, desired);
        const auto& members = lists[list];
        const auto desiredIds = pathChildren(field) ? orderedChildren(desired, members.paths, budget) :
                                                      orderedChildren(desired, members.tokens, budget);
        const auto currentIds = pathChildren(field) ? orderedChildren(current, members.paths, budget) :
                                                      orderedChildren(current, members.tokens, budget);
        if (desiredIds.empty()) {
            if (!desired.IsEmpty() || !current.IsEmpty()) {
                throw UnsupportedOperation(UnsupportedOperation::Reason::MultiUpdate);
            }
            continue;
        }
        if (!reordered && desiredIds != currentIds) {
            reordered = true;
            reorder.childList = uint8_t(list);
        }
        reorder.children.insert(reorder.children.end(), desiredIds.begin(), desiredIds.end());
    }
    if (reordered) {
        // A shadowed duplicate sibling owns no spec, so the final graph omits
        // it; keep it listed at its stored position so the service never
        // drops it from the parent's children.
        const auto stored = planned->listedChildren(parent.id);
        product(stored.size() + reorder.children.size() + 1, sizeof(uint64_t), budget.work.remaining);
        const std::set<uint64_t> named(reorder.children.begin(), reorder.children.end());
        for (size_t i = 0; i < stored.size(); ++i)
            if (!named.count(stored[i]))
                reorder.children.insert(reorder.children.begin() + std::min(i, reorder.children.size()), stored[i]);
        if (metadata.commands.size() >= limits.maxStructuralCommands)
            fail(wire::ErrorKind::ResourceLimit, "intent structural command limit");
        metadata.commands.push_back(std::move(reorder));
    }
}
// Every parent of the complete final graph, in path order.
void normalizeChildren(const IdentityMap& graph, const WireScenePtr& planned, const SdfAbstractData& replay,
    wire::EnvelopeInput& metadata, const FieldEditLimits& limits, Budget& budget) {
    product(planned->sectionCount() + 1, sizeof(uint64_t), budget.work.remaining);
    std::map<uint64_t, ChildLists> groups;
    for (const auto& [path, child] : graph) {
        (void)path;
        if (child.id == 1) {
            spend(256, budget.work.remaining);
            continue;
        }
        const auto parent = graph.find(namespaceParentPath(child.type, child.path));
        if (parent == graph.end()) fail(wire::ErrorKind::Malformed, "dangling final intent parent");
        addMember(groups[parent->second.id], parent->second.type, child, budget);
    }
    for (const auto& [path, parent] : graph)
        normalizeParent(path, parent, groups[parent.id], planned, replay, metadata, limits, budget);
}
} // namespace

SectionIdAllocator::SectionIdAllocator(size_t maxIds, size_t maxAttempts, std::function<uint64_t()> random)
    : maxIds_(maxIds), maxAttempts_(maxAttempts), random_(std::move(random)) {
    if (!random_)
        random_ = [] {
            std::random_device source;
            if (!source.entropy()) fail(wire::ErrorKind::ResourceLimit, "section ID entropy unavailable");
            return std::uniform_int_distribution<uint64_t>{}(source);
        };
}
void SectionIdAllocator::observe(uint64_t id) {
    if (id <= 1) return;
    std::lock_guard<std::mutex> lock(mutex_);
    if (observed_.count(id)) return;
    if (observed_.size() >= maxIds_) fail(wire::ErrorKind::ResourceLimit, "section ID history exhausted");
    observed_.insert(id);
}
uint64_t SectionIdAllocator::allocate(WireSceneWorkBudget* work, const std::function<bool(uint64_t)>& taken) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (observed_.size() >= maxIds_) fail(wire::ErrorKind::ResourceLimit, "section ID history exhausted");
    for (size_t attempt = 0; attempt < maxAttempts_; ++attempt) {
        if (work) spend(128, work->remaining);
        uint64_t id;
        try {
            id = random_();
        } catch (const wire::DecodeError&) { throw; } catch (const std::bad_alloc&) {
            throw;
        } catch (...) { fail(wire::ErrorKind::ResourceLimit, "section ID entropy failure"); }
        if (id <= 1 || observed_.count(id)) continue;
        if (observed_.size() >= maxIds_) fail(wire::ErrorKind::ResourceLimit, "section ID history exhausted");
        observed_.insert(id);
        if (!taken || !taken(id)) return id;
    }
    fail(wire::ErrorKind::ResourceLimit, "section ID collision attempts exhausted");
}
size_t SectionIdAllocator::observedCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return observed_.size();
}

FieldEditIntentPtr FieldEditCompiler::compile(const WireScenePtr& base,
    const std::shared_ptr<const SdfOperationLog>& original, const FieldEditLimits& limits,
    const std::shared_ptr<SectionIdAllocator>& suppliedIds, WireScenePtr* applied, std::vector<uint64_t>* issued) {
    if (!base || !original || !original->operations)
        fail(wire::ErrorKind::Malformed, "field intent requires retained base and sealed owners");
    // The log limit bounds the raw log before any per-operation work. The
    // operation limit counts the operations whose writes the update emits,
    // checked once the records are built: a write that collapses into an
    // earlier record of its field or sample does not count. The work and
    // retention budgets charge every operation besides.
    const auto& operations = *original->operations;
    if (operations.size() > limits.maxLogOperations)
        fail(wire::ErrorKind::ResourceLimit, "field intent log operation limit exceeded");
    Budget budget(limits);
    spend(sizeof(FieldEditIntent), budget.retained);
    product(operations.size(), sizeof(Operation), budget.retained);
    product(operations.size(), sizeof(Operation), budget.work.remaining);
    wire::DecodedValues inputValues;
    size_t fields = 0, samples = 0;
    bool structural = false, childFields = false;
    PathMeasurements pathMeasures;
    for (const auto& op : operations) {
        supported(op, inputValues.fields.size()); // This operation's log position.
        structural = structural || structuralKind(op.kind);
        childFields = childFields || anyChildField(op.field);
        spend(pathCost(op.path, limits, budget, pathMeasures), budget.retained);
        if (!op.destination.IsEmpty()) spend(pathCost(op.destination, limits, budget, pathMeasures), budget.retained);
        for (const auto* text : {&op.field.GetString(), &op.key.GetString()}) {
            spend(text->size(), budget.retained);
            spend(text->size(), budget.work.remaining);
        }
        if (dictionaryKind(op.kind)) {
            try {
                preflightDictionaryKey(op, limits, budget);
            } catch (const wire::DecodeError& error) {
                unsupportedValue(error, inputValues.fields.size()); // This operation's log position.
            }
        }
        inputValues.fields.push_back(op.value);
        if (sampleKind(op.kind)) ++samples;
        else if (!structuralKind(op.kind)) ++fields;
    }
    // The record limit counts emitted field and sample records.
    const auto maxRecords = std::min(limits.scene.envelope.maxRecords, limits.envelope.content.maxRecords);
    try {
        spend(measure(inputValues, limits.values, budget), budget.retained);
    } catch (const wire::DecodeError& error) {
        if (error.kind != wire::ErrorKind::Unsupported) throw;
        // The batch measure loses which value failed. On this failure path
        // only, measure each value alone to name the first unencodable one.
        for (size_t i = 0; i < inputValues.fields.size(); ++i) {
            try {
                (void)wire::measureValues(wire::DecodedValues{{inputValues.fields[i]}, {}}, limits.values);
            } catch (const wire::DecodeError& single) {
                if (single.kind == wire::ErrorKind::Unsupported) unsupportedValue(single, i);
            }
        }
        throw;
    }
    // A log with no structural operation and no child-field record leaves the
    // graph and every child list as the base has them, so the namespace plan
    // is empty and no child order needs normalizing.
    const bool planned = structural || childFields || !limits.footprint;
    // At the footprint, records change only the specs they address and the
    // plan changes only the sections its commands name, so compiling costs
    // work in proportion to the log: the replay store shares the base's
    // storage, identities resolve through the base's path index, the plan and
    // child normalization follow the paths the log creates, erases and moves,
    // and the projections are compared at the addressed specs and the specs
    // the candidate changes. The whole-scene compilation enumerates the
    // base's identities, plans and normalizes the complete graph and verifies
    // every spec: work in proportion to the scene, retaining neither.
    const bool whole = !limits.footprint;
    if (whole) {
        const auto baseCost = base->retainedCharge();
        spend(baseCost, budget.work.remaining);
        spend(baseCost, budget.work.remaining);
    }
    // The whole-scene compilation observes the base's projected identities in
    // the allocator; at the footprint a temporary allocator leaves room for
    // them instead, so both exhaust its history at the same allocation.
    const auto allocator = suppliedIds ?
        suppliedIds :
        std::make_shared<SectionIdAllocator>(
            whole ? limits.maxSectionIds : limits.maxSectionIds - std::min(limits.maxSectionIds, base->pathCount() - 1),
            limits.maxIdAttempts);
    // An identity the base holds is never allocated, whether or not the
    // allocator has observed it.
    const auto allocate = [&](WireSceneWorkBudget* work) {
        const auto id = allocator->allocate(work, [&](uint64_t id) { return base->containsSection(id); });
        if (issued) issued->push_back(id);
        return id;
    };
    // Whole, the graph holds every spec. At the footprint it holds the base
    // identities the log addresses, resolved through the base's path index
    // when a record needs them, and the identities the log creates or moves;
    // `vacated` holds the base paths the log erases or moves a spec from, and
    // `erasedIds` the base identities it erases. Every other base path keeps
    // its base identity.
    auto graph = whole ? identities(base, limits, budget, *allocator) : IdentityMap{};
    std::set<SdfPath> vacated;
    std::set<uint64_t> erasedIds;
    const auto inBase = [&](uint64_t id) { return !base->pathForSection(id).IsEmpty(); };
    // The graph entry of `path`, resolved from the base when the log has not
    // vacated it; the graph's end when it holds none.
    const auto resolve = [&](const SdfPath& path) {
        auto found = graph.find(path);
        if (whole || found != graph.end() || vacated.count(path)) return found;
        if (const auto id = base->sectionForPath(path)) {
            spend(sizeof(NamespaceSpec) + 128, budget.work.remaining);
            found = graph.emplace(path, NamespaceSpec{id, base->data()->GetSpecType(path), path}).first;
        }
        return found;
    };
    const auto replayData = TfCreateRefPtr(new SdfSceneData);
    replayData->CopyFrom(base->data());
    auto& replayScene = *replayData;
    struct FieldRecord {
        uint64_t section;
        size_t index;
        TfToken field;
        VtValue value;
        bool sample;
        double time;
    };
    std::vector<FieldRecord> records;
    records.reserve(std::min(fields + samples, size_t(maxRecords)));
    // The record of each section's field. Repeated writes to one field of one
    // identity collapse into it, carrying the last value and write position,
    // including across a move of that identity: the wire applies every
    // structural command before field records keyed by section, and a field's
    // last record wins, so the update is unchanged. A recreated path is a new
    // section with its own records, and the records of an erased identity are
    // dropped with it, so each record keeps the identity its writes addressed.
    std::unordered_map<FieldKey, size_t, FieldKeyHash> fieldPositions;
    // Writes whose field or sample records all collapsed into earlier records.
    size_t collapsed = 0;
    // The latest record for each equivalent sample time. A set keeps an existing key's bits, so a set after an
    // erase of an equivalent time is its own record after the erase: it authors the key's bits.
    std::map<std::pair<uint64_t, double>, size_t> samplePositions;
    const auto sampleRecord = [&](uint64_t section, size_t index, double time, const VtValue& value) {
        if (!std::isfinite(time)) fail(wire::ErrorKind::Unsupported, "nonfinite sample time cannot persist");
        spend(sizeof(FieldRecord) + 64, budget.work.remaining);
        const auto found = samplePositions.find({section, time});
        if (found != samplePositions.end() && (value.IsEmpty() || !records[found->second].value.IsEmpty())) {
            auto& record = records[found->second];
            if (value.IsEmpty()) record.time = time;
            record.value = value;
            record.index = index;
            return;
        }
        if (records.size() >= maxRecords) fail(wire::ErrorKind::ResourceLimit, "expanded sample record limit");
        samplePositions[{section, time}] = records.size();
        records.push_back({section, index, {}, value, true, time});
    };
    std::set<uint64_t> movedIds;
    // Consecutive sample records apply to a working copy of each path's map,
    // made and charged on the run's first record for the path, exactly as the
    // replay store's SetTimeSample and EraseTimeSample apply them one by one.
    // Each changed map is written back before the next other record and after
    // the last record: a map that ends empty erases the field, and a map whose
    // records erased only absent keys keeps its field as it is.
    struct SampleRun {
        SdfTimeSampleMap samples;
        bool changed = false;
    };
    std::map<SdfPath, SampleRun> sampleRuns;
    const auto writeSampleRuns = [&] {
        for (auto& [path, run] : sampleRuns) {
            if (!run.changed) continue;
            if (run.samples.empty()) replayScene.Erase(path, SdfDataTokens->TimeSamples);
            else replayScene.Set(path, SdfDataTokens->TimeSamples, VtValue::Take(run.samples));
        }
        sampleRuns.clear();
    };
    const auto superseded = supersededWrites(operations);
    // Whether the replay store holds a spec at a path, which only structural
    // records change.
    std::unordered_map<SdfPath, bool, SdfPath::Hash> held;
    for (size_t i = 0; i < operations.size(); ++i) {
        const auto& op = operations[i];
        spend(sizeof(Operation) + sizeof(FieldRecord), budget.work.remaining);
        if (!sampleKind(op.kind)) writeSampleRuns();
        if (structuralKind(op.kind)) held.clear();
        if (ignoredByHostStorage(replayScene, op, held)) continue; // The final snapshot check still covers it.
        auto found = resolve(op.path);
        uint64_t section = found == graph.end() ? 0 : found->second.id;
        if (op.kind == Kind::CreateSpec) {
            if (found != graph.end() && found->second.type != op.specType)
                throw UnsupportedOperation(UnsupportedOperation::Reason::IdentityTypeChange, i);
            if (!section) {
                if (graph.size() >= limits.scene.maxSections)
                    fail(wire::ErrorKind::ResourceLimit, "intent section limit");
                section = allocate(&budget.work);
                graph.emplace(op.path, NamespaceSpec{section, op.specType, op.path});
            }
        }
        if (dictionaryKind(op.kind))
            (void)measure(replayScene.Get(op.path, op.field), dictionaryLimits(limits), budget);
        SampleRun* sampleRun = nullptr;
        if (sampleKind(op.kind)) {
            auto run = sampleRuns.find(op.path);
            if (run == sampleRuns.end()) {
                const auto existing = replayScene.Get(op.path, SdfDataTokens->TimeSamples);
                (void)measure(existing, limits.scene.retainedValues, budget);
                run = sampleRuns.emplace(op.path, SampleRun{}).first;
                if (existing.IsHolding<SdfTimeSampleMap>())
                    run->second.samples = existing.UncheckedGet<SdfTimeSampleMap>();
            }
            sampleRun = &run->second;
        }
        const bool wholeMap = !sampleKind(op.kind) && op.field == SdfDataTokens->TimeSamples;
        VtValue previousSamples;
        if (wholeMap) {
            previousSamples = replayScene.Get(op.path, SdfDataTokens->TimeSamples);
            (void)measure(previousSamples, limits.scene.retainedValues, budget);
        }
        try {
            if (!sampleRun) {
                if (!superseded[i]) replay(replayScene, op);
            } else if (op.kind == Kind::SetSample && !op.value.IsEmpty()) {
                sampleRun->samples[op.time] = op.value;
                sampleRun->changed = true;
            } else sampleRun->changed |= sampleRun->samples.erase(op.time) != 0;
            if (op.kind == Kind::EraseSpec && found != graph.end()) {
                if (!whole) {
                    vacated.insert(op.path);
                    if (inBase(section)) erasedIds.insert(section);
                }
                graph.erase(found);
            } else if (op.kind == Kind::MoveSpec) {
                if (found == graph.end()) fail(wire::ErrorKind::Malformed, "missing move identity");
                spend(64, budget.work.remaining);
                movedIds.insert(section);
                auto moved = found->second;
                moved.path = op.destination;
                graph.erase(found);
                if (!whole) vacated.insert(op.path);
                graph.emplace(op.destination, std::move(moved));
            }
        } catch (const std::invalid_argument&) { fail(wire::ErrorKind::Malformed, "invalid intent operation"); }
        if (structuralKind(op.kind) || !section) continue; // Idempotent erasure of missing specs has no wire target.
        if (sampleKind(op.kind)) {
            const auto before = records.size();
            sampleRecord(section, i, op.time, op.kind == Kind::SetSample ? op.value : VtValue{});
            if (records.size() == before) ++collapsed;
        } else {
            auto value = dictionaryKind(op.kind) ? replayScene.Get(op.path, op.field) :
                op.kind == Kind::SetField        ? op.value :
                                                   VtValue{};
            const auto type = found->second.type;
            if (childList(type, op.field) >= 0) {
                childValue(op.field, value);
            } else if (wholeMap) {
                const auto before = records.size();
                if (!value.IsEmpty() && !value.IsHolding<SdfTimeSampleMap>())
                    fail(wire::ErrorKind::Malformed, "timeSamples field must be a map");
                const SdfTimeSampleMap empty;
                const auto& desired = value.IsEmpty() ? empty : value.UncheckedGet<SdfTimeSampleMap>();
                // The new map replaces each key it lacks bit for bit, including an equivalent key of other bits.
                if (!previousSamples.IsEmpty())
                    for (const auto& [time, sample] : previousSamples.UncheckedGet<SdfTimeSampleMap>()) {
                        (void)sample;
                        spend(1, budget.work.remaining);
                        const auto kept = desired.find(time);
                        if (kept == desired.end() || std::signbit(kept->first) != std::signbit(time))
                            sampleRecord(section, i, time, {});
                    }
                for (const auto& [time, sample] : desired) sampleRecord(section, i, time, sample);
                if (records.size() == before) ++collapsed;
            } else {
                spend(64, budget.work.remaining);
                const auto [position, added] = fieldPositions.emplace(FieldKey{section, op.field}, records.size());
                if (!added) {
                    auto& record = records[position->second];
                    record.value = std::move(value);
                    record.index = i;
                    ++collapsed;
                    continue;
                }
                if (records.size() >= maxRecords) fail(wire::ErrorKind::ResourceLimit, "field intent record limit");
                records.push_back({section, i, op.field, std::move(value), false, 0});
            }
        }
    }
    writeSampleRuns();
    if (operations.size() - collapsed > limits.maxOperations)
        fail(wire::ErrorKind::ResourceLimit, "field intent operation limit exceeded");
    product(graph.size(), sizeof(NamespaceSpec) + 128, budget.work.remaining);
    std::vector<NamespaceSpec> desired;
    NamespaceChanges changes;
    // Identities of the final graph: whole, every one; at the footprint, the
    // ones the log creates, beside the base identities it does not erase.
    std::set<uint64_t> retainedIds;
    for (const auto& [path, spec] : graph) {
        const auto sampleMap = replayScene.Get(path, SdfDataTokens->TimeSamples);
        if (sampleMap.IsHolding<SdfTimeSampleMap>() && sampleMap.UncheckedGet<SdfTimeSampleMap>().empty())
            fail(wire::ErrorKind::Unsupported, "explicit empty timeSamples has no persistent representation");
        if (whole) {
            desired.push_back(spec);
            retainedIds.insert(spec.id);
        } else if (!inBase(spec.id)) {
            changes.placed.push_back(spec);
            retainedIds.insert(spec.id);
        } else if (movedIds.count(spec.id)) changes.placed.push_back(spec);
    }
    changes.removed.assign(erasedIds.begin(), erasedIds.end());
    const auto retained = [&](uint64_t id) {
        return retainedIds.count(id) || (!whole && !erasedIds.count(id) && inBase(id));
    };
    NamespacePlan plan;
    if (planned) {
        SceneNamespaceLimits planning;
        planning.scene = limits.scene;
        planning.maxCommands = limits.maxStructuralCommands;
        planning.maxMetadataBytes = limits.scene.maxMetadataBytes;
        // The planner owns a local meter until it returns. Reserve a disjoint eighth
        // of remaining work for scratch entropy attempts instead of modifying its
        // shared counter concurrently or resetting another full transaction budget.
        WireSceneWorkBudget scratch{structural ? budget.work.remaining / 8 : 0};
        spend(scratch.remaining, budget.work.remaining);
        planning.maxWork = budget.work.remaining;
        const auto allocateScratch = [&] { return allocate(&scratch); };
        plan = whole ? planSceneNamespace(base, desired, allocateScratch, planning, &budget.work) :
                       planSceneNamespaceChanges(base, changes, allocateScratch, planning, &budget.work);
        budget.work.remaining += scratch.remaining; // Refund unconsumed reserved work only.
    }

    wire::EnvelopeInput metadata;
    metadata.commands = std::move(plan.commands);
    wire::DecodedValues values;
    metadata.fields.reserve(std::min(fields, records.size()));
    values.fields.reserve(std::min(fields, records.size()));
    metadata.samples.reserve(std::min(samples, records.size()));
    values.samples.reserve(std::min(samples, records.size()));
    // Records addressing a removed identity are dropped; only its Delete
    // compiles. A peer that moves a descendant out of the deleted subtree before
    // the Delete executes keeps it without the dropped edit, which is superseded.
    for (const auto id : movedIds) {
        (void)id;
        spend(1, budget.work.remaining);
    }
    for (auto& record : records) {
        spend(sizeof(FieldRecord), budget.work.remaining);
        if (!retained(record.section)) continue;
        if (record.sample) {
            metadata.samples.push_back({record.section, record.time, {}});
            values.samples.push_back(std::move(record.value));
        } else {
            metadata.fields.push_back({record.section, uint64_t(record.index) + 1, record.field.GetString(), {}});
            values.fields.push_back(std::move(record.value));
        }
    }
    if (planned && whole) normalizeChildren(graph, plan.candidate, replayScene, metadata, limits, budget);
    else if (planned) {
        // Only a parent whose final children, authored child fields or
        // planned child lists differ from the base's can need a Reorder or
        // fail membership: one whose child field the log authors, one the
        // log creates, erases or moves a spec at, under or away from, or one
        // a planned command changes. Every other parent keeps the base's
        // children, whose child fields the scene derives from them.
        std::set<SdfPath> parents;
        const auto touch = [&](const SdfPath& path) {
            spend(64, budget.work.remaining);
            parents.insert(path);
            if (path.IsAbsoluteRootPath()) return;
            parents.insert(path.GetParentPath());
            if (path.IsPrimVariantSelectionPath() && !path.GetVariantSelection().second.empty())
                parents.insert(namespaceParentPath(SdfSpecTypeVariant, path));
        };
        for (const auto& op : operations) {
            if (structuralKind(op.kind)) {
                touch(op.path);
                if (!op.destination.IsEmpty()) touch(op.destination);
            } else if (anyChildField(op.field)) {
                spend(64, budget.work.remaining);
                parents.insert(op.path);
            }
        }
        for (const auto& command : metadata.commands)
            for (const auto id : {command.section, command.parent, command.oldParent}) {
                spend(64, budget.work.remaining);
                const auto path = id ? plan.candidate->pathForSection(id) : SdfPath();
                if (!path.IsEmpty()) parents.insert(path);
            }
        const auto finalAt = [&](const SdfPath& path) -> std::optional<NamespaceSpec> {
            if (const auto found = graph.find(path); found != graph.end()) return found->second;
            if (vacated.count(path)) return std::nullopt;
            const auto id = base->sectionForPath(path);
            if (!id) return std::nullopt;
            return NamespaceSpec{id, base->data()->GetSpecType(path), path};
        };
        std::map<SdfPath, std::vector<SdfPath>> placedUnder;
        for (const auto& [path, spec] : graph) {
            spend(64, budget.work.remaining);
            if (spec.id != 1) placedUnder[namespaceParentPath(spec.type, path)].push_back(path);
        }
        for (const auto& path : parents) {
            const auto parent = finalAt(path);
            if (!parent) continue;
            // The final children: the base's children at this path and the
            // specs the graph holds under it, each where it still is.
            std::set<SdfPath> children;
            if (const auto owner = base->sectionForPath(path))
                for (const auto id : base->listedChildren(owner)) {
                    spend(64, budget.work.remaining);
                    auto child = base->pathForSection(id);
                    if (!child.IsEmpty()) children.insert(std::move(child));
                }
            if (const auto found = placedUnder.find(path); found != placedUnder.end())
                children.insert(found->second.begin(), found->second.end());
            ChildLists lists;
            for (const auto& child : children)
                if (const auto spec = finalAt(child)) addMember(lists, parent->type, *spec, budget);
            normalizeParent(path, *parent, lists, plan.candidate, replayScene, metadata, limits, budget);
        }
    }
    // The update the service receives is exactly these records; one over its
    // entry limits is refused before any value is encoded.
    checkDeltaEntries(metadata.commands, metadata.fields.size(), metadata.samples.size(), limits.entries);
    const auto valueCost = measure(values, limits.values, budget);
    spend(valueCost, budget.work.remaining);
    auto framing = framingLimits(limits, budget);
    auto bytes = wire::encodeUsdEnvelope(metadata, values, framing, limits.values);
    spend(bytes.size(), budget.retained);
    spend(bytes.size(), budget.work.remaining);
    const auto candidate = buildSceneCandidate(base, bytes, limits.scene, &budget.work,
        WireSceneBatchPolicy::OrdinaryUpdate);

    // Host storage holds its baseline, which is `base`, plus the log's edits:
    // every edit it stores is recorded, and each flush rebaselines it on the
    // scene this compilation verifies. The replay store and the sealed copy of
    // host storage differ from the base only at the paths the log addresses
    // or moves a spec to, and the candidate only where diffProjectedScenes
    // reports, so at the footprint the three are compared at those paths,
    // after the copy's spec count. Other sealed data and an unsealed host
    // buffer prefix are compared with the replay at every spec.
    const auto* host = original->data ? dynamic_cast<const HostSceneStorage*>(original->data.operator->()) : nullptr;
    if (whole || !host) {
        verifyProjection(candidate, replayScene, limits, budget);
        if (original->data) verifyProjection(candidate, *original->data, limits, budget, wire::ErrorKind::Malformed);
    } else {
        if (host->specCount() != candidate->pathCount())
            fail(wire::ErrorKind::Malformed, "field intent projection added specs");
        // The paths in path order, each once, however often the log names it.
        std::unordered_set<SdfPath, SdfPath::Hash> named;
        for (const auto& op : operations) {
            named.insert(op.path);
            if (!op.destination.IsEmpty()) named.insert(op.destination);
        }
        std::set<SdfPath> addressed(named.begin(), named.end());
        if (planned) {
            const auto difference = diffProjectedScenes(*base, *candidate);
            addressed.insert(difference.erased.begin(), difference.erased.end());
            addressed.insert(difference.written.begin(), difference.written.end());
            for (const auto& field : difference.fields) addressed.insert(field.path);
        }
        CandidateSpec spec;
        for (const auto& path : addressed) {
            readCandidate(*candidate->data(), path, spec);
            verifySpec(spec, replayScene, path, limits, budget, wire::ErrorKind::Integrity);
            verifySpec(spec, *original->data, path, limits, budget, wire::ErrorKind::Malformed);
        }
    }
    auto result = std::shared_ptr<FieldEditIntent>(new FieldEditIntent);
    result->empty_ = metadata.commands.empty() && metadata.fields.empty() && metadata.samples.empty();
    for (const auto& command : metadata.commands) {
        spend(1, budget.work.remaining);
        // Fresh authored identities have an authoritative collision contract.
        // Scratch identities used only to break namespace cycles still need
        // separate concurrency qualification; they are absent from the final graph.
        if (command.kind == wire::Command::Kind::Create && !retained(command.section))
            result->unsupportedStructure_ = UnsupportedOperation::Reason::MoveCycle;
    }
    result->concurrentStructureSupported_ = result->unsupportedStructure_ == UnsupportedOperation::Reason{};
    result->operations_ = original->operations;
    result->planned_ = planned;
    result->bytes_ = std::move(bytes);
    result->retainedBytes_ = limits.maxRetainedBytes - budget.retained;
    result->workUsed_ = limits.maxWork - budget.work.remaining;
    if (applied) *applied = candidate;
    return result;
}
namespace {
std::atomic<bool> compileCheck{false};
std::atomic<uint64_t> compilations{0};
// The class of a compilation failure: its type and its kind or reason.
std::string failureClass(const std::exception_ptr& failure) {
    if (!failure) return "none";
    try {
        std::rethrow_exception(failure);
    } catch (const UnsupportedOperation& error) {
        return "unsupported " + std::to_string(uint32_t(error.reason()));
    } catch (const DeltaEntryLimitExceeded& error) {
        return "entries " + std::to_string(int(error.reason()));
    } catch (const wire::DecodeError& error) {
        return "decode " + std::to_string(int(error.kind));
    } catch (const std::exception& error) { return std::string("other ") + typeid(error).name(); } catch (...) {
        return "unknown";
    }
}
// A compilation failure's class and message. The two compilations verify
// different spec sets, so one divergence can fail different checks of the
// same class.
std::string failureDescription(const std::exception_ptr& failure) {
    auto result = failureClass(failure);
    try {
        if (failure) std::rethrow_exception(failure);
    } catch (const std::exception& error) { result += std::string(" (") + error.what() + ")"; } catch (...) {
    }
    return result;
}
} // namespace
void setFieldEditCompileCheck(bool enabled) noexcept { compileCheck.store(enabled, std::memory_order_relaxed); }
FieldEditIntentPtr compileFieldEdits(const WireScenePtr& base, const std::shared_ptr<const SdfOperationLog>& original,
    const FieldEditLimits& limits, const std::shared_ptr<SectionIdAllocator>& suppliedIds, WireScenePtr* applied) {
    compilations.fetch_add(1, std::memory_order_relaxed);
    if (!limits.footprint || !compileCheck.load(std::memory_order_relaxed))
        return FieldEditCompiler::compile(base, original, limits, suppliedIds, applied, nullptr);
    std::vector<uint64_t> issued;
    FieldEditIntentPtr result;
    std::exception_ptr failure;
    try {
        result = FieldEditCompiler::compile(base, original, limits, suppliedIds, applied, &issued);
    } catch (...) { failure = std::current_exception(); }
    // The whole-scene compilation draws the identities the footprint drew, in
    // order, and then deterministic ones.
    auto whole = limits;
    whole.footprint = false;
    size_t next = 0;
    std::mt19937_64 rest(0x666f6f747072696eull);
    const auto replayed = std::make_shared<SectionIdAllocator>(limits.maxSectionIds, limits.maxIdAttempts,
        [&] { return next < issued.size() ? issued[next++] : rest(); });
    FieldEditIntentPtr expected;
    std::exception_ptr expectedFailure;
    try {
        expected = FieldEditCompiler::compile(base, original, whole, replayed, nullptr, nullptr);
    } catch (...) { expectedFailure = std::current_exception(); }
    // Each charges its own work and draws identities on its own, so a
    // resource limit on either side is not compared.
    const auto limited = [](const std::exception_ptr& error) { return failureClass(error).rfind("decode 2", 0) == 0; };
    if (limited(failure) || limited(expectedFailure)) {
        if (failure) std::rethrow_exception(failure);
        return result;
    }
    if (failureClass(failure) != failureClass(expectedFailure))
        throw std::logic_error("footprint compilation failed as " + failureDescription(failure) +
            ", the whole-scene one as " + failureDescription(expectedFailure));
    if (failure) std::rethrow_exception(failure);
    if (result->bytes() != expected->bytes() || result->retainedBytes() != expected->retainedBytes() ||
        result->empty() != expected->empty() ||
        result->concurrentStructureSupported() != expected->concurrentStructureSupported() ||
        result->unsupportedStructure() != expected->unsupportedStructure())
        throw std::logic_error("footprint compilation differs from the whole-scene compilation");
    return result;
}
uint64_t fieldEditCompilationsForTest() noexcept { return compilations.load(std::memory_order_relaxed); }
bool matchesWireProjection(const WireScenePtr& scene, const SdfAbstractData& data, const FieldEditLimits& limits) {
    if (!scene) return false;
    Budget budget(limits);
    try {
        verifyProjection(scene, data, limits, budget);
        return true;
    } catch (const wire::DecodeError&) { return false; }
}
} // namespace omni::usd::live
