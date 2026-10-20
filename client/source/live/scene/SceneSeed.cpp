// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "SceneSeed.h"
#include "SceneNamespace.h"
#include "OmpfEncodeInternal.h"
#include <algorithm>
#include <map>
#include <limits>
#include <pxr/usd/sdf/schema.h>
#include <cmath>
#include <tuple>

using namespace PXR_NS;
namespace omni::usd::live {
namespace {
[[noreturn]] void fail(wire::ErrorKind kind, const char* message) { throw wire::DecodeError(kind, message); }
void require(bool valid, const char* message) {
    if (!valid) fail(wire::ErrorKind::Malformed, message);
}
void spend(size_t amount, size_t& remaining) {
    if (amount > remaining) fail(wire::ErrorKind::ResourceLimit, "creation seed budget exceeded");
    remaining -= amount;
}
struct Budget {
    WireSceneWorkBudget work;
    size_t metadata, paths, fields, records, strings;
    explicit Budget(const SceneSeedLimits& limits)
        : work{limits.maxWork}, metadata(limits.maxMetadataBytes), paths(limits.scene.maxPathBytes),
          fields(limits.scene.maxFields),
          records(std::min(limits.scene.envelope.maxRecords, limits.envelope.content.maxRecords)),
          strings(std::min({limits.scene.maxMetadataBytes, limits.scene.envelope.maxStringBytes,
              limits.envelope.content.maxStringBytes})) {}
    void record(size_t bytes) {
        spend(bytes, metadata);
        spend(bytes, work.remaining);
        spend(1, records);
    }
    void text(size_t bytes) {
        spend(bytes, metadata);
        spend(bytes, strings);
        spend(bytes, work.remaining);
    }
    void values(const wire::DecodedValues& source, wire::EncodeLimits limits, size_t passes) {
        limits.maxEncodedBytes = std::min(limits.maxEncodedBytes, work.remaining / passes);
        limits.maxElements = std::min(limits.maxElements, work.remaining / (passes * 128));
        size_t elements = 0;
        const auto bytes = wire::measureValues(source, limits, &elements);
        for (size_t i = 0; i < passes; ++i) {
            spend(bytes, work.remaining);
            if (elements > work.remaining / 128)
                fail(wire::ErrorKind::ResourceLimit, "creation seed value work limit exceeded");
            spend(elements * 128, work.remaining);
        }
    }
};
void pathTopology(const SdfPath& path, size_t& remaining, Budget& budget) {
    for (const auto& ancestor : path.GetAncestorsRange()) {
        spend(1, budget.work.remaining);
        if (!ancestor.IsAbsoluteRootPath()) spend(1, remaining);
        if (ancestor.IsTargetPath() || ancestor.IsMapperPath())
            pathTopology(ancestor.GetTargetPath(), remaining, budget);
    }
}
void pathShape(SdfSpecType type, const SdfPath& path) {
    bool valid = false;
    switch (type) {
    case SdfSpecTypePseudoRoot: valid = path.IsAbsoluteRootPath(); break;
    case SdfSpecTypePrim: valid = path.IsPrimPath() && !path.IsAbsoluteRootPath(); break;
    case SdfSpecTypeAttribute: valid = path.IsPrimPropertyPath() || path.IsRelationalAttributePath(); break;
    case SdfSpecTypeRelationship: valid = path.IsPrimPropertyPath(); break;
    case SdfSpecTypeConnection:
    case SdfSpecTypeRelationshipTarget: valid = path.IsTargetPath(); break;
    case SdfSpecTypeExpression: valid = path.IsExpressionPath(); break;
    case SdfSpecTypeMapper: valid = path.IsMapperPath(); break;
    case SdfSpecTypeMapperArg: valid = path.IsMapperArgPath(); break;
    case SdfSpecTypeVariantSet:
        valid = path.IsPrimVariantSelectionPath() && path.GetVariantSelection().second.empty();
        break;
    case SdfSpecTypeVariant:
        valid = path.IsPrimVariantSelectionPath() && !path.GetVariantSelection().second.empty();
        break;
    default: break;
    }
    require(valid, "invalid creation seed spec type/path");
}
bool absoluteTargets(const SdfPath& path, Budget& budget) {
    if (!path.IsAbsolutePath()) return false;
    for (const auto& ancestor : path.GetAncestorsRange()) {
        spend(1, budget.work.remaining);
        if ((ancestor.IsTargetPath() || ancestor.IsMapperPath()) && !absoluteTargets(ancestor.GetTargetPath(), budget))
            return false;
    }
    return true;
}
struct FieldLess {
    bool operator()(const TfToken& a, const TfToken& b) const { return a.GetString() < b.GetString(); }
};
struct Node {
    uint64_t id = 0;
    SdfSpecType type;
    std::map<TfToken, VtValue, FieldLess> fields;
};
struct Plan final : SdfAbstractDataSpecVisitor {
    const SceneSeedLimits& limits;
    Budget& budget;
    std::map<SdfPath, Node> nodes;
    Plan(const SceneSeedLimits& limits, Budget& budget) : limits(limits), budget(budget) {}
    bool VisitSpec(const SdfAbstractData& source, const SdfPath& path) override {
        if (nodes.size() >= limits.scene.maxSections)
            fail(wire::ErrorKind::ResourceLimit, "creation seed spec count exceeded");
        budget.record(512);
        require(!path.IsEmpty() && path.IsAbsolutePath(), "creation seed requires absolute spec paths");
        size_t depth = limits.scene.maxPathDepth;
        pathTopology(path, depth, budget);
        auto encoding = limits.scene.retainedValues;
        encoding.maxDepth = limits.scene.maxPathDepth;
        encoding.maxStringBytes = std::min(encoding.maxStringBytes, budget.paths);
        // Bounded path codec traversal precedes SDK cached text and map comparisons.
        budget.values(wire::DecodedValues{{VtValue(path)}, {}}, encoding, 1);
        spend(path.GetString().size(), budget.paths);
        const auto type = source.GetSpecType(path);
        pathShape(type, path);
        if (type == SdfSpecTypeMapper && !absoluteTargets(path.GetTargetPath(), budget))
            fail(wire::ErrorKind::Unsupported, "creation seed mapper requires absolute target identities");
        auto inserted = nodes.emplace(path, Node{0, type, {}});
        require(inserted.second, "duplicate creation seed spec path");
        const auto names = source.List(path);
        if (names.size() > budget.fields) fail(wire::ErrorKind::ResourceLimit, "creation seed field count exceeded");
        for (const auto& name : names) {
            spend(1, budget.fields);
            budget.record(256);
            budget.text(name.GetString().size());
            require(!name.IsEmpty(), "empty creation seed field name");
            VtValue value;
            require(source.Has(path, name, &value), "listed creation seed field is absent");
            if (value.IsEmpty())
                fail(wire::ErrorKind::Unsupported, "present empty field has no lossless wire representation");
            require(inserted.first->second.fields.emplace(name, std::move(value)).second,
                "duplicate creation seed field");
        }
        return true;
    }
    void Done(const SdfAbstractData&) override {}
};
wire::EnvelopeEncodeLimits framingLimits(const SceneSeedLimits& limits) {
    auto result = limits.envelope;
    auto& out = result.content;
    const auto& in = limits.scene.envelope;
    out.maxBytes = std::min(out.maxBytes, in.maxBytes);
    out.maxParts = std::min(out.maxParts, in.maxParts);
    out.maxRecords = std::min(out.maxRecords, in.maxRecords);
    out.maxStringBytes = std::min(out.maxStringBytes, in.maxStringBytes);
    out.maxReferencedValueBytes = std::min(out.maxReferencedValueBytes, in.maxReferencedValueBytes);
    return result;
}
wire::EncodeLimits valueLimits(const SceneSeedLimits& limits, const wire::EnvelopeEncodeLimits& framing) {
    auto result = limits.values;
    const auto& retained = limits.scene.retainedValues;
    const auto& decoded = limits.scene.values;
    result.maxEncodedBytes = std::min({result.maxEncodedBytes, retained.maxEncodedBytes, decoded.maxEncodedBytes,
        framing.content.maxBytes, framing.content.maxReferencedValueBytes});
    result.maxStringBytes = std::min({result.maxStringBytes, retained.maxStringBytes, decoded.maxStringBytes});
    result.maxElements = std::min({result.maxElements, retained.maxElements, decoded.maxElements});
    result.maxDepth = std::min({result.maxDepth, retained.maxDepth, decoded.maxDepth});
    return result;
}
} // namespace
SceneSeed compileSceneSeed(const SdfAbstractData& source, std::string_view sourceFormat,
    const SceneSeedLimits& limits) {
    if (limits.scene.maxPathDepth > 256) throw std::invalid_argument("invalid creation seed path depth limit");
    Budget budget(limits);
    budget.text(sourceFormat.size());
    Plan plan(limits, budget);
    source.VisitSpecs(&plan);
    const auto root = plan.nodes.find(SdfPath::AbsoluteRootPath());
    require(root != plan.nodes.end() && root->second.type == SdfSpecTypePseudoRoot,
        "creation seed requires pseudoroot");
    root->second.id = 1;
    uint64_t next = 2;
    for (auto& [path, node] : plan.nodes)
        if (!path.IsAbsoluteRootPath()) node.id = next++;
    wire::EnvelopeInput input;
    input.isDiff = true;
    input.sourceFormat = std::string(sourceFormat);
    wire::DecodedValues values, authored;
    const auto encoding = valueLimits(limits, framingLimits(limits));
    for (const auto& [path, node] : plan.nodes) {
        (void)path;
        for (const auto& [name, value] : node.fields) {
            (void)name;
            spend(128, budget.work.remaining);
            authored.fields.push_back(value);
        }
    }
    // Validate complete source values before traversing authored child keys.
    budget.values(authored, encoding, 2);
    const auto original = wire::encodeValues(authored, encoding);
    using Child = std::tuple<uint64_t, unsigned, std::string>;
    std::map<Child, uint32_t> orders;
    for (const auto& [path, node] : plan.nodes) {
        (void)path;
        uint32_t order = 0;
        for (unsigned list = 0; list < 3; ++list) {
            const auto field = namespaceChildField(node.type, list);
            const auto found = node.fields.find(field);
            if (field.IsEmpty() || found == node.fields.end()) continue;
            const auto add = [&](const auto& keys) {
                if (keys.empty())
                    fail(wire::ErrorKind::Unsupported,
                        "explicit empty child membership has no persistent representation");
                for (const auto& key : keys) {
                    budget.text(key.GetString().size());
                    spend(128, budget.work.remaining);
                    if (order == UINT32_MAX)
                        fail(wire::ErrorKind::ResourceLimit, "creation seed sibling order exhausted");
                    if (!orders.emplace(Child{node.id, list, key.GetString()}, order++).second)
                        fail(wire::ErrorKind::Unsupported, "duplicate creation seed child membership");
                }
            };
            const auto& value = found->second;
            if (value.IsHolding<std::vector<TfToken>>()) add(value.UncheckedGet<std::vector<TfToken>>());
            else if (value.IsHolding<std::vector<SdfPath>>()) add(value.UncheckedGet<std::vector<SdfPath>>());
            else fail(wire::ErrorKind::Malformed, "invalid creation seed child membership type");
        }
    }
    for (const auto& [path, node] : plan.nodes) {
        spend(256, budget.work.remaining);
        wire::Command command{wire::Command::Kind::DiffChange};
        command.section = node.id;
        command.sectionType = uint8_t(node.type);
        if (node.id != 1) {
            const auto parent = plan.nodes.find(namespaceParentPath(node.type, path));
            require(parent != plan.nodes.end(), "creation seed spec has no parent");
            const auto list = namespaceChildList(parent->second.type, node.type);
            const auto field = namespaceChildField(parent->second.type, list);
            if (parent->second.fields.find(field) == parent->second.fields.end())
                fail(wire::ErrorKind::Unsupported, "creation seed child graph lacks authored membership");
            command.parent = parent->second.id;
            const auto key = namespaceChildKey(node.type, path);
            const auto& text = key.IsHolding<TfToken>() ? key.UncheckedGet<TfToken>().GetString() :
                                                          key.UncheckedGet<SdfPath>().GetString();
            const auto order = orders.find(Child{command.parent, list, text});
            if (order == orders.end())
                fail(wire::ErrorKind::Unsupported, "creation seed child graph differs from authored membership");
            command.sectionOrder = order->second;
            budget.text(text.size());
            command.name = text;
        }
        input.commands.push_back(std::move(command));
        for (const auto& [name, value] : node.fields) {
            spend(128, budget.work.remaining);
            bool child = false;
            for (unsigned list = 0; list < 3; ++list) child = child || namespaceChildField(node.type, list) == name;
            if (child) continue;
            if (name == SdfFieldKeys->TimeSamples) {
                require(value.IsHolding<SdfTimeSampleMap>(), "creation seed timeSamples field must be a map");
                const auto& samples = value.UncheckedGet<SdfTimeSampleMap>();
                if (samples.empty())
                    fail(wire::ErrorKind::Unsupported, "explicit empty timeSamples has no persistent representation");
                for (const auto& [time, sample] : samples) {
                    if (!std::isfinite(time) || sample.IsEmpty())
                        fail(wire::ErrorKind::Unsupported, "creation seed sample has no persistent representation");
                    budget.record(128);
                    input.samples.push_back({node.id, time, {}});
                    values.samples.push_back(sample);
                }
            } else {
                input.fields.push_back({node.id, uint64_t(input.fields.size()) + 1, name.GetString(), {}});
                values.fields.push_back(value);
            }
        }
    }
    // The CreateLayer the service receives holds exactly these records; a seed
    // over its entry limits is refused before any value is encoded.
    checkDeltaEntries(input.commands, input.fields.size(), input.samples.size(), limits.entries);
    auto framing = framingLimits(limits);
    framing.content.maxBytes = std::min(framing.content.maxBytes, budget.work.remaining / 2);
    // Validate metadata and intersect downstream limits before allocating any
    // encoded value buffers. One shared value quota covers the complete source.
    wire::detail::validateEnvelopeInput(input, framing);
    budget.values(values, encoding, 2);
    const auto encoded = wire::encodeValues(values, encoding);
    for (size_t i = 0; i < input.fields.size(); ++i) input.fields[i].value = encoded.fields[i];
    for (size_t i = 0; i < input.samples.size(); ++i) input.samples[i].value = encoded.samples[i];
    framing.content.maxBytes = std::min(framing.content.maxBytes, budget.work.remaining / 2);
    SceneSeed result;
    result.bytes = wire::encodeEnvelope(input, framing);
    spend(result.bytes.size(), budget.work.remaining);
    result.scene = buildSceneCandidate({}, result.bytes, limits.scene, &budget.work);
    if (result.scene->sectionCount() != plan.nodes.size() || result.scene->sourceFormat() != sourceFormat)
        fail(wire::ErrorKind::Unsupported, "creation seed projection changed source identity");
    wire::DecodedValues projected;
    for (const auto& [path, node] : plan.nodes) {
        spend(256, budget.work.remaining);
        const auto& data = *result.scene->data();
        if (result.scene->sectionForPath(path) != node.id || data.GetSpecType(path) != node.type ||
            data.List(path).size() != node.fields.size())
            fail(wire::ErrorKind::Unsupported, "creation seed projection changed spec or field presence");
        for (const auto& [name, value] : node.fields) {
            (void)value;
            spend(128, budget.work.remaining);
            VtValue candidate;
            if (!data.Has(path, name, &candidate))
                fail(wire::ErrorKind::Unsupported, "creation seed projection lost field");
            projected.fields.push_back(std::move(candidate));
        }
    }
    budget.values(projected, encoding, 2);
    const auto comparison = wire::encodeValues(projected, encoding);
    // Byte equality preserves supported typed values including NaN/signed-zero
    // bits; ordinary VtValue equality is not an exact authored-value oracle.
    for (size_t i = 0; i < original.fields.size(); ++i) {
        spend(original.fields[i].size(), budget.work.remaining);
        if (original.fields[i] != comparison.fields[i])
            fail(wire::ErrorKind::Unsupported, "creation seed projection changed authored value");
    }
    result.workUsed = limits.maxWork - budget.work.remaining;
    return result;
}
} // namespace omni::usd::live
