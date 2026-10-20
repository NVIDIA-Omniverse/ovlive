// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "Probe.h"
#include "LiveUri.h"
#include <pxr/pxr.h>
#include <pxr/usd/ar/defineResolver.h>
#include <pxr/usd/ar/resolvedPath.h>
#include <pxr/usd/ar/timestamp.h>
#include <pxr/usd/sdf/data.h>
#include <pxr/usd/sdf/fileFormat.h>
#include <pxr/usd/sdf/layer.h>
#include <pxr/usd/sdf/schema.h>
#include <mutex>
#include <iostream>
#include <functional>
#include <stdexcept>

static_assert(PXR_VERSION == 2511, "Resolver hook probe is qualified only for USD 25.11");
namespace live_resolver_probe {
namespace {
std::mutex mutex;
std::vector<Event> log;
bool endpointRegistered = true;
std::string creationCase;
PXR_NS::SdfAbstractDataConstRefPtr capturedData;
PXR_NS::SdfLayerHandle capturedTarget;
std::vector<std::function<void()>> deferred;
}
void creationMode(std::string value) { creationCase = std::move(value); }
void marker(const char* value) { std::cout << "{\"marker\":\"" << value << "\"}" << std::endl; }
std::string capturedComment() {
    if (!capturedData) return {};
    const auto value = capturedData->Get(PXR_NS::SdfPath::AbsoluteRootPath(), PXR_NS::SdfFieldKeys->Comment);
    return value.IsHolding<std::string>() ? value.UncheckedGet<std::string>() : std::string{};
}
bool capturedTargetExpired() { return !capturedTarget; }
void enqueueAnonymous() {
    deferred.emplace_back([] {
        marker("before-deferred-anonymous");
        const auto nested = PXR_NS::SdfLayer::CreateAnonymous("deferred.usda");
        if (!nested) throw std::runtime_error("deferred anonymous creation failed");
        marker("after-deferred-anonymous");
    });
    marker("queued-anonymous-work");
}
size_t drainDeferred() {
    auto jobs = std::move(deferred);
    for (const auto& job : jobs) job();
    return jobs.size();
}
void clear() { std::lock_guard<std::mutex> lock(mutex); log.clear(); }
std::vector<Event> events() { std::lock_guard<std::mutex> lock(mutex); return log; }
void registered(bool value) { std::lock_guard<std::mutex> lock(mutex); endpointRegistered = value; }
void record(std::string hook, const std::string& path, std::string detail = {}) {
    std::lock_guard<std::mutex> lock(mutex); log.push_back({std::move(hook), path, std::move(detail)});
}
std::string canonical(const std::string& path, bool requireRegistration) {
    try {
        const auto parsed = omni::usd::live::parseUri(path);
        if (parsed.layerName.empty()) return {};
        std::lock_guard<std::mutex> lock(mutex);
        if (requireRegistration && (!endpointRegistered || parsed.authority.canonical != "probe.example"))
            return {};
        return parsed.canonical;
    } catch (const std::invalid_argument&) { return {}; }
    catch (const std::length_error&) { return {}; }
}
}

PXR_NAMESPACE_OPEN_SCOPE
class LiveHookProbeResolver final : public ArResolver {
    std::string _CreateIdentifier(const std::string& path, const ArResolvedPath& anchor) const override {
        live_resolver_probe::record("CreateIdentifier", path, anchor.GetPathString());
        return live_resolver_probe::canonical(path, false);
    }
    std::string _CreateIdentifierForNewAsset(const std::string& path, const ArResolvedPath& anchor) const override {
        live_resolver_probe::record("CreateIdentifierForNewAsset", path, anchor.GetPathString());
        return live_resolver_probe::canonical(path, false);
    }
    ArResolvedPath _Resolve(const std::string& path) const override {
        live_resolver_probe::record("Resolve", path);
        return ArResolvedPath(live_resolver_probe::canonical(path, true));
    }
    ArResolvedPath _ResolveForNewAsset(const std::string& path) const override {
        live_resolver_probe::record("ResolveForNewAsset", path);
        return ArResolvedPath(live_resolver_probe::canonical(path, true));
    }
    ArTimestamp _GetModificationTimestamp(const std::string& path, const ArResolvedPath& resolved) const override {
        live_resolver_probe::record("GetModificationTimestamp", path, resolved.GetPathString());
        return ArTimestamp(); // Invalid timestamp deliberately requires reload.
    }
    std::shared_ptr<ArAsset> _OpenAsset(const ArResolvedPath& path) const override {
        live_resolver_probe::record("OpenAsset", path.GetPathString());
        return {}; // The custom file format supplies its own in-memory content.
    }
    bool _CanWriteAssetToPath(const ArResolvedPath& path, std::string*) const override {
        live_resolver_probe::record("CanWriteAssetToPath", path.GetPathString());
        return !live_resolver_probe::canonical(path.GetPathString(), true).empty();
    }
    std::shared_ptr<ArWritableAsset> _OpenAssetForWrite(const ArResolvedPath& path, WriteMode) const override {
        live_resolver_probe::record("OpenAssetForWrite", path.GetPathString());
        return {};
    }
};
AR_DEFINE_RESOLVER(LiveHookProbeResolver, ArResolver);

class LiveHookProbeFormat final : public SdfFileFormat {
public:
    LiveHookProbeFormat() : SdfFileFormat(TfToken("liveprobe"), TfToken("1"), TfToken("usd"), "live") {}
    bool CanRead(const std::string& path) const override {
        live_resolver_probe::record("CanRead", path);
        return !live_resolver_probe::canonical(path, true).empty();
    }
    bool Read(SdfLayer* layer, const std::string& path, bool metadataOnly) const override {
        live_resolver_probe::record("Read", path, metadataOnly ? "metadata" : "full");
        if (live_resolver_probe::canonical(path, true).empty()) return false;
        SdfAbstractDataRefPtr data = TfCreateRefPtr(new SdfData);
        data->CreateSpec(SdfPath::AbsoluteRootPath(), SdfSpecTypePseudoRoot);
        data->Set(SdfPath::AbsoluteRootPath(), SdfFieldKeys->Comment, VtValue(std::string("probe-read")));
        _SetLayerData(layer, data);
        return true;
    }
    bool SaveToFile(const SdfLayer& layer, const std::string& path, const std::string&,
                    const FileFormatArguments&) const override {
        live_resolver_probe::record("SaveToFile", path);
        namespace probe = live_resolver_probe;
        const auto& mode = probe::creationCase;
        if (mode == "create-deferred-work") probe::enqueueAnonymous();
        if (mode == "create-install" || mode == "create-freeze-failure") {
            SdfAbstractDataRefPtr data = TfCreateRefPtr(new SdfData);
            data->CreateSpec(SdfPath::AbsoluteRootPath(), SdfSpecTypePseudoRoot);
            data->Set(SdfPath::AbsoluteRootPath(), SdfFieldKeys->Comment, VtValue(std::string("hook-bootstrap")));
            probe::marker("before-initial-data-swap");
            _SetLayerData(const_cast<SdfLayer*>(&layer), data);
            probe::marker("after-initial-data-swap");
            if (mode == "create-freeze-failure") {
                // Pin backing only while the caller still protects the target.
                probe::capturedData = TfCreateRefPtrFromProtectedWeakPtr(_GetLayerData(layer));
                probe::capturedTarget = SdfLayerHandle(const_cast<SdfLayer*>(&layer));
                return false;
            }
        }
        if (mode == "create-release-failure") {
            probe::capturedTarget = SdfLayerHandle(const_cast<SdfLayer*>(&layer));
            return false;
        }
        return !live_resolver_probe::canonical(path, true).empty();
    }
    bool WriteToFile(const SdfLayer&, const std::string& path, const std::string&,
                     const FileFormatArguments&) const override {
        live_resolver_probe::record("WriteToFile", path);
        if (live_resolver_probe::creationCase == "export-nested") {
            live_resolver_probe::marker("before-nested-anonymous");
            const auto nested = SdfLayer::CreateAnonymous("nested.usda");
            if (!nested) throw std::runtime_error("nested anonymous creation failed");
            live_resolver_probe::marker("after-nested-anonymous");
        }
        return !live_resolver_probe::canonical(path, true).empty();
    }
};
TF_REGISTRY_FUNCTION(TfType) { SDF_DEFINE_FILE_FORMAT(LiveHookProbeFormat, SdfFileFormat); }
PXR_NAMESPACE_CLOSE_SCOPE
