// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "Probe.h"
#include <pxr/pxr.h>
#include <pxr/base/plug/registry.h>
#include <pxr/base/tf/errorMark.h>
#include <pxr/base/tf/notice.h>
#include <pxr/base/tf/weakBase.h>
#include <pxr/usd/ar/resolver.h>
#include <pxr/usd/ar/resolvedPath.h>
#include <pxr/usd/sdf/layer.h>
#include <pxr/usd/sdf/notice.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace PXR_NS;
namespace probe = live_resolver_probe;
namespace {
struct Step { std::string name; std::vector<probe::Event> events; };
std::vector<Step> steps;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
size_t position(const std::vector<probe::Event>& events, const char* hook) {
    const auto found = std::find_if(events.begin(), events.end(), [&](const auto& e) { return e.hook == hook; });
    return static_cast<size_t>(found - events.begin());
}
bool has(const std::vector<probe::Event>& events, const char* hook) { return position(events, hook) != events.size(); }
std::vector<probe::Event> finish(const char* step) {
    auto events = probe::events(); steps.push_back({step, events}); probe::clear(); return events;
}
void noAssetOpen(const std::vector<probe::Event>& events) {
    require(!has(events, "OpenAsset") && !has(events, "OpenAssetForWrite"),
            "USD unexpectedly requires resolver byte assets for custom format I/O");
}
void quoted(std::ostream& out, const std::string& value) {
    out << '"';
    const char* hex = "0123456789abcdef";
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << static_cast<char>(c);
        else if (c < 0x20) out << "\\u00" << hex[c >> 4] << hex[c & 15];
        else out << static_cast<char>(c);
    }
    out << '"';
}
void report(std::ostream& out, const std::string& error) {
    out << "{\"schema_version\":1,\"boundary\":\"usd_resolver_hook_probe\",\"usd_version\":2511,"
           "\"service\":false,\"production_plugin\":false,\"passed\":" << (error.empty() ? "true" : "false")
        << ",\"error\":"; quoted(out, error); out << ",\"steps\":[";
    bool first = true;
    for (const auto& step : steps) {
        if (!first) out << ',';
        first = false; out << "{\"name\":"; quoted(out, step.name); out << ",\"events\":[";
        bool firstEvent = true;
        for (const auto& event : step.events) {
            if (!firstEvent) out << ',';
            firstEvent = false; out << "{\"hook\":"; quoted(out, event.hook);
            out << ",\"path\":"; quoted(out, event.path); out << ",\"detail\":";
            quoted(out, event.detail); out << '}';
        }
        out << "]}";
    }
    out << "]}\n";
}
void run(const std::string& plugins, const std::filesystem::path& artifact) {
    static_assert(PXR_VERSION == 2511, "Probe supports only USD 25.11");
    // Must precede the first Ar access. No competing live handler is registered
    // by this executable, and no plugin/runtime from the old delegate is linked.
    PlugRegistry::GetInstance().RegisterPlugins(plugins);
    probe::clear();
    const std::string canonical = "live://probe.example/session/root.live";
    const std::string alias = "live://PROBE.EXAMPLE:443/session/%72oot.live";
    auto layer = SdfLayer::FindOrOpen(alias);
    const auto opening = finish("find_or_open");
    require(bool(layer), "FindOrOpen failed before custom format Read");
    require(layer->GetIdentifier() == canonical && layer->GetResolvedPath().GetPathString() == canonical,
            "URI normalization did not define canonical layer identity");
    require(has(opening, "CreateIdentifier") && has(opening, "Resolve") && has(opening, "Read"),
            "open omitted an expected identifier/resolve/read hook");
    require(position(opening, "CreateIdentifier") < position(opening, "Read") &&
            position(opening, "Resolve") < position(opening, "Read"), "Read ran before required resolution hooks");
    noAssetOpen(opening);
    require(layer->GetComment() == "probe-read", "custom Read did not supply scene data");

    auto duplicate = SdfLayer::FindOrOpen(canonical);
    auto aliasDuplicate = SdfLayer::FindOrOpen(alias);
    const auto cached = finish("canonical_duplicate_open");
    require(duplicate == layer && aliasDuplicate == layer && !has(cached, "Read"), "canonical URI duplicate reopened content");

    SdfLayer::FileFormatArguments args{{"alpha", "one"}, {"beta", "two"}};
    auto withArgs = SdfLayer::FindOrOpen(alias, args);
    const auto argumentOpen = finish("format_arguments_open");
    require(withArgs && withArgs != layer && withArgs->GetFileFormatArguments() == args && has(argumentOpen, "Read"),
            "format arguments did not create their own Sdf identity");
    noAssetOpen(argumentOpen);
    auto sameArgs = SdfLayer::FindOrOpen(SdfLayer::CreateIdentifier(canonical, args));
    const auto argumentDuplicate = finish("format_arguments_duplicate");
    require(sameArgs == withArgs && !has(argumentDuplicate, "Read"), "encoded and explicit arguments did not share a layer");
    for (const auto& event : argumentOpen)
        if (event.hook == "CreateIdentifier" || event.hook == "Resolve" || event.hook == "Read")
            require(event.path.find("SDF_FORMAT_ARGS") == std::string::npos, "Sdf arguments leaked into resolver asset paths");

    const auto created = SdfLayer::CreateNew("live://probe.example/session/created.live");
    const auto creating = finish("create_new");
    require(bool(created), "CreateNew failed");
    require(has(creating, "CreateIdentifierForNewAsset") && has(creating, "ResolveForNewAsset") && !has(creating, "Read"),
            "CreateNew did not use new-asset hooks or unexpectedly read existing data");
    noAssetOpen(creating);
    created->SetComment("created-edit");
    require(created->Save(), "Save failed");
    const auto saving = finish("save");
    require(has(saving, "SaveToFile") && !has(saving, "Read"), "Save did not reach SaveToFile");
    noAssetOpen(saving);

    require(layer->Export("live://probe.example/session/export.live"), "live-format Export failed");
    const auto exporting = finish("export_live_format");
    require(has(exporting, "WriteToFile") && !has(exporting, "SaveToFile"), "Export did not select WriteToFile separately from Save");
    noAssetOpen(exporting);

    const auto localFile = std::filesystem::absolute(artifact).parent_path() / "resolver-probe-export.usda";
    require(layer->Export(localFile.string()), "ordinary USDA export failed");
    auto local = SdfLayer::FindOrOpen(localFile.string());
    require(local && local->GetComment() == "probe-read", "ordinary USDA export did not preserve data");
    const auto localExport = finish("export_stock_usda");
    require(!has(localExport, "Read") && !has(localExport, "WriteToFile"), "stock USDA export used custom format");

    layer->SetComment("discard-on-reload");
    require(layer->Reload(true), "forced Reload failed");
    const auto reloading = finish("reload_force");
    require(has(reloading, "Read") && layer->GetComment() == "probe-read", "Reload did not reread custom data");
    noAssetOpen(reloading);

    for (const auto& rejected : {std::string("live://probe.example/session/bad%Q0.live"),
                                 std::string("live://unregistered.example/session/root.live")}) {
        TfErrorMark mark;
        const auto invalid = SdfLayer::FindOrOpen(rejected);
        const auto events = finish(rejected.find("bad%") != std::string::npos ? "malformed_uri" : "unregistered_authority");
        require(!invalid && !has(events, "Read"), "invalid/unregistered URI reached file-format Read");
        mark.Clear();
    }
    probe::registered(false);
    {
        TfErrorMark mark;
        const auto invalid = SdfLayer::FindOrOpen("live://probe.example/session/after-unregister.live");
        const auto events = finish("endpoint_unregistered");
        require(!invalid && !has(events, "Read"), "unregistered endpoint admitted a new layer");
        mark.Clear();
    }
    probe::registered(true);
    // Positive instrumentation control: an explicit asset open really reaches
    // the hook. Thus its absence above is not a broken recorder/registration.
    require(!ArGetResolver().OpenAsset(ArResolvedPath(canonical)), "probe unexpectedly supplies byte assets");
    require(has(finish("explicit_open_asset_control"), "OpenAsset"), "resolver asset-open hook was not registered");
}
struct CreationNotices : TfWeakBase {
    const bool deferSave;
    size_t changes = 0, saves = 0;
    TfNotice::Key changesKey, savedKey;
    explicit CreationNotices(bool defer) : deferSave(defer) {
        changesKey = TfNotice::Register(TfCreateWeakPtr(this), &CreationNotices::changed);
        savedKey = TfNotice::Register(TfCreateWeakPtr(this), &CreationNotices::saved);
    }
    ~CreationNotices() { TfNotice::Revoke(changesKey); TfNotice::Revoke(savedKey); }
    void changed(const SdfNotice::LayersDidChange&) { ++changes; probe::marker("content-notice"); }
    void saved(const SdfNotice::LayerDidSaveLayerToFile&) {
        ++saves;
        if (deferSave) {
            probe::marker("save-notice-queued");
            probe::enqueueAnonymous();
        }
    }
};
void creationContract(const std::string& plugins, const std::string& mode) {
    PlugRegistry::GetInstance().RegisterPlugins(plugins);
    probe::clear(); probe::creationMode(mode);
    const std::string path = "live://probe.example/session/hook-creation.live";
    // Preload all ordinary format/plugin setup before the instrumented call.
    { const auto control = SdfLayer::CreateAnonymous("control.usda");
      require(bool(control), "anonymous positive control failed"); }
    probe::marker("anonymous-control-passed");
    CreationNotices notices(mode == "create-deferred-save-notice");
    { const auto control = SdfLayer::CreateAnonymous("notice-control.usda");
      require(bool(control), "notice control layer creation failed");
      control->SetComment("prove-content-notice");
      require(notices.changes > 0, "content notice listener did not fire"); }
    notices.changes = 0;
    probe::marker("content-notice-control-passed");
    if (mode == "export-nested") {
        const auto source = SdfLayer::CreateAnonymous("source.usda");
        require(source->Export(path), "export hook failed");
    } else {
        TfErrorMark errors;
        const auto target = SdfLayer::CreateNew(path);
        errors.Clear();
        probe::marker("outer-create-returned");
        const auto drained = probe::drainDeferred();
        require(drained == ((mode == "create-deferred-work" || mode == "create-deferred-save-notice") ? 1u : 0u),
                "deferred work did not run exactly once after outer call returned");
        if (mode == "create-release-failure" || mode == "create-freeze-failure") {
            require(!target, "failed save returned successful CreateNew");
            require(probe::capturedTargetExpired(), "failed target remains alive");
            probe::marker("failed-target-destroyed");
            if (mode == "create-freeze-failure") {
                require(probe::capturedComment() == "hook-bootstrap", "frozen backing lost installed state");
                probe::marker("frozen-backing-survived-target");
            }
            probe::marker("before-find-after-failure");
            const auto reopened = SdfLayer::FindOrOpen(path);
            require(reopened && reopened->GetComment() == "probe-read", "failure did not permit a fresh read");
            probe::marker("after-find-after-failure");
            if (mode == "create-freeze-failure")
                require(probe::capturedComment() == "hook-bootstrap", "fresh read changed frozen failed host");
        } else {
            require(bool(target), "CreateNew hook failed");
            require(notices.saves == 1, "save notice positive control missing");
            if (mode == "create-install") {
                require(target->GetComment() == "hook-bootstrap", "initial backing swap not installed");
                require(notices.changes == 0, "initial backing swap emitted a content notice");
                require(SdfLayer::FindOrOpen(path) == target, "initialized target was not reusable");
            }
        }
    }
    finish(mode.c_str());
    probe::marker("completed");
}
}
int main(int argc, char** argv) {
    if (argc != 3 && argc != 4) { std::cerr << "usage: live_resolver_probe PLUGIN_INFO TRACE_JSON [CREATION_CASE]\n"; return 2; }
    std::string error;
    try { if (argc == 4) creationContract(argv[1], argv[3]); else run(argv[1], argv[2]); }
    catch (const std::exception& exception) { error = exception.what(); finish("failure_tail"); }
    report(std::cout, error);
    std::ofstream artifact(argv[2], std::ios::binary | std::ios::trunc);
    if (!artifact) { std::cerr << "cannot write hook trace artifact\n"; return 2; }
    report(artifact, error); artifact.close();
    return artifact && error.empty() ? 0 : 1;
}
