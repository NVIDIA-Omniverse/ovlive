// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "LiveMetadata.h"
#include "LiveRuntime.h"
#include "LiveStatus.h"
#include "LiveUri.h"
#include "LiveExceptionResult.h"
#include <pxr/pxr.h>
#include <pxr/base/tf/diagnostic.h>
#include <pxr/base/tf/enum.h>
#include <pxr/usd/ar/defineResolver.h>
#include <pxr/usd/ar/resolvedPath.h>
#include <pxr/usd/ar/timestamp.h>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

PXR_NAMESPACE_OPEN_SCOPE
// A typed USD diagnostic, with the precise public result retained in its
// TfDiagnosticInfo. Messages never contain URI, token, or remote error text.
enum class OmniUsdLiveResolverError { Failure };
TF_REGISTRY_FUNCTION(TfEnum) { TF_ADD_ENUM_NAME(OmniUsdLiveResolverError::Failure); }

namespace {
namespace live = ::omni::usd::live;
struct ResolverFailure {
    OmniUsdLiveResult result;
    const char* stage; // Fixed source text only, never an input/server string.
};

void reportFailure(const char* operation, OmniUsdLiveResult result, std::string* whyNot) noexcept {
    // Diagnostic allocation failure must not allow an exception to escape a USD
    // hook, nor turn a failed operation into a successful resolved asset.
    try {
        if (whyNot)
            *whyNot = std::string("Live resolver ") + operation + " rejected (result=" + std::to_string(result) + ").";
        TF_ERROR(TfDiagnosticInfo(result), TfEnum(OmniUsdLiveResolverError::Failure),
            "Live resolver %s failed (result=%u)", operation, unsigned(result));
    } catch (...) {}
}

template<class Result, class Action>
Result guarded(const char* operation, Action&& action, std::string* whyNot = nullptr) noexcept {
    try {
        return action();
    } catch (const ResolverFailure& failure) { reportFailure(failure.stage, failure.result, whyNot); } catch (...) {
        reportFailure(operation, live::currentExceptionResult(), whyNot);
    }
    return Result{};
}

live::LiveUri layerUri(const std::string& path) {
    auto uri = live::parseUri(path);
    if (uri.layerName.empty()) throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
    return uri;
}

bool liveLayerPath(const std::string& path) noexcept {
    static constexpr std::string_view suffix = ".live";
    return path.size() >= suffix.size() && path.compare(path.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string identifier(const std::string& path, const ArResolvedPath& anchor) {
    if (path.compare(0, 7, "live://") == 0) return layerUri(path).canonical;
    // A live URI is not the storage directory of the base scene. Only relative
    // live-layer paths anchor here; an ordinary asset (texture, payload, USD
    // file) anchored in a live layer has no live identifier. An empty result
    // with no diagnostic lets USD report it unresolved and issues no RPC.
    if (!liveLayerPath(path)) return {};
    if (path.size() > live::kMaxUriBytes) throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    if (path.front() == '/' || path.find(':') != std::string::npos || anchor.IsEmpty())
        throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
    const auto base = layerUri(anchor.GetPathString());
    const auto slash = base.canonical.rfind('/');
    const auto prefixSize = slash + 1;
    if (path.size() > live::kMaxUriBytes - prefixSize) throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    // Combine encoded URI components, then let the one canonical codec decode
    // and validate them. Decoding the session or layer before joining could
    // mistake an opaque escaped session slash for a directory boundary.
    return layerUri(base.canonical.substr(0, prefixSize) + path).canonical;
}

struct ActiveEndpoint {
    // Runtime must outlive its endpoint lease, including failure unwinding.
    std::shared_ptr<live::Runtime> runtime;
    live::EndpointLease endpoint;
};

ActiveEndpoint endpointFor(const live::LiveUri& uri) {
    auto runtime = live::getRuntime();
    if (!runtime) throw OmniUsdLiveResult(OMNI_USD_LIVE_NOT_INITIALIZED);
    OmniUsdLiveEndpoint handle;
    {
        std::lock_guard<std::mutex> lock(runtime->mutex);
        if (runtime->closing) throw OmniUsdLiveResult(OMNI_USD_LIVE_BUSY);
        const auto found = runtime->authorities.find(uri.authority.canonical);
        if (found == runtime->authorities.end())
            throw ResolverFailure{OMNI_USD_LIVE_NOT_FOUND, "configuration: endpoint authority is not registered"};
        handle = found->second;
    }
    auto lease = runtime->acquireEndpoint(handle);
    if (!lease) throw OmniUsdLiveResult(OMNI_USD_LIVE_BUSY);
    return {std::move(runtime), std::move(lease)};
}

uint64_t metadata(const live::LiveUri& uri, std::shared_ptr<live::Runtime>* runtime = nullptr) {
    // Pin the same registered endpoint through stat and interpretation. A
    // concurrent unregister cannot replace its configuration under this query.
    const auto active = endpointFor(uri);
    OmniUsdLiveLayerInfo info{};
    info.struct_size = sizeof(info);
    info.abi_version = OMNI_USD_LIVE_ABI_VERSION;
    uint64_t modifiedTimeUs = 0;
    const auto result = live::statLayerMetadata(uri.canonical, info, &modifiedTimeUs);
    if (result != OMNI_USD_LIVE_OK) throw result;
    if (runtime) *runtime = active.runtime;
    return modifiedTimeUs;
}

// The subject ID of the layer attached for a canonical URI, or zero. A
// closing or closed backing is not attached: its reopen stats again.
uint64_t attachedSubject(const live::Runtime& runtime, const std::string& canonical) noexcept {
    if (runtime.closing) return 0;
    OmniUsdLiveStatus status{};
    status.struct_size = sizeof(status);
    status.abi_version = OMNI_USD_LIVE_ABI_VERSION;
    if (runtime.statuses->layer(canonical, &status) != OMNI_USD_LIVE_OK ||
        status.state == OMNI_USD_LIVE_STATE_CLOSING || status.state == OMNI_USD_LIVE_STATE_CLOSED)
        return 0;
    return status.subject_id;
}

// The last successful StatLayer modification time per canonical URI of one
// runtime instance. An entry serves only the attached subject it is bound to;
// a stat taken while no subject was attached binds to the next attached
// subject that reads it. Entries are dropped when a lookup finds no attached
// subject, and the map holds at most max_status_subjects entries.
class ModificationTimes {
    struct Entry {
        uint64_t subject;
        uint64_t modifiedTimeUs;
    };
    std::mutex mutex_;
    std::string instance_;
    std::unordered_map<std::string, Entry> entries_;

    void adopt(const live::Runtime& runtime) { // mutex_ held
        if (instance_ == runtime.instanceNonce) return;
        entries_.clear();
        instance_ = runtime.instanceNonce;
    }

public:
    void record(const live::Runtime& runtime, const std::string& uri, uint64_t subject, uint64_t modifiedTimeUs) {
        std::lock_guard<std::mutex> lock(mutex_);
        adopt(runtime);
        const auto found = entries_.find(uri);
        if (found != entries_.end()) {
            found->second = {subject, modifiedTimeUs};
            return;
        }
        if (entries_.size() >= runtime.options.values.max_status_subjects) entries_.clear();
        entries_.emplace(uri, Entry{subject, modifiedTimeUs});
    }
    std::optional<uint64_t> find(const live::Runtime& runtime, const std::string& uri, uint64_t subject) {
        std::lock_guard<std::mutex> lock(mutex_);
        adopt(runtime);
        const auto found = entries_.find(uri);
        if (found == entries_.end()) return std::nullopt;
        if (!subject) {
            entries_.erase(found);
            return std::nullopt;
        }
        if (found->second.subject && found->second.subject != subject) return std::nullopt;
        found->second.subject = subject;
        return found->second.modifiedTimeUs;
    }
    void forget(const live::Runtime& runtime, const std::string& uri) {
        std::lock_guard<std::mutex> lock(mutex_);
        adopt(runtime);
        entries_.erase(uri);
    }
};
}

class OmniUsdLiveResolver final : public ArResolver {
    mutable ModificationTimes times_;

    // Stat the layer and cache its modification time for the subject attached
    // when the stat began (zero when none was).
    uint64_t stat(const live::LiveUri& uri, const std::shared_ptr<live::Runtime>& runtime, uint64_t subject) const {
        std::shared_ptr<live::Runtime> used;
        const auto modifiedTimeUs = metadata(uri, &used);
        times_.record(*used, uri.canonical, used == runtime ? subject : 0, modifiedTimeUs);
        return modifiedTimeUs;
    }

    std::string _CreateIdentifier(const std::string& path, const ArResolvedPath& anchor) const override {
        return guarded<std::string>("CreateIdentifier", [&] { return identifier(path, anchor); });
    }

    std::string _CreateIdentifierForNewAsset(const std::string& path, const ArResolvedPath& anchor) const override {
        return guarded<std::string>("CreateIdentifierForNewAsset", [&] { return identifier(path, anchor); });
    }

    ArResolvedPath _Resolve(const std::string& path) const override {
        return guarded<ArResolvedPath>("Resolve", [&] {
            const auto uri = layerUri(path);
            // An attached layer is existence evidence: a cached open issues no
            // RPC. Without one, the stat proves existence and drops any cached
            // time of a previous attachment.
            const auto runtime = live::getRuntime();
            if (runtime) {
                if (attachedSubject(*runtime, uri.canonical)) return ArResolvedPath(uri.canonical);
                times_.forget(*runtime, uri.canonical);
            }
            (void)stat(uri, runtime, 0);
            return ArResolvedPath(uri.canonical);
        });
    }

    ArResolvedPath _ResolveForNewAsset(const std::string& path) const override {
        return guarded<ArResolvedPath>("ResolveForNewAsset", [&] {
            const auto uri = layerUri(path);
            const auto active = endpointFor(uri);
            // Eligibility is local configuration, not creation or permission
            // evidence. The file format must perform its own mutation admission.
            // A new asset's layer never reads a time cached before its creation.
            times_.forget(*active.runtime, uri.canonical);
            return ArResolvedPath(uri.canonical);
        });
    }

    bool _CanWriteAssetToPath(const ArResolvedPath& path, std::string* whyNot) const override {
        return guarded<bool>("CanWriteAssetToPath", [&] {
            const auto active = endpointFor(layerUri(path.GetPathString()));
            if (whyNot) whyNot->clear();
            return true;
        }, whyNot);
    }

    ArTimestamp _GetModificationTimestamp(const std::string& path, const ArResolvedPath& resolved) const override {
        return guarded<ArTimestamp>("GetModificationTimestamp", [&] {
            const auto uri = layerUri(resolved.GetPathString());
            if (path.compare(0, 7, "live://") == 0 && layerUri(path).canonical != uri.canonical)
                throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
            // An attached layer reads the time of its last successful stat. A
            // forced reload of an attached layer is a no-op, so a time older
            // than the service's only suppresses a reload that does nothing.
            const auto runtime = live::getRuntime();
            const auto subject = runtime ? attachedSubject(*runtime, uri.canonical) : 0;
            std::optional<uint64_t> cached;
            if (runtime) cached = times_.find(*runtime, uri.canonical, subject);
            const auto modifiedTimeUs = cached ? *cached : stat(uri, runtime, subject);
            // The protocol has no presence bit: successful zero is Unix epoch.
            // Preserve the wire's microseconds internally; truncating to the
            // public ABI's milliseconds could hide a change during reload.
            return ArTimestamp(static_cast<double>(modifiedTimeUs) / 1000000.0);
        });
    }

    std::shared_ptr<ArAsset> _OpenAsset(const ArResolvedPath&) const override {
        reportFailure("OpenAsset", OMNI_USD_LIVE_UNSUPPORTED, nullptr);
        return {};
    }

    std::shared_ptr<ArWritableAsset> _OpenAssetForWrite(const ArResolvedPath&, WriteMode) const override {
        reportFailure("OpenAssetForWrite", OMNI_USD_LIVE_UNSUPPORTED, nullptr);
        return {};
    }
};
AR_DEFINE_RESOLVER(OmniUsdLiveResolver, ArResolver);
PXR_NAMESPACE_CLOSE_SCOPE
