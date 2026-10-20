// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <omni/usd/live/LiveUsd.h>
#include "LiveSdfFileFormat.h"
#include "UsdLayerCreation.h"
#include "UsdCreationRegistry.h"
#include "LiveRuntime.h"
#include "LiveUri.h"
#include "LiveExceptionResult.h"
#include <memory>
#include <mutex>

namespace omni::usd::live::host {
namespace {
OmniUsdLiveResult prepareRetained(const PXR_NS::SdfLayer& seed, const char* uri, uint64_t* subject,
    std::shared_ptr<const void> expectedContext) {
    if (!subject || !uri) return OMNI_USD_LIVE_INVALID_ARGUMENT;
    size_t uriSize = 0;
    while (uriSize <= kMaxUriBytes && uri[uriSize]) ++uriSize;
    if (uriSize > kMaxUriBytes) return OMNI_USD_LIVE_RESOURCE_EXHAUSTED;
    auto lease = reserveUsdCreation();
    auto workflow = UsdLayerCreation::prepare(getRuntime(), seed, std::string(uri, uriSize), {},
        std::move(expectedContext));
    const auto id = workflow->creation()->status()->id();
    lease.install(workflow);
    *subject = id;
    return OMNI_USD_LIVE_OK;
}
}
OmniUsdLiveResult setPublicationContext(PublicationDispatcher dispatch, std::function<bool()> isCurrent) noexcept {
    return resultBoundary([&] {
        setUsdPublicationDispatcher(std::move(dispatch), std::move(isCurrent));
        return OmniUsdLiveResult(OMNI_USD_LIVE_OK);
    });
}
struct PublicationContext::Impl {
    std::mutex mutex;
    std::shared_ptr<const void> context;
    bool retiring = false;
    bool closing = false;
};
PublicationContext::PublicationContext(Impl* value) noexcept : impl_(value) {}
PublicationContext::~PublicationContext() {
    unpublishUsdPublicationContext(impl_->context);
    delete impl_; // Captures retire without the global or owner mutex held.
}
OmniUsdLiveResult PublicationContext::close() noexcept {
    std::shared_ptr<const void> retired;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->closing = true;
        if (!impl_->context) return impl_->retiring ? OMNI_USD_LIVE_BUSY : OMNI_USD_LIVE_OK;
        unpublishUsdPublicationContext(impl_->context);
        // No weak Host promotions exist. After removing global publication,
        // every possible future callback is rooted in another strong owner.
        if (impl_->context.use_count() != 1) return OMNI_USD_LIVE_BUSY;
        retired = std::move(impl_->context);
        impl_->retiring = true;
    }
    retired.reset();
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->retiring = false;
    }
    return OMNI_USD_LIVE_OK;
}
OmniUsdLiveResult createPublicationContext(PublicationDispatcher dispatch, std::function<bool()> isCurrent,
    PublicationContext** output) noexcept {
    return resultBoundary([&] {
        if (!output || *output || !dispatch || !isCurrent) return OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
        auto implementation = std::make_unique<PublicationContext::Impl>();
        std::unique_ptr<PublicationContext> owner(new PublicationContext(implementation.get()));
        implementation.release();
        owner->impl_->context = createUsdPublicationContext(std::move(dispatch), std::move(isCurrent));
        *output = owner.release();
        return OmniUsdLiveResult(OMNI_USD_LIVE_OK);
    });
}
bool dispatchingLocalPublication() noexcept { return LocalPublicationScope::active(); }
OmniUsdLiveResult save(const PXR_NS::SdfLayer& layer, uint32_t timeoutMs) noexcept {
    return resultBoundary([&] { return saveAttachedLiveScene(layer, timeoutMs); });
}
struct Creation::Impl {
    std::shared_ptr<UsdLayerCreation> workflow;
    std::shared_ptr<SceneCreation> finite; // Off-host observation never touches mutable USD ownership.
};
Creation::Creation(Impl* value) noexcept : impl_(value) {}
Creation::~Creation() { delete impl_; }
OmniUsdLiveResult prepareCreation(const PXR_NS::SdfLayer& seed, const char* uri, Creation** output) noexcept {
    return Creation::prepare(seed, uri, output, {});
}
OmniUsdLiveResult PublicationContext::prepareCreation(const PXR_NS::SdfLayer& seed, const char* uri,
    Creation** output) noexcept {
    return resultBoundary([&] {
        std::shared_ptr<const void> context;
        {
            std::lock_guard<std::mutex> lock(impl_->mutex);
            if (impl_->closing || !impl_->context) return OmniUsdLiveResult(OMNI_USD_LIVE_NOT_INITIALIZED);
            context = impl_->context;
        }
        return Creation::prepare(seed, uri, output, std::move(context));
    });
}
OmniUsdLiveResult PublicationContext::prepareRetainedCreation(const PXR_NS::SdfLayer& seed, const char* uri,
    uint64_t* subject) noexcept {
    return resultBoundary([&] {
        std::shared_ptr<const void> context;
        {
            std::lock_guard<std::mutex> lock(impl_->mutex);
            if (impl_->closing || !impl_->context) return OmniUsdLiveResult(OMNI_USD_LIVE_NOT_INITIALIZED);
            context = impl_->context;
        }
        return prepareRetained(seed, uri, subject, std::move(context));
    });
}
OmniUsdLiveResult prepareRetainedCreation(const PXR_NS::SdfLayer& seed, const char* uri, uint64_t* subject) noexcept {
    return resultBoundary([&] { return prepareRetained(seed, uri, subject, {}); });
}
OmniUsdLiveResult nextRetainedCreation(uint64_t afterSubject, RetainedCreationInfo* output) noexcept {
    return resultBoundary([&] {
        if (!output) return OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
        RetainedUsdCreationInfo value;
        if (!nextRetainedUsdCreation(afterSubject, value)) return OmniUsdLiveResult(OMNI_USD_LIVE_NOT_FOUND);
        output->subjectId = value.subjectId;
        output->liveUri = std::move(value.liveUri);
        return OmniUsdLiveResult(OMNI_USD_LIVE_OK);
    });
}
OmniUsdLiveResult acquireRetainedCreation(uint64_t subject, Creation** output) noexcept {
    return resultBoundary([&] {
        if (!output || *output) return OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
        auto lease = acquireUsdCreation(subject);
        lease.workflow().requireTransferHost();
        auto implementation = std::make_unique<Creation::Impl>();
        std::unique_ptr<Creation> owner(new Creation(implementation.get()));
        implementation.release();
        owner->impl_->finite = lease.workflow().creation();
        owner->impl_->workflow = lease.take();
        *output = owner.release();
        return OmniUsdLiveResult(OMNI_USD_LIVE_OK);
    });
}
OmniUsdLiveResult releaseRetainedCreation(uint64_t subject) noexcept {
    return resultBoundary([&] { return acquireUsdCreation(subject).close(); });
}
OmniUsdLiveResult Creation::prepare(const PXR_NS::SdfLayer& seed, const char* uri, Creation** output,
    std::shared_ptr<const void> expectedContext) noexcept {
    return resultBoundary([&] {
        if (!output || *output || !uri) return OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
        size_t uriSize = 0;
        while (uriSize <= kMaxUriBytes && uri[uriSize]) ++uriSize;
        if (uriSize > kMaxUriBytes) return OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
        auto implementation = std::make_unique<Creation::Impl>();
        std::unique_ptr<Creation> owner(new Creation(implementation.get()));
        implementation.release();
        owner->impl_->workflow = UsdLayerCreation::prepare(getRuntime(), seed, std::string(uri, uriSize), {},
            std::move(expectedContext));
        owner->impl_->finite = owner->impl_->workflow->creation();
        *output = owner.release();
        return OmniUsdLiveResult(OMNI_USD_LIVE_OK);
    });
}
OmniUsdLiveResult Creation::start() noexcept {
    return resultBoundary([&] { return impl_->workflow->start(); });
}
OmniUsdLiveResult Creation::wait(uint32_t timeoutMs) noexcept { return impl_->finite->wait(timeoutMs); }
OmniUsdLiveResult Creation::getOutcome(CreationOutcome* output) const noexcept {
    return resultBoundary([&] {
        if (!output) return OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
        const auto snapshot = impl_->finite->snapshot();
        CreationOutcome value;
        switch (snapshot.phase) {
        case SceneCreation::Phase::Prepared: value.phase = CreationPhase::Prepared; break;
        case SceneCreation::Phase::Creating: value.phase = CreationPhase::Creating; break;
        case SceneCreation::Phase::Confirmed: value.phase = CreationPhase::Confirmed; break;
        case SceneCreation::Phase::UnissuedFailure: value.phase = CreationPhase::UnissuedFailure; break;
        case SceneCreation::Phase::Rejected: value.phase = CreationPhase::Rejected; break;
        case SceneCreation::Phase::OutcomeUnknown: value.phase = CreationPhase::OutcomeUnknown; break;
        }
        value.result = snapshot.result;
        value.error = snapshot.error;
        value.subjectId = impl_->finite->status()->id();
        value.started = snapshot.started;
        value.closing = snapshot.closing;
        value.quiescent = snapshot.quiescent;
        *output = value;
        return OmniUsdLiveResult(OMNI_USD_LIVE_OK);
    });
}
OmniUsdLiveResult Creation::prepareLayer(uint32_t timeoutMs) noexcept {
    return impl_->workflow->prepareAttachment(timeoutMs);
}
OmniUsdLiveResult Creation::activate() noexcept { return impl_->workflow->activate(); }
OmniUsdLiveResult Creation::getLayer(PXR_NS::SdfLayerRefPtr* output) const noexcept {
    return resultBoundary([&] {
        if (!output) return OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
        const auto layer = impl_->workflow->layer();
        if (!layer) return OmniUsdLiveResult(OMNI_USD_LIVE_BUSY);
        *output = layer;
        return OmniUsdLiveResult(OMNI_USD_LIVE_OK);
    });
}
OmniUsdLiveResult Creation::close() noexcept { return impl_->workflow->close(); }
OmniUsdLiveResult Creation::waitForClose(uint32_t timeoutMs) noexcept {
    return impl_->workflow->waitForClose(timeoutMs);
}
} // namespace omni::usd::live::host
