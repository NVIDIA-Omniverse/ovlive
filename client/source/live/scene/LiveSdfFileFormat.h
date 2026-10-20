// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "SceneAttachment.h"
#include <pxr/usd/sdf/fileFormat.h>
#include <optional>
#include <chrono>
#include <functional>

namespace omni::usd::live {
class UsdLayerCreation;
class UsdCreationWorkflow;
void prepareInitializingCreationTarget(PXR_NS::SdfLayer&, const std::shared_ptr<UsdCreationWorkflow>&);
void bindInitializingCreationAttachment(PXR_NS::SdfLayer&, const std::shared_ptr<UsdCreationWorkflow>&,
    const std::shared_ptr<SceneAttachment>&);
SceneAttachment::Publication initializingCreationPublication(PXR_NS::SdfLayer&,
    const std::shared_ptr<UsdCreationWorkflow>&);
using UsdPublicationDispatcher = std::function<void(std::function<void()>)>;
// Private host integration boundary. A replacement affects future opens only;
// existing bindings retain their dispatcher and its captured host lifetime.
// FindOrOpen/Read and all publication MUST run on this same serial host
// context. Dispatch enqueues without waiting. Publication and flush jobs must
// wait until the enclosing USD call, notices and change blocks return, including
// during nested event pumps. Accepted jobs must run or release captures.
// No default worker may mutate an attached stage without host synchronization.
void setUsdPublicationDispatcher(UsdPublicationDispatcher, std::function<bool()> isCurrent = {});
// Opaque Host owner used by the exported lifecycle adapter. Unpublish does not
// destroy caller ownership and retires global ownership outside its mutex.
std::shared_ptr<const void> createUsdPublicationContext(UsdPublicationDispatcher, std::function<bool()>);
void unpublishUsdPublicationContext(const std::shared_ptr<const void>&) noexcept;
// Test seam: journal limits for attachments that later Reads open. Empty
// restores the defaults. Existing attachments keep their limits.
void setLiveLayerJournalLimitsForTest(std::optional<SceneJournal::Limits>);
// Test seam: the service's entry limits for attachments that later Reads open
// and for seeds that later Export and CreateNew compile, beside the journal
// limits above. Empty restores the service's limits. Existing attachments and
// prepared creations keep their limits.
void setLiveLayerEntryLimitsForTest(std::optional<DeltaEntryLimits>);
// Test seam: a host edit buffer quota in bytes for each layer that later Reads
// open, beside the endpoint's shared budget. Empty restores the default, no
// quota of the layer's own. Existing layers keep their quota.
void setLiveLayerHostEditQuotaForTest(std::optional<size_t> maxPendingBytes);
// Test seam: progress observer for attachments that later Reads or creation
// activations open. It receives every progress change as SceneAttachment's
// Changed callback does, on the layer's actor, and must only signal. Empty
// removes it. Existing attachments keep the observer they opened with.
void setLiveLayerProgressObserverForTest(SceneAttachment::Changed);
// Test seam: the monotonic clock Export (WriteToFile) and CreateNew
// (SaveToFile) read for their creation deadline, including a queued CreateNew
// finalizer. It runs on the host, must only read state and must return
// std::chrono::steady_clock times. Empty restores std::chrono::steady_clock.
void setLiveLayerCreationClockForTest(std::function<std::chrono::steady_clock::time_point()>);
std::shared_ptr<SceneAttachment> attachedLiveScene(const PXR_NS::SdfLayer&);
OmniUsdLiveResult saveAttachedLiveScene(const PXR_NS::SdfLayer&, uint32_t timeoutMs);
}

PXR_NAMESPACE_OPEN_SCOPE
class OmniUsdLiveNativeFileFormat final : public SdfFileFormat {
public:
    SdfAbstractDataRefPtr InitData(const FileFormatArguments&) const override;
    bool CanRead(const std::string&) const override;
    bool Read(SdfLayer*, const std::string&, bool) const override;
    bool WriteToFile(const SdfLayer&, const std::string&, const std::string&,
        const FileFormatArguments&) const override;
    bool SaveToFile(const SdfLayer&, const std::string&, const std::string&, const FileFormatArguments&) const override;
    bool WriteToString(const SdfLayer&, std::string*, const std::string&) const override;
    bool WriteToStream(const SdfSpecHandle&, std::ostream&, size_t) const override;
protected:
    friend class omni::usd::live::UsdLayerCreation;
    friend void omni::usd::live::prepareInitializingCreationTarget(PXR_NS::SdfLayer&,
        const std::shared_ptr<omni::usd::live::UsdCreationWorkflow>&);
    friend void omni::usd::live::bindInitializingCreationAttachment(PXR_NS::SdfLayer&,
        const std::shared_ptr<omni::usd::live::UsdCreationWorkflow>&,
        const std::shared_ptr<omni::usd::live::SceneAttachment>&);
    friend omni::usd::live::SceneAttachment::Publication omni::usd::live::initializingCreationPublication(
        PXR_NS::SdfLayer&, const std::shared_ptr<omni::usd::live::UsdCreationWorkflow>&);
    template<class T> friend class Sdf_FileFormatFactory;
    OmniUsdLiveNativeFileFormat();
    ~OmniUsdLiveNativeFileFormat() override = default;
    SdfLayer* _InstantiateNewLayer(const SdfFileFormatConstPtr&, const std::string&, const std::string&,
        const ArAssetInfo&, const FileFormatArguments&) const override;
    SdfAbstractDataRefPtr _InitDetachedData(const FileFormatArguments&) const override;
    bool _ReadDetached(SdfLayer*, const std::string&, bool) const override;
private:
    static omni::usd::live::SceneAttachment::Publication _PublicationFor(SdfLayer*);
    // An opened layer gains edit permission here; a creation layer is
    // attached unopened and gains it at activation.
    static void _Attach(SdfLayer*, const std::shared_ptr<omni::usd::live::SceneAttachment>&, bool opened = true);
};
PXR_NAMESPACE_CLOSE_SCOPE
