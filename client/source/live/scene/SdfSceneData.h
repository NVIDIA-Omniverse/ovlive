// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ReaderWriterLock.h"
#include <pxr/usd/sdf/abstractData.h>
#include <map>
#include <memory>
#include <utility>
#include <vector>

namespace omni::usd::live {
// Exact, transport-free scene storage. USD owns child fields and traverses namespace
// edits spec by spec; no derived hierarchy or transport acknowledgment lives here.
// The attachment will publish/reconcile this state with work-ID intents.
class SdfSceneData final : public PXR_NS::SdfAbstractData, public threading::ReaderWriterLock<SdfSceneData> {
public:
    SdfSceneData();
    ~SdfSceneData() override;
    bool StreamsData() const override { return false; }
    bool IsDetached() const override { return true; }
    void CopyFrom(const PXR_NS::SdfAbstractDataConstPtr&) override;
    void CreateSpec(const PXR_NS::SdfPath&, PXR_NS::SdfSpecType) override;
    bool HasSpec(const PXR_NS::SdfPath&) const override;
    void EraseSpec(const PXR_NS::SdfPath&) override;
    void MoveSpec(const PXR_NS::SdfPath&, const PXR_NS::SdfPath&) override;
    PXR_NS::SdfSpecType GetSpecType(const PXR_NS::SdfPath&) const override;
    bool Has(const PXR_NS::SdfPath&, const PXR_NS::TfToken&, PXR_NS::VtValue* = nullptr) const override;
    bool Has(const PXR_NS::SdfPath&, const PXR_NS::TfToken&, PXR_NS::SdfAbstractDataValue*) const override;
    bool HasSpecAndField(const PXR_NS::SdfPath&, const PXR_NS::TfToken&, PXR_NS::VtValue*,
        PXR_NS::SdfSpecType*) const override;
    bool HasSpecAndField(const PXR_NS::SdfPath&, const PXR_NS::TfToken&, PXR_NS::SdfAbstractDataValue*,
        PXR_NS::SdfSpecType*) const override;
    PXR_NS::VtValue Get(const PXR_NS::SdfPath&, const PXR_NS::TfToken&) const override;
    void Set(const PXR_NS::SdfPath&, const PXR_NS::TfToken&, const PXR_NS::VtValue&) override;
    void Set(const PXR_NS::SdfPath&, const PXR_NS::TfToken&, const PXR_NS::SdfAbstractDataConstValue&) override;
    void Erase(const PXR_NS::SdfPath&, const PXR_NS::TfToken&) override;
    std::vector<PXR_NS::TfToken> List(const PXR_NS::SdfPath&) const override;
    void SetDictValueByKey(const PXR_NS::SdfPath&, const PXR_NS::TfToken&, const PXR_NS::TfToken&,
        const PXR_NS::VtValue&) override;
    void SetDictValueByKey(const PXR_NS::SdfPath&, const PXR_NS::TfToken&, const PXR_NS::TfToken&,
        const PXR_NS::SdfAbstractDataConstValue&) override;
    void EraseDictValueByKey(const PXR_NS::SdfPath&, const PXR_NS::TfToken&, const PXR_NS::TfToken&) override;
    std::set<double> ListAllTimeSamples() const override;
    std::set<double> ListTimeSamplesForPath(const PXR_NS::SdfPath&) const override;
    bool GetBracketingTimeSamples(double, double*, double*) const override;
    size_t GetNumTimeSamplesForPath(const PXR_NS::SdfPath&) const override;
    bool GetBracketingTimeSamplesForPath(const PXR_NS::SdfPath&, double, double*, double*) const override;
    bool GetPreviousTimeSampleForPath(const PXR_NS::SdfPath&, double, double*) const override;
    bool QueryTimeSample(const PXR_NS::SdfPath&, double, PXR_NS::VtValue*) const override;
    bool QueryTimeSample(const PXR_NS::SdfPath&, double, PXR_NS::SdfAbstractDataValue*) const override;
    void SetTimeSample(const PXR_NS::SdfPath&, double, const PXR_NS::VtValue&) override;
    void EraseTimeSample(const PXR_NS::SdfPath&, double) override;
    // Creates the spec at `path`, or replaces the one there, holding exactly
    // the nonempty values of `fields` in their order: the content that erasing
    // the spec, creating it and setting each field would leave, in one edit of
    // the spec index.
    void PutSpec(const PXR_NS::SdfPath&, PXR_NS::SdfSpecType, const std::map<PXR_NS::TfToken, PXR_NS::VtValue>& fields);
    // The spec at `path` from one lookup: its type and its fields' names and
    // values in List order. False, leaving both unchanged, when it is absent.
    bool ReadSpec(const PXR_NS::SdfPath&, PXR_NS::SdfSpecType* type,
        std::vector<std::pair<PXR_NS::TfToken, PXR_NS::VtValue>>* fields) const;
private:
    void _VisitSpecs(PXR_NS::SdfAbstractDataSpecVisitor*) const override;
    // Borrowed map; callers must hold this data's read/write guard throughout.
    const PXR_NS::SdfTimeSampleMap* _Samples(const PXR_NS::SdfPath&) const;
    struct State;
    std::unique_ptr<State> state_;
};
} // namespace omni::usd::live
