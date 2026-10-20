// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "AbstractValue.h"
#include <pxr/usd/sdf/abstractData.h>
#include <atomic>
#include <memory>
#include <utility>
#include <vector>

namespace omni::usd::live {
// Host-confined mutable storage with immutable per-spec records. Replaced records
// retire synchronously after mutation guards unwind; no SDK destruction task is
// scheduled. This does not make arbitrary custom VtValue destructors safe.
class HostSceneStorage final : public PXR_NS::SdfAbstractData {
public:
    HostSceneStorage();
    ~HostSceneStorage() override;
    bool StreamsData() const override { return false; }
    bool IsDetached() const override { return true; }
    // A copy that shares this storage's immutable spec records and the
    // structure of its spec index: constant time. Later edits of either copy
    // copy only what they change. CopyFrom of another host storage into empty
    // storage shares the same way.
    PXR_NS::TfRefPtr<HostSceneStorage> clone() const;
    size_t specCount() const;
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
    // The spec at `path` from one lookup: its type and its fields' names and
    // values in List order. False, leaving both unchanged, when it is absent.
    bool ReadSpec(const PXR_NS::SdfPath&, PXR_NS::SdfSpecType* type,
        std::vector<std::pair<PXR_NS::TfToken, PXR_NS::VtValue>>* fields) const;
private:
    void _VisitSpecs(PXR_NS::SdfAbstractDataSpecVisitor*) const override;
    struct State;
    std::unique_ptr<State> state_;
    // Frozen owners allow concurrent const traversal. Mutable owners remain
    // host-confined for writes: this counter is reentry accounting, not a
    // reader/writer lock.
    mutable std::atomic<uint64_t> active_{0};
};
} // namespace omni::usd::live
