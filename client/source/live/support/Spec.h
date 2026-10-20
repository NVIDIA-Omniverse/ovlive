// SPDX-FileCopyrightText: Copyright (c) 2018-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "SpecField.h"
#include "ReaderWriterLock.h"
struct Spec;

struct Spec : public std::enable_shared_from_this<Spec>, public threading::ReaderWriterLock<Spec> {
    using Fields_t = std::unordered_map<PXR_NS::TfToken, SpecField, PXR_NS::TfToken::HashFunctor>;
    using OnAddField_t = std::function<SpecField(const PXR_NS::TfToken&)>;
    using FieldConditional_t = std::pair<SpecField&, bool>;

    explicit Spec(uint64_t specId);
    Spec(Spec&& other) = delete;
    Spec(const Spec& other) = delete;
    ~Spec() = default;

    Spec& operator=(Spec& other) = delete;
    Spec& operator=(Spec&& other) = delete;

    uint64_t SpecId() const;
    bool IsValid() const;

    static Spec& Default();

    FieldConditional_t EmplaceField(const PXR_NS::TfToken& fieldName, const OnAddField_t& onAdd);
    FieldConditional_t EmplaceField(const PXR_NS::TfToken& fieldName);
    Fields_t::size_type EraseField(const PXR_NS::TfToken& fieldName);
    const SpecField* FindField(const PXR_NS::TfToken& field);
    // Sdf exact-storage operations preserve authored empty containers and do
    // not retain version/tombstone history. Callers keep a guard across reads.
    const SpecField* FindExactField(const PXR_NS::TfToken& field) const;
    void SetExactField(const PXR_NS::TfToken& field, PXR_NS::VtValue value);
    void EraseExactField(const PXR_NS::TfToken& field);
    void VisitFields(const std::function<bool(const PXR_NS::TfToken&, const SpecField&)>& visitor) const;

    void EraseTimeSample(double time, uint64_t version);
    void InsertOrAssignTimeSample(double time, const PXR_NS::VtValue& value, uint64_t version);
    bool IsEmpty() const { return _fields.empty(); }
    void Reset();

private:
    Fields_t _fields;
    std::unordered_map<double, uint64_t> _timeSampleSaveCounts;
};
