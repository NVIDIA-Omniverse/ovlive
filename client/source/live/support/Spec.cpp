// SPDX-FileCopyrightText: Copyright (c) 2018-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "Spec.h"

#include "SpecField.h"
#include "SdfIncludes.h"

Spec& Spec::Default() {
    static auto defaultValue = std::make_shared<Spec>(0);
    return *defaultValue;
}

const SpecField* Spec::FindField(const TfToken& field) {
    auto lock = AcquireReadLock("Spec(%)::FindField field: '%'", SpecId(), field.GetString());
    const auto it = _fields.find(field);
    if (it == _fields.end() || SpecField::IsEmpty(it->second.CurrentValue())) { return nullptr; }

    return &it->second;
}

Spec::FieldConditional_t Spec::EmplaceField(const TfToken& fieldName, const OnAddField_t& onAdd) {
    auto writeLock = AcquireWriteLock("Spec(%)::EmplaceField field: '%'", SpecId(), fieldName.GetString());
    const auto [it, added] = _fields.emplace(fieldName, onAdd(fieldName));
    return std::make_pair(std::reference_wrapper<SpecField>(it->second), added);
}

const SpecField* Spec::FindExactField(const TfToken& field) const {
    auto lock = AcquireReadLock();
    const auto it = _fields.find(field);
    return it == _fields.end() || it->second.CurrentValue().IsEmpty() ? nullptr : &it->second;
}

void Spec::SetExactField(const TfToken& name, VtValue value) {
    auto lock = AcquireWriteLock();
    if (value.IsEmpty()) {
        _fields.erase(name);
        return;
    }
    if (const auto it = _fields.find(name); it != _fields.end()) it->second.ReplaceValue(std::move(value));
    else _fields.emplace(name, SpecField{weak_from_this(), SpecId(), name, std::move(value)});
}

void Spec::EraseExactField(const TfToken& field) {
    auto lock = AcquireWriteLock();
    _fields.erase(field);
}

Spec::FieldConditional_t Spec::EmplaceField(const TfToken& fieldName) {
    auto onAdd = [&](const TfToken& name) {
        // An uninitialized field is distinct from an authored empty sample map.
        // The latter represents a local edit to CommitValue's version model.
        return SpecField{weak_from_this(), this->SpecId(), name};
    };

    return EmplaceField(fieldName, onAdd);
}

Spec::Fields_t::size_type Spec::EraseField(const TfToken& fieldName) {
    auto lock = AcquireWriteLock("Spec(%)::EraseField field: '%'", SpecId(), fieldName.GetString());
    if (const auto it = _fields.find(fieldName); it != _fields.end()) {
        it->second.SetValue(fieldName == SdfFieldKeys->TimeSamples ? VtValue(SdfTimeSampleMap()) : VtValue());
    }

    return 0;
}

void Spec::EraseTimeSample(const double time, const uint64_t version) {
    InsertOrAssignTimeSample(time, VtValue{}, version);
}

void Spec::InsertOrAssignTimeSample(const double time, const VtValue& value, const uint64_t version) {
    auto lock = AcquireWriteLock("Spec(%)::InsertOrAssignTimeSample", SpecId());
    const auto [field, added] = EmplaceField(SdfFieldKeys->TimeSamples);
    if (const auto timeSampleVersion = _timeSampleSaveCounts[time]; version == 0 || timeSampleVersion <= version) {
        field.InsertOrAssignTimeSample(time, value, version);
        if (version != 0) _timeSampleSaveCounts[time] = version;
    }
}

Spec::Spec(const uint64_t specId) : threading::ReaderWriterLock<Spec>(specId) {}

uint64_t Spec::SpecId() const { return Id(); }

bool Spec::IsValid() const { return Id(); }

void Spec::Reset() {
    auto lock = AcquireWriteLock("Spec(%)::Reset", SpecId());

    _fields.clear();
    _timeSampleSaveCounts.clear();
}

void Spec::VisitFields(const std::function<bool(const TfToken&, const SpecField&)>& visitor) const {
    auto lock = AcquireReadLock("Spec(%)::VisitFields", SpecId());
    auto shouldContinue = true;
    for (auto idx = _fields.begin(); shouldContinue && idx != _fields.end(); ++idx) {
        shouldContinue = visitor(idx->first, idx->second);
    }
}
