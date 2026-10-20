// SPDX-FileCopyrightText: Copyright (c) 2018-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "SpecField.h"

#include "Spec.h"
#include "SdfIncludes.h"

#include <iomanip>
#include <stdexcept>

const VtValue SpecField::DefaultValue = VtValue{};

std::string GetFieldPath(const uint64_t specId, const TfToken& fieldName) {
    std::ostringstream stream;
    stream << "'" << specId << ":" << fieldName.GetString() << "'";
    return stream.str();
}

SpecField::SpecField(const std::weak_ptr<Spec>& spec, const uint64_t specId, const TfToken& fieldName)
    : SpecField(spec, specId, fieldName, VtValue{}) {}

SpecField::SpecField(std::weak_ptr<Spec> spec, const uint64_t specId, const TfToken& fieldName, VtValue&& value)
    : _spec(std::move(spec)), _versions()
#if !defined(_NDEBUG) && !defined(NDEBUG)
      ,
      _fieldPath(GetFieldPath(specId, fieldName))
#endif
{
    _versions.emplace_back(std::move(value), 0);
}

void swap(SpecField& a, SpecField& b) noexcept {
    using std::swap;
    auto lockA = a._spec.lock()->AcquireWriteLock("SpecField::swap()");
    auto lockB = b._spec.lock()->AcquireWriteLock("SpecField::swap()");
    swap(a._versions, b._versions);

    a._fieldPath.swap(b._fieldPath);
}

SpecField::operator const VtValue&() const {
    auto lock = _spec.lock()->AcquireReadLock("SpecField::operator const VtValue&()");
    return _versions.cbegin()->value;
}

SpecField::Version_t::value_type& SpecField::Current() {
    auto lock = _spec.lock()->AcquireReadLock("SpecField::Current()");
    return *_versions.begin();
}

const VtValue& SpecField::CurrentValue() const {
    auto lock = _spec.lock()->AcquireReadLock("SpecField::CurrentValue()");
    return _versions.cbegin()->value;
}

uint64_t SpecField::CurrentVersion() const {
    auto lock = _spec.lock()->AcquireReadLock("SpecField::CurrentVersion()");
    return _versions.cbegin()->version;
}

const VtValue& SpecField::CommittedValue() const {
    auto lock = _spec.lock()->AcquireReadLock("SpecField::CommittedValue()");
    return _versions.crbegin()->version ? _versions.crbegin()->value : DefaultValue;
}

uint64_t SpecField::CommittedVersion() const {
    auto lock = _spec.lock()->AcquireReadLock("SpecField::CommittedVersion()");
    return _versions.crbegin()->version;
}

void SpecField::EraseTimeSample(const double time, const uint64_t version) {
    InsertOrAssignTimeSample(time, VtValue{}, version);
}

void SpecField::InsertOrAssignTimeSample(const double time, const VtValue& value, const uint64_t version) {
    auto lock = _spec.lock()->AcquireWriteLock("SpecField::InsertOrAssignTimeSample()");
    // Never remove the current/committed map while preparing a candidate:
    // CommitValue can reject the candidate or retain an outstanding local edit.
    if (version != 0 && version < CommittedVersion()) return;
    SdfTimeSampleMap samples;
    const auto current = _versions.begin();
    if (current->value.IsHolding<SdfTimeSampleMap>()) samples = current->value.UncheckedGet<SdfTimeSampleMap>();
    else if (!current->value.IsEmpty()) throw std::invalid_argument("timeSamples field is not a time-sample map");
    if (value.IsEmpty()) {
        samples.erase(time);
    } else {
        samples[time] = value;
    }
    CommitValue(VtValue(samples), version);
}

bool SpecField::IsEmpty(const VtValue& value) {
    return value.IsEmpty() || (value.IsHolding<SdfTimeSampleMap>() && value.UncheckedGet<SdfTimeSampleMap>().empty());
}


void SpecField::SetValue(VtValue&& otherValue) {
    auto lock = _spec.lock()->AcquireWriteLock("SpecField::SetValue()");
    if (const auto current = _versions.begin(); current->value.IsEmpty()) {
        std::swap(current->value, otherValue);
    } else {
        _versions.emplace_front(std::move(otherValue), 0);
    }
}

void SpecField::ReplaceValue(VtValue value) {
    Version_t replacement;
    replacement.emplace_back(std::move(value), 0);
    auto lock = _spec.lock()->AcquireWriteLock("SpecField::ReplaceValue");
    _versions.swap(replacement);
}

void SpecField::CommitVersion(const uint64_t version) {
    auto lock = _spec.lock()->AcquireWriteLock("SpecField::SetValue()");
    const auto current = _versions.begin();
    current->version = version;
    _versions.erase(std::next(current), _versions.end());
}

bool SpecField::CommitValue(VtValue&& value, uint64_t version) {
    using std::swap;
    if (version == 0) {
        SetValue(std::move(value));
        return false;
    }

    auto lock = _spec.lock()->AcquireWriteLock("SpecField::CommitValue()");
    if (const auto current = _versions.begin(); current->version == 0) {
        // Time-sample updates are committed as one field value and follow this path.
        if (const auto unassigned = current->value.IsEmpty(); unassigned || value == current->value) {
            auto& entry = *current;
            _versions.erase(std::next(current), _versions.end());
            // current either has the value being committed or is unassigned
            // unassigned means it's arriving from a remote or async operation
            if (unassigned) { std::swap(entry.value, value); }

            entry.version = version;
            return unassigned;
        }

        // the value being committed is a previous version of the field
        Version_t::iterator iterator;
        for (iterator = _versions.begin(); iterator != _versions.end(); ++iterator) {
            if (iterator->version == 0) {
                if (iterator->value == value) {
                    iterator->version = version;
                    _versions.erase(std::next(iterator), _versions.end());
                    return false;
                }
            } else if (version > iterator->version) {
                // more recent committed update
                iterator->version = version;
                std::swap(iterator->value, value);
                _versions.erase(std::next(iterator), _versions.end());
                return false;
            }
        }

        if (iterator == _versions.end()) { _versions.emplace_back(std::move(value), version); }
    } else if (version >= current->version) {
        // an updated remote value has arrived
        std::swap(current->value, value);
        current->version = version;
        _versions.erase(std::next(current), _versions.end());
        return true;
    }
    return false;
}

std::string SpecField::GetString(const bool currentValueOnly) const {
    auto lock = _spec.lock()->AcquireReadLock("SpecField::GetString()");
    const auto current = _versions.begin();
    std::ostringstream stream;
    if (currentValueOnly) {
        stream << "{" << VtValueToStr(CurrentValue()) << "}";
    } else {
        stream << "base: {" << VtValueToStr(CommittedValue()) << ", " << CommittedVersion() << "} current: {"
               << VtValueToStr(CurrentValue()) << ", " << CurrentVersion() << "}";
    }

    return stream.str();
}

template<typename V> std::ostringstream& DumpVector(std::ostringstream& stream, const V& vector) {
    stream << "[";
    for (auto token = vector.begin(); token != vector.end(); ++token) {
        stream << "'" << *token << "'";
        if (token != vector.end() - 1) stream << ", ";
    }
    stream << "]";

    return stream;
}

std::string SpecField::VtValueToStr(const VtValue& value) {
    std::ostringstream stream;
    // create a copy of the value because it may be changing
    if (const auto copy = VtValue{value}; copy.GetTypeid() == typeid(TfTokenVector)) {
        DumpVector(stream, copy.UncheckedGet<TfTokenVector>());
    } else if (copy.IsHolding<VtArray<GfVec3d>>()) {
        stream << copy;
    } else if (copy.IsHolding<std::vector<SdfPath>>()) {
        DumpVector(stream, copy.UncheckedGet<std::vector<SdfPath>>());
    } else if (copy.IsHolding<VtDictionary>()) {
        stream << copy;
    } else {
        stream << "'" << copy << "'";
    }
    return stream.str();
}
