// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "HostSceneStorage.h"
#include "PersistentMap.h"
#include <omni/usd/live/Live.h>
#include <pxr/base/tf/diagnostic.h>
#include <pxr/base/vt/dictionary.h>
#include <pxr/usd/sdf/schema.h>
#include <algorithm>
#include <cmath>
#include <map>
#include <limits>
#include <stdexcept>

using namespace PXR_NS;
namespace omni::usd::live {
namespace {
// Host-only exclusion, not a mutex. Foreign callbacks may read, but may not
// invalidate a pending mutation or a visitor's traversal by writing recursively.
struct Active {
    const TfRefPtr<const HostSceneStorage> owner;
    std::atomic<uint64_t>& depth;
    Active(const HostSceneStorage& storage, std::atomic<uint64_t>& value, bool write = true)
        : owner(&storage), depth(value) {
        uint64_t prior = 0;
        if (write) {
            if (!depth.compare_exchange_strong(prior, 1)) throw OmniUsdLiveResult(OMNI_USD_LIVE_BUSY);
        } else {
            prior = depth.load();
            do {
                if (prior == std::numeric_limits<uint64_t>::max())
                    throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
            } while (!depth.compare_exchange_weak(prior, prior + 1));
        }
    }
    ~Active() { depth.fetch_sub(1); }
};
SdfTimeSampleMap samples(const VtValue& value) {
    return value.IsHolding<SdfTimeSampleMap>() ? value.UncheckedGet<SdfTimeSampleMap>() : SdfTimeSampleMap{};
}
bool bracket(const std::set<double>& times, double time, double* low, double* high) {
    if (times.empty() || std::isnan(time)) return false;
    auto next = times.lower_bound(time);
    if (next == times.end()) *low = *high = *times.rbegin();
    else if (next == times.begin() || *next == time) *low = *high = *next;
    else {
        *low = *std::prev(next);
        *high = *next;
    }
    return true;
}
}
struct HostSceneStorage::State {
    struct Record {
        SdfSpecType type = SdfSpecTypeUnknown;
        std::vector<std::pair<TfToken, VtValue>> fields;
        auto find(const TfToken& field) {
            return std::find_if(fields.begin(), fields.end(), [&](const auto& item) { return item.first == field; });
        }
        const VtValue* get(const TfToken& field) const {
            for (const auto& item : fields)
                if (item.first == field) return &item.second;
            return nullptr;
        }
        void set(const TfToken& field, const VtValue& value, bool allowEmpty = false) {
            const auto it = find(field);
            if (value.IsEmpty() && !allowEmpty) {
                if (it != fields.end()) fields.erase(it);
            } else if (it == fields.end()) fields.emplace_back(field, value);
            else it->second = value;
        }
    };
    using Owner = std::shared_ptr<const Record>;
    // Copies of the storage share this map's structure and its records.
    PersistentMap<SdfPath, Owner> specs;
    Owner record(const SdfPath& path) const {
        const auto* found = specs.find(path);
        return found ? *found : Owner{};
    }
    // Keep retired ownership outside the Active scope, including on exceptions.
    // A `foreign` edit runs foreign code, which may read the storage and copy
    // its index, sharing the node that holds the spec's slot; the slot is then
    // acquired again after the edit, so the write changes this storage's own
    // node and never a copy's.
    template<class Edit> void edit(HostSceneStorage& owner, const SdfPath& path, Edit edit, bool foreign = false) {
        const TfRefPtr<const HostSceneStorage> pin(&owner);
        Owner retired;
        std::shared_ptr<Record> next;
        {
            Active guard(owner, owner.active_);
            auto* slot = specs.edit(path);
            if (!slot) {
                TF_CODING_ERROR("Cannot edit an absent host spec");
                return;
            }
            retired = *slot;
            next = std::make_shared<Record>(*retired);
            edit(*next);
            if (foreign) slot = specs.edit(path);
            *slot = std::move(next);
        }
    }
};
HostSceneStorage::HostSceneStorage() : state_(std::make_unique<State>()) {}
HostSceneStorage::~HostSceneStorage() = default;
TfRefPtr<HostSceneStorage> HostSceneStorage::clone() const {
    auto copy = TfCreateRefPtr(new HostSceneStorage);
    Active guard(*this, active_, false);
    copy->state_->specs = state_->specs;
    return copy;
}
size_t HostSceneStorage::specCount() const { return state_->specs.size(); }
void HostSceneStorage::CopyFrom(const SdfAbstractDataConstPtr& source) {
    if (!source) throw std::invalid_argument("missing host storage source");
    if (source.operator->() == this) return;
    const TfRefPtr<const HostSceneStorage> self(this);
    const SdfAbstractDataConstRefPtr sourceOwner(source.operator->());
    // The temporary also uses synchronous plugin-owned storage during unwinding.
    const auto replacement = TfCreateRefPtr(new HostSceneStorage);
    {
        Active guard(*this, active_);
        replacement->state_->specs = state_->specs;
        // A copy overlays the source on this storage's specs and fields, as
        // stock USD's does. Into empty storage, another host storage shares
        // its immutable records and structure.
        const auto* host = dynamic_cast<const HostSceneStorage*>(source.operator->());
        if (host && state_->specs.empty()) {
            Active reading(*host, host->active_, false);
            replacement->state_->specs = host->state_->specs;
        } else replacement->SdfAbstractData::CopyFrom(sourceOwner);
        state_.swap(replacement->state_);
    }
}
void HostSceneStorage::CreateSpec(const SdfPath& path, SdfSpecType type) {
    if (type == SdfSpecTypeUnknown) {
        TF_CODING_ERROR("Unknown host spec type");
        return;
    }
    if (HasSpec(path)) {
        state_->edit(*this, path, [&](auto& next) { next.type = type; });
    } else {
        Active guard(*this, active_);
        auto next = std::make_shared<State::Record>();
        next->type = type;
        state_->specs.insert(path, std::move(next));
    }
}
bool HostSceneStorage::HasSpec(const SdfPath& path) const { return state_->specs.count(path) != 0; }
SdfSpecType HostSceneStorage::GetSpecType(const SdfPath& path) const {
    const auto record = state_->record(path);
    return record ? record->type : SdfSpecTypeUnknown;
}
void HostSceneStorage::EraseSpec(const SdfPath& path) {
    const TfRefPtr<const HostSceneStorage> self(this);
    State::Owner retired;
    {
        Active guard(*this, active_);
        if (!HasSpec(path)) {
            TF_CODING_ERROR("Cannot erase an absent host spec");
            return;
        }
        retired = state_->record(path);
        state_->specs.erase(path);
    }
}
void HostSceneStorage::MoveSpec(const SdfPath& path, const SdfPath& destination) {
    Active guard(*this, active_);
    const auto record = state_->record(path);
    if (!record || state_->specs.count(destination)) {
        TF_CODING_ERROR("Host spec move requires an existing source and absent destination");
        return;
    }
    // Both edits apply to a copy that replaces the index only when both
    // succeed. The destination retains the exact immutable record.
    auto specs = state_->specs;
    specs.insert(destination, record);
    specs.erase(path);
    std::swap(state_->specs, specs);
}
void HostSceneStorage::_VisitSpecs(SdfAbstractDataSpecVisitor* visitor) const {
    if (!visitor) throw std::invalid_argument("missing host storage visitor");
    Active guard(*this, active_, false);
    for (auto it = state_->specs.begin(); it != state_->specs.end(); ++it)
        if (!visitor->VisitSpec(*this, it.key())) break;
}
bool HostSceneStorage::Has(const SdfPath& path, const TfToken& field, VtValue* output) const {
    const TfRefPtr<const HostSceneStorage> self(this);
    const auto owner = state_->record(path);
    const auto value = owner ? owner->get(field) : nullptr;
    if (!value) return false;
    if (output) *output = *value;
    return true;
}
bool HostSceneStorage::Has(const SdfPath& path, const TfToken& field, SdfAbstractDataValue* output) const {
    const TfRefPtr<const HostSceneStorage> self(this);
    const auto owner = state_->record(path);
    const auto value = owner ? owner->get(field) : nullptr;
    if (!value) return false;
    Active guard(*this, active_, false);
    return !output || output->StoreValue(*value);
}
bool HostSceneStorage::HasSpecAndField(const SdfPath& path, const TfToken& field, VtValue* output,
    SdfSpecType* type) const {
    *type = GetSpecType(path);
    return Has(path, field, output);
}
bool HostSceneStorage::HasSpecAndField(const SdfPath& path, const TfToken& field, SdfAbstractDataValue* output,
    SdfSpecType* type) const {
    *type = GetSpecType(path);
    return Has(path, field, output);
}
VtValue HostSceneStorage::Get(const SdfPath& path, const TfToken& field) const {
    VtValue value;
    Has(path, field, &value);
    return value;
}
void HostSceneStorage::Set(const SdfPath& path, const TfToken& field, const VtValue& value) {
    if (value.IsEmpty()) {
        Erase(path, field);
        return;
    }
    state_->edit(*this, path, [&](auto& next) { next.set(field, value); });
}
void HostSceneStorage::Set(const SdfPath& path, const TfToken& field, const SdfAbstractDataConstValue& value) {
    const TfRefPtr<const HostSceneStorage> self(this);
    VtValue copy; // Failed foreign output retires after the edit guard unwinds.
    state_->edit(*this, path, [&](auto& next) {
        const auto previous = next.get(field);
        if (!extractAbstractValue(value, previous ? *previous : VtValue{}, copy))
            throw OmniUsdLiveResult(OMNI_USD_LIVE_UNSUPPORTED);
        // Unlike Set(VtValue), the stock abstract-value overload retains an
        // authored empty slot. Failed extraction here preserves prior contents.
        next.set(field, copy, true);
    }, true);
}
void HostSceneStorage::Erase(const SdfPath& path, const TfToken& field) {
    if (!HasSpec(path)) {
        Active guard(*this, active_);
        return;
    }
    state_->edit(*this, path, [&](auto& next) {
        const auto it = next.find(field);
        if (it != next.fields.end()) next.fields.erase(it);
    });
}
bool HostSceneStorage::ReadSpec(const SdfPath& path, SdfSpecType* type,
    std::vector<std::pair<TfToken, VtValue>>* fields) const {
    const auto owner = state_->record(path);
    if (!owner) return false;
    *type = owner->type;
    *fields = owner->fields;
    return true;
}
std::vector<TfToken> HostSceneStorage::List(const SdfPath& path) const {
    const auto owner = state_->record(path);
    std::vector<TfToken> result;
    if (owner)
        for (const auto& item : owner->fields) result.push_back(item.first);
    return result;
}
void HostSceneStorage::SetDictValueByKey(const SdfPath& path, const TfToken& field, const TfToken& key,
    const VtValue& value) {
    if (value.IsEmpty()) {
        EraseDictValueByKey(path, field, key);
        return;
    }
    state_->edit(*this, path, [&](auto& next) {
        const auto previous = next.get(field);
        VtDictionary dict = previous && previous->template IsHolding<VtDictionary>() ?
            previous->template UncheckedGet<VtDictionary>() :
            VtDictionary{};
        dict.SetValueAtPath(key, value);
        next.set(field, VtValue(std::move(dict)));
    });
}
void HostSceneStorage::SetDictValueByKey(const SdfPath& path, const TfToken& field, const TfToken& key,
    const SdfAbstractDataConstValue& value) {
    const TfRefPtr<const HostSceneStorage> self(this);
    VtValue copy;
    {
        Active guard(*this, active_);
        if (!extractAbstractValue(value, VtValue{}, copy)) throw OmniUsdLiveResult(OMNI_USD_LIVE_UNSUPPORTED);
    }
    SetDictValueByKey(path, field, key, copy);
}
void HostSceneStorage::EraseDictValueByKey(const SdfPath& path, const TfToken& field, const TfToken& key) {
    if (!HasSpec(path)) {
        Active guard(*this, active_);
        return;
    }
    state_->edit(*this, path, [&](auto& next) {
        const auto previous = next.get(field);
        if (!previous || !previous->template IsHolding<VtDictionary>()) return;
        auto dict = previous->template UncheckedGet<VtDictionary>();
        dict.EraseValueAtPath(key);
        next.set(field, dict.empty() ? VtValue{} : VtValue(std::move(dict)));
    });
}
std::set<double> HostSceneStorage::ListTimeSamplesForPath(const SdfPath& path) const {
    const auto value = Get(path, SdfFieldKeys->TimeSamples);
    std::set<double> times;
    if (value.IsHolding<SdfTimeSampleMap>())
        for (const auto& item : value.UncheckedGet<SdfTimeSampleMap>()) times.insert(item.first);
    return times;
}
std::set<double> HostSceneStorage::ListAllTimeSamples() const {
    const TfRefPtr<const HostSceneStorage> self(this);
    std::set<double> times;
    for (auto it = state_->specs.begin(); it != state_->specs.end(); ++it) {
        auto pathTimes = ListTimeSamplesForPath(it.key());
        times.insert(pathTimes.begin(), pathTimes.end());
    }
    return times;
}
size_t HostSceneStorage::GetNumTimeSamplesForPath(const SdfPath& path) const {
    const auto value = Get(path, SdfFieldKeys->TimeSamples);
    return value.IsHolding<SdfTimeSampleMap>() ? value.UncheckedGet<SdfTimeSampleMap>().size() : 0;
}
bool HostSceneStorage::GetBracketingTimeSamples(double time, double* low, double* high) const {
    return bracket(ListAllTimeSamples(), time, low, high);
}
bool HostSceneStorage::GetBracketingTimeSamplesForPath(const SdfPath& path, double time, double* low,
    double* high) const {
    return bracket(ListTimeSamplesForPath(path), time, low, high);
}
bool HostSceneStorage::GetPreviousTimeSampleForPath(const SdfPath& path, double time, double* previous) const {
    if (std::isnan(time)) return false;
    const auto times = ListTimeSamplesForPath(path);
    auto next = times.lower_bound(time);
    if (next == times.begin()) return false;
    *previous = *std::prev(next);
    return true;
}
bool HostSceneStorage::QueryTimeSample(const SdfPath& path, double time, VtValue* output) const {
    if (std::isnan(time)) return false;
    const auto value = Get(path, SdfFieldKeys->TimeSamples);
    if (!value.IsHolding<SdfTimeSampleMap>()) return false;
    const auto& map = value.UncheckedGet<SdfTimeSampleMap>();
    const auto it = map.find(time);
    if (it == map.end()) return false;
    if (output) *output = it->second;
    return true;
}
bool HostSceneStorage::QueryTimeSample(const SdfPath& path, double time, SdfAbstractDataValue* output) const {
    const TfRefPtr<const HostSceneStorage> self(this);
    VtValue value;
    if (!QueryTimeSample(path, time, &value)) return false;
    Active guard(*this, active_, false);
    return !output || output->StoreValue(value);
}
void HostSceneStorage::SetTimeSample(const SdfPath& path, double time, const VtValue& value) {
    if (std::isnan(time)) throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
    if (value.IsEmpty()) {
        EraseTimeSample(path, time);
        return;
    }
    state_->edit(*this, path, [&](auto& next) {
        const auto previous = next.get(SdfFieldKeys->TimeSamples);
        auto map = samples(previous ? *previous : VtValue{});
        map[time] = value;
        next.set(SdfFieldKeys->TimeSamples, VtValue(std::move(map)));
    });
}
void HostSceneStorage::EraseTimeSample(const SdfPath& path, double time) {
    if (std::isnan(time)) throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
    if (!HasSpec(path)) {
        Active guard(*this, active_);
        return;
    }
    state_->edit(*this, path, [&](auto& next) {
        const auto previous = next.get(SdfFieldKeys->TimeSamples);
        auto map = samples(previous ? *previous : VtValue{});
        if (!map.erase(time)) return;
        next.set(SdfFieldKeys->TimeSamples, map.empty() ? VtValue{} : VtValue(std::move(map)));
    });
}
} // namespace omni::usd::live
