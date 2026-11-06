// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "HostSceneStorage.h"
#include "PersistentMap.h"
#include "SceneWork.h"
#include <omni/usd/live/Live.h>
#include <pxr/base/tf/diagnostic.h>
#include <pxr/base/vt/dictionary.h>
#include <pxr/usd/sdf/schema.h>
#include <algorithm>
#include <cmath>
#include <map>
#include <unordered_map>
#include <limits>
#include <stdexcept>

using namespace PXR_NS;
namespace omni::usd::live {
namespace {
// A reference to `storage` for the caller's scope.
TfRefPtr<const HostSceneStorage> pinned(const HostSceneStorage& storage) {
    sceneWork::synchronize(2);
    return TfRefPtr<const HostSceneStorage>(&storage);
}
// Host-only exclusion, not a mutex. Foreign callbacks may read, but may not
// invalidate a pending mutation or a visitor's traversal by writing recursively.
struct Active {
    const TfRefPtr<const HostSceneStorage> owner;
    std::atomic<uint64_t>& depth;
    Active(const HostSceneStorage& storage, std::atomic<uint64_t>& value, bool write = true)
        : owner(pinned(storage)), depth(value) {
        sceneWork::synchronize(2);
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
            return std::find_if(fields.begin(), fields.end(), [&](const auto& item) {
                sceneWork::count(1);
                return item.first == field;
            });
        }
        const VtValue* get(const TfToken& field) const {
            for (const auto& item : fields) {
                sceneWork::count(1);
                if (item.first == field) return &item.second;
            }
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
        if (!found) return {};
        sceneWork::synchronize(2);
        return *found;
    }
    // Keep retired ownership outside the Active scope, including on exceptions.
    // A `foreign` edit runs foreign code, which may read the storage and copy
    // its index, sharing the node that holds the spec's slot; the slot is then
    // acquired again after the edit, so the write changes this storage's own
    // node and never a copy's.
    template<class Edit> void edit(HostSceneStorage& owner, const SdfPath& path, Edit edit, bool foreign = false) {
        const auto pin = pinned(owner);
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
            sceneWork::count(retired->fields.size());
            next = std::make_shared<Record>(*retired);
            edit(*next);
            if (foreign) slot = specs.edit(path);
            *slot = std::move(next);
        }
    }
    // Edits the spec's sample map in place when no copy of the storage or
    // reader shares its record, and otherwise in a copy of the record that
    // replaces it. The map itself is copied only while a reader's value holds
    // it, so repeated edits cost the map's own operations. `mutate` returns
    // whether it changed the map. Everything the edit replaces, in `retired`
    // or here, is destroyed after the guard unwinds.
    template<class Mutate>
    void editSamples(HostSceneStorage& owner, const SdfPath& path, VtValue& retired, Mutate mutate) {
        const auto pin = pinned(owner);
        Owner shared;
        std::shared_ptr<Record> copy;
        Active guard(owner, owner.active_);
        auto* slot = specs.edit(path);
        if (!slot) {
            TF_CODING_ERROR("Cannot edit an absent host spec");
            return;
        }
        // Every record is created mutable and published as const; one that
        // only this slot owns is observable by no one else. A copy of the
        // storage on another thread may have read it before releasing its
        // owner, so the fence orders those reads before these writes, as
        // PersistentMap::edit does.
        Record* record = nullptr;
        if (slot->use_count() == 1) {
            std::atomic_thread_fence(std::memory_order_acquire);
            record = const_cast<Record*>(slot->get());
        } else {
            sceneWork::count((*slot)->fields.size());
            copy = std::make_shared<Record>(**slot);
            record = copy.get();
        }
        const auto field = record->find(SdfFieldKeys->TimeSamples);
        const bool present = field != record->fields.end();
        const bool held = present && field->second.IsHolding<SdfTimeSampleMap>();
        SdfTimeSampleMap map;
        if (held) field->second.UncheckedSwap(map);
        bool changed = false;
        try {
            changed = mutate(map);
        } catch (...) {
            if (held) field->second.UncheckedSwap(map);
            throw;
        }
        if (!changed) {
            if (held) field->second.UncheckedSwap(map);
            return;
        }
        if (map.empty()) {
            if (present) {
                retired = std::move(field->second);
                record->fields.erase(field);
            }
        } else if (held) field->second.UncheckedSwap(map);
        else if (present) retired = std::exchange(field->second, VtValue::Take(map));
        else record->fields.emplace_back(SdfFieldKeys->TimeSamples, VtValue::Take(map));
        if (copy) {
            shared = std::move(*slot);
            *slot = std::move(copy);
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
    const auto self = pinned(*this);
    const SdfAbstractDataConstRefPtr sourceOwner(source.operator->());
    sceneWork::synchronize(2);
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
        } else {
            // Each source spec's type and fields, as one edit of its record.
            struct Overlay final : SdfAbstractDataSpecVisitor {
                HostSceneStorage& target;
                explicit Overlay(HostSceneStorage& storage) : target(storage) {}
                bool VisitSpec(const SdfAbstractData& data, const SdfPath& path) override {
                    std::vector<std::pair<TfToken, VtValue>> fields;
                    for (const auto& field : data.List(path)) fields.emplace_back(field, data.Get(path, field));
                    target.OverlaySpec(path, data.GetSpecType(path), fields);
                    return true;
                }
                void Done(const SdfAbstractData&) override {}
            } overlay(*replacement);
            sourceOwner->VisitSpecs(&overlay);
        }
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
void HostSceneStorage::OverlaySpec(const SdfPath& path, SdfSpecType type,
    const std::vector<std::pair<TfToken, VtValue>>& fields) {
    if (type == SdfSpecTypeUnknown) {
        TF_CODING_ERROR("Unknown host spec type");
        return;
    }
    const auto self = pinned(*this);
    State::Owner retired;
    {
        Active guard(*this, active_);
        auto* slot = state_->specs.edit(path);
        auto next = slot ? std::make_shared<State::Record>(**slot) : std::make_shared<State::Record>();
        next->type = type;
        // Each field's position in the record, so each write is one lookup.
        std::unordered_map<TfToken, size_t, TfToken::HashFunctor> index;
        index.reserve(next->fields.size() + fields.size());
        sceneWork::count(next->fields.size());
        for (size_t i = 0; i < next->fields.size(); ++i) index.emplace(next->fields[i].first, i);
        // The positions this overlay erases; an authored empty slot it does
        // not name stays, as it does through Set.
        std::vector<bool> erased(next->fields.size());
        bool erasing = false;
        for (const auto& [field, value] : fields) {
            sceneWork::count(1);
            const auto found = index.find(field);
            if (found != index.end()) {
                next->fields[found->second].second = value;
                erased[found->second] = value.IsEmpty();
                erasing = erasing || value.IsEmpty();
            } else if (!value.IsEmpty()) {
                index.emplace(field, next->fields.size());
                next->fields.emplace_back(field, value);
                erased.push_back(false);
            }
        }
        if (erasing) {
            sceneWork::count(next->fields.size());
            size_t kept = 0;
            for (size_t i = 0; i < next->fields.size(); ++i)
                if (!erased[i]) next->fields[kept++] = std::move(next->fields[i]);
            next->fields.erase(next->fields.begin() + std::ptrdiff_t(kept), next->fields.end());
        }
        if (slot) {
            retired = std::move(*slot);
            *slot = std::move(next);
        } else state_->specs.insert(path, std::move(next));
    }
}
bool HostSceneStorage::HasSpec(const SdfPath& path) const { return state_->specs.count(path) != 0; }
SdfSpecType HostSceneStorage::GetSpecType(const SdfPath& path) const {
    const auto record = state_->record(path);
    return record ? record->type : SdfSpecTypeUnknown;
}
void HostSceneStorage::EraseSpec(const SdfPath& path) {
    const auto self = pinned(*this);
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
    const auto self = pinned(*this);
    const auto owner = state_->record(path);
    const auto value = owner ? owner->get(field) : nullptr;
    if (!value) return false;
    if (output) *output = *value;
    return true;
}
bool HostSceneStorage::Has(const SdfPath& path, const TfToken& field, SdfAbstractDataValue* output) const {
    const auto self = pinned(*this);
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
    const auto self = pinned(*this);
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
    const auto self = pinned(*this);
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
    const auto self = pinned(*this);
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
    const auto self = pinned(*this);
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
    VtValue retired, replaced;
    state_->editSamples(*this, path, retired, [&](SdfTimeSampleMap& map) {
        const auto [it, inserted] = map.try_emplace(time, value);
        if (!inserted) replaced = std::exchange(it->second, value);
        return true;
    });
}
void HostSceneStorage::EraseTimeSample(const SdfPath& path, double time) {
    if (std::isnan(time)) throw OmniUsdLiveResult(OMNI_USD_LIVE_INVALID_ARGUMENT);
    if (!HasSpec(path)) {
        Active guard(*this, active_);
        return;
    }
    VtValue retired;
    SdfTimeSampleMap::node_type erased;
    state_->editSamples(*this, path, retired, [&](SdfTimeSampleMap& map) {
        erased = map.extract(time);
        return !erased.empty();
    });
}
} // namespace omni::usd::live
