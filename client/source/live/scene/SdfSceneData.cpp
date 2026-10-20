// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "SdfSceneData.h"
#include "AbstractValue.h"
#include "PersistentMap.h"
#include <pxr/base/tf/diagnostic.h>
#include <pxr/usd/sdf/schema.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <vector>

using namespace PXR_NS;
namespace omni::usd::live {
namespace {
void validateSamples(const VtValue& value);
}
struct SdfSceneData::State {
    // A spec's type and its fields in the order they were first set. A record
    // of at most kScanned entries is searched in order. A larger one indexes
    // its fields by name, and an erased field leaves an empty entry in place
    // until empty entries outnumber the rest, so a lookup, set or erase costs
    // constant time, amortized, however many fields the spec holds. Every
    // entry a search, compaction or indexing visits counts toward the scene
    // work meter.
    class Record {
    public:
        SdfSpecType type = SdfSpecTypeUnknown;
        void reserve(size_t count) { entries_.reserve(count); }
        void read(std::vector<std::pair<TfToken, VtValue>>& fields) const {
            fields.clear();
            fields.reserve(size());
            for (const auto& entry : entries_)
                if (!entry.second.IsEmpty()) fields.push_back(entry);
        }
        const VtValue* find(const TfToken& name) const {
            const auto at = position(name);
            return at == kAbsent ? nullptr : &entries_[at].second;
        }
        void set(const TfToken& name, VtValue value) {
            if (value.IsEmpty()) {
                erase(name);
                return;
            }
            if (const auto at = position(name); at != kAbsent) {
                entries_[at].second = std::move(value);
                return;
            }
            entries_.emplace_back(name, std::move(value));
            if (index_.empty()) {
                if (entries_.size() > kScanned) reindex();
                return;
            }
            try {
                index_.emplace(name, entries_.size() - 1);
            } catch (...) {
                entries_.pop_back();
                throw;
            }
        }
        void erase(const TfToken& name) {
            const auto at = position(name);
            if (at == kAbsent) return;
            if (index_.empty()) {
                entries_.erase(entries_.begin() + std::ptrdiff_t(at));
                return;
            }
            index_.erase(name);
            entries_[at].second = VtValue();
            if (2 * index_.size() < entries_.size()) compact();
        }
        size_t size() const { return index_.empty() ? entries_.size() : index_.size(); }
        // Calls `visit(name)` for each field in the order it was first set.
        template<class Visit> void names(Visit&& visit) const {
            for (const auto& [name, value] : entries_)
                if (!value.IsEmpty()) visit(name);
        }

    private:
        static constexpr size_t kScanned = 8, kAbsent = std::numeric_limits<size_t>::max();
        // The entry holding `name`'s value. Only an indexed record holds empty
        // entries, and its index names none of them.
        size_t position(const TfToken& name) const {
            if (!index_.empty()) {
                const auto found = index_.find(name);
                return found == index_.end() ? kAbsent : found->second;
            }
            for (size_t i = 0; i < entries_.size(); ++i) {
                sceneWork::count(1);
                if (entries_[i].first == name) return i;
            }
            return kAbsent;
        }
        // Drops the empty entries; the record is searched in order until it
        // indexes its entries again.
        void compact() {
            sceneWork::count(entries_.size());
            index_.clear();
            entries_.erase(std::remove_if(entries_.begin(), entries_.end(),
                               [](const auto& entry) { return entry.second.IsEmpty(); }),
                entries_.end());
            if (entries_.size() > kScanned) reindex();
        }
        // Indexes the entries of a record that holds no empty entry.
        void reindex() {
            sceneWork::count(entries_.size());
            std::unordered_map<TfToken, size_t, TfToken::HashFunctor> index;
            index.reserve(entries_.size());
            for (size_t i = 0; i < entries_.size(); ++i) index.emplace(entries_[i].first, i);
            index_.swap(index);
        }
        std::vector<std::pair<TfToken, VtValue>> entries_;
        std::unordered_map<TfToken, size_t, TfToken::HashFunctor> index_;
    };
    // Copies of the data share this map's structure and its unedited records.
    PersistentMap<SdfPath, Record> specs;
    void create(const SdfPath& path, SdfSpecType type) {
        if (const auto* found = specs.find(path)) {
            if (found->type != type) specs.edit(path)->type = type;
            return;
        }
        Record record;
        record.type = type;
        specs.insert(path, std::move(record));
    }
};
SdfSceneData::SdfSceneData() : state_(std::make_unique<State>()) {}
SdfSceneData::~SdfSceneData() = default;

void SdfSceneData::CreateSpec(const SdfPath& path, SdfSpecType type) {
    if (path.IsEmpty() || type <= SdfSpecTypeUnknown || type >= SdfNumSpecTypes) {
        TF_CODING_ERROR("Invalid scene spec path or type");
        return;
    }
    auto guard = AcquireWriteLock();
    state_->create(path, type);
}
void SdfSceneData::PutSpec(const SdfPath& path, SdfSpecType type, const std::map<TfToken, VtValue>& fields) {
    if (path.IsEmpty() || type <= SdfSpecTypeUnknown || type >= SdfNumSpecTypes) {
        TF_CODING_ERROR("Invalid scene spec path or type");
        return;
    }
    State::Record record;
    record.type = type;
    record.reserve(fields.size());
    for (const auto& [name, value] : fields) {
        if (name == SdfFieldKeys->TimeSamples) validateSamples(value);
        record.set(name, value);
    }
    auto guard = AcquireWriteLock();
    state_->specs.assign(path, std::move(record));
}
bool SdfSceneData::ReadSpec(const SdfPath& path, SdfSpecType* type,
    std::vector<std::pair<TfToken, VtValue>>* fields) const {
    auto guard = AcquireReadLock();
    const auto* record = state_->specs.find(path);
    if (!record) return false;
    *type = record->type;
    record->read(*fields);
    return true;
}
bool SdfSceneData::HasSpec(const SdfPath& path) const {
    auto guard = AcquireReadLock();
    return state_->specs.find(path) != nullptr;
}
SdfSpecType SdfSceneData::GetSpecType(const SdfPath& path) const {
    auto guard = AcquireReadLock();
    const auto* found = state_->specs.find(path);
    return found ? found->type : SdfSpecTypeUnknown;
}
void SdfSceneData::EraseSpec(const SdfPath& path) {
    auto guard = AcquireWriteLock();
    state_->specs.erase(path);
}
void SdfSceneData::MoveSpec(const SdfPath& oldPath, const SdfPath& newPath) {
    auto guard = AcquireWriteLock();
    if (oldPath == newPath) return;
    const auto* old = state_->specs.find(oldPath);
    if (!old || newPath.IsEmpty() || state_->specs.find(newPath)) {
        TF_CODING_ERROR("Scene spec move requires an existing source and absent destination");
        return;
    }
    // Both edits apply to a copy that replaces the index only when both
    // succeed, so a failure leaves the source intact.
    auto specs = state_->specs;
    specs.insert(newPath, *old);
    specs.erase(oldPath);
    std::swap(state_->specs, specs);
}

void SdfSceneData::_VisitSpecs(SdfAbstractDataSpecVisitor* visitor) const {
    // Visit a snapshot of the paths without holding the read lock, as stock
    // SdfData does; a visitor that reads another layer's data (TransferContent)
    // then never nests two scene locks.
    std::vector<SdfPath> paths;
    {
        auto guard = AcquireReadLock();
        paths.reserve(state_->specs.size());
        for (auto it = state_->specs.begin(); it != state_->specs.end(); ++it) paths.push_back(it.key());
    }
    for (const auto& path : paths)
        if (!visitor->VisitSpec(*this, path)) break;
}
void SdfSceneData::CopyFrom(const SdfAbstractDataConstPtr& source) {
    if (!source) {
        TF_CODING_ERROR("Cannot copy null scene data");
        return;
    }
    if (source.operator->() == this) return;
    // Another scene's data shares its structure and records: a copy in
    // constant time whose later edits copy only what they change.
    if (const auto* scene = dynamic_cast<const SdfSceneData*>(source.operator->())) {
        PersistentMap<SdfPath, State::Record> shared;
        {
            auto guard = scene->AcquireReadLock();
            shared = scene->state_->specs;
        }
        auto guard = AcquireWriteLock();
        std::swap(state_->specs, shared);
        return;
    }
    struct Snapshot : SdfAbstractDataSpecVisitor {
        struct Record {
            SdfSpecType type;
            std::map<TfToken, VtValue> fields;
        };
        std::map<SdfPath, Record> records;
        bool VisitSpec(const SdfAbstractData& src, const SdfPath& path) override {
            Record record{src.GetSpecType(path), {}};
            for (const auto& name : src.List(path)) record.fields.emplace(name, src.Get(path, name));
            records.emplace(path, std::move(record));
            return true;
        }
        void Done(const SdfAbstractData&) override {}
    } snapshot;
    // Foreign sources follow their own documented read synchronization.
    source->VisitSpecs(&snapshot);
    auto replacement = std::make_unique<State>();
    sceneWork::count(snapshot.records.size());
    for (const auto& [path, record] : snapshot.records) {
        State::Record copy;
        copy.type = record.type;
        for (const auto& [name, value] : record.fields) {
            if (name == SdfFieldKeys->TimeSamples) validateSamples(value);
            copy.set(name, value);
        }
        replacement->specs.insert(path, std::move(copy));
    }
    {
        auto guard = AcquireWriteLock();
        state_.swap(replacement);
    }
    // Retired values/destructors run outside the destination guard.
}

bool SdfSceneData::Has(const SdfPath& path, const TfToken& name, VtValue* value) const {
    auto guard = AcquireReadLock();
    const auto* record = state_->specs.find(path);
    const auto* field = record ? record->find(name) : nullptr;
    if (!field) return false;
    if (value) *value = *field;
    return true;
}
bool SdfSceneData::Has(const SdfPath& path, const TfToken& name, SdfAbstractDataValue* value) const {
    VtValue copy;
    const bool found = Has(path, name, &copy);
    return found && (!value || value->StoreValue(copy));
}
bool SdfSceneData::HasSpecAndField(const SdfPath& path, const TfToken& name, VtValue* value, SdfSpecType* type) const {
    auto guard = AcquireReadLock();
    *type = GetSpecType(path);
    return *type != SdfSpecTypeUnknown && Has(path, name, value);
}
bool SdfSceneData::HasSpecAndField(const SdfPath& path, const TfToken& name, SdfAbstractDataValue* value,
    SdfSpecType* type) const {
    VtValue copy;
    const bool found = HasSpecAndField(path, name, &copy, type);
    return found && (!value || value->StoreValue(copy));
}
VtValue SdfSceneData::Get(const SdfPath& path, const TfToken& name) const {
    VtValue value;
    Has(path, name, &value);
    return value;
}
namespace {
void validateSamples(const VtValue& value) {
    if (value.IsEmpty()) return;
    if (!value.IsHolding<SdfTimeSampleMap>()) throw std::invalid_argument("timeSamples must be a map");
    for (const auto& [time, sample] : value.UncheckedGet<SdfTimeSampleMap>()) {
        if (std::isnan(time) || sample.IsEmpty()) throw std::invalid_argument("invalid authored time sample");
    }
}
SdfTimeSampleMap sampleMap(const VtValue& value) {
    return value.IsHolding<SdfTimeSampleMap>() ? value.UncheckedGet<SdfTimeSampleMap>() : SdfTimeSampleMap{};
}
bool bracket(const std::set<double>& samples, double time, double* low, double* high) {
    if (samples.empty() || std::isnan(time)) return false;
    auto next = samples.lower_bound(time);
    if (next == samples.end()) *low = *high = *samples.rbegin();
    else if (next == samples.begin() || *next == time) *low = *high = *next;
    else {
        *high = *next;
        *low = *std::prev(next);
    }
    return true;
}
}
void SdfSceneData::Set(const SdfPath& path, const TfToken& name, const VtValue& value) {
    if (name == SdfFieldKeys->TimeSamples) validateSamples(value);
    auto guard = AcquireWriteLock();
    auto* record = state_->specs.edit(path);
    if (!record) {
        TF_CODING_ERROR("Cannot set a field on an absent spec");
        return;
    }
    record->set(name, value);
}
void SdfSceneData::Set(const SdfPath& path, const TfToken& name, const SdfAbstractDataConstValue& value) {
    VtValue copy;
    if (!extractAbstractValue(value, VtValue{}, copy)) throw std::invalid_argument("cannot extract scene field value");
    Set(path, name, copy);
}
void SdfSceneData::Erase(const SdfPath& path, const TfToken& name) {
    auto guard = AcquireWriteLock();
    const auto* record = state_->specs.find(path);
    if (record && record->find(name)) state_->specs.edit(path)->erase(name);
}
std::vector<TfToken> SdfSceneData::List(const SdfPath& path) const {
    auto guard = AcquireReadLock();
    std::vector<TfToken> result;
    if (const auto* record = state_->specs.find(path)) {
        result.reserve(record->size());
        record->names([&](const TfToken& name) { result.push_back(name); });
    }
    return result;
}
void SdfSceneData::SetDictValueByKey(const SdfPath& path, const TfToken& field, const TfToken& key,
    const VtValue& value) {
    auto guard = AcquireWriteLock();
    SdfAbstractData::SetDictValueByKey(path, field, key, value);
}
void SdfSceneData::SetDictValueByKey(const SdfPath& path, const TfToken& field, const TfToken& key,
    const SdfAbstractDataConstValue& value) {
    VtValue copy;
    if (!extractAbstractValue(value, VtValue{}, copy))
        throw std::invalid_argument("cannot extract scene dictionary value");
    SetDictValueByKey(path, field, key, copy);
}
void SdfSceneData::EraseDictValueByKey(const SdfPath& path, const TfToken& field, const TfToken& key) {
    auto guard = AcquireWriteLock();
    SdfAbstractData::EraseDictValueByKey(path, field, key);
}
std::set<double> SdfSceneData::ListTimeSamplesForPath(const SdfPath& path) const {
    auto guard = AcquireReadLock();
    std::set<double> result;
    if (const auto* samples = _Samples(path))
        for (const auto& [time, ignored] : *samples) {
            (void)ignored;
            result.insert(time);
        }
    return result;
}
const SdfTimeSampleMap* SdfSceneData::_Samples(const SdfPath& path) const {
    const auto* record = state_->specs.find(path);
    const auto* field = record ? record->find(SdfFieldKeys->TimeSamples) : nullptr;
    return field ? &field->UncheckedGet<SdfTimeSampleMap>() : nullptr;
}
std::set<double> SdfSceneData::ListAllTimeSamples() const {
    auto guard = AcquireReadLock();
    std::set<double> result;
    for (auto it = state_->specs.begin(); it != state_->specs.end(); ++it) {
        const auto times = ListTimeSamplesForPath(it.key());
        result.insert(times.begin(), times.end());
    }
    return result;
}
size_t SdfSceneData::GetNumTimeSamplesForPath(const SdfPath& path) const {
    auto guard = AcquireReadLock();
    const auto* samples = _Samples(path);
    return samples ? samples->size() : 0;
}
bool SdfSceneData::GetBracketingTimeSamples(double time, double* low, double* high) const {
    return bracket(ListAllTimeSamples(), time, low, high);
}
bool SdfSceneData::GetBracketingTimeSamplesForPath(const SdfPath& path, double time, double* low, double* high) const {
    auto guard = AcquireReadLock();
    const auto* samples = _Samples(path);
    if (!samples || samples->empty() || std::isnan(time)) return false;
    auto next = samples->lower_bound(time);
    if (next == samples->end()) *low = *high = samples->rbegin()->first;
    else if (next == samples->begin() || next->first == time) *low = *high = next->first;
    else {
        *high = next->first;
        *low = std::prev(next)->first;
    }
    return true;
}
bool SdfSceneData::GetPreviousTimeSampleForPath(const SdfPath& path, double time, double* previous) const {
    if (std::isnan(time)) return false;
    auto guard = AcquireReadLock();
    const auto* samples = _Samples(path);
    if (!samples) return false;
    const auto it = samples->lower_bound(time);
    if (it == samples->begin()) return false;
    *previous = std::prev(it)->first;
    return true;
}
bool SdfSceneData::QueryTimeSample(const SdfPath& path, double time, VtValue* value) const {
    if (std::isnan(time)) return false;
    auto guard = AcquireReadLock();
    const auto* samples = _Samples(path);
    if (!samples) return false;
    const auto it = samples->find(time);
    if (it == samples->end()) return false;
    if (value) *value = it->second;
    return true;
}
bool SdfSceneData::QueryTimeSample(const SdfPath& path, double time, SdfAbstractDataValue* value) const {
    VtValue copy;
    const bool found = QueryTimeSample(path, time, &copy);
    return found && (!value || value->StoreValue(copy));
}
void SdfSceneData::SetTimeSample(const SdfPath& path, double time, const VtValue& value) {
    if (std::isnan(time)) throw std::invalid_argument("time sample is NaN");
    if (value.IsEmpty()) {
        EraseTimeSample(path, time);
        return;
    }
    auto guard = AcquireWriteLock();
    auto samples = sampleMap(Get(path, SdfFieldKeys->TimeSamples));
    samples[time] = value;
    Set(path, SdfFieldKeys->TimeSamples, VtValue(std::move(samples)));
}
void SdfSceneData::EraseTimeSample(const SdfPath& path, double time) {
    if (std::isnan(time)) throw std::invalid_argument("time sample is NaN");
    auto guard = AcquireWriteLock();
    auto samples = sampleMap(Get(path, SdfFieldKeys->TimeSamples));
    if (!samples.erase(time)) return;
    if (samples.empty()) Erase(path, SdfFieldKeys->TimeSamples);
    else Set(path, SdfFieldKeys->TimeSamples, VtValue(std::move(samples)));
}
} // namespace omni::usd::live
