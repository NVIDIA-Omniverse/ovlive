// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "HostSceneProjection.h"
#include <pxr/base/tf/diagnostic.h>
#include <pxr/base/tf/type.h>
#include <pxr/base/vt/dictionary.h>
#include <pxr/usd/sdf/assetPath.h>
#include <pxr/usd/sdf/listOp.h>
#include <pxr/usd/sdf/payload.h>
#include <pxr/usd/sdf/reference.h>
#include <pxr/usd/sdf/types.h>
#include <algorithm>
#include <new>
#include <stdexcept>
#include <string>

using namespace PXR_NS;
namespace omni::usd::live {
namespace {
OmniUsdLiveResult currentFailure() noexcept {
    try {
        throw;
    } catch (OmniUsdLiveResult result) { return result; } catch (const LayerOperationRejected& rejected) {
        return rejected.result();
    } catch (const std::bad_alloc&) { return OMNI_USD_LIVE_RESOURCE_EXHAUSTED; } catch (const std::length_error&) {
        return OMNI_USD_LIVE_RESOURCE_EXHAUSTED;
    } catch (...) { return OMNI_USD_LIVE_INTERNAL_ERROR; }
}
// A storage throw before any change reaches the host as a typed rejection
// (design section 8, failure containment: a mutator that throws in storage
// removes its record, and a rejected operation reports through its call and a
// USD runtime error). Allocation failures and unknown exceptions propagate unchanged.
[[noreturn]] void rethrowRejected(const char* what) {
    try {
        throw;
    } catch (OmniUsdLiveResult result) {
        TF_RUNTIME_ERROR("Live layer %s rejected before mutation (result=%u)", what, unsigned(result));
        throw LayerOperationRejected(result);
    }
}
}
namespace {
// The value estimate saturates at the largest size_t, so it never wraps and
// never decreases as a value grows.
constexpr size_t kSaturated = std::numeric_limits<size_t>::max();
size_t sum(size_t a, size_t b) noexcept { return a > kSaturated - b ? kSaturated : a + b; }
size_t product(size_t a, size_t b) noexcept { return b && a > kSaturated / b ? kSaturated : a * b; }
// A map entry's node: its links, key and value storage.
constexpr size_t kNodeBytes = 64;
// A nested value deeper than this charges nothing more.
constexpr unsigned kMaxDepth = 64;
size_t valueBytes(const VtValue& value, unsigned depth) noexcept;
// The heap bytes of one item beyond its own storage: string characters, a
// reference's custom data and an unregistered value's value. Other items
// are fixed-size or interned by USD.
size_t itemBytes(const std::string& item, unsigned) noexcept { return item.size(); }
size_t itemBytes(const SdfAssetPath& item, unsigned) noexcept {
    return sum(item.GetAssetPath().size(), item.GetResolvedPath().size());
}
size_t itemBytes(const SdfReference& item, unsigned depth) noexcept {
    return sum(item.GetAssetPath().size(),
        depth < kMaxDepth ? valueBytes(VtValue(item.GetCustomData()), depth + 1) : 0);
}
size_t itemBytes(const SdfPayload& item, unsigned) noexcept { return item.GetAssetPath().size(); }
size_t itemBytes(const SdfUnregisteredValue& item, unsigned depth) noexcept {
    return depth < kMaxDepth ? valueBytes(item.GetValue(), depth + 1) : 0;
}
template<class T> size_t itemBytes(const T&, unsigned) noexcept { return 0; }
// Each item at its type's size and at least 16 bytes, plus its heap bytes.
template<class Items> size_t itemsBytes(const Items& items, unsigned depth) noexcept {
    using Item = typename Items::value_type;
    size_t total = product(items.size(), std::max<size_t>(16, sizeof(Item)));
    for (const auto& item : items) total = sum(total, itemBytes(item, depth));
    return total;
}
// Every item of a list operation's six lists.
template<class T> size_t listOpBytes(const SdfListOp<T>& op, unsigned depth) noexcept {
    size_t total = 0;
    for (const auto* items : {&op.GetExplicitItems(), &op.GetAddedItems(), &op.GetPrependedItems(),
             &op.GetAppendedItems(), &op.GetDeletedItems(), &op.GetOrderedItems()})
        total = sum(total, itemsBytes(*items, depth));
    return total;
}
// The bytes of `value` if it holds a vector or a list operation of T.
template<class T> bool compositeBytes(const VtValue& value, unsigned depth, size_t& bytes) noexcept {
    if (value.IsHolding<std::vector<T>>()) bytes = itemsBytes(value.UncheckedGet<std::vector<T>>(), depth);
    else if (value.IsHolding<SdfListOp<T>>()) bytes = listOpBytes(value.UncheckedGet<SdfListOp<T>>(), depth);
    else return false;
    return true;
}
size_t valueBytes(const VtValue& value, unsigned depth) noexcept {
    try {
        if (value.IsEmpty()) return 0;
        if (value.IsHolding<std::string>()) return value.UncheckedGet<std::string>().size();
        if (value.IsHolding<SdfAssetPath>()) return itemBytes(value.UncheckedGet<SdfAssetPath>(), depth);
        if (value.IsHolding<SdfPayload>()) return itemBytes(value.UncheckedGet<SdfPayload>(), depth);
        if (value.IsHolding<SdfUnregisteredValue>())
            return itemBytes(value.UncheckedGet<SdfUnregisteredValue>(), depth);
        if (value.IsHolding<VtDictionary>()) {
            // Each entry is a map node holding a key string and a value.
            size_t total = 0;
            for (const auto& [key, entry] : value.UncheckedGet<VtDictionary>())
                total = sum(total, sum(kNodeBytes + key.size(), depth < kMaxDepth ? valueBytes(entry, depth + 1) : 0));
            return total;
        }
        if (value.IsHolding<SdfTimeSampleMap>()) {
            // Each sample is a map node holding its time and value.
            size_t total = 0;
            for (const auto& [time, sample] : value.UncheckedGet<SdfTimeSampleMap>())
                total = sum(total, sum(kNodeBytes, depth < kMaxDepth ? valueBytes(sample, depth + 1) : 0));
            return total;
        }
        if (value.IsHolding<SdfVariantSelectionMap>()) {
            size_t total = 0;
            for (const auto& [set, selection] : value.UncheckedGet<SdfVariantSelectionMap>())
                total = sum(total, kNodeBytes + set.size() + selection.size());
            return total;
        }
        size_t bytes = 0;
        if (compositeBytes<TfToken>(value, depth, bytes) || compositeBytes<SdfPath>(value, depth, bytes) ||
            compositeBytes<std::string>(value, depth, bytes) || compositeBytes<SdfReference>(value, depth, bytes) ||
            compositeBytes<SdfPayload>(value, depth, bytes) || compositeBytes<int>(value, depth, bytes) ||
            compositeBytes<int64_t>(value, depth, bytes) || compositeBytes<unsigned int>(value, depth, bytes) ||
            compositeBytes<uint64_t>(value, depth, bytes) ||
            compositeBytes<SdfUnregisteredValue>(value, depth, bytes) || compositeBytes<double>(value, depth, bytes) ||
            compositeBytes<SdfLayerOffset>(value, depth, bytes))
            return bytes;
        if (value.IsHolding<VtArray<std::string>>())
            return itemsBytes(value.UncheckedGet<VtArray<std::string>>(), depth);
        if (value.IsHolding<VtArray<SdfAssetPath>>())
            return itemsBytes(value.UncheckedGet<VtArray<SdfAssetPath>>(), depth);
        if (value.IsArrayValued()) {
            const auto type = TfType::Find(value.GetElementTypeid());
            return product(value.GetArraySize(), std::max<size_t>(16, type.IsUnknown() ? 0 : type.GetSizeof()));
        }
        return 0;
    } catch (...) { return 0; }
}
}
size_t hostEditRecordBytes(const SdfOperation& operation) noexcept {
    const auto value = valueBytes(operation.value, 0);
    constexpr size_t record = sizeof(SdfOperation) + 2 * sizeof(size_t);
    return value > std::numeric_limits<size_t>::max() - record ? std::numeric_limits<size_t>::max() : record + value;
}
LayerOperationRejected::LayerOperationRejected(OmniUsdLiveResult result)
    : std::runtime_error(
          "live layer operation rejected before mutation (result=" + std::to_string(unsigned(result)) + ")"),
      result_(result) {}
HostSceneProjection::HostSceneProjection(WireScenePtr baseline, std::shared_ptr<HostFailureLatch> fault, Limits limits)
    : baseline_(std::move(baseline)), fault_(std::move(fault)), data_(TfCreateRefPtr(new HostSceneStorage)),
      limits_(limits) {
    if (!baseline_ || !baseline_->data() || !fault_)
        throw std::invalid_argument("missing host projection baseline or fault owner");
    if (limits_.maxPendingBytes == 0) throw std::invalid_argument("host edit buffer quota must be positive");
    // Copy on preparation, never lazily after a USD mutation has already begun.
    // This only touches the private owned container and cannot latch a fault.
    data_->CopyFrom(baseline_->data());
    // Everything constructed here happens before any forwarded call.
    LIVE_TSAN_RELEASE(this);
}
HostSceneProjection::~HostSceneProjection() {
    // Each forwarded call decrements the depth after its last read of this
    // projection and its storage, and holds a reference to the projection, so
    // every call has retired when this runs. TfRefPtr's deleting decrement only
    // releases, and USD (often on TBB workers) orders the last decrement
    // through synchronization outside this code, so this acquire is what
    // orders those reads before the members, and the storage records shared
    // with no other owner, are freed. reseat() gets the same edge from its
    // depth load.
    (void)forwardDepth_.load(std::memory_order_acquire);
}
SdfAbstractDataConstRefPtr HostSceneProjection::capture() const {
    DataCall call(*this);
    const auto snapshot = TfCreateRefPtr(new HostSceneStorage);
    snapshot->CopyFrom(call.data);
    return snapshot;
}
void HostSceneProjection::CopyFrom(const SdfAbstractDataConstPtr&) {
    // Whole-content replacement of a live layer is unsupported (design section 8).
    // Reject before any storage change or source read; content, records and
    // the host fault stay unchanged.
    TF_RUNTIME_ERROR("Live layer content replacement is unsupported (result=%u)", unsigned(OMNI_USD_LIVE_UNSUPPORTED));
    throw LayerOperationRejected(OMNI_USD_LIVE_UNSUPPORTED);
}
void HostSceneProjection::setLocalEditSink(std::function<void(size_t)> onRecorded) {
    onRecorded_ = std::move(onRecorded);
}
void HostSceneProjection::setQuotaSink(std::function<bool(size_t)> onQuota) { onQuota_ = std::move(onQuota); }
void HostSceneProjection::setBudgetSink(std::function<HostEditBudget(size_t)> onBudget) {
    onBudget_ = std::move(onBudget);
}
HostSceneProjection::Reseat HostSceneProjection::prepareReseat(WireScenePtr next,
    const WireSceneDifference& difference) const {
    if (!next || !next->data()) throw std::invalid_argument("missing host projection baseline");
    DataCall call(*this);
    Reseat prepared{std::move(next), call.data->clone()};
    auto& storage = *prepared.storage;
    const auto& source = *prepared.scene->data();
    for (const auto& path : difference.erased)
        if (storage.HasSpec(path)) storage.EraseSpec(path);
    for (const auto& path : difference.written) {
        if (storage.HasSpec(path)) storage.EraseSpec(path);
        storage.CreateSpec(path, source.GetSpecType(path));
        for (const auto& field : source.List(path)) storage.Set(path, field, source.Get(path, field));
    }
    for (const auto& field : difference.fields)
        if (!field.rewritten) storage.Set(field.path, field.field, field.after);
    return prepared;
}
bool HostSceneProjection::keepsBaseline(const WireScenePtr& next, const WireSceneDifference& difference) const {
    return next == baseline_ || (difference.empty() && next && next->sourceFormat() == baseline_->sourceFormat());
}
void HostSceneProjection::reseat(Reseat&& prepared) {
    if (!prepared.scene || !prepared.storage) throw std::logic_error("empty host projection reseat");
    if (!pending_.empty()) throw std::logic_error("host projection reseat over buffered edits");
    if (forwardDepth_.load() != 0) throw std::logic_error("host projection reseat inside a forwarded call");
    baseline_ = std::move(prepared.scene);
    data_.swap(prepared.storage);
    // Readers reach the projection through synchronization ThreadSanitizer
    // cannot see; published() acquires this release, as it does the
    // constructor's.
    LIVE_TSAN_RELEASE(this);
}
HostSceneProjection::LocalEdits HostSceneProjection::pendingLocalEdits() const {
    LocalEdits edits;
    edits.data = capture();
    edits.operations = std::make_shared<const std::vector<SdfOperation>>(pending_);
    edits.base = baseline_;
    return edits;
}
void HostSceneProjection::acceptLocalEdits(size_t count, WireScenePtr next) {
    if (count > pending_.size() || !next) throw std::logic_error("accepted host edits exceed the buffer");
    size_t accepted = 0;
    for (size_t i = 0; i < count; ++i) accepted += hostEditRecordBytes(pending_[i]);
    pending_.erase(pending_.begin(), pending_.begin() + static_cast<std::ptrdiff_t>(count));
    pendingBytes_ -= std::min(pendingBytes_, accepted);
    if (pending_.empty()) pendingBytes_ = 0;
    baseline_ = std::move(next);
}
void HostSceneProjection::discardLocalEdits() noexcept {
    discarding_ = true;
    std::vector<SdfOperation>().swap(pending_);
    pendingBytes_ = 0;
}
bool HostSceneProjection::appendRecord(SdfOperation&& operation) {
    // A discarding projection belongs to a FAILED layer: the edit applies,
    // its owner counts it, and no record is kept.
    if (discarding_) {
        turnOpen_ = true;
        return false;
    }
    // A FAILED layer's edits always apply, so a latched host fault disables
    // the quota; the sink decides for a layer its owner has failed.
    if (!turnOpen_ && fault_->result().load(std::memory_order_acquire) == OMNI_USD_LIVE_OK) {
        bool full = pendingBytes_ >= limits_.maxPendingBytes;
        HostEditBudget budget;
        if (!full && onBudget_) {
            try {
                budget = onBudget_(pendingBytes_);
                full = budget.full;
            } catch (...) {
                fault_->fail(OMNI_USD_LIVE_INTERNAL_ERROR);
                full = true;
            }
        }
        if (full) {
            bool refuse = true;
            auto cause = OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
            if (onQuota_ && fault_->result().load(std::memory_order_acquire) == OMNI_USD_LIVE_OK) {
                try {
                    refuse = onQuota_(pending_.size());
                } catch (...) {
                    fault_->fail(OMNI_USD_LIVE_INTERNAL_ERROR);
                    refuse = true;
                }
            }
            if (refuse) {
                // The rejection reports the cause the layer latched, if any: a
                // throwing sink, or one that failed an off-host mutation.
                if (const auto latched = fault_->result().load(std::memory_order_acquire); latched != OMNI_USD_LIVE_OK)
                    cause = latched;
                // The error names the bound the buffer reached.
                if (budget.full)
                    TF_RUNTIME_ERROR("Live layer host edit buffer is full: the endpoint's buffered host edits hold "
                                     "%llu bytes of max_pending_bytes_per_endpoint (%llu) (result=%u)",
                        (unsigned long long)budget.charged, (unsigned long long)budget.limit, unsigned(cause));
                else if (pendingBytes_ >= limits_.maxPendingBytes)
                    TF_RUNTIME_ERROR("Live layer host edit buffer is full: its %zu buffered bytes reach the layer's "
                                     "quota of %zu bytes (result=%u)",
                        pendingBytes_, limits_.maxPendingBytes, unsigned(cause));
                else TF_RUNTIME_ERROR("Live layer host edit buffer refused a host turn (result=%u)", unsigned(cause));
                throw LayerOperationRejected(cause);
            }
        }
    }
    // Append before the storage edit, so an edit that reaches storage always
    // has its record. USD does not mark the record's move noexcept (TfRefPtr),
    // so this append may throw; it does so before storage changes.
    pending_.push_back(std::move(operation));
    turnOpen_ = true;
    return true;
}
void HostSceneProjection::rejectNestedMutation() const {
    // Storage also rejects a mutation nested in one of its own calls. The
    // projection rejects one from its own record path (abstract-value
    // extraction) too, so the record appended last is always this call's own.
    // The depth is loaded once: this refuses a mutation that begins while a
    // forwarded call is in progress and detects no other overlap.
    if (forwardDepth_.load() != 0) throw LayerOperationRejected(OMNI_USD_LIVE_BUSY);
}
template<class Edit> void HostSceneProjection::forward(SdfOperation&& operation, Edit&& edit) {
    rejectNestedMutation();
    // The call pins this projection and its storage through notification.
    DataCall call(*this);
    const bool appended = appendRecord(std::move(operation));
    try {
        edit(*call.data);
    } catch (OmniUsdLiveResult) {
        if (appended) pending_.pop_back();
        rethrowRejected("host edit");
    } catch (...) {
        if (appended) pending_.pop_back();
        throw;
    }
    notifyRecorded(appended);
}
void HostSceneProjection::notifyRecorded(bool appended) noexcept {
    if (appended) {
        const auto bytes = hostEditRecordBytes(pending_.back());
        pendingBytes_ = bytes > std::numeric_limits<size_t>::max() - pendingBytes_ ?
            std::numeric_limits<size_t>::max() :
            pendingBytes_ + bytes;
    }
    if (!onRecorded_) return;
    try {
        onRecorded_(pending_.size());
    } catch (...) { fault_->fail(OMNI_USD_LIVE_INTERNAL_ERROR); }
}
void HostSceneProjection::CreateSpec(const SdfPath& path, SdfSpecType type) {
    if (announcing_) return;
    SdfOperation operation(SdfOperationKind::CreateSpec);
    operation.path = path;
    operation.specType = type;
    forward(std::move(operation), [&](HostSceneStorage& data) { data.CreateSpec(path, type); });
}
void HostSceneProjection::EraseSpec(const SdfPath& path) {
    if (announcing_) return;
    SdfOperation operation(SdfOperationKind::EraseSpec);
    operation.path = path;
    forward(std::move(operation), [&](HostSceneStorage& data) { data.EraseSpec(path); });
}
void HostSceneProjection::MoveSpec(const SdfPath& path, const SdfPath& destination) {
    if (announcing_) return;
    SdfOperation operation(SdfOperationKind::MoveSpec);
    operation.path = path;
    operation.destination = destination;
    forward(std::move(operation), [&](HostSceneStorage& data) { data.MoveSpec(path, destination); });
}
void HostSceneProjection::Set(const SdfPath& path, const TfToken& field, const VtValue& value) {
    if (announcing_) return;
    SdfOperation operation(SdfOperationKind::SetField);
    operation.path = path;
    operation.field = field;
    operation.value = value;
    forward(std::move(operation), [&](HostSceneStorage& data) { data.Set(path, field, value); });
}
void HostSceneProjection::Set(const SdfPath& path, const TfToken& field, const SdfAbstractDataConstValue& value) {
    if (announcing_) return;
    SdfOperation operation(SdfOperationKind::SetField);
    operation.path = path;
    operation.field = field;
    rejectNestedMutation();
    DataCall call(*this);
    const bool appended = appendRecord(std::move(operation));
    try {
        call.data->Set(path, field, value);
    } catch (OmniUsdLiveResult) {
        if (appended) pending_.pop_back();
        rethrowRejected("host edit");
    } catch (...) {
        if (appended) pending_.pop_back();
        throw;
    }
    // A discarding projection keeps no record, so it extracts nothing.
    if (!appended) {
        notifyRecorded(false);
        return;
    }
    // Storage rejects an unextractable value. A value whose second extraction
    // fails or throws is applied but unrecordable, so it fails the layer instead.
    bool extracted = false;
    try {
        extracted = extractAbstractValue(value, VtValue{}, pending_.back().value);
    } catch (...) {
        pending_.pop_back();
        fault_->fail(currentFailure());
        throw;
    }
    if (!extracted) {
        pending_.pop_back();
        fault_->fail(OMNI_USD_LIVE_UNSUPPORTED);
        return;
    }
    notifyRecorded(true);
}
void HostSceneProjection::Erase(const SdfPath& path, const TfToken& field) {
    if (announcing_) return;
    SdfOperation operation(SdfOperationKind::EraseField);
    operation.path = path;
    operation.field = field;
    forward(std::move(operation), [&](HostSceneStorage& data) { data.Erase(path, field); });
}
void HostSceneProjection::SetDictValueByKey(const SdfPath& path, const TfToken& field, const TfToken& key,
    const VtValue& value) {
    if (announcing_) return;
    SdfOperation operation(SdfOperationKind::SetDictKey);
    operation.path = path;
    operation.field = field;
    operation.key = key;
    operation.value = value;
    forward(std::move(operation), [&](HostSceneStorage& data) { data.SetDictValueByKey(path, field, key, value); });
}
void HostSceneProjection::SetDictValueByKey(const SdfPath& path, const TfToken& field, const TfToken& key,
    const SdfAbstractDataConstValue& value) {
    if (announcing_) return;
    SdfOperation operation(SdfOperationKind::SetDictKey);
    operation.path = path;
    operation.field = field;
    operation.key = key;
    rejectNestedMutation();
    DataCall call(*this);
    const bool appended = appendRecord(std::move(operation));
    try {
        call.data->SetDictValueByKey(path, field, key, value);
    } catch (OmniUsdLiveResult) {
        if (appended) pending_.pop_back();
        rethrowRejected("host edit");
    } catch (...) {
        if (appended) pending_.pop_back();
        throw;
    }
    // Same record and extraction contract as the abstract-value Set overload.
    if (!appended) {
        notifyRecorded(false);
        return;
    }
    bool extracted = false;
    try {
        extracted = extractAbstractValue(value, VtValue{}, pending_.back().value);
    } catch (...) {
        pending_.pop_back();
        fault_->fail(currentFailure());
        throw;
    }
    if (!extracted) {
        pending_.pop_back();
        fault_->fail(OMNI_USD_LIVE_UNSUPPORTED);
        return;
    }
    notifyRecorded(true);
}
void HostSceneProjection::EraseDictValueByKey(const SdfPath& path, const TfToken& field, const TfToken& key) {
    if (announcing_) return;
    SdfOperation operation(SdfOperationKind::EraseDictKey);
    operation.path = path;
    operation.field = field;
    operation.key = key;
    forward(std::move(operation), [&](HostSceneStorage& data) { data.EraseDictValueByKey(path, field, key); });
}
void HostSceneProjection::SetTimeSample(const SdfPath& path, double time, const VtValue& value) {
    if (announcing_) return;
    SdfOperation operation(SdfOperationKind::SetSample);
    operation.path = path;
    operation.time = time;
    operation.value = value;
    forward(std::move(operation), [&](HostSceneStorage& data) { data.SetTimeSample(path, time, value); });
}
void HostSceneProjection::EraseTimeSample(const SdfPath& path, double time) {
    if (announcing_) return;
    SdfOperation operation(SdfOperationKind::EraseSample);
    operation.path = path;
    operation.time = time;
    forward(std::move(operation), [&](HostSceneStorage& data) { data.EraseTimeSample(path, time); });
}
bool HostSceneProjection::HasSpec(const SdfPath& path) const {
    DataCall call(*this);
    return call.data->HasSpec(path);
}
SdfSpecType HostSceneProjection::GetSpecType(const SdfPath& path) const {
    DataCall call(*this);
    return call.data->GetSpecType(path);
}
bool HostSceneProjection::Has(const SdfPath& path, const TfToken& field, VtValue* value) const {
    DataCall call(*this);
    return call.data->Has(path, field, value);
}
bool HostSceneProjection::Has(const SdfPath& path, const TfToken& field, SdfAbstractDataValue* value) const {
    DataCall call(*this);
    return call.data->Has(path, field, value);
}
bool HostSceneProjection::HasSpecAndField(const SdfPath& path, const TfToken& field, VtValue* value,
    SdfSpecType* type) const {
    DataCall call(*this);
    return call.data->HasSpecAndField(path, field, value, type);
}
bool HostSceneProjection::HasSpecAndField(const SdfPath& path, const TfToken& field, SdfAbstractDataValue* value,
    SdfSpecType* type) const {
    DataCall call(*this);
    return call.data->HasSpecAndField(path, field, value, type);
}
VtValue HostSceneProjection::Get(const SdfPath& path, const TfToken& field) const {
    DataCall call(*this);
    return call.data->Get(path, field);
}
std::vector<TfToken> HostSceneProjection::List(const SdfPath& path) const {
    DataCall call(*this);
    return call.data->List(path);
}
std::set<double> HostSceneProjection::ListAllTimeSamples() const {
    DataCall call(*this);
    return call.data->ListAllTimeSamples();
}
std::set<double> HostSceneProjection::ListTimeSamplesForPath(const SdfPath& path) const {
    DataCall call(*this);
    return call.data->ListTimeSamplesForPath(path);
}
bool HostSceneProjection::GetBracketingTimeSamples(double time, double* lower, double* upper) const {
    DataCall call(*this);
    return call.data->GetBracketingTimeSamples(time, lower, upper);
}
size_t HostSceneProjection::GetNumTimeSamplesForPath(const SdfPath& path) const {
    DataCall call(*this);
    return call.data->GetNumTimeSamplesForPath(path);
}
bool HostSceneProjection::GetBracketingTimeSamplesForPath(const SdfPath& path, double time, double* lower,
    double* upper) const {
    DataCall call(*this);
    return call.data->GetBracketingTimeSamplesForPath(path, time, lower, upper);
}
bool HostSceneProjection::GetPreviousTimeSampleForPath(const SdfPath& path, double time, double* previous) const {
    DataCall call(*this);
    return call.data->GetPreviousTimeSampleForPath(path, time, previous);
}
bool HostSceneProjection::QueryTimeSample(const SdfPath& path, double time, VtValue* value) const {
    DataCall call(*this);
    return call.data->QueryTimeSample(path, time, value);
}
bool HostSceneProjection::QueryTimeSample(const SdfPath& path, double time, SdfAbstractDataValue* value) const {
    DataCall call(*this);
    return call.data->QueryTimeSample(path, time, value);
}
void HostSceneProjection::_VisitSpecs(SdfAbstractDataSpecVisitor* visitor) const {
    if (!visitor) throw std::invalid_argument("missing Sdf visitor");
    struct Forward final : SdfAbstractDataSpecVisitor {
        const HostSceneProjection& view;
        SdfAbstractDataSpecVisitor& visitor;
        Forward(const HostSceneProjection& source, SdfAbstractDataSpecVisitor& target)
            : view(source), visitor(target) {}
        bool VisitSpec(const SdfAbstractData&, const SdfPath& path) override { return visitor.VisitSpec(view, path); }
        void Done(const SdfAbstractData&) override {} // The outer VisitSpecs owns Done.
    } forward(*this, *visitor);
    DataCall call(*this);
    call.data->VisitSpecs(&forward);
}
void publishSceneDifference(SdfLayerStateDelegateBase& delegate, HostSceneProjection& projection,
    HostSceneProjection::Reseat&& prepared, const WireSceneDifference& difference) {
    const HostSceneProjection::Announcement announcement(projection);
    // A removed or created spec is reported as significant, as a host's own
    // non-inert edit is: its fields arrive with it and have no notices.
    // Removals and moves traverse the old content: USD visits the old
    // subtree to relocate every retained spec identity under a move.
    for (const auto& path : difference.removed) delegate.DeleteSpec(path, false);
    for (const auto& [from, to] : difference.moved) delegate.MoveSpec(from, to);
    projection.reseat(std::move(prepared));
    for (const auto& [path, type] : difference.created) delegate.CreateSpec(path, type, false);
    for (const auto& field : difference.fields) {
        VtValue before = field.before;
        delegate.SetField(field.path, field.field, field.after, &before);
    }
}
} // namespace omni::usd::live
