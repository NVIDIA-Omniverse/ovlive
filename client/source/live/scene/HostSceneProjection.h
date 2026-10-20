// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "WireScene.h"
#include "HostSceneStorage.h"
#include "SdfOperation.h"
#include "LiveTsan.h"
#include <omni/usd/live/Live.h>
#include <pxr/usd/sdf/data.h>
#include <pxr/usd/sdf/layerStateDelegate.h>
#include <atomic>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

namespace omni::usd::live {
// A live-layer operation rejected before any mutation (design section 8): whole-
// content replacement through CopyFrom, a mutation refused by the host edit
// quota, a mutation that begins while a forwarded call on the projection is in
// progress (rejectNestedMutation: typed BUSY, no USD runtime error), or a
// storage rejection before mutation (a NaN time sample, an unextractable
// foreign value, or a write while a read or another forwarded call holds the
// storage). It derives from std::exception, so a host that
// catches std::exception around an Sdf call continues with the layer's content
// and status unchanged. `result()` is the typed cause.
class LayerOperationRejected final : public std::runtime_error {
public:
    explicit LayerOperationRejected(OmniUsdLiveResult result);
    OmniUsdLiveResult result() const noexcept { return result_; }
private:
    OmniUsdLiveResult result_;
};
// Sticky host failure: the first typed cause (an unrecordable write, an
// unsupported flush or authoring outside the host context) fails the layer;
// the host content keeps every edit. Atomic observation is thread-safe;
// mutable USD storage still requires the original serial host context. No
// callbacks, allocation, or waiting occur here.
class HostFailureLatch final {
public:
    // Thread-neutral sticky failure cause. An aliasing shared_ptr can retain
    // this atomic for actor gates without retaining mutable USD data or callbacks.
    const std::atomic<OmniUsdLiveResult>& result() const noexcept { return result_; }
    // Latch the first failure cause. Later causes never replace it.
    void fail(OmniUsdLiveResult cause) noexcept {
        if (cause == OMNI_USD_LIVE_OK) return;
        auto expected = OmniUsdLiveResult(OMNI_USD_LIVE_OK);
        result_.compare_exchange_strong(expected, cause, std::memory_order_acq_rel, std::memory_order_acquire);
    }
private:
    static_assert(std::atomic<OmniUsdLiveResult>::is_always_lock_free);
    std::atomic<OmniUsdLiveResult> result_{OMNI_USD_LIVE_OK};
};

// The bytes one buffered record charges to the host edit buffer quota and to
// its endpoint's shared buffer budget (design section 8, Quotas): the record
// itself, two turn-boundary entries and an estimate of its value's heap
// bytes: string and asset path characters; each array, vector and list
// operation item at its type's size and at least 16 bytes, plus its own
// characters or nested value; and 64 bytes for each dictionary entry, time
// sample and variant selection, plus its key characters and value. Paths
// and names are interned by USD and charge nothing of their own. The
// estimate saturates at the largest size_t.
size_t hostEditRecordBytes(const SdfOperation& operation) noexcept;

// Host edit buffer quota (design section 8). A host turn ends when a flush
// attempt ends. A mutation that would start a new turn while the buffer holds
// maxPendingBytes or more, or while the budget sink reports its shared budget
// full, is refused before it changes anything, unless the layer has
// failed; mutations inside a started turn always record, so one USD call
// never applies partially. The endpoint's shared budget bounds every layer's
// buffer together; a layer's own quota defaults to 1 GiB, the default of
// max_pending_bytes_per_endpoint, so a buffer no endpoint budget charges,
// such as a created layer's before activation, is bounded as one layer on a
// default endpoint is.
struct HostEditBufferLimits {
    size_t maxPendingBytes = size_t(1) << 30;
};
// The shared budget a buffer is charged to, as the budget sink reports it:
// whether it is full, and its charged bytes and limit for the refusal's
// USD runtime error.
struct HostEditBudget {
    bool full = false;
    uint64_t charged = 0, limit = 0;
};

// Mutable backing for production live layers; immutable service data is separate.
// Prepare plugin-owned mutable storage eagerly, before exposing it to a host.
// The authoritative baseline stays immutable. All projection access, including
// capture, is host-confined.
// Mutations can allocate and throw; this class does not contain failures of
// complete USD operations.
class HostSceneProjection final : public PXR_NS::SdfAbstractData {
public:
    using Limits = HostEditBufferLimits;
    // A zero quota throws std::invalid_argument.
    HostSceneProjection(WireScenePtr baseline, std::shared_ptr<HostFailureLatch> fault, Limits limits = {});
    // Every forwarded call's reads happen before the storage is freed.
    ~HostSceneProjection() override;
    HostSceneProjection(const HostSceneProjection&) = delete;
    HostSceneProjection& operator=(const HostSceneProjection&) = delete;
    const WireScenePtr& baseline() const noexcept { return baseline_; }
    const std::shared_ptr<HostFailureLatch>& fault() const noexcept { return fault_; }
    const Limits& limits() const noexcept { return limits_; }
    // Host mutations applied to storage and not yet admitted, as one sealed unit.
    struct LocalEdits {
        std::shared_ptr<const std::vector<SdfOperation>> operations;             // capture order
        PXR_NS::SdfAbstractDataConstRefPtr data;                                 // capture() taken at peek time
        WireScenePtr base;                                                       // baseline the records apply to
    };
    // Host-confined. Called after every recorded operation with the buffer size.
    // The sink only schedules; it never reenters this projection. A throwing
    // sink fails the layer with INTERNAL_ERROR instead of reaching the mutator.
    void setLocalEditSink(std::function<void(size_t pending)> onRecorded);
    // Host-confined. Called with the buffer size when a mutation would start a
    // host turn at the quota on a layer whose host fault is clear, before any
    // change. It returns whether to refuse: false records the mutation (its
    // owner already failed the layer). Without a sink the mutation is refused.
    // Like the record sink it only schedules and never reenters this
    // projection; a throwing sink fails the layer with INTERNAL_ERROR and the
    // mutation is refused with that cause. A refusal reports the cause the
    // layer latched, if any, else RESOURCE_EXHAUSTED.
    void setQuotaSink(std::function<bool(size_t pending)> onQuota);
    // Host-confined. Called with the buffered bytes when a mutation would
    // start a host turn below the layer's own quota on a layer whose host
    // fault is clear, before any change. It charges the buffer to its shared
    // budget and reports whether that budget is full; a full
    // budget applies the quota, and the quota sink decides as above. It
    // never reenters this projection; a throwing sink fails the layer with
    // INTERNAL_ERROR and the mutation is refused with that cause. Without a
    // sink only the layer's own quota applies.
    void setBudgetSink(std::function<HostEditBudget(size_t pendingBytes)> onBudget);
    // Host-confined. Ends the current host turn; the flush path calls it
    // after every flush attempt.
    void endHostTurn() noexcept { turnOpen_ = false; }
    bool hasPendingLocalEdits() const noexcept { return !pending_.empty(); } // Host-confined.
    size_t pendingLocalEditCount() const noexcept { return pending_.size(); } // Host-confined.
    // Host-confined. The buffered records' hostEditRecordBytes total.
    size_t pendingLocalEditBytes() const noexcept { return pendingBytes_; }
    // Host-confined. Drop every buffered record and retain no later one. A
    // later mutation still applies to storage and notifies the record sink,
    // and the quota never refuses it. The owner calls this once the layer is
    // FAILED, whose records are never admitted, and keeps the buffered count
    // itself. Idempotent.
    void discardLocalEdits() noexcept;
    bool discardsLocalEdits() const noexcept { return discarding_; } // Host-confined.
    // Host-confined. The buffered records, a capture of storage and the
    // baseline they were applied to. The buffer is unchanged.
    LocalEdits pendingLocalEdits() const;
    // Host-confined. Drop the first `count` records (all admitted by one flush)
    // and rebaseline without capturing. `count` zero only rebaselines.
    void acceptLocalEdits(size_t count, WireScenePtr next);
    // Content that replaces this projection's own in place, prepared before
    // any change. Host-confined.
    struct Reseat {
        WireScenePtr scene;
        PXR_NS::TfRefPtr<HostSceneStorage> storage;
    };
    // `difference` is diffProjectedScenes(*baseline(), *next). The storage
    // shares this projection's unchanged spec records and index structure
    // and takes `next`'s for the specs the difference changes, so the cost is
    // proportional to the changed specs' fields, with a logarithmic factor in
    // the spec count, never in field values. The host content equals the
    // baseline, so the result equals `next`.
    Reseat prepareReseat(WireScenePtr next, const WireSceneDifference& difference) const;
    // Whether a publication of `next`, with `difference` from the baseline,
    // keeps the baseline and content as they are: `next` is the baseline, or
    // the difference is empty and the source format unchanged. The content
    // then equals its baseline exactly, even where the comparison treats a
    // representation change as equal; the next flush compiles against the
    // baseline and the journal replays the intent onto its visible scene.
    bool keepsBaseline(const WireScenePtr& next, const WireSceneDifference& difference) const;
    // Host-confined. Replace the content and baseline with a prepared reseat;
    // nothing is recorded and USD is not told. Throws std::logic_error, with
    // nothing changed, while the buffer holds records, a forwarded call is in
    // progress or the reseat is empty. A published reseat is visible to
    // readers that start after it.
    void reseat(Reseat&&);
    // Host-confined. While an Announcement lives, a mutation that reaches the
    // projection changes and records nothing: the content already holds its
    // result, and SdfLayer's primitive edit only sends its change notice.
    class Announcement final {
    public:
        explicit Announcement(HostSceneProjection& projection) noexcept : projection_(projection) {
            projection_.announcing_ = true;
        }
        ~Announcement() { projection_.announcing_ = false; }
        Announcement(const Announcement&) = delete;
        Announcement& operator=(const Announcement&) = delete;
    private:
        HostSceneProjection& projection_;
    };
    // Caller pins/synchronizes this projection for the entire copy. Failure
    // leaves all owners unchanged. The capture shares the storage's immutable
    // spec records and index structure, so it costs constant time and later
    // edits of the projection never reach it. This is stock USD value-copy
    // independence, not serialization/deep-copy of arbitrary custom VtValue
    // pointer payloads.
    PXR_NS::SdfAbstractDataConstRefPtr capture() const;
    bool StreamsData() const override { return true; }
    bool IsDetached() const override { return false; }
    // Posts a USD runtime error and throws LayerOperationRejected(UNSUPPORTED)
    // before any storage change. USD 25.11 TransferContent reaches it on fresh
    // InitData before the layer's backing changes.
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
private:
    void _VisitSpecs(PXR_NS::SdfAbstractDataSpecVisitor*) const override;
    // Before any storage change: refuse a mutation that would start a host
    // turn at the quota, then append its record. Returns whether it appended
    // one; a discarding projection appends nothing and refuses nothing.
    // Throws with nothing changed.
    bool appendRecord(SdfOperation&& operation);
    // Apply the storage edit for the record appendRecord just appended, if
    // any. An edit that throws removes that record; one that applies
    // notifies the sink.
    template<class Edit> void forward(SdfOperation&& operation, Edit&& edit);
    // Charges the record appended last, when `appended`, then notifies the
    // record sink.
    void notifyRecorded(bool appended) noexcept;
    // Throws LayerOperationRejected(BUSY) for a mutation that begins while a
    // forwarded call on this projection is in progress, before any change.
    void rejectNestedMutation() const;
    WireScenePtr baseline_;
    std::vector<SdfOperation> pending_;
    size_t pendingBytes_ = 0; // hostEditRecordBytes of pending_, host-confined.
    std::function<void(size_t)> onRecorded_;
    std::function<bool(size_t)> onQuota_;
    std::function<HostEditBudget(size_t)> onBudget_;
    bool turnOpen_ = false;
    // Host-thread state, like pending_: an off-host mutator racing a discard is
    // the same contract violation as an off-host record and has stock SdfData
    // semantics only.
    bool discarding_ = false;
    bool announcing_ = false; // Host-confined; see Announcement.
    const std::shared_ptr<HostFailureLatch> fault_;
    // Replaced only by reseat, on the host.
    PXR_NS::TfRefPtr<HostSceneStorage> data_;
    // Forwarded calls in progress. Concurrent readers (USD composition reads
    // from worker threads) may overlap and retire in any order, so this is a
    // depth count, not a saved flag; it is reentry accounting, not a lock. The
    // host serializes its authoring against reads (design section 7); a
    // mutation concurrent with a reader is outside that contract. The guard
    // loads the depth once, so it refuses only a mutation that begins while a
    // read is already in progress, throwing LayerOperationRejected(BUSY)
    // through USD before any change. An edit that reaches storage while a read
    // holds it is refused by the storage write guard with the same typed
    // rejection and a USD runtime error; any other overlap is an undetected
    // race.
    mutable std::atomic<uint64_t> forwardDepth_{0};
    // The constructed projection as a forwarded call sees it. USD hands the
    // projection to its callers, including TBB workers, through synchronization
    // ThreadSanitizer cannot see; this acquires the constructor's release.
    static const HostSceneProjection* published(const HostSceneProjection& projection) noexcept {
        LIVE_TSAN_ACQUIRE(&projection);
        return &projection;
    }
    // Pin this projection and its storage through callbacks, and count the
    // forwarded call until it returns; nested read calls keep it counted.
    struct DataCall {
        const PXR_NS::TfRefPtr<const HostSceneProjection> owner;
        const PXR_NS::TfRefPtr<HostSceneStorage> data;
        std::atomic<uint64_t>& active;
        explicit DataCall(const HostSceneProjection& projection)
            : owner(published(projection)), data(projection.data_), active(projection.forwardDepth_) {
            active.fetch_add(1);
        }
        ~DataCall() { active.fetch_sub(1); }
        DataCall(const DataCall&) = delete;
    };
    const Limits limits_;
};
// Host-confined. Installs `prepared` into `projection`, the backing of the
// layer whose state delegate is `delegate`, and sends `difference`, from
// which it was prepared, as USD's per-path change notices. Each change goes
// through the delegate's primitive Sdf edit, which records its notice in the
// caller's SdfChangeBlock and reaches the projection inside an Announcement,
// so the edit changes and records nothing. Removals and moves run against the
// old content, which USD traverses to relocate the identities of a moved
// subtree's retained spec handles; then the reseat installs the new content,
// and creations and field changes run against it, where USD reads the new
// values. A removal or creation reports a significant change, so its
// subtree's fields send no notices of their own.
void publishSceneDifference(PXR_NS::SdfLayerStateDelegateBase& delegate, HostSceneProjection& projection,
    HostSceneProjection::Reseat&& prepared, const WireSceneDifference& difference);
} // namespace omni::usd::live
