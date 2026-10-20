// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "DeltaEntries.h"
#include "SdfOperation.h"
#include "UnsupportedOperation.h"
#include "WireScene.h"
#include <cstdint>
#include <functional>
#include <mutex>
#include <set>

namespace omni::usd::live {
struct FieldEditLimits {
    WireSceneLimits scene;
    wire::EnvelopeEncodeLimits envelope;
    wire::EncodeLimits values;
    // The service's entry limits for the compiled update; an update over them
    // throws DeltaEntryLimitExceeded before it is encoded.
    DeltaEntryLimits entries;
    // The most operations whose writes one compiled update emits. Repeated
    // writes to one field or sample collapse into one record, and only the
    // first of them counts, so a log may hold more operations than this.
    size_t maxOperations = 65536;
    // The most operations a log holds, checked before any per-operation work,
    // so the cost of compiling a log is bounded before collapse is known. The
    // host edit buffer is bounded by bytes and may hold more records; a flush
    // admits the most whole turns within this limit, so only one turn over
    // it fails, as the operation limit does. 16 times maxOperations leaves a
    // turn room for many repeated writes.
    size_t maxLogOperations = size_t(1) << 20;
    size_t maxStructuralCommands = 65536;
    size_t maxSectionIds =
        size_t(1) << 26; // Observed IDs, live and retired; a layer holds as many as the service sends.
    size_t maxIdAttempts = 64;
    size_t maxWork = 512u * 1024 * 1024;
    size_t maxRetainedBytes = 512u * 1024 * 1024;
    // Compile and verify each log at its footprint, so its cost follows the
    // log, not the scene: a log with no structural operation and no
    // child-field record skips namespace planning, whose plan is empty;
    // identities resolve through the base's path index; the namespace plan
    // and child-order normalization follow the paths the log creates, erases
    // and moves; and the candidate, the replay store and the host capture
    // are compared at the paths the log addresses and the specs the candidate
    // changes. False plans, normalizes and verifies every log against the
    // whole scene; the compiled bytes are the same either way.
    bool footprint = true;
};
// Incarnation-scoped allocation history. Full-width random IDs provide only
// probabilistic uniqueness across clients/unseen history, so they also rely on
// service collision rejection. Never reuse observed or already allocated IDs,
// even after failed preparation. The optional private generator must not reenter.
class SectionIdAllocator final {
public:
    explicit SectionIdAllocator(size_t maxIds = size_t(1) << 26, size_t maxAttempts = 64,
        std::function<uint64_t()> random = {});
    // Optional exclusive transaction budget charges every entropy/collision
    // attempt. An identity `taken` reports is observed, as if seen before, and
    // drawn again within the same attempts.
    uint64_t allocate(WireSceneWorkBudget* = nullptr, const std::function<bool(uint64_t)>& taken = {});
    void observe(uint64_t);
    size_t observedCount() const;
private:
    const size_t maxIds_, maxAttempts_;
    std::function<uint64_t()> random_;
    mutable std::mutex mutex_;
    std::set<uint64_t> observed_;
};
class FieldEditIntent;
using FieldEditIntentPtr = std::shared_ptr<const FieldEditIntent>;
// The compiler behind compileFieldEdits; `issued`, when given, receives every
// identity a compilation allocates, in order.
struct FieldEditCompiler {
    static FieldEditIntentPtr compile(const WireScenePtr&, const std::shared_ptr<const SdfOperationLog>&,
        const FieldEditLimits&, const std::shared_ptr<SectionIdAllocator>&, WireScenePtr* applied,
        std::vector<uint64_t>* issued);
};
// Standalone identity-aware one-spec structural, field, dictionary and sample compiler.
// CopyFrom and changing an existing identity's type remain unsupported.
// Input owners must remain immutable throughout compilation; only the operation log is retained afterward.
class FieldEditIntent final {
public:
    // The original ordered operation log. The intent retains neither its base
    // scene nor the host's sealed copy; both are compile-time inputs only.
    const std::shared_ptr<const std::vector<SdfOperation>>& operations() const { return operations_; }
    const std::string& bytes() const { return bytes_; }
    // Conservative logical representation of what the intent retains: its
    // operations' paths, names and encoded values with element/record
    // overhead, and its compiled bytes. Proportional to the edit, never to the
    // layer. This is not an allocator/RSS or SDK-cache bound.
    size_t retainedBytes() const { return retainedBytes_; }
    // Successful compilation's shared logical work charge; callers must carry
    // it into subsequent admission/replay instead of resetting the budget.
    size_t workUsed() const { return workUsed_; }
    // False for scratch Create commands or structural transactions needing
    // whole child-membership field overrides. Those need additional service
    // contracts before admission. Qualified Move/Delete commands deliberately
    // replay against the latest subtree without freezing base membership.
    // A log's edits, child order or moves addressing identities it removes are
    // dropped and do not clear it.
    bool concurrentStructureSupported() const { return concurrentStructureSupported_; }
    // Why concurrentStructureSupported() is false: MoveCycle for scratch
    // Creates, else MultiUpdate. Zero when it is true.
    UnsupportedOperation::Reason unsupportedStructure() const { return unsupportedStructure_; }
    // True when the compiled envelope carries no command, field or sample.
    bool empty() const { return empty_; }
    // Whether compilation planned the namespace (see FieldEditLimits::footprint).
    bool planned() const { return planned_; }
private:
    friend struct FieldEditCompiler;
    std::shared_ptr<const std::vector<SdfOperation>> operations_;
    std::string bytes_;
    bool concurrentStructureSupported_ = true, empty_ = false, planned_ = true;
    UnsupportedOperation::Reason unsupportedStructure_{};
    size_t retainedBytes_ = 0, workUsed_ = 0;
};
// The original operation log is authoritative. Compilation starts from `base`;
// sealed.data must match its complete final wire projection, but is never
// used to infer operations or resolve paths. Dictionary-key edits lower to whole-field
// replacements; concurrent per-key merge is NOT promised. Original operations
// remain retained even when wire phase normalization suppresses older samples.
// Dictionary key components (including empty components preserved by pinned
// USD's ':' splitting) and their combined depth with the inserted value are
// preflighted before USD expansion, including erase/no-op key traversals.
// DecodeError classifies malformed input, unsupported scope and quota failure.
// No scene publication, admission, or stale-base reconciliation occurs here.
// SceneJournal rejects unqualified compiled structural shapes. Supply one
// allocator for the entire incarnation when compiling multiple logs; the
// default temporary allocator only protects this standalone compilation.
// A non-null `applied` receives the verified scene of base plus this intent,
// which the compiled intent does not retain. A log whose `data` is null, a
// prefix of a host edit buffer whose capture holds later records too, is
// verified only against its operations replayed onto the base; the rest of
// the buffer is verified against its capture when it compiles.
FieldEditIntentPtr compileFieldEdits(const WireScenePtr&, const std::shared_ptr<const SdfOperationLog>&,
    const FieldEditLimits& = {}, const std::shared_ptr<SectionIdAllocator>& = {}, WireScenePtr* applied = nullptr);
// Test hook, off by default: while enabled, every compilation at the
// footprint also compiles the log against the whole scene, drawing the same
// identities, and throws std::logic_error unless both compile the same bytes,
// retained size, emptiness and structure support, or fail with the same
// error class. Production code never enables it.
void setFieldEditCompileCheck(bool enabled) noexcept;
// True when `data` holds exactly `scene`'s projected specs, field presence and
// wire-encoded values. A null scene, or a limit or codec failure, returns false.
bool matchesWireProjection(const WireScenePtr& scene, const PXR_NS::SdfAbstractData& data,
    const FieldEditLimits& limits = {});
// Test seam: the compileFieldEdits calls this process has made.
uint64_t fieldEditCompilationsForTest() noexcept;
} // namespace omni::usd::live
