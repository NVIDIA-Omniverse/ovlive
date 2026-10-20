// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "SceneJournal.h"
#include "LiveLayerAdmission.h"
#include "LiveLayerStatus.h"

namespace omni::usd::live {
// Private owned finite CreateLayer transaction. Create replaces an existing
// service layer; calling start() is explicit mutation consent, never a retry.
// Shares a typed status subject with its confirmed attachment. UsdLayerCreation
// binds it to a USD layer for host::Creation and the Export and CreateNew hooks.
class SceneCreation final {
public:
    enum class Phase { Prepared, Creating, Confirmed, UnissuedFailure, Rejected, OutcomeUnknown };
    struct Snapshot {
        Phase phase = Phase::Prepared;
        OmniUsdLiveResult result = OMNI_USD_LIVE_OK;
        OmniUsdLiveRequestError error{};
        uint64_t object = 0, sequence = 0;
        OmniUsdLiveRequest request = 0;
        bool started = false, closing = false, quiescent = false;
    };
    // Complete owned input, validated against an empty scene. A diff seed must
    // have baseVersion zero. Reserves canonical admission and scheduler capacity
    // before returning, without issuing RPCs. Throws typed results on failure.
    // With an admission deadline, a closing backing of the destination is
    // waited for until then; the caller must be allowed to block.
    static std::shared_ptr<SceneCreation> prepare(std::shared_ptr<Runtime>, OmniUsdLiveEndpoint, std::string session,
        std::string layer, std::string seedBytes, SceneJournal::Limits = {},
        std::optional<std::chrono::steady_clock::time_point> admissionDeadline = {});
    ~SceneCreation();
    SceneCreation(const SceneCreation&) = delete;
    SceneCreation& operator=(const SceneCreation&) = delete;
    // At most one attempt, including a preissuance failure. A repeated call
    // returns ALREADY_EXISTS. OK is submission, not confirmation of execution.
    OmniUsdLiveResult start() noexcept;
    // Consume the one attempt without transport (e.g. a hook's deadline expired
    // during preparation or a host mutation fault latched on its target).
    // Retain an exact unissued result and wake quiescence. The result carries
    // no request or uncertain mutation; an already admitted attempt is never
    // overwritten.
    OmniUsdLiveResult failBeforeStart(OmniUsdLiveResult) noexcept;
    // OK means request/charged callback captures have retired; inspect snapshot
    // for execution outcome. Prepared wait returns BUSY. Timeout never cancels.
    OmniUsdLiveResult wait(uint32_t timeoutMs) noexcept;
    // Cancel/quiesce only. Keep seed, bytes, receipt and admission until this
    // owner is released. A timeout retains asynchronous cleanup ownership.
    OmniUsdLiveResult close(uint32_t timeoutMs) noexcept;
    Snapshot snapshot() const;
    // Retained creation provenance, not an applied journal frontier. Only after
    // Confirmed may a caller claim admission for one actor using object from
    // snapshot(), resume=0 and discover=false. Its initial journal is empty.
    // Retain this creation owner separately through any bootstrap failure.
    const WireScenePtr& seed() const noexcept;
    const std::string& bytes() const noexcept;
    const std::shared_ptr<LayerAdmission>& admission() const noexcept;
    // Stable canonical subject, shared with the confirmed attachment. Closing
    // this finite owner after handoff cannot replace the actor's status.
    const std::shared_ptr<LayerStatus>& status() const noexcept;
    // Bounds every reissue after a refusal that applied nothing (design
    // section 13) by the caller's initial-open deadline, which replaces
    // initial_open_timeout_ms counted from start(). A reissue whose backoff
    // would end at or after it, or that is still pending when it passes, is
    // not issued and the refusal's UNAVAILABLE stands.
    void boundReissues(std::chrono::steady_clock::time_point deadline) noexcept;
    // The opening this creation serves was abandoned: no later reissue goes
    // out. An attempt in flight still settles.
    void abandonReissues() noexcept;
    // Test seam: runs on the dispatch worker before a re-send or a reissue is
    // submitted; returning true fails the submission. Null removes it.
    static void setBeforeResendForTest(bool (*hook)(void*) noexcept, void* context) noexcept;
private:
    struct State;
    explicit SceneCreation(std::shared_ptr<State> state) : state_(std::move(state)) {}
    const std::shared_ptr<State> state_;
};
} // namespace omni::usd::live
