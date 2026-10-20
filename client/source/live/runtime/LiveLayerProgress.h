// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <omni/usd/live/Live.h>
#include <array>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace omni::usd::live {

// One controller per canonical attachment, never shared by object ID alone.
struct LayerIdentity {
    OmniUsdLiveEndpoint endpoint;
    std::string session, layer, writer;
    uint64_t object;
};

// Pure layer control/retention state. The caller supplies complete immutable
// intent/coverage owners, not transport completion tags or partial buffers.
// The eventual layer actor still owns transport scheduling, OMPF validation,
// ordered application and canonical attachment registration.
class LayerProgress {
public:
    enum class Gate { Recovering, Ready, Failed };
    // Committed: the writer's chain proves a local write committed, at a
    // position the transport never reported. A read started after the proof
    // delivers its commit as ordinary coverage, and its intent retires with
    // the application of the next local write that has a position.
    enum class Disposition { Peer, Queued, Issued, Acknowledged, Unknown, Rejected, Committed };
    enum class CoverageKind { Update, Aggregate, Snapshot };
    using ContentHash = std::array<uint8_t, 32>;
    struct Coverage {
        std::shared_ptr<const void> owner;
        ContentHash fingerprint; // SHA-256 of the complete original payload.
        CoverageKind kind = CoverageKind::Update;
    };
    struct Work {
        uint64_t id = 0, first = 0, last = 0;
        Disposition disposition = Disposition::Queued;
        bool applied = false;
        std::shared_ptr<const void> owner;
        ContentHash fingerprint{};
        CoverageKind kind = CoverageKind::Update;
        uint64_t retainedBytes = 0; // Local complete-intent logical charge.
        uint64_t localOrdinal = 0; // Consecutive local admission identity; peers are zero.
        bool authReissued = false; // The single post-refresh reissue has been offered.
        bool resent = false; // The single operation-ID re-send after an unknown outcome has been offered.
        uint32_t refusals = 0; // Consecutive attempts the service refused with nothing applied.
        bool fingerprinted = false; // A local intent's issued bytes are fingerprinted.
        // A local write's attempts that reached transport, the highest
        // position observed when the first of them was taken (its position is
        // above it), and whether an attempt's outcome was unknown, so that
        // the write may have committed although it is queued again.
        uint64_t attempts = 0, floor = 0;
        bool mayHaveCommitted = false;
        // A local write: the work whose application retires it, an aggregate
        // containing its effect or, for a committed write, the next local
        // write's application.
        uint64_t absorbedBy = 0;
        // The local writes this work's application retires before its own
        // effect, and the last of their local ordinals.
        uint64_t absorbedLocal = 0, absorbedThrough = 0;
        // Committed: the first read generation started after the chain proved it.
        uint64_t readAfter = 0;
        // An acknowledged local write, or peer coverage an earlier read
        // delivered, that a resumed read's net-empty catch-up passed: the
        // service's net result already accounts for it, so it retires without
        // applying its bytes.
        bool netted = false;
    };
    // AwaitAuth: the service rejected the attempt UNAUTHENTICATED before
    // executing it; the intent is queued again at the FIFO head for one
    // reissue after the caller obtains a usable token generation, and the
    // attempt holds no position, so the positions the read skipped that no
    // other unsettled write may hold fill as empty commits. When a filled
    // hole is wider than the gap bound, the layer fails RESOURCE_EXHAUSTED
    // and the write settles without the reissue.
    enum class WriteSettlement { Settled, AwaitAuth };
    struct ReadCursor {
        uint64_t generation, object, sequence;
    };
    struct ReadySignal {
        bool accepted = false, initialLatest = false;
    };
    struct Publication {
        void (*commit)(void*, uint64_t) noexcept = nullptr;
        void* context = nullptr;
    };
    struct Snapshot {
        Gate gate;
        OmniUsdLiveResult result;
        uint64_t generation, covered, applied, readSequence;
        uint64_t issued; // Local writes in flight.
        size_t retained, queued;
        bool readStarted, initialLatestPublished;
        uint64_t pendingWrites = 0, pendingBytes = 0;
        uint64_t object = 0; // Zero only before validated initial discovery.
        // Attachment-lifetime ledger, independent of peer/shared work IDs.
        // Retained local ordinals are exactly (localRetired, localAdmitted].
        uint64_t initialApplied = 0, localAdmitted = 0, localRetired = 0;
        // Failed by stopIssuance and never stopped: the read subscription
        // stays (or reconnects) and applied work keeps advancing.
        bool readingAfterFailure = false;
        // Local writes the chain proves committed, which retire without a
        // reported position; and queued writes an earlier attempt of which has
        // an unknown outcome, with committed writes that no later local write
        // with a position bounds yet.
        uint64_t unplaced = 0, uncertain = 0;
        // Committed writes whose commit only a read started after the proof
        // delivers, while an older read is open: the layer resubscribes.
        uint64_t awaitingRead = 0;
    };

    // A nonzero resume point requires the caller's retained layer-state owner.
    // This is the caller's explicit cursor, never another attachment's cache.
    // Discovery requires object/resume zero. Only retained complete bootstrap
    // coverage adopts an object; local admission is BUSY until that happens.
    LayerProgress(LayerIdentity identity, uint64_t resume, std::shared_ptr<const void> initialState, size_t maxRetained,
        uint64_t maxGap, bool discover = false);
    // The immutable requested identity; objectId()/snapshot().object report an
    // incarnation adopted by a new discovery attachment.
    const LayerIdentity identity;
    uint64_t objectId() const;

    // Admission precedes async commit acceptance. The owner must retain the
    // complete edit, including values/dependencies needed for recovery export.
    // When supplied, commit runs synchronously exactly once after both retained
    // work and FIFO storage are allocated, before takeWrite can observe the ID.
    // It runs under the progress mutex: only preallocated ID assignment/pointer
    // swaps are permitted; no reentry, locking, allocation or owner destruction.
    // The caller retains displaced owners for retirement after its own locks.
    // Rejected admission never calls commit or consumes an ID/local ordinal.
    // A non-null fingerprint is the SHA-256 of the bytes every issue of this
    // intent sends.
    uint64_t admit(std::shared_ptr<const void> intent, Publication publication = {nullptr, nullptr},
        uint64_t retainedBytes = 0, const ContentHash* fingerprint = nullptr);
    // Reserves the next queued write, in FIFO order, while fewer than
    // `window` writes are in flight: once the read is ready or, with the read
    // closed, while a queued write's outcome is unknown, so that its re-send
    // settles it before the read resumes (design section 12). A caller that
    // never issues transport must defer it; once execution is possible it
    // must report an outcome. Acknowledgments of in-flight writes settle in
    // any order.
    Work takeWrite(size_t window = 1); // id==0 when the gate, window or FIFO prevents issuance.
    // Reserves one queued write out of FIFO order, as takeWrite does: a
    // broken chain's latest write whose outcome is unknown, whose answer
    // locates the chain without waiting on an earlier one (design section 13).
    Work takeWriteOutOfOrder(uint64_t id, size_t window);
    // A queued write that has not committed and cannot commit before its next
    // attempt, since an earlier write's every attempt applied nothing:
    // positions are not held for it.
    void clearUncertain(uint64_t id);
    // Returns a taken write that never reached transport to the FIFO at its
    // admission position. It holds no position, so the positions the read
    // skipped that no other unsettled write may hold fill as empty commits;
    // a filled hole wider than the gap bound fails the layer.
    void deferWrite(uint64_t id);
    // Returns an in-flight write whose attempt applied nothing, or whose
    // outcome a later attempt resolves, to the FIFO at its admission
    // position, keeping its first issue's floor. `mayHaveCommitted` marks an
    // attempt whose outcome is unknown: until the write settles, a position
    // the read skips above its floor may be its commit and is never filled
    // as empty. False, with nothing changed, once issuance has stopped.
    bool requeueWrite(uint64_t id, bool mayHaveCommitted);
    // A queued write that an earlier attempt reached transport for, which
    // the writer's chain proves committed (a later write of the chain
    // committed, or the service named it or a later write as the writer's
    // last committed operation). Its position is never inferred: it holds the
    // positions above its floor and below the next local write with a
    // position until a read started after the proof passes the position
    // before that write. That read delivers its commit as ordinary coverage,
    // and the write's intent then retires with the next write's application
    // (design section 12).
    void confirmCommitted(uint64_t id);
    // A definitive rejection retires the intent as Rejected and stops
    // issuance with that result (stopIssuance); OUTCOME_UNKNOWN retires it as
    // Unknown and also closes the read (fail), since the unknown write's
    // sequence is never learned (design section 13). So does a definitive
    // rejection of a write that was re-sent or requeued with
    // `mayHaveCommitted`, keeping its typed result: the earlier attempt may
    // have committed at a position that is never learned. A re-sent write's
    // UNAUTHENTICATED or UNAVAILABLE result is OUTCOME_UNKNOWN: it earns no
    // further issue, and its first attempt's outcome stays unknown. So is
    // that of a write requeued with `mayHaveCommitted`, apart from the
    // UNAUTHENTICATED reissue. An OK at a position strictly after the first
    // position of a retained, unapplied aggregate is absorbed by it, as
    // acceptCoverage absorbs an earlier acknowledgment; any other position
    // retained coverage, an empty commit or another local write's chain
    // order contradicts, or one at or below the write's floor or the applied
    // frontier, is OUTCOME_UNKNOWN.
    WriteSettlement completeWrite(uint64_t id, OmniUsdLiveResult result, uint64_t sequence = 0);
    // Offers the in-flight write, whose attempt the service refused
    // UNAVAILABLE with nothing applied (design section 13), a reissue of the
    // same bytes: it is queued again at its FIFO position, as for the
    // UNAUTHENTICATED reissue, and this attempt holds no position, so the
    // positions the read skipped that no other unsettled write may hold fill
    // as empty commits. Returns its consecutive refusals, this one included,
    // or zero when it was re-sent, this refusal reaches `limit`, issuance
    // stopped, or a filled hole exceeds the gap bound; the caller then
    // completes it with the refusal's result.
    uint32_t refuseWrite(uint64_t id, uint32_t limit);
    // Offers the in-flight write, whose attempt carried an operation ID and
    // ended with an unknown outcome, its single re-send of the same bytes with
    // that ID (design section 13). The write stays issued until the re-send
    // completes, so no position a read skips meanwhile is filled as empty, and
    // its acknowledgment is checked against the floor captured when it was
    // first taken. False when it was already re-sent or issuance stopped; the
    // caller then completes it OUTCOME_UNKNOWN. The re-send's result completes
    // the write as any attempt's does.
    bool resendWrite(uint64_t id);

    // Invalidates the old stream immediately; issued writes may still settle.
    uint64_t disconnect();
    // BUSY until issued writes settle, no queued write's outcome is unknown
    // and every committed write has a later local write with a position; no
    // global gate.
    ReadCursor startRead();

    // Retains complete validated semantic coverage BEFORE pipeline admission.
    // A read delivers in order, so a position it skipped below `first` is an
    // empty commit unless an issued write may hold it; an acknowledgment's
    // position covers nothing below it. `first..last` must be
    // independently justified (singleton, validated aggregate, or initial
    // snapshot covering 1..last without synthetic gap expansion). The actor
    // queues only nonzero returned work IDs, in semantic order. Zero means a
    // stale generation or replay already retained/applied; it must not apply
    // the payload again. Exact retained replay requires matching fingerprints.
    // Aggregate coverage [first,last] is the service's state at last, so it
    // contains every local write acknowledged inside it: an acknowledgment
    // at a position after first and at or before last, whether it arrives
    // before or after this coverage (completeWrite), is absorbed and retires
    // when this coverage applies (convergence rule 1, design section 12), as
    // does a committed write whose next local write it absorbs. An aggregate
    // starting on an acknowledged position is an overlap. Local writes between
    // two it absorbs whose acknowledgments are still to come are absorbed as
    // they arrive; any other gap in the absorbed local order is an overlap. A
    // singleton Update that repeats an acknowledged intent's issued bytes at
    // its position is a replay; every other update applies as delivered,
    // whatever its bytes.
    // Peer coverage of an earlier read that the aggregate contains, after its
    // first position, retires with it. Every other overlap stops UNSUPPORTED
    // with work retained; a re-delivered peer range must match its
    // fingerprint.
    uint64_t acceptCoverage(uint64_t generation, uint64_t object, uint64_t first, uint64_t last, Coverage data);
    // An empty header: above coverage, the positions through it are empty
    // commits, and one after the read's catch-up target names an empty commit
    // even where a local write may hold its position. A catch-up target
    // above the read's position is a net-empty catch-up: each acknowledged
    // local write and each earlier read's peer coverage it passes retires
    // without applying. Separates
    // per-generation internal readiness from the one initial external
    // OK_LATEST notification.
    ReadySignal caughtUp(uint64_t generation, uint64_t object, uint64_t sequence);
    // Complete content may establish readiness without an empty header, but
    // only after the caller has established the stream's recovery boundary.
    ReadySignal recoveryReady(uint64_t generation, uint64_t object, uint64_t sequence);

    // Called after effects are represented in the retained layer state. An ack
    // alone never frees intents. Out-of-order applied records remain retained
    // until the contiguous applied frontier covers them. Local retirement
    // advances its ordinal prefix only when acknowledged records are erased.
    // Absorbed acknowledgments apply with their coverage. Returns the owners
    // of absorbed local work this call erased; the caller completes and
    // releases them after its own locks.
    std::vector<std::shared_ptr<const void>> applied(uint64_t id);
    // Terminal failure: issuance and admission stop, pending work stays
    // retained, and the read closes, freezing the applied frontier. The first
    // cause is kept unless a later one is OUTCOME_UNKNOWN.
    void fail(OmniUsdLiveResult result);
    // Issuance-only failure (definitive rejection, host failure): as fail,
    // except that an open read stays open and may reconnect with a new
    // generation, and accepted coverage keeps applying. Readiness never
    // reopens the gate. It never reopens a read that fail already closed.
    void stopIssuance(OmniUsdLiveResult result);
    Snapshot snapshot() const;
    std::vector<Work> retained() const;
    // The ordered run of unapplied work that may apply next, at most `max`:
    // the work starting at the applied frontier plus one, then each work
    // starting right after the previous one's last position and the empty
    // commits and applied work that follow it. It ends before work that
    // retires a local write after a committed write whose intent has not yet
    // been given work to retire with, and before an aggregate of more than one
    // position that may contain the commit of a write whose acknowledgment is
    // still to come after its first position. Changes nothing.
    std::vector<Work> applicable(size_t max) const;

private:
    mutable std::mutex mutex_;
    std::shared_ptr<const void> initialState_;
    const size_t maxRetained_;
    const uint64_t maxGap_;
    const uint64_t initialApplied_;
    uint64_t localAdmitted_ = 0, localRetired_ = 0;
    Gate gate_ = Gate::Recovering;
    OmniUsdLiveResult result_ = OMNI_USD_LIVE_OK;
    uint64_t generation_ = 1, covered_, applied_, readSequence_ = 0, resume_ = 0;
    uint64_t readyGeneration_ = 0; // The read generation whose catch-up target arrived.
    uint64_t pendingWrites_ = 0, pendingBytes_ = 0;
    uint64_t object_ = 0;
    uint64_t nextId_ = 1, issued_ = 0, highestObserved_;
    bool readStarted_ = false, initialLatestPublished_ = false, readingAfterFailure_ = false;
    std::map<uint64_t, Work> work_;
    std::deque<uint64_t> queued_;
    // Position ranges above the applied frontier that are empty commits; they count as covered and applied.
    std::vector<std::pair<uint64_t, uint64_t>> empty_;

    bool stopped() const;
    uint64_t allocateId() const;
    using Ranges = std::vector<std::pair<uint64_t, uint64_t>>;
    uint64_t frontier(uint64_t base, bool onlyApplied, const Work* extra = nullptr,
        const Ranges* empty = nullptr) const;
    Ranges holes(uint64_t through, const Work* extra = nullptr) const;
    // Position ranges an unsettled local write may hold: above its floor and
    // below the next acknowledged local write of the chain.
    Ranges held() const;
    // The next local write after `work` in chain order that has a position.
    const Work* bounding(const Work& work) const;
    // Whether a committed write still waits for work to retire with.
    static bool unreleased(const Work& work);
    // Whether a queued write's outcome is unknown, or a committed write has
    // no later local write with a position.
    bool uncertain() const;
    bool issuable(size_t window) const;
    // Gives each committed write whose release condition holds the work it
    // retires with: once a read started after the proof has passed the
    // position before the next local write with a position, that write's
    // application, or the aggregate absorbing it.
    void release();
    // Records a delivered position and releases the committed writes it
    // passes.
    void advance(uint64_t sequence);
    void fill(const Ranges& holes);
    bool currentRead(uint64_t generation, uint64_t object);
    ReadySignal ready(uint64_t generation, uint64_t object, uint64_t sequence, bool header);
    // Marks every queued write whose outcome is unknown Unknown and removes it
    // from the FIFO; whether there was one.
    bool settleUncertain();
    void stop(OmniUsdLiveResult result);
    void haltIssuance(OmniUsdLiveResult result);
};
} // namespace omni::usd::live
