// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "LiveLayerProgress.h"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace omni::usd::live {
LayerProgress::LayerProgress(LayerIdentity key, uint64_t resume, std::shared_ptr<const void> initialState,
    size_t maxRetained, uint64_t maxGap, bool discover)
    : identity(std::move(key)), initialState_(std::move(initialState)), maxRetained_(maxRetained), maxGap_(maxGap),
      initialApplied_(resume), covered_(resume), applied_(resume), object_(identity.object), highestObserved_(resume) {
    if (!identity.endpoint || identity.session.empty() || identity.layer.empty() || identity.writer.empty() ||
        (!identity.object != discover) || (discover && resume) || !maxRetained || (resume && !initialState_))
        throw std::invalid_argument("invalid layer progress options");
}

bool LayerProgress::stopped() const { return gate_ == Gate::Failed; }

bool LayerProgress::settleUncertain() {
    // A queued write an earlier attempt of which has an unknown outcome is
    // never issued again once issuance stops, so its outcome stays unknown.
    bool found = false;
    for (auto& [id, entry] : work_)
        if (entry.disposition == Disposition::Queued && entry.mayHaveCommitted) {
            entry.disposition = Disposition::Unknown;
            queued_.erase(std::find(queued_.begin(), queued_.end(), id));
            found = true;
        }
    return found;
}

void LayerProgress::stop(OmniUsdLiveResult result) {
    if (settleUncertain()) result = OMNI_USD_LIVE_OUTCOME_UNKNOWN;
    if (!stopped()) {
        gate_ = Gate::Failed;
        result_ = result;
    } else if (result == OMNI_USD_LIVE_OUTCOME_UNKNOWN) {
        // An issued mutation can become uncertain after another failure. Never
        // hide that stronger preservation obligation behind the earlier error.
        result_ = result;
    }
    readStarted_ = false;
    readingAfterFailure_ = false;
}

void LayerProgress::haltIssuance(OmniUsdLiveResult result) {
    // An unknown write outcome closes the read, since its sequence is never learned.
    if (settleUncertain()) {
        stop(OMNI_USD_LIVE_OUTCOME_UNKNOWN);
        return;
    }
    const bool reading = !stopped() || readingAfterFailure_;
    const bool started = readStarted_;
    stop(result);
    readingAfterFailure_ = reading;
    readStarted_ = reading && started;
}

uint64_t LayerProgress::allocateId() const {
    if (work_.size() >= maxRetained_ || nextId_ == std::numeric_limits<uint64_t>::max())
        throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    return nextId_;
}

uint64_t LayerProgress::admit(std::shared_ptr<const void> intent, Publication publication, uint64_t retainedBytes,
    const ContentHash* fingerprint) {
    if (!intent) throw std::invalid_argument("missing retained intent");
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopped()) throw OmniUsdLiveResult(result_);
    if (!object_) throw OmniUsdLiveResult(OMNI_USD_LIVE_BUSY);
    const auto id = allocateId();
    if (retainedBytes > UINT64_MAX - pendingBytes_ || pendingWrites_ == UINT64_MAX || localAdmitted_ == UINT64_MAX)
        throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    // Both allocations must succeed before this edit becomes accepted.
    auto entry = work_.emplace(id, Work{id, 0, 0, Disposition::Queued, false, intent});
    try {
        queued_.push_back(id);
    } catch (...) {
        work_.erase(entry.first);
        throw;
    }
    entry.first->second.retainedBytes = retainedBytes;
    if (fingerprint) {
        entry.first->second.fingerprint = *fingerprint;
        entry.first->second.fingerprinted = true;
    }
    entry.first->second.localOrdinal = ++localAdmitted_;
    ++pendingWrites_;
    pendingBytes_ += retainedBytes;
    ++nextId_;
    if (publication.commit) publication.commit(publication.context, id);
    return id;
}

bool LayerProgress::unreleased(const Work& work) {
    return work.disposition == Disposition::Committed && !work.absorbedBy;
}

const LayerProgress::Work* LayerProgress::bounding(const Work& work) const {
    // Local ordinals increase with work IDs.
    for (auto it = work_.upper_bound(work.id); it != work_.end(); ++it)
        if (it->second.localOrdinal && it->second.first) return &it->second;
    return nullptr;
}

bool LayerProgress::uncertain() const {
    // A committed write is bounded once a later local write has a position;
    // settling ends only then, so the read that releases it can resume.
    return std::any_of(work_.begin(), work_.end(), [&](const auto& item) {
        return (item.second.disposition == Disposition::Queued && item.second.mayHaveCommitted) ||
            (unreleased(item.second) && !bounding(item.second));
    });
}

bool LayerProgress::issuable(size_t window) const {
    // While a write's outcome is unknown the read stays closed, and writes
    // issue to settle it (design section 12).
    return issued_ < window &&
        (gate_ == Gate::Ready || (gate_ == Gate::Recovering && object_ && !readStarted_ && uncertain()));
}

LayerProgress::Work LayerProgress::takeWrite(size_t window) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!issuable(window)) return {};
    // The FIFO holds every unsettled write in admission order; the writes in
    // flight precede the first queued one except while a refused write waits
    // to be issued again ahead of them.
    const auto next = std::find_if(queued_.begin(), queued_.end(),
        [&](uint64_t id) { return work_.at(id).disposition == Disposition::Queued; });
    if (next == queued_.end()) return {};
    auto& entry = work_.at(*next);
    entry.disposition = Disposition::Issued;
    // A write's position is above the highest position observed when it was
    // first taken; a later attempt cannot narrow that, since an earlier
    // attempt may have committed first.
    if (!entry.attempts) entry.floor = highestObserved_;
    ++entry.attempts;
    ++issued_;
    // Keep the entry until completion, making pre-issuance deferral allocation-free.
    return entry;
}

LayerProgress::Work LayerProgress::takeWriteOutOfOrder(uint64_t id, size_t window) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto entry = work_.find(id);
    if (!issuable(window) || entry == work_.end() || entry->second.disposition != Disposition::Queued) return {};
    entry->second.disposition = Disposition::Issued;
    if (!entry->second.attempts) entry->second.floor = highestObserved_;
    ++entry->second.attempts;
    ++issued_;
    return entry->second;
}

void LayerProgress::clearUncertain(uint64_t id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto entry = work_.find(id);
    if (entry == work_.end() || entry->second.disposition != Disposition::Queued) return;
    entry->second.mayHaveCommitted = false;
    try {
        fill(holes(readSequence_));
        release();
    } catch (const OmniUsdLiveResult& result) { stop(result); }
}

void LayerProgress::deferWrite(uint64_t id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto entry = work_.find(id);
    if (!id || entry == work_.end() || entry->second.disposition != Disposition::Issued)
        throw std::invalid_argument("write is not in flight");
    entry->second.disposition = Disposition::Queued;
    --entry->second.attempts;
    --issued_;
    // The write never reached transport, so the positions the read skipped
    // while it was taken hold nothing of it.
    try {
        fill(holes(readSequence_));
    } catch (const OmniUsdLiveResult& result) { stop(result); }
}

bool LayerProgress::requeueWrite(uint64_t id, bool mayHaveCommitted) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto entry = work_.find(id);
    if (!id || entry == work_.end() || entry->second.disposition != Disposition::Issued)
        throw std::invalid_argument("write is not in flight");
    if (stopped()) return false;
    entry->second.disposition = Disposition::Queued;
    entry->second.mayHaveCommitted |= mayHaveCommitted;
    --issued_;
    // A write whose attempts all applied nothing holds no position.
    try {
        fill(holes(readSequence_));
    } catch (const OmniUsdLiveResult& result) { stop(result); }
    return true;
}

void LayerProgress::confirmCommitted(uint64_t id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto entry = work_.find(id);
    if (!id || entry == work_.end() || entry->second.disposition != Disposition::Queued || !entry->second.attempts)
        throw std::invalid_argument("write was never issued");
    queued_.erase(std::find(queued_.begin(), queued_.end(), id));
    entry->second.disposition = Disposition::Committed;
    entry->second.mayHaveCommitted = true;
    // A read open at the proof may have filtered the commit; only a read started
    // after the proof is known to deliver it.
    entry->second.readAfter = readStarted_ && generation_ < std::numeric_limits<uint64_t>::max() ? generation_ + 1 :
                                                                                                   generation_;
}

LayerProgress::Ranges LayerProgress::held() const {
    // One pass in each direction over local work in chain order (local
    // ordinals increase with work IDs): the last position acknowledged
    // before each write, and the first acknowledged after it.
    Ranges result;
    std::vector<std::pair<const Work*, uint64_t>> holders;
    uint64_t before = 0;
    for (const auto& [id, entry] : work_) {
        if (!entry.localOrdinal) continue;
        const bool unsettled = entry.disposition == Disposition::Issued || unreleased(entry) ||
            (entry.disposition == Disposition::Queued && entry.mayHaveCommitted);
        if (unsettled) holders.emplace_back(&entry, std::max(entry.floor, before));
        if (entry.first) before = std::max(before, entry.last);
    }
    if (holders.empty()) return result;
    result.reserve(holders.size());
    uint64_t after = std::numeric_limits<uint64_t>::max();
    auto holder = holders.rbegin();
    for (auto it = work_.rbegin(); it != work_.rend() && holder != holders.rend(); ++it) {
        if (!it->second.localOrdinal) continue;
        if (&it->second == holder->first) {
            if (after > holder->second + 1) result.emplace_back(holder->second + 1, after - 1);
            ++holder;
        }
        if (it->second.first) after = std::min(after, it->second.first);
    }
    std::sort(result.begin(), result.end());
    return result;
}

uint64_t LayerProgress::frontier(uint64_t base, bool onlyApplied, const Work* extra, const Ranges* empty) const {
    // Bounded by maxRetained, not sequence distance. Work stays immutable during
    // this calculation, and allocations precede publication of a new frontier.
    Ranges ranges;
    ranges.reserve(work_.size() + (extra ? 1 : 0) + empty_.size() + (empty ? empty->size() : 0));
    auto append = [&](const Work& entry) {
        if (entry.first && (!onlyApplied || entry.applied)) ranges.emplace_back(entry.first, entry.last);
    };
    for (const auto& entry : work_) append(entry.second);
    if (extra) append(*extra);
    // An empty commit is covered and has nothing to apply.
    ranges.insert(ranges.end(), empty_.begin(), empty_.end());
    if (empty) ranges.insert(ranges.end(), empty->begin(), empty->end());
    std::sort(ranges.begin(), ranges.end());
    for (const auto& range : ranges) {
        if (range.second <= base) continue;
        if (base == std::numeric_limits<uint64_t>::max() || range.first > base + 1) break;
        base = range.second;
    }
    return base;
}

namespace {
using PositionRanges = std::vector<std::pair<uint64_t, uint64_t>>;
// The parts of each of `ranges` outside every range of `remove`; both sorted.
PositionRanges subtract(const PositionRanges& ranges, const PositionRanges& remove) {
    PositionRanges result;
    result.reserve(ranges.size() + remove.size());
    for (auto [first, last] : ranges) {
        for (const auto& [from, to] : remove) {
            if (to < first || from > last) continue;
            if (from > first) result.emplace_back(first, from - 1);
            if (to >= last) {
                first = 1;
                last = 0;
                break;
            }
            first = to + 1;
        }
        if (first <= last) result.emplace_back(first, last);
    }
    return result;
}
} // namespace

// gRPC delivers a read in order, so a position the read skips is an empty
// commit, except one an unsettled local write may hold: the read never
// carries this writer's own commits. `through` is a position the read
// delivered. Returns the positions through it that no retained work, `extra`,
// known empty commit or unsettled write holds, or none before an incarnation
// is adopted. A hole wider than the synthetic gap bound fails like an
// oversized aggregate.
LayerProgress::Ranges LayerProgress::holes(uint64_t through, const Work* extra) const {
    Ranges result;
    if (!object_ || through <= covered_ || (stopped() && !readingAfterFailure_)) return result;
    Ranges taken(empty_);
    taken.reserve(work_.size() + empty_.size() + 1);
    for (const auto& item : work_)
        if (item.second.first) taken.emplace_back(item.second.first, item.second.last);
    if (extra && extra->first) taken.emplace_back(extra->first, extra->last);
    std::sort(taken.begin(), taken.end());
    auto next = covered_ + 1;
    bool open = true;
    for (const auto& [first, last] : taken) {
        if (next > through) break;
        if (last < next) continue;
        if (first > next) result.emplace_back(next, std::min(first - 1, through));
        if (last == std::numeric_limits<uint64_t>::max()) {
            open = false;
            break;
        }
        next = std::max(next, last + 1);
    }
    if (open && next <= through) result.emplace_back(next, through);
    if (issued_ || std::any_of(work_.begin(), work_.end(), [](const auto& item) {
        return unreleased(item.second) ||
            (item.second.disposition == Disposition::Queued && item.second.mayHaveCommitted);
    }))
        result = subtract(result, held());
    for (const auto& [first, last] : result)
        if (last - first >= maxGap_) throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    return result;
}

// A committed write's commit is ordinary coverage of a read started after
// the chain proved it, at a position no later than the one before the next
// local write with a position. Once such a read has passed that position, the
// write retires with that next write's application, or with the aggregate
// absorbing it, and holds no position any more. Its position is never
// inferred (design section 12).
void LayerProgress::release() {
    bool released = false;
    for (auto& [id, entry] : work_) {
        if (!unreleased(entry) || !readStarted_ || generation_ < entry.readAfter) continue;
        const auto* next = bounding(entry);
        if (!next || readSequence_ + 1 < next->first) continue;
        auto& carrier = work_.at(next->absorbedBy ? next->absorbedBy : next->id);
        entry.absorbedBy = carrier.id;
        ++carrier.absorbedLocal;
        if (carrier.localOrdinal) carrier.absorbedThrough = carrier.localOrdinal - 1;
        released = true;
    }
    if (released) fill(holes(readSequence_));
}

void LayerProgress::advance(uint64_t sequence) {
    readSequence_ = sequence;
    try {
        release();
    } catch (const OmniUsdLiveResult& result) {
        stop(result);
        throw;
    }
}

void LayerProgress::fill(const Ranges& holes) {
    if (holes.empty()) return;
    const auto covered = frontier(covered_, false, nullptr, &holes);
    const auto applied = frontier(applied_, true, nullptr, &holes);
    empty_.reserve(empty_.size() + holes.size());
    empty_.insert(empty_.end(), holes.begin(), holes.end());
    covered_ = covered;
    applied_ = applied;
    empty_.erase(
        std::remove_if(empty_.begin(), empty_.end(), [&](const auto& range) { return range.second <= applied_; }),
        empty_.end());
}

LayerProgress::WriteSettlement LayerProgress::completeWrite(uint64_t id, OmniUsdLiveResult result, uint64_t sequence) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto found = work_.find(id);
    if (!id || found == work_.end() || found->second.disposition != Disposition::Issued)
        throw std::invalid_argument("write is not in flight");
    auto& entry = found->second;
    // A write is re-sent at most once, so an UNAUTHENTICATED re-send, or one
    // refused with nothing applied, earns no reissue; its first attempt's
    // outcome stays unknown.
    if (entry.resent && (result == OMNI_USD_LIVE_UNAUTHENTICATED || result == OMNI_USD_LIVE_UNAVAILABLE))
        result = OMNI_USD_LIVE_OUTCOME_UNKNOWN;
    if (result == OMNI_USD_LIVE_UNAUTHENTICATED && !entry.authReissued && !stopped()) {
        // The service authenticates before it executes the update, so this
        // attempt never ran. Keep the intent at its FIFO position for exactly
        // one reissue with a refreshed token generation; that reissue is not a
        // retry of an executed mutation (design section 13).
        entry.disposition = Disposition::Queued;
        --issued_;
        // The attempt holds no position, so the holes below the read's
        // position held for it fill here, since the read may stay quiet
        // throughout the auth wait. A hole wider than the gap bound fails the
        // layer, and the write settles below without the reissue.
        try {
            fill(holes(readSequence_));
        } catch (const OmniUsdLiveResult& failure) {
            stop(failure);
            entry.disposition = Disposition::Issued;
            ++issued_;
        }
        if (!stopped()) {
            entry.authReissued = true;
            entry.refusals = 0;
            return WriteSettlement::AwaitAuth;
        }
    }
    // A write an earlier attempt of which may have committed, and may still
    // commit, keeps that outcome when a later attempt ends its issues
    // without a definitive rejection: a second UNAUTHENTICATED, or the
    // refusal that reaches the reissue limit.
    if (entry.mayHaveCommitted && (result == OMNI_USD_LIVE_UNAUTHENTICATED || result == OMNI_USD_LIVE_UNAVAILABLE))
        result = OMNI_USD_LIVE_OUTCOME_UNKNOWN;
    uint64_t absorber = 0;
    if (result == OMNI_USD_LIVE_OK) {
        // Positions at/below the captured baseline cannot be this new write.
        // Nor may distinct local writes or already accepted peer coverage claim
        // a position for an issued write without explicit reconciliation. The
        // writer's chain commits in admission order, so an earlier local write
        // holds a lower position and a later one a higher position. An
        // unapplied aggregate is the service's state at its last position, so
        // a position strictly after its first one is this write's commit
        // inside it: the write is absorbed and retires when the aggregate
        // applies, as when the acknowledgment arrives first (design section 12).
        bool collision = sequence <= applied_ || sequence <= entry.floor;
        for (const auto& item : work_) {
            const auto& other = item.second;
            if (other.first && other.first <= sequence && other.last >= sequence) {
                if (other.disposition == Disposition::Peer && other.kind == CoverageKind::Aggregate &&
                    other.first < sequence && !other.applied && !other.absorbedBy && !other.netted && !absorber)
                    absorber = other.id;
                else collision = true;
            }
            if (other.first && other.localOrdinal && other.id != id)
                collision |= other.localOrdinal < entry.localOrdinal ? other.last >= sequence : other.first <= sequence;
        }
        for (const auto& [first, last] : empty_) collision |= first <= sequence && last >= sequence;
        if (!sequence || collision) result = OMNI_USD_LIVE_OUTCOME_UNKNOWN;
    }
    if (result == OMNI_USD_LIVE_OK) {
        auto confirmed = entry;
        confirmed.first = confirmed.last = sequence;
        confirmed.disposition = Disposition::Acknowledged;
        // If allocation fails, the in-flight entry/intent remains intact. The
        // actor must retry this local bookkeeping, never the mutation RPC.
        const auto covered = frontier(covered_, false, &confirmed);
        if (absorber) {
            auto& carrier = work_.at(absorber);
            confirmed.absorbedBy = absorber;
            ++carrier.absorbedLocal;
            carrier.absorbedThrough = std::max(carrier.absorbedThrough, confirmed.localOrdinal);
        }
        entry = std::move(confirmed);
        covered_ = covered;
        highestObserved_ = std::max(highestObserved_, sequence);
    } else {
        if (result == OMNI_USD_LIVE_OUTCOME_UNKNOWN) {
            entry.disposition = Disposition::Unknown;
            stop(result);
        } else {
            entry.disposition = Disposition::Rejected;
            // A definitive rejection of a later attempt says nothing of an
            // earlier attempt whose outcome is unknown: that attempt may
            // have committed at a position the read skipped, which is never
            // learned, so the read closes as for an unknown outcome and no
            // hole fill or journal application crosses that position
            // (design section 13).
            if (entry.resent || entry.mayHaveCommitted) stop(result);
            else haltIssuance(result);
        }
    }
    queued_.erase(std::find(queued_.begin(), queued_.end(), id));
    --issued_;
    // Holes below the read's position are empty once no write can hold them. An
    // acknowledgment proves nothing about the positions below it: the read may
    // not have delivered them yet.
    try {
        fill(holes(readSequence_));
        release();
    } catch (const OmniUsdLiveResult& result) { stop(result); }
    return WriteSettlement::Settled;
}

uint32_t LayerProgress::refuseWrite(uint64_t id, uint32_t limit) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto found = work_.find(id);
    if (!id || found == work_.end() || found->second.disposition != Disposition::Issued)
        throw std::invalid_argument("write is not in flight");
    auto& entry = found->second;
    if (entry.resent || stopped() || entry.refusals + 1 >= limit) return 0;
    // The service applied nothing, so the attempt holds no position. Keep the
    // intent at its FIFO position for its reissue; that reissue is not a retry
    // of an executed mutation (design section 13).
    entry.disposition = Disposition::Queued;
    --issued_;
    // Holes below the read's position held for this attempt are empty and
    // fill here, since the read may stay quiet throughout the backoff. A hole
    // wider than the gap bound fails the layer, and the write stays in flight
    // for the caller to settle with the refusal's result.
    try {
        fill(holes(readSequence_));
    } catch (const OmniUsdLiveResult& result) {
        stop(result);
        entry.disposition = Disposition::Issued;
        ++issued_;
        return 0;
    }
    return ++entry.refusals;
}

bool LayerProgress::resendWrite(uint64_t id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto found = work_.find(id);
    if (!id || found == work_.end() || found->second.disposition != Disposition::Issued)
        throw std::invalid_argument("write is not in flight");
    auto& entry = found->second;
    if (entry.resent || stopped()) return false;
    // The service answers a re-send of a committed operation with that
    // commit's sequence, which the read skipped and may have passed, so the
    // write keeps its issue and its floor.
    entry.resent = true;
    return true;
}

uint64_t LayerProgress::disconnect() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopped() && !readingAfterFailure_) return generation_;
    if (generation_ == std::numeric_limits<uint64_t>::max()) {
        stop(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
        return generation_;
    }
    ++generation_;
    if (!stopped()) gate_ = Gate::Recovering;
    readStarted_ = false;
    return generation_;
}

LayerProgress::ReadCursor LayerProgress::startRead() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopped() && !readingAfterFailure_) throw OmniUsdLiveResult(result_);
    // A read started while a write's outcome is unknown could repeat that
    // write's commit in its catch-up, which names no writer.
    if (issued_ || readStarted_ || uncertain() || (!stopped() && gate_ != Gate::Recovering))
        throw OmniUsdLiveResult(OMNI_USD_LIVE_BUSY);
    readStarted_ = true;
    resume_ = readSequence_ = covered_;
    return {generation_, object_, resume_};
}

bool LayerProgress::currentRead(uint64_t generation, uint64_t object) {
    if ((stopped() && !readingAfterFailure_) || generation != generation_ || !readStarted_) return false;
    if (!object) {
        stop(OMNI_USD_LIVE_MALFORMED_STREAM);
        throw OmniUsdLiveResult(OMNI_USD_LIVE_MALFORMED_STREAM);
    }
    if (object_ && object != object_) {
        stop(OMNI_USD_LIVE_OBJECT_REPLACED);
        throw OmniUsdLiveResult(OMNI_USD_LIVE_OBJECT_REPLACED);
    }
    return true;
}

uint64_t LayerProgress::acceptCoverage(uint64_t generation, uint64_t object, uint64_t first, uint64_t last,
    Coverage data) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!currentRead(generation, object)) return 0;
    if (!data.owner || !first || first > last || last <= readSequence_ || first <= resume_) {
        stop(OMNI_USD_LIVE_MALFORMED_STREAM);
        throw OmniUsdLiveResult(OMNI_USD_LIVE_MALFORMED_STREAM);
    }
    const bool snapshot = data.kind == CoverageKind::Snapshot;
    // A discovery stream must establish the complete initial history before
    // adopting its incarnation. Empty/high-water headers cannot do this.
    if (!object_ && (first != 1 || (!snapshot && (last != 1 || data.kind != CoverageKind::Update)))) {
        stop(OMNI_USD_LIVE_MALFORMED_STREAM);
        throw OmniUsdLiveResult(OMNI_USD_LIVE_MALFORMED_STREAM);
    }
    if ((snapshot && (first != 1 || resume_ || readSequence_ || covered_)) ||
        (data.kind == CoverageKind::Update && first != last)) {
        stop(OMNI_USD_LIVE_MALFORMED_STREAM);
        throw OmniUsdLiveResult(OMNI_USD_LIVE_MALFORMED_STREAM);
    }
    if (!snapshot && last - first > maxGap_) {
        stop(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
        throw OmniUsdLiveResult(OMNI_USD_LIVE_RESOURCE_EXHAUSTED);
    }
    if (last <= applied_) {
        // This incarnation's immutable sequence positions are already present
        // in the layer state. They can replay while an older stream catches up.
        advance(last);
        return 0;
    }
    std::vector<std::pair<uint64_t, uint64_t>> absorbed; // (localOrdinal, id); allocated before any mutation.
    // Peer coverage an earlier read delivered above this read's resume point
    // waits behind a hole. An aggregate of this read that contains it holds
    // its effect, so it retires with the aggregate, as do the local writes it
    // absorbed.
    std::vector<uint64_t> superseded;
    if (data.kind == CoverageKind::Aggregate)
        for (const auto& [id, previous] : work_)
            if (previous.disposition == Disposition::Peer && !previous.absorbedBy && previous.first > readSequence_ &&
                first < previous.first && previous.last <= last)
                superseded.push_back(id);
    const auto supersedes = [&](uint64_t id) {
        return std::find(superseded.begin(), superseded.end(), id) != superseded.end();
    };
    for (const auto& item : work_) {
        const auto& previous = item.second;
        if (previous.absorbedBy && supersedes(previous.absorbedBy)) {
            if (previous.localOrdinal) absorbed.emplace_back(previous.localOrdinal, previous.id);
            continue;
        }
        if (!previous.first || previous.last < first || previous.first > last || supersedes(previous.id)) continue;
        if (previous.disposition == Disposition::Peer && previous.first == first && previous.last == last) {
            if (previous.fingerprint != data.fingerprint || previous.kind != data.kind) {
                stop(OMNI_USD_LIVE_MALFORMED_STREAM);
                throw OmniUsdLiveResult(OMNI_USD_LIVE_MALFORMED_STREAM);
            }
            advance(last);
            return 0;
        }
        if (previous.disposition == Disposition::Acknowledged && data.kind == CoverageKind::Update && first == last &&
            previous.first == first && previous.fingerprinted && previous.fingerprint == data.fingerprint) {
            // A catch-up re-sent this layer's own acknowledged write. The
            // retained intent carries the same bytes and retires through its
            // own ordered application, which may already be in flight.
            advance(last);
            return 0;
        }
        // Aggregate coverage [first,last] is the service's state at last, so it
        // contains the effect of every write acknowledged inside it. Only an
        // acknowledgment strictly after first is absorbed: one at first would
        // be ready at the same applied position as the aggregate and apply
        // alone, leaving the aggregate behind the applied frontier.
        const bool contained = first < previous.first && previous.last <= last;
        if (previous.disposition != Disposition::Acknowledged || previous.applied || previous.absorbedBy ||
            !contained || data.kind != CoverageKind::Aggregate) {
            stop(OMNI_USD_LIVE_UNSUPPORTED);
            throw OmniUsdLiveResult(OMNI_USD_LIVE_UNSUPPORTED);
        }
        absorbed.emplace_back(previous.localOrdinal, previous.id);
    }
    // The aggregate holds the commit of a committed write whose next local
    // write it absorbs, when this read started after the proof, and the
    // effect of every local write that retires with a write it absorbs.
    if (data.kind == CoverageKind::Aggregate) {
        const auto direct = absorbed.size();
        const auto holds = [&](uint64_t id, size_t count) {
            return std::any_of(absorbed.begin(), absorbed.begin() + std::ptrdiff_t(count),
                [&](const auto& pair) { return pair.second == id; });
        };
        for (const auto& [id, entry] : work_) {
            if (unreleased(entry)) {
                const auto* next = bounding(entry);
                if (generation_ >= entry.readAfter && next && holds(next->id, direct))
                    absorbed.emplace_back(entry.localOrdinal, id);
            } else if (entry.localOrdinal && entry.absorbedBy && !holds(id, direct) &&
                holds(entry.absorbedBy, direct)) {
                absorbed.emplace_back(entry.localOrdinal, id);
            }
        }
    }
    std::sort(absorbed.begin(), absorbed.end());
    // Between two local writes it absorbs, the aggregate holds every local
    // write's commit, since the chain commits in order: a gap in their local
    // ordinals is only a write whose acknowledgment is still to come, which
    // completeWrite absorbs when it arrives.
    const auto awaited = [&](uint64_t ordinal) {
        return std::any_of(work_.begin(), work_.end(), [&](const auto& item) {
            const auto& entry = item.second;
            return entry.localOrdinal == ordinal &&
                (entry.disposition == Disposition::Issued ||
                    (entry.disposition == Disposition::Queued && entry.mayHaveCommitted));
        });
    };
    for (size_t i = 1; i < absorbed.size(); ++i) {
        bool contiguous = absorbed[i].first > absorbed[i - 1].first;
        for (auto ordinal = absorbed[i - 1].first + 1; contiguous && ordinal < absorbed[i].first; ++ordinal)
            contiguous = awaited(ordinal);
        if (!contiguous) {
            stop(OMNI_USD_LIVE_UNSUPPORTED);
            throw OmniUsdLiveResult(OMNI_USD_LIVE_UNSUPPORTED);
        }
    }
    if (first <= applied_) {
        stop(OMNI_USD_LIVE_UNSUPPORTED);
        throw OmniUsdLiveResult(OMNI_USD_LIVE_UNSUPPORTED);
    }
    const auto id = allocateId();
    Work entry{id, first, last, Disposition::Peer, false, data.owner, data.fingerprint, data.kind};
    entry.absorbedLocal = absorbed.size();
    if (!absorbed.empty()) entry.absorbedThrough = absorbed.back().first;
    Ranges empty;
    try {
        empty = holes(last, &entry);
    } catch (const OmniUsdLiveResult& result) {
        stop(result);
        throw;
    }
    const auto covered = frontier(covered_, false, &entry, &empty);
    const auto applied = frontier(applied_, true, nullptr, &empty);
    empty_.reserve(empty_.size() + empty.size());
    work_.emplace(id, std::move(entry));
    for (const auto& pair : absorbed) work_.at(pair.second).absorbedBy = id;
    for (const auto previous : superseded) work_.at(previous).absorbedBy = id;
    if (!object_) object_ = object; // Publication follows all fallible admission.
    ++nextId_;
    empty_.insert(empty_.end(), empty.begin(), empty.end());
    covered_ = covered;
    applied_ = applied;
    highestObserved_ = std::max(highestObserved_, last);
    advance(last);
    return id;
}

LayerProgress::ReadySignal LayerProgress::ready(uint64_t generation, uint64_t object, uint64_t sequence, bool header) {
    if (!currentRead(generation, object)) return {};
    if (!sequence || sequence < readSequence_) {
        stop(OMNI_USD_LIVE_MALFORMED_STREAM);
        throw OmniUsdLiveResult(OMNI_USD_LIVE_MALFORMED_STREAM);
    }
    // An empty header above coverage, a caught-up header with nothing owed or an
    // empty commit, makes the positions through it empty, unless an issued write
    // may hold one; those fill once it settles. A header-only frame after the
    // read's catch-up target names an empty commit even where an unsettled
    // write may hold its position, since the read never carries this writer's
    // commits made while it is open (design section 3).
    const auto named = [&] {
        return std::any_of(work_.begin(), work_.end(),
                   [&](const auto& item) {
            return item.second.first && item.second.first <= sequence && item.second.last >= sequence;
        }) ||
            std::any_of(empty_.begin(), empty_.end(),
                [&](const auto& range) { return range.first <= sequence && range.second >= sequence; });
    };
    try {
        if (header && readyGeneration_ == generation && object_ && sequence > readSequence_ && sequence > covered_ &&
            !named())
            fill({{sequence, sequence}});
        fill(holes(sequence));
    } catch (const OmniUsdLiveResult& result) {
        stop(result);
        throw;
    }
    if (sequence > covered_ && held().empty()) {
        // Before an incarnation is adopted, only complete content establishes it.
        stop(OMNI_USD_LIVE_MALFORMED_STREAM);
        throw OmniUsdLiveResult(OMNI_USD_LIVE_MALFORMED_STREAM);
    }
    highestObserved_ = std::max(highestObserved_, sequence);
    // A catch-up target above the read's position delivers no commit through
    // it: the commits there net to nothing, so every unapplied coverage among
    // them retires without applying: an acknowledged local write, a committed
    // write that retires with it, and peer coverage an earlier read delivered
    // above this read's resume point, behind a position held for a write.
    if (header && readyGeneration_ != generation)
        for (auto& [id, entry] : work_)
            if (!entry.applied && !entry.absorbedBy && entry.first > readSequence_ && entry.first <= sequence &&
                (entry.disposition == Disposition::Peer ||
                    (entry.localOrdinal && entry.disposition == Disposition::Acknowledged)))
                entry.netted = true;
    advance(sequence);
    readyGeneration_ = generation;
    if (stopped()) return {true, false}; // Reading after failure never reopens issuance.
    gate_ = Gate::Ready;
    const bool publish = !initialLatestPublished_;
    initialLatestPublished_ = true;
    return {true, publish};
}

LayerProgress::ReadySignal LayerProgress::caughtUp(uint64_t generation, uint64_t object, uint64_t sequence) {
    std::lock_guard<std::mutex> lock(mutex_);
    return ready(generation, object, sequence, true);
}

LayerProgress::ReadySignal LayerProgress::recoveryReady(uint64_t generation, uint64_t object, uint64_t sequence) {
    std::lock_guard<std::mutex> lock(mutex_);
    return ready(generation, object, sequence, false);
}

std::vector<std::shared_ptr<const void>> LayerProgress::applied(uint64_t id) {
    // Final release can enter endpoint/payload cleanup. Never do it while
    // holding this controller's mutex, including exceptional paths.
    std::vector<Work> retired;
    std::vector<std::shared_ptr<const void>> absorbed;
    std::unique_lock<std::mutex> lock(mutex_);
    auto it = work_.find(id);
    if (it == work_.end() || !it->second.first) throw std::invalid_argument("work has no confirmed coverage");
    auto updated = it->second;
    updated.applied = true;
    const auto applied = frontier(applied_, true, &updated);
    // Service coverage and local admission order are different axes. A later
    // ACK may be marked applied while an earlier coverage hole keeps it alive.
    // Only the set erased at the prospective contiguous frontier can advance
    // the local ledger. Preflight that whole set before any state mutation.
    auto localRetired = localRetired_;
    size_t absorbing = 0;
    for (const auto& item : work_) {
        const auto& work = item.second;
        absorbing += work.absorbedBy != 0;
        if ((work.applied || work.id == id || work.absorbedBy == id) && work.last <= applied &&
            work.disposition != Disposition::Peer) {
            if ((work.disposition != Disposition::Acknowledged &&
                    (work.disposition != Disposition::Committed || !work.absorbedBy)) ||
                localRetired == UINT64_MAX || work.localOrdinal != localRetired + 1)
                throw std::logic_error("noncontiguous local acknowledgment retirement");
            ++localRetired;
        }
    }
    retired.reserve(work_.size());
    absorbed.reserve(absorbing);
    it->second.applied = true;
    // An absorbed acknowledgment's effect is part of its coverage's effect.
    for (auto& item : work_)
        if (item.second.absorbedBy == id) item.second.applied = true;
    applied_ = applied;
    localRetired_ = localRetired;
    empty_.erase(
        std::remove_if(empty_.begin(), empty_.end(), [&](const auto& range) { return range.second <= applied_; }),
        empty_.end());
    for (auto iter = work_.begin(); iter != work_.end();) {
        if (iter->second.applied && iter->second.last <= applied_) {
            if (iter->second.disposition != Disposition::Peer) {
                --pendingWrites_;
                pendingBytes_ -= iter->second.retainedBytes;
            }
            if (iter->second.absorbedBy) absorbed.push_back(std::move(iter->second.owner));
            retired.push_back(std::move(iter->second));
            iter = work_.erase(iter);
        } else ++iter;
    }
    lock.unlock();
    return absorbed;
}

void LayerProgress::fail(OmniUsdLiveResult result) {
    if (result == OMNI_USD_LIVE_OK) throw std::invalid_argument("failure must be non-success");
    std::lock_guard<std::mutex> lock(mutex_);
    stop(result);
}

void LayerProgress::stopIssuance(OmniUsdLiveResult result) {
    if (result == OMNI_USD_LIVE_OK) throw std::invalid_argument("failure must be non-success");
    std::lock_guard<std::mutex> lock(mutex_);
    haltIssuance(result);
}

LayerProgress::Snapshot LayerProgress::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return {gate_, result_, generation_, covered_, applied_, readSequence_, issued_, work_.size(),
        queued_.size() - size_t(issued_), readStarted_, initialLatestPublished_, pendingWrites_, pendingBytes_, object_,
        initialApplied_, localAdmitted_, localRetired_, readingAfterFailure_,
        uint64_t(std::count_if(work_.begin(), work_.end(),
            [](const auto& item) { return item.second.disposition == Disposition::Committed; })),
        uint64_t(std::count_if(work_.begin(), work_.end(),
            [&](const auto& item) {
        return (item.second.disposition == Disposition::Queued && item.second.mayHaveCommitted) ||
            (unreleased(item.second) && !bounding(item.second));
    })),
        uint64_t(std::count_if(work_.begin(), work_.end(), [&](const auto& item) {
        return unreleased(item.second) && readStarted_ && generation_ < item.second.readAfter;
    }))};
}

uint64_t LayerProgress::objectId() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return object_;
}

std::vector<LayerProgress::Work> LayerProgress::retained() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<Work> result;
    result.reserve(work_.size());
    for (const auto& entry : work_) result.push_back(entry.second);
    return result;
}

std::vector<LayerProgress::Work> LayerProgress::applicable(size_t max) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<Work> result;
    // Local intents retire in chain order, so nothing retires a write after
    // a committed write that has no work to retire with yet.
    uint64_t blocked = std::numeric_limits<uint64_t>::max();
    for (const auto& [id, entry] : work_)
        if (unreleased(entry)) blocked = std::min(blocked, entry.localOrdinal);
    // The positions an acknowledgment still to come may name: above the
    // write's floor and the last earlier local position, and below the next
    // later local write with a position. An aggregate with such a position
    // after its first one may contain that write's commit, which retires the
    // write only through the aggregate's application, so it applies once the
    // write settles.
    Ranges awaited;
    uint64_t before = 0;
    for (const auto& [id, entry] : work_) {
        if (!entry.localOrdinal) continue;
        if (entry.disposition == Disposition::Issued ||
            (entry.disposition == Disposition::Queued && entry.mayHaveCommitted)) {
            const auto* next = bounding(entry);
            const auto low = std::max(entry.floor, before);
            const auto high = next ? next->first : std::numeric_limits<uint64_t>::max();
            if (high > low && high - low > 1) awaited.emplace_back(low + 1, high - 1);
        }
        if (entry.first) before = std::max(before, entry.last);
    }
    const auto awaits = [&](const Work& work) {
        return work.kind == CoverageKind::Aggregate && work.first < work.last &&
            std::any_of(awaited.begin(), awaited.end(),
                [&](const auto& range) { return range.first <= work.last && range.second > work.first; });
    };
    auto position = applied_;
    while (result.size() < max && position != std::numeric_limits<uint64_t>::max()) {
        const auto next = std::find_if(work_.begin(), work_.end(),
            [&](const auto& entry) { return entry.second.first == position + 1 && !entry.second.applied; });
        if (next == work_.end()) break;
        const auto& work = next->second;
        if (std::max(work.localOrdinal, work.absorbedLocal ? work.absorbedThrough : uint64_t(0)) > blocked ||
            awaits(work))
            break;
        result.push_back(next->second);
        // The frontier once this work applies: its range, then the empty
        // commits and applied work contiguous with it.
        position = frontier(next->second.last, true);
    }
    return result;
}
} // namespace omni::usd::live
