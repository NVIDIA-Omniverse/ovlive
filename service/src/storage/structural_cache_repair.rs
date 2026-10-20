// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! The structural cache's stream repair: bring a cached decoded state current
//! by replaying the layer-stream entries it missed, accept the result only on
//! an exact `(seq, commit_chain)` match, and otherwise name why the entry must
//! be read again (structural-edits SDD SR-4.1, SR-4.2, SR-4.2a, SR-3.2).
//!
//! Section references: `SR-n.m` is the structural-edits spec SDD (Spec Bee
//! `b.xes`); `§n.m` is `docs-internal/sdd.md`.
//!
//! # The loop (SR-4.1)
//!
//! Each round is one `structural_cache_sync.lua` read
//! ([`read_structural_cache_sync`]) from the entry's `seq`: the header's facts,
//! the call's commit-token record, and a chunk of at most
//! [`STRUCTURAL_REPAIR_MAX_ENTRIES`] (256) stream entries under the read's
//! byte bound. The first read asks at most
//! [`STRUCTURAL_REPAIR_INITIAL_ENTRIES`] entries, the later reads up to
//! the 256, and the first read's byte bound is
//! [`STRUCTURAL_REPAIR_INITIAL_BYTES`] (256 KiB): its RP0 is what every hit
//! holds before it learns what is pending, so both its bounds are kept to a
//! reply priced within 1 MiB. When the first pending entry alone is larger (the script's
//! `TOO_LARGE`) and within [`STRUCTURAL_REPAIR_MAX_BYTES`] (4 MiB), the next
//! read's bound grows to that entry's size; above that cap the repair reloads
//! `too_large`. When a chunk stops at its byte bound with entries still
//! pending (fewer than the read's entry count and short of the range's end),
//! the next read's bound doubles, up to that cap, so a backlog of large
//! records is read in batches rather than one round trip each. Either way the
//! later reads keep the grown bound, and each read's RP0 charges its bound
//! before it is issued ("The charge points", below). The rounds continue until
//! the entry reaches the header's `seq` — chasing it as commits land, within
//! a budget of [`STRUCTURAL_REPAIR_MAX_TOTAL`] (1,024) replayed entries per
//! repair. Each round re-checks only that the header
//! `seq` has not gone down and the object is not superseded; the
//! `commit_chain` folded over the replayed payloads is compared with the
//! header's only at the round whose chunk reached that round's header `seq`.
//! A rollback between rounds still fails that comparison, because the mixed
//! fold cannot match.
//!
//! The entry is advanced **in place**, one record at a time
//! (`CheckedOut::mutate` for a cache checkout): the record decoded
//! ([`decode_stream_record`]), then replayed ([`apply_decoded_record_into`])
//! onto the state at `seq - 1`, the children lists normalized after every
//! record ([`normalize_children`], since the next record was planned on a
//! state without lists under dangling parents), then `seq` and
//! `commit_chain` advanced. After each round's chunk the whole state is
//! normalized ([`normalize_state`]), so between rounds the entry is a valid,
//! normalized cut. An entry holding its plan indexes ([`CacheEntry::plan`]:
//! every cached entry, not the sampled audit's copy) replays through them
//! instead ([`apply_decoded_record_indexed`]), normalizes only the lists each
//! record wrote (`domain::plan_index::normalize_touched`), and needs no
//! per-round [`normalize_state`]: a compact state holds no external-value
//! index (`replay_chunk`), so it is a normalized cut after every record.
//! The points a call can stop at without an error are between rounds and,
//! for a refused RP2 (below), between two records of a round, where an entry
//! without plan indexes is normalized before the repair returns.
//! A replay error leaves the state partly applied: the repair answers a reload
//! and the caller discards the entry.
//!
//! # The charge points (SDD §5.4)
//!
//! A structural `UpdateLayer`'s repair runs with the call's commit charge
//! (`storage::commit_charge`) and holds each round's memory in it before
//! allocating it, never waiting, grow-only. A read's reply is priced in two
//! terms, per stream entry and per byte:
//!
//! - **RP0**, before each read: the held entry at its current weight, state
//!   and plan indexes ([`CacheEntry::weights_now`]), as the handler's hint
//!   holds it, plus the read's reply at its entry bound and its byte bound;
//! - **RP1**, once the chunk passed step 2's checks and before any of it
//!   replays: the same entry, plus the chunk's reply at its actual entries
//!   and bytes, the parse of its largest record's delta part (records parse
//!   one at a time) and the entry's growth by all of the chunk's delta parts.
//!   The lengths come from the zero-copy OMPF container parse; an OMPF whose
//!   container does not parse counts whole.
//! - **RP2**, inside the replay, before each record applies and once it is
//!   decoded: the entry as it stands after the records before it, plus what
//!   that record can grow it by that does not scale with its delta bytes —
//!   the tables at their load limit it doubles, the children lists and
//!   name-index vectors it pushes onto (each trimmed or compacted, so the
//!   first push reallocates), and its listings and depth walk as rewrite
//!   counts — plus the scratch its apply allocates and frees: the walks,
//!   sets and purge of every cascade it can make, bounded by the subtrees it
//!   can delete, a reset's registrations, and the list-delete sink's ids and
//!   copies ([`estimate_replay_record_growth`], `growth` and `scratch`) —
//!   and the chunk's reply with the record's own parse and per-byte growth.
//!   RP1 front-loads the round's parse and per-byte growth, so most refusals
//!   still fall at a round boundary; RP2 adds the per-record growth and
//!   scratch no per-byte factor bounds.
//!
//! The entry's weight is read through [`CacheEntry::weights_now`] — in O(1)
//! for an entry holding its plan indexes, each hash table at its held
//! capacity, so never below what a table thinned by the replayed records
//! still holds, and equal to its stored weight until a round replays into
//! it — once per round that follows a replay, and at RP2 before each record.
//! An entry without plan indexes (the sampled audit's copy) is never
//! re-weighed, since that is an O(state) walk: its held term is its stored
//! state weight `W0` plus `G`, the growth RP2 priced for every record
//! replayed so far (`estimate_replay_record_growth_bounded`'s `growth`, each
//! table priced from the copy's bucket bound, `CopyBucketBound`; never its
//! `scratch`, which is freed once its record applies),
//! and each point's per-byte growth term also carries `D`, the delta parts
//! replayed so far, so the copy's growth beyond `G` stays priced
//! ([`ObjectStateNeed::Cached`] at RP0: `W0 + G` and `D`; at RP1: `W0 + G`
//! and `D` plus the chunk's; at RP2: `W0 + G` before the record and `D`
//! plus the record's). A refused RP0 or RP1 answers
//! `RepairEnd::NeedsCharge` between rounds — the round has not replayed — and
//! a refused RP2 between two records of the round, the refused record not
//! applied; either way the entry is a valid, normalized cut at its `seq` and
//! `commit_chain`: the caller puts it back, the call waits holding nothing,
//! and its re-run is a hit whose repair continues from that cut, reading the
//! refused record first. Nothing awaits inside a round's replay, so a
//! cancellation, the deadline or a shutdown lands only at a read or at the
//! call's wait. [`repair_to_target`] charges the same three points when it is
//! given a charge (the sampled audit's), and answers a refusal
//! [`TargetRepair::ChargeRefused`] at the cut it reached.
//!
//! On a hit the first RP0 runs before the repair's first read, which is
//! where the call's commit token is first checked (below). So a call whose
//! own commit already landed — a counted retry, or a recovery re-issue's
//! first attempt — can wait at RP0 before it learns that, as it can at the
//! WATCH path's WA0. The outcome is unchanged: the next attempt's read
//! answers the landed token.
//!
//! # Acceptance and the reload reasons (SR-4.2)
//!
//! A repair is accepted only when the header exists and is not superseded,
//! its `seq` is not below the entry's, the entries `cached+1 ..= seq` are all
//! present with ids `{n}-0`, each an `update` payload of this object at its
//! id's `seq`, every one replays, and the fold of the entry's `commit_chain`
//! over their payloads equals the header's. Anything else answers
//! `RepairEnd::Reload` with the first [`ReloadReason`] its checks find, run
//! in SR-4.2's order:
//!
//! 1. per round, against the header: `regressed` (the header `seq` is below
//!    the entry's, or below an earlier round's), then `identity` (same `seq`,
//!    other `commit_chain`, nothing replayed yet);
//! 2. per round, over the chunk before any replay: `gap` (an id out of
//!    sequence, a range that ends short of its last `seq` — `reached_end` is
//!    not contiguity — or an entry at `seq` 0, whose successor is
//!    `CreateLayer`'s `1-0` seed, which carries its content without an
//!    `ompf`), then `regressed` (the fold mismatches at the round that reached
//!    the header), `tombstone` (a `deleted` payload), `parse` (an id, payload
//!    or object that does not decode), then `too_far` when the header is
//!    already beyond the budget, then `too_large` (the script's `TOO_LARGE`
//!    for one pending entry alone above [`STRUCTURAL_REPAIR_MAX_BYTES`];
//!    within it, the next read grows instead, "The loop" above);
//! 3. per record, during replay: `parse` (its OMPF or OODF bytes) or
//!    `replay_err` (`apply_delta` refused it), and `dangling_adopted` (the
//!    record created a section the entry holds a dangling parent for);
//! 4. after the round: `too_far` once the budget is spent short of the header
//!    — a backstop that cannot be reached, since step 2's `too_far` already
//!    answers any round whose goal lies beyond the remaining budget.
//!
//! Three consequences of running the cheap checks before replay, and of
//! deciding `too_far` from the header alone:
//!
//! - `too_far` decided at step 2 never replays the round, so a replay error
//!   in it goes unclassified;
//! - a round whose header is already beyond the remaining budget reads only
//!   that round's chunk, so any anomaly past it — a restore's missing
//!   entries, a tombstone, an undecodable entry, a fold mismatch — is
//!   reported as `too_far`. An entry 1,100 commits behind with a restore jump
//!   400 entries in reloads `too_far` where SR-4.2's order alone would say
//!   `gap`;
//! - `dangling_adopted` ends the repair at the adopting record, so a later
//!   record's error is not classified either.
//!
//! Each is a reload all the same; only the label differs.
//!
//! The other two reasons, `diverged` and `uncacheable`, come from the commit,
//! not from repair ([`ReloadReason`]).
//!
//! # The header mapping and the commit token (SR-4.2, SR-4.2a)
//!
//! In the commit paths' order: an absent header ends the repair with
//! `NotFound(EMPTY_HEADER_SENTINEL)`; a header without a decimal `seq` with
//! `FailedPrecondition(PARTIAL_HEADER_SENTINEL)` (never `NotFound`, which
//! would route the handler into checkpoint recovery); then the commit token;
//! then `superseded` with `FailedPrecondition(SUPERSEDED_SENTINEL)`; then a
//! chained call's predecessor. The caller drops the entry on each
//! `RepairEnd::Gone`.
//!
//! The commit token is checked on every round's reply, before anything is
//! planned, with the verdicts `update_layer_fast.lua` step 1b and its
//! PREDECESSOR step reach and in their order (own-key match, then the start
//! record, then `superseded`, then the predecessor): a record that begins
//! with this call's match key — its commit token, or `op.<operation_id>` with
//! a client operation — answers `RepairEnd::Answered` with the record's
//! verdict (this call's own commit, whose reply was lost and which the repair
//! may just have replayed; or, with an operation, an earlier call's); with an
//! unchained operation a record that has moved off the call's start record
//! answers it overtaken; and, after the superseded check, with a chained
//! operation a record that does not name the call's predecessor answers it
//! `PredecessorPending` ([`check_predecessor`]). A wrong-typed record key
//! answers the byte-identical `WRONGTYPE_KEY` refusal both commit backends
//! return (`RepairEnd::CommitTokenWrongType`), for the caller to count and
//! log.
//!
//! # Dangling parents (SR-3.2)
//!
//! The entry holds no children list whose parent is neither the root nor a
//! live section, and records those parents in `dangling_parents`. The engine
//! keeps no list under a parent with no row (SDD §7.7), so the stored lists
//! behind the set are ones an earlier build wrote. Replay keeps the set: a
//! list written under a missing parent adds the id (the children
//! normalization drops the list and returns it; no plan of this engine writes
//! one), and a `DeleteChildrenList` of a listed parent, or an empty
//! `SetChildren` (the store's delete), removes it — this engine's plans reach
//! it only through a delta that creates a listed parent and deletes it again,
//! whose cascade emits the `DeleteChildrenList`. Nothing in the replayed
//! state shows such a delete, so while the set is non-empty each record is
//! replayed through the list-delete sink (`MutationSink::ListDeletes`),
//! which keeps the id of each children list the record deletes and builds
//! every other mutation only to drop it, and the set is kept from those ids
//! ([`advance_to_successor`]); with an empty set nothing can be removed and
//! the record is replayed through the discarding sink. A replayed record
//! that makes a listed parent a live section reloads `dangling_adopted`: a
//! commit of this build deletes the stored list as it creates the parent
//! (SDD §7.7), but one by an earlier build leaves it in place as the
//! parent's list, which the entry does not hold;
//! [`plan_adopts_dangling_parent`] is the same test for a call's own plan.
//!
//! A call's own committed plan advances the entry by the same per-record
//! step, [`advance_to_successor`]: `CheckedOut::advance_committed` applies
//! it to the held entry the plan left in place, with the lists the plan's
//! mutations delete (`StorageMutation::deleted_list`, the one definition the
//! list-delete sink also keeps), exactly as a repair replaying that commit
//! would, so the two cannot disagree about the dangling set.
//! (`successor_entry`, compiled into test and test-support builds only, is
//! the same step on a planned copy: the shadow check's oracle.)
//!
//! # Repair to a target
//!
//! [`repair_to_target`] runs the same loop, capped at a given `(seq,
//! commit_chain)` cut instead of the header's, with no commit-token check. It
//! is for the sampled audit (SR-3.6), which repairs a copy of an entry (state
//! only, no plan indexes: `CacheEntry::copy_without_plan`) to the
//! cut of its own read, holding its repair on the upload budget through the
//! charge it is given (the audit's `AuditCharge`, "The charge points"). It
//! ticks no metric and touches no cache entry: the caller owns the entry it
//! passes.
//!
//! Nothing here ticks a metric; the read side (`storage::structural_cache_read`)
//! counts the repairs it runs.

#[cfg(any(test, feature = "test-support"))]
use std::time::Instant;

use crate::domain::delta::MutationSink;
#[cfg(any(test, feature = "test-support"))]
use crate::domain::delta::StorageMutation;
use crate::domain::ompf::OmpfContainer;
use crate::domain::pass_timing::{self, Pass};
use crate::domain::plan_index::{normalize_touched, PlanTouches};
use crate::domain::replay::{
    apply_decoded_record_indexed, apply_decoded_record_into, decode_stream_record,
    normalize_children, normalize_state, DecodedRecord, ReplayErrorClass,
};
use crate::domain::types::CompactState;
use crate::error::ServiceError;
use crate::observability::metrics::{
    STRUCTURAL_CACHE_RELOAD_DANGLING_ADOPTED, STRUCTURAL_CACHE_RELOAD_DIVERGED,
    STRUCTURAL_CACHE_RELOAD_GAP, STRUCTURAL_CACHE_RELOAD_IDENTITY, STRUCTURAL_CACHE_RELOAD_PARSE,
    STRUCTURAL_CACHE_RELOAD_REGRESSED, STRUCTURAL_CACHE_RELOAD_REPLAY_ERR,
    STRUCTURAL_CACHE_RELOAD_TOMBSTONE, STRUCTURAL_CACHE_RELOAD_TOO_FAR,
    STRUCTURAL_CACHE_RELOAD_TOO_LARGE, STRUCTURAL_CACHE_RELOAD_UNCACHEABLE,
};
use crate::storage::client::ValkeyClient;
use crate::storage::commit_chain::commit_chain_fold;
use crate::storage::commit_charge::{ObjectStateNeed, RepairTransient, SharedCommitCharge};
use crate::storage::keys::KeySchema;
use crate::storage::layer::{
    commit_token_record_wrong_type_refusal, empty_header_not_found, parse_commit_token_record,
    partial_header_failed_precondition, superseded_failed_precondition, CommitIdentity, Operation,
    RecordedCommit,
};
use crate::storage::notifications::{LayerNotification, DELETED_STREAM_PAYLOAD};
use crate::storage::structural_cache::{
    estimate_replay_record_growth, estimate_replay_record_growth_bounded, CacheEntry, CheckedOut,
    CopyBucketBound, HeldCapacities, PricedBuckets, ReplayRecordGrowth,
};
use crate::storage::structural_cache_sync::{
    read_structural_cache_sync, StructuralSyncRequest, SyncChunk, SyncCommitToken, SyncHeader,
    SyncMode, SyncStreamEntry, STRUCTURAL_REPAIR_INITIAL_BYTES, STRUCTURAL_REPAIR_INITIAL_ENTRIES,
    STRUCTURAL_REPAIR_MAX_BYTES, STRUCTURAL_REPAIR_MAX_ENTRIES, STRUCTURAL_REPAIR_PAGE_ENTRIES,
};

/// The most stream entries one repair replays (SR-4.1): 1,024. Beyond it the
/// entry reloads `too_far`.
///
/// A reload happens only on a real gap, trim or anomaly, or after more than
/// about 1,000 commits missed between this replica's structural edits on the
/// object. Replay is cheaper than the spec assumed when it chose 1,024. At
/// 5,000 sections a field-only entry replays in about 17 µs, so 1,024 of them
/// cost about 17.5 ms of replica CPU, against a 188–368 ms reload, with its
/// whole-layer shard block, at 5,003 sections. That reload range spans two
/// runs whose 2× gap is unexplained. Structural entries replay in 0.2–0.7 ms
/// each at 5,000 sections
/// (`docs-internal/validation/structural-edits-b2wo-20261002/epic4-measurements.md`).
pub const STRUCTURAL_REPAIR_MAX_TOTAL: u64 = 1_024;

/// The bounds one repair runs under. Production uses
/// [`RepairBounds::PRODUCTION`] (also the [`Default`]); a test narrows them
/// to reach a bound without writing hundreds of entries or megabytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RepairBounds {
    /// At most this many stream entries per repair read
    /// ([`STRUCTURAL_REPAIR_MAX_ENTRIES`]).
    pub max_entries: u64,
    /// The entry bound of a repair's first read, capped at `max_entries`
    /// ([`STRUCTURAL_REPAIR_INITIAL_ENTRIES`]); the later reads ask up to
    /// `max_entries`. Only the first read is capped: its RP0 is what every
    /// cache hit holds before it knows what is pending.
    pub initial_entries: u64,
    /// The byte bound of a repair's first read, capped at `max_bytes`
    /// ([`STRUCTURAL_REPAIR_INITIAL_BYTES`]). A read's bound grows, and the
    /// later reads keep it (module documentation, "The loop"): to a pending
    /// entry's size when that entry alone is larger than the bound and at
    /// most `max_bytes`, and to twice the bound, at most `max_bytes`, when a
    /// chunk stops at the bound with entries still pending. Set equal to
    /// `max_bytes`, no read grows.
    pub initial_bytes: u64,
    /// At most this many stream-entry bytes per repair read: the hard cap on
    /// a read's byte bound ([`STRUCTURAL_REPAIR_MAX_BYTES`]), past which no
    /// read grows. A pending entry above it reloads `too_large`.
    pub max_bytes: u64,
    /// At most this many stream entries replayed per repair
    /// ([`STRUCTURAL_REPAIR_MAX_TOTAL`]).
    pub max_total: u64,
    /// Entries per `XRANGE` page inside the read
    /// ([`STRUCTURAL_REPAIR_PAGE_ENTRIES`]).
    pub page_entries: u64,
}

impl RepairBounds {
    /// The production bounds: 256 entries per read, a first read of
    /// [`STRUCTURAL_REPAIR_INITIAL_ENTRIES`] entries and 256 KiB, the byte bound growing to at most 4 MiB, 1,024
    /// entries per repair, one entry per page. The sampled audit repairs
    /// under them too.
    pub const PRODUCTION: Self = Self {
        max_entries: STRUCTURAL_REPAIR_MAX_ENTRIES,
        initial_entries: STRUCTURAL_REPAIR_INITIAL_ENTRIES,
        initial_bytes: STRUCTURAL_REPAIR_INITIAL_BYTES,
        max_bytes: STRUCTURAL_REPAIR_MAX_BYTES,
        max_total: STRUCTURAL_REPAIR_MAX_TOTAL,
        page_entries: STRUCTURAL_REPAIR_PAGE_ENTRIES,
    };

    /// The entry bound a repair read asks for: the first read's
    /// `initial_entries`, every later read's `max_entries`, either capped at
    /// `max_entries` and at the repair's remaining entry budget, and at least
    /// one.
    fn read_entries(&self, first: bool, budget_left: u64) -> u64 {
        let bound = if first {
            self.initial_entries.min(self.max_entries)
        } else {
            self.max_entries
        };
        bound.min(budget_left).max(1)
    }

    /// What a repair's first read holds at its RP0 beside the entry: its
    /// reply at its entry bound and its byte bound, nothing replayed yet.
    /// The structural call's suspension after a refused plan growth waits for
    /// at least this on the entry it resumes on (`storage::layer`, the
    /// `Overflow` arm), since the resumed hit's first charge point is that
    /// RP0.
    pub fn first_read(&self) -> RepairTransient {
        RepairTransient {
            reply_entry_count: self.read_entries(true, self.max_total),
            reply_bytes: self.initial_bytes.min(self.max_bytes),
            ..RepairTransient::default()
        }
    }
}

impl Default for RepairBounds {
    fn default() -> Self {
        Self::PRODUCTION
    }
}

/// Why a cache entry is dropped and the object read again: the `reason` label
/// of `live_edit_structural_cache_reloads_total` (SR-2.4, SR-4.2).
///
/// The repair answers every reason but [`Self::Diverged`] and
/// [`Self::Uncacheable`]; the module documentation gives the order its checks
/// run in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReloadReason {
    /// The header `seq` equals the entry's but its `commit_chain` differs: a
    /// re-materialization or rollback landing at the same `seq`.
    Identity,
    /// The header `seq` is below the entry's or went down between two repair
    /// reads, or the `commit_chain` folded over the replayed payloads does not
    /// reach the header's (or the target's): a re-materialization or rollback
    /// at another `seq`, or a commit that did not fold.
    Regressed,
    /// Stream entries the repair needs are missing: trimmed by the stream's
    /// `MAXLEN`, or never written (a restore advances `seq` without them).
    Gap,
    /// A `deleted` entry sits in the repaired range.
    Tombstone,
    /// A stream entry's id, payload, OMPF container or OODF delta does not
    /// decode, or its payload names another object or `seq`.
    Parse,
    /// A decoded stream entry did not replay onto the entry.
    ReplayErr,
    /// More than the repair's budget ([`STRUCTURAL_REPAIR_MAX_TOTAL`]) of
    /// entries separate the entry from the header.
    TooFar,
    /// One pending stream entry alone is larger than the repair reads' hard
    /// cap ([`RepairBounds::max_bytes`]: [`STRUCTURAL_REPAIR_MAX_BYTES`],
    /// 4 MiB, in production). An entry larger
    /// only than a read's current bound ([`RepairBounds::initial_bytes`] at
    /// first) grows the next read to its size instead, with no reload.
    TooLarge,
    /// A replayed commit created a section in the entry's `dangling_parents`
    /// — or this replica's own structural commit did, whose caller then drops
    /// the entry instead of swapping its successor in (SR-3.2) and counts it.
    DanglingAdopted,
    /// A structural commit answered `CACHE_DIVERGED` (SR-6.4). Counted by the
    /// commit's caller; never a repair outcome.
    Diverged,
    /// The entry was dropped because the object became uncacheable: an
    /// explicit put-back refused by the per-entry cap
    /// (`structural_cache::PutBack::OverCap`). Counted by the put-back's
    /// caller (`structural_cache_read::count_put_back`); never a repair
    /// outcome.
    Uncacheable,
}

impl ReloadReason {
    /// The metric label (a
    /// [`crate::observability::metrics::STRUCTURAL_CACHE_RELOAD_REASONS`]
    /// value).
    pub fn label(self) -> &'static str {
        match self {
            Self::Identity => STRUCTURAL_CACHE_RELOAD_IDENTITY,
            Self::Regressed => STRUCTURAL_CACHE_RELOAD_REGRESSED,
            Self::Gap => STRUCTURAL_CACHE_RELOAD_GAP,
            Self::Tombstone => STRUCTURAL_CACHE_RELOAD_TOMBSTONE,
            Self::Parse => STRUCTURAL_CACHE_RELOAD_PARSE,
            Self::ReplayErr => STRUCTURAL_CACHE_RELOAD_REPLAY_ERR,
            Self::TooFar => STRUCTURAL_CACHE_RELOAD_TOO_FAR,
            Self::TooLarge => STRUCTURAL_CACHE_RELOAD_TOO_LARGE,
            Self::DanglingAdopted => STRUCTURAL_CACHE_RELOAD_DANGLING_ADOPTED,
            Self::Diverged => STRUCTURAL_CACHE_RELOAD_DIVERGED,
            Self::Uncacheable => STRUCTURAL_CACHE_RELOAD_UNCACHEABLE,
        }
    }
}

/// **Test-only.** An async hook a test hands the read side to run at a fixed
/// point of a structural call, so it can change the store there.
#[cfg(any(test, feature = "test-support"))]
pub type TestHook =
    std::sync::Arc<dyn Fn() -> futures::future::BoxFuture<'static, ()> + Send + Sync>;

/// The test seams of one repair. Empty in a build without `test-support`,
/// and empty by default in every build.
#[derive(Clone, Default)]
pub(crate) struct RepairHooks {
    /// Run before every repair read after the first: between two reads of
    /// one repair, after the earlier one's chunk was replayed. Lets a test
    /// land a commit (this call's own included) that only a later read sees.
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) between_reads: Option<TestHook>,
    /// Check, after every charged record replayed onto an entry without
    /// plan indexes (the sampled audit's copy), that its bucket bound is at
    /// or above each table's true bucket count, its tombstone bound at or
    /// above the true tombstones at that count (`CopyBucketBound`), and the
    /// same of the bound it starts from: the copy's twin of the shadow
    /// check's weight floor on an indexed entry. The bound's soundness rests
    /// on the hand-kept list of the engine's removal paths the record's
    /// estimate reads (`structural_cache::PricedBuckets`), so a removal path
    /// missing from it fails here. On for [`repair_to_target`]'s callers;
    /// the sampled audit follows its cache's shadow check, which a
    /// memory-measuring harness turns off, since the check clones every
    /// table.
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) copy_bound_check: bool,
}

/// What one repair read and replayed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RepairStats {
    /// Repair reads (`structural_cache_sync.lua` calls) issued.
    pub reads: u32,
    /// Stream entries replayed onto the entry.
    pub replayed_entries: u64,
    /// The reads' stream-byte tallies, summed (entry ids plus every field name
    /// and value; 0 for a read that returned no entry).
    pub bytes: u64,
}

/// How a repair toward the header ended.
#[derive(Debug)]
pub(crate) enum RepairEnd {
    /// The entry is at the header's `(seq, commit_chain)` and normalized.
    Current,
    /// The call's commit-token check answered before planning (see
    /// [`TokenAnswer`]). The entry is a valid cut (possibly short of the
    /// header): the caller puts it back.
    Answered(TokenAnswer),
    /// The commit-token record key holds a foreign type: `refusal` is the
    /// commit backends' byte-identical `WRONGTYPE_KEY` refusal. The entry is a
    /// valid cut: the caller puts it back.
    CommitTokenWrongType { refusal: ServiceError },
    /// The header shows the object gone or replaced: `NotFound` or
    /// `FailedPrecondition` with the commit paths' sentinel. The caller drops
    /// the entry.
    Gone(ServiceError),
    /// An anomaly: the caller drops the entry and reads the object again.
    Reload(ReloadReason),
    /// With a commit charge only: a repair charge point (RP0 before a read,
    /// RP1 before replaying its chunk, RP2 before applying one of its
    /// records) could not hold its need without waiting. `target` is that
    /// need's whole object-state target. The round did not replay (RP0,
    /// RP1), or replayed only the records before the refused one (RP2), so
    /// the entry is a valid cut, normalized: the caller puts it back, and the
    /// call waits for `target` holding nothing; its re-run is a hit whose
    /// repair continues from that cut.
    NeedsCharge { target: u64 },
}

/// How a [`repair_to_target`] ended.
#[derive(Debug)]
pub enum TargetRepair {
    /// The entry is at the target's `(seq, commit_chain)` and normalized.
    Reached,
    /// The repair could not reach the target, for the reason given (the
    /// module documentation's checks, with the target in place of the
    /// header). A target below the entry's `seq`, or above the header's, is
    /// [`ReloadReason::Regressed`]. The entry may be partly replayed.
    Unreachable(ReloadReason),
    /// The header shows the object gone or replaced (`NotFound` or
    /// `FailedPrecondition`, as the header mapping answers).
    Gone(ServiceError),
    /// With a charge only: a charge point (RP0, RP1 or RP2) could not hold
    /// its need, whose whole target is `target`, without waiting. The entry
    /// is a valid, normalized cut short of the target; the caller drops it or
    /// keeps it, as it likes.
    ChargeRefused { target: u64 },
}

/// Whether a planned successor state adopts one of the entry's dangling
/// parents: a section in `entry.dangling_parents` is live in `planned`
/// (SR-3.2).
///
/// For the structural call's own plan: such a creation meets a stored
/// children list the planned state does not hold (the plan was made, like
/// today's full read, without it). The plan itself is right — the commit
/// lands, deleting that list (SDD §7.7) — but on `Committed` the caller drops
/// the entry ([`CheckedOut::discard`]) instead of keeping the planned one;
/// the next call reloads. The repair applies the same test to every replayed
/// record, and a committed plan's step (`CheckedOut::advance_committed`)
/// answers [`ReloadReason::DanglingAdopted`] by it.
pub fn plan_adopts_dangling_parent(entry: &CacheEntry, planned: &CompactState) -> bool {
    adopts_dangling_parent(&entry.dangling_parents, planned)
}

fn adopts_dangling_parent(
    dangling_parents: &std::collections::HashSet<u64>,
    state: &CompactState,
) -> bool {
    dangling_parents
        .iter()
        .any(|parent_id| state.sections.contains_key(parent_id))
}

/// Repair `entry`, a caller-owned copy of a cache entry, to the `(target_seq,
/// target_commit_chain)` cut instead of the header's (module documentation,
/// "Repair to a target"). Reads with `bounds`; checks no commit token; ticks
/// no metric. With `charge` (the sampled audit's), holds each read, round and
/// record in it first, never waiting, as a structural call's repair does
/// (module documentation, "The charge points"); a refusal ends the repair
/// [`TargetRepair::ChargeRefused`]. `None` charges nothing.
///
/// `copy_start`, for an `entry` without plan indexes (the sampled audit's
/// copy), is the capacities the entry it was copied from weighed its tables
/// at (`CacheEntry::held_state_capacities`, read before the copy): the
/// copy's bucket bound starts there (`CopyBucketBound`), since its tables
/// can hide buckets behind the tombstones the clone kept. `None` starts it
/// at the copy's own `capacity()` readings.
///
/// On [`TargetRepair::Reached`] the entry is at the target and normalized.
/// On any other answer, or an error, it may be partly replayed and is the
/// caller's to drop.
///
/// A test-support build checks the copy's bucket bound against its tables'
/// true buckets after every charged record (`RepairHooks::copy_bound_check`).
///
/// # Errors
///
/// The repair read's own classified error (transient → `Aborted`, permanent
/// → `Internal`), or `Internal` for a reply that contradicts itself.
#[allow(clippy::too_many_arguments)]
pub async fn repair_to_target(
    client: &ValkeyClient,
    keys: &KeySchema,
    entry: &mut CacheEntry,
    target_seq: u64,
    target_commit_chain: &str,
    bounds: RepairBounds,
    charge: Option<&dyn SharedCommitCharge>,
    copy_start: Option<HeldCapacities>,
) -> Result<(TargetRepair, RepairStats), ServiceError> {
    #[cfg(any(test, feature = "test-support"))]
    let hooks = RepairHooks {
        between_reads: None,
        copy_bound_check: true,
    };
    #[cfg(not(any(test, feature = "test-support")))]
    let hooks = RepairHooks::default();
    repair_to_target_with(
        client,
        keys,
        entry,
        target_seq,
        target_commit_chain,
        bounds,
        charge,
        copy_start,
        &hooks,
    )
    .await
}

/// [`repair_to_target`] with the test seams `hooks`: the sampled audit's,
/// which checks its copy's bucket bound only while its cache's shadow check
/// is on (`storage::structural_cache_audit`).
#[allow(clippy::too_many_arguments)]
pub(crate) async fn repair_to_target_with(
    client: &ValkeyClient,
    keys: &KeySchema,
    entry: &mut CacheEntry,
    target_seq: u64,
    target_commit_chain: &str,
    bounds: RepairBounds,
    charge: Option<&dyn SharedCommitCharge>,
    copy_start: Option<HeldCapacities>,
    hooks: &RepairHooks,
) -> Result<(TargetRepair, RepairStats), ServiceError> {
    let goal = Goal::Target {
        seq: target_seq,
        commit_chain: target_commit_chain,
    };
    let (end, stats) = repair(client, keys, entry, goal, bounds, charge, copy_start, hooks).await?;
    let answer = match end {
        RepairEnd::Current => TargetRepair::Reached,
        RepairEnd::Reload(reason) => TargetRepair::Unreachable(reason),
        RepairEnd::Gone(error) => TargetRepair::Gone(error),
        RepairEnd::Answered(_) | RepairEnd::CommitTokenWrongType { .. } => {
            return Err(ServiceError::Internal(
                "structural cache repair to a target answered a commit-token verdict it never \
                 checks"
                    .to_owned(),
            ))
        }
        RepairEnd::NeedsCharge { target } => TargetRepair::ChargeRefused { target },
    };
    Ok((answer, stats))
}

/// Where a repair stops.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Goal<'a> {
    /// At the header's current `(seq, commit_chain)`, checking the call's
    /// commit token on every read.
    Header {
        /// The call's `client_id`: the commit-token record's field.
        client_id: &'a str,
        /// The call's commit token.
        commit_token: &'a str,
        /// The call's client operation, with its writer guard, when it
        /// carried one.
        operation: Option<&'a Operation>,
    },
    /// At a given cut, with no commit-token check.
    Target { seq: u64, commit_chain: &'a str },
}

/// Something holding the entry a repair advances: a cache checkout, or a
/// caller-owned entry.
pub(crate) trait RepairHolder {
    /// The entry as it stands.
    fn held(&self) -> &CacheEntry;
    /// Run `f` on the entry in place.
    fn mutate_held<R>(&mut self, f: impl FnOnce(&mut CacheEntry) -> R) -> R;
}

impl RepairHolder for CheckedOut {
    fn held(&self) -> &CacheEntry {
        self.entry()
    }

    fn mutate_held<R>(&mut self, f: impl FnOnce(&mut CacheEntry) -> R) -> R {
        self.mutate(f)
    }
}

impl RepairHolder for CacheEntry {
    fn held(&self) -> &CacheEntry {
        self
    }

    fn mutate_held<R>(&mut self, f: impl FnOnce(&mut CacheEntry) -> R) -> R {
        f(self)
    }
}

/// What a call's commit-token record answers before the call plans
/// (SR-4.2a): the verdict `update_layer_fast.lua` step 1b and its
/// PREDECESSOR step — and the txn path's step 2a and its predecessor check —
/// reach on the same record, from the same parser.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenAnswer {
    /// A commit under the call's match key landed; `recorded.by` says whose:
    /// this call's own (report it), or with a client operation an earlier
    /// call's (`ALREADY_COMMITTED` / `ALREADY_EXISTS`, and the call's
    /// pre-flight increments are owed back).
    Recorded(RecordedCommit),
    /// Unchained client operations only: the writer's record has moved off
    /// the call's start record, so the call is overtaken (`UNAVAILABLE`).
    /// Plan nothing; the caller's shared verdict handling settles the call's
    /// releases.
    Overtaken,
    /// Chained client operations only, checked after the superseded check
    /// ([`check_predecessor`]): the writer's record does not name the call's
    /// predecessor (for a chain's first write: a record exists), so the call
    /// is refused `PREDECESSOR_NOT_COMMITTED` (a plain `UNAVAILABLE` after a
    /// counted retry). Plan nothing; the caller's shared verdict handling
    /// settles the call's releases. `record` is the record as read, `None`
    /// when absent; it carries a commit token.
    PredecessorPending { record: Option<String> },
}

/// Whether a call's commit token is recorded (SR-4.2a).
#[derive(Debug)]
pub(crate) enum TokenVerdict {
    /// No record under the call's match key, and (with an unchained
    /// operation) the record still holds the call's start record: plan, once
    /// the superseded check and — for a chained operation —
    /// [`check_predecessor`] have passed.
    NotLanded,
    /// The record answers the call; see [`TokenAnswer`].
    Answered(TokenAnswer),
    /// The record key holds a foreign type; the commit backends' refusal.
    WrongType(ServiceError),
}

/// Check a repair read's commit-token facts for the call holding
/// `commit_token` and, when it carried one, `operation` (SR-4.2a): a
/// foreign-typed key is the `WRONGTYPE_KEY` refusal, else the record decides
/// ([`check_commit_token_record`]).
///
/// # Errors
///
/// `Internal` for a record under this call's match key with an unparseable
/// tail ([`parse_commit_token_record`]).
pub(crate) fn check_commit_token(
    keys: &KeySchema,
    object_id: u64,
    token: &SyncCommitToken,
    commit_token: &str,
    operation: Option<&Operation>,
) -> Result<TokenVerdict, ServiceError> {
    if token.is_wrong_type() {
        let wtok_key = keys.obj_write_token(&object_id.to_string());
        return Ok(TokenVerdict::WrongType(
            commit_token_record_wrong_type_refusal(&wtok_key, &token.key_type),
        ));
    }
    Ok(
        match check_commit_token_record(token.record.as_deref(), commit_token, operation)? {
            Some(answer) => TokenVerdict::Answered(answer),
            None => TokenVerdict::NotLanded,
        },
    )
}

/// What the commit-token `record` (absent: `None`) answers the call holding
/// `commit_token` and, when it carried one, `operation`, in step 1b's order:
/// a record under the call's match key (the token, or `op.<operation_id>`)
/// is [`TokenAnswer::Recorded`] with [`parse_commit_token_record`]'s verdict;
/// otherwise, with an unchained operation, a record that no longer holds the
/// call's start record is [`TokenAnswer::Overtaken`]; otherwise `None` (a
/// chained call's predecessor is [`check_predecessor`]'s). Shape-blind
/// on the read side: the repair script returns the record verbatim, the
/// `<token>:<seq>:<server_ts>` and the
/// `op.<id>:<seq>:<server_ts>:<token>:<request_digest>` shapes alike.
///
/// # Errors
///
/// As [`check_commit_token`].
pub(crate) fn check_commit_token_record(
    record: Option<&str>,
    commit_token: &str,
    operation: Option<&Operation>,
) -> Result<Option<TokenAnswer>, ServiceError> {
    let identity = CommitIdentity::from_recorded_token(commit_token, operation);
    if let Some(recorded) = record
        .and_then(|record| parse_commit_token_record(record, &identity))
        .transpose()?
    {
        return Ok(Some(TokenAnswer::Recorded(recorded)));
    }
    Ok(identity
        .overtaken_by(record)
        .then_some(TokenAnswer::Overtaken))
}

/// The predecessor guard on the read side: for a chained call (holding
/// `commit_token` and `operation`) whose commit-token `record` (absent:
/// `None`) did not answer it ([`check_commit_token_record`] gave `None`) and
/// whose object is not superseded, [`TokenAnswer::PredecessorPending`] when
/// the record does not name its predecessor
/// (`CommitIdentity::predecessor_pending`); otherwise `None` — always `None`
/// without a chained operation.
///
/// Deliberately separate from [`check_commit_token_record`], which every
/// caller runs before its superseded check: the canonical order is own-key
/// match → start record → SUPERSEDED → PREDECESSOR → FENCED, so each caller
/// runs this after that check and before planning.
pub(crate) fn check_predecessor(
    record: Option<&str>,
    commit_token: &str,
    operation: Option<&Operation>,
) -> Option<TokenAnswer> {
    CommitIdentity::from_recorded_token(commit_token, operation)
        .predecessor_pending(record)
        .then(|| TokenAnswer::PredecessorPending {
            record: record.map(str::to_owned),
        })
}

/// The repair loop (module documentation): advance `holder`'s entry toward
/// `goal` under `bounds`, and say how it ended, with what it read and
/// replayed. With `charge` (the structural call's commit charge), each round
/// holds its read and its replay in it first, and each record its own
/// growth, never waiting (RP0, RP1, RP2), and a refusal ends the repair with
/// [`RepairEnd::NeedsCharge`]; `None` charges nothing. `copy_start` starts
/// the bucket bound of an entry without plan indexes ([`CopyWeight`],
/// [`repair_to_target`]); the cached route passes `None`, its entries holding
/// their indexes. `hooks` are test seams; production passes the default
/// (none).
///
/// # Errors
///
/// A repair read's classified error, a commit-token record with this call's
/// token and an unparseable tail (`Internal`), or `Internal` for a reply that
/// contradicts itself. The entry is then a valid cut (an error is never raised
/// mid-replay).
#[allow(clippy::too_many_arguments)]
pub(crate) async fn repair<H: RepairHolder + Send>(
    client: &ValkeyClient,
    keys: &KeySchema,
    holder: &mut H,
    goal: Goal<'_>,
    bounds: RepairBounds,
    charge: Option<&dyn SharedCommitCharge>,
    copy_start: Option<HeldCapacities>,
    hooks: &RepairHooks,
) -> Result<(RepairEnd, RepairStats), ServiceError> {
    #[cfg(not(any(test, feature = "test-support")))]
    let _ = hooks;
    let object_id = holder.held().object_id;
    let (client_id, target_seq) = match goal {
        Goal::Header { client_id, .. } => (client_id, None),
        Goal::Target { seq, .. } => ("", Some(seq)),
    };
    let mut stats = RepairStats::default();
    let mut previous_header_seq: Option<u64> = None;
    // The reads' byte bound: the first asks `initial_bytes` (at most
    // `max_bytes`); a larger pending entry within `max_bytes` grows it to that
    // entry's size (step 2's `TOO_LARGE`), and a chunk the bound cut short
    // doubles it (after step 3), each capped at `max_bytes` and kept for the
    // later reads.
    let mut read_bytes = bounds.initial_bytes.min(bounds.max_bytes);
    // The held entry's estimated weight, state and plan indexes, for the
    // charge points: read once per round that follows a replay
    // (`CacheEntry::weights_now`, O(1) for an indexed entry), since a replay
    // leaves the stored weights stale.
    let mut weight: Option<u64> = None;
    // An entry without plan indexes (the sampled audit's copy) is never
    // re-weighed — that is an O(state) walk — so its charge points hold its
    // stored weight plus what RP2 priced and the delta parts replayed
    // (module documentation, "The charge points"), RP2 pricing each record's
    // tables from the copy's bucket bound.
    let mut copy = holder
        .held()
        .plan
        .is_none()
        .then(|| CopyWeight::of(holder.held(), copy_start));
    #[cfg(any(test, feature = "test-support"))]
    if let Some(copy) = copy.as_mut() {
        copy.check_bound = hooks.copy_bound_check;
        if copy.check_bound {
            copy.bound
                .assert_holds_for_test(&holder.held().state, "the copy's starting bound");
        }
    }
    loop {
        #[cfg(any(test, feature = "test-support"))]
        if stats.reads > 0 {
            if let Some(hook) = hooks.between_reads.as_ref() {
                hook().await;
            }
        }
        let entry_seq = holder.held().seq;
        let budget_left = bounds.max_total.saturating_sub(stats.replayed_entries);
        // This read's entry bound, which RP0 prices: the first read's is
        // capped at `initial_entries`, the later reads ask up to
        // `max_entries`.
        let max_entries = bounds.read_entries(stats.reads == 0, budget_left);

        // RP0, before the read: the held entry, state and plan indexes (as
        // the hint holds it), plus this read's reply at its entry bound and
        // its byte bound. Grow-only, never waiting. Refused, the round has
        // not begun: the entry is the last round's cut (or the checked-out
        // one), so the caller puts it back and the call waits.
        //
        // The reply term prices the reply as this call's memory, nothing
        // more. The read buffer the reply grows is the connection's: what a
        // connection keeps once drained is the connection buffers' at-rest
        // bound `C_conn` (`storage::connection_buffers::at_rest_bound_bytes`),
        // outside `B`, and growth above that is the call's until the drain.
        // `C_conn` counts each general client at one warm connection plus
        // (P − 1) idle ones, and it holds on a cluster (P > 1) too, whichever
        // primary's connection a repair read lands on: fred's retention
        // (`vendor/fred-10.1.0/src/live_edit/retention.rs`) elects at most one
        // warm connection per client (`elect_streaming`, :425-450, a single
        // index), and cuts every other drained connection that carried
        // traffic to the idle shape — no in-flight queue allocation, no write
        // buffer, a fresh 8 KiB read buffer (`action`, :453-463; `cut`,
        // :578-588) — at the router's next sweep (`sweep`, :600-644) or, for
        // a reply that drains it, at the reply pop before the caller is
        // answered (`on_reply`, :671-679). Warmth moves only from a drained
        // connection, and the one it leaves is cut. So the read's growth
        // above the class limits is transient in-call memory, which this
        // term covers, and no further term is added here for a cluster
        // client.
        if let Some(charge) = charge {
            let (weight, replayed) = match copy {
                Some(copy) => (copy.held(), copy.replayed),
                None => (*weight.get_or_insert_with(|| held_weight(&*holder)), 0),
            };
            let need = ObjectStateNeed::cached_repair(
                weight,
                RepairTransient {
                    reply_entry_count: max_entries,
                    reply_bytes: read_bytes,
                    replay_delta_bytes: replayed,
                    ..RepairTransient::default()
                },
            );
            if let Err(target) = charge.try_hold_at_least_need(&need) {
                return Ok((RepairEnd::NeedsCharge { target }, stats));
            }
        }

        let request = StructuralSyncRequest {
            object_id,
            client_id,
            mode: SyncMode::Repair {
                from_seq: entry_seq,
                target_seq,
            },
            max_entries,
            max_bytes: read_bytes,
            page_entries: bounds.page_entries,
        };
        let reply = read_structural_cache_sync(client, keys, &request).await?;
        stats.reads = stats.reads.saturating_add(1);

        // The header mapping, in the commit paths' order (SR-4.2).
        let facts = match reply.header {
            SyncHeader::Absent => return Ok((RepairEnd::Gone(empty_header_not_found()), stats)),
            SyncHeader::Present(facts) => facts,
        };
        let Some(header_seq) = facts.seq else {
            return Ok((RepairEnd::Gone(partial_header_failed_precondition()), stats));
        };
        if let Goal::Header {
            commit_token,
            operation,
            ..
        } = goal
        {
            match check_commit_token(
                keys,
                object_id,
                &reply.commit_token,
                commit_token,
                operation,
            )? {
                TokenVerdict::NotLanded => {}
                TokenVerdict::Answered(answer) => return Ok((RepairEnd::Answered(answer), stats)),
                TokenVerdict::WrongType(refusal) => {
                    return Ok((RepairEnd::CommitTokenWrongType { refusal }, stats))
                }
            }
        }
        if facts.superseded.is_some() {
            return Ok((RepairEnd::Gone(superseded_failed_precondition()), stats));
        }
        // A chained call's predecessor, after the superseded check (the
        // canonical order), before anything is planned.
        if let Goal::Header {
            commit_token,
            operation,
            ..
        } = goal
        {
            if let Some(answer) = check_predecessor(
                reply.commit_token.record.as_deref(),
                commit_token,
                operation,
            ) {
                return Ok((RepairEnd::Answered(answer), stats));
            }
        }

        // Step 1: the header against the entry.
        if header_seq < entry_seq || previous_header_seq.is_some_and(|seen| header_seq < seen) {
            return Ok((RepairEnd::Reload(ReloadReason::Regressed), stats));
        }
        previous_header_seq = Some(header_seq);
        let (goal_seq, goal_chain) = match goal {
            Goal::Header { .. } => (header_seq, facts.commit_chain.as_str()),
            Goal::Target { seq, commit_chain } => {
                if header_seq < seq || entry_seq > seq {
                    return Ok((RepairEnd::Reload(ReloadReason::Regressed), stats));
                }
                (seq, commit_chain)
            }
        };
        if entry_seq == goal_seq {
            if holder.held().commit_chain == goal_chain {
                return Ok((RepairEnd::Current, stats));
            }
            let reason = if stats.replayed_entries == 0 {
                ReloadReason::Identity
            } else {
                ReloadReason::Regressed
            };
            return Ok((RepairEnd::Reload(reason), stats));
        }

        // Step 2: the chunk, before any replay. From here `entry_seq <
        // goal_seq`: entries are needed.
        let beyond_budget = goal_seq - entry_seq > budget_left;
        let (entries, chains, chunk_bytes, stopped_at_byte_bound) = match reply.chunk {
            SyncChunk::Probe(_) | SyncChunk::NoRange => {
                return Err(contradiction(
                    object_id,
                    "a repair read with a decimal header seq answered no range",
                ))
            }
            SyncChunk::TooLarge {
                end_seq,
                entry_id,
                entry_bytes,
            } => {
                check_end_seq(object_id, end_seq, goal_seq)?;
                let reason = match parse_stream_id(&entry_id) {
                    None => ReloadReason::Parse,
                    Some(id) if entry_seq == 0 || id != (entry_seq + 1, 0) => ReloadReason::Gap,
                    Some(_) if beyond_budget => ReloadReason::TooFar,
                    // The script answers `TOO_LARGE` only for an entry above
                    // the read's bound; anything else would grow the read
                    // forever.
                    Some(_) if entry_bytes <= read_bytes => {
                        return Err(contradiction(
                            object_id,
                            &format!(
                                "a repair read bounded at {read_bytes} bytes answered TOO_LARGE \
                                 for an entry of {entry_bytes} bytes"
                            ),
                        ))
                    }
                    // Within the hard cap: the next read grows to the entry's
                    // size (after its RP0) instead of reloading, and keeps it.
                    Some(_) if entry_bytes <= bounds.max_bytes => {
                        read_bytes = entry_bytes;
                        continue;
                    }
                    Some(_) => ReloadReason::TooLarge,
                };
                return Ok((RepairEnd::Reload(reason), stats));
            }
            SyncChunk::Entries {
                end_seq,
                entries,
                reached_end,
                bytes,
            } => {
                stats.bytes = stats.bytes.saturating_add(bytes);
                check_end_seq(object_id, end_seq, goal_seq)?;
                if entries.is_empty() && !reached_end {
                    return Err(contradiction(
                        object_id,
                        "a repair read stopped at its bounds before its first entry",
                    ));
                }
                let scan = ChunkScan {
                    object_id,
                    entry_seq,
                    entry_chain: &holder.held().commit_chain,
                    goal_seq,
                    goal_chain,
                    reached_end,
                    beyond_budget,
                };
                // The script stops a chunk short of the range's end at one of
                // two bounds: the entry count (the chunk then holds exactly
                // the request's `max_entries`) or the byte bound (fewer).
                let stopped_at_byte_bound =
                    !reached_end && (entries.len() as u64) < request.max_entries;
                match scan.check(&entries) {
                    Ok(chains) => (entries, chains, bytes, stopped_at_byte_bound),
                    Err(reason) => return Ok((RepairEnd::Reload(reason), stats)),
                }
            }
        };

        // The round's transient: its reply, its largest record's delta part
        // and its delta parts summed.
        let transient = replay_transient(chunk_bytes, &entries);

        // RP1, before the replay: the held entry, state and plan indexes,
        // plus the chunk's reply at its actual entries and bytes, the parse
        // of its largest record's delta part and the entry's growth by every
        // delta part in it (on the audit's copy, also by those replayed
        // before). Grow-only, never waiting. Refused, nothing of the round is
        // replayed: the entry is still the last round's cut, so the caller
        // puts it back, and the re-run reads this chunk again. RP1
        // front-loads the round's parse and per-byte growth, so most refusals
        // fall here, at a round boundary; RP2, per record inside the replay,
        // adds what a record grows beyond its delta bytes.
        if let Some(charge) = charge {
            let need = match copy {
                Some(copy) => ObjectStateNeed::cached_repair(
                    copy.held(),
                    RepairTransient {
                        replay_delta_bytes: transient
                            .replay_delta_bytes
                            .saturating_add(copy.replayed),
                        ..transient
                    },
                ),
                None => ObjectStateNeed::cached_repair(
                    *weight.get_or_insert_with(|| held_weight(&*holder)),
                    transient,
                ),
            };
            if let Err(target) = charge.try_hold_at_least_need(&need) {
                return Ok((RepairEnd::NeedsCharge { target }, stats));
            }
        }

        // Step 3: replay the chunk in place, RP2 before each record.
        let record_charge = charge.map(|charge| RecordCharge {
            charge,
            reply_entry_count: transient.reply_entry_count,
            reply_bytes: transient.reply_bytes,
        });
        let replayed = holder
            .mutate_held(|held| replay_chunk(held, &entries, chains, record_charge, copy.as_mut()));
        match replayed {
            Err(reason) => return Ok((RepairEnd::Reload(reason), stats)),
            Ok(ChunkEnd::Replayed) => {}
            // RP2 refused record `replayed`: the records before it are
            // replayed and the entry is the normalized cut after the last
            // of them, at its `seq` and `commit_chain`. The caller puts it
            // back and the call waits; its re-run reads from that cut, the
            // refused record first.
            Ok(ChunkEnd::Refused { target, replayed }) => {
                stats.replayed_entries = stats.replayed_entries.saturating_add(replayed as u64);
                return Ok((RepairEnd::NeedsCharge { target }, stats));
            }
        }
        stats.replayed_entries = stats.replayed_entries.saturating_add(entries.len() as u64);
        // The replay grew the entry: the next round's charge points weigh it
        // again.
        weight = None;
        // A chunk the byte bound cut short with more pending: the next read
        // asks twice the bound, up to `max_bytes`, and the later reads keep
        // it, so a backlog of large records is read in batches rather than
        // one round trip each. The next round's RP0 charges the grown bound
        // before it reads.
        if stopped_at_byte_bound {
            read_bytes = read_bytes.saturating_mul(2).min(bounds.max_bytes);
        }

        // Step 4: done, or another round within the budget.
        if holder.held().seq == goal_seq {
            // The scan compared the fold with the goal's `commit_chain` at
            // this round, which reached the goal.
            return Ok((RepairEnd::Current, stats));
        }
        // A deliberate backstop that cannot be reached: a round replays only
        // when its goal is within the remaining budget (`beyond_budget` in
        // step 2 answers `too_far` before any replay, re-checked each round
        // against that round's header, so commits landing between reads are
        // covered), and its read asks for at most that budget. So a round
        // either reaches its goal or leaves budget over. Kept, rather than
        // turned into an assertion, so a change to either rule still ends
        // the loop with a reload instead of a read past the budget.
        if stats.replayed_entries >= bounds.max_total {
            return Ok((RepairEnd::Reload(ReloadReason::TooFar), stats));
        }
    }
}

/// The held entry's estimated weight for a repair charge point: its state
/// and plan indexes as it stands ([`CacheEntry::weights_now`]), in O(1) for
/// an entry holding its plan indexes. Before any replay it equals the stored
/// weight ([`CacheEntry::weight_bytes`], set when the entry was last admitted
/// or put back); after one, the stored weights are stale and this is current.
fn held_weight<H: RepairHolder>(holder: &H) -> u64 {
    let (state, indexes) = holder.held().weights_now();
    state.saturating_add(indexes)
}

/// The transient of a checked chunk of `chunk_bytes` (the read's byte
/// tally), which RP1 holds and whose summed delta parts count against the
/// repair's replayed-bytes budget: its reply at its entries and bytes, its
/// largest record's delta part (one record is parsed at a time) and its
/// delta parts summed (what the replay adds to the entry).
fn replay_transient(chunk_bytes: u64, entries: &[SyncStreamEntry]) -> RepairTransient {
    let (largest, summed) = entries
        .iter()
        .map(|entry| delta_part_bytes(entry.ompf.as_deref()))
        .fold((0u64, 0u64), |(largest, summed), part| {
            (largest.max(part), summed.saturating_add(part))
        });
    RepairTransient {
        reply_entry_count: u64::try_from(entries.len()).unwrap_or(u64::MAX),
        reply_bytes: chunk_bytes,
        parse_delta_bytes: largest,
        replay_delta_bytes: summed,
    }
}

/// RP2's inputs for one round (module documentation, "The charge points"):
/// the call's commit charge and the round's reply, which stays held while
/// its records replay.
#[derive(Clone, Copy)]
struct RecordCharge<'c> {
    charge: &'c dyn SharedCommitCharge,
    /// The chunk's entries.
    reply_entry_count: u64,
    /// The chunk's byte tally.
    reply_bytes: u64,
}

impl RecordCharge<'_> {
    /// RP2 for `record`, decoded and about to replay onto `held` through the
    /// list-delete sink when `list_deletes`: hold at least the entry as it
    /// stands plus what the record can grow it by beyond its delta bytes and
    /// the scratch its apply allocates and frees
    /// ([`estimate_replay_record_growth`]: `growth` + `scratch`) and the
    /// round's reply with the record's own parse and per-byte growth at
    /// `delta_bytes`, its delta part. Grow-only, never waiting. Answers the
    /// record's priced growth alone — the scratch is gone once the record is
    /// applied — and, on an entry without plan indexes, what its bucket
    /// bound's raise reads once the record applies ([`PricedBuckets`]); or
    /// `Err` with the need's target when refused.
    ///
    /// The entry as it stands is, with plan indexes, its weight now
    /// ([`held_weight`], O(1)); without them (the sampled audit's copy,
    /// `copy`), its stored weight plus the growth priced for the records
    /// before ([`CopyWeight`]), the per-byte term then carrying the delta
    /// parts replayed before as well, and the record's tables priced from
    /// the copy's bucket bound.
    fn try_hold(
        &self,
        held: &CacheEntry,
        copy: Option<&CopyWeight>,
        record: &DecodedRecord,
        delta_bytes: u64,
        list_deletes: bool,
    ) -> Result<(u64, Option<PricedBuckets>), u64> {
        let (estimate, priced) = match held.plan.as_ref() {
            Some(indexes) => (
                estimate_replay_record_growth(
                    &held.state,
                    Some(indexes),
                    record.delta(),
                    list_deletes,
                ),
                None,
            ),
            None => {
                let (estimate, priced) =
                    unindexed_record_growth(held, copy.map(|c| &c.bound), record, list_deletes);
                (estimate, Some(priced))
            }
        };
        let (weight, replayed_before) = match copy {
            Some(copy) => (copy.held(), copy.replayed),
            None => (held_weight(held), 0),
        };
        let need = ObjectStateNeed::cached_repair_record(
            weight,
            estimate.growth.saturating_add(estimate.scratch),
            estimate.rewrite,
            RepairTransient {
                reply_entry_count: self.reply_entry_count,
                reply_bytes: self.reply_bytes,
                parse_delta_bytes: delta_bytes,
                replay_delta_bytes: replayed_before.saturating_add(delta_bytes),
            },
        );
        self.charge
            .try_hold_at_least_need(&need)
            .map(|_| (estimate.growth, priced))
    }
}

/// The per-record growth estimate of an entry without plan indexes — the
/// sampled audit's copy, and a test's index-less entry — each of its tables
/// priced from `bound`, the copy's bucket bound
/// ([`estimate_replay_record_growth_bounded`], whose doc carries the
/// soundness argument), or, with none, from a bound at the entry's own
/// `capacity()` readings. Also answers what the bound's raise reads once the
/// record applies ([`CopyWeight::applied`]). The one estimate on an
/// index-less entry, so a bound on that copy's estimate has one place to go.
fn unindexed_record_growth(
    held: &CacheEntry,
    bound: Option<&CopyBucketBound>,
    record: &DecodedRecord,
    list_deletes: bool,
) -> (ReplayRecordGrowth, PricedBuckets) {
    let live;
    let bound = match bound {
        Some(bound) => bound,
        None => {
            live = CopyBucketBound::new(&held.state, HeldCapacities::live(&held.state));
            &live
        }
    };
    estimate_replay_record_growth_bounded(&held.state, bound, record.delta(), list_deletes)
}

/// What an entry without plan indexes is charged as holding, kept without
/// re-weighing it (an O(state) walk): its stored state weight when the
/// repair began, `base` (`W0`); the growth RP2 priced for every record it
/// replayed since, `grown` (`G`); and those records' delta parts,
/// `replayed` (`D`), whose growth beyond `G` the per-byte term prices.
/// `G` sums each record's `growth` only, what the entry can keep: a record's
/// `scratch` (`ReplayRecordGrowth::scratch`) is held at its own RP2 and is
/// freed by the time the record is applied. It also carries the entry's
/// bucket bound, `bound` ([`CopyBucketBound`]), which RP2 prices each
/// record's tables from and each applied record raises; a stack value,
/// dropped with the repair.
#[derive(Debug, Clone, Copy)]
struct CopyWeight {
    base: u64,
    grown: u64,
    replayed: u64,
    bound: CopyBucketBound,
    /// Whether the bound is checked against the true buckets after every
    /// charged record (`RepairHooks::copy_bound_check`).
    #[cfg(any(test, feature = "test-support"))]
    check_bound: bool,
}

impl CopyWeight {
    /// The weight of `entry` as the repair finds it: its stored state weight;
    /// and its bucket bound from `start`, the capacities the entry it was
    /// copied from weighed its tables at, or its own `capacity()` readings
    /// without them.
    fn of(entry: &CacheEntry, start: Option<HeldCapacities>) -> Self {
        let start = start.unwrap_or_else(|| HeldCapacities::live(&entry.state));
        Self {
            base: entry.state_weight_bytes,
            grown: 0,
            replayed: 0,
            bound: CopyBucketBound::new(&entry.state, start),
            #[cfg(any(test, feature = "test-support"))]
            check_bound: false,
        }
    }

    /// `W0 + G`: the held term of its charge points.
    fn held(&self) -> u64 {
        self.base.saturating_add(self.grown)
    }

    /// A record of `delta_bytes`, priced at `growth` by RP2 (0 uncharged),
    /// was applied, leaving `state`: `G` gains its growth — never its
    /// scratch — and `D` its delta part, and the bucket bound moves by its
    /// raise rule from `priced`, the buckets the record's estimate priced
    /// ([`CopyBucketBound::applied`]). `priced` is `None` only on a repair
    /// with no charge, which prices no record and never reads `W0 + G` or the
    /// bound, so the bound is left as it is. A test-support build with the
    /// check on then asserts the moved bound against the tables' true
    /// buckets after the record (`RepairHooks::copy_bound_check`).
    fn applied(
        &mut self,
        growth: u64,
        delta_bytes: u64,
        priced: Option<&PricedBuckets>,
        state: &CompactState,
    ) {
        self.grown = self.grown.saturating_add(growth);
        self.replayed = self.replayed.saturating_add(delta_bytes);
        if let Some(priced) = priced {
            self.bound.applied(priced, state);
            #[cfg(any(test, feature = "test-support"))]
            if self.check_bound {
                self.bound.assert_holds_for_test(
                    state,
                    &format!("after the replayed record at seq {}", state.sequence_num),
                );
            }
        }
    }
}

/// How [`replay_chunk`] ended, short of a reload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChunkEnd {
    /// Every record replayed.
    Replayed,
    /// RP2 refused the record at index `replayed` before it applied: the
    /// records before it replayed and the entry is the normalized cut after
    /// them. `target` is the refused need's whole object-state target.
    Refused { target: u64, replayed: usize },
}

/// The delta part's length in one stream entry's OMPF, from the zero-copy
/// container parse (no part is copied or decoded); an absent OMPF has none,
/// and one whose container does not parse counts whole.
fn delta_part_bytes(ompf: Option<&[u8]>) -> u64 {
    let Some(ompf) = ompf else {
        return 0;
    };
    let len = match OmpfContainer::parse(ompf) {
        Ok(container) => container.delta_bytes.len(),
        Err(_) => ompf.len(),
    };
    u64::try_from(len).unwrap_or(u64::MAX)
}

/// The `Internal` error for a reply that contradicts itself or the request.
fn contradiction(object_id: u64, what: &str) -> ServiceError {
    ServiceError::Internal(format!(
        "structural cache repair of object {object_id}: {what}"
    ))
}

/// The script computes a chunk's last `seq` from the header it read in the
/// same call and the request's target, exactly as [`repair`] computes the
/// goal; anything else is a script this binary does not carry.
fn check_end_seq(object_id: u64, end_seq: u64, goal_seq: u64) -> Result<(), ServiceError> {
    if end_seq == goal_seq {
        return Ok(());
    }
    Err(contradiction(
        object_id,
        &format!("a repair read's range ends at seq {end_seq}, not at its goal {goal_seq}"),
    ))
}

/// A stream id `{ms}-{n}` as its two numbers; `None` for anything else.
fn parse_stream_id(id: &str) -> Option<(u64, u64)> {
    let (ms, n) = id.split_once('-')?;
    Some((ms.parse().ok()?, n.parse().ok()?))
}

/// The pre-replay checks of one round's chunk (module documentation, step 2).
struct ChunkScan<'a> {
    object_id: u64,
    /// The entry's `seq` before the round.
    entry_seq: u64,
    /// The entry's `commit_chain` before the round.
    entry_chain: &'a str,
    goal_seq: u64,
    goal_chain: &'a str,
    reached_end: bool,
    /// More entries separate the entry from the goal than the repair's
    /// remaining budget.
    beyond_budget: bool,
}

impl ChunkScan<'_> {
    /// Check `entries` in SR-4.2's order and, when they pass, return the
    /// `commit_chain` after each one.
    fn check(&self, entries: &[SyncStreamEntry]) -> Result<Vec<String>, ReloadReason> {
        // `gap`: CreateLayer's seed is never replayed; ids `{n}-0` in
        // sequence; a range that ran out of entries must have ended at its
        // last `seq` (`reached_end` alone is not contiguity).
        if self.entry_seq == 0 {
            return Err(ReloadReason::Gap);
        }
        let mut expected = self.entry_seq + 1;
        let mut unparseable_id = false;
        for entry in entries {
            match parse_stream_id(&entry.id) {
                Some(id) if id == (expected, 0) => expected += 1,
                Some(_) => return Err(ReloadReason::Gap),
                None => {
                    unparseable_id = true;
                    break;
                }
            }
        }
        let last_seq = expected - 1;
        if !unparseable_id && self.reached_end && last_seq != self.goal_seq {
            return Err(ReloadReason::Gap);
        }

        // `regressed`: the fold, compared only at the round that reached the
        // goal. Without every payload's text there is no fold to compare.
        let payloads: Option<Vec<&str>> = entries.iter().map(|e| e.payload.as_deref()).collect();
        let mut chains = Vec::with_capacity(entries.len());
        if let Some(payloads) = &payloads {
            let mut chain = self.entry_chain.to_owned();
            for payload in payloads {
                chain = commit_chain_fold(&chain, payload);
                chains.push(chain.clone());
            }
            if !unparseable_id && last_seq == self.goal_seq && chain != self.goal_chain {
                return Err(ReloadReason::Regressed);
            }
        }

        // `tombstone`, then `parse`: every entry an `update` of this object
        // at its id's `seq`.
        let Some(payloads) = payloads else {
            if entries
                .iter()
                .any(|e| e.payload.as_deref() == Some(DELETED_STREAM_PAYLOAD))
            {
                return Err(ReloadReason::Tombstone);
            }
            return Err(ReloadReason::Parse);
        };
        if payloads.contains(&DELETED_STREAM_PAYLOAD) {
            return Err(ReloadReason::Tombstone);
        }
        if unparseable_id {
            return Err(ReloadReason::Parse);
        }
        for (seq, payload) in (self.entry_seq + 1..).zip(&payloads) {
            match LayerNotification::parse(payload) {
                Ok(LayerNotification::Update {
                    object_id,
                    sequence_num,
                    ..
                }) if object_id == self.object_id && sequence_num == seq => {}
                _ => return Err(ReloadReason::Parse),
            }
        }

        // `too_far`, decided before replaying a round that cannot reach the
        // goal within the budget.
        if self.beyond_budget {
            return Err(ReloadReason::TooFar);
        }
        Ok(chains)
    }
}

/// Replay `entries` — checked by [`ChunkScan::check`], with `chains` the
/// `commit_chain` after each — onto `held` in place, keeping its dangling
/// parents, and normalize it (module documentation, step 3).
///
/// An entry holding its plan indexes ([`CacheEntry::plan`]) replays each
/// record through them ([`apply_decoded_record_indexed`]): no per-record
/// index build or full depth walk, and only the lists the record wrote are
/// normalized. A compact state holds no external-value index, so that path
/// needs no per-round `normalize_state` either. An entry without them — the
/// sampled audit's copy — replays through the per-delta indexes and
/// normalizes every list, as before.
///
/// Each record is decoded ([`decode_stream_record`]) before it is applied.
/// With `record_charge` (the structural call's or the sampled audit's
/// charge), RP2 runs between the two ([`RecordCharge::try_hold`]); refused,
/// the record is dropped unapplied and the chunk ends [`ChunkEnd::Refused`]
/// at the cut after the records before it — already normalized on an indexed
/// entry, and normalized here (`normalize_state`) on one without indexes.
/// `copy`, an entry without plan indexes' charged weight ([`CopyWeight`]),
/// gains each applied record's priced growth and delta part, and its bucket
/// bound is raised from the buckets the record's RP2 priced, read once the
/// record and its children normalization have run.
///
/// On an error `held`'s state may be partly applied and must be discarded.
fn replay_chunk(
    held: &mut CacheEntry,
    entries: &[SyncStreamEntry],
    chains: Vec<String>,
    record_charge: Option<RecordCharge<'_>>,
    mut copy: Option<&mut CopyWeight>,
) -> Result<ChunkEnd, ReloadReason> {
    for (index, (entry, chain)) in entries.iter().zip(chains).enumerate() {
        let seq = held.seq + 1;
        // Only a listed dangling parent can lose its stored list, and nothing
        // in the state shows it: read the deletes off the list-delete sink
        // then, which keeps the ids of the lists the record deletes and
        // nothing else.
        let track_deletes = !held.dangling_parents.is_empty();
        let mut sink = if track_deletes {
            MutationSink::list_deletes()
        } else {
            MutationSink::Discard
        };
        let mut touches = PlanTouches::default();
        let ompf = entry.ompf.as_deref();
        let delta_bytes = delta_part_bytes(ompf);
        // The record's priced growth once applied (0 uncharged) and, on an
        // entry without plan indexes, the buckets its estimate priced for the
        // copy's bucket bound (none uncharged); or the refused target.
        let step = pass_timing::time(Pass::RepairReplay, || {
            let record = decode_stream_record(held.seq, ompf, seq)?;
            // RP2, between the decode and the apply.
            let mut growth = 0;
            let mut priced = None;
            if let Some(record_charge) = record_charge.as_ref() {
                let held_now: &CacheEntry = held;
                let copy_now = copy.as_deref();
                let charged = pass_timing::time(Pass::RepairCharge, || {
                    record_charge.try_hold(held_now, copy_now, &record, delta_bytes, track_deletes)
                });
                match charged {
                    Ok((record_growth, record_buckets)) => {
                        growth = record_growth;
                        priced = record_buckets;
                    }
                    Err(target) => return Ok(Err(target)),
                }
            }
            let CacheEntry { state, plan, .. } = &mut *held;
            match plan.as_mut() {
                Some(indexes) => {
                    apply_decoded_record_indexed(state, record, &mut sink, indexes, &mut touches)
                        .map(drop)
                }
                None => apply_decoded_record_into(state, record, &mut sink),
            }
            .map(|()| Ok((growth, priced)))
        })
        .map_err(|error| match error.class() {
            ReplayErrorClass::Parse => ReloadReason::Parse,
            ReplayErrorClass::Replay => ReloadReason::ReplayErr,
        })?;
        let (growth, priced) = match step {
            Ok(step) => step,
            Err(target) => {
                if held.plan.is_none() {
                    normalize_round(held);
                }
                return Ok(ChunkEnd::Refused {
                    target,
                    replayed: index,
                });
            }
        };
        // The ids the list-delete sink kept are exactly the
        // `StorageMutation::deleted_list` of each mutation a collecting sink
        // would have held, in the same order (`MutationSink::ListDeletes`
        // keeps that id of each mutation it builds and drops the rest), and
        // the dangling upkeep reads nothing else off the mutations; with the
        // discarding sink (no dangling parents) there is nothing to read.
        let deleted_lists: &[u64] = match &sink {
            MutationSink::ListDeletes(ids) => ids,
            MutationSink::Collect(_) | MutationSink::Discard => &[],
        };
        let touched = held.plan.is_some().then_some(&touches.lists);
        pass_timing::time(Pass::RepairNormalizeChildren, || {
            advance_to_successor(held, deleted_lists.iter().copied(), seq, chain, touched)
        })?;
        // After the record's children normalization, the last of its
        // removals: the bound reads the tables as the record left them.
        if let Some(copy) = copy.as_deref_mut() {
            copy.applied(growth, delta_bytes, priced.as_ref(), &held.state);
        }
    }
    if held.plan.is_none() {
        normalize_round(held);
    }
    Ok(ChunkEnd::Replayed)
}

/// The per-round normalization of an entry holding no plan indexes: zero the
/// external-value indexes the round's records left (no apply step reads
/// them), so the entry is normalized between rounds — and at a cut RP2
/// stopped a round at.
fn normalize_round(held: &mut CacheEntry) {
    let dangling = pass_timing::time(Pass::RepairNormalizeState, || {
        normalize_state(&mut held.state)
    });
    held.dangling_parents.extend(dangling);
}

/// The successor of `entry` after a commit, from a plan made on a **copy**
/// (SR-3.2, SR-3.4): `planned` — the state the commit's plan produced on a
/// clone of `entry.state` ([`CheckedOut::plan_state`]), at `seq` — with
/// `entry`'s dangling parents carried over and kept exactly as the stream
/// repair keeps them when it replays the same commit, and normalized.
/// `mutations` are the persistence mutations the plan sent to its sink, which
/// the commit wrote; `seq` and `commit_chain` are the commit's, the header's
/// values after it. The successor holds no plan indexes
/// ([`CacheEntry::plan`]); its next plan in place builds them.
///
/// The cached route plans in place and settles a commit with
/// [`CheckedOut::advance_committed`] instead, the same step on the held
/// entry; this is the copy-based oracle its shadow check compares against,
/// and the tests', so it is compiled into test and test-support builds only.
///
/// Built by the same step the repair applies per replayed record, so the
/// entry a commit swaps in equals the entry another replica's repair reaches
/// across that commit, which the sampled audit compares with the store:
///
/// - a `DeleteChildrenList`, or an empty `SetChildren` (the store's delete),
///   of a dangling parent removes it from the set — the stored list is gone,
///   and nothing in `planned` shows that;
/// - a children list in `planned` under a parent that is neither the root nor
///   a live section is dropped, and its parent added to the set;
/// - external-value indexes are zeroed and empty lists dropped
///   ([`normalize_state`]).
///
/// The mutations are read only while the set is non-empty, as the repair
/// does; a plan run with the discarding sink may pass an empty slice then.
///
/// **Not weighed here.** The successor's `state_weight_bytes` is carried
/// over from `entry`, a placeholder, and its index weight is 0 (it holds no
/// indexes): [`CheckedOut::put_back`] compacts the state and sets both
/// weights before the per-entry cap check, so weighing here would walk the
/// whole state once more for a value nothing reads.
///
/// # Errors
///
/// [`ReloadReason::DanglingAdopted`] when `planned` makes one of the entry's
/// dangling parents a live section ([`plan_adopts_dangling_parent`]): a
/// stored list sat under that id which `planned` does not hold, so no
/// successor is swapped in. The commit still landed; the caller discards the
/// guard ([`CheckedOut::discard`]) and the next call reloads.
#[cfg(any(test, feature = "test-support"))]
pub fn successor_entry(
    entry: &CacheEntry,
    planned: CompactState,
    mutations: &[StorageMutation],
    seq: u64,
    commit_chain: String,
) -> Result<CacheEntry, ReloadReason> {
    // At `entry`'s cut until the step advances it. The weight is `entry`'s,
    // a placeholder that put-back replaces (doc above).
    let mut successor = CacheEntry {
        object_id: entry.object_id,
        commit_chain: entry.commit_chain.clone(),
        seq: entry.seq,
        dangling_parents: entry.dangling_parents.clone(),
        state: planned,
        state_weight_bytes: entry.state_weight_bytes,
        index_weight_bytes: 0,
        last_used: Instant::now(),
        plan: None,
    };
    pass_timing::time(Pass::SuccessorNormalize, || {
        advance_to_successor(
            &mut successor,
            mutations.iter().filter_map(StorageMutation::deleted_list),
            seq,
            commit_chain,
            None,
        )?;
        let dangling = normalize_state(&mut successor.state);
        successor.dangling_parents.extend(dangling);
        Ok::<(), ReloadReason>(())
    })?;
    Ok(successor)
}

/// One commit's step on an entry whose state already has the commit applied
/// — by replay in the repair, by the plan for [`successor_entry`]: refuse an
/// adopted dangling parent, keep the dangling-parent set from
/// `deleted_lists`, the ids of the children lists the commit deleted, normalize
/// the children lists, and advance the cut to `seq` and `commit_chain`.
/// External-value indexes are left for the caller's [`normalize_state`].
///
/// `deleted_lists` is each [`StorageMutation::deleted_list`](crate::domain::delta::StorageMutation::deleted_list) of the commit's
/// mutations (a `DeleteChildrenList`, or an empty `SetChildren`): the ids the
/// repair's list-delete sink kept (`MutationSink::ListDeletes`), or those a
/// caller holding the mutations reads off them. It is iterated only while
/// the dangling set is non-empty, so a caller whose plan ran with the
/// discarding sink may pass none then.
///
/// With `touched`, the lists the commit wrote, only those are normalized
/// (`domain::plan_index::normalize_touched`), and the entry's plan indexes
/// keep their list sum; without, every list is.
pub(crate) fn advance_to_successor(
    held: &mut CacheEntry,
    deleted_lists: impl IntoIterator<Item = u64>,
    seq: u64,
    commit_chain: String,
    touched: Option<&std::collections::HashSet<u64>>,
) -> Result<(), ReloadReason> {
    debug_assert!(
        touched.is_some() || held.plan.is_none(),
        "an entry holding plan indexes names the lists it touched, so they keep their list sum"
    );
    if adopts_dangling_parent(&held.dangling_parents, &held.state) {
        return Err(ReloadReason::DanglingAdopted);
    }
    if !held.dangling_parents.is_empty() {
        for section_id in deleted_lists {
            held.dangling_parents.remove(&section_id);
        }
    }
    // A list written under a missing parent: dropped, and its parent
    // listed (again, if this record rewrote a listed parent's list).
    let dangling = match (touched, held.plan.as_mut()) {
        (Some(lists), Some(indexes)) => normalize_touched(&mut held.state, indexes, lists),
        _ => normalize_children(&mut held.state),
    };
    held.dangling_parents.extend(dangling);
    held.seq = seq;
    held.commit_chain = commit_chain;
    Ok(())
}

/// [`successor_entry`]: the dangling-parent upkeep a commit's mutations
/// drive, the normalization, the adoption refusal and the advanced cut; and
/// the repair's indexed replay of one record on an entry with dangling
/// parents, the same through its list-delete sink as through a collecting
/// one. The repair's use of the same step is otherwise covered by the cache
/// equivalence proptest and the repair's integration tests.
#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use oodf_ffi::{CommandCreateSection, OwnedDelta, StructCommand};

    use super::*;
    use crate::domain::delta::apply_delta;
    use crate::domain::types::{compact, FieldRows, ObjectState};
    use crate::domain::types::{Field, FieldValue, Section};

    const OBJECT_ID: u64 = 42;
    /// The entry's cut.
    const SEQ: u64 = 7;
    const CHAIN: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    /// The commit's `commit_chain`.
    const NEXT_CHAIN: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    /// Two dangling parents: neither the root nor a live section.
    const DANGLING: u64 = 9;
    const OTHER_DANGLING: u64 = 8;
    /// A live section under the root.
    const LIVE: u64 = 1;
    /// An orphan: a live section whose parent is [`DANGLING`].
    const ORPHAN: u64 = 5;

    fn section(section_id: u64, parent_id: u64) -> Section {
        Section {
            section_id,
            parent_id,
            name: format!("s{section_id}"),
            section_type: 6,
            section_order: section_id as u32,
            first_version: 1,
            last_version: 1,
        }
    }

    /// A normalized entry at [`SEQ`]: [`LIVE`] under the root, [`ORPHAN`]
    /// under [`DANGLING`] (whose stored list the entry does not hold), and
    /// both [`DANGLING`] and [`OTHER_DANGLING`] listed as dangling.
    fn entry() -> CacheEntry {
        let mut state = ObjectState {
            object_id: OBJECT_ID,
            sequence_num: SEQ,
            first_version: 1,
            ..ObjectState::default()
        };
        state.sections.insert(LIVE, section(LIVE, 0));
        state.sections.insert(ORPHAN, section(ORPHAN, DANGLING));
        state.children.insert(0, vec![LIVE]);
        CacheEntry::new(
            OBJECT_ID,
            CHAIN.to_owned(),
            SEQ,
            HashSet::from([DANGLING, OTHER_DANGLING]),
            compact(&state),
        )
    }

    /// `entry`'s state as a plan that changed nothing structural leaves it,
    /// at the next `seq`.
    fn planned(entry: &CacheEntry) -> CompactState {
        let mut state = entry.state.clone();
        state.sequence_num = SEQ + 1;
        state
    }

    fn successor(
        entry: &CacheEntry,
        planned: CompactState,
        mutations: &[StorageMutation],
    ) -> Result<CacheEntry, ReloadReason> {
        successor_entry(entry, planned, mutations, SEQ + 1, NEXT_CHAIN.to_owned())
    }

    #[test]
    fn successor_entry_delete_children_list_of_a_dangling_parent_removes_it() {
        let entry = entry();
        let next = successor(
            &entry,
            planned(&entry),
            &[StorageMutation::DeleteChildrenList {
                section_id: DANGLING,
            }],
        )
        .expect("no adoption");
        assert_eq!(next.dangling_parents, HashSet::from([OTHER_DANGLING]));
        assert_eq!(
            (next.seq, next.commit_chain.as_str()),
            (SEQ + 1, NEXT_CHAIN)
        );
        assert_eq!(next.object_id, OBJECT_ID);
    }

    /// An empty `SetChildren` is the store's delete of the list; a non-empty
    /// one rewrites it, and its parent stays dangling.
    #[test]
    fn successor_entry_empty_set_children_of_a_dangling_parent_removes_it() {
        let entry = entry();
        let next = successor(
            &entry,
            planned(&entry),
            &[
                StorageMutation::SetChildren {
                    section_id: DANGLING,
                    children: Vec::new(),
                },
                StorageMutation::SetChildren {
                    section_id: OTHER_DANGLING,
                    children: vec![ORPHAN],
                },
            ],
        )
        .expect("no adoption");
        assert_eq!(next.dangling_parents, HashSet::from([OTHER_DANGLING]));
    }

    /// Mutations that delete no dangling parent's list leave the set as it
    /// was.
    #[test]
    fn successor_entry_without_a_children_delete_keeps_the_dangling_parents() {
        let entry = entry();
        let next = successor(
            &entry,
            planned(&entry),
            &[StorageMutation::DeleteChildrenList { section_id: LIVE }],
        )
        .expect("no adoption");
        assert_eq!(next.dangling_parents, entry.dangling_parents);
        assert_eq!(next.state, planned(&entry), "nothing to normalize");
    }

    /// A planned list under a parent that is neither the root nor a live
    /// section is dropped and its parent listed; empty lists are dropped.
    /// The successor is exactly the planned state less those two lists: the
    /// compact form holds no `ext_value_index` to zero (a planned external
    /// row keeps its key and hash only), so normalization changes nothing
    /// else. The weight is not recomputed: it is the predecessor's
    /// placeholder, which put-back replaces.
    #[test]
    fn successor_entry_normalizes_the_planned_state_and_lists_a_new_dangling_parent() {
        const NEW_DANGLING: u64 = 77;
        let entry = entry();
        let mut planned = planned(&entry);
        planned.fields.insert_row(Field {
            section_id: LIVE,
            key_name: "ext".to_owned(),
            value: FieldValue::External {
                hash: [3; 32],
                size: 10,
                ext_value_index: 4,
            },
            set_order: 1,
            authoring_set_order: 1,
            version: SEQ + 1,
        });
        // The state normalization must reach: the plan without the two lists.
        let want = planned.clone();
        planned.children.insert(NEW_DANGLING, vec![ORPHAN]);
        planned.children.insert(LIVE, Vec::new());

        let next = successor(&entry, planned, &[]).expect("no adoption");

        assert_eq!(
            next.dangling_parents,
            HashSet::from([DANGLING, OTHER_DANGLING, NEW_DANGLING])
        );
        assert!(!next.state.children.contains_key(&NEW_DANGLING));
        assert!(
            !next.state.children.contains_key(&LIVE),
            "empty list dropped"
        );
        assert_eq!(
            next.state.fields.ext_hash(LIVE, "ext"),
            Some([3; 32]),
            "the compact row keeps the external hash"
        );
        assert_eq!(
            next.state, want,
            "normalization drops the two lists and changes nothing else"
        );
        assert_eq!(
            next.state_weight_bytes, entry.state_weight_bytes,
            "the predecessor's state weight, carried over as a placeholder"
        );
    }

    /// A real plan that creates a section with a dangling parent's id adopts
    /// it: no successor, whatever the mutations say.
    #[test]
    fn successor_entry_plan_adopting_a_dangling_parent_is_refused() {
        let entry = entry();
        let mut planned = entry.state.clone();
        let mutations = apply_delta(
            &mut planned,
            Some(OwnedDelta {
                struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
                    parent_id: 0,
                    section_id: DANGLING,
                    section_name: "adopted".to_owned(),
                    section_type: 6,
                })],
                ..OwnedDelta::default()
            }),
        )
        .expect("the plan applies");
        assert!(plan_adopts_dangling_parent(&entry, &planned));

        let answer = successor(&entry, planned, &mutations);

        assert_eq!(answer, Err(ReloadReason::DanglingAdopted));
    }

    /// The repair's step for a replayed record through the list-delete sink
    /// (`MutationSink::ListDeletes`), which hands `advance_to_successor` the
    /// ids of the lists the record deleted rather than its mutations: each
    /// dangling parent among them leaves the set, any other id (a live
    /// section's list) changes nothing, a repeat or the order changes
    /// nothing, and no id at all keeps the set — and the cut advances in
    /// every case.
    #[test]
    fn advance_to_successor_removes_each_deleted_list_id_from_the_dangling_parents() {
        let cases: [(&str, Vec<u64>, HashSet<u64>); 4] = [
            (
                "one dangling parent's list",
                vec![DANGLING],
                HashSet::from([OTHER_DANGLING]),
            ),
            (
                "both, beside a live section's list and a repeat",
                vec![LIVE, OTHER_DANGLING, DANGLING, DANGLING],
                HashSet::new(),
            ),
            (
                "a live section's list only",
                vec![LIVE],
                HashSet::from([DANGLING, OTHER_DANGLING]),
            ),
            (
                "none",
                Vec::new(),
                HashSet::from([DANGLING, OTHER_DANGLING]),
            ),
        ];
        for (what, deleted_lists, want) in cases {
            let mut held = entry();
            held.state.sequence_num = SEQ + 1;
            // The record wrote no list; an entry holding plan indexes names
            // the lists it touched, as the repair does.
            let touched = HashSet::new();
            let touched = held.plan.is_some().then_some(&touched);

            advance_to_successor(
                &mut held,
                deleted_lists,
                SEQ + 1,
                NEXT_CHAIN.to_owned(),
                touched,
            )
            .expect("no adoption");

            assert_eq!(held.dangling_parents, want, "{what}");
            assert_eq!(
                (held.seq, held.commit_chain.as_str()),
                (SEQ + 1, NEXT_CHAIN),
                "{what}: the cut advances"
            );
        }
    }

    /// With no dangling parent left to remove, `advance_to_successor` never
    /// reads the deleted lists — so a caller whose record replayed through
    /// the discarding sink, which keeps none, may pass none. An iterator
    /// that panics if read stands in for them.
    #[test]
    fn advance_to_successor_with_no_dangling_parents_never_reads_the_deleted_lists() {
        let mut held = entry();
        held.dangling_parents.clear();
        held.state.sequence_num = SEQ + 1;
        let unread = std::iter::from_fn(|| -> Option<u64> {
            panic!("the deleted lists are read with no dangling parent left")
        });
        let touched = HashSet::new();
        let touched = held.plan.is_some().then_some(&touched);

        advance_to_successor(&mut held, unread, SEQ + 1, NEXT_CHAIN.to_owned(), touched)
            .expect("no adoption");

        assert!(held.dangling_parents.is_empty());
        assert_eq!(held.seq, SEQ + 1);
    }

    /// [`LIVE`]'s child in [`indexed_entry_with_orphans`].
    const LIVE_CHILD: u64 = 2;
    /// [`ORPHAN`]'s child in [`indexed_entry_with_orphans`].
    const ORPHAN_CHILD: u64 = 6;
    /// Parents no section of [`indexed_entry_with_orphans`] has, nor any
    /// dangling one: where the move and the re-parent send a section.
    const MOVE_TARGET: u64 = 77;
    const REPARENT_TARGET: u64 = 66;

    /// An entry with plan indexes and dangling parents, as a cache hit's
    /// repair holds it: [`LIVE`] under the root with [`LIVE_CHILD`] below
    /// it, and [`ORPHAN`] under the dangling [`DANGLING`] (whose stored list
    /// the entry does not hold) with [`ORPHAN_CHILD`] below it;
    /// [`DANGLING`] and [`OTHER_DANGLING`] listed as dangling.
    fn indexed_entry_with_orphans() -> CacheEntry {
        let mut state = ObjectState {
            object_id: OBJECT_ID,
            sequence_num: SEQ,
            first_version: 1,
            ..ObjectState::default()
        };
        for (id, parent) in [
            (LIVE, 0),
            (LIVE_CHILD, LIVE),
            (ORPHAN, DANGLING),
            (ORPHAN_CHILD, ORPHAN),
        ] {
            state.sections.insert(id, section(id, parent));
        }
        state.children.insert(0, vec![LIVE]);
        state.children.insert(LIVE, vec![LIVE_CHILD]);
        state.children.insert(ORPHAN, vec![ORPHAN_CHILD]);
        CacheEntry::new(
            OBJECT_ID,
            CHAIN.to_owned(),
            SEQ,
            HashSet::from([DANGLING, OTHER_DANGLING]),
            compact(&state),
        )
    }

    /// One indexed replay arm: `delta` replayed onto a clone of `base` as
    /// `replay_chunk` replays a record on an entry holding plan indexes —
    /// decoded from its OMPF, applied through the entry's indexes
    /// (`apply_decoded_record_indexed`) into `sink`, then stepped by
    /// `advance_to_successor` with the record's touched lists — with the
    /// deleted-list ids read off the sink: the list-delete sink's own ids, or
    /// a collecting sink's mutations' `StorageMutation::deleted_list`, as a
    /// caller holding the mutations reads them (`successor_entry`). Answers
    /// the successor entry, those ids and the record's touches.
    fn replay_indexed(
        base: &CacheEntry,
        delta: &OwnedDelta,
        mut sink: MutationSink,
    ) -> (CacheEntry, Vec<u64>, PlanTouches) {
        let mut held = base.clone();
        let ompf =
            OmpfContainer::serialize(&oodf_ffi::build_delta(delta).expect("build_delta"), &[])
                .expect("serialize OMPF");
        let seq = held.seq + 1;
        let record = decode_stream_record(held.seq, Some(&ompf), seq).expect("the record decodes");
        let mut touches = PlanTouches::default();
        {
            let CacheEntry { state, plan, .. } = &mut held;
            let indexes = plan
                .as_mut()
                .expect("fixture: the entry holds plan indexes");
            apply_decoded_record_indexed(state, record, &mut sink, indexes, &mut touches)
                .expect("the record applies");
        }
        let deleted: Vec<u64> = match &sink {
            MutationSink::ListDeletes(ids) => ids.clone(),
            MutationSink::Collect(mutations) => mutations
                .iter()
                .filter_map(StorageMutation::deleted_list)
                .collect(),
            MutationSink::Discard => panic!("fixture: a list-delete or collecting sink"),
        };
        advance_to_successor(
            &mut held,
            deleted.iter().copied(),
            seq,
            NEXT_CHAIN.to_owned(),
            Some(&touches.lists),
        )
        .expect("no adoption");
        (held, deleted, touches)
    }

    /// Each children list's `(length, capacity)`, by parent.
    fn list_shapes(state: &CompactState) -> std::collections::BTreeMap<u64, (usize, usize)> {
        state
            .children
            .iter()
            .map(|(&parent, list)| (parent, (list.len(), list.capacity())))
            .collect()
    }

    /// The indexed replay the repair runs on an entry with dangling parents
    /// — through the list-delete sink (`MutationSink::ListDeletes`), which
    /// `replay_chunk` takes whenever the entry lists one — reaches exactly
    /// what the same replay reaches through a collecting sink: the same
    /// deleted-list ids, state, children lists (each at the same length and
    /// capacity), plan indexes (the same content, each equal to a fresh
    /// build of its state, and the same depth verdict), touched lists and
    /// listed children, dangling parents and cut. Records: a delete of an
    /// orphan under a dangling parent, a delete of a live section with a
    /// child (a cascade), a move to a missing parent, a `DiffChangeSection`
    /// re-parent under a missing parent, and a dangling parent created and
    /// deleted again in one record, whose list delete takes it off the
    /// dangling set — the upkeep the list-delete sink exists for: that
    /// record's list-delete ids carry `DANGLING` and its successor no longer
    /// lists it. Self-check: the records delete lists, and some record
    /// removes a dangling parent (adding one would not do).
    /// The unindexed engine's twin of this is `domain::delta`'s
    /// `list_deletes_sink_keeps_exactly_the_collecting_sinks_list_delete_ids`.
    ///
    /// Holds at hashbrown group widths 8 and 16 by construction: both arms
    /// start from clones of one state, which keep its tables' hashers and
    /// bucket layouts, and every assertion reads content — map and set
    /// equality, the plan indexes' sorted name vectors and allocation sums,
    /// `Vec` order, length and capacity — never a hash table's capacity,
    /// layout or iteration order, the only things a group width changes.
    #[test]
    fn indexed_replay_through_the_list_delete_sink_matches_the_collecting_sink() {
        use oodf_ffi::{CommandDeleteSection, CommandDiffChangeSection, CommandMoveSection};

        use crate::domain::plan_index::PlanIndexes;

        let base = indexed_entry_with_orphans();
        assert!(
            base.plan.is_some() && !base.dangling_parents.is_empty(),
            "fixture self-check: an indexed entry with dangling parents, which the repair \
             replays through the list-delete sink"
        );
        let one = |command: StructCommand| OwnedDelta {
            struct_commands: vec![command],
            ..OwnedDelta::default()
        };
        let cases = [
            (
                "a delete of an orphan under a dangling parent",
                one(StructCommand::DeleteSection(CommandDeleteSection {
                    parent_id: DANGLING,
                    section_id: ORPHAN,
                })),
            ),
            (
                "a delete of a live section with a child",
                one(StructCommand::DeleteSection(CommandDeleteSection {
                    parent_id: 0,
                    section_id: LIVE,
                })),
            ),
            (
                "a move to a missing parent",
                one(StructCommand::MoveSection(CommandMoveSection {
                    old_parent_id: 0,
                    new_parent_id: MOVE_TARGET,
                    section_id: LIVE,
                    new_name: "moved".to_owned(),
                })),
            ),
            (
                "a DiffChangeSection re-parent under a missing parent",
                one(StructCommand::DiffChangeSection(CommandDiffChangeSection {
                    parent_id: REPARENT_TARGET,
                    section_id: LIVE,
                    section_name: "reparented".to_owned(),
                    section_type: 6,
                    section_order: 1,
                })),
            ),
            (
                "a dangling parent created and deleted again",
                OwnedDelta {
                    struct_commands: vec![
                        StructCommand::CreateSection(CommandCreateSection {
                            parent_id: 0,
                            section_id: DANGLING,
                            section_name: "transient".to_owned(),
                            section_type: 6,
                        }),
                        StructCommand::DeleteSection(CommandDeleteSection {
                            parent_id: 0,
                            section_id: DANGLING,
                        }),
                    ],
                    ..OwnedDelta::default()
                },
            ),
        ];
        // The dangling parent a record's list delete must take off the set:
        // the last record's `DANGLING`.
        let removes = [None, None, None, None, Some(DANGLING)];
        let mut deleted_ids = 0;
        let mut removals = 0;
        for ((what, delta), removes) in cases.into_iter().zip(removes) {
            let (listed, listed_ids, listed_touches) =
                replay_indexed(&base, &delta, MutationSink::list_deletes());
            let (collected, collected_ids, collected_touches) =
                replay_indexed(&base, &delta, MutationSink::collecting());
            assert_eq!(listed_ids, collected_ids, "{what}: the deleted-list ids");
            assert_eq!(listed.state, collected.state, "{what}: the state");
            assert_eq!(
                list_shapes(&listed.state),
                list_shapes(&collected.state),
                "{what}: each children list's length and capacity"
            );
            let (listed_indexes, collected_indexes) = (
                listed.plan.as_ref().expect("indexed"),
                collected.plan.as_ref().expect("indexed"),
            );
            assert!(
                listed_indexes.same_content_as(collected_indexes),
                "{what}: the plan indexes"
            );
            assert_eq!(
                listed_indexes.depth_ok(),
                collected_indexes.depth_ok(),
                "{what}: the plan indexes' depth verdict"
            );
            for (arm, entry) in [("list-delete", &listed), ("collecting", &collected)] {
                assert!(
                    entry
                        .plan
                        .as_ref()
                        .expect("indexed")
                        .same_content_as(&PlanIndexes::build(&entry.state)),
                    "{what}: the {arm} arm's plan indexes equal a fresh build of its state"
                );
            }
            assert_eq!(
                (
                    &listed_touches.lists,
                    &listed_touches.listed,
                    listed_touches.reset
                ),
                (
                    &collected_touches.lists,
                    &collected_touches.listed,
                    collected_touches.reset
                ),
                "{what}: the touched lists and listed children"
            );
            assert_eq!(
                listed.dangling_parents, collected.dangling_parents,
                "{what}: the dangling parents"
            );
            assert_eq!(
                (listed.seq, listed.commit_chain.as_str()),
                (collected.seq, collected.commit_chain.as_str()),
                "{what}: the cut"
            );
            if let Some(parent) = removes {
                assert!(
                    base.dangling_parents.contains(&parent) && listed_ids.contains(&parent),
                    "{what}: the list-delete sink keeps the delete of dangling parent {parent}'s \
                     list: {listed_ids:?}"
                );
                assert!(
                    !listed.dangling_parents.contains(&parent),
                    "{what}: that delete takes {parent} off the dangling parents: {:?}",
                    listed.dangling_parents
                );
            }
            deleted_ids += listed_ids.len();
            if base
                .dangling_parents
                .iter()
                .any(|parent| !listed.dangling_parents.contains(parent))
            {
                removals += 1;
            }
        }
        assert!(
            deleted_ids > 0 && removals > 0,
            "fixture self-check: the records delete lists ({deleted_ids} ids) and some record \
             removes a dangling parent ({removals} records)"
        );
    }
}
