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

//! The commit-charge seam: how a structural `UpdateLayer`'s commit charges
//! the memory it holds for the object it edits — the checked-out cache entry
//! it plans on in place and what that plan grows, its stream repair's reads
//! and replay, the rows it rewrites, a whole-object read, the commands its transaction queues — into the call's
//! upload reservation (SDD §5.4).
//!
//! That memory's size is known only inside the commit, where the call holds
//! per-object permits, a transaction checkout, a cold-start permit or a
//! checked-out cache entry, and nothing may wait on the upload gate while
//! holding any of them (the gate's lock order,
//! `crate::grpc::upload_admission`). So the seam has non-waiting growth —
//! [`CommitCharge::try_hold`], and [`SharedCommitCharge::try_hold_at_least`]
//! over it — for those places, and a waiting one, [`CommitCharge::hold`],
//! only for a caller that holds nothing else.
//!
//! **The contract.** Storage counts; the upload side prices. At each charge
//! point the commit describes what it is about to hold as an
//! [`ObjectStateNeed`] — counts only — and asks the seam for its
//! [`CommitCharge::target`], the whole object-state term in bytes, which it
//! then holds before it allocates that memory, in one of two ways:
//!
//! - **Grow-only, before the attempt's peak**
//!   ([`SharedCommitCharge::try_hold_at_least`]): hold at least the target,
//!   never less than is held. The cached route's CR0, its stream repair's
//!   RP0, RP1 and RP2, and CR1, the WATCH path's WA0 and WA1, its read's
//!   pre-issue and after-window growth, and a plan's growth through its
//!   budget are grow-only, so a resumed attempt keeps what its recharge
//!   waited for until its peak, and a cached attempt's charge carries into
//!   its WATCH fallback.
//! - **Exact, after the peak** ([`SharedCommitCharge::try_hold`]): hold
//!   exactly the target, which may give back what the peak's transients
//!   needed. Only CR2 and WA3 trim, once the plan is made and the read's
//!   windows are gone. CR2's target keeps the one transient still ahead of
//!   a cached attempt: the table and name-index-vector rebuilds its plan's
//!   rollback could need (`CheckedOut::rollback_rebuild_bytes`, an upper
//!   bound), held in its growth term.
//!
//! The cached route's stream repair (`storage::structural_cache_repair`)
//! charges per round, before each read and before replaying what it read,
//! and per record before applying it:
//!
//! - **RP0**, before each repair read: the held entry at its current weight,
//!   state and plan indexes (`CacheEntry::weights_now`), plus that read's
//!   reply at its entry and byte bounds
//!   ([`RepairTransient::reply_entry_count`], [`RepairTransient::reply_bytes`]).
//! - **RP1**, once the read's chunk passed its checks and before any of it
//!   is replayed: the same entry, plus the chunk's reply at its actual
//!   entries and bytes, the parse of its largest record's delta part
//!   ([`RepairTransient::parse_delta_bytes`]) and the entry's growth by the
//!   replay of every record in it ([`RepairTransient::replay_delta_bytes`]).
//! - **RP2**, before each record of the chunk is applied, once it is
//!   decoded ([`ObjectStateNeed::cached_repair_record`]): the entry as it
//!   stands after the records before it, plus what that record can grow it
//!   by beyond its delta bytes — the tables, children lists and name-index
//!   vectors it grows, and its listings and depth walk as rewrite counts —
//!   and the chunk's reply with the record's own parse and per-byte growth.
//!
//! All three are grow-only, so what they held stays held into the plan,
//! which may spend it as rewrite allowance (what is held beyond CR1's need,
//! which carries no repair transient), until CR2 or WA3 trims it. A refused
//! RP0 or RP1 ends the repair between rounds, and a refused RP2 between two
//! records of a round; either way the entry is a valid cut: the read side
//! puts it back and the attempt suspends.
//!
//! A plan grows the charge as it plans: [`CommitCharge::rewrite_allowance`]
//! is what is held beyond the need's target with no rewrite, priced at the
//! rewrite constants — what the plan may spend on rows it rewrites, lists it
//! rewrites or lists, the name index it builds, the ids its adoption index
//! records, the sections its depth check walks and, 1:1, the tables and
//! name-index vectors its rollback could have to rebuild — and
//! [`ChargeRewriteBudget`] answers the
//! plan's [`RewriteBudget::extend`] by holding at least the need with 5/4 of
//! its rows, list entries, adoption entries and walked sections so far and
//! its index and rollback rebuilds exactly, never waiting. The plan asks at
//! an emission and at each build site before it allocates a structure whose
//! size follows the object (a parent's listing, the name index, the adoption
//! index's ids, a reset's, a cascade's or an adoption's rows, the depth
//! walk), and before a write that can leave a table without the room its
//! rollback needs, so the charge covers that structure before it exists.
//!
//! A refused growth changes nothing; the commit then releases everything it
//! holds (puts its cache entry back, `UNWATCH`es and returns its transaction
//! checkout) and its attempt answers [`NeedsCharge`] with the best-known
//! whole target — after a refused plan extension, the need with the plan's
//! exact need, which the plan counts on to its end (its rollback rebuilds
//! included), its scratch and its rollback running above what is held until
//! the synchronous plan and its rollback return; on the cached route that
//! need is priced on CR1 of the entry as put back, which the resumed attempt
//! holds (a rollback restores contents, not layout), and the target is at
//! least the resumed hit's first charge point, its repair's first RP0 on that
//! entry, so the resumed attempt is not refused there. A resumed plan's
//! rollback-rebuild projections follow the layout it meets, so it can still
//! ask once more.
//! `update_layer`'s retry loop suspends on it, gives the per-object permits
//! back, waits on [`CommitCharge::hold`] for that target, re-admits and
//! resumes the same attempt, so a recharge spends no retry.
//!
//! Declared here and implemented outside `storage`, as
//! [`crate::storage::counters::ObjectIdCounterRepair`] is: the upload
//! implementation (`crate::grpc::upload_admission::UploadCommitCharge`) wraps
//! the call's `UploadReservation` and carries the charge constants and
//! formulas, so `storage` names an abstract seam and gains no dependency on
//! `grpc`. Callers with no upload — every `LayerStore` caller other than the
//! `UpdateLayer` handler — pass [`NoCommitCharge`].
//!
//! The structural cache's sampled audit is no upload but holds its repair on
//! `B` too, through its own holder ([`AuditChargeSource`] opens one
//! [`AuditCharge`] per audit): the same seam, net of the copy's own weight,
//! never waiting, never on the overflow slot, yielding to queued uploads and
//! capped per audit.
//!
//! Inside the call the attempts reach the charge through a
//! [`CommitChargeCell`], a `std::sync::Mutex` locked only inside the
//! synchronous calls of [`SharedCommitCharge`] and never across an await.

use std::sync::{Mutex, PoisonError};

use async_trait::async_trait;

use crate::domain::delta::{MutationAllowance, RewriteBudget, RewriteCount};
use crate::error::ServiceError;

/// What a commit is about to hold for the object it edits, in counts: the
/// input of [`CommitCharge::target`]. Storage fills it in; the upload side
/// prices it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectStateNeed {
    /// The cached structural route, which plans in place on the held cache
    /// entry: the entry, what the plan can allocate beside it, the plan's
    /// rewrite, and the stream repair's transient (RP0, RP1, RP2).
    Cached {
        /// The held entry's estimated weight, all of it held outside every
        /// other charge, since a checked-out entry's resident slot can be
        /// evicted under the call: its state and plan indexes
        /// (`CacheEntry::weight_bytes`) at the handler's hint and from CR1
        /// on; at RP0, RP1 and RP2 the entry as it stands
        /// (`CacheEntry::weights_now`, state and plan indexes); the decoded
        /// state's (`estimate_state_bytes`) at CR0.
        held: u64,
        /// What the attempt can allocate beside the held entry outside its
        /// rewrite, priced 1:1: 0 at RP0 and RP1; at RP2 the tables, children
        /// lists and name-index vectors the record grows, plus the scratch
        /// its apply allocates and frees — its cascades' walks, sets and
        /// purge, a reset's registrations, the list-delete sink's ids and
        /// copies (`estimate_replay_record_growth`, `ReplayRecordGrowth`'s
        /// `growth` + `scratch`); at CR0 the plan-index build's peak
        /// (`PlanIndexes::build_peak_bytes`); at CR1 the tables the plan can
        /// grow in place — each table whose length plus the delta's insert
        /// calls passes its capacity, at the tables its growth holds at once
        /// (at least the doubled table), and a reset's re-allocated
        /// tables (`estimate_in_place_growth_bytes`); 0 at the hint; at CR2,
        /// when the plan's growth is in `held`, the tables and the
        /// name-index vectors the plan's rollback could still have to
        /// rebuild at their pre-plan length
        /// (`CheckedOut::rollback_rebuild_bytes`: what it could rebuild, an
        /// upper bound, each rebuild priced exactly; usually 0).
        growth: u64,
        /// The plan's rewrite: zero before the plan, except the walk a
        /// full depth check will make (`walked_sections`, held from CR1 when
        /// the entry's indexes cannot answer it incrementally); while it
        /// plans, its kept rewrite plus its unretired build-site projections
        /// (scaled by its budget) and its projected rollback rebuilds; after
        /// it, its exact kept count (no index, adoption, walk or rebuild
        /// entries). At RP2 the replayed record's own scratch: its parents'
        /// listings, its depth check's walk and, on an entry without plan
        /// indexes, its per-delta indexes (`estimate_replay_record_growth`).
        rewrite: RewriteCount,
        /// The stream repair's transient: one read's reply, and while its
        /// chunk replays the parse of a record and the entry's growth (RP0,
        /// RP1, RP2). Zero outside the repair, at CR0, CR1 and CR2: the held
        /// term then carries over grow-only into the plan as rewrite
        /// allowance, and CR2 trims it.
        repair: RepairTransient,
    },
    /// The WATCH/MULTI/EXEC path: one structural attempt's whole-object
    /// read, its fixed cost, the plan's rewrite and the commands its commit
    /// queues.
    Watch {
        /// The whole-object read, while it runs; zero outside it.
        snapshot: WatchSnapshot,
        /// Whether the attempt's fixed per-attempt cost is held: every
        /// structural WATCH attempt's, from its head.
        fixed: bool,
        /// The plan's rewrite: zero before the plan; while it plans, its kept
        /// rewrite plus its unretired build-site projections (scaled by its
        /// budget); after it, its exact kept count (no index or adoption
        /// entries).
        rewrite: RewriteCount,
        /// The commands the commit queues at once, by connection.
        queued: QueuedCommands,
    },
}

impl ObjectStateNeed {
    /// The cached route's need for a held entry of estimated weight `held`
    /// with `rewrite` and no growth beside it: the handler's hint.
    pub fn cached(held: u64, rewrite: RewriteCount) -> Self {
        Self::cached_growing(held, 0, rewrite)
    }

    /// The cached route's need at a stream-repair charge point (RP0, RP1):
    /// the held entry of estimated weight `held`, state and plan indexes, as
    /// the hint holds it, with no growth and no rewrite, plus `repair`.
    pub fn cached_repair(held: u64, repair: RepairTransient) -> Self {
        Self::Cached {
            held,
            growth: 0,
            rewrite: RewriteCount::default(),
            repair,
        }
    }

    /// The cached route's need at the stream repair's per-record charge
    /// point (RP2): the held entry of estimated weight `held` as it stands
    /// before the record, the `growth` and `rewrite` the record's replay can
    /// allocate beside it, and `repair`, the round's reply with the record's
    /// parse and per-byte growth.
    pub fn cached_repair_record(
        held: u64,
        growth: u64,
        rewrite: RewriteCount,
        repair: RepairTransient,
    ) -> Self {
        Self::Cached {
            held,
            growth,
            rewrite,
            repair,
        }
    }

    /// The cached route's need for a held entry of estimated weight `held`,
    /// with `growth` bytes the attempt can allocate beside it and `rewrite`:
    /// CR0, CR1 and CR2 (whose growth is the rollback rebuilds still
    /// possible).
    pub fn cached_growing(held: u64, growth: u64, rewrite: RewriteCount) -> Self {
        Self::Cached {
            held,
            growth,
            rewrite,
            repair: RepairTransient::default(),
        }
    }

    /// A structural WATCH attempt's fixed cost alone: its need from the head
    /// of the attempt until its read counts the object.
    pub fn watch_fixed() -> Self {
        Self::Watch {
            snapshot: WatchSnapshot::default(),
            fixed: true,
            rewrite: RewriteCount::default(),
            queued: QueuedCommands::default(),
        }
    }

    /// The same need with `rewrite` in place of its own; a cached need keeps
    /// its repair transient.
    pub fn with_rewrite(self, rewrite: RewriteCount) -> Self {
        match self {
            Self::Cached {
                held,
                growth,
                repair,
                ..
            } => Self::Cached {
                held,
                growth,
                rewrite,
                repair,
            },
            Self::Watch {
                snapshot,
                fixed,
                queued,
                ..
            } => Self::Watch {
                snapshot,
                fixed,
                rewrite,
                queued,
            },
        }
    }

    /// The need's rewrite.
    pub fn rewrite(&self) -> RewriteCount {
        match self {
            Self::Cached { rewrite, .. } | Self::Watch { rewrite, .. } => *rewrite,
        }
    }
}

/// The cached route's stream-repair transient, in counts and bytes
/// ([`ObjectStateNeed::Cached`]): what one round of the repair holds beside
/// the entry while it reads and replays (`storage::structural_cache_repair`,
/// RP0, RP1 and RP2). Zero outside the repair.
///
/// The reply is priced in two terms, one per entry and one per byte, so a
/// read's bound is priced by both its entry and its byte bound rather than
/// at the per-byte cost of the smallest records.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RepairTransient {
    /// The read's stream entries: at RP0 the read's entry bound (its
    /// `max_entries`), before it is issued; at RP1 and RP2 the chunk's
    /// entries.
    pub reply_entry_count: u64,
    /// The read's reply: at RP0 the read's byte bound, before it is issued;
    /// at RP1 and RP2 the chunk's own byte tally (its ids and every field
    /// name and value, its OMPFs whole).
    pub reply_bytes: u64,
    /// The delta part one replayed record parses (records parse one at a
    /// time; zero at RP0): at RP1 the chunk's largest record's, at RP2 the
    /// record's own. A record whose OMPF container does not parse counts its
    /// whole OMPF.
    pub parse_delta_bytes: u64,
    /// The delta parts the replay grows the entry by, beyond what RP2 prices
    /// per record (zero at RP0), counted as for `parse_delta_bytes`: at RP1
    /// the chunk's, summed; at RP2 the record's own.
    pub replay_delta_bytes: u64,
}

/// A WATCH attempt's whole-object read, in counts
/// ([`ObjectStateNeed::Watch`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WatchSnapshot {
    /// The object's entries as the read's count window reports them:
    /// sections + tombstones + fields + time samples, plus two per children
    /// list — its `LLEN` and its ranged `LRANGE` — for the larger of
    /// sections + 2 and the `obj:chs` count. Each is a round-1 member or a
    /// command the read holds until its round is issued.
    pub entries: u64,
    /// Wire commands in flight at once: `min(commands, FANOUT_WINDOW)`, a
    /// value row's two-command read counting two.
    pub in_flight: u64,
    /// Plain wire commands in flight at once, each holding its own future:
    /// the most any plain window of the read has issued so far — a round-1
    /// index page, a lone children list too long for a script window, or a
    /// window issued plain after the read fell back from its window script.
    /// A script window holds one future whatever its entries and adds none.
    /// 0 until the count windows' charge (WA1), which holds round 1's index
    /// pages, one plain command per index with members; grow-only.
    pub plain_in_flight: u64,
    /// Items the read has tallied so far (hash fields and members), which
    /// its decoded state retains; before a window or page is issued, plus
    /// the items it can add (its commands' item widths).
    pub items: u64,
    /// One window's bytes: at the count windows (WA1), the read's projected
    /// reply and script-request bound from its counts, at most
    /// `FANOUT_WINDOW_BYTES`; before a window or page is issued, the sum of
    /// its commands' per-command reply bounds, plus a script window's request
    /// bytes (its entries' ids and the column lists); after it, the bytes its
    /// replies held.
    pub window_bytes: u64,
    /// Name bytes the read retains: the round-1 field-index member bytes and
    /// the section-name bytes it has read so far. A name is kept as several
    /// copies (the member, its round-2 key, the command's id, the decoded
    /// name), so it is charged per byte rather than per entry.
    pub name_bytes: u64,
}

/// The commands a WATCH commit queues at once, by the connection whose
/// queue holds them ([`ObjectStateNeed::Watch`]). Each count is of commands
/// in one connection's queues at the same time, already capped at what the
/// commit can have in flight there.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct QueuedCommands {
    /// The `MULTI` … `EXEC` pipeline on the transaction connection: `MULTI`,
    /// every mutation's commands, the external-value index writes, the
    /// commit-token record, the `XADD`, the `XTRIM`, the `CLIENT ID` witness
    /// and `EXEC`, plus, for a commit that writes a value row, the commit
    /// release ring's `HDEL`, `HSET` and `PEXPIRE`.
    pub txn: u64,
    /// The external-value pre-image read's commands in flight at once (a
    /// fast-shaped commit only; zero on the snapshot branch).
    pub pre_image: u64,
    /// The keyspace type probe's keys in flight at once, plus the commit
    /// release ring's window `HMGET` that runs beside it when the commit
    /// writes a value row.
    pub probe: u64,
}

/// An attempt that could not hold the charge it needed without waiting, and
/// released everything it held: `update_layer`'s retry loop waits on
/// [`CommitCharge::hold`] for `target`, with the per-object permits given
/// back, and resumes the same attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NeedsCharge {
    /// The best-known whole object-state target: what the next run of the
    /// attempt is expected to need.
    pub target: u64,
    /// The attempt's `path` label, one of
    /// [`crate::observability::metrics::UPDATE_LAYER_PATHS`]:
    /// `live_edit_update_layer_recharges_total{path}`.
    pub path: &'static str,
}

/// A commit's handle on the memory charge of the call it runs in.
///
/// Every amount is the **object-state term**: bytes on top of the call's
/// base processing charge, which the implementation holds throughout. A
/// target is the whole term the commit needs held from now on, not an
/// increment: a target below [`Self::held`] shrinks the term to it, giving
/// the difference back. One caller holds another amount through it: between
/// its calls, the `UpdateLayer` handler holds one extra copy of the upload's
/// external values on top of the term while the checkpoint-recovery
/// re-issue's pre-store runs, and gives it back before any commit runs (see
/// `grpc::upload_admission::UploadCommitCharge`).
#[async_trait]
pub trait CommitCharge: Send {
    /// The object-state bytes held now. Saturating arithmetic only:
    /// [`NoCommitCharge`] answers `u64::MAX`.
    fn held(&self) -> u64;

    /// The object-state target, in bytes, of `need`: what holding it costs.
    /// [`NoCommitCharge`] answers 0.
    fn target(&self, need: &ObjectStateNeed) -> u64;

    /// What a plan may spend on its rewrite now: [`Self::held`] less the
    /// target of `need` with no rewrite, priced at the rewrite constants.
    /// [`NoCommitCharge`] answers an unbounded allowance.
    fn rewrite_allowance(&self, need: &ObjectStateNeed) -> MutationAllowance;

    /// Hold exactly `target` object-state bytes, at once or not at all —
    /// never waits, so it may be called holding per-object permits, a
    /// transaction checkout, a cold-start permit or a checked-out entry.
    /// Returns whether the target is now held; `false` changes nothing. A
    /// target below [`Self::held`] shrinks the term to it: the exact trim,
    /// for a charge point after the attempt's peak (CR2, WA3). A charge point
    /// before the peak grows only
    /// ([`SharedCommitCharge::try_hold_at_least`]).
    fn try_hold(&mut self, target: u64) -> bool;

    /// Hold exactly `target` object-state bytes, waiting for room when it
    /// does not fit now. **May wait on the upload gate, so call it only while
    /// holding nothing else** — no per-object permit, transaction checkout,
    /// cold-start permit or checked-out entry — or it can deadlock with the
    /// gate's overflow slot.
    ///
    /// # Errors
    ///
    /// `UNAVAILABLE "server shutting down"`, carrying the
    /// `SERVER_SHUTTING_DOWN` refusal detail and its retry hint
    /// ([`ServiceError::UnavailableWithDetail`]), when the service shuts down
    /// while it waits; the wait is otherwise bounded only by the RPC
    /// deadline, which drops the call.
    async fn hold(&mut self, target: u64) -> Result<(), ServiceError>;
}

/// The charge of a commit that runs in no upload: nothing is reserved, so
/// every target counts as held — [`CommitCharge::held`] is `u64::MAX`,
/// [`CommitCharge::target`] is 0, the rewrite allowance is unbounded,
/// [`CommitCharge::try_hold`] always succeeds and [`CommitCharge::hold`]
/// never waits.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoCommitCharge;

#[async_trait]
impl CommitCharge for NoCommitCharge {
    fn held(&self) -> u64 {
        u64::MAX
    }

    fn target(&self, _need: &ObjectStateNeed) -> u64 {
        0
    }

    fn rewrite_allowance(&self, _need: &ObjectStateNeed) -> MutationAllowance {
        MutationAllowance::UNBOUNDED
    }

    fn try_hold(&mut self, _target: u64) -> bool {
        true
    }

    async fn hold(&mut self, _target: u64) -> Result<(), ServiceError> {
        Ok(())
    }
}

/// Shared access to a call's [`CommitCharge`] from inside its attempts: the
/// synchronous calls only, so none can wait on the gate. Implemented by
/// [`CommitChargeCell`]; the attempts take it as `&dyn SharedCommitCharge`.
pub trait SharedCommitCharge: Sync {
    /// [`CommitCharge::held`].
    fn held(&self) -> u64;
    /// [`CommitCharge::target`].
    fn target(&self, need: &ObjectStateNeed) -> u64;
    /// [`CommitCharge::rewrite_allowance`].
    fn rewrite_allowance(&self, need: &ObjectStateNeed) -> MutationAllowance;
    /// [`CommitCharge::try_hold`]: hold exactly `target`, never waiting —
    /// the exact trim after an attempt's peak (CR2, WA3).
    fn try_hold(&self, target: u64) -> bool;
    /// Hold at least `target`: `max(held, target)`, never waiting. A no-op
    /// answering `true` when `target` is at most what is held; otherwise
    /// [`CommitCharge::try_hold`] of `target`. Every charge point before an
    /// attempt's peak is grow-only, so the room a recharge waited for is not
    /// given back to the gate before the attempt reaches the memory it is
    /// for.
    fn try_hold_at_least(&self, target: u64) -> bool;
    /// [`Self::try_hold`] of `need`'s target: the exact trim. Returns the
    /// target, held, or `Err` with it when refused.
    fn try_hold_need(&self, need: &ObjectStateNeed) -> Result<u64, u64> {
        let target = self.target(need);
        if self.try_hold(target) {
            Ok(target)
        } else {
            Err(target)
        }
    }
    /// [`Self::try_hold_at_least`] of `need`'s target: grow-only. Returns the
    /// target, held at least, or `Err` with it when refused.
    fn try_hold_at_least_need(&self, need: &ObjectStateNeed) -> Result<u64, u64> {
        let target = self.target(need);
        if self.try_hold_at_least(target) {
            Ok(target)
        } else {
            Err(target)
        }
    }
}

/// The [`RewriteBudget`] a bounded plan grows the call's commit charge
/// through, as it plans (SDD §5.4): created per plan call over the call's
/// charge and the need the plan's memory is part of, and dropped with the
/// plan.
///
/// Asked for more because the plan's rewrite has reached `needed` — its kept
/// rewrite plus the projections its build sites made before allocating (a
/// parent's listing, the name index, the adoption index's ids, a reset's, a
/// cascade's or an adoption's rows, the depth walk) — it holds at least the
/// target of `need` with that rewrite, rows, list entries, adoption entries
/// and walked sections scaled by 5/4
/// and each rounded up, the index's sections and the rollback rebuilds
/// exactly (each one's size is known when it is projected), never waiting
/// ([`SharedCommitCharge::try_hold_at_least`]), and answers the new
/// allowance: what is now held beyond `need`'s target with no rewrite
/// ([`CommitCharge::rewrite_allowance`]). The 5/4 headroom keeps the calls
/// to a few dozen for a whole-list rewrite of tens of thousands of children,
/// or an adoption index recording them one id at a time, and over-holds at
/// most a quarter of the rewrite until the exact trim (CR2, WA3). A refused
/// growth answers `None`: the plan overflows and counts on to its exact need
/// (kept plus unretired projections, at its peak, every rollback rebuild it
/// could need included), and the attempt suspends for the target of `need`
/// with it, unscaled — on the cached route, priced on CR1 of the entry as put
/// back after the rollback, not the pre-plan CR1, since that is what the
/// resumed attempt's CR1 holds, and never below its repair's first RP0 on
/// that entry (module docs). The refused plan's scratch, and
/// its rollback, run above what is held until the synchronous plan and its
/// rollback return (SDD §5.4's residual).
pub struct ChargeRewriteBudget<'a> {
    charge: &'a dyn SharedCommitCharge,
    need: ObjectStateNeed,
}

impl<'a> ChargeRewriteBudget<'a> {
    /// A budget on `charge` for a plan whose memory, beside its rewrite, is
    /// `need` (its own rewrite is ignored).
    pub fn new(charge: &'a dyn SharedCommitCharge, need: ObjectStateNeed) -> Self {
        Self {
            charge,
            need: need.with_rewrite(RewriteCount::default()),
        }
    }
}

/// `n × 5/4`, rounded up, saturating.
fn with_quarter_headroom(n: u64) -> u64 {
    n.saturating_add(n.div_ceil(4))
}

impl RewriteBudget for ChargeRewriteBudget<'_> {
    fn extend(&mut self, needed: RewriteCount) -> Option<u64> {
        // The name index and the rollback rebuilds exactly: each one's size
        // is known when it is projected, and a rebuild is asked at most once
        // per table. The adoption index's ids and the depth walk's sections
        // grow a little at a time, like rows and list entries, so they take
        // the same headroom and the asks stay few.
        let scaled = RewriteCount {
            rows: with_quarter_headroom(needed.rows),
            list_entries: with_quarter_headroom(needed.list_entries),
            indexed_sections: needed.indexed_sections,
            adoption_entries: with_quarter_headroom(needed.adoption_entries),
            walked_sections: with_quarter_headroom(needed.walked_sections),
            rollback_rebuild_bytes: needed.rollback_rebuild_bytes,
        };
        self.charge
            .try_hold_at_least_need(&self.need.with_rewrite(scaled))
            .ok()
            .map(|_| self.charge.rewrite_allowance(&self.need).bytes)
    }
}

/// A call's [`CommitCharge`] behind a `std::sync::Mutex`, shared by its
/// attempts ([`SharedCommitCharge`]). The lock is taken only inside those
/// synchronous calls, never across an await; the waiting
/// [`CommitCharge::hold`] goes through [`Self::get_mut`], between attempts,
/// with no attempt alive to share the cell.
pub struct CommitChargeCell<'c> {
    inner: Mutex<&'c mut (dyn CommitCharge + 'c)>,
}

impl<'c> CommitChargeCell<'c> {
    /// Share `charge` with the call's attempts.
    pub fn new(charge: &'c mut (dyn CommitCharge + 'c)) -> Self {
        Self {
            inner: Mutex::new(charge),
        }
    }

    /// The charge itself, for the waiting [`CommitCharge::hold`] between
    /// attempts.
    pub fn get_mut(&mut self) -> &mut (dyn CommitCharge + 'c) {
        &mut **self.inner.get_mut().unwrap_or_else(PoisonError::into_inner)
    }

    fn with<R>(&self, f: impl FnOnce(&mut (dyn CommitCharge + 'c)) -> R) -> R {
        let mut guard = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        f(&mut **guard)
    }
}

impl SharedCommitCharge for CommitChargeCell<'_> {
    fn held(&self) -> u64 {
        self.with(|charge| charge.held())
    }

    fn target(&self, need: &ObjectStateNeed) -> u64 {
        self.with(|charge| charge.target(need))
    }

    fn rewrite_allowance(&self, need: &ObjectStateNeed) -> MutationAllowance {
        self.with(|charge| charge.rewrite_allowance(need))
    }

    fn try_hold(&self, target: u64) -> bool {
        self.with(|charge| charge.try_hold(target))
    }

    // One lock for the comparison and the growth, so nothing changes what is
    // held between them.
    fn try_hold_at_least(&self, target: u64) -> bool {
        self.with(|charge| target <= charge.held() || charge.try_hold(target))
    }
}

/// The sampled audit's charge for one audit: what its repair of the entry's
/// copy to the read's cut holds (`storage::structural_cache_audit`,
/// "Memory"), on the upload budget `B` beside the uploads, through the same
/// non-waiting seam as a structural call's repair ([`SharedCommitCharge`]).
///
/// The audit is no upload, so its holder differs from a call's in three
/// ways, which the implementation keeps (`grpc::upload_admission`'s
/// `UploadBudgetAuditCharges`): a growth never waits and never takes the
/// overflow slot; it yields to every queued upload and always leaves one
/// upload admission's room free; and it is refused above [`Self::cap`]. A
/// refused growth changes nothing, and the audit skips. Its targets are
/// net of the copy's own weight, which the line's audit term already funds
/// ([`AuditChargeSource::open`]'s `base`). Dropping it gives back everything
/// it holds and serves the queues.
pub trait AuditCharge: SharedCommitCharge + Send {
    /// The most this audit may hold: a growth whose target is above it is
    /// refused, whatever room `B` has.
    fn cap(&self) -> u64;

    /// The most it has held at once since it was opened.
    fn peak(&self) -> u64;
}

/// Where the sampled audit opens its [`AuditCharge`]: one per audit, held
/// from the repair's first charge point until the compare ends. Declared
/// here, implemented on the upload side
/// (`grpc::upload_admission::UploadBudgetAuditCharges`), and joined in
/// `main.rs`, as [`CommitCharge`] is, so `storage` gains no dependency on
/// `grpc`.
pub trait AuditChargeSource: Send + Sync {
    /// An empty charge for one audit whose copy of the entry weighs `base`
    /// (its stored state weight): every target it answers is the need's
    /// price less `base`, which the line's audit term funds outside `B`.
    fn open(&self, base: u64) -> Box<dyn AuditCharge>;
}

/// An [`AuditChargeSource`] whose charges refuse every growth: what an audit
/// that never starts holds (`StructuralCacheAudit::disabled`). Its targets
/// are `u64::MAX`, above its cap of 0, since `storage` holds no prices.
#[derive(Debug, Clone, Copy, Default)]
pub struct RefusingAuditCharges;

impl AuditChargeSource for RefusingAuditCharges {
    fn open(&self, _base: u64) -> Box<dyn AuditCharge> {
        Box::new(RefusingAuditCharge)
    }
}

/// [`RefusingAuditCharges`]' charge.
struct RefusingAuditCharge;

impl SharedCommitCharge for RefusingAuditCharge {
    fn held(&self) -> u64 {
        0
    }

    fn target(&self, _need: &ObjectStateNeed) -> u64 {
        u64::MAX
    }

    fn rewrite_allowance(&self, _need: &ObjectStateNeed) -> MutationAllowance {
        MutationAllowance {
            bytes: 0,
            ..MutationAllowance::UNBOUNDED
        }
    }

    fn try_hold(&self, target: u64) -> bool {
        target == 0
    }

    fn try_hold_at_least(&self, target: u64) -> bool {
        target == 0
    }
}

impl AuditCharge for RefusingAuditCharge {
    fn cap(&self) -> u64 {
        0
    }

    fn peak(&self) -> u64 {
        0
    }
}

/// **Test-support only.** An [`AuditChargeSource`] whose charges hold
/// everything already, as [`NoCommitCharge`] does: every target is 0, every
/// growth is granted and nothing is held on any budget. For a test whose
/// audit must not be limited by memory.
#[cfg(any(test, feature = "test-support"))]
#[derive(Debug, Clone, Copy, Default)]
pub struct UnboundedAuditCharges;

#[cfg(any(test, feature = "test-support"))]
impl AuditChargeSource for UnboundedAuditCharges {
    fn open(&self, _base: u64) -> Box<dyn AuditCharge> {
        Box::new(UnboundedAuditCharge)
    }
}

/// [`UnboundedAuditCharges`]' charge.
#[cfg(any(test, feature = "test-support"))]
struct UnboundedAuditCharge;

#[cfg(any(test, feature = "test-support"))]
impl SharedCommitCharge for UnboundedAuditCharge {
    fn held(&self) -> u64 {
        u64::MAX
    }

    fn target(&self, _need: &ObjectStateNeed) -> u64 {
        0
    }

    fn rewrite_allowance(&self, _need: &ObjectStateNeed) -> MutationAllowance {
        MutationAllowance::UNBOUNDED
    }

    fn try_hold(&self, _target: u64) -> bool {
        true
    }

    fn try_hold_at_least(&self, _target: u64) -> bool {
        true
    }
}

#[cfg(any(test, feature = "test-support"))]
impl AuditCharge for UnboundedAuditCharge {
    fn cap(&self) -> u64 {
        u64::MAX
    }

    fn peak(&self) -> u64 {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A commit outside any upload holds everything already: it is never
    /// refused and never waits, whatever the target.
    #[tokio::test]
    async fn no_commit_charge_holds_every_target_and_never_refuses() {
        let mut charge = NoCommitCharge;
        assert_eq!(charge.held(), u64::MAX);
        for target in [0, 1, u64::MAX] {
            assert!(charge.try_hold(target), "try_hold({target})");
            assert!(charge.hold(target).await.is_ok(), "hold({target})");
            assert_eq!(charge.held(), u64::MAX, "after {target}");
        }
    }

    /// Every need is free for a commit outside any upload, and its plan's
    /// rewrite is unbounded, so its budget is never asked.
    #[test]
    fn no_commit_charge_prices_every_need_at_zero_with_an_unbounded_allowance() {
        let charge = NoCommitCharge;
        let rewrite = RewriteCount {
            rows: 9,
            list_entries: 9,
            indexed_sections: 9,
            adoption_entries: 9,
            walked_sections: 9,
            rollback_rebuild_bytes: 9,
        };
        for need in [
            ObjectStateNeed::cached(u64::MAX, rewrite),
            ObjectStateNeed::watch_fixed().with_rewrite(rewrite),
        ] {
            assert_eq!(charge.target(&need), 0, "{need:?}");
            assert_eq!(
                charge.rewrite_allowance(&need).bytes,
                u64::MAX,
                "{need:?}: unbounded"
            );
        }
    }

    /// The cached route's stream-repair need (RP0, RP1) is the entry at
    /// `held`, as [`ObjectStateNeed::cached`] holds it, with no growth and no
    /// rewrite, plus the transient. `with_rewrite` replaces the rewrite alone,
    /// so a need's transient survives it; `cached` and `cached_growing` carry
    /// none.
    #[test]
    fn cached_repair_need_keeps_its_transient_through_with_rewrite() {
        let repair = RepairTransient {
            reply_entry_count: 4,
            reply_bytes: 3,
            parse_delta_bytes: 2,
            replay_delta_bytes: 1,
        };
        let need = ObjectStateNeed::cached_repair(10, repair);
        assert_eq!(
            need,
            ObjectStateNeed::Cached {
                held: 10,
                growth: 0,
                rewrite: RewriteCount::default(),
                repair,
            }
        );
        let rewrite = RewriteCount {
            rows: 4,
            ..RewriteCount::default()
        };
        let rewritten = need.with_rewrite(rewrite);
        assert_eq!(
            rewritten,
            ObjectStateNeed::Cached {
                held: 10,
                growth: 0,
                rewrite,
                repair,
            },
            "with_rewrite keeps the repair transient"
        );
        assert_eq!(rewritten.rewrite(), rewrite);
        for plain in [
            ObjectStateNeed::cached(10, rewrite),
            ObjectStateNeed::cached_growing(10, 12, rewrite),
        ] {
            assert!(
                matches!(plain, ObjectStateNeed::Cached { repair, .. }
                    if repair == RepairTransient::default()),
                "{plain:?} carries no repair transient"
            );
        }
    }

    /// A test [`CommitCharge`] that never refuses and prices a need at its
    /// rewrite's rows + list entries + index sections + adoption entries, one
    /// byte each: enough to read back what a charge point asked to hold.
    #[derive(Default)]
    struct CountingCharge {
        held: u64,
        holds: Vec<u64>,
    }

    #[async_trait]
    impl CommitCharge for CountingCharge {
        fn held(&self) -> u64 {
            self.held
        }

        fn target(&self, need: &ObjectStateNeed) -> u64 {
            need.rewrite().bytes(1, 1, 1, 1, 1)
        }

        fn rewrite_allowance(&self, need: &ObjectStateNeed) -> MutationAllowance {
            MutationAllowance {
                bytes: self
                    .held
                    .saturating_sub(self.target(&need.with_rewrite(RewriteCount::default()))),
                row_bytes: 1,
                list_entry_bytes: 1,
                index_entry_bytes: 1,
                adoption_entry_bytes: 1,
                walk_entry_bytes: 1,
            }
        }

        fn try_hold(&mut self, target: u64) -> bool {
            self.held = target;
            self.holds.push(target);
            true
        }

        async fn hold(&mut self, target: u64) -> Result<(), ServiceError> {
            self.held = target;
            Ok(())
        }
    }

    /// Grow-only against exact: through the cell,
    /// `try_hold_at_least` of a target at or below what is held is a no-op
    /// that never reaches the charge, and above it holds the target; the
    /// exact `try_hold` — CR2's and WA3's trim — sets the target, down too.
    #[test]
    fn commit_charge_cell_grows_only_at_least_and_trims_exactly() {
        let mut inner = CountingCharge::default();
        {
            let cell = CommitChargeCell::new(&mut inner);
            assert!(cell.try_hold_at_least(100));
            assert_eq!(cell.held(), 100);
            assert!(
                cell.try_hold_at_least(40),
                "below what is held: answered held"
            );
            assert!(cell.try_hold_at_least(100), "equal: answered held");
            assert_eq!(
                cell.held(),
                100,
                "a grow-only point never shrinks the charge"
            );
            assert!(cell.try_hold(30), "the exact trim");
            assert_eq!(cell.held(), 30, "the exact trim gives back the rest");
        }
        assert_eq!(
            inner.holds,
            vec![100, 30],
            "the no-op growths never reached the charge"
        );
    }

    /// [`ChargeRewriteBudget`] holds at least the need with 5/4 of the plan's
    /// rows, list entries and adoption-index ids, each rounded up, and its
    /// name-index sections exactly, then answers the new
    /// allowance: what is held beyond the need with no rewrite.
    #[test]
    fn charge_rewrite_budget_holds_five_quarters_of_its_counts_and_the_index_exactly() {
        let mut inner = CountingCharge::default();
        let asked = RewriteCount {
            rows: 4,
            list_entries: 10,
            indexed_sections: 7,
            adoption_entries: 1_000,
            walked_sections: 0,
            rollback_rebuild_bytes: 0,
        };
        let held_bytes = {
            let cell = CommitChargeCell::new(&mut inner);
            let mut budget = ChargeRewriteBudget::new(&cell, ObjectStateNeed::cached(1, asked));
            budget.extend(asked)
        };
        // ⌈4 × 5/4⌉ = 5, ⌈10 × 5/4⌉ = 13 (12.5 up), 7 exactly, ⌈1,000 × 5/4⌉ = 1,250.
        let want = 5 + 13 + 7 + 1_250;
        assert_eq!(inner.holds, vec![want]);
        assert_eq!(
            held_bytes,
            Some(want),
            "the allowance is the scaled rewrite held, the need's own rewrite ignored"
        );
        let adoption_only = RewriteCount {
            adoption_entries: 3,
            ..RewriteCount::default()
        };
        let mut inner = CountingCharge::default();
        let held_bytes = {
            let cell = CommitChargeCell::new(&mut inner);
            let mut budget = ChargeRewriteBudget::new(&cell, ObjectStateNeed::watch_fixed());
            budget.extend(adoption_only)
        };
        assert_eq!(inner.holds, vec![4], "⌈3 × 5/4⌉ = 4 adoption ids");
        assert_eq!(held_bytes, Some(4));
    }

    /// The charge a disabled audit opens (`StructuralCacheAudit::disabled`)
    /// refuses every growth: every need's target is above its cap of 0, so a
    /// repair holding it is refused at its first charge point, and it holds
    /// and has held nothing. Holding nothing (a target of 0) is all it
    /// grants.
    #[test]
    fn refusing_audit_charges_refuse_every_growth_and_hold_nothing() {
        let charge = RefusingAuditCharges.open(1_000);
        let need = ObjectStateNeed::cached_repair(
            0,
            RepairTransient {
                reply_entry_count: 1,
                ..RepairTransient::default()
            },
        );
        let target = charge.target(&need);
        assert!(target > charge.cap(), "its targets are above its cap");
        assert_eq!(charge.try_hold_at_least_need(&need), Err(target));
        for target in [1, u64::MAX] {
            assert!(!charge.try_hold(target), "try_hold({target})");
            assert!(
                !charge.try_hold_at_least(target),
                "try_hold_at_least({target})"
            );
        }
        assert!(charge.try_hold(0) && charge.try_hold_at_least(0));
        assert_eq!((charge.held(), charge.peak(), charge.cap()), (0, 0, 0));
        assert_eq!(charge.rewrite_allowance(&need).bytes, 0);
    }

    /// The test-support charge every audit test without a budget uses grants
    /// every growth at a target of 0, as `NoCommitCharge` does: nothing is
    /// held on any budget and its cap is unbounded.
    #[test]
    fn unbounded_audit_charges_grant_everything_at_no_target() {
        let charge = UnboundedAuditCharges.open(1_000);
        let need = ObjectStateNeed::cached_repair(
            u64::MAX,
            RepairTransient {
                reply_entry_count: u64::MAX,
                ..RepairTransient::default()
            },
        );
        assert_eq!(charge.target(&need), 0);
        assert_eq!(charge.try_hold_at_least_need(&need), Ok(0));
        assert!(charge.try_hold(u64::MAX));
        assert_eq!((charge.cap(), charge.peak()), (u64::MAX, 0));
        assert_eq!(charge.rewrite_allowance(&need).bytes, u64::MAX);
    }
}
