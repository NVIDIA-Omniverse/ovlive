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

//! The structural cache's read side: the one entry point a structural
//! `UpdateLayer` attempt calls to get an entry current at the header's
//! `(seq, commit_chain)` to plan on, or to learn that it must not plan at all
//! (structural-edits SDD SR-4.1–SR-4.3, SR-3.4, SR-2.4; the plan's breakdown
//! decision E6).
//!
//! Section references: `SR-n.m` is the structural-edits spec SDD (Spec Bee
//! `b.xes`); `§n.m` is `docs-internal/sdd.md`.
//!
//! # Who calls it
//!
//! Every attempt of a structural or reset `UpdateLayer` call under the `lua`
//! backend calls [`read_for_structural_edit`] first, once per attempt
//! (`LayerStore::update_layer_once_structural`), then plans and commits on
//! the entry it answers and settles it by the commit's outcome. A re-attempt
//! after a snapshot-fence wait calls it again within the same attempt.
//!
//! # The sequence
//!
//! 1. **Lookup** ([`StructuralCache::lookup`]). An object with a live
//!    uncacheable verdict, or a disabled cache, answers
//!    [`StructuralRead::Uncacheable`] with no read: the uncacheable path's own
//!    commit-token check covers it. A verdict no one has reported yet is
//!    reported here, once ("Metrics" below).
//! 2. **Hit: repair** the checked-out entry from the stream
//!    (`storage::structural_cache_repair`), which also maps the header and
//!    checks the call's commit token on every read, and, with a commit
//!    charge, holds each read, each chunk's replay and each record's apply
//!    in it first (RP0, RP1, RP2). A repair anomaly drops the entry and goes
//!    to step 3 as a reload; a refused charge point (RP0, RP1 or RP2) puts
//!    it back at the cut the repair reached — between rounds for RP0 and
//!    RP1, between two records of a round for RP2, a valid, normalized cut
//!    either way — and answers [`StructuralRead::NeedsCharge`].
//! 3. **Miss or reload: probe, then cold start.** The repair script in probe
//!    mode answers the header mapping, the commit token, `superseded` (then a
//!    chained call's predecessor, from the same token record) and
//!    `chs_complete`. An object without `chs_complete` gets an
//!    [`UncacheableReason::NoChsComplete`] negative entry with no full read,
//!    counted under `live_edit_structural_cold_start_reads_total{route="uncacheable"}`,
//!    and answers [`StructuralRead::Uncacheable`] with the probe's size when
//!    its item bound is over the plain item ceiling, as a probed cold start
//!    would (`structural_cache_cold_start::probed_above_ceiling`), so its
//!    WATCH fallback registers its writer intent first (P12). A reset
//!    ([`StructuralReadRequest::with_reset_content_rows`]) whose work,
//!    estimated from the probe's cardinalities, is over the structural
//!    commit's work limit answers [`StructuralRead::OverWorkLimit`] here,
//!    with no cold start and nothing recorded
//!    (`storage::structural_size_gate`). Otherwise
//!    [`cold_start_probed`](crate::storage::structural_cache_cold_start::cold_start_probed)'s
//!    route reads the object, chosen on the probe's index cardinalities (that
//!    module's "The route"; when an index key holds a foreign type the probe
//!    reports none, and the unprobed cold start runs, whose read refuses that
//!    index); its answers map as:
//!    - `Loaded` → the token record it read (a chained call's predecessor
//!      included), then a repair to the current header (normally zero
//!      entries);
//!    - `Invalidated` (the object was retired or audit-dropped while it read)
//!      or `NotFound` (the atomic read cannot tell an absent header from one
//!      without a numeric `seq`) → probe again, which classifies it;
//!    - `Superseded` → the token record it read, then
//!      `FailedPrecondition(SUPERSEDED_SENTINEL)` (never a pending
//!      predecessor: superseded comes first);
//!    - `Uncacheable` → [`StructuralRead::Uncacheable`], with the layer's
//!      size when the cold start knows it is above the plain ceilings — a
//!      layer too large for its fenced read to fit the configured RPC
//!      deadline ([`UncacheableReason::OverDeadline`], recorded for the
//!      object) and a fenced read that failed for this call
//!      ([`UncacheableReason::FencedReadFailed`]) among them;
//!    - `CommitTokenRecordWrongType` (the record key changed type between
//!      the probe and the cold start's `HGET`) →
//!      [`StructuralRead::CommitTokenRecordWrongType`];
//!    - `NeedsCharge` (with a commit charge only: the cold start could not
//!      hold its decoded entry in the call's charge without waiting, stored
//!      the entry and dropped its permits) → [`StructuralRead::NeedsCharge`]
//!      under `structural_full`.
//!
//!    The repair after a cold start charges as a hit's does; a refused
//!    charge point there puts the entry back and answers
//!    [`StructuralRead::NeedsCharge`] under `structural_full` too. A repair
//!    anomaly after a cold start reloads again, at most
//!    [`STRUCTURAL_READ_MAX_RELOADS_AFTER_COLD_START`] times per call; past
//!    that the call answers [`StructuralRead::Uncacheable`] with
//!    [`UncacheableReason::ReloadLimit`], recording no negative entry (the
//!    verdict is still reported, "Metrics" below). The reload's
//!    cold start takes its ticket after the anomaly's `discard`, so that
//!    discard never refuses the reload's admission.
//!
//! # Answers
//!
//! - [`StructuralRead::Ready`]: the entry, checked out and current, with the
//!   path hint `structural` (a hit, repaired or not) or `structural_full` (a
//!   cold start or reload). The caller plans in place on it
//!   ([`CheckedOut::plan_in_place`]) and settles the guard by outcome
//!   (SR-3.4): on `Committed`, [`CheckedOut::advance_committed`] and then
//!   [`CheckedOut::put_back_held`] — or [`CheckedOut::discard`] when the
//!   plan adopts a dangling parent, which `advance_committed` answers as
//!   `Err(ReloadReason::DanglingAdopted)` and
//!   [`plan_adopts_dangling_parent`](crate::storage::structural_cache_repair::plan_adopts_dangling_parent)
//!   tests on its own — and otherwise [`CheckedOut::put_back_held`], which
//!   rolls the plan back first, or `discard`.
//! - [`StructuralRead::AlreadyLanded`]: this call's commit token is recorded;
//!   the call reports that commit and plans nothing (SR-4.2a), on every route.
//!   Any entry held is put back.
//! - [`StructuralRead::Recorded`] / [`StructuralRead::Overtaken`] /
//!   [`StructuralRead::PredecessorPending`]: with a client operation
//!   ([`StructuralReadRequest::with_operation`]) only — an earlier call's
//!   commit of the same `operation_id`; for an unchained call, the writer's
//!   record moved off its start record; for a chained call, a record that
//!   does not name its predecessor, checked after the superseded check (on a
//!   hit or repair, after the probe, and on a cold start's loaded entry; a
//!   superseded object answers `SUPERSEDED` first). The call plans nothing
//!   and answers them as the fast and WATCH paths do (§7.5.3). Any entry held
//!   is put back.
//! - [`StructuralRead::Uncacheable`]: take the uncacheable path (SR-7.4) —
//!   for the object while its negative entry lives, or for this call alone
//!   when the reason is [`UncacheableReason::ReloadLimit`] or
//!   [`UncacheableReason::FencedReadFailed`].
//! - [`StructuralRead::OverWorkLimit`]: a reset too large for one structural
//!   commit, found before its cold start; take the WATCH path for this call.
//!   No entry was read or recorded.
//! - [`StructuralRead::CommitTokenRecordWrongType`]: the commit backends'
//!   byte-identical `WRONGTYPE_KEY` refusal, for the caller to count, log and
//!   return: the structural route counts it on the update script's
//!   structural-mode series, `inc_update_script_keyspace_type_error` with
//!   `mode="structural"` (`keyspace_type_error_total{backend="lua",
//!   site="update", mode="structural"}`), as the fast path's own `obj:wtok`
//!   refusal ticks `mode="fast"`. Any entry held is put back.
//! - [`StructuralRead::NeedsCharge`]: with a commit charge only
//!   ([`StructuralReadRequest::with_commit_charge`]) — the cold start stored
//!   its entry uncharged (CR0), or, from a plain read whose hold was over
//!   its allowance, loaded nothing; or a repair's charge point (RP0 before a
//!   read, RP1 before replaying its chunk, RP2 before applying one of its
//!   records) was refused — RP0 and RP1 between rounds, RP2 between two
//!   records of a round — and the entry was put back at the cut it reached,
//!   a valid, normalized cut. The caller waits for the charge,
//!   holding nothing, under the answer's path (`structural` on a hit,
//!   `structural_full` after a cold start), and reads again: a hit whose
//!   repair continues from that cut, or a cold start under the charge.
//! - `Err`: the header mapping — `NotFound(EMPTY_HEADER_SENTINEL)`,
//!   `FailedPrecondition(PARTIAL_HEADER_SENTINEL)` or
//!   `FailedPrecondition(SUPERSEDED_SENTINEL)`, any entry dropped — or a
//!   store error (an entry held is put back by its guard: it is a valid cut).
//!
//! # Bounded by the call's deadline
//!
//! The entry point has no timer of its own. It runs inside `update_layer`,
//! under the RPC deadline, which drops the future on expiry: an outstanding
//! cold-start ticket is then released (it retains every invalidation record
//! newer than its epoch while it lives), any held entry goes back by its
//! guard, and the re-probe loop ends with it. Each of the cold start's
//! whole-layer reads also waits for and holds one of the cache's per-replica
//! plain permits, and a fenced read also holds the one-at-a-time cold-start
//! permit, taken first (`storage::structural_cache_cold_start`;
//! `storage::structural_cache`, "Cold-start permits"), so however
//! many calls loop through probe, cold start and repair at once, at most
//! [`STRUCTURAL_COLD_START_READ_PERMITS`](crate::storage::structural_cache::STRUCTURAL_COLD_START_READ_PERMITS)
//! of those reads run on the replica together; the probe takes none, and
//! the deadline ends a permit wait too. A fenced read also caps its own
//! waiting against what is left of that deadline when it begins
//! ([`StructuralReadRequest::with_remaining_deadline`]), so that a read that
//! cannot finish in time gives up early enough for the call's WATCH attempt
//! (`storage::structural_cache_fenced`). A layer whose fenced read could not
//! finish in time even with the whole configured deadline
//! ([`StructuralReadRequest::with_rpc_deadline`]) is refused before that read
//! and recorded [`UncacheableReason::OverDeadline`], so its later edits pay
//! no probe until the negative entry expires. The reload loop after a failed
//! cold start is also bounded by count
//! ([`STRUCTURAL_READ_MAX_RELOADS_AFTER_COLD_START`]), so a cause that
//! recurs — an older replica's non-folding commits during a rolling deploy —
//! cannot keep it re-reading the object until the deadline.
//!
//! # Metrics (SR-2.4)
//!
//! - `live_edit_structural_cache_lookups_total{outcome}` once per call of
//!   this entry point that consulted the cache — so once per structural
//!   attempt, and again for a retry after `STALE` or `CACHE_DIVERGED` and for
//!   a re-attempt after a fence wait (decider ruling, Q3: the metric counts
//!   attempts): `cold` on a miss, before the probe; on a hit, once the repair
//!   ends, `reload` for an anomaly, else `repaired` if it replayed an entry,
//!   else `hit`. A hit whose repair failed on a store error, and an
//!   uncacheable lookup, tick nothing.
//! - `live_edit_structural_cache_reloads_total{reason}` for each repair
//!   anomaly, after a hit or after a cold start, and `reason="uncacheable"`
//!   for each explicit put-back the per-entry cap refused ([`count_put_back`],
//!   the read side's and the structural route's alike). A header-mapped drop
//!   (gone, superseded, partial) is not a reload.
//! - `live_edit_structural_cache_replayed_entries` and
//!   `live_edit_structural_cache_repair_bytes` once per repair that ran to an
//!   end, whatever the end.
//! - The cold start ticks its own metrics; the probe's `NoChsComplete`
//!   verdict ticks `cold_start_reads_total{route="uncacheable"}` through it.
//! - `live_edit_structural_cache_uncacheable_total{reason}` and its INFO
//!   line once per verdict this call reaches
//!   (`structural_cache_cold_start::report_uncacheable`, which names the
//!   request's layer when [`StructuralReadRequest::with_layer`] set it): the
//!   probe's `no_chs_complete`, the cold start's verdicts, `reload_limit`, and
//!   each explicit put-back the per-entry cap refused (`over_weight`). An
//!   uncacheable lookup reports only a verdict no one has reported yet —
//!   one the implicit put-back of a dropped call recorded — once
//!   ([`StructuralCache::take_unreported`]); a verdict already reported, and
//!   a disabled cache, report nothing. `reload_limit` and
//!   `fenced_read_failed` log at DEBUG (verdicts about one call, never
//!   recorded, so their rate is unbounded while they recur); every other
//!   reason, `over_deadline` among them, at INFO.

use std::time::{Duration, Instant};

use crate::error::ServiceError;
use crate::observability::metrics::{
    SharedMetrics, STRUCTURAL_CACHE_LOOKUP_COLD, STRUCTURAL_CACHE_LOOKUP_HIT,
    STRUCTURAL_CACHE_LOOKUP_RELOAD, STRUCTURAL_CACHE_LOOKUP_REPAIRED,
};
use crate::storage::client::ValkeyClient;
use crate::storage::commit_charge::SharedCommitCharge;
use crate::storage::diff_read::{SnapshotCardinalities, SnapshotCeilings};
use crate::storage::keys::KeySchema;
use crate::storage::layer::{
    empty_header_not_found, partial_header_failed_precondition, superseded_failed_precondition,
    Operation, RecordedBy, RecordedCommit,
};
use crate::storage::snapshot_fence::writer::RemainingDeadlineFn;
use crate::storage::snapshot_fence::LayerSize;
use crate::storage::structural_cache::{
    CacheLookup, CheckedOut, PutBack, StructuralCache, UncacheableReason,
};
use crate::storage::structural_cache_cold_start::{
    cold_start_for, probed_above_ceiling, record_verdict, report_uncacheable, ColdStart,
    ColdStartCharge, Verdict, VerdictLayer,
};
#[cfg(any(test, feature = "test-support"))]
use crate::storage::structural_cache_repair::TestHook;
use crate::storage::structural_cache_repair::{
    check_commit_token, check_commit_token_record, check_predecessor, repair, Goal, ReloadReason,
    RepairBounds, RepairEnd, RepairHooks, RepairStats, TokenAnswer, TokenVerdict,
};
use crate::storage::structural_cache_sync::{
    read_structural_cache_sync, StructuralSyncRequest, SyncChunk, SyncHeader,
};
use crate::storage::structural_size_gate::{reset_work_lower_bound, STRUCTURAL_COMMIT_WORK_LIMIT};

/// A test hook run after every probe that lets a cold start proceed, before
/// the cold start's first read: the one point at which a test can change the
/// object between the two (see
/// [`StructuralReadRequest::with_after_probe_hook_for_test`]).
#[cfg(any(test, feature = "test-support"))]
pub type AfterProbeHook = TestHook;

/// A test hook run between two repair reads of one repair, after the
/// earlier read's chunk was replayed (see
/// [`StructuralReadRequest::with_between_repair_reads_hook_for_test`]).
#[cfg(any(test, feature = "test-support"))]
pub type BetweenRepairReadsHook = TestHook;

/// A test hook run after every cold start that loaded an entry and found the
/// call's commit token not landed, before the repair that follows it (see
/// [`StructuralReadRequest::with_after_cold_start_hook_for_test`]).
#[cfg(any(test, feature = "test-support"))]
pub type AfterColdStartHook = TestHook;

/// How many times one call reloads an object after one of its own cold
/// starts failed its repair, before it gives up on the cache for that call
/// and answers [`StructuralRead::Uncacheable`] with
/// [`UncacheableReason::ReloadLimit`]: 1.
///
/// **Why a cap.** A cold start's entry is repaired to the current header at
/// once, and that repair normally replays nothing. It fails only when the
/// object changed in the moment between the cold start's read and the repair
/// in a way that reloads — and one way recurs: during a rolling deploy an
/// older replica's commit does not fold `commit_chain`, so every such commit
/// landing in that moment fails the repair with `regressed`. Under a busy
/// older writer each reload's cold start can meet a new one, and without a
/// cap the call would read the whole object again and again until its RPC
/// deadline, each read blocking the object's shard. One reload covers the
/// ordinary race; a second failure means the object is changing faster than
/// a cold start can catch it, and the uncacheable path commits without the
/// cache.
///
/// **One call only.** The verdict is not recorded: no negative entry, and
/// the failed entry was discarded as every reload discards it, so the next
/// call on the object looks it up and cold-starts afresh. Reloads that follow
/// a cache hit's failed repair, and re-probes after a cold start that found
/// the object invalidated or unclassifiable, do not count: neither is a
/// failed cold start.
pub const STRUCTURAL_READ_MAX_RELOADS_AFTER_COLD_START: u32 = 1;

/// One structural call's read-side request.
#[derive(Clone)]
pub struct StructuralReadRequest<'a> {
    object_id: u64,
    client_id: &'a str,
    commit_token: &'a str,
    operation: Option<&'a Operation>,
    layer: Option<VerdictLayer<'a>>,
    repair_bounds: RepairBounds,
    rpc_deadline: Option<Duration>,
    remaining_deadline: Option<RemainingDeadlineFn>,
    reset_content_rows: Option<u64>,
    commit_charge: Option<&'a dyn SharedCommitCharge>,
    #[cfg(any(test, feature = "test-support"))]
    work_limit_for_test: Option<u64>,
    #[cfg(any(test, feature = "test-support"))]
    cold_start_hold_allowance_for_test: Option<u64>,
    #[cfg(any(test, feature = "test-support"))]
    after_probe_hook: Option<AfterProbeHook>,
    #[cfg(any(test, feature = "test-support"))]
    between_repair_reads_hook: Option<BetweenRepairReadsHook>,
    #[cfg(any(test, feature = "test-support"))]
    after_cold_start_hook: Option<AfterColdStartHook>,
}

impl<'a> StructuralReadRequest<'a> {
    /// A request for `object_id` by the call whose `client_id` and commit
    /// token are given, under the production repair bounds.
    pub fn new(object_id: u64, client_id: &'a str, commit_token: &'a str) -> Self {
        Self {
            object_id,
            client_id,
            commit_token,
            operation: None,
            layer: None,
            repair_bounds: RepairBounds::PRODUCTION,
            rpc_deadline: None,
            remaining_deadline: None,
            reset_content_rows: None,
            commit_charge: None,
            #[cfg(any(test, feature = "test-support"))]
            work_limit_for_test: None,
            #[cfg(any(test, feature = "test-support"))]
            cold_start_hold_allowance_for_test: None,
            #[cfg(any(test, feature = "test-support"))]
            after_probe_hook: None,
            #[cfg(any(test, feature = "test-support"))]
            between_repair_reads_hook: None,
            #[cfg(any(test, feature = "test-support"))]
            after_cold_start_hook: None,
        }
    }

    /// The same request for a call carrying a client `operation_id`:
    /// `operation` (with the call's writer guard already set, once per call)
    /// makes the commit-token check match on `op.<operation_id>` and answer
    /// the record's verdict — this call's own commit, an earlier call's, or a
    /// conflict — and, when no record matches, check the writer guard: an
    /// unchained call's start record before the superseded check, a chained
    /// call's predecessor after it. The same verdicts, in the same order, as
    /// the commit script's step 1b, 1c and PREDECESSOR step.
    pub fn with_operation(mut self, operation: &'a Operation) -> Self {
        self.operation = Some(operation);
        self
    }

    /// The same request for a call on `session_id`'s layer `layer_name`,
    /// which the verdict line of every uncacheable verdict the call reaches
    /// (INFO; DEBUG for `reload_limit` and `fenced_read_failed`) names beside
    /// the object (`structural_cache_cold_start::report_uncacheable`).
    /// Without it the line names the object only. The structural
    /// `UpdateLayer` attempt always sets it.
    pub fn with_layer(mut self, session_id: &'a str, layer_name: &'a str) -> Self {
        self.layer = Some(VerdictLayer {
            session_id,
            layer_name,
        });
        self
    }

    /// The same request with the calling RPC's deadline: `remaining` reads
    /// how much of it is left. A fenced cold start reads it once, when its
    /// fenced read begins, and caps its waiting so time is left for the read
    /// and one WATCH attempt (`storage::structural_cache_fenced`). Without it
    /// that cap is the cost model's `reader_wait_max(L)` alone. The structural
    /// `UpdateLayer` attempt sets it whenever its store has the deadline
    /// wired.
    pub fn with_remaining_deadline(mut self, remaining: RemainingDeadlineFn) -> Self {
        self.remaining_deadline = Some(remaining);
        self
    }

    /// The same request on a replica whose configured RPC deadline
    /// (`LIVE_EDIT_RPC_DEADLINE_MS`) is `rpc_deadline`. A cold start bound
    /// for the fenced read first checks the layer's size against it: when
    /// the fenced read could not fit even a whole deadline, the object is
    /// recorded [`UncacheableReason::OverDeadline`] with no fenced read
    /// (`storage::structural_cache_cold_start`, "The route"). Without it that
    /// check is skipped. The structural `UpdateLayer` attempt sets it, with
    /// [`Self::with_remaining_deadline`], whenever its store has the
    /// deadline wired.
    pub fn with_rpc_deadline(mut self, rpc_deadline: Duration) -> Self {
        self.rpc_deadline = Some(rpc_deadline);
        self
    }

    /// The same request for a reset whose delta carries `content_rows` field
    /// and time-sample rows. Before a cold start — the object has no cache
    /// entry, or its entry was dropped for a reload — the read side then
    /// estimates the reset's work from the probe's index cardinalities and
    /// these rows
    /// ([`reset_work_lower_bound`](crate::storage::structural_size_gate::reset_work_lower_bound)),
    /// and answers [`StructuralRead::OverWorkLimit`] without the cold start's
    /// full read when it is over the structural commit's work limit. A
    /// cached entry is never gated here: the structural route gates its plan.
    /// The structural `UpdateLayer` attempt sets it for every reset.
    pub fn with_reset_content_rows(mut self, content_rows: u64) -> Self {
        self.reset_content_rows = Some(content_rows);
        self
    }

    /// The same request for a call whose memory is charged to `charge`, the
    /// structural `UpdateLayer`'s upload commit charge (SDD §5.4). A cold
    /// start then holds its decoded entry in it at its decode, before its
    /// read permits drop (`storage::structural_cache_cold_start`, seam A,
    /// CR0), and every repair holds each read before issuing it, each
    /// chunk's replay before replaying it and each record before applying it
    /// (`storage::structural_cache_repair`, RP0, RP1, RP2) — never waiting.
    /// Refused, the cold start answers [`StructuralRead::NeedsCharge`],
    /// having stored its entry unless a plain read's hold is over its
    /// allowance (`plain_cold_start_hold_allowance_bytes`), or the repair
    /// puts the entry back at the cut it reached (between rounds for RP0 and
    /// RP1, between two records of a round for RP2; a valid, normalized cut
    /// either way) and the read answers the same. Without it
    /// nothing is charged, as for every caller but the structural route.
    pub fn with_commit_charge(mut self, charge: &'a dyn SharedCommitCharge) -> Self {
        self.commit_charge = Some(charge);
        self
    }

    /// **Test-only.** Judge a refused plain cold start's hold
    /// ([`Self::with_commit_charge`]) against `allowance` bytes instead of
    /// `plain_cold_start_hold_allowance_bytes` of the plain byte ceiling
    /// (`storage::structural_cache_cold_start`, seam A). The structural
    /// `UpdateLayer` attempt passes its store's override
    /// (`LayerStore::with_plain_cold_start_hold_allowance_for_test`).
    #[cfg(any(test, feature = "test-support"))]
    pub fn with_plain_cold_start_hold_allowance_for_test(mut self, allowance: u64) -> Self {
        self.cold_start_hold_allowance_for_test = Some(allowance);
        self
    }

    /// **Test-only.** Gate a reset ([`Self::with_reset_content_rows`]) against
    /// `limit` work units instead of [`STRUCTURAL_COMMIT_WORK_LIMIT`]. The
    /// structural `UpdateLayer` attempt passes its store's limit
    /// (`LayerStore::with_structural_commit_work_limit_for_test`), so one
    /// override reaches both size gates.
    #[cfg(any(test, feature = "test-support"))]
    pub fn with_work_limit_for_test(mut self, limit: u64) -> Self {
        self.work_limit_for_test = Some(limit);
        self
    }

    /// The commit charge a cold start grows at seam A
    /// ([`Self::with_commit_charge`]), with a test's hold allowance
    /// ([`Self::with_plain_cold_start_hold_allowance_for_test`]).
    fn cold_start_charge(&self) -> Option<ColdStartCharge<'a>> {
        let charge = ColdStartCharge::new(self.commit_charge?);
        #[cfg(any(test, feature = "test-support"))]
        let charge = charge.with_hold_allowance_for_test(self.cold_start_hold_allowance_for_test);
        Some(charge)
    }

    /// The work limit the pre-read reset gate compares with:
    /// [`STRUCTURAL_COMMIT_WORK_LIMIT`], or a test's override
    /// ([`Self::with_work_limit_for_test`]).
    fn work_limit(&self) -> u64 {
        #[cfg(any(test, feature = "test-support"))]
        if let Some(limit) = self.work_limit_for_test {
            return limit;
        }
        STRUCTURAL_COMMIT_WORK_LIMIT
    }

    /// The same request under other repair bounds. Production keeps
    /// [`RepairBounds::PRODUCTION`]; a test narrows a bound to reach it.
    pub fn with_repair_bounds(mut self, bounds: RepairBounds) -> Self {
        self.repair_bounds = bounds;
        self
    }

    /// **Test-only.** Run `hook` after each probe that lets a cold start
    /// proceed, before the cold start reads — so a test can make the header
    /// partial or superseded between the two and observe the cold start's
    /// answer re-classified.
    #[cfg(any(test, feature = "test-support"))]
    pub fn with_after_probe_hook_for_test(mut self, hook: AfterProbeHook) -> Self {
        self.after_probe_hook = Some(hook);
        self
    }

    /// **Test-only.** Run `hook` between two reads of every repair this call
    /// runs — before each repair read after the first, once the earlier
    /// read's chunk is replayed — so a test can land a commit, this call's own
    /// included, that only a later read sees, and observe the per-read
    /// commit-token check answer it. A repair that finishes in one read never
    /// runs it; narrow [`Self::with_repair_bounds`] to force more reads.
    #[cfg(any(test, feature = "test-support"))]
    pub fn with_between_repair_reads_hook_for_test(mut self, hook: BetweenRepairReadsHook) -> Self {
        self.between_repair_reads_hook = Some(hook);
        self
    }

    /// **Test-only.** Run `hook` after each cold start that loaded an entry
    /// and found this call's commit token not landed, before that entry's
    /// repair — so a test can land a commit the cold start's read did not
    /// see and the repair must cross: a commit that does not fold
    /// `commit_chain` (`storage::test_support::commit_without_fold`) makes
    /// that repair reload `regressed`, and doing so on every cold start
    /// drives the call to [`STRUCTURAL_READ_MAX_RELOADS_AFTER_COLD_START`].
    #[cfg(any(test, feature = "test-support"))]
    pub fn with_after_cold_start_hook_for_test(mut self, hook: AfterColdStartHook) -> Self {
        self.after_cold_start_hook = Some(hook);
        self
    }

    /// The repair's test seams for this request; empty in production.
    fn repair_hooks(&self) -> RepairHooks {
        RepairHooks {
            #[cfg(any(test, feature = "test-support"))]
            between_reads: self.between_repair_reads_hook.clone(),
            // A cached entry always holds its plan indexes, so `repair`
            // builds no copy bound on this route and the check has nothing
            // to assert.
            #[cfg(any(test, feature = "test-support"))]
            copy_bound_check: false,
        }
    }

    /// The object read.
    pub fn object_id(&self) -> u64 {
        self.object_id
    }

    /// The bounds this request's repairs run under
    /// ([`Self::with_repair_bounds`]).
    pub fn repair_bounds(&self) -> RepairBounds {
        self.repair_bounds
    }
}

impl std::fmt::Debug for StructuralReadRequest<'_> {
    /// The commit token is not printed: it stays out of logs above DEBUG.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StructuralReadRequest")
            .field("object_id", &self.object_id)
            .field("client_id", &self.client_id)
            .field("layer", &self.layer)
            .field("repair_bounds", &self.repair_bounds)
            .field("rpc_deadline", &self.rpc_deadline)
            .field("remaining_deadline", &self.remaining_deadline.is_some())
            .field("reset_content_rows", &self.reset_content_rows)
            .field("commit_charge", &self.commit_charge.is_some())
            .finish_non_exhaustive()
    }
}

/// Which read the entry an attempt plans on came from: the commit path label
/// the attempt commits under (SR-2.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructuralPath {
    /// A cache hit, repaired or not: `path="structural"`.
    Structural,
    /// A cold start or reload read: `path="structural_full"`.
    StructuralFull,
}

impl StructuralPath {
    /// The `path` label value.
    pub fn label(self) -> &'static str {
        match self {
            Self::Structural => "structural",
            Self::StructuralFull => "structural_full",
        }
    }
}

/// An entry ready to plan on.
#[derive(Debug)]
pub struct ReadyEntry {
    /// The checked-out entry, current at the header's `(seq, commit_chain)`
    /// as of the last repair read, and normalized. Its `seq` and
    /// `commit_chain` are the structural commit's fence (SR-6.4).
    pub entry: CheckedOut,
    /// Which read it came from.
    pub path: StructuralPath,
    /// What the last repair read and replayed.
    pub repair: RepairStats,
}

/// What the read side answers a structural call (module documentation,
/// "Answers").
//
// `clippy::large_enum_variant` allowed: `Ready` carries the checkout guard by
// value, like `CacheLookup::Hit`. The value is built and matched once per
// call and never stored.
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub enum StructuralRead {
    /// Plan on this entry.
    Ready(ReadyEntry),
    /// This call's commit landed at `seq`, stamped `server_ts`: report it and
    /// plan nothing (SR-4.2a).
    AlreadyLanded { seq: u64, server_ts: u64 },
    /// With a client operation only: the record names an earlier call that
    /// carried the same `operation_id` (`RecordedBy::EarlierCall` →
    /// `ALREADY_COMMITTED`, `RecordedBy::Conflict` → `ALREADY_EXISTS`; never
    /// `ThisCall`, which is [`Self::AlreadyLanded`]). Plan nothing; the
    /// caller's shared verdict handling answers it and releases the call's
    /// pre-flight external-value increments.
    Recorded(RecordedCommit),
    /// With an unchained client operation only: the writer's record has moved
    /// off the call's start record. Plan nothing; answered `UNAVAILABLE`
    /// (`OPERATION_OVERTAKEN_MESSAGE`). The caller's shared verdict handling
    /// hands back the call's pre-flight increments on its first attempt, or
    /// looks up its own landed commit's releases after a counted retry.
    Overtaken,
    /// With a chained client operation only, after the superseded check: the
    /// writer's record does not name the call's predecessor (for a chain's
    /// first write: a record exists). `record` is that record as read, `None`
    /// when absent. Plan nothing; answered `PREDECESSOR_NOT_COMMITTED` (a
    /// plain `UNAVAILABLE` after a counted retry), with the same release
    /// handling as [`Self::Overtaken`].
    PredecessorPending { record: Option<String> },
    /// The object is uncacheable — or, with [`UncacheableReason::ReloadLimit`]
    /// or [`UncacheableReason::FencedReadFailed`], this call gave up on the
    /// cache — so take the uncacheable path (SR-7.4).
    Uncacheable {
        reason: UncacheableReason,
        /// The layer's size when this call's cold start knows the layer is
        /// above the plain atomic-read ceilings
        /// (`storage::structural_cache_cold_start`, "What the read side
        /// learns of the size"); `None` otherwise, a lookup that hit a
        /// negative entry included. The structural route classifies its WATCH
        /// fallback from it, so the fallback's first attempt registers its
        /// writer intent before `WATCH` (policy decision P12).
        above_ceiling: Option<LayerSize>,
    },
    /// A reset ([`StructuralReadRequest::with_reset_content_rows`]) of an
    /// object with no cache entry whose estimated work is over the structural
    /// commit's work limit: take the WATCH path for this call, with no cold
    /// start. Nothing is recorded: the object stays cacheable, and the next
    /// call looks it up afresh.
    OverWorkLimit {
        /// The reset's estimated work, in units
        /// ([`reset_work_lower_bound`](crate::storage::structural_size_gate::reset_work_lower_bound)).
        estimate: u64,
        /// The probe's index cardinalities the estimate was taken from.
        cardinalities: SnapshotCardinalities,
        /// The probe's size when its item bound is over the plain item
        /// ceiling, as [`Self::Uncacheable`] carries it, so the WATCH
        /// fallback registers its writer intent before its first `WATCH`
        /// (policy decision P12); `None` otherwise.
        above_ceiling: Option<LayerSize>,
    },
    /// The commit-token record key holds a foreign type. `refusal` is the
    /// `Internal` refusal both commit backends return, byte for byte; the
    /// caller counts it on the update script's structural-mode series
    /// (`inc_update_script_keyspace_type_error` with `mode="structural"`:
    /// `keyspace_type_error_total{backend="lua", site="update",
    /// mode="structural"}`), logs it, and returns it.
    CommitTokenRecordWrongType { refusal: ServiceError },
    /// With a commit charge only ([`StructuralReadRequest::with_commit_charge`]):
    /// a cold start's decoded entry (CR0), or a repair's next read, chunk
    /// replay or record apply (RP0, RP1, RP2), could not be held in the
    /// charge without waiting. The entry was stored — a cold start's after
    /// dropping its read permits and grant, unless a plain read's state and
    /// build were over its allowance, when nothing was loaded; a repaired one
    /// at the last cut its repair reached, between rounds (RP0, RP1) or
    /// between two records of a round (RP2), a valid, normalized cut —
    /// and the caller waits for `target` object-state bytes, holding nothing;
    /// its next read is a hit, whose repair continues from that cut, or a
    /// cold start under the held charge (SDD §5.4). `path` is the read's:
    /// [`StructuralPath::Structural`] on a hit,
    /// [`StructuralPath::StructuralFull`] after a cold start.
    NeedsCharge { target: u64, path: StructuralPath },
}

/// Get `request`'s object ready for one structural attempt (module
/// documentation): an entry current at the header, or an answer that plans
/// nothing.
///
/// # Errors
///
/// - `NotFound(EMPTY_HEADER_SENTINEL)` for an absent header,
///   `FailedPrecondition(PARTIAL_HEADER_SENTINEL)` for a header without a
///   numeric `seq`, `FailedPrecondition(SUPERSEDED_SENTINEL)` for a
///   superseded one (after the token check); any entry is dropped.
/// - A read's classified store error (transient → `Aborted`, permanent →
///   `Internal`), `Internal` for a commit-token record with this call's token
///   and an unparseable tail, or `Internal` for a reply that contradicts
///   itself.
pub async fn read_for_structural_edit(
    client: &ValkeyClient,
    keys: &KeySchema,
    cache: &StructuralCache,
    request: &StructuralReadRequest<'_>,
) -> Result<StructuralRead, ServiceError> {
    let metrics = client.metrics_snapshot();
    let metrics = metrics.as_ref();
    match cache.lookup(request.object_id) {
        CacheLookup::Uncacheable { reason } => {
            // A verdict no one reported yet — recorded by the implicit
            // put-back of a dropped call — is reported by this first lookup
            // it routes, once; a reported one (and a disabled cache) gives
            // nothing.
            if let Some(unreported) = cache.take_unreported(request.object_id) {
                report_uncacheable(
                    metrics,
                    cache,
                    &Verdict::new(request.object_id, request.layer, unreported),
                );
            }
            return Ok(StructuralRead::Uncacheable {
                reason,
                above_ceiling: None,
            });
        }
        CacheLookup::Miss => tick_lookup(metrics, STRUCTURAL_CACHE_LOOKUP_COLD),
        CacheLookup::Hit(mut entry) => {
            let (end, stats) = repair(
                client,
                keys,
                &mut entry,
                header_goal(request),
                request.repair_bounds,
                request.commit_charge,
                None,
                &request.repair_hooks(),
            )
            .await?;
            observe_repair(metrics, &stats);
            tick_lookup(
                metrics,
                match end {
                    RepairEnd::Reload(_) => STRUCTURAL_CACHE_LOOKUP_RELOAD,
                    _ if stats.replayed_entries > 0 => STRUCTURAL_CACHE_LOOKUP_REPAIRED,
                    _ => STRUCTURAL_CACHE_LOOKUP_HIT,
                },
            );
            if let Some(answer) = settle(
                entry,
                end,
                stats,
                StructuralPath::Structural,
                metrics,
                cache,
                request,
            ) {
                return answer;
            }
        }
    }
    read_cold(client, keys, cache, request, metrics).await
}

/// Step 3: probe, cold start and repair, until an answer or the reload cap
/// (module documentation, "The sequence").
async fn read_cold(
    client: &ValkeyClient,
    keys: &KeySchema,
    cache: &StructuralCache,
    request: &StructuralReadRequest<'_>,
    metrics: Option<&SharedMetrics>,
) -> Result<StructuralRead, ServiceError> {
    let object_id = request.object_id;
    // Reloads after this call's own cold starts failed their repair, capped
    // at `STRUCTURAL_READ_MAX_RELOADS_AFTER_COLD_START`.
    let mut reloads_after_cold_start: u32 = 0;
    loop {
        let started = Instant::now();
        let probe = read_structural_cache_sync(
            client,
            keys,
            &StructuralSyncRequest::probe(object_id, request.client_id),
        )
        .await?;
        let facts = match probe.header {
            SyncHeader::Absent => return Err(empty_header_not_found()),
            SyncHeader::Present(facts) => facts,
        };
        // The index cardinalities the cold start routes on (its step 0);
        // `None` when an index key holds a foreign type.
        let SyncChunk::Probe(cardinalities) = probe.chunk else {
            // The script's tag for the chunk, never the chunk itself: an
            // `Entries` answer carries up to the request's byte bound of
            // stream payload (user scene data), which has no place in an
            // error message or the log line that carries it. No `_` arm, so
            // a new variant must name its tag; `Probe` cannot reach here.
            let kind = match probe.chunk {
                SyncChunk::Probe(_) => "PROBE",
                SyncChunk::NoRange => "NO_RANGE",
                SyncChunk::Entries { .. } => "OK",
                SyncChunk::TooLarge { .. } => "TOO_LARGE",
            };
            return Err(ServiceError::Internal(format!(
                "structural_cache_sync.lua answered a probe of object {object_id} with a {kind} \
                 chunk, not PROBE"
            )));
        };
        if facts.seq.is_none() {
            return Err(partial_header_failed_precondition());
        }
        match check_commit_token(
            keys,
            object_id,
            &probe.commit_token,
            request.commit_token,
            request.operation,
        )? {
            TokenVerdict::NotLanded => {}
            TokenVerdict::Answered(answer) => return Ok(answered(answer)),
            TokenVerdict::WrongType(refusal) => {
                return Ok(StructuralRead::CommitTokenRecordWrongType { refusal })
            }
        }
        if facts.superseded.is_some() {
            return Err(superseded_failed_precondition());
        }
        // A chained call's predecessor, after the superseded check (the
        // canonical order), before the size gate, the cold start and planning.
        if let Some(answer) = check_predecessor(
            probe.commit_token.record.as_deref(),
            request.commit_token,
            request.operation,
        ) {
            return Ok(answered(answer));
        }
        if !facts.chs_complete_marked() {
            let reason = UncacheableReason::NoChsComplete;
            record_verdict(
                cache,
                metrics,
                &Verdict::new(object_id, request.layer, reason),
                started,
            );
            // The probe's own bound, when it is over the item ceiling, as a
            // probed cold start would carry it: the WATCH fallback of a large
            // unmarked layer registers its writer intent before its first
            // `WATCH` (P12). `None` when the probe reported no cardinalities.
            let above_ceiling = cardinalities.and_then(|cards| {
                probed_above_ceiling(
                    &cards.snapshot_cardinalities(false),
                    SnapshotCeilings::of_client(client),
                )
            });
            return Ok(StructuralRead::Uncacheable {
                reason,
                above_ceiling,
            });
        }

        // A reset too large for one structural commit goes to the WATCH path
        // now, before the cold start reads the whole layer only for the plan
        // to be refused by the size gate (`storage::structural_size_gate`).
        // The estimate is a lower bound on the plan's work, so a reset it lets
        // through is still gated after planning. Without cardinalities (a
        // foreign-typed index) there is no estimate, and the cold start runs.
        if let (Some(content_rows), Some(cards)) = (request.reset_content_rows, cardinalities) {
            let cards = cards.snapshot_cardinalities(facts.chs_complete_marked());
            let estimate = reset_work_lower_bound(&cards, content_rows);
            if estimate > request.work_limit() {
                return Ok(StructuralRead::OverWorkLimit {
                    estimate,
                    cardinalities: cards,
                    above_ceiling: probed_above_ceiling(
                        &cards,
                        SnapshotCeilings::of_client(client),
                    ),
                });
            }
        }

        #[cfg(any(test, feature = "test-support"))]
        if let Some(hook) = request.after_probe_hook.as_ref() {
            hook().await;
        }

        let loaded = match cold_start_for(
            client,
            keys,
            cache,
            object_id,
            request.client_id,
            request.layer,
            // A foreign-typed index leaves no cardinalities to route on: the
            // cold start then runs unprobed, plain read first, whose script
            // refuses that index with its per-family error and ticks
            // `live_edit_snapshot_read_type_error_total{key_family}`.
            cardinalities.map(|cards| cards.snapshot_cardinalities(facts.chs_complete_marked())),
            request.rpc_deadline,
            request.remaining_deadline.as_ref(),
            request.cold_start_charge(),
        )
        .await
        {
            Ok(ColdStart::Loaded(loaded)) => loaded,
            // The decoded entry was stored, uncharged — unless a plain read's
            // hold was over its allowance, when nothing was — and the call
            // waits for its charge under the cold start's path; its next read
            // is a hit, or a cold start under the charge.
            Ok(ColdStart::NeedsCharge { target }) => {
                return Ok(StructuralRead::NeedsCharge {
                    target,
                    path: StructuralPath::StructuralFull,
                })
            }
            // Retired or audit-dropped during the read, or a header the read
            // could not classify: the next probe tells which.
            Ok(ColdStart::Invalidated) | Err(ServiceError::NotFound(_)) => continue,
            Ok(ColdStart::Superseded { commit_token }) => {
                if let Some(answer) = check_commit_token_record(
                    commit_token.as_deref(),
                    request.commit_token,
                    request.operation,
                )? {
                    return Ok(answered(answer));
                }
                return Err(superseded_failed_precondition());
            }
            Ok(ColdStart::Uncacheable {
                reason,
                above_ceiling,
            }) => {
                return Ok(StructuralRead::Uncacheable {
                    reason,
                    above_ceiling,
                })
            }
            // The token record's key changed type after the probe read it.
            Ok(ColdStart::CommitTokenRecordWrongType { refusal }) => {
                return Ok(StructuralRead::CommitTokenRecordWrongType { refusal })
            }
            Err(error) => return Err(error),
        };

        let mut entry = loaded.entry;
        // An `Err` drops the guard, which puts the loaded entry back: it is a
        // valid cut. A loaded entry's object was not superseded (that is
        // `ColdStart::Superseded`, above), so the own-key match is followed
        // directly by a chained call's predecessor, as in the canonical order.
        if let Some(answer) = check_commit_token_record(
            loaded.commit_token.as_deref(),
            request.commit_token,
            request.operation,
        )?
        .or_else(|| {
            check_predecessor(
                loaded.commit_token.as_deref(),
                request.commit_token,
                request.operation,
            )
        }) {
            count_put_back(metrics, cache, request, entry.put_back_held());
            return Ok(answered(answer));
        }

        #[cfg(any(test, feature = "test-support"))]
        if let Some(hook) = request.after_cold_start_hook.as_ref() {
            hook().await;
        }

        let (end, stats) = repair(
            client,
            keys,
            &mut entry,
            header_goal(request),
            request.repair_bounds,
            request.commit_charge,
            None,
            &request.repair_hooks(),
        )
        .await?;
        observe_repair(metrics, &stats);
        if let Some(answer) = settle(
            entry,
            end,
            stats,
            StructuralPath::StructuralFull,
            metrics,
            cache,
            request,
        ) {
            return answer;
        }
        // `settle` discarded the entry and counted the reload's reason.
        reloads_after_cold_start += 1;
        if reloads_after_cold_start > STRUCTURAL_READ_MAX_RELOADS_AFTER_COLD_START {
            // This call only: nothing recorded (the constant's docs), and no
            // cold-start `uncacheable` route tick — each cold start that ran
            // loaded its entry and ticked the route that loaded it. Reported
            // as this call's verdict.
            let reason = UncacheableReason::ReloadLimit;
            report_uncacheable(
                metrics,
                cache,
                &Verdict::new(object_id, request.layer, reason),
            );
            return Ok(StructuralRead::Uncacheable {
                reason,
                above_ceiling: None,
            });
        }
    }
}

/// The repair goal for `request`: the header, checking its commit token.
fn header_goal<'r>(request: &'r StructuralReadRequest<'_>) -> Goal<'r> {
    Goal::Header {
        client_id: request.client_id,
        commit_token: request.commit_token,
        operation: request.operation,
    }
}

/// The read's answer for a commit-token check that answered the call: this
/// call's own commit is [`StructuralRead::AlreadyLanded`], an earlier call's
/// [`StructuralRead::Recorded`], a moved start record
/// [`StructuralRead::Overtaken`], a chained call's pending predecessor
/// [`StructuralRead::PredecessorPending`].
fn answered(answer: TokenAnswer) -> StructuralRead {
    match answer {
        TokenAnswer::Recorded(RecordedCommit {
            by: RecordedBy::ThisCall,
            seq,
            server_ts,
        }) => StructuralRead::AlreadyLanded { seq, server_ts },
        TokenAnswer::Recorded(recorded) => StructuralRead::Recorded(recorded),
        TokenAnswer::Overtaken => StructuralRead::Overtaken,
        TokenAnswer::PredecessorPending { record } => StructuralRead::PredecessorPending { record },
    }
}

/// Settle `entry` by how its repair ended, and give the call's answer — or
/// `None` for a reload: the entry is discarded and the reason counted, and the
/// caller reads the object again.
fn settle(
    entry: CheckedOut,
    end: RepairEnd,
    stats: RepairStats,
    path: StructuralPath,
    metrics: Option<&SharedMetrics>,
    cache: &StructuralCache,
    request: &StructuralReadRequest<'_>,
) -> Option<Result<StructuralRead, ServiceError>> {
    Some(match end {
        RepairEnd::Current => Ok(StructuralRead::Ready(ReadyEntry {
            entry,
            path,
            repair: stats,
        })),
        RepairEnd::Answered(answer) => {
            count_put_back(metrics, cache, request, entry.put_back_held());
            Ok(answered(answer))
        }
        RepairEnd::CommitTokenWrongType { refusal } => {
            count_put_back(metrics, cache, request, entry.put_back_held());
            Ok(StructuralRead::CommitTokenRecordWrongType { refusal })
        }
        RepairEnd::Gone(error) => {
            entry.discard();
            Err(error)
        }
        RepairEnd::Reload(reason) => {
            if let Some(m) = metrics {
                m.inc_structural_cache_reload(reason.label());
            }
            entry.discard();
            return None;
        }
        // A repair charge point (RP0, RP1 or RP2) refused — RP0 and RP1
        // between rounds, RP2 between two records of a round: the entry is a
        // valid, normalized cut, so it goes back, and the call waits under
        // `path`.
        RepairEnd::NeedsCharge { target } => {
            count_put_back(metrics, cache, request, entry.put_back_held());
            Ok(StructuralRead::NeedsCharge { target, path })
        }
    })
}

/// Count how an explicit put-back of a structural call's entry settled
/// (structural-edits spec SR-2.4; C3 amendment, 2026-10-04, decider-approved):
/// a put-back the per-entry cap refused ([`PutBack::OverCap`]) ticks
/// `live_edit_structural_cache_reloads_total{reason="uncacheable"}` — the
/// entry grew past the cap while cached, the object now has an `OverWeight`
/// negative entry, and its structural edits take the uncacheable path until
/// that expires. That negative entry is a verdict, so it is also reported
/// once, as `over_weight`, by taking it
/// ([`StructuralCache::take_unreported`]; `structural_cache_cold_start::report_uncacheable`:
/// `live_edit_structural_cache_uncacheable_total` and the INFO line naming
/// `request`'s layer and object, with the weight the negative entry recorded
/// against the per-entry cap). Stored and epoch-refused put-backs tick and
/// log nothing. Every explicit put-back of the structural route goes through
/// here: the read side's own and `LayerStore::update_layer_once_structural`'s.
pub(crate) fn count_put_back(
    metrics: Option<&SharedMetrics>,
    cache: &StructuralCache,
    request: &StructuralReadRequest<'_>,
    settled: PutBack,
) {
    if settled != PutBack::OverCap {
        return;
    }
    if let Some(m) = metrics {
        m.inc_structural_cache_reload(ReloadReason::Uncacheable.label());
    }
    // The refusal recorded the verdict, with the weight, on the object's
    // negative entry, unreported; take it to report it. Should the object
    // have been invalidated since (a retirement or audit drop racing the
    // put-back), there is no verdict left to report.
    if let Some(reason) = cache.take_unreported(request.object_id) {
        report_uncacheable(
            metrics,
            cache,
            &Verdict::new(request.object_id, request.layer, reason),
        );
    }
}

fn tick_lookup(metrics: Option<&SharedMetrics>, outcome: &'static str) {
    if let Some(m) = metrics {
        m.inc_structural_cache_lookup(outcome);
    }
}

fn observe_repair(metrics: Option<&SharedMetrics>, stats: &RepairStats) {
    if let Some(m) = metrics {
        m.observe_structural_cache_repair(stats.replayed_entries, stats.bytes);
    }
}
