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

//! A per-replica byte budget, [`MemoryBudget`] (Spec SDD SR-7.2), of which a
//! replica holds two instances:
//!
//! - **the catch-up memory budget `T`** (`ValkeyClient::catchup_build_budget`),
//!   which catch-up flights, structural cache cold starts and audits reserve
//!   from — what the rest of these docs describe unless they name the other;
//! - **the retirement allowance `R_ref`** (`LayerStore::retirement_budget`),
//!   which external-value clean-up of a retiring object reserves from:
//!   `DeleteLayer`, the `CreateLayer` overwrite's cleanup body and the orphan
//!   sweep (`storage::retirement`). Each reserves its charge — a page sized
//!   from its largest structural index, at most the page bound, plus a charge
//!   per reference it releases — tops up for references beyond those
//!   `obj:ext` counted and for a page widened past an index that grew, and
//!   shrinks to its page once its release returns.
//!
//! Each instance ticks its own metric series ([`BudgetSeries`]): `R_ref`'s
//! traffic never moves the catch-up gauges operators read against `T`.
//!
//! A budget of bytes, not a count of catch-ups: layer sizes span orders of
//! magnitude, so a count cannot bound memory (the catch-up slot,
//! `CatchUpGate`, stays the CPU bound beside it). It is not a bare
//! `tokio::sync::Semaphore` because it must accept never-blocking top-ups that
//! take it past its total, and a semaphore's permits cannot go negative.
//!
//! - **Total `T`** (derived from the pod memory limit, `config::memory_plan`;
//!   `R_ref` likewise), held in KiB.
//!   Every request is in bytes, rounded up to whole KiB, and **clamped to
//!   `T`**, so an oversized request runs alone rather than never.
//! - **Grants in arrival order.** A reservation of `n` is granted when
//!   `reserved + overcommit + n ≤ T`. With `n` clamped, a request of the whole
//!   `T` is granted only when nothing else is held, and while it is held no
//!   other grant fits: it runs alone. A request that arrives while others
//!   queue joins the back even if it would fit, so a large request at the head
//!   holds smaller ones behind it instead of starving; every wait is bounded by
//!   its caller's deadline, or by shutdown or the caller dropping it.
//! - **Non-blocking grant** ([`MemoryBudget::try_grant`]): granted only when no
//!   request is queued and it fits; never joins the queue.
//! - **Forced top-up** ([`MemoryBudget::top_up`]): adds to `overcommit` at
//!   once and never waits or fails. It is for memory already resident (a
//!   snapshot read that came out heavier than its grant), where refusing
//!   would only waste the paid read.
//! - **A grant's own top-up.** A fenced snapshot read holds at most one
//!   top-up beside its grant, one guard that grows and shrinks in place:
//!   [`MemoryGrant::top_up_to`] grows it as the read reports its progress
//!   (after each fan-out window, to `SNAPSHOT_MEMORY_FACTOR` × the reply
//!   bytes read so far), never past `T` for grant and top-up together, and
//!   [`MemoryGrant::adjust_with_top_up`] sets the pair to the measured
//!   weight once the read is verified: the grant shrunk and the top-up
//!   dropped, or the grant kept and the top-up exactly the difference. So the
//!   memory a read holds resident is accounted while it reads, not only
//!   after, and no top-up is counted twice.
//! - **Guards.** Every grant and top-up is a guard that repays exactly what it
//!   holds when dropped, from `reserved` or `overcommit` respectively, and
//!   wakes the queue. Debt is cancelled with the memory that caused it; it is
//!   never turned into grantable room.
//!
//! **The bound** (the catch-up budget's; `R_ref`'s is in `storage::retirement`).
//! Every grant requires `reserved + overcommit + n ≤ T`, and that
//! sum counts every top-up still held, so resident memory under the budget is
//! at most `T` plus the top-ups taken since the last grant. That term is small
//! because only a budgeted fenced read (a catch-up flight's, or a structural
//! cache cold start's or audit's) grows memory past its grant, and a budgeted
//! fenced read runs only under a fenced read permit (`CatchUpReadGate`, class
//! `fenced`), each holder one top-up that keeps its grant and top-up together
//! within `T` however often it grows, so at most `P_f` top-ups, each at most
//! `T`, are outstanding beyond `T`. (The checkpoint assembler's fenced read
//! takes no permit and no budget, so it is outside this bound.) A top-up is
//! never withheld: it records memory the read already holds, and withholding
//! it would only hide that memory from admission. One more term sits beside
//! them: a fenced read stage's fan-out window transient `X` (the per-command
//! read state and the replies of the window in flight, held as client
//! values, which scale with the window's commands, not with `reply_bytes`,
//! and the allocator pages they leave dirty), which the flight's grant does
//! not reserve. Every fenced fan-out pass runs on the one `fanout-read` thread and
//! the snapshot lane's routers on the one `snapshot-io` thread
//! (`storage::fanout_thread`), so `X` is one read stage in one allocator
//! arena whatever the replica's worker count: polled on the service's
//! workers, a read left dirty pages in every worker arena it moved through,
//! and its resident memory beyond its heap grew with the CPU limit. With one
//! read stage per fenced permit, resident above-ceiling catch-up memory is at
//! most `T × (1 + P_f) + P_f × X` for layers whose measured weight is within
//! `T`; the pod memory plan funds the `P_f × X` term beside `T × (1 + P_f)`
//! (`config::memory_plan`).
//!
//! **Waits** ([`MemoryBudget::reserve`]) take an optional deadline (a caller
//! with no capacity budget of its own passes none, as every retirement does)
//! and end [`CatchUpRefusal::Closed`] once the budget's shutdown call runs:
//! [`MemoryBudget::close`] for the catch-up budget, which refuses everything
//! after it, and [`MemoryBudget::end_waits`] for `R_ref` (the first step of
//! `LayerStore::wait_for_cleanup`), which refuses only a reservation that
//! would have to wait and still grants one that fits.
//! Like the catch-up slot and read-permit waits, session cancellation and
//! client disconnect are the caller's to race: dropping a wait leaves the
//! queue and wakes whoever it was holding back.

use std::collections::{HashSet, VecDeque};
use std::sync::{Arc, Mutex as StdMutex, MutexGuard, PoisonError};
use std::time::Instant;

use tokio::sync::Notify;

use crate::observability::metrics::{
    SharedMetrics, CATCHUP_BUILD_ADMISSION_RESULT_CANCELLED,
    CATCHUP_BUILD_ADMISSION_RESULT_IMMEDIATE, CATCHUP_BUILD_ADMISSION_RESULT_REFUSED_TIMEOUT,
    CATCHUP_BUILD_ADMISSION_RESULT_WAITED, RETIREMENT_WAIT_OUTCOME_CANCELLED,
    RETIREMENT_WAIT_OUTCOME_IMMEDIATE, RETIREMENT_WAIT_OUTCOME_WAITED,
};
use crate::storage::subscriber_admission::CatchUpRefusal;

/// Memory a decoded catch-up snapshot holds, as a multiple of the snapshot
/// read's `reply_bytes` tally (`AtomicSnapshot::reply_bytes`), for sizing a
/// reservation (Spec SDD SR-7.2): 21, measured as part of a combined factor
/// and kept.
///
/// The snapshot and one build together are 42 × `reply_bytes`, split evenly
/// with [`BUILD_MEMORY_FACTOR`]. Task 10 measured the pair on the implemented
/// fenced reader as one combined factor (the read's peak covers the read and
/// the build; the split is not separable, because every measured read's
/// largest allocation sample fell inside its read stage): a read's peak fits a
/// fixed term plus a per-byte part, about 5.3 × `reply_bytes` allocated and
/// 16.8 × RSS at four times the item ceiling and above. The fixed term is
/// taken to be the read stage's fan-out window transient (`X`, which the
/// bound funds beside `T`; see the module docs), an inference from the fit.
/// 42 over-covers that per-byte part at every size, so the factors are kept.
/// Recorded in the Task 10 section of
/// `docs-internal/validation/fenced-snapshot-reads-20261002/report.md` (full
/// precision in that folder's `raw/task10-results/results.json`, `memory`).
/// Epic 5 re-fits the grant's form (a window term per command, the factors on
/// RSS) for both read arms.
pub const SNAPSHOT_MEMORY_FACTOR: u64 = 21;

/// Peak memory of one catch-up build, as a multiple of the snapshot read's
/// `reply_bytes` tally, for sizing a reservation (Spec SDD SR-7.2): 21, the
/// other half of the combined 42 (see [`SNAPSHOT_MEMORY_FACTOR`], same
/// record). An extra build's part, sized from this factor alone, would be
/// under-sized if the build were the larger share; a combined
/// snapshot-plus-build grant is right either way. The measured bursts bound
/// the build's share from above: 50 joins of a layer of ten times the item
/// ceiling grew peak memory by one read plus at most about 85 MiB for 49
/// builds and their payloads, under 2 MiB per build against 21 ×
/// `reply_bytes` (about 88 MiB) per build.
pub const BUILD_MEMORY_FACTOR: u64 = 21;

/// The budget's unit, in bytes.
const KIB: u64 = 1024;

/// `bytes` in the budget's unit, whole KiB, rounded up so a reservation never
/// covers less than it was asked for.
pub fn kib_for_bytes(bytes: u64) -> u64 {
    bytes.div_ceil(KIB)
}

/// `kib` as a byte gauge delta, saturating rather than wrapping.
fn gauge_bytes(kib: u64) -> i64 {
    i64::try_from(kib.saturating_mul(KIB)).unwrap_or(i64::MAX)
}

/// How one [`MemoryBudget::reserve`] call ended, for its instance's series
/// ([`BudgetSeries::record_wait`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetWaitOutcome {
    /// Granted without queueing.
    Immediate,
    /// Granted after queueing for memory to free.
    Waited,
    /// Refused at the caller's deadline.
    TimedOut,
    /// Ended without memory: the budget closed or its waits ended
    /// ([`MemoryBudget::end_waits`]), or the caller dropped the wait.
    Cancelled,
}

/// The metric series one [`MemoryBudget`] instance ticks: its grants,
/// top-ups and waits, and nothing of any other instance's, so one budget's
/// traffic never moves another's gauges. Each instance names its own
/// implementation when it is built ([`MemoryBudget::with_series`]); a new
/// budget instance adds an implementation of its own, beside
/// [`CATCH_UP_BUILD_SERIES`] and [`RETIREMENT_SERIES`].
pub trait BudgetSeries: std::fmt::Debug + Send + Sync {
    /// Move this series' reserved-bytes gauge by `delta`.
    fn add_reserved(&self, m: &SharedMetrics, delta: i64);

    /// Move this series' over-commit gauge by `delta`.
    fn add_overcommit(&self, m: &SharedMetrics, delta: i64);

    /// Record one [`MemoryBudget::reserve`] call's wait, `waited` long, which
    /// ended with `outcome`.
    fn record_wait(
        &self,
        m: &SharedMetrics,
        waited: std::time::Duration,
        outcome: BudgetWaitOutcome,
    );

    /// Record one successful non-blocking grant ([`MemoryBudget::try_grant`]).
    fn record_try_grant(&self, m: &SharedMetrics);
}

/// The catch-up memory budget `T`'s series:
/// `live_edit_catchup_build_budget_{reserved,overcommit}_bytes`,
/// `live_edit_catchup_build_wait_seconds` and
/// `live_edit_catchup_build_admission_total{result}` (a non-blocking grant
/// counts `immediate`).
#[derive(Debug)]
pub struct CatchUpBuildSeries;

/// [`CatchUpBuildSeries`], the series [`MemoryBudget::new`] ticks.
pub static CATCH_UP_BUILD_SERIES: CatchUpBuildSeries = CatchUpBuildSeries;

impl BudgetSeries for CatchUpBuildSeries {
    fn add_reserved(&self, m: &SharedMetrics, delta: i64) {
        m.add_catchup_build_budget_reserved_bytes(delta);
    }

    fn add_overcommit(&self, m: &SharedMetrics, delta: i64) {
        m.add_catchup_build_budget_overcommit_bytes(delta);
    }

    fn record_wait(
        &self,
        m: &SharedMetrics,
        waited: std::time::Duration,
        outcome: BudgetWaitOutcome,
    ) {
        m.observe_catchup_build_wait(waited);
        m.inc_catchup_build_admission(match outcome {
            BudgetWaitOutcome::Immediate => CATCHUP_BUILD_ADMISSION_RESULT_IMMEDIATE,
            BudgetWaitOutcome::Waited => CATCHUP_BUILD_ADMISSION_RESULT_WAITED,
            BudgetWaitOutcome::TimedOut => CATCHUP_BUILD_ADMISSION_RESULT_REFUSED_TIMEOUT,
            BudgetWaitOutcome::Cancelled => CATCHUP_BUILD_ADMISSION_RESULT_CANCELLED,
        });
    }

    fn record_try_grant(&self, m: &SharedMetrics) {
        m.inc_catchup_build_admission(CATCHUP_BUILD_ADMISSION_RESULT_IMMEDIATE);
    }
}

/// The retirement allowance `R_ref`'s series (`storage::retirement`):
/// `live_edit_retirement_budget_{reserved,overcommit}_bytes` and
/// `live_edit_retirement_wait_seconds{outcome}`. It has no admission counter,
/// and a non-blocking grant records nothing on it. A refusal at a deadline —
/// which no retirement passes — is recorded as the caller ending its own
/// wait, `cancelled`.
#[derive(Debug)]
pub struct RetirementSeries;

/// [`RetirementSeries`], the series the retirement allowance ticks.
pub static RETIREMENT_SERIES: RetirementSeries = RetirementSeries;

impl BudgetSeries for RetirementSeries {
    fn add_reserved(&self, m: &SharedMetrics, delta: i64) {
        m.add_retirement_budget_reserved_bytes(delta);
    }

    fn add_overcommit(&self, m: &SharedMetrics, delta: i64) {
        m.add_retirement_budget_overcommit_bytes(delta);
    }

    fn record_wait(
        &self,
        m: &SharedMetrics,
        waited: std::time::Duration,
        outcome: BudgetWaitOutcome,
    ) {
        m.observe_retirement_wait(
            waited,
            match outcome {
                BudgetWaitOutcome::Immediate => RETIREMENT_WAIT_OUTCOME_IMMEDIATE,
                BudgetWaitOutcome::Waited => RETIREMENT_WAIT_OUTCOME_WAITED,
                BudgetWaitOutcome::TimedOut | BudgetWaitOutcome::Cancelled => {
                    RETIREMENT_WAIT_OUTCOME_CANCELLED
                }
            },
        );
    }

    fn record_try_grant(&self, _m: &SharedMetrics) {}
}

/// A per-replica memory budget; see the module docs.
pub struct MemoryBudget {
    total_kib: u64,
    state: StdMutex<State>,
    metrics: Arc<StdMutex<Option<SharedMetrics>>>,
    series: &'static dyn BudgetSeries,
}

struct State {
    /// KiB held by grant guards.
    reserved_kib: u64,
    /// KiB held by top-up guards.
    overcommit_kib: u64,
    closed: bool,
    /// Set once by [`MemoryBudget::end_waits`]: no reservation waits any more
    /// — one that fits is still granted at once, one that would wait is
    /// refused [`CatchUpRefusal::Closed`]. Never cleared.
    waits_ended: bool,
    /// Waiting reservations, oldest first.
    queue: VecDeque<Waiter>,
    /// Waiters granted from the queue whose wait has not yet picked the grant
    /// up. The grant is already counted in `reserved_kib`; a wait dropped
    /// before picking it up gives it back.
    handed: HashSet<u64>,
    next_id: u64,
}

struct Waiter {
    id: u64,
    kib: u64,
    notify: Arc<Notify>,
    /// The handle the waiting call captured, so the gauge moves on the same
    /// handle at grant time as at the grant guard's drop.
    metrics: Option<SharedMetrics>,
}

impl std::fmt::Debug for MemoryBudget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let st = self.lock();
        f.debug_struct("MemoryBudget")
            .field("total_kib", &self.total_kib)
            .field("reserved_kib", &st.reserved_kib)
            .field("overcommit_kib", &st.overcommit_kib)
            .field("waiters", &st.queue.len())
            .field("closed", &st.closed)
            .field("waits_ended", &st.waits_ended)
            .finish()
    }
}

impl MemoryBudget {
    /// A catch-up budget ([`CATCH_UP_BUILD_SERIES`]) of `total_kib`. At
    /// least 1 KiB: configuration rejects 0 (Spec SDD SR-7.2), and a budget
    /// built with 0 anyway is raised to 1 KiB, because every request clamps to
    /// the total, so a zero total would clamp each to nothing and bound
    /// nothing while still reading as a budget.
    pub fn new(total_kib: u64, metrics: Arc<StdMutex<Option<SharedMetrics>>>) -> Self {
        Self::with_series(total_kib, metrics, &CATCH_UP_BUILD_SERIES)
    }

    /// A budget of `total_kib` (at least 1 KiB, as [`Self::new`]) ticking
    /// `series`.
    pub fn with_series(
        total_kib: u64,
        metrics: Arc<StdMutex<Option<SharedMetrics>>>,
        series: &'static dyn BudgetSeries,
    ) -> Self {
        Self {
            total_kib: total_kib.max(1),
            state: StdMutex::new(State {
                reserved_kib: 0,
                overcommit_kib: 0,
                closed: false,
                waits_ended: false,
                queue: VecDeque::new(),
                handed: HashSet::new(),
                next_id: 0,
            }),
            metrics,
            series,
        }
    }

    /// The series this budget ticks.
    pub fn series(&self) -> &'static dyn BudgetSeries {
        self.series
    }

    /// The total `T`, in bytes.
    pub fn total_bytes(&self) -> u64 {
        self.total_kib.saturating_mul(KIB)
    }

    /// Bytes currently held by grants.
    pub fn reserved_bytes(&self) -> u64 {
        self.lock().reserved_kib.saturating_mul(KIB)
    }

    /// Bytes currently held by top-ups (over-commit).
    pub fn overcommit_bytes(&self) -> u64 {
        self.lock().overcommit_kib.saturating_mul(KIB)
    }

    /// Reservations waiting in the queue.
    pub fn waiters(&self) -> usize {
        self.lock().queue.len()
    }

    pub fn is_closed(&self) -> bool {
        self.lock().closed
    }

    /// Whether [`Self::end_waits`] ran.
    pub fn waits_ended(&self) -> bool {
        self.lock().waits_ended
    }

    /// `bytes` as the KiB a request of that size takes: rounded up, then
    /// clamped to the total.
    pub fn clamped_kib(&self, bytes: u64) -> u64 {
        kib_for_bytes(bytes).min(self.total_kib)
    }

    /// Refuses every later reservation and non-blocking grant, and ends every
    /// queued wait [`CatchUpRefusal::Closed`]. Grants and top-ups already held
    /// stay valid until dropped, and a top-up still succeeds: it accounts for
    /// memory that is already resident. `ValkeyClient::quit` calls it on the
    /// catch-up budget at shutdown, beside closing the read-permit gates. The
    /// retirement allowance is never closed: shutdown ends its waits instead
    /// ([`Self::end_waits`]).
    pub fn close(&self) {
        let mut st = self.lock();
        st.closed = true;
        for waiter in st.queue.drain(..) {
            waiter.notify.notify_one();
        }
    }

    /// Ends every queued wait [`CatchUpRefusal::Closed`], and from then on
    /// lets no reservation wait: one that fits is still granted at once (the
    /// queue is empty), one that does not is refused `Closed` at once and
    /// never queued. Unlike [`Self::close`], memory that is free stays
    /// grantable. A grant already handed to a waiter is kept, grants and
    /// top-ups held stay valid, and [`Self::try_grant`] and [`Self::top_up`]
    /// are unchanged. One-way. `LayerStore::wait_for_cleanup` calls it on the
    /// retirement allowance as its first step, so only a clean-up that is
    /// queued, or would have to queue, ends without memory at shutdown.
    pub fn end_waits(&self) {
        let mut st = self.lock();
        st.waits_ended = true;
        for waiter in st.queue.drain(..) {
            waiter.notify.notify_one();
        }
    }

    /// Reserves `bytes` (rounded up to KiB, clamped to the total): at once when
    /// nothing is queued and it fits, otherwise after a FIFO wait for memory to
    /// free. With a `deadline`, the wait ends [`CatchUpRefusal::TimedOut`] there,
    /// and a deadline already passed never joins the queue; with none, it waits
    /// until granted, closed or dropped. Refuses [`CatchUpRefusal::Closed`] once
    /// [`Self::close`] ran, and — when it would have to wait — once
    /// [`Self::end_waits`] ran. The caller races this against its own endings;
    /// dropping it mid-wait leaves the queue and wakes the next waiter.
    ///
    /// Records one `live_edit_catchup_build_wait_seconds` observation and one
    /// `live_edit_catchup_build_admission_total{result}` per call, whatever the
    /// outcome: `immediate`, `waited`, `refused_timeout`, or `cancelled` for a
    /// closed budget, ended waits or a dropped wait. A retirement allowance
    /// records one
    /// `live_edit_retirement_wait_seconds{outcome}` instead
    /// ([`RETIREMENT_SERIES`]).
    pub async fn reserve(
        self: &Arc<Self>,
        bytes: u64,
        deadline: Option<tokio::time::Instant>,
    ) -> Result<MemoryGrant, CatchUpRefusal> {
        let kib = self.clamped_kib(bytes);
        let metrics = self.metrics();
        let mut observation = BuildWaitObservation::start(metrics.clone(), self.series);
        let (id, notify) = {
            let mut st = self.lock();
            if st.closed {
                observation.finish(BudgetWaitOutcome::Cancelled);
                return Err(CatchUpRefusal::Closed);
            }
            if st.queue.is_empty() && self.fits(&st, kib) {
                add_reserved(&mut st, kib, metrics.as_ref(), self.series);
                drop(st);
                observation.finish(BudgetWaitOutcome::Immediate);
                return Ok(MemoryGrant::new(self, kib, metrics));
            }
            if st.waits_ended {
                observation.finish(BudgetWaitOutcome::Cancelled);
                return Err(CatchUpRefusal::Closed);
            }
            if deadline.is_some_and(|d| d <= tokio::time::Instant::now()) {
                observation.finish(BudgetWaitOutcome::TimedOut);
                return Err(CatchUpRefusal::TimedOut);
            }
            let id = st.next_id;
            st.next_id = st.next_id.wrapping_add(1);
            let notify = Arc::new(Notify::new());
            st.queue.push_back(Waiter {
                id,
                kib,
                notify: Arc::clone(&notify),
                metrics: metrics.clone(),
            });
            (id, notify)
        };

        let mut ticket = QueuedReservation {
            budget: self,
            id,
            kib,
            metrics: metrics.clone(),
            settled: false,
        };
        let outcome = match deadline {
            None => ticket.wait(&notify).await,
            Some(deadline) => {
                let waited = tokio::time::timeout_at(deadline, ticket.wait(&notify)).await;
                match waited {
                    Ok(outcome) => outcome,
                    Err(_elapsed) => ticket.settle_at_deadline(),
                }
            }
        };
        match outcome {
            Ok(()) => {
                observation.finish(BudgetWaitOutcome::Waited);
                Ok(MemoryGrant::new(self, kib, metrics))
            }
            Err(refusal) => {
                observation.finish(match refusal {
                    CatchUpRefusal::TimedOut => BudgetWaitOutcome::TimedOut,
                    CatchUpRefusal::Closed => BudgetWaitOutcome::Cancelled,
                });
                Err(refusal)
            }
        }
    }

    /// Grants `bytes` (rounded up to KiB, clamped to the total) only if the
    /// budget is open, no reservation is queued and it fits; never waits. A
    /// grant counts `live_edit_catchup_build_admission_total{result="immediate"}`
    /// on the catch-up series (nothing on the retirement series); a decline
    /// counts nothing, since nothing was refused that was waiting.
    pub fn try_grant(self: &Arc<Self>, bytes: u64) -> Option<MemoryGrant> {
        let kib = self.clamped_kib(bytes);
        let metrics = self.metrics();
        {
            let mut st = self.lock();
            if st.closed || !st.queue.is_empty() || !self.fits(&st, kib) {
                return None;
            }
            add_reserved(&mut st, kib, metrics.as_ref(), self.series);
        }
        if let Some(m) = &metrics {
            self.series.record_try_grant(m);
        }
        Some(MemoryGrant::new(self, kib, metrics))
    }

    /// Records `bytes` (rounded up to KiB, clamped to the total) as
    /// over-commit at once: never waits, never fails, closed budget included.
    /// The memory is repaid when the returned guard drops.
    pub fn top_up(self: &Arc<Self>, bytes: u64) -> MemoryTopUp {
        self.top_up_kib(self.clamped_kib(bytes))
    }

    fn top_up_kib(self: &Arc<Self>, kib: u64) -> MemoryTopUp {
        let metrics = self.metrics();
        {
            let mut st = self.lock();
            st.overcommit_kib = st.overcommit_kib.saturating_add(kib);
        }
        if let Some(m) = &metrics {
            self.series.add_overcommit(m, gauge_bytes(kib));
        }
        MemoryTopUp {
            budget: Arc::clone(self),
            kib,
            metrics,
        }
    }

    /// Whether `kib` (already clamped) fits beside what is held.
    fn fits(&self, st: &State, kib: u64) -> bool {
        st.reserved_kib
            .saturating_add(st.overcommit_kib)
            .saturating_add(kib)
            <= self.total_kib
    }

    /// Grants queued reservations from the head, in order, while the head
    /// fits. Called after anything that frees memory or removes a waiter, so
    /// a waiter that was held back only by the one ahead of it is not left
    /// waiting.
    fn grant_queued(&self, st: &mut State) {
        loop {
            let head_fits = match st.queue.front() {
                Some(head) => self.fits(st, head.kib),
                None => return,
            };
            if !head_fits {
                return;
            }
            let Some(waiter) = st.queue.pop_front() else {
                return;
            };
            add_reserved(st, waiter.kib, waiter.metrics.as_ref(), self.series);
            st.handed.insert(waiter.id);
            waiter.notify.notify_one();
        }
    }

    /// Gives `kib` of grants back and wakes the queue.
    fn release_reserved(&self, kib: u64, metrics: Option<&SharedMetrics>) {
        let mut st = self.lock();
        st.reserved_kib = st.reserved_kib.saturating_sub(kib);
        if let Some(m) = metrics {
            self.series.add_reserved(m, -gauge_bytes(kib));
        }
        self.grant_queued(&mut st);
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        // Nothing panics while holding the lock, and every update leaves the
        // state consistent, so a poisoned lock's state is still sound.
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn metrics(&self) -> Option<SharedMetrics> {
        self.metrics.lock().ok().and_then(|g| g.clone())
    }
}

fn add_reserved(
    st: &mut State,
    kib: u64,
    metrics: Option<&SharedMetrics>,
    series: &'static dyn BudgetSeries,
) {
    st.reserved_kib = st.reserved_kib.saturating_add(kib);
    if let Some(m) = metrics {
        series.add_reserved(m, gauge_bytes(kib));
    }
}

/// A reservation sitting in the queue. Until it is settled, dropping it (the
/// caller stopped waiting) removes it from the queue, or gives back a grant
/// it was handed but had not picked up, and wakes the queue either way.
struct QueuedReservation<'a> {
    budget: &'a Arc<MemoryBudget>,
    id: u64,
    kib: u64,
    metrics: Option<SharedMetrics>,
    settled: bool,
}

impl QueuedReservation<'_> {
    /// Waits until this reservation is granted (`Ok`), or the budget closes or
    /// its waits end ([`MemoryBudget::end_waits`]) before it was.
    async fn wait(&mut self, notify: &Notify) -> Result<(), CatchUpRefusal> {
        loop {
            notify.notified().await;
            let mut st = self.budget.lock();
            if st.handed.remove(&self.id) {
                self.settled = true;
                return Ok(());
            }
            if st.closed || st.waits_ended {
                self.settled = true;
                return Err(CatchUpRefusal::Closed);
            }
        }
    }

    /// The deadline passed: keep a grant handed at the last moment, otherwise
    /// leave the queue.
    fn settle_at_deadline(&mut self) -> Result<(), CatchUpRefusal> {
        let mut st = self.budget.lock();
        self.settled = true;
        if st.handed.remove(&self.id) {
            return Ok(());
        }
        if st.closed || st.waits_ended {
            return Err(CatchUpRefusal::Closed);
        }
        st.queue.retain(|w| w.id != self.id);
        self.budget.grant_queued(&mut st);
        Err(CatchUpRefusal::TimedOut)
    }
}

impl Drop for QueuedReservation<'_> {
    fn drop(&mut self) {
        if self.settled {
            return;
        }
        let mut st = self.budget.lock();
        if st.handed.remove(&self.id) {
            st.reserved_kib = st.reserved_kib.saturating_sub(self.kib);
            if let Some(m) = &self.metrics {
                self.budget.series.add_reserved(m, -gauge_bytes(self.kib));
            }
        } else {
            st.queue.retain(|w| w.id != self.id);
        }
        self.budget.grant_queued(&mut st);
    }
}

/// Memory granted by a [`MemoryBudget`]; given back when dropped, on every
/// exit: completion, error, cancellation of its owner, or a task abort.
pub struct MemoryGrant {
    budget: Arc<MemoryBudget>,
    kib: u64,
    metrics: Option<SharedMetrics>,
}

impl std::fmt::Debug for MemoryGrant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryGrant")
            .field("bytes", &self.bytes())
            .finish()
    }
}

impl MemoryGrant {
    fn new(budget: &Arc<MemoryBudget>, kib: u64, metrics: Option<SharedMetrics>) -> Self {
        Self {
            budget: Arc::clone(budget),
            kib,
            metrics,
        }
    }

    /// Bytes this grant holds, in whole KiB.
    pub fn bytes(&self) -> u64 {
        self.kib.saturating_mul(KIB)
    }

    /// Gives back everything above `bytes` (rounded up to KiB) and wakes the
    /// queue. A grant never grows this way: a target at or above what it
    /// holds changes nothing.
    pub fn shrink_to(&mut self, bytes: u64) {
        let target = kib_for_bytes(bytes);
        if target >= self.kib {
            return;
        }
        let freed = self.kib - target;
        self.kib = target;
        self.budget.release_reserved(freed, self.metrics.as_ref());
    }

    /// Grows `top_up`, this grant's held top-up, so that the two together
    /// cover `bytes` (rounded up to KiB, **clamped to the total**): a forced
    /// top-up for whatever the grant and the held top-up do not already cover,
    /// taken at once — it never waits and never fails, closed budget included.
    /// Never shrinks either. The held top-up grows in place; one is minted
    /// only when none is held, so the caller holds one guard however often it
    /// calls.
    ///
    /// For a read reporting its progress: memory already resident beyond the
    /// grant. The clamp is cumulative — the grant and its top-up together
    /// never exceed `T` — so repeated calls never take more than `T` for one
    /// grant. It is for memory a budgeted fenced read already holds; the
    /// permit that gates that read bounds the term (module docs, the bound).
    pub fn top_up_to(&self, top_up: &mut Option<MemoryTopUp>, bytes: u64) {
        let target = self.budget.clamped_kib(bytes);
        let held = top_up.as_ref().map_or(0, |t| t.kib);
        if target <= self.kib.saturating_add(held) {
            return;
        }
        self.set_top_up_kib(top_up, target - self.kib);
    }

    /// Adjusts this grant and its held top-up, `top_up`, to a measured weight,
    /// `measured_bytes` (rounded up to KiB, clamped to the total), so that
    /// together they hold exactly that weight:
    ///
    /// - at or below the grant, the grant shrinks to it and the top-up is
    ///   dropped;
    /// - above it, the grant is kept and the top-up becomes exactly the
    ///   difference: the held one grown or shrunk in place, or a forced one
    ///   minted (never waiting) when none is held.
    ///
    /// Never two top-ups for one grant, so a top-up taken from a read's
    /// progress ([`Self::top_up_to`]) is never counted twice, and calling it
    /// again with another measure re-adjusts the same pair. Whatever it gives
    /// back wakes the queue. The caller keeps both for as long as the memory
    /// they account for stays resident.
    pub fn adjust_with_top_up(&mut self, top_up: &mut Option<MemoryTopUp>, measured_bytes: u64) {
        let measured = self.budget.clamped_kib(measured_bytes);
        if measured <= self.kib {
            *top_up = None;
            self.shrink_to(measured.saturating_mul(KIB));
        } else {
            self.set_top_up_kib(top_up, measured - self.kib);
        }
    }

    /// Make the held top-up exactly `kib` (at least 1; a caller wanting none
    /// drops it): resized in place when one is held, minted otherwise.
    fn set_top_up_kib(&self, top_up: &mut Option<MemoryTopUp>, kib: u64) {
        match top_up {
            Some(held) => {
                debug_assert!(
                    Arc::ptr_eq(&held.budget, &self.budget),
                    "a grant's top-up comes from the grant's own budget"
                );
                held.resize_kib(kib);
            }
            None => *top_up = Some(self.budget.top_up_kib(kib)),
        }
    }
}

impl Drop for MemoryGrant {
    fn drop(&mut self) {
        self.budget
            .release_reserved(self.kib, self.metrics.as_ref());
    }
}

/// A forced top-up recorded as over-commit; repaid when dropped.
pub struct MemoryTopUp {
    budget: Arc<MemoryBudget>,
    kib: u64,
    metrics: Option<SharedMetrics>,
}

impl std::fmt::Debug for MemoryTopUp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryTopUp")
            .field("bytes", &self.bytes())
            .finish()
    }
}

impl MemoryTopUp {
    /// Bytes this top-up holds, in whole KiB.
    pub fn bytes(&self) -> u64 {
        self.kib.saturating_mul(KIB)
    }

    /// Make this top-up hold exactly `kib`: growing adds the difference to
    /// `overcommit` at once (never waiting), shrinking repays it and wakes the
    /// queue, and the over-commit gauge moves by the difference either way.
    fn resize_kib(&mut self, kib: u64) {
        if kib == self.kib {
            return;
        }
        let mut st = self.budget.lock();
        if kib > self.kib {
            let grown = kib - self.kib;
            st.overcommit_kib = st.overcommit_kib.saturating_add(grown);
            if let Some(m) = &self.metrics {
                self.budget.series.add_overcommit(m, gauge_bytes(grown));
            }
        } else {
            let freed = self.kib - kib;
            st.overcommit_kib = st.overcommit_kib.saturating_sub(freed);
            if let Some(m) = &self.metrics {
                self.budget.series.add_overcommit(m, -gauge_bytes(freed));
            }
            self.budget.grant_queued(&mut st);
        }
        self.kib = kib;
    }
}

impl Drop for MemoryTopUp {
    fn drop(&mut self) {
        let mut st = self.budget.lock();
        st.overcommit_kib = st.overcommit_kib.saturating_sub(self.kib);
        if let Some(m) = &self.metrics {
            self.budget.series.add_overcommit(m, -gauge_bytes(self.kib));
        }
        self.budget.grant_queued(&mut st);
    }
}

/// The memory a decoded snapshot is accounted under (Spec SDD SR-7.2 step 1):
/// the grant its read reserved and, when the read came out heavier than the
/// grant, the grant's one top-up for the difference (taken as the read
/// progressed and set to the measured weight after it). The snapshot owns it
/// (`storage::diff_read::AtomicSnapshot::memory`), so both are repaid exactly
/// when the last holder of the snapshot drops it — on a shared fenced-read
/// flight, the last attached catch-up that still holds it
/// (`storage::snapshot_flights`). Empty (the default) on a snapshot read
/// without a reservation.
#[derive(Debug, Default)]
pub struct SnapshotMemory {
    grant: Option<MemoryGrant>,
    top_up: Option<MemoryTopUp>,
}

impl SnapshotMemory {
    /// Memory held by `grant`, extended by `top_up`.
    pub fn new(grant: MemoryGrant, top_up: Option<MemoryTopUp>) -> Self {
        Self {
            grant: Some(grant),
            top_up,
        }
    }

    /// Bytes held, grant and top-up together, in whole KiB.
    pub fn bytes(&self) -> u64 {
        self.grant
            .as_ref()
            .map_or(0, MemoryGrant::bytes)
            .saturating_add(self.top_up.as_ref().map_or(0, MemoryTopUp::bytes))
    }
}

/// Compile-time check that a reservation wait and the guards can live in a
/// spawned task and be owned by a shared snapshot: the wait future must be
/// `Send` (no `std::sync::MutexGuard` held across an await), and the guards
/// `Send + Sync`.
#[allow(dead_code)]
fn _assert_send(budget: Arc<MemoryBudget>) {
    fn send<T: Send>(_: &T) {}
    fn send_sync<T: Send + Sync>() {}
    send(&async move { budget.reserve(0, None).await });
    send_sync::<MemoryGrant>();
    send_sync::<MemoryTopUp>();
    send_sync::<SnapshotMemory>();
}

/// One [`MemoryBudget::reserve`] call's wait observation and admission
/// result, recorded once when the call ends, on its budget's series: with the
/// result it finished with, or `cancelled` when its caller dropped it first.
struct BuildWaitObservation {
    started: Instant,
    metrics: Option<SharedMetrics>,
    series: &'static dyn BudgetSeries,
    recorded: bool,
}

impl BuildWaitObservation {
    fn start(metrics: Option<SharedMetrics>, series: &'static dyn BudgetSeries) -> Self {
        Self {
            started: Instant::now(),
            metrics,
            series,
            recorded: false,
        }
    }

    fn finish(&mut self, outcome: BudgetWaitOutcome) {
        if self.recorded {
            return;
        }
        self.recorded = true;
        if let Some(m) = &self.metrics {
            self.series.record_wait(m, self.started.elapsed(), outcome);
        }
    }
}

impl Drop for BuildWaitObservation {
    fn drop(&mut self) {
        self.finish(BudgetWaitOutcome::Cancelled);
    }
}

#[cfg(test)]
mod tests {
    //! Type-level tests of [`MemoryBudget`] (Spec SDD SR-7.2, SR-8.5), on the
    //! catch-up budget's series ([`MemoryBudget::new`]) but for the
    //! `end_waits` test (the retirement allowance's shutdown call, on its own
    //! series) and the last section, which pins that each instance ticks only
    //! its own series (the retirement allowance's, [`RETIREMENT_SERIES`],
    //! b.fsz MR-D4). Every
    //! hand-off here is synchronous — a freed grant is handed to the queue's
    //! head under the budget's lock before the freeing call returns — so a
    //! queued reservation is checked with a single poll rather than a timed
    //! wait. Deadline tests run on paused tokio time. The model-based property
    //! over random operation sequences is `memory_budget_model` in
    //! `tests/proptests.rs`.

    use super::*;
    use crate::observability::metrics::test_util::{render, sample_value};
    use std::future::Future;
    use std::pin::Pin;
    use std::task::Poll;
    use std::time::Duration;

    const RESERVED_GAUGE: &str = "live_edit_catchup_build_budget_reserved_bytes";
    const OVERCOMMIT_GAUGE: &str = "live_edit_catchup_build_budget_overcommit_bytes";
    const ADMISSIONS: &str = "live_edit_catchup_build_admission_total";
    const WAIT_COUNT: &str = "live_edit_catchup_build_wait_seconds_count";

    /// A request far heavier than any budget here, so it clamps to the total.
    const HUGE: u64 = 1 << 40;

    fn budget(total_kib: u64) -> Arc<MemoryBudget> {
        Arc::new(MemoryBudget::new(total_kib, Arc::new(StdMutex::new(None))))
    }

    fn budget_with_metrics(total_kib: u64) -> (Arc<MemoryBudget>, SharedMetrics) {
        let metrics: SharedMetrics =
            Arc::new(crate::observability::metrics::metrics_scrape_only("test"));
        let b = Arc::new(MemoryBudget::new(
            total_kib,
            Arc::new(StdMutex::new(Some(metrics.clone()))),
        ));
        (b, metrics)
    }

    fn gauge(metrics: &SharedMetrics, name: &str) -> i64 {
        sample_value(&render(metrics), name, &[]).expect("pre-registered at zero") as i64
    }

    fn admissions(metrics: &SharedMetrics, result: &str) -> u64 {
        sample_value(&render(metrics), ADMISSIONS, &[("result", result)])
            .expect("every result cell is pre-registered") as u64
    }

    fn waits_observed(metrics: &SharedMetrics) -> u64 {
        sample_value(&render(metrics), WAIT_COUNT, &[]).map_or(0, |v| v as u64)
    }

    type Wait<'a> = Pin<Box<dyn Future<Output = Result<MemoryGrant, CatchUpRefusal>> + Send + 'a>>;

    /// A reservation of `bytes` with no deadline, boxed so a test can drop it
    /// mid-wait.
    fn wait(b: &Arc<MemoryBudget>, bytes: u64) -> Wait<'_> {
        Box::pin(b.reserve(bytes, None))
    }

    async fn poll_once<F: Future + Unpin>(fut: &mut F) -> Poll<F::Output> {
        futures::poll!(fut)
    }

    /// Polls a reservation that must still be queued.
    async fn assert_queued(w: &mut Wait<'_>, why: &str) {
        assert!(poll_once(w).await.is_pending(), "{why}");
    }

    /// Polls a reservation that must have been granted by now (every hand-off
    /// is synchronous, so one poll picks it up).
    async fn assert_granted(w: &mut Wait<'_>, why: &str) -> MemoryGrant {
        match poll_once(w).await {
            Poll::Ready(Ok(g)) => g,
            Poll::Ready(Err(e)) => panic!("{why}: refused {e:?}"),
            Poll::Pending => panic!("{why}: still queued"),
        }
    }

    /// A reservation the test expects to fit at once, granted on its first
    /// poll. A single poll rather than an `.await`: on broken accounting the
    /// test fails here instead of queueing forever.
    async fn reserve_now(b: &Arc<MemoryBudget>, bytes: u64) -> MemoryGrant {
        assert_granted(
            &mut wait(b, bytes),
            "the request fits and nothing is queued",
        )
        .await
    }

    // ── Sizing ───────────────────────────────────────────────────────────────

    /// Every request is rounded up to whole KiB, so a grant never covers less
    /// than asked, and clamped to the total, so an oversized one is grantable
    /// at all — by `reserve`, `try_grant` and `top_up` alike.
    #[tokio::test]
    async fn memory_budget_requests_round_up_to_whole_kib_and_clamp_to_the_total() {
        assert_eq!(kib_for_bytes(0), 0);
        assert_eq!(kib_for_bytes(1), 1);
        assert_eq!(kib_for_bytes(KIB), 1);
        assert_eq!(kib_for_bytes(KIB + 1), 2);

        let b = budget(8);
        assert_eq!(b.total_bytes(), 8 * KIB);
        assert_eq!(b.clamped_kib(8 * KIB + 1), 8);
        assert_eq!(b.clamped_kib(u64::MAX), 8);

        let g = b.try_grant(KIB + 1).expect("an empty budget grants");
        assert_eq!(g.bytes(), 2 * KIB, "1 KiB + 1 B rounds up to 2 KiB");
        assert_eq!(b.reserved_bytes(), 2 * KIB);
        drop(g);

        let g = reserve_now(&b, HUGE).await;
        assert_eq!(
            g.bytes(),
            8 * KIB,
            "a request above the total is clamped to it"
        );
        let t = b.top_up(HUGE);
        assert_eq!(
            t.bytes(),
            8 * KIB,
            "a top-up above the total is clamped to it"
        );
        drop((g, t));
        assert_eq!((b.reserved_bytes(), b.overcommit_bytes()), (0, 0));
    }

    /// Configuration rejects a zero budget; one built with 0 anyway holds
    /// 1 KiB, so requests still clamp to something that bounds them.
    #[tokio::test]
    async fn memory_budget_built_with_a_zero_total_holds_one_kib() {
        let b = budget(0);
        assert_eq!(b.total_bytes(), KIB);
        let g = reserve_now(&b, HUGE).await;
        assert_eq!(g.bytes(), KIB);
        assert!(b.try_grant(1).is_none(), "the one KiB is held");
    }

    // ── FIFO admission ───────────────────────────────────────────────────────

    /// Grants come in arrival order: a request that would fit joins the back
    /// while another is queued, and the head is granted before it even when
    /// the head is the larger one.
    #[tokio::test]
    async fn memory_budget_reserve_grants_queued_requests_in_arrival_order() {
        let b = budget(10);
        let first = reserve_now(&b, 5 * KIB).await;

        let mut head = wait(&b, 10 * KIB);
        assert_queued(&mut head, "10 KiB does not fit beside 5 KiB of 10").await;
        let mut behind = wait(&b, KIB);
        assert_queued(
            &mut behind,
            "1 KiB fits beside 5 KiB, but joins the back while another request is queued",
        )
        .await;
        assert_eq!(b.waiters(), 2);

        drop(first);
        let head_grant = assert_granted(&mut head, "freeing 5 KiB lets the head in").await;
        assert_eq!(head_grant.bytes(), 10 * KIB);
        assert_queued(&mut behind, "the head now holds the whole budget").await;
        assert_eq!(b.waiters(), 1);

        drop(head_grant);
        let behind_grant = assert_granted(&mut behind, "the head's release admits the next").await;
        drop(behind_grant);
        assert_eq!((b.reserved_bytes(), b.waiters()), (0, 0));
    }

    /// Freeing memory admits every queued request that then fits, head first,
    /// not just one: a waiter held back only by the one ahead of it is not left
    /// waiting.
    #[tokio::test]
    async fn memory_budget_a_release_admits_every_queued_request_that_fits_in_order() {
        let b = budget(10);
        let held = reserve_now(&b, 10 * KIB).await;
        let mut a = wait(&b, 4 * KIB);
        let mut c = wait(&b, 4 * KIB);
        let mut d = wait(&b, 4 * KIB);
        for w in [&mut a, &mut c, &mut d] {
            assert_queued(w, "the budget is full").await;
        }

        drop(held);
        let ga = assert_granted(&mut a, "4 KiB fits an empty budget").await;
        let gc = assert_granted(&mut c, "4 + 4 KiB fit in 10").await;
        assert_queued(&mut d, "4 + 4 + 4 KiB do not").await;
        assert_eq!(b.reserved_bytes(), 8 * KIB);
        drop((ga, gc));
        drop(assert_granted(&mut d, "the releases admit the last").await);
        assert_eq!(b.reserved_bytes(), 0);
    }

    // ── Clamp runs alone ─────────────────────────────────────────────────────

    /// A request clamped to the whole total is granted only when nothing is
    /// held (`reserved + overcommit == 0`), and while it is held nothing else
    /// is granted.
    #[tokio::test]
    async fn memory_budget_a_clamped_request_is_granted_only_when_nothing_is_held() {
        let b = budget(10);
        let small = b.try_grant(KIB).expect("an empty budget grants");
        let mut clamped = wait(&b, HUGE);
        assert_queued(&mut clamped, "a 1 KiB grant is held").await;

        drop(small);
        let alone = assert_granted(&mut clamped, "nothing is held any more").await;
        assert_eq!(alone.bytes(), 10 * KIB);
        assert!(b.try_grant(KIB).is_none(), "nothing runs beside it");
        let mut beside = wait(&b, KIB);
        assert_queued(&mut beside, "nothing runs beside it").await;

        drop(alone);
        drop(assert_granted(&mut beside, "its release admits the next").await);
    }

    /// An outstanding top-up holds a clamped request back on its own, even
    /// with no grant held, and repaying it admits the request.
    #[tokio::test]
    async fn memory_budget_a_clamped_request_waits_while_only_a_top_up_is_outstanding() {
        let b = budget(10);
        let top = b.top_up(KIB);
        assert_eq!((b.reserved_bytes(), b.overcommit_bytes()), (0, KIB));

        let mut clamped = wait(&b, HUGE);
        assert_queued(&mut clamped, "a top-up counts against admission").await;

        drop(top);
        let g = assert_granted(&mut clamped, "repaying the top-up wakes the queue").await;
        assert_eq!(g.bytes(), 10 * KIB);
    }

    // ── Non-blocking grant ───────────────────────────────────────────────────

    /// `try_grant` grants only when the budget is open, no request is queued
    /// and the request fits, and never joins the queue.
    #[tokio::test]
    async fn memory_budget_try_grant_succeeds_only_when_open_unqueued_and_fitting() {
        let b = budget(10);
        let held = b.try_grant(6 * KIB).expect("fits an empty budget");
        assert!(b.try_grant(5 * KIB).is_none(), "6 + 5 KiB do not fit in 10");
        assert_eq!(b.waiters(), 0, "a declined grant never queues");

        let mut queued = wait(&b, 10 * KIB);
        assert_queued(&mut queued, "10 KiB does not fit beside 6").await;
        assert!(
            b.try_grant(KIB).is_none(),
            "1 KiB fits beside 6, but a request is queued"
        );

        drop(queued);
        let fits = b.try_grant(KIB).expect("queue empty and 6 + 1 KiB fit");
        assert_eq!(b.reserved_bytes(), 7 * KIB);

        b.close();
        assert!(b.try_grant(KIB).is_none(), "a closed budget grants nothing");
        drop((held, fits));
        assert_eq!(b.reserved_bytes(), 0);
    }

    // ── Forced top-up ────────────────────────────────────────────────────────

    /// A top-up is taken at once past the total, counts toward every later
    /// admission until its guard drops, and still succeeds on a closed budget.
    #[tokio::test]
    async fn memory_budget_top_up_never_waits_and_counts_as_overcommit_until_dropped() {
        let b = budget(4);
        let full = reserve_now(&b, 4 * KIB).await;
        let top = b.top_up(3 * KIB);
        assert_eq!(top.bytes(), 3 * KIB);
        assert_eq!(
            (b.reserved_bytes(), b.overcommit_bytes()),
            (4 * KIB, 3 * KIB),
            "a top-up is taken past a full budget"
        );

        drop(full);
        let mut after = wait(&b, 2 * KIB);
        assert_queued(&mut after, "3 KiB of over-commit + 2 KiB do not fit in 4").await;
        drop(top);
        drop(assert_granted(&mut after, "repaying the top-up admits the waiter").await);

        b.close();
        let late = b.top_up(KIB);
        assert_eq!(
            b.overcommit_bytes(),
            KIB,
            "a top-up accounts for resident memory, so a closed budget takes it"
        );
        drop(late);
        assert_eq!((b.reserved_bytes(), b.overcommit_bytes()), (0, 0));
    }

    // ── Deadlines ────────────────────────────────────────────────────────────

    /// A queued reservation is refused `TimedOut` at its deadline, not before,
    /// leaves the queue, and records one wait and one `refused_timeout`.
    #[tokio::test(start_paused = true)]
    async fn memory_budget_reserve_past_its_deadline_is_refused_timed_out_and_leaves_the_queue() {
        const BUDGET: Duration = Duration::from_millis(300);
        let (b, metrics) = budget_with_metrics(4);
        let held = reserve_now(&b, 4 * KIB).await;

        let started = tokio::time::Instant::now();
        let refused = b.reserve(KIB, Some(started + BUDGET)).await;
        assert!(matches!(refused, Err(CatchUpRefusal::TimedOut)));
        assert!(
            tokio::time::Instant::now() >= started + BUDGET,
            "the refusal comes at the deadline"
        );
        assert_eq!(b.waiters(), 0, "the timed-out wait left the queue");
        assert_eq!(b.reserved_bytes(), 4 * KIB, "and took nothing");
        assert_eq!(
            admissions(&metrics, CATCHUP_BUILD_ADMISSION_RESULT_REFUSED_TIMEOUT),
            1
        );
        assert_eq!(
            waits_observed(&metrics),
            2,
            "the holder's and the refused wait"
        );

        drop(held);
        assert!(
            b.try_grant(4 * KIB).is_some(),
            "no queue position was left behind"
        );
    }

    /// A deadline already passed takes free memory at once, and with none free
    /// is refused on its first poll without joining the queue.
    #[tokio::test]
    async fn memory_budget_reserve_with_a_passed_deadline_takes_free_memory_and_never_queues() {
        let b = budget(4);
        let passed = tokio::time::Instant::now();
        let held = b
            .reserve(4 * KIB, Some(passed))
            .await
            .expect("free memory is taken with no time left");

        let mut refused = Box::pin(b.reserve(KIB, Some(passed)));
        match poll_once(&mut refused).await {
            Poll::Ready(Err(CatchUpRefusal::TimedOut)) => {}
            Poll::Ready(other) => panic!("expected TimedOut, got {:?}", other.map(|_| ())),
            Poll::Pending => panic!("a passed deadline must be refused at once, never queued"),
        }
        assert_eq!(b.waiters(), 0);
        drop(held);
    }

    /// An oversized (clamped) request at the head of the queue neither
    /// deadlocks the queue nor outlives its deadline: it is refused there, and
    /// its departure admits the request it was holding back.
    #[tokio::test(start_paused = true)]
    async fn memory_budget_an_oversized_head_timing_out_admits_the_request_behind_it() {
        let b = budget(10);
        let held = reserve_now(&b, 5 * KIB).await;
        let deadline = tokio::time::Instant::now() + Duration::from_millis(300);
        let mut oversized = Box::pin(b.reserve(HUGE, Some(deadline)));
        assert!(poll_once(&mut oversized).await.is_pending());
        let mut behind = wait(&b, KIB);
        assert_queued(&mut behind, "1 KiB queues behind the head").await;

        assert!(matches!(oversized.await, Err(CatchUpRefusal::TimedOut)));
        let g = assert_granted(&mut behind, "the head's timeout admits the next").await;
        assert_eq!(b.reserved_bytes(), 6 * KIB);
        drop((held, g));
    }

    /// A reservation with a deadline that queues and is granted before the
    /// deadline (the shape of a caller with a capacity budget of its own; a
    /// shared fenced-read flight reserves with none) returns its grant,
    /// counted in `reserved`, recorded `waited`;
    /// dropping the grant repays it. A release that lands on the deadline
    /// instant either keeps the hand-off (`Ok`, counted) or gives it back
    /// (`TimedOut`, nothing reserved): never memory reserved with no owner.
    #[tokio::test(start_paused = true)]
    async fn memory_budget_a_deadline_bearing_wait_granted_in_time_counts_as_waited() {
        const BUDGET: Duration = Duration::from_millis(300);
        let (b, metrics) = budget_with_metrics(4);

        let held = reserve_now(&b, 4 * KIB).await;
        let deadline = tokio::time::Instant::now() + BUDGET;
        let mut in_time = Box::pin(b.reserve(KIB, Some(deadline)));
        assert!(
            poll_once(&mut in_time).await.is_pending(),
            "the budget is full"
        );
        tokio::time::advance(BUDGET / 3).await;
        drop(held);
        let g = match poll_once(&mut in_time).await {
            Poll::Ready(Ok(g)) => g,
            other => panic!(
                "a release before the deadline grants the wait, got {:?}",
                other.map(|r| r.map(|_| ()))
            ),
        };
        assert_eq!(g.bytes(), KIB);
        assert_eq!(b.reserved_bytes(), KIB, "the grant is counted");
        assert_eq!(
            admissions(&metrics, CATCHUP_BUILD_ADMISSION_RESULT_WAITED),
            1
        );
        assert_eq!(
            admissions(&metrics, CATCHUP_BUILD_ADMISSION_RESULT_REFUSED_TIMEOUT),
            0
        );
        drop(g);
        assert_eq!(b.reserved_bytes(), 0, "dropping the grant repays it");

        let held = reserve_now(&b, 4 * KIB).await;
        let deadline = tokio::time::Instant::now() + BUDGET;
        let mut last_moment = Box::pin(b.reserve(KIB, Some(deadline)));
        assert!(
            poll_once(&mut last_moment).await.is_pending(),
            "the budget is full"
        );
        tokio::time::advance(BUDGET).await;
        drop(held);
        match poll_once(&mut last_moment).await {
            Poll::Ready(Ok(g)) => {
                assert_eq!(g.bytes(), KIB);
                assert_eq!(b.reserved_bytes(), KIB, "a kept hand-off is counted");
                drop(g);
            }
            Poll::Ready(Err(CatchUpRefusal::TimedOut)) => {}
            other => panic!(
                "a release on the deadline instant either grants or times out, got {:?}",
                other.map(|r| r.map(|_| ()))
            ),
        }
        assert_eq!(
            (b.reserved_bytes(), b.waiters()),
            (0, 0),
            "nothing is left reserved without an owner"
        );
    }

    /// Closing the budget while a deadline-bearing reservation is queued ends
    /// it `Closed` (not `TimedOut`), reserving nothing for it.
    #[tokio::test(start_paused = true)]
    async fn memory_budget_close_ends_a_deadline_bearing_wait_as_closed() {
        let (b, metrics) = budget_with_metrics(4);
        let held = reserve_now(&b, 4 * KIB).await;
        let deadline = tokio::time::Instant::now() + Duration::from_millis(300);
        let mut parked = Box::pin(b.reserve(KIB, Some(deadline)));
        assert!(
            poll_once(&mut parked).await.is_pending(),
            "the budget is full"
        );

        b.close();
        assert!(matches!(
            poll_once(&mut parked).await,
            Poll::Ready(Err(CatchUpRefusal::Closed))
        ));
        assert_eq!(
            (b.reserved_bytes(), b.waiters()),
            (4 * KIB, 0),
            "only the holder's grant is reserved"
        );
        assert_eq!(
            admissions(&metrics, CATCHUP_BUILD_ADMISSION_RESULT_CANCELLED),
            1
        );
        drop(held);
        assert_eq!(b.reserved_bytes(), 0);
    }

    /// A queued reservation as `reserve` enqueues it, for driving
    /// [`QueuedReservation::settle_at_deadline`] directly.
    fn enqueue(b: &Arc<MemoryBudget>, kib: u64) -> QueuedReservation<'_> {
        let mut st = b.lock();
        let id = st.next_id;
        st.next_id += 1;
        st.queue.push_back(Waiter {
            id,
            kib,
            notify: Arc::new(Notify::new()),
            metrics: None,
        });
        QueuedReservation {
            budget: b,
            id,
            kib,
            metrics: None,
            settled: false,
        }
    }

    /// The race branches of the deadline settle, driven directly: through
    /// `reserve` they are reached only when another thread hands the grant
    /// out, or closes the budget, between the deadline firing and the settle
    /// taking the lock. A single-threaded test never gets there, because
    /// tokio's `Timeout` polls the wait before its timer and `Notify` takes no
    /// coop budget, so a hand-off or close is always seen by the wait first.
    ///
    /// A grant handed at the last moment is kept for the caller (who wraps it
    /// in a `MemoryGrant`, as `reserve` does) rather than given back while
    /// still returned; a budget closed at the last moment ends the wait
    /// `Closed`, reserving nothing for it.
    #[tokio::test]
    async fn memory_budget_settle_at_deadline_keeps_a_last_moment_hand_off_and_reports_close() {
        let b = budget(4);
        let held = reserve_now(&b, 4 * KIB).await;
        let mut handed = enqueue(&b, 1);
        drop(held);
        assert_eq!(
            b.reserved_bytes(),
            KIB,
            "the release handed 1 KiB to the queue"
        );
        assert!(matches!(handed.settle_at_deadline(), Ok(())));
        drop(handed);
        assert_eq!(
            b.reserved_bytes(),
            KIB,
            "a kept hand-off stays reserved for the caller's grant"
        );
        drop(MemoryGrant::new(&b, 1, None));
        assert_eq!(b.reserved_bytes(), 0, "and the caller's grant repays it");

        let held = reserve_now(&b, 4 * KIB).await;
        let mut closed = enqueue(&b, 1);
        b.close();
        assert!(matches!(
            closed.settle_at_deadline(),
            Err(CatchUpRefusal::Closed)
        ));
        drop(closed);
        assert_eq!(
            (b.reserved_bytes(), b.waiters()),
            (4 * KIB, 0),
            "only the holder's grant is reserved"
        );
        drop(held);
        assert_eq!(b.reserved_bytes(), 0);
    }

    // ── Dropped and closed waits ─────────────────────────────────────────────

    /// A dropped wait leaves the queue without taking memory, and wakes the
    /// request it was holding back.
    #[tokio::test]
    async fn memory_budget_a_dropped_wait_leaves_the_queue_and_wakes_the_one_behind_it() {
        let b = budget(10);
        let held = reserve_now(&b, 5 * KIB).await;
        let mut head = wait(&b, 10 * KIB);
        assert_queued(&mut head, "10 KiB does not fit beside 5").await;
        let mut behind = wait(&b, KIB);
        assert_queued(&mut behind, "1 KiB queues behind the head").await;

        drop(head);
        assert_eq!(b.reserved_bytes(), 6 * KIB, "the dropped wait took nothing");
        let g = assert_granted(&mut behind, "dropping the head admits the next").await;
        assert_eq!(b.waiters(), 0);
        drop((held, g));
        assert_eq!(b.reserved_bytes(), 0);
    }

    /// A grant handed to a queued wait that is dropped before it picks the
    /// grant up is given back, and goes to the next waiter.
    #[tokio::test]
    async fn memory_budget_a_grant_handed_to_a_dropped_wait_is_given_back() {
        let b = budget(10);
        let held = reserve_now(&b, 10 * KIB).await;
        let mut handed = wait(&b, 4 * KIB);
        assert_queued(&mut handed, "the budget is full").await;
        let mut next = wait(&b, 10 * KIB);
        assert_queued(&mut next, "the budget is full").await;

        drop(held);
        assert_eq!(
            (b.reserved_bytes(), b.waiters()),
            (4 * KIB, 1),
            "the release hands 4 KiB to the head before it is polled"
        );
        drop(handed);
        let g = assert_granted(&mut next, "the unclaimed 4 KiB came back").await;
        assert_eq!(b.reserved_bytes(), 10 * KIB);
        drop(g);
        assert_eq!(b.reserved_bytes(), 0);
    }

    /// Closing the budget (shutdown) ends every queued wait `Closed`, refuses
    /// every later reservation even with memory free, and leaves held grants
    /// valid until they drop.
    #[tokio::test]
    async fn memory_budget_close_ends_queued_waits_and_later_reservations_as_closed() {
        let b = budget(4);
        let held = reserve_now(&b, 4 * KIB).await;
        let mut parked = wait(&b, KIB);
        assert_queued(&mut parked, "the budget is full").await;

        b.close();
        assert!(b.is_closed());
        assert!(matches!(
            poll_once(&mut parked).await,
            Poll::Ready(Err(CatchUpRefusal::Closed))
        ));
        assert_eq!(b.waiters(), 0);
        assert_eq!(b.reserved_bytes(), 4 * KIB, "a held grant stays valid");

        drop(held);
        assert_eq!(b.reserved_bytes(), 0);
        assert!(
            matches!(
                poll_once(&mut wait(&b, KIB)).await,
                Poll::Ready(Err(CatchUpRefusal::Closed))
            ),
            "a closed budget refuses even with memory free"
        );
    }

    /// `end_waits` — the retirement allowance's shutdown call (b.fsz MR-D4) —
    /// ends every queued wait `Closed`, a deadline-bearing one included
    /// (`Closed`, not `TimedOut`, even once its deadline has passed), and from
    /// then on lets nothing wait: a request that fits is still granted at once,
    /// one that does not is refused `Closed` on its first poll and never
    /// queued. The budget is not closed, so `try_grant` and `top_up` are
    /// unchanged, and a `close()` afterwards still refuses everything. Fails on
    /// `close()` semantics (the request that fits would be refused) and on an
    /// end that only drains the queue it finds (the request that does not fit
    /// would queue). On the retirement series each refusal records one
    /// `cancelled` wait.
    #[tokio::test(start_paused = true)]
    async fn memory_budget_end_waits_ends_queued_waits_grants_what_fits_and_refuses_what_would_wait(
    ) {
        const BUDGET: Duration = Duration::from_millis(300);
        let (b, metrics) = retirement_budget_with_metrics(8);
        let held = reserve_now(&b, 6 * KIB).await;
        let mut queued = Box::pin(b.reserve(4 * KIB, Some(tokio::time::Instant::now() + BUDGET)));
        assert!(
            poll_once(&mut queued).await.is_pending(),
            "4 KiB does not fit beside 6"
        );
        assert_eq!(b.waiters(), 1);

        b.end_waits();
        assert_eq!(b.waiters(), 0, "the queued wait left the queue");
        assert!(b.waits_ended());
        assert!(!b.is_closed(), "ending the waits does not close the budget");
        tokio::time::advance(BUDGET * 2).await;
        match poll_once(&mut queued).await {
            Poll::Ready(Err(CatchUpRefusal::Closed)) => {}
            other => panic!(
                "a queued deadline-bearing wait ends Closed, not TimedOut, got {:?}",
                other.map(|r| r.map(|_| ()))
            ),
        }
        assert_eq!(b.reserved_bytes(), 6 * KIB, "it took nothing");

        let fits = reserve_now(&b, 2 * KIB).await;
        assert_eq!(
            b.reserved_bytes(),
            8 * KIB,
            "a request that fits is granted at once"
        );

        let mut would_wait = wait(&b, KIB);
        assert!(
            matches!(
                poll_once(&mut would_wait).await,
                Poll::Ready(Err(CatchUpRefusal::Closed))
            ),
            "a request that would wait is refused on its first poll"
        );
        assert_eq!(b.waiters(), 0, "and never queued");
        assert_eq!(
            retirement_waits(&metrics),
            [2, 0, 2],
            "two immediate grants; the ended wait and the refused request are cancelled"
        );

        drop(fits);
        let try_granted = b
            .try_grant(2 * KIB)
            .expect("try_grant is unchanged: 2 KiB fits and nothing is queued");
        let top = b.top_up(4 * KIB);
        assert_eq!(top.bytes(), 4 * KIB, "a top-up is unchanged");
        drop((try_granted, top));

        b.close();
        assert!(b.is_closed());
        assert!(
            b.try_grant(KIB).is_none(),
            "a close afterwards refuses a non-blocking grant"
        );
        assert!(
            matches!(
                poll_once(&mut wait(&b, KIB)).await,
                Poll::Ready(Err(CatchUpRefusal::Closed))
            ),
            "and a reservation that would fit"
        );
        drop(held);
        assert_eq!(
            (b.reserved_bytes(), b.overcommit_bytes(), b.waiters()),
            (0, 0, 0)
        );
    }

    // ── Grant adjustment ─────────────────────────────────────────────────────

    /// `shrink_to` gives back everything above its target (rounded up to KiB)
    /// and wakes the queue; a target at or above the grant changes nothing.
    #[tokio::test]
    async fn memory_grant_shrink_to_gives_back_the_excess_and_wakes_the_queue() {
        let b = budget(10);
        let mut g = reserve_now(&b, 8 * KIB).await;
        let mut waiter = wait(&b, 5 * KIB);
        assert_queued(&mut waiter, "5 KiB does not fit beside 8").await;

        g.shrink_to(3 * KIB - 100);
        assert_eq!(g.bytes(), 3 * KIB, "the target rounds up to whole KiB");
        let w = assert_granted(&mut waiter, "3 + 5 KiB now fit").await;

        g.shrink_to(9 * KIB);
        assert_eq!(g.bytes(), 3 * KIB, "a grant never grows by shrinking");
        assert_eq!(b.reserved_bytes(), 8 * KIB);
        drop((g, w));
        assert_eq!(b.reserved_bytes(), 0);
    }

    /// `adjust_with_top_up` with no top-up held: a measure at or below the
    /// grant shrinks it and takes no top-up; a measure above it takes a
    /// never-waiting top-up for the difference, leaving the grant as it was; a
    /// measure above the total tops up only to the total.
    #[tokio::test]
    async fn memory_grant_adjust_with_no_top_up_held_shrinks_or_tops_up_for_the_difference() {
        let b = budget(10);

        let mut lower = reserve_now(&b, 8 * KIB).await;
        let mut top = None;
        lower.adjust_with_top_up(&mut top, 2 * KIB);
        assert!(top.is_none());
        assert_eq!(lower.bytes(), 2 * KIB);
        assert_eq!((b.reserved_bytes(), b.overcommit_bytes()), (2 * KIB, 0));
        drop(lower);

        let mut equal = reserve_now(&b, 4 * KIB).await;
        equal.adjust_with_top_up(&mut top, 4 * KIB);
        assert!(top.is_none());
        assert_eq!(equal.bytes(), 4 * KIB);
        drop(equal);

        let mut higher = reserve_now(&b, 4 * KIB).await;
        higher.adjust_with_top_up(&mut top, 7 * KIB);
        let held = top.take().expect("a heavier measure tops up");
        assert_eq!(held.bytes(), 3 * KIB, "the top-up is the difference");
        assert_eq!(higher.bytes(), 4 * KIB, "the grant itself is unchanged");
        assert_eq!(
            (b.reserved_bytes(), b.overcommit_bytes()),
            (4 * KIB, 3 * KIB)
        );
        drop((higher, held));

        let mut above = reserve_now(&b, 4 * KIB).await;
        above.adjust_with_top_up(&mut top, HUGE);
        let held = top.take().expect("a measure above the total tops up");
        assert_eq!(
            above.bytes() + held.bytes(),
            10 * KIB,
            "grant and top-up together never exceed the total"
        );
        drop((above, held));
        assert_eq!((b.reserved_bytes(), b.overcommit_bytes()), (0, 0));
    }

    // ── A grant's own top-up (progress top-ups and the adjust) ──────────────

    /// R2 — a measure at or below the grant, with a progress top-up held:
    /// the grant shrinks to it and the held top-up is dropped (repaid to
    /// `overcommit`, the queue woken), whatever the top-up had grown to.
    #[tokio::test]
    async fn memory_grant_adjust_below_the_grant_drops_the_held_top_up_and_shrinks_the_grant() {
        let b = budget(16);
        let mut grant = reserve_now(&b, 6 * KIB).await;
        let mut top = None;
        grant.top_up_to(&mut top, 11 * KIB);
        assert_eq!(top.as_ref().map(MemoryTopUp::bytes), Some(5 * KIB));
        let mut behind = wait(&b, 8 * KIB);
        assert_queued(
            &mut behind,
            "6 KiB reserved + 5 KiB over-committed + 8 > 16",
        )
        .await;

        grant.adjust_with_top_up(&mut top, 4 * KIB - 10);
        assert!(top.is_none(), "a measure under the grant drops the top-up");
        assert_eq!(grant.bytes(), 4 * KIB, "the grant shrinks to the measure");
        let g = assert_granted(&mut behind, "4 KiB + 8 now fit, the top-up repaid").await;
        assert_eq!(
            (b.reserved_bytes(), b.overcommit_bytes()),
            (12 * KIB, 0),
            "the grant's 4 KiB and the waiter's 8, nothing over-committed"
        );
        drop((grant, g));
        assert_eq!((b.reserved_bytes(), b.overcommit_bytes()), (0, 0));
    }

    /// R2 — a measure above the grant, with a progress top-up held: the grant
    /// is kept and the one held guard becomes exactly the difference, grown or
    /// shrunk in place; never a second guard, so the over-commit is the
    /// difference once, not the progress top-up plus the difference.
    #[tokio::test]
    async fn memory_grant_adjust_above_the_grant_resizes_the_one_held_top_up_to_the_difference() {
        let b = budget(64);
        let mut grant = reserve_now(&b, 8 * KIB).await;
        let mut top = None;
        grant.top_up_to(&mut top, 20 * KIB);
        assert_eq!(b.overcommit_bytes(), 12 * KIB);

        // Grown: past what progress took.
        grant.adjust_with_top_up(&mut top, 30 * KIB);
        assert_eq!(grant.bytes(), 8 * KIB, "the grant is kept");
        assert_eq!(
            top.as_ref().map(MemoryTopUp::bytes),
            Some(22 * KIB),
            "one guard of exactly measure − grant"
        );
        assert_eq!(
            (b.reserved_bytes(), b.overcommit_bytes()),
            (8 * KIB, 22 * KIB),
            "no double counting: 22 KiB over-committed, not 12 + 22"
        );

        // Shrunk: below what is held, still above the grant; wakes the queue.
        let mut behind = wait(&b, 40 * KIB);
        assert_queued(&mut behind, "8 + 22 + 40 > 64").await;
        grant.adjust_with_top_up(&mut top, 9 * KIB + 1);
        assert_eq!(
            top.as_ref().map(MemoryTopUp::bytes),
            Some(2 * KIB),
            "10 KiB (9 KiB + 1 B rounded up) − 8 KiB, in place"
        );
        assert_eq!(b.overcommit_bytes(), 2 * KIB);
        let g = assert_granted(&mut behind, "8 + 2 + 40 fit in 64").await;

        // Again at the same measure: nothing moves.
        grant.adjust_with_top_up(&mut top, 10 * KIB);
        assert_eq!(
            (b.reserved_bytes(), b.overcommit_bytes()),
            (48 * KIB, 2 * KIB)
        );
        drop((grant, top, g));
        assert_eq!((b.reserved_bytes(), b.overcommit_bytes()), (0, 0));
    }

    /// R2 — the over-commit gauge follows a held top-up as it grows from
    /// progress, is resized by the adjust and is dropped, and returns to its
    /// value before the read (another holder's top-up still counted).
    #[tokio::test]
    async fn memory_grant_held_top_up_moves_the_overcommit_gauge_and_returns_it_on_drop() {
        let (b, metrics) = budget_with_metrics(64);
        let other = b.top_up(3 * KIB);
        let before = gauge(&metrics, OVERCOMMIT_GAUGE);
        assert_eq!(before, 3 * KIB as i64);

        let mut grant = reserve_now(&b, 4 * KIB).await;
        let mut top = None;
        grant.top_up_to(&mut top, 10 * KIB);
        assert_eq!(gauge(&metrics, OVERCOMMIT_GAUGE), before + 6 * KIB as i64);
        grant.top_up_to(&mut top, 25 * KIB);
        assert_eq!(gauge(&metrics, OVERCOMMIT_GAUGE), before + 21 * KIB as i64);
        grant.adjust_with_top_up(&mut top, 12 * KIB);
        assert_eq!(
            gauge(&metrics, OVERCOMMIT_GAUGE),
            before + 8 * KIB as i64,
            "the adjust shrinks the gauge with the guard"
        );
        assert_eq!(b.overcommit_bytes(), 11 * KIB);

        drop(top);
        assert_eq!(gauge(&metrics, OVERCOMMIT_GAUGE), before, "back on drop");
        assert_eq!(b.overcommit_bytes(), 3 * KIB);

        let mut top = None;
        grant.top_up_to(&mut top, 30 * KIB);
        grant.adjust_with_top_up(&mut top, KIB);
        assert!(top.is_none());
        assert_eq!(
            gauge(&metrics, OVERCOMMIT_GAUGE),
            before,
            "an adjust under the grant drops it, gauge included"
        );
        drop((grant, other));
        assert_eq!(gauge(&metrics, OVERCOMMIT_GAUGE), 0);
        assert_eq!(gauge(&metrics, RESERVED_GAUGE), 0);
    }

    /// `top_up_to` only ever grows: a target at or below what the grant and
    /// its held top-up cover takes nothing, and a target above takes only the
    /// shortfall, into the one held guard.
    #[tokio::test]
    async fn memory_grant_top_up_to_takes_only_the_shortfall_and_never_shrinks() {
        let b = budget(64);
        let grant = reserve_now(&b, 8 * KIB).await;
        let mut top = None;
        grant.top_up_to(&mut top, 8 * KIB);
        assert!(top.is_none(), "the grant covers it");
        grant.top_up_to(&mut top, 5 * KIB);
        assert!(top.is_none(), "the grant covers it");
        grant.top_up_to(&mut top, 8 * KIB + 1);
        assert_eq!(
            top.as_ref().map(MemoryTopUp::bytes),
            Some(KIB),
            "one byte past the grant is one KiB of top-up"
        );
        grant.top_up_to(&mut top, 20 * KIB);
        grant.top_up_to(&mut top, 12 * KIB);
        assert_eq!(
            top.as_ref().map(MemoryTopUp::bytes),
            Some(12 * KIB),
            "grown to 20 − 8, not shrunk by a lower target"
        );
        assert_eq!(
            (b.reserved_bytes(), b.overcommit_bytes()),
            (8 * KIB, 12 * KIB),
            "one guard's worth of over-commit"
        );
        drop((grant, top));
        assert_eq!((b.reserved_bytes(), b.overcommit_bytes()), (0, 0));
    }

    /// R3a — progress top-ups never wait: with the budget fully reserved by
    /// other grants and a reservation queued behind them, `top_up_to` takes
    /// its shortfall at once, past the total, without joining or disturbing
    /// the queue; on a closed budget too.
    #[tokio::test]
    async fn memory_grant_top_up_to_never_waits_on_a_fully_reserved_budget() {
        let b = budget(16);
        let grant = reserve_now(&b, 4 * KIB).await;
        let others = reserve_now(&b, 12 * KIB).await;
        let mut queued = wait(&b, KIB);
        assert_queued(&mut queued, "the budget is fully reserved").await;

        let mut top = None;
        grant.top_up_to(&mut top, 10 * KIB);
        assert_eq!(top.as_ref().map(MemoryTopUp::bytes), Some(6 * KIB));
        assert_eq!(
            (b.reserved_bytes(), b.overcommit_bytes(), b.waiters()),
            (16 * KIB, 6 * KIB, 1),
            "taken past a full budget at once; the queued reservation still queued"
        );
        assert_queued(&mut queued, "over-commit holds it back").await;

        b.close();
        grant.top_up_to(&mut top, 14 * KIB);
        assert_eq!(
            b.overcommit_bytes(),
            10 * KIB,
            "a closed budget still takes it: the memory is resident"
        );
        drop((grant, others, top, queued));
        assert_eq!((b.reserved_bytes(), b.overcommit_bytes()), (0, 0));
    }

    /// R3b — the cumulative clamp: however many progress reports push it, a
    /// grant and its held top-up together never exceed `T`, and reach it
    /// exactly once the target passes it; the adjust after the read clamps
    /// the same way.
    #[tokio::test]
    async fn memory_grant_top_up_to_keeps_grant_and_top_up_within_the_total() {
        let b = budget(32);
        let mut grant = reserve_now(&b, 5 * KIB).await;
        let mut top = None;
        for step in 1..=40u64 {
            grant.top_up_to(&mut top, step * 3 * KIB);
            let held = grant.bytes() + top.as_ref().map_or(0, MemoryTopUp::bytes);
            assert!(
                held <= 32 * KIB,
                "step {step}: grant + top-up = {held} B past T"
            );
            assert_eq!(
                held,
                (step * 3 * KIB).clamp(5 * KIB, 32 * KIB),
                "step {step}"
            );
        }
        assert_eq!(
            grant.bytes() + top.as_ref().map_or(0, MemoryTopUp::bytes),
            32 * KIB,
            "the target passed T: grant + top-up is exactly T"
        );
        grant.top_up_to(&mut top, HUGE);
        grant.adjust_with_top_up(&mut top, HUGE);
        assert_eq!(
            (b.reserved_bytes(), b.overcommit_bytes()),
            (5 * KIB, 27 * KIB),
            "still exactly T after the adjust to a weight above it"
        );
        drop((grant, top));
        assert_eq!((b.reserved_bytes(), b.overcommit_bytes()), (0, 0));
    }

    // ── Guards ───────────────────────────────────────────────────────────────

    /// A grant and a top-up held by a task that is aborted are repaid.
    #[tokio::test]
    async fn memory_budget_guards_held_by_an_aborted_task_are_repaid() {
        let b = budget(10);
        let holder = {
            let b = Arc::clone(&b);
            tokio::spawn(async move {
                let _grant = reserve_now(&b, 4 * KIB).await;
                let _top = b.top_up(2 * KIB);
                std::future::pending::<()>().await;
            })
        };
        for _ in 0..10_000 {
            if b.overcommit_bytes() != 0 {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(b.overcommit_bytes(), 2 * KIB, "the holder took its guards");
        assert_eq!(b.reserved_bytes(), 4 * KIB);
        holder.abort();
        assert!(holder.await.unwrap_err().is_cancelled());
        assert_eq!((b.reserved_bytes(), b.overcommit_bytes()), (0, 0));
    }

    /// `SnapshotMemory` owns a grant and its top-up (what a shared fenced-read
    /// flight hands its snapshot): it reports both together, and dropping it
    /// repays both — the grant to `reserved` (waking the queue), the top-up to
    /// `overcommit`. The default holds nothing.
    #[tokio::test]
    async fn snapshot_memory_reports_and_repays_its_grant_and_top_up_together() {
        let b = budget(8);
        assert_eq!(SnapshotMemory::default().bytes(), 0);

        let mut grant = reserve_now(&b, 6 * KIB).await;
        let mut top = None;
        grant.adjust_with_top_up(&mut top, 9 * KIB);
        assert!(top.is_some(), "a heavier measure tops up");
        let memory = SnapshotMemory::new(grant, top);
        assert_eq!(memory.bytes(), 8 * KIB, "the grant and the top-up together");
        assert_eq!(
            (b.reserved_bytes(), b.overcommit_bytes()),
            (6 * KIB, 2 * KIB)
        );
        let mut behind = wait(&b, 2 * KIB);
        assert_queued(&mut behind, "nothing fits beside the snapshot's memory").await;

        drop(memory);
        assert_eq!(
            b.overcommit_bytes(),
            0,
            "the top-up is repaid with the snapshot"
        );
        let g = assert_granted(&mut behind, "the repaid grant wakes the queue").await;
        assert_eq!(b.reserved_bytes(), 2 * KIB);
        drop(g);

        let memory = SnapshotMemory::new(reserve_now(&b, 3 * KIB).await, None);
        assert_eq!(memory.bytes(), 3 * KIB, "a grant with no top-up");
        drop(memory);
        assert_eq!((b.reserved_bytes(), b.overcommit_bytes()), (0, 0));
    }

    // ── Metrics (Spec SDD SR-8.5) ────────────────────────────────────────────

    /// The reserved and over-commit gauges move with every grant, shrink,
    /// top-up and guard drop — a grant handed to a dropped wait included — and
    /// read zero once everything is released.
    #[tokio::test]
    async fn memory_budget_gauges_follow_every_grant_and_top_up_back_to_zero() {
        let (b, metrics) = budget_with_metrics(8);
        assert_eq!(gauge(&metrics, RESERVED_GAUGE), 0);
        assert_eq!(gauge(&metrics, OVERCOMMIT_GAUGE), 0);

        let mut reserved = reserve_now(&b, 3 * KIB).await;
        let granted = b.try_grant(KIB).unwrap();
        let top = b.top_up(2 * KIB);
        assert_eq!(gauge(&metrics, RESERVED_GAUGE), 4 * KIB as i64);
        assert_eq!(gauge(&metrics, OVERCOMMIT_GAUGE), 2 * KIB as i64);

        reserved.shrink_to(KIB);
        assert_eq!(gauge(&metrics, RESERVED_GAUGE), 2 * KIB as i64);

        let mut handed = wait(&b, 8 * KIB);
        assert_queued(&mut handed, "2 KiB held and 2 KiB over-committed").await;
        drop((reserved, granted));
        assert_eq!(gauge(&metrics, RESERVED_GAUGE), 0);
        drop(top);
        assert_eq!(
            gauge(&metrics, RESERVED_GAUGE),
            8 * KIB as i64,
            "the hand-off to the queued wait is counted at once"
        );
        assert_eq!(gauge(&metrics, OVERCOMMIT_GAUGE), 0);
        drop(handed);
        assert_eq!(
            gauge(&metrics, RESERVED_GAUGE),
            0,
            "an unclaimed hand-off is given back on the gauge too"
        );

        let mut adjusted = reserve_now(&b, 2 * KIB).await;
        let mut top = None;
        adjusted.adjust_with_top_up(&mut top, 5 * KIB);
        assert_eq!(gauge(&metrics, OVERCOMMIT_GAUGE), 3 * KIB as i64);
        drop((adjusted, top));
        assert_eq!(gauge(&metrics, RESERVED_GAUGE), 0);
        assert_eq!(gauge(&metrics, OVERCOMMIT_GAUGE), 0);
    }

    /// Every `reserve` call records one wait observation and one admission
    /// result, whatever its outcome; a successful `try_grant` counts
    /// `immediate` and observes no wait; a declined `try_grant` and a top-up
    /// count nothing.
    #[tokio::test]
    async fn memory_budget_admission_results_count_each_reserve_and_each_successful_try_grant() {
        let (b, metrics) = budget_with_metrics(4);
        let cells = |m: &SharedMetrics| {
            [
                CATCHUP_BUILD_ADMISSION_RESULT_IMMEDIATE,
                CATCHUP_BUILD_ADMISSION_RESULT_WAITED,
                CATCHUP_BUILD_ADMISSION_RESULT_REFUSED_TIMEOUT,
                CATCHUP_BUILD_ADMISSION_RESULT_CANCELLED,
            ]
            .map(|r| admissions(m, r))
        };
        assert_eq!(
            cells(&metrics),
            [0, 0, 0, 0],
            "every cell is seeded at zero"
        );
        assert_eq!(waits_observed(&metrics), 0);

        let a = reserve_now(&b, 2 * KIB).await;
        assert_eq!(
            cells(&metrics),
            [1, 0, 0, 0],
            "an uncontended reserve is immediate"
        );
        assert_eq!(waits_observed(&metrics), 1, "and observes its (zero) wait");

        let g = b.try_grant(2 * KIB).unwrap();
        assert_eq!(
            cells(&metrics),
            [2, 0, 0, 0],
            "a non-blocking grant is immediate"
        );
        assert_eq!(waits_observed(&metrics), 1, "but observes no wait");

        assert!(b.try_grant(KIB).is_none());
        let top = b.top_up(KIB);
        assert_eq!(
            cells(&metrics),
            [2, 0, 0, 0],
            "a decline and a top-up count nothing"
        );
        assert_eq!(waits_observed(&metrics), 1);
        drop(top);

        let mut queued = wait(&b, 2 * KIB);
        assert_queued(&mut queued, "the budget is full").await;
        drop(a);
        let w = assert_granted(&mut queued, "the release admits it").await;
        assert_eq!(
            cells(&metrics),
            [2, 1, 0, 0],
            "granted after queueing: waited"
        );
        assert_eq!(waits_observed(&metrics), 2);

        let refused = b.reserve(KIB, Some(tokio::time::Instant::now())).await;
        assert!(matches!(refused, Err(CatchUpRefusal::TimedOut)));
        assert_eq!(
            cells(&metrics),
            [2, 1, 1, 0],
            "a passed deadline: refused_timeout"
        );
        assert_eq!(waits_observed(&metrics), 3);

        let mut dropped = wait(&b, KIB);
        assert_queued(&mut dropped, "the budget is full").await;
        drop(dropped);
        assert_eq!(cells(&metrics), [2, 1, 1, 1], "a dropped wait: cancelled");
        assert_eq!(waits_observed(&metrics), 4);

        let mut closed = wait(&b, KIB);
        assert_queued(&mut closed, "the budget is full").await;
        b.close();
        assert!(matches!(
            poll_once(&mut closed).await,
            Poll::Ready(Err(CatchUpRefusal::Closed))
        ));
        assert!(matches!(
            poll_once(&mut wait(&b, KIB)).await,
            Poll::Ready(Err(CatchUpRefusal::Closed))
        ));
        assert_eq!(
            cells(&metrics),
            [2, 1, 1, 3],
            "a wait ended by close and a reserve on a closed budget: cancelled"
        );
        assert_eq!(waits_observed(&metrics), 6);
        drop((g, w));
    }

    // ── Per-instance series (b.fsz MR-D4) ────────────────────────────────────

    const RETIREMENT_RESERVED_GAUGE: &str = "live_edit_retirement_budget_reserved_bytes";
    const RETIREMENT_OVERCOMMIT_GAUGE: &str = "live_edit_retirement_budget_overcommit_bytes";
    const RETIREMENT_WAIT_COUNT: &str = "live_edit_retirement_wait_seconds_count";

    /// A retirement allowance ([`RETIREMENT_SERIES`]) of `total_kib` and the
    /// registry it ticks.
    fn retirement_budget_with_metrics(total_kib: u64) -> (Arc<MemoryBudget>, SharedMetrics) {
        let metrics: SharedMetrics =
            Arc::new(crate::observability::metrics::metrics_scrape_only("test"));
        let b = Arc::new(MemoryBudget::with_series(
            total_kib,
            Arc::new(StdMutex::new(Some(metrics.clone()))),
            &RETIREMENT_SERIES,
        ));
        (b, metrics)
    }

    /// `live_edit_retirement_wait_seconds_count{outcome}` for each of
    /// `immediate`, `waited`, `cancelled`, 0 before the first observation.
    fn retirement_waits(metrics: &SharedMetrics) -> [u64; 3] {
        let text = render(metrics);
        ["immediate", "waited", "cancelled"].map(|outcome| {
            sample_value(&text, RETIREMENT_WAIT_COUNT, &[("outcome", outcome)])
                .map_or(0, |v| v as u64)
        })
    }

    /// The retirement allowance ticks its own series, and nothing of the
    /// catch-up budget's though both tick one registry: every grant, release,
    /// top-up and shrink moves the retirement gauges and returns them to 0;
    /// every `reserve` call records one `live_edit_retirement_wait_seconds`
    /// observation under its outcome — `immediate`, `waited`, and
    /// `cancelled` for a dropped wait, a wait the close ended and a reserve
    /// on a closed budget; a non-blocking grant records nothing; and the
    /// catch-up gauges, admission cells and wait histogram never move.
    #[tokio::test]
    async fn memory_budget_with_the_retirement_series_ticks_its_own_series_and_never_the_catch_up_series(
    ) {
        let (b, metrics) = retirement_budget_with_metrics(16);

        let mut grant = reserve_now(&b, 12 * KIB).await;
        assert_eq!(
            gauge(&metrics, RETIREMENT_RESERVED_GAUGE),
            (12 * KIB) as i64,
            "a grant moves the retirement allowance's reserved gauge"
        );
        let top = b.top_up(3 * KIB);
        assert_eq!(
            gauge(&metrics, RETIREMENT_OVERCOMMIT_GAUGE),
            (3 * KIB) as i64,
            "a top-up moves its over-commit gauge"
        );
        let mut queued = wait(&b, 4 * KIB);
        assert_queued(&mut queued, "12 KiB + 3 KiB of over-commit leave 1 KiB").await;
        drop(top);
        grant.shrink_to(4 * KIB);
        let waited =
            assert_granted(&mut queued, "the shrink and the repaid top-up free room").await;
        assert_eq!(
            gauge(&metrics, RETIREMENT_RESERVED_GAUGE),
            (8 * KIB) as i64,
            "the shrink gave back 8 KiB and the queued 4 KiB was granted"
        );
        assert_eq!(gauge(&metrics, RETIREMENT_OVERCOMMIT_GAUGE), 0);

        let try_granted = b.try_grant(KIB).expect("1 KiB fits, nothing queued");
        let mut dropped = wait(&b, 16 * KIB);
        assert_queued(&mut dropped, "the clamped request waits for the whole").await;
        drop(dropped);
        let mut closed = wait(&b, 16 * KIB);
        assert_queued(&mut closed, "the clamped request waits for the whole").await;
        b.close();
        assert!(matches!(
            poll_once(&mut closed).await,
            Poll::Ready(Err(CatchUpRefusal::Closed))
        ));
        assert!(matches!(
            poll_once(&mut wait(&b, KIB)).await,
            Poll::Ready(Err(CatchUpRefusal::Closed))
        ));
        assert_eq!(
            retirement_waits(&metrics),
            [1, 1, 3],
            "one immediate grant, one waited grant, three cancelled reservations; the \
             non-blocking grant recorded nothing"
        );

        drop((grant, waited, try_granted));
        assert_eq!(
            (
                gauge(&metrics, RETIREMENT_RESERVED_GAUGE),
                gauge(&metrics, RETIREMENT_OVERCOMMIT_GAUGE)
            ),
            (0, 0),
            "everything given back"
        );
        assert_eq!(
            (
                gauge(&metrics, RESERVED_GAUGE),
                gauge(&metrics, OVERCOMMIT_GAUGE),
                waits_observed(&metrics)
            ),
            (0, 0, 0),
            "the catch-up budget's gauges and wait histogram never moved"
        );
        for result in crate::observability::metrics::CATCHUP_BUILD_ADMISSION_RESULTS {
            assert_eq!(admissions(&metrics, result), 0, "result={result}");
        }
    }

    /// The catch-up budget, the mirror case: on one registry its traffic
    /// never moves the retirement allowance's gauges or wait histogram.
    #[tokio::test]
    async fn memory_budget_with_the_catch_up_series_never_ticks_the_retirement_series() {
        let (b, metrics) = budget_with_metrics(16);
        let grant = reserve_now(&b, 12 * KIB).await;
        let top = b.top_up(8 * KIB);
        assert_eq!(gauge(&metrics, RESERVED_GAUGE), (12 * KIB) as i64);
        assert_eq!(
            (
                gauge(&metrics, RETIREMENT_RESERVED_GAUGE),
                gauge(&metrics, RETIREMENT_OVERCOMMIT_GAUGE),
                retirement_waits(&metrics)
            ),
            (0, 0, [0, 0, 0]),
            "the catch-up budget's traffic must not reach the retirement series"
        );
        drop((grant, top));
    }
}
