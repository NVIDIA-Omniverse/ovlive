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

//! The structural cache's fenced reads: how a cold start — and, in a
//! no-wait mode, the sampled audit — waits for and holds capacity around
//! b.y3m's fenced snapshot read (`storage::snapshot_fence::read`), and how a
//! cold start gives up for one call.
//!
//! Section references: `SR-n.m` is the structural-edits spec SDD (Spec Bee
//! `b.xes`); `§n.m` is `docs-internal/sdd.md`.
//!
//! # Where it sits
//!
//! A cold start (`storage::structural_cache_cold_start`) takes the fenced read
//! for a layer its probe puts above the plain atomic-read ceilings, or
//! predicts over the plain byte ceiling, and after a plain read that answered
//! `TOO_LARGE`. It drives [`read_object_snapshot_fenced`] for the
//! `structural_cache` consumer with a [`ColdStartFenceHook`] of its own: a
//! catch-up flight's hook carries flight state, and
//! [`CappedFenceWait`](crate::storage::snapshot_fence::read::CappedFenceWait)
//! takes no capacity. A cold start never joins a catch-up flight, and no
//! reader ever reads the structural cache.
//!
//! The sampled audit (`storage::structural_cache_audit`) reads a layer above
//! the plain ceilings with the same hook ([`ColdStartFenceHook::for_audit`]):
//! the same capacity in the same order, under the `structural_cache_audit`
//! consumer, with no deadline in scope, and **no capacity wait**
//! ([`CapacityWait::None`]): the pair is tried without joining the permits'
//! queues, `P_f` with a deadline already passed (which never queues), and the
//! grant only if it fits now (`MemoryBudget::try_grant`, which counts no
//! refusal on the catch-up admission series). Anything not free at once ends
//! the read with [`FencedFailure::CapacityTimeout`], so an audit never gets
//! ahead of a waiting cold start or catch-up. Once it holds them, it holds
//! them for its read like a cold start.
//!
//! # What the hook takes, in order
//!
//! Before each attempt, and again after each `RETRY_AFTER` sleep, the hook
//! takes what it does not already hold:
//!
//! 1. **The cache's permit pair**
//!    ([`StructuralCache::acquire_fenced_cold_start_read_permits`]): the
//!    one-at-a-time cold-start permit, then a plain permit, both FIFO, so at
//!    most one fenced cold start per replica holds or waits for the bounds
//!    below.
//! 2. **A fenced-class catch-up read permit** (`P_f`,
//!    `ValkeyClient::catchup_fenced_read_gate`), shared with the catch-up
//!    flights' fenced reads. Any memory grant is dropped first: nothing waits
//!    for a permit while holding memory (Spec SDD SR-7.5).
//! 3. **A memory grant** of `b × (SNAPSHOT_MEMORY_FACTOR + 1)` from the
//!    catch-up build budget `T` (`ValkeyClient::catchup_build_budget`): the
//!    snapshot plus one fan-out window's replies, bounded by `b`; no build.
//!    The decoded state rides inside the snapshot factor's over-coverage
//!    (project SDD §12.6, *One grant per fenced read of the structural
//!    cache*). The budget clamps a request to `T`, so a layer whose grant would exceed it
//!    runs alone. `b` is the cold start's estimate: the item bound times the
//!    route predictor's bytes per item on a probed route, or a `TOO_LARGE`
//!    answer's scaled tally after a plain read.
//!
//! **The grant is topped up as the read goes.** An estimate can be far
//! below the layer's real bytes: a layer above the item ceiling whose values
//! are much denser than the predictor's figure (large inline arrays, posed or
//! recorded rigs). So after the fan-out's first round and after each later
//! window ([`FencedReadHook::on_read_progress`]), the hook tops the grant up,
//! never waiting, to `SNAPSHOT_MEMORY_FACTOR` × the reply bytes read so far,
//! clamped so that the grant and its one top-up together never exceed `T`
//! (`MemoryGrant::top_up_to`). It holds `P_f` throughout, so the top-up is
//! inside the over-commit `T × (1 + P_f)` funds. Every read of a layer whose
//! weight is within `T` is accounted while it reads, the first included;
//! only a layer heavier than `T` runs past it.
//!
//! Each wait ends at `started + wait_cap(L)` (below), or at once for the
//! audit. On a `RETRY_AFTER` it
//! gives everything back — the grant and its top-up, then `P_f`, then the
//! pair — sleeps the hint, and takes them again in order (policy decision
//! P6). A broken attempt keeps all of them, top-up included, for the retry
//! that follows at once. After a verified read it sets the grant and its
//! top-up to `SNAPSHOT_MEMORY_FACTOR × reply_bytes`
//! (`MemoryGrant::adjust_with_top_up`: the grant shrunk and the top-up
//! dropped, or the one top-up resized to the difference) and then gives
//! `P_f` back, as a flight does; the pair and the memory go to the cold
//! start's `load` as a [`ColdStartHold`].
//!
//! A catch-up never waits on the cache's permits, so the shared bounds form
//! no wait cycle; and because the pair is taken first, a catch-up flight
//! waits for `P_f` behind at most one fenced read of the cache's — a cold
//! start's or an audit's — per replica.
//!
//! # The waiting cap
//!
//! `wait_cap(L) = min(reader_wait_max(L), remaining − hold(L) − S(L))`, where
//! `remaining` is the calling RPC's deadline left when the fenced route began
//! (read once), `hold(L)` the fenced read's expected time, and `S(L)` one
//! structural WATCH attempt's: the time left after waiting must still cover
//! this read and, should it fail, one WATCH attempt. `S` is taken at no less
//! than zero: its fitted structural term `s1·max(e, S_MIN_ITEMS) + s0` can be
//! negative, which leaves `S` below zero for a layer whose hold is short.
//! With no deadline in scope the cap is
//! `reader_wait_max(L)`. Every figure comes from the cost model at run time
//! (`storage::snapshot_fence::cost`); `L` follows the latest size the read was
//! told of.
//!
//! The cap bounds everything from the first attempt on: the capacity waits,
//! the `RETRY_AFTER` sleeps, and the reads themselves. A cap already at zero
//! at the first attempt refuses before any permit is asked for or any acquire
//! sent. A `RETRY_AFTER` whose sleep would end past the cap refuses at once,
//! and so does an attempt after a broken one that starts past it.
//!
//! A layer whose cap would be zero even with the replica's whole configured
//! RPC deadline left — the deadline is at or below `hold(L) + S(L)` at the
//! cold start's estimate ([`wait_reserve`]) — never reaches this hook: the
//! cold start records it `UncacheableReason::OverDeadline` before any fenced
//! read (`storage::structural_cache_cold_start`, "The route"). A cap of zero
//! here — the call's own deadline left being short, or a larger size the read
//! learned from the store — fails that call only and records nothing.
//!
//! # How it fails
//!
//! The hook records why it ended the read ([`FencedFailure`]); the cold start
//! maps it as below (the audit maps a capacity timeout to `skipped` `busy`,
//! the cap or broken attempts to `skipped`, and a closed capacity to a
//! `skipped` it does not count as a fenced result):
//!
//! - [`FencedFailure::Cap`], [`FencedFailure::CapacityTimeout`], or the read's
//!   own `Unavailable` after [`FENCE_READ_ATTEMPTS`] broken attempts
//!   ([`ColdStartFenceHook::broken`]): `UncacheableReason::FencedReadFailed`
//!   for this call only — it takes the WATCH path, nothing is recorded, and
//!   the next call cold-starts afresh.
//! - [`FencedFailure::Closed`] (the budget or the gate closed at shutdown):
//!   the shutdown error ([`shutting_down`]), never the WATCH path. The
//!   cache's own permits are never closed, so a cold start still queued for
//!   the pair at shutdown sees `Closed` only once it reaches `P_f`; if its cap
//!   runs out first it fails for its call instead, and its WATCH attempt then
//!   meets the shutdown like any other write.
//!
//! # Every guard released on every exit
//!
//! The cold start's future owns the hook, which owns the grant and any
//! top-up, `P_f` and the pair; they drop in that order. Dropping the future at
//! any await — the RPC deadline, a cancel — drops them all, aborts the read's
//! fan-out pass on the `fanout-read` thread (`storage::fanout_thread`), and
//! the fenced read's lease guard spawns its own release. A panic unwinds the
//! same way; one in the pass is re-raised in the cold start's own task.

use std::time::Duration;

use tokio::time::Instant;

use crate::error::ServiceError;
use crate::storage::client::ValkeyClient;
use crate::storage::diff_read::SlimSnapshotReadConsumer;
use crate::storage::memory_budget::{
    MemoryGrant, MemoryTopUp, SnapshotMemory, SNAPSHOT_MEMORY_FACTOR,
};
use crate::storage::snapshot_fence::read::{
    FenceAttempt, FenceAttemptOutcome, FenceGrant, FenceReleased, FenceRetry, FencedReadHook,
};
use crate::storage::snapshot_fence::{
    coefficients, ms_to_duration, FenceCoefficients, LayerSize, FENCE_READ_ATTEMPTS,
};
use crate::storage::structural_cache::{ColdStartReadPermit, StructuralCache};
use crate::storage::subscriber_admission::{CatchUpReadPermit, CatchUpRefusal};

/// Why a [`ColdStartFenceHook`] ended its fenced read (module docs, "How it
/// fails").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FencedFailure {
    /// The read's waiting reached [`ColdStartFenceHook::wait_cap`]: at the
    /// first attempt (no time left at all), on a `RETRY_AFTER` whose sleep
    /// would end past it, or before an attempt after a broken one.
    Cap,
    /// A capacity wait — the permit pair, `P_f` or the memory grant — did not
    /// end before the cap.
    CapacityTimeout,
    /// The catch-up read gate or memory budget is closed: the replica is
    /// shutting down.
    Closed,
}

impl FencedFailure {
    /// The `cause` field of the `fenced_read_failed` verdict line.
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Cap => "cap",
            Self::CapacityTimeout => "capacity_timeout",
            Self::Closed => "closed",
        }
    }
}

/// How a [`ColdStartFenceHook`]'s capacity waits end.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CapacityWait {
    /// At `started + wait_cap(L)`: the cold start's.
    UpToTheCap,
    /// At once: whatever is not free when asked for is a
    /// [`FencedFailure::CapacityTimeout`]. The audit's.
    None,
}

/// The `cause` field of a `fenced_read_failed` verdict reached because
/// [`FENCE_READ_ATTEMPTS`] attempts in a row were broken.
pub(crate) const FENCED_FAILURE_BROKEN: &str = "broken";

/// What a cold start holds from the end of its read into `load`: the cache's
/// read permits (the plain permit, or a fenced read's pair) and, for a fenced
/// read, its memory grant and top-up. `load` drops the permits at seam A —
/// after a structural `UpdateLayer`'s commit charge has taken the decoded
/// entry — and the memory at seam B (`storage::structural_cache_cold_start`).
/// When that charge refuses the entry (CR0), neither drops at either seam:
/// both are kept until the admitted entry is put back, and drop only then —
/// except on the plain route, when the read's decoded state and index build
/// are over `plain_cold_start_hold_allowance_bytes`: then both drop at seam
/// A and nothing is stored. A fenced read's refused hold is unchanged.
#[derive(Debug)]
pub(crate) struct ColdStartHold {
    /// The cache's read permits the read held.
    pub(crate) read_permits: ColdStartReadPermit,
    /// The read's memory: a fenced read's grant and top-up; empty for a plain
    /// read, whose reply is bounded by the plain byte ceiling.
    pub(crate) memory: SnapshotMemory,
}

impl ColdStartHold {
    /// A plain read's hold: its plain permit, no memory.
    pub(crate) fn plain(read_permits: ColdStartReadPermit) -> Self {
        Self {
            read_permits,
            memory: SnapshotMemory::default(),
        }
    }
}

/// The error a cold start answers when the replica is shutting down during
/// its fenced read: retryable, and never the WATCH path.
pub(crate) fn shutting_down() -> ServiceError {
    ServiceError::Unavailable(
        "this replica is shutting down; its structural cold-start capacity is closed".to_owned(),
    )
}

/// The cold start's [`FencedReadHook`] (module docs).
///
/// Field order is drop order: the memory, then `P_f`, then the pair.
pub(crate) struct ColdStartFenceHook<'a> {
    grant: Option<MemoryGrant>,
    top_up: Option<MemoryTopUp>,
    read_permit: Option<CatchUpReadPermit>,
    pair: Option<ColdStartReadPermit>,
    client: &'a ValkeyClient,
    cache: &'a StructuralCache,
    coefficients: FenceCoefficients,
    /// The consumer its `P_f` wait is observed under.
    consumer: SlimSnapshotReadConsumer,
    /// Whether its capacity waits wait at all ([`CapacityWait`]).
    capacity_wait: CapacityWait,
    /// When the fenced route began: the cap's origin.
    started: Instant,
    /// The calling RPC's deadline left at `started`; `None` when none is in
    /// scope.
    remaining_at_start: Option<Duration>,
    /// The size the cold start estimated before the read.
    estimate: LayerSize,
    /// The latest layer size the read was told of.
    size: LayerSize,
    /// The largest size a completed fan-out measured, broken attempts
    /// included.
    measured: Option<LayerSize>,
    /// The largest size the fence scripts named that is not the cold start's
    /// own estimate echoed back: the fence key's stored size, from a
    /// `RETRY_AFTER` or a grant.
    reported: Option<LayerSize>,
    /// When the read was first granted a lease: from then on the layer's
    /// colliding writers wait.
    first_grant: Option<Instant>,
    /// Why the hook ended the read, if it did.
    failure: Option<FencedFailure>,
    /// Broken attempts so far.
    broken: u32,
    /// The latest attempt number the read started.
    attempts: u32,
}

impl<'a> ColdStartFenceHook<'a> {
    /// A hook for one fenced cold start of a layer estimated at `estimate`,
    /// its clock started now, with `remaining` of the calling RPC's deadline
    /// left (`None` when none is in scope).
    pub(crate) fn new(
        client: &'a ValkeyClient,
        cache: &'a StructuralCache,
        estimate: LayerSize,
        remaining: Option<Duration>,
    ) -> Self {
        Self::with(
            client,
            cache,
            estimate,
            remaining,
            SlimSnapshotReadConsumer::StructuralCache,
            CapacityWait::UpToTheCap,
        )
    }

    /// A hook for the sampled audit's fenced read of a layer estimated at
    /// `estimate` (`storage::structural_cache_audit`): the same capacity, in
    /// the same order, observed under the `structural_cache_audit` consumer,
    /// with no deadline in scope (so the cap is `reader_wait_max(L)`), and
    /// **no capacity wait** ([`CapacityWait::None`]): a pair, `P_f` or grant
    /// that is not free at once ends the read with
    /// [`FencedFailure::CapacityTimeout`], so an audit never queues ahead of
    /// a cold start or a catch-up. The audit is a sample: it skips.
    pub(crate) fn for_audit(
        client: &'a ValkeyClient,
        cache: &'a StructuralCache,
        estimate: LayerSize,
    ) -> Self {
        Self::with(
            client,
            cache,
            estimate,
            None,
            SlimSnapshotReadConsumer::StructuralCacheAudit,
            CapacityWait::None,
        )
    }

    fn with(
        client: &'a ValkeyClient,
        cache: &'a StructuralCache,
        estimate: LayerSize,
        remaining: Option<Duration>,
        consumer: SlimSnapshotReadConsumer,
        capacity_wait: CapacityWait,
    ) -> Self {
        Self {
            grant: None,
            top_up: None,
            read_permit: None,
            pair: None,
            client,
            cache,
            coefficients: coefficients(client),
            consumer,
            capacity_wait,
            started: Instant::now(),
            remaining_at_start: remaining,
            estimate,
            size: estimate,
            measured: None,
            reported: None,
            first_grant: None,
            failure: None,
            broken: 0,
            attempts: 0,
        }
    }

    /// The waiting cap for a layer of `size` (module docs, "The waiting
    /// cap"): `min(reader_wait_max(L), remaining − hold(L) − max(S(L), 0))`,
    /// or `reader_wait_max(L)` with no deadline in scope. Zero when the
    /// deadline left is already below `hold + S`.
    pub(crate) fn wait_cap(&self, size: LayerSize) -> Duration {
        wait_cap(&self.coefficients, self.remaining_at_start, size)
    }

    /// Why the hook ended the read, if it did.
    pub(crate) fn failure(&self) -> Option<FencedFailure> {
        self.failure
    }

    /// Broken attempts so far.
    pub(crate) fn broken(&self) -> u32 {
        self.broken
    }

    /// The latest attempt the read started.
    pub(crate) fn attempts(&self) -> u32 {
        self.attempts
    }

    /// The latest layer size the read was told of.
    pub(crate) fn size(&self) -> LayerSize {
        self.size
    }

    /// The largest size a completed fan-out measured, if any did.
    #[cfg(test)]
    pub(crate) fn measured(&self) -> Option<LayerSize> {
        self.measured
    }

    /// The largest size the read learned from the store — a completed
    /// fan-out's measure, or a size the fence scripts named other than the
    /// cold start's own estimate — if it learned any.
    pub(crate) fn learned(&self) -> Option<LayerSize> {
        max_size(self.measured, self.reported)
    }

    /// Note a size the fence scripts named: kept unless it is the estimate
    /// echoed back (a fence key with no stored size names the asked size).
    fn note_reported(&mut self, size: LayerSize) {
        if size != self.estimate {
            self.reported = max_size(self.reported, Some(size));
        }
    }

    /// When the read was first granted a lease, if it was: the moment its
    /// pause of the layer's colliding writers began.
    pub(crate) fn first_grant(&self) -> Option<Instant> {
        self.first_grant
    }

    /// How long the fenced route has waited and read so far.
    pub(crate) fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }

    /// The pair and the memory a verified read holds, for `load`. `None` when
    /// the pair is not held, which a verified read rules out.
    pub(crate) fn take_hold(&mut self) -> Option<ColdStartHold> {
        let read_permits = self.pair.take()?;
        let memory = match self.grant.take() {
            Some(grant) => SnapshotMemory::new(grant, self.top_up.take()),
            None => SnapshotMemory::default(),
        };
        Some(ColdStartHold {
            read_permits,
            memory,
        })
    }

    /// End the read for `failure`, remembering why.
    fn fail(&mut self, failure: FencedFailure) -> ServiceError {
        self.failure = Some(failure);
        match failure {
            FencedFailure::Closed => shutting_down(),
            FencedFailure::Cap | FencedFailure::CapacityTimeout => {
                ServiceError::Unavailable(format!(
                    "structural cache's fenced read gave up waiting ({}): waited {} ms of \
                     its {} ms cap",
                    failure.label(),
                    self.elapsed().as_millis(),
                    self.wait_cap(self.size).as_millis()
                ))
            }
        }
    }

    /// The failure a capacity wait's refusal is.
    fn refused(&mut self, refusal: CatchUpRefusal) -> ServiceError {
        self.fail(match refusal {
            CatchUpRefusal::TimedOut => FencedFailure::CapacityTimeout,
            CatchUpRefusal::Closed => FencedFailure::Closed,
        })
    }

    /// Give back everything held idle, in order: the grant and its top-up,
    /// `P_f`, then the pair.
    fn release_capacity(&mut self) {
        self.top_up = None;
        self.grant = None;
        self.read_permit = None;
        self.pair = None;
    }

    /// Take what an attempt on a layer of `size` needs and is not already
    /// held: the pair, `P_f`, then the grant, each wait ending at the cap —
    /// or, for the audit ([`CapacityWait::None`]), none waiting at all.
    async fn take_capacity(&mut self, size: LayerSize) -> Result<(), ServiceError> {
        let deadline = match self.capacity_wait {
            CapacityWait::UpToTheCap => self.started + self.wait_cap(size),
            // Each wait below takes what is free at once and refuses at once
            // otherwise.
            CapacityWait::None => Instant::now(),
        };
        if self.pair.is_none() {
            let pair = match self.capacity_wait {
                // The future is polled once before the deadline is checked, so
                // a free pair is taken even at the deadline.
                CapacityWait::UpToTheCap => tokio::time::timeout_at(
                    deadline,
                    self.cache.acquire_fenced_cold_start_read_permits(),
                )
                .await
                .ok(),
                // Never joins the permits' queues.
                CapacityWait::None => self.cache.try_acquire_fenced_cold_start_read_permits(),
            };
            match pair {
                Some(pair) => self.pair = Some(pair),
                None => return Err(self.fail(FencedFailure::CapacityTimeout)),
            }
        }
        if self.read_permit.is_none() {
            // Never wait for a permit while holding memory (Spec SDD SR-7.5).
            self.top_up = None;
            self.grant = None;
            let permit = self
                .client
                .catchup_fenced_read_gate
                .acquire(deadline, self.consumer)
                .await;
            match permit {
                Ok(permit) => self.read_permit = Some(permit),
                Err(refusal) => return Err(self.refused(refusal)),
            }
        }
        if self.grant.is_none() {
            let bytes = cold_start_grant_bytes(size.bytes);
            match self.capacity_wait {
                CapacityWait::UpToTheCap => {
                    let grant = self
                        .client
                        .catchup_build_budget
                        .reserve(bytes, Some(deadline))
                        .await;
                    match grant {
                        Ok(grant) => self.grant = Some(grant),
                        Err(refusal) => return Err(self.refused(refusal)),
                    }
                }
                // Granted only if it fits now; a decline counts nothing on
                // the catch-up budget's admission series, since nothing
                // waited for memory.
                CapacityWait::None => match self.client.catchup_build_budget.try_grant(bytes) {
                    Some(grant) => self.grant = Some(grant),
                    // A closed budget is the shutdown, not a busy one.
                    None if self.client.catchup_build_budget.is_closed() => {
                        return Err(self.fail(FencedFailure::Closed))
                    }
                    None => return Err(self.fail(FencedFailure::CapacityTimeout)),
                },
            }
        }
        Ok(())
    }
}

/// The waiting cap of a fenced cold start of a layer of `size`, under
/// `coefficients`, with `remaining` of the calling RPC's deadline left when
/// it began (module docs, "The waiting cap"): `min(reader_wait_max(L),
/// remaining − hold(L) − max(S(L), 0))`, or `reader_wait_max(L)` when
/// `remaining` is `None`. Zero when `remaining` is below `hold + S`.
pub fn wait_cap(
    coefficients: &FenceCoefficients,
    remaining: Option<Duration>,
    size: LayerSize,
) -> Duration {
    let reader_wait_max = ms_to_duration(coefficients.timing(size).reader_wait_max_ms);
    match remaining {
        None => reader_wait_max,
        Some(remaining) => {
            reader_wait_max.min(remaining.saturating_sub(wait_reserve(coefficients, size)))
        }
    }
}

/// The deadline [`wait_cap`] keeps back for a layer of `size`: `hold(L) +
/// max(S(L), 0)`, the fenced read's expected time and one structural WATCH
/// attempt's, each taken at no less than zero. A deadline left at or below
/// it leaves a cap of zero.
pub fn wait_reserve(coefficients: &FenceCoefficients, size: LayerSize) -> Duration {
    let timing = coefficients.timing(size);
    ms_to_duration(timing.hold_ms.max(0.0) + timing.structural_attempt_ms.max(0.0))
}

/// The larger of two sizes, item count and bytes each, when either is known.
pub(crate) fn max_size(a: Option<LayerSize>, b: Option<LayerSize>) -> Option<LayerSize> {
    match (a, b) {
        (Some(a), Some(b)) => Some(LayerSize {
            items: a.items.max(b.items),
            bytes: a.bytes.max(b.bytes),
        }),
        (a, None) => a,
        (None, b) => b,
    }
}

/// The memory a fenced cold start reserves before its acquire, for a layer
/// of `b` reply bytes: `b × (SNAPSHOT_MEMORY_FACTOR + 1)`, the snapshot plus
/// one window (module docs).
pub fn cold_start_grant_bytes(b: u64) -> u64 {
    b.saturating_mul(SNAPSHOT_MEMORY_FACTOR + 1)
}

/// What a verified fenced cold start keeps of its grant and top-up: `b ×
/// SNAPSHOT_MEMORY_FACTOR` of the read's measured reply bytes, held until the
/// snapshot is decoded (seam B); and, of the reply bytes read so far, what
/// the read tops its grant up to as it goes.
pub fn cold_start_weight_bytes(reply_bytes: u64) -> u64 {
    reply_bytes.saturating_mul(SNAPSHOT_MEMORY_FACTOR)
}

impl FencedReadHook for ColdStartFenceHook<'_> {
    async fn before_attempt(&mut self, attempt: FenceAttempt) -> Result<(), ServiceError> {
        self.attempts = attempt.attempt;
        self.size = attempt.size;
        let cap = self.wait_cap(attempt.size);
        if !attempt.follows_broken_read() && cap.is_zero() {
            // No time for this read and a WATCH attempt after it: refuse
            // before any permit is asked for or any acquire sent.
            return Err(self.fail(FencedFailure::Cap));
        }
        if attempt.follows_broken_read() && self.elapsed() > cap {
            return Err(self.fail(FencedFailure::Cap));
        }
        self.take_capacity(attempt.size).await
    }

    async fn on_retry_after(&mut self, retry: FenceRetry) -> Result<(), ServiceError> {
        // Nothing held idle through the sleep (P6).
        self.release_capacity();
        self.size = retry.size;
        self.note_reported(retry.size);
        let would_wait = self.elapsed().saturating_add(retry.hint);
        if would_wait > self.wait_cap(retry.size) {
            return Err(self.fail(FencedFailure::Cap));
        }
        tokio::time::sleep(retry.hint).await;
        self.take_capacity(retry.size).await
    }

    async fn on_grant(&mut self, grant: FenceGrant) {
        self.first_grant.get_or_insert_with(Instant::now);
        self.size = grant.size;
        self.note_reported(grant.size);
    }

    fn on_read_progress(&mut self, bytes_read: u64) {
        // The read holds `P_f` throughout (asserted). A top-up records memory
        // already resident, so it is taken whenever a grant is held:
        // withholding it would only hide that memory from admission.
        debug_assert!(self.read_permit.is_some(), "a fenced read holds P_f");
        if let Some(grant) = &self.grant {
            grant.top_up_to(&mut self.top_up, cold_start_weight_bytes(bytes_read));
        }
    }

    async fn before_release(&mut self) {}

    async fn after_release(&mut self, released: FenceReleased) {
        self.measured = max_size(self.measured, released.measured);
        match released.outcome {
            FenceAttemptOutcome::Verified => {
                // Set the grant and its one top-up to the measured weight
                // while `P_f` is still held, and only then give `P_f` back:
                // at most `P_f` top-ups are outstanding past `T`.
                if let (Some(grant), Some(measured)) = (self.grant.as_mut(), released.measured) {
                    grant.adjust_with_top_up(
                        &mut self.top_up,
                        cold_start_weight_bytes(measured.bytes),
                    );
                }
                self.read_permit = None;
            }
            // Everything is kept for the retry that follows at once.
            FenceAttemptOutcome::Broken => self.broken += 1,
            FenceAttemptOutcome::Ended => {}
        }
    }
}

/// Whether a read that ended with `error` under `hook` failed for its call
/// (module docs, "How it fails"): the hook's cap or capacity timeout, or the
/// read's `Unavailable` after [`FENCE_READ_ATTEMPTS`] broken attempts. Returns
/// the verdict line's `cause`.
pub(crate) fn failed_for_the_call(
    hook: &ColdStartFenceHook<'_>,
    error: &ServiceError,
) -> Option<&'static str> {
    match hook.failure() {
        Some(failure @ (FencedFailure::Cap | FencedFailure::CapacityTimeout)) => {
            Some(failure.label())
        }
        Some(FencedFailure::Closed) => None,
        None => (matches!(error, ServiceError::Unavailable(_))
            && hook.broken() >= FENCE_READ_ATTEMPTS)
            .then_some(FENCED_FAILURE_BROKEN),
    }
}

#[cfg(test)]
mod tests {
    //! The waiting cap's arithmetic ([`wait_cap`], pure), and the hook's
    //! events driven one at a time against a real client's gates and budget
    //! (`P_f` and `T`) and a real cache's permit pair: what it takes, in what
    //! order, what it gives back and when, and how each failure is
    //! classified. Through the cold start, end to end, these are in
    //! `tests/integration/storage/structural_cache_cold_start_test.rs`; here
    //! each event's own effect is pinned — in particular the grant's adjust
    //! after a verified read, which has no await between it and the end of
    //! the read and so cannot be seen from outside the hook. The progress
    //! top-ups are pinned both one report at a time and through a real
    //! fenced read of a byte-dense layer, the hook wrapped to record what it
    //! holds at each report and before the release
    //! (`storage::test_support::fenced_read_progress`).
    //!
    //! No figure the cost model or configuration owns is restated: every
    //! timing comes from the coefficients, `T` and the `P_f` limit from the
    //! client, the memory factor from its constant.

    use std::future::Future;
    use std::pin::pin;
    use std::sync::Arc;
    use std::task::Poll;
    use std::time::Duration;

    use super::*;
    use crate::storage::diff_read::SnapshotReadConsumer;
    use crate::storage::memory_budget::kib_for_bytes;
    use crate::storage::snapshot_fence::{ReleaseVerdict, RetryReason};
    use crate::storage::test_support::fenced_read_progress::{
        as_held, assert_accounted_while_reading, lowered_client, posed_rig_layer, read_fenced,
        Held, ObservedHook,
    };
    use crate::storage::test_support::TestValkey;

    /// A budget no estimate here comes near; the cache's only role is its
    /// permit pair.
    const CACHE_BUDGET: u64 = 64 * 1024 * 1024;
    /// The test hooks' layer estimate.
    const ESTIMATE: LayerSize = LayerSize {
        items: 5_000,
        bytes: 90_000,
    };

    fn recorded() -> FenceCoefficients {
        FenceCoefficients::recorded(crate::config::SNAPSHOT_ATOMIC_READ_MAX_ITEMS_DEFAULT)
    }

    /// `hold(L) + max(S(L), 0)` as the cap's reserve, computed independently
    /// of [`wait_cap`]'s own expression only in clamping each term.
    fn reserve(coefficients: &FenceCoefficients, size: LayerSize) -> Duration {
        let t = coefficients.timing(size);
        ms_to_duration(t.hold_ms.max(0.0) + t.structural_attempt_ms.max(0.0))
    }

    fn reader_wait_max(coefficients: &FenceCoefficients, size: LayerSize) -> Duration {
        ms_to_duration(coefficients.timing(size).reader_wait_max_ms)
    }

    const SIZES: [LayerSize; 4] = [
        LayerSize { items: 0, bytes: 0 },
        LayerSize {
            items: 1_000,
            bytes: 12_000,
        },
        LayerSize {
            items: 40_000,
            bytes: 800_000,
        },
        LayerSize {
            items: 2_000_000,
            bytes: 40_000_000,
        },
    ];

    // ── wait_cap (pure) ─────────────────────────────────────────────────────

    /// No deadline in scope: the cap is `reader_wait_max(L)`, exactly
    /// `CappedFenceWait::cap`'s expression on the same coefficients, at every
    /// item ceiling (the ceiling feeds `min_gap` and `Q`) and size.
    #[test]
    fn wait_cap_with_no_deadline_is_reader_wait_max_as_capped_fence_wait_caps() {
        for item_ceiling in [crate::config::SNAPSHOT_ATOMIC_READ_MAX_ITEMS_DEFAULT, 64, 1] {
            let coefficients = FenceCoefficients::recorded(item_ceiling);
            for size in SIZES {
                assert_eq!(
                    wait_cap(&coefficients, None, size),
                    reader_wait_max(&coefficients, size),
                    "item ceiling {item_ceiling}, size {size:?}"
                );
            }
        }
    }

    /// A deadline that leaves more than `reader_wait_max(L)` past the
    /// reserve: the cap is `reader_wait_max(L)` (the minimum's other side).
    #[test]
    fn wait_cap_with_a_long_deadline_is_reader_wait_max() {
        let coefficients = recorded();
        for size in SIZES {
            let rwm = reader_wait_max(&coefficients, size);
            let remaining = reserve(&coefficients, size) + rwm + Duration::from_secs(1);
            assert_eq!(
                wait_cap(&coefficients, Some(remaining), size),
                rwm,
                "size {size:?}"
            );
        }
    }

    /// A deadline under `reader_wait_max(L)` past the reserve: the cap is the
    /// deadline left less `hold(L)` and `S(L)` — time kept for this read and
    /// one WATCH attempt.
    #[test]
    fn wait_cap_is_the_deadline_left_less_hold_and_s() {
        const LEFT: Duration = Duration::from_millis(250);
        let coefficients = recorded();
        for size in SIZES {
            assert!(
                reader_wait_max(&coefficients, size) > LEFT,
                "fixture: reader_wait_max is over {LEFT:?} at {size:?}"
            );
            let remaining = reserve(&coefficients, size) + LEFT;
            assert_eq!(
                wait_cap(&coefficients, Some(remaining), size),
                LEFT,
                "size {size:?}: remaining − hold − S"
            );
        }
    }

    /// A deadline at or under `hold(L) + S(L)` leaves no time: zero, never a
    /// wrapped or negative figure.
    #[test]
    fn wait_cap_is_zero_once_the_deadline_left_is_at_or_under_hold_plus_s() {
        let coefficients = recorded();
        for size in SIZES {
            let reserve = reserve(&coefficients, size);
            assert!(reserve > Duration::ZERO, "fixture: a positive reserve");
            // Just under: 1 ns, since a refit `F` and `S` can put a small
            // size's reserve under a millisecond.
            for remaining in [reserve, reserve - Duration::from_nanos(1), Duration::ZERO] {
                assert_eq!(
                    wait_cap(&coefficients, Some(remaining), size),
                    Duration::ZERO,
                    "size {size:?}, remaining {remaining:?}"
                );
            }
        }
    }

    /// Each reserve term is taken at no less than zero: a negative `S` (the
    /// fitted chord below its range) or a negative `hold` reserves nothing
    /// for itself and never lengthens the cap by its negative value. Built on
    /// coefficient sets whose `F` and `S − F` terms are constants.
    #[test]
    fn wait_cap_takes_hold_and_s_at_no_less_than_zero() {
        const REMAINING: Duration = Duration::from_millis(150);
        // hold = 50 ms, S = hold − 80 = −30 ms: reserve 50 ms, cap 100 ms
        // (an unclamped S would leave 130 ms).
        let negative_s = FenceCoefficients {
            f0_ms: 50.0,
            f1_ms_per_item: 0.0,
            f2_ms_per_byte: 0.0,
            s1_ms_per_item: 0.0,
            s0_ms: -80.0,
            ..recorded()
        };
        // hold = −30 ms, S = hold + 80 = 50 ms: reserve 50 ms, cap 100 ms
        // (an unclamped hold would leave 130 ms).
        let negative_hold = FenceCoefficients {
            f0_ms: -30.0,
            s0_ms: 80.0,
            ..negative_s
        };
        for (what, coefficients, hold, s) in [
            ("negative S", negative_s, 50.0, -30.0),
            ("negative hold", negative_hold, -30.0, 50.0),
        ] {
            let t = coefficients.timing(ESTIMATE);
            assert_eq!(
                (t.hold_ms, t.structural_attempt_ms),
                (hold, s),
                "fixture: {what}"
            );
            assert!(t.reader_wait_max_ms > REMAINING.as_millis() as f64);
            assert_eq!(
                wait_cap(&coefficients, Some(REMAINING), ESTIMATE),
                Duration::from_millis(100),
                "{what}: only the positive term is reserved"
            );
        }
    }

    // ── The hook, one event at a time ───────────────────────────────────────

    /// What the hook holds, as the client's gate and budget and the cache's
    /// seam see it: `(fenced reads in flight, P_f held, reserved bytes,
    /// over-committed bytes)`.
    fn held(client: &ValkeyClient, cache: &StructuralCache) -> (usize, usize, u64, u64) {
        let gate = &client.catchup_fenced_read_gate;
        (
            cache.cold_start_reads_in_flight_for_test().fenced,
            gate.limit() - gate.available(),
            client.catchup_build_budget.reserved_bytes(),
            client.catchup_build_budget.overcommit_bytes(),
        )
    }

    const NOTHING: (usize, usize, u64, u64) = (0, 0, 0, 0);

    /// `bytes` as the budget grants it: whole KiB, clamped to `T` (read from
    /// the client).
    fn granted(client: &ValkeyClient, bytes: u64) -> u64 {
        let total_kib = client.catchup_build_budget.total_bytes() / 1024;
        kib_for_bytes(bytes).min(total_kib) * 1024
    }

    /// `(SNAPSHOT_MEMORY_FACTOR + 1) × b`, as granted.
    fn grant_for(client: &ValkeyClient, b: u64) -> u64 {
        granted(client, b * (SNAPSHOT_MEMORY_FACTOR + 1))
    }

    fn first_attempt(size: LayerSize) -> FenceAttempt {
        FenceAttempt { attempt: 1, size }
    }

    fn released(outcome: FenceAttemptOutcome, measured: LayerSize) -> FenceReleased {
        FenceReleased {
            verdict: Some(match outcome {
                FenceAttemptOutcome::Broken => ReleaseVerdict::Broken,
                _ => ReleaseVerdict::Verified,
            }),
            outcome,
            measured: Some(measured),
        }
    }

    /// Poll `fut` once.
    async fn poll_once<F: Future>(fut: std::pin::Pin<&mut F>) -> Poll<F::Output> {
        let mut fut = fut;
        std::future::poll_fn(|cx| Poll::Ready(fut.as_mut().poll(cx))).await
    }

    /// Before an attempt the hook takes the pair, `P_f` and a grant of the
    /// snapshot plus one window, `(SNAPSHOT_MEMORY_FACTOR + 1) × b`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn before_attempt_takes_the_pair_p_f_and_the_snapshot_plus_one_window_grant() {
        let tv = TestValkey::setup().await;
        let client = tv.client();
        let cache = StructuralCache::new(CACHE_BUDGET);
        let mut hook = ColdStartFenceHook::new(client, &cache, ESTIMATE, None);

        hook.before_attempt(first_attempt(ESTIMATE))
            .await
            .expect("free capacity is taken");

        assert_eq!(
            held(client, &cache),
            (1, 1, grant_for(client, ESTIMATE.bytes), 0),
            "the pair, one P_f permit and (factor + 1) × b"
        );
        drop(hook);
        assert_eq!(held(client, &cache), NOTHING, "dropping the hook frees all");
    }

    /// With every `P_f` permit held elsewhere, the hook waits for `P_f`
    /// holding its pair and no memory: the grant is never taken before `P_f`
    /// (nothing waits for a permit while holding memory).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn before_attempt_waits_for_p_f_holding_the_pair_and_no_memory() {
        let tv = TestValkey::setup().await;
        let client = tv.client();
        let cache = StructuralCache::new(CACHE_BUDGET);
        let gate = &client.catchup_fenced_read_gate;
        let mut others = Vec::new();
        for _ in 0..gate.limit() {
            others.push(
                gate.acquire(
                    Instant::now() + Duration::from_secs(3_600),
                    SnapshotReadConsumer::Catchup,
                )
                .await
                .expect("fixture: every P_f permit taken"),
            );
        }
        let mut hook = ColdStartFenceHook::new(client, &cache, ESTIMATE, None);
        let mut attempt = pin!(hook.before_attempt(first_attempt(ESTIMATE)));

        assert!(poll_once(attempt.as_mut()).await.is_pending());
        assert_eq!(
            held(client, &cache),
            (1, gate.limit(), 0, 0),
            "waiting for P_f: the pair, no memory"
        );
        drop(others);
        tokio::time::timeout(Duration::from_secs(20), attempt)
            .await
            .expect("P_f freed: the attempt's capacity is taken")
            .expect("and granted");
        assert_eq!(
            held(client, &cache),
            (1, 1, grant_for(client, ESTIMATE.bytes), 0)
        );
    }

    /// A verified read adjusts the grant to `SNAPSHOT_MEMORY_FACTOR ×
    /// reply_bytes` of the measured size and gives `P_f` back, keeping the
    /// pair; the hold the cold start takes carries the pair and exactly that
    /// memory, and dropping it frees all. Measured lighter than the estimate
    /// the grant shrinks; heavier, a top-up covers the difference.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn verified_release_adjusts_the_grant_to_the_measured_weight_and_frees_p_f() {
        let tv = TestValkey::setup().await;
        let client = tv.client();
        for (what, measured_bytes) in [
            ("lighter", ESTIMATE.bytes / 3),
            ("heavier", ESTIMATE.bytes * 3),
        ] {
            let cache = StructuralCache::new(CACHE_BUDGET);
            let mut hook = ColdStartFenceHook::new(client, &cache, ESTIMATE, None);
            hook.before_attempt(first_attempt(ESTIMATE))
                .await
                .expect("free capacity is taken");
            let measured = LayerSize {
                items: ESTIMATE.items,
                bytes: measured_bytes,
            };
            let grant = grant_for(client, ESTIMATE.bytes);
            let weight = granted(client, measured_bytes * SNAPSHOT_MEMORY_FACTOR);
            let (reserved, overcommit) = if weight <= grant {
                (weight, 0)
            } else {
                (grant, weight - grant)
            };

            hook.after_release(released(FenceAttemptOutcome::Verified, measured))
                .await;

            assert_eq!(
                held(client, &cache),
                (1, 0, reserved, overcommit),
                "{what}: the pair kept, P_f given back, the grant adjusted to factor × the \
                 measured reply bytes"
            );
            assert_eq!(hook.measured(), Some(measured), "{what}");
            let hold = hook.take_hold().expect("a verified read holds its pair");
            assert_eq!(
                hold.memory.bytes(),
                weight,
                "{what}: the hold carries the adjusted memory to load"
            );
            drop(hook);
            assert_eq!(
                held(client, &cache),
                (1, 0, reserved, overcommit),
                "{what}: the hook dropped, the hold still holds the pair and the memory"
            );
            drop(hold);
            assert_eq!(held(client, &cache), NOTHING, "{what}");
        }
    }

    // ── Progress top-ups (the fenced read's per-window reports) ─────────────

    /// The bytes `hook` holds for its read: its grant and its one top-up.
    fn hook_memory(hook: &ColdStartFenceHook<'_>) -> u64 {
        hook.grant.as_ref().map_or(0, MemoryGrant::bytes)
            + hook.top_up.as_ref().map_or(0, MemoryTopUp::bytes)
    }

    /// The probed cold start's estimate for a layer whose `TOO_LARGE` answer
    /// bounds it at `answer.item_bound` items: the bound × the route
    /// predictor's bytes per item.
    fn probed_estimate(answer: &crate::storage::diff_read::SnapshotTooLarge) -> LayerSize {
        LayerSize {
            items: answer.item_bound,
            bytes: answer.item_bound
                * crate::storage::structural_cache_cold_start::STRUCTURAL_COLD_START_ROUTE_BYTES_PER_ITEM,
        }
    }

    /// One fenced cold start of a posed-rig layer through `client` (its
    /// budget the read's own), observed: returns the observed hook after the
    /// verified read, still holding what the read left it.
    async fn observed_rig_cold_start<'a>(
        tv: &TestValkey,
        client: &'a ValkeyClient,
        cache: &'a StructuralCache,
        layer: &str,
    ) -> ObservedHook<ColdStartFenceHook<'a>> {
        let (oid, answer) = posed_rig_layer(tv, client, layer).await;
        let estimate = probed_estimate(&answer);
        let mut hook = ObservedHook::new(
            ColdStartFenceHook::new(client, cache, estimate, None),
            hook_memory,
            Arc::clone(&client.catchup_build_budget),
        );
        read_fenced(
            client,
            tv,
            oid,
            SlimSnapshotReadConsumer::StructuralCache,
            estimate,
            &mut hook,
        )
        .await;
        hook
    }

    /// R1 — a byte-dense layer (posed-rig shaped: few items, large inline
    /// arrays, above the item ceiling) under-sizes the cold start's grant
    /// from its estimate, so as the read goes the hook tops the grant up to
    /// `SNAPSHOT_MEMORY_FACTOR` × the bytes read so far: at the end of the
    /// read, before the adjust, the read holds at least that factor of its
    /// measured bytes, all of it accounted in the budget, where the grant
    /// alone was below it. The verified adjust then leaves exactly the
    /// measured weight, in one grant and one top-up.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_byte_dense_reads_grant_is_topped_up_to_the_snapshot_factor_of_the_bytes_read() {
        let tv = TestValkey::setup().await;
        let client = lowered_client(&tv, None);
        let total = client.catchup_build_budget.total_bytes();
        let cache = StructuralCache::new(CACHE_BUDGET);
        let mut observed = Box::pin(observed_rig_cold_start(
            &tv,
            &client,
            &cache,
            "rig-cold-start",
        ))
        .await;

        assert_accounted_while_reading(&observed, total);
        let measured = observed.measured.expect("measured");
        let weight = as_held(cold_start_weight_bytes(measured.bytes), total);
        assert_eq!(
            observed.held(),
            Held {
                hook: weight,
                budget: weight
            },
            "the verified adjust leaves exactly factor × the measured bytes: no top-up counted \
             twice"
        );
        let hold = observed
            .inner
            .take_hold()
            .expect("a verified read holds its pair");
        assert_eq!(hold.memory.bytes(), weight, "and hands it to load");
        drop(hold);
        drop(observed);
        assert_eq!(
            client.catchup_build_budget.reserved_bytes()
                + client.catchup_build_budget.overcommit_bytes(),
            0
        );
    }

    /// R3b — the cumulative clamp: with a `T` well below the dense layer's
    /// weight, the grant and its top-up never exceed `T` at any report, and
    /// equal it by the end of the read and after the adjust.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn progress_top_ups_never_take_grant_and_top_up_past_a_small_t_and_reach_it() {
        const SMALL_T_KIB: u64 = 1024;
        let tv = TestValkey::setup().await;
        let client = lowered_client(&tv, Some(SMALL_T_KIB));
        let total = SMALL_T_KIB * 1024;
        let cache = StructuralCache::new(CACHE_BUDGET);
        let observed = Box::pin(observed_rig_cold_start(
            &tv,
            &client,
            &cache,
            "rig-cold-start-small-t",
        ))
        .await;

        let measured = observed.measured.expect("measured");
        assert!(
            cold_start_weight_bytes(measured.bytes) > total,
            "fixture: the rig's weight must be past the small T"
        );
        let end = assert_accounted_while_reading(&observed, total);
        assert_eq!(end, total, "the grant and its top-up reach T exactly");
        assert!(
            observed.reports.iter().all(|r| r.held.hook <= total),
            "never past T"
        );
        assert_eq!(
            observed.held().hook,
            total,
            "the adjust to a weight past T clamps the same way"
        );
    }

    /// R3a — a progress report's top-up never waits: with the rest of `T`
    /// reserved by another grant and a reservation queued behind it, the
    /// report (synchronous) returns having taken the top-up past the total,
    /// and the queued reservation is still queued.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_progress_top_up_never_waits_on_a_fully_reserved_budget() {
        let tv = TestValkey::setup().await;
        let client = tv.client();
        let cache = StructuralCache::new(CACHE_BUDGET);
        let budget = &client.catchup_build_budget;
        let mut hook = ColdStartFenceHook::new(client, &cache, ESTIMATE, None);
        hook.before_attempt(first_attempt(ESTIMATE))
            .await
            .expect("free capacity is taken");
        let grant = grant_for(client, ESTIMATE.bytes);
        let others = budget
            .try_grant(budget.total_bytes() - grant)
            .expect("fixture: the rest of T");
        let mut queued = pin!(budget.reserve(1024, None));
        assert!(poll_once(queued.as_mut()).await.is_pending(), "T is full");

        let bytes_read = ESTIMATE.bytes * 3;
        hook.on_read_progress(bytes_read);

        let top_up = granted(client, cold_start_weight_bytes(bytes_read)) - grant;
        assert!(top_up > 0, "fixture: the report needs a top-up");
        assert_eq!(
            held(client, &cache),
            (1, 1, budget.total_bytes(), top_up),
            "taken at once, past a full T"
        );
        assert!(
            poll_once(queued.as_mut()).await.is_pending(),
            "the queued reservation is still queued"
        );
        assert_eq!(budget.waiters(), 1);
        drop(others);
    }

    /// R2, at the hook — a verified read after progress top-ups sets the
    /// grant and its one top-up to the measured weight: measured lighter than
    /// the grant, the grant shrinks and the progress top-up is dropped;
    /// heavier, the grant is kept and the one top-up is exactly the
    /// difference, whatever progress had taken.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_verified_release_sets_the_grant_and_its_progress_top_up_to_the_measured_weight() {
        let tv = TestValkey::setup().await;
        let client = tv.client();
        for (what, progress_bytes, measured_bytes) in [
            ("lighter", ESTIMATE.bytes * 2, ESTIMATE.bytes / 3),
            (
                "heavier, past progress",
                ESTIMATE.bytes * 2,
                ESTIMATE.bytes * 4,
            ),
            (
                "heavier, under progress",
                ESTIMATE.bytes * 4,
                ESTIMATE.bytes * 2,
            ),
        ] {
            let cache = StructuralCache::new(CACHE_BUDGET);
            let mut hook = ColdStartFenceHook::new(client, &cache, ESTIMATE, None);
            hook.before_attempt(first_attempt(ESTIMATE))
                .await
                .expect("free capacity is taken");
            hook.on_read_progress(progress_bytes);
            let grant = grant_for(client, ESTIMATE.bytes);
            assert!(hook.top_up.is_some(), "{what}: fixture: progress topped up");
            let weight = granted(client, cold_start_weight_bytes(measured_bytes));
            let (reserved, overcommit) = if weight <= grant {
                (weight, 0)
            } else {
                (grant, weight - grant)
            };

            hook.after_release(released(
                FenceAttemptOutcome::Verified,
                LayerSize {
                    items: ESTIMATE.items,
                    bytes: measured_bytes,
                },
            ))
            .await;

            assert_eq!(
                held(client, &cache),
                (1, 0, reserved, overcommit),
                "{what}: exactly the measured weight, P_f given back"
            );
            assert_eq!(
                hook.top_up.as_ref().map_or(0, MemoryTopUp::bytes),
                overcommit,
                "{what}: one top-up of exactly weight − grant"
            );
            drop(hook);
            assert_eq!(held(client, &cache), NOTHING, "{what}");
        }
    }

    /// A `RETRY_AFTER` after a broken attempt that kept a progress top-up
    /// gives the top-up back with the grant: nothing is held through the
    /// sleep.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn retry_after_gives_a_kept_progress_top_up_back_with_the_grant() {
        let tv = TestValkey::setup().await;
        let client = tv.client();
        let cache = StructuralCache::new(CACHE_BUDGET);
        let mut hook = ColdStartFenceHook::new(client, &cache, ESTIMATE, None);
        hook.before_attempt(first_attempt(ESTIMATE))
            .await
            .expect("free capacity is taken");
        hook.on_read_progress(ESTIMATE.bytes * 3);
        hook.after_release(released(FenceAttemptOutcome::Broken, ESTIMATE))
            .await;
        assert!(
            held(client, &cache).3 > 0,
            "a broken attempt keeps its top-up for the retry"
        );
        hook.before_attempt(FenceAttempt {
            attempt: 2,
            size: ESTIMATE,
        })
        .await
        .expect("the retry's attempt");
        let retry = FenceRetry {
            hint: hook.wait_cap(ESTIMATE) / 2,
            reason: RetryReason::Epoch,
            size: ESTIMATE,
        };

        let mut sleeping = pin!(hook.on_retry_after(retry));
        assert!(poll_once(sleeping.as_mut()).await.is_pending(), "it sleeps");
        assert_eq!(
            held(client, &cache),
            NOTHING,
            "nothing held through the sleep"
        );
    }

    /// A broken attempt keeps everything for the retry that follows at once:
    /// the pair, `P_f` and the grant, unadjusted; the next attempt takes
    /// nothing more.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn broken_release_keeps_everything_for_the_retry() {
        let tv = TestValkey::setup().await;
        let client = tv.client();
        let cache = StructuralCache::new(CACHE_BUDGET);
        let mut hook = ColdStartFenceHook::new(client, &cache, ESTIMATE, None);
        hook.before_attempt(first_attempt(ESTIMATE))
            .await
            .expect("free capacity is taken");
        let holding = (1, 1, grant_for(client, ESTIMATE.bytes), 0);

        hook.after_release(released(
            FenceAttemptOutcome::Broken,
            LayerSize {
                items: ESTIMATE.items,
                bytes: ESTIMATE.bytes / 3,
            },
        ))
        .await;
        assert_eq!(hook.broken(), 1);
        assert_eq!(
            held(client, &cache),
            holding,
            "all kept, the grant unadjusted"
        );
        hook.before_attempt(FenceAttempt {
            attempt: 2,
            size: ESTIMATE,
        })
        .await
        .expect("the retry's attempt");
        assert_eq!(
            held(client, &cache),
            holding,
            "the retry takes nothing more"
        );
    }

    /// On `RETRY_AFTER` the hook gives everything back before it sleeps: in
    /// the sleep it holds no pair, no `P_f` and no grant.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn retry_after_gives_everything_back_before_its_sleep() {
        let tv = TestValkey::setup().await;
        let client = tv.client();
        let cache = StructuralCache::new(CACHE_BUDGET);
        let mut hook = ColdStartFenceHook::new(client, &cache, ESTIMATE, None);
        hook.before_attempt(first_attempt(ESTIMATE))
            .await
            .expect("free capacity is taken");
        // Half the cap: a sleep well within it, whatever the cost model's
        // coefficients make `reader_wait_max(L)`.
        let retry = FenceRetry {
            hint: hook.wait_cap(ESTIMATE) / 2,
            reason: RetryReason::Epoch,
            size: ESTIMATE,
        };
        assert!(
            hook.elapsed() + retry.hint < hook.wait_cap(ESTIMATE),
            "fixture: the sleep is within the cap"
        );

        let mut sleeping = pin!(hook.on_retry_after(retry));
        assert!(poll_once(sleeping.as_mut()).await.is_pending(), "it sleeps");
        assert_eq!(
            held(client, &cache),
            NOTHING,
            "nothing held through the sleep"
        );
    }

    /// A `RETRY_AFTER` whose sleep would end past the cap refuses at once,
    /// without sleeping: `Cap`, everything given back, the call's failure.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn retry_after_past_the_cap_refuses_at_once_with_cap() {
        let tv = TestValkey::setup().await;
        let client = tv.client();
        let cache = StructuralCache::new(CACHE_BUDGET);
        let remaining = reserve(&coefficients(client), ESTIMATE) + Duration::from_millis(100);
        let mut hook = ColdStartFenceHook::new(client, &cache, ESTIMATE, Some(remaining));
        assert_eq!(hook.wait_cap(ESTIMATE), Duration::from_millis(100));
        hook.before_attempt(first_attempt(ESTIMATE))
            .await
            .expect("free capacity is taken");

        let refused = {
            let mut retry = pin!(hook.on_retry_after(FenceRetry {
                hint: Duration::from_secs(10),
                reason: RetryReason::Epoch,
                size: ESTIMATE,
            }));
            poll_once(retry.as_mut()).await
        };

        let Poll::Ready(Err(error)) = refused else {
            panic!("a sleep past the cap is refused at its first poll; got {refused:?}");
        };
        assert!(matches!(error, ServiceError::Unavailable(_)), "{error:?}");
        assert_eq!(hook.failure(), Some(FencedFailure::Cap));
        assert_eq!(failed_for_the_call(&hook, &error), Some("cap"));
        assert_eq!(held(client, &cache), NOTHING, "everything given back");
    }

    /// A deadline that leaves no cap refuses the first attempt before any
    /// permit is asked for: `Cap`, and nothing was ever taken.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_zero_cap_refuses_the_first_attempt_before_taking_any_capacity() {
        let tv = TestValkey::setup().await;
        let client = tv.client();
        let cache = StructuralCache::new(CACHE_BUDGET);
        let remaining = reserve(&coefficients(client), ESTIMATE);
        let mut hook = ColdStartFenceHook::new(client, &cache, ESTIMATE, Some(remaining));

        let error = hook
            .before_attempt(first_attempt(ESTIMATE))
            .await
            .expect_err("no time for the read and a WATCH attempt");

        assert_eq!(hook.failure(), Some(FencedFailure::Cap));
        assert_eq!(failed_for_the_call(&hook, &error), Some("cap"));
        assert_eq!(held(client, &cache), NOTHING, "nothing was taken");
    }

    /// A capacity wait that does not end before the cap — `P_f` held
    /// elsewhere — is `CapacityTimeout`, the call's failure, with no memory
    /// taken; dropping the hook gives back what it holds.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_p_f_wait_past_the_cap_is_a_capacity_timeout() {
        let tv = TestValkey::setup().await;
        let client = tv.client();
        let cache = StructuralCache::new(CACHE_BUDGET);
        let gate = &client.catchup_fenced_read_gate;
        let mut others = Vec::new();
        for _ in 0..gate.limit() {
            others.push(
                gate.acquire(
                    Instant::now() + Duration::from_secs(3_600),
                    SnapshotReadConsumer::Catchup,
                )
                .await
                .expect("fixture: every P_f permit taken"),
            );
        }
        let remaining = reserve(&coefficients(client), ESTIMATE) + Duration::from_millis(200);
        let mut hook = ColdStartFenceHook::new(client, &cache, ESTIMATE, Some(remaining));

        let error = tokio::time::timeout(
            Duration::from_secs(20),
            hook.before_attempt(first_attempt(ESTIMATE)),
        )
        .await
        .expect("the wait ends at the cap")
        .expect_err("P_f never freed before the cap");

        assert_eq!(hook.failure(), Some(FencedFailure::CapacityTimeout));
        assert_eq!(failed_for_the_call(&hook, &error), Some("capacity_timeout"));
        assert_eq!(held(client, &cache).2, 0, "no memory was taken");
        drop(hook);
        assert_eq!(held(client, &cache), (0, gate.limit(), 0, 0));
    }

    /// A closed budget (shutdown) is `Closed`: the shutdown error, never the
    /// call's WATCH fallback.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_closed_budget_is_closed_and_not_the_calls_failure() {
        let tv = TestValkey::setup().await;
        let client = tv.client();
        let cache = StructuralCache::new(CACHE_BUDGET);
        client.catchup_build_budget.close();
        let mut hook = ColdStartFenceHook::new(client, &cache, ESTIMATE, None);

        let error = hook
            .before_attempt(first_attempt(ESTIMATE))
            .await
            .expect_err("a closed budget refuses");

        assert!(matches!(error, ServiceError::Unavailable(_)), "{error:?}");
        assert_eq!(hook.failure(), Some(FencedFailure::Closed));
        assert_eq!(failed_for_the_call(&hook, &error), None);
    }

    /// The read's `Unavailable` is the call's failure only after
    /// `FENCE_READ_ATTEMPTS` broken attempts; fewer, or another error, is
    /// returned as it is.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_reads_unavailable_is_the_calls_failure_only_after_every_attempt_broke() {
        let tv = TestValkey::setup().await;
        let client = tv.client();
        let cache = StructuralCache::new(CACHE_BUDGET);
        let mut hook = ColdStartFenceHook::new(client, &cache, ESTIMATE, None);
        let unavailable = ServiceError::Unavailable("broken".to_owned());
        let aborted = ServiceError::Aborted("transient".to_owned());
        for attempt in 1..=FENCE_READ_ATTEMPTS {
            assert_eq!(
                failed_for_the_call(&hook, &unavailable),
                None,
                "{} broken attempts",
                attempt - 1
            );
            hook.before_attempt(FenceAttempt {
                attempt,
                size: ESTIMATE,
            })
            .await
            .expect("capacity");
            hook.after_release(released(FenceAttemptOutcome::Broken, ESTIMATE))
                .await;
        }
        assert_eq!(hook.broken(), FENCE_READ_ATTEMPTS);
        assert_eq!(
            failed_for_the_call(&hook, &unavailable),
            Some(FENCED_FAILURE_BROKEN)
        );
        assert_eq!(failed_for_the_call(&hook, &aborted), None);
    }
}
