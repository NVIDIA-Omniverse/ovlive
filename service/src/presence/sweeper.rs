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

//! Crash-recovery presence sweeper (SDD §13.3).
//!
//! A connection whose replica crashed leaves its membership behind in
//! `le:{sid}:presence` with no lease keeper to renew its lease. Every
//! `LIVE_EDIT_SWEEPER_INTERVAL_SECS` each replica tries to take the
//! presence-sweeper leader lock (`le:presence:sweeper:leader`, a lock of its
//! own so the presence and GC sweeps can run on different replicas); the one
//! that wins runs one **pass**:
//!
//! 1. Sample `Instant::now()`, then take the lock with the acquire-or-refresh
//!    probe on the maintenance pool. On success the pass holds a
//!    [`LeaderLease`] whose local deadline counts from that sample.
//! 2. Page through the session index (`le:presence:active_sessions`) with
//!    `SSCAN`, [`SweeperBudget::session_page`] at a time.
//! 3. For each indexed session, page through its member set with `SSCAN`,
//!    [`SweeperBudget::member_page`] at a time, and split every page into
//!    chunks of at most [`SweeperBudget::member_batch`] connection ids:
//!    `SSCAN COUNT` is only a hint, and a compactly stored set comes back
//!    whole in one page.
//! 4. For each chunk, in this order: the test seam's park point (test builds
//!    only); the cancellation check; [`LeaderLease::ensure_held`], whose
//!    `Lost` ends the pass before the chunk's script; then one
//!    `storage::presence::sweep_member_batch`. That script removes, in one
//!    atomic step, every chunk member whose lease has expired and which is
//!    still a member, deletes its info hash and publishes its `left` — so
//!    the sweeper publishes nothing itself, a connection that renewed its
//!    lease or was re-registered meanwhile is never removed, and every
//!    removal is announced exactly once, by the execution that made it.
//! 5. When a session's member iteration ends with the last chunk reporting no
//!    members left, or having seen no ids at all (an empty session still in
//!    the index), remove the session from the index through
//!    `storage::presence::remove_session_from_index`, the shared
//!    remove-then-recheck protocol (SDD SR-3.8). Nothing else here removes an
//!    index entry.
//!
//! **Bounds.** A pass is one sequential task: at most one presence command
//! (on the data pool) is in flight at a time, and it holds one session page
//! and one member page of ids. No set of ids seen is kept across pages: an id
//! `SSCAN` returns twice re-runs an idempotent chunk, which removes nothing the
//! second time.
//!
//! **Fencing.** A chunk's script runs only after `ensure_held` proved the
//! lease, which renews the lock every third of its TTL. A lost lease ends the
//! pass before its next script call, logs one WARN and counts on
//! `live_edit_presence_sweep_lease_total`. The script's own lease and
//! membership checks keep a removal safe even if two replicas sweep at once;
//! the lease keeps the sweep load to one replica.
//!
//! **Errors and cancellation.** A store error on a session's member page or
//! chunk is logged (a timeout is counted where it is received) and ends that
//! session's iteration; the pass goes on with the next session. An error on
//! the session index's `SSCAN` ends the pass. The service's background-task
//! cancellation is checked at every page and chunk boundary, so a pass stops
//! between chunks, never inside one. Nothing is kept between passes: every
//! pass restarts from cursor 0, and the lease is dropped at its end.
//!
//! Metrics: `live_edit_presence_sweep_duration_seconds`,
//! `live_edit_presence_sweep_lease_total`,
//! `live_edit_presence_stale_removed_total`,
//! `live_edit_presence_stale_removal_delay_seconds` and
//! `live_edit_presence_sweep_oldest_stale_age_seconds` (see their fields on
//! `observability::metrics::Metrics`).

#[cfg(any(test, feature = "test-fault-injection"))]
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::time::MissedTickBehavior;
use tokio_util::sync::CancellationToken;

use crate::observability::metrics::{
    SharedMetrics, PRESENCE_SWEEP_LEASE_LOST, PRESENCE_SWEEP_LEASE_REFRESH_ERROR,
    PRESENCE_SWEEP_LEASE_RENEWED, PRESENCE_SWEEP_OUTCOME_CANCELLED,
    PRESENCE_SWEEP_OUTCOME_COMPLETED, PRESENCE_SWEEP_OUTCOME_ERROR,
    PRESENCE_SWEEP_OUTCOME_LEASE_LOST,
};
use crate::presence::PresenceEvent;
#[cfg(any(test, feature = "test-fault-injection"))]
use crate::presence::PresenceSeamGate;
use crate::storage::client::{FredErrOrigin, ValkeyClient, POOL_LABEL_DATA};
use crate::storage::external_values::{try_acquire_leader_lock, LEADER_LOCK_TTL_MS};
use crate::storage::keys::KeySchema;
use crate::storage::leader_lease::{
    LeaderIdentity, LeaderLease, LeaseOutcome, LeaseState, LostCause,
};
#[cfg(any(test, feature = "test-fault-injection"))]
use crate::storage::presence::ScanPage;
use crate::storage::presence::{
    remove_session_from_index, scan_member_page, scan_session_index_page, sweep_member_batch,
    SWEEP_MEMBER_BATCH, SWEEP_MEMBER_PAGE, SWEEP_SESSION_PAGE,
};

/// Default sweeper interval in seconds (SDD §13.3).
pub const DEFAULT_SWEEPER_INTERVAL_SECS: u64 = 15;

/// The sizes and the lock TTL one sweeper pass works within (SDD SR-3).
///
/// Production passes [`Self::production`]: the 30 s leader-lock TTL and the
/// `storage::presence` page and batch constants. Tests pass small values, to
/// drive a pass across many pages and chunks, or past a third of its TTL.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SweeperBudget {
    /// TTL of the presence-sweeper leader lock, applied by the acquire and by
    /// every renewal. The pass renews it every third of this.
    pub lock_ttl: Duration,
    /// `SSCAN COUNT` hint for the session index. At least 1.
    pub session_page: usize,
    /// `SSCAN COUNT` hint for a session's member set. At least 1.
    pub member_page: usize,
    /// The most connection ids one sweep script call takes. 1 to
    /// `SWEEP_MEMBER_BATCH`.
    pub member_batch: usize,
}

impl SweeperBudget {
    /// The production budget: `LEADER_LOCK_TTL_MS`, `SWEEP_SESSION_PAGE`,
    /// `SWEEP_MEMBER_PAGE` and `SWEEP_MEMBER_BATCH`.
    pub fn production() -> Self {
        Self {
            lock_ttl: Duration::from_millis(LEADER_LOCK_TTL_MS),
            session_page: SWEEP_SESSION_PAGE,
            member_page: SWEEP_MEMBER_PAGE,
            member_batch: SWEEP_MEMBER_BATCH,
        }
    }

    /// The lock TTL in milliseconds, as the lock scripts take it.
    fn lock_ttl_ms(&self) -> u64 {
        u64::try_from(self.lock_ttl.as_millis()).unwrap_or(u64::MAX)
    }

    /// Panic on a budget no pass can run with. Every value is chosen in code
    /// (production's constants or a test's), so a bad one is a caller bug,
    /// caught at spawn rather than as a failure every pass.
    fn assert_valid(&self) {
        assert!(
            self.lock_ttl_ms() > 0,
            "SweeperBudget::lock_ttl must be at least 1 ms, got {:?}",
            self.lock_ttl
        );
        assert!(
            self.session_page > 0 && self.member_page > 0,
            "SweeperBudget page sizes must be at least 1, got {self:?}"
        );
        assert!(
            (1..=SWEEP_MEMBER_BATCH).contains(&self.member_batch),
            "SweeperBudget::member_batch must be 1 to {SWEEP_MEMBER_BATCH}, got {}",
            self.member_batch
        );
    }
}

/// Spawn the crash-recovery presence sweeper as a background task (see the
/// module doc).
///
/// Every `interval` the task runs one tick: the leader probe and, if this
/// replica won the lock, one pass within `budget`. The first tick is one
/// `interval` after spawn. A pass that outlasts `interval` delays the next
/// tick rather than being followed by back-to-back ones
/// (`MissedTickBehavior::Delay`).
///
/// `leader` is the process-wide [`LeaderIdentity`]: its lock value is what the
/// lock stores, so two replicas sharing a service label never both lead.
/// `metrics` takes the sweep families; metric labels carry the service label,
/// never the lock value.
///
/// The task exits when `cancel` fires: at once between ticks, and at the next
/// page or chunk boundary during a pass.
///
/// # Panics
///
/// If `interval` is zero (`Config::from_env` rejects
/// `LIVE_EDIT_SWEEPER_INTERVAL_SECS=0`), or `budget` holds a zero TTL or page
/// size, or a `member_batch` outside 1 to `SWEEP_MEMBER_BATCH`.
pub fn spawn_presence_sweeper(
    client: ValkeyClient,
    key_schema: KeySchema,
    leader: LeaderIdentity,
    budget: SweeperBudget,
    metrics: SharedMetrics,
    interval: Duration,
    cancel: CancellationToken,
) -> tokio::task::JoinHandle<()> {
    let sweeper = Sweeper::new(client, key_schema, leader, budget, metrics, cancel);
    sweeper.spawn(interval)
}

/// **Test-only.** [`spawn_presence_sweeper`] with `seam` installed: every pass
/// parks at its gate and records its observations (see
/// [`PresenceSweeperSeam`]).
#[cfg(any(test, feature = "test-fault-injection"))]
// The seam is the one argument beyond `spawn_presence_sweeper`'s.
#[allow(clippy::too_many_arguments)]
pub fn spawn_presence_sweeper_with_seam(
    client: ValkeyClient,
    key_schema: KeySchema,
    leader: LeaderIdentity,
    budget: SweeperBudget,
    metrics: SharedMetrics,
    interval: Duration,
    cancel: CancellationToken,
    seam: Arc<PresenceSweeperSeam>,
) -> tokio::task::JoinHandle<()> {
    let mut sweeper = Sweeper::new(client, key_schema, leader, budget, metrics, cancel);
    sweeper.seam = Some(seam);
    sweeper.spawn(interval)
}

/// How a leader pass ended; its `outcome` label on
/// `live_edit_presence_sweep_duration_seconds`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PassEnd {
    /// The whole session index was walked.
    Completed,
    /// `ensure_held` reported the lease lost before a chunk.
    LeaseLost,
    /// `cancel` fired; seen at a page or chunk boundary.
    Cancelled,
    /// The session index's `SSCAN` failed.
    Error,
}

impl PassEnd {
    fn label(self) -> &'static str {
        match self {
            Self::Completed => PRESENCE_SWEEP_OUTCOME_COMPLETED,
            Self::LeaseLost => PRESENCE_SWEEP_OUTCOME_LEASE_LOST,
            Self::Cancelled => PRESENCE_SWEEP_OUTCOME_CANCELLED,
            Self::Error => PRESENCE_SWEEP_OUTCOME_ERROR,
        }
    }
}

/// What one chunk did.
enum ChunkEnd {
    /// The script ran; members left in the session's set afterwards.
    Swept { remaining: u64 },
    /// The script call failed: the session's iteration ends, the pass goes on.
    SessionFailed,
    /// The pass ends before this chunk's script.
    Pass(PassEnd),
}

/// One leader pass's state. Lives only for the pass: the lease is dropped at
/// its end and nothing in it is persisted.
struct Pass {
    lease: LeaderLease,
    /// Sessions taken from the index.
    sessions: u64,
    /// Connections removed.
    removed: u64,
    /// The largest known removal delay.
    oldest_delay: Duration,
    /// `sweep_member_batch` calls issued.
    script_calls: usize,
}

/// The sweeper task's fixed inputs.
struct Sweeper {
    client: ValkeyClient,
    keys: KeySchema,
    leader: LeaderIdentity,
    budget: SweeperBudget,
    metrics: SharedMetrics,
    cancel: CancellationToken,
    #[cfg(any(test, feature = "test-fault-injection"))]
    seam: Option<Arc<PresenceSweeperSeam>>,
}

impl Sweeper {
    fn new(
        client: ValkeyClient,
        keys: KeySchema,
        leader: LeaderIdentity,
        budget: SweeperBudget,
        metrics: SharedMetrics,
        cancel: CancellationToken,
    ) -> Self {
        budget.assert_valid();
        Self {
            client,
            keys,
            leader,
            budget,
            metrics,
            cancel,
            #[cfg(any(test, feature = "test-fault-injection"))]
            seam: None,
        }
    }

    fn spawn(self, interval: Duration) -> tokio::task::JoinHandle<()> {
        assert!(
            !interval.is_zero(),
            "the presence sweeper interval must be positive"
        );
        tokio::spawn(self.run(interval))
    }

    async fn run(self, interval: Duration) {
        tracing::info!(
            interval_secs = interval.as_secs(),
            replica_id = %self.leader.service_label,
            lock_value = %self.leader,
            "presence sweeper background task started"
        );
        let mut timer = tokio::time::interval(interval);
        timer.set_missed_tick_behavior(MissedTickBehavior::Delay);
        // An interval's first tick completes at once; the first pass waits
        // one interval.
        timer.tick().await;

        loop {
            // A pass runs in the arm's body, not in the race: it checks
            // `cancel` itself at its page and chunk boundaries, so it is never
            // dropped inside a chunk.
            tokio::select! {
                biased;
                () = self.cancel.cancelled() => {
                    tracing::info!("presence sweeper shutting down");
                    break;
                }
                _ = timer.tick() => {
                    self.sweep_once().await;
                }
            }
        }
    }

    /// One tick: the leader probe and, if this replica won the lock, one pass.
    /// Returns the pass's outcome label, or `None` when no pass ran.
    async fn sweep_once(&self) -> Option<&'static str> {
        // Sampled BEFORE the acquire round-trip, so the lease's local deadline
        // under-estimates the server-side expiry (`LeaderLease::new`).
        let started = Instant::now();
        let lock_key = self.keys.presence_sweeper_leader_lock();
        match try_acquire_leader_lock(
            &self.client,
            &lock_key,
            &self.leader,
            self.budget.lock_ttl_ms(),
        )
        .await
        {
            Ok(true) => {}
            Ok(false) => {
                tracing::debug!(
                    replica_id = %self.leader.service_label,
                    lock_value = %self.leader,
                    "presence sweep skipped: another replica holds the leader lock"
                );
                // A replica that is not the leader publishes 0, so a demoted
                // leader stops exporting the last maximum it observed.
                self.metrics.record_presence_sweep_oldest_stale_age(0.0);
                return None;
            }
            Err(e) => {
                // The wrapper counted a timeout. An unknown leader state
                // publishes nothing.
                tracing::warn!(
                    replica_id = %self.leader.service_label,
                    lock_value = %self.leader,
                    error = %e,
                    "presence sweeper leader lock acquisition failed; skipping tick"
                );
                return None;
            }
        }

        let mut pass = Pass {
            lease: LeaderLease::new(lock_key, self.leader.clone(), started, self.budget.lock_ttl),
            sessions: 0,
            removed: 0,
            oldest_delay: Duration::ZERO,
            script_calls: 0,
        };
        let end = self.walk_index(&mut pass).await;
        let elapsed = started.elapsed();

        self.metrics
            .observe_presence_sweep_duration(end.label(), elapsed.as_secs_f64());
        // Only a pass that walked the whole index publishes its maximum; a
        // partial pass would understate it.
        if end == PassEnd::Completed {
            self.metrics
                .record_presence_sweep_oldest_stale_age(pass.oldest_delay.as_secs_f64());
        }
        #[cfg(any(test, feature = "test-fault-injection"))]
        if let Some(seam) = self.seam.as_deref() {
            seam.pass_ended(end.label(), pass.script_calls);
        }
        tracing::debug!(
            replica_id = %self.leader.service_label,
            outcome = end.label(),
            sessions = pass.sessions,
            removed = pass.removed,
            script_calls = pass.script_calls,
            elapsed = ?elapsed,
            "presence sweep pass ended"
        );
        Some(end.label())
    }

    /// Steps 2–5 of a pass (module doc): page through the session index and
    /// sweep each session.
    async fn walk_index(&self, pass: &mut Pass) -> PassEnd {
        let origin = FredErrOrigin::client(&self.client, POOL_LABEL_DATA);
        let mut cursor: Option<String> = None;
        loop {
            let read = async {
                #[cfg(any(test, feature = "test-fault-injection"))]
                let _in_flight = self
                    .seam
                    .as_deref()
                    .map(PresenceSweeperSeam::command_started);
                scan_session_index_page(
                    &self.client,
                    origin,
                    &self.keys,
                    cursor.as_deref(),
                    self.budget.session_page,
                )
                .await
            };
            // Page boundary. The read is raced, not just preceded, by the
            // check: it is read-only, so a shutdown need not wait out a slow
            // reply.
            let page = tokio::select! {
                biased;
                () = self.cancel.cancelled() => return PassEnd::Cancelled,
                page = read => page,
            };
            let page = match page {
                Ok(page) => page,
                Err(e) => {
                    // `scan_session_index_page` counted a timeout.
                    tracing::warn!(
                        replica_id = %self.leader.service_label,
                        error = %e,
                        "presence sweep: SSCAN of the session index failed; ending the pass"
                    );
                    return PassEnd::Error;
                }
            };
            #[cfg(any(test, feature = "test-fault-injection"))]
            if let Some(seam) = self.seam.as_deref() {
                seam.session_page(page.ids.len());
            }

            for session_id in &page.ids {
                pass.sessions += 1;
                if let Err(end) = self.sweep_session(session_id, page.ids.len(), pass).await {
                    return end;
                }
            }

            match page.next_cursor {
                Some(next) => cursor = Some(next),
                None => return PassEnd::Completed,
            }
        }
    }

    /// Steps 3–5 for one session. `Err` ends the pass; a store error on this
    /// session ends only its iteration and returns `Ok`.
    ///
    /// `session_page_len` is the size of the index page this session came
    /// from, held for the whole session, for the seam's held-ids observation.
    async fn sweep_session(
        &self,
        session_id: &str,
        #[cfg_attr(
            not(any(test, feature = "test-fault-injection")),
            allow(unused_variables)
        )]
        session_page_len: usize,
        pass: &mut Pass,
    ) -> Result<(), PassEnd> {
        let origin = FredErrOrigin::client(&self.client, POOL_LABEL_DATA);
        let mut cursor: Option<String> = None;
        let mut saw_ids = false;
        let mut last_remaining: Option<u64> = None;
        loop {
            let read = async {
                #[cfg(any(test, feature = "test-fault-injection"))]
                let _in_flight = self
                    .seam
                    .as_deref()
                    .map(PresenceSweeperSeam::command_started);
                scan_member_page(
                    &self.client,
                    origin,
                    &self.keys,
                    session_id,
                    cursor.as_deref(),
                    self.budget.member_page,
                )
                .await
            };
            // Page boundary; raced for the same reason as the index read's.
            let page = tokio::select! {
                biased;
                () = self.cancel.cancelled() => return Err(PassEnd::Cancelled),
                page = read => page,
            };
            let page = match page {
                Ok(page) => page,
                Err(e) => {
                    // `scan_member_page` counted a timeout.
                    tracing::warn!(
                        session_id = %session_id,
                        error = %e,
                        "presence sweep: SSCAN of a session's member set failed; skipping the \
                         rest of this session for this pass"
                    );
                    return Ok(());
                }
            };
            // Test-only: the seam may re-deliver an earlier page of this
            // session inside this one (`repeat_next_member_page`), as `SSCAN`
            // does when the set's table shrinks mid-scan. Real re-delivery
            // depends on the server's hash seed and is blocked by a fork child,
            // so a test cannot rely on it. Rewritten before the observation so
            // the observed size is the size chunked.
            #[cfg(any(test, feature = "test-fault-injection"))]
            let page = match self.seam.as_deref() {
                Some(seam) => seam.rewrite_member_page(session_id, page),
                None => page,
            };
            #[cfg(any(test, feature = "test-fault-injection"))]
            if let Some(seam) = self.seam.as_deref() {
                seam.member_page(page.ids.len(), session_page_len + page.ids.len());
            }
            saw_ids |= !page.ids.is_empty();

            for chunk in page.ids.chunks(self.budget.member_batch) {
                match self.sweep_chunk(session_id, chunk, pass).await {
                    ChunkEnd::Swept { remaining } => last_remaining = Some(remaining),
                    ChunkEnd::SessionFailed => return Ok(()),
                    ChunkEnd::Pass(end) => return Err(end),
                }
            }

            match page.next_cursor {
                Some(next) => cursor = Some(next),
                None => break,
            }
        }

        // Step 5. `last_remaining` is the last chunk's count: the set as of
        // the iteration's end. An empty, indexed session is one whose join
        // failed or crashed between indexing and registering.
        if last_remaining == Some(0) || !saw_ids {
            #[cfg(any(test, feature = "test-fault-injection"))]
            let _in_flight = self
                .seam
                .as_deref()
                .map(PresenceSweeperSeam::command_started);
            match remove_session_from_index(&self.client, origin, &self.keys, session_id).await {
                Ok(false) => tracing::debug!(
                    session_id = %session_id,
                    "presence sweep: removed an empty session from the session index"
                ),
                Ok(true) => tracing::debug!(
                    session_id = %session_id,
                    "presence sweep: an emptied session gained a member meanwhile; it stays \
                     in the session index"
                ),
                // The helper counted a timeout. The entry is re-examined next
                // pass; a member it left un-indexed is re-added by its keeper,
                // or, if its replica is gone, by the orphan sweep's index
                // backstop.
                Err(e) => tracing::warn!(
                    session_id = %session_id,
                    error = %e,
                    "presence sweep: removing an empty session from the session index failed"
                ),
            }
        }
        Ok(())
    }

    /// Step 4 for one chunk of at most `member_batch` connection ids.
    async fn sweep_chunk(&self, session_id: &str, chunk: &[String], pass: &mut Pass) -> ChunkEnd {
        #[cfg(any(test, feature = "test-fault-injection"))]
        if let Some(seam) = self.seam.as_deref() {
            seam.before_chunk.arrive().await;
        }

        // Chunk boundary: the last point at which the pass may stop before
        // this chunk. Nothing is in flight here, so there is nothing to race.
        if self.cancel.is_cancelled() {
            return ChunkEnd::Pass(PassEnd::Cancelled);
        }

        let check = pass.lease.ensure_held(&self.client).await;
        self.count_lease(check.outcome);
        if check.state == LeaseState::Lost {
            self.warn_lease_lost(&pass.lease, check.cause);
            return ChunkEnd::Pass(PassEnd::LeaseLost);
        }

        let left_payloads: Vec<String> = chunk
            .iter()
            .map(|cid| {
                PresenceEvent::PeerLeft {
                    connection_id: cid.clone(),
                }
                .encode()
            })
            .collect();
        pass.script_calls += 1;
        #[cfg(any(test, feature = "test-fault-injection"))]
        if let Some(seam) = self.seam.as_deref() {
            seam.script_call(chunk.len());
        }
        let result = {
            #[cfg(any(test, feature = "test-fault-injection"))]
            let _in_flight = self
                .seam
                .as_deref()
                .map(PresenceSweeperSeam::command_started);
            sweep_member_batch(
                &self.client,
                FredErrOrigin::client(&self.client, POOL_LABEL_DATA),
                &self.keys,
                session_id,
                chunk,
                &left_payloads,
            )
            .await
        };
        match result {
            Ok(batch) => {
                self.record_removals(session_id, &batch.removed, pass);
                ChunkEnd::Swept {
                    remaining: batch.remaining,
                }
            }
            Err(e) => {
                // `sweep_member_batch` counted a timeout. Nothing is
                // published here: a script that ran despite the error already
                // published its own `left`s.
                tracing::warn!(
                    session_id = %session_id,
                    chunk_len = chunk.len(),
                    error = %e,
                    "presence sweep: a sweep script call failed; skipping the rest of this \
                     session for this pass"
                );
                ChunkEnd::SessionFailed
            }
        }
    }

    /// Count and log one chunk's removals, and fold their delays into the
    /// pass's maximum.
    fn record_removals(
        &self,
        session_id: &str,
        removed: &[(String, Option<Duration>)],
        pass: &mut Pass,
    ) {
        if removed.is_empty() {
            return;
        }
        let n = u64::try_from(removed.len()).unwrap_or(u64::MAX);
        self.metrics.add_presence_stale_removed(n);
        pass.removed += n;
        for (connection_id, delay) in removed {
            if let Some(delay) = delay {
                self.metrics
                    .observe_presence_stale_removal_delay(delay.as_secs_f64());
                pass.oldest_delay = pass.oldest_delay.max(*delay);
            }
            tracing::info!(
                session_id = %session_id,
                connection_id = %connection_id,
                removal_delay = ?delay,
                "presence sweeper removed a stale connection (lease expired) and announced its \
                 departure"
            );
        }
    }

    /// Count a lease decision on `live_edit_presence_sweep_lease_total`.
    /// `RefreshError` counts on both `refresh_error` and `lost`, so `lost` is
    /// the number of passes a lease loss ended.
    fn count_lease(&self, outcome: Option<LeaseOutcome>) {
        match outcome {
            None => {}
            Some(LeaseOutcome::Renewed) => {
                self.metrics
                    .inc_presence_sweep_lease(PRESENCE_SWEEP_LEASE_RENEWED);
            }
            Some(LeaseOutcome::Lost) => {
                self.metrics
                    .inc_presence_sweep_lease(PRESENCE_SWEEP_LEASE_LOST);
            }
            Some(LeaseOutcome::RefreshError) => {
                self.metrics
                    .inc_presence_sweep_lease(PRESENCE_SWEEP_LEASE_REFRESH_ERROR);
                self.metrics
                    .inc_presence_sweep_lease(PRESENCE_SWEEP_LEASE_LOST);
            }
        }
    }

    /// The pass's one WARN for a lost lease. A lost lease ends the pass, so
    /// this runs at most once per pass.
    fn warn_lease_lost(&self, lease: &LeaderLease, cause: Option<LostCause>) {
        let lock_key = lease.lock_key();
        let leader = lease.identity();
        match cause {
            Some(LostCause::DeadlineOverrun) => tracing::warn!(
                lock_key = %lock_key,
                replica_id = %leader.service_label,
                lock_value = %leader,
                "presence sweep pass overran its leader lease before a refresh could land; \
                 ending the pass before its next chunk (a lapsed lease cannot be re-proven, \
                 only re-acquired next tick)"
            ),
            Some(LostCause::NotHeld) => tracing::warn!(
                lock_key = %lock_key,
                replica_id = %leader.service_label,
                lock_value = %leader,
                "presence sweeper leader lease is no longer held by this replica (key lapsed \
                 or claimed by a peer); ending the pass before its next chunk"
            ),
            Some(LostCause::StoreError(e)) => tracing::warn!(
                lock_key = %lock_key,
                replica_id = %leader.service_label,
                lock_value = %leader,
                error = %e,
                "presence sweeper leader lease refresh failed, so the lease cannot be proven \
                 held; ending the pass before its next chunk"
            ),
            // `LeaseCheck::cause` is `Some` whenever the state is `Lost`.
            None => tracing::warn!(
                lock_key = %lock_key,
                replica_id = %leader.service_label,
                lock_value = %leader,
                "presence sweeper leader lease lost; ending the pass before its next chunk"
            ),
        }
    }
}

/// Test-only park point in, and observations of, the presence sweeper's
/// passes, plus a one-shot member-page replay
/// ([`Self::repeat_next_member_page`]). Installed with
/// [`spawn_presence_sweeper_with_seam`]; the gate starts disarmed (see
/// [`PresenceSeamGate`]) and the replay off.
///
/// **Observations** accumulate across every pass of the sweeper the seam is
/// installed on; nothing clears them. A "presence command" is one data-pool
/// command of a pass: an index or member-set `SSCAN` page, a
/// `sweep_member_batch` script call, or a `remove_session_from_index` call
/// (its three commands are issued one after another, so it counts as one).
/// The leader probe and lease refreshes run on the maintenance pool and are
/// not presence commands.
///
/// **Test-only.** Gated behind `#[cfg(any(test, feature =
/// "test-fault-injection"))]`, as is the spawn function that installs it; the
/// release binary has neither.
#[cfg(any(test, feature = "test-fault-injection"))]
pub struct PresenceSweeperSeam {
    /// First in each chunk's order — this park, then the cancellation check,
    /// then `ensure_held`, then the chunk's script — so a test can park a
    /// chunk and cancel it before its script runs, or hold it past a third of
    /// the lock TTL so that chunk's `ensure_held` renews (or, past the TTL,
    /// finds the lease lost). Reached once per chunk, including a chunk that
    /// is about to stop.
    pub before_chunk: PresenceSeamGate,
    /// Presence commands in flight now.
    commands_in_flight: std::sync::atomic::AtomicUsize,
    /// The most presence commands ever in flight at once.
    max_commands_in_flight: std::sync::atomic::AtomicUsize,
    /// Connection ids per script call, in call order.
    script_batch_sizes: std::sync::Mutex<Vec<usize>>,
    /// Ids per session-index page, in read order.
    session_page_sizes: std::sync::Mutex<Vec<usize>>,
    /// Ids per member-set page, in read order, replayed ids included.
    member_page_sizes: std::sync::Mutex<Vec<usize>>,
    /// The most ids a pass has held at once: one index page plus one member
    /// page, replayed ids included.
    max_ids_held: std::sync::atomic::AtomicUsize,
    /// Every leader pass that ended, in order.
    passes: tokio::sync::watch::Sender<Vec<ObservedSweepPass>>,
    /// The one-shot member-page replay ([`Self::repeat_next_member_page`]).
    member_page_replay: std::sync::Mutex<MemberPageReplay>,
}

/// Where [`PresenceSweeperSeam`]'s one-shot member-page replay stands.
#[cfg(any(test, feature = "test-fault-injection"))]
#[derive(Debug, Default)]
enum MemberPageReplay {
    /// Never armed: every page passes through.
    #[default]
    Off,
    /// Armed: the next member page read is copied and passed through.
    Armed,
    /// Holding a copy of the page read while armed, and the session it came
    /// from, to prepend to that session's next page.
    Holding {
        session_id: String,
        ids: Vec<String>,
    },
    /// Spent — replayed, or abandoned: every page passes through.
    Done,
}

/// One leader pass, as [`PresenceSweeperSeam`] observes it.
#[cfg(any(test, feature = "test-fault-injection"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObservedSweepPass {
    /// How it ended: its `outcome` label on
    /// `live_edit_presence_sweep_duration_seconds`, one of
    /// `observability::metrics::PRESENCE_SWEEP_OUTCOMES`.
    pub outcome: &'static str,
    /// `sweep_member_batch` calls it issued.
    pub script_calls: usize,
}

#[cfg(any(test, feature = "test-fault-injection"))]
impl Default for PresenceSweeperSeam {
    fn default() -> Self {
        Self {
            before_chunk: PresenceSeamGate::new(),
            commands_in_flight: std::sync::atomic::AtomicUsize::new(0),
            max_commands_in_flight: std::sync::atomic::AtomicUsize::new(0),
            script_batch_sizes: std::sync::Mutex::new(Vec::new()),
            session_page_sizes: std::sync::Mutex::new(Vec::new()),
            member_page_sizes: std::sync::Mutex::new(Vec::new()),
            max_ids_held: std::sync::atomic::AtomicUsize::new(0),
            passes: tokio::sync::watch::channel(Vec::new()).0,
            member_page_replay: std::sync::Mutex::new(MemberPageReplay::Off),
        }
    }
}

#[cfg(any(test, feature = "test-fault-injection"))]
impl PresenceSweeperSeam {
    /// The gate disarmed and nothing observed.
    pub fn new() -> Self {
        Self::default()
    }

    /// Test side: the most presence commands the sweeper has had in flight at
    /// once.
    pub fn max_commands_in_flight(&self) -> usize {
        self.max_commands_in_flight
            .load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Test side: the connection ids each script call carried, in call order.
    pub fn script_batch_sizes(&self) -> Vec<usize> {
        lock(&self.script_batch_sizes).clone()
    }

    /// Test side: the ids each session-index page returned, in read order.
    pub fn session_page_sizes(&self) -> Vec<usize> {
        lock(&self.session_page_sizes).clone()
    }

    /// Test side: the ids each member-set page returned, in read order. A page
    /// the replay rewrote counts its replayed ids.
    pub fn member_page_sizes(&self) -> Vec<usize> {
        lock(&self.member_page_sizes).clone()
    }

    /// Test side: the most ids a pass has held at once — the session-index
    /// page it was on plus the member page it was chunking. A page the replay
    /// rewrote counts its replayed ids, so a replay can lift this past the
    /// production bound (one index page plus one member page); a test that
    /// arms the replay must allow for that.
    pub fn max_ids_held(&self) -> usize {
        self.max_ids_held.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Test side: deliver the next member page twice within its session's
    /// iteration, as `SSCAN` may when a set's table shrinks mid-scan. The next
    /// member page read is passed through unchanged and a copy of its ids is
    /// held; the next page of the **same session in the same pass** is
    /// delivered with the held ids prepended (its cursor unchanged), so its
    /// chunks re-run members the first page's chunks may already have
    /// removed. One-shot: after that every page passes through again.
    ///
    /// The replay is abandoned, with nothing prepended, if the held page's
    /// session reads no further page in that pass: if a page of a different
    /// session arrives first, or the pass ends first. Either way the page
    /// sizes show it — no page is inflated. A test that needs the replay must
    /// therefore give the first session swept at least two member pages.
    /// Calling this again once armed, holding or spent has no effect.
    ///
    /// Real re-delivery cannot be relied on in a test: whether a shrink makes
    /// `SSCAN` repeat a page depends on the server's hash seed, and a fork
    /// child (RDB/AOF rewrite) blocks the shrink.
    pub fn repeat_next_member_page(&self) {
        let mut replay = lock(&self.member_page_replay);
        if matches!(*replay, MemberPageReplay::Off) {
            *replay = MemberPageReplay::Armed;
        }
    }

    /// Test side: every leader pass that has ended, in order.
    pub fn passes(&self) -> Vec<ObservedSweepPass> {
        self.passes.borrow().clone()
    }

    /// Test side: block until at least `n` leader passes have ended, and
    /// return them all. Level-triggered.
    pub async fn wait_for_passes(&self, n: usize) -> Vec<ObservedSweepPass> {
        let mut passes = self.passes.subscribe();
        let passes = passes
            .wait_for(|passes| passes.len() >= n)
            .await
            .expect("the seam holds the passes sender, so the channel cannot close");
        passes.clone()
    }

    /// Production side: a presence command is being sent. Holds an in-flight
    /// slot until the returned guard drops — when the command completes, or a
    /// cancel race drops it.
    fn command_started(&self) -> SweepCommandInFlight<'_> {
        use std::sync::atomic::Ordering::SeqCst;
        let now = self.commands_in_flight.fetch_add(1, SeqCst) + 1;
        self.max_commands_in_flight.fetch_max(now, SeqCst);
        SweepCommandInFlight(&self.commands_in_flight)
    }

    /// Production side: a script call carrying `ids` connection ids.
    fn script_call(&self, ids: usize) {
        lock(&self.script_batch_sizes).push(ids);
    }

    /// Production side: a session-index page of `ids` ids.
    fn session_page(&self, ids: usize) {
        lock(&self.session_page_sizes).push(ids);
        self.max_ids_held
            .fetch_max(ids, std::sync::atomic::Ordering::SeqCst);
    }

    /// Production side: a member-set page of `ids` ids, with `held` ids held
    /// in all (this page plus its session-index page).
    fn member_page(&self, ids: usize, held: usize) {
        lock(&self.member_page_sizes).push(ids);
        self.max_ids_held
            .fetch_max(held, std::sync::atomic::Ordering::SeqCst);
    }

    /// Production side: `page`, a member-set page of `session_id` just read,
    /// as the sweep is to chunk it — rewritten when the replay
    /// ([`Self::repeat_next_member_page`]) is due.
    fn rewrite_member_page(&self, session_id: &str, mut page: ScanPage) -> ScanPage {
        let mut replay = lock(&self.member_page_replay);
        match std::mem::take(&mut *replay) {
            MemberPageReplay::Armed => {
                *replay = MemberPageReplay::Holding {
                    session_id: session_id.to_owned(),
                    ids: page.ids.clone(),
                };
            }
            MemberPageReplay::Holding {
                session_id: held_session,
                mut ids,
            } => {
                // A different session's page abandons the replay: the held
                // page's session iteration has ended.
                if held_session == session_id {
                    ids.append(&mut page.ids);
                    page.ids = ids;
                }
                *replay = MemberPageReplay::Done;
            }
            state @ (MemberPageReplay::Off | MemberPageReplay::Done) => *replay = state,
        }
        page
    }

    /// Production side: a leader pass ended. A replay still holding a page
    /// is abandoned, so a held page is never re-delivered into a later pass.
    fn pass_ended(&self, outcome: &'static str, script_calls: usize) {
        {
            let mut replay = lock(&self.member_page_replay);
            if matches!(*replay, MemberPageReplay::Holding { .. }) {
                *replay = MemberPageReplay::Done;
            }
        }
        self.passes.send_modify(|passes| {
            passes.push(ObservedSweepPass {
                outcome,
                script_calls,
            });
        });
    }
}

/// Lock an observation, ignoring poisoning: a panicking test thread must not
/// hide the observations from the assertion that explains it.
#[cfg(any(test, feature = "test-fault-injection"))]
fn lock<T>(m: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// One presence command in flight, for [`PresenceSweeperSeam`]'s high-water
/// mark; releases its slot on drop.
#[cfg(any(test, feature = "test-fault-injection"))]
struct SweepCommandInFlight<'a>(&'a std::sync::atomic::AtomicUsize);

#[cfg(any(test, feature = "test-fault-injection"))]
impl Drop for SweepCommandInFlight<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    //! One tick at a time through [`Sweeper::sweep_once`] against a live
    //! `TestValkey`, so each test reads exactly one pass's outcome, metrics and
    //! store effect. A test that holds a pass mid-way installs a
    //! [`PresenceSweeperSeam`] and runs the tick on a task of its own.
    //!
    //! The tests that need a chunk's `ensure_held` to refresh hold the chunk
    //! for [`HOLD_PAST_RENEWAL`] on purpose: the lease's renewal window is
    //! measured on `std::time::Instant`, which `tokio::time::pause` cannot move.
    //! Every other wait is a bounded wait on a seam latch or the tick's own
    //! return.
    //!
    //! The seam's member-page replay and the budget checks are tested without
    //! a store.

    use std::sync::Arc;

    use fred::interfaces::{HashesInterface, KeysInterface, ListInterface, SetsInterface};

    use super::*;
    use crate::observability::metrics::test_util::{
        has_metric, histogram_bucket, render, sample_value,
    };
    use crate::observability::metrics::{metrics_scrape_only, Metrics};
    use crate::storage::presence::store_time_ms;
    use crate::storage::test_support::{
        command_timeout_row, command_timeouts_only_on, release_all, TestValkey,
    };

    /// The `service` label of every test's metrics.
    const SERVICE: &str = "sweeper";

    /// How long before the seed a [`Seed::Stale`] member's recorded lease
    /// expired: every removal delay the sweep reports for one is at least this.
    const STALE_FOR_MS: u64 = 7_000;

    /// The lock TTL of the tests that hold a chunk past a third of it.
    const SHORT_LOCK_TTL: Duration = Duration::from_secs(3);

    /// A hold past a third of [`SHORT_LOCK_TTL`] (1 s), so the held chunk's
    /// `ensure_held` is due to refresh: the lease's clock starts before the
    /// leader probe, so the hold alone puts the check past the third. The
    /// slack under the whole TTL — 1.8 s for the probe, the two page reads
    /// before the park, the park handoff and the sleep's overshoot — keeps
    /// the lease's local deadline unpassed, so the check refreshes rather
    /// than finding the lease overrun.
    const HOLD_PAST_RENEWAL: Duration = Duration::from_millis(1_200);

    /// Bound on every wait. Unreachable on the healthy path.
    const SETTLE: Duration = Duration::from_secs(10);

    const OLDEST_AGE: &str = "live_edit_presence_sweep_oldest_stale_age_seconds";
    const REMOVAL_DELAY: &str = "live_edit_presence_stale_removal_delay_seconds";

    /// How [`seed`] writes one member.
    #[derive(Clone, Copy)]
    enum Seed {
        /// A lease (600 s) and an info hash recording an expiry 600 s ahead.
        Live,
        /// No lease, and an info hash recording a lease that expired
        /// [`STALE_FOR_MS`] ago: a connection whose replica crashed.
        Stale,
        /// No lease and no info hash, so no recorded expiry.
        StaleUnrecorded,
    }

    /// Write `members` into `sid`'s presence state straight through the store,
    /// as joins on a replica that then crashed (or still lives) would have
    /// left them, and index the session. The store clock is read on the data
    /// pool; the writes go through the maintenance pool. A test that stalls
    /// the data pool seeds first.
    async fn seed(tv: &TestValkey, sid: &str, members: &[(&str, Seed)]) {
        let ks = tv.key_schema();
        let now_ms = store_time_ms(
            tv.client(),
            FredErrOrigin::client(tv.client(), POOL_LABEL_DATA),
            ks,
            sid,
        )
        .await
        .expect("store_time_ms must succeed");
        let conn = tv.client().maintenance_pool.next();
        for (cid, seed) in members {
            let _: i64 = conn
                .sadd(ks.presence_members(sid), *cid)
                .await
                .expect("SADD member must succeed");
            let expires_at_ms = match seed {
                Seed::Live => Some(now_ms + 600_000),
                Seed::Stale => Some(now_ms - STALE_FOR_MS),
                Seed::StaleUnrecorded => None,
            };
            if let Some(expires_at_ms) = expires_at_ms {
                let _: i64 = conn
                    .hset(
                        ks.presence_user(sid, cid),
                        vec![
                            ("user_name", "Seeded".to_owned()),
                            ("user_application", "seed-app".to_owned()),
                            ("lease_expires_at_ms", expires_at_ms.to_string()),
                        ],
                    )
                    .await
                    .expect("HSET info must succeed");
            }
            if matches!(seed, Seed::Live) {
                let _: () = conn
                    .set(
                        ks.presence_lease(sid, cid),
                        "1",
                        Some(fred::types::Expiration::EX(600)),
                        None,
                        false,
                    )
                    .await
                    .expect("SET lease must succeed");
            }
        }
        let _: i64 = conn
            .sadd(ks.presence_active_sessions(), sid)
            .await
            .expect("SADD active_sessions must succeed");
    }

    async fn is_member(tv: &TestValkey, sid: &str, cid: &str) -> bool {
        tv.client()
            .maintenance_pool
            .next()
            .sismember(tv.key_schema().presence_members(sid), cid)
            .await
            .expect("SISMEMBER must succeed")
    }

    /// A budget with small pages and the given lock TTL and chunk size.
    fn budget(lock_ttl: Duration, member_batch: usize) -> SweeperBudget {
        SweeperBudget {
            lock_ttl,
            session_page: 10,
            member_page: 10,
            member_batch,
        }
    }

    /// The leader identity every test's sweeper runs as.
    fn ours() -> LeaderIdentity {
        LeaderIdentity::new(SERVICE, "instance-ours")
    }

    /// A sweeper over `tv` with a [`PresenceSweeperSeam`] installed.
    fn seamed(
        tv: &TestValkey,
        budget: SweeperBudget,
        metrics: &SharedMetrics,
        cancel: CancellationToken,
    ) -> (Arc<Sweeper>, Arc<PresenceSweeperSeam>) {
        let seam = Arc::new(PresenceSweeperSeam::new());
        let mut sweeper = Sweeper::new(
            tv.client().clone(),
            tv.key_schema().clone(),
            ours(),
            budget,
            Arc::clone(metrics),
            cancel,
        );
        sweeper.seam = Some(Arc::clone(&seam));
        (Arc::new(sweeper), seam)
    }

    /// Run one tick on its own task.
    fn tick(sweeper: &Arc<Sweeper>) -> tokio::task::JoinHandle<Option<&'static str>> {
        let sweeper = Arc::clone(sweeper);
        tokio::spawn(async move { sweeper.sweep_once().await })
    }

    async fn tick_outcome(
        task: tokio::task::JoinHandle<Option<&'static str>>,
    ) -> Option<&'static str> {
        tokio::time::timeout(SETTLE, task)
            .await
            .expect("the tick must end once released")
            .expect("the tick must not panic")
    }

    async fn wait_parked(seam: &PresenceSweeperSeam) {
        tokio::time::timeout(SETTLE, seam.before_chunk.wait_until_parked())
            .await
            .expect("the pass must park at its first chunk");
    }

    /// `live_edit_presence_sweep_lease_total{outcome}`. Every cell is seeded.
    fn lease_total(metrics: &Metrics, outcome: &str) -> f64 {
        sample_value(
            &render(metrics),
            "live_edit_presence_sweep_lease_total",
            &[("outcome", outcome), ("service", SERVICE)],
        )
        .expect("every sweep lease cell is seeded")
    }

    /// `[renewed, lost, refresh_error]` on `live_edit_presence_sweep_lease_total`.
    fn lease_row(metrics: &Metrics) -> [f64; 3] {
        [
            lease_total(metrics, PRESENCE_SWEEP_LEASE_RENEWED),
            lease_total(metrics, PRESENCE_SWEEP_LEASE_LOST),
            lease_total(metrics, PRESENCE_SWEEP_LEASE_REFRESH_ERROR),
        ]
    }

    /// Passes recorded on `live_edit_presence_sweep_duration_seconds` with
    /// `outcome`; `None` while that series has no observation.
    fn passes_ended(metrics: &Metrics, outcome: &str) -> Option<f64> {
        sample_value(
            &render(metrics),
            "live_edit_presence_sweep_duration_seconds_count",
            &[("outcome", outcome), ("service", SERVICE)],
        )
    }

    fn stale_removed(metrics: &Metrics) -> f64 {
        sample_value(
            &render(metrics),
            "live_edit_presence_stale_removed_total",
            &[("service", SERVICE)],
        )
        .expect("the stale-removed cell is seeded")
    }

    /// A pass whose session-index read gets no reply within the command
    /// timeout counts exactly one
    /// `valkey_errors_total{pool="data", kind="command_timeout"}` and ends with
    /// the `error` outcome. The read is held behind a server-side `BLPOP` on
    /// every data-pool connection, so the `Timeout` is fred's real one; the
    /// leader probe runs on the maintenance pool, which is not held.
    ///
    /// SR-3.4: an error on the session index's `SSCAN` ends the pass — not
    /// just a session — so nothing is swept: the stale member it would have
    /// reached stays, no script is called, the pass is recorded with the
    /// `error` outcome, and, the pass not having completed, no oldest stale
    /// age is published.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_timed_out_sweep_read_counts_one_data_command_timeout() {
        let tv = TestValkey::setup().await;
        seed(&tv, "s", &[("stale", Seed::Stale)]).await;
        let metrics: SharedMetrics = Arc::new(metrics_scrape_only(SERVICE));
        tv.client()
            .install_observability_hooks(Arc::clone(&metrics))
            .await;
        let (sweeper, seam) = seamed(
            &tv,
            SweeperBudget::production(),
            &metrics,
            CancellationToken::new(),
        );

        let stalls = tv.stall_data_pool().await;
        let outcome = sweeper.sweep_once().await;
        release_all(stalls).await;

        assert_eq!(
            outcome,
            Some(PRESENCE_SWEEP_OUTCOME_ERROR),
            "a failed index read ends the pass with the error outcome"
        );
        assert_eq!(
            command_timeout_row(&metrics),
            command_timeouts_only_on("data", 1),
            "the timed-out SSCAN must count once, on the data pool"
        );
        assert_eq!(
            seam.passes(),
            vec![ObservedSweepPass {
                outcome: PRESENCE_SWEEP_OUTCOME_ERROR,
                script_calls: 0,
            }]
        );
        assert!(
            seam.session_page_sizes().is_empty() && seam.member_page_sizes().is_empty(),
            "no page was read"
        );
        assert!(
            is_member(&tv, "s", "stale").await,
            "a pass whose index read failed must sweep nothing"
        );
        assert_eq!(
            passes_ended(&metrics, PRESENCE_SWEEP_OUTCOME_ERROR),
            Some(1.0)
        );
        assert_eq!(stale_removed(&metrics), 0.0);
        assert!(
            !has_metric(&render(&metrics), OLDEST_AGE),
            "a pass that did not complete must publish no oldest stale age (unknown is not 0)"
        );
    }

    /// SR-6: a completed pass counts every removal on
    /// `live_edit_presence_stale_removed_total`, observes the removal delay of
    /// each removal whose info hash recorded its expiry (and only those) on
    /// `live_edit_presence_stale_removal_delay_seconds`, and publishes the
    /// largest delay as the oldest stale age. A live member is not removed,
    /// and its session, not emptied, stays indexed.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_pass_with_removals_counts_them_and_observes_each_recorded_delay() {
        let tv = TestValkey::setup().await;
        seed(
            &tv,
            "s",
            &[
                ("stale-a", Seed::Stale),
                ("stale-b", Seed::Stale),
                ("stale-unrecorded", Seed::StaleUnrecorded),
                ("live", Seed::Live),
            ],
        )
        .await;
        let metrics: SharedMetrics = Arc::new(metrics_scrape_only(SERVICE));
        let (sweeper, _seam) = seamed(
            &tv,
            budget(SweeperBudget::production().lock_ttl, 2),
            &metrics,
            CancellationToken::new(),
        );

        assert_eq!(
            sweeper.sweep_once().await,
            Some(PRESENCE_SWEEP_OUTCOME_COMPLETED)
        );

        for cid in ["stale-a", "stale-b", "stale-unrecorded"] {
            assert!(!is_member(&tv, "s", cid).await, "{cid} must be removed");
        }
        assert!(is_member(&tv, "s", "live").await, "the live member stays");
        let indexed: bool = tv
            .client()
            .maintenance_pool
            .next()
            .sismember(tv.key_schema().presence_active_sessions(), "s")
            .await
            .expect("SISMEMBER active_sessions must succeed");
        assert!(indexed, "a session that still has a member stays indexed");

        let text = render(&metrics);
        let service = [("service", SERVICE)];
        assert_eq!(stale_removed(&metrics), 3.0, "every removal is counted");
        assert_eq!(
            sample_value(&text, &format!("{REMOVAL_DELAY}_count"), &service),
            Some(2.0),
            "a delay is observed for each removal with a recorded expiry, and only those"
        );
        let stale_for_secs = (STALE_FOR_MS / 1000) as f64;
        assert_eq!(
            histogram_bucket(&text, REMOVAL_DELAY, "5", &service),
            Some(0.0),
            "both recorded delays are at least {stale_for_secs} s"
        );
        let oldest = sample_value(&text, OLDEST_AGE, &service)
            .expect("a completed pass publishes its oldest stale age");
        assert!(
            (stale_for_secs..60.0).contains(&oldest),
            "the oldest stale age is the largest delay, at least {stale_for_secs} s; got {oldest}"
        );
        assert_eq!(
            passes_ended(&metrics, PRESENCE_SWEEP_OUTCOME_COMPLETED),
            Some(1.0)
        );
        assert_eq!(
            lease_row(&metrics),
            [0.0, 0.0, 0.0],
            "a pass shorter than a third of the lock TTL decides nothing about its lease"
        );
    }

    /// A replica that loses the leader probe runs no pass and publishes 0 as
    /// the oldest stale age, replacing whatever maximum it published while it
    /// led.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_tick_that_loses_the_leader_probe_publishes_zero_oldest_age_and_runs_no_pass() {
        let tv = TestValkey::setup().await;
        seed(&tv, "s", &[("stale", Seed::Stale)]).await;
        let peer = LeaderIdentity::new(SERVICE, "instance-peer");
        let _: () = tv
            .client()
            .maintenance_pool
            .next()
            .set(
                tv.key_schema().presence_sweeper_leader_lock(),
                peer.lock_value(),
                Some(fred::types::Expiration::PX(30_000)),
                None,
                false,
            )
            .await
            .expect("planting the peer's lock must succeed");
        let metrics: SharedMetrics = Arc::new(metrics_scrape_only(SERVICE));
        metrics.record_presence_sweep_oldest_stale_age(30.0);
        let (sweeper, seam) = seamed(
            &tv,
            SweeperBudget::production(),
            &metrics,
            CancellationToken::new(),
        );

        assert_eq!(
            sweeper.sweep_once().await,
            None,
            "a non-leader runs no pass"
        );

        let text = render(&metrics);
        assert_eq!(
            sample_value(&text, OLDEST_AGE, &[("service", SERVICE)]),
            Some(0.0),
            "a non-leader publishes 0, so a demoted leader stops exporting its last maximum"
        );
        assert!(
            !has_metric(&text, "live_edit_presence_sweep_duration_seconds"),
            "a non-leader records no pass"
        );
        assert!(seam.passes().is_empty() && seam.session_page_sizes().is_empty());
        assert!(is_member(&tv, "s", "stale").await);
    }

    /// A leader probe that errors leaves the leader state unknown: the tick
    /// runs no pass and publishes nothing, so the oldest stale age this
    /// replica published before stays as it was (unknown is not 0). The error
    /// is forced by making the lock key a LIST, which the acquire script's
    /// `GET` rejects.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_tick_whose_leader_probe_errors_runs_no_pass_and_leaves_the_oldest_age_alone() {
        let tv = TestValkey::setup().await;
        seed(&tv, "s", &[("stale", Seed::Stale)]).await;
        let conn = tv.client().maintenance_pool.next();
        let _: i64 = conn
            .rpush(tv.key_schema().presence_sweeper_leader_lock(), "tripwire")
            .await
            .expect("RPUSH tripwire must succeed");
        let metrics: SharedMetrics = Arc::new(metrics_scrape_only(SERVICE));
        metrics.record_presence_sweep_oldest_stale_age(30.0);
        let (sweeper, seam) = seamed(
            &tv,
            SweeperBudget::production(),
            &metrics,
            CancellationToken::new(),
        );

        assert_eq!(
            sweeper.sweep_once().await,
            None,
            "a tick whose leader probe failed runs no pass"
        );

        let text = render(&metrics);
        assert_eq!(
            sample_value(&text, OLDEST_AGE, &[("service", SERVICE)]),
            Some(30.0),
            "a failed probe must leave the published oldest stale age as it was, not 0"
        );
        assert!(
            !has_metric(&text, "live_edit_presence_sweep_duration_seconds"),
            "a failed probe records no pass"
        );
        assert!(seam.passes().is_empty() && seam.session_page_sizes().is_empty());
        assert!(is_member(&tv, "s", "stale").await);
    }

    /// PRD FR-7: a chunk held at the seam past a third of the lock TTL, but
    /// under all of it, renews the lease at that chunk's `ensure_held`
    /// (`renewed`), and the pass goes on to complete and remove every stale
    /// member. The lock is still this process's afterwards.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_chunk_held_past_a_third_of_the_lock_ttl_renews_the_lease_and_the_pass_completes() {
        let tv = TestValkey::setup().await;
        seed(
            &tv,
            "s",
            &[("stale-1", Seed::Stale), ("stale-2", Seed::Stale)],
        )
        .await;
        let metrics: SharedMetrics = Arc::new(metrics_scrape_only(SERVICE));
        let (sweeper, seam) = seamed(
            &tv,
            budget(SHORT_LOCK_TTL, 1),
            &metrics,
            CancellationToken::new(),
        );

        seam.before_chunk.arm();
        let task = tick(&sweeper);
        wait_parked(&seam).await;
        tokio::time::sleep(HOLD_PAST_RENEWAL).await;
        seam.before_chunk.release();

        assert_eq!(
            tick_outcome(task).await,
            Some(PRESENCE_SWEEP_OUTCOME_COMPLETED)
        );
        let [renewed, lost, refresh_error] = lease_row(&metrics);
        assert!(
            renewed >= 1.0,
            "the chunk held past a third of the TTL must renew the lease; renewed = {renewed}"
        );
        assert_eq!((lost, refresh_error), (0.0, 0.0));
        assert_eq!(seam.script_batch_sizes(), vec![1, 1]);
        for cid in ["stale-1", "stale-2"] {
            assert!(!is_member(&tv, "s", cid).await, "{cid} must be removed");
        }
        let holder: Option<String> = tv
            .client()
            .maintenance_pool
            .next()
            .get(tv.key_schema().presence_sweeper_leader_lock())
            .await
            .expect("GET lock must succeed");
        assert_eq!(holder, Some(ours().lock_value()));
    }

    /// Hold a pass at its first chunk, run `steal` against the lock key while
    /// it is parked, hold it past a third of the TTL so the chunk's
    /// `ensure_held` refreshes, release it, and return the tick's outcome and
    /// the pass's seam.
    async fn lose_the_lease_at_a_parked_chunk(
        tv: &TestValkey,
        metrics: &SharedMetrics,
        steal: impl std::future::Future<Output = ()>,
    ) -> (Option<&'static str>, Arc<PresenceSweeperSeam>) {
        let (sweeper, seam) = seamed(
            tv,
            budget(SHORT_LOCK_TTL, 1),
            metrics,
            CancellationToken::new(),
        );
        seam.before_chunk.arm();
        let task = tick(&sweeper);
        wait_parked(&seam).await;
        steal.await;
        tokio::time::sleep(HOLD_PAST_RENEWAL).await;
        seam.before_chunk.release();
        (tick_outcome(task).await, seam)
    }

    /// SR-3.5: a lock another replica holds when a chunk's due refresh runs
    /// ends the pass before that chunk's script, with the `lease_lost`
    /// outcome, one `lost` and no `refresh_error`. The stale member that chunk
    /// held is not removed.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_lease_another_replica_holds_ends_the_pass_before_its_chunk_counting_lost() {
        let tv = TestValkey::setup().await;
        seed(&tv, "s", &[("stale", Seed::Stale)]).await;
        let metrics: SharedMetrics = Arc::new(metrics_scrape_only(SERVICE));
        let lock_key = tv.key_schema().presence_sweeper_leader_lock();
        let peer = LeaderIdentity::new(SERVICE, "instance-peer");
        let (outcome, seam) = lose_the_lease_at_a_parked_chunk(&tv, &metrics, async {
            let _: () = tv
                .client()
                .maintenance_pool
                .next()
                .set(
                    &lock_key,
                    peer.lock_value(),
                    Some(fred::types::Expiration::PX(30_000)),
                    None,
                    false,
                )
                .await
                .expect("handing the lock to the peer must succeed");
        })
        .await;

        assert_eq!(outcome, Some(PRESENCE_SWEEP_OUTCOME_LEASE_LOST));
        assert_eq!(
            lease_row(&metrics),
            [0.0, 1.0, 0.0],
            "[renewed, lost, refresh_error]: a lock held by another replica is lost, not an error"
        );
        assert!(
            seam.script_batch_sizes().is_empty(),
            "a lost lease ends the pass before its chunk's script"
        );
        assert!(is_member(&tv, "s", "stale").await);
        assert_eq!(
            passes_ended(&metrics, PRESENCE_SWEEP_OUTCOME_LEASE_LOST),
            Some(1.0)
        );
        assert!(
            !has_metric(&render(&metrics), OLDEST_AGE),
            "a pass that did not complete publishes no oldest stale age"
        );
    }

    /// SR-3.5: a chunk's due refresh that errors ends the pass before that
    /// chunk's script, counting `refresh_error` and, additively, `lost`. The
    /// error is forced by making the lock key a LIST, which the refresh
    /// script's `GET` rejects.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_lease_refresh_that_errors_ends_the_pass_counting_refresh_error_and_lost() {
        let tv = TestValkey::setup().await;
        seed(&tv, "s", &[("stale", Seed::Stale)]).await;
        let metrics: SharedMetrics = Arc::new(metrics_scrape_only(SERVICE));
        let lock_key = tv.key_schema().presence_sweeper_leader_lock();
        let (outcome, seam) = lose_the_lease_at_a_parked_chunk(&tv, &metrics, async {
            let conn = tv.client().maintenance_pool.next();
            let _: i64 = conn.del(&lock_key).await.expect("DEL lock must succeed");
            let _: i64 = conn
                .rpush(&lock_key, "tripwire")
                .await
                .expect("RPUSH tripwire must succeed");
        })
        .await;

        assert_eq!(outcome, Some(PRESENCE_SWEEP_OUTCOME_LEASE_LOST));
        assert_eq!(
            lease_row(&metrics),
            [0.0, 1.0, 1.0],
            "[renewed, lost, refresh_error]: an errored refresh counts refresh_error and lost"
        );
        assert!(seam.script_batch_sizes().is_empty());
        assert!(is_member(&tv, "s", "stale").await);
    }

    /// SR-3.3: a pass parked at a chunk boundary when `cancel` fires, then
    /// released, stops at that boundary: it calls no sweep script, records one
    /// `cancelled` pass, and the stale member stays.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_cancel_at_a_parked_chunk_ends_the_pass_before_its_script() {
        let tv = TestValkey::setup().await;
        seed(&tv, "s", &[("stale", Seed::Stale)]).await;
        let metrics: SharedMetrics = Arc::new(metrics_scrape_only(SERVICE));
        let cancel = CancellationToken::new();
        let (sweeper, seam) = seamed(&tv, SweeperBudget::production(), &metrics, cancel.clone());

        seam.before_chunk.arm();
        let task = tick(&sweeper);
        wait_parked(&seam).await;
        cancel.cancel();
        seam.before_chunk.release();

        assert_eq!(
            tick_outcome(task).await,
            Some(PRESENCE_SWEEP_OUTCOME_CANCELLED)
        );
        assert!(
            seam.script_batch_sizes().is_empty(),
            "no script after the cancel"
        );
        assert_eq!(
            seam.passes(),
            vec![ObservedSweepPass {
                outcome: PRESENCE_SWEEP_OUTCOME_CANCELLED,
                script_calls: 0,
            }]
        );
        assert_eq!(
            passes_ended(&metrics, PRESENCE_SWEEP_OUTCOME_CANCELLED),
            Some(1.0)
        );
        assert!(is_member(&tv, "s", "stale").await);
    }

    /// SR-3.3: a cancel while a page read is in flight ends the pass at once,
    /// without waiting for the reply. The read is held behind a stall of every
    /// data-pool connection, so a pass that awaited it would end at the 5 s
    /// command timeout, with the `error` outcome and a counted timeout; this
    /// one ends `cancelled` and counts none. (No timing is asserted: the
    /// outcome and the timeout count tell the two apart.)
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_cancel_during_a_page_read_ends_the_pass_without_awaiting_the_reply() {
        let tv = TestValkey::setup().await;
        seed(&tv, "s", &[("stale", Seed::Stale)]).await;
        let metrics: SharedMetrics = Arc::new(metrics_scrape_only(SERVICE));
        tv.client()
            .install_observability_hooks(Arc::clone(&metrics))
            .await;
        let cancel = CancellationToken::new();
        let (sweeper, seam) = seamed(&tv, SweeperBudget::production(), &metrics, cancel.clone());

        let stalls = tv.stall_data_pool().await;
        let task = tick(&sweeper);
        let in_flight = tokio::time::timeout(SETTLE, async {
            while seam.max_commands_in_flight() == 0 {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await;
        cancel.cancel();
        let outcome = tokio::time::timeout(SETTLE, task).await;
        release_all(stalls).await;

        in_flight.expect("precondition: the pass's index read was sent");
        let outcome = outcome
            .expect("the cancelled tick must end")
            .expect("the tick must not panic");
        assert_eq!(outcome, Some(PRESENCE_SWEEP_OUTCOME_CANCELLED));
        assert_eq!(
            command_timeout_row(&metrics),
            command_timeouts_only_on("data", 0),
            "the abandoned read must not have been awaited to its timeout"
        );
        assert!(seam.session_page_sizes().is_empty() && seam.script_batch_sizes().is_empty());
        assert!(is_member(&tv, "s", "stale").await);
    }

    /// SR-3.3: a pass that starts after `cancel` has fired reads no page: the
    /// boundary check before the first index read stops it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_pass_on_a_cancelled_token_reads_no_page() {
        let tv = TestValkey::setup().await;
        seed(&tv, "s", &[("stale", Seed::Stale)]).await;
        let metrics: SharedMetrics = Arc::new(metrics_scrape_only(SERVICE));
        let cancel = CancellationToken::new();
        cancel.cancel();
        let (sweeper, seam) = seamed(&tv, SweeperBudget::production(), &metrics, cancel);

        assert_eq!(
            sweeper.sweep_once().await,
            Some(PRESENCE_SWEEP_OUTCOME_CANCELLED)
        );
        assert_eq!(
            seam.max_commands_in_flight(),
            0,
            "no presence command may be sent after the cancel"
        );
        assert!(is_member(&tv, "s", "stale").await);
    }

    // ── Member-page replay (seam) ────────────────────────────────────────────

    /// A member page of `ids` whose next cursor is `cursor`.
    fn page(ids: &[&str], cursor: Option<&str>) -> ScanPage {
        ScanPage {
            ids: ids.iter().map(|id| (*id).to_owned()).collect(),
            next_cursor: cursor.map(str::to_owned),
        }
    }

    /// Whether the seam's replay is spent.
    fn replay_done(seam: &PresenceSweeperSeam) -> bool {
        matches!(*lock(&seam.member_page_replay), MemberPageReplay::Done)
    }

    #[test]
    fn a_replay_holds_the_armed_page_and_prepends_it_to_the_same_sessions_next_page_once() {
        let seam = PresenceSweeperSeam::new();
        assert_eq!(
            seam.rewrite_member_page("s", page(&["x"], Some("3"))),
            page(&["x"], Some("3")),
            "unarmed, a page passes through"
        );
        assert!(matches!(
            *lock(&seam.member_page_replay),
            MemberPageReplay::Off
        ));

        seam.repeat_next_member_page();
        assert!(matches!(
            *lock(&seam.member_page_replay),
            MemberPageReplay::Armed
        ));
        assert_eq!(
            seam.rewrite_member_page("s", page(&["a", "b"], Some("7"))),
            page(&["a", "b"], Some("7")),
            "the armed page passes through unchanged"
        );
        assert!(matches!(
            &*lock(&seam.member_page_replay),
            MemberPageReplay::Holding { session_id, ids }
                if session_id == "s" && *ids == ["a", "b"]
        ));
        assert_eq!(
            seam.rewrite_member_page("s", page(&["c"], Some("9"))),
            page(&["a", "b", "c"], Some("9")),
            "the session's next page carries the held ids first, its cursor unchanged"
        );
        assert!(replay_done(&seam));
        assert_eq!(
            seam.rewrite_member_page("s", page(&["d"], None)),
            page(&["d"], None),
            "one-shot: the page after the replay passes through"
        );
    }

    #[test]
    fn a_page_of_another_session_abandons_the_held_replay() {
        let seam = PresenceSweeperSeam::new();
        seam.repeat_next_member_page();
        seam.rewrite_member_page("s1", page(&["a"], Some("5")));
        assert_eq!(
            seam.rewrite_member_page("s2", page(&["x"], Some("6"))),
            page(&["x"], Some("6")),
            "another session's page gets nothing prepended"
        );
        assert!(replay_done(&seam));
        assert_eq!(
            seam.rewrite_member_page("s1", page(&["b"], None)),
            page(&["b"], None),
            "the abandoned replay does not reach the held session's later page"
        );
    }

    #[test]
    fn the_end_of_a_pass_abandons_a_held_replay_but_not_an_armed_one() {
        let seam = PresenceSweeperSeam::new();
        seam.repeat_next_member_page();
        seam.rewrite_member_page("s", page(&["a"], Some("5")));
        seam.pass_ended(PRESENCE_SWEEP_OUTCOME_COMPLETED, 1);
        assert!(replay_done(&seam));
        assert_eq!(
            seam.rewrite_member_page("s", page(&["b"], None)),
            page(&["b"], None),
            "a held page is never re-delivered into a later pass"
        );

        let seam = PresenceSweeperSeam::new();
        seam.repeat_next_member_page();
        seam.pass_ended(PRESENCE_SWEEP_OUTCOME_COMPLETED, 0);
        assert!(
            matches!(*lock(&seam.member_page_replay), MemberPageReplay::Armed),
            "a pass that read no member page leaves the replay armed"
        );
        seam.rewrite_member_page("s", page(&["a"], Some("5")));
        assert_eq!(
            seam.rewrite_member_page("s", page(&["b"], None)),
            page(&["a", "b"], None)
        );
    }

    #[test]
    fn re_arming_while_armed_holding_or_spent_has_no_effect() {
        // Armed: a second arm does not make the replay fire twice.
        let seam = PresenceSweeperSeam::new();
        seam.repeat_next_member_page();
        seam.repeat_next_member_page();
        seam.rewrite_member_page("s", page(&["a"], Some("1")));
        assert_eq!(
            seam.rewrite_member_page("s", page(&["b"], Some("2"))),
            page(&["a", "b"], Some("2"))
        );
        assert_eq!(
            seam.rewrite_member_page("s", page(&["c"], None)),
            page(&["c"], None),
            "re-arming while armed"
        );

        // Holding: an arm neither drops nor re-captures the held page.
        let seam = PresenceSweeperSeam::new();
        seam.repeat_next_member_page();
        seam.rewrite_member_page("s", page(&["a"], Some("1")));
        seam.repeat_next_member_page();
        assert_eq!(
            seam.rewrite_member_page("s", page(&["b"], Some("2"))),
            page(&["a", "b"], Some("2")),
            "re-arming while holding"
        );
        assert_eq!(
            seam.rewrite_member_page("s", page(&["c"], None)),
            page(&["c"], None)
        );

        // Done: a spent replay stays spent.
        seam.repeat_next_member_page();
        assert!(replay_done(&seam));
        seam.rewrite_member_page("s", page(&["d"], Some("3")));
        assert_eq!(
            seam.rewrite_member_page("s", page(&["e"], None)),
            page(&["e"], None),
            "re-arming once spent"
        );
    }

    #[test]
    fn an_empty_armed_page_replays_nothing_and_spends_the_replay() {
        let seam = PresenceSweeperSeam::new();
        seam.repeat_next_member_page();
        assert_eq!(
            seam.rewrite_member_page("s", page(&[], Some("4"))),
            page(&[], Some("4"))
        );
        assert_eq!(
            seam.rewrite_member_page("s", page(&["b"], None)),
            page(&["b"], None),
            "an empty held page prepends nothing"
        );
        assert!(replay_done(&seam));
    }

    // ── SweeperBudget ────────────────────────────────────────────────────────

    #[test]
    fn the_production_budget_is_the_leader_lock_ttl_and_the_sweep_constants() {
        assert_eq!(
            SweeperBudget::production(),
            SweeperBudget {
                lock_ttl: Duration::from_millis(LEADER_LOCK_TTL_MS),
                session_page: SWEEP_SESSION_PAGE,
                member_page: SWEEP_MEMBER_PAGE,
                member_batch: SWEEP_MEMBER_BATCH,
            }
        );
        SweeperBudget::production().assert_valid();
    }

    #[test]
    fn a_budget_at_its_bounds_is_valid() {
        for member_batch in [1, SWEEP_MEMBER_BATCH] {
            SweeperBudget {
                lock_ttl: Duration::from_millis(1),
                session_page: 1,
                member_page: 1,
                member_batch,
            }
            .assert_valid();
        }
    }

    #[test]
    #[should_panic(expected = "lock_ttl must be at least 1 ms")]
    fn a_lock_ttl_under_one_millisecond_is_refused() {
        SweeperBudget {
            lock_ttl: Duration::from_micros(999),
            ..SweeperBudget::production()
        }
        .assert_valid();
    }

    #[test]
    #[should_panic(expected = "page sizes must be at least 1")]
    fn a_zero_session_page_is_refused() {
        SweeperBudget {
            session_page: 0,
            ..SweeperBudget::production()
        }
        .assert_valid();
    }

    #[test]
    #[should_panic(expected = "page sizes must be at least 1")]
    fn a_zero_member_page_is_refused() {
        SweeperBudget {
            member_page: 0,
            ..SweeperBudget::production()
        }
        .assert_valid();
    }

    #[test]
    #[should_panic(expected = "member_batch must be 1 to")]
    fn a_zero_member_batch_is_refused() {
        SweeperBudget {
            member_batch: 0,
            ..SweeperBudget::production()
        }
        .assert_valid();
    }

    #[test]
    #[should_panic(expected = "member_batch must be 1 to")]
    fn a_member_batch_over_the_script_cap_is_refused() {
        SweeperBudget {
            member_batch: SWEEP_MEMBER_BATCH + 1,
            ..SweeperBudget::production()
        }
        .assert_valid();
    }
}
