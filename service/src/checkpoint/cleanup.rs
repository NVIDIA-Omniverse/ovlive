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

//! The checkpoint clean-up task (b.g65, SDD §22.7.2, §22.17.5).
//!
//! The backup store's clean-up — reaping orphaned partial publishes, reaping
//! the generations of deleted layers, and collecting the backup copies of
//! large values no retained backup names — runs as its **own** background
//! task, spawned in `main.rs` beside the external-value backup repair. It is
//! not a timer of the checkpoint loop and takes no part in
//! `le:checkpoint:leader`, so the backup loop never pauses for it. One
//! layer's backup can still defer briefly: the clean-up fence (R7) refuses a
//! publish that names a value the clean-up just deleted, or a stem it is
//! deleting, and that backup is written on a later pass (counted as
//! `checkpoint_reconcile_total{outcome="publish_deferred"}`).
//!
//! - **Its lock is `le:gc:leader`**, the load-shed lock of the GC, the orphan
//!   sweep, the reference-count reconciler and the backup repair: this task is
//!   its fifth consumer, taken with the process's one [`LeaderIdentity`]
//!   ([`try_acquire_gc_leader`]). Every `reconcile_interval` a replica probes
//!   it; the holder runs a pass, a non-holder publishes 0 on both backlog
//!   gauges, and a failed probe publishes nothing. The pass re-checks the
//!   lock at its start, at each phase start and at each drain iteration with
//!   a plain read that never refreshes it, and ends when this process no longer holds it
//!   (or the read fails). During the pass the GC task's progress-gated
//!   refresh ticker alone keeps the key, so a replica whose GC loop died or
//!   wedged loses it on schedule and its clean-up pass ends. That is
//!   load-shedding only: no safety rests on it, and two replicas running a
//!   pass at once cost duplicate work, never a wrong delete.
//! - **Its safety is in Valkey.** Every delete is licensed by its own
//!   `{recon}` script (`storage::recon_fence`): the collect's delete-time
//!   fence against writers' re-arms and Valkey liveness, the orphan reap's
//!   commit against a publish of the same stem, and the deleters'
//!   announcements that explain a listed sidecar the collect then reads absent.
//!   None of it depends on holding a lock while deciding.
//! - **It waits while an older build, or no replica, leads backups**
//!   ([`CleanupGate`]): an
//!   older build's backup writers do not re-arm before their PUTs, so the
//!   collect's delete rule does not cover them. The pass reads
//!   `le:checkpoint:leader` at its start, at each phase start and at each
//!   drain iteration (every [`CleanupGate::may_continue`] call) and runs
//!   only while it carries the marked value of a current build
//!   ([`LeaderIdentity::checkpoint_lock_value`], version 1 or more). On an
//!   unmarked or absent value it skips the rest of the pass with one INFO and
//!   `checkpoint_reconcile_total{outcome="deferred_older_build"}`, and still
//!   publishes the backlog gauges.
//!
//! The pass body — the orphan, dead-object and collect phases and their
//! budget — is `checkpoint::checkpointer::run_cleanup_pass`.

use std::sync::Arc;
use std::time::{Duration, Instant};

use fred::interfaces::KeysInterface;
use tokio::task::JoinHandle;
use tokio::time::MissedTickBehavior;
use tokio_util::sync::CancellationToken;

use crate::checkpoint::checkpointer::{run_cleanup_pass, CheckpointLoopConfig};
use crate::checkpoint::ext_mirror::{ExtMirrorSender, ReserveRefusal};
use crate::checkpoint::store::reconcile_grace;
use crate::checkpoint::CheckpointStore;
use crate::observability::metrics::SharedMetrics;
use crate::storage::client::{FredErrOrigin, ValkeyClient, POOL_LABEL_MAINTENANCE};
use crate::storage::ext_backup::ExtBackupMarks;
use crate::storage::external_values::try_acquire_gc_leader;
use crate::storage::keys::KeySchema;
use crate::storage::leader_lease::{checkpoint_lock_version, LeaderIdentity};

/// The lowest `le:checkpoint:leader` protocol version under which the
/// clean-up runs: the first build whose backup writers re-arm before every
/// PUT.
const CLEANUP_MIN_CHECKPOINT_LOCK_VERSION: u32 = 1;

/// How long the collect's heal waits for room in the ext mirror's byte budget
/// before it gives up and keeps the value's `:collected` record for a later
/// heal (SDD §22.17.5, R5).
pub(super) const HEAL_RESERVE_WAIT: Duration = Duration::from_secs(5);

/// How often the heal asks the budget again while it waits.
const HEAL_RESERVE_RETRY: Duration = Duration::from_millis(250);

/// What the collect's heal needs (b.g65, R5 and R6): the ext mirror's sender,
/// whose byte budget the heal's read is charged to, as the backup repair's
/// re-send is, and the marks it confirms the re-put copy with. Without it the
/// collect cannot heal, and a value it would have healed keeps its
/// `:collected` record for a later pass.
#[derive(Clone)]
pub struct CollectHeal {
    mirror: ExtMirrorSender,
    marks: ExtBackupMarks,
}

impl CollectHeal {
    /// The heal over `mirror`'s byte budget, confirming under `marks`.
    pub fn new(mirror: ExtMirrorSender, marks: ExtBackupMarks) -> Self {
        Self { mirror, marks }
    }

    /// A reservation of `len` bytes from the mirror's byte budget, asking
    /// again while it has no room, for at most [`HEAL_RESERVE_WAIT`]. `None`
    /// when no room came (or the value is larger than the whole budget).
    pub(super) async fn reserve(
        &self,
        len: u64,
    ) -> Option<crate::checkpoint::ExtMirrorReservation> {
        let started = Instant::now();
        loop {
            match self.mirror.try_reserve(len) {
                Ok(reservation) => return Some(reservation),
                Err(ReserveRefusal::NoRoom) if started.elapsed() < HEAL_RESERVE_WAIT => {
                    tokio::time::sleep(HEAL_RESERVE_RETRY).await;
                }
                Err(_) => return None,
            }
        }
    }

    /// The marks the heal confirms its re-put copy with.
    pub(super) fn marks(&self) -> &ExtBackupMarks {
        &self.marks
    }
}

/// The clean-up pass's entitlement to continue (b.g65, N1 and N5): the
/// `le:gc:leader` re-check (a read, never a refresh; load-shed only) and the
/// current-build check on
/// `le:checkpoint:leader`. Once either fails the gate stays closed for the
/// rest of the pass.
pub(super) struct CleanupGate<'a> {
    client: &'a ValkeyClient,
    keys: &'a KeySchema,
    leader: &'a LeaderIdentity,
    metrics: Option<&'a SharedMetrics>,
    closed: bool,
    /// Whether the pass stopped because an older build, or no replica, leads
    /// backups — `le:checkpoint:leader` unmarked, a lower version, or absent
    /// (the pass summary's `deferred_older_build` field).
    older_build_leads: bool,
}

impl<'a> CleanupGate<'a> {
    /// The gate for one pass of this replica, `leader`.
    pub(super) fn new(
        client: &'a ValkeyClient,
        keys: &'a KeySchema,
        leader: &'a LeaderIdentity,
        metrics: Option<&'a SharedMetrics>,
    ) -> Self {
        Self {
            client,
            keys,
            leader,
            metrics,
            closed: false,
            older_build_leads: false,
        }
    }

    /// Whether the pass stopped because an older build, or no replica, leads
    /// backups.
    pub(super) fn older_build_leads(&self) -> bool {
        self.older_build_leads
    }

    /// Whether the pass may go on: this replica still holds `le:gc:leader`
    /// and `le:checkpoint:leader` carries a current build's marked value.
    /// Called at the pass start, at each phase start and at each drain
    /// iteration.
    ///
    /// The `le:gc:leader` check is a plain read, never a refresh: one `GET`
    /// compared with this process's [`LeaderIdentity::gc_lock_value`]. The
    /// lock is taken (and refreshed) only by the tick's acquire before the
    /// pass ([`cleanup_tick`]); during the pass, the GC task's progress-gated
    /// refresh ticker is the only thing that keeps it. So a replica whose GC
    /// loop died or wedged loses the key on schedule and a peer takes it over,
    /// and this pass then ends at its next check instead of keeping the key
    /// alive and starving GC fleet-wide.
    pub(super) async fn may_continue(&mut self) -> bool {
        if self.closed {
            return false;
        }
        let replica_id = self.leader.service_label.as_str();
        let own_value = self.leader.gc_lock_value();
        match read_gc_lock(self.client, self.keys).await {
            Ok(Some(value)) if value == own_value => {}
            Ok(holder) => {
                tracing::info!(
                    replica_id = %replica_id,
                    lock_value = %own_value,
                    holder = holder.as_deref().unwrap_or(""),
                    "checkpoint clean-up lost the maintenance leader lock mid-pass; ending the pass \
                     (what it finished stays done, and the next holder's pass resumes)"
                );
                self.closed = true;
                return false;
            }
            Err(e) => {
                tracing::warn!(
                    replica_id = %replica_id,
                    lock_value = %own_value,
                    error = %e,
                    "checkpoint clean-up could not re-probe the maintenance leader lock; ending the \
                     pass"
                );
                self.closed = true;
                return false;
            }
        }
        match read_checkpoint_lock(self.client, self.keys).await {
            Ok(Some(value))
                if checkpoint_lock_version(&value)
                    .is_some_and(|v| v >= CLEANUP_MIN_CHECKPOINT_LOCK_VERSION) =>
            {
                true
            }
            Ok(holder) => {
                tracing::info!(
                    replica_id = %replica_id,
                    checkpoint_lock_value = holder.as_deref().unwrap_or(""),
                    "checkpoint clean-up skips the rest of this pass: the replica leading backups \
                     runs an older build, or none leads them now, and an older build's backup \
                     writers do not follow the clean-up's delete rule. Expected during a rolling \
                     upgrade, and briefly when no replica holds the backup lock (just after \
                     startup, or while backup leadership fails over); clean-up resumes once a \
                     current build leads backups"
                );
                if let Some(m) = self.metrics {
                    m.add_checkpoint_reconcile("deferred_older_build", 1, replica_id);
                }
                self.older_build_leads = true;
                self.closed = true;
                false
            }
            Err(e) => {
                tracing::warn!(
                    replica_id = %replica_id,
                    error = %e,
                    "checkpoint clean-up could not read which build leads backups; ending the pass"
                );
                self.closed = true;
                false
            }
        }
    }
}

/// `GET le:gc:leader` on the maintenance pool: the holder's lock value,
/// `None` when no replica holds the maintenance leader lock. A read only: it
/// never refreshes or takes the lock.
async fn read_gc_lock(
    client: &ValkeyClient,
    keys: &KeySchema,
) -> Result<Option<String>, fred::error::Error> {
    let origin = FredErrOrigin::client(client, POOL_LABEL_MAINTENANCE);
    let (maint_pool, _maint_token) = client.acquire_maintenance();
    maint_pool
        .next()
        .get(keys.gc_leader_lock())
        .await
        .inspect_err(|e| origin.observe_error(e))
}

/// `GET le:checkpoint:leader` on the maintenance pool: the holder's lock
/// value, `None` when no replica leads backups.
async fn read_checkpoint_lock(
    client: &ValkeyClient,
    keys: &KeySchema,
) -> Result<Option<String>, fred::error::Error> {
    let origin = FredErrOrigin::client(client, POOL_LABEL_MAINTENANCE);
    let (maint_pool, _maint_token) = client.acquire_maintenance();
    maint_pool
        .next()
        .get(keys.checkpoint_leader_lock())
        .await
        .inspect_err(|e| origin.observe_error(e))
}

/// One interval of the clean-up task: probe `le:gc:leader` and run one pass
/// as its holder ([`run_cleanup_pass`]); as a non-holder, publish 0 on both
/// backlog gauges (an OTel gauge never decays, so a replica that led and then
/// lost the lock would otherwise pin the fleet-wide `max()` at a backlog that
/// is no longer its to report); on a failed probe, publish nothing (unknown is
/// not zero).
pub(super) async fn cleanup_tick(
    client: &ValkeyClient,
    keys: &KeySchema,
    store: &Arc<dyn CheckpointStore>,
    config: &CheckpointLoopConfig,
    metrics: Option<&SharedMetrics>,
    cancel: &CancellationToken,
    heal: Option<&CollectHeal>,
) {
    let leader = &config.leader;
    match try_acquire_gc_leader(client, keys, leader, metrics).await {
        Ok(true) => run_cleanup_pass(client, keys, store, config, metrics, cancel, heal).await,
        Ok(false) => {
            if let Some(m) = metrics {
                m.record_checkpoint_ext_tombstone_backlog(0, &leader.service_label);
                m.record_checkpoint_dead_object_backlog(0, &leader.service_label);
            }
            tracing::debug!(
                replica_id = %leader.service_label,
                lock_value = %leader.gc_lock_value(),
                "checkpoint clean-up skipped: another replica holds the maintenance leader lock"
            );
        }
        Err(e) => tracing::warn!(
            replica_id = %leader.service_label,
            lock_value = %leader.gc_lock_value(),
            error = %e,
            "checkpoint clean-up leader lock acquisition failed; skipping this interval"
        ),
    }
}

/// Spawn the checkpoint clean-up task: one [`cleanup_tick`] every
/// `config.reconcile_interval` (`LIVE_EDIT_CHECKPOINT_RECONCILE_INTERVAL_SECS`),
/// the first at once, `MissedTickBehavior::Skip` so a long pass never replays
/// a burst. `config.leader` is the process's one identity, shared with every
/// other leader-gated task. `heal` is the ext mirror's sender and the backup
/// marks, present whenever the mirror is.
///
/// Exits on `cancel`. The task's `select!` is biased toward the cancel, so
/// once it fires no interval that is also ready starts another pass. A pass
/// in progress checks it before each member, each dead object's generation
/// and each batch, and races every read it makes before a member's first
/// write against it (the gate's reads, the claims that register nothing, the
/// listings, the liveness probes, the sidecar reads, the live-set walk and its
/// announcement check), abandoning the read in flight; a
/// delete and its post-delete step are one unit with no check between them,
/// and no write is abandoned once sent (`checkpointer::run_reconcile_tick`,
/// *Cancellation*). Everything a pass finished stays done, and the next pass
/// resumes from the durable `{recon}` work lists.
pub fn spawn_checkpoint_cleanup(
    client: ValkeyClient,
    keys: KeySchema,
    store: Arc<dyn CheckpointStore>,
    config: CheckpointLoopConfig,
    heal: Option<CollectHeal>,
    metrics: Option<SharedMetrics>,
    cancel: CancellationToken,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        tracing::info!(
            reconcile_interval_secs = config.reconcile_interval.as_secs(),
            reconcile_grace_secs = reconcile_grace(store.publish_worst_case()).as_secs(),
            heal = heal.is_some(),
            replica_id = %config.leader.service_label,
            lock_value = %config.leader.gc_lock_value(),
            "checkpoint clean-up task started"
        );
        let mut timer = tokio::time::interval(config.reconcile_interval);
        timer.set_missed_tick_behavior(MissedTickBehavior::Skip);
        loop {
            tokio::select! {
                biased;
                () = cancel.cancelled() => {
                    tracing::info!("checkpoint clean-up shutting down");
                    return;
                }
                _ = timer.tick() => {
                    cleanup_tick(
                        &client,
                        &keys,
                        &store,
                        &config,
                        metrics.as_ref(),
                        &cancel,
                        heal.as_ref(),
                    )
                    .await;
                }
            }
        }
    })
}
