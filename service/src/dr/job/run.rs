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

//! The Job frame: the run order every restore and accept-loss Job follows
//! (`super`'s module docs list it), around a [`JobBody`]: the restore body
//! (`super::restore`) or the accept-loss body (`super::accept_loss`).

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use async_trait::async_trait;
use tokio::net::TcpListener;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use super::absence::{
    check_counters_sidecar, run_list_route_canary, CanaryCheck, OwnPuts, SidecarCheck, StoreClock,
    LIST_ROUTE_CAUSE, SIDECAR_ABSENT_CAUSE,
};
use super::args::{JobArgs, JobMode};
use super::execute::discard::log_value_removal_purge;
use super::execute::memory::RestoreMemory;
use super::lease::{JobLease, LeaseAcquireError, LeaseRefresh, LeaseTimings};
use super::permit::{RevokeCause, WritePermit};
use super::preconditions::{read_precondition, Precondition};
use super::quiet::{not_found_cache_lifetime, QuietPeriod, StoreDeadlines};
use super::JobExit;
use crate::checkpoint::{CheckpointAdapter, CheckpointStore};
use crate::config::Config;
use crate::observability::metrics::Metrics;
use crate::observability::metrics_server::serve_metrics;
use crate::storage::allocator_purge::AllocatorPurge;
use crate::storage::client::ValkeyClient;
use crate::storage::keys::KeySchema;
use crate::storage::leader_lease::LeaderIdentity;

/// How often a long wait on the store's clock logs what is left: 30 s.
pub const WAIT_LOG_INTERVAL: Duration = Duration::from_secs(30);

/// Every timing the frame uses. [`JobTimings::production`] in the binary;
/// tests shorten them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JobTimings {
    /// The lease's TTL, refresh interval and acquire wait.
    pub lease: LeaseTimings,
    /// How often a wait on the store's clock logs what is left. A failed
    /// read of the store's clock during the quiet period is retried at the
    /// same cadence.
    pub wait_log_interval: Duration,
}

impl JobTimings {
    /// The production timings.
    pub fn production() -> Self {
        Self {
            lease: LeaseTimings::production(),
            wait_log_interval: WAIT_LOG_INTERVAL,
        }
    }
}

/// Why a Job has no backup store to read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoBackupStore {
    /// `LIVE_EDIT_CHECKPOINT_ADAPTER=disabled`: there are no backups.
    Disabled,
    /// `inmemory`: the backups lived inside one server process, so this
    /// process's own store is empty.
    InMemory,
}

/// What the Job may read backups from, decided by the configured adapter,
/// never by an error or an empty answer.
#[derive(Clone)]
pub enum BackupAccess {
    /// No backup store: no listing, no canary, no sidecar read and no
    /// `ext_blob_exists` probe; the run treats it as "no backups".
    None(NoBackupStore),
    /// A persistent backup store (`LocalDisk` or `StorageApi`).
    Store(Arc<dyn CheckpointStore>),
}

/// Everything a Job needs from the process that runs it.
pub struct JobContext {
    /// The accepted command line.
    pub args: JobArgs,
    /// The configuration, read with `Config::from_env`.
    pub config: Arc<Config>,
    /// The Valkey client, connected by `main.rs`.
    pub valkey: ValkeyClient,
    /// The key schema `main.rs` built for this process.
    pub keys: KeySchema,
    /// The backup store, built as the server builds it.
    pub checkpoint_store: Arc<dyn CheckpointStore>,
    /// The process's metrics, served on `metrics_listener`.
    pub metrics: Arc<Metrics>,
    /// The metrics port, bound by the caller; `None` serves no metrics.
    pub metrics_listener: Option<TcpListener>,
    /// The frame's timings.
    pub timings: JobTimings,
    /// Resolves on SIGTERM (or SIGINT). Its signal handler must already be
    /// installed when the context is built.
    pub shutdown: Pin<Box<dyn Future<Output = ()> + Send>>,
    /// The value this Job holds its lease with.
    pub identity: LeaderIdentity,
    /// The process's allocator purge (`storage::allocator_purge`), which
    /// accept-loss's value removal runs around each of its transactions;
    /// `None` when the process installed none. The binary installs one on
    /// linux-gnu (`main.rs`'s `run_dr_job`, every jemalloc arena) and
    /// `None` elsewhere. With `None`, or once a purge has failed, the value
    /// removal charges each removal its un-purged admission
    /// (`execute::memory::value_removal_admission`).
    pub allocator_purge: Option<AllocatorPurge>,
}

/// What a [`JobBody`] works with. Built by the frame after the lease is taken
/// and the precondition re-read.
pub struct JobRun {
    /// Restore or accept-loss.
    pub mode: JobMode,
    /// The incident the store is held under.
    pub incident: String,
    /// The configuration.
    pub config: Arc<Config>,
    /// The Valkey client.
    pub valkey: ValkeyClient,
    /// The key schema.
    pub keys: KeySchema,
    /// The backup store, or why there is none.
    pub backup: BackupAccess,
    /// The quiet period, term by term.
    pub quiet: QuietPeriod,
    /// L, how old a by-address answer from the backup store may be: a
    /// cached not-found, or a cached earlier body.
    pub not_found_cache: Duration,
    /// The store's clock and the Job's deadlines on it.
    pub clock: StoreClock,
    /// The write permit. Every write unit calls `begin_unit` first.
    pub permit: WritePermit,
    /// The external values this run stored in the backup store itself.
    pub own_puts: OwnPuts,
    /// The process's metrics.
    pub metrics: Arc<Metrics>,
    /// The value this Job holds its lease with (`JobLease::holder`): the
    /// un-hold checks the lease still holds it.
    pub lease_holder: String,
    /// The counters sidecar's `max_object_id_observed` as this run's PD2
    /// check read it, set by the frame when the check passed; unset with no
    /// backup store. The counter floor's sidecar term.
    pub counters_sidecar_max: OnceLock<u64>,
    /// The process's allocator purge, from [`JobContext::allocator_purge`]:
    /// accept-loss's value removal runs it around each transaction; a
    /// restore never calls it.
    pub allocator_purge: Option<AllocatorPurge>,
}

impl JobRun {
    /// The Job's deadlines on the store's clock.
    pub fn deadlines(&self) -> StoreDeadlines {
        self.clock.deadlines()
    }
}

/// The part of a Job each body builds: what it plans, and what it does
/// once the frame has opened the write permit and the run-wide absence
/// preconditions have passed.
#[async_trait]
pub trait JobBody: Send + Sync {
    /// What [`Self::plan`] produces and [`Self::execute`] consumes.
    type Plan: Send;

    /// Reads only: no write, no call of the permit. Called once, as the
    /// quiet period begins, and runs in the background while it lasts; the
    /// frame keeps its plan. A plan not finished when the permit opens is
    /// awaited then. Return an exit to end the run (having logged why).
    ///
    /// Because it starts during Q, reads it took then could predate late
    /// writes that Q bounds: a `StorageApi` checkpoint PUT applied within C
    /// after entry, or a Valkey write inside the hold's write bound. So a
    /// body must take, or re-check, any listing it picks generations from at
    /// or after Q's end, before it writes. Otherwise "newer wins" could
    /// restore an older generation. Both bodies take every read at or after
    /// Q's end: their planner first waits on the store's clock for Q's end
    /// (`plan::RestorePlanner::plan`), and accept-loss's, with a backup
    /// store, for `T_safe`, so nothing needs a re-check.
    async fn plan(&self, run: &JobRun) -> Result<Self::Plan, JobExit>;

    /// The writes, and the finish (`super::finish`: the recount, then stay
    /// held or un-hold). Called after the permit opened and, with a backup
    /// store, after the counters sidecar read present and parsed
    /// (`run.counters_sidecar_max` then holds its reading) and the
    /// list-route canary passed. Every write unit calls
    /// `run.permit.begin_unit()` first. The frame drops this future when the
    /// permit is revoked.
    async fn execute(&self, run: &JobRun, plan: Self::Plan) -> JobExit;
}

/// Run a Job to its exit: [`super`]'s run order, around `body`.
pub async fn run_job<B: JobBody>(ctx: JobContext, body: B) -> JobExit {
    let JobContext {
        args,
        config,
        valkey,
        keys,
        checkpoint_store,
        metrics,
        metrics_listener,
        timings,
        shutdown,
        identity,
        allocator_purge,
    } = ctx;
    let mode = args.mode;
    // The Job's restore memory (`execute::memory`): `main.rs` refuses a limit
    // below the minimum before connecting (exit 2), so an error here is
    // reachable only in-process. Serial mode's WARN is logged once, here.
    match RestoreMemory::derive(&config, mode) {
        Ok(memory) => memory.log_serial_warn(mode),
        Err(refusal) => {
            tracing::error!(
                mode = mode.name(),
                refusal = %refusal,
                "{refusal}; the {mode} Job wrote nothing and exits 1"
            );
            return JobExit::Failed;
        }
    }
    // Accept-loss: whether its value removal purges the allocator, and the
    // admission that sets, once per run and before anything else, so a run
    // that removes nothing says it too.
    if mode == JobMode::AcceptLoss {
        log_value_removal_purge(mode, allocator_purge.as_ref());
    }
    let metrics_task = metrics_listener
        .map(|listener| tokio::spawn(serve_metrics(listener, metrics.prometheus_exporter())));
    let sigterm = CancellationToken::new();
    let signal_task = tokio::spawn({
        let sigterm = sigterm.clone();
        async move {
            shutdown.await;
            tracing::info!("the DR Job received SIGTERM");
            sigterm.cancel();
        }
    });

    let backup = match config.checkpoint_adapter {
        CheckpointAdapter::Disabled => BackupAccess::None(NoBackupStore::Disabled),
        CheckpointAdapter::InMemory => BackupAccess::None(NoBackupStore::InMemory),
        CheckpointAdapter::LocalDisk | CheckpointAdapter::StorageApi => {
            BackupAccess::Store(checkpoint_store)
        }
    };
    let frame = Frame {
        mode,
        args,
        config,
        valkey,
        keys,
        backup,
        metrics,
        timings,
        sigterm,
        allocator_purge,
    };
    let exit = frame.run(identity, &body).await;

    signal_task.abort();
    if let Some(task) = metrics_task {
        task.abort();
    }
    tracing::info!(
        mode = mode.name(),
        exit_code = exit.code(),
        "the {mode} Job exits {}",
        exit.code()
    );
    exit
}

struct Frame {
    mode: JobMode,
    args: JobArgs,
    config: Arc<Config>,
    valkey: ValkeyClient,
    keys: KeySchema,
    backup: BackupAccess,
    metrics: Arc<Metrics>,
    timings: JobTimings,
    sigterm: CancellationToken,
    allocator_purge: Option<AllocatorPurge>,
}

impl Frame {
    async fn run<B: JobBody>(&self, identity: LeaderIdentity, body: &B) -> JobExit {
        let mode = self.mode;
        // 1. The precondition, before the lease: writes nothing.
        if let Err(exit) = self.precondition("").await {
            return exit;
        }
        if mode == JobMode::Restore
            && matches!(self.backup, BackupAccess::None(NoBackupStore::InMemory))
        {
            tracing::error!(
                mode = mode.name(),
                "the {mode} Job wrote nothing and exits 3; the service stays held: the in-memory \
                 backup store cannot be read by a restore Job; use accept-loss"
            );
            return JobExit::StaysHeld;
        }

        // 2. The lease.
        let lease = match JobLease::acquire(
            &self.valkey,
            &self.keys,
            identity,
            self.timings.lease,
            &self.sigterm,
        )
        .await
        {
            Ok(lease) => lease,
            Err(LeaseAcquireError::Busy { holder }) => {
                tracing::error!(
                    mode = mode.name(),
                    holder = %holder,
                    "another DR Job holds the lease: the {mode} Job wrote nothing and exits 1; \
                     Kubernetes retries it"
                );
                return JobExit::Failed;
            }
            Err(LeaseAcquireError::Store(e)) => {
                tracing::error!(
                    mode = mode.name(),
                    error = %e,
                    "the {mode} Job could not take its lease, wrote nothing and exits 1"
                );
                return JobExit::Failed;
            }
            Err(LeaseAcquireError::Cancelled) => {
                tracing::info!(
                    mode = mode.name(),
                    "the {mode} Job stopped before taking its lease; it wrote nothing"
                );
                return JobExit::Failed;
            }
        };
        tracing::info!(
            mode = mode.name(),
            lease = lease.key(),
            "took the DR Job lease"
        );
        let refresh = lease.spawn_refresh();

        let exit = self.with_lease(body, &refresh, lease.holder()).await;

        // Every exit after the lease was taken releases it, so a replacement
        // Job acquires at once (a lost lease is no longer this Job's, and the
        // compare-and-delete leaves it alone).
        refresh.stop();
        lease.release().await;
        exit
    }

    /// Read the precondition and turn every refusal into its exit, logged.
    /// `phase` names when it is read, for the logs.
    async fn precondition(&self, phase: &str) -> Result<(String, u64), JobExit> {
        let mode = self.mode;
        let incident = &self.args.incident;
        match read_precondition(&self.valkey, &self.keys, incident).await {
            Ok(Precondition::Held {
                incident,
                detected_at_ms,
            }) => Ok((incident, detected_at_ms)),
            Ok(Precondition::NotHeld) => {
                tracing::info!(
                    mode = mode.name(),
                    incident = %incident,
                    "the store is not held{phase}: nothing to do; the {mode} Job wrote nothing and \
                     exits 0"
                );
                Err(JobExit::NothingToDo)
            }
            Ok(Precondition::DifferentIncident { held }) => {
                tracing::error!(
                    mode = mode.name(),
                    incident = %incident,
                    held_incident = %held,
                    "held under a different incident {held}: the {mode} Job was started for \
                     incident {incident}, wrote nothing and exits 4"
                );
                Err(JobExit::DifferentIncident)
            }
            Ok(Precondition::MarkerIncomplete { field }) => {
                tracing::error!(
                    mode = mode.name(),
                    incident = %incident,
                    field,
                    "the {mode} Job wrote nothing and exits 3; the service stays held: the hold \
                     marker's {field} field is missing or unreadable, and the Job never guesses it"
                );
                Err(JobExit::StaysHeld)
            }
            Err(e) => {
                tracing::error!(
                    mode = mode.name(),
                    error = %e,
                    "the {mode} Job could not read the hold marker{phase}, wrote nothing and \
                     exits 1"
                );
                Err(JobExit::Failed)
            }
        }
    }

    async fn with_lease<B: JobBody>(
        &self,
        body: &B,
        refresh: &LeaseRefresh,
        lease_holder: String,
    ) -> JobExit {
        let mode = self.mode;
        // 3. The precondition again, under the lease: an un-hold or a new
        // incident between the first read and the lease is seen here.
        let (incident, detected_at_ms) =
            match self.precondition(" (read again under the lease)").await {
                Ok(held) => held,
                Err(exit) => return exit,
            };
        let quiet = QuietPeriod::from_config(&self.config);
        let not_found_cache = not_found_cache_lifetime(&self.config);
        let deadlines = StoreDeadlines::new(detected_at_ms, quiet.total(), not_found_cache);
        tracing::info!(
            mode = mode.name(),
            incident = %incident,
            detected_at_ms,
            quiet_period_secs = quiet.total().as_secs_f64(),
            stale_threshold_secs = quiet.stale_threshold.as_secs(),
            poll_interval_secs = quiet.poll_interval.as_secs(),
            hold_write_bound_secs = quiet.hold_write_bound.as_secs(),
            join_teardown_bound_secs = quiet.join_teardown_bound.as_secs(),
            adapter_call_ceiling_secs = quiet.adapter_call_ceiling.as_secs(),
            shutdown_bound_secs = quiet.shutdown_bound.as_secs_f64(),
            not_found_cache_secs = not_found_cache.as_secs(),
            quiet_ends_at_ms = deadlines.quiet_ends_at_ms,
            t_safe_ms = deadlines.t_safe_ms,
            "the store is held under this incident; the {mode} Job writes nothing until the \
             store's clock passes the quiet period"
        );

        let permit = WritePermit::new();
        let run = JobRun {
            mode,
            incident,
            config: Arc::clone(&self.config),
            valkey: self.valkey.clone(),
            keys: self.keys.clone(),
            backup: self.backup.clone(),
            quiet,
            not_found_cache,
            clock: StoreClock::new(
                self.valkey.clone(),
                self.keys.clone(),
                deadlines,
                self.timings.wait_log_interval,
            ),
            permit: permit.clone(),
            own_puts: OwnPuts::default(),
            metrics: Arc::clone(&self.metrics),
            lease_holder,
            counters_sidecar_max: OnceLock::new(),
            allocator_purge: self.allocator_purge.clone(),
        };

        // Steps 4–5: wait out Q, planning in the background.
        let pending = match self.until_permit(&run, body, refresh).await {
            Ok(pending) => pending,
            Err(exit) => return exit,
        };
        permit.open();
        tracing::info!(
            mode = mode.name(),
            "the quiet period is over: the {mode} Job's write permit is open"
        );

        let outcome = tokio::select! {
            biased;
            () = refresh.lost().cancelled() => {
                let why = refresh.reason();
                permit.revoke(RevokeCause::LeaseLost(why.clone()));
                Err(RevokeCause::LeaseLost(why))
            }
            () = self.sigterm.cancelled() => {
                permit.revoke(RevokeCause::Shutdown);
                Err(RevokeCause::Shutdown)
            }
            exit = self.after_permit(&run, body, pending) => Ok(exit),
        };
        match outcome {
            Ok(exit) => exit,
            Err(cause) => self.revoked_exit(cause),
        }
    }

    /// Wait out the quiet period on the store's clock with the body's
    /// planning running in the background, until the first reading of the
    /// store's clock at or past Q's end.
    async fn until_permit<'a, B: JobBody>(
        &self,
        run: &'a JobRun,
        body: &'a B,
        refresh: &LeaseRefresh,
    ) -> Result<PendingPlan<'a, B::Plan>, JobExit> {
        let mode = self.mode;
        let deadlines = run.deadlines();
        let mut next_wait_log = Instant::now();
        let mut plan: Option<B::Plan> = None;
        let mut planning: Option<PlanFuture<'a, B::Plan>> = Some(body.plan(run));
        loop {
            // Wake for the next wait log, and for the quiet period's end on
            // this process's clock (the store's clock is read again on
            // waking). A failed read is retried at the wait-log cadence.
            let wake = match run.clock.now_ms().await {
                Ok(now_ms) => {
                    let remaining_ms = deadlines.quiet_ends_at_ms.saturating_sub(now_ms);
                    if remaining_ms == 0 {
                        return Ok(match (plan, planning) {
                            (Some(plan), _) => PendingPlan::Ready(plan),
                            (None, Some(planning)) => PendingPlan::Running(planning),
                            (None, None) => unreachable!("planning yields a plan or an exit"),
                        });
                    }
                    if Instant::now() >= next_wait_log {
                        next_wait_log = Instant::now() + self.timings.wait_log_interval;
                        tracing::info!(
                            mode = mode.name(),
                            remaining_secs = remaining_ms / 1000,
                            "the {mode} Job is waiting out the quiet period: {}s left on the \
                             store's clock before its first write",
                            remaining_ms / 1000
                        );
                    }
                    next_wait_log.min(Instant::now() + Duration::from_millis(remaining_ms))
                }
                Err(e) => {
                    tracing::warn!(
                        mode = mode.name(),
                        error = %e,
                        "the {mode} Job could not read the store's clock; trying again"
                    );
                    Instant::now() + self.timings.wait_log_interval
                }
            };
            tokio::select! {
                biased;
                () = self.sigterm.cancelled() => return Err(self.stopped_before_writing()),
                () = refresh.lost().cancelled() => return Err(self.lost_before_writing(refresh)),
                result = async { planning.as_mut().expect("guarded").await },
                    if planning.is_some() =>
                {
                    planning = None;
                    match result {
                        Ok(done) => plan = Some(done),
                        Err(exit) => return Err(exit),
                    }
                }
                () = tokio::time::sleep_until(wake) => {}
            }
        }
    }

    /// After the permit opened: finish the plan, check the counters sidecar
    /// and run the list-route canary (with a backup store), then the body's
    /// writes.
    async fn after_permit<'a, B: JobBody>(
        &self,
        run: &'a JobRun,
        body: &'a B,
        pending: PendingPlan<'a, B::Plan>,
    ) -> JobExit {
        let mode = self.mode;
        let plan = match pending {
            PendingPlan::Ready(plan) => Ok(plan),
            PendingPlan::Running(planning) => planning.await,
        };
        let plan = match plan {
            Ok(plan) => plan,
            Err(exit) => return exit,
        };
        if let BackupAccess::Store(store) = &run.backup {
            match check_counters_sidecar(store.as_ref(), &run.clock).await {
                Ok(SidecarCheck::Present(sidecar)) => {
                    let _ = run.counters_sidecar_max.set(sidecar.max_object_id_observed);
                }
                Ok(SidecarCheck::Unreadable(error)) => {
                    tracing::error!(
                        mode = mode.name(),
                        error = %error,
                        "the {mode} Job wrote nothing and exits 3; the service stays held: the \
                         backup store's counters record cannot be read ({error})"
                    );
                    return JobExit::StaysHeld;
                }
                Ok(SidecarCheck::Absent) => {
                    tracing::error!(
                        mode = mode.name(),
                        "the {mode} Job wrote nothing and exits 3; the service stays held: \
                         {SIDECAR_ABSENT_CAUSE}"
                    );
                    return JobExit::StaysHeld;
                }
                Err(e) => {
                    tracing::error!(
                        mode = mode.name(),
                        error = %e,
                        "the {mode} Job could not read the backup store's counters record, wrote \
                         nothing and exits 1"
                    );
                    return JobExit::Failed;
                }
            }
            match run_list_route_canary(store.as_ref(), &run.permit, &run.clock).await {
                Ok(CanaryCheck::Passed) => {
                    tracing::info!(
                        mode = mode.name(),
                        "the backup store's listing route listed back the list-route canary"
                    );
                }
                Ok(CanaryCheck::Failed(e)) => {
                    tracing::error!(
                        mode = mode.name(),
                        error = %e,
                        "the {mode} Job exits 3 and the service stays held: {LIST_ROUTE_CAUSE}"
                    );
                    return JobExit::StaysHeld;
                }
                Err(e) => {
                    tracing::error!(
                        mode = mode.name(),
                        error = %e,
                        "the {mode} Job's list-route canary failed with a backup-store fault; it \
                         exits 1 and the service stays held"
                    );
                    return JobExit::Failed;
                }
            }
        }
        body.execute(run, plan).await
    }

    fn stopped_before_writing(&self) -> JobExit {
        tracing::info!(
            mode = self.mode.name(),
            "the {mode} Job received SIGTERM before writing; it wrote nothing, releases its lease \
             and exits 1",
            mode = self.mode
        );
        JobExit::Failed
    }

    fn lost_before_writing(&self, refresh: &LeaseRefresh) -> JobExit {
        tracing::error!(
            mode = self.mode.name(),
            reason = %refresh.reason(),
            "the {mode} Job lost its lease before writing; it wrote nothing and exits 1",
            mode = self.mode
        );
        JobExit::Failed
    }

    fn revoked_exit(&self, cause: RevokeCause) -> JobExit {
        let mode = self.mode;
        match cause {
            RevokeCause::LeaseLost(why) => {
                tracing::error!(
                    mode = mode.name(),
                    reason = %why,
                    "the {mode} Job lost its lease, stopped writing and exits 1; the service \
                     stays held, and a re-run continues from what it wrote"
                );
                JobExit::Failed
            }
            RevokeCause::Shutdown => {
                tracing::info!(
                    mode = mode.name(),
                    "the {mode} Job received SIGTERM: it stopped writing, releases its lease and \
                     exits 1; the service stays held, and a re-run continues from what it wrote"
                );
                JobExit::Failed
            }
        }
    }
}

type PlanFuture<'a, P> = Pin<Box<dyn Future<Output = Result<P, JobExit>> + Send + 'a>>;

/// The plan as the permit opens.
enum PendingPlan<'a, P> {
    Ready(P),
    Running(PlanFuture<'a, P>),
}
