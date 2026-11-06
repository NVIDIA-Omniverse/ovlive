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

//! Hold entry, stale-verdict entry, the held heartbeat and the exits: what a
//! replica does with each hold verdict and with its age (SDD SR-6.1–6.5,
//! SR-7).

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::Duration;

use futures::future::BoxFuture;
use tokio::sync::watch;
use tokio::task::{AbortHandle, JoinHandle};
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;
use tokio_util::task::AbortOnDropHandle;

use super::gate::HoldGate;
use crate::dr::hold::{
    unhold_exit_jitter, HoldObservation, HoldVerdict, HoldWriter, HOLD_WRITERS,
    STALE_VERDICT_THRESHOLD,
};
use crate::grpc::graceful_shutdown::{GracefulShutdown, HoldCause};
use crate::observability::metrics::Metrics;
use crate::storage::client::UNRESPONSIVE_MAX_TIMEOUT;

/// How often a held replica repeats its WARN that it is still held (SDD SR-7).
pub const HOLD_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(5 * 60);

/// The code a replica exits with when it cannot act on the hold verdict (SDD
/// SR-6.5): no successful hold read for [`STALE_VERDICT_THRESHOLD`] before
/// its serving point, a hold poll that ended before shutdown, or a hold
/// reactor that failed (held mode included); and `main.rs`'s exit when the
/// one startup hold read fails. Non-zero, so the restart reads as a failure;
/// the new process starts with a new poll and reactor. A replica that is
/// serving or held never exits because reads fail: it keeps polling.
pub const NO_VERDICT_EXIT_CODE: i32 = 1;

/// How a replica ends its own process when the hold reactor decides to exit:
/// the exit seam. Production uses [`ProcessExit`]; a test injects one that
/// records the code instead of ending the test runner.
///
/// The reactor calls it once and then stops, whether or not the call returns.
pub trait HoldExit: Send + Sync + 'static {
    /// End the process with `code`.
    fn exit(&self, code: i32) -> BoxFuture<'static, ()>;
}

/// The production [`HoldExit`]: flush the trace exporter (bounded by
/// [`POST_BUDGET_FLUSH_BOUND`](crate::grpc::transport::POST_BUDGET_FLUSH_BOUND)),
/// then `std::process::exit(code)`. Log lines are written synchronously, so
/// every line logged before the call is already out.
#[derive(Clone, Copy, Debug, Default)]
pub struct ProcessExit;

impl HoldExit for ProcessExit {
    fn exit(&self, code: i32) -> BoxFuture<'static, ()> {
        Box::pin(async move {
            let _ = tokio::time::timeout(
                crate::grpc::transport::POST_BUDGET_FLUSH_BOUND,
                crate::observability::shutdown_tracing(),
            )
            .await;
            std::process::exit(code)
        })
    }
}

/// The hold reactor's settings, and the two signals it publishes.
#[derive(Clone)]
pub struct HoldReactorSettings {
    /// The interval of the WARN a held replica repeats:
    /// [`HOLD_HEARTBEAT_INTERVAL`] in production, shorter in a test. Must be
    /// non-zero: the heartbeat's ticker panics on a zero period.
    pub heartbeat_interval: Duration,
    /// The exit seam.
    pub exit: Arc<dyn HoldExit>,
    /// The readiness input (SDD SR-5.4): the reactor sets it `true` while this
    /// replica's hold verdict is stale, held or serving, and `false` when a
    /// held replica's read succeeds again. `main.rs`, or held mode, hands a
    /// receiver to the health supervisor
    /// (`HealthSupervisorConfig::verdict_stale`), which reports readiness
    /// `NOT_SERVING` while it is `true`.
    pub verdict_stale: watch::Sender<bool>,
    /// The scaling snapshot's `held` (SDD SR-7a): the reactor sets it `true`
    /// at hold entry ([`spawn_held_mode_reactor`]: at spawn), and `false` once
    /// the latest verdict is `Serving` again, just before the un-hold exit. A
    /// stale verdict leaves it as it is. `main.rs`, or held mode, hands a
    /// receiver to the `/scaling/v1` source, so the scaler freezes the replica
    /// count while it is `true`.
    pub held: watch::Sender<bool>,
}

impl HoldReactorSettings {
    /// The production settings: the 5-minute heartbeat, [`ProcessExit`], and
    /// both signals starting `false`, with no receiver yet (`subscribe` one).
    pub fn production() -> Self {
        Self {
            heartbeat_interval: HOLD_HEARTBEAT_INTERVAL,
            exit: Arc::new(ProcessExit),
            verdict_stale: watch::Sender::new(false),
            held: watch::Sender::new(false),
        }
    }
}

impl std::fmt::Debug for HoldReactorSettings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HoldReactorSettings")
            .field("heartbeat_interval", &self.heartbeat_interval)
            .field("verdict_stale", &*self.verdict_stale.borrow())
            .field("held", &*self.held.borrow())
            .finish_non_exhaustive()
    }
}

// ── The writer abort set ─────────────────────────────────────────────────────

#[derive(Default)]
struct WriterEntry {
    tasks: Vec<AbortHandle>,
    tokens: Vec<CancellationToken>,
    /// Why this writer was not spawned on this replica, when it was not.
    not_spawned: Option<&'static str>,
}

#[derive(Default)]
struct AbortsInner {
    entries: BTreeMap<HoldWriter, WriterEntry>,
    aborted: bool,
}

/// The abort routine's handles on the writer inventory
/// (`crate::dr::hold::HOLD_WRITERS`): for each [`HoldWriter`], the tasks to
/// abort and the tokens to cancel, registered where they are spawned.
///
/// [`Self::abort_all`] aborts every registered task first — so a task stops at
/// its next await instead of running the drain its token's cancel would start
/// (the external-value mirror workers drain their queue on cancel) — then
/// cancels every registered token, which stops what has no handle here: the
/// tasks a tracker spawns per call, the ones spawned after the abort, which
/// are born on a cancelled token, and the on-access repairs and restores,
/// whose hold write fence the token is. Anything registered after the abort is
/// aborted or cancelled at once.
///
/// A task aborted here ends with a cancelled `JoinError` at the shutdown
/// join, which `grpc::transport::run_post_server_sequence` reports as
/// aborted, not as a failure.
#[derive(Default)]
pub struct HoldWriterAborts {
    inner: Mutex<AbortsInner>,
}

impl std::fmt::Debug for HoldWriterAborts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.lock();
        f.debug_struct("HoldWriterAborts")
            .field("registered", &inner.entries.keys().collect::<Vec<_>>())
            .field("aborted", &inner.aborted)
            .finish()
    }
}

impl HoldWriterAborts {
    /// An empty set: nothing registered, nothing aborted.
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> MutexGuard<'_, AbortsInner> {
        // Every operation under the lock is a push, a flag set or a drain, so
        // a panic while it was held leaves it consistent.
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Register one task of `writer`, to be aborted at hold entry.
    pub fn register_task(&self, writer: HoldWriter, task: AbortHandle) {
        let mut inner = self.lock();
        if inner.aborted {
            task.abort();
        }
        inner.entries.entry(writer).or_default().tasks.push(task);
    }

    /// Register a token of `writer`, to be cancelled at hold entry, after
    /// every registered task is aborted.
    pub fn register_token(&self, writer: HoldWriter, token: CancellationToken) {
        let mut inner = self.lock();
        if inner.aborted {
            token.cancel();
        }
        inner.entries.entry(writer).or_default().tokens.push(token);
    }

    /// Record that `writer` is not running on this replica, and `why`, so the
    /// serving point's completeness check does not report it.
    pub fn register_not_spawned(&self, writer: HoldWriter, why: &'static str) {
        self.lock().entries.entry(writer).or_default().not_spawned = Some(why);
    }

    /// The inventory entries with neither a task, a token nor a
    /// not-spawned record: writers this replica would leave running at hold
    /// entry. Empty when the wiring covers the whole inventory.
    pub fn unregistered(&self) -> Vec<HoldWriter> {
        let inner = self.lock();
        HOLD_WRITERS
            .into_iter()
            .filter(|writer| {
                inner.entries.get(writer).is_none_or(|entry| {
                    entry.tasks.is_empty() && entry.tokens.is_empty() && entry.not_spawned.is_none()
                })
            })
            .collect()
    }

    /// Whether `writer` has a task or a token registered.
    pub fn is_registered(&self, writer: HoldWriter) -> bool {
        self.lock()
            .entries
            .get(&writer)
            .is_some_and(|entry| !entry.tasks.is_empty() || !entry.tokens.is_empty())
    }

    /// Whether [`Self::abort_all`] has run.
    pub fn is_aborted(&self) -> bool {
        self.lock().aborted
    }

    /// Abort every registered task, then cancel every registered token, and
    /// abort or cancel at once anything registered later. Returns the writers
    /// that had something registered, in inventory order. Idempotent.
    pub fn abort_all(&self) -> Vec<HoldWriter> {
        let mut inner = self.lock();
        inner.aborted = true;
        for entry in inner.entries.values() {
            for task in &entry.tasks {
                task.abort();
            }
        }
        for entry in inner.entries.values() {
            for token in &entry.tokens {
                token.cancel();
            }
        }
        HOLD_WRITERS
            .into_iter()
            .filter(|writer| {
                inner
                    .entries
                    .get(writer)
                    .is_some_and(|entry| !entry.tasks.is_empty() || !entry.tokens.is_empty())
            })
            .collect()
    }
}

// ── The serving point ────────────────────────────────────────────────────────

/// What the hold entry routine acts on. Every part exists only once the
/// replica has wired its service, so they are handed over together at the
/// serving point ([`HoldServingPoint::reach`]).
#[derive(Clone)]
pub struct HoldEntryParts {
    /// The LiveService gate.
    pub gate: HoldGate,
    /// The graceful-shutdown handle whose hold ending ends the live streams.
    pub streams: GracefulShutdown,
    /// The handles on the writer inventory.
    pub writers: Arc<HoldWriterAborts>,
    /// The hold signals' registry.
    pub metrics: Arc<Metrics>,
}

impl std::fmt::Debug for HoldEntryParts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HoldEntryParts")
            .field("gate", &self.gate)
            .field("streams", &self.streams)
            .field("writers", &self.writers)
            .finish_non_exhaustive()
    }
}

/// The replica's serving point, as the hold reactor sees it: unreached until
/// `main.rs` hands over the [`HoldEntryParts`], immediately before it opens
/// readiness and the full-API server takes the socket (Step 8). Cloning
/// shares it.
///
/// Before it is reached a `Held` verdict makes the process exit 0 at once;
/// after, it runs the hold entry routine. Reached once, never unreached.
#[derive(Clone, Default)]
pub struct HoldServingPoint {
    slot: Arc<OnceLock<HoldEntryParts>>,
}

impl std::fmt::Debug for HoldServingPoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HoldServingPoint")
            .field("reached", &self.is_reached())
            .finish()
    }
}

impl HoldServingPoint {
    /// An unreached serving point.
    pub fn new() -> Self {
        Self::default()
    }

    /// Reach the serving point with `parts`. Logs one ERROR naming every
    /// inventory entry the parts' abort set has nothing registered for: a
    /// store- or backup-writing task a hold would leave running. Returns
    /// `false`, keeping the first parts and logging nothing, when it was
    /// already reached.
    pub fn reach(&self, parts: HoldEntryParts) -> bool {
        let writers = Arc::clone(&parts.writers);
        if self.slot.set(parts).is_err() {
            return false;
        }
        let unregistered = writers.unregistered();
        if !unregistered.is_empty() {
            let names: Vec<&str> = unregistered.into_iter().map(HoldWriter::name).collect();
            tracing::error!(
                writers = ?names,
                "hold: no task or token is registered for these store- or backup-writing \
                 background tasks, so a hold entry would leave them running"
            );
        }
        true
    }

    /// Whether the serving point has been reached.
    pub fn is_reached(&self) -> bool {
        self.slot.get().is_some()
    }

    fn parts(&self) -> Option<&HoldEntryParts> {
        self.slot.get()
    }
}

// ── Entry and exits ──────────────────────────────────────────────────────────

/// The hold entry routine of a serving replica (SDD SR-6.1–6.3, SR-7), for a
/// `Held` `verdict`. In order:
///
/// 1. close the LiveService gate ([`HoldGate::close_held`]);
/// 2. end every live stream with the hold's cause
///    ([`GracefulShutdown::end_streams_for_hold`] with [`HoldCause::Held`]
///    and the verdict's incident), and refuse every later enrollment with it;
/// 3. abort the writer inventory ([`HoldWriterAborts::abort_all`]);
/// 4. set `live_edit_dr_held` to 1 and tick
///    `live_edit_dr_hold_entered_total{reason}` once
///    ([`Metrics::record_dr_hold_entered`]);
/// 5. log one ERROR naming the incident, reason and evidence.
///
/// The held heartbeat is the reactor's ([`spawn_hold_reactor`]). Touches no
/// per-session token, no shutdown token and no readiness status.
pub fn enter_hold(parts: &HoldEntryParts, verdict: &HoldVerdict) {
    parts.gate.close_held();
    parts.streams.end_streams_for_hold(HoldCause::Held {
        incident: verdict.incident().map(str::to_owned),
    });
    let aborted = parts.writers.abort_all();
    parts.metrics.record_dr_hold_entered(verdict.reason());
    let aborted: Vec<&str> = aborted.into_iter().map(HoldWriter::name).collect();
    let detected_at_ms = match verdict {
        HoldVerdict::Held { detected_at_ms, .. } => *detected_at_ms,
        HoldVerdict::Serving => None,
    };
    tracing::error!(
        incident = verdict.incident().unwrap_or("unknown"),
        reason = verdict.reason().unwrap_or("unknown"),
        evidence = verdict.evidence().unwrap_or("unknown"),
        detected_at_ms,
        aborted_writers = ?aborted,
        "the fleet is held for disaster recovery: this replica refuses every LiveService RPC \
         SERVICE_HELD, has ended its live streams and aborted its store- and backup-writing \
         tasks, and writes nothing more to the backing store from LIVE_EDIT_RPC_DEADLINE_MS + \
         LIVE_EDIT_VALKEY_COMMAND_TIMEOUT_MS + {wait_secs} s after now: its in-flight calls end \
         by their deadline and its streams' work ends now, each clean-up they leave is cut at \
         the command timeout, a write not yet sent when its caller gives up is never sent, and \
         one already being sent leaves within {wait_secs} s (startup refuses a \
         LIVE_EDIT_VALKEY_COMMAND_TIMEOUT_MS of 0). This sets aside the teardown of the streams \
         it ended (a few writes, each cut at the command timeout) and a write already sent when \
         the network or the backing store stalls. It stays held until an operator restores the \
         backing store or accepts its loss, then restarts",
        wait_secs = UNRESPONSIVE_MAX_TIMEOUT.as_secs(),
    );
}

/// The exit of a normal-mode replica that sees `Held` after its startup hold
/// evaluation but before its serving point: log one WARN naming the incident,
/// then exit 0 at once through `exit`, so the replica restarts into held mode.
/// Called by the reactor ([`spawn_hold_reactor`]). A startup evaluation that
/// already answers `Held` starts held mode instead
/// ([`run_held_mode`](super::run_held_mode)).
pub async fn exit_held_before_serving(observation: &HoldObservation, exit: &dyn HoldExit) {
    tracing::warn!(
        incident = observation.verdict.incident().unwrap_or("unknown"),
        reason = observation.verdict.reason().unwrap_or("unknown"),
        "the fleet is held for disaster recovery and this replica has not started serving; \
         exiting 0 so it restarts held"
    );
    exit.exit(0).await;
}

/// The stale-verdict entry routine of a serving replica that is not held
/// (SDD SR-6.5): the hold entry routine's steps with the stale verdict's
/// cause, in order:
///
/// 1. close the LiveService gate for a stale verdict
///    ([`HoldGate::close_stale`]): every RPC is refused
///    `UPSTREAM_UNAVAILABLE`;
/// 2. end every live stream with [`HoldCause::StaleVerdict`]
///    ([`GracefulShutdown::end_streams_for_hold`]), and refuse every later
///    enrollment with it;
/// 3. abort the writer inventory ([`HoldWriterAborts::abort_all`]), the same
///    set a hold aborts;
/// 4. set `live_edit_dr_hold_verdict_stale` to 1
///    ([`Metrics::set_dr_hold_verdict_stale`]);
/// 5. log one WARN.
///
/// Writes no marker: this is not a fleet hold. Never for a replica already
/// held: its gate stays `SERVICE_HELD`, its streams already ended and its
/// writers are already aborted, and a second hold ending would replace the
/// late-enrolment cause `Held` with `StaleVerdict`, so a late stream would
/// end `UPSTREAM_UNAVAILABLE` while the gate answers `SERVICE_HELD`. The
/// reactor handles that case on its own (its stale WARN and gauge only).
/// Touches no per-session token and no shutdown token; the readiness input
/// is the reactor's ([`HoldReactorSettings::verdict_stale`]).
pub fn enter_stale_verdict(parts: &HoldEntryParts) {
    parts.gate.close_stale();
    parts.streams.end_streams_for_hold(HoldCause::StaleVerdict);
    let aborted = parts.writers.abort_all();
    parts.metrics.set_dr_hold_verdict_stale(true);
    let aborted: Vec<&str> = aborted.into_iter().map(HoldWriter::name).collect();
    tracing::warn!(
        aborted_writers = ?aborted,
        "this replica has had no successful disaster-recovery hold read for {threshold_secs} s, \
         so it cannot tell whether the service is held: it now refuses every LiveService RPC \
         UPSTREAM_UNAVAILABLE, has ended its live streams and aborted its store- and \
         backup-writing tasks, and reports readiness NOT_SERVING. It keeps reading, with no time \
         limit, and restarts once a read succeeds",
        threshold_secs = STALE_VERDICT_THRESHOLD.as_secs(),
    );
}

/// Spawn the hold reactor: the task that acts on every verdict the hold poll
/// publishes on `verdicts`, and on the age of the latest one (SDD SR-6, the
/// lifecycle rows "Hold poll task" and "Local stale-verdict gate").
///
/// On a verdict:
///
/// - `Held` before `serving_point` is reached: [`exit_held_before_serving`].
/// - `Held` once it is reached: [`enter_hold`], once, then
///   `settings.held` set `true` and a WARN every `settings.heartbeat_interval`
///   while the latest verdict is `Held`, naming its incident.
/// - `Serving` after a hold entry: `settings.held` set `false`, a wait of
///   [`unhold_exit_jitter`], then exit 0 through `settings.exit`. Never
///   reopens the gate.
///
/// On the latest verdict's age (SDD SR-6.5), measured from its observation's
/// `at` on the reactor's own timer:
///
/// - [`STALE_VERDICT_THRESHOLD`] old before the serving point: exit
///   [`NO_VERDICT_EXIT_CODE`] at once, as a `Held` verdict there exits 0
///   (Task 3's rule B1): nothing a stale entry stops exists yet.
/// - [`STALE_VERDICT_THRESHOLD`] old on a serving replica that is not held:
///   [`enter_stale_verdict`], then `settings.verdict_stale` set `true`. Its
///   next published verdict, whatever it says, waits [`unhold_exit_jitter`]
///   and exits 0, so the replica restarts into normal service or, if that
///   verdict is `Held`, held. Never reopens the gate.
/// - [`STALE_VERDICT_THRESHOLD`] old on a held replica:
///   `settings.verdict_stale` set `true`, `live_edit_dr_hold_verdict_stale`
///   1 and one WARN; the gate, the streams and the writers stay as the hold
///   left them. Its next published verdict clears both signals and is then
///   acted on as above (`Serving` exits 0 after the jitter; `Held` keeps it
///   held).
///
/// Once stale, held or serving, a replica waits on no clock: it keeps reading,
/// with no time limit, and only a publication or shutdown moves it.
///
/// The current value of `verdicts` is acted on at once, then every change.
/// Stops, acting on nothing more, when `shutdown` is cancelled — `main.rs`
/// cancels it on SIGTERM or SIGINT, so a SIGTERM while held runs the normal
/// paced shutdown, one during an exit wait cancels the exit, and a paced
/// shutdown that outlasts the stale threshold is never read as a stale
/// verdict (the poll stops on the same token, so its verdict ages). Shutting
/// down is read from that token only, and before the channel. A closed
/// channel — the poll ended, or panicked, without the token — is not a
/// shutdown: the replica can no longer see the verdict, so the reactor logs
/// one ERROR and exits [`NO_VERDICT_EXIT_CODE`] at once, before or after the
/// serving point. Stops after calling the exit seam.
///
/// The returned handle is a watcher's, and it resolves only once the reactor
/// task has ended: the watcher waits on that task, and if it fails (panics
/// or is cancelled) before `shutdown`, logs one ERROR and exits
/// [`NO_VERDICT_EXIT_CODE`] through `settings.exit`, as held mode does on its
/// reactor's failure. A replica whose reactor has failed would no longer
/// react to a hold. Once `shutdown` is cancelled the watcher only waits for
/// the reactor to stop on that token, and ignores how it ended. Aborting the
/// returned handle aborts the reactor too, even before the watcher first
/// runs.
pub fn spawn_hold_reactor(
    verdicts: watch::Receiver<HoldObservation>,
    serving_point: HoldServingPoint,
    shutdown: CancellationToken,
    settings: HoldReactorSettings,
) -> JoinHandle<()> {
    let exit = Arc::clone(&settings.exit);
    // Owned by the watcher's future from here on, so dropping that future,
    // whether the watcher was aborted before or after it first ran, aborts
    // the reactor.
    let reactor = AbortOnDropHandle::new(tokio::spawn(run_hold_reactor(
        verdicts,
        serving_point,
        shutdown.clone(),
        settings,
    )));
    tokio::spawn(watch_hold_reactor(reactor, shutdown, exit))
}

/// Wait on the normal-mode reactor task until it ends, and exit
/// [`NO_VERDICT_EXIT_CODE`] through `exit` if it failed before `shutdown`.
/// Once `shutdown` is cancelled, wait for the reactor to stop on that token
/// and ignore its result: a failure during shutdown exits nothing. Returns
/// only after the reactor has ended, so the handle `spawn_hold_reactor`
/// returns means "the reactor has stopped".
async fn watch_hold_reactor(
    mut reactor: AbortOnDropHandle<()>,
    shutdown: CancellationToken,
    exit: Arc<dyn HoldExit>,
) {
    let joined = tokio::select! {
        biased;
        () = shutdown.cancelled() => None,
        joined = &mut reactor => Some(joined),
    };
    let Some(joined) = joined else {
        let _ended = reactor.await;
        return;
    };
    if let Err(e) = joined {
        tracing::error!(
            error = %e,
            "hold: the hold reactor failed, so this replica no longer acts on the hold verdict or \
             its age; exiting non-zero so it restarts"
        );
        exit.exit(NO_VERDICT_EXIT_CODE).await;
    }
}

/// Where a replica stands on its verdict's age (SDD SR-6.5).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Staleness {
    /// The latest verdict is younger than the stale threshold, or no stale
    /// entry has run since the last publication.
    Fresh,
    /// A serving replica that was not held went stale and ran
    /// [`enter_stale_verdict`]: its next publication exits 0.
    StaleServing,
    /// A held replica went stale: its next publication clears the signals
    /// and is acted on as a verdict.
    StaleHeld,
}

/// Wait out a jitter of [`unhold_exit_jitter`] and exit 0 through `exit`,
/// unless `shutdown` is cancelled first. `what` opens the INFO line, which
/// carries the jitter as `jitter_ms`.
async fn exit_zero_after_jitter(exit: &dyn HoldExit, shutdown: &CancellationToken, what: &str) {
    let jitter = unhold_exit_jitter();
    tracing::info!(
        jitter_ms = u64::try_from(jitter.as_millis()).unwrap_or(u64::MAX),
        "{what}",
    );
    tokio::select! {
        biased;
        () = shutdown.cancelled() => return,
        () = tokio::time::sleep(jitter) => {}
    }
    exit.exit(0).await;
}

async fn run_hold_reactor(
    mut verdicts: watch::Receiver<HoldObservation>,
    serving_point: HoldServingPoint,
    shutdown: CancellationToken,
    settings: HoldReactorSettings,
) {
    // Set at hold entry: the heartbeat, aborted when the reactor returns.
    let mut heartbeat: Option<AbortOnDropHandle<()>> = None;
    let mut staleness = Staleness::Fresh;
    // Whether the value about to be read was published since the last read:
    // a successful hold read. `false` for the first read, which is the
    // startup evaluation's, and after the stale threshold passed.
    let mut published = false;
    loop {
        if shutdown.is_cancelled() {
            return;
        }
        let observation = verdicts.borrow_and_update().clone();
        if std::mem::take(&mut published) {
            match staleness {
                Staleness::Fresh => {}
                Staleness::StaleServing => {
                    exit_zero_after_jitter(
                        settings.exit.as_ref(),
                        &shutdown,
                        "the disaster-recovery hold read succeeded again after this replica's \
                         verdict went stale; it exits 0 after its jitter and restarts, held if \
                         the service is held",
                    )
                    .await;
                    return;
                }
                Staleness::StaleHeld => {
                    staleness = Staleness::Fresh;
                    clear_held_verdict_stale(
                        serving_point.parts().map(|parts| parts.metrics.as_ref()),
                        &settings,
                        &observation.verdict,
                    );
                }
            }
        }
        match (&observation.verdict, heartbeat.is_some()) {
            (HoldVerdict::Held { .. }, false) => match serving_point.parts() {
                None => {
                    exit_held_before_serving(&observation, settings.exit.as_ref()).await;
                    return;
                }
                Some(parts) => {
                    enter_hold(parts, &observation.verdict);
                    settings.held.send_replace(true);
                    heartbeat = Some(AbortOnDropHandle::new(tokio::spawn(run_heartbeat(
                        verdicts.clone(),
                        settings.heartbeat_interval,
                        shutdown.clone(),
                    ))));
                }
            },
            (HoldVerdict::Serving, true) => {
                settings.held.send_replace(false);
                exit_zero_after_jitter(
                    settings.exit.as_ref(),
                    &shutdown,
                    "the disaster-recovery hold cleared; this replica exits 0 after its jitter \
                     and restarts into normal service",
                )
                .await;
                return;
            }
            _ => {}
        }
        // The one clock this replica acts on, from its last successful read:
        // the stale threshold, while fresh. Once stale it waits on no clock.
        let deadline = match staleness {
            Staleness::Fresh => Some(observation.stale_at()),
            Staleness::StaleServing | Staleness::StaleHeld => None,
        };
        tokio::select! {
            biased;
            () = shutdown.cancelled() => return,
            changed = verdicts.changed() => match changed {
                Ok(()) => published = true,
                Err(_) => {
                    exit_hold_poll_ended(
                        settings.exit.as_ref(),
                        heartbeat.is_some(),
                        staleness != Staleness::Fresh,
                    )
                    .await;
                    return;
                }
            },
            // Fires only while fresh: a stale replica has no deadline.
            () = sleep_until_if_any(deadline) => {
                let since_last_success_secs = observation.at.elapsed().as_secs();
                match serving_point.parts() {
                    None => {
                        tracing::error!(
                            since_last_success_secs,
                            "no successful disaster-recovery hold read for {stale_secs} s before \
                             this replica started serving; it exits non-zero so it restarts with \
                             a fresh connection to the store",
                            stale_secs = STALE_VERDICT_THRESHOLD.as_secs(),
                        );
                        settings.exit.exit(NO_VERDICT_EXIT_CODE).await;
                        return;
                    }
                    Some(parts) if heartbeat.is_some() => {
                        staleness = Staleness::StaleHeld;
                        mark_held_verdict_stale(&parts.metrics, &settings, &observation.verdict);
                    }
                    Some(parts) => {
                        staleness = Staleness::StaleServing;
                        enter_stale_verdict(parts);
                        settings.verdict_stale.send_replace(true);
                    }
                }
            }
        }
    }
}

/// Spawn the held-mode reactor: the hold reactor of a process that found the
/// fleet held at startup and runs held mode
/// ([`run_held_mode`](super::run_held_mode); SDD SR-5, SR-6.4, SR-6.5). It
/// acts on every verdict the hold poll publishes on `verdicts`, and on the
/// age of the latest one.
///
/// Not [`spawn_hold_reactor`]: a held-mode process has no serving point and
/// never reaches one, and that reactor's arms before the serving point would
/// make a restart loop of a hold (a `Held` verdict exits 0 at once) and exit
/// a stale one at 30 s instead of reporting readiness `NOT_SERVING`. Here:
///
/// - **At spawn,** `settings.held` is set `true` (the scaling snapshot's
///   `held`, so the scaler freezes the replica count; SDD SR-7a) and the held
///   heartbeat starts: a WARN every `settings.heartbeat_interval` while the
///   latest verdict is `Held`.
/// - **`Held`:** nothing more. The refuse-all stub reads the incident from
///   the same channel at each refusal.
/// - **`Serving`:** `settings.held` set `false`, a wait of
///   [`unhold_exit_jitter`], then exit 0 through `settings.exit`, so the
///   replica restarts into normal service. The first value of `verdicts` is
///   acted on at once like any other, so a `Serving` startup observation
///   exits too.
/// - **[`STALE_VERDICT_THRESHOLD`] with no successful read:**
///   `settings.verdict_stale` set `true` (readiness `NOT_SERVING`),
///   `live_edit_dr_hold_verdict_stale` 1, one WARN. The replica stays held,
///   refusing `SERVICE_HELD`, and does not exit.
/// - **No successful read for any longer:** nothing more. The replica keeps
///   reading, with no time limit.
/// - **A successful read after that:** both signals cleared, one INFO, then
///   the verdict is acted on as above.
/// - **The channel closes** (the poll ended, or panicked, without the
///   shutdown token): one ERROR, then exit [`NO_VERDICT_EXIT_CODE`] at once.
///
/// The stale threshold runs on the reactor's own timer from the latest
/// observation's `at`. Stops, acting on nothing more, when `shutdown` is
/// cancelled (held mode's shutdown path, on the first SIGTERM or SIGINT; read
/// before the channel), and after calling the exit seam.
pub fn spawn_held_mode_reactor(
    verdicts: watch::Receiver<HoldObservation>,
    metrics: Arc<Metrics>,
    shutdown: CancellationToken,
    settings: HoldReactorSettings,
) -> JoinHandle<()> {
    settings.held.send_replace(true);
    tokio::spawn(run_held_mode_reactor(verdicts, metrics, shutdown, settings))
}

async fn run_held_mode_reactor(
    mut verdicts: watch::Receiver<HoldObservation>,
    metrics: Arc<Metrics>,
    shutdown: CancellationToken,
    settings: HoldReactorSettings,
) {
    let _heartbeat = AbortOnDropHandle::new(tokio::spawn(run_heartbeat(
        verdicts.clone(),
        settings.heartbeat_interval,
        shutdown.clone(),
    )));
    // Whether the latest verdict has gone stale: set at the stale threshold,
    // cleared by the next successful read.
    let mut stale = false;
    // Whether the value about to be read was published since the last read:
    // a successful hold read. `false` for the first read, the startup
    // evaluation's.
    let mut published = false;
    loop {
        if shutdown.is_cancelled() {
            return;
        }
        let observation = verdicts.borrow_and_update().clone();
        if std::mem::take(&mut published) && stale {
            stale = false;
            clear_held_verdict_stale(Some(&metrics), &settings, &observation.verdict);
        }
        if !observation.verdict.is_held() {
            settings.held.send_replace(false);
            exit_zero_after_jitter(
                settings.exit.as_ref(),
                &shutdown,
                "the disaster-recovery hold cleared; this replica exits 0 after its jitter and \
                 restarts into normal service",
            )
            .await;
            return;
        }
        // The stale threshold, while fresh; once stale, no clock.
        let deadline = (!stale).then(|| observation.stale_at());
        tokio::select! {
            biased;
            () = shutdown.cancelled() => return,
            changed = verdicts.changed() => match changed {
                Ok(()) => published = true,
                Err(_) => {
                    exit_hold_poll_ended(settings.exit.as_ref(), true, stale).await;
                    return;
                }
            },
            () = sleep_until_if_any(deadline) => {
                stale = true;
                mark_held_verdict_stale(&metrics, &settings, &observation.verdict);
            }
        }
    }
}

/// A held replica's verdict went stale (SDD SR-6.5): set
/// `live_edit_dr_hold_verdict_stale` and the `verdict_stale` signal, and log
/// one WARN. Nothing else changes: it stays held.
fn mark_held_verdict_stale(
    metrics: &Metrics,
    settings: &HoldReactorSettings,
    verdict: &HoldVerdict,
) {
    metrics.set_dr_hold_verdict_stale(true);
    settings.verdict_stale.send_replace(true);
    tracing::warn!(
        incident = verdict.incident().unwrap_or("unknown"),
        "this held replica has had no successful disaster-recovery hold read for {stale_secs} s: \
         it stays held, refusing every LiveService RPC SERVICE_HELD, reports readiness \
         NOT_SERVING, and keeps reading, with no time limit, until a read succeeds",
        stale_secs = STALE_VERDICT_THRESHOLD.as_secs(),
    );
}

/// A held replica whose verdict was stale read successfully again: clear the
/// `verdict_stale` signal and, when `metrics` is given,
/// `live_edit_dr_hold_verdict_stale`, and log one INFO. The verdict read is
/// acted on afterwards by the caller.
fn clear_held_verdict_stale(
    metrics: Option<&Metrics>,
    settings: &HoldReactorSettings,
    verdict: &HoldVerdict,
) {
    settings.verdict_stale.send_replace(false);
    if let Some(metrics) = metrics {
        metrics.set_dr_hold_verdict_stale(false);
    }
    tracing::info!(
        held = verdict.is_held(),
        "the disaster-recovery hold read succeeded again; this held replica's verdict is no \
         longer stale"
    );
}

/// The hold poll's channel closed before shutdown: this replica can no
/// longer see the hold verdict. Log one ERROR and exit
/// [`NO_VERDICT_EXIT_CODE`] through `exit`, so it restarts with a new poll.
async fn exit_hold_poll_ended(exit: &dyn HoldExit, held: bool, stale: bool) {
    tracing::error!(
        held,
        stale,
        "hold: the hold poll ended before shutdown, so this replica can no longer see the hold \
         verdict; it exits non-zero so it restarts with a new poll"
    );
    exit.exit(NO_VERDICT_EXIT_CODE).await;
}

/// Sleep until `deadline`, or forever when there is none.
async fn sleep_until_if_any(deadline: Option<Instant>) {
    match deadline {
        Some(deadline) => tokio::time::sleep_until(deadline).await,
        None => std::future::pending().await,
    }
}

/// The held heartbeat: a WARN every `interval` while the latest verdict is
/// `Held`, naming its incident and how long this replica has been held, until
/// `shutdown`.
async fn run_heartbeat(
    verdicts: watch::Receiver<HoldObservation>,
    interval: Duration,
    shutdown: CancellationToken,
) {
    let entered = Instant::now();
    let mut ticker = tokio::time::interval_at(entered + interval, interval);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            biased;
            () = shutdown.cancelled() => return,
            _ = ticker.tick() => {}
        }
        let verdict = verdicts.borrow().verdict.clone();
        if let HoldVerdict::Held { .. } = verdict {
            tracing::warn!(
                incident = verdict.incident().unwrap_or("unknown"),
                reason = verdict.reason().unwrap_or("unknown"),
                held_for_secs = entered.elapsed().as_secs(),
                "this replica is still held for disaster recovery: it refuses every LiveService \
                 RPC SERVICE_HELD and has stopped its store- and backup-writing tasks, until an \
                 operator restores the backing store or accepts its loss"
            );
        }
    }
}
