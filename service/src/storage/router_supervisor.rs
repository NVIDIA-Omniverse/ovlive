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

//! Supervision of every pooled Valkey client's fred router task.
//!
//! Each fred client runs one **router task**: the task that owns the client's
//! connections, writes the commands callers queue on its channel and routes the
//! replies back. `Client::connect()` spawns it and returns its `ConnectHandle`
//! (a `JoinHandle`). When that task ends, nothing else in fred marks it: the
//! client still reports `Connected`, its command channel stays open, and no
//! `on_error` / `on_reconnect` event fires. Every later command on the client
//! waits for its timeout and fails, and `force_reconnection()` never answers,
//! because there is no router left to serve it. In a pool, fred keeps routing
//! that member's share of the traffic to it. The client stays like that until
//! the process restarts. The `ConnectHandle` is the only signal that the task
//! has ended.
//!
//! This module watches that signal for every client the service opens: each
//! member of the `data`, `txn`, `maintenance` and `snapshot` pools, the pub/sub
//! `subscriber`, and each stream reader's current client (`subscriber_pool`).
//! One watcher task per member, on one [`TaskTracker`] and one
//! [`CancellationToken`] (a retirable watch — a stream reader's, or a `data`,
//! `maintenance` or `snapshot` member's — runs on a child of it, so it can be
//! retired alone). When a member's handle ends:
//!
//! - **`Ok(Ok(()))`** — the router stopped because the client was `quit()`.
//!   Never restarted. For a `txn` slot that is a slot recycle in progress
//!   (`ValkeyClient::repair_held_txn_slot` quits the client and connects it
//!   again), and the watcher waits for the new handle the recycle hands over
//!   ([`RouterSupervisor::hand_over`]) — unless the slot's client was replaced
//!   (below), in which case nothing is handed over and the watcher exits.
//! - **Any end of a `txn` router marked retired** — never counted or
//!   restarted. Two things mark one. A repair that aborted it
//!   ([`RouterSupervisor::abort_current`]): a recycle in progress, handled as
//!   above, the watcher waiting for the hand-over. Or a slot replacement
//!   ([`RouterSupervisor::retire_slot`]): the client left its slot and its
//!   watch was forgotten with its hand-over channel. The router it marks is
//!   the slot's newest, so the watcher, which always moves on to the newest
//!   router handed over before it reads an end, reaches that one, sees it end
//!   marked, finds the channel closed and exits.
//! - **`Ok(Err(_))`** (the router task returned an error) **or `Err(JoinError)`**
//!   (it panicked or was aborted) — unless supervision is being shut down, the
//!   watcher logs one ERROR line (`Valkey client router task ended; restarting
//!   it`, with `pool`, `member`, how it ended and the restart count), increments
//!   `live_edit_valkey_router_restarts_total{pool}`, waits the storm-guard delay
//!   and restarts the router.
//!
//! **How a router is restarted.** A multiplexed member (`data`,
//! `subscriber_pool`, `maintenance`, `snapshot`, the pub/sub `subscriber`) is
//! restarted with `client.connect()`. It does not block and it returns the new
//! handle. After an `Ok(Err(_))` end it reuses the command channel the ended
//! task stored back, so commands queued on the client while it had no router
//! are served by the new one. After a panic or abort the receiver died with
//! the task: `connect()` builds a fresh channel, and the commands queued on the
//! old one fail instead of being written. The client's event hooks need no
//! re-arm: only `quit()` swaps fred's
//! broadcast senders. Two other calls look like fixes and are not:
//! `force_reconnection()` has no timeout and is served by the router itself, so
//! on an ended router it never answers; `init()` blocks until the connection is
//! up and, if it fails, drops the handle the watcher needs. A `txn` slot is
//! restarted through the checkout instead, under a permit and the slot's own
//! guard, so nothing borrows it mid-rebuild: `ValkeyClient` takes the slot by
//! client id (`ExclusiveCheckout::checkout_slot_for_router_restart`), runs the
//! same repair a dirty checkout gets, and the repair hands the new handle over.
//! When `storage::member_replacement` swaps a fresh client into a `txn` slot,
//! the old client's watch is ended by its id ([`RouterSupervisor::retire_slot`]:
//! forgotten, its router marked retired so its end counts no restart, and its
//! hand-over channel closed, so its watcher exits instead of waiting for a
//! recycle, per the `Ok(Ok(()))` and retired-router cases above) and the fresh
//! client is watched under its own id, so the slot's watch follows its client.
//! A restart of the old client's router that was already under way finds the
//! slot no longer holds that client and is skipped.
//!
//! **A repair aborts the slot's old router before it starts the new one.**
//! `quit()` alone does not end a router in step with the repair. When the
//! repair's bounded `QUIT` times out (a `probe_timeout` recycle, whose
//! connection is not answering), the old router is still running, draining
//! the command that hung. When `QUIT` succeeds, fred answers it once the
//! command is written, and the router ends only after a further await, so it
//! can still be alive too. An old router still alive when `connect()` starts
//! the new one ends by returning later, and fred then stores its command
//! receiver back on the client: a receiver whose sender `connect()` has
//! already replaced. A later
//! repair's `quit()` would then see a stored receiver and return at once, and
//! its `connect()` would start a router on that dead channel, the one the
//! watcher then watches, while the router the earlier repair started kept
//! carrying the slot's commands, unrepaired and unwatched. So every repair
//! calls [`RouterSupervisor::abort_current`] between `quit()` and `connect()`.
//! A router still running is marked as retired, aborted, and awaited, bounded
//! by [`ROUTER_ABORT_TIMEOUT`]. A router the abort has ended drops its
//! receiver instead of storing it, so `connect()` builds a fresh channel, and
//! the watcher counts no restart for it. The residual is a router that has not
//! ended within that bound: the repair logs a WARN and connects anyway, and
//! that router can still store its receiver back once it does end. An abort
//! takes effect at the router's next await, so the bound is reached only by a
//! router caught mid-poll, which keeps the residual rare. A router that had
//! already ended when the repair reached
//! it is neither aborted nor marked, so its watcher reads that end by the
//! policy above. To make this hold, the supervisor registers each `txn` slot's
//! router when the repair hands it over, not when the watcher next polls. The
//! abort therefore always reaches the router the previous repair started.
//!
//! **Stream readers: watched from first connect, across replacement.** A
//! stream reader connects lazily, the first time it serves a subscription, and
//! on a cluster replaces its client with one to another primary when the slot
//! it serves is owned elsewhere (`storage::stream_mux`). Each reader client is
//! watched from the moment its first connect succeeds
//! ([`RouterSupervisor::watch_retirable`]) and restarted like any multiplexed
//! member if its router dies. When the reader replaces or drops that client it
//! retires the watch first ([`RouterSupervisor::retire`]): the watcher is
//! cancelled and awaited, so the client's end — its `QUIT`, or the abort of a
//! router whose `QUIT` did not finish — is never read as a death, never
//! counted and never restarted. The retire hands back the abort handle of the
//! client's current router (the watcher may have restarted it), which is the
//! one the reader aborts once its `QUIT` has returned, failed or overrun, so a
//! discarded client's router never keeps reconnecting it.
//!
//! **`data`, `maintenance` and `snapshot` members: retirable too.** A member of
//! these pools can be replaced by a fresh client
//! (`storage::member_replacement`), so each is watched the same way from
//! startup, and the fresh client from the connect that made it. The
//! replacement retires the old client's watch before anything ends that
//! client, so its end is not counted as a restart either.
//!
//! **Storm guard.** The first restart of a member is immediate. A member whose
//! router ends again within [`ROUTER_RESTART_QUICK_WINDOW`] of its previous
//! restart waits [`ROUTER_RESTART_INITIAL_DELAY`] first, doubling with each
//! further quick restart up to [`ROUTER_RESTART_MAX_DELAY`]; a router that stays
//! up for the window resets the count ([`RestartBackoff`]). A restart that
//! fails to connect ends the new task at once (fred's `fail_fast`), which the
//! watcher sees as one more end and backs off the same way. Fixed constants,
//! no configuration: the cap is also the longest a dead member blackholes its
//! share of a pool's traffic in one stretch.
//!
//! **Shutdown.** `ValkeyClient::quit()` cancels the token and drains the
//! tracker ([`RouterSupervisor::shutdown`]) before it closes any pool, so the
//! `QUIT`s that follow end routers that nothing will restart. When the
//! client's last owner drops without `quit()`, the token is cancelled without
//! the drain ([`RouterSupervisor::release`]).
//!
//! **Relation to the vendored fred patch.** The measured cause of a router
//! ending was fred 10.1.0's `process_ask` returning the error of an `-ASK` the
//! command could not follow; the patch in `vendor/fred-10.1.0` removes it
//! (`vendor/fred-10.1.0.SOURCE.md`). Other ways out of the router loop remain in
//! fred — a responder error the loop propagates, "Missing expected
//! connection.", an error fred will not reconnect on, a panic — and each would
//! otherwise stall its member for the life of the pod. This module turns any of
//! them into a counted, logged restart.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use fred::types::ConnectHandle;
use futures::future::{BoxFuture, FutureExt, Shared};
use tokio::sync::mpsc;
use tokio::task::AbortHandle;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;

use crate::observability::metrics::SharedMetrics;

/// A member whose router ends again within this long of its previous restart
/// restarts after a delay ([`router_restart_delay`]); a router that has stayed
/// up this long resets the member's back-off.
pub const ROUTER_RESTART_QUICK_WINDOW: Duration = Duration::from_secs(60);

/// The delay before the first quick restart (the second restart within
/// [`ROUTER_RESTART_QUICK_WINDOW`]); it doubles with each further one.
pub const ROUTER_RESTART_INITIAL_DELAY: Duration = Duration::from_millis(100);

/// The cap on the delay before a restart.
pub const ROUTER_RESTART_MAX_DELAY: Duration = Duration::from_secs(5);

/// How long [`RouterSupervisor::shutdown`] waits for the watchers to exit. They
/// exit at their next await once cancelled, so this only bounds a watcher stuck
/// in a slot repair's synchronous part. One of `ValkeyClient::quit`'s
/// sequential bounds, summed against the shutdown-budget headroom by a `const`
/// assertion in `storage::client`.
pub(crate) const ROUTER_SUPERVISOR_DRAIN_TIMEOUT: Duration = Duration::from_secs(2);

/// How long [`RouterSupervisor::abort_current`] waits for an aborted `txn`
/// router to end. An abort ends a task at its next await, which for a router
/// waiting on its socket or its command channel is at once, so this only
/// bounds a router that is mid-poll when the abort lands. It is part of a
/// dirty checkout's worst case (`ValkeyClient`'s budget assertion).
pub(crate) const ROUTER_ABORT_TIMEOUT: Duration = Duration::from_millis(500);

/// The delay before a restart, given how many quick restarts in a row this one
/// makes: none for the first restart of a storm (`0`), then
/// [`ROUTER_RESTART_INITIAL_DELAY`] doubling per quick restart, capped at
/// [`ROUTER_RESTART_MAX_DELAY`].
pub fn router_restart_delay(quick_restarts: u32) -> Duration {
    if quick_restarts == 0 {
        return Duration::ZERO;
    }
    // 100 ms × 2^(n-1); the shift is clamped well past the cap so it cannot
    // overflow.
    let factor = 1u32 << (quick_restarts - 1).min(16);
    ROUTER_RESTART_INITIAL_DELAY
        .saturating_mul(factor)
        .min(ROUTER_RESTART_MAX_DELAY)
}

/// One member's storm-guard state: when its router was last restarted, and how
/// many restarts in a row came within [`ROUTER_RESTART_QUICK_WINDOW`] of the
/// previous one.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct RestartBackoff {
    last_restart: Option<Instant>,
    quick_restarts: u32,
}

impl RestartBackoff {
    /// A router ended at `now`: return how long to wait before restarting it.
    ///
    /// The restart counts as quick when the previous restart was less than
    /// [`ROUTER_RESTART_QUICK_WINDOW`] ago; otherwise the member's count resets
    /// and the restart is immediate.
    pub fn next_delay(&mut self, now: Instant) -> Duration {
        let quick = self.last_restart.is_some_and(|previous| {
            now.saturating_duration_since(previous) < ROUTER_RESTART_QUICK_WINDOW
        });
        self.quick_restarts = if quick {
            self.quick_restarts.saturating_add(1)
        } else {
            0
        };
        router_restart_delay(self.quick_restarts)
    }

    /// Record that the router was restarted at `at`.
    pub fn mark_restarted(&mut self, at: Instant) {
        self.last_restart = Some(at);
    }

    /// The restart about to happen, counted within its storm: 1 for the first
    /// restart, 2 for the first quick one, and so on. Read after
    /// [`Self::next_delay`].
    pub fn storm_restarts(&self) -> u32 {
        self.quick_restarts.saturating_add(1)
    }
}

/// How a watcher restarts its member's router.
pub(crate) enum RouterRestart {
    /// A multiplexed client: call `connect()` on it and watch the handle it
    /// returns.
    Connect(Box<dyn Fn() -> ConnectHandle + Send + Sync>),
    /// A `txn` slot: run the repair under the checkout. The new handle arrives
    /// through [`RouterSupervisor::hand_over`], as every slot recycle's does.
    Repair(Box<dyn Fn() -> BoxFuture<'static, ()> + Send + Sync>),
}

/// How a router task ended: its `ConnectHandle`'s output, reduced to what the
/// watcher logs, so it can be [`Shared`] between the watcher and
/// [`RouterSupervisor::abort_current`] (a `JoinError` is not `Clone`).
#[derive(Debug, Clone)]
enum RouterEnd {
    /// `Ok(Ok(()))`: the router stopped because the client was `quit()`.
    Quit,
    /// `Ok(Err(_))`: the router task returned an error.
    Error { kind: &'static str, detail: String },
    /// `Err(JoinError)`: the task panicked (`how` `panic`) or was aborted
    /// (`how` `aborted`).
    Join { how: &'static str, detail: String },
}

/// One router task under supervision. Clones share the task.
#[derive(Clone)]
struct SupervisedRouter {
    /// Resolves once the task has ended. Both the watcher and
    /// [`RouterSupervisor::abort_current`] await it.
    ended: Shared<BoxFuture<'static, RouterEnd>>,
    abort: AbortHandle,
    /// Marks a `txn` slot router whose end is not a router death, so the
    /// watcher counts no restart for it. Set by
    /// [`RouterSupervisor::abort_current`] before it aborts the task (a slot
    /// repair in progress, which hands the next router over), and by
    /// [`RouterSupervisor::retire_slot`] when a slot replacement swaps the
    /// client out (no repair follows and nothing is handed over: the watcher
    /// exits). A router marked so is never recorded for the test-only
    /// `RouterSupervisor::end_router_for_test`.
    retired: Arc<AtomicBool>,
}

impl SupervisedRouter {
    fn new(handle: ConnectHandle) -> Self {
        let abort = handle.abort_handle();
        let ended = async move {
            match handle.await {
                Ok(Ok(())) => RouterEnd::Quit,
                Ok(Err(error)) => RouterEnd::Error {
                    kind: error.kind().to_str(),
                    detail: error.details().to_owned(),
                },
                Err(join_error) => RouterEnd::Join {
                    how: if join_error.is_panic() {
                        "panic"
                    } else {
                        "aborted"
                    },
                    detail: join_error.to_string(),
                },
            }
        }
        .boxed()
        .shared();
        Self {
            ended,
            abort,
            retired: Arc::new(AtomicBool::new(false)),
        }
    }
}

/// A `txn` slot's watcher, as [`RouterSupervisor::hand_over`] and
/// [`RouterSupervisor::abort_current`] reach it.
struct SlotWatch {
    /// The watcher's hand-over channel.
    handover: mpsc::UnboundedSender<SupervisedRouter>,
    /// The router the slot's latest repair started (or, before any repair, the
    /// one the slot was connected with). Updated when the router is handed
    /// over, not when the watcher receives it, so a repair that follows at
    /// once still reaches it.
    current: SupervisedRouter,
}

/// What [`RouterSupervisor::abort_current`] found and did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AbortCurrent {
    /// The router was running. It was aborted, and has ended (by the abort, or
    /// on its own just before it).
    Ended,
    /// The router had already ended before the call: nothing was aborted, and
    /// the watcher treats the end as it would have without the repair.
    AlreadyEnded,
    /// The router was aborted but had not ended within
    /// [`ROUTER_ABORT_TIMEOUT`]. It may still store its command receiver back
    /// once it does end.
    StillRunning,
    /// No router is registered for the client id: supervision was shut down
    /// (or the id names no `txn` slot). Nothing was aborted.
    Unsupervised,
}

/// One retirable watch ([`RouterSupervisor::watch_retirable`]): the
/// supervision of a client its member replaces — a stream reader's, or a
/// `data`, `maintenance` or `snapshot` pool member's
/// (`storage::member_replacement`) — ended by [`RouterSupervisor::retire`]
/// before the member quits or drops that client.
pub(crate) struct RetirableWatch {
    pool: &'static str,
    member: usize,
    client_id: String,
    /// Cancels this watcher alone; a child of the supervisor's token, so
    /// [`RouterSupervisor::shutdown`] ends it too.
    cancel: CancellationToken,
    /// The watcher task, awaited by the retire.
    watcher: tokio::task::JoinHandle<()>,
    /// The abort handle of the client's current router: the one it connected
    /// with, or the latest restart the watcher made. The watcher's restart
    /// holds this lock from its cancellation check through `connect()` to
    /// storing the new handle, so a reader of it under the lock after a
    /// cancellation sees the client's last router.
    current: Arc<Mutex<AbortHandle>>,
    /// Set by [`AbandonHandle::abandon`] (never by [`RouterSupervisor::retire`])
    /// before it cancels, so the watcher's cancel arm aborts the router it is
    /// watching rather than leaving it for a `QUIT`.
    abandoned: Arc<AtomicBool>,
}

impl RetirableWatch {
    /// A handle that ends this watch without awaiting anything, for a drop
    /// guard that cannot await [`RouterSupervisor::retire`] (a member
    /// replacement aborted mid-way, `storage::member_replacement`).
    pub(crate) fn abandon_handle(&self) -> AbandonHandle {
        AbandonHandle {
            cancel: self.cancel.clone(),
            current: Arc::clone(&self.current),
            abandoned: Arc::clone(&self.abandoned),
        }
    }
}

/// Ends one [`RetirableWatch`] synchronously ([`RetirableWatch::abandon_handle`]).
#[derive(Clone)]
pub(crate) struct AbandonHandle {
    cancel: CancellationToken,
    current: Arc<Mutex<AbortHandle>>,
    abandoned: Arc<AtomicBool>,
}

impl AbandonHandle {
    /// End the watch without awaiting anything: mark it abandoned, cancel the
    /// watcher, so nothing below is read as a router death, then — under the
    /// `current` lock — read the client's current router (the one it connected
    /// with, or the watcher's latest restart of it) and abort it. Only a drop
    /// guard that cannot await [`RouterSupervisor::retire`] uses it.
    ///
    /// **No router outlives it.** A retirable watch's restart runs under the
    /// same `current` lock: it checks for cancellation, calls `connect()` and
    /// stores the new handle before releasing the lock. So either the restart
    /// takes the lock after this cancellation and starts nothing, or it
    /// finished storing its router before this read, and this aborts that
    /// router. The returned handle is therefore the client's last router, and
    /// a caller that waits on it (`pool::retire`'s release) waits for every
    /// router the client had.
    ///
    /// **The flag.** The `abandoned` mark is set before the cancellation, so
    /// the watcher's cancel arm, seeing it, aborts the router it watches before
    /// returning, where a plain [`RouterSupervisor::retire`] (which never sets
    /// it) leaves the router to the caller's `QUIT`. With the lock in place
    /// that abort only repeats this one: defense in depth.
    ///
    /// Blocks only while a restart holds the lock, which spans a synchronous
    /// `connect()` (it spawns the router task) and no await.
    pub(crate) fn abandon(&self) -> AbortHandle {
        self.abandoned.store(true, Ordering::SeqCst);
        self.cancel.cancel();
        let router = self
            .current
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        router.abort();
        router
    }
}

impl std::fmt::Debug for RetirableWatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RetirableWatch")
            .field("pool", &self.pool)
            .field("member", &self.member)
            .field("client_id", &self.client_id)
            .finish()
    }
}

/// What a retirable watch's watcher shares with its [`RetirableWatch`] and
/// [`AbandonHandle`]: the `current` router handle and the `abandoned` mark.
struct Retirable {
    current: Arc<Mutex<AbortHandle>>,
    abandoned: Arc<AtomicBool>,
}

/// Supervises the router task of every pooled Valkey client; see the module
/// docs. One per `ValkeyClient`, shared by its clones through an `Arc`.
pub struct RouterSupervisor {
    tracker: TaskTracker,
    cancel: CancellationToken,
    /// Shared with `ValkeyClient`; `None` until `install_observability_hooks`
    /// runs, in which case a restart is logged but not counted.
    metrics: Arc<Mutex<Option<SharedMetrics>>>,
    /// Each `txn` slot's watcher, by fred client id.
    slots: Mutex<HashMap<String, SlotWatch>>,
    /// Test-only: the abort handle of each member's current router task, by
    /// `(pool, member)`, for [`Self::end_router_for_test`].
    #[cfg(any(test, feature = "test-support"))]
    routers: Mutex<HashMap<(&'static str, usize), tokio::task::AbortHandle>>,
    /// Test-only: the fred client id each live retirable watch supervises, by
    /// `(pool, member)`, for [`Self::retirable_watches_for_test`].
    #[cfg(any(test, feature = "test-support"))]
    retirable: Mutex<HashMap<(&'static str, usize), String>>,
}

impl std::fmt::Debug for RouterSupervisor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RouterSupervisor")
            .field("watchers", &self.tracker.len())
            .field("cancelled", &self.cancel.is_cancelled())
            .finish()
    }
}

impl RouterSupervisor {
    /// A supervisor with no watchers yet. `metrics` is the slot `ValkeyClient`
    /// fills in `install_observability_hooks`.
    pub(crate) fn new(metrics: Arc<Mutex<Option<SharedMetrics>>>) -> Self {
        Self {
            tracker: TaskTracker::new(),
            cancel: CancellationToken::new(),
            metrics,
            slots: Mutex::new(HashMap::new()),
            #[cfg(any(test, feature = "test-support"))]
            routers: Mutex::new(HashMap::new()),
            #[cfg(any(test, feature = "test-support"))]
            retirable: Mutex::new(HashMap::new()),
        }
    }

    /// Watch `handle`, the router task of member `member` of pool `pool` (a
    /// `POOL_LABEL_*` value), restarting it with `restart` when it ends.
    ///
    /// `txn_client_id` is the fred client id of a `txn` slot, whose restart is
    /// [`RouterRestart::Repair`]: the watcher is registered for
    /// [`Self::hand_over`] and [`Self::abort_current`] under it. `None` for
    /// every multiplexed member.
    pub(crate) fn watch(
        self: &Arc<Self>,
        pool: &'static str,
        member: usize,
        handle: ConnectHandle,
        restart: RouterRestart,
        txn_client_id: Option<String>,
    ) {
        let router = SupervisedRouter::new(handle);
        // Now, not when the watcher first polls, as `watch_retirable` does: a
        // slot replacement's fresh router must replace the old client's entry
        // before anything can end it.
        self.record_router(pool, member, &router, None);
        let handovers = txn_client_id.map(|client_id| {
            let (tx, rx) = mpsc::unbounded_channel();
            if let Ok(mut map) = self.slots.lock() {
                map.insert(
                    client_id,
                    SlotWatch {
                        handover: tx,
                        current: router.clone(),
                    },
                );
            }
            rx
        });
        let supervisor = Arc::clone(self);
        let cancel = self.cancel.clone();
        self.tracker.spawn(async move {
            supervisor
                .run_watcher(pool, member, router, restart, handovers, cancel, None)
                .await;
        });
    }

    /// Watch `handle`, the router task of the client `client_id` that member
    /// `member` of pool `pool` currently uses, restarting it with `restart`
    /// when it ends — until [`Self::retire`] ends the watch.
    ///
    /// For a member that replaces its client (a stream reader, or a `data`,
    /// `maintenance` or `snapshot` pool member, `storage::member_replacement`):
    /// register each client once its first connect succeeded, so a first
    /// connect is never read as a restart, and retire the watch before quitting
    /// or dropping the client. A watch registered after [`Self::shutdown`]
    /// exits at once.
    pub(crate) fn watch_retirable(
        self: &Arc<Self>,
        pool: &'static str,
        member: usize,
        client_id: &str,
        handle: ConnectHandle,
        restart: RouterRestart,
    ) -> RetirableWatch {
        let router = SupervisedRouter::new(handle);
        let current = Arc::new(Mutex::new(router.abort.clone()));
        let abandoned = Arc::new(AtomicBool::new(false));
        let cancel = self.cancel.child_token();
        // Now, not when the watcher first polls, so a test can end the router
        // as soon as the connect that registered it returned.
        self.record_router(pool, member, &router, None);
        #[cfg(any(test, feature = "test-support"))]
        if let Ok(mut map) = self.retirable.lock() {
            map.insert((pool, member), client_id.to_owned());
        }
        let supervisor = Arc::clone(self);
        let watcher = {
            let cancel = cancel.clone();
            let retirable = Retirable {
                current: Arc::clone(&current),
                abandoned: Arc::clone(&abandoned),
            };
            self.tracker.spawn(async move {
                supervisor
                    .run_watcher(pool, member, router, restart, None, cancel, Some(retirable))
                    .await;
            })
        };
        RetirableWatch {
            pool,
            member,
            client_id: client_id.to_owned(),
            cancel,
            watcher,
            current,
            abandoned,
        }
    }

    /// End a [`Self::watch_retirable`] watch before its client is quit or
    /// dropped: cancel the watcher and wait for it to exit (bounded by
    /// [`ROUTER_ABORT_TIMEOUT`]; it exits at its next await), so whatever ends
    /// the client's router from here on is neither counted nor restarted.
    ///
    /// Returns the abort handle of the client's current router — the one it
    /// connected with, or the watcher's latest restart of it — for the caller
    /// to abort once its `QUIT` is done, however it ended.
    pub(crate) async fn retire(&self, watch: RetirableWatch) -> AbortHandle {
        let RetirableWatch {
            pool,
            member,
            client_id,
            cancel,
            watcher,
            current,
            abandoned: _,
        } = watch;
        cancel.cancel();
        if tokio::time::timeout(ROUTER_ABORT_TIMEOUT, watcher)
            .await
            .is_err()
        {
            tracing::warn!(
                pool,
                member,
                timeout_ms = ROUTER_ABORT_TIMEOUT.as_millis() as u64,
                "a retired router watcher did not exit in time; it exits at its next await"
            );
        }
        let current = current
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        self.forget_retired(pool, member, &client_id, &current);
        current
    }

    #[cfg(any(test, feature = "test-support"))]
    fn forget_retired(
        &self,
        pool: &'static str,
        member: usize,
        client_id: &str,
        router: &AbortHandle,
    ) {
        if let Ok(mut routers) = self.routers.lock() {
            if routers
                .get(&(pool, member))
                .is_some_and(|recorded| recorded.id() == router.id())
            {
                routers.remove(&(pool, member));
            }
        }
        if let Ok(mut map) = self.retirable.lock() {
            if map.get(&(pool, member)).is_some_and(|id| id == client_id) {
                map.remove(&(pool, member));
            }
        }
    }

    #[cfg(not(any(test, feature = "test-support")))]
    fn forget_retired(
        &self,
        _pool: &'static str,
        _member: usize,
        _client_id: &str,
        _router: &AbortHandle,
    ) {
    }

    /// The fred client id each live [`Self::watch_retirable`] watch of pool
    /// `pool` supervises, by member. A watch leaves the map when it is
    /// retired; [`Self::shutdown`] clears it. **Test-only.**
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) fn retirable_watches_for_test(&self, pool: &str) -> HashMap<usize, String> {
        self.retirable
            .lock()
            .map(|map| {
                map.iter()
                    .filter(|((p, _), _)| *p == pool)
                    .map(|((_, member), id)| (*member, id.clone()))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Hand the router task a `txn` slot repair just started to that slot's
    /// watcher, and record it as the slot's current router for the next
    /// [`Self::abort_current`]. Called by every repair, probe-driven or
    /// router-restart. The repair aborted the slot's previous router first, so
    /// the router handed over here is the one serving the slot.
    ///
    /// With no watcher for `client_id` (supervision shut down) the handle is
    /// dropped, which detaches the task without ending it.
    pub(crate) fn hand_over(&self, client_id: &str, handle: ConnectHandle) {
        let router = SupervisedRouter::new(handle);
        let Ok(mut map) = self.slots.lock() else {
            return;
        };
        let Some(slot) = map.get_mut(client_id) else {
            return;
        };
        if slot.handover.send(router.clone()).is_ok() {
            slot.current = router;
        } else {
            // The watcher has exited (shutdown): the handle is dropped and the
            // task detached, as with no watcher at all, and nothing is left
            // for a later repair to abort through this entry.
            map.remove(client_id);
        }
    }

    /// End the router the `txn` slot `client_id` currently runs, so the slot's
    /// repair can start its replacement without leaving it running beside the
    /// new one. Called by every repair between its `quit()` and its
    /// `connect()`, while the repair holds the slot; see the module docs for
    /// why `quit()` alone does not do this.
    ///
    /// A router still running is marked retired, so its watcher reads the end
    /// as the repair in progress and counts no restart, and is then aborted
    /// and awaited for up to [`ROUTER_ABORT_TIMEOUT`]. Once it has ended it can
    /// no longer store its command receiver back on the client: an aborted
    /// router drops it, and one that returned on its own first stored it before
    /// it ended, while that receiver was still the client's current one.
    ///
    /// The wait does not involve the watcher, so it is safe from inside the
    /// watcher's own router restart (whose router has already ended).
    pub(crate) async fn abort_current(&self, client_id: &str) -> AbortCurrent {
        let current = match self.slots.lock() {
            Ok(map) => map.get(client_id).map(|slot| slot.current.clone()),
            Err(_) => None,
        };
        let Some(current) = current else {
            return AbortCurrent::Unsupervised;
        };
        if current.abort.is_finished() {
            // Ended before this repair reached it: a death the watcher
            // handles as one, or the `quit()` above ending it.
            return AbortCurrent::AlreadyEnded;
        }
        // Before the abort, so the watcher can never see the abort's end
        // without the mark.
        current.retired.store(true, Ordering::SeqCst);
        current.abort.abort();
        match tokio::time::timeout(ROUTER_ABORT_TIMEOUT, current.ended).await {
            Ok(_) => AbortCurrent::Ended,
            Err(_) => AbortCurrent::StillRunning,
        }
    }

    /// End supervision of the `txn` slot client `client_id` because its slot's
    /// client is being replaced (`storage::member_replacement`): forget its
    /// watch, so nothing is left under the old id, and mark its current router
    /// retired, so however that router now ends — the retirement's `QUIT`, or
    /// its abort — the watcher counts no restart and exits (it finds no
    /// hand-over coming). Returns the abort handle of that router, for the
    /// retirement to end it. `None` when no watch has `client_id`
    /// (supervision shut down).
    pub(crate) fn retire_slot(&self, client_id: &str) -> Option<AbortHandle> {
        let slot = match self.slots.lock() {
            Ok(mut map) => map.remove(client_id),
            Err(_) => None,
        }?;
        // Before anything ends it, so the watcher never sees the end unmarked.
        slot.current.retired.store(true, Ordering::SeqCst);
        // Dropping `slot` drops the hand-over sender: the watcher records no
        // router from here on (`record_router`), and once the router ends it
        // finds the channel closed and exits.
        Some(slot.current.abort.clone())
    }

    /// The fred client ids of the `txn` slots under supervision. A replaced
    /// slot's id leaves it as the fresh client's joins. **Test-only.**
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) fn slot_watch_ids_for_test(&self) -> Vec<String> {
        self.slots
            .lock()
            .map(|map| map.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// Stop supervising: cancel every watcher and wait (bounded) for them to
    /// exit. Called by `ValkeyClient::quit` before any pool is closed, so the
    /// routers its `QUIT`s end are not restarted. Idempotent.
    pub(crate) async fn shutdown(&self) {
        self.release();
        if tokio::time::timeout(ROUTER_SUPERVISOR_DRAIN_TIMEOUT, self.tracker.wait())
            .await
            .is_err()
        {
            tracing::warn!(
                watchers_remaining = self.tracker.len(),
                timeout_secs = ROUTER_SUPERVISOR_DRAIN_TIMEOUT.as_secs(),
                "timed out draining the Valkey router supervisor; closing the pools anyway"
            );
        }
        if let Ok(mut map) = self.slots.lock() {
            map.clear();
        }
        #[cfg(any(test, feature = "test-support"))]
        if let Ok(mut map) = self.retirable.lock() {
            map.clear();
        }
    }

    /// Stop supervising without waiting: cancel every watcher, which exits at
    /// its next await as on shutdown and drops what it holds — a `txn` slot's
    /// restart holds an internal clone of the `ValkeyClient`. The first part
    /// of [`Self::shutdown`], and what the drop of the client's last owner
    /// does here (`ValkeyClient`'s `ClientOwners`), which cannot wait.
    /// Idempotent.
    pub(crate) fn release(&self) {
        self.cancel.cancel();
        self.tracker.close();
    }

    /// The number of live watcher tasks. Zero once [`Self::shutdown`] has
    /// drained them. **Test-only.**
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) fn watchers(&self) -> usize {
        self.tracker.len()
    }

    /// Abort the current router task of member `member` of pool `pool` (a
    /// `POOL_LABEL_*` value), as a panic in it would end it. Returns whether
    /// such a router was registered. **Test-only.**
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) fn end_router_for_test(&self, pool: &str, member: usize) -> bool {
        let routers = match self.routers.lock() {
            Ok(routers) => routers,
            Err(_) => return false,
        };
        match routers
            .iter()
            .find(|((p, m), _)| *p == pool && *m == member)
        {
            Some((_, abort)) => {
                abort.abort();
                true
            }
            None => false,
        }
    }

    /// Record `router` as member `member` of pool `pool`'s current router for
    /// [`Self::end_router_for_test`] — unless it is marked retired, or
    /// `handovers`, its `txn` slot watcher's hand-over channel, is closed. A
    /// slot replacement ([`Self::retire_slot`]) does both before the fresh
    /// client's [`Self::watch`] records the fresh router, and the check runs
    /// under the record's lock, so the old client's watcher, taking a router a
    /// repair handed over before the swap, never overwrites the fresh entry.
    #[cfg(any(test, feature = "test-support"))]
    fn record_router(
        &self,
        pool: &'static str,
        member: usize,
        router: &SupervisedRouter,
        handovers: Option<&mpsc::UnboundedReceiver<SupervisedRouter>>,
    ) {
        if let Ok(mut routers) = self.routers.lock() {
            let superseded = router.retired.load(Ordering::SeqCst)
                || handovers.is_some_and(mpsc::UnboundedReceiver::is_closed);
            if !superseded {
                routers.insert((pool, member), router.abort.clone());
            }
        }
    }

    #[cfg(not(any(test, feature = "test-support")))]
    fn record_router(
        &self,
        _pool: &'static str,
        _member: usize,
        _router: &SupervisedRouter,
        _handovers: Option<&mpsc::UnboundedReceiver<SupervisedRouter>>,
    ) {
    }

    /// One member's watcher; see the module docs for the policy. `cancel` is
    /// the supervisor's token, or a retirable watch's child of it; `retirable`,
    /// for a retirable watch, carries the `current` handle it keeps pointing at
    /// the router being watched and the `abandoned` mark its cancel arm reads.
    #[allow(clippy::too_many_arguments)]
    async fn run_watcher(
        &self,
        pool: &'static str,
        member: usize,
        mut router: SupervisedRouter,
        restart: RouterRestart,
        mut handovers: Option<mpsc::UnboundedReceiver<SupervisedRouter>>,
        cancel: CancellationToken,
        retirable: Option<Retirable>,
    ) {
        let mut backoff = RestartBackoff::default();
        loop {
            self.record_router(pool, member, &router, handovers.as_ref());
            if let Some(retirable) = &retirable {
                *retirable
                    .current
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = router.abort.clone();
            }
            let ended = tokio::select! {
                biased;
                () = cancel.cancelled() => {
                    // An abandoned watch (`AbandonHandle::abandon`) aborts the
                    // router it watches; a retired one (`retire`) leaves it to
                    // the caller's `QUIT`.
                    if retirable
                        .as_ref()
                        .is_some_and(|r| r.abandoned.load(Ordering::SeqCst))
                    {
                        router.abort.abort();
                    }
                    return;
                }
                Some(next) = next_handover(&mut handovers) => {
                    // A slot recycle replaced the router; the old handle is
                    // dropped, which detaches whatever is left of that task
                    // (nothing, when the recycle aborted it first).
                    router = next;
                    continue;
                }
                ended = &mut router.ended => ended,
            };

            // A `txn` slot's router marked retired or `quit()`: never counted,
            // never restarted. A repair that aborted it (`abort_current`) or a
            // recycle that quit it hands over the router it starts next, which
            // the watcher takes. A slot replacement that marked it
            // (`retire_slot`) marked the newest router handed over before the
            // swap and dropped the hand-over sender, so the channel reads
            // closed and the watcher exits. A multiplexed member's router is
            // never marked so (a retirable watch is cancelled instead), and
            // once `quit()` it is done.
            let retired = router.retired.load(Ordering::SeqCst);
            let (how, kind, detail) = match ended {
                _ if retired => match self.await_handover(&mut handovers).await {
                    Some(next) => {
                        router = next;
                        continue;
                    }
                    None => return,
                },
                RouterEnd::Quit => {
                    if handovers.is_none() {
                        return;
                    }
                    match self.await_handover(&mut handovers).await {
                        Some(next) => {
                            router = next;
                            continue;
                        }
                        None => return,
                    }
                }
                RouterEnd::Error { kind, detail } => ("error", kind, detail),
                RouterEnd::Join { how, detail } => (how, "", detail),
            };
            if cancel.is_cancelled() {
                return;
            }

            let delay = backoff.next_delay(Instant::now());
            tracing::error!(
                pool,
                member,
                ended = how,
                error_kind = kind,
                error = %detail,
                storm_restarts = backoff.storm_restarts(),
                restart_delay_ms = delay.as_millis() as u64,
                "Valkey client router task ended; restarting it"
            );
            if let Some(metrics) = self.metrics.lock().ok().and_then(|slot| slot.clone()) {
                metrics.inc_valkey_router_restart(pool);
            }
            if !delay.is_zero() {
                tokio::select! {
                    biased;
                    () = cancel.cancelled() => return,
                    () = tokio::time::sleep(delay) => {}
                }
            }

            router = match &restart {
                // A retirable watch restarts under its `current` lock — check
                // for cancellation, `connect()`, store the handle, unlock — so
                // `AbandonHandle::abandon`, which cancels and then reads
                // `current` under the same lock, either keeps this restart from
                // starting anything or reads and aborts the router it started.
                // `connect()` only spawns the router task; nothing awaits
                // under the lock (the guard is dropped at the block's end).
                // The closure decides the runtime: a snapshot lane member's
                // restarts on `snapshot-io`, where the lane's routers run so
                // the fan-out's reply frames come from one arena
                // (`storage::fanout_thread::connect_router_on`).
                RouterRestart::Connect(connect) => match &retirable {
                    Some(retirable) => {
                        let mut current = retirable
                            .current
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        if cancel.is_cancelled() {
                            return;
                        }
                        let next = SupervisedRouter::new(connect());
                        *current = next.abort.clone();
                        next
                    }
                    None => SupervisedRouter::new(connect()),
                },
                RouterRestart::Repair(repair) => {
                    tokio::select! {
                        biased;
                        () = cancel.cancelled() => return,
                        () = repair() => {}
                    }
                    match self.await_handover(&mut handovers).await {
                        Some(next) => next,
                        None => return,
                    }
                }
            };
            backoff.mark_restarted(Instant::now());
        }
    }

    /// The newest handle waiting on `handovers`, else the next one to arrive;
    /// `None` on cancellation or when the channel is gone.
    async fn await_handover(
        &self,
        handovers: &mut Option<mpsc::UnboundedReceiver<SupervisedRouter>>,
    ) -> Option<SupervisedRouter> {
        let receiver = handovers.as_mut()?;
        let mut newest = None;
        // A repair hands over exactly one handle, but two repairs of the same
        // slot can complete before the watcher reads (a recycle and this
        // restart, serialised on the slot's guard); the later one is current.
        while let Ok(next) = receiver.try_recv() {
            newest = Some(next);
        }
        if newest.is_some() {
            return newest;
        }
        tokio::select! {
            biased;
            () = self.cancel.cancelled() => None,
            next = receiver.recv() => next,
        }
    }
}

/// The next handed-over handle, or never for a member without a hand-over
/// channel (every multiplexed member).
async fn next_handover(
    handovers: &mut Option<mpsc::UnboundedReceiver<SupervisedRouter>>,
) -> Option<SupervisedRouter> {
    match handovers.as_mut() {
        Some(receiver) => receiver.recv().await,
        None => std::future::pending().await,
    }
}

#[cfg(test)]
mod tests {
    //! b.v6d T37 — the storm guard. The policy is operator-facing (SDD §15.3):
    //! the first restart of a member is immediate; a
    //! member whose router ends again within 60 s of its previous restart
    //! waits 100 ms, doubling per further quick restart up to 5 s; 60 s of
    //! uptime resets it. The numbers are restated here rather than read from
    //! the constants, so a change to the policy is a deliberate edit of both.
    //!
    //! Also three `abort_current` cases (b.v6d defer-9), below the storm guard.

    use super::*;

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    #[test]
    fn router_restart_delay_is_immediate_then_doubles_from_100_ms_to_a_5_s_cap() {
        let delays: Vec<Duration> = (0..=8).map(router_restart_delay).collect();
        assert_eq!(
            delays,
            [
                ms(0),
                ms(100),
                ms(200),
                ms(400),
                ms(800),
                ms(1_600),
                ms(3_200),
                ms(5_000),
                ms(5_000),
            ],
            "the n-th quick restart in a row waits 100 ms × 2^(n-1), capped at 5 s; the first \
             restart of a storm waits nothing"
        );
        assert_eq!(
            router_restart_delay(u32::MAX),
            ms(5_000),
            "the doubling must saturate at the cap, never overflow"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn restart_backoff_counts_quick_restarts_and_resets_after_60_s_of_uptime() {
        let mut backoff = RestartBackoff::default();
        let start = Instant::now();

        let mut ends = Vec::new();
        // A storm: each end 10 ms after the previous restart.
        let mut now = start;
        for _ in 0..3 {
            let delay = backoff.next_delay(now);
            ends.push((delay, backoff.storm_restarts()));
            backoff.mark_restarted(now + delay);
            now = now + delay + ms(10);
        }
        assert_eq!(
            ends,
            [(ms(0), 1), (ms(100), 2), (ms(200), 3)],
            "a first end restarts at once; each end within the window of the previous restart is \
             one more quick restart"
        );

        // The router then stays up for the whole window: the storm is over.
        let last_restart = now - ms(10);
        let after_window = last_restart + Duration::from_secs(60);
        assert_eq!(
            (backoff.next_delay(after_window), backoff.storm_restarts()),
            (ms(0), 1),
            "60 s after the previous restart an end counts as a fresh storm"
        );
        backoff.mark_restarted(after_window);
        assert_eq!(
            backoff.next_delay(after_window + Duration::from_secs(60) - ms(1)),
            ms(100),
            "an end just inside the window is quick again"
        );
    }

    /// The watcher itself applies the guard: with paused time, a router that
    /// keeps ending is restarted at once, then 100 ms later, then 200 ms after
    /// that. Nothing else advances the clock, so the gaps are exact.
    #[tokio::test(start_paused = true)]
    async fn a_router_that_ends_again_quickly_waits_the_storm_delay_before_its_restart() {
        fn ended(detail: &'static str) -> ConnectHandle {
            tokio::spawn(
                async move { Err(fred::error::Error::new(fred::error::ErrorKind::IO, detail)) },
            )
        }

        let supervisor = Arc::new(RouterSupervisor::new(Arc::new(Mutex::new(None))));
        let (restarted_tx, mut restarted) = mpsc::unbounded_channel::<Instant>();
        let restarts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let restart = {
            let restarts = Arc::clone(&restarts);
            RouterRestart::Connect(Box::new(move || {
                let n = restarts.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
                let _ = restarted_tx.send(Instant::now());
                if n < 3 {
                    ended("the restarted router ended again")
                } else {
                    tokio::spawn(std::future::pending())
                }
            }))
        };

        let start = Instant::now();
        supervisor.watch("data", 0, ended("the router ended"), restart, None);
        let mut at = Vec::new();
        for _ in 0..3 {
            at.push(
                restarted
                    .recv()
                    .await
                    .expect("the watcher restarts the router"),
            );
        }
        assert_eq!(
            [at[0] - start, at[1] - at[0], at[2] - at[1]],
            [ms(0), ms(100), ms(200)],
            "first restart immediate; the second end inside the window waits 100 ms, the third \
             200 ms"
        );

        supervisor.shutdown().await;
        assert_eq!(supervisor.watchers(), 0, "shutdown drains the watcher");
    }

    // b.v6d defer-9 — `abort_current`. The end-to-end behaviour (a repair
    // whose `QUIT` timed out leaves the next repair able to rebuild the slot;
    // an aborted router is neither counted nor restarted) is in
    // `tests/integration/storage/b_v6d_router_test.rs`'s `old_router_abort`.
    // These three pin what no real fred router can be made to show there.

    /// A `txn` slot's restart, for the tests that never reach one: a repair
    /// that does nothing.
    fn idle_repair() -> RouterRestart {
        RouterRestart::Repair(Box::new(|| async {}.boxed()))
    }

    /// A router task that never ends on its own.
    fn running_router() -> ConnectHandle {
        tokio::spawn(std::future::pending())
    }

    /// The router a repair hands over is the one the NEXT repair aborts, even
    /// when that repair follows before the slot's watcher has read the
    /// hand-over: `hand_over` records it as the slot's current router itself.
    /// On this current-thread runtime nothing between the hand-over and the
    /// abort yields, so the watcher has not run. Aborting the router the slot
    /// started with instead would leave the handed-over one running beside the
    /// next repair's router, to store its command receiver back later.
    #[tokio::test]
    async fn abort_current_ends_the_router_handed_over_last_before_the_watcher_reads_it() {
        let supervisor = Arc::new(RouterSupervisor::new(Arc::new(Mutex::new(None))));
        let first = running_router();
        let first_task = first.abort_handle();
        supervisor.watch("txn", 0, first, idle_repair(), Some("slot-0".to_owned()));
        let handed_over = running_router();
        let handed_over_task = handed_over.abort_handle();
        supervisor.hand_over("slot-0", handed_over);

        let outcome = supervisor.abort_current("slot-0").await;

        assert_eq!(
            (
                outcome,
                handed_over_task.is_finished(),
                first_task.is_finished()
            ),
            (AbortCurrent::Ended, true, false),
            "the abort must end the router handed over last, and leave the one the slot started \
             with alone"
        );
        supervisor.shutdown().await;
        first_task.abort();
    }

    /// A router that ended before the repair reached it is neither aborted nor
    /// marked retired: `abort_current` reports `AlreadyEnded`, and the
    /// watcher handles the end as the router death it was, running the slot's
    /// restart once. Marked retired instead, the death would be read as a
    /// repair in progress and the slot left waiting for a hand-over no repair
    /// sends. The router has finished before `watch` is called, and nothing
    /// between `watch` and the abort yields, so on this current-thread runtime
    /// the abort runs before the watcher first polls.
    #[tokio::test]
    async fn abort_current_leaves_a_router_that_already_ended_to_its_watcher() {
        let supervisor = Arc::new(RouterSupervisor::new(Arc::new(Mutex::new(None))));
        let router: ConnectHandle = tokio::spawn(async {
            Err(fred::error::Error::new(
                fred::error::ErrorKind::IO,
                "the router ended on its own",
            ))
        });
        let router_task = router.abort_handle();
        for _ in 0..1_000 {
            if router_task.is_finished() {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert!(
            router_task.is_finished(),
            "precondition: the router has ended before the repair reaches it"
        );
        let repairs = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let restart = {
            let repairs = Arc::clone(&repairs);
            RouterRestart::Repair(Box::new(move || {
                repairs.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                async {}.boxed()
            }))
        };
        supervisor.watch("txn", 0, router, restart, Some("slot-0".to_owned()));

        let outcome = supervisor.abort_current("slot-0").await;

        assert_eq!(
            outcome,
            AbortCurrent::AlreadyEnded,
            "a router that had already ended must be reported as such, not aborted"
        );
        let deadline = Instant::now() + Duration::from_secs(10);
        while repairs.load(std::sync::atomic::Ordering::SeqCst) == 0 {
            assert!(
                Instant::now() < deadline,
                "the watcher must handle the already-ended router as a death and run the slot's \
                 restart; a router marked retired gets none"
            );
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        assert_eq!(
            repairs.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "the death is restarted exactly once"
        );
        supervisor.shutdown().await;
    }

    /// A router still mid-poll when the abort lands cannot end until that poll
    /// returns. A real fred router is always parked on its socket or its
    /// command channel, so this stand-in blocks its worker thread instead, until
    /// the test releases it. `abort_current` waits [`ROUTER_ABORT_TIMEOUT`] and
    /// reports `StillRunning` (the repair's WARN arm); the abort is still
    /// issued, so the router ends as soon as its poll returns.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn abort_current_reports_still_running_for_a_router_that_outlives_its_budget() {
        let supervisor = Arc::new(RouterSupervisor::new(Arc::new(Mutex::new(None))));
        let (polling_tx, polling) = tokio::sync::oneshot::channel::<()>();
        let (release, released) = std::sync::mpsc::channel::<()>();
        let router: ConnectHandle = tokio::spawn(async move {
            let _ = polling_tx.send(());
            // Blocks this worker thread inside the task's first poll.
            let _ = released.recv();
            std::future::pending().await
        });
        let router_task = router.abort_handle();
        supervisor.watch("txn", 0, router, idle_repair(), Some("slot-0".to_owned()));
        tokio::time::timeout(Duration::from_secs(10), polling)
            .await
            .expect("the router's first poll must start")
            .expect("the router task sends before it blocks");

        let outcome = supervisor.abort_current("slot-0").await;
        release
            .send(())
            .expect("the router is still blocked on the release");

        assert_eq!(
            outcome,
            AbortCurrent::StillRunning,
            "a router that has not ended within ROUTER_ABORT_TIMEOUT of its abort must be \
             reported as still running"
        );
        let deadline = Instant::now() + Duration::from_secs(10);
        while !router_task.is_finished() {
            assert!(
                Instant::now() < deadline,
                "the abort must have been issued: once its poll returns the router must end"
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        supervisor.shutdown().await;
    }

    // b.azh — retirable watches: a stream reader's client is supervised from
    // its first connect until the reader retires it (a retarget, an idle node
    // that stopped being a primary, shutdown), across every restart in
    // between. The end-to-end behaviour on real reader clients is in
    // `tests/integration/storage/stream_mux_reader_connect_test.rs`; these pin
    // the supervisor's half, which no real client can stage deterministically:
    // which router a retire hands back after a restart, and that nothing a
    // retired watch's router does afterwards is restarted.

    /// A restart closure that counts its calls and starts a router that never
    /// ends on its own, reporting each new router's abort handle.
    fn counted_restart() -> (
        RouterRestart,
        Arc<std::sync::atomic::AtomicUsize>,
        mpsc::UnboundedReceiver<AbortHandle>,
    ) {
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (started_tx, started) = mpsc::unbounded_channel::<AbortHandle>();
        let restart = {
            let calls = Arc::clone(&calls);
            RouterRestart::Connect(Box::new(move || {
                calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let router = running_router();
                let _ = started_tx.send(router.abort_handle());
                router
            }))
        };
        (restart, calls, started)
    }

    /// Polls (bounded) until `task` has ended.
    async fn await_finished(task: &AbortHandle, what: &str) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while !task.is_finished() {
            assert!(Instant::now() < deadline, "{what}");
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    }

    /// A retirable watch restarts its client's router when it dies, and the
    /// retire hands back the router running NOW — the restart, not the one
    /// the client connected with — for the reader to abort once its `QUIT`
    /// has returned, failed or overrun. Handing back the first router would
    /// leave the restarted one running beside a client the reader has
    /// discarded.
    #[tokio::test]
    async fn a_retirable_watch_restarts_a_dead_router_and_its_retire_hands_back_the_restart() {
        let supervisor = Arc::new(RouterSupervisor::new(Arc::new(Mutex::new(None))));
        let (restart, calls, mut started) = counted_restart();
        let first = running_router();
        let first_task = first.abort_handle();
        let watch = supervisor.watch_retirable("subscriber_pool", 0, "client-a", first, restart);
        assert_eq!(
            supervisor.retirable_watches_for_test("subscriber_pool"),
            HashMap::from([(0, "client-a".to_owned())]),
            "the watch is registered for its reader, under its client's id"
        );

        assert!(
            supervisor.end_router_for_test("subscriber_pool", 0),
            "the connected router is the reader's current one"
        );
        let restarted = tokio::time::timeout(Duration::from_secs(10), started.recv())
            .await
            .expect("the watcher must restart the dead router")
            .expect("the restart reports its router");
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
        // The watcher records the restart as current before it awaits it.
        for _ in 0..1_000 {
            tokio::task::yield_now().await;
        }

        let handed_back = supervisor.retire(watch).await;
        assert_eq!(
            (handed_back.id(), first_task.is_finished()),
            (restarted.id(), true),
            "the retire hands back the restarted router, the one the client runs now"
        );
        assert!(
            !restarted.is_finished(),
            "the retire itself aborts nothing: the reader aborts the handed-back router after \
             its QUIT"
        );
        assert!(
            supervisor
                .retirable_watches_for_test("subscriber_pool")
                .is_empty(),
            "a retired watch leaves the registry"
        );
        restarted.abort();
        supervisor.shutdown().await;
    }

    /// After a retire, the client's router may end any way at all — cleanly,
    /// as its `QUIT` ends it, or by the reader's abort after the `QUIT` — and
    /// nothing restarts it: the watcher has exited, and the supervisor no
    /// longer reaches the router. One watch per ending, each retired first.
    #[tokio::test]
    async fn a_retired_watchs_router_is_never_restarted_however_it_ends() {
        let supervisor = Arc::new(RouterSupervisor::new(Arc::new(Mutex::new(None))));

        // Ended cleanly, as a `QUIT` ends a router: `Ok(())`.
        let (quit_restart, quit_calls, _quit_started) = counted_restart();
        let (quit_tx, quit_rx) = tokio::sync::oneshot::channel::<()>();
        let quitting: ConnectHandle = tokio::spawn(async move {
            let _ = quit_rx.await;
            Ok(())
        });
        let watchers_before = supervisor.watchers();
        let watch =
            supervisor.watch_retirable("subscriber_pool", 2, "client-q", quitting, quit_restart);
        assert_eq!(supervisor.watchers(), watchers_before + 1);
        let current = supervisor.retire(watch).await;
        assert_eq!(
            supervisor.watchers(),
            watchers_before,
            "the retire waits for the watcher to exit"
        );
        quit_tx.send(()).expect("the router is still waiting");
        await_finished(&current, "the router ended by its QUIT must end").await;

        // Aborted, as the reader aborts the router after its `QUIT`.
        let (abort_restart, abort_calls, _abort_started) = counted_restart();
        let watch = supervisor.watch_retirable(
            "subscriber_pool",
            3,
            "client-b",
            running_router(),
            abort_restart,
        );
        let current = supervisor.retire(watch).await;
        assert_eq!(supervisor.watchers(), watchers_before);
        current.abort();
        await_finished(&current, "the aborted router must end").await;

        for _ in 0..1_000 {
            tokio::task::yield_now().await;
        }
        assert_eq!(
            (
                quit_calls.load(std::sync::atomic::Ordering::SeqCst),
                abort_calls.load(std::sync::atomic::Ordering::SeqCst)
            ),
            (0, 0),
            "a retired client's router is never restarted, whether its QUIT or an abort ended it"
        );
        assert!(
            !supervisor.end_router_for_test("subscriber_pool", 2)
                && !supervisor.end_router_for_test("subscriber_pool", 3),
            "the supervisor no longer reaches a retired router"
        );
        supervisor.shutdown().await;
    }

    /// A reader that retargets retires one watch and registers the next: the
    /// new client's dead router is restarted, the old client's is not, and the
    /// registry follows the reader to its new client.
    #[tokio::test]
    async fn after_a_replacement_only_the_new_clients_router_is_restarted() {
        let supervisor = Arc::new(RouterSupervisor::new(Arc::new(Mutex::new(None))));
        let (old_restart, old_calls, _old_started) = counted_restart();
        let (new_restart, new_calls, mut new_started) = counted_restart();
        let old = supervisor.watch_retirable(
            "subscriber_pool",
            0,
            "client-old",
            running_router(),
            old_restart,
        );
        let old_router = supervisor.retire(old).await;
        let new = supervisor.watch_retirable(
            "subscriber_pool",
            0,
            "client-new",
            running_router(),
            new_restart,
        );
        assert_eq!(
            supervisor.retirable_watches_for_test("subscriber_pool"),
            HashMap::from([(0, "client-new".to_owned())])
        );

        old_router.abort();
        assert!(supervisor.end_router_for_test("subscriber_pool", 0));
        let restarted = tokio::time::timeout(Duration::from_secs(10), new_started.recv())
            .await
            .expect("the new client's dead router must be restarted")
            .expect("the restart reports its router");
        for _ in 0..1_000 {
            tokio::task::yield_now().await;
        }
        assert_eq!(
            (
                old_calls.load(std::sync::atomic::Ordering::SeqCst),
                new_calls.load(std::sync::atomic::Ordering::SeqCst)
            ),
            (0, 1),
            "only the current client's router is restarted"
        );
        supervisor.retire(new).await.abort();
        restarted.abort();
        supervisor.shutdown().await;
    }

    /// A watch registered after shutdown — a reader whose connect completed
    /// while `quit()` was shutting the supervisor down — exits at once, on its
    /// own: its watcher ends before anything retires the watch, with its
    /// router dead and nothing restarted. A watch cancelled by a token of its
    /// own rather than the shut-down supervisor's would see that router end,
    /// restart it and keep watching, so the watcher count would never reach 0.
    /// The retire runs only afterwards, for cleanup.
    #[tokio::test]
    async fn a_retirable_watch_registered_after_shutdown_exits_at_once() {
        let supervisor = Arc::new(RouterSupervisor::new(Arc::new(Mutex::new(None))));
        supervisor.shutdown().await;
        let (restart, calls, _started) = counted_restart();
        let router = running_router();
        let router_task = router.abort_handle();
        let watch =
            supervisor.watch_retirable("subscriber_pool", 0, "client-late", router, restart);
        router_task.abort();
        await_finished(&router_task, "the late client's router must end").await;

        let deadline = Instant::now() + Duration::from_secs(10);
        while supervisor.watchers() > 0 {
            assert!(
                Instant::now() < deadline,
                "a watch registered after shutdown must exit by itself, before any retire; {} \
                 watcher(s) still running (restarts so far: {})",
                supervisor.watchers(),
                calls.load(std::sync::atomic::Ordering::SeqCst)
            );
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "the watcher exited without restarting the router that ended under it"
        );

        supervisor.retire(watch).await;
    }

    /// How long the T-9 restart waits, inside its `connect()`, for an
    /// `abandon` racing it on another thread to return.
    const ABANDON_RACE_WAIT: Duration = Duration::from_secs(1);

    /// b.zt3 T-9 (I-9): an `abandon` that races a retirable watch's restart
    /// returns the router that restart started, and that router is aborted —
    /// so `pool::retire`'s release, which waits on the handle `abandon`
    /// returns, follows every router the client had.
    ///
    /// The race is staged inside the restart itself, with no seam: the
    /// test's `RouterRestart::Connect` closure runs between the watcher's
    /// cancellation check and the store of the new handle. It calls `abandon`
    /// on a std thread and waits up to [`ABANDON_RACE_WAIT`] for it to return
    /// before starting its router. With the restart under the `current` lock
    /// (Step 2.2) the `abandon` blocks on that lock until the new handle is
    /// stored, then reads and aborts it. Without it, `abandon` returns at once
    /// with the stale handle of the router that already ended, and the router
    /// the restart then starts keeps running.
    ///
    /// Fail-on-old, by in-place perturbation: restarting a retirable watch
    /// with a bare `SupervisedRouter::new(connect())` (Step 2.2's lock and
    /// cancellation check gone) together with removing Step 2.1's abort on the
    /// watcher's cancel arm turns the `returned.id() == restarted.id()`
    /// assertion red: `abandon` returns the stale router's handle. With 2.2 in
    /// place, removing 2.1 alone leaves this test green — the router 2.1
    /// aborts is the one `abandon` has already aborted — and no test can fail
    /// on it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_abandon_racing_a_restart_returns_and_aborts_the_router_the_restart_started() {
        let supervisor = Arc::new(RouterSupervisor::new(Arc::new(Mutex::new(None))));
        let abandon: Arc<std::sync::OnceLock<AbandonHandle>> = Arc::new(std::sync::OnceLock::new());
        let (abandoned_tx, abandoned_rx) = std::sync::mpsc::channel::<AbortHandle>();
        let (started_tx, mut started) = mpsc::unbounded_channel::<AbortHandle>();
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let restart = {
            let abandon = Arc::clone(&abandon);
            let calls = Arc::clone(&calls);
            let abandoned_tx = Mutex::new(abandoned_tx);
            RouterRestart::Connect(Box::new(move || {
                calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let handle = abandon
                    .get()
                    .expect("the abandon handle is stashed before the first router ends")
                    .clone();
                let to_test = abandoned_tx
                    .lock()
                    .expect("the sender lock is not poisoned")
                    .clone();
                let (returned_tx, returned) = std::sync::mpsc::channel::<()>();
                std::thread::spawn(move || {
                    let router = handle.abandon();
                    let _ = returned_tx.send(());
                    let _ = to_test.send(router);
                });
                // Without the lock, `abandon` returns well inside the wait;
                // with it, it cannot return until this closure has.
                let _ = returned.recv_timeout(ABANDON_RACE_WAIT);
                let router = running_router();
                let _ = started_tx.send(router.abort_handle());
                router
            }))
        };
        let first = running_router();
        let first_task = first.abort_handle();
        let watchers = supervisor.watchers();
        let watch = supervisor.watch_retirable("data", 0, "client-a", first, restart);
        assert!(
            abandon.set(watch.abandon_handle()).is_ok(),
            "the handle is stashed once"
        );

        assert!(
            supervisor.end_router_for_test("data", 0),
            "the first router is the client's current one"
        );
        let restarted = tokio::time::timeout(Duration::from_secs(10), started.recv())
            .await
            .expect("the watcher must restart the ended router")
            .expect("the restart reports its router");
        let returned =
            tokio::task::spawn_blocking(move || abandoned_rx.recv_timeout(Duration::from_secs(10)))
                .await
                .expect("the receiving thread does not panic")
                .expect("the racing abandon must return");

        assert_eq!(
            returned.id(),
            restarted.id(),
            "abandon must return the router the racing restart started — the client's last \
             router — not the stale one that had already ended ({:?})",
            first_task.id()
        );
        await_finished(
            &restarted,
            "the router the racing restart started must be aborted",
        )
        .await;
        let deadline = Instant::now() + Duration::from_secs(10);
        while supervisor.watchers() > watchers {
            assert!(
                Instant::now() < deadline,
                "the abandoned watch's watcher must exit"
            );
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "nothing restarts after the abandon"
        );
        supervisor.retire(watch).await;
        supervisor.shutdown().await;
    }
}
