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

//! Replacing a pool member that keeps failing with a fresh fred client.
//!
//! A pooled fred client can keep failing after its connection is rebuilt: a
//! fast unresponsive verdict closes the silent socket and fred reconnects it,
//! and the fresh socket fails again. Whatever per-client state does it, a
//! reconnect keeps it. So a member of the `data`, `maintenance` or `snapshot`
//! pool, or a `txn` slot, whose client reaches its second **fast** verdict
//! ([`MEMBER_REPLACE_AFTER_FAST_VERDICTS`]) within [`MEMBER_REPLACE_WINDOW`] is
//! replaced: a fresh client, built from the same configuration with nothing
//! carried over, takes the member over, and the old one is retired.
//!
//! **The trigger.** Only fast verdicts count (`fred::live_edit::liveness::
//! verdict_of`): one connection silent while its shard answered others. A slow
//! verdict or an expired bounded wait is a shard silent for everyone, which a
//! new client cannot cure, and counting it would replace every member of a pool
//! at once during an outage.
//!
//! **One at a time per pool, then the next at once.** A replacement holds its
//! pool's slot from the trigger through the swap (step 5 below; for a `txn`
//! slot, through the wait for the slot and the swap), so a pool connects one
//! fresh client at a time. A member that comes due while the slot is held is
//! marked pending and started as soon as the swap is made, lowest member index
//! first, without waiting for another verdict, so the k-th of k members due at
//! once is replaced within k connect times (plus each `txn` slot's wait). The
//! old client's retirement (step 6) runs after the swap, in the same task,
//! without the slot; several can be in progress, at most
//! [`MEMBER_RETIRING_CAP`] per replica, of any pool, a swap past it ending
//! the oldest other one early.
//! After a member is replaced it is not replaced again for
//! [`MEMBER_REPLACE_COOLDOWN`]; a fast verdict on the fresh client inside the
//! cooldown logs a WARN, "member N's replacement did not cure it", naming both
//! client ids: the cause is then not per-client fred state.
//!
//! **The flow, across files.**
//!
//! 1. `ValkeyClient::connect` (`storage::client`) builds one [`MemberReplacer`]
//!    over the four pools; `start_router_supervision` watches each multiplexed
//!    member as a retirable watch (`storage::router_supervisor`) and hands the
//!    watch to the replacer ([`MemberReplacer::adopt_watch`]), and watches each
//!    `txn` slot under its client id, with the slot's router restart, which it
//!    also hands to the replacer for the slots it will swap
//!    ([`MemberReplacer::set_txn_restart`]).
//! 2. `install_observability_hooks` arms each member's reconnect and error
//!    hooks through the per-connection hook registry (`PoolHooks`), keyed by
//!    client id, with a [`VerdictSink`] naming the member — for a `txn` slot
//!    through `ValkeyClient::rearm_pool_hooks`, which every slot recycle also
//!    uses ([`MemberReplacer::arm_txn`]). The error hook
//!    (`spawn_error_metric`) classifies each connection error, ticks
//!    `live_edit_valkey_unresponsive_total{pool,verdict}`, and hands a fast
//!    verdict to the sink ([`MemberReplacer::on_fast_verdict`]).
//! 3. On the trigger, one task per replacement:
//!    1. builds the fresh client (`Client::clone_new`: the same config,
//!       credential provider and peer-reply registry, a new id);
//!    2. connects it, bounded by the connect timeout — on failure its router is
//!       aborted, the member keeps its client, outcome `connect_failed`;
//!    3. arms its hooks under its own id, after the connect, so the first
//!       connect is not counted as a reconnect;
//!    4. watches its router (`RouterSupervisor::watch_retirable`);
//!    5. points the member at it (`fred::live_edit::pool::replace_member`):
//!       new commands go to it from here on, outcome `replaced`;
//!    6. retires the old client (for a `txn` slot, see below), as a stream
//!       reader retires a connection:
//!       [`ReplaceKnobs::grace`] runs from the swap, so the commands already
//!       handed to it finish or time out, and inside it the old client's
//!       watch is retired (so its end is never counted as a router restart);
//!       when the grace ends, its hooks are disarmed and it
//!       is sent `QUIT` whatever its state reads, which ends its router
//!       through fred's own path: owed replies drained, commands queued
//!       behind the `QUIT` failed with `Canceled` (so a `SCAN` stream on it
//!       ends in an error, never a silent end of stream). Only if that has not
//!       ended the router within [`MEMBER_RETIRE_QUIT_BOUND`] is the router
//!       aborted. Then, once that router has finished, what fred would leave
//!       on a client its pool keeps is released
//!       (`fred::live_edit::pool::retire`): the command channel is closed
//!       and anything still queued in it cancelled, so a later send through a
//!       `&Client` borrowed before the replacement — a background walk asking
//!       for its next `SCAN` page — fails at once instead of waiting forever
//!       (with an `Unknown` error from fred's closed-channel path, which the
//!       service treats as permanent; those walks retry on their next tick);
//!       the credential-refresh task is stopped; and the backchannel
//!       connection is closed — on a detached task bounded at about 6 s
//!       (`RETIRE_ROUTER_WAIT` plus `RETIRE_BACKCHANNEL_CLOSE_TIMEOUT`), with
//!       one rare residual: a close that times out before it gets the
//!       backchannel's lock leaves that socket open until the client is
//!       dropped (see `fred::live_edit::pool::retire` and
//!       [`MemberReplacer::quit_and_release`]). Nothing in flight on it moves
//!       to the fresh client, so a replacement never re-sends a command. The
//!       retirement is bounded by the grace plus [`MEMBER_RETIRE_QUIT_BOUND`]:
//!       about 56 s at the defaults. The watch's retire (at most
//!       `ROUTER_ABORT_TIMEOUT`, 0.5 s) runs inside the grace, which is never
//!       shorter than 1 s in production; only a test grace below 0.5 s lets
//!       it add to the bound. It does not hold the pool's slot, so it never
//!       delays the next replacement; a swap that would make more than
//!       [`MEMBER_RETIRING_CAP`] retirements in progress across the replica
//!       ends the oldest other one early, whatever its pool, as shutdown does.
//! 4. The readers' routing view (`ReaderConnector`) resolves data member 0 on
//!    every call, so it follows a replacement of that member.
//!
//! **A `txn` slot** is swapped under its own guard instead of steps 4–5. The
//! replacement takes the slot through the checkout like any caller
//! (`ExclusiveCheckout::checkout_slot_for_replacement`: a permit, then that
//! slot's mutex), so it waits until no caller holds the slot — never mid-
//! transaction, never past a `WatchGuard` still holding it for its `UNWATCH`.
//! Holding the guard it writes the fresh client into the slot
//! (`ExclusiveCheckout::swap_slot_client`, which re-keys the slot's id), ends
//! the old client's slot watch (`RouterSupervisor::retire_slot`: forgotten, its
//! router marked retired so its end counts no restart) and watches the fresh
//! client's router under the fresh id with the slot's router restart. The
//! hook pair was re-keyed at step 3 and the old one is disarmed in step 6.
//! Then it releases the slot and permit, and the next checkout of that slot
//! gets the fresh client, through the same `acquire_txn_for` probe and
//! `WatchGuard` hygiene as any. The permit count, the next-free scan and
//! fred's `ExclusivePool` are untouched; fred's `ExclusivePool::acquire()` is
//! never called. Its retirement deliberately skips [`ReplaceKnobs::grace`]:
//! the grace exists for commands already handed to a multiplexed member's old
//! client, and a slot's old client has none — the swap waited until no caller
//! held the slot, and no caller can reach a client that has left its slot
//! (every caller reaches a slot's client only through a checkout). So it goes
//! straight to disarming its hooks and `QUIT`; the rest of step 6 is the same.
//! fred keeps no husk of it, but the `txn` pool has a budget of its own, of the
//! same size: a count of `max(size, 4)` replacements, so a client that keeps
//! failing cannot churn connections without bound.
//! 5. `ValkeyClient::quit` shuts the replacer down first, before the router
//!    supervisor ([`MemberReplacer::shutdown`]): every replacement task stops
//!    at its next await, all of them at once. A replacement not yet active
//!    aborts the fresh client's router and disarms its hooks; every retirement
//!    in progress aborts its old client's router at once. Either way, once
//!    that router has finished, the discarded client is released as in step 6
//!    (command channel closed, credential-refresh task stopped, backchannel
//!    closed within its bound, with the same rare residual). A drop guard does
//!    the same when a replacement task is aborted or panics, and for the
//!    retirement the per-replica cap evicts. When the client's last owner
//!    drops without `quit`, the same cancellation runs without the wait
//!    ([`MemberReplacer::release`], from `ValkeyClient`'s `ClientOwners`).
//!
//! **The budget.** fred keeps every retired multiplexed client in its pool's
//! spare cells so that a `&Client` borrowed before the replacement stays valid;
//! a pool has `max(size, 4)` of them (`fred::live_edit::pool`). The `txn` pool
//! has no spare cells (a slot's old client is swapped out, not kept); its
//! budget is a count of `max(size, 4)` replacements, kept here. When a pool's
//! budget is used up a due replacement is refused (outcome `budget_exhausted`,
//! one WARN per pool) and the member keeps relying on fast verdicts; a pod
//! restart resets it.
//!
//! Metric: `live_edit_valkey_member_replacements_total{pool,outcome}`.
//! Constants, not configuration (no environment variable).

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;

use fred::clients::{Client, Pool};

use crate::storage::checkout::ExclusiveCheckout;
use fred::interfaces::ClientLike;
use tokio::task::AbortHandle;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;

use crate::observability::metrics::{
    VALKEY_MEMBER_REPLACEMENT_BUDGET_EXHAUSTED, VALKEY_MEMBER_REPLACEMENT_CONNECT_FAILED,
    VALKEY_MEMBER_REPLACEMENT_REPLACED,
};
use crate::storage::client::{
    PoolHooks, POOL_LABEL_DATA, POOL_LABEL_MAINTENANCE, POOL_LABEL_SNAPSHOT, POOL_LABEL_TXN,
    UNRESPONSIVE_MAX_TIMEOUT,
};
use crate::storage::fanout_thread::connect_router_on;
use crate::storage::router_supervisor::{
    AbandonHandle, RetirableWatch, RouterRestart, RouterSupervisor,
};

/// How many fast unresponsive verdicts on one member's client, within
/// [`MEMBER_REPLACE_WINDOW`], replace it. One fast verdict is already cured by
/// fred's reconnect; a second means the fresh socket failed too. Under the fast
/// verdict, a client that stays bad also stalls the command re-sent after its
/// first verdict, so the second usually follows within seconds.
pub const MEMBER_REPLACE_AFTER_FAST_VERDICTS: usize = 2;

/// The window [`MEMBER_REPLACE_AFTER_FAST_VERDICTS`] counts over. In the field
/// a bad client failed again after its reconnect within one second to an hour;
/// an hour also catches an hourly pattern.
pub const MEMBER_REPLACE_WINDOW: Duration = Duration::from_secs(60 * 60);

/// How long after a member is replaced it is not replaced again. A fast
/// verdict on the fresh client inside it logs the "did not cure" WARN.
pub const MEMBER_REPLACE_COOLDOWN: Duration = Duration::from_secs(10 * 60);

/// How long [`MemberReplacer::shutdown`] waits for replacement tasks to exit.
/// They exit at their next await once cancelled. One of `ValkeyClient::quit`'s
/// sequential bounds, summed against the shutdown-budget headroom by a `const`
/// assertion in `storage::client`.
pub(crate) const MEMBER_REPLACER_DRAIN_TIMEOUT: Duration = Duration::from_secs(2);

/// How long a retired client's `QUIT`, and the end of its router after it, may
/// take before the router is aborted: the slow unresponsive bound, which bounds
/// the drain `QUIT` runs first on each of its connections, plus a margin.
/// Nothing waits on a retirement — it runs after its swap has freed the pool's
/// slot, and shutdown cuts it short — so the bound only has to be finite.
pub const MEMBER_RETIRE_QUIT_BOUND: Duration =
    Duration::from_secs(UNRESPONSIVE_MAX_TIMEOUT.as_secs() + 5);

/// How often a retirement checks whether the retired client's router ended
/// after its `QUIT`.
const ROUTER_END_POLL: Duration = Duration::from_millis(20);

/// The most old clients one replacer (one replica) keeps retiring at once,
/// across all its pools: the bound on old clients a replica holds above its
/// configured pool sizes while they retire. A swap that would make it five ends
/// the **oldest** retirement other than its own, whatever its pool, early
/// through its last-resort path — its router aborted, its
/// hooks disarmed, its client released (`fred::live_edit::pool::retire`) — as
/// shutdown would, and logs one WARN naming that client and its pool. Commands
/// still in flight on that client then fail as fred fails a command its
/// aborted router dropped (`Unknown`, "channel closed"), which every caller
/// answers as an unknown outcome (`INTERNAL`), never as a refusal that applied
/// nothing: only a store write refusal's text
/// (`storage::client::store_write_refusal`) earns that hint. The fresh
/// client's connect is never delayed by the cap: the check runs at the swap,
/// and the swap's own new retirement is never the one ended.
///
/// A `maintenance` retirement is ended like any other. What the eviction kills
/// there costs at most a delayed pass, a leak the reference-count reconciler
/// drains, or a misleading `ERROR` line: the GC decrement is at-most-once in
/// its script, so a decrement killed in flight is re-driven from `inflight`
/// and answers `replayed` if it had run (`storage::external_values`), and every
/// other command on that pool keeps fred's retries and so already tolerates a
/// second run. Shutdown still cuts every retirement.
pub const MEMBER_RETIRING_CAP: usize = 4;

/// How long a retired client of a `data`, `maintenance` or `snapshot` member is
/// left, counted from its swap, to finish the commands already handed to it
/// before its hooks are disarmed and it is sent `QUIT` (a `txn` slot's old
/// client gets no grace: no caller can reach it once it has left its slot):
/// the command timeout plus one second, or — when the command timeout is zero
/// (disabled) — the slow unresponsive bound plus one second, the longest a
/// command on it can then wait before fred fails its connection.
pub fn member_retire_grace(command_timeout: Duration) -> Duration {
    if command_timeout.is_zero() {
        UNRESPONSIVE_MAX_TIMEOUT + Duration::from_secs(1)
    } else {
        command_timeout + Duration::from_secs(1)
    }
}

/// The replacement trigger, cooldown and retirement grace. The production
/// values are the constants above and [`member_retire_grace`]; tests shorten
/// them (`ValkeyClient::set_member_replacement_knobs_for_test`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReplaceKnobs {
    /// [`MEMBER_REPLACE_AFTER_FAST_VERDICTS`].
    pub after_fast_verdicts: usize,
    /// [`MEMBER_REPLACE_WINDOW`].
    pub window: Duration,
    /// [`MEMBER_REPLACE_COOLDOWN`].
    pub cooldown: Duration,
    /// [`member_retire_grace`].
    pub grace: Duration,
}

impl ReplaceKnobs {
    /// The production knobs for a client whose command timeout is
    /// `command_timeout`.
    pub fn production(command_timeout: Duration) -> Self {
        Self {
            after_fast_verdicts: MEMBER_REPLACE_AFTER_FAST_VERDICTS,
            window: MEMBER_REPLACE_WINDOW,
            cooldown: MEMBER_REPLACE_COOLDOWN,
            grace: member_retire_grace(command_timeout),
        }
    }
}

/// Where a member's error hook hands its client's fast verdicts: the replacer
/// (held weakly, so a hook task never keeps it alive), the member, and the
/// client the hook is armed on.
#[derive(Clone)]
pub(crate) struct VerdictSink {
    replacer: Weak<MemberReplacer>,
    pool: &'static str,
    member: usize,
    client_id: String,
}

impl VerdictSink {
    /// A fast verdict closed a connection of the client this sink names.
    pub(crate) fn fast_verdict(&self) {
        if let Some(replacer) = self.replacer.upgrade() {
            replacer.on_fast_verdict(self.pool, self.member, &self.client_id);
        }
    }
}

/// How a `txn` slot's router is restarted when it ends, for a client id:
/// `ValkeyClient`'s slot repair under the checkout, which needs the
/// `ValkeyClient` itself.
pub(crate) type TxnRestart = Arc<dyn Fn(String) -> RouterRestart + Send + Sync>;

/// One replaceable pool.
struct ReplaceablePool {
    label: &'static str,
    kind: PoolKind,
    state: Mutex<PoolReplaceState>,
}

/// How a pool's members are reached, and so how one is replaced.
enum PoolKind {
    /// A multiplexed fred `Pool`: a member is re-pointed in fred's arena.
    Multiplexed(Pool),
    /// The `txn` pool, reached only through its checkout: a slot's client is
    /// swapped under the slot's guard.
    Txn(Arc<ExclusiveCheckout>),
}

#[derive(Default)]
struct PoolReplaceState {
    /// The pool's slot is held: one replacement of one of its members is
    /// between its trigger and its swap (the connect, and for a `txn` slot the
    /// wait for the slot). Released at the swap, or when the attempt ends,
    /// and only by the replacement that holds it ([`Replacement::release_slot`]);
    /// the old client's retirement runs after it, without the slot.
    in_flight: bool,
    /// The budget-exhausted WARN has been logged for this pool.
    budget_warned: bool,
    /// For the `txn` pool: replacements left in its budget (`max(size, 4)`).
    /// A multiplexed pool's budget is its spare cells, read from fred.
    txn_replacements_left: usize,
    members: Vec<MemberSlot>,
}

struct MemberSlot {
    /// The fred client id of the member's current client.
    client_id: String,
    /// For a `txn` slot: a clone of its current client (a clone shares the
    /// client, and holding it locks nothing), the template of the fresh one.
    /// A multiplexed member's is read from its pool.
    client: Option<Client>,
    /// The supervision of the member's current client; taken by the
    /// replacement that retires it.
    watch: Option<RetirableWatch>,
    /// The fast verdicts on the current client within the window, oldest first.
    fast_verdicts: VecDeque<Instant>,
    /// The member's latest replacement, for the cooldown.
    replaced: Option<Replaced>,
    /// The member came due while another replacement held the pool's slot.
    /// Set by [`MemberReplacer::on_fast_verdict`] in place of starting it;
    /// cleared when its replacement starts or is refused for budget, when the
    /// re-check at the slot's release finds it no longer due
    /// ([`PoolReplaceState::take_next_due`]), and at a swap that replaces it (a
    /// verdict on its old client while that replacement held the slot marked
    /// it). Dropped with the replacer. A member in its cooldown is never
    /// marked: its verdict returns before the threshold is checked.
    pending: bool,
}

impl MemberSlot {
    /// Drop the fast verdicts older than `window` before `now`.
    fn prune_verdicts(&mut self, now: Instant, window: Duration) {
        while self
            .fast_verdicts
            .front()
            .is_some_and(|at| now.saturating_duration_since(*at) > window)
        {
            self.fast_verdicts.pop_front();
        }
    }
}

impl PoolReplaceState {
    /// The re-check when the pool's slot frees: the lowest-index member marked
    /// `pending` whose fast verdicts, pruned to `window` before `now`, still
    /// reach `after_fast_verdicts`, with its mark cleared — or `None`. Each
    /// pending member it passes over on the way (no longer due: its verdicts
    /// aged out) has its mark cleared too; members after the one it returns
    /// keep theirs, for the next release.
    fn take_next_due(
        &mut self,
        now: Instant,
        window: Duration,
        after_fast_verdicts: usize,
    ) -> Option<usize> {
        for (member, slot) in self.members.iter_mut().enumerate() {
            if !slot.pending {
                continue;
            }
            slot.pending = false;
            slot.prune_verdicts(now, window);
            if slot.fast_verdicts.len() >= after_fast_verdicts {
                return Some(member);
            }
        }
        None
    }
}

struct Replaced {
    at: Instant,
    previous_id: String,
}

/// Replaces failing members of the `data`, `maintenance` and `snapshot` pools,
/// and failing `txn` slots' clients; see the module docs. One per
/// `ValkeyClient`, shared by its clones.
pub(crate) struct MemberReplacer {
    pools: Vec<ReplaceablePool>,
    /// The `txn` slots' router restart ([`Self::set_txn_restart`]).
    ///
    /// The closure holds an internal clone of the `ValkeyClient` that built
    /// it (`ValkeyClient::internal_clone`), and that client holds this
    /// replacer through an `Arc`: a reference cycle, which an internal clone
    /// cannot keep alive past the client's owners. [`Self::release`] takes it
    /// out — at `ValkeyClient::quit`, through [`Self::shutdown`], or when the
    /// client's last owner drops without `quit` — and so frees the client.
    txn_restart: Mutex<Option<TxnRestart>>,
    supervisor: Arc<RouterSupervisor>,
    hooks: PoolHooks,
    connect_timeout: Duration,
    knobs: Mutex<ReplaceKnobs>,
    /// The replacer's shutdown: cancelled by [`Self::shutdown`], and the parent
    /// of every retirement's own stop token.
    cancel: CancellationToken,
    /// Every replacement task, from its trigger through its old client's
    /// retirement; drained by [`Self::shutdown`].
    tracker: TaskTracker,
    /// The retirements in progress, oldest first, capped at
    /// [`MEMBER_RETIRING_CAP`].
    retiring: Mutex<Retiring>,
    /// The `snapshot-io` runtime a `snapshot` member's fresh client's router
    /// is started and restarted on (`storage::fanout_thread`); `None` only
    /// for a replacer built without a client (unit tests), whose routers
    /// start on the calling runtime.
    snapshot_io: Option<tokio::runtime::Handle>,
}

/// The retirements in progress across a replacer's pools, oldest first.
#[derive(Default)]
struct Retiring {
    /// The id the next retirement gets.
    next_id: u64,
    entries: VecDeque<RetiringEntry>,
}

/// One old client's retirement in progress: from the swap that retired it to
/// the end of its `QUIT` (or its cut-short cleanup), when its guard drops.
struct RetiringEntry {
    id: u64,
    pool: &'static str,
    member: usize,
    client_id: String,
    /// The retirement's own stop: a child of the replacer's shutdown token,
    /// cancelled alone when [`MEMBER_RETIRING_CAP`] evicts it. Either way the
    /// retirement returns at its next await and its guard cuts it short.
    stop: CancellationToken,
}

impl Retiring {
    /// While more than [`MEMBER_RETIRING_CAP`] retirements are in progress,
    /// end the oldest one other than `spare` (the swap's own new retirement)
    /// — cancelling its stop token and logging one WARN. Above a cap of at
    /// least 1 there is always such a one, so the loop ends at the cap.
    fn evict_over_cap(&mut self, spare: u64) {
        while self.entries.len() > MEMBER_RETIRING_CAP {
            let Some(index) = self.entries.iter().position(|entry| entry.id != spare) else {
                break;
            };
            let Some(evicted) = self.entries.remove(index) else {
                break;
            };
            tracing::warn!(
                pool = evicted.pool,
                member = evicted.member,
                client_id = %evicted.client_id,
                retiring_cap = MEMBER_RETIRING_CAP,
                "ending a retired Valkey client's retirement early: more old clients were \
                 retiring than the per-replica cap allows, so the oldest one has its router \
                 aborted now instead of finishing its retirement; commands still in flight on \
                 it fail"
            );
            evicted.stop.cancel();
        }
    }
}

impl std::fmt::Debug for MemberReplacer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemberReplacer")
            .field("tasks", &self.tracker.len())
            .field("retiring", &self.retiring_count())
            .field("shut_down", &self.cancel.is_cancelled())
            .finish()
    }
}

impl MemberReplacer {
    /// A replacer over the three multiplexed pools and the `txn` pool, whose
    /// slots hold `txn_clients` (clones, in slot order), every member on the
    /// client it was built with and unwatched until [`Self::adopt_watch`].
    /// A `snapshot` member's fresh router runs on `snapshot_io`
    /// ([`Self::snapshot_io`]).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        data: Pool,
        maintenance: Pool,
        snapshot: Pool,
        txn: Arc<ExclusiveCheckout>,
        txn_clients: Vec<Client>,
        supervisor: Arc<RouterSupervisor>,
        hooks: PoolHooks,
        connect_timeout: Duration,
        command_timeout: Duration,
        snapshot_io: Option<tokio::runtime::Handle>,
    ) -> Arc<Self> {
        let slot = |client: &Client, keep: bool| MemberSlot {
            client_id: ClientLike::id(client).to_owned(),
            client: keep.then(|| client.clone()),
            watch: None,
            fast_verdicts: VecDeque::new(),
            replaced: None,
            pending: false,
        };
        let mut pools: Vec<ReplaceablePool> = [
            (POOL_LABEL_DATA, data),
            (POOL_LABEL_MAINTENANCE, maintenance),
            (POOL_LABEL_SNAPSHOT, snapshot),
        ]
        .into_iter()
        .map(|(label, pool)| {
            let members = pool
                .clients()
                .iter()
                .map(|client| slot(client, false))
                .collect();
            ReplaceablePool {
                label,
                kind: PoolKind::Multiplexed(pool),
                state: Mutex::new(PoolReplaceState {
                    members,
                    ..PoolReplaceState::default()
                }),
            }
        })
        .collect();
        pools.push(ReplaceablePool {
            label: POOL_LABEL_TXN,
            state: Mutex::new(PoolReplaceState {
                txn_replacements_left: txn_clients
                    .len()
                    .max(fred::live_edit::pool::MIN_SPARE_CELLS),
                members: txn_clients
                    .iter()
                    .map(|client| slot(client, true))
                    .collect(),
                ..PoolReplaceState::default()
            }),
            kind: PoolKind::Txn(txn),
        });
        Arc::new(Self {
            pools,
            txn_restart: Mutex::new(None),
            supervisor,
            hooks,
            connect_timeout,
            knobs: Mutex::new(ReplaceKnobs::production(command_timeout)),
            cancel: CancellationToken::new(),
            tracker: TaskTracker::new(),
            retiring: Mutex::new(Retiring::default()),
            snapshot_io,
        })
    }

    fn pool(&self, label: &str) -> Option<&ReplaceablePool> {
        self.pools.iter().find(|pool| pool.label == label)
    }

    fn knobs(&self) -> ReplaceKnobs {
        *self
            .knobs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Replace the trigger, cooldown and grace. **Test-only.**
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) fn set_knobs_for_test(&self, knobs: ReplaceKnobs) {
        *self
            .knobs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = knobs;
    }

    /// Hand over the startup watch of member `member` of pool `label`
    /// (`ValkeyClient::start_router_supervision`), for the replacement that
    /// retires the client to retire it.
    pub(crate) fn adopt_watch(&self, label: &'static str, member: usize, watch: RetirableWatch) {
        let Some(pool) = self.pool(label) else {
            return;
        };
        let mut state = pool
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(slot) = state.members.get_mut(member) {
            slot.watch = Some(watch);
        }
    }

    /// Arm the reconnect and error hooks on every member's current client,
    /// keyed by its id, with a [`VerdictSink`] naming the member. Called by
    /// `install_observability_hooks` once the metrics exist; a no-op before
    /// that, as every `PoolHooks` arm is.
    ///
    /// The `data`, `maintenance` and `snapshot` members only: the `txn` slots
    /// are armed through [`Self::arm_txn`].
    pub(crate) fn arm_hooks(self: &Arc<Self>) {
        for pool in &self.pools {
            let PoolKind::Multiplexed(members) = &pool.kind else {
                continue;
            };
            for (member, client) in members.clients().iter().enumerate() {
                self.arm(pool.label, member, client);
            }
        }
    }

    /// Arm the reconnect and error hooks on `client`, a `txn` slot's client,
    /// with a [`VerdictSink`] naming its slot, aborting the pair armed on it
    /// before. The one installer of a `txn` slot's hooks
    /// (`ValkeyClient::rearm_pool_hooks`: the startup install and every slot
    /// recycle). A client no slot holds is armed without a sink.
    pub(crate) fn arm_txn(self: &Arc<Self>, client: &Client) {
        let member = self.pool(POOL_LABEL_TXN).and_then(|pool| match &pool.kind {
            PoolKind::Txn(txn) => txn.slot_of(ClientLike::id(client)),
            PoolKind::Multiplexed(_) => None,
        });
        match member {
            Some(member) => self.arm(POOL_LABEL_TXN, member, client),
            None => self.hooks.rearm_with(client, POOL_LABEL_TXN, None),
        }
    }

    /// Hand over how a `txn` slot's router is restarted
    /// (`ValkeyClient::start_router_supervision`), for the watch a slot
    /// replacement starts under the fresh client's id. `restart` holds an
    /// internal clone of that `ValkeyClient`, which holds this replacer: a
    /// reference cycle until [`Self::release`] (see the `txn_restart` field).
    pub(crate) fn set_txn_restart(&self, restart: TxnRestart) {
        *self
            .txn_restart
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(restart);
    }

    /// Count a fast verdict on member `member` of pool `label`'s current
    /// client, as its error hook would. **Test-only.**
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) fn fast_verdict_for_test(self: &Arc<Self>, label: &str, member: usize) {
        let Some(pool) = self.pool(label) else {
            return;
        };
        let client_id = pool
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .members
            .get(member)
            .map(|slot| slot.client_id.clone());
        if let Some(client_id) = client_id {
            self.on_fast_verdict(pool.label, member, &client_id);
        }
    }

    fn arm(self: &Arc<Self>, label: &'static str, member: usize, client: &Client) {
        let sink = VerdictSink {
            replacer: Arc::downgrade(self),
            pool: label,
            member,
            client_id: ClientLike::id(client).to_owned(),
        };
        self.hooks.rearm_with(client, label, Some(sink));
    }

    /// A fast unresponsive verdict closed a connection of `client_id`, member
    /// `member` of pool `label`: count it, and start a replacement when it is
    /// due (see the module docs).
    pub(crate) fn on_fast_verdict(
        self: &Arc<Self>,
        label: &'static str,
        member: usize,
        client_id: &str,
    ) {
        let Some(pool) = self.pool(label) else {
            return;
        };
        let knobs = self.knobs();
        let now = Instant::now();
        let mut state = pool
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if self.cancel.is_cancelled() {
            return;
        }
        let in_flight = state.in_flight;
        let Some(slot) = state.members.get_mut(member) else {
            return;
        };
        // A retired client's late verdict: the member has moved on.
        if slot.client_id != client_id {
            return;
        }
        slot.fast_verdicts.push_back(now);
        slot.prune_verdicts(now, knobs.window);
        if let Some(replaced) = slot
            .replaced
            .as_ref()
            .filter(|replaced| now.saturating_duration_since(replaced.at) < knobs.cooldown)
        {
            tracing::warn!(
                pool = label,
                member,
                previous_client_id = %replaced.previous_id,
                client_id,
                replaced_secs_ago = now.saturating_duration_since(replaced.at).as_secs(),
                "member {member}'s replacement did not cure it: the fresh client had a fast \
                 unresponsive verdict too, so the cause is not per-client fred state"
            );
            return;
        }
        let verdicts = slot.fast_verdicts.len();
        if verdicts < knobs.after_fast_verdicts {
            return;
        }
        if in_flight {
            // Another replacement holds the pool's slot: remember this one, so
            // the slot's release starts it without waiting for another verdict.
            slot.pending = true;
            return;
        }
        self.try_start(pool, &mut state, member, verdicts);
    }

    /// Start the replacement of member `member` of `pool`, due on `verdicts`
    /// fast verdicts, under the pool's lock (`state`) with the slot free: take
    /// the slot (`in_flight`) and spawn the replacement task — or, when the
    /// pool's budget is used up, refuse it (outcome `budget_exhausted`, counted
    /// once per refusal; the WARN once per pool). Either way the member's
    /// `pending` mark is cleared. The one place a replacement starts: the
    /// trigger ([`Self::on_fast_verdict`]) and the re-check when the slot frees
    /// ([`Self::release_slot`]) both come here.
    fn try_start(
        self: &Arc<Self>,
        pool: &ReplaceablePool,
        state: &mut PoolReplaceState,
        member: usize,
        verdicts: usize,
    ) {
        let label = pool.label;
        let Some(slot) = state.members.get_mut(member) else {
            return;
        };
        slot.pending = false;
        let client_id = slot.client_id.clone();
        let (left, budget) = match &pool.kind {
            PoolKind::Multiplexed(members) => (
                fred::live_edit::pool::spare_cells_left(members),
                fred::live_edit::pool::spare_cells(members),
            ),
            PoolKind::Txn(txn) => (
                state.txn_replacements_left,
                txn.slots().max(fred::live_edit::pool::MIN_SPARE_CELLS),
            ),
        };
        if left == 0 {
            // Counted once per refused replacement; the WARN once per pool.
            self.count(label, VALKEY_MEMBER_REPLACEMENT_BUDGET_EXHAUSTED);
            if !state.budget_warned {
                state.budget_warned = true;
                match &pool.kind {
                    PoolKind::Multiplexed(_) => tracing::warn!(
                        pool = label,
                        member,
                        client_id,
                        spare_cells = budget,
                        "a pool member is due for replacement but the pool's spare cells are \
                         used up; it keeps its client, and fast unresponsive verdicts keep \
                         rebuilding its connections, until the process restarts (logged once \
                         per pool)"
                    ),
                    PoolKind::Txn(_) => tracing::warn!(
                        pool = label,
                        member,
                        client_id,
                        replacements = budget,
                        "a txn slot is due for replacement but the txn pool's replacement \
                         budget is used up; it keeps its client, and fast unresponsive verdicts \
                         keep rebuilding its connections, until the process restarts (logged \
                         once per pool)"
                    ),
                }
            }
            return;
        }
        // Counted afresh on the fresh client, or, if this attempt fails, from
        // the next verdict on.
        state.members[member].fast_verdicts.clear();
        let old = match &pool.kind {
            PoolKind::Multiplexed(members) => members.member(member).clone(),
            PoolKind::Txn(_) => match state.members[member].client.clone() {
                Some(client) => client,
                None => return,
            },
        };
        state.in_flight = true;
        let replacer = Arc::clone(self);
        self.tracker.spawn(async move {
            replacer.replace(label, member, old, verdicts).await;
        });
    }

    /// Free pool `label`'s slot, then start the replacement of the
    /// lowest-index member that came due while it was held and is still due
    /// ([`PoolReplaceState::take_next_due`]), if any — at most one, since that
    /// one takes the slot again (or, refused for budget, the budget is spent
    /// for every member). Nothing starts once the replacer is shut down. Only
    /// the replacement holding the slot calls it ([`Replacement::release_slot`]).
    fn release_slot(self: &Arc<Self>, label: &str) {
        let Some(pool) = self.pool(label) else {
            return;
        };
        let knobs = self.knobs();
        let mut state = pool
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.in_flight = false;
        if self.cancel.is_cancelled() {
            return;
        }
        if let Some(member) =
            state.take_next_due(Instant::now(), knobs.window, knobs.after_fast_verdicts)
        {
            let verdicts = state.members[member].fast_verdicts.len();
            self.try_start(pool, &mut state, member, verdicts);
        }
    }

    /// Register the retirement of `client_id`, member `member` of pool `pool`,
    /// which a swap just retired, and return its id and its stop token (a
    /// child of the shutdown token). If that makes more than
    /// [`MEMBER_RETIRING_CAP`] retirements in progress, the oldest one other
    /// than this new one is ended early, whatever its pool: its stop token is
    /// cancelled, so it returns at its next await and its guard cuts it short
    /// (router aborted, hooks disarmed, client released), and one WARN names
    /// it.
    fn begin_retirement(
        &self,
        pool: &'static str,
        member: usize,
        client_id: &str,
    ) -> (u64, CancellationToken) {
        let mut retiring = self
            .retiring
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let id = retiring.next_id;
        retiring.next_id += 1;
        let stop = self.cancel.child_token();
        retiring.entries.push_back(RetiringEntry {
            id,
            pool,
            member,
            client_id: client_id.to_owned(),
            stop: stop.clone(),
        });
        retiring.evict_over_cap(id);
        (id, stop)
    }

    /// Forget retirement `id` when it ends, whichever way (its guard's drop).
    /// A no-op for one the cap already evicted.
    fn end_retirement(&self, id: u64) {
        self.retiring
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .entries
            .retain(|entry| entry.id != id);
    }

    fn retiring_count(&self) -> usize {
        self.retiring
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .entries
            .len()
    }

    fn count(&self, label: &'static str, outcome: &'static str) {
        if let Some(metrics) = self.hooks.metrics() {
            metrics.inc_valkey_member_replacement(label, outcome);
        }
    }

    /// One replacement of member `member` of pool `label`, whose current client
    /// is `old`; `verdicts` fast verdicts triggered it, and it holds the pool's
    /// slot from its trigger. See the module docs for the steps. Every exit — a
    /// return, a cancellation at an await, an abort or a panic — goes through
    /// the [`Replacement`] guard. The slot is released at the swap; this frees
    /// it once more, idempotently, for every attempt that ended before one (a
    /// failed connect, a cancellation, a `txn` slot's early return).
    async fn replace(
        self: Arc<Self>,
        label: &'static str,
        member: usize,
        old: Client,
        verdicts: usize,
    ) {
        let mut guard = Replacement {
            replacer: Arc::clone(&self),
            label,
            holds_slot: true,
            retirement: None,
            fresh_router: None,
            fresh_watch: None,
            fresh: None,
            old_watch: None,
            old_router: None,
            old: None,
        };
        self.replace_steps(&mut guard, label, member, old, verdicts)
            .await;
        guard.release_slot();
    }

    /// The steps of [`Self::replace`], every early return leaving the cleanup
    /// to `guard`.
    async fn replace_steps(
        self: &Arc<Self>,
        guard: &mut Replacement,
        label: &'static str,
        member: usize,
        old: Client,
        verdicts: usize,
    ) {
        let Some(pool) = self.pool(label) else {
            return;
        };
        let knobs = self.knobs();
        let old_id = ClientLike::id(&old).to_owned();

        // 1–2. Build and connect the fresh client, its router's handle kept
        // from the spawn so a connect that is cut off can still be aborted.
        // A `snapshot` member's router starts on `snapshot-io`, where the
        // lane's routers run (`storage::fanout_thread`), and is restarted
        // there (step 4).
        let fresh = old.clone_new();
        let fresh_id = ClientLike::id(&fresh).to_owned();
        guard.fresh = Some(fresh.clone());
        let spawn_on = if label == POOL_LABEL_SNAPSHOT {
            self.snapshot_io.clone()
        } else {
            None
        };
        let (router, outcome) = {
            let mut connected = std::pin::pin!(fresh.wait_for_connect());
            let before = futures::poll!(connected.as_mut());
            let router = connect_router_on(spawn_on.as_ref(), &fresh);
            guard.fresh_router = Some(router.abort_handle());
            let outcome = match before {
                std::task::Poll::Ready(result) => Some(result),
                std::task::Poll::Pending => tokio::select! {
                    biased;
                    () = self.cancel.cancelled() => None,
                    result = tokio::time::timeout(self.connect_timeout, connected) => Some(
                        result.unwrap_or_else(|_| {
                            Err(fred::error::Error::new(
                                fred::error::ErrorKind::Timeout,
                                format!(
                                    "the fresh client was not connected within {} s \
                                     (LIVE_EDIT_VALKEY_CONNECT_TIMEOUT_SECS)",
                                    self.connect_timeout.as_secs()
                                ),
                            ))
                        }),
                    ),
                },
            };
            (router, outcome)
        };
        match outcome {
            None => return,
            Some(Ok(())) => {}
            Some(Err(error)) => {
                self.count(label, VALKEY_MEMBER_REPLACEMENT_CONNECT_FAILED);
                tracing::warn!(
                    pool = label,
                    member,
                    client_id = %old_id,
                    fresh_client_id = %fresh_id,
                    error = %error,
                    "pool member replacement failed: the fresh client did not connect; the \
                     member keeps its client"
                );
                return;
            }
        }

        // 3. Hooks under the fresh id, after its first connect.
        self.arm(label, member, &fresh);

        let members = match &pool.kind {
            PoolKind::Multiplexed(members) => members,
            PoolKind::Txn(txn) => {
                self.swap_txn_slot(guard, pool, txn, member, old_id, fresh, router, verdicts)
                    .await;
                return;
            }
        };

        // 4. Supervise its router from here on.
        let restart = fresh.clone();
        let watch = self.supervisor.watch_retirable(
            label,
            member,
            &fresh_id,
            router,
            RouterRestart::Connect(Box::new(move || {
                connect_router_on(spawn_on.as_ref(), &restart)
            })),
        );
        guard.fresh_router = None;
        guard.fresh_watch = Some(watch.abandon_handle());

        // 5. Point the member at it.
        let (retired, old_watch, spare_left) = {
            let mut state = pool
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if self.cancel.is_cancelled() {
                return;
            }
            let retired = match fred::live_edit::pool::replace_member(members, member, fresh) {
                Ok(retired) => retired,
                Err(fred::live_edit::pool::BudgetExhausted) => {
                    // The trigger checked for a spare cell and this is the
                    // replacement holding the pool's slot, the only one that
                    // can take a spare cell, so this is not reached; handled
                    // as the trigger's refusal.
                    self.count(label, VALKEY_MEMBER_REPLACEMENT_BUDGET_EXHAUSTED);
                    return;
                }
            };
            let slot = &mut state.members[member];
            let old_watch = slot.watch.replace(watch);
            slot.client_id = fresh_id.clone();
            slot.replaced = Some(Replaced {
                at: Instant::now(),
                previous_id: old_id.clone(),
            });
            slot.fast_verdicts.clear();
            // A verdict on the old client while this replacement held the slot
            // marked it pending; the member is replaced now.
            slot.pending = false;
            (
                retired,
                old_watch,
                fred::live_edit::pool::spare_cells_left(members),
            )
        };
        // Active: from here on the guard retires the old client, not the fresh.
        guard.fresh_watch = None;
        guard.fresh = None;
        guard.old_watch = old_watch.as_ref().map(RetirableWatch::abandon_handle);
        guard.old = Some(retired.clone());
        // The retirement runs on without the pool's slot: register it under
        // the per-replica cap, then free the slot, which starts the next
        // member that came due meanwhile.
        let (retirement, stop) = self.begin_retirement(label, member, &old_id);
        // The grace runs from the swap: the watch's retire below waits inside
        // it, not before it, so the retirement lasts one grace before its
        // `QUIT`.
        let grace_ends = Instant::now() + knobs.grace;
        guard.retirement = Some(retirement);
        guard.release_slot();
        self.count(label, VALKEY_MEMBER_REPLACEMENT_REPLACED);
        tracing::warn!(
            pool = label,
            member,
            client_id = %old_id,
            fresh_client_id = %fresh_id,
            fast_verdicts = verdicts,
            spare_cells_left = spare_left,
            "replaced a pool member that kept failing with a fresh Valkey client; the old \
             client is retired after the commands handed to it finish"
        );

        // 6. Retire the old client. `stop` ends it early: shutdown, or the
        // per-replica cap evicting it.
        let router: Option<AbortHandle> = match old_watch {
            Some(watch) => tokio::select! {
                biased;
                () = stop.cancelled() => return,
                router = self.supervisor.retire(watch) => Some(router),
            },
            None => None,
        };
        tokio::select! {
            biased;
            () = stop.cancelled() => return,
            () = tokio::time::sleep_until(grace_ends) => {}
        }
        if !self.quit_and_release(&retired, router, &stop).await {
            return;
        }
        guard.old_watch = None;
        guard.old = None;
    }

    /// Steps 4–6 for a `txn` slot (see the module docs): take the slot under
    /// its guard once no caller holds it, swap `fresh` (connected, hooks armed,
    /// its router `router`) in, re-key the slot's watch, release the slot, then
    /// retire the old client. Every early return leaves the cleanup to `guard`.
    #[allow(clippy::too_many_arguments)]
    async fn swap_txn_slot(
        &self,
        guard: &mut Replacement,
        pool: &ReplaceablePool,
        txn: &Arc<ExclusiveCheckout>,
        member: usize,
        old_id: String,
        fresh: Client,
        router: fred::types::ConnectHandle,
        verdicts: usize,
    ) {
        let label = pool.label;
        let fresh_id = ClientLike::id(&fresh).to_owned();

        // 4. The slot itself, through the checkout like any caller (a permit,
        // then this slot's mutex): this waits for as long as a caller holds the
        // slot, so a swap never lands mid-transaction.
        let checkout = tokio::select! {
            biased;
            () = self.cancel.cancelled() => return,
            checkout = txn.checkout_slot_for_replacement(&old_id) => checkout,
        };
        let Some(mut checkout) = checkout else {
            tracing::warn!(
                pool = label,
                member,
                client_id = %old_id,
                fresh_client_id = %fresh_id,
                "txn slot replacement abandoned: the pool is shutting down, or the slot no \
                 longer holds this client"
            );
            return;
        };
        let restart = self
            .txn_restart
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        let Some(restart) = restart else {
            // Shut down (or supervision never started): nothing to watch with.
            return;
        };

        // 5. Under the slot's guard: swap the client in, end the old client's
        // slot watch and watch the fresh one under its own id.
        let (retired, old_router, left) = {
            let mut state = pool
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if self.cancel.is_cancelled() {
                return;
            }
            let Some(retired) = txn.swap_slot_client(&mut checkout, fresh.clone()) else {
                return;
            };
            let old_router = self.supervisor.retire_slot(&old_id);
            self.supervisor.watch(
                label,
                member,
                router,
                restart(fresh_id.clone()),
                Some(fresh_id.clone()),
            );
            // Active: from here on the guard retires the old client.
            guard.fresh_router = None;
            guard.fresh = None;
            guard.old_router = old_router.clone();
            guard.old = Some(retired.clone());
            state.txn_replacements_left = state.txn_replacements_left.saturating_sub(1);
            let left = state.txn_replacements_left;
            let slot = &mut state.members[member];
            slot.client_id = fresh_id.clone();
            slot.client = Some(fresh);
            slot.replaced = Some(Replaced {
                at: Instant::now(),
                previous_id: old_id.clone(),
            });
            slot.fast_verdicts.clear();
            // A verdict on the old client while this replacement held the slot
            // marked it pending; the member is replaced now.
            slot.pending = false;
            (retired, old_router, left)
        };
        // The slot, then its permit: the next checkout of it gets the fresh
        // client.
        drop(checkout);
        // The retirement runs on without the pool's slot, as a multiplexed
        // member's does.
        let (retirement, stop) = self.begin_retirement(label, member, &old_id);
        guard.retirement = Some(retirement);
        guard.release_slot();
        self.count(label, VALKEY_MEMBER_REPLACEMENT_REPLACED);
        tracing::warn!(
            pool = label,
            member,
            client_id = %old_id,
            fresh_client_id = %fresh_id,
            fast_verdicts = verdicts,
            replacements_left = left,
            "replaced a txn slot's Valkey client that kept failing with a fresh one; the old \
             client is retired"
        );

        // 6. No grace: the old client left the slot, and every caller reaches a
        // slot's client only through a checkout, so nothing can be using it.
        if !self.quit_and_release(&retired, old_router, &stop).await {
            return;
        }
        guard.old_router = None;
        guard.old = None;
    }

    /// The end of a retirement, shared by both pool kinds: disarm the old
    /// client's hooks, then end its router through fred's own `QUIT` path,
    /// abort it only as the last resort, and release what fred would leave
    /// behind. `false` when `stop` (the retirement's own token: shutdown, or
    /// the per-replica cap evicting it) fired meanwhile; the caller then
    /// returns and its guard cuts the retirement short.
    ///
    /// **The release (`fred::live_edit::pool::retire`) is detached**: it runs
    /// on a task spawned on the runtime, not on this replacer's tracker, so
    /// neither this retirement nor shutdown waits for it. That is safe because
    /// it is bounded and holds nothing of ours: it waits at most
    /// `RETIRE_ROUTER_WAIT` (5 s) for the router to finish, its only I/O is one
    /// socket's close — the client's backchannel connection — bounded by
    /// `RETIRE_BACKCHANNEL_CLOSE_TIMEOUT` (1 s), so about 6 s in all, it holds
    /// no pool or slot, and process exit ends it. One residual: a close that
    /// times out while still waiting for the backchannel's lock takes nothing,
    /// so that socket stays open until the client is dropped — for a
    /// multiplexed member's old client, with its pool (fred logs a WARN when it
    /// happens; `pool::retire`'s doc says when it can).
    async fn quit_and_release(
        &self,
        retired: &Client,
        router: Option<AbortHandle>,
        stop: &CancellationToken,
    ) -> bool {
        self.hooks.disarm(ClientLike::id(retired));
        // End the router through fred's own `QUIT` path, whatever the client's
        // state reads: it drains the replies still owed and fails every queued
        // command with an error, so a caller still walking a `SCAN` stream on
        // this client sees an error, not an end of stream that reads as "walked
        // everything". An abort drops all of that unanswered, so it is only the
        // last resort, once [`MEMBER_RETIRE_QUIT_BOUND`] has passed. The `QUIT`
        // carries the same bound as its own timeout: fred skips a command whose
        // caller timed out, and the default command timeout would cut it first.
        let deadline = Instant::now() + MEMBER_RETIRE_QUIT_BOUND;
        let options = fred::types::config::Options {
            timeout: Some(MEMBER_RETIRE_QUIT_BOUND),
            ..Default::default()
        };
        let quit = retired.with_options(&options);
        tokio::select! {
            biased;
            () = stop.cancelled() => return false,
            _ = tokio::time::timeout_at(deadline, ClientLike::quit(&quit)) => {}
        }
        if let Some(router) = router.as_ref() {
            tokio::select! {
                biased;
                () = stop.cancelled() => return false,
                () = router_ended(router, deadline) => {}
            }
            // The last resort; nothing to do when the `QUIT` ended it.
            router.abort();
        }
        // After the router has finished (`retire` waits for it): an aborted
        // router that had just connected could otherwise re-arm the credential
        // task, and a `QUIT` exit puts the command receiver back.
        fred::live_edit::pool::retire(retired, router);
        true
    }

    /// Stop replacing: no replacement starts from here on (a slot released
    /// after this starts no pending member), and every replacement and every
    /// retirement in progress — however many — is cancelled at its next await,
    /// all of them in parallel (a retirement's stop token is a child of this
    /// one; its guard cleans up), and waited for, bounded by
    /// [`MEMBER_REPLACER_DRAIN_TIMEOUT`]. Called by
    /// `ValkeyClient::quit` before the router supervisor shuts down. Its
    /// first, synchronous part is [`Self::release`]. Idempotent.
    pub(crate) async fn shutdown(&self) {
        self.release();
        if tokio::time::timeout(MEMBER_REPLACER_DRAIN_TIMEOUT, self.tracker.wait())
            .await
            .is_err()
        {
            tracing::warn!(
                replacements_remaining = self.tracker.len(),
                timeout_secs = MEMBER_REPLACER_DRAIN_TIMEOUT.as_secs(),
                "timed out waiting for in-flight pool member replacements to stop; closing the \
                 pools anyway"
            );
        }
    }

    /// Stop replacing without waiting: no replacement starts from here on,
    /// every one in progress and every retirement is cancelled at its next
    /// await, and the `txn` slots' router restart (see the `txn_restart`
    /// field) is dropped. The first part of [`Self::shutdown`], and all of
    /// what the drop of the client's last owner does here
    /// (`ValkeyClient`'s `ClientOwners`), which cannot wait. Idempotent.
    pub(crate) fn release(&self) {
        {
            // Under every pool's lock, so no trigger that read the token as
            // live is still about to spawn.
            let _held: Vec<_> = self
                .pools
                .iter()
                .map(|pool| {
                    pool.state
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                })
                .collect();
            self.cancel.cancel();
        }
        self.txn_restart
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        self.tracker.close();
    }

    /// Whether pool `label`'s slot is held: a replacement of one of its
    /// members is between its trigger and its swap. An old client's retirement
    /// does not hold it. **Test-only.**
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) fn in_flight_for_test(&self, label: &str) -> bool {
        self.pool(label).is_some_and(|pool| {
            pool.state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .in_flight
        })
    }

    /// The replacement tasks still running, across every pool: replacements
    /// before their swap and old clients' retirements after it. **Test-only.**
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) fn tasks_for_test(&self) -> usize {
        self.tracker.len()
    }

    /// The old clients' retirements in progress, across every pool: at most
    /// [`MEMBER_RETIRING_CAP`]. **Test-only.**
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) fn retiring_for_test(&self) -> usize {
        self.retiring_count()
    }
}

/// Wait until the router behind `router` has ended, or `deadline`.
async fn router_ended(router: &AbortHandle, deadline: Instant) {
    while !router.is_finished() && Instant::now() < deadline {
        tokio::time::sleep(ROUTER_END_POLL).await;
    }
}

/// The cleanup of one replacement, whichever way it ends. Before the member is
/// pointed at the fresh client, the fresh client is the one torn down (its
/// router aborted, its watch abandoned, its hooks disarmed); after, the old
/// client's retirement is cut short (its watch abandoned, its current router
/// aborted at once, its hooks disarmed) — on shutdown, on a panic, or when the
/// per-replica cap ([`MEMBER_RETIRING_CAP`]) evicts it. Either client is then
/// released (`fred::live_edit::pool::retire`, once the aborted router has
/// finished: command channel closed, credential-refresh task stopped,
/// backchannel closed within a bound — see
/// [`MemberReplacer::quit_and_release`] for its rare residual). A completed
/// replacement has cleared every client field.
///
/// **The pool's slot.** The replacement holds it from the trigger
/// (`holds_slot`) until [`Self::release_slot`] frees it — at the swap, or when
/// the attempt ends before one. Only the holder ever clears the pool's
/// `in_flight`: a guard that already released the slot never touches it
/// again, so a retirement that ends later cannot clear the flag a later
/// replacement of the same pool now holds. A guard dropped while still holding
/// it (a panic, or the task dropped by the runtime, before the swap) clears the flag but starts
/// nothing: a drop never spawns, so a member that came due meanwhile waits for
/// its next fast verdict.
struct Replacement {
    replacer: Arc<MemberReplacer>,
    label: &'static str,
    /// The replacement still holds its pool's slot.
    holds_slot: bool,
    /// The old client's retirement, once the swap registered it
    /// ([`MemberReplacer::begin_retirement`]); forgotten when this drops.
    retirement: Option<u64>,
    fresh_router: Option<AbortHandle>,
    fresh_watch: Option<AbandonHandle>,
    fresh: Option<Client>,
    old_watch: Option<AbandonHandle>,
    /// A `txn` slot's old client's router (a multiplexed member's is reached
    /// through `old_watch`).
    old_router: Option<AbortHandle>,
    old: Option<Client>,
}

impl Replacement {
    /// Free the pool's slot, if this replacement still holds it, and start the
    /// next member that came due meanwhile ([`MemberReplacer::release_slot`]).
    /// Idempotent: called at the swap and again when the replacement's steps
    /// return.
    fn release_slot(&mut self) {
        if std::mem::take(&mut self.holds_slot) {
            self.replacer.release_slot(self.label);
        }
    }
}

impl Drop for Replacement {
    fn drop(&mut self) {
        // Each client is released only once the router aborted here has
        // finished (`retire` waits for it on a spawned task, so this does not
        // block on it). `abandon` returns the client's last router: it waits
        // out, on the watch's `current` lock, at most a restart's synchronous
        // `connect()`, and no restart starts after it.
        let mut fresh_router = self.fresh_watch.take().map(|watch| watch.abandon());
        if let Some(router) = self.fresh_router.take() {
            router.abort();
            fresh_router = Some(router);
        }
        if let Some(fresh) = self.fresh.take() {
            self.replacer.hooks.disarm(ClientLike::id(&fresh));
            fred::live_edit::pool::retire(&fresh, fresh_router);
        }
        let mut old_router = self.old_watch.take().map(|watch| watch.abandon());
        if let Some(router) = self.old_router.take() {
            router.abort();
            old_router = Some(router);
        }
        if let Some(old) = self.old.take() {
            self.replacer.hooks.disarm(ClientLike::id(&old));
            fred::live_edit::pool::retire(&old, old_router);
        }
        if let Some(retirement) = self.retirement.take() {
            self.replacer.end_retirement(retirement);
        }
        // Only while this replacement still holds the slot; never a re-check.
        if self.holds_slot {
            if let Some(pool) = self.replacer.pool(self.label) {
                pool.state
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .in_flight = false;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    //! b.zt3 postcomp-1 — the scheduling decisions and the retiring cap, with
    //! no Valkey.
    //!
    //! The end-to-end behaviour (a pending member started at the swap, the
    //! cap evicting a real retirement, slot ownership against a real
    //! retirement's end) is in
    //! `tests/integration/storage/b_zt3_member_replacement_test.rs`, on real
    //! time: tokio's paused clock fires fred's connect and command timeouts
    //! during real socket round trips, so a paused-clock test can only reach
    //! the decisions themselves. These pin those, exactly:
    //!
    //! | Test | Design item | What it pins |
    //! |---|---|---|
    //! | [`the_recheck_starts_the_lowest_index_pending_member_still_due_and_leaves_later_ones_marked`] | T-R3 | `take_next_due`: lowest index first, a due member never marked is never taken, a pending member no longer due is passed over with its mark cleared |
    //! | [`a_pending_member_whose_verdicts_aged_out_of_the_window_is_skipped_and_unmarked`] | T-R3 | after `tokio::time::advance` past the window: skipped, its mark cleared, its verdicts pruned |
    //! | [`a_member_due_while_the_slot_is_held_is_marked_and_a_release_with_the_budget_spent_starts_nothing`] | T-R3 | `on_fast_verdict` marks instead of starting; `release_slot` with the budget spent frees the slot, starts nothing, counts `budget_exhausted` once per release |
    //! | [`a_release_after_shutdown_frees_the_slot_and_starts_nothing`] | T-R3 | the re-check never starts anything once the replacer is shut down |
    //! | [`a_swap_past_the_cap_ends_the_oldest_retirement_and_never_the_new_one`] | Rider A | the cap at 4: the oldest is evicted (its stop token cancelled), the new one never |
    //! | [`maintenance_retirements_between_older_and_newer_ones_are_evicted_in_age_order`] | Rider A, no pool exempt | a `maintenance` pool of 2 retiring between a data and a snapshot retirement, no time passing before the swaps (so still inside their grace): three swaps evict data, then each `maintenance` one, oldest first |
    //! | [`four_maintenance_retirements_never_hold_the_count_above_the_cap`] | Rider A bound, no pool exempt | four `maintenance` retirements, no time passing before the swaps (so still inside their grace), fill the cap: each later swap evicts the oldest of them, and the count is 4 after every swap, never 5 |
    //!
    //! The replacer is built over unconnected fred pools: nothing here sends a
    //! command, and the `txn` pool's budget (a count kept here, set to zero
    //! where a test needs it spent) is what keeps a release from spawning a
    //! replacement that would connect.
    //!
    //! Fail-on-old, by in-place perturbation of the named site, each turning
    //! the named tests red on their `cancelled(..)`, count or mark assertions:
    //! making `Retiring::evict_over_cap` pass over `maintenance` entries (the
    //! in-grace exemption the cap once had) — the two `maintenance` tests;
    //! removing the eviction from `begin_retirement` — the cap test and the
    //! two `maintenance` tests; deleting the re-check from `release_slot` —
    //! the budget-spent release test.

    use std::collections::HashMap;

    use fred::prelude::Builder;

    use super::*;
    use crate::observability::metrics::test_util::{render, sample_value};
    use crate::observability::metrics::{metrics_scrape_only, Metrics, SharedMetrics};

    /// A slot whose client is named `id`, with `verdicts` fast verdicts, marked
    /// `pending` or not.
    fn member(id: &str, verdicts: &[Instant], pending: bool) -> MemberSlot {
        MemberSlot {
            client_id: id.to_owned(),
            client: None,
            watch: None,
            fast_verdicts: verdicts.iter().copied().collect(),
            replaced: None,
            pending,
        }
    }

    fn state(members: Vec<MemberSlot>) -> PoolReplaceState {
        PoolReplaceState {
            members,
            ..PoolReplaceState::default()
        }
    }

    fn pending_marks(state: &PoolReplaceState) -> Vec<bool> {
        state.members.iter().map(|slot| slot.pending).collect()
    }

    /// A replacer over unconnected pools — `maintenance` members in the
    /// `maintenance` pool, two `txn` slots — counting on a fresh registry.
    fn offline_replacer(maintenance: usize) -> (Arc<MemberReplacer>, SharedMetrics) {
        let metrics: SharedMetrics = Arc::new(metrics_scrape_only("zt3-replacer"));
        let metrics_slot = Arc::new(Mutex::new(Some(Arc::clone(&metrics))));
        let pool = |size: usize| {
            Builder::default_centralized()
                .build_pool(size)
                .expect("an unconnected pool builds")
        };
        let exclusive = Builder::default_centralized()
            .build_exclusive_pool(2)
            .expect("an unconnected exclusive pool builds");
        let txn_clients: Vec<Client> = exclusive
            .clients()
            .iter()
            .map(|slot| {
                slot.try_lock()
                    .expect("nothing holds a slot of a pool just built")
                    .clone()
            })
            .collect();
        let txn = Arc::new(ExclusiveCheckout::new(
            exclusive,
            POOL_LABEL_TXN,
            Arc::clone(&metrics_slot),
        ));
        let replacer = MemberReplacer::new(
            pool(2),
            pool(maintenance),
            pool(1),
            txn,
            txn_clients,
            Arc::new(RouterSupervisor::new(Arc::clone(&metrics_slot))),
            PoolHooks {
                metrics: metrics_slot,
                background_tasks: Arc::new(Mutex::new(Some(Vec::new()))),
                registry: Arc::new(Mutex::new(HashMap::new())),
                diagnostics: None,
            },
            Duration::from_secs(5),
            Duration::from_secs(5),
            None,
        );
        (replacer, metrics)
    }

    fn txn_state(replacer: &MemberReplacer) -> std::sync::MutexGuard<'_, PoolReplaceState> {
        replacer
            .pool(POOL_LABEL_TXN)
            .expect("the txn pool is replaceable")
            .state
            .lock()
            .expect("the pool lock is not poisoned")
    }

    fn txn_budget_exhausted(metrics: &Metrics) -> f64 {
        sample_value(
            &render(metrics),
            "live_edit_valkey_member_replacements_total",
            &[
                ("pool", POOL_LABEL_TXN),
                ("outcome", VALKEY_MEMBER_REPLACEMENT_BUDGET_EXHAUSTED),
            ],
        )
        .expect("every member-replacement cell is zero-seeded")
    }

    /// T-R3: the re-check at a slot's release takes the lowest-index member
    /// still due among those marked pending, clearing its mark, and leaves
    /// later marked members for the next release. A member that is due but was
    /// never marked (its verdict was not refused by a held slot) is never
    /// taken, and a marked member no longer due is passed over with its mark
    /// cleared.
    #[tokio::test(start_paused = true)]
    async fn the_recheck_starts_the_lowest_index_pending_member_still_due_and_leaves_later_ones_marked(
    ) {
        let now = Instant::now();
        let due = [now, now];
        let mut state = state(vec![
            member("never-marked", &due, false),
            member("first", &due, true),
            member("not-due", &[now], true),
            member("second", &due, true),
        ]);

        assert_eq!(
            state.take_next_due(
                now,
                MEMBER_REPLACE_WINDOW,
                MEMBER_REPLACE_AFTER_FAST_VERDICTS
            ),
            Some(1),
            "the lowest-index pending member still due"
        );
        assert_eq!(
            pending_marks(&state),
            [false, false, true, true],
            "its mark is cleared; the later ones keep theirs for the next release"
        );
        assert_eq!(
            state.take_next_due(
                now,
                MEMBER_REPLACE_WINDOW,
                MEMBER_REPLACE_AFTER_FAST_VERDICTS
            ),
            Some(3),
            "the next release passes over the member no longer due"
        );
        assert_eq!(
            pending_marks(&state),
            [false, false, false, false],
            "the member passed over has its mark cleared too"
        );
        assert_eq!(
            state.take_next_due(
                now,
                MEMBER_REPLACE_WINDOW,
                MEMBER_REPLACE_AFTER_FAST_VERDICTS
            ),
            None,
            "a due member that was never marked is never started by the re-check"
        );
        assert_eq!(state.members[0].fast_verdicts.len(), 2);
    }

    /// T-R3: a pending member whose verdicts aged out of the 60-minute window
    /// while it waited is skipped and its mark cleared, its stale verdicts
    /// pruned; a member due inside the window behind it is the one taken.
    #[tokio::test(start_paused = true)]
    async fn a_pending_member_whose_verdicts_aged_out_of_the_window_is_skipped_and_unmarked() {
        let before = Instant::now();
        tokio::time::advance(MEMBER_REPLACE_WINDOW + Duration::from_secs(1)).await;
        let now = Instant::now();
        let mut state = state(vec![
            member("aged-out", &[before, before], true),
            member("half-aged", &[before, now], true),
            member("due", &[now, now], true),
        ]);

        assert_eq!(
            state.take_next_due(
                now,
                MEMBER_REPLACE_WINDOW,
                MEMBER_REPLACE_AFTER_FAST_VERDICTS
            ),
            Some(2)
        );
        assert_eq!(pending_marks(&state), [false, false, false]);
        assert_eq!(
            state
                .members
                .iter()
                .map(|slot| slot.fast_verdicts.len())
                .collect::<Vec<_>>(),
            [0, 1, 2],
            "the verdicts older than the window are pruned on the way"
        );
    }

    /// T-R3: a fast verdict that brings a member due while another replacement
    /// holds the pool's slot marks it pending instead of starting it; the
    /// slot's release, with the pool's budget spent, frees the slot, starts
    /// nothing, clears the refused member's mark and counts one
    /// `budget_exhausted` — once per release, for the one member it tried —
    /// leaving the next marked member for the next release.
    #[tokio::test]
    async fn a_member_due_while_the_slot_is_held_is_marked_and_a_release_with_the_budget_spent_starts_nothing(
    ) {
        let (replacer, metrics) = offline_replacer(1);
        txn_state(&replacer).in_flight = true;

        replacer.fast_verdict_for_test(POOL_LABEL_TXN, 0);
        assert_eq!(
            pending_marks(&txn_state(&replacer)),
            [false, false],
            "one verdict is not due"
        );
        replacer.fast_verdict_for_test(POOL_LABEL_TXN, 0);
        for _ in 0..MEMBER_REPLACE_AFTER_FAST_VERDICTS {
            replacer.fast_verdict_for_test(POOL_LABEL_TXN, 1);
        }
        assert_eq!(
            pending_marks(&txn_state(&replacer)),
            [true, true],
            "each member due while the slot is held is marked, not started"
        );
        assert_eq!(replacer.tasks_for_test(), 0, "nothing started");

        txn_state(&replacer).txn_replacements_left = 0;
        replacer.release_slot(POOL_LABEL_TXN);

        {
            let state = txn_state(&replacer);
            assert!(!state.in_flight, "the release frees the slot");
            assert_eq!(
                pending_marks(&state),
                [false, true],
                "the refused member's mark is cleared; the next keeps its own"
            );
            assert!(state.budget_warned);
        }
        assert_eq!(replacer.tasks_for_test(), 0, "nothing started");
        assert_eq!(txn_budget_exhausted(&metrics), 1.0, "one refusal counted");

        txn_state(&replacer).in_flight = true;
        replacer.release_slot(POOL_LABEL_TXN);
        assert_eq!(pending_marks(&txn_state(&replacer)), [false, false]);
        assert_eq!(
            txn_budget_exhausted(&metrics),
            2.0,
            "the next release refuses the next marked member, counted once"
        );
        assert_eq!(replacer.tasks_for_test(), 0);
    }

    /// T-R3: once the replacer is shut down, a slot's release frees the slot
    /// and starts nothing, refused or otherwise: a marked member stays marked
    /// and nothing is counted.
    #[tokio::test]
    async fn a_release_after_shutdown_frees_the_slot_and_starts_nothing() {
        let (replacer, metrics) = offline_replacer(1);
        txn_state(&replacer).in_flight = true;
        for _ in 0..MEMBER_REPLACE_AFTER_FAST_VERDICTS {
            replacer.fast_verdict_for_test(POOL_LABEL_TXN, 0);
        }
        txn_state(&replacer).txn_replacements_left = 0;
        replacer.cancel.cancel();

        replacer.release_slot(POOL_LABEL_TXN);

        let state = txn_state(&replacer);
        assert!(!state.in_flight);
        assert_eq!(pending_marks(&state), [true, false]);
        assert_eq!(txn_budget_exhausted(&metrics), 0.0, "nothing was tried");
        assert_eq!(replacer.tasks_for_test(), 0);
    }

    /// Register a retirement of `pool`'s member `member`, client `id`.
    fn retire(
        replacer: &MemberReplacer,
        pool: &'static str,
        member: usize,
        id: &str,
    ) -> (u64, CancellationToken) {
        replacer.begin_retirement(pool, member, id)
    }

    fn cancelled(stops: &[&CancellationToken]) -> Vec<bool> {
        stops.iter().map(|stop| stop.is_cancelled()).collect()
    }

    /// Rider A: a swap that would make five retirements in progress ends the
    /// oldest one — its stop token cancelled, so its task returns and its guard
    /// cuts it short — and never the swap's own new one.
    #[test]
    fn a_swap_past_the_cap_ends_the_oldest_retirement_and_never_the_new_one() {
        let (replacer, _metrics) = offline_replacer(1);
        let olds: Vec<(u64, CancellationToken)> = (0..MEMBER_RETIRING_CAP)
            .map(|m| retire(&replacer, POOL_LABEL_DATA, m, &format!("data-{m}")))
            .collect();
        assert_eq!(replacer.retiring_for_test(), MEMBER_RETIRING_CAP);
        assert!(
            olds.iter().all(|(_, stop)| !stop.is_cancelled()),
            "at the cap nothing is evicted"
        );

        let (_, new) = retire(&replacer, POOL_LABEL_SNAPSHOT, 0, "snapshot-0");

        assert_eq!(replacer.retiring_for_test(), MEMBER_RETIRING_CAP);
        assert_eq!(
            cancelled(&[&olds[0].1, &olds[1].1, &olds[2].1, &olds[3].1, &new]),
            [true, false, false, false, false],
            "only the oldest is ended"
        );
    }

    /// Rider A, no pool exempt: a `maintenance` pool of two, both members
    /// retiring between an older data retirement and a newer snapshot one. No
    /// time passes here between registration and eviction, so each is still
    /// inside its grace when it is ended — the case the cap once exempted.
    /// Each swap past the cap ends the oldest retirement left, so the data one
    /// goes first and then each `maintenance` one in turn, older first; the
    /// snapshot retirement and every swap's own new one survive, and the count
    /// stays at the cap.
    #[test]
    fn maintenance_retirements_between_older_and_newer_ones_are_evicted_in_age_order() {
        let (replacer, _metrics) = offline_replacer(2);
        let (_, data_0) = retire(&replacer, POOL_LABEL_DATA, 0, "data-0");
        let (_, maintenance_0) = retire(&replacer, POOL_LABEL_MAINTENANCE, 0, "maintenance-0");
        let (_, maintenance_1) = retire(&replacer, POOL_LABEL_MAINTENANCE, 1, "maintenance-1");
        let (_, snapshot_0) = retire(&replacer, POOL_LABEL_SNAPSHOT, 0, "snapshot-0");
        assert_eq!(replacer.retiring_for_test(), MEMBER_RETIRING_CAP);

        let mut swaps = Vec::new();
        let mut ended = Vec::new();
        for member in 0..3 {
            swaps.push(retire(
                &replacer,
                POOL_LABEL_TXN,
                member,
                &format!("txn-{member}"),
            ));
            assert_eq!(
                replacer.retiring_for_test(),
                MEMBER_RETIRING_CAP,
                "after swap {member} past the cap"
            );
            ended.push(cancelled(&[
                &data_0,
                &maintenance_0,
                &maintenance_1,
                &snapshot_0,
            ]));
        }

        assert_eq!(
            ended,
            [
                [true, false, false, false],
                [true, true, false, false],
                [true, true, true, false],
            ],
            "the data retirement, then maintenance member 0's, then member 1's"
        );
        assert!(
            swaps.iter().all(|(_, stop)| !stop.is_cancelled()),
            "no swap's own retirement is ended"
        );
    }

    /// Rider A bound, no pool exempt: four `maintenance` retirements (a
    /// `maintenance` pool sized past the default) fill the cap. No time passes
    /// here between registration and eviction, so each is still inside its
    /// grace when it is ended. Each later swap ends the oldest of them at
    /// once, so the count is the cap after every swap — never five — and the
    /// four go in age order while every swap's own retirement survives.
    #[test]
    fn four_maintenance_retirements_never_hold_the_count_above_the_cap() {
        let (replacer, _metrics) = offline_replacer(MEMBER_RETIRING_CAP);
        let maintenance: Vec<(u64, CancellationToken)> = (0..MEMBER_RETIRING_CAP)
            .map(|m| {
                retire(
                    &replacer,
                    POOL_LABEL_MAINTENANCE,
                    m,
                    &format!("maintenance-{m}"),
                )
            })
            .collect();
        assert_eq!(replacer.retiring_for_test(), MEMBER_RETIRING_CAP);
        let maintenance_stops: Vec<&CancellationToken> =
            maintenance.iter().map(|(_, stop)| stop).collect();

        let mut swaps = Vec::new();
        for swap in 0..MEMBER_RETIRING_CAP {
            swaps.push(retire(
                &replacer,
                POOL_LABEL_DATA,
                swap,
                &format!("data-{swap}"),
            ));
            assert_eq!(
                replacer.retiring_for_test(),
                MEMBER_RETIRING_CAP,
                "swap {swap} past the cap leaves the count at the cap"
            );
            let expected: Vec<bool> = (0..MEMBER_RETIRING_CAP).map(|m| m <= swap).collect();
            assert_eq!(
                cancelled(&maintenance_stops),
                expected,
                "swap {swap} ends maintenance member {swap}'s retirement, the oldest left"
            );
        }
        assert!(
            swaps.iter().all(|(_, stop)| !stop.is_cancelled()),
            "no swap's own retirement is ended"
        );
    }
}
