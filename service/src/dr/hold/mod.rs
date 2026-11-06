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

//! The fleet hold: whether this replica may serve, as the store says.
//!
//! # The hold flow
//!
//! 1. **The marker.** The fleet is held while the hold marker exists
//!    (`storage::keys::KeySchema::dr_hold_marker`, a HASH in the object-id
//!    counter's slot, fields in `storage::dr_hold`). Only the operator Job's
//!    un-hold script (`scripts/dr_unhold.lua`, through
//!    `storage::dr_hold::unhold`) deletes it (SDD SR-4.2). Automatic loss detection,
//!    which widens the predicate to "marker, or counter absent on an
//!    established store", is not built yet; until then a test or an operator
//!    writes the marker.
//! 2. **Startup evaluation.** After the Valkey connect and the durability
//!    guard, and before it builds any store, dispatcher or store- or
//!    backup-writing task (the metrics server and the service-token
//!    providers are already running; SDD SR-5.1a), a process reads the
//!    marker once ([`evaluate_hold_at_startup`]). `Held` starts it in held
//!    mode (`crate::grpc::hold::run_held_mode`: a refuse-all LiveService
//!    stub, and no store or store- or backup-writing task ever built),
//!    `Serving` in normal mode, and an error means it must not serve: the
//!    evaluation is fail-closed, and the process exits non-zero at once. The
//!    kubelet's restart is the retry; there is no in-process wait.
//! 3. **The poll.** In both modes the process then reads the marker every
//!    `LIVE_EDIT_DR_HOLD_POLL_INTERVAL_MS` ([`spawn_hold_poll`]). Each read
//!    is one `EVALSHA` on the counter's slot, bounded by
//!    [`poll_call_timeout`]. A read that answers publishes a
//!    [`HoldObservation`]: the [`HoldVerdict`] and the instant it was
//!    observed, on a `watch` channel. **An error is never a verdict** (SDD
//!    SR-2.7): a connection error, a timeout, `-LOADING`, `-BUSY`,
//!    `CLUSTERDOWN` or a failed redirect publishes nothing, so the last
//!    observation's instant stops advancing. The poll stops when the
//!    shutdown token is cancelled.
//! 4. **Reactions** (outside this module, in `crate::grpc::hold` and
//!    `main.rs`). A serving process that sees `Held` closes its LiveService
//!    gate (`SERVICE_HELD`), ends its streams and aborts every store- or
//!    backup-writing task — the writer inventory, [`HOLD_WRITERS`] — and
//!    keeps polling. A normal-mode process that sees `Held` after its startup
//!    evaluation but before its serving point exits 0 at once instead, and
//!    restarts into held mode. A held-mode process, or one that held in
//!    place, exits 0 after a random [`unhold_exit_jitter`] once the verdict
//!    turns `Serving`, and restarts into a normal startup. Every consumer
//!    reads the incident from the latest observation, never from a copy taken
//!    when the hold began: a new incident can replace the old one while held.
//! 5. **Staleness** ([`VerdictAge`]; the reactions are in `crate::grpc::hold`
//!    too, on the reactor's own timer). A serving
//!    process that is not held, with no successful poll for
//!    [`STALE_VERDICT_THRESHOLD`], fails closed on its own
//!    (`UPSTREAM_UNAVAILABLE`, streams ended, writers aborted, readiness
//!    `NOT_SERVING`) and exits 0 on its next successful poll; a held one
//!    stays held and reports readiness `NOT_SERVING`. No process that is
//!    serving or held ever exits because reads fail: it keeps polling, with
//!    no time limit, through the data pool, whose connections fred and the
//!    pool's own supervision rebuild. A poll task that ends before shutdown
//!    makes its process exit non-zero at once, so it restarts with a new
//!    poll.
//!
//! This module provides the parts of steps 2, 3 and 5 that read the store or
//! keep time — the verdict, the channel, the poll, the startup evaluation, the
//! staleness clock, the shared constants and the hold's after-entry bounds
//! ([`hold_write_bound`] and [`join_teardown_write_bound`], which the restore
//! Job's quiet period is derived from) — and, as data, the
//! writer inventory step 4 aborts ([`HoldWriter`], [`HOLD_WRITERS`]). `main.rs` runs
//! the startup evaluation, then, on `Held`, held mode (which spawns its own
//! poll), or, on `Serving`, spawns the poll before any other background task.

mod poll;
mod verdict;
mod writers;

use std::time::Duration;

use crate::storage::client::UNRESPONSIVE_MAX_TIMEOUT;

pub use poll::{
    evaluate_hold_at_startup, poll_hold_once, spawn_hold_poll, HoldPollError, HoldPollSettings,
};
pub use verdict::{
    poll_call_timeout, unhold_exit_jitter, verdict_age, verdict_from_marker, HoldObservation,
    HoldVerdict, VerdictAge,
};
pub use writers::{HoldWriter, HOLD_WRITERS};

/// How long a replica may go without a successful hold poll before its
/// verdict counts as stale (SDD SR-6.5): 30 s, fifteen polls at the default
/// interval. A serving replica then fails closed on its own.
///
/// A replica that cannot see the marker reaches stale-verdict entry up to
/// this threshold plus one poll interval after the marker's `detected_at`,
/// and its after-entry bounds ([`hold_write_bound`] and
/// [`join_teardown_write_bound`]) only start at that entry. So the operator
/// Job's quiet period before its first write must cover this threshold, plus
/// one poll interval, plus the largest after-entry bound (SDD SR-9.1).
pub const STALE_VERDICT_THRESHOLD: Duration = Duration::from_secs(30);

/// Upper bound of the random wait before a held replica exits once the hold
/// clears, or a stale one exits on recovery (SDD SR-6.4): 5 s. The wait is
/// drawn uniformly from 0 to this ([`unhold_exit_jitter`]) so the fleet's
/// restarts, and its clients' reconnects, spread out.
pub const UNHOLD_EXIT_JITTER_MAX: Duration = Duration::from_secs(5);

const _: () = assert!(
    crate::config::DR_HOLD_POLL_INTERVAL_MS_MAX as u128 * 5 == STALE_VERDICT_THRESHOLD.as_millis(),
    "the largest hold-poll interval is the stale-verdict threshold divided by five (SDD SR-6.5 \
     \"Poll bounds\")"
);

const _: () = assert!(
    crate::config::DR_HOLD_POLL_INTERVAL_MS_DEFAULT <= crate::config::DR_HOLD_POLL_INTERVAL_MS_MAX,
    "the default hold-poll interval is within its bound"
);

/// How long after hold (or stale-verdict) entry a serving replica can still
/// send a store write — apart from a join's teardown
/// ([`join_teardown_write_bound`]), which the restore Job's quiet period
/// counts beside it (SDD SR-9.1, `crate::dr::job::quiet`): `rpc_deadline +
/// command_timeout + UNRESPONSIVE_MAX_TIMEOUT`, saturating.
///
/// The bound counts when callers give up, not when writes land. Once a
/// call's caller gives up — its timer fires, fred answers it with an error,
/// or its future is dropped — fred writes no frame of that call it has not
/// already written, whether the frame waits behind a reconnect, in the retry
/// buffer or behind a redirect (SDD §15.3 *Vendored fred* change 9). The
/// latest give-up after entry, by class:
///
/// - a deadline-bound call already past the LiveService gate, an upload's
///   commit included: `rpc_deadline` (`LIVE_EDIT_RPC_DEADLINE_MS`), where the
///   server drops its handler, and the drop is a give-up. So the upload
///   commits' size-scaled `txn_timeout` no longer reaches past it;
/// - a deadline-exempt stream's own store work (a `JoinSession`'s SR-4 steps
///   2, 4 and 5 and its `PeerMessage` publish, a `SubscribeReadLayer`
///   catch-up's build): entry, where the hold ending ends the stream and each
///   of those awaits races the stream's token; only its teardown runs on
///   ([`join_teardown_write_bound`]);
/// - a clean-up a drop leaves (the writer inventory's audit: a read hold's
///   `HDEL`, an out-of-memory upload hold's `DEL`, a writer intent's drop
///   `HDEL`, a writer-intent renewal's clear, a fence lease's release): cut
///   at `command_timeout` (`LIVE_EDIT_VALKEY_COMMAND_TIMEOUT_MS`) after its
///   spawn, its `NOSCRIPT` reload included, and every spawn comes by
///   `rpc_deadline`, so `rpc_deadline + command_timeout`;
/// - the store out-of-memory request: one in flight at entry gives up within
///   `command_timeout`; none is sent after entry, being behind the hold write
///   fence (`storage::store_oom`);
/// - an on-access repair or restore write already handed to fred at entry
///   (`crate::checkpoint` `hold_fence`): `command_timeout`, or sooner;
/// - an aborted inventory task, a maintenance-lock refresh included: entry
///   itself, the abort being a drop.
///
/// A frame fred had already written before its caller gave up is not
/// recalled: it reaches the socket within one of fred's bounded write or
/// flush waits, `UNRESPONSIVE_MAX_TIMEOUT` (45 s, `crate::storage::client`),
/// or is dropped with its connection — hence the last term. Reads that
/// constant, so a change to the wait moves this bound with it. From the
/// kernel's send queue on, when the store runs a frame is the network's and
/// the store's: a recorded residual outside every bound (SDD §15.3
/// *Vendored fred* change 9).
///
/// Assumes `command_timeout` > 0: a zero command timeout is fred's "no
/// timeout", which bounds none of these writes, so there is no bound then.
/// `Config::from_env` refuses a zero `LIVE_EDIT_VALKEY_COMMAND_TIMEOUT_MS`
/// at startup, for the service and the restore Jobs alike.
pub fn hold_write_bound(rpc_deadline: Duration, command_timeout: Duration) -> Duration {
    rpc_deadline
        .saturating_add(command_timeout)
        .saturating_add(UNRESPONSIVE_MAX_TIMEOUT)
}

/// How long a `JoinSession` teardown waits for the connection's lease keeper
/// beyond one store command timeout (`crate::grpc::presence`'s teardown, SDD
/// SR-4.3 step 2). A keeper step already under way runs to completion, so the
/// wait covers a step's refresh plus scheduling slack. A step that takes
/// longer — a refresh retried by fred, a renewal's index re-add on top, or a
/// rejoin's clock read, register and index re-add after the refresh —
/// outlives the wait; teardown logs that and goes on. A late refresh or
/// rejoin register is then fenced by the teardown marker and the register's
/// deadline, and a late index re-add can leave only an index entry for an
/// empty session, which the sweeper removes.
///
/// Here, beside [`hold_write_bound`], because the restore Job's quiet period
/// counts it ([`join_teardown_write_bound`]); `dr` must not depend on `grpc`.
pub const KEEPER_STOP_SLACK: Duration = Duration::from_secs(1);

/// The store commands a `JoinSession` teardown sends after it stops the lease
/// keeper, each given up within one command timeout, the unregister's
/// `NOSCRIPT` reload included by its cut: the unregister script, then
/// the session-index remove-then-recheck's `SREM`, `SCARD` and `SADD`
/// (`storage::presence::remove_session_from_index`). The `UNSUBSCRIBE` that
/// follows changes no key, so it is not counted.
pub const JOIN_TEARDOWN_COMMANDS: u32 = 4;

/// How long after hold entry a `JoinSession` teardown can still send a store
/// write: the keeper stop's wait, `command_timeout` + [`KEEPER_STOP_SLACK`],
/// then [`JOIN_TEARDOWN_COMMANDS`] commands given up within `command_timeout`
/// each (the unregister cut at `command_timeout`, its `NOSCRIPT` reload
/// included, by `crate::grpc::presence`'s teardown), then
/// `UNRESPONSIVE_MAX_TIMEOUT` (45 s) for the last command's frame to reach
/// the socket, as in [`hold_write_bound`] —
/// `5 × command_timeout + KEEPER_STOP_SLACK + UNRESPONSIVE_MAX_TIMEOUT`,
/// saturating.
///
/// The hold ending ends every join's stream at entry (SDD SR-6.3), and each
/// join then runs this teardown on its own task. A join still registering at
/// entry starts it within 2 × `command_timeout`: its SR-4 steps 2, 4 and 5
/// race the stream's token, and only step 3's pub/sub (at most a prior
/// leaver's `UNSUBSCRIBE` wait and one `SUBSCRIBE`) is not raced; with no
/// register issued, only the index removal's three commands follow, so it
/// too ends within the bound. A lease-keeper step already under way when the
/// keeper stop's wait gives up runs on — a refresh with its reload, then an
/// index re-add, or a register with its reload, then a re-add — and ends
/// within entry + 4 × `command_timeout`, inside the bound. It can outlast
/// [`hold_write_bound`] once the command timeout is large: at the default
/// 30 s RPC deadline, above about 7 s. The restore Job's quiet period covers
/// the larger of the two (SDD SR-9.1). Assumes `command_timeout` > 0, as
/// [`hold_write_bound`] does.
pub fn join_teardown_write_bound(command_timeout: Duration) -> Duration {
    command_timeout
        .saturating_add(KEEPER_STOP_SLACK)
        .saturating_add(command_timeout.saturating_mul(JOIN_TEARDOWN_COMMANDS))
        .saturating_add(UNRESPONSIVE_MAX_TIMEOUT)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// At the defaults (D = 30 s, T = 5 s): the deadline, one command
    /// timeout, then fred's 45 s write wait — 30 s + 5 s + 45 s = 80 s.
    #[test]
    fn hold_write_bound_at_defaults_is_deadline_plus_command_timeout_plus_write_wait() {
        assert_eq!(UNRESPONSIVE_MAX_TIMEOUT, Duration::from_secs(45));
        assert_eq!(
            hold_write_bound(Duration::from_secs(30), Duration::from_secs(5)),
            Duration::from_secs(80)
        );
    }

    /// The command timeout adds to the deadline rather than competing with
    /// the 45 s wait, at every size.
    #[test]
    fn hold_write_bound_adds_the_command_timeout_at_every_size() {
        assert_eq!(
            hold_write_bound(Duration::from_secs(30), Duration::from_secs(60)),
            Duration::from_secs(135)
        );
        assert_eq!(
            hold_write_bound(Duration::from_secs(30), UNRESPONSIVE_MAX_TIMEOUT),
            Duration::from_secs(120)
        );
    }

    /// The keeper stop (T + 1 s), four commands of T each, then fred's 45 s
    /// write wait: 5T + 1 s + 45 s.
    #[test]
    fn join_teardown_write_bound_is_five_command_timeouts_plus_the_keeper_slack_and_write_wait() {
        assert_eq!(KEEPER_STOP_SLACK, Duration::from_secs(1));
        assert_eq!(
            join_teardown_write_bound(Duration::from_secs(5)),
            Duration::from_secs(71)
        );
        assert_eq!(
            join_teardown_write_bound(Duration::from_secs(60)),
            Duration::from_secs(346)
        );
        assert_eq!(join_teardown_write_bound(Duration::MAX), Duration::MAX);
    }

    /// Saturates rather than overflowing.
    #[test]
    fn hold_write_bound_saturates() {
        assert_eq!(
            hold_write_bound(Duration::MAX, Duration::from_secs(5)),
            Duration::MAX
        );
    }
}
